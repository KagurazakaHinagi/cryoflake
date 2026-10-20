import contextlib
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

import numpy as np

from cryoflake.io.mrc import normalized_grid_size

_FLOAT32_BYTES = 4
_FLOAT64_BYTES = 8
_COMPLEX128_BYTES = 16
_GIB = 1024**3
_DEFAULT_MEMORY_GUARD_FRACTION = 0.75
_FALLBACK_MEMORY_GUARD_BYTES = 128 * _GIB
_CGROUP_UNLIMITED_THRESHOLD_BYTES = 1 << 60
_PROC_SELF_CGROUP_PATH = Path("/proc/self/cgroup")
_PROC_SELF_MOUNTINFO_PATH = Path("/proc/self/mountinfo")


@dataclass(frozen=True)
class PreprocessMemoryEstimate:
    input_shape: tuple[int, int, int]
    input_voxel_size: float
    cubic_size: int
    output_size: int
    estimated_peak_bytes: int


class PreprocessMemoryAdmissionCancelled(RuntimeError):
    """Raised when a waiting preprocessing job is cancelled."""


class PreprocessMemoryAdmission:
    """Bound aggregate estimated preprocessing memory across worker threads."""

    def __init__(
        self,
        limit_bytes: int,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.limit_bytes = int(limit_bytes)
        if self.limit_bytes <= 0:
            raise ValueError("limit_bytes must be a positive integer")
        self._cancel_event = cancel_event
        self._condition = threading.Condition()
        self._reserved_bytes = 0

    @property
    def reserved_bytes(self) -> int:
        with self._condition:
            return self._reserved_bytes

    def acquire(self, requested_bytes: int) -> None:
        requested_bytes = int(requested_bytes)
        if requested_bytes <= 0:
            raise ValueError("requested_bytes must be a positive integer")
        if requested_bytes > self.limit_bytes:
            raise ValueError(
                f"requested_bytes ({requested_bytes}) exceeds memory admission "
                f"limit ({self.limit_bytes})"
            )

        with self._condition:
            while self._reserved_bytes + requested_bytes > self.limit_bytes:
                if self._cancel_event is not None and self._cancel_event.is_set():
                    raise PreprocessMemoryAdmissionCancelled(
                        "Preprocessing memory admission was cancelled"
                    )
                self._condition.wait(timeout=0.1)
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise PreprocessMemoryAdmissionCancelled(
                    "Preprocessing memory admission was cancelled"
                )
            self._reserved_bytes += requested_bytes

    def release(self, requested_bytes: int) -> None:
        requested_bytes = int(requested_bytes)
        with self._condition:
            if requested_bytes <= 0 or requested_bytes > self._reserved_bytes:
                raise ValueError("released bytes exceed reserved memory")
            self._reserved_bytes -= requested_bytes
            self._condition.notify_all()

    @contextlib.contextmanager
    def reserve(self, requested_bytes: int):
        requested_bytes = int(requested_bytes)
        self.acquire(requested_bytes)
        try:
            yield
        finally:
            self.release(requested_bytes)


def _physical_memory_bytes() -> Optional[int]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    total = int(page_size) * int(physical_pages)
    if total <= 0:
        return None
    return total


def _decode_mountinfo_path(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _parse_proc_cgroups(text: str) -> list[tuple[str, PurePosixPath]]:
    cgroups = []
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controllers = fields[1].split(",") if fields[1] else []
        if not controllers:
            cgroups.append(("v2", PurePosixPath(fields[2] or "/")))
        elif "memory" in controllers:
            cgroups.append(("v1", PurePosixPath(fields[2] or "/")))
    return cgroups


def _parse_memory_cgroup_mounts(
    text: str,
) -> list[tuple[str, PurePosixPath, Path]]:
    mounts = []
    for line in text.splitlines():
        try:
            before_separator, after_separator = line.split(" - ", 1)
        except ValueError:
            continue
        mount_fields = before_separator.split()
        filesystem_fields = after_separator.split()
        if len(mount_fields) < 5 or len(filesystem_fields) < 3:
            continue
        filesystem_type = filesystem_fields[0]
        super_options = filesystem_fields[2].split(",")
        if filesystem_type == "cgroup2":
            kind = "v2"
        elif filesystem_type == "cgroup" and "memory" in super_options:
            kind = "v1"
        else:
            continue
        mounts.append(
            (
                kind,
                PurePosixPath(_decode_mountinfo_path(mount_fields[3])),
                Path(_decode_mountinfo_path(mount_fields[4])),
            )
        )
    return mounts


def _resolve_cgroup_directory(
    mount_root: PurePosixPath,
    mount_point: Path,
    process_cgroup: PurePosixPath,
) -> Path:
    try:
        relative_path = process_cgroup.relative_to(mount_root)
    except ValueError:
        relative_path = process_cgroup.relative_to("/")
    if str(relative_path) == ".":
        return mount_point
    return mount_point.joinpath(*relative_path.parts)


def _read_finite_cgroup_limit(path: Path) -> Optional[int]:
    try:
        raw_value = path.read_text().strip()
    except (OSError, UnicodeError):
        return None
    if raw_value in {"", "max", "-1"}:
        return None
    try:
        limit_bytes = int(raw_value)
    except ValueError:
        return None
    if limit_bytes <= 0 or limit_bytes >= _CGROUP_UNLIMITED_THRESHOLD_BYTES:
        return None
    return limit_bytes


def _cgroup_memory_limit_bytes(
    cgroup_path: Path = _PROC_SELF_CGROUP_PATH,
    mountinfo_path: Path = _PROC_SELF_MOUNTINFO_PATH,
) -> Optional[int]:
    try:
        cgroups = _parse_proc_cgroups(cgroup_path.read_text())
        mounts = _parse_memory_cgroup_mounts(mountinfo_path.read_text())
    except (OSError, UnicodeError):
        return None

    limits = []
    seen_limit_paths = set()
    for cgroup_kind, process_cgroup in cgroups:
        limit_file_name = (
            "memory.max" if cgroup_kind == "v2" else "memory.limit_in_bytes"
        )
        matching_mounts = [
            (mount_root, mount_point)
            for mount_kind, mount_root, mount_point in mounts
            if mount_kind == cgroup_kind
            and (process_cgroup == mount_root or mount_root in process_cgroup.parents)
        ]
        if not matching_mounts:
            same_kind_mounts = [
                (mount_root, mount_point)
                for mount_kind, mount_root, mount_point in mounts
                if mount_kind == cgroup_kind
            ]
            if len(same_kind_mounts) == 1:
                matching_mounts = same_kind_mounts
        for mount_root, mount_point in matching_mounts:
            current_dir = _resolve_cgroup_directory(
                mount_root,
                mount_point,
                process_cgroup,
            )
            while True:
                limit_path = current_dir / limit_file_name
                if limit_path not in seen_limit_paths:
                    seen_limit_paths.add(limit_path)
                    limit_bytes = _read_finite_cgroup_limit(limit_path)
                    if limit_bytes is not None:
                        limits.append(limit_bytes)
                if current_dir == mount_point:
                    break
                parent_dir = current_dir.parent
                if parent_dir == current_dir:
                    break
                current_dir = parent_dir
    return min(limits) if limits else None


def preprocess_memory_guard_limit_bytes() -> int:
    physical_memory_bytes = _physical_memory_bytes()
    cgroup_memory_limit_bytes = _cgroup_memory_limit_bytes()
    available_limits = [
        limit
        for limit in (physical_memory_bytes, cgroup_memory_limit_bytes)
        if limit is not None
    ]
    if not available_limits:
        return _FALLBACK_MEMORY_GUARD_BYTES
    effective_memory_bytes = min(available_limits)
    return max(
        1,
        int(effective_memory_bytes * _DEFAULT_MEMORY_GUARD_FRACTION),
    )


def _estimated_preprocess_peak_bytes(
    input_shape: tuple[int, int, int],
    cubic_size: int,
    output_size: int,
) -> int:
    initial_fft_elements = cubic_size * cubic_size * (cubic_size // 2 + 1)
    output_fft_elements = output_size * output_size * (output_size // 2 + 1)

    initial_fft_bytes = initial_fft_elements * _COMPLEX128_BYTES
    original_grid_bytes = int(np.prod(input_shape)) * _FLOAT32_BYTES
    cubic_grid_bytes = cubic_size**3 * _FLOAT32_BYTES
    resident_input_bytes = cubic_grid_bytes
    if input_shape != (cubic_size, cubic_size, cubic_size):
        resident_input_bytes += original_grid_bytes
    output_grid_bytes = output_size**3 * _FLOAT32_BYTES
    output_fft_bytes = output_fft_elements * _COMPLEX128_BYTES
    spectral_workspace_bytes = output_fft_elements * _FLOAT64_BYTES * 6

    # NumPy keeps the original and shifted FFTs alive together. Upscaling also
    # holds both the allocated output and its final shifted copy; downscaling
    # replaces that allocation with a view before making the final copy.
    if output_size == cubic_size:
        rescale_peak_bytes = resident_input_bytes
    else:
        output_fft_copies = 1 if output_size < cubic_size else 2
        rescale_peak_bytes = (
            resident_input_bytes
            + initial_fft_bytes * 2
            + output_fft_bytes * output_fft_copies
        )
    inverse_fft_peak_bytes = (
        resident_input_bytes + initial_fft_bytes + output_fft_bytes * 2
    )
    # get_spectral_indices can hold three mesh grids plus three expression/
    # shift temporaries at once, all float64 half-grid arrays.
    spectral_peak_bytes = (
        output_grid_bytes + output_fft_bytes + spectral_workspace_bytes
    )
    return max(
        rescale_peak_bytes,
        inverse_fft_peak_bytes,
        spectral_peak_bytes,
    )


def estimate_preprocess_memory(
    map_path: os.PathLike,
    target_voxel_size: float,
) -> PreprocessMemoryEstimate:
    import mrcfile

    map_path = Path(map_path)
    with mrcfile.open(str(map_path), "r", header_only=True) as mrc_file:
        input_voxel_size = float(mrc_file.voxel_size.x)
        input_shape = (
            int(mrc_file.header.nz),
            int(mrc_file.header.ny),
            int(mrc_file.header.nx),
        )

    if not np.isfinite(input_voxel_size) or input_voxel_size <= 0:
        raise ValueError(f"Invalid voxel size in MRC header: {input_voxel_size}")
    if min(input_shape) <= 0:
        raise ValueError(f"Invalid MRC grid shape: {input_shape}")

    cubic_size = max(max(input_shape), 128)
    cubic_size += cubic_size % 2
    output_size = normalized_grid_size(
        cubic_size,
        input_voxel_size,
        target_voxel_size,
    )

    estimated_peak_bytes = _estimated_preprocess_peak_bytes(
        input_shape,
        cubic_size,
        output_size,
    )

    return PreprocessMemoryEstimate(
        input_shape=input_shape,
        input_voxel_size=input_voxel_size,
        cubic_size=cubic_size,
        output_size=output_size,
        estimated_peak_bytes=estimated_peak_bytes,
    )


def estimate_preprocess_memory_or_none(
    map_path: os.PathLike,
    target_voxel_size: float,
) -> Optional[PreprocessMemoryEstimate]:
    try:
        return estimate_preprocess_memory(map_path, target_voxel_size)
    except Exception:
        return None
