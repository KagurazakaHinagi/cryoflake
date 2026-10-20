"""Streaming integrity validation for on-disk NPZ+JSON stores."""

from __future__ import annotations

import hashlib
import io
import math
import zipfile
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import BinaryIO

import numpy as np

from cryoflake.storage.codec import (
    STORE_GENERATION_KEY,
    _collect_array_specs,
    read_metadata,
)


def _read_array_header(member: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(member)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
    elif version in {(2, 0), (3, 0)}:
        # v3 changes header text encoding only. Portable artifacts use plain
        # dtypes with ASCII descriptors; structured Unicode names are excluded.
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(member)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    return shape, fortran_order, dtype


def _read_array_payload(
    member: BinaryIO,
    shape: tuple[int, ...],
    fortran_order: bool,
    dtype: np.dtype,
) -> np.ndarray:
    # Allocate only after the header and member length have been validated.
    # ndarray preserves zero-width string dtypes, unlike np.empty().
    array = np.ndarray(math.prod(shape), dtype=dtype)
    if dtype.itemsize:
        target = memoryview(array.view(np.uint8))
        offset = 0
        while offset < len(target):
            # ZipExtFile.readinto() may allocate an intermediate bytes object;
            # keep that temporary bounded independently of the array size.
            count = member.readinto(target[offset : offset + 256 * 1024])
            if not count:
                raise ValueError("NPZ+JSON array has a truncated serialized payload")
            offset += count
    if fortran_order:
        return array.reshape(shape[::-1]).transpose()
    return array.reshape(shape)


def _validate_store_archive(
    archive: zipfile.ZipFile,
    store: str,
    declared_specs: Mapping[
        tuple[str, str],
        tuple[np.dtype, tuple[int, ...]],
    ],
    expected_generation: str,
    *,
    selected_keys: Collection[str] | None = None,
    load_arrays: bool = False,
) -> dict[str, np.ndarray]:
    declared = {
        key: descriptor
        for (array_store, key), descriptor in declared_specs.items()
        if array_store == store
    }
    expected_members = {
        f"{key}.npy": descriptor for key, descriptor in declared.items()
    }
    member_names = archive.namelist()
    if len(member_names) != len(set(member_names)):
        raise ValueError(f"NPZ+JSON {store} store contains duplicate members")
    available = set(member_names)
    generation_member = f"{STORE_GENERATION_KEY}.npy"
    if generation_member not in available:
        raise ValueError(f"NPZ+JSON {store} store is missing its generation")
    available.remove(generation_member)
    if available != set(expected_members):
        missing = set(expected_members) - available
        unexpected = available - set(expected_members)
        if missing:
            raise ValueError(
                f"NPZ+JSON {store} store is missing arrays: {sorted(missing)}"
            )
        raise ValueError(
            f"NPZ+JSON {store} store has unexpected arrays: {sorted(unexpected)}"
        )

    generation_info = archive.getinfo(generation_member)
    if generation_info.file_size > 1024:
        raise ValueError(f"NPZ+JSON {store} store has an invalid generation")
    with archive.open(generation_info) as generation_file:
        generation_bytes = generation_file.read()
    generation_source = io.BytesIO(generation_bytes)
    shape, _, dtype = _read_array_header(generation_source)
    if (
        shape != ()
        or dtype.kind != "U"
        or generation_source.tell() + dtype.itemsize != len(generation_bytes)
    ):
        raise ValueError(f"NPZ+JSON {store} store has an invalid generation")
    generation_source.seek(0)
    generation = np.load(generation_source, allow_pickle=False)
    if str(generation.item()) != expected_generation:
        raise ValueError(f"NPZ+JSON {store} store generation does not match metadata")

    arrays = {}
    for member_name, (
        expected_dtype,
        expected_shape,
    ) in expected_members.items():
        if selected_keys is not None and member_name[:-4] not in selected_keys:
            continue
        member_info = archive.getinfo(member_name)
        with archive.open(member_info) as member:
            shape, fortran_order, dtype = _read_array_header(member)
            payload_offset = member.tell()
            if dtype != expected_dtype:
                raise ValueError(
                    f"Array {store}/{member_name[:-4]} has dtype {dtype}, "
                    f"expected {expected_dtype}"
                )
            if shape != expected_shape:
                raise ValueError(
                    f"Array {store}/{member_name[:-4]} has shape {shape}, "
                    f"expected {expected_shape}"
                )
            expected_size = payload_offset + (
                math.prod(expected_shape) * expected_dtype.itemsize
            )
            if member_info.file_size != expected_size:
                raise ValueError(
                    f"Array {store}/{member_name[:-4]} has an invalid "
                    "serialized payload size"
                )
            if load_arrays:
                arrays[member_name[:-4]] = _read_array_payload(
                    member, shape, fortran_order, dtype
                )
            else:
                for _ in iter(lambda: member.read(1024 * 1024), b""):
                    pass
    return arrays


def _validate_store_file(
    path: Path,
    store: str,
    declared_specs: Mapping[
        tuple[str, str],
        tuple[np.dtype, tuple[int, ...]],
    ],
    expected_generation: str,
    expected_hash: str,
) -> None:
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != expected_hash:
            raise ValueError(f"NPZ+JSON {store} store SHA-256 does not match metadata")
        source.seek(0)
        with zipfile.ZipFile(source) as archive:
            _validate_store_archive(archive, store, declared_specs, expected_generation)


def validate_npz_json_store_files(
    metadata_data: bytes | Path,
    core_path: Path,
    sequence_path: Path | None = None,
) -> None:
    """Validate every declared archive member without retaining the payload."""

    _validate_npz_json_store_files_metadata(
        read_metadata(metadata_data), core_path, sequence_path
    )


def _validate_npz_json_store_files_metadata(
    metadata: Mapping[str, object],
    core_path: Path,
    sequence_path: Path | None = None,
) -> None:
    declared_specs: dict[
        tuple[str, str],
        tuple[np.dtype, tuple[int, ...]],
    ] = {}
    _collect_array_specs(metadata["tree"], declared_specs)
    _validate_store_file(
        Path(core_path),
        "core",
        declared_specs,
        metadata["store_generations"]["core"],
        metadata["store_hashes"]["core"],
    )
    if "sequence" in metadata["files"]:
        if sequence_path is None:
            raise ValueError("NPZ+JSON sequence array store is required")
        _validate_store_file(
            Path(sequence_path),
            "sequence",
            declared_specs,
            metadata["store_generations"]["sequence"],
            metadata["store_hashes"]["sequence"],
        )
