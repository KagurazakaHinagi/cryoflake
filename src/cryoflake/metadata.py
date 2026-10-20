from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Optional

import numpy as np

from cryoflake.common import array_all_finite


def validate_density_grid(
    grid_payload,
    label: str = "density_grid",
) -> np.ndarray:
    try:
        if isinstance(grid_payload, np.ndarray):
            grid = grid_payload
        else:
            if isinstance(grid_payload, (bytes, bytearray)):
                grid_payload = io.BytesIO(grid_payload)
            if hasattr(grid_payload, "seek"):
                grid_payload.seek(0)
            loaded_grid = np.load(grid_payload, allow_pickle=False)
            if isinstance(loaded_grid, np.lib.npyio.NpzFile):
                try:
                    grid = loaded_grid["arr_0"]
                finally:
                    loaded_grid.close()
            else:
                grid = loaded_grid
        grid = np.asarray(grid)
    except Exception as exc:
        raise ValueError(
            f"{label} must be a readable density grid array or NPY/NPZ input."
        ) from exc

    if grid.dtype.kind in {"f", "i", "u"}:
        with np.errstate(over="ignore", invalid="ignore"):
            grid = grid.astype(np.float32, copy=False)
    if (
        grid.ndim != 3
        or grid.size == 0
        or grid.dtype.kind not in {"f", "i", "u"}
        or not array_all_finite(grid)
    ):
        raise ValueError(f"{label} must be a finite numeric 3D density grid.")
    return grid


def validate_coordinates(
    coordinates,
    label: str = "backbone_coordinates",
    allow_nan: bool = False,
) -> np.ndarray:
    try:
        coordinates = np.asarray(coordinates)
    except Exception as exc:
        raise ValueError(f"{label} must be a numeric coordinate array.") from exc

    if coordinates.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{label} must be a numeric coordinate array.")
    if coordinates.shape == (0,):
        return np.zeros((0, 3), dtype=np.float32)
    if coordinates.ndim == 1:
        if coordinates.shape[0] != 3:
            raise ValueError(f"{label} must have shape (num_atoms, 3).")
        coordinates = coordinates.reshape(1, 3)
    elif coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"{label} must have shape (num_atoms, 3).")

    with np.errstate(over="ignore", invalid="ignore"):
        coordinates = coordinates.astype(np.float32, copy=False)
    if allow_nan:
        if np.isinf(coordinates).any():
            raise ValueError(f"{label} must not contain infinite coordinates.")
    elif not array_all_finite(coordinates):
        raise ValueError(f"{label} must contain finite coordinates.")
    return coordinates


def validate_global_origin(
    global_origin,
    label: str = "global_origin",
) -> np.ndarray:
    global_origin = np.asarray(global_origin)
    if global_origin.dtype.kind in {"f", "i", "u"}:
        with np.errstate(over="ignore", invalid="ignore"):
            global_origin = global_origin.astype(np.float32, copy=False)
    if (
        global_origin.shape != (3,)
        or global_origin.dtype.kind not in {"f", "i", "u"}
        or not array_all_finite(global_origin)
    ):
        raise ValueError(f"{label} must be a finite numeric 3-vector.")
    return global_origin


def validate_voxel_size(
    voxel_size,
    label: str = "voxel_size",
) -> float:
    voxel_size = np.asarray(voxel_size)
    if voxel_size.shape != () or voxel_size.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{label} must be a positive finite numeric scalar.")

    voxel_size_value = float(voxel_size)
    if not np.isfinite(voxel_size_value) or voxel_size_value <= 0:
        raise ValueError(f"{label} must be a positive finite numeric scalar.")
    return voxel_size_value


def validate_residue_labels(
    labels,
    expected_count: int,
    allowed_labels: set[str],
    label: str = "residue labels",
) -> list[str]:
    if isinstance(labels, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of residue labels.")
    try:
        normalized_labels = [str(residue_label).upper() for residue_label in labels]
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence of residue labels.") from exc

    if len(normalized_labels) != int(expected_count):
        raise ValueError(
            f"{label} length must match expected count {int(expected_count)}."
        )

    invalid_labels = [
        residue_label
        for residue_label in normalized_labels
        if residue_label not in allowed_labels
    ]
    if invalid_labels:
        raise ValueError(f"{label} contains unrecognized residue labels.")
    return normalized_labels


def validate_preprocessed_artifact_mapping(
    artifact_mapping,
    source: Optional[object] = None,
) -> Mapping:
    """Return an artifact mapping without validating its individual fields."""

    label = _metadata_label("preprocessed artifact", source)
    if not isinstance(artifact_mapping, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return artifact_mapping


def get_preprocessed_artifact_global_origin(
    artifact_mapping: Mapping,
    source: Optional[object] = None,
) -> np.ndarray:
    artifact_mapping = validate_preprocessed_artifact_mapping(
        artifact_mapping, source=source
    )
    label = _metadata_label("global_origin", source)
    return validate_global_origin(artifact_mapping.get("global_origin"), label=label)


def get_preprocessed_artifact_voxel_size(
    artifact_mapping: Mapping,
    source: Optional[object] = None,
) -> float:
    artifact_mapping = validate_preprocessed_artifact_mapping(
        artifact_mapping, source=source
    )
    label = _metadata_label("voxel_size", source)
    return validate_voxel_size(artifact_mapping.get("voxel_size"), label=label)


def get_preprocessed_artifact_grid(
    artifact_mapping: Mapping,
    source: Optional[object] = None,
) -> np.ndarray:
    artifact_mapping = validate_preprocessed_artifact_mapping(
        artifact_mapping, source=source
    )
    label = _metadata_label("density_grid", source)
    return validate_density_grid(artifact_mapping.get("density_grid"), label=label)


def _metadata_label(name: str, source: Optional[object]) -> str:
    if source is None:
        return name
    return f"{name} in {source}"
