"""Summary support for manifest preprocessing."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from cryoflake.manifests.manifest import (
    normalize_emdb_id,
)
from cryoflake.metadata import (
    validate_density_grid,
    validate_voxel_size,
)
from cryoflake.storage.transports import (
    is_remote_uri,
)

PREPROCESS_SUMMARY_FIELDNAMES = [
    "emdb_id",
    "wwpdb_id",
    "map_url",
    "structure_url",
    "map_path",
    "structure_path",
    "artifact_reference",
    "cached_artifact_dir",
    "source_resolution",
    "input_voxel_size",
    "artifact_voxel_size",
    "grid_shape",
    "component_node_count",
    "component_interface_count",
    "component_group_ids",
    "interface_group_ids",
    "split",
    "split_seed",
    "success",
    "status",
    "error",
    "finished_at",
]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _summary_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_exception(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _read_input_voxel_size(map_path: os.PathLike) -> str:
    map_path = Path(map_path)
    if not map_path.is_file():
        return ""
    try:
        import mrcfile

        with mrcfile.open(str(map_path), "r", header_only=True) as mrc_file:
            voxel_size = float(mrc_file.voxel_size.x)
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            return ""
    except Exception:
        return ""
    return str(voxel_size)


def _artifact_voxel_size(artifact_mapping: Mapping[str, object]) -> str:
    try:
        voxel_size = validate_voxel_size(artifact_mapping.get("voxel_size"))
    except Exception:
        return ""
    return str(float(voxel_size))


def _artifact_grid_shape(artifact_mapping: Mapping[str, object]) -> str:
    try:
        grid = validate_density_grid(artifact_mapping.get("density_grid"))
    except Exception:
        return ""
    return "x".join(str(int(axis_length)) for axis_length in grid.shape)


def _preprocess_summary_row(
    manifest_row: Mapping[str, str],
    map_path: os.PathLike,
    structure_path: os.PathLike,
    artifact_reference: os.PathLike,
    split_seed: int,
) -> dict[str, str]:
    return {
        "emdb_id": normalize_emdb_id(manifest_row["emdb_id"]),
        "wwpdb_id": manifest_row.get("wwpdb_id", "").strip(),
        "map_url": manifest_row.get("map_url", "").strip(),
        "structure_url": manifest_row.get("structure_url", "").strip(),
        "map_path": str(map_path),
        "structure_path": str(structure_path) if str(structure_path) else "",
        "artifact_reference": str(artifact_reference),
        "cached_artifact_dir": "",
        "source_resolution": manifest_row.get("source_resolution", "").strip(),
        "input_voxel_size": _read_input_voxel_size(map_path),
        "artifact_voxel_size": "",
        "grid_shape": "",
        "component_node_count": "",
        "component_interface_count": "",
        "component_group_ids": "",
        "interface_group_ids": "",
        "split": "",
        "split_seed": str(split_seed),
        "success": "false",
        "status": "",
        "error": "",
        "finished_at": "",
    }


def _finish_summary_row(
    summary_row: dict[str, str],
    status: str,
    success: bool,
    error: str = "",
    artifact_voxel_size: str = "",
    grid_shape: str = "",
) -> None:
    summary_row["status"] = status
    summary_row["success"] = _summary_bool(success)
    summary_row["error"] = error
    summary_row["artifact_voxel_size"] = artifact_voxel_size
    summary_row["grid_shape"] = grid_shape
    summary_row["finished_at"] = _utc_timestamp()


def _assign_summary_splits(
    summary_rows: list[dict[str, str]],
    artifact_reference_splits: Mapping[Path | str, set[str]],
) -> None:
    for summary_row in summary_rows:
        artifact_reference = summary_row.get("artifact_reference", "")
        if not artifact_reference:
            continue
        artifact_reference_key = (
            artifact_reference
            if is_remote_uri(artifact_reference)
            else Path(artifact_reference)
        )
        split_names = artifact_reference_splits.get(artifact_reference_key, set())
        if split_names:
            summary_row["split"] = ",".join(sorted(split_names))


def write_preprocess_summary_csv(
    summary_rows: Iterable[Mapping[str, str]],
    summary_csv: os.PathLike,
) -> None:
    summary_path = Path(summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=PREPROCESS_SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    fieldname: row.get(fieldname, "")
                    for fieldname in PREPROCESS_SUMMARY_FIELDNAMES
                }
            )
