"""Reuse support for manifest preprocessing."""

from __future__ import annotations

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

from cryoflake.constants import (
    BACKBONE_ATOM_NAMES,
    PREPROCESSED_NUCLEOTIDE_RESIDUE_LABELS,
    PREPROCESSED_PROTEIN_RESIDUE_LABELS,
)
from cryoflake.manifests.manifest import (
    get_preprocessed_artifact_name,
)
from cryoflake.metadata import (
    validate_coordinates,
    validate_density_grid,
    validate_global_origin,
    validate_residue_labels,
    validate_voxel_size,
)
from cryoflake.sequence.validation import (
    validate_sequence_embedding_payload,
)
from cryoflake.storage.store import (
    load_preprocessed_artifact,
)
from cryoflake.targets.full_atom import validate_full_atom_targets


def _resolve_manifest_path(path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value.strip())
    if not path.is_absolute():
        path = manifest_dir / path
    return path


def _materialize_preprocess_input(input_path: Path, temporary_dir: Path) -> Path:
    if input_path.suffix.lower() != ".gz":
        return input_path

    materialized_path = temporary_dir / input_path.with_suffix("").name
    with gzip.open(input_path, "rb") as input_file:
        with materialized_path.open("wb") as output_file:
            shutil.copyfileobj(input_file, output_file)
    return materialized_path


@dataclass(frozen=True)
class _ValidatedTrainingArtifact:
    grid_shape: tuple[int, int, int]
    voxel_size: float
    protein_nodes: int
    nucleotide_nodes: int
    component_targets: dict
    sequence_shape: tuple[int, int] | None


def _coordinates_have_grid_targets(
    coordinate_arrays: Iterable[np.ndarray],
    grid_shape: tuple[int, int, int],
    global_origin: np.ndarray,
    voxel_size: float,
) -> bool:
    """Check validated coordinates with workspace bounded by one chunk."""

    with np.errstate(over="ignore"):
        grid_extent_xyz = np.asarray(grid_shape[::-1], dtype=np.float64) * voxel_size
    for coordinates in coordinate_arrays:
        for start in range(0, len(coordinates), 65536):
            # Compare world-space differences without division: very small or
            # large positive spacing must not turn zeros into NaNs or negative
            # coordinates into -0. Float64 also prevents float32 subtraction
            # overflow. NaNs fail the comparisons without a filtered copy.
            shifted_coordinates_xyz = np.subtract(
                coordinates[start : start + 65536], global_origin, dtype=np.float64
            )
            inside_mask = np.logical_and(
                shifted_coordinates_xyz >= 0,
                shifted_coordinates_xyz < grid_extent_xyz,
            ).all(axis=-1)
            if np.any(inside_mask):
                return True
    return False


def _validate_training_artifact_mapping(
    artifact_mapping: object,
    *,
    expected_interface_contact_cutoff: Optional[float] = None,
    require_sequence_embeddings: bool = False,
) -> _ValidatedTrainingArtifact:
    """Validate once and retain only details needed by callers and summaries."""

    if not isinstance(artifact_mapping, Mapping):
        raise ValueError("Preprocessed artifact must be a mapping.")
    from cryoflake.targets.schema import validate_component_extension_targets

    component_targets = validate_component_extension_targets(
        artifact_mapping.get("extension_targets")
    )["component"]
    if expected_interface_contact_cutoff is not None and not np.isclose(
        component_targets["interface_contact_cutoff"],
        float(expected_interface_contact_cutoff),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("Component interface contact cutoff does not match.")
    sequence_shape = None
    if require_sequence_embeddings or {"sequence", "sequence_mask"}.intersection(
        artifact_mapping
    ):
        sequence_shape = tuple(
            int(size)
            for size in validate_sequence_embedding_payload(artifact_mapping)[
                "sequence"
            ].shape
        )
    grid = validate_density_grid(artifact_mapping.get("density_grid"))
    global_origin = validate_global_origin(artifact_mapping.get("global_origin"))
    voxel_size = validate_voxel_size(artifact_mapping.get("voxel_size"))

    backbone_coordinates = artifact_mapping.get("backbone_coordinates")
    residue_info = artifact_mapping.get("residue_info")
    backbone_atoms = {*BACKBONE_ATOM_NAMES, "P"}
    if not isinstance(backbone_coordinates, Mapping) or not backbone_atoms.issubset(
        backbone_coordinates
    ):
        raise ValueError("Preprocessed artifact must contain all backbone atom arrays.")
    coordinates = {
        atom_name: validate_coordinates(backbone_coordinates[atom_name], allow_nan=True)
        for atom_name in backbone_atoms
    }
    num_protein = len(coordinates["CA"])
    num_nucleotide = len(coordinates["P"])
    if any(len(coordinates[name]) != num_protein for name in BACKBONE_ATOM_NAMES):
        raise ValueError("Protein backbone atom arrays must have matching row counts.")
    validate_full_atom_targets(
        artifact_mapping.get("full_atom_targets"),
        num_protein=num_protein,
        num_nucleotide=num_nucleotide,
    )
    if not isinstance(residue_info, Mapping):
        raise ValueError("Preprocessed artifact must contain residue_info.")
    validate_residue_labels(
        residue_info.get("aa_type"),
        expected_count=num_protein,
        allowed_labels=PREPROCESSED_PROTEIN_RESIDUE_LABELS,
    )
    validate_residue_labels(
        residue_info.get("nuc_type"),
        expected_count=num_nucleotide,
        allowed_labels=PREPROCESSED_NUCLEOTIDE_RESIDUE_LABELS,
    )
    if not _coordinates_have_grid_targets(
        (coordinates["CA"], coordinates["P"]),
        grid.shape,
        global_origin,
        voxel_size,
    ):
        raise ValueError(
            "Preprocessed artifact must have a backbone target in the grid."
        )
    return _ValidatedTrainingArtifact(
        grid_shape=tuple(int(size) for size in grid.shape),
        voxel_size=voxel_size,
        protein_nodes=num_protein,
        nucleotide_nodes=num_nucleotide,
        component_targets=component_targets,
        sequence_shape=sequence_shape,
    )


def _preprocessed_artifact_is_reusable(
    artifact_mapping: object,
    *,
    expected_interface_contact_cutoff: Optional[float] = None,
    require_sequence_embeddings: bool = False,
) -> bool:
    try:
        _validate_training_artifact_mapping(
            artifact_mapping,
            expected_interface_contact_cutoff=expected_interface_contact_cutoff,
            require_sequence_embeddings=require_sequence_embeddings,
        )
    except (ImportError, TypeError, ValueError, OverflowError):
        return False
    return True


def _find_reusable_preprocessed_artifact(
    manifest_row: Mapping[str, str],
    target_artifact_reference: Path | str,
    reuse_dirs: Iterable[Path],
    s3_endpoint_url: Optional[str] = None,
    expected_interface_contact_cutoff: Optional[float] = None,
    allow_external_reuse: bool = True,
    replace_sequence_embeddings: bool = False,
) -> Optional[tuple[Path | str, dict[str, object]]]:
    """Return a reusable path and its already loaded payload for inspection."""

    artifact_name = get_preprocessed_artifact_name(manifest_row)
    seen_paths = set()
    candidate_paths = [target_artifact_reference]
    if allow_external_reuse:
        candidate_paths.extend(directory / artifact_name for directory in reuse_dirs)
    for candidate_path in candidate_paths:
        path_key_value = str(candidate_path)
        if path_key_value in seen_paths:
            continue
        seen_paths.add(path_key_value)
        try:
            artifact_mapping = load_preprocessed_artifact(
                candidate_path,
                include_sequence_embeddings=not replace_sequence_embeddings,
                s3_endpoint_url=s3_endpoint_url,
            )
        except Exception:
            continue
        if _preprocessed_artifact_is_reusable(
            artifact_mapping,
            expected_interface_contact_cutoff=expected_interface_contact_cutoff,
        ):
            return candidate_path, artifact_mapping
        # Failed candidates can contain very large grids. Release their arrays
        # before the next archive is decompressed into memory.
        del artifact_mapping
    return None
