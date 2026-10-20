"""Reuse support for manifest preprocessing."""

from __future__ import annotations

import gzip
import shutil
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


def _coordinate_count(
    coordinates,
    allow_nan: bool = False,
) -> Optional[int]:
    try:
        coordinates = validate_coordinates(coordinates, allow_nan=allow_nan)
    except Exception:
        return None
    return int(coordinates.shape[0])


def _coordinates_have_grid_targets(
    coordinate_payloads: Iterable[object],
    grid_shape: tuple[int, int, int],
    global_origin: np.ndarray,
    voxel_size: float,
    allow_nan: bool = False,
) -> bool:
    grid_shape_xyz = np.asarray(grid_shape[::-1], dtype=np.float32)
    global_origin = np.asarray(global_origin, dtype=np.float32)
    for coordinates in coordinate_payloads:
        try:
            coordinates = validate_coordinates(coordinates, allow_nan=allow_nan)
        except Exception:
            return False
        if coordinates.size == 0:
            continue
        finite_mask = np.isfinite(coordinates).all(axis=-1)
        if not np.any(finite_mask):
            continue
        voxel_coordinates_xyz = (coordinates[finite_mask] - global_origin) / float(
            voxel_size
        )
        inside_mask = np.logical_and(
            voxel_coordinates_xyz >= 0,
            voxel_coordinates_xyz < grid_shape_xyz,
        ).all(axis=-1)
        if np.any(inside_mask):
            return True
    return False


def _residue_labels_are_valid(
    labels,
    expected_count: int,
    allowed_labels: set[str],
) -> bool:
    try:
        validate_residue_labels(
            labels,
            expected_count=expected_count,
            allowed_labels=allowed_labels,
        )
    except ValueError:
        return False
    return True


def _residue_info_matches_counts(
    residue_info,
    num_protein: int,
    num_nucleotide: int,
) -> bool:
    if not isinstance(residue_info, Mapping):
        return False
    return _residue_labels_are_valid(
        residue_info.get("aa_type"),
        expected_count=num_protein,
        allowed_labels=PREPROCESSED_PROTEIN_RESIDUE_LABELS,
    ) and _residue_labels_are_valid(
        residue_info.get("nuc_type"),
        expected_count=num_nucleotide,
        allowed_labels=PREPROCESSED_NUCLEOTIDE_RESIDUE_LABELS,
    )


def _preprocessed_artifact_is_reusable(
    artifact_mapping: object,
    *,
    expected_interface_contact_cutoff: Optional[float] = None,
    require_sequence_embeddings: bool = False,
) -> bool:
    if not isinstance(artifact_mapping, Mapping):
        return False
    try:
        from cryoflake.targets.schema import (
            validate_component_extension_targets,
        )

        extension_targets = validate_component_extension_targets(
            artifact_mapping.get("extension_targets")
        )
        if expected_interface_contact_cutoff is not None and not np.isclose(
            extension_targets["component"]["interface_contact_cutoff"],
            float(expected_interface_contact_cutoff),
            rtol=0.0,
            atol=1e-8,
        ):
            return False
        if require_sequence_embeddings or {"sequence", "sequence_mask"}.intersection(
            artifact_mapping
        ):
            validate_sequence_embedding_payload(artifact_mapping)
    except (ImportError, TypeError, ValueError):
        return False
    if "density_grid" not in artifact_mapping:
        return False
    try:
        grid = validate_density_grid(artifact_mapping["density_grid"])
        global_origin = validate_global_origin(artifact_mapping.get("global_origin"))
        voxel_size = validate_voxel_size(artifact_mapping.get("voxel_size"))
    except Exception:
        return False

    backbone_coordinates = artifact_mapping.get("backbone_coordinates")
    residue_info = artifact_mapping.get("residue_info")
    backbone_atoms = {*BACKBONE_ATOM_NAMES, "P"}
    has_full_backbone = False
    if isinstance(backbone_coordinates, Mapping) and backbone_atoms.issubset(
        backbone_coordinates
    ):
        atom_counts = {
            atom_name: _coordinate_count(
                backbone_coordinates[atom_name], allow_nan=True
            )
            for atom_name in backbone_atoms
        }
        if all(atom_count is not None for atom_count in atom_counts.values()):
            num_protein = atom_counts["CA"]
            num_nucleotide = atom_counts["P"]
            try:
                validate_full_atom_targets(
                    artifact_mapping.get("full_atom_targets"),
                    num_protein=num_protein,
                    num_nucleotide=num_nucleotide,
                )
                has_full_atom_targets = True
            except ValueError:
                has_full_atom_targets = False
            has_matching_backbone_lengths = all(
                atom_counts[atom_name] == num_protein
                for atom_name in BACKBONE_ATOM_NAMES
            )
            has_full_backbone = (
                has_matching_backbone_lengths
                and has_full_atom_targets
                and _residue_info_matches_counts(
                    residue_info,
                    num_protein=num_protein,
                    num_nucleotide=num_nucleotide,
                )
                and _coordinates_have_grid_targets(
                    (backbone_coordinates["CA"], backbone_coordinates["P"]),
                    grid.shape,
                    global_origin,
                    voxel_size,
                    allow_nan=True,
                )
            )
    return has_full_backbone


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
