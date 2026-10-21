"""CPU preprocessing from fitted maps and structures to canonical artifacts."""

from __future__ import annotations

import gzip
import shutil
import tempfile
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile

from cryoflake.io.fasta import read_fasta
from cryoflake.io.mrc import load_mrc, prepare_grid, spectral_amplitude
from cryoflake.io.structure import load_full_backbone_from_structure
from cryoflake.metadata import (
    validate_density_grid,
    validate_global_origin,
    validate_voxel_size,
)
from cryoflake.preprocess.reuse import _preprocessed_artifact_is_reusable
from cryoflake.storage import preprocessed_artifact_root, write_preprocessed_artifact
from cryoflake.targets.components import build_component_extension_targets
from cryoflake.targets.full_atom import validate_full_atom_targets
from cryoflake.targets.structure import load_full_atom_targets_from_structure


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not np.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return normalized


def _input_path(value: str | Path, name: str, suffixes: set[str]) -> Path:
    path = Path(value)
    suffix = (
        path.with_suffix("").suffix if path.suffix.lower() == ".gz" else path.suffix
    )
    if suffixes and suffix.lower() not in suffixes:
        raise ValueError(f"Unsupported {name} extension: {suffix!r}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not identify a local file: {path}")
    return path


def _materialize_input(path: Path, temporary_dir: Path, name: str) -> Path:
    if path.suffix.lower() != ".gz":
        return path
    destination = temporary_dir / f"{name}{path.with_suffix('').suffix}"
    with gzip.open(path, "rb") as source, destination.open("wb") as output:
        shutil.copyfileobj(source, output)
    return destination


def _component_structure_path(structure_path: Path, temporary_dir: Path) -> Path:
    if structure_path.suffix.lower() in {".cif", ".mmcif"}:
        return structure_path
    # PDB has no mmCIF entity/chemical-component tables. Preserve its deposited
    # atom records and author identities; component classification uses the
    # remaining residue, element and ATOM/HETATM information.
    source = PDBFile.read(structure_path)
    atoms = source.get_structure(
        model=1, altloc="all", extra_fields=["occupancy", "b_factor"]
    )
    atoms.chain_id = np.where(atoms.chain_id == "", " ", atoms.chain_id)
    converted = pdbx.CIFFile()
    pdbx.set_structure(converted, atoms, data_block="structure")
    converted["structure"]["atom_site"]["label_alt_id"] = atoms.altloc_id
    destination = temporary_dir / "components.cif"
    converted.write(destination)
    return destination


def _anchor_centroid(coordinates: dict[str, np.ndarray], name: str) -> np.ndarray:
    anchors = np.concatenate([coordinates["CA"], coordinates["P"]], axis=0)
    anchors = anchors[np.isfinite(anchors).all(axis=-1)]
    if not len(anchors):
        raise ValueError(
            f"{name} has no finite CA/P anchors for origin-shift correction"
        )
    return np.mean(anchors, axis=0, dtype=np.float64)


def run(
    *,
    map_path: str | Path,
    structure_path: str | Path,
    artifact_reference: str | Path,
    target_voxel_size: float = 1.0,
    original_structure_path: str | Path | None = None,
    fasta_path: str | Path | None = None,
    fix_shift: bool = True,
    interface_contact_cutoff: float = 5.0,
    s3_endpoint_url: str | None = None,
) -> None:
    """Write one complete NPZ+JSON artifact from local raw inputs.

    An optional original structure supplies all stored structure targets. With
    ``fix_shift``, the map origin is translated from the fitted structure's
    coordinate frame to that original structure's CA/P anchor-centroid frame.
    Inputs may be gzip-compressed. FASTA records are portable metadata and do
    not supply sequence embeddings.
    """
    target_voxel_size = _positive_finite(target_voxel_size, "target_voxel_size")
    interface_contact_cutoff = _positive_finite(
        interface_contact_cutoff, "interface_contact_cutoff"
    )
    if not isinstance(fix_shift, (bool, np.bool_)):
        raise ValueError("fix_shift must be a boolean")
    # Resolve every supplied input and reject unsupported output spellings
    # before creating or publishing any artifact content.
    map_path = _input_path(map_path, "map_path", {".mrc", ".map"})
    structure_suffixes = {".cif", ".mmcif", ".pdb", ".ent"}
    structure_path = _input_path(structure_path, "structure_path", structure_suffixes)
    if original_structure_path is not None:
        original_structure_path = _input_path(
            original_structure_path, "original_structure_path", structure_suffixes
        )
    if fasta_path is not None:
        fasta_path = _input_path(fasta_path, "fasta_path", set())
    preprocessed_artifact_root(artifact_reference)

    with tempfile.TemporaryDirectory(prefix="cryoflake_preprocess_") as temporary:
        temporary_dir = Path(temporary)
        map_path = _materialize_input(map_path, temporary_dir, "density")
        structure_path = _materialize_input(structure_path, temporary_dir, "fitted")
        target_structure_path = (
            _materialize_input(original_structure_path, temporary_dir, "original")
            if original_structure_path is not None
            else structure_path
        )
        density_grid, voxel_size, global_origin = load_mrc(map_path)
        density_grid = validate_density_grid(density_grid)
        global_origin = validate_global_origin(global_origin)
        voxel_size = validate_voxel_size(voxel_size)
        density_grid, voxel_size, global_origin = prepare_grid(
            density_grid, voxel_size, global_origin, target_voxel_size
        )
        coordinates, residue_info = load_full_backbone_from_structure(
            target_structure_path
        )
        if original_structure_path is not None and fix_shift:
            fitted_coordinates, _ = load_full_backbone_from_structure(structure_path)
            global_origin = validate_global_origin(
                np.asarray(global_origin, dtype=np.float64)
                + _anchor_centroid(coordinates, "original_structure_path")
                - _anchor_centroid(fitted_coordinates, "structure_path")
            )
        full_atom_targets = validate_full_atom_targets(
            load_full_atom_targets_from_structure(target_structure_path),
            num_protein=len(coordinates["CA"]),
            num_nucleotide=len(coordinates["P"]),
        )
        atom_mask = full_atom_targets["atom_mask"]
        polymer_atom_node_indices = np.broadcast_to(
            np.arange(len(atom_mask), dtype=np.int64)[:, None], atom_mask.shape
        )[atom_mask]
        extension_targets = build_component_extension_targets(
            _component_structure_path(target_structure_path, temporary_dir),
            global_origin=global_origin,
            voxel_size=voxel_size,
            grid_shape=density_grid.shape,
            polymer_atom_positions=full_atom_targets["atom_positions"][atom_mask],
            polymer_atom_node_indices=polymer_atom_node_indices,
            polymer_chain_indices=full_atom_targets["chain_index"],
            polymer_residue_indices=full_atom_targets["residue_index"],
            polymer_aatype=full_atom_targets["aatype"],
            interface_contact_cutoff=interface_contact_cutoff,
        )
        shape = np.asarray(density_grid.shape, dtype=np.int64)
        artifact_mapping = {
            "density_grid": density_grid,
            "voxel_size": voxel_size,
            "global_origin": global_origin,
            "shape": shape,
            "cella": shape * voxel_size,
            "grid_spectral_amplitude": spectral_amplitude(np.fft.rfftn(density_grid)),
            "backbone_coordinates": coordinates,
            "residue_info": residue_info,
            "full_atom_targets": full_atom_targets,
            "extension_targets": extension_targets,
        }
        if fasta_path is not None:
            fasta_path = _materialize_input(fasta_path, temporary_dir, "sequence")
            sequences, sequence_names = read_fasta(fasta_path)
            artifact_mapping["fasta"] = {
                "sequences": [sequence._asdict() for sequence in sequences],
                "sequence_names": sequence_names,
                "chain_to_seqid": {
                    chain: sequence_index
                    for sequence_index, sequence in enumerate(sequences)
                    for chain in sequence.chain_ids
                },
            }
        if not _preprocessed_artifact_is_reusable(artifact_mapping):
            raise ValueError(
                "Default preprocessing does not satisfy the canonical preprocessing "
                "contract; check the structure's finite CA/P anchors and map alignment"
            )
        write_preprocessed_artifact(
            artifact_reference, artifact_mapping, s3_endpoint_url=s3_endpoint_url
        )
