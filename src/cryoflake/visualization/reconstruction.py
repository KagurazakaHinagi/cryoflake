"""Reconstruct portable molecular views from stored artifact values."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from cryoflake.constants import (
    BACKBONE_ATOM_NAMES,
    FULL_ATOM_COUNT,
    PREPROCESSED_NUCLEOTIDE_RESIDUE_LABELS,
    PREPROCESSED_PROTEIN_RESIDUE_LABELS,
    NodeKind,
    PolymerKind,
)
from cryoflake.io.mrc import save_mrc
from cryoflake.metadata import (
    validate_coordinates,
    validate_density_grid,
    validate_global_origin,
    validate_residue_labels,
    validate_voxel_size,
)
from cryoflake.storage.store import load_preprocessed_artifact
from cryoflake.targets.schema import validate_component_extension_targets

_VISUALIZATION_ROOT_KEYS = (
    "density_grid",
    "global_origin",
    "voxel_size",
    "backbone_coordinates",
    "residue_info",
    "full_atom_targets",
    "extension_targets",
)
_PROTEIN_LETTERS = dict(
    zip(
        (
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
            "UNK",
        ),
        "ARNDCQEGHILKMFPSTWYVX",
        strict=True,
    )
)
_ATOM_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\"*+_-]{0,7}\Z")


@dataclass(frozen=True)
class VisualizationBundle:
    """Local structure, map, observed sequence, and view summary exports."""

    structure_path: Path
    density_path: Path
    sequence_path: Path
    summary_path: Path


@dataclass(frozen=True)
class _Residue:
    chain_id: str
    residue_index: int
    residue_name: str
    atom_names: list[str]
    positions: np.ndarray
    elements: list[str]
    b_factors: np.ndarray
    occupancies: np.ndarray
    insertion_code: str = ""
    polymer_kind: str | None = None
    hetero: bool = False
    chem_comp_type: str = "NON-POLYMER"
    entity_poly_type: str | None = None


def _integer_topology(targets: Mapping, key: str, residue_count: int) -> np.ndarray:
    values = np.asarray(targets[key])
    if (
        values.shape != (residue_count,)
        or values.dtype.kind not in {"i", "u"}
        or (values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max))
        or (key == "chain_index" and np.any(values < 0))
    ):
        raise ValueError(
            f"full_atom_targets[{key!r}] must contain {residue_count} "
            "valid integer topology values"
        )
    return values.astype(np.int64, copy=False)


def _polymer_topology(
    targets: Mapping | None,
    protein_count: int,
    nucleotide_count: int,
    reserved_chain_ids: set[str],
) -> tuple[list[str], np.ndarray]:
    residue_count = protein_count + nucleotide_count
    has_topology = targets is not None and any(
        key in targets for key in ("chain_index", "residue_index")
    )
    if has_topology:
        if not all(key in targets for key in ("chain_index", "residue_index")):
            raise ValueError(
                "full_atom_targets topology requires chain_index and residue_index"
            )
        chain_indices = _integer_topology(targets, "chain_index", residue_count)
        residue_indices = _integer_topology(targets, "residue_index", residue_count)
    else:
        chain_indices = np.concatenate(
            (np.zeros(protein_count, dtype=int), np.ones(nucleotide_count, dtype=int))
        )
        residue_indices = np.concatenate(
            (np.arange(1, protein_count + 1), np.arange(1, nucleotide_count + 1))
        )
    names = {}
    chain_ids = []
    used_chain_ids = set(reserved_chain_ids)
    for row_index, chain_index in enumerate(chain_indices):
        protein = row_index < protein_count
        key = (int(chain_index), protein)
        if key not in names:
            base_name = (
                f"POLYMER_{int(chain_index)}"
                if has_topology
                else "PROTEIN"
                if protein
                else "NUCLEIC_ACID"
            )
            chain_id = base_name
            suffix = 2
            while chain_id in used_chain_ids:
                chain_id = f"{base_name}_{suffix}"
                suffix += 1
            names[key] = chain_id
            used_chain_ids.add(chain_id)
        chain_ids.append(names[key])
    return chain_ids, residue_indices


def _full_atom_arrays(targets: Mapping, residue_count: int):
    arrays = {}
    for key, shape in (
        ("atom_positions", (residue_count, FULL_ATOM_COUNT, 3)),
        ("atom_mask", (residue_count, FULL_ATOM_COUNT)),
        ("b_factors", (residue_count, FULL_ATOM_COUNT)),
    ):
        value = np.asarray(targets.get(key))
        if (
            value.shape != shape
            or value.dtype.kind not in {"b", "f", "i", "u"}
            or not np.isfinite(value).all()
            or (key != "atom_mask" and value.dtype.kind == "b")
        ):
            raise ValueError(
                f"full_atom_targets[{key!r}] must be finite numeric {shape}"
            )
        if key == "atom_mask":
            if not np.isin(value, (0, 1)).all():
                raise ValueError(
                    "full_atom_targets['atom_mask'] must contain only boolean values"
                )
            value = value.astype(bool, copy=False)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                value = value.astype(np.float32, copy=False)
            if not np.isfinite(value).all():
                raise ValueError(
                    f"full_atom_targets[{key!r}] must remain finite as float32"
                )
        arrays[key] = value
    return arrays


def _load_atom_names(
    atom_names_path: str | Path, atom_mask: np.ndarray
) -> list[list[str]]:
    try:
        names = json.loads(Path(atom_names_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            "atom_names_path must contain a readable JSON atom-name list"
        ) from exc
    residue_count = len(atom_mask)
    if (
        isinstance(names, list)
        and len(names) == FULL_ATOM_COUNT
        and all(isinstance(value, str) for value in names)
    ):
        names = [names] * residue_count
    elif not (
        isinstance(names, list)
        and len(names) == residue_count
        and all(
            isinstance(row, list)
            and len(row) == FULL_ATOM_COUNT
            and all(isinstance(value, str) for value in row)
            for row in names
        )
    ):
        raise ValueError(
            f"atom_names_path must contain {FULL_ATOM_COUNT} slot names or "
            f"{residue_count} rows of {FULL_ATOM_COUNT} slot names"
        )
    normalized_names = [[name.strip().upper() for name in row] for row in names]
    for row_index, (row, mask) in enumerate(
        zip(normalized_names, atom_mask, strict=True)
    ):
        present_names = [
            name for name, present in zip(row, mask, strict=True) if present
        ]
        if any(not _ATOM_NAME.fullmatch(name) for name in present_names):
            raise ValueError(
                f"atom_names_path has a missing or invalid present atom name in residue row {row_index}"
            )
        if len(set(present_names)) != len(present_names):
            raise ValueError(
                f"atom_names_path has duplicate present atom names in residue row {row_index}"
            )
    return normalized_names


def _element_from_polymer_atom_name(atom_name: str) -> str:
    # Polymer CA is carbon, not calcium. Atom-slot order carries no element information.
    letters = atom_name.lstrip("0123456789")
    if not letters or not letters[0].isalpha():
        raise ValueError(f"Cannot determine an element for polymer atom {atom_name!r}")
    return "SE" if letters == "SE" else letters[0]


def _polymer_residues(
    artifact_mapping: Mapping, component: Mapping | None, atom_names_path
):
    backbone = artifact_mapping.get("backbone_coordinates")
    residue_info = artifact_mapping.get("residue_info")
    if not isinstance(backbone, Mapping) or not isinstance(residue_info, Mapping):
        raise ValueError(
            "Visualization requires backbone_coordinates and residue_info mappings"
        )
    coordinates = {
        name: validate_coordinates(
            backbone.get(name), f"backbone_coordinates[{name!r}]", allow_nan=True
        )
        for name in (*BACKBONE_ATOM_NAMES, "P")
    }
    protein_count = len(coordinates["CA"])
    if any(len(coordinates[name]) != protein_count for name in BACKBONE_ATOM_NAMES):
        raise ValueError(
            "Protein backbone coordinate arrays must have matching row counts"
        )
    protein_labels = validate_residue_labels(
        residue_info.get("aa_type"),
        protein_count,
        PREPROCESSED_PROTEIN_RESIDUE_LABELS,
        "residue_info['aa_type']",
    )
    nucleotide_labels = validate_residue_labels(
        residue_info.get("nuc_type"),
        len(coordinates["P"]),
        PREPROCESSED_NUCLEOTIDE_RESIDUE_LABELS,
        "residue_info['nuc_type']",
    )
    labels = protein_labels + nucleotide_labels
    targets = artifact_mapping.get("full_atom_targets")
    if targets is not None and not isinstance(targets, Mapping):
        raise ValueError("full_atom_targets must be a mapping when present")
    reserved_chain_ids = (
        set(component["chain_id"].tolist()) if component is not None else set()
    )
    chain_ids, residue_indices = _polymer_topology(
        targets, protein_count, len(nucleotide_labels), reserved_chain_ids
    )
    full_atom = None
    atom_slot_names = None
    if atom_names_path is not None:
        if targets is None:
            raise ValueError(
                "atom_names_path requires full_atom_targets in the artifact"
            )
        full_atom = _full_atom_arrays(targets, len(labels))
        atom_slot_names = _load_atom_names(atom_names_path, full_atom["atom_mask"])
    residues = []
    skipped_count = 0
    for row_index, residue_name in enumerate(labels):
        protein = row_index < protein_count
        if full_atom is not None:
            present = full_atom["atom_mask"][row_index]
            names = [
                name
                for name, selected in zip(
                    atom_slot_names[row_index], present, strict=True
                )
                if selected
            ]
            positions = full_atom["atom_positions"][row_index, present]
            b_factors = full_atom["b_factors"][row_index, present]
        else:
            candidate_names = BACKBONE_ATOM_NAMES if protein else ("P",)
            coordinate_index = row_index if protein else row_index - protein_count
            names = [
                name
                for name in candidate_names
                if np.isfinite(coordinates[name][coordinate_index]).all()
            ]
            positions = np.asarray(
                [coordinates[name][coordinate_index] for name in names],
                dtype=np.float32,
            )
            b_factors = np.zeros(len(names), dtype=np.float32)
        if not names:
            skipped_count += 1
            continue
        residues.append(
            _Residue(
                chain_id=chain_ids[row_index],
                residue_index=int(residue_indices[row_index]),
                residue_name=residue_name,
                atom_names=names,
                positions=positions,
                elements=[_element_from_polymer_atom_name(name) for name in names],
                b_factors=b_factors,
                occupancies=np.ones(len(names), dtype=np.float32),
                polymer_kind="protein" if protein else "nucleic_acid",
                chem_comp_type="L-PEPTIDE LINKING"
                if protein
                else "RNA LINKING"
                if not residue_name.startswith("D")
                else "DNA LINKING",
                entity_poly_type="polypeptide(L)"
                if protein
                else "polyribonucleotide"
                if not residue_name.startswith("D")
                else "polydeoxyribonucleotide",
            )
        )
    return residues, skipped_count


def _component_residues(component: Mapping | None) -> list[_Residue]:
    if component is None:
        return []
    residues = []
    polymer_types = {
        int(PolymerKind.PROTEIN): "polypeptide(L)",
        int(PolymerKind.RNA): "polyribonucleotide",
        int(PolymerKind.DNA): "polydeoxyribonucleotide",
        int(
            PolymerKind.NUCLEIC_ACID
        ): "polydeoxyribonucleotide/polyribonucleotide hybrid",
    }
    for row_index, residue_name in enumerate(component["component_id"]):
        present = component["atom_mask"][row_index]
        residues.append(
            _Residue(
                chain_id=str(component["chain_id"][row_index]),
                residue_index=int(component["residue_index"][row_index]),
                insertion_code=str(component["insertion_code"][row_index]),
                residue_name=str(residue_name),
                atom_names=component["atom_names"][row_index, present].tolist(),
                positions=component["atom_positions"][row_index, present],
                elements=component["atom_elements"][row_index, present].tolist(),
                b_factors=component["b_factors"][row_index, present],
                occupancies=component["occupancies"][row_index, present],
                hetero=component["node_kind"][row_index] != int(NodeKind.POLYMER),
                chem_comp_type=str(component["source_chem_comp_type"][row_index])
                or "OTHER",
                entity_poly_type=polymer_types.get(
                    int(component["polymer_kind"][row_index]), "other"
                )
                if component["node_kind"][row_index] == int(NodeKind.POLYMER)
                else None,
            )
        )
    return residues


def _write_structure(residues: list[_Residue], structure_path: Path) -> None:
    atom_count = sum(len(residue.atom_names) for residue in residues)
    atoms = AtomArray(atom_count)
    atoms.coord = np.concatenate([residue.positions for residue in residues])
    annotations = {
        "chain_id": "chain_id",
        "res_id": "residue_index",
        "ins_code": "insertion_code",
        "res_name": "residue_name",
        "hetero": "hetero",
    }
    for annotation, field in annotations.items():
        atoms.set_annotation(
            annotation,
            np.repeat(
                [getattr(residue, field) for residue in residues],
                [len(residue.atom_names) for residue in residues],
            ),
        )
    for annotation, field in (
        ("atom_name", "atom_names"),
        ("element", "elements"),
        ("b_factor", "b_factors"),
        ("occupancy", "occupancies"),
    ):
        atoms.set_annotation(
            annotation,
            np.concatenate([getattr(residue, field) for residue in residues]),
        )
    structure = pdbx.CIFFile()
    pdbx.set_structure(structure, atoms, data_block="cryoflake")
    block = structure["cryoflake"]
    # Give every stored residue instance an unambiguous label identity, while
    # retaining signed author residue indices, insertion codes, and chain IDs.
    label_indices = []
    chain_counters = {}
    polymer_types_by_chain = {}
    for residue in residues:
        chain_counters[residue.chain_id] = chain_counters.get(residue.chain_id, 0) + 1
        label_indices.extend(
            [chain_counters[residue.chain_id]] * len(residue.atom_names)
        )
        if residue.entity_poly_type is not None:
            polymer_types_by_chain.setdefault(residue.chain_id, set()).add(
                residue.entity_poly_type
            )
    block["atom_site"]["label_seq_id"] = np.asarray(label_indices, dtype=np.int64)
    # Biotite's atom_site writer preserves atom values; these chemistry metadata
    # help standalone viewers recognize polymer entities in sparse backbone views.
    chain_ids = list(dict.fromkeys(residue.chain_id for residue in residues))
    entity_ids = {chain_id: str(index + 1) for index, chain_id in enumerate(chain_ids)}
    block["atom_site"]["label_entity_id"] = np.asarray(
        [entity_ids[chain_id] for chain_id in atoms.chain_id]
    )
    block["struct_asym"] = pdbx.CIFCategory(
        {"id": chain_ids, "entity_id": list(entity_ids.values())}
    )
    block["entity"] = pdbx.CIFCategory(
        {
            "id": list(entity_ids.values()),
            "type": [
                "polymer" if chain_id in polymer_types_by_chain else "non-polymer"
                for chain_id in chain_ids
            ],
        }
    )
    components = {residue.residue_name: residue.chem_comp_type for residue in residues}
    block["chem_comp"] = pdbx.CIFCategory(
        {"id": list(components), "type": list(components.values())}
    )
    if polymer_types_by_chain:
        polymer_chain_ids = [
            chain_id for chain_id in chain_ids if chain_id in polymer_types_by_chain
        ]
        polymer_types = []
        for chain_id in polymer_chain_ids:
            types = polymer_types_by_chain[chain_id]
            polymer_types.append(
                next(iter(types))
                if len(types) == 1
                else "polydeoxyribonucleotide/polyribonucleotide hybrid"
                if types <= {"polyribonucleotide", "polydeoxyribonucleotide"}
                else "other"
            )
        block["entity_poly"] = pdbx.CIFCategory(
            {
                "entity_id": [entity_ids[chain_id] for chain_id in polymer_chain_ids],
                "type": polymer_types,
            }
        )
        polymer_residues = [
            residue for residue in residues if residue.entity_poly_type is not None
        ]
        chain_counters = {}
        sequence_indices = []
        for residue in residues:
            chain_counters[residue.chain_id] = (
                chain_counters.get(residue.chain_id, 0) + 1
            )
            if residue.entity_poly_type is not None:
                sequence_indices.append(chain_counters[residue.chain_id])
        block["entity_poly_seq"] = pdbx.CIFCategory(
            {
                "entity_id": [
                    entity_ids[residue.chain_id] for residue in polymer_residues
                ],
                "num": sequence_indices,
                "mon_id": [residue.residue_name for residue in polymer_residues],
                "hetero": ["n"] * len(polymer_residues),
            }
        )
    structure.write(structure_path)


def _chain_sequences(residues: list[_Residue]) -> list[dict]:
    chains = {}
    sequence_letters = {}
    for residue in residues:
        if residue.polymer_kind is None:
            continue
        if residue.chain_id not in chains:
            chains[residue.chain_id] = {
                "chain_id": residue.chain_id,
                "polymer_kind": residue.polymer_kind,
                "sequence": "",
                "residue_count": 0,
                "residue_indices": [],
            }
            sequence_letters[residue.chain_id] = []
        chain = chains[residue.chain_id]
        sequence_letters[residue.chain_id].append(
            _PROTEIN_LETTERS[residue.residue_name]
            if residue.polymer_kind == "protein"
            else residue.residue_name.removeprefix("D")
        )
        chain["residue_count"] += 1
        chain["residue_indices"].append(residue.residue_index)
    for chain_id, chain in chains.items():
        chain["sequence"] = "".join(sequence_letters[chain_id])
    return list(chains.values())


def reconstruct_artifact(
    artifact_reference: str | Path,
    visualization_output_dir: str | Path,
    *,
    atom_names_path: str | Path | None = None,
    s3_endpoint_url: str | None = None,
) -> VisualizationBundle:
    """Export an artifact's world-space structure, density, and observed sequences.

    Named backbone atoms are reconstructed by default. Full-atom slots have no
    built-in naming convention: supply a JSON list of 65 names, or one list per
    protein-then-nucleotide residue, to export their masked positions instead.
    Original source maps, structures, and embedding arrays are never read.
    """
    artifact_mapping = load_preprocessed_artifact(
        artifact_reference,
        root_keys=_VISUALIZATION_ROOT_KEYS,
        include_sequence_embeddings=False,
        s3_endpoint_url=s3_endpoint_url,
    )
    density_grid = validate_density_grid(artifact_mapping.get("density_grid"))
    global_origin = validate_global_origin(artifact_mapping.get("global_origin"))
    voxel_size = validate_voxel_size(artifact_mapping.get("voxel_size"))
    extension_targets = artifact_mapping.get("extension_targets")
    component = (
        validate_component_extension_targets(extension_targets)["component"]
        if extension_targets is not None
        else None
    )
    polymer_residues, skipped_count = _polymer_residues(
        artifact_mapping, component, atom_names_path
    )
    component_residues = _component_residues(component)
    residues = polymer_residues + component_residues
    if not residues:
        raise ValueError("Artifact has no present finite named atoms to visualize")
    chains = _chain_sequences(polymer_residues)
    limitations = [
        "Sequences describe only exported canonical polymer residues; they are not embedding sequences.",
        "Canonical polymer chain names are synthetic; original chain names and insertion codes are not defined by the stored backbone contract.",
    ]
    if atom_names_path is None:
        limitations.insert(
            0,
            "Backbone reconstruction contains finite CA, C, N, O and nucleotide P atoms only; full-atom slot names require atom_names_path.",
        )
    if skipped_count:
        limitations.append(
            f"Skipped {skipped_count} polymer residues with no present finite atoms."
        )
    summary = {
        "schema_version": 1,
        "artifact_reference": str(artifact_reference),
        "coordinate_units": "angstrom",
        "density": {
            "shape_zyx": list(density_grid.shape),
            "origin_xyz": global_origin.tolist(),
            "voxel_size_angstrom": voxel_size,
        },
        "reconstruction_mode": "backbone" if atom_names_path is None else "full_atom",
        "limitations": limitations,
        "atom_count": sum(len(residue.atom_names) for residue in residues),
        "residue_count": len(residues),
        "chain_count": len({residue.chain_id for residue in residues}),
        "component_count": len(component_residues),
        "sequence_count": len(chains),
        "chains": chains,
    }
    output_dir = Path(visualization_output_dir)
    bundle = VisualizationBundle(
        structure_path=output_dir / "structure.cif",
        density_path=output_dir / "density.mrc",
        sequence_path=output_dir / "sequence.fasta",
        summary_path=output_dir / "summary.json",
    )
    for output_path in (
        bundle.structure_path,
        bundle.density_path,
        bundle.sequence_path,
        bundle.summary_path,
    ):
        if output_path.exists():
            raise FileExistsError(
                f"Visualization output already exists: {output_path}; choose a new visualization_output_dir"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_structure(residues, bundle.structure_path)
    save_mrc(density_grid, voxel_size, global_origin, bundle.density_path)
    bundle.sequence_path.write_text(
        "".join(
            f">{chain['chain_id']} observed_{chain['polymer_kind']}\n{chain['sequence']}\n"
            for chain in chains
        ),
        encoding="utf-8",
    )
    bundle.summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return bundle
