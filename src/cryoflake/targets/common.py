"""Shared molecular-component records, classification, and geometry helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cryoflake.constants import (
    COMMON_CARBOHYDRATE_IDS,
    ION_ELEMENTS,
    UNKNOWN_COMPONENT_IDS,
    WATER_COMPONENT_IDS,
    AnchorKind,
    NodeKind,
    PolymerKind,
)


@dataclass(frozen=True)
class _Atom:
    name: str
    element: str
    position: np.ndarray
    occupancy: float
    b_factor: float
    altloc: str
    source_index: int
    group_pdb: str = ""


@dataclass(frozen=True)
class _Node:
    component_id: str
    instance_id: str
    chain_id: str
    residue_index: int
    insertion_code: str
    source_chem_comp_type: str
    node_kind: int
    polymer_kind: int
    atoms: tuple[_Atom, ...]


def _infer_element(atom_name: str) -> str:
    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        return "X"
    two_letter = letters[:2].upper()
    if two_letter in ION_ELEMENTS:
        return two_letter
    return letters[0].upper()


def _select_altloc(atoms: Sequence[_Atom]) -> _Atom:
    return sorted(
        atoms,
        key=lambda atom: (
            -atom.occupancy,
            0 if not atom.altloc else 1,
            atom.altloc,
            atom.source_index,
        ),
    )[0]


def _deduplicate_atoms(atoms: Sequence[_Atom]) -> tuple[_Atom, ...]:
    by_name: dict[str, list[_Atom]] = defaultdict(list)
    for atom in atoms:
        by_name[atom.name].append(atom)
    return tuple(_select_altloc(by_name[name]) for name in sorted(by_name))


def _heavy_atoms(atoms: Sequence[_Atom]) -> tuple[_Atom, ...]:
    heavy = tuple(atom for atom in atoms if atom.element not in {"H", "D"})
    return heavy or tuple(atoms)


def _classify(
    *,
    component_id: str,
    atoms: Sequence[_Atom],
    entity_type: str = "",
    polymer_type: str = "",
    branch_type: str = "",
    chem_comp_type: str = "",
    group_pdb: str = "",
) -> tuple[int, int]:
    entity_type = entity_type.lower()
    polymer_type = polymer_type.lower()
    branch_type = branch_type.lower()
    chem_comp_type = chem_comp_type.lower()

    if entity_type == "water" or component_id in WATER_COMPONENT_IDS:
        return int(NodeKind.WATER), int(PolymerKind.NONE)
    if (
        entity_type == "branched"
        or "oligosaccharide" in branch_type
        or "saccharide" in polymer_type
        or "saccharide" in chem_comp_type
        or component_id in COMMON_CARBOHYDRATE_IDS
    ):
        return int(NodeKind.POLYMER), int(PolymerKind.CARBOHYDRATE)
    if entity_type == "polymer" or polymer_type or "linking" in chem_comp_type:
        if "polypeptide" in polymer_type or "peptide" in chem_comp_type:
            polymer_kind = PolymerKind.PROTEIN
        elif "hybrid" in polymer_type:
            polymer_kind = PolymerKind.NUCLEIC_ACID
        elif "deoxyribonucleotide" in polymer_type or "dna" in chem_comp_type:
            polymer_kind = PolymerKind.DNA
        elif "ribonucleotide" in polymer_type or "rna" in chem_comp_type:
            polymer_kind = PolymerKind.RNA
        else:
            polymer_kind = PolymerKind.OTHER
        return int(NodeKind.POLYMER), int(polymer_kind)

    if component_id in UNKNOWN_COMPONENT_IDS:
        return int(NodeKind.UNKNOWN), int(PolymerKind.NONE)
    if entity_type in {"non-polymer", "macrolide"} or group_pdb == "HETATM":
        heavy_atoms = _heavy_atoms(atoms)
        if len(heavy_atoms) == 1 and heavy_atoms[0].element in ION_ELEMENTS:
            return int(NodeKind.ION), int(PolymerKind.NONE)
        return int(NodeKind.NON_POLYMER), int(PolymerKind.NONE)
    if group_pdb == "ATOM":
        return int(NodeKind.POLYMER), int(PolymerKind.OTHER)
    return int(NodeKind.UNKNOWN), int(PolymerKind.NONE)


def _canonical_reference(
    atoms: Sequence[_Atom],
    anchor: np.ndarray,
    preferred_names: tuple[str, str] = (),
) -> np.ndarray:
    """Return an anchor-centred, rigid-transform-invariant three-point geometry."""

    by_name = {atom.name: atom for atom in atoms}
    candidates = [
        atom
        for atom in _heavy_atoms(atoms)
        if np.linalg.norm(atom.position - anchor) > 1e-6
    ]
    if not candidates:
        return np.zeros((3, 3), dtype=np.float32)

    preferred_first = by_name.get(preferred_names[0]) if preferred_names else None
    if (
        preferred_first is not None
        and np.linalg.norm(preferred_first.position - anchor) > 1e-6
    ):
        first = preferred_first
    else:
        first = sorted(
            candidates,
            key=lambda atom: (
                -float(np.linalg.norm(atom.position - anchor)),
                atom.name,
            ),
        )[0]
    first_vector = first.position - anchor
    first_length = float(np.linalg.norm(first_vector))
    x_axis = -first_vector / first_length

    preferred_third = by_name.get(preferred_names[1]) if preferred_names else None
    if (
        preferred_third is not None
        and np.linalg.norm(preferred_third.position - anchor) > 1e-6
    ):
        third = preferred_third
    else:
        remaining = [atom for atom in candidates if atom.name != first.name]
        if not remaining:
            return np.asarray(
                [[-first_length, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            )
        third = sorted(
            remaining,
            key=lambda atom: (
                -float(
                    np.linalg.norm(
                        (atom.position - anchor)
                        - np.dot(atom.position - anchor, x_axis) * x_axis
                    )
                ),
                -float(np.linalg.norm(atom.position - anchor)),
                atom.name,
            ),
        )[0]

    third_vector = third.position - anchor
    third_x = float(np.dot(third_vector, x_axis))
    third_y = float(np.linalg.norm(third_vector - third_x * x_axis))
    return np.asarray(
        [
            [-first_length, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [third_x, third_y, 0.0],
        ],
        dtype=np.float32,
    )


def _anchor_and_reference(node: _Node) -> tuple[np.ndarray, int, np.ndarray]:
    atoms = node.atoms
    by_name = {atom.name: atom for atom in atoms}
    heavy = _heavy_atoms(atoms)

    if node.polymer_kind == int(PolymerKind.PROTEIN) and {"N", "CA", "C"}.issubset(
        by_name
    ):
        anchor = by_name["CA"].position
        anchor_kind = AnchorKind.C_ALPHA
        preferred = ("N", "C")
    elif (
        node.polymer_kind
        in {
            int(PolymerKind.RNA),
            int(PolymerKind.DNA),
            int(PolymerKind.NUCLEIC_ACID),
        }
        and "P" in by_name
    ):
        anchor = by_name["P"].position
        anchor_kind = AnchorKind.PHOSPHORUS
        preferred = ("OP1", "OP2")
    elif node.polymer_kind == int(PolymerKind.CARBOHYDRATE) and (
        "C1" in by_name or "C1'" in by_name
    ):
        anchor_name = "C1" if "C1" in by_name else "C1'"
        anchor = by_name[anchor_name].position
        anchor_kind = AnchorKind.ATOM
        preferred = ("O5", "C2")
    elif len(heavy) == 1:
        anchor = heavy[0].position
        anchor_kind = AnchorKind.ATOM
        preferred = ()
    else:
        anchor = np.mean(np.stack([atom.position for atom in heavy], axis=0), axis=0)
        anchor_kind = AnchorKind.CENTROID
        preferred = ()

    if node.node_kind == int(NodeKind.UNKNOWN):
        anchor_kind = AnchorKind.UNSPECIFIED
    reference = _canonical_reference(atoms, anchor, preferred)
    return (
        np.asarray(anchor, dtype=np.float32),
        int(anchor_kind),
        reference,
    )


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=np.str_)
