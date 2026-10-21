"""Construct portable full-atom supervision directly from deposited structures."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from cryoflake import constants
from cryoflake.io.structure import _backbone_residues, _structure_models
from cryoflake.targets._chemistry import (
    ATOM_NAMES,
    CHI_ATOMS,
    NUCLEOTIDE_TORSIONS,
    PI_PERIODIC_CHI,
    RESIDUE_ATOMS,
    RESIDUE_NAMES,
)
from cryoflake.targets.full_atom import validate_full_atom_targets

_ATOM_INDEX = {name: index for index, name in enumerate(ATOM_NAMES)}
_RESIDUE_INDEX = {name: index for index, name in enumerate(RESIDUE_NAMES)}
_PROTEIN_NAMES = constants.PREPROCESSED_PROTEIN_RESIDUE_LABELS
_GEOMETRY_EPSILON = 1e-8


def _frame(points: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a right-handed frame from negative x, origin, and xy points."""
    negative_x, origin, xy_point = points
    x_axis = origin - negative_x
    x_length = np.linalg.norm(x_axis)
    if x_length <= _GEOMETRY_EPSILON:
        return np.zeros((3, 4), dtype=np.float32), False
    x_axis = x_axis / x_length
    y_axis = xy_point - origin
    y_axis -= np.dot(y_axis, x_axis) * x_axis
    y_length = np.linalg.norm(y_axis)
    if y_length <= _GEOMETRY_EPSILON:
        return np.zeros((3, 4), dtype=np.float32), False
    y_axis = y_axis / y_length
    return np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis), origin)).astype(
        np.float32
    ), True


def _torsion(points: np.ndarray) -> tuple[np.ndarray, bool]:
    """Encode a dihedral in the common rigid-frame sine/cosine convention."""
    frame, valid = _frame(points[[1, 2, 0]])
    if not valid:
        return np.zeros(2, dtype=np.float32), False
    local = frame[:, :3].T @ (points[3] - frame[:, 3])
    sine_cosine = local[[2, 1]]
    length = np.linalg.norm(sine_cosine)
    if length <= _GEOMETRY_EPSILON:
        return np.zeros(2, dtype=np.float32), False
    return (sine_cosine / length).astype(np.float32), True


def _frame_atoms(residue_name: str) -> dict[int, tuple[str, ...]]:
    if residue_name in _PROTEIN_NAMES:
        groups = {0: ("C", "CA", "N"), 3: ("CA", "C", "O")}
        groups.update(
            (4 + index, tuple(atoms[1:]))
            for index, atoms in enumerate(CHI_ATOMS[residue_name])
        )
        return groups
    groups = {0: ("OP2", "P", "OP1")}
    groups.update(
        (index + 1, tuple(atoms[1:]))
        for index, atoms in enumerate(NUCLEOTIDE_TORSIONS)
        if index != 6 or not residue_name.startswith("D")
    )
    groups[8] = tuple(CHI_ATOMS[residue_name][0][1:])
    return groups


def _empty_targets(residue_count: int) -> dict[str, np.ndarray]:
    float_shapes = {
        "atom_positions": (constants.FULL_ATOM_COUNT, 3),
        "b_factors": (constants.FULL_ATOM_COUNT,),
        "rigidgroups_gt_frames": (constants.RIGID_GROUP_COUNT, 3, 4),
        "rigidgroups_alt_gt_frames": (constants.RIGID_GROUP_COUNT, 3, 4),
        "torsion_angles_sin_cos": (constants.TORSION_COUNT, 2),
        "alt_torsion_angles_sin_cos": (constants.TORSION_COUNT, 2),
    }
    mask_shapes = {
        "prot_mask": (),
        "atom_mask": (constants.FULL_ATOM_COUNT,),
        "rigidgroups_gt_exists": (constants.RIGID_GROUP_COUNT,),
        "rigidgroups_group_exists": (constants.RIGID_GROUP_COUNT,),
        "rigidgroups_group_is_ambiguous": (constants.RIGID_GROUP_COUNT,),
        "torsion_angles_mask": (constants.TORSION_COUNT,),
    }
    targets = {
        name: np.zeros((residue_count, *shape), dtype=np.float32)
        for name, shape in float_shapes.items()
    }
    targets.update(
        (name, np.zeros((residue_count, *shape), dtype=np.bool_))
        for name, shape in mask_shapes.items()
    )
    targets.update(
        (name, np.zeros(residue_count, dtype=np.int64))
        for name in ("aatype", "chain_index", "residue_index")
    )
    return targets


def _set_geometry(targets, row: int, residue_name: str, previous_row: int | None):
    positions = targets["atom_positions"][row]
    mask = targets["atom_mask"][row]
    for group, names in _frame_atoms(residue_name).items():
        targets["rigidgroups_group_exists"][row, group] = True
        indices = [_ATOM_INDEX[name] for name in names]
        if not mask[indices].all():
            continue
        frame, valid = _frame(positions[indices])
        if group == 0:
            # The backbone convention mirrors local x and z, retaining chirality.
            frame[:, [0, 2]] *= -1
        targets["rigidgroups_gt_frames"][row, group] = frame
        targets["rigidgroups_gt_exists"][row, group] = valid
    targets["rigidgroups_alt_gt_frames"][row] = targets["rigidgroups_gt_frames"][row]
    ambiguous_chi = PI_PERIODIC_CHI.get(residue_name)
    if ambiguous_chi is not None:
        group = ambiguous_chi + 4
        targets["rigidgroups_group_is_ambiguous"][row, group] = True
        targets["rigidgroups_alt_gt_frames"][row, group, :, 1:3] *= -1

    def store_torsion(index, points, atoms_exist, *, mirror=False):
        if not atoms_exist:
            return
        angle, valid = _torsion(points)
        targets["torsion_angles_sin_cos"][row, index] = -angle if mirror else angle
        targets["torsion_angles_mask"][row, index] = valid

    if residue_name in _PROTEIN_NAMES:
        if previous_row is not None:
            previous_positions = targets["atom_positions"][previous_row]
            previous_mask = targets["atom_mask"][previous_row]
            store_torsion(
                0,
                np.concatenate((previous_positions[[1, 2]], positions[[0, 1]])),
                previous_mask[[1, 2]].all() and mask[[0, 1]].all(),
            )
            store_torsion(
                1,
                np.concatenate((previous_positions[[2]], positions[[0, 1, 2]])),
                previous_mask[2] and mask[[0, 1, 2]].all(),
            )
        store_torsion(2, positions[:4], mask[:4].all(), mirror=True)
        torsions = (
            (3 + index, atoms) for index, atoms in enumerate(CHI_ATOMS[residue_name])
        )
    else:
        torsions = (
            *(
                (index, atoms)
                for index, atoms in enumerate(NUCLEOTIDE_TORSIONS)
                if index != 6 or not residue_name.startswith("D")
            ),
            (7, CHI_ATOMS[residue_name][0]),
        )
    for index, atoms in torsions:
        indices = [_ATOM_INDEX[name] for name in atoms]
        store_torsion(index, positions[indices], mask[indices].all())
    targets["alt_torsion_angles_sin_cos"][row] = targets["torsion_angles_sin_cos"][row]
    if ambiguous_chi is not None:
        targets["alt_torsion_angles_sin_cos"][row, 3 + ambiguous_chi] *= -1


def _previous_protein_row(targets, residues, row: int) -> int | None:
    """Avoid supervising backbone torsions across chain breaks or sequence gaps."""
    if row == 0 or not targets["prot_mask"][row - 1]:
        return None
    if targets["chain_index"][row] != targets["chain_index"][row - 1]:
        return None
    difference = targets["residue_index"][row] - targets["residue_index"][row - 1]
    insertion_step = difference == 0 and (
        str(residues[row].ins_code[0]) != str(residues[row - 1].ins_code[0])
    )
    if difference != 1 and not insertion_step:
        return None
    if not targets["atom_mask"][row - 1, 2] or not targets["atom_mask"][row, 0]:
        return None
    distance = np.linalg.norm(
        targets["atom_positions"][row - 1, 2] - targets["atom_positions"][row, 0]
    )
    return row - 1 if 0.5 <= distance <= 2.0 else None


def load_full_atom_targets_from_structure(
    structure_path: str | Path,
) -> dict[str, np.ndarray]:
    """Load first-model PDB/mmCIF full-atom targets, protein rows before nucleotide.

    Residue selection and per-atom alternate locations match the backbone reader.
    Missing atoms, undefined torsions, and degenerate frames have false masks and
    zero values. Unavailable B factors are zero. Protein backbone torsions require
    adjacent author residue identities and a 0.5–2.0 Å peptide C–N distance.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = next(
            _structure_models(
                structure_path, all_models=False, extra_fields=["b_factor"]
            )
        )
    residues = [
        residue
        for residue in _backbone_residues(model)
        if str(residue.res_name[0]).strip().upper() in _RESIDUE_INDEX
    ]
    targets = _empty_targets(len(residues))
    chain_names = sorted({str(residue.chain_id[0]) for residue in residues})
    chain_indices = {name: index for index, name in enumerate(chain_names)}
    residue_names = []
    for row, residue in enumerate(residues):
        name = str(residue.res_name[0]).strip().upper()
        residue_names.append(name)
        protein = name in _PROTEIN_NAMES
        targets["aatype"][row] = _RESIDUE_INDEX[name]
        targets["prot_mask"][row] = protein
        targets["chain_index"][row] = chain_indices[str(residue.chain_id[0])]
        targets["residue_index"][row] = int(residue.res_id[0])
        atom_names = (*RESIDUE_ATOMS[name], "OXT") if protein else RESIDUE_ATOMS[name]
        for atom_name in atom_names:
            indices = np.flatnonzero(residue.atom_name == atom_name)
            if not len(indices):
                continue
            chosen = min(
                indices, key=lambda index: (str(residue.altloc_id[index]), int(index))
            )
            position = np.asarray(residue.coord[chosen], dtype=np.float32)
            if not np.isfinite(position).all():
                raise ValueError(
                    "Structure atom coordinates must contain only finite values."
                )
            slot = _ATOM_INDEX[atom_name]
            targets["atom_positions"][row, slot] = position
            targets["atom_mask"][row, slot] = True
            b_factor = residue.b_factor[chosen]
            if np.isfinite(b_factor):
                targets["b_factors"][row, slot] = b_factor
    for row, name in enumerate(residue_names):
        previous = (
            _previous_protein_row(targets, residues, row)
            if name in _PROTEIN_NAMES
            else None
        )
        _set_geometry(targets, row, name, previous)
    node_order = np.concatenate(
        (np.flatnonzero(targets["prot_mask"]), np.flatnonzero(~targets["prot_mask"]))
    )
    return validate_full_atom_targets(
        {name: value[node_order] for name, value in targets.items()}
    )


__all__ = ["load_full_atom_targets_from_structure"]
