"""Full-atom structure parsing, observed geometry, and missing-data supervision."""

import numpy as np
import pytest

from cryoflake.constants import FULL_ATOM_TARGET_KEYS, FULL_ATOM_TOPOLOGY_KEYS
from cryoflake.io.structure import load_full_backbone_from_structure
from cryoflake.targets.structure import load_full_atom_targets_from_structure


def _atom(
    serial,
    name,
    residue,
    chain,
    residue_id,
    xyz,
    *,
    altloc="",
    insertion="",
    b_factor=21.5,
    occupancy=1,
):
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {name:^4s}{altloc:1s}{residue:>3s} {chain:1s}"
        f"{residue_id:4d}{insertion:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{b_factor:6.2f}          {name[0]:>2s}  \n"
    )


def _structure(tmp_path, residues, filename="structure.pdb"):
    records = []
    for residue, chain, residue_id, atoms in residues:
        records.extend(
            _atom(len(records) + index + 1, name, residue, chain, residue_id, xyz)
            for index, (name, xyz) in enumerate(atoms.items())
        )
    path = tmp_path / filename
    path.write_text("".join(records) + "END\n")
    return path


_ASPARTATE = {
    "N": (0, 1, 0),
    "CA": (0, 0, 0),
    "C": (1, 0, 0),
    "O": (1, 0, 1),
    "CB": (0, 0, -1),
    "CG": (0, 1, -1),
    "OD1": (1, 1, -1),
    "OD2": (-1, 1, -1),
    "OXT": (1, -1, 0),
}


def test_protein_frames_torsions_ambiguity_and_atom_slots(tmp_path):
    path = _structure(tmp_path, [("ASP", "A", 7, _ASPARTATE)])
    targets = load_full_atom_targets_from_structure(path)
    assert set(targets) == set(FULL_ATOM_TARGET_KEYS + FULL_ATOM_TOPOLOGY_KEYS)
    np.testing.assert_array_equal(targets["aatype"], [3])
    np.testing.assert_array_equal(targets["prot_mask"], [True])
    assert targets["atom_positions"].shape == (1, 65, 3)
    assert targets["atom_positions"].dtype == np.float32
    assert targets["atom_mask"].dtype == np.bool_
    np.testing.assert_allclose(targets["atom_positions"][0, 64], [1, -1, 0])
    assert targets["atom_mask"][0, 64]
    np.testing.assert_allclose(targets["b_factors"][targets["atom_mask"]], 21.5)
    np.testing.assert_allclose(targets["b_factors"][~targets["atom_mask"]], 0)
    np.testing.assert_allclose(
        targets["rigidgroups_gt_frames"][0, 0],
        np.column_stack((np.eye(3), np.zeros(3))),
    )
    np.testing.assert_array_equal(
        targets["rigidgroups_group_exists"][0], [1, 0, 0, 1, 1, 1, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        targets["rigidgroups_gt_exists"], targets["rigidgroups_group_exists"]
    )
    np.testing.assert_array_equal(
        targets["torsion_angles_mask"][0], [0, 0, 1, 1, 1, 0, 0, 0]
    )
    np.testing.assert_allclose(
        targets["torsion_angles_sin_cos"][0, 2:5], [[-1, 0], [0, 1], [1, 0]], atol=1e-6
    )
    np.testing.assert_allclose(
        targets["alt_torsion_angles_sin_cos"][0, 4], [-1, 0], atol=1e-6
    )
    assert np.flatnonzero(targets["rigidgroups_group_is_ambiguous"][0]).tolist() == [5]
    frame = targets["rigidgroups_gt_frames"][0, 5]
    alternative = targets["rigidgroups_alt_gt_frames"][0, 5]
    np.testing.assert_allclose(alternative[:, [0, 3]], frame[:, [0, 3]])
    np.testing.assert_allclose(alternative[:, 1:3], -frame[:, 1:3])


def test_missing_atoms_and_degenerate_geometry_are_masked(tmp_path):
    missing = {name: xyz for name, xyz in _ASPARTATE.items() if name not in {"O", "CG"}}
    degenerate = {"N": (0, 0, 0), "CA": (1, 0, 0), "C": (2, 0, 0), "O": (3, 0, 0)}
    path = _structure(tmp_path, [("ASP", "A", 1, missing), ("GLY", "B", 1, degenerate)])
    targets = load_full_atom_targets_from_structure(path)
    assert not targets["atom_mask"][0, 3]
    assert not targets["torsion_angles_mask"][0].any()
    assert not targets["rigidgroups_gt_exists"][0, 3:].any()
    assert not targets["rigidgroups_gt_exists"][1].any()
    assert not targets["torsion_angles_mask"][1].any()
    np.testing.assert_allclose(targets["atom_positions"][~targets["atom_mask"]], 0)
    np.testing.assert_allclose(
        targets["torsion_angles_sin_cos"][~targets["torsion_angles_mask"]], 0
    )
    np.testing.assert_allclose(
        targets["rigidgroups_gt_frames"][~targets["rigidgroups_gt_exists"]], 0
    )
    assert all(np.isfinite(value).all() for value in targets.values())


def _cytidine(*, dna=False):
    # A non-planar sugar/phosphate fragment exercises every distinct frame axis.
    atoms = {
        "P": (0, 0, 0),
        "OP1": (0, 1, 0),
        "OP2": (1, 0, 0),
        "O5'": (0, 0, 1),
        "C5'": (1, 1, 1),
        "C4'": (2, 1, 0),
        "O4'": (2, 0, 0),
        "C3'": (3, 2, 1),
        "O3'": (3, 3, 0),
        "C2'": (2, 3, 2),
        "C1'": (1, 2, 2),
        "O2'": (3, 3, 3),
        "N1": (0, 3, 2),
        "C2": (0, 4, 3),
        "O2": (0, 5, 3),
        "N3": (-1, 4, 3),
        "C4": (-2, 3, 2),
        "N4": (-3, 3, 2),
        "C5": (-2, 2, 1),
        "C6": (-1, 2, 1),
    }
    if dna:
        del atoms["O2'"]
    return atoms


def test_unknown_residues_and_nucleotides_keep_chemistry_and_backbone_order(tmp_path):
    protein = {
        name: xyz for name, xyz in _ASPARTATE.items() if name in {"N", "CA", "C", "O"}
    }
    path = _structure(
        tmp_path,
        [
            ("DN", "Z", 41, _cytidine(dna=True)),
            ("UNK", "Z", 42, protein),
            ("N", "A", 3, _cytidine()),
            ("C", "A", 4, _cytidine()),
        ],
    )
    targets = load_full_atom_targets_from_structure(path)
    coordinates, labels = load_full_backbone_from_structure(path)
    np.testing.assert_array_equal(targets["aatype"], [28, 29, 30, 25])
    np.testing.assert_array_equal(targets["prot_mask"], [1, 0, 0, 0])
    np.testing.assert_array_equal(targets["chain_index"], [1, 1, 0, 0])
    np.testing.assert_array_equal(targets["residue_index"], [42, 41, 3, 4])
    assert labels["aa_type"] == ["UNK"]
    assert labels["nuc_type"] == ["DN", "N", "C"]
    np.testing.assert_allclose(targets["atom_positions"][:1, 1], coordinates["CA"])
    np.testing.assert_allclose(targets["atom_positions"][1:, 37], coordinates["P"])
    np.testing.assert_array_equal(
        targets["rigidgroups_group_exists"][0], [1, 0, 0, 1, 0, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        targets["rigidgroups_group_exists"][1], [1, 1, 1, 1, 1, 1, 1, 0, 1]
    )
    assert targets["rigidgroups_gt_exists"][2:].all()
    assert targets["torsion_angles_mask"][2:].all()
    np.testing.assert_array_equal(
        targets["torsion_angles_mask"][1], [1, 1, 1, 1, 1, 1, 0, 1]
    )
    assert targets["torsion_angles_mask"][0, 2]  # Unknown protein psi uses O, never CB.
    assert targets["atom_mask"][2, 63] and not targets["atom_mask"][1, 63]


@pytest.mark.parametrize("change", ["chain", "gap", "distance", "nucleotide"])
def test_backbone_torsions_do_not_cross_structural_breaks(tmp_path, change):
    glycine = {"N": (1.8, 1, 0), "CA": (2, 1, 1), "C": (3, 2, 1), "O": (3, 2, 2)}
    first = ("ASP", "A", 1, _ASPARTATE)
    second = ("GLY", "A", 2, glycine)
    connected = load_full_atom_targets_from_structure(
        _structure(tmp_path, [first, second], "connected.pdb")
    )
    assert connected["torsion_angles_mask"][1, :3].all()
    if change == "chain":
        second = ("GLY", "B", 2, glycine)
    elif change == "gap":
        second = ("GLY", "A", 3, glycine)
    elif change == "distance":
        second = (
            "GLY",
            "A",
            2,
            {name: np.array(xyz) + 20 for name, xyz in glycine.items()},
        )
    elif change == "nucleotide":
        first = ("C", "A", 1, _cytidine())
    broken = load_full_atom_targets_from_structure(
        _structure(tmp_path, [first, second])
    )
    protein_row = 0 if change == "nucleotide" else 1
    assert not broken["torsion_angles_mask"][protein_row, :2].any()
    assert broken["torsion_angles_mask"][protein_row, 2]


def test_geometry_is_equivariant_under_rigid_coordinate_transform(tmp_path):
    path = _structure(tmp_path, [("ASP", "A", 1, _ASPARTATE)])
    original = load_full_atom_targets_from_structure(path)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([3, 7, -2])
    moved_atoms = {name: rotation @ xyz + shift for name, xyz in _ASPARTATE.items()}
    moved = load_full_atom_targets_from_structure(
        _structure(tmp_path, [("ASP", "A", 1, moved_atoms)], "moved.pdb")
    )
    for field in ("rigidgroups_gt_frames", "rigidgroups_alt_gt_frames"):
        present = original["rigidgroups_gt_exists"]
        frames = original[field][present]
        expected = np.concatenate(
            (
                rotation @ frames[:, :, :3],
                (rotation @ frames[:, :, 3, None] + shift[:, None]),
            ),
            axis=-1,
        )
        np.testing.assert_allclose(moved[field][present], expected, atol=1e-6)
    for field in ("torsion_angles_sin_cos", "alt_torsion_angles_sin_cos"):
        np.testing.assert_allclose(moved[field], original[field], atol=1e-6)


def test_backbone_torsions_follow_insertion_code_residue_steps(tmp_path):
    glycine = {
        "N": (1.8, 1, 0),
        "CA": (2, 1, 1),
        "C": (3, 2, 1),
        "O": (3, 2, 2),
    }
    records = [
        _atom(index + 1, name, "ASP", "A", 1, xyz)
        for index, (name, xyz) in enumerate(_ASPARTATE.items())
    ]
    records.extend(
        _atom(20 + index, name, "GLY", "A", 1, xyz, insertion="A")
        for index, (name, xyz) in enumerate(glycine.items())
    )
    path = tmp_path / "inserted.pdb"
    path.write_text("".join(records) + "END\n")
    targets = load_full_atom_targets_from_structure(path)
    np.testing.assert_array_equal(targets["residue_index"], [1, 1])
    assert targets["torsion_angles_mask"][1, :3].all()


def test_mmcif_author_ids_label_chemistry_altloc_b_factors_and_first_model(tmp_path):
    path = tmp_path / "structure.mmcif"
    path.write_text("""data_structure
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.auth_atom_id
_atom_site.label_comp_id
_atom_site.auth_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.label_alt_id
_atom_site.pdbx_PDB_model_num
_atom_site.B_iso_or_equiv
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM C CA XX ALA LIG label author 1 10 B 5 90 9 9 9
ATOM C CA XX ALA LIG label author 1 10 A 5 13 1 2 3
ATOM N N XX ALA LIG label author 1 10 A 5 14 2 2 3
ATOM P P XX DA LIG rna ? 7 ? . 5 15 7 8 9
ATOM C CA XX SER LIG second Z 2 20 . 9 16 10 11 12
""")
    targets = load_full_atom_targets_from_structure(path)
    np.testing.assert_array_equal(targets["aatype"], [0, 20])
    np.testing.assert_array_equal(targets["residue_index"], [10, 7])
    np.testing.assert_array_equal(targets["chain_index"], [0, 1])
    np.testing.assert_allclose(targets["atom_positions"][0, 1], [1, 2, 3])
    assert targets["b_factors"][0, 1] == 13
    assert targets["b_factors"][1, 37] == 15


def test_pdb_altloc_insertion_and_missing_b_factor_mmcif(tmp_path):
    path = tmp_path / "alternate.pdb"
    path.write_text(
        _atom(1, "CA", "ALA", "A", 1, (9, 9, 9), altloc="B", occupancy=0.9, b_factor=90)
        + _atom(
            2, "CA", "ALA", "A", 1, (1, 2, 3), altloc="A", occupancy=0.1, b_factor=12
        )
        + _atom(3, "CA", "GLY", "A", 1, (4, 5, 6), insertion="A")
        + "END\n"
    )
    targets = load_full_atom_targets_from_structure(path)
    np.testing.assert_allclose(targets["atom_positions"][:, 1], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(targets["residue_index"], [1, 1])
    assert targets["b_factors"][0, 1] == 12
    missing = tmp_path / "missing.cif"
    missing.write_text("""data_structure
_atom_site.group_PDB ATOM
_atom_site.type_symbol C
_atom_site.label_atom_id CA
_atom_site.label_comp_id ALA
_atom_site.label_asym_id A
_atom_site.label_seq_id 1
_atom_site.label_alt_id .
_atom_site.Cartn_x 1
_atom_site.Cartn_y 2
_atom_site.Cartn_z 3
""")
    missing_targets = load_full_atom_targets_from_structure(missing)
    assert missing_targets["atom_mask"][0, 1]
    assert not missing_targets["b_factors"].any()
