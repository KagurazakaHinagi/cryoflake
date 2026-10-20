"""Biotite PDB/mmCIF parsing and serialized backbone label contracts."""

import numpy as np
import pytest

from cryoflake.io.structure import (
    CIFSecondaryStructures,
    DSSPSecondaryStructures,
    load_full_backbone_from_structure,
)


def _pdb_atom(
    serial, name, residue, chain, resid, x, *, altloc="", insertion="", occupancy=1
):
    return (
        f"ATOM  {serial:5d} {name:^4s}{altloc:1s}{residue:>3s} {chain:1s}"
        f"{resid:4d}{insertion:1s}   {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{occupancy:6.2f}{20.0:6.2f}          {name[0]:>2s}  \n"
    )


def test_pdb_per_atom_altlocs_models_and_insertion_codes(tmp_path):
    path = tmp_path / "models.pdb"
    path.write_text(
        "MODEL        1\n"
        + _pdb_atom(1, "CA", "ALA", "A", 1, 9, altloc="B", occupancy=0.9)
        + _pdb_atom(2, "CA", "ALA", "A", 1, 1, altloc="A", occupancy=0.1)
        + _pdb_atom(3, "N", "ALA", "A", 1, 2, altloc="B", occupancy=0.9)
        + _pdb_atom(4, "N", "ALA", "A", 1, 3, altloc="A", occupancy=0.1)
        + _pdb_atom(5, "CA", "GLY", "A", 1, 4, insertion="A")
        + "ENDMDL\nMODEL        7\n"
        + _pdb_atom(6, "CA", "SER", "B", 2, 5)
        + "ENDMDL\nEND\n"
    )
    first, labels = load_full_backbone_from_structure(path)
    assert labels["aa_type"] == ["ALA", "GLY"]
    np.testing.assert_allclose(first["CA"][:, 0], [1, 4])
    np.testing.assert_allclose(first["N"][0, 0], 3)
    assert np.isnan(first["O"]).all()
    assert first["P"].shape == (0, 3)

    all_models, labels = load_full_backbone_from_structure(path, all_models=True)
    assert labels["aa_type"] == ["ALA", "GLY", "SER"]
    assert labels["chain"] == ["A", "A", "B"]
    np.testing.assert_allclose(all_models["CA"][:, 0], [1, 4, 5])


def test_mmcif_author_identity_label_chemistry_and_unequal_models(tmp_path):
    path = tmp_path / "models.mmcif"
    path.write_text("""data_models
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
_atom_site.pdbx_PDB_ins_code
_atom_site.label_alt_id
_atom_site.pdbx_PDB_model_num
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM C CA XX ALA LIG label author 1 10 ? . 5 1 2 3
ATOM C CA XX GLY LIG label author 1 10 A . 5 4 5 6
ATOM P P XX DA LIG rna ? 7 ? ? . 5 7 8 9
ATOM C CA XX SER LIG second Z 2 20 ? . 9 10 11 12
""")
    first, labels = load_full_backbone_from_structure(path)
    assert labels["aa_type"] == ["ALA", "GLY"]
    assert labels["nuc_type"] == ["DA"]
    assert labels["chain"] == ["author", "author", "rna"]
    np.testing.assert_allclose(first["CA"], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(first["P"], [[7, 8, 9]])
    all_models, labels = load_full_backbone_from_structure(path, all_models=True)
    assert labels["aa_type"] == ["ALA", "GLY", "SER"]
    np.testing.assert_allclose(all_models["CA"][-1], [10, 11, 12])


def test_mmcif_secondary_structure_uses_named_columns_and_quoted_tokens(tmp_path):
    path = tmp_path / "secondary.cif"
    path.write_text("""data_secondary
loop_
_struct_conf.end_auth_seq_id
_struct_conf.beg_auth_asym_id
_struct_conf.conf_type_id
_struct_conf.end_auth_asym_id
_struct_conf.beg_auth_seq_id
_struct_conf.pdbx_PDB_helix_class
_struct_conf.details
12 'chain A' HELX_P 'chain A' 10 5
;
A multiline description with spaces
;
18 'chain A' TURN_P 'chain A' 16 ? 'turn description'
21 'chain A' HELX_P 'chain B' 20 1 'cross-chain range'
loop_
_struct_sheet_range.end_auth_seq_id
_struct_sheet_range.beg_auth_seq_id
_struct_sheet_range.end_auth_asym_id
_struct_sheet_range.beg_auth_asym_id
15 13 'chain A' 'chain A'
""")
    deposited = CIFSecondaryStructures(path)
    dssp = DSSPSecondaryStructures(path)
    assert deposited.get_residue_secondary_structure("chain A", 11) == "HELIX_5"
    assert deposited.get_residue_secondary_structure("chain A", 14) == "SHEET"
    assert deposited.get_residue_secondary_structure("chain A", 17) == "NULL"
    assert deposited.get_residue_secondary_structure("chain A", 20) == "NULL"
    assert dssp.get_residue_secondary_structure("chain A", 11) == "HELX_P"
    assert dssp.get_residue_secondary_structure("chain A", 17) == "TURN_P"
    assert dssp.get_residue_secondary_structure("chain A", 20) == "OTHER"


def test_empty_or_unsupported_structure_is_rejected(tmp_path):
    empty = tmp_path / "empty.pdb"
    empty.write_text("END\n")
    with pytest.raises(ValueError, match="contains no models"):
        load_full_backbone_from_structure(empty)
    unsupported = tmp_path / "structure.xyz"
    with pytest.raises(ValueError, match="Unsupported structure"):
        load_full_backbone_from_structure(unsupported)


def test_pdb_backbone_keeps_distinct_hetero_residues_with_same_author_number(tmp_path):
    path = tmp_path / "hetero.pdb"
    path.write_text(
        _pdb_atom(1, "CA", "ALA", "A", 1, 1)
        + _pdb_atom(2, "CA", "ALA", "A", 1, 2).replace("ATOM  ", "HETATM", 1)
        + _pdb_atom(3, "CA", "GLY", "A", 1, 3).replace("ATOM  ", "HETATM", 1)
        + "END\n"
    )
    coordinates, labels = load_full_backbone_from_structure(path)
    assert labels["aa_type"] == ["ALA", "ALA", "GLY"]
    np.testing.assert_allclose(coordinates["CA"][:, 0], [1, 2, 3])


def test_pdb_backbone_selects_disordered_residue_variant_before_atom_altloc(tmp_path):
    path = tmp_path / "variants.pdb"
    path.write_text(
        _pdb_atom(1, "CA", "SER", "A", 1, 1, altloc="A", occupancy=0.9)
        + _pdb_atom(2, "N", "SER", "A", 1, 2, altloc="A", occupancy=0.9)
        + _pdb_atom(3, "CA", "CYS", "A", 1, 3, altloc="B", occupancy=0.1)
        + _pdb_atom(4, "N", "CYS", "A", 1, 4, altloc="B", occupancy=0.1)
        + "END\n"
    )
    coordinates, labels = load_full_backbone_from_structure(path)
    assert labels["aa_type"] == ["CYS"]
    np.testing.assert_allclose(coordinates["CA"][:, 0], [3])
    np.testing.assert_allclose(coordinates["N"][:, 0], [4])


def test_pdb_backbone_groups_discontinuous_chains_and_preserves_blank_chain(tmp_path):
    path = tmp_path / "chains.pdb"
    path.write_text(
        _pdb_atom(1, "CA", "ALA", " ", 1, 1)
        + _pdb_atom(2, "CA", "GLY", "B", 1, 2)
        + _pdb_atom(3, "N", "ALA", " ", 1, 3)
        + _pdb_atom(4, "CA", "SER", " ", 2, 4)
        + "END\n"
    )
    coordinates, labels = load_full_backbone_from_structure(path)
    assert labels["chain"] == [" ", " ", "B"]
    assert labels["aa_type"] == ["ALA", "SER", "GLY"]
    np.testing.assert_allclose(coordinates["CA"][:, 0], [1, 4, 2])
    np.testing.assert_allclose(coordinates["N"][0, 0], 3)


def test_mmcif_uses_first_data_block_for_coordinates_and_secondary_labels(tmp_path):
    path = tmp_path / "blocks.cif"
    path.write_text("""data_structure
_atom_site.group_PDB ATOM
_atom_site.type_symbol C
_atom_site.label_atom_id CA
_atom_site.label_alt_id .
_atom_site.label_comp_id ALA
_atom_site.label_asym_id A
_atom_site.label_seq_id 1
_atom_site.Cartn_x 1
_atom_site.Cartn_y 2
_atom_site.Cartn_z 3
_struct_conf.conf_type_id HELX_P
_struct_conf.beg_label_asym_id A
_struct_conf.end_label_asym_id A
_struct_conf.beg_label_seq_id 1
_struct_conf.end_label_seq_id 1
data_dictionary
_chem_comp.id ALA
_chem_comp.type 'L-PEPTIDE LINKING'
""")
    coordinates, labels = load_full_backbone_from_structure(path)
    assert labels["aa_type"] == ["ALA"]
    assert labels["secondary_structure"] == ["HELIX_1"]
    np.testing.assert_allclose(coordinates["CA"], [[1, 2, 3]])
