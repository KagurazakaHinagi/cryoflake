import copy
import hashlib

import numpy as np
import pytest

from cryoflake.constants import AnchorKind, NodeKind, PolymerKind
from cryoflake.targets import components
from cryoflake.targets.components import build_component_extension_targets
from cryoflake.targets.schema import validate_component_extension_targets


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_blocked_interface_contacts_preserve_exact_dense_results(dtype):
    rng = np.random.default_rng(731)
    component = rng.normal(size=(257, 3)).astype(dtype)
    polymer = rng.normal(size=(2053, 3)).astype(dtype)
    # Include the cutoff itself and the next representable value outside it.
    component[0] = 0
    polymer[:2] = [[0, 0, 0.5], [0, 0, np.nextafter(dtype(0.5), dtype(1))]]
    delta = component[:, None, :] - polymer[None, :, :]
    expected = np.any(np.sum(delta * delta, axis=-1) <= 0.25, axis=0)
    actual = components._contacting_polymer_atoms(component, polymer, 0.25)
    np.testing.assert_array_equal(actual, expected)


def test_interface_contact_workspace_is_bounded_across_both_atom_axes(monkeypatch):
    rng = np.random.default_rng(732)
    component = rng.normal(size=(41, 3)).astype(np.float32)
    polymer = rng.normal(size=(103, 3)).astype(np.float32)
    delta = component[:, None, :] - polymer[None, :, :]
    expected = np.any(np.sum(delta * delta, axis=-1) <= 0.01, axis=0)
    square = np.square
    block_shapes = []

    def record_square(values, **kwargs):
        block_shapes.append(values.shape)
        return square(values, **kwargs)

    monkeypatch.setattr(components, "_CONTACT_PAIR_BLOCK_SIZE", 17)
    monkeypatch.setattr(components.np, "square", record_square)
    actual = components._contacting_polymer_atoms(component, polymer, 0.01)
    np.testing.assert_array_equal(actual, expected)
    assert len(block_shapes) > 1
    assert all(len(shape) == 2 for shape in block_shapes)
    assert max(shape[0] * shape[1] for shape in block_shapes) <= 17


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interface_contact_cutoff_preserves_reduction_rounding(dtype):
    component = np.zeros((1, 3), dtype=dtype)
    polymer = np.asarray([[0.003, 0.051, 0.007]], dtype=dtype)
    cutoff = np.sum(polymer[0] * polymer[0])
    assert components._contacting_polymer_atoms(component, polymer, cutoff).item()
    assert not components._contacting_polymer_atoms(
        component, polymer, np.nextafter(cutoff, dtype(0))
    ).item()


def _write_mixed_cif(path):
    path.write_text(
        """data_mixed
loop_
_chem_comp.id
_chem_comp.type
ALA 'L-PEPTIDE LINKING'
MSE 'L-PEPTIDE LINKING'
LIG NON-POLYMER
ZN NON-POLYMER
HOH NON-POLYMER
UNL NON-POLYMER
loop_
_entity.id
_entity.type
1 polymer
2 non-polymer
3 non-polymer
4 water
5 non-polymer
loop_
_entity_poly.entity_id
_entity_poly.type
1 polypeptide(L)
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_PDB_model_num
_atom_site.type_symbol
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM N . ALA A 1 1 A 1 ? 1 N 1.0 20.0 0.0 0.0 0.0
ATOM CA . ALA A 1 1 A 1 ? 1 C 1.0 20.0 1.0 0.0 0.0
ATOM C . ALA A 1 1 A 1 ? 1 C 1.0 20.0 2.0 0.0 0.0
HETATM N . MSE A 1 2 A 2 ? 1 N 1.0 20.0 10.0 0.0 0.0
HETATM CA . MSE A 1 2 A 2 ? 1 C 1.0 20.0 11.0 0.0 0.0
HETATM C . MSE A 1 2 A 2 ? 1 C 1.0 20.0 12.0 1.0 0.0
HETATM C1 A LIG B 2 . B 3 ? 1 C 0.4 20.0 1.0 0.0 0.0
HETATM C1 B LIG B 2 . B 3 ? 1 C 0.8 20.0 5.0 0.0 0.0
HETATM O1 . LIG B 2 . B 3 ? 1 O 1.0 20.0 3.0 0.0 0.0
HETATM ZN . ZN C 3 . C 4 ? 1 ZN 1.0 20.0 25.0 0.0 0.0
HETATM O . HOH D 4 . D 5 ? 1 O 1.0 20.0 16.0 0.0 0.0
HETATM C1 . UNL E 5 . E 6 ? 1 C 1.0 20.0 3.0 0.0 0.0
HETATM C2 . UNL E 5 . E 6 ? 1 C 1.0 20.0 5.0 0.0 0.0
"""
    )


def _write_canonical_cif(path):
    path.write_text(
        """data_canonical
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_PDB_model_num
_atom_site.type_symbol
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM N . ALA A 1 1 A 1 ? 1 N 1.0 20.0 0.0 0.0 0.0
ATOM CA . ALA A 1 1 A 1 ? 1 C 1.0 20.0 1.0 0.0 0.0
ATOM C . ALA A 1 1 A 1 ? 1 C 1.0 20.0 2.0 0.0 0.0
"""
    )


def _build(path, **kwargs):
    return build_component_extension_targets(
        path,
        global_origin=np.zeros(3, dtype=np.float32),
        voxel_size=1.0,
        grid_shape=(20, 20, 20),
        **kwargs,
    )


@pytest.mark.parametrize("field", ["positions", "atom_positions", "b_factors"])
def test_component_targets_reject_values_that_overflow_float32(tmp_path, field):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = _build(structure_path)
    component = targets["component"]
    component[field] = component[field].astype(np.float64)
    component[field].flat[0] = 1e100

    with pytest.raises(ValueError, match="finite"):
        validate_component_extension_targets(targets)


def test_component_residue_indices_do_not_wrap_uint64(tmp_path):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = _build(structure_path)
    component = targets["component"]
    component["residue_index"] = np.full(
        component["residue_index"].shape, 2**63, dtype=np.uint64
    )
    with pytest.raises(ValueError, match="int64|representable"):
        validate_component_extension_targets(targets)


def test_build_component_targets_covers_typed_nodes_and_interfaces(tmp_path):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)

    targets = _build(
        structure_path,
        polymer_atom_positions=np.asarray([[0.0, 0.0, 0.0]]),
        polymer_atom_node_indices=np.asarray([0]),
        polymer_chain_indices=np.asarray([2]),
        polymer_residue_indices=np.asarray([10]),
        polymer_aatype=np.asarray([5]),
        interface_contact_cutoff=5.0,
    )
    component = targets["component"]

    assert component["component_id"].tolist() == [
        "MSE",
        "LIG",
        "ZN",
        "HOH",
        "UNL",
    ]
    assert component["node_kind"].tolist() == [
        int(NodeKind.POLYMER),
        int(NodeKind.NON_POLYMER),
        int(NodeKind.ION),
        int(NodeKind.WATER),
        int(NodeKind.UNKNOWN),
    ]
    assert component["polymer_kind"].tolist() == [
        int(PolymerKind.PROTEIN),
        int(PolymerKind.NONE),
        int(PolymerKind.NONE),
        int(PolymerKind.NONE),
        int(PolymerKind.NONE),
    ]
    assert component["anchor_kind"].tolist() == [
        int(AnchorKind.C_ALPHA),
        int(AnchorKind.CENTROID),
        int(AnchorKind.ATOM),
        int(AnchorKind.ATOM),
        int(AnchorKind.UNSPECIFIED),
    ]
    assert component["reference_local_positions"].shape == (5, 3, 3)
    np.testing.assert_allclose(
        component["reference_local_positions"][:, 1],
        0.0,
    )

    ligand_index = component["component_id"].tolist().index("LIG")
    ligand_atom_names = component["atom_names"][ligand_index]
    ligand_atom_mask = component["atom_mask"][ligand_index]
    assert ligand_atom_names[ligand_atom_mask].tolist() == ["C1", "O1"]
    np.testing.assert_allclose(
        component["atom_positions"][ligand_index, ligand_atom_mask, 0],
        [5.0, 3.0],
    )
    np.testing.assert_allclose(
        component["occupancies"][ligand_index, ligand_atom_mask],
        [0.8, 1.0],
    )

    expected_digest = hashlib.sha256(repr([(2, 10, 5)]).encode()).hexdigest()[:16]
    assert component["interface_group_id"][ligand_index] == (f"LIG|{expected_digest}")
    assert component["interface_contact_cutoff"] == 5.0
    assert component["split_groups"]["component"] == (
        "LIG",
        "MSE",
        "UNL",
        "ZN",
    )
    assert component["split_groups"]["interface"] == (
        f"LIG|{expected_digest}",
        f"UNL|{expected_digest}",
    )

    proposal = component["proposal_targets"]
    assert proposal["node_indices"].tolist() == list(range(5))
    np.testing.assert_array_equal(proposal["positions"], component["positions"])
    assert proposal["valid_mask"].tolist() == [
        True,
        True,
        False,
        True,
        True,
    ]
    np.testing.assert_allclose(
        component["positions"][ligand_index],
        component["positions"][-1],
    )


def test_build_component_targets_writes_valid_empty_payload(tmp_path):
    structure_path = tmp_path / "canonical.cif"
    _write_canonical_cif(structure_path)

    component = _build(structure_path)["component"]

    assert component["positions"].shape == (0, 3)
    assert component["reference_local_positions"].shape == (0, 3, 3)
    assert component["atom_positions"].shape == (0, 0, 3)
    assert component["atom_mask"].shape == (0, 0)
    assert component["proposal_targets"]["positions"].shape == (0, 3)
    assert component["split_groups"] == {"component": (), "interface": ()}


@pytest.mark.parametrize(
    "file_name",
    ("components.pdb", "components.ent", "components.cif.gz"),
)
def test_build_component_targets_rejects_non_cif_structures(
    tmp_path,
    file_name,
):
    structure_path = tmp_path / file_name
    structure_path.write_text("unsupported")

    with pytest.raises(ValueError, match=r"uncompressed \.cif"):
        _build(structure_path)


def test_build_component_targets_uses_mmcif_chemistry_metadata(tmp_path):
    structure_path = tmp_path / "components.cif"
    structure_path.write_text(
        """data_components
loop_
_chem_comp.id
_chem_comp.type
MSE 'L-PEPTIDE LINKING'
NAG 'D-SACCHARIDE'
PSU 'RNA LINKING'
D5M 'DNA LINKING'
HBR 'RNA/DNA LINKING'
PXL OTHER
loop_
_entity.id
_entity.type
1 polymer
2 branched
3 polymer
4 polymer
5 polymer
6 polymer
loop_
_entity_poly.entity_id
_entity_poly.type
1 polypeptide(L)
3 polyribonucleotide
4 polydeoxyribonucleotide
5 polydeoxyribonucleotide/polyribonucleotide_hybrid
6 other
loop_
_pdbx_entity_branch.entity_id
_pdbx_entity_branch.type
2 oligosaccharide
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_PDB_model_num
_atom_site.type_symbol
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
HETATM N . MSE A 1 1 A 11 ? 1 N 1.0 20.0 1.0 1.0 1.0
HETATM CA . MSE A 1 1 A 11 ? 1 C 1.0 20.0 2.0 1.0 1.0
HETATM C . MSE A 1 1 A 11 ? 1 C 1.0 20.0 3.0 2.0 1.0
HETATM C1 . NAG B 2 1 B 21 ? 1 C 1.0 20.0 5.0 5.0 5.0
HETATM O5 . NAG B 2 1 B 21 ? 1 O 1.0 20.0 6.0 5.0 5.0
HETATM C2 . NAG B 2 1 B 21 ? 1 C 1.0 20.0 5.0 6.0 5.0
HETATM P . PSU C 3 1 C 31 ? 1 P 1.0 20.0 7.0 7.0 7.0
HETATM OP1 . PSU C 3 1 C 31 ? 1 O 1.0 20.0 8.0 7.0 7.0
HETATM OP2 . PSU C 3 1 C 31 ? 1 O 1.0 20.0 7.0 8.0 7.0
HETATM P . D5M D 4 1 D 41 ? 1 P 1.0 20.0 9.0 9.0 9.0
HETATM OP1 . D5M D 4 1 D 41 ? 1 O 1.0 20.0 10.0 9.0 9.0
HETATM OP2 . D5M D 4 1 D 41 ? 1 O 1.0 20.0 9.0 10.0 9.0
HETATM P . HBR E 5 1 E 51 ? 1 P 1.0 20.0 11.0 11.0 11.0
HETATM OP1 . HBR E 5 1 E 51 ? 1 O 1.0 20.0 12.0 11.0 11.0
HETATM OP2 . HBR E 5 1 E 51 ? 1 O 1.0 20.0 11.0 12.0 11.0
HETATM C1 . PXL F 6 1 F 61 ? 1 C 1.0 20.0 13.0 13.0 13.0
HETATM C2 . PXL F 6 1 F 61 ? 1 C 1.0 20.0 14.0 13.0 13.0
"""
    )

    component = _build(structure_path)["component"]

    assert component["component_id"].tolist() == [
        "MSE",
        "NAG",
        "PSU",
        "D5M",
        "HBR",
        "PXL",
    ]
    assert component["node_kind"].tolist() == [
        int(NodeKind.POLYMER),
        int(NodeKind.POLYMER),
        int(NodeKind.POLYMER),
        int(NodeKind.POLYMER),
        int(NodeKind.POLYMER),
        int(NodeKind.POLYMER),
    ]
    assert component["polymer_kind"].tolist() == [
        int(PolymerKind.PROTEIN),
        int(PolymerKind.CARBOHYDRATE),
        int(PolymerKind.RNA),
        int(PolymerKind.DNA),
        int(PolymerKind.NUCLEIC_ACID),
        int(PolymerKind.OTHER),
    ]
    assert component["anchor_kind"].tolist() == [
        int(AnchorKind.C_ALPHA),
        int(AnchorKind.ATOM),
        int(AnchorKind.PHOSPHORUS),
        int(AnchorKind.PHOSPHORUS),
        int(AnchorKind.PHOSPHORUS),
        int(AnchorKind.CENTROID),
    ]
    assert component["source_chem_comp_type"].tolist() == [
        "L-PEPTIDE LINKING",
        "D-SACCHARIDE",
        "RNA LINKING",
        "DNA LINKING",
        "RNA/DNA LINKING",
        "OTHER",
    ]
    assert component["residue_index"].tolist() == [11, 21, 31, 41, 51, 61]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda component: component["node_kind"].__setitem__(0, 99),
            "node_kind contains an unsupported value",
        ),
        (
            lambda component: component["reference_local_positions"].__setitem__(
                (0, 1, 0),
                1.0,
            ),
            "anchor-centred",
        ),
        (
            lambda component: component["split_groups"].__setitem__(
                "component",
                ("OTHER",),
            ),
            "split groups do not match",
        ),
        (
            lambda component: component["anchor_kind"].__setitem__(
                -1,
                int(AnchorKind.CENTROID),
            ),
            "UNKNOWN nodes require UNSPECIFIED",
        ),
        (
            lambda component: component["anchor_kind"].__setitem__(
                2,
                int(AnchorKind.CENTROID),
            ),
            "ION and WATER nodes require ATOM",
        ),
    ],
)
def test_validate_component_targets_rejects_corrupt_payload(
    tmp_path,
    mutate,
    message,
):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = copy.deepcopy(_build(structure_path))
    mutate(targets["component"])

    with pytest.raises(ValueError, match=message):
        validate_component_extension_targets(targets)


@pytest.mark.parametrize("component_id", [" ", "\t\n"])
def test_component_ids_remain_nonempty_after_normalization(tmp_path, component_id):
    structure = tmp_path / "mixed.cif"
    _write_mixed_cif(structure)
    targets = _build(structure)
    component = targets["component"]
    water_index = component["component_id"].tolist().index("HOH")
    component["component_id"][water_index] = component_id

    with pytest.raises(ValueError, match="component_id must not contain empty values"):
        validate_component_extension_targets(targets)


def test_validate_component_targets_rejects_malformed_interface_id(tmp_path):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = _build(
        structure_path,
        polymer_atom_positions=np.asarray([[0.0, 0.0, 0.0]]),
        polymer_atom_node_indices=np.asarray([0]),
        polymer_chain_indices=np.asarray([2]),
        polymer_residue_indices=np.asarray([10]),
        polymer_aatype=np.asarray([5]),
    )
    component = targets["component"]
    ligand_index = component["component_id"].tolist().index("LIG")
    component["interface_group_id"][ligand_index] = "LIG|not-a-digest"
    component["split_groups"]["interface"] = tuple(
        sorted(component["interface_group_id"][component["interface_mask"]])
    )

    with pytest.raises(ValueError, match="16 lowercase hex"):
        validate_component_extension_targets(targets)


def test_component_context_requires_exact_polymer_topology(tmp_path):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)

    with pytest.raises(ValueError, match="polymer chain, residue, and aatype"):
        _build(
            structure_path,
            polymer_atom_positions=np.asarray([[0.0, 0.0, 0.0]]),
            polymer_atom_node_indices=np.asarray([0]),
        )


def test_component_mmcif_preserves_first_model_and_residue_instances(tmp_path):
    structure_path = tmp_path / "instances.mmcif"
    structure_path.write_text("""data_instances
_chem_comp.id LIG
_chem_comp.type 'NON-POLYMER'
_chem_comp.name
;
A multiline component name with spaces
;
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_PDB_model_num
_atom_site.type_symbol
_atom_site.occupancy
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
HETATM 'C1' B LIG label1 1 . 'chain A' 10 ? 7 C 0.5 1 1 1
HETATM 'C1' A LIG label1 1 . 'chain A' 10 ? 7 C 0.5 2 2 2
HETATM 'O1' . LIG label1 1 . 'chain A' 10 ? 7 O 1.0 3 3 3
HETATM 'C1' . LIG label2 1 . 'chain A' 10 ? 7 C 1.0 4 4 4
HETATM 'C1' . LIG label1 1 . 'chain A' 10 A 7 C 1.0 5 5 5
HETATM 'C1' . LIG label1 1 . 'chain A' 10 ? 9 C 1.0 99 99 99
""")
    component = _build(structure_path)["component"]
    assert component["component_id"].tolist() == ["LIG", "LIG", "LIG"]
    assert component["chain_id"].tolist() == ["chain A"] * 3
    assert component["residue_index"].tolist() == [10, 10, 10]
    assert component["insertion_code"].tolist() == ["", "", "A"]
    assert len(set(component["instance_id"].tolist())) == 3
    # Altloc occupancy ties use the identifier; models are selected by appearance.
    np.testing.assert_allclose(component["atom_positions"][0, 0], [2, 2, 2])
    np.testing.assert_allclose(component["positions"][1:], [[4, 4, 4], [5, 5, 5]])
    assert component["source_chem_comp_type"].tolist() == ["NON-POLYMER"] * 3


def test_component_mmcif_ignores_invalid_coordinates_and_selects_residue_variant(
    tmp_path,
):
    structure_path = tmp_path / "variants.cif"
    structure_path.write_text("""data_variants
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_model_num
_atom_site.type_symbol
_atom_site.occupancy
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
HETATM C1 AAA A 1 1 C 0.1 1 1 1
HETATM C1 BBB A 1 1 C 0.9 2 2 2
HETATM O1 BBB A 1 1 O 0.9 3 3 3
HETATM C2 BBB A 1 1 C 1.0 ? 4 4
""")
    component = _build(structure_path)["component"]
    assert component["component_id"].tolist() == ["BBB"]
    assert component["atom_mask"].sum() == 2
    assert np.isfinite(component["atom_positions"]).all()
    np.testing.assert_allclose(component["positions"], [[2.5, 2.5, 2.5]])
