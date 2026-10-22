import copy
import hashlib

import numpy as np
import pytest
from standard_target import standard_expected

from cryoflake.constants import AnchorKind, NodeKind, PolymerKind
from cryoflake.targets import components, schema
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


def test_build_component_targets_recovers_real_7ae1_ions_cluster_and_interfaces(
    standard_structure_path, standard_artifact_mapping
):
    targets = standard_artifact_mapping["full_atom_targets"]
    mask = targets["atom_mask"]
    result = build_component_extension_targets(
        standard_structure_path,
        global_origin=standard_artifact_mapping["global_origin"],
        voxel_size=standard_artifact_mapping["voxel_size"],
        grid_shape=standard_artifact_mapping["density_grid"].shape,
        polymer_atom_positions=targets["atom_positions"][mask],
        polymer_atom_node_indices=np.nonzero(mask)[0],
        polymer_chain_indices=targets["chain_index"],
        polymer_residue_indices=targets["residue_index"],
        polymer_aatype=targets["aatype"],
    )
    component = result["component"]
    expected = standard_expected()
    assert result["schema_version"] == 1
    assert len(component["positions"]) == expected["component_nodes"] == 9
    assert component["component_id"].tolist() == ["ZN"] * 7 + ["SF4", "MG"]
    assert component["node_kind"].tolist() == [int(NodeKind.ION)] * 7 + [
        int(NodeKind.NON_POLYMER),
        int(NodeKind.ION),
    ]
    assert (component["atom_names"] != "").sum(axis=1).tolist() == [1] * 7 + [8, 1]
    assert (
        int(np.count_nonzero(component["atom_names"] != ""))
        == expected["component_atom_count"]
    )
    np.testing.assert_array_equal(
        component["positions"][0],
        np.asarray([198.726, 155.234, 195.417], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        component["positions"][-1],
        np.asarray([175.353, 165.911, 152.429], dtype=np.float32),
    )
    assert np.any(component["reference_local_positions"][7] != 0)
    assert component["interface_group_id"].tolist() == [
        "ZN|55f867cb5b4d421e",
        "ZN|79e030771e58759b",
        "ZN|c1d9dfd5dc23f3ed",
        "ZN|4da705497e704cf2",
        "ZN|65f912b955d4ed7d",
        "ZN|30a69ffa40103cb3",
        "ZN|0a8e5519b9837b2b",
        "SF4|aac4bd62fec34823",
        "MG|deb4a608147bcf00",
    ]
    np.testing.assert_array_equal(
        component["proposal_targets"]["node_indices"], np.arange(9)
    )
    assert component["proposal_targets"]["valid_mask"].all()
    assert not {"atom_mask", "interface_mask", "split_groups"}.intersection(component)
    assert set(component["proposal_targets"]) == {"node_indices", "valid_mask"}


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
    ligand_atom_mask = ligand_atom_names != ""
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
    component_group_ids = tuple(
        sorted(
            set(
                component["component_id"][component["node_kind"] != int(NodeKind.WATER)]
            )
        )
    )
    assert component_group_ids == (
        "LIG",
        "MSE",
        "UNL",
        "ZN",
    )
    interface_group_ids = tuple(
        sorted(
            set(component["interface_group_id"][component["interface_group_id"] != ""])
        )
    )
    assert interface_group_ids == (
        f"LIG|{expected_digest}",
        f"UNL|{expected_digest}",
    )

    proposal = component["proposal_targets"]
    assert proposal["node_indices"].tolist() == list(range(5))
    np.testing.assert_array_equal(
        component["positions"][proposal["node_indices"]], component["positions"]
    )
    assert set(proposal) == {"node_indices", "valid_mask"}
    assert not {"atom_mask", "interface_mask", "split_groups"}.intersection(component)
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
    assert component["atom_names"].shape == component["atom_elements"].shape == (0, 0)
    assert component["interface_group_id"].shape == (0,)
    assert component["proposal_targets"]["node_indices"].shape == (0,)
    assert component["proposal_targets"]["valid_mask"].shape == (0,)
    assert not {"atom_mask", "interface_mask", "split_groups"}.intersection(component)
    assert set(component["proposal_targets"]) == {"node_indices", "valid_mask"}


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
    assert (component["atom_names"] != "").sum() == 2
    assert np.isfinite(component["atom_positions"]).all()
    np.testing.assert_allclose(component["positions"], [[2.5, 2.5, 2.5]])


@pytest.mark.parametrize("empty_components", [False, True])
@pytest.mark.parametrize(
    "removed_field", ["atom_mask", "interface_mask", "split_groups"]
)
def test_component_schema_rejects_removed_fields_even_for_empty_payloads(
    tmp_path, empty_components, removed_field
):
    structure_path = tmp_path / "structure.cif"
    (_write_canonical_cif if empty_components else _write_mixed_cif)(structure_path)
    targets = _build(structure_path)
    component = targets["component"]
    values = {
        "atom_mask": component["atom_names"] != "",
        "interface_mask": component["interface_group_id"] != "",
        "split_groups": {"component": (), "interface": ()},
    }
    component[removed_field] = values[removed_field]
    with pytest.raises(
        ValueError, match="component payload has missing or unexpected fields"
    ):
        validate_component_extension_targets(targets)


@pytest.mark.parametrize("empty_components", [False, True])
def test_component_proposals_reject_removed_duplicate_positions(
    tmp_path, empty_components
):
    structure_path = tmp_path / "structure.cif"
    (_write_canonical_cif if empty_components else _write_mixed_cif)(structure_path)
    targets = _build(structure_path)
    component = targets["component"]
    component["proposal_targets"]["positions"] = component["positions"].copy()
    with pytest.raises(
        ValueError, match="proposal_targets has missing or unexpected fields"
    ):
        validate_component_extension_targets(targets)


def test_component_atom_presence_uses_normalized_names_and_preserves_atom_values(
    standard_artifact_mapping,
):
    targets = standard_artifact_mapping["extension_targets"]
    expected = copy.deepcopy(targets["component"])
    component = targets["component"]
    for field in ("atom_names", "atom_elements"):
        component[field] = np.asarray(
            [
                [f" {name.lower()} " if name else "\t " for name in row]
                for row in component[field]
            ]
        )
    normalized = validate_component_extension_targets(targets)["component"]
    for field in (
        "atom_names",
        "atom_elements",
        "atom_positions",
        "b_factors",
        "occupancies",
        "positions",
        "reference_local_positions",
    ):
        np.testing.assert_array_equal(normalized[field], expected[field])
    assert (normalized["atom_names"] != "").sum(axis=1).tolist() == [1] * 7 + [8, 1]
    assert not {"atom_mask", "interface_mask", "split_groups"}.intersection(normalized)


@pytest.mark.parametrize(
    "field,index,value,message",
    [
        ("atom_names", (0, 1), "", "contiguous present prefix"),
        ("atom_elements", (0, 0), "", "present atoms require names and elements"),
        ("atom_elements", (1, 2), "C", "padded atom fields"),
        ("atom_positions", (1, 2, 0), 1.0, "padded atom fields"),
        ("b_factors", (1, 2), 1.0, "padded atom fields"),
        ("occupancies", (1, 2), 1.0, "padded atom fields"),
        ("atom_positions", (1, 2, 0), np.nan, "finite"),
        ("occupancies", (0, 0), -1.0, "occupancies must be non-negative"),
    ],
)
def test_component_presence_from_names_retains_padding_and_numeric_validation(
    tmp_path, field, index, value, message
):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = _build(structure_path)
    targets["component"][field][index] = value
    with pytest.raises(ValueError, match=message):
        validate_component_extension_targets(targets)


def test_component_row_requires_a_present_named_atom(tmp_path):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = _build(structure_path)
    targets["component"]["atom_names"][0] = ""
    with pytest.raises(
        ValueError, match="each component node must contain at least one atom"
    ):
        validate_component_extension_targets(targets)


@pytest.mark.parametrize(
    "names", [np.zeros((5, 3)), np.full((5,), "CA"), np.full((4, 3), "CA")]
)
def test_component_names_define_only_valid_padded_atom_dimensions(tmp_path, names):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = _build(structure_path)
    targets["component"]["atom_names"] = names
    with pytest.raises(ValueError, match="atom_names must be a padded string"):
        validate_component_extension_targets(targets)


def test_component_proposal_subset_order_and_custom_validity_are_preserved(
    standard_artifact_mapping,
):
    targets = standard_artifact_mapping["extension_targets"]
    expected_positions = targets["component"]["positions"][[2, 0]].copy()
    targets["component"]["proposal_targets"] = {
        "node_indices": np.asarray([2, 0], dtype=np.int32),
        # Intentionally differs from geometric grid validity for these anchors.
        "valid_mask": np.asarray([1, 0], dtype=np.float64),
    }
    normalized = validate_component_extension_targets(targets)["component"]
    proposals = normalized["proposal_targets"]
    assert set(proposals) == {"node_indices", "valid_mask"}
    assert proposals["node_indices"].dtype == np.int64
    assert proposals["valid_mask"].dtype == np.bool_
    assert proposals["node_indices"].tolist() == [2, 0]
    assert proposals["valid_mask"].tolist() == [True, False]
    np.testing.assert_array_equal(
        normalized["positions"][proposals["node_indices"]], expected_positions
    )


def test_component_proposals_allow_an_empty_subset_of_nonempty_components(
    standard_artifact_mapping,
):
    targets = standard_artifact_mapping["extension_targets"]
    targets["component"]["proposal_targets"] = {
        "node_indices": np.empty(0, dtype=np.int64),
        "valid_mask": np.empty(0, dtype=bool),
    }
    normalized = validate_component_extension_targets(targets)["component"]
    assert len(normalized["positions"]) == standard_expected()["component_nodes"]
    assert normalized["proposal_targets"]["node_indices"].shape == (0,)
    assert normalized["proposal_targets"]["valid_mask"].shape == (0,)


@pytest.mark.parametrize("indices", [[0, 0], [-1, 0], [0, 5]])
def test_component_proposal_indices_remain_unique_and_in_range(tmp_path, indices):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = _build(structure_path)
    targets["component"]["proposal_targets"] = {
        "node_indices": np.asarray(indices),
        "valid_mask": np.ones(2, dtype=bool),
    }
    with pytest.raises(
        ValueError, match="proposal node_indices must be unique and in range"
    ):
        validate_component_extension_targets(targets)


@pytest.mark.parametrize(
    "validity",
    [
        np.asarray([True]),
        np.asarray([True, False, True]),
        np.asarray([0.5, 1.0]),
        np.asarray([np.nan, 1.0]),
    ],
)
def test_component_proposal_validity_still_requires_matching_binary_mask(
    tmp_path, validity
):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = _build(structure_path)
    targets["component"]["proposal_targets"] = {
        "node_indices": np.asarray([2, 0]),
        "valid_mask": validity,
    }
    with pytest.raises(ValueError, match="proposal valid_mask"):
        validate_component_extension_targets(targets)


@pytest.mark.parametrize(
    "interface_id", ["OTHER|0123456789abcdef", "LIG|0123456789ABCDEF", "LIG|123"]
)
def test_component_nonempty_interface_ids_keep_component_prefix_and_digest_rules(
    tmp_path, interface_id
):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = _build(structure_path)
    component = targets["component"]
    component["interface_group_id"] = component["interface_group_id"].astype(object)
    component["interface_group_id"][1] = interface_id
    with pytest.raises(ValueError, match="16 lowercase hex"):
        validate_component_extension_targets(targets)


def test_component_interface_presence_is_fully_defined_by_group_id(tmp_path):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = _build(structure_path)
    component = targets["component"]
    component["interface_group_id"] = component["interface_group_id"].astype(object)
    component["interface_group_id"][1] = "LIG|0123456789abcdef"
    normalized = validate_component_extension_targets(targets)["component"]
    assert (normalized["interface_group_id"] != "").tolist() == [
        False,
        True,
        False,
        False,
        False,
    ]
    assert "interface_mask" not in normalized


@pytest.mark.parametrize("schema_version", [0, 2, True, 1.0, "1", None])
def test_deduplicated_component_payload_rejects_unsupported_schema_versions(
    tmp_path, schema_version
):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    targets = _build(structure_path)
    assert targets["schema_version"] == 1
    targets["schema_version"] = schema_version
    with pytest.raises(
        ValueError, match="unsupported component extension schema_version"
    ):
        validate_component_extension_targets(targets)


@pytest.mark.parametrize(
    "field",
    [
        "component_id",
        "instance_id",
        "chain_id",
        "insertion_code",
        "source_chem_comp_type",
        "atom_names",
        "atom_elements",
        "interface_group_id",
    ],
)
@pytest.mark.parametrize("invalid_value", [None, 7, 1.5, True])
@pytest.mark.parametrize("container", ["object_array", "list"])
def test_component_string_fields_reject_nonstrings_without_coercion(
    standard_artifact_mapping, field, invalid_value, container
):
    targets = standard_artifact_mapping["extension_targets"]
    values = targets["component"][field].astype(object)
    values.flat[0] = invalid_value
    targets["component"][field] = values.tolist() if container == "list" else values

    with pytest.raises(ValueError, match=f"{field} must contain only strings"):
        validate_component_extension_targets(targets)


@pytest.mark.parametrize("field", ["atom_names", "atom_elements", "instance_id"])
def test_component_string_fields_reject_invalid_utf8(standard_artifact_mapping, field):
    targets = standard_artifact_mapping["extension_targets"]
    values = targets["component"][field].astype(object)
    values.flat[0] = b"\xff"
    targets["component"][field] = values

    with pytest.raises(ValueError, match=f"{field} must contain valid UTF-8 strings"):
        validate_component_extension_targets(targets)


def test_component_strings_decode_utf8_and_preserve_normalized_width(
    standard_artifact_mapping,
):
    targets = standard_artifact_mapping["extension_targets"]
    component = targets["component"]
    expected = copy.deepcopy(component)
    for field in (
        "component_id",
        "instance_id",
        "chain_id",
        "insertion_code",
        "source_chem_comp_type",
        "atom_names",
        "atom_elements",
        "interface_group_id",
    ):
        component[field] = np.asarray(
            [value.encode("utf-8") for value in component[field].flat], dtype=object
        ).reshape(component[field].shape)
    component["component_id"][1] = " ß ".encode()
    component["atom_names"][1, 0] = " straße ".encode()
    component["instance_id"][1] = "Identität".encode()
    # This deliberate string-normalization probe changes a component identity,
    # so remove its old interface ID rather than asserting a false prefix.
    component["interface_group_id"][1] = b""
    normalized = validate_component_extension_targets(targets)["component"]
    assert normalized["component_id"][1] == "SS"
    assert normalized["atom_names"][1, 0] == "STRASSE"
    assert normalized["instance_id"][1] == "Identität"
    for field in ("positions", "atom_positions", "reference_local_positions"):
        np.testing.assert_array_equal(normalized[field], expected[field])
    assert normalized["atom_names"].dtype.kind == "U"


def test_component_interface_ids_strip_before_presence_and_format_checks(
    standard_artifact_mapping,
):
    targets = standard_artifact_mapping["extension_targets"]
    component = targets["component"]
    node_count = len(component["positions"])
    component["interface_group_id"] = np.full(node_count, "\t ", dtype=object)
    component["interface_group_id"][1] = " ZN|0123456789abcdef\n"
    normalized = validate_component_extension_targets(targets)["component"]
    expected_ids = [""] * node_count
    expected_ids[1] = "ZN|0123456789abcdef"
    assert normalized["interface_group_id"].tolist() == expected_ids
    expected_presence = np.zeros(node_count, dtype=bool)
    expected_presence[1] = True
    np.testing.assert_array_equal(
        normalized["interface_group_id"] != "", expected_presence
    )


def _polymer_context():
    return {
        "polymer_atom_positions": np.asarray([[3.0, 0.0, 0.0]], dtype=np.float64),
        "polymer_atom_node_indices": np.asarray([0]),
        "polymer_chain_indices": np.asarray([2]),
        "polymer_residue_indices": np.asarray([10]),
        "polymer_aatype": np.asarray([5]),
    }


@pytest.mark.parametrize("value", [1e100, np.inf, np.nan])
def test_component_context_rejects_positions_not_finite_in_float32(tmp_path, value):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    context = _polymer_context()
    context["polymer_atom_positions"][0, 0] = value
    with pytest.raises(ValueError, match="polymer_atom_positions.*finite.*float32"):
        _build(structure_path, **context)


@pytest.mark.parametrize(
    "field",
    [
        "polymer_atom_node_indices",
        "polymer_chain_indices",
        "polymer_residue_indices",
        "polymer_aatype",
    ],
)
def test_component_context_rejects_topology_values_that_cannot_fit_int64(
    tmp_path, field
):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    context = _polymer_context()
    context[field] = np.asarray([2**63], dtype=np.uint64)
    with pytest.raises(ValueError, match="representable as int64"):
        _build(structure_path, **context)


def test_component_context_keeps_representable_uint64_topology_hash(tmp_path):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    context = _polymer_context()
    maximum = np.iinfo(np.int64).max
    for field in ("polymer_chain_indices", "polymer_residue_indices", "polymer_aatype"):
        context[field] = np.asarray([maximum], dtype=np.uint64)
    component = _build(structure_path, **context)["component"]
    expected_digest = hashlib.sha256(
        repr([(int(maximum), int(maximum), int(maximum))]).encode()
    ).hexdigest()[:16]
    assert component["interface_group_id"][1] == f"LIG|{expected_digest}"


@pytest.mark.parametrize("voxel_size", [True, "1", [1.0], 1 + 0j])
def test_component_builder_requires_canonical_numeric_voxel_size(tmp_path, voxel_size):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    with pytest.raises(ValueError, match="voxel_size.*positive finite numeric scalar"):
        build_component_extension_targets(
            structure_path,
            global_origin=np.zeros(3),
            voxel_size=voxel_size,
            grid_shape=(20, 20, 20),
        )


def test_component_builder_rejects_origin_that_overflows_float32(tmp_path):
    structure_path = tmp_path / "mixed.cif"
    _write_mixed_cif(structure_path)
    with pytest.raises(ValueError, match="global_origin.*finite numeric 3-vector"):
        build_component_extension_targets(
            structure_path,
            global_origin=np.full(3, 1e100),
            voxel_size=1.0,
            grid_shape=(20, 20, 20),
        )


def test_component_numeric_validation_reuses_arrays_and_scans_each_once(
    standard_artifact_mapping, monkeypatch
):
    targets = standard_artifact_mapping["extension_targets"]
    checked = []
    finite = schema.array_all_finite

    def record_finite(array):
        checked.append(array)
        return finite(array)

    monkeypatch.setattr(schema, "array_all_finite", record_finite)
    normalized = validate_component_extension_targets(targets)["component"]
    fields = (
        "positions",
        "reference_local_positions",
        "atom_positions",
        "b_factors",
        "occupancies",
    )
    assert len(checked) == len(fields)
    assert {id(array) for array in checked} == {
        id(normalized[field]) for field in fields
    }
    for field in fields:
        assert normalized[field] is targets["component"][field]


@pytest.mark.parametrize(
    "positions,origin,voxel_size,grid_shape,expected",
    [
        (
            [[0, 0, 0], [1e-30, 0, 0], [-1e-30, 0, 0]],
            [0, 0, 0],
            1e-100,
            (2, 2, 2),
            [True, False, False],
        ),
        (
            [[0, 0, 0], [1, 0, 0], [-1, 0, 0]],
            [0, 0, 0],
            1e100,
            (2, 2, 2),
            [True, True, False],
        ),
        (
            [[0, 0, 0], [1e-30, 0, 0], [-1e-30, 0, 0]],
            [0, 0, 0],
            1e30,
            (2, 2, 2),
            [True, True, False],
        ),
        (
            [
                [-1, 2, 3],
                [np.nextafter(np.float32(1.5), np.float32(-np.inf)), 2, 3],
                [1.5, 2, 3],
                [np.nextafter(np.float32(1.5), np.float32(np.inf)), 2, 3],
                [-1, 4, 3],
                [-1, 2, 4.5],
            ],
            [-1, 2, 3],
            0.5,
            (3, 4, 5),
            [True, True, False, False, False, False],
        ),
        (
            [[3e38, 0, 0], [-3e38, 0, 0]],
            [-3e38, 0, 0],
            1e38,
            (2, 2, 10),
            [True, True],
        ),
        (
            [[3e38, 0, 0], [-3e38, 0, 0]],
            [-3e38, 0, 0],
            1e38,
            (2, 2, 5),
            [False, True],
        ),
        (
            [[0, 0, 0], [3e38, 0, 0], [-3e38, 0, 0]],
            [0, 0, 0],
            1e308,
            (2, 2, 2),
            [True, True, False],
        ),
    ],
    ids=[
        "tiny-spacing",
        "huge-spacing",
        "negative-coordinate-division-underflow",
        "noncubic-boundaries-and-nextafter",
        "subtraction-exceeds-float32-inside",
        "subtraction-exceeds-float32-outside",
        "extent-exceeds-float64",
    ],
)
def test_component_grid_bounds_preserve_world_space_boundaries(
    positions, origin, voxel_size, grid_shape, expected
):
    positions = np.asarray(positions, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    positions_before = positions.copy()
    origin_before = origin.copy()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        valid_mask = components._grid_valid_mask(
            positions, origin, voxel_size, grid_shape
        )
    assert valid_mask.dtype == np.bool_
    np.testing.assert_array_equal(valid_mask, expected)
    np.testing.assert_array_equal(positions, positions_before)
    np.testing.assert_array_equal(origin, origin_before)
