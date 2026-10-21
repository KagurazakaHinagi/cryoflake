import copy
import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from biotite.structure.io import pdbx

from cryoflake.constants import NodeKind, PolymerKind
from cryoflake.storage import store
from cryoflake.storage.store import write_preprocessed_artifact
from cryoflake.targets.components import build_component_extension_targets
from cryoflake.visualization import reconstruction
from cryoflake.visualization.reconstruction import reconstruct_artifact


def _artifact_mapping():
    origin = np.asarray([17.5, -8.0, 100.0], dtype=np.float32)
    coordinates = {
        name: origin + np.asarray([[index / 8, 0.5, 1], [2.5 + index / 8, 2, 1]])
        for index, name in enumerate(("CA", "C", "N", "O"))
    }
    coordinates["O"][1] = np.nan
    coordinates["P"] = origin + np.asarray([[4.5, 1, 0.5]])
    return {
        "density_grid": np.arange(30, dtype=np.float32).reshape(2, 3, 5),
        "global_origin": origin,
        "voxel_size": 1.75,
        "backbone_coordinates": coordinates,
        "residue_info": {
            "aa_type": ["ALA", "ARG"],
            "nuc_type": ["DG"],
            # This optional field follows original traversal, not full-atom row order.
            "chain": ["original_nucleotide", "original_protein", "original_protein"],
        },
        "full_atom_targets": {
            "chain_index": np.asarray([2, 2, 7]),
            "residue_index": np.asarray([-1, 80, 12]),
        },
        "sequence": np.zeros((8, 2), dtype=np.float32),
        "sequence_mask": np.ones(8, dtype=bool),
        "structure_path": "/missing/original.cif",
        "map_path": "/missing/original.mrc",
    }


def _read_structure(path, *, author_fields=True):
    return pdbx.get_structure(
        pdbx.CIFFile.read(path),
        model=1,
        use_author_fields=author_fields,
        extra_fields=["b_factor", "occupancy"],
    )


def _write_artifact(tmp_path, artifact_mapping):
    artifact_reference = tmp_path / "artifact"
    write_preprocessed_artifact(artifact_reference, artifact_mapping)
    return artifact_reference


def _component_targets(tmp_path):
    component_source_path = tmp_path / "components.cif"
    component_source_path.write_text("""data_components
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.type_symbol
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
HETATM C1 LIG label1 POLYMER_2 52 A C 0.4 13.5 20.5 -6.5 100.5
HETATM O1 LIG label1 POLYMER_2 52 A O 1.2 14.5 21.5 -6.0 101.0
HETATM ZN ZN label2 'chain with spaces' -3 ? ZN 1.0 7.5 19.5 -5.5 101.5
HETATM C1 LIG label3 POLYMER_2 52 A C 0.6 15.5 22.5 -6.5 100.5
""")
    targets = build_component_extension_targets(
        component_source_path,
        global_origin=np.asarray([17.5, -8, 100]),
        voxel_size=1.75,
        grid_shape=(2, 3, 5),
    )
    component_source_path.unlink()
    return targets


def test_backbone_reconstruction_preserves_world_geometry_and_observed_sequences(
    tmp_path,
):
    artifact_mapping = _artifact_mapping()
    artifact_reference = _write_artifact(tmp_path, artifact_mapping)
    # Embedded sequence arrays are deliberately unrelated, and are not needed.
    (artifact_reference / "sequence.npz").unlink()

    bundle = reconstruct_artifact(
        artifact_reference / "metadata.json", tmp_path / "view"
    )

    atoms = _read_structure(bundle.structure_path)
    assert atoms.atom_name.tolist() == ["CA", "C", "N", "O", "CA", "C", "N", "P"]
    assert atoms.res_name.tolist() == ["ALA"] * 4 + ["ARG"] * 3 + ["DG"]
    assert atoms.res_id.tolist() == [-1] * 4 + [80] * 3 + [12]
    assert atoms.chain_id.tolist() == ["POLYMER_2"] * 7 + ["POLYMER_7"]
    expected_coordinates = np.asarray(
        [
            artifact_mapping["backbone_coordinates"][name][row]
            for row in range(2)
            for name in ("CA", "C", "N", "O")
            if np.isfinite(artifact_mapping["backbone_coordinates"][name][row]).all()
        ]
        + [artifact_mapping["backbone_coordinates"]["P"][0]]
    )
    np.testing.assert_array_equal(atoms.coord, expected_coordinates)
    assert np.isfinite(atoms.coord).all()
    with mrcfile.open(bundle.density_path) as density:
        np.testing.assert_array_equal(density.data, artifact_mapping["density_grid"])
        np.testing.assert_array_equal(
            density.header.origin.tolist(), artifact_mapping["global_origin"]
        )
        assert density.voxel_size.tolist() == (1.75, 1.75, 1.75)
        assert (
            int(density.header.mapc),
            int(density.header.mapr),
            int(density.header.maps),
        ) == (1, 2, 3)
    summary = json.loads(bundle.summary_path.read_text())
    assert summary["density"] == {
        "shape_zyx": [2, 3, 5],
        "origin_xyz": [17.5, -8, 100],
        "voxel_size_angstrom": 1.75,
    }
    assert summary["coordinate_units"] == "angstrom"
    assert summary["atom_count"] == 8
    assert summary["residue_count"] == 3
    assert summary["chain_count"] == summary["sequence_count"] == 2
    assert summary["reconstruction_mode"] == "backbone"
    assert "Backbone reconstruction" in summary["limitations"][0]
    assert summary["chains"] == [
        {
            "chain_id": "POLYMER_2",
            "polymer_kind": "protein",
            "sequence": "AR",
            "residue_count": 2,
            "residue_indices": [-1, 80],
        },
        {
            "chain_id": "POLYMER_7",
            "polymer_kind": "nucleic_acid",
            "sequence": "G",
            "residue_count": 1,
            "residue_indices": [12],
        },
    ]
    assert (
        bundle.sequence_path.read_text()
        == ">POLYMER_2 observed_protein\nAR\n>POLYMER_7 observed_nucleic_acid\nG\n"
    )
    block = pdbx.CIFFile.read(bundle.structure_path)["cryoflake"]
    assert block["entity_poly"]["type"].as_array(str).tolist() == [
        "polypeptide(L)",
        "polydeoxyribonucleotide",
    ]
    assert block["entity_poly_seq"]["mon_id"].as_array(str).tolist() == [
        "ALA",
        "ARG",
        "DG",
    ]
    assert block["entity_poly_seq"]["num"].as_array(int).tolist() == [1, 2, 1]


def test_interleaved_chains_preserve_sequence_order_and_mixed_polymer_chemistry(
    tmp_path,
):
    artifact_mapping = _artifact_mapping()
    artifact_mapping["backbone_coordinates"] = {
        name: np.arange(12, dtype=np.float32).reshape(4, 3) + atom_index / 8
        for atom_index, name in enumerate(("CA", "C", "N", "O", "P"))
    }
    artifact_mapping["residue_info"] = {
        "aa_type": ["ALA", "ARG", "ASN", "ASP"],
        "nuc_type": ["DA", "U", "C", "DG"],
    }
    artifact_mapping["full_atom_targets"] = {
        "chain_index": np.asarray([4, 2, 4, 2, 7, 8, 7, 9]),
        "residue_index": np.asarray([-10, 90, 5, 91, 8, -3, 11, 2]),
    }
    bundle = reconstruct_artifact(
        _write_artifact(tmp_path, artifact_mapping), tmp_path / "view"
    )
    summary = json.loads(bundle.summary_path.read_text())
    assert summary["chains"] == [
        {
            "chain_id": "POLYMER_4",
            "polymer_kind": "protein",
            "sequence": "AN",
            "residue_count": 2,
            "residue_indices": [-10, 5],
        },
        {
            "chain_id": "POLYMER_2",
            "polymer_kind": "protein",
            "sequence": "RD",
            "residue_count": 2,
            "residue_indices": [90, 91],
        },
        {
            "chain_id": "POLYMER_7",
            "polymer_kind": "nucleic_acid",
            "sequence": "AC",
            "residue_count": 2,
            "residue_indices": [8, 11],
        },
        {
            "chain_id": "POLYMER_8",
            "polymer_kind": "nucleic_acid",
            "sequence": "U",
            "residue_count": 1,
            "residue_indices": [-3],
        },
        {
            "chain_id": "POLYMER_9",
            "polymer_kind": "nucleic_acid",
            "sequence": "G",
            "residue_count": 1,
            "residue_indices": [2],
        },
    ]
    assert bundle.sequence_path.read_text() == (
        ">POLYMER_4 observed_protein\nAN\n"
        ">POLYMER_2 observed_protein\nRD\n"
        ">POLYMER_7 observed_nucleic_acid\nAC\n"
        ">POLYMER_8 observed_nucleic_acid\nU\n"
        ">POLYMER_9 observed_nucleic_acid\nG\n"
    )
    block = pdbx.CIFFile.read(bundle.structure_path)["cryoflake"]
    assert block["struct_asym"]["id"].as_array(str).tolist() == [
        "POLYMER_4",
        "POLYMER_2",
        "POLYMER_7",
        "POLYMER_8",
        "POLYMER_9",
    ]
    assert block["entity_poly"]["type"].as_array(str).tolist() == [
        "polypeptide(L)",
        "polypeptide(L)",
        "polydeoxyribonucleotide/polyribonucleotide hybrid",
        "polyribonucleotide",
        "polydeoxyribonucleotide",
    ]
    assert block["entity_poly_seq"]["num"].as_array(int).tolist() == [
        1,
        1,
        2,
        2,
        1,
        1,
        2,
        1,
    ]
    assert block["entity_poly_seq"]["entity_id"].as_array(str).tolist() == [
        "1",
        "2",
        "1",
        "2",
        "3",
        "4",
        "3",
        "5",
    ]
    atoms = _read_structure(bundle.structure_path)
    assert atoms.res_id.tolist() == [-10] * 4 + [90] * 4 + [5] * 4 + [91] * 4 + [
        8,
        -3,
        11,
        2,
    ]


def test_components_preserve_masks_topology_elements_occupancy_and_distinct_instances(
    tmp_path,
):
    artifact_mapping = _artifact_mapping()
    artifact_mapping["extension_targets"] = _component_targets(tmp_path)
    artifact_reference = _write_artifact(tmp_path, artifact_mapping)
    bundle = reconstruct_artifact(artifact_reference, tmp_path / "view")
    atoms = _read_structure(bundle.structure_path)
    component = artifact_mapping["extension_targets"]["component"]
    assert atoms.chain_id[:7].tolist() == ["POLYMER_2_2"] * 7
    assert atoms.chain_id[8:].tolist() == [
        "POLYMER_2",
        "POLYMER_2",
        "chain with spaces",
        "POLYMER_2",
    ]
    assert atoms.res_id[8:].tolist() == [52, 52, -3, 52]
    assert atoms.ins_code[8:].tolist() == ["A", "A", "", "A"]
    assert atoms.atom_name[8:].tolist() == ["C1", "O1", "ZN", "C1"]
    assert atoms.element[8:].tolist() == ["C", "O", "ZN", "C"]
    assert atoms.hetero[8:].all()
    np.testing.assert_array_equal(
        atoms.coord[8:], component["atom_positions"][component["atom_mask"]]
    )
    np.testing.assert_allclose(atoms.occupancy[8:], [0.4, 1.2, 1.0, 0.6])
    np.testing.assert_allclose(atoms.b_factor[8:], [13.5, 14.5, 7.5, 15.5])
    # Distinct stored instances that share author topology retain unique labels.
    label_atoms = _read_structure(bundle.structure_path, author_fields=False)
    assert label_atoms.res_id[8:].tolist() == [1, 1, 1, 2]
    summary = json.loads(bundle.summary_path.read_text())
    assert summary["atom_count"] == 12
    assert summary["residue_count"] == 6
    assert summary["component_count"] == 3
    assert summary["chain_count"] == 4
    assert [chain["sequence"] for chain in summary["chains"]] == ["AR", "G"]


def test_noncanonical_polymer_components_retain_polymer_chemistry(tmp_path):
    artifact_mapping = _artifact_mapping()
    artifact_mapping["extension_targets"] = _component_targets(tmp_path)
    component = artifact_mapping["extension_targets"]["component"]
    component["node_kind"][0] = int(NodeKind.POLYMER)
    component["polymer_kind"][0] = int(PolymerKind.OTHER)
    bundle = reconstruct_artifact(
        _write_artifact(tmp_path, artifact_mapping), tmp_path / "view"
    )
    atoms = _read_structure(bundle.structure_path)
    assert not atoms.hetero[8:10].any()
    assert atoms.hetero[10:].all()
    block = pdbx.CIFFile.read(bundle.structure_path)["cryoflake"]
    assert block["entity_poly"]["type"].as_array(str).tolist() == [
        "polypeptide(L)",
        "polydeoxyribonucleotide",
        "other",
    ]
    assert block["entity_poly_seq"]["mon_id"].as_array(str).tolist() == [
        "ALA",
        "ARG",
        "DG",
        "LIG",
    ]


def test_missing_atoms_never_become_phantom_coordinates_or_sequence_rows(tmp_path):
    artifact_mapping = _artifact_mapping()
    artifact_mapping.pop("full_atom_targets")
    for name in ("CA", "C", "N", "O"):
        artifact_mapping["backbone_coordinates"][name][0] = np.nan
    bundle = reconstruct_artifact(
        _write_artifact(tmp_path, artifact_mapping), tmp_path / "view"
    )
    atoms = _read_structure(bundle.structure_path)
    assert atoms.atom_name.tolist() == ["CA", "C", "N", "P"]
    assert atoms.res_id.tolist() == [2, 2, 2, 1]
    assert atoms.chain_id.tolist() == ["PROTEIN"] * 3 + ["NUCLEIC_ACID"]
    assert (
        bundle.sequence_path.read_text()
        == ">PROTEIN observed_protein\nR\n>NUCLEIC_ACID observed_nucleic_acid\nG\n"
    )
    assert (
        "Skipped 1 polymer residues"
        in json.loads(bundle.summary_path.read_text())["limitations"][-1]
    )


def _with_full_atom_arrays(artifact_mapping):
    artifact_mapping = copy.deepcopy(artifact_mapping)
    targets = artifact_mapping["full_atom_targets"]
    targets["atom_positions"] = (
        np.arange(3 * 65 * 3, dtype=np.float32).reshape(3, 65, 3) / 8
    )
    targets["atom_mask"] = np.zeros((3, 65), dtype=bool)
    targets["atom_mask"][0, [0, 4, 64]] = True
    targets["atom_mask"][1, [0, 4]] = True
    targets["atom_mask"][2, 64] = True
    targets["b_factors"] = np.arange(3 * 65, dtype=np.float32).reshape(3, 65) / 4
    return artifact_mapping


@pytest.mark.parametrize("per_residue_names", [False, True])
def test_full_atom_export_uses_explicit_names_and_exact_masked_values(
    tmp_path, per_residue_names
):
    artifact_mapping = _with_full_atom_arrays(_artifact_mapping())
    names = [""] * 65
    names[0], names[4], names[64] = "N", "CA", "P"
    if per_residue_names:
        names = [names.copy() for _ in range(3)]
        names[0][64] = "SG"
        names[1][4] = "C"
    atom_names_path = tmp_path / "names.json"
    atom_names_path.write_text(json.dumps(names))
    bundle = reconstruct_artifact(
        _write_artifact(tmp_path, artifact_mapping),
        tmp_path / "view",
        atom_names_path=atom_names_path,
    )
    atoms = _read_structure(bundle.structure_path)
    assert atoms.atom_name.tolist() == (
        ["N", "CA", "SG", "N", "C", "P"]
        if per_residue_names
        else ["N", "CA", "P", "N", "CA", "P"]
    )
    targets = artifact_mapping["full_atom_targets"]
    np.testing.assert_array_equal(
        atoms.coord, targets["atom_positions"][targets["atom_mask"]]
    )
    np.testing.assert_array_equal(
        atoms.b_factor, targets["b_factors"][targets["atom_mask"]]
    )
    assert len(atoms) == 6
    summary = json.loads(bundle.summary_path.read_text())
    assert summary["atom_count"] == 6
    assert summary["reconstruction_mode"] == "full_atom"
    assert not any("Backbone reconstruction" in text for text in summary["limitations"])


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["N"] * 65, "duplicate present atom names"),
        ([""] * 65, "missing or invalid present atom name"),
        (["bad name"] * 65, "missing or invalid present atom name"),
        (["bad\nname"] * 65, "missing or invalid present atom name"),
        (["N"] * 64, "65 slot names"),
        ({"N": 0}, "65 slot names"),
        ([None] * 65, "65 slot names"),
    ],
)
def test_invalid_full_atom_names_are_rejected_before_output_io(
    tmp_path, names, message
):
    artifact_mapping = _with_full_atom_arrays(_artifact_mapping())
    atom_names_path = tmp_path / "names.json"
    atom_names_path.write_text(json.dumps(names))
    with pytest.raises(ValueError, match=message):
        reconstruct_artifact(
            _write_artifact(tmp_path, artifact_mapping),
            tmp_path / "view",
            atom_names_path=atom_names_path,
        )
    assert not (tmp_path / "view").exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda mapping: mapping.pop("global_origin"), "global_origin"),
        (lambda mapping: mapping.__setitem__("voxel_size", 0), "voxel_size"),
        (
            lambda mapping: mapping["backbone_coordinates"]["N"].__setitem__(
                (0, 0), np.inf
            ),
            "infinite coordinates",
        ),
        (
            lambda mapping: mapping["residue_info"].__setitem__("aa_type", ["ALA"]),
            "length must match",
        ),
        (
            lambda mapping: mapping["full_atom_targets"].__setitem__(
                "chain_index", [-1, 2, 7]
            ),
            "valid integer topology",
        ),
        (
            lambda mapping: mapping["full_atom_targets"].pop("residue_index"),
            "requires chain_index and residue_index",
        ),
    ],
)
def test_incomplete_or_malformed_visualization_values_are_rejected(
    tmp_path, mutate, message
):
    artifact_mapping = _artifact_mapping()
    mutate(artifact_mapping)
    with pytest.raises(ValueError, match=message):
        reconstruct_artifact(
            _write_artifact(tmp_path, artifact_mapping), tmp_path / "view"
        )
    assert not (tmp_path / "view").exists()


def test_s3_selected_store_reads_skip_embeddings_and_preserve_endpoint(
    tmp_path, monkeypatch
):
    artifact_reference = _write_artifact(tmp_path, _artifact_mapping())
    remote_members = {
        f"s3://bucket/atomic/{path.name}": path.read_bytes()
        for path in artifact_reference.iterdir()
        if path.is_file()
    }
    requests = []

    def read_member(reference, limit, *, s3_endpoint_url):
        requests.append((str(reference), s3_endpoint_url))
        assert len(remote_members[str(reference)]) <= limit
        return remote_members[str(reference)]

    def copy_member(reference, destination, *, s3_endpoint_url):
        requests.append((str(reference), s3_endpoint_url))
        Path(destination).write_bytes(remote_members[str(reference)])

    monkeypatch.setattr(store, "read_file_bytes_limited", read_member)
    monkeypatch.setattr(store, "copy_file_to_path", copy_member)
    bundle = reconstruct_artifact(
        "s3://bucket/atomic/metadata.json",
        tmp_path / "view",
        s3_endpoint_url="https://storage.invalid",
    )
    assert requests == [
        ("s3://bucket/atomic/metadata.json", "https://storage.invalid"),
        ("s3://bucket/atomic/core.npz", "https://storage.invalid"),
    ]
    assert _read_structure(bundle.structure_path).array_length() == 8
    assert (
        json.loads(bundle.summary_path.read_text())["artifact_reference"]
        == "s3://bucket/atomic/metadata.json"
    )


def test_loader_selects_only_reconstruction_roots_and_excludes_embedding_arrays(
    tmp_path, monkeypatch
):
    calls = []

    def load(artifact_reference, **kwargs):
        calls.append((artifact_reference, kwargs))
        return _artifact_mapping()

    monkeypatch.setattr(reconstruction, "load_preprocessed_artifact", load)
    reconstruct_artifact("s3://bucket/atomic", tmp_path / "view")
    reference, options = calls.pop()
    assert reference == "s3://bucket/atomic"
    assert options["include_sequence_embeddings"] is False
    assert set(options["root_keys"]) == {
        "density_grid",
        "global_origin",
        "voxel_size",
        "backbone_coordinates",
        "residue_info",
        "full_atom_targets",
        "extension_targets",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("atom_mask", 0.5, "only boolean values"),
        ("atom_positions", np.nan, "finite numeric"),
        ("b_factors", np.inf, "finite numeric"),
    ],
)
def test_full_atom_export_rejects_invalid_masks_and_nonfinite_values(
    tmp_path, field, value, message
):
    artifact_mapping = _with_full_atom_arrays(_artifact_mapping())
    artifact_mapping["full_atom_targets"][field] = artifact_mapping[
        "full_atom_targets"
    ][field].astype(float)
    artifact_mapping["full_atom_targets"][field].flat[0] = value
    atom_names_path = tmp_path / "names.json"
    names = [""] * 65
    names[0], names[4], names[64] = "N", "CA", "P"
    atom_names_path.write_text(json.dumps(names))
    with pytest.raises(ValueError, match=message):
        reconstruct_artifact(
            _write_artifact(tmp_path, artifact_mapping),
            tmp_path / "view",
            atom_names_path=atom_names_path,
        )
    assert not (tmp_path / "view").exists()


def test_existing_export_is_not_overwritten(tmp_path):
    artifact_reference = _write_artifact(tmp_path, _artifact_mapping())
    bundle = reconstruct_artifact(artifact_reference, tmp_path / "view")
    bundle.structure_path.write_text("User-edited structure\n")
    with pytest.raises(FileExistsError, match="Visualization output already exists"):
        reconstruct_artifact(artifact_reference, tmp_path / "view")
    assert bundle.structure_path.read_text() == "User-edited structure\n"


def test_all_missing_atoms_rejected_without_exporting_an_empty_structure(tmp_path):
    artifact_mapping = _artifact_mapping()
    for coordinates in artifact_mapping["backbone_coordinates"].values():
        coordinates[:] = np.nan
    with pytest.raises(ValueError, match="no present finite named atoms"):
        reconstruct_artifact(
            _write_artifact(tmp_path, artifact_mapping), tmp_path / "view"
        )
    assert not (tmp_path / "view").exists()


def test_unsupported_component_schema_is_rejected_before_output(tmp_path):
    artifact_mapping = _artifact_mapping()
    artifact_mapping["extension_targets"] = _component_targets(tmp_path)
    artifact_mapping["extension_targets"]["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        reconstruct_artifact(
            _write_artifact(tmp_path, artifact_mapping), tmp_path / "view"
        )
    assert not (tmp_path / "view").exists()
