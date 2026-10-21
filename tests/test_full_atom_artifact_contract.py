"""Full-atom names survive storage and define canonical artifact reuse."""

from __future__ import annotations

import io
import json
from copy import deepcopy

import numpy as np
import pytest

from cryoflake.common import json_sha256
from cryoflake.constants import (
    FULL_ATOM_COUNT,
    FULL_ATOM_TARGET_SCHEMA_VERSION,
    RIGID_GROUP_COUNT,
    TORSION_COUNT,
)
from cryoflake.preprocess.artifact_identity import preprocess_checkpoint_contract
from cryoflake.preprocess.checkpoint import (
    PREPROCESS_CHECKPOINT_FORMAT_VERSION,
    CheckpointContractMismatchError,
    PreprocessCheckpoint,
)
from cryoflake.preprocess.reuse import _preprocessed_artifact_is_reusable
from cryoflake.storage.artifacts import validate_preprocessed_artifact
from cryoflake.storage.store import (
    load_preprocessed_artifact,
    write_preprocessed_artifact,
)
from cryoflake.targets.components import build_component_extension_targets
from cryoflake.targets.full_atom import validate_full_atom_targets

_POLYMER_CIF = """data_contract
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_model_num
_atom_site.type_symbol
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM CA ALA A 1 1 A 10 1 C 1 1 1
ATOM P DA B 2 1 B 20 1 P 2 2 2
"""


def _full_atom_targets():
    targets = {
        "schema_version": FULL_ATOM_TARGET_SCHEMA_VERSION,
        "aatype": np.asarray([0, 24], dtype=np.int64),
        "chain_index": np.asarray([0, 1], dtype=np.int64),
        "residue_index": np.asarray([10, 20], dtype=np.int64),
        "atom_names": np.asarray([f"C{slot + 1}" for slot in range(FULL_ATOM_COUNT)]),
    }
    shapes = {
        "atom_positions": (FULL_ATOM_COUNT, 3),
        "atom_mask": (FULL_ATOM_COUNT,),
        "b_factors": (FULL_ATOM_COUNT,),
        "rigidgroups_gt_frames": (RIGID_GROUP_COUNT, 3, 4),
        "rigidgroups_gt_exists": (RIGID_GROUP_COUNT,),
        "rigidgroups_group_exists": (RIGID_GROUP_COUNT,),
        "rigidgroups_group_is_ambiguous": (RIGID_GROUP_COUNT,),
        "rigidgroups_alt_gt_frames": (RIGID_GROUP_COUNT, 3, 4),
        "torsion_angles_sin_cos": (TORSION_COUNT, 2),
        "alt_torsion_angles_sin_cos": (TORSION_COUNT, 2),
        "torsion_angles_mask": (TORSION_COUNT,),
    }
    for key, shape in shapes.items():
        targets[key] = np.zeros((2, *shape), dtype=np.float32)
    # Names follow the supplied coordinates rather than an assumed slot table.
    for residue, slots, names, positions in (
        (0, [0, 7, 22], ["N", "CA", "CB"], [[1, 0, 0], [1, 1, 1], [2, 1, 1]]),
        (1, [3, 12], ["P", "O5'"], [[2, 2, 2], [3, 2, 2]]),
    ):
        targets["atom_names"][slots] = names
        targets["atom_positions"][residue, slots] = positions
        targets["atom_mask"][residue, slots] = 1
        targets["b_factors"][residue, slots] = [10 + slot for slot in slots]
    return validate_full_atom_targets(targets, num_protein=1, num_nucleotide=1)


@pytest.fixture
def artifact_mapping(tmp_path):
    structure_path = tmp_path / "polymers.cif"
    structure_path.write_text(_POLYMER_CIF)
    missing = np.full((1, 3), np.nan, dtype=np.float32)
    return {
        "density_grid": np.arange(64, dtype=np.float32).reshape(4, 4, 4),
        "global_origin": np.zeros(3, dtype=np.float32),
        "voxel_size": 1.0,
        "backbone_coordinates": {
            "CA": np.asarray([[1, 1, 1]], dtype=np.float32),
            "N": np.asarray([[1, 0, 0]], dtype=np.float32),
            "C": missing.copy(),
            "O": missing.copy(),
            "P": np.asarray([[2, 2, 2]], dtype=np.float32),
        },
        "residue_info": {"aa_type": ["ALA"], "nuc_type": ["DA"]},
        "full_atom_targets": _full_atom_targets(),
        "extension_targets": build_component_extension_targets(
            structure_path,
            global_origin=np.zeros(3, dtype=np.float32),
            voxel_size=1.0,
            grid_shape=(4, 4, 4),
        ),
    }


def _assert_redundant_target_fields_are_not_serialized(artifact_reference):
    metadata = json.loads((artifact_reference / "metadata.json").read_text())
    tree = metadata["tree"]["items"]
    full_atom_fields = tree["full_atom_targets"]["items"]
    component_fields = tree["extension_targets"]["items"]["component"]["items"]
    assert "prot_mask" not in full_atom_fields
    assert not {"atom_mask", "interface_mask", "split_groups"}.intersection(
        component_fields
    )
    assert "positions" not in component_fields["proposal_targets"]["items"]
    removed_array_paths = (
        "full_atom_targets/prot_mask",
        "extension_targets/component/atom_mask",
        "extension_targets/component/interface_mask",
        "extension_targets/component/proposal_targets/positions",
        "extension_targets/component/split_groups/",
    )
    with np.load(artifact_reference / "core.npz", allow_pickle=False) as archive:
        assert not any(key.startswith(removed_array_paths) for key in archive.files)


def test_storage_preserves_full_atom_names_and_their_observation_alignment(
    tmp_path, artifact_mapping
):
    artifact_reference = tmp_path / "artifact"
    write_preprocessed_artifact(artifact_reference, artifact_mapping)
    _assert_redundant_target_fields_are_not_serialized(artifact_reference)
    expected = artifact_mapping["full_atom_targets"]
    metadata = json.loads((artifact_reference / "metadata.json").read_text())
    names_descriptor = metadata["tree"]["items"]["full_atom_targets"]["items"][
        "atom_names"
    ]
    assert names_descriptor == {
        "kind": "array",
        "store": "core",
        "key": "full_atom_targets/atom_names",
        "dtype": expected["atom_names"].dtype.str,
        "shape": [65],
    }
    assert metadata["format_version"] == 1

    with np.load(artifact_reference / "core.npz", allow_pickle=False) as archive:
        names = archive["full_atom_targets/atom_names"]
        assert names.shape == (65,)
        assert names.dtype.kind == "U"
        np.testing.assert_array_equal(names, expected["atom_names"])
        for key, values in expected.items():
            if isinstance(values, np.ndarray):
                stored = archive[f"full_atom_targets/{key}"]
                assert stored.dtype == values.dtype
                np.testing.assert_array_equal(stored, values)

    restored_mapping = load_preprocessed_artifact(artifact_reference)
    for key in ("density_grid", "global_origin"):
        np.testing.assert_array_equal(restored_mapping[key], artifact_mapping[key])
    assert restored_mapping["voxel_size"] == artifact_mapping["voxel_size"]
    assert restored_mapping["residue_info"] == artifact_mapping["residue_info"]
    for key, coordinates in artifact_mapping["backbone_coordinates"].items():
        np.testing.assert_array_equal(
            restored_mapping["backbone_coordinates"][key], coordinates
        )
    loaded = restored_mapping["full_atom_targets"]
    assert loaded["schema_version"] == 1
    assert loaded["atom_names"].dtype.kind == "U"
    assert loaded["atom_mask"].dtype == np.bool_
    assert loaded["atom_positions"].dtype == np.float32
    for key, values in expected.items():
        if isinstance(values, np.ndarray):
            assert loaded[key].dtype == values.dtype
            np.testing.assert_array_equal(loaded[key], values)
    observed = loaded["atom_mask"]
    assert [loaded["atom_names"][row_mask].tolist() for row_mask in observed] == [
        ["N", "CA", "CB"],
        ["P", "O5'"],
    ]
    assert np.broadcast_to(loaded["atom_names"], observed.shape)[observed].tolist() == [
        "N",
        "CA",
        "CB",
        "P",
        "O5'",
    ]
    np.testing.assert_array_equal(
        loaded["atom_positions"][observed],
        [[1, 0, 0], [1, 1, 1], [2, 1, 1], [2, 2, 2], [3, 2, 2]],
    )
    assert (loaded["atom_names"] != "").all()
    assert len(set(loaded["atom_names"].tolist())) == 65


def test_canonical_validation_and_reuse_accept_named_full_atom_artifact(
    tmp_path, artifact_mapping
):
    assert _preprocessed_artifact_is_reusable(artifact_mapping)
    artifact_reference = tmp_path / "artifact"
    write_preprocessed_artifact(artifact_reference, artifact_mapping)

    result = validate_preprocessed_artifact(artifact_reference)

    assert result.artifact_reference == str(artifact_reference)
    assert result.grid_shape == (4, 4, 4)
    assert result.protein_nodes == result.nucleotide_nodes == 1
    assert result.component_nodes == 0
    assert result.sequence_length is result.sequence_feature_dim is None
    assert _preprocessed_artifact_is_reusable(
        load_preprocessed_artifact(artifact_reference)
    )


@pytest.mark.parametrize("grid_payload", ["array", "npy_bytes"])
def test_canonical_validation_reuses_normalized_details_without_rescanning(
    monkeypatch, artifact_mapping, grid_payload
):
    from cryoflake.preprocess import reuse
    from cryoflake.storage import artifacts
    from cryoflake.targets import schema

    artifact_mapping["sequence"] = np.ones((2, 8), dtype=np.float64)
    # Single-row vectors are accepted by the coordinate contract too.
    for name, coordinates in artifact_mapping["backbone_coordinates"].items():
        artifact_mapping["backbone_coordinates"][name] = coordinates[0]
    if grid_payload == "npy_bytes":
        buffer = io.BytesIO()
        np.save(buffer, artifact_mapping["density_grid"], allow_pickle=False)
        artifact_mapping["density_grid"] = buffer.getvalue()
    monkeypatch.setattr(
        artifacts, "load_preprocessed_artifact", lambda *_, **__: artifact_mapping
    )
    calls = {}

    def count_validation(module, name):
        original = getattr(module, name)

        def counted(*args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    for name in (
        "validate_coordinates",
        "validate_density_grid",
        "validate_full_atom_targets",
        "validate_sequence_embedding_payload",
    ):
        count_validation(reuse, name)
    count_validation(schema, "validate_component_extension_targets")

    result = validate_preprocessed_artifact(
        "artifact", require_sequence_embeddings=True
    )

    assert result.grid_shape == (4, 4, 4)
    assert result.protein_nodes == result.nucleotide_nodes == 1
    assert (result.sequence_length, result.sequence_feature_dim) == (2, 8)
    assert calls == {
        "validate_coordinates": 5,
        "validate_density_grid": 1,
        "validate_full_atom_targets": 1,
        "validate_sequence_embedding_payload": 1,
        "validate_component_extension_targets": 1,
    }
    assert artifact_mapping["sequence"].dtype == np.float64


@pytest.mark.parametrize(
    "removed_field",
    ["prot_mask", "atom_mask", "interface_mask", "proposal_positions", "split_groups"],
)
def test_reuse_and_canonical_validation_reject_removed_redundant_target_fields(
    tmp_path, artifact_mapping, removed_field
):
    component = artifact_mapping["extension_targets"]["component"]
    if removed_field == "prot_mask":
        parent = artifact_mapping["full_atom_targets"]
        key = removed_field
        parent[key] = np.asarray([True, False])
    elif removed_field == "proposal_positions":
        parent = component["proposal_targets"]
        key = "positions"
        parent[key] = component["positions"][parent["node_indices"]]
    else:
        parent = component
        key = removed_field
        parent[key] = {
            "atom_mask": component["atom_names"] != "",
            "interface_mask": component["interface_group_id"] != "",
            "split_groups": {"component": (), "interface": ()},
        }[removed_field]
    assert not _preprocessed_artifact_is_reusable(artifact_mapping)
    artifact_reference = tmp_path / "redundant-artifact"
    write_preprocessed_artifact(artifact_reference, artifact_mapping)
    loaded = load_preprocessed_artifact(artifact_reference)
    if removed_field == "prot_mask":
        restored_parent = loaded["full_atom_targets"]
    else:
        restored_parent = loaded["extension_targets"]["component"]
        if removed_field == "proposal_positions":
            restored_parent = restored_parent["proposal_targets"]
    assert key in restored_parent
    assert not _preprocessed_artifact_is_reusable(loaded)
    with pytest.raises(ValueError, match="canonical preprocessing contract"):
        validate_preprocessed_artifact(artifact_reference)


@pytest.mark.parametrize(
    "missing_fields",
    [("schema_version",), ("atom_names",), ("schema_version", "atom_names")],
)
def test_legacy_full_atom_payloads_are_rejected_by_reuse_and_validation(
    tmp_path, artifact_mapping, missing_fields
):
    for key in missing_fields:
        artifact_mapping["full_atom_targets"].pop(key)
    assert not _preprocessed_artifact_is_reusable(artifact_mapping)
    artifact_reference = tmp_path / "legacy-artifact"
    write_preprocessed_artifact(artifact_reference, artifact_mapping)

    # Valid NPZ+JSON storage does not make an old supervision payload reusable.
    loaded = load_preprocessed_artifact(artifact_reference)
    assert not _preprocessed_artifact_is_reusable(loaded)
    with pytest.raises(ValueError, match="canonical preprocessing contract"):
        validate_preprocessed_artifact(artifact_reference)


@pytest.mark.parametrize("schema_version", [0, 2, -1, True, 1.0, "1", None])
def test_unsupported_full_atom_versions_are_never_reused(
    tmp_path, artifact_mapping, schema_version
):
    artifact_mapping["full_atom_targets"]["schema_version"] = schema_version
    assert not _preprocessed_artifact_is_reusable(artifact_mapping)
    artifact_reference = tmp_path / "unsupported-artifact"
    write_preprocessed_artifact(artifact_reference, artifact_mapping)
    with pytest.raises(ValueError, match="canonical preprocessing contract"):
        validate_preprocessed_artifact(artifact_reference)


def test_per_residue_atom_name_tables_are_rejected_without_silent_broadcast_or_flattening(
    tmp_path, artifact_mapping
):
    targets = artifact_mapping["full_atom_targets"]
    targets["atom_names"] = np.tile(targets["atom_names"], (len(targets["aatype"]), 1))
    with pytest.raises(ValueError, match="atom_names.*shape"):
        validate_full_atom_targets(targets)
    assert not _preprocessed_artifact_is_reusable(artifact_mapping)
    artifact_reference = tmp_path / "per-residue-artifact"
    write_preprocessed_artifact(artifact_reference, artifact_mapping)
    _assert_redundant_target_fields_are_not_serialized(artifact_reference)
    loaded = load_preprocessed_artifact(artifact_reference)
    assert loaded["full_atom_targets"]["atom_names"].shape == (2, 65)
    assert not _preprocessed_artifact_is_reusable(loaded)
    with pytest.raises(ValueError, match="canonical preprocessing contract"):
        validate_preprocessed_artifact(artifact_reference)


@pytest.mark.parametrize("invalid_name", ["", "N", "invalid name"])
def test_global_atom_names_must_describe_unobserved_slots_too(
    tmp_path, artifact_mapping, invalid_name
):
    targets = artifact_mapping["full_atom_targets"]
    assert not targets["atom_mask"][:, 64].any()
    targets["atom_names"] = targets["atom_names"].astype(object)
    targets["atom_names"][64] = invalid_name
    with pytest.raises(ValueError, match="atom_names"):
        validate_full_atom_targets(targets)
    assert not _preprocessed_artifact_is_reusable(artifact_mapping)


def test_component_atoms_keep_their_independent_variable_width_namespace(
    tmp_path, artifact_mapping
):
    structure_path = tmp_path / "mixed.cif"
    structure_path.write_text(
        _POLYMER_CIF
        + "".join(
            f"HETATM C{index + 1} LIG C 3 . C 30 1 C {1 + index / 100} 1 1\n"
            for index in range(67)
        )
    )
    artifact_mapping["extension_targets"] = build_component_extension_targets(
        structure_path, global_origin=np.zeros(3), voxel_size=1.0, grid_shape=(4, 4, 4)
    )
    expected = artifact_mapping["extension_targets"]["component"]
    assert expected["atom_names"].shape == (1, 67)
    artifact_reference = tmp_path / "mixed-artifact"
    write_preprocessed_artifact(artifact_reference, artifact_mapping)
    _assert_redundant_target_fields_are_not_serialized(artifact_reference)
    loaded = load_preprocessed_artifact(artifact_reference)
    assert loaded["full_atom_targets"]["atom_names"].shape == (65,)
    assert loaded["extension_targets"]["schema_version"] == 1
    component = loaded["extension_targets"]["component"]
    assert component["component_id"].tolist() == ["LIG"]
    assert component["atom_names"].shape == (1, 67)
    assert component["atom_positions"].shape == (1, 67, 3)
    assert "atom_mask" not in component
    assert (component["atom_names"] != "").shape == (1, 67)
    assert (component["atom_names"] != "").all()
    assert component["atom_elements"].tolist() == [["C"] * 67]
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            assert component[key].dtype == value.dtype
            np.testing.assert_array_equal(component[key], value)
    assert component["interface_contact_cutoff"] == expected["interface_contact_cutoff"]
    assert _preprocessed_artifact_is_reusable(loaded)
    assert validate_preprocessed_artifact(artifact_reference).component_nodes == 1


def test_storage_preserves_custom_proposal_selection_without_copying_positions(
    tmp_path, artifact_mapping
):
    structure_path = tmp_path / "components.cif"
    structure_path.write_text(
        _POLYMER_CIF
        + "".join(
            f"HETATM C1 LIG {chain} {entity} . {chain} {entity} 1 C {entity / 10} 1 1\n"
            for chain, entity in (("C", 3), ("D", 4), ("E", 5))
        )
    )
    artifact_mapping["extension_targets"] = build_component_extension_targets(
        structure_path, global_origin=np.zeros(3), voxel_size=1.0, grid_shape=(4, 4, 4)
    )
    component = artifact_mapping["extension_targets"]["component"]
    component["proposal_targets"] = {
        "node_indices": np.asarray([2, 0], dtype=np.int64),
        "valid_mask": np.asarray([False, True]),
    }
    expected_positions = component["positions"][[2, 0]].copy()
    artifact_reference = tmp_path / "custom-proposals"
    write_preprocessed_artifact(artifact_reference, artifact_mapping)
    _assert_redundant_target_fields_are_not_serialized(artifact_reference)
    loaded = load_preprocessed_artifact(artifact_reference)
    restored_component = loaded["extension_targets"]["component"]
    proposal = restored_component["proposal_targets"]
    assert set(proposal) == {"node_indices", "valid_mask"}
    np.testing.assert_array_equal(proposal["node_indices"], [2, 0])
    np.testing.assert_array_equal(proposal["valid_mask"], [False, True])
    np.testing.assert_array_equal(
        restored_component["positions"][proposal["node_indices"]], expected_positions
    )
    assert _preprocessed_artifact_is_reusable(loaded)
    assert validate_preprocessed_artifact(artifact_reference).component_nodes == 3


@pytest.mark.parametrize(
    "incompatible_contract",
    [
        "legacy_semantics",
        "full_atom_version",
        "missing_target_fields",
        "redundant_fields",
    ],
)
def test_checkpoint_identity_includes_full_atom_schema_and_rejects_old_progress(
    tmp_path, incompatible_contract
):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("emdb_id,wwpdb_id,map_path,structure_path\n")
    contract = preprocess_checkpoint_contract(
        manifest_csv=manifest_path,
        manifest_rows=[],
        manifest_dir=tmp_path,
        artifact_output_dir=tmp_path / "artifacts",
        artifact_s3_prefix=None,
        s3_endpoint_url=None,
        artifact_reuse_dirs=(),
        target_voxel_size=1.0,
        fix_shift=True,
        preprocess_adapter=abs,
        interface_contact_cutoff=5.0,
        require_sequence_embeddings=False,
        sequence_embedding_loader=len,
        download_errors={},
        preprocess_checkpoint_token="full-atom-schema-regression",
    )
    assert contract["artifact_contract_version"] == 2
    assert contract["format_contracts"]["targets"]["full_atom_schema_version"] == 1
    assert contract["format_contracts"]["targets"]["schema_version"] == 1
    target_fields = contract["target_fields"]
    assert "prot_mask" not in target_fields["full_atom_targets"]
    assert not {"atom_mask", "interface_mask", "split_groups"}.intersection(
        target_fields["extension_targets.component"]
    )
    assert (
        "positions" not in target_fields["extension_targets.component.proposal_targets"]
    )
    current_fingerprint = json_sha256(contract)
    legacy_contract = deepcopy(contract)
    if incompatible_contract == "legacy_semantics":
        legacy_contract["artifact_contract_version"] = 3
        legacy_contract["format_contracts"]["targets"].pop("full_atom_schema_version")
    elif incompatible_contract == "full_atom_version":
        legacy_contract["format_contracts"]["targets"]["full_atom_schema_version"] = 2
    elif incompatible_contract == "missing_target_fields":
        legacy_contract.pop("target_fields")
    else:
        legacy_contract["target_fields"]["full_atom_targets"].append("prot_mask")
        legacy_contract["target_fields"]["extension_targets.component"].extend(
            ["atom_mask", "interface_mask", "split_groups"]
        )
        legacy_contract["target_fields"][
            "extension_targets.component.proposal_targets"
        ].append("positions")
        for fields in legacy_contract["target_fields"].values():
            fields.sort()
    summary_path = tmp_path / "summary.csv"
    with PreprocessCheckpoint(summary_path, legacy_contract) as checkpoint:
        checkpoint.record({"row_index": 1, "state": "durable_success"})
        metadata_path = checkpoint.metadata_path
        journal_path = checkpoint.journal_path
        assert checkpoint.fingerprint != current_fingerprint
    original_metadata = metadata_path.read_bytes()
    original_journal = journal_path.read_bytes()
    assert json.loads(original_metadata)["format_version"] == 5
    assert PREPROCESS_CHECKPOINT_FORMAT_VERSION == 5

    with pytest.raises(CheckpointContractMismatchError, match="does not match"):
        PreprocessCheckpoint(summary_path, contract)

    assert metadata_path.read_bytes() == original_metadata
    assert journal_path.read_bytes() == original_journal

    current_summary_path = tmp_path / "current-summary.csv"
    with PreprocessCheckpoint(current_summary_path, contract) as current_checkpoint:
        assert current_checkpoint.fingerprint == current_fingerprint
        current_checkpoint.record({"row_index": 1, "state": "durable_success"})
    with PreprocessCheckpoint(current_summary_path, contract) as recovered_checkpoint:
        assert recovered_checkpoint.records[1]["state"] == "durable_success"
