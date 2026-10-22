"""Canonical artifact identifiers have no forwarding names or wire aliases."""

from __future__ import annotations

import inspect
import json
from importlib import import_module

import numpy as np
import pytest
from typer.testing import CliRunner

import cryoflake
import cryoflake.storage as storage
from cryoflake import cli, metadata
from cryoflake.config import compose_config
from cryoflake.manifests import splits
from cryoflake.manifests.manifest import split_artifact_references
from cryoflake.preprocess.checkpoint import (
    CheckpointContractMismatchError,
    PreprocessCheckpoint,
    checkpoint_record_for_result,
    grouped_record_from_checkpoint,
)
from cryoflake.preprocess.models import _PreprocessManifestRowResult
from cryoflake.storage import store, transports


@pytest.fixture(autouse=True)
def no_credential_loading(monkeypatch):
    monkeypatch.setattr(transports, "load_s3_environment", lambda: False)


@pytest.mark.parametrize(
    "name",
    [
        "preprocessed_artifact_member_reference",
        "replace_preprocessed_artifact_sequence",
        "load_preprocessed_artifact",
        "write_preprocessed_artifact",
        "upload_preprocessed_artifact",
    ],
)
def test_public_artifact_storage_names_share_the_canonical_implementation(name):
    implementation = getattr(store, name)
    assert getattr(storage, name) is implementation
    assert getattr(cryoflake, name) is implementation
    assert implementation.__name__ == name


def test_artifact_signatures_describe_references_and_mappings():
    assert list(inspect.signature(store.load_preprocessed_artifact).parameters)[0] == (
        "artifact_reference"
    )
    assert list(inspect.signature(store.write_preprocessed_artifact).parameters)[
        :2
    ] == [
        "artifact_reference",
        "artifact_mapping",
    ]
    assert list(inspect.signature(store.upload_preprocessed_artifact).parameters)[
        :2
    ] == [
        "local_artifact_reference",
        "remote_artifact_reference",
    ]
    for name in (
        "get_preprocessed_artifact_grid",
        "get_preprocessed_artifact_voxel_size",
        "get_preprocessed_artifact_global_origin",
    ):
        function = getattr(metadata, name)
        assert getattr(cryoflake, name) is function
        assert list(inspect.signature(function).parameters)[0] == "artifact_mapping"


@pytest.mark.parametrize(
    "module_name, removed_names",
    [
        (
            "cryoflake.storage.store",
            ("preprocessed_member_reference", "replace_preprocessed_sequence"),
        ),
        (
            "cryoflake.metadata",
            (
                "get_preprocessed_grid",
                "get_preprocessed_voxel_size",
                "get_preprocessed_global_origin",
                "validate_preprocessed_grid",
                "validate_preprocessed_coordinates",
                "validate_preprocessed_global_origin",
                "validate_preprocessed_voxel_size",
                "validate_preprocessed_residue_labels",
            ),
        ),
        (
            "cryoflake.manifests.splits",
            (
                "ComponentSplitRecord",
                "ComponentSplitResult",
                "SplitStrategy",
                "split_component_records",
                "write_component_split_report",
                "read_holdout_groups",
            ),
        ),
        (
            "cryoflake.manifests.manifest",
            (
                "MapListEntry",
                "read_maplist_csv_entries",
                "read_dataset_list",
                "write_dataset_list",
                "split_preprocessed_paths",
            ),
        ),
        ("cryoflake.storage.artifacts", ("validate_dataset_artifacts",)),
        ("cryoflake.storage.locks", ("preprocessed_write_locks",)),
        ("cryoflake.storage.upload_cache", ("BatchedUploadCache",)),
        (
            "cryoflake.storage.transports",
            (
                "path_exists",
                "path_identity",
                "download_file_to_path",
                "upload_file_to_uri",
            ),
        ),
        ("cryoflake.targets.full_atom", ("full_atom_targets_from_protein",)),
        (
            "cryoflake.preprocess.checkpoint",
            ("component_record_from_checkpoint", "path_from_checkpoint"),
        ),
    ],
)
def test_superseded_api_names_are_absent(module_name, removed_names):
    module = import_module(module_name)
    assert all(not hasattr(module, name) for name in removed_names)


@pytest.mark.parametrize(
    "module_name", ["cryoflake.manifests.maplist", "cryoflake.preprocess.maplist"]
)
def test_superseded_module_paths_are_absent(module_name):
    with pytest.raises(ModuleNotFoundError):
        import_module(module_name)


def test_artifact_list_and_validation_operations_have_one_public_name():
    manifest = import_module("cryoflake.manifests.manifest")
    artifacts = import_module("cryoflake.storage.artifacts")
    for name in ("read_artifact_list", "write_artifact_list"):
        assert getattr(cryoflake, name) is getattr(manifest, name)
    assert cryoflake.validate_artifact_list is artifacts.validate_artifact_list
    assert list(inspect.signature(artifacts.validate_artifact_list).parameters)[0] == (
        "artifact_list_path"
    )
    for name in (
        "read_dataset_list",
        "write_dataset_list",
        "validate_dataset_artifacts",
    ):
        assert not hasattr(cryoflake, name)


@pytest.mark.parametrize(
    "old_name",
    [
        "maplist_csv",
        "output_csv",
        "preprocessor",
        "preprocess_output_dir",
        "preprocess_output_s3_prefix",
        "preprocess_reuse_dirs",
        "dataset_list_output",
        "valid_dataset_list_output",
        "validation_split_by",
        "holdout_groups_file",
        "original_structure",
        "fasta_sequence",
    ],
)
def test_superseded_configuration_keys_are_not_aliases(old_name):
    with pytest.raises(ValueError, match=old_name):
        compose_config(values={"pipeline": {old_name: None}})


@pytest.mark.parametrize(
    "command, old_option",
    [
        ("prepare-manifest", "--input-csv"),
        ("prepare-manifest", "--output-csv"),
        ("prepare-manifest", "--preprocess-output-dir"),
        ("prepare-manifest", "--preprocess-output-s3-prefix"),
        ("prepare-manifest", "--preprocess-reuse-dir"),
        ("prepare-manifest", "--preprocessor"),
        ("prepare-manifest", "--dataset-list-output"),
        ("prepare-manifest", "--valid-dataset-list-output"),
        ("prepare-manifest", "--validation-split-by"),
        ("prepare-manifest", "--holdout-groups-file"),
        ("preprocess", "--original-structure"),
        ("preprocess", "--fasta-sequence"),
        ("preprocess", "--dont-fix-shift"),
        ("prepare-manifest", "--dont-fix-shift"),
    ],
)
def test_superseded_cli_options_are_rejected(command, old_option):
    result = CliRunner().invoke(cli.app, [command, old_option, "unused"])
    assert result.exit_code == 2
    assert "No such option" in result.output


def test_superseded_prepare_command_and_pipeline_are_rejected():
    result = CliRunner().invoke(cli.app, ["prepare-maplist", "--help"])
    assert result.exit_code == 2
    assert "No such command" in result.output
    with pytest.raises(ValueError):
        compose_config(pipeline="maplist")


def test_artifact_reference_split_deduplicates_remote_member_references():
    train, validation = split_artifact_references(
        ["s3://bucket/first", "s3://bucket/first/metadata.json", "s3://bucket/second"],
        validation_fraction=0.5,
        seed=1,
    )
    assert set(train + validation) == {"s3://bucket/first", "s3://bucket/second"}
    assert len(train) == len(validation) == 1


@pytest.mark.parametrize("split_strategy", ["component", "interface"])
def test_grouped_split_report_uses_reference_names_and_version_three(
    tmp_path, split_strategy
):
    records = (
        splits.GroupedSplitRecord("s3://bucket/first", "1abc", ("c-a",), ("i-a",)),
        splits.GroupedSplitRecord("s3://bucket/second", "2def", ("c-b",), ("i-b",)),
    )
    selected = "c-a" if split_strategy == "component" else "i-a"
    result = splits.split_grouped_records(
        records,
        split_strategy=split_strategy,
        validation_fraction=0.5,
        seed=1,
        holdout_group_ids=[selected],
    )
    assert result.train_artifact_references == ("s3://bucket/second",)
    assert result.validation_artifact_references == ("s3://bucket/first",)
    assert not hasattr(result, "strategy")
    assert not hasattr(result, "train_paths")
    assert not hasattr(records[0], "path")
    report = tmp_path / "split.json"
    splits.write_grouped_split_report(result, report)
    payload = json.loads(report.read_text())
    assert set(payload) == {
        "schema_version",
        "split_strategy",
        "validation_fraction",
        "seed",
        "selected_holdout_group_ids",
        "num_records",
        "num_train_records",
        "num_validation_records",
        "assignments",
    }
    assert payload["schema_version"] == "G2610210"
    assert payload["split_strategy"] == split_strategy
    assert payload["num_train_records"] == payload["num_validation_records"] == 1
    assert [row["artifact_reference"] for row in payload["assignments"]] == [
        record.artifact_reference for record in records
    ]
    assert all("path" not in row for row in payload["assignments"])
    assert splits.read_holdout_group_ids(
        report, expected_split_strategy=split_strategy
    ) == (selected,)
    payload["schema_version"] = 2
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_version"):
        splits.read_holdout_group_ids(report)


def test_grouped_split_does_not_accept_old_field_or_keyword_names(tmp_path):
    with pytest.raises(TypeError, match="path"):
        splits.GroupedSplitRecord(path="artifact", artifact_group_id="1abc")
    with pytest.raises(TypeError, match="strategy"):
        splits.split_grouped_records(
            [], strategy="component", validation_fraction=0.5, seed=1
        )
    with pytest.raises(TypeError, match="expected_strategy"):
        splits.read_holdout_group_ids(
            tmp_path / "missing", expected_strategy="component"
        )


def test_checkpoint_date_version_uses_canonical_reference_fields(tmp_path):
    grouped_record = splits.GroupedSplitRecord(tmp_path / "artifact", "1abc")
    result = _PreprocessManifestRowResult(
        row_index=1,
        artifact_reference=grouped_record.artifact_reference,
        summary_row={"success": "true", "artifact_reference": "s3://bucket/target"},
        grouped_split_record=grouped_record,
    )
    record = checkpoint_record_for_result(result)
    assert record["artifact_reference"] == str(grouped_record.artifact_reference)
    assert record["target_artifact_reference"] == "s3://bucket/target"
    assert record["cached_artifact_dir"] is None
    assert (
        grouped_record_from_checkpoint(record["grouped_split_record"]) == grouped_record
    )
    assert not {
        "output_path",
        "target_output_path",
        "upload_cache_path",
        "component_split_record",
    }.intersection(record)

    summary = tmp_path / "summary.csv"
    with PreprocessCheckpoint(summary, {}) as checkpoint:
        checkpoint.record(record)
        metadata_path = checkpoint.metadata_path
        journal_path = checkpoint.journal_path
    metadata_document = json.loads(metadata_path.read_text())
    assert metadata_document["format_version"] == "C2610210"
    with PreprocessCheckpoint(summary, {}) as recovered:
        restored = recovered.records[1]
        assert restored["grouped_split_record"] == record["grouped_split_record"]

    metadata_document["format_version"] = 4
    metadata_path.write_text(json.dumps(metadata_document))
    original_metadata = metadata_path.read_bytes()
    original_journal = journal_path.read_bytes()
    with pytest.raises(CheckpointContractMismatchError):
        PreprocessCheckpoint(summary, {})
    assert metadata_path.read_bytes() == original_metadata
    assert journal_path.read_bytes() == original_journal


def test_density_grid_and_backbone_coordinates_roundtrip_with_canonical_keywords(
    tmp_path,
):
    artifact = tmp_path / "artifact"
    density_grid = np.ones((2, 2, 2), dtype=np.float64)
    backbone_coordinates = {"CA": np.zeros((1, 3), dtype=np.float32)}
    store.write_preprocessed_artifact(
        artifact_reference=artifact,
        artifact_mapping={
            "density_grid": density_grid,
            "backbone_coordinates": backbone_coordinates,
        },
    )
    mapping = store.load_preprocessed_artifact(artifact_reference=artifact)
    assert set(mapping) == {"density_grid", "backbone_coordinates"}
    assert mapping["density_grid"].dtype == np.float32
    np.testing.assert_array_equal(mapping["density_grid"], density_grid)
    np.testing.assert_array_equal(
        mapping["backbone_coordinates"]["CA"], backbone_coordinates["CA"]
    )
