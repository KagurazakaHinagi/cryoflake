"""Artifact terminology is canonical in split APIs and persisted progress."""

from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from cryoflake.config import CryoflakeConfig, compose_config, validate_config
from cryoflake.manifests.splits import (
    GroupedSplitRecord,
    read_holdout_group_ids,
    split_grouped_records,
    write_grouped_split_report,
)
from cryoflake.preprocess import manifest
from cryoflake.preprocess.checkpoint import (
    CheckpointContractMismatchError,
    PreprocessCheckpoint,
    checkpoint_record_for_result,
    grouped_record_from_checkpoint,
)
from cryoflake.preprocess.models import _PreprocessManifestRowResult
from cryoflake.storage import transports


@pytest.mark.parametrize("overrides", [(), ("pipeline.split_strategy=artifact",)])
def test_artifact_split_default_and_explicit_configuration(overrides, monkeypatch):
    # Config composition normally loads checkout credentials; these tests need none.
    monkeypatch.setattr(transports, "load_s3_environment", lambda: False)
    config = compose_config(overrides)

    assert config.pipeline.split_strategy == "artifact"
    assert CryoflakeConfig().pipeline.split_strategy == "artifact"
    validate_config(config)


def test_configuration_rejects_example_split_choice():
    config = OmegaConf.structured(CryoflakeConfig)
    config.pipeline.split_strategy = "example"

    with pytest.raises(ValueError, match="split_strategy.*artifact"):
        validate_config(config)


def test_manifest_rejects_example_split_before_reading_or_creating_outputs(
    tmp_path, monkeypatch
):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("Invalid split options must fail before manifest I/O")

    monkeypatch.setattr(manifest, "read_manifest_csv", unexpected_read)
    output = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="split_strategy.*artifact"):
        manifest.preprocess_manifest_csv(
            tmp_path / "missing.csv",
            output,
            split_strategy="example",
        )

    assert not output.exists()


def test_grouped_split_record_has_one_artifact_group_identifier():
    record = GroupedSplitRecord("artifact", artifact_group_id="  1abc  ")

    assert record.artifact_group_id == "1abc"
    assert not hasattr(record, "example_group_id")
    with pytest.raises(TypeError, match="example_group_id"):
        GroupedSplitRecord("artifact", example_group_id="1abc")


@pytest.mark.parametrize("strategy", ["component", "interface"])
def test_grouped_reports_use_version_three_and_artifact_group_keys(tmp_path, strategy):
    records = [
        GroupedSplitRecord("first", "1abc", ("component-a",), ("interface-a",)),
        GroupedSplitRecord("second", "2def", ("component-b",), ("interface-b",)),
    ]
    selected = f"{strategy}-a"
    split = split_grouped_records(
        records,
        split_strategy=strategy,
        validation_fraction=0.5,
        seed=7,
        holdout_group_ids=[selected],
    )
    report = tmp_path / "split.json"

    write_grouped_split_report(result=split, split_report_path=report)
    payload = json.loads(report.read_text())

    assert payload["schema_version"] == 3
    assert [row["artifact_group_id"] for row in payload["assignments"]] == [
        "1abc",
        "2def",
    ]
    assert all("example_group_id" not in row for row in payload["assignments"])
    assert read_holdout_group_ids(
        holdout_group_ids_path=report, expected_split_strategy=strategy
    ) == (selected,)

    payload["schema_version"] = 1
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_version"):
        read_holdout_group_ids(report, expected_split_strategy=strategy)


def test_checkpoint_roundtrips_the_artifact_group_identifier(tmp_path):
    group = GroupedSplitRecord(
        tmp_path / "artifact", "1abc", ("component-a",), ("interface-a",)
    )
    result = _PreprocessManifestRowResult(
        row_index=1,
        artifact_reference=group.artifact_reference,
        summary_row={"success": "true"},
        grouped_split_record=group,
    )
    record = checkpoint_record_for_result(result)

    assert record["grouped_split_record"] == {
        "artifact_reference": str(group.artifact_reference),
        "artifact_group_id": "1abc",
        "component_group_ids": ["component-a"],
        "interface_group_ids": ["interface-a"],
    }
    summary = tmp_path / "summary.csv"
    with PreprocessCheckpoint(summary, {}) as checkpoint:
        checkpoint.record(record)
        metadata = json.loads(checkpoint.metadata_path.read_text())
        assert metadata["format_version"] == 5

    with PreprocessCheckpoint(summary, {}) as recovered:
        restored = recovered.records[1]["grouped_split_record"]
        assert "example_group_id" not in restored
        assert grouped_record_from_checkpoint(restored) == group


def test_version_four_checkpoint_fails_before_decoding_old_group_keys(tmp_path):
    summary = tmp_path / "summary.csv"
    with PreprocessCheckpoint(summary, {}) as checkpoint:
        metadata_path = checkpoint.metadata_path
        journal_path = checkpoint.journal_path
        checkpoint.record(
            {
                "row_index": 1,
                "grouped_split_record": {
                    "artifact_reference": "artifact",
                    "artifact_group_id": "1abc",
                    "component_group_ids": [],
                    "interface_group_ids": [],
                },
            }
        )

    record = json.loads(journal_path.read_text())
    grouped_record = record["grouped_split_record"]
    grouped_record["example_group_id"] = grouped_record.pop("artifact_group_id")
    journal_path.write_text(json.dumps(record) + "\n")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["format_version"] == 5
    metadata["format_version"] = 4
    metadata_path.write_text(json.dumps(metadata))
    original_metadata = metadata_path.read_bytes()
    original_journal = journal_path.read_bytes()

    with pytest.raises(CheckpointContractMismatchError, match="does not match"):
        PreprocessCheckpoint(summary, {})

    assert metadata_path.read_bytes() == original_metadata
    assert journal_path.read_bytes() == original_journal
