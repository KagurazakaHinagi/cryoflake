"""Checkpoint recovery validates renamed records before reuse or publication."""

from __future__ import annotations

import json
from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path

import pytest

from cryoflake.preprocess.checkpoint import (
    CheckpointContractMismatchError,
    PreprocessCheckpoint,
    _local_cache_identity,
    grouped_record_from_checkpoint,
)
from cryoflake.preprocess.recovery import (
    _durable_result_from_checkpoint,
    _pending_result_from_checkpoint,
)


def _record(reference="s3://bucket/artifact", *, state="durable_success"):
    return {
        "row_index": 1,
        "state": state,
        "artifact_reference": reference,
        "target_artifact_reference": reference,
        "summary_row": {
            "artifact_reference": reference,
            "success": "true" if state == "durable_success" else "false",
            "split": "valid",
            "split_seed": "1",
            "status": "processed" if state == "durable_success" else "upload_failed",
            "error": "" if state == "durable_success" else "interrupted upload",
        },
        "grouped_split_record": {
            "artifact_reference": reference,
            "artifact_group_id": "1abc",
            "component_group_ids": ["component-a"],
            "interface_group_ids": ["interface-a"],
        },
        "artifact_identity": {"kind": "canonical_artifact", "members": {}},
        "success_status": "processed",
        "artifact_voxel_size": "1.5",
        "grid_shape": "4x4x4",
        "entry_label": "EMD-1234 / 1abc",
    }


class _Uploads:
    def __init__(self):
        self.calls = []
        self.future = Future()

    def stage(self, cached_artifact_dir, remote_artifact_reference):
        self.calls.append((cached_artifact_dir, remote_artifact_reference))
        return self.future


@pytest.fixture
def pending_record(tmp_path):
    artifact = tmp_path / ".upload_cache_previous/row_00000001/artifact"
    artifact.mkdir(parents=True)
    (artifact / "metadata.json").write_text("cached metadata")
    (artifact / "core.npz").write_bytes(b"cached arrays")
    record = _record(state="upload_pending")
    record["cached_artifact_dir"] = str(artifact)
    record["cache_identity"] = _local_cache_identity(artifact)
    record["summary_row"]["cached_artifact_dir"] = str(artifact)
    return record


@pytest.mark.parametrize(
    "changes",
    [
        {"artifact_group_id": ""},
        {"artifact_group_id": None},
        {"artifact_group_id": 7},
        {"artifact_reference": None},
        {"artifact_reference": 7},
        {"artifact_reference": "https://bucket/artifact"},
        {"component_group_ids": [None]},
        {"component_group_ids": ["   "]},
        {"interface_group_ids": [7]},
    ],
)
def test_malformed_grouped_checkpoint_records_are_not_coerced_or_raised(changes):
    payload = _record()["grouped_split_record"]
    payload.update(changes)

    assert grouped_record_from_checkpoint(payload) is None


@pytest.mark.parametrize("grouped_record", [None, {}, "old record"])
def test_durable_recovery_requires_grouped_record_for_complete_split_outputs(
    grouped_record,
):
    record = _record()
    record["grouped_split_record"] = grouped_record

    assert _durable_result_from_checkpoint(record, split_seed=7) is None


@pytest.mark.parametrize(
    "field", ["summary_row", "grouped_split_record", "target_artifact_reference"]
)
def test_durable_recovery_rejects_disagreeing_artifact_references(field):
    record = _record()
    if isinstance(record[field], dict):
        record[field]["artifact_reference"] = "s3://bucket/other"
    else:
        record[field] = "s3://bucket/other"

    assert _durable_result_from_checkpoint(record, split_seed=7) is None


def test_durable_recovery_preserves_group_membership_and_updates_summary_aliases():
    record = _record()
    record["summary_row"]["artifact_reference"] += "/metadata.json"
    record["grouped_split_record"]["artifact_reference"] += "/metadata.json"
    original = deepcopy(record)

    result = _durable_result_from_checkpoint(record, split_seed=7)

    assert result is not None
    assert result.artifact_reference == "s3://bucket/artifact"
    assert result.grouped_split_record.artifact_reference == result.artifact_reference
    assert result.grouped_split_record.component_group_ids == ("component-a",)
    assert result.grouped_split_record.interface_group_ids == ("interface-a",)
    assert result.summary_row["artifact_reference"] == str(result.artifact_reference)
    assert result.summary_row["split"] == ""
    assert result.summary_row["split_seed"] == "7"
    assert record == original


@pytest.mark.parametrize("success", ["false", None, True])
def test_durable_recovery_requires_a_successful_canonical_summary(success):
    record = _record()
    record["summary_row"]["success"] = success

    assert _durable_result_from_checkpoint(record, split_seed=7) is None


@pytest.mark.parametrize(
    "field", ["grouped_split_record", "summary_row", "artifact_reference"]
)
def test_pending_recovery_validates_references_before_staging_upload(
    tmp_path, pending_record, field
):
    if field == "grouped_split_record":
        pending_record[field]["artifact_group_id"] = ""
    elif field == "summary_row":
        pending_record[field]["artifact_reference"] = "s3://bucket/other"
    else:
        pending_record[field] = "s3://bucket/other"
    uploads = _Uploads()

    assert (
        _pending_result_from_checkpoint(
            pending_record,
            upload_cache=uploads,
            artifact_output_dir=tmp_path,
            split_seed=7,
        )
        is None
    )

    assert uploads.calls == []
    assert Path(pending_record["cached_artifact_dir"]).is_dir()


@pytest.mark.parametrize("state", ["upload_pending", "terminal_failure"])
def test_pending_recovery_stages_verified_cache_with_consistent_summary(
    tmp_path, pending_record, state
):
    pending_record["state"] = state
    if state == "terminal_failure":
        pending_record["artifact_reference"] = None
    pending_record["summary_row"]["artifact_reference"] += "/metadata.json"
    pending_record["grouped_split_record"]["artifact_reference"] += "/metadata.json"
    original = deepcopy(pending_record)
    uploads = _Uploads()

    result = _pending_result_from_checkpoint(
        pending_record,
        upload_cache=uploads,
        artifact_output_dir=tmp_path,
        split_seed=7,
    )

    assert result is not None
    assert uploads.calls == [
        (Path(pending_record["cached_artifact_dir"]), "s3://bucket/artifact")
    ]
    assert result.artifact_reference == result.grouped_split_record.artifact_reference
    assert result.summary_row["artifact_reference"] == result.artifact_reference
    assert result.upload_future is uploads.future
    assert result.summary_row["success"] == "false"
    assert result.summary_row["status"] == result.summary_row["error"] == ""
    assert result.summary_row["split"] == ""
    assert result.summary_row["split_seed"] == "7"
    assert pending_record == original


def test_pending_recovery_rejects_changed_cache_without_staging_or_deleting_it(
    tmp_path, pending_record
):
    artifact = Path(pending_record["cached_artifact_dir"])
    (artifact / "core.npz").write_bytes(b"modified cached arrays")
    uploads = _Uploads()

    assert (
        _pending_result_from_checkpoint(
            pending_record,
            upload_cache=uploads,
            artifact_output_dir=tmp_path,
            split_seed=7,
        )
        is None
    )

    assert uploads.calls == []
    assert (artifact / "core.npz").read_bytes() == b"modified cached arrays"


@pytest.mark.parametrize("extra", ["path", "example_group_id", "unknown_field"])
def test_grouped_records_reject_extra_fields_even_with_canonical_fields_present(extra):
    grouped_record = _record()["grouped_split_record"]
    grouped_record[extra] = "conflicting legacy value"

    assert grouped_record_from_checkpoint(grouped_record) is None


@pytest.mark.parametrize(
    "extra",
    [
        "output_path",
        "target_output_path",
        "upload_cache_path",
        "component_split_record",
        "post_voxel_size",
    ],
)
def test_recovery_rejects_conflicting_old_journal_keys(extra):
    record = _record()
    record[extra] = "legacy value"

    assert _durable_result_from_checkpoint(record, split_seed=7) is None


@pytest.mark.parametrize(
    "extra",
    [
        "preprocessed_path",
        "upload_cache_path",
        "original_resolution",
        "pre_voxel_size",
        "post_voxel_size",
    ],
)
def test_recovery_rejects_conflicting_old_summary_keys(extra):
    record = _record()
    record["summary_row"][extra] = "legacy value"

    assert _durable_result_from_checkpoint(record, split_seed=7) is None


def test_pending_legacy_fields_are_rejected_without_staging_or_deleting_cache(
    tmp_path, pending_record
):
    pending_record["upload_cache_path"] = pending_record["cached_artifact_dir"]
    uploads = _Uploads()

    assert (
        _pending_result_from_checkpoint(
            pending_record,
            upload_cache=uploads,
            artifact_output_dir=tmp_path,
            split_seed=7,
        )
        is None
    )

    assert uploads.calls == []
    assert Path(pending_record["cached_artifact_dir"]).is_dir()


def _progress_files(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with PreprocessCheckpoint(summary_path, {}) as checkpoint:
        checkpoint.record(_record())
        metadata_path = checkpoint.metadata_path
        journal_path = checkpoint.journal_path
    return summary_path, metadata_path, journal_path


@pytest.mark.parametrize(
    "version",
    [
        4,
        5,
        5.0,
        True,
        None,
        "5",
        "",
        "C2610211",
        "C2610200",
        "C2610320",
        "C26102100",
        "S2610210",
        "c2610210",
        " C2610210",
        ["C2610210"],
        {"version": "C2610210"},
    ],
)
def test_checkpoint_rejects_legacy_or_unsupported_versions_without_mutation(
    tmp_path, version
):
    summary_path, metadata_path, journal_path = _progress_files(tmp_path)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["format_version"] == "C2610210"
    metadata["format_version"] = version
    metadata_path.write_text(json.dumps(metadata))
    original_metadata = metadata_path.read_bytes()
    original_journal = journal_path.read_bytes()

    with pytest.raises(CheckpointContractMismatchError):
        PreprocessCheckpoint(summary_path, {})

    assert metadata_path.read_bytes() == original_metadata
    assert journal_path.read_bytes() == original_journal


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_version",
        "float_version",
        "nonfinite",
        "contract_mismatch",
        "extra_key",
    ],
)
def test_checkpoint_metadata_requires_unambiguous_canonical_json(tmp_path, change):
    summary_path, metadata_path, journal_path = _progress_files(tmp_path)
    metadata = json.loads(metadata_path.read_text())
    if change == "duplicate_version":
        serialized = json.dumps(metadata).replace(
            '"format_version": "C2610210"',
            '"format_version": 5, "format_version": "C2610210"',
        )
    else:
        if change == "float_version":
            metadata["format_version"] = 5.0
        elif change == "nonfinite":
            metadata["contract"] = {"invalid": float("nan")}
        elif change == "contract_mismatch":
            metadata["contract"] = {"different": "contract"}
        else:
            metadata["version"] = 4
        serialized = json.dumps(metadata)
    metadata_path.write_text(serialized)
    original_journal = journal_path.read_bytes()

    with pytest.raises(CheckpointContractMismatchError):
        PreprocessCheckpoint(summary_path, {})

    assert metadata_path.read_text() == serialized
    assert journal_path.read_bytes() == original_journal


@pytest.mark.parametrize("change", ["duplicate_reference", "nonfinite"])
def test_checkpoint_skips_ambiguous_or_nonfinite_json_rows(tmp_path, change):
    summary_path, _, journal_path = _progress_files(tmp_path)
    record = json.loads(journal_path.read_text())
    if change == "duplicate_reference":
        serialized = json.dumps(record).replace(
            '"artifact_reference": "s3://bucket/artifact"',
            '"artifact_reference": "s3://bucket/wrong", '
            '"artifact_reference": "s3://bucket/artifact"',
            1,
        )
    else:
        record["artifact_voxel_size"] = float("inf")
        serialized = json.dumps(record)
    journal_path.write_text(serialized + "\n")

    with PreprocessCheckpoint(summary_path, {}) as recovered:
        assert recovered.records == {}

    assert journal_path.read_text() == serialized + "\n"


@pytest.mark.parametrize(
    "change", ["duplicate_reference", "nonfinite", "overflow_float"]
)
def test_ambiguous_latest_row_retires_only_its_older_progress(tmp_path, change):
    summary_path, _, journal_path = _progress_files(tmp_path)
    record = json.loads(journal_path.read_text())
    with PreprocessCheckpoint(summary_path, {}) as checkpoint:
        checkpoint.record({"row_index": 2, "state": "terminal_failure"})
    if change == "duplicate_reference":
        serialized = json.dumps(record).replace(
            '"artifact_reference": "s3://bucket/artifact"',
            '"artifact_reference": "s3://bucket/wrong", '
            '"artifact_reference": "s3://bucket/artifact"',
            1,
        )
    else:
        record["artifact_voxel_size"] = float("inf")
        serialized = json.dumps(record)
        if change == "overflow_float":
            serialized = serialized.replace("Infinity", "1e999")
    with journal_path.open("a") as output:
        output.write(serialized + "\n")

    with PreprocessCheckpoint(summary_path, {}) as recovered:
        assert set(recovered.records) == {2}


def test_duplicate_row_indices_do_not_select_either_ambiguous_row_for_recovery(
    tmp_path,
):
    summary_path, _, journal_path = _progress_files(tmp_path)
    record = json.loads(journal_path.read_text())
    with PreprocessCheckpoint(summary_path, {}) as checkpoint:
        checkpoint.record({"row_index": 2, "state": "terminal_failure"})
    serialized = json.dumps(record).replace(
        '"row_index": 1', '"row_index": 1, "row_index": 2'
    )
    with journal_path.open("a") as output:
        output.write(serialized + "\n")

    with PreprocessCheckpoint(summary_path, {}) as recovered:
        assert recovered.records == {}


def test_later_noncanonical_row_retires_older_progress_for_retry(tmp_path):
    summary_path, _, journal_path = _progress_files(tmp_path)
    record = json.loads(journal_path.read_text())
    record["output_path"] = record["artifact_reference"]
    with journal_path.open("a") as output:
        output.write(json.dumps(record) + "\n")

    with PreprocessCheckpoint(summary_path, {}) as recovered:
        assert recovered.records == {}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_checkpoint_writer_rejects_nonfinite_rows_without_changing_progress(
    tmp_path, value
):
    with PreprocessCheckpoint(tmp_path / "summary.csv", {}) as checkpoint:
        original = checkpoint.journal_path.read_bytes()
        with pytest.raises(ValueError, match="JSON compliant"):
            checkpoint.record({"row_index": 1, "artifact_voxel_size": value})
        assert checkpoint.records == {}
        assert checkpoint.journal_path.read_bytes() == original


def test_checkpoint_writer_rejects_old_keys_before_journaling(tmp_path):
    with PreprocessCheckpoint(tmp_path / "summary.csv", {}) as checkpoint:
        with pytest.raises(ValueError, match="unexpected or malformed fields"):
            checkpoint.record({"row_index": 1, "output_path": "artifact"})
        assert checkpoint.records == {}
        assert checkpoint.journal_path.read_bytes() == b""


def test_nonfinite_checkpoint_contract_is_rejected_before_creating_progress(tmp_path):
    summary_path = tmp_path / "new/summary.csv"
    with pytest.raises(ValueError, match="JSON compliant"):
        PreprocessCheckpoint(summary_path, {"invalid": float("nan")})
    assert not summary_path.parent.exists()
