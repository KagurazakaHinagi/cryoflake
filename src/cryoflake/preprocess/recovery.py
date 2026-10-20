"""Recovery support for manifest preprocessing."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional

from cryoflake.manifests.splits import GroupedSplitRecord
from cryoflake.preprocess.checkpoint import (
    _checkpoint_record_fields_are_canonical,
    _checkpoint_row_index,
    artifact_reference_from_checkpoint,
    checkpoint_cache_matches,
    grouped_record_from_checkpoint,
)
from cryoflake.preprocess.models import _PreprocessManifestRowResult
from cryoflake.storage.transports import (
    is_remote_uri,
)
from cryoflake.storage.upload_cache import BatchedArtifactUploadCache


def _reference_identity(reference: object) -> Path | str:
    if not isinstance(reference, (str, Path)):
        raise ValueError("Checkpoint artifact reference must be a string")
    root = artifact_reference_from_checkpoint(os.fspath(reference))
    return root if isinstance(root, str) else root.resolve()


def _validated_recovery_fields(
    record: Mapping[str, object],
    *,
    reference_field: str,
    split_seed: int,
) -> Optional[tuple[int, Path | str, dict[str, str], GroupedSplitRecord]]:
    """Reject incomplete rows before they can omit split peers or publish data."""
    if not _checkpoint_record_fields_are_canonical(record):
        return None
    try:
        row_index = _checkpoint_row_index(record.get("row_index"))
        artifact_reference = artifact_reference_from_checkpoint(
            record.get(reference_field)
        )
        reference_identity = _reference_identity(artifact_reference)
        summary_payload = record.get("summary_row")
        grouped_record = grouped_record_from_checkpoint(
            record.get("grouped_split_record")
        )
        if not isinstance(summary_payload, Mapping) or grouped_record is None:
            return None
        references = [
            summary_payload.get("artifact_reference"),
            record.get("target_artifact_reference"),
            grouped_record.artifact_reference,
        ]
        stored_reference = record.get("artifact_reference")
        if stored_reference is not None:
            references.append(stored_reference)
        elif record.get("state") != "terminal_failure":
            return None
        if any(
            _reference_identity(value) != reference_identity for value in references
        ):
            return None
        summary_row = {str(key): str(value) for key, value in summary_payload.items()}
        summary_row["artifact_reference"] = str(artifact_reference)
        summary_row["split"] = ""
        summary_row["split_seed"] = str(split_seed)
        grouped_record = replace(grouped_record, artifact_reference=artifact_reference)
        return row_index, artifact_reference, summary_row, grouped_record
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _durable_result_from_checkpoint(
    record: Mapping[str, object],
    *,
    split_seed: int,
) -> Optional[_PreprocessManifestRowResult]:
    if record.get("state") != "durable_success":
        return None
    summary_payload = record.get("summary_row")
    if (
        not isinstance(summary_payload, Mapping)
        or summary_payload.get("success") != "true"
    ):
        return None
    fields = _validated_recovery_fields(
        record, reference_field="artifact_reference", split_seed=split_seed
    )
    if fields is None:
        return None
    row_index, artifact_reference, summary_row, grouped_record = fields
    return _PreprocessManifestRowResult(
        row_index=row_index,
        artifact_reference=artifact_reference,
        summary_row=summary_row,
        grouped_split_record=grouped_record,
        preprocess_succeeded=True,
        artifact_identity=(
            dict(record["artifact_identity"])
            if isinstance(record.get("artifact_identity"), Mapping)
            else None
        ),
    )


def _pending_result_from_checkpoint(
    record: Mapping[str, object],
    *,
    upload_cache: BatchedArtifactUploadCache,
    artifact_output_dir: Path,
    split_seed: int,
) -> Optional[_PreprocessManifestRowResult]:
    if record.get("state") not in {"upload_pending", "terminal_failure"}:
        return None
    if not checkpoint_cache_matches(record, cache_parent=artifact_output_dir):
        return None
    fields = _validated_recovery_fields(
        record, reference_field="target_artifact_reference", split_seed=split_seed
    )
    if fields is None:
        return None
    row_index, target_artifact_reference, summary_row, grouped_record = fields
    cache_path_value = record.get("cached_artifact_dir")
    success_status = record.get("success_status")
    if (
        not is_remote_uri(target_artifact_reference)
        or not isinstance(cache_path_value, str)
        or not isinstance(success_status, str)
        or not success_status.strip()
    ):
        return None
    summary_row["cached_artifact_dir"] = ""
    summary_row["error"] = ""
    summary_row["success"] = "false"
    summary_row["status"] = ""
    cache_path = Path(cache_path_value)
    upload_future = upload_cache.stage(cache_path, target_artifact_reference)
    return _PreprocessManifestRowResult(
        row_index=row_index,
        artifact_reference=target_artifact_reference,
        summary_row=summary_row,
        grouped_split_record=grouped_record,
        preprocess_succeeded=True,
        upload_future=upload_future,
        cached_artifact_dir=cache_path,
        success_status=success_status,
        artifact_voxel_size=str(record.get("artifact_voxel_size") or ""),
        grid_shape=str(record.get("grid_shape") or ""),
        entry_label=str(record.get("entry_label") or ""),
    )
