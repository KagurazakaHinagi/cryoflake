"""Results support for manifest preprocessing."""

from __future__ import annotations

import os
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

from cryoflake.manifests.manifest import (
    split_artifact_references,
    write_artifact_list,
)
from cryoflake.manifests.splits import (
    read_holdout_group_ids,
    split_grouped_records,
    write_grouped_split_report,
)
from cryoflake.preprocess.models import _PreprocessManifestRowResult
from cryoflake.preprocess.summary import (
    _assign_summary_splits,
    _finish_summary_row,
    _format_exception,
    write_preprocess_summary_csv,
)


def _resolve_pending_uploads(
    results: list[_PreprocessManifestRowResult],
    continue_on_preprocess_error: bool,
) -> list[_PreprocessManifestRowResult]:
    resolved_results = []
    first_upload_error = None
    for result in results:
        if result.upload_future is None:
            resolved_results.append(result)
            continue
        try:
            cleanup_warning = result.upload_future.result()
        except Exception as exc:
            error = _format_exception(exc)
            result.summary_row["cached_artifact_dir"] = str(result.cached_artifact_dir)
            _finish_summary_row(
                result.summary_row,
                status="upload_failed",
                success=False,
                error=error,
                artifact_voxel_size=result.artifact_voxel_size,
                grid_shape=result.grid_shape,
            )
            warning_message = (
                f"Skipping {result.entry_label}: remote upload failed; {error}"
            )
            if result.warning_message is not None:
                warning_message = f"{result.warning_message}; {warning_message}"
            resolved_results.append(
                replace(
                    result,
                    artifact_reference=None,
                    warning_message=warning_message,
                    preprocess_succeeded=False,
                    upload_future=None,
                )
            )
            if first_upload_error is None:
                first_upload_error = exc
            continue

        _finish_summary_row(
            result.summary_row,
            status=result.success_status,
            success=True,
            artifact_voxel_size=result.artifact_voxel_size,
            grid_shape=result.grid_shape,
        )
        warning_message = result.warning_message
        if cleanup_warning is not None:
            result.summary_row["cached_artifact_dir"] = str(result.cached_artifact_dir)
            cleanup_warning = f"{result.entry_label}: {cleanup_warning}"
            if warning_message is None:
                warning_message = cleanup_warning
            else:
                warning_message = f"{warning_message}; {cleanup_warning}"
        resolved_results.append(
            replace(
                result,
                warning_message=warning_message,
                upload_future=None,
                cached_artifact_dir=(
                    result.cached_artifact_dir if cleanup_warning is not None else None
                ),
            )
        )

    if first_upload_error is not None and not continue_on_preprocess_error:
        raise first_upload_error
    return resolved_results


def _finalize_preprocess_manifest_results(
    results: list[_PreprocessManifestRowResult],
    *,
    artifact_list_path: Optional[os.PathLike],
    validation_artifact_list_path: Optional[os.PathLike],
    validation_fraction: float,
    split_seed: int,
    split_strategy: str,
    holdout_group_ids_path: Optional[os.PathLike],
    split_report_path: Optional[os.PathLike],
    summary_csv: Optional[os.PathLike],
    progress_callback: Optional[Callable[[str], None]],
) -> list[Path | str]:
    artifact_references = []
    summary_rows = []
    grouped_split_records = []
    for result in results:
        if result.warning_message is not None:
            warnings.warn(result.warning_message, RuntimeWarning, stacklevel=3)
            if progress_callback is not None:
                progress_callback(f"Warning: {result.warning_message}")
        if result.artifact_reference is not None:
            artifact_references.append(result.artifact_reference)
        if (
            result.artifact_reference is not None
            and result.grouped_split_record is not None
        ):
            grouped_split_records.append(result.grouped_split_record)
        summary_rows.append(result.summary_row)

    artifact_reference_splits: dict[Path | str, set[str]] = {}
    if artifact_list_path is not None:
        if validation_artifact_list_path is None:
            write_artifact_list(artifact_references, artifact_list_path)
            for artifact_reference in artifact_references:
                artifact_reference_splits.setdefault(artifact_reference, set()).add(
                    "train"
                )
        elif split_strategy != "artifact":
            holdout_group_ids = (
                None
                if holdout_group_ids_path is None
                else read_holdout_group_ids(
                    holdout_group_ids_path,
                    expected_split_strategy=split_strategy,
                )
            )
            split_result = split_grouped_records(
                grouped_split_records,
                split_strategy=split_strategy,
                validation_fraction=validation_fraction,
                seed=split_seed,
                holdout_group_ids=holdout_group_ids,
            )
            train_artifact_references = list(split_result.train_artifact_references)
            validation_artifact_references = list(
                split_result.validation_artifact_references
            )
            write_artifact_list(train_artifact_references, artifact_list_path)
            write_artifact_list(
                validation_artifact_references, validation_artifact_list_path
            )
            if split_report_path is not None:
                write_grouped_split_report(split_result, split_report_path)
            for artifact_reference in train_artifact_references:
                artifact_reference_splits.setdefault(artifact_reference, set()).add(
                    "train"
                )
            for artifact_reference in validation_artifact_references:
                artifact_reference_splits.setdefault(artifact_reference, set()).add(
                    "valid"
                )
        else:
            train_artifact_references, validation_artifact_references = (
                split_artifact_references(
                    artifact_references,
                    validation_fraction=validation_fraction,
                    seed=split_seed,
                )
            )
            write_artifact_list(train_artifact_references, artifact_list_path)
            write_artifact_list(
                validation_artifact_references, validation_artifact_list_path
            )
            for artifact_reference in train_artifact_references:
                artifact_reference_splits.setdefault(artifact_reference, set()).add(
                    "train"
                )
            for artifact_reference in validation_artifact_references:
                artifact_reference_splits.setdefault(artifact_reference, set()).add(
                    "valid"
                )
    elif validation_artifact_list_path is not None:
        raise ValueError(
            "validation_artifact_list_path requires artifact_list_path so both split "
            "outputs can be written."
        )

    if summary_csv is not None:
        _assign_summary_splits(summary_rows, artifact_reference_splits)
        write_preprocess_summary_csv(summary_rows, summary_csv)
    return artifact_references
