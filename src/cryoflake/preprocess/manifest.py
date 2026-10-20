"""Manifest support for manifest preprocessing."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

import numpy as np

from cryoflake.common import (
    paths_must_be_distinct,
    validate_num_workers,
)
from cryoflake.manifests.manifest import (
    get_preprocessed_artifact_name,
    read_manifest_csv,
)
from cryoflake.preprocess.artifact_identity import (
    preprocess_checkpoint_contract,
    preprocessed_artifact_identity,
)
from cryoflake.preprocess.checkpoint import (
    PreprocessCheckpoint,
    checkpoint_record_for_result,
)
from cryoflake.preprocess.memory_guard import (
    PreprocessMemoryAdmission,
    preprocess_memory_guard_limit_bytes,
)
from cryoflake.preprocess.models import (
    _PreprocessManifestRowResult,
    _PreprocessManifestRowTask,
)
from cryoflake.preprocess.recovery import (
    _durable_result_from_checkpoint,
    _pending_result_from_checkpoint,
)
from cryoflake.preprocess.results import (
    _finalize_preprocess_manifest_results,
    _resolve_pending_uploads,
)
from cryoflake.preprocess.worker import _run_preprocess_manifest_row_tasks
from cryoflake.sequence.sidecars import (
    load_sequence_embedding_file,
)
from cryoflake.storage.store import (
    upload_preprocessed_artifact,
)
from cryoflake.storage.transports import (
    is_remote_uri,
    validate_remote_configuration,
)
from cryoflake.storage.upload_cache import BatchedArtifactUploadCache


def _validate_manifest_file_destinations(
    manifest_rows: list[dict[str, str]],
    *,
    manifest_dir: Path,
    output_files: Mapping[str, os.PathLike | str | None],
    artifact_roots: Iterable[Path],
) -> None:
    resolved_outputs = {
        name: Path(value).resolve()
        for name, value in output_files.items()
        if value is not None
    }
    resolved_artifact_roots = tuple(path.resolve() for path in artifact_roots)
    for name, artifact_reference in resolved_outputs.items():
        if any(
            artifact_reference.is_relative_to(root) for root in resolved_artifact_roots
        ):
            raise ValueError(
                f"{name} cannot be inside a preprocessed artifact directory: "
                f"{artifact_reference}"
            )
    for row_index, row in enumerate(manifest_rows, start=1):
        for fieldname in ("map_path", "structure_path", "sequence_embedding_path"):
            source_value = row.get(fieldname, "").strip()
            if not source_value:
                continue
            source_path = Path(source_value)
            if not source_path.is_absolute():
                source_path = manifest_dir / source_path
            for name, artifact_reference in resolved_outputs.items():
                paths_must_be_distinct(
                    {
                        f"row {row_index} {fieldname}": source_path,
                        name: artifact_reference,
                    }
                )


def preprocess_manifest_csv(
    manifest_csv: os.PathLike,
    artifact_output_dir: os.PathLike,
    artifact_s3_prefix: Optional[str] = None,
    s3_endpoint_url: Optional[str] = None,
    artifact_reuse_dirs: Optional[Iterable[os.PathLike]] = None,
    artifact_list_path: Optional[os.PathLike] = None,
    validation_artifact_list_path: Optional[os.PathLike] = None,
    validation_fraction: float = 0.1,
    split_seed: int = 0,
    target_voxel_size: float = 1.0,
    force_preprocess: bool = False,
    skip_missing: bool = False,
    fix_shift: bool = True,
    preprocess_adapter: Optional[Callable[..., None]] = None,
    require_sequence_embeddings: bool = False,
    sequence_embedding_loader: Callable[[os.PathLike], Mapping[str, np.ndarray]] = (
        load_sequence_embedding_file
    ),
    summary_csv: Optional[os.PathLike] = None,
    continue_on_preprocess_error: bool = True,
    download_errors: Optional[Mapping[str, str]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    num_workers: int = 1,
    split_strategy: str = "artifact",
    holdout_group_ids_path: Optional[os.PathLike] = None,
    split_report_path: Optional[os.PathLike] = None,
    interface_contact_cutoff: float = 5.0,
    upload_workers: Optional[int] = None,
    reset_preprocess_checkpoint: bool = False,
    preprocess_checkpoint_token: Optional[str] = None,
) -> list[Path | str]:
    validation_fraction = float(validation_fraction)
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be greater than 0 and less than 1")
    target_voxel_size = float(target_voxel_size)
    if not np.isfinite(target_voxel_size) or target_voxel_size <= 0:
        raise ValueError("target_voxel_size must be a positive finite number")
    interface_contact_cutoff = float(interface_contact_cutoff)
    if not np.isfinite(interface_contact_cutoff) or interface_contact_cutoff <= 0:
        raise ValueError("interface_contact_cutoff must be a positive finite number")
    if split_strategy not in {"artifact", "component", "interface"}:
        raise ValueError("split_strategy must be artifact, component, or interface")
    if split_strategy != "artifact" and validation_artifact_list_path is None:
        raise ValueError(
            "component/interface validation splitting requires validation_artifact_list_path"
        )
    if holdout_group_ids_path is not None and split_strategy == "artifact":
        raise ValueError(
            "holdout_group_ids_path requires component or interface splitting"
        )
    if split_report_path is not None and split_strategy == "artifact":
        raise ValueError("split_report_path requires component or interface splitting")
    if validation_artifact_list_path is not None and artifact_list_path is None:
        raise ValueError("validation_artifact_list_path requires artifact_list_path")
    num_workers = validate_num_workers(num_workers, "num_workers")
    if upload_workers is None:
        upload_workers = num_workers
    upload_workers = validate_num_workers(upload_workers, "upload_workers")

    if preprocess_adapter is None:
        raise ValueError(
            "preprocess_adapter is required for raw map/structure preprocessing. "
            "Pass a callable or select an adapter with the CLI --preprocess-adapter "
            "module:callable option."
        )
    for name, callback in (
        ("preprocess_adapter", preprocess_adapter),
        ("sequence_embedding_loader", sequence_embedding_loader),
        ("progress_callback", progress_callback),
    ):
        if not callable(callback) and (
            name != "progress_callback" or callback is not None
        ):
            raise ValueError(f"{name} must be callable")
    file_paths = {
        "manifest_csv": manifest_csv,
        "holdout_group_ids_path": holdout_group_ids_path,
        "artifact_list_path": artifact_list_path,
        "validation_artifact_list_path": validation_artifact_list_path,
        "summary_csv": summary_csv,
        "split_report_path": split_report_path,
    }
    if summary_csv is not None:
        for suffix in ("checkpoint.json", "checkpoint.jsonl", "checkpoint.lock"):
            file_paths[f"summary_csv.{suffix}"] = f"{os.fspath(summary_csv)}.{suffix}"
    paths_must_be_distinct(file_paths)

    manifest_csv = Path(manifest_csv)
    manifest_dir = manifest_csv.parent
    artifact_output_dir = Path(artifact_output_dir)
    if artifact_s3_prefix is not None and not is_remote_uri(artifact_s3_prefix):
        raise ValueError("artifact_s3_prefix must be an s3:// URI")
    artifact_reuse_dirs = tuple(
        Path(artifact_reuse_dir) for artifact_reuse_dir in (artifact_reuse_dirs or ())
    )

    download_errors = dict(download_errors or {})
    manifest_rows = read_manifest_csv(manifest_csv)
    artifact_rows: dict[str, list[int]] = {}
    for row_index, row in enumerate(manifest_rows, start=1):
        artifact_name = get_preprocessed_artifact_name(row)
        artifact_rows.setdefault(artifact_name, []).append(row_index)
    duplicate_artifacts = {
        artifact_name: row_indices
        for artifact_name, row_indices in artifact_rows.items()
        if len(row_indices) > 1
    }
    if duplicate_artifacts:
        duplicate_details = ", ".join(
            f"{artifact_name} (rows {', '.join(map(str, row_indices))})"
            for artifact_name, row_indices in sorted(duplicate_artifacts.items())
        )
        raise ValueError(
            "Manifest rows resolve to duplicate preprocessed artifact roots: "
            f"{duplicate_details}"
        )
    _validate_manifest_file_destinations(
        manifest_rows,
        manifest_dir=manifest_dir,
        output_files={
            name: value
            for name, value in file_paths.items()
            if name not in {"manifest_csv", "holdout_group_ids_path"}
        },
        artifact_roots=(
            (artifact_output_dir / name for name in artifact_rows)
            if artifact_s3_prefix is None
            else ()
        ),
    )
    if artifact_s3_prefix is not None:
        validate_remote_configuration(
            artifact_s3_prefix, s3_endpoint_url=s3_endpoint_url
        )
    total_rows = len(manifest_rows)
    artifact_output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_stop_event = threading.Event()
    memory_admission = PreprocessMemoryAdmission(
        preprocess_memory_guard_limit_bytes(),
        cancel_event=preprocess_stop_event,
    )
    checkpoint = None
    if summary_csv is not None:
        checkpoint = PreprocessCheckpoint(
            summary_csv,
            preprocess_checkpoint_contract(
                manifest_csv=manifest_csv,
                manifest_rows=manifest_rows,
                manifest_dir=manifest_dir,
                artifact_output_dir=artifact_output_dir,
                artifact_s3_prefix=artifact_s3_prefix,
                s3_endpoint_url=s3_endpoint_url,
                artifact_reuse_dirs=artifact_reuse_dirs,
                target_voxel_size=target_voxel_size,
                fix_shift=fix_shift,
                preprocess_adapter=preprocess_adapter,
                interface_contact_cutoff=interface_contact_cutoff,
                require_sequence_embeddings=require_sequence_embeddings,
                sequence_embedding_loader=sequence_embedding_loader,
                download_errors=download_errors,
                preprocess_checkpoint_token=preprocess_checkpoint_token,
            ),
            reset=force_preprocess or reset_preprocess_checkpoint,
        )
    upload_cache = None
    try:
        if artifact_s3_prefix is not None:
            upload_cache = BatchedArtifactUploadCache(
                artifact_output_dir,
                lambda local_path, remote_uri: upload_preprocessed_artifact(
                    local_path,
                    remote_uri,
                    s3_endpoint_url=s3_endpoint_url,
                ),
                num_workers=upload_workers,
                stop_on_failure=not continue_on_preprocess_error,
            )
    except BaseException:
        if checkpoint is not None:
            checkpoint.close()
        raise

    def close_resources() -> None:
        try:
            if upload_cache is not None:
                upload_cache.close()
        finally:
            if checkpoint is not None:
                checkpoint.close()

    resumed_results: list[_PreprocessManifestRowResult] = []
    resumed_row_indices = set()
    artifact_identities: dict[int, dict[str, object]] = {}
    artifact_identity_lock = threading.Lock()
    try:
        if checkpoint is not None and not force_preprocess:
            for row_index, record in sorted(checkpoint.records.items()):
                if row_index > total_rows:
                    continue
                durable_result = _durable_result_from_checkpoint(
                    record,
                    split_seed=split_seed,
                )
                stored_identity = record.get("artifact_identity")
                if durable_result is not None and isinstance(
                    stored_identity,
                    Mapping,
                ):
                    current_identity = preprocessed_artifact_identity(
                        durable_result.artifact_reference,
                        s3_endpoint_url=s3_endpoint_url,
                    )
                    if current_identity == dict(stored_identity):
                        resumed_results.append(durable_result)
                        resumed_row_indices.add(row_index)
                        artifact_identities[row_index] = current_identity
                        continue

                if upload_cache is not None:
                    pending_result = _pending_result_from_checkpoint(
                        record,
                        upload_cache=upload_cache,
                        artifact_output_dir=artifact_output_dir,
                        split_seed=split_seed,
                    )
                    if pending_result is not None:
                        resumed_results.append(pending_result)
                        resumed_row_indices.add(row_index)
    except BaseException:
        close_resources()
        raise

    try:
        if resumed_row_indices and progress_callback is not None:
            progress_callback(
                "Resuming "
                f"{len(resumed_row_indices)}/{total_rows} rows from "
                f"{checkpoint.journal_path}"
            )
    except BaseException:
        close_resources()
        raise

    try:
        tasks = [
            _PreprocessManifestRowTask(
                row_index=row_index,
                row=row,
                manifest_dir=manifest_dir,
                artifact_output_dir=artifact_output_dir,
                artifact_s3_prefix=artifact_s3_prefix,
                s3_endpoint_url=s3_endpoint_url,
                artifact_reuse_dirs=artifact_reuse_dirs,
                target_voxel_size=target_voxel_size,
                force_preprocess=force_preprocess,
                skip_missing=skip_missing,
                fix_shift=fix_shift,
                preprocess_adapter=preprocess_adapter,
                interface_contact_cutoff=interface_contact_cutoff,
                require_sequence_embeddings=require_sequence_embeddings,
                sequence_embedding_loader=sequence_embedding_loader,
                continue_on_preprocess_error=continue_on_preprocess_error,
                download_errors=download_errors,
                split_seed=split_seed,
                upload_cache=upload_cache,
                memory_admission=memory_admission,
            )
            for row_index, row in enumerate(manifest_rows, start=1)
            if row_index not in resumed_row_indices
        ]
    except BaseException:
        close_resources()
        raise

    def checkpoint_result(result: _PreprocessManifestRowResult) -> None:
        if checkpoint is None:
            return
        identity = result.artifact_identity
        if (
            identity is None
            and result.upload_future is None
            and result.artifact_reference is not None
            and result.summary_row.get("success") == "true"
        ):
            identity = preprocessed_artifact_identity(
                result.artifact_reference,
                s3_endpoint_url=s3_endpoint_url,
            )
        if identity is not None:
            with artifact_identity_lock:
                artifact_identities[result.row_index] = identity
        checkpoint.record(
            checkpoint_record_for_result(
                result,
                artifact_identity=identity,
            )
        )
        if result.upload_future is not None:

            def record_completed_upload(_future) -> None:
                try:
                    resolved_result = _resolve_pending_uploads(
                        [result],
                        continue_on_preprocess_error=True,
                    )[0]
                    resolved_identity = None
                    if resolved_result.artifact_reference is not None:
                        resolved_identity = preprocessed_artifact_identity(
                            resolved_result.artifact_reference,
                            s3_endpoint_url=s3_endpoint_url,
                        )
                        if resolved_identity is not None:
                            with artifact_identity_lock:
                                artifact_identities[resolved_result.row_index] = (
                                    resolved_identity
                                )
                    checkpoint.record_from_callback(
                        checkpoint_record_for_result(
                            resolved_result,
                            artifact_identity=resolved_identity,
                        )
                    )
                except BaseException as exc:
                    checkpoint.fail_from_callback(exc)

            result.upload_future.add_done_callback(record_completed_upload)

    try:
        for resumed_result in resumed_results:
            if resumed_result.upload_future is not None:
                checkpoint_result(resumed_result)
        try:
            new_results = _run_preprocess_manifest_row_tasks(
                tasks,
                num_workers=num_workers,
                stop_event=preprocess_stop_event,
                result_callback=checkpoint_result,
            )
        finally:
            if upload_cache is not None:
                upload_cache.close()
    except BaseException:
        close_resources()
        raise

    results = resumed_results + new_results
    try:
        results = _resolve_pending_uploads(
            results,
            continue_on_preprocess_error=continue_on_preprocess_error,
        )
    except BaseException:
        close_resources()
        raise
    results.sort(key=lambda result: result.row_index)
    if len(results) != total_rows or [result.row_index for result in results] != list(
        range(1, total_rows + 1)
    ):
        close_resources()
        raise RuntimeError(
            "Preprocessing did not produce exactly one result for every manifest row"
        )
    if checkpoint is not None:
        try:
            checkpoint.raise_if_failed()
            checkpoint_records = checkpoint.records
            for result in results:
                with artifact_identity_lock:
                    identity = artifact_identities.get(result.row_index)
                if (
                    identity is None
                    and result.artifact_reference is not None
                    and result.summary_row.get("success") == "true"
                ):
                    identity = preprocessed_artifact_identity(
                        result.artifact_reference,
                        s3_endpoint_url=s3_endpoint_url,
                    )
                checkpoint_record = checkpoint_record_for_result(
                    result,
                    artifact_identity=identity,
                )
                existing_record = checkpoint_records.get(result.row_index)
                if existing_record is not None:
                    existing_record = dict(existing_record)
                    existing_record.pop("checkpoint_generation", None)
                if existing_record == checkpoint_record:
                    continue
                checkpoint.record(checkpoint_record)
                checkpoint_records[result.row_index] = checkpoint_record
        except BaseException:
            close_resources()
            raise
    try:
        artifact_references = _finalize_preprocess_manifest_results(
            results,
            artifact_list_path=artifact_list_path,
            validation_artifact_list_path=validation_artifact_list_path,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
            split_strategy=split_strategy,
            holdout_group_ids_path=holdout_group_ids_path,
            split_report_path=split_report_path,
            summary_csv=summary_csv,
            progress_callback=progress_callback,
        )
    except BaseException:
        close_resources()
        raise

    if checkpoint is not None:
        if all(result.summary_row.get("success") == "true" for result in results):
            checkpoint.complete()
        else:
            checkpoint.close()

    return artifact_references
