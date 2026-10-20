"""Worker support for manifest preprocessing."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from cryoflake.common import (
    format_gib,
    path_key,
    run_tasks_with_progress,
)
from cryoflake.manifests.manifest import (
    get_preprocessed_artifact_name,
    normalize_emdb_id,
    normalize_wwpdb_id,
)
from cryoflake.manifests.splits import (
    GroupedSplitRecord,
)
from cryoflake.preprocess.memory_guard import (
    estimate_preprocess_memory_or_none,
)
from cryoflake.preprocess.models import (
    _PreprocessManifestRowResult,
    _PreprocessManifestRowTask,
)
from cryoflake.preprocess.reuse import (
    _find_reusable_preprocessed_artifact,
    _materialize_preprocess_input,
    _preprocessed_artifact_is_reusable,
    _resolve_manifest_path,
)
from cryoflake.preprocess.summary import (
    _artifact_grid_shape,
    _artifact_voxel_size,
    _finish_summary_row,
    _format_exception,
    _preprocess_summary_row,
)
from cryoflake.sequence.sidecars import (
    attach_sequence_embeddings_to_preprocessed_artifact,
)
from cryoflake.storage.store import (
    load_preprocessed_artifact,
)
from cryoflake.storage.transports import (
    join_remote_uri,
)


def _preprocess_manifest_row(
    task: _PreprocessManifestRowTask,
) -> _PreprocessManifestRowResult:
    row = task.row
    entry_label = row.get("emdb_accession") or (
        f"EMD-{normalize_emdb_id(row['emdb_id'])}"
    )
    map_path = Path(row["map_path"].strip())
    if not map_path.is_absolute():
        map_path = task.manifest_dir / map_path
    structure_path_value = row.get("structure_path", "").strip()
    structure_path = Path(structure_path_value)
    if structure_path_value and not structure_path.is_absolute():
        structure_path = task.manifest_dir / structure_path
    artifact_name = get_preprocessed_artifact_name(row)
    final_local_artifact_dir = task.artifact_output_dir / artifact_name
    target_artifact_reference: Path | str = final_local_artifact_dir
    if task.artifact_s3_prefix is not None:
        target_artifact_reference = join_remote_uri(
            task.artifact_s3_prefix, artifact_name
        )
    summary_row = _preprocess_summary_row(
        row,
        map_path=map_path,
        structure_path=structure_path if structure_path_value else "",
        artifact_reference=target_artifact_reference,
        split_seed=task.split_seed,
    )

    if not structure_path_value:
        _finish_summary_row(
            summary_row,
            status="skipped_no_structure",
            success=False,
            error="No fitted structure path",
        )
        return _PreprocessManifestRowResult(
            row_index=task.row_index,
            artifact_reference=None,
            summary_row=summary_row,
        )

    row_download_errors = []
    for path in (map_path, structure_path):
        download_error = task.download_errors.get(path_key(path))
        if download_error:
            row_download_errors.append(download_error)
    if row_download_errors:
        _finish_summary_row(
            summary_row,
            status="download_failed",
            success=False,
            error="; ".join(row_download_errors),
        )
        return _PreprocessManifestRowResult(
            row_index=task.row_index,
            artifact_reference=None,
            summary_row=summary_row,
        )

    missing_structure_path = structure_path_value and not structure_path.is_file()
    if missing_structure_path:
        error = f"Manifest row references missing structure file: {structure_path}"
        _finish_summary_row(
            summary_row,
            status="skipped_missing",
            success=False,
            error=error,
        )
        return _PreprocessManifestRowResult(
            row_index=task.row_index,
            artifact_reference=None,
            summary_row=summary_row,
            warning_message=f"Skipping {entry_label}: {error}",
        )

    missing_paths = [str(path) for path in (map_path,) if not path.is_file()]
    if missing_paths:
        if task.skip_missing:
            _finish_summary_row(
                summary_row,
                status="skipped_missing",
                success=False,
                error="Manifest row references missing files: "
                + ", ".join(missing_paths),
            )
            return _PreprocessManifestRowResult(
                row_index=task.row_index,
                artifact_reference=None,
                summary_row=summary_row,
            )
        raise FileNotFoundError(
            "Manifest row references missing files: " + ", ".join(missing_paths)
        )

    warning_message = None
    sequence_embedding_path = None
    sequence_embedding_path_value = row.get("sequence_embedding_path", "").strip()
    sequence_sidecar_configured = bool(sequence_embedding_path_value)
    if sequence_embedding_path_value:
        sequence_embedding_path = _resolve_manifest_path(
            sequence_embedding_path_value, task.manifest_dir
        )
        if not sequence_embedding_path.is_file():
            if task.require_sequence_embeddings:
                raise FileNotFoundError(
                    "Manifest row references missing required sequence "
                    f"embedding file: {sequence_embedding_path}"
                )
            warning_message = (
                f"{entry_label}: referenced sequence embedding file is missing; "
                f"continuing without sequence embeddings: {sequence_embedding_path}"
            )
            sequence_embedding_path = None
    elif task.require_sequence_embeddings:
        raise ValueError(
            "Manifest row is missing sequence_embedding_path while "
            "require_sequence_embeddings is enabled."
        )

    grouped_split_record = None
    cached_artifact_dir = None
    upload_future = None
    memory_request = None
    artifact_mapping = None
    extension_targets = None
    component_targets = None
    try:
        if task.upload_cache is not None:
            task.upload_cache.raise_if_failed()
        memory_estimate = estimate_preprocess_memory_or_none(
            map_path,
            task.target_voxel_size,
        )
        estimated_memory_request = (
            task.memory_admission.limit_bytes
            if memory_estimate is None
            else min(
                memory_estimate.estimated_peak_bytes,
                task.memory_admission.limit_bytes,
            )
        )
        task.memory_admission.acquire(estimated_memory_request)
        memory_request = estimated_memory_request
        reusable_artifact_reference = None
        if not task.force_preprocess:
            if not (sequence_sidecar_configured and sequence_embedding_path is None):
                reusable_output = _find_reusable_preprocessed_artifact(
                    row,
                    target_artifact_reference,
                    task.artifact_reuse_dirs,
                    s3_endpoint_url=task.s3_endpoint_url,
                    expected_interface_contact_cutoff=(task.interface_contact_cutoff),
                    allow_external_reuse=not sequence_sidecar_configured,
                    replace_sequence_embeddings=sequence_embedding_path is not None,
                )
                if reusable_output is not None:
                    reusable_artifact_reference, artifact_mapping = reusable_output
                del reusable_output
        artifact_reference = reusable_artifact_reference or target_artifact_reference
        summary_row["artifact_reference"] = str(artifact_reference)
        should_preprocess = reusable_artifact_reference is None
        uses_external_reuse = (
            reusable_artifact_reference is not None
            and reusable_artifact_reference != target_artifact_reference
        )
        if should_preprocess:
            memory_limit_bytes = task.memory_admission.limit_bytes
            if (
                memory_estimate is not None
                and memory_estimate.estimated_peak_bytes > memory_limit_bytes
            ):
                target_shape = "x".join([str(memory_estimate.output_size)] * 3)
                error = (
                    "Estimated preprocessing memory "
                    f"{format_gib(memory_estimate.estimated_peak_bytes)} "
                    f"exceeds guard limit {format_gib(memory_limit_bytes)} "
                    f"for target grid {target_shape} "
                    f"(input grid "
                    f"{'x'.join(str(axis) for axis in memory_estimate.input_shape)}, "
                    f"voxel size {memory_estimate.input_voxel_size:g}, "
                    f"target voxel size {task.target_voxel_size:g})."
                )
                _finish_summary_row(
                    summary_row,
                    status="skipped_oversized_map",
                    success=False,
                    error=error,
                )
                return _PreprocessManifestRowResult(
                    row_index=task.row_index,
                    artifact_reference=None,
                    summary_row=summary_row,
                    warning_message=f"Skipping {entry_label}: {error}",
                )

            with tempfile.TemporaryDirectory(
                prefix=".preprocess_inputs_", dir=str(task.artifact_output_dir)
            ) as temporary_dir_name:
                temporary_dir = Path(temporary_dir_name)
                adapter_artifact_dir = final_local_artifact_dir
                if task.artifact_s3_prefix is not None:
                    if task.upload_cache is None:
                        raise RuntimeError(
                            "Remote preprocessing requires an upload cache"
                        )
                    cached_artifact_dir = task.upload_cache.allocate_artifact_dir(
                        task.row_index, artifact_name
                    )
                    adapter_artifact_dir = cached_artifact_dir
                preprocess_map_path = _materialize_preprocess_input(
                    map_path, temporary_dir
                )
                preprocess_structure_path = _materialize_preprocess_input(
                    structure_path, temporary_dir
                )
                preprocess_kwargs = dict(
                    map_path=str(preprocess_map_path),
                    structure_path=str(preprocess_structure_path),
                    artifact_reference=str(adapter_artifact_dir),
                    target_voxel_size=task.target_voxel_size,
                    original_structure_path=None,
                    fasta_path=None,
                    fix_shift=task.fix_shift,
                    interface_contact_cutoff=task.interface_contact_cutoff,
                )
                task.preprocess_adapter(**preprocess_kwargs)
                if sequence_embedding_path is not None:
                    attach_sequence_embeddings_to_preprocessed_artifact(
                        adapter_artifact_dir,
                        sequence_embedding_path,
                        sequence_embedding_loader=task.sequence_embedding_loader,
                    )
            inspection_artifact_reference = adapter_artifact_dir
            status = "processed"
        elif uses_external_reuse:
            inspection_artifact_reference = artifact_reference
            status = "reused_external"
        else:
            inspection_artifact_reference = artifact_reference
            status = "reused"
        if (
            sequence_embedding_path is not None
            and not should_preprocess
            and not uses_external_reuse
        ):
            # Release the old core arrays before loading alignment metadata and
            # inspecting the newly attached sequence generation.
            artifact_mapping = None
            attach_sequence_embeddings_to_preprocessed_artifact(
                artifact_reference,
                sequence_embedding_path,
                sequence_embedding_loader=task.sequence_embedding_loader,
                s3_endpoint_url=task.s3_endpoint_url,
            )
        from cryoflake.targets.schema import (
            validate_component_extension_targets,
        )

        if artifact_mapping is None:
            artifact_mapping = load_preprocessed_artifact(
                inspection_artifact_reference,
                s3_endpoint_url=task.s3_endpoint_url,
            )
        if not _preprocessed_artifact_is_reusable(
            artifact_mapping,
            expected_interface_contact_cutoff=task.interface_contact_cutoff,
            require_sequence_embeddings=task.require_sequence_embeddings,
        ):
            raise ValueError(
                "Preprocessed artifact does not satisfy the training contract: "
                f"{inspection_artifact_reference}. Check grid geometry, backbone/residue labels, "
                "full-atom targets, component targets, and any sequence embeddings."
            )
        extension_targets = validate_component_extension_targets(
            artifact_mapping.get("extension_targets")
        )
        component_targets = extension_targets["component"]
        split_groups = component_targets["split_groups"]
        component_group_ids = tuple(split_groups["component"])
        interface_group_ids = tuple(split_groups["interface"])
        summary_row["component_node_count"] = str(len(component_targets["positions"]))
        summary_row["component_interface_count"] = str(
            int(np.asarray(component_targets["interface_mask"]).sum())
        )
        summary_row["component_group_ids"] = ";".join(component_group_ids)
        summary_row["interface_group_ids"] = ";".join(interface_group_ids)
        wwpdb_id = row.get("wwpdb_id", "").strip()
        artifact_group_id = (
            normalize_wwpdb_id(wwpdb_id) if wwpdb_id else str(artifact_reference)
        )
        grouped_split_record = GroupedSplitRecord(
            artifact_reference=artifact_reference,
            artifact_group_id=artifact_group_id,
            component_group_ids=component_group_ids,
            interface_group_ids=interface_group_ids,
        )
        artifact_voxel_size = _artifact_voxel_size(artifact_mapping)
        grid_shape = _artifact_grid_shape(artifact_mapping)
        if cached_artifact_dir is not None:
            upload_future = task.upload_cache.stage(
                cached_artifact_dir, str(artifact_reference)
            )
    except Exception as exc:
        error = _format_exception(exc)
        _finish_summary_row(
            summary_row,
            status="failed",
            success=False,
            error=error,
        )
        if not task.continue_on_preprocess_error:
            raise
        warning_message = f"Skipping {entry_label}: preprocessing failed; {error}"
        return _PreprocessManifestRowResult(
            row_index=task.row_index,
            artifact_reference=None,
            summary_row=summary_row,
            warning_message=warning_message,
        )
    finally:
        # A reservation can admit another large artifact as soon as it is
        # released. Drop both the payload and validation views first, even on
        # errors whose traceback keeps this frame alive.
        artifact_mapping = None
        extension_targets = None
        component_targets = None
        try:
            # Before stage() returns, this worker still owns its cache row.
            # Interrupts such as KeyboardInterrupt must clean it up as well.
            if cached_artifact_dir is not None and upload_future is None:
                task.upload_cache.discard(cached_artifact_dir)
        finally:
            if memory_request is not None:
                task.memory_admission.release(memory_request)

    if upload_future is not None:
        return _PreprocessManifestRowResult(
            row_index=task.row_index,
            artifact_reference=artifact_reference,
            summary_row=summary_row,
            warning_message=warning_message,
            grouped_split_record=grouped_split_record,
            preprocess_succeeded=True,
            upload_future=upload_future,
            cached_artifact_dir=cached_artifact_dir,
            success_status=status,
            artifact_voxel_size=artifact_voxel_size,
            grid_shape=grid_shape,
            entry_label=entry_label,
        )

    _finish_summary_row(
        summary_row,
        status=status,
        success=True,
        artifact_voxel_size=artifact_voxel_size,
        grid_shape=grid_shape,
    )
    return _PreprocessManifestRowResult(
        row_index=task.row_index,
        artifact_reference=artifact_reference,
        summary_row=summary_row,
        warning_message=warning_message,
        grouped_split_record=grouped_split_record,
        preprocess_succeeded=True,
    )


def _run_preprocess_manifest_row_tasks(
    tasks: list[_PreprocessManifestRowTask],
    num_workers: int,
    *,
    stop_event: Optional[threading.Event] = None,
    result_callback: Optional[Callable[[_PreprocessManifestRowResult], None]] = None,
) -> list[_PreprocessManifestRowResult]:
    return run_tasks_with_progress(
        tasks,
        _preprocess_manifest_row,
        num_workers=num_workers,
        progress_description="Preprocessing manifest rows",
        success_predicate=lambda result: result.preprocess_succeeded,
        wait_for_workers_on_error=True,
        stop_event=stop_event,
        result_callback=result_callback,
    )
