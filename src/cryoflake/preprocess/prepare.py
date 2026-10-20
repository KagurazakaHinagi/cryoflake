"""Execute manifest preparation from a validated Hydra configuration."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from cryoflake.config import validate_config
from cryoflake.manifests.download import download_manifest_entries
from cryoflake.manifests.manifest import (
    ManifestEntry,
    build_manifest,
    write_manifest_csv,
)
from cryoflake.preprocess.manifest import preprocess_manifest_csv


def _log_progress(message: str) -> None:
    print(f"[prepare-manifest] {message}", flush=True)


def run(
    config: DictConfig,
    *,
    preprocess_adapter: Callable[..., Any] | None = None,
    sequence_embedding_generator: Callable[..., Any] | None = None,
    progress_callback: Callable[[str], None] | None = _log_progress,
) -> list[ManifestEntry]:
    """Prepare a manifest, optionally downloading and preprocessing its rows.

    An injected adapter is required when preprocessing is enabled. All option
    validation and source-header contract checks finish before output creation.
    Sequence embeddings are read from supplied sidecars, optionally generated
    by an injected adapter after downloads and before preprocessing. Adapters
    own their model dependencies and receive the configured paths and options.
    """
    for name, callback in (
        ("preprocess_adapter", preprocess_adapter),
        ("sequence_embedding_generator", sequence_embedding_generator),
        ("progress_callback", progress_callback),
    ):
        if callback is not None and not callable(callback):
            raise ValueError(f"{name} must be callable")
    validate_config(
        config,
        require_inputs=True,
        preprocess_adapter_provided=preprocess_adapter is not None,
        sequence_embedding_generator_provided=sequence_embedding_generator is not None,
    )
    args = config.pipeline
    if args.name != "manifest":
        raise ValueError("Manifest preparation requires the manifest pipeline")
    if args.artifact_output_dir is not None and preprocess_adapter is None:
        raise ValueError("Raw preprocessing requires an injected preprocessing adapter")
    if args.prepare_sequence_embeddings and sequence_embedding_generator is None:
        raise ValueError(
            "Sequence embedding generation requires an injected "
            "sequence_embedding_generator adapter"
        )
    with Path(args.source_csv).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        headers = [name.strip() for name in reader.fieldnames or []]
        if any(not name for name in headers) or len(set(headers)) != len(headers):
            raise ValueError("Source CSV column names must be unique and nonempty")
        missing = set(config.contracts.manifest.required_source_columns) - set(headers)
        if missing:
            raise ValueError(
                f"Source CSV violates manifest contract; missing columns: {sorted(missing)}"
            )

    def progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    progress(f"Reading map/PDB associations from {args.source_csv}")
    entries = build_manifest(
        args.source_csv,
        include_without_structure=args.include_without_structure,
    )
    progress(f"Resolved {len(entries)} entries; writing {args.manifest_csv}")
    write_manifest_csv(
        entries,
        args.manifest_csv,
        maps_dir=args.maps_dir,
        structures_dir=args.structures_dir,
        sequence_embeddings_dir=args.sequence_embeddings_dir,
        sequence_embedding_suffix=args.sequence_embedding_suffix,
        sequence_embedding_name_template=args.sequence_embedding_name_template,
    )
    download_errors = {}
    if args.download:
        progress(f"Downloading maps and structures for {len(entries)} entries")
        download_errors = download_manifest_entries(
            entries,
            maps_dir=args.maps_dir,
            structures_dir=args.structures_dir,
            force_download=args.force_download,
            progress_callback=progress,
            num_workers=args.download_workers or args.workers,
        )
    if args.prepare_sequence_embeddings:
        progress(f"Generating sequence embeddings into {args.sequence_embeddings_dir}")
        assert sequence_embedding_generator is not None
        sequence_embedding_generator(
            structures_dir=args.structures_dir,
            sequence_embeddings_dir=args.sequence_embeddings_dir,
            device=args.sequence_embedding_device,
            max_chain_length=args.sequence_embedding_max_chain_length,
            num_workers=args.sequence_embedding_workers,
        )
        progress("Sequence embedding generation complete")
    if args.artifact_output_dir is not None:
        artifact_list_path = args.artifact_list_path
        if artifact_list_path is None:
            artifact_list_path = Path(args.artifact_output_dir) / "artifact_list.txt"
        summary_csv = args.summary_csv
        if summary_csv is None:
            output_csv_path = Path(args.manifest_csv)
            summary_csv = output_csv_path.with_name(
                f"{output_csv_path.stem}_summary.csv"
            )
        progress(f"Preprocessing manifest rows into {args.artifact_output_dir}")
        artifact_references = preprocess_manifest_csv(
            args.manifest_csv,
            artifact_output_dir=args.artifact_output_dir,
            artifact_s3_prefix=args.artifact_s3_prefix,
            s3_endpoint_url=config.storage.s3_endpoint_url,
            artifact_reuse_dirs=list(args.artifact_reuse_dirs),
            artifact_list_path=artifact_list_path,
            validation_artifact_list_path=args.validation_artifact_list_path,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
            split_strategy=args.split_strategy,
            holdout_group_ids_path=args.holdout_group_ids_path,
            split_report_path=args.split_report_path,
            target_voxel_size=args.target_voxel_size,
            interface_contact_cutoff=args.interface_contact_cutoff,
            force_preprocess=args.force_preprocess,
            skip_missing=args.skip_missing,
            fix_shift=args.fix_shift,
            require_sequence_embeddings=args.require_sequence_embeddings,
            summary_csv=summary_csv,
            reset_preprocess_checkpoint=args.reset_preprocess_checkpoint,
            continue_on_preprocess_error=not args.fail_fast,
            download_errors=download_errors,
            progress_callback=progress,
            num_workers=args.preprocess_workers or args.workers,
            upload_workers=args.upload_workers or args.workers,
            preprocess_adapter=preprocess_adapter,
        )
        progress(
            f"Prepared {len(artifact_references)} artifacts; artifact list: {artifact_list_path}"
        )
        progress(f"Wrote preprocessing summary to {summary_csv}")
    progress("Manifest preparation completed")
    return entries
