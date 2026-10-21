"""Typer commands backed by strict, composable Hydra configurations."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from cryoflake.config import compose_config, validate_config

app = typer.Typer(
    help="Prepare and validate portable cryo-EM training data.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

ConfigDirectory = Annotated[
    Path | None,
    typer.Option("--config-dir", help="Custom Hydra configuration directory."),
]
ConfigName = Annotated[
    str,
    typer.Option("--config-name", help="Hydra root YAML name without its extension."),
]
Overrides = Annotated[
    list[str] | None,
    typer.Option(
        "--override",
        "-o",
        help="Hydra override KEY=VALUE; repeat for multiple overrides.",
    ),
]
Adapter = Annotated[
    str | None,
    typer.Option(
        "--preprocess-adapter",
        metavar="default|MODULE:CALLABLE",
        help="Raw preprocessing adapter: built-in default or an importable callable.",
    ),
]


def _load_adapter(
    specification: str, *, label: str = "preprocess_adapter"
) -> Callable[..., Any]:
    module_name, separator, attribute_path = specification.partition(":")
    if not separator or not module_name.strip() or not attribute_path.strip():
        raise ValueError(f"{label} must use the MODULE:CALLABLE form")
    try:
        value: Any = importlib.import_module(module_name.strip())
        for name in attribute_path.strip().split("."):
            if not name:
                raise ValueError(f"{label} must use the MODULE:CALLABLE form")
            value = getattr(value, name)
    except Exception as exc:
        raise ValueError(f"Cannot load {label} {specification!r}: {exc}") from exc
    if not callable(value):
        raise ValueError(f"{label.capitalize()} {specification!r} is not callable")
    return value


def _configuration(
    pipeline: str | None,
    config_dir: Path | None,
    config_name: str,
    overrides: list[str] | None,
    values: dict[str, Any],
    *,
    s3_endpoint_url: str | None = None,
    require_inputs: bool = True,
) -> DictConfig:
    try:
        explicit: dict[str, Any] = {
            "pipeline": {
                name: value for name, value in values.items() if value is not None
            },
        }
        if s3_endpoint_url is not None:
            explicit["storage"] = {"s3_endpoint_url": s3_endpoint_url}
        composition_overrides = list(overrides or ())
        selected_pipeline = pipeline
        if not require_inputs and pipeline is not None:
            # This is an explicit inspection flag, so it wins over -o selections.
            composition_overrides.append(f"pipeline={pipeline}")
            selected_pipeline = None
        elif require_inputs and (config_dir is not None or config_name != "config"):
            # An external root chooses its own group; validate its entry-point
            # kind below rather than silently replacing those configured values.
            selected_pipeline = None
        config = compose_config(
            composition_overrides,
            config_dir=config_dir,
            config_name=config_name,
            pipeline=selected_pipeline,
            values=explicit,
        )
        if require_inputs and config.pipeline.name != pipeline:
            raise ValueError(
                f"This command requires pipeline.name={pipeline}; "
                f"the selected configuration has pipeline.name={config.pipeline.name}"
            )
        validate_config(config, require_inputs=require_inputs)
    except (ValueError, OmegaConfBaseException) as exc:
        raise typer.BadParameter(str(exc)) from exc
    return config


def _adapter(
    specification: str | None, *, option: str = "--preprocess-adapter"
) -> Callable[..., Any] | None:
    if specification is None:
        return None
    if option == "--preprocess-adapter" and specification == "default":
        from cryoflake.adapters.preprocess.default import run

        return run
    try:
        return _load_adapter(specification, label=option.removeprefix("--"))
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint=option) from exc


def _runtime_failure(exc: Exception) -> None:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(1) from exc


@app.command("config")
def show_config(
    pipeline: Annotated[
        str | None,
        typer.Option(
            help="Pipeline group; defaults to the selected root configuration."
        ),
    ] = None,
    config_dir: ConfigDirectory = None,
    config_name: ConfigName = "config",
    override: Overrides = None,
) -> None:
    """Print the resolved pipeline and runtime data contracts as YAML."""
    config = _configuration(
        pipeline, config_dir, config_name, override, {}, require_inputs=False
    )
    typer.echo(OmegaConf.to_yaml(config))


@app.command("validate-artifact")
def validate_artifact(
    artifact_reference: Annotated[
        str | None, typer.Argument(help="Local artifact root or s3://bucket/key.")
    ] = None,
    require_sequence_embeddings: Annotated[
        bool | None,
        typer.Option("--require-sequence-embeddings/--no-require-sequence-embeddings"),
    ] = None,
    s3_endpoint_url: str | None = None,
    config_dir: ConfigDirectory = None,
    config_name: ConfigName = "config",
    override: Overrides = None,
) -> None:
    """Validate storage integrity and the configured artifact contract."""
    config = _configuration(
        "validate",
        config_dir,
        config_name,
        override,
        {
            "artifact_reference": artifact_reference,
            "require_sequence_embeddings": require_sequence_embeddings,
        },
        s3_endpoint_url=s3_endpoint_url,
    )
    from cryoflake.storage.artifacts import validate_preprocessed_artifact

    try:
        result = validate_preprocessed_artifact(
            config.pipeline.artifact_reference,
            require_sequence_embeddings=config.pipeline.require_sequence_embeddings,
            s3_endpoint_url=config.storage.s3_endpoint_url,
        )
    except Exception as exc:
        _runtime_failure(exc)
    typer.echo(result.to_json())


@app.command("preprocess")
def preprocess(
    map_path: Annotated[str | None, typer.Argument(help="Density map path.")] = None,
    structure_path: Annotated[
        str | None, typer.Argument(help="Structure file path.")
    ] = None,
    artifact_reference: Annotated[
        str | None, typer.Argument(help="Local or S3 artifact root.")
    ] = None,
    preprocess_adapter: Adapter = None,
    original_structure_path: str | None = None,
    fasta_path: str | None = None,
    target_voxel_size: float | None = None,
    interface_contact_cutoff: float | None = None,
    fix_shift: Annotated[
        bool | None,
        typer.Option(
            "--fix-shift/--no-fix-shift", help="Correct the map origin shift."
        ),
    ] = None,
    s3_endpoint_url: str | None = None,
    config_dir: ConfigDirectory = None,
    config_name: ConfigName = "config",
    override: Overrides = None,
) -> None:
    """Run an explicitly configured adapter for one map and structure."""
    config = _configuration(
        "preprocess",
        config_dir,
        config_name,
        override,
        {
            "map_path": map_path,
            "structure_path": structure_path,
            "artifact_reference": artifact_reference,
            "preprocess_adapter": preprocess_adapter,
            "original_structure_path": original_structure_path,
            "fasta_path": fasta_path,
            "target_voxel_size": target_voxel_size,
            "interface_contact_cutoff": interface_contact_cutoff,
            "fix_shift": fix_shift,
        },
        s3_endpoint_url=s3_endpoint_url,
    )
    adapter = _adapter(config.pipeline.preprocess_adapter)
    p = config.pipeline
    try:
        assert adapter is not None
        adapter(
            map_path=p.map_path,
            structure_path=p.structure_path,
            artifact_reference=p.artifact_reference,
            target_voxel_size=p.target_voxel_size,
            original_structure_path=p.original_structure_path,
            fasta_path=p.fasta_path,
            fix_shift=p.fix_shift,
            s3_endpoint_url=config.storage.s3_endpoint_url,
            interface_contact_cutoff=p.interface_contact_cutoff,
        )
        from cryoflake.storage.artifacts import validate_preprocessed_artifact

        validate_preprocessed_artifact(
            p.artifact_reference,
            require_sequence_embeddings=p.require_sequence_embeddings,
            s3_endpoint_url=config.storage.s3_endpoint_url,
        )
    except Exception as exc:
        _runtime_failure(exc)


@app.command("prepare-manifest")
def prepare_manifest(
    source_csv: Annotated[str | None, typer.Option("--source-csv")] = None,
    manifest_csv: str | None = None,
    maps_dir: str | None = None,
    structures_dir: str | None = None,
    sequence_embeddings_dir: str | None = None,
    sequence_embedding_suffix: str | None = None,
    sequence_embedding_name_template: str | None = None,
    require_sequence_embeddings: Annotated[
        bool | None,
        typer.Option("--require-sequence-embeddings/--no-require-sequence-embeddings"),
    ] = None,
    prepare_sequence_embeddings: Annotated[
        bool | None,
        typer.Option("--prepare-sequence-embeddings/--no-prepare-sequence-embeddings"),
    ] = None,
    sequence_embedding_generator: Annotated[
        str | None,
        typer.Option(
            "--sequence-embedding-generator",
            metavar="MODULE:CALLABLE",
            help="Adapter that generates sidecars before preprocessing.",
        ),
    ] = None,
    sequence_embedding_device: str | None = None,
    sequence_embedding_workers: int | None = None,
    sequence_embedding_max_chain_length: int | None = None,
    include_without_structure: Annotated[
        bool | None,
        typer.Option("--include-without-structure/--exclude-without-structure"),
    ] = None,
    download: Annotated[bool | None, typer.Option("--download/--no-download")] = None,
    force_download: Annotated[
        bool | None, typer.Option("--force-download/--no-force-download")
    ] = None,
    workers: int | None = None,
    download_workers: int | None = None,
    artifact_output_dir: str | None = None,
    artifact_s3_prefix: str | None = None,
    s3_endpoint_url: str | None = None,
    artifact_reuse_dir: Annotated[
        list[str] | None, typer.Option("--artifact-reuse-dir")
    ] = None,
    preprocess_workers: int | None = None,
    upload_workers: int | None = None,
    summary_csv: str | None = None,
    reset_preprocess_checkpoint: Annotated[
        bool | None,
        typer.Option("--reset-preprocess-checkpoint/--no-reset-preprocess-checkpoint"),
    ] = None,
    artifact_list_path: str | None = None,
    validation_artifact_list_path: str | None = None,
    validation_fraction: float | None = None,
    split_seed: int | None = None,
    split_strategy: str | None = None,
    holdout_group_ids_path: str | None = None,
    split_report_path: str | None = None,
    target_voxel_size: float | None = None,
    interface_contact_cutoff: float | None = None,
    force_preprocess: Annotated[
        bool | None, typer.Option("--force-preprocess/--no-force-preprocess")
    ] = None,
    skip_missing: Annotated[
        bool | None, typer.Option("--skip-missing/--no-skip-missing")
    ] = None,
    fail_fast: Annotated[
        bool | None, typer.Option("--fail-fast/--no-fail-fast")
    ] = None,
    fix_shift: Annotated[
        bool | None,
        typer.Option(
            "--fix-shift/--no-fix-shift", help="Correct the map origin shift."
        ),
    ] = None,
    preprocess_adapter: Adapter = None,
    config_dir: ConfigDirectory = None,
    config_name: ConfigName = "config",
    override: Overrides = None,
) -> None:
    """Build a manifest and optionally download or preprocess its entries.

    Explicit flags override YAML values. Use --override for any Hydra field.
    Sequence embeddings use sidecars, optionally generated by an explicit adapter.
    """
    values = {
        "source_csv": source_csv,
        "manifest_csv": manifest_csv,
        "maps_dir": maps_dir,
        "structures_dir": structures_dir,
        "sequence_embeddings_dir": sequence_embeddings_dir,
        "sequence_embedding_suffix": sequence_embedding_suffix,
        "sequence_embedding_name_template": sequence_embedding_name_template,
        "require_sequence_embeddings": require_sequence_embeddings,
        "prepare_sequence_embeddings": prepare_sequence_embeddings,
        "sequence_embedding_generator": sequence_embedding_generator,
        "sequence_embedding_device": sequence_embedding_device,
        "sequence_embedding_workers": sequence_embedding_workers,
        "sequence_embedding_max_chain_length": sequence_embedding_max_chain_length,
        "include_without_structure": include_without_structure,
        "download": download,
        "force_download": force_download,
        "workers": workers,
        "download_workers": download_workers,
        "artifact_output_dir": artifact_output_dir,
        "artifact_s3_prefix": artifact_s3_prefix,
        "artifact_reuse_dirs": artifact_reuse_dir or None,
        "preprocess_workers": preprocess_workers,
        "upload_workers": upload_workers,
        "summary_csv": summary_csv,
        "reset_preprocess_checkpoint": reset_preprocess_checkpoint,
        "artifact_list_path": artifact_list_path,
        "validation_artifact_list_path": validation_artifact_list_path,
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "split_strategy": split_strategy,
        "holdout_group_ids_path": holdout_group_ids_path,
        "split_report_path": split_report_path,
        "target_voxel_size": target_voxel_size,
        "interface_contact_cutoff": interface_contact_cutoff,
        "force_preprocess": force_preprocess,
        "skip_missing": skip_missing,
        "fail_fast": fail_fast,
        "fix_shift": fix_shift,
        "preprocess_adapter": preprocess_adapter,
    }
    config = _configuration(
        "manifest",
        config_dir,
        config_name,
        override,
        values,
        s3_endpoint_url=s3_endpoint_url,
    )
    adapter = _adapter(config.pipeline.preprocess_adapter)
    generator = _adapter(
        config.pipeline.sequence_embedding_generator,
        option="--sequence-embedding-generator",
    )
    from cryoflake.preprocess.prepare import run

    try:
        run(
            config,
            preprocess_adapter=adapter,
            sequence_embedding_generator=generator,
        )
    except Exception as exc:
        _runtime_failure(exc)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``cryoflake`` console script."""
    app(args=None if argv is None else list(argv), prog_name="cryoflake")


if __name__ == "__main__":
    main()
