"""Compose and validate the packaged Hydra pipeline and data contracts.

``compose_config()`` never changes the working directory or launches a job.
Use ``validate_config(..., require_inputs=True)`` before performing I/O.
Wire-format fields describe the supported format; unsupported changes fail
explicitly instead of producing artifacts that other readers cannot decode.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from string import Formatter
from urllib.parse import urlsplit

from hydra import compose, initialize_config_dir, initialize_config_module
from hydra.errors import HydraException
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from cryoflake.common import paths_must_be_distinct


@dataclass
class ArtifactContract:
    format_name: str = "cryoflake-preprocessed"
    format_version: int = 1
    generation_key: str = "__cryoflake_store_generation__"
    metadata_file_name: str = "metadata.json"
    core_file_name: str = "core.npz"
    sequence_file_name: str = "sequence.npz"
    max_metadata_bytes: int = 16 * 1024 * 1024


@dataclass
class TargetContract:
    schema_version: int = 1
    full_atom_count: int = 65
    rigid_group_count: int = 9
    torsion_count: int = 8
    unknown_protein_index: int = 28
    full_residue_type_count: int = 31


@dataclass
class ManifestContract:
    required_source_columns: list[str] = field(
        default_factory=lambda: ["EMD-", "Associated PDB"]
    )


@dataclass
class Contracts:
    artifact: ArtifactContract = field(default_factory=ArtifactContract)
    targets: TargetContract = field(default_factory=TargetContract)
    manifest: ManifestContract = field(default_factory=ManifestContract)


@dataclass
class PipelineConfig:
    name: str = "manifest"
    source_csv: str | None = None
    manifest_csv: str | None = None
    maps_dir: str | None = None
    structures_dir: str | None = None
    sequence_embeddings_dir: str | None = None
    sequence_embedding_suffix: str = ".npz"
    sequence_embedding_name_template: str = "{wwpdb_id}"
    require_sequence_embeddings: bool = False
    prepare_sequence_embeddings: bool = False
    sequence_embedding_generator: str | None = None
    sequence_embedding_device: str | None = None
    sequence_embedding_workers: int = 1
    sequence_embedding_max_chain_length: int = 1000
    include_without_structure: bool = False
    download: bool = False
    force_download: bool = False
    workers: int = 1
    download_workers: int | None = None
    preprocess_workers: int | None = None
    upload_workers: int | None = None
    artifact_output_dir: str | None = None
    artifact_s3_prefix: str | None = None
    artifact_reuse_dirs: list[str] = field(default_factory=list)
    summary_csv: str | None = None
    reset_preprocess_checkpoint: bool = False
    artifact_list_path: str | None = None
    validation_artifact_list_path: str | None = None
    validation_fraction: float = 0.1
    split_seed: int = 0
    split_strategy: str = "artifact"
    holdout_group_ids_path: str | None = None
    split_report_path: str | None = None
    target_voxel_size: float = 1.0
    interface_contact_cutoff: float = 5.0
    force_preprocess: bool = False
    skip_missing: bool = False
    fail_fast: bool = False
    fix_shift: bool = True
    preprocess_adapter: str | None = None
    map_path: str | None = None
    structure_path: str | None = None
    artifact_reference: str | None = None
    original_structure_path: str | None = None
    fasta_path: str | None = None


@dataclass
class StorageConfig:
    s3_endpoint_url: str | None = None


@dataclass
class CryoflakeConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    contracts: Contracts = field(default_factory=Contracts)
    storage: StorageConfig = field(default_factory=StorageConfig)


def _contract_from_yaml(kind: str, name: str, schema):
    resource = files("cryoflake.configs").joinpath("contracts", kind, f"{name}.yaml")
    with resource.open("r", encoding="utf-8") as stream:
        value = OmegaConf.merge(OmegaConf.structured(schema), OmegaConf.load(stream))
    return OmegaConf.to_object(value)


def artifact_contract(config: DictConfig | None = None) -> ArtifactContract:
    """Return the artifact contract, loaded from the packaged YAML by default."""
    if config is not None:
        return OmegaConf.to_object(config.contracts.artifact)
    return _contract_from_yaml("artifact", "npz_json", ArtifactContract)


def target_contract(config: DictConfig | None = None) -> TargetContract:
    """Return the component and full-atom target format contract."""
    if config is not None:
        return OmegaConf.to_object(config.contracts.targets)
    return _contract_from_yaml("targets", "component", TargetContract)


def compose_config(
    overrides: Sequence[str] = (),
    *,
    config_dir: str | Path | None = None,
    config_name: str = "config",
    pipeline: str | None = None,
    values: Mapping[str, object] | None = None,
) -> DictConfig:
    """Compose Hydra groups and overrides, then apply a strict typed schema.

    A custom directory must contain ``config_name.yaml`` and its referenced
    groups, or include ``pkg://cryoflake.configs`` in ``hydra.searchpath``.
    Overrides use Hydra syntax, for example ``pipeline.workers=4``. ``values``
    applies typed Python values after Hydra overrides, before validation.
    """
    arguments = list(overrides)
    if pipeline is not None:
        arguments.insert(0, f"pipeline={pipeline}")
    try:
        # Raw adapters also need the root .env, even if they do not use the
        # built-in storage client. Existing process variables retain priority.
        from cryoflake.storage.transports import load_s3_environment

        load_s3_environment()
        initializer = (
            initialize_config_module(
                config_module="cryoflake.configs", version_base="1.3"
            )
            if config_dir is None
            else initialize_config_dir(
                config_dir=str(Path(config_dir).expanduser().resolve()),
                version_base="1.3",
            )
        )
        with initializer:
            composed = compose(config_name=config_name, overrides=arguments)
        if values:
            composed = OmegaConf.merge(composed, values)
        for name in (
            "workers",
            "download_workers",
            "preprocess_workers",
            "upload_workers",
            "sequence_embedding_workers",
            "sequence_embedding_max_chain_length",
            "split_seed",
        ):
            if isinstance(OmegaConf.select(composed, f"pipeline.{name}"), bool):
                raise ValueError(f"pipeline.{name} must be an integer, not a boolean")
        for section, names in (
            ("artifact", ("format_version", "max_metadata_bytes")),
            ("targets", tuple(TargetContract.__dataclass_fields__)),
        ):
            for name in names:
                if isinstance(
                    OmegaConf.select(composed, f"contracts.{section}.{name}"), bool
                ):
                    raise ValueError(
                        f"contracts.{section}.{name} must be an integer, not a boolean"
                    )
        config = OmegaConf.merge(OmegaConf.structured(CryoflakeConfig), composed)
        OmegaConf.resolve(config)
        validate_config(config)
    except (HydraException, OmegaConfBaseException, OSError) as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    return config


def _validate_reference(value: str | None, name: str, *, s3_only: bool = False) -> None:
    if value is None:
        return
    if not value.strip():
        raise ValueError(f"{name} must be a nonempty path or S3 reference")
    from cryoflake.storage.transports import join_remote_uri

    parsed = urlsplit(value)
    if parsed.scheme and (parsed.scheme != "s3" or not parsed.netloc):
        raise ValueError(f"{name} supports local paths and s3://bucket/key references")
    if s3_only and (parsed.scheme != "s3" or not parsed.netloc):
        raise ValueError(f"{name} must be an s3://bucket/prefix reference")
    if parsed.scheme:
        try:
            join_remote_uri(value, "configuration-check")
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc


def _validate_manifest_destinations(p: DictConfig) -> None:
    """Prevent generated files from overwriting one another or source inputs."""
    if p.name != "manifest":
        return
    outputs = {"manifest_csv": p.manifest_csv}
    inputs = {"source_csv": p.source_csv}
    if p.artifact_output_dir:
        manifest_csv = Path(p.manifest_csv) if p.manifest_csv else None
        summary = p.summary_csv or (
            str(manifest_csv.with_name(f"{manifest_csv.stem}_summary.csv"))
            if manifest_csv is not None
            else None
        )
        outputs.update(
            summary_csv=summary,
            artifact_list_path=p.artifact_list_path
            or str(Path(p.artifact_output_dir) / "artifact_list.txt"),
            validation_artifact_list_path=p.validation_artifact_list_path,
            split_report_path=p.split_report_path,
        )
        if summary is not None:
            for suffix in ("checkpoint.json", "checkpoint.jsonl", "checkpoint.lock"):
                outputs[f"summary_csv.{suffix}"] = f"{summary}.{suffix}"
        inputs["holdout_group_ids_path"] = p.holdout_group_ids_path
    paths_must_be_distinct(
        {f"pipeline.{name}": value for name, value in (inputs | outputs).items()}
    )


def validate_config(
    config: DictConfig,
    *,
    require_inputs: bool = False,
    preprocess_adapter_provided: bool = False,
    sequence_embedding_generator_provided: bool = False,
) -> None:
    """Reject unsupported formats and invalid pipeline values before side effects."""
    p = config.pipeline
    if p.name not in {"manifest", "preprocess", "validate"}:
        raise ValueError("pipeline.name must be manifest, preprocess, or validate")
    if p.sequence_embedding_suffix.lower() not in {".npy", ".npz"}:
        raise ValueError("pipeline.sequence_embedding_suffix must be .npy or .npz")
    local_paths = (
        "source_csv",
        "manifest_csv",
        "maps_dir",
        "structures_dir",
        "sequence_embeddings_dir",
        "artifact_output_dir",
        "summary_csv",
        "artifact_list_path",
        "validation_artifact_list_path",
        "holdout_group_ids_path",
        "split_report_path",
        "map_path",
        "structure_path",
        "original_structure_path",
        "fasta_path",
    )
    local_directories = {
        "maps_dir",
        "structures_dir",
        "sequence_embeddings_dir",
        "artifact_output_dir",
    }
    for name in local_paths:
        value = p[name]
        if value is not None and not value.strip():
            raise ValueError(f"pipeline.{name} must be a nonempty path")
        if value is not None and urlsplit(value).scheme:
            raise ValueError(f"pipeline.{name} must be a local filesystem path")
        if value is not None:
            path = Path(value)
            if name in local_directories:
                if (path.exists() or path.is_symlink()) and not path.is_dir():
                    raise ValueError(f"pipeline.{name} must be a directory")
            elif path.is_dir():
                raise ValueError(f"pipeline.{name} must be a file path")
    for value in p.artifact_reuse_dirs:
        if not value.strip() or urlsplit(value).scheme:
            raise ValueError(
                "pipeline.artifact_reuse_dirs must contain nonempty local paths"
            )
        directory = Path(value)
        if (directory.exists() or directory.is_symlink()) and not directory.is_dir():
            raise ValueError(
                "pipeline.artifact_reuse_dirs must contain local directory paths"
            )
    endpoint = config.storage.s3_endpoint_url
    # The documented .env template leaves this optional value blank for AWS.
    if endpoint not in (None, ""):
        try:
            parsed_endpoint = urlsplit(endpoint)
            valid_endpoint = (
                parsed_endpoint.scheme in {"http", "https"}
                and parsed_endpoint.hostname
                and not parsed_endpoint.query
                and not parsed_endpoint.fragment
                and not any(character.isspace() for character in endpoint)
            )
            parsed_endpoint.port  # Validate an explicitly supplied port too.
        except ValueError as exc:
            raise ValueError(
                f"storage.s3_endpoint_url must be a valid HTTP(S) URL: {exc}"
            ) from exc
        if not valid_endpoint:
            raise ValueError("storage.s3_endpoint_url must be a valid HTTP(S) URL")
    template_values = {
        "emdb_id": "1234",
        "emdb_accession": "EMD-1234",
        "artifact_name": "emd_1234_1abc",
        "wwpdb_id": "1abc",
    }
    try:
        template = p.sequence_embedding_name_template
        for _, name, _, _ in Formatter().parse(template):
            if name is not None and name not in template_values:
                raise ValueError(f"unsupported field {name!r}")
        if not template.format(**template_values).strip():
            raise ValueError("template produces an empty name")
    except (KeyError, IndexError, ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid pipeline.sequence_embedding_name_template: {exc}"
        ) from exc
    _validate_manifest_destinations(p)
    supported_artifact = artifact_contract()
    for name in (
        "format_name",
        "format_version",
        "generation_key",
        "metadata_file_name",
        "core_file_name",
        "sequence_file_name",
        "max_metadata_bytes",
    ):
        if config.contracts.artifact[name] != getattr(supported_artifact, name):
            raise ValueError(f"Unsupported artifact contract field: {name}")
    supported_targets = target_contract()
    for name in supported_targets.__dataclass_fields__:
        if config.contracts.targets[name] != getattr(supported_targets, name):
            raise ValueError(f"Unsupported target contract field: {name}")
    columns = list(config.contracts.manifest.required_source_columns)
    if (
        not {"EMD-", "Associated PDB"}.issubset(columns)
        or any(not column.strip() for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError(
            "Manifest contract requires unique, nonempty EMD- and Associated PDB columns"
        )
    for name in (
        "workers",
        "download_workers",
        "preprocess_workers",
        "upload_workers",
        "sequence_embedding_workers",
        "sequence_embedding_max_chain_length",
    ):
        value = p[name]
        if value is not None and (isinstance(value, bool) or value <= 0):
            raise ValueError(f"pipeline.{name} must be a positive integer")
    for name in ("target_voxel_size", "interface_contact_cutoff"):
        value = p[name]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"pipeline.{name} must be a positive finite number")
    if not math.isfinite(p.validation_fraction) or not 0 < p.validation_fraction < 1:
        raise ValueError(
            "pipeline.validation_fraction must be greater than 0 and less than 1"
        )
    if p.split_strategy not in {"artifact", "component", "interface"}:
        raise ValueError(
            "pipeline.split_strategy must be artifact, component, or interface"
        )
    if p.split_strategy != "artifact" and not p.validation_artifact_list_path:
        raise ValueError(
            "Grouped splitting requires pipeline.validation_artifact_list_path"
        )
    if (
        p.holdout_group_ids_path or p.split_report_path
    ) and p.split_strategy == "artifact":
        raise ValueError(
            "Holdout groups and split reports require component or interface splitting"
        )
    if p.download and (not p.maps_dir or not p.structures_dir):
        raise ValueError(
            "Downloading requires pipeline.maps_dir and pipeline.structures_dir"
        )
    if p.prepare_sequence_embeddings and (
        not p.sequence_embeddings_dir or not p.structures_dir
    ):
        raise ValueError(
            "Sequence embedding generation requires pipeline.sequence_embeddings_dir "
            "and pipeline.structures_dir"
        )
    if p.artifact_s3_prefix and not p.artifact_output_dir:
        raise ValueError(
            "Remote preprocessing requires pipeline.artifact_output_dir scratch storage"
        )
    _validate_reference(p.artifact_s3_prefix, "artifact_s3_prefix", s3_only=True)
    _validate_reference(p.artifact_reference, "pipeline.artifact_reference")
    if p.artifact_reference is not None:
        from cryoflake.storage.store import preprocessed_artifact_root

        try:
            root = preprocessed_artifact_root(p.artifact_reference)
        except ValueError as exc:
            raise ValueError(f"pipeline.artifact_reference: {exc}") from exc
        if p.name == "preprocess" and isinstance(root, Path):
            resolved_root = root.resolve()
            if resolved_root == Path.cwd().resolve() or resolved_root == Path(
                resolved_root.anchor
            ):
                raise ValueError(
                    "pipeline.artifact_reference must be a dedicated subdirectory, "
                    "not the current directory or a filesystem root"
                )
    for name in ("preprocess_adapter", "sequence_embedding_generator"):
        if p[name] is not None:
            if name == "preprocess_adapter" and p[name] == "default":
                continue
            module, separator, attribute = p[name].partition(":")
            if not separator or not all(
                part.isidentifier()
                for dotted_name in (module.strip(), attribute.strip())
                for part in dotted_name.split(".")
            ):
                choices = (
                    "be 'default' or use the MODULE:CALLABLE form"
                    if name == "preprocess_adapter"
                    else "use the MODULE:CALLABLE form"
                )
                raise ValueError(f"{name} must {choices}")
    if require_inputs:
        required = {
            "manifest": ("source_csv", "manifest_csv"),
            "preprocess": (
                "map_path",
                "structure_path",
                "artifact_reference",
                "preprocess_adapter",
            ),
            "validate": ("artifact_reference",),
        }[p.name]
        for name in required:
            if name == "preprocess_adapter" and preprocess_adapter_provided:
                continue
            if not p[name]:
                raise ValueError(f"pipeline.{name} is required")
        if (
            p.artifact_output_dir
            and not p.preprocess_adapter
            and not preprocess_adapter_provided
        ):
            raise ValueError(
                "--artifact-output-dir requires --preprocess-adapter "
                "default or MODULE:CALLABLE"
            )
        if (
            p.prepare_sequence_embeddings
            and not p.sequence_embedding_generator
            and not sequence_embedding_generator_provided
        ):
            raise ValueError(
                "Sequence embedding generation requires "
                "--sequence-embedding-generator MODULE:CALLABLE or an injected "
                "sequence_embedding_generator"
            )
