from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cryoflake.config import (
    artifact_contract,
    compose_config,
    target_contract,
    validate_config,
)


def test_packaged_contracts_and_groups_preserve_dataset_wire_format():
    artifact = artifact_contract()
    targets = target_contract()
    assert (artifact.format_name, artifact.format_version) == (
        "cryoflake-preprocessed",
        1,
    )
    assert (
        artifact.metadata_file_name,
        artifact.core_file_name,
        artifact.sequence_file_name,
    ) == (
        "metadata.json",
        "core.npz",
        "sequence.npz",
    )
    assert artifact.generation_key == "__cryoflake_store_generation__"
    assert targets.schema_version == 1
    assert (
        targets.full_atom_count,
        targets.rigid_group_count,
        targets.torsion_count,
    ) == (65, 9, 8)
    for group in ("manifest", "preprocess", "validate"):
        config = compose_config(pipeline=group)
        assert config.pipeline.name == group
        assert config.pipeline.sequence_embedding_suffix == ".npz"
        assert OmegaConf.is_struct(config)


@pytest.mark.parametrize("field", ["metadata_file", "core_file", "sequence_file"])
def test_artifact_contract_file_names_do_not_keep_legacy_aliases(field):
    assert not hasattr(artifact_contract(), field)
    with pytest.raises(ValueError, match=field):
        compose_config(values={"contracts": {"artifact": {field: "legacy.npz"}}})


def test_split_report_path_does_not_keep_legacy_config_alias():
    with pytest.raises(ValueError, match="split_report"):
        compose_config(values={"pipeline": {"split_report": None}})


def test_injected_preprocessing_adapter_uses_operation_specific_keyword():
    config = compose_config(
        pipeline="preprocess",
        values={
            "pipeline": {
                "map_path": "map.mrc",
                "structure_path": "structure.cif",
                "artifact_reference": "artifact",
            }
        },
    )
    validate_config(config, require_inputs=True, preprocess_adapter_provided=True)
    with pytest.raises(ValueError, match="preprocess_adapter is required"):
        validate_config(
            config, require_inputs=True, sequence_embedding_generator_provided=True
        )
    with pytest.raises(TypeError, match="adapter_provided"):
        validate_config(config, adapter_provided=True)


def test_hydra_overrides_and_interpolation_do_not_change_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CRYOFLAKE_S3_ENDPOINT_URL", "https://objects.example.test")
    config = compose_config(
        ["pipeline.workers=3", "pipeline.preprocess_workers=${pipeline.workers}"]
    )
    assert config.pipeline.workers == config.pipeline.preprocess_workers == 3
    assert config.storage.s3_endpoint_url == "https://objects.example.test"
    assert Path.cwd() == tmp_path
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "override",
    [
        "pipeline.workers=0",
        "pipeline.workers=true",
        "pipeline.workers=1.5",
        "pipeline.target_voxel_size=nan",
        "pipeline.interface_contact_cutoff=-1",
        "pipeline.validation_fraction=0",
        "pipeline.validation_fraction=1",
        "pipeline.validation_fraction=inf",
        "pipeline.split_strategy=invalid",
        "contracts.artifact.format_version=2",
        "contracts.artifact.format_version=true",
        "contracts.artifact.core_file_name=changed.npz",
        "contracts.artifact.generation_key=__changed_generation__",
        "+contracts.artifact.require_sequence_embeddings=true",
        "pipeline.sequence_embedding_workers=0",
        "pipeline.sequence_embedding_workers=true",
        "pipeline.sequence_embedding_max_chain_length=-1",
        "pipeline.sequence_embedding_max_chain_length=1.5",
        "pipeline.sequence_embedding_generator=missing-separator",
        "pipeline.sequence_embedding_suffix=.prot",
        "pipeline.sequence_embedding_suffix=.pkl",
        "pipeline.sequence_embedding_suffix=.pickle",
        "pipeline.sequence_embedding_suffix=.json",
        "pipeline.sequence_embedding_suffix=''",
        "contracts.targets.schema_version=2",
        "contracts.targets.full_atom_count=99",
        "+pipeline.unrecognized=true",
        "pipeline.artifact_reference=hf://buckets/user/artifact",
        "pipeline.artifact_reference=https://example.com/artifact",
    ],
)
def test_invalid_or_unsupported_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        compose_config([override])


@pytest.mark.parametrize("field", ["artifact_reference"])
@pytest.mark.parametrize("prefix", ["", "s3://bucket/"])
@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".prot", ".npy", ".npz"])
def test_artifact_configuration_rejects_single_file_references(field, prefix, suffix):
    with pytest.raises(ValueError, match=f"pipeline.{field}.*single-file"):
        compose_config(values={"pipeline": {field: f"{prefix}artifact{suffix}"}})


@pytest.mark.parametrize("field", ["artifact_reference"])
def test_artifact_configuration_rejects_existing_local_files(tmp_path, field):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"unsupported standalone artifact")
    with pytest.raises(ValueError, match=f"pipeline.{field}.*must be a directory"):
        compose_config(values={"pipeline": {field: str(path)}})


@pytest.mark.parametrize("field", ["artifact_reference"])
@pytest.mark.parametrize(
    "reference",
    [
        "artifact",
        "artifact/metadata.json",
        "s3://bucket/artifact",
        "s3://bucket/artifact/metadata.json",
        "s3://bucket",
    ],
)
def test_artifact_configuration_accepts_roots_and_metadata_references(field, reference):
    config = compose_config(values={"pipeline": {field: reference}})
    assert config.pipeline[field] == reference


@pytest.mark.parametrize("reference_kind", ["cwd", "cwd_metadata", "root", "alias"])
def test_preprocessing_artifact_reference_requires_a_dedicated_directory(
    monkeypatch, tmp_path, reference_kind
):
    monkeypatch.chdir(tmp_path)
    if reference_kind == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
    artifact_reference = {
        "cwd": str(tmp_path),
        "cwd_metadata": str(tmp_path / "metadata.json"),
        "root": tmp_path.anchor,
        "alias": str(tmp_path / "alias"),
    }[reference_kind]
    with pytest.raises(
        ValueError, match="pipeline.artifact_reference.*dedicated subdirectory"
    ):
        compose_config(
            pipeline="preprocess",
            values={"pipeline": {"artifact_reference": artifact_reference}},
        )


@pytest.mark.parametrize("artifact_reference", [".", "metadata.json", "/"])
def test_validation_artifact_reference_keeps_read_only_root_semantics(
    monkeypatch, tmp_path, artifact_reference
):
    monkeypatch.chdir(tmp_path)
    config = compose_config(
        pipeline="validate",
        values={"pipeline": {"artifact_reference": artifact_reference}},
    )
    assert config.pipeline.artifact_reference == artifact_reference


def test_artifact_configuration_does_not_restrict_raw_inputs_or_parent_directories():
    config = compose_config(
        values={
            "pipeline": {
                "map_path": "input.npy",
                "structure_path": "input.pkl",
                "artifact_output_dir": "scratch.npz",
                "artifact_s3_prefix": "s3://bucket/collection.npz",
            }
        }
    )
    assert config.pipeline.map_path == "input.npy"
    assert config.pipeline.structure_path == "input.pkl"
    assert config.pipeline.artifact_output_dir == "scratch.npz"
    assert config.pipeline.artifact_s3_prefix == "s3://bucket/collection.npz"


def test_missing_required_pipeline_inputs_are_reported_before_io():
    config = compose_config(pipeline="preprocess")
    with pytest.raises(ValueError, match="pipeline.map_path is required"):
        validate_config(config, require_inputs=True)


def test_manifest_requires_an_explicit_source_csv():
    config = compose_config(values={"pipeline": {"manifest_csv": "manifest.csv"}})
    assert config.pipeline.source_csv is None
    with pytest.raises(ValueError, match="pipeline.source_csv is required"):
        validate_config(config, require_inputs=True)


def test_sequence_requirement_is_pipeline_policy_not_wire_contract():
    config = compose_config(["pipeline.require_sequence_embeddings=true"])
    assert config.pipeline.require_sequence_embeddings is True
    assert "require_sequence_embeddings" not in config.contracts.artifact


def test_typed_python_values_override_hydra_before_cross_field_validation():
    config = compose_config(
        ["pipeline.download=true"],
        values={
            "pipeline": {"maps_dir": "maps", "structures_dir": "structures"},
        },
    )
    assert config.pipeline.download
    assert config.pipeline.maps_dir == "maps"


def test_unknown_hydra_override_has_actionable_error():
    with pytest.raises(ValueError, match="Invalid configuration"):
        compose_config(["pipeline.wrokers=2"])


def test_repository_project_config_reuses_packaged_groups():
    config_dir = Path(__file__).parents[1] / "configs"
    config = compose_config(["pipeline=project"], config_dir=config_dir)
    assert config.pipeline.name == "manifest"
    assert config.pipeline.workers == 4
    assert config.pipeline.manifest_csv == "data/manifest.csv"
    assert config.contracts.artifact.format_version == 1


def test_config_loads_selected_dotenv_before_resolving_storage_endpoint(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text("CRYOFLAKE_S3_ENDPOINT_URL=https://dotenv.example.test\n")
    monkeypatch.setenv("CRYOFLAKE_ENV_FILE", str(env_file))
    monkeypatch.delenv("CRYOFLAKE_S3_ENDPOINT_URL", raising=False)
    config = compose_config()
    assert config.storage.s3_endpoint_url == "https://dotenv.example.test"
    monkeypatch.setenv("CRYOFLAKE_S3_ENDPOINT_URL", "https://process.example.test")
    assert compose_config().storage.s3_endpoint_url == "https://process.example.test"


def test_missing_selected_dotenv_is_a_configuration_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYOFLAKE_ENV_FILE", str(tmp_path / "missing.env"))
    with pytest.raises(ValueError, match="S3 environment file does not exist"):
        compose_config()
    monkeypatch.delenv("CRYOFLAKE_ENV_FILE")
    assert compose_config().pipeline.name == "manifest"


@pytest.mark.parametrize(
    "field", ["source_csv", "manifest_csv", "maps_dir", "artifact_reference"]
)
def test_configuration_rejects_blank_paths(field):
    with pytest.raises(ValueError, match=f"pipeline.{field}.*nonempty"):
        compose_config(values={"pipeline": {field: "   "}})


@pytest.mark.parametrize(
    "field", ["map_path", "structure_path", "original_structure_path", "fasta_path"]
)
@pytest.mark.parametrize("scheme", ["s3://bucket/", "https://example.test/"])
def test_raw_input_paths_reject_remote_references(field, scheme):
    with pytest.raises(ValueError, match=rf"pipeline.{field}.*local filesystem"):
        compose_config(values={"pipeline": {field: f"{scheme}raw-input"}})


@pytest.mark.parametrize(
    "field", ["map_path", "structure_path", "original_structure_path", "fasta_path"]
)
def test_raw_input_paths_reject_existing_directories(tmp_path, field):
    with pytest.raises(ValueError, match=rf"pipeline.{field}.*file path"):
        compose_config(values={"pipeline": {field: str(tmp_path)}})


def test_artifact_reuse_search_parents_reject_existing_files(tmp_path):
    file_path = tmp_path / "artifact.bin"
    file_path.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="artifact_reuse_dirs.*local directory"):
        compose_config(values={"pipeline": {"artifact_reuse_dirs": [str(file_path)]}})


@pytest.mark.parametrize("field", ["maps_dir", "structures_dir", "artifact_output_dir"])
def test_local_pipeline_directories_reject_remote_uris(field):
    with pytest.raises(ValueError, match=f"pipeline.{field}.*local"):
        compose_config(values={"pipeline": {field: "s3://bucket/directory"}})


@pytest.mark.parametrize("suffix", ["?version=1", "#fragment"])
def test_remote_preprocessing_prefix_rejects_uri_components(suffix):
    with pytest.raises(ValueError, match="artifact_s3_prefix"):
        compose_config(
            values={
                "pipeline": {
                    "artifact_output_dir": "scratch",
                    "artifact_s3_prefix": f"s3://bucket/artifacts{suffix}",
                }
            }
        )


@pytest.mark.parametrize(
    "endpoint",
    ["invalid", "ftp://objects.test", "https://", "https://objects.test:bad"],
)
def test_storage_endpoint_requires_a_valid_http_url(endpoint):
    with pytest.raises(ValueError, match="storage.s3_endpoint_url"):
        compose_config(values={"storage": {"s3_endpoint_url": endpoint}})


def test_blank_optional_storage_endpoint_preserves_default_aws_configuration():
    assert (
        compose_config(
            values={"storage": {"s3_endpoint_url": ""}}
        ).storage.s3_endpoint_url
        == ""
    )


@pytest.mark.parametrize(
    "specification", ["os:path..join", "os..path:join", "os:path:join"]
)
def test_adapter_configuration_rejects_malformed_dotted_names(specification):
    with pytest.raises(ValueError, match="MODULE:CALLABLE"):
        compose_config(values={"pipeline": {"preprocess_adapter": specification}})


@pytest.mark.parametrize(
    "template", ["{unknown}", "{wwpdb_id", "{}", "{wwpdb_id.__class__}"]
)
def test_embedding_template_errors_are_configuration_errors(template):
    with pytest.raises(ValueError, match="sequence_embedding_name_template"):
        compose_config(
            values={"pipeline": {"sequence_embedding_name_template": template}}
        )


@pytest.mark.parametrize("link_kind", ["same", "symlink", "hardlink"])
def test_manifest_output_cannot_overwrite_source_csv(tmp_path, link_kind):
    source = tmp_path / "source.csv"
    source.write_text("EMD-,Associated PDB\n1234,1abc\n")
    output = source if link_kind == "same" else tmp_path / "manifest.csv"
    if link_kind == "symlink":
        output.symlink_to(source)
    elif link_kind == "hardlink":
        output.hardlink_to(source)
    with pytest.raises(ValueError, match="source_csv.*manifest_csv"):
        compose_config(
            values={
                "pipeline": {"source_csv": str(source), "manifest_csv": str(output)}
            }
        )


def test_preprocessing_output_files_must_be_distinct(tmp_path):
    with pytest.raises(
        ValueError, match="artifact_list_path.*validation_artifact_list_path"
    ):
        compose_config(
            values={
                "pipeline": {
                    "artifact_output_dir": str(tmp_path / "artifacts"),
                    "artifact_list_path": str(tmp_path / "train.txt"),
                    "validation_artifact_list_path": str(tmp_path / "train.txt"),
                }
            }
        )


@pytest.mark.parametrize(
    "suffix", ["checkpoint.json", "checkpoint.jsonl", "checkpoint.lock"]
)
def test_output_files_cannot_overwrite_checkpoint_state(tmp_path, suffix):
    summary = tmp_path / "summary.csv"
    with pytest.raises(ValueError, match="artifact_list_path.*summary_csv"):
        compose_config(
            values={
                "pipeline": {
                    "artifact_output_dir": str(tmp_path / "artifacts"),
                    "summary_csv": str(summary),
                    "artifact_list_path": f"{summary}.{suffix}",
                }
            }
        )
