"""Run the CLI and pipeline examples directly from the user documentation."""

from __future__ import annotations

import csv
import gzip
import json
import os
import shlex
import shutil
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from dotenv import dotenv_values
from omegaconf import OmegaConf

from cryoflake.manifests.manifest import read_artifact_list, read_manifest_csv
from cryoflake.storage.artifacts import validate_preprocessed_artifact
from cryoflake.storage.codec import read_metadata
from cryoflake.storage.store import load_preprocessed_artifact

from .examples import get_example

pytestmark = pytest.mark.documentation


def _options(example_id: str) -> list[str]:
    return shlex.split(get_example(example_id).code)


def _batch_command() -> list[str]:
    (command,) = get_example("guide-batch").commands()
    return command


def _summary(root: Path) -> list[dict[str, str]]:
    with (root / "data/summary.csv").open(newline="") as source:
        return list(csv.DictReader(source))


def _assert_default_artifact(artifact_reference: Path | str) -> dict:
    """Check observed fixture data rather than accepting any valid artifact."""
    artifact_mapping = load_preprocessed_artifact(artifact_reference)
    assert {"density_grid", "backbone_coordinates", "full_atom_targets"}.issubset(
        artifact_mapping
    )
    assert {"grid_io", "cas"}.isdisjoint(artifact_mapping)
    assert artifact_mapping["density_grid"].dtype == np.float32
    assert np.isfinite(artifact_mapping["density_grid"]).all()
    assert len(set(artifact_mapping["density_grid"].shape)) == 1
    assert artifact_mapping["density_grid"].shape[0] % 2 == 0
    assert artifact_mapping["shape"].dtype == np.int64
    np.testing.assert_array_equal(
        artifact_mapping["shape"], artifact_mapping["density_grid"].shape
    )
    np.testing.assert_allclose(
        artifact_mapping["cella"],
        artifact_mapping["shape"] * artifact_mapping["voxel_size"],
    )
    amplitude = artifact_mapping["grid_spectral_amplitude"]
    assert amplitude.dtype == np.float64
    assert amplitude.shape == (artifact_mapping["density_grid"].shape[-1] // 2 + 1,)
    assert np.isfinite(amplitude).all()
    np.testing.assert_allclose(artifact_mapping["global_origin"], [-62.0] * 3)
    np.testing.assert_array_equal(
        artifact_mapping["backbone_coordinates"]["CA"], [[1.0, 1.0, 1.0]]
    )
    np.testing.assert_array_equal(
        artifact_mapping["backbone_coordinates"]["P"], [[2.0, 2.0, 2.0]]
    )
    targets = artifact_mapping["full_atom_targets"]
    np.testing.assert_array_equal(targets["aatype"], [0, 20])
    np.testing.assert_array_equal(targets["prot_mask"], [True, False])
    np.testing.assert_array_equal(targets["chain_index"], [0, 1])
    np.testing.assert_array_equal(targets["residue_index"], [10, 20])
    assert targets["atom_positions"].shape == (2, 65, 3)
    assert targets["atom_mask"].dtype == np.bool_
    np.testing.assert_array_equal(targets["atom_mask"].sum(axis=1), [4, 1])
    assert targets["rigidgroups_gt_exists"][0].any()
    assert not targets["rigidgroups_gt_exists"][1].any()
    assert not targets["torsion_angles_mask"][1].any()
    assert artifact_mapping["extension_targets"]["schema_version"] == 1
    assert "sequence" not in artifact_mapping
    return artifact_mapping


def _assert_complete_batch(workspace, *, remote: bool = False) -> list[Path | str]:
    paths = read_artifact_list(workspace.root / "data/train.txt")
    assert len(paths) == 2
    assert len(set(map(str, paths))) == 2
    rows = _summary(workspace.root)
    assert len(rows) == 2
    assert all(row["success"].lower() == "true" for row in rows)
    assert all(row["status"] in {"processed", "reused"} for row in rows)
    assert all(row["cached_artifact_dir"] == "" for row in rows)
    assert [float(row["source_resolution"]) for row in rows] == [3.2, 2.8]
    assert all(float(row["input_voxel_size"]) == 1.0 for row in rows)
    assert all(float(row["artifact_voxel_size"]) > 0 for row in rows)
    summary_references = {
        row["artifact_reference"]
        if remote
        else str(Path(row["artifact_reference"]).resolve())
        for row in rows
    }
    assert summary_references == set(map(str, paths))
    for path in paths:
        result = validate_preprocessed_artifact(path)
        assert result.protein_nodes > 0
        assert result.component_nodes > 0
        assert all(size > 0 for size in result.grid_shape)
        if remote:
            assert str(path).startswith("s3://my-bucket/artifacts/")
        else:
            assert Path(path).is_dir()
    return paths


@pytest.mark.parametrize(
    "example_id", ["readme-install", "guide-install", "guide-wheel-install"]
)
def test_installation_snippets_exercise_the_installed_cli(doc_workspace, example_id):
    # Environment-mutating uv commands are checked separately by CI installation.
    commands = get_example(example_id).commands()
    assert commands
    outputs = [doc_workspace.invoke(command).output for command in commands]
    assert any("prepare-manifest" in output for output in outputs)
    if example_id == "guide-install":
        config = OmegaConf.create(outputs[-1])
        assert config.pipeline.name == "manifest"
        assert config.contracts.artifact.format_name == "cryoflake-preprocessed"


@pytest.mark.parametrize("example_id", ["guide-manifest", "readme-manifest"])
def test_manifest_examples_preserve_associations_and_resolve_paths(
    doc_workspace, example_id
):
    example = get_example(example_id)
    if example_id == "guide-manifest":
        assert example.python_setup, "The standalone guide must create its source CSV"
        source = doc_workspace.root / "data/source.csv"
        source.unlink()
        exec(example.python_setup, {})
        assert source.is_file()
    for command in example.commands():
        doc_workspace.invoke(command)
    manifest = doc_workspace.root / "data/manifest.csv"
    rows = read_manifest_csv(manifest)
    assert [(row["emdb_id"], row["wwpdb_id"]) for row in rows] == [
        ("1234", "1abc"),
        ("5678", "2def"),
    ]
    assert [float(row["source_resolution"]) for row in rows] == [3.2, 2.8]
    for row in rows:
        assert row["emdb_accession"] == f"EMD-{row['emdb_id']}"
        assert row["map_url"].endswith(f"/emd_{row['emdb_id']}.map.gz")
        assert row["structure_url"].endswith(f"/{row['wwpdb_id']}.cif.gz")
        assert not Path(row["map_path"]).is_absolute()
        assert (manifest.parent / row["map_path"]).is_file()
        assert (manifest.parent / row["structure_path"]).is_file()


def test_download_example_fetches_real_gzip_inputs(doc_workspace):
    for command in get_example("guide-download").commands():
        doc_workspace.invoke(command)
    assert len(doc_workspace.http_calls) == 4
    for relative in (
        "maps/emd_1234.map.gz",
        "maps/emd_5678.map.gz",
        "structures/1abc.cif.gz",
        "structures/2def.cif.gz",
    ):
        with gzip.open(doc_workspace.root / "data" / relative, "rb") as source:
            assert source.read(1)
    rows = read_manifest_csv(doc_workspace.root / "data/manifest.csv")
    # Manifest selection precedes downloads and prefers existing plain inputs.
    assert all(row["map_path"].endswith(".map") for row in rows)


def test_single_example_runs_default_adapter_and_validates_actual_artifact(
    doc_workspace,
):
    artifact = doc_workspace.root / "data/artifacts/example"
    shutil.rmtree(artifact)
    results = [
        doc_workspace.invoke(command)
        for command in get_example("guide-single").commands()
    ]
    assert not doc_workspace.adapter_calls
    result = json.loads(results[-1].output)
    actual = validate_preprocessed_artifact(artifact)
    assert result["grid_shape"] == list(actual.grid_shape)
    assert result["protein_nodes"] == actual.protein_nodes == 1
    assert result["nucleotide_nodes"] == actual.nucleotide_nodes == 1
    assert result["component_nodes"] == actual.component_nodes == 1
    artifact_mapping = _assert_default_artifact(artifact)
    assert actual.grid_shape == (86, 86, 86)
    assert artifact_mapping["voxel_size"] == pytest.approx(128 / 86)
    metadata = read_metadata(artifact / "metadata.json")
    assert metadata["format"] == "cryoflake-preprocessed"
    assert metadata["format_version"] == 1
    assert set(metadata["files"]) == {"core"}


@pytest.mark.parametrize(
    ("flag", "expected"), [("--fix-shift", True), ("--no-fix-shift", False)]
)
def test_documented_shift_flags_control_default_origin_translation(
    doc_workspace, flag, expected
):
    example = get_example("guide-single")
    assert f"`{flag}`" in example.path.read_text()
    command = next(
        command for command in example.commands() if command[0] == "preprocess"
    )

    original_structure = doc_workspace.root / "data/original.cif"
    lines = (doc_workspace.root / "data/model.cif").read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith(("ATOM ", "HETATM ")):
            fields = line.split()
            fields[-3] = str(float(fields[-3]) + 10.0)
            lines[index] = " ".join(fields)
    original_structure.write_text("\n".join(lines) + "\n")

    doc_workspace.invoke(
        [*command, "--original-structure-path", str(original_structure), flag]
    )

    artifact_mapping = load_preprocessed_artifact("data/artifacts/example")
    np.testing.assert_allclose(
        artifact_mapping["global_origin"],
        [-52.0 if expected else -62.0, -62.0, -62.0],
    )
    np.testing.assert_allclose(
        artifact_mapping["backbone_coordinates"]["CA"], [[11.0, 1.0, 1.0]]
    )
    assert not doc_workspace.adapter_calls


def test_single_default_adapter_accepts_gzip_inputs_and_fasta_metadata(doc_workspace):
    command = next(
        command
        for command in get_example("guide-single").commands()
        if command[0] == "preprocess"
    )
    for argument in (1, 2):
        source = doc_workspace.root / command[argument]
        compressed = source.with_name(source.name + ".gz")
        compressed.write_bytes(gzip.compress(source.read_bytes()))
        command[argument] = str(compressed)
    fasta_path = doc_workspace.root / "data/chains.fasta.gz"
    fasta_path.write_bytes(gzip.compress(b">protein\nA\n"))

    doc_workspace.invoke([*command, "--fasta-path", str(fasta_path)])

    artifact_mapping = _assert_default_artifact("data/artifacts/example")
    assert artifact_mapping["fasta"] == {
        "sequences": [{"sequence": "A", "chain_count": 1, "chain_ids": ["A"]}],
        "sequence_names": ["protein"],
        "chain_to_seqid": {"A": 0},
    }
    assert (
        validate_preprocessed_artifact("data/artifacts/example").sequence_length is None
    )


@pytest.mark.parametrize(
    "argument, value",
    [
        (1, "s3://bucket/raw-map.mrc"),
        (1, "data/maps"),
        (2, "s3://bucket/raw-structure.cif"),
        (2, "data/structures"),
        ("--original-structure-path", "s3://bucket/original.cif"),
        ("--fasta-path", "data/structures"),
    ],
)
def test_single_artifact_guide_rejects_invalid_raw_file_paths_before_adapter_io(
    doc_workspace, argument, value
):
    command = next(
        command
        for command in get_example("guide-single").commands()
        if command[0] == "preprocess"
    )
    if isinstance(argument, int):
        command[argument] = value
    else:
        command += [argument, value]
    metadata_path = doc_workspace.root / "data/artifacts/example/metadata.json"
    original_metadata = metadata_path.read_bytes()

    result = doc_workspace.invoke(command, check=False)

    assert result.exit_code == 2, result.output
    field = (
        {1: "map_path", 2: "structure_path"}[argument]
        if isinstance(argument, int)
        else argument.removeprefix("--").replace("-", "_")
    )
    assert f"pipeline.{field}" in result.output
    if value.startswith("s3://"):
        assert "local" in result.output and "filesystem" in result.output
    else:
        assert "file path" in result.output
    assert not doc_workspace.adapter_calls
    assert metadata_path.read_bytes() == original_metadata


def test_cli_batch_example_produces_two_successful_training_artifacts(doc_workspace):
    doc_workspace.invoke(_batch_command())
    paths = _assert_complete_batch(doc_workspace)
    assert {Path(path).name for path in paths} == {
        "emd_1234_1abc",
        "emd_5678_2def",
    }
    assert not doc_workspace.adapter_calls
    for artifact_reference in paths:
        artifact_mapping = _assert_default_artifact(artifact_reference)
        assert artifact_mapping["density_grid"].shape == (128, 128, 128)
        assert artifact_mapping["voxel_size"] == 1.0
    assert len(doc_workspace.http_calls) == 4
    assert not (doc_workspace.root / "data/summary.csv.checkpoint.json").exists()
    assert not (doc_workspace.root / "data/summary.csv.checkpoint.jsonl").exists()


def test_python_batch_example_uses_manifest_relative_input_paths(doc_workspace):
    doc_workspace.prepare_manifest()
    namespace = get_example("guide-batch-python").execute()
    paths = _assert_complete_batch(doc_workspace)
    assert {str(Path(path).resolve()) for path in namespace["paths"]} == {
        str(Path(path).resolve()) for path in paths
    }
    assert not doc_workspace.adapter_calls
    for artifact_reference in paths:
        _assert_default_artifact(artifact_reference)


def test_cli_embedding_options_generate_and_attach_real_sidecars(doc_workspace):
    doc_workspace.invoke(_batch_command() + _options("guide-embeddings-options"))
    assert doc_workspace.generator_calls == [
        {
            "structures_dir": "data/structures",
            "sequence_embeddings_dir": "data/embeddings",
            "device": "cpu",
            "max_chain_length": 1000,
            "num_workers": 2,
        }
    ]
    paths = _assert_complete_batch(doc_workspace)
    for path in paths:
        name = Path(path).name.rsplit("_", 1)[1]
        with np.load(doc_workspace.root / f"data/embeddings/{name}.npz") as sidecar:
            actual = load_preprocessed_artifact(path)
            np.testing.assert_array_equal(actual["sequence"], sidecar["sequence"])
            assert actual["sequence_mask"].all()
        stats = validate_preprocessed_artifact(path, require_sequence_embeddings=True)
        assert stats.sequence_length > 0
        assert stats.sequence_feature_dim > 0


def test_python_embedding_generator_example_uses_documented_signature(doc_workspace):
    namespace = get_example("guide-embeddings-python").execute()
    assert namespace["config"].pipeline.prepare_sequence_embeddings is True
    assert doc_workspace.generator_calls == [
        {
            "structures_dir": "data/structures",
            "sequence_embeddings_dir": "data/embeddings",
            "device": None,
            "max_chain_length": 1000,
            "num_workers": 1,
        }
    ]
    assert doc_workspace.adapter_calls == []
    rows = read_manifest_csv(doc_workspace.root / "data/manifest.csv")
    assert len(rows) == 2
    for row in rows:
        sidecar = doc_workspace.root / "data" / row["sequence_embedding_path"]
        with np.load(sidecar) as values:
            assert values["sequence"].ndim == 2
            assert np.isfinite(values["sequence"]).all()


@pytest.mark.parametrize("split_strategy", ["artifact", "component"])
def test_split_options_produce_disjoint_nonempty_datasets(
    doc_workspace, split_strategy
):
    options = _options("guide-split-options")
    if split_strategy == "component":
        options += _options("guide-group-options")
    doc_workspace.invoke(_batch_command() + options)
    train = set(map(str, read_artifact_list(doc_workspace.root / "data/train.txt")))
    valid = set(map(str, read_artifact_list(doc_workspace.root / "data/valid.txt")))
    assert train and valid
    assert train.isdisjoint(valid)
    assert len(train | valid) == 2
    summary = _summary(doc_workspace.root)
    assert {row["split"] for row in summary} == {"train", "valid"}
    if split_strategy == "component":
        report = json.loads((doc_workspace.root / "data/split.json").read_text())
        assert report["schema_version"] == 3
        assert report["split_strategy"] == "component"
        assert report["num_records"] == 2
        assert report["num_train_records"] == report["num_validation_records"] == 1
        holdouts = set(report["selected_holdout_group_ids"])
        assert holdouts
        assert {
            str(Path(record["artifact_reference"]).resolve())
            for record in report["assignments"]
        } == train | valid
        for record in report["assignments"]:
            if record["split"] == "train":
                assert holdouts.isdisjoint(record["component_group_ids"])


def test_replay_options_preserve_report_selection_and_assignments(doc_workspace):
    doc_workspace.invoke(
        _batch_command()
        + _options("guide-split-options")
        + _options("guide-group-options")
    )
    train_before = (doc_workspace.root / "data/train.txt").read_text()
    valid_before = (doc_workspace.root / "data/valid.txt").read_text()
    original_report = json.loads((doc_workspace.root / "data/split.json").read_text())
    doc_workspace.invoke(
        _batch_command()
        + _options("guide-split-options")
        + _options("guide-replay-options")
    )
    replayed_report = json.loads(
        (doc_workspace.root / "data/split-replayed.json").read_text()
    )
    assert (
        replayed_report["selected_holdout_group_ids"]
        == original_report["selected_holdout_group_ids"]
    )
    assert replayed_report["assignments"] == original_report["assignments"]
    assert (doc_workspace.root / "data/train.txt").read_text() == train_before
    assert (doc_workspace.root / "data/valid.txt").read_text() == valid_before
    assert not doc_workspace.adapter_calls


def test_remote_batch_options_upload_artifacts_and_release_local_staging(doc_workspace):
    doc_workspace.invoke(_batch_command() + _options("guide-remote-options"))
    paths = _assert_complete_batch(doc_workspace, remote=True)
    assert not doc_workspace.adapter_calls
    for path in paths:
        payload = load_preprocessed_artifact(path)
        assert payload["density_grid"].ndim == 3
    assert not list((doc_workspace.root / "data/scratch").rglob("metadata.json"))
    assert doc_workspace.s3.close_count > 0


@pytest.mark.parametrize("example_id", ["guide-validate", "readme-inspect"])
def test_validation_examples_check_real_local_and_remote_training_artifacts(
    doc_workspace, example_id
):
    doc_workspace.seed_training("data/artifacts/example")
    doc_workspace.seed_training("data/artifacts/emd_1001_1abc")
    doc_workspace.seed_training("s3://my-bucket/artifacts/example")
    validations = []
    for command in get_example(example_id).commands():
        result = doc_workspace.invoke(command)
        if command[0] == "config":
            assert OmegaConf.create(result.output).pipeline.name == "preprocess"
        else:
            payload = json.loads(result.output)
            validations.append(payload)
            assert payload["protein_nodes"] > 0
            assert payload["component_nodes"] > 0
    assert validations
    if example_id == "guide-validate":
        assert len(validations) == 2
        assert validations[-1]["sequence_length"] > 0
        assert validations[-1]["sequence_feature_dim"] > 0


def test_environment_example_lists_supported_credential_settings():
    values = dotenv_values(stream=StringIO(get_example("guide-env").code))
    assert set(values) == {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "CRYOFLAKE_S3_ENDPOINT_URL",
        "CRYOFLAKE_S3_REGION",
    }
    assert values["CRYOFLAKE_S3_ENDPOINT_URL"].startswith("https://")


def test_required_embedding_validation_example_rejects_absent_sequence(doc_workspace):
    reference = "s3://my-bucket/artifacts/example"
    doc_workspace.seed_training(reference, include_sequence=False)
    assert validate_preprocessed_artifact(reference).sequence_length is None
    commands = get_example("guide-validate").commands()
    (remote_command,) = [command for command in commands if reference in command]
    result = doc_workspace.invoke(remote_command, check=False)
    assert result.exit_code == 1, result.output
    assert "canonical preprocessing contract" in result.output


def test_explicit_environment_file_example_loads_selected_file(
    doc_workspace, monkeypatch
):
    selected = doc_workspace.root / "credentials.env"
    selected.write_text("CRYOFLAKE_S3_REGION=documentation-region\n")
    example = get_example("guide-env-file")
    assignment = shlex.split(example.code)[0]
    name, placeholder = assignment.split("=", 1)
    assert name == "CRYOFLAKE_ENV_FILE"
    monkeypatch.setenv(name, str(selected))
    for command in example.commands(replacements={placeholder: str(selected)}):
        result = doc_workspace.invoke(command)
    assert OmegaConf.create(result.output).pipeline.name == "manifest"
    assert os.environ["CRYOFLAKE_S3_REGION"] == "documentation-region"


def test_configuration_inspection_examples_resolve_groups_and_interpolation(
    doc_workspace,
):
    configs = [
        OmegaConf.create(doc_workspace.invoke(command).output)
        for command in get_example("config-inspect").commands()
    ]
    assert [config.pipeline.name for config in configs] == [
        "manifest",
        "preprocess",
        "validate",
        "manifest",
    ]
    assert configs[-1].pipeline.workers == 4
    assert configs[-1].pipeline.preprocess_workers == 4
    assert all(config.contracts.artifact.format_version == 1 for config in configs)


def test_project_configuration_examples_compose_and_execute(doc_workspace):
    commands = get_example("config-project").commands()
    config = OmegaConf.create(doc_workspace.invoke(commands[0]).output)
    assert config.pipeline.workers == 4
    assert config.pipeline.source_csv == "data/source.csv"
    for command in commands[1:]:
        doc_workspace.invoke(command)
    assert len(read_manifest_csv(doc_workspace.root / "data/manifest.csv")) == 2


def test_custom_yaml_group_example_composes_with_packaged_defaults(doc_workspace):
    group = doc_workspace.root / "configs/pipeline/lab.yaml"
    group.write_text(get_example("config-lab").code)
    result = doc_workspace.invoke(
        ["config", "--config-dir", "configs", "-o", "pipeline=lab"]
    )
    config = OmegaConf.create(result.output)
    assert config.pipeline.name == "manifest"
    assert config.pipeline.workers == 4
    assert config.pipeline.source_csv == "data/source.csv"
    assert config.pipeline.manifest_csv == "data/manifest.csv"
    assert config.contracts.artifact.format_version == 1


def test_python_configuration_example_validates_and_runs(doc_workspace):
    namespace = get_example("config-python").execute()
    assert namespace["config"].pipeline.workers == 4
    entries = namespace["entries"]
    assert [(entry.emdb_id, entry.wwpdb_id) for entry in entries] == [
        ("1234", "1abc"),
        ("5678", "2def"),
    ]
    assert len(read_manifest_csv(doc_workspace.root / "data/manifest.csv")) == 2
