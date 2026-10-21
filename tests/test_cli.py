from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest
from omegaconf import OmegaConf
from typer import rich_utils
from typer.testing import CliRunner

from cryoflake import cli
from cryoflake.storage.artifacts import ArtifactValidationResult

runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "160"})


@pytest.fixture(autouse=True)
def stable_cli_rendering(monkeypatch):
    # Typer captures CI color flags at import time. Forced terminal rendering
    # makes TERM=dumb use 80 columns regardless of COLUMNS and can add ANSI
    # styles within option names. Keep help/error assertions independent of it.
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", False)
    monkeypatch.setattr(rich_utils, "MAX_WIDTH", 160)


def _install_adapter(monkeypatch, function):
    module = ModuleType("_cryoflake_test_adapter")
    module.run = function
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return f"{module.__name__}:run"


def _source_csv(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("EMD-,Associated PDB,resolution\n1234,1ABC,2.5\n")
    return source


def test_help_lists_typer_commands_and_configuration_options():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert all(
        name in result.output
        for name in ("validate-artifact", "prepare-manifest", "preprocess", "config")
    )
    result = runner.invoke(cli.app, ["prepare-manifest", "--help"])
    assert result.exit_code == 0
    assert all(
        option in result.output
        for option in (
            "--preprocess-adapter",
            "--artifact-output-dir",
            "--source-csv",
            "--artifact-s3-prefix",
            "--reset-preprocess-checkpoint",
            "--config-dir",
            "--override",
        )
    )
    assert "--manifest " not in result.output
    assert "--preprocess-output-remote-prefix" not in result.output


@pytest.mark.parametrize(
    "option",
    [
        "--manifest",
        "--preprocess-output-remote-prefix",
        "--force",
        "--no-force",
        "--split-report",
    ],
)
def test_prepare_manifest_rejects_removed_option_aliases(option):
    result = runner.invoke(cli.app, ["prepare-manifest", option, "input"])
    assert result.exit_code == 2
    assert "No such option" in result.output


def test_validate_artifact_prints_machine_readable_result(monkeypatch):
    calls = []

    def fake_validate(reference, **kwargs):
        calls.append((reference, kwargs))
        return ArtifactValidationResult(
            artifact_reference=str(reference),
            grid_shape=(4, 5, 6),
            protein_nodes=7,
            nucleotide_nodes=8,
            component_nodes=9,
            sequence_length=10,
            sequence_feature_dim=11,
        )

    monkeypatch.setattr(
        "cryoflake.storage.artifacts.validate_preprocessed_artifact", fake_validate
    )
    result = runner.invoke(
        cli.app,
        [
            "validate-artifact",
            "s3://bucket/example",
            "--require-sequence-embeddings",
            "--s3-endpoint-url",
            "https://objects.example.test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "s3://bucket/example",
            {
                "require_sequence_embeddings": True,
                "s3_endpoint_url": "https://objects.example.test",
            },
        )
    ]
    assert json.loads(result.output)["grid_shape"] == [4, 5, 6]


def test_pipeline_config_controls_sequence_requirement(monkeypatch):
    calls = []

    def validate(reference, **kwargs):
        calls.append(kwargs)
        return ArtifactValidationResult(reference, (1, 1, 1), 0, 0, 0, None, None)

    monkeypatch.setattr(
        "cryoflake.storage.artifacts.validate_preprocessed_artifact", validate
    )
    result = runner.invoke(
        cli.app,
        [
            "validate-artifact",
            "example",
            "-o",
            "pipeline.require_sequence_embeddings=true",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["require_sequence_embeddings"] is True


def test_obsolete_contract_sequence_option_fails_before_artifact_read(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "cryoflake.storage.artifacts.validate_preprocessed_artifact",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    result = runner.invoke(
        cli.app,
        [
            "validate-artifact",
            "example",
            "-o",
            "+contracts.artifact.require_sequence_embeddings=true",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "require_sequence_embeddings" in result.output
    assert calls == []


def test_preprocess_loads_adapter_and_forwards_flags(monkeypatch):
    calls = []
    validation_calls = []
    adapter = _install_adapter(monkeypatch, lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        "cryoflake.storage.artifacts.validate_preprocessed_artifact",
        lambda *args, **kwargs: validation_calls.append((args, kwargs)),
    )
    result = runner.invoke(
        cli.app,
        [
            "preprocess",
            "map.mrc",
            "model.cif",
            "artifact",
            "--preprocess-adapter",
            adapter,
            "--original-structure-path",
            "original.cif",
            "--fasta-path",
            "sequence.fasta",
            "--no-fix-shift",
            "--interface-contact-cutoff",
            "4.5",
            "--s3-endpoint-url",
            "https://objects.example.test",
            "-o",
            "pipeline.target_voxel_size=1.25",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "map_path": "map.mrc",
            "structure_path": "model.cif",
            "artifact_reference": "artifact",
            "target_voxel_size": 1.25,
            "original_structure_path": "original.cif",
            "fasta_path": "sequence.fasta",
            "fix_shift": False,
            "s3_endpoint_url": "https://objects.example.test",
            "interface_contact_cutoff": 4.5,
        }
    ]
    assert validation_calls == [
        (
            ("artifact",),
            {
                "require_sequence_embeddings": False,
                "s3_endpoint_url": "https://objects.example.test",
            },
        )
    ]


def test_preprocess_accepts_paths_and_adapter_entirely_from_yaml_overrides(monkeypatch):
    calls = []
    adapter = _install_adapter(monkeypatch, lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        "cryoflake.storage.artifacts.validate_preprocessed_artifact",
        lambda *args, **kwargs: None,
    )
    result = runner.invoke(
        cli.app,
        [
            "preprocess",
            "-o",
            "pipeline.map_path=map.mrc",
            "-o",
            "pipeline.structure_path=model.cif",
            "-o",
            "pipeline.artifact_reference=example",
            "-o",
            f"pipeline.preprocess_adapter={adapter}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["artifact_reference"] == "example"


@pytest.mark.parametrize("damage", ["missing", "storage", "contract"])
def test_preprocess_fails_when_adapter_does_not_create_a_valid_artifact(
    monkeypatch, tmp_path, damage
):
    import numpy as np

    from cryoflake.storage.store import write_preprocessed_artifact

    output = tmp_path / "artifact"

    def invalid_adapter(**kwargs):
        if damage == "storage":
            output.mkdir()
            (output / "metadata.json").write_text("invalid metadata")
        elif damage == "contract":
            write_preprocessed_artifact(output, {"density_grid": np.zeros((2, 2, 2))})

    adapter = _install_adapter(monkeypatch, invalid_adapter)
    result = runner.invoke(
        cli.app,
        [
            "preprocess",
            "map.mrc",
            "model.cif",
            str(output),
            "--preprocess-adapter",
            adapter,
        ],
    )
    assert result.exit_code == 1, result.output
    assert "Error:" in result.output
    assert "Traceback" not in result.output
    if damage == "contract":
        assert "canonical preprocessing contract" in result.output


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".prot", ".npy", ".npz"])
@pytest.mark.parametrize("remote", [False, True])
def test_preprocess_rejects_single_file_output_before_adapter(
    monkeypatch, tmp_path, suffix, remote
):
    calls = []
    adapter = _install_adapter(monkeypatch, lambda **kwargs: calls.append(kwargs))
    output = (
        f"s3://bucket/artifact{suffix}"
        if remote
        else str(tmp_path / f"artifact{suffix}")
    )
    result = runner.invoke(
        cli.app,
        [
            "preprocess",
            "map.mrc",
            "model.cif",
            output,
            "--preprocess-adapter",
            adapter,
        ],
    )
    assert result.exit_code == 2, result.output
    assert "pipeline.artifact_reference" in result.output
    plain_output = " ".join(result.output.replace("│", "").split())
    assert "single-file artifacts are unsupported" in plain_output
    assert calls == []
    assert not list(tmp_path.iterdir())


def test_preprocess_rejects_existing_file_output_before_adapter(monkeypatch, tmp_path):
    calls = []
    adapter = _install_adapter(monkeypatch, lambda **kwargs: calls.append(kwargs))
    output = tmp_path / "artifact.bin"
    output.write_bytes(b"standalone artifact")
    result = runner.invoke(
        cli.app,
        [
            "preprocess",
            "map.mrc",
            "model.cif",
            str(output),
            "--preprocess-adapter",
            adapter,
        ],
    )
    assert result.exit_code == 2, result.output
    assert "must be a directory" in result.output
    assert calls == []
    assert output.read_bytes() == b"standalone artifact"


@pytest.mark.parametrize(
    "adapter", ["missing-separator", "not_a_real_module:run", "sys:version"]
)
def test_preprocess_rejects_invalid_adapter_without_traceback(adapter):
    result = runner.invoke(
        cli.app,
        ["preprocess", "map", "model", "artifact", "--preprocess-adapter", adapter],
    )
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_prepare_manifest_runs_manifest_generation_without_adapter(tmp_path):
    output = tmp_path / "manifest.csv"
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            str(_source_csv(tmp_path)),
            "--manifest-csv",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "emd_1234.map.gz" in output.read_text()


def test_prepare_manifest_requires_explicit_source_before_io(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        cli.app, ["prepare-manifest", "--manifest-csv", "manifest.csv"]
    )
    assert result.exit_code == 2, result.output
    assert "pipeline.source_csv is required" in result.output
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "options, message",
    [
        (["--artifact-output-dir", "artifacts"], "MODULE:CALLABLE"),
        (["--download"], "maps_dir"),
        (["--workers", "0"], "positive integer"),
        (["--target-voxel-size", "nan"], "positive finite"),
        (["--validation-fraction", "1"], "less than 1"),
        (
            ["--split-strategy", "example"],
            "must be artifact, component, or interface",
        ),
        (["--split-strategy", "component"], "validation_artifact_list_path"),
        (["--artifact-s3-prefix", "s3://bucket/key"], "scratch storage"),
        (
            ["-o", "contracts.artifact.format_version=2"],
            "Unsupported artifact contract",
        ),
    ],
)
def test_invalid_options_fail_before_manifest_side_effects(tmp_path, options, message):
    output = tmp_path / "manifest.csv"
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            str(tmp_path / "missing.csv"),
            "--manifest-csv",
            str(output),
            *options,
        ],
    )
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert not output.exists()


def test_prepare_manifest_forwards_loaded_adapter_and_worker_overrides(
    monkeypatch, tmp_path
):
    adapter = _install_adapter(monkeypatch, lambda **kwargs: None)
    calls = []

    def fake_prepare(config, *, preprocess_adapter=None, **kwargs):
        calls.append((config, preprocess_adapter, kwargs))

    monkeypatch.setattr("cryoflake.preprocess.prepare.run", fake_prepare)
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            "source.csv",
            "--manifest-csv",
            "manifest.csv",
            "--preprocess-adapter",
            adapter,
            "--artifact-output-dir",
            str(tmp_path / "scratch"),
            "--artifact-s3-prefix",
            "s3://bucket/artifacts",
            "--workers",
            "3",
            "-o",
            "pipeline.workers=2",
            "-o",
            "pipeline.upload_workers=4",
        ],
    )
    assert result.exit_code == 0, result.output
    config, function, _ = calls[0]
    assert config.pipeline.workers == 3  # Explicit flags take precedence.
    assert config.pipeline.upload_workers == 4
    assert config.pipeline.artifact_s3_prefix == "s3://bucket/artifacts"
    assert callable(function)


def test_custom_yaml_directory_and_manifest_contract_are_consumed(tmp_path):
    import cryoflake.configs

    config_dir = tmp_path / "configs"
    shutil.copytree(Path(cryoflake.configs.__file__).parent, config_dir)
    (config_dir / "pipeline" / "manifest.yaml").write_text(
        "defaults: [base, _self_]\nname: manifest\ninclude_without_structure: true\n"
    )
    (config_dir / "contracts" / "manifest" / "emdb.yaml").write_text(
        "required_source_columns: [EMD-, Associated PDB, approved]\n"
    )
    output = tmp_path / "manifest.csv"
    source = _source_csv(tmp_path)
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--config-dir",
            str(config_dir),
            "--source-csv",
            str(source),
            "--manifest-csv",
            str(output),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "approved" in result.output
    assert not output.exists()
    source.write_text("EMD-,Associated PDB,approved\n1234,,yes\n")
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--config-dir",
            str(config_dir),
            "--source-csv",
            str(source),
            "--manifest-csv",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1234" in output.read_text()


def test_config_command_prints_resolved_yaml_without_running_pipeline():
    result = runner.invoke(
        cli.app,
        ["config", "--pipeline", "preprocess", "-o", "pipeline.target_voxel_size=1.5"],
    )
    assert result.exit_code == 0, result.output
    config = OmegaConf.create(result.output)
    assert config.pipeline.name == "preprocess"
    assert config.pipeline.target_voxel_size == 1.5
    assert config.contracts.artifact.format_name == "cryoflake-preprocessed"


@pytest.mark.parametrize("selection", ["directory", "name"])
def test_custom_root_pipeline_is_shared_by_inspection_and_execution(
    monkeypatch, tmp_path, selection
):
    import cryoflake.configs
    from cryoflake.preprocess.prepare import run

    config_dir = tmp_path / "configs"
    shutil.copytree(Path(cryoflake.configs.__file__).parent, config_dir)
    config_name = "config" if selection == "directory" else "production"
    root_yaml = (
        (config_dir / "config.yaml")
        .read_text()
        .replace("pipeline: manifest", "pipeline: lab")
    )
    (config_dir / f"{config_name}.yaml").write_text(root_yaml)
    source = _source_csv(tmp_path)
    manifest = tmp_path / "results" / "manifest.csv"
    (config_dir / "pipeline" / "lab.yaml").write_text(
        "defaults: [base, _self_]\nname: manifest\nworkers: 9\n"
        f"source_csv: {source}\nmanifest_csv: {manifest}\n"
    )
    if selection == "directory":
        arguments = ["--config-dir", str(config_dir)]
    else:
        # Resolve a named packaged root from the isolated test resources.
        compose = cli.compose_config

        def compose_named_root(*args, **kwargs):
            assert kwargs["config_dir"] is None
            kwargs["config_dir"] = config_dir
            return compose(*args, **kwargs)

        monkeypatch.setattr(cli, "compose_config", compose_named_root)
        arguments = ["--config-name", config_name]
    inspected = runner.invoke(cli.app, ["config", *arguments])
    assert inspected.exit_code == 0, inspected.output
    inspected_config = OmegaConf.create(inspected.output)
    seen = []

    def prepare(config, **kwargs):
        seen.append(OmegaConf.to_container(config.pipeline, resolve=True))
        return run(config, **kwargs)

    monkeypatch.setattr("cryoflake.preprocess.prepare.run", prepare)
    executed = runner.invoke(cli.app, ["prepare-manifest", *arguments])
    assert executed.exit_code == 0, executed.output
    assert seen == [OmegaConf.to_container(inspected_config.pipeline, resolve=True)]
    assert seen[0]["workers"] == 9
    assert "emd_1234.map.gz" in manifest.read_text()


def test_explicit_inspection_pipeline_flag_wins_over_group_override():
    result = runner.invoke(
        cli.app, ["config", "--pipeline", "preprocess", "-o", "pipeline=validate"]
    )
    assert result.exit_code == 0, result.output
    assert OmegaConf.create(result.output).pipeline.name == "preprocess"


@pytest.mark.parametrize("pipeline_name", ["preprocess", "validate"])
def test_execution_rejects_wrong_custom_root_pipeline_before_io(
    tmp_path, pipeline_name
):
    import cryoflake.configs

    config_dir = tmp_path / "configs"
    shutil.copytree(Path(cryoflake.configs.__file__).parent, config_dir)
    root = config_dir / "config.yaml"
    root.write_text(root.read_text().replace("pipeline: manifest", "pipeline: lab"))
    output = tmp_path / "results" / "manifest.csv"
    (config_dir / "pipeline" / "lab.yaml").write_text(
        f"defaults: [base, _self_]\nname: {pipeline_name}\n"
        f"source_csv: {tmp_path / 'missing.csv'}\nmanifest_csv: {output}\n"
    )
    result = runner.invoke(
        cli.app, ["prepare-manifest", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 2, result.output
    assert "This command requires pipeline.name=manifest" in result.output
    assert f"pipeline.name={pipeline_name}" in result.output
    assert not output.parent.exists()


@pytest.mark.parametrize("reference_kind", ["cwd", "cwd_metadata", "root"])
def test_preprocess_rejects_unsafe_artifact_root_before_adapter_io(
    monkeypatch, tmp_path, reference_kind
):
    monkeypatch.chdir(tmp_path)
    artifact_reference = {
        "cwd": ".",
        "cwd_metadata": "metadata.json",
        "root": tmp_path.anchor,
    }[reference_kind]
    calls = []
    adapter = _install_adapter(monkeypatch, lambda **kwargs: calls.append(kwargs))
    result = runner.invoke(
        cli.app,
        [
            "preprocess",
            "map.mrc",
            "model.cif",
            artifact_reference,
            "--preprocess-adapter",
            adapter,
        ],
    )
    assert result.exit_code == 2, result.output
    assert "dedicated subdirectory" in result.output
    assert calls == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "field", ["map_path", "structure_path", "original_structure_path", "fasta_path"]
)
def test_preprocess_rejects_remote_raw_inputs_before_adapter_or_output(
    monkeypatch, tmp_path, field
):
    calls = []
    adapter = _install_adapter(monkeypatch, lambda **kwargs: calls.append(kwargs))
    output = tmp_path / "artifact"
    arguments = ["map.mrc", "structure.cif", str(output)]
    if field in {"map_path", "structure_path"}:
        arguments[0 if field == "map_path" else 1] = "s3://bucket/raw-input"
    else:
        arguments += [f"--{field.replace('_', '-')}", "s3://bucket/raw-input"]
    result = runner.invoke(
        cli.app, ["preprocess", *arguments, "--preprocess-adapter", adapter]
    )
    assert result.exit_code == 2, result.output
    assert f"pipeline.{field}" in result.output
    assert "local filesystem" in result.output
    assert calls == []
    assert not output.exists()


def test_artifact_reuse_files_fail_before_manifest_or_adapter_io(monkeypatch, tmp_path):
    calls = []
    adapter = _install_adapter(monkeypatch, lambda **kwargs: calls.append(kwargs))
    reuse_file = tmp_path / "file.bin"
    reuse_file.write_bytes(b"not a search directory")
    output = tmp_path / "output"
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            str(tmp_path / "missing.csv"),
            "--manifest-csv",
            str(output / "manifest.csv"),
            "--preprocess-adapter",
            adapter,
            "--artifact-output-dir",
            str(output / "artifacts"),
            "--artifact-reuse-dir",
            str(reuse_file),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "artifact_reuse_dirs" in result.output
    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize("command", ["preprocess", "prepare-manifest"])
@pytest.mark.parametrize(
    "composed, flag, expected",
    [
        (False, None, False),
        (False, "--fix-shift", True),
        (True, "--no-fix-shift", False),
    ],
)
def test_canonical_shift_flags_preserve_or_override_composed_policy(
    monkeypatch, command, composed, flag, expected
):
    calls = []
    adapter = _install_adapter(monkeypatch, lambda **kwargs: calls.append(kwargs))
    if command == "preprocess":
        arguments = [
            "map.mrc",
            "model.cif",
            "artifact",
            "--preprocess-adapter",
            adapter,
        ]
        monkeypatch.setattr(
            "cryoflake.storage.artifacts.validate_preprocessed_artifact",
            lambda *args, **kwargs: None,
        )
    else:
        arguments = ["--source-csv", "source.csv", "--manifest-csv", "manifest.csv"]

        def prepare(config, **kwargs):
            calls.append({"fix_shift": config.pipeline.fix_shift})

        monkeypatch.setattr("cryoflake.preprocess.prepare.run", prepare)
    result = runner.invoke(
        cli.app,
        [
            command,
            *arguments,
            "-o",
            f"pipeline.fix_shift={str(composed).lower()}",
            *([flag] if flag else []),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["fix_shift"] is expected


@pytest.mark.parametrize(
    "composed, flag, expected",
    [
        (True, None, True),
        (False, "--force-download", True),
        (True, "--no-force-download", False),
    ],
)
def test_canonical_download_force_flags_reach_downloader(
    monkeypatch, tmp_path, composed, flag, expected
):
    calls = []

    def download(entries, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(
        "cryoflake.preprocess.prepare.download_manifest_entries", download
    )
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            str(_source_csv(tmp_path)),
            "--manifest-csv",
            str(tmp_path / "manifest.csv"),
            "--maps-dir",
            str(tmp_path / "maps"),
            "--structures-dir",
            str(tmp_path / "structures"),
            "--download",
            "-o",
            f"pipeline.force_download={str(composed).lower()}",
            *([flag] if flag else []),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["force_download"] is expected


def test_runtime_errors_are_concise_and_nonzero(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("corrupt artifact")

    monkeypatch.setattr(
        "cryoflake.storage.artifacts.validate_preprocessed_artifact", fail
    )
    result = runner.invoke(cli.app, ["validate-artifact", "broken"])
    assert result.exit_code == 1
    assert "corrupt artifact" in result.output
    assert "Traceback" not in result.output


def test_package_metadata_registers_typer_cli():
    import tomllib

    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert metadata["project"]["scripts"]["cryoflake"] == "cryoflake.cli:main"
    dependencies = " ".join(metadata["project"]["dependencies"])
    assert "hydra-core" in dependencies
    assert "typer" in dependencies


def test_download_overrides_can_be_completed_with_explicit_directory_flags(
    monkeypatch, tmp_path
):
    calls = []

    def download(entries, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(
        "cryoflake.preprocess.prepare.download_manifest_entries", download
    )
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            str(_source_csv(tmp_path)),
            "--manifest-csv",
            str(tmp_path / "manifest.csv"),
            "--maps-dir",
            str(tmp_path / "maps"),
            "--structures-dir",
            str(tmp_path / "structures"),
            "-o",
            "pipeline.download=true",
            "-o",
            "pipeline.workers=3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["num_workers"] == 3


def test_manifest_runner_uses_injected_adapter_and_pipeline_requirements(
    monkeypatch, tmp_path
):
    from cryoflake.config import compose_config
    from cryoflake.preprocess.prepare import run

    calls = []

    def adapter(**kwargs):
        return None

    def preprocess(manifest, **kwargs):
        calls.append((manifest, kwargs))
        return []

    monkeypatch.setattr(
        "cryoflake.preprocess.prepare.preprocess_manifest_csv", preprocess
    )
    config = compose_config(
        [
            "pipeline.workers=2",
            "pipeline.preprocess_workers=3",
            "pipeline.upload_workers=4",
            "pipeline.require_sequence_embeddings=true",
        ],
        values={
            "pipeline": {
                "source_csv": str(_source_csv(tmp_path)),
                "manifest_csv": str(tmp_path / "manifest.csv"),
                "artifact_output_dir": str(tmp_path / "artifacts"),
            }
        },
    )
    assert run(config, preprocess_adapter=adapter, progress_callback=None)
    _, arguments = calls[0]
    assert arguments["preprocess_adapter"] is adapter
    assert arguments["num_workers"] == 3
    assert arguments["upload_workers"] == 4
    assert arguments["require_sequence_embeddings"] is True
    assert (
        arguments["artifact_list_path"] == tmp_path / "artifacts" / "artifact_list.txt"
    )
    assert arguments["summary_csv"] == tmp_path / "manifest_summary.csv"


def test_sequence_generation_requires_paths_without_side_effects(
    tmp_path,
):
    output = tmp_path / "manifest.csv"
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--manifest-csv",
            str(output),
            "--prepare-sequence-embeddings",
        ],
    )
    assert result.exit_code == 2
    assert "sequence_embeddings_dir" in result.output
    assert not output.exists()


def test_sequence_generation_requires_explicit_adapter_before_io(tmp_path):
    output = tmp_path / "manifest.csv"
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            str(tmp_path / "missing.csv"),
            "--manifest-csv",
            str(output),
            "--structures-dir",
            "structures",
            "--sequence-embeddings-dir",
            "embeddings",
            "--prepare-sequence-embeddings",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "--sequence-embedding-generator MODULE:CALLABLE" in result.output
    assert not output.exists()


@pytest.mark.parametrize("suffix", [".prot", ".pkl", ".pickle", ".json"])
def test_unsupported_embedding_suffix_is_rejected_before_generator_or_io(
    monkeypatch, tmp_path, suffix
):
    calls = []
    adapter = _install_adapter(monkeypatch, lambda **kwargs: calls.append(kwargs))
    output_dir = tmp_path / "output"
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            str(tmp_path / "missing.csv"),
            "--manifest-csv",
            str(output_dir / "manifest.csv"),
            "--structures-dir",
            str(tmp_path / "structures"),
            "--sequence-embeddings-dir",
            str(output_dir / "embeddings"),
            "--sequence-embedding-suffix",
            suffix,
            "--prepare-sequence-embeddings",
            "--sequence-embedding-generator",
            adapter,
        ],
    )
    assert result.exit_code == 2, result.output
    assert "sequence_embedding_suffix must be .npy or .npz" in result.output
    assert calls == []
    assert not output_dir.exists()


def test_sequence_generation_cli_loads_adapter_from_hydra(monkeypatch, tmp_path):
    calls = []

    def generate(
        *,
        structures_dir,
        sequence_embeddings_dir,
        device,
        max_chain_length,
        num_workers,
    ):
        calls.append(
            {
                "structures_dir": structures_dir,
                "sequence_embeddings_dir": sequence_embeddings_dir,
                "device": device,
                "max_chain_length": max_chain_length,
                "num_workers": num_workers,
            }
        )

    adapter = _install_adapter(monkeypatch, generate)
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            str(_source_csv(tmp_path)),
            "--manifest-csv",
            str(tmp_path / "manifest.csv"),
            "--structures-dir",
            "structures",
            "--sequence-embeddings-dir",
            "embeddings",
            "--prepare-sequence-embeddings",
            "--sequence-embedding-device",
            "cpu",
            "--sequence-embedding-workers",
            "2",
            "--sequence-embedding-max-chain-length",
            "512",
            "-o",
            f"pipeline.sequence_embedding_generator={adapter}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "structures_dir": "structures",
            "sequence_embeddings_dir": "embeddings",
            "device": "cpu",
            "max_chain_length": 512,
            "num_workers": 2,
        }
    ]
    assert "1abc.npz" in (tmp_path / "manifest.csv").read_text()


def test_sequence_generation_runner_preserves_phase_order(monkeypatch, tmp_path):
    from cryoflake.config import compose_config
    from cryoflake.preprocess.prepare import run

    phases = []

    def download(entries, **kwargs):
        phases.append("download")
        return {}

    def generate(**kwargs):
        phases.append("sequence")
        assert kwargs == {
            "structures_dir": "structures",
            "sequence_embeddings_dir": "embeddings",
            "device": None,
            "max_chain_length": 1000,
            "num_workers": 1,
        }

    def preprocess(manifest, **kwargs):
        phases.append("preprocess")
        assert "sequence_embedding_path" in Path(manifest).read_text()
        return []

    monkeypatch.setattr(
        "cryoflake.preprocess.prepare.download_manifest_entries", download
    )
    monkeypatch.setattr(
        "cryoflake.preprocess.prepare.preprocess_manifest_csv", preprocess
    )
    config = compose_config(
        values={
            "pipeline": {
                "source_csv": str(_source_csv(tmp_path)),
                "manifest_csv": str(tmp_path / "manifest.csv"),
                "maps_dir": "maps",
                "structures_dir": "structures",
                "sequence_embeddings_dir": "embeddings",
                "download": True,
                "prepare_sequence_embeddings": True,
                "artifact_output_dir": str(tmp_path / "artifacts"),
            }
        }
    )
    entries = run(
        config,
        preprocess_adapter=lambda **kwargs: None,
        sequence_embedding_generator=generate,
        progress_callback=None,
    )
    assert len(entries) == 1
    assert phases == ["download", "sequence", "preprocess"]


def test_manifest_cannot_overwrite_source_csv(tmp_path):
    source = _source_csv(tmp_path)
    original = source.read_bytes()
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            str(source),
            "--manifest-csv",
            str(source),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "distinct files" in result.output
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--artifact-s3-prefix", "s3://bucket/artifacts?query=1"], "query"),
        (["--sequence-embedding-name-template", "{typo}"], "name_template"),
        (["--s3-endpoint-url", "objects.invalid"], "s3_endpoint_url"),
        (["--maps-dir", "s3://bucket/maps"], "local filesystem"),
    ],
)
def test_invalid_configuration_fails_before_adapter_or_output(
    monkeypatch, tmp_path, arguments, message
):
    calls = []
    adapter = _install_adapter(monkeypatch, lambda **kwargs: calls.append(kwargs))
    output_dir = tmp_path / "output"
    source = _source_csv(tmp_path)
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            str(source),
            "--manifest-csv",
            str(output_dir / "manifest.csv"),
            "--artifact-output-dir",
            str(output_dir / "artifacts"),
            "--preprocess-adapter",
            adapter,
            *arguments,
        ],
    )
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "headers", ["EMD-,Associated PDB,EMD-", "EMD-,Associated PDB,"]
)
def test_invalid_source_headers_fail_before_output_creation(tmp_path, headers):
    source = tmp_path / "source.csv"
    source.write_text(f"{headers}\n1234,1abc,2345\n")
    output = tmp_path / "output" / "manifest.csv"
    result = runner.invoke(
        cli.app,
        [
            "prepare-manifest",
            "--source-csv",
            str(source),
            "--manifest-csv",
            str(output),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "unique and nonempty" in result.output
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "callback",
    ["preprocess_adapter", "sequence_embedding_generator", "progress_callback"],
)
def test_manifest_runner_requires_callable_callbacks_before_io(tmp_path, callback):
    from cryoflake.config import compose_config
    from cryoflake.preprocess.prepare import run

    output = tmp_path / "output" / "manifest.csv"
    config = compose_config(
        values={
            "pipeline": {
                "source_csv": str(_source_csv(tmp_path)),
                "manifest_csv": str(output),
            }
        }
    )
    with pytest.raises(ValueError, match=f"{callback} must be callable"):
        run(config, **{callback: object()})
    assert not output.parent.exists()
