"""Optional viewer dispatch, process lifetime, and CLI validation."""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

from cryoflake import visualization
from cryoflake.cli import app


def _bundle(directory):
    directory.mkdir(exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps({"chains": [{"chain_id": "POLYMER_0"}, {"chain_id": "POLYMER_1"}]})
    )
    return SimpleNamespace(
        structure_path=directory / "structure.cif",
        density_path=directory / "density.mrc",
        sequence_path=directory / "sequence.fasta",
        summary_path=directory / "summary.json",
    )


def test_missing_extra_gives_install_guidance_before_artifact_io(monkeypatch):
    original = importlib.import_module

    def without_bottle(name, *args, **kwargs):
        if name == "bottle":
            raise ModuleNotFoundError("No module named 'bottle'", name="bottle")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without_bottle)
    result = CliRunner().invoke(app, ["visualize", "nonexistent-artifact"])
    assert result.exit_code == 1
    assert "uv pip install 'cryoflake[vis]'" in result.output
    assert "Traceback" not in result.output


def test_cli_passes_visualization_options(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        visualization,
        "visualize_artifact",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    result = CliRunner().invoke(
        app,
        [
            "visualize",
            "s3://bucket/artifact/metadata.json",
            "--viewer",
            "chimerax",
            "--visualization-output-dir",
            str(tmp_path),
            "--atom-names-path",
            "names.json",
            "--port",
            "4567",
            "--no-open-browser",
            "--chimerax-path",
            "/gui/ChimeraX",
            "--s3-endpoint-url",
            "https://objects.example.test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        (
            ("s3://bucket/artifact/metadata.json",),
            {
                "viewer": "chimerax",
                "visualization_output_dir": tmp_path,
                "atom_names_path": Path("names.json"),
                "port": 4567,
                "open_browser": False,
                "chimerax_path": Path("/gui/ChimeraX"),
                "s3_endpoint_url": "https://objects.example.test",
            },
        )
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--viewer", "pymol"],
        ["--port", "-1"],
        ["--port", "65536"],
    ],
)
def test_cli_rejects_invalid_viewer_options_before_dispatch(monkeypatch, arguments):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid CLI options must fail before dispatch")

    monkeypatch.setattr(visualization, "visualize_artifact", unexpected)
    result = CliRunner().invoke(app, ["visualize", "artifact", *arguments])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "options",
    [
        {"viewer": "unknown"},
        {"port": True},
        {"port": 1.5},
        {"port": -1},
        {"port": 65536},
        {"chimerax_path": "/executable"},
    ],
)
def test_python_api_rejects_invalid_options_before_dependencies(monkeypatch, options):
    monkeypatch.setattr(
        visualization,
        "_require_visualization_dependencies",
        lambda: pytest.fail("invalid options must fail before dependency checks"),
    )
    with pytest.raises(ValueError):
        visualization.visualize_artifact("artifact", **options)


def test_temporary_exports_live_through_viewer_and_are_cleaned(monkeypatch):
    monkeypatch.setattr(
        visualization, "_require_visualization_dependencies", lambda: None
    )
    from cryoflake.visualization import reconstruction

    directories = []

    def reconstruct(reference, directory, **kwargs):
        assert reference == "s3://bucket/artifact"
        assert kwargs == {"atom_names_path": None, "s3_endpoint_url": None}
        directories.append(directory)
        bundle = _bundle(directory)
        bundle.structure_path.write_text("data_test")
        return bundle

    monkeypatch.setattr(reconstruction, "reconstruct_artifact", reconstruct)

    def serve(bundle, **kwargs):
        assert bundle.structure_path.exists()
        assert kwargs == {"port": 0, "open_browser": True}

    monkeypatch.setattr(visualization, "_serve_molstar", serve)
    visualization.visualize_artifact("s3://bucket/artifact")
    assert len(directories) == 1
    assert not directories[0].exists()


def test_missing_chimerax_fails_before_artifact_or_output_io(monkeypatch, tmp_path):
    monkeypatch.setattr(
        visualization, "_require_visualization_dependencies", lambda: None
    )
    directory = tmp_path / "exports"
    with pytest.raises(ValueError, match="executable"):
        visualization.visualize_artifact(
            "missing-artifact",
            viewer="chimerax",
            chimerax_path=tmp_path / "missing-executable",
            visualization_output_dir=directory,
        )
    assert not directory.exists()


def test_chimerax_discovery_uses_path_and_explicit_override(monkeypatch, tmp_path):
    executable = tmp_path / "ChimeraX"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr(visualization.shutil, "which", lambda name: "/bin/chimerax")
    assert visualization._find_chimerax(None) == "/bin/chimerax"
    assert visualization._find_chimerax(executable) == str(executable)


def test_chimerax_script_quotes_paths_and_waits_for_gui(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path / 'a "quote" and space')
    executable = tmp_path / "ChimeraX"
    executable.touch(mode=0o755)
    calls = []

    def run(arguments, **kwargs):
        script = Path(arguments[1])
        assert script.exists()
        ast.parse(script.read_text())
        calls.append((arguments, kwargs, script.read_text()))

    monkeypatch.setattr(visualization.subprocess, "run", run)
    visualization._launch_chimerax(bundle, chimerax_path=executable)
    assert calls[0][0] == [
        str(executable),
        str(bundle.structure_path.parent / "chimerax.py"),
    ]
    assert calls[0][1] == {"check": True}
    source = calls[0][2]
    assert "StringArg.unparse" in source
    assert "volume #2 style mesh transparency 45" in source
    assert "sequence chain #1/POLYMER_0" in source
    assert "sequence chain #1/POLYMER_1" in source


def test_chimerax_process_failure_propagates(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(visualization, "_find_chimerax", lambda path: "/gui/ChimeraX")

    def failure(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(visualization.subprocess, "run", failure)
    with pytest.raises(subprocess.CalledProcessError):
        visualization._launch_chimerax(bundle, chimerax_path=None)


@pytest.mark.parametrize("chain_ids", [[], ["POLYMER_0", "POLYMER_1"]])
def test_chimerax_startup_script_opens_each_sequence_individually(
    monkeypatch, tmp_path, chain_ids
):
    bundle = _bundle(tmp_path / 'space and "quote"')
    bundle.summary_path.write_text(
        json.dumps({"chains": [{"chain_id": chain_id} for chain_id in chain_ids]})
    )
    monkeypatch.setattr(visualization, "_find_chimerax", lambda path: "/gui/ChimeraX")
    commands = ModuleType("chimerax.core.commands")
    observed_commands = []
    commands.run = lambda session, command: observed_commands.append(command)
    commands.StringArg = SimpleNamespace(unparse=json.dumps)
    monkeypatch.setitem(sys.modules, "chimerax", ModuleType("chimerax"))
    monkeypatch.setitem(sys.modules, "chimerax.core", ModuleType("chimerax.core"))
    monkeypatch.setitem(sys.modules, "chimerax.core.commands", commands)

    def run(arguments, **kwargs):
        script = Path(arguments[1])
        exec(compile(script.read_text(), str(script), "exec"), {"session": object()})

    monkeypatch.setattr(visualization.subprocess, "run", run)
    visualization._launch_chimerax(bundle, chimerax_path=None)
    assert observed_commands[:2] == [
        "open " + json.dumps(str(bundle.structure_path)),
        "open " + json.dumps(str(bundle.density_path)),
    ]
    assert [
        command for command in observed_commands if command.startswith("sequence")
    ] == [f"sequence chain #1/{chain_id}" for chain_id in chain_ids]
    assert observed_commands[-1] == "view"


def test_molstar_prints_actual_url_and_closes_server(monkeypatch, tmp_path, capsys):
    from cryoflake.visualization import server

    events = []

    class FakeServer:
        server_address = ("127.0.0.1", 43621)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("closed")

        def serve_forever(self):
            events.append("serving")
            raise KeyboardInterrupt

    monkeypatch.setattr(
        server, "create_visualization_server", lambda *a, **k: FakeServer()
    )
    monkeypatch.setattr(
        visualization.webbrowser, "open", lambda url: events.append(url)
    )
    visualization._serve_molstar(_bundle(tmp_path), port=0, open_browser=True)
    assert events == ["http://127.0.0.1:43621/", "serving", "closed"]
    assert "http://127.0.0.1:43621/" in capsys.readouterr().out
