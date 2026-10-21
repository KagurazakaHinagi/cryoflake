"""Exercise the wheel installation example in a separate environment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

from .examples import PROJECT_ROOT, get_example

pytestmark = pytest.mark.documentation


def _run(arguments, *, cwd, env=None):
    result = subprocess.run(
        arguments, cwd=cwd, env=env, text=True, capture_output=True, timeout=90
    )
    assert result.returncode == 0, f"{arguments!r}\n{result.stdout}\n{result.stderr}"
    return result


def _wheel_install_arguments(uv: str, command: list[str], wheel: Path) -> list[str]:
    assert command[:3] == ["uv", "pip", "install"]
    placeholders = [token for token in command[3:] if Path(token).name == wheel.name]
    assert len(placeholders) == 1, "The install example must name the built wheel"
    return [
        uv,
        *command[1:3],
        "--offline",
        "--no-deps",
        *(str(wheel) if token == placeholders[0] else token for token in command[3:]),
    ]


def _visualization_install_arguments(
    uv: str, command: list[str], wheel: Path
) -> list[str]:
    assert command[:3] == ["uv", "pip", "install"]
    assert command[3:].count("cryoflake[vis]") == 1
    return [
        uv,
        *command[1:3],
        "--offline",
        "--no-deps",
        *(
            str(wheel) + "[vis]" if token == "cryoflake[vis]" else token
            for token in command[3:]
        ),
    ]


def test_wheel_installation_preserves_all_documented_arguments(tmp_path):
    wheel = tmp_path / "cryoflake-0.1.0-py3-none-any.whl"
    assert _wheel_install_arguments(
        "/bin/uv",
        ["uv", "pip", "install", "--verbose", "/placeholder/" + wheel.name, "--strict"],
        wheel,
    ) == [
        "/bin/uv",
        "pip",
        "install",
        "--offline",
        "--no-deps",
        "--verbose",
        str(wheel),
        "--strict",
    ]


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory):
    output = tmp_path_factory.mktemp("documentation-build")
    uv = shutil.which("uv")
    assert uv is not None, "Documentation installation tests require uv (as does CI)"
    _run([uv, "build", "--offline", "--out-dir", str(output)], cwd=PROJECT_ROOT)
    (wheel,) = output.glob("*.whl")
    (source,) = output.glob("*.tar.gz")
    return uv, wheel, source


@pytest.mark.parametrize("example_id", ["guide-wheel-install", "visualization-install"])
def test_documented_wheel_installation_works_outside_checkout(
    tmp_path, built_distributions, example_id
):
    uv, wheel, _ = built_distributions
    working = tmp_path / "application"
    working.mkdir()
    environment = tmp_path / "environment"
    _run(
        [uv, "venv", "--offline", "--python", sys.executable, str(environment)],
        cwd=working,
    )
    env_file = working / "credentials.env"
    env_file.write_text("")
    child_env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("AWS_", "CRYOFLAKE_"))
    }
    child_env.update(
        VIRTUAL_ENV=str(environment),
        PATH=str(environment / "bin") + os.pathsep + os.environ.get("PATH", ""),
        CRYOFLAKE_ENV_FILE=str(env_file),
        UV_OFFLINE="1",
        # Reuse resolved dependencies but do not process the checkout's editable
        # .pth file. The installed cryoflake must come from the new wheel.
        PYTHONPATH=sysconfig.get_path("purelib"),
    )
    install, help_command = get_example(example_id).shell_commands()
    install_arguments = (
        _visualization_install_arguments(uv, install, wheel)
        if example_id == "visualization-install"
        else _wheel_install_arguments(uv, install, wheel)
    )
    _run(
        install_arguments,
        cwd=working,
        env=child_env,
    )
    help_result = _run(help_command, cwd=working, env=child_env)
    if example_id == "visualization-install":
        assert "artifact_reference" in help_result.stdout
        assert "molstar" in help_result.stdout
        assert "chimerax" in help_result.stdout
    else:
        assert "prepare-manifest" in help_result.stdout

    python = environment / "bin/python"
    probe = _run(
        [
            str(python),
            "-c",
            (
                "import json; import cryoflake; "
                "from cryoflake.config import compose_config; "
                "print(json.dumps({'file': cryoflake.__file__, 'pipelines': "
                "[compose_config(pipeline=name).pipeline.name "
                "for name in ('manifest', 'preprocess', 'validate')]}))"
            ),
        ],
        cwd=working,
        env=child_env,
    )
    loaded = json.loads(probe.stdout)
    assert Path(loaded["file"]).is_relative_to(environment)
    assert loaded["pipelines"] == ["manifest", "preprocess", "validate"]

    # Even a shared dependency directory must not hide the base-install path.
    # Block the optional server import, then exercise the installed CLI before
    # any artifact can be read. Core commands and help remain available.
    missing_extra_probe = _run(
        [
            str(python),
            "-c",
            "\n".join(
                [
                    "import importlib.abc, json, sys",
                    "class WithoutBottle(importlib.abc.MetaPathFinder):",
                    "    def find_spec(self, fullname, path=None, target=None):",
                    "        if fullname == 'bottle':",
                    "            raise ModuleNotFoundError(name='bottle')",
                    "sys.meta_path.insert(0, WithoutBottle())",
                    "from typer.testing import CliRunner",
                    "from cryoflake.cli import app",
                    "runner = CliRunner()",
                    "help_result = runner.invoke(app, ['--help'])",
                    "result = runner.invoke(app, ['visualize', 'missing-artifact'])",
                    "print(json.dumps({'help_exit': help_result.exit_code, "
                    "'exit': result.exit_code, 'output': result.output}))",
                ]
            ),
        ],
        cwd=working,
        env=child_env,
    )
    missing_extra = json.loads(missing_extra_probe.stdout)
    assert missing_extra["help_exit"] == 0
    assert missing_extra["exit"] == 1
    assert "uv pip install 'cryoflake[vis]'" in missing_extra["output"]
    assert "missing-artifact" not in missing_extra["output"]


def test_visualization_extra_and_viewer_assets_are_packaged(built_distributions):
    _, wheel, _ = built_distributions
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
        assert "vis" in metadata.get_all("Provides-Extra", [])
        optional_dependencies = metadata.get_all("Requires-Dist", [])
        server_dependencies = [
            requirement
            for requirement in optional_dependencies
            if requirement.startswith("bottle")
        ]
        assert len(server_dependencies) == 1
        assert 'extra == "vis"' in server_dependencies[0].replace("'", '"')
        assert ">=0.13" in server_dependencies[0]
        assert "<0.14" in server_dependencies[0]
        for source in sorted((PROJECT_ROOT / "src/cryoflake/visualization").rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            package_path = str(source.relative_to(PROJECT_ROOT / "src"))
            assert archive.read(package_path) == source.read_bytes()


def test_source_distribution_keeps_documentation_tests_and_public_guides(
    built_distributions,
):
    _, _, source = built_distributions
    with tarfile.open(source) as archive:
        names = archive.getnames()
    assert not any("/internal/" in name or name.endswith("/.env") for name in names)
    for path in [
        PROJECT_ROOT / ".github/workflows/ci.yml",
        PROJECT_ROOT / "AGENTS.md",
        *sorted((PROJECT_ROOT / "docs").rglob("*.md")),
        *sorted((PROJECT_ROOT / "tests/documentation").glob("*.py")),
    ]:
        assert any(
            name.endswith("/" + str(path.relative_to(PROJECT_ROOT))) for name in names
        ), path
