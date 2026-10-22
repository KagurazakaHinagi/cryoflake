"""Exercise the wheel installation example in a separate environment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
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


def test_documented_wheel_installation_works_outside_checkout(
    tmp_path, built_distributions
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
    install, help_command = get_example("guide-wheel-install").shell_commands()
    _run(
        _wheel_install_arguments(uv, install, wheel),
        cwd=working,
        env=child_env,
    )
    help_result = _run(help_command, cwd=working, env=child_env)
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
        *(
            PROJECT_ROOT / relative
            for relative in (
                "tests/conftest.py",
                "tests/standard_target.py",
                "tests/test_standard_target.py",
                "tests/fixtures/emd_11736_7ae1/7ae1.cif.gz",
                "tests/fixtures/emd_11736_7ae1/subset.cif.gz",
                "tests/fixtures/emd_11736_7ae1/standard_arrays.npz",
                "tests/fixtures/emd_11736_7ae1/full_entry_targets.npz",
                "tests/fixtures/emd_11736_7ae1/provenance.json",
                "tests/fixtures/emd_11736_7ae1/README.md",
                "tests/fixtures/emd_11736_7ae1/regenerate.py",
            )
        ),
        *sorted((PROJECT_ROOT / "docs").rglob("*.md")),
        *sorted((PROJECT_ROOT / "tests/documentation").glob("*.py")),
    ]:
        assert any(
            name.endswith("/" + str(path.relative_to(PROJECT_ROOT))) for name in names
        ), path
