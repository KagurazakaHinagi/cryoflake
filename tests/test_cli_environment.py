"""CLI assertions remain meaningful under CI's import-time terminal settings."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cli_help_and_validation_under_ci_terminal_settings(tmp_path):
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("AWS_", "CRYOFLAKE_"))
        and name != "_TYPER_FORCE_DISABLE_TERMINAL"
    }
    env_file = tmp_path / "credentials.env"
    env_file.write_text("")
    environment.update(
        CRYOFLAKE_ENV_FILE=str(env_file),
        AWS_EC2_METADATA_DISABLED="true",
        GITHUB_ACTIONS="true",
        CI="true",
        FORCE_COLOR="1",
        PY_COLORS="1",
        NO_COLOR="1",
        TERM="dumb",
        COLUMNS="80",
        TERMINAL_WIDTH="80",
    )
    # Typer reads CI flags at import time. A fresh process is required to
    # reproduce the runner; changing os.environ after importing it is too late.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_cli.py::test_help_lists_typer_commands_and_configuration_options",
            "tests/test_cli.py::test_preprocess_rejects_existing_file_output_before_adapter",
            "tests/test_cli.py::test_invalid_options_fail_before_manifest_side_effects",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
