"""Optional artifact reconstruction and interactive molecular viewers."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import webbrowser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryoflake.visualization.reconstruction import VisualizationBundle


def _require_visualization_dependencies() -> None:
    try:
        importlib.import_module("bottle")
    except ModuleNotFoundError as exc:
        if exc.name != "bottle":
            raise
        raise RuntimeError(
            "Visualization requires the vis extra. "
            "Install it with: uv pip install 'cryoflake[vis]'"
        ) from exc


def _find_chimerax(chimerax_path: str | Path | None) -> str:
    if chimerax_path is not None:
        executable = Path(chimerax_path).expanduser().resolve()
        if executable.is_file() and os.access(executable, os.X_OK):
            return str(executable)
        raise ValueError(f"ChimeraX executable is not executable: {executable}")
    for name in ("chimerax", "ChimeraX"):
        if executable := shutil.which(name):
            return executable
    for application_dir in (Path("/Applications"), Path.home() / "Applications"):
        executable = application_dir / "ChimeraX.app/Contents/MacOS/ChimeraX"
        if executable.is_file() and os.access(executable, os.X_OK):
            return str(executable)
    raise ValueError(
        "ChimeraX GUI was not found. Install the standalone UCSF ChimeraX "
        "application and pass its executable with --chimerax-path."
    )


def _launch_chimerax(
    bundle: VisualizationBundle, *, chimerax_path: str | Path | None
) -> None:
    executable = _find_chimerax(chimerax_path)
    summary = json.loads(bundle.summary_path.read_text(encoding="utf-8"))
    sequence_commands = "".join(
        f"run(session, {('sequence chain #1/' + chain['chain_id'])!r})\n"
        for chain in summary["chains"]
    )
    # Let ChimeraX quote its own command arguments. Python repr safely embeds
    # paths in the startup script, including spaces, quotes, and newlines.
    script_path = bundle.structure_path.parent / "chimerax.py"
    script_path.write_text(
        "from chimerax.core.commands import run, StringArg\n"
        f"run(session, 'open ' + StringArg.unparse({str(bundle.structure_path)!r}))\n"
        f"run(session, 'open ' + StringArg.unparse({str(bundle.density_path)!r}))\n"
        "run(session, 'show #1 atoms')\n"
        "run(session, 'style #1 stick')\n"
        "run(session, 'color #1 bychain')\n"
        "run(session, 'volume #2 style surface transparency 0.45 show')\n"
        # Open individual sequences: a multi-FASTA can mix polymer types and
        # chain lengths, so treating it as one alignment would be misleading.
        + sequence_commands
        + "run(session, 'view')\n",
        encoding="utf-8",
    )
    # Waiting keeps temporary reconstructed data alive until the GUI exits.
    subprocess.run([executable, str(script_path)], check=True)


def _serve_molstar(
    bundle: VisualizationBundle, *, port: int, open_browser: bool
) -> None:
    from cryoflake.visualization.server import create_visualization_server

    with create_visualization_server(bundle, port=port) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"MolStar visualization: {url}", flush=True)
        print("Press Ctrl-C to stop the visualization server.", flush=True)
        if open_browser:
            try:
                opened = webbrowser.open(url)
            except (webbrowser.Error, OSError):
                opened = False
            if not opened:
                print("Open the URL above in your browser.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def visualize_artifact(
    artifact_reference: str | Path,
    *,
    viewer: str = "molstar",
    visualization_output_dir: str | Path | None = None,
    atom_names_path: str | Path | None = None,
    port: int = 0,
    open_browser: bool = True,
    chimerax_path: str | Path | None = None,
    s3_endpoint_url: str | None = None,
) -> None:
    """Reconstruct one artifact and run a MolStar server or standalone ChimeraX."""
    if viewer not in {"molstar", "chimerax"}:
        raise ValueError("viewer must be molstar or chimerax")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    if viewer != "chimerax" and chimerax_path is not None:
        raise ValueError("chimerax_path requires viewer=chimerax")
    _require_visualization_dependencies()
    if viewer == "chimerax":
        # Fail before downloading or exporting an artifact if the GUI is absent.
        chimerax_path = _find_chimerax(chimerax_path)

    from cryoflake.visualization.reconstruction import reconstruct_artifact

    def launch(output_dir: Path) -> None:
        bundle = reconstruct_artifact(
            artifact_reference,
            output_dir,
            atom_names_path=atom_names_path,
            s3_endpoint_url=s3_endpoint_url,
        )
        if viewer == "molstar":
            _serve_molstar(bundle, port=port, open_browser=open_browser)
        else:
            _launch_chimerax(bundle, chimerax_path=chimerax_path)

    if visualization_output_dir is None:
        with TemporaryDirectory(prefix="cryoflake-visualization-") as temporary_dir:
            launch(Path(temporary_dir))
    else:
        launch(Path(visualization_output_dir).expanduser().resolve())


__all__ = ["visualize_artifact"]
