"""Exercise visualization commands directly from the public guide offline."""

from __future__ import annotations

import json

import mrcfile
import numpy as np
import pytest
from biotite.structure.io import pdbx

from cryoflake import visualization
from cryoflake.storage import transports
from cryoflake.storage.store import (
    load_preprocessed_artifact,
    write_preprocessed_artifact,
)

from .examples import get_example

pytestmark = pytest.mark.documentation


def _assert_reconstruction(bundle, artifact_mapping, *, mode="backbone"):
    assert bundle.structure_path.name == "structure.cif"
    assert bundle.density_path.name == "density.mrc"
    assert bundle.sequence_path.name == "sequence.fasta"
    assert bundle.summary_path.name == "summary.json"
    structure = pdbx.get_structure(pdbx.CIFFile.read(bundle.structure_path), model=1)
    assert len(structure) == 7
    assert set(structure.atom_name) == {"N", "CA", "C", "O", "P", "C1", "O1"}
    assert set(structure.res_name) == {"ALA", "DA", "LIG"}
    np.testing.assert_array_equal(
        structure.coord[structure.atom_name == "CA"],
        artifact_mapping["backbone_coordinates"]["CA"],
    )
    np.testing.assert_array_equal(
        structure.coord[structure.atom_name == "P"],
        artifact_mapping["backbone_coordinates"]["P"],
    )
    with mrcfile.open(bundle.density_path) as density:
        np.testing.assert_array_equal(density.data, artifact_mapping["density_grid"])
        assert float(density.voxel_size.x) == artifact_mapping["voxel_size"]
        np.testing.assert_array_equal(
            [float(density.header.origin[name]) for name in "xyz"],
            artifact_mapping["global_origin"],
        )
    summary = json.loads(bundle.summary_path.read_text())
    assert summary["schema_version"] == 1
    assert summary["coordinate_units"] == "angstrom"
    assert summary["reconstruction_mode"] == mode
    assert summary["atom_count"] == 7
    assert summary["component_count"] == 1
    assert summary["sequence_count"] == 2
    assert {chain["chain_id"] for chain in summary["chains"]} == {
        "POLYMER_0",
        "POLYMER_1",
    }
    assert {chain["sequence"] for chain in summary["chains"]} == {"A"}
    assert {chain["polymer_kind"] for chain in summary["chains"]} == {
        "protein",
        "nucleic_acid",
    }
    fasta = bundle.sequence_path.read_text()
    assert len([line for line in fasta.splitlines() if line.startswith(">")]) == 2
    assert [line for line in fasta.splitlines() if not line.startswith(">")] == [
        "A",
        "A",
    ]
    return structure, summary


@pytest.mark.parametrize(
    "example_id", ["visualization-install", "visualization-checkout"]
)
def test_visualization_installation_examples_expose_the_cli(doc_workspace, example_id):
    (command,) = get_example(example_id).commands()
    result = doc_workspace.invoke(command)
    assert "artifact_reference" in result.output
    assert "molstar" in result.output
    assert "chimerax" in result.output


@pytest.mark.parametrize(
    "example_id, port, open_browser, remote",
    [
        ("visualization-molstar", 0, True, False),
        ("visualization-manual-browser", 8765, False, False),
        ("visualization-s3", 0, False, True),
    ],
)
def test_molstar_examples_reconstruct_aligned_data_before_serving(
    doc_workspace, monkeypatch, example_id, port, open_browser, remote
):
    artifact_reference = (
        "s3://my-bucket/artifacts/example" if remote else "data/artifacts/example"
    )
    artifact_mapping = load_preprocessed_artifact(artifact_reference)
    metadata_path = doc_workspace.root / "data/artifacts/example/metadata.json"
    initial_metadata = metadata_path.read_bytes()
    served = []
    transport_endpoints = []
    get_s3_client = transports._get_s3_client

    def s3_client(endpoint=None):
        transport_endpoints.append(endpoint)
        return get_s3_client(endpoint)

    def serve(bundle, **options):
        _assert_reconstruction(bundle, artifact_mapping)
        served.append((bundle, options))

    monkeypatch.setattr(visualization, "_serve_molstar", serve)
    monkeypatch.setattr(transports, "_get_s3_client", s3_client)
    (command,) = get_example(example_id).commands()
    doc_workspace.invoke(command)

    assert len(served) == 1
    bundle, options = served[0]
    assert options == {"port": port, "open_browser": open_browser}
    assert bundle.structure_path.exists()
    assert metadata_path.read_bytes() == initial_metadata
    initial_exports = {
        path: path.read_bytes()
        for path in (
            bundle.structure_path,
            bundle.density_path,
            bundle.sequence_path,
            bundle.summary_path,
        )
    }
    repeated = doc_workspace.invoke(command, check=False)
    assert repeated.exit_code == 1
    assert "already exist" in repeated.output
    assert len(served) == 1
    assert all(
        path.read_bytes() == contents for path, contents in initial_exports.items()
    )
    if remote:
        assert doc_workspace.s3.calls
        assert all(bucket == "my-bucket" for _, bucket, _ in doc_workspace.s3.calls)
        assert transport_endpoints
        assert set(transport_endpoints) == {"https://s3.example.test"}
    else:
        assert not transport_endpoints


def test_chimerax_example_selects_executable_and_exports_the_same_data(
    doc_workspace, monkeypatch
):
    executable = doc_workspace.root / "ChimeraX"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    artifact_mapping = load_preprocessed_artifact("data/artifacts/example")
    launched = []

    def launch(bundle, *, chimerax_path):
        _assert_reconstruction(bundle, artifact_mapping)
        launched.append(chimerax_path)

    monkeypatch.setattr(visualization, "_launch_chimerax", launch)
    (command,) = get_example("visualization-chimerax").commands(
        {"/opt/ChimeraX/bin/ChimeraX": str(executable)}
    )
    doc_workspace.invoke(command)

    assert launched == [str(executable)]


def test_atom_name_example_restores_masked_polymer_atoms(doc_workspace, monkeypatch):
    artifact_reference = "data/artifacts/example"
    artifact_mapping = load_preprocessed_artifact(artifact_reference)
    backbone = artifact_mapping["backbone_coordinates"]
    full_atom = artifact_mapping["full_atom_targets"]
    full_atom["atom_positions"][0, :4] = np.asarray(
        [backbone[name][0] for name in ("N", "CA", "C", "O")]
    )
    full_atom["atom_mask"][0, :4] = True
    full_atom["atom_positions"][1, 0] = backbone["P"][0]
    full_atom["atom_mask"][1, 0] = True
    write_preprocessed_artifact(artifact_reference, artifact_mapping)
    namespace = get_example("visualization-atom-names").execute()
    atom_names = json.loads((doc_workspace.root / "data/atom_names.json").read_text())
    assert atom_names == namespace["atom_names"]
    assert len(atom_names) == 2
    assert all(len(names) == 65 for names in atom_names)
    served = []

    def serve(bundle, **options):
        structure, summary = _assert_reconstruction(
            bundle, artifact_mapping, mode="full_atom"
        )
        assert set(structure.res_id) == {10, 20, 30}
        assert all("embedding" not in chain for chain in summary["chains"])
        served.append(options)

    monkeypatch.setattr(visualization, "_serve_molstar", serve)
    (command,) = get_example("visualization-full-atom").commands()
    doc_workspace.invoke(command)
    assert served == [{"port": 0, "open_browser": False}]


def test_omitting_export_directory_keeps_data_only_while_viewer_runs(
    doc_workspace, monkeypatch
):
    served = []

    def serve(bundle, **options):
        assert all(
            path.is_file()
            for path in (
                bundle.structure_path,
                bundle.density_path,
                bundle.sequence_path,
                bundle.summary_path,
            )
        )
        served.append(bundle.structure_path.parent)

    monkeypatch.setattr(visualization, "_serve_molstar", serve)
    (command,) = get_example("visualization-molstar").commands()
    output_option = command.index("--visualization-output-dir")
    del command[output_option : output_option + 2]
    doc_workspace.invoke(command)
    assert len(served) == 1
    assert not served[0].exists()
