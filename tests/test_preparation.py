from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cryoflake.io.mrc import (
    make_cubic,
    prepare_grid,
)
from cryoflake.io.structure import load_full_backbone_from_structure
from cryoflake.manifests.manifest import (
    read_artifact_list,
    write_artifact_list,
)
from cryoflake.preprocess.manifest import (
    preprocess_manifest_csv,
)


def test_dataset_list_round_trip_resolves_relative_paths(tmp_path):
    artifact = tmp_path / "artifacts" / "example"
    dataset_list = tmp_path / "lists" / "train.txt"

    write_artifact_list([artifact], dataset_list)

    assert read_artifact_list(dataset_list) == [artifact]


def test_dataset_list_reader_preserves_remote_roots(tmp_path):
    dataset_list = tmp_path / "train.txt"
    dataset_list.write_text("s3://bucket/examples/one\n")

    assert read_artifact_list(dataset_list) == ["s3://bucket/examples/one"]


def test_manifest_preprocessing_requires_explicit_adapter(tmp_path):
    with pytest.raises(ValueError, match="preprocess_adapter is required"):
        preprocess_manifest_csv(
            tmp_path / "manifest.csv",
            artifact_output_dir=tmp_path / "output",
        )


def test_cpu_grid_preparation_is_float32_and_adjusts_origin():
    grid = np.ones((4, 6, 8), dtype=np.float32)
    cubic, shift, original_end = make_cubic(grid)

    assert cubic.shape == (128, 128, 128)
    np.testing.assert_array_equal(original_end - shift, grid.shape)

    prepared = prepare_grid(
        grid,
        voxel_size=2.0,
        global_origin=np.asarray([10.0, 20.0, 30.0]),
        target_voxel_size=2.0,
    )
    assert prepared.grid.dtype == np.float32
    np.testing.assert_allclose(
        prepared.global_origin,
        np.asarray([10.0, 20.0, 30.0]) - shift * 2.0,
    )


def test_structure_reader_classifies_nodes_and_marks_missing_atoms(tmp_path):
    structure_path = tmp_path / "example.pdb"
    structure_path.write_text(
        "ATOM      1  N   ALA A   1      10.000  11.000  12.000"
        "  1.00 20.00           N  \n"
        "ATOM      2  CA  ALA A   1      11.000  12.000  13.000"
        "  1.00 20.00           C  \n"
        "ATOM      3  C   ALA A   1      12.000  13.000  14.000"
        "  1.00 20.00           C  \n"
        "HETATM    4  P    DA B   2      20.000  21.000  22.000"
        "  1.00 20.00           P  \n"
        "END\n"
    )

    coordinates, residue_info = load_full_backbone_from_structure(structure_path)

    assert residue_info == {
        "aa_type": ["ALA"],
        "secondary_structure": ["NULL", "NULL"],
        "chain": ["A", "B"],
        "nuc_type": ["DA"],
    }
    assert coordinates["CA"].shape == (1, 3)
    assert coordinates["P"].shape == (1, 3)
    assert coordinates["CA"].dtype == np.float32
    np.testing.assert_allclose(coordinates["P"][0], [20.0, 21.0, 22.0])
    assert np.isnan(coordinates["O"][0]).all()


def test_runtime_package_does_not_import_model_stack():
    package_root = Path(__file__).parents[1] / "src" / "cryoflake"
    offending = []
    for source_path in package_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offending.extend(
                (source_path.name, name)
                for name in names
                if name.split(".", 1)[0] in {"torch", "Bio"}
            )
    assert offending == []


def test_base_import_does_not_load_model_stack():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, cryoflake; "
                "assert 'torch' not in sys.modules; "
                "assert 'Bio' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
