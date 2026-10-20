"""Exercise compressed raw inputs through real manifest workers and storage."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import weakref
from pathlib import Path

import numpy as np
import pytest

from cryoflake.constants import FULL_ATOM_COUNT, RIGID_GROUP_COUNT, TORSION_COUNT
from cryoflake.io.mrc import load_mrc, save_mrc
from cryoflake.io.structure import load_full_backbone_from_structure
from cryoflake.manifests.manifest import read_artifact_list
from cryoflake.preprocess import reuse, worker
from cryoflake.preprocess.manifest import preprocess_manifest_csv
from cryoflake.preprocess.reuse import _preprocessed_artifact_is_reusable
from cryoflake.storage.artifacts import (
    validate_artifact_list,
    validate_preprocessed_artifact,
)
from cryoflake.storage.store import (
    load_preprocessed_artifact,
    write_preprocessed_artifact,
)
from cryoflake.targets.components import build_component_extension_targets
from cryoflake.targets.full_atom import validate_full_atom_targets

_MIXED_CIF = """data_pipeline
loop_
_entity.id
_entity.type
1 polymer
2 polymer
3 non-polymer
loop_
_entity_poly.entity_id
_entity_poly.type
1 polypeptide(L)
2 polydeoxyribonucleotide
loop_
_chem_comp.id
_chem_comp.type
ALA 'L-PEPTIDE LINKING'
DA 'DNA LINKING'
LIG 'NON-POLYMER'
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_PDB_model_num
_atom_site.type_symbol
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM N . ALA A 1 1 A 10 ? 1 N 1.0 20.0 1.0 0.0 1.0
ATOM CA . ALA A 1 1 A 10 ? 1 C 1.0 20.0 1.0 1.0 1.0
ATOM C . ALA A 1 1 A 10 ? 1 C 1.0 20.0 2.0 1.0 1.0
ATOM O . ALA A 1 1 A 10 ? 1 O 1.0 20.0 2.0 2.0 1.0
ATOM P . DA B 2 1 B 20 ? 1 P 1.0 20.0 2.0 2.0 2.0
HETATM C1 . LIG C 3 . C 30 ? 1 C 1.0 20.0 1.5 1.0 1.0
HETATM O1 . LIG C 3 . C 30 ? 1 O 1.0 20.0 1.5 2.0 1.0
"""


def _full_atom_fixture():
    """Two canonical nodes with unavailable geometry explicitly masked artifact_reference."""
    targets = {
        "aatype": np.asarray([0, 24], dtype=np.int64),
        "prot_mask": np.asarray([True, False]),
        "chain_index": np.asarray([0, 1], dtype=np.int64),
        "residue_index": np.asarray([10, 20], dtype=np.int64),
    }
    shapes = {
        "atom_positions": (FULL_ATOM_COUNT, 3),
        "atom_mask": (FULL_ATOM_COUNT,),
        "b_factors": (FULL_ATOM_COUNT,),
        "rigidgroups_gt_frames": (RIGID_GROUP_COUNT, 3, 4),
        "rigidgroups_gt_exists": (RIGID_GROUP_COUNT,),
        "rigidgroups_group_exists": (RIGID_GROUP_COUNT,),
        "rigidgroups_group_is_ambiguous": (RIGID_GROUP_COUNT,),
        "rigidgroups_alt_gt_frames": (RIGID_GROUP_COUNT, 3, 4),
        "torsion_angles_sin_cos": (TORSION_COUNT, 2),
        "alt_torsion_angles_sin_cos": (TORSION_COUNT, 2),
        "torsion_angles_mask": (TORSION_COUNT,),
    }
    targets.update(
        {
            name: np.zeros((2, *shape), dtype=np.float32)
            for name, shape in shapes.items()
        }
    )
    return validate_full_atom_targets(targets, num_protein=1, num_nucleotide=1)


@pytest.mark.parametrize("vector_fields", [("CA",), ("P",), ("CA", "P")])
def test_artifact_validation_counts_normalized_single_coordinates(
    tmp_path, vector_fields
):
    structure = tmp_path / "structure.cif"
    structure.write_text(_MIXED_CIF)
    grid = np.ones((4, 4, 4), dtype=np.float32)
    origin = np.zeros(3, dtype=np.float32)
    coordinates = {
        name: np.ones((1, 3), dtype=np.float32) for name in ("CA", "C", "N", "O", "P")
    }
    for name in vector_fields:
        coordinates[name] = coordinates[name].reshape(3)
    root = tmp_path / "artifact"
    write_preprocessed_artifact(
        root,
        {
            "density_grid": grid,
            "global_origin": origin,
            "voxel_size": 1.0,
            "backbone_coordinates": coordinates,
            "residue_info": {"aa_type": ["ALA"], "nuc_type": ["DA"]},
            "full_atom_targets": _full_atom_fixture(),
            "extension_targets": build_component_extension_targets(
                structure, global_origin=origin, voxel_size=1.0, grid_shape=grid.shape
            ),
        },
    )

    result = validate_preprocessed_artifact(root)

    assert result.protein_nodes == result.nucleotide_nodes == 1


class _CPUAdapter:
    """A real NumPy/Biotite adapter with one intentional transient interruption."""

    def __init__(self):
        self.calls = []
        self.interrupted = False

    def __call__(
        self,
        *,
        map_path,
        structure_path,
        artifact_reference,
        target_voxel_size,
        interface_contact_cutoff,
        original_structure_path,
        fasta_path,
        fix_shift,
    ):
        self.calls.append(Path(structure_path).name)
        assert Path(map_path).suffix == ".mrc"
        assert Path(structure_path).suffix in {".cif", ".mmcif"}
        assert original_structure_path is None and fasta_path is None and fix_shift
        if Path(structure_path).name == "2abc.mmcif" and not self.interrupted:
            self.interrupted = True
            raise RuntimeError("intentional adapter interruption")

        volume = load_mrc(map_path)
        assert volume.voxel_size == target_voxel_size == 1
        coordinates, residue_info = load_full_backbone_from_structure(structure_path)
        full_atom_targets = _full_atom_fixture()
        extension_targets = build_component_extension_targets(
            structure_path,
            global_origin=volume.global_origin,
            voxel_size=volume.voxel_size,
            grid_shape=volume.grid.shape,
            polymer_atom_positions=np.concatenate(
                (coordinates["CA"], coordinates["P"])
            ),
            polymer_atom_node_indices=np.arange(2, dtype=np.int64),
            polymer_chain_indices=full_atom_targets["chain_index"],
            polymer_residue_indices=full_atom_targets["residue_index"],
            polymer_aatype=full_atom_targets["aatype"],
            interface_contact_cutoff=interface_contact_cutoff,
        )
        write_preprocessed_artifact(
            artifact_reference,
            {
                "density_grid": volume.grid,
                "voxel_size": volume.voxel_size,
                "global_origin": volume.global_origin,
                "backbone_coordinates": coordinates,
                "residue_info": residue_info,
                "full_atom_targets": full_atom_targets,
                "extension_targets": extension_targets,
            },
        )


def _compressed_inputs(tmp_path):
    grid = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    raw_map = tmp_path / "density.mrc"
    save_mrc(grid, 1, np.zeros(3), raw_map)
    compressed_map = tmp_path / "density.mrc.gz"
    compressed_map.write_bytes(gzip.compress(raw_map.read_bytes()))
    raw_map.unlink()
    for name in ("1abc.cif.gz", "2abc.mmcif.gz"):
        (tmp_path / name).write_bytes(gzip.compress(_MIXED_CIF.encode()))
    sequence = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.savez_compressed(
        tmp_path / "sequence.npz",
        sequence=sequence,
        sequence_mask=np.asarray([True, False]),
        unified_seq=np.asarray("A|||A"),
        unified_seq_len=np.asarray(2),
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path,sequence_embedding_path\n"
        "1001,1abc,density.mrc.gz,1abc.cif.gz,sequence.npz\n"
        "1002,2abc,density.mrc.gz,2abc.mmcif.gz,sequence.npz\n"
    )
    return manifest, grid, sequence


def test_reused_artifact_is_loaded_once_per_manifest_row(monkeypatch, tmp_path):
    manifest, _, _ = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path\n"
        "1001,1abc,density.mrc.gz,1abc.cif.gz\n"
    )
    adapter = _CPUAdapter()
    arguments = dict(
        manifest_csv=manifest,
        artifact_output_dir=tmp_path / "artifacts",
        preprocess_adapter=adapter,
        continue_on_preprocess_error=False,
    )
    artifact = preprocess_manifest_csv(**arguments)[0]
    loaded_paths = []

    def tracked_load(path, **kwargs):
        loaded_paths.append(path)
        return load_preprocessed_artifact(path, **kwargs)

    monkeypatch.setattr(reuse, "load_preprocessed_artifact", tracked_load)
    monkeypatch.setattr(worker, "load_preprocessed_artifact", tracked_load)

    assert preprocess_manifest_csv(**arguments) == [artifact]
    assert loaded_paths == [artifact]
    assert adapter.calls == ["1abc.cif"]


def test_reuse_releases_rejected_payload_before_loading_next_candidate(
    monkeypatch, tmp_path
):
    target = tmp_path / "target"
    previous_grid = None
    loads = 0

    def candidate_load(path, **kwargs):
        nonlocal previous_grid, loads
        assert previous_grid is None or previous_grid() is None
        loads += 1
        grid = np.zeros((4, 4, 4), dtype=np.float32)
        previous_grid = weakref.ref(grid)
        return {"density_grid": grid, "valid": path != target}

    monkeypatch.setattr(reuse, "load_preprocessed_artifact", candidate_load)
    monkeypatch.setattr(
        reuse,
        "_preprocessed_artifact_is_reusable",
        lambda payload, **kwargs: payload["valid"],
    )
    matched = reuse._find_reusable_preprocessed_artifact(
        {"emdb_id": "1001", "wwpdb_id": "1abc"},
        target,
        [tmp_path / "reusable"],
    )
    assert loads == 2
    assert matched[0] == tmp_path / "reusable" / "emd_1001_1abc"
    assert matched[1]["density_grid"] is previous_grid()


def test_compressed_manifest_pipeline_validates_resumes_and_reuses(tmp_path):
    manifest, grid, sequence = _compressed_inputs(tmp_path)
    output_dir = tmp_path / "artifacts"
    dataset_list = tmp_path / "dataset.txt"
    summary = tmp_path / "summary.csv"
    journal = Path(f"{summary}.checkpoint.jsonl")
    adapter = _CPUAdapter()
    progress = []
    arguments = dict(
        manifest_csv=manifest,
        artifact_output_dir=output_dir,
        artifact_list_path=dataset_list,
        summary_csv=summary,
        preprocess_adapter=adapter,
        # The callable tracks execution counts, rather than configuration state.
        preprocess_checkpoint_token="cpu-integration-v1",
        require_sequence_embeddings=True,
        continue_on_preprocess_error=False,
        progress_callback=progress.append,
    )
    first_artifact = output_dir / "emd_1001_1abc"
    with pytest.raises(RuntimeError, match="intentional adapter interruption"):
        preprocess_manifest_csv(**arguments)
    assert adapter.calls == ["1abc.cif", "2abc.mmcif"]
    assert journal.is_file()
    assert (
        validate_preprocessed_artifact(
            first_artifact, require_sequence_embeddings=True
        ).sequence_length
        == 2
    )
    first_core_hash = hashlib.sha256(
        (first_artifact / "core.npz").read_bytes()
    ).digest()

    paths = preprocess_manifest_csv(**arguments)
    assert adapter.calls == ["1abc.cif", "2abc.mmcif", "2abc.mmcif"]
    assert any(message.startswith("Resuming 1/2 rows") for message in progress)
    assert paths == [first_artifact, output_dir / "emd_1002_2abc"]
    assert not journal.exists()
    results = validate_artifact_list(dataset_list, require_sequence_embeddings=True)
    assert len(results) == 2
    for artifact, result in zip(paths, results):
        assert result.grid_shape == (4, 4, 4)
        assert (
            result.protein_nodes
            == result.nucleotide_nodes
            == result.component_nodes
            == 1
        )
        assert result.sequence_length == 2 and result.sequence_feature_dim == 4
        loaded = load_preprocessed_artifact(artifact)
        np.testing.assert_array_equal(loaded["sequence"], sequence)
        np.testing.assert_array_equal(loaded["sequence_mask"], [True, False])
        np.testing.assert_array_equal(loaded["backbone_coordinates"]["CA"], [[1, 1, 1]])
        assert loaded["extension_targets"]["component"]["component_id"].tolist() == [
            "LIG"
        ]
        assert loaded["extension_targets"]["component"]["interface_mask"].tolist() == [
            True
        ]
        with np.load(artifact / "core.npz", allow_pickle=False) as archive:
            assert archive.files
        assert validate_preprocessed_artifact(artifact).grid_shape == grid.shape

    # Completion clears the checkpoint, so another run exercises actual reuse.
    assert preprocess_manifest_csv(**arguments) == paths
    assert len(adapter.calls) == 3
    assert (
        hashlib.sha256((first_artifact / "core.npz").read_bytes()).digest()
        == first_core_hash
    )
    with summary.open(newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    assert [row["status"] for row in summary_rows] == ["reused", "reused"]
    assert [row["success"] for row in summary_rows] == ["true", "true"]
    assert [row["split"] for row in summary_rows] == ["train", "train"]
    assert (
        len(validate_artifact_list(dataset_list, require_sequence_embeddings=True)) == 2
    )


@pytest.mark.parametrize("split_strategy", ["component", "interface"])
def test_recovered_grouped_split_report_replays_with_canonical_references(
    tmp_path, split_strategy
):
    manifest, _, _ = _compressed_inputs(tmp_path)
    # Independent ligand/interface groups permit a nonempty grouped split.
    (tmp_path / "2abc.mmcif.gz").write_bytes(
        gzip.compress(_MIXED_CIF.replace("LIG", "ATP").encode())
    )
    adapter = _CPUAdapter()
    summary = tmp_path / "summary.csv"
    train_list = tmp_path / "train.txt"
    validation_list = tmp_path / "valid.txt"
    report = tmp_path / "split.json"
    replayed_report = tmp_path / "replayed.json"
    arguments = dict(
        manifest_csv=manifest,
        artifact_output_dir=tmp_path / "artifacts",
        artifact_list_path=train_list,
        validation_artifact_list_path=validation_list,
        summary_csv=summary,
        split_strategy=split_strategy,
        split_report_path=report,
        preprocess_adapter=adapter,
        preprocess_checkpoint_token="grouped-naming-integration-v1",
        require_sequence_embeddings=True,
        continue_on_preprocess_error=False,
    )
    with pytest.raises(RuntimeError, match="intentional adapter interruption"):
        preprocess_manifest_csv(**arguments)

    journal = Path(f"{summary}.checkpoint.jsonl")
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    durable = next(record for record in records if record["state"] == "durable_success")
    assert (
        durable["artifact_reference"]
        == (durable["grouped_split_record"]["artifact_reference"])
    )
    assert durable["summary_row"]["artifact_reference"] == durable["artifact_reference"]
    assert "component_split_record" not in durable
    first_core = Path(durable["artifact_reference"]) / "core.npz"
    first_core_bytes = first_core.read_bytes()

    references = preprocess_manifest_csv(**arguments)

    assert adapter.calls == ["1abc.cif", "2abc.mmcif", "2abc.mmcif"]
    assert not journal.exists()
    assert first_core.read_bytes() == first_core_bytes
    original = json.loads(report.read_text())
    assert original["schema_version"] == 3
    assert original["split_strategy"] == split_strategy
    assert {row["artifact_reference"] for row in original["assignments"]} == {
        str(reference) for reference in references
    }
    original_train = read_artifact_list(train_list)
    original_validation = read_artifact_list(validation_list)
    assert len(original_train) == len(original_validation) == 1
    assert set(original_train).isdisjoint(original_validation)

    replay_arguments = dict(
        arguments, holdout_group_ids_path=report, split_report_path=replayed_report
    )
    assert preprocess_manifest_csv(**replay_arguments) == references
    replayed = json.loads(replayed_report.read_text())
    assert replayed["assignments"] == original["assignments"]
    assert (
        replayed["selected_holdout_group_ids"] == original["selected_holdout_group_ids"]
    )
    assert read_artifact_list(train_list) == original_train
    assert read_artifact_list(validation_list) == original_validation
    assert len(adapter.calls) == 3
    assert first_core.read_bytes() == first_core_bytes
    with summary.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assignments = {
        row["artifact_reference"]: row["split"] for row in replayed["assignments"]
    }
    assert {row["artifact_reference"]: row["split"] for row in rows} == assignments


def test_manifest_ignores_pickle_files_and_reuses_npz_json_artifacts(
    tmp_path, monkeypatch
):
    manifest, grid, _ = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path\n"
        "1001,1abc,density.mrc.gz,1abc.cif.gz\n"
    )
    artifact_name = "emd_1001_1abc"
    reuse_dir = tmp_path / "reuse"
    reuse_dir.mkdir()
    for suffix in (".pkl", ".pickle"):
        # A protocol-4 Pickle of None must never reach a deserializer.
        (reuse_dir / f"{artifact_name}{suffix}").write_bytes(b"\x80\x04N.")

    opened_single_files = []
    original_open = Path.open

    def record_open(path, *args, **kwargs):
        if path.parent == reuse_dir and path.suffix in {".pkl", ".pickle"}:
            opened_single_files.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)
    adapter = _CPUAdapter()
    output_dir = tmp_path / "artifacts"
    summary = tmp_path / "summary.csv"
    paths = preprocess_manifest_csv(
        manifest,
        output_dir,
        artifact_reuse_dirs=[reuse_dir],
        preprocess_adapter=adapter,
        summary_csv=summary,
        continue_on_preprocess_error=False,
    )

    artifact = output_dir / artifact_name
    assert paths == [artifact]
    assert adapter.calls == ["1abc.cif"]
    assert opened_single_files == []
    assert {path.name for path in artifact.iterdir()} == {
        "core.npz",
        "metadata.json",
    }
    assert validate_preprocessed_artifact(artifact).grid_shape == grid.shape
    np.testing.assert_array_equal(
        load_preprocessed_artifact(artifact)["density_grid"], grid
    )
    with summary.open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["status"] == "processed"

    # Canonical external reuse still works alongside ignored single-file data.
    assert preprocess_manifest_csv(
        manifest,
        tmp_path / "other_artifacts",
        artifact_reuse_dirs=[reuse_dir, output_dir],
        preprocess_adapter=adapter,
        summary_csv=summary,
        continue_on_preprocess_error=False,
    ) == [artifact]
    assert adapter.calls == ["1abc.cif"]
    assert opened_single_files == []
    with summary.open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["status"] == "reused_external"


def test_required_sidecar_enriches_reused_target_without_preprocessing(tmp_path):
    manifest, _, sequence = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path\n"
        "1001,1abc,density.mrc.gz,1abc.cif.gz\n"
    )
    adapter = _CPUAdapter()
    output_dir = tmp_path / "artifacts"
    summary = tmp_path / "summary.csv"
    arguments = dict(
        manifest_csv=manifest,
        artifact_output_dir=output_dir,
        preprocess_adapter=adapter,
        summary_csv=summary,
        continue_on_preprocess_error=False,
    )
    artifact = preprocess_manifest_csv(**arguments)[0]
    assert "sequence" not in load_preprocessed_artifact(artifact)
    core_before = (artifact / "core.npz").read_bytes()

    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path,sequence_embedding_path\n"
        "1001,1abc,density.mrc.gz,1abc.cif.gz,sequence.npz\n"
    )
    assert preprocess_manifest_csv(**arguments, require_sequence_embeddings=True) == [
        artifact
    ]

    assert adapter.calls == ["1abc.cif"]
    assert (artifact / "core.npz").read_bytes() == core_before
    np.testing.assert_array_equal(
        load_preprocessed_artifact(artifact)["sequence"], sequence
    )
    assert (
        validate_preprocessed_artifact(
            artifact, require_sequence_embeddings=True
        ).sequence_length
        == 2
    )
    with summary.open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["status"] == "reused"


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_reuse_rebuilds_artifact_with_unreadable_optional_sequence(tmp_path, damage):
    manifest, _, _ = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path,sequence_embedding_path\n"
        "1001,1abc,density.mrc.gz,1abc.cif.gz,sequence.npz\n"
    )
    adapter = _CPUAdapter()
    arguments = dict(
        manifest_csv=manifest,
        artifact_output_dir=tmp_path / "artifacts",
        preprocess_adapter=adapter,
        continue_on_preprocess_error=False,
    )
    artifact = preprocess_manifest_csv(**arguments)[0]
    sequence_store = artifact / "sequence.npz"
    if damage == "missing":
        sequence_store.unlink()
    else:
        sequence_store.write_bytes(b"broken archive")
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path\n"
        "1001,1abc,density.mrc.gz,1abc.cif.gz\n"
    )

    assert preprocess_manifest_csv(**arguments) == [artifact]

    assert adapter.calls == ["1abc.cif", "1abc.cif"]
    assert "sequence" not in load_preprocessed_artifact(artifact)
    assert validate_preprocessed_artifact(artifact).sequence_length is None


def test_reuse_repairs_corrupt_sequence_from_replacement_without_reprocessing(
    tmp_path,
):
    manifest, _, sequence = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path,sequence_embedding_path\n"
        "1001,1abc,density.mrc.gz,1abc.cif.gz,sequence.npz\n"
    )
    adapter = _CPUAdapter()
    arguments = dict(
        manifest_csv=manifest,
        artifact_output_dir=tmp_path / "artifacts",
        preprocess_adapter=adapter,
        continue_on_preprocess_error=False,
        require_sequence_embeddings=True,
    )
    artifact = preprocess_manifest_csv(**arguments)[0]
    core_before = (artifact / "core.npz").read_bytes()
    (artifact / "sequence.npz").write_bytes(b"broken archive")

    assert preprocess_manifest_csv(**arguments) == [artifact]

    assert adapter.calls == ["1abc.cif"]
    assert (artifact / "core.npz").read_bytes() == core_before
    np.testing.assert_array_equal(
        load_preprocessed_artifact(artifact)["sequence"], sequence
    )


@pytest.mark.parametrize(
    "sequence_fields",
    [
        {"sequence_mask": np.ones(2, dtype=bool)},
        {"sequence": None},
        {"sequence": np.zeros((0, 4), dtype=np.float32)},
    ],
)
def test_reuse_rejects_present_invalid_optional_sequence(
    tmp_path, sequence_fields, monkeypatch
):
    manifest, _, _ = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path\n"
        "1001,1abc,density.mrc.gz,1abc.cif.gz\n"
    )
    artifact = preprocess_manifest_csv(
        manifest,
        tmp_path / "artifacts",
        preprocess_adapter=_CPUAdapter(),
        continue_on_preprocess_error=False,
    )[0]
    payload = load_preprocessed_artifact(artifact)
    assert _preprocessed_artifact_is_reusable(payload)
    payload.update(sequence_fields)

    assert not _preprocessed_artifact_is_reusable(payload)
    monkeypatch.setattr(
        "cryoflake.storage.artifacts.load_preprocessed_artifact",
        lambda *args, **kwargs: payload,
    )
    with pytest.raises(ValueError, match="canonical preprocessing contract"):
        validate_preprocessed_artifact(artifact)


@pytest.mark.parametrize("damage", ["grid", "full_atom_targets", "contact_cutoff"])
def test_manifest_rejects_adapter_outputs_that_violate_training_contract(
    tmp_path, damage
):
    manifest, _, _ = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path\n"
        "1001,1abc,density.mrc.gz,1abc.cif.gz\n"
    )
    adapter = _CPUAdapter()

    def invalid_adapter(**kwargs):
        adapter(**kwargs)
        payload = load_preprocessed_artifact(kwargs["artifact_reference"])
        if damage == "grid":
            payload["density_grid"] = np.zeros((1, 1, 1), dtype=np.float32)
        elif damage == "full_atom_targets":
            payload.pop("full_atom_targets")
        else:
            payload["extension_targets"]["component"]["interface_contact_cutoff"] = 9.0
        write_preprocessed_artifact(kwargs["artifact_reference"], payload)

    with pytest.raises(ValueError, match="does not satisfy the training contract"):
        preprocess_manifest_csv(
            manifest,
            tmp_path / "artifacts",
            preprocess_adapter=invalid_adapter,
            continue_on_preprocess_error=False,
        )


def test_cli_preprocess_validates_real_adapter_output(monkeypatch, tmp_path):
    import sys
    from types import ModuleType

    from typer.testing import CliRunner

    from cryoflake.cli import app

    _compressed_inputs(tmp_path)
    raw_map = tmp_path / "density.mrc"
    raw_map.write_bytes(gzip.decompress((tmp_path / "density.mrc.gz").read_bytes()))
    raw_structure = tmp_path / "1abc.cif"
    raw_structure.write_text(_MIXED_CIF)
    output = tmp_path / "artifact"
    adapter = _CPUAdapter()
    module = ModuleType("_cryoflake_cli_integration_adapter")

    def preprocess(**kwargs):
        kwargs.pop("s3_endpoint_url")
        adapter(**kwargs)

    module.preprocess = preprocess
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = CliRunner().invoke(
        app,
        [
            "preprocess",
            str(raw_map),
            str(raw_structure),
            str(output),
            "--preprocess-adapter",
            f"{module.__name__}:preprocess",
        ],
    )
    assert result.exit_code == 0, result.output
    assert adapter.calls == ["1abc.cif"]
    validated = validate_preprocessed_artifact(output)
    assert validated.protein_nodes == validated.nucleotide_nodes == 1
