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
from standard_target import (
    load_standard_artifact_mapping,
    load_standard_full_atom_targets,
    standard_expected,
    write_standard_structure,
)

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


@pytest.mark.parametrize("vector_fields", [("CA",), ("P",), ("CA", "P")])
def test_artifact_validation_counts_normalized_single_coordinates(
    tmp_path, vector_fields
):
    payload = load_standard_artifact_mapping()
    targets = payload["full_atom_targets"]
    protein_count = len(payload["backbone_coordinates"]["CA"])
    selected_rows = [0, protein_count]
    payload["full_atom_targets"] = {
        key: value[selected_rows]
        if isinstance(value, np.ndarray) and key != "atom_names"
        else value
        for key, value in targets.items()
    }
    coordinates = {
        name: values[:1].copy()
        for name, values in payload["backbone_coordinates"].items()
    }
    payload["backbone_coordinates"] = coordinates
    payload["residue_info"] = {
        "aa_type": payload["residue_info"]["aa_type"][:1],
        "nuc_type": payload["residue_info"]["nuc_type"][:1],
    }
    for name in vector_fields:
        coordinates[name] = coordinates[name].reshape(3)
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, payload)

    result = validate_preprocessed_artifact(root)

    assert result.protein_nodes == result.nucleotide_nodes == 1
    restored_targets = load_preprocessed_artifact(root)["full_atom_targets"]
    assert restored_targets["atom_names"].shape == (65,)
    assert restored_targets["atom_names"].dtype.kind == "U"
    np.testing.assert_array_equal(restored_targets["atom_names"], targets["atom_names"])
    assert restored_targets["atom_mask"].any()


class _StandardTargetAdapter:
    """Round-trip the canonical real fixture with one controlled interruption.

    Map resampling is tested separately; this injected adapter preserves the
    fixture's actual spacing and only validates the requested spacing value.
    """

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
        if Path(structure_path).name == "9tst.mmcif" and not self.interrupted:
            self.interrupted = True
            raise RuntimeError("intentional adapter interruption")

        volume = load_mrc(map_path)
        assert target_voxel_size > 0
        assert np.isclose(volume.voxel_size, standard_expected()["voxel_size"])
        coordinates, residue_info = load_full_backbone_from_structure(structure_path)
        full_atom_targets = load_standard_full_atom_targets()
        extension_targets = build_component_extension_targets(
            structure_path,
            global_origin=volume.global_origin,
            voxel_size=volume.voxel_size,
            grid_shape=volume.grid.shape,
            polymer_atom_positions=full_atom_targets["atom_positions"][
                full_atom_targets["atom_mask"]
            ],
            polymer_atom_node_indices=np.broadcast_to(
                np.arange(len(full_atom_targets["aatype"]))[:, None],
                full_atom_targets["atom_mask"].shape,
            )[full_atom_targets["atom_mask"]],
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
    payload = load_standard_artifact_mapping()
    grid = payload["density_grid"]
    raw_map = tmp_path / "density.mrc"
    save_mrc(grid, payload["voxel_size"], payload["global_origin"], raw_map)
    compressed_map = tmp_path / "density.mrc.gz"
    compressed_map.write_bytes(gzip.compress(raw_map.read_bytes()))
    raw_map.unlink()
    structure_path = write_standard_structure(tmp_path / "standard.cif")
    # The second routing record is an explicit test duplicate of 7AE1, used to
    # exercise interruption/recovery. 90001/9tst is not another biological entry.
    for name in ("7ae1.cif.gz", "9tst.mmcif.gz"):
        (tmp_path / name).write_bytes(gzip.compress(structure_path.read_bytes()))
    protein_count = len(payload["backbone_coordinates"]["CA"])
    ca_slot = payload["full_atom_targets"]["atom_names"].tolist().index("CA")
    # Geometry-derived numeric test features stand in for a sequence model;
    # no language-model output is claimed by these storage/attachment checks.
    sequence = np.column_stack(
        (
            payload["backbone_coordinates"]["CA"],
            payload["full_atom_targets"]["b_factors"][:protein_count, ca_slot],
        )
    ).astype(np.float32)
    np.savez_compressed(
        tmp_path / "sequence.npz",
        sequence=sequence,
        sequence_mask=np.arange(protein_count) % 2 == 0,
        unified_seq_len=np.asarray(protein_count),
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path,sequence_embedding_path\n"
        "11736,7ae1,density.mrc.gz,7ae1.cif.gz,sequence.npz\n"
        "90001,9tst,density.mrc.gz,9tst.mmcif.gz,sequence.npz\n"
    )
    return manifest, grid, sequence


def test_reused_artifact_is_loaded_once_per_manifest_row(monkeypatch, tmp_path):
    manifest, _, _ = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path\n"
        "11736,7ae1,density.mrc.gz,7ae1.cif.gz\n"
    )
    adapter = _StandardTargetAdapter()
    arguments = dict(
        manifest_csv=manifest,
        artifact_output_dir=tmp_path / "artifacts",
        preprocess_adapter=adapter,
        target_voxel_size=standard_expected()["voxel_size"],
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
    assert adapter.calls == ["7ae1.cif"]


def test_standard_pipeline_fits_compact_fixture_memory_budget(monkeypatch, tmp_path):
    """The real coarse map must not imply an unperformed fine-grid resampling."""
    manifest, _, _ = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path\n"
        "11736,7ae1,density.mrc.gz,7ae1.cif.gz\n"
    )
    monkeypatch.setattr(
        "cryoflake.preprocess.manifest.preprocess_memory_guard_limit_bytes",
        lambda: 512 * 1024**2,
    )
    references = preprocess_manifest_csv(
        manifest,
        tmp_path / "artifacts",
        preprocess_adapter=_StandardTargetAdapter(),
        target_voxel_size=standard_expected()["voxel_size"],
        continue_on_preprocess_error=False,
    )
    assert len(references) == 1
    result = validate_preprocessed_artifact(references[0])
    assert result.grid_shape == standard_expected()["grid_shape"]
    assert result.protein_nodes == standard_expected()["protein_nodes"]


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
        {"emdb_id": "11736", "wwpdb_id": "7ae1"},
        target,
        [tmp_path / "reusable"],
    )
    assert loads == 2
    assert matched[0] == tmp_path / "reusable" / "emd_11736_7ae1"
    assert matched[1]["density_grid"] is previous_grid()


def test_compressed_manifest_pipeline_validates_resumes_and_reuses(tmp_path):
    manifest, grid, sequence = _compressed_inputs(tmp_path)
    output_dir = tmp_path / "artifacts"
    dataset_list = tmp_path / "dataset.txt"
    summary = tmp_path / "summary.csv"
    journal = Path(f"{summary}.checkpoint.jsonl")
    adapter = _StandardTargetAdapter()
    progress = []
    arguments = dict(
        manifest_csv=manifest,
        artifact_output_dir=output_dir,
        artifact_list_path=dataset_list,
        summary_csv=summary,
        preprocess_adapter=adapter,
        target_voxel_size=standard_expected()["voxel_size"],
        # The callable tracks execution counts, rather than configuration state.
        preprocess_checkpoint_token="cpu-integration-v1",
        require_sequence_embeddings=True,
        continue_on_preprocess_error=False,
        progress_callback=progress.append,
    )
    first_artifact = output_dir / "emd_11736_7ae1"
    with pytest.raises(RuntimeError, match="intentional adapter interruption"):
        preprocess_manifest_csv(**arguments)
    assert adapter.calls == ["7ae1.cif", "9tst.mmcif"]
    assert journal.is_file()
    assert validate_preprocessed_artifact(
        first_artifact, require_sequence_embeddings=True
    ).sequence_length == len(sequence)
    first_core_hash = hashlib.sha256(
        (first_artifact / "core.npz").read_bytes()
    ).digest()

    paths = preprocess_manifest_csv(**arguments)
    assert adapter.calls == ["7ae1.cif", "9tst.mmcif", "9tst.mmcif"]
    assert any(message.startswith("Resuming 1/2 rows") for message in progress)
    assert paths == [first_artifact, output_dir / "emd_90001_9tst"]
    assert not journal.exists()
    results = validate_artifact_list(dataset_list, require_sequence_embeddings=True)
    assert len(results) == 2
    for artifact, result in zip(paths, results):
        assert result.grid_shape == tuple(standard_expected()["grid_shape"])
        assert result.protein_nodes == standard_expected()["protein_nodes"]
        assert result.nucleotide_nodes == standard_expected()["nucleotide_nodes"]
        assert result.component_nodes == standard_expected()["component_nodes"]
        assert (
            result.sequence_length == len(sequence) and result.sequence_feature_dim == 4
        )
        loaded = load_preprocessed_artifact(artifact)
        np.testing.assert_array_equal(loaded["sequence"], sequence)
        np.testing.assert_array_equal(
            loaded["sequence_mask"], np.arange(len(sequence)) % 2 == 0
        )
        np.testing.assert_array_equal(
            loaded["backbone_coordinates"]["CA"],
            load_standard_artifact_mapping()["backbone_coordinates"]["CA"],
        )
        assert sorted(
            set(loaded["extension_targets"]["component"]["component_id"])
        ) == ["MG", "SF4", "ZN"]
        component = loaded["extension_targets"]["component"]
        assert (component["interface_group_id"] != "").all()
        assert "prot_mask" not in loaded["full_atom_targets"]
        assert not {"atom_mask", "interface_mask", "split_groups"}.intersection(
            component
        )
        assert "positions" not in component["proposal_targets"]
        with np.load(artifact / "core.npz", allow_pickle=False) as archive:
            assert archive.files
            assert not {
                "full_atom_targets/prot_mask",
                "extension_targets/component/atom_mask",
                "extension_targets/component/interface_mask",
                "extension_targets/component/proposal_targets/positions",
            }.intersection(archive.files)
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
    # Deliberate component subsets of the same deposition provide disjoint
    # groups for this algorithm test; no fictitious ligand chemistry is added.
    for name, component_ids in (("7ae1.cif.gz", ["ZN"]), ("9tst.mmcif.gz", ["SF4"])):
        structure_path = write_standard_structure(
            tmp_path / "selected.cif", component_ids=component_ids
        )
        (tmp_path / name).write_bytes(gzip.compress(structure_path.read_bytes()))
    adapter = _StandardTargetAdapter()
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
        target_voxel_size=standard_expected()["voxel_size"],
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

    assert adapter.calls == ["7ae1.cif", "9tst.mmcif", "9tst.mmcif"]
    assert not journal.exists()
    assert first_core.read_bytes() == first_core_bytes
    original = json.loads(report.read_text())
    assert original["schema_version"] == "G2610210"
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
        "11736,7ae1,density.mrc.gz,7ae1.cif.gz\n"
    )
    artifact_name = "emd_11736_7ae1"
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
    adapter = _StandardTargetAdapter()
    output_dir = tmp_path / "artifacts"
    summary = tmp_path / "summary.csv"
    paths = preprocess_manifest_csv(
        manifest,
        output_dir,
        artifact_reuse_dirs=[reuse_dir],
        preprocess_adapter=adapter,
        target_voxel_size=standard_expected()["voxel_size"],
        summary_csv=summary,
        continue_on_preprocess_error=False,
    )

    artifact = output_dir / artifact_name
    assert paths == [artifact]
    assert adapter.calls == ["7ae1.cif"]
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
        target_voxel_size=standard_expected()["voxel_size"],
        summary_csv=summary,
        continue_on_preprocess_error=False,
    ) == [artifact]
    assert adapter.calls == ["7ae1.cif"]
    assert opened_single_files == []
    with summary.open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["status"] == "reused_external"


def test_required_sidecar_enriches_reused_target_without_preprocessing(tmp_path):
    manifest, _, sequence = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path\n"
        "11736,7ae1,density.mrc.gz,7ae1.cif.gz\n"
    )
    adapter = _StandardTargetAdapter()
    output_dir = tmp_path / "artifacts"
    summary = tmp_path / "summary.csv"
    arguments = dict(
        manifest_csv=manifest,
        artifact_output_dir=output_dir,
        preprocess_adapter=adapter,
        target_voxel_size=standard_expected()["voxel_size"],
        summary_csv=summary,
        continue_on_preprocess_error=False,
    )
    artifact = preprocess_manifest_csv(**arguments)[0]
    assert "sequence" not in load_preprocessed_artifact(artifact)
    core_before = (artifact / "core.npz").read_bytes()

    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path,sequence_embedding_path\n"
        "11736,7ae1,density.mrc.gz,7ae1.cif.gz,sequence.npz\n"
    )
    assert preprocess_manifest_csv(**arguments, require_sequence_embeddings=True) == [
        artifact
    ]

    assert adapter.calls == ["7ae1.cif"]
    assert (artifact / "core.npz").read_bytes() == core_before
    np.testing.assert_array_equal(
        load_preprocessed_artifact(artifact)["sequence"], sequence
    )
    assert validate_preprocessed_artifact(
        artifact, require_sequence_embeddings=True
    ).sequence_length == len(sequence)
    with summary.open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["status"] == "reused"


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_reuse_rebuilds_artifact_with_unreadable_optional_sequence(tmp_path, damage):
    manifest, _, _ = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path,sequence_embedding_path\n"
        "11736,7ae1,density.mrc.gz,7ae1.cif.gz,sequence.npz\n"
    )
    adapter = _StandardTargetAdapter()
    arguments = dict(
        manifest_csv=manifest,
        artifact_output_dir=tmp_path / "artifacts",
        preprocess_adapter=adapter,
        target_voxel_size=standard_expected()["voxel_size"],
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
        "11736,7ae1,density.mrc.gz,7ae1.cif.gz\n"
    )

    assert preprocess_manifest_csv(**arguments) == [artifact]

    assert adapter.calls == ["7ae1.cif", "7ae1.cif"]
    assert "sequence" not in load_preprocessed_artifact(artifact)
    assert validate_preprocessed_artifact(artifact).sequence_length is None


def test_reuse_repairs_corrupt_sequence_from_replacement_without_reprocessing(
    tmp_path,
):
    manifest, _, sequence = _compressed_inputs(tmp_path)
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path,sequence_embedding_path\n"
        "11736,7ae1,density.mrc.gz,7ae1.cif.gz,sequence.npz\n"
    )
    adapter = _StandardTargetAdapter()
    arguments = dict(
        manifest_csv=manifest,
        artifact_output_dir=tmp_path / "artifacts",
        preprocess_adapter=adapter,
        target_voxel_size=standard_expected()["voxel_size"],
        continue_on_preprocess_error=False,
        require_sequence_embeddings=True,
    )
    artifact = preprocess_manifest_csv(**arguments)[0]
    core_before = (artifact / "core.npz").read_bytes()
    (artifact / "sequence.npz").write_bytes(b"broken archive")

    assert preprocess_manifest_csv(**arguments) == [artifact]

    assert adapter.calls == ["7ae1.cif"]
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
        "11736,7ae1,density.mrc.gz,7ae1.cif.gz\n"
    )
    artifact = preprocess_manifest_csv(
        manifest,
        tmp_path / "artifacts",
        preprocess_adapter=_StandardTargetAdapter(),
        target_voxel_size=standard_expected()["voxel_size"],
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
        "11736,7ae1,density.mrc.gz,7ae1.cif.gz\n"
    )
    adapter = _StandardTargetAdapter()

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
            target_voxel_size=standard_expected()["voxel_size"],
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
    raw_structure = tmp_path / "7ae1.cif"
    write_standard_structure(raw_structure)
    output = tmp_path / "artifact"
    adapter = _StandardTargetAdapter()
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
    assert adapter.calls == ["7ae1.cif"]
    validated = validate_preprocessed_artifact(output)
    assert validated.protein_nodes == standard_expected()["protein_nodes"]
    assert validated.nucleotide_nodes == standard_expected()["nucleotide_nodes"]
