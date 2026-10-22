"""Resource lifetime regressions for parallel manifest workers."""

from __future__ import annotations

import weakref
from pathlib import Path

import numpy as np
import pytest

from cryoflake.constants import NodeKind
from cryoflake.preprocess import manifest as preprocess_manifest
from cryoflake.preprocess import worker
from cryoflake.preprocess.checkpoint import PreprocessCheckpoint
from cryoflake.preprocess.memory_guard import PreprocessMemoryAdmission
from cryoflake.preprocess.models import _PreprocessManifestRowTask
from cryoflake.preprocess.reuse import _ValidatedTrainingArtifact
from cryoflake.storage.upload_cache import BatchedArtifactUploadCache


def _task(tmp_path, *, preprocess_adapter, admission, upload_cache=None):
    map_path = tmp_path / "map.mrc"
    structure_path = tmp_path / "structure.cif"
    map_path.write_bytes(b"map")
    structure_path.write_text("data_test\n")
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir(exist_ok=True)
    return _PreprocessManifestRowTask(
        row_index=1,
        row={
            "emdb_id": "1001",
            "wwpdb_id": "1abc",
            "map_path": str(map_path),
            "structure_path": str(structure_path),
        },
        manifest_dir=tmp_path,
        artifact_output_dir=output_dir,
        artifact_s3_prefix=("s3://bucket/artifacts" if upload_cache else None),
        s3_endpoint_url=None,
        artifact_reuse_dirs=(),
        target_voxel_size=1.0,
        force_preprocess=True,
        skip_missing=False,
        fix_shift=True,
        preprocess_adapter=preprocess_adapter,
        interface_contact_cutoff=5.0,
        require_sequence_embeddings=False,
        sequence_embedding_loader=lambda _: {},
        continue_on_preprocess_error=False,
        download_errors={},
        split_seed=0,
        upload_cache=upload_cache,
        memory_admission=admission,
    )


@pytest.mark.parametrize("validation_fails", [False, True])
def test_worker_releases_loaded_arrays_before_memory_reservation(
    monkeypatch, tmp_path, validation_fails
):
    array_refs = []

    class CheckingAdmission(PreprocessMemoryAdmission):
        def release(self, requested_bytes):
            # A waiting worker can allocate as soon as this reservation is
            # released, so the preceding worker must already drop its arrays.
            assert array_refs and all(ref() is None for ref in array_refs)
            super().release(requested_bytes)

    def load_payload(*args, **kwargs):
        grid = np.zeros((2, 2, 2), dtype=np.float32)
        positions = np.zeros((0, 3), dtype=np.float32)
        array_refs.extend((weakref.ref(grid), weakref.ref(positions)))
        return {
            "density_grid": grid,
            "voxel_size": 1.0,
            "extension_targets": {
                "component": {
                    "positions": positions,
                    "component_id": np.empty(0, dtype=str),
                    "node_kind": np.empty(0, dtype=np.int64),
                    "interface_group_id": np.empty(0, dtype=str),
                }
            },
        }

    admission = CheckingAdmission(10)
    task = _task(tmp_path, preprocess_adapter=lambda **_: None, admission=admission)
    monkeypatch.setattr(worker, "estimate_preprocess_memory_or_none", lambda *_: None)
    monkeypatch.setattr(worker, "load_preprocessed_artifact", load_payload)

    def validate_payload(mapping, **kwargs):
        if validation_fails:
            normalized = np.zeros((2, 3), dtype=np.float32)
            array_refs.append(weakref.ref(normalized))
            raise ValueError("Invalid target array")
        return _ValidatedTrainingArtifact(
            grid_shape=mapping["density_grid"].shape,
            voxel_size=1.0,
            protein_nodes=0,
            nucleotide_nodes=0,
            component_targets=mapping["extension_targets"]["component"],
            sequence_shape=None,
        )

    monkeypatch.setattr(worker, "_validate_training_artifact_mapping", validate_payload)

    if validation_fails:
        with pytest.raises(ValueError, match="does not satisfy the training contract"):
            worker._preprocess_manifest_row(task)
    else:
        result = worker._preprocess_manifest_row(task)
        assert result.preprocess_succeeded
    assert admission.reserved_bytes == 0


def test_worker_derives_sorted_groups_and_counts_interface_nodes_from_lean_targets(
    monkeypatch, tmp_path
):
    component = {
        "positions": np.zeros((5, 3), dtype=np.float32),
        "component_id": np.asarray(["LIG", "HOH", "LIG", "ATP", "DOD"]),
        "node_kind": np.asarray(
            [
                NodeKind.NON_POLYMER,
                NodeKind.WATER,
                NodeKind.NON_POLYMER,
                NodeKind.NON_POLYMER,
                NodeKind.WATER,
            ],
            dtype=np.int64,
        ),
        "interface_group_id": np.asarray(
            [
                "LIG|bbbbbbbbbbbbbbbb",
                "",
                "LIG|bbbbbbbbbbbbbbbb",
                "ATP|aaaaaaaaaaaaaaaa",
                "DOD|cccccccccccccccc",
            ]
        ),
    }
    original = {key: value.copy() for key, value in component.items()}
    artifact_mapping = {
        "density_grid": np.zeros((2, 2, 2), dtype=np.float32),
        "voxel_size": 1.0,
        "extension_targets": {"component": component},
    }
    admission = PreprocessMemoryAdmission(10)
    task = _task(tmp_path, preprocess_adapter=lambda **_: None, admission=admission)
    monkeypatch.setattr(worker, "estimate_preprocess_memory_or_none", lambda *_: None)
    monkeypatch.setattr(
        worker, "load_preprocessed_artifact", lambda *_, **__: artifact_mapping
    )
    monkeypatch.setattr(
        worker,
        "_validate_training_artifact_mapping",
        lambda mapping, **_: _ValidatedTrainingArtifact(
            grid_shape=mapping["density_grid"].shape,
            voxel_size=1.0,
            protein_nodes=0,
            nucleotide_nodes=0,
            component_targets=mapping["extension_targets"]["component"],
            sequence_shape=None,
        ),
    )

    result = worker._preprocess_manifest_row(task)

    assert result.preprocess_succeeded
    assert result.summary_row["component_node_count"] == "5"
    assert result.summary_row["component_interface_count"] == "4"
    assert result.summary_row["component_group_ids"] == "ATP;LIG"
    expected_interfaces = (
        "ATP|aaaaaaaaaaaaaaaa",
        "DOD|cccccccccccccccc",
        "LIG|bbbbbbbbbbbbbbbb",
    )
    assert result.summary_row["interface_group_ids"] == ";".join(expected_interfaces)
    assert result.grouped_split_record.component_group_ids == ("ATP", "LIG")
    assert result.grouped_split_record.interface_group_ids == expected_interfaces
    assert set(component) == set(original)
    for key, value in original.items():
        np.testing.assert_array_equal(component[key], value)
    assert admission.reserved_bytes == 0


def test_interrupted_adapter_discards_unsubmitted_upload_and_releases_memory(
    monkeypatch, tmp_path
):
    class AdapterInterrupted(BaseException):
        pass

    def interrupted_adapter(*, artifact_reference, **kwargs):
        artifact = Path(artifact_reference)
        artifact.mkdir()
        (artifact / "partial.npz").write_bytes(b"unfinished")
        raise AdapterInterrupted("interrupted before upload")

    admission = PreprocessMemoryAdmission(10)
    cache_parent = tmp_path / "artifacts"
    uploaded = []
    with BatchedArtifactUploadCache(
        cache_parent, lambda *args: uploaded.append(args), num_workers=1
    ) as cache:
        task = _task(
            tmp_path,
            preprocess_adapter=interrupted_adapter,
            admission=admission,
            upload_cache=cache,
        )
        monkeypatch.setattr(
            worker, "estimate_preprocess_memory_or_none", lambda *_: None
        )

        with pytest.raises(AdapterInterrupted, match="interrupted before upload"):
            worker._preprocess_manifest_row(task)

        assert not (cache.cache_dir / "row_00000001").exists()
        assert admission.reserved_bytes == 0
    assert uploaded == []


def test_task_allocation_failure_releases_checkpoint_ownership(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "emdb_id,wwpdb_id,map_path,structure_path\n1001,1abc,map.mrc,structure.cif\n"
    )
    summary = tmp_path / "summary.csv"
    created_checkpoints = []

    def create_checkpoint(*args, **kwargs):
        checkpoint = PreprocessCheckpoint(*args, **kwargs)
        created_checkpoints.append(checkpoint)
        return checkpoint

    def allocation_failure(**kwargs):
        raise MemoryError("task allocation failed")

    monkeypatch.setattr(preprocess_manifest, "PreprocessCheckpoint", create_checkpoint)
    monkeypatch.setattr(
        preprocess_manifest, "_PreprocessManifestRowTask", allocation_failure
    )

    try:
        with pytest.raises(MemoryError, match="task allocation failed"):
            preprocess_manifest.preprocess_manifest_csv(
                manifest,
                tmp_path / "artifacts",
                summary_csv=summary,
                preprocess_adapter=lambda **_: None,
            )
        # Reopening the journal requires the failed invocation to have closed
        # both its journal stream and the ownership lock.
        with PreprocessCheckpoint(summary, created_checkpoints[0].contract):
            pass
    finally:
        for checkpoint in created_checkpoints:
            checkpoint.close()
