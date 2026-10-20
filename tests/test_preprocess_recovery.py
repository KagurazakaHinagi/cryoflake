from __future__ import annotations

import functools
import gzip
import hashlib
import json
import os
import re
import shutil
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import mrcfile
import numpy as np
import pytest

from cryoflake.manifests.manifest import get_preprocessed_artifact_name
from cryoflake.manifests.splits import GroupedSplitRecord
from cryoflake.preprocess import checkpoint_identity, results, summary, worker
from cryoflake.preprocess import manifest as preprocess_manifest
from cryoflake.preprocess.artifact_identity import preprocessed_artifact_identity
from cryoflake.preprocess.checkpoint import (
    CheckpointContractMismatchError,
    PreprocessCheckpoint,
    _local_cache_identity,
    checkpoint_cache_matches,
)
from cryoflake.preprocess.checkpoint_identity import callable_identity
from cryoflake.preprocess.memory_guard import (
    PreprocessMemoryAdmission,
    PreprocessMemoryAdmissionCancelled,
    _cgroup_memory_limit_bytes,
    estimate_preprocess_memory,
)
from cryoflake.preprocess.models import _PreprocessManifestRowResult
from cryoflake.storage.upload_cache import BatchedArtifactUploadCache

_CALLABLE_TEST_MODE = 1


def _global_mode_preprocess():
    return _CALLABLE_TEST_MODE


def _calls_global_mode_preprocess():
    return _global_mode_preprocess()


def _write_manifest(tmp_path: Path, rows: int = 2) -> Path:
    manifest_path = tmp_path / "manifest.csv"
    lines = ["emdb_id,wwpdb_id,map_path,structure_path"]
    for row_index in range(1, rows + 1):
        map_path = tmp_path / f"map_{row_index}.mrc"
        structure_path = tmp_path / f"{row_index}abc.cif"
        map_path.write_bytes(b"test map")
        structure_path.write_text("data_test\n")
        lines.append(
            f"{1000 + row_index},{row_index}abc,{map_path.name},{structure_path.name}"
        )
    manifest_path.write_text("\n".join(lines) + "\n")
    return manifest_path


def _successful_result(task, artifact_reference: Path | str):
    map_path = task.manifest_dir / task.row["map_path"]
    structure_path = task.manifest_dir / task.row["structure_path"]
    summary_row = summary._preprocess_summary_row(
        task.row,
        map_path=map_path,
        structure_path=structure_path,
        artifact_reference=artifact_reference,
        split_seed=task.split_seed,
    )
    summary._finish_summary_row(
        summary_row,
        status="processed",
        success=True,
    )
    return _PreprocessManifestRowResult(
        row_index=task.row_index,
        artifact_reference=artifact_reference,
        summary_row=summary_row,
        grouped_split_record=GroupedSplitRecord(
            artifact_reference, task.row["wwpdb_id"]
        ),
        preprocess_succeeded=True,
    )


def _dummy_preprocess(**kwargs) -> None:
    del kwargs


@pytest.mark.parametrize("corruption", ["missing_group", "wrong_reference"])
def test_invalid_grouped_checkpoint_record_reprocesses_without_dropping_split_rows(
    tmp_path, monkeypatch, corruption
):
    manifest_path = _write_manifest(tmp_path, rows=3)
    summary_path = tmp_path / "summary.csv"
    calls = []

    def row_worker(task):
        calls.append(task.row_index)
        artifact_reference = task.artifact_output_dir / get_preprocessed_artifact_name(
            task.row
        )
        artifact_reference.mkdir(parents=True, exist_ok=True)
        (artifact_reference / "core.npz").write_bytes(b"core")
        (artifact_reference / "metadata.json").write_text('{"committed": true}\n')
        result = _successful_result(task, artifact_reference)
        return _PreprocessManifestRowResult(
            row_index=result.row_index,
            artifact_reference=result.artifact_reference,
            summary_row=result.summary_row,
            preprocess_succeeded=True,
            grouped_split_record=GroupedSplitRecord(
                artifact_reference,
                task.row["wwpdb_id"],
                (f"component-{task.row_index}",),
            ),
        )

    finalize = preprocess_manifest._finalize_preprocess_manifest_results
    interrupted = False

    def interrupt_first_finalization(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise OSError("interrupted before splitting")
        return finalize(*args, **kwargs)

    monkeypatch.setattr(worker, "_preprocess_manifest_row", row_worker)
    monkeypatch.setattr(
        preprocess_manifest,
        "_finalize_preprocess_manifest_results",
        interrupt_first_finalization,
    )
    options = {
        "artifact_output_dir": tmp_path / "artifacts",
        "summary_csv": summary_path,
        "preprocess_adapter": _dummy_preprocess,
        "artifact_list_path": tmp_path / "train.txt",
        "validation_artifact_list_path": tmp_path / "valid.txt",
        "split_report_path": tmp_path / "split.json",
        "split_strategy": "component",
        "validation_fraction": 0.5,
    }
    with pytest.raises(OSError, match="interrupted before splitting"):
        preprocess_manifest.preprocess_manifest_csv(manifest_path, **options)

    journal = Path(f"{summary_path}.checkpoint.jsonl")
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    first = next(record for record in records if record["row_index"] == 1)
    if corruption == "missing_group":
        first["grouped_split_record"] = None
    else:
        first["grouped_split_record"]["artifact_reference"] = str(tmp_path / "wrong")
    journal.write_text("".join(json.dumps(record) + "\n" for record in records))

    references = preprocess_manifest.preprocess_manifest_csv(manifest_path, **options)

    assert calls == [1, 2, 3, 1]
    train = set((tmp_path / "train.txt").read_text().splitlines())
    valid = set((tmp_path / "valid.txt").read_text().splitlines())
    assert train and valid and train.isdisjoint(valid)
    assert train | valid == set(map(str, references))
    report = json.loads((tmp_path / "split.json").read_text())
    assert report["num_records"] == len(references) == 3


def test_callable_identity_hashes_each_source_once_per_snapshot(monkeypatch, tmp_path):
    package = tmp_path / "fingerprint_cache_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    adapter_path = package / "adapter.py"
    source = (
        "def first():\n    return 1\n"
        "def second():\n    return 2\n"
        "def adapter():\n    return first() + second()\n"
    )
    adapter_path.write_text(source)
    extra_source = package / "extra.py"
    extra_source.write_text("SETTING = 1\n")
    module = ModuleType("fingerprint_cache_pkg.adapter")
    module.__file__ = str(adapter_path)
    exec(compile(source, str(adapter_path), "exec"), module.__dict__)
    reads = []
    original_hash = checkpoint_identity.sha256_file

    def tracked_hash(path):
        reads.append(path)
        return original_hash(path)

    monkeypatch.setattr(checkpoint_identity, "sha256_file", tracked_hash)
    first = callable_identity(module.adapter)
    assert sorted(reads) == sorted(package.glob("*.py"))
    assert checkpoint_identity._identity_cache.get() is None
    reads.clear()
    assert callable_identity(module.adapter) == first
    assert sorted(reads) == sorted(package.glob("*.py"))
    extra_source.write_text("SETTING = 2\n")
    assert callable_identity(module.adapter) != first


@pytest.mark.parametrize(
    "values",
    [
        np.arange(60, dtype=np.float32).reshape(3, 4, 5).transpose(2, 0, 1),
        np.arange(60, dtype=">i8").reshape(3, 4, 5)[::-1, :, ::2],
        np.asarray(["ALA", "GLY", "VAL", "ASP"])[::-1],
        np.asarray(7),
        np.empty((2, 0), dtype=np.float32),
    ],
)
def test_checkpoint_array_hash_matches_c_order_bytes(values):
    contiguous = np.ascontiguousarray(values)
    expected = hashlib.sha256(
        memoryview(contiguous).cast("B") if contiguous.nbytes else b""
    ).hexdigest()
    assert checkpoint_identity._array_data_sha256(values) == expected


def test_checkpoint_array_hash_bounds_noncontiguous_temporary(monkeypatch):
    values = np.arange(1024 * 1024 * 8, dtype=np.float32).reshape(1024, -1).T
    original_contiguous = np.ascontiguousarray
    expected = hashlib.sha256(memoryview(original_contiguous(values)).cast("B"))
    chunk_sizes = []

    def bounded_contiguous(chunk):
        chunk_sizes.append(chunk.nbytes)
        assert chunk.nbytes <= checkpoint_identity._ARRAY_HASH_CHUNK_BYTES
        return original_contiguous(chunk)

    monkeypatch.setattr(checkpoint_identity.np, "ascontiguousarray", bounded_contiguous)
    assert checkpoint_identity._array_data_sha256(values) == expected.hexdigest()
    assert len(chunk_sizes) > 1


@pytest.mark.parametrize(
    "invalid_options",
    [
        {"validation_fraction": float("nan")},
        {"validation_fraction": float("inf")},
        {"validation_fraction": 0},
        {"validation_fraction": 1},
        {"preprocess_adapter": "module:adapter"},
        {"sequence_embedding_loader": None},
        {"progress_callback": 5},
        {"validation_artifact_list_path": "validation.txt"},
    ],
)
def test_direct_manifest_preflight_rejects_invalid_options_before_outputs(
    tmp_path, invalid_options
):
    manifest_path = _write_manifest(tmp_path, rows=1)
    output_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.csv"
    arguments = {
        "manifest_csv": manifest_path,
        "artifact_output_dir": output_dir,
        "summary_csv": summary_path,
        "preprocess_adapter": _dummy_preprocess,
        **invalid_options,
    }

    with pytest.raises(ValueError):
        preprocess_manifest.preprocess_manifest_csv(**arguments)

    assert not output_dir.exists()
    assert not summary_path.exists()
    assert not Path(f"{summary_path}.checkpoint.lock").exists()


@pytest.mark.parametrize(
    "collision",
    [
        "manifest",
        "train",
        "summary",
        "checkpoint.json",
        "checkpoint.jsonl",
        "checkpoint.lock",
    ],
)
def test_direct_manifest_preflight_rejects_colliding_outputs(tmp_path, collision):
    manifest_path = _write_manifest(tmp_path, rows=1)
    manifest_before = manifest_path.read_bytes()
    output_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.csv"
    train_path = tmp_path / "train.txt"
    arguments = {
        "manifest_csv": manifest_path,
        "artifact_output_dir": output_dir,
        "summary_csv": summary_path,
        "artifact_list_path": train_path,
        "preprocess_adapter": _dummy_preprocess,
    }
    if collision == "manifest":
        arguments["artifact_list_path"] = manifest_path
    elif collision == "train":
        arguments["validation_artifact_list_path"] = train_path
    elif collision == "summary":
        arguments["artifact_list_path"] = summary_path
    else:
        arguments["artifact_list_path"] = Path(f"{summary_path}.{collision}")

    with pytest.raises(ValueError, match="must use distinct files"):
        preprocess_manifest.preprocess_manifest_csv(**arguments)

    assert manifest_path.read_bytes() == manifest_before
    assert not output_dir.exists()
    assert not summary_path.exists()
    assert not Path(f"{summary_path}.checkpoint.lock").exists()


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_direct_manifest_preflight_detects_aliased_input_output_files(
    tmp_path, link_type
):
    manifest_path = _write_manifest(tmp_path, rows=1)
    alias = tmp_path / "dataset.txt"
    if link_type == "symlink":
        alias.symlink_to(manifest_path)
    else:
        alias.hardlink_to(manifest_path)
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="must use distinct files"):
        preprocess_manifest.preprocess_manifest_csv(
            manifest_path,
            tmp_path / "artifacts",
            artifact_list_path=alias,
            preprocess_adapter=_dummy_preprocess,
        )

    assert manifest_path.read_bytes() == before
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("output_name", "artifact_member"),
    [
        ("artifact_list_path", "core.npz"),
        ("validation_artifact_list_path", "metadata.json"),
        ("summary_csv", ""),
        ("summary_csv", "nested/summary.csv"),
        ("split_report_path", "sequence.npz"),
    ],
)
def test_direct_manifest_preflight_protects_artifact_directories(
    tmp_path, monkeypatch, output_name, artifact_member
):
    manifest_path = _write_manifest(tmp_path, rows=1)
    output_dir = tmp_path / "artifacts"
    artifact_root = output_dir / "emd_1001_1abc"
    arguments = {
        "manifest_csv": manifest_path,
        "artifact_output_dir": output_dir,
        "artifact_list_path": tmp_path / "train.txt",
        "validation_artifact_list_path": tmp_path / "valid.txt",
        "summary_csv": tmp_path / "summary.csv",
        "split_strategy": "component",
        "preprocess_adapter": _dummy_preprocess,
        output_name: artifact_root / artifact_member,
    }
    monkeypatch.setattr(
        preprocess_manifest,
        "_run_preprocess_manifest_row_tasks",
        lambda *args, **kwargs: pytest.fail("Invalid output reached row workers"),
    )

    with pytest.raises(ValueError, match="inside a preprocessed artifact directory"):
        preprocess_manifest.preprocess_manifest_csv(**arguments)

    assert not output_dir.exists()
    assert not (tmp_path / "summary.csv.checkpoint.lock").exists()


def test_direct_manifest_preflight_resolves_symlinks_into_artifact_directories(
    tmp_path, monkeypatch
):
    manifest_path = _write_manifest(tmp_path, rows=1)
    output_dir = tmp_path / "artifacts"
    alias = tmp_path / "dataset.txt"
    alias.symlink_to(output_dir / "emd_1001_1abc" / "metadata.json")
    monkeypatch.setattr(
        preprocess_manifest,
        "_run_preprocess_manifest_row_tasks",
        lambda *args, **kwargs: pytest.fail("Invalid output reached row workers"),
    )

    with pytest.raises(ValueError, match="inside a preprocessed artifact directory"):
        preprocess_manifest.preprocess_manifest_csv(
            manifest_path,
            output_dir,
            artifact_list_path=alias,
            preprocess_adapter=_dummy_preprocess,
        )

    assert alias.is_symlink()
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "source_name", ["map_path", "structure_path", "sequence_embedding_path"]
)
@pytest.mark.parametrize("link_type", ["direct", "symlink", "hardlink"])
def test_direct_manifest_preflight_protects_manifest_member_inputs(
    tmp_path, monkeypatch, source_name, link_type
):
    manifest_path = _write_manifest(tmp_path, rows=1)
    sequence_path = tmp_path / "sequence.npz"
    sequence_path.write_bytes(b"source embeddings")
    manifest_path.write_text(
        manifest_path.read_text()
        .replace(
            "map_path,structure_path\n",
            "map_path,structure_path,sequence_embedding_path\n",
        )
        .replace("map_1.mrc,1abc.cif\n", "map_1.mrc,1abc.cif,sequence.npz\n")
    )
    source_path = {
        "map_path": tmp_path / "map_1.mrc",
        "structure_path": tmp_path / "1abc.cif",
        "sequence_embedding_path": sequence_path,
    }[source_name]
    destination = source_path
    if link_type != "direct":
        destination = tmp_path / "dataset.txt"
        if link_type == "symlink":
            destination.symlink_to(source_path)
        else:
            destination.hardlink_to(source_path)
    before = source_path.read_bytes()
    output_dir = tmp_path / "artifacts"
    monkeypatch.setattr(
        preprocess_manifest,
        "_run_preprocess_manifest_row_tasks",
        lambda *args, **kwargs: pytest.fail("Invalid output reached row workers"),
    )

    with pytest.raises(ValueError, match="must use distinct files"):
        preprocess_manifest.preprocess_manifest_csv(
            manifest_path,
            output_dir,
            artifact_list_path=destination,
            preprocess_adapter=_dummy_preprocess,
        )

    assert source_path.read_bytes() == before
    assert not output_dir.exists()


def test_checkpoint_resumes_durable_local_rows_after_interruption(
    tmp_path,
    monkeypatch,
):
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.csv"
    calls = []
    fail_second_row = True

    def fake_row_worker(task):
        nonlocal fail_second_row
        calls.append(task.row_index)
        if task.row_index == 2 and fail_second_row:
            raise RuntimeError("simulated interruption")
        artifact_reference = task.artifact_output_dir / get_preprocessed_artifact_name(
            task.row
        )
        artifact_reference.mkdir(parents=True, exist_ok=True)
        (artifact_reference / "core.npz").write_bytes(b"core")
        (artifact_reference / "metadata.json").write_text('{"committed": true}\n')
        return _successful_result(task, artifact_reference)

    monkeypatch.setattr(
        worker,
        "_preprocess_manifest_row",
        fake_row_worker,
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        preprocess_manifest.preprocess_manifest_csv(
            manifest_path,
            artifact_output_dir=output_dir,
            summary_csv=summary_path,
            preprocess_adapter=_dummy_preprocess,
        )

    checkpoint_metadata = Path(f"{summary_path}.checkpoint.json")
    checkpoint_journal = Path(f"{summary_path}.checkpoint.jsonl")
    assert checkpoint_metadata.is_file()
    assert checkpoint_journal.is_file()
    assert calls == [1, 2]
    checkpoint_records = [
        json.loads(line) for line in checkpoint_journal.read_text().splitlines()
    ]
    assert checkpoint_records[0]["state"] == "durable_success"
    assert "preprocess_succeeded" not in checkpoint_records[0]

    fail_second_row = False
    artifact_references = preprocess_manifest.preprocess_manifest_csv(
        manifest_path,
        artifact_output_dir=output_dir,
        summary_csv=summary_path,
        preprocess_adapter=_dummy_preprocess,
    )

    assert calls == [1, 2, 2]
    assert len(artifact_references) == 2
    assert summary_path.is_file()
    assert not checkpoint_metadata.exists()
    assert not checkpoint_journal.exists()


def test_checkpoint_token_stabilizes_stateful_pipeline_callables(
    tmp_path,
    monkeypatch,
):
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.csv"
    calls = []
    runtime_state = {"preprocess_calls": 0, "loader_calls": 0}
    fail_second_row = True

    def stateful_preprocess(**kwargs):
        del kwargs
        runtime_state["preprocess_calls"] += 1

    def stateful_loader(path):
        del path
        runtime_state["loader_calls"] += 1
        return {}

    def fake_row_worker(task):
        nonlocal fail_second_row
        calls.append(task.row_index)
        if task.row_index == 2 and fail_second_row:
            raise RuntimeError("simulated interruption")
        task.preprocess_adapter()
        task.sequence_embedding_loader(Path("unused"))
        artifact_reference = task.artifact_output_dir / get_preprocessed_artifact_name(
            task.row
        )
        artifact_reference.mkdir(parents=True, exist_ok=True)
        (artifact_reference / "core.npz").write_bytes(b"core")
        (artifact_reference / "metadata.json").write_text('{"committed": true}\n')
        return _successful_result(task, artifact_reference)

    monkeypatch.setattr(
        worker,
        "_preprocess_manifest_row",
        fake_row_worker,
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        preprocess_manifest.preprocess_manifest_csv(
            manifest_path,
            artifact_output_dir=output_dir,
            summary_csv=summary_path,
            preprocess_adapter=stateful_preprocess,
            sequence_embedding_loader=stateful_loader,
            preprocess_checkpoint_token="pipeline-v1",
        )

    with pytest.raises(CheckpointContractMismatchError, match="does not match"):
        preprocess_manifest.preprocess_manifest_csv(
            manifest_path,
            artifact_output_dir=output_dir,
            summary_csv=summary_path,
            preprocess_adapter=stateful_preprocess,
            sequence_embedding_loader=stateful_loader,
            preprocess_checkpoint_token="pipeline-v2",
        )

    fail_second_row = False
    artifact_references = preprocess_manifest.preprocess_manifest_csv(
        manifest_path,
        artifact_output_dir=output_dir,
        summary_csv=summary_path,
        preprocess_adapter=stateful_preprocess,
        sequence_embedding_loader=stateful_loader,
        preprocess_checkpoint_token="pipeline-v1",
    )

    assert calls == [1, 2, 2]
    assert runtime_state == {"preprocess_calls": 2, "loader_calls": 2}
    assert len(artifact_references) == 2


def test_checkpoint_rejects_changed_download_error_state(
    tmp_path,
    monkeypatch,
):
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.csv"
    fail_second_row = True

    def fake_row_worker(task):
        nonlocal fail_second_row
        if task.row_index == 2 and fail_second_row:
            raise RuntimeError("simulated interruption")
        artifact_reference = task.artifact_output_dir / get_preprocessed_artifact_name(
            task.row
        )
        artifact_reference.mkdir(parents=True, exist_ok=True)
        (artifact_reference / "core.npz").write_bytes(b"core")
        (artifact_reference / "metadata.json").write_text('{"committed": true}\n')
        return _successful_result(task, artifact_reference)

    monkeypatch.setattr(
        worker,
        "_preprocess_manifest_row",
        fake_row_worker,
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        preprocess_manifest.preprocess_manifest_csv(
            manifest_path,
            artifact_output_dir=output_dir,
            summary_csv=summary_path,
            preprocess_adapter=_dummy_preprocess,
        )

    first_map_path = (tmp_path / "map_1.mrc").resolve()
    with pytest.raises(CheckpointContractMismatchError, match="does not match"):
        preprocess_manifest.preprocess_manifest_csv(
            manifest_path,
            artifact_output_dir=output_dir,
            summary_csv=summary_path,
            preprocess_adapter=_dummy_preprocess,
            download_errors={
                str(first_map_path): "simulated failed refresh",
            },
        )


def test_changed_artifact_member_invalidates_checkpoint_resume(
    tmp_path,
    monkeypatch,
):
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.csv"
    calls = []
    fail_second_row = True

    def fake_row_worker(task):
        nonlocal fail_second_row
        calls.append(task.row_index)
        if task.row_index == 2 and fail_second_row:
            raise RuntimeError("simulated interruption")
        artifact_reference = task.artifact_output_dir / get_preprocessed_artifact_name(
            task.row
        )
        artifact_reference.mkdir(parents=True, exist_ok=True)
        (artifact_reference / "core.npz").write_bytes(f"core {len(calls)}".encode())
        (artifact_reference / "metadata.json").write_text('{"committed": true}\n')
        return _successful_result(task, artifact_reference)

    monkeypatch.setattr(
        worker,
        "_preprocess_manifest_row",
        fake_row_worker,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        preprocess_manifest.preprocess_manifest_csv(
            manifest_path,
            artifact_output_dir=output_dir,
            summary_csv=summary_path,
            preprocess_adapter=_dummy_preprocess,
        )

    first_artifact = output_dir / "emd_1001_1abc"
    stored_identity = preprocessed_artifact_identity(first_artifact)
    assert stored_identity is not None
    (first_artifact / "core.npz").unlink()
    assert preprocessed_artifact_identity(first_artifact) is None

    fail_second_row = False
    artifact_references = preprocess_manifest.preprocess_manifest_csv(
        manifest_path,
        artifact_output_dir=output_dir,
        summary_csv=summary_path,
        preprocess_adapter=_dummy_preprocess,
    )

    assert calls == [1, 2, 1, 2]
    assert len(artifact_references) == 2


def test_incompatible_checkpoint_is_preserved_until_explicit_reset(tmp_path):
    summary_path = tmp_path / "summary.csv"
    original_contract = {"target": "s3://old-bucket/artifacts", "version": 1}
    with PreprocessCheckpoint(summary_path, original_contract) as checkpoint:
        checkpoint.record({"row_index": 1, "state": "terminal_failure"})

    metadata_path = Path(f"{summary_path}.checkpoint.json")
    journal_path = Path(f"{summary_path}.checkpoint.jsonl")
    original_metadata = metadata_path.read_bytes()
    original_journal = journal_path.read_bytes()

    with pytest.raises(CheckpointContractMismatchError, match="does not match"):
        PreprocessCheckpoint(
            summary_path,
            {"target": "s3://new-bucket/npz", "version": 2},
        )

    assert metadata_path.read_bytes() == original_metadata
    assert journal_path.read_bytes() == original_journal

    replacement = PreprocessCheckpoint(
        summary_path,
        {"target": "s3://new-bucket/npz", "version": 2},
        reset=True,
    )
    try:
        assert replacement.records == {}
    finally:
        replacement.close()


def test_checkpoint_repairs_torn_journal_tail(tmp_path):
    summary_path = tmp_path / "summary.csv"
    contract = {"test": "torn-tail"}
    with PreprocessCheckpoint(summary_path, contract) as checkpoint:
        checkpoint.record({"row_index": 1, "state": "terminal_failure"})

    journal_path = Path(f"{summary_path}.checkpoint.jsonl")
    with journal_path.open("ab") as journal:
        journal.write(b'{"row_index":2')

    resumed = PreprocessCheckpoint(summary_path, contract)
    try:
        assert set(resumed.records) == {1}
        assert journal_path.read_bytes().endswith(b"\n")
        resumed.record({"row_index": 2, "state": "terminal_failure"})
    finally:
        resumed.close()

    verified = PreprocessCheckpoint(summary_path, contract)
    try:
        assert set(verified.records) == {1, 2}
    finally:
        verified.close()


@pytest.mark.parametrize("metadata", [[], None, "invalid", 42])
def test_checkpoint_rejects_non_object_metadata_without_losing_progress(
    tmp_path, metadata
):
    summary_path = tmp_path / "summary.csv"
    contract = {"test": "invalid-metadata"}
    with PreprocessCheckpoint(summary_path, contract) as checkpoint:
        checkpoint.record({"row_index": 1, "state": "terminal_failure"})
    metadata_path = Path(f"{summary_path}.checkpoint.json")
    journal_path = Path(f"{summary_path}.checkpoint.jsonl")
    metadata_path.write_text(json.dumps(metadata))
    original_journal = journal_path.read_bytes()

    with pytest.raises(CheckpointContractMismatchError, match="does not match"):
        PreprocessCheckpoint(summary_path, contract)

    assert metadata_path.read_text() == json.dumps(metadata)
    assert journal_path.read_bytes() == original_journal
    # A failed recovery attempt must release ownership for an explicit reset.
    with PreprocessCheckpoint(summary_path, contract, reset=True) as checkpoint:
        assert checkpoint.records == {}


def test_checkpoint_reset_skips_non_object_journal_records(tmp_path):
    summary_path = tmp_path / "summary.csv"
    contract = {"test": "invalid-record"}
    with PreprocessCheckpoint(summary_path, contract):
        pass
    journal_path = Path(f"{summary_path}.checkpoint.jsonl")
    journal_path.write_text('[]\nnull\n"invalid"\n42\n')

    with PreprocessCheckpoint(summary_path, contract, reset=True) as checkpoint:
        assert checkpoint.records == {}


def test_checkpoint_skips_malformed_row_identifiers_without_shadowing_progress(
    tmp_path,
):
    summary_path = tmp_path / "summary.csv"
    contract = {"test": "invalid-row-identifier"}
    with PreprocessCheckpoint(summary_path, contract) as checkpoint:
        checkpoint.record({"row_index": 1, "state": "durable_success"})
        generation = checkpoint.records[1]["checkpoint_generation"]
    journal_path = Path(f"{summary_path}.checkpoint.jsonl")
    with journal_path.open("a") as journal:
        for row_index in (True, 1.5, "1", float("inf"), [], {}, None):
            journal.write(
                json.dumps(
                    {
                        "row_index": row_index,
                        "state": "terminal_failure",
                        "checkpoint_generation": generation,
                    }
                )
                + "\n"
            )

    with PreprocessCheckpoint(summary_path, contract) as checkpoint:
        assert set(checkpoint.records) == {1}
        assert checkpoint.records[1]["state"] == "durable_success"


@pytest.mark.parametrize("row_index", [True, 1.5, "1", float("inf"), 0, -1])
def test_checkpoint_refuses_invalid_row_identifiers_without_writing(
    tmp_path, row_index
):
    with PreprocessCheckpoint(tmp_path / "summary.csv", {}) as checkpoint:
        with pytest.raises(ValueError, match="row_index must be a positive integer"):
            checkpoint.record({"row_index": row_index})
        assert checkpoint.records == {}
        assert checkpoint.journal_path.read_bytes() == b""


def test_checkpoint_skips_non_utf8_journal_lines_and_keeps_later_progress(tmp_path):
    summary_path = tmp_path / "summary.csv"
    contract = {"test": "non-utf8-line"}
    with PreprocessCheckpoint(summary_path, contract) as checkpoint:
        checkpoint.record({"row_index": 1, "state": "durable_success"})
        generation = checkpoint.records[1]["checkpoint_generation"]
    with Path(f"{summary_path}.checkpoint.jsonl").open("ab") as journal:
        journal.write(b"\xff\xfe\n")
        journal.write(
            json.dumps(
                {
                    "row_index": 2,
                    "state": "terminal_failure",
                    "checkpoint_generation": generation,
                }
            ).encode()
            + b"\n"
        )

    with PreprocessCheckpoint(summary_path, contract) as checkpoint:
        assert set(checkpoint.records) == {1, 2}
    with PreprocessCheckpoint(summary_path, contract, reset=True) as checkpoint:
        assert checkpoint.records == {}


def test_finalization_failure_resumes_without_reprocessing(
    tmp_path,
    monkeypatch,
):
    manifest_path = _write_manifest(tmp_path, rows=1)
    output_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.csv"
    calls = []

    def fake_row_worker(task):
        calls.append(task.row_index)
        artifact_reference = task.artifact_output_dir / "emd_1001_1abc"
        artifact_reference.mkdir(parents=True, exist_ok=True)
        (artifact_reference / "core.npz").write_bytes(b"core")
        (artifact_reference / "metadata.json").write_text('{"committed": true}\n')
        return _successful_result(task, artifact_reference)

    finalize = results._finalize_preprocess_manifest_results
    finalize_calls = 0

    def fail_first_finalization(*args, **kwargs):
        nonlocal finalize_calls
        finalize_calls += 1
        if finalize_calls == 1:
            raise OSError("simulated summary write failure")
        return finalize(*args, **kwargs)

    monkeypatch.setattr(
        worker,
        "_preprocess_manifest_row",
        fake_row_worker,
    )
    monkeypatch.setattr(
        preprocess_manifest,
        "_finalize_preprocess_manifest_results",
        fail_first_finalization,
    )

    with pytest.raises(OSError, match="summary write failure"):
        preprocess_manifest.preprocess_manifest_csv(
            manifest_path,
            artifact_output_dir=output_dir,
            summary_csv=summary_path,
            preprocess_adapter=_dummy_preprocess,
        )

    assert calls == [1]
    assert Path(f"{summary_path}.checkpoint.jsonl").is_file()

    artifact_references = preprocess_manifest.preprocess_manifest_csv(
        manifest_path,
        artifact_output_dir=output_dir,
        summary_csv=summary_path,
        preprocess_adapter=_dummy_preprocess,
    )

    assert calls == [1]
    assert artifact_references == [output_dir / "emd_1001_1abc"]
    assert summary_path.is_file()


def test_failed_remote_upload_resumes_from_retained_cache(
    tmp_path,
    monkeypatch,
):
    manifest_path = _write_manifest(tmp_path, rows=1)
    output_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.csv"
    remote_prefix = "s3://test-bucket/preprocessed"
    calls = []
    uploaded = set()
    fail_upload = True
    retained_cache_path = None

    def fake_row_worker(task):
        nonlocal retained_cache_path
        calls.append(task.row_index)
        artifact_name = get_preprocessed_artifact_name(task.row)
        remote_path = f"{remote_prefix}/{artifact_name}"
        cache_path = task.upload_cache.allocate_artifact_dir(
            task.row_index,
            artifact_name,
        )
        cache_path.mkdir(parents=True)
        (cache_path / "core.npz").write_bytes(b"core")
        (cache_path / "metadata.json").write_text('{"committed": true}\n')
        retained_cache_path = cache_path
        upload_future = task.upload_cache.stage(cache_path, remote_path)
        result = _successful_result(task, remote_path)
        result.summary_row["success"] = "false"
        result.summary_row["status"] = ""
        result.summary_row["finished_at"] = ""
        return _PreprocessManifestRowResult(
            row_index=result.row_index,
            artifact_reference=result.artifact_reference,
            summary_row=result.summary_row,
            grouped_split_record=result.grouped_split_record,
            preprocess_succeeded=True,
            upload_future=upload_future,
            cached_artifact_dir=cache_path,
            success_status="processed",
            entry_label="EMD-1001",
        )

    def fake_upload(local_path, remote_path, **kwargs):
        del local_path, kwargs
        if fail_upload:
            raise OSError("simulated R2 outage")
        uploaded.add(remote_path)

    def fake_identity(reference, **kwargs):
        del kwargs
        if str(reference) in uploaded:
            return {
                "kind": "s3",
                "content_length": 20,
                "etag": "committed",
                "version_id": "",
            }
        return None

    monkeypatch.setattr(
        worker,
        "_preprocess_manifest_row",
        fake_row_worker,
    )
    monkeypatch.setattr(
        preprocess_manifest,
        "validate_remote_configuration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        preprocess_manifest,
        "upload_preprocessed_artifact",
        fake_upload,
    )
    monkeypatch.setattr(
        preprocess_manifest,
        "preprocessed_artifact_identity",
        fake_identity,
    )

    with pytest.warns(RuntimeWarning, match="remote upload failed"):
        assert (
            preprocess_manifest.preprocess_manifest_csv(
                manifest_path,
                artifact_output_dir=output_dir,
                artifact_s3_prefix=remote_prefix,
                summary_csv=summary_path,
                preprocess_adapter=_dummy_preprocess,
            )
            == []
        )

    assert calls == [1]
    assert retained_cache_path is not None
    assert retained_cache_path.is_dir()
    assert Path(f"{summary_path}.checkpoint.jsonl").is_file()

    fail_upload = False
    artifact_references = preprocess_manifest.preprocess_manifest_csv(
        manifest_path,
        artifact_output_dir=output_dir,
        artifact_s3_prefix=remote_prefix,
        summary_csv=summary_path,
        preprocess_adapter=_dummy_preprocess,
    )

    assert calls == [1]
    assert artifact_references == ["s3://test-bucket/preprocessed/emd_1001_1abc"]
    assert not retained_cache_path.exists()
    assert not Path(f"{summary_path}.checkpoint.json").exists()
    assert not Path(f"{summary_path}.checkpoint.jsonl").exists()


def test_memory_admission_blocks_until_aggregate_capacity_is_available():
    admission = PreprocessMemoryAdmission(10)
    admission.acquire(7)
    waiter_started = threading.Event()
    waiter_acquired = threading.Event()

    def wait_for_capacity():
        waiter_started.set()
        admission.acquire(5)
        waiter_acquired.set()
        admission.release(5)

    waiter = threading.Thread(target=wait_for_capacity)
    waiter.start()
    assert waiter_started.wait(timeout=1)
    assert not waiter_acquired.wait(timeout=0.05)

    admission.release(7)
    assert waiter_acquired.wait(timeout=1)
    waiter.join(timeout=1)
    assert not waiter.is_alive()
    assert admission.reserved_bytes == 0


def test_memory_admission_wait_can_be_cancelled():
    cancel_event = threading.Event()
    admission = PreprocessMemoryAdmission(10, cancel_event=cancel_event)
    admission.acquire(7)
    failure = []

    def wait_for_capacity():
        try:
            admission.acquire(5)
        except BaseException as exc:
            failure.append(exc)

    waiter = threading.Thread(target=wait_for_capacity)
    waiter.start()
    cancel_event.set()
    waiter.join(timeout=1)
    admission.release(7)

    assert not waiter.is_alive()
    assert len(failure) == 1
    assert isinstance(failure[0], PreprocessMemoryAdmissionCancelled)


@pytest.mark.parametrize(
    ("estimated_peak_bytes", "expected_max_active"),
    [(6, 1), (None, 1), (4, 2)],
)
def test_manifest_workers_share_the_aggregate_memory_budget(
    tmp_path,
    monkeypatch,
    estimated_peak_bytes,
    expected_max_active,
):
    manifest_path = _write_manifest(tmp_path)
    active = 0
    max_active = 0
    invocation_count = 0
    lock = threading.Lock()
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    both_estimates_called = threading.Event()
    estimate_calls = 0

    def fake_preprocess(**kwargs):
        nonlocal active, invocation_count, max_active
        del kwargs
        with lock:
            active += 1
            invocation_count += 1
            max_active = max(max_active, active)
            current_invocation = invocation_count
        if current_invocation == 1:
            first_entered.set()
            assert release_first.wait(timeout=5)
        else:
            second_entered.set()
        with lock:
            active -= 1

    monkeypatch.setattr(
        preprocess_manifest,
        "preprocess_memory_guard_limit_bytes",
        lambda: 10,
    )

    def fake_estimate(*args, **kwargs):
        nonlocal estimate_calls
        del args, kwargs
        with lock:
            estimate_calls += 1
            if estimate_calls == 2:
                both_estimates_called.set()
        if estimated_peak_bytes is None:
            return None
        return SimpleNamespace(
            estimated_peak_bytes=estimated_peak_bytes,
            input_shape=(4, 4, 4),
            input_voxel_size=1.0,
            output_size=4,
        )

    monkeypatch.setattr(
        worker,
        "estimate_preprocess_memory_or_none",
        fake_estimate,
    )
    monkeypatch.setattr(
        worker,
        "load_preprocessed_artifact",
        lambda *args, **kwargs: {
            "density_grid": np.zeros((2, 2, 2), dtype=np.float32),
            "voxel_size": 1.0,
        },
    )
    monkeypatch.setattr(
        worker, "_preprocessed_artifact_is_reusable", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        "cryoflake.targets.schema.validate_component_extension_targets",
        lambda value: {
            "component": {
                "positions": np.zeros((0, 3), dtype=np.float32),
                "interface_mask": np.zeros(0, dtype=bool),
                "split_groups": {"component": (), "interface": ()},
            }
        },
    )

    artifact_references = []
    failures = []

    def run_manifest():
        try:
            artifact_references.extend(
                preprocess_manifest.preprocess_manifest_csv(
                    manifest_path,
                    artifact_output_dir=tmp_path / "artifacts",
                    preprocess_adapter=fake_preprocess,
                    num_workers=2,
                )
            )
        except BaseException as exc:
            failures.append(exc)

    runner = threading.Thread(target=run_manifest)
    runner.start()
    assert first_entered.wait(timeout=2)
    assert both_estimates_called.wait(timeout=2)
    try:
        if expected_max_active == 1:
            assert not second_entered.wait(timeout=0.2)
        else:
            assert second_entered.wait(timeout=2)
    finally:
        release_first.set()
    runner.join(timeout=5)

    assert not runner.is_alive()
    assert failures == []
    assert len(artifact_references) == 2
    assert max_active == expected_max_active


def test_gzip_memory_estimate_reads_only_the_mrc_header(tmp_path):
    map_path = tmp_path / "map.mrc"
    compressed_path = tmp_path / "map.mrc.gz"
    with mrcfile.new(map_path) as mrc:
        mrc.set_data(np.zeros((6, 8, 10), dtype=np.float32))
        mrc.voxel_size = 1.5
    with map_path.open("rb") as source:
        with gzip.open(compressed_path, "wb") as destination:
            shutil.copyfileobj(source, destination)

    estimate = estimate_preprocess_memory(
        compressed_path,
        target_voxel_size=1.0,
    )

    assert estimate.input_shape == (6, 8, 10)
    assert estimate.input_voxel_size == pytest.approx(1.5)
    assert estimate.cubic_size == 128
    assert estimate.output_size == 192
    assert estimate.estimated_peak_bytes > 0


def test_cgroup_limit_ignores_unrelated_same_kind_mount(tmp_path):
    cgroup_path = tmp_path / "cgroup"
    mountinfo_path = tmp_path / "mountinfo"
    correct_mount = tmp_path / "correct"
    unrelated_mount = tmp_path / "unrelated"
    (correct_mount / "job").mkdir(parents=True)
    unrelated_mount.mkdir()
    (correct_mount / "job" / "memory.max").write_text("500\n")
    (unrelated_mount / "memory.max").write_text("100\n")
    cgroup_path.write_text("0::/user.slice/job\n")
    mountinfo_path.write_text(
        f"36 29 0:32 /user.slice {correct_mount} rw - cgroup2 cgroup rw\n"
        f"37 29 0:33 /other {unrelated_mount} rw - cgroup2 cgroup rw\n"
    )

    assert _cgroup_memory_limit_bytes(cgroup_path, mountinfo_path) == 500


def test_cgroup_limit_includes_every_ancestor_mount(tmp_path):
    cgroup_path = tmp_path / "cgroup"
    mountinfo_path = tmp_path / "mountinfo"
    outer_mount = tmp_path / "outer"
    inner_mount = tmp_path / "inner"
    (outer_mount / "b" / "c").mkdir(parents=True)
    (inner_mount / "c").mkdir(parents=True)
    (outer_mount / "memory.max").write_text("100\n")
    (inner_mount / "c" / "memory.max").write_text("500\n")
    cgroup_path.write_text("0::/a/b/c\n")
    mountinfo_path.write_text(
        f"36 29 0:32 /a {outer_mount} rw - cgroup2 cgroup rw\n"
        f"37 29 0:33 /a/b {inner_mount} rw - cgroup2 cgroup rw\n"
    )

    assert _cgroup_memory_limit_bytes(cgroup_path, mountinfo_path) == 100


def test_callable_identity_includes_partial_and_closure_state():
    def configured_preprocess(*, mode):
        return mode

    mode_one = functools.partial(configured_preprocess, mode=1)
    mode_two = functools.partial(configured_preprocess, mode=2)

    assert callable_identity(mode_one) != callable_identity(mode_two)

    def make_closure(mode):
        return lambda: mode

    assert callable_identity(make_closure(1)) != callable_identity(make_closure(2))

    def make_configured_closure(configured_callable):
        return lambda: configured_callable()

    assert callable_identity(make_configured_closure(mode_one)) != callable_identity(
        make_configured_closure(mode_two)
    )


def test_callable_identity_includes_referenced_global_state():
    global _CALLABLE_TEST_MODE

    original_mode = _CALLABLE_TEST_MODE
    try:
        _CALLABLE_TEST_MODE = 1
        mode_one_identity = callable_identity(_global_mode_preprocess)
        _CALLABLE_TEST_MODE = 2
        mode_two_identity = callable_identity(_global_mode_preprocess)
    finally:
        _CALLABLE_TEST_MODE = original_mode

    assert mode_one_identity != mode_two_identity


def test_callable_identity_includes_transitive_and_partial_global_state():
    global _CALLABLE_TEST_MODE

    configured_partial = functools.partial(_global_mode_preprocess)
    original_mode = _CALLABLE_TEST_MODE
    try:
        _CALLABLE_TEST_MODE = 1
        partial_mode_one = callable_identity(configured_partial)
        transitive_mode_one = callable_identity(_calls_global_mode_preprocess)
        _CALLABLE_TEST_MODE = 2
        partial_mode_two = callable_identity(configured_partial)
        transitive_mode_two = callable_identity(_calls_global_mode_preprocess)
    finally:
        _CALLABLE_TEST_MODE = original_mode

    assert partial_mode_one != partial_mode_two
    assert transitive_mode_one != transitive_mode_two


def test_callable_identity_includes_bound_and_callable_object_behavior():
    class Adapter:
        def method(self):
            return _CALLABLE_TEST_MODE

        def __call__(self):
            return _CALLABLE_TEST_MODE

    def make_bound_method(mode):
        class ClosureAdapter:
            def method(self):
                return mode

        return ClosureAdapter().method

    global _CALLABLE_TEST_MODE

    adapter = Adapter()
    original_mode = _CALLABLE_TEST_MODE
    try:
        _CALLABLE_TEST_MODE = 1
        method_mode_one = callable_identity(adapter.method)
        callable_mode_one = callable_identity(adapter)
        _CALLABLE_TEST_MODE = 2
        method_mode_two = callable_identity(adapter.method)
        callable_mode_two = callable_identity(adapter)
    finally:
        _CALLABLE_TEST_MODE = original_mode

    assert method_mode_one != method_mode_two
    assert callable_mode_one != callable_mode_two
    assert callable_identity(make_bound_method(1)) != callable_identity(
        make_bound_method(2)
    )

    mapping = {"input": {"mode": 1}}
    bound_builtin = mapping.get
    mapping_mode_one = callable_identity(bound_builtin)
    mapping["input"]["mode"] = 2
    assert callable_identity(bound_builtin) != mapping_mode_one


def test_callable_identity_includes_exact_function_code():
    mode_one = lambda: 1  # noqa: E731 - exercise anonymous function code
    mode_two = lambda: 2  # noqa: E731 - exercise anonymous function code
    namespace_one = {}
    namespace_two = {}
    exec("def generated(): return 1", namespace_one)
    exec("def generated(): return 2", namespace_two)

    assert callable_identity(mode_one) != callable_identity(mode_two)
    assert callable_identity(namespace_one["generated"]) != callable_identity(
        namespace_two["generated"]
    )


def test_callable_identity_is_repeatable_for_unchanged_function():
    first_identity = callable_identity(_dummy_preprocess)

    assert callable_identity(_dummy_preprocess) == first_identity
    assert callable_identity(
        _dummy_preprocess,
        state_token="stable-v1",
    ) == callable_identity(
        _dummy_preprocess,
        state_token="stable-v1",
    )


def test_callable_identity_hashes_complete_numpy_array_state():
    values_one = np.zeros(10_000, dtype=np.float32)
    values_two = values_one.copy()
    values_two[500] = 1.0

    def make_closure(values):
        return lambda: values[500]

    assert repr(values_one) == repr(values_two)
    assert callable_identity(make_closure(values_one)) != callable_identity(
        make_closure(values_two)
    )

    masked_one = np.ma.array(
        [10, 20],
        mask=[False, True],
        fill_value=-1,
    )
    masked_two = np.ma.array(
        [10, 20],
        mask=[True, False],
        fill_value=-1,
    )
    assert callable_identity(make_closure(masked_one)) != callable_identity(
        make_closure(masked_two)
    )


def test_callable_identity_handles_empty_arrays_without_losing_shape_or_dtype():
    def make_closure(value):
        return lambda: value

    empty = np.zeros((0, 3), dtype=np.float32)
    identity = callable_identity(make_closure(empty))

    assert identity == callable_identity(make_closure(empty.copy()))
    assert identity != callable_identity(
        make_closure(np.zeros((2, 0), dtype=np.float32))
    )
    assert identity != callable_identity(
        make_closure(np.zeros((0, 3), dtype=np.float64))
    )


def test_callable_identity_normalizes_opaque_object_addresses():
    class OpaqueConfiguration:
        __slots__ = ()

    def make_closure(configuration):
        return lambda: configuration

    assert callable_identity(make_closure(OpaqueConfiguration())) == callable_identity(
        make_closure(OpaqueConfiguration())
    )


def test_callable_identity_preserves_semantic_hex_values():
    def make_closure(pattern):
        return lambda value: pattern.search(value)

    assert callable_identity(
        make_closure(re.compile("0xdeadbeef"))
    ) != callable_identity(make_closure(re.compile("0xcafebabe")))
    assert callable_identity(
        make_closure(re.compile(" at 0xdeadbeef>"))
    ) != callable_identity(make_closure(re.compile(" at 0xcafebabe>")))


def test_callable_identity_accepts_moduleless_dynamic_function():
    namespace = {}
    exec("def dynamic_preprocess(**kwargs): return kwargs", namespace)

    identity = callable_identity(namespace["dynamic_preprocess"])

    assert "dynamic_preprocess" in identity


def test_callable_identity_token_ignores_mutating_runtime_state():
    runtime_state = {"calls": 0}

    def stateful_preprocess():
        runtime_state["calls"] += 1

    first_identity = callable_identity(
        stateful_preprocess,
        state_token="implementation-v1",
    )
    stateful_preprocess()

    assert (
        callable_identity(
            stateful_preprocess,
            state_token="implementation-v1",
        )
        == first_identity
    )
    assert (
        callable_identity(
            stateful_preprocess,
            state_token="implementation-v2",
        )
        != first_identity
    )


def test_upload_cache_removes_resumed_artifact_directory(tmp_path):
    old_cache_dir = tmp_path / ".upload_cache_old"
    retained_path = old_cache_dir / "row_00000001" / "example"
    retained_path.mkdir(parents=True)
    (retained_path / "metadata.json").write_text("{}\n")

    upload_cache = BatchedArtifactUploadCache(
        tmp_path,
        lambda local_path, remote_uri: None,
        num_workers=1,
        batch_size=1,
    )
    result = upload_cache.stage(retained_path, "s3://bucket/example")
    upload_cache.close()

    assert result.result(timeout=1) is None
    assert not retained_path.exists()
    assert not old_cache_dir.exists()


def test_checkpoint_cache_rejects_same_size_same_mtime_replacement(tmp_path):
    cache_dir = tmp_path / ".upload_cache_old" / "row_00000001"
    retained_path = cache_dir / "example"
    retained_path.mkdir(parents=True)
    core_path = retained_path / "core.npz"
    metadata_path = retained_path / "metadata.json"
    core_path.write_bytes(b"core-one")
    metadata_path.write_bytes(b'{"value":1}')
    original_identity = _local_cache_identity(retained_path)
    original_times = {
        path: path.stat().st_mtime_ns for path in (core_path, metadata_path)
    }

    core_path.write_bytes(b"core-two")
    metadata_path.write_bytes(b'{"value":2}')
    for path, mtime_ns in original_times.items():
        os.utime(path, ns=(mtime_ns, mtime_ns))

    record = {
        "row_index": 1,
        "cached_artifact_dir": str(retained_path),
        "cache_identity": original_identity,
    }
    assert _local_cache_identity(retained_path) != original_identity
    assert not checkpoint_cache_matches(record, cache_parent=tmp_path)


def test_upload_cache_refuses_to_delete_unmanaged_directory(tmp_path):
    unmanaged_path = tmp_path / "important"
    unmanaged_path.mkdir()
    (unmanaged_path / "data").write_bytes(b"keep")
    upload_cache = BatchedArtifactUploadCache(
        tmp_path,
        lambda local_path, remote_uri: None,
        num_workers=1,
        batch_size=1,
    )

    result = upload_cache.stage(unmanaged_path, "s3://bucket/example")
    upload_cache.close()

    warning = result.result(timeout=1)
    assert "refusing to remove" in warning
    assert (unmanaged_path / "data").read_bytes() == b"keep"


def test_checkpoint_rejects_unmanaged_retained_cache_path(tmp_path):
    unmanaged_path = tmp_path / "important"
    unmanaged_path.mkdir()
    (unmanaged_path / "metadata.json").write_text("{}\n")
    record = {
        "row_index": 1,
        "cached_artifact_dir": str(unmanaged_path),
        "cache_identity": _local_cache_identity(unmanaged_path),
    }

    assert not checkpoint_cache_matches(
        record,
        cache_parent=tmp_path / "artifacts",
    )


def test_checkpoint_rejects_single_file_upload_cache(tmp_path):
    cache_dir = tmp_path / ".upload_cache_old" / "row_00000001"
    cache_dir.mkdir(parents=True)
    retained_file = cache_dir / "example.npz"
    retained_file.write_bytes(b"single-file cache")
    record = {
        "row_index": 1,
        "cached_artifact_dir": str(retained_file),
        "cache_identity": {"kind": "local", "size": retained_file.stat().st_size},
    }

    assert _local_cache_identity(retained_file) is None
    assert not checkpoint_cache_matches(record, cache_parent=tmp_path)
    assert retained_file.read_bytes() == b"single-file cache"
