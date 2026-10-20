"""Regression coverage for malformed and unusual portable artifact contents."""

from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np
import pytest

from cryoflake.common import sha256_file
from cryoflake.storage import reader, writer
from cryoflake.storage.codec import STORE_GENERATION_KEY, read_metadata, required_stores
from cryoflake.storage.reader import read_npz_json
from cryoflake.storage.store import (
    load_preprocessed_artifact,
    write_preprocessed_artifact,
)
from cryoflake.storage.validation import validate_npz_json_store_files


@pytest.mark.parametrize("key", ["file", "allow_pickle"])
def test_array_keys_do_not_collide_with_numpy_writer_arguments(tmp_path, key):
    root = tmp_path / "artifact"
    expected = np.arange(6, dtype=np.float32)
    write_preprocessed_artifact(root, {key: expected})

    np.testing.assert_array_equal(load_preprocessed_artifact(root)[key], expected)
    validate_npz_json_store_files(root / "metadata.json", root / "core.npz")


@pytest.mark.parametrize("root_keys", [None, ["array.npy"]])
def test_array_keys_with_npy_suffixes_are_unambiguous(tmp_path, root_keys):
    root = tmp_path / "artifact"
    first = np.arange(4, dtype=np.float32)
    second = first + 100
    write_preprocessed_artifact(root, {"array": first, "array.npy": second})

    actual = load_preprocessed_artifact(root, root_keys=root_keys)

    np.testing.assert_array_equal(actual["array.npy"], second)
    if root_keys is None:
        np.testing.assert_array_equal(actual["array"], first)


@pytest.mark.parametrize("validator", ["read", "stream"])
def test_array_members_reject_trailing_serialized_data(tmp_path, validator):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(4, dtype=np.float32)})
    store = root / "core.npz"
    with zipfile.ZipFile(store) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["array.npy"] += b"unexpected payload bytes"
    with zipfile.ZipFile(store, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["store_hashes"]["core"] = sha256_file(store)
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="serialized payload size"):
        if validator == "read":
            load_preprocessed_artifact(root)
        else:
            validate_npz_json_store_files(metadata_path, store)


@pytest.mark.parametrize("validator", ["read", "stream"])
def test_array_members_require_the_npy_extension(tmp_path, validator):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(4, dtype=np.float32)})
    store = root / "core.npz"
    with zipfile.ZipFile(store) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["array"] = members.pop("array.npy")
    with zipfile.ZipFile(store, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["store_hashes"]["core"] = sha256_file(store)
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="missing arrays"):
        if validator == "read":
            load_preprocessed_artifact(root)
        else:
            validate_npz_json_store_files(metadata_path, store)


@pytest.mark.parametrize("validator", ["metadata", "read", "stream"])
def test_sequence_store_declaration_must_match_array_tree(tmp_path, validator):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2)})
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["files"]["sequence"] = "sequence.npz"
    metadata["store_generations"]["sequence"] = "0" * 32
    metadata["store_hashes"]["sequence"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="sequence store declaration"):
        if validator == "metadata":
            read_metadata(metadata_path)
        elif validator == "read":
            read_npz_json(metadata_path, root / "core.npz")
        else:
            validate_npz_json_store_files(metadata_path, root / "core.npz")


@pytest.mark.parametrize("root_keys", ["array", [1], ["array", None], 1])
@pytest.mark.parametrize("include_sequence_embeddings", [True, False])
def test_selected_root_keys_require_a_collection_of_strings(
    tmp_path, root_keys, include_sequence_embeddings
):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2)})

    with pytest.raises(ValueError, match="root_keys"):
        load_preprocessed_artifact(
            root,
            root_keys=root_keys,
            include_sequence_embeddings=include_sequence_embeddings,
        )


@pytest.mark.parametrize("root_keys", ["array", [1], 1])
def test_lowlevel_selection_rejects_invalid_root_keys(tmp_path, root_keys):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2)})
    with pytest.raises(ValueError, match="root_keys"):
        read_npz_json(root / "metadata.json", root / "core.npz", root_keys=root_keys)
    with pytest.raises(ValueError, match="root_keys"):
        required_stores(root / "metadata.json", root_keys=root_keys)


def test_absent_string_root_keys_are_ignored(tmp_path):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2)})
    assert load_preprocessed_artifact(root, root_keys=["missing"]) == {}


@pytest.mark.parametrize(
    "node",
    [
        {"kind": []},
        {"kind": "array", "store": [], "key": "array", "dtype": "<i8", "shape": [2]},
        {"kind": "sequence", "sequence_type": [], "items": []},
    ],
)
def test_unhashable_metadata_values_raise_validation_errors(tmp_path, node):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2)})
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["tree"]["items"]["array"] = node

    with pytest.raises(ValueError):
        read_metadata(json.dumps(metadata).encode())


def test_selective_loading_does_not_decode_unselected_arrays(tmp_path, monkeypatch):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2), "other": np.arange(3)})
    original_open = zipfile.ZipFile.open
    opened = []

    def checked_open(archive, member, *args, **kwargs):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        opened.append(name)
        assert name != "other.npy"
        return original_open(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", checked_open)
    actual = load_preprocessed_artifact(root, root_keys=["array"])
    assert set(actual) == {"array"}
    np.testing.assert_array_equal(actual["array"], np.arange(2))
    assert opened == [f"{STORE_GENERATION_KEY}.npy", "array.npy"]


@pytest.mark.parametrize("validator", ["read", "stream"])
def test_generation_headers_are_checked_before_decoding(
    tmp_path, monkeypatch, validator
):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2)})
    store = root / "core.npz"
    with zipfile.ZipFile(store) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    # A tiny member with a huge declared shape must never reach np.load,
    # which would allocate its declared payload before reporting truncation.
    generation = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        generation,
        {"descr": "<U32", "fortran_order": False, "shape": (10**12,)},
    )
    members[f"{STORE_GENERATION_KEY}.npy"] = generation.getvalue()
    with zipfile.ZipFile(store, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["store_hashes"]["core"] = sha256_file(store)
    metadata_path.write_text(json.dumps(metadata))

    def unexpected_load(*args, **kwargs):
        pytest.fail("Unvalidated generation headers must not reach np.load")

    monkeypatch.setattr(np, "load", unexpected_load)
    with pytest.raises(ValueError, match="invalid generation"):
        if validator == "read":
            load_preprocessed_artifact(root)
        else:
            validate_npz_json_store_files(metadata_path, store)


@pytest.mark.parametrize("failure_index", [0, 1, 2])
@pytest.mark.parametrize("fail_after_replace", [True, False])
def test_failed_local_commit_restores_previous_artifact(
    tmp_path, monkeypatch, failure_index, fail_after_replace
):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(
        root, {"array": np.arange(2), "sequence": np.ones((2, 3))}
    )
    original = {path.name: path.read_bytes() for path in root.iterdir()}
    original_commit = writer._commit_staged_file
    calls = 0

    def interrupted_commit(staged, target):
        nonlocal calls
        call_index = calls
        calls += 1
        if call_index == failure_index:
            if fail_after_replace:
                original_commit(staged, target)
            raise OSError("simulated commit failure")
        original_commit(staged, target)

    monkeypatch.setattr(writer, "_commit_staged_file", interrupted_commit)
    with pytest.raises(OSError, match="simulated commit failure"):
        write_preprocessed_artifact(
            root, {"array": np.arange(3), "sequence": np.ones((3, 4))}
        )

    assert {path.name: path.read_bytes() for path in root.iterdir()} == original
    np.testing.assert_array_equal(
        load_preprocessed_artifact(root)["array"], np.arange(2)
    )


def test_failed_commit_restores_dangling_member_symlinks(tmp_path, monkeypatch):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2)})
    original = {path.name: path.read_bytes() for path in root.iterdir()}
    sequence_path = root / "sequence.npz"
    sequence_path.symlink_to("not-present.npz")
    original_commit = writer._commit_staged_file

    def interrupted_commit(staged, target):
        if target.name == "metadata.json":
            raise OSError("simulated commit failure")
        original_commit(staged, target)

    monkeypatch.setattr(writer, "_commit_staged_file", interrupted_commit)
    with pytest.raises(OSError, match="simulated commit failure"):
        write_preprocessed_artifact(
            root, {"array": np.arange(3), "sequence": np.ones((2, 3))}
        )

    assert sequence_path.is_symlink()
    assert os.readlink(sequence_path) == "not-present.npz"
    assert {
        path.name: path.read_bytes()
        for path in root.iterdir()
        if path.name != "sequence.npz"
    } == original


def test_read_uses_the_same_store_snapshot_for_validation_and_decoding(
    tmp_path, monkeypatch
):
    root = tmp_path / "artifact"
    replacement = tmp_path / "replacement"
    expected = np.arange(4, dtype=np.float32)
    write_preprocessed_artifact(root, {"array": expected})
    write_preprocessed_artifact(replacement, {"array": expected + 100})
    original_validate = reader._validate_store_archive

    def validate_then_replace(*args, **kwargs):
        os.replace(replacement / "core.npz", root / "core.npz")
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(reader, "_validate_store_archive", validate_then_replace)
    actual = load_preprocessed_artifact(root)

    np.testing.assert_array_equal(actual["array"], expected)
