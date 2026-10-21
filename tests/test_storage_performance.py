"""Deterministic checks for bounded, single-pass artifact reconstruction."""

from __future__ import annotations

import io
import json
import struct
import zipfile

import numpy as np
import pytest

from cryoflake.common import sha256_file
from cryoflake.storage import codec, reader
from cryoflake.storage.store import (
    load_preprocessed_artifact,
    write_preprocessed_artifact,
)
from cryoflake.storage.validation import _read_array_payload


@pytest.mark.parametrize("root_keys", [None, ["array"]])
@pytest.mark.parametrize("include_sequence_embeddings", [True, False])
def test_facade_parses_and_validates_metadata_once(
    tmp_path, monkeypatch, root_keys, include_sequence_embeddings
):
    root = tmp_path / "artifact"
    expected = np.arange(4, dtype=np.float32)
    write_preprocessed_artifact(root, {"array": expected})
    original_loads = codec.json.loads
    calls = 0

    def counted_loads(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_loads(*args, **kwargs)

    monkeypatch.setattr(codec.json, "loads", counted_loads)
    loaded = load_preprocessed_artifact(
        root,
        root_keys=root_keys,
        include_sequence_embeddings=include_sequence_embeddings,
    )
    np.testing.assert_array_equal(loaded["array"], expected)
    assert calls == 1


@pytest.mark.parametrize(
    "dtype",
    ["<f4", ">f8", ">i2", "c16", "U3", "S3", "M8[ns]", "m8[us]", "V3"],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_single_pass_reader_preserves_dtype_layout_and_bytes(tmp_path, dtype, order):
    expected = np.array(np.arange(12).reshape(3, 4), dtype=dtype, order=order)
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": expected})

    actual = load_preprocessed_artifact(root)["array"]

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.flags.f_contiguous == expected.flags.f_contiguous
    assert actual.flags.writeable
    assert actual.tobytes(order="A") == expected.tobytes(order="A")


@pytest.mark.parametrize("dtype", ["f4", "S0", "U0", "V0"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (2, 3)])
def test_single_pass_reader_preserves_empty_arrays_and_zero_width_dtypes(
    tmp_path, dtype, shape
):
    expected = np.ndarray(shape, dtype=dtype)
    expected[...] = np.zeros((), dtype=dtype)
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": expected})

    actual = load_preprocessed_artifact(root)["array"]

    assert actual.dtype == expected.dtype
    assert actual.shape == shape
    assert actual.tobytes() == expected.tobytes()


def test_array_payload_reads_are_bounded_and_handle_partial_readinto():
    expected = np.arange(400_000, dtype=np.float64)

    class PartialReads(io.BytesIO):
        def readinto(self, target):
            assert len(target) <= 256 * 1024
            return super().readinto(target[: min(len(target), 13_127)])

    actual = _read_array_payload(
        PartialReads(expected.tobytes()), expected.shape, False, expected.dtype
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mutable", [False, True])
def test_flattening_shares_immutable_bytes_and_snapshots_mutable_input(mutable):
    value = b"large immutable buffers do not need a duplicate"
    if mutable:
        value = bytearray(value)
    _, stores = codec.flatten_payload({"data": value})
    stored = stores["core"]["data"]
    if mutable:
        value[0] = 0
        assert stored.tobytes().startswith(b"large")
        assert stored.flags.owndata
    else:
        assert stored.base is value
        assert not stored.flags.owndata


def test_immutable_bytes_round_trip(tmp_path):
    value = bytes(range(256)) * 4096
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"data": value})

    assert load_preprocessed_artifact(root)["data"] == value


def test_single_pass_reader_checks_zip_crc(tmp_path):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(4, dtype=np.float32)})
    path = root / "core.npz"
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("array.npy")
    contents = bytearray(path.read_bytes())
    name_length, extra_length = struct.unpack_from(
        "<HH", contents, info.header_offset + 26
    )
    payload_offset = info.header_offset + 30 + name_length + extra_length
    contents[payload_offset + info.file_size - 1] ^= 1
    path.write_bytes(contents)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["store_hashes"]["core"] = sha256_file(path)
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        load_preprocessed_artifact(root)


@pytest.mark.parametrize("root_keys", [None, ["branch"], ["label"], []])
@pytest.mark.parametrize("include_sequence_embeddings", [True, False])
def test_array_descriptors_are_collected_only_for_metadata_validation_and_reading(
    tmp_path, monkeypatch, root_keys, include_sequence_embeddings
):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(
        root,
        {
            "branch": {"values": np.arange(4), "nested": [np.arange(3), b"bytes"]},
            "sequence": np.ones((2, 3)),
            "label": "portable",
        },
    )
    original_collect = codec._collect_array_specs
    visits = {}

    def count_descriptors(node, specs):
        if isinstance(node, dict) and node.get("kind") in {"array", "bytes"}:
            key = (node["store"], node["key"])
            visits[key] = visits.get(key, 0) + 1
        return original_collect(node, specs)

    monkeypatch.setattr(codec, "_collect_array_specs", count_descriptors)
    monkeypatch.setattr(reader, "_collect_array_specs", count_descriptors)
    loaded = load_preprocessed_artifact(
        root,
        root_keys=root_keys,
        include_sequence_embeddings=include_sequence_embeddings,
    )
    metadata_only = root_keys in (["label"], [])
    if metadata_only:
        assert loaded == ({"label": "portable"} if root_keys else {})
    else:
        np.testing.assert_array_equal(loaded["branch"]["values"], np.arange(4))
        np.testing.assert_array_equal(loaded["branch"]["nested"][0], np.arange(3))
        assert loaded["branch"]["nested"][1] == b"bytes"
    expected_visits = 1 if metadata_only else 2
    assert visits == {
        ("core", "branch/values"): expected_visits,
        ("core", "branch/nested/0"): expected_visits,
        ("core", "branch/nested/1"): expected_visits,
        ("sequence", "sequence"): expected_visits,
    }
