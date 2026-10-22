from __future__ import annotations

import io
import pickle
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from cryoflake.sequence.sidecars import (
    attach_sequence_embeddings_to_preprocessed_artifact,
    load_sequence_embedding_file,
)
from cryoflake.sequence.validation import (
    validate_sequence_embedding_metadata,
    validate_sequence_embedding_payload,
)
from cryoflake.storage.store import (
    load_preprocessed_artifact,
    write_preprocessed_artifact,
)


def test_attach_sidecar_to_artifact_reference_preserves_core(tmp_path):
    root = tmp_path / "artifact"
    features = np.arange(4, dtype=np.float32)
    write_preprocessed_artifact(root, {"features": features})
    core_before = (root / "core.npz").read_bytes()
    sidecar = tmp_path / "embeddings.npz"
    sequence = np.arange(6, dtype=np.float64).reshape(2, 3)
    np.savez(sidecar, sequence=sequence, sequence_mask=[True, False])

    attach_sequence_embeddings_to_preprocessed_artifact(
        artifact_reference=root / "metadata.json", sequence_embedding_path=sidecar
    )

    payload = load_preprocessed_artifact(root)
    np.testing.assert_array_equal(payload["features"], features)
    np.testing.assert_array_equal(payload["sequence"], sequence)
    np.testing.assert_array_equal(payload["sequence_mask"], [True, False])
    assert payload["sequence"].dtype == np.float32
    assert (root / "core.npz").read_bytes() == core_before


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_supported_sidecar_formats_preserve_embeddings_and_mask(tmp_path, suffix):
    path = tmp_path / f"embedding{suffix}"
    sequence = np.arange(12, dtype=np.float64).reshape(3, 4)
    mask = np.array([1, 0, 1], dtype=np.uint8)
    payload = {
        "sequence": sequence,
        "sequence_mask": mask,
        "unified_seq": "AC|||D",
        "unified_seq_len": 3,
    }
    if suffix == ".npy":
        np.save(path, sequence)
    else:
        np.savez(path, **payload)

    loaded = load_sequence_embedding_file(path)
    np.testing.assert_array_equal(loaded["sequence"], sequence.astype(np.float32))
    assert loaded["sequence"].dtype == np.float32
    np.testing.assert_array_equal(
        loaded["sequence_mask"], np.ones(3, dtype=bool) if suffix == ".npy" else mask
    )
    assert loaded["sequence_mask"].dtype == bool


@pytest.mark.parametrize("field", ["residue_to_lm_embedding", "lm_embeddings", "arr_0"])
def test_npz_sidecars_require_canonical_sequence_field(tmp_path, field):
    path = tmp_path / "embedding.npz"
    np.savez(path, **{field: np.ones((3, 4))})

    with pytest.raises(ValueError, match="must contain sequence"):
        load_sequence_embedding_file(path)


@pytest.mark.parametrize(
    "payload", [SimpleNamespace(unified_seq="ACD", unified_seq_len=3), np.ones((3, 4))]
)
def test_sequence_metadata_requires_mapping(payload):
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_sequence_embedding_metadata(payload, 3)


@pytest.mark.parametrize("payload", [np.ones((3, 4)), SimpleNamespace(sequence=[[1]])])
def test_sequence_payload_requires_mapping(payload):
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_sequence_embedding_payload(payload)


@pytest.mark.parametrize(
    "key", ["residue_to_lm_embedding", "lm_embeddings", "arr_0", "protein"]
)
def test_sequence_payload_requires_canonical_top_level_field(key):
    sequence = np.ones((2, 3), dtype=np.float32)
    payload = {key: {"sequence": sequence} if key == "protein" else sequence}
    with pytest.raises(ValueError, match="must contain sequence"):
        validate_sequence_embedding_payload(payload)


def test_sequence_payload_normalizes_array_and_optional_mask():
    sequence = np.arange(6, dtype=np.float64).reshape(2, 3)
    payload = {"sequence": sequence, "unified_seq": "A|||C", "unified_seq_len": 2}

    actual = validate_sequence_embedding_payload(payload)
    assert set(actual) == {"sequence", "sequence_mask"}
    assert actual["sequence"].dtype == np.float32
    assert actual["sequence_mask"].dtype == bool
    np.testing.assert_array_equal(actual["sequence"], sequence)
    np.testing.assert_array_equal(actual["sequence_mask"], [True, True])
    assert "sequence_mask" not in payload

    payload["sequence_mask"] = [1, 0]
    np.testing.assert_array_equal(
        validate_sequence_embedding_payload(payload)["sequence_mask"], [True, False]
    )


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"unified_seq": "A"}, "must match unified_seq"),
        ({"unified_seq_len": 3}, "must match unified_seq_len"),
        ({"sequence_mask": [True]}, "length must match"),
        ({"sequence_mask": [False, False]}, "at least one true"),
    ],
)
def test_sequence_payload_validates_metadata_and_mask(metadata, message):
    with pytest.raises(ValueError, match=message):
        validate_sequence_embedding_payload({"sequence": np.ones((2, 3)), **metadata})


@pytest.mark.parametrize("suffix", [".prot", ".pkl", ".pickle", ".json", ""])
def test_unsupported_sidecars_are_rejected_without_reading(
    tmp_path, monkeypatch, suffix
):
    path = tmp_path / f"embedding{suffix}"

    def unexpected_read(*args, **kwargs):
        pytest.fail("Unsupported sidecars must be rejected before reading")

    monkeypatch.setattr(type(path), "open", unexpected_read)
    monkeypatch.setattr(pickle, "load", unexpected_read)
    monkeypatch.setattr(pickle, "loads", unexpected_read)
    with pytest.raises(ValueError, match="Expected .npy or .npz"):
        load_sequence_embedding_file(path)


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_object_array_sidecars_fail_without_unpickling(tmp_path, monkeypatch, suffix):
    path = tmp_path / f"embedding{suffix}"
    array = np.array([[object()]], dtype=object)
    if suffix == ".npy":
        np.save(path, array)
    else:
        np.savez(path, sequence=np.ones((3, 4)), unrecognized_metadata=array)

    def unexpected_unpickle(*args, **kwargs):
        pytest.fail("Object arrays must be rejected before unpickling")

    monkeypatch.setattr(pickle, "load", unexpected_unpickle)
    monkeypatch.setattr(pickle, "loads", unexpected_unpickle)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_sequence_embedding_file(path)


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"unified_seq": "AC"}, "unified_seq"),
        ({"unified_seq_len": 4}, "unified_seq_len"),
        ({"unified_seq_len": True}, "scalar integer"),
        ({"sequence_mask": [0, 0, 0]}, "at least one true"),
        ({"sequence_mask": [1, 2, 0]}, "0/1"),
        ({"sequence_mask": [1, 0]}, "length must match"),
    ],
)
def test_sidecars_reject_inconsistent_metadata(tmp_path, metadata, message):
    path = tmp_path / "embedding.npz"
    np.savez(path, sequence=np.ones((3, 4)), **metadata)
    with pytest.raises(ValueError, match=message):
        load_sequence_embedding_file(path)


def test_npz_sidecar_streams_numeric_extras_without_materializing(
    tmp_path, monkeypatch
):
    path = tmp_path / "embedding.npz"
    sequence = np.arange(12, dtype=np.float64).reshape(3, 4)
    np.savez_compressed(
        path,
        sequence=sequence,
        sequence_mask=[True, False, True],
        unified_seq="AC|||D",
        unified_seq_len=3,
        unrelated_array=np.arange(512 * 1024, dtype=np.float64),
    )
    with np.load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    original_getitem = archive_type.__getitem__
    original_read = zipfile.ZipExtFile.read
    materialized_keys = []
    ignored_reads = []

    def checked_getitem(archive, key):
        assert key != "unrelated_array"
        materialized_keys.append(key)
        return original_getitem(archive, key)

    def checked_read(member, size=-1):
        if member.name == "unrelated_array.npy":
            assert 0 <= size <= 256 * 1024
            ignored_reads.append(size)
        return original_read(member, size)

    monkeypatch.setattr(archive_type, "__getitem__", checked_getitem)
    monkeypatch.setattr(zipfile.ZipExtFile, "read", checked_read)

    actual = load_sequence_embedding_file(path)

    np.testing.assert_array_equal(actual["sequence"], sequence)
    np.testing.assert_array_equal(actual["sequence_mask"], [True, False, True])
    assert actual["sequence"].dtype == np.float32
    assert materialized_keys == [
        "sequence",
        "sequence_mask",
        "unified_seq",
        "unified_seq_len",
    ]
    assert len(ignored_reads) > 16


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize(
    "extra",
    [
        np.array([np.nan, np.inf]),
        np.asfortranarray(np.arange(12).reshape(3, 4)),
        np.zeros(2, dtype=[("value", "f8"), ("index", "i4")]),
        np.zeros((0, 3)),
        np.ndarray((2, 3), dtype="S0"),
    ],
    ids=["nonfinite", "fortran", "structured", "empty", "zero-width"],
)
def test_npz_sidecars_preserve_supported_ignored_array_formats(
    tmp_path, version, extra
):
    path = tmp_path / "embedding.npz"
    np.savez(path, sequence=np.ones((2, 3)))
    serialized = io.BytesIO()
    np.lib.format.write_array(serialized, extra, version=version, allow_pickle=False)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("extra.npy", serialized.getvalue())

    actual = load_sequence_embedding_file(path)

    np.testing.assert_array_equal(actual["sequence"], np.ones((2, 3)))
    np.testing.assert_array_equal(actual["sequence_mask"], [True, True])


@pytest.mark.parametrize("object_field", [False, True])
def test_npz_sidecars_validate_utf8_structured_extras_without_unpickling(
    tmp_path, monkeypatch, object_field
):
    path = tmp_path / "embedding.npz"
    np.savez(path, sequence=np.ones((2, 3)))
    extra = np.zeros(1, dtype=[("原子", object if object_field else "f8")])
    serialized = io.BytesIO()
    np.lib.format.write_array(serialized, extra, version=(3, 0))
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("extra.npy", serialized.getvalue())

    def unexpected_unpickle(*args, **kwargs):
        pytest.fail("Ignored object arrays must be rejected before unpickling")

    monkeypatch.setattr(pickle, "load", unexpected_unpickle)
    monkeypatch.setattr(pickle, "loads", unexpected_unpickle)
    if object_field:
        with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
            load_sequence_embedding_file(path)
    else:
        np.testing.assert_array_equal(
            load_sequence_embedding_file(path)["sequence"], np.ones((2, 3))
        )


@pytest.mark.parametrize("member_name", ["notes.txt", "notes.npy"])
def test_npz_sidecars_preserve_raw_non_npy_extras(tmp_path, member_name):
    path = tmp_path / "embedding.npz"
    np.savez(path, sequence=np.ones((2, 3)))
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, b"extra information" * 100_000)

    actual = load_sequence_embedding_file(path)

    np.testing.assert_array_equal(actual["sequence"], np.ones((2, 3)))


@pytest.mark.parametrize("invalid_kind", ["header", "version", "shape", "truncated"])
def test_npz_sidecars_reject_malformed_ignored_arrays(tmp_path, invalid_kind):
    path = tmp_path / "embedding.npz"
    np.savez(path, sequence=np.ones((2, 3)))
    serialized = io.BytesIO()
    if invalid_kind == "shape":
        np.lib.format.write_array_header_1_0(
            serialized,
            {"descr": "<f4", "shape": (-1,), "fortran_order": False},
        )
        content = serialized.getvalue()
    else:
        np.lib.format.write_array(serialized, np.arange(1024), allow_pickle=False)
        content = serialized.getvalue()
        if invalid_kind == "header":
            content = content[:10] + b"invalid header"
        elif invalid_kind == "version":
            content = content[:6] + b"\x04\x00" + content[8:]
        else:
            content = content[:-1]
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("extra.npy", content)

    with pytest.raises(ValueError):
        load_sequence_embedding_file(path)


@pytest.mark.parametrize("member_name", ["extra.npy", "notes.txt"])
def test_npz_sidecars_reject_ignored_member_crc_corruption(tmp_path, member_name):
    path = tmp_path / "embedding.npz"
    np.savez(path, sequence=np.ones((2, 3)))
    extra = io.BytesIO()
    np.lib.format.write_array(extra, np.arange(128 * 1024), allow_pickle=False)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            member_name,
            extra.getvalue()
            if member_name.endswith(".npy")
            else b"raw bytes" * 100_000,
        )
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(member_name)
    content = bytearray(path.read_bytes())
    filename_size = int.from_bytes(
        content[member.header_offset + 26 : member.header_offset + 28], "little"
    )
    extra_size = int.from_bytes(
        content[member.header_offset + 28 : member.header_offset + 30], "little"
    )
    payload_offset = member.header_offset + 30 + filename_size + extra_size
    content[payload_offset + member.file_size - 1] ^= 1
    path.write_bytes(content)

    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        load_sequence_embedding_file(path)


_NPZ_MEMBER_COLLISIONS = [
    "ignored_alias",
    "supported_alias",
    "suffix_alias",
    "ignored_duplicate",
    "supported_duplicate",
]


def _npy_bytes(array):
    serialized = io.BytesIO()
    np.lib.format.write_array(serialized, array)
    return serialized.getvalue()


def _write_shadowed_npz(path, collision, shadowed_content):
    # Synthetic embeddings isolate ZIP name precedence and physical integrity.
    sequence = np.arange(12, dtype=np.float64).reshape(3, 4)
    names = {
        "ignored_alias": ("extra.npy", "extra"),
        "supported_alias": ("sequence.npy", "sequence"),
        "suffix_alias": ("extra.npy.npy", "extra.npy"),
        "ignored_duplicate": ("extra.npy", "extra.npy"),
        "supported_duplicate": ("sequence.npy", "sequence.npy"),
    }
    shadowed_name, preferred_name = names[collision]
    preferred_content = (
        _npy_bytes(sequence)
        if collision.startswith("supported_")
        else b"unrelated raw metadata"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        if not collision.startswith("supported_"):
            archive.writestr("sequence.npy", _npy_bytes(sequence))
        archive.writestr(shadowed_name, shadowed_content)
        shadowed_member = archive.infolist()[-1]
        if shadowed_name == preferred_name:
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(preferred_name, preferred_content)
        else:
            archive.writestr(preferred_name, preferred_content)
    return sequence, shadowed_member


@pytest.mark.parametrize("collision", _NPZ_MEMBER_COLLISIONS)
def test_npz_sidecars_preserve_preferred_member_with_name_collisions(
    tmp_path, monkeypatch, collision
):
    path = tmp_path / "embedding.npz"
    expected, _ = _write_shadowed_npz(
        path, collision, _npy_bytes(np.full((3, 4), 500.0))
    )
    with np.load(path, allow_pickle=False) as archive:
        # NpzFile prefers exact names and the last physical duplicate.
        np.testing.assert_array_equal(archive["sequence"], expected)
        archive_type = type(archive)
    original_getitem = archive_type.__getitem__
    materialized_keys = []

    def checked_getitem(archive, key):
        materialized_keys.append(key)
        return original_getitem(archive, key)

    monkeypatch.setattr(archive_type, "__getitem__", checked_getitem)

    actual = load_sequence_embedding_file(path)

    np.testing.assert_array_equal(actual["sequence"], expected)
    assert actual["sequence"].dtype == np.float32
    np.testing.assert_array_equal(actual["sequence_mask"], [True, True, True])
    assert materialized_keys == ["sequence"]


def test_npz_sidecar_prefers_exact_name_before_later_npy_alias(tmp_path):
    # Synthetic embeddings distinguish exact-name preference from ZIP order.
    path = tmp_path / "embedding.npz"
    sequence = np.arange(12, dtype=np.float64).reshape(3, 4)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sequence", _npy_bytes(sequence))
        archive.writestr("sequence.npy", _npy_bytes(np.full((3, 4), 500.0)))

    actual = load_sequence_embedding_file(path)

    np.testing.assert_array_equal(actual["sequence"], sequence)


@pytest.mark.parametrize("collision", _NPZ_MEMBER_COLLISIONS)
@pytest.mark.parametrize("invalid_kind", ["object", "header"])
def test_npz_sidecars_reject_shadowed_invalid_npy_without_unpickling(
    tmp_path, monkeypatch, collision, invalid_kind
):
    path = tmp_path / "embedding.npz"
    shadowed_content = (
        _npy_bytes(np.array([object()], dtype=object))
        if invalid_kind == "object"
        else np.lib.format.magic(1, 0) + b"\xff\xffinvalid header"
    )
    _write_shadowed_npz(path, collision, shadowed_content)

    def unexpected_unpickle(*args, **kwargs):
        pytest.fail("Shadowed object arrays must be rejected before unpickling")

    monkeypatch.setattr(pickle, "load", unexpected_unpickle)
    monkeypatch.setattr(pickle, "loads", unexpected_unpickle)
    message = "Object arrays cannot be loaded" if invalid_kind == "object" else None
    with pytest.raises(ValueError, match=message):
        load_sequence_embedding_file(path)


@pytest.mark.parametrize("collision", _NPZ_MEMBER_COLLISIONS)
def test_npz_sidecars_reject_shadowed_physical_member_crc_corruption(
    tmp_path, collision
):
    path = tmp_path / "embedding.npz"
    _, member = _write_shadowed_npz(
        path, collision, _npy_bytes(np.arange(32 * 1024, dtype=np.float64))
    )
    content = bytearray(path.read_bytes())
    filename_size = int.from_bytes(
        content[member.header_offset + 26 : member.header_offset + 28], "little"
    )
    extra_size = int.from_bytes(
        content[member.header_offset + 28 : member.header_offset + 30], "little"
    )
    payload_offset = member.header_offset + 30 + filename_size + extra_size
    content[payload_offset + member.file_size - 1] ^= 1
    path.write_bytes(content)

    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        load_sequence_embedding_file(path)
