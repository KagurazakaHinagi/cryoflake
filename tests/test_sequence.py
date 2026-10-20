from __future__ import annotations

import pickle
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
