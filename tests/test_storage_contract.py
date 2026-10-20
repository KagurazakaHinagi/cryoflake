"""Artifact readers and writers enforce only canonical cryoFlake identifiers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cryoflake.common import sha256_file
from cryoflake.storage.codec import FORMAT_NAME, STORE_GENERATION_KEY
from cryoflake.storage.store import (
    load_preprocessed_artifact,
    replace_preprocessed_artifact_sequence,
    write_preprocessed_artifact,
)
from cryoflake.storage.validation import validate_npz_json_store_files
from cryoflake.storage.writer import write_npz_json


def _validate(root, validator):
    if validator == "read":
        return load_preprocessed_artifact(root)
    return validate_npz_json_store_files(root / "metadata.json", root / "core.npz")


def test_artifacts_use_canonical_wire_identifiers(tmp_path):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(4)})
    metadata = json.loads((root / "metadata.json").read_text())
    assert FORMAT_NAME == metadata["format"] == "cryoflake-preprocessed"
    assert STORE_GENERATION_KEY == "__cryoflake_store_generation__"
    with np.load(root / "core.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"array", STORE_GENERATION_KEY}


def test_direct_writer_uses_the_single_split_layout(tmp_path):
    root = tmp_path / "artifact"
    sequence = np.ones((2, 3), dtype=np.float32)
    assert (
        write_npz_json(artifact_dir=root, artifact_mapping={"sequence": sequence})
        is None
    )
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["split_sequence"] is True
    with np.load(root / "core.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {STORE_GENERATION_KEY}
    with np.load(root / "sequence.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"sequence", STORE_GENERATION_KEY}
    np.testing.assert_array_equal(
        load_preprocessed_artifact(root)["sequence"], sequence
    )


def test_writer_does_not_offer_an_unsplit_format(tmp_path):
    with pytest.raises(TypeError, match="split_sequence"):
        write_npz_json(tmp_path / "artifact", {}, split_sequence=False)


@pytest.mark.parametrize("validator", ["read", "stream"])
@pytest.mark.parametrize("split_sequence", [False, None, 0, 1, "true"])
def test_metadata_requires_the_canonical_split_layout(
    tmp_path, validator, split_sequence
):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2)})
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["split_sequence"] = split_sequence
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="require split_sequence=true"):
        _validate(root, validator)


def test_sequence_replacement_preserves_canonical_core_and_identifiers(tmp_path):
    root = tmp_path / "artifact"
    payload = {
        "density_grid": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        "sequence": np.ones((2, 3), dtype=np.float32),
        "sequence_mask": np.asarray([True, False]),
    }
    write_preprocessed_artifact(root, payload)
    core = (root / "core.npz").read_bytes()
    replacement = np.full((2, 5), 7, dtype=np.float32)
    replace_preprocessed_artifact_sequence(root, {"sequence": replacement})
    assert (root / "core.npz").read_bytes() == core
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["format"] == FORMAT_NAME
    with np.load(root / "sequence.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"sequence", "sequence_mask", STORE_GENERATION_KEY}
    loaded = load_preprocessed_artifact(root)
    np.testing.assert_array_equal(loaded["sequence"], replacement)
    np.testing.assert_array_equal(loaded["density_grid"], payload["density_grid"])
    validate_npz_json_store_files(
        root / "metadata.json", root / "core.npz", root / "sequence.npz"
    )


@pytest.mark.parametrize("validator", ["read", "stream"])
@pytest.mark.parametrize("replace_marker", [True, False])
def test_noncanonical_archive_markers_fail_closed(tmp_path, validator, replace_marker):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2)})
    path = root / "core.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["__unsupported_store_generation__"] = arrays[STORE_GENERATION_KEY]
    if replace_marker:
        del arrays[STORE_GENERATION_KEY]
    np.savez_compressed(path, **arrays)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["store_hashes"]["core"] = sha256_file(path)
    metadata_path.write_text(json.dumps(metadata))
    error = "missing its generation" if replace_marker else "unexpected arrays"
    with pytest.raises(ValueError, match=error):
        _validate(root, validator)


@pytest.mark.parametrize("validator", ["read", "stream"])
def test_generation_marker_cannot_be_declared_as_payload(tmp_path, validator):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2)})
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["tree"]["items"][STORE_GENERATION_KEY] = {
        "kind": "array",
        "store": "core",
        "key": STORE_GENERATION_KEY,
        "shape": [],
        "dtype": "<U32",
    }
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="reserved store generation key"):
        _validate(root, validator)


@pytest.mark.parametrize("validator", ["read", "stream"])
@pytest.mark.parametrize("format_name", [[], {}, None, 1, "unrecognized-format"])
def test_unsupported_metadata_formats_fail_closed(tmp_path, validator, format_name):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(2)})
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["format"] = format_name
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r"Unsupported NPZ\+JSON format"):
        _validate(root, validator)
