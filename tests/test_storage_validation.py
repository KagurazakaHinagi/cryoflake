"""Portable archive validation works with supported public NumPy APIs."""

import io
import json
import zipfile

import numpy as np
import pytest

from cryoflake.common import sha256_file
from cryoflake.storage.store import write_preprocessed_artifact
from cryoflake.storage.validation import validate_npz_json_store_files


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_archive_validator_reads_numeric_and_unicode_headers(tmp_path, version):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(
        root,
        {
            "density_grid": np.zeros((2, 2, 2), dtype=np.float32),
            "labels": np.array(["ALA", "α"], dtype="U3"),
        },
    )
    path = root / "core.npz"
    with np.load(path, allow_pickle=False) as store:
        arrays = {name: store[name] for name in store.files}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(
                member, array, version=version, allow_pickle=False
            )
            archive.writestr(f"{name}.npy", member.getvalue())
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["store_hashes"]["core"] = sha256_file(path)
    metadata_path.write_text(json.dumps(metadata))
    validate_npz_json_store_files(metadata_path, path)


def test_archive_validator_detects_tampering_without_loading_payload(tmp_path):
    root = tmp_path / "artifact"
    write_preprocessed_artifact(root, {"array": np.arange(4, dtype=np.float32)})
    path = root / "core.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        validate_npz_json_store_files(root / "metadata.json", path)
