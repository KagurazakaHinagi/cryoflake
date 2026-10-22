from __future__ import annotations

import ast
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from standard_target import load_standard_artifact_mapping, standard_expected

from cryoflake import (
    FASTASequence,
    load_preprocessed_artifact,
    preprocessed_artifact_exists,
    replace_preprocessed_artifact_sequence,
    write_preprocessed_artifact,
)
from cryoflake.storage import transports
from cryoflake.storage.artifacts import validate_preprocessed_artifact
from cryoflake.storage.codec import STORE_GENERATION_KEY
from cryoflake.storage.store import upload_preprocessed_artifact
from cryoflake.storage.transports import join_remote_uri, split_s3_uri


def _payload(*, include_sequence: bool = True) -> dict[str, object]:
    payload = load_standard_artifact_mapping()
    # Explicit dtype-conversion probe on the canonical real density.
    payload["density_grid"] = payload["density_grid"].astype(np.float64)
    if include_sequence:
        # Controlled numeric embeddings test storage, not model inference.
        payload.update(
            {
                "sequence": np.arange(12, dtype=np.float32).reshape(2, 6),
                "sequence_mask": np.asarray([True, False]),
            }
        )
    return payload


def test_local_round_trip_uses_split_stores_and_float32_grid(tmp_path):
    artifact = tmp_path / "example"
    payload = _payload()

    assert (
        write_preprocessed_artifact(
            artifact / "metadata.json",
            payload,
        )
        == artifact
    )
    loaded = load_preprocessed_artifact(artifact)

    assert {path.name for path in artifact.iterdir()} == {
        "metadata.json",
        "core.npz",
        "sequence.npz",
    }
    np.testing.assert_array_equal(
        loaded["density_grid"],
        np.asarray(payload["density_grid"], dtype=np.float32),
    )
    assert loaded["density_grid"].dtype == np.float32
    np.testing.assert_array_equal(loaded["sequence"], payload["sequence"])

    with np.load(artifact / "core.npz", allow_pickle=False) as core:
        assert "sequence" not in core.files
        assert STORE_GENERATION_KEY in core.files
    with np.load(artifact / "sequence.npz", allow_pickle=False) as sequence:
        assert set(sequence.files) == {
            "sequence",
            "sequence_mask",
            STORE_GENERATION_KEY,
        }

    metadata = json.loads((artifact / "metadata.json").read_text())
    assert metadata["format"] == "cryoflake-preprocessed"
    assert metadata["format_version"] == "N2610210"
    assert (
        metadata["store_hashes"]["core"]
        == hashlib.sha256((artifact / "core.npz").read_bytes()).hexdigest()
    )
    assert (
        metadata["store_hashes"]["sequence"]
        == hashlib.sha256((artifact / "sequence.npz").read_bytes()).hexdigest()
    )
    assert preprocessed_artifact_exists(artifact)
    validated = validate_preprocessed_artifact(
        artifact, require_sequence_embeddings=True
    )
    expected = standard_expected()
    assert validated.grid_shape == expected["grid_shape"]
    assert validated.protein_nodes == expected["protein_nodes"]
    assert validated.nucleotide_nodes == expected["nucleotide_nodes"]
    assert validated.component_nodes == expected["component_nodes"]
    np.testing.assert_array_equal(
        loaded["full_atom_targets"]["atom_names"],
        payload["full_atom_targets"]["atom_names"],
    )


@pytest.mark.parametrize("representation", ["array", "npy", "npz"])
def test_grid_inputs_share_normalization_before_storage(tmp_path, representation):
    artifact = tmp_path / "example"
    payload = _payload(include_sequence=False)
    expected = payload["density_grid"].astype(np.float32)
    if representation != "array":
        buffer = io.BytesIO()
        save = np.save if representation == "npy" else np.savez_compressed
        save(buffer, payload["density_grid"])
        # Input normalization rewinds streams before decoding them.
        payload["density_grid"] = buffer

    write_preprocessed_artifact(artifact, payload)
    loaded = load_preprocessed_artifact(artifact)

    assert loaded["density_grid"].dtype == np.float32
    np.testing.assert_array_equal(loaded["density_grid"], expected)


def test_invalid_grid_is_rejected_before_artifact_creation(tmp_path):
    artifact = tmp_path / "example"
    with pytest.raises(ValueError, match="finite numeric 3D density grid"):
        write_preprocessed_artifact(artifact, {"density_grid": np.zeros((2, 2))})
    assert not artifact.exists()


def test_core_only_read_does_not_require_sequence_store(tmp_path):
    artifact = tmp_path / "example"
    write_preprocessed_artifact(artifact, _payload())
    (artifact / "sequence.npz").unlink()

    loaded = load_preprocessed_artifact(
        artifact,
        include_sequence_embeddings=False,
    )

    assert "density_grid" in loaded
    assert "sequence" not in loaded
    assert "sequence_mask" not in loaded
    assert not preprocessed_artifact_exists(artifact)


def test_sequence_replacement_preserves_core_store(tmp_path):
    artifact = tmp_path / "example"
    write_preprocessed_artifact(artifact, _payload())
    original_core_digest = hashlib.sha256(
        (artifact / "core.npz").read_bytes()
    ).hexdigest()
    replacement = np.full((3, 4), 7.0, dtype=np.float32)

    replace_preprocessed_artifact_sequence(
        artifact,
        {
            "sequence": replacement,
            "sequence_mask": np.asarray([True, False, True]),
        },
    )
    loaded = load_preprocessed_artifact(artifact)

    assert (
        hashlib.sha256((artifact / "core.npz").read_bytes()).hexdigest()
        == original_core_digest
    )
    np.testing.assert_array_equal(loaded["sequence"], replacement)
    np.testing.assert_array_equal(
        loaded["sequence_mask"],
        np.asarray([True, False, True]),
    )


def test_reader_reconstructs_portable_fasta_sequence(tmp_path):
    artifact = tmp_path / "example"
    fasta = FASTASequence(sequence="ACD", chain_count=2, chain_ids=["A", "B"])
    payload = _payload(include_sequence=False)
    payload["fasta"] = {"sequences": [fasta]}

    write_preprocessed_artifact(artifact, payload)
    loaded = load_preprocessed_artifact(artifact)

    assert loaded["fasta"]["sequences"] == [fasta]
    assert isinstance(loaded["fasta"]["sequences"][0], FASTASequence)


def test_store_corruption_fails_closed(tmp_path):
    artifact = tmp_path / "example"
    write_preprocessed_artifact(artifact, _payload())
    with (artifact / "core.npz").open("ab") as output:
        output.write(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_preprocessed_artifact(artifact)


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".prot", ".npy", ".npz", ".PKL"])
def test_single_file_artifacts_are_rejected_without_reading_or_overwriting(
    tmp_path, monkeypatch, suffix
):
    reference = tmp_path / f"unsupported{suffix}"
    contents = b"single-file artifacts are unsupported"
    reference.write_bytes(contents)

    def unexpected_open(*args, **kwargs):
        pytest.fail("Single-file artifact contents must never be opened")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", unexpected_open)
        for operation in [
            lambda: load_preprocessed_artifact(reference),
            lambda: write_preprocessed_artifact(reference, _payload()),
            lambda: replace_preprocessed_artifact_sequence(reference, {}),
            lambda: validate_preprocessed_artifact(reference),
            lambda: upload_preprocessed_artifact(reference, "s3://bucket/artifact"),
        ]:
            with pytest.raises(ValueError, match=r"NPZ\+JSON"):
                operation()
        assert not preprocessed_artifact_exists(reference)
    assert reference.read_bytes() == contents
    output = tmp_path / f"new{suffix}"
    with pytest.raises(ValueError, match=r"NPZ\+JSON"):
        write_preprocessed_artifact(output, _payload())
    assert not output.exists()


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".prot", ".npy", ".npz"])
def test_remote_single_file_artifacts_are_rejected_before_s3_access(
    monkeypatch, suffix
):
    reference = f"s3://bucket/unsupported{suffix}"

    def unexpected_client(*args, **kwargs):
        pytest.fail("Single-file artifact references must not access S3")

    monkeypatch.setattr(transports, "_get_s3_client", unexpected_client)
    for operation in [
        lambda: load_preprocessed_artifact(reference),
        lambda: write_preprocessed_artifact(reference, _payload()),
        lambda: replace_preprocessed_artifact_sequence(reference, {}),
        lambda: validate_preprocessed_artifact(reference),
        lambda: upload_preprocessed_artifact("local-artifact", reference),
    ]:
        with pytest.raises(ValueError, match=r"NPZ\+JSON"):
            operation()
    assert not preprocessed_artifact_exists(reference)


def test_arbitrary_existing_files_are_not_artifact_roots(tmp_path):
    reference = tmp_path / "serialized-data"
    reference.write_bytes(b"not an artifact directory")
    with pytest.raises(ValueError, match="root must be a directory"):
        load_preprocessed_artifact(reference)
    assert not preprocessed_artifact_exists(reference)


def test_public_storage_api_exposes_canonical_artifact_operations():
    import inspect

    import cryoflake
    import cryoflake.storage as storage
    from cryoflake.storage import store

    for module in (cryoflake, storage, store):
        assert not hasattr(module, "LegacyPicklePolicy")
        assert not hasattr(module, "LegacyPreprocessedPickleWarning")
        assert not hasattr(module, "load_canonical_preprocessed_example")
        assert not hasattr(module, "load_preprocessed_example")
        assert not hasattr(module, "write_preprocessed_example")
        assert module.load_preprocessed_artifact is load_preprocessed_artifact
        assert module.write_preprocessed_artifact is write_preprocessed_artifact
    for function in (load_preprocessed_artifact, write_preprocessed_artifact):
        assert function.__name__.endswith("_preprocessed_artifact")
    for function in (
        load_preprocessed_artifact,
        replace_preprocessed_artifact_sequence,
    ):
        assert "legacy_pickle_policy" not in inspect.signature(function).parameters


def test_remote_uri_helpers_validate_and_join_paths():
    assert split_s3_uri("s3://bucket/path/object") == (
        "bucket",
        "path/object",
    )
    assert join_remote_uri("s3://bucket/prefix", "metadata.json") == (
        "s3://bucket/prefix/metadata.json"
    )
    with pytest.raises(ValueError, match="Unsupported URI scheme"):
        join_remote_uri("hf://org/bucket/prefix/", "/core.npz")
    with pytest.raises(ValueError, match="bucket and key"):
        split_s3_uri("s3://bucket")
    with pytest.raises(ValueError, match="query"):
        split_s3_uri("s3://bucket/key?version=1")


def test_source_and_import_boundary_excludes_model_stack():
    package_root = Path(__file__).parents[1] / "src" / "cryoflake"
    forbidden_imports = []
    for source_path in package_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            forbidden_imports.extend(
                name
                for name in names
                if name.split(".", 1)[0] in {"torch", "Bio", "pickle", "_pickle"}
            )
    assert forbidden_imports == []

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cryoflake; "
                "assert 'torch' not in sys.modules; "
                "assert 'Bio' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "reference", ["hf://org/bucket/artifact", "https://example.test/artifact"]
)
def test_storage_rejects_unsupported_remote_schemes(reference):
    for operation in [
        lambda: write_preprocessed_artifact(reference, _payload()),
        lambda: load_preprocessed_artifact(reference),
        lambda: preprocessed_artifact_exists(reference),
        lambda: preprocessed_artifact_exists(f"{reference}.pkl"),
        lambda: replace_preprocessed_artifact_sequence(reference, {}),
    ]:
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            operation()


def test_s3_artifact_round_trip_and_sequence_replacement(monkeypatch):
    objects = {}
    uploads = []

    class Client:
        def close(self):
            pass

        def upload_file(self, Filename, Bucket, Key):
            objects[(Bucket, Key)] = Path(Filename).read_bytes()
            uploads.append(Key)

        def download_file(self, Bucket, Key, Filename):
            Path(Filename).write_bytes(objects[(Bucket, Key)])

        def get_object(self, Bucket, Key):
            payload = objects[(Bucket, Key)]
            return {"Body": io.BytesIO(payload), "ContentLength": len(payload)}

        def head_object(self, Bucket, Key):
            if (Bucket, Key) not in objects:
                raise FileNotFoundError(Key)
            return {"ContentLength": len(objects[(Bucket, Key)])}

    client = Client()
    monkeypatch.setattr(transports, "_get_s3_client", lambda endpoint=None: client)
    root = "s3://bucket/artifacts/example"
    assert not preprocessed_artifact_exists(root)
    assert write_preprocessed_artifact(root, _payload()) == root
    assert (
        json.loads(objects[("bucket", "artifacts/example/metadata.json")])[
            "format_version"
        ]
        == "N2610210"
    )
    assert uploads == [
        "artifacts/example/core.npz",
        "artifacts/example/sequence.npz",
        "artifacts/example/metadata.json",
    ]
    assert preprocessed_artifact_exists(root)
    loaded = load_preprocessed_artifact(root)
    np.testing.assert_array_equal(loaded["sequence"], _payload()["sequence"])
    validated = validate_preprocessed_artifact(root, require_sequence_embeddings=True)
    assert validated.protein_nodes == standard_expected()["protein_nodes"]
    assert validated.component_nodes == standard_expected()["component_nodes"]
    core_before = objects[("bucket", "artifacts/example/core.npz")]
    sequence = np.ones((2, 4), dtype=np.float32)
    replace_preprocessed_artifact_sequence(root, {"sequence": sequence})
    np.testing.assert_array_equal(
        load_preprocessed_artifact(root)["sequence"], sequence
    )
    assert objects[("bucket", "artifacts/example/core.npz")] == core_before
    assert (
        json.loads(objects[("bucket", "artifacts/example/metadata.json")])[
            "format_version"
        ]
        == "N2610210"
    )
    assert uploads[-3:] == [
        "artifacts/example/metadata.json",
        "artifacts/example/sequence.npz",
        "artifacts/example/metadata.json",
    ]
