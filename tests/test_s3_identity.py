"""S3 destination settings must participate in resumable run identities."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from cryoflake.common import json_sha256
from cryoflake.preprocess import artifact_identity
from cryoflake.preprocess.artifact_identity import preprocess_checkpoint_contract
from cryoflake.storage import transports


def _preprocess(**kwargs):
    pass


def _load_embeddings(path):
    return {}


def _contract(tmp_path):
    return preprocess_checkpoint_contract(
        manifest_csv=tmp_path / "manifest.csv",
        manifest_rows=[],
        manifest_dir=tmp_path,
        artifact_output_dir=tmp_path / "scratch",
        artifact_s3_prefix="s3://bucket/artifacts",
        s3_endpoint_url=None,
        artifact_reuse_dirs=(),
        target_voxel_size=1.0,
        fix_shift=True,
        preprocess_adapter=_preprocess,
        interface_contact_cutoff=5.0,
        require_sequence_embeddings=False,
        sequence_embedding_loader=_load_embeddings,
        download_errors={},
    )


@pytest.mark.parametrize(
    "variable, expected_key",
    [
        ("CRYOFLAKE_S3_ENDPOINT_URL", "s3_endpoint_url"),
        ("AWS_ENDPOINT_URL_S3", "s3_endpoint_url"),
        ("AWS_ENDPOINT_URL", "s3_endpoint_url"),
        ("CRYOFLAKE_S3_REGION", "s3_region"),
        ("AWS_REGION", "s3_region"),
        ("AWS_DEFAULT_REGION", "s3_region"),
    ],
)
def test_effective_s3_destination_changes_checkpoint_fingerprint(
    monkeypatch, tmp_path, variable, expected_key
):
    monkeypatch.setattr(transports, "load_s3_environment", lambda: False)
    for name in (
        "CRYOFLAKE_S3_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL",
        "CRYOFLAKE_S3_REGION",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "destination-one")
    first = _contract(tmp_path)
    monkeypatch.setenv(variable, "destination-two")
    second = _contract(tmp_path)
    assert first[expected_key] == "destination-one"
    assert second[expected_key] == "destination-two"
    assert json_sha256(first) != json_sha256(second)


def test_checkpoint_identity_uses_dotenv_destination_and_excludes_credentials(
    monkeypatch, tmp_path
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AWS_ENDPOINT_URL_S3=https://objects.example.test\n"
        "AWS_DEFAULT_REGION=test-region\n"
        "AWS_ACCESS_KEY_ID=test-key-one\n"
        "AWS_SECRET_ACCESS_KEY=test-secret-one\n"
        "AWS_SESSION_TOKEN=test-session-one\n"
    )
    for name in (
        "CRYOFLAKE_S3_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL",
        "CRYOFLAKE_S3_REGION",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        monkeypatch.setenv(name, "temporary")
        monkeypatch.delenv(name)
    monkeypatch.delenv(transports.ENV_FILE_ENV, raising=False)
    monkeypatch.setattr(transports, "_source_env_file", lambda: env_file)
    first = _contract(tmp_path)
    assert first["s3_endpoint_url"] == "https://objects.example.test"
    assert first["s3_region"] == "test-region"
    settings = transports.resolve_s3_settings()
    assert settings == {
        "endpoint_url": first["s3_endpoint_url"],
        "region_name": first["s3_region"],
    }
    serialized = json.dumps(first)
    assert "test-key-one" not in serialized
    assert "test-secret-one" not in serialized
    assert "test-session-one" not in serialized
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(name, "rotated-credential")
    assert json_sha256(_contract(tmp_path)) == json_sha256(first)


@pytest.mark.parametrize("field", ["schema_version", "full_atom_schema_version"])
def test_yaml_target_contract_change_invalidates_checkpoint(
    monkeypatch, tmp_path, field
):
    first = _contract(tmp_path)
    original = artifact_identity.target_contract()
    monkeypatch.setattr(
        artifact_identity,
        "target_contract",
        lambda: replace(original, **{field: f"{getattr(original, field)[:-1]}1"}),
    )
    second = _contract(tmp_path)
    assert (
        first["format_contracts"]["targets"][field]
        != second["format_contracts"]["targets"][field]
    )
    assert json_sha256(first) != json_sha256(second)


def test_dated_versions_are_recorded_in_checkpoint_contract(tmp_path):
    contract = _contract(tmp_path)
    assert contract["artifact_contract_version"] == "S2610210"
    assert contract["format_contracts"]["artifact"]["format_version"] == "N2610210"
    targets = contract["format_contracts"]["targets"]
    assert targets["full_atom_schema_version"] == "F2610210"
    assert targets["schema_version"] == "E2610210"


def test_semantic_contract_change_invalidates_checkpoint(monkeypatch, tmp_path):
    first = _contract(tmp_path)
    monkeypatch.setattr(
        artifact_identity, "PREPROCESSED_ARTIFACT_CONTRACT_VERSION", "S2610211"
    )
    second = _contract(tmp_path)
    assert first["artifact_contract_version"] == "S2610210"
    assert second["artifact_contract_version"] == "S2610211"
    assert json_sha256(first) != json_sha256(second)


def test_storage_contract_change_invalidates_checkpoint(monkeypatch, tmp_path):
    first = _contract(tmp_path)
    original = artifact_identity.artifact_contract()
    monkeypatch.setattr(
        artifact_identity,
        "artifact_contract",
        lambda: replace(original, format_version="N2610211"),
    )
    second = _contract(tmp_path)
    assert first["format_contracts"]["artifact"]["format_version"] == "N2610210"
    assert second["format_contracts"]["artifact"]["format_version"] == "N2610211"
    assert json_sha256(first) != json_sha256(second)
