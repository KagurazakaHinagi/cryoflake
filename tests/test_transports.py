from __future__ import annotations

import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from cryoflake.storage import transports


@pytest.fixture(autouse=True)
def isolated_s3_sessions(monkeypatch):
    monkeypatch.setattr(transports, "_S3_SESSIONS", threading.local())


class _MissingObject(Exception):
    response = {
        "Error": {"Code": "NoSuchKey"},
        "ResponseMetadata": {"HTTPStatusCode": 404},
    }


class _FakeS3Client:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.close_count = 0

    def close(self):
        self.close_count += 1

    def upload_file(self, Filename, Bucket, Key):
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()

    def download_file(self, Bucket, Key, Filename):
        try:
            payload = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise _MissingObject() from exc
        Path(Filename).write_bytes(payload)

    def get_object(self, Bucket, Key):
        try:
            payload = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise _MissingObject() from exc
        return {
            "Body": io.BytesIO(payload),
            "ContentLength": len(payload),
        }

    def head_object(self, Bucket, Key):
        try:
            payload = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise _MissingObject() from exc
        return {
            "ContentLength": len(payload),
            "ETag": f'"etag-{len(payload)}"',
        }


def test_local_transport_reads_copies_and_enforces_limits(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "nested" / "destination"
    source.write_bytes(b"artifact")

    assert transports.file_reference_exists(file_reference=source)
    assert transports.read_file_bytes(file_reference=source) == b"artifact"
    assert (
        transports.read_file_bytes_limited(file_reference=source, max_bytes=8)
        == b"artifact"
    )
    with pytest.raises(ValueError, match="size limit"):
        transports.read_file_bytes_limited(source, 7)

    transports.copy_file_to_path(
        source_file_reference=source, destination_path=destination
    )
    assert destination.read_bytes() == b"artifact"


@pytest.mark.parametrize(
    "source",
    ["s3:///key", "s3://bucket/key?version=1", "s3://bucket/key#fragment"],
)
def test_invalid_remote_download_does_not_create_destination(source, tmp_path):
    destination = tmp_path / "new" / "copy"
    with pytest.raises(ValueError, match="S3 URI"):
        transports.copy_file_to_path(source, destination)
    assert not destination.parent.exists()


def test_s3_object_keys_preserve_leading_slashes():
    assert transports.split_s3_uri("s3://bucket//key") == ("bucket", "/key")


@pytest.mark.parametrize(
    "authority", ["user@bucket", "user:secret@bucket", "bucket:9000"]
)
def test_s3_uri_rejects_credentials_and_ports_before_client_creation(authority):
    uri = f"s3://{authority}/key"
    with pytest.raises(ValueError, match="bucket|authority"):
        transports.split_s3_uri(uri)
    with pytest.raises(ValueError, match="bucket|authority"):
        transports.join_remote_uri(uri, "metadata.json")


def test_s3_transport_round_trip_with_explicit_endpoint(monkeypatch, tmp_path):
    fake_client = _FakeS3Client()
    endpoints = []

    def fake_get_s3_client(endpoint_url=None):
        endpoints.append(endpoint_url)
        return fake_client

    monkeypatch.setattr(
        transports,
        "_get_s3_client",
        fake_get_s3_client,
    )
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"artifact")
    uri = "s3://training-data/preprocessed/core.npz"

    assert not transports.file_reference_exists(
        uri,
        s3_endpoint_url="https://r2.test",
    )
    transports.upload_file_to_s3(
        source,
        uri,
        s3_endpoint_url="https://r2.test",
    )
    assert transports.file_reference_identity(
        uri,
        s3_endpoint_url="https://r2.test",
    ) == {
        "kind": "s3",
        "content_length": 8,
        "etag": "etag-8",
        "version_id": "",
    }
    assert transports.file_reference_exists(
        uri,
        s3_endpoint_url="https://r2.test",
    )
    assert (
        transports.read_file_bytes_limited(
            uri,
            8,
            s3_endpoint_url="https://r2.test",
        )
        == b"artifact"
    )
    transports.copy_file_to_path(
        uri,
        destination,
        s3_endpoint_url="https://r2.test",
    )
    assert destination.read_bytes() == b"artifact"
    assert endpoints == ["https://r2.test"] * 6
    assert fake_client.close_count == 6


@pytest.mark.parametrize("scheme", ["hf", "https", "gs", "file"])
def test_unsupported_uri_schemes_fail_before_local_access(scheme, tmp_path):
    uri = f"{scheme}://bucket/object"
    operations = [
        lambda: transports.is_remote_uri(uri),
        lambda: transports.file_reference_exists(uri),
        lambda: transports.file_reference_identity(uri),
        lambda: transports.read_file_bytes(uri),
        lambda: transports.read_file_bytes_limited(uri, 8),
        lambda: transports.copy_file_to_path(uri, tmp_path / "nested" / "copy"),
        lambda: transports.upload_file_to_s3(tmp_path / "source", uri),
        lambda: transports.validate_remote_configuration(uri),
        lambda: transports.join_remote_uri(uri, "core.npz"),
        lambda: transports.resolve_artifact_list_entry(uri, tmp_path),
    ]
    for operation in operations:
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            operation()
    assert not (tmp_path / "nested").exists()


def test_local_transport_identity_and_missing_file(tmp_path):
    source = tmp_path / "source"
    assert not transports.file_reference_exists(source)
    assert transports.file_reference_identity(source) is None
    source.write_bytes(b"artifact")
    identity = transports.file_reference_identity(source)
    assert identity["kind"] == "local"
    assert identity["size"] == 8
    assert transports.file_reference_identity(tmp_path) is None


def test_s3_transport_does_not_hide_permission_errors(monkeypatch):
    def forbidden(**kwargs):
        raise PermissionError("access denied")

    client = SimpleNamespace(head_object=forbidden, close=lambda: None)
    monkeypatch.setattr(transports, "_get_s3_client", lambda endpoint=None: client)
    for operation in [
        transports.file_reference_exists,
        transports.file_reference_identity,
    ]:
        with pytest.raises(PermissionError, match="access denied"):
            operation("s3://bucket/object")


def test_s3_stream_limit_closes_body_on_failure(monkeypatch):
    bodies = []

    def get_object(**kwargs):
        body = io.BytesIO(b"larger than the limit")
        bodies.append(body)
        return {"Body": body}

    closed_clients = []
    client = SimpleNamespace(
        get_object=get_object, close=lambda: closed_clients.append(True)
    )
    monkeypatch.setattr(transports, "_get_s3_client", lambda endpoint=None: client)
    with pytest.raises(ValueError, match="size limit"):
        transports.read_file_bytes_limited("s3://bucket/object", 8)
    assert bodies[0].closed
    assert closed_clients == [True]


@pytest.mark.parametrize("limited", [False, True])
def test_s3_read_interrupt_closes_body_before_client(monkeypatch, limited):
    closed = []

    class Body:
        def read(self, *args):
            raise KeyboardInterrupt("interrupted stream")

        def close(self):
            closed.append("body")

    client = SimpleNamespace(
        get_object=lambda **_: {"Body": Body()},
        close=lambda: closed.append("client"),
    )
    monkeypatch.setattr(transports, "_get_s3_client", lambda endpoint=None: client)
    with pytest.raises(KeyboardInterrupt, match="interrupted stream"):
        if limited:
            transports.read_file_bytes_limited("s3://bucket/object", 8)
        else:
            transports.read_file_bytes("s3://bucket/object")
    assert closed == ["body", "client"]


def test_s3_transport_uses_sdk_file_transfers(monkeypatch, tmp_path):
    calls = []
    client = SimpleNamespace(
        download_file=lambda *args: calls.append(("download", args)),
        upload_file=lambda *args: calls.append(("upload", args)),
        close=lambda: None,
    )
    monkeypatch.setattr(transports, "_get_s3_client", lambda endpoint=None: client)
    source = tmp_path / "source"
    source.write_bytes(b"artifact")
    destination = tmp_path / "nested" / "destination"
    transports.upload_file_to_s3(source, "s3://bucket/path/object")
    transports.copy_file_to_path("s3://bucket/path/object", destination)
    assert calls == [
        ("upload", (str(source), "bucket", "path/object")),
        ("download", ("bucket", "path/object", str(destination))),
    ]


@pytest.mark.parametrize(
    "operation, fails",
    [("validate", False)]
    + [
        (operation, fails)
        for operation in ["exists", "identity", "read", "limited", "download", "upload"]
        for fails in [False, True]
    ],
)
def test_s3_operations_close_client_on_success_and_failure(
    monkeypatch, tmp_path, operation, fails
):
    body = io.BytesIO(b"artifact")

    class Client(_FakeS3Client):
        def request(self, *args, **kwargs):
            if fails:
                raise RuntimeError("remote request failed")
            return {"Body": body, "ContentLength": 8, "ETag": '"etag"'}

        head_object = get_object = download_file = upload_file = request

    client = Client()
    monkeypatch.setattr(transports, "_get_s3_client", lambda endpoint=None: client)
    uri = "s3://bucket/object"
    local = tmp_path / "object"
    local.write_bytes(b"artifact")
    operations = {
        "validate": lambda: transports.validate_remote_configuration(uri),
        "exists": lambda: transports.file_reference_exists(uri),
        "identity": lambda: transports.file_reference_identity(uri),
        "read": lambda: transports.read_file_bytes(uri),
        "limited": lambda: transports.read_file_bytes_limited(uri, 8),
        "download": lambda: transports.copy_file_to_path(uri, local),
        "upload": lambda: transports.upload_file_to_s3(local, uri),
    }
    if fails:
        with pytest.raises(RuntimeError, match="remote request failed"):
            operations[operation]()
    else:
        operations[operation]()
    assert client.close_count == 1
    if operation in {"read", "limited"} and not fails:
        assert body.closed


def test_s3_environment_file_preserves_existing_values(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AWS_ACCESS_KEY_ID=test-file-key\n"
        "AWS_SECRET_ACCESS_KEY=test-file-secret\n"
        "CRYOFLAKE_S3_ENDPOINT_URL=https://s3.example.test\n"
        "CRYOFLAKE_S3_REGION=test-region\n"
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-process-key")
    # Register dotenv's new variables with monkeypatch so cleanup restores them.
    for name in [
        "AWS_SECRET_ACCESS_KEY",
        transports.S3_ENDPOINT_URL_ENV,
        transports.S3_REGION_ENV,
    ]:
        monkeypatch.setenv(name, "temporary")
        monkeypatch.delenv(name)
    calls = []

    def client(service, **kwargs):
        calls.append((service, kwargs))
        assert transports.os.environ["AWS_ACCESS_KEY_ID"] == "test-process-key"
        assert transports.os.environ["AWS_SECRET_ACCESS_KEY"] == "test-file-secret"
        return "client"

    def session():
        assert transports.os.environ["AWS_SECRET_ACCESS_KEY"] == "test-file-secret"
        return SimpleNamespace(client=client)

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=session))
    monkeypatch.setenv(transports.ENV_FILE_ENV, str(env_file))
    assert transports._get_s3_client() == "client"
    assert calls == [
        (
            "s3",
            {
                "endpoint_url": "https://s3.example.test",
                "region_name": "test-region",
            },
        )
    ]
    transports._get_s3_client("https://explicit.example.test")
    assert calls[-1][1]["endpoint_url"] == "https://explicit.example.test"


def test_environment_path_argument_overrides_env_setting(monkeypatch, tmp_path):
    env_file = tmp_path / "credentials.env"
    env_file.write_text("CRYOFLAKE_TEST_ENV=explicit\n")
    monkeypatch.setenv("CRYOFLAKE_TEST_ENV", "temporary")
    monkeypatch.delenv("CRYOFLAKE_TEST_ENV")
    monkeypatch.setenv(transports.ENV_FILE_ENV, str(tmp_path / "missing.env"))
    assert transports.load_s3_environment(env_file)
    assert transports.os.environ["CRYOFLAKE_TEST_ENV"] == "explicit"
    with pytest.raises(FileNotFoundError, match="environment file"):
        transports.load_s3_environment()


def test_environment_does_not_search_working_directory_or_ancestors(
    monkeypatch, tmp_path
):
    parent_env = tmp_path / ".env"
    parent_env.write_text("CRYOFLAKE_TEST_ENV=foreign\n")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    monkeypatch.delenv(transports.ENV_FILE_ENV, raising=False)
    monkeypatch.delenv("CRYOFLAKE_TEST_ENV", raising=False)
    monkeypatch.setattr(transports, "_source_env_file", lambda: None)
    assert not transports.load_s3_environment()
    assert "CRYOFLAKE_TEST_ENV" not in transports.os.environ


def test_installed_package_has_no_implicit_env_file(monkeypatch, tmp_path):
    installed_module = tmp_path / "site-packages/cryoflake/storage/transports.py"
    monkeypatch.setattr(transports, "__file__", str(installed_module))
    assert transports._source_env_file() is None


def test_source_checkout_uses_only_its_root_env(monkeypatch, tmp_path):
    project_root = tmp_path / "cryoflake"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text('[project]\nname = "cryoflake"\n')
    module = project_root / "src/cryoflake/storage/transports.py"
    monkeypatch.setattr(transports, "__file__", str(module))
    assert transports._source_env_file() == project_root / ".env"
    (project_root / ".env").write_text("CRYOFLAKE_TEST_ENV=source-root\n")
    (tmp_path / ".env").write_text("CRYOFLAKE_TEST_ENV=foreign-parent\n")
    monkeypatch.delenv(transports.ENV_FILE_ENV, raising=False)
    monkeypatch.setenv("CRYOFLAKE_TEST_ENV", "temporary")
    monkeypatch.delenv("CRYOFLAKE_TEST_ENV")
    assert transports.load_s3_environment()
    assert transports.os.environ["CRYOFLAKE_TEST_ENV"] == "source-root"


@pytest.mark.parametrize(
    ("endpoint_variable", "region_variable"),
    [
        ("CRYOFLAKE_S3_ENDPOINT_URL", "CRYOFLAKE_S3_REGION"),
        ("AWS_ENDPOINT_URL_S3", "AWS_REGION"),
        ("AWS_ENDPOINT_URL", "AWS_DEFAULT_REGION"),
    ],
)
def test_s3_client_supports_standard_aws_environment(
    monkeypatch, endpoint_variable, region_variable
):
    for name in [
        "CRYOFLAKE_S3_ENDPOINT_URL",
        "CRYOFLAKE_S3_REGION",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(transports, "load_s3_environment", lambda: False)
    monkeypatch.setenv(endpoint_variable, "https://s3.example.test")
    monkeypatch.setenv(region_variable, "test-region")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(
            Session=lambda: SimpleNamespace(
                client=lambda *args, **kwargs: calls.append((args, kwargs))
            )
        ),
    )
    transports._get_s3_client()
    assert calls == [
        (
            ("s3",),
            {
                "endpoint_url": "https://s3.example.test",
                "region_name": "test-region",
            },
        )
    ]


def test_s3_sessions_are_reused_only_within_their_thread_and_process(monkeypatch):
    ready = threading.Barrier(2)
    sessions = []

    class Session:
        def __init__(self):
            self.thread = threading.get_ident()
            sessions.append(self)

        def client(self, service, **kwargs):
            assert threading.get_ident() == self.thread
            return self

    monkeypatch.setattr(transports, "resolve_s3_settings", lambda _: {})
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=Session))

    def create_clients():
        first = transports._get_s3_client()
        ready.wait(timeout=2)
        assert transports._get_s3_client() is first
        return first

    with ThreadPoolExecutor(max_workers=2) as executor:
        clients = [executor.submit(create_clients) for _ in range(2)]
        assert clients[0].result(timeout=5) is not clients[1].result(timeout=5)
    assert len(sessions) == 2

    parent_session = transports._get_s3_client()
    parent_pid = transports.os.getpid()
    monkeypatch.setattr(transports.os, "getpid", lambda: parent_pid + 1)
    assert transports._get_s3_client() is not parent_session
