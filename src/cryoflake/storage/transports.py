"""Local filesystem and S3-compatible object storage transports."""

from __future__ import annotations

import os
import shutil
import threading
from contextlib import closing
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

S3_ENDPOINT_URL_ENV = "CRYOFLAKE_S3_ENDPOINT_URL"
S3_REGION_ENV = "CRYOFLAKE_S3_REGION"
ENV_FILE_ENV = "CRYOFLAKE_ENV_FILE"
_S3_SESSIONS = threading.local()


def is_remote_uri(reference: os.PathLike | str) -> bool:
    """Identify supported remote references, rejecting unsupported URI schemes."""

    scheme = urlparse(os.fspath(reference)).scheme.lower()
    if scheme and scheme != "s3":
        raise ValueError(
            f"Unsupported URI scheme '{scheme}'. "
            "cryoFlake remote storage supports only s3://."
        )
    return scheme == "s3"


def _validate_s3_authority(bucket: str) -> None:
    if "@" in bucket or ":" in bucket or any(char.isspace() for char in bucket):
        raise ValueError(
            "S3 URI authority must be a bucket without credentials or a port"
        )


def split_s3_uri(uri: os.PathLike | str) -> tuple[str, str]:
    parsed = urlparse(os.fspath(uri))
    if parsed.scheme.lower() != "s3":
        raise ValueError(f"Expected s3:// URI, got {uri}")
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError(f"S3 URI must not include params, query, or fragment: {uri}")
    bucket = parsed.netloc
    key = parsed.path.removeprefix("/")
    if not bucket or not key:
        raise ValueError(f"S3 URI must include bucket and key: {uri}")
    _validate_s3_authority(bucket)
    return bucket, key


def join_remote_uri(prefix: str, name: str) -> str:
    if not is_remote_uri(prefix):
        raise ValueError(f"Remote URI prefix must use s3://: {prefix}")
    parsed = urlparse(prefix)
    if not parsed.netloc:
        raise ValueError(f"S3 URI prefix must include a bucket: {prefix}")
    _validate_s3_authority(parsed.netloc)
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError(
            f"Remote URI prefix must not include params, query, or fragment: {prefix}"
        )
    return f"{prefix.rstrip('/')}/{name.lstrip('/')}"


def resolve_artifact_list_entry(entry: str, base_dir: Path) -> Path | str:
    entry = entry.strip()
    if is_remote_uri(entry):
        return entry
    path = Path(entry)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _source_env_file() -> Path | None:
    """Locate this source checkout's .env without searching other directories."""

    source_file = Path(__file__).resolve()
    package_root = source_file.parents[1]
    source_root = package_root.parent
    project_root = source_root.parent
    if source_root.name != "src" or not (project_root / "pyproject.toml").is_file():
        return None
    return project_root / ".env"


def load_s3_environment(env_file: os.PathLike | str | None = None) -> bool:
    """Load S3 environment settings without replacing existing variables.

    Source checkouts use the package root's ``.env``. Installed packages load
    no implicit file; call this function with a path or set
    ``CRYOFLAKE_ENV_FILE`` to select one. No ancestor or working-directory
    search is performed. Standard AWS credential variables are left for
    boto3's credential provider chain to consume.
    """

    explicit_path = env_file if env_file is not None else os.environ.get(ENV_FILE_ENV)
    if explicit_path is not None:
        dotenv_path = Path(explicit_path).expanduser()
        if not dotenv_path.is_file():
            raise FileNotFoundError(
                f"S3 environment file does not exist: {dotenv_path}"
            )
    else:
        dotenv_path = _source_env_file()
        if dotenv_path is None or not dotenv_path.is_file():
            return False
    return load_dotenv(dotenv_path=dotenv_path, override=False)


def resolve_s3_settings(s3_endpoint_url: str | None = None) -> dict[str, str]:
    """Resolve effective client settings for I/O and checkpoint identities.

    Credentials stay in boto3's credential provider chain and are never
    included in the returned settings.
    """
    load_s3_environment()
    endpoint_url = (
        s3_endpoint_url
        or os.environ.get(S3_ENDPOINT_URL_ENV)
        or os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("AWS_ENDPOINT_URL")
    )
    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    region_name = (
        os.environ.get(S3_REGION_ENV)
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    if region_name:
        kwargs["region_name"] = region_name
    return kwargs


def _get_s3_client(s3_endpoint_url: str | None = None):
    import boto3

    settings = resolve_s3_settings(s3_endpoint_url)
    owner_pid = os.getpid()
    session = getattr(_S3_SESSIONS, "session", None)
    if session is None or _S3_SESSIONS.owner_pid != owner_pid:
        # boto3.client() shares a default Session whose lazy components are
        # unsafe to initialize concurrently. Reuse only within one thread and
        # process; each operation still owns and closes its own client.
        session = boto3.Session()
        _S3_SESSIONS.session = session
        _S3_SESSIONS.owner_pid = owner_pid
    return session.client("s3", **settings)


def validate_remote_configuration(
    remote_uri: os.PathLike | str,
    s3_endpoint_url: str | None = None,
) -> None:
    if not is_remote_uri(remote_uri):
        raise ValueError(f"Remote URI must use s3://: {remote_uri}")
    # Prefixes may refer to an entire bucket rather than a single object.
    join_remote_uri(os.fspath(remote_uri), "configuration-check")
    with closing(_get_s3_client(s3_endpoint_url)):
        pass


def _is_not_found_error(exc: BaseException) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    response = getattr(exc, "response", {})
    error_code = str(response.get("Error", {}).get("Code", ""))
    status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return error_code in {"404", "NoSuchKey", "NotFound"} or status_code == 404


def file_reference_exists(
    file_reference: os.PathLike | str, s3_endpoint_url: str | None = None
) -> bool:
    if is_remote_uri(file_reference):
        bucket, key = split_s3_uri(file_reference)
        with closing(_get_s3_client(s3_endpoint_url)) as client:
            try:
                client.head_object(Bucket=bucket, Key=key)
            except Exception as exc:
                if _is_not_found_error(exc):
                    return False
                raise
        return True
    return Path(file_reference).is_file()


def file_reference_identity(
    file_reference: os.PathLike | str,
    s3_endpoint_url: str | None = None,
) -> dict[str, object] | None:
    """Return a stable identity for a local file or S3 object, if available."""

    if is_remote_uri(file_reference):
        bucket, key = split_s3_uri(file_reference)
        with closing(_get_s3_client(s3_endpoint_url)) as client:
            try:
                response = client.head_object(Bucket=bucket, Key=key)
            except Exception as exc:
                if _is_not_found_error(exc):
                    return None
                raise
        content_length = response.get("ContentLength")
        etag = str(response.get("ETag") or "").strip('"')
        version_id = str(response.get("VersionId") or "")
        if content_length is None or not (etag or version_id):
            return None
        return {
            "kind": "s3",
            "content_length": int(content_length),
            "etag": etag,
            "version_id": version_id,
        }

    local_path = Path(file_reference)
    try:
        stat_result = local_path.stat()
    except FileNotFoundError:
        return None
    if not local_path.is_file():
        return None
    return {
        "kind": "local",
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "ctime_ns": stat_result.st_ctime_ns,
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
    }


def read_file_bytes(
    file_reference: os.PathLike | str,
    s3_endpoint_url: str | None = None,
) -> bytes:
    """Read one local or S3 object without interpreting its contents."""

    if is_remote_uri(file_reference):
        bucket, key = split_s3_uri(file_reference)
        with closing(_get_s3_client(s3_endpoint_url)) as client:
            response = client.get_object(Bucket=bucket, Key=key)
            with closing(response["Body"]) as body:
                return body.read()
    return Path(file_reference).read_bytes()


def _read_limited_stream(body, max_bytes: int) -> bytes:
    chunks = []
    total = 0
    while total <= max_bytes:
        chunk = body.read(min(1024 * 1024, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    payload = b"".join(chunks)
    if len(payload) > max_bytes:
        raise ValueError(f"Object exceeds the configured {max_bytes}-byte size limit")
    return payload


def read_file_bytes_limited(
    file_reference: os.PathLike | str,
    max_bytes: int,
    s3_endpoint_url: str | None = None,
) -> bytes:
    """Read a small object while enforcing a hard byte limit."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if is_remote_uri(file_reference):
        bucket, key = split_s3_uri(file_reference)
        with closing(_get_s3_client(s3_endpoint_url)) as client:
            response = client.get_object(Bucket=bucket, Key=key)
            content_length = response.get("ContentLength")
            with closing(response["Body"]) as body:
                if content_length is not None and int(content_length) > max_bytes:
                    raise ValueError(
                        f"Object exceeds the configured {max_bytes}-byte size limit"
                    )
                return _read_limited_stream(body, max_bytes)

    local_path = Path(file_reference)
    if local_path.stat().st_size > max_bytes:
        raise ValueError(f"Object exceeds the configured {max_bytes}-byte size limit")
    with local_path.open("rb") as input_file:
        return _read_limited_stream(input_file, max_bytes)


def copy_file_to_path(
    source_file_reference: os.PathLike | str,
    destination_path: os.PathLike | str,
    s3_endpoint_url: str | None = None,
) -> None:
    """Stream one local or S3 object into a local file."""

    remote_source = is_remote_uri(source_file_reference)
    if is_remote_uri(destination_path):
        raise ValueError("Copy destination must be a local file path")
    if remote_source:
        bucket, key = split_s3_uri(source_file_reference)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if remote_source:
        with closing(_get_s3_client(s3_endpoint_url)) as client:
            client.download_file(bucket, key, str(destination_path))
        return
    shutil.copyfile(Path(source_file_reference), destination_path)


def upload_file_to_s3(
    source_path: os.PathLike | str,
    destination_s3_uri: str,
    s3_endpoint_url: str | None = None,
) -> None:
    """Upload one local file to an S3 object."""

    if not is_remote_uri(destination_s3_uri):
        raise ValueError(f"Remote URI must use s3://: {destination_s3_uri}")
    if is_remote_uri(source_path):
        raise ValueError("Upload source must be a local path")
    bucket, key = split_s3_uri(destination_s3_uri)
    with closing(_get_s3_client(s3_endpoint_url)) as client:
        client.upload_file(str(Path(source_path)), bucket, key)
