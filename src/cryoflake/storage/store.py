"""Storage facade for canonical preprocessed training artifacts."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cryoflake.common import sha256_file
from cryoflake.sequence.validation import (
    validate_sequence_embedding_payload,
)
from cryoflake.storage.codec import (
    CORE_FILE_NAME,
    MAX_METADATA_BYTES,
    METADATA_FILE_NAME,
    SEQUENCE_FILE_NAME,
    SEQUENCE_ROOT_KEYS,
    _required_stores,
    incomplete_metadata_bytes,
    read_metadata,
)
from cryoflake.storage.locks import (
    preprocessed_artifact_write_locks,
)
from cryoflake.storage.reader import _read_npz_json_metadata
from cryoflake.storage.transports import (
    copy_file_to_path,
    file_reference_exists,
    is_remote_uri,
    join_remote_uri,
    read_file_bytes_limited,
    upload_file_to_s3,
)
from cryoflake.storage.validation import _validate_npz_json_store_files_metadata
from cryoflake.storage.writer import write_npz_json, write_sequence_npz_json

_SINGLE_FILE_ARTIFACT_SUFFIXES = {".pkl", ".pickle", ".prot", ".npy", ".npz"}


def preprocessed_artifact_root(
    artifact_reference: os.PathLike | str,
) -> Path | str:
    """Normalize an artifact root or metadata member reference to its root."""

    value = os.fspath(artifact_reference)
    if not value:
        raise ValueError("Preprocessed artifact reference must not be empty")
    remote = is_remote_uri(value)
    if remote:
        value = value.rstrip("/")
        if value.rsplit("/", 1)[-1] == METADATA_FILE_NAME:
            value = value.rsplit("/", 1)[0]
    path = Path(value)
    if not remote and path.name == METADATA_FILE_NAME:
        path = path.parent
    if path.suffix.lower() in _SINGLE_FILE_ARTIFACT_SUFFIXES:
        raise ValueError(
            "NPZ+JSON artifacts require a directory root or metadata.json reference; "
            "single-file artifacts are unsupported"
        )
    if not remote and path.exists() and not path.is_dir():
        raise ValueError(f"NPZ+JSON artifact root must be a directory: {path}")
    return value if remote else path


def preprocessed_artifact_member_reference(
    artifact_reference: os.PathLike | str,
    file_name: str,
) -> Path | str:
    """Resolve a fixed member beneath a local or remote artifact root."""

    if file_name not in {
        METADATA_FILE_NAME,
        CORE_FILE_NAME,
        SEQUENCE_FILE_NAME,
    }:
        raise ValueError(f"Unsupported preprocessed artifact member: {file_name}")
    root = preprocessed_artifact_root(artifact_reference)
    if is_remote_uri(root):
        return join_remote_uri(str(root), file_name)
    return Path(root) / file_name


def load_preprocessed_artifact(
    artifact_reference: os.PathLike | str,
    *,
    root_keys: Collection[str] | None = None,
    include_sequence_embeddings: bool = True,
    s3_endpoint_url: str | None = None,
) -> Mapping[str, object]:
    """Load an NPZ+JSON artifact, optionally selecting top-level artifact_mapping keys."""

    if root_keys is not None and (
        isinstance(root_keys, (str, bytes))
        or not isinstance(root_keys, Collection)
        or not all(isinstance(key, str) for key in root_keys)
    ):
        raise ValueError("root_keys must be a collection of strings")
    metadata_reference = preprocessed_artifact_member_reference(
        artifact_reference, METADATA_FILE_NAME
    )
    metadata_data = read_file_bytes_limited(
        metadata_reference,
        MAX_METADATA_BYTES,
        s3_endpoint_url=s3_endpoint_url,
    )
    metadata = read_metadata(metadata_data)
    selected_root_keys = root_keys
    if selected_root_keys is None and not include_sequence_embeddings:
        selected_root_keys = tuple(
            key for key in metadata["tree"]["items"] if key not in SEQUENCE_ROOT_KEYS
        )
    elif selected_root_keys is not None and not include_sequence_embeddings:
        selected_root_keys = tuple(
            key for key in selected_root_keys if key not in SEQUENCE_ROOT_KEYS
        )

    stores = _required_stores(
        metadata,
        root_keys=selected_root_keys,
    )
    core_data = None
    sequence_data = None
    store_members = {
        store: file_name
        for store, file_name in (
            ("core", CORE_FILE_NAME),
            ("sequence", SEQUENCE_FILE_NAME),
        )
        if store in stores
    }
    if not is_remote_uri(artifact_reference):
        if "core" in stores:
            core_data = Path(
                preprocessed_artifact_member_reference(
                    artifact_reference, CORE_FILE_NAME
                )
            )
        if "sequence" in stores:
            sequence_data = Path(
                preprocessed_artifact_member_reference(
                    artifact_reference, SEQUENCE_FILE_NAME
                )
            )
        return _read_npz_json_metadata(
            metadata,
            core_data,
            sequence_data,
            root_keys=selected_root_keys,
        )

    with tempfile.TemporaryDirectory(
        prefix="cryoflake_preprocessed_read_"
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)

        def download_store(file_name: str) -> tuple[Path, str]:
            destination = temporary_root / file_name
            copy_file_to_path(
                preprocessed_artifact_member_reference(artifact_reference, file_name),
                destination,
                s3_endpoint_url=s3_endpoint_url,
            )
            return destination, sha256_file(destination)

        if len(store_members) > 1:
            with ThreadPoolExecutor(
                max_workers=len(store_members),
                thread_name_prefix="cryoflake-preprocessed-read",
            ) as executor:
                downloads = {
                    store: executor.submit(download_store, file_name)
                    for store, file_name in store_members.items()
                }
                downloaded_stores = {
                    store: future.result() for store, future in downloads.items()
                }
        else:
            downloaded_stores = {
                store: download_store(file_name)
                for store, file_name in store_members.items()
            }
        return _read_npz_json_metadata(
            metadata,
            (downloaded_stores["core"][0] if "core" in downloaded_stores else None),
            (
                downloaded_stores["sequence"][0]
                if "sequence" in downloaded_stores
                else None
            ),
            root_keys=selected_root_keys,
            verified_store_hashes={
                store: result[1] for store, result in downloaded_stores.items()
            },
        )


def preprocessed_artifact_exists(
    artifact_reference: os.PathLike | str,
    *,
    s3_endpoint_url: str | None = None,
) -> bool:
    """Return whether a canonical artifact has every declared member."""

    # URI scheme errors remain explicit; unsupported single-file references
    # and existing local files are never considered dataset artifacts.
    is_remote_uri(artifact_reference)
    try:
        metadata_reference = preprocessed_artifact_member_reference(
            artifact_reference, METADATA_FILE_NAME
        )
    except ValueError:
        return False
    if not file_reference_exists(
        metadata_reference,
        s3_endpoint_url=s3_endpoint_url,
    ):
        return False
    try:
        metadata_data = read_file_bytes_limited(
            metadata_reference,
            MAX_METADATA_BYTES,
            s3_endpoint_url=s3_endpoint_url,
        )
        files = read_metadata(metadata_data)["files"]
    except Exception:
        return False
    return all(
        file_reference_exists(
            preprocessed_artifact_member_reference(artifact_reference, file_name),
            s3_endpoint_url=s3_endpoint_url,
        )
        for file_name in files.values()
    )


def upload_preprocessed_artifact(
    local_artifact_reference: os.PathLike,
    remote_artifact_reference: str,
    *,
    s3_endpoint_url: str | None = None,
) -> None:
    """Upload one artifact, committing its metadata object last."""

    if is_remote_uri(local_artifact_reference):
        raise ValueError("local_artifact_reference must be a local artifact directory")
    with preprocessed_artifact_write_locks(
        preprocessed_artifact_root(local_artifact_reference),
        preprocessed_artifact_root(remote_artifact_reference),
    ):
        _upload_preprocessed_artifact_unlocked(
            local_artifact_reference,
            remote_artifact_reference,
            s3_endpoint_url=s3_endpoint_url,
        )


def _upload_preprocessed_artifact_unlocked(
    local_artifact_reference: os.PathLike,
    remote_artifact_reference: str,
    *,
    s3_endpoint_url: str | None,
) -> None:
    if not is_remote_uri(remote_artifact_reference):
        raise ValueError("remote_artifact_reference must be an s3:// URI")
    local_root = Path(preprocessed_artifact_root(local_artifact_reference))
    if not local_root.is_dir():
        raise ValueError(
            "Canonical preprocessed uploads require an artifact directory: "
            f"{local_root}"
        )

    metadata_path = local_root / METADATA_FILE_NAME
    metadata_data = read_file_bytes_limited(
        metadata_path,
        MAX_METADATA_BYTES,
        s3_endpoint_url=None,
    )
    metadata = read_metadata(metadata_data)
    files = metadata["files"]
    ordered_roles = ["core", "sequence"]
    local_store_paths = {}
    for role in ordered_roles:
        file_name = files.get(role)
        if file_name is None:
            continue
        local_path = local_root / file_name
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Preprocessed artifact is missing {role} store: {local_path}"
            )
        local_store_paths[role] = (file_name, local_path)

    _validate_npz_json_store_files_metadata(
        metadata,
        local_store_paths["core"][1],
        (local_store_paths["sequence"][1] if "sequence" in local_store_paths else None),
    )
    _publish_incomplete_remote_marker_if_needed(
        remote_artifact_reference,
        s3_endpoint_url=s3_endpoint_url,
    )
    for role in ordered_roles:
        if role not in local_store_paths:
            continue
        file_name, local_path = local_store_paths[role]
        upload_file_to_s3(
            local_path,
            str(
                preprocessed_artifact_member_reference(
                    remote_artifact_reference, file_name
                )
            ),
            s3_endpoint_url=s3_endpoint_url,
        )
    upload_file_to_s3(
        metadata_path,
        str(
            preprocessed_artifact_member_reference(
                remote_artifact_reference, METADATA_FILE_NAME
            )
        ),
        s3_endpoint_url=s3_endpoint_url,
    )


def _publish_incomplete_remote_marker_if_needed(
    artifact_reference: str,
    *,
    s3_endpoint_url: str | None,
) -> None:
    metadata_reference = str(
        preprocessed_artifact_member_reference(artifact_reference, METADATA_FILE_NAME)
    )
    if not file_reference_exists(
        metadata_reference,
        s3_endpoint_url=s3_endpoint_url,
    ):
        return
    with tempfile.TemporaryDirectory(prefix="cryoflake_incomplete_") as temporary_dir:
        marker_path = Path(temporary_dir) / METADATA_FILE_NAME
        marker_path.write_bytes(incomplete_metadata_bytes())
        upload_file_to_s3(
            marker_path,
            metadata_reference,
            s3_endpoint_url=s3_endpoint_url,
        )


def write_preprocessed_artifact(
    artifact_reference: os.PathLike | str,
    artifact_mapping: Mapping[str, object],
    *,
    s3_endpoint_url: str | None = None,
) -> Path | str:
    """Write one canonical split NPZ+JSON artifact and return its root."""

    root = preprocessed_artifact_root(artifact_reference)
    if is_remote_uri(root):
        with tempfile.TemporaryDirectory(
            prefix="cryoflake_preprocessed_"
        ) as temporary_dir:
            local_root = Path(temporary_dir) / "artifact"
            write_npz_json(local_root, artifact_mapping)
            upload_preprocessed_artifact(
                local_root,
                str(root),
                s3_endpoint_url=s3_endpoint_url,
            )
        return root

    local_root = Path(root)
    resolved_root = local_root.resolve()
    if resolved_root == Path.cwd().resolve() or resolved_root == Path(
        resolved_root.anchor
    ):
        raise ValueError(
            "Preprocessed artifact root must be a dedicated subdirectory, "
            "not the current directory or a filesystem root"
        )
    if local_root.exists() and not local_root.is_dir():
        raise ValueError(
            f"Preprocessed artifact root must be a directory: {local_root}"
        )
    write_npz_json(local_root, artifact_mapping)
    return local_root


def replace_preprocessed_artifact_sequence(
    artifact_reference: os.PathLike | str,
    sequence_payload: Mapping[str, object],
    *,
    s3_endpoint_url: str | None = None,
) -> None:
    """Replace sequence data while leaving the canonical core NPZ untouched."""

    with preprocessed_artifact_write_locks(
        preprocessed_artifact_root(artifact_reference)
    ):
        _replace_preprocessed_artifact_sequence_unlocked(
            artifact_reference,
            sequence_payload,
            s3_endpoint_url=s3_endpoint_url,
        )


def _replace_preprocessed_artifact_sequence_unlocked(
    artifact_reference: os.PathLike | str,
    sequence_payload: Mapping[str, object],
    *,
    s3_endpoint_url: str | None,
) -> None:
    normalized_sequence_payload = validate_sequence_embedding_payload(sequence_payload)
    metadata_reference = preprocessed_artifact_member_reference(
        artifact_reference, METADATA_FILE_NAME
    )
    if is_remote_uri(artifact_reference):
        metadata_data = read_file_bytes_limited(
            metadata_reference,
            MAX_METADATA_BYTES,
            s3_endpoint_url=s3_endpoint_url,
        )
        with tempfile.TemporaryDirectory(prefix="cryoflake_sequence_") as temporary_dir:
            local_root = Path(temporary_dir)
            local_metadata = local_root / METADATA_FILE_NAME
            local_metadata.write_bytes(metadata_data)
            write_sequence_npz_json(
                local_metadata,
                normalized_sequence_payload,
            )
            _publish_incomplete_remote_marker_if_needed(
                str(preprocessed_artifact_root(artifact_reference)),
                s3_endpoint_url=s3_endpoint_url,
            )
            upload_file_to_s3(
                local_root / SEQUENCE_FILE_NAME,
                str(
                    preprocessed_artifact_member_reference(
                        artifact_reference, SEQUENCE_FILE_NAME
                    )
                ),
                s3_endpoint_url=s3_endpoint_url,
            )
            upload_file_to_s3(
                local_metadata,
                str(metadata_reference),
                s3_endpoint_url=s3_endpoint_url,
            )
        return

    write_sequence_npz_json(
        Path(metadata_reference),
        normalized_sequence_payload,
    )
