from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Optional

from cryoflake.common import json_sha256
from cryoflake.config import artifact_contract, target_contract
from cryoflake.constants import FULL_ATOM_TARGET_KEYS, FULL_ATOM_TOPOLOGY_KEYS
from cryoflake.preprocess.checkpoint_identity import callable_identity
from cryoflake.storage.codec import (
    CORE_FILE_NAME,
    METADATA_FILE_NAME,
    SEQUENCE_FILE_NAME,
)
from cryoflake.storage.store import preprocessed_artifact_member_reference
from cryoflake.storage.transports import (
    file_reference_identity,
    resolve_s3_settings,
)
from cryoflake.targets.schema import _COMPONENT_KEYS, _PROPOSAL_KEYS

PREPROCESSED_ARTIFACT_CONTRACT_VERSION = "S2610210"


def manifest_source_identity(
    manifest_rows: list[Mapping[str, str]],
    manifest_dir: Path,
) -> str:
    digest = hashlib.sha256()
    for row_index, row in enumerate(manifest_rows, start=1):
        for fieldname in (
            "map_path",
            "structure_path",
            "sequence_embedding_path",
        ):
            path_value = row.get(fieldname, "").strip()
            identity: dict[str, object] = {
                "row_index": row_index,
                "fieldname": fieldname,
                "path": path_value,
                "exists": False,
            }
            if path_value:
                path = Path(path_value)
                if not path.is_absolute():
                    path = manifest_dir / path
                identity["path"] = str(path.resolve())
                try:
                    stat_result = path.stat()
                except OSError:
                    pass
                else:
                    identity.update(
                        {
                            "exists": path.is_file(),
                            "size": stat_result.st_size,
                            "mtime_ns": stat_result.st_mtime_ns,
                            "ctime_ns": stat_result.st_ctime_ns,
                            "device": stat_result.st_dev,
                            "inode": stat_result.st_ino,
                        }
                    )
            digest.update(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            digest.update(b"\n")
    return digest.hexdigest()


def preprocess_checkpoint_contract(
    *,
    manifest_csv: Path,
    manifest_rows: list[Mapping[str, str]],
    manifest_dir: Path,
    artifact_output_dir: Path,
    artifact_s3_prefix: Optional[str],
    s3_endpoint_url: Optional[str],
    artifact_reuse_dirs: tuple[Path, ...],
    target_voxel_size: float,
    fix_shift: bool,
    preprocess_adapter: Callable,
    interface_contact_cutoff: float,
    require_sequence_embeddings: bool,
    sequence_embedding_loader: Callable,
    download_errors: Mapping[str, str],
    preprocess_checkpoint_token: Optional[str] = None,
) -> dict[str, object]:
    s3_settings = resolve_s3_settings(s3_endpoint_url)
    return {
        "artifact_contract_version": PREPROCESSED_ARTIFACT_CONTRACT_VERSION,
        "format_contracts": {
            "artifact": asdict(artifact_contract()),
            "targets": asdict(target_contract()),
        },
        "target_fields": {
            "full_atom_targets": sorted(
                ("schema_version", *FULL_ATOM_TARGET_KEYS, *FULL_ATOM_TOPOLOGY_KEYS)
            ),
            "extension_targets.component": sorted(_COMPONENT_KEYS),
            "extension_targets.component.proposal_targets": sorted(_PROPOSAL_KEYS),
        },
        "manifest_path": str(manifest_csv.resolve()),
        "manifest_sha256": json_sha256([dict(row) for row in manifest_rows]),
        "manifest_source_identity": manifest_source_identity(
            manifest_rows,
            manifest_dir,
        ),
        "artifact_output_dir": str(artifact_output_dir.resolve()),
        "artifact_s3_prefix": artifact_s3_prefix or "",
        "s3_endpoint_url": s3_settings.get("endpoint_url", ""),
        "s3_region": s3_settings.get("region_name", ""),
        "artifact_reuse_dirs": [str(path.resolve()) for path in artifact_reuse_dirs],
        "target_voxel_size": target_voxel_size,
        "fix_shift": bool(fix_shift),
        "preprocess_adapter": callable_identity(
            preprocess_adapter,
            state_token=preprocess_checkpoint_token,
        ),
        "interface_contact_cutoff": interface_contact_cutoff,
        "require_sequence_embeddings": bool(require_sequence_embeddings),
        "download_errors": dict(
            sorted((str(path), str(error)) for path, error in download_errors.items())
        ),
        "sequence_embedding_loader": callable_identity(
            sequence_embedding_loader,
            state_token=preprocess_checkpoint_token,
        ),
    }


def preprocessed_artifact_identity(
    artifact_reference: os.PathLike | str,
    *,
    s3_endpoint_url: Optional[str] = None,
) -> Optional[dict[str, object]]:
    members = {}
    for file_name in (
        METADATA_FILE_NAME,
        CORE_FILE_NAME,
        SEQUENCE_FILE_NAME,
    ):
        member_reference = preprocessed_artifact_member_reference(
            artifact_reference, file_name
        )
        members[file_name] = file_reference_identity(
            member_reference,
            s3_endpoint_url=s3_endpoint_url,
        )
    if members[METADATA_FILE_NAME] is None or members[CORE_FILE_NAME] is None:
        return None
    return {
        "kind": "canonical_artifact",
        "members": members,
    }
