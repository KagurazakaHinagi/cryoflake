"""Models support for manifest preprocessing."""

from __future__ import annotations

import os
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

from cryoflake.manifests.splits import (
    GroupedSplitRecord,
)
from cryoflake.preprocess.memory_guard import (
    PreprocessMemoryAdmission,
)
from cryoflake.storage.upload_cache import BatchedArtifactUploadCache


@dataclass(frozen=True)
class _PreprocessManifestRowTask:
    row_index: int
    row: Mapping[str, str]
    manifest_dir: Path
    artifact_output_dir: Path
    artifact_s3_prefix: Optional[str]
    s3_endpoint_url: Optional[str]
    artifact_reuse_dirs: tuple[Path, ...]
    target_voxel_size: float
    force_preprocess: bool
    skip_missing: bool
    fix_shift: bool
    preprocess_adapter: Callable[..., None]
    interface_contact_cutoff: float
    require_sequence_embeddings: bool
    sequence_embedding_loader: Callable[[os.PathLike], Mapping[str, np.ndarray]]
    continue_on_preprocess_error: bool
    download_errors: Mapping[str, str]
    split_seed: int
    upload_cache: Optional[BatchedArtifactUploadCache]
    memory_admission: PreprocessMemoryAdmission


@dataclass(frozen=True)
class _PreprocessManifestRowResult:
    row_index: int
    artifact_reference: Optional[Path | str]
    summary_row: dict[str, str]
    warning_message: Optional[str] = None
    grouped_split_record: Optional[GroupedSplitRecord] = None
    preprocess_succeeded: bool = False
    upload_future: Optional[Future[Optional[str]]] = None
    cached_artifact_dir: Optional[Path] = None
    success_status: str = ""
    artifact_voxel_size: str = ""
    grid_shape: str = ""
    entry_label: str = ""
    artifact_identity: Optional[dict[str, object]] = None
