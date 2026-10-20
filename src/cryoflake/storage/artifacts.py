"""Model-independent validation for canonical preprocessed artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from cryoflake.metadata import validate_coordinates
from cryoflake.preprocess.reuse import (
    _preprocessed_artifact_is_reusable,
)
from cryoflake.sequence.validation import validate_sequence_embedding_payload
from cryoflake.storage.store import load_preprocessed_artifact


@dataclass(frozen=True)
class ArtifactValidationResult:
    artifact_reference: str
    grid_shape: tuple[int, int, int]
    protein_nodes: int
    nucleotide_nodes: int
    component_nodes: int
    sequence_length: int | None
    sequence_feature_dim: int | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def validate_preprocessed_artifact(
    artifact_reference: os.PathLike | str,
    *,
    require_sequence_embeddings: bool = False,
    s3_endpoint_url: str | None = None,
) -> ArtifactValidationResult:
    """Validate storage integrity and the model-independent data contract."""

    payload = load_preprocessed_artifact(
        artifact_reference,
        include_sequence_embeddings=True,
        s3_endpoint_url=s3_endpoint_url,
    )
    if not _preprocessed_artifact_is_reusable(
        payload,
        require_sequence_embeddings=require_sequence_embeddings,
    ):
        raise ValueError(
            f"Artifact does not satisfy the canonical preprocessing contract: "
            f"{artifact_reference!s}"
        )

    grid = payload["density_grid"]
    coordinates = payload["backbone_coordinates"]
    protein_coordinates = validate_coordinates(coordinates["CA"], allow_nan=True)
    nucleotide_coordinates = validate_coordinates(coordinates["P"], allow_nan=True)
    component = payload["extension_targets"]["component"]

    sequence_length = None
    sequence_feature_dim = None
    if "sequence" in payload or "sequence_mask" in payload:
        sequence_payload = validate_sequence_embedding_payload(payload)
        sequence_length, sequence_feature_dim = map(
            int, sequence_payload["sequence"].shape
        )

    return ArtifactValidationResult(
        artifact_reference=str(artifact_reference),
        grid_shape=tuple(int(size) for size in grid.shape),
        protein_nodes=int(len(protein_coordinates)),
        nucleotide_nodes=int(len(nucleotide_coordinates)),
        component_nodes=int(len(component["positions"])),
        sequence_length=sequence_length,
        sequence_feature_dim=sequence_feature_dim,
    )


def validate_artifact_list(
    artifact_list_path: os.PathLike,
    *,
    require_sequence_embeddings: bool = False,
    s3_endpoint_url: str | None = None,
) -> list[ArtifactValidationResult]:
    from cryoflake.manifests.manifest import read_artifact_list

    return [
        validate_preprocessed_artifact(
            artifact_reference,
            require_sequence_embeddings=require_sequence_embeddings,
            s3_endpoint_url=s3_endpoint_url,
        )
        for artifact_reference in read_artifact_list(Path(artifact_list_path))
    ]
