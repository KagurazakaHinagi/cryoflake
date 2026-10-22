"""Model-independent validation for canonical preprocessed artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from cryoflake.preprocess.reuse import (
    _validate_training_artifact_mapping,
)
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
    try:
        validated = _validate_training_artifact_mapping(
            payload,
            require_sequence_embeddings=require_sequence_embeddings,
        )
    except (ImportError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Artifact does not satisfy the canonical preprocessing contract: "
            f"{artifact_reference!s}"
        ) from exc

    sequence_length = None
    sequence_feature_dim = None
    if validated.sequence_shape is not None:
        sequence_length, sequence_feature_dim = validated.sequence_shape

    return ArtifactValidationResult(
        artifact_reference=str(artifact_reference),
        grid_shape=validated.grid_shape,
        protein_nodes=validated.protein_nodes,
        nucleotide_nodes=validated.nucleotide_nodes,
        component_nodes=len(validated.component_targets["positions"]),
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
