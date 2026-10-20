import os
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from cryoflake.sequence.validation import (
    validate_sequence_embedding_payload,
)
from cryoflake.storage.store import (
    replace_preprocessed_artifact_sequence,
)


def load_sequence_embedding_file(
    sequence_embedding_path: os.PathLike,
) -> dict[str, np.ndarray]:
    """Load a numeric NPY array or NPZ input with the ``sequence`` field.

    NPZ inputs may also contain ``sequence_mask`` and unified sequence metadata.
    Object arrays are rejected without deserialization.
    """
    sequence_embedding_path = Path(sequence_embedding_path)
    suffix = sequence_embedding_path.suffix.lower()

    if suffix == ".npy":
        payload = {"sequence": np.load(sequence_embedding_path, allow_pickle=False)}
    elif suffix == ".npz":
        with np.load(sequence_embedding_path, allow_pickle=False) as npz_file:
            payload = {key: npz_file[key] for key in npz_file.files}
    else:
        raise ValueError(
            "Unsupported sequence embedding file type: "
            f"{sequence_embedding_path.suffix}. Expected .npy or .npz."
        )

    return validate_sequence_embedding_payload(payload)


def attach_sequence_embeddings_to_preprocessed_artifact(
    artifact_reference: os.PathLike | str,
    sequence_embedding_path: os.PathLike,
    sequence_embedding_loader: Callable[[os.PathLike], Mapping[str, np.ndarray]] = (
        load_sequence_embedding_file
    ),
    s3_endpoint_url: str | None = None,
) -> None:
    """Load an embedding sidecar and replace sequence fields in an artifact."""
    sequence_payload = dict(sequence_embedding_loader(sequence_embedding_path))
    replace_preprocessed_artifact_sequence(
        artifact_reference,
        sequence_payload,
        s3_endpoint_url=s3_endpoint_url,
    )
