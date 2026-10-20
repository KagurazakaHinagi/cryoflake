from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from cryoflake.common import array_all_finite


def validate_sequence_embedding_array(
    sequence,
    label: str = "Sequence embeddings",
) -> np.ndarray:
    sequence = np.asarray(sequence)
    if sequence.ndim != 2 or sequence.shape[0] == 0 or sequence.shape[1] == 0:
        raise ValueError(f"{label} must have shape (sequence_length, feature_dim).")
    if sequence.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{label} must be numeric.")
    sequence = sequence.astype(np.float32, copy=False)
    if not array_all_finite(sequence):
        raise ValueError(f"{label} must contain only finite values.")
    return sequence


def validate_sequence_mask(
    sequence_mask,
    sequence_length: int,
    label: str = "sequence_mask",
) -> np.ndarray:
    mask = np.asarray(sequence_mask).reshape(-1)
    if len(mask) != sequence_length:
        raise ValueError(f"{label} length must match sequence embedding length.")

    if mask.dtype.kind == "b":
        mask = mask.astype(bool, copy=False)
    elif mask.dtype.kind in {"f", "i", "u"}:
        if not np.isfinite(mask).all():
            raise ValueError(f"{label} must contain only finite values.")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{label} must be boolean or contain only 0/1 values.")
        mask = mask.astype(bool)
    else:
        raise ValueError(f"{label} must be boolean or contain only 0/1 values.")

    if not np.any(mask):
        raise ValueError(f"{label} must contain at least one true value.")
    return mask


def _coerce_scalar(value):
    value = np.asarray(value)
    if value.shape != ():
        raise ValueError("Expected a scalar value.")
    return value.item()


def _coerce_non_negative_int(value, label: str) -> int:
    value = _coerce_scalar(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be a scalar integer.")
    parsed_value = int(value)
    if parsed_value < 0:
        raise ValueError(f"{label} must be non-negative.")
    return parsed_value


def _unified_sequence_length(unified_seq) -> int:
    unified_seq = _coerce_scalar(unified_seq)
    if isinstance(unified_seq, bytes):
        unified_seq = unified_seq.decode()
    if not isinstance(unified_seq, str):
        raise ValueError("unified_seq must be a string.")
    return sum(len(sequence) for sequence in unified_seq.split("|||"))


def validate_sequence_embedding_metadata(
    payload: Mapping[str, object],
    sequence_length: int,
    label: str = "Sequence embeddings",
) -> None:
    """Validate optional sequence metadata against the embedding length."""
    if not isinstance(payload, Mapping):
        raise TypeError("Sequence embedding metadata must be a mapping.")

    unified_seq_len_value = payload.get("unified_seq_len")
    if unified_seq_len_value is not None:
        unified_seq_len = _coerce_non_negative_int(
            unified_seq_len_value, f"{label} unified_seq_len"
        )
        if unified_seq_len != sequence_length:
            raise ValueError(f"{label} length must match unified_seq_len.")

    unified_seq_value = payload.get("unified_seq")
    if unified_seq_value is not None:
        unified_seq_len = _unified_sequence_length(unified_seq_value)
        if unified_seq_len != sequence_length:
            raise ValueError(f"{label} length must match unified_seq.")


def validate_sequence_embedding_payload(
    payload: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Normalize canonical embeddings and their optional mask and metadata.

    The returned payload contains float32 ``sequence`` and boolean
    ``sequence_mask`` arrays. A missing mask marks every residue as valid.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("Sequence embedding payload must be a mapping.")
    if "sequence" not in payload:
        raise ValueError("Sequence embedding payload must contain sequence.")
    sequence = validate_sequence_embedding_array(payload["sequence"])
    validate_sequence_embedding_metadata(payload, sequence.shape[0])
    sequence_mask = payload.get("sequence_mask")
    if sequence_mask is None:
        sequence_mask = np.ones(sequence.shape[0], dtype=bool)
    else:
        sequence_mask = validate_sequence_mask(sequence_mask, sequence.shape[0])
    return {"sequence": sequence, "sequence_mask": sequence_mask}
