"""Portable full-atom target extraction and validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np

from cryoflake import constants

_FLOAT_TARGET_SHAPES = {
    "atom_positions": (constants.FULL_ATOM_COUNT, 3),
    "b_factors": (constants.FULL_ATOM_COUNT,),
    "rigidgroups_gt_frames": (constants.RIGID_GROUP_COUNT, 3, 4),
    "rigidgroups_alt_gt_frames": (constants.RIGID_GROUP_COUNT, 3, 4),
    "torsion_angles_sin_cos": (constants.TORSION_COUNT, 2),
    "alt_torsion_angles_sin_cos": (constants.TORSION_COUNT, 2),
}
_MASK_TARGET_SHAPES = {
    "atom_mask": (constants.FULL_ATOM_COUNT,),
    "rigidgroups_gt_exists": (constants.RIGID_GROUP_COUNT,),
    "rigidgroups_group_exists": (constants.RIGID_GROUP_COUNT,),
    "rigidgroups_group_is_ambiguous": (constants.RIGID_GROUP_COUNT,),
    "torsion_angles_mask": (constants.TORSION_COUNT,),
}


def _protein_mask_from_aatype(aatype: np.ndarray) -> np.ndarray:
    return (aatype < 20) | (aatype == constants.UNKNOWN_PROTEIN_INDEX)


def _target_array(
    targets: Mapping,
    key: str,
    expected_shape: tuple[int, ...],
    dtype,
    label: str,
) -> np.ndarray:
    try:
        value = np.asarray(targets[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}[{key!r}] must be a numeric array.") from exc
    if value.shape != expected_shape or value.dtype.kind not in {"b", "f", "i", "u"}:
        raise ValueError(
            f"{label}[{key!r}] must have shape {expected_shape} and be numeric."
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{label}[{key!r}] must contain only finite values.")
    if dtype is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            value = value.astype(dtype, copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"{label}[{key!r}] must contain only finite values.")
    return value


def _mask_target_array(
    targets: Mapping,
    key: str,
    expected_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    value = _target_array(targets, key, expected_shape, None, label)
    if not np.isin(value, (0.0, 1.0)).all():
        raise ValueError(f"{label}[{key!r}] must contain only boolean values.")
    return value.astype(np.bool_, copy=False)


def validate_full_atom_targets(
    targets,
    num_protein: Optional[int] = None,
    num_nucleotide: Optional[int] = None,
    label: str = "full_atom_targets",
) -> dict[str, np.ndarray]:
    """Validate and normalize serialized full-atom supervision arrays."""

    if not isinstance(targets, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    missing_keys = [
        key for key in constants.FULL_ATOM_TARGET_KEYS if key not in targets
    ]
    if missing_keys:
        raise ValueError(f"{label} is missing required target arrays.")

    aatype = np.asarray(targets["aatype"])
    if aatype.ndim != 1 or aatype.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label}['aatype'] must be a one-dimensional integer array.")
    aatype = aatype.astype(np.int64, copy=False)
    if np.any(aatype < 0) or np.any(aatype >= constants.FULL_RESIDUE_TYPE_COUNT):
        raise ValueError(f"{label}['aatype'] contains an invalid residue type index.")

    num_nodes = len(aatype)
    normalized = {"aatype": aatype}
    prot_mask = _mask_target_array(targets, "prot_mask", (num_nodes,), label)
    expected_prot_mask = _protein_mask_from_aatype(aatype)
    if not np.array_equal(prot_mask, expected_prot_mask):
        raise ValueError(f"{label}['prot_mask'] does not match aatype.")
    normalized["prot_mask"] = prot_mask

    for key, trailing_shape in _FLOAT_TARGET_SHAPES.items():
        normalized[key] = _target_array(
            targets, key, (num_nodes, *trailing_shape), np.float32, label
        )
    for key, trailing_shape in _MASK_TARGET_SHAPES.items():
        normalized[key] = _mask_target_array(
            targets, key, (num_nodes, *trailing_shape), label
        )
    missing_topology = [
        key for key in constants.FULL_ATOM_TOPOLOGY_KEYS if key not in targets
    ]
    if missing_topology:
        raise ValueError(f"{label} must contain chain_index and residue_index.")
    for key in constants.FULL_ATOM_TOPOLOGY_KEYS:
        value = np.asarray(targets[key])
        if value.shape != (num_nodes,) or value.dtype.kind not in {"i", "u"}:
            raise ValueError(
                f"{label}[{key!r}] must be a one-dimensional integer array."
            )
        if value.dtype.kind == "u" and np.any(value > np.iinfo(np.int64).max):
            raise ValueError(f"{label}[{key!r}] must be representable as int64.")
        normalized[key] = value.astype(np.int64, copy=False)
    if np.any(normalized["chain_index"] < 0):
        raise ValueError(f"{label}['chain_index'] must be non-negative.")

    if num_protein is not None and int(prot_mask.sum()) != int(num_protein):
        raise ValueError(f"{label} protein count does not match CA coordinates.")
    if num_nucleotide is not None and int((~prot_mask).sum()) != int(num_nucleotide):
        raise ValueError(f"{label} nucleotide count does not match P coordinates.")
    expected_order = np.arange(num_nodes) < int(prot_mask.sum())
    if not np.array_equal(prot_mask, expected_order):
        raise ValueError(f"{label} must store protein rows before nucleotide rows.")
    return normalized


def extract_full_atom_targets(target_source) -> dict[str, np.ndarray]:
    """Extract target arrays in the GNN node order: protein, then nucleotide."""

    aatype = np.asarray(target_source.aatype)
    if aatype.ndim != 1:
        raise ValueError("Target source aatype must be a one-dimensional array.")
    prot_mask = _protein_mask_from_aatype(aatype)
    node_order = np.concatenate(
        (np.flatnonzero(prot_mask), np.flatnonzero(~prot_mask)), axis=0
    )
    targets = {
        "aatype": aatype[node_order],
        "prot_mask": prot_mask[node_order],
    }
    for key in constants.FULL_ATOM_TARGET_KEYS[2:]:
        try:
            targets[key] = np.asarray(getattr(target_source, key))[node_order]
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Target source is missing aligned {key} targets."
            ) from exc
    for key in constants.FULL_ATOM_TOPOLOGY_KEYS:
        try:
            targets[key] = np.asarray(getattr(target_source, key))[node_order]
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Target source is missing aligned {key} targets."
            ) from exc
    return validate_full_atom_targets(targets)


__all__ = [
    "extract_full_atom_targets",
    "validate_full_atom_targets",
]
