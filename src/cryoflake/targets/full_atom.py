"""Portable full-atom target extraction and validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Optional

import numpy as np

from cryoflake import constants
from cryoflake.common import array_all_finite
from cryoflake.targets.common import _all_boolean_values

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
_ATOM_NAME = re.compile(r"[A-Z0-9][A-Z0-9'\"*+_-]*\Z")


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
    if dtype is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            value = value.astype(dtype, copy=False)
        if not array_all_finite(value):
            raise ValueError(f"{label}[{key!r}] must contain only finite values.")
    return value


def _mask_target_array(
    targets: Mapping,
    key: str,
    expected_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    value = _target_array(targets, key, expected_shape, None, label)
    if not _all_boolean_values(value):
        raise ValueError(f"{label}[{key!r}] must contain only boolean values.")
    return value.astype(np.bool_, copy=False)


def _atom_name_array(
    targets: Mapping,
    label: str,
) -> np.ndarray:
    expected_shape = (constants.FULL_ATOM_COUNT,)
    value = targets["atom_names"]
    if isinstance(value, np.ndarray) and value.dtype.kind not in {"U", "S", "O"}:
        raise ValueError(f"{label}['atom_names'] must contain only strings.")
    try:
        # Preserve input element types: NumPy's inferred string dtype would
        # silently turn a mixed string/integer sequence into valid-looking names.
        names = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label}['atom_names'] must be a string array with shape {expected_shape}."
        ) from exc
    if names.shape != expected_shape:
        raise ValueError(f"{label}['atom_names'] must have shape {expected_shape}.")
    normalized_names = []
    for name in names.flat:
        if not isinstance(name, (str, bytes)):
            raise ValueError(f"{label}['atom_names'] must contain only strings.")
        if isinstance(name, bytes):
            try:
                name = name.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{label}['atom_names'] must contain valid UTF-8 strings."
                ) from exc
        name = name.strip().upper()
        if not name:
            raise ValueError(
                f"{label}['atom_names'] requires nonempty names for all slots."
            )
        if not _ATOM_NAME.fullmatch(name):
            raise ValueError(f"{label}['atom_names'] contains an invalid atom name.")
        normalized_names.append(name)
    # Let NumPy determine the required Unicode width; fixed-width conversion
    # must not truncate scientific atom labels or manufacture duplicate names.
    if len(set(normalized_names)) != constants.FULL_ATOM_COUNT:
        raise ValueError(
            f"{label}['atom_names'] has duplicate atom names in its slot mapping."
        )
    return np.asarray(normalized_names, dtype=str)


def validate_full_atom_targets(
    targets,
    num_protein: Optional[int] = None,
    num_nucleotide: Optional[int] = None,
    label: str = "full_atom_targets",
) -> dict[str, object]:
    """Validate and normalize serialized full-atom supervision arrays."""

    if not isinstance(targets, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    if "prot_mask" in targets:
        raise ValueError(
            f"{label}['prot_mask'] is derived from aatype and must not be stored."
        )
    missing_keys = [
        key for key in constants.FULL_ATOM_TARGET_KEYS if key not in targets
    ]
    if missing_keys:
        raise ValueError(f"{label} is missing required target arrays.")
    schema_version = targets.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, (int, np.integer))
        or int(schema_version) != constants.FULL_ATOM_TARGET_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} has unsupported full-atom schema_version.")

    aatype = np.asarray(targets["aatype"])
    if aatype.ndim != 1 or aatype.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label}['aatype'] must be a one-dimensional integer array.")
    aatype = aatype.astype(np.int64, copy=False)
    if np.any(aatype < 0) or np.any(aatype >= constants.FULL_RESIDUE_TYPE_COUNT):
        raise ValueError(f"{label}['aatype'] contains an invalid residue type index.")

    num_nodes = len(aatype)
    normalized = {
        "schema_version": constants.FULL_ATOM_TARGET_SCHEMA_VERSION,
        "aatype": aatype,
    }
    prot_mask = _protein_mask_from_aatype(aatype)

    for key, trailing_shape in _FLOAT_TARGET_SHAPES.items():
        normalized[key] = _target_array(
            targets, key, (num_nodes, *trailing_shape), np.float32, label
        )
    for key, trailing_shape in _MASK_TARGET_SHAPES.items():
        normalized[key] = _mask_target_array(
            targets, key, (num_nodes, *trailing_shape), label
        )
    normalized["atom_names"] = _atom_name_array(targets, label)
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

    protein_count = int(np.count_nonzero(prot_mask))
    if num_protein is not None and protein_count != int(num_protein):
        raise ValueError(f"{label} protein count does not match CA coordinates.")
    if num_nucleotide is not None and num_nodes - protein_count != int(num_nucleotide):
        raise ValueError(f"{label} nucleotide count does not match P coordinates.")
    if not prot_mask[:protein_count].all():
        raise ValueError(f"{label} must store protein rows before nucleotide rows.")
    return normalized


def extract_full_atom_targets(target_source) -> dict[str, object]:
    """Extract residue targets in GNN node order and preserve their shared atom table."""

    try:
        aatype = np.asarray(target_source.aatype)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Target source aatype must be a one-dimensional integer array."
        ) from exc
    if aatype.ndim != 1 or aatype.dtype.kind not in {"i", "u"}:
        raise ValueError(
            "Target source aatype must be a one-dimensional integer array."
        )
    num_nodes = len(aatype)
    prot_mask = _protein_mask_from_aatype(aatype)
    node_order = np.concatenate(
        (np.flatnonzero(prot_mask), np.flatnonzero(~prot_mask)), axis=0
    )
    targets = {
        "schema_version": constants.FULL_ATOM_TARGET_SCHEMA_VERSION,
        "aatype": aatype[node_order],
    }
    for key in constants.FULL_ATOM_TARGET_KEYS[1:]:
        try:
            value = getattr(target_source, key)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Target source is missing {key} targets.") from exc
        if key == "atom_names":
            # The slot mapping is shared by every residue. Keep its element
            # types until validation and do not index it by residue order.
            targets[key] = value
            continue
        try:
            array = np.asarray(value)
            if array.ndim == 0 or len(array) != num_nodes:
                raise ValueError("unaligned residue rows")
            targets[key] = array[node_order]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Target source is missing aligned {key} targets."
            ) from exc
    for key in constants.FULL_ATOM_TOPOLOGY_KEYS:
        try:
            array = np.asarray(getattr(target_source, key))
            if array.ndim == 0 or len(array) != num_nodes:
                raise ValueError("unaligned residue rows")
            targets[key] = array[node_order]
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Target source is missing aligned {key} targets."
            ) from exc
    return validate_full_atom_targets(targets)


__all__ = [
    "extract_full_atom_targets",
    "validate_full_atom_targets",
]
