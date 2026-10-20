"""Strict validation for the versioned component-target extension payload."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

from cryoflake.constants import (
    COMPONENT_EXTENSION_SCHEMA_VERSION,
    AnchorKind,
    NodeKind,
    PolymerKind,
)
from cryoflake.targets.common import _string_array

_COMPONENT_KEYS = {
    "positions",
    "node_kind",
    "polymer_kind",
    "anchor_kind",
    "reference_local_positions",
    "component_id",
    "instance_id",
    "chain_id",
    "residue_index",
    "insertion_code",
    "source_chem_comp_type",
    "atom_positions",
    "atom_mask",
    "atom_names",
    "atom_elements",
    "b_factors",
    "occupancies",
    "interface_mask",
    "interface_group_id",
    "interface_contact_cutoff",
    "proposal_targets",
    "split_groups",
}
_PROPOSAL_KEYS = {"node_indices", "positions", "valid_mask"}
_SPLIT_GROUP_KEYS = {"component", "interface"}


def _numeric_array(value, shape, name: str, dtype) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in {"b", "f", "i", "u"}:
        raise ValueError(f"{name} must be a numeric array with shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(dtype, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integer_array(value, shape, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be an integer array with shape {shape}")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must be representable as int64")
    return array.astype(np.int64, copy=False)


def _boolean_array(value, shape, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in {"b", "f", "i", "u"}:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only boolean values")
    return array.astype(bool, copy=False)


def _validated_strings(value, length: int, name: str, *, nonempty: bool) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype.kind not in {"U", "S", "O"}:
        raise ValueError(f"{name} must be a string array with shape ({length},)")
    normalized = _string_array(array.tolist())
    if nonempty and any(not item for item in normalized.tolist()):
        raise ValueError(f"{name} must not contain empty values")
    return normalized


def _normalized_group_ids(value, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of group IDs")
    normalized = tuple(sorted({str(item).strip() for item in value}))
    if any(not item for item in normalized):
        raise ValueError(f"{name} must not contain empty group IDs")
    return normalized


def validate_component_extension_targets(extension_targets) -> dict:
    """Strictly validate and normalize a serialized component extension payload."""

    if not isinstance(extension_targets, Mapping):
        raise ValueError("extension_targets must be a mapping")
    if set(extension_targets) != {"schema_version", "component"}:
        raise ValueError("extension_targets must contain schema_version and component")
    schema_version = extension_targets["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, (int, np.integer))
        or int(schema_version) != COMPONENT_EXTENSION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported component extension schema_version")
    component = extension_targets["component"]
    if not isinstance(component, Mapping) or set(component) != _COMPONENT_KEYS:
        raise ValueError("component payload has missing or unexpected fields")

    positions_value = np.asarray(component["positions"])
    if positions_value.ndim != 2 or positions_value.shape[1:] != (3,):
        raise ValueError("component positions must have shape (N, 3)")
    num_nodes = len(positions_value)
    positions = _numeric_array(
        positions_value, (num_nodes, 3), "component positions", np.float32
    )
    node_kind = _integer_array(component["node_kind"], (num_nodes,), "node_kind")
    polymer_kind = _integer_array(
        component["polymer_kind"], (num_nodes,), "polymer_kind"
    )
    anchor_kind = _integer_array(component["anchor_kind"], (num_nodes,), "anchor_kind")
    enum_fields = (
        ("node_kind", node_kind, len(NodeKind)),
        ("polymer_kind", polymer_kind, len(PolymerKind)),
        ("anchor_kind", anchor_kind, len(AnchorKind)),
    )
    for name, values, upper_bound in enum_fields:
        if np.any(values < 0) or np.any(values >= upper_bound):
            raise ValueError(f"{name} contains an unsupported value")
    is_polymer = node_kind == int(NodeKind.POLYMER)
    if np.any(is_polymer & (polymer_kind == int(PolymerKind.NONE))):
        raise ValueError("polymer nodes require a polymer_kind")
    if np.any((~is_polymer) & (polymer_kind != int(PolymerKind.NONE))):
        raise ValueError("non-polymer node kinds require polymer_kind NONE")
    is_unknown = node_kind == int(NodeKind.UNKNOWN)
    is_unspecified = anchor_kind == int(AnchorKind.UNSPECIFIED)
    if np.any(is_unknown & ~is_unspecified):
        raise ValueError("UNKNOWN nodes require UNSPECIFIED anchors")
    if np.any(~is_unknown & is_unspecified):
        raise ValueError("non-UNKNOWN nodes cannot use UNSPECIFIED anchors")
    c_alpha_anchor = anchor_kind == int(AnchorKind.C_ALPHA)
    is_protein = is_polymer & (polymer_kind == int(PolymerKind.PROTEIN))
    if np.any(c_alpha_anchor & ~is_protein):
        raise ValueError("C_ALPHA anchors require POLYMER/PROTEIN nodes")
    phosphorus_anchor = anchor_kind == int(AnchorKind.PHOSPHORUS)
    is_nucleic_acid = is_polymer & np.isin(
        polymer_kind,
        (
            int(PolymerKind.RNA),
            int(PolymerKind.DNA),
            int(PolymerKind.NUCLEIC_ACID),
        ),
    )
    if np.any(phosphorus_anchor & ~is_nucleic_acid):
        raise ValueError("PHOSPHORUS anchors require nucleic-acid polymer nodes")
    atom_anchor = anchor_kind == int(AnchorKind.ATOM)
    is_single_atom_kind = np.isin(
        node_kind,
        (int(NodeKind.ION), int(NodeKind.WATER)),
    )
    if np.any(is_single_atom_kind & ~atom_anchor):
        raise ValueError("ION and WATER nodes require ATOM anchors")

    reference = _numeric_array(
        component["reference_local_positions"],
        (num_nodes, 3, 3),
        "reference_local_positions",
        np.float32,
    )
    if not np.allclose(reference[:, 1], 0.0, atol=1e-5):
        raise ValueError("reference_local_positions must be anchor-centred")
    if (
        not np.allclose(reference[:, 0, 1:], 0.0, atol=1e-5)
        or not np.allclose(reference[:, 2, 2], 0.0, atol=1e-5)
        or np.any(reference[:, 0, 0] > 1e-5)
        or np.any(reference[:, 2, 1] < -1e-5)
    ):
        raise ValueError("reference_local_positions must use the canonical frame")

    component_id = _validated_strings(
        component["component_id"], num_nodes, "component_id", nonempty=True
    )
    component_id = np.char.upper(np.char.strip(component_id))
    if np.any(component_id == ""):
        raise ValueError("component_id must not contain empty values")
    instance_id = _validated_strings(
        component["instance_id"], num_nodes, "instance_id", nonempty=True
    )
    if len(set(instance_id.tolist())) != num_nodes:
        raise ValueError("instance_id values must be unique")
    chain_id = _validated_strings(
        component["chain_id"], num_nodes, "chain_id", nonempty=False
    )
    residue_index = _integer_array(
        component["residue_index"],
        (num_nodes,),
        "residue_index",
    )
    insertion_code = _validated_strings(
        component["insertion_code"],
        num_nodes,
        "insertion_code",
        nonempty=False,
    )
    source_chem_comp_type = _validated_strings(
        component["source_chem_comp_type"],
        num_nodes,
        "source_chem_comp_type",
        nonempty=False,
    )
    source_chem_comp_type = np.char.upper(np.char.strip(source_chem_comp_type))

    atom_mask_value = np.asarray(component["atom_mask"])
    if atom_mask_value.ndim != 2 or atom_mask_value.shape[0] != num_nodes:
        raise ValueError("atom_mask must have shape (N, max_atoms)")
    max_atoms = atom_mask_value.shape[1]
    atom_mask = _boolean_array(atom_mask_value, (num_nodes, max_atoms), "atom_mask")
    atom_positions = _numeric_array(
        component["atom_positions"],
        (num_nodes, max_atoms, 3),
        "atom_positions",
        np.float32,
    )
    b_factors = _numeric_array(
        component["b_factors"],
        (num_nodes, max_atoms),
        "b_factors",
        np.float32,
    )
    occupancies = _numeric_array(
        component["occupancies"],
        (num_nodes, max_atoms),
        "occupancies",
        np.float32,
    )
    if np.any(occupancies < 0):
        raise ValueError("occupancies must be non-negative")
    atom_names = np.asarray(component["atom_names"])
    atom_elements = np.asarray(component["atom_elements"])
    if atom_names.shape != (num_nodes, max_atoms) or atom_names.dtype.kind not in {
        "U",
        "S",
        "O",
    }:
        raise ValueError("atom_names must be a padded string (N, max_atoms) array")
    if atom_elements.shape != (
        num_nodes,
        max_atoms,
    ) or atom_elements.dtype.kind not in {"U", "S", "O"}:
        raise ValueError("atom_elements must be a padded string (N, max_atoms) array")
    atom_names = np.char.upper(np.char.strip(atom_names.astype(np.str_)))
    atom_elements = np.char.upper(np.char.strip(atom_elements.astype(np.str_)))
    if num_nodes and np.any(atom_mask.sum(axis=1) == 0):
        raise ValueError("each component node must contain at least one atom")
    for row in range(num_nodes):
        atom_count = int(atom_mask[row].sum())
        if not atom_mask[row, :atom_count].all() or atom_mask[row, atom_count:].any():
            raise ValueError("atom_mask must use contiguous left padding")
    if np.any((atom_names == "") & atom_mask) or np.any(
        (atom_elements == "") & atom_mask
    ):
        raise ValueError("present atoms require names and elements")
    if (
        np.any((atom_names != "") & ~atom_mask)
        or np.any((atom_elements != "") & ~atom_mask)
        or not np.allclose(atom_positions[~atom_mask], 0.0)
        or not np.allclose(b_factors[~atom_mask], 0.0)
        or not np.allclose(occupancies[~atom_mask], 0.0)
    ):
        raise ValueError("padded atom fields must be empty or zero")

    for index in range(num_nodes):
        present_positions = atom_positions[index, atom_mask[index]]
        present_names = atom_names[index, atom_mask[index]]
        present_elements = atom_elements[index, atom_mask[index]]
        anchor = positions[index]
        anchor_type = anchor_kind[index]
        if node_kind[index] == int(NodeKind.ION):
            heavy_count = int((~np.isin(present_elements, ("H", "D"))).sum())
            if heavy_count != 1:
                raise ValueError("ion nodes must contain one heavy atom")
        if node_kind[index] == int(NodeKind.WATER):
            heavy_elements = present_elements[~np.isin(present_elements, ("H", "D"))]
            if heavy_elements.tolist() != ["O"]:
                raise ValueError("water nodes must contain one oxygen heavy atom")

        if anchor_type == int(AnchorKind.C_ALPHA):
            candidates = present_positions[present_names == "CA"]
        elif anchor_type == int(AnchorKind.PHOSPHORUS):
            candidates = present_positions[present_names == "P"]
        elif anchor_type == int(AnchorKind.ATOM):
            candidates = present_positions
        elif anchor_type == int(AnchorKind.CENTROID):
            heavy = present_positions[~np.isin(present_elements, ("H", "D"))]
            candidates = heavy if len(heavy) else present_positions
            if not np.allclose(anchor, candidates.mean(axis=0), atol=1e-4):
                raise ValueError("centroid anchors must match atom positions")
            continue
        else:
            continue
        if not len(candidates) or not np.any(
            np.all(np.isclose(candidates, anchor, atol=1e-4), axis=-1)
        ):
            raise ValueError("atom anchors must match a component atom")

    interface_mask = _boolean_array(
        component["interface_mask"], (num_nodes,), "interface_mask"
    )
    interface_group_id = _validated_strings(
        component["interface_group_id"],
        num_nodes,
        "interface_group_id",
        nonempty=False,
    )
    if any(
        bool(mask) != bool(group_id)
        for mask, group_id in zip(interface_mask, interface_group_id)
    ):
        raise ValueError(
            "interface_group_id must be present exactly for interface nodes"
        )
    for index in np.flatnonzero(interface_mask):
        expected_pattern = rf"{re.escape(str(component_id[index]))}\|[0-9a-f]{{16}}"
        if re.fullmatch(expected_pattern, str(interface_group_id[index])) is None:
            raise ValueError(
                "interface_group_id must use "
                "<component_id>|<16 lowercase hex characters>"
            )
    cutoff_value = np.asarray(component["interface_contact_cutoff"])
    if cutoff_value.shape != () or cutoff_value.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("interface_contact_cutoff must be a numeric scalar")
    interface_contact_cutoff = float(cutoff_value)
    if not np.isfinite(interface_contact_cutoff) or interface_contact_cutoff <= 0:
        raise ValueError("interface_contact_cutoff must be positive and finite")

    proposal = component["proposal_targets"]
    if not isinstance(proposal, Mapping) or set(proposal) != _PROPOSAL_KEYS:
        raise ValueError("proposal_targets has missing or unexpected fields")
    proposal_indices_value = np.asarray(proposal["node_indices"])
    if proposal_indices_value.ndim != 1:
        raise ValueError("proposal node_indices must be one-dimensional")
    num_proposals = len(proposal_indices_value)
    proposal_indices = _integer_array(
        proposal_indices_value,
        (num_proposals,),
        "proposal node_indices",
    )
    if (
        np.any(proposal_indices < 0)
        or np.any(proposal_indices >= num_nodes)
        or len(set(proposal_indices.tolist())) != num_proposals
    ):
        raise ValueError("proposal node_indices must be unique and in range")
    proposal_positions = _numeric_array(
        proposal["positions"],
        (num_proposals, 3),
        "proposal positions",
        np.float32,
    )
    if not np.array_equal(
        proposal_positions,
        positions[proposal_indices],
    ):
        raise ValueError("proposal positions must match component positions")
    proposal_valid_mask = _boolean_array(
        proposal["valid_mask"],
        (num_proposals,),
        "proposal valid_mask",
    )

    split_groups = component["split_groups"]
    if not isinstance(split_groups, Mapping) or set(split_groups) != _SPLIT_GROUP_KEYS:
        raise ValueError("split_groups has missing or unexpected fields")
    component_groups = _normalized_group_ids(
        split_groups["component"],
        "component split groups",
    )
    interface_groups = _normalized_group_ids(
        split_groups["interface"],
        "interface split groups",
    )
    expected_component_groups = tuple(
        sorted(
            {
                str(component_id[index])
                for index in range(num_nodes)
                if node_kind[index] != int(NodeKind.WATER)
            }
        )
    )
    expected_interface_groups = tuple(
        sorted(
            {
                str(interface_group_id[index])
                for index in range(num_nodes)
                if interface_mask[index]
            }
        )
    )
    if component_groups != expected_component_groups:
        raise ValueError("component split groups do not match component IDs")
    if interface_groups != expected_interface_groups:
        raise ValueError("interface split groups do not match interface IDs")

    normalized_component = {
        "positions": positions,
        "node_kind": node_kind,
        "polymer_kind": polymer_kind,
        "anchor_kind": anchor_kind,
        "reference_local_positions": reference,
        "component_id": component_id,
        "instance_id": instance_id,
        "chain_id": chain_id,
        "residue_index": residue_index,
        "insertion_code": insertion_code,
        "source_chem_comp_type": source_chem_comp_type,
        "atom_positions": atom_positions,
        "atom_mask": atom_mask,
        "atom_names": atom_names,
        "atom_elements": atom_elements,
        "b_factors": b_factors,
        "occupancies": occupancies,
        "interface_mask": interface_mask,
        "interface_group_id": interface_group_id,
        "interface_contact_cutoff": interface_contact_cutoff,
        "proposal_targets": {
            "node_indices": proposal_indices,
            "positions": proposal_positions,
            "valid_mask": proposal_valid_mask,
        },
        "split_groups": {
            "component": component_groups,
            "interface": interface_groups,
        },
    }
    return {
        "schema_version": COMPONENT_EXTENSION_SCHEMA_VERSION,
        "component": normalized_component,
    }
