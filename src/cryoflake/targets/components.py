"""Preprocessing targets for molecular components outside canonical CA/P supervision.

Biotite handles mmCIF parsing; shared geometry and schema validation live in
focused modules, using the centralized chemistry and target format constants.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from cryoflake import constants
from cryoflake.targets import schema
from cryoflake.targets.common import (
    _anchor_and_reference,
    _heavy_atoms,
    _Node,
    _string_array,
)
from cryoflake.targets.parsers import _read_component_nodes

_CONTACT_PAIR_BLOCK_SIZE = 131_072


def _contacting_polymer_atoms(
    component_positions: np.ndarray,
    polymer_positions: np.ndarray,
    cutoff_square: float,
) -> np.ndarray:
    """Match atoms with bounded pairwise workspace and unchanged arithmetic."""
    contacts = np.zeros(len(polymer_positions), dtype=bool)
    component_block_size = min(len(component_positions), 128, _CONTACT_PAIR_BLOCK_SIZE)
    if not component_block_size:
        return contacts
    polymer_block_size = max(1, _CONTACT_PAIR_BLOCK_SIZE // component_block_size)
    for polymer_start in range(0, len(polymer_positions), polymer_block_size):
        polymer_stop = polymer_start + polymer_block_size
        polymer_block = polymer_positions[polymer_start:polymer_stop]
        block_contacts = contacts[polymer_start:polymer_stop]
        for component_start in range(0, len(component_positions), component_block_size):
            component_block = component_positions[
                component_start : component_start + component_block_size
            ]
            distance_square = component_block[:, None, 0] - polymer_block[None, :, 0]
            np.square(distance_square, out=distance_square)
            axis_square = np.empty_like(distance_square)
            # Match sum(delta * delta, axis=-1)'s x/y/z reduction order for
            # float32 coordinates, using two pair planes instead of a 3D delta.
            for axis in (1, 2):
                np.subtract(
                    component_block[:, None, axis],
                    polymer_block[None, :, axis],
                    out=axis_square,
                )
                np.square(axis_square, out=axis_square)
                distance_square += axis_square
            block_contacts |= np.any(distance_square <= cutoff_square, axis=0)
            if block_contacts.all():
                break
    return contacts


def _normalize_polymer_context(
    polymer_atom_positions,
    polymer_atom_node_indices,
    polymer_chain_indices,
    polymer_residue_indices,
    polymer_aatype,
):
    topology_values = (
        polymer_chain_indices,
        polymer_residue_indices,
        polymer_aatype,
    )
    if polymer_atom_positions is None:
        if polymer_atom_node_indices is not None or any(
            value is not None for value in topology_values
        ):
            raise ValueError("polymer topology requires polymer_atom_positions")
        return None

    positions = np.asarray(polymer_atom_positions)
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or positions.dtype.kind not in {"f", "i", "u"}
        or not np.isfinite(positions).all()
    ):
        raise ValueError("polymer_atom_positions must be a finite numeric (A, 3) array")
    positions = positions.astype(np.float32, copy=False)
    if polymer_atom_node_indices is None:
        if len(positions) == 0 and not any(
            value is not None for value in topology_values
        ):
            return positions, np.zeros(0, dtype=np.int64), ()
        raise ValueError(
            "polymer_atom_node_indices and polymer topology are required "
            "with polymer atoms"
        )
    if any(value is None for value in topology_values):
        raise ValueError(
            "polymer chain, residue, and aatype arrays are required with polymer atoms"
        )

    node_indices = np.asarray(polymer_atom_node_indices)
    if (
        node_indices.shape != (len(positions),)
        or node_indices.dtype.kind not in {"i", "u"}
        or np.any(node_indices < 0)
    ):
        raise ValueError("polymer_atom_node_indices must be non-negative integer (A,)")
    node_indices = node_indices.astype(np.int64, copy=False)
    minimum_nodes = int(node_indices.max()) + 1 if len(node_indices) else 0
    topology = []
    for name, value in zip(
        ("chain", "residue", "aatype"),
        topology_values,
    ):
        array = np.asarray(value)
        if (
            array.ndim != 1
            or len(array) < minimum_nodes
            or array.dtype.kind not in {"i", "u"}
        ):
            raise ValueError(
                f"polymer_{name}_indices must cover every polymer node"
                if name != "aatype"
                else "polymer_aatype must cover every polymer node"
            )
        topology.append(array.astype(np.int64, copy=False))
    if len({len(array) for array in topology}) != 1:
        raise ValueError("polymer topology arrays must have matching lengths")
    return positions, node_indices, tuple(topology)


def _interface_metadata(
    nodes: Sequence[_Node],
    polymer_context,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    interface_mask = np.zeros(len(nodes), dtype=bool)
    signatures = [""] * len(nodes)
    if polymer_context is None or len(polymer_context[0]) == 0:
        return interface_mask, _string_array(signatures)

    polymer_positions, atom_node_indices, topology = polymer_context
    cutoff_square = float(cutoff) ** 2
    for node_index, node in enumerate(nodes):
        component_positions = np.stack(
            [atom.position for atom in _heavy_atoms(node.atoms)], axis=0
        )
        contacting_atoms = _contacting_polymer_atoms(
            component_positions, polymer_positions, cutoff_square
        )
        if not np.any(contacting_atoms):
            continue
        interface_mask[node_index] = True
        contacting_nodes = np.unique(atom_node_indices[contacting_atoms])
        fingerprint_rows = sorted(
            {
                (
                    int(topology[0][polymer_node]),
                    int(topology[1][polymer_node]),
                    int(topology[2][polymer_node]),
                )
                for polymer_node in contacting_nodes
            }
        )
        digest = hashlib.sha256(repr(fingerprint_rows).encode()).hexdigest()[:16]
        signatures[node_index] = f"{node.component_id}|{digest}"
    return interface_mask, _string_array(signatures)


def _grid_valid_mask(
    positions: np.ndarray,
    global_origin,
    voxel_size,
    grid_shape,
) -> np.ndarray:
    origin = np.asarray(global_origin)
    shape = np.asarray(grid_shape)
    if (
        origin.shape != (3,)
        or origin.dtype.kind not in {"f", "i", "u"}
        or not np.isfinite(origin).all()
    ):
        raise ValueError("global_origin must be a finite numeric 3-vector")
    if shape.shape != (3,) or shape.dtype.kind not in {"i", "u"} or np.any(shape <= 0):
        raise ValueError("grid_shape must contain three positive integers")
    try:
        voxel_size = float(voxel_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("voxel_size must be positive and finite") from exc
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive and finite")
    voxel_positions = (positions - origin.astype(np.float32, copy=False)) / voxel_size
    shape_xyz = shape[::-1].astype(np.float32, copy=False)
    return np.logical_and(
        voxel_positions >= 0,
        voxel_positions < shape_xyz,
    ).all(axis=-1)


def build_component_extension_targets(
    structure_path,
    *,
    global_origin,
    voxel_size,
    grid_shape,
    polymer_atom_positions=None,
    polymer_atom_node_indices=None,
    polymer_chain_indices=None,
    polymer_residue_indices=None,
    polymer_aatype=None,
    interface_contact_cutoff=5.0,
) -> dict:
    """Extract component targets without changing canonical CA/P rows."""

    try:
        interface_contact_cutoff = float(interface_contact_cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "interface_contact_cutoff must be positive and finite"
        ) from exc
    if not np.isfinite(interface_contact_cutoff) or interface_contact_cutoff <= 0:
        raise ValueError("interface_contact_cutoff must be positive and finite")

    nodes = _read_component_nodes(structure_path)
    num_nodes = len(nodes)
    max_atoms = max((len(node.atoms) for node in nodes), default=0)
    positions = np.zeros((num_nodes, 3), dtype=np.float32)
    node_kind = np.zeros(num_nodes, dtype=np.int64)
    polymer_kind = np.zeros(num_nodes, dtype=np.int64)
    anchor_kind = np.zeros(num_nodes, dtype=np.int64)
    reference = np.zeros((num_nodes, 3, 3), dtype=np.float32)
    atom_positions = np.zeros((num_nodes, max_atoms, 3), dtype=np.float32)
    atom_mask = np.zeros((num_nodes, max_atoms), dtype=bool)
    atom_name_width = max(
        (len(atom.name) for node in nodes for atom in node.atoms),
        default=1,
    )
    element_width = max(
        (len(atom.element) for node in nodes for atom in node.atoms),
        default=1,
    )
    atom_names = np.full(
        (num_nodes, max_atoms),
        "",
        dtype=f"<U{atom_name_width}",
    )
    atom_elements = np.full(
        (num_nodes, max_atoms),
        "",
        dtype=f"<U{element_width}",
    )
    b_factors = np.zeros((num_nodes, max_atoms), dtype=np.float32)
    occupancies = np.zeros((num_nodes, max_atoms), dtype=np.float32)

    for node_index, node in enumerate(nodes):
        positions[node_index], anchor_kind[node_index], reference[node_index] = (
            _anchor_and_reference(node)
        )
        node_kind[node_index] = node.node_kind
        polymer_kind[node_index] = node.polymer_kind
        for atom_index, atom in enumerate(node.atoms):
            atom_positions[node_index, atom_index] = atom.position
            atom_mask[node_index, atom_index] = True
            atom_names[node_index, atom_index] = atom.name
            atom_elements[node_index, atom_index] = atom.element
            b_factors[node_index, atom_index] = atom.b_factor
            occupancies[node_index, atom_index] = atom.occupancy

    polymer_context = _normalize_polymer_context(
        polymer_atom_positions,
        polymer_atom_node_indices,
        polymer_chain_indices,
        polymer_residue_indices,
        polymer_aatype,
    )
    interface_mask, interface_group_id = _interface_metadata(
        nodes,
        polymer_context,
        interface_contact_cutoff,
    )
    proposal_valid_mask = _grid_valid_mask(
        positions,
        global_origin,
        voxel_size,
        grid_shape,
    )
    component_groups = tuple(
        sorted(
            {
                node.component_id
                for node in nodes
                if node.node_kind != int(constants.NodeKind.WATER)
            }
        )
    )
    interface_groups = tuple(
        sorted(
            {
                str(group_id)
                for group_id, is_interface in zip(
                    interface_group_id,
                    interface_mask,
                )
                if is_interface
            }
        )
    )
    extension_targets = {
        "schema_version": constants.COMPONENT_EXTENSION_SCHEMA_VERSION,
        "component": {
            "positions": positions,
            "node_kind": node_kind,
            "polymer_kind": polymer_kind,
            "anchor_kind": anchor_kind,
            "reference_local_positions": reference,
            "component_id": _string_array([node.component_id for node in nodes]),
            "instance_id": _string_array([node.instance_id for node in nodes]),
            "chain_id": _string_array([node.chain_id for node in nodes]),
            "residue_index": np.asarray(
                [node.residue_index for node in nodes],
                dtype=np.int64,
            ),
            "insertion_code": _string_array([node.insertion_code for node in nodes]),
            "source_chem_comp_type": _string_array(
                [node.source_chem_comp_type for node in nodes]
            ),
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
                "node_indices": np.arange(num_nodes, dtype=np.int64),
                "positions": positions.copy(),
                "valid_mask": proposal_valid_mask,
            },
            "split_groups": {
                "component": component_groups,
                "interface": interface_groups,
            },
        },
    }
    return schema.validate_component_extension_targets(extension_targets)


__all__ = ["build_component_extension_targets"]
