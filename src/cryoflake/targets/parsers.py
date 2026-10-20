"""mmCIF reader for additive molecular-component targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cryoflake.common import clean_token, finite_float, integer_token
from cryoflake.constants import CANONICAL_COMPONENT_IDS
from cryoflake.io.structure import _cif_column, _read_cif_data
from cryoflake.targets.common import (
    _Atom,
    _classify,
    _deduplicate_atoms,
    _infer_element,
    _Node,
)


def _category_mapping(data, id_key: str, value_key: str) -> dict[str, str]:
    ids = data.get(id_key, ())
    values = data.get(value_key, ())
    return {
        clean_token(identifier): clean_token(value)
        for identifier, value in zip(ids, values)
    }


def _read_component_nodes(structure_path: str | Path) -> list[_Node]:
    path = Path(structure_path)
    if path.suffix.lower() not in {".cif", ".mmcif"}:
        raise ValueError(
            "Component target structures must be uncompressed .cif or .mmcif files: "
            f"{path}"
        )
    data = _read_cif_data(path)

    atom_names = list(data.get("_atom_site.label_atom_id", ()))
    num_atoms = len(atom_names)
    if num_atoms == 0:
        return []
    columns = {
        "component": _cif_column(data, "_atom_site.label_comp_id", num_atoms),
        "entity": _cif_column(data, "_atom_site.label_entity_id", num_atoms),
        "label_asym": _cif_column(data, "_atom_site.label_asym_id", num_atoms),
        "auth_asym": _cif_column(data, "_atom_site.auth_asym_id", num_atoms),
        "label_seq": _cif_column(data, "_atom_site.label_seq_id", num_atoms),
        "auth_seq": _cif_column(data, "_atom_site.auth_seq_id", num_atoms),
        "icode": _cif_column(data, "_atom_site.pdbx_PDB_ins_code", num_atoms),
        "model": _cif_column(data, "_atom_site.pdbx_PDB_model_num", num_atoms, "1"),
        "altloc": _cif_column(data, "_atom_site.label_alt_id", num_atoms),
        "element": _cif_column(data, "_atom_site.type_symbol", num_atoms),
        "occupancy": _cif_column(data, "_atom_site.occupancy", num_atoms, "1"),
        "b_factor": _cif_column(data, "_atom_site.B_iso_or_equiv", num_atoms, "0"),
        "group": _cif_column(data, "_atom_site.group_PDB", num_atoms),
        "x": _cif_column(data, "_atom_site.Cartn_x", num_atoms),
        "y": _cif_column(data, "_atom_site.Cartn_y", num_atoms),
        "z": _cif_column(data, "_atom_site.Cartn_z", num_atoms),
    }
    first_model = columns["model"][0]
    entity_types = _category_mapping(data, "_entity.id", "_entity.type")
    polymer_types = _category_mapping(
        data, "_entity_poly.entity_id", "_entity_poly.type"
    )
    branch_types = _category_mapping(
        data, "_pdbx_entity_branch.entity_id", "_pdbx_entity_branch.type"
    )
    chem_types = {
        component_id.upper(): component_type.upper()
        for component_id, component_type in _category_mapping(
            data,
            "_chem_comp.id",
            "_chem_comp.type",
        ).items()
    }

    locations: dict[tuple[str, ...], dict[str, list[_Atom]]] = {}
    location_metadata: dict[tuple[str, ...], tuple[str, str]] = {}
    for index, atom_name in enumerate(atom_names):
        if columns["model"][index] != first_model:
            continue
        component_id = clean_token(columns["component"][index]).upper()
        if not component_id or component_id in CANONICAL_COMPONENT_IDS:
            continue
        position = np.asarray(
            [
                finite_float(columns["x"][index], np.nan),
                finite_float(columns["y"][index], np.nan),
                finite_float(columns["z"][index], np.nan),
            ],
            dtype=np.float32,
        )
        if not np.isfinite(position).all():
            continue
        entity_id = clean_token(columns["entity"][index])
        label_asym = clean_token(columns["label_asym"][index])
        auth_asym = clean_token(columns["auth_asym"][index])
        label_seq = clean_token(columns["label_seq"][index])
        auth_seq = clean_token(columns["auth_seq"][index])
        insertion_code = clean_token(columns["icode"][index])
        location_key = (
            clean_token(first_model),
            entity_id,
            label_asym,
            label_seq,
            auth_asym,
            auth_seq,
            insertion_code,
        )
        atom_name = clean_token(atom_name).upper()
        element = clean_token(columns["element"][index]).upper()
        atom = _Atom(
            name=atom_name,
            element=element or _infer_element(atom_name),
            position=position,
            occupancy=max(
                finite_float(columns["occupancy"][index], 0.0),
                0.0,
            ),
            b_factor=finite_float(columns["b_factor"][index], 0.0),
            altloc=clean_token(columns["altloc"][index]),
            source_index=index,
            group_pdb=clean_token(columns["group"][index]).upper(),
        )
        locations.setdefault(location_key, {}).setdefault(component_id, []).append(atom)
        location_metadata[location_key] = (
            entity_id,
            auth_asym or label_asym,
        )

    nodes = []
    for location_key, component_variants in locations.items():
        variants = []
        for component_id, atoms in component_variants.items():
            selected_atoms = _deduplicate_atoms(atoms)
            mean_occupancy = float(np.mean([atom.occupancy for atom in selected_atoms]))
            variants.append((-mean_occupancy, component_id, selected_atoms))
        _, component_id, atoms = sorted(variants)[0]
        entity_id, chain_id = location_metadata[location_key]
        residue_index = integer_token(
            location_key[5],
            location_key[3],
            default=0,
        )
        insertion_code = location_key[6]
        group_pdb = atoms[0].group_pdb if atoms else ""
        chem_comp_type = chem_types.get(component_id, "")
        node_kind, polymer_kind = _classify(
            component_id=component_id,
            atoms=atoms,
            entity_type=entity_types.get(entity_id, ""),
            polymer_type=polymer_types.get(entity_id, ""),
            branch_type=branch_types.get(entity_id, ""),
            chem_comp_type=chem_comp_type,
            group_pdb=group_pdb,
        )
        instance_id = json.dumps(
            ["mmcif", *location_key, component_id],
            separators=(",", ":"),
        )
        nodes.append(
            _Node(
                component_id=component_id,
                instance_id=instance_id,
                chain_id=chain_id,
                residue_index=residue_index,
                insertion_code=insertion_code,
                source_chem_comp_type=chem_comp_type,
                node_kind=node_kind,
                polymer_kind=polymer_kind,
                atoms=atoms,
            )
        )
    return nodes
