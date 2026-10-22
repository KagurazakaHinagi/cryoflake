"""Regenerate the offline fixture from the recorded local real-data inputs.

This is a development utility, never run by CI. It does not download inputs,
invoke NextAngelo, load weights, or synthesize supervision.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import mrcfile
import numpy as np
from biotite.structure.io import pdbx

from cryoflake.io.mrc import rescale_real
from cryoflake.io.structure import load_full_backbone_from_structure
from cryoflake.targets.components import build_component_extension_targets
from cryoflake.targets.full_atom import extract_full_atom_targets


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _flatten(mapping, prefix=""):
    arrays = {}
    for name, value in mapping.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            arrays.update(_flatten(value, key))
        else:
            arrays[key] = np.asarray(value)
    return arrays


def regenerate(source_dir: Path, fixture_dir: Path):
    fixture_dir.mkdir(parents=True, exist_ok=True)
    source_cif = source_dir / "inputs/structures/7ae1.cif"
    source_map = source_dir / "inputs/maps/emd_11736.map"
    with np.load(
        source_dir / "raw_nextangelo_targets.npz", allow_pickle=False
    ) as archive:
        raw = {name: archive[name] for name in archive.files}
    full_targets = extract_full_atom_targets(SimpleNamespace(**raw))
    with mrcfile.mmap(source_map, mode="r") as volume:
        original_shape = tuple(volume.data.shape)
        original_spacing = float(volume.voxel_size.x)
        origin = np.array(
            [volume.header.origin.x, volume.header.origin.y, volume.header.origin.z],
            dtype=np.float32,
        )
        if original_shape != (336, 336, 336) or np.any(origin):
            raise ValueError("Unexpected EMD-11736 deposited map geometry")
        original_mean = float(np.mean(volume.data, dtype=np.float64))
        normalization = (64 / 336) ** 3
        density = np.asarray(
            rescale_real(volume.data, 64) * normalization, dtype=np.float32
        )
    spacing = original_spacing * 336 / 64
    mask = full_targets["atom_mask"]
    context = dict(
        polymer_atom_positions=full_targets["atom_positions"][mask],
        polymer_atom_node_indices=np.nonzero(mask)[0],
        polymer_chain_indices=full_targets["chain_index"],
        polymer_residue_indices=full_targets["residue_index"],
        polymer_aatype=full_targets["aatype"],
    )
    component = build_component_extension_targets(
        source_cif,
        global_origin=origin,
        voxel_size=spacing,
        grid_shape=density.shape,
        **context,
    )["component"]
    component_atoms = component["atom_positions"][component["atom_names"] != ""]
    contact_rows = np.zeros(len(mask), dtype=bool)
    for position in component_atoms:
        square_distances = np.sum(
            (full_targets["atom_positions"].astype(np.float64) - position) ** 2,
            axis=-1,
        )
        contact_rows |= np.any((square_distances <= 5.0**2) & mask, axis=-1)
    protein_mask = (full_targets["aatype"] < 20) | (full_targets["aatype"] == 28)
    seeds = contact_rows | ~protein_mask
    selected = seeds.copy()
    for row in np.flatnonzero(seeds):
        for neighbor in (row - 1, row + 1):
            if 0 <= neighbor < len(mask) and (
                raw["chain_index"][neighbor] == raw["chain_index"][row]
            ):
                selected[neighbor] = True
    rows = np.flatnonzero(selected)
    targets = {
        name: (value.copy() if name == "atom_names" else value[rows].copy())
        if isinstance(value, np.ndarray)
        else value
        for name, value in full_targets.items()
    }

    identities = {
        (
            str(raw["source_chain_ids"][row]),
            str(raw["residue_index"][row]),
            str(raw["source_insertion_codes"][row]),
            str(raw["source_residue_names"][row]),
        )
        for row in rows
    }
    identities.update(
        zip(
            component["chain_id"].tolist(),
            component["residue_index"].astype(str).tolist(),
            component["insertion_code"].tolist(),
            component["component_id"].tolist(),
            strict=True,
        )
    )
    cif = pdbx.CIFFile.read(source_cif)
    block = next(iter(cif.values()))
    atom_site = block["atom_site"]
    columns = {
        name: atom_site[name].as_array(str)
        for name in (
            "auth_asym_id",
            "auth_seq_id",
            "pdbx_PDB_ins_code",
            "label_comp_id",
        )
    }
    included = np.asarray(
        [
            (chain, residue, "" if insertion in {"?", "."} else insertion, name)
            in identities
            for chain, residue, insertion, name in zip(*columns.values(), strict=True)
        ]
    )
    block["atom_site"] = pdbx.CIFCategory(
        {name: column.as_array(str)[included] for name, column in atom_site.items()}
    )
    subset_stream = io.StringIO()
    cif.write(subset_stream)
    subset_text = subset_stream.getvalue()
    subset_path = fixture_dir / "subset.cif.gz"
    subset_path.write_bytes(gzip.compress(subset_text.encode(), mtime=0))
    # The public structure reader requires a local uncompressed CIF.
    temporary_cif = fixture_dir / "_subset_generation.cif"
    temporary_cif.write_text(subset_text)
    try:
        backbone, residue_info = load_full_backbone_from_structure(temporary_cif)
        subset_mask = targets["atom_mask"]
        extension = build_component_extension_targets(
            temporary_cif,
            global_origin=origin,
            voxel_size=spacing,
            grid_shape=density.shape,
            polymer_atom_positions=targets["atom_positions"][subset_mask],
            polymer_atom_node_indices=np.nonzero(subset_mask)[0],
            polymer_chain_indices=targets["chain_index"],
            polymer_residue_indices=targets["residue_index"],
            polymer_aatype=targets["aatype"],
        )
    finally:
        temporary_cif.unlink()
    for name in ("CA", "C", "N", "O", "P"):
        slot = targets["atom_names"].tolist().index(name)
        row_mask = ~protein_mask[rows] if name == "P" else protein_mask[rows]
        positions = targets["atom_positions"][row_mask, slot].copy()
        positions[~targets["atom_mask"][row_mask, slot]] = np.nan
        np.testing.assert_array_equal(positions, backbone[name])
    np.testing.assert_array_equal(
        extension["component"]["interface_group_id"], component["interface_group_id"]
    )
    mapping = {
        "density_grid": density,
        "global_origin": origin,
        "voxel_size": spacing,
        "backbone_coordinates": backbone,
        "residue_info": residue_info,
        "full_atom_targets": targets,
        "extension_targets": extension,
    }
    arrays = _flatten(mapping)
    arrays.update(
        source_row_indices=rows,
        source_chain_ids=raw["source_chain_ids"][rows],
        source_residue_names=raw["source_residue_names"][rows],
        source_insertion_codes=raw["source_insertion_codes"][rows],
        source_selected_altloc_ids=raw["source_selected_altloc_ids"][rows],
    )
    np.savez_compressed(fixture_dir / "standard_arrays.npz", **arrays)
    shutil.copyfile(
        source_dir / "inputs/structures/7ae1.cif.gz", fixture_dir / "7ae1.cif.gz"
    )
    shutil.copyfile(
        source_dir / "raw_nextangelo_targets.npz",
        fixture_dir / "full_entry_targets.npz",
    )
    protein_count = int(protein_mask[rows].sum())

    def coordinate_sample(row, name):
        slot = targets["atom_names"].tolist().index(name)
        return {
            "row": row,
            "source_row": int(rows[row]),
            "slot": slot,
            "chain_id": str(raw["source_chain_ids"][rows[row]]),
            "residue_index": int(targets["residue_index"][row]),
            "residue_name": str(raw["source_residue_names"][rows[row]]),
            "atom_name": name,
            "coordinate": targets["atom_positions"][row, slot].tolist(),
        }

    frame_row, frame_group = np.argwhere(targets["rigidgroups_gt_exists"])[0]
    torsion_row, torsion_index = np.argwhere(targets["torsion_angles_mask"])[0]
    expected = {
        "grid_shape": list(density.shape),
        "voxel_size": spacing,
        "global_origin": origin.tolist(),
        "protein_nodes": protein_count,
        "nucleotide_nodes": int((~protein_mask[rows]).sum()),
        "full_atom_residue_count": len(rows),
        "present_atom_count": int(targets["atom_mask"].sum()),
        "component_nodes": len(component["positions"]),
        "component_atom_count": len(component_atoms),
        "structure_atom_count": int(included.sum()),
        "chain_count": len(set(raw["source_chain_ids"][rows].tolist())),
        "component_ids": sorted(set(component["component_id"].tolist())),
        "component_id_counts": {
            name: int(np.sum(component["component_id"] == name))
            for name in sorted(set(component["component_id"].tolist()))
        },
        "interface_nodes": int(np.count_nonzero(component["interface_group_id"])),
        "proposal_valid_nodes": int(component["proposal_targets"]["valid_mask"].sum()),
        "protein_ca_sample": coordinate_sample(0, "CA"),
        "nucleotide_p_sample": coordinate_sample(protein_count, "P"),
        "rigid_frame_sample": {
            "row": int(frame_row),
            "group": int(frame_group),
            "matrix": targets["rigidgroups_gt_frames"][frame_row, frame_group].tolist(),
        },
        "torsion_sample": {
            "row": int(torsion_row),
            "torsion": int(torsion_index),
            "sin_cos": targets["torsion_angles_sin_cos"][
                torsion_row, torsion_index
            ].tolist(),
        },
        "density_mean": float(np.mean(density, dtype=np.float64)),
        "density_samples": {
            "0,0,0": float(density[0, 0, 0]),
            "32,32,32": float(density[32, 32, 32]),
            "63,63,63": float(density[63, 63, 63]),
        },
    }
    chain_starts = np.flatnonzero(raw["chain_index"][1:] != raw["chain_index"][:-1]) + 1
    affected_starts = chain_starts[protein_mask[chain_starts]]
    inputs = json.loads((source_dir / "inputs.json").read_text())
    caveat = json.loads((source_dir / "nextangelo_torsion_caveat.json").read_text())
    provenance = {
        "emdb_id": "11736",
        "pdb_id": "7ae1",
        "resolution_angstrom": 2.8,
        "source_entries": {
            "emdb": inputs["emdb_entry"],
            "pdb": "https://www.rcsb.org/structure/7AE1",
        },
        "sources": inputs["sources"],
        "source_structure_uncompressed_sha256": _digest(source_cif),
        "source_map_uncompressed_sha256": _digest(source_map),
        "target_parser": {
            "module": "nextangelo.utils.protein.get_protein_from_file_path",
            "commit": inputs["nextangelo_commit"],
            "version": "1.0.17",
            "atom_names": "nextangelo.utils.residue_constants.atom_types",
            "model_weights_used": False,
        },
        "schema_versions": {
            "full_atom_targets": targets["schema_version"],
            "extension_targets": extension["schema_version"],
        },
        "full_entry": {
            "protein_nodes": int(protein_mask.sum()),
            "nucleotide_nodes": int((~protein_mask).sum()),
            "present_atom_count": int(mask.sum()),
            "structure_atom_count": 42608,
            "component_nodes": 9,
        },
        "selection": {
            "rule": "All 44 nucleotide rows and canonical residues with any present atom within 5.0 angstrom of any component atom; add each seed's immediately previous/next observed source row in the same chain, once, without recursive expansion.",
            "source_row_indices": rows.tolist(),
            "contact_source_row_indices": np.flatnonzero(contact_rows).tolist(),
            "seed_source_row_indices": np.flatnonzero(seeds).tolist(),
            "source_order_preserved": True,
            "topology_indices_renumbered": False,
            "cached_geometry_recomputed_after_subsetting": False,
            "all_component_atoms_retained": True,
        },
        "density_derivation": {
            "operation": "cryoflake.io.mrc.rescale_real(deposited_grid, 64) * (64 / 336)**3, converted to float32",
            "source_shape_zyx": list(original_shape),
            "source_voxel_size_angstrom": original_spacing,
            "source_mean": original_mean,
            "fourier_amplitude_normalization": normalization,
            "extent_xyz_angstrom": [original_spacing * 336] * 3,
            "origin_xyz_angstrom": origin.tolist(),
            "purpose": "Compact derived volume for storage/geometry tests; not the full-resolution deposited map.",
        },
        "upstream_torsion_caveat": {
            "description": "The authentic NextAngelo parser masks pre-omega/phi from previous source-row atom presence without checking chain boundaries. Artifact conformance does not certify biochemical torsion correctness; cached targets are preserved without correction.",
            "source": "nextangelo/utils/protein.py",
            "source_sha256": caveat["source_sha256"],
            "source_line_anchors": caveat["source_line_anchors"],
            "full_entry_affected_protein_chain_starts": affected_starts.tolist(),
            "compact_affected_source_rows": np.intersect1d(
                rows, affected_starts
            ).tolist(),
            "cross_chain_previous_C_to_current_N_distance_angstrom": caveat[
                "interchain_previous_C_to_current_N_distance_angstrom"
            ],
        },
        "expected": expected,
    }
    provenance["assets"] = {
        path.name: {"sha256": _digest(path), "bytes": path.stat().st_size}
        for path in sorted(fixture_dir.iterdir())
        if path.suffix in {".gz", ".npz"}
    }
    (fixture_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(expected, indent=2))
    print(
        "Binary asset bytes:",
        sum(item["bytes"] for item in provenance["assets"].values()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    regenerate(args.source_dir, Path(__file__).resolve().parent)
