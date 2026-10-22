"""Offline real-data provenance, alignment and integration checks."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter

import numpy as np
import pytest
from biotite.structure.io import pdbx
from standard_target import (
    EMD_ID,
    FIXTURE_DIR,
    FULL_ENTRY_TARGETS_PATH,
    FULL_STRUCTURE_GZIP_PATH,
    PDB_ID,
    PROVENANCE_PATH,
    load_full_entry_full_atom_targets,
    load_standard_artifact_mapping,
    load_standard_full_atom_targets,
    load_standard_source_metadata,
    standard_expected,
    write_full_entry_structure,
    write_standard_structure,
)

from cryoflake.constants import CANONICAL_COMPONENT_IDS
from cryoflake.io.structure import load_full_backbone_from_structure
from cryoflake.storage.artifacts import validate_preprocessed_artifact
from cryoflake.storage.store import (
    load_preprocessed_artifact,
    write_preprocessed_artifact,
)
from cryoflake.targets.components import build_component_extension_targets


def _read_atoms(path):
    return pdbx.get_structure(
        pdbx.CIFFile.read(path),
        model=1,
        altloc="all",
        use_author_fields=True,
        extra_fields=["b_factor", "occupancy"],
    )


def _polymer_context(targets):
    mask = targets["atom_mask"]
    return {
        "polymer_atom_positions": targets["atom_positions"][mask],
        "polymer_atom_node_indices": np.nonzero(mask)[0],
        "polymer_chain_indices": targets["chain_index"],
        "polymer_residue_indices": targets["residue_index"],
        "polymer_aatype": targets["aatype"],
    }


def _assert_named_atom_alignment(targets, metadata, atoms):
    lookup = {
        (
            str(atom.chain_id),
            int(atom.res_id),
            str(atom.ins_code),
            str(atom.res_name),
            str(atom.atom_name),
        ): index
        for index, atom in enumerate(atoms)
    }
    rows, slots = np.nonzero(targets["atom_mask"])
    indices = [
        lookup[
            (
                str(metadata["source_chain_ids"][row]),
                int(targets["residue_index"][row]),
                str(metadata["source_insertion_codes"][row]),
                str(metadata["source_residue_names"][row]),
                str(targets["atom_names"][slot]),
            )
        ]
        for row, slot in zip(rows, slots, strict=True)
    ]
    np.testing.assert_array_equal(
        targets["atom_positions"][rows, slots], atoms.coord[indices]
    )
    np.testing.assert_array_equal(
        targets["b_factors"][rows, slots], atoms.b_factor[indices].astype(np.float32)
    )
    assert len(indices) == len(set(indices))


def test_standard_asset_provenance_and_size_are_pinned():
    provenance = json.loads(PROVENANCE_PATH.read_text())
    assert EMD_ID == provenance["emdb_id"] == "11736"
    assert PDB_ID == provenance["pdb_id"] == "7ae1"
    assert (
        provenance["target_parser"]["commit"]
        == "e52b1bd482696a5378deaa14eff6c6fe9155f880"
    )
    assert provenance["target_parser"]["model_weights_used"] is False
    assert provenance["selection"]["topology_indices_renumbered"] is False
    assert (
        provenance["selection"]["cached_geometry_recomputed_after_subsetting"] is False
    )
    total = 0
    for name, expected in provenance["assets"].items():
        path = FIXTURE_DIR / name
        assert path.stat().st_size == expected["bytes"]
        with path.open("rb") as stream:
            assert (
                hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"]
            )
        total += path.stat().st_size
    assert total < 6_000_000
    assert provenance["assets"]["7ae1.cif.gz"]["sha256"] == (
        "86052f1572b2764d9ca389dba227b992412bc1a2d9e72ba81faceb4c934bc578"
    )
    assert (
        hashlib.sha256(
            gzip.decompress(FULL_STRUCTURE_GZIP_PATH.read_bytes())
        ).hexdigest()
        == (provenance["source_structure_uncompressed_sha256"])
    )


def test_standard_helpers_return_fresh_mutable_payloads():
    first = load_standard_artifact_mapping()
    second = load_standard_artifact_mapping()
    first["density_grid"][0, 0, 0] = 999
    first["full_atom_targets"]["atom_positions"][0, 0] = -999
    first["full_atom_targets"]["atom_names"][0] = "X"
    first["residue_info"]["aa_type"][0] = "UNK"
    first["extension_targets"]["component"]["component_id"][0] = "X"
    fresh = load_standard_artifact_mapping()
    assert set(fresh) == set(second)
    assert "sequence" not in fresh and "sequence_mask" not in fresh
    assert (
        second["residue_info"]["aa_type"][0]
        == fresh["residue_info"]["aa_type"][0]
        == "MET"
    )
    for mapping in (second, fresh):
        assert mapping["density_grid"][0, 0, 0] != 999
        assert mapping["full_atom_targets"]["atom_names"][0] == "N"
        assert not np.any(mapping["full_atom_targets"]["atom_positions"] == -999)
        assert "X" not in mapping["extension_targets"]["component"]["component_id"]
    targets = load_standard_full_atom_targets()
    targets["aatype"][0] = 28
    assert load_standard_full_atom_targets()["aatype"][0] == 12
    expected = standard_expected()
    expected["protein_ca_sample"]["coordinate"][0] = -999
    assert (
        standard_expected()["protein_ca_sample"]["coordinate"][0] == 199.11599731445312
    )


def test_standard_mapping_retains_real_density_and_geometry(standard_artifact_mapping):
    mapping = standard_artifact_mapping
    expected = standard_expected()
    provenance = json.loads(PROVENANCE_PATH.read_text())
    density = mapping["density_grid"]
    assert density.shape == expected["grid_shape"] == (64, 64, 64)
    assert density.dtype == np.float32
    assert np.isfinite(density).all()
    assert density.min() < 0 < density.max()
    assert mapping["voxel_size"] == expected["voxel_size"] == 5.512499749660492
    np.testing.assert_array_equal(mapping["global_origin"], [0, 0, 0])
    np.testing.assert_array_equal(
        np.asarray(density.shape[::-1]) * mapping["voxel_size"],
        provenance["density_derivation"]["extent_xyz_angstrom"],
    )
    assert np.mean(density, dtype=np.float64) == pytest.approx(
        provenance["density_derivation"]["source_mean"], rel=1e-6, abs=1e-12
    )
    assert float(density[32, 32, 32]) == 0.00788439903408289
    targets = mapping["full_atom_targets"]
    assert targets["atom_positions"].shape == (189, 65, 3)
    assert targets["atom_names"].shape == (65,)
    assert targets["atom_names"][:5].tolist() == ["N", "CA", "C", "O", "CB"]
    assert targets["atom_names"][37] == "P"
    assert targets["atom_mask"].sum() == expected["present_atom_count"] == 2001
    for sample_name in ("protein_ca_sample", "nucleotide_p_sample"):
        sample = expected[sample_name]
        assert targets["atom_names"][sample["slot"]] == sample["atom_name"]
        np.testing.assert_array_equal(
            targets["atom_positions"][sample["row"], sample["slot"]],
            sample["coordinate"],
        )
    frame = expected["rigid_frame_sample"]
    assert targets["rigidgroups_gt_exists"][frame["row"], frame["group"]]
    np.testing.assert_array_equal(
        targets["rigidgroups_gt_frames"][frame["row"], frame["group"]], frame["matrix"]
    )
    torsion = expected["torsion_sample"]
    assert targets["torsion_angles_mask"][torsion["row"], torsion["torsion"]]
    np.testing.assert_array_equal(
        targets["torsion_angles_sin_cos"][torsion["row"], torsion["torsion"]],
        torsion["sin_cos"],
    )


def test_compact_targets_align_with_deposited_named_atoms(
    standard_artifact_mapping, standard_structure_path
):
    atoms = _read_atoms(standard_structure_path)
    assert len(atoms) == standard_expected()["structure_atom_count"] == 2017
    assert len(set(atoms.chain_id.tolist())) == 11
    targets = standard_artifact_mapping["full_atom_targets"]
    metadata = load_standard_source_metadata()
    _assert_named_atom_alignment(targets, metadata, atoms)
    coordinates, labels = load_full_backbone_from_structure(standard_structure_path)
    assert len(coordinates["CA"]) == 145
    assert len(coordinates["P"]) == 44
    assert labels == standard_artifact_mapping["residue_info"]
    for name, array in coordinates.items():
        np.testing.assert_array_equal(
            array, standard_artifact_mapping["backbone_coordinates"][name]
        )
    noncanonical = ~np.isin(atoms.res_name, tuple(CANONICAL_COMPONENT_IDS))
    assert Counter(atoms.res_name[noncanonical]) == {"ZN": 7, "MG": 1, "SF4": 8}


def test_compact_selection_and_caches_are_exact_subsets():
    provenance = json.loads(PROVENANCE_PATH.read_text())
    targets = load_standard_full_atom_targets()
    metadata = load_standard_source_metadata()
    rows = metadata["source_row_indices"]
    np.testing.assert_array_equal(rows, provenance["selection"]["source_row_indices"])
    with np.load(FULL_ENTRY_TARGETS_PATH, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    for name, array in targets.items():
        if name == "schema_version":
            assert array == 1
            continue
        expected = raw[name] if name == "atom_names" else raw[name][rows]
        np.testing.assert_array_equal(array, expected.astype(array.dtype))
    mapping = load_standard_artifact_mapping()
    component = mapping["extension_targets"]["component"]
    component_positions = component["atom_positions"][component["atom_names"] != ""]
    contact = np.zeros(len(raw["aatype"]), dtype=bool)
    for position in component_positions:
        distance_squared = np.sum((raw["atom_positions"] - position) ** 2, axis=-1)
        contact |= np.any(
            (distance_squared <= 25) & raw["atom_mask"].astype(bool), axis=1
        )
    assert contact.sum() == 70
    is_protein = (raw["aatype"] < 20) | (raw["aatype"] == 28)
    seed = contact | ~is_protein
    selected = seed.copy()
    for row in np.flatnonzero(seed):
        for neighbor in (row - 1, row + 1):
            if (
                0 <= neighbor < len(selected)
                and raw["chain_index"][neighbor] == raw["chain_index"][row]
            ):
                selected[neighbor] = True
    np.testing.assert_array_equal(rows, np.flatnonzero(selected))
    assert np.all(np.diff(rows) > 0)
    assert (~is_protein[rows]).sum() == 44
    caveat = provenance["upstream_torsion_caveat"]
    assert len(caveat["full_entry_affected_protein_chain_starts"]) == 16
    assert caveat["compact_affected_source_rows"] == []
    assert not np.intersect1d(
        rows, caveat["full_entry_affected_protein_chain_starts"]
    ).size


@pytest.mark.parametrize(
    "component_ids,expected_counts",
    [
        (None, {"ZN": 7, "MG": 1, "SF4": 1}),
        ({"ZN"}, {"ZN": 7}),
        ({"SF4"}, {"SF4": 1}),
        ({"MG"}, {"MG": 1}),
        (set(), {}),
    ],
)
def test_structure_filters_only_genuine_components(
    tmp_path, component_ids, expected_counts
):
    path = write_standard_structure(tmp_path / "7ae1.cif", component_ids=component_ids)
    mapping = load_standard_artifact_mapping()
    coordinates, labels = load_full_backbone_from_structure(path)
    assert labels == mapping["residue_info"]
    for name, expected in mapping["backbone_coordinates"].items():
        np.testing.assert_array_equal(coordinates[name], expected)
    extension = build_component_extension_targets(
        path,
        global_origin=mapping["global_origin"],
        voxel_size=mapping["voxel_size"],
        grid_shape=mapping["density_grid"].shape,
        **_polymer_context(mapping["full_atom_targets"]),
    )
    component = extension["component"]
    assert Counter(component["component_id"]) == expected_counts
    assert np.all(component["proposal_targets"]["valid_mask"])
    assert np.all(component["interface_group_id"] != "")
    if component_ids is None:
        cached = mapping["extension_targets"]["component"]
        for name, values in component.items():
            if isinstance(values, np.ndarray):
                np.testing.assert_array_equal(values, cached[name])


@pytest.mark.parametrize("invalid_ids", [{"LIG"}, {"ATP"}, "ZN", [1]])
def test_structure_filter_rejects_fabricated_chemistry(tmp_path, invalid_ids):
    path = tmp_path / "7ae1.cif"
    with pytest.raises(ValueError, match="component IDs|component_ids"):
        write_standard_structure(path, component_ids=invalid_ids)
    assert not path.exists()


def test_compact_artifact_roundtrip_and_validation(tmp_path, standard_artifact_mapping):
    reference = tmp_path / "emd_11736_7ae1"
    write_preprocessed_artifact(reference, standard_artifact_mapping)
    result = validate_preprocessed_artifact(reference)
    assert result.grid_shape == (64, 64, 64)
    assert result.protein_nodes == 145
    assert result.nucleotide_nodes == 44
    assert result.component_nodes == 9
    restored = load_preprocessed_artifact(reference)
    for name, values in standard_artifact_mapping["full_atom_targets"].items():
        np.testing.assert_array_equal(restored["full_atom_targets"][name], values)
    np.testing.assert_array_equal(
        restored["density_grid"], standard_artifact_mapping["density_grid"]
    )


def test_full_entry_reference_alignment_and_artifact_roundtrip(tmp_path):
    path = write_full_entry_structure(tmp_path / "7ae1.cif")
    atoms = _read_atoms(path)
    targets = load_full_entry_full_atom_targets()
    with np.load(FULL_ENTRY_TARGETS_PATH, allow_pickle=False) as archive:
        metadata = {
            name: archive[name]
            for name in (
                "source_chain_ids",
                "source_insertion_codes",
                "source_residue_names",
            )
        }
    _assert_named_atom_alignment(targets, metadata, atoms)
    assert len(atoms) == 42608
    assert targets["atom_mask"].sum() == 42592
    mapping = load_standard_artifact_mapping()
    coordinates, labels = load_full_backbone_from_structure(path)
    assert len(coordinates["CA"]) == 5243 and len(coordinates["P"]) == 44
    mapping.update(
        backbone_coordinates=coordinates,
        residue_info=labels,
        full_atom_targets=targets,
        extension_targets=build_component_extension_targets(
            path,
            global_origin=mapping["global_origin"],
            voxel_size=mapping["voxel_size"],
            grid_shape=mapping["density_grid"].shape,
            **_polymer_context(targets),
        ),
    )
    reference = tmp_path / "full_entry"
    write_preprocessed_artifact(reference, mapping)
    result = validate_preprocessed_artifact(reference)
    assert result.protein_nodes == 5243
    assert result.nucleotide_nodes == 44
    assert result.component_nodes == 9
    restored = load_preprocessed_artifact(reference, root_keys=["full_atom_targets"])
    for name, values in targets.items():
        np.testing.assert_array_equal(restored["full_atom_targets"][name], values)
