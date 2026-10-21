"""Built-in raw preprocessing produces complete portable training artifacts."""

from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from cryoflake.cli import app
from cryoflake.io.mrc import save_mrc
from cryoflake.preprocess.adapters import default
from cryoflake.storage import load_preprocessed_artifact, transports
from cryoflake.storage.artifacts import validate_preprocessed_artifact
from cryoflake.targets.structure import load_full_atom_targets_from_structure


def _write_structure(path, *, shift=(0, 0, 0), protein="ALA", ligand=True):
    records = [
        ("ATOM", "N", protein, "A", 1, "N", (0, 0, 0)),
        ("ATOM", "CA", protein, "A", 1, "C", (1, 0, 0)),
        ("ATOM", "C", protein, "A", 1, "C", (1, 1, 0)),
        ("ATOM", "P", "DA", "B", 2, "P", (3, 3, 3)),
    ]
    if ligand:
        records.extend(
            [
                ("HETATM", "C1", "LIG", "C", 3, "C", (2, 0, 0)),
                ("HETATM", "O1", "LIG", "C", 3, "O", (2, 1, 0)),
            ]
        )
    header = """data_structure
loop_
_chem_comp.id
_chem_comp.type
ALA 'L-PEPTIDE LINKING'
GLY 'L-PEPTIDE LINKING'
DA 'DNA LINKING'
LIG NON-POLYMER
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_PDB_model_num
_atom_site.type_symbol
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""
    lines = []
    for group, atom, residue, chain, residue_index, element, position in records:
        x, y, z = np.asarray(position) + shift
        lines.append(
            f"{group} {atom} . {residue} {chain} {residue_index} {residue_index} "
            f"{chain} {residue_index} ? 1 {element} 1 20 {x} {y} {z}\n"
        )
    path.write_text(header + "".join(lines))


@pytest.fixture
def raw_inputs(tmp_path):
    map_path = tmp_path / "emd_1234.mrc"
    density = np.arange(4**3, dtype=np.float32).reshape(4, 4, 4)
    save_mrc(density, 1, np.zeros(3), map_path)
    structure_path = tmp_path / "1abc.cif"
    _write_structure(structure_path)
    return map_path, structure_path


def test_default_adapter_round_trip_includes_geometry_masks_and_interfaces(
    raw_inputs, tmp_path
):
    map_path, structure_path = raw_inputs
    artifact_reference = tmp_path / "artifact"
    default.run(
        map_path=map_path,
        structure_path=structure_path,
        artifact_reference=artifact_reference,
        target_voxel_size=2,
        interface_contact_cutoff=1.5,
    )
    result = validate_preprocessed_artifact(artifact_reference)
    assert result.grid_shape == (64, 64, 64)
    assert (result.protein_nodes, result.nucleotide_nodes, result.component_nodes) == (
        1,
        1,
        1,
    )
    artifact_mapping = load_preprocessed_artifact(artifact_reference)
    assert artifact_mapping["density_grid"].dtype == np.float32
    assert artifact_mapping["voxel_size"] == 2
    np.testing.assert_array_equal(artifact_mapping["global_origin"], [-62] * 3)
    np.testing.assert_array_equal(artifact_mapping["shape"], [64] * 3)
    np.testing.assert_array_equal(artifact_mapping["cella"], [128] * 3)
    assert artifact_mapping["grid_spectral_amplitude"].shape == (33,)
    assert np.isfinite(artifact_mapping["grid_spectral_amplitude"]).all()
    assert np.isnan(artifact_mapping["backbone_coordinates"]["O"]).all()
    assert artifact_mapping["residue_info"]["aa_type"] == ["ALA"]
    assert artifact_mapping["residue_info"]["nuc_type"] == ["DA"]
    full_atom = artifact_mapping["full_atom_targets"]
    assert full_atom["prot_mask"].tolist() == [True, False]
    assert full_atom["atom_mask"].sum(axis=1).tolist() == [3, 1]
    assert np.isfinite(full_atom["atom_positions"]).all()
    assert np.count_nonzero(full_atom["rigidgroups_gt_exists"][1]) == 0
    component = artifact_mapping["extension_targets"]["component"]
    assert component["component_id"].tolist() == ["LIG"]
    assert component["interface_contact_cutoff"] == 1.5
    assert component["interface_mask"].tolist() == [True]
    assert component["proposal_targets"]["valid_mask"].tolist() == [True]
    assert set(path.name for path in artifact_reference.iterdir()) == {
        "metadata.json",
        "core.npz",
    }
    assert "grid_io" not in artifact_mapping
    assert "cas" not in artifact_mapping
    metadata = json.loads((artifact_reference / "metadata.json").read_text())
    assert metadata["format_version"] == 1
    assert "nextangelo" not in json.dumps(metadata).lower()
    with np.load(artifact_reference / "core.npz", allow_pickle=False) as arrays:
        assert all(arrays[key].dtype.kind != "O" for key in arrays.files)


def test_original_structure_supplies_all_targets_in_its_own_frame(raw_inputs, tmp_path):
    map_path, fitted_structure_path = raw_inputs
    shift = np.asarray([10, 20, 30])
    _write_structure(fitted_structure_path, shift=shift, ligand=False)
    original_structure_path = tmp_path / "original.cif"
    _write_structure(original_structure_path, protein="GLY")
    save_mrc(np.ones((4, 4, 4), dtype=np.float32), 1, shift, map_path)
    artifact_reference = tmp_path / "artifact"
    default.run(
        map_path=map_path,
        structure_path=fitted_structure_path,
        original_structure_path=original_structure_path,
        artifact_reference=artifact_reference,
    )
    payload = load_preprocessed_artifact(artifact_reference)
    np.testing.assert_array_equal(payload["global_origin"], [-62] * 3)
    np.testing.assert_array_equal(payload["backbone_coordinates"]["CA"], [[1, 0, 0]])
    np.testing.assert_array_equal(payload["backbone_coordinates"]["P"], [[3, 3, 3]])
    assert payload["residue_info"]["aa_type"] == ["GLY"]
    expected_targets = load_full_atom_targets_from_structure(original_structure_path)
    for field, expected in expected_targets.items():
        np.testing.assert_array_equal(payload["full_atom_targets"][field], expected)
    component = payload["extension_targets"]["component"]
    np.testing.assert_array_equal(
        component["atom_positions"][0], [[2, 0, 0], [2, 1, 0]]
    )
    assert component["interface_mask"].tolist() == [True]


def test_original_structure_can_leave_origin_uncorrected(raw_inputs, tmp_path):
    map_path, structure_path = raw_inputs
    original_structure_path = tmp_path / "original.cif"
    _write_structure(original_structure_path, shift=[1, 2, 3])
    default.run(
        map_path=map_path,
        structure_path=structure_path,
        original_structure_path=original_structure_path,
        artifact_reference=tmp_path / "artifact",
        fix_shift=False,
    )
    payload = load_preprocessed_artifact(tmp_path / "artifact")
    np.testing.assert_array_equal(payload["global_origin"], [-62] * 3)
    np.testing.assert_array_equal(payload["backbone_coordinates"]["CA"], [[2, 2, 3]])


def test_gzip_inputs_and_fasta_are_portable(raw_inputs, tmp_path):
    paths = []
    for path in raw_inputs:
        compressed_path = path.with_suffix(path.suffix + ".gz")
        with gzip.open(compressed_path, "wb") as compressed:
            compressed.write(path.read_bytes())
        paths.append(compressed_path)
    fasta_path = tmp_path / "sequences.fasta.gz"
    with gzip.open(fasta_path, "wt") as compressed:
        compressed.write(
            ">protein|Chains X [auth A], Z [auth C]\nAG\n>nucleotide|Chain B\nAC\n"
        )
    artifact_reference = tmp_path / "artifact"
    default.run(
        map_path=paths[0],
        structure_path=paths[1],
        fasta_path=fasta_path,
        artifact_reference=artifact_reference,
    )
    payload = load_preprocessed_artifact(artifact_reference)
    assert payload["fasta"] == {
        "sequences": [
            {"sequence": "AG", "chain_count": 2, "chain_ids": ["A", "C"]},
            {"sequence": "AC", "chain_count": 1, "chain_ids": ["B"]},
        ],
        "sequence_names": ["protein", "nucleotide"],
        "chain_to_seqid": {"A": 0, "C": 0, "B": 1},
    }
    assert "sequence" not in payload
    assert not (artifact_reference / "sequence.npz").exists()
    validate_preprocessed_artifact(artifact_reference)


def test_pdb_inputs_preserve_ligand_altloc_occupancy_and_residue_identity(
    raw_inputs, tmp_path
):
    map_path, _ = raw_inputs
    structure_path = tmp_path / "structure.pdb"

    def atom(serial, name, residue, chain, residue_index, x, *, altloc="", occupancy=1):
        group = "ATOM  " if residue == "ALA" else "HETATM"
        return (
            f"{group}{serial:5d} {name:^4s}{altloc:1s}{residue:>3s} {chain:1s}"
            f"{residue_index:4d}A   {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
            f"{occupancy:6.2f}{12.0:6.2f}          {name[0]:>2s}  \n"
        )

    structure_path.write_text(
        atom(1, "N", "ALA", "A", 1, 0)
        + atom(2, "CA", "ALA", "A", 1, 1)
        + atom(3, "C", "ALA", "A", 1, 2)
        + atom(4, "C1", "LIG", "Z", -2, 3, altloc="B", occupancy=0.8)
        + atom(5, "C1", "LIG", "Z", -2, 9, altloc="A", occupancy=0.2)
        + "END\n"
    )
    artifact_reference = tmp_path / "artifact"
    default.run(
        map_path=map_path,
        structure_path=structure_path,
        artifact_reference=artifact_reference,
    )
    payload = load_preprocessed_artifact(artifact_reference)
    component = payload["extension_targets"]["component"]
    assert component["component_id"].tolist() == ["LIG"]
    assert component["chain_id"].tolist() == ["Z"]
    assert component["residue_index"].tolist() == [-2]
    assert component["insertion_code"].tolist() == ["A"]
    np.testing.assert_array_equal(component["atom_positions"], [[[3, 0, 0]]])
    np.testing.assert_allclose(component["occupancies"], [[0.8]])
    validate_preprocessed_artifact(artifact_reference)


@pytest.mark.parametrize("name", ["target_voxel_size", "interface_contact_cutoff"])
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, "invalid", None, True])
def test_invalid_geometry_is_rejected_before_input_or_output_io(
    tmp_path, monkeypatch, name, value
):
    def unexpected_io(*args, **kwargs):
        pytest.fail("Invalid geometry must be rejected before input I/O")

    monkeypatch.setattr(default, "_input_path", unexpected_io)
    with pytest.raises(ValueError, match=f"{name} must be a positive finite number"):
        default.run(
            map_path=tmp_path / "missing.mrc",
            structure_path=tmp_path / "missing.cif",
            artifact_reference=tmp_path / "artifact",
            **{name: value},
        )
    assert not (tmp_path / "artifact").exists()


@pytest.mark.parametrize(
    "output_name", ["artifact.pkl", "artifact.npz", "artifact.npy"]
)
def test_single_file_output_names_are_rejected_before_map_read(
    raw_inputs, tmp_path, monkeypatch, output_name
):
    def unexpected_io(*args, **kwargs):
        pytest.fail("Invalid output must be rejected before map loading")

    monkeypatch.setattr(default, "load_mrc", unexpected_io)
    with pytest.raises(ValueError, match="single-file artifacts are unsupported"):
        default.run(
            map_path=raw_inputs[0],
            structure_path=raw_inputs[1],
            artifact_reference=tmp_path / output_name,
        )


@pytest.mark.parametrize("invalid_input", ["density", "anchors", "alignment", "fasta"])
def test_invalid_payload_is_rejected_without_creating_artifact(
    raw_inputs, tmp_path, invalid_input
):
    map_path, structure_path = raw_inputs
    kwargs = {}
    if invalid_input == "density":
        density = np.ones((4, 4, 4), dtype=np.float32)
        density[0, 0, 0] = np.nan
        save_mrc(density, 1, np.zeros(3), map_path)
    elif invalid_input == "anchors":
        _write_structure(structure_path, ligand=False)
        structure_path.write_text(
            structure_path.read_text()
            .replace("ATOM CA", "ATOM CB")
            .replace("ATOM P", "ATOM O")
        )
    elif invalid_input == "alignment":
        _write_structure(structure_path, shift=[1000, 1000, 1000])
    else:
        fasta_path = tmp_path / "empty.fasta"
        fasta_path.write_text("")
        kwargs["fasta_path"] = fasta_path
    artifact_reference = tmp_path / "artifact"
    with pytest.raises((ValueError, RuntimeError)):
        default.run(
            map_path=map_path,
            structure_path=structure_path,
            artifact_reference=artifact_reference,
            **kwargs,
        )
    assert not artifact_reference.exists()


def test_default_adapter_runs_through_batch_cli(raw_inputs, tmp_path):
    source_csv = tmp_path / "source.csv"
    source_csv.write_text("EMD-,Associated PDB\nEMD-1234,1ABC\n")
    manifest_csv = tmp_path / "manifest.csv"
    artifact_output_dir = tmp_path / "artifacts"
    summary_csv = tmp_path / "summary.csv"
    result = CliRunner().invoke(
        app,
        [
            "prepare-manifest",
            "--source-csv",
            str(source_csv),
            "--manifest-csv",
            str(manifest_csv),
            "--maps-dir",
            str(tmp_path),
            "--structures-dir",
            str(tmp_path),
            "--artifact-output-dir",
            str(artifact_output_dir),
            "--preprocess-adapter",
            "default",
            "--summary-csv",
            str(summary_csv),
        ],
    )
    assert result.exit_code == 0, result.output
    result = validate_preprocessed_artifact(artifact_output_dir / "emd_1234_1abc")
    assert (result.protein_nodes, result.nucleotide_nodes, result.component_nodes) == (
        1,
        1,
        1,
    )
    with summary_csv.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert rows[0]["status"] == "processed"
    assert rows[0]["success"] == "true"
    assert rows[0]["component_interface_count"] == "1"


def test_default_adapter_writes_and_validates_s3_artifact_offline(
    raw_inputs, monkeypatch
):
    objects = {}
    endpoints = []

    class Client:
        def close(self):
            pass

        def upload_file(self, Filename, Bucket, Key):
            objects[(Bucket, Key)] = Path(Filename).read_bytes()

        def download_file(self, Bucket, Key, Filename):
            Path(Filename).write_bytes(objects[(Bucket, Key)])

        def get_object(self, Bucket, Key):
            payload = objects[(Bucket, Key)]
            return {"Body": io.BytesIO(payload), "ContentLength": len(payload)}

        def head_object(self, Bucket, Key):
            if (Bucket, Key) not in objects:
                raise FileNotFoundError(Key)
            return {"ContentLength": len(objects[(Bucket, Key)])}

    def client(endpoint=None):
        endpoints.append(endpoint)
        return Client()

    monkeypatch.setattr(transports, "_get_s3_client", client)
    artifact_reference = "s3://bucket/artifacts/density"
    endpoint = "https://s3.example.test"
    default.run(
        map_path=raw_inputs[0],
        structure_path=raw_inputs[1],
        artifact_reference=artifact_reference,
        s3_endpoint_url=endpoint,
    )
    assert set(objects) == {
        ("bucket", "artifacts/density/core.npz"),
        ("bucket", "artifacts/density/metadata.json"),
    }
    result = validate_preprocessed_artifact(
        artifact_reference, s3_endpoint_url=endpoint
    )
    assert result.grid_shape == (128, 128, 128)
    assert result.protein_nodes == 1
    assert result.nucleotide_nodes == 1
    assert result.component_nodes == 1
    assert endpoints and all(selected == endpoint for selected in endpoints)


def test_origin_correction_handles_nucleotide_only_structures(raw_inputs, tmp_path):
    map_path, structure_path = raw_inputs
    shift = np.asarray([1, 2, 3])
    _write_structure(structure_path, shift=shift, ligand=False)
    original_structure_path = tmp_path / "original.cif"
    _write_structure(original_structure_path, ligand=False)
    for path in (structure_path, original_structure_path):
        path.write_text(
            "\n".join(
                line
                for line in path.read_text().splitlines()
                if not (line.startswith("ATOM") and "ALA" in line)
            )
            + "\n"
        )
    artifact_reference = tmp_path / "artifact"
    default.run(
        map_path=map_path,
        structure_path=structure_path,
        original_structure_path=original_structure_path,
        artifact_reference=artifact_reference,
    )
    payload = load_preprocessed_artifact(artifact_reference)
    np.testing.assert_array_equal(payload["global_origin"], [-63, -64, -65])
    np.testing.assert_array_equal(payload["backbone_coordinates"]["P"], [[3, 3, 3]])
    assert payload["full_atom_targets"]["prot_mask"].tolist() == [False]
    assert payload["extension_targets"]["component"]["positions"].shape == (0, 3)
    result = validate_preprocessed_artifact(artifact_reference)
    assert result.protein_nodes == 0
    assert result.nucleotide_nodes == 1
