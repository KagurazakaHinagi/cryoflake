"""Small real inputs and offline adapters for executable documentation."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import socket
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from typer.testing import CliRunner

from cryoflake.cli import app
from cryoflake.constants import FULL_ATOM_COUNT, RIGID_GROUP_COUNT, TORSION_COUNT
from cryoflake.io.mrc import (
    load_mrc,
    normalized_grid_size,
    rescale_real,
    save_mrc,
)
from cryoflake.io.structure import load_full_backbone_from_structure
from cryoflake.manifests import download, manifest
from cryoflake.storage import transports
from cryoflake.storage.store import write_preprocessed_artifact
from cryoflake.targets.components import build_component_extension_targets
from cryoflake.targets.full_atom import validate_full_atom_targets

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CIF = """data_documentation
loop_
_entity.id
_entity.type
1 polymer
2 polymer
3 non-polymer
loop_
_entity_poly.entity_id
_entity_poly.type
1 polypeptide(L)
2 polydeoxyribonucleotide
loop_
_chem_comp.id
_chem_comp.type
ALA 'L-PEPTIDE LINKING'
DA 'DNA LINKING'
{ligand} 'NON-POLYMER'
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
ATOM N . ALA A 1 1 A 10 ? 1 N 1.0 20.0 1.0 0.0 1.0
ATOM CA . ALA A 1 1 A 10 ? 1 C 1.0 20.0 1.0 1.0 1.0
ATOM C . ALA A 1 1 A 10 ? 1 C 1.0 20.0 2.0 1.0 1.0
ATOM O . ALA A 1 1 A 10 ? 1 O 1.0 20.0 2.0 2.0 1.0
ATOM P . DA B 2 1 B 20 ? 1 P 1.0 20.0 2.0 2.0 2.0
HETATM C1 . {ligand} C 3 . C 30 ? 1 C 1.0 20.0 1.5 1.0 1.0
HETATM O1 . {ligand} C 3 . C 30 ? 1 O 1.0 20.0 1.5 2.0 1.0
"""


def _full_atom_targets():
    """Label both nodes and explicitly mask unavailable detailed geometry."""
    targets = {
        "aatype": np.asarray([0, 24], dtype=np.int64),
        "prot_mask": np.asarray([True, False]),
        "chain_index": np.asarray([0, 1], dtype=np.int64),
        "residue_index": np.asarray([10, 20], dtype=np.int64),
    }
    shapes = {
        "atom_positions": (FULL_ATOM_COUNT, 3),
        "atom_mask": (FULL_ATOM_COUNT,),
        "b_factors": (FULL_ATOM_COUNT,),
        "rigidgroups_gt_frames": (RIGID_GROUP_COUNT, 3, 4),
        "rigidgroups_gt_exists": (RIGID_GROUP_COUNT,),
        "rigidgroups_group_exists": (RIGID_GROUP_COUNT,),
        "rigidgroups_group_is_ambiguous": (RIGID_GROUP_COUNT,),
        "rigidgroups_alt_gt_frames": (RIGID_GROUP_COUNT, 3, 4),
        "torsion_angles_sin_cos": (TORSION_COUNT, 2),
        "alt_torsion_angles_sin_cos": (TORSION_COUNT, 2),
        "torsion_angles_mask": (TORSION_COUNT,),
    }
    targets.update(
        {
            name: np.zeros((2, *shape), dtype=np.float32)
            for name, shape in shapes.items()
        }
    )
    return validate_full_atom_targets(targets, num_protein=1, num_nucleotide=1)


class _MissingObject(Exception):
    response = {
        "Error": {"Code": "NoSuchKey"},
        "ResponseMetadata": {"HTTPStatusCode": 404},
    }


class _MemoryS3:
    """Implement the SDK operations exercised by examples without credentials."""

    def __init__(self):
        self.objects = {}
        self.calls = []
        self.close_count = 0
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            self.close_count += 1

    def _read(self, operation, bucket, key):
        with self._lock:
            self.calls.append((operation, bucket, key))
            try:
                return self.objects[bucket, key]
            except KeyError as exc:
                raise _MissingObject() from exc

    def get_object(self, Bucket, Key):
        data = self._read("get", Bucket, Key)
        return {"Body": io.BytesIO(data), "ContentLength": len(data)}

    def head_object(self, Bucket, Key):
        data = self._read("head", Bucket, Key)
        return {
            "ContentLength": len(data),
            "ETag": f'"{hashlib.sha256(data).hexdigest()}"',
        }

    def download_file(self, Bucket, Key, Filename):
        Path(Filename).write_bytes(self._read("download", Bucket, Key))

    def upload_file(self, Filename, Bucket, Key):
        data = Path(Filename).read_bytes()
        with self._lock:
            self.calls.append(("upload", Bucket, Key))
            self.objects[Bucket, Key] = data


@dataclass
class _DocWorkspace:
    root: Path
    runner: CliRunner = field(default_factory=CliRunner)
    adapter_calls: list = field(default_factory=list)
    generator_calls: list = field(default_factory=list)
    http_calls: list = field(default_factory=list)
    s3: _MemoryS3 = field(default_factory=_MemoryS3)
    _calls_lock: threading.Lock = field(default_factory=threading.Lock)

    def invoke(self, args, *, check=True):
        result = self.runner.invoke(app, list(args))
        if check:
            assert result.exit_code == 0, (
                f"cryoflake {' '.join(map(str, args))}\n{result.output}\n"
                f"{result.exception!r}"
            )
        return result

    def _payload(self, mrc_path, structure_path, voxel_size, cutoff):
        volume = load_mrc(mrc_path)
        grid = volume.grid
        if voxel_size != volume.voxel_size:
            size = normalized_grid_size(len(grid), volume.voxel_size, voxel_size)
            grid = rescale_real(grid, size).astype(np.float32)
        coordinates, residue_info = load_full_backbone_from_structure(structure_path)
        full_atom_targets = _full_atom_targets()
        return {
            "density_grid": grid,
            "global_origin": volume.global_origin,
            "voxel_size": voxel_size,
            "backbone_coordinates": coordinates,
            "residue_info": residue_info,
            "full_atom_targets": full_atom_targets,
            "extension_targets": build_component_extension_targets(
                structure_path,
                global_origin=volume.global_origin,
                voxel_size=voxel_size,
                grid_shape=grid.shape,
                polymer_atom_positions=np.concatenate(
                    (coordinates["CA"], coordinates["P"])
                ),
                polymer_atom_node_indices=np.arange(2, dtype=np.int64),
                polymer_chain_indices=full_atom_targets["chain_index"],
                polymer_residue_indices=full_atom_targets["residue_index"],
                polymer_aatype=full_atom_targets["aatype"],
                interface_contact_cutoff=cutoff,
            ),
        }

    def run_adapter(
        self,
        *,
        map_path,
        structure_path,
        artifact_reference,
        target_voxel_size=1.0,
        original_structure_path=None,
        fasta_path=None,
        fix_shift=True,
        interface_contact_cutoff=5.0,
        s3_endpoint_url=None,
    ):
        arguments = {
            "map_path": map_path,
            "structure_path": structure_path,
            "artifact_reference": artifact_reference,
            "target_voxel_size": target_voxel_size,
            "original_structure_path": original_structure_path,
            "fasta_path": fasta_path,
            "fix_shift": fix_shift,
            "interface_contact_cutoff": interface_contact_cutoff,
            "s3_endpoint_url": s3_endpoint_url,
        }
        with self._calls_lock:
            self.adapter_calls.append(arguments)
        payload = self._payload(
            map_path, structure_path, target_voxel_size, interface_contact_cutoff
        )
        write_preprocessed_artifact(
            artifact_reference, payload, s3_endpoint_url=s3_endpoint_url
        )

    def generate_embeddings(
        self,
        *,
        structures_dir,
        sequence_embeddings_dir,
        device,
        max_chain_length,
        num_workers,
    ):
        self.generator_calls.append(
            {
                "structures_dir": structures_dir,
                "sequence_embeddings_dir": sequence_embeddings_dir,
                "device": device,
                "max_chain_length": max_chain_length,
                "num_workers": num_workers,
            }
        )
        output = Path(sequence_embeddings_dir)
        output.mkdir(parents=True, exist_ok=True)
        for source in sorted(Path(structures_dir).glob("*.cif")):
            np.savez(
                output / f"{source.stem}.npz",
                sequence=np.ones((2, 8), dtype=np.float32),
                sequence_mask=np.ones(2, dtype=np.bool_),
            )

    def seed_training(self, artifact_reference, *, include_sequence=True):
        payload = self._payload(
            self.root / "data/map.mrc", self.root / "data/model.cif", 1.0, 5.0
        )
        if include_sequence:
            payload.update(
                sequence=np.ones((2, 8), dtype=np.float32),
                sequence_mask=np.ones(2, dtype=np.bool_),
            )
        return write_preprocessed_artifact(artifact_reference, payload)

    def prepare_manifest(self):
        output = self.root / "data/manifest.csv"
        manifest.write_manifest_csv(
            manifest.build_manifest(self.root / "data/source.csv"),
            output,
            maps_dir=self.root / "data/maps",
            structures_dir=self.root / "data/structures",
        )
        return output

    def urlopen(self, url, *args, **kwargs):
        url = str(url)
        with self._calls_lock:
            self.http_calls.append(url)
        name = url.rsplit("/", 1)[-1]
        if name in {"emd_1234.map.gz", "emd_5678.map.gz"}:
            return io.BytesIO(gzip.compress((self.root / "data/map.mrc").read_bytes()))
        if name in {"1abc.cif.gz", "2def.cif.gz"}:
            structure = self.root / "data/structures" / name.removesuffix(".gz")
            return io.BytesIO(gzip.compress(structure.read_bytes()))
        if "/api/entry/fitted/" in url and name in {"1234", "5678"}:
            pdb_id = {"1234": "1abc", "5678": "2def"}[name]
            return io.BytesIO(
                json.dumps(
                    {
                        "crossreferences": {
                            "pdb_list": {"pdb_reference": {"pdb_id": pdb_id}}
                        }
                    }
                ).encode()
            )
        raise AssertionError(f"Unstubbed documentation HTTP request: {url}")


@pytest.fixture
def doc_workspace(tmp_path, monkeypatch):
    """Run each guide scenario with fresh raw data and no external I/O."""
    monkeypatch.chdir(tmp_path)
    for name in tuple(os.environ):
        if name.startswith(("AWS_", "CRYOFLAKE_")):
            monkeypatch.delenv(name)
    environment = tmp_path / "credentials.env"
    environment.write_text("")
    monkeypatch.setenv("CRYOFLAKE_ENV_FILE", str(environment))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    def no_network(*args, **kwargs):
        raise AssertionError(
            "Documentation tests must not access live network services"
        )

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    workspace = _DocWorkspace(tmp_path)
    monkeypatch.setattr(
        transports, "_get_s3_client", lambda endpoint=None: workspace.s3
    )
    monkeypatch.setattr(download, "urlopen", workspace.urlopen)
    monkeypatch.setattr(manifest, "urlopen", workspace.urlopen)

    parent = ModuleType("my_app")
    parent.__path__ = []
    adapter = ModuleType("my_app.preprocess")
    adapter.run = workspace.run_adapter
    generator = ModuleType("my_app.embeddings")
    generator.generate = workspace.generate_embeddings
    parent.preprocess = adapter
    parent.embeddings = generator
    for module in (parent, adapter, generator):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    data = tmp_path / "data"
    (data / "maps").mkdir(parents=True)
    (data / "structures").mkdir()
    (data / "source.csv").write_text(
        "EMD-,Associated PDB,resolution\nEMD-1234,1ABC,3.2\nEMD-5678,2DEF,2.8\n"
    )
    save_mrc(
        np.arange(64, dtype=np.float32).reshape(4, 4, 4),
        1.0,
        np.zeros(3, dtype=np.float32),
        data / "map.mrc",
    )
    (data / "model.cif").write_text(_CIF.format(ligand="LIG"))
    for emdb_id, pdb_id, ligand in (("1234", "1abc", "LIG"), ("5678", "2def", "ATP")):
        shutil.copyfile(data / "map.mrc", data / "maps" / f"emd_{emdb_id}.map")
        (data / "structures" / f"{pdb_id}.cif").write_text(_CIF.format(ligand=ligand))
    shutil.copytree(_PROJECT_ROOT / "configs", tmp_path / "configs")
    workspace.seed_training(data / "artifacts/example")
    workspace.seed_training("s3://my-bucket/artifacts/example")
    workspace.s3.calls.clear()
    workspace.s3.close_count = 0
    initial_threads = {thread.ident for thread in threading.enumerate()}
    yield workspace
    leaked_uploads = [
        thread
        for thread in threading.enumerate()
        if thread.ident not in initial_threads
        and thread.name.startswith("cryoflake-upload")
    ]
    assert not leaked_uploads, (
        f"Documentation example leaked upload threads: {leaked_uploads}"
    )
