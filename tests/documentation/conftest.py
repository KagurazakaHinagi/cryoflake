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
from standard_target import (
    load_standard_artifact_mapping,
    load_standard_full_atom_targets,
    standard_expected,
    write_standard_structure,
)
from typer.testing import CliRunner

from cryoflake.cli import app
from cryoflake.io.mrc import load_mrc, save_mrc
from cryoflake.io.structure import load_full_backbone_from_structure
from cryoflake.manifests import download, manifest
from cryoflake.preprocess import worker
from cryoflake.preprocess.memory_guard import estimate_preprocess_memory_or_none
from cryoflake.storage import transports
from cryoflake.storage.store import write_preprocessed_artifact
from cryoflake.targets.components import build_component_extension_targets

EXPECTED = standard_expected()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

    def _payload(self, mrc_path, structure_path, target_voxel_size, cutoff):
        # The injected documentation adapter records the requested spacing but
        # retains the compact fixture's real sampling; resampling has separate
        # primitive coverage and must not expand every guide case to 352 cubed.
        volume = load_mrc(mrc_path)
        coordinates, residue_info = load_full_backbone_from_structure(structure_path)
        full_atom_targets = load_standard_full_atom_targets()
        atom_mask = full_atom_targets["atom_mask"]
        node_indices = np.broadcast_to(
            np.arange(len(atom_mask))[:, None], atom_mask.shape
        )[atom_mask]
        return {
            "density_grid": volume.grid,
            "global_origin": volume.global_origin,
            "voxel_size": volume.voxel_size,
            "backbone_coordinates": coordinates,
            "residue_info": residue_info,
            "full_atom_targets": full_atom_targets,
            "extension_targets": build_component_extension_targets(
                structure_path,
                global_origin=volume.global_origin,
                voxel_size=volume.voxel_size,
                grid_shape=volume.grid.shape,
                polymer_atom_positions=full_atom_targets["atom_positions"][atom_mask],
                polymer_atom_node_indices=node_indices,
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
                sequence=np.ones(
                    (EXPECTED["full_atom_residue_count"], 8), dtype=np.float32
                ),
                sequence_mask=np.ones(
                    EXPECTED["full_atom_residue_count"], dtype=np.bool_
                ),
            )

    def seed_training(self, artifact_reference, *, include_sequence=False):
        payload = load_standard_artifact_mapping()
        if include_sequence:
            # Synthetic embeddings exist only in explicit sequence scenarios.
            residue_count = EXPECTED["full_atom_residue_count"]
            payload.update(
                sequence=np.ones((residue_count, 8), dtype=np.float32),
                sequence_mask=np.ones(residue_count, dtype=np.bool_),
            )
        return write_preprocessed_artifact(artifact_reference, payload)

    def use_independent_component_groups(self):
        # Two explicitly filtered copies of 7AE1 exercise grouped splitting;
        # placeholder route names do not identify additional depositions.
        write_standard_structure(
            self.root / "data/structures/1abc.cif", component_ids={"ZN"}
        )
        write_standard_structure(
            self.root / "data/structures/2def.cif", component_ids={"SF4"}
        )

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

    def estimate_documentation_workload(map_path, target_voxel_size):
        # Match the injected adapter's retained sampling rather than estimating
        # an interpolation the test double does not perform. Admission and
        # release continue through the real worker memory guard.
        return estimate_preprocess_memory_or_none(map_path, EXPECTED["voxel_size"])

    monkeypatch.setattr(
        worker, "estimate_preprocess_memory_or_none", estimate_documentation_workload
    )

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
    standard = load_standard_artifact_mapping()
    save_mrc(
        standard["density_grid"],
        standard["voxel_size"],
        standard["global_origin"],
        data / "map.mrc",
    )
    write_standard_structure(data / "model.cif")
    # The guides keep illustrative routing IDs. Both routes use explicitly
    # duplicated canonical inputs, without claiming two real depositions.
    for emdb_id, pdb_id in (("1234", "1abc"), ("5678", "2def")):
        shutil.copyfile(data / "map.mrc", data / "maps" / f"emd_{emdb_id}.map")
        write_standard_structure(data / "structures" / f"{pdb_id}.cif")
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
