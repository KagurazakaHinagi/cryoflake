"""Fresh offline test payloads derived from the real EMD-11736 / 7AE1 entry.

The compact target retains deposited atom identities/coordinates and cached
NextAngelo geometry. See the fixture provenance for the exact selection,
Fourier-derived density, and upstream torsion-mask limitation.
"""

from __future__ import annotations

import gzip
import io
import json
from copy import deepcopy
from functools import cache
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx

EMD_ID = "11736"
PDB_ID = "7ae1"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "emd_11736_7ae1"
PROVENANCE_PATH = FIXTURE_DIR / "provenance.json"
FULL_STRUCTURE_GZIP_PATH = FIXTURE_DIR / "7ae1.cif.gz"
FULL_ENTRY_TARGETS_PATH = FIXTURE_DIR / "full_entry_targets.npz"
STANDARD_STRUCTURE_GZIP_PATH = FIXTURE_DIR / "subset.cif.gz"
_ARTIFACT_ROOTS = {
    "density_grid",
    "global_origin",
    "voxel_size",
    "backbone_coordinates",
    "residue_info",
    "full_atom_targets",
    "extension_targets",
}


@cache
def _provenance() -> dict:
    return json.loads(PROVENANCE_PATH.read_text())


def standard_expected() -> dict:
    """Return independently recorded counts, geometry and named atom samples."""
    expected = deepcopy(_provenance()["expected"])
    expected["grid_shape"] = tuple(expected["grid_shape"])
    return expected


@cache
def _standard_arrays() -> dict[str, np.ndarray]:
    with np.load(FIXTURE_DIR / "standard_arrays.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


@cache
def _standard_mapping() -> dict:
    mapping = {}
    for path, array in _standard_arrays().items():
        fields = path.split("/")
        if fields[0] not in _ARTIFACT_ROOTS:
            continue
        node = mapping
        for field in fields[:-1]:
            node = node.setdefault(field, {})
        if fields[0] == "residue_info":
            value = array.tolist()
        elif array.ndim == 0:
            value = array.item()
        else:
            value = array.copy()
        node[fields[-1]] = value
    return mapping


def load_standard_artifact_mapping() -> dict:
    """Return a fresh mutable complete training mapping, without embeddings."""
    return deepcopy(_standard_mapping())


def load_standard_full_atom_targets() -> dict:
    """Return fresh normalized cached supervision for the compact real subset."""
    return deepcopy(_standard_mapping()["full_atom_targets"])


def load_standard_source_metadata() -> dict[str, np.ndarray]:
    """Return original full-entry row identities for the compact subset."""
    return {
        name: array.copy()
        for name, array in _standard_arrays().items()
        if name.startswith("source_")
    }


def write_standard_structure(
    path: str | Path,
    *,
    component_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Write actual subset CIF records, optionally filtering genuine components.

    Canonical polymer rows are always retained. Filtering accepts only the
    deposited component IDs ZN, MG and SF4; it never renames chemistry.
    """
    path = Path(path)
    text = gzip.decompress(STANDARD_STRUCTURE_GZIP_PATH.read_bytes()).decode()
    if component_ids is None:
        path.write_text(text)
        return path
    if isinstance(component_ids, str) or any(
        not isinstance(name, str) for name in component_ids
    ):
        raise ValueError("component_ids must be a collection of genuine component IDs")
    requested = set(component_ids)
    known = set(_provenance()["expected"]["component_ids"])
    if not requested <= known:
        raise ValueError(f"Unknown 7AE1 component IDs: {sorted(requested - known)}")
    cif = pdbx.CIFFile.read(io.StringIO(text))
    block = next(iter(cif.values()))
    atom_site = block["atom_site"]
    names = atom_site["label_comp_id"].as_array(str)
    keep = ~np.isin(names, tuple(known - requested))
    block["atom_site"] = pdbx.CIFCategory(
        {name: column.as_array(str)[keep] for name, column in atom_site.items()}
    )
    cif.write(path)
    return path


def write_full_entry_structure(path: str | Path) -> Path:
    """Write the unchanged downloaded full-entry CIF for one integration check."""
    path = Path(path)
    path.write_bytes(gzip.decompress(FULL_STRUCTURE_GZIP_PATH.read_bytes()))
    return path


def load_full_entry_full_atom_targets() -> dict:
    """Load authentic full-entry cached targets with current normalization."""
    from types import SimpleNamespace

    from cryoflake.targets.full_atom import extract_full_atom_targets

    with np.load(FULL_ENTRY_TARGETS_PATH, allow_pickle=False) as archive:
        source = SimpleNamespace(**{name: archive[name] for name in archive.files})
    return extract_full_atom_targets(source)
