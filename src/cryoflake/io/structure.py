"""Biotite structure readers and named-column secondary-structure parsing."""

from __future__ import annotations

import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
from biotite.structure import get_residue_starts
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile

from cryoflake.common import clean_token, integer_token
from cryoflake.constants import (
    BACKBONE_ATOM_NAMES,
    PREPROCESSED_NUCLEOTIDE_RESIDUE_LABELS,
    PREPROCESSED_PROTEIN_RESIDUE_LABELS,
)

Helix = namedtuple(
    "Helix", ["chain", "start_res", "end_chain", "end_res", "helix_type"]
)
SheetStrand = namedtuple("SheetStrand", ["chain", "start_res", "end_chain", "end_res"])
SecondaryStructure = namedtuple(
    "SecondaryStructure", ["chain", "start_res", "end_res", "type"]
)


def _read_cif_data(path: str | Path) -> dict[str, list[str]]:
    """Read the first CIF block using Biotite's quoted/multiline-aware parser."""
    source = pdbx.CIFFile.read(path)
    if not source:
        raise ValueError(f"CIF file contains no data blocks: {path!s}")
    block = source[next(iter(source))]
    return {
        f"_{category_name}.{column_name}": column.as_array(str).tolist()
        for category_name, category in block.items()
        for column_name, column in category.items()
    }


def _cif_column(data, key: str, length: int, default: str = "") -> list[str]:
    values = list(data.get(key, ()))
    if not values:
        return [default] * length
    if len(values) != length:
        raise ValueError(f"mmCIF column {key} has an inconsistent length")
    return values


def _secondary_ranges(data, category: str):
    """Yield chain/residue ranges with per-row author-to-label fallback."""
    prefix = f"_{category}."
    lengths = [len(value) for key, value in data.items() if key.startswith(prefix)]
    if not lengths:
        return
    count = lengths[0]
    columns = {
        f"{end}_{namespace}_{field}": _cif_column(
            data, f"{prefix}{end}_{namespace}_{field}", count
        )
        for end in ("beg", "end")
        for namespace in ("auth", "label")
        for field in ("asym_id", "seq_id")
    }
    for index in range(count):
        chains = [
            clean_token(columns[f"{end}_auth_asym_id"][index])
            or clean_token(columns[f"{end}_label_asym_id"][index])
            for end in ("beg", "end")
        ]
        residue_tokens = [
            clean_token(columns[f"{end}_auth_seq_id"][index])
            or clean_token(columns[f"{end}_label_seq_id"][index])
            for end in ("beg", "end")
        ]
        try:
            start_res, end_res = map(int, residue_tokens)
        except ValueError:
            continue
        yield index, chains[0], chains[1], start_res, end_res


class CIFSecondaryStructures:
    """Read deposited mmCIF helix/sheet ranges while retaining stored labels."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = file_path
        self.helices = []
        self.sheets = []
        self.residue_to_type: dict[str, dict[int, str]] = {}
        self.parse_cif_file()

    def add_helix(self, start_chain, end_chain, start_res, end_res, helix_type):
        if start_chain != end_chain:
            return
        self.helices.append(
            Helix(start_chain, start_res, end_chain, end_res, helix_type)
        )
        records = self.residue_to_type.setdefault(start_chain, {})
        for residue in range(start_res, end_res + 1):
            records[residue] = f"HELIX_{helix_type}"

    def add_sheet_strand(self, start_chain, end_chain, start_res, end_res):
        if start_chain != end_chain:
            return
        self.sheets.append(SheetStrand(start_chain, start_res, end_chain, end_res))
        records = self.residue_to_type.setdefault(start_chain, {})
        for residue in range(start_res, end_res + 1):
            records[residue] = "SHEET"

    def parse_cif_file(self) -> None:
        if Path(self.file_path).suffix.lower() not in {".cif", ".mmcif"}:
            return
        data = _read_cif_data(self.file_path)
        types = data.get("_struct_conf.conf_type_id", ())
        classes = data.get("_struct_conf.pdbx_PDB_helix_class", ())
        for index, start_chain, end_chain, start_res, end_res in _secondary_ranges(
            data, "struct_conf"
        ):
            if index >= len(types) or not types[index].upper().startswith("HELX"):
                continue
            helix_class = clean_token(classes[index]) if index < len(classes) else ""
            self.add_helix(
                start_chain, end_chain, start_res, end_res, helix_class or "1"
            )
        for _, start_chain, end_chain, start_res, end_res in _secondary_ranges(
            data, "struct_sheet_range"
        ):
            self.add_sheet_strand(start_chain, end_chain, start_res, end_res)

    def get_residue_secondary_structure(self, chain_id: str, residue_index: int) -> str:
        return self.residue_to_type.get(chain_id, {}).get(residue_index, "NULL")


class DSSPSecondaryStructures:
    """Read DSSP mmCIF struct_conf type identifiers without fixed column order."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = file_path
        self.residue_to_type: dict[str, dict[int, str]] = {}
        self.structures = []
        self.parse_cif_file()

    def parse_cif_file(self) -> None:
        if Path(self.file_path).suffix.lower() not in {".cif", ".mmcif"}:
            return
        data = _read_cif_data(self.file_path)
        types = data.get("_struct_conf.conf_type_id", ())
        for index, start_chain, end_chain, start_res, end_res in _secondary_ranges(
            data, "struct_conf"
        ):
            if start_chain != end_chain or index >= len(types):
                continue
            structure = SecondaryStructure(
                start_chain, start_res, end_res, types[index]
            )
            self.structures.append(structure)
            records = self.residue_to_type.setdefault(start_chain, {})
            for residue in range(start_res, end_res + 1):
                records[residue] = structure.type

    def get_residue_secondary_structure(self, chain_id: str, residue_index: int) -> str:
        return self.residue_to_type.get(chain_id, {}).get(residue_index, "OTHER")


def _structure_models(path: str | Path, *, all_models: bool):
    suffix = Path(path).suffix.lower()
    if suffix in {".pdb", ".ent"}:
        source = PDBFile.read(path)
        model_count = source.get_model_count()
        reader = source.get_structure
    elif suffix in {".cif", ".mmcif"}:
        source = pdbx.CIFFile.read(path)
        if not source:
            raise ValueError(f"Structure contains no models: {path!s}")
        block = source[next(iter(source))]
        atom_site = block.get("atom_site")
        if atom_site is None or atom_site.row_count == 0:
            raise ValueError(f"Structure contains no models: {path!s}")
        if "pdbx_PDB_model_num" not in atom_site:
            atom_site["pdbx_PDB_model_num"] = ["1"] * atom_site.row_count
        # The source contract uses author chain/residue IDs but label chemistry.
        for field in ("atom_id", "comp_id"):
            if f"label_{field}" in atom_site:
                atom_site[f"auth_{field}"] = atom_site[f"label_{field}"]
        for field in ("asym_id", "seq_id"):
            author = atom_site.get(f"auth_{field}")
            label = atom_site.get(f"label_{field}")
            if author is not None and label is not None:
                atom_site[f"auth_{field}"] = [
                    clean_token(a) or clean_token(b)
                    for a, b in zip(author.as_array(str), label.as_array(str))
                ]
        model_count = pdbx.get_model_count(block)

        def reader(**kwargs):
            return pdbx.get_structure(block, use_author_fields=True, **kwargs)
    else:
        raise ValueError(f"Unsupported structure file extension: {suffix!r}")
    if not model_count:
        raise ValueError(f"Structure contains no models: {path!s}")
    for model in range(1, (model_count if all_models else 1) + 1):
        # Read separately: models can have different atoms and residue topology.
        atoms = reader(model=model, altloc="all")
        if suffix in {".pdb", ".ent"}:
            # Biotite strips the PDB blank-chain identifier; retain stored labels.
            atoms.chain_id = np.where(atoms.chain_id == "", " ", atoms.chain_id)
        yield atoms


def _backbone_residues(model):
    """Group author residue identities in chain order, including hetero identity.

    AtomArray residue boundaries alone merge ATOM/HETATM records that share a
    residue number, and split chains whose records are discontinuous. Retain the
    established chain/residue traversal and the last deposited variant of a
    disordered polymer residue, independently of per-atom altloc selection.
    """
    chains = {}
    starts = get_residue_starts(
        model, add_exclusive_stop=True, extra_categories=("hetero", "res_name")
    )
    for start, stop in zip(starts[:-1], starts[1:]):
        chain = chains.setdefault(str(model.chain_id[start]), {})
        name = str(model.res_name[start])
        key = (
            name if model.hetero[start] else "",
            int(model.res_id[start]),
            str(model.ins_code[start]),
        )
        variants = chain.setdefault(key, {})
        if name not in variants and variants and len(variants) == 1:
            first_indices = next(iter(variants.values()))
            if not all(clean_token(model.altloc_id[i]) for i in first_indices):
                # A duplicate residue with blank altlocs is malformed. Match the
                # permissive reader's treatment by retaining its original row.
                continue
            # Creating a disordered residue appends it to the chain's records.
            chain.pop(key)
            chain[key] = variants
        variants.setdefault(name, []).extend(range(start, stop))
        # Repeated variants select the one most recently observed in the file.
        variants[name] = variants.pop(name)
    for chain in chains.values():
        for variants in chain.values():
            yield model[next(reversed(variants.values()))]


def _atom_position(residue, name: str) -> np.ndarray:
    indices = np.flatnonzero(residue.atom_name == name)
    if not len(indices):
        return np.full(3, np.nan, dtype=np.float32)
    # Preserve the original backbone policy: first altloc by identifier, per atom.
    # Component targets instead select highest occupancy (see targets.common).
    chosen = min(indices, key=lambda index: (str(residue.altloc_id[index]), int(index)))
    return np.asarray(residue.coord[chosen], dtype=np.float32)


def load_full_backbone_from_structure(
    structure_path: str | Path,
    *,
    all_models: bool = False,
    quiet: bool = True,
    dssp_secondary_structures: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Read canonical protein backbone and nucleotide-P coordinates with Biotite.

    The first model is used by default; each model is parsed independently when
    requested. Missing atoms become NaN coordinates. Residue vocabularies and
    secondary-structure strings retain the serialized dataset contract.
    """
    coordinates: dict[str, list[np.ndarray]] = {
        name: [] for name in (*BACKBONE_ATOM_NAMES, "P")
    }
    residue_info: dict[str, list[str]] = {
        "aa_type": [],
        "secondary_structure": [],
        "chain": [],
        "nuc_type": [],
    }
    secondary_structures = (
        DSSPSecondaryStructures(structure_path)
        if dssp_secondary_structures
        else CIFSecondaryStructures(structure_path)
    )
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore", UserWarning)
        for model in _structure_models(structure_path, all_models=all_models):
            for residue in _backbone_residues(model):
                residue_name = str(residue.res_name[0]).strip().upper()
                if residue_name in PREPROCESSED_PROTEIN_RESIDUE_LABELS:
                    residue_info["aa_type"].append(residue_name)
                    for atom_name in BACKBONE_ATOM_NAMES:
                        coordinates[atom_name].append(
                            _atom_position(residue, atom_name)
                        )
                elif residue_name in PREPROCESSED_NUCLEOTIDE_RESIDUE_LABELS:
                    residue_info["nuc_type"].append(residue_name)
                    coordinates["P"].append(_atom_position(residue, "P"))
                else:
                    continue
                chain_id = str(residue.chain_id[0])
                residue_info["chain"].append(chain_id)
                residue_info["secondary_structure"].append(
                    secondary_structures.get_residue_secondary_structure(
                        chain_id, integer_token(residue.res_id[0])
                    )
                )
    return (
        {
            key: np.asarray(values, dtype=np.float32).reshape((-1, 3))
            for key, values in coordinates.items()
        },
        residue_info,
    )
