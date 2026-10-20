"""Shared chemistry vocabularies and the serialized target format contract.

The numeric values are part of the serialized artifact and model-checkpoint
contract. Add new values only at the end of an enum; never reorder or reuse an
existing value.
"""

from enum import IntEnum

from cryoflake.config import target_contract


class NodeKind(IntEnum):
    """Coarse node representation used by the shared geometric trunk."""

    POLYMER = 0
    NON_POLYMER = 1
    ION = 2
    WATER = 3
    UNKNOWN = 4


class PolymerKind(IntEnum):
    """Polymer chemistry without tying the schema to residue vocabularies."""

    NONE = 0
    PROTEIN = 1
    RNA = 2
    DNA = 3
    NUCLEIC_ACID = 4
    CARBOHYDRATE = 5
    OTHER = 6


class AnchorKind(IntEnum):
    """Geometric anchor represented by a structural node."""

    UNSPECIFIED = 0
    C_ALPHA = 1
    PHOSPHORUS = 2
    ATOM = 3
    CENTROID = 4


PREPROCESSED_PROTEIN_RESIDUE_LABELS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
}
PREPROCESSED_NUCLEOTIDE_RESIDUE_LABELS = {
    "DA",
    "DC",
    "DG",
    "DT",
    "A",
    "C",
    "G",
    "U",
    "DN",
    "N",
}


CANONICAL_COMPONENT_IDS = (
    PREPROCESSED_PROTEIN_RESIDUE_LABELS | PREPROCESSED_NUCLEOTIDE_RESIDUE_LABELS
)
BACKBONE_ATOM_NAMES = ("CA", "C", "N", "O")

WATER_COMPONENT_IDS = {"DOD", "H2O", "HOH", "WAT"}
UNKNOWN_COMPONENT_IDS = {"UNL", "UNX"}
ION_ELEMENTS = {
    "AL",
    "BA",
    "BE",
    "BR",
    "CA",
    "CD",
    "CE",
    "CL",
    "CO",
    "CR",
    "CS",
    "CU",
    "DY",
    "ER",
    "EU",
    "F",
    "FE",
    "GA",
    "GD",
    "HG",
    "HO",
    "IN",
    "IR",
    "K",
    "LA",
    "LI",
    "LU",
    "MG",
    "MN",
    "MO",
    "NA",
    "ND",
    "NI",
    "OS",
    "PB",
    "PD",
    "PR",
    "PT",
    "RB",
    "RE",
    "RH",
    "RU",
    "SB",
    "SC",
    "SE",
    "SM",
    "SN",
    "SR",
    "TB",
    "TC",
    "TI",
    "TL",
    "TM",
    "V",
    "W",
    "Y",
    "YB",
    "ZN",
    "ZR",
}
COMMON_CARBOHYDRATE_IDS = {
    "A2G",
    "BGC",
    "BMA",
    "FUC",
    "GAL",
    "GLC",
    "MAN",
    "NAG",
    "NDG",
    "SIA",
}


_TARGET_CONTRACT = target_contract()
FULL_ATOM_COUNT = _TARGET_CONTRACT.full_atom_count
RIGID_GROUP_COUNT = _TARGET_CONTRACT.rigid_group_count
TORSION_COUNT = _TARGET_CONTRACT.torsion_count
UNKNOWN_PROTEIN_INDEX = _TARGET_CONTRACT.unknown_protein_index
FULL_RESIDUE_TYPE_COUNT = _TARGET_CONTRACT.full_residue_type_count

FULL_ATOM_TARGET_KEYS = (
    "aatype",
    "prot_mask",
    "atom_positions",
    "atom_mask",
    "b_factors",
    "rigidgroups_gt_frames",
    "rigidgroups_gt_exists",
    "rigidgroups_group_exists",
    "rigidgroups_group_is_ambiguous",
    "rigidgroups_alt_gt_frames",
    "torsion_angles_sin_cos",
    "alt_torsion_angles_sin_cos",
    "torsion_angles_mask",
)
FULL_ATOM_TOPOLOGY_KEYS = ("chain_index", "residue_index")


COMPONENT_EXTENSION_SCHEMA_VERSION = _TARGET_CONTRACT.schema_version
