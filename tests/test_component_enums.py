import ast
from pathlib import Path

from cryoflake.constants import AnchorKind, NodeKind, PolymerKind


def test_component_enum_values_match_the_serialized_contract():
    assert {member.name: member.value for member in NodeKind} == {
        "POLYMER": 0,
        "NON_POLYMER": 1,
        "ION": 2,
        "WATER": 3,
        "UNKNOWN": 4,
    }
    assert {member.name: member.value for member in PolymerKind} == {
        "NONE": 0,
        "PROTEIN": 1,
        "RNA": 2,
        "DNA": 3,
        "NUCLEIC_ACID": 4,
        "CARBOHYDRATE": 5,
        "OTHER": 6,
    }
    assert {member.name: member.value for member in AnchorKind} == {
        "UNSPECIFIED": 0,
        "C_ALPHA": 1,
        "PHOSPHORUS": 2,
        "ATOM": 3,
        "CENTROID": 4,
    }


def test_portable_target_modules_do_not_import_model_stack():
    source_root = Path(__file__).parents[1] / "src" / "cryoflake"
    target_modules = (
        "targets/common.py",
        "targets/parsers.py",
        "targets/schema.py",
        "constants.py",
        "targets/components.py",
        "targets/full_atom.py",
        "io/structure.py",
        "io/fasta.py",
    )

    forbidden = {"torch", "Bio"}
    for module_name in target_modules:
        tree = ast.parse((source_root / module_name).read_text())
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        assert imported_roots.isdisjoint(forbidden), module_name
