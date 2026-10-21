"""Execute the published storage examples and check the format reference tables."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cryoflake
from cryoflake import (
    FULL_ATOM_TARGET_SCHEMA_VERSION,
    FASTASequence,
    build_component_extension_targets,
    load_preprocessed_artifact,
    preprocessed_artifact_root,
    validate_component_extension_targets,
    validate_full_atom_targets,
    validate_preprocessed_artifact,
    write_preprocessed_artifact,
)
from cryoflake.config import compose_config
from cryoflake.constants import AnchorKind, NodeKind, PolymerKind
from cryoflake.manifests.manifest import read_source_associations
from cryoflake.manifests.splits import (
    SPLIT_REPORT_SCHEMA_VERSION,
    GroupedSplitRecord,
    read_holdout_group_ids,
    split_grouped_records,
    write_grouped_split_report,
)
from cryoflake.preprocess.artifact_identity import (
    PREPROCESSED_ARTIFACT_CONTRACT_VERSION,
)
from cryoflake.preprocess.checkpoint import PREPROCESS_CHECKPOINT_FORMAT_VERSION
from cryoflake.preprocess.summary import PREPROCESS_SUMMARY_FIELDNAMES
from cryoflake.sequence import validation as sequence_validation
from cryoflake.storage.codec import read_metadata
from cryoflake.storage.transports import is_remote_uri
from cryoflake.targets.full_atom import extract_full_atom_targets

from .examples import get_example

pytestmark = pytest.mark.documentation


def _reference_section(example_id: str, heading: str) -> str:
    """Find a named section without depending on a code block's position."""

    source = get_example(example_id).path.read_text(encoding="utf-8")
    level = len(heading) - len(heading.lstrip("#"))
    section = source.split(heading + "\n", maxsplit=1)[1]
    return re.split(rf"^#{{1,{level}}} ", section, maxsplit=1, flags=re.M)[0]


def _table_rows(section: str) -> list[list[str]]:
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section.splitlines()
        if line.startswith("| ")
    ]


def test_readme_storage_example(doc_workspace):
    namespace = get_example("readme-storage").execute()

    payload = namespace["payload"]
    assert set(payload) == {"features", "label"}
    assert payload["label"] == "example"
    assert payload["features"].dtype == np.float32
    np.testing.assert_array_equal(payload["features"], np.ones((4, 8)))
    assert (doc_workspace.root / "data/artifacts/example/metadata.json").is_file()


def test_api_storage_example(doc_workspace):
    namespace = get_example("api-storage").execute()

    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(namespace["selected"]["features"], features)
    payload = namespace["loaded"]
    assert set(payload) == {"features", "sequence", "sequence_mask"}
    assert payload["features"].dtype == np.float32
    np.testing.assert_array_equal(payload["features"], features)
    assert payload["sequence"].dtype == np.float32
    np.testing.assert_array_equal(payload["sequence"], np.zeros((3, 16)))
    assert payload["sequence_mask"].dtype == np.bool_
    np.testing.assert_array_equal(payload["sequence_mask"], [True] * 3)


def test_fasta_reading_and_portable_record_roundtrip_example():
    namespace = get_example("api-fasta").execute()
    restored = namespace["restored"]["fasta_sequences"]

    assert len(restored) == 1
    assert type(restored[0]) is FASTASequence
    assert restored[0].sequence == "ACD"
    assert restored[0].chain_count == 2
    assert restored[0].chain_ids == ["X", "Y"]
    assert namespace["label_sequences"][0].chain_ids == ["A", "B"]


def test_artifact_reference_examples_load_the_same_payload(doc_workspace, monkeypatch):
    doc_workspace.seed_training("artifacts/example")
    references = []
    payloads = []

    def load_and_record(reference):
        payload = load_preprocessed_artifact(reference)
        references.append(reference)
        payloads.append(payload)
        return payload

    monkeypatch.setattr(cryoflake, "load_preprocessed_artifact", load_and_record)
    get_example("artifact-references").execute()

    assert len(payloads) == 3
    root_reference, metadata_reference, remote_reference = references
    assert not is_remote_uri(root_reference)
    assert Path(root_reference).is_dir()
    assert Path(metadata_reference) == Path(root_reference) / "metadata.json"
    assert preprocessed_artifact_root(root_reference) == preprocessed_artifact_root(
        metadata_reference
    )
    assert str(remote_reference).startswith("s3://")
    assert preprocessed_artifact_root(remote_reference) == remote_reference
    for payload in payloads:
        assert payload.keys() == payloads[0].keys()
        np.testing.assert_array_equal(
            payload["density_grid"], payloads[0]["density_grid"]
        )
        np.testing.assert_array_equal(payload["sequence"], payloads[0]["sequence"])
        np.testing.assert_array_equal(
            payload["sequence_mask"], payloads[0]["sequence_mask"]
        )
    assert doc_workspace.s3.close_count > 0


def test_documented_artifact_layout_matches_split_artifact_members(doc_workspace):
    layout = get_example("artifact-layout").code
    documented_members = re.findall(r"^[├└]── ([^\s#]+)", layout, re.M)
    assert documented_members
    assert len(documented_members) == len(set(documented_members))
    artifact = doc_workspace.root / "data/artifacts/example"
    assert layout.splitlines()[0] == artifact.name + "/"
    assert set(documented_members) == {member.name for member in artifact.iterdir()}
    metadata = read_metadata(artifact / "metadata.json")
    assert set(documented_members) == {"metadata.json", *metadata["files"].values()}


def test_documented_payload_tree_matches_written_metadata(tmp_path):
    tree = json.loads(get_example("artifact-tree").code)
    root = tmp_path / "example"
    values = np.arange(6, dtype=np.dtype("<f4")).reshape(2, 3)
    write_preprocessed_artifact(root, {"values": values, "label": "example"})

    metadata = read_metadata(root / "metadata.json")
    assert metadata["tree"] == tree
    assert set(path.name for path in root.iterdir()) == {"metadata.json", "core.npz"}
    loaded = load_preprocessed_artifact(root)
    assert loaded["values"].dtype.str == tree["items"]["values"]["dtype"]
    np.testing.assert_array_equal(loaded["values"], values)


def test_artifact_roundtrip_example(doc_workspace):
    namespace = get_example("artifact-roundtrip").execute()

    payload = namespace["loaded"]
    assert set(payload) == {"values", "label", "notes", "sequence", "sequence_mask"}
    assert payload["values"].dtype == np.float32
    assert payload["label"] == "example"
    assert type(payload["notes"]) is tuple
    assert type(payload["notes"][1]) is bytes
    np.testing.assert_array_equal(payload["sequence_mask"], [True, False])
    assert set(namespace["updated"]) == {"sequence_mask"}
    assert namespace["updated"]["sequence_mask"].dtype == np.bool_


def test_documented_validation_levels_distinguish_storage_from_training(doc_workspace):
    section = _reference_section("artifact-roundtrip", "## Choose a validation level")
    operation_names = {
        row[0].strip("`").removesuffix("()")
        for row in _table_rows(section)
        if row[0].startswith("`")
    }
    assert operation_names == {
        "write_preprocessed_artifact",
        "load_preprocessed_artifact",
        "validate_preprocessed_artifact",
        "preprocessed_artifact_exists",
        "validate_preprocessed_artifact_mapping",
    }
    operations = {name: getattr(cryoflake, name) for name in operation_names}
    artifact_mapping = {
        "density_grid": np.arange(24, dtype=np.int16).reshape(2, 3, 4),
        "label": "storage-only",
    }
    assert (
        operations["validate_preprocessed_artifact_mapping"](artifact_mapping)
        is artifact_mapping
    )
    with pytest.raises(ValueError, match="must be a mapping"):
        operations["validate_preprocessed_artifact_mapping"]([])

    artifact_reference = doc_workspace.root / "storage-only"
    operations["write_preprocessed_artifact"](artifact_reference, artifact_mapping)
    loaded = operations["load_preprocessed_artifact"](artifact_reference)
    assert loaded["density_grid"].dtype == np.float32
    np.testing.assert_array_equal(
        loaded["density_grid"], artifact_mapping["density_grid"]
    )
    assert operations["preprocessed_artifact_exists"](artifact_reference)
    with pytest.raises(ValueError, match="canonical preprocessing contract"):
        operations["validate_preprocessed_artifact"](artifact_reference)

    core_path = artifact_reference / "core.npz"
    core_path.write_bytes(core_path.read_bytes()[:-1])
    assert operations["preprocessed_artifact_exists"](artifact_reference)
    with pytest.raises(ValueError, match="SHA-256"):
        operations["load_preprocessed_artifact"](artifact_reference)


def test_validate_command_checks_float64_embeddings_once_and_reports_dimensions(
    doc_workspace, monkeypatch
):
    artifact_reference = "data/artifacts/example"
    artifact_mapping = load_preprocessed_artifact(artifact_reference)
    sequence = np.arange(16, dtype=np.float64).reshape(2, 8)
    artifact_mapping["sequence"] = sequence
    write_preprocessed_artifact(artifact_reference, artifact_mapping)
    normalized_dtypes = []
    validate_array = sequence_validation.validate_sequence_embedding_array

    def validate_and_record(sequence, *args, **kwargs):
        normalized_dtypes.append(np.asarray(sequence).dtype)
        return validate_array(sequence, *args, **kwargs)

    monkeypatch.setattr(
        sequence_validation,
        "validate_sequence_embedding_array",
        validate_and_record,
    )

    result = doc_workspace.invoke(get_example("guide-validate").commands()[0])

    summary = json.loads(result.output)
    assert summary["grid_shape"] == [4, 4, 4]
    assert summary["protein_nodes"] == 1
    assert summary["nucleotide_nodes"] == 1
    assert summary["component_nodes"] == 1
    assert summary["sequence_length"] == 2
    assert summary["sequence_feature_dim"] == 8
    assert normalized_dtypes == [np.dtype(np.float64)]
    stored_sequence = load_preprocessed_artifact(artifact_reference)["sequence"]
    assert stored_sequence.dtype == np.float64
    np.testing.assert_array_equal(stored_sequence, sequence)


def test_source_csv_example_normalizes_associations(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(get_example("contract-source-csv").code, encoding="utf-8")

    entries = read_source_associations(source)

    assert [entry.emdb_id for entry in entries] == ["1234", "12345"]
    assert [entry.wwpdb_id for entry in entries] == ["1abc", "2xyz"]
    assert [entry.source_resolution for entry in entries] == ["3.2", "2.8"]


@pytest.mark.parametrize(
    ("position", "expected_inside"),
    [
        pytest.param([10.0, 20.0, 30.0], True, id="inclusive-lower-corner"),
        pytest.param([19.5, 25.5, 33.5], True, id="noncubic-grid-interior"),
        pytest.param([20.0, 20.0, 30.0], False, id="exclusive-x-upper"),
        pytest.param([10.0, 26.0, 30.0], False, id="exclusive-y-upper"),
        pytest.param([10.0, 20.0, 34.0], False, id="exclusive-z-upper"),
        pytest.param([9.5, 20.0, 30.0], False, id="below-x-lower"),
        pytest.param([10.0, 19.5, 30.0], False, id="below-y-lower"),
        pytest.param([10.0, 20.0, 29.5], False, id="below-z-lower"),
        pytest.param([np.nan, 20.0, 30.0], False, id="missing-anchor"),
        pytest.param([np.inf, 20.0, 30.0], False, id="infinite-anchor"),
    ],
)
def test_coordinate_example_agrees_with_semantic_validation(
    doc_workspace, position, expected_inside
):
    # Distinct axis sizes, nonzero origin, and nonunit spacing expose xyz/zyx mixups.
    grid = np.zeros((2, 3, 5), dtype=np.float32)
    origin = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    coordinate = np.asarray(position, dtype=np.float32)
    namespace = get_example("contract-geometry").execute(
        {
            "position_xyz": coordinate,
            "global_origin": origin,
            "voxel_size": 2.0,
            "density_grid": grid,
        }
    )
    assert bool(namespace["inside"]) is expected_inside

    payload = load_preprocessed_artifact("data/artifacts/example")
    payload.update(density_grid=grid, global_origin=origin, voxel_size=2.0)
    payload["backbone_coordinates"]["CA"] = coordinate.reshape(1, 3)
    # The contract needs one CA/P anchor inside; the other anchor must not hide failures.
    payload["backbone_coordinates"]["P"] = np.full((1, 3), np.nan, dtype=np.float32)
    reference = doc_workspace.root / "geometry-example"
    write_preprocessed_artifact(reference, payload)

    if expected_inside:
        result = validate_preprocessed_artifact(reference)
        assert result.grid_shape == grid.shape
        assert result.protein_nodes == result.nucleotide_nodes == 1
    else:
        with pytest.raises(ValueError, match="preprocessing contract"):
            validate_preprocessed_artifact(reference)


@pytest.mark.parametrize("voxel_size", [1e-100, 1e100])
@pytest.mark.parametrize("at_origin", [True, False], ids=["origin", "below-origin"])
def test_geometry_example_and_validation_preserve_extreme_spacing_bounds(
    doc_workspace, voxel_size, at_origin
):
    coordinate = np.zeros(3, dtype=np.float32)
    if not at_origin:
        coordinate[0] = np.nextafter(np.float32(0), np.float32(-1))
    _assert_geometry_example_and_validate_command(
        doc_workspace, voxel_size, coordinate, expected_inside=at_origin
    )


@pytest.mark.parametrize("voxel_size", [float(np.float32(1e-40)), float(2**120)])
@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_geometry_example_and_validation_exclude_extreme_exact_upper_bounds(
    doc_workspace, voxel_size, axis
):
    # These spacings keep each tested upper coordinate exactly representable
    # in the canonical float32 coordinates, including subnormal values.
    coordinate = np.zeros(3, dtype=np.float32)
    coordinate[axis] = [5, 3, 2][axis] * voxel_size
    assert float(coordinate[axis]) == [5, 3, 2][axis] * voxel_size
    _assert_geometry_example_and_validate_command(
        doc_workspace, voxel_size, coordinate, expected_inside=False
    )


def _assert_geometry_example_and_validate_command(
    doc_workspace, voxel_size, coordinate, *, expected_inside
):
    grid = np.zeros((2, 3, 5), dtype=np.float32)
    origin = np.zeros(3, dtype=np.float32)
    namespace = get_example("contract-geometry").execute(
        {
            "position_xyz": coordinate,
            "global_origin": origin,
            "voxel_size": voxel_size,
            "density_grid": grid,
        }
    )
    assert bool(namespace["inside"]) is expected_inside
    artifact_reference = "data/artifacts/example"
    artifact_mapping = load_preprocessed_artifact(artifact_reference)
    artifact_mapping.update(
        density_grid=grid, global_origin=origin, voxel_size=voxel_size
    )
    artifact_mapping["backbone_coordinates"]["CA"] = coordinate.reshape(1, 3)
    artifact_mapping["backbone_coordinates"]["P"] = np.full(
        (1, 3), np.nan, dtype=np.float32
    )
    write_preprocessed_artifact(artifact_reference, artifact_mapping)

    result = doc_workspace.invoke(
        get_example("guide-validate").commands()[0], check=False
    )

    if expected_inside:
        assert result.exit_code == 0, result.output
        summary = json.loads(result.output)
        assert summary["grid_shape"] == [2, 3, 5]
        assert summary["protein_nodes"] == summary["nucleotide_nodes"] == 1
    else:
        assert result.exit_code == 1
        assert "canonical preprocessing contract" in result.output


def test_metadata_required_field_table_matches_real_artifacts(doc_workspace):
    section = _reference_section("artifact-tree", "## Metadata object")
    rows = _table_rows(section)
    documented_fields = {row[0].strip("`") for row in rows if row[0].startswith("`")}
    metadata = read_metadata(
        doc_workspace.root / "data/artifacts/example/metadata.json"
    )

    assert documented_fields == set(metadata)
    assert metadata["split_sequence"] is True
    assert set(metadata["files"]) == {"core", "sequence"}
    assert set(metadata["files"]) == set(metadata["store_generations"])
    assert set(metadata["files"]) == set(metadata["store_hashes"])


def test_full_atom_shape_and_dtype_table_matches_validated_targets(doc_workspace):
    section = _reference_section("contract-geometry", "### Full-atom supervision")
    payload = load_preprocessed_artifact("data/artifacts/example")
    normalized = validate_full_atom_targets(payload["full_atom_targets"])
    residue_count = len(normalized["aatype"])
    rows = [row for row in _table_rows(section) if row[0].startswith("`")]

    assert {row[0].strip("`") for row in rows} == set(normalized)
    for field, shape, dtype in rows:
        name = field.strip("`")
        if shape == "Scalar":
            assert name == "schema_version"
            assert dtype == "`int`"
            assert type(normalized[name]) is int
            assert normalized[name] == FULL_ATOM_TARGET_SCHEMA_VERSION == 1
            continue
        expected_shape = ast.literal_eval(
            shape.strip("`").replace("R", str(residue_count))
        )
        assert normalized[name].shape == expected_shape, name
        if dtype == "Unicode":
            assert normalized[name].dtype.kind == "U"
            assert not normalized[name].dtype.hasobject
            longest_name = max(map(len, normalized[name].flat), default=0)
            assert normalized[name].dtype.itemsize >= 4 * longest_name
        else:
            assert normalized[name].dtype == np.dtype(dtype.strip("`")), name


def _sparse_named_full_atom_mapping():
    artifact_mapping = load_preprocessed_artifact("data/artifacts/example")
    targets = artifact_mapping["full_atom_targets"]
    targets["atom_names"] = np.asarray(
        [f"slot_{slot}".encode() for slot in range(65)], dtype="S8"
    )
    targets["atom_names"][[2, 7, 23, 31, 64]] = [b" n ", b"ca", b"o", b"p", b"c1'"]
    targets["atom_mask"][0, [2, 7, 23]] = True
    targets["atom_mask"][1, [31, 64]] = True
    targets["atom_positions"][0, [2, 7, 23]] = [
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 1.0],
    ]
    targets["atom_positions"][1, [31, 64]] = [
        [2.0, 2.0, 2.0],
        [3.0, 2.0, 2.0],
    ]
    return artifact_mapping


def test_named_full_atom_roundtrip_reconstructs_sparse_atoms_from_the_guide(
    doc_workspace,
):
    artifact_mapping = _sparse_named_full_atom_mapping()
    original_targets = artifact_mapping["full_atom_targets"]
    write_preprocessed_artifact("data/artifacts/example", artifact_mapping)

    namespace = get_example("contract-full-atom-roundtrip").execute()

    restored_targets = namespace["restored_targets"]
    np.testing.assert_array_equal(namespace["protein_mask"], [True, False])
    assert "prot_mask" not in restored_targets
    assert type(restored_targets["schema_version"]) is int
    assert restored_targets["schema_version"] == FULL_ATOM_TARGET_SCHEMA_VERSION
    assert restored_targets["atom_names"].dtype.kind == "U"
    assert restored_targets["atom_names"].shape == (65,)
    assert restored_targets["atom_names"].dtype.itemsize == 4 * len("SLOT_63")
    assert restored_targets["atom_names"][[2, 7, 23, 31, 64]].tolist() == [
        "N",
        "CA",
        "O",
        "P",
        "C1'",
    ]
    assert len(set(restored_targets["atom_names"])) == 65
    assert all(restored_targets["atom_names"])
    np.testing.assert_array_equal(
        restored_targets["atom_mask"], original_targets["atom_mask"]
    )
    np.testing.assert_array_equal(
        restored_targets["atom_positions"], original_targets["atom_positions"]
    )
    assert not restored_targets["atom_mask"][0, :2].any()
    protein, nucleotide = namespace["named_residues"]
    assert (protein["chain_index"], protein["residue_index"]) == (0, 10)
    assert (nucleotide["chain_index"], nucleotide["residue_index"]) == (1, 20)
    assert set(protein["atom_positions_by_name"]) == {"N", "CA", "O"}
    assert set(nucleotide["atom_positions_by_name"]) == {"P", "C1'"}
    np.testing.assert_array_equal(protein["atom_positions_by_name"]["CA"], [1, 1, 1])
    np.testing.assert_array_equal(nucleotide["atom_positions_by_name"]["P"], [2, 2, 2])
    np.testing.assert_array_equal(
        nucleotide["atom_positions_by_name"]["C1'"], [3, 2, 2]
    )

    artifact_dir = doc_workspace.root / "data/artifacts/named-roundtrip"
    metadata = read_metadata(artifact_dir / "metadata.json")
    target_tree = metadata["tree"]["items"]["full_atom_targets"]["items"]
    assert target_tree["schema_version"] == {"kind": "scalar", "value": 1}
    assert "prot_mask" not in target_tree
    assert target_tree["atom_names"]["shape"] == [65]
    assert (
        target_tree["atom_names"]["dtype"] == restored_targets["atom_names"].dtype.str
    )
    with np.load(
        artifact_dir / metadata["files"]["core"], allow_pickle=False
    ) as arrays:
        assert "full_atom_targets/schema_version" not in arrays.files
        assert "full_atom_targets/prot_mask" not in arrays.files
        assert arrays.files.count("full_atom_targets/atom_names") == 1
        assert arrays["full_atom_targets/atom_names"].shape == (65,)
        assert arrays["full_atom_targets/atom_names"].dtype.kind == "U"
        np.testing.assert_array_equal(
            arrays["full_atom_targets/atom_names"], restored_targets["atom_names"]
        )
        np.testing.assert_array_equal(
            arrays["full_atom_targets/atom_mask"], original_targets["atom_mask"]
        )
    result = validate_preprocessed_artifact(artifact_dir)
    assert (result.protein_nodes, result.nucleotide_nodes) == (1, 1)


def test_documented_full_atom_extractor_preserves_shared_names_and_reorders_rows(
    doc_workspace,
):
    targets = _sparse_named_full_atom_mapping()["full_atom_targets"]
    source = SimpleNamespace(
        **{
            name: value if name == "atom_names" else value[::-1]
            for name, value in targets.items()
            if isinstance(value, np.ndarray)
        }
    )
    assert not hasattr(source, "schema_version")

    extracted = extract_full_atom_targets(source)

    assert extracted["schema_version"] == FULL_ATOM_TARGET_SCHEMA_VERSION == 1
    assert "prot_mask" not in extracted
    np.testing.assert_array_equal(extracted["aatype"], [0, 24])
    np.testing.assert_array_equal(extracted["chain_index"], [0, 1])
    np.testing.assert_array_equal(extracted["residue_index"], [10, 20])
    np.testing.assert_array_equal(extracted["atom_mask"], targets["atom_mask"])
    np.testing.assert_array_equal(
        extracted["atom_names"], validate_full_atom_targets(targets)["atom_names"]
    )
    assert extracted["atom_names"].shape == (65,)
    assert extracted["atom_names"][[2, 7, 23]].tolist() == ["N", "CA", "O"]
    assert extracted["atom_names"][[31, 64]].tolist() == ["P", "C1'"]
    np.testing.assert_array_equal(
        extracted["atom_positions"], targets["atom_positions"]
    )
    del source.atom_names
    with pytest.raises(ValueError, match="atom_names"):
        extract_full_atom_targets(source)


@pytest.mark.parametrize(
    "field", ["atom_positions", "atom_mask", "rigidgroups_gt_frames", "chain_index"]
)
@pytest.mark.parametrize("row_difference", [-1, 1])
def test_documented_extractor_requires_aligned_residue_rows(
    doc_workspace, field, row_difference
):
    targets = _sparse_named_full_atom_mapping()["full_atom_targets"]
    source = SimpleNamespace(
        **{key: value for key, value in targets.items() if key != "schema_version"}
    )
    values = getattr(source, field)
    malformed = (
        values[:-1] if row_difference < 0 else np.concatenate([values, values[:1]])
    )
    setattr(source, field, malformed)

    with pytest.raises(ValueError, match=f"aligned {field}"):
        extract_full_atom_targets(source)


@pytest.mark.parametrize(
    "missing_fields, schema_version",
    [
        (("schema_version", "atom_names"), None),
        (("schema_version",), None),
        (("atom_names",), 1),
        ((), 0),
        ((), 2),
        ((), True),
        ((), 1.0),
    ],
)
def test_documented_full_atom_schema_rejects_legacy_or_unsupported_mappings(
    doc_workspace, missing_fields, schema_version
):
    artifact_mapping = _sparse_named_full_atom_mapping()
    targets = artifact_mapping["full_atom_targets"]
    targets["schema_version"] = schema_version
    for field in missing_fields:
        targets.pop(field)
    with pytest.raises(ValueError, match="full_atom_targets"):
        validate_full_atom_targets(targets)
    artifact_reference = "data/artifacts/rejected"
    write_preprocessed_artifact(artifact_reference, artifact_mapping)

    result = doc_workspace.invoke(
        ["validate-artifact", artifact_reference], check=False
    )

    assert result.exit_code == 1
    assert "canonical preprocessing contract" in result.output


@pytest.mark.parametrize(
    "bad_name", ["", "  ", "ca", "slot_1", "C A", "α", 7, None, b"\xff"]
)
@pytest.mark.parametrize("slot", [0, 2])
def test_documented_full_atom_names_reject_invalid_even_fully_masked_slots(
    doc_workspace, bad_name, slot
):
    targets = _sparse_named_full_atom_mapping()["full_atom_targets"]
    targets["atom_names"] = targets["atom_names"].astype(object)
    assert not targets["atom_mask"][:, 0].any()
    targets["atom_names"][slot] = bad_name
    with pytest.raises(ValueError, match="atom_names"):
        validate_full_atom_targets(targets)


@pytest.mark.parametrize("residue_count", [0, 1, 2])
def test_documented_full_atom_table_rejects_residue_indexed_names(
    doc_workspace, residue_count
):
    targets = _sparse_named_full_atom_mapping()["full_atom_targets"]
    for field, value in targets.items():
        if isinstance(value, np.ndarray) and field != "atom_names":
            targets[field] = value[:residue_count]
    shared_names = targets["atom_names"]
    normalized = validate_full_atom_targets(targets)
    assert normalized["atom_names"].shape == (65,)
    assert normalized["atom_positions"].shape == (residue_count, 65, 3)
    targets["atom_names"] = np.tile(shared_names, (residue_count, 1))
    with pytest.raises(ValueError, match="atom_names.*shape"):
        validate_full_atom_targets(targets)
    source = SimpleNamespace(
        **{
            field: value
            for field, value in targets.items()
            if field != "schema_version"
        }
    )
    with pytest.raises(ValueError, match="atom_names.*shape"):
        extract_full_atom_targets(source)


def test_component_fields_and_enum_tables_match_validated_schema(doc_workspace):
    section = _reference_section(
        "contract-geometry", "### Component extension, schema version 1"
    )
    payload = load_preprocessed_artifact("data/artifacts/example")
    extension = validate_component_extension_targets(payload["extension_targets"])
    rows = _table_rows(section)
    # The field table combines atom_names and atom_elements into one display row.
    documented_fields = {
        name
        for row in rows
        if row[0].startswith("`") and len(row) == 3
        for name in re.findall(r"`([^`]+)`", row[0])
    }
    assert documented_fields == set(extension["component"])
    component = extension["component"]
    component_atom_mask = component["atom_names"] != ""
    assert component["atom_names"].ndim == component["atom_elements"].ndim == 2
    assert component["atom_names"].shape == component_atom_mask.shape
    assert component["atom_elements"].shape == component_atom_mask.shape
    dimension_sizes = {
        "N": len(component["positions"]),
        "A": component["atom_positions"].shape[1],
    }
    for names, shape, rule in (row for row in rows if len(row) == 3):
        fields = re.findall(r"`([^`]+)`", names)
        for field in fields:
            value = component[field]
            if shape.startswith("`("):
                dimensions = re.sub(
                    r"\b[NA]\b",
                    lambda match: str(dimension_sizes[match[0]]),
                    shape.strip("`"),
                )
                assert value.shape == ast.literal_eval(dimensions), field
            if dtype := re.search(r"`(float32|int64|bool)`", rule):
                assert value.dtype == np.dtype(dtype[1]), field
            if shape == "Mapping":
                assert set(value) == set(re.findall(r"`([^`]+)`", rule)), field

    for enum in (NodeKind, PolymerKind, AnchorKind):
        row = next(row for row in rows if row[0] == f"`{enum.__name__}`")
        documented_values = {
            name: int(value) for name, value in re.findall(r"`([A-Z_]+)=(\d+)`", row[1])
        }
        assert documented_values == {item.name: item.value for item in enum}


def test_component_reader_example_derives_values_and_keeps_stored_proposals(
    doc_workspace,
):
    source_path = doc_workspace.root / "data/derived-components.cif"
    source = (doc_workspace.root / "data/model.cif").read_text()
    source = source.replace(
        "3 non-polymer\n", "3 non-polymer\n4 water\n5 non-polymer\n"
    ).replace(
        "LIG 'NON-POLYMER'\n",
        "LIG 'NON-POLYMER'\nHOH WATER\nZN 'NON-POLYMER'\n",
    )
    source += (
        "HETATM O . HOH D 4 . D 40 ? 1 O 1.0 20.0 3.8 3.8 3.8\n"
        "HETATM ZN . ZN E 5 . E 50 ? 1 ZN 1.0 20.0 2.4 2.0 2.0\n"
    )
    source_path.write_text(source)
    artifact_mapping = load_preprocessed_artifact("data/artifacts/example")
    extension = build_component_extension_targets(
        source_path,
        global_origin=artifact_mapping["global_origin"],
        voxel_size=artifact_mapping["voxel_size"],
        grid_shape=artifact_mapping["density_grid"].shape,
        polymer_atom_positions=np.asarray([[1, 1, 1], [2, 2, 2]], dtype=np.float32),
        polymer_atom_node_indices=np.asarray([0, 1], dtype=np.int64),
        polymer_chain_indices=np.asarray([0, 1], dtype=np.int64),
        polymer_residue_indices=np.asarray([10, 20], dtype=np.int64),
        polymer_aatype=np.asarray([0, 24], dtype=np.int64),
        interface_contact_cutoff=0.75,
    )
    component = extension["component"]
    component["proposal_targets"]["node_indices"] = np.asarray([2, 0], dtype=np.int64)
    component["proposal_targets"]["valid_mask"] = np.asarray([False, True])
    identity_text = {
        field: np.asarray([f"  {value}  " for value in component[field]])
        for field in ("instance_id", "chain_id", "insertion_code")
    }
    for field in (
        "component_id",
        "instance_id",
        "chain_id",
        "insertion_code",
        "source_chem_comp_type",
        "atom_names",
        "atom_elements",
        "interface_group_id",
    ):
        values = component[field]
        lowercase = field in {
            "component_id",
            "source_chem_comp_type",
            "atom_names",
            "atom_elements",
        }
        component[field] = np.asarray(
            [
                f"  {str(value).lower() if lowercase else value}  ".encode("utf-8")
                for value in values.flat
            ]
        ).reshape(values.shape)
    artifact_mapping["extension_targets"] = extension
    write_preprocessed_artifact("data/artifacts/example", artifact_mapping)

    namespace = get_example("contract-component-derived").execute()

    component = namespace["component"]
    assert component["component_id"].tolist() == ["LIG", "HOH", "ZN"]
    assert component["atom_names"].tolist() == [["C1", "O1"], ["O", ""], ["ZN", ""]]
    assert component["atom_elements"].tolist() == [["C", "O"], ["O", ""], ["ZN", ""]]
    assert component["interface_group_id"][1] == ""
    for field, values in identity_text.items():
        np.testing.assert_array_equal(component[field], values)
    assert component["node_kind"].tolist() == [
        int(NodeKind.NON_POLYMER),
        int(NodeKind.WATER),
        int(NodeKind.ION),
    ]
    np.testing.assert_array_equal(
        namespace["component_atom_mask"], [[True, True], [True, False], [True, False]]
    )
    np.testing.assert_array_equal(
        namespace["component_interface_mask"], [True, False, True]
    )
    ligand_digest = hashlib.sha256(repr([(0, 10, 0)]).encode()).hexdigest()[:16]
    ion_digest = hashlib.sha256(repr([(1, 20, 24)]).encode()).hexdigest()[:16]
    assert namespace["component_groups"] == ("LIG", "ZN")
    assert namespace["interface_groups"] == (f"LIG|{ligand_digest}", f"ZN|{ion_digest}")
    np.testing.assert_array_equal(
        namespace["proposal_positions"], component["positions"][[2, 0]]
    )
    np.testing.assert_array_equal(component["proposal_targets"]["node_indices"], [2, 0])
    np.testing.assert_array_equal(
        component["proposal_targets"]["valid_mask"], [False, True]
    )
    # The ion anchor is inside the grid, but custom proposal validity stays false.
    assert np.all((component["positions"][2] >= 0) & (component["positions"][2] < 4))
    assert {"atom_mask", "interface_mask", "split_groups"}.isdisjoint(component)
    assert set(component["proposal_targets"]) == {"node_indices", "valid_mask"}

    artifact_dir = doc_workspace.root / "data/artifacts/example"
    metadata = read_metadata(artifact_dir / "metadata.json")
    component_tree = metadata["tree"]["items"]["extension_targets"]["items"][
        "component"
    ]["items"]
    assert {"atom_mask", "interface_mask", "split_groups"}.isdisjoint(component_tree)
    assert set(component_tree["proposal_targets"]["items"]) == {
        "node_indices",
        "valid_mask",
    }
    with np.load(
        artifact_dir / metadata["files"]["core"], allow_pickle=False
    ) as arrays:
        absent_arrays = {
            "extension_targets/component/atom_mask",
            "extension_targets/component/interface_mask",
            "extension_targets/component/proposal_targets/positions",
        }
        assert absent_arrays.isdisjoint(arrays.files)
        assert not any(
            key.startswith("extension_targets/component/split_groups/")
            for key in arrays.files
        )
        np.testing.assert_array_equal(
            arrays["extension_targets/component/proposal_targets/node_indices"], [2, 0]
        )
        np.testing.assert_array_equal(
            arrays["extension_targets/component/proposal_targets/valid_mask"],
            [False, True],
        )


@pytest.mark.parametrize(
    "field",
    [
        "component_id",
        "instance_id",
        "chain_id",
        "insertion_code",
        "source_chem_comp_type",
        "atom_names",
        "atom_elements",
        "interface_group_id",
    ],
)
@pytest.mark.parametrize("bad_value", [None, 7, b"\xff"])
def test_component_reader_example_rejects_non_string_values(
    doc_workspace, monkeypatch, field, bad_value
):
    artifact_mapping = load_preprocessed_artifact("data/artifacts/example")
    component = artifact_mapping["extension_targets"]["component"]
    component[field] = component[field].astype(object)
    component[field].flat[0] = bad_value
    monkeypatch.setattr(
        cryoflake,
        "load_preprocessed_artifact",
        lambda *args, **kwargs: artifact_mapping,
    )

    with pytest.raises(ValueError, match=field):
        get_example("contract-component-derived").execute()


@pytest.mark.parametrize(
    "mapping_name, field",
    [
        ("full_atom_targets", "prot_mask"),
        ("component", "atom_mask"),
        ("component", "interface_mask"),
        ("component", "split_groups"),
        ("proposal_targets", "positions"),
    ],
)
def test_removed_duplicate_target_fields_are_rejected(
    doc_workspace, mapping_name, field
):
    artifact_mapping = load_preprocessed_artifact("data/artifacts/example")
    full_atom = artifact_mapping["full_atom_targets"]
    extension = artifact_mapping["extension_targets"]
    component = extension["component"]
    proposal = component["proposal_targets"]
    removed_values = {
        ("full_atom_targets", "prot_mask"): (full_atom["aatype"] < 20)
        | (full_atom["aatype"] == 28),
        ("component", "atom_mask"): component["atom_names"] != "",
        ("component", "interface_mask"): component["interface_group_id"] != "",
        ("component", "split_groups"): {
            "component": tuple(
                sorted(
                    set(
                        component["component_id"][
                            component["node_kind"] != int(NodeKind.WATER)
                        ].tolist()
                    )
                )
            ),
            "interface": tuple(
                sorted(
                    set(
                        component["interface_group_id"][
                            component["interface_group_id"] != ""
                        ].tolist()
                    )
                )
            ),
        },
        ("proposal_targets", "positions"): component["positions"][
            proposal["node_indices"]
        ],
    }
    target_mapping = {
        "full_atom_targets": full_atom,
        "component": component,
        "proposal_targets": proposal,
    }[mapping_name]
    target_mapping[field] = removed_values[mapping_name, field]
    with pytest.raises(ValueError):
        if mapping_name == "full_atom_targets":
            validate_full_atom_targets(full_atom)
        else:
            validate_component_extension_targets(extension)
    artifact_reference = "data/artifacts/rejected-duplicate"
    write_preprocessed_artifact(artifact_reference, artifact_mapping)
    result = doc_workspace.invoke(
        ["validate-artifact", artifact_reference], check=False
    )
    assert result.exit_code == 1
    assert "canonical preprocessing contract" in result.output


def test_split_report_and_assignment_fields_match_the_specification(tmp_path):
    section = _reference_section(
        "contract-geometry", "## Artifact lists and split reports"
    )
    documented_fields = {
        field
        for row in _table_rows(section)
        for field in re.findall(r"`([a-z_]+)`", row[0])
    }
    assignment_description = re.search(
        r"Each assignment has (.*?)\.\s+Report row indices", section, re.S
    )
    assert assignment_description is not None
    assignment_fields = set(re.findall(r"`([a-z_]+)`", assignment_description[1]))
    records = [
        GroupedSplitRecord("s3://bucket/repeated", "first", ("a",)),
        GroupedSplitRecord("s3://bucket/repeated", "second", ("b",)),
        GroupedSplitRecord("s3://bucket/other", "other", ("b",)),
    ]
    split = split_grouped_records(
        records,
        split_strategy="component",
        validation_fraction=0.5,
        seed=7,
        holdout_group_ids=["a"],
    )
    split_report_path = tmp_path / "split.json"
    write_grouped_split_report(split, split_report_path)
    report = json.loads(split_report_path.read_text())

    assert set(report) == documented_fields
    assert all(set(row) == assignment_fields for row in report["assignments"])
    assert [row["row_index"] for row in report["assignments"]] == [1, 2, 3]
    assert report["num_records"] == len(records) == 3
    assert report["num_validation_records"] == 2
    assert report["num_train_records"] == 1
    assert len({row["artifact_reference"] for row in report["assignments"]}) == 2
    assert read_holdout_group_ids(
        split_report_path, expected_split_strategy="component"
    ) == ("a",)


def test_configuration_tables_match_all_pipeline_defaults(doc_workspace):
    section = _reference_section("config-inspect", "## Pipeline reference")
    default_tables, single_commands = section.split("### Single-artifact commands", 1)
    pipeline = compose_config().pipeline
    documented_settings = set()
    for setting, default, _ in _table_rows(default_tables):
        if not setting.startswith("`"):
            continue
        name = setting.strip("`")
        value = default.strip("`")
        documented_settings.add(name)
        actual = pipeline[name]
        expected = value if isinstance(actual, str) else json.loads(value)
        assert actual == expected, name
        if isinstance(actual, bool):
            assert type(expected) is bool, name
    for setting, _, _ in _table_rows(single_commands):
        if setting.startswith("`"):
            name = setting.strip("`")
            documented_settings.add(name)
            assert pipeline[name] is None, name

    assert documented_settings == set(pipeline)


def test_training_root_table_matches_the_canonical_semantic_contract(doc_workspace):
    section = _reference_section("contract-geometry", "## Complete training payload")
    section = re.split(r"^### ", section, maxsplit=1, flags=re.M)[0]
    documented_fields = {
        field
        for row in _table_rows(section)
        if row[0].startswith("`")
        for field in re.findall(r"`([^`]+)`", row[0])
    }
    artifact_reference = doc_workspace.root / "data/artifacts/example"
    artifact_mapping = load_preprocessed_artifact(artifact_reference=artifact_reference)

    assert documented_fields == set(artifact_mapping)
    assert {"density_grid", "backbone_coordinates"}.issubset(artifact_mapping)
    result = validate_preprocessed_artifact(artifact_reference=artifact_reference)
    assert result.artifact_reference == str(artifact_reference)


def test_validation_result_field_table_matches_serialized_result(doc_workspace):
    section = _reference_section("contract-geometry", "## Artifact validation results")
    documented_fields = {
        field
        for row in _table_rows(section)
        if row[0].startswith("`")
        for field in re.findall(r"`([^`]+)`", row[0])
    }
    artifact_reference = doc_workspace.root / "data/artifacts/example"
    result = validate_preprocessed_artifact(artifact_reference=artifact_reference)

    assert documented_fields == set(json.loads(result.to_json()))


def test_summary_field_table_matches_the_csv_contract():
    section = _reference_section("contract-geometry", "## Preprocessing summary CSV")
    field_table = section.split("\n| Status |", maxsplit=1)[0]
    documented_fields = {
        field
        for row in _table_rows(field_table)
        if row[0].startswith("`")
        for field in re.findall(r"`([^`]+)`", row[0])
    }

    assert documented_fields == set(PREPROCESS_SUMMARY_FIELDNAMES)
    assert {
        "artifact_reference",
        "cached_artifact_dir",
        "source_resolution",
        "input_voxel_size",
        "artifact_voxel_size",
    }.issubset(documented_fields)


def test_documented_versions_match_runtime_and_written_artifacts(doc_workspace):
    source = get_example("contract-geometry").path.read_text()
    introduction = source.split("\n## ", maxsplit=1)[0]
    documented_versions = {
        row[0]: int(row[1].strip("`"))
        for row in _table_rows(introduction)
        if row[1].startswith("`")
    }
    artifact_reference = doc_workspace.root / "data/artifacts/example"
    metadata = read_metadata(artifact_reference / "metadata.json")
    artifact_mapping = load_preprocessed_artifact(artifact_reference)
    assert documented_versions == {
        "Semantic preprocessing": PREPROCESSED_ARTIFACT_CONTRACT_VERSION,
        "NPZ+JSON storage": metadata["format_version"],
        "Full-atom targets": artifact_mapping["full_atom_targets"]["schema_version"],
        "Component extension": artifact_mapping["extension_targets"]["schema_version"],
    }
    checkpoint = re.search(r"`format_version: (\d+)`", source)
    assert checkpoint is not None
    assert int(checkpoint[1]) == PREPROCESS_CHECKPOINT_FORMAT_VERSION
    report_section = _reference_section(
        "contract-geometry", "## Artifact lists and split reports"
    )
    report_version = next(
        row[1] for row in _table_rows(report_section) if row[0] == "`schema_version`"
    )
    assert int(report_version.strip("`")) == SPLIT_REPORT_SCHEMA_VERSION
