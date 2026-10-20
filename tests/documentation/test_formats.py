"""Execute the published storage examples and check the format reference tables."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest

from cryoflake import (
    FASTASequence,
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
from cryoflake.storage.codec import read_metadata
from cryoflake.storage.transports import is_remote_uri

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


def test_artifact_reference_examples_load_the_same_payload(doc_workspace):
    doc_workspace.seed_training("artifacts/example")
    references = []
    payloads = []

    def load_and_record(reference):
        payload = load_preprocessed_artifact(reference)
        references.append(reference)
        payloads.append(payload)
        return payload

    get_example("artifact-references").execute(
        {"load_preprocessed_artifact": load_and_record}
    )

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
        expected_shape = ast.literal_eval(
            shape.strip("`").replace("R", str(residue_count))
        )
        assert normalized[name].shape == expected_shape, name
        assert normalized[name].dtype == np.dtype(dtype.strip("`")), name


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


def test_documented_semantic_and_recovery_versions_match_runtime():
    source = get_example("contract-geometry").path.read_text()
    semantic = re.search(
        r"semantic preprocessing contract is version \*\*(\d+)\*\*", source
    )
    checkpoint = re.search(r"`format_version: (\d+)`", source)
    report = re.search(r"\| `schema_version` \| `(\d+)` \|", source)

    assert semantic is not None
    assert checkpoint is not None
    assert report is not None
    assert int(semantic[1]) == PREPROCESSED_ARTIFACT_CONTRACT_VERSION
    assert int(checkpoint[1]) == PREPROCESS_CHECKPOINT_FORMAT_VERSION
    assert int(report[1]) == SPLIT_REPORT_SCHEMA_VERSION
