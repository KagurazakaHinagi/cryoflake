"""Canonical report replay rejects mixed schemas and silently changed holdouts."""

import json
from dataclasses import replace

import pytest

from cryoflake.manifests import splits


@pytest.fixture
def canonical_report(tmp_path):
    records = (
        splits.GroupedSplitRecord("first", "shared", ("a",), ("i-a",)),
        splits.GroupedSplitRecord("peer", "shared", ("b",), ("i-b",)),
        splits.GroupedSplitRecord("other", "other", ("b", "c"), ("i-b", "i-c")),
    )
    result = splits.split_grouped_records(
        records,
        split_strategy="component",
        validation_fraction=0.5,
        seed=7,
        holdout_group_ids=["a"],
    )
    path = tmp_path / "report.json"
    splits.write_grouped_split_report(result, path)
    return result, path, json.loads(path.read_text())


def _rewrite(path, payload):
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize("split_strategy", ["component", "interface"])
def test_canonical_report_replays_holdouts_and_atomic_peers(
    tmp_path, canonical_report, split_strategy
):
    original, _, _ = canonical_report
    selected = "a" if split_strategy == "component" else "i-a"
    result = splits.split_grouped_records(
        original.records,
        split_strategy=split_strategy,
        validation_fraction=0.5,
        seed=7,
        holdout_group_ids=[selected],
    )
    report = tmp_path / "selected.json"
    splits.write_grouped_split_report(result, report)

    replayed = splits.split_grouped_records(
        original.records,
        split_strategy=split_strategy,
        validation_fraction=0.1,
        seed=12345,
        holdout_group_ids=splits.read_holdout_group_ids(
            report, expected_split_strategy=split_strategy
        ),
    )

    assert replayed.assignments == result.assignments == ("valid", "valid", "train")
    assert replayed.validation_artifact_references == ("first", "peer")
    assert replayed.selected_holdout_group_ids == (selected,)


def test_reading_report_does_not_resolve_historical_artifact_references(
    canonical_report, tmp_path, monkeypatch
):
    _, report, _ = canonical_report
    unrelated = tmp_path / "another-application"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    def do_not_resolve_artifact(reference):
        raise AssertionError(
            f"Historical artifacts are not inputs to replay: {reference}"
        )

    monkeypatch.setattr(
        splits, "_artifact_reference_group_key", do_not_resolve_artifact
    )

    assert splits.read_holdout_group_ids(report) == ("a",)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("schema_version", 3, "schema_version"),
        ("schema_version", 3.0, "schema_version"),
        ("schema_version", True, "schema_version"),
        ("schema_version", None, "schema_version"),
        ("schema_version", "3", "schema_version"),
        ("schema_version", "G2610211", "schema_version"),
        ("schema_version", "G2610200", "schema_version"),
        ("schema_version", "G2610320", "schema_version"),
        ("schema_version", "G26102100", "schema_version"),
        ("schema_version", "S2610210", "schema_version"),
        ("schema_version", "g2610210", "schema_version"),
        ("schema_version", " G2610210", "schema_version"),
        ("schema_version", ["G2610210"], "schema_version"),
        ("schema_version", {"version": "G2610210"}, "schema_version"),
        ("seed", False, "seed"),
        ("seed", 7.5, "seed"),
        ("validation_fraction", True, "validation_fraction"),
        ("validation_fraction", "0.5", "validation_fraction"),
        ("validation_fraction", float("nan"), "nonfinite"),
        ("num_records", 3.0, "num_records"),
        ("num_train_records", True, "num_train_records"),
        ("num_validation_records", 2.0, "num_validation_records"),
    ],
)
def test_report_scalar_types_are_strict(canonical_report, field, value, message):
    _, report, payload = canonical_report
    payload[field] = value
    _rewrite(report, payload)

    with pytest.raises(ValueError, match=message):
        splits.read_holdout_group_ids(report)


@pytest.mark.parametrize(
    "alias", ["strategy", "selected_holdout_groups", "num_train", "num_valid"]
)
def test_version_three_report_rejects_mixed_legacy_top_level_fields(
    canonical_report, alias
):
    _, report, payload = canonical_report
    payload[alias] = "unused legacy value"
    _rewrite(report, payload)

    with pytest.raises(ValueError, match="unexpected fields"):
        splits.read_holdout_group_ids(report)


@pytest.mark.parametrize("alias", ["path", "example_group_id"])
def test_version_three_report_rejects_legacy_assignment_fields(canonical_report, alias):
    _, report, payload = canonical_report
    payload["assignments"][0][alias] = "old-field-value"
    _rewrite(report, payload)

    with pytest.raises(ValueError, match="assignment.*unexpected fields"):
        splits.read_holdout_group_ids(report)


@pytest.mark.parametrize(
    "field", ["artifact_reference", "artifact_group_id", "row_index"]
)
def test_report_assignments_require_canonical_fields(canonical_report, field):
    _, report, payload = canonical_report
    del payload["assignments"][0][field]
    _rewrite(report, payload)

    with pytest.raises(ValueError, match="assignment.*missing"):
        splits.read_holdout_group_ids(report)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("row_index", False, "row_index"),
        ("row_index", 1.0, "row_index"),
        ("artifact_reference", None, "artifact_reference"),
        ("artifact_reference", "  ", "artifact_reference"),
        ("artifact_group_id", 12, "artifact_group_id"),
        ("component_group_ids", "a", "component_group_ids"),
        ("interface_group_ids", [None], "interface_group_ids"),
        ("split", "val", "split must"),
        ("split", ["valid"], "split must"),
    ],
)
def test_report_assignment_values_are_validated(
    canonical_report, field, value, message
):
    _, report, payload = canonical_report
    payload["assignments"][0][field] = value
    _rewrite(report, payload)

    with pytest.raises(ValueError, match=message):
        splits.read_holdout_group_ids(report)


@pytest.mark.parametrize(
    "field, value",
    [("num_records", 4), ("num_train_records", 2), ("num_validation_records", 1)],
)
def test_report_counts_agree_with_its_assignments(canonical_report, field, value):
    _, report, payload = canonical_report
    payload[field] = value
    _rewrite(report, payload)

    with pytest.raises(ValueError, match="assignments|counts"):
        splits.read_holdout_group_ids(report)


@pytest.mark.parametrize(
    "selected, message", [([], "at least one"), (["missing"], "unknown")]
)
def test_report_selected_groups_exist_in_its_records(
    canonical_report, selected, message
):
    _, report, payload = canonical_report
    payload["selected_holdout_group_ids"] = selected
    _rewrite(report, payload)

    with pytest.raises(ValueError, match=message):
        splits.read_holdout_group_ids(report)


def test_report_cannot_put_selected_holdout_groups_in_training(canonical_report):
    _, report, payload = canonical_report
    payload["selected_holdout_group_ids"] = ["b"]
    _rewrite(report, payload)

    with pytest.raises(ValueError, match="selected holdout group in training"):
        splits.read_holdout_group_ids(report)


@pytest.mark.parametrize("field", ["artifact_reference", "artifact_group_id"])
def test_report_cannot_split_identical_artifact_peers(canonical_report, field):
    _, report, payload = canonical_report
    payload["assignments"][2][field] = payload["assignments"][0][field]
    _rewrite(report, payload)

    with pytest.raises(ValueError, match="atomic artifact peers"):
        splits.read_holdout_group_ids(report)


@pytest.mark.parametrize("nested", [False, True])
def test_duplicate_report_json_keys_cannot_override_replay_fields(
    canonical_report, nested
):
    _, report, payload = canonical_report
    serialized = json.dumps(payload)
    field = "artifact_reference" if nested else "split_strategy"
    value = payload["assignments"][0][field] if nested else payload[field]
    original = f'"{field}": {json.dumps(value)}'
    report.write_text(
        serialized.replace(original, f'"{field}": "overridden", {original}', 1)
    )

    with pytest.raises(ValueError, match=f"repeats key '{field}'"):
        splits.read_holdout_group_ids(report)


@pytest.mark.parametrize("is_report", [False, True])
def test_holdout_ids_still_strip_deduplicate_and_sort(canonical_report, is_report):
    _, report, payload = canonical_report
    values = [" a ", "a"]
    if is_report:
        payload["selected_holdout_group_ids"] = values
        values = payload
    _rewrite(report, values)

    assert splits.read_holdout_group_ids(report) == ("a",)


def test_invalid_result_does_not_publish_a_nonreplayable_report(
    canonical_report, tmp_path
):
    result, _, _ = canonical_report
    invalid = replace(result, selected_holdout_group_ids=("missing",))
    destination = tmp_path / "uncreated" / "invalid.json"

    with pytest.raises(ValueError, match="unknown"):
        splits.write_grouped_split_report(invalid, destination)

    assert not destination.parent.exists()


def test_report_writer_normalizes_selected_group_ids(canonical_report, tmp_path):
    result, _, _ = canonical_report
    result = replace(result, selected_holdout_group_ids=(" a ", "a"))
    destination = tmp_path / "normalized.json"

    splits.write_grouped_split_report(result, destination)

    assert json.loads(destination.read_text())["selected_holdout_group_ids"] == ["a"]
    assert splits.read_holdout_group_ids(destination) == ("a",)
