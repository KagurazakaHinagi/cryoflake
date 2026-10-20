from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

from cryoflake.storage.transports import is_remote_uri

GroupedSplitStrategy = Literal["component", "interface"]
SplitAssignment = Literal["train", "valid"]
SPLIT_REPORT_SCHEMA_VERSION = 3
_REPORT_FIELDS = {
    "schema_version",
    "split_strategy",
    "validation_fraction",
    "seed",
    "selected_holdout_group_ids",
    "num_records",
    "num_train_records",
    "num_validation_records",
    "assignments",
}
_ASSIGNMENT_FIELDS = {
    "row_index",
    "artifact_reference",
    "artifact_group_id",
    "component_group_ids",
    "interface_group_ids",
    "split",
}


def _normalize_group_ids(
    values: Iterable[str] | str | None,
    *,
    label: str,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    if values is None:
        normalized_values: list[str] = []
    elif isinstance(values, str):
        normalized_values = [values.strip()]
    else:
        normalized_values = [str(value).strip() for value in values]

    if any(not value for value in normalized_values):
        raise ValueError(f"{label} must not contain empty group IDs")
    normalized = tuple(sorted(set(normalized_values)))
    if require_nonempty and not normalized:
        raise ValueError(f"{label} must contain at least one group ID")
    return normalized


@dataclass(frozen=True)
class GroupedSplitRecord:
    """One preprocessed artifact and its leakage-control group identifiers."""

    artifact_reference: os.PathLike[str] | str
    artifact_group_id: str
    component_group_ids: tuple[str, ...] = ()
    interface_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        path_value = os.fspath(self.artifact_reference)
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(
                "artifact_reference must be a non-empty filesystem path or remote URI"
            )
        artifact_group_id = str(self.artifact_group_id).strip()
        if not artifact_group_id:
            raise ValueError("artifact_group_id must be a non-empty stable identifier")

        object.__setattr__(self, "artifact_group_id", artifact_group_id)
        object.__setattr__(
            self,
            "component_group_ids",
            _normalize_group_ids(
                self.component_group_ids,
                label="component_group_ids",
            ),
        )
        object.__setattr__(
            self,
            "interface_group_ids",
            _normalize_group_ids(
                self.interface_group_ids,
                label="interface_group_ids",
            ),
        )


@dataclass(frozen=True)
class GroupedSplitResult:
    """A deterministic split whose assignments align with ``records``."""

    records: tuple[GroupedSplitRecord, ...]
    assignments: tuple[SplitAssignment, ...]
    split_strategy: GroupedSplitStrategy
    validation_fraction: float
    seed: int
    selected_holdout_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.records) != len(self.assignments):
            raise ValueError("records and assignments must have the same length")
        if any(split not in {"train", "valid"} for split in self.assignments):
            raise ValueError("assignments must contain only 'train' or 'valid'")

    @property
    def train_artifact_references(self) -> tuple[os.PathLike[str] | str, ...]:
        return tuple(
            record.artifact_reference
            for record, split in zip(self.records, self.assignments)
            if split == "train"
        )

    @property
    def validation_artifact_references(self) -> tuple[os.PathLike[str] | str, ...]:
        return tuple(
            record.artifact_reference
            for record, split in zip(self.records, self.assignments)
            if split == "valid"
        )


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            self.parent[index] = self.find(parent)
        return self.parent[index]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def _artifact_reference_group_key(artifact_reference: os.PathLike[str] | str) -> str:
    reference_value = os.fspath(artifact_reference)
    if is_remote_uri(reference_value):
        return reference_value
    return str(Path(reference_value).expanduser().resolve())


def _atomic_peer_sets(
    records: Sequence[GroupedSplitRecord],
) -> tuple[tuple[int, ...], ...]:
    disjoint_set = _DisjointSet(len(records))
    reference_owners: dict[str, int] = {}
    artifact_group_owners: dict[str, int] = {}

    for index, record in enumerate(records):
        reference_key = _artifact_reference_group_key(record.artifact_reference)
        reference_owner = reference_owners.setdefault(reference_key, index)
        disjoint_set.union(index, reference_owner)

        artifact_group_owner = artifact_group_owners.setdefault(
            record.artifact_group_id, index
        )
        disjoint_set.union(index, artifact_group_owner)

    peer_sets: dict[int, list[int]] = {}
    for index in range(len(records)):
        peer_sets.setdefault(disjoint_set.find(index), []).append(index)
    return tuple(tuple(indices) for indices in peer_sets.values())


def _group_record_indices(
    records: Sequence[GroupedSplitRecord],
    split_strategy: GroupedSplitStrategy,
) -> dict[str, set[int]]:
    attribute = (
        "component_group_ids"
        if split_strategy == "component"
        else "interface_group_ids"
    )
    group_indices: dict[str, set[int]] = {}
    for index, record in enumerate(records):
        for group_id in getattr(record, attribute):
            group_indices.setdefault(group_id, set()).add(index)
    return group_indices


def _expand_to_atomic_peers(
    indices: set[int],
    peer_sets: Sequence[Sequence[int]],
) -> set[int]:
    expanded = set(indices)
    for peers in peer_sets:
        if expanded.intersection(peers):
            expanded.update(peers)
    return expanded


def _automatic_holdout_groups(
    group_indices: dict[str, set[int]],
    peer_sets: Sequence[Sequence[int]],
    *,
    num_records: int,
    validation_fraction: float,
    seed: int,
) -> tuple[str, ...]:
    target_count = round(num_records * validation_fraction)
    target_count = min(max(1, target_count), num_records - 1)

    candidate_groups = sorted(group_indices)
    random.Random(seed).shuffle(candidate_groups)
    candidate_rank = {group_id: rank for rank, group_id in enumerate(candidate_groups)}
    selected: list[str] = []
    valid_indices: set[int] = set()
    remaining = set(candidate_groups)

    while remaining:
        choices = []
        for group_id in remaining:
            proposed_indices = _expand_to_atomic_peers(
                valid_indices | group_indices[group_id],
                peer_sets,
            )
            if len(proposed_indices) == len(valid_indices):
                continue
            if len(proposed_indices) >= num_records:
                continue
            choices.append(
                (
                    abs(len(proposed_indices) - target_count),
                    candidate_rank[group_id],
                    group_id,
                    proposed_indices,
                )
            )

        if not choices:
            break
        best_error, _, group_id, proposed_indices = min(
            choices, key=lambda choice: (choice[0], choice[1])
        )
        current_error = abs(len(valid_indices) - target_count)
        if selected and best_error >= current_error:
            break
        selected.append(group_id)
        valid_indices = proposed_indices
        remaining.remove(group_id)
        if len(valid_indices) == target_count:
            break

    if not selected:
        raise ValueError(
            "Cannot create a non-empty grouped validation split without "
            "emptying the training split"
        )
    return tuple(sorted(selected))


def _validate_fraction(validation_fraction: float) -> float:
    validation_fraction = float(validation_fraction)
    if (
        not math.isfinite(validation_fraction)
        or validation_fraction <= 0
        or validation_fraction >= 1
    ):
        raise ValueError(
            "validation_fraction must be finite, greater than 0, and less than 1"
        )
    return validation_fraction


def _validate_split_invariants(
    records: Sequence[GroupedSplitRecord],
    assignments: Sequence[SplitAssignment],
    *,
    split_strategy: GroupedSplitStrategy,
    selected_groups: Sequence[str],
) -> None:
    train_indices = {
        index for index, assignment in enumerate(assignments) if assignment == "train"
    }
    valid_indices = set(range(len(records))) - train_indices
    if not train_indices or not valid_indices:
        raise ValueError(
            "Grouped split must leave both training and validation non-empty"
        )

    train_artifact_references = {
        _artifact_reference_group_key(records[index].artifact_reference)
        for index in train_indices
    }
    validation_artifact_references = {
        _artifact_reference_group_key(records[index].artifact_reference)
        for index in valid_indices
    }
    if train_artifact_references.intersection(validation_artifact_references):
        raise RuntimeError(
            "Grouped split placed the same artifact reference in both outputs"
        )

    train_artifact_groups = {
        records[index].artifact_group_id for index in train_indices
    }
    valid_artifact_groups = {
        records[index].artifact_group_id for index in valid_indices
    }
    if train_artifact_groups.intersection(valid_artifact_groups):
        raise RuntimeError(
            "Grouped split placed the same artifact group in both outputs"
        )

    attribute = (
        "component_group_ids"
        if split_strategy == "component"
        else "interface_group_ids"
    )
    selected_group_set = set(selected_groups)
    train_groups = {
        group_id
        for index in train_indices
        for group_id in getattr(records[index], attribute)
    }
    if selected_group_set.intersection(train_groups):
        raise RuntimeError("A selected holdout group remains in the training split")


def split_grouped_records(
    records: Iterable[GroupedSplitRecord],
    *,
    split_strategy: GroupedSplitStrategy,
    validation_fraction: float,
    seed: int,
    holdout_group_ids: Iterable[str] | str | None = None,
) -> GroupedSplitResult:
    """Split records without leaking selected groups or atomic artifact peers.

    Co-occurring component or interface IDs are not implicitly selected. A
    record can therefore move to validation because it contains one selected
    group while its other, unselected groups remain represented in training.
    """

    records = tuple(records)
    if len(records) < 2:
        raise ValueError("Grouped splitting requires at least two records")
    if any(not isinstance(record, GroupedSplitRecord) for record in records):
        raise TypeError("records must contain GroupedSplitRecord instances")
    if split_strategy not in {"component", "interface"}:
        raise ValueError("split_strategy must be 'component' or 'interface'")
    validation_fraction = _validate_fraction(validation_fraction)
    seed = int(seed)

    group_indices = _group_record_indices(records, split_strategy)
    if not group_indices:
        raise ValueError(f"No {split_strategy} group IDs are available for splitting")
    peer_sets = _atomic_peer_sets(records)

    if holdout_group_ids is None:
        selected_groups = _automatic_holdout_groups(
            group_indices,
            peer_sets,
            num_records=len(records),
            validation_fraction=validation_fraction,
            seed=seed,
        )
    else:
        selected_groups = _normalize_group_ids(
            holdout_group_ids,
            label="holdout_group_ids",
            require_nonempty=True,
        )
        unknown_groups = sorted(set(selected_groups) - set(group_indices))
        if unknown_groups:
            raise ValueError(
                f"Unknown {split_strategy} holdout groups: {', '.join(unknown_groups)}"
            )

    selected_indices = {
        index for group_id in selected_groups for index in group_indices[group_id]
    }
    valid_indices = _expand_to_atomic_peers(selected_indices, peer_sets)
    assignments: tuple[SplitAssignment, ...] = tuple(
        "valid" if index in valid_indices else "train" for index in range(len(records))
    )
    _validate_split_invariants(
        records,
        assignments,
        split_strategy=split_strategy,
        selected_groups=selected_groups,
    )
    return GroupedSplitResult(
        records=records,
        assignments=assignments,
        split_strategy=split_strategy,
        validation_fraction=validation_fraction,
        seed=seed,
        selected_holdout_group_ids=tuple(sorted(selected_groups)),
    )


def read_holdout_group_ids(
    holdout_group_ids_path: os.PathLike[str] | str,
    expected_split_strategy: GroupedSplitStrategy | None = None,
) -> tuple[str, ...]:
    """Read one group per line, a JSON list, or a split-report JSON object."""

    if expected_split_strategy not in {None, "component", "interface"}:
        raise ValueError(
            "expected_split_strategy must be component, interface, or None"
        )
    holdout_group_ids_path = Path(holdout_group_ids_path)
    text = holdout_group_ids_path.read_text().strip()
    if not text:
        raise ValueError(f"Holdout group file is empty: {holdout_group_ids_path}")

    if text[0] in "[{":
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_unique_report_object,
                parse_constant=_reject_report_json_constant,
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid holdout group JSON: {holdout_group_ids_path}: {exc}"
            ) from exc
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            if "selected_holdout_group_ids" in payload:
                return _validate_split_report(payload, expected_split_strategy)
            else:
                raise ValueError(
                    "Holdout group JSON object must be a split report "
                    "containing selected_holdout_group_ids"
                )
        else:
            raise ValueError("Holdout group JSON must be a list or object")
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]

    if isinstance(values, str) or not isinstance(values, list):
        raise ValueError("Holdout groups must be represented as a list of strings")
    if any(not isinstance(value, str) for value in values):
        raise ValueError("Holdout group IDs must be strings")
    return _normalize_group_ids(
        values,
        label="holdout groups",
        require_nonempty=True,
    )


def _unique_report_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Split report JSON repeats key {key!r}")
        result[key] = value
    return result


def _reject_report_json_constant(value: str) -> None:
    raise ValueError(f"Split report JSON contains nonfinite number {value}")


def _report_group_ids(value: object, label: str, *, required=False) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Split report {label} must be a list of strings")
    return _normalize_group_ids(value, label=label, require_nonempty=required)


def _validate_split_report(
    payload: dict[str, object],
    expected_split_strategy: GroupedSplitStrategy | None = None,
) -> tuple[str, ...]:
    """Check the versioned report without resolving its historical artifact paths."""

    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != SPLIT_REPORT_SCHEMA_VERSION
    ):
        raise ValueError("Split report has an unsupported or missing schema_version")
    if set(payload) != _REPORT_FIELDS:
        raise ValueError("Split report has missing or unexpected fields")
    split_strategy = payload["split_strategy"]
    if not isinstance(split_strategy, str) or split_strategy not in {
        "component",
        "interface",
    }:
        raise ValueError(
            "Split report must declare component or interface split_strategy"
        )
    if (
        expected_split_strategy is not None
        and split_strategy != expected_split_strategy
    ):
        raise ValueError(
            f"Split report split_strategy {split_strategy!r} does not "
            f"match requested split_strategy {expected_split_strategy!r}"
        )
    fraction = payload["validation_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError("Split report validation_fraction must be a finite number")
    _validate_fraction(fraction)
    if type(payload["seed"]) is not int:
        raise ValueError("Split report seed must be an integer")
    for name in ("num_records", "num_train_records", "num_validation_records"):
        if type(payload[name]) is not int or payload[name] <= 0:
            raise ValueError(f"Split report {name} must be a positive integer")
    assignments = payload["assignments"]
    if not isinstance(assignments, list) or len(assignments) != payload["num_records"]:
        raise ValueError("Split report assignments must match num_records")

    selected = _report_group_ids(
        payload["selected_holdout_group_ids"],
        "selected_holdout_group_ids",
        required=True,
    )
    selected_set = set(selected)
    available_groups = set()
    reference_splits = {}
    artifact_group_splits = {}
    train_count = 0
    attribute = (
        "component_group_ids"
        if split_strategy == "component"
        else "interface_group_ids"
    )
    for index, assignment in enumerate(assignments, start=1):
        if not isinstance(assignment, dict) or set(assignment) != _ASSIGNMENT_FIELDS:
            raise ValueError("Split report assignment has missing or unexpected fields")
        if type(assignment["row_index"]) is not int or assignment["row_index"] != index:
            raise ValueError(
                "Split report row_index must preserve one-based input order"
            )
        for name in ("artifact_reference", "artifact_group_id"):
            if not isinstance(assignment[name], str) or not assignment[name].strip():
                raise ValueError(f"Split report {name} must be a nonempty string")
        group_ids = {
            name: _report_group_ids(assignment[name], name)
            for name in ("component_group_ids", "interface_group_ids")
        }
        split = assignment["split"]
        if not isinstance(split, str) or split not in {"train", "valid"}:
            raise ValueError("Split report split must be train or valid")
        if split == "train":
            train_count += 1
            if selected_set.intersection(group_ids[attribute]):
                raise ValueError(
                    "Split report places a selected holdout group in training"
                )
        available_groups.update(group_ids[attribute])
        for key, owners in (
            (assignment["artifact_reference"], reference_splits),
            (assignment["artifact_group_id"].strip(), artifact_group_splits),
        ):
            if owners.setdefault(key, split) != split:
                raise ValueError(
                    "Split report places atomic artifact peers in both outputs"
                )
    if (
        train_count != payload["num_train_records"]
        or len(assignments) - train_count != payload["num_validation_records"]
    ):
        raise ValueError("Split report assignment counts do not match declared counts")
    if not selected_set.issubset(available_groups):
        raise ValueError("Split report selected_holdout_group_ids contains unknown IDs")
    return selected


def write_grouped_split_report(
    result: GroupedSplitResult,
    split_report_path: os.PathLike[str] | str,
) -> None:
    """Write a reproducible JSON report including every row assignment."""

    if not isinstance(result, GroupedSplitResult):
        raise TypeError("result must be a GroupedSplitResult")
    payload = {
        "schema_version": SPLIT_REPORT_SCHEMA_VERSION,
        "split_strategy": result.split_strategy,
        "validation_fraction": result.validation_fraction,
        "seed": result.seed,
        "selected_holdout_group_ids": list(result.selected_holdout_group_ids),
        "num_records": len(result.records),
        "num_train_records": len(result.train_artifact_references),
        "num_validation_records": len(result.validation_artifact_references),
        "assignments": [
            {
                "row_index": row_index,
                "artifact_reference": os.fspath(record.artifact_reference),
                "artifact_group_id": record.artifact_group_id,
                "component_group_ids": list(record.component_group_ids),
                "interface_group_ids": list(record.interface_group_ids),
                "split": assignment,
            }
            for row_index, (record, assignment) in enumerate(
                zip(result.records, result.assignments),
                start=1,
            )
        ],
    }
    payload["selected_holdout_group_ids"] = list(_validate_split_report(payload))
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    split_report_path = Path(split_report_path)
    split_report_path.parent.mkdir(parents=True, exist_ok=True)
    split_report_path.write_text(serialized)
