"""Validated reconstruction for versioned NPZ+JSON artifacts."""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from cryoflake.storage.codec import (
    _collect_array_specs,
    _selected_tree,
    read_metadata,
)
from cryoflake.storage.validation import _validate_store_archive
from cryoflake.types import FASTASequence


def _load_array_store(
    data: bytes | Path | None,
    store: str,
    selected_specs: Mapping[
        tuple[str, str],
        tuple[np.dtype, tuple[int, ...]],
    ],
    declared_specs: Mapping[
        tuple[str, str],
        tuple[np.dtype, tuple[int, ...]],
    ],
    expected_generation: str,
    expected_hash: str,
    verified_hash: str | None,
    require_store: bool = False,
) -> dict[tuple[str, str], np.ndarray]:
    selected = {
        key: descriptor
        for (array_store, key), descriptor in selected_specs.items()
        if array_store == store
    }
    if not selected and not require_store:
        return {}
    if data is None:
        raise ValueError(f"NPZ+JSON {store} array store is required")
    # Keep one open file across hashing, validation, and reconstruction. A
    # concurrent atomic writer may replace the path between these operations.
    with (
        io.BytesIO(data) if isinstance(data, bytes) else Path(data).open("rb")
    ) as source:
        if verified_hash is None:
            verified_hash = hashlib.file_digest(source, "sha256").hexdigest()
        if verified_hash != expected_hash:
            raise ValueError(f"NPZ+JSON {store} store SHA-256 does not match metadata")
        source.seek(0)
        with zipfile.ZipFile(source) as archive:
            arrays = _validate_store_archive(
                archive,
                store,
                declared_specs,
                expected_generation,
                selected_keys=selected,
                load_arrays=True,
            )
    return {(store, key): array for key, array in arrays.items()}


def _reconstruct_value(
    node: dict[str, object],
    arrays: Mapping[tuple[str, str], np.ndarray],
) -> object:
    kind = node["kind"]
    if kind in {"array", "bytes"}:
        value = arrays[(node["store"], node["key"])]
        return value.tobytes() if kind == "bytes" else value
    if kind == "mapping":
        return {
            key: _reconstruct_value(child, arrays)
            for key, child in node["items"].items()
        }
    if kind == "namedtuple":
        values = {
            key: _reconstruct_value(child, arrays)
            for key, child in node["items"].items()
        }
        if node.get("type") == "FASTASequence" and set(values) == {
            "sequence",
            "chain_count",
            "chain_ids",
        }:
            return FASTASequence(**values)
        return values
    if kind == "sequence":
        values = [_reconstruct_value(child, arrays) for child in node["items"]]
        return tuple(values) if node.get("sequence_type") == "tuple" else values
    if kind == "scalar":
        return node["value"]
    raise AssertionError(f"Validated tree has unknown kind: {kind!r}")


def read_npz_json(
    metadata_data: bytes | Path,
    core_data: bytes | Path | None,
    sequence_data: bytes | Path | None = None,
    *,
    root_keys: Sequence[str] | None = None,
    verified_store_hashes: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Load and validate a complete payload or selected top-level fields."""

    return _read_npz_json_metadata(
        read_metadata(metadata_data),
        core_data,
        sequence_data,
        root_keys=root_keys,
        verified_store_hashes=verified_store_hashes,
    )


def _read_npz_json_metadata(
    metadata: Mapping[str, object],
    core_data: bytes | Path | None,
    sequence_data: bytes | Path | None = None,
    *,
    root_keys: Sequence[str] | None = None,
    verified_store_hashes: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Reconstruct from metadata already validated by the storage facade."""

    tree = _selected_tree(metadata["tree"], root_keys)
    selected_specs: dict[
        tuple[str, str],
        tuple[np.dtype, tuple[int, ...]],
    ] = {}
    _collect_array_specs(tree, selected_specs)
    declared_specs: dict[
        tuple[str, str],
        tuple[np.dtype, tuple[int, ...]],
    ] = {}
    _collect_array_specs(metadata["tree"], declared_specs)
    verified_store_hashes = verified_store_hashes or {}
    arrays = _load_array_store(
        core_data,
        "core",
        selected_specs,
        declared_specs,
        metadata["store_generations"]["core"],
        metadata["store_hashes"]["core"],
        verified_store_hashes.get("core"),
        require_store=root_keys is None,
    )
    arrays.update(
        _load_array_store(
            sequence_data,
            "sequence",
            selected_specs,
            declared_specs,
            metadata["store_generations"].get("sequence", ""),
            metadata["store_hashes"].get("sequence", ""),
            verified_store_hashes.get("sequence"),
        )
    )
    payload = _reconstruct_value(tree, arrays)
    if not isinstance(payload, Mapping):
        raise AssertionError("Validated NPZ+JSON root did not reconstruct a mapping")
    return payload
