"""Safe, versioned NPZ+JSON storage for preprocessed artifacts."""

from __future__ import annotations

import json
import uuid
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from urllib.parse import quote

import numpy as np

from cryoflake.config import artifact_contract
from cryoflake.metadata import validate_density_grid

_CONTRACT = artifact_contract()
FORMAT_VERSION = _CONTRACT.format_version
FORMAT_NAME = _CONTRACT.format_name
MAX_METADATA_BYTES = _CONTRACT.max_metadata_bytes
METADATA_FILE_NAME = _CONTRACT.metadata_file_name
CORE_FILE_NAME = _CONTRACT.core_file_name
SEQUENCE_FILE_NAME = _CONTRACT.sequence_file_name
STORE_GENERATION_KEY = _CONTRACT.generation_key
SEQUENCE_ROOT_KEYS = {"sequence", "sequence_mask"}


def _array_key(path: tuple[str, ...]) -> str:
    return "/".join(quote(part, safe="") for part in path)


def _store_array(
    value: np.ndarray,
    path: tuple[str, ...],
    stores: dict[str, dict[str, np.ndarray]],
    *,
    kind: str = "array",
) -> dict[str, object]:
    if value.dtype.hasobject or value.dtype.fields is not None:
        raise TypeError(
            f"Object or structured array is not supported at {'/'.join(path)}"
        )
    store = "sequence" if path and path[0] in SEQUENCE_ROOT_KEYS else "core"
    key = _array_key(path)
    stores[store][key] = value
    return {
        "kind": kind,
        "store": store,
        "key": key,
        "dtype": value.dtype.str,
        "shape": list(value.shape),
    }


def _flatten_value(
    value: object,
    path: tuple[str, ...],
    stores: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    if path == ("density_grid",):
        value = validate_density_grid(value)

    if isinstance(value, np.ndarray):
        return _store_array(value, path, stores)

    if isinstance(value, (bytes, bytearray)):
        array = np.frombuffer(value, dtype=np.uint8)
        if isinstance(value, bytearray):
            array = array.copy()
        return _store_array(
            array,
            path,
            stores,
            kind="bytes",
        )

    if isinstance(value, Mapping):
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise TypeError(
                f"NPZ+JSON mapping keys must be strings at {'/'.join(path) or '<root>'}"
            )
        return {
            "kind": "mapping",
            "items": {
                key: _flatten_value(
                    item,
                    (*path, key),
                    stores,
                )
                for key, item in value.items()
            },
        }

    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {
            "kind": "namedtuple",
            "type": type(value).__name__,
            "items": {
                field: _flatten_value(
                    getattr(value, field),
                    (*path, field),
                    stores,
                )
                for field in value._fields
            },
        }

    if isinstance(value, Sequence) and not isinstance(value, str):
        return {
            "kind": "sequence",
            "sequence_type": "tuple" if isinstance(value, tuple) else "list",
            "items": [
                _flatten_value(
                    item,
                    (*path, str(index)),
                    stores,
                )
                for index, item in enumerate(value)
            ],
        }

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "value": value}
    raise TypeError(
        f"Unsupported {type(value).__name__} at {'/'.join(path) or '<root>'}"
    )


def flatten_payload(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    if not isinstance(payload, Mapping):
        raise TypeError("NPZ+JSON payload root must be a mapping")
    stores: dict[str, dict[str, np.ndarray]] = {"core": {}, "sequence": {}}
    tree = _flatten_value(payload, (), stores)
    store_generations = {}
    for store in ("core", "sequence"):
        if store == "sequence" and not stores[store]:
            continue
        if STORE_GENERATION_KEY in stores[store]:
            raise ValueError(f"Payload array path reserves {STORE_GENERATION_KEY!r}")
        generation = uuid.uuid4().hex
        store_generations[store] = generation
        stores[store][STORE_GENERATION_KEY] = np.asarray(generation)
    metadata = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "split_sequence": True,
        "files": {
            "core": CORE_FILE_NAME,
            **({"sequence": SEQUENCE_FILE_NAME} if stores["sequence"] else {}),
        },
        "store_generations": store_generations,
        "tree": tree,
    }
    return metadata, stores


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"NPZ+JSON metadata repeats key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"NPZ+JSON metadata contains invalid number {value}")


def read_metadata(data: bytes | Path) -> dict[str, object]:
    """Read and validate metadata for a preprocessed artifact."""

    if isinstance(data, bytes):
        serialized = data
        if len(serialized) > MAX_METADATA_BYTES:
            raise ValueError(
                f"NPZ+JSON metadata exceeds the {MAX_METADATA_BYTES}-byte format limit"
            )
    else:
        data = Path(data)
        if data.stat().st_size > MAX_METADATA_BYTES:
            raise ValueError(
                f"NPZ+JSON metadata exceeds the {MAX_METADATA_BYTES}-byte format limit"
            )
        with data.open("rb") as input_file:
            serialized = input_file.read(MAX_METADATA_BYTES + 1)
        if len(serialized) > MAX_METADATA_BYTES:
            raise ValueError(
                f"NPZ+JSON metadata exceeds the {MAX_METADATA_BYTES}-byte format limit"
            )
    value = json.loads(
        serialized,
        object_pairs_hook=_json_object_without_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("NPZ+JSON metadata must be a JSON object")
    if value.get("format") != FORMAT_NAME:
        raise ValueError(f"Unsupported NPZ+JSON format: {value.get('format')!r}")
    if (
        type(value.get("format_version")) is not str
        or value.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError(
            f"Unsupported NPZ+JSON format version: {value.get('format_version')!r}"
        )
    tree = value.get("tree")
    if not isinstance(tree, dict) or tree.get("kind") != "mapping":
        raise ValueError("NPZ+JSON root tree must be a mapping")
    if value.get("split_sequence") is not True:
        raise ValueError("NPZ+JSON artifacts require split_sequence=true")
    files = value.get("files")
    if not isinstance(files, dict) or files.get("core") != CORE_FILE_NAME:
        raise ValueError(
            f"NPZ+JSON metadata must declare {CORE_FILE_NAME!r} as its core store"
        )
    if set(files) - {"core", "sequence"}:
        raise ValueError("NPZ+JSON metadata declares an unsupported array store")
    if "sequence" in files and files["sequence"] != SEQUENCE_FILE_NAME:
        raise ValueError("NPZ+JSON metadata declares an unsupported sequence store")
    generations = value.get("store_generations")
    if not isinstance(generations, dict) or set(generations) != set(files):
        raise ValueError("NPZ+JSON store generations must match declared stores")
    for store, generation in generations.items():
        if not isinstance(generation, str) or len(generation) != 32:
            raise ValueError(f"NPZ+JSON {store} store has an invalid generation")
        try:
            int(generation, 16)
        except ValueError as exc:
            raise ValueError(
                f"NPZ+JSON {store} store has an invalid generation"
            ) from exc
    store_hashes = value.get("store_hashes")
    if not isinstance(store_hashes, dict) or set(store_hashes) != set(files):
        raise ValueError("NPZ+JSON store hashes must match declared stores")
    for store, store_hash in store_hashes.items():
        if not isinstance(store_hash, str) or len(store_hash) != 64:
            raise ValueError(f"NPZ+JSON {store} store has an invalid SHA-256 hash")
        try:
            int(store_hash, 16)
        except ValueError as exc:
            raise ValueError(
                f"NPZ+JSON {store} store has an invalid SHA-256 hash"
            ) from exc
    specs = _validate_tree_schema(tree)
    if any(store == "sequence" for store, _ in specs) != ("sequence" in files):
        raise ValueError(
            "NPZ+JSON sequence store declaration does not match its array tree"
        )
    return value


def _selected_tree(
    tree: dict[str, object],
    root_keys: Sequence[str] | None,
) -> dict[str, object]:
    if root_keys is None:
        return tree
    if (
        isinstance(root_keys, (str, bytes))
        or not isinstance(root_keys, Collection)
        or not all(isinstance(key, str) for key in root_keys)
    ):
        raise ValueError("root_keys must be a collection of strings")
    items = tree["items"]
    selected = {key: items[key] for key in root_keys if key in items}
    return {"kind": "mapping", "items": selected}


def _collect_array_specs(
    node: object,
    specs: dict[tuple[str, str], tuple[np.dtype, tuple[int, ...]]],
) -> None:
    if not isinstance(node, dict):
        raise ValueError("NPZ+JSON tree node must be an object")
    kind = node.get("kind")
    if not isinstance(kind, str):
        raise ValueError("NPZ+JSON tree node has an invalid kind")
    if kind in {"array", "bytes"}:
        store = node.get("store")
        key = node.get("key")
        dtype_text = node.get("dtype")
        shape_value = node.get("shape")
        if (
            not isinstance(store, str)
            or store not in {"core", "sequence"}
            or not isinstance(key, str)
        ):
            raise ValueError("Array node has invalid store or key")
        if key == STORE_GENERATION_KEY:
            raise ValueError("Array node declares the reserved store generation key")
        if not isinstance(dtype_text, str):
            raise ValueError(f"Array {key!r} has invalid dtype")
        try:
            dtype = np.dtype(dtype_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Array {key!r} has invalid dtype") from exc
        if dtype.hasobject or dtype.fields is not None:
            raise ValueError(f"Array {key!r} declares an object or structured dtype")
        if not isinstance(shape_value, list) or not all(
            type(axis) is int and axis >= 0 for axis in shape_value
        ):
            raise ValueError(f"Array {key!r} has invalid shape")
        identity = (store, key)
        if identity in specs:
            raise ValueError(f"Duplicate array reference: {store}/{key}")
        specs[identity] = (dtype, tuple(shape_value))
        return
    if kind in {"mapping", "namedtuple"}:
        items = node.get("items")
        if not isinstance(items, dict):
            raise ValueError(f"{kind} node has invalid items")
        for child in items.values():
            _collect_array_specs(child, specs)
        return
    if kind == "sequence":
        items = node.get("items")
        if not isinstance(items, list):
            raise ValueError("Sequence node has invalid items")
        for child in items:
            _collect_array_specs(child, specs)
        return
    if kind == "scalar":
        if "value" not in node:
            raise ValueError("Scalar node is missing its value")
        return
    raise ValueError(f"Unsupported NPZ+JSON tree node kind: {kind!r}")


def _validate_tree_schema(
    tree: dict[str, object],
) -> dict[tuple[str, str], tuple[np.dtype, tuple[int, ...]]]:
    def validate_node(
        node: object,
        path: tuple[str, ...],
        expected_store: str,
    ) -> None:
        if not isinstance(node, dict):
            raise ValueError("NPZ+JSON tree node must be an object")
        kind = node.get("kind")
        if not isinstance(kind, str):
            raise ValueError("NPZ+JSON tree node has an invalid kind")
        if kind in {"array", "bytes"}:
            if node.get("store") != expected_store:
                raise ValueError(
                    "NPZ+JSON array store does not match its top-level root"
                )
            if node.get("key") != _array_key(path):
                raise ValueError("NPZ+JSON array key does not match its tree path")
            if kind == "bytes" and node.get("dtype") != np.dtype(np.uint8).str:
                raise ValueError("NPZ+JSON bytes node must use uint8 storage")
            return
        if kind in {"mapping", "namedtuple"}:
            items = node.get("items")
            if not isinstance(items, dict) or not all(
                isinstance(key, str) for key in items
            ):
                raise ValueError(f"{kind} node has invalid items")
            if kind == "namedtuple" and not isinstance(node.get("type"), str):
                raise ValueError("Namedtuple node has an invalid type")
            for key, child in items.items():
                validate_node(child, (*path, key), expected_store)
            return
        if kind == "sequence":
            if node.get("sequence_type") not in ("list", "tuple"):
                raise ValueError("Sequence node has an invalid sequence_type")
            items = node.get("items")
            if not isinstance(items, list):
                raise ValueError("Sequence node has invalid items")
            for index, child in enumerate(items):
                validate_node(
                    child,
                    (*path, str(index)),
                    expected_store,
                )
            return
        if kind == "scalar":
            scalar = node.get("value")
            if scalar is not None and type(scalar) not in {
                bool,
                int,
                float,
                str,
            }:
                raise ValueError("Scalar node has a non-scalar value")
            if isinstance(scalar, float) and not np.isfinite(scalar):
                raise ValueError("Scalar node has a non-finite value")
            return
        raise ValueError(f"Unsupported NPZ+JSON tree node kind: {kind!r}")

    items = tree.get("items")
    if not isinstance(items, dict):
        raise ValueError("NPZ+JSON root tree must have mapping items")
    for root_key, node in items.items():
        expected_store = "sequence" if root_key in SEQUENCE_ROOT_KEYS else "core"
        validate_node(node, (root_key,), expected_store)
    specs: dict[tuple[str, str], tuple[np.dtype, tuple[int, ...]]] = {}
    _collect_array_specs(tree, specs)
    return specs


def metadata_root_keys(data: bytes | Path) -> tuple[str, ...]:
    """Return the top-level payload keys declared by metadata."""

    metadata = read_metadata(data)
    return tuple(metadata["tree"]["items"])


def required_stores(
    data: bytes | Path,
    *,
    root_keys: Sequence[str] | None = None,
) -> set[str]:
    """Return array stores needed to reconstruct the selected top-level keys."""

    return _required_stores(read_metadata(data), root_keys=root_keys)


def _required_stores(
    metadata: Mapping[str, object],
    *,
    root_keys: Sequence[str] | None = None,
) -> set[str]:
    # Metadata has already established that the store declaration and complete
    # array tree agree. A full read needs every declared store, including an
    # otherwise empty core archive containing only its generation marker.
    if root_keys is None:
        return set(metadata["files"])
    tree = _selected_tree(metadata["tree"], root_keys)

    def contains_stored_data(node: Mapping[str, object]) -> bool:
        kind = node["kind"]
        if kind in {"array", "bytes"}:
            return True
        if kind in {"mapping", "namedtuple"}:
            return any(contains_stored_data(child) for child in node["items"].values())
        if kind == "sequence":
            return any(contains_stored_data(child) for child in node["items"])
        return False

    stores = set()
    for root_key, node in tree["items"].items():
        store = "sequence" if root_key in SEQUENCE_ROOT_KEYS else "core"
        if store not in stores and contains_stored_data(node):
            stores.add(store)
    return stores


def declared_store_files(data: bytes | Path) -> dict[str, str]:
    """Return the fixed array-store members declared by metadata."""

    metadata = read_metadata(data)
    return dict(metadata["files"])


def _metadata_bytes(metadata: Mapping[str, object]) -> bytes:
    return json.dumps(
        metadata,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def incomplete_metadata_bytes() -> bytes:
    """Return a marker that makes an interrupted remote overwrite fail closed."""

    return json.dumps(
        {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "state": "incomplete",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
