# NPZ+JSON artifact format

cryoFlake stores one artifact as a directory, or as an S3 object prefix, containing JSON metadata and NumPy arrays. Use `load_preprocessed_artifact()` as the loader and `write_preprocessed_artifact()` as the writer. These functions are available from `cryoflake` and `cryoflake.storage`.

This reference specifies storage format **`cryoflake-preprocessed`, version `1`**. The [data contracts](data-contracts.md) specify the additional fields, shapes, units, and relationships required for a training artifact. A mapping can round-trip through storage without being a valid training artifact.

For an end-to-end workflow, see the [user guide](user-guide.md). See [configuration](configuration.md) for Hydra configuration and S3 credentials, and the [API reference](api-reference.md) for function signatures.

## Members and references

<!-- doc-example: artifact-layout; mode=reference -->
```text
example/
├── metadata.json
├── core.npz
└── sequence.npz      # Present only when sequence roots contain arrays or bytes.
```

| Member | Contents |
| --- | --- |
| `metadata.json` | Format identifiers, array-store hashes and generations, and the complete payload tree. |
| `core.npz` | Arrays beneath all top-level keys except `sequence` and `sequence_mask`, plus a generation marker. Always present, even for an empty or scalar-only payload. |
| `sequence.npz` | Arrays beneath the top-level `sequence` and `sequence_mask` keys, plus a separate generation marker. Omitted when these roots contain no arrays or bytes. |

Pass either the directory/prefix or its `metadata.json` member to the public storage APIs:

<!-- doc-example: artifact-references; mode=python -->
```python
load_preprocessed_artifact("artifacts/example")
load_preprocessed_artifact("artifacts/example/metadata.json")
load_preprocessed_artifact("s3://my-bucket/artifacts/example")
```

An individual NPZ member is not an artifact reference. Artifact roots must be dedicated directories; writing to the current working directory or a filesystem root is rejected. S3 is the supported remote transport.

## Metadata object

`metadata.json` is a JSON object with these required fields:

| Field | Type | Contract |
| --- | --- | --- |
| `format` | string | Exactly `"cryoflake-preprocessed"`. |
| `format_version` | integer | Exactly `1`; a JSON boolean is not an integer version. |
| `split_sequence` | boolean | Exactly `true`. |
| `files` | object | Exactly `{"core": "core.npz"}`, optionally with `"sequence": "sequence.npz"`. Other store names or member filenames are rejected. |
| `store_generations` | object | The same keys as `files`. Each value is a 32-character hexadecimal generation identifier. Writers generate a fresh lowercase UUID hex value for every rewritten store. |
| `store_hashes` | object | The same keys as `files`. Each value is the SHA-256 digest of the entire corresponding NPZ file, represented by 64 lowercase hexadecimal characters. Readers compare the text to the computed lowercase digest; uppercase digests do not match. |
| `tree` | object | A root node of kind `mapping`, as specified below. |

Metadata is limited to **16,777,216 bytes (16 MiB)**. Readers reject duplicate JSON keys, nonstandard JSON numbers such as `NaN` and `Infinity`, invalid tree nodes, and unsupported format identifiers or versions. Writers emit compact JSON with sorted object keys and finite scalar numbers.

The packaged Hydra contract is [contracts/artifact/npz_json.yaml](../src/cryoflake/configs/contracts/artifact/npz_json.yaml). Its identifiers, filenames, and size limit describe the supported wire format. Overriding them to other values is rejected by configuration validation.

## Payload tree

The payload root must be a mapping with string keys. Each entry in `tree.items` describes the corresponding Python value. Children recursively use the following node schemas:

| `kind` | Required fields beyond `kind` | Reconstructed Python value |
| --- | --- | --- |
| `mapping` | `items`: object mapping string keys to child nodes | `dict` |
| `array` | `store`: `"core"` or `"sequence"`; `key`: string; `dtype`: NumPy dtype string; `shape`: list of nonnegative integers | `numpy.ndarray` |
| `bytes` | The same fields as `array`, with `dtype` equal to NumPy's `uint8` dtype string (`"\|u1"`) | `bytes` |
| `sequence` | `sequence_type`: `"list"` or `"tuple"`; `items`: list of child nodes | `list` or `tuple`, respectively |
| `namedtuple` | `type`: original class name; `items`: object mapping field names to child nodes | A `FASTASequence` for the recognized schema below; otherwise a `dict` of fields. |
| `scalar` | `value`: JSON null, boolean, integer, finite number, or string | `None`, `bool`, `int`, `float`, or `str` |

For example, this tree describes an array and a label:

<!-- doc-example: artifact-tree; mode=data -->
```json
{
  "kind": "mapping",
  "items": {
    "values": {
      "kind": "array",
      "store": "core",
      "key": "values",
      "dtype": "<f4",
      "shape": [2, 3]
    },
    "label": {"kind": "scalar", "value": "example"}
  }
}
```

The example is a `tree` field, not a complete metadata document. A valid document also needs the file, hash, and generation fields tied to its actual NPZ files.

### Array keys and routing

An array's `key` is derived from its full path in the tree. Percent-encode each path component with UTF-8 URI encoding and join components with `/`. Mapping keys and namedtuple field names are path components; sequence indices use decimal strings. For example:

| Payload location | NPZ key | ZIP member |
| --- | --- | --- |
| `payload["values"]` | `values` | `values.npy` |
| `payload["backbone_coordinates"]["CA"]` | `backbone_coordinates/CA` | `backbone_coordinates/CA.npy` |
| `payload["custom/root"]["x y"][0]` | `custom%2Froot/x%20y/0` | `custom%2Froot/x%20y/0.npy` |

`store` is `sequence` only when the first path component is exactly `sequence` or `sequence_mask`; it is `core` otherwise. A nested key named `sequence` under another root stays in `core`. Readers verify both the encoded path and this routing. Duplicate array references are rejected.

The NPZ key `__cryoflake_store_generation__` is reserved. Each declared archive includes a member named `__cryoflake_store_generation__.npy` containing a scalar Unicode NumPy array whose value matches that store's `store_generations` entry. This marker is not part of the payload tree or returned payload.

### Supported values and normalization

Arrays preserve the plain dtype represented by `dtype.str`, shape, and values, including byte order. Both C-order and Fortran-order NPY arrays can be read. Object dtypes and structured dtypes are rejected; loading never enables NumPy Pickle deserialization. Numeric (including complex), boolean, fixed-width string, datetime, timedelta, and unstructured void NumPy dtypes can be stored. Zero-length dimensions, zero-width dtypes, and scalar arrays are supported. Custom dtype metadata, array subclasses, arbitrary strides, and shared-memory relationships are not preserved.

`bytes` and `bytearray` inputs are stored as one-dimensional `uint8` arrays and load as immutable `bytes`. Mapping implementations load as plain dictionaries. Tuples retain their container type; other supported `collections.abc.Sequence` inputs, such as `range`, load as lists. NumPy scalar values become Python scalars when their `.item()` result is a supported scalar. A complex array is supported, but a Python or NumPy complex scalar is not. Sets, paths, dataclass instances, and other arbitrary Python objects are rejected. Cyclic containers are not supported.

Namedtuple values retain a field mapping and a type name. The reader reconstructs `cryoflake.FASTASequence` only for a node whose type is `FASTASequence` and whose fields are exactly `sequence`, `chain_count`, and `chain_ids`. It does not import or instantiate other classes from metadata.

The top-level key `density_grid` has an additional writer rule: it is validated as a nonempty, finite numeric 3D grid and normalized to a `float32` array before serialization. Supply a NumPy array, or an NPY/NPZ input as bytes, a path, or a binary file-like object; an NPZ input must contain `arr_0`. A nested Python list is not a supported grid input. This rule applies to generic payloads as well as training artifacts. Other array roots do not gain training-specific shape or finiteness checks merely by being written: they may retain `NaN` or infinity even though scalar JSON numbers must be finite. See the [data contracts](data-contracts.md) for semantic validation.

## NPZ integrity and validation

Writers create ZIP archives with DEFLATE-compressed NPY members. Readers also accept other ZIP compression methods supported by the running Python installation. Readers accept NPY versions 1.0, 2.0, and 3.0, subject to NumPy's default NPY header-size limit. The metadata dtype is NumPy's `dtype.str`, such as `<f4`, `<i8`, `|b1`, or `<U32`; `shape` must match the NPY header exactly and be representable by NumPy.

For each loaded store, the loader:

1. Checks the SHA-256 digest of the complete archive against metadata.
2. Rejects duplicate archive members, missing declared arrays, and undeclared extra members.
3. Validates the generation marker and its exact agreement with metadata.
4. Checks the dtype, shape, and exact serialized size of each selected array before allocating it, then reads its payload. ZIP decoding also checks CRCs for members that are read.

A generation member larger than 1,024 bytes, a nonscalar generation value, or a non-Unicode generation dtype is rejected. Invalid or mixed metadata/store generations fail with an error instead of silently yielding a combined payload.

`preprocessed_artifact_exists()` checks valid metadata and the presence of its declared members. It does **not** check their hashes or the training contract. To check a complete training artifact, use `validate_preprocessed_artifact()` or the CLI validation workflow described in the [user guide](user-guide.md). Generic mappings should be checked by loading them; they need not satisfy the training contract.

For custom producers that need full storage validation without the training contract, `cryoflake.storage.validation.validate_npz_json_store_files(metadata_data, core_path, sequence_path=None)` checks every declared member without retaining reconstructed payload arrays. `metadata_data` accepts JSON bytes or a metadata `Path`; supply `sequence_path` when metadata declares that store. This is a local-file operation; the high-level loader handles S3 downloads.

The metadata limit is not a limit on array bytes, array dimensions, or total process memory. Load only the roots needed for an application and size concurrent consumers for the data they retain.

## Selected reads

`load_preprocessed_artifact(artifact_reference, root_keys=[...])` returns only the requested top-level fields. `root_keys` must be a collection of strings, not a single string. Unknown keys are ignored; an empty selection returns an empty mapping. Nested field selection is not supported.

`include_sequence_embeddings=False` removes `sequence` and `sequence_mask` from the requested roots, even when they were explicitly listed in `root_keys`. This permits core-only use when an unneeded sequence store is unavailable.

The complete metadata tree is always validated. Only stores needed for selected array or bytes nodes are opened or downloaded. A default full load also requires `core.npz` for a scalar-only payload. Within a loaded store, its whole-file hash, member list, and generation are checked, but only selected arrays have their NPY headers and payloads decoded. A selected read therefore does not replace full validation of unselected fields or stores.

S3 reads download the required stores into temporary local files; returned arrays remain usable after those files are removed. The loader does not return memory-mapped arrays or open file handles.

## Sequence updates

`replace_preprocessed_artifact_sequence(artifact_reference, payload)` replaces the `sequence` and `sequence_mask` roots and the split sequence archive while preserving `core.npz` byte for byte. It requires a nonempty real numeric embedding matrix of shape `(sequence_length, feature_dim)` and converts it to finite `float32` values. Boolean and complex embeddings are rejected. A missing or `None` mask becomes an all-true boolean vector; an explicit mask is flattened and must have matching length, contain only boolean or real numeric 0/1 values, and contain at least one true value.

Optional `unified_seq` and `unified_seq_len` inputs are checked against embedding length. The replacement writes only normalized `sequence` and `sequence_mask` fields. It does not update sequence metadata or other roots already stored in `core.npz`; the caller must keep them consistent with the [training contract](data-contracts.md).

Replacement validates the existing JSON metadata and the supplied embedding payload. It does not read or validate the existing array stores. Use full artifact validation when the integrity of an existing training artifact needs to be established before or after an update.

The public replacement API requires embeddings and does not remove them. To create an artifact without sequence roots, write the desired complete payload with `write_preprocessed_artifact()`.

## Write ordering and concurrent access

Local writers stage and `fsync` new files, then replace the array stores before replacing `metadata.json` last. If an ordinary write or commit operation fails, the writer attempts to restore the previous files and removes its temporary files. Local writes, uploads, and sequence replacements sharing an artifact root are serialized within one process. An upload also holds the local source's write lock while validating and sending its members.

An artifact update spans multiple files. It is not a filesystem-wide atomic transaction, and process-local locks do not coordinate independent writer processes. A concurrent reader can encounter a transient hash or generation mismatch; retry after the writer finishes. A process termination or power loss during commit can leave a partial update. Use separate roots for independent producers and publish completed artifacts rather than having multiple processes overwrite one root.

S3 uploads validate local stores, upload `core.npz` and any `sequence.npz`, and publish `metadata.json` last. Before overwriting an existing remote artifact, the uploader replaces its metadata with an incomplete marker. Readers reject that marker while the update is in progress. A failed upload can leave that marker or partial stores; retry the upload to publish a complete artifact. S3 sequence replacement follows the same metadata-last rule while uploading only the new sequence store.

The package does not provide a distributed writer lock or a multi-object S3 transaction. Coordinate writers to a shared S3 prefix externally. Unreferenced remote objects can remain after an overwrite; the metadata declaration determines which stores belong to the artifact.

## Round-trip example

This example writes a generic payload, selects individual roots, and replaces embeddings without rewriting the core store. It is a storage example; it is not a complete training artifact.

<!-- doc-example: artifact-roundtrip; mode=python -->
```python
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cryoflake import (
    load_preprocessed_artifact,
    replace_preprocessed_artifact_sequence,
    write_preprocessed_artifact,
)

with TemporaryDirectory() as temporary_directory:
    root = Path(temporary_directory) / "example"
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    write_preprocessed_artifact(
        root,
        {
            "values": values,
            "label": "example",
            "notes": ("portable", b"bytes"),
            "sequence": np.ones((2, 4), dtype=np.float32),
            "sequence_mask": np.array([True, False]),
        },
    )

    loaded = load_preprocessed_artifact(root / "metadata.json")
    np.testing.assert_array_equal(loaded["values"], values)
    assert loaded["notes"] == ("portable", b"bytes")
    assert load_preprocessed_artifact(root, root_keys=["label"]) == {"label": "example"}
    core_only = load_preprocessed_artifact(root, include_sequence_embeddings=False)
    assert "sequence" not in core_only

    core_hash = hashlib.sha256((root / "core.npz").read_bytes()).digest()
    replace_preprocessed_artifact_sequence(root, {"sequence": np.full((2, 8), 3.0)})
    assert hashlib.sha256((root / "core.npz").read_bytes()).digest() == core_hash
    updated = load_preprocessed_artifact(root, root_keys=["sequence_mask"])
    np.testing.assert_array_equal(updated["sequence_mask"], [True, True])
```
