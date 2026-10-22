# NPZ+JSON artifact format

An artifact stores one Python payload mapping with the metadata needed to
reconstruct it. cryoFlake writes the mapping's arrays to NumPy archives and its
structure to JSON. The same format works in a local directory or an S3 prefix.

Use `write_preprocessed_artifact()` to save an artifact and
`load_preprocessed_artifact()` to read it. Both functions are exported from
`cryoflake` and `cryoflake.storage`. This guide starts with those operations;
the metadata, array encoding, and publication rules follow below.

The storage format is **`cryoflake-preprocessed`, version `"N2610210"`**.
The separate [data contracts](data-contracts.md#complete-training-payload)
define semantic preprocessing contract **`"S2610210"`**, including the fields
a training artifact needs.
A portable mapping can round-trip through storage without satisfying that
training contract.

For the preprocessing workflow, see the [user guide](user-guide.md). Use
[configuration](configuration.md) for Hydra settings and S3 credentials, and
the [API reference](api-reference.md) for function signatures.

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

An artifact reference identifies the directory or S3 prefix, or its
`metadata.json` member. For an existing artifact, the loader accepts all three
forms below; replace the illustrative references with your own:

<!-- doc-example: artifact-references; mode=python -->
```python
from cryoflake import load_preprocessed_artifact

artifact_mapping = load_preprocessed_artifact("artifacts/example")
load_preprocessed_artifact("artifacts/example/metadata.json")
load_preprocessed_artifact("s3://my-bucket/artifacts/example")
```

Individual NPZ members cannot be used as artifact references. Choose a
dedicated directory for each artifact; writing to the current working
directory or a filesystem root is rejected. S3 is the supported remote
transport.

## Choose a validation level

Choose the check that matches how you intend to use the payload:

| Operation | What it establishes |
| --- | --- |
| `write_preprocessed_artifact()` | The mapping can be encoded in NPZ+JSON. A top-level `density_grid`, when present, is also validated and normalized before writing. |
| `load_preprocessed_artifact()` | The requested payload can be reconstructed after the storage integrity checks described below. A default full load checks every declared store; selected reads check the stores they need. |
| `validate_preprocessed_artifact()` | Storage integrity and the complete training contract, including geometry, full-atom targets, components, and any sequence embeddings present. |
| `preprocessed_artifact_exists()` | Metadata is valid and every declared member exists. It does not check store hashes or the training contract. |
| `validate_preprocessed_artifact_mapping()` | The input implements `Mapping`. It does not validate its fields or storage. |

Use loading for a generic portable mapping and semantic validation for an
artifact intended for training. Manifest preprocessing also requires the
training contract before accepting an artifact. The nested full-atom schema
is `"F2610210"` and the component extension schema is `"E2610210"`. Their
identifiers are independent of storage format `"N2610210"` and semantic
preprocessing contract `"S2610210"`. The
[data contract version table](data-contracts.md) explains their `PYYMMDDX`
format and exact-match validation.

## Round-trip example

Run this self-contained example to write a generic payload, read it back, and
replace its embeddings. It creates a temporary workspace and checks that the
core store stays byte-for-byte unchanged during sequence replacement. The
payload demonstrates storage operations; it is not a complete training artifact.

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

## Selected reads

Use `load_preprocessed_artifact(artifact_reference, root_keys=[...])` when you
need particular top-level fields. The round-trip example selects `label`
without loading the arrays under `values`. `root_keys` must be a collection of
strings, not a single string. Unknown keys are ignored; an empty selection
returns an empty mapping. Nested field selection is not supported.

Set `include_sequence_embeddings=False` to exclude `sequence` and
`sequence_mask`, including when they were explicitly listed in `root_keys`.
This permits core-only use when an unneeded sequence store is unavailable.

Every load validates the complete metadata tree. Only stores needed for the
selected array or bytes nodes are opened or downloaded; a default full load
also requires `core.npz` for a scalar-only payload. Each loaded store still
undergoes a whole-file hash, member-list, and generation check. Only selected
arrays have their NPY headers and payloads decoded, so selected reads do not
establish full validation of unselected fields or stores.

S3 reads download the required stores into temporary local files. Returned
arrays remain usable after those files are removed; the loader does not return
memory-mapped arrays or open file handles.

## Sequence updates

Use `replace_preprocessed_artifact_sequence(artifact_reference, payload)` to
replace the `sequence` and `sequence_mask` roots and their sequence archive.
The core archive remains byte-for-byte unchanged, as the round-trip example
checks.

Supply embeddings and, optionally, a mask:

- `sequence` must be a nonempty real numeric matrix of shape
  `(sequence_length, feature_dim)`. It is converted to finite `float32` values;
  boolean and complex embeddings are rejected.
- A missing or `None` `sequence_mask` becomes an all-true boolean vector. An
  explicit mask is flattened and must have matching length, contain only
  boolean or real numeric 0/1 values, and contain at least one true value.
- Optional `unified_seq` and `unified_seq_len` inputs are checked against the
  embedding length. The replacement writes only normalized `sequence` and
  `sequence_mask` fields.

Other stored roots, including sequence metadata you previously added, remain
unchanged. Keep them consistent with the [training contract](data-contracts.md).
Replacement validates the existing JSON metadata and the supplied embedding
payload, but does not read or validate the existing array stores. Use full
artifact validation to establish a training artifact's integrity before or
after an update.

The replacement API requires embeddings and does not remove them. To create an
artifact without sequence roots, write the desired payload with
`write_preprocessed_artifact()`.

## Metadata object

The remaining sections describe the wire format for readers inspecting
artifacts or implementing a producer. `metadata.json` is a JSON object with
these required fields:

| Field | Type | Contract |
| --- | --- | --- |
| `format` | string | Exactly `"cryoflake-preprocessed"`. |
| `format_version` | string | Exactly `"N2610210"`; integer versions and other identifiers are rejected. |
| `split_sequence` | boolean | Exactly `true`. |
| `files` | object | Exactly `{"core": "core.npz"}`, optionally with `"sequence": "sequence.npz"`. Other store names or member filenames are rejected. |
| `store_generations` | object | The same keys as `files`. Each value is a 32-character hexadecimal generation identifier. Writers generate a fresh lowercase UUID hex value for every rewritten store. |
| `store_hashes` | object | The same keys as `files`. Each value is the SHA-256 digest of the entire corresponding NPZ file, represented by 64 lowercase hexadecimal characters. Readers compare the text to the computed lowercase digest; uppercase digests do not match. |
| `tree` | object | A root node of kind `mapping`, as specified below. |

Metadata is limited to **16,777,216 bytes (16 MiB)**. Readers reject:

- Duplicate JSON keys.
- Nonstandard JSON numbers such as `NaN` and `Infinity`.
- Invalid tree nodes or unsupported format identifiers and versions.

Writers emit compact JSON with sorted object keys and finite scalar numbers.
The packaged Hydra contract is
[contracts/artifact/npz_json.yaml](../src/cryoflake/configs/contracts/artifact/npz_json.yaml).
Its identifiers, filenames, and size limit describe the supported wire format;
configuration validation rejects overrides to other values.

## Payload tree

The payload root must be a mapping with string keys. Each entry in `tree.items`
describes the corresponding Python value. Children recursively use the
following node schemas:

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

The example shows a `tree` field. A complete metadata document also needs the
file, hash, and generation fields tied to its actual NPZ archives.

### Array keys and routing

An array's `key` comes from its full path in the tree. Percent-encode each path
component with UTF-8 URI encoding, then join components with `/`. Mapping keys
and namedtuple field names are path components; sequence indices use decimal
strings:

| Payload location | NPZ key | ZIP member |
| --- | --- | --- |
| `payload["values"]` | `values` | `values.npy` |
| `payload["backbone_coordinates"]["CA"]` | `backbone_coordinates/CA` | `backbone_coordinates/CA.npy` |
| `payload["custom/root"]["x y"][0]` | `custom%2Froot/x%20y/0` | `custom%2Froot/x%20y/0.npy` |

`store` is `sequence` only when the first path component is exactly `sequence`
or `sequence_mask`; it is `core` otherwise. A nested key named `sequence` under
another root stays in `core`. Readers verify both the encoded path and this
routing, and reject duplicate array references.

The NPZ key `__cryoflake_store_generation__` is reserved. Each declared archive
includes a member named `__cryoflake_store_generation__.npy` containing a scalar
Unicode NumPy array. Its value matches that store's `store_generations` entry.
This marker is separate from the payload tree and is not returned to callers.

### Supported values and normalization

For NumPy arrays, storage preserves the plain dtype represented by `dtype.str`,
shape, and values, including byte order. Both C-order and Fortran-order NPY
arrays can be read.

- Supported dtypes include numeric types, including complex, plus boolean,
  fixed-width string, datetime, timedelta, and unstructured void types.
- Zero-length dimensions, zero-width dtypes, and scalar arrays are supported.
- Object and structured dtypes are rejected. Loading never enables NumPy
  Pickle deserialization.

Custom dtype metadata, array subclasses, arbitrary strides, and shared-memory
relationships are not preserved.

Other values are reconstructed as follows:

- `bytes` and `bytearray` inputs are stored as one-dimensional `uint8` arrays
  and load as immutable `bytes`.
- Mapping implementations load as plain dictionaries. Tuples retain their
  container type; other supported `collections.abc.Sequence` inputs, such as
  `range`, load as lists.
- NumPy scalar values become Python scalars when their `.item()` result is a
  supported scalar. A complex array is supported, but a Python or NumPy complex
  scalar is not.

Sets, paths, dataclass instances, and other arbitrary Python objects are
rejected. Cyclic containers are not supported.

Namedtuples retain their field mapping and type name. The reader reconstructs
`cryoflake.FASTASequence` only when the type is `FASTASequence` and the fields
are exactly `sequence`, `chain_count`, and `chain_ids`. Other namedtuple nodes
load as dictionaries; the reader does not import or instantiate classes from
metadata.

The top-level `density_grid` has an additional writer rule: it must be a
nonempty, finite real numeric 3D grid and is normalized to a `float32` array before
serialization. Supply a NumPy array, or an NPY/NPZ input as bytes, a path, or a
binary file-like object. An NPZ input must contain `arr_0`; a nested Python
list is not a supported grid input. This rule also applies to generic payloads.

Other array roots keep their supplied dtypes and do not gain training-specific
shape or finiteness checks merely by being written. They may retain `NaN` or
infinity, although scalar JSON numbers must be finite. Apply the
[data contracts](data-contracts.md) when semantic validation is needed.

## NPZ integrity and validation

Writers create ZIP archives with DEFLATE-compressed NPY members. Readers also
accept other ZIP compression methods supported by the running Python
installation.

Readers accept NPY versions 1.0, 2.0, and 3.0, subject to NumPy's default NPY
header-size limit. The metadata dtype is NumPy's `dtype.str`, such as `<f4`,
`<i8`, `|b1`, or `<U32`. The shape must match the NPY header exactly and be
representable by NumPy.

For each loaded store, the loader:

1. Checks the SHA-256 digest of the complete archive against metadata.
2. Rejects duplicate archive members, missing declared arrays, and undeclared extra members.
3. Validates the generation marker and its exact agreement with metadata.
4. Checks the dtype, shape, and exact serialized size of each selected array before allocating it, then reads its payload. ZIP decoding also checks CRCs for members that are read.

A generation member larger than 1,024 bytes, a nonscalar generation value, or a
non-Unicode generation dtype is rejected. Invalid or mixed metadata/store
generations fail with an error.

As described under validation levels, `preprocessed_artifact_exists()` checks
metadata and member presence. Use loading to check a generic mapping's storage
integrity, or `validate_preprocessed_artifact()` to check a complete training
artifact. The [user guide](user-guide.md) also describes CLI validation.

For custom producers, use
`cryoflake.storage.validation.validate_npz_json_store_files(metadata_data, core_path, sequence_path=None)`
to check every declared member without retaining reconstructed payload arrays
or applying the training contract. `metadata_data` accepts JSON bytes or a
metadata `Path`; supply `sequence_path` when metadata declares that store.
This is a local-file operation; the high-level loader handles S3 downloads.

The metadata limit does not limit array bytes, array dimensions, or total
process memory. Load the roots an application needs and size concurrent
consumers for the data they retain.

## Write ordering and concurrent access

Local writes follow this order:

1. Stage and `fsync` the new files.
2. Replace the array stores.
3. Replace `metadata.json` last.

If an ordinary write or commit fails, the writer attempts to restore the
previous files and removes its temporary files. Local writes, uploads, and
sequence replacements sharing an artifact root are serialized within one
process. An upload also holds the local source's write lock while validating
and sending its members.

An update spans multiple files and is not a filesystem-wide atomic transaction.
Process-local locks do not coordinate independent writer processes. Concurrent
readers can encounter a transient hash or generation mismatch; retry after the
writer finishes. Process termination or power loss during commit can leave a
partial update. Use separate roots for independent producers and publish
completed artifacts.

S3 uploads follow the same publication order:

1. Validate the local stores.
2. When overwriting an existing remote artifact, replace its metadata with an
   incomplete marker that readers reject.
3. Upload `core.npz` and any `sequence.npz`.
4. Publish the complete `metadata.json` last.

A failed upload can leave an incomplete marker or partial stores; retry the
upload to publish a complete artifact. S3 sequence replacement also publishes
metadata last while uploading only the new sequence store.

The package does not provide a distributed writer lock or a multi-object S3
transaction. Coordinate writers to a shared S3 prefix externally. Unreferenced
remote objects can remain after an overwrite; the metadata declaration
determines which stores belong to the artifact.
