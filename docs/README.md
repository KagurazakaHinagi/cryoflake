# cryoFlake documentation

cryoFlake manages cryo-EM dataset preparation and storage. Select the built-in
`default` adapter to turn raw maps and structures into training artifacts, or
supply a custom preprocessing callable. Sequence-model inference uses an
application-supplied adapter.

## Choose a guide

| Task | Documentation |
| --- | --- |
| Install and run the CLI | [User guide](user-guide.md) |
| Prepare manifests, download inputs, and produce training artifacts | [User guide](user-guide.md) |
| Add preprocessing and embedding adapters | [User guide](user-guide.md), [Python API](api-reference.md) |
| Resume interrupted jobs, configure workers, or publish to S3 | [User guide](user-guide.md) |
| Configure paths, pipelines, splitting, and environment settings | [Configuration reference](configuration.md) |
| Produce or consume training arrays | [Data contracts](data-contracts.md) |
| Implement an NPZ+JSON reader or inspect artifact integrity | [Artifact format specification](artifact-format.md) |
| Use the package from Python | [API reference](api-reference.md) |
| Contribute changes and run checks | [Development](development.md) |

## Two contracts to keep separate

The [artifact format](artifact-format.md) defines how portable mappings and
NumPy arrays are serialized, identified, hashed, and reconstructed. A generic
mapping can be valid for storage without representing a training artifact.

The [training data contract](data-contracts.md) defines the grid, coordinates,
labels, full-atom arrays, component targets, and optional embeddings that
preprocessing and artifact validation require. Adapter authors must satisfy
both contracts. Array shapes alone do not establish biological alignment;
the adapter also owns the model's residue and atom conventions.

Examples use the `cryoflake` Python import and CLI command. The display name is
cryoFlake. Relative CLI and YAML paths resolve from the working directory;
relative file references inside manifests and artifact lists resolve from the
containing file as described in the contracts reference.
