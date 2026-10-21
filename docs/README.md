# cryoFlake documentation

cryoFlake manages cryo-EM dataset preparation and storage. It does not ship a
raw preprocessing model or an embedding model; those are supplied by your
application through explicit adapters.

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
| Inspect an artifact's structure, sequence, and density in MolStar or ChimeraX | [Visualization](visualization.md) |
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
