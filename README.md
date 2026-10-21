# cryoFlake

cryoFlake prepares, stores, and validates cryo-EM datasets. It provides EMDB
manifests and downloads, Biotite structure and FASTA readers, molecular targets,
sequence sidecars, dataset splits, resumable preprocessing, and local or
S3-compatible NPZ+JSON storage. The built-in `default` preprocessing adapter
produces training artifacts without model weights. Applications can supply
custom preprocessing and optional embedding-model adapters.

## Get started

Use Python 3.12 or newer. From this checkout:

<!-- doc-example: readme-install; mode=setup -->
```bash
uv sync --locked
uv run cryoflake --help
```

Prepare a manifest from an association CSV with `EMD-` and `Associated PDB`
columns:

<!-- doc-example: readme-manifest; mode=cli -->
```bash
uv run cryoflake prepare-manifest \
  --source-csv data/source.csv --manifest-csv data/manifest.csv \
  --maps-dir data/maps --structures-dir data/structures
```

Add `--download --workers 4` to fetch the source files. To preprocess them,
select `--preprocess-adapter default` and an artifact directory with
`--artifact-output-dir`. Custom adapters use `MODULE:CALLABLE` instead of
`default`. The [user guide](docs/user-guide.md)
covers adapter requirements, embeddings, splits, checkpoint recovery, and S3.

Inspect configuration or validate a complete training artifact:

<!-- doc-example: readme-inspect; mode=cli -->
```bash
uv run cryoflake config --pipeline preprocess
uv run cryoflake validate-artifact data/artifacts/emd_1001_1abc
```

## Read and write artifacts in Python

An artifact is a directory or S3 prefix containing `metadata.json`, `core.npz`,
and optional `sequence.npz`. Storage supports mappings of arrays and portable
values:

<!-- doc-example: readme-storage; mode=python -->
```python
import numpy as np
from cryoflake import load_preprocessed_artifact, write_preprocessed_artifact

write_preprocessed_artifact(
    "data/artifacts/example",
    {"features": np.ones((4, 8), dtype=np.float32), "label": "example"},
)
payload = load_preprocessed_artifact("data/artifacts/example")
assert payload["features"].shape == (4, 8)
```

This storage example is a generic mapping. Preprocessing and
`validate-artifact` require the complete [training data contract](docs/data-contracts.md),
including grid geometry, backbone labels, full-atom targets, and component targets.
See the [artifact format specification](docs/artifact-format.md) for serialization,
integrity checks, and selective loading.

## Documentation

Start at the [documentation index](docs/README.md), or choose a reference:

| Guide | Contents |
| --- | --- |
| [User guide](docs/user-guide.md) | Installation, CLI workflows, adapters, recovery, S3, and troubleshooting |
| [Configuration](docs/configuration.md) | Hydra groups, typed settings, defaults, overrides, and credentials |
| [Data contracts](docs/data-contracts.md) | CSVs, artifact lists, training arrays, embeddings, enums, and split reports |
| [Artifact format](docs/artifact-format.md) | Versioned NPZ+JSON metadata, tree nodes, archives, hashes, and publication rules |
| [Python API](docs/api-reference.md) | Storage, validation, readers, targets, manifests, and extension points |
| [Development](docs/development.md) | uv setup, Ruff, tests, and distribution builds |

Project YAML overrides live in [`configs/`](configs/); installed defaults and
format contracts live in [`src/cryoflake/configs/`](src/cryoflake/configs/).
Use the ignored root `.env` for local S3 credentials, with
[`.env.example`](.env.example) as a template. Installed packages can select a
file through `CRYOFLAKE_ENV_FILE`. See the user guide for credential precedence.
