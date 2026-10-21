# User guide

cryoFlake prepares map/structure manifests, downloads raw inputs, runs a
preprocessing adapter, validates training artifacts, and writes artifact lists.
Artifacts are local directories or S3 prefixes holding NPZ arrays and JSON
metadata. Use the built-in `default` adapter for raw preprocessing or supply
your own callable. Sequence-model inference requires an application adapter.

Use the [configuration reference](configuration.md) for settings,
[data contracts](data-contracts.md) for payload/tabular schemas, the
[artifact specification](artifact-format.md) for serialization, and the
[API reference](api-reference.md) for Python interfaces.

## Installation

Python 3.12 or newer and uv are required. From this checkout:

<!-- doc-example: guide-install; mode=setup -->
```bash
uv sync --locked
uv run cryoflake --help
uv run cryoflake config
```

To install into an application's environment, build a wheel with `uv build`,
then install it there:

<!-- doc-example: guide-wheel-install; mode=setup -->
```bash
uv pip install /absolute/path/to/cryoflake-0.1.0-py3-none-any.whl
cryoflake --help
```

Packaged configuration groups and the `default` adapter work outside the
checkout. Install custom adapters and their dependencies in the same environment. `python -m cryoflake`
also runs the CLI; each subcommand has `--help`.

## Prepare a manifest

A source association CSV requires `EMD-` and `Associated PDB` columns. This
small example exercises manifest creation without network access or an adapter:

<!-- doc-example: guide-manifest; mode=cli -->
```bash
uv run python - <<'PY'
from pathlib import Path

Path("data").mkdir(exist_ok=True)
Path("data/source.csv").write_text(
    "EMD-,Associated PDB,resolution\n"
    "EMD-1234,1ABC,3.2\n"
    "EMD-5678,2DEF,2.8\n"
)
PY
uv run cryoflake prepare-manifest \
  --source-csv data/source.csv --manifest-csv data/manifest.csv \
  --maps-dir data/maps --structures-dir data/structures
```

These are illustrative associations: choose actual dataset associations before
downloading. The command keeps one row per EMDB ID, uses the first associated
wwPDB ID, and preserves source resolution when supplied. It does not look up
missing associations online. Unassociated rows are excluded unless
`--include-without-structure` is set; preprocessing skips retained rows without
structures. See [manifest specifications](data-contracts.md) for normalization
and exact columns.

The generated CSV records URLs and local input paths. Relative paths resolve
from its directory. Existing `emd_<id>.map` or `.mrc` files are preferred to
`.map.gz`; existing `<pdb>.cif` files are preferred to `.cif.gz`. Having both
`.map` and `.mrc` files for one ID is ambiguous and raises an error.

## Download raw inputs

With real source associations, add download options:

<!-- doc-example: guide-download; mode=cli -->
```bash
uv run cryoflake prepare-manifest \
  --source-csv data/source.csv --manifest-csv data/manifest.csv \
  --maps-dir data/maps --structures-dir data/structures \
  --download --download-workers 4
```

Download mode requires both local directories. A nonempty existing destination
is skipped unless `--force-download` is set; gzip files must also yield at least one
decompressed byte. This is a quick reuse check, not a full integrity check.
Downloads use isolated temporary files and replace destinations after success.
HTTP 404 and DNS “name not found” errors warn and are returned to the batch;
combined preprocessing records affected rows as `download_failed`. Other
HTTP/network errors abort the command, including timeouts and server errors.
Downloads target the standard `.map.gz` and `.cif.gz` filenames even when
manifest creation selected an existing uncompressed input.

## Choose a preprocessing adapter

Select `--preprocess-adapter default` to use the built-in CPU adapter. Adapter
selection is explicit: `pipeline.preprocess_adapter` remains `null` until set.
The built-in adapter reads MRC maps and fitted structures, pads maps to an even
cube of at least 128 voxels per side, and resamples to the nearest supported
even grid size for `target_voxel_size`. The artifact records the actual voxel
size, which may differ from the requested spacing. It writes backbone labels,
observed full-atom targets, and component/interface targets in cryoFlake's
NPZ+JSON contract. It does not generate sequence embeddings or require model
weights. See the [full-atom conventions](data-contracts.md#full-atom-supervision).

Raw maps use `.mrc` or `.map`; structures use `.cif`, `.mmcif`, `.pdb`, or `.ent`.
The built-in adapter accepts gzip-compressed inputs with an additional `.gz`
suffix, including optional FASTA input, and uses the first structure model.
`original_structure_path`, when supplied, provides the output structure
coordinates and targets. With `fix_shift=True`, the map origin is translated
by the difference between the original and fitted structures' finite CA/P
centroids. Without an original structure there is no centroid translation.
FASTA input adds sequence records and chain associations, not embeddings.

For custom preprocessing, the CLI imports your callable using `MODULE:CALLABLE`, for example
`my_app.preprocess:run`. It must be importable in the active environment.
Its return value is ignored: it must write a complete training artifact at
`artifact_reference`, which cryoFlake loads and validates before reporting success.

The callable accepts these keyword arguments:

| Argument | Meaning |
| --- | --- |
| `map_path` | Density map path. |
| `structure_path` | Fitted structure path. |
| `artifact_reference` | Artifact output root. |
| `target_voxel_size` | Requested target spacing, default `1.0`. |
| `original_structure_path` | Optional original structure; batch workers pass `None`. |
| `fasta_path` | Optional FASTA path; batch workers pass `None`. |
| `fix_shift` | Request origin-shift correction, default `True`. |
| `interface_contact_cutoff` | Component contact distance, default `5.0`. |
| `s3_endpoint_url` | Passed by the CLI when processing one artifact; give it an optional default when sharing an adapter with batch processing. |

Batch workers unpack `.gz` map/structure inputs into temporary local files.
Processing one artifact requires local raw-input file paths and passes them
through unchanged, so its adapter owns format and compression handling.
Multiple batch threads may call one adapter concurrently;
protect shared model state or use one preprocessing worker.

Write with `write_preprocessed_artifact(artifact_reference, artifact_mapping,
s3_endpoint_url=s3_endpoint_url)`. The writer accepts portable mappings and
arrays; successful serialization alone does not establish a complete training
artifact. The required payload includes the density grid and geometry, aligned
protein/nucleotide backbone coordinates and labels, full-atom targets, and
component/interface targets. Present sequence data must also validate. The
[data contracts](data-contracts.md) specify keys, shapes, masks, units, and
enums. `cryoflake.io` supplies MRC/structure readers and `cryoflake.targets`
supplies target-building helpers; consult the [API reference](api-reference.md).

## Process one artifact

With real local inputs:

<!-- doc-example: guide-single; mode=cli -->
```bash
uv run cryoflake preprocess data/map.mrc data/model.cif data/artifacts/example \
  --preprocess-adapter default --target-voxel-size 1.5
uv run cryoflake validate-artifact data/artifacts/example
```

Other options include `--original-structure-path`, `--fasta-path`,
`--interface-contact-cutoff`, and the paired `--fix-shift` / `--no-fix-shift` flags. Add
`-o pipeline.require_sequence_embeddings=true` to require sequence data.
The built-in adapter supports S3 output and the configured endpoint. Custom
adapters must use their endpoint argument when writing remotely.

## Process a batch

With source associations and downloadable inputs:

<!-- doc-example: guide-batch; mode=cli -->
```bash
uv run cryoflake prepare-manifest \
  --source-csv data/source.csv --manifest-csv data/manifest.csv \
  --maps-dir data/maps --structures-dir data/structures \
  --download --download-workers 4 \
  --artifact-output-dir data/artifacts \
  --preprocess-adapter default --preprocess-workers 2 \
  --summary-csv data/summary.csv --artifact-list-path data/train.txt
```

The order is manifest creation, optional downloads, optional embedding
generation, then preprocessing. Omit `--download` for existing input files.
Artifact roots are named `emd_<id>_<pdb>` or `emd_<id>` without an extension.
The artifact list contains successful roots, one per line. Local relative
entries resolve from the list's directory; S3 entries remain URIs.

Defaults are `<artifact_output_dir>/artifact_list.txt` for the list and
`<manifest_stem>_summary.csv` beside the manifest for the summary. Summary
rows include status, errors, geometry, target counts, group IDs, output/cache
locations, and split assignment. Inspect `success` and `status`: a continuing
batch may exit successfully despite failed rows. `--fail-fast` aborts on
preprocessing/upload errors and drains already running work before releasing
ownership. Missing structure files are skipped. Missing maps raise unless
`--skip-missing` is set. Missing required sidecars or missing sidecar references
also abort, regardless of `--fail-fast`; that flag controls errors inside
preprocessing, artifact validation, and upload handling.

Valid existing outputs are reused unless `--force-preprocess` is set. Repeat
`--artifact-reuse-dir data/archive` to search other local artifact
directories. Lists can then point to those existing roots; reuse does not copy
each artifact into the new output directory. Invalid artifacts or incompatible
component contact settings are regenerated. Reuse checks the training contract
and contact cutoff; it does not independently compare the original inputs,
adapter, requested voxel size, or shift policy. Use `--force-preprocess` after
changing those settings when no matching checkpoint protects the run.

For a generated or hand-edited manifest, use the direct Python API:

<!-- doc-example: guide-batch-python; mode=python -->
```python
from cryoflake.preprocess.adapters.default import run as adapter
from cryoflake.preprocess.manifest import preprocess_manifest_csv

paths = preprocess_manifest_csv(
    "data/manifest.csv",
    artifact_output_dir="data/artifacts",
    artifact_list_path="data/train.txt",
    summary_csv="data/summary.csv",
    num_workers=2,
    preprocess_adapter=adapter,
)
```

`prepare-manifest` expects the source association CSV and creates a manifest;
it does not accept a generated manifest as its source.

## Sequence embeddings

Set `--sequence-embeddings-dir data/embeddings` to attach existing sidecars.
The default filename is `<wwpdb_id>.npz`. Override it with
`--sequence-embedding-suffix .npy` or
`--sequence-embedding-name-template '{artifact_name}'`. Template fields
are `{emdb_id}`, `{emdb_accession}`, `{artifact_name}`, and `{wwpdb_id}`.

A `.npy` sidecar holds the numeric embedding matrix. An `.npz` sidecar requires
`sequence` and optionally contains `sequence_mask`, `unified_seq`, and
`unified_seq_len`. Object arrays are rejected, and alternative embedding keys
cannot replace `sequence`. Additional numeric fields are ignored. See
[sequence contracts](data-contracts.md) for dimensions and normalization.
These are input files; output embeddings use the artifact's `sequence.npz`.

`--require-sequence-embeddings` requires a sidecar reference and an existing
file for every structured batch row. An adapter's own embeddings do not replace
that batch-sidecar requirement. Without it, a referenced missing sidecar warns
and forces fresh preprocessing without attaching that file; embeddings already
written by the adapter may remain. Malformed present data still fails
validation. Attaching a supplied sidecar replaces existing sequence fields.
When a manifest references a sidecar, external reuse directories are excluded
so attaching it cannot mutate an external artifact. Valid output already at
the destination can still be reused and have its embeddings replaced.

To generate sidecars before preprocessing, add these options to the batch
command, using an installed application generator:

<!-- doc-example: guide-embeddings-options; mode=options -->
```text
--sequence-embeddings-dir data/embeddings
--prepare-sequence-embeddings
--sequence-embedding-generator my_app.embeddings:generate
--sequence-embedding-device cpu
--sequence-embedding-workers 2
--sequence-embedding-max-chain-length 1000
```

The generator receives `structures_dir` (structure directory), `sequence_embeddings_dir`
(sidecar directory), `device`, `max_chain_length`, and `num_workers`. It must
create files matching manifest names. There is no built-in sequence model.
Python callers can inject it:

<!-- doc-example: guide-embeddings-python; mode=python -->
```python
from my_app.embeddings import generate

from cryoflake.config import compose_config
from cryoflake.preprocess.prepare import run

config = compose_config(
    values={
        "pipeline": {
            "source_csv": "data/source.csv",
            "manifest_csv": "data/manifest.csv",
            "structures_dir": "data/structures",
            "sequence_embeddings_dir": "data/embeddings",
            "prepare_sequence_embeddings": True,
        }
    }
)
run(config, sequence_embedding_generator=generate)
```

## Splitting datasets

Add these options to a preprocessing command:

<!-- doc-example: guide-split-options; mode=options -->
```text
--artifact-list-path data/train.txt
--validation-artifact-list-path data/valid.txt
--validation-fraction 0.1
--split-seed 42
```

The default `artifact` strategy partitions unique artifact references for the
given input order and seed. Artifact aliases cannot cross the split. Different
artifacts from the same wwPDB structure can cross this default split; grouped
strategies keep those structure peers together. A singleton stays in training
and produces an empty validation file;
`read_artifact_list()` rejects an empty file when reading it.

For group holdout also add:

<!-- doc-example: guide-group-options; mode=options -->
```text
--split-strategy component
--split-report-path data/split.json
```

Use `interface` for interface groups. Group IDs come from component targets
and appear in summaries/reports. Automatic selection approximates the fraction
while leaving both outputs nonempty. At least two independent artifacts with
usable group IDs are needed, and the fraction may not be achievable exactly.

All records containing a selected group move to validation, together with
records sharing their artifact reference or `artifact_group_id`. Selected
groups cannot remain in training. Other unselected groups co-occurring in a
validation artifact may still appear in training. Select groups explicitly or
replay a previous report with these grouped-split options:

<!-- doc-example: guide-replay-options; mode=options -->
```text
--split-strategy component
--holdout-group-ids-path data/split.json
--split-report-path data/split-replayed.json
```

Keep the training/validation list options above. The input holdout report and
new report output must use distinct paths; alternatively omit `--split-report-path`
when replaying. Holdout input may instead be a JSON string list or one ID per
nonempty line. Report input must match the chosen strategy. Reports and summary
CSVs label validation assignments `valid`.

## Resume, retry, or reset

A summary path enables sibling files `<summary>.checkpoint.json`,
`<summary>.checkpoint.jsonl`, and `<summary>.checkpoint.lock`. Ownership
requires POSIX `fcntl` locks; two processes cannot own the same journal. The
lock file remains after completion, so its existence alone does not indicate
an active run. Checkpoint metadata uses format version `5`. Use
`--reset-preprocess-checkpoint` to start a new journal when the stored version
differs.

Rerun the same command after interruption or failed rows/uploads. Completed
rows are recovered when their recorded artifacts remain current; changed or
missing artifacts are checked again. Failed rows are retried. Failed S3
uploads retain local artifacts, allowing a compatible rerun to retry uploads
without preprocessing again. Keep scratch/cache and checkpoint files until
recovery finishes.

Recovery also checks the stored group record and agreement between artifact
references in the row and its summary. Missing, malformed, or contradictory
records are retried through preprocessing rather than omitted from split
outputs; pending uploads are staged only after these checks pass.

The run fingerprint covers the manifest and source identities, output paths,
adapter implementation/state, sequence policy/loader, geometry, and resolved
S3 endpoint/region. A conflicting checkpoint raises an error. Review the
changes, then add `--reset-preprocess-checkpoint` to start a new journal.
Reset discards progress but can still reuse valid outputs;
`--force-preprocess` also bypasses reuse. Worker counts, split selection,
list/report destinations, and the fail-fast policy can change without changing
the preprocessing contract. The summary path selects the checkpoint itself;
use the same path to recover its progress.

When adapter behavior depends on external state that callable inspection
cannot capture, Python callers can supply `preprocess_checkpoint_token` to
`preprocess_manifest_csv()` and change it when that state changes. A fully
successful batch removes journal/metadata after writing final lists/summary.
Skipped or failed rows retain recovery state.

## Workers, memory, and uploads

`--workers` supplies the default for downloads, preprocessing, and uploads;
`--download-workers`, `--preprocess-workers`, and `--upload-workers` override
those counts. These phases use threads. `--sequence-embedding-workers`
independently defaults to `1` and is passed to the generator, which owns its
scheduling. Adapters own model/device scheduling and internal pools. Start with
one worker for a shared GPU model unless its adapter safely supports concurrent
calls.

Batch memory admission estimates MRC resampling requirements and uses 75% of
the smaller detected physical-memory/cgroup limit, or 128 GiB if neither is
detectable. It does not bound application model memory. Jobs whose estimate
exceeds the limit become `skipped_oversized_map`. Unknown estimates reserve
the entire budget and run one at a time. There is no CLI/YAML memory knob.

S3 processing writes local cache artifacts and uploads asynchronously with
bounded pending work and producer backpressure. Shutdown waits for accepted
uploads and bookkeeping. Success removes managed cache artifacts; failure
retains them and records `cached_artifact_dir`. Keep enough scratch space for
staged artifacts and decompressed inputs. Do not share an artifact root among
independent concurrent writers: write locks protect threads in one process,
while checkpoint ownership protects one batch journal.

## S3 storage

Copy `.env.example` to the ignored checkout-root `.env` and fill in required
settings:

<!-- doc-example: guide-env; mode=data -->
```dotenv
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_SESSION_TOKEN=your-optional-session-token
CRYOFLAKE_S3_ENDPOINT_URL=https://your-service.example
CRYOFLAKE_S3_REGION=your-region
```

Omit unused session tokens. With no endpoint in cryoFlake or AWS environment
settings, omit/leave blank the endpoint for normal Amazon S3 configuration.
A blank endpoint does not disable other endpoint environment variables. AWS
profiles and role credentials also work through boto3. Existing process
variables override `.env`. Keep credentials out of YAML, CLI arguments, and
committed files.

For an installed wheel to load a credentials file, select it explicitly:

<!-- doc-example: guide-env-file; mode=cli -->
```bash
CRYOFLAKE_ENV_FILE=/absolute/path/credentials.env cryoflake config
```

Python callers may use
`cryoflake.storage.load_s3_environment("/absolute/path/credentials.env")`.
Installed packages do not search the working directory or its ancestors for
`.env`. See [endpoint precedence](configuration.md#s3-environment-and-endpoint-precedence).

For remote batches add these options, keeping a local directory for staging:

<!-- doc-example: guide-remote-options; mode=options -->
```text
--artifact-s3-prefix s3://my-bucket/artifacts
--artifact-output-dir data/scratch
```

Newly processed artifacts use S3 roots in the list; artifacts
reused from external directories retain their original references. Use
`--s3-endpoint-url https://service.example` to override the endpoint explicitly.
Artifact references use `s3://bucket/key`; credentials and service hosts do
not belong inside them.

## Validate and troubleshoot

<!-- doc-example: guide-validate; mode=cli -->
```bash
uv run cryoflake validate-artifact data/artifacts/example
uv run cryoflake validate-artifact s3://my-bucket/artifacts/example \
  --require-sequence-embeddings
```

Validation checks storage integrity and the complete training contract. Its
JSON output contains the reference, grid shape, protein/nucleotide/component
counts, and optional embedding dimensions. Present embeddings are checked
even when optional. Validate a list using
`validate_artifact_list("data/train.txt")` from
`cryoflake.storage.artifacts`.

| Symptom | Action |
| --- | --- |
| Cannot load adapter/generator | Install its module and dependencies in this environment; check `MODULE:CALLABLE`. |
| Missing CSV headers | Use an association CSV for `prepare-manifest` and the Python API for a generated manifest; compare the data contracts. |
| Missing input after moving a manifest | Resolve relative paths from the CSV directory and correct the CSV or restore its layout. |
| Checkpoint mismatch | Review sources, adapter state, geometry, and destination before resetting explicitly. |
| Another run owns the checkpoint | Wait for or stop the owner; use separate output/summary paths for independent work. Do not delete an active lock file. |
| Artifact generation/hash/member mismatch | Restore a complete artifact or regenerate it; do not mix metadata and stores from different writes. |
| Training contract validation failure | Compare the adapter payload with all required fields; serializable mappings may still lack targets. |
| Invalid grouped split | Inspect IDs/holdouts, add independent artifacts, or select groups leaving both partitions nonempty. |
| `skipped_oversized_map` | Check map spacing and requested target; use a scientifically suitable coarser target or more available memory. |
| Upload failure | Check endpoint, credentials, permissions, and the error; retain the cache and rerun for recovery. |

`load_preprocessed_artifact()` reads all fields by default. Use
`root_keys=["density_grid"]` for selected top-level fields or
`include_sequence_embeddings=False` to exclude sequence fields. Selective
reads validate selected stores/arrays; full artifact validation checks a
complete training artifact. NPZ arrays load into memory; this API does not
provide lazy or memory-mapped arrays.
