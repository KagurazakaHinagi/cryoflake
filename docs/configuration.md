# Configuration reference

Hydra composes YAML groups; OmegaConf applies a strict typed schema. CLI and
Python callers share that configuration. Composition never launches a Hydra
job or changes directories. Relative configuration paths resolve from the
process working directory; paths inside manifests and artifact lists resolve
from the directory containing that file. Paths written inside YAML are not
automatically expanded from `~`; use absolute paths or OmegaConf environment
interpolation such as `${oc.env:HOME}/data/maps`.

Start with the [user guide](user-guide.md) for workflows. See
[data contracts](data-contracts.md) for payload and tabular schemas, and the
[artifact specification](artifact-format.md) for NPZ+JSON serialization.

## Inspection and precedence

<!-- doc-example: config-inspect; mode=cli -->
```bash
uv run cryoflake config
uv run cryoflake config --pipeline preprocess
uv run cryoflake config --pipeline validate
uv run cryoflake config -o pipeline.workers=4 \
  -o 'pipeline.preprocess_workers=${pipeline.workers}'
```

Each `-o` / `--override` accepts one Hydra expression. Quote shell-sensitive
expressions. Lists use Hydra syntax, for example
`-o 'pipeline.artifact_reuse_dirs=[data/archive,data/other]'`.

Without `--pipeline`, `config` honors the group selected by the root YAML or
`-o pipeline=...`. An explicit `config --pipeline GROUP` takes precedence over
a conflicting group override. Execution commands select their packaged group
by default. When `--config-dir` or a nondefault `--config-name` selects a custom
root, they preserve its selected group instead; its `pipeline.name` must match
the command. Use `-o pipeline=...` to select another group in that root.

Precedence, lowest to highest:

1. Typed defaults.
2. Composed YAML groups and root YAML.
3. Hydra overrides supplied with `-o`.
4. Explicit CLI flags or Python `compose_config(values=...)` values.

Omitted boolean flags preserve YAML values. Paired flags such as `--download`
and `--no-download` override them. `--fix-shift` and `--no-fix-shift` explicitly
enable or disable `fix_shift`; omitting both preserves the composed value. Each execution
command requires its corresponding pipeline `name`, including custom groups.

`config` validates values without requiring entry-point inputs. Execution
also requires inputs and adapters. Unknown fields, invalid types, unsupported
wire-format changes, nonpositive worker counts, nonfinite numeric settings,
and configuration-level path conflicts fail before pipeline outputs are
created. Checks requiring generated manifest rows run later, before
preprocessing starts; a manifest may already have been written. Configuration
errors exit with status 2; runtime failures exit with status 1. Continuing row
failures can coexist with a successful command exit: inspect the summary.

Single-artifact preprocessing rejects an output root resolving to the current
directory or a filesystem root before invoking the adapter. Validation may
read those locations when they contain an artifact.

## YAML groups and project settings

These resources ship in installed wheels:

<!-- doc-example: config-layout; mode=reference -->
```text
src/cryoflake/configs/
  config.yaml
  pipeline/
    base.yaml
    manifest.yaml
    preprocess.yaml
    validate.yaml
  contracts/
    artifact/npz_json.yaml
    targets/component.yaml
    manifest/emdb.yaml
```

The root composes groups into `pipeline`, `contracts`, and `storage`.
Pipeline groups inherit `pipeline/base.yaml`. The checkout's
[`configs/config.yaml`](../configs/config.yaml) imports packaged groups with
`hydra.searchpath: [pkg://cryoflake.configs]`. Its
[`project` group](../configs/pipeline/project.yaml) is a working configuration
example:

<!-- doc-example: config-project; mode=cli -->
```bash
uv run cryoflake config --config-dir configs --pipeline project
uv run cryoflake prepare-manifest --config-dir configs -o pipeline=project
```

Execution requires the source CSV specified by the group. Add another group
as `configs/pipeline/lab.yaml`, for example:

<!-- doc-example: config-lab; mode=data -->
```yaml
defaults:
  - base
  - _self_

name: manifest
source_csv: data/source.csv
manifest_csv: data/manifest.csv
maps_dir: data/maps
structures_dir: data/structures
workers: 4
```

Select it with `--config-dir configs -o pipeline=lab`. `--config-name NAME`
selects `NAME.yaml` instead of `config.yaml`. A custom directory must contain
that root and its referenced groups, or import the package search path as the
checkout does. Installed packages also accept absolute configuration paths.

## Pipeline reference

All settings below live under `pipeline`. CLI names normally replace
underscores with hyphens. Exceptions: `source_csv` uses `--source-csv`, and
`artifact_reuse_dirs` uses repeatable `--artifact-reuse-dir`. Any setting
can be supplied through `-o pipeline.KEY=VALUE`.

### Manifest and downloads

| Setting | Default | Meaning |
| --- | --- | --- |
| `name` | `manifest` | Entry-point kind: `manifest`, `preprocess`, or `validate`. |
| `source_csv` | `null` | Required local source association CSV. |
| `manifest_csv` | `null` | Required local generated manifest CSV. |
| `maps_dir` | `null` | Local map directory; required for downloads. |
| `structures_dir` | `null` | Local structure directory; required for downloads or embedding generation. |
| `include_without_structure` | `false` | Keep unassociated source rows; preprocessing skips them. |
| `download` | `false` | Download manifest inputs. |
| `force_download` | `false` | Download again even when destination files pass the quick reuse check. |
| `workers` | `1` | Positive shared worker count. |
| `download_workers` | `null` | Download threads; falls back to `workers`. |

The default source contract requires unique, nonempty headers including
`EMD-` and `Associated PDB`. A project may add required source columns through
`contracts.manifest.required_source_columns` while retaining those two.
Manifest creation uses supplied associations without online lookup.
Existing downloads must be nonempty; gzip destinations must yield a
decompressed byte to qualify for reuse. HTTP 404 and selected DNS resolution
failures warn and continue, while other download exceptions abort. `fail_fast`
does not change download handling.

### Sequence embeddings

| Setting | Default | Meaning |
| --- | --- | --- |
| `sequence_embeddings_dir` | `null` | Local sidecar directory recorded in the manifest. |
| `sequence_embedding_suffix` | `.npz` | Input sidecar suffix: `.npy` or `.npz`. |
| `sequence_embedding_name_template` | `{wwpdb_id}` | Filename stem template. |
| `require_sequence_embeddings` | `false` | Require batch sidecars and valid artifact embeddings during preprocessing/validation. |
| `prepare_sequence_embeddings` | `false` | Run an explicit generator after downloads. |
| `sequence_embedding_generator` | `null` | Generator as `MODULE:CALLABLE`. |
| `sequence_embedding_device` | `null` | Device passed unchanged to the generator. |
| `sequence_embedding_workers` | `1` | Positive worker count passed to the generator. |
| `sequence_embedding_max_chain_length` | `1000` | Positive maximum chain length passed to the generator. |

Templates accept `{emdb_id}`, `{emdb_accession}`, `{artifact_name}`, and
`{wwpdb_id}` only, and must produce a nonempty name. Generation requires the
structure/sidecar directories and a generator. The application owns model
dependencies and interpretation of device/length settings. See
[sequence workflows](user-guide.md#sequence-embeddings) and
[sidecar contracts](data-contracts.md).
`sequence_embedding_workers` is independent of `workers`; it does not fall
back to the shared worker count. When processing or validating one artifact,
required embeddings must be in the artifact; no separate sidecar is required.

### Batch preprocessing and recovery

| Setting | Default | Meaning |
| --- | --- | --- |
| `preprocess_adapter` | `null` | Explicit raw adapter as `MODULE:CALLABLE`. |
| `artifact_output_dir` | `null` | Enable preprocessing using this local output/scratch directory. |
| `artifact_s3_prefix` | `null` | Publish under an S3 prefix; requires local scratch. |
| `artifact_reuse_dirs` | `[]` | Local directories searched for valid existing artifacts. |
| `preprocess_workers` | `null` | Preprocessing threads; falls back to `workers`. |
| `upload_workers` | `null` | Asynchronous upload threads; falls back to `workers`. |
| `target_voxel_size` | `1.0` | Positive finite target spacing passed to the adapter. |
| `interface_contact_cutoff` | `5.0` | Positive finite contact distance, also checked on reuse. |
| `fix_shift` | `true` | Request map origin-shift correction. |
| `force_preprocess` | `false` | Bypass artifact reuse and reset checkpoint progress. |
| `skip_missing` | `false` | Skip missing maps; absent structures are always skipped. |
| `fail_fast` | `false` | Abort on adapter, artifact-validation, or upload failure instead of continuing affected rows. Missing maps/required sidecars already abort unless their applicable skip policy allows them. |
| `summary_csv` | `null` | Summary/checkpoint base; runner derives `<manifest_csv_stem>_summary.csv` if unset. |
| `reset_preprocess_checkpoint` | `false` | Discard progress when the run contract changes. |

The memory admission limit is automatic; there is no CLI/YAML memory field.
See [operational behavior](user-guide.md#workers-memory-and-uploads).
Checkpoints require POSIX `fcntl` locks. Direct Python
`preprocess_manifest_csv()` calls enable recovery only when `summary_csv` is
provided; the manifest runner always selects a summary during preprocessing.
Reuse checks artifact validity and the contact cutoff, not whether every source
or preprocessing setting matches. Outside matching checkpoint recovery, use
`force_preprocess` to regenerate outputs after changing inputs, adapters, voxel
size, or shift policy. See [reuse behavior](user-guide.md#process-a-batch).

### Artifact lists and splits

| Setting | Default | Meaning |
| --- | --- | --- |
| `artifact_list_path` | `null` | Training list; runner defaults to `<artifact_output_dir>/artifact_list.txt`. |
| `validation_artifact_list_path` | `null` | Optional validation list; enables splitting. |
| `validation_fraction` | `0.1` | Finite fraction strictly between 0 and 1. |
| `split_seed` | `0` | Integer selection seed. |
| `split_strategy` | `artifact` | `artifact`, `component`, or `interface`. |
| `holdout_group_ids_path` | `null` | Explicit selection for grouped splitting. |
| `split_report_path` | `null` | Grouped report JSON output. |

Grouped strategies require a validation list. Holdout input and reports require
`component` or `interface`. Input accepts one group per nonempty line, a JSON
string list, or a versioned report with the matching strategy. See
[splitting datasets](user-guide.md#splitting-datasets) for guarantees and
[data contracts](data-contracts.md) for report fields.

### Single-artifact commands

These settings all default to `null`:

| Setting | Used by | Meaning |
| --- | --- | --- |
| `map_path` | `preprocess` | Required map path; first positional argument. |
| `structure_path` | `preprocess` | Required structure path; second positional argument. |
| `artifact_reference` | `preprocess`, `validate-artifact` | Required artifact root or S3 prefix; third positional argument for preprocessing, first for validation. |
| `original_structure_path` | `preprocess` | Optional original structure passed to the adapter. |
| `fasta_path` | `preprocess` | Optional FASTA path passed to the adapter. |

Single preprocessing honors the shared voxel-size, shift, contact, and
embedding-requirement settings. Validation honors the embedding requirement.
The adapter for processing one artifact owns input handling; batch workers
additionally decompress `.gz` inputs into temporary local files.
Raw map, structure, original-structure, and FASTA inputs must use local file
paths; remote URIs and existing directories are rejected before adapter I/O.
Reuse parents in `artifact_reuse_dirs` must be local directory paths, not files.

## Contracts and path checks

Artifact and target groups describe the supported wire formats. Changes to
their identities, versions, filenames, marker, metadata limit, or target
dimensions are rejected. Supporting another wire format requires code changes.
Use the [artifact](artifact-format.md) and [data](data-contracts.md) references
to interpret these descriptors.

Source/output CSVs, summaries, lists, reports, holdout inputs, and checkpoint
sidecars must use distinct files, including symlink and existing hardlink
aliases. Control outputs cannot overwrite manifest inputs or be placed inside
a generated artifact. Manifest/download/embedding paths, batch scratch,
reports, summaries, and lists must be local. Artifact references and
single-artifact output support local directories or `s3://` only. S3 URIs
cannot contain credentials, ports, queries, or fragments; service hosts belong
in `storage.s3_endpoint_url` as HTTP(S) URLs.

## S3 environment and endpoint precedence

`storage.s3_endpoint_url` defaults to `${oc.env:CRYOFLAKE_S3_ENDPOINT_URL,null}`.
The explicit `--s3-endpoint-url` flag overrides composition. During I/O a
nonempty explicit endpoint wins, followed by `CRYOFLAKE_S3_ENDPOINT_URL`,
`AWS_ENDPOINT_URL_S3`, and `AWS_ENDPOINT_URL`. Empty values are skipped; a blank
CLI/YAML endpoint does not disable endpoints from the environment. If none is
set, the SDK supplies its normal service configuration. Region precedence is
`CRYOFLAKE_S3_REGION`, `AWS_REGION`, then `AWS_DEFAULT_REGION`.

Credentials use boto3's provider chain, including environment, profiles, and
roles. The ignored checkout-root `.env` is loaded before composition/S3 access
without replacing existing variables. Installed wheels do not search the
working directory or its ancestors: set
`CRYOFLAKE_ENV_FILE=/absolute/path/credentials.env` or call
`load_s3_environment(path)`. A selected file must exist. See
[S3 setup](user-guide.md#s3-storage).

## Python composition

<!-- doc-example: config-python; mode=python -->
```python
from cryoflake.config import compose_config, validate_config
from cryoflake.preprocess.prepare import run

config = compose_config(
    ["pipeline.workers=4"],
    values={
        "pipeline": {
            "source_csv": "data/source.csv",
            "manifest_csv": "data/manifest.csv",
        }
    },
)
validate_config(config, require_inputs=True)
entries = run(config)
```

`pipeline="preprocess"` or `pipeline="validate"` selects another composition;
`run()` above specifically executes the manifest pipeline. Python callers
inject `preprocess_adapter=adapter` and/or `sequence_embedding_generator=generator`
into `run()` when enabling those phases. Injection satisfies adapter validation
without string settings. `run()` does not import callables from string values
automatically. See the [API reference](api-reference.md) for direct operations.
