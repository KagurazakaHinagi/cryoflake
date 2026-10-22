# cryoFlake project standards

These instructions apply throughout this repository. Keep this file current
when project conventions change.

cryoFlake is a Python package for cryo-EM data preparation and validation.
Use `cryoflake` for Python imports and CLI identifiers, uv for dependencies,
Ruff for lint/formatting, Biotite for structures, Hydra for configuration, and
Typer for the CLI. Artifact storage is NPZ+JSON, locally or on S3.

## Non-ambiguous terminology

Use a name that identifies the value's role, type, and scope. Follow the
existing canonical vocabulary across APIs, CLI flags, YAML, serialized fields,
errors, tests, and documentation.

| Term or name | Meaning |
| --- | --- |
| Artifact | One stored NPZ+JSON unit, including its metadata and declared array stores. |
| `artifact_reference`, `artifact_references` | Local or S3 artifact roots, or their `metadata.json` references. |
| `artifact_mapping` | In-memory mapping containing an artifact's payload. |
| `*_path` | A local file, such as a raw map, structure, CSV, or sidecar. |
| `*_dir` | A local directory. |
| `artifact_name` | Extensionless artifact root name; not a file suffix or filesystem path. |
| `artifact_list_path` | Local text file listing artifact references. |
| `source_csv`, `manifest_csv` | Source association CSV and generated manifest CSV, respectively. |
| `artifact_output_dir` | Local artifact parent, also used for staging S3 outputs. |
| `artifact_s3_prefix` | Remote parent prefix beneath which artifacts are published. |
| `artifact_reuse_dirs`, `cached_artifact_dir` | Local reuse search parents and one retained cached artifact directory. |
| `GroupedSplit*` | Splitting that supports both component and interface groups. |

- Do not call stored artifacts "examples", standalone "files", or local
  "paths" when the operation also accepts S3 references. "Example" is suitable
  for illustrative documentation, fixtures, and placeholder names.
- Distinguish raw inputs, embedding sidecars, artifact mappings, artifact
  references, and files that list references. Do not interchange their names.
- Use descriptive words instead of opaque abbreviations such as `out`, `fn`,
  or `stu`. Preserve established scientific atom labels, units, and model tensor
  names when their meanings are defined by the data contract.
- Name counts by what they count: split records can contain repeated artifact
  references, so a record count is not necessarily a unique-artifact count.
- Keep names consistent across boundaries. CLI flags normally use the YAML
  field name with underscores replaced by hyphens; document deliberate
  exceptions. Use positive boolean names and explicit enable/disable flags.
- Distinguish a Hydra pipeline group from its entry-point kind, `pipeline.name`.
  A custom group may have `name: manifest`; preserve selected groups and
  explicit flag precedence, and validate inputs before output or adapter I/O.
- Keep one canonical API per operation. When renaming, update exports, callers,
  adapter keywords, fixtures, monkeypatch targets, contracts, and guides in the
  same change. Remove old names; do not add compatibility aliases, wrappers,
  deprecated flags, or silent fallback to legacy serialized keys.
  Legacy names remain valid in tests that assert their absence or rejection;
  avoid blind substring replacements.
- Review schema versions and checkpoint identities when serialized names or
  their meanings change. Add regression tests proving removed names and
  unsupported versions are rejected.

## Documentation changes require CI test coverage

Documentation is part of the supported interface. Any change to a documented
command, Python example, API signature, option, configuration default,
serialized field, data contract, output, or workflow must update or add its
regression coverage in the same change. A documentation-only change does not
exempt a behavioral claim from testing.

- Maintain the source-derived suite in `tests/documentation/`. Execute snippets
  from Markdown instead of copying them into tests. Keep existing
  `doc-example` IDs stable; classify new fenced blocks with an ID and mode,
  register them in `test_inventory.py`, and provide a meaningful scenario.
- Assert actual behavior: parsed settings, artifacts, arrays, masks, shapes,
  dtypes, geometry, serialized keys, versions, and reports as appropriate.
  Do not rely only on exit codes or weaken expectations to hide a mismatch.
- Check reference tables and signatures against the implementation. Preserve
  link/anchor checks and isolated wheel-installation coverage.
- Keep tests offline and independent of credentials, model weights, and remote
  services. Use temporary workspaces and injected adapters/transports.
- CI must continue running `uv run pytest tests/documentation` for every
  supported Python version, including changes confined to documentation.
  Update `.github/workflows/ci.yml` when a new documented workflow needs a
  check outside this suite; do not skip documentation checks with path filters.
- For spelling, formatting, links, or policy prose that changes no executable
  behavior or contract, run the existing documentation checks and state why
  existing coverage is sufficient. Do not add tests that merely match prose.

## Standard offline molecular target

Use the compact EMD-11736 / 7AE1 fixture through
[`tests/standard_target.py`](tests/standard_target.py) for general molecular
artifact, validation, and workflow tests. Its
[README](tests/fixtures/emd_11736_7ae1/README.md) and
[provenance](tests/fixtures/emd_11736_7ae1/provenance.json) describe the real
structure subset, extracted targets, and downsampled density. Keep genuine
coordinates, atom names, masks, and cached geometry as the reference values.
Load independent copies before making changes.

Synthetic inputs and mutations remain appropriate when the behavior under
test is an invalid value, boundary, format, or rare molecular case. State that
purpose locally; do not introduce an unrelated invented molecule or empty
geometry as a general reference payload. Synthetic embeddings must be labeled
as test data. Keep routine tests offline and independent of model weights.

## Verification and maintenance

Run `uv sync --locked --dev` to prepare the environment. After documentation
changes, run `uv run pytest tests/documentation`, `uv run ruff check .`, and
`uv run ruff format --check .`. Changes to runtime behavior or contracts also
require focused regression tests and the full `uv run pytest` suite. Changes
to packaging require `uv build` and the distribution checks.

Use [docs/development.md](docs/development.md) for the test workflow and
[docs/data-contracts.md](docs/data-contracts.md) for semantic contracts. Keep
public guides under `docs/`; migration notes, performance investigations,
benchmarks, and results belong in ignored `internal/`. Keep credentials in the
ignored root `.env`; never put secret values in code, docs, fixtures, or logs.
