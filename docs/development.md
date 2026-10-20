# Development

For installation and normal operation, start with the [user guide](user-guide.md).
This page describes working on the package itself.

Follow the project-wide [agent and contributor standards](../AGENTS.md) for
precise terminology and documentation changes that must update CI regression
coverage.

## Set up the checkout

Use Python 3.12 or newer and uv. The lockfile records runtime and development
dependencies; the `dev` group contains pytest, pytest-cov, and Ruff.

<!-- doc-example: development-checks; mode=setup -->
```bash
uv sync --locked --dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests/documentation
uv run pytest --cov=cryoflake --cov-report=term-missing
uv build
```

Apply formatting with `uv run ruff format .` and safe lint fixes with
`uv run ruff check --fix .`. After dependency changes, run `uv lock`, then
verify the environment with `uv sync --locked --dev`. Declare dependencies
imported by runtime code directly.

## Tests

Tests generate small structures, density maps, manifests, and archives in
temporary directories. S3 clients and model adapters are replaced with test
doubles, so the suite does not need cloud credentials, network downloads, or
model weights. Structure tests exercise Biotite parsing directly.

Storage tests cover the [NPZ+JSON format](artifact-format.md), integrity,
selective loading, and sequence replacement. Training validators are checked
against the [data contracts](data-contracts.md). Workflow tests cover
downloads, adapter invocation, embedding preparation, splitting, checkpoint
recovery, and retained upload artifacts.

Concurrency tests coordinate workers with events and barriers and use bounded
waits. Resource checks cover startup failure, overlapping shutdown, callbacks,
cache ownership, file locks, interrupted downloads, and memory admission.
Allocation regressions use operation counts, allocation bounds, and exact
output comparisons. CI does not depend on wall-clock throughput thresholds.

## Documentation regression tests

`uv run pytest tests/documentation` reads examples directly from `README.md`
and the public guides. CI runs this suite explicitly on Python 3.12, 3.13,
and 3.14 before the remaining tests. A normal `uv run pytest` also includes it;
`-m documentation` selects these cases by marker.

Every fenced block has an HTML annotation immediately before it, such as
`<!-- doc-example: readme-storage; mode=python -->`. Keep IDs stable when moving
examples. Supported modes are `python`, `cli`, `options`, `data`, `setup`, and
`reference`. New or unclassified fences fail the inventory check. Add a scenario
in `tests/documentation/` and register its ID/mode in `test_inventory.py` when
adding an example; do not copy the documented code into the test.
Use top-level fences: quoted or list-embedded fences are rejected explicitly
so they cannot bypass example coverage.

Tests execute Python blocks verbatim and pass CLI command arguments to Typer.
Temporary workspaces supply real MRC/mmCIF inputs and NPZ+JSON artifacts, with
injected application adapters, generators, and in-memory HTTP/S3 transports.
They block live network calls and isolate credentials. Option fragments are
combined with their documented base command, and examples assert artifacts,
arrays, reports, geometry boundaries, and settings rather than only exit codes.
Reference diagrams and signatures are classified separately; public signatures
and documentation links are checked too.

Environment setup, lint, build, and recursive pytest commands are checked
against CI rather than executed inside the running test environment. The wheel
installation example builds distributions offline and installs the wheel into
a separate temporary environment, reusing resolved dependencies. Run these
tests through uv so its cached build backend is available; no model weights,
cloud credentials, or live downloads are required.

Run focused tests while editing, then the full suite before submitting changes
to runtime behavior or data contracts. GitHub Actions checks Python 3.12,
3.13, and 3.14, Ruff, pytest, and distribution builds.

## Package layout and contracts

Place shared helpers in `src/cryoflake/common.py`, molecular vocabularies and
enums in `src/cryoflake/constants.py`, and pipeline-specific logic in its
subpackage. Keep model integrations behind injected callables. The
[API reference](api-reference.md) identifies the owning modules.

Public guides live in `docs/`. Project YAML overrides live in `configs/`,
while installed defaults and format definitions live in
`src/cryoflake/configs/`. Keep documentation tables and examples consistent
with those definitions and their validators. Changes to serialized contracts
must account for format/schema versions and checkpoint identities.

Wheels include packaged YAML resources. Source distributions also include
project configuration, public guides, tests, and `.env.example`. Credentials
and local working material are excluded from distributions. Inspect both
archives when changing build include/exclude rules.
