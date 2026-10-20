# Data contracts

This reference describes the data accepted and produced by cryoFlake. Use it
when preparing manifests, implementing a preprocessing adapter, or consuming a
dataset. The [artifact format](artifact-format.md) describes how these values
are encoded in NPZ+JSON; this page describes their meaning.

Two validation levels serve different purposes:

- `write_preprocessed_artifact()` and `load_preprocessed_artifact()` handle
  portable mappings and storage integrity. A mapping can be useful for storage
  without being a complete training artifact. The writer validates and
  normalizes a top-level `density_grid` when present, but does not require the
  training fields below.
- Manifest preprocessing, `validate_preprocessed_artifact()`, and
  `validate_artifact_list()` require the complete training contract,
  including valid full-atom and component targets. They also validate sequence
  fields whenever present, even when embeddings are optional.

`validate_preprocessed_artifact_mapping(artifact_mapping)` only checks that its
input implements `Mapping`. It does not establish either contract above; see
the [API reference](api-reference.md#targets-and-metadata).

Runtime descriptors live in the packaged Hydra contracts:
[`manifest/emdb.yaml`](../src/cryoflake/configs/contracts/manifest/emdb.yaml),
[`targets/component.yaml`](../src/cryoflake/configs/contracts/targets/component.yaml),
and [`artifact/npz_json.yaml`](../src/cryoflake/configs/contracts/artifact/npz_json.yaml).
See [configuration](configuration.md) for how they are selected and validated.

## Source association CSV

`prepare-manifest` reads a UTF-8 CSV; a UTF-8 byte-order mark is accepted.
Column names and cell values are stripped of surrounding whitespace.

| Column | Required | Interpretation |
| --- | --- | --- |
| `EMD-` | Yes | Four or five digits, optionally prefixed with `EMD-`; normalized to the digits. |
| `Associated PDB` | Yes | The first word-boundary-delimited, four-character alphanumeric identifier in the cell is used and lowercased. Placeholder tokens `none`, `null`, `n/a`, and `na` are ignored; an empty cell or a cell without an identifier has no associated structure. |
| First column whose name starts with `resolution`, case-insensitively | No | Copied as text to `source_resolution`; no numeric or unit conversion is applied. |
| Other columns | No by default | Ignored by the built-in reader. A project contract may require additional headers. |

For example:

<!-- doc-example: contract-source-csv; mode=data -->
```csv
EMD-,Associated PDB,Resolution (Å)
EMD-1234,1ABC,3.2
12345,2XYZ; 3XYZ,2.8
```

Blank rows are ignored. Repeated normalized EMDB IDs retain the first row.
Rows without an associated structure are excluded by default;
`include_without_structure` retains them in the manifest. It does not make
them suitable for structure-supervised preprocessing.

The source reader uses the cell association; it does not automatically query
EMDB to fill a missing association. `get_fitted_structure_id_from_emdb_entry()`
is a separate Python operation. See the
[manifest implementation](../src/cryoflake/manifests/manifest.py) for parsing and
URL construction.

## Preprocessing manifest CSV

`write_manifest_csv()` produces these columns. Handwritten manifests must have
the four required headers exactly as shown; the generated manifest reader does
not apply source-CSV header normalization.

| Column | Required by manifest reader | Meaning |
| --- | --- | --- |
| `emdb_accession` | No | Display accession, such as `EMD-1234`. |
| `emdb_id` | Yes | EMDB identifier used for artifact naming; accepts the same normalization as the source CSV. |
| `wwpdb_id` | Yes | Optional structure identifier as a cell value; when nonempty it must be four alphanumeric characters and is normalized to lowercase for naming. |
| `source_resolution` | No | Source resolution text, later copied to summary `source_resolution`. |
| `map_url` | No | Source download URL recorded in the manifest and summary. |
| `structure_url` | No | Source structure URL, empty when no association exists. |
| `map_path` | Yes | Local density map path. |
| `structure_path` | Yes | Local structure path; a blank value produces `skipped_no_structure`. |
| `sequence_embedding_path` | No | Local `.npy` or `.npz` embedding input; generated only when an embedding directory is configured. |

Relative input paths are resolved from the manifest's directory. Generated
relative paths are written relative to that same directory. Paths remain local
even when the final artifacts are published to S3. Manifest preprocessing
materializes `.gz` map and structure inputs into a temporary directory before
calling the adapter. The generated-manifest reader checks required headers,
then returns cells as text without stripping or validating their contents;
naming and preprocessing validate the fields they consume. Extra columns are
retained by the reader but ignored by the built-in preprocessing workflow.

Generated map filenames are `emd_<id>.map.gz`; an existing
`emd_<id>.map` or `emd_<id>.mrc` is preferred. Having both uncompressed variants
is an error. Structure filenames are `<wwpdb_id>.cif.gz`, with an existing
`<wwpdb_id>.cif` preferred. Artifact roots are extensionless:
`emd_<id>_<wwpdb_id>` when a structure ID exists, otherwise `emd_<id>`.
Use `get_preprocessed_artifact_name()` to derive them.

Embedding names use the configured template and suffix. The template accepts
`{emdb_id}`, `{emdb_accession}`, `{wwpdb_id}`, and `{artifact_name}`; the
default is `{wwpdb_id}` with `.npz`.

## Complete training payload

The semantic preprocessing contract is version **3**. Its density and backbone
roots are `density_grid` and `backbone_coordinates`. Alternative names cannot
substitute for these required roots. This semantic version is
independent of NPZ+JSON format version `1` and component schema version `1`.

The following top-level keys are required for a successful preprocessing row
and semantic artifact validation. Extra portable fields are allowed.

The normalized dtypes below describe the return values of the individual
validators. Apart from `density_grid`, storage preserves array dtypes supplied to
the writer. Semantic artifact validation checks normalized views without
rewriting the saved payload. Adapters that need canonical stored target dtypes
should save the mappings returned by `validate_full_atom_targets()` and
`validate_component_extension_targets()`, and use normalized sequence fields.

| Key | Required value |
| --- | --- |
| `density_grid` | Nonempty, finite, numeric density array with shape `(Z, Y, X)`. Validated grids are `float32`. |
| `global_origin` | Finite numeric `(3,)` array in Cartesian `(x, y, z)` order, in ångströms. Normalized to `float32`. |
| `voxel_size` | Positive finite numeric scalar, in ångströms per voxel; spacing is isotropic. |
| `backbone_coordinates` | Mapping containing `CA`, `C`, `N`, `O`, and `P` coordinate arrays. |
| `residue_info` | Mapping containing `aa_type` and `nuc_type` label sequences. |
| `full_atom_targets` | Full-atom supervision mapping described below. |
| `extension_targets` | Versioned component-target mapping described below; required even when there are no component nodes. |
| `sequence`, `sequence_mask` | Optional embedding fields described below. `sequence_mask` cannot stand alone. |

Use world coordinates for atom and anchor positions. The relation between a
Cartesian position and the density array is:

<!-- doc-example: contract-geometry; mode=python -->
```python
voxel_xyz = (position_xyz - global_origin) / voxel_size
# density_grid is indexed as density_grid[z, y, x].
inside = ((voxel_xyz >= 0) & (voxel_xyz < density_grid.shape[::-1])).all()
```

The upper boundary is exclusive. At least one finite `CA` or `P` anchor must
fall inside the grid. Validation does not require all anchors to be inside.
Density values have no required normalization or amplitude range. The semantic
contract requires a nonempty 3D grid, not a cubic grid; grid preparation helpers
may impose their own cubic/even-size requirements. Grid, backbone, origin,
voxel-size, and embedding validators accept real integer or floating dtypes;
boolean, complex, object, and string inputs are rejected for these fields.

The grid validator also accepts a readable safe NumPy grid payload, including
compressed `arr_0` input. The artifact writer converts `density_grid` into an actual
numeric array. For new adapters, supply the array directly.

### Backbone and residue labels

Let `P` be the protein-residue count and `Q` the nucleotide-residue count.

| Field | Shape | Rules |
| --- | --- | --- |
| `backbone_coordinates["CA"]`, `backbone_coordinates["C"]`, `backbone_coordinates["N"]`, `backbone_coordinates["O"]` | `(P, 3)` each | Matching row counts; numeric coordinates normalized to `float32`. |
| `backbone_coordinates["P"]` | `(Q, 3)` | Nucleotide phosphorus coordinates, normalized to `float32`. |
| `residue_info["aa_type"]` | Length `P` | Protein residue labels from the vocabulary below. |
| `residue_info["nuc_type"]` | Length `Q` | Nucleotide labels from the vocabulary below. |

Backbone arrays may contain NaN for missing atoms, but cannot contain infinity.
An empty coordinate sequence is normalized to `(0, 3)`; a single `(3,)`
coordinate is accepted and normalized to `(1, 3)`. Empty protein or nucleotide
groups are allowed, provided the artifact still has a finite in-grid CA/P target.

Protein labels are `ALA`, `ARG`, `ASN`, `ASP`, `CYS`, `GLN`, `GLU`, `GLY`,
`HIS`, `ILE`, `LEU`, `LYS`, `MET`, `PHE`, `PRO`, `SER`, `THR`, `TRP`, `TYR`,
`VAL`, and `UNK`. Nucleotide labels are `DA`, `DC`, `DG`, `DT`, `A`, `C`, `G`,
`U`, `DN`, and `N`. Label validation uppercases values but does not strip
whitespace.

`load_full_backbone_from_structure()` additionally returns `chain` and
`secondary_structure` in `residue_info`. These follow the reader's structure
traversal across both residue classes; do not assume that they have already
been reordered to protein-then-nucleotide full-atom rows. They are useful
annotations, but are not required or checked by semantic artifact validation.
The reader uses the first model unless `all_models=True`, writes NaN for
missing backbone atoms, and selects per-atom alternate locations by identifier.

Authoritative rules:
[`metadata.py`](../src/cryoflake/metadata.py),
[`constants.py`](../src/cryoflake/constants.py),
[`io/structure.py`](../src/cryoflake/io/structure.py), and
[`preprocess/reuse.py`](../src/cryoflake/preprocess/reuse.py).

### Full-atom supervision

Let `R = P + Q`. Every required array has the same leading residue dimension.
Rows must be ordered as all protein residues followed by all nucleotides,
matching the order of the corresponding `backbone_coordinates` arrays.

| Field | Shape | Normalized dtype |
| --- | --- | --- |
| `aatype` | `(R,)` | `int64` |
| `prot_mask` | `(R,)` | `bool` |
| `chain_index` | `(R,)` | `int64` |
| `residue_index` | `(R,)` | `int64` |
| `atom_positions` | `(R, 65, 3)` | `float32` |
| `atom_mask` | `(R, 65)` | `bool` |
| `b_factors` | `(R, 65)` | `float32` |
| `rigidgroups_gt_frames` | `(R, 9, 3, 4)` | `float32` |
| `rigidgroups_alt_gt_frames` | `(R, 9, 3, 4)` | `float32` |
| `rigidgroups_gt_exists` | `(R, 9)` | `bool` |
| `rigidgroups_group_exists` | `(R, 9)` | `bool` |
| `rigidgroups_group_is_ambiguous` | `(R, 9)` | `bool` |
| `torsion_angles_sin_cos` | `(R, 8, 2)` | `float32` |
| `alt_torsion_angles_sin_cos` | `(R, 8, 2)` | `float32` |
| `torsion_angles_mask` | `(R, 8)` | `bool` |

All floating target values must be finite, including values at masked
positions, and must remain finite after `float32` conversion. Masks accept
boolean or numeric 0/1 inputs and reject other values before conversion.
`aatype`, `chain_index`, and `residue_index` require integer arrays.
`chain_index` is nonnegative; residue indices may be negative. Topology values
must fit `int64`.

`aatype` must be in `[0, 31)`. Indices `0` through `19` and `28` are protein
types; `28` is the unknown-protein index. Other accepted indices are treated
as nucleotide types. `prot_mask` must exactly match that rule and contain `P`
true entries followed by `Q` false entries.

The package fixes these dimensions and the protein/nucleotide index rule. The
adapter and consuming model must agree on the individual residue-index
vocabulary, 65 atom-slot names/order, rigid-group meanings, frame convention,
and torsion order. cryoFlake does not define or infer those mappings. The
validator checks shapes, finiteness, masks, counts, and ordering; it does not
prove that a frame is orthonormal, a sine/cosine pair has unit norm, or a
masked atom has a zero coordinate. It also does not compare full-atom positions
with backbone coordinates, map integer `aatype` values back to residue labels,
or verify that topology indices identify the deposited structure. Those
alignments are the adapter's responsibility. Extra full-atom mapping keys are
ignored and omitted by the normalizing validator. Full-atom padding is
controlled by masks and does not use the component padding rule below.

`extract_full_atom_targets()` extracts these fields from an application's
protein object, reorders its rows into protein-then-nucleotide order, and
validates the result. See
[`targets/full_atom.py`](../src/cryoflake/targets/full_atom.py).

### Component extension, schema version 1

`extension_targets` contains exactly `schema_version` and `component`.
`schema_version` is the integer `1`, not a boolean. The `component` mapping
contains exactly the fields below; missing and unexpected keys are errors.
This extension adds nodes outside the canonical residue-label sets rather
than replacing the backbone/full-atom nodes.

Let `N` be the number of component nodes, `A` the padded atom count, and `K`
the number of proposals. `N = 0` is valid; the builder uses `A = 0` then.

| Field | Shape/value | Normalized dtype or rule |
| --- | --- | --- |
| `positions` | `(N, 3)` | Finite `float32` world-space anchors. |
| `node_kind` | `(N,)` | `int64`, `NodeKind` values below. |
| `polymer_kind` | `(N,)` | `int64`, `PolymerKind` values below. |
| `anchor_kind` | `(N,)` | `int64`, `AnchorKind` values below. |
| `reference_local_positions` | `(N, 3, 3)` | Finite `float32`, canonical anchor-centred local geometry. |
| `component_id` | `(N,)` | Nonempty strings; stripped and uppercased. |
| `instance_id` | `(N,)` | Nonempty unique strings identifying individual instances. |
| `chain_id` | `(N,)` | Strings; empty allowed. |
| `residue_index` | `(N,)` | Integers fitting `int64`; signed values allowed. |
| `insertion_code` | `(N,)` | Strings; empty allowed. |
| `source_chem_comp_type` | `(N,)` | Strings; stripped and uppercased; empty allowed. |
| `atom_positions` | `(N, A, 3)` | Finite `float32` world-space positions. |
| `atom_mask` | `(N, A)` | Boolean/0-or-1 input, normalized to `bool`. |
| `atom_names`, `atom_elements` | `(N, A)` | String arrays; stripped and uppercased. |
| `b_factors` | `(N, A)` | Finite `float32`. |
| `occupancies` | `(N, A)` | Finite nonnegative `float32`; not restricted to at most 1. |
| `interface_mask` | `(N,)` | Boolean/0-or-1 input, normalized to `bool`. |
| `interface_group_id` | `(N,)` | Nonempty exactly where `interface_mask` is true. |
| `interface_contact_cutoff` | Scalar | Positive finite number, in ångströms. |
| `proposal_targets` | Mapping | Exactly `node_indices`, `positions`, `valid_mask`. |
| `split_groups` | Mapping | Exactly `component`, `interface`; each is a sequence of group-ID strings. |

Enum numbers are serialized identifiers. Import their definitions from
`cryoflake.constants`.

| Enum | Values |
| --- | --- |
| `NodeKind` | `POLYMER=0`, `NON_POLYMER=1`, `ION=2`, `WATER=3`, `UNKNOWN=4` |
| `PolymerKind` | `NONE=0`, `PROTEIN=1`, `RNA=2`, `DNA=3`, `NUCLEIC_ACID=4`, `CARBOHYDRATE=5`, `OTHER=6` |
| `AnchorKind` | `UNSPECIFIED=0`, `C_ALPHA=1`, `PHOSPHORUS=2`, `ATOM=3`, `CENTROID=4` |

Cross-field rules are enforced:

- A `POLYMER` node has a non-`NONE` polymer kind; all other node kinds use
  `NONE`. Only `UNKNOWN` nodes use `UNSPECIFIED` anchors, and every unknown
  node must use one.
- `C_ALPHA` anchors require protein polymers; `PHOSPHORUS` anchors require
  RNA, DNA, or nucleic-acid polymers. Ions and water require `ATOM` anchors.
- Each node has at least one present atom. Present atoms occupy consecutive
  columns from zero, followed by masked padding. Present atoms have nonempty
  names and elements; padded names/elements are empty and padded positions,
  B factors, and occupancies are zero. An ion has exactly one non-H/D atom;
  water has exactly one non-H/D atom, an oxygen.
- CA/P anchors coincide with a present atom with the corresponding name;
  `ATOM` anchors coincide with a present atom. A `CENTROID` anchor is the mean
  of non-H/D atom positions, falling back to all atoms if there are no heavy
  atoms. These comparisons use NumPy `isclose`/`allclose` with `atol=1e-4`
  and their default `rtol=1e-5`. Padded numeric fields use `allclose` against
  zero with its default `atol=1e-8`.
- Local reference point 1 is the origin. Point 0 lies on the nonpositive
  x-axis; point 2 lies in the xy-plane with nonnegative y. Canonical-frame
  checks use a `1e-5` absolute tolerance. Degenerate zero geometry is allowed.
- An interface ID has the form `<component_id>|<16 lowercase hex characters>`.
  `split_groups.component` is the sorted unique set of non-water component
  IDs. `split_groups.interface` is the sorted unique set of IDs on interface
  nodes. Group-ID sequences are stripped, deduplicated, sorted, and checked
  against these sets.

`proposal_targets.node_indices` is an integer `(K,)` array with unique values
in `[0, N)`. `proposal_targets.positions` is finite `float32` `(K, 3)` and
must exactly equal `positions[node_indices]` after normalization.
`proposal_targets.valid_mask` is boolean `(K,)`. The builder proposes every
node and computes validity using the grid bounds above; the schema validator
checks the mask's type/shape, not its agreement with a particular grid.

`build_component_extension_targets()` reads the first data block and first
encountered model in an uncompressed `.cif` or `.mmcif` file. It excludes
canonical residue IDs and skips atoms with missing or nonfinite coordinates.
For each atom name it selects the highest-occupancy alternative, breaking ties
by blank alternate ID, then lexicographic alternate ID, then source order.
Atoms are stored in sorted atom-name order. Competing component identities at
one residue location are resolved by highest mean selected-atom occupancy,
then lexicographic component ID. Negative or invalid occupancies become zero;
invalid B factors become zero. Node order follows first encountered residue
locations, independently of protein/nucleotide backbone ordering.

The builder prefers author chain and residue IDs, falling back to label IDs
(and residue index zero when neither residue ID parses). `instance_id` is a
compact JSON string containing the model, entity, label chain/residue, author
chain/residue, insertion code, and component ID. Treat it as an opaque identity
within the artifact. Classification uses mmCIF entity/polymer/chemistry
metadata and the shared component/element vocabularies. Protein nodes use CA
anchors when N/CA/C are present; nucleic-acid nodes use P when present;
carbohydrates prefer C1 or C1'; other nodes use a single heavy atom or the
heavy-atom centroid. Unknown nodes retain their computed position but use
`UNSPECIFIED` anchor kind.

Supplying polymer atom positions and their node/topology arrays enables
interface detection: a component heavy atom at or below the cutoff from a
supplied polymer atom marks an interface. A component with no heavy atoms uses
all its atoms. The context requires finite `(M, 3)` atom positions, nonnegative
integer `(M,)` atom-to-node indices, and matching one-dimensional chain,
residue, and `aatype` integer arrays covering every referenced node. Without
context, all interface flags are false and interface IDs/groups are empty.
IDs use the first 16 lowercase hex characters of SHA-256 over
`repr(sorted(set(contacting_topology_tuples))).encode()`, where each tuple is
`(chain_index, residue_index, aatype)` converted to Python integers.

Component validation does not infer classification from chemistry, compare
reference geometry with atom positions, or recompute interfaces from the
full-atom targets. Supply the builder's normalized result when those derived
values are needed.

See [`targets/schema.py`](../src/cryoflake/targets/schema.py),
[`targets/components.py`](../src/cryoflake/targets/components.py), and
[`targets/parsers.py`](../src/cryoflake/targets/parsers.py) for the full rules.

### Sequence embedding inputs and stored fields

`load_sequence_embedding_file()` accepts a local numeric `.npy` array or a
`.npz` file with a `sequence` field. NumPy object arrays are rejected without
deserialization. Let `L` be the embedding row count and `D` the feature width.

| Field | Required | Rules |
| --- | --- | --- |
| `sequence` | Yes | Numeric `(L, D)` array, `L > 0`, `D > 0`; normalized to finite `float32`. |
| `sequence_mask` | No | Boolean or numeric 0/1 values; flattened length must be `L`, with at least one true value. Missing means all true. Normalized to `(L,)` `bool`. |
| `unified_seq` | No, input metadata | Scalar string or UTF-8 bytes; the sum of sequence lengths after splitting on `|||` must equal `L`. Separators do not count as residues. |
| `unified_seq_len` | No, input metadata | Nonnegative scalar integer, excluding booleans; must equal `L`. |

For an NPZ input, optional metadata is validated before normalization.
`validate_sequence_embedding_payload()` returns only normalized `sequence`
and `sequence_mask`. Sidecar loading and attachment therefore do not carry
`unified_seq`, `unified_seq_len`, or unrelated NPZ fields into the artifact.
`attach_sequence_embeddings_to_preprocessed_artifact()` replaces the stored
sequence fields without changing the core store.

The sequence contract checks embedding/metadata length consistency. It does
not align embeddings to chains, compare sequence letters with structure
residues, or require `L = P + Q`. The application must define that alignment
and the embedding model/feature meaning. A valid mask may exclude rows but
cannot exclude every row. `require_sequence_embeddings` turns missing
embeddings into a validation error; present malformed embeddings always fail.

See [`sequence/validation.py`](../src/cryoflake/sequence/validation.py) and
[`sequence/sidecars.py`](../src/cryoflake/sequence/sidecars.py).

## Artifact lists and split reports

An artifact list is a text file with one artifact root or `metadata.json`
reference per nonblank line. Entries may be local paths or S3 URIs. Local
relative entries resolve from the list file's directory; S3 URIs retain their
URI spelling. Blank lines are ignored; comments have no special syntax. An
empty list can be written, but `read_artifact_list()` rejects an empty input.

The default `artifact` split strategy deduplicates artifact aliases before
partitioning. It uses `validation_fraction` strictly between 0 and 1 and a
deterministic seed. With two or more unique artifacts it keeps both outputs
nonempty. A singleton remains in training and yields an empty validation file.

Component/interface splitting uses `GroupedSplitRecord` values containing
`artifact_reference`, a nonempty `artifact_group_id`, and normalized `component_group_ids`
and `interface_group_ids`. Records sharing an artifact reference or `artifact_group_id` stay
together. Here local path identity uses the resolved filesystem path and remote
path identity uses the exact URI string. Unlike the `artifact` strategy,
grouped splitting does not normalize root versus `metadata.json` artifact
aliases or existing hardlinks; use consistent references and stable artifact
group IDs for these peers.
Every record containing a selected holdout group moves to validation, together
with those peers. Co-occurring unselected groups are not automatically held
out, so they may also appear in training. Grouped splitting requires at least
two records and enough independent groups to leave both outputs nonempty.

`holdout_group_ids_path` accepts one nonempty group ID per line, a JSON string
list, or a version-3 grouped split report. IDs are stripped, deduplicated, and
sorted. An explicit selection must be nonempty and contain only known IDs.
A report used as input must declare a matching `component` or `interface`
strategy.

Report inputs must contain the complete version-3 schema below. Missing or
unexpected fields, duplicate JSON keys, nonfinite JSON numbers, invalid scalar
or list types, inconsistent counts or row order, and conflicting holdout or
atomic-peer assignments are rejected. Historical artifact references are not
resolved or loaded when extracting holdout IDs, so a report can be reused in
a different working directory or dataset run. Plain text and JSON string-list
inputs remain available when only group IDs are needed.

`write_grouped_split_report()` writes the following JSON structure:

| Field | Value |
| --- | --- |
| `schema_version` | `3` |
| `split_strategy` | `"component"` or `"interface"` |
| `validation_fraction`, `seed` | Requested fraction and deterministic seed. |
| `selected_holdout_group_ids` | Sorted list of selected IDs; reusable as a holdout input. |
| `num_records`, `num_train_records`, `num_validation_records` | Record and assignment counts. |
| `assignments` | One object per input record, preserving input order. |

Each assignment has `row_index` (one-based), `artifact_reference`, `artifact_group_id`,
`component_group_ids`, `interface_group_ids`, and `split` (`"train"` or
`"valid"`). Report row indices describe the split-record sequence, which can
exclude unsuccessful preprocessing rows; they are not original manifest row
numbers. See [`manifests/splits.py`](../src/cryoflake/manifests/splits.py).

## Artifact validation results

`validate_preprocessed_artifact()` returns an `ArtifactValidationResult`;
`validate_artifact_list()` returns one result per artifact-list entry in
list order and stops at the first invalid artifact. They do not deduplicate
list entries. `validate-artifact` and `result.to_json()` serialize these fields:

| Field | Value |
| --- | --- |
| `artifact_reference` | Artifact reference supplied to validation, as a string. |
| `grid_shape` | Three dimensions in `(Z, Y, X)` order; tuple in Python, list in JSON. |
| `protein_nodes`, `nucleotide_nodes`, `component_nodes` | Counts of backbone protein rows, backbone nucleotide rows, and component nodes. |
| `sequence_length`, `sequence_feature_dim` | Embedding row count and feature width, or Python `None` / JSON `null` when embeddings are absent. |

These are computed summaries, not an additional on-disk artifact schema.
See [`storage/artifacts.py`](../src/cryoflake/storage/artifacts.py).

## Preprocessing summary CSV

After a completed run, the summary has one row per manifest row, including
skipped and continuing failed rows. Interrupted or fail-fast runs can leave
checkpoint state without writing a new summary; resume the run to finalize
it. All cells are text; missing/not-applicable values are empty.

| Columns | Meaning |
| --- | --- |
| `emdb_id`, `wwpdb_id`, `map_url`, `structure_url`, `map_path`, `structure_path` | Input identity and source locations. |
| `artifact_reference` | Intended local artifact root or remote S3 prefix; consult `success` before consuming it. |
| `cached_artifact_dir` | Retained local artifact directory when upload failed or cleanup could not finish. |
| `source_resolution` | Manifest `source_resolution`, with surrounding whitespace stripped. |
| `input_voxel_size`, `artifact_voxel_size` | Input header x-axis spacing when readable and output isotropic spacing, in ångströms per voxel. |
| `grid_shape` | Output dimensions joined with `x`, in `(Z, Y, X)` order, such as `64x80x96`. |
| `component_node_count`, `component_interface_count` | Component-node and interface-node counts. |
| `component_group_ids`, `interface_group_ids` | Semicolon-separated group IDs, empty when no groups exist. |
| `split`, `split_seed` | Assigned `train` or `valid` and seed; blank split when no artifact list was requested. |
| `success` | Lowercase `true` or `false`. |
| `status` | Outcome from the table below. |
| `error` | Error/skip description; exceptions include their class name. |
| `finished_at` | UTC ISO-8601 timestamp with seconds precision. |

| Status | Meaning |
| --- | --- |
| `processed` | Adapter output passed validation and any requested upload finished. |
| `reused` | Existing target artifact reused successfully. |
| `reused_external` | Artifact from a configured reuse directory reused successfully. |
| `skipped_no_structure` | No structure path was supplied. |
| `download_failed` | A preceding download failed. |
| `skipped_missing` | Required local input was missing. |
| `skipped_oversized_map` | Estimated map preprocessing memory exceeded the configured admission limit. |
| `failed` | Preprocessing or validation failed. |
| `upload_failed` | Remote publication failed; cached output is retained when available. |

Only successful artifacts enter artifact lists. Checkpoint journals next to
the summary are managed recovery state rather than a dataset interchange
format. Their metadata uses `format_version: 5`; serialized grouped split
records use `artifact_reference` and `artifact_group_id`. Unsupported checkpoint
versions require an
explicit reset before starting a new journal. These recovery and split-report
versions are independent of the NPZ+JSON format and component-target schema,
which remain at version `1`. See the [user guide](user-guide.md) for retrying
interrupted runs and the
[summary implementation](../src/cryoflake/preprocess/summary.py) for the fixed
output column order.

Recovery validates each row's canonical group record and agreement between
its artifact, target, summary, and grouped-record references before resuming
it. Invalid rows fall back to preprocessing; a malformed pending row is not
staged for upload. Valid root and `metadata.json` references to the same
artifact are normalized consistently in recovered results.
