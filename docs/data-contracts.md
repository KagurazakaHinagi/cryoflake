# Data contracts

Use this reference to produce or read a cryoFlake training artifact: its density
grid, molecular coordinates, supervision targets, and optional sequence
embeddings. The [artifact format](artifact-format.md) defines their NPZ+JSON
encoding; this page defines their shapes, units, and relationships.

Choose the section for your task:

- Produce or read training arrays: [complete training payload](#complete-training-payload).
- Prepare inputs: [source association CSV](#source-association-csv) and
  [preprocessing manifest CSV](#preprocessing-manifest-csv).
- Inspect outputs: [artifact lists and split reports](#artifact-lists-and-split-reports),
  [validation results](#artifact-validation-results), and
  [preprocessing summary CSV](#preprocessing-summary-csv).

| Contract | Current version | Scope |
| --- | --- | --- |
| Semantic preprocessing | `"S2610210"` | The complete training payload and its validation rules. |
| NPZ+JSON storage | `"N2610210"` | Metadata, array stores, and storage integrity. |
| Full-atom targets | `"F2610210"` | `full_atom_targets.schema_version`. |
| Component extension | `"E2610210"` | `extension_targets.schema_version`. |
| Grouped split reports | `"G2610210"` | Split-report JSON `schema_version`. |
| Preprocessing checkpoints | `"C2610210"` | Checkpoint metadata `format_version`. |

Schema identifiers are strings in the form `PYYMMDDX`: a domain prefix, the
release date as `YYMMDD`, and a single revision digit from `0` through `9`.
For example, `"F2610210"` means full-atom targets released on October 21,
2026, revision 0. Use `S` for semantic preprocessing, `N` for NPZ+JSON
storage, `F` for full-atom targets, `E` for component extensions, `G` for
grouped split reports, and `C` for preprocessing checkpoints. Each domain
advances independently. Quote these identifiers in JSON, YAML, and
Python; do not convert them to numbers.

Readers require the exact supported identifier for each domain. A different
prefix, date, or revision is unsupported even if it follows the naming
pattern. Previous integer versions are rejected; there is no automatic
conversion. The semantic preprocessing version describes validation rules;
no semantic-version field is required in the payload or `metadata.json`.
The metadata field `format_version` is the NPZ+JSON storage identifier
`"N2610210"`.

These identifiers describe separate contracts. The runtime descriptors are
[artifact/npz_json.yaml](../src/cryoflake/configs/contracts/artifact/npz_json.yaml),
[targets/component.yaml](../src/cryoflake/configs/contracts/targets/component.yaml),
and [manifest/emdb.yaml](../src/cryoflake/configs/contracts/manifest/emdb.yaml).
See the [configuration reference](configuration.md) for their composition.

## Complete training payload

A preprocessing adapter must write the required roots below. Use these exact
names for local and S3 artifacts. Extra portable top-level fields are allowed.
Sequence embeddings are optional unless the pipeline requires them.

| Key | Required | Value |
| --- | --- | --- |
| `density_grid` | Yes | Nonempty, finite numeric array with shape `(Z, Y, X)`. |
| `global_origin` | Yes | Finite numeric `(3,)` Cartesian `(x, y, z)` array, in ångströms. |
| `voxel_size` | Yes | Positive finite numeric scalar, in ångströms per voxel; spacing is isotropic. |
| `backbone_coordinates` | Yes | Mapping with `CA`, `C`, `N`, `O`, and `P` coordinate arrays. |
| `residue_info` | Yes | Mapping with `aa_type` and `nuc_type` residue-label sequences. |
| `full_atom_targets` | Yes | Full-atom supervision, topology, and the shared atom-name table below. |
| `extension_targets` | Yes | Component extension below, including when there are no component nodes. |
| `sequence`, `sequence_mask` | Optional | Embedding fields below; `sequence_mask` requires `sequence`. |

### Validation and saved values

`validate_preprocessed_artifact()` and `validate_artifact_list()` check the
complete training contract, as does manifest preprocessing. Sequence fields
are checked whenever present, even when embeddings are optional.

Storage has a broader role: `write_preprocessed_artifact()` and
`load_preprocessed_artifact()` accept generic portable mappings and check
storage integrity. A generic mapping does not need the training roots above.
The writer validates and converts a top-level `density_grid` to `float32` when
present. Other arrays retain their supplied dtypes.

The dtype tables on this page describe the normalized results of individual
validators. Semantic artifact validation checks normalized views and does not
rewrite the saved arrays. To store canonical target dtypes, write the mappings
returned by `validate_full_atom_targets()` and
`validate_component_extension_targets()`, and normalize sequence fields before
writing them.

`validate_preprocessed_artifact_mapping(artifact_mapping)` checks only that the
input implements `Mapping`. Use semantic artifact validation to check a
complete training artifact, or the storage loader to check a generic mapping.
See the [API reference](api-reference.md#targets-and-metadata).

### Dimensions and coordinate units

| Symbol | Meaning |
| --- | --- |
| `Z`, `Y`, `X` | Density-array dimensions, in array index order. |
| `P`, `Q`, `R` | Protein-residue count, nucleotide-residue count, and their sum `R = P + Q`. |
| `N`, `A` | Component-node count and padded atom count per component. |
| `K` | Number of component proposals. |
| `L`, `D` | Sequence-embedding row count and feature width. |

Atom coordinates and component anchors use world-space Cartesian `(x, y, z)`
positions in ångströms. Density arrays use `(z, y, x)` indexing. Check whether
a world position lies inside the density grid with:

<!-- doc-example: contract-geometry; mode=python -->
```python
import numpy as np

relative_position_xyz = np.asarray(position_xyz, dtype=np.float64) - global_origin
# density_grid is indexed as density_grid[z, y, x].
extent_xyz = np.asarray(density_grid.shape[::-1], dtype=np.float64) * voxel_size
inside = ((relative_position_xyz >= 0) & (relative_position_xyz < extent_xyz)).all()
```

The lower grid boundary is inclusive and the upper boundary is exclusive.
At least one finite CA or P backbone anchor must fall inside the density grid;
other anchors may fall outside it. A valid grid is nonempty and three-dimensional
and may be noncubic. Density amplitudes have no required normalization or range.
Individual grid-preparation helpers may impose cubic or even-size requirements.

Grid, backbone, origin, voxel-size, and embedding validators accept real integer
or floating dtypes. They reject boolean, complex, object, and string dtypes for
these numeric fields. Coordinates, origins, and normalized grids are `float32`;
voxel size is a Python `float`.

For new adapters, supply the density array directly. The grid validator also
accepts readable safe NumPy grid payloads, including compressed `arr_0` input;
the writer converts them into a numeric array.

### Backbone and residue labels

The protein coordinate arrays contain `P` rows and nucleotide phosphorus
coordinates contain `Q` rows. Keep their label sequences aligned with those
rows:

| Field | Shape | Rules |
| --- | --- | --- |
| `backbone_coordinates["CA"]`, `backbone_coordinates["C"]`, `backbone_coordinates["N"]`, `backbone_coordinates["O"]` | `(P, 3)` each | Matching row counts; numeric coordinates normalized to `float32`. |
| `backbone_coordinates["P"]` | `(Q, 3)` | Nucleotide phosphorus coordinates, normalized to `float32`. |
| `residue_info["aa_type"]` | Length `P` | Protein residue labels from the vocabulary below. |
| `residue_info["nuc_type"]` | Length `Q` | Nucleotide labels from the vocabulary below. |

Backbone coordinates may contain NaN for missing atoms, but not infinity.
An empty coordinate sequence normalizes to `(0, 3)` and a single `(3,)`
coordinate normalizes to `(1, 3)`. Either residue class may be empty as long as
the artifact still has a finite CA or P anchor inside the grid.

Protein labels are `ALA`, `ARG`, `ASN`, `ASP`, `CYS`, `GLN`, `GLU`, `GLY`,
`HIS`, `ILE`, `LEU`, `LYS`, `MET`, `PHE`, `PRO`, `SER`, `THR`, `TRP`, `TYR`,
`VAL`, and `UNK`. Nucleotide labels are `DA`, `DC`, `DG`, `DT`, `A`, `C`, `G`,
`U`, `DN`, and `N`. Label validation uppercases values without stripping
whitespace.

`load_full_backbone_from_structure()` uses the first model unless
`all_models=True`, writes NaN for missing backbone atoms, and selects per-atom
alternate locations by identifier. Its additional `residue_info.chain` and
`residue_info.secondary_structure` annotations follow structure traversal
across both residue classes. They are not guaranteed to follow the
protein-then-nucleotide full-atom row order, and semantic artifact validation
does not require or check them.

See [`metadata.py`](../src/cryoflake/metadata.py),
[`constants.py`](../src/cryoflake/constants.py),
[`io/structure.py`](../src/cryoflake/io/structure.py), and
[`preprocess/reuse.py`](../src/cryoflake/preprocess/reuse.py).

### Full-atom supervision

`full_atom_targets` contains the fields below. Its residue-indexed arrays have
`R = P + Q` rows: all proteins first, followed by all nucleotides, matching the
corresponding backbone coordinate arrays. It also contains one shared
`atom_names` table for all 65 slots.

`schema_version` must be the string `"F2610210"`; missing or unsupported
identifiers and non-string values are rejected. Import
`FULL_ATOM_TARGET_SCHEMA_VERSION` from `cryoflake` or `cryoflake.constants` when producing this mapping.

| Field | Shape/value | Normalized dtype |
| --- | --- | --- |
| `schema_version` | Scalar | `str` |
| `aatype` | `(R,)` | `int64` |
| `chain_index` | `(R,)` | `int64` |
| `residue_index` | `(R,)` | `int64` |
| `atom_positions` | `(R, 65, 3)` | `float32` |
| `atom_mask` | `(R, 65)` | `bool` |
| `atom_names` | `(65,)` | Unicode |
| `b_factors` | `(R, 65)` | `float32` |
| `rigidgroups_gt_frames` | `(R, 9, 3, 4)` | `float32` |
| `rigidgroups_alt_gt_frames` | `(R, 9, 3, 4)` | `float32` |
| `rigidgroups_gt_exists` | `(R, 9)` | `bool` |
| `rigidgroups_group_exists` | `(R, 9)` | `bool` |
| `rigidgroups_group_is_ambiguous` | `(R, 9)` | `bool` |
| `torsion_angles_sin_cos` | `(R, 8, 2)` | `float32` |
| `alt_torsion_angles_sin_cos` | `(R, 8, 2)` | `float32` |
| `torsion_angles_mask` | `(R, 8)` | `bool` |

#### Atom slots and presence

The shared table has shape `(65,)`, including when `R = 0`. Its entry at slot
`s` names `atom_positions[row, s]` for every full-atom residue row. The adapter
supplies the complete table and the consuming model must use the same slot
convention. cryoFlake does not infer names from slot numbers.

All 65 entries must contain actual strings or UTF-8 byte strings. Validation
strips whitespace, uppercases names, and returns a Unicode array wide enough
for the longest name. Every normalized name must be nonempty, unique across
the entire table, and match `[A-Z0-9][A-Z0-9'"*+_-]*`. These requirements also
apply to slots masked in every residue. Non-string scalars are rejected.

`atom_mask` selects present atoms independently for each residue. Names never
reorder slots or fill sparse masks. Masked full-atom coordinates are not
required to be zero; this differs from component padding below.

#### Labels, topology, and numerical rules

`aatype` must be an integer array with values in `[0, 31)`. Indices `0` through
`19` and `28` are protein types; `28` is the unknown-protein index. Other
accepted indices are treated as nucleotide types. Derive protein membership
with `(aatype < 20) | (aatype == 28)`: it must produce `P` true entries followed
by `Q` false entries. A `prot_mask` field is not accepted in this mapping.

`chain_index` is a nonnegative integer array. `residue_index` is an integer
array and may contain negative values. Both must fit `int64`.

All floating target values must be finite, including masked values, and remain
finite after conversion to `float32`. Masks accept booleans or numeric 0/1
values and reject other values before conversion.

#### Adapter responsibilities

The adapter and model must agree on the residue-index vocabulary, rigid-group
meanings, frame convention, and torsion order. Validation checks shapes,
finiteness, masks, counts, and row order. It does not establish orthonormal
frames or unit-length sine/cosine pairs, compare full-atom positions with
backbone coordinates, map integer `aatype` values back to residue labels, or
verify that topology indices identify the deposited structure.

Extra full-atom keys other than the rejected `prot_mask` field are omitted by
the normalizing validator. `extract_full_atom_targets()` requires an
application object with the target arrays, topology, and shared atom-name
table. It reorders residue arrays into protein-then-nucleotide order, preserves
the table's slot order, and emits the current schema version itself. See
[`targets/full_atom.py`](../src/cryoflake/targets/full_atom.py).

#### Read named atom coordinates

Normalize the target mapping before writing it, then use each residue's mask
with the shared name table to recover named world-space atom coordinates:

<!-- doc-example: contract-full-atom-roundtrip; mode=python -->
```python
import numpy as np

from cryoflake import (
    load_preprocessed_artifact,
    validate_full_atom_targets,
    write_preprocessed_artifact,
)

artifact_mapping = load_preprocessed_artifact("data/artifacts/example")
targets = validate_full_atom_targets(artifact_mapping["full_atom_targets"])
artifact_mapping["full_atom_targets"] = targets
write_preprocessed_artifact("data/artifacts/named-roundtrip", artifact_mapping)
restored = load_preprocessed_artifact("data/artifacts/named-roundtrip")
restored_targets = restored["full_atom_targets"]
assert restored_targets["schema_version"] == "F2610210"
np.testing.assert_array_equal(restored_targets["atom_names"], targets["atom_names"])
protein_mask = (restored_targets["aatype"] < 20) | (restored_targets["aatype"] == 28)

named_residues = []
for row in range(len(restored_targets["aatype"])):
    atom_mask = restored_targets["atom_mask"][row]
    atom_positions_by_name = dict(
        zip(
            restored_targets["atom_names"][atom_mask].tolist(),
            restored_targets["atom_positions"][row, atom_mask],
            strict=True,
        )
    )
    named_residues.append(
        {
            "chain_index": int(restored_targets["chain_index"][row]),
            "residue_index": int(restored_targets["residue_index"][row]),
            "atom_positions_by_name": atom_positions_by_name,
        }
    )
```

### Component extension

`extension_targets` has exactly two keys: string `schema_version: "E2610210"`
and `component`. Non-string or unsupported identifiers, missing keys, and
extra keys are rejected. The component mapping must contain exactly the
fields in the following table.

Components describe nodes outside the canonical backbone residue-label sets.
They have their own padded `(N, A)` atom names and elements; the shared
full-atom name table does not apply to them. An artifact with no components
still supplies this extension with `N = 0`; the builder then uses `A = 0`.

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
| `atom_names`, `atom_elements` | `(N, A)` | String arrays; stripped and uppercased. |
| `b_factors` | `(N, A)` | Finite `float32`. |
| `occupancies` | `(N, A)` | Finite nonnegative `float32`; not restricted to at most 1. |
| `interface_group_id` | `(N,)` | Stripped strings; nonempty IDs identify interface nodes. |
| `interface_contact_cutoff` | Scalar | Positive finite number, in ångströms. |
| `proposal_targets` | Mapping | Exactly `node_indices`, `valid_mask`. |

Enum numbers are serialized identifiers. Import them from `cryoflake.constants`:

| Enum | Values |
| --- | --- |
| `NodeKind` | `POLYMER=0`, `NON_POLYMER=1`, `ION=2`, `WATER=3`, `UNKNOWN=4` |
| `PolymerKind` | `NONE=0`, `PROTEIN=1`, `RNA=2`, `DNA=3`, `NUCLEIC_ACID=4`, `CARBOHYDRATE=5`, `OTHER=6` |
| `AnchorKind` | `UNSPECIFIED=0`, `C_ALPHA=1`, `PHOSPHORUS=2`, `ATOM=3`, `CENTROID=4` |

String fields accept actual strings or UTF-8 byte strings and normalize to
Unicode without coercing non-string scalars. Component IDs, source
chemical-component types, atom names, and elements are stripped of surrounding
whitespace and uppercased. Interface group IDs are stripped without changing
case. Instance IDs, chain IDs, and insertion codes preserve their text.

#### Atoms, padding, and anchors

Derive atom presence from normalized `atom_names != ""`. Each node has at
least one present atom. Present atoms occupy consecutive columns starting at
zero and have nonempty names and elements. Padding follows those atoms, with
empty names/elements and zero positions, B factors, and occupancies. An ion
has exactly one non-H/D atom; water has exactly one non-H/D atom, an oxygen.

The validator also enforces these relationships:

- A `POLYMER` node has a non-`NONE` polymer kind; all other node kinds use
  `NONE`. Exactly the `UNKNOWN` nodes use `UNSPECIFIED` anchors.
- `C_ALPHA` anchors require protein polymers; `PHOSPHORUS` anchors require
  RNA, DNA, or nucleic-acid polymers. Ions and water use `ATOM` anchors.
- A CA/P anchor coincides with a present atom of that name. An `ATOM` anchor
  coincides with a present atom. A `CENTROID` anchor is the mean of non-H/D
  atom positions, or all atoms when none are heavy. Comparisons use NumPy
  `isclose`/`allclose` with `atol=1e-4` and default `rtol=1e-5`. Padded numeric
  fields use `allclose` against zero with default `atol=1e-8`.
- Local reference point 1 is the origin. Point 0 lies on the nonpositive
  x-axis; point 2 lies in the xy-plane with nonnegative y. Canonical-frame
  checks use absolute tolerance `1e-5`; degenerate zero geometry is allowed.

#### Interfaces and proposals

A nonempty `interface_group_id` marks an interface node. It has the form
`<component_id>|<16 lowercase hex characters>`. Component groups are the sorted
unique non-water component IDs. Interface groups are the sorted unique
nonempty interface IDs.

`proposal_targets` contains exactly `node_indices` and `valid_mask`.
`node_indices` is an integer `(K,)` array with unique values in `[0, N)`; its
subset and order are preserved. Obtain proposal coordinates from
`positions[node_indices]`. `valid_mask` is a stored boolean `(K,)` array,
accepting boolean or numeric 0/1 input. The validator checks its type and shape
and preserves its values without recomputing them from a density grid.

The builder proposes every component node and computes validity from the grid
bounds described above. Consumers may supply their own proposal subset,
ordering, and validity values.

#### Derive reader values

Atom and interface membership, proposal coordinates, and group lists are
computed reader values. The component mapping does not accept `atom_mask`,
`interface_mask`, or `split_groups`, and the proposal mapping does not accept
`positions`. Derive the values from the normalized stored fields:

<!-- doc-example: contract-component-derived; mode=python -->
```python
from cryoflake import (
    NodeKind,
    load_preprocessed_artifact,
    validate_component_extension_targets,
)

artifact_mapping = load_preprocessed_artifact("data/artifacts/example")
component = validate_component_extension_targets(artifact_mapping["extension_targets"])[
    "component"
]
component_atom_mask = component["atom_names"] != ""
component_interface_mask = component["interface_group_id"] != ""
proposal_positions = component["positions"][
    component["proposal_targets"]["node_indices"]
]
component_groups = tuple(
    sorted(
        set(
            component["component_id"][
                component["node_kind"] != int(NodeKind.WATER)
            ].tolist()
        )
    )
)
interface_groups = tuple(
    sorted(set(component["interface_group_id"][component_interface_mask].tolist()))
)
```

#### Build components from a structure

`build_component_extension_targets()` reads the first data block and first
encountered model in an uncompressed `.cif` or `.mmcif` file. It excludes
canonical residue IDs and skips atoms with missing or nonfinite coordinates.
For each atom name it picks the highest-occupancy alternative, breaking ties
by blank alternate ID, lexicographic alternate ID, then source order. Atom
columns are sorted by name. Competing component identities at one residue
location are resolved by highest mean selected-atom occupancy, then
lexicographic component ID. Invalid or negative occupancies become zero;
invalid B factors become zero. Nodes follow first encountered residue locations
independently of backbone row order.

The builder prefers author chain and residue IDs, falls back to label IDs,
and uses residue index zero when neither index parses. `instance_id` is an
opaque identity within the artifact: a compact JSON string containing model,
entity, label chain/residue, author chain/residue, insertion code, and
component ID.

Classification uses mmCIF entity/polymer/chemistry metadata and shared
component/element vocabularies. Protein nodes use CA anchors when N/CA/C are
present; nucleic-acid nodes use P when present; carbohydrates prefer C1 or C1'.
Other nodes use a single heavy atom or the heavy-atom centroid. Unknown nodes
retain their computed positions with `UNSPECIFIED` anchor kind.

To detect interfaces, supply finite `(M, 3)` polymer atom positions,
nonnegative integer `(M,)` atom-to-node indices, and matching one-dimensional
chain, residue, and `aatype` integer arrays covering every referenced node.
Context coordinates must remain finite after `float32` conversion, and
context integers must fit `int64`.
A component is at an interface when one of its heavy atoms is at or below the
cutoff from a supplied polymer atom; components without heavy atoms use all
atoms. Without context, interface IDs are empty. Each ID suffix is the first
16 lowercase hexadecimal characters of SHA-256 over
`repr(sorted(set(contacting_topology_tuples))).encode()`, with each tuple
`(chain_index, residue_index, aatype)` converted to Python integers.

Component validation checks the supplied fields; it does not infer chemical
classification, compare local reference geometry with atom positions, or
recompute interfaces from full-atom targets. Use the builder's normalized
result when producing those values from a structure.

See [`targets/schema.py`](../src/cryoflake/targets/schema.py),
[`targets/components.py`](../src/cryoflake/targets/components.py), and
[`targets/parsers.py`](../src/cryoflake/targets/parsers.py).

### Sequence embedding inputs and stored fields

`load_sequence_embedding_file()` accepts a local numeric `.npy` array or a
`.npz` file with a `sequence` field. NumPy object arrays are rejected without
deserialization. Let `L` be the embedding row count and `D` the feature width.

| Field | Required | Rules |
| --- | --- | --- |
| `sequence` | Yes | Numeric `(L, D)` array, `L > 0`, `D > 0`; normalized to finite `float32`. |
| `sequence_mask` | No | Boolean or numeric 0/1 values; flattened length must be `L`, with at least one true value. Missing or `None` in an embedding payload means all true. Normalized to `(L,)` `bool`. |
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
`artifact_reference`, a nonempty `artifact_group_id`, and normalized
`component_group_ids` and `interface_group_ids`. Records sharing an artifact
reference or `artifact_group_id` stay together. Local identity uses the
resolved filesystem path; remote identity uses the exact URI string.

For grouped splitting, use consistent references and stable artifact group
IDs. This strategy does not normalize root versus `metadata.json` artifact
aliases or existing hardlinks as the `artifact` strategy does.

Every record containing a selected holdout group moves to validation, together
with those peers. Co-occurring unselected groups are not automatically held
out, so they may also appear in training. Grouped splitting requires at least
two records and enough independent groups to leave both outputs nonempty.

`holdout_group_ids_path` accepts one nonempty group ID per line, a JSON string
list, or a `"G2610210"` grouped split report. IDs are stripped, deduplicated, and
sorted. An explicit selection must be nonempty and contain only known IDs.
A report used as input must declare a matching `component` or `interface`
strategy.

Report inputs must contain the complete `"G2610210"` schema below. Missing or
unexpected fields, duplicate JSON keys, nonfinite JSON numbers, invalid scalar
or list types, inconsistent counts or row order, and conflicting holdout or
atomic-peer assignments are rejected. Historical artifact references are not
resolved or loaded when extracting holdout IDs, so a report can be reused in
a different working directory or dataset run. Plain text and JSON string-list
inputs remain available when only group IDs are needed.

`write_grouped_split_report()` writes the following JSON structure:

| Field | Value |
| --- | --- |
| `schema_version` | String `"G2610210"`. |
| `split_strategy` | `"component"` or `"interface"` |
| `validation_fraction`, `seed` | Requested fraction and deterministic seed. |
| `selected_holdout_group_ids` | Sorted list of selected IDs; reusable as a holdout input. |
| `num_records`, `num_train_records`, `num_validation_records` | Record and assignment counts. |
| `assignments` | One object per input record, preserving input order. |

Each assignment has `row_index` (one-based), `artifact_reference`,
`artifact_group_id`, `component_group_ids`, `interface_group_ids`, and `split`
(`"train"` or `"valid"`). Report row indices describe the split-record sequence,
which can exclude unsuccessful preprocessing rows. They are not original
manifest row numbers. See
[`manifests/splits.py`](../src/cryoflake/manifests/splits.py).

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

Only successful artifacts enter artifact lists. Checkpoint journals beside
the summary hold recovery state; they are not a dataset interchange format.
Checkpoint metadata uses `format_version: "C2610210"`. Serialized grouped
split records use `artifact_reference` and `artifact_group_id`. An unsupported
checkpoint version requires an explicit reset before starting a new journal.

The checkpoint run identity includes a `target_fields` descriptor containing
the canonical full-atom, component, and nested proposal field sets. A
checkpoint from a different target layout requires an explicit reset even
when the schema identifiers are unchanged.

Recovery validates each row's canonical group record and agreement between
its artifact, target, summary, and grouped-record references before resuming
it. Invalid rows fall back to preprocessing; a malformed pending row is not
staged for upload. Valid root and `metadata.json` references to the same
artifact are normalized consistently in recovered results.

See the [user guide](user-guide.md) for retrying interrupted runs and the
[summary implementation](../src/cryoflake/preprocess/summary.py) for the fixed
output column order.
