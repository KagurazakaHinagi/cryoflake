# Python API reference

Use `import cryoflake` for high-level artifact storage and training validators.
Import workflow, structure-reader, and low-level format functions from their
owning modules below. The [user guide](user-guide.md) covers complete workflows;
the [data contracts](data-contracts.md) specify array shapes and semantics.

## Artifact storage

These functions are available from both `cryoflake` and `cryoflake.storage`
as the same function objects. Their implementation is
[`cryoflake.storage.store`](../src/cryoflake/storage/store.py).

<!-- doc-example: api-storage-signatures; mode=reference -->
```text
load_preprocessed_artifact(
    artifact_reference, *, root_keys=None, include_sequence_embeddings=True,
    s3_endpoint_url=None,
)
write_preprocessed_artifact(artifact_reference, artifact_mapping, *, s3_endpoint_url=None)
replace_preprocessed_artifact_sequence(artifact_reference, sequence_payload, *, s3_endpoint_url=None)
upload_preprocessed_artifact(local_artifact_reference, remote_artifact_reference, *, s3_endpoint_url=None)
```

| Operation | Return value and behavior |
| --- | --- |
| `load_preprocessed_artifact()` | Reconstructed mapping. `artifact_reference` is a local artifact directory, an S3 prefix, or its `metadata.json`. Storage integrity is validated; training semantics are checked separately. |
| `write_preprocessed_artifact()` | Artifact root as `Path` locally or `str` for S3. Creates or replaces the fixed NPZ+JSON layout from a portable mapping. |
| `replace_preprocessed_artifact_sequence()` | `None`. Requires a valid embedding mapping, normalizes it, replaces sequence fields, and preserves `core.npz` bytes. |
| `upload_preprocessed_artifact()` | `None`. Validates local store integrity and publishes to an S3 destination, placing complete metadata last. The local artifact is retained. |
| `preprocessed_artifact_exists(artifact_reference, *, s3_endpoint_url=None)` | `bool`. Checks readable format metadata and existence of required stores; does not replace full array/hash or training validation. |
| `preprocessed_artifact_root(artifact_reference)` | Normalized local `Path` or S3 `str`. Accepts a root or `metadata.json` reference. |
| `preprocessed_artifact_member_reference(artifact_reference, file_name)` | Member `Path` or S3 URI for `metadata.json`, `core.npz`, or `sequence.npz`. |

`root_keys` selects top-level string keys. Unknown keys are ignored. Passing
`include_sequence_embeddings=False` excludes `sequence` and `sequence_mask` even if
selected by `root_keys`. Selected reads do not check every unselected array;
see the [format specification](artifact-format.md) for the validation boundary.
Invalid references and metadata schemas normally raise `ValueError`;
unsupported Python values can raise `TypeError`. Exceptions from local I/O,
ZIP/NumPy decoding, and the S3 SDK may propagate directly, such as `OSError`
or `zipfile.BadZipFile`. There is no single exception class for every storage
failure. Artifact roots must be directories or prefixes, not standalone NumPy
files.

This self-contained example stores a generic mapping, selects fields, and
replaces embeddings:

<!-- doc-example: api-storage; mode=python -->
```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from cryoflake import (
    load_preprocessed_artifact,
    replace_preprocessed_artifact_sequence,
    write_preprocessed_artifact,
)

with TemporaryDirectory() as temporary:
    root = Path(temporary) / "example"
    write_preprocessed_artifact(
        root,
        {
            "features": np.arange(12, dtype=np.float32).reshape(3, 4),
            "sequence": np.ones((3, 8), dtype=np.float32),
        },
    )
    selected = load_preprocessed_artifact(root, root_keys=["features"])
    assert set(selected) == {"features"}
    replace_preprocessed_artifact_sequence(
        root,
        {
            "sequence": np.zeros((3, 16), dtype=np.float32),
        },
    )
    loaded = load_preprocessed_artifact(root)
    assert loaded["sequence"].shape == (3, 16)
    assert loaded["sequence_mask"].all()
```

## Validate training artifacts

Import these from `cryoflake` or
[`cryoflake.storage.artifacts`](../src/cryoflake/storage/artifacts.py):

<!-- doc-example: api-validation-signatures; mode=reference -->
```text
validate_preprocessed_artifact(
    artifact_reference, *, require_sequence_embeddings=False, s3_endpoint_url=None,
)
validate_artifact_list(
    artifact_list_path, *, require_sequence_embeddings=False, s3_endpoint_url=None,
)
```

Both load all stores and enforce the complete training contract, including any
present sequence fields. Missing optional embeddings are accepted unless
`require_sequence_embeddings=True`. A generic mapping from the storage example
above does not satisfy this contract.

The first function returns `ArtifactValidationResult`; the second returns a
list of these results in artifact-list order and stops at the first failure.
The result contains `artifact_reference`, `grid_shape`, `protein_nodes`,
`nucleotide_nodes`, `component_nodes`, `sequence_length`, and
`sequence_feature_dim`. The last two fields are `None` without embeddings.
`result.to_json()` returns a JSON summary.

## Compose and run pipelines

[`cryoflake.config`](../src/cryoflake/config.py) provides:

| Function | Purpose |
| --- | --- |
| `compose_config(overrides=(), *, config_dir=None, config_name="config", pipeline=None, values=None)` | Compose Hydra groups and overrides into a typed `DictConfig`. Python `values` are applied after overrides. Loads the selected environment file before composition. |
| `validate_config(config, *, require_inputs=False, preprocess_adapter_provided=False, sequence_embedding_generator_provided=False)` | Validate typed settings and option combinations. `require_inputs=True` checks required paths and adapter configuration; the provided flags indicate adapters injected by the Python caller. |
| `artifact_contract(config=None)` | Read the artifact contract as an `ArtifactContract` dataclass. Without a config, reads packaged YAML. |
| `target_contract(config=None)` | Read dimensions and schema identifiers as a `TargetContract` dataclass. Without a config, reads packaged YAML. |

`cryoflake.preprocess.prepare.run(config, *, preprocess_adapter=None,
sequence_embedding_generator=None, progress_callback=...)` prepares a source
association CSV, then performs the configured download, embedding, and
preprocessing phases. It returns a `list[ManifestEntry]` for the selected
source associations, and writes the configured manifest. `progress_callback` accepts
messages, defaults to printing progress to stdout, and may be `None`. See the
[configuration reference](configuration.md) for defaults and precedence.
Unlike the CLI, `run()` does not import adapters from configuration strings;
Python callers must inject the callables when those phases are enabled.

For an existing generated manifest, call
[`cryoflake.preprocess.manifest.preprocess_manifest_csv`](../src/cryoflake/preprocess/manifest.py).
It requires `manifest_csv`, `artifact_output_dir`, and an explicit `preprocess_adapter`.
Useful keyword arguments include:

| Keywords | Purpose |
| --- | --- |
| `artifact_s3_prefix`, `s3_endpoint_url` | Publish through a local staging cache to S3. |
| `artifact_list_path`, `validation_artifact_list_path` | Write all successful references or separate train/validation lists. |
| `validation_fraction`, `split_seed`, `split_strategy`, `holdout_group_ids_path`, `split_report_path` | Configure `artifact` (default), `component`, or `interface` splits. |
| `target_voxel_size`, `fix_shift`, `interface_contact_cutoff` | Adapter geometry and interface settings. |
| `require_sequence_embeddings`, `sequence_embedding_loader` | Validate or load sidecar embeddings. |
| `num_workers`, `upload_workers`, `continue_on_preprocess_error` | Control parallel work and failure handling. |
| `summary_csv`, `reset_preprocess_checkpoint`, `preprocess_checkpoint_token` | Enable recovery, explicitly reset progress, or distinguish external adapter state. |
| `force_preprocess`, `skip_missing`, `artifact_reuse_dirs` | Force fresh processing/reset checkpoint progress, skip missing maps, and search validated reuse candidates. Missing structure files are always skipped. |

It returns successful artifact references as `list[Path | str]`. Checkpointing
requires a `summary_csv`; the direct Python function does not choose a summary
path for you. Manifest preparation through `run()` supplies workflow defaults.
Destination collisions and invalid split combinations are rejected before
output creation. Without a compatible live checkpoint, reuse checks artifact
validity, sequence requirements, and contact cutoff rather than matching every
source, adapter, shift, or target-voxel setting. Use `force_preprocess=True` when those
changes require regeneration. The [user guide](user-guide.md) specifies adapter
signatures, failure handling, and recovery rules.

## Sequence embeddings

| API | Behavior |
| --- | --- |
| `cryoflake.sequence.sidecars.load_sequence_embedding_file(sequence_embedding_path)` | Load `.npy` or `.npz` input and return normalized embeddings and mask. |
| `cryoflake.sequence.sidecars.attach_sequence_embeddings_to_preprocessed_artifact(artifact_reference, sequence_embedding_path, sequence_embedding_loader=..., s3_endpoint_url=None)` | Load a sidecar and replace sequence fields in an artifact. |
| `cryoflake.validate_sequence_embedding_payload(payload)` | Return only float32 `sequence` and boolean `sequence_mask`; optional metadata is validated, then omitted from the returned mapping. |
| `cryoflake.validate_sequence_embedding_array(sequence)` | Normalize a finite, nonempty two-dimensional numeric embedding to float32. |
| `cryoflake.validate_sequence_mask(sequence_mask, sequence_length)` | Normalize a 0/1 or boolean mask of matching length; require at least one true row. |
| `cryoflake.validate_sequence_embedding_metadata(payload, sequence_length)` | Check optional `unified_seq` and `unified_seq_len` against the embedding row count. |

See [embedding contracts](data-contracts.md) for mask, length, and metadata rules.

## Structure, FASTA, and density readers

| API | Return value |
| --- | --- |
| `cryoflake.io.structure.load_full_backbone_from_structure(structure_path, *, all_models=False, quiet=True, dssp_secondary_structures=False)` | `(coordinates, residue_info)` mappings. Protein backbone arrays `CA`, `C`, `N`, `O` and nucleotide `P` arrays have shape `(n, 3)` and float32 dtype. Missing atoms use NaNs. |
| `cryoflake.read_fasta(fasta_path, use_author_chain_ids=True)` | `(sequences, names)` lists in file order. `sequences` contains `FASTASequence(sequence, chain_count, chain_ids)` values. Empty files or files without nonempty records raise `FASTAEmptyError`. |
| `cryoflake.io.mrc.load_mrc(mrc_path, *, multiply_global_origin=True)` | `MRCObject(grid, voxel_size, global_origin)` with an owned float32 grid, scalar spacing, and xyz origin. |
| `cryoflake.io.mrc.save_mrc(grid, voxel_size, origin, path)` | Write an MRC map, replacing an existing output. Returns `None`. |
| `cryoflake.io.mrc.prepare_grid(grid, voxel_size, global_origin, target_voxel_size=1.5)` | Return an `MRCObject` after cubic padding and voxel normalization, adjusting origin for the padding shift. |

Backbone readers accept uncompressed `.pdb`, `.ent`, `.cif`, and `.mmcif`.
They use the first model unless `all_models=True`, which concatenates the
per-model results. Residues are grouped using author chain/residue identifiers;
mmCIF uses the first data block. Returned `residue_info` includes chain labels
but does not expose original residue numbers or insertion codes. Adapters must
supply their own full-atom topology. Backbone alternatives use the first
alternate identifier; component targets use highest occupancy with
deterministic ties. The manifest pipeline materializes compressed inputs
before invoking its adapter.

`dssp_secondary_structures=True` reads mmCIF `struct_conf` type identifiers;
it does not run the DSSP program. The default reader also uses `struct_sheet_range`
annotations and returns `HELIX_<class>` (class defaults to `1`), `SHEET`, or
`NULL`. The DSSP mode returns deposited type identifiers or `OTHER` for
unannotated residues. PDB inputs have no mmCIF secondary annotations.

The MRC loader accepts only `(mapc, mapr, maps)` arrangements `(1, 2, 3)`,
`(3, 2, 1)`, and `(2, 1, 3)`, and returns the grid in `(Z, Y, X)` order.
It uses the header's x voxel size as isotropic spacing. Its origin convention
is `header.origin + [nxstart, nystart, nzstart]`, multiplied by that spacing
when `multiply_global_origin=True` (the default). Inspect that convention when
using nonunit spacing or nonzero header offsets: `save_mrc()` writes its
`origin` argument directly to the header and is not an inverse origin transform.

FASTA reads preserve duplicate records and normalize sequence case. RCSB chain
headers can map label chains to author chains; plain headers use chain `A`.
The portable type lives in [`cryoflake.types`](../src/cryoflake/types.py).

This generic storage example preserves FASTA records and their chain identities:

<!-- doc-example: api-fasta; mode=python -->
```python
from pathlib import Path
from tempfile import TemporaryDirectory

from cryoflake import (
    FASTASequence,
    load_preprocessed_artifact,
    read_fasta,
    write_preprocessed_artifact,
)

with TemporaryDirectory() as temporary:
    fasta_path = Path(temporary) / "chains.fasta"
    fasta_path.write_text(
        ">1ABC_1|Chains A [auth X], B [auth Y]|Protein\nacD\n", encoding="utf-8"
    )
    sequences, names = read_fasta(fasta_path)
    assert names == ["1ABC_1"]
    assert sequences == [
        FASTASequence(sequence="ACD", chain_count=2, chain_ids=["X", "Y"])
    ]
    label_sequences, _ = read_fasta(fasta_path, use_author_chain_ids=False)
    assert label_sequences[0].chain_ids == ["A", "B"]

    artifact_reference = Path(temporary) / "fasta-artifact"
    write_preprocessed_artifact(artifact_reference, {"fasta_sequences": sequences})
    restored = load_preprocessed_artifact(artifact_reference)
    assert restored["fasta_sequences"] == sequences
```

## Targets and metadata

| Module/API | Use |
| --- | --- |
| `cryoflake.build_component_extension_targets(structure_path, *, global_origin, voxel_size, grid_shape, ...)` | Build component targets from mmCIF, with optional polymer context and interface cutoff. |
| `cryoflake.validate_component_extension_targets(extension_targets)` | Validate and normalize schema, masks, indices, and component metadata. |
| `cryoflake.targets.full_atom.extract_full_atom_targets(target_source)` | Extract aligned target attributes from an application object and reorder protein rows before nucleotide rows. The object must already supply the full target arrays and topology; this is not a structure parser or model. |
| `cryoflake.validate_full_atom_targets(targets, num_protein=None, num_nucleotide=None)` | Validate full-atom targets and optional backbone row counts. |
| `cryoflake.validate_preprocessed_artifact_mapping(artifact_mapping, source=None)` | Check that the input implements `Mapping` and return the same object. This checks the input type only. |
| `cryoflake.get_preprocessed_artifact_grid(artifact_mapping, source=None)`, `get_preprocessed_artifact_voxel_size(artifact_mapping, source=None)`, `get_preprocessed_artifact_global_origin(artifact_mapping, source=None)` | Read and normalize individual geometry fields. These do not validate the complete training artifact. |
| `cryoflake.metadata` | Independent validators for grids, coordinate arrays, origin, voxel size, residue labels, and mapping inputs. |
| `cryoflake.constants` | Enums, molecular vocabularies, required field tuples, and target dimensions. |

The [data contracts](data-contracts.md) define shapes, enum values, coordinate
frames, valid masks, and the model-specific conventions the adapter must supply.
`validate_preprocessed_artifact_mapping()` does not check required keys, array
values, or storage integrity, and it raises `ValueError` for a non-mapping input.
Use `validate_preprocessed_artifact(artifact_reference)` for the complete stored training
contract. The mapping validator and geometry getters are available from both
`cryoflake` and `cryoflake.metadata`. They accept `artifact_mapping` as a keyword;
their optional `source` argument adds context to error messages.

## Manifests and splits

Use [`cryoflake.manifests.manifest`](../src/cryoflake/manifests/manifest.py) for
`ManifestEntry`, `build_manifest()`, `manifest_rows()`, `write_manifest_csv()`,
and `read_manifest_csv()`. `get_preprocessed_artifact_name(manifest_row)` normalizes
EMDB/wwPDB identifiers and returns an extensionless name such as
`emd_1234_1abc`, or `emd_1234` without a structure.
`read_source_associations(source_csv)` parses and normalizes source association
rows into `ManifestEntry` objects, including rows without a structure;
`build_manifest()` excludes those rows by default.

`cryoflake.read_artifact_list(artifact_list_path)` resolves relative local paths from the list
file and preserves S3 strings. `cryoflake.write_artifact_list(artifact_references, artifact_list_path)`
creates parent directories and writes one reference per line. Reading an empty
list raises `ValueError`. `split_artifact_references(artifact_references, validation_fraction=0.1,
seed=0)` deduplicates artifact aliases and returns `(train, valid)` lists; a
singleton stays in training.

For grouped splits, use `GroupedSplitRecord`, `split_grouped_records(records, *, split_strategy, validation_fraction, seed, holdout_group_ids=None)`,
`read_holdout_group_ids()`, and `write_grouped_split_report()` from
[`cryoflake.manifests.splits`](../src/cryoflake/manifests/splits.py).
Each record requires `artifact_reference` and `artifact_group_id`, with optional
`component_group_ids` and `interface_group_ids`.
`GroupedSplitResult.assignments` aligns with input records;
`train_artifact_references` and `validation_artifact_references` expose each partition. See the
[data contracts](data-contracts.md) for report fields and selected-group rules.
`read_holdout_group_ids(holdout_group_ids_path, expected_split_strategy=None)` checks that a report
matches the requested grouped split strategy when that argument is supplied.
`cryoflake.manifests.download.download_manifest_entries()` handles the
download phase independently and returns a mapping of recoverable path errors.

## Low-level operations

Use these when implementing custom storage workflows rather than complete
training pipelines. Detailed signatures and contracts are in the linked sources.

| Operation | Owning API/module |
| --- | --- |
| Direct local format reads | [`storage.reader.read_npz_json()`](../src/cryoflake/storage/reader.py) |
| Direct local writes or sequence-store removal | [`storage.writer.write_npz_json()`, `write_sequence_npz_json()`](../src/cryoflake/storage/writer.py) |
| Metadata schema inspection | [`storage.codec.read_metadata()`, `metadata_root_keys()`, `required_stores()`, `declared_store_files()`](../src/cryoflake/storage/codec.py) |
| Validate archives without reconstructing payload arrays | [`storage.validation.validate_npz_json_store_files()`](../src/cryoflake/storage/validation.py) |
| Serialize writes within one process | [`storage.locks.preprocessed_artifact_write_locks()`](../src/cryoflake/storage/locks.py) |
| Local/S3 object I/O and environment loading | [`storage.transports`](../src/cryoflake/storage/transports.py) |
| Bounded asynchronous staging | [`storage.upload_cache.BatchedArtifactUploadCache`](../src/cryoflake/storage/upload_cache.py) |
| File/JSON hashes and bounded task execution | [`common.sha256_file()`, `json_sha256()`, `run_tasks_with_progress()`](../src/cryoflake/common.py) |
| Density padding, Fourier rescaling, and shell statistics | [`io.mrc`](../src/cryoflake/io/mrc.py) |

`BatchedArtifactUploadCache.allocate_artifact_dir(row_index, artifact_name)`
reserves a local artifact directory path inside the managed cache.
`stage(local_artifact_dir, remote_artifact_reference)` queues its upload and
returns a future; closing the cache drains outstanding uploads and callbacks.

S3 environment loading is also available as
`cryoflake.storage.load_s3_environment(env_file=None)`. Use the
[configuration reference](configuration.md) for credential and endpoint
precedence. Process-local artifact locks do not provide distributed locking;
coordinate writers across processes and hosts externally.
