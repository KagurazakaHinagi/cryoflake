# Offline EMD-11736 / 7AE1 standard test target

These assets come from the deposited EMD-11736 map and its fitted 7AE1 model
(human RNA polymerase III elongation complex, reported resolution 2.8 Å).
CI reads these local assets and never downloads inputs, imports NextAngelo,
runs a predictor, or loads model weights.

The complete source records, hashes, selection indices, density operation,
independent named-coordinate samples, and known upstream limitation are in
[provenance.json](provenance.json). Public sources are
[EMDB EMD-11736](https://www.ebi.ac.uk/emdb/EMD-11736) and
[PDB 7AE1](https://www.rcsb.org/structure/7AE1). The original download URLs and
compressed checksums are retained in the provenance.

## Compact target used by ordinary tests

The subset contains **145 protein residues, all 44 nucleotide residues, and all
nine molecular components**: seven ZN ions, one MG ion, and one eight-atom SF4
cluster. There are 2,001 observed canonical atoms and 16 component atoms, or
2,017 deposited atom records across 11 author chains.

Selection starts with all nucleotide rows and every canonical residue with any
observed atom within 5.0 Å of a component atom. It adds each seed's immediately
preceding and following observed source row in the same chain, once, without
recursive expansion. All atom records for the selected residues are retained.
Original atom names, coordinates, B-factors, occupancies, author chain/residue
identities, atom IDs, and source row order are preserved. Chain/residue numbers
are not compressed or invented. The CIF retains original metadata categories;
its `atom_site` category contains the selected complete residues/components.

The full-atom arrays and shared 65-name table are cached outputs from the
existing NextAngelo 1.0.17 structure parser at the commit recorded in the
provenance. Geometry arrays are selected from the complete-entry calculation;
they are not recomputed using the discontinuous subset as a new chain.
Values undergo only cryoFlake's documented numeric normalization. No fabricated
slot names, zero-filled replacement frames, or synthetic coordinates are used.
Sequence embeddings are absent. Tests of embedding behavior explicitly add
their own small numerical inputs.

The density is a **derived 64 × 64 × 64 volume**, not the deposited-resolution
map. Generation applies `cryoflake.io.mrc.rescale_real(deposited_grid, 64)`,
multiplies by `(64 / 336)**3` to preserve Fourier DC/mean amplitude, and stores
float32 voxels. Origin remains `(0, 0, 0)` Å. Spacing changes from
`1.0499999523162842` Å to `5.512499749660492` Å, preserving the original
`352.7999839782715` Å extent along each axis. It exercises storage and world/grid
geometry; it does not establish predictor accuracy on a coarse map.

## Upstream torsion-mask limitation

The original NextAngelo parser constructs protein pre-omega/phi torsions using
the preceding source row without checking chain identity. For the full entry,
all 16 later protein chain starts retain those masks across distinct chains;
their preceding C-to-current N distances range from 37.57 to 143.64 Å.
The compact selection includes **none of those 16 source chain starts**.
All cached targets, including the full-entry reference, remain unchanged.
Passing artifact/coordinate checks is not a certification that every upstream
torsion is biochemically correct. Source anchors and the exact affected rows
are recorded in `provenance.json`.

## Assets and regeneration

- `subset.cif.gz`: compact structure with unchanged deposited atom records.
- `standard_arrays.npz`: derived density, complete compact training mapping,
  and original full-entry row/identity metadata; numeric/string arrays only.
- `7ae1.cif.gz`: unchanged original downloaded full-entry structure.
- `full_entry_targets.npz`: unchanged authentic extraction bridge for one
  complete-entry alignment/storage regression; no model checkpoint or weights.

Binary assets occupy 5,433,098 bytes in total, below 6 MB. The original 336³
map is not bundled. The shared helper is `tests/standard_target.py`; it returns
fresh mutable mappings and can write either the compact or full-entry CIF.
Optional `component_ids` filtering keeps all real compact polymer rows while
selecting only genuine ZN, MG, or SF4 components for split scenarios.

To regenerate, first make the original decompressed map/CIF, their downloaded
gzip inputs, the authentic extraction bridge, and the recorded investigation
JSON available in a local source directory. Run
`uv run python tests/fixtures/emd_11736_7ae1/regenerate.py SOURCE_DIRECTORY`.
The utility performs no downloads or NextAngelo/model execution. Changes to
the resulting hashes, scientific values, or selection must be reviewed and
verified by `tests/test_standard_target.py`.
