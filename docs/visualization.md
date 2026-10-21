# Visualize an artifact

The optional visualization tools reconstruct an atomic structure, observed
polymer sequences, and a density map from a complete training artifact. You can
view them in MolStar in a browser or in a separately installed ChimeraX GUI.
The original structure and map files are not required.

## Install visualization support

Install the `vis` extra in your Python environment:

<!-- doc-example: visualization-install; mode=setup -->
```bash
uv pip install 'cryoflake[vis]'
cryoflake visualize --help
```

When working from this checkout, install the same extra with uv:

<!-- doc-example: visualization-checkout; mode=setup -->
```bash
uv sync --locked --dev --extra vis
uv run cryoflake visualize --help
```

The extra supplies the lightweight web server. ChimeraX itself is a standalone
desktop application and must be installed separately.

## Open MolStar

Run the following command with an artifact root or its `metadata.json`
reference:

<!-- doc-example: visualization-molstar; mode=cli -->
```bash
cryoflake visualize data/artifacts/example \
  --visualization-output-dir data/visualizations/example
```

MolStar is the default viewer. cryoFlake starts a server on `127.0.0.1`, chooses
an available port, and opens the viewer in your browser. Keep the command
running while you use the viewer; press Ctrl-C to stop the server. The browser
loads pinned MolStar assets from a CDN, so it needs internet access. Artifact
data is served locally and is not uploaded to the CDN.

Use the structure and sequence controls to inspect residues alongside the
density overlay. The page provides density visibility, contour, and opacity
controls, along with downloads of the reconstructed data.
The light/dark switch in the upper-right corner changes the page and viewer
appearance. The page starts with your system's color setting and follows system
changes until you choose a mode. Your choice lasts until you reload the page.
Browser download filenames include the artifact folder name as a prefix:
an artifact in `data/artifacts/example` saves as `example_structure.cif`,
`example_density.mrc`, `example_sequence.fasta`, and `example_summary.json`.
The sequence panel inserts spaces every five residues for readability.
Click a sequence to remove the grouping spaces and select all residues in a
read-only text box. Clicking elsewhere restores the five-residue grouping.

The density contour starts at 1 σ and opacity at 25%. Adjust either slider, or
click its displayed number to open a synchronized text box without increment
or decrement buttons. Valid text previews the change as you type; press Enter
or click away to finish. Press Escape to restore the value from before editing.
Invalid text keeps the editor open with an inline validation message and
preserves the last valid value.

Dragging the contour slider rounds and displays its value to three decimal
places in σ. The opacity slider uses whole percentages. Text entry keeps the
precision you type.

Contour levels are standard deviations from the map mean: the absolute contour
is `mean + level × standard deviation`. Any finite level is valid, including
fractional, zero, and negative levels; values beyond the slider's initial range
expand that range. Opacity accepts percentages from 0 to 100, including
fractional percentages.

To open the displayed URL yourself and use a specific port:

<!-- doc-example: visualization-manual-browser; mode=cli -->
```bash
cryoflake visualize data/artifacts/example/metadata.json \
  --visualization-output-dir data/visualizations/manual \
  --port 8765 --no-open-browser
```

S3 artifact references work through the same storage transport as other CLI
commands. Credentials follow the [user guide](user-guide.md#s3-storage).
For an S3-compatible service, supply its endpoint explicitly:

<!-- doc-example: visualization-s3; mode=cli -->
```bash
cryoflake visualize s3://my-bucket/artifacts/example \
  --visualization-output-dir data/visualizations/remote \
  --s3-endpoint-url https://s3.example.test --no-open-browser
```

## Open ChimeraX

Install the ChimeraX GUI, then choose it explicitly. cryoFlake searches for
`chimerax` or `ChimeraX` on your `PATH`, and checks the standard macOS application
location. Use `--chimerax-path` to select another installed executable:

<!-- doc-example: visualization-chimerax; mode=cli -->
```bash
cryoflake visualize data/artifacts/example --viewer chimerax \
  --chimerax-path /opt/ChimeraX/bin/ChimeraX \
  --visualization-output-dir data/visualizations/chimerax
```

The generated ChimeraX session opens the structure, sequence viewer, and density
map together. The density is shown as a surface with 45% transparency. The command
waits for ChimeraX to exit.

## Understand the reconstruction

The export directory contains `structure.cif`, `density.mrc`, `sequence.fasta`,
and `summary.json`. The map preserves the artifact's density grid, voxel size,
and global origin so that structure and density share the same coordinates.
The summary records what was reconstructed and any limitations.
Choose a fresh export directory; existing reconstructed assets are not overwritten.

By default, the structure contains the named protein backbone atoms
(`CA`, `C`, `N`, `O`), nucleotide `P` atoms, and available named component atoms.
The artifact's 65 polymer atom slots do not define a universal atom-name order.
To reconstruct those slots, provide the atom names used by your preprocessing
adapter as a JSON list of 65 strings, or one such list per polymer node in the
artifact's CA-then-P order. Only atoms with a true `atom_mask` are exported;
their names must be supplied. Do not infer the slot order from this example:

<!-- doc-example: visualization-atom-names; mode=python -->
```python
import json
from pathlib import Path

# For a two-node artifact: one protein residue followed by one nucleotide.
# Every other slot is masked out. Use your own adapter's naming convention.
atom_names = [[""] * 65 for _ in range(2)]
atom_names[0][:4] = ["N", "CA", "C", "O"]
atom_names[1][0] = "P"
Path("data/atom_names.json").write_text(json.dumps(atom_names), encoding="utf-8")
```

<!-- doc-example: visualization-full-atom; mode=cli -->
```bash
cryoflake visualize data/artifacts/example \
  --atom-names-path data/atom_names.json \
  --visualization-output-dir data/visualizations/full-atom \
  --no-open-browser
```

FASTA contains exported observed polymer residues, grouped into synthetic chain
names derived from the stored `chain_index` values when available.
Sequence embedding arrays are not amino-acid or nucleotide sequences. Missing
residues, unresolved atoms, original author chain names, and bonds absent from
the artifact cannot be recovered.

If you omit `--visualization-output-dir`, cryoFlake uses a temporary directory
and removes it when the command exits. Specify a dedicated export directory
to retain the reconstructed data. This directory is separate from the artifact
and does not change the stored artifact.
