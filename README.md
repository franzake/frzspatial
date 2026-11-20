# frzspatial

CosMx (AtoMx) exports to [SpatialData](https://spatialdata.scverse.org), for TMA slides: fast loading and caching,
TMA cores and case metadata, per-cell QC, R tools run from Python, and plots. The functions take a single slide or
several slides merged into one object (with `slide=`), and change it in place.

- **Read** an AtoMx flat-file export (counts, metadata, polygons, transcripts, per-FOV composites and masks) into one
  SpatialData object in AtoMx global pixels: in parallel, with transcripts cached as Parquet and images read lazily.
  `load_slides` reopens saved slides in seconds and builds only the missing ones; `merge_slides` puts them in one
  object; `drop_cells` removes flagged cells.
- **TMA cores**: `key_matrix`, `assign_cores`, `plot_core_grid`, and case metadata (`cases`, `add_clinical`).
- **QC**: `fov_integrity`, `compute_sbr` (regional signal-to-background ratio), `qc_flags`, `plot_qc`,
  `plot_qc_flags`.
- **R tools** (`frzspatial.rtools`): InSituType, FastReseg inputs and results, scPearsonPCA, HieraType markers and
  heatmap, SpNeigh structures. Reference profiles from an atlas (`frzspatial.reference`).
- **DeSpotX** decontamination input and result (`despotx_input`, `apply_despotx`).
- **Plots** (`frzspatial.pl`): one core with its outlines, transcripts or any obs column (`plot_core`), UMAPs
  (`plot_umap`), cell-type composition (`plot_composition`).

## Installation

```bash
pip install "git+https://github.com/franzake/frzspatial"
```

From R with reticulate:

```r
reticulate::py_require("frzspatial @ git+https://github.com/franzake/frzspatial")
```

It installs from wheels only (nothing to compile) on Linux, Windows and macOS 14+, with Python 3.12 to 3.14. On Intel
Macs, numba and llvmlite are kept at 0.62 / 0.45, the last versions with Intel Mac wheels: use Python 3.12 or 3.13
there. The R tools need R with Matrix, data.table and the tool's package.

## Quick start

```python
import frzspatial as fs
import matplotlib.pyplot as plt
import pandas as pd
from frzspatial.pl import plot_core

# Each slide's export is in data/<slide>_inputs/ (flatFiles/, CellComposite/, CellLabels/)
slides = fs.load_slides(["MN2", "MN3"], input_dir="data", zarr_dir="data/zarr", cache_dir="data/cache")
for s, sdata in slides.items():
    sdata.tables["table"].obs["slide"] = s
usc = fs.merge_slides(slides, drop=None)   # elements suffixed -<slide>, coordinate systems global-<slide>
```

The first call builds and saves `data/zarr/MN2_flat.zarr` (minutes); later calls only open it (seconds).

Cores come from the TMA key (rows and columns oriented like the slide), then QC and case metadata, per slide:

```python
key = fs.key_matrix(pd.read_csv("MN2_key.csv", sep=";"))   # key[r - 1, c - 1] = sample of core C<c>R<r>
fs.assign_cores(usc, key, slide="MN2")                     # obs core_str, sample_id (in place)
fs.plot_core_grid(sdata=usc, slide="MN2", by="core"); plt.show()
fs.fov_integrity(usc, "Hs_6k", slide="MN2")                # obs flag_fov_integrity, fov_signal_loss
fs.compute_sbr(usc, slide="MN2")                           # obs SBR, flag_sbr
fs.qc_flags(usc, slide="MN2")                              # obs flag_*, flag_overall
fs.drop_cells(usc, "flag_overall")

plot_core(usc, "C4R7", slide="MN2", genes={"NPHS2": "magenta", "LRP2": "orange"}); plt.show()
plot_core(usc, "C4R7", slide="MN2", color="sample_id", image=False, legend_outside=True); plt.show()
```

`plot_core` also takes a Series as `color` (e.g. a column masked with `.where(...)`; the other cells are grey).

## Help

Every function has a docstring: `help(fs.assign_cores)`, or `fs.assign_cores?` in IPython. The module docstrings
(`help(frzspatial.io)`, `help(frzspatial.core)`) describe the expected export layout and coordinate systems. The
full reference is at <https://franzake.github.io/frzspatial>, or build it locally with
`pip install "frzspatial[docs]" && mkdocs serve`.

## Conventions

- Positions are AtoMx global px (0.120280945 µm/px, `fs.PX_UM`); each FOV is placed with
  `x = dx + col`, `y = cy - row`, so FOVs stitch without resampling.
- Cells are named by AtoMx's `cell` (`c_<slide_ID>_<fov>_<cell_ID>`).

## Development

```bash
git clone https://github.com/franzake/frzspatial && cd frzspatial
pip install -e ".[test,docs]"
pytest
```

The tests build a small synthetic AtoMx export, so they need no data.

## License

MIT
