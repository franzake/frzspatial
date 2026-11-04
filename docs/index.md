# frzspatial

CosMx (AtoMx) exports to [SpatialData](https://spatialdata.scverse.org), for TMA slides: fast loading and caching,
cores stitched into one image, plots, and hand annotation of regions in napari that ends up as a cell-level column.

- **Read** an AtoMx flat-file export (counts, metadata, polygons, transcripts, per-FOV composites and masks) into one
  SpatialData object in AtoMx global pixels: in parallel, with transcripts cached as Parquet and images read lazily.
- **Reopen** saved slides in seconds (`load_slides`), building only the missing ones, with a progress bar.
- **Stitch** a TMA core, or a whole slide at reduced resolution, into one composite and one mask layer for napari
  (`build_core`).
- **Plot** a slide overview or one core with its outlines, transcripts or any obs column (`frzspatial.pl`).
- **Annotate** regions (glomeruli, arteries, ...) or single cells in napari; polygons are saved as GeoJSON after
  every change and each cell gets the class of the polygon around its centroid (`annotate_core`,
  `annotations_to_obs`).
- **QC**: regional signal-to-background ratio per cell (`compute_sbr`).
- **Register** a whole-slide H&E onto the cells from tissue shape alone (`frzspatial.he`).
- **Export** count matrices and metadata as plain files for a Seurat object in R (`write_seurat_inputs`).

## Installation

```bash
pip install "git+https://github.com/franzake/frzspatial"
```

This installs everything: plotting (spatialdata-plot), napari with PyQt5, and H&E registration (scikit-image).

From R with reticulate:

```r
reticulate::py_require("frzspatial @ git+https://github.com/franzake/frzspatial")
```

It installs from wheels only (nothing to compile) on Linux, Windows and macOS 14+, with Python 3.12 to 3.14:

- **Intel Macs**: numba and llvmlite are kept at 0.62 / 0.45, the last versions with Intel Mac wheels. Use Python
  3.12 or 3.13 there: numba has no Intel Mac wheels for Python 3.14.
- **napari** stays < 0.10, the last versions that work with PyQt5. On ARM Linux, where PyQt5 has no wheels, napari is
  installed without a Qt backend (`pip install pyqt6` to open windows).

Tested with Python 3.12, spatialdata 0.8.0, spatialdata-plot 0.4.3, napari 0.9.2 and napari-spatialdata 0.7.2.

## Quick start

```python
import frzspatial as fs

# Each slide's export is in data/<slide>_inputs/ (flatFiles/, CellComposite/, CellLabels/)
slides = fs.load_slides(["MN2", "MN3"], input_dir="data", zarr_dir="data/zarr", cache_dir="data/cache")
sdata = slides["MN2"]
```

The first call builds and saves `data/zarr/MN2_flat.zarr` (minutes); later calls only open it (seconds).

Cores need an obs column with each cell's core name (`core_str` by default) and, for annotations, its slide name
(`slide`):

```python
obs = sdata.tables["table"].obs
obs["slide"] = "MN2"
obs["core_str"] = ...  # e.g. from your TMA key

# Plots
import matplotlib.pyplot as plt
from frzspatial.pl import plot_core, plot_slide_overview

plot_slide_overview(sdata, "core_str"); plt.show()
plot_core(sdata, "C4R7", genes={"NPHS2": "magenta", "LRP2": "orange"}); plt.show()

# napari: one core at full resolution, or the whole slide (core=None) at 1/8
from frzspatial.viewer import open_napari
open_napari(fs.build_core(sdata, "MN2", "C4R7"))
```

### Annotating regions

```python
from frzspatial.viewer import annotate_core
from frzspatial.pl import annotation_palette, plot_core

CLASSES = ["Glomerulus", "Sclerotic_glomerulus", "Tubulointerstitium", "Artery", "Inflammation", "Exclude"]
annotate_core(sdata, "MN2", "C4R7", CLASSES)     # opens napari; returns when the window is closed

obs = sdata.tables["table"].obs
obs["manual_annotation"].value_counts()
plot_core(sdata, "C4R7", color="manual_annotation", palette=annotation_palette(CLASSES, obs["manual_annotation"]))

fs.save_table(sdata)                             # store the new column in MN2_flat.zarr
```

In napari, click the `annot: <class>` layer of a class, then draw:

| To do                  | How                                                                 |
|------------------------|---------------------------------------------------------------------|
| draw a region          | **P** (polygon), click around it, double-click to close             |
| mark single cells      | zoom in, **R** (rectangle), drag a small box over the cell's centre |
| move or delete a shape | **5** (select), click it, then drag or press Backspace              |
| colour the outlines    | select `cell_boundaries`, pick a column in the *View* panel         |

Each change is saved to `results/annotations/<slide>_<core>.geojson`. Close the window with its own close button.
When Python runs inside R or Jupyter, **don't quit with Cmd-Q**: it quits the host's Qt loop and can hang it, and
force-quitting the window kills the host. Rerunning `annotate_core` on the same core reopens its shapes for
editing; `fs.annotations_to_obs(sdata, CLASSES)` rebuilds the column from the files at any time.

### Merged slides

After `spatialdata.concatenate` with each slide's elements suffixed `-<slide>` and in coordinate system
`global-<slide>`, the same functions take the merged object and a slide name:

```python
open_napari(fs.build_core(merged, "MN2", "C4R7", colors={"cell_type": {"Podocyte": "#e377c2", ...}}))
fs.annotations_to_obs(merged, CLASSES)  # every slide's files, matched on obs["slide"] and obs["core_str"]
```

## Command line

```bash
frzspatial build data/MN2_inputs MN2 -o MN2_flat.zarr --cache-dir cache
frzspatial info MN2_flat.zarr
frzspatial view MN2_flat.zarr --slide MN2 --core C4R7
frzspatial view MN2_flat.zarr --slide MN2 --core C4R7 --annotate Glomerulus Artery
```

`frzspatial <command> --help` lists every option.

## Help

Every function has a docstring: `help(fs.build_core)`, or `fs.build_core?` in IPython. The module docstrings
(`help(frzspatial.io)`, `help(frzspatial.core)`) describe the expected export layout and coordinate systems. The
full reference is at <https://franzake.github.io/frzspatial>, or build it locally with
`pip install "frzspatial[docs]" && mkdocs serve`.

## Conventions

- Positions are AtoMx global px (0.120280945 µm/px, `fs.PX_UM`); each FOV is placed with
  `x = dx + col`, `y = cy - row`, so FOVs stitch without resampling.
- Cells are named by AtoMx's `cell` (`c_<slide_ID>_<fov>_<cell_ID>`).
- `fs.set_verbose(False)` silences the progress messages.

## Development

```bash
git clone https://github.com/franzake/frzspatial && cd frzspatial
pip install -e ".[test,docs]"
pytest
```

The tests build a small synthetic AtoMx export, so they need no data.

## License

MIT
