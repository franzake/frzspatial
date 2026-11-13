# Changelog

## Unreleased

- `frzspatial.rtools.structure_zones`: multicellular structures (e.g. glomeruli from their tuft cells) and the
  cells inside and in two rings around them (R package SpNeigh, convex outlines).
- `frzspatial.rtools.decontx`: ambient RNA removal by decontX (R package celda) with given clusters, per batch;
  returns the decontaminated counts and each cell's contaminated fraction.

## 0.3.0 (2026-11-03)

- QC: `qc_flags` (Barisano's `QCfiltering()` flags and summary), `plot_qc` (`QCoverview()` panels) and
  `plot_qc_flags` (UpSet plot of the flags).
- `merge_slides`: the slides without their flagged cells, one coordinate system per slide.
- `frzspatial.rtools`: R tools run from Python through files and `Rscript` (works from Jupyter, Quarto and
  reticulate): `insitutype` (supervised InSituType), `fastreseg_inputs` / `load_fastreseg` (FastReseg's cluster
  inputs and results), `apply_reseg` (the resegmented object), `pearson_pca` (scPearsonPCA batch-corrected PCA),
  `cluster_markers` and `marker_heatmap` (HieraType).
- `frzspatial.despotx`: `despotx_input` and `apply_despotx` (DeSpotX decontamination).
- Plots: `plot_umap`, `plot_composition`; `plot_core` and `plot_slide_overview` take a merged object's `slide`;
  overviews with more than 102 categories get distinct colours (spatialdata-plot drew them grey).
- `load_fastreseg` caches the merged results (Parquet + npz). Dependencies: igraph (Leiden), adjustText.

## 0.2.0 (2026-11-03)

- `frzspatial.tma`: `key_matrix`, `assign_cores` (gbspatial's `assign_fovs_to_cores()`, with manual FOV fixes),
  `plot_core_grid` (orientation check), `cases` and `add_clinical`.
- `slide_with_cells` (later results, e.g. niches, on a single slide) and `frzspatial.pl.plot_annotations`.
- `fov_integrity` (gbspatial's `runFOVQC()`) and `panel_barcodes` (the panels' reporter barcodes) in `frzspatial.qc`.
- Both identical to the R versions on four kidney TMA slides (cores, samples and flagged FOVs cell for cell; signal
  loss within 1e-15).

## 0.1.2 (2026-11-03)

- Intel Macs: numba < 0.63 and llvmlite < 0.46 (the newest with Intel Mac wheels) instead of fixed versions; numpy
  follows numba's own limit.
- ARM Linux: napari without PyQt5 (no wheels there), so the install no longer fails.
- Install notes per platform in the README.

## 0.1.1 (2026-11-03)

- Intel Macs: numba 0.61.2, llvmlite 0.44.0 and numpy < 2.3 are pinned automatically (newer llvmlite has no wheels).
- Everything is installed by default (plots, napari with PyQt5, H&E): `pip install frzspatial`, no extras needed
  (`frzspatial[all]` still works).
- napari < 0.10, the last versions that work with PyQt5.

## 0.1.0 (2026-11-03)

First release: AtoMx flat files to SpatialData (`build_sdata`, `load_slides`, `save_table`), stitched cores and
slides (`build_core`), plots (`frzspatial.pl`), napari viewing and annotation (`frzspatial.viewer`), GeoJSON
annotations to obs (`annotations_to_obs`), H&E registration (`frzspatial.he`), Seurat export
(`write_seurat_inputs`), per-cell SBR (`compute_sbr`) and the `frzspatial` command.
