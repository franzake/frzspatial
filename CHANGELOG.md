# Changelog

## 0.1.0 (2026-11-13)

First release.

- I/O: AtoMx flat files to SpatialData (`build_sdata`, `load_slides`, `save_table`), stitched cores and slides
  (`build_core`), `slide_with_cells` (later results, e.g. niches, on a single slide), Seurat export
  (`write_seurat_inputs`) and the `frzspatial` command.
- TMA (`frzspatial.tma`): `key_matrix`, `assign_cores` (gbspatial's `assign_fovs_to_cores()`, with manual FOV fixes),
  `plot_core_grid` (orientation check), `cases` and `add_clinical`.
- QC (`frzspatial.qc`): per-cell SBR (`compute_sbr`), `fov_integrity` (gbspatial's `runFOVQC()`), `panel_barcodes`,
  `qc_flags` (Barisano's `QCfiltering()` flags and summary), `plot_qc` (`QCoverview()` panels), `plot_qc_flags`
  (UpSet plot of the flags) and `merge_slides` (the slides without their flagged cells, one coordinate system per
  slide). Identical to the R versions on four kidney TMA slides (cores, samples and flagged FOVs cell for cell; signal
  loss within 1e-15).
- R tools (`frzspatial.rtools`), run from Python through files and `Rscript` (works from Jupyter, Quarto and
  reticulate): `insitutype` (supervised InSituType), `fastreseg_inputs` / `load_fastreseg` (FastReseg's cluster inputs
  and results, cached as Parquet + npz), `apply_reseg`, `pearson_pca` (scPearsonPCA batch-corrected PCA),
  `cluster_markers` and `marker_heatmap` (HieraType), `structure_zones` (multicellular structures, e.g. glomeruli,
  and two rings around them; SpNeigh) and `decontx` (ambient RNA removal per batch; celda).
- Decontamination (`frzspatial.despotx`): `despotx_input` and `apply_despotx`. Niches (`frzspatial.niches`) and
  reference profiles (`frzspatial.reference`).
- Plots (`frzspatial.pl`): slide overviews and cores (take a merged object's `slide`; more than 102 categories get
  distinct colours), `plot_annotations`, `plot_umap`, `plot_composition`. napari viewing and annotation
  (`frzspatial.viewer`), GeoJSON annotations to obs (`annotations_to_obs`), H&E registration (`frzspatial.he`).
- Install: everything by default (`pip install frzspatial`; `frzspatial[all]` still works). Intel Macs get
  numba < 0.63 and llvmlite < 0.46 (the newest with Intel Mac wheels); ARM Linux gets napari without PyQt5 (no wheels
  there); napari < 0.10, the last versions that work with PyQt5. Dependencies include squidpy, igraph (Leiden) and
  adjustText.
