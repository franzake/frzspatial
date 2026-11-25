# Changelog

## Unreleased

- `frzspatial.smide` (`frzspatial.de`): smiDE spatial differential expression on any SpatialData table, any cell-type
  column and any design. Per target cell type and gene, an NB2 count mixed model of the raw counts with the cell's
  total count as offset, the formula's fixed effects and a random intercept, and the RankNorm-transformed expression
  of the same gene in neighbouring cells of other types (radius or k nearest, within blocks); Wald contrasts, BH per
  family counting failed fits. Defaults reproduce the CHARU MN glomerular smiDE analysis (V. Charu, 2026-10). The
  default engine is pure Python (Laplace-approximated ML as in glmmTMB, `frzspatial._nbglmm`), parallel with
  checkpoints; `engine="glmmtmb"` runs glmmTMB in R (`R/smide.R`). New dependencies: formulaic, joblib,
  threadpoolctl. `pl.plot_smide_volcano`.
- `smide`: NanoString smiDE's contamination screen (`overlap_ratio_metric`) on the same neighbour graph: columns
  `avg_celltype`, `avg_neighbor_other`, `overlap_ratio` (mean expression in the type's other-type neighbours / mean in
  the type; above 1, a change may be spillover), and `max_overlap_ratio=` to test only genes below it: on by
  default at 1.5 (smiDE's example cut-off of 1 drops about half of a CosMx 6K panel: broadly expressed genes sit near
  1); `None` turns it off. `de.overlap_ratio` matches smiDE's algorithm exactly (tests).
- `smide(cells=...)`: restrict the tested cells (e.g. a tissue niche) while the neighbour covariate still counts
  every cell.
- `pl.plot_smide_heatmap(res)`: top genes x all families (log2FC, FDR dots, clustered genes).
  `pl.plot_smide_patients(sdata, res, celltype_col, contrast, group_col)`: per cell type, patients' pseudobulk of the
  contrast's top genes as robust z-scores, patients clustered within each group with their trees. `cells=` (here and
  in `smide`) also takes any obs column for its non-missing cells, e.g. `"ng_glom_id"`.
  Both heatmaps: the same blue - white - red palette with saturated ends, a border around every cell
  (`cell_borders`, `border_color`), one `fontsize` for all text; `plot_smide_heatmap` clips automatically
  (`clip="auto"`), stars FDR < 0.05 and gives each gene's log2FC in its label.
- Figure helpers: `pl.plot_group_proportions` (per-patient cell-type shares by group, Kruskal-Wallis + BH-adjusted
  pairwise Wilcoxon brackets), `pl.plot_group_values` (one value per patient by group, same tests),
  `pl.plot_radial_position` (ridges of radial position, e.g. `ng_dist` in glomeruli).
- `frzspatial.enrich`: preranked GSEA of DE results per family (`smide_gsea`, ranked on smide's z of every tested
  gene: the right universe for a targeted panel, no FDR cut-off), Enrichr libraries cached as JSON (`gene_sets`:
  MSigDB Hallmark, GO Biological Process 2025, Reactome 2024 by default); `pl.plot_gsea_dotplot` (gene sets x
  families, NES colour, FDR size) and `pl.plot_gsea_bars` (one family's top sets per library). New dependency: gseapy.
- Kept to the functions of a CosMx TMA analysis from flat files to cell types and glomeruli. Removed: stitched cores
  (`build_core`), napari viewing and annotation (`frzspatial.viewer`, `frzspatial.annotate`), niches (`frzspatial.niches`), the Seurat export (`frzspatial.export`), the `frzspatial` command,
  `save_table`, `slide_with_cells`, `decontx`, `plot_boundary`, `set_verbose`, and the plots `plot_slide_overview`,
  `plot_annotations`, `annotation_palette`, `plot_niche_composition`, `plot_share_by_group`. napari is no longer a
  dependency. They are in 0.1.0 and on the `main-sp` branch's first commit.
- `frzspatial.he.align_he_cores`: the post-CosMx H&E as one aligned image per core (slide affine fitted on core
  landmarks with spatialdata's `get_transformation_between_landmarks`, then each core's shift by scikit-image's
  `phase_cross_correlation`); `plot_core` draws it with `image="he_"`. scikit-image is a dependency again.
- Functions that work per slide take a merged object and `slide=` and change it in place; `assign_cores` stores each
  slide's FOV table in `uns["tma_fovs"]`; `drop_cells`; `plot_core` takes a Series as `color`, `legend_outside`,
  `legend_fontsize`.

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
