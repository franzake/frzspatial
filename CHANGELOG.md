# Changelog

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
