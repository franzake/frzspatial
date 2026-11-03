"""Export for R/Seurat without conversion packages.

:func:`write_seurat_inputs` writes plain files that R reads directly. A matching R reader::

    read_csc = function(dir, assay, genes, cells) {
      i = readBin(file.path(dir, paste0(assay, "_i.bin")), "integer", n = 2^31 - 1)
      p = readBin(file.path(dir, paste0(assay, "_p.bin")), "integer", n = length(cells) + 1)
      x = readBin(file.path(dir, paste0(assay, "_x.bin")), "double", n = length(i))
      Matrix::sparseMatrix(i = i, p = p, x = x, index1 = FALSE, dims = c(length(genes), length(cells)),
                           dimnames = list(genes, cells))
    }
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["write_seurat_inputs"]


def write_seurat_inputs(out_dir, counts, genes, cells, obs, xy_px, umap=None):
    """Write the inputs of a Seurat object as plain files.

    Parameters
    ----------
    out_dir
        Output folder (created).
    counts
        ``{assay: cells × genes sparse matrix}``; the first is the object's main assay.
    genes, cells
        Gene and cell names.
    obs
        Cell metadata, cells as rows; categoricals are written as text (NA as "").
    xy_px
        (cells, 2) positions, e.g. AtoMx global px.
    umap
        (cells, 2) embedding, or None.

    Returns
    -------
    Path
        `out_dir`, holding ``<assay>_i.bin``, ``<assay>_p.bin``, ``<assay>_x.bin`` (the genes × cells matrix as CSC
        components: int32, int32, float64), ``assays.txt``, ``genes.txt``, ``cells.txt``, ``metadata.csv``,
        ``xy_px.csv`` and ``umap.csv``.

    Raises
    ------
    ValueError
        If a matrix has more non-zeros than an R dgCMatrix holds (2^31 - 1).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for assay, m in counts.items():
        m = sp.csc_matrix(m.T)  # genes x cells, column = cell
        m.sum_duplicates()
        if m.nnz >= 2**31 - 1:
            raise ValueError(f"{assay}: {m.nnz:,} non-zero counts, more than an R dgCMatrix holds")
        m.indices.astype(np.int32).tofile(out / f"{assay}_i.bin")
        m.indptr.astype(np.int32).tofile(out / f"{assay}_p.bin")
        m.data.astype(np.float64).tofile(out / f"{assay}_x.bin")
    (out / "assays.txt").write_text("\n".join(counts) + "\n")
    (out / "genes.txt").write_text("\n".join(map(str, genes)) + "\n")
    (out / "cells.txt").write_text("\n".join(map(str, cells)) + "\n")
    md = obs.copy()
    for c in md.columns[[isinstance(d, pd.CategoricalDtype) for d in md.dtypes]]:
        md[c] = md[c].astype(str).where(md[c].notna(), "")
    md.index = list(cells)
    md.to_csv(out / "metadata.csv", index_label="cell_id")
    pd.DataFrame(np.asarray(xy_px, float), index=list(cells), columns=["x", "y"]).to_csv(
        out / "xy_px.csv", index_label="cell_id")
    if umap is not None:
        pd.DataFrame(np.asarray(umap, float), index=list(cells), columns=["umap_1", "umap_2"]).to_csv(
            out / "umap.csv", index_label="cell_id")
    return out
