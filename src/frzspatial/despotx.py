"""DeSpotX decontamination (run on a GPU cluster): its input file and the decontaminated object."""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from ._utils import PX_UM, _log

__all__ = ["despotx_input", "apply_despotx"]


def _ctrl(names):
    names = pd.Index(names).astype(str)
    return names.str.startswith("Negative") | names.str.startswith("SystemControl")


def despotx_input(sdata, out_dir, label="ist_lineage", merge=None, min_cells=20, slide_gap_mm=100, table="table"):
    """Write DeSpotX's input, ``<out_dir>/despotx_in.h5ad``, as main.Rmd §6.1.

    Raw counts of the genes (no control probes), each cell's label (``obs["dx_celltype"]``) and its position in mm
    (``obsm["spatial"]``), slides side by side `slide_gap_mm` apart so that cells of different slides are never
    neighbours.

    Parameters
    ----------
    sdata
        The merged object (obs ``slide``, ``obsm["spatial"]`` in global px).
    label
        obs column of the cell types DeSpotX models contamination for.
    merge
        ``{small label: label it joins}`` for labels with fewer than `min_cells` cells.

    Returns
    -------
    Series
        Cells per label. Raises ValueError if a label still has fewer than `min_cells` cells.
    """
    import anndata as ad

    tb = sdata.tables[table]
    labels = tb.obs[label].astype(str).replace(merge or {})
    small = labels.value_counts()[lambda n: n < min_cells]
    if not small.empty:
        raise ValueError(f"labels with < {min_cells} cells, add them to merge: {small.to_dict()}")
    slides = list(tb.obs["slide"].cat.categories)
    idx = tb.obs["slide"].astype(str).map({s: i for i, s in enumerate(slides)})
    if idx.isna().any():
        raise ValueError("cells without a known slide")
    xy = np.asarray(tb.obsm["spatial"], float) * PX_UM / 1000
    xy[:, 0] += idx.to_numpy() * slide_gap_mm
    keep = ~_ctrl(tb.var_names)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Plain (object) string names: the cluster's anndata (< 0.11) cannot write the nullable strings a newer one reads
    # them back as, and DeSpotX failed saving adjusted.h5ad at the end of the 2026-11-05 run
    names = lambda ix: pd.Index(np.asarray(ix, dtype=object), dtype=object)
    ad.AnnData(X=sp.csr_matrix(tb.X)[:, keep].astype("int32"),
               obs=pd.DataFrame({"dx_celltype": labels.to_numpy()}, index=names(tb.obs_names)),
               var=pd.DataFrame(index=names(tb.var_names[keep])), obsm={"spatial": xy}).write_h5ad(out / "despotx_in.h5ad")
    _log(f"wrote {out / 'despotx_in.h5ad'}: {tb.n_obs} cells, {int(keep.sum())} genes")
    return labels.value_counts()


def apply_despotx(sdata, out_dir, table="table"):
    """A copy of the merged object with DeSpotX's corrected counts, as main.Rmd §6.3.

    Reads ``<out_dir>/despotx_out`` (``adjusted.mtx``, ``cells.tsv``, ``genes.tsv``, ``contamination.tsv``).

    Returns
    -------
    SpatialData
        Same images, masks, transcripts and polygons; its table has the corrected genes (control probes unchanged)
        in ``X``, the original counts in ``layers["counts"]`` and each cell's contamination in
        ``obs["despotx_eps"]``.
    """
    out = Path(out_dir) / "despotx_out"
    adj = sp.csr_matrix(scipy.io.mmread(out / "adjusted.mtx").T)
    cells = pd.read_csv(out / "cells.tsv", header=None)[0].astype(str)
    genes = pd.read_csv(out / "genes.tsv", header=None)[0].astype(str)
    eps = pd.read_csv(out / "contamination.tsv", sep="\t").set_index("cell_id")["contamination"]
    eps.index = eps.index.astype(str)

    tb = sdata.tables[table].copy()
    tb.layers["counts"] = tb.X.copy()
    rows = pd.Index(cells).get_indexer(tb.obs_names)
    if (rows < 0).any():
        raise ValueError(f"{(rows < 0).sum()} cells of the table are missing from DeSpotX's output")
    ctrl = _ctrl(tb.var_names)
    X = sp.hstack([adj[rows], sp.csr_matrix(tb.X)[:, ctrl]]).tocsr()
    tb.X = X[:, pd.Index(list(genes) + list(tb.var_names[ctrl])).get_indexer(tb.var_names)]
    tb.obs["despotx_eps"] = eps.reindex(tb.obs_names).to_numpy()
    new = sdata.subset([name for _, name, _ in sdata.gen_elements()])
    new.tables[table] = tb
    _log(f"DeSpotX: {tb.n_obs} cells, median contamination {np.nanmedian(tb.obs['despotx_eps']):.3f}")
    return new
