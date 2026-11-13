"""Reference profiles for supervised cell typing (InSituType) from a single-cell atlas."""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._utils import _log

__all__ = ["reference_profiles", "probe_genes"]


def probe_genes(probe, genes):
    """The atlas genes a panel probe measures: ``"IGHG1/2"`` -> IGHG1 + IGHG2, ``"LILRB3/A6"`` -> LILRB3 + LILRA6,
    ``"IFNA4/10/17"`` -> IFNA4 + IFNA10 + IFNA17. Each part after the first replaces the end of the first gene name;
    parts not found in `genes` are dropped."""
    genes = set(genes)
    parts = probe.split("/")
    first = parts[0]
    out = [first] if first in genes else []
    for p in parts[1:]:
        for k in range(1, len(first)):
            cand = first[:-k] + p
            if cand in genes:
                out.append(cand)
                break
    return out


def reference_profiles(atlas, label="cell_type", panel_genes=None, min_cells=20, gene_symbols="feature_name",
                       layer=None, scale=1e4):
    """Mean expression of each cell type of a single-cell atlas, on the panel's genes: InSituType reference profiles.

    Each cell's counts are scaled to `scale` total, then averaged per cell type (as CellProfileLibrary's profiles).
    Combined panel probes (e.g. ``IGHG1/2``) get the sum of their genes (:func:`probe_genes`).

    Parameters
    ----------
    atlas
        AnnData or the path of an ``.h5ad`` (e.g. a CELLxGENE download; read backed, one cell type at a time).
    label
        obs column of the cell types.
    panel_genes
        Gene (probe) names of the panel; None keeps every atlas gene.
    min_cells
        Cell types with fewer cells are left out.
    gene_symbols
        var column with gene symbols (CELLxGENE: ``feature_name``; var names are Ensembl IDs); None uses var names.
    layer
        Raw counts: a layer name, ``"raw"`` for ``atlas.raw.X``, or None to use ``raw.X`` when present, else ``X``.

    Returns
    -------
    DataFrame
        Genes x cell types.
    """
    import anndata as ad

    a = ad.read_h5ad(atlas, backed="r") if isinstance(atlas, (str, bytes)) or hasattr(atlas, "__fspath__") else atlas
    symbols = pd.Index(a.var[gene_symbols].astype(str) if gene_symbols else a.var_names.astype(str))
    use_raw = layer == "raw" or (layer is None and a.raw is not None)
    labels = a.obs[label].astype(str)
    counts = labels.value_counts()
    types = sorted(counts[counts >= min_cells].index)
    _log(f"atlas: {a.n_obs:,} cells, {len(types)} cell types with >= {min_cells} cells"
         + (f" (left out: {', '.join(sorted(counts[counts < min_cells].index))})" if (counts < min_cells).any() else ""))
    prof = np.zeros((len(symbols), len(types)))
    for j, t in enumerate(types):
        idx = np.flatnonzero((labels == t).to_numpy())
        m = a.raw[idx].X if use_raw else (a[idx].layers[layer] if layer else a[idx].X)
        m = sp.csr_matrix(m, dtype=np.float64)
        tot = np.asarray(m.sum(1)).ravel()
        prof[:, j] = np.asarray((sp.diags(scale / np.maximum(tot, 1)) @ m).mean(0)).ravel()
    prof = pd.DataFrame(prof, index=symbols, columns=types).groupby(level=0).sum()  # duplicate symbols summed
    if panel_genes is None:
        return prof
    rows, missing = {}, []
    for g in panel_genes:
        hit = probe_genes(g, prof.index) if "/" in g else ([g] if g in prof.index else [])
        if hit:
            rows[g] = prof.loc[hit].sum()
        else:
            missing.append(g)
    _log(f"profiles: {len(rows)} of {len(panel_genes)} panel genes found in the atlas")
    return pd.DataFrame(rows).T
