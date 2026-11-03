"""Per-cell QC of a SpatialData table."""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from ._utils import PX_UM

__all__ = ["compute_sbr"]

PX_MM = PX_UM / 1000  # mm per AtoMx global px


def compute_sbr(sdata, bandwidth=0.01, weight_cutoff=0.08, bg_quantile=0.01, sbr_min=0, sbr_max=None, table="table"):
    """Regional signal-to-background ratio (SBR) of each cell.

    Mean gene counts over mean negative-probe counts, each smoothed over the cell's neighbours with Gaussian weights.
    Same algorithm and defaults as ``computeSBR()`` in Barisano's CosMx QC (R).

    Parameters
    ----------
    sdata
        A SpatialData object whose table has ``CenterX/Y_global_px`` and probes named ``Negative*`` (negative
        probes) and ``SystemControl*`` (excluded).
    bandwidth
        Gaussian bandwidth, mm.
    weight_cutoff
        Neighbours with a smaller weight are ignored (sets the search radius).
    bg_quantile
        The smoothed background is floored at this quantile of its positive values: a near-zero background
        otherwise gives ratios of ~1e9.
    sbr_min, sbr_max
        Bounds on log2(SBR) for ``flag_sbr``; None disables a bound.
    table
        Table element name.

    Returns
    -------
    ndarray
        SBR per cell. ``obs`` gets ``SBR``, ``log2SBR`` and ``flag_sbr`` in place.
    """
    tb = sdata.tables[table]
    obs = tb.obs
    xy = obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float) * PX_MM

    # 0. Gaussian weights of the neighbours within the cutoff radius (and 1 for the cell itself), rows summing to 1
    tree = cKDTree(xy)
    pairs = tree.sparse_distance_matrix(tree, np.sqrt(-2 * bandwidth**2 * np.log(weight_cutoff)), output_type="ndarray")
    pairs = pairs[pairs["i"] != pairs["j"]]
    n = len(xy)
    conn = sp.csr_matrix((np.exp(-pairs["v"] ** 2 / (2 * bandwidth**2)), (pairs["i"], pairs["j"])), shape=(n, n))
    conn = conn + sp.identity(n, format="csr")
    conn = sp.diags(1 / np.asarray(conn.sum(axis=1)).ravel()) @ conn

    # 1. Smoothed mean counts of genes and negative probes; background floored at a low quantile
    probes = tb.var_names.astype(str)
    neg_probe = probes.str.startswith("Negative")
    gene = ~(neg_probe | probes.str.startswith("SystemControl"))
    X = sp.csr_matrix(tb.X)
    signal = conn @ np.asarray(X[:, gene].mean(axis=1)).ravel()
    neg = conn @ np.asarray(X[:, neg_probe].mean(axis=1)).ravel()
    sbr = signal / np.maximum(neg, np.quantile(neg[neg > 0], bg_quantile))

    # 2. Flag cells outside [sbr_min, sbr_max] on the log2 scale
    log2 = np.log2(sbr)
    flag = np.zeros(n, bool)
    if sbr_min is not None:
        flag |= log2 < sbr_min
    if sbr_max is not None:
        flag |= log2 > sbr_max
    obs["SBR"], obs["log2SBR"], obs["flag_sbr"] = sbr, log2, flag
    print(f"flag_sbr (log2SBR outside [{sbr_min}, {sbr_max}]): {flag.sum()} / {n} cells")
    return sbr
