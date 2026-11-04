"""Per-cell QC of a SpatialData table."""

import warnings
from importlib.resources import files

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from ._utils import PX_UM

__all__ = ["compute_sbr", "fov_integrity", "panel_barcodes"]

PX_MM = PX_UM / 1000  # mm per AtoMx global px


def compute_sbr(sdata, bandwidth=0.01, weight_cutoff=0.08, bg_quantile=0.01, sbr_min=0, sbr_max=None, table="table",
                label=""):
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
    label
        Prefix of the printed summary (e.g. the slide name).

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
    prefix = f"{label}: " if label else ""
    tqdm.write(f"{prefix}flag_sbr (log2SBR outside [{sbr_min}, {sbr_max}]): {flag.sum()} / {n} cells")
    return sbr


def panel_barcodes(panel="Hs_6k"):
    """Reporter barcodes of a CosMx panel (``gene``, ``barcode``), as shipped with gbspatial.

    Panels: ``Hs_6k``, ``Hs_IO``, ``Hs_UCC``, ``Hs_WTX``, ``Mm_Neuro``, ``Mm_UCC``. Each barcode has two characters
    per reporter cycle; the second is the colour read in that cycle (``.`` = none).
    """
    d = pd.read_csv(files("frzspatial").joinpath("data/barcodes_by_panel.csv.gz"), compression="gzip")
    if panel not in set(d["panel"]):
        raise ValueError(f"unknown panel {panel!r}; choose from {sorted(set(d['panel']))}")
    return d.loc[d["panel"] == panel, ["gene", "barcode"]].reset_index(drop=True)


def _cut(x, n):
    """Bin of each value among n equal intervals over its range, as R's cut(x, n) (right-closed, ends widened)."""
    lo, hi = x.min(), x.max()
    dx = hi - lo
    if dx == 0:
        dx = abs(lo) if lo != 0 else 1
        breaks = lo - dx / 1000 + np.arange(n + 1) * ((hi - lo + 2 * dx / 1000) / n)
    else:
        breaks = lo + np.arange(n + 1) * (dx / n)
        breaks[0], breaks[-1] = lo - dx / 1000, hi + dx / 1000
    return np.searchsorted(breaks, x, side="left") - 1


def fov_integrity(sdata, panel="Hs_6k", max_prop_loss=0.6, max_totalcounts_loss=0.6, slide="", table="table"):
    """Flag FOVs that lost signal: in total counts, or in one reporter cycle of the barcodes.

    The FOV QC of gbspatial's ``runFOVQC()`` (after NanoString's CosMx FOV QC): each FOV is cut into a 7 x 7 grid
    of squares (squares with fewer than 10 cells are skipped), and each square is compared with the 10 most similar
    squares of other FOVs (nearest neighbours in mean counts per barcode bit). A FOV is flagged

    - for total counts when > 75% of its squares have lost more than `max_totalcounts_loss` of the counts of
      their neighbours;
    - for bias when, for half of a reporter cycle's colours, its squares have lost signal in that bit
      (mean log2 ratio below log2(1 - `max_prop_loss`), p < 0.01, in at least half of its squares).

    Parameters
    ----------
    sdata
        A single slide: probes named ``Negative*`` / ``SystemControl*`` are excluded.
    panel
        Panel of the barcodes (:func:`panel_barcodes`).
    max_prop_loss, max_totalcounts_loss
        Signal-loss limits, as fractions (0.6 = 60% lost).
    slide
        Slide name: FOV ids are ``<slide><fov>``, as in gbspatial.
    table
        Table element.

    Returns
    -------
    dict
        ``flagged`` (FOV ids), ``flagged_totalcounts``, ``flagged_bias``, ``totalcounts_resid`` (log2 ratio per
        square) and ``bias`` / ``p`` (FOVs x bits). Sets obs ``flag_fov_integrity``, ``flag_fov_totalcounts``,
        ``flag_fov_bias``, ``fov_signal_loss`` (log2 ratio of the cell's square) and ``fov_signal_loss_cat`` in place.
    """
    if not 0 <= max_prop_loss <= 1:
        raise ValueError("max_prop_loss must be in [0, 1]")
    tb = sdata.tables[table]
    obs = tb.obs
    probes = tb.var_names.astype(str)
    gene = ~(probes.str.startswith("Negative") | probes.str.startswith("SystemControl"))
    X = sp.csr_matrix(tb.X)[:, gene].astype(float)
    genes = probes[gene]
    xy = obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float) * PX_MM
    xy -= xy.min(axis=0)
    fovid = slide + obs["fov"].astype(int).astype(str).to_numpy()

    # 0. 7 x 7 squares per FOV; squares with < 10 cells are left out
    square = np.full(len(obs), -1)
    square_fov = []
    for f in pd.unique(fovid):
        i = np.flatnonzero(fovid == f)
        local = _cut(xy[i, 0], 7) * 7 + _cut(xy[i, 1], 7)
        codes, inv = np.unique(local, return_inverse=True)
        square[i] = len(square_fov) + inv
        square_fov += [f] * len(codes)
    square_fov = np.array(square_fov)
    size = np.bincount(square[square >= 0], minlength=len(square_fov))
    small = size < 10
    keep_sq = np.flatnonzero(~small)
    renum = np.full(len(square_fov), -1)
    renum[keep_sq] = np.arange(len(keep_sq))
    square = np.where(square >= 0, renum[square], -1)
    square_fov = square_fov[keep_sq]
    n_sq = len(square_fov)

    # 1. Mean counts per barcode bit in each square, rows scaled to the same total
    bc = panel_barcodes(panel)
    bc = bc[bc["gene"].isin(set(genes))]
    chars = "".join(bc["barcode"])
    colours = [c for c in dict.fromkeys(chars) if c != "."]
    n_cycles = len(bc["barcode"].iat[0]) // 2
    bits = [f"reportercycle{i + 1}{c}" for i in range(n_cycles) for c in colours]
    B = np.zeros((len(bc), len(bits)))
    for i in range(n_cycles):
        here = bc["barcode"].str[2 * i + 1].to_numpy()
        for k, c in enumerate(colours):
            B[here == c, i * len(colours) + k] = 1
    in_sq = square >= 0
    M = sp.csr_matrix((np.ones(in_sq.sum()), (square[in_sq], np.flatnonzero(in_sq))), shape=(n_sq, len(obs)))
    M = sp.diags(1 / np.asarray(M.sum(axis=1)).ravel()) @ M
    col = pd.Index(genes).get_indexer(bc["gene"])
    bitcounts = np.asarray((M @ X[:, col]) @ B)
    tot = np.maximum(bitcounts.sum(axis=1), 1)
    bitcounts = bitcounts / tot[:, None] * tot.mean()

    # 2. The 10 most similar squares of other FOVs (one per FOV), among the 500 nearest
    from sklearn.neighbors import NearestNeighbors

    k = min(10 * 50, n_sq)
    top = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(bitcounts).kneighbors(bitcounts,
                                                                                       return_distance=False)
    comp = np.full((n_sq, 10), -1)
    for i in range(n_sq):
        f = square_fov[top[i]]
        _, first = np.unique(f, return_index=True)
        first = np.sort(first)
        first = first[f[first] != square_fov[i]][:10]
        comp[i, :len(first)] = top[i, first]
    have = comp >= 0

    def neighbour_mean(v):
        vals = np.where(have[..., None] if v.ndim == 2 else have, v[comp], np.nan)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # squares without comparators: NaN
            return np.nanmean(vals, axis=1)

    # 3. Total counts of each square against its neighbours
    total = np.asarray(X.sum(axis=1)).ravel()
    sq_total = np.bincount(square[in_sq], weights=total[in_sq], minlength=n_sq) / np.bincount(square[in_sq],
                                                                                              minlength=n_sq)
    tc_resid = np.log2(sq_total / neighbour_mean(sq_total))
    lost = np.where(np.isnan(tc_resid), np.nan, tc_resid < np.log2(1 - max_totalcounts_loss))
    lost_per_fov = pd.Series(lost).groupby(square_fov, sort=False).agg(lambda v: v.mean(skipna=False))
    flagged_tc = list(lost_per_fov.index[lost_per_fov.to_numpy() > 0.75])

    # 4. Bias per FOV and bit: mean log2 ratio of its squares (one-way linear model, as R's lm), p and agreement
    resid = np.log2((bitcounts + 1) / (neighbour_mean(bitcounts) + 1))
    fovs = pd.unique(square_fov)
    g = pd.Index(fovs).get_indexer(square_fov)
    bias = np.full((len(fovs), len(bits)), np.nan)
    p = bias.copy()
    agree = bias.copy()
    for b in range(len(bits)):
        r = resid[:, b]
        ok = ~np.isnan(r)
        n_f = np.bincount(g[ok], minlength=len(fovs))
        mean_f = np.bincount(g[ok], weights=r[ok], minlength=len(fovs)) / np.where(n_f > 0, n_f, np.nan)
        df = ok.sum() - (n_f > 0).sum()
        sigma = np.sqrt(((r[ok] - mean_f[g[ok]]) ** 2).sum() / df) if df > 0 else np.nan
        t = mean_f / (sigma / np.sqrt(n_f))
        bias[:, b], p[:, b] = mean_f, 2 * stats.t.sf(np.abs(t), df) if df > 0 else np.nan
        below = pd.Series(np.where(ok, r < np.log2(1 - max_prop_loss) / 3, np.nan))
        agree[:, b] = below.groupby(g).agg(lambda v: v.mean(skipna=False)).reindex(range(len(fovs))).to_numpy()
    with np.errstate(invalid="ignore"):
        flag = (bias < np.log2(1 - max_prop_loss)) & (p < 0.01) & (agree >= 0.5)
    flag = np.where(np.isnan(bias) | np.isnan(p) | np.isnan(agree), np.nan, flag)
    per_cycle = flag.reshape(len(fovs), n_cycles, len(colours)).mean(axis=2)
    with np.errstate(invalid="ignore"):
        flagged_bias = list(fovs[(per_cycle >= 0.5).any(axis=1)])
    flagged = flagged_tc + [f for f in flagged_bias if f not in flagged_tc]

    # 5. Per cell
    loss = np.where(in_sq, tc_resid[np.maximum(square, 0)], np.nan)
    obs["flag_fov_integrity"] = np.isin(fovid, flagged)
    obs["flag_fov_totalcounts"] = np.isin(fovid, flagged_tc)
    obs["flag_fov_bias"] = np.isin(fovid, flagged_bias)
    obs["fov_signal_loss"] = loss
    obs["fov_signal_loss_cat"] = pd.cut(loss, [-np.inf, -2, -1, 0, 1, 2, np.inf],
                                        labels=["<-2", "-2:-1", "-1:0", "0:1", "1:2", ">2"])
    prefix = f"{slide}: " if slide else ""
    tqdm.write(f"{prefix}FOVs flagged: {len(flagged)} / {len(pd.unique(fovid))}"
               + (f" (total counts: {', '.join(flagged_tc) or '-'}; bias: {', '.join(flagged_bias) or '-'})"
                  if flagged else ""))
    return {"flagged": flagged, "flagged_totalcounts": flagged_tc, "flagged_bias": flagged_bias,
            "totalcounts_resid": pd.Series(tc_resid, index=[f"{f}_{i}" for i, f in enumerate(square_fov)]),
            "bias": pd.DataFrame(bias, index=fovs, columns=bits), "p": pd.DataFrame(p, index=fovs, columns=bits)}
