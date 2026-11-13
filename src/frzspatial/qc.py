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

__all__ = ["compute_sbr", "fov_integrity", "panel_barcodes", "qc_flags", "plot_qc", "plot_qc_flags"]

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


_QC_FILTERS = {  # flag column: (name in the summary, rule)
    "flag_nCount": ("RNA counts", "nCount_RNA"),
    "flag_nFeature": ("Detected genes", "nFeature_RNA"),
    "flag_area": ("Cell area", "Area"),
    "flag_fov_integrity": ("FOV integrity", "signal loss"),
    "flag_boundary": ("FOV boundary", "log2 split ratio"),
    "flag_sbr": ("Regional SBR", "log2 SBR"),
}


def _outside(v, lo, hi):
    """v < lo or v > hi (None: no bound); NaN is never flagged."""
    v = np.asarray(v, float)
    with np.errstate(invalid="ignore"):
        flag = np.zeros(len(v), bool)
        if lo is not None:
            flag |= v < lo
        if hi is not None:
            flag |= v > hi
    return flag & ~np.isnan(v)


def _rule(lo, hi):
    parts = ([f"< {lo:.4g}"] if lo is not None else []) + ([f"> {hi:.4g}"] if hi is not None else [])
    return " or ".join(parts) or "-"


def qc_flags(sdata, count_min=30, count_max=None, feature_min=None, feature_max=None, area_min=None, area_max=80e3,
             split_ratio_min=np.log2(0.5), split_ratio_max=np.log2(4), sbr_min=0, sbr_max=None, table="table",
             label=""):
    """Flag cells failing the QC thresholds, as ``QCfiltering()`` of Barisano's CosMx QC (R), without removing them.

    Each bound is optional (None: none). The FOV-integrity flag comes from :func:`fov_integrity`; the boundary flag
    applies to FOV-border cells only (``SplitRatioToLocal`` > 0: area over the FOV's mean area, from AtoMx), on the
    log2 scale; the SBR flag uses ``log2SBR`` (:func:`compute_sbr`).

    Parameters
    ----------
    sdata
        A single slide, after :func:`fov_integrity` and :func:`compute_sbr`.
    count_min, count_max, feature_min, feature_max, area_min, area_max
        Bounds on ``nCount_RNA``, ``nFeature_RNA`` and ``Area`` (px).
    split_ratio_min, split_ratio_max, sbr_min, sbr_max
        Bounds on log2(``SplitRatioToLocal``) of border cells and on ``log2SBR``.
    table
        Table element.
    label
        Prefix of the printed summary (e.g. the slide name).

    Returns
    -------
    DataFrame
        Cells flagged per filter and overall. Sets obs ``flag_nCount``, ``flag_nFeature``, ``flag_area``,
        ``flag_boundary``, ``flag_sbr`` (and ``flag_density``, ``flag_extranuclear``: False, as in R) and
        ``flag_overall`` in place.
    """
    obs = sdata.tables[table].obs
    sr = obs["SplitRatioToLocal"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_sr = np.where(sr > 0, np.log2(sr), np.nan)
    obs["flag_nCount"] = _outside(obs["nCount_RNA"], count_min, count_max)
    obs["flag_nFeature"] = _outside(obs["nFeature_RNA"], feature_min, feature_max)
    obs["flag_area"] = _outside(obs["Area"], area_min, area_max)
    obs["flag_boundary"] = _outside(log2_sr, split_ratio_min, split_ratio_max)
    obs["flag_sbr"] = _outside(obs["log2SBR"], sbr_min, sbr_max)
    obs["flag_fov_integrity"] = obs["flag_fov_integrity"].astype(bool)
    obs["flag_density"] = obs["flag_extranuclear"] = False
    obs["flag_overall"] = obs[list(_QC_FILTERS)].any(axis=1)

    rules = [_rule(count_min, count_max), _rule(feature_min, feature_max), _rule(area_min, area_max),
             "flagged FOV", _rule(split_ratio_min, split_ratio_max), _rule(sbr_min, sbr_max)]
    summary = pd.DataFrame({
        "filter": [n for n, _ in _QC_FILTERS.values()] + ["Overall"],
        "rule": [f"{v} {r}" for (_, v), r in zip(_QC_FILTERS.values(), rules)] + ["any filter"],
        "flagged": [int(obs[c].sum()) for c in _QC_FILTERS] + [int(obs["flag_overall"].sum())]})
    summary["pct"] = (100 * summary["flagged"] / len(obs)).round(2)
    prefix = f"{label}: " if label else ""
    tqdm.write(f"{prefix}{int(obs['flag_overall'].sum())} / {len(obs)} cells flagged")
    return summary


def plot_qc(sdata, count_min=None, count_max=None, feature_min=None, feature_max=None, area_min=None, area_max=None,
            sbr_min=None, sbr_max=None, split_ratio_min=None, split_ratio_max=None, title="", bins=40, color="0.4",
            table="table"):
    """The QC panels of ``QCoverview()``: counts, genes, area, log2 SBR and log2 split ratio (border cells), with
    red lines at the thresholds.

    Returns
    -------
    Figure
        4 x 2 panels; call ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    obs = sdata.tables[table].obs
    d = pd.DataFrame({"nCounts": obs["nCount_RNA"].to_numpy(float), "nGenes": obs["nFeature_RNA"].to_numpy(float),
                      "area": obs["Area"].to_numpy(float), "log2SBR": obs["log2SBR"].to_numpy(float),
                      "split": obs["SplitRatioToLocal"].to_numpy(float)})
    border = d[d["split"] > 0].assign(log2split=lambda b: np.log2(b["split"]))
    fig, axes = plt.subplots(4, 2, figsize=(8, 10))

    def lines(ax, xs=(), ys=()):
        for v in xs:
            if v is not None:
                ax.axvline(v, c="red", lw=0.8)
        for v in ys:
            if v is not None:
                ax.axhline(v, c="red", lw=0.8)

    def hist(ax, v, t, xs):
        ax.hist(v[np.isfinite(v)], bins=bins, color=color)
        ax.set(title=t, ylabel="nCells")
        lines(ax, xs)

    def scatter(ax, x, y, t, xs, ys):
        ax.scatter(x, y, s=0.4, c=color, lw=0, rasterized=True)
        ax.set(title=t)
        lines(ax, xs, ys)

    c, f, a, s, r = (count_min, count_max), (feature_min, feature_max), (area_min, area_max), (sbr_min, sbr_max), \
        (split_ratio_min, split_ratio_max)
    hist(axes[0, 0], d["nCounts"], "nCounts x nCells", c)
    scatter(axes[0, 1], d["nCounts"], d["nGenes"], "nCount_RNA x nFeature_RNA", c, f)
    hist(axes[1, 0], d["area"], "cellArea x nCells", a)
    scatter(axes[1, 1], d["area"], d["nCounts"], "cellArea x nCounts", a, c)
    hist(axes[2, 0], d["log2SBR"], "log2SBR x nCells", s)
    scatter(axes[2, 1], d["log2SBR"], d["nCounts"], "log2SBR x nCounts", s, c)
    hist(axes[3, 0], border["log2split"], "log2(SplitRatio) x nCells (border cells)", r)
    scatter(axes[3, 1], border["log2split"], border["nCounts"], "log2(SplitRatio) x nCounts (border cells)", r, c)
    for ax in axes.flat:
        ax.title.set_fontsize(9)
        ax.tick_params(labelsize=7)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_qc_flags(sdata, title="", n_intersections=20, table="table"):
    """Cells flagged by each QC filter and by each combination of filters (UpSet plot), with the kept / flagged
    share, as ``QCfiltering()``'s plot. Needs :func:`qc_flags`.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    obs = sdata.tables[table].obs
    cols = [c for c in _QC_FILTERS if obs[c].any()]
    names = [_QC_FILTERS[c][0] for c in cols]
    flags = obs.loc[obs["flag_overall"].astype(bool), cols].astype(bool)
    combos = flags.value_counts().head(n_intersections)
    n, k = len(combos), len(cols)
    x = np.arange(n)

    # 0. Layout: pie | intersection sizes (top); set sizes | dots with the filter names on their left (bottom)
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 0.38], width_ratios=[0.22, 0.78], hspace=0.04, wspace=0.28)
    pie, bars, sizes, dots = (fig.add_subplot(gs[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)])

    # 1. Intersection sizes
    bars.bar(x, combos.to_numpy(), color="0.25", width=0.6)
    for i, v in enumerate(combos.to_numpy()):
        bars.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    bars.set(xticks=[], xlim=(-0.6, n - 0.4), ylabel="Cells")
    bars.set_title("Flagged cells per combination of filters", fontsize=10, loc="left")

    # 2. Dot matrix, one shaded band per filter
    for j in range(k):
        if j % 2 == 0:
            dots.axhspan(j - 0.5, j + 0.5, color="0.95", zorder=0)
        dots.scatter(x, [j] * n, c="0.85", s=36, zorder=1)
    for i, combo in enumerate(combos.index):
        on = [j for j, v in enumerate(combo) if v]
        if len(on) > 1:
            dots.plot([i, i], [min(on), max(on)], c="0.15", lw=1.5, zorder=2)
        dots.scatter([i] * len(on), on, c="0.15", s=36, zorder=3)
    dots.set(xticks=[], xlim=bars.get_xlim(), ylim=(-0.5, k - 0.5))
    dots.set_yticks(range(k), names, fontsize=9)
    dots.tick_params(axis="y", length=0, pad=6)

    # 3. Cells per filter (any combination), growing to the left
    totals = [int(obs[c].sum()) for c in cols]
    sizes.barh(range(k), totals, color="0.25", height=0.6)
    for j, v in enumerate(totals):
        sizes.text(v, j, f"{v:,} ", ha="right", va="center", fontsize=7, color="0.2")
    sizes.set(yticks=[], ylim=dots.get_ylim(), xlim=(0, max(totals) * 1.3), xlabel="Cells per filter")
    sizes.invert_xaxis()

    # 4. Kept / flagged share, counts in the legend
    n_flag = int(obs["flag_overall"].sum())
    wedges, _ = pie.pie([n_flag, len(obs) - n_flag], colors=["#D73027", "0.88"], startangle=90, counterclock=False,
                        wedgeprops={"linewidth": 1, "edgecolor": "white"})
    pie.legend(wedges, [f"Flagged {n_flag:,} ({100 * n_flag / len(obs):.1f}%)",
                        f"Kept {len(obs) - n_flag:,} ({100 * (1 - n_flag / len(obs)):.1f}%)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=8)
    pie.set_aspect("equal")

    for ax in (bars, dots, sizes):
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible((ax is bars and side == "left") or (ax is sizes and side == "bottom"))
    bars.tick_params(labelsize=8)
    sizes.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.02, ha="left")
    return fig
