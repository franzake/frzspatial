"""Spatial differential expression: smiDE (count mixed models with a neighbour-expression covariate).

``smide`` tests, for every gene and every target cell type, a term of a count model fitted to the raw counts of the
cells of that type, adjusted for the expression of the same gene in neighbouring cells of other types. That neighbour
covariate absorbs transcripts that segmentation assigns to the wrong cell and the local tissue context, so the tested
effect is closer to what the target cells themselves express. It works on any SpatialData table (or AnnData) with raw
counts, a cell-type column, cell coordinates and the columns of the formula.

For target cell type T, gene g and cell i of type T (one family = one cell type x one contrast):

    y_gi | b ~ NB2(mu_gi, theta_g)
    log mu_gi = log(U_i) + x_i' beta_g + gamma_g Z_gi + b_g,group(i),   b ~ N(0, tau_g^2)
    N_gi = sum over neighbours j of i (other types, within radius_um, same block) of 1e4 * y_gj / U_j
    Z_gi = RankNorm(N_gi) over the cells of the family: qnorm((rank - 0.375) / (n + 0.25)), average ties

U is the cell's total count over the whole panel (before any gene filter, control probes excluded). A gene is tested
when it is detected (count > 0) in more than `min_detection` of the cells of at least one compared group. Inference is
a Wald test on the contrast; BH is applied within each family, counting failed fits in the family size (their p is NA),
plus a secondary BH over all cell types within each contrast. The defaults are those of the glomerular smiDE analysis
of the CHARU MN study (V. Charu, 2026-10): 30 um, sum of other-type expression, RankNorm, union 10 % detection,
patient random intercept, pairwise refits.

Engines: ``"python"`` (default, installed with frzspatial) fits NB2 with at most one random-intercept factor by the
same Laplace-approximated maximum likelihood as glmmTMB (``frzspatial._nbglmm``); ``"glmmtmb"`` runs glmmTMB itself in
R (``frzspatial/R/smide.R``) and takes any lme4-style random-effect terms; it needs R with glmmTMB, Matrix and
data.table (``check_r()``).
"""

import hashlib
import os
import pickle
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import ndtri
from scipy.stats import norm, rankdata

from ._utils import PX_UM, _log

__all__ = ["smide", "neighbor_graph", "neighbor_expression", "ranknorm", "check_r"]

_PROBE_PREFIXES = ("Negative", "SystemControl", "FalseCode", "NegPrb")
_ENGINE_VERSION = "smide-1"


def _engine_hash():
    """Hash of the model engine's source: checkpoints of an edited engine are never reused."""
    from . import _nbglmm

    return hashlib.sha1(Path(_nbglmm.__file__).read_bytes()).hexdigest()[:12]


# --------------------------------------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------------------------------------
def ranknorm(x):
    """Rank-based inverse-normal transform (Blom): ``qnorm((rank(x, average ties) - 0.375) / (n + 0.25))``."""
    x = np.asarray(x, float)
    return ndtri((rankdata(x, method="average") - 0.375) / (x.size + 0.25))


def neighbor_graph(xy, blocks=None, radius_um=30.0, n_neighbors=None):
    """Spatial neighbours as a sparse 0/1 matrix (cells x cells, no self loops).

    Parameters
    ----------
    xy
        Coordinates in µm (n, 2).
    blocks
        Labels (n,): cells are neighbours only within the same block (e.g. slide, or slide + sample + core). None: one
        block.
    radius_um
        All cells within this distance (used when `n_neighbors` is None).
    n_neighbors
        The k nearest cells instead of a radius.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, float)
    n = len(xy)
    codes = np.zeros(n, np.intp) if blocks is None else pd.factorize(pd.Series(blocks).astype(str))[0]
    rows, cols = [], []
    for b in np.unique(codes):
        idx = np.flatnonzero(codes == b)
        if idx.size < 2:
            continue
        tree = cKDTree(xy[idx])
        if n_neighbors:
            k = min(int(n_neighbors) + 1, idx.size)
            _, nn = tree.query(xy[idx], k=k)
            r = np.repeat(np.arange(idx.size), k)
            c = nn.ravel()
            keep = r != c
            rows.append(idx[r[keep]])
            cols.append(idx[c[keep]])
        else:
            pairs = tree.query_pairs(radius_um, output_type="ndarray")
            if len(pairs):
                rows += [idx[pairs[:, 0]], idx[pairs[:, 1]]]
                cols += [idx[pairs[:, 1]], idx[pairs[:, 0]]]
    if not rows:
        return sp.csr_matrix((n, n))
    r, c = np.concatenate(rows), np.concatenate(cols)
    A = sp.csr_matrix((np.ones(r.size), (r, c)), shape=(n, n))
    A.data[:] = 1.0
    return A


def neighbor_expression(A, counts, size, target, allowed, scale=1e4, agg="sum"):
    """Expression of each gene in the allowed neighbours of each target cell (sparse, targets x genes).

    ``N[i, g] = sum_j A[i, j] * allowed[j] * scale * counts[j, g] / size[j]`` (``agg="mean"``: divided by the number of
    allowed neighbours; cells without any are 0).
    """
    w = np.where(allowed, scale / np.where(size > 0, size, np.inf), 0.0)
    At = A[target]
    N = (At @ sp.diags(w)) @ counts   # CSR or CSC as given: no copy of the whole matrix
    if agg == "mean":
        k = np.asarray(At @ allowed.astype(float)).ravel()
        N = sp.diags(np.where(k > 0, 1.0 / np.maximum(k, 1), 0.0)) @ N
    elif agg != "sum":
        raise ValueError(f"neighbor_agg must be 'sum' or 'mean', not {agg!r}")
    return sp.csc_matrix(N)


def _check_integer_counts(data, chunk=10_000_000):
    """Raise unless all values are non-negative integers (in chunks: no full-size temporary arrays)."""
    for i in range(0, data.size, chunk):
        d = data[i:i + chunk]
        if d.size and (d.min() < 0 or np.any(d != np.round(d))):
            raise ValueError("counts must be raw non-negative integers (pass the raw-count `layer`)")


def _bh(p, n):
    """Benjamini-Hochberg q-values with family size n >= number of p (NaN p stay NaN)."""
    p = np.asarray(p, float)
    q = np.full(p.size, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not ok.size:
        return q
    o = ok[np.argsort(p[ok])]
    ranks = np.arange(1, o.size + 1)
    v = np.minimum.accumulate((p[o] * n / ranks)[::-1])[::-1]
    q[o] = np.minimum(v, 1.0)
    return q


def _parse_formula(formula):
    """(fixed-effect right-hand side, [(re lhs, re group), ...]) of a ``~ a + b + (1|g)`` formula."""
    rhs = formula.split("~", 1)[-1]
    re_terms = [(a.strip(), b.strip()) for a, b in re.findall(r"\(\s*([^|()]+?)\s*\|\s*([^()]+?)\s*\)", rhs)]
    fixed = re.sub(r"\(\s*[^|()]+?\|[^()]+?\)", "", rhs)
    fixed = re.sub(r"\+\s*(\+\s*)+", "+ ", fixed).strip().strip("+").strip() or "1"
    return fixed, re_terms


def _formula_vars(fixed, obs):
    """obs columns named in the fixed-effect formula."""
    names = set(re.findall(r"[A-Za-z_.][A-Za-z0-9_.]*", fixed))
    return [c for c in obs.columns if c in names]


def _first_var(fixed, obs):
    for tok in re.split(r"[+*:]", fixed):
        tok = tok.strip()
        for c in obs.columns:
            if re.search(rf"(^|[^A-Za-z0-9_.]){re.escape(c)}([^A-Za-z0-9_.]|$)", tok):
                return c
    raise ValueError(f"no obs column found in the formula {fixed!r}")


def _normalise_contrasts(contrasts, fixed, obs):
    """{name: (var, case, ref)} or {name: (design column,)}."""
    if contrasts is None:
        var = _first_var(fixed, obs)
        col = obs[var]
        if pd.api.types.is_numeric_dtype(col) and not isinstance(col.dtype, pd.CategoricalDtype):
            return {var: (var,)}
        levels = list(col.cat.categories) if isinstance(col.dtype, pd.CategoricalDtype) else sorted(col.dropna().unique())
        levels = [lv for lv in levels if (col == lv).any()]
        return {f"{lv}_vs_{levels[0]}": (var, lv, levels[0]) for lv in levels[1:]}
    out = {}
    for name, spec in (contrasts.items() if isinstance(contrasts, dict) else ((None, c) for c in contrasts)):
        spec = tuple(spec) if isinstance(spec, (list, tuple)) else (spec,)
        if len(spec) == 3:
            name = name or f"{spec[1]}_vs_{spec[2]}"
        elif len(spec) == 1:
            name = name or str(spec[0])
        else:
            raise ValueError(f"contrast {spec!r}: give (variable, case, reference) or (design column,)")
        out[name] = spec
    return out


def _design(fixed, data):
    """Fixed-effect design matrix (formulaic), all-zero columns dropped."""
    import formulaic

    mm = formulaic.model_matrix(fixed, data, na_action="raise")
    X = np.asarray(mm, float)
    cols = list(mm.columns)
    keep = np.any(X != 0, axis=0)
    return X[:, keep], [c for c, k in zip(cols, keep) if k]


def _level_col(cols, var, level):
    for c in (f"{var}[T.{level}]", f"{var}[{level}]"):
        if c in cols:
            return cols.index(c)
    return None


# --------------------------------------------------------------------------------------------------------------------
# fitting one batch of genes (runs in the worker processes)
# --------------------------------------------------------------------------------------------------------------------
def _gene_rows(fit, gene, L_list, z_included):
    rows = []
    beta, V = fit["beta"], fit["vcov"]
    p_user = len(L_list[0][1]) if L_list else 0
    for cname, L in L_list:
        Lf = np.zeros(beta.size)
        Lf[:p_user] = L
        est = float(Lf @ beta) if np.all(np.isfinite(beta)) else np.nan
        var = float(Lf @ V @ Lf) if np.all(np.isfinite(V)) else np.nan
        se = np.sqrt(var) if var > 0 else np.nan
        z = est / se if np.isfinite(se) else np.nan
        p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
        rows.append(dict(gene=gene, contrast=cname, beta=est, SE=se, z=z, p=p,
                         neighbor_beta=float(beta[-1]) if z_included else np.nan,
                         neighbor_SE=float(np.sqrt(V[-1, -1])) if z_included and np.isfinite(V[-1, -1]) else np.nan,
                         re_sd=fit["re_sd"], dispersion=fit["theta"], loglik=fit["loglik"],
                         converged=fit["converged"], pdHess=fit["pdHess"], max_grad=fit["max_grad"],
                         valid=fit["valid"], optimizer=fit["optimizer"], n_attempts=fit["n_attempts"],
                         error=fit["error"]))
    return rows


def _z_column(nvec, transform):
    if transform == "ranknorm":
        z = ranknorm(nvec)
    elif transform == "log1p":
        z = np.log1p(nvec)
    elif transform is None or transform == "none":
        z = np.asarray(nvec, float)
    else:
        raise ValueError(f"neighbor_transform must be 'ranknorm', 'log1p' or None, not {transform!r}")
    return None if np.ptp(z) < 1e-12 else z


def _fit_batch(genes, Yb, Nb, X, offset, groups, n_groups, L_list, zero_groups, transform, max_grad):
    """Fit `genes` (columns of Yb / Nb) with the Python engine; one row per gene x contrast."""
    from threadpoolctl import threadpool_limits

    from ._nbglmm import fit_nb2

    out = []
    with threadpool_limits(1):
        for j, gene in enumerate(genes):
            y = Yb[:, j].toarray().ravel()
            z = None if Nb is None else _z_column(Nb[:, j].toarray().ravel(), transform)
            # a whole compared group without a transcript: no finite contrast, not fitted
            empty = [cn for cn, idx in zero_groups if any(y[ix].sum() == 0 for ix in idx)]
            if len(empty) == len(L_list):
                fit = dict(beta=np.full(X.shape[1], np.nan), vcov=np.full((X.shape[1],) * 2, np.nan), theta=np.nan,
                           re_sd=np.nan, loglik=np.nan, converged=False, pdHess=False, max_grad=np.nan,
                           valid=False, optimizer="not_fitted", n_attempts=0,
                           error="a whole compared group has zero transcripts")
                rows = _gene_rows(fit, gene, L_list, False)
            else:
                Xg = X if z is None else np.column_stack([X, z])
                fit = fit_nb2(y, Xg, offset, groups, n_groups, max_grad=max_grad)
                rows = _gene_rows(fit, gene, L_list, z is not None)
            for r in rows:
                r["neighbor_term"] = z is not None
                r["zero_entire_group"] = r["contrast"] in empty
                if r["zero_entire_group"]:
                    r.update(p=np.nan, valid=False, error="a whole compared group has zero transcripts")
            out += rows
    return out


# --------------------------------------------------------------------------------------------------------------------
# glmmTMB engine (R)
# --------------------------------------------------------------------------------------------------------------------
def check_r(rscript_bin=None):
    """Whether R and the glmmTMB engine's packages are available: ``{"Rscript": path, "missing": [...]}``."""
    import subprocess

    exe = rscript_bin or os.environ.get("FRZSPATIAL_RSCRIPT") or shutil.which("Rscript")
    if exe is None:
        return {"Rscript": None, "missing": ["R"]}
    code = ("p <- c('glmmTMB','Matrix','data.table'); "
            "cat(p[!vapply(p, requireNamespace, logical(1), quietly=TRUE)], sep='\\n')")
    r = subprocess.run([exe, "-e", code], capture_output=True, text=True)
    return {"Rscript": exe, "missing": [m for m in r.stdout.split() if m]}


def _fit_family_r(genes, Y, N, X, xcols, offset, re_data, re_terms, transform, n_jobs, max_grad, rscript_bin=None):
    """Fit all genes of one family with glmmTMB in R; returns {gene: fit dict} in the Python engine's format."""
    from .rtools import rscript, write_counts

    tmp = Path(tempfile.mkdtemp(prefix="frz_smide_"))
    try:
        write_counts(tmp, "y", Y, [str(i) for i in range(Y.shape[0])], genes)
        Z = np.full((Y.shape[0], len(genes)), np.nan)
        if N is not None:
            for j in range(len(genes)):
                z = _z_column(N[:, j].toarray().ravel(), transform)
                if z is not None:
                    Z[:, j] = z
        Z.astype(np.float64).tofile(tmp / "z.bin")  # row-major n x G
        d = pd.DataFrame(X, columns=[f"X{k + 1}" for k in range(X.shape[1])])
        d["log_offset"] = offset
        for c in re_data.columns:
            d[c] = re_data[c].astype(str).to_numpy()
        d.to_csv(tmp / "design.csv", index=False)
        re_str = " + ".join(f"({a} | {b})" for a, b in re_terms)
        (tmp / "re.txt").write_text(re_str + "\n")
        workers = os.cpu_count() if n_jobs in (None, -1) else int(n_jobs)
        rscript("smide.R", tmp, workers, max_grad, rscript_bin=rscript_bin)
        res = pd.read_csv(tmp / "fits.csv")
        vc = np.fromfile(tmp / "vcov.bin", np.float64)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = {}
    for _, r in res.iterrows():
        p = int(r["n_coef"])
        beta = np.array([r[f"b{k + 1}"] for k in range(p)], float)
        off = int(r["vcov_offset"])
        V = vc[off:off + p * p].reshape(p, p)
        out[r["gene"]] = (dict(beta=beta, vcov=V, theta=r["theta"], re_sd=r["re_sd"], loglik=r["loglik"],
                               converged=bool(r["converged"]), pdHess=bool(r["pdHess"]), max_grad=r["max_grad"],
                               valid=bool(r["valid"]), optimizer=str(r["optimizer"]), n_attempts=int(r["n_attempts"]),
                               error="" if pd.isna(r["error"]) else str(r["error"])), bool(r["neighbor_term"]))
    return out


# --------------------------------------------------------------------------------------------------------------------
# main function
# --------------------------------------------------------------------------------------------------------------------
def smide(sdata, celltype_col, formula, contrasts=None, celltypes=None, table="table", layer=None, genes=None,
          size="total", radius_um=30.0, n_neighbors=None, neighbor_types="other", block_cols="auto",
          neighbor_agg="sum", neighbor_scale=1e4, neighbor_transform="ranknorm", coords="spatial", px_um=PX_UM,
          min_detection=0.10, min_cells=50, max_cells_per_group=None, contrast_mode="subset", engine="python",
          n_jobs=-1, batch_size=32, checkpoint_dir=None, max_grad=1e-2, seed=0, store=True, rscript_bin=None):
    """smiDE differential expression for every gene in any cell types of a SpatialData object.

    One family = one target cell type x one contrast. Per gene, an NB2 count model of the raw counts of the family's
    cells with the cell's total count as offset, the fixed and random effects of `formula`, and the RankNorm-
    transformed expression of the same gene in neighbouring cells of other types (see the module docstring).

    Parameters
    ----------
    sdata
        SpatialData (its table `table`) or an AnnData, with raw integer counts (``X`` or `layer`).
    celltype_col
        obs column of the cell types (any annotation level).
    formula
        Right-hand side of the model, lme4 style, e.g. ``"~ disease + slide + (1|patient)"`` or ``"~ age"``. The
        offset and the neighbour covariate are added automatically. Python engine: at most one random-intercept
        term ``(1|g)``.
    contrasts
        ``{name: (variable, case, reference)}`` for a categorical variable, ``{name: (design_column,)}`` for any
        coefficient (e.g. a continuous ``("age",)``). None: every level of the formula's first variable against its
        first level (categorical), or its coefficient (numeric). Positive beta = higher in `case`.
    celltypes
        Target cell types (default: every type with at least `min_cells` cells in a family).
    genes
        Genes to consider (default: all, control probes excluded).
    size
        Per-cell size factor of the offset: ``"total"`` (counts over all genes, before any filter) or an obs column.
    radius_um, n_neighbors
        Neighbourhood: cells within `radius_um` µm, or the `n_neighbors` nearest cells.
    neighbor_types
        Neighbours that enter the covariate: ``"other"`` (cells of other types: the smiDE default), ``"all"`` or a
        list of cell types.
    block_cols
        obs columns that cells must share to be neighbours (``"auto"``: ``slide`` if present). Use e.g.
        ``("slide", "sample_id", "core_str")`` to keep neighbourhoods within one core.
    neighbor_agg, neighbor_scale
        ``"sum"`` (default) or ``"mean"`` of ``neighbor_scale * count / size`` over the neighbours.
    neighbor_transform
        ``"ranknorm"`` (default, recomputed within each family), ``"log1p"``, or None. ``neighbor_types=None``
        drops the covariate (plain NB2 mixed-model DE).
    coords, px_um
        obsm key of the cell centroids (or an (n, 2) array) and its µm per unit (frzspatial tables store AtoMx
        pixels: ``fs.PX_UM``; use 1 for coordinates already in µm).
    min_detection
        A gene is tested when its detection rate exceeds this in at least one compared group (union filter;
        all cells for a continuous contrast).
    min_cells
        Families with fewer cells (or a compared group with fewer than 3 cells) are skipped.
    max_cells_per_group
        Speed: randomly keep at most this many cells per compared group (stratified by the random-effect group).
        None (default) keeps all cells.
    contrast_mode
        ``"subset"`` (default): each pairwise contrast is fitted on the cells of its two groups only.
        ``"joint"``: one fit per cell type on all levels; every contrast is a linear combination of its coefficients.
    engine
        ``"python"`` (default) or ``"glmmtmb"`` (R; any random-effect terms; see ``check_r()``).
    rscript_bin
        Rscript of the R with glmmTMB for ``engine="glmmtmb"`` (default: ``$FRZSPATIAL_RSCRIPT``, else ``Rscript``
        on the PATH), e.g. a conda R kept apart from the R of the other R tools.
    n_jobs, batch_size
        Parallel workers (-1: all cores) and genes per task. One BLAS thread per worker.
    checkpoint_dir
        Folder of finished gene batches, keyed by a hash of their inputs and settings: an interrupted run resumes.
    max_grad
        Largest |gradient| of a valid fit.
    seed
        For `max_cells_per_group`.
    store
        Store the results and settings in ``table.uns["smide"]``.

    Returns
    -------
    DataFrame
        One row per cell type x contrast x tested gene: ``beta`` (natural-log rate ratio), ``SE``, ``z``, ``p``,
        ``p_for_inference`` (NA for invalid fits), ``FDR_BH`` (within the family), ``FDR_BH_all_celltypes``,
        ``log2FC`` with its 95 % CI, ``neighbor_beta``, ``re_sd``, ``dispersion``, cell / group counts, detection
        and mean raw count per compared group, and fit diagnostics (``valid``, ``converged``, ``pdHess``,
        ``max_grad``, ``error``).
    """
    tb = sdata.tables[table] if hasattr(sdata, "tables") else sdata
    obs = tb.obs
    if engine not in ("python", "glmmtmb"):
        raise ValueError("engine must be 'python' or 'glmmtmb'")
    if contrast_mode not in ("subset", "joint"):
        raise ValueError("contrast_mode must be 'subset' or 'joint'")
    fixed, re_terms = _parse_formula(formula)
    if engine == "python":
        if len(re_terms) > 1 or any(a != "1" for a, _ in re_terms):
            raise ValueError("the python engine takes at most one random intercept (1|g); use engine='glmmtmb'")
    re_vars = sorted({v for _, b in re_terms for v in re.split(r"[/:]", b) if v.strip()}, key=str)
    re_vars = [v.strip() for v in re_vars]
    for v in [celltype_col, *re_vars]:
        if v not in obs:
            raise KeyError(f"obs has no column {v!r}")
    fvars = _formula_vars(fixed, obs)
    cons = _normalise_contrasts(contrasts, fixed, obs)

    # 1. Counts, size factors
    names = tb.var_names.astype(str)
    is_gene = ~names.str.startswith(_PROBE_PREFIXES)
    # memory: one sparse matrix in its own dtype (no float64 / CSC copy of the whole table; families are small)
    Xall = tb.layers[layer] if layer else tb.X
    Xall = Xall if sp.isspmatrix_csr(Xall) or isinstance(Xall, sp.csr_array) else sp.csr_matrix(Xall)
    _check_integer_counts(Xall.data)
    G = Xall if is_gene.all() else Xall[:, np.flatnonzero(is_gene)]
    gnames = np.asarray(names[is_gene])
    U = np.asarray(G.sum(axis=1, dtype=np.float64)).ravel() if size == "total" else obs[size].to_numpy(float)
    candidates = np.arange(G.shape[1]) if genes is None else np.flatnonzero(np.isin(gnames, list(genes)))
    ctype = obs[celltype_col].astype(object).where(obs[celltype_col].notna(), None).to_numpy()
    ok_cell = (U > 0) & pd.notna(ctype)

    # 2. Neighbour graph, once (cells with a type and counts)
    use_nb = neighbor_types is not None
    if use_nb:
        xy = np.asarray(tb.obsm[coords] if isinstance(coords, str) else coords, float)[:, :2] * float(px_um)
        if block_cols == "auto":
            block_cols = ("slide",) if "slide" in obs else ()
        block_cols = (block_cols,) if isinstance(block_cols, str) else tuple(block_cols or ())
        ok_cell &= np.all(np.isfinite(xy), axis=1)
        for c in block_cols:
            ok_cell &= obs[c].notna().to_numpy()
        blocks = obs[list(block_cols)].astype(str).agg("::".join, axis=1).to_numpy() if block_cols else None
        A = neighbor_graph(xy, blocks, radius_um, n_neighbors)
        A = A.multiply(ok_cell[:, None]).multiply(ok_cell[None, :]).tocsr()
        _log(f"smide: neighbour graph {A.nnz:,} edges ({'k=' + str(n_neighbors) if n_neighbors else f'{radius_um} µm'}"
             f", blocks {block_cols or 'none'})")

    types = celltypes if celltypes is not None else [t for t in pd.unique(ctype[ok_cell])]
    rng = np.random.default_rng(seed)
    if engine == "glmmtmb":
        chk = check_r(rscript_bin)
        if chk["missing"]:
            raise RuntimeError(f"engine='glmmtmb' needs R ({chk['Rscript']}) with: {', '.join(chk['missing'])}")
    params = dict(celltype_col=celltype_col, formula=formula, contrasts={k: list(v) for k, v in cons.items()},
                  size=size, radius_um=radius_um, n_neighbors=n_neighbors,
                  neighbor_types=neighbor_types if isinstance(neighbor_types, (str, type(None))) else list(neighbor_types),
                  block_cols=list(block_cols) if use_nb else None, neighbor_agg=neighbor_agg,
                  neighbor_scale=neighbor_scale, neighbor_transform=neighbor_transform, px_um=px_um,
                  min_detection=min_detection, min_cells=min_cells, max_cells_per_group=max_cells_per_group,
                  contrast_mode=contrast_mode, engine=engine, max_grad=max_grad, layer=layer, seed=seed)

    # families: (cell type, fit id, {contrast: spec})
    if contrast_mode == "subset":
        fits = [(name, {name: spec}) for name, spec in cons.items()]
    else:
        fits = [("joint", cons)]
    results, skipped = [], []
    for T in types:
        tmask = ok_cell & (ctype == T)
        if use_nb:
            if neighbor_types == "other":
                allowed = ok_cell & (ctype != T)
            elif neighbor_types == "all":
                allowed = ok_cell.copy()
            else:
                allowed = ok_cell & np.isin(ctype, list(neighbor_types))
            tidx_all = np.flatnonzero(tmask)
            Gc = G if candidates.size == G.shape[1] else G[:, candidates]
            Nall = neighbor_expression(A, Gc, U, tidx_all, allowed, neighbor_scale, neighbor_agg)
            del Gc
            pos_in_T = np.full(len(obs), -1)
            pos_in_T[tidx_all] = np.arange(tidx_all.size)
        for fit_id, fcons in fits:
            # 3. Cells of the family
            fam = tmask.copy()
            for c in fvars + re_vars:
                fam &= obs[c].notna().to_numpy()
            pair_vars = {spec[0] for spec in fcons.values() if len(spec) == 3}
            if contrast_mode == "subset":
                for spec in fcons.values():
                    if len(spec) == 3:
                        fam &= obs[spec[0]].isin([spec[1], spec[2]]).to_numpy()
            if max_cells_per_group:
                fam = _subsample(fam, obs, fcons, re_vars, max_cells_per_group, rng)
            idx = np.flatnonzero(fam)
            label = f"{T} / {fit_id}"
            if idx.size < min_cells:
                skipped.append((T, fit_id, f"{idx.size} cells < min_cells"))
                continue
            data = obs.iloc[idx][fvars].copy()
            for c in data.columns:
                if isinstance(data[c].dtype, pd.CategoricalDtype):
                    data[c] = data[c].cat.remove_unused_categories()
                elif not pd.api.types.is_numeric_dtype(data[c]):
                    data[c] = pd.Categorical(data[c].astype(str))
            for spec in fcons.values():
                if len(spec) == 3 and contrast_mode == "subset":
                    data[spec[0]] = pd.Categorical(data[spec[0]].astype(object), categories=[spec[2], spec[1]])
            X, xcols = _design(fixed, data)
            if np.linalg.matrix_rank(X) < X.shape[1]:
                skipped.append((T, fit_id, "fixed-effect design not full rank (a term is confounded with another, "
                                           "e.g. the contrast with slide)"))
                _log(f"smide: skip {label}: design not full rank")
                continue
            # contrast vectors and compared groups
            L_list, groups_of, bad = [], {}, None
            for cname, spec in fcons.items():
                L = np.zeros(X.shape[1])
                if len(spec) == 1:
                    if spec[0] not in xcols:
                        bad = f"contrast {cname}: no design column {spec[0]!r} (columns: {xcols})"
                        break
                    L[xcols.index(spec[0])] = 1.0
                    groups_of[cname] = None
                else:
                    var, case, ref = spec
                    vals = obs.iloc[idx][var].astype(object).to_numpy()
                    ic, ir = np.flatnonzero(vals == case), np.flatnonzero(vals == ref)
                    if min(ic.size, ir.size) < 3:
                        bad = f"contrast {cname}: fewer than 3 cells in a group"
                        break
                    kc, kr = _level_col(xcols, var, case), _level_col(xcols, var, ref)
                    if kc is None and kr is None:
                        bad = f"contrast {cname}: levels {case!r}/{ref!r} not in the design"
                        break
                    if kc is not None:
                        L[kc] += 1.0
                    if kr is not None:
                        L[kr] -= 1.0
                    groups_of[cname] = (ic, ir)
                L_list.append((cname, L))
            if bad:
                skipped.append((T, fit_id, bad))
                _log(f"smide: skip {label}: {bad}")
                continue
            # 4. Genes: union detection filter per contrast
            Yf = sp.csc_matrix(G[idx][:, candidates], dtype=np.float64)
            det = (Yf > 0)
            tested = {}
            for cname, grp in groups_of.items():
                if grp is None:
                    rate = np.asarray(det.mean(axis=0)).ravel()
                else:
                    rate = np.maximum(np.asarray(det[grp[0]].mean(axis=0)).ravel(),
                                      np.asarray(det[grp[1]].mean(axis=0)).ravel())
                tested[cname] = rate > min_detection
            fit_genes = np.flatnonzero(np.any(np.column_stack(list(tested.values())), axis=1))
            if not fit_genes.size:
                skipped.append((T, fit_id, "no gene passes min_detection"))
                continue
            offset = np.log(U[idx])
            if re_terms and engine == "python":
                gcodes, glev = pd.factorize(obs.iloc[idx][re_terms[0][1]].astype(str))
                n_groups = len(glev)
            else:
                gcodes, n_groups = None, 0
            Nf = Nall[pos_in_T[idx]][:, fit_genes].tocsc() if use_nb else None
            Yg = Yf[:, fit_genes].tocsc()
            gl = gnames[candidates][fit_genes].tolist()
            zero_groups = [(cn, list(gr)) for cn, gr in groups_of.items() if gr is not None]
            _log(f"smide: {label}: {idx.size:,} cells, {fit_genes.size:,} genes"
                 + (f", {n_groups} random-effect groups" if n_groups else ""))
            # 5. Fit
            if engine == "python":
                rows = _run_batches(gl, Yg, Nf, X, offset, gcodes, n_groups, L_list, zero_groups,
                                    neighbor_transform, max_grad, n_jobs, batch_size, checkpoint_dir,
                                    (T, fit_id, params))
            else:
                re_data = obs.iloc[idx][re_vars]
                fitted = _fit_family_r(gl, Yg, Nf, X, xcols, offset, re_data, re_terms, neighbor_transform, n_jobs,
                                       max_grad, rscript_bin)
                rows = []
                for j, gname in enumerate(gl):
                    f, zin = fitted[gname]
                    rr = _gene_rows(f, gname, L_list, zin)
                    y = Yg[:, j].toarray().ravel()
                    for r in rr:
                        r["neighbor_term"] = zin
                        gr = groups_of[r["contrast"]]
                        r["zero_entire_group"] = gr is not None and any(y[ix].sum() == 0 for ix in gr)
                        if r["zero_entire_group"]:
                            r.update(p=np.nan, valid=False, error="a whole compared group has zero transcripts")
                    rows += rr
            res = pd.DataFrame(rows)
            # 6. Per-contrast tables: tested genes only, group summaries, BH
            for cname, _ in L_list:
                keep_genes = set(np.asarray(gl)[tested[cname][fit_genes]])
                r = res[(res["contrast"] == cname) & res["gene"].isin(keep_genes)].copy()
                gidx = {g: k for k, g in enumerate(gl)}
                cols = [gidx[g] for g in r["gene"]]
                grp = groups_of[cname]
                spec = fcons[cname]
                r.insert(0, "celltype", T)
                r["case"] = spec[1] if len(spec) == 3 else np.nan
                r["reference"] = spec[2] if len(spec) == 3 else np.nan
                r["n_cells"] = idx.size
                if gcodes is not None or re_vars:
                    rv = obs.iloc[idx][re_vars[0]].astype(str).to_numpy() if re_vars else None
                    r["n_re_groups"] = len(pd.unique(rv)) if rv is not None else np.nan
                if grp is not None:
                    ic, ir = grp
                    r["n_case_cells"], r["n_ref_cells"] = ic.size, ir.size
                    if re_vars:
                        r["n_case_groups"] = len(pd.unique(rv[ic]))
                        r["n_ref_groups"] = len(pd.unique(rv[ir]))
                    Yc = Yg[ic][:, cols]
                    Yr = Yg[ir][:, cols]
                    r["detect_case"] = np.asarray((Yc > 0).mean(axis=0)).ravel()
                    r["detect_ref"] = np.asarray((Yr > 0).mean(axis=0)).ravel()
                    r["mean_case"] = np.asarray(Yc.mean(axis=0)).ravel()
                    r["mean_ref"] = np.asarray(Yr.mean(axis=0)).ravel()
                else:
                    r["detect_all"] = np.asarray((Yg[:, cols] > 0).mean(axis=0)).ravel()
                r["p_for_inference"] = np.where(r["valid"], r["p"], np.nan)
                r["FDR_BH"] = _bh(r["p_for_inference"].to_numpy(), len(r))
                r["beta_low"] = r["beta"] - 1.959963984540054 * r["SE"]
                r["beta_high"] = r["beta"] + 1.959963984540054 * r["SE"]
                for c in ("", "_low", "_high"):
                    r[f"log2FC{c}"] = r[f"beta{c}"] / np.log(2)
                r["re_sd_near_zero"] = r["re_sd"] < 1e-4
                results.append(r)
                n_sig = int((r["FDR_BH"] < 0.05).sum())
                _log(f"smide: {T} / {cname}: {len(r)} genes, {int(r['valid'].sum())} valid, {n_sig} at FDR < 0.05")
    if not results:
        raise ValueError(f"no family could be fitted: {skipped}")
    out = pd.concat(results, ignore_index=True)
    out["FDR_BH_all_celltypes"] = np.nan
    for cname, ix in out.groupby("contrast").groups.items():
        out.loc[ix, "FDR_BH_all_celltypes"] = _bh(out.loc[ix, "p_for_inference"].to_numpy(), len(ix))
    first = ["celltype", "contrast", "gene", "beta", "SE", "z", "p", "p_for_inference", "FDR_BH",
             "FDR_BH_all_celltypes", "log2FC", "log2FC_low", "log2FC_high", "beta_low", "beta_high", "case", "reference"]
    out = out[first + [c for c in out.columns if c not in first]]
    if store:
        tb.uns["smide"] = {"results": out, "params": _jsonable(params),
                           "skipped": pd.DataFrame(skipped, columns=["celltype", "fit", "reason"]).astype(str)}
    return out


def _jsonable(d):
    return {k: (str(v) if v is None or isinstance(v, (dict, tuple)) else v) for k, v in d.items()}


def _subsample(fam, obs, fcons, re_vars, cap, rng):
    """Keep at most `cap` cells per compared group, stratified by the random-effect group."""
    idx = np.flatnonzero(fam)
    keys = [spec[0] for spec in fcons.values() if len(spec) == 3]
    grp = obs.iloc[idx][keys].astype(str).agg("::".join, axis=1).to_numpy() if keys else np.zeros(idx.size, str)
    strat = obs.iloc[idx][re_vars].astype(str).agg("::".join, axis=1).to_numpy() if re_vars else np.zeros(idx.size, str)
    keep = np.zeros(idx.size, bool)
    for gv in np.unique(grp):
        ig = np.flatnonzero(grp == gv)
        if ig.size <= cap:
            keep[ig] = True
            continue
        frac = cap / ig.size
        for sv in np.unique(strat[ig]):
            iss = ig[strat[ig] == sv]
            keep[rng.choice(iss, max(1, int(np.ceil(frac * iss.size))), replace=False)] = True
    out = np.zeros_like(fam)
    out[idx[keep]] = True
    return out


def _hash(*parts):
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
        elif sp.issparse(p):
            p = sp.csc_matrix(p)
            for a in (p.data, p.indices, p.indptr):
                h.update(np.ascontiguousarray(a).tobytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:20]


def _run_batches(genes, Y, N, X, offset, groups, n_groups, L_list, zero_groups, transform, max_grad, n_jobs,
                 batch_size, checkpoint_dir, key):
    from joblib import Parallel, delayed

    batches = [list(range(i, min(i + batch_size, len(genes)))) for i in range(0, len(genes), batch_size)]
    common = _hash(X, offset, groups if groups is not None else "noRE", n_groups, [(c, L) for c, L in L_list],
                   transform, max_grad, _ENGINE_VERSION, _engine_hash(), key[2]) if checkpoint_dir else None
    done, todo = {}, []
    for k, b in enumerate(batches):
        f = None
        if checkpoint_dir:
            bh = _hash(common, [genes[j] for j in b], Y[:, b], N[:, b] if N is not None else "noN")
            f = Path(checkpoint_dir) / re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{key[0]}__{key[1]}") / f"{bh}.pkl"
            if f.exists():
                done[k] = pickle.loads(f.read_bytes())
                continue
        todo.append((k, b, f))
    if done:
        _log(f"smide: {len(done)} of {len(batches)} gene batches from checkpoints")

    jobs = 1 if n_jobs == 1 or len(todo) <= 1 else n_jobs
    outs = Parallel(n_jobs=jobs, backend="loky")(
        delayed(_fit_batch_checkpoint)(f, [genes[j] for j in b], Y[:, b], None if N is None else N[:, b], X, offset,
                                       groups, n_groups, L_list, zero_groups, transform, max_grad)
        for _, b, f in todo) if todo else []
    for (k, _, _), o in zip(todo, outs):
        done[k] = o
    return [r for k in range(len(batches)) for r in done[k]]


def _fit_batch_checkpoint(f, *args):
    """_fit_batch, saving its rows to `f` (atomically) as soon as the batch is done."""
    rows = _fit_batch(*args)
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(pickle.dumps(rows))
        tmp.replace(f)
    return rows
