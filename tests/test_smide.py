"""frzspatial.de.smide and its NB2 GLMM engine, on simulated data."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.optimize import approx_fprime
from scipy.stats import norm

from frzspatial import de
from frzspatial._nbglmm import _Laplace, fit_nb2


def _simulate(seed=0, n_per=250, n_pat=12, G=12, effect=1.0, theta=3.0, tau=0.3):
    """Patients in 3 groups (N, G1, G2), one block each; types A/B/C; genes g0-g2 higher in A cells of G1."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    groups = ["N", "G1", "G2"]
    obs = pd.DataFrame({
        "patient": np.repeat([f"P{i}" for i in range(n_pat)], n_per),
        "disease": np.repeat([groups[i % 3] for i in range(n_pat)], n_per),
        "slide": np.repeat([f"S{i % 2}" for i in range(n_pat)], n_per),
        "ct": rng.choice(["A", "B", "C"], n_pat * n_per, p=[0.45, 0.3, 0.25]),
        "age": np.repeat(rng.normal(50, 10, n_pat), n_per),
    })
    n = len(obs)
    xy = rng.uniform(0, 300, (n, 2)) + np.repeat(np.arange(n_pat) * 1000.0, n_per)[:, None]
    pat = np.repeat(np.arange(n_pat), n_per)
    b = rng.normal(0, tau, (n_pat, G))
    base = np.log(rng.uniform(0.3, 1.5, G))
    U = rng.integers(150, 600, n)
    eff = np.zeros(G)
    eff[:3] = effect
    hit = ((obs["ct"] == "A") & (obs["disease"] == "G1")).to_numpy()[:, None]
    mu = np.exp(base[None, :] + b[pat] + np.log(U / 300.0)[:, None] + np.where(hit, eff[None, :], 0.0))
    Y = rng.negative_binomial(theta, theta / (theta + mu))
    a = ad.AnnData(sp.csr_matrix(Y.astype(np.float32)),
                   obs=obs.astype({"disease": "category", "ct": "category", "slide": "category"}))
    a.var_names = [f"g{i}" for i in range(G)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obsm["spatial"] = xy
    return a


# --- building blocks -------------------------------------------------------------------------------------------------
def test_ranknorm_blom_with_ties():
    x = np.array([0, 0, 1, 2, 10.0])
    z = de.ranknorm(x)
    expected = norm.ppf((np.array([1.5, 1.5, 3, 4, 5]) - 0.375) / (5 + 0.25))
    assert np.allclose(z, expected) and z[0] == z[1] and np.all(np.diff(z) >= 0)


def test_neighbor_graph_radius_blocks_knn():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 100, (300, 2))
    blocks = rng.choice(["a", "b"], 300)
    A = de.neighbor_graph(xy, blocks, radius_um=12.0)
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    brute = (d <= 12.0) & (blocks[:, None] == blocks[None]) & ~np.eye(300, dtype=bool)
    assert (A.toarray().astype(bool) == brute).all()
    K = de.neighbor_graph(xy, None, n_neighbors=5)
    assert (np.asarray(K.sum(axis=1)).ravel() == 5).all() and K.diagonal().sum() == 0


def test_neighbor_expression_direct_replay():
    """The bundle's 'direct spatial covariate replay': sum over other-type neighbours of 1e4 * y / U."""
    rng = np.random.default_rng(2)
    n, G = 200, 6
    xy = rng.uniform(0, 80, (n, 2))
    Y = sp.csr_matrix(rng.poisson(1.0, (n, G)).astype(float))
    U = np.asarray(Y.sum(axis=1)).ravel() + 5
    ctype = rng.choice(["T", "O1", "O2"], n)
    A = de.neighbor_graph(xy, None, radius_um=15.0)
    target = np.flatnonzero(ctype == "T")
    N = de.neighbor_expression(A, Y, U, target, ctype != "T").toarray()
    Yd = Y.toarray()
    for k, i in enumerate(target[:20]):
        jj = np.flatnonzero((ctype != "T") & (np.hypot(*(xy - xy[i]).T) <= 15.0) & (np.arange(n) != i))
        expected = (Yd[jj] * (1e4 / U[jj])[:, None]).sum(0) if jj.size else np.zeros(G)
        assert np.allclose(N[k], expected)


def test_bh_counts_failed_fits():
    p = np.array([0.01, 0.02, np.nan, 0.04])
    q = de._bh(p, 4)
    assert np.isnan(q[2]) and np.isclose(q[0], 0.01 * 4 / 1) and np.isclose(q[3], 0.04 * 4 / 3)


def test_parse_formula():
    assert de._parse_formula("~ disease + slide + (1|patient)") == ("disease + slide", [("1", "patient")])
    assert de._parse_formula("y ~ (1 | a) + x") == ("x", [("1", "a")])
    assert de._parse_formula("~ 1 + (1|g)")[0] == "1"


# --- engine ----------------------------------------------------------------------------------------------------------
def test_engine_gradient_and_recovery():
    rng = np.random.default_rng(3)
    n, K = 4000, 16
    g = rng.integers(0, K, n)
    X = np.c_[np.ones(n), (g % 2).astype(float), rng.normal(size=n)]
    off = np.log(rng.integers(50, 500, n))
    beta, tau, theta = np.array([-4.5, 0.6, 0.3]), 0.4, 2.0
    mu = np.exp(X @ beta + off + rng.normal(0, tau, K)[g])
    y = rng.negative_binomial(theta, theta / (theta + mu))
    obj = _Laplace(y, X, off, g, K)
    par = np.r_[beta + 0.1, np.log(theta) - 0.2, np.log(tau) + 0.3]
    _, grad = obj(par)

    def f(p):
        obj.b = np.zeros(K)
        return obj(p)[0]

    assert np.max(np.abs(approx_fprime(par, f, 1e-6) - grad) / np.maximum(1, np.abs(grad))) < 1e-4
    r = fit_nb2(y, X, off, g, K)
    se = np.sqrt(np.diag(r["vcov"]))
    assert r["valid"] and np.all(np.abs(r["beta"] - beta) < 4 * se)
    assert 1.2 < r["theta"] < 3.5 and 0.15 < r["re_sd"] < 0.8
    glm = fit_nb2(y, X, off)
    assert glm["valid"] and np.isnan(glm["re_sd"])


# --- smide -----------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim():
    return _simulate()


def test_smide_finds_de_genes(sim, tmp_path):
    res = de.smide(sim, "ct", "~ disease + (1|patient)", contrasts={"G1_vs_N": ("disease", "G1", "N")},
                   px_um=1, block_cols=("patient",), min_cells=20, n_jobs=2, batch_size=4, checkpoint_dir=tmp_path)
    a = res[res["celltype"] == "A"].set_index("gene")
    assert set(a.index[a["FDR_BH"] < 0.05]) >= {"g0", "g1", "g2"}
    assert (a.loc[["g0", "g1", "g2"], "beta"] > 0).all()
    assert a["valid"].mean() > 0.8 and (res["n_case_groups"] == 4).all() and (res["n_ref_groups"] == 4).all()
    assert np.allclose(res["log2FC"], res["beta"] / np.log(2), equal_nan=True)
    assert "smide" in sim.uns and len(sim.uns["smide"]["results"]) == len(res)
    # resumed from the checkpoints: identical
    again = de.smide(sim, "ct", "~ disease + (1|patient)", contrasts={"G1_vs_N": ("disease", "G1", "N")},
                     px_um=1, block_cols=("patient",), min_cells=20, n_jobs=1, batch_size=4, checkpoint_dir=tmp_path)
    pd.testing.assert_frame_equal(res, again)


def test_smide_joint_continuous_and_no_random_effect(sim):
    joint = de.smide(sim, "ct", "~ disease + (1|patient)", celltypes=["A"], contrast_mode="joint", px_um=1,
                     block_cols=("patient",), min_cells=20, n_jobs=1, store=False)
    assert set(joint["contrast"]) == {"G2_vs_G1", "N_vs_G1"}  # every level against the first category
    cont = de.smide(sim, "ct", "~ age + (1|patient)", celltypes=["B"], px_um=1, block_cols=("patient",),
                    min_cells=20, n_jobs=1, store=False)
    assert set(cont["contrast"]) == {"age"} and cont["case"].isna().all()
    plain = de.smide(sim, "ct", "~ disease", contrasts=[("disease", "G1", "N")], celltypes=["A"],
                     neighbor_types=None, min_cells=20, n_jobs=1, store=False)
    assert not plain["neighbor_term"].any() and plain["re_sd"].isna().all()


def test_smide_rejects_confounded_design(sim):
    with pytest.raises(ValueError, match="no family could be fitted"):
        de.smide(sim, "ct", "~ disease + patient", contrasts={"G1_vs_N": ("disease", "G1", "N")}, px_um=1,
                 celltypes=["A"], min_cells=20, n_jobs=1, store=False)


def test_smide_glmmtmb_agrees():
    if de.check_r()["missing"]:
        pytest.skip("R with glmmTMB not installed")
    a = _simulate(seed=5, n_per=150, G=6)
    kw = dict(contrasts={"G1_vs_N": ("disease", "G1", "N")}, celltypes=["A"], px_um=1, block_cols=("patient",),
              min_cells=20, n_jobs=2, store=False)
    py = de.smide(a, "ct", "~ disease + (1|patient)", **kw).set_index("gene")
    r = de.smide(a, "ct", "~ disease + (1|patient)", engine="glmmtmb", **kw).set_index("gene")
    ok = py["valid"] & r["valid"]
    assert ok.sum() >= 4
    assert np.allclose(py.loc[ok, "beta"], r.loc[ok, "beta"], atol=2e-3)
    assert np.allclose(py.loc[ok, "SE"], r.loc[ok, "SE"], rtol=2e-2)


def test_overlap_ratio_matches_nanostring():
    """smiDE::overlap_ratio_metric, step by step: column-standardised adjacency over the other-type cells, mean over
    the cell type's cells (cells without such neighbours count 0), data = counts x mean(total) / total."""
    rng = np.random.default_rng(4)
    n, G = 300, 5
    xy = rng.uniform(0, 100, (n, 2))
    Y = sp.csr_matrix(rng.poisson(2.0, (n, G)).astype(float))
    U = np.asarray(Y.sum(axis=1)).ravel() + 3
    ctype = rng.choice(["T", "O1", "O2"], n)
    W = de.neighbor_graph(xy, None, radius_um=9.0).toarray()
    data = Y.toarray() * (U.mean() / U)[:, None]                          # assay_matrix (cells x genes here)
    cells, other = ctype == "T", ctype != "T"
    rs = W[other].sum(axis=0)                                             # colSums(W[other_cells, ])
    rs[rs == 0] = np.inf
    Wcs = W / rs[None, :]                                                 # W %*% Diagonal(1 / rs)
    other_ii = data[other].T @ Wcs[np.ix_(other, cells)]                  # genes x cells_ii
    avg_n_ref, avg_t_ref = other_ii.mean(axis=1), data[cells].mean(axis=0)
    avg_t, avg_n = de.overlap_ratio(sp.csr_matrix(W), Y, U, np.flatnonzero(cells), other)
    assert np.allclose(avg_t, avg_t_ref) and np.allclose(avg_n, avg_n_ref)


def test_smide_overlap_ratio_column_and_screen(sim):
    kw = dict(contrasts={"G1_vs_N": ("disease", "G1", "N")}, celltypes=["A"], px_um=1, block_cols=("patient",),
              min_cells=20, n_jobs=1, store=False)
    full = de.smide(sim, "ct", "~ disease + (1|patient)", **kw)
    assert {"overlap_ratio", "avg_celltype", "avg_neighbor_other"} <= set(full.columns)
    assert full["overlap_ratio"].notna().all()
    cut = float(full["overlap_ratio"].median())
    screened = de.smide(sim, "ct", "~ disease + (1|patient)", max_overlap_ratio=cut, **kw)
    assert set(screened["gene"]) == set(full.loc[full["overlap_ratio"] < cut, "gene"])
    # the screened genes leave the BH family; the kept genes' fits are unchanged
    m = full.set_index("gene").loc[screened["gene"]]
    assert np.allclose(m["beta"].to_numpy(), screened["beta"].to_numpy())


def test_smide_cells_restricts_tested_cells_only(sim):
    """cells= limits the tested cells; neighbours still come from all cells (so it differs from subsetting)."""
    kw = dict(contrasts={"G1_vs_N": ("disease", "G1", "N")}, celltypes=["A"], px_um=1, block_cols=("patient",),
              min_cells=20, n_jobs=1, store=False, max_overlap_ratio=None)
    keep = sim.obsm["spatial"][:, 1] % 300 < 150          # half of each patient's area
    r = de.smide(sim, "ct", "~ disease + (1|patient)", cells=keep, **kw)
    o = sim.obs
    expect = (keep & (o["ct"] == "A") & o["disease"].isin(["G1", "N"])).sum()
    assert (r["n_cells"] == expect).all()
    sub = de.smide(sim[keep].copy(), "ct", "~ disease + (1|patient)", **kw)   # neighbours cut at the border
    m = r.set_index("gene")["neighbor_beta"].reindex(sub["gene"])
    assert not np.allclose(m.to_numpy(), sub["neighbor_beta"].to_numpy())


def test_smide_plots(sim):
    import matplotlib
    matplotlib.use("Agg")
    from frzspatial.pl import plot_smide_heatmap, plot_smide_patients

    sim.layers["counts"] = sim.X
    sim.obs["in_glom"] = np.where(sim.obsm["spatial"][:, 0] % 1000 < 200, "g1", None)   # a glomerulus-id-like column
    res = de.smide(sim, "ct", "~ disease + (1|patient)",
                   contrasts={"G1_vs_N": ("disease", "G1", "N"), "G2_vs_N": ("disease", "G2", "N")}, px_um=1,
                   block_cols=("patient",), min_cells=20, n_jobs=1, store=False, cells="in_glom")
    assert (res["n_cells"] < (sim.obs["ct"] == "A").sum()).all()
    fig = plot_smide_heatmap(res, n_top=3)
    assert fig.axes
    fig = plot_smide_patients(sim, res, "ct", "G1_vs_N", group_col="disease", patient_col="patient",
                              cells="in_glom", n_genes=6, min_cells=3, column_labels="slide")
    assert len(fig.axes) >= 3
    assert [t.get_text() for t in fig.legends[0].get_texts()][:2] == ["N", "G1"]   # reference first, then case
