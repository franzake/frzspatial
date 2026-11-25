"""Figure helpers of frzspatial.pl: group proportions and values with statistics, radial position."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from frzspatial.pl import plot_group_proportions, plot_group_values, plot_radial_position


def _obs(seed=0):
    """30 patients in 3 groups; type P is depleted in group B (10 % vs 30 %); a glomerulus id on 80 % of cells."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(30):
        g = ["A", "B", "C"][k % 3]
        p_p = 0.1 if g == "B" else 0.3
        n = 300
        t = rng.choice(["P", "M", "E"], n, p=[p_p, (1 - p_p) / 2, (1 - p_p) / 2])
        glom = np.where(rng.random(n) < 0.8, "g1", None)
        dist = np.where(t == "M", rng.uniform(0, 0.5, n), rng.uniform(0.6, 1.1, n))
        rows.append(pd.DataFrame({"ct": t, "grp": g, "pat": f"p{k}", "glom": glom, "dist": dist}))
    return pd.concat(rows, ignore_index=True)


def test_group_proportions_finds_depletion():
    obs = _obs()
    fig, stats = plot_group_proportions(obs, "ct", "grp", patient_col="pat", cells="glom", celltypes=["P", "M", "E"],
                                        palette={"A": "g", "B": "r", "C": "b"})
    p = stats[stats["celltype"] == "P"]
    assert p["kruskal_p"].iloc[0] < 1e-3
    q = p.set_index(["group_a", "group_b"])["q"]
    assert q[("A", "B")] < 0.05 and q[("B", "C")] < 0.05 and q[("A", "C")] > 0.05
    assert len(fig.axes) == 3
    star_texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("*" in s for s in star_texts)


def test_group_values_and_radial():
    obs = _obs(1)
    per = obs.groupby("pat").agg(v=("dist", "mean"), g=("grp", "first"))
    fig, stats = plot_group_values(per["v"], per["g"], ylabel="mean")
    assert len(stats) == 3 and fig.axes
    fig = plot_radial_position(obs, "ct", dist_col="dist", cells="glom")
    labels = [t.get_text().split(" (")[0] for t in fig.axes[0].texts]
    assert labels[0] == "M"   # the most central type on top


def test_smide_gsea_and_plots():
    """A planted set (genes ranked at the top of one family) is found; the plots draw."""
    from frzspatial.enrich import smide_gsea
    from frzspatial.pl import plot_gsea_bars, plot_gsea_dotplot

    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(400)]
    rows = []
    for t in ["A", "B"]:
        z = rng.normal(size=400)
        if t == "A":
            z[:25] += 3   # set "up" enriched in A only
        rows.append(pd.DataFrame({"celltype": t, "contrast": "X_vs_Y", "gene": genes, "z": z, "valid": True}))
    res = pd.concat(rows, ignore_index=True)
    sets = {"lib": {"up": genes[:25], "rand1": genes[100:140], "rand2": genes[200:230]}}
    g = smide_gsea(res, sets, min_size=10, permutations=200, threads=1)
    a = g[(g["celltype"] == "A") & (g["term"] == "up")].iloc[0]
    b = g[(g["celltype"] == "B") & (g["term"] == "up")].iloc[0]
    assert a["nes"] > 1.5 and a["fdr"] < 0.05 and b["fdr"] > 0.05 and a["size"] == 25
    assert plot_gsea_dotplot(g, "lib", n_top=3).axes
    assert plot_gsea_bars(g, "A", "X_vs_Y").axes
