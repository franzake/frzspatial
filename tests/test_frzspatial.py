import numpy as np
import pandas as pd
import pytest
import shapely
from spatialdata.transformations import get_transformation

import frzspatial as fs


def test_build_sdata(sdata):
    assert set(sdata.images) == {"fov1_image", "fov2_image"}
    assert set(sdata.labels) == {"fov1_labels", "fov2_labels"}
    table = sdata.tables["table"]
    assert table.shape == (8, 5)
    assert len(sdata.shapes["cell_boundaries"]) == 8
    assert len(sdata.points["transcripts"].compute()) == 24
    m = get_transformation(sdata.images["fov2_image"], "global").to_affine_matrix(("x", "y"), ("x", "y"))
    assert (m[0, 2], m[1, 2], m[1, 1]) == (1100, 2100, -1)  # pixel (0, 0) at (dx, cy), y flipped
    assert "slide" in sdata.coordinate_systems


def test_polygons_contain_centroids(sdata):
    obs = sdata.tables["table"].obs
    shp = sdata.shapes["cell_boundaries"].loc[obs.index]
    assert shapely.contains_xy(shp.geometry.to_numpy(), obs["CenterX_global_px"], obs["CenterY_global_px"]).all()


def test_load_slides(export_dir, tmp_path):
    zarr_dir = tmp_path / "zarr"
    slides = fs.load_slides(["S1"], input_dir=export_dir.parent, zarr_dir=zarr_dir, cache_dir=tmp_path / "cache")
    assert (zarr_dir / "S1_flat.zarr").exists() and slides["S1"].tables["table"].n_obs == 8
    again = fs.load_slides(["S1"], input_dir=export_dir.parent, zarr_dir=zarr_dir, cache_dir=tmp_path / "cache")
    assert again["S1"].tables["table"].n_obs == 8   # reopened from the flat zarr


def test_load_slides_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        fs.load_slides(["nope"], input_dir=tmp_path, zarr_dir=tmp_path / "zarr", cache_dir=tmp_path)


def test_plots(sdata):
    pytest.importorskip("spatialdata_plot")
    import matplotlib

    matplotlib.use("Agg")
    from frzspatial.pl import plot_core

    sdata.tables["table"].obs["manual_annotation"] = pd.Categorical(["Glomerulus"] * 2 + [None] * 6,
                                                                    categories=["Glomerulus", "Artery"])
    pal = {"Glomerulus": "#1f77b4"}
    assert plot_core(sdata, "C1R1", genes={"G1": "magenta"}) is not None
    assert plot_core(sdata, "C1R1", color="manual_annotation", palette=pal) is not None
    # A Series (cells where it is NA grey) is plotted without staying in the table; the legend can go outside
    obs = sdata.tables["table"].obs
    sel = obs["manual_annotation"].where(obs["manual_annotation"] == "Glomerulus").rename("glomeruli")
    ax = plot_core(sdata, "C1R1", color=sel, palette=pal, legend_outside=True, legend_fontsize=7)
    assert "_plot_core" not in obs and ax.get_title().endswith("glomeruli")
    assert ax.get_legend() is None or ax.get_legend().get_texts()[0].get_fontsize() == 7


def test_compute_sbr(sdata):
    import anndata as ad
    import scipy.sparse as sp
    from types import SimpleNamespace

    t = sdata.tables["table"]
    X = sp.hstack([t.X, sp.csr_matrix(np.ones((t.n_obs, 2)))]).tocsr()  # two negative probes, 1 count each
    t2 = ad.AnnData(X, obs=t.obs.copy(), var=pd.DataFrame(index=list(t.var_names) + ["Negative1", "Negative2"]))
    assert fs.compute_sbr(SimpleNamespace(tables={"table": t2}), bandwidth=0.005) is None
    sbr = t2.obs["SBR"].to_numpy()
    assert np.isfinite(sbr).all() and (sbr > 0).all()
    assert {"SBR", "log2SBR", "flag_sbr"} <= set(t2.obs)
    fs.compute_sbr(SimpleNamespace(tables={"table": t2}), bandwidth=1e-6)  # no neighbours: own counts
    assert np.allclose(t2.obs["SBR"], np.asarray(t.X.mean(axis=1)).ravel())

    # Two slides on top of each other (same coordinates): per slide, the same SBR as the slide alone
    fs.compute_sbr(SimpleNamespace(tables={"table": t2}), bandwidth=0.005)
    alone = t2.obs["SBR"].to_numpy()
    one = t2.copy()
    one.obs = one.obs.drop(columns=["slide", "SBR", "log2SBR", "flag_sbr"])
    t3 = ad.concat({"S1": one, "S2": one}, label="slide", index_unique="-")
    s2 = t3.obs["slide"].to_numpy() == "S2"
    t3.X = (sp.diags(np.where(s2, 2.0, 1.0)) @ t3.X).tocsr()  # S2 has twice the counts, so a mix would show
    merged = SimpleNamespace(tables={"table": t3})
    fs.compute_sbr(merged, bandwidth=0.005, slide="S1")
    assert np.allclose(t3.obs["SBR"].to_numpy()[: t2.n_obs], alone) and t3.obs["SBR"].iloc[t2.n_obs:].isna().all()
    fs.compute_sbr(merged, bandwidth=0.005, slide="S2")
    assert np.allclose(t3.obs["SBR"].to_numpy()[t2.n_obs:], alone)  # counts and background both doubled


def test_key_matrix_and_clinical(sdata):
    from frzspatial import add_clinical, cases, key_matrix

    key = key_matrix(pd.DataFrame([[37128.0, None]]))
    assert key.tolist() == [["37128", "control1"]]
    obs = sdata.tables["table"].obs
    obs["sample_id"] = pd.Categorical(np.where(obs["fov"] == 1, "37128", "control1"))
    clinical = cases("S1", ["37128"], "PLA2R")
    add_clinical(sdata, "S1", clinical, ["S1", "S2"])
    assert obs.loc[obs["fov"] == 1, "group"].eq("MN").all() and obs.loc[obs["fov"] == 2, "group"].isna().all()
    assert list(obs["slide"].cat.categories) == ["S1", "S2"] and obs["core_id"].iloc[0] == "S1_C1R1"

    # A merged object: each slide gets its own cases; the other slide's rows and the slide column are kept
    import anndata as ad
    from types import SimpleNamespace

    one = sdata.tables["table"].copy()
    one.obs = one.obs.drop(columns=["slide", "group", "group_mn", "sid", "core_id"])
    merged = ad.concat({"S1": one, "S2": one}, label="slide", index_unique="-")
    m = SimpleNamespace(tables={"table": merged})
    clinical = pd.concat([cases("S1", ["37128"], "PLA2R"), cases("S2", ["37128"], "NELL1")])
    add_clinical(m, "S1", clinical, ["S1", "S2"])
    mo, s2 = merged.obs, merged.obs["slide"].to_numpy() == "S2"
    assert mo.loc[~s2 & (mo["fov"] == 1).to_numpy(), "group_mn"].eq("PLA2R").all() and mo.loc[s2, "group_mn"].isna().all()
    add_clinical(m, "S2", clinical, ["S1", "S2"])
    assert mo.loc[s2 & (mo["fov"] == 1).to_numpy(), "group_mn"].eq("NELL1").all()
    assert mo.loc[~s2 & (mo["fov"] == 1).to_numpy(), "group_mn"].eq("PLA2R").all()
    assert set(mo["slide"]) == {"S1", "S2"} and mo.loc[s2, "core_id"].astype(str).str.startswith("S2_").all()


def test_assign_cores(sdata):
    import matplotlib

    matplotlib.use("Agg")
    from frzspatial import assign_cores, key_matrix, plot_core_grid

    sdata.tables["table"].obs = sdata.tables["table"].obs.drop(columns=["core_str"])
    obs = sdata.tables["table"].obs
    assert assign_cores(sdata, key_matrix([["A", "B"]]), "S1", min_cells=1, fov_size=100) is None
    fovs = sdata.tables["table"].uns["tma_fovs"]["S1"]["fovs"]
    assert fovs.set_index("fov")["core_str"].astype(str).to_dict() == {1: "C1R1", 2: "C2R1"}
    assert obs.groupby("fov", observed=True)["sample_id"].first().astype(str).to_dict() == {1: "A", 2: "B"}
    assign_cores(sdata, key_matrix([["A", "B"]]), "S1", min_cells=1, fov_size=100, fov_overrides={2: "C1R1"})
    assert set(obs["sample_id"]) == {"A"} and set(obs["core_col"]) == {1.0}
    plot_core_grid(by="column", sdata=sdata)
    ax = plot_core_grid(by="core", sdata=sdata, fontsize=10)
    assert {t.get_text() for t in ax.texts} == {"1", "2", "C1R1"}
    fovs = sdata.tables["table"].uns["tma_fovs"]["S1"]["fovs"].copy()   # an explicit table, without its grid
    fovs.attrs = {"slide": "S1", **sdata.tables["table"].uns["tma_fovs"]["S1"]["grid"]}
    # The table kept in uns draws the grid, also after a Zarr round trip
    import tempfile

    import anndata as ad

    tb = sdata.tables["table"]
    assert tb.uns["tma_fovs"]["S1"]["fovs"]["core_str"].astype(str).tolist() == ["C1R1", "C1R1"]
    with tempfile.TemporaryDirectory() as d:
        tb.write_zarr(f"{d}/t.zarr")
        sdata.tables["table"] = ad.read_zarr(f"{d}/t.zarr")
    for by in ("core", "row"):
        ax = plot_core_grid(by=by, sdata=sdata, slide="S1")
        assert [t.get_text() for t in ax.get_xticklabels()] == ["C1", "C2"]
    # Without it: cores from obs, no grid; the sweeps need the table
    del sdata.tables["table"].uns["tma_fovs"]
    ax = plot_core_grid(by="core", sdata=sdata, slide="S1")
    assert {t.get_text() for t in ax.texts} == {"1", "2", "C1R1"}
    with pytest.raises(ValueError, match="give sdata"):
        plot_core_grid(fovs, "core")
    with pytest.raises(ValueError, match="assign_cores worked on"):
        plot_core_grid(by="row", sdata=sdata)


def test_fov_integrity():
    import anndata as ad
    import scipy.sparse as sp
    from frzspatial import fov_integrity, panel_barcodes

    # 6 FOVs of 7 x 7 squares, 12 cells each; FOV 0 keeps 10% of the counts
    rng = np.random.default_rng(0)
    genes = panel_barcodes("Hs_6k")["gene"][:300].tolist() + ["NegPrb1"]
    rows = []
    for f in range(6):
        for gx in range(7):
            for gy in range(7):
                for _ in range(12):
                    rows.append((f, f * 10000 + gx * 600 + rng.uniform(50, 550), gy * 600 + rng.uniform(50, 550)))
    obs = pd.DataFrame(rows, columns=["fov", "CenterX_global_px", "CenterY_global_px"])
    lam = np.where(obs["fov"].to_numpy()[:, None] == 0, 0.1, 1.0) * rng.uniform(0.5, 3, len(genes))[None, :]
    X = sp.csr_matrix(rng.poisson(lam))
    obs.index = obs.index.astype(str)
    table = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))

    class Slide:
        tables = {"table": table}

    assert fov_integrity(Slide, slide="T") is None
    assert table.obs.loc[table.obs["fov"] == 0, "flag_fov_integrity"].all()
    assert table.obs.loc[table.obs["fov"] == 0, "flag_fov_totalcounts"].all()
    assert not table.obs.loc[table.obs["fov"] != 0, "flag_fov_totalcounts"].any()
    assert str(table.obs["fov_signal_loss_cat"].iloc[0]) == "<-2"
    assert list(table.obs["fov_signal_loss_cat"].cat.categories) == ["<-2", "-2:-1", "-1:0", "0:1", "1:2", ">2"]

    # A merged object: slide T next to a healthy copy U; each checked on its own gives T's result and no flags on U
    alone = table.obs.copy()
    healthy = ad.AnnData(X=sp.csr_matrix(rng.poisson(np.ones_like(lam) * lam[1:2])), obs=obs.copy(), var=table.var)
    merged = ad.concat({"T": ad.AnnData(X=X, obs=obs.copy(), var=table.var), "U": healthy}, label="slide",
                       index_unique="-")

    class Merged:
        tables = {"table": merged}

    fov_integrity(Merged, slide="T")
    t = (merged.obs["slide"] == "T").to_numpy()
    for col in ("flag_fov_integrity", "flag_fov_totalcounts", "fov_signal_loss"):
        assert np.allclose(merged.obs.loc[t, col].astype(float), alone[col].astype(float), equal_nan=True), col
    assert not merged.obs.loc[~t, "flag_fov_integrity"].any()   # U not checked yet: False
    fov_integrity(Merged, slide="U")
    assert not merged.obs.loc[~t, "flag_fov_integrity"].any() and merged.obs.loc[t, "flag_fov_integrity"].sum() > 0


def test_qc_flags_and_merge(sdata):
    import matplotlib

    matplotlib.use("Agg")
    from frzspatial import merge_slides, plot_qc, plot_qc_flags, qc_flags

    obs = sdata.tables["table"].obs
    obs["log2SBR"] = np.where(obs["fov"] == 1, 1.0, -1.0)
    obs["flag_fov_integrity"] = False
    obs["SplitRatioToLocal"] = 0.0
    obs["nCount_RNA"], obs["nFeature_RNA"], obs["Area"] = 100.0, 5.0, 400.0
    summary = qc_flags(sdata, count_min=None, area_max=None, sbr_min=0)
    assert obs["flag_overall"].tolist() == (obs["fov"] == 2).tolist()
    assert summary.set_index("filter").loc["Overall", "flagged"] == 4
    plot_qc(sdata, sbr_min=0)
    plot_qc_flags(sdata, title="S1")
    merged = merge_slides({"S1": sdata})
    t = merged.tables["table"]
    assert t.n_obs == 4 and "cell_boundaries-S1" in merged.shapes and list(t.obs["slide"].cat.categories) == ["S1"]
    assert list(t.obs_names) == t.obs["cell"].astype(str).tolist()

    # Merged first, QC per slide, then drop_cells: the same cells kept as merge_slides(drop="flag_overall")
    from frzspatial import drop_cells

    merged = merge_slides({"S1": sdata}, drop=None)
    t = merged.tables["table"]
    t.obs["flag_overall"] = False   # cleared: qc_flags must set it again for this slide
    summary2 = qc_flags(merged, count_min=None, area_max=None, sbr_min=0, slide="S1")
    assert summary2.equals(summary) and t.obs["flag_overall"].sum() == 4
    assert plot_qc(merged, sbr_min=0, slide="S1") is not None and plot_qc_flags(merged, slide="S1") is not None
    drop_cells(merged)
    t = merged.tables["table"]
    assert t.n_obs == 4 and not t.obs["flag_overall"].any()
    assert len(t.uns["dropped_cells"]) == 4 and not set(t.uns["dropped_cells"]) & set(t.obs_names)
    assert sorted(merged.shapes["cell_boundaries-S1"].index) == sorted(t.obs_names)


def test_rtools_roundtrip(tmp_path):
    import shutil
    import subprocess

    import pytest
    import scipy.sparse as sp
    from frzspatial import rtools

    if shutil.which("Rscript") is None:
        pytest.skip("R not installed")
    X = sp.random(30, 7, density=0.3, format="csr", random_state=0) * 10
    rtools.write_counts(tmp_path, "m", X, [f"c{i}" for i in range(30)], [f"g{j}" for j in range(7)])
    code = (f'source("{rtools.R_DIR.joinpath("io.R")}"); m = read_counts("{tmp_path}", "m"); '
            f'stopifnot(dim(m) == c(30, 7), rownames(m)[3] == "c2"); write_counts(m * 2, "{tmp_path}", "m2")')
    subprocess.run(["Rscript", "-e", code], check=True)
    Y, rows, cols = rtools.read_counts(tmp_path, "m2")
    assert rows[3] == "c3" and cols == [f"g{j}" for j in range(7)] and abs(Y - 2 * X).max() < 1e-12


def test_umap_composition_despotx(sdata, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import scipy.io
    import scipy.sparse as sp
    from frzspatial import apply_despotx, despotx_input, merge_slides
    from frzspatial.pl import plot_composition, plot_umap

    obs = sdata.tables["table"].obs
    obs["ist_lineage"] = pd.Categorical(np.where(obs["fov"] == 1, "A", "B"))
    obs["sample_id"], obs["group_mn"] = np.where(obs["fov"] == 1, "s1", "s2"), "MN"
    merged = merge_slides({"S1": sdata}, drop=None)
    t = merged.tables["table"]
    t.obsm["X_umap"] = np.random.default_rng(0).normal(size=(t.n_obs, 2))
    plot_umap(t, "ist_lineage", palette={"A": "#ff0000", "B": "#0000ff"}, legend=True)
    plot_composition(t.obs, "ist_lineage")

    # DeSpotX: input written, then an "output" of halved counts read back
    counts = despotx_input(merged, tmp_path, min_cells=1)
    assert counts.to_dict() == {"A": 4, "B": 4} and (tmp_path / "despotx_in.h5ad").exists()
    out = tmp_path / "despotx_out"
    out.mkdir()
    X = sp.csr_matrix(t.X)
    scipy.io.mmwrite(out / "adjusted.mtx", (X / 2).T)
    pd.Series(t.obs_names).to_csv(out / "cells.tsv", header=False, index=False)
    pd.Series(t.var_names).to_csv(out / "genes.tsv", header=False, index=False)
    pd.DataFrame({"cell_id": t.obs_names, "contamination": 0.1}).to_csv(out / "contamination.tsv", sep="\t", index=False)
    t3 = apply_despotx(merged, tmp_path).tables["table"]
    assert abs(sp.csr_matrix(t3.X) - X / 2).max() < 1e-9 and (t3.layers["counts"] != X).nnz == 0
    assert np.allclose(t3.obs["despotx_eps"], 0.1)
