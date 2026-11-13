"""R tools run from Python: InSituType (cell typing), FastReseg (resegmentation, inputs and results), scPearsonPCA
(batch-corrected PCA), HieraType (cluster markers, marker heatmap), decontX (ambient RNA) and SpNeigh (structures and the rings around them).

Each tool runs in its own ``Rscript`` process on files written here (counts as the CSC components of an R
dgCMatrix, tables as CSV), and its results are read back, so it works the same from a Jupyter kernel, a Quarto render
or reticulate. The R scripts are in ``frzspatial/R``; they need R with Matrix, data.table and the tool's package
(InSituType, FastReseg, Seurat + scPearsonPCA, HieraType + ggplot2, celda, SpNeigh + sf).
"""

import os
import shutil
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._utils import PX_UM, _log

__all__ = ["rscript", "write_counts", "read_counts", "insitutype", "fastreseg_inputs", "load_fastreseg",
           "apply_reseg", "pearson_pca", "cluster_markers", "marker_heatmap", "decontx", "structure_zones", "plot_boundary"]

R_DIR = files("frzspatial").joinpath("R")


def rscript(script, *args, rscript_bin=None):
    """Run one of frzspatial's R scripts (``frzspatial/R/<script>``) with `args`, printing its output.

    `rscript_bin` defaults to ``$FRZSPATIAL_RSCRIPT``, else ``Rscript`` on the PATH.

    Raises
    ------
    RuntimeError
        If R ends with an error (its last lines are in the message).
    """
    exe = rscript_bin or os.environ.get("FRZSPATIAL_RSCRIPT") or shutil.which("Rscript")
    if exe is None:
        raise RuntimeError("Rscript not found: install R, or set FRZSPATIAL_RSCRIPT to its path")
    cmd = [exe, str(R_DIR.joinpath(script)), *map(str, args)]
    tail = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail = (tail + [line])[-20:]
    if proc.returncode:
        raise RuntimeError(f"{script} failed (exit {proc.returncode}):\n{''.join(tail)}")


def write_counts(folder, name, X, rows, cols):
    """Write a sparse matrix for R's ``read_counts(folder, name)``: CSC components and row / column names."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    m = sp.csc_matrix(X)
    m.sum_duplicates()
    if m.nnz >= 2**31 - 1:
        raise ValueError(f"{name}: {m.nnz:,} non-zeros, more than an R dgCMatrix holds")
    m.indices.astype(np.int32).tofile(folder / f"{name}_i.bin")
    m.indptr.astype(np.int32).tofile(folder / f"{name}_p.bin")
    m.data.astype(np.float64).tofile(folder / f"{name}_x.bin")
    (folder / f"{name}_rows.txt").write_text("\n".join(map(str, rows)) + "\n")
    (folder / f"{name}_cols.txt").write_text("\n".join(map(str, cols)) + "\n")


def read_counts(folder, name):
    """A matrix written by R's ``write_counts(m, folder, name)``: (CSR matrix, row names, column names)."""
    folder = Path(folder)
    rows = (folder / f"{name}_rows.txt").read_text().splitlines()
    cols = (folder / f"{name}_cols.txt").read_text().splitlines()
    m = sp.csc_matrix((np.fromfile(folder / f"{name}_x.bin", np.float64), np.fromfile(folder / f"{name}_i.bin", np.int32),
                       np.fromfile(folder / f"{name}_p.bin", np.int32)), shape=(len(rows), len(cols)))
    return m.tocsr(), rows, cols


def _split_probes(tb, layer=None):
    probes = tb.var_names.astype(str)
    neg = probes.str.startswith("Negative")
    gene = ~(neg | probes.str.startswith("SystemControl"))
    X = sp.csr_matrix(tb.layers[layer] if layer else tb.X).astype(np.float64)
    return X, probes, gene, neg


def insitutype(sdata, ref_profiles, if_cols=("Mean.PanCK", "Mean.Membrane", "Mean.DAPI"), anchor_cells=50000,
               ncores=6, seed=1, save_rds=None, table="table", workdir=None, layer=None, round_counts=False):
    """Supervised InSituType cell typing (R package InSituType), as main.Rmd §4.1.

    Steps, in R: cohorts from the immunofluorescence columns (``fastCohorting``); reference profiles rescaled to
    this panel on anchors among `anchor_cells` random cells (``updateReferenceProfiles``), keeping the cell types
    with too few anchors at their atlas profile rescaled per gene by the median ratio of the refitted types; then
    insitutypeML's likelihoods (split across `ncores`), cohort frequencies and the most likely type per cell.

    Parameters
    ----------
    sdata
        Object whose table holds raw counts (probes ``Negative*`` give the background, ``SystemControl*`` are left
        out) and `if_cols`.
    ref_profiles
        Reference profiles (genes x cell types): an ``.rds`` (e.g. CellProfileLibrary's Kidney_HCA), a ``.csv`` or a
        DataFrame (e.g. :func:`frzspatial.reference.reference_profiles` of an atlas).
    if_cols
        obs columns for the cohorts.
    anchor_cells, ncores, seed
        Cells searched for anchors, cores for the likelihoods, random seed.
    save_rds
        Optional ``.rds`` to keep R's results (updated profiles, labels, cohorts), as main.Rmd's
        ``insitutype_usc1.rds``.
    workdir
        Folder for the exchanged files (default: a temporary one, removed afterwards).
    layer, round_counts
        Counts to type: a table layer (default ``X``), rounded to integers with `round_counts` (InSituType's
        likelihoods need integer counts, e.g. for DeSpotX-corrected ones).

    Returns
    -------
    DataFrame
        Per cell (obs order): ``cohort``, ``celltype`` and ``prob`` (posterior of the chosen type).
    """
    tb = sdata.tables[table]
    X, probes, gene, neg = _split_probes(tb, layer)
    if round_counts:
        X.data = np.rint(X.data)
        X.eliminate_zeros()
    tmp = Path(workdir or tempfile.mkdtemp(prefix="frz_ist_"))
    try:
        if isinstance(ref_profiles, pd.DataFrame):  # e.g. frzspatial.reference.reference_profiles()
            ref_profiles.to_csv(tmp / "ref_profiles.csv")
            ref_profiles = tmp / "ref_profiles.csv"
        cells = tb.obs_names.astype(str)
        write_counts(tmp, "x", X[:, gene], cells, probes[gene])
        pd.DataFrame({"cell": cells, "neg": np.asarray(X[:, neg].mean(axis=1)).ravel(),
                      **{c: tb.obs[c].to_numpy(float) for c in if_cols}}).to_csv(tmp / "cells.csv", index=False)
        _log(f"InSituType in R: {len(cells)} cells, {int(gene.sum())} genes")
        rscript("insitutype.R", tmp, Path(ref_profiles).resolve(), anchor_cells, ncores, seed,
                Path(save_rds).resolve() if save_rds else "")
        res = pd.read_csv(tmp / "ist.csv", dtype={"cell": str}).set_index("cell").reindex(cells)
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    return res


def fastreseg_inputs(sdata, clusters, left_out, out_dir, table="table"):
    """Write ``fastreseg_inputs.rds`` for the FastReseg cluster job (``src/run_fastreseg_remote.R``), as main.Rmd §4.2.

    Parameters
    ----------
    sdata
        The merged object: raw counts (control probes are passed as ``ctrl_genes``) and obs ``slide``, ``fov``.
    clusters
        Reference cluster per cell (obs order), e.g. InSituType lineages.
    left_out
        Cell names FastReseg treats as extracellular (cell 0 of each FOV, QC-removed cells).
    out_dir
        Folder of the inputs (created).

    Returns
    -------
    Path
        The ``.rds``: counts (cells x genes, without control probes), clust, left_out, ctrl_genes and the FOVs of
        each slide.
    """
    tb = sdata.tables[table]
    X, probes, gene, _ = _split_probes(tb)
    out_dir = Path(out_dir)
    tmp = Path(tempfile.mkdtemp(prefix="frz_fr_"))
    try:
        cells = tb.obs_names.astype(str)
        write_counts(tmp, "counts", X[:, gene], cells, probes[gene])
        pd.DataFrame({"cell": cells, "clust": np.asarray(clusters, dtype=str)}).to_csv(tmp / "clust.csv", index=False)
        (tmp / "left_out.txt").write_text("\n".join(map(str, left_out)) + "\n")
        (tmp / "ctrl_genes.txt").write_text("\n".join(probes[~gene]) + "\n")
        obs = tb.obs
        fovs = pd.DataFrame({"slide": obs["slide"].astype(str), "fov": obs["fov"].astype(int)}).drop_duplicates()
        fovs["slide"] = pd.Categorical(fovs["slide"], categories=list(obs["slide"].cat.categories)
                                       if hasattr(obs["slide"], "cat") else pd.unique(fovs["slide"]))
        fovs.sort_values(["slide", "fov"]).to_csv(tmp / "fovs.csv", index=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        rscript("fastreseg_inputs.R", tmp, (out_dir / "fastreseg_inputs.rds").resolve())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_dir / "fastreseg_inputs.rds"


def load_fastreseg(out_dir, save_rds=False, cache=True):
    """The FastReseg task results downloaded into `out_dir` (``reseg_task_<t>.rds``), merged in R, as main.Rmd §4.2.

    Every task also resegments the first FOV (for the same distance cutoff), so its duplicates are dropped; the
    number of results is checked against ``fastreseg_ntasks.txt``. Merging takes a few minutes (R reads every task
    file), so with `cache` the result is kept in `out_dir` as ``reseg_cells.parquet`` and ``reseg_counts.npz`` and
    read from there (seconds) until a task file is newer. With `save_rds`, R also saves the merge as
    ``reseg_sdata.rds``.

    Returns
    -------
    (DataFrame, csr_matrix, list)
        FastReseg's cells (``updated_perCellDT``, indexed by ``updated_cellID``), their counts (cells x probes) and
        the probe names.
    """
    out_dir = Path(out_dir)
    cells_f, counts_f = out_dir / "reseg_cells.parquet", out_dir / "reseg_counts.npz"
    tasks = list(out_dir.glob("reseg_task_*.rds"))
    fresh = cells_f.exists() and counts_f.exists() and all(
        f.stat().st_mtime < min(cells_f.stat().st_mtime, counts_f.stat().st_mtime) for f in tasks)
    if cache and fresh and not save_rds:
        cells = pd.read_parquet(cells_f)
        z = np.load(counts_f, allow_pickle=False)
        X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        probes = list(z["probes"])
        _log(f"FastReseg (cached {cells_f.name}): {len(cells)} cells")
    else:
        _log(f"FastReseg: merging {len(tasks)} task results in R (a few minutes)")
        tmp = Path(tempfile.mkdtemp(prefix="frz_fr_"))
        try:
            rscript("load_fastreseg.R", out_dir.resolve(), tmp, "TRUE" if save_rds else "FALSE")
            cells = pd.read_csv(tmp / "cells.csv", dtype={"updated_cellID": str}).set_index("updated_cellID")
            X, rows, probes = read_counts(tmp, "exprs")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if list(rows) != list(cells.index):
            raise ValueError("FastReseg's cells and counts are not in the same order")
        if cache:
            cells.to_parquet(cells_f)
            np.savez(counts_f, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
                     probes=np.array(probes))
    _log(f"FastReseg: {len(cells)} cells, {cells['reSeg_action'].value_counts(dropna=False).to_dict()}")
    return cells, X, probes


def apply_reseg(sdata, cells, X, probes, px_um=None, table="table"):
    """A copy of the merged object with FastReseg's cells, as main.Rmd §4.3.

    Split children (``<parent>_g<n>``) inherit their parent's metadata, sit at their transcripts' mean and have no
    polygon; counts, ``nCount_RNA`` and ``nFeature_RNA`` are recomputed; obs gets ``parent_cell_id``,
    ``reseg_child``, ``reSeg_action`` and ``reseg_celltype``.

    Parameters
    ----------
    sdata
        The merged object FastReseg ran on (shapes ``cell_boundaries-<slide>``, obs ``slide``).
    cells, X, probes
        From :func:`load_fastreseg`.
    px_um
        µm per px (FastReseg's coordinates are in µm); default :data:`frzspatial.PX_UM`.

    Returns
    -------
    SpatialData
    """
    import anndata as ad
    from spatialdata.models import TableModel

    from ._utils import PX_UM

    px_um = px_um or PX_UM
    tb = sdata.tables[table]
    idx = pd.Index(probes).get_indexer(tb.var_names)
    if (idx < 0).any():
        raise ValueError(f"{(idx < 0).sum()} probes of the table are missing from FastReseg's counts")
    X = sp.csr_matrix(X)[:, idx]
    gene = ~(tb.var_names.str.startswith("Negative") | tb.var_names.str.startswith("SystemControl"))

    # 0. Metadata from the parent cell, then the recomputed columns
    parent = cells.index.str.replace(r"_g[0-9]+$", "", regex=True)
    obs = tb.obs.reindex(parent).set_axis(cells.index).assign(
        cell=cells.index, parent_cell_id=parent, reseg_child=cells.index != parent,
        reSeg_action=cells["reSeg_action"].to_numpy(), reseg_celltype=cells["updated_celltype"].to_numpy(),
        nCount_RNA=np.asarray(X[:, gene].sum(axis=1)).ravel(),
        nFeature_RNA=np.asarray((X[:, gene] > 0).sum(axis=1)).ravel())
    child = obs["reseg_child"].to_numpy()
    xy = ["CenterX_global_px", "CenterY_global_px"]
    obs[xy] = obs[xy].astype(float)  # children sit between pixels (pandas 3 does not upcast integer columns)
    obs.loc[child, xy] = cells.loc[child, ["x", "y"]].to_numpy() / px_um

    # 1. The new table; only cells that were not split keep their polygons
    slides = list(tb.obs["slide"].cat.categories)
    regions = [f"cell_boundaries-{s}" for s in slides]
    obs["region"] = pd.Categorical("cell_boundaries-" + obs["slide"].astype(str), categories=regions)
    if obs["region"].isna().any():
        raise ValueError("FastReseg cells without a parent in the table")
    out = sdata.subset([name for _, name, _ in sdata.gen_elements()])
    out.tables[table] = TableModel.parse(
        ad.AnnData(X=X, obs=obs, var=tb.var,
                   obsm={"spatial": obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)}),
        region=regions, region_key="region", instance_key="cell")
    for region in regions:
        shapes = sdata.shapes[region]
        out.shapes[region] = shapes[shapes.index.isin(obs.index)]
    return out


def pearson_pca(sdata, batch_vars=("sample_id",), nfeatures=2000, npcs=50, ncores=8, exclude_genes=None,
                table="table"):
    """Batch-corrected Pearson-residual PCA (R packages Seurat and scPearsonPCA), as main.Rmd §5.1.

    Seurat's vst variable genes, per-batch gene frequencies and scPearsonPCA's quasi-Poisson Pearson-residual PCA with
    the residuals standardised within each batch (integration). Control probes (``Negative*``, ``SystemControl*``)
    are left out.

    Parameters
    ----------
    sdata
        Object whose table holds raw counts.
    batch_vars
        obs columns whose combination is the batch (e.g. ``("slide", "sample_id")``).
    nfeatures, npcs, ncores
        Variable genes, principal components, cores.
    exclude_genes
        Genes never chosen as variable genes.

    Returns
    -------
    list
        The variable genes. Sets ``obsm["X_pca"]`` and ``obs["batch"]`` in place.
    """
    tb = sdata.tables[table]
    X, probes, gene, _ = _split_probes(tb)
    cells = tb.obs_names.astype(str)
    batch = tb.obs[list(batch_vars)].astype(str).agg("-".join, axis=1)
    tmp = Path(tempfile.mkdtemp(prefix="frz_pca_"))
    try:
        write_counts(tmp, "x", X[:, gene], cells, probes[gene])
        pd.DataFrame({"cell": cells, "batch": batch.to_numpy()}).to_csv(tmp / "batch.csv", index=False)
        if exclude_genes:
            (tmp / "exclude_genes.txt").write_text("\n".join(exclude_genes) + "\n")
        _log(f"scPearsonPCA in R: {len(cells)} cells, {batch.nunique()} batches")
        rscript("pearson_pca.R", tmp, nfeatures, npcs, ncores)
        shape = tuple(int(v) for v in (tmp / "pca_dim.txt").read_text().split())
        pca = np.fromfile(tmp / "pca.bin", np.float64).reshape(shape, order="F")
        hvgs = (tmp / "hvgs.txt").read_text().splitlines()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tb.obsm["X_pca"] = pca
    tb.obs["batch"] = pd.Categorical(batch.to_numpy())
    return hvgs


def cluster_markers(sdata, cluster_column="clusters", n_top=None, pseudocount=0.025, table="table"):
    """Markers of each cluster against all other cells (R package HieraType), as main.Rmd §7.1.

    Fold changes on total-count-normalised gene counts (``HieraType::clusterwise_foldchange_metrics``), ranked by
    ``prioritystat`` = (cluster_expr + pseudocount) / (clusterprime_expr + pseudocount), which penalises low
    expressers.

    Parameters
    ----------
    sdata
        Object whose table holds counts and `cluster_column` (categorical: its order is the clusters' order).
    n_top
        Genes kept per cluster in ``top`` (None: all).

    Returns
    -------
    dict
        ``markers`` (every gene and cluster), ``top`` (the `n_top` best per cluster) and ``mean_counts`` (mean counts
        per cell of the top genes in each cluster, genes x clusters).
    """
    tb = sdata.tables[table]
    X, probes, gene, _ = _split_probes(tb)
    cells = tb.obs_names.astype(str)
    cl = tb.obs[cluster_column]
    levels = [str(c) for c in (cl.cat.categories if hasattr(cl, "cat") else sorted(cl.astype(str).unique()))]
    tmp = Path(tempfile.mkdtemp(prefix="frz_mk_"))
    try:
        write_counts(tmp, "x", X[:, gene], cells, probes[gene])
        pd.DataFrame({"cell": cells, "cluster": cl.astype(str).to_numpy()}).to_csv(tmp / "clusters.csv", index=False)
        (tmp / "levels.txt").write_text("\n".join(levels) + "\n")
        _log(f"HieraType markers in R: {len(levels)} clusters of {cluster_column}")
        rscript("cluster_markers.R", tmp, "all" if n_top is None else n_top, pseudocount)
        out = {k: pd.read_csv(tmp / f"{k}.csv", dtype={"cluster": str}) for k in ("markers", "top", "mean_counts")}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for k in ("markers", "top"):
        out[k]["cluster"] = pd.Categorical(out[k]["cluster"], categories=levels)
    return out


def decontx(sdata, cluster_column, batch_column="slide", max_iter=500, delta=(10, 10), estimate_delta=True, seed=12345,
            table="table", layer=None):
    """Ambient RNA removal by decontX (R package celda), with clusters given.

    Each cell's counts are modelled as a mixture of its cluster's expression profile and a contamination profile
    from the other clusters, estimated separately in each batch (``celda::decontX`` with ``z`` and ``batch``).

    Parameters
    ----------
    sdata
        Object whose table holds raw counts (probes ``Negative*`` / ``SystemControl*`` are left out), `cluster_column`
        and `batch_column`.
    cluster_column
        obs column with the clusters (``z``).
    batch_column
        obs column with the batches, each with its own contamination profile (None: one batch).
    max_iter, delta, estimate_delta, seed
        decontX's ``maxIter``, ``delta`` (beta prior of the contamination), ``estimateDelta`` and ``seed``.
    layer
        Table layer with the counts (default ``X``).

    Returns
    -------
    dict
        ``counts`` (CSR, cells x genes: decontaminated counts, not integers), ``genes`` (their names) and
        ``contamination`` (Series: each cell's estimated contaminated fraction, indexed like the table).
    """
    tb = sdata.tables[table]
    X, probes, gene, _ = _split_probes(tb, layer)
    cells = tb.obs_names.astype(str)
    batch = tb.obs[batch_column].astype(str).to_numpy() if batch_column else np.repeat("all", len(cells))
    tmp = Path(tempfile.mkdtemp(prefix="frz_dx_"))
    try:
        write_counts(tmp, "x", X[:, gene], cells, probes[gene])
        pd.DataFrame({"cell": cells, "cluster": tb.obs[cluster_column].astype(str).to_numpy(), "batch": batch}).to_csv(
            tmp / "cells.csv", index=False)
        _log(f"decontX in R: {len(cells):,} cells, clusters from {cluster_column}")
        rscript("decontx.R", tmp, max_iter, delta[0], delta[1], int(bool(estimate_delta)), seed)
        counts, rows, genes = read_counts(tmp, "decont")
        cont = pd.read_csv(tmp / "contamination.csv", dtype={"cell": str}).set_index("cell")["contamination"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    order = pd.Index(rows).get_indexer(cells)
    return {"counts": counts[order], "genes": list(genes), "contamination": cont.reindex(cells).set_axis(tb.obs_names)}


def marker_heatmap(markers, blocks, width=22, height=9, out_png=None, show=True):
    """HieraType's marker heatmap (R, ggplot2) of fold changes, genes in named blocks, as main.Rmd §7.3.

    Parameters
    ----------
    markers
        ``cluster_markers(...)["markers"]`` (its ``cluster`` categories set the row order).
    blocks
        ``{block name: [genes]}``, drawn left to right with dashed lines between blocks.
    width, height
        Size in inches.
    out_png
        Where to keep the image (default: a temporary file).
    show
        Draw it with matplotlib (``plt.show()`` follows).

    Returns
    -------
    Path
        The PNG.
    """
    tmp = Path(tempfile.mkdtemp(prefix="frz_hm_"))
    out = Path(out_png) if out_png else tmp / "marker_heatmap.png"
    try:
        markers.to_csv(tmp / "markers.csv", index=False)
        (tmp / "levels.txt").write_text("\n".join(map(str, markers["cluster"].cat.categories)) + "\n")
        pd.DataFrame([(b, g) for b, genes in blocks.items() for g in genes], columns=["block", "gene"]) \
            .to_csv(tmp / "blocks.csv", index=False)
        rscript("marker_heatmap.R", tmp, out.resolve(), width, height)
        if show:
            import matplotlib.pyplot as plt

            img = plt.imread(out)
            fig = plt.figure(figsize=(width, height))
            fig.add_axes([0, 0, 1, 1]).imshow(img)
            fig.axes[0].axis("off")
    finally:
        if out_png:
            shutil.rmtree(tmp, ignore_errors=True)
    return out


def structure_zones(table, label, types, eps_um=35, min_cells=20, rings_um=(15, 30),
                    zones=("inside", "ring1", "ring2"), slide_key="slide", core_key="core_id", pad_um=1, outlier_um=30,
                    max_dist_um=100, px_um=PX_UM):
    """Multicellular structures and the cells inside and around them (R package SpNeigh), e.g. glomeruli from their
    tuft cells, with Bowman's capsule as the first ring.

    Per slide, in R: SpNeigh's ``getBoundary`` drops cells of `types` whose mean distance to their 5 nearest such
    cells exceeds `outlier_um` (its default 30; ``np.inf`` keeps all) and groups the rest by DBSCAN (`eps_um`, 4 points); each group is outlined by its convex hull padded by `pad_um` (SpNeigh's concave outline cuts into compact
    structures) and kept with at least `min_cells` of `types` inside; ``getCellsInside`` then assigns every cell to the
    structure it lies in, the ring 0-``rings_um[0]`` outside it or the ring ``rings_um[0]``-``rings_um[1]``
    (``getRingRegion``), the first match winning.

    Parameters
    ----------
    table
        AnnData with obs `label`, `slide_key`, `core_key` and ``obsm["spatial"]`` in global px.
    types
        Labels whose cells make the structures (e.g. ``["Podocytes", "Mesangial", "Glom_EC"]``).
    zones
        Names of the three zones: inside, first ring, second ring.
    max_dist_um
        Cells further than this from every outline get no ``dist``.

    Returns
    -------
    zones : DataFrame
        Per cell (obs order): ``zone`` (categorical, NA away from every structure), ``unit``, the structure
        (``<core>_<n>``, numbered by size within the core), and ``dist``, the signed distance (µm) to the nearest
        structure's convex outline of the same slide (negative inside; NA beyond `max_dist_um`).
    outlines : dict
        ``{unit: shapely Polygon}`` in µm.
    """
    import shapely

    xy = np.asarray(table.obsm["spatial"], float) * px_um
    cells = table.obs_names.astype(str)
    member = table.obs[label].astype(str).isin(list(types)).to_numpy()
    tmp = Path(tempfile.mkdtemp(prefix="frz_sz_"))
    try:
        pd.DataFrame({"cell": cells, "slide": table.obs[slide_key].astype(str).to_numpy(), "x": xy[:, 0], "y": xy[:, 1],
                      "member": member.astype(int)}).to_csv(tmp / "cells.csv", index=False)
        _log(f"SpNeigh structures in R: {member.sum():,} cells of {len(types)} types")
        rscript("structure_zones.R", tmp, eps_um, 4, min_cells, pad_um, "Inf" if np.isinf(outlier_um) else outlier_um, *rings_um)
        z = pd.read_csv(tmp / "zones.csv", dtype={"cell": str}).set_index("cell").reindex(cells)
        lines = pd.read_csv(tmp / "outlines.csv")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    inside = z[z["zone"] == "inside"].join(pd.Series(table.obs[core_key].astype(str).to_numpy(), index=cells, name="core"))
    core = inside.groupby("region_id")["core"].agg(lambda c: c.mode()[0])
    size = inside.groupby("region_id").size()
    names = {r: f"{c}_{n}" for c, rs in core.groupby(core) for n, r in enumerate(size[rs.index].sort_values(ascending=False).index, 1)}
    out = pd.DataFrame({"zone": pd.Categorical(z["zone"].map(dict(zip(("inside", "ring1", "ring2"), zones))), categories=list(zones)),
                        "unit": z["region_id"].map(names).astype("category")}, index=table.obs_names)
    outlines = {names[r]: shapely.Polygon(d[["x", "y"]].to_numpy()) for r, d in lines.groupby("region_id") if r in names}
    # signed distance to the nearest outline of the same slide, measured from the convex hull (the padding added back)
    slide = table.obs[slide_key].astype(str).to_numpy()
    dist = np.full(len(cells), np.inf)
    for r, u in names.items():
        poly = outlines[u]
        x0, y0, x1, y1 = poly.bounds
        i = np.flatnonzero((slide == r.rsplit("_r", 1)[0]) & (xy[:, 0] > x0 - max_dist_um) & (xy[:, 0] < x1 + max_dist_um)
                           & (xy[:, 1] > y0 - max_dist_um) & (xy[:, 1] < y1 + max_dist_um))
        d = shapely.distance(poly.boundary, shapely.points(xy[i])) * np.where(shapely.contains_xy(poly, xy[i, 0], xy[i, 1]), -1, 1) + pad_um
        dist[i] = np.minimum(dist[i], d)
    dist[(out["zone"] == zones[0]).to_numpy()] = np.minimum(dist[(out["zone"] == zones[0]).to_numpy()], 0)  # cells in the padding: on the outline
    out["dist"] = np.where(dist <= max_dist_um, dist, np.nan)
    _log(f"{len(outlines)} structures; cells: " + ", ".join(f"{k} {v:,}" for k, v in out["zone"].value_counts(sort=False).items()))
    return out, outlines


def plot_boundary(table, label, outlines, core, palette=None, rings_um=(15, 30), core_key="core_id", point_size=1.2,
                  width=8, height=8, out_png=None, show=True, px_um=PX_UM):
    """One core's cells and structure outlines as SpNeigh's ``plotBoundary`` draws them (R, ggplot2): cells as points
    coloured by `label`, each outline in black and its rings (`rings_um`) dashed.

    Parameters
    ----------
    table
        AnnData with obs `label`, `core_key` and ``obsm["spatial"]`` in global px.
    outlines
        ``{unit: shapely Polygon}`` in µm (e.g. from :func:`structure_zones`); those of `core` are drawn.
    core
        The core (``obs[core_key]``).
    palette
        ``{label value: colour}`` (default: SpNeigh's colours).
    width, height
        Size in inches.
    out_png, show
        Where to keep the image (default: a temporary file); draw it with matplotlib (``plt.show()`` follows).

    Returns
    -------
    Path
        The PNG.
    """
    tmp = Path(tempfile.mkdtemp(prefix="frz_pb_"))
    out = Path(out_png) if out_png else tmp / "plot_boundary.png"
    try:
        sel = (table.obs[core_key].astype(str) == str(core)).to_numpy()
        xy = np.asarray(table.obsm["spatial"], float)[sel] * px_um
        lab = table.obs[label][sel]
        levels = [str(c) for c in (lab.cat.remove_unused_categories().cat.categories if hasattr(lab, "cat") else sorted(lab.astype(str).unique()))]
        pd.DataFrame({"cell": table.obs_names[sel].astype(str), "x": xy[:, 0], "y": xy[:, 1], "cluster": lab.astype(str).to_numpy()}) \
            .to_csv(tmp / "cells.csv", index=False)
        pd.DataFrame({"cluster": levels, "color": [(palette or {}).get(c, "") for c in levels]}).to_csv(tmp / "colors.csv", index=False)
        pd.concat([pd.DataFrame({"region_id": u, "x": np.asarray(p.exterior.coords)[:, 0], "y": np.asarray(p.exterior.coords)[:, 1]})
                   for u, p in outlines.items() if u.startswith(f"{core}_")] or [pd.DataFrame(columns=["region_id", "x", "y"])]) \
            .to_csv(tmp / "outlines.csv", index=False)
        rscript("plot_boundary.R", tmp, out.resolve(), point_size, width, height, f"{core}", *rings_um)
        if show:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(width, height))
            fig.add_axes([0, 0, 1, 1]).imshow(plt.imread(out))
            fig.axes[0].axis("off")
    finally:
        if out_png:
            shutil.rmtree(tmp, ignore_errors=True)
    return out
