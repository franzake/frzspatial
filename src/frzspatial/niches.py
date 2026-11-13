"""Cellular neighbourhoods and niches of annotated cells (squidpy, scanpy), as main.Rmd §8."""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._utils import PX_UM, _log

__all__ = ["neighbourhood_graph", "compute_niches", "niche_composition", "name_niches", "niche_palette"]


def neighbourhood_graph(table, label="celltypes_dx", exclude_prefix="Low_quality", radius_um=50,
                        obs_cols=("slide", "sample_id", "core_id", "group", "group_mn"), slide_key="slide"):
    """A light table of the annotated cells (no counts) with each cell's neighbours within `radius_um`, per slide.

    Parameters
    ----------
    table
        AnnData with obs `label` and ``obsm["spatial"]`` in global px (e.g. the usc3 table).
    exclude_prefix
        Cells whose label starts with it are left out (None: keep all).
    radius_um
        Neighbourhood radius (50 µm: around a tubule or glomerulus).
    obs_cols
        obs columns kept besides `label`.

    Returns
    -------
    AnnData
        With ``obsp["spatial_connectivities"]`` (squidpy's radius graph, one per slide) and the label colours of
        `table` (``uns[f"{label}_colors"]``) when it has them.
    """
    import anndata as ad
    import squidpy as sq

    keep = np.ones(table.n_obs, bool)
    if exclude_prefix:
        keep = ~table.obs[label].astype(str).str.startswith(exclude_prefix).to_numpy()
    cols = [label] + [c for c in obs_cols if c in table.obs]
    nb = ad.AnnData(obs=table.obs.loc[keep, cols].copy(), obsm={"spatial": np.asarray(table.obsm["spatial"])[keep]})
    nb.obs[label] = nb.obs[label].cat.remove_unused_categories()
    nb.obs[slide_key] = nb.obs[slide_key].cat.remove_unused_categories()  # a graph per slide present
    if f"{label}_colors" in table.uns:
        pal = dict(zip(table.obs[label].cat.categories, table.uns[f"{label}_colors"]))
        nb.uns[f"{label}_colors"] = [pal[c] for c in nb.obs[label].cat.categories]
    sq.gr.spatial_neighbors_radius(nb, library_key=slide_key, radius=radius_um / PX_UM)
    deg = np.asarray(nb.obsp["spatial_connectivities"].sum(1)).ravel()
    _log(f"{nb.n_obs:,} cells; neighbours within {radius_um} µm: median {np.median(deg):.0f}, "
         f"{(deg == 0).mean():.1%} with none")
    return nb


def compute_niches(nb, label="celltypes_dx", resolution=0.2, n_neighbors=15, min_cells=200, key_added="niche"):
    """Niches: Leiden clusters of each cell's neighbourhood composition, as squidpy's ``calculate_niche``.

    Each cell's profile is the label composition of its neighbours (:func:`neighbourhood_graph`; cells without any:
    their own label), standardised per label (``calculate_niche``'s "neighborhood" flavour, ``scale=True``); niches
    are Leiden clusters of the profiles' kNN graph, with igraph (much faster than ``calculate_niche``'s leidenalg).
    Niches with fewer than `min_cells` cells become ``not_a_niche``.

    Returns
    -------
    Series
        Cells per niche. Sets ``nb.obs[key_added]`` in place.
    """
    import anndata as ad
    import scanpy as sc
    from sklearn.preprocessing import StandardScaler

    onehot = sp.csr_matrix(pd.get_dummies(nb.obs[label]).to_numpy(float))
    profile = nb.obsp["spatial_connectivities"] @ onehot
    n_nb = np.asarray(profile.sum(1)).ravel()
    profile = sp.csr_matrix(profile.multiply(1 / np.maximum(n_nb, 1)[:, None]))
    profile[n_nb == 0] = onehot[n_nb == 0]
    prof = ad.AnnData(X=StandardScaler().fit_transform(profile.toarray()), obs=nb.obs[[]].copy())
    sc.pp.neighbors(prof, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.leiden(prof, resolution=resolution, flavor="igraph", n_iterations=2, key_added=key_added)
    niche = prof.obs[key_added].astype(str)
    small = niche.value_counts()[lambda n: n < min_cells].index
    niche[niche.isin(small)] = "not_a_niche"
    nb.obs[key_added] = pd.Categorical(niche.to_numpy())
    return nb.obs[key_added].value_counts()


def niche_composition(nb, niche="niche", label="celltypes_dx", pseudo=1e-3):
    """Label share of each niche, log2 enrichment of each label over its overall share, and each niche's three most
    enriched labels.

    Returns
    -------
    (DataFrame, DataFrame, Series)
        ``comp`` (niches x labels, rows sum to 1), ``enrich`` and ``top`` (``"A / B / C"`` per niche).
    """
    comp = pd.crosstab(nb.obs[niche], nb.obs[label], normalize="index")
    enrich = np.log2((comp + pseudo) / (nb.obs[label].value_counts(normalize=True)[comp.columns] + pseudo))
    top = enrich.apply(lambda r: " / ".join(r.sort_values(ascending=False).index[:3]), axis=1)
    return comp, enrich, top


def name_niches(nb, names, top, niche="niche", key_added="niche_label"):
    """Hand-written niche names into ``obs[key_added]``; unnamed niches become ``"<niche>: <top labels>"``.

    Parameters
    ----------
    names
        ``{niche number: name}``; niches sharing a name are one niche afterwards.
    top
        From :func:`niche_composition`.

    Returns
    -------
    list
        The niches still unnamed.
    """
    niches = list(top.index)
    auto = {n: f"{n}: {top[n]}" for n in niches if n != "not_a_niche"}
    mapping = {**auto, **names, "not_a_niche": "not_a_niche"}
    order = list(dict.fromkeys(list(names.values()) + [auto[n] for n in niches if n in auto and n not in names]
                               + ["not_a_niche"]))
    nb.obs[key_added] = pd.Categorical(nb.obs[niche].astype(str).map(mapping), categories=order)
    unnamed = sorted(set(niches) - set(names) - {"not_a_niche"}, key=lambda n: int(n) if n.isdigit() else 0)
    _log(f"{len(names)} niches named, {len(unnamed)} still tagged by their top labels: {unnamed}")
    return unnamed


def niche_palette(categories, none="not_a_niche"):
    """``{niche: colour}``, fixed over all niches (scanpy's default_102), `none` grey."""
    from scanpy.pl.palettes import default_102

    pal = dict(zip(categories, default_102 * (len(categories) // 102 + 1)))
    if none in pal:
        pal[none] = "#BDBDBD"
    return pal
