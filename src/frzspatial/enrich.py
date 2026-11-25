"""Gene set enrichment of differential-expression results (e.g. ``frzspatial.de.smide``): preranked GSEA per family.

Why preranked GSEA, ranked on the DE statistic of every tested gene (``z = beta / SE``), rather than an over-
representation test of the significant genes: imaging panels (CosMx 6K, Xenium) measure a selected subset of genes, so
a genome-wide background would make any panel-rich pathway look enriched; the ranked list is exactly the genes tested in
that family (the right universe by construction), no FDR cut-off is needed, and families with few significant genes
(small cell types, weak contrasts) still give a result. Gene sets are counted on their genes in the ranked list
(``min_size`` / ``max_size``), so the scores are relative to the panel.

Gene sets come from Enrichr's libraries (gseapy), e.g. ``MSigDB_Hallmark_2020`` (50 broad, little redundant: the
first look), ``GO_Biological_Process_2025`` and ``Reactome_Pathways_2024`` (detail); ``gene_sets`` caches them as
JSON so a rerun is offline and reproducible.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ._utils import _log

__all__ = ["gene_sets", "smide_gsea", "gene_themes", "THEMES", "cluster_enrichment", "cluster_labels"]

DEFAULT_LIBRARIES = ("MSigDB_Hallmark_2020", "GO_Biological_Process_2025", "Reactome_Pathways_2024")

#: Broad programmes for annotating genes, each defined by words matched (case-insensitive) in gene set names.
THEMES = {
    "Basement membrane / ECM": ["extracellular matrix", "ecm ", "collagen", "basement membrane", "proteoglycan",
                                "epithelial mesenchymal", "extracellular structure", "encapsulating structure"],
    "Adhesion / integrins": ["adhesion", "integrin", "junction", "focal"],
    "Interferon / antigen presentation": ["interferon", "antigen", "mhc", "allograft", "immune"],
    "Coagulation / wound healing": ["coagulation", "wound", "platelet", "hemostasis"],
    "Angiogenesis / endothelium": ["angiogenesis", "vasculature", "endothelial", "vegf"],
    "Stress / TNF / hypoxia": ["tnf", "nf-kb", "hypoxia", "il-6", "stress", "glucocorticoid", "unfolded"],
}
THEME_COLORS = {"Basement membrane / ECM": "#8C510A", "Adhesion / integrins": "#D8B365",
                "Interferon / antigen presentation": "#762A83", "Coagulation / wound healing": "#E7298A",
                "Angiogenesis / endothelium": "#1B9E77", "Stress / TNF / hypoxia": "#666666"}


def gene_sets(libraries=DEFAULT_LIBRARIES, organism="Human", cache_dir=None):
    """Gene set libraries as ``{library: {term: [genes]}}``, from Enrichr (gseapy), cached as JSON in `cache_dir`.

    A library already in `cache_dir` is read from there (no download), so results stay reproducible once cached.
    """
    import gseapy as gp

    out = {}
    cache = Path(cache_dir) if cache_dir else None
    for lib in libraries:
        f = cache / f"{lib}.json" if cache else None
        if f is not None and f.exists():
            out[lib] = json.loads(f.read_text())
            continue
        out[lib] = {k: list(v) for k, v in gp.get_library(lib, organism=organism).items()}
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(out[lib]))
        _log(f"gene sets: {lib}, {len(out[lib])} sets")
    return out


def smide_gsea(res, sets, stat="z", celltypes=None, contrasts=None, min_size=10, max_size=500, permutations=1000,
               seed=0, threads=4):
    """Preranked GSEA of every family (cell type x contrast) of a DE table, for every gene set library.

    Parameters
    ----------
    res
        DE results with columns ``celltype``, ``contrast``, ``gene``, `stat` and ``valid`` (``smide``'s table).
    sets
        ``{library: {term: [genes]}}`` (``gene_sets``).
    stat
        Ranking statistic: ``"z"`` (default, signed: positive = higher in the first group named) or e.g. ``"beta"``.
    celltypes, contrasts
        Families to test (default: all).
    min_size, max_size
        Gene sets with fewer / more genes among the ranked (tested) genes are skipped.
    permutations, seed, threads
        Gene-set permutations of gseapy's prerank, their seed, and threads.

    Returns
    -------
    DataFrame
        One row per family x library x gene set: ``nes``, ``es``, ``pval``, ``fdr`` (BH within the family and library,
        gseapy's), ``size`` (set genes among the ranked), ``leading_edge`` (``;``-separated), ``n_ranked``.
    """
    import gseapy as gp

    d = res[res["valid"].astype(bool) & res[stat].notna()]
    rows = []
    for (t, c), f in d.groupby(["celltype", "contrast"], sort=False):
        if (celltypes is not None and t not in celltypes) or (contrasts is not None and c not in contrasts):
            continue
        rnk = f.drop_duplicates("gene").set_index("gene")[stat].astype(float).sort_values(ascending=False)
        for lib, gs in sets.items():
            try:
                r = gp.prerank(rnk=rnk, gene_sets=gs, min_size=min_size, max_size=max_size,
                               permutation_num=permutations, seed=seed, threads=threads, outdir=None,
                               verbose=False).res2d
            except Exception as e:  # noqa: BLE001  (e.g. no set within min_size / max_size)
                _log(f"gsea: {t} / {c} / {lib}: {type(e).__name__}: {e}")
                continue
            tag = r["Tag %"].astype(str).str.split("/")
            rows.append(pd.DataFrame({
                "celltype": t, "contrast": c, "library": lib, "term": r["Term"].astype(str),
                "nes": r["NES"].astype(float), "es": r["ES"].astype(float),
                "pval": r["NOM p-val"].astype(float), "fdr": r["FDR q-val"].astype(float),
                "size": tag.str[1].astype(int), "leading_edge": r["Lead_genes"].astype(str), "n_ranked": len(rnk)}))
        n = sum(len(x) for x in rows if x["celltype"].iloc[0] == t and x["contrast"].iloc[0] == c)
        _log(f"gsea: {t} / {c}: {len(rnk):,} ranked genes, {n} gene sets tested")
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out.sort_values(["celltype", "contrast", "library", "fdr", "nes"], ascending=[True, True, True, True, False],
                           ignore_index=True) if len(out) else out


def gene_themes(gsea, themes=None, fdr=0.05, celltypes=None, contrasts=None, libraries=None):
    """One broad programme per gene, from the leading edges of the significant gene sets.

    A gene set belongs to a theme when its name contains one of the theme's words (``THEMES`` by default: basement
    membrane / ECM, adhesion, interferon / antigen presentation, coagulation, angiogenesis, stress); each gene goes to
    the theme whose significant sets (FDR < `fdr`) list it most often in their leading edge (ties: the theme listed
    first). Genes in no such leading edge are left out.

    Parameters
    ----------
    gsea
        ``smide_gsea``'s table.
    themes
        ``{theme: [words]}`` (default ``THEMES``).
    celltypes, contrasts, libraries
        Restrict the gene sets used (default: all).

    Returns
    -------
    Series
        Theme per gene (categorical, in the themes' order), named ``"theme"``.
    """
    themes = themes or THEMES
    g = gsea[gsea["fdr"] < fdr]
    if celltypes is not None:
        g = g[g["celltype"].isin(celltypes)]
    if contrasts is not None:
        g = g[g["contrast"].isin(contrasts)]
    if libraries is not None:
        g = g[g["library"].isin(libraries)]
    counts = {}
    names = g["term"].astype(str).str.lower()
    for k, (theme, words) in enumerate(themes.items()):
        hit = names.apply(lambda n: any(w in n for w in words)).to_numpy()
        for le in g.loc[hit, "leading_edge"].astype(str):
            for gene in le.split(";"):
                if gene:
                    counts.setdefault(gene, np.zeros(len(themes)))[k] += 1
    cats = list(themes)
    out = pd.Series({gene: cats[int(np.argmax(c))] for gene, c in counts.items()}, name="theme", dtype=object)
    return out.astype(pd.CategoricalDtype(cats))


def cluster_enrichment(clusters, sets, background, min_size=5, max_size=500):
    """Over-representation of gene sets in each gene cluster (hypergeometric test), within a background.

    For annotating clusters of a heatmap (``frzspatial.pl.smide_heatmap_clusters``): a cluster is a short gene list
    without a ranking, so an over-representation test, against the genes that could have been in the heatmap (e.g.
    every gene tested in at least one smiDE family: a targeted panel, not the genome).

    Parameters
    ----------
    clusters
        Series gene -> cluster.
    sets
        ``{library: {term: [genes]}}`` (``gene_sets``).
    background
        The universe of genes (cluster genes outside it are dropped).
    min_size, max_size
        Gene sets with fewer / more genes in the background are skipped.

    Returns
    -------
    DataFrame
        One row per cluster x library x gene set with at least one cluster gene: ``overlap`` (cluster genes in the
        set), ``cluster_size``, ``set_size`` (in the background), ``fold`` (observed / expected), ``pval``, ``fdr`` (BH
        within the cluster and library, over all tested sets), ``genes``.
    """
    from scipy.stats import hypergeom

    from .de import _bh

    bg = set(map(str, background))
    N = len(bg)
    cl = pd.Series(clusters).astype(object)
    cl = cl[cl.index.astype(str).isin(bg)]
    rows = []
    for lib, gs in sets.items():
        terms = {t: set(g) & bg for t, g in gs.items()}
        terms = {t: g for t, g in terms.items() if min_size <= len(g) <= max_size}
        for c, genes in cl.groupby(cl).groups.items():
            members = set(map(str, genes))
            n = len(members)
            res = []
            for t, g in terms.items():
                k = len(members & g)
                res.append((t, k, len(g), hypergeom.sf(k - 1, N, len(g), n) if k else 1.0,
                            ";".join(sorted(members & g))))
            if not res:
                continue
            d = pd.DataFrame(res, columns=["term", "overlap", "set_size", "pval", "genes"])
            d["fdr"] = _bh(d["pval"].to_numpy(), len(d))
            d = d[d["overlap"] > 0]
            d.insert(0, "cluster", c)
            d.insert(1, "library", lib)
            d["cluster_size"] = n
            d["fold"] = d["overlap"] / (n * d["set_size"] / N)
            rows.append(d)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out.sort_values(["cluster", "fdr", "pval"], ignore_index=True) if len(out) else out


def cluster_labels(enrichment, n_terms=2, fdr=0.05, libraries=None, max_chars=60, clean=True, nominal_p=0.01,
                   min_overlap=2):
    """``{cluster: text}`` naming each cluster by its `n_terms` most significant gene sets (FDR < `fdr`), for
    ``plot_smide_heatmap(cluster_labels=...)``. A cluster without one is named by its best set if its nominal p is
    below `nominal_p` (shown as "term (p = ...)", suggestive only), else "n.s."; sets need `min_overlap` cluster
    genes. `clean` drops GO ids from the names."""
    import re

    e = enrichment if libraries is None else enrichment[enrichment["library"].isin(libraries)]
    e = e[e["overlap"] >= min_overlap]
    out = {}
    for c in sorted(enrichment["cluster"].unique()):
        d = e[(e["cluster"] == c) & (e["fdr"] < fdr)].sort_values(["fdr", "pval"])
        if d.empty and nominal_p:
            best = e[(e["cluster"] == c) & (e["pval"] < nominal_p)].sort_values("pval").head(1)
            if len(best):
                t = re.sub(r"\s*\(GO:\d+\)", "", best["term"].iloc[0]) if clean else best["term"].iloc[0]
                out[c] = f"{t[:max_chars]} (p = {best['pval'].iloc[0]:.1g})"
                continue
        names = []
        for t in d["term"]:
            t = re.sub(r"\s*\(GO:\d+\)", "", t) if clean else t
            t = t if len(t) <= max_chars else t[: max_chars - 1] + "…"
            if t not in names:
                names.append(t)
            if len(names) == n_terms:
                break
        out[c] = "; ".join(names) if names else "n.s."
    return out
