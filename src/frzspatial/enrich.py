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

__all__ = ["gene_sets", "smide_gsea"]

DEFAULT_LIBRARIES = ("MSigDB_Hallmark_2020", "GO_Biological_Process_2025", "Reactome_Pathways_2024")


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
