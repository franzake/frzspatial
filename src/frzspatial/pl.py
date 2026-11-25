"""Static plots of single slides with spatialdata-plot, y up as in the AtoMx slide view.

Each function returns the matplotlib axes without showing them: call ``plt.show()`` or ``ax.figure.savefig(...)``.
"""

import warnings

import numpy as np
import pandas as pd
import spatialdata_plot  # noqa: F401  (adds .pl to SpatialData objects)

from .core import coordinate_system, core_box

__all__ = ["plot_core", "plot_composition", "plot_umap", "plot_smide_volcano", "plot_smide_heatmap",
           "plot_smide_patients", "plot_group_proportions", "plot_group_values", "plot_radial_position",
           "plot_gsea_dotplot", "plot_gsea_bars", "smide_heatmap_clusters"]


def plot_core(sdata, core, genes=None, color=None, palette=None, title=None, core_key="core_str", slide=None,
              fill_alpha=0.5, image=True, outline_alpha=0.6, ax=None, cmap="viridis", layer=None, outline_width=None,
              legend_outside=False, legend_fontsize=None):
    """One core of a single slide: its composite image, cell outlines and selected transcripts.

    Parameters
    ----------
    sdata
        A single slide, or a merged object with `slide`.
    core
        Core name (``obs[core_key]``).
    genes
        ``{gene: colour}`` of transcripts to draw.
    color
        obs column to fill the cells by (e.g. ``"manual_annotation"``), or a gene: each cell filled by its counts
        (`layer`, default ``X``) on `cmap`. Without it, cells are outlined in yellow. A list draws one panel per
        entry, side by side (returns their axes). Or a Series over the table's cells (e.g. a column masked with
        ``.where(...)``): plotted as a column, with the cells where it is NA in grey, and not kept in the table.
    palette
        ``{value: colour}`` for `color` (default: the column's stored colours, e.g. from a UMAP, else scanpy's); cells
        with other values or NA are drawn grey.
    title
        Plot title (default: ``"<slide> <core>: <color>"``).
    slide
        For a merged object: the core's slide (its elements ``*-<slide>``, coordinate system ``global-<slide>``).
    fill_alpha, outline_alpha
        Opacity of the cells' fill (with `color`) and outlines.
    image
        Draw the composite image under the cells; or a prefix of a per-core image to draw instead, e.g. ``"he_"``
        for ``he_<slide>_<core>`` (the aligned H&E of :func:`frzspatial.he.align_he_cores`).
    ax
        Matplotlib axes to draw into (default: a new figure).
    cmap, layer
        Colour map and table layer for a gene `color`.
    outline_width
        Width of the cell borders (default: 0.3 on the image, 0.6 without it).
    legend_outside, legend_fontsize
        Put the legend to the right of the core, without a frame; its font size (default: matplotlib's).

    Examples
    --------
    >>> plot_core(slides["MN2"], "C4R7", genes={"NPHS2": "magenta"})  # doctest: +SKIP
    >>> plot_core(usc, "C4R7", slide="MN2", color=types.where(in_glom), legend_outside=True)  # doctest: +SKIP
    >>> plot_core(usc, "C4R7", slide="MN2", color=["CFH", "VCAM1"], image=False)  # doctest: +SKIP
    """
    kw = dict(genes=genes, palette=palette, title=title, core_key=core_key, slide=slide, fill_alpha=fill_alpha,
              image=image, outline_alpha=outline_alpha, cmap=cmap, layer=layer, outline_width=outline_width,
              legend_outside=legend_outside, legend_fontsize=legend_fontsize)
    if isinstance(color, pd.Series):   # a temporary column, removed afterwards
        table, tmp = sdata.tables["table"], "_plot_core"
        table.obs[tmp] = color.reindex(table.obs_names)
        kw["title"] = title or f"{slide + ' ' if slide else ''}{core}: {color.name or 'selection'}"
        try:
            return plot_core(sdata, core, color=tmp, ax=ax, **kw)
        finally:
            table.obs.drop(columns=tmp, inplace=True, errors="ignore")
            table.uns.pop(f"{tmp}_colors", None)
    if isinstance(color, (list, tuple)):
        import matplotlib.pyplot as plt

        if ax is None:
            h = plt.rcParams["figure.figsize"][1]   # square panels at the session's figure height
            _, ax = plt.subplots(1, len(color), figsize=(h * len(color), h), squeeze=False)
            ax = ax[0]
        return [plot_core(sdata, core, color=c, ax=a, **kw) for c, a in zip(color, ax)]
    suffix, cs = coordinate_system(sdata, slide) if slide else ("", "global")
    shapes = f"cell_boundaries{suffix}"
    if isinstance(image, str):   # a per-core image, e.g. the aligned H&E
        comps = [n for n in (f"{image}{slide}_{core}", f"{image}{core}") if n in sdata.images][:1]
        if not comps:
            raise KeyError(f"no image {image}{slide + '_' if slide else ''}{core}")
    else:
        comps = [name for name in sdata.images if name.endswith(f"_image{suffix}")] if image else []
    elements = comps + [shapes, "table"] + ([f"transcripts{suffix}"] if genes else [])
    with warnings.catch_warnings():  # a merged table also annotates the other slides' polygons, left out here
        warnings.filterwarnings("ignore", message="The table is annotating", category=UserWarning)
        sub = sdata.subset(elements).query.bounding_box(**core_box(sdata, core, core_key, slide=slide))

    base = sub.pl.render_images() if image else sub
    width = outline_width if outline_width is not None else (0.3 if image else 0.6)
    if color and color not in sub.tables["table"].obs and color in sub.tables["table"].var_names:
        pl = base.pl.render_shapes(
            shapes,
            color=color,
            table_layer=layer,
            cmap=cmap,
            fill_alpha=fill_alpha if image else 1,
            outline_alpha=outline_alpha,
            outline_width=width)
    elif color:
        from ._palette import column_palette

        pal = column_palette(sdata.tables["table"], color, palette)
        present = set(sub.tables["table"].obs[color].dropna().astype(str))
        groups = [g for g in pal if g in present]
        pl = base.pl.render_shapes(
            shapes,
            color=color,
            groups=groups,
            palette=[pal[g] for g in groups],
            na_color="#DEDEDE",
            fill_alpha=fill_alpha,
            outline_alpha=outline_alpha,
            outline_width=width)
    else:
        pl = base.pl.render_shapes(
            shapes,
            fill_alpha=0,
            outline_alpha=outline_alpha,
            outline_width=width,
            outline_color="#FFFF00" if image else "#4D4D4D")  # yellow shows on the dark composite, dark grey on white
    if genes:
        pl = pl.pl.render_points(
            f"transcripts{suffix}",
            color="target",
            groups=list(genes),
            palette=genes,
            size=0.3 if image else 2,
            alpha=0.6 if image else 0.9)
    title = title or f"{slide + ' ' if slide else ''}{core}: {color or ('transcripts' if genes else 'composite')}"
    if ax is not None:
        pl.pl.show(coordinate_systems=cs, title=title, ax=ax)
    else:
        ax = pl.pl.show(coordinate_systems=cs, title=title, show=False, return_ax=True)

    ax.invert_yaxis()
    leg = ax.get_legend()
    if leg is not None and legend_outside:
        ax.legend(handles=leg.legend_handles, labels=[t.get_text() for t in leg.get_texts()], loc="center left",
                  bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=legend_fontsize)
    elif leg is not None and legend_fontsize:
        for text in leg.get_texts():
            text.set_fontsize(legend_fontsize)
    return ax


def plot_composition(obs, color, sample="sample_id", group="group_mn", palette=None, figsize=(14, 7),
                     exclude_groups=("Placenta",), groups=None, legend_fontsize=8):
    """Composition of each sample in `color` (e.g. cell types), stacked bars of fractions, one panel per `group`.

    Parameters
    ----------
    obs
        Cell metadata with `sample`, `group` and `color` (rows without a sample, or in `exclude_groups`, are left out).
    groups
        Order of the panels (default: as in `obs`); groups not listed are left out.
    legend_fontsize
        Size of the legend's entries and title.
    palette
        ``{value: colour}`` (default: scanpy's colours over all values of `color`).

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    d = obs[[sample, group, color]].dropna(subset=[sample, group])
    d = d[~d[group].astype(str).isin(exclude_groups)]
    frac = pd.crosstab([d[group].astype(str), d[sample].astype(str)], d[color], normalize="index")
    frac = frac.loc[:, frac.sum() > 0]
    present = list(dict.fromkeys(frac.index.get_level_values(0)))
    groups = [g for g in groups if g in present] if groups is not None else present
    from ._palette import scanpy_palette

    cats = obs[color].cat.categories if hasattr(obs[color], "cat") else frac.columns
    cols = palette or scanpy_palette(cats)
    widths = [len(frac.loc[g]) for g in groups]
    fig, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=True, gridspec_kw={"width_ratios": widths})
    for ax, g in zip(np.atleast_1d(axes), groups):
        f = frac.loc[g]
        bottom = np.zeros(len(f))
        for c in f.columns:
            ax.bar(range(len(f)), f[c].to_numpy(), bottom=bottom, color=cols.get(c, "0.7"), width=0.9, label=c)
            bottom += f[c].to_numpy()
        ax.set_xticks(range(len(f)), f.index, rotation=90, fontsize=7)
        ax.set_title(g, fontsize=11)
        ax.set_xlim(-0.6, len(f) - 0.4)
    np.atleast_1d(axes)[0].set_ylabel("fraction of cells")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols.get(c, "0.7")) for c in frac.columns]
    fig.legend(handles, frac.columns, title=color, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize,
               title_fontsize=legend_fontsize, frameon=False)
    fig.tight_layout()
    return fig


def _densest_point(xy, bins=150, extent=None):
    """Centre of the most crowded bin of a 2D histogram (lightly smoothed): where most of a group's cells are."""
    from scipy.ndimage import gaussian_filter

    h, ex, ey = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=extent)
    i, j = np.unravel_index(np.argmax(gaussian_filter(h, 1.5)), h.shape)
    return (ex[i] + ex[i + 1]) / 2, (ey[j] + ey[j + 1]) / 2


def _place_labels(ax, xy, group, fontsize):
    """Each group's name at its densest point, largest groups first; a name that would overlap one already placed
    goes to the nearest free spot on rings around that point. adjustText was dropped (2026-11-16): it also pushes a
    name away from its own point, and moved names far from their cells."""
    import matplotlib.patheffects as pe

    extent = [[xy[:, 0].min(), xy[:, 0].max()], [xy[:, 1].min(), xy[:, 1].max()]]
    g = group.to_numpy()
    sizes = group.value_counts()
    renderer = ax.figure.canvas.get_renderer()
    to_px, placed = ax.transData.transform, []
    for c in sizes[sizes > 0].index:
        x0, y0 = to_px(_densest_point(xy[g == c], extent=extent))
        t = ax.text(0, 0, c, fontsize=fontsize, weight="bold", ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=3, foreground="w")], transform=None)
        t.set_position((x0, y0))
        w, h = t.get_window_extent(renderer).width, t.get_window_extent(renderer).height
        best = (x0, y0)
        for r in np.arange(0, 40) * h / 2:  # rings half a text height apart
            ring = [(x0, y0)] if r == 0 else [(x0 + r * np.cos(a), y0 + r * np.sin(a))
                                              for a in np.linspace(0, 2 * np.pi, int(8 + r / h * 8), endpoint=False)]
            free = [(x, y) for x, y in ring
                    if all(abs(x - px) > (w + pw) / 2 or abs(y - py) > (h + ph) / 2 for px, py, pw, ph in placed)]
            if free:
                best = free[0]
                break
        placed.append((*best, w, h))
        t.set_transform(ax.transData)
        t.set_position(ax.transData.inverted().transform(best))


def plot_umap(table, color, palette=None, labels=True, legend=False, title="", size=3, fontsize=12,
              legend_fontsize=None, figsize=None, dpi=None, min_label_frac=0):
    """UMAP coloured by an obs column, as main.Rmd's UMAP panels.

    Parameters
    ----------
    table
        AnnData with ``obsm["X_umap"]``.
    color
        obs column(s): one, or a list for side-by-side panels (then without labels).
    palette
        ``{value: colour}`` for a single categorical `color` (stored in ``uns[f"{color}_colors"]``).
    labels
        Name each group on the plot where most of its cells are (the densest point, not the median: a group spread over
        several islands has its median between them). A name that would cover another moves by the smallest step that
        clears it.
    min_label_frac
        Name only groups with at least this fraction of the cells (0: all). With many groups (97 atlas types) the small
        ones' names crowd the plot's edges.
    legend
        Also a legend (useful with `palette`): True or ``"top"`` above the plot in 4 columns, ``"right"`` to its right
        in one column.
    fontsize, legend_fontsize
        Names on the plot; legend entries (None: `fontsize`, its title a bit larger).
    figsize
        (width, height) in inches of a new figure (None: matplotlib's default). Give the chunk's ``fig-width`` and
        ``fig-height``: a figure of another shape is shrunk into that box, with blank margins.
    dpi
        Resolution of that new figure (None: matplotlib's default, blurry at 366k cells).

    Returns
    -------
    Axes or list of Axes
    """
    import matplotlib.pyplot as plt
    import scanpy as sc

    if isinstance(color, str) and palette:
        cats = table.obs[color].cat.categories
        table.uns[f"{color}_colors"] = [palette.get(c, "#BDBDBD") for c in cats]
    single = isinstance(color, str)
    if single and figsize:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        ax = None
    ax = sc.pl.umap(table, color=color, title=title if single else None, size=size, frameon=True, show=False,
                    legend_loc="none" if single else "right margin", ax=ax)
    if single and legend:
        cats = table.obs[color].cat.categories
        handles = [plt.Line2D([], [], marker="o", ls="", color=c) for c in table.uns[f"{color}_colors"]]
        where = ({"loc": "center left", "bbox_to_anchor": (1.01, 0.5), "ncols": 1} if legend == "right" else
                 {"loc": "lower center", "bbox_to_anchor": (0.5, 1.01), "ncols": 4})
        lf = legend_fontsize or fontsize
        ax.legend(handles, cats, title=color, fontsize=lf, title_fontsize=lf * 1.1, markerscale=lf / 10,
                  frameon=False, columnspacing=0.6, handletextpad=0.2, **where)
    plt.tight_layout()
    # Names after the layout: they are spaced in screen pixels, which the layout would change
    if single and labels:
        group = table.obs[color]
        if min_label_frac:
            share = group.value_counts(normalize=True)
            group = group.where(group.isin(share.index[share >= min_label_frac]))
        _place_labels(ax, np.asarray(table.obsm["X_umap"]), group, fontsize)
    return ax




def plot_smide_volcano(res, celltype, contrast, fdr=0.05, y="FDR_BH", n_labels=12, title=None, figsize=(9, 8),
                       fontsize=12, repel=True, ax=None):
    """Volcano plot of one ``frzspatial.de.smide`` family: log2 fold change against -log10 `y`.

    Parameters
    ----------
    res
        The DataFrame returned by ``smide`` (or ``table.uns["smide"]["results"]``).
    celltype, contrast
        The family.
    fdr
        Threshold on ``FDR_BH`` that colours the genes (higher in case: red, lower: blue).
    y
        ``"FDR_BH"`` (default) or ``"p_for_inference"`` (nominal p) on the y axis.
    n_labels
        The genes with the smallest FDR that get a name.
    fontsize
        Size of all text: tick labels, axis labels and gene names; the title is a little larger, the legend a little
        smaller.
    repel
        Spread the gene names apart (adjustText), each joined to its point by a thin line.

    Returns
    -------
    Axes
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    d = res[(res["celltype"] == celltype) & (res["contrast"] == contrast) & res["valid"]].copy()
    if d.empty:
        raise ValueError(f"no valid fits for {celltype!r} / {contrast!r}")
    d["_y"] = -np.log10(d[y].clip(lower=1e-300))
    sig = d["FDR_BH"] < fdr
    up, down = sig & (d["beta"] > 0), sig & (d["beta"] < 0)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    fs = fontsize
    ax.scatter(d.loc[~sig, "log2FC"], d.loc[~sig, "_y"], s=fs * 0.7, c="0.78", lw=0, label=f"FDR >= {fdr}")
    case, ref = d["case"].iloc[0], d["reference"].iloc[0]
    hi = f"higher in {case}" if isinstance(case, str) else "positive"
    lo = f"higher in {ref}" if isinstance(ref, str) else "negative"
    ax.scatter(d.loc[up, "log2FC"], d.loc[up, "_y"], s=fs * 1.5, c="#c0392b", lw=0, label=f"{hi} ({int(up.sum())})")
    ax.scatter(d.loc[down, "log2FC"], d.loc[down, "_y"], s=fs * 1.5, c="#2a6fbb", lw=0,
               label=f"{lo} ({int(down.sum())})")
    ax.axvline(0, c="0.6", lw=0.8)
    if y == "FDR_BH":
        ax.axhline(-np.log10(fdr), c="0.6", lw=0.8, ls="--")
    texts = [ax.text(r["log2FC"], r["_y"], r["gene"], fontsize=fs * 0.9,
                     path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])
             for _, r in d[sig].nsmallest(n_labels, "FDR_BH").iterrows()]
    ax.set_xlabel(f"adjusted log2 fold change ({case} vs {ref})" if isinstance(case, str) else "coefficient / log(2)",
                  fontsize=fs)
    ax.set_ylabel("-log10 FDR (BH)" if y == "FDR_BH" else "-log10 p", fontsize=fs)
    ax.tick_params(labelsize=fs * 0.9)
    nice = contrast.replace("_vs_", " vs ")
    ax.set_title(title or f"{celltype}: {nice} ({len(d):,} genes)", fontsize=fs * 1.15)
    ax.margins(x=0.08, y=0.1)
    leg = ax.legend(loc="best", fontsize=fs * 0.8, frameon=True, framealpha=0.85, edgecolor="none",
                    markerscale=1.5, handletextpad=0.3, borderpad=0.4)
    leg.set_zorder(5)
    if repel and texts:
        try:
            from adjustText import adjust_text

            adjust_text(texts, ax=ax, expand=(1.15, 1.3), arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
        except ImportError:
            pass
    return ax


def _smide_order(res, celltypes, contrasts):
    types = list(celltypes) if celltypes is not None else list(dict.fromkeys(res["celltype"]))
    cons = list(contrasts) if contrasts is not None else list(dict.fromkeys(res["contrast"]))
    return types, cons


def _smide_cmap():
    """The smide heatmaps' blue - white - red palette (deep ends)."""
    import seaborn as sns

    return sns.diverging_palette(250, 12, s=90, l=42, sep=8, as_cmap=True)


def _sat_colorbar(fig, im, lim, label, cax=None, ax=None, fontsize=12):
    """Colour bar with saturated ends labelled <= -lim and >= lim."""
    cb = fig.colorbar(im, cax=cax, ax=ax, ticks=[-lim, -lim / 2, 0, lim / 2, lim],
                      **({} if cax is not None else dict(fraction=0.03, pad=0.02)))
    cb.ax.set_yticklabels([f"<= -{lim:g}", f"{-lim / 2:g}", "0", f"{lim / 2:g}", f">= {lim:g}"], fontsize=fontsize)
    cb.set_label(label, fontsize=fontsize * 1.15)
    return cb


def _cell_borders(ax, nrow, ncol, color, lw=0.5):
    """A thin line between every two cells of an imshow heatmap, inside the heatmap only (not over its headers)."""
    ax.grid(False)
    ax.vlines(np.arange(-0.5, ncol + 0.5, 1), -0.5, nrow - 0.5, colors=color, linewidths=lw, zorder=2)
    ax.hlines(np.arange(-0.5, nrow + 0.5, 1), -0.5, ncol - 0.5, colors=color, linewidths=lw, zorder=2)


def _smide_heatmap_data(res, celltypes, contrasts, n_top, fdr, clip, n_clusters):
    """Genes, log2FC / FDR matrices, tree and clusters (numbered top to bottom) of ``plot_smide_heatmap``."""
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage

    types, cons = _smide_order(res, celltypes, contrasts)
    fam = res.assign(family=res["celltype"].astype(str) + " | " + res["contrast"].astype(str))
    order = [f"{t} | {c}" for t in types for c in cons]
    sig = fam[(fam["FDR_BH"] < fdr) & fam["family"].isin(order)].sort_values("FDR_BH")
    top = sig.groupby("family", sort=False).head(n_top)["gene"].unique()
    if not len(top):
        raise ValueError(f"no gene with FDR < {fdr}")
    l2fc = fam.pivot_table(index="gene", columns="family", values="log2FC").reindex(index=top, columns=order)
    q = fam.pivot_table(index="gene", columns="family", values="FDR_BH").reindex(index=top, columns=order)
    if clip == "auto":
        clip = max(0.25, float(np.ceil(np.nanpercentile(np.abs(l2fc.to_numpy()), 90) * 4) / 4))
    tree = None
    if len(l2fc) > 2:
        tree = linkage(l2fc.fillna(0).clip(-clip, clip).to_numpy(), "average", metric="correlation")
        rows = l2fc.index[leaves_list(tree)]
        l2fc, q = l2fc.loc[rows], q.loc[rows]
    clusters = None
    if tree is not None and n_clusters:
        clusters = fcluster(tree, n_clusters, "maxclust")[leaves_list(tree)]
        first = {c: k + 1 for k, c in enumerate(dict.fromkeys(clusters))}   # numbered top to bottom
        clusters = np.array([first[c] for c in clusters])
    return dict(types=types, cons=cons, order=order, sig=sig, l2fc=l2fc, q=q, clip=clip, tree=tree, clusters=clusters)


def smide_heatmap_clusters(res, n_clusters, celltypes=None, contrasts=None, n_top=50, fdr=0.05, clip="auto"):
    """The gene clusters of ``plot_smide_heatmap`` with the same arguments: ``Series`` gene -> cluster (1 = top), in
    the heatmap's row order, for annotating the clusters (e.g. ``frzspatial.enrich.cluster_enrichment``)."""
    d = _smide_heatmap_data(res, celltypes, contrasts, n_top, fdr, clip, n_clusters)
    if d["clusters"] is None:
        raise ValueError("fewer than 3 genes: no clustering")
    return pd.Series(d["clusters"], index=d["l2fc"].index, name="cluster")


def plot_smide_heatmap(res, celltypes=None, contrasts=None, n_top=50, clip="auto", fdr=0.05, fontsize=11,
                       cell_borders=True, border_color="0.3", figsize=None, title=None, row_tree=True,
                       n_clusters=None, row_annotation=None, annotation_palette=None, annotation_title="pathway",
                       cluster_labels=None, cluster_label_chars=38):
    """Overview of a ``frzspatial.de.smide`` run: each family's top genes x all families (cell type x contrast).

    Colour: adjusted log2 fold change (red higher in the first group named), on the same blue - white - red scale as
    ``plot_smide_patients``, saturated beyond `clip`; star: FDR < `fdr`; grey: gene not tested in that cell type.
    Gene labels give the log2FC in the gene's most significant family. Genes are ordered by hierarchical clustering
    of their fold-change profiles.

    Parameters
    ----------
    res
        The DataFrame returned by ``smide``.
    celltypes, contrasts
        Order of the column blocks and of the columns within them (default: as in `res`).
    n_top
        Significant genes with the smallest FDR taken from each family.
    clip
        Colour limit (|log2FC|): a number, or ``"auto"`` (the 90th percentile of |log2FC| over the cells shown, so a
        few large changes don't wash out the others).
    fontsize
        Base size of all text (gene and contrast labels; title, colour bar and cell-type labels scale from it). The
        figure is sized from it (rows, columns, margins), so nothing overlaps at any size; its size is logged (in
        Quarto, set the chunk's fig-width / fig-height to it). `figsize` overrides.
    cell_borders, border_color
        Draw a thin line between every two cells, in this colour.
    row_tree
        Draw the genes' clustering tree (average linkage, correlation of their log2FC profiles) right of the heatmap.
    n_clusters
        Cut the tree into this many gene clusters, separated by lines (and numbered on the tree).
    row_annotation, annotation_palette, annotation_title
        A strip left of the heatmap colouring each gene by a category, e.g. ``frzspatial.enrich.gene_themes(gsea)``
        (pathway programme from the GSEA leading edges); genes without one stay blank. Palette ``{category: colour}``
        (default ``enrich.THEME_COLORS`` for the themes, else scanpy's).
    cluster_labels, cluster_label_chars
        ``{cluster: text}`` written beside each cluster's number (with `n_clusters`), e.g.
        ``enrich.cluster_labels(enrich.cluster_enrichment(smide_heatmap_clusters(...), sets, background))``: the
        clusters' enriched pathways; wrapped at `cluster_label_chars` characters.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from scipy.cluster.hierarchy import dendrogram

    d = _smide_heatmap_data(res, celltypes, contrasts, n_top, fdr, clip, n_clusters)
    types, cons, order, sig = d["types"], d["cons"], d["order"], d["sig"]
    l2fc, q, clip, tree, clusters = d["l2fc"], d["q"], d["clip"], d["tree"], d["clusters"]
    ann = None
    if row_annotation is not None:
        ann = pd.Series(row_annotation).reindex(l2fc.index)
        cats = (list(ann.cat.categories) if isinstance(ann.dtype, pd.CategoricalDtype)
                else sorted(ann.dropna().astype(str).unique()))
        cats = [c for c in cats if (ann.astype(object) == c).any()]
        if annotation_palette is None:
            from .enrich import THEME_COLORS
            from ._palette import scanpy_palette

            annotation_palette = {c: THEME_COLORS.get(c) for c in cats}
            missing = [c for c in cats if annotation_palette[c] is None]
            annotation_palette |= scanpy_palette(missing) if missing else {}
    best = sig.drop_duplicates("gene").set_index("gene")["log2FC"]   # log2FC in each gene's most significant family
    import textwrap

    from ._utils import _log

    # layout in inches, every size from fontsize
    fs = fontsize
    gene_fs, col_fs, head_fs, title_fs = 0.85 * fs, fs, 1.2 * fs, 1.3 * fs
    labels = [f"{g} ({best[g]:+.1f})" for g in l2fc.index]
    xlabels = [o.split(" | ")[1].replace("_vs_", " vs ") for o in order]
    row_h = gene_fs * 1.35 / 72
    head_w = max(len(t) for t in types) * head_fs * 0.62 / 72 + 0.1     # bold cell-type header over its columns
    col_w = max(fs * 2.0 / 72, head_w / len(cons))
    axes_h = row_h * (len(l2fc) + 1.3)                                   # genes + the cell-type header
    heat_w = col_w * len(order)
    label_w = max(len(x) for x in labels) * gene_fs * 0.6 / 72 + 0.3
    xlab_h = max(len(x) for x in xlabels) * col_fs * 0.62 / 72 + 0.3
    cbar_w = 0.5 + fs * 5.5 / 72
    ann_w = col_w * 0.8 if ann is not None else 0.0                       # pathway strip
    num_w = fs * 1.5 / 72 if n_clusters else 0                            # cluster numbers right of the tree
    lab_w = (cluster_label_chars * fs * 0.85 * 0.55 / 72 + 0.15) if (cluster_labels and n_clusters) else 0
    tree_w = (0.35 + fs * 0.05 + num_w + lab_w) if (row_tree and tree is not None) else 0.0
    leg_w = (max(len(c) for c in cats) * fs * 0.55 / 72 + 0.4) if ann is not None else 0.0
    x_heat = label_w + ann_w + (0.05 if ann is not None else 0)
    W = x_heat + heat_w + tree_w + max(cbar_w, leg_w) + 0.2
    ttl = title or (f"smiDE: top {n_top} genes per family (* FDR < {fdr}; grey: not tested; "
                    "label: log2FC in its most significant family)")
    ttl = "\n".join(textwrap.wrap(ttl, width=max(30, int(W * 72 / (title_fs * 0.55)))))
    title_h = (ttl.count("\n") + 1) * title_fs * 1.45 / 72 + 0.3
    H = title_h + axes_h + xlab_h + 0.2
    if figsize is not None:
        W, H = figsize
    cmap = _smide_cmap().copy()
    cmap.set_bad("0.88")
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([x_heat / W, xlab_h / H, heat_w / W, axes_h / H])
    im = ax.imshow(l2fc.clip(-clip, clip).to_numpy(), cmap=cmap, vmin=-clip, vmax=clip, aspect="auto",
                   interpolation="none")
    rows_h = row_h * len(l2fc)                                             # the gene rows, below the header
    if ann is not None:
        aax = fig.add_axes([label_w / W, xlab_h / H, ann_w / W, rows_h / H])
        code = ann.astype(object).map({c: k for k, c in enumerate(cats)}).to_numpy(dtype=float)
        amap = ListedColormap([annotation_palette[c] for c in cats])
        amap.set_bad("white")
        aax.imshow(np.ma.masked_invalid(code)[:, None], cmap=amap, vmin=-0.5, vmax=len(cats) - 0.5, aspect="auto",
                   interpolation="none")
        aax.set_xticks([0], [annotation_title], rotation=90, fontsize=col_fs)
        aax.set_yticks([])
        for side in aax.spines.values():
            side.set_linewidth(0.5)
    if row_tree and tree is not None:
        tax = fig.add_axes([(x_heat + heat_w + 0.05) / W, xlab_h / H, (tree_w - 0.1 - num_w - lab_w) / W,
                            rows_h / H])
        dendrogram(tree, ax=tax, orientation="right", no_labels=True, color_threshold=0,
                   above_threshold_color="0.3")
        tax.set_ylim(10 * len(l2fc), 0)
        tax.set_axis_off()
        if clusters is not None:
            for k in np.unique(clusters):
                idx = np.flatnonzero(clusters == k)
                tax.text(max(tax.get_xlim()) * 1.04, 10 * idx.mean() + 5, str(k), fontsize=fs, ha="left",
                         va="center", weight="bold", color="#B2182B", clip_on=False)
                if cluster_labels and k in cluster_labels:
                    import textwrap as _tw

                    txt = "\n".join(_tw.wrap(str(cluster_labels[k]), cluster_label_chars))
                    x_lab = (x_heat + heat_w + tree_w - lab_w) / W
                    y_lab = (xlab_h + rows_h * (1 - (idx.mean() + 0.5) / len(l2fc))) / H
                    fig.text(x_lab, y_lab, txt, fontsize=fs * 0.85, va="center", ha="left", color="#B2182B")
    yy, xx = np.nonzero((q < fdr).to_numpy())
    ax.scatter(xx, yy, s=(row_h * 72 * 0.6) ** 2, c="k", marker="*", linewidths=0, zorder=3)
    ax.set_xticks(range(len(order)), xlabels, rotation=90, fontsize=col_fs)
    ax.set_yticks(range(len(l2fc)), labels, fontsize=gene_fs)
    if ann is not None:                                                    # gene names left of the pathway strip
        ax.tick_params(axis="y", pad=(ann_w + 0.05) * 72 + 2)
    if cell_borders:
        _cell_borders(ax, len(l2fc), len(order), border_color)
    else:
        ax.grid(False)
    for k, t in enumerate(types):
        ax.text(len(cons) * k + (len(cons) - 1) / 2, -0.8, t, ha="center", va="bottom", fontsize=head_fs,
                weight="bold")
        if k:
            ax.axvline(len(cons) * k - 0.5, c="k", lw=1.5, zorder=4)
    if clusters is not None:
        for b in np.flatnonzero(np.diff(clusters)) + 0.5:
            ax.axhline(b, c="k", lw=1.2, zorder=4)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(len(l2fc) - 0.5, -1.8)
    x_side = x_heat + heat_w + tree_w + 0.25
    cax = fig.add_axes([x_side / W, xlab_h / H + 0.55 * axes_h / H, 0.22 / W, 0.35 * axes_h / H])
    _sat_colorbar(fig, im, clip, "adjusted log2 fold change", cax=cax, fontsize=fs)
    if ann is not None:
        handles = [plt.Line2D([], [], marker="s", ls="", ms=fs * 0.9, color=annotation_palette[c], label=c)
                   for c in cats]
        fig.legend(handles=handles, title=annotation_title, loc="upper left", fontsize=fs * 0.9,
                   title_fontsize=fs, frameon=False, bbox_to_anchor=((x_side - 0.1) / W, (xlab_h + 0.5 * axes_h) / H),
                   handletextpad=0.3, borderaxespad=0)
    fig.suptitle(ttl, fontsize=title_fs, y=1 - 0.1 / H, va="top")
    _log(f"plot_smide_heatmap: figure {W:.1f} x {H:.1f} in (in Quarto, set the chunk's fig-width / fig-height to these)")
    return fig


def plot_smide_patients(sdata, res, celltype_col, contrast, group_col, patient_col="sample_id", celltypes=None,
                        cells=None, n_genes=50, min_cells=10, layer="counts", groups=None, palette=None, z_lim=2,
                        cluster="within", column_labels=None, fontsize=13, cell_borders=True, border_color="0.3",
                        table="table", title=None):
    """The top genes of one ``smide`` contrast at the level of the patients: one heatmap per cell type.

    Each column is one patient's pseudobulk of the cell type (sum of raw counts over its cells, log CPM), each row a
    gene (the `n_genes` smallest p of `contrast` in that type, ordered by fold change; label: log2FC and * when
    FDR < 0.05). Values are robust z-scores per gene across patients (median / MAD), so a few extreme patients don't
    flatten the others' contrast. Patients are grouped by `group_col` and ordered within each group by hierarchical
    clustering (average linkage, Euclidean) of their gene profiles, each group's tree drawn above it: an effect carried
    by most patients of a group reads as one block, subgroups or outlier patients separate within it.

    Parameters
    ----------
    sdata
        SpatialData (its table `table`) or AnnData with raw counts (`layer`, or ``X`` if None).
    res
        The DataFrame returned by ``smide``.
    celltype_col, contrast, group_col, patient_col
        obs columns of the cell types, the contrast name in `res`, the group variable and the patient.
    celltypes
        Cell types, one panel each (default: those of `res` for `contrast`).
    cells
        The cells to pool, as ``smide(cells=...)``: e.g. ``"ng_glom_id"`` (the cells of a glomerulus).
    n_genes, min_cells
        Genes per panel; patients with fewer cells of the type are left out.
    groups, palette
        Group order (default: the contrast's reference, its case, then the other groups) and ``{group: colour}``.
    z_lim
        Colour limit of the robust z-scores.
    cluster
        ``"within"`` (default): patients clustered within each group, one tree per group, groups side by side;
        ``"all"``: one tree over all patients (the group strip shows who is who); None: patients by group,
        unclustered.
    column_labels
        obs column whose value labels each patient's column (e.g. ``"slide"``). None (default): no labels, since
        patient ids can be identifiers.
    fontsize
        Base size of all text: gene and column labels, panel titles, legend, colour bar and figure title scale from it,
        and so does the figure (rows, columns and gaps are sized from the text), so nothing overlaps at any size.
    cell_borders, border_color
        Draw a thin line between every two cells, in this colour.

    Returns
    -------
    Figure
        Sized from `fontsize`, `n_genes` and the number of patients; its size is logged (in Quarto, set the chunk's
        fig-width / fig-height to it, or the figure is redrawn at another size).
    """
    import matplotlib.pyplot as plt
    import scipy.sparse as sp
    from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

    from .de import _PROBE_PREFIXES, _cell_mask

    tb = sdata.tables[table] if hasattr(sdata, "tables") else sdata
    obs = tb.obs
    r_all = res[res["contrast"] == contrast]
    if r_all.empty:
        raise ValueError(f"no contrast {contrast!r} in res")
    types = list(celltypes) if celltypes is not None else list(dict.fromkeys(r_all["celltype"]))
    case, ref = r_all["case"].iloc[0], r_all["reference"].iloc[0]
    present = [g for g in pd.unique(obs[group_col].dropna().astype(str))]
    if groups is None:
        first = [g for g in (ref, case) if isinstance(g, str) and g in present]
        groups = first + sorted(g for g in present if g not in first)
    from ._palette import scanpy_palette
    palette = palette or scanpy_palette(groups)
    X = sp.csr_matrix(tb.layers[layer] if layer else tb.X)
    names = tb.var_names.astype(str)
    total = np.asarray(X[:, np.flatnonzero(~names.str.startswith(_PROBE_PREFIXES))].sum(axis=1)).ravel()
    gidx = pd.Index(names)
    keep_cells = _cell_mask(obs, cells) & obs[group_col].astype(str).isin(groups).to_numpy() \
        & obs[patient_col].notna().to_numpy()
    import textwrap

    from ._utils import _log

    cmap = _smide_cmap()
    mode = {True: "within", False: None}.get(cluster, cluster)
    if mode not in ("all", "within", None):
        raise ValueError("cluster must be 'all', 'within' or None")

    # 1. Per cell type: patients' pseudobulk, robust z, column order and trees
    panels = []
    for t in types:
        r = r_all[r_all["celltype"] == t]
        top = r.nsmallest(n_genes, "p_for_inference").sort_values("log2FC", ascending=False)
        m = keep_cells & (obs[celltype_col].astype(str) == t).to_numpy()
        o = obs[m]
        n = o.groupby(patient_col, observed=True).size()
        cols_keep = [group_col] + ([column_labels] if column_labels and column_labels != group_col else [])
        pats = (o.drop_duplicates(patient_col).set_index(patient_col).loc[n.index[n >= min_cells], cols_keep]
                .assign(g=lambda d: pd.Categorical(d[group_col].astype(str), groups)).sort_values("g"))
        code = pd.Categorical(o[patient_col].astype(str), categories=pats.index.astype(str)).codes
        ok = code >= 0
        P = sp.csr_matrix((np.ones(ok.sum()), (code[ok], np.flatnonzero(ok))), shape=(len(pats), ok.size))
        idx = np.flatnonzero(m)
        pb = np.asarray((P @ X[idx][:, gidx.get_indexer(top["gene"])]).todense()) / (P @ total[idx])[:, None] * 1e6
        z = np.log1p(pb)
        med = np.median(z, axis=0)
        mad = 1.4826 * np.median(np.abs(z - med), axis=0)
        z = ((z - med) / np.where(mad > 1e-9, mad, z.std(axis=0).clip(min=1e-9))).T   # genes x patients
        zc = np.clip(z, -z_lim, z_lim)
        order, edges, trees = list(range(len(pats))), [], []
        if mode == "all" and len(pats) > 2:
            tree = linkage(zc.T, "average", metric="euclidean")
            order = list(leaves_list(tree))
            trees = [(0, len(pats), tree)]
        elif mode == "within" or mode is None:
            order = []
            for grp in groups:
                cols = np.flatnonzero(pats["g"].astype(str).to_numpy() == grp)
                if mode == "within" and len(cols) > 2:
                    tree = linkage(zc[:, cols].T, "average", metric="euclidean")
                    cols = cols[leaves_list(tree)]
                    trees.append((len(order), len(cols), tree))
                order += list(cols)
                if len(cols):
                    edges.append(len(order) - 0.5)
            edges = edges[:-1]
        labels = [f"{gname} ({l:+.1f}){' *' if q < 0.05 else ''}"
                  for gname, l, q in zip(top["gene"], top["log2FC"], top["FDR_BH"])]
        panels.append(dict(t=t, z=zc[:, order], pats=pats.iloc[order], labels=labels, trees=trees, edges=edges))

    # 2. Layout in inches, every size from fontsize
    fs = fontsize
    gene_fs, col_fs, title_fs = 0.85 * fs, 0.75 * fs, 1.3 * fs
    row_h = gene_fs * 1.35 / 72                                          # one gene row
    col_w = fs * 1.15 / 72                                               # one patient column
    n_pat = max(len(pp["pats"]) for pp in panels)
    heat_w = max(fs * 0.3, col_w * n_pat)
    axes_h = max(fs * 0.15, row_h * (n_genes + 1.3))                     # genes + the group strip
    label_w = max(len(x) for pp in panels for x in pp["labels"]) * gene_fs * 0.6 / 72 + 0.3
    tree_h = max(fs * 0.06, 0.12 * axes_h)
    title_h = title_fs * 1.9 / 72
    xlab_h = (max((len(str(v)) for pp in panels for v in pp["pats"][column_labels]), default=0) * col_fs * 0.62 / 72
              + 0.3) if column_labels else 0.25
    block_h = title_h + tree_h + axes_h + xlab_h
    ncol = 2 if len(panels) > 1 else 1
    nrow = int(np.ceil(len(panels) / ncol))
    cell_w = label_w + heat_w + 0.6
    cbar_w = 0.5 + fs * 5.5 / 72                                         # bar, its tick labels and 2-line label
    W = ncol * cell_w + cbar_w
    sup = title or (f"smiDE {contrast.replace('_vs_', ' vs ')}: patient pseudobulk (robust z of log CPM; "
                    + {"all": "patients clustered by hierarchical clustering",
                       "within": "patients clustered within each group", None: "patients by group"}[mode]
                    + "; label: log2FC, * FDR < 0.05)")
    sup = "\n".join(textwrap.wrap(sup, width=max(40, int(W * 72 / (1.5 * fs * 0.55)))))
    header_h = (sup.count("\n") + 1) * 1.5 * fs * 1.45 / 72 + 1.3 * fs * 2.2 / 72 + 0.4
    H = header_h + nrow * block_h + (nrow - 1) * 0.5 + 0.3
    fig = plt.figure(figsize=(W, H))

    # 3. Panels
    im, plotted = None, set()
    for k, pp in enumerate(panels):
        rr, cc = divmod(k, ncol)
        x0 = cc * cell_w + label_w + 0.3
        y0 = H - header_h - (rr + 1) * block_h - rr * 0.5 + xlab_h
        ax = fig.add_axes([x0 / W, y0 / H, heat_w / W, axes_h / H])
        z, pats = pp["z"], pp["pats"]
        plotted |= set(pats["g"].astype(str))
        im = ax.imshow(z, cmap=cmap, vmin=-z_lim, vmax=z_lim, aspect="auto", interpolation="none")
        ax.scatter(range(len(pats)), [-1.1] * len(pats), c=[palette.get(g, "0.6") for g in pats["g"].astype(str)],
                   marker="s", s=(col_w * 72 * 0.55) ** 2, clip_on=False, zorder=3)
        ax.set_yticks(range(len(pp["labels"])), pp["labels"], fontsize=gene_fs)
        if column_labels:
            ax.set_xticks(range(len(pats)), pats[column_labels].astype(str), rotation=90, fontsize=col_fs)
        else:
            ax.set_xticks([])
        for e in pp["edges"]:
            ax.axvline(e, c="k", lw=2, zorder=4)
        # trees and title in figure fractions, like the heatmap: a figure drawn larger than made (e.g. a Quarto
        # chunk taller than the logged size) gains space instead of sliding titles into trees and panels
        npat = max(len(pats), 1)
        for start, nk, tree in pp["trees"]:
            dax = fig.add_axes([(x0 + heat_w * start / npat) / W, (y0 + axes_h) / H, heat_w * nk / npat / W,
                                tree_h * 0.92 / H])
            dendrogram(tree, ax=dax, no_labels=True, color_threshold=0, above_threshold_color="0.25")
            dax.set_xlim(0, 10 * nk)
            dax.set_axis_off()
        if cell_borders:
            _cell_borders(ax, len(pp["labels"]), len(pats), border_color)
        else:
            ax.grid(False)
        ax.set_xlim(-0.5, len(pats) - 0.5)
        ax.set_ylim(len(pp["labels"]) - 0.5, -1.8)
        fig.text((x0 + heat_w / 2) / W, (y0 + axes_h + tree_h + 0.12) / H,
                 f"{pp['t']}: {len(pats)} patients (>= {min_cells} cells), top {len(pp['labels'])} genes",
                 fontsize=title_fs, ha="center", va="bottom")

    # 4. Colour bar, legend, title
    cax = fig.add_axes([(ncol * cell_w + 0.3) / W, 0.35, 0.22 / W, 0.3])
    _sat_colorbar(fig, im, z_lim, "robust z-score across patients\n(log CPM, median / MAD)", cax=cax, fontsize=fs)
    shown = [g for g in groups if g in plotted]   # groups with at least one patient column
    handles = [plt.Line2D([], [], marker="s", ls="", ms=fs * 1.1, color=palette.get(g, "0.6"), label=g) for g in shown]
    fig.legend(handles=handles, loc="upper center", ncol=max(len(shown), 1), fontsize=1.3 * fs, frameon=False,
               bbox_to_anchor=(0.5, 1 - (header_h - 1.3 * fs * 2.2 / 72 - 0.2) / H))
    fig.suptitle(sup, fontsize=1.5 * fs, y=1 - 0.15 / H, va="top")
    _log(f"plot_smide_patients: figure {W:.1f} x {H:.1f} in (in Quarto, set the chunk's fig-width / fig-height to these)")
    return fig


def _stars(q):
    return "****" if q < 1e-4 else "***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else ""


def _group_tests(values, groups):
    """Kruskal-Wallis across groups and pairwise two-sided Mann-Whitney (Wilcoxon rank-sum) p-values."""
    from itertools import combinations

    from scipy.stats import kruskal, mannwhitneyu

    vals = [np.asarray(values[g], float) for g in groups]
    ok = [v for v in vals if len(v) > 1]
    kw = kruskal(*ok).pvalue if len(ok) > 1 and np.ptp(np.concatenate(ok)) > 0 else np.nan
    pairs = []
    for a, b in combinations(range(len(groups)), 2):
        va, vb = vals[a], vals[b]
        p = mannwhitneyu(va, vb, alternative="two-sided").pvalue if len(va) > 1 and len(vb) > 1 else np.nan
        pairs.append((groups[a], groups[b], p))
    return kw, pairs


def _bh_q(p):
    """Benjamini-Hochberg q-values (NaN p stay NaN)."""
    from .de import _bh

    p = np.asarray(p, float)
    return _bh(p, int(np.isfinite(p).sum()))


def _box_by_group(ax, values, groups, palette, fontsize, ylabel=None, title=None, brackets=(), percent=False,
                  seed=0):
    """Box (no outliers) + jittered points per group on `ax`; `brackets`: [(group a, group b, label)]."""
    rng = np.random.default_rng(seed)
    data = [np.asarray(values.get(g, []), float) for g in groups]
    bp = ax.boxplot(data, positions=range(len(groups)), widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color="k", lw=0.8), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                    boxprops=dict(lw=0.6))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(palette.get(g, "0.7"))
        patch.set_alpha(0.75)
    for k, d in enumerate(data):
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(d)), d, s=fontsize * 0.9, c="k", lw=0, zorder=3)
    ax.set_xticks(range(len(groups)), groups, fontsize=fontsize, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=fontsize * 0.9)
    if percent:
        from matplotlib.ticker import PercentFormatter

        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    nonempty = [d for d in data if len(d)]
    allv = np.concatenate(nonempty) if nonempty else np.array([0.0])
    top = float(np.max(allv))
    span = float(np.ptp(allv)) or max(abs(top), 1e-3)
    for k, (a, b, label) in enumerate(brackets):
        ia, ib = groups.index(a), groups.index(b)
        y = top + span * (0.08 + 0.12 * k)
        ax.plot([ia, ia, ib, ib], [y - span * 0.02, y, y, y - span * 0.02], c="k", lw=0.6)
        ax.text((ia + ib) / 2, y, label, ha="center", va="bottom", fontsize=fontsize)
    if brackets:
        ax.set_ylim(top=top + span * (0.12 + 0.12 * len(brackets)))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(False)


def _facet_grid(n, ncols, figsize, fontsize):
    import matplotlib.pyplot as plt

    ncols = ncols or min(n, 3)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (ncols * fontsize * 0.27, nrows * fontsize * 0.33)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    return fig, axes


def plot_group_proportions(obs, celltype_col, group_col, patient_col="sample_id", cells=None, celltypes=None,
                           groups=None, palette=None, min_cells=20, ncols=None, fontsize=7, figsize=None,
                           show_ns=False):
    """Share of each cell type among a patient's cells, by group: one facet per cell type, box + patient points.

    The denominator is each patient's selected cells (`cells`, e.g. ``"ng_glom_id"``: the cells of the tagged
    glomeruli, so the shares are of glomerular cells). Per facet: Kruskal-Wallis across the groups (in the title) and
    pairwise Wilcoxon rank-sum tests, BH-adjusted over all facets' pairs; brackets with stars mark q < 0.05.

    Parameters
    ----------
    obs
        Cell metadata (or an AnnData: its ``obs``).
    celltype_col, group_col, patient_col
        obs columns of the cell types, the group (e.g. diagnosis) and the patient.
    cells
        The cells counted: an obs column (booleans, or any column for its non-missing cells) or a boolean array.
    celltypes, groups
        Facets and their order (default: the types among the cells, by abundance) and group order (default: sorted).
    palette
        ``{group: colour}``.
    min_cells
        Patients with fewer selected cells are left out.
    fontsize, figsize, ncols
        Text size (all text), figure size in inches (default from fontsize), facets per row.
    show_ns
        Also draw brackets for q >= 0.05 (labelled "ns").

    Returns
    -------
    (Figure, DataFrame)
        The figure and the statistics: per cell type the Kruskal-Wallis p, each pair's p and BH q.
    """
    from ._palette import scanpy_palette
    from .de import _cell_mask

    obs = getattr(obs, "obs", obs)
    m = _cell_mask(obs, cells) & obs[group_col].notna().to_numpy() & obs[patient_col].notna().to_numpy()
    o = obs.loc[m, [celltype_col, group_col, patient_col]].astype(str)
    if groups is None:
        groups = sorted(o[group_col].unique())
    groups = list(groups)
    o = o[o[group_col].isin(groups)]
    n = o.groupby(patient_col).size()
    o = o[o[patient_col].isin(n.index[n >= min_cells])]
    share = pd.crosstab(o[patient_col], o[celltype_col], normalize="index")
    grp = o.drop_duplicates(patient_col).set_index(patient_col)[group_col].reindex(share.index)
    types = list(celltypes) if celltypes is not None else list(o[celltype_col].value_counts().index)
    palette = palette or scanpy_palette(groups)

    def vals(t):
        return {g: (share.loc[grp == g, t].to_numpy() if t in share else np.array([])) for g in groups}

    rows = []
    for t in types:
        kw, pairs = _group_tests(vals(t), groups)
        rows += [dict(celltype=t, kruskal_p=kw, group_a=a, group_b=b, p=p) for a, b, p in pairs]
    stats = pd.DataFrame(rows)
    stats["q"] = _bh_q(stats["p"].to_numpy())
    fig, axes = _facet_grid(len(types), ncols, figsize, fontsize)
    for k, (ax, t) in enumerate(zip(axes.ravel(), types)):
        st = stats[stats["celltype"] == t]
        br = [(r.group_a, r.group_b, _stars(r.q) or "ns") for r in st.itertuples()
              if np.isfinite(r.q) and (r.q < 0.05 or show_ns)]
        kw = st["kruskal_p"].iloc[0] if len(st) else np.nan
        _box_by_group(ax, vals(t), groups, palette, fontsize, title=f"{t}\nKW p = {kw:.2g}", brackets=br,
                      percent=True, ylabel="share of cells" if k % axes.shape[1] == 0 else None)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.5)
    return fig, stats


def plot_group_values(values, group, groups=None, palette=None, ylabel=None, title=None, fontsize=7, figsize=None,
                      show_ns=False):
    """One value per patient (or per unit) by group: box + points, Kruskal-Wallis in the title, BH-adjusted pairwise
    Wilcoxon brackets for q < 0.05 (e.g. the mean number of cells per glomerulus of each patient).

    Parameters
    ----------
    values, group
        Series of the values and of their groups (same index).
    groups, palette
        Group order and ``{group: colour}``.

    Returns
    -------
    (Figure, DataFrame)
    """
    import matplotlib.pyplot as plt

    from ._palette import scanpy_palette

    d = pd.DataFrame({"v": values, "g": group}).dropna()
    d["g"] = d["g"].astype(str)
    groups = list(groups) if groups is not None else sorted(d["g"].unique())
    palette = palette or scanpy_palette(groups)
    v = {g: d.loc[d["g"] == g, "v"].to_numpy() for g in groups}
    kw, pairs = _group_tests(v, groups)
    stats = pd.DataFrame([dict(kruskal_p=kw, group_a=a, group_b=b, p=p) for a, b, p in pairs])
    stats["q"] = _bh_q(stats["p"].to_numpy())
    br = [(r.group_a, r.group_b, _stars(r.q) or "ns") for r in stats.itertuples()
          if np.isfinite(r.q) and (r.q < 0.05 or show_ns)]
    fig, ax = plt.subplots(figsize=figsize or (fontsize * 0.3, fontsize * 0.33))
    _box_by_group(ax, v, groups, palette, fontsize, ylabel=ylabel,
                  title=(f"{title}\n" if title else "") + f"KW p = {kw:.2g}", brackets=br)
    fig.tight_layout(pad=0.3)
    return fig, stats


def plot_radial_position(obs, celltype_col, dist_col="ng_dist", celltypes=None, palette=None, cells=None, xmax=1.3,
                         fontsize=7, figsize=None, xlabel="radial position (0 = centre, 1 = edge)"):
    """Where each cell type sits inside a structure: density of its cells' radial position (0 = centre, 1 = edge,
    e.g. ``ng_dist`` of the tagged glomeruli), one ridge per cell type, its median marked.

    Parameters
    ----------
    obs
        Cell metadata (or an AnnData: its ``obs``).
    celltype_col, dist_col
        obs columns of the cell types and of the radial position.
    celltypes
        Ridges, top to bottom (default: by median position, the most central first).
    palette
        ``{cell type: colour}``.
    cells
        Cells to include, as ``plot_group_proportions``.
    xmax
        Positions beyond are drawn at `xmax`.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    from ._palette import scanpy_palette
    from .de import _cell_mask

    obs = getattr(obs, "obs", obs)
    m = _cell_mask(obs, cells) & obs[dist_col].notna().to_numpy()
    o = obs.loc[m, [celltype_col, dist_col]]
    o = o.assign(t=o[celltype_col].astype(str), d=o[dist_col].astype(float).clip(upper=xmax))
    types = list(celltypes) if celltypes is not None else list(o.groupby("t")["d"].median().sort_values().index)
    palette = palette or scanpy_palette(types)
    x = np.linspace(0, xmax, 300)
    fig, ax = plt.subplots(figsize=figsize or (fontsize * 0.38, fontsize * 0.05 * (len(types) + 2)))
    for k, t in enumerate(types):
        d = o.loc[o["t"] == t, "d"].to_numpy()
        if len(d) < 5:
            continue
        dens = gaussian_kde(d)(x)
        y = 0.85 * dens / dens.max()
        base = len(types) - 1 - k
        ax.fill_between(x, base, base + y, color=palette.get(t, "0.6"), alpha=0.8, lw=0)
        ax.plot(x, base + y, c="k", lw=0.5)
        med = float(np.median(d))
        ax.plot([med, med], [base, base + float(np.interp(med, x, y))], c="k", lw=0.8)
        ax.text(xmax * 1.01, base + 0.1, f"{t} (n = {len(d):,})", fontsize=fontsize, va="bottom")
    ax.set_yticks([])
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.1, len(types))
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize * 0.9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(False)
    fig.tight_layout(pad=0.3)
    return fig


def _short(term, n):
    term = str(term)
    return term if len(term) <= n else term[: n - 1].rstrip() + "…"


def plot_gsea_dotplot(gsea, library, celltypes=None, contrasts=None, n_top=5, fdr=0.05, clip="auto", max_chars=55,
                      fontsize=11, title=None):
    """Gene set enrichment across DE families: gene sets (rows) x families (cell type x contrast, columns).

    Rows: the union of each family's `n_top` gene sets with FDR < `fdr` (smallest FDR first) in `library`, ordered by
    hierarchical clustering of their NES profiles. Dot colour: NES (red = enriched among genes higher in the first group
    named), on the smide heatmaps' blue - white - red scale, saturated beyond `clip`; dot size: -log10 FDR; a dot with a
    black edge is FDR < `fdr`; no dot: the set was not tested in that family (too few of its genes ranked).

    Parameters
    ----------
    gsea
        ``frzspatial.enrich.smide_gsea``'s table.
    library
        The gene set library shown (e.g. ``"MSigDB_Hallmark_2020"``).
    clip
        |NES| colour limit: a number, or ``"auto"`` (the 95th percentile of |NES| shown, at least 1.5).
    max_chars
        Longer gene set names are shortened.

    Returns
    -------
    Figure
    """
    import textwrap

    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage

    from ._utils import _log

    g = gsea[gsea["library"] == library].copy()
    if g.empty:
        raise ValueError(f"no results for library {library!r}")
    types = list(celltypes) if celltypes is not None else list(dict.fromkeys(g["celltype"]))
    cons = list(contrasts) if contrasts is not None else list(dict.fromkeys(g["contrast"]))
    g["family"] = g["celltype"].astype(str) + " | " + g["contrast"].astype(str)
    order = [f"{t} | {c}" for t in types for c in cons]
    sig = g[(g["fdr"] < fdr) & g["family"].isin(order)].sort_values(["fdr", "pval"])
    terms = sig.groupby("family", sort=False).head(n_top)["term"].unique()
    if not len(terms):
        raise ValueError(f"no gene set with FDR < {fdr} in {library}")
    nes = g.pivot_table(index="term", columns="family", values="nes").reindex(index=terms, columns=order)
    q = g.pivot_table(index="term", columns="family", values="fdr").reindex(index=terms, columns=order)
    if len(nes) > 2:
        rows = nes.index[leaves_list(linkage(nes.fillna(0).to_numpy(), "average", metric="euclidean"))]
        nes, q = nes.loc[rows], q.loc[rows]
    if clip == "auto":
        clip = max(1.5, float(np.ceil(np.nanpercentile(np.abs(nes.to_numpy()), 95) * 2) / 2))
    fs = fontsize
    labels = [_short(t, max_chars) for t in nes.index]
    row_h = fs * 1.5 / 72
    head_w = max(len(t) for t in types) * 1.1 * fs * 0.62 / 72 + 0.1   # bold cell-type header over its columns
    col_w = max(fs * 2.0 / 72, head_w / len(cons))
    axes_h, heat_w = row_h * (len(nes) + 1.3), col_w * len(order)
    label_w = max(len(x) for x in labels) * fs * 0.55 / 72 + 0.2
    xlab_h = max(len(o.split(" | ")[1]) for o in order) * fs * 0.6 / 72 + 0.3
    legend_w = 0.6 + fs * 9 / 72
    W = label_w + heat_w + legend_w
    ttl = title or f"{library}: top {n_top} gene sets per family (FDR < {fdr}; edge: significant)"
    ttl = "\n".join(textwrap.wrap(ttl, width=max(30, int(W * 72 / (1.2 * fs * 0.55)))))
    title_h = (ttl.count("\n") + 1) * 1.2 * fs * 1.45 / 72 + 0.25
    H = title_h + axes_h + xlab_h + 0.2
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([label_w / W, xlab_h / H, heat_w / W, axes_h / H])
    yy, xx = np.meshgrid(range(len(nes)), range(len(order)), indexing="ij")
    v, qq = nes.to_numpy(), q.to_numpy()
    ok = np.isfinite(v)
    smax = (min(row_h, col_w) * 72 * 0.95) ** 2
    size = np.clip(-np.log10(np.clip(qq, 1e-10, 1)), 0, 4) / 4 * smax * 0.85 + smax * 0.15
    im = ax.scatter(xx[ok], yy[ok], c=np.clip(v[ok], -clip, clip), s=size[ok], cmap=_smide_cmap(), vmin=-clip,
                    vmax=clip, edgecolors=np.where(qq[ok] < fdr, "k", "none"), linewidths=0.6, zorder=3)
    ax.set_xticks(range(len(order)), [o.split(" | ")[1].replace("_vs_", " vs ") for o in order], rotation=90,
                  fontsize=fs)
    ax.set_yticks(range(len(nes)), labels, fontsize=fs)
    _cell_borders(ax, len(nes), len(order), "0.85", lw=0.4)
    for k, t in enumerate(types):
        ax.text(len(cons) * k + (len(cons) - 1) / 2, -0.8, t, ha="center", va="bottom", fontsize=1.1 * fs,
                weight="bold")
        if k:
            ax.axvline(len(cons) * k - 0.5, c="k", lw=1.2, zorder=4)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(len(nes) - 0.5, -1.8)
    # colour bar and size legend
    x0 = (label_w + heat_w + 0.3) / W
    cax = fig.add_axes([x0, xlab_h / H + 0.55 * axes_h / H, 0.18 / W, 0.35 * axes_h / H])
    _sat_colorbar(fig, im, clip, "NES", cax=cax, fontsize=fs)
    for k, (qv, lab) in enumerate([(0.05, "0.05"), (1e-2, "0.01"), (1e-4, "1e-4")]):
        s = np.clip(-np.log10(qv), 0, 4) / 4 * smax * 0.85 + smax * 0.15
        ax_l = fig.add_axes([x0, xlab_h / H + (0.4 - 0.1 * k) * axes_h / H, 0.3 / W, 0.08 * axes_h / H])
        ax_l.scatter([0.5], [0.5], s=s, c="0.5", edgecolors="k", linewidths=0.6)
        ax_l.text(1.6, 0.5, f"FDR {lab}", va="center", fontsize=fs)
        ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1); ax_l.set_axis_off()
    fig.suptitle(ttl, fontsize=1.2 * fs, y=1 - 0.08 / H, va="top")
    _log(f"plot_gsea_dotplot: figure {W:.1f} x {H:.1f} in (in Quarto, set the chunk's fig-width / fig-height to these)")
    return fig


def plot_gsea_bars(gsea, celltype, contrast, libraries=None, n_top=8, fdr=0.05, max_chars=60, fontsize=11,
                   title=None):
    """The top enriched gene sets of one DE family: NES bars (red: higher in the first group named), FDR on the right,
    one block per library, the `n_top` sets with the smallest FDR of each (only FDR < `fdr`).

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    g = gsea[(gsea["celltype"] == celltype) & (gsea["contrast"] == contrast) & (gsea["fdr"] < fdr)]
    libs = list(libraries) if libraries is not None else list(dict.fromkeys(g["library"]))
    blocks = [(lib, g[g["library"] == lib].sort_values(["fdr", "pval"]).head(n_top).sort_values("nes"))
              for lib in libs]
    blocks = [(lib, b) for lib, b in blocks if len(b)]
    if not blocks:
        raise ValueError(f"no gene set with FDR < {fdr} for {celltype} / {contrast}")
    fs = fontsize
    n = sum(len(b) for _, b in blocks) + len(blocks)
    labels = [_short(t, max_chars) for _, b in blocks for t in b["term"]]
    label_w = max(len(x) for x in labels) * fs * 0.55 / 72 + 0.2
    W, H = label_w + 3.0, n * fs * 1.6 / 72 + 1.1
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([label_w / W, 0.6 / H, 2.3 / W, (H - 0.9) / H])
    y, ticks, ylabels, heads = 0, [], [], []
    shown = pd.concat([b for _, b in blocks])
    lo = min(0.0, float(shown["nes"].min())) * 1.1
    hi = max(0.0, float(shown["nes"].max())) * 1.1
    lo, hi = (lo, hi + 0.3) if lo == 0 else (lo - 0.3, hi) if hi == 0 else (lo, hi)
    for lib, b in blocks:
        ticks.append(y)
        ylabels.append(lib.replace("_", " "))
        heads.append(len(ticks) - 1)
        y += 1
        for _, r in b.iterrows():
            ax.barh(y, r["nes"], color="#c0392b" if r["nes"] > 0 else "#2a6fbb", height=0.7)
            ax.text(hi + (hi - lo) * 0.03, y, f"{r['fdr']:.1g}" if r["fdr"] > 0 else "<0.001", fontsize=fs * 0.9,
                    va="center")
            ticks.append(y)
            ylabels.append(_short(r["term"], max_chars))
            y += 1
    ax.set_yticks(ticks, ylabels, fontsize=fs)
    for k in heads:
        ax.get_yticklabels()[k].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(lo, hi)
    ax.axvline(0, c="k", lw=0.6)
    ax.set_xlabel("NES", fontsize=fs)
    ax.tick_params(axis="x", labelsize=fs * 0.9)
    ax.text(hi + (hi - lo) * 0.03, -0.9, "FDR", fontsize=fs * 0.9, weight="bold", va="center")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(False)
    fig.suptitle(title or f"{celltype}: {contrast.replace('_vs_', ' vs ')}", fontsize=1.15 * fs, y=1 - 0.05 / H,
                 va="top")
    return fig
