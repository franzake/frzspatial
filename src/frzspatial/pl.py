"""Static plots of single slides with spatialdata-plot, y up as in the AtoMx slide view.

Each function returns the matplotlib axes without showing them: call ``plt.show()`` or ``ax.figure.savefig(...)``.
"""

import warnings

import numpy as np
import pandas as pd
import spatialdata_plot  # noqa: F401  (adds .pl to SpatialData objects)

from .core import coordinate_system, core_box

__all__ = ["plot_core", "plot_composition", "plot_umap", "plot_smide_volcano", "plot_smide_heatmap",
           "plot_smide_patients"]


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
                     exclude_groups=("Placenta",)):
    """Composition of each sample in `color` (e.g. cell types), stacked bars of fractions, one panel per `group`.

    Parameters
    ----------
    obs
        Cell metadata with `sample`, `group` and `color` (rows without a sample, or in `exclude_groups`, are left out).
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
    groups = list(dict.fromkeys(frac.index.get_level_values(0)))
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
    fig.legend(handles, frac.columns, title=color, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, frameon=False)
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
                       fontsize=12, ax=None):
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
    ax.scatter(d.loc[~sig, "log2FC"], d.loc[~sig, "_y"], s=8, c="0.75", lw=0, label=f"FDR >= {fdr}")
    case, ref = d["case"].iloc[0], d["reference"].iloc[0]
    hi = f"higher in {case}" if isinstance(case, str) else "positive"
    lo = f"higher in {ref}" if isinstance(ref, str) else "negative"
    ax.scatter(d.loc[up, "log2FC"], d.loc[up, "_y"], s=14, c="#c0392b", lw=0, label=f"{hi} ({int(up.sum())})")
    ax.scatter(d.loc[down, "log2FC"], d.loc[down, "_y"], s=14, c="#2a6fbb", lw=0, label=f"{lo} ({int(down.sum())})")
    ax.axvline(0, c="0.6", lw=0.8)
    if y == "FDR_BH":
        ax.axhline(-np.log10(fdr), c="0.6", lw=0.8, ls="--")
    for _, r in d[sig].nsmallest(n_labels, "FDR_BH").iterrows():
        ax.annotate(r["gene"], (r["log2FC"], r["_y"]), xytext=(3, 3), textcoords="offset points",
                    fontsize=fontsize - 2, path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])
    ax.set_xlabel(f"adjusted log2 fold change ({case} vs {ref})" if isinstance(case, str) else "coefficient / log(2)",
                  fontsize=fontsize)
    ax.set_ylabel("-log10 FDR (BH)" if y == "FDR_BH" else "-log10 p", fontsize=fontsize)
    ax.set_title(title or f"smiDE: {celltype}, {contrast} ({len(d)} valid genes)", fontsize=fontsize + 2)
    ax.legend(frameon=False, fontsize=fontsize - 2)
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


def plot_smide_heatmap(res, celltypes=None, contrasts=None, n_top=8, clip="auto", fdr=0.05, fontsize=11,
                       cell_borders=True, border_color="0.3", figsize=None, title=None):
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
        Base size of all text (gene and contrast labels; titles, colour bar and cell-type labels scale from it).
    cell_borders, border_color
        Draw a thin line between every two cells, in this colour.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage

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
    if len(l2fc) > 2:
        rows = l2fc.index[leaves_list(linkage(l2fc.fillna(0).clip(-clip, clip).to_numpy(), "average",
                                              metric="correlation"))]
        l2fc, q = l2fc.loc[rows], q.loc[rows]
    best = sig.drop_duplicates("gene").set_index("gene")["log2FC"]   # log2FC in each gene's most significant family
    cmap = _smide_cmap().copy()
    cmap.set_bad("0.88")
    fs = fontsize
    fig, ax = plt.subplots(figsize=figsize or (4.5 + 0.85 * len(order), 3 + 0.032 * fs * len(l2fc)))
    im = ax.imshow(l2fc.clip(-clip, clip).to_numpy(), cmap=cmap, vmin=-clip, vmax=clip, aspect="auto",
                   interpolation="none")
    yy, xx = np.nonzero((q < fdr).to_numpy())
    ax.scatter(xx, yy, s=5 * fs, c="k", marker="*", linewidths=0, zorder=3)
    ax.set_xticks(range(len(order)), [o.split(" | ")[1].replace("_vs_", " vs ") for o in order], rotation=90,
                  fontsize=fs)
    ax.set_yticks(range(len(l2fc)), [f"{g} ({best[g]:+.1f})" for g in l2fc.index], fontsize=fs * 0.85)
    if cell_borders:
        _cell_borders(ax, len(l2fc), len(order), border_color)
    else:
        ax.grid(False)
    for k, t in enumerate(types):
        ax.text(len(cons) * k + (len(cons) - 1) / 2, -1.0, t, ha="center", va="bottom", fontsize=fs * 1.3,
                weight="bold")
        if k:
            ax.axvline(len(cons) * k - 0.5, c="k", lw=1.5, zorder=4)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(len(l2fc) - 0.5, -1.6)
    _sat_colorbar(fig, im, clip, "adjusted log2 fold change", ax=ax, fontsize=fs)
    ax.set_title(title or f"smiDE: top {n_top} genes per family (* FDR < {fdr}; grey: not tested; "
                 "label: log2FC in its most significant family)", fontsize=fs * 1.3, pad=3 * fs)
    fig.tight_layout()
    return fig


def plot_smide_patients(sdata, res, celltype_col, contrast, group_col, patient_col="sample_id", celltypes=None,
                        cells=None, n_genes=24, min_cells=10, layer="counts", groups=None, palette=None, z_lim=2,
                        cluster=True, column_labels=None, fontsize=13, cell_borders=True, border_color="0.3",
                        table="table", title=None):
    """The top genes of one ``smide`` contrast at the level of the patients: one heatmap per cell type.

    Each column is one patient's pseudobulk of the cell type (sum of raw counts over its cells, log CPM), each row a
    gene (the `n_genes` smallest p of `contrast` in that type, ordered by fold change; label: log2FC and * when
    FDR < 0.05). Values are robust z-scores per gene across patients (median / MAD), so a few extreme patients don't
    flatten the others' contrast. Patients are grouped by `group_col` and, with `cluster`, ordered by hierarchical
    clustering within each group (average linkage, Euclidean), with each group's tree drawn above it. An effect
    carried by most patients of a group reads as a block.

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
    column_labels
        obs column whose value labels each patient's column (e.g. ``"slide"``). None (default): no labels, since
        patient ids can be identifiers.
    fontsize
        Base size of all text (gene labels; panel titles, legend, colour bar and figure title scale from it). Gene
        labels shrink further when many genes would not fit.
    cell_borders, border_color
        Draw a thin line between every two cells, in this colour.

    Returns
    -------
    Figure
        Its height grows with `n_genes` (two panel rows: about 8 + 0.58 x n_genes inches; in Quarto set the chunk's
        fig-height to ``fig.get_size_inches()[1]``).
    """
    import matplotlib.pyplot as plt
    import scipy.sparse as sp
    from matplotlib.transforms import blended_transform_factory
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
    cmap = _smide_cmap()
    ncol = 2 if len(types) > 1 else 1
    nrow = int(np.ceil(len(types) / ncol))
    # layout in inches: each panel row = its heatmap, with its trees, title and column labels around it
    heat_h = max(3.5, 0.25 * n_genes + 1.2)
    tree_h = 0.16 * heat_h
    top_h, gap_h, bot_h = 1.6 + tree_h + 0.8, tree_h + 1.8, 1.2
    fig_h = top_h + nrow * heat_h + (nrow - 1) * gap_h + bot_h
    fig, axes = plt.subplots(nrow, ncol, figsize=(15 * ncol, fig_h), squeeze=False)
    fs = fontsize
    label_size = min(fs, 520 / n_genes)
    im, plotted = None, set()
    for ax, t in zip(axes.ravel(), types):
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
        z = ((z - med) / np.where(mad > 1e-9, mad, z.std(axis=0).clip(min=1e-9))).T
        order, edges, trees = [], [], []
        for grp in groups:
            cols = np.flatnonzero(pats["g"].astype(str).to_numpy() == grp)
            if cluster and len(cols) > 2:
                tree = linkage(np.clip(z[:, cols], -z_lim, z_lim).T, "average", metric="euclidean")
                cols = cols[leaves_list(tree)]
                trees.append((len(order), len(cols), tree))
            order += list(cols)
            edges.append(len(order) - 0.5)
        z, pats = z[:, order], pats.iloc[order]
        plotted |= set(pats["g"].astype(str))
        im = ax.imshow(np.clip(z, -z_lim, z_lim), cmap=cmap, vmin=-z_lim, vmax=z_lim, aspect="auto",
                       interpolation="none")
        ax.scatter(range(len(pats)), [-1.1] * len(pats), c=[palette.get(g, "0.6") for g in pats["g"].astype(str)],
                   marker="s", s=180, clip_on=False)
        ax.set_yticks(range(len(top)), [f"{gname} ({l:+.1f}){' *' if q < 0.05 else ''}" for gname, l, q
                                        in zip(top["gene"], top["log2FC"], top["FDR_BH"])], fontsize=label_size)
        if column_labels:
            ax.set_xticks(range(len(pats)), pats[column_labels].astype(str), rotation=90, fontsize=fs * 0.7)
        else:
            ax.set_xticks([])
        for e in edges[:-1]:
            ax.axvline(e, c="k", lw=2, zorder=4)
        for start, k, tree in trees:
            dax = ax.inset_axes([start - 0.5, 1.0, k, 0.16], transform=blended_transform_factory(ax.transData, ax.transAxes))
            dendrogram(tree, ax=dax, no_labels=True, color_threshold=0, above_threshold_color="0.25")
            dax.set_xlim(0, 10 * k)
            dax.set_axis_off()
        if cell_borders:
            _cell_borders(ax, len(top), len(pats), border_color)
        else:
            ax.grid(False)
        ax.set_xlim(-0.5, len(pats) - 0.5)
        ax.set_ylim(len(top) - 0.5, -1.8)
        ax.set_title(f"{t}: {len(pats)} patients (>= {min_cells} cells), top {len(top)} genes", fontsize=fs * 1.4,
                     pad=(tree_h + 0.15) * 72)
    for ax in axes.ravel()[len(types):]:
        ax.set_axis_off()
    H = fig.get_size_inches()[1]
    fig.subplots_adjust(left=0.07, right=0.92, bottom=bot_h / H, top=1 - top_h / H, wspace=0.22,
                        hspace=gap_h / heat_h)
    cax = fig.add_axes([0.94, 0.35, 0.01, 0.3])
    _sat_colorbar(fig, im, z_lim, "robust z-score across patients\n(log CPM, median / MAD)", cax=cax, fontsize=fs)
    shown = [g for g in groups if g in plotted]   # groups with at least one patient column
    handles = [plt.Line2D([], [], marker="s", ls="", ms=14, color=palette.get(g, "0.6"), label=g) for g in shown]
    fig.legend(handles=handles, loc="upper center", ncol=max(len(shown), 1), fontsize=fs * 1.4, frameon=False,
               bbox_to_anchor=(0.5, 1 - 1.0 / H))
    fig.suptitle(title or f"smiDE {contrast.replace('_vs_', ' vs ')}: patient pseudobulk (robust z of log CPM, "
                 "patients clustered within each group; label: log2FC, * FDR < 0.05)", fontsize=fs * 1.7, y=1 - 0.3 / H)
    return fig
