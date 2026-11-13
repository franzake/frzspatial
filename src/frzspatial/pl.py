"""Static plots of single slides with spatialdata-plot, y up as in the AtoMx slide view.

Each function returns the matplotlib axes without showing them: call ``plt.show()`` or ``ax.figure.savefig(...)``.
"""

import warnings

import pandas as pd
import spatialdata_plot  # noqa: F401  (adds .pl to SpatialData objects)

from .core import coordinate_system, core_box

__all__ = ["plot_slide_overview", "plot_core", "annotation_palette", "plot_annotations", "plot_composition",
           "plot_umap", "plot_niche_composition", "plot_share_by_group"]


def plot_slide_overview(sdata, color, title=None, figsize=(12, 20), dpi=200, fontsize=8, slide=None, palette=None,
                        labels=True):
    """All cells of a single slide coloured by an obs column, each group labelled at its median position.

    Cells are rasterised by datashader at ``figsize × dpi`` pixels: at ~1000 px across a slide one pixel is about
    one cell, so raise `dpi` for sharper cores.

    Parameters
    ----------
    sdata
        A single slide, or a merged object with `slide`.
    color
        obs column, e.g. ``"core_str"`` or ``"fov"``.
    title
        Plot title (default: `color`).
    figsize, dpi
        Figure size (inches) and resolution.
    fontsize
        Size of the group labels.
    slide
        For a merged object: the slide to draw (its ``cell_boundaries-<slide>`` in ``global-<slide>``).
    palette
        ``{value: colour}`` for `color` (default: the column's stored colours, e.g. from a UMAP, else scanpy's); cells
        with other values are drawn grey, e.g. to highlight a few cell types.
    labels
        Name each group at its median position (e.g. cores); False shows a legend instead (e.g. cell types).
    """
    from ._palette import column_palette

    suffix, cs = coordinate_system(sdata, slide) if slide else ("", "global")
    table = sdata.tables["table"]
    obs = table.obs
    if slide and "slide" in obs:
        obs = obs[obs["slide"].astype(str) == slide]
    extra = {}
    if palette is not None or hasattr(obs[color], "cat"):  # categorical: values not in the palette are drawn grey
        pal = column_palette(table, color, palette)
        groups = [g for g in pal if g in set(obs[color].dropna().astype(str))]
        extra = {"groups": groups, "palette": [pal[g] for g in groups], "na_color": "#DEDEDE"}
    ax = (
        sdata.pl.render_shapes(f"cell_boundaries{suffix}", color=color, method="datashader", **extra)
        .pl.show(
            coordinate_systems=cs,
            legend_loc=None if labels else "right margin",
            title=title or (f"{slide}: {color}" if slide else color),
            figsize=figsize,
            dpi=dpi,
            show=False,
            return_ax=True))

    if labels:
        centres = obs.groupby(color, observed=True)[["CenterX_global_px", "CenterY_global_px"]].median()
        for label, (x, y) in centres.iterrows():
            ax.text(x, y, label, fontsize=fontsize, ha="center")

    ax.invert_yaxis()
    return ax


def plot_core(sdata, core, genes=None, color=None, palette=None, title=None, core_key="core_str", slide=None,
              fill_alpha=0.5, image=True, outline_alpha=0.6, ax=None, cmap="viridis", layer=None, outline_width=None):
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
        entry, side by side (returns their axes).
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
        Draw the composite image under the cells.
    ax
        Matplotlib axes to draw into (default: a new figure).
    cmap, layer
        Colour map and table layer for a gene `color`.
    outline_width
        Width of the cell borders (default: 0.3 on the image, 0.6 without it).

    Examples
    --------
    >>> plot_core(slides["MN2"], "C4R7", genes={"NPHS2": "magenta"})  # doctest: +SKIP
    >>> plot_core(slides["MN2"], "C4R7", color="manual_annotation",
    ...           palette=annotation_palette(classes))  # doctest: +SKIP
    >>> plot_core(usc, "C4R7", slide="MN2", color=["CFH", "VCAM1"], image=False)  # doctest: +SKIP
    """
    if isinstance(color, (list, tuple)):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, len(color), figsize=(8 * len(color), 8), squeeze=False)
            ax = ax[0]
        return [plot_core(sdata, core, genes=genes, color=c, palette=palette, title=title, core_key=core_key,
                          slide=slide, fill_alpha=fill_alpha, image=image, outline_alpha=outline_alpha, ax=a,
                          cmap=cmap, layer=layer, outline_width=outline_width)
                for c, a in zip(color, ax)]
    suffix, cs = coordinate_system(sdata, slide) if slide else ("", "global")
    shapes = f"cell_boundaries{suffix}"
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
    return ax


def annotation_palette(classes, used=None):
    """``{class: colour}`` in the colours of :func:`frzspatial.viewer.annotate_core`'s layers (scanpy's default_20).

    used: an obs column; only the classes that occur in it are returned (for :func:`plot_core`'s `palette`).
    """
    from ._palette import DEFAULT_20

    pal = {c: DEFAULT_20[i % len(DEFAULT_20)] for i, c in enumerate(classes)}
    if used is not None:
        present = set(used.dropna().astype(str))
        pal = {c: v for c, v in pal.items() if c in present}
    return pal


def plot_annotations(sdata, classes, column="manual_annotation", title="", core_key="core_str"):
    """Each annotated core, its cells filled by class in the colours of the napari layers.

    To check which cells each drawn polygon took. Shows one figure per core with annotated cells.

    Parameters
    ----------
    sdata
        A single slide, after :func:`frzspatial.annotations_to_obs`.
    classes
        The annotation classes, in the order given to :func:`frzspatial.viewer.annotate_core`.
    column
        obs column of the annotations.
    title
        Prefix of each figure's title (e.g. the slide name).
    """
    import matplotlib.pyplot as plt

    obs = sdata.tables["table"].obs
    palette = annotation_palette(classes, used=obs[column])
    for core in obs.loc[obs[column].notna(), core_key].unique():
        plot_core(sdata, core, color=column, palette=palette, title=f"{title} {core}: {column}".strip(),
                  core_key=core_key)
        plt.show()


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


def plot_umap(table, color, palette=None, labels=True, legend=False, title="", size=3, fontsize=12):
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
        Name each group on the plot, overlaps moved apart (adjustText).
    legend
        Also a legend above the plot (useful with `palette`).

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
    ax = sc.pl.umap(table, color=color, title=title if single else None, size=size, frameon=True, show=False,
                    legend_loc="on data" if single and labels else "right margin",
                    legend_fontsize=fontsize, legend_fontoutline=2)
    if single and labels:
        from adjustText import adjust_text

        adjust_text(ax.texts, ax=ax)
    if single and legend:
        cats = table.obs[color].cat.categories
        handles = [plt.Line2D([], [], marker="o", ls="", color=c) for c in table.uns[f"{color}_colors"]]
        ax.legend(handles, cats, title=color, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=4, fontsize=10,
                  frameon=False, columnspacing=0.6, handletextpad=0.2)
    plt.tight_layout()
    return ax


def plot_niche_composition(comp, enrich, figsize=(16, 7)):
    """Heatmaps of each niche's label shares and log2 enrichments (:func:`frzspatial.niches.niche_composition`).

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    sns.heatmap(comp, cmap="Blues", ax=axs[0], cbar_kws={"label": "share of the niche's cells"})
    sns.heatmap(enrich, cmap="RdBu_r", center=0, vmin=-3, vmax=3, ax=axs[1], cbar_kws={"label": "log2 enrichment"})
    axs[0].set_title("Niche composition")
    axs[1].set_title("Cell-type enrichment per niche")
    fig.tight_layout()
    return fig


def plot_share_by_group(obs, column, group="group_mn", case=("slide", "sample_id"), exclude=("Placenta",),
                        ncols=5, figsize=(14, 8)):
    """Each case's share of cells in each value of `column` (e.g. niches), by `group`: boxplots with one point per
    case, one panel per value (every value counted for every case, 0 when absent).

    Parameters
    ----------
    obs
        Cell metadata with `column`, `group` and the `case` columns.
    exclude
        Groups left out (e.g. placenta orientation cores); cells without a group or value are left out too.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    d = obs[[*case, group, column]].dropna()
    d = d[~d[group].astype(str).isin(exclude)]
    values = [v for v in obs[column].cat.categories if v in set(d[column])] if hasattr(obs[column], "cat") \
        else sorted(d[column].unique())
    share = pd.crosstab([d[c].astype(str) for c in case] + [d[group].astype(str)], d[column], normalize="index")
    share = share.reindex(columns=values, fill_value=0)
    groups = sorted(share.index.get_level_values(-1).unique())
    from ._palette import scanpy_palette

    colours = scanpy_palette(groups)
    nrows = int(np.ceil(len(values) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, v in zip(axs.flat, values):
        per = [share.loc[share.index.get_level_values(-1) == g, v].to_numpy() for g in groups]
        ax.boxplot(per, showfliers=False, widths=0.6)
        for i, (g, y) in enumerate(zip(groups, per), start=1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(y)), y, s=8, color=colours[g], zorder=3)
        ax.set_xticks(range(1, len(groups) + 1), groups, rotation=45, ha="right", fontsize=8)
        ax.set_title(str(v), fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    for ax in list(axs.flat)[len(values):]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel("share of the case's cells", fontsize=8)
    fig.tight_layout()
    return fig
