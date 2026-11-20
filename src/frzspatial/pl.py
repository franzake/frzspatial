"""Static plots of single slides with spatialdata-plot, y up as in the AtoMx slide view.

Each function returns the matplotlib axes without showing them: call ``plt.show()`` or ``ax.figure.savefig(...)``.
"""

import warnings

import numpy as np
import pandas as pd
import spatialdata_plot  # noqa: F401  (adds .pl to SpatialData objects)

from .core import coordinate_system, core_box

__all__ = ["plot_core", "plot_composition", "plot_umap"]


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


