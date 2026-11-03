"""Static plots of single slides with spatialdata-plot, y up as in the AtoMx slide view.

Each function returns the matplotlib axes without showing them: call ``plt.show()`` or ``ax.figure.savefig(...)``.
"""

import spatialdata_plot  # noqa: F401  (adds .pl to SpatialData objects)

from .core import core_box

__all__ = ["plot_slide_overview", "plot_core", "annotation_palette"]


def plot_slide_overview(sdata, color, title=None, figsize=(12, 20), dpi=200, fontsize=8):
    """All cells of a single slide coloured by an obs column, each group labelled at its median position.

    Cells are rasterised by datashader at ``figsize × dpi`` pixels: at ~1000 px across a slide one pixel is about
    one cell, so raise `dpi` for sharper cores.

    Parameters
    ----------
    sdata
        A single slide.
    color
        obs column, e.g. ``"core_str"`` or ``"fov"``.
    title
        Plot title (default: `color`).
    figsize, dpi
        Figure size (inches) and resolution.
    fontsize
        Size of the group labels.
    """
    ax = (
        sdata.pl.render_shapes("cell_boundaries", color=color, method="datashader")
        .pl.show(
            coordinate_systems="global",
            legend_loc=None,
            title=title or color,
            figsize=figsize,
            dpi=dpi,
            show=False,
            return_ax=True))

    obs = sdata.tables["table"].obs
    centres = obs.groupby(color, observed=True)[["CenterX_global_px", "CenterY_global_px"]].median()
    for label, (x, y) in centres.iterrows():
        ax.text(x, y, label, fontsize=fontsize, ha="center")

    ax.invert_yaxis()
    return ax


def plot_core(sdata, core, genes=None, color=None, palette=None, title=None, core_key="core_str"):
    """One core of a single slide: its composite image, cell outlines and selected transcripts.

    Parameters
    ----------
    sdata
        A single slide.
    core
        Core name (``obs[core_key]``).
    genes
        ``{gene: colour}`` of transcripts to draw.
    color
        obs column to fill the cells by (e.g. ``"manual_annotation"``); without it, cells are outlined in yellow.
    palette
        ``{value: colour}`` for `color`; cells with other values or NA are left unfilled.
    title
        Plot title (default: `core`).

    Examples
    --------
    >>> plot_core(slides["MN2"], "C4R7", genes={"NPHS2": "magenta"})  # doctest: +SKIP
    >>> plot_core(slides["MN2"], "C4R7", color="manual_annotation",
    ...           palette=annotation_palette(classes))  # doctest: +SKIP
    """
    comps = [name for name in sdata.images if name.endswith("_image")]
    elements = comps + ["cell_boundaries", "table"] + (["transcripts"] if genes else [])
    sub = sdata.subset(elements).query.bounding_box(**core_box(sdata, core, core_key))

    if color:
        pl = sub.pl.render_images().pl.render_shapes(
            "cell_boundaries",
            color=color,
            groups=list(palette) if palette else None,
            palette=list(palette.values()) if palette else None,
            fill_alpha=0.5,
            outline_alpha=0.6,
            outline_width=0.3)
    else:
        pl = sub.pl.render_images().pl.render_shapes(
            "cell_boundaries",
            fill_alpha=0,
            outline_alpha=0.6,
            outline_width=0.3,
            outline_color="yellow")
    if genes:
        pl = pl.pl.render_points(
            "transcripts",
            color="target",
            groups=list(genes),
            palette=genes,
            size=0.3,
            alpha=0.6)
    ax = pl.pl.show(coordinate_systems="global", title=title or core, show=False, return_ax=True)

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
