"""Cores of a TMA slide: their bounding box, and a slide's elements in a merged object.

Two kinds of objects are supported everywhere:

* a **single slide** (e.g. one value of :func:`frzspatial.load_slides`): unsuffixed elements (``fov<N>_image``,
  ``cell_boundaries``, ``transcripts``) in the coordinate system ``"global"``;
* a **merged object** (several slides concatenated with ``spatialdata.concatenate``): each slide's elements
  suffixed ``-<slide>`` and in their own coordinate system ``"global-<slide>"``, so slides never overlap.
"""

__all__ = ["core_box", "coordinate_system"]


def coordinate_system(sdata, slide):
    """``("-<slide>", "global-<slide>")`` for a merged object, ``("", "global")`` for a single slide."""
    merged = f"cell_boundaries-{slide}" in sdata.shapes
    return (f"-{slide}", f"global-{slide}") if merged else ("", "global")


def core_box(sdata, core, core_key="core_str", table="table", slide=None, slide_key="slide"):
    """Bounding-box query arguments for one core: the rectangle around its cells.

    Use as ``sdata.query.bounding_box(**core_box(sdata, "C4R7"))``. For a merged object, give the `slide`: its cells
    (``obs[slide_key] == slide``) and coordinate system ``global-<slide>`` are used.

    Raises
    ------
    ValueError
        If no cell has ``obs[core_key] == core`` (on that slide).
    """
    obs = sdata.tables[table].obs
    sel = obs[core_key].astype(str) == core
    if slide is not None and slide_key in obs:
        sel &= obs[slide_key].astype(str) == slide
    xy = obs.loc[sel, ["CenterX_global_px", "CenterY_global_px"]]
    if xy.empty:
        raise ValueError(f"no cells in core {core}" + (f" of slide {slide}" if slide else ""))
    return dict(
        axes=("x", "y"),
        min_coordinate=xy.min().to_numpy(),
        max_coordinate=xy.max().to_numpy(),
        target_coordinate_system=coordinate_system(sdata, slide)[1] if slide else "global")


