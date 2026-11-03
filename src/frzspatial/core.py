"""Cores of a TMA slide: their bounding box and a stitched, napari-ready SpatialData per core or slide.

Two kinds of objects are supported everywhere:

* a **single slide** (e.g. one value of :func:`frzspatial.load_slides`): unsuffixed elements (``fov<N>_image``,
  ``cell_boundaries``, ``transcripts``) in the coordinate system ``"global"``;
* a **merged object** (several slides concatenated with ``spatialdata.concatenate``): each slide's elements
  suffixed ``-<slide>`` and in their own coordinate system ``"global-<slide>"``, so slides never overlap.
"""

import numpy as np
import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, TableModel
from spatialdata.transformations import Affine, get_transformation

from ._utils import _log

__all__ = ["build_core", "core_box", "coordinate_system"]


def coordinate_system(sdata, slide):
    """``("-<slide>", "global-<slide>")`` for a merged object, ``("", "global")`` for a single slide."""
    merged = f"cell_boundaries-{slide}" in sdata.shapes
    return (f"-{slide}", f"global-{slide}") if merged else ("", "global")


def core_box(sdata, core, core_key="core_str", table="table"):
    """Bounding-box query arguments for one core of a single slide: the rectangle around its cells.

    Use as ``sdata.query.bounding_box(**core_box(sdata, "C4R7"))``.

    Raises
    ------
    ValueError
        If no cell has ``obs[core_key] == core``.
    """
    obs = sdata.tables[table].obs
    xy = obs.loc[obs[core_key].astype(str) == core, ["CenterX_global_px", "CenterY_global_px"]]
    if xy.empty:
        raise ValueError(f"no cells in core {core}")
    return dict(
        axes=("x", "y"),
        min_coordinate=xy.min().to_numpy(),
        max_coordinate=xy.max().to_numpy(),
        target_coordinate_system="global")


def build_core(sdata, slide, core=None, table="table", colors=None, scale_factors=(2, 2, 2), downsample=None,
               core_key="core_str", slide_key="slide"):
    """One core, or a whole slide, as a small SpatialData object for napari.

    The core's FOVs are stitched into one composite and one mask layer. FOVs are placed by their AtoMx offsets
    (x = col + dx, y = cy - row) in integer px, so stitching needs no resampling.

    Parameters
    ----------
    sdata
        A single slide or a merged object (see the module docs).
    slide
        Slide name: selects the elements and coordinate system of a merged object, and the cells with
        ``obs[slide_key] == slide``.
    core
        Core name (``obs[core_key]``), or None for the whole slide.
    table
        Table element name.
    colors
        ``{column: {category: colour}}``, stored as the table's ``<column>_colors``: napari-spatialdata only colours
        a categorical column reliably with them.
    scale_factors
        Pyramid of the image and masks: each level halves.
    downsample
        Keep every n-th pixel of each FOV (the transformation scales it back to the slide's px). Default 1 for a
        core and 8 for a whole slide (a full-resolution slide is ~40 GB of RGB).
    core_key, slide_key
        obs columns of the core and slide names (`slide_key` is only used for a merged object).

    Returns
    -------
    SpatialData
        ``core_image`` (multiscale RGB), ``core_labels`` (each cell's mask value = its integer id in the table and
        polygons), ``cell_boundaries``, ``transcripts`` and ``table`` (only cells with a polygon, as napari needs one
        shape per row; unused categories dropped; the original names in ``obs["cell_name"]``), in the slide's
        coordinate system. Needs ``obs["fov"]`` and ``obs["slide_ID"]``.

    Examples
    --------
    >>> from frzspatial.viewer import open_napari  # doctest: +SKIP
    >>> open_napari(build_core(slides["MN2"], "MN2", "C4R7"))  # doctest: +SKIP
    """
    sfx, cs = coordinate_system(sdata, slide)
    tb = sdata.tables[table]
    obs = tb.obs
    in_slide = obs[slide_key].astype(str) == slide if sfx else pd.Series(True, index=obs.index)
    in_core = in_slide if core is None else in_slide & (obs[core_key].astype(str) == core)
    if not in_core.any():
        raise ValueError(f"no cells in {slide} {core or ''}")
    ds = downsample or (8 if core is None else 1)
    fovs = sorted(obs.loc[in_core, "fov"].astype(int).unique())

    # 0. FOV offsets from the elements' transformations, and the canvas covering all the core's FOVs
    place = {}
    for fov in fovs:
        name = f"fov{fov}_image{sfx}"
        if name not in sdata.images:
            continue
        m = get_transformation(sdata.images[name], cs).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        h, w = sdata.images[name].shape[-2:]
        place[fov] = (int(round(m[0, 2])), int(round(m[1, 2])), h, w)  # dx, cy, height, width
    if not place:
        raise ValueError(f"no FOV images for {slide} {core or ''}")
    x0 = min(dx for dx, cy, h, w in place.values())
    x1 = max(dx + w for dx, cy, h, w in place.values())
    y1 = max(cy for dx, cy, h, w in place.values())
    y0 = min(cy - h for dx, cy, h, w in place.values())
    H, W = -(-(y1 - y0) // ds), -(-(x1 - x0) // ds)  # canvas px, rounded up
    to_cs = {cs: Affine(np.array([[ds, 0, x0], [0, -ds, y1], [0, 0, 1]]), input_axes=("x", "y"),
                        output_axes=("x", "y"))}

    # 1. Cells with a polygon in the core's box, renumbered 1..n (0 = background in the masks)
    #    (whole slide: no query, which would take minutes and read every transcript into memory)
    if core is None:
        shp = sdata.shapes[f"cell_boundaries{sfx}"]
        t = tb[in_slide.to_numpy()]
        pts = sdata.points[f"transcripts{sfx}"]
    else:
        box = dict(axes=("x", "y"), min_coordinate=[x0, y0], max_coordinate=[x1, y1], target_coordinate_system=cs)
        sub = sdata.subset([f"cell_boundaries{sfx}", table]).query.bounding_box(**box)
        shp = sub.shapes[f"cell_boundaries{sfx}"]
        t = sub.tables[table]
        pts = sdata.subset([f"transcripts{sfx}"]).query.bounding_box(**box).points[f"transcripts{sfx}"]
    ids = pd.Series(np.arange(1, len(shp) + 1), index=shp.index)

    # 2. Stitched composite and masks; mask values (cell_ID within the FOV) mapped to the cells' ids
    image = np.zeros((3, H, W), np.uint8)
    labels = np.zeros((H, W), np.int32)
    slide_id = int(obs.loc[in_slide, "slide_ID"].iloc[0])
    for fov, (dx, cy, h, w) in place.items():
        img = np.asarray(sdata.images[f"fov{fov}_image{sfx}"].data)[:, ::ds, ::ds]
        r, c = (y1 - cy) // ds, (dx - x0) // ds
        h, w = min(img.shape[1], H - r), min(img.shape[2], W - c)
        image[:, r:r + h, c:c + w] = np.maximum(image[:, r:r + h, c:c + w], img[:, :h, :w])
        lab_name = f"fov{fov}_labels{sfx}"
        if lab_name in sdata.labels:
            lab = np.asarray(sdata.labels[lab_name].data)[::ds, ::ds][:h, :w]
            lut = np.zeros(int(lab.max()) + 1, np.int32)
            names = pd.Index([f"c_{slide_id}_{fov}_{v}" for v in range(len(lut))])
            hit = names.isin(ids.index)
            lut[hit] = ids[names[hit]].to_numpy()
            mapped = lut[lab]
            tile = labels[r:r + h, c:c + w]
            tile[mapped > 0] = mapped[mapped > 0]
    _log(f"{slide} {core or 'whole slide'}: {len(place)} FOVs stitched, {W} x {H} px (1/{ds}), {len(ids)} cells")

    # 3. Table rows with a polygon, ids as instance key, annotating the polygons
    t = t[t.obs["cell"].astype(str).isin(ids.index)].copy()
    t.obs["cell_name"] = t.obs["cell"].astype(str).to_numpy()
    t.obs["cell"] = ids[t.obs["cell_name"]].to_numpy()
    t.obs["region"] = pd.Categorical(["cell_boundaries"] * t.n_obs)
    t.uns.pop("spatialdata_attrs", None)
    for col in t.obs.columns[[isinstance(d, pd.CategoricalDtype) for d in t.obs.dtypes]]:
        kept = t.obs[col].cat.remove_unused_categories()
        if f"{col}_colors" in t.uns and len(t.uns[f"{col}_colors"]) == len(t.obs[col].cat.categories):
            used = t.obs[col].cat.categories.isin(kept.cat.categories)
            t.uns[f"{col}_colors"] = list(np.asarray(t.uns[f"{col}_colors"])[used])
        t.obs[col] = kept
    for col, palette in (colors or {}).items():
        t.uns[f"{col}_colors"] = [palette[c] for c in t.obs[col].cat.categories]

    return SpatialData(
        images={"core_image": Image2DModel.parse(image, dims=("c", "y", "x"), c_coords=["r", "g", "b"],
                                                  scale_factors=list(scale_factors), transformations=to_cs)},
        labels={"core_labels": Labels2DModel.parse(labels, dims=("y", "x"), scale_factors=list(scale_factors),
                                                   transformations=to_cs)},
        shapes={"cell_boundaries": shp.set_axis(ids.to_numpy())},
        points={"transcripts": pts},
        tables={"table": TableModel.parse(t, region="cell_boundaries", region_key="region", instance_key="cell")})
