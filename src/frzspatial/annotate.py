"""Hand annotations: polygons saved as GeoJSON, turned into a cell-level obs column.

Annotations are stored one file per core in an annotation folder, named ``<slide>_<core>.geojson``
(``<slide>_slide.geojson`` for shapes drawn on a whole slide), with one ``class`` property per polygon and
coordinates in the slide's global px. They are drawn with :func:`frzspatial.viewer.annotate_core`, but any GeoJSON
in that layout works (e.g. exported from QuPath after converting to global px).

This module does not need napari.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

__all__ = ["annotation_file", "read_annotations", "write_annotations", "annotations_to_obs"]


def annotation_file(annot_dir, slide, core=None):
    """Path of a core's annotation file: ``<annot_dir>/<slide>_<core>.geojson`` (core None: ``<slide>_slide``)."""
    return Path(annot_dir) / f"{slide}_{core or 'slide'}.geojson"


def read_annotations(path):
    """Polygons of one annotation file as a GeoDataFrame (``class``, ``geometry``), one row per polygon.

    A missing file gives an empty GeoDataFrame; multipolygons are split into polygons.
    """
    path = Path(path)
    if not path.exists():
        return gpd.GeoDataFrame(columns=["class", "geometry"], geometry="geometry")
    return gpd.read_file(path).explode(index_parts=False).reset_index(drop=True)


def write_annotations(annot, path):
    """Save polygons (GeoDataFrame with ``class`` and ``geometry``) to `path`; an empty one deletes the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(annot):
        annot.to_file(path, driver="GeoJSON")
    elif path.exists():
        path.unlink()


def annotations_to_obs(sdata, classes, annot_dir="results/annotations", column="manual_annotation", table="table",
                       slide_key="slide", core_key="core_str"):
    """Label each cell with the class of the annotated polygon around its centroid.

    Every ``<slide>_<core>.geojson`` in `annot_dir` is applied to the cells with ``obs[slide_key] == slide`` and
    ``obs[core_key] == core`` (all cells of the slide for ``<slide>_slide.geojson``). Files of slides or cores not
    in the table are skipped, so the same folder can serve each slide and a merged object.

    Parameters
    ----------
    sdata
        A single slide or a merged object.
    classes
        Class names in priority order: where polygons overlap, the class later in the list wins.
    annot_dir
        Folder of the GeoJSON files.
    column
        obs column to write.
    table, slide_key, core_key
        Table element and its obs columns of slide and core names. Positions are ``CenterX/Y_global_px``.

    Returns
    -------
    Series
        Cells per class. ``obs[column]`` is set in place: categorical with every class as a category (used or not,
        so that tables annotated separately still share them when merged), NA outside every polygon.

    Examples
    --------
    >>> annotations_to_obs(slides["MN2"], ["Glomerulus", "Artery"])  # doctest: +SKIP
    Glomerulus    126
    Artery          0
    """
    obs = sdata.tables[table].obs
    order = {c: i for i, c in enumerate(classes)}
    manual = pd.Series(pd.NA, index=obs.index, dtype="object")
    for f in sorted(Path(annot_dir).glob("*.geojson")):
        slide, core = f.stem.split("_", 1)
        in_core = (obs[slide_key].astype(str) == slide).to_numpy()
        if core != "slide":
            in_core &= (obs[core_key].astype(str) == core).to_numpy()
        if not in_core.any():
            continue
        xy = obs.loc[in_core, ["CenterX_global_px", "CenterY_global_px"]].to_numpy()
        for c, g in sorted(read_annotations(f).groupby("class")["geometry"], key=lambda cg: order.get(cg[0], -1)):
            hit = shapely.contains_xy(shapely.union_all(np.asarray(g)), xy[:, 0], xy[:, 1])
            manual.loc[obs.index[in_core][hit]] = c
    extra = sorted(set(manual.dropna()) - set(classes))
    obs[column] = pd.Categorical(manual, categories=list(classes) + extra)
    return obs[column].value_counts()
