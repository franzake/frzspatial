"""napari windows: view a SpatialData object, and draw annotations on a core or slide.

Needs the ``napari`` extra: ``pip install "frzspatial[napari]"``. Importing this module loads Qt, so import it only
where a window is opened.

Notes
-----
When Python runs inside another program (R through reticulate, Jupyter), the napari window belongs to that
program: close it with the window's own close button. Quitting with Cmd-Q quits the host application's Qt loop and
can hang it, and force-quitting the window kills the host (e.g. the R session). Annotations are saved after every
change, so a hang loses nothing.
"""

import anndata as ad
import geopandas as gpd
import napari
import numpy as np
import shapely
from napari_spatialdata import Interactive

from ._palette import DEFAULT_20
from .annotate import annotation_file, annotations_to_obs, read_annotations, write_annotations
from .core import build_core, coordinate_system

__all__ = ["open_napari", "annotate_core", "patch_anndata"]


def patch_anndata():
    """Restore ``AnnData._get_X``, removed in anndata 0.13 but still called by napari-spatialdata 0.7 (colour by
    gene). Called by :func:`open_napari` and :func:`annotate_core`; call it yourself before opening napari-spatialdata
    another way."""
    if not hasattr(ad.AnnData, "_get_X"):
        ad.AnnData._get_X = lambda self, use_raw=False, layer=None: self.X if layer is None else self.layers[layer]


def open_napari(sdata):
    """Show `sdata` in a napari-spatialdata window; returns when the window is closed.

    The window can open behind other windows (look for it with Cmd-Tab / Alt-Tab).

    Examples
    --------
    >>> open_napari(build_core(slides["MN2"], "MN2", "C4R7"))  # doctest: +SKIP
    """
    patch_anndata()
    Interactive(sdata).run()


def annotate_core(sdata, slide, core, classes, annot_dir="results/annotations", column="manual_annotation",
                  colors=None, table="table", downsample=None):
    """Draw regions on a core (or a whole slide) in napari, one layer per class; label the cells on closing.

    The core is stitched by :func:`frzspatial.build_core` and opened with its composite, cell outlines (not for a
    whole slide: ~70k outlines are slow) and one Shapes layer ``annot: <class>`` per class, holding this core's
    saved polygons. Draw polygons (P) or rectangles (R) in the layer of a class; ellipses are ignored. Each change
    is saved to ``<annot_dir>/<slide>_<core>.geojson`` (global px). When the window is closed, ``obs[column]`` is
    rebuilt from every saved file (:func:`frzspatial.annotations_to_obs`).

    Parameters
    ----------
    sdata
        A single slide or a merged object.
    slide
        Slide name.
    core
        Core name, or None for the whole slide (downsampled; saved to ``<slide>_slide.geojson``).
    classes
        Class names, in priority order for overlaps; classes found only in the saved file are added.
    annot_dir, column, table
        As in :func:`frzspatial.annotations_to_obs`.
    colors, downsample
        As in :func:`frzspatial.build_core`.

    Returns
    -------
    GeoDataFrame
        The core's polygons (``class``, ``geometry``).
    """
    path = annotation_file(annot_dir, slide, core)
    _, cs = coordinate_system(sdata, slide)

    # 0. The stitched core, composite and outlines first (an empty Shapes layer added first would fix the camera at
    #    the origin, far from the core)
    patch_anndata()
    viz = Interactive(build_core(sdata, slide, core, table=table, colors=colors, downsample=downsample),
                      headless=True)
    viewer = napari.current_viewer()
    viz.switch_coordinate_system(cs)
    for element in ["core_image", "cell_boundaries"] if core else ["core_image"]:
        viz.add_element(element, cs)

    # 1. One Shapes layer per class with the saved polygons; napari's world coordinates are global px, (y, x)
    saved = read_annotations(path)
    classes = list(classes) + [c for c in saved["class"].unique() if c not in classes]
    for i, c in enumerate(classes):
        polys = [np.asarray(g.exterior.coords)[:-1, ::-1] for g in saved.loc[saved["class"] == c, "geometry"]]
        colour = DEFAULT_20[i % len(DEFAULT_20)]
        viewer.add_shapes(polys or None, shape_type="polygon", name=f"annot: {c}", edge_color=colour,
                          face_color=colour + "40", edge_width=8)
    viewer.reset_view()

    # 2. Polygons and rectangles back to global px (x, y), saved after every change
    def save(event=None):
        rows = []
        for c in classes:
            layer = viewer.layers[f"annot: {c}"]
            for verts, kind in zip(layer.data, layer.shape_type):
                if kind in ("polygon", "rectangle") and len(verts) >= 3:
                    rows.append({"class": c,
                                 "geometry": shapely.Polygon(np.asarray(verts)[:, -2:][:, ::-1]).buffer(0)})
        annot = gpd.GeoDataFrame(rows, columns=["class", "geometry"], geometry="geometry")
        write_annotations(annot, path)
        return annot

    for c in classes:
        viewer.layers[f"annot: {c}"].events.data.connect(save)

    # 3. The window; returns when it is closed
    viz.run()
    annot = save()
    viewer.close()
    counts = annot["class"].value_counts().to_dict() if len(annot) else ""
    print(f"{slide} {core or 'whole slide'}: {len(annot)} polygons", counts)

    # 4. The obs column from every saved file
    annotations_to_obs(sdata, classes, annot_dir, column, table)
    return annot
