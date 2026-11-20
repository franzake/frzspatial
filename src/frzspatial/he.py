"""Post-CosMx H&E aligned to the cells, one crop per TMA core, with spatialdata transformations.

The alignment is stored the scverse way: each core's H&E crop is a SpatialData image whose transformation
(``Affine`` then the slide's own) puts it on the cells. The points it is fitted on come from tissue shape:

1. **Slide**: tissue masks of the H&E overview (saturation, Otsu; the post-CosMx H&E is pale) and of the cells, on a
   `grid_um` grid; the 8 right-angle orientations, each with its best shift (``phase_cross_correlation``); then each
   core paired with its H&E tissue blob, and the affine fitted on those landmarks with spatialdata's
   ``get_transformation_between_landmarks`` (refitted twice without outliers).
2. **Core**: each core's cell footprint registered on the H&E tissue around it at `level` (``phase_cross_correlation``
   on a `core_grid_um` grid), which corrects the 15-40 µm a single affine leaves on a TMA.
"""

import re

import numpy as np
import pandas as pd

from ._utils import PX_UM, _log, _slide_rows

__all__ = ["align_he_cores"]


def _tissue(rgb, sigma):
    """Tissue mask of an RGB H&E image: smoothed saturation above its Otsu threshold, holes filled."""
    from scipy.ndimage import binary_fill_holes
    from skimage.color import rgb2hsv
    from skimage.filters import gaussian, threshold_otsu

    sat = gaussian(rgb2hsv(rgb)[..., 1], sigma=sigma)
    return binary_fill_holes(sat > threshold_otsu(sat))


def _on_grid(mask, um_per_px, grid_um):
    from skimage.transform import resize

    shape = np.maximum((np.array(mask.shape) * um_per_px / grid_um).astype(int), 1)
    return resize(mask.astype(float), shape, anti_aliasing=True) > 0.5


def _footprint(points_um, shape, origin_um, grid_um, close=1):
    """Cells as a mask on the grid (rows = y): the cells' grid squares, closed and filled."""
    from scipy.ndimage import binary_closing, binary_fill_holes

    ii = np.floor((points_um - origin_um) / grid_um).astype(int)
    ok = (ii >= 0).all(1) & (ii[:, 0] < shape[1]) & (ii[:, 1] < shape[0])
    m = np.zeros(shape, bool)
    m[ii[ok, 1], ii[ok, 0]] = True
    return binary_fill_holes(binary_closing(m, iterations=close))


def _slide_affine(cells_px, cores, he_mask, grid_um, px_um, min_blob_mm2):
    """Affine (3 x 3) from CosMx px to H&E µm, and the number of landmark pairs it was fitted on."""
    from skimage.measure import label, regionprops
    from skimage.registration import phase_cross_correlation
    from spatialdata.models import PointsModel
    from spatialdata.transformations import get_transformation_between_landmarks

    um = cells_px * px_um
    ctr = um.mean(0)
    best = None
    for k in range(4):
        for flip in (1, -1):
            th = np.pi / 2 * k
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([1, flip])
            P = (um - ctr) @ R.T
            origin = P.min(0)
            cm = _footprint(P, he_mask.shape, origin, grid_um)
            shift = phase_cross_correlation(he_mask.astype(float), cm.astype(float), normalization=None)[0]
            score = (np.roll(cm, shift.astype(int), axis=(0, 1)) & he_mask).sum() / max(cm.sum(), 1)
            if best is None or score > best[0]:
                best = (score, R, origin, shift)
    score, R, origin, shift = best

    blobs = regionprops(label(he_mask))
    blob_xy = np.array([b.centroid[::-1] for b in blobs]) * grid_um
    big = np.array([b.area for b in blobs]) * grid_um ** 2 >= min_blob_mm2 * 1e6
    core_px = pd.DataFrame(cells_px, index=cores).groupby(level=0).mean()
    guess = (core_px.to_numpy() * px_um - ctr) @ R.T - origin + shift[::-1] * grid_um   # core centroids, H&E µm
    limit = 300.0
    for _ in range(3):   # pair each core with its nearest large blob, fit, then pair again with the fit
        d = np.hypot(*(blob_xy[None] - guess[:, None]).transpose(2, 0, 1))
        d[:, ~big] = np.inf
        j, dmin = d.argmin(1), d.min(1)
        keep = dmin < limit
        if keep.sum() < 3:
            raise ValueError(f"only {keep.sum()} cores paired with H&E tissue: is it the right H&E?")
        A = get_transformation_between_landmarks(references_coords=PointsModel.parse(blob_xy[j[keep]]),
                                                 moving_coords=PointsModel.parse(core_px.to_numpy()[keep]))
        M = A.to_affine_matrix(("x", "y"), ("x", "y"))
        guess = (np.c_[core_px.to_numpy(), np.ones(len(core_px))] @ M.T)[:, :2]
        res = np.hypot(*(guess[keep] - blob_xy[j[keep]]).T)
        limit = max(3 * np.median(res), 60.0)
    return M, int(keep.sum()), float(score)


def align_he_cores(sdata, svs, slide, level=1, grid_um=20, core_grid_um=5, half_um=900, pad_um=50, max_shift_um=150,
                   min_overlap=0.3, min_cells=50, min_blob_mm2=0.05, core_key="core_id", prefix="he_", table="table",
                   px_um=PX_UM):
    """Add a slide's post-CosMx H&E to `sdata`, one image per core, each aligned to that core's cells.

    Parameters
    ----------
    sdata
        A single slide, or a merged object with `slide`; its cell polygons give the coordinate systems.
    svs
        The H&E whole-slide image (Aperio SVS; only its main series is read, never the label or macro images).
    slide
        The slide's name (``obs["slide"]``).
    level
        Pyramid level of the crops (1: 0.55 µm/px on a 0.14 µm/px scan) and of the core registration.
    grid_um, core_grid_um
        Grid of the slide-level and of the core-level tissue masks.
    half_um, pad_um
        Half-width of the H&E window searched around each core; margin of each crop around its cells.
    max_shift_um, min_overlap, min_cells
        A core gets an image when its own shift is at most `max_shift_um` and at least `min_overlap` of its cells
        fall on H&E tissue (else its tissue is missing or moved on the H&E); cores with fewer cells are skipped.
    min_blob_mm2
        Smallest H&E tissue blob used as a landmark.
    core_key, prefix
        obs column of the cores; the images are named ``<prefix><core>`` (e.g. ``he_MN2_C4R7``), replacing any of
        that name.

    Returns
    -------
    DataFrame
        One row per core: cells, shift (µm), the share of its cells on H&E tissue before and after its own shift,
        and whether it got an image. The images are added to ``sdata.images`` in place.
    """
    import tifffile
    import zarr
    from skimage.registration import phase_cross_correlation
    from spatialdata.models import Image2DModel
    from spatialdata.transformations import Affine, Sequence, get_transformation

    from .core import coordinate_system

    obs = sdata.tables[table].obs
    rows = _slide_rows(obs, slide)
    rows = np.ones(len(obs), bool) if rows is None else rows
    cells_px = np.asarray(sdata.tables[table].obsm["spatial"], float)[rows]
    cores = obs.loc[rows, core_key].astype(str).to_numpy()
    suffix, _ = coordinate_system(sdata, slide)
    shapes_cs = get_transformation(sdata.shapes[f"cell_boundaries{suffix}"], get_all=True)

    with tifffile.TiffFile(svs) as tif:
        levels = tif.series[0].levels
        mpp = float(re.search(r"MPP\s*=\s*([\d.]+)", tif.pages[0].description).group(1))
        um_at = lambda lv: mpp * levels[0].shape[0] / levels[lv].shape[0]   # noqa: E731
        overview = levels[-1].asarray()
        he_mask = _on_grid(_tissue(overview, sigma=4), um_at(len(levels) - 1), grid_um)
        M, n_land, score = _slide_affine(cells_px, cores, he_mask, grid_um, px_um, min_blob_mm2)
        _log(f"{slide}: H&E slide fit on {n_land} cores ({score:.0%} of cells on tissue before the fit), "
             f"{np.sqrt(abs(np.linalg.det(M[:2, :2]))):.4f} µm per CosMx px")

        img = zarr.open(tif.series[0].aszarr(level=level), mode="r")
        umh = um_at(level)
        out = []
        for core in pd.unique(cores):
            P = cells_px[cores == core]
            if len(P) < min_cells:
                out.append((core, len(P), np.nan, np.nan, np.nan, np.nan, False))
                continue
            p = (np.c_[P, np.ones(len(P))] @ M.T)[:, :2]   # the cells in H&E µm
            c = p.mean(0)
            x0, y0 = np.maximum(((c - half_um) / umh).astype(int), 0)
            x1, y1 = ((c + half_um) / umh).astype(int)
            tg = _on_grid(_tissue(np.asarray(img[y0:y1, x0:x1]), sigma=3), umh, core_grid_um)
            cm = _footprint(p, tg.shape, np.array([x0, y0]) * umh, core_grid_um, close=2)
            sh = phase_cross_correlation(tg.astype(float), cm.astype(float), normalization=None,
                                         upsample_factor=4)[0]
            shift = sh[::-1] * core_grid_um   # x, y in µm
            before = (cm & tg).sum() / max(cm.sum(), 1)
            after = (np.roll(cm, np.round(sh).astype(int), axis=(0, 1)) & tg).sum() / max(cm.sum(), 1)
            ok = np.hypot(*shift) <= max_shift_um and after >= min_overlap
            out.append((core, len(P), shift[0], shift[1], before, after, ok))
            if not ok:
                continue
            # Crop around the shifted cells; crop px -> H&E µm -> (minus the core's shift) -> CosMx px -> slide cs
            cx0, cy0 = np.maximum(((p.min(0) + shift - pad_um) / umh).astype(int), 0)
            cx1, cy1 = ((p.max(0) + shift + pad_um) / umh).astype(int)
            to_px = np.linalg.inv(M) @ np.array([[1, 0, cx0 * umh - shift[0]], [0, 1, cy0 * umh - shift[1]],
                                                 [0, 0, 1]]) @ np.diag([umh, umh, 1])
            A = Affine(to_px, input_axes=("x", "y"), output_axes=("x", "y"))
            sdata.images[f"{prefix}{core}"] = Image2DModel.parse(
                np.asarray(img[cy0:cy1, cx0:cx1]).transpose(2, 0, 1), dims=("c", "y", "x"),
                transformations={cs: Sequence([A, t]) for cs, t in shapes_cs.items()})

    res = pd.DataFrame(out, columns=["core", "cells", "shift_x_um", "shift_y_um", "on_tissue_slide_fit",
                                     "on_tissue_core_fit", "aligned"]).set_index("core")
    _log(f"{slide}: {res['aligned'].sum()} of {len(res)} cores aligned to the H&E")
    return res
