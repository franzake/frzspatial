"""Register a whole-slide H&E (Aperio SVS) onto the CosMx cells, from tissue shape alone.

Needs the ``he`` extra (scikit-image): ``pip install "frzspatial[he]"``.
"""

import re
import warnings

import dask.array as da
import numpy as np
import pandas as pd
import tifffile
import zarr
from spatialdata.models import Image2DModel, ShapesModel
from spatialdata.transformations import Affine, Scale, Sequence

from ._utils import PX_UM, _log
from .io import add_slide_view

__all__ = ["register_he", "add_he_image"]


def _points_mask(xy_um, grid_um):
    """Tissue mask (rows = y) of points on a grid_um grid, its origin (µm) and each point's cell."""
    from scipy.ndimage import binary_closing, binary_fill_holes

    origin = xy_um.min(0)
    idx = np.floor((xy_um - origin) / grid_um).astype(int)
    mask = np.zeros(idx.max(0)[::-1] + 1, dtype=bool)
    mask[idx[:, 1], idx[:, 0]] = True
    return binary_fill_holes(binary_closing(mask, iterations=2)), origin, idx


def _xcorr_fit(xy_um, he_mask, grid_um):
    """Best translation of points onto the H&E mask: (fraction of point tissue on H&E tissue, t_um)."""
    from scipy.signal import fftconvolve

    mask, origin, _ = _points_mask(xy_um, grid_um)
    cc = fftconvolve(he_mask.astype(np.float32), mask[::-1, ::-1].astype(np.float32), mode="full")
    a, b = np.unravel_index(np.argmax(cc), cc.shape)
    shift = np.array([b - mask.shape[1] + 1, a - mask.shape[0] + 1]) * grid_um
    return cc[a, b] / mask.sum(), shift - origin


def _rotation(theta, flip):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    return np.array([[c, -s], [s, c]]) @ np.diag([1, flip])


def register_he(cells_px, he_overview, he_um_per_overview_px, px_um=PX_UM, grid_um=50, min_blob_mm2=0.05):
    """Fit the CosMx → H&E transform from tissue shape alone.

    1. Tissue masks on a `grid_um` grid: the H&E by Otsu threshold, the cells by occupancy.
    2. Rigid search: the 8 right-angle rotations and flips, then a ±4° sweep, each with its best translation by FFT
       cross-correlation (pixel sizes are known, so the scale is fixed).
    3. Landmarks: each tissue blob (e.g. a TMA core) fully covered by cells is paired with its H&E blob; with at
       least 3 pairs, outliers are dropped and the affine is refitted on their centroids.

    Parameters
    ----------
    cells_px
        (n, 2) cell centroids in global px.
    he_overview
        A coarse H&E level, (y, x, 3).
    he_um_per_overview_px
        µm per pixel of `he_overview`.
    px_um
        µm per CosMx global px.
    grid_um
        Mask grid, µm.
    min_blob_mm2
        Smallest blob used as a landmark, mm².

    Returns
    -------
    affine : Affine
        H&E µm → global px.
    pairs : DataFrame
        Landmark pairs (``x_px``, ``y_px``, ``he_x_um``, ``he_y_um``, ``resid_um``).
    score : float
        Fraction of the cells' tissue on H&E tissue.
    """
    from scipy.ndimage import binary_fill_holes, binary_opening
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.transform import resize
    from spatialdata.transformations import get_transformation_between_landmarks

    # 0. Masks
    grey = he_overview.mean(axis=2)
    he_px = binary_fill_holes(binary_opening(grey < threshold_otsu(grey), iterations=1))
    shape = np.round(np.array(he_px.shape) * he_um_per_overview_px / grid_um).astype(int)
    he_mask = resize(he_px.astype(np.float32), shape, anti_aliasing=True) > 0.5
    xy = cells_px * px_um

    # 1. Rigid search: coarse, then fine around the best right angle
    fit = lambda th, fl: (*_xcorr_fit(xy @ _rotation(th, fl).T, he_mask, grid_um), th, fl)
    score, t, theta, flip = max((fit(th, fl) for th in (0, 90, 180, 270) for fl in (1, -1)), key=lambda f: f[0])
    score, t, theta, flip = max((fit(theta + d, flip) for d in np.arange(-4, 4.01, 0.5)), key=lambda f: f[0])
    R = _rotation(theta, flip)
    to_he_um = np.r_[np.c_[R, t], [[0, 0, 1]]]

    # 2. Landmarks: blobs of cells (in the H&E frame) paired with the H&E blob under their centroid
    he_lab = label(he_mask)
    xy_he = xy @ R.T + t
    cell_mask, origin, idx = _points_mask(xy_he, grid_um)
    cell_lab = label(cell_mask)[idx[:, 1], idx[:, 0]]
    min_cells = min_blob_mm2 * 1e6 / grid_um**2
    pairs = []
    for k in np.unique(cell_lab):
        c_he = xy_he[cell_lab == k].mean(0)
        ix, iy = np.floor(c_he / grid_um).astype(int)
        if not (0 <= iy < he_lab.shape[0] and 0 <= ix < he_lab.shape[1]) or he_lab[iy, ix] == 0:
            continue
        blob = regionprops((he_lab == he_lab[iy, ix]).astype(int))[0]
        cell_area = np.unique(idx[cell_lab == k], axis=0).shape[0]
        if blob.area >= min_cells and cell_area >= 0.8 * blob.area:
            pairs.append([*cells_px[cell_lab == k].mean(0), *(np.array(blob.centroid[::-1]) + 0.5) * grid_um])
    pairs = pd.DataFrame(pairs, columns=["x_px", "y_px", "he_x_um", "he_y_um"]).drop_duplicates(
        ["he_x_um", "he_y_um"], keep=False)

    # 3. px -> µm -> H&E µm, as H&E µm -> global px
    to_px = lambda m: np.diag([1 / px_um, 1 / px_um, 1]) @ np.linalg.inv(m)
    affine = Affine(to_px(to_he_um), input_axes=("x", "y"), output_axes=("x", "y"))
    if len(pairs) >= 3:
        fit_lm = lambda p: get_transformation_between_landmarks(
            references_coords=ShapesModel.parse(p[["x_px", "y_px"]].to_numpy(), geometry=0, radius=1),
            moving_coords=ShapesModel.parse(p[["he_x_um", "he_y_um"]].to_numpy(), geometry=0, radius=1))

        def res(a, p):
            m = a.to_affine_matrix(("x", "y"), ("x", "y"))
            moved = (np.c_[p[["he_x_um", "he_y_um"]], np.ones(len(p))] @ m.T)[:, :2]
            return np.hypot(*(moved - p[["x_px", "y_px"]].to_numpy()).T) * px_um

        a = fit_lm(pairs)
        pairs["resid_um"] = res(a, pairs)
        pairs = pairs[pairs["resid_um"] <= max(3 * pairs["resid_um"].median(), 2 * grid_um)]
        if len(pairs) >= 3:
            affine = fit_lm(pairs)
            pairs["resid_um"] = res(affine, pairs)
    return affine, pairs, score


def add_he_image(sdata, svs, name="he", px_um=PX_UM, grid_um=50):
    """Add a whole-slide H&E (Aperio SVS) as a lazy multiscale image, registered onto the cells.

    The SVS's own pyramid levels become the scales, so nothing is read until plotted or written, except the
    coarsest level, used by :func:`register_he`. The pixel size comes from the SVS header (MPP). Landmark pairs are
    kept as shapes ``<name>_landmarks`` (H&E) and ``cell_landmarks``.

    Parameters
    ----------
    sdata
        A single slide, e.g. from :func:`frzspatial.build_sdata`. Modified in place and returned.
    svs
        Path of the SVS file.
    name
        Image element name.
    """
    from spatialdata.models.pyramids_utils import dask_arrays_to_datatree
    from spatialdata.transformations import Identity
    from spatialdata.transformations._utils import _set_transformations, compute_coordinates

    # 0. Lazy pyramid and pixel size
    tif = tifffile.TiffFile(svs)
    levels = [da.from_array(zarr.open(tif.series[0].aszarr(level=i), mode="r"), chunks=(2048, 2048, 3))
              for i in range(len(tif.series[0].levels))]  # from_zarr would look like a backing .zarr to spatialdata
    mpp = float(re.search(r"MPP\s*=\s*([\d.]+)", tif.pages[0].description).group(1))
    he = compute_coordinates(dask_arrays_to_datatree([lv.transpose(2, 0, 1) for lv in levels], dims=("c", "y", "x"),
                                                     channels=["r", "g", "b"]))

    # 1. Registration on the coarsest level, in H&E µm, then as H&E base px -> global px
    obs = sdata.tables["table"].obs
    cells_px = obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)
    overview = levels[-1].compute()
    affine, pairs, score = register_he(cells_px, overview, mpp * levels[0].shape[0] / overview.shape[0], px_um,
                                       grid_um)
    affine = Sequence([Scale([mpp, mpp], axes=("x", "y")), affine])
    _log(f"H&E registered: {score:.0%} of cell tissue on H&E tissue, {len(pairs)} landmarks"
         + (f", median residual {pairs['resid_um'].median():.0f} µm" if "resid_um" in pairs else ""))
    if len(pairs) < 3:
        warnings.warn("H&E: < 3 landmarks, rigid fit only. A small region (e.g. one TMA core) can match anywhere: "
                      "check the overlay, or register on the whole slide.")

    # 2. Image and landmarks
    _set_transformations(he, {"global": affine})
    Image2DModel().validate(he)
    sdata.images[name] = he
    if len(pairs):
        he_lm = ShapesModel.parse(pairs[["he_x_um", "he_y_um"]].to_numpy() / mpp, geometry=0, radius=250)
        _set_transformations(he_lm, {"global": affine})
        sdata.shapes[f"{name}_landmarks"] = he_lm
        sdata.shapes["cell_landmarks"] = ShapesModel.parse(pairs[["x_px", "y_px"]].to_numpy(), geometry=0,
                                                           radius=500, transformations={"global": Identity()})
    if "slide" in sdata.coordinate_systems:
        add_slide_view(sdata)
    return sdata
