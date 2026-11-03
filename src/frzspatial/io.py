"""Read CosMx AtoMx flat-file exports into SpatialData, in AtoMx global pixels.

An AtoMx export folder (``<inputs>``) is expected to look like::

    <inputs>/
      flatFiles/<id>_metadata_file.csv[.gz]
      flatFiles/<id>_exprMat_file.csv[.gz]
      flatFiles/<id>-polygons.csv[.gz]
      flatFiles/<id>_tx_file.csv[.gz]
      CellComposite/CellComposite_F00001.jpg ...
      CellLabels/CellLabels_F00001.tif ...
      Morphology2D/*_F00001.TIF ...              (optional)

where ``<id>`` is the dataset id (usually the slide name). :func:`build_sdata` turns it into one SpatialData
object with these elements, all in the ``"global"`` coordinate system:

=====================  ==============================================================================
``table``              counts (sparse int32) and cell metadata; cells named by ``cell`` (``c_<slide>_<fov>_<id>``)
``cell_boundaries``    one polygon per cell
``transcripts``        lazy points from a Parquet cache (x, y, z_raw, target, cell, CellComp, fov)
``fov<N>_image``       CellComposite of each FOV (lazy)
``fov<N>_labels``      CellLabels mask of each FOV (lazy)
``fov<N>_morphology``  Morphology2D of each FOV, one channel per marker (optional, lazy)
=====================  ==============================================================================

Each FOV is placed with ``x_global = x_local + dx`` and ``y_global = cy - y_local`` (image row = ``y_local``).

Speed: gzip is decompressed by ``pigz`` in a separate process when it is installed, CSVs are parsed by
multithreaded pyarrow, transcripts are streamed to Parquet once (reused while newer than the CSV) and read lazily,
and table, polygons and transcripts are read in parallel.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dask
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import shapely
import tifffile
from anndata import AnnData
from PIL import Image
from scipy.sparse import csr_matrix
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, Scale, Sequence, Translation, get_transformation, set_transformation
from tqdm.auto import tqdm

from ._utils import _log, _reset_clock

__all__ = [
    "build_sdata",
    "load_slides",
    "save_table",
    "add_slide_view",
    "read_table",
    "read_polygons",
    "transcripts_parquet",
    "read_transcripts",
    "read_images_labels",
    "flat",
]

_BLOCK = 64 << 20  # pyarrow CSV block size
_KEY = 1 << 20  # (fov, cell_ID) -> fov * _KEY + cell_ID; cell_ID < 2**20


def flat(folder, name):
    """Path of the flat file ``<name>.csv`` or ``<name>.csv.gz`` in `folder`, whichever exists.

    Raises
    ------
    FileNotFoundError
        If neither exists.
    """
    hits = [p for p in (Path(folder) / f"{name}.csv", Path(folder) / f"{name}.csv.gz") if p.exists()]
    if not hits:
        raise FileNotFoundError(f"{name}.csv[.gz] not found in {folder}")
    return hits[0]


class _Source:
    """Binary stream of a CSV, decompressed by pigz when available."""

    def __init__(self, path, threads=4):
        self.path, self.proc = Path(path), None
        if self.path.suffix != ".gz":
            self.fh = open(self.path, "rb")
        elif shutil.which("pigz"):
            self.proc = subprocess.Popen(["pigz", "-dc", f"-p{threads}", str(self.path)], stdout=subprocess.PIPE,
                                         bufsize=1 << 20)
            self.fh = self.proc.stdout
        else:
            self.fh = gzip.open(self.path, "rb")

    def __enter__(self):
        return self.fh

    def __exit__(self, *exc):
        self.fh.close()
        if self.proc is not None:
            self.proc.wait()


def _header(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return fh.readline().rstrip("\n").split(",")


def _read_csv(path, threads=4, **convert):
    with _Source(path, threads) as fh:
        return pv.read_csv(fh, read_options=pv.ReadOptions(block_size=_BLOCK),
                           convert_options=pv.ConvertOptions(**convert))


def _keys(fov, cell_id):
    return np.asarray(fov, np.int64) * _KEY + np.asarray(cell_id, np.int64)


def read_table(flat_dir, dataset_id):
    """Cell table: counts joined to the cell metadata on (fov, cell_ID).

    Parameters
    ----------
    flat_dir
        The export's ``flatFiles`` folder.
    dataset_id
        Prefix of the flat files (``<dataset_id>_metadata_file.csv`` ...).

    Returns
    -------
    AnnData
        Sparse int32 counts, the metadata as ``obs`` (cells named by ``cell``), centroids in ``obsm["spatial"]``,
        annotating the ``cell_boundaries`` shapes.
    """
    meta = _read_csv(flat(flat_dir, f"{dataset_id}_metadata_file")).to_pandas()
    meta = meta.drop(columns=["cell_id"], errors="ignore")  # duplicate of `cell`; case variant of cell_ID

    path = flat(flat_dir, f"{dataset_id}_exprMat_file")
    genes = [c for c in _header(path) if c not in ("fov", "cell_ID")]
    counts = _read_csv(path, column_types={g: pa.int32() for g in genes})
    rows = pd.Index(_keys(counts["fov"], counts["cell_ID"])).get_indexer(_keys(meta["fov"], meta["cell_ID"]))
    if (rows < 0).any():
        raise ValueError(f"{(rows < 0).sum()} metadata cells have no counts")
    X = np.column_stack([counts[g].to_numpy() for g in genes])[rows]
    del counts

    meta.index = meta["cell"].to_numpy()
    meta["region"] = pd.Categorical(["cell_boundaries"] * len(meta))
    table = AnnData(csr_matrix(X), obs=meta, var=pd.DataFrame(index=genes))
    table.obsm["spatial"] = meta[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)
    return TableModel.parse(table, region="cell_boundaries", region_key="region", instance_key="cell")


def read_polygons(flat_dir, dataset_id, cell_names):
    """Cell outlines, one polygon per cell with at least 3 vertices.

    Parameters
    ----------
    flat_dir, dataset_id
        As in :func:`read_table`.
    cell_names
        Series of ``cell`` names indexed by the (fov, cell_ID) key ``fov * 2**20 + cell_ID``.

    Returns
    -------
    GeoDataFrame
        Polygons in global px, indexed by ``cell``.
    """
    v = _read_csv(
        flat(flat_dir, f"{dataset_id}-polygons"),
        include_columns=["fov", "cellID", "x_global_px", "y_global_px"],
        column_types={"fov": pa.int32(), "cellID": pa.int32(), "x_global_px": pa.float64(),
                      "y_global_px": pa.float64()},
    )
    key = _keys(v["fov"], v["cellID"])
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])  # vertices come in one block per cell
    n = np.diff(np.r_[start, len(key)])
    if len(np.unique(key[start])) != len(start):
        raise ValueError("Polygon vertices are not grouped by cell.")
    codes = np.repeat(np.arange(len(start)), n)
    keep = (n >= 3)[codes]
    _, codes = np.unique(codes[keep], return_inverse=True)
    xy = np.c_[v["x_global_px"].to_numpy(), v["y_global_px"].to_numpy()][keep]
    names = cell_names.reindex(key[start][n >= 3]).to_numpy()
    polygons = shapely.polygons(shapely.linearrings(xy, indices=codes))
    return ShapesModel.parse(gpd.GeoDataFrame(geometry=polygons, index=pd.Index(names)))


_TX_TYPES = {
    "fov": pa.int16(),
    "cell": pa.dictionary(pa.int32(), pa.string()),
    "x_global_px": pa.float32(),
    "y_global_px": pa.float32(),
    "z": pa.int8(),
    "target": pa.dictionary(pa.int32(), pa.string()),
    "CellComp": pa.dictionary(pa.int32(), pa.string()),  # CSV reader supports int32 indices only
}


def transcripts_parquet(flat_dir, dataset_id, cache_dir, threads=4):
    """Stream the transcript CSV to ``<cache_dir>/<dataset_id>_tx.parquet`` once.

    The Parquet file is reused while it is newer than the CSV. Columns are renamed ``x``, ``y`` and ``z_raw``
    (a ``z`` column would be read by spatialdata as a third axis).

    Returns
    -------
    Path
        The Parquet file.
    """
    src = flat(flat_dir, f"{dataset_id}_tx_file")
    out = Path(cache_dir) / f"{dataset_id}_tx.parquet"
    if out.exists() and out.stat().st_mtime > src.stat().st_mtime:
        _log(f"transcripts: reusing cache {out}")
        return out
    _log(f"transcripts: streaming {src.name} to the Parquet cache (one time)")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    convert = pv.ConvertOptions(include_columns=list(_TX_TYPES), column_types=_TX_TYPES)
    with _Source(src, threads) as fh:
        reader = pv.open_csv(fh, read_options=pv.ReadOptions(block_size=_BLOCK), convert_options=convert)
        names = {"x_global_px": "x", "y_global_px": "y", "z": "z_raw"}
        schema = pa.schema([reader.schema.field(c).with_name(names.get(c, c)) for c in _TX_TYPES])
        rows, step = 0, 25_000_000
        with pq.ParquetWriter(tmp, schema, compression="lz4") as writer:
            for batch in reader:
                writer.write_batch(pa.RecordBatch.from_arrays([batch.column(c) for c in _TX_TYPES], schema=schema))
                if (rows + batch.num_rows) // step > rows // step:
                    _log(f"transcripts: {(rows + batch.num_rows) / 1e6:.0f}M rows")
                rows += batch.num_rows
    tmp.replace(out)
    _log(f"transcripts: {rows:,} rows cached")
    return out


_CATEGORICAL = ("target", "cell", "CellComp")


def read_transcripts(parquet):
    """Lazy transcript points (dask) from a Parquet cache written by :func:`transcripts_parquet`."""
    ddf = dd.read_parquet(parquet, engine="pyarrow")
    # known categories: required to write the points, and avoids dask recomputing them
    for column, categories in _parquet_categories(parquet, _CATEGORICAL).items():
        ddf[column] = ddf[column].cat.set_categories(categories)
    # sort=True: for dask input spatialdata only uses it to skip a full scan that checks index order
    return PointsModel.parse(ddf, coordinates={"x": "x", "y": "y"}, feature_key="target", instance_key="cell",
                             sort=True)


def _parquet_categories(parquet, columns):
    """Union of the per-chunk dictionaries of dictionary columns, without decoding them."""
    f = pq.ParquetFile(parquet)
    values = {c: set() for c in columns}
    for i in range(f.num_row_groups):
        group = f.read_row_group(i, columns=list(columns))
        for c in columns:
            for chunk in group.column(c).chunks:
                values[c].update(chunk.dictionary.to_pylist())
    return {c: sorted(v) for c, v in values.items()}


def _read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _lazy_image(path):
    """(y, x, c) uint8 dask array from a JPEG/PNG; only the header is read now."""
    with Image.open(path) as im:
        w, h = im.size
    return da.from_delayed(dask.delayed(_read_rgb)(str(path)), shape=(h, w, 3), dtype=np.uint8)


def _lazy_tiff(path):
    """(y, x) dask array from a single-page TIFF; only the header is read now."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        shape, dtype = page.shape, page.dtype
    return da.from_delayed(dask.delayed(tifffile.imread)(str(path)), shape=shape, dtype=dtype)


def _morphology_channels(path):
    """Marker names of a Morphology2D TIFF in page order (ChannelOrder), e.g. PanCK, G, Membrane, CD45, DNA."""
    with tifffile.TiffFile(path) as tif:
        meta = json.loads(tif.pages[0].description)
    targets = {r["Fluorophore"]["ChannelId"]: r["BiologicalTarget"]
               for r in meta["MorphologyKit"]["MorphologyReagents"]}
    return [targets.get(uid, uid) for uid in meta["ChannelOrder"]]  # a channel without a reagent keeps its id


def _lazy_stack(path):
    """(c, y, x) dask array from a multi-page TIFF; only the header is read now."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape, dtype = series.shape, series.dtype
    return da.from_delayed(dask.delayed(tifffile.imread)(str(path)), shape=shape, dtype=dtype)


def read_images_labels(meta, image_dir, morphology=False):
    """Lazy per-FOV images, placed in global px.

    Parameters
    ----------
    meta
        Cell metadata with ``fov``, ``CenterX/Y_global_px`` and ``CenterX/Y_local_px``: each FOV's offset is the
        median over its cells.
    image_dir
        The export folder holding ``CellComposite/``, ``CellLabels/`` and ``Morphology2D/``.
    morphology
        Also read the Morphology2D stacks (one channel per marker).

    Returns
    -------
    images, labels : dict
        ``{"fov<N>_image": ..., "fov<N>_morphology": ...}`` and ``{"fov<N>_labels": ...}``.
    """
    images, labels = {}, {}
    morph = {int(re.search(r"_F(\d+)\.TIF$", p.name).group(1)): p
             for p in (Path(image_dir) / "Morphology2D").glob("*_F*.TIF")} if morphology else {}
    if morphology and not morph:
        raise FileNotFoundError(f"morphology=True but no Morphology2D/*_F*.TIF in {image_dir}")
    channels = _morphology_channels(next(iter(morph.values()))) if morph else None
    off = pd.DataFrame({
        "dx": meta.CenterX_global_px - meta.CenterX_local_px,
        "cy": meta.CenterY_global_px + meta.CenterY_local_px,
        "fov": meta.fov,
    }).groupby("fov").median()
    for fov, o in off.iterrows():
        to_global = {"global": Affine(np.array([[1, 0, o.dx], [0, -1, o.cy], [0, 0, 1]]),
                                      input_axes=("x", "y"), output_axes=("x", "y"))}
        im = Path(image_dir) / "CellComposite" / f"CellComposite_F{fov:05d}.jpg"
        la = Path(image_dir) / "CellLabels" / f"CellLabels_F{fov:05d}.tif"
        if im.exists():
            images[f"fov{fov}_image"] = Image2DModel.parse(_lazy_image(im), dims=("y", "x", "c"),
                                                           transformations=to_global)
        if la.exists():
            labels[f"fov{fov}_labels"] = Labels2DModel.parse(_lazy_tiff(la), dims=("y", "x"),
                                                             transformations=to_global)
        if fov in morph:  # same pixel grid as CellLabels
            images[f"fov{fov}_morphology"] = Image2DModel.parse(
                _lazy_stack(morph[fov]), dims=("c", "y", "x"), c_coords=channels, transformations=to_global)
    return images, labels


def add_slide_view(sdata, name="slide"):
    """Add a coordinate system `name`: ``"global"`` mirrored in y (``y = y_max - y_global``).

    spatialdata-plot draws y downwards (the image convention); in this system plots show y upwards, as in the
    AtoMx slide view. ``"global"`` itself is left unchanged. Modifies `sdata` in place and returns it.
    """
    y_max = float(np.ceil(sdata.tables["table"].obs["CenterY_global_px"].max() / 1000) * 1000 + 5000)
    mirror = Sequence([Scale([1, -1], axes=("x", "y")), Translation([0, y_max], axes=("x", "y"))])
    for _, _, element in sdata.gen_spatial_elements():
        set_transformation(element, Sequence([get_transformation(element, "global"), mirror]),
                           to_coordinate_system=name)
    return sdata


def build_sdata(inputs, cache_dir, dataset_id, *, images=True, morphology=False, transcripts=True, threads=8,
                flip_y=True):
    """Build one SpatialData object from an AtoMx export folder.

    Parameters
    ----------
    inputs
        Export folder holding ``flatFiles/``, ``CellComposite/`` and ``CellLabels/`` (see the module docs).
    cache_dir
        Folder for the transcript Parquet cache (``<dataset_id>_tx.parquet``).
    dataset_id
        Prefix of the flat files, usually the slide name.
    images
        Read the per-FOV composites and masks (lazily).
    morphology
        Also read ``Morphology2D/`` as ``fov<N>_morphology`` (one channel per marker).
    transcripts
        Read the transcripts (via the Parquet cache).
    threads
        Threads for pigz and the CSV parser.
    flip_y
        Add the ``"slide"`` coordinate system, y upwards (:func:`add_slide_view`).

    Returns
    -------
    SpatialData
        Lazy where possible: write it with ``sdata.write(path)`` and reopen it with :func:`load_slides` or
        ``spatialdata.read_zarr``.

    Examples
    --------
    >>> sdata = build_sdata("data/MN2_inputs", cache_dir="cache", dataset_id="MN2")  # doctest: +SKIP
    >>> sdata.write("MN2_flat.zarr")  # doctest: +SKIP
    """
    _reset_clock()
    inputs = Path(inputs).expanduser()
    flat_dir, image_dir = inputs / "flatFiles", (inputs if images else None)
    _log(f"reading {inputs}")
    with ThreadPoolExecutor(3) as pool:
        tx = pool.submit(transcripts_parquet, flat_dir, dataset_id, cache_dir, threads) if transcripts else None
        table = pool.submit(read_table, flat_dir, dataset_id).result()
        _log(f"table: {table.n_obs:,} cells x {table.n_vars:,} genes")
        obs = table.obs
        cell_names = pd.Series(obs["cell"].to_numpy(), index=_keys(obs["fov"], obs["cell_ID"]))
        shapes = pool.submit(read_polygons, flat_dir, dataset_id, cell_names)
        imgs, labels = read_images_labels(obs, image_dir, morphology) if image_dir is not None else ({}, {})
        n_morph = sum(name.endswith("_morphology") for name in imgs)
        _log(f"images and masks: {len(imgs) - n_morph} composites + {n_morph} morphology + {len(labels)} masks (lazy)")
        shapes = shapes.result()
        _log(f"polygons: {len(shapes):,} cells")
        points = {"transcripts": read_transcripts(tx.result())} if transcripts else {}
    sdata = SpatialData(images=imgs, labels=labels, points=points,
                        shapes={"cell_boundaries": shapes}, tables={"table": table})
    if flip_y:
        add_slide_view(sdata)
    _log("done")
    return sdata


def load_slides(slides, input_dir, zarr_dir, cache_dir, *, rebuild=False, morphology=False):
    """Open several slides, building the missing ones.

    Each slide is reopened from ``<zarr_dir>/<slide>_flat.zarr``. A slide without a zarr is first built from
    ``<input_dir>/<slide>_inputs`` with :func:`build_sdata` and saved there.

    Parameters
    ----------
    slides
        Slide names, also used as the flat files' dataset ids.
    input_dir
        Folder holding the ``<slide>_inputs`` export folders.
    zarr_dir
        Folder of the ``<slide>_flat.zarr`` files.
    cache_dir
        Transcript Parquet cache, as in :func:`build_sdata`.
    rebuild
        True rebuilds every slide; a list rebuilds only those.
    morphology
        Passed to :func:`build_sdata`.

    Returns
    -------
    dict
        ``{slide: SpatialData}``. Images and masks are read lazily from the zarr.

    Notes
    -----
    ome_zarr's "no parent found" warnings (one per FOV mask, harmless) are silenced.
    """
    logging.getLogger("ome_zarr.reader").setLevel(logging.ERROR)
    rebuild = set(slides) if rebuild is True else set(rebuild or ())
    Path(zarr_dir).mkdir(parents=True, exist_ok=True)
    out = {}
    for s in (bar := tqdm(list(slides), desc="slides", unit="slide")):
        zarr_path = Path(zarr_dir) / f"{s}_flat.zarr"
        if s in rebuild or not zarr_path.exists():
            if not (Path(input_dir) / f"{s}_inputs" / "flatFiles").is_dir():
                raise FileNotFoundError(f"{s}: no {zarr_path} and no {input_dir}/{s}_inputs/flatFiles to build it from")
            bar.set_postfix_str(f"{s} (building)")
            build_sdata(Path(input_dir) / f"{s}_inputs", cache_dir=cache_dir, dataset_id=s,
                        morphology=morphology).write(zarr_path, overwrite=True)
        bar.set_postfix_str(s)
        out[s] = read_zarr(zarr_path)
    return out


def save_table(sdata, table="table"):
    """Write `sdata`'s table back into the zarr it was read from (e.g. after adding obs columns).

    spatialdata does not overwrite an element of its own backing zarr, so the table is deleted on disk and written
    again; images, masks, polygons and transcripts on disk are left unchanged. Takes seconds per slide. Do not
    interrupt it: between the two steps the zarr has no table.
    """
    sdata.delete_element_from_disk(table)
    sdata.write_element(table)

