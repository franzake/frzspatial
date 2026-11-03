"""frzspatial: CosMx (AtoMx) exports to SpatialData, with stitched TMA cores, plots and napari annotation.

Modules
-------
frzspatial.io        build_sdata, load_slides, save_table: AtoMx flat files to SpatialData and back to zarr
frzspatial.core      build_core, core_box: a core or slide stitched into one image and mask
frzspatial.annotate  annotations_to_obs and GeoJSON helpers (no napari needed)
frzspatial.pl        plot_slide_overview, plot_core, annotation_palette       (extra: plot)
frzspatial.viewer    open_napari, annotate_core                                 (extra: napari; loads Qt)
frzspatial.he        register_he, add_he_image: H&E registration            (extra: he)
frzspatial.qc        compute_sbr: regional signal-to-background ratio
frzspatial.export    write_seurat_inputs: plain files for a Seurat object

The functions of io, core, annotate, qc and export are also available at the top level, e.g. ``frzspatial.load_slides``.
``pl``, ``viewer`` and ``he`` are imported on demand, since they need optional dependencies.

Use ``help(frzspatial.load_slides)`` (or any function) for its documentation.
"""

from importlib.metadata import PackageNotFoundError, version

from . import _utils
from ._utils import PX_UM
from .annotate import annotation_file, annotations_to_obs, read_annotations, write_annotations
from .core import build_core, coordinate_system, core_box
from .export import write_seurat_inputs
from .qc import compute_sbr
from .io import add_slide_view, build_sdata, load_slides, save_table

try:
    __version__ = version("frzspatial")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0"


def set_verbose(verbose=True):
    """Turn frzspatial's progress messages on or off."""
    _utils.VERBOSE = bool(verbose)


__all__ = [
    "PX_UM",
    "__version__",
    "add_slide_view",
    "annotation_file",
    "annotations_to_obs",
    "build_core",
    "build_sdata",
    "compute_sbr",
    "coordinate_system",
    "core_box",
    "load_slides",
    "read_annotations",
    "save_table",
    "set_verbose",
    "write_annotations",
    "write_seurat_inputs",
]
