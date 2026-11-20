"""frzspatial: CosMx (AtoMx) exports to SpatialData, with stitched TMA cores, plots and napari annotation.

Modules
-------
frzspatial.io        build_sdata, load_slides, save_table, merge_slides, drop_cells, slide_with_cells: reading and merging
frzspatial.core      build_core, core_box: a core or slide stitched into one image and mask
frzspatial.annotate  annotations_to_obs and GeoJSON helpers (no napari needed)
frzspatial.pl        plot_slide_overview, plot_core, plot_annotations, annotation_palette
frzspatial.viewer    open_napari, annotate_core (loads Qt)
frzspatial.he        register_he, add_he_image: H&E registration
frzspatial.qc        compute_sbr, fov_integrity, qc_flags, plot_qc, plot_qc_flags: per-cell QC
frzspatial.rtools    insitutype, fastreseg_inputs, load_fastreseg, apply_reseg: R tools run from Python
frzspatial.tma       key_matrix, assign_cores, plot_core_grid, cases, add_clinical: TMA cores and case metadata
frzspatial.niches    neighbourhood_graph, compute_niches, niche_composition, name_niches: §8 niches
frzspatial.despotx   despotx_input, apply_despotx: DeSpotX decontamination input and result
frzspatial.export    write_seurat_inputs: plain files for a Seurat object

The functions of io, core, annotate, qc, tma and export are also available at the top level, e.g. ``frzspatial.load_slides``.
``pl``, ``viewer`` and ``he`` are imported on demand, since they are slow to import (``viewer`` loads Qt).

Use ``help(frzspatial.load_slides)`` (or any function) for its documentation.
"""

from importlib.metadata import PackageNotFoundError, version

from . import _utils
from ._utils import PX_UM
from .annotate import annotation_file, annotations_to_obs, read_annotations, write_annotations
from .core import build_core, coordinate_system, core_box
from .export import write_seurat_inputs
from .qc import compute_sbr, fov_integrity, panel_barcodes, plot_qc, plot_qc_flags, qc_flags
from .tma import add_clinical, assign_cores, cases, key_matrix, plot_core_grid
from . import niches, reference, rtools
from .despotx import apply_despotx, despotx_input
from .io import add_slide_view, build_sdata, drop_cells, load_slides, merge_slides, save_table, slide_with_cells

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
    "add_clinical",
    "add_slide_view",
    "apply_despotx",
    "assign_cores",
    "annotation_file",
    "annotations_to_obs",
    "build_core",
    "build_sdata",
    "cases",
    "compute_sbr",
    "coordinate_system",
    "core_box",
    "despotx_input",
    "fov_integrity",
    "key_matrix",
    "load_slides",
    "merge_slides",
    "drop_cells",
    "niches",
    "panel_barcodes",
    "plot_core_grid",
    "plot_qc",
    "plot_qc_flags",
    "qc_flags",
    "reference",
    "rtools",
    "read_annotations",
    "save_table",
    "set_verbose",
    "slide_with_cells",
    "write_annotations",
    "write_seurat_inputs",
]
