"""frzspatial: CosMx (AtoMx) exports to SpatialData, with TMA cores, QC, R tools and plots.

Modules
-------
frzspatial.io         build_sdata, load_slides, merge_slides, drop_cells: reading and merging
frzspatial.core       core_box, coordinate_system: a core's bounding box, a slide's elements
frzspatial.qc         compute_sbr, fov_integrity, qc_flags, plot_qc, plot_qc_flags: per-cell QC
frzspatial.tma        key_matrix, assign_cores, plot_core_grid, cases, add_clinical: TMA cores and case metadata
frzspatial.reference  reference_profiles: InSituType reference profiles from an atlas
frzspatial.rtools     insitutype, fastreseg_inputs, load_fastreseg, apply_reseg, pearson_pca, cluster_markers,
                      marker_heatmap, structure_zones: R tools run from Python
frzspatial.despotx    despotx_input, apply_despotx: DeSpotX decontamination input and result
frzspatial.he         align_he_cores: post-CosMx H&E, one aligned crop per core
frzspatial.de         smide: spatial differential expression (count mixed models + neighbour covariate)
frzspatial.enrich     gene_sets, smide_gsea: preranked GSEA of DE results per family
frzspatial.pl         plot_core, plot_umap, plot_composition, plot_smide_volcano

The functions of io, core, qc, tma, despotx and ``de.smide`` are also available at the top level, e.g. ``frzspatial.load_slides``.
``pl`` is imported on demand, since it is slow to import.

Use ``help(frzspatial.load_slides)`` (or any function) for its documentation.
"""

from importlib.metadata import PackageNotFoundError, version

from ._utils import PX_UM
from .core import coordinate_system, core_box
from .qc import compute_sbr, fov_integrity, panel_barcodes, plot_qc, plot_qc_flags, qc_flags
from .tma import add_clinical, assign_cores, cases, key_matrix, plot_core_grid
from . import de, enrich, he, reference, rtools
from .de import smide
from .despotx import apply_despotx, despotx_input
from .io import add_slide_view, build_sdata, drop_cells, load_slides, merge_slides

try:
    __version__ = version("frzspatial")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0"


__all__ = [
    "PX_UM",
    "__version__",
    "add_clinical",
    "add_slide_view",
    "apply_despotx",
    "assign_cores",
    "build_sdata",
    "cases",
    "compute_sbr",
    "coordinate_system",
    "core_box",
    "de",
    "enrich",
    "despotx_input",
    "fov_integrity",
    "key_matrix",
    "load_slides",
    "merge_slides",
    "drop_cells",
    "he",
    "panel_barcodes",
    "plot_core_grid",
    "plot_qc",
    "plot_qc_flags",
    "qc_flags",
    "reference",
    "rtools",
    "smide",
]
