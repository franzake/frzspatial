# One core's cells and structure outlines as SpNeigh's plotBoundary draws them, for frzspatial.rtools.plot_boundary():
#   Rscript plot_boundary.R <dir> <out.png> <point_size> <width> <height> <title> <ring1> <ring2>
# <dir> holds cells.csv (cell, x, y in µm, cluster), colors.csv (cluster, color; empty: SpNeigh's) and outlines.csv
# (region_id, x, y). Each outline in black, its rings dashed; y pointing down as in the image.
suppressPackageStartupMessages({library(SpNeigh); library(sf); library(ggplot2); library(data.table)})
args = commandArgs(trailingOnly = TRUE)
DIR = args[1]; OUT = args[2]; PT = as.numeric(args[3]); W = as.numeric(args[4]); H = as.numeric(args[5])
TITLE = args[6]; RINGS = as.numeric(args[7:length(args)])

cells = as.data.frame(fread(file.path(DIR, "cells.csv"), colClasses = c(cell = "character", cluster = "character")))
cols = fread(file.path(DIR, "colors.csv"), colClasses = "character")
cells$cluster = factor(cells$cluster, levels = cols$cluster)
colors = if (all(nzchar(cols$color))) cols$color else colors15_cheng
lines = as.data.frame(fread(file.path(DIR, "outlines.csv"), colClasses = c(region_id = "character")))
cells$y = -cells$y; lines$y = -lines$y   # image orientation (y down); axis labels un-negated below

p = plotBoundary(cells, cluster_col = "cluster", boundary = if (nrow(lines)) lines else NULL, colors = colors,
                 point_size = PT, linewidth_boundary = 0.9)
if (nrow(lines)) {
  polys = buildBoundaryPoly(lines)
  for (k in seq_along(RINGS))
    p = p + geom_sf(data = getOuterBoundary(polys, dist = RINGS[k]), inherit.aes = FALSE, fill = NA,
                    color = c("#E41A1C", "#FFB000", "grey40")[k], linewidth = 0.6, linetype = "dashed")
}
p = p + scale_y_continuous(labels = function(b) -b) + coord_sf(expand = FALSE) + labs(title = TITLE, color = NULL, x = "x (µm)", y = "y (µm)") +
  theme(legend.position = "bottom", legend.text = element_text(size = 11), plot.title = element_text(size = 14))
ggsave(OUT, p, width = W, height = H, dpi = 110, bg = "white")
