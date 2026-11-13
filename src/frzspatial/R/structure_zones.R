# Structures and the rings around them (SpNeigh) for frzspatial.rtools.structure_zones():
#   Rscript structure_zones.R <dir> <eps> <min_pts> <min_cells> <pad> <outlier> <ring1> <ring2>
# <dir> holds cells.csv (cell, slide, x, y in µm, member 0/1); writes zones.csv (cell, region_id, zone: inside, ring1,
# ring2) and outlines.csv (region_id, x, y). Per slide, as each slide has its own coordinates.
suppressPackageStartupMessages({library(SpNeigh); library(sf); library(data.table)})
say = function(...) message(format(Sys.time(), "[%H:%M:%S] "), ...)
args = commandArgs(trailingOnly = TRUE)
DIR = args[1]; EPS = as.numeric(args[2]); MIN_PTS = as.integer(args[3]); MIN_CELLS = as.integer(args[4])
PAD = as.numeric(args[5]); OUTLIER = as.numeric(args[6]); RING1 = as.numeric(args[7]); RING2 = as.numeric(args[8])

cells = as.data.frame(fread(file.path(DIR, "cells.csv"), colClasses = c(cell = "character", slide = "character")))
cells$member = ifelse(cells$member == 1, "member", "other")

res = lapply(split(cells, cells$slide), function(d) {
  if (sum(d$member == "member") < MIN_CELLS) return(NULL)
  # 1. Member cells less SpNeigh's outliers (mean distance to their 5 nearest members > OUTLIER), grouped by DBSCAN;
  #    each group outlined by its convex hull, padded so the cells on the outline count as inside (SpNeigh's concave
  #    outline cuts into compact structures such as glomerular tufts)
  b = getBoundary(d, cluster_col = "member", one_cluster = "member", eps = EPS, minPts = MIN_PTS, distance_cutoff = OUTLIER)
  b$region_id = paste0(d$slide[1], "_r", b$region_id)
  inner = st_buffer(st_convex_hull(buildBoundaryPoly(b)), PAD)
  n = table(getCellsInside(d[d$member == "member", ], "member", inner)$region_id)
  inner = inner[inner$region_id %in% names(n)[n >= MIN_CELLS], ]
  if (nrow(inner) == 0) return(NULL)
  # 2. Cells inside, in the ring 0-RING1 outside and in the ring RING1-RING2 (first match wins)
  #    (ring widths from the convex hull, so less the padding)
  zones = list(inside = inner, ring1 = getRingRegion(inner, dist = RING1 - PAD),
               ring2 = getRingRegion(getOuterBoundary(inner, dist = RING1 - PAD), dist = RING2 - RING1))
  z = rbindlist(lapply(names(zones), function(k)
    data.table(st_drop_geometry(getCellsInside(d, "member", zones[[k]]))[, c("cell", "region_id")], zone = k)))
  xy = st_coordinates(inner)
  say(d$slide[1], ": ", nrow(inner), " structures")
  list(zones = z[!duplicated(z$cell)], outlines = data.table(region_id = inner$region_id[xy[, "L2"]], x = xy[, "X"], y = xy[, "Y"]))
})
res = Filter(Negate(is.null), res)
fwrite(rbindlist(lapply(res, `[[`, "zones")), file.path(DIR, "zones.csv"))
fwrite(rbindlist(lapply(res, `[[`, "outlines")), file.path(DIR, "outlines.csv"))
