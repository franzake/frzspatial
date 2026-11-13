# Ambient RNA removal by decontX (R package celda) for frzspatial.rtools.decontx():
#   Rscript decontx.R <dir> <max_iter> <delta1> <delta2> <estimate_delta 0/1> <seed>
# <dir> holds x_* (cells x genes counts) and cells.csv (cell, cluster, batch); writes decont_* (cells x genes,
# decontaminated counts) and contamination.csv (cell, contamination).
here = dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(here, "io.R"))
suppressPackageStartupMessages(library(celda))
args = commandArgs(trailingOnly = TRUE)
DIR = args[1]; MAX_ITER = as.integer(args[2]); DELTA = as.numeric(args[3:4]); EST_DELTA = args[5] == "1"
SEED = as.integer(args[6])

counts = Matrix::t(read_counts(DIR, "x"))
cells = fread(file.path(DIR, "cells.csv"), colClasses = "character")
cells = cells[match(colnames(counts), cells$cell)]
say(ncol(counts), " cells, ", nrow(counts), " genes, ", uniqueN(cells$cluster), " clusters, ", uniqueN(cells$batch), " batches")

# Each cell's counts as a mixture of its cluster's profile and the contamination from the other clusters, per batch
res = decontX(counts, z = cells$cluster, batch = cells$batch, maxIter = MAX_ITER, delta = DELTA,
              estimateDelta = EST_DELTA, seed = SEED, verbose = TRUE)
write_counts(Matrix::t(res$decontXcounts), DIR, "decont")
fwrite(data.table(cell = colnames(counts), contamination = res$contamination), file.path(DIR, "contamination.csv"))
say("done")
