# The FastReseg task results (reseg_task_<t>.rds) merged, for frzspatial.rtools.load_fastreseg(), as main.Rmd §4.2:
#   Rscript load_fastreseg.R <fastreseg_dir> <exchange_dir> <save: TRUE | FALSE>
# Writes <exchange_dir>/cells.csv (updated_perCellDT) and exprs_* (cells x probes); with save, reseg_sdata.rds.
here = dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(here, "io.R"))
args = commandArgs(trailingOnly = TRUE)
FR_OUT = args[1]; DIR = args[2]; SAVE = as.logical(args[3])

# 0. One result per array task
n_tasks = as.integer(readLines(file.path(FR_OUT, "fastreseg_ntasks.txt")))
files = sort(list.files(FR_OUT, "^reseg_task_[0-9]+[.]rds$", full.names = TRUE))
if (length(files) != n_tasks) stop(length(files), " task results in ", FR_OUT, ", expected ", n_tasks)
parts = lapply(files, readRDS)

# 1. Merged: every task also resegmented the first FOV, so its duplicates are dropped
cells = rbindlist(lapply(parts, `[[`, "updated_perCellDT"), fill = TRUE)
cells = cells[!duplicated(cells$updated_cellID)]
genes = rownames(parts[[1]]$updated_perCellExprs)
exprs = do.call(cbind, lapply(parts, function(p) p$updated_perCellExprs[genes, , drop = FALSE]))[, cells$updated_cellID]
print(table(cells$reSeg_action, useNA = "ifany"))
if (SAVE) saveRDS(list(updated_perCellDT = cells, updated_perCellExprs = exprs), file.path(FR_OUT, "reseg_sdata.rds"))

# 2. For Python
fwrite(cells, file.path(DIR, "cells.csv"))
write_counts(Matrix::t(exprs), DIR, "exprs")
say(nrow(cells), " cells, ", length(genes), " probes")
