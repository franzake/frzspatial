# fastreseg_inputs.rds for the FastReseg cluster job (run_fastreseg_remote.R), for frzspatial.rtools.fastreseg_inputs():
#   Rscript fastreseg_inputs.R <dir> <out.rds>
# <dir> holds counts_* (cells x genes), clust.csv (cell, clust), left_out.txt, ctrl_genes.txt and fovs.csv (slide, fov).
here = dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(here, "io.R"))
args = commandArgs(trailingOnly = TRUE)
DIR = args[1]; OUT = args[2]

counts = read_counts(DIR, "counts")
clust = fread(file.path(DIR, "clust.csv"), colClasses = "character")
fovs = fread(file.path(DIR, "fovs.csv"), colClasses = list(character = "slide"))
slides = unique(fovs$slide)
saveRDS(list(
  counts     = counts,
  clust      = setNames(clust$clust, clust$cell),
  left_out   = readLines(file.path(DIR, "left_out.txt")),
  ctrl_genes = readLines(file.path(DIR, "ctrl_genes.txt")),
  fovs       = setNames(lapply(slides, function(s) as.integer(fovs$fov[fovs$slide == s])), slides)), OUT)
say("wrote ", OUT, ": ", nrow(counts), " cells, ", ncol(counts), " genes, ", nrow(fovs), " FOVs")
