# Cluster markers by HieraType fold change for frzspatial.rtools.cluster_markers(), as main.Rmd §7.1
# (clusterMarkersSdata() of src/sdata_qc.R):
#   Rscript cluster_markers.R <dir> <n_top: number or all> <pseudocount>
# <dir> holds x_* (cells x genes counts) and clusters.csv (cell, cluster) and levels.txt; writes markers.csv, top.csv and
# mean_counts.csv.
here = dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(here, "io.R"))
args = commandArgs(trailingOnly = TRUE)
DIR = args[1]; N_TOP = if (args[2] == "all") NULL else as.integer(args[2]); PSEUDO = as.numeric(args[3])

counts = Matrix::t(read_counts(DIR, "x"))
cl = fread(file.path(DIR, "clusters.csv"), colClasses = "character")
cl = factor(cl$cluster[match(colnames(counts), cl$cell)], levels = readLines(file.path(DIR, "levels.txt")))
say(ncol(counts), " cells, ", nrow(counts), " genes, ", nlevels(cl), " clusters")

# 0. Fold change of each cluster against all other cells, ranked by prioritystat (penalises low expressers)
fc = HieraType::clusterwise_foldchange_metrics(counts, metadata = data.frame(cell_ID = colnames(counts), cluster = cl),
                                               cluster_column = "cluster", cellid_column = "cell_ID")
fc[, prioritystat := (cluster_expr + PSEUDO) / (clusterprime_expr + PSEUDO)]
fc[, cluster := factor(cluster, levels(cl))]
setorder(fc, cluster, -prioritystat)
top = if (is.null(N_TOP)) fc else fc[, head(.SD, N_TOP), by = cluster]

# 1. Mean counts per cell of the top genes in each cluster
genes = unique(top$gene)
avg = Matrix::sparseMatrix(i = seq_along(cl), j = as.integer(cl), x = 1 / tabulate(cl)[cl])
mean_counts = data.table(gene = genes, as.matrix(counts[genes, ] %*% avg))
setnames(mean_counts, c("gene", levels(cl)))

fwrite(fc, file.path(DIR, "markers.csv")); fwrite(top, file.path(DIR, "top.csv"))
fwrite(mean_counts, file.path(DIR, "mean_counts.csv"))
say("done")
