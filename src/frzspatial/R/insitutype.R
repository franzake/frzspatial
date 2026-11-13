# Supervised InSituType cell typing for frzspatial.rtools.insitutype(), as main.Rmd §4.1:
#   Rscript insitutype.R <dir> <ref_profiles.rds> <anchor_cells> <ncores> <seed> [<save.rds>]
# <dir> holds x_* (cells x genes counts) and cells.csv (cell, neg, immunofluorescence columns); writes <dir>/ist.csv.
here = dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(here, "io.R"))
suppressPackageStartupMessages(library(InSituType))
args = commandArgs(trailingOnly = TRUE)
DIR = args[1]; REF = args[2]; ANCHOR_CELLS = as.integer(args[3]); NCORES = as.integer(args[4])
SEED = as.integer(args[5]); SAVE = if (length(args) > 5 && nzchar(args[6])) args[6] else NULL

# 0. Query: counts, background and immunofluorescence; reference profiles; genes shared with them
x = read_counts(DIR, "x")
cells = as.data.frame(fread(file.path(DIR, "cells.csv"), colClasses = list(character = "cell")))
neg = setNames(cells$neg, cells$cell)
ref_profiles = if (grepl("[.]csv$", REF)) as.matrix(read.csv(REF, row.names = 1, check.names = FALSE)) else readRDS(REF)
genes = intersect(colnames(x), rownames(ref_profiles))
say(nrow(x), " cells, ", length(genes), " genes shared with ", ncol(ref_profiles), " reference profiles")

# 1. Cohorts from immunofluorescence (protein is not affected by RNA spillover)
set.seed(SEED)
cohort = fastCohorting(as.matrix(cells[, setdiff(names(cells), c("cell", "neg"))]), gaussian_transform = TRUE)
names(cohort) = cells$cell
say(length(unique(cohort)), " cohorts")

# 2. Reference profiles rescaled to this panel, with anchors from a random subset of cells
set.seed(SEED)
sub = sample(nrow(x), min(ANCHOR_CELLS, nrow(x)))
upd = updateReferenceProfiles(reference_profiles = ref_profiles[genes, , drop = FALSE], reference_sds = NULL,
                              counts = x[sub, genes, drop = FALSE], neg = neg[sub], assay_type = "rna")$updated_profiles

# 2b. Cell types with too few anchors (dropped by updateReferenceProfiles) keep their atlas profile, rescaled per gene
#     by the median updated/atlas ratio of the refitted types
dropped = setdiff(colnames(ref_profiles), colnames(upd)); kept = colnames(upd); g = rownames(upd)
ratio = upd[g, kept] / ref_profiles[g, kept]
ratio[!is.finite(ratio)] = NA
scale = apply(ratio, 1, median, na.rm = TRUE)
scale[!is.finite(scale)] = median(scale, na.rm = TRUE)
upd = cbind(upd[g, kept], ref_profiles[g, dropped, drop = FALSE] * scale)
say("refitted: ", length(kept), " cell types; atlas profiles rescaled: ", paste(dropped, collapse = ", "))

# 3. insitutypeML's steps, its likelihoods (lldist) split across cores; background and cohort frequencies use all
#    cells, so labels and probabilities equal one insitutypeML() call
xg = x[, g, drop = FALSE]
bg = estimateBackground(counts = xg, neg = neg)
xg = InSituType:::alignGenes(counts = xg, profiles = upd)
prof = upd[colnames(xg), , drop = FALSE]
chunks = split(seq_len(nrow(xg)), cut(seq_len(nrow(xg)), NCORES, labels = FALSE))
ll = parallel::mclapply(chunks, function(i) InSituType:::lldist(x = prof, xsd = NULL, mat = xg[i, , drop = FALSE],
                                                               bg = bg[i], size = 10, assay_type = "rna"),
                        mc.cores = NCORES)
bad = vapply(ll, inherits, logical(1), "try-error")
if (any(bad)) stop("lldist failed in a worker: ", ll[bad][[1]])
logliks = InSituType:::update_logliks_with_cohort_freqs(do.call(rbind, ll), cohort = cohort, minfreq = 1e-04,
                                                        nbaselinecells = 100)
logliks = logliks[, colnames(logliks) != "undefined", drop = FALSE]
logliks = cbind(logliks, undefined = ifelse(Matrix::rowSums(xg) == 0, 0, -Inf))  # cells with no counts
ist = list(clust = setNames(colnames(logliks)[apply(logliks, 1, which.max)], rownames(logliks)),
           prob = setNames(apply(InSituType:::logliks2probs(logliks), 1, max), rownames(logliks)))

# 4. Results
fwrite(data.table(cell = rownames(x), cohort = cohort[rownames(x)], celltype = ist$clust[rownames(x)],
                  prob = ist$prob[rownames(x)]), file.path(DIR, "ist.csv"))
if (!is.null(SAVE)) saveRDS(list(ref_profiles_upd = upd, ist = ist, cohort = cohort), SAVE)
say("done")
