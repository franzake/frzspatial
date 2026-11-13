# Batch-corrected Pearson-residual PCA (scPearsonPCA) for frzspatial.rtools.pearson_pca(), as main.Rmd §5.1
# (runScPearsonPCASdata() of src/sdata_qc.R):
#   Rscript pearson_pca.R <dir> <nfeatures> <npcs> <ncores>
# <dir> holds x_* (cells x genes counts), batch.csv (cell, batch) and optionally exclude_genes.txt; writes pca.bin
# (cells x npcs, float64, column-major) and hvgs.txt.
here = dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(here, "io.R"))
args = commandArgs(trailingOnly = TRUE)
DIR = args[1]; NFEATURES = as.integer(args[2]); NPCS = as.integer(args[3]); NCORES = as.integer(args[4])

# 0. Counts (genes x cells) and batches
x = Matrix::t(read_counts(DIR, "x"))
b = fread(file.path(DIR, "batch.csv"), colClasses = "character")
batch = data.table(cell = colnames(x), batch = b$batch[match(colnames(x), b$cell)])
exclude = if (file.exists(file.path(DIR, "exclude_genes.txt"))) readLines(file.path(DIR, "exclude_genes.txt")) else NULL
say(ncol(x), " cells, ", nrow(x), " genes, ", length(unique(batch$batch)), " batches")

# 1. Variable genes (Seurat vst, as FindVariableFeatures() on the object)
hvf = Seurat::FindVariableFeatures(x[setdiff(rownames(x), exclude), ], verbose = FALSE)
hvgs = head(rownames(hvf)[order(-hvf$vst.variance.standardized)], NFEATURES)
say(length(hvgs), " variable genes (vst)")

# 2. Per-batch gene frequencies and the batch-corrected PCA
grate = scPearsonPCA::gene_frequency(x, obs = batch, batch_variable = "batch", cellid_colname = "cell")
say("batch-corrected Pearson-residual PCA (", NPCS, " PCs, ", NCORES, " cores)")
pca = scPearsonPCA:::sparse_quasipoisson_pca_seurat_batch(
  x[hvgs, ], obs = batch, batch_variable = "batch", cellid_colname = "cell", totalcounts = Matrix::colSums(x),
  grate = grate[hvgs, ], npcs = NPCS, ncores = NCORES, verbose = TRUE, return_seurat_reduction = FALSE)

# 3. For Python
emb = unname(pca$cell.embeddings[colnames(x), , drop = FALSE])
writeBin(as.double(emb), file.path(DIR, "pca.bin"), size = 8)
writeLines(as.character(dim(emb)), file.path(DIR, "pca_dim.txt"))
writeLines(hvgs, file.path(DIR, "hvgs.txt"))
say("done")
