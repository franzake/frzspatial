# HieraType marker heatmap for frzspatial.rtools.marker_heatmap(), as main.Rmd §7.3:
#   Rscript marker_heatmap.R <dir> <out.png> <width_in> <height_in>
# <dir> holds markers.csv (HieraType fold changes), levels.txt (cluster order) and blocks.csv (block, gene).
here = dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(here, "io.R"))
suppressPackageStartupMessages(library(ggplot2))
args = commandArgs(trailingOnly = TRUE)
DIR = args[1]; OUT = args[2]; W = as.numeric(args[3]); H = as.numeric(args[4])

fc = fread(file.path(DIR, "markers.csv"))
fc[, cluster := factor(cluster, readLines(file.path(DIR, "levels.txt")))]
blocks = fread(file.path(DIR, "blocks.csv"))
blocks[, block := factor(block, unique(block))]
genes = blocks$gene
n = as.vector(table(blocks$block)); ends = cumsum(n)

# Fold changes of each cluster against all other cells; dashed lines between blocks, block names on top
p = HieraType::marker_heatmap(fc, featsuse = genes, geneorder = genes, clusterorder = levels(fc$cluster),
                              orient_diagonal = FALSE) +
  geom_vline(xintercept = head(ends, -1) + 0.5, linetype = 2) +
  annotate("text", x = ends - n / 2 + 0.5, y = nlevels(fc$cluster) + 0.7, label = levels(blocks$block), angle = 90,
           hjust = 0, size = 5, fontface = "bold") +
  coord_cartesian(clip = "off") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 14),
        axis.text.y = element_text(size = 14), legend.text = element_text(size = 12),
        legend.title = element_text(size = 14), plot.margin = margin(110, 5, 5, 5))
ggsave(OUT, p, width = W, height = H, dpi = 100, bg = "white")
say("wrote ", OUT)
