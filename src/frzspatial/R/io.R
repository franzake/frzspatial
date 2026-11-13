# File exchange with frzspatial.rtools: sparse matrices as the CSC components of a dgCMatrix, with row / column names.
suppressPackageStartupMessages({library(Matrix); library(data.table)})
say = function(...) message(format(Sys.time(), "[%H:%M:%S] "), ...)

read_counts = function(dir, name) {
  f = function(s) file.path(dir, paste0(name, s))
  rd = function(s, what, size) readBin(f(s), what, n = file.size(f(s)) / size, size = size)
  rows = readLines(f("_rows.txt")); cols = readLines(f("_cols.txt"))
  new("dgCMatrix", i = rd("_i.bin", "integer", 4), p = rd("_p.bin", "integer", 4), x = rd("_x.bin", "double", 8),
      Dim = c(length(rows), length(cols)), Dimnames = list(rows, cols))
}

write_counts = function(m, dir, name) {
  m = as(m, "CsparseMatrix"); m = as(m, "dMatrix")
  f = function(s) file.path(dir, paste0(name, s))
  writeBin(as.integer(m@i), f("_i.bin"), size = 4); writeBin(as.integer(m@p), f("_p.bin"), size = 4)
  writeBin(as.double(m@x), f("_x.bin"), size = 8)
  writeLines(rownames(m), f("_rows.txt")); writeLines(colnames(m), f("_cols.txt"))
}
