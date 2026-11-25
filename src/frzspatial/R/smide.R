# glmmTMB engine of frzspatial.de.smide(engine="glmmtmb"): one NB2 (log link) mixed model per gene.
#   Rscript smide.R <dir> <workers> <max_grad>
# <dir> holds y_* (cells x genes counts), z.bin (cells x genes neighbour covariate, row-major float64, NaN = term
# omitted), design.csv (X1..Xp: fixed-effect design built in Python; log_offset; the random-effect grouping columns)
# and re.txt (random-effect terms, lme4 syntax). Writes fits.csv (one row per gene) and vcov.bin (each gene's
# fixed-effect covariance, row-major, at vcov_offset).
# The fit follows the original glomerular smiDE pipeline (183_nephron_model_functions.R, V. Charu 2026-10): no zero
# inflation, constant dispersion, nlminb then BFGS, valid only with convergence, a positive-definite Hessian, finite
# positive SEs, max |gradient| < max_grad, a positive dispersion and a finite random-effect SD.
here = dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(here, "io.R"))
suppressPackageStartupMessages(library(glmmTMB))
args = commandArgs(trailingOnly = TRUE)
DIR = args[1]; WORKERS = as.integer(args[2]); MAX_GRAD = as.numeric(args[3])
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")

# 0. Inputs
y = read_counts(DIR, "y")
n = nrow(y); G = ncol(y); genes = colnames(y)
Z = matrix(readBin(file.path(DIR, "z.bin"), "double", n = n * G, size = 8), nrow = n, ncol = G, byrow = TRUE)
d = as.data.frame(fread(file.path(DIR, "design.csv")))
xcols = grep("^X[0-9]+$", names(d), value = TRUE)
re = trimws(readLines(file.path(DIR, "re.txt"))[1])
for (v in setdiff(names(d), c(xcols, "log_offset"))) d[[v]] = factor(d[[v]])
rhs = paste(c("0", xcols, "offset(log_offset)", if (nzchar(re)) re), collapse = " + ")
say(n, " cells, ", G, " genes, ", length(xcols), " fixed-effect columns, random effects: ", if (nzchar(re)) re else "none")

# 1. One gene
fit_gene = function(j) {
  dd = d; dd$y = as.numeric(y[, j]); z = Z[, j]; has_z = all(is.finite(z))
  if (has_z) dd$Z = z
  form = as.formula(paste("y ~", rhs, if (has_z) "+ Z"), env = globalenv())
  res = list(gene = genes[j], neighbor_term = has_z, n_coef = length(xcols) + has_z, beta = rep(NA_real_, length(xcols) + has_z),
             vcov = matrix(NA_real_, length(xcols) + has_z, length(xcols) + has_z), theta = NA_real_, re_sd = NA_real_,
             loglik = NA_real_, converged = FALSE, pdHess = FALSE, max_grad = NA_real_, valid = FALSE,
             optimizer = "", n_attempts = 0L, error = "")
  for (opt in c("nlminb", "BFGS")) {
    res$n_attempts = res$n_attempts + 1L; res$optimizer = opt
    ctl = if (opt == "nlminb") glmmTMBControl(parallel = 1L, rank_check = "stop", optCtrl = list(iter.max = 1000L, eval.max = 1500L)) else
      glmmTMBControl(parallel = 1L, rank_check = "stop", optimizer = stats::optim, optArgs = list(method = "BFGS"), optCtrl = list(maxit = 2000L))
    err = ""
    fit = tryCatch(suppressWarnings(glmmTMB(form, data = dd, family = nbinom2(link = "log"), ziformula = ~0, dispformula = ~1,
                                            control = ctl)), error = function(e) { err <<- conditionMessage(e); NULL })
    res$error = err
    if (is.null(fit)) next
    b = fixef(fit)$cond; V = tryCatch(as.matrix(vcov(fit)$cond), error = function(e) NULL)
    res$beta = unname(b)
    if (!is.null(V)) res$vcov = unname(V)
    res$theta = tryCatch(as.numeric(sigma(fit)), error = function(e) NA_real_)
    res$re_sd = if (nzchar(re)) tryCatch(as.numeric(attr(VarCorr(fit)$cond[[1]], "stddev"))[1], error = function(e) NA_real_) else NA_real_
    res$loglik = tryCatch(as.numeric(logLik(fit)), error = function(e) NA_real_)
    res$converged = isTRUE(fit$fit$convergence == 0L)
    res$pdHess = isTRUE(fit$sdr$pdHess)
    gr = tryCatch(as.numeric(fit$obj$gr(fit$fit$par)), error = function(e) NA_real_)
    res$max_grad = if (all(is.finite(gr))) max(abs(gr)) else NA_real_
    se = sqrt(diag(res$vcov))
    res$valid = res$converged && res$pdHess && all(is.finite(c(res$beta, se, res$max_grad, res$theta))) && all(se > 0) &&
      res$max_grad < MAX_GRAD && res$theta > 0 && (!nzchar(re) || (is.finite(res$re_sd) && res$re_sd >= 0))
    if (res$valid) { res$error = ""; break }
    if (!nzchar(res$error)) res$error = "invalid fit (convergence, Hessian or gradient)"
  }
  res
}

# 2. All genes in parallel (forked workers; one core on Windows)
workers = if (.Platform$OS.type == "windows") 1L else max(1L, min(WORKERS, G))
say("fitting with ", workers, " workers")
fits = parallel::mclapply(seq_len(G), fit_gene, mc.cores = workers, mc.preschedule = FALSE)
bad = vapply(fits, function(f) inherits(f, "try-error") || is.null(f), logical(1))
if (any(bad)) stop("worker failures for ", sum(bad), " genes")

# 3. For Python
K = max(vapply(fits, function(f) f$n_coef, integer(1)))
off = 0
rows = lapply(fits, function(f) {
  b = c(f$beta, rep(NA_real_, K - length(f$beta)))
  r = data.frame(gene = f$gene, n_coef = f$n_coef, neighbor_term = f$neighbor_term, theta = f$theta, re_sd = f$re_sd,
                 loglik = f$loglik, converged = f$converged, pdHess = f$pdHess, max_grad = f$max_grad, valid = f$valid,
                 optimizer = f$optimizer, n_attempts = f$n_attempts, error = f$error, vcov_offset = off)
  r = cbind(r, as.data.frame(as.list(setNames(b, paste0("b", seq_len(K))))))
  off <<- off + f$n_coef^2
  r
})
fwrite(rbindlist(rows), file.path(DIR, "fits.csv"))
writeBin(as.double(unlist(lapply(fits, function(f) as.vector(t(f$vcov))))), file.path(DIR, "vcov.bin"), size = 8)
say("done: ", sum(vapply(fits, function(f) f$valid, logical(1))), " / ", G, " valid fits")
