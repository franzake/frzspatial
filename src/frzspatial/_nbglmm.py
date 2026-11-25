"""Negative-binomial (NB2, log link) GLM / GLMM with one random-intercept factor, in numpy.

The model of ``frzspatial.de.smide``'s default engine, for one gene:

    y_i | b ~ NB2(mu_i, theta),  log mu_i = x_i' beta + offset_i + b_g(i),  b_g ~ N(0, tau^2)
    Var(y_i | b) = mu_i + mu_i^2 / theta

It maximises the same Laplace-approximated marginal likelihood as glmmTMB (TMB): the random intercepts are integrated
out by a Laplace approximation at their conditional mode, and beta, log(theta) and log(tau) are the outer parameters
(ML, not REML). With a single random-intercept factor the inner problem separates into one 1-D Newton problem per group,
solved for all groups at once. The gradient of the Laplace objective is exact (implicit differentiation through the
conditional modes), the covariance comes from a central-difference Hessian of that gradient, and the fit is judged by
the same rules as the original glmmTMB pipeline (convergence, positive-definite Hessian, max |gradient| < 0.01).
Without a grouping factor it is a plain NB2 GLM.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

_CLIP = 700.0  # exp() overflow guard on the linear predictor
#: Bounds of log(theta) and log(tau): theta up to e^25 (the Poisson limit), tau from e^-20 (no group effect) to e^5.
_ALPHA_BOUNDS = (-20.0, 25.0)
_LAM_BOUNDS = (-20.0, 5.0)


class _Laplace:
    """Objective (minus the Laplace log-likelihood) and its exact gradient, with warm-started conditional modes."""

    def __init__(self, y, X, offset, groups=None, n_groups=0):
        self.y = np.asarray(y, float)
        self.X = np.asarray(X, float)
        self.off = np.asarray(offset, float)
        self.g = None if groups is None else np.asarray(groups, np.intp)
        self.K = int(n_groups) if groups is not None else 0
        self.p = self.X.shape[1]
        self.lgy1 = gammaln(self.y + 1.0).sum()
        self.b = np.zeros(self.K)
        self.n_eval = 0

    # per-observation NB2 pieces at linear predictor eta
    def _pieces(self, eta, theta, alpha):
        eta = np.clip(eta, -_CLIP, _CLIP)
        L = np.logaddexp(alpha, eta)  # log(theta + mu)
        pt = np.exp(alpha - L)  # theta / (theta + mu)
        pm = np.exp(eta - L)  # mu / (theta + mu)
        return eta, L, pt, pm

    def _inner(self, eta0, theta, alpha, tau2):
        """Conditional modes b_g (Newton, all groups at once)."""
        y, g, K = self.y, self.g, self.K
        b = self.b.copy()
        for _ in range(200):
            _, _, pt, pm = self._pieces(eta0 + b[g], theta, alpha)
            s = y * pt - theta * pm
            h = (y + theta) * pt * pm
            grad = np.bincount(g, s, K) - b / tau2
            H = np.bincount(g, h, K) + 1.0 / tau2
            step = np.clip(grad / H, -3.0, 3.0)
            b += step
            if np.max(np.abs(step)) < 1e-10:
                break
        return b

    def __call__(self, par):
        """(f, grad) with f = -log L (Laplace)."""
        self.n_eval += 1
        y, X, p = self.y, self.X, self.p
        beta, alpha = par[:p], float(np.clip(par[p], *_ALPHA_BOUNDS))
        theta = np.exp(alpha)
        eta0 = X @ beta + self.off
        if self.K:
            lam = float(np.clip(par[p + 1], *_LAM_BOUNDS))
            tau2 = np.exp(2.0 * lam)
            b = self._inner(eta0, theta, alpha, tau2)
            self.b = b
            eta = eta0 + b[self.g]
        else:
            eta = eta0
        eta, L, pt, pm = self._pieces(eta, theta, alpha)
        ll = (gammaln(y + theta) - gammaln(theta)).sum() - self.lgy1 + (theta * (alpha - L) + y * (eta - L)).sum()
        s = y * pt - theta * pm  # d ll_i / d eta
        dl_dtheta = digamma(y + theta) - digamma(theta) + (alpha - L) + 1.0 - (y + theta) * np.exp(-L)
        grad = np.empty_like(par)
        if not self.K:
            grad[:p] = X.T @ s
            grad[p] = theta * dl_dtheta.sum()
            return -ll, -grad
        g, K = self.g, self.K
        h = (y + theta) * pt * pm  # -d2 ll_i / d eta2
        t = h * (pt - pm)  # d h_i / d eta
        H = np.bincount(g, h, K) + 1.0 / tau2
        T = np.bincount(g, t, K)
        ll += np.sum(-b * b / (2.0 * tau2) - lam - 0.5 * np.log(H))
        Hg, Tg = H[g], T[g]
        # beta: d/dbeta [sum ll_i] - 1/2 d/dbeta [sum log H_k], with db_k/dbeta = -sum h x / H_k
        grad[:p] = X.T @ (s - 0.5 * t / Hg + 0.5 * Tg * h / (Hg * Hg))
        # log(theta)
        e_L = np.exp(-L)
        ds_dtheta = (y - np.exp(eta)) * pm * e_L  # (y - mu) mu / (theta + mu)^2
        dh_dtheta = pm * (y * (pm - pt) * e_L + 2.0 * pt * pm)
        Sth = theta * np.bincount(g, ds_dtheta, K)
        Hth = theta * np.bincount(g, dh_dtheta, K)
        grad[p] = theta * dl_dtheta.sum() - 0.5 * np.sum((Hth + T * Sth / H) / H)
        # log(tau)
        grad[p + 1] = np.sum(b * b / tau2 - 1.0 - 0.5 * (-2.0 / tau2 + T * (2.0 * b / tau2) / H) / H)
        return -ll, -grad


def _start(y, X, offset):
    """Starting values: intercept-only rate and a moment estimate of theta, other coefficients 0."""
    beta = np.zeros(X.shape[1])
    rate = np.log(max(y.sum(), 0.5) / np.exp(offset).sum())
    icpt = np.flatnonzero(np.all(X == 1.0, axis=0))
    if icpt.size:
        beta[icpt[0]] = rate
    mu = np.exp(rate + offset)
    excess = np.sum((y - mu) ** 2 - mu)
    theta = np.clip(np.sum(mu ** 2) / excess, 1e-2, 1e3) if excess > 0 else 10.0
    return beta, np.log(theta)


def _bounds(p, has_re):
    return [(None, None)] * p + [_ALPHA_BOUNDS] + ([_LAM_BOUNDS] if has_re else [])


def _hessian(obj, par, rel=1e-4):
    """Central-difference Hessian of the exact gradient, symmetrised."""
    n = par.size
    Hm = np.empty((n, n))
    b_saved = obj.b.copy()
    for j in range(n):
        e = np.zeros(n)
        e[j] = rel * max(1.0, abs(par[j]))
        obj.b = b_saved.copy()
        gp = obj(par + e)[1]
        obj.b = b_saved.copy()
        gm = obj(par - e)[1]
        Hm[:, j] = (gp - gm) / (2.0 * e[j])
    obj.b = b_saved
    return 0.5 * (Hm + Hm.T)


def _pd_hessian(obj, par):
    """The Hessian, with larger difference steps when it is not positive definite: a nearly flat direction (theta
    towards the Poisson limit, tau towards 0) has a curvature below the finite-difference noise of small steps, where
    glmmTMB's exact (AD) Hessian is still positive definite."""
    for rel in (1e-4, 1e-3, 1e-2):
        Hm = _hessian(obj, par, rel)
        try:
            np.linalg.cholesky(Hm)
            return Hm
        except np.linalg.LinAlgError:
            continue
    return _hessian(obj, par, 1e-4)


def fit_nb2(y, X, offset, groups=None, n_groups=0, max_grad=1e-2, optimizers=("L-BFGS-B", "BFGS")):
    """Fit one gene. Returns a dict of estimates and diagnostics (never raises on a numerical failure).

    Parameters
    ----------
    y
        Counts (n,).
    X
        Fixed-effect design (n, p), full column rank.
    offset
        Offset on the log scale (n,), e.g. log(total counts).
    groups, n_groups
        Integer codes 0..n_groups-1 of the random-intercept factor, or None for a GLM.
    max_grad
        Largest |gradient| (on the optimiser's scale: beta, log theta, log tau) for a valid fit.
    optimizers
        Tried in order until a valid fit (like nlminb then BFGS in glmmTMB).

    Returns
    -------
    dict
        ``beta`` (p,), ``vcov`` (p, p), ``theta``, ``re_sd``, ``loglik``, ``converged``, ``pdHess``,
        ``max_grad``, ``valid``, ``optimizer``, ``n_attempts``, ``error``.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    offset = np.asarray(offset, float)
    p = X.shape[1]
    has_re = groups is not None and n_groups > 0
    out = dict(beta=np.full(p, np.nan), vcov=np.full((p, p), np.nan), theta=np.nan, re_sd=np.nan, loglik=np.nan,
               converged=False, pdHess=False, max_grad=np.nan, valid=False, optimizer="", n_attempts=0, error="")
    try:
        # 1. NB2 GLM (also the starting point of the GLMM)
        beta0, alpha0 = _start(y, X, offset)
        glm = _Laplace(y, X, offset)
        with np.errstate(all="ignore"):
            r = minimize(glm, np.r_[beta0, alpha0], jac=True, method="L-BFGS-B", bounds=_bounds(p, False),
                         options=dict(maxiter=2000, gtol=1e-8, ftol=1e-14))
        start = r.x if np.all(np.isfinite(r.x)) else np.r_[beta0, alpha0]
        obj = _Laplace(y, X, offset, groups, n_groups) if has_re else glm
        if has_re:
            start = np.r_[start, np.log(0.3)]
    except Exception as e:  # noqa: BLE001  (numerical failure: report, do not raise)
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    for k, method in enumerate(optimizers, start=1):
        out["n_attempts"] = k
        out["optimizer"] = method
        try:
            obj.b = np.zeros(obj.K)
            opts = dict(maxiter=3000, gtol=1e-9) if method == "BFGS" else dict(maxiter=3000, gtol=1e-9, ftol=1e-15)
            bounds = _bounds(p, has_re) if method == "L-BFGS-B" else None
            with np.errstate(all="ignore"):
                r = minimize(obj, start, jac=True, method=method, options=opts, bounds=bounds)
                f, gr = obj(r.x)
                Hm = _pd_hessian(obj, r.x)
            par = r.x
            out["loglik"] = -f
            out["max_grad"] = float(np.max(np.abs(gr)))
            out["converged"] = bool(np.isfinite(f) and (r.success or out["max_grad"] < max_grad))
            out["theta"] = float(np.exp(par[p]))
            out["re_sd"] = float(np.exp(par[p + 1])) if has_re else np.nan
            out["beta"] = par[:p].copy()
            try:
                np.linalg.cholesky(Hm)
                out["pdHess"] = True
                out["vcov"] = np.linalg.inv(Hm)[:p, :p]
            except np.linalg.LinAlgError:
                out["pdHess"] = False
                out["vcov"] = np.full((p, p), np.nan)
            se = np.sqrt(np.diag(out["vcov"]))
            out["valid"] = bool(out["converged"] and out["pdHess"] and np.all(np.isfinite(out["beta"]))
                                and np.all(np.isfinite(se)) and np.all(se > 0) and out["max_grad"] < max_grad
                                and np.isfinite(out["theta"]) and out["theta"] > 0
                                and (not has_re or (np.isfinite(out["re_sd"]) and out["re_sd"] >= 0)))
            out["error"] = "" if out["valid"] else (r.message if isinstance(r.message, str) else str(r.message))
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
        if out["valid"]:
            break
        start = out["beta"].tolist() + [np.log(out["theta"])] if np.all(np.isfinite(out["beta"])) else start
        if has_re and len(start) == p + 1:
            start = start + [np.log(max(out["re_sd"], 1e-3)) if np.isfinite(out["re_sd"]) else np.log(0.3)]
        start = np.asarray(start, float)
    return out
