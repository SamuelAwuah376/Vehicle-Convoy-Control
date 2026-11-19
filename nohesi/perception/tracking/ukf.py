"""Unscented Kalman filter primitives for the team tracker (design section 6).

API (the tracker, imm.py and belief.py build on it):
- sigma_points(mean, cov, alpha=ALPHA, beta=BETA, kappa=KAPPA) -> (X (2n+1, n), wm (2n+1,), wc (2n+1,)):
  the scaled unscented transform's points [mean, mean + cols of sqrt(c P), mean - cols], c = alpha^2 (n + kappa),
  from a robust Cholesky factor (symmetrized, eigenvalue-floored and jittered when the plain factor fails).
- ukf_predict(mean, cov, f, Q) -> (mean, cov): f maps sigma points (k, n) -> (k, n) (one time step already bound in,
  e.g. lambda X: model(X, dt)); Q is added to the propagated covariance.
- ukf_update(mean, cov, z, R, h, residual=None) -> UpdateResult(mean, cov, innovation, S, nis, loglik): h maps sigma
  points (k, n) -> (k, m); residual(a, b) -> a - b with any angle components wrapped (default plain subtraction), used
  for the predicted measurement's mean, its spread and the innovation, so angles near +-pi average correctly.
  The covariance update is cov - K S K^T, symmetrized, with an eigenvalue floor (EIG_REL x the largest, at least
  EIG_ABS). nis = innovation^T S^-1 innovation (chi-square with m dof when consistent); loglik = log N(z; zhat, S).
- robust_cholesky(P), floor_eigenvalues(P): the numerical safeguards, exposed for the tracker's own matrices.

Batching: every function also takes stacked filters, mean (..., n) and cov (..., n, n); f and h still receive one
2-D array of all sigma points (k = batch x (2n+1) rows) and the outputs keep the batch shape. z (m,) or (..., m) and
R (m, m) or (..., m, m) broadcast against it. A single small filter is dominated by numpy call overhead (~0.1 ms),
so the tracker should batch: one call for 45 tracks x 3 IMM models costs about five single filters.

Default sigma-point parameters alpha 1, beta 2, kappa 0: the points sit at +-sqrt(n) sigma with equal positive
weights (wm0 = 0, wc0 = 2), which keeps every covariance weight positive (a small alpha gives a large negative wc0
and can lose positive definiteness on the box-geometry and image-clipping nonlinearities of the sensor models).
"""
from functools import lru_cache
from typing import NamedTuple
import numpy as np

__all__ = ['ALPHA', 'BETA', 'KAPPA', 'EIG_REL', 'EIG_ABS', 'UpdateResult', 'sigma_points', 'ukf_predict', 'ukf_update',
           'robust_cholesky', 'floor_eigenvalues', 'symmetrize']

ALPHA, BETA, KAPPA = 1.0, 2.0, 0.0
EIG_REL, EIG_ABS = 1e-12, 1e-18     # covariance eigenvalue floor: relative to the largest, and absolute
JITTER, JITTER_STEPS = 1e-12, 14    # Cholesky retries add JITTER x 10^i x mean variance to the diagonal


class UpdateResult(NamedTuple):
    """ukf_update's output. Shapes for one filter: mean (n,), cov (n, n), innovation (m,), S (m, m), nis and loglik
    scalars (batched: leading batch dims on each)."""
    mean: np.ndarray
    cov: np.ndarray
    innovation: np.ndarray
    S: np.ndarray
    nis: np.ndarray
    loglik: np.ndarray


def symmetrize(P):
    P = np.asarray(P, float)
    return 0.5*(P+np.swapaxes(P, -1, -2))


def floor_eigenvalues(P):
    """P symmetrized, with eigenvalues below max(EIG_REL x the largest, EIG_ABS) raised to that floor (per matrix).
    Matrices already above the floor come back only symmetrized."""
    P = symmetrize(P)
    w, V = np.linalg.eigh(P)
    floor = np.maximum(EIG_REL*np.abs(w).max(-1), EIG_ABS)
    bad = w[..., 0] < floor
    if not np.any(bad):
        return P
    fixed = symmetrize((V*np.maximum(w, floor[..., None])[..., None, :]) @ np.swapaxes(V, -1, -2))
    return np.where(bad[..., None, None], fixed, P)


def robust_cholesky(P):
    """Lower Cholesky factor L (L L^T = P) of symmetric P (..., n, n). On failure (not positive definite from round-off
    or a degenerate matrix) the eigenvalues are floored and growing diagonal jitter is tried. Raises ValueError for a
    non-finite matrix."""
    P = symmetrize(P)
    if not np.all(np.isfinite(P)):
        raise ValueError('non-finite covariance')
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        pass
    P = floor_eigenvalues(P)
    n = P.shape[-1]
    scale = np.maximum(np.trace(P, axis1=-2, axis2=-1)/n, EIG_ABS)[..., None, None]*np.eye(n)
    for i in range(JITTER_STEPS):
        try:
            return np.linalg.cholesky(P+JITTER*10.0**i*scale)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError('covariance could not be factorized')


@lru_cache(maxsize=32)
def _weights(n, alpha, beta, kappa):
    c = alpha**2*(n+kappa)
    if c <= 0:
        raise ValueError('sigma-point scale alpha^2 (n + kappa) must be positive')
    wm = np.full(2*n+1, 1/(2*c))
    wm[0] = 1-n/c
    wc = wm.copy()
    wc[0] += 1-alpha**2+beta
    wm.setflags(write=False)
    wc.setflags(write=False)
    return c, wm, wc


def sigma_points(mean, cov, alpha=ALPHA, beta=BETA, kappa=KAPPA):
    """Scaled unscented sigma points of N(mean, cov): X (..., 2n+1, n) and the mean and covariance weights (2n+1,)
    (read-only, shared)."""
    mean, cov = np.asarray(mean, float), np.asarray(cov, float)
    n = mean.shape[-1]
    c, wm, wc = _weights(n, float(alpha), float(beta), float(kappa))
    if mean.shape[:-1] != cov.shape[:-2]:
        batch = np.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])
        mean, cov = np.broadcast_to(mean, batch+(n,)), np.broadcast_to(cov, batch+(n, n))
    D = np.swapaxes(robust_cholesky(cov), -1, -2)*np.sqrt(c)     # rows: the factor's columns
    m = mean[..., None, :]
    return np.concatenate((m, m+D, m-D), axis=-2), wm, wc


def _apply(fn, X, name):
    """fn on the sigma points flattened to (k, n), reshaped back to X's batch shape."""
    flat = X.reshape(-1, X.shape[-1])
    Y = np.asarray(fn(flat), float)
    if Y.ndim != 2 or Y.shape[0] != flat.shape[0]:
        raise ValueError(f'{name} must map sigma points (k, n) to (k, m), got {Y.shape} for {flat.shape}')
    return Y.reshape(X.shape[:-1]+Y.shape[-1:])


def _subtract(a, b):
    return np.subtract(a, b)


def _cross(w, A, B):
    """sum_i w_i A_i B_i^T over the sigma-point axis: (..., k, p), (..., k, q) -> (..., p, q)."""
    return np.swapaxes(A*w[:, None], -1, -2) @ B


def ukf_predict(mean, cov, f, Q, alpha=ALPHA, beta=BETA, kappa=KAPPA):
    """Propagate N(mean, cov) through f (sigma points (k, n) -> (k, n)) and add process noise Q."""
    X, wm, wc = sigma_points(mean, cov, alpha, beta, kappa)
    Y = _apply(f, X, 'f')
    m = wm @ Y
    dY = Y-m[..., None, :]
    return m, symmetrize(_cross(wc, dY, dY)+Q)


def ukf_update(mean, cov, z, R, h, residual=None, alpha=ALPHA, beta=BETA, kappa=KAPPA):
    """Measurement update of N(mean, cov) with z ~ N(h(x), R). See the module docstring for residual and the
    outputs."""
    res = _subtract if residual is None else residual
    mean = np.asarray(mean, float)
    X, wm, wc = sigma_points(mean, cov, alpha, beta, kappa)
    Z = _apply(h, X, 'h')
    ref = Z[..., :1, :]
    zhat = ref[..., 0, :]+wm @ res(Z, ref)                       # the mean taken relative to the centre point
    dZ = res(Z, zhat[..., None, :])
    L = robust_cholesky(_cross(wc, dZ, dZ)+R)
    S = L @ np.swapaxes(L, -1, -2)
    Linv = np.linalg.inv(L)
    Pxz = _cross(wc, X-mean[..., None, :], dZ)
    K = Pxz @ (np.swapaxes(Linv, -1, -2) @ Linv)
    z = np.asarray(z, float)
    nu = res(z if z.shape == zhat.shape else np.broadcast_to(z, zhat.shape), zhat)
    white = (Linv @ nu[..., None])[..., 0]
    nis = (white*white).sum(-1)
    logdet = 2*np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1)
    loglik = -0.5*(nis+logdet+nu.shape[-1]*np.log(2*np.pi))
    post = mean+(K @ nu[..., None])[..., 0]
    P = floor_eigenvalues(np.asarray(cov, float)-K @ np.swapaxes(Pxz, -1, -2))
    return UpdateResult(post, P, nu, S, nis, loglik)
