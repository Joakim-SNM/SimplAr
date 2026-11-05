"""GLS + ML combination (arar_math.combine_ages_vermeesch) against independent
references.  No data files needed.

1. ml_overdispersion against a published random-effects fit: the BCG vaccine
   trials (Colditz et al. 1994), the standard example of the R package metafor
   (rma, REML: tau^2 = 0.3132, mu = -0.7145, se = 0.1798).
2. The ML overdispersion, GLS mean and external tier against a separate
   implementation (joint optimisation over mu and omega, direct matrix
   inversion) on random data sets with shared J and calibration blocks.
3. Without random effects and without covariance, GLS + ML must equal the
   ordinary inverse-variance weighted mean.
"""
import math
import os
import sys

import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arar_math as am


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'BAD'} {label:44s} {got:.6g}  vs  {want:.6g}")
    assert ok, label


def test_bcg_reml():
    print('REML overdispersion vs metafor (BCG trials, log risk ratio):')
    bcg = [(4, 119, 11, 128), (6, 300, 29, 274), (3, 228, 11, 209),
           (62, 13536, 248, 12619), (33, 5036, 47, 5761), (180, 1361, 372, 1079),
           (8, 2537, 10, 619), (505, 87886, 499, 87892), (29, 7470, 45, 7232),
           (17, 1699, 65, 1600), (186, 50448, 141, 27197), (5, 2493, 3, 2338),
           (27, 16886, 29, 17825)]
    yi = np.array([math.log((a / (a + b)) / (c / (c + d))) for a, b, c, d in bcg])
    vi = np.array([1 / a - 1 / (a + b) + 1 / c - 1 / (c + d) for a, b, c, d in bcg])
    o2 = am.ml_overdispersion(yi, np.sqrt(vi), reml=True)
    w = 1 / (vi + o2)
    check('tau^2', o2, 0.3132, 5e-5)
    check('mu', (w * yi).sum() / w.sum(), -0.7145, 5e-5)
    check('se', math.sqrt(1 / w.sum()), 0.1798, 5e-5)


def _independent_ml(t, V0):
    """(mu, omega^2) maximising the normal likelihood with V = V0 + omega^2 I."""
    n = len(t)

    def nll(p):
        mu, log_om = p
        V = V0 + math.exp(2 * log_om) * np.eye(n)
        r = t - mu
        return 0.5 * (np.linalg.slogdet(V)[1] + r @ np.linalg.inv(V) @ r)

    best = None
    for lo0 in np.linspace(math.log(1e-6 * t.std() + 1e-12), math.log(3 * t.std() + 1e-12), 9):
        r = optimize.minimize(nll, [t.mean(), lo0], method='Nelder-Mead',
                              options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
        if best is None or r.fun < best.fun:
            best = r

    def nll0(mu):
        r = t - mu
        return 0.5 * (np.linalg.slogdet(V0)[1] + r @ np.linalg.inv(V0) @ r)

    b0 = optimize.minimize_scalar(nll0)
    if b0.fun <= best.fun:
        return b0.x, 0.0
    return best.x[0], math.exp(2 * best.x[1])


def test_against_independent_ml(trials=60):
    print(f'GLS + ML vs an independent implementation ({trials} random data sets):')
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(4, 25))
        a = rng.uniform(0.005, 0.08, n)                    # analytical 1σ, Ma
        pos = rng.integers(0, 3, n)                        # three irradiation positions
        p = a * rng.uniform(0.2, 1.5, n)                   # J part of the internal tier
        s = np.full(n, 0.03)                               # calibration part
        t = 55.0 + rng.normal(0, rng.choice([0, 0.01, 0.03, 0.06]), n) + rng.normal(0, a)
        si = np.sqrt(a ** 2 + p ** 2)
        se = np.sqrt(si ** 2 + s ** 2)
        res = am.combine_ages_vermeesch(t, a, si, se, position_group=list(pos),
                                        return_matrices=True)
        M = res['matrices']
        V0 = np.diag(a ** 2) + M['P'] + M['S']
        _, om2 = _independent_ml(t, V0)
        V = V0 + om2 * np.eye(n)
        Vi = np.linalg.inv(V)
        w = Vi.sum(1) / Vi.sum()
        worst = max(worst,
                    abs(res['wm'] - w @ t),
                    abs(math.sqrt(M['omega2']) - math.sqrt(om2)),
                    abs(res['sigma_ext'] - math.sqrt(w @ V @ w)))
    check('largest difference (Ma)', worst, 0.0, 1e-6)


def test_reduces_to_weighted_mean():
    print('GLS without random effects or covariance = inverse-variance WM:')
    rng = np.random.default_rng(2)
    a = rng.uniform(0.01, 0.05, 12)
    t = 55.0 + rng.normal(0, a)
    g = am.combine_ages_vermeesch(t, a, random_effects=False)
    w = am.combine_ages(t, a, scale_overdispersion=False)
    check('mean (Ma)', g['wm'], w['wm'], 1e-12)
    check('sigma (Ma)', g['sigma_ana'], w['sigma_ana'], 1e-12)


if __name__ == '__main__':
    test_bcg_reml()
    test_against_independent_ml()
    test_reduces_to_weighted_mean()
    print('ALL PASSED')
