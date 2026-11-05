"""Mixture models for single-crystal ⁴⁰Ar/³⁹Ar ages: the maths and the drawing
behind the Mixture view of the Age distribution tool.

Discrete (finite) mixture: the analyses are taken to come from k age
components; each component's age and the proportion of analyses in it are
found by maximum likelihood (Galbraith & Green 1990; Galbraith & Laslett 1993),
by the EM algorithm, with uncertainties from the observed information matrix
(Galbraith 2005), as in IsoplotR's peakfit (Vermeesch 2018).  The number of
components is chosen with the Bayes Information Criterion.

Minimum age model: one youngest discrete age plus a continuous spread of older
analyses, truncated at it (Galbraith et al. 1999), with the normal likelihood
of Galbraith (2005, section 6.11).

Everything works on z with standard error s:
    'linear'  z = t,      s = σ_t      (keeps zero and negative ages)
    'log'     z = ln t,   s = σ_t / t  (IsoplotR's default; needs t > 0)
The inputs are ages (Ma) with their ANALYTICAL 1σ.  J and the calibration are
shared by every analysis of a sample, so they move all components together and
are added to each component age afterwards (component_tiers), as for a
weighted mean.
"""
import math
import numpy as np
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy import optimize, stats
from scipy.special import logsumexp

from arar_math import internal_sigma_array, external_sigma_array, mswd_critical


# ---------------------------------------------------------------------------
# transformation
# ---------------------------------------------------------------------------
def to_z(t, s, space):
    t = np.asarray(t, float)
    s = np.abs(np.asarray(s, float))
    ok = np.isfinite(t) & np.isfinite(s) & (s > 0)
    if space == 'log':
        ok &= t > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(np.where(ok, t, 1.0)), s / np.where(ok, t, 1.0), ok
    return t, s, ok


def from_z(mu, se, space):
    """Component value and 1σ back in Ma."""
    if space == 'log':
        t = np.exp(mu)
        return t, t * se
    return np.asarray(mu, float), np.asarray(se, float)


# ---------------------------------------------------------------------------
# discrete (finite) mixture
# ---------------------------------------------------------------------------
def _loglik_terms(z, s, mu, pi):
    """log(π_i f_ui) for every grain u and component i  (n × k)."""
    return (np.log(np.maximum(pi, 1e-300))[None, :]
            + stats.norm.logpdf(z[:, None], mu[None, :], s[:, None]))


def _em(z, s, mu, pi, max_iter=5000, tol=1e-12):
    w = 1.0 / s**2
    L_old = -np.inf
    for it in range(max_iter):
        lp = _loglik_terms(z, s, mu, pi)
        L = logsumexp(lp, axis=1)
        post = np.exp(lp - L[:, None])                      # p_ui, membership
        pi = post.mean(axis=0)
        mu = (post * (w * z)[:, None]).sum(0) / np.maximum((post * w[:, None]).sum(0), 1e-300)
        Ls = L.sum()
        if abs(Ls - L_old) <= tol * max(1.0, abs(Ls)):
            break
        L_old = Ls
    lp = _loglik_terms(z, s, mu, pi)
    L = logsumexp(lp, axis=1)
    post = np.exp(lp - L[:, None])
    return mu, pi, post, L.sum(), it + 1


def _starts(z, s, k, rng, prev=None):
    """Starting locations: evenly over the range (IsoplotR), weighted
    quantiles, random picks among the more precise grains and, given the
    (k−1)-component fit, that fit plus one component at the worst-fitting
    grain or with its most scattered component split in two."""
    out = []
    if k == 1:
        return [np.array([np.median(z)])]
    if prev is not None:
        mu, pi = prev
        lp = _loglik_terms(z, s, mu, pi)
        L = logsumexp(lp, axis=1)
        out.append(np.sort(np.r_[mu, z[np.argmin(L)]]))
        post = np.exp(lp - L[:, None])
        r2 = (post * ((z[:, None] - mu[None, :]) / s[:, None])**2).sum(0) / np.maximum(post.sum(0), 1e-9)
        j = int(np.argmax(r2))
        spread = math.sqrt(max(np.sum(post[:, j] * (z - mu[j])**2) / max(post[:, j].sum(), 1e-9), 1e-12))
        out.append(np.sort(np.r_[np.delete(mu, j), mu[j] - spread, mu[j] + spread]))
    out.append(np.linspace(z.min(), z.max(), k))
    order = np.argsort(z)
    w = 1.0 / s[order]**2
    cw = np.cumsum(w) / w.sum()
    out.append(np.interp((np.arange(k) + 0.5) / k, cw, z[order]))
    p = (1.0 / s**2) / (1.0 / s**2).sum()
    for _ in range(20):
        idx = rng.choice(len(z), size=k, replace=False, p=p) if (p > 0).sum() >= k \
            else rng.choice(len(z), size=k, replace=False)
        out.append(np.sort(z[idx]))
    return out


def information_cov(z, s, mu, pi, post):
    """Covariance of (π_1..π_{k-1}, μ_1..μ_k): the inverse of the observed
    information matrix (Galbraith 2005; IsoplotR get_peakfit_covmat)."""
    k = len(mu)
    a = (z[:, None] - mu[None, :]) / s[:, None]**2               # ∂ ln f / ∂μ
    b = a**2 - 1.0 / s[:, None]**2                                # ∂² ln f / ∂μ² + a²
    C = (post[:, :, None] * post[:, None, :] * a[:, :, None] * a[:, None, :]).sum(0)
    C -= np.diag((b * post).sum(0))
    if k == 1:
        return np.linalg.inv(C)
    if np.any(pi < 1e-12):
        return np.full((2 * k - 1, 2 * k - 1), np.nan)
    d = post[:, :k - 1] / pi[None, :k - 1] - (post[:, k - 1] / pi[k - 1])[:, None]
    A = d.T @ d
    B = np.zeros((k - 1, k))
    for i in range(k - 1):
        for j in range(k):
            B[i, j] = np.sum(post[:, j] * a[:, j] * (d[:, i] - (i == j) / pi[j] + (j == k - 1) / pi[k - 1]))
    I = np.block([[A, B], [B.T, C]])
    if not np.all(np.isfinite(I)):
        return np.full(I.shape, np.nan)
    try:
        return np.linalg.inv(I)
    except np.linalg.LinAlgError:
        return np.full(I.shape, np.nan)


def fit_fmm(t, s, k, space='linear', seed=0, prev=None):
    """Discrete mixture with k components.  Returns a dict in Ma, components
    sorted young to old.  prev: the (k−1)-component fit, used for starts."""
    z, sz, ok = to_z(t, s, space)
    z, sz = z[ok], sz[ok]
    n = len(z)
    if n < 2 * k:
        return None
    rng = np.random.default_rng(seed)
    # short runs from every start, then the three best run to convergence
    short = [_em(z, sz, np.array(mu0, float), np.full(k, 1.0 / k), max_iter=100)
             for mu0 in _starts(z, sz, k, rng,
                                None if prev is None else (prev['mu'], prev['prop']))]
    short.sort(key=lambda r: -r[3])
    best = None
    for mu0, pi0, *_ in short[:3]:
        mu, pi, post, L, it = _em(z, sz, mu0, pi0, max_iter=3000, tol=1e-11)
        if best is None or L > best[3] + 1e-9:
            best = (mu, pi, post, L, it)
    mu, pi, post, L, it = best
    o = np.argsort(mu)
    mu, pi, post = mu[o], pi[o], post[:, o]
    E = information_cov(z, sz, mu, pi, post)
    se_mu = np.sqrt(np.clip(np.diag(E)[k - 1:], 0, None))
    if k > 1:
        Ep = E[:k - 1, :k - 1]
        se_pi = np.r_[np.sqrt(np.clip(np.diag(Ep), 0, None)), math.sqrt(max(Ep.sum(), 0.0))]
    else:
        se_pi = np.zeros(1)
    age, se_age = from_z(mu, se_mu, space)
    r = (z[:, None] - mu[None, :]) / sz[:, None]
    neff = post.sum(0)
    # membership-weighted MSWD of each component (diagnostic, not part of the
    # model); like any MSWD it needs at least two analyses
    mswd = np.where(neff > 2 - 1e-6, (post * r**2).sum(0) / np.maximum(neff - 1, 1e-9), np.nan)
    full_post = np.full((len(ok), k), np.nan)
    full_post[ok] = post
    # an empty or duplicated component is not a component: such a k is never chosen
    degenerate = bool(np.any(neff < 0.5) or np.any(np.diff(mu) < 1e-6 * max(1.0, np.ptp(z)))
                      or not np.all(np.isfinite(se_mu)))
    return dict(model='fmm', k=k, space=space, n=n, used=ok, L=L, iterations=it, degenerate=degenerate,
                bic=-2 * L + (2 * k - 1) * math.log(n), aic=-2 * L + 2 * (2 * k - 1),
                mu=mu, se_mu=se_mu, age=age, se_age=se_age, prop=pi, se_prop=se_pi,
                post=full_post, n_eff=neff,
                n_assigned=np.bincount(post.argmax(1), minlength=k), mswd=mswd)


def select_k(t, s, kmax=10, space='linear', delta=2.0):
    """Fits for k = 1, 2, … and the k with the lowest BIC.  Stops at kmax or
    when two k in a row have not lowered the BIC.  (IsoplotR tries up to 5
    and stops at the first k that does not lower it; single old xenocrysts
    each take a component, so 5 is too few for detrital-type samples.)
    The k returned is the smallest within `delta` of the lowest BIC."""
    fits = []
    best = np.inf
    worse = 0
    for k in range(1, kmax + 1):
        f = fit_fmm(t, s, k, space, prev=fits[-1] if fits else None)
        if f is None:
            break
        fits.append(f)
        if not f['degenerate'] and f['bic'] < best:
            best, worse = f['bic'], 0
        else:
            worse += 1
            if worse >= 2:
                break
    if not fits:
        return None, []
    ok = [f for f in fits if not f['degenerate']] or fits[:1]
    bmin = min(f['bic'] for f in ok)
    # the fewest components within `delta` of the lowest BIC: a BIC difference
    # below 2 is "not worth more than a bare mention" (Kass & Raftery 1995)
    kbest = min(f['k'] for f in ok if f['bic'] <= bmin + delta)
    return kbest, fits


# ---------------------------------------------------------------------------
# minimum age model (Galbraith et al. 1999; Galbraith 2005 §6.11)
# ---------------------------------------------------------------------------
def _mam_nll(gam, prop, sig, mu, z, s):
    A = prop * stats.norm.pdf(z, gam, s)
    s2 = sig**2 + s**2
    mu0 = (mu / sig**2 + z / s**2) / (1 / sig**2 + 1 / s**2)
    s0 = 1 / np.sqrt(1 / sig**2 + 1 / s**2)
    num = stats.norm.sf((gam - mu0) / s0)
    den = max(stats.norm.sf((gam - mu) / sig), 1e-300)
    Bc = (1 - prop) * stats.norm.pdf(z, mu, np.sqrt(s2)) * num / den
    f = np.clip(A + Bc, 1e-300, 1e300)
    return -np.log(f).sum()


def _central(z, s):
    """Central (random-effects) model, Galbraith 2005 p.100 (IsoplotR
    continuous_mixture): mean and overdispersion σ."""
    sig = max(np.std(z), 1e-6)
    for _ in range(60):
        w = 1 / (sig**2 + s**2)
        m = np.sum(w * z) / np.sum(w)
        f = lambda kap: (1 - np.sum((1 / (np.exp(kap)**2 + s**2) * (z - m))**2)
                         / np.sum(1 / (np.exp(kap)**2 + s**2)))**2
        r = optimize.minimize_scalar(f, bounds=(math.log(sig) - 3, math.log(sig) + 3), method='bounded')
        sig = math.exp(r.x)
    return m, sig


def _num_hessian(f, x, h=1e-4):
    x = np.asarray(x, float)
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ej = np.zeros(n)
            hi = h * max(1.0, abs(x[i])); hj = h * max(1.0, abs(x[j]))
            ei[i] = hi; ej[j] = hj
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * hi * hj)
    return H


def fit_mam(t, s, npar=3, space='log'):
    """Minimum age model.  Returns the minimum (youngest discrete) age, its
    analytical 1σ, the proportion of grains in it and the spread σ of the rest."""
    z, sz, ok = to_z(t, s, space)
    z, sz = z[ok], sz[ok]
    zs = np.sort(z)
    mz, Mz = zs[0], zs[-1]
    cm, csig = _central(z, sz)

    def mp(p):
        gam = p[0]
        prop = 1 / (1 + math.exp(-p[1]))
        sig = math.exp(p[2])
        mu = gam if npar < 4 else gam + (Mz - gam) / (1 + math.exp(-p[3]))
        return gam, prop, sig, mu

    nll = lambda p: _mam_nll(*mp(p), z, sz)
    best = None
    # several starting minima: IsoplotR's (mean of the two youngest) plus the
    # weighted quantiles of the younger half
    g0s = [np.mean(zs[:2])] + list(np.quantile(z, [0.05, 0.15, 0.3]))
    for g0 in g0s:
        init = np.array([g0, 0.0, math.log(max(csig, 1e-6)), 0.0][:npar])
        lo = [mz, -20, init[2] - 20, -20][:npar]
        hi = [max(cm, mz + 1e-9), 20, init[2] + 2, 20][:npar]
        init = np.clip(init, lo, hi)
        r = optimize.minimize(nll, init, method='L-BFGS-B', bounds=list(zip(lo, hi)))
        if best is None or r.fun < best.fun:
            best = r
    p = best.x.copy()
    if p[2] < -10:
        p[1] = 10            # σ → 0: all grains in the minimum
    gam, prop, sig, mu = mp(p)
    H = _num_hessian(nll, p)
    try:
        lE = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        lE = np.full((npar, npar), np.nan)
    J = np.eye(npar)
    J[1, 1] = math.exp(p[1]) / (1 + math.exp(p[1]))**2
    J[2, 2] = math.exp(p[2])
    if npar == 4:
        e4 = math.exp(p[3])
        J[3, 0] = 1 - e4 / (1 + e4)
        J[3, 3] = (Mz - gam) * e4 / (1 + e4)**2
    E = J @ lE @ J.T
    se = np.sqrt(np.where(np.diag(E) > 0, np.diag(E), np.nan))
    age, se_age = from_z(np.array([gam]), np.array([se[0]]), space)
    note = ''
    if sig < 0.05 * np.min(sz):
        # no spread above the minimum: the model is the weighted mean of all
        # grains, and its γ uncertainty from the Hessian is not meaningful
        f1 = fit_fmm(t, s, 1, space)
        age, se_age = f1['age'], f1['se_age']
        prop, se = 1.0, np.r_[se[0], 0.0, se[2:]]
        note = 'no grains above the minimum: minimum age = weighted mean'
    # membership of each grain in the minimum component
    A = prop * stats.norm.pdf(z, gam, sz)
    tot = np.exp(-np.array([_mam_nll(gam, prop, sig, mu, np.array([zz]), np.array([ss]))
                            for zz, ss in zip(z, sz)]))
    pm = np.clip(A / tot, 0, 1)
    pmin = np.full(len(ok), np.nan)
    pmin[ok] = pm
    neff = float(pm.sum())
    zg = math.log(age[0]) if space == 'log' and age[0] > 0 else (gam if space == 'log' else age[0])
    mswd = (float(np.sum(pm * ((z - zg) / sz)**2)) / (neff - 1)) if neff > 2 - 1e-6 else float('nan')
    return dict(model='mam', npar=npar, space=space, n=len(z), used=ok, L=-best.fun,
                n_eff=neff, mswd=mswd,
                gamma=gam, se_gamma=se[0], age=age[0], se_age=se_age[0],
                prop=prop, se_prop=se[1], sigma=sig, se_sigma=se[2],
                mu=mu, post_min=pmin, converged=best.success, note=note)




# ---------------------------------------------------------------------------
# component uncertainties: the three tiers
# ---------------------------------------------------------------------------
def components(fit):
    """[(age, model 1σ, MSWD, n_eff), ...] of a fit, youngest first."""
    if fit['model'] == 'fmm':
        return list(zip(fit['age'], fit['se_age'], fit['mswd'], fit['n_eff']))
    return [(fit['age'], fit['se_age'], fit['mswd'], fit['n_eff'])]


def component_tiers(fit, cfg):
    """Analytical, internal and external 1σ of every component age.

    The analytical σ is the model's, multiplied by √MSWD when the analyses of
    the component (weighted by how likely they belong to it) scatter by more
    than the Wendt & Carl (1991) limit for their number, the rule used for
    every weighted mean.  J and the calibration are then added as for a
    weighted mean of the sample.  Returns a list of dicts, youngest first."""
    out = []
    for age, se, mswd, neff in components(fit):
        scale = 1.0
        if neff is not None and mswd == mswd and mswd > mswd_critical(max(neff, 2.0)):
            scale = math.sqrt(mswd)
        ana = se * scale
        si = float(internal_sigma_array([age], [ana], cfg)[0])
        sx = float(external_sigma_array([age], [si], cfg)[0])
        out.append(dict(age=float(age), se_model=float(se), scale=scale, mswd=mswd,
                        n_eff=neff, ana=float(ana), int=si, ext=sx))
    return out


def default_xlim(t, s, fit, ds=2):
    """An age range around the components that hold most of the analyses
    (≥ 5 %), so single old xenocrysts do not set the scale."""
    t = np.asarray(t, float)
    s = np.abs(np.asarray(s, float))
    used = fit['used']
    if fit['model'] == 'fmm':
        comp = np.nanargmax(np.nan_to_num(fit['post'], nan=-1), axis=1)
        sel = used & np.isin(comp, np.where(fit['prop'] >= 0.05)[0])
    else:
        sel = used & ((np.nan_to_num(fit['post_min']) >= 0.5) | (t <= np.quantile(t[used], 0.9)))
    if not sel.any():
        sel = used
    tt, ss = t[sel], s[sel] * ds
    span = np.ptp(tt) + 1e-3
    lo = np.quantile(tt - np.minimum(ss, span), 0.02)
    hi = np.quantile(tt + np.minimum(ss, span), 0.98)
    pad = 0.08 * (hi - lo + 1e-3)
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e',
          '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
GREY = '#9a9a9a'


def _colour(fit, i):
    if fit['model'] == 'mam' and i > 0:
        return GREY
    return COLORS[i % len(COLORS)]


def _pdf(t, s, axis, w=None):
    w = np.ones(len(t)) if w is None else w
    out = np.zeros_like(axis)
    for a, sg, ww in zip(t, s, w):
        if ww > 0 and sg > 0:
            out += ww * np.exp(-0.5 * ((axis - a) / sg)**2) / (sg * math.sqrt(2 * math.pi))
    return out


def membership(fit):
    """(most likely component, its probability, n × k membership matrix) of
    every analysis; component −1 for analyses the model did not use.  For the
    minimum age model the columns are (minimum, older spread)."""
    if fit['model'] == 'fmm':
        P = fit['post']
    else:
        p = fit['post_min']
        P = np.c_[p, 1 - p]
    unused = np.isnan(P[:, 0])
    comp = np.where(unused, -1, np.argmax(np.nan_to_num(P, nan=-1), axis=1))
    prob = np.where(unused, np.nan, np.max(np.nan_to_num(P, nan=0), axis=1))
    return comp, prob, P


def legend_label(fit, i, tier, ds=2, dec=3, tiers_mode='all', with_n=True):
    """'C1  56.794  ±2σ 0.107/0.109/0.118  (47 %, n = 41)' and its variants."""
    af = f'.{dec}f'
    name = 'Minimum' if fit['model'] == 'mam' else f'C{i + 1}'
    a = tier['age']
    if tiers_mode == 'off':
        lab = name
    elif tiers_mode in ('ana', 'int', 'ext'):
        lab = f'{name}  {a:{af}} ± {tier[tiers_mode] * ds:{af}}'
    else:
        lab = (f'{name}  {a:{af}}  ±{ds}σ {tier["ana"] * ds:{af}}/'
               f'{tier["int"] * ds:{af}}/{tier["ext"] * ds:{af}}')
    if with_n and tiers_mode != 'off':
        if fit['model'] == 'fmm':
            lab += f'  ({100 * fit["prop"][i]:.0f} %, n = {fit["n_assigned"][i]})'
        else:
            lab += f'  ({100 * fit["prop"]:.0f} %)'
    return lab


def draw_components(ax, t, s, fit, xlim, tiers, ds=2, dec=3, title=None, show=None,
                    tiers_mode='all', with_n=True, lw=1.0):
    """The ranked analyses and their summed-Gaussian curve (display σ), as in
    the main plot, with the curve split into the membership-weighted curves of
    the components, which add up to it.  Each analysis is coloured by its
    most likely component, faded when its membership is uncertain."""
    show = {} if show is None else show
    on = lambda k: show.get(k, True)
    t = np.asarray(t, float)
    s = np.abs(np.asarray(s, float))
    n = len(t)
    comp, prob, P = membership(fit)
    lo, hi = xlim
    axis = np.linspace(lo, hi, 4000)
    total = _pdf(t, s * ds, axis)
    scale = n / total.max() if total.max() > 0 else 1.0
    if on('fill'):
        base = np.zeros_like(axis)
        for i in range(P.shape[1]):
            c = _pdf(t, s * ds, axis, np.nan_to_num(P[:, i])) * scale
            ax.fill_between(axis, base, base + c, color=_colour(fit, i), alpha=0.35,
                            lw=0, zorder=1)
            base = base + c
    if on('curve'):
        ax.plot(axis, total * scale, color='k', lw=1.4 * lw, zorder=6)
    if on('points'):
        rank = np.empty(n)
        rank[np.argsort(t)] = np.arange(1, n + 1)
        for j in range(n):
            if not (lo - 3 * s[j] * ds <= t[j] <= hi + 3 * s[j] * ds):
                continue
            ci = comp[j]
            col = GREY if ci < 0 else _colour(fit, ci)
            alpha = 0.3 if ci < 0 else 0.35 + 0.65 * prob[j]
            ax.errorbar(t[j], rank[j], xerr=ds * s[j], fmt='o', ms=3.5 * lw, color=col,
                        mfc=col if ci >= 0 else 'none', alpha=alpha, lw=0.9 * lw,
                        capsize=0, zorder=4)
    for i, tr in enumerate(tiers):
        a = tr['age']
        if not (lo <= a <= hi):
            continue
        col = _colour(fit, i) if fit['model'] == 'fmm' else COLORS[0]
        if on('bands'):
            ax.axvspan(a - ds * tr['ana'], a + ds * tr['ana'], color=col, alpha=0.25,
                       lw=0, zorder=2)
            ax.axvline(a, color=col, lw=1.3 * lw, ls='--', zorder=5)
        ax.plot([], [], color=col, lw=6, alpha=0.6,
                label=legend_label(fit, i, tr, ds, dec, tiers_mode, with_n))
    off = [tr['age'] for tr in tiers if not (lo <= tr['age'] <= hi)]
    if off:
        lab = (f'+1 component outside the age range ({off[0]:.{dec}f} Ma)' if len(off) == 1 else
               f'+{len(off)} components outside the age range ({min(off):.0f}–{max(off):.0f} Ma)')
        ax.plot([], [], color='none', label=lab)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, n * 1.08)
    ax.set_xlabel('Age (Ma)')
    ax.set_ylabel('Rank')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)
    leg = ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0,
                    frameon=False, prop={'size': 8}, title=title, alignment='left',
                    title_fontproperties={'size': 9, 'weight': 'bold'})
    leg.set_zorder(20)
    return leg


def draw_radial(ax, t, s, fit, xlim, space='linear'):
    """Radial plot (Galbraith 1988): x = 1/σ, y = (z − z0)/σ, with z0 the
    weighted mean.  An analysis of value z lies on the line from the origin
    with slope z − z0, so its value is read where that line meets the arc;
    each component is drawn as such a line.  Only analyses within xlim are
    drawn, and the arc spans the analyses that carry weight."""
    t = np.asarray(t, float)
    s = np.abs(np.asarray(s, float))
    comp, prob, P = membership(fit)
    z, sz, ok = to_z(t, s, space)
    lo, hi = xlim
    inr = ok & (t >= lo) & (t <= hi)
    if inr.sum() < 1:
        ax.text(0.5, 0.5, 'No analyses in the age range', transform=ax.transAxes, ha='center')
        return
    zf = np.log if space == 'log' else (lambda v: np.asarray(v, float))
    ages = np.atleast_1d(fit['age'])
    # arc range: the analyses with at least 10 % of the best precision, and
    # the component ages in view; imprecise analyses sit near the origin and
    # would only stretch the scale
    prec = 1 / sz[inr]
    zz = np.r_[z[inr][prec >= 0.1 * prec.max()], zf(ages[(ages >= lo) & (ages <= hi) & (ages > 0 if space == 'log' else True)])]
    pad = 0.08 * max(zz.max() - zz.min(), 1e-6)
    zm, zM = zz.min() - pad, zz.max() + pad
    w = 1 / sz[inr]**2
    z0 = np.sum(w * z[inr]) / np.sum(w)
    rx, ry = 1 / sz[inr], (z[inr] - z0) / sz[inr]
    # the arc is an ellipse in data units that looks round on the axes
    # (IsoplotR radial_scale)
    bb = ax.get_window_extent()
    e = (bb.height / max(bb.width, 1)) / (zM - zm)
    th = np.arctan(e * (z[inr] - z0))
    xM = 1.08 * math.sqrt(np.max((rx**2 + ry**2) / (np.cos(th)**2 + (np.sin(th) / e)**2)))

    def arc(zv, r):
        a = np.arctan(e * (np.asarray(zv) - z0))
        return r * np.cos(a), (r / e) * np.sin(a)

    ax.plot(*arc(np.linspace(zm, zM, 200), xM), color='k', lw=0.9)
    tlo, thi = (math.exp(zm), math.exp(zM)) if space == 'log' else (zm, zM)
    for v in [v for v in MaxNLocator(6).tick_values(tlo, thi) if tlo <= v <= thi]:
        x0, y0 = arc(zf(v), xM)
        x1, y1 = arc(zf(v), xM * 1.025)
        ax.plot([x0, x1], [y0, y1], color='k', lw=0.9)
        ax.text(x1 * 1.01, y1, f'{v:g}', va='center', ha='left', fontsize=8)
    for i, a in enumerate(ages):
        if lo <= a <= hi and (a > 0 or space != 'log'):
            x1, y1 = arc(zf(a), xM)
            ax.plot([0, x1], [0, y1], color=_colour(fit, i) if fit['model'] == 'fmm' else COLORS[0],
                    lw=1.4, ls='--', zorder=3)
    for jj, j in enumerate(np.where(inr)[0]):
        ci = comp[j]
        col = GREY if ci < 0 else _colour(fit, ci)
        ax.plot(rx[jj], ry[jj], 'o', ms=4.5, mfc=col, mec='k', mew=0.4,
                alpha=0.35 + 0.65 * (prob[j] if ci >= 0 else 0), zorder=4)
    ax.set_xlim(0, xM * 1.25)
    ys = np.r_[arc(np.array([zm, zM]), xM)[1], ry]
    ax.set_ylim(min(ys.min(), -3) * 1.05, max(ys.max(), 3) * 1.05)
    ax.spines[['top', 'right']].set_visible(False)
    yl = ax.get_ylim()
    ax.set_yticks([-2, 0, 2] if yl[1] - yl[0] < 30 else [0])
    ax.tick_params(labelsize=8)
    ax.set_ylabel('Standardised estimate')
    ax.set_xlabel('Precision (1/σ)' if space == 'linear' else 'Precision (t/σ)')
    nout = int((ok & ~inr).sum())
    if nout:
        ax.text(0.02, 0.02, f'{nout} analyses outside {lo:.4g}–{hi:.4g} Ma not shown',
                transform=ax.transAxes, fontsize=7, color='#666')


def draw_bic(ax, fits, k_shown, k_bic):
    """ΔBIC (BIC minus the lowest) for every number of components tried.
    The bar of the fit shown is coloured; the BIC choice is marked ▼."""
    ks = [f['k'] for f in fits]
    d = np.array([f['bic'] for f in fits]) - min(f['bic'] for f in fits)
    # a floor on the bar height, so a bar at ΔBIC ≈ 0 still shows its colour
    ax.bar(ks, np.clip(d, 0.6, 30), color=[COLORS[0] if k == k_shown else '#bbbbbb' for k in ks])
    for k, v in zip(ks, d):
        ax.text(k, min(v, 30) + 0.8, ('>30' if v > 30 else f'{v:.1f}') + (' ▼' if k == k_bic else ''),
                ha='center', fontsize=6.5)
    ax.set_xticks(ks)
    ax.set_ylim(0, 36)
    ax.set_xlabel('Number of components', fontsize=7.5)
    ax.set_ylabel('ΔBIC', fontsize=7.5)
    ax.tick_params(labelsize=7)
