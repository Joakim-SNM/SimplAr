"""Reference implementation of the astronomically anchored 40Ar/39Ar framework.

Written directly from the published equations and deliberately independent of
arar_math.py, so that the toolchain can be checked against something that does
not share its code.

    Renne et al. (1994) Geology 22, eq. 1-3        age, R-value, back-solve
    Renne et al. (1998) Chem. Geol. 145, eq. 5-6   chain form and propagation
    Kuiper et al. (2008) Science 320, SOM          explicit partial derivatives
    Phillips et al. (2022) Chem. Geol. 597         R(FCs/A1Ts), recommended value
    Rivera et al. (2011) EPSL 311                  A1 tephra age 6.943 +- 0.005 Ma

Chain for an unknown irradiated with Fish Canyon sanidine:

    R_u/A1 = (F_u / F_FCs) * R_FCs/A1
    t      = ln(1 + R_u/A1 * (exp(lam t_A1) - 1)) / lam

Every input is independently measured: two isotope ratios from the user's own
irradiation (F_u, F_FCs), one from Phillips (R_FCs/A1), one orbital age (t_A1),
and lambda.  All sigmas below are 1-sigma; all ages in years unless named _ma.
"""

import math

MIN_LAMBDA, MIN_LAMBDA_1S = 5.463e-10, 0.054e-10          # Min et al. (2000)

# (t_A1 Ma, 1s Ma, R_FCs/A1, 1s).  R is the value consistent with each paper's
# published FCs age at the Min lambda (Phillips print 4.0819, which rounds to
# 28.176473 Ma; the published FCs age is 28.176).
ANCHORS = {
    'Phillips et al. (2022)': (6.943, 0.0025, None, 0.0014 / 2.0, 28.176),
    'Rivera et al. (2011)':   (6.943, 0.0025, None, 0.0026 / 2.0, 28.172),
    'Kuiper et al. (2008)':   (6.500, 0.005,  None, 0.0009,       28.201),
}


def _consistent_R(t_a1_ma, t_fc_ma, lam=MIN_LAMBDA):
    return math.expm1(lam * t_fc_ma * 1e6) / math.expm1(lam * t_a1_ma * 1e6)


def anchor(name):
    t_a1, s_a1, _, s_r, t_fc = ANCHORS[name]
    return t_a1, s_a1, _consistent_R(t_a1, t_fc), s_r


def ingrowth(anc, lam):
    """exp(lam t_FCs) - 1 for the standard, from the chain: R_FCs/A1 (e^(lam t_A1) - 1)."""
    t_a1, _, r_cal, _ = anc
    return r_cal * math.expm1(lam * t_a1 * 1e6)


def age(R_u_fcs, anc, lam=MIN_LAMBDA):
    """Age (yr) of an unknown with R_u/FCs = F_u/F_FCs."""
    return math.log1p(R_u_fcs * ingrowth(anc, lam)) / lam


def partials(t, anc, lam=MIN_LAMBDA):
    """(dt/dlnR, dt/dt_A1, dt/dlam) at age t (yr), for the chain in R_u/A1.

    dt/dlnR is the same for a fractional change in ANY ratio in the product
    (F_u, F_FCs, R_FCs/A1) and for J.
    """
    t_a1 = anc[0] * 1e6
    E = math.exp(lam * t_a1)
    X = math.exp(lam * t)
    P = (X - 1.0) / (E - 1.0)                                   # R_u/A1
    d_lnR = (1.0 - math.exp(-lam * t)) / lam
    d_ta1 = P * E / X
    d_lam = -t / lam + P * t_a1 * E / (lam * X)
    return d_lnR, d_ta1, d_lam


def tiers(t, rel_ana, rel_flux, anc, lam=MIN_LAMBDA, s_lam=MIN_LAMBDA_1S):
    """1-sigma (ana, int, ext) in years, plus the external components.

    rel_ana  : sigma(F_u)/F_u         (or the relative SEM of a mean ratio)
    rel_flux : sigma(F_FCs)/F_FCs     (= sigma_J / J)
    """
    d_lnR, d_ta1, d_lam = partials(t, anc, lam)
    t_a1, s_a1, r_cal, s_r = anc
    ana = d_lnR * rel_ana
    flux = d_lnR * rel_flux
    comps = dict(R_cal=d_lnR * s_r / r_cal,
                 t_A1=d_ta1 * s_a1 * 1e6,
                 lam=d_lam * s_lam)
    intr = math.hypot(ana, flux)
    ext = math.sqrt(intr**2 + sum(c * c for c in comps.values()))
    return ana, intr, ext, comps, flux


def combine_shared(ages_yr, ana, flux_same_position, comps_list):
    """Analytical-weight mean with exact shared-term propagation (1 sigma, yr).

    ana                : per-age analytical sigma (independent)
    flux_same_position : per-age fluence sigma, all ages in ONE position (shared)
    comps_list         : per-age dict of signed systematic components (shared)
    """
    w = [1.0 / a**2 for a in ana]
    W = sum(w)
    wh = [x / W for x in w]
    mean = sum(h * t for h, t in zip(wh, ages_yr))
    v_ana = sum((h * a)**2 for h, a in zip(wh, ana))
    v_flux = sum(h * f for h, f in zip(wh, flux_same_position))**2
    v_sys = sum(sum(h * c[k] for h, c in zip(wh, comps_list))**2
                for k in comps_list[0])
    return mean, math.sqrt(v_ana), math.sqrt(v_ana + v_flux), \
        math.sqrt(v_ana + v_flux + v_sys)
