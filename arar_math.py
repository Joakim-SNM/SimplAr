"""Shared math for the SimplAr Ar-Ar tools.

Everything the SimplAr tools have in common lives here: weighted means and
their covariance-aware variants, J-factor error propagation, York
regression, and the physical constants / calibration presets.

Main references:
  Koppers (2002), Computers & Geosciences 28, ArArCALC error conventions.
  Wendt & Carl (1991), Chem. Geol. 86, MSWD confidence limits.
  Vermeesch (2015), GCA 171, covariance-aware weighted means (eqs. 91-94).
  York (1969), EPSL 5, error-weighted regression.

All uncertainties passed in and returned are 1-sigma unless a name says
otherwise. The 1s/2s choice belongs in the GUIs, at display time.
"""

import math

import numpy as np

from scipy.optimize import minimize_scalar
from scipy.stats import chi2 as _chi2, t as _t


def chi2_cdf(x, df):
    return float(_chi2.cdf(max(float(x), 0.0), float(df)))


def t_crit(dof):
    """Two-tailed 95% Student-t critical value."""
    return float(_t.ppf(0.975, max(int(dof), 1)))


# --------------------------------------------------------------------------
# Physical constants and calibration presets.
#
# Defaults follow Phillips et al. (2022), paired with the Min et al. (2000)
# decay constant it is anchored to (see DEFAULT_FC_NAME below). Preset errors
# are 1-sigma; values published at 2-sigma (Kuiper 2008, Rivera 2011,
# Phillips 2022) have been halved here.
#
# external_flag on the FC presets: True means the calibration itself fixes how
# lambda enters (an orbital anchor, or a joint fit with lambda), so sigma_lambda
# is not added at full leverage.  How it does enter is decided in
# systematic_components: on the anchored chain for Kuiper, Rivera and Phillips,
# with the joint covariance for Carter.
# --------------------------------------------------------------------------

# (name, lambda yr^-1, 1s error)
LAMBDA_PRESETS = [
    ('Steiger & Jäger (1977)', 5.543e-10,  0.0),
    ('Min et al. (2000)',      5.463e-10,  0.054e-10),
    ('Renne et al. (2010)',    5.5492e-10, 0.0093e-10),
    ('Renne et al. (2011)',    5.5305e-10, 0.0132e-10),
    ('Carter et al. (2025)',   5.5042e-10, 0.0054e-10),
]

# (name, FCs age Ma, 1s error Ma, external_flag)
#
# The astronomically anchored ages (Kuiper 2008, Rivera 2011, Phillips 2022)
# carry external_flag=True.  Their ages and errors are the published values
# (2-sigma halved) and identify the preset; the external uncertainty of an
# age dated against them is not propagated from the FCs age and its error but
# from the anchor (FC_ANCHORS: R_FCs/A, t_A) and lambda on the two-step chain.
# Phillips et al. (2022): 28.176 +- 0.023 Ma (2s), A1 route.
FC_PRESETS = [
    ('Renne et al. (1998)  [S&J λ]', 28.02,  0.14,   False),
    ('Kuiper et al. (2008)',         28.201, 0.023,  True),
    ('Renne et al. (2010)',          28.305, 0.036,  False),
    ('Renne et al. (2011)',          28.294, 0.036,  False),
    ('Rivera et al. (2011)',         28.172, 0.014,  True),
    ('Phillips et al. (2022)',       28.176, 0.0115, True),
    ('Carter et al. (2025)',         28.183, 0.017,  True),
    ('Historical S&J — 27.84 Ma',    27.84,  0.0,    False),
]

# The λ each FC calibration is anchored to.  The astronomically tuned ages
# were all derived with the Min et al. (2000) decay constant and should be
# used together with it; the joint statistical calibrations (Renne 2010/11,
# Carter 2025) come with their own λ.  Renne (1998) and the historical value
# predate Min and pair with Steiger & Jäger.
FC_PAIRED_LAMBDA = {
    'Renne et al. (1998)  [S&J λ]': 'Steiger & Jäger (1977)',
    'Kuiper et al. (2008)':         'Min et al. (2000)',
    'Renne et al. (2010)':          'Renne et al. (2010)',
    'Renne et al. (2011)':          'Renne et al. (2011)',
    'Rivera et al. (2011)':         'Min et al. (2000)',
    'Phillips et al. (2022)':       'Min et al. (2000)',
    'Carter et al. (2025)':         'Carter et al. (2025)',
    'Historical S&J — 27.84 Ma':    'Steiger & Jäger (1977)',
}


# --------------------------------------------------------------------------
# Program-wide default calibration.
#
# Every tool opens on this one, and the four scalar defaults below are read
# out of the preset tables rather than typed again, so the monitor age, its
# error and the decay constant can never drift apart from the named
# calibration.  To move the whole suite onto a different calibration, change
# DEFAULT_FC_NAME alone; the anchored lambda follows from FC_PAIRED_LAMBDA.
# --------------------------------------------------------------------------

DEFAULT_FC_NAME     = 'Phillips et al. (2022)'
DEFAULT_LAMBDA_NAME = FC_PAIRED_LAMBDA[DEFAULT_FC_NAME]

DEFAULT_FC_INDEX     = next(i for i, r in enumerate(FC_PRESETS)
                            if r[0] == DEFAULT_FC_NAME)
DEFAULT_LAMBDA_INDEX = next(i for i, r in enumerate(LAMBDA_PRESETS)
                            if r[0] == DEFAULT_LAMBDA_NAME)

_, STD_AGE_DEFAULT, STD_AGE_ERR, DEFAULT_FC_EXTERNAL = FC_PRESETS[DEFAULT_FC_INDEX]
_, LAMBDA_DEFAULT, LAMBDA_ERR = LAMBDA_PRESETS[DEFAULT_LAMBDA_INDEX]

# 'external' when the monitor age was astronomically anchored: its quoted error
# already carries the lambda contribution, so sigma_lambda must not be added on
# top.  Used as the default stdAgeErrType wherever a fresh cfg is built.
DEFAULT_STD_AGE_ERR_TYPE = 'external' if DEFAULT_FC_EXTERNAL else 'internal'


def paired_lambda_index(fc_name):
    """Index into LAMBDA_PRESETS of the λ that fc_name is anchored to,
    or None for unknown / custom entries.  Tolerates labels with an age
    suffix ('Kuiper et al. (2008) — 28.201 Ma')."""
    name = str(fc_name).strip()
    lam_name = FC_PAIRED_LAMBDA.get(name)
    if lam_name is None:
        for k, v in FC_PAIRED_LAMBDA.items():
            stem = k.split('  ')[0].split(' — ')[0]
            if name.startswith(stem):
                lam_name = v
                break
    if lam_name is None:
        return None
    for i, (n, _, _) in enumerate(LAMBDA_PRESETS):
        if n == lam_name:
            return i
    return None


def paired_lambda(fc_name):
    """(name, lambda, 1s error) of the λ preset paired with fc_name, or None."""
    i = paired_lambda_index(fc_name)
    return LAMBDA_PRESETS[i] if i is not None else None


def calibration_for(fc_age_ma, lam, fc_err_ma=None):
    """cfg calibration keys (stdAgeMa, stdAgeErrMa, stdAgeErrType, lambdaYr,
    lambdaErrYr) for a monitor age and total decay constant read from a data
    file.  A pair matching a preset (FC age within 0.5 ka, its anchored λ
    within 1e-4 relative) takes that preset's values and errors; any other
    pair is used as given, with fc_err_ma and the error of a λ preset of the
    same value when there is one."""
    for name, age, err, ext in FC_PRESETS:
        pl = paired_lambda(name)
        if pl is not None and abs(fc_age_ma - age) < 5e-4 and abs(lam - pl[1]) <= 1e-4 * pl[1]:
            return dict(stdAgeMa=age, stdAgeErrMa=err,
                        stdAgeErrType='external' if ext else 'internal',
                        lambdaYr=pl[1], lambdaErrYr=pl[2])
    lam_err = next((e for _, l, e in LAMBDA_PRESETS if abs(lam - l) <= 1e-4 * l),
                   LAMBDA_ERR)
    return dict(stdAgeMa=float(fc_age_ma),
                stdAgeErrMa=float(fc_err_ma) if fc_err_ma else STD_AGE_ERR,
                stdAgeErrType='internal', lambdaYr=float(lam), lambdaErrYr=lam_err)


def file_age_calibration(cfg, steps):
    """Calibration keys that label stored step ages with the calibration they
    were calculated on, or None when cfg already does.

    Data-converter files carry the source file's monitor age and total decay
    constant in cfg['constants'] (file_std_age, file_lambda_total), and their
    step ages are ln(1 + J·R)/λ with that λ.  Files from older converter
    versions were labelled with the tool default calibration instead.  When
    cfg names a different λ and every step age agrees with the file's λ (to
    1e-4, not with cfg's), the ages were never recast and cfg is a mislabel:
    the file's calibration is returned (calibration_for).  Ages recast by
    Calibration… agree with cfg's λ and are left alone.  Files whose λ equals
    cfg's cannot be told apart this way and are left alone too."""
    c = cfg.get('constants') or {}
    try:
        t_f = float(c.get('file_std_age') or 0.0)
        l_f = float(c.get('file_lambda_total') or 0.0)
        l_c = float(cfg.get('lambdaYr') or 0.0)
    except (TypeError, ValueError):
        return None
    if t_f <= 0 or l_f <= 0 or (l_c > 0 and abs(l_c - l_f) <= 1e-6 * l_f):
        return None
    n = 0
    for s in steps:
        try:
            R, J, a = (float(s.get(k) or 0.0) for k in ('r4039', 'j_val', 'age'))
        except (TypeError, ValueError):
            continue
        if R > 0 and J > 0 and a > 0:
            n += 1
            if abs(math.log1p(J * R) / l_f / 1e6 - a) > 1e-4 * a:
                return None
    if n == 0:
        return None
    return calibration_for(t_f, l_f, c.get('file_std_age_err'))


# FC ages set by orbital tuning.  With the preset's own age and error they are
# propagated on the anchored chain (FC_ANCHORS); with an edited error, on the
# single-step R-value form.  Either way λ nearly cancels.
ASTRONOMICAL_FC_NAMES = {
    'Kuiper et al. (2008)',
    'Rivera et al. (2011)',
    'Phillips et al. (2022)',
}


# The orbital anchor each astronomically calibrated FC age was back-solved from,
# via Renne et al. (1994) eq. 3.  Fields: (t_astro Ma, 1s Ma, R_FCs/astro, 1s R).
#
# Why an anchor: t_FCs is not an independent input: it is the output of an
# equation containing lambda, so sigma_lambda and sigma(t_FCs) are correlated and
# Renne et al. (1998) eq. 6 is not valid on (t_FCs, lambda) -- they say so on
# p. 120.  Anchoring on t_astro instead makes every input independently measured
# (two ratios and a cycle count), which is the n=2 chain eq. 5/6 were written for.
#
# Only calibrations whose anchor is published verbatim belong here.  Kuiper's
# values are from the SOM section "Age and error equations for astronomically
# calibrated standard"; with them, external_error reproduces the paper's own
# 0.023 / 0.044 / 0.087 Ma at sigma_t_astro = 5 / 10 / 20 kyr (see tests).
# Tuffs and sigmas differ between calibrations and are easy to
# conflate.  Kuiper (2008) is anchored on the Messadit/Melilla tephras, Morocco;
# Rivera (2011) and Phillips (2022) on the A1 tephra of the Faneromeni section,
# Crete.  Kuiper's SOM inputs are 1s (its 6.500 Ma is a composite reference for
# the Melilla suite, not a single tuff); Phillips and Rivera publish at 2s.
# Everything below is stored at 1s, per this module's convention.
#
# Each entry reproduces its own paper's published FCs age and both uncertainty
# tiers -- see tests/test_min_external.py.
# R_FCs/astro is stored as the value consistent with each paper's published FCs
# age, not the value they print. The two differ in the 5th significant figure
# (Phillips print 4.0819, which implies 28.176473 Ma against their published
# 28.176) purely from rounding. Using the printed R would shift every age by
# ~0.9 kyr at 55 Ma and put the age on a different FCs age from the one the
# preset advertises. The relative sigma on R -- the only thing the external
# uses it for -- changes by 0.002%, i.e. not at all.
#
# The consistent R is computed below from each preset's FCs age at the Min et al.
# (2000) lambda the three calibrations were derived with, so each anchor returns
# its preset FCs age to machine precision. The value each paper prints is kept
# alongside for reference.
#                          t_anchor  1s     R printed  1s R      anchor tephra
_FC_ANCHOR_INPUTS = {
    # Melilla composite, Kuiper et al. (2008) SOM (SOM inputs are already 1s)
    'Kuiper et al. (2008)':   (6.500, 0.005,  4.3664, 0.00090),
    # A1 tephra, Faneromeni, Crete; t_anchor revised by Rivera et al. (2011).
    # Rivera print R = 4.0813 +- 0.0026 (2s).
    'Rivera et al. (2011)':   (6.943, 0.0025, 4.0813, 0.00130),
    # Phillips et al. (2022) recommended inter-laboratory value, A1 route:
    # R = 4.0819 +- 0.0014 (2s), t(A1) = 6.943 +- 0.005 Ma (2s).
    'Phillips et al. (2022)': (6.943, 0.0025, 4.0819, 0.00070),
}
_ANCHOR_NATIVE_LAMBDA = 5.463e-10     # Min et al. (2000), used by all three


def _consistent_anchor_R(name):
    t_a = _FC_ANCHOR_INPUTS[name][0]
    t_fc = next(a for n, a, e, x in FC_PRESETS if n == name)
    return (math.expm1(_ANCHOR_NATIVE_LAMBDA * t_fc * 1e6)
            / math.expm1(_ANCHOR_NATIVE_LAMBDA * t_a * 1e6))


# (t_anchor Ma, 1s Ma, R_FCs/anchor consistent with the preset, 1s R)
FC_ANCHORS = {name: (v[0], v[1], _consistent_anchor_R(name), v[3])
              for name, v in _FC_ANCHOR_INPUTS.items()}


def calibration_class(fc_name):
    """'astronomical' (orbital-tuned age, R-value external applies),
    'joint' (age and λ co-calibrated, e.g. Carter 2025, use its own coupled
    error), or 'other'. Tolerates an age suffix on the name."""
    name = str(fc_name).strip()
    for a in ASTRONOMICAL_FC_NAMES:
        if name.startswith(a):
            return 'astronomical'
    if name.startswith('Carter'):
        return 'joint'
    return 'other'


# --------------------------------------------------------------------------
# Anchored calibration framework
#
# For an astronomically calibrated monitor the age of an unknown is computed
# on the two-step chain of Renne et al. (1998) eq. 5,
#
#     t = (1/λ) ln[1 + R_u/FCs · R_FCs/A · (e^{λ t_A} − 1)]
#
# with every input measured independently: the unknown and monitor ratios from
# the user's own irradiation, R_FCs/A from the calibration paper, and t_A, the
# orbital age of the anchor tephra.  The Fish Canyon age never enters as an
# input, so its λ dependence cannot correlate with the λ term (Renne et al.
# 1998, p. 120).  Everything in this module that needs "e^{λ t_FCs} − 1" gets it
# from fc_ingrowth(), and every external uncertainty from
# systematic_components(), so the tools cannot drift apart.
# --------------------------------------------------------------------------

def anchor_for(t_fc_ma, sigma_fc_ma=None):
    """Orbital anchor (t_A, σ_A, R_FCs/A, σ_R; 1σ) for an FCs age, or None.

    Matches an anchored preset by its FCs age, and by its 1σ error as well when
    one is given, so a user-edited error falls back to the single-step route
    rather than being silently replaced by the preset's.
    """
    try:
        t = float(t_fc_ma)
    except (TypeError, ValueError):
        return None
    for name, age, err, _ in FC_PRESETS:
        if name not in FC_ANCHORS or abs(age - t) >= 5e-4:
            continue
        if sigma_fc_ma is not None and abs(float(sigma_fc_ma) - err) >= 5e-5:
            continue
        return FC_ANCHORS[name]
    return None


def cfg_anchor(cfg):
    """Orbital anchor for a flux config, or None (see anchor_for).
    cfg['fcAnchor'], when present, overrides the preset lookup."""
    a = cfg.get('fcAnchor')
    if a:
        return tuple(float(x) for x in a)
    return anchor_for(cfg.get('stdAgeMa', STD_AGE_DEFAULT),
                      cfg.get('stdAgeErrMa') if 'stdAgeErrMa' in cfg
                      else (None if 'stdAgeMa' in cfg else STD_AGE_ERR))


def fc_ingrowth(t_fc_ma, lam, sigma_fc_ma=None):
    """e^{λ t_FCs} − 1, the monitor ingrowth factor that turns R into an age.

    For an anchored preset it is computed on the chain as
    R_FCs/A · (e^{λ t_A} − 1): identical to e^{λ t_FCs} − 1 at the Min et al.
    (2000) λ the calibration was derived with, and the correct value at any
    other λ, where the FCs age is not an independent constant.
    """
    anc = anchor_for(t_fc_ma, sigma_fc_ma)
    if anc is not None:
        return anc[2] * math.expm1(lam * anc[0] * 1e6)
    return math.expm1(lam * float(t_fc_ma) * 1e6)


def cfg_ingrowth(cfg, lam=None):
    """fc_ingrowth for a flux config."""
    if lam is None:
        lam = cfg.get('lambdaYr', LAMBDA_DEFAULT)
    anc = cfg_anchor(cfg)
    if anc is not None:
        return anc[2] * math.expm1(lam * anc[0] * 1e6)
    return math.expm1(lam * cfg.get('stdAgeMa', STD_AGE_DEFAULT) * 1e6)


def effective_fc_age(cfg):
    """The FCs age (Ma) implied by the config's calibration and λ."""
    lam = cfg.get('lambdaYr', LAMBDA_DEFAULT)
    return math.log1p(cfg_ingrowth(cfg, lam)) / lam / 1e6


def dt_dlnR(age_ma, lam=LAMBDA_DEFAULT):
    """∂t/∂ln R in Ma: the age change per fractional change in any ratio in the
    chain (sample, monitor, calibration) or in J.  Exact form (1 − e^{−λt})/λ;
    age × σ_R/R overstates it by ~1.5% at 55 Ma."""
    return -math.expm1(-lam * age_ma * 1e6) / lam / 1e6


def anchored_partials(age_ma, anchor, lam=LAMBDA_DEFAULT):
    """(∂t/∂ln R [Ma], ∂t/∂t_A [dimensionless], ∂t/∂λ [yr²]) on the chain.
    Kuiper et al. (2008) SOM, with the product R_u/A = R_u/FCs · R_FCs/A."""
    t = age_ma * 1e6
    t_a = anchor[0] * 1e6
    E = math.exp(lam * t_a)
    X = math.exp(lam * t)
    P = math.expm1(lam * t) / math.expm1(lam * t_a)          # R_u/A
    return (dt_dlnR(age_ma, lam),
            P * E / X,
            -t / lam + P * t_a * E / (lam * X))


def fluence_sigma(ages, cfg):
    """1σ (Ma) contributed by the monitor / J precision: ∂t/∂ln J · σ_J/J."""
    ages = np.asarray(ages, float)
    jf = cfg_j_err_frac(cfg)
    lam = cfg.get('lambdaYr', LAMBDA_DEFAULT)
    return np.array([dt_dlnR(a, lam) * jf for a in ages.ravel()]).reshape(ages.shape)


def internal_sigma_array(ages, sigma_ana, cfg):
    """Internal tier (Ma, 1σ): analytical ⊕ fluence, with the exact partial."""
    return np.sqrt(np.asarray(sigma_ana, float)**2 + fluence_sigma(ages, cfg)**2)


# --------------------------------------------------------------------------
# Weighted means
# --------------------------------------------------------------------------

def weighted_mean(ages, err1s):
    """Inverse-variance weighted mean.

    Returns a dict with mean, sem (1s error on the mean), mswd, chi2, prob
    (chi-squared p-value), n, overdispersed (Wendt & Carl 1991 threshold)
    and sem_scaled (sem * sqrt(MSWD) when overdispersed, else sem).
    Returns None when no point has a positive error.
    """
    a = np.asarray(ages,  dtype=float)
    e = np.asarray(err1s, dtype=float)
    w = np.where(e > 0, 1.0 / e**2, 0.0)
    W = w.sum()
    if W == 0:
        return None
    m = (w * a).sum() / W
    sem = 1.0 / np.sqrt(W)
    n = len(a)
    if n > 1:
        chi2 = float((w * (a - m)**2).sum())
        mswd = chi2 / (n - 1)
        prob = float(1.0 - chi2_cdf(chi2, n - 1))
        over = mswd > mswd_critical(n)   # Wendt & Carl 1991, n-dependent
    else:
        chi2, mswd, prob, over = 0.0, 0.0, 1.0, False
    scaled = sem * math.sqrt(mswd) if over else sem
    return dict(mean=m, sem=sem, mswd=mswd, chi2=chi2, prob=prob, n=n,
                overdispersed=over, sem_scaled=scaled)


def hole_candidates(disk, hole):
    """The labels a sample's hole_id could have in a .jflux, in order: the hole
    itself, the hole with the sample's own disk prefix stripped ('E17' -> '17'),
    then its trailing digit run ('9E17-25' -> '25', 'E17' -> '17')."""
    if hole is None:
        return []
    hole = str(hole)
    disk_s = str(disk) if disk else ''
    cands = [hole]
    if disk_s and hole.upper().startswith(disk_s.upper()):
        cands.append(hole[len(disk_s):])
    tail = ''
    for ch in reversed(hole):
        if ch.isdigit():
            tail = ch + tail
        else:
            break
    if tail:
        cands.append(tail)
    seen = set()
    return [h for h in cands if h and not (h in seen or seen.add(h))]


def flux_hole_matches(disk_flux, disk, hole):
    """Every loaded disk-flux hole a sample's hole_id could be, as
    [((irradiation, disk_id), hole_index), ...], before any disambiguation
    (candidates from hole_candidates)."""
    if not disk_flux or hole is None:
        return []
    hole_cands = hole_candidates(disk, hole)
    matches = []
    for (kirr, kdisk), fx in disk_flux.items():
        for h in hole_cands:
            if h in fx['idx']:
                matches.append(((kirr, kdisk), fx['idx'][h]))
                break
    return matches


def match_flux_hole(disk_flux, disk, hole, irr=None, preferred=None):
    """The loaded disk-flux hole a sample (disk_id, hole_id, irradiation) was
    irradiated in: ((irradiation, disk_id), hole_index), or None.

    Used by the Plateau & isochron and Age distribution tools alike.  Matching is
    by HOLE; the disk name and irradiation only disambiguate when several loaded
    holes share the number.  A tie is resolved by the user's preferred disk (if
    one is set), then the same irradiation, then an exact disk-name match.  If
    that still leaves more than one (hole 17 on both 'E inner' and 'E outer'),
    it does not guess: None, and the user assigns the hole."""
    matches = flux_hole_matches(disk_flux, disk, hole)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if preferred is not None:
        for m in matches:
            if m[0] == tuple(preferred):
                return m
    if irr is not None:
        irr_m = [m for m in matches if m[0][0] is None or str(m[0][0]) == str(irr)]
        if len(irr_m) == 1:
            return irr_m[0]
        if irr_m:
            matches = irr_m
    if len(matches) == 1:
        return matches[0]
    disk_s = str(disk) if disk else ''
    exact = [m for m in matches if str(m[0][1]).upper() == disk_s.upper()]
    if len(exact) == 1:
        return exact[0]
    return None


# --------------------------------------------------------------------------
# Disk-flux registry: .jflux files, saved copies, and what a hole encodes.
#
# The registry maps (irradiation, disk_id) to one entry per disk:
#   idx {hole: index}, holes, J, J_sigma, J_corr, hole_xy, design,
#   R, R_sigma (newer files only), lamFlux, fcFlux, fcFluxErr (the λ / FCs age
#   the J values were computed with), fit (the fit_id of the .jflux, newer
#   files only) and xcorr {(irradiation, other disk): n_this × n_other
#   correlation} for the disks of a global fit through several disks.
# Every tool reads and writes it through the functions below.
# --------------------------------------------------------------------------

def disk_flux_entry(holes, J, J_sigma, J_corr, hole_xy=(), design='',
                    R=None, R_sigma=None, lam=None, fc=None, fc_err=None,
                    fit=None, xcorr=None):
    """One registry entry from plain lists."""
    holes = [str(h) for h in holes]
    entry = dict(idx={h: i for i, h in enumerate(holes)}, holes=holes,
                 J=np.asarray(J, float), J_sigma=np.asarray(J_sigma, float),
                 J_corr=np.asarray(J_corr, float),
                 hole_xy=np.asarray(hole_xy if hole_xy is not None else [], float),
                 design=design or '', lamFlux=lam, fcFlux=fc, fcFluxErr=fc_err)
    if R is not None:
        entry['R'] = np.asarray(R, float)
    if R_sigma is not None:
        entry['R_sigma'] = np.asarray(R_sigma, float)
    if fit:
        entry['fit'] = str(fit)
    entry['xcorr'] = {tuple(k): np.asarray(v, float) for k, v in (xcorr or {}).items()}
    return entry


def jflux_entries(cfg):
    """{(irradiation, disk_id): entry} from a parsed .jflux dict.
    Raises ValueError when it is not a jflux_config."""
    if cfg.get('type') != 'jflux_config':
        raise ValueError('not a jflux_config file.')
    irr = str(cfg.get('irradiation'))
    fit = cfg.get('fit_id')
    out = {}
    for d in cfg.get('disks', []):
        out[(irr, str(d['disk_id']))] = disk_flux_entry(
            d['holes'], d['J'], d['J_sigma'], d['J_corr'], d.get('hole_xy', []),
            d.get('disk_design', ''), d.get('R'), d.get('R_sigma'),
            cfg.get('lambdaYr'), cfg.get('stdAgeMa'), cfg.get('stdAgeErrMa'), fit)
    # Global fit through several disks: correlation between holes on
    # different disks, stored once per pair and kept both ways here.
    if fit:
        for c in cfg.get('cross_disk_corr', []):
            ka, kb = (irr, str(c['disk_a'])), (irr, str(c['disk_b']))
            if ka in out and kb in out:
                m = np.asarray(c['J_corr'], float)
                out[ka]['xcorr'][kb] = m
                out[kb]['xcorr'][ka] = m.T
    return out


def read_jflux(path):
    """{(irradiation, disk_id): entry} from a .jflux file (see jflux_entries)."""
    import json
    with open(path, encoding='utf-8') as fh:
        return jflux_entries(json.load(fh))


def serialize_disk_flux(disk_flux):
    """The registry as a JSON-safe dict keyed 'irr|disk' (the .ararstack /
    .arar format)."""
    out = {}
    for (irr, disk), fx in disk_flux.items():
        out[f'{irr}|{disk}'] = {
            'irr': irr, 'disk': disk,
            'idx': fx['idx'], 'holes': fx['holes'],
            'J': fx['J'].tolist(), 'J_sigma': fx['J_sigma'].tolist(),
            'J_corr': fx['J_corr'].tolist(),
            'R': (fx['R'].tolist() if fx.get('R') is not None else None),
            'R_sigma': (fx['R_sigma'].tolist() if fx.get('R_sigma') is not None else None),
            'hole_xy': np.asarray(fx.get('hole_xy', []), float).tolist(),
            'design': fx.get('design', ''),
            'lamFlux': fx.get('lamFlux'), 'fcFlux': fx.get('fcFlux'),
            'fcFluxErr': fx.get('fcFluxErr'),
            'fit': fx.get('fit'),
            'xcorr': serialize_xcorr(fx),
        }
    return out


def serialize_xcorr(fx):
    """A registry entry's cross-disk correlations as a JSON-safe list."""
    return [{'irr': k[0], 'disk': k[1], 'J_corr': np.asarray(m, float).tolist()}
            for k, m in (fx.get('xcorr') or {}).items()]


def _xcorr_from(data):
    return {(str(c.get('irr')), str(c.get('disk'))): c.get('J_corr', [])
            for c in (data or [])}


def deserialize_disk_flux(data):
    """{(irr, disk): entry} from a saved registry: the 'irr|disk' dict of
    serialize_disk_flux, or the list the Age distribution sessions write."""
    out = {}
    if isinstance(data, dict):
        items = [(str(fx.get('irr')), str(fx.get('disk')), fx) for fx in data.values()]
    else:
        items = [(str(fx.get('irradiation', '')), str(fx.get('disk_id', '')), fx)
                 for fx in (data or [])]
    for irr, disk, fx in items:
        out[(irr, disk)] = disk_flux_entry(
            fx.get('holes', []), fx.get('J', []), fx.get('J_sigma', []),
            fx.get('J_corr', []), fx.get('hole_xy', []), fx.get('design', ''),
            fx.get('R'), fx.get('R_sigma'), fx.get('lamFlux'), fx.get('fcFlux'),
            fx.get('fcFluxErr'), fx.get('fit'), _xcorr_from(fx.get('xcorr')))
    return out


def merge_disk_flux(target, new):
    """Add the disks of `new` that `target` does not have yet.  Returns the
    number added; disks already loaded are kept as they are."""
    n = 0
    for key, fx in new.items():
        if key not in target:
            target[key] = fx
            n += 1
    return n


def flux_hole_R(fx, i):
    """The monitor ratio R a disk-flux hole encodes: the R stored in the .jflux
    when present, else (e^{λ_f t_f} − 1)/J from the flux's own calibration.
    R is what the tools apply, so a flux computed on another calibration still
    gives the right age.  None when the flux carries no calibration or J ≤ 0."""
    R_arr = fx.get('R')
    if R_arr is not None:
        try:
            R = float(R_arr[i])
            if R > 0:
                return R
        except (IndexError, TypeError, ValueError):
            pass
    try:
        J = float(fx['J'][i])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    lam_f, fc_f = fx.get('lamFlux'), fx.get('fcFlux')
    if J > 0 and lam_f and fc_f:
        return fc_ingrowth(fc_f, lam_f, fx.get('fcFluxErr')) / J
    return None


def flux_hole_rel(fx, i):
    """Relative 1σ of a disk-flux hole (σ_R/R, equal to σ_J/J), or None."""
    try:
        if fx.get('R_sigma') is not None and fx.get('R') is not None:
            return float(fx['R_sigma'][i]) / float(fx['R'][i])
        return float(fx['J_sigma'][i]) / float(fx['J'][i])
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return None


def flux_hole_J(fx, i, cfg):
    """(J, σ_J) of a disk-flux hole for the calibration in cfg: the hole's R
    turned into J with cfg's monitor ingrowth, σ_J keeping its relative size.
    A flux without a stored calibration gives its raw J.  (None, None) for a
    non-positive J."""
    Jf = float(fx['J'][i])
    Jsf = float(fx['J_sigma'][i])
    if Jf <= 0:
        return None, None
    R = flux_hole_R(fx, i)
    if R and cfg.get('lambdaYr') and cfg.get('stdAgeMa'):
        J = cfg_ingrowth(cfg) / R
        if J > 0:
            return J, Jsf * (J / Jf)
    return Jf, Jsf


def position_ident(disk_id=None, hole_id=None, group=None, cfg=None):
    """Identity of an irradiation position for j_correlation when no loaded
    .jflux hole is known: the disk label, else the hole label, else a J group
    the user typed, else the J calibration itself (aliquots on the same J and
    σ_J were measured against the same monitor).  None when nothing is known;
    the caller then gives the point an identity of its own.

    Shared by the Plateau & isochron stack and the Age distribution tool, so the same
    aliquots fall in the same position groups in both."""
    for v in (disk_id, hole_id, group):
        if v not in (None, ''):
            return str(v)
    if cfg:
        J, rel = cfg_J(cfg)
        if J:
            return f'J={J:.10g}±{rel:.6g}'
    return None


def flux_pair_corr(disk_flux, ma, mb):
    """Correlation of the J error between two loaded disk-flux holes
    ma, mb = ((irradiation, disk), hole_index): J_corr within one disk; the
    cross-disk block when both disks come from the same global fit (same
    fit_id); otherwise 0, as disks fitted separately are independent."""
    fa = disk_flux[ma[0]]
    if ma[0] == mb[0]:
        return float(fa['J_corr'][ma[1], mb[1]])
    fb = disk_flux.get(mb[0], {})
    x = (fa.get('xcorr') or {}).get(mb[0])
    if x is not None and fa.get('fit') and fa.get('fit') == fb.get('fit'):
        return float(x[ma[1], mb[1]])
    return 0.0


def j_correlation(points, disk_flux, force=False):
    """N x N correlation of the J (flux) error between analyses.

    The one rule used by both the Plateau & isochron tool (aliquots) and the
    Age distribution tool (points / samples), so the same analyses get the same
    covariance in either.  ``points`` is a sequence of dicts:

      match  ((irradiation, disk), hole_index) of the loaded .jflux hole the
             point was irradiated in, or None when no loaded hole is known
      ident  hashable identity of its irradiation position (disk / hole label)
      irr    irradiation name, '' when unknown

    Two points correlate as follows (first rule that applies):
      force            -> 1
      both matched to the same loaded disk
                       -> J_corr[hole_i, hole_j] from the .jflux (1 for the
                          same hole; between 0 and 1 for neighbours; it can
                          be negative)
      both matched to different disks of one global fit (a 3D fit through
      several disks, same .jflux)
                       -> the cross-disk J_corr from the .jflux
      both matched, but to disks fitted separately (individual-disk fits,
      other files, or two irradiations)
                       -> 0: separate flux monitors, independent
      one matched, the other not
                       -> 0: the unmatched one has its J from elsewhere
                          (another lab, a typed J), independent of the disk
      neither matched: same ident, and the same irradiation (an unknown
      irradiation counts as the same)
                       -> 1
      anything else    -> 0
    """
    n = len(points)
    rho = np.eye(n)
    for a in range(n):
        pa = points[a]
        for b in range(a + 1, n):
            pb = points[b]
            if force:
                r = 1.0
            else:
                ma, mb = pa.get('match'), pb.get('match')
                if ma is not None and mb is not None:
                    r = flux_pair_corr(disk_flux, ma, mb)
                elif ma is not None or mb is not None:
                    r = 0.0
                else:
                    ia, ib = str(pa.get('irr') or ''), str(pb.get('irr') or '')
                    same_irr = (not ia) or (not ib) or ia == ib
                    r = 1.0 if (same_irr and pa.get('ident') == pb.get('ident')) else 0.0
            rho[a, b] = rho[b, a] = r
    return rho


def combine_ages(ages, sigma_ana, sigma_int=None, sigma_ext=None,
                 position_group=None, standard_group=None,
                 scale_overdispersion=True, od_mode='threshold', p_corr=None,
                 sys_components=None):
    """Weighted mean of N ages with a three-tier error budget.

    The covariance is Sigma_ij = a^2 d_ij + p_i p_j [same position]
    + s_i s_j [same standard], where a = sigma_ana,
    p = sqrt(sigma_int^2 - sigma_ana^2) and s = sqrt(sigma_ext^2 - sigma_int^2).
    The mean itself uses analytical weights, so it does not depend on the
    covariance; only the tier errors do.

    od_mode selects the overdispersion trigger: 'threshold' (Wendt & Carl
    1991 95% limit) or 'mswd_gt1'. When triggered and scale_overdispersion
    is set, the analytical tier is inflated by MSWD.

    p_corr, if given, is an NxN correlation matrix for the J-flux block and
    overrides position_group there, used when a loaded .jflux file provides
    the true cross-hole correlation.

    sys_components, if given, is the n × k matrix of signed calibration
    components from systematic_matrix.  The standard block is then propagated
    one shared parameter at a time, Σ_k (Σ_i ŵ_i c_ik)², which is exact when the
    ages differ; the lumped s_i s_j form is exact only for equal ages.
    """
    ages = np.asarray(ages, float)
    a = np.asarray(sigma_ana, float)
    n = len(ages)
    si = np.asarray(sigma_int, float) if sigma_int is not None else a.copy()
    se = np.asarray(sigma_ext, float) if sigma_ext is not None else si.copy()
    p = np.sqrt(np.maximum(0.0, si**2 - a**2))
    s = np.sqrt(np.maximum(0.0, se**2 - si**2))
    if position_group is None:
        position_group = list(range(n))
    if standard_group is None:
        standard_group = [0] * n
    w = 1.0 / a**2
    Sw = w.sum()
    wh = w / Sw
    wm = float((wh * ages).sum())
    if n > 1:
        chi2 = float((w * (ages - wm)**2).sum())
        mswd = chi2 / (n - 1)
        thr = mswd_critical(n)           # Wendt & Carl 1991, n-dependent
        over = (mswd > 1.0) if od_mode == 'mswd_gt1' else (mswd > thr)
    else:
        chi2, mswd, over = 0.0, 1.0, False
    prob = float(1.0 - chi2_cdf(chi2, n - 1)) if n > 1 else 1.0
    mfac = mswd if (scale_overdispersion and over) else 1.0
    var_ana = float((wh**2 * a**2).sum()) * mfac

    def _block(comp, groups):
        idx = {}
        for i, g in enumerate(groups):
            idx.setdefault(g, []).append(i)
        return sum((sum(wh[i] * comp[i] for i in m))**2 for m in idx.values())

    if p_corr is not None:
        pc = np.asarray(p_corr, float)
        wp = wh * p
        var_p = float(wp @ pc @ wp)
    else:
        var_p = _block(p, position_group)
    var_int = var_ana + var_p
    if sys_components is not None:
        C = np.asarray(sys_components, float).reshape(n, -1)
        var_s = sum(_block(C[:, k], standard_group) for k in range(C.shape[1]))
    else:
        var_s = _block(s, standard_group)
    var_ext = var_int + var_s
    return dict(wm=wm, n=n, mswd=mswd, chi2=chi2, prob=prob,
                overdispersed=over, mswd_scale=math.sqrt(mfac),
                sigma_ana=math.sqrt(var_ana), sigma_int=math.sqrt(var_int),
                sigma_ext=math.sqrt(var_ext))


def ml_overdispersion(t, a, V0=None, reml=False):
    """Overdispersion ω² ≥ 0 for t ~ N(μ·1, V0 + ω²·I), by maximum likelihood
    (Vermeesch 2015 eq. 94 / IsoplotR) or, with reml=True, restricted ML.

    V0 is the covariance the ages are already known to carry: the analytical
    variances on the diagonal, plus the J block between positions and the
    calibration block.  Fitting ω on V0 (not on the analytical variances alone)
    stops scatter that the J or calibration terms already explain, ages from
    holes or irradiations with independent J, from being counted a second time
    as ω.  With V0 = diag(a²) this is the original analytical-only fit.
    """
    t = np.asarray(t, float)
    n = len(t)
    if n < 2:
        return 0.0
    V0 = (np.diag(np.asarray(a, float)**2) if V0 is None
          else np.asarray(V0, float))
    one = np.ones(n)
    eye = np.eye(n)

    def neg_ll(o2):
        V = V0 + o2 * eye
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return np.inf
        Vi1 = np.linalg.solve(V, one)
        s = float(one @ Vi1)
        r = t - float(Vi1 @ t) / s
        val = 2.0 * np.log(np.diag(L)).sum() + float(r @ np.linalg.solve(V, r))
        if reml:
            val += math.log(s)
        return 0.5 * val

    hi = (t.var(ddof=1) + float(np.max(np.diag(V0)))) * 20.0
    x = float(minimize_scalar(neg_ll, bounds=(0.0, hi), method='bounded',
                              options={'xatol': hi * 1e-12}).x)
    # the boundary ω = 0 wins when it is at least as likely
    return max(0.0, x) if neg_ll(x) < neg_ll(0.0) else 0.0


def combine_ages_vermeesch(ages, sigma_ana, sigma_int=None, sigma_ext=None,
                           position_group=None, standard_group=None,
                           p_corr=None, random_effects=True,
                           return_matrices=False, sys_components=None,
                           reml=False):
    """Generalised-least-squares weighted mean, Vermeesch (2015) eqs. 91-94.

    Same inputs and return keys as combine_ages.  The mean is the GLS estimate
    on the full covariance V = A + P + S (eq. 92); overdispersion is an additive
    random effect ω² fitted by ML (or REML) on that same covariance, so J and
    calibration scatter is not counted again as ω.  The three tiers are the
    variance of this mean under A, A + P and A + P + S, wᵀAw, wᵀ(A+P)w and
    wᵀVw with w the GLS weights, so they describe the reported age.

    With return_matrices=True the result also carries the A, P, S blocks and V.
    sys_components (n × k, from systematic_matrix) builds the standard block as
    S_ij = Σ_k c_ik c_jk within a standard group.
    """
    t = np.asarray(ages, float)
    a = np.asarray(sigma_ana, float)
    n = len(t)
    si = np.asarray(sigma_int, float) if sigma_int is not None else a.copy()
    se = np.asarray(sigma_ext, float) if sigma_ext is not None else si.copy()
    p = np.sqrt(np.maximum(0.0, si**2 - a**2))
    s = np.sqrt(np.maximum(0.0, se**2 - si**2))
    if position_group is None:
        position_group = list(range(n))
    if standard_group is None:
        standard_group = [0] * n

    w0 = 1.0 / a**2
    mu_o = float((w0 * t).sum() / w0.sum())
    chi2 = float((w0 * (t - mu_o)**2).sum()) if n > 1 else 0.0
    mswd = chi2 / (n - 1) if n > 1 else 1.0
    prob = float(1.0 - chi2_cdf(chi2, n - 1)) if n > 1 else 1.0

    if p_corr is not None:
        P = np.outer(p, p) * np.asarray(p_corr, float)
    else:
        P = np.array([[p[i] * p[j] if position_group[i] == position_group[j] else 0.0
                       for j in range(n)] for i in range(n)])
    same_std = np.array([[standard_group[i] == standard_group[j]
                          for j in range(n)] for i in range(n)], float)
    if sys_components is not None:
        C = np.asarray(sys_components, float).reshape(n, -1)
        S = (C @ C.T) * same_std
    else:
        S = np.outer(s, s) * same_std

    A0 = np.diag(a**2)
    om2 = (ml_overdispersion(t, a, V0=A0 + P + S, reml=reml)
           if random_effects else 0.0)
    A = A0 + om2 * np.eye(n)
    V = A + P + S
    one = np.ones(n)
    w = np.linalg.solve(V, one)
    w = w / w.sum()                                   # GLS weights (eq. 92)
    wm = float(w @ t)
    var_ana = float(w @ A @ w)
    var_int = float(w @ (A + P) @ w)
    var_ext = float(w @ V @ w)

    result = dict(wm=wm, n=n, mswd=mswd, chi2=chi2, prob=prob,
                  overdispersed=(om2 > 0.0), overdispersion=math.sqrt(om2),
                  mswd_scale=1.0, weights=w,
                  sigma_ana=math.sqrt(var_ana), sigma_int=math.sqrt(var_int),
                  sigma_ext=math.sqrt(var_ext))
    if return_matrices:
        result['matrices'] = dict(A=A, P=P, S=S, V=V, omega2=om2,
                                  p=p, s=s, sigma_ana=a, sigma_int=si,
                                  sigma_ext=se)
    return result


# Combining methods, as stored in session files.  Older files used the names
# on the right; method_key() reads both.
METHOD_WM = 'wm'          # WM + √MSWD   (combine_ages)
METHOD_GLS = 'gls_ml'     # GLS + ML     (combine_ages_vermeesch)
_OLD_METHOD_KEYS = {'schaen': METHOD_WM, 'koppers': METHOD_WM, 'vermeesch': METHOD_GLS}


def method_key(name):
    """'wm' or 'gls_ml' for a stored method name, old or new; 'wm' if unknown."""
    name = _OLD_METHOD_KEYS.get(name, name)
    return name if name in (METHOD_WM, METHOD_GLS) else METHOD_WM


def combine(method, ages, sigma_ana, sigma_int=None, sigma_ext=None, **kw):
    """combine_ages (METHOD_WM: analytical weights + √MSWD) or
    combine_ages_vermeesch (METHOD_GLS: GLS + ML random effects)."""
    if method_key(method) == METHOD_GLS:
        kw.pop('od_mode', None)
        kw.pop('scale_overdispersion', None)
        return combine_ages_vermeesch(ages, sigma_ana, sigma_int, sigma_ext, **kw)
    return combine_ages(ages, sigma_ana, sigma_int, sigma_ext, **kw)


def mswd_critical(n):
    """Upper 95% confidence limit on the MSWD for ``n`` data points
    (Wendt & Carl 1991): 1 + 2*sqrt(2/(n-1)), where n-1 is the degrees of
    freedom. Scatter with MSWD above this limit is 'significant' overdispersion
    (only then should the uncertainty be magnified by sqrt(MSWD)); below it the
    scatter is consistent with the stated analytical errors at that n. Returns
    +inf for n < 2 so the criterion can never trigger. This is the single,
    n-dependent overdispersion trigger shared by every tool."""
    if n is None or n < 2:
        return float('inf')
    return 1.0 + 2.0 * math.sqrt(2.0 / (n - 1))


def mswd_classify(mswd, n):
    """'overdispersed' / 'underdispersed' / 'acceptable' at the Wendt & Carl
    (1991) 95% limits, 1 +/- 2*sqrt(2/(n-1)). None when n < 2."""
    if mswd is None or n is None or n < 2:
        return None
    hi = mswd_critical(n)
    lo = max(0.0, 1.0 - 2.0 * math.sqrt(2.0 / (n - 1)))
    if mswd > hi:
        return 'overdispersed'
    if mswd < lo:
        return 'underdispersed'
    return 'acceptable'


# --------------------------------------------------------------------------
# J factor and age conversion
# --------------------------------------------------------------------------

def cfg_j_err_frac(cfg):
    """Relative 1s J error for the internal tier: flux measurement only,
    no monitor-age or lambda systematics."""
    mode = cfg.get('ageFluxMode', 'imported')
    if mode == 'ratio_R_std':
        r_err = cfg.get('rStdErrPct', 0.10)
        r_std = cfg.get('rStd', 70.0)
        if cfg.get('rStdErrMode', 'pct') == 'pct':
            return r_err / 100.0
        return r_err / r_std if r_std > 0 else 0.0
    if mode == 'ratio_J':
        if cfg.get('jErrMode', 'pct') == 'pct':
            return cfg.get('jErrPct', 0.0) / 100.0
        j_val = cfg.get('jVal', 0.005)
        return cfg.get('jErrPct', 0.0) / j_val if j_val > 0 else 0.0
    if mode == 'ratio_R_FCs':
        # only add the R_FCs% error when the pasted sigmas are analytical-only;
        # a pasted internal tier already contains it
        if cfg.get('pasteTier', 'analytical') == 'internal':
            return 0.0
        return cfg.get('rfcsErrPct', 0.0) / 100.0
    return 0.0


def cfg_J(cfg):
    """(J, relative 1σ J error) that a flux config turns ⁴⁰Ar*/³⁹Ar_K into an
    age with, or (None, 0.0) when it has none (imported ages, R_unk/R_FCs).

    ratio_R_std: J = (e^{λ t_FCs} − 1)/R_std, the ingrowth from cfg_ingrowth
    (anchored chain where the preset has one); ratio_J: the typed J.  The
    error is the flux measurement only (cfg_j_err_frac); an applied disk-flux
    hole (jForce) carries its σ_J in percent.  Used wherever a J is needed, so
    a stored jVal that went stale after a recalibration is never read in place
    of the J the ages are computed with."""
    mode = cfg.get('ageFluxMode', 'imported')
    if mode == 'ratio_R_std':
        r_std = float(cfg.get('rStd', 0.0) or 0.0)
        J = cfg_ingrowth(cfg) / r_std if r_std > 0 else 0.0
    elif mode == 'ratio_J':
        J = float(cfg.get('jVal', 0.0) or 0.0)
    else:
        return None, 0.0
    if not J > 0:
        return None, 0.0
    rel = (float(cfg.get('jErrPct', 0.0) or 0.0) / 100.0
           if mode == 'ratio_J' and cfg.get('jForce')
           else float(cfg_j_err_frac(cfg)))
    return J, rel


ERR_MODES = ('sem_mswd', 'sem_mswd_2s', 'sem')


# σ level of step errors in a file that carries neither marker below: the
# data converter wrote raw 1σ before it started marking its files.
UNMARKED_ERR_DIV = 1.0


def file_err_div(cfg, unmarked=None):
    """Divisor that brings a saved session's measured step errors to 1σ, the
    one rule every tool reads files with.  errScale=2 marks files written by
    the Plateau & isochron tool (2σ), stepSigma=1 converter files (1σ).  A file with
    neither uses ``unmarked`` (a tool's own setting), else UNMARKED_ERR_DIV."""
    if cfg.get('errScale') == 2:
        return 2.0
    if cfg.get('stepSigma') == 1:
        return 1.0
    return float(unmarked) if unmarked else UNMARKED_ERR_DIV


def scale_sem(sem, mswd, prob, n, err_mode='sem_mswd_2s', p_cut=5.0):
    """Analytical 1σ of a weighted mean under the Plateau error modes.

      'sem_mswd'     SEM·√MSWD·t₉₅(n−1)/2 when p is below the cutoff (%), so the
                     2σ display is the Student-t 95% interval (Isoplot); else SEM
      'sem_mswd_2s'  SEM·√MSWD when MSWD exceeds the Wendt & Carl (1991) 95%
                     limit for n, else SEM
      'sem'          SEM

    Used for single plateaus, stacked plateaus (both n bases) and the plateau
    ages the Age distribution tool reads from a stack."""
    if err_mode == 'sem_mswd':
        if n > 1 and mswd > 0 and prob * 100.0 < p_cut:
            return sem * math.sqrt(mswd) * t_crit(n - 1) / 2.0
        return sem
    if err_mode == 'sem_mswd_2s':
        if n > 1 and mswd > mswd_critical(n):
            return sem * math.sqrt(mswd)
        return sem
    return sem


def r_to_age(R, err1s_R, J, J_err_frac, lam=LAMBDA_DEFAULT):
    """Age (Ma) and 1s error (Ma) from a single R = 40Ar*/39Ar_K value."""
    if R <= 0 or J <= 0 or lam <= 0:
        return 0.0, 0.0
    age_yr       = np.log(J * R + 1.0) / lam
    denom        = lam * (J * R + 1.0)
    sigma_J      = J * J_err_frac
    sigma_age_yr = np.sqrt((J * err1s_R)**2 + (R * sigma_J)**2) / denom
    return age_yr * 1e-6, sigma_age_yr * 1e-6


def rvalue_age_external(R, sigma_R, t_std_ma, sigma_t_std_ma,
                        lam=LAMBDA_DEFAULT, sigma_lam=LAMBDA_ERR):
    """Age and full external error by the R-value intercalibration method
    (Renne et al. 2010), for a standard whose age does not depend on the decay
    constant (an astronomically tuned age):

        T = (1/λ) · ln[ (e^{λ·t_std} − 1)·R + 1 ]
        σ²_T = (∂T/∂λ)²·σ²_λ + (∂T/∂t_std)²·σ²_t_std + (∂T/∂R)²·σ²_R

    The ∂T/∂λ term nearly cancels, so the external is dominated by σ_R and
    σ_t_std.  Not valid for a jointly calibrated standard (Carter et al. 2025),
    where the standard age and λ are covariant.

    Parameters
    ----------
    R              : weighted-mean F_sample / F_standard (intercalibration ratio)
    sigma_R        : 1σ on R
    t_std_ma       : astronomical age of the standard (Ma)
    sigma_t_std_ma : 1σ on the standard age (Ma)
    lam, sigma_lam : ⁴⁰K decay constant (yr⁻¹) and its 1σ

    Returns a dict: age_ma, ext_ma, ext_frac (ext/age), the three variance
    shares (var_lambda / var_tstd / var_R as fractions), and the partials.
    """
    t_std = t_std_ma * 1e6
    s_std = sigma_t_std_ma * 1e6
    e = math.exp(lam * t_std)
    X = (e - 1.0) * R + 1.0
    if X <= 0:
        return dict(age_ma=float('nan'), ext_ma=float('nan'), ext_frac=float('nan'))
    t = math.log(X) / lam
    dT_dlam  = -math.log(X) / lam**2 + R * t_std * e / (lam * X)
    dT_dtstd = R * e / X
    dT_dR    = (e - 1.0) / (lam * X)
    v_lam  = (dT_dlam  * sigma_lam)**2
    v_tstd = (dT_dtstd * s_std)**2
    v_R    = (dT_dR    * sigma_R)**2
    var = v_lam + v_tstd + v_R
    ext = math.sqrt(var)
    return dict(
        age_ma=t / 1e6, ext_ma=ext / 1e6,
        ext_frac=(ext / t) if t > 0 else float('nan'),
        var_lambda=(v_lam / var) if var > 0 else 0.0,
        var_tstd=(v_tstd / var) if var > 0 else 0.0,
        var_R=(v_R / var) if var > 0 else 0.0,
        dT_dlam=dT_dlam, dT_dtstd=dT_dtstd, dT_dR=dT_dR,
    )


def rvalue_external_sigma(age_ma, t_std_ma, sigma_t_std_ma,
                          lam=LAMBDA_DEFAULT, sigma_lam=LAMBDA_ERR,
                          cov_lam_tstd=0.0):
    """Shared systematic (Ma, 1σ) the standard age + decay constant add on top
    of the internal tier, in the R-value framework, to combine in quadrature
    with the internal error to get the external tier.

    Uses the R-value partials with R back-computed from the age, so λ enters the
    way it does in rvalue_age_external (nearly cancelling) rather than the naive
    full-leverage route.  cov_lam_tstd is cov(λ, t_std): 0 for an independent
    astronomical anchor, Carter's joint value for the Carter calibration.
    Returns 0 for a degenerate standard age.
    """
    dT_dtstd, dT_dlam, T = _rvalue_partials(age_ma, t_std_ma, lam)
    if T == 0.0:
        return 0.0
    s_std = sigma_t_std_ma * 1e6
    var = ((dT_dtstd * s_std)**2 + (dT_dlam * sigma_lam)**2
           + 2.0 * dT_dtstd * dT_dlam * cov_lam_tstd)
    return math.sqrt(max(0.0, var)) / 1e6


def cfg_is_astronomical(cfg):
    """True if the cfg's FC age + λ match an astronomically tuned preset."""
    age = cfg.get('stdAgeMa')
    lam = cfg.get('lambdaYr')
    if age is None or lam is None:
        return False
    for name, a, e, ext in FC_PRESETS:
        if name in ASTRONOMICAL_FC_NAMES and abs(a - age) < 5e-4:
            pl = paired_lambda(name)
            if pl is not None and abs(pl[1] - lam) < 1e-14:
                return True
    return False


# Carter et al. (2025) posterior covariance cov(λ_tot, t_FCs), from their
# Ar_ages_and_uncertainties.ipynb (figshare doi:10.6084/m9.figshare.27284775).
# Units: (yr⁻¹·yr).  With Carter's posterior σ's it corresponds to ρ ≈ 0, so it
# is numerically negligible, but it is the covariance term of their joint
# calibration and is carried here for completeness.
CARTER_COV_LAM_FCS = 1.866205e-20


def cfg_is_carter(cfg):
    """True if the cfg's FC age + λ match the Carter et al. (2025) joint preset."""
    age = cfg.get('stdAgeMa')
    lam = cfg.get('lambdaYr')
    if age is None or lam is None:
        return False
    for name, a, _e, _x in FC_PRESETS:
        if name.startswith('Carter') and abs(a - age) < 5e-4:
            pl = paired_lambda(name)
            return pl is not None and abs(pl[1] - lam) < 1e-14
    return False


def cfg_covariance(cfg):
    """cov(λ, t_std) for the calibration: Carter's joint value, else 0."""
    return CARTER_COV_LAM_FCS if cfg_is_carter(cfg) else 0.0


def _rvalue_partials(age_ma, t_std_ma, lam):
    """(dT/dt_std, dT/dλ) in SI-ish units (yr per unit), with R implied by the age."""
    T = age_ma * 1e6
    t_std = t_std_ma * 1e6
    e_std = math.exp(lam * t_std)
    e_T = math.exp(lam * T)
    if e_std <= 1.0:
        return 0.0, 0.0, 0.0
    R = (e_T - 1.0) / (e_std - 1.0)
    X = e_T
    dT_dtstd = R * e_std / X
    dT_dlam = -T / lam + R * t_std * e_std / (lam * X)
    return dT_dtstd, dT_dlam, T


def rvalue_external_terms(age_ma, t_std_ma, sigma_t_std_ma,
                          lam=LAMBDA_DEFAULT, sigma_lam=LAMBDA_ERR):
    """(FCs-age term, λ term) in Ma of the R-value external systematic."""
    dT_dtstd, dT_dlam, T = _rvalue_partials(age_ma, t_std_ma, lam)
    if T == 0.0:
        return 0.0, 0.0
    return (abs(dT_dtstd * sigma_t_std_ma * 1e6) / 1e6,
            abs(dT_dlam * sigma_lam) / 1e6)


def rvalue_external_sigma_anchored(age_ma, anchor,
                                   lam=LAMBDA_DEFAULT, sigma_lam=LAMBDA_ERR):
    """Shared systematic (Ma, 1s) anchored on the orbital age, not on t_FCs.

    Renne et al. (1998) eq. (6) applied to the n=2 chain of eq. (5):

        t = (1/lam) * ln[ R_sam/FCs * R_FCs/astro * (e^(lam*t_astro) - 1) + 1 ]

    Every input is then independently measured -- an orbital cycle count and two
    isotope ratios -- so the sigma_lambda / sigma(t_FCs) correlation that eq. (6)
    cannot carry simply does not arise.  The product R is implied by the age, so
    only t_astro, its sigma, and the relative sigma on R_FCs/astro are needed.

    Returns (sigma_sys_ma, t_astro_term, lam_term, R_term), all 1s Ma.  The
    sample's own R (the internal tier) is deliberately excluded.
    """
    t_astro_ma, s_astro_ma, r_fc, s_r_fc = anchor
    if age_ma <= 0 or t_astro_ma <= 0:
        return 0.0, 0.0, 0.0, 0.0
    d_lnR, d_ta, d_lam = anchored_partials(age_ma, anchor, lam)
    rel_r = (s_r_fc / r_fc) if r_fc else 0.0
    tt = abs(d_ta * s_astro_ma)
    lt = abs(d_lam * sigma_lam) / 1e6
    rt = abs(d_lnR * rel_r)
    return math.sqrt(tt**2 + lt**2 + rt**2), tt, lt, rt


def systematic_components(age_ma, cfg, include_cross=True):
    """The calibration systematic for one age, as signed 1σ components (Ma).

    Returns a list of (label, value).  The components are independent of each
    other, so the external systematic is the root-sum-square of the values; they
    are signed so that several ages can be combined with exact linear
    propagation of each shared parameter (see combine_ages).

    Anchored calibration (Kuiper 2008, Rivera 2011, Phillips 2022), Renne et al.
    (1998) eq. 6 on the chain of eq. 5, partials as in Kuiper et al. (2008) SOM:
        R(FCs/A) calibration        ∂t/∂ln R · σ_R/R
        anchor astronomical age     ∂t/∂t_A  · σ_tA
        decay constant λ            ∂t/∂λ    · σ_λ      (λ nearly cancels)

    Joint calibration (Carter 2025): single-step R-value partials on (t_FCs, λ)
    with their covariance, decomposed into two independent components.

    Other (legacy physical-λ presets, custom values): the J-fraction route with
    full λ leverage, which is what those calibrations imply.

    cfg['fullSystematics'] (Box 2) replaces the λ component by the full physical
    40K decay scheme of Min et al. (2000), propagated without cancellation.
    cfg['crossChronometer'] (legacy) adds an age·σ_λ/λ component.
    """
    lam = cfg.get('lambdaYr', LAMBDA_DEFAULT)
    slam = cfg.get('lambdaErrYr', LAMBDA_ERR)
    t_std = cfg.get('stdAgeMa', STD_AGE_DEFAULT)
    s_std = cfg.get('stdAgeErrMa', STD_AGE_ERR)
    full = bool(cfg.get('fullSystematics'))
    anc = cfg_anchor(cfg)
    comps = []
    lam_label = 'Decay constant λ'
    if anc is not None:
        d_lnR, d_ta, d_lam = anchored_partials(age_ma, anc, lam)
        comps = [('R(FCs/anchor) calibration', d_lnR * anc[3] / anc[2]),
                 ('Anchor tephra astronomical age', d_ta * anc[1]),
                 (lam_label, d_lam * slam / 1e6)]
        cross_ok = True
    else:
        cls, cov = calibration_box(cfg)
        if cls in ('astronomical', 'joint'):
            d_t0, d_lam, _ = _rvalue_partials(age_ma, t_std, lam)
            st = s_std * 1e6
            if st > 0:
                c1 = d_t0 * st + d_lam * cov / st
                c2 = d_lam * math.sqrt(max(0.0, slam**2 - cov**2 / st**2))
            else:
                c1, c2 = 0.0, d_lam * slam
            comps = [('FCs standard age', c1 / 1e6), (lam_label, c2 / 1e6)]
            cross_ok = True
        else:
            e = math.exp(lam * t_std * 1e6)
            f = e / (e - 1.0)
            ext_age = cfg.get('stdAgeErrType', 'internal') == 'external'
            d_lnR = dt_dlnR(age_ma, lam)
            # A standard age that scales as 1/λ (dt_std/dλ = −t_std/λ) gives the
            # total derivative dt/dλ = ∂t/∂λ − (t_std/λ) ∂t/∂t_std = −t/λ.
            comps = [('FCs standard age', d_lnR * f * lam * s_std * 1e6),
                     (lam_label, 0.0 if ext_age else -age_ma * slam / lam)]
            cross_ok = ext_age
    if full:
        decay = min_full_external(
            age_ma, decay_set=cfg.get('minDecaySet', MIN_DECAY_DEFAULT))['decay_ma']
        comps = [c for c in comps if c[0] != lam_label]
        comps.append(('Decay constant (Min full external)', decay))
    elif include_cross and cfg.get('crossChronometer') and cross_ok:
        comps.append(('Decay constant λ (cross-chronometer)',
                      age_ma * slam / lam if lam > 0 else 0.0))
    return comps


def external_route(cfg):
    """Short label for how systematic_components propagates the external."""
    if cfg.get('fullSystematics'):
        return 'full external (Min: physical ⁴⁰K decay scheme)'
    anc = cfg_anchor(cfg)
    if anc is not None:
        return (f'astronomical anchor, t_A = {anc[0]:g} Ma via R(FCs/A) '
                f'(λ nearly cancels)')
    cls, _ = calibration_box(cfg)
    route = {'astronomical': 'R-value on the FCs age (λ nearly cancels)',
             'joint': 'joint calibration (Carter-style)'}.get(
                 cls, 'single-step J route (full λ)')
    if calibration_warning(cfg):
        route += '; overstated for Renne et al., see About'
    return route


def systematic_matrix(ages, cfgs):
    """(n × k array of signed 1σ components in Ma, list of k labels).

    cfgs is one config for every age or a sequence of per-age configs.  Rows
    with different calibrations get zeros in each other's columns; combine them
    with separate standard groups.
    """
    ages = [float(a) for a in np.asarray(ages, float).ravel()]
    if isinstance(cfgs, dict):
        cfgs = [cfgs] * len(ages)
    rows = [systematic_components(a, c) for a, c in zip(ages, cfgs)]
    labels = []
    for r in rows:
        for lbl, _ in r:
            if lbl not in labels:
                labels.append(lbl)
    C = np.zeros((len(ages), len(labels)))
    for i, r in enumerate(rows):
        for lbl, v in r:
            C[i, labels.index(lbl)] = v
    return C, labels


def age_error_budget(sigma_ana, sigma_int, sigma_ext, age_ma, cfg):
    """Split the external variance into named sources that sum to sigma_ext².

    Returns a list of (label, variance_Ma²): analytical, J / neutron flux, and
    one slice per calibration component of systematic_components (for an
    anchored calibration: the R(FCs/anchor) measurement, the anchor tephra's
    astronomical age, and λ).  The systematic slices are the exact component
    variances, rescaled only if sigma_ext was computed elsewhere.
    """
    v_ana = sigma_ana**2
    v_flux = max(0.0, sigma_int**2 - sigma_ana**2)
    v_sys = max(0.0, sigma_ext**2 - sigma_int**2)
    comps = [(lbl, v * v) for lbl, v in systematic_components(age_ma, cfg)]
    tot = sum(v for _, v in comps)
    budget = [('Analytical (mass-spec)', v_ana), ('J / neutron flux', v_flux)]
    if tot > 0:
        budget += [(lbl, v_sys * v / tot) for lbl, v in comps]
    elif v_sys > 0:
        budget.append(('Calibration systematic', v_sys))
    return budget


# --------------------------------------------------------------------------
# Min et al. (2000) full external error
# --------------------------------------------------------------------------
#
# Min et al. (2000, GCA 64, 73-98) propagate the measured 40K
# decay scheme rather than a single pinned total lambda.  The two partial
# decay constants are measured independently:
#   lambda_e  = electron-capture branch  40K -> 40Ar   ("lambda_40Ar")
#   lambda_b  = beta-minus branch        40K -> 40Ca
# and the age equation carries the branching factor lambda/lambda_e:
#   t = (1/lambda) * ln[ 1 + (lambda/lambda_e) * D ],   lambda = lambda_e + lambda_b
# where D = 40Ar*/40K of the sample.  Because lambda_e is poorly known and does
# not cancel, its full uncertainty propagates almost one-to-one into the age.
# This is the source of the large "full external" errors reported by ArArCALC.
#
# Constants below are the sets tabulated by Min et al. (2000).  The paper quotes
# 2-sigma uncertainties; stored here as 1-sigma (halved) to match the rest of
# this module.
MIN_DECAY_SETS = {
    # name: (lambda_e, lambda_e_1s, lambda_b, lambda_b_1s)  [yr^-1]
    'Audi et al. (1997)':          (0.580e-10, 0.007e-10, 4.884e-10, 0.0495e-10),
    'Endt & Van der Leun (1990)':  (0.580e-10, 0.0085e-10, 4.950e-10, 0.043e-10),
}
MIN_DECAY_DEFAULT = 'Audi et al. (1997)'   # Min's provisionally recommended set
# 40K/K abundance (Min et al. 2000, Audi 1997): 1.17e-4 +- 0.02e-4 (2s) -> 1s.
MIN_K40_K = 1.17e-4
MIN_K40_K_1S = 0.01e-4


def min_full_external(age_ma, sigma_int_ma=0.0, sigma_std_ma=0.0,
                      decay_set=MIN_DECAY_DEFAULT, include_abundance=False):
    """Min et al. (2000) full external error (Ma, 1sigma).

    Propagates the two 40K partial decay constants independently (Min's
    hierarchical scheme, no covariance) through the K-Ar-equivalent age
    equation with branching,
        t = (1/lambda) * ln[1 + (lambda/lambda_e) * D],   lambda = lambda_e + lambda_b,
    where D = 40Ar*/40K of the sample, back-computed from the age with the same
    constants.  This is the non-cancelling decay-constant term; unlike the
    intercalibrated route (rvalue_age_external) lambda does not cancel here, so
    the sensitivity d(ln t)/d(ln lambda) ~ -1.

    The decay-constant term is combined in quadrature with the supplied internal
    (analytical + J) term and the standard (K/Ar of the primary standard) term,
    both 1sigma Ma, to give the full external.  include_abundance adds the
    40K/K term (small; only meaningful for an absolute primary-standard chain,
    it cancels in an intercalibrated ratio, so it is off by default).

    Returns a dict: age_ma, decay_ma (decay-constant 1sigma term), ext_ma (full
    external 1sigma, if internal/std supplied), and the branch sub-terms
    (le_ma, lb_ma, f_ma) and total lambda used.

    Validated against Min et al. (2000) F-239 (see tests/test_min_external.py):
    age ~1090 Ma, decay-constant 2sigma ~21 Ma; combined with the paper's
    +-10 Ma (2s) standard term the full external is ~+-23 Ma (2s), reproducing
    their 1090.6 +- 23.3 Ma (Audi 1997 constants).
    """
    if decay_set not in MIN_DECAY_SETS:
        raise ValueError('unknown Min decay set: %r' % (decay_set,))
    le, le_s, lb, lb_s = MIN_DECAY_SETS[decay_set]
    T = age_ma * 1e6
    lam = le + lb
    if T <= 0 or le <= 0 or lam <= 0:
        return dict(age_ma=age_ma, decay_ma=0.0, ext_ma=0.0,
                    le_ma=0.0, lb_ma=0.0, f_ma=0.0, lam_total=lam)
    D = (le / lam) * math.expm1(lam * T)          # 40Ar*/40K implied by the age

    def _age(le_, lb_, D_):
        l = le_ + lb_
        return math.log1p((l / le_) * D_) / l      # yr

    h = 1e-6
    dle = (_age(le * (1 + h), lb, D) - _age(le * (1 - h), lb, D)) / (2 * le * h)
    dlb = (_age(le, lb * (1 + h), D) - _age(le, lb * (1 - h), D)) / (2 * lb * h)
    le_term = dle * le_s                            # yr
    lb_term = dlb * lb_s
    var = le_term**2 + lb_term**2
    f_term = 0.0
    if include_abundance:
        # 40K/K enters via D = (40Ar*/K)/f with 40Ar*/K held fixed: dD/df = -D/f
        dD = (_age(le, lb, D * (1 + h)) - _age(le, lb, D * (1 - h))) / (2 * D * h)
        f_term = dD * (-D / MIN_K40_K) * MIN_K40_K_1S
        var += f_term**2
    decay = math.sqrt(var)                          # yr
    ext = math.sqrt(decay**2 + (sigma_int_ma * 1e6)**2 + (sigma_std_ma * 1e6)**2)
    return dict(age_ma=age_ma,
                decay_ma=decay / 1e6,
                ext_ma=ext / 1e6,
                le_ma=abs(le_term) / 1e6,
                lb_ma=abs(lb_term) / 1e6,
                f_ma=abs(f_term) / 1e6,
                lam_total=lam)


# --------------------------------------------------------------------------
# External error: which calibration is propagated which way
# --------------------------------------------------------------------------
#
# Box 1, intercalibrated (default).  The standard age is fixed independently
#   of lambda (astronomical tuning) or jointly with it (Carter), so lambda
#   nearly cancels in the R-value form and the external is dominated by the
#   standard age.  The two cases differ only in cov(lambda, t_std): 0 for an
#   astronomical age, the calibration's own value for a joint fit.
#
# Box 2, Min et al. (2000) full external (opt-in).  Propagates the measured
#   40K decay scheme (lambda_e, lambda_b) without cancellation.  It treats the
#   decay constants as independent of the standard, so it should not be
#   combined with a jointly calibrated lambda (see full_systematics_warning).

def calibration_box(cfg):
    """(regime, cov) for a calibration's default (Box 1) external.

    regime is 'astronomical' (cov=0), 'joint' (cov=Carter's value), or 'other'
    (legacy physical-lambda presets that fall back to the J-fraction route).
    cov is cov(lambda, t_std) for the intercalibrated propagation.
    Uses cfg['calibClass'] when set (authoritative), else detects from age+lambda.
    """
    cls = cfg.get('calibClass')
    if cls is None:
        cls = ('astronomical' if cfg_is_astronomical(cfg)
               else 'joint' if cfg_is_carter(cfg) else 'other')
    return cls, cfg_covariance(cfg)


def calibration_group_key(cfg):
    """Hashable identity of a calibration, used to decide which samples share
    their systematic components when pooling.

    Missing flags count as their defaults and the calibration class is resolved
    as systematic_components resolves it, so a config read from a file and one
    filled in from the calibration bar compare equal when they describe the same
    calibration.  The constants are compared to 12 significant digits.
    """
    def num(k):
        v = cfg.get(k)
        return None if v is None else float(f'{float(v):.12g}')
    return (num('stdAgeMa'), num('stdAgeErrMa'), num('lambdaYr'), num('lambdaErrYr'),
            bool(cfg.get('fullSystematics')), bool(cfg.get('crossChronometer')),
            calibration_box(cfg)[0])


# Renne et al. (2010, 2011) calibrated the FCs age and the 40K decay constants
# jointly, but the covariances they publish (2010 Table 4b; 2011 Table 1) are
# rounded too far to propagate below about 1 Ga: the matrix is not positive
# definite and the age uncertainty from it falls well short of their own Monte
# Carlo values (2010 Table 5).  Their Monte Carlo triplets (2010, EA-3) are not
# included, so these presets go through the single-step route with the FCs age
# and λ independent, which overstates the external error.
RENNE_JOINT_NAMES = ('Renne et al. (2010)', 'Renne et al. (2011)')


def calibration_warning(cfg):
    """A warning when the FCs age is that of Renne et al. (2010) or (2011),
    else ''."""
    age = cfg.get('stdAgeMa')
    if age is None:
        return ''
    for name, a, _e, _x in FC_PRESETS:
        if name in RENNE_JOINT_NAMES and abs(a - float(age)) < 5e-4:
            return (f'{name}: the FCs age and λ were calibrated together, but their '
                    'published covariances are too rounded to propagate, so SimplAr '
                    'treats them as independent. The external error is overstated, '
                    'by about a factor of four at 56 Ma. Internal errors are not '
                    'affected.')
    return ''


def full_systematics_warning(cfg):
    """A warning when the Min full external is combined with a jointly
    calibrated lambda (Carter), else ''.  The joint lambda was fixed with the
    help of U-Pb ages; the Min uncertainty treats lambda as independent of
    them, so the two do not belong together."""
    cls, _ = calibration_box(cfg)
    if cls == 'joint':
        return ('Full-systematics (Min) uses the independent physical decay '
                'constant, which is inconsistent with a jointly calibrated '
                'lambda (e.g. Carter). Use Box 1 (default) for this '
                'calibration, or switch to a physical-lambda preset for Min.')
    return ''


def external_error(age_ma, cfg, full_systematics=None):
    """Calibration systematic (Ma, 1σ) for one age: the root-sum-square of
    systematic_components, without the internal tier and without the legacy
    cross-chronometer term.

    full_systematics=True/False overrides cfg['fullSystematics'] (Box 2, the
    Min et al. 2000 physical decay scheme in place of the λ component).
    """
    if full_systematics is not None:
        cfg = dict(cfg, fullSystematics=bool(full_systematics))
    return math.sqrt(sum(v * v for _, v in
                         systematic_components(age_ma, cfg, include_cross=False)))


def external_sigma_array(ages, sigma_int, cfg):
    """Per-analysis external σ (Ma, 1σ) = internal ⊕ calibration systematic.

    The systematic is the root-sum-square of systematic_components: for an
    anchored calibration the R(FCs/A) calibration, the anchor's orbital age
    and λ on the two-step chain; for the Carter joint calibration the
    decomposed (t_FCs, λ) covariance; otherwise the single-step route with
    full λ leverage.  cfg['fullSystematics'] and cfg['crossChronometer'] are
    honoured there.
    """
    ages = np.asarray(ages, float)
    sigma_int = np.asarray(sigma_int, float)
    sysv = np.array([math.sqrt(sum(v * v for _, v in systematic_components(a, cfg)))
                     for a in ages.ravel()]).reshape(ages.shape)
    return np.sqrt(sigma_int**2 + sysv**2)


def r_to_age_cfg(R_vals, sigma_vals, cfg):
    """Vectorised R -> age using a flux config dict.

    Returns ((ages_ma, sigmas_ma), None) on success or (None, message) on a
    bad configuration. Sigmas come out at the same level they went in.

    ratio_R_FCs:  t = ln(1 + R*em1) / lam,  em1 = e^{λ t_FCs} − 1 (cfg_ingrowth,
                  on the anchored chain R_FCs/A (e^{λ t_A} − 1) where one exists)
    ratio_R_std:  t = ln(1 + J*R) / lam,    J = em1 / R_std
    ratio_J:      t = ln(1 + J*R) / lam
    """
    mode = cfg.get('ageFluxMode', 'imported')
    lam  = cfg.get('lambdaYr', LAMBDA_DEFAULT)
    R    = np.asarray(R_vals, float)
    sig  = np.asarray(sigma_vals, float)
    if mode == 'ratio_R_FCs':
        em1   = cfg_ingrowth(cfg, lam)
        arg   = 1.0 + R * em1
        if np.any(arg <= 0):
            return None, 'Some R_unk/R_FCs give ln(...) <= 0.'
        return (np.log(arg) / lam / 1e6, (em1 / (lam * arg) / 1e6) * sig), None
    if mode == 'ratio_R_std':
        r_std = cfg.get('rStd', 70.0)
        if r_std <= 0:
            return None, 'R_std must be > 0.'
        J = cfg_ingrowth(cfg, lam) / r_std
    elif mode == 'ratio_J':
        J = cfg.get('jVal', 0.005)
        if J <= 0:
            return None, 'J must be > 0.'
    else:
        return None, 'Set flux mode to ratio_R_std, ratio_J or ratio_R_FCs to use R input.'
    arg = 1.0 + J * R
    if np.any(arg <= 0):
        return None, 'Some R give ln(1+J*R) <= 0.'
    return (np.log(arg) / lam / 1e6, (J / (lam * arg) / 1e6) * sig), None


def rr_from_age(age_ma, cfg):
    """(R_unk/R_std, dR/d_age in per-Ma) from an age and a flux config.
    Returns (None, None) if the FC standard age is unusable."""
    lam    = cfg.get('lambdaYr', LAMBDA_DEFAULT)
    em1_fc = cfg_ingrowth(cfg, lam)
    if em1_fc <= 0:
        return None, None
    em1_unk = math.exp(lam * age_ma * 1e6) - 1.0
    rr      = em1_unk / em1_fc
    dr_dt   = lam * math.exp(lam * age_ma * 1e6) * 1e6 / em1_fc
    return rr, dr_dt


# --------------------------------------------------------------------------
# York (1969) regression
# --------------------------------------------------------------------------

def york_regression(x, sx, y, sy, ri=None):
    """Iterative error-weighted straight-line fit with correlated errors.

    x, sx, y, sy are values and 1s errors; ri the optional per-point error
    correlations. Returns a dict with slope, intercept, their 1s errors,
    cov_ab, x_bar, y_bar, mswd, prob and n, or None if n < 3 or the fit
    degenerates.
    """
    x  = np.asarray(x,  dtype=float)
    sx = np.asarray(sx, dtype=float)
    y  = np.asarray(y,  dtype=float)
    sy = np.asarray(sy, dtype=float)
    n  = len(x)
    if n < 3:
        return None

    ri = np.zeros(n) if ri is None else np.asarray(ri, dtype=float)
    ri = np.clip(ri, -0.9999, 0.9999)

    wx = np.where(sx > 0, 1.0 / sx**2, 1e30)
    wy = np.where(sy > 0, 1.0 / sy**2, 1e30)
    alpha = np.sqrt(wx * wy)

    # start from the ordinary least-squares slope
    x_m = np.mean(x)
    y_m = np.mean(y)
    denom = np.sum((x - x_m)**2)
    b = np.sum((x - x_m) * (y - y_m)) / denom if denom > 0 else 1.0

    beta = np.zeros(n)
    for _ in range(500):
        denom_W = wx + b**2 * wy - 2.0 * b * ri * alpha
        denom_W = np.where(denom_W > 0, denom_W, 1e-30)
        W = alpha**2 / denom_W
        Ws = W.sum()
        if Ws == 0:
            return None
        x_bar = (W * x).sum() / Ws
        y_bar = (W * y).sum() / Ws
        U = x - x_bar
        V = y - y_bar
        beta = W * (U / wy + b * V / wx - (b * U + V) * ri / alpha)
        num = (W * beta * V).sum()
        den = (W * beta * U).sum()
        if den == 0:
            return None
        b_new = num / den
        if abs(b_new - b) < 1e-12 * (abs(b) + 1e-30):
            b = b_new
            break
        b = b_new

    # final pass with the converged slope
    denom_W = wx + b**2 * wy - 2.0 * b * ri * alpha
    denom_W = np.where(denom_W > 0, denom_W, 1e-30)
    W = alpha**2 / denom_W
    Ws = W.sum()
    x_bar = (W * x).sum() / Ws
    y_bar = (W * y).sum() / Ws
    a = y_bar - b * x_bar

    # adjusted x values (York 1969 eq. 13) for the uncertainties
    x_adj = x_bar + beta
    x_bar_adj = (W * x_adj).sum() / Ws
    Sxx_adj = (W * (x_adj - x_bar_adj)**2).sum()
    if Sxx_adj <= 0:
        return None

    slope_1s = 1.0 / np.sqrt(Sxx_adj)
    intercept_1s = np.sqrt(1.0 / Ws + x_bar_adj**2 * slope_1s**2)
    cov_ab = -x_bar_adj * slope_1s**2

    S = (W * (y - a - b * x)**2).sum()
    mswd = float(S / (n - 2)) if n > 2 else 0.0
    prob = float(1.0 - chi2_cdf(S, n - 2)) if n > 2 else 1.0

    return dict(slope=b, intercept=a,
                slope_1s=slope_1s, intercept_1s=intercept_1s, cov_ab=cov_ab,
                x_bar=x_bar, y_bar=y_bar, mswd=mswd, prob=prob, n=n)


def isochron_age(fit, J, lam, inverse):
    """Age and analytical 1σ (Ma) of a York isochron fit, with the initial
    ⁴⁰Ar/³⁶Ar and its 1σ.

    inverse=True: ³⁶Ar/⁴⁰Ar vs ³⁹Ar/⁴⁰Ar, ⁴⁰Ar*/³⁹Ar = −b/a, (⁴⁰Ar/³⁶Ar)₀ = 1/a.
    inverse=False: ⁴⁰Ar/³⁶Ar vs ³⁹Ar/³⁶Ar, ⁴⁰Ar*/³⁹Ar = b, (⁴⁰Ar/³⁶Ar)₀ = a.
    Both errors are multiplied by √MSWD when MSWD > 1 (the ArArCALC isochron
    convention).  Returns dict(age_ma, s_ana, init_ratio, init_ratio_1s) with
    None for whatever cannot be computed (no fit, J ≤ 0, a ≤ 0, ...)."""
    out = dict(age_ma=None, s_ana=None, init_ratio=None, init_ratio_1s=None)
    if fit is None:
        return out
    b, a = fit['slope'], fit['intercept']
    sb, sa, cov = fit['slope_1s'], fit['intercept_1s'], fit['cov_ab']
    mf = math.sqrt(fit['mswd']) if fit['mswd'] > 1.0 else 1.0
    if inverse:
        if a > 0:
            out['init_ratio'] = 1.0 / a
            out['init_ratio_1s'] = sa / a**2 * mf
        if J and J > 0 and b != 0 and a > 0:
            R = -b / a
            X = R * J
            if X > 0:
                var_R = (sb / a)**2 + (b * sa / a**2)**2 - 2.0 * b * cov / a**3
                out['age_ma'] = math.log1p(X) / lam * 1e-6
                out['s_ana'] = J * math.sqrt(max(var_R, 0.0)) / (lam * (1.0 + X)) * 1e-6 * mf
    else:
        out['init_ratio'] = a
        out['init_ratio_1s'] = sa * mf
        if J and J > 0 and b > 0:
            X = b * J
            out['age_ma'] = math.log1p(X) / lam * 1e-6
            out['s_ana'] = J * sb / (lam * (1.0 + X)) * 1e-6 * mf
    return out


# --------------------------------------------------------------------------
# Number formatting for tables and CSV export
# --------------------------------------------------------------------------

def fsig(v, n):
    """v to exactly n significant figures in fixed-decimal notation, keeping
    trailing zeros (:.Ng strips them). '' for None/NaN/inf."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ''
    if v == 0:
        return '0.' + '0' * (n - 1)
    mag = int(math.floor(math.log10(abs(v))))
    dp = max(0, n - 1 - mag)
    return f"{v:.{dp}f}"


def ff(v, d):
    """v with exactly d decimal places; '' for None/NaN/inf."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ''
    return f"{v:.{d}f}"


# --------------------------------------------------------------------------
# Irradiation-disk drawing, shared by the flux dialogs of both plotting GUIs
# --------------------------------------------------------------------------

def render_disk(fig, hole_tbl, fx, key, ringed):
    """Draw a disk's holes and fill hole_tbl.

    Holes are coloured by R, the monitor ⁴⁰Ar*/³⁹Ar ratio the flux encodes,
    R = (e^{λ_f·t_f} − 1)/J from the flux's own calibration.  R is the measured
    (calibration-independent) monitor ratio, so it lines up directly with the
    plateau window's R standard, which is easier to eyeball than a J.  When the
    flux carries no stored calibration (legacy load) it falls back to showing J.
    The relative error is identical for R and J.  Ringed holes are circled in
    red.  Returns the Axes."""
    from PyQt5.QtWidgets import QTableWidgetItem
    from PyQt5.QtCore import Qt

    fig.clear()
    ax = fig.add_subplot(111)
    hole_tbl.setRowCount(0)
    if fx is None:
        return ax
    J = np.asarray(fx['J'], float)
    Js = np.asarray(fx['J_sigma'], float)
    holes = fx['holes']
    if fx.get('R') is not None or (fx.get('lamFlux') and fx.get('fcFlux')):
        val = np.array([flux_hole_R(fx, k) or np.nan for k in range(len(holes))], float)
        vlabel, vfmt = 'R', '.6g'
    else:
        val = J
        vlabel, vfmt = 'J', '.8g'
    xy = fx.get('hole_xy', np.array([]))
    if getattr(xy, 'size', 0) and xy.shape[0] == len(holes):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, s=260, cmap='viridis',
                        edgecolors='k', zorder=3)
        for k, h in enumerate(holes):
            ax.annotate(h, (xy[k, 0], xy[k, 1]), ha='center', va='center',
                        fontsize=7, color='white', zorder=4)
        ax.set_aspect('equal')
        ax.set_title(f"Disk {key[1]}  ({fx.get('design', '')})", fontsize=9)
        fig.colorbar(sc, ax=ax, label=vlabel, fraction=0.046, pad=0.04)
        mi = sorted(ringed)
        if mi:
            ax.scatter(xy[mi, 0], xy[mi, 1], s=480, facecolors='none',
                       edgecolors='red', linewidths=2.0, zorder=5)
    else:
        ax.text(0.5, 0.5, 'No hole coordinates in this file.\n'
                '(re-export from the flux calculator to draw the disk)',
                transform=ax.transAxes, ha='center', va='center',
                color='#888', style='italic', fontsize=9)
    fig.tight_layout()
    hole_tbl.setColumnCount(3)
    hole_tbl.setHorizontalHeaderLabels(['Hole', vlabel, f'±{vlabel}%'])
    hole_tbl.setRowCount(len(holes))
    for k, h in enumerate(holes):
        pct = (Js[k] / J[k] * 100.0) if J[k] else 0.0
        vtxt = f'{val[k]:{vfmt}}' if np.isfinite(val[k]) else '—'
        for c, txt in enumerate([h, vtxt, f'{pct:.4g}']):
            it = QTableWidgetItem(txt)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            hole_tbl.setItem(k, c, it)
    return ax


BUDGET_COLORS = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2', '#9D755D']


def draw_budget_pie(ax, budget, scale, sig_lbl, centre_fmt='.4f', legend_values=True):
    """Doughnut of the external variance split by age_error_budget, as both
    plotting tools show it.  scale (1 or 2) and sig_lbl ('1σ' / '2σ') set the
    display level.  Returns the external error at that level (Ma)."""
    total = sum(v for _, v in budget) or 1.0
    parts = [(lbl, v) for lbl, v in budget if v / total > 1e-6]
    fracs = [v / total for _, v in parts]
    wedges, _ = ax.pie(fracs, colors=BUDGET_COLORS[:len(parts)], startangle=90,
                       counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor='white'))
    ax.set_aspect('equal')
    ext = math.sqrt(total) * scale
    ax.text(0, 0, f'{sig_lbl} ext\n{ext:{centre_fmt}}\nMa', ha='center',
            va='center', fontsize=10, fontweight='bold')
    if legend_values:
        leg = [f'{lbl}:  {fr*100:.1f}%   ({sig_lbl} {math.sqrt(fr*total)*scale:.4f} Ma)'
               for (lbl, _), fr in zip(parts, fracs)]
    else:
        leg = [f'{lbl}:  {fr*100:.1f}%' for (lbl, _), fr in zip(parts, fracs)]
    ax.legend(wedges, leg, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              fontsize=8, frameon=False)
    return ext


# --------------------------------------------------------------------------
# Shared UI: force a light Fusion theme so the tools never follow macOS
# dark mode. Called from each tool's main(). Kept here so all three windows
# get the exact same palette and never drift apart.
# --------------------------------------------------------------------------

def force_light(app):
    """Force a light Fusion palette so the app never follows macOS dark mode."""
    from PyQt5.QtGui import QPalette, QColor
    app.setStyle('Fusion')
    p = QPalette()
    white = QColor('#ffffff')
    near = QColor('#f4f4f4')
    text = QColor('#1a1a1a')
    btn = QColor('#e9e9e9')
    hi = QColor('#3573b9')
    p.setColor(QPalette.Window, near)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, white)
    p.setColor(QPalette.AlternateBase, near)
    p.setColor(QPalette.ToolTipBase, white)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, btn)
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.BrightText, QColor('#d62728'))
    p.setColor(QPalette.Highlight, hi)
    p.setColor(QPalette.HighlightedText, white)
    p.setColor(QPalette.Disabled, QPalette.Text, QColor('#9a9a9a'))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor('#9a9a9a'))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor('#9a9a9a'))
    app.setPalette(p)
    app.setStyleSheet(
        'QWidget { color:#1a1a1a; }'
        'QLineEdit, QTextEdit, QPlainTextEdit { background:#ffffff; color:#1a1a1a;'
        ' border:1px solid #bbb; border-radius:3px; }'
        'QComboBox { background:#ffffff; color:#1a1a1a; border:1px solid #bbb;'
        ' border-radius:3px; padding:1px 4px; }'
        'QComboBox QAbstractItemView { background:#ffffff; color:#1a1a1a;'
        ' selection-background-color:#3573b9; selection-color:#ffffff; }'
        'QPushButton { background:#ededed; color:#1a1a1a; border:1px solid #b9b9b9;'
        ' border-radius:4px; padding:4px 8px; }'
        'QPushButton:hover { background:#e3e3e3; }'
        'QPushButton:checked { background:#cfe0f2; border-color:#3573b9; }'
        'QCheckBox { color:#1a1a1a; }'
        'QLabel { color:#1a1a1a; }'
        'QScrollArea { background:#f4f4f4; border:none; }'
    )


_MONO_FAMILIES = ('Menlo', 'Consolas', 'DejaVu Sans Mono', 'Liberation Mono',
                  'Courier New', 'Courier')


def mono_font(size=9):
    """The system's fixed-width font at the given point size; when the platform
    names none, the first installed of a few common ones."""
    from PyQt5.QtGui import QFont, QFontDatabase, QFontInfo
    f = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    if not QFontInfo(f).fixedPitch():
        db = QFontDatabase()
        families = db.families()
        family = (next((x for x in _MONO_FAMILIES if x in families), None)
                  or next((x for x in families if db.isFixedPitch(x)), None))
        if family:
            f = QFont(family)
    f.setPointSize(size)
    return f
