"""Calibrations, external errors, the recalculator, the standards library and
monitors other than Fish Canyon sanidine, checked against published values
and against tests/reference_framework.py, which is written independently from
the published equations.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_calibration.py
"""
import os
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('SIMPLAR_DATA_DIR', tempfile.mkdtemp())
os.environ.setdefault('SIMPLAR_STANDARDS_FILE', os.path.join(tempfile.mkdtemp(), 'standards.json'))
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import copy
import json
import math

import numpy as np

import arar_math as am
import reference_framework as ref
import standards as stdlib
from synthetic_data import load_ratios, CFG, CFG_R, LAM, PHILLIPS, FLUX_PCT

_APP = []


def _qt():
    """One QApplication for the GUI tests, kept alive."""
    from PyQt5.QtWidgets import QApplication
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


def _comp(comps, start):
    """The systematic component whose label starts with start."""
    d = dict(comps)
    return next(v for k, v in d.items() if k.startswith(start))


def close(a, b, tol, what):
    ok = abs(a - b) <= tol
    print(f"  {'ok ' if ok else 'BAD'} {what:<48} {a:.10g}  vs  {b:.10g}")
    assert ok, what


# --------------------------------------------------------------------------
# Validation tests for the external-error methods in arar_math.
# --------------------------------------------------------------------------

def _close(a, b, tol, msg):
    assert abs(a - b) <= tol, f"{msg}: got {a}, expected {b} (tol {tol})"
    print(f"  OK  {msg}: {a:.6g} ~ {b:.6g}")


def test_carter_spreadsheet():
    """rvalue_age_external reproduces the Carter (2025) mmc2.xlsx 'Linear' sheet."""
    print("Carter intercalibrated (spreadsheet mmc2.xlsx):")
    R, sR = 2.315973987133238, 0.000821191405421399
    d = am.rvalue_age_external(R, sR, t_std_ma=28.183, sigma_t_std_ma=0.017,
                               lam=5.5042e-10, sigma_lam=0.0054e-10)
    _close(d['age_ma'], 64.61713151529224, 1e-6, "age (Ma)")
    # sheet Z (full external, incl. covariance) = 0.044674 Ma; the independent
    # form omits the negligible covariance term, so match to <1e-4 Ma.
    _close(d['ext_ma'], 0.044674268944092724, 5e-5, "external Z (Ma)")
    # dominant share is the standard age, not lambda
    assert d['var_tstd'] > d['var_R'] > d['var_lambda']
    print(f"  OK  variance shares  FCs={d['var_tstd']:.3f} R={d['var_R']:.3f} "
          f"lambda={d['var_lambda']:.5f}")


def test_astronomical_covariance_zero():
    """Astronomical route: same partials, covariance term = 0, small lambda term."""
    print("Astronomical (Kuiper 2008, cov=0):")
    age = 64.6
    sig = am.rvalue_external_sigma(age, t_std_ma=28.201, sigma_t_std_ma=0.023,
                                   lam=5.463e-10, sigma_lam=0.054e-10,
                                   cov_lam_tstd=0.0)
    tt, lt = am.rvalue_external_terms(age, 28.201, 0.023, 5.463e-10, 0.054e-10)
    # the intercalibrated lambda term is ~1% of the full term
    # age*sigma_lambda/lambda (sensitivity ~0.01, not 1).
    full_term = age * (0.054e-10 / 5.463e-10)
    assert lt < 0.03 * full_term, f"lambda term too large: {lt} vs {full_term}"
    print(f"  OK  std-age term={tt:.4f} Ma, lambda term={lt:.5f} Ma; "
          f"lambda term is {lt/full_term:.3f} of the full term "
          f"({full_term:.4f} Ma); combined sys={sig:.4f} Ma")


def test_min_f239():
    """Min et al. (2000) full external reproduces F-239 = 1090.6 +- 23.3 Ma (2s)."""
    print("Min et al. (2000) full external, F-239:")
    # Paper: analytical/standard term +-10 Ma (2s) -> 5.0 Ma (1s).
    d = am.min_full_external(age_ma=1090.0, sigma_int_ma=0.0, sigma_std_ma=5.0,
                             decay_set='Audi et al. (1997)')
    decay_2s = 2 * d['decay_ma']
    ext_2s = 2 * d['ext_ma']
    print(f"  decay-constant only: {decay_2s:.1f} Ma (2s)")
    print(f"  full external:       {ext_2s:.1f} Ma (2s)")
    _close(decay_2s, 21.1, 1.5, "decay-constant 2s (Ma)")
    _close(ext_2s, 23.3, 1.5, "full external 2s (Ma)")
    # sanity: decay dominates, and lambda_e (e.c. branch) is the largest part
    assert d['le_ma'] > d['lb_ma'], "lambda_e should dominate over lambda_b"
    print(f"  OK  le term={d['le_ma']:.2f} Ma > lb term={d['lb_ma']:.2f} Ma (1s)")


def test_min_vs_intercalibrated_gap():
    """Same age, two conventions: Min full external >> intercalibrated external."""
    print("Convention gap at 65 Ma:")
    inter = am.rvalue_age_external(2.316, 0.00082, 28.183, 0.017,
                                   5.5042e-10, 0.0054e-10)['ext_ma']
    mn = am.min_full_external(64.6, sigma_std_ma=0.0,
                              decay_set='Audi et al. (1997)')['decay_ma']
    print(f"  intercalibrated (Carter) external = {inter*1000:.1f} kyr")
    print(f"  Min decay-constant external       = {mn*1000:.0f} kyr")
    assert mn > 10 * inter, "Min external should dwarf the intercalibrated one"
    print(f"  OK  ratio ~{mn/inter:.0f}x")


def test_external_dispatch():
    """external_error takes the right route for each calibration."""
    print("External-error routes (external_error):")
    carter = dict(stdAgeMa=28.183, stdAgeErrMa=0.017,
                  lambdaYr=5.5042e-10, lambdaErrYr=0.0054e-10)
    # The intercalibrated default matches rvalue_external_sigma with the joint covariance.
    b1 = am.external_error(64.6, carter, full_systematics=False)
    ref = am.rvalue_external_sigma(64.6, 28.183, 0.017, 5.5042e-10, 0.0054e-10,
                                   am.cfg_covariance(carter))
    _close(b1, ref, 1e-12, "default == intercalibrated")
    # The full external (Min) is far larger and independent of the pinned lambda.
    b2 = am.external_error(64.6, carter, full_systematics=True)
    assert b2 > 10 * b1, f"full external should be far larger: {b2} vs {b1}"
    print(f"  OK  intercalibrated={b1*1000:.1f} kyr  full external (Min)={b2*1000:.0f} kyr  "
          f"(~{b2/b1:.0f}x)")
    # astronomical -> cov 0
    kuiper = dict(calibClass='astronomical', stdAgeMa=28.201, stdAgeErrMa=0.023,
                  lambdaYr=5.463e-10, lambdaErrYr=0.054e-10)
    _, cov = am.calibration_box(kuiper)
    assert cov == 0.0, "astronomical covariance must be 0"
    print("  OK  astronomical cov=0, joint cov!=0:",
          am.calibration_box(carter)[1] != 0.0)


def test_full_systematics_warning():
    """Min-over-pinned-lambda raises a coherence warning; astronomical does not."""
    print("Coherence warning:")
    carter = dict(calibClass='joint', stdAgeMa=28.183, lambdaYr=5.5042e-10)
    kuiper = dict(calibClass='astronomical', stdAgeMa=28.201, lambdaYr=5.463e-10)
    assert am.full_systematics_warning(carter), "should warn for Carter+Min"
    assert not am.full_systematics_warning(kuiper), "should not warn for astro"
    print("  OK  warns for joint (Carter), silent for astronomical")


def test_full_systematics_enlarges_external():
    """external_sigma_array with fullSystematics on is far larger than the
    intercalibrated default."""
    print("external_sigma_array, full external vs default:")
    cfg = dict(calibClass='joint', stdAgeMa=28.183, stdAgeErrMa=0.017,
               lambdaYr=5.5042e-10, lambdaErrYr=0.0054e-10)
    ages = [64.6, 30.0]
    sig_int = [0.02, 0.01]
    base = am.external_sigma_array(ages, sig_int, cfg)
    cfg_on = dict(cfg, fullSystematics=True)
    on = am.external_sigma_array(ages, sig_int, cfg_on)
    assert on[0] > 5 * base[0], "fullSystematics should enlarge the external"
    print(f"  OK  default[0]={base[0]*1000:.1f} kyr -> "
          f"fullSystematics[0]={on[0]*1000:.0f} kyr")


def test_anchored_external_reproduces_kuiper():
    """Kuiper et al. (2008) SOM states sigma(t_FCs) = 0.023 / 0.044 / 0.087 Ma
    (1s) for sigma_t_astro = 5 / 10 / 20 kyr. Reproducing all three is what
    proves the partials are literally theirs."""
    print("Anchored external vs Kuiper et al. (2008) SOM:")
    lam, s_lam = 5.463e-10, 0.107e-10
    for s_astro, expect in ((0.005, 0.023), (0.010, 0.044), (0.020, 0.087)):
        anchor = (6.500, s_astro, 4.3664, 0.0009)
        got = am.rvalue_external_sigma_anchored(28.201, anchor, lam, s_lam)[0]
        assert abs(got - expect) < 6e-4, (s_astro, got, expect)
        print(f"  OK  sigma_t_astro {s_astro*1000:4.0f} kyr -> "
              f"{got:.4f} Ma ~ {expect}")


def test_every_anchor_reproduces_its_own_paper():
    """Each FC_ANCHORS entry must reproduce the FCs age and the external
    uncertainty its own paper publishes. Guards against mixing up tuffs
    (Melilla vs A1/Faneromeni) or sigma levels (Kuiper 1s vs Phillips 2s)."""
    lam, s_lam = 5.463e-10, 0.054e-10          # Min et al. (2000), 1s
    print("Each anchor vs its own published FCs age and external (1s):")
    cases = [('Kuiper et al. (2008)',   28.201, 0.023),
             ('Rivera et al. (2011)',   28.172, 0.014),
             ('Phillips et al. (2022)', 28.176, 0.0115)]
    for name, age_pub, ext_pub in cases:
        anchor = am.FC_ANCHORS[name]
        got, tt, lt, rt = am.rvalue_external_sigma_anchored(
            age_pub, anchor, lam, s_lam)
        assert abs(got - ext_pub) < 1.2e-3, (name, got, ext_pub)
        # the preset age must agree with what the anchor implies
        implied = next(a for n, a, e, x in am.FC_PRESETS if n == name)
        assert abs(implied - age_pub) < 1e-3, (name, implied, age_pub)
        print(f"  OK  {name:<24} ext {got:.4f} ~ {ext_pub}   "
              f"[R {rt*1e3:.1f}, t_astro {tt*1e3:.1f}, lam {lt*1e3:.1f} kyr]")
    # lambda is always small but never zero
    for name, _, _ in cases:
        _, _, lt, _ = am.rvalue_external_sigma_anchored(
            28.18, am.FC_ANCHORS[name], lam, s_lam)
        assert 1e-4 < lt < 5e-3, (name, lt)
    print("  OK  lambda term nonzero and < 5 kyr for all three")


def test_anchor_beats_uncorrelated_route():
    """Anchoring on t_astro gives a slightly larger systematic than the
    single-step t_FCs route: the λ dependence of the anchor chain adds to that
    of the age equation (Renne et al. 1998, p. 120). Same sign, small size."""
    lam, s_lam = 5.463e-10, 0.107e-10
    anchor = (6.500, 0.005, 4.3664, 0.0009)
    age = 56.0667
    anch = am.rvalue_external_sigma_anchored(age, anchor, lam, s_lam)[0]
    legacy = am.rvalue_external_sigma(age, 28.201, 0.023, lam, s_lam, 0.0)
    assert anch > legacy, (anch, legacy)
    assert (anch - legacy) * 1e3 < 2.0, (anch, legacy)
    print("Anchored vs uncorrelated single-step:")
    print(f"  OK  anchored {anch*1e3:.2f} kyr > single-step {legacy*1e3:.2f} kyr "
          f"(by {(anch-legacy)*1e3:.2f} kyr, 1s)")


def test_external_error_uses_anchor_when_available():
    """external_error must take the anchored route for a cfg naming a
    calibration with a registered anchor, and the legacy route otherwise."""
    lam, s_lam = 5.463e-10, 0.107e-10
    base = dict(lambdaYr=lam, lambdaErrYr=s_lam, calibClass='astronomical')
    with_anchor = dict(base, fcName='Kuiper et al. (2008)',
                       stdAgeMa=28.201, stdAgeErrMa=0.023)
    without = dict(base, fcName='Renne et al. (1998)  [S&J λ]',
                   stdAgeMa=28.02, stdAgeErrMa=0.14)
    a = am.external_error(56.0, with_anchor)
    b = am.external_error(56.0, without)
    exp_a = am.rvalue_external_sigma_anchored(
        56.0, am.FC_ANCHORS['Kuiper et al. (2008)'], lam, s_lam)[0]
    exp_b = am.rvalue_external_sigma(56.0, 28.02, 0.14, lam, s_lam, 0.0)
    assert abs(a - exp_a) < 1e-12, (a, exp_a)
    assert abs(b - exp_b) < 1e-12, (b, exp_b)
    print("external_error routing:")
    print(f"  OK  Kuiper (anchor known)   -> anchored  {a*1e3:.2f} kyr")
    print(f"  OK  Renne 1998 (no anchor)    -> legacy    {b*1e3:.2f} kyr")


def test_anchor_reproduces_its_preset_fc_age():
    """Every anchor must reproduce, through the age equation, exactly the FCs age
    its own preset advertises. Guards against the age and the uncertainty being
    computed from two slightly different FCs ages (a 0.5 kyr rounding mismatch
    in R_FCs/astro shows up as ~1 kyr at 55 Ma)."""
    import math as _m
    lam = 5.463e-10
    print("Anchor vs preset FCs age:")
    for name, anchor in am.FC_ANCHORS.items():
        t_astro, _, r, _ = anchor
        implied = _m.log(1 + r * (_m.exp(lam * t_astro * 1e6) - 1)) / lam / 1e6
        preset = next(a for n, a, e, x in am.FC_PRESETS if n == name)
        assert abs(implied - preset) < 5e-5, (name, implied, preset)
        print(f"  OK  {name:<24} anchor -> {implied:.6f} == preset {preset}")


def test_single_step_route_full_lambda():
    """A standard age that scales as 1/λ: the λ term is the total derivative
    dt/dλ = −t/λ, i.e. t·σλ/λ at any age, checked by moving λ and the standard
    age together."""
    import math as _m
    lam, slam, t_std = 5.543e-10, 0.0554e-10, 28.02
    cfg = dict(stdAgeMa=t_std, stdAgeErrMa=0.14, lambdaYr=lam, lambdaErrYr=slam,
               stdAgeErrType='internal', calibClass='other')

    def age(Q, l):
        return _m.log1p(Q * _m.expm1(l * (t_std * lam / l) * 1e6)) / l / 1e6

    for t in (1.0, 55.0, 500.0, 2000.0):
        Q = _m.expm1(lam * t * 1e6) / _m.expm1(lam * t_std * 1e6)
        h = 1e-6
        num = (age(Q, lam * (1 + h)) - age(Q, lam * (1 - h))) / (2 * h * lam) * slam
        got = _comp(am.systematic_components(t, cfg), 'λ, ')
        _close(got, num, 1e-6, f'lambda term at {t} Ma')
        _close(got, -t * slam / lam, 1e-12, f'equals -t·σλ/λ at {t} Ma')
        print(f"  OK  {t:7.1f} Ma: λ term {got:.5f} Ma = −t·σλ/λ")


# --------------------------------------------------------------------------
# Checks the toolchain against tests/reference_framework.py, which is
# written independently from the published equations.
# --------------------------------------------------------------------------

def test_anchor_reproduces_presets():
    print('anchors return their published FCs ages at the native lambda:')
    for name, age, err, _ in am.FC_PRESETS:
        if name in am.FC_ANCHORS:
            close(am.effective_fc_age(dict(stdAgeMa=age, stdAgeErrMa=err,
                                           lambdaYr=LAM)), age, 1e-9, name)
            close(am.fc_ingrowth(age, LAM, err), math.expm1(LAM * age * 1e6),
                  1e-15, name + ' ingrowth = e^(lam t)-1')
            anc = am.FC_ANCHORS[name]
            r = ref.anchor(name)
            for x, y in zip(anc, r):
                close(x, y, 1e-12, name + ' anchor tuple')


def test_single_age_tiers():
    print('single age, synthetic R, all three tiers:')
    R_ = 2.0
    t = ref.age(R_, ref.anchor(PHILLIPS))
    (ages, _s), err = am.r_to_age_cfg([R_], [0.0], CFG_R)
    ages = float(ages[0])
    close(ages * 1e6, t, 1e-4, 'age (yr)')
    rel_ana = 0.0004 / R_
    rel_flux = FLUX_PCT / 100
    a, i, e, comps, _ = ref.tiers(t, rel_ana, rel_flux, ref.anchor(PHILLIPS))
    s_int = am.internal_sigma_array([ages], [am.dt_dlnR(ages, LAM) * rel_ana], CFG_R)[0]
    s_ext = am.external_sigma_array([ages], [s_int], CFG_R)[0]
    close(s_int * 1e6, i, 1e-4, 'internal 1s (yr)')
    close(s_ext * 1e6, e, 1e-4, 'external 1s (yr)')
    got = dict(am.systematic_components(ages, CFG))
    close(_comp(got, 'R(FCs/') * 1e6, comps['R_cal'], 1e-6, 'R cal comp')
    close(_comp(got, 'A1 astronomical age') * 1e6, comps['t_A1'], 1e-6, 't_A1 comp')
    close(_comp(got, 'λ, ') * 1e6, comps['lam'], 1e-6, 'lambda comp')
    b = am.age_error_budget(0.0, s_int, s_ext, ages, CFG)
    close(sum(v for _, v in b), s_ext**2, 1e-18, 'budget sums to ext^2')


def test_stack():
    print('43 synthetic analyses, WM with exact shared propagation:')
    R, sR, F, sF = load_ratios()
    n = len(R)
    (ages, s_ana), _ = am.r_to_age_cfg(R, sR, CFG_R)
    (agesF, s_anaF), _ = am.r_to_age_cfg(F, sF, CFG)
    close(float(np.max(np.abs(agesF - ages))) * 1e6, 0.0, 1e-3,
          'F and R routes agree, max |dt| yr')
    s_int = am.internal_sigma_array(ages, s_ana, CFG_R)
    s_ext = am.external_sigma_array(ages, s_int, CFG_R)
    C, labels = am.systematic_matrix(ages, CFG_R)
    res = am.combine_ages(ages, s_ana, s_int, s_ext, position_group=[0] * n,
                          sys_components=C, scale_overdispersion=False)
    anc = ref.anchor(PHILLIPS)
    t_ref = [ref.age(r, anc) for r in R]
    ana_ref = [ref.partials(t, anc)[0] * s / r for t, s, r in zip(t_ref, sR, R)]
    flux_ref = [ref.partials(t, anc)[0] * FLUX_PCT * 1e-2 for t in t_ref]
    comps_ref = [ref.tiers(t, 0, 0, anc)[3] for t in t_ref]
    m, a, i, e = ref.combine_shared(t_ref, ana_ref, flux_ref, comps_ref)
    close(res['wm'] * 1e6, m, 1e-3, 'WM (yr)')
    close(res['sigma_ana'] * 1e6, a, 1e-4, 'ana 1s (yr)')
    close(res['sigma_int'] * 1e6, i, 1e-4, 'int 1s (yr)')
    close(res['sigma_ext'] * 1e6, e, 1e-4, 'ext 1s (yr)')
    rv = am.combine_ages_vermeesch(ages, s_ana, s_int, s_ext, position_group=[0] * n,
                                   sys_components=C, random_effects=False)
    # GLS on the full covariance: must agree with the analytical-weight mean to
    # within its own precision, and give tiers no larger
    print(f"  GLS: wm {rv['wm']:.8f}  ana {2e3*rv['sigma_ana']:.3f}  "
          f"int {2e3*rv['sigma_int']:.3f}  ext {2e3*rv['sigma_ext']:.3f} kyr")
    assert abs(rv['wm'] - res['wm']) <= res['sigma_ana'], (rv['wm'], res['wm'])
    assert rv['sigma_ext'] <= res['sigma_ext'] * (1 + 1e-9), (rv['sigma_ext'], res['sigma_ext'])


def test_other_lambda():
    print('anchored preset with a non-native lambda: age follows the chain')
    cfg = dict(CFG, lambdaYr=5.5305e-10, lambdaErrYr=0.0132e-10)
    anc = ref.anchor(PHILLIPS)
    (a, _s), _ = am.r_to_age_cfg([2.0], [0.0], dict(cfg, ageFluxMode='ratio_R_FCs'))
    a = float(a[0])
    close(a * 1e6, ref.age(2.0, anc, 5.5305e-10), 1e-4, 'age (yr)')


def test_rr_roundtrip():
    print('rr_from_age inverts r_to_age_cfg:')
    for lam in (LAM, 5.5305e-10):
        cfg = dict(CFG, ageFluxMode='ratio_R_FCs', lambdaYr=lam)
        (a, _s), _ = am.r_to_age_cfg([0.345], [0.0], cfg)
        a = float(a[0])
        rr, _ = am.rr_from_age(a, cfg)
        close(rr, 0.345, 1e-12, f'lam={lam}')


# --------------------------------------------------------------------------
# Recalculator maths and the standards library.
# --------------------------------------------------------------------------

def cfg(name, full=False):
    n, a, e, x = next(r for r in am.FC_PRESETS if r[0] == name)
    pl = am.paired_lambda(name)
    return dict(stdAgeMa=a, stdAgeErrMa=e, stdAgeErrType='external' if x else 'internal',
                lambdaYr=pl[1], lambdaErrYr=pl[2], calibClass=am.calibration_class(name),
                fullSystematics=full)


def test_library_reproduces_published_ages():
    # Renne et al. (2010) Table 5 and Renne et al. (2011) Table 2: R -> age
    c10, c11 = cfg('Renne et al. (2010)'), cfg('Renne et al. (2011)')
    for R, a10, a11 in [(0.04229, 1.2061, 1.2056), (1.0112, 28.619, 28.608),
                        (3.5958, 99.769, 99.738), (51.878, 1080.4, 1081.0),
                        (211.42, 2648.1, 2651.8)]:
        close(am.standard_age_on(c10, R), a10, 0.6 * 10 ** -(len(str(a10).split('.')[1])), f'R={R} on 2010')
        close(am.standard_age_on(c11, R), a11, 0.6 * 10 ** -(len(str(a11).split('.')[1])), f'R={R} on 2011')
    # Renne et al. (1998): ages on FCs 28.02 Ma (rounded in the paper)
    c98 = cfg('Renne et al. (1998)  [S&J λ]')
    for R, age, tol in [(21.4876, 523.1, 0.1), (3.5957, 98.79, 0.01), (0.04229, 1.194, 0.001),
                        (3.8367, 105.2, 0.05), (1.0112, 28.34, 0.01)]:
        close(am.standard_age_on(c98, R), age, tol, f'R={R} on Renne 1998')
    # and FCs from GA1550 at 98.79 Ma, the way Renne et al. (1998) derived it
    lam = 5.543e-10
    close(math.log1p(math.expm1(lam * 98.79e6) / 3.5957) / lam / 1e6, 28.02, 0.005, 'FCs from GA1550')
    # Niespolo et al. (2017): ACs on FCs 28.201 Ma (Kuiper et al. 2008) = 1.1849 Ma
    close(am.standard_age_on(cfg('Kuiper et al. (2008)'), 0.041707), 1.1849, 0.0001, 'ACs on Kuiper')


def test_library_entries():
    lib = stdlib.load_default()
    assert not stdlib.check(lib), stdlib.check(lib)
    ids = [s['id'] for s in lib['standards']]
    for sid in ('FCs', 'ACs', 'TCs', 'GA1550', 'MMhb-1', 'Hb3gr', 'A1'):
        assert sid in ids, sid
    for s in lib['standards']:
        if s['id'] == 'FCs':                    # R(FCs/FCs) = 1 needs no entry
            assert stdlib.recommended_R(s) is None and s['published_ages']
            continue
        rec = stdlib.recommended_R(s)
        assert rec is not None and rec[0] > 0, s['id']
    # the presets used by the tools come from the library
    assert [r[0] for r in am.FC_PRESETS] == [c['name'] for c in lib['calibrations']
                                            if c.get('standard', 'FCs') == 'FCs']
    assert am.DEFAULT_FC_NAME == 'Phillips et al. (2022)'
    # the anchors give back their published FCs ages at Min et al. (2000)
    for name, anc in am.FC_ANCHORS.items():
        age = next(a for n, a, e, x in am.FC_PRESETS if n == name)
        close(am.effective_fc_age(cfg(name)), age, 1e-9, name)


def test_save_ignores_published_changes():
    """An entry left as it was when the editor opened is not an edit, even if
    the published file changed in the meantime."""
    import copy, tempfile
    base = stdlib.load_default()
    lib = copy.deepcopy(base)
    newer = copy.deepcopy(base)
    next(x for x in newer['standards'] if x['id'] == 'Hb3gr')['locality'] = 'changed'
    next(x for x in lib['standards'] if x['id'] == 'TCs')['notes'] = 'my note'
    old_path, old_load = stdlib.user_path, stdlib.load_default
    d = tempfile.mkdtemp()
    stdlib.user_path = lambda: os.path.join(d, 'standards.json')
    stdlib.load_default = lambda: copy.deepcopy(newer)
    try:
        stdlib.save(lib, base)
        saved = json.load(open(stdlib.user_path()))
    finally:
        stdlib.user_path, stdlib.load_default = old_path, old_load
    assert [x['id'] for x in saved['standards']] == ['TCs'], saved['standards']


def test_identity_and_round_trip():
    ph = cfg('Phillips et al. (2022)')
    r = am.recalculate(55.8, 28.176, ph['lambdaYr'], ph)
    close(r['age'], 55.8, 1e-9, 'identity')
    close(r['factor'], 1.0, 1e-9, 'identity factor')
    assert r['components'] == am.systematic_components(r['age'], ph)
    # Renne 1998 (S&J) -> Carter -> back
    c98, car = cfg('Renne et al. (1998)  [S&J λ]'), cfg('Carter et al. (2025)')
    a = am.recalculate(65.5, 28.02, c98['lambdaYr'], car)['age']
    b = am.recalculate(a, car['stdAgeMa'], car['lambdaYr'], c98)['age']
    close(b, 65.5, 1e-9, 'round trip')
    # an independent evaluation of the recast
    lam0, lam = 5.543e-10, car['lambdaYr']
    Ru = math.expm1(lam0 * 65.5e6) / math.expm1(lam0 * 28.02e6)
    close(a, math.log1p(Ru * math.expm1(lam * car['stdAgeMa'] * 1e6)) / lam / 1e6, 1e-9, 'by hand')


def test_monitor_own_age():
    ph = cfg('Phillips et al. (2022)')
    R = 0.041707
    r = am.recalculate(1.194, 1.194, 5.543e-10, ph, R, 0.000011, 'R(ACs)')
    close(r['age'], am.standard_age_on(ph, R), 1e-12, 'ACs own age')
    comp = dict(r['components'])['R(ACs)']
    close(comp, am.dt_dlnR(r['age'], ph['lambdaYr']) * 0.000011 / R, 1e-15, 'R component')


def test_factor_is_derivative():
    tgt = cfg('Kuiper et al. (2008)')
    h = 1e-6
    for t in (0.5, 28.0, 250.0, 1800.0):
        r = am.recalculate(t, 1.193, 5.543e-10, tgt, 0.0417, 0.0)
        up = am.recalculate(t + h, 1.193, 5.543e-10, tgt, 0.0417, 0.0)['age']
        dn = am.recalculate(t - h, 1.193, 5.543e-10, tgt, 0.0417, 0.0)['age']
        close(r['factor'], (up - dn) / (2 * h), 1e-6, f'factor at {t}')


def test_combine_shares_calibration():
    ph = cfg('Phillips et al. (2022)')
    rs = [am.recalculate(56.0, 28.02, 5.543e-10, ph), am.recalculate(56.0, 28.02, 5.543e-10, ph)]
    labels = [lbl for lbl, _ in rs[0]['components']]
    C = np.array([[v for _, v in r['components']] for r in rs])
    sys1 = math.sqrt((C[0] ** 2).sum())
    a = np.array([0.02, 0.02])
    res = am.combine(am.METHOD_WM, [rs[0]['age'], rs[1]['age']], a, a, np.hypot(a, sys1),
                     standard_group=[0, 0], sys_components=C, p_corr=np.eye(2))
    close(res['wm'], rs[0]['age'], 1e-12, 'mean')
    close(res['sigma_ana'], 0.02 / math.sqrt(2), 1e-12, 'analytical averages down')
    close(res['sigma_ext'] ** 2 - res['sigma_int'] ** 2, sys1 ** 2, 1e-12, 'calibration does not')
    assert len(labels) == C.shape[1]


def test_lambda_kept_with_another_lambda():
    """An astronomical or joint FCs age paired with another λ keeps its λ term
    (on the R-value form), it is not dropped."""
    for age, err, lam, s_lam in [(28.176, 0.02, 5.5305e-10, 0.0132e-10),   # Phillips, edited error
                                 (28.183, 0.017, 5.463e-10, 0.054e-10)]:    # Carter age, Min λ
        c = dict(stdAgeMa=age, stdAgeErrMa=err, stdAgeErrType='external', lambdaYr=lam,
                 lambdaErrYr=s_lam)
        assert am.calibration_box(c)[0] in ('astronomical', 'joint'), c
        lam_term = _comp(am.systematic_components(56.0, c), 'λ, ')
        exp = am._rvalue_partials(56.0, age, lam)[1] * s_lam / 1e6
        close(lam_term, exp, 1e-12, f'λ term for {age}')
        assert abs(lam_term) > 0
    # the Carter covariance only with Carter's own λ
    assert am.cfg_covariance(dict(stdAgeMa=28.183, lambdaYr=5.463e-10)) == 0.0
    assert am.cfg_covariance(dict(stdAgeMa=28.183, lambdaYr=5.5042e-10)) == am.CARTER_COV_LAM_FCS


def test_ratio_orientation():
    a1 = stdlib.find(stdlib.load_default(), 'standards', 'A1')
    R, s, ic = stdlib.recommended_R(a1)
    assert ic['ratio'] == 'FCs/std' and ic['R'] == 4.0819
    close(R, 1 / 4.0819, 1e-15, 'A1/FCs')
    close(s, 0.0007 / 4.0819 ** 2, 1e-15, 'its 1σ')
    # the A1 R gives back the A1 age on Phillips et al. (2022), to the rounding of
    # the printed R (the preset uses the R that reproduces 28.176 Ma exactly)
    close(am.standard_age_on(cfg('Phillips et al. (2022)'), R), 6.943, 0.0002, 'A1 on Phillips')


def test_user_edits_merge():
    tmp = tempfile.mkdtemp()
    os.environ['SIMPLAR_STANDARDS_FILE'] = os.path.join(tmp, 'standards.json')
    lib = stdlib.load()
    acs = stdlib.find(lib, 'standards', 'ACs')
    acs['intercalibrations'][0]['R'] = 0.0417
    lib['calibrations'] = [c for c in lib['calibrations'] if not c['name'].startswith('Historical')]
    stdlib.save(lib)
    back = stdlib.load()
    assert stdlib.find(back, 'standards', 'ACs')['intercalibrations'][0]['R'] == 0.0417
    assert stdlib.find(back, 'standards', 'ACs')['edited']
    assert not any(c['name'].startswith('Historical') for c in back['calibrations'])
    assert len(back['standards']) == len(stdlib.load_default()['standards'])
    stdlib.reset()
    assert stdlib.load() == stdlib.merge(stdlib.load_default(), {})


# --------------------------------------------------------------------------
# Monitors other than Fish Canyon sanidine.
# --------------------------------------------------------------------------

PH = dict(stdAgeMa=28.176, stdAgeErrMa=0.0115, stdAgeErrType='external',
          lambdaYr=5.463e-10, lambdaErrYr=0.054e-10)


def acs(**kw):
    c = dict(PH, **am.monitor_keys('ACs'))
    c.update(kw)
    return c


def test_keys_and_age():
    k = am.monitor_keys('ACs')
    assert k == dict(monitor='ACs', monitorR=0.041707, monitorRErr=0.000011)
    assert am.cfg_monitor(PH) == ('FCs', 1.0, 0.0)
    close(am.monitor_age(acs()), am.standard_age_on(PH, 0.041707), 1e-15, 'ACs age')
    # a monitor age as files and papers quote it maps back onto the FCs age
    t_m = am.monitor_age(acs())
    close(am.fcs_equivalent_age(t_m, PH['lambdaYr'], 0.041707), 28.176, 1e-9, 'FCs equivalent')


def test_j_from_measured_monitor_ratio():
    """J = (e^{λ t_ACs} − 1)/R_ACs, the same J from the FCs-equivalent R."""
    R_meas = 1.7
    cfg = acs(ageFluxMode='ratio_R_std', rStd=am.to_fcs_r(R_meas, acs()), rStdErrPct=0.1)
    J, rel = am.cfg_J(cfg)
    t_acs = am.monitor_age(cfg)
    close(J, math.expm1(PH['lambdaYr'] * t_acs * 1e6) / R_meas, 1e-15, 'J')
    close(am.from_fcs_r(cfg['rStd'], cfg), R_meas, 1e-12, 'round trip')
    # an absolute R error is in the measured ratio's units
    cfg2 = dict(cfg, rStdErrMode='abs', rStdErrPct=0.0017)
    close(am.cfg_j_err_frac(cfg2), 0.0017 / R_meas, 1e-12, 'relative J error')


def test_ratio_to_monitor_mode():
    """R(unk)/R(std) relative to ACs gives the age of R × R(ACs/FCs) relative to FCs."""
    r_acs = 3.0
    (a1, _), _ = am.r_to_age_cfg([r_acs], [0.0], acs(ageFluxMode='ratio_R_FCs'))
    (a2, _), _ = am.r_to_age_cfg([r_acs * 0.041707], [0.0], dict(PH, ageFluxMode='ratio_R_FCs'))
    close(a1[0], a2[0], 1e-12, 'age')
    rr, _ = am.rr_from_age(a1[0], acs())
    close(rr, r_acs, 1e-9, 'inverse')


def test_external_term():
    comps = dict(am.systematic_components(1.2, acs()))
    close(_comp(comps, 'R(ACs/FCs)'), am.dt_dlnR(1.2, PH['lambdaYr']) * 0.000011 / 0.041707, 1e-15, 'R term')
    assert not any(k.startswith('R(ACs/FCs)') for k in dict(am.systematic_components(1.2, PH)))
    assert 'monitor ACs' in am.external_route(acs())
    # shared between ages on the same monitor when combining
    C, labels = am.systematic_matrix([1.2, 1.3], acs())
    lab = next(k for k in labels if k.startswith('R(ACs/FCs)'))
    assert (C[:, labels.index(lab)] > 0).all()


def test_converter_monitor():
    import importlib.util
    _qt()
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        'dcm', os.path.join(ROOT, '1_data_converter', 'data_converter.py'))
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    rows = [{'Sample': 'S1', 'Run_ID': f'1-0{k}', 'Age': 1.0, 'Age_Er': 0.01,
             'Ar39_Moles': 1e-15, 'J': 0.0004, 'J_Er': 4e-7, 'Ar40Rad_Over_Ar39': 1.4,
             'Ar40Rad_Over_Ar39_Er': 0.002, 'PctAr40Rad': 95, 'Ca_Over_K': 1,
             'Ca_Over_K_Er': 0.1, 'Filename': f'f{k}', 'Irrad': 'X1'} for k in range(3)]
    out = tempfile.mkdtemp()
    t_acs = am.monitor_age(acs())
    written, warn = dc._convert_mass_spec_raw(pd.DataFrame(rows), out, {}, t_acs, 0.0115,
                                              monitor='ACs')
    cfg = json.load(open(written[0], encoding='utf-8'))['cfg']
    assert cfg['monitor'] == 'ACs' and not warn
    close(cfg['stdAgeMa'], 28.176, 1e-9, 'FCs-equivalent age')
    # the FCs-equivalent R gives back the file's J on the file's own constants
    J = am.cfg_J(dict(cfg, ageFluxMode='ratio_R_std'))[0]
    close(J / 0.0004, 1.0, 1e-6, 'J recovered')


def test_calibration_notes():
    # a preset age with another λ: the anchor gives another FCs age
    n = am.calibration_notes(dict(PH, lambdaYr=5.543e-10))
    assert len(n) == 1 and 'anchor gives FCs = 28.17' in n[0], n
    # an astronomical age with an edited error: the anchor is not used
    n = am.calibration_notes(dict(PH, stdAgeErrMa=0.02))
    assert len(n) == 1 and 'not the astronomical anchor' in n[0], n
    # the usual path: nothing
    assert am.calibration_notes(PH) == []
    assert am.monitor_mismatch_note(acs(), 'ACs') == ''
    assert 'was measured on FCs' in am.monitor_mismatch_note(acs(), 'FCs', 'the .jflux')
    assert am.edited_standards_note() == ''


def test_manual_flux_in_combine():
    import importlib.util
    _qt()
    spec = importlib.util.spec_from_file_location(
        'rcm', os.path.join(ROOT, '6_recalculator', 'recalculator.py'))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    c = rc.RecalculatorWindow().combine_tab
    c.table.setCurrentCell(0, 0)
    c.table.paste('A\t55.95\t0.02\t\nB\t56.01\t0.03\t\nC\t55.99\t0.03\t')
    host = rc._RecalcFluxHost(c)
    for r, v in enumerate([('J', 0.0027, 2.7e-6), ('J', 0.0027, 2.7e-6), ('R', 40.0, 0.08)]):
        host.set_manual(r, r, v)
    c.recalc()
    a = c.items[0]
    close(a['int'], math.hypot(a['ana'], am.dt_dlnR(a['age'], PH['lambdaYr']) * 0.001), 1e-12,
          'internal from the manual J')
    rho = c.result['rho']
    assert rho[0, 1] == 1.0 and rho[0, 2] == 0.0 and rho[1, 2] == 0.0, rho
    assert c._j_groups(rho) == ['Same J for A, B.'], c._j_groups(rho)


def test_combine_flux_session_and_analytical_from_internal():
    import importlib.util
    _qt()
    spec = importlib.util.spec_from_file_location(
        'rcm2', os.path.join(ROOT, '6_recalculator', 'recalculator.py'))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    c = rc.RecalculatorWindow().combine_tab
    c.table.setCurrentCell(0, 0)
    # B has only ± internal; its J error comes from the .jflux hole
    c.table.paste('A\t55.95\t0.02\t\nB\t56.01\t\t0.05')
    c.disk_flux.update({('I1', 'D1'): am.disk_flux_entry(
        ['1', '2'], [0.0027, 0.0027], [2.7e-6, 2.7e-7], [[1, 0.5], [0.5, 1]])})
    c.overrides = {0: ('I1', 'D1', '1'), 1: ('I1', 'D1', '2')}
    c.recalc()
    b = c.items[1]
    p = am.dt_dlnR(b['age'], PH['lambdaYr']) * 1e-4
    assert 0.0245 < b['int'] < 0.0255, b['int']   # input at 2σ
    close(b['ana'], math.sqrt(b['int'] ** 2 - p ** 2), 1e-12, 'analytical = internal minus J')
    assert c.result['rho'][0, 1] == 0.5
    st = c.state()
    import json
    st = json.loads(json.dumps(st))
    c2 = rc.RecalculatorWindow().combine_tab
    c2.set_state(st)
    assert ('I1', 'D1') in c2.disk_flux and c2.match(1) is not None
    close(c2.result['sigma_int'], c.result['sigma_int'], 1e-12, 'session keeps the flux')


def test_converter_step_order():
    """Steps are labelled in run order even when the file lists them otherwise
    (the last, fusion run first)."""
    import importlib.util
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        'dco', os.path.join(ROOT, '1_data_converter', 'data_converter.py'))
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    runs = [(30.0, 7, 55.65), (0.5, 0, 54.95), (0.7, 1, 55.05), (1.1, 2, 55.39)]
    rows = [{'Sample': 'S1', 'Run_ID': f'100-01{"ABCDEFGH"[k]}', 'Age': age, 'Age_Er': 0.1,
             'Ar39_Moles': 1e-15, 'J': 0.0027, 'J_Er': 2.7e-6, 'Ar40Rad_Over_Ar39': 11.3,
             'Ar40Rad_Over_Ar39_Er': 0.02, 'PctAr40Rad': 95, 'Ca_Over_K': 1,
             'Ca_Over_K_Er': 0.1, 'Filename': f'f{k}', 'Irrad': 'X1', 'Pwr_Requested': pwr,
             'Run_Date': '2026-07-24', 'Run_Hour': 4 + k / 4}
            for pwr, k, age in runs]
    out = tempfile.mkdtemp()
    written, warn = dc._convert_mass_spec_raw(pd.DataFrame(rows), out, {}, 28.176, 0.0115)
    steps = json.load(open(written[0], encoding='utf-8'))['steps']
    assert [s['label'] for s in steps] == ['A', 'B', 'C', 'D']
    assert [s['power'] for s in steps] == [0.5, 0.7, 1.1, 30.0], [s['power'] for s in steps]
    assert steps[-1]['age'] == 55.65
    # no run times: file order, and a missing power is the step's position
    for r in rows:
        del r['Run_Date'], r['Run_Hour'], r['Pwr_Requested']
    written, warn = dc._convert_mass_spec_raw(pd.DataFrame(rows), tempfile.mkdtemp(), {},
                                              28.176, 0.0115)
    steps = json.load(open(written[0], encoding='utf-8'))['steps']
    assert [s['power'] for s in steps] == [1.0, 2.0, 3.0, 4.0]
    assert steps[0]['age'] == 55.65


def test_plateau_apply_typed_flux_to_all():
    """'Apply flux to all aliquots' with a typed R standard copies it to every
    aliquot, not only disk-flux holes."""
    import importlib.util
    import pandas as pd
    _qt()
    spec = importlib.util.spec_from_file_location(
        'dca', os.path.join(ROOT, '1_data_converter', 'data_converter.py'))
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    rows = [{'Sample': f'S{a}', 'Run_ID': f'10{a}-01{"ABCD"[k]}', 'Age': 55.0 + 0.01 * k,
             'Age_Er': 0.1, 'Ar39_Moles': 1e-15, 'J': 0.0027, 'J_Er': 2.7e-6,
             'Ar40Rad_Over_Ar39': 11.3, 'Ar40Rad_Over_Ar39_Er': 0.02, 'PctAr40Rad': 95,
             'Ca_Over_K': 1, 'Ca_Over_K_Er': 0.1, 'Filename': f'f{a}{k}', 'Irrad': 'X1',
             'Pwr_Requested': k + 1.0}
            for a in range(3) for k in range(4)]
    out = tempfile.mkdtemp()
    written, _ = dc._convert_mass_spec_raw(pd.DataFrame(rows), out, {}, 28.176, 0.0115)
    spec = importlib.util.spec_from_file_location(
        'pia', os.path.join(ROOT, '4_plateau_isochron', 'plateau_isochron.py'))
    pi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pi)
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    w = pi.PlateauWindow()
    pi.dfold.get_open_file_names = lambda *a, **k: (sorted(written), '')
    w.load_session()
    assert len(w._carousel) == 3
    w._sp_flux_hole = None
    w.cfg.update(ageFluxMode='ratio_R_std', rStd=5.7, rStdErrPct=0.035, rStdErrMode='pct')
    w._carousel_apply_flux_to_all()
    for sl in w._carousel:
        assert (sl['cfg']['ageFluxMode'], sl['cfg']['rStd'], sl['cfg']['rStdErrPct']) == \
            ('ratio_R_std', 5.7, 0.035), sl['cfg']
    ages = [sl['plateau']['r']['mean'] for sl in w._carousel]
    close(ages[1], ages[0], 1e-9, 'same R, same ages')

    # 'Apply calibration to all aliquots' shares the calibration, never the flux
    flux = {}
    for i, sl in enumerate(w._carousel):
        c = w.cfg if i == w._carousel_idx else sl['cfg']
        c.update(rStd=5.70 + 0.01 * i, rStdErrPct=0.030 + 0.001 * i)
        flux[i] = (c['ageFluxMode'], c['rStd'], c['rStdErrPct'], sl.get('flux_hole'))
    leg = dict(stdAgeMa=28.201, stdAgeErrMa=0.023, stdAgeErrType='external',
               lambdaYr=5.463e-10, lambdaErrYr=5.4e-12, **am.monitor_keys('FCs'))
    keys = am.route_cfg(am.route_from_cfg(leg))
    w.cfg.update({k: keys[k] for k in am.CALIB_KEYS if k in keys})
    am.set_route(w.cfg, {k: keys[k] for k in am.ROUTE_KEYS if k in keys})
    w._carousel_apply_calib_to_all(quiet=True)
    for i, sl in enumerate(w._carousel):
        if i == w._carousel_idx:
            continue
        c = sl['cfg']
        assert (c['ageFluxMode'], c['rStd'], c['rStdErrPct'], sl.get('flux_hole')) == flux[i], (c, flux[i])
        assert c['stdAgeMa'] == 28.201 and c['route'] == w.cfg['route'], c
    print('  ok  Apply calibration to all: calibration shared, each aliquot keeps its flux')

    # Save XLSX / CSV asks 'all samples or this one' before the file dialog
    order = []
    out = os.path.join(tempfile.mkdtemp(), 'x.xlsx')
    def _save(*a, **k):
        order.append('file')
        return out, 'Excel (*.xlsx)'
    def _exec(box):
        order.append('ask')
        box._pick = next(b for b in box.buttons() if b.text() == 'All samples')
        return 0
    pi.dfold.get_save_file_name = _save
    old_exec, old_clicked = QMessageBox.exec_, QMessageBox.clickedButton
    QMessageBox.exec_ = _exec
    QMessageBox.clickedButton = lambda box: getattr(box, '_pick', None)
    try:
        w.export_csv()
    finally:
        QMessageBox.exec_, QMessageBox.clickedButton = old_exec, old_clicked
    from openpyxl import load_workbook
    ws = load_workbook(out)['Results']
    n = sum(1 for r in ws.iter_rows(values_only=True) if r and r[0] == 'Sample')
    assert order == ['ask', 'file'] and n == 3, (order, n)
    print('  ok  Save XLSX / CSV: asked first, all three samples written')


def test_routes():
    print('Calibration routes:')
    lib = stdlib.load_default()
    import itertools
    n = 0
    for (name, age, err, ext), (_, lam, sl) in itertools.product(am.FC_PRESETS, am.LAMBDA_PRESETS):
        for mon in ('FCs', 'TCs', 'ACs'):
            leg = dict(stdAgeMa=age, stdAgeErrMa=err, stdAgeErrType='external' if ext else 'internal',
                       lambdaYr=lam, lambdaErrYr=sl, **am.monitor_keys(mon))
            rc = am.route_cfg(am.route_from_cfg(leg))
            for t in (0.5, 55.0, 1090.0):
                va = [v for _, v in am.systematic_components(t, leg)]
                vb = [v for _, v in am.systematic_components(t, rc)]
                assert len(va) == len(vb) and all(abs(x - y) <= 1e-12 * max(1, abs(x))
                                                  for x, y in zip(va, vb)), (name, mon, t)
            assert abs(am.cfg_ingrowth(leg) - am.cfg_ingrowth(rc)) <= 1e-15
            n += 1
    print(f'  ok  {n} calibration, λ and monitor combinations equal their old form')

    # TCs → FCs → Phillips et al. (2022), by hand
    tcs = stdlib.steps_from(lib, 'TCs')
    assert [st['to'] for st in tcs] == ['FCs'], tcs
    route = dict(name='TCs', monitor='TCs', steps=tcs, calibration='Phillips et al. (2022)',
                 **{'lambda': 'Min et al. (2000)'})
    assert not stdlib.route_problems(route, lib), stdlib.route_problems(route, lib)
    c = am.route_cfg(route)
    lam = 5.463e-10
    ing = 1.0112 * am.FC_ANCHORS['Phillips et al. (2022)'][2] * math.expm1(lam * 6.943e6)
    close(c['monitorR'] * am.cfg_ingrowth(c), ing, 1e-14, 'TCs ingrowth = R(TCs/FCs) R(FCs/A1) (e^λtA − 1)')
    close(am.monitor_age(c), math.log1p(ing) / lam / 1e6, 1e-12, 'TCs age on the route')
    comps = dict(am.systematic_components(55.0, c))
    close(comps['R(TCs/FCs), Renne et al. (1998)'], am.dt_dlnR(55.0, lam) * 0.001 / 1.0112, 1e-12,
          'step term ∂t/∂lnR · σR/R')
    # shared with an FCs age on Phillips: A1 age, R(FCs/A1) and λ; not the TCs step
    f = am.route_cfg(stdlib.default_route(lib, 'FCs'))
    shared = set(dict(am.systematic_components(55.0, f))) & set(comps)
    assert shared == {'A1 astronomical age', 'R(FCs/A1), Phillips et al. (2022)',
                      'λ, Min et al. (2000)'}, shared
    print('  ok  TCs → FCs → A1: ingrowth, age and terms by hand; shares A1, R(FCs/A1) and λ with FCs')

    # a lab R straight from ACs to A1, and the A1 calibration at the end
    lab = copy.deepcopy(lib)
    acs = next(st for st in lab['standards'] if st['id'] == 'ACs')
    acs['intercalibrations'].append(dict(R=0.17045, R_1s=0.00005, ratio='std/FCs', other='A1',
                                         reference='Our lab (2026)', recommended=False))
    assert not stdlib.check(lab), stdlib.check(lab)
    st = [x for x in stdlib.steps_from(lab, 'ACs') if x['to'] == 'A1' and 'Our lab' in x['source']]
    assert len(st) == 1 and st[0]['num'] == 'ACs'
    direct = dict(name='ACs direct', monitor='ACs', steps=st, calibration='A1 astronomical age',
                  **{'lambda': 'Min et al. (2000)'})
    assert not stdlib.route_problems(direct, lab), stdlib.route_problems(direct, lab)
    c = am.route_cfg(direct, lab)
    close(c['monitorR'] * am.cfg_ingrowth(c), 0.17045 * math.expm1(lam * 6.943e6), 1e-14,
          'ACs ingrowth = R(ACs/A1) (e^λtA − 1)')
    t = 1.185
    comps = dict(am.systematic_components(t, c))
    Rp = math.expm1(lam * t * 1e6) / math.expm1(lam * 6.943e6)
    ref = am.rvalue_age_external(Rp, 0.0, 6.943, 0.0025, lam, 5.4e-12)
    close(math.hypot(comps['A1 astronomical age'], comps['λ, Min et al. (2000)']), ref['ext_ma'],
          1e-9, 'A1 age and λ terms = R-value form on the chain')
    assert 'R(ACs/A1), Our lab (2026)' in comps and not any('FCs' in k for k in comps), comps
    print('  ok  ACs → A1 on a lab R: no FCs anywhere, terms equal the R-value form')

    # the path builder refuses what the data do not support
    bad = dict(direct, steps=[dict(st[0], R=0.2)])
    assert stdlib.route_problems(bad, lab)
    assert stdlib.route_problems(dict(direct, calibration='Phillips et al. (2022)'), lab)
    loop = dict(name='loop', monitor='TCs', calibration='Phillips et al. (2022)',
                steps=tcs + [dict(tcs[0], **{'from': 'FCs', 'to': 'TCs'})],
                **{'lambda': 'Min et al. (2000)'})
    assert stdlib.route_problems(loop, lib)
    assert stdlib.default_route(lib, 'A1')['calibration'] == 'A1 astronomical age'
    assert stdlib.default_route(lib, 'GA1550')['steps'][0]['to'] == 'FCs'
    print('  ok  unsupported R, wrong end, loops refused; defaults for A1 and GA1550')

    # a .jflux hole is read as a ratio of the route's monitor
    fx = am.disk_flux_entry(['1'], [0.0027], [2.7e-6], [[1.0]], (), '', [5.7], [0.0057],
                            5.463e-10, 28.176, 0.0115, None)
    fx['monitor'], fx['monitorR'] = 'FCs', 1.0
    fcs = am.route_cfg(stdlib.default_route(lib, 'FCs'))
    tcs = am.route_cfg(stdlib.default_route(lib, 'TCs'))
    close(am.flux_hole_rstd(fx, 0, fcs), 5.7, 1e-12, 'FCs flux on an FCs route: R as stored')
    close(am.flux_hole_rstd(fx, 0, tcs), 5.7 / 1.0112, 1e-12, 'FCs flux on a TCs route: read as TCs')
    close(am.flux_hole_J(fx, 0, tcs)[0], am.monitor_ingrowth(tcs) / 5.7, 1e-15, '  J = e^(λ t_TCs) − 1 / R')
    fx['monitor'], fx['monitorR'] = 'TCs', 1.0112      # a TCs regression stores R / 1.0112
    close(am.flux_hole_rstd(fx, 0, tcs), 5.7, 1e-12, 'TCs flux on a TCs route: as before')
    assert am.monitor_mismatch_note(fcs, 'TCs', 'the flux') and not am.monitor_mismatch_note(tcs, 'TCs')
    print('  ok  mismatch noted; matching monitors silent')

    # FCs through the K–Ar standards and Mes4: the published FCs ages come back
    lib = stdlib.load_default()
    for cal, lam_name, R, age, tol in (('GA1550 K–Ar age', 'Steiger & Jäger (1977)', 3.5957, 28.02, 0.005),
                                       ('Mes4 astronomical age', 'Min et al. (2000)', 4.1738, 28.179, 0.0006)):
        to = stdlib.find(lib, 'calibrations', cal)['standard']
        st = [x for x in stdlib.steps_from(lib, 'FCs') if x['to'] == to and x['R'] == R]
        r = dict(name='', monitor='FCs', steps=st, calibration=cal, **{'lambda': lam_name})
        assert not stdlib.route_problems(r, lib)
        close(am.monitor_age(am.route_cfg(r, lib)), age, tol, f'FCs through {to} (printed R)')


if __name__ == '__main__':
    test_carter_spreadsheet()
    test_astronomical_covariance_zero()
    test_min_f239()
    test_min_vs_intercalibrated_gap()
    test_external_dispatch()
    test_full_systematics_warning()
    test_full_systematics_enlarges_external()
    test_anchored_external_reproduces_kuiper()
    test_every_anchor_reproduces_its_own_paper()
    test_anchor_reproduces_its_preset_fc_age()
    test_anchor_beats_uncorrelated_route()
    test_external_error_uses_anchor_when_available()
    test_single_step_route_full_lambda()
    test_anchor_reproduces_presets()
    test_single_age_tiers()
    test_stack()
    test_other_lambda()
    test_rr_roundtrip()
    test_library_reproduces_published_ages()
    test_library_entries()
    test_save_ignores_published_changes()
    test_identity_and_round_trip()
    test_monitor_own_age()
    test_factor_is_derivative()
    test_combine_shares_calibration()
    test_lambda_kept_with_another_lambda()
    test_ratio_orientation()
    test_user_edits_merge()
    test_keys_and_age()
    test_j_from_measured_monitor_ratio()
    test_ratio_to_monitor_mode()
    test_external_term()
    test_converter_monitor()
    test_calibration_notes()
    test_manual_flux_in_combine()
    test_combine_flux_session_and_analytical_from_internal()
    test_converter_step_order()
    test_plateau_apply_typed_flux_to_all()
    test_routes()
    print('ALL PASSED')
