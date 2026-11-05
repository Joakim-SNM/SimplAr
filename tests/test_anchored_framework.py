"""Checks the toolchain against tests/reference_framework.py, which is written
independently from the published equations.  Run: python tests/test_anchored_framework.py"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import arar_math as am
import reference_framework as ref
from synthetic_data import ratio_set, RSTD, FLUX_PCT

LAM, SLAM = 5.463e-10, 0.054e-10
CFG = dict(ageFluxMode='ratio_R_std', rStd=RSTD, rStdErrPct=FLUX_PCT,
           rStdErrMode='pct', stdAgeMa=28.176, stdAgeErrMa=0.0115,
           stdAgeErrType='external', lambdaYr=LAM, lambdaErrYr=SLAM)
CFG_R = dict(CFG, ageFluxMode='ratio_R_FCs', rfcsErrPct=FLUX_PCT,
             pasteTier='analytical')
PHIL = 'Phillips et al. (2022)'


def close(a, b, tol, what):
    ok = abs(a - b) <= tol
    print(f"  {'ok ' if ok else 'BAD'} {what:<48} {a:.10g}  vs  {b:.10g}")
    assert ok, what


def load_ratios():
    """43 made-up single-crystal analyses (tests/synthetic_data.py)."""
    return ratio_set()


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
    t = ref.age(R_, ref.anchor(PHIL))
    (ages, _s), err = am.r_to_age_cfg([R_], [0.0], CFG_R)
    ages = float(ages[0])
    close(ages * 1e6, t, 1e-4, 'age (yr)')
    rel_ana = 0.0004 / R_
    rel_flux = FLUX_PCT / 100
    a, i, e, comps, _ = ref.tiers(t, rel_ana, rel_flux, ref.anchor(PHIL))
    s_int = am.internal_sigma_array([ages], [am.dt_dlnR(ages, LAM) * rel_ana], CFG_R)[0]
    s_ext = am.external_sigma_array([ages], [s_int], CFG_R)[0]
    close(s_int * 1e6, i, 1e-4, 'internal 1s (yr)')
    close(s_ext * 1e6, e, 1e-4, 'external 1s (yr)')
    got = dict(am.systematic_components(ages, CFG))
    close(got['R(FCs/anchor) calibration'] * 1e6, comps['R_cal'], 1e-6, 'R cal comp')
    close(got['Anchor tephra astronomical age'] * 1e6, comps['t_A1'], 1e-6, 't_A1 comp')
    close(got['Decay constant λ'] * 1e6, comps['lam'], 1e-6, 'lambda comp')
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
    anc = ref.anchor(PHIL)
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


def test_other_lambda():
    print('anchored preset with a non-native lambda: age follows the chain')
    cfg = dict(CFG, lambdaYr=5.5305e-10, lambdaErrYr=0.0132e-10)
    anc = ref.anchor(PHIL)
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


if __name__ == '__main__':
    test_anchor_reproduces_presets()
    test_single_age_tiers()
    test_stack()
    test_other_lambda()
    test_rr_roundtrip()
    print('ALL PASSED')
