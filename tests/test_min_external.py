"""Validation tests for the external-error methods in arar_math.

Run:  python -m pytest tests/test_min_external.py   (or: python tests/test_min_external.py)

Covers three routes:
  1. Carter/Renne intercalibrated  -> reproduces the Carter (2025) spreadsheet.
  2. Astronomical                  -> same form, covariance = 0.
  3. Min et al. (2000) full external -> reproduces the paper's F-239 result.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arar_math as am


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
    """Astronomical route: same partials, covariance term = 0, lambda cancels."""
    print("Astronomical (Kuiper 2008, cov=0):")
    age = 64.6
    sig = am.rvalue_external_sigma(age, t_std_ma=28.201, sigma_t_std_ma=0.023,
                                   lam=5.463e-10, sigma_lam=0.054e-10,
                                   cov_lam_tstd=0.0)
    tt, lt = am.rvalue_external_terms(age, 28.201, 0.023, 5.463e-10, 0.054e-10)
    # cancellation check: the intercalibrated lambda term is ~1% of the
    # full-leverage value age*sigma_lambda/lambda (sensitivity ~0.01, not 1).
    full_leverage = age * (0.054e-10 / 5.463e-10)
    assert lt < 0.03 * full_leverage, f"lambda not cancelling: {lt} vs {full_leverage}"
    print(f"  OK  std-age term={tt:.4f} Ma, lambda term={lt:.5f} Ma; "
          f"lambda is {lt/full_leverage:.3f} of full leverage "
          f"({full_leverage:.4f} Ma); combined sys={sig:.4f} Ma")


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


def test_box_dispatch():
    """external_error routes each calibration to the right box."""
    print("Box dispatch (external_error):")
    carter = dict(stdAgeMa=28.183, stdAgeErrMa=0.017,
                  lambdaYr=5.5042e-10, lambdaErrYr=0.0054e-10)
    # Box 1 default matches rvalue_external_sigma with the joint covariance.
    b1 = am.external_error(64.6, carter, full_systematics=False)
    ref = am.rvalue_external_sigma(64.6, 28.183, 0.017, 5.5042e-10, 0.0054e-10,
                                   am.cfg_covariance(carter))
    _close(b1, ref, 1e-12, "Box 1 == intercalibrated")
    # Box 2 (Min) is far larger and independent of the pinned lambda.
    b2 = am.external_error(64.6, carter, full_systematics=True)
    assert b2 > 10 * b1, f"Box 2 should dwarf Box 1: {b2} vs {b1}"
    print(f"  OK  Box1={b1*1000:.1f} kyr  Box2(Min)={b2*1000:.0f} kyr  "
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


def test_external_sigma_array_default_unchanged():
    """fullSystematics off leaves external_sigma_array behaviour identical."""
    print("external_sigma_array default unchanged:")
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
    """Anchoring on t_astro must give a slightly LARGER systematic than the
    single-step t_FCs route, because the two lambda paths compound rather than
    cancel (Renne et al. 1998, p. 120). Same sign, small size."""
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
    without = dict(base, fcName='Renne et al. (1998)  [S&J lambda]',
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


if __name__ == '__main__':
    test_carter_spreadsheet()
    test_astronomical_covariance_zero()
    test_min_f239()
    test_min_vs_intercalibrated_gap()
    test_box_dispatch()
    test_full_systematics_warning()
    test_external_sigma_array_default_unchanged()
    test_anchored_external_reproduces_kuiper()
    test_every_anchor_reproduces_its_own_paper()
    test_anchor_reproduces_its_preset_fc_age()
    test_anchor_beats_uncorrelated_route()
    test_external_error_uses_anchor_when_available()
    print("\nAll checks passed.")
