"""End-to-end: the Age distribution Sample, the plateau tiers, the pooled age
and the Plateau stack on synthetic data must all equal the independent
reference (tests/reference_framework.py)."""
import os, sys, math, importlib.util, tempfile
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('SIMPLAR_DATA_DIR', tempfile.mkdtemp())
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import reference_framework as ref
from test_anchored_framework import load_ratios, CFG_R, close, PHIL
from synthetic_data import FLUX_PCT

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

R, sR, F, sF = load_ratios()
anc = ref.anchor(PHIL)
t_ref = [ref.age(r, anc) for r in R]
d = [ref.partials(t, anc)[0] for t in t_ref]
m_ref, a_ref, i_ref, e_ref = ref.combine_shared(
    t_ref, [di * s / r for di, s, r in zip(d, sR, R)],
    [di * FLUX_PCT * 1e-2 for di in d], [ref.tiers(t, 0, 0, anc)[3] for t in t_ref])

print('Age distribution, Sample.wm() (ratio_R_FCs input):')
pp = load('5_age_distribution/age_distribution.py', 'pp')
s = pp.Sample('synthetic')
s.input_mode = 'ratio'; s.reject = 'none'
s.j_cfg.update(CFG_R)
s.r_vals = np.array(R); s.r_uncs = np.array(sR)
assert s.recompute() is None
w = s.wm()
close(w['wm'] * 1e6, m_ref, 1e-3, 'WM (yr)')
close(w['sigma_ana'] * 1e6 / math.sqrt(w['mswd_scale']**2), a_ref, 1e-3, 'ana 1s (yr, unscaled)')
if w['mswd_scale'] == 1.0:
    close(w['sigma_int'] * 1e6, i_ref, 1e-3, 'int 1s (yr)')
    close(w['sigma_ext'] * 1e6, e_ref, 1e-3, 'ext 1s (yr)')
else:
    print('  (overdispersed: ana scaled by', w['mswd_scale'], ')')

print('Age distribution, ratio_R_std input (F values, R_std):')
s2 = pp.Sample('synthetic F')
s2.input_mode = 'ratio'; s2.reject = 'none'
s2.j_cfg.update(dict(CFG_R, ageFluxMode='ratio_R_std'))
s2.r_vals = np.array(F); s2.r_uncs = np.array(sF)
assert s2.recompute() is None
w2 = s2.wm()
close(w2['wm'] * 1e6, m_ref, 1e-3, 'WM (yr)')
close(w2['sigma_ext'] * 1e6, w['sigma_ext'] * 1e6, 0.05, 'ext equal to R route (yr)')

print('Plateau _compute_tiers via find_plateau on a synthetic 3-step plateau:')
pl = load('4_plateau_isochron/plateau_isochron.py', 'pl')
steps = [dict(label=str(k), power=k, age=0, err1s=0, pct39Ar=33.3, caK=0, pct40=99,
              r4039=float(F[k]), err_r4039=float(sF[k]), j_val=0, j_err=0,
              status='normal') for k in range(3)]
pcfg = dict(CFG_R, ageFluxMode='ratio_R_std', minPct=50, minN=3, pCut=5,
            errMode='sem')
st = pl._compute_steps_with_age_mode(steps, pcfg)
res = pl.find_plateau(st, pcfg)
p = res if isinstance(res, dict) else res[0]
t = p['r']['mean'] * 1e6
a1 = p['e1s_ana'] * 1e6
_, i1, e1, _, _ = ref.tiers(t, a1 / ref.partials(t, anc)[0], FLUX_PCT * 1e-2, anc)
close(p['e1s_int'] * 1e6, i1, 1e-3, 'int 1s (yr)')
close(p['e1s_ext'] * 1e6, e1, 1e-3, 'ext 1s (yr)')
print('ALL GUI PATHS PASSED')


def _pool_test():
    print('Age distribution pool(): the synthetic set split into two samples, same calibration:')
    win = pp.ProbabilityWindow()
    for sm in list(win.samples):
        win.remove_sample(sm)
    half = len(R) // 2
    for nm, sl in (('A', slice(0, half)), ('B', slice(half, None))):
        smp = win.add_sample()
        smp.name = nm
        smp.input_mode = 'ratio'; smp.reject = 'none'
        smp.j_cfg.update(CFG_R)
        smp.r_vals = np.array(R[sl]); smp.r_uncs = np.array(sR[sl])
        smp.recompute()
    win.pool_on = True
    pr = win.pool()
    close(pr['wm'] * 1e6, m_ref, 1e-3, 'pooled WM (yr)')
    close(pr['sigma_int'] * 1e6, i_ref, 1e-3, 'pooled int 1s (yr)  [same J position]')
    close(pr['sigma_ext'] * 1e6, e_ref, 1e-3, 'pooled ext 1s (yr)')


_pool_test()
print('POOL PASSED')


def _stack_test():
    print('Plateau stack: aliquot combination with exact shared systematics:')
    class Fake:
        _eff_cfg_for = staticmethod(lambda si: dict(CFG_R))
    ages = [55.30, 55.10, 55.50]
    grp, C = pl.PlateauWindow._stack_systematics(Fake(), [0, 1, 2], ages)
    assert grp == [0, 0, 0]
    sa = np.array([0.01, 0.02, 0.015]); si = sa * 1.2
    import arar_math as am
    se = am.external_sigma_array(ages, si, CFG_R)
    res = am.combine_ages(ages, sa, si, se, standard_group=grp, sys_components=C,
                          scale_overdispersion=False)
    w = (1 / sa**2) / np.sum(1 / sa**2)
    comps = [ref.tiers(a * 1e6, 0, 0, anc)[3] for a in ages]
    v_sys = sum(sum(wi * c[k] for wi, c in zip(w, comps))**2 for k in comps[0])
    v_p = float(np.sum((w * np.sqrt(si**2 - sa**2))**2))   # different positions
    exp = math.sqrt(float(np.sum((w * sa)**2)) * 1e12 + v_p * 1e12 + v_sys)
    close(res['sigma_ext'] * 1e6, exp, 1e-3, 'ext 1s (yr)')


_stack_test()
print('STACK PASSED')
