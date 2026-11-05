"""Cross-tool consistency: the Plateau & isochron stack and the Age distribution
tool must give the same answer for the same aliquots, and every path inside the
Plateau & isochron tool must use the same J, error mode and J correlation.

Run: python tests/test_consistency.py
"""
import os, sys, json, math, tempfile, importlib.util
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('SIMPLAR_DATA_DIR', tempfile.mkdtemp())
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import arar_math as am
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog
app = QApplication.instance() or QApplication([])
for _n in ('information', 'warning', 'critical'):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: QMessageBox.Ok))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


pl = load('4_plateau_isochron/plateau_isochron.py', 'pl')
pp = load('5_age_distribution/age_distribution.py', 'pp')


def close(a, b, tol, what):
    ok = abs(a - b) <= tol * max(1.0, abs(b))
    print(f"  {'ok ' if ok else 'BAD'} {what:<58} {a:.10g}  vs  {b:.10g}")
    assert ok, what


PHIL = dict(stdAgeMa=28.176, stdAgeErrMa=0.0115, stdAgeErrType='external',
            lambdaYr=5.463e-10, lambdaErrYr=0.054e-10)
CARTER_INGROWTH = math.expm1(5.5042e-10 * 28.183e6)


def steps(R0, n=8, scatter=0.0, seed=1, err=0.004):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        R = R0 * (1 + scatter * rng.normal())
        out.append(dict(label=chr(65 + k), power=k, age=0, err1s=0, pct39Ar=100.0 / n,
                        caK=1, pct40=99, r4039=R, err_r4039=err, j_val=0, j_err=0,
                        status='normal'))
    return out


JFLUX = dict(type='jflux_config', version=1, irradiation='IRR', lambdaYr=5.5042e-10,
             stdAgeMa=28.183, stdAgeErrMa=0.017,
             disks=[dict(disk_id='D', holes=['1', '2'], hole_xy=[[0, 0], [1, 0]],
                         J=[CARTER_INGROWTH / 5.70, CARTER_INGROWTH / 5.72],
                         J_sigma=[CARTER_INGROWTH / 5.70 * 2e-4, CARTER_INGROWTH / 5.72 * 3e-4],
                         J_corr=[[1, 0.6], [0.6, 1]])])


def make_stack(err_mode='sem_mswd_2s', jflux=True):
    """Three aliquots: two on hole D1, one on hole D2, each R standard baked
    from its hole the way the flux dialog's Apply does."""
    w = pl.PlateauWindow()
    w._toggle_stack_mode(True)
    if jflux:
        w._disk_flux.update(am.jflux_entries(JFLUX))
    for k, (hole, rstd, rel, R0) in enumerate([('D1', 5.70, 0.02, 11.30), ('D1', 5.70, 0.02, 11.31),
                                               ('D2', 5.72, 0.03, 11.36)]):
        cfg = dict(pl._SESSION_CFG_DEFAULTS, **PHIL, ageFluxMode='ratio_R_std', rStd=rstd,
                   rStdErrPct=rel, errMode=err_mode, disk_id='D', hole_id=hole)
        w._sessions.append({'title': f'al{k}', 'cfg': cfg, 'orig_title': f'al{k}',
                            'raw_steps': steps(R0, scatter=4e-4, seed=k)})
    w._global_cfg.update(dict(PHIL, errMode=err_mode))
    w.cfg.update(dict(PHIL, errMode=err_mode))
    w._update_flux_groups()
    w._recompute_all_sessions()
    return w


def save_stack(w):
    path = os.path.join(tempfile.mkdtemp(), 'stack.ararstack')
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (path, ''))
    w._stack_save()
    return path


def prob_pool(path, answer):
    win = pp.ProbabilityWindow()
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([path], ''))
    QMessageBox.question = staticmethod(lambda *a, **k: answer)
    win.load_into(win.samples[0])
    win.pool_on = True
    return win


def test_stack_vs_probability():
    print('Plateau stack (n = aliquots) vs the Age distribution tool, same .ararstack:')
    w = make_stack()
    path = save_stack(w)
    for meth, pmeth in ((am.METHOD_WM, am.METHOD_WM), (am.METHOD_GLS, am.METHOD_GLS)):
        w._stack_method, w._wm_n_basis = meth, 'aliquots'
        r, a, i, e = w._combined_plateau()
        pp._STAT_METHOD = pmeth
        for answer, how in ((QMessageBox.Yes, 'one sample per aliquot'),
                            (QMessageBox.No, 'one combined sample')):
            win = prob_pool(path, answer)
            res = win.pool() if answer == QMessageBox.Yes else win.samples[0].wm()
            close(res['wm'], r['mean'], 1e-9, f'{meth}: age, {how}')
            close(res['sigma_int'], i, 1e-6, f'{meth}: internal 1s, {how}')
            close(res['sigma_ext'], e, 1e-6, f'{meth}: external 1s, {how}')
    pp._STAT_METHOD = am.METHOD_WM
    # the stack's .jflux reached the probability plot: hole 1 and 2 correlate
    # at the file's 0.6, not as one disk (1)
    win = prob_pool(path, QMessageBox.Yes)
    infos = [win._j_point_info(s, [0])[0] for s in win._active()]
    rho = win._j_corr(infos)
    close(rho[0, 2], 0.6, 1e-12, 'J correlation D1-D2 from the stack .jflux')


def test_error_modes():
    print('Error modes: every path scales the SEM with arar_math.scale_sem:')
    for em in am.ERR_MODES:
        w = make_stack(em)
        w._stack_method, w._wm_n_basis = am.METHOD_WM, 'steps'
        r, a, i, e = w._combined_plateau()
        exp = am.scale_sem(r['sem_raw'], r['mswd'], r['prob'], r['n'], em, 5.0)
        close(a, exp, 1e-12, f'n = steps, {em}')
        for s in w._sessions:
            p = s['plateau']
            close(p['e1s_ana'], am.scale_sem(p['r']['sem'], p['r']['mswd'], p['r']['prob'],
                                             p['r']['n'], em, 5.0), 1e-12, f'plateau {s["title"]}, {em}')
    # sem_mswd: 1s carries t/2, so the 2s display is the Student-t 95% interval
    v = am.scale_sem(1.0, 4.0, 0.001, 10, 'sem_mswd', 5.0)
    close(v, 2.0 * am.t_crit(9) / 2.0, 1e-12, 'sem_mswd: SEM*sqrt(MSWD)*t/2')
    # Wendt & Carl limit: MSWD 1.5 at n = 10 is within it
    close(am.scale_sem(1.0, 1.5, 0.2, 10, 'sem_mswd_2s'), 1.0, 1e-12, 'sem_mswd_2s below the limit')


def test_one_J_everywhere():
    print('One J: plateau, saved plateau, isochron and export all use the aged J:')
    w = make_stack()
    for si, s in enumerate(w._sessions):
        eff = w._eff_cfg_for(si)
        J, rel = am.cfg_J(eff)
        close(s['plateau']['J'], J, 1e-15, f'plateau J, {s["title"]}')
        close(pl._plateau_summary(s['plateau'], eff)['J'], J, 1e-15, '  saved plateau J')
        close(w._iso_J(eff, s['steps_c']), J, 1e-15, '  isochron / CSV J')
        close(float(pp.dexp.flux_cells(eff)[0]), J, 1e-7, '  exported J column')
    # a stale jVal in the session cfg must not leak into any of them
    w._sessions[0]['cfg']['jVal'] = 0.005
    w._recompute_all_sessions()
    close(w._iso_J(w._eff_cfg_for(0), w._sessions[0]['steps_c']),
          am.cfg_J(w._eff_cfg_for(0))[0], 1e-15, 'stale jVal ignored')


def test_converter_file_from_R():
    print('Converter .arar on the mass-spec calibration (Carter J, Phillips cfg):')
    rstd = 5.725
    J_file = CARTER_INGROWTH / rstd
    st = []
    for k, R in enumerate([11.40, 11.41, 11.39, 11.405]):
        t_file = math.log1p(J_file * R) / 5.5042e-10 / 1e6
        st.append(dict(label=chr(65 + k), age=t_file, err2s=0.05, r4039=R, err_r4039=0.004,
                       j_val=J_file, j_err=J_file * 2e-4, pct39Ar=25, status='normal'))
    data = dict(title='conv', cfg=dict(ageFluxMode='imported', jVal=J_file, jErrPct=0.02,
                                       rStd=rstd, rStdErrPct=0.02, stepSigma=1, **PHIL),
                steps=st)
    path = os.path.join(tempfile.mkdtemp(), 'conv.arar')
    json.dump(data, open(path, 'w', encoding='utf-8'))
    d = pp.parse_arar(path)
    assert d['j_cfg']['ageFluxMode'] == 'ratio_R_std'
    win = pp.ProbabilityWindow()
    win._load_docs(win.samples[0], [d])
    age_prob = win.samples[0].ages[0]
    cfg = dict(PHIL, ageFluxMode='ratio_R_std', rStd=rstd)
    (ages_R, _s), _ = am.r_to_age_cfg([11.40], [0.0], cfg)
    close(age_prob, float(ages_R[0]), 1e-12, 'Probability age = R-standard age at the cfg calibration')
    close(win.samples[0].j_cfg['jVal'], am.cfg_J(cfg)[0], 1e-15, 'jVal synced to the R standard')


def test_wendt_carl_threshold():
    print('Wendt & Carl trigger: MSWD above 1 but below the limit is not scaled:')
    ages = [55.00, 55.02, 54.99, 55.03]
    sa = [0.015] * 4
    r = am.combine_ages(ages, sa, od_mode='threshold')
    assert 1.0 < r['mswd'] < am.mswd_critical(4) and r['mswd_scale'] == 1.0
    print('  ok')


def test_position_ident():
    print('position_ident: disk, hole, group, then the J itself:')
    assert am.position_ident('E', 'E17') == 'E'
    assert am.position_ident(None, 'E17') == 'E17'
    assert am.position_ident(None, None, 'grp') == 'grp'
    c1 = dict(PHIL, ageFluxMode='ratio_R_std', rStd=5.7, rStdErrPct=0.02)
    assert am.position_ident(cfg=c1) == am.position_ident(cfg=dict(c1))
    assert am.position_ident(cfg=c1) != am.position_ident(cfg=dict(c1, rStd=5.71))
    assert am.position_ident() is None
    print('  ok')


def test_disk_flux_round_trip():
    print('Disk-flux registry: .jflux, stack and session formats read back alike:')
    a = am.jflux_entries(JFLUX)
    b = am.deserialize_disk_flux(am.serialize_disk_flux(a))
    lst = [dict(irradiation=k[0], disk_id=k[1], holes=v['holes'], J=v['J'].tolist(),
                J_sigma=v['J_sigma'].tolist(), J_corr=v['J_corr'].tolist(),
                lamFlux=v['lamFlux'], fcFlux=v['fcFlux'], fcFluxErr=v['fcFluxErr'])
           for k, v in a.items()]
    c = am.deserialize_disk_flux(lst)
    for reg in (b, c):
        fx = reg[('IRR', 'D')]
        close(am.flux_hole_R(fx, 1), 5.72, 1e-12, 'hole R')
        J, sJ = am.flux_hole_J(fx, 1, PHIL)
        close(J, am.cfg_ingrowth(PHIL) / 5.72, 1e-15, 'hole J at the Phillips calibration')
        close(sJ / J, 3e-4, 1e-9, 'relative sigma kept')


def test_imported_J_in_internal():
    print('Imported ages: the J error stored with the steps is in the internal tier:')
    st = []
    for k in range(6):
        st.append(dict(label=chr(65 + k), power=k, age=55.0 + 0.002 * (k % 3), err1s=0.02,
                       pct39Ar=100 / 6, caK=1, pct40=99, r4039=11.3, err_r4039=0.004,
                       j_val=0.0027192, j_err=0.0027192 * 3.5e-4, status='normal'))
    cfg = dict(pl._SESSION_CFG_DEFAULTS, **PHIL, ageFluxMode='imported', minPct=50, minN=3)
    p = pl.find_plateau(pl.add_cumulative(st), cfg)
    exp = math.hypot(p['e1s_ana'], am.dt_dlnR(p['r']['mean'], PHIL['lambdaYr']) * 3.5e-4)
    close(p['e1s_int'], exp, 1e-12, 'internal = analytical (+) dt/dlnJ * sJ/J')
    assert p['e1s_int'] > p['e1s_ana']
    close(pl._plateau_summary(p, cfg)['jErrFrac'], 3.5e-4, 1e-12, 'saved jErrFrac')
    assert 'J' not in pl._plateau_summary(p, cfg)   # a reader rebuilds from the R standard


def test_flux_on_disk_vs_external():
    print('Flux: rows on the disk use its J and rho; the others keep their J, independent:')
    w = make_stack()
    for title, disk, hole, rstd in (('own J on D1', 'D', 'D1', 5.80), ('other lab', 'Q', 'Q3', 5.75)):
        cfg = dict(pl._SESSION_CFG_DEFAULTS, **PHIL, ageFluxMode='ratio_R_std', rStd=rstd,
                   rStdErrPct=0.03, disk_id=disk, hole_id=hole)
        w._sessions.append({'title': title, 'cfg': cfg, 'orig_title': title,
                            'raw_steps': steps(11.33, scatter=4e-4, seed=7)})
    w._flux_override[3] = 'none'          # unticked: J from elsewhere
    pl._StackFluxHost(w).apply()
    ing = am.cfg_ingrowth(PHIL)
    close(am.cfg_J(w._eff_cfg_for(0))[0], ing / 5.70, 1e-15, 'on D1: hole J')
    close(am.cfg_J(w._eff_cfg_for(3))[0], ing / 5.80, 1e-15, "'none' on D1: own J")
    close(am.cfg_J(w._eff_cfg_for(4))[0], ing / 5.75, 1e-15, 'not on a loaded disk: own J')
    rho = w._build_stack_pcorr(list(range(5)))
    close(rho[0, 1], 1.0, 1e-12, 'same hole: 1')
    close(rho[0, 2], 0.6, 1e-12, 'D1-D2: .jflux rho')
    close(rho[0, 3], 0.0, 1e-12, "on the disk vs 'none' on the same disk id: 0")
    close(rho[0, 4], 0.0, 1e-12, 'on the disk vs another lab: 0')
    path = save_stack(w)
    assert json.load(open(path, encoding='utf-8'))['flux_override']['3'] == 'none'
    w2 = pl.PlateauWindow(); w2._toggle_stack_mode(True)
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([path], ''))
    w2._stack_load()
    assert w2._flux_override.get(3) == 'none'
    # legacy files wrote 'none' as its letters
    assert pl._flux_override_from_disk(list('none')) == 'none'
    win = prob_pool(path, QMessageBox.Yes)
    act = win._active()
    infos = [win._j_point_info(s, [0])[0] for s in act]
    rp = win._j_corr(infos)
    for a_, b_ in ((0, 2), (0, 3), (0, 4), (3, 4)):
        close(rp[a_, b_], rho[a_, b_], 1e-12, f'probability rho[{a_},{b_}] = plateau')
    close(am.cfg_J(act[3].j_cfg)[0], ing / 5.80, 1e-12, "probability: 'none' row keeps its J")


if __name__ == '__main__':
    test_position_ident()
    test_disk_flux_round_trip()
    test_error_modes()
    test_one_J_everywhere()
    test_converter_file_from_R()
    test_wendt_carl_threshold()
    test_stack_vs_probability()
    test_imported_J_in_internal()
    test_flux_on_disk_vs_external()
    print('ALL CONSISTENCY CHECKS PASSED')
