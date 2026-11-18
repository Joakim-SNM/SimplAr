"""Every tool opens, saves and reads its files on any operating system, and the
tools agree with each other: the Plateau & isochron stack and the Age
distribution tool give the same answer for the same aliquots, and the window
paths equal the independent reference (tests/reference_framework.py).

Run:  QT_QPA_PLATFORM=offscreen python tests/test_tools.py
"""
import os
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
DATA = tempfile.mkdtemp()
os.environ['SIMPLAR_DATA_DIR'] = DATA
os.environ.setdefault('SIMPLAR_STANDARDS_FILE', os.path.join(tempfile.mkdtemp(), 'standards.json'))
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import importlib.util
import json
import math
import subprocess

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog
from PyQt5.QtGui import QFontDatabase, QFontInfo

import arar_math as am
import data_folders as dfold
import reference_framework as ref
import synthetic_data as synth
from synthetic_data import load_ratios, CFG_R, PHILLIPS, FLUX_PCT

app = QApplication.instance() or QApplication([])
shown = []          # message boxes the tools would have shown: (kind, (title, text))
for _n in ('information', 'warning', 'critical'):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: shown.append((_n, a[1:3])) or 0))


def load(rel, name):
    """A tool's module, loaded from its file."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def ok(what):
    print(f'  ok  {what}')


def close_abs(a, b, tol, what):
    ok = abs(a - b) <= tol
    print(f"  {'ok ' if ok else 'BAD'} {what:<48} {a:.10g}  vs  {b:.10g}")
    assert ok, what


# --------------------------------------------------------------------------
# Every tool opens, saves and reads its files, on any operating system.
# --------------------------------------------------------------------------

def test_launcher_and_about():
    print('Launcher and the Methods & Reference document:')
    L = load('launcher.py', 'launcher')
    w = L.ToolLauncher()
    for label, rel in L.TOOLS:
        assert os.path.isfile(os.path.join(ROOT, *rel)), rel
    ok(f'{len(L.TOOLS)} tool scripts where the launcher looks for them')
    html = L.build_about_html(ROOT)
    assert 'Recalculator and age combiner; standards' in html and '<img src=' in html
    L.AboutDialog(w)
    ok('About document builds, equations rendered to the cache folder')
    assert DATA in w.data_lbl.text()
    ok('launcher shows the data folder')


def test_every_window_opens():
    print('Every tool window opens:')
    names = {}
    names['Flux monitors'] = load('2_flux_monitors/flux_monitors.py', 'fm').FluxMonitorsWindow()
    names['Flux regression'] = load('3_flux_regression/flux_regression.py', 'fr').FluxRegressionCalculator()
    names['Plateau & isochron'] = load('4_plateau_isochron/plateau_isochron.py', 'pl').PlateauWindow()
    names['Age distribution'] = load('5_age_distribution/age_distribution.py', 'ad').ProbabilityWindow()
    names['Recalculator and age combiner'] = load('6_recalculator/recalculator.py', 'rc').RecalculatorWindow()
    names['Standards'] = load('standards.py', 'se').StandardsEditor()
    dc = load('1_data_converter/data_converter.py', 'dc')
    names['Data converter'] = dc.ConvertDialog('x.xlsx', ['Sample', 'Age'], 'massspec')
    for title, w in names.items():
        assert w.windowTitle() == title, (title, w.windowTitle())
        w.show()
    app.processEvents()
    ok(', '.join(names))


def test_data_folders():
    print('Data folder, one subfolder per tool:')
    dfold.use_tool('plateau_isochron')
    assert dfold.start_dir() == os.path.join(DATA, '4 Plateau & isochron')
    p = dfold._start_path('Sample: 1/2 *final*?', None)
    assert os.path.dirname(p) == os.path.join(DATA, '4 Plateau & isochron')
    assert os.path.basename(p) == 'Sample_ 1_2 _final__', os.path.basename(p)
    ok('dialogs open in the tool folder; suggested names made safe for Windows')
    assert dfold.safe_filename('CON.') == 'CON_' and dfold.safe_filename('nul.final') == 'nul_.final'
    assert dfold.safe_filename('CONSTANT') == 'CONSTANT' and dfold.safe_filename('  ') == 'untitled'
    dfold.create_all()
    assert sorted(os.listdir(DATA)) == sorted(dfold.TOOL_FOLDERS.values())
    ok(f'all {len(dfold.TOOL_FOLDERS)} tool folders created')


def test_unicode_session_files():
    print('Session files with σ, λ, ± and Ø in them, written and read back:')
    pl = load('4_plateau_isochron/plateau_isochron.py', 'pl2')
    w = pl.PlateauWindow()
    w.raw_steps = synth.plateau_steps()
    w.title_edit.setText('Træll Ø σ±λ')
    w._update_all()
    path = os.path.join(DATA, '4 Plateau & isochron', 'unicode.arar')
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (path, ''))
    w.save_session()
    with open(path, encoding='utf-8') as f:
        assert json.load(f)['title'] == 'Træll Ø σ±λ'
    w2 = pl.PlateauWindow()
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([path], ''))
    w2.load_session()
    assert w2.title_edit.text() == 'Træll Ø σ±λ', w2.title_edit.text()
    ok('Plateau & isochron session')
    ad = load('5_age_distribution/age_distribution.py', 'ad2')
    a = ad.ProbabilityWindow()
    a.samples[0].name = 'Ø σ±'
    p2 = os.path.join(DATA, '5 Age distribution', 'unicode.arprob')
    QFileDialog.getSaveFileName = staticmethod(lambda *a_, **k: (p2, ''))
    a.save_session()
    a2 = ad.ProbabilityWindow()
    QFileDialog.getOpenFileName = staticmethod(lambda *a_, **k: (p2, ''))
    a2.load_session()
    assert a2.samples[0].name == 'Ø σ±', a2.samples[0].name
    ok('Age distribution session')
    rc = load('6_recalculator/recalculator.py', 'rc2')
    r = rc.RecalculatorWindow()
    r.recalc_tab.inp.setCurrentCell(0, 0)
    r.recalc_tab.inp.paste('Ø σ±\t55.8\t0.08\t0.1')
    p3 = os.path.join(DATA, '6 Recalculator and age combiner', 'unicode.arrc')
    QFileDialog.getSaveFileName = staticmethod(lambda *a_, **k: (p3, ''))
    r.save_session()
    r2 = rc.RecalculatorWindow()
    QFileDialog.getOpenFileName = staticmethod(lambda *a_, **k: (p3, ''))
    r2.load_session()
    assert r2.recalc_tab.results[0]['label'] == 'Ø σ±', r2.recalc_tab.results
    assert abs(r2.recalc_tab.results[0]['age'] - r.recalc_tab.results[0]['age']) < 1e-12
    ok('Recalculator session')


def test_converter_file_names():
    print('Data converter: sample names with characters Windows forbids:')
    dc = load('1_data_converter/data_converter.py', 'dc2')
    rows = []
    for k in range(3):
        rows.append({'Sample': 'Ash: 17/b*', 'Run_ID': f'100-01{"ABC"[k]}', 'Age': 55.0,
                     'Age_Er': 0.1, 'Ar39_Moles': 1e-15, 'J': 0.0027, 'J_Er': 2.7e-6,
                     'Ar40Rad_Over_Ar39': 11.3, 'Ar40Rad_Over_Ar39_Er': 0.02, 'PctAr40Rad': 95,
                     'Ca_Over_K': 1.0, 'Ca_Over_K_Er': 0.1, 'Filename': f'f{k}', 'Irrad': 'IRR?1'})
    out = os.path.join(DATA, '1 Data converter')
    written, warnings = dc._convert_mass_spec_raw(pd.DataFrame(rows), out, {}, 28.176, 0.0115)
    assert len(written) == 1 and not warnings, (written, warnings)
    name = os.path.basename(written[0])
    assert not set(name) & set('<>:"/\\|?*'), name
    ok(f'written as {name}')


_FONT_CHECK = ('import sys; sys.path.insert(0, sys.argv[1]); import arar_math as am; '
               'from PyQt5.QtWidgets import QApplication; from PyQt5.QtGui import QFontInfo; '
               'app = QApplication([]); i = QFontInfo(am.mono_font(9)); '
               'print(i.fixedPitch(), i.family())')


def test_fonts():
    print('Fixed-width font for tables and reports:')
    # The off-screen platform has no font settings (and on Windows no fonts at
    # all), so on Windows and macOS ask the platform the tools normally run on.
    if sys.platform in ('win32', 'darwin'):
        env = {k: v for k, v in os.environ.items() if k != 'QT_QPA_PLATFORM'}
        try:
            r = subprocess.run([sys.executable, '-c', _FONT_CHECK, ROOT], env=env, timeout=120,
                               capture_output=True, text=True, encoding='utf-8', errors='replace')
        except subprocess.TimeoutExpired:
            r = subprocess.CompletedProcess([], 1, '', 'timed out')
        if r.returncode == 0 and r.stdout.strip():
            fixed, family = r.stdout.strip().splitlines()[-1].split(' ', 1)
            assert fixed == 'True', family
            ok(f'{family} (native platform)')
            return
        print('  native platform not available:', r.stderr.strip()[-300:])
    db = QFontDatabase()
    if not any(db.isFixedPitch(x) for x in db.families()):
        print('  skipped: no fixed-width font installed for the off-screen platform')
        return
    f = am.mono_font(9)
    assert QFontInfo(f).fixedPitch(), QFontInfo(f).family()
    ok(QFontInfo(f).family())


# --------------------------------------------------------------------------
# Cross-tool consistency: the Plateau & isochron stack and the Age
# distribution tool must give the same answer for the same aliquots, and
# every path inside the Plateau & isochron tool must use the same J, error
# mode and J correlation.
# --------------------------------------------------------------------------

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
        close(float(pp.export.flux_cells(eff)[0]), J, 1e-7, '  exported J column')
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


# --------------------------------------------------------------------------
# End-to-end: the Age distribution Sample, the plateau tiers, the pooled age
# and the Plateau stack on synthetic data must all equal the independent
# reference (tests/reference_framework.py).
# --------------------------------------------------------------------------

def test_gui_paths():
    """End-to-end: the Age distribution Sample, the plateau tiers, the
    pooled age and the Plateau stack on synthetic data must all equal the
    independent reference (tests/reference_framework.py)."""
    R, sR, F, sF = load_ratios()
    anc = ref.anchor(PHILLIPS)
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
    close_abs(w['wm'] * 1e6, m_ref, 1e-3, 'WM (yr)')
    close_abs(w['sigma_ana'] * 1e6 / math.sqrt(w['mswd_scale']**2), a_ref, 1e-3, 'ana 1s (yr, unscaled)')
    assert w['mswd_scale'] == 1.0, 'the synthetic set has no excess scatter'
    close_abs(w['sigma_int'] * 1e6, i_ref, 1e-3, 'int 1s (yr)')
    close_abs(w['sigma_ext'] * 1e6, e_ref, 1e-3, 'ext 1s (yr)')

    print('Age distribution, ratio_R_std input (F values, R_std):')
    s2 = pp.Sample('synthetic F')
    s2.input_mode = 'ratio'; s2.reject = 'none'
    s2.j_cfg.update(dict(CFG_R, ageFluxMode='ratio_R_std'))
    s2.r_vals = np.array(F); s2.r_uncs = np.array(sF)
    assert s2.recompute() is None
    w2 = s2.wm()
    close_abs(w2['wm'] * 1e6, m_ref, 1e-3, 'WM (yr)')
    close_abs(w2['sigma_ext'] * 1e6, w['sigma_ext'] * 1e6, 0.05, 'ext equal to R route (yr)')

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
    close_abs(p['e1s_int'] * 1e6, i1, 1e-3, 'int 1s (yr)')
    close_abs(p['e1s_ext'] * 1e6, e1, 1e-3, 'ext 1s (yr)')
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
        close_abs(pr['wm'] * 1e6, m_ref, 1e-3, 'pooled WM (yr)')
        close_abs(pr['sigma_int'] * 1e6, i_ref, 1e-3, 'pooled int 1s (yr)  [same J position]')
        close_abs(pr['sigma_ext'] * 1e6, e_ref, 1e-3, 'pooled ext 1s (yr)')

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
        se = am.external_sigma_array(ages, si, CFG_R)
        res = am.combine_ages(ages, sa, si, se, standard_group=grp, sys_components=C,
                              scale_overdispersion=False)
        w = (1 / sa**2) / np.sum(1 / sa**2)
        comps = [ref.tiers(a * 1e6, 0, 0, anc)[3] for a in ages]
        v_sys = sum(sum(wi * c[k] for wi, c in zip(w, comps))**2 for k in comps[0])
        v_p = float(np.sum((w * np.sqrt(si**2 - sa**2))**2))   # different positions
        exp = math.sqrt(float(np.sum((w * sa)**2)) * 1e12 + v_p * 1e12 + v_sys)
        close_abs(res['sigma_ext'] * 1e6, exp, 1e-3, 'ext 1s (yr)')

    _stack_test()
    print('STACK PASSED')


def test_age_distribution_data_box():
    """The data box reads back what it shows: editing one line at σ-in 2σ
    leaves the other uncertainties alone; Append into an empty sample keeps R."""
    print('Age distribution data box and Append:')
    ad = load('5_age_distribution/age_distribution.py', 'ad_box')
    win = ad.ProbabilityWindow()
    p = win.panels[0]
    win.input_sigma = 2
    p.data.setPlainText('55.0\t0.2\n56.0\t0.2\n55.5\t0.2')
    p._apply_paste()
    close_abs(float(p.s.unc_2s[0]), 0.1, 1e-15, 'pasted at 2σ, stored at 1σ')
    p.data.setPlainText(p.data.toPlainText().replace('56.0', '56.1'))
    p._apply_paste()
    assert np.allclose(p.s.unc_2s, 0.1) and abs(p.s.ages[1] - 56.1) < 1e-12, p.s.unc_2s
    ok('an edit of one line at σ-in 2σ leaves every error at 0.1')
    path = synth.write_arar_files(tempfile.mkdtemp())[0]
    s2 = win.add_sample()
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([path], ''))
    win.append_into(s2)
    assert s2.has_R() and s2.input_mode == 'ratio', (len(s2.r_vals), s2.input_mode)
    ok('Append into an empty sample keeps the R values')


if __name__ == '__main__':
    test_launcher_and_about()
    test_every_window_opens()
    test_data_folders()
    test_unicode_session_files()
    test_converter_file_names()
    test_fonts()
    test_position_ident()
    test_disk_flux_round_trip()
    test_error_modes()
    test_one_J_everywhere()
    test_converter_file_from_R()
    test_wendt_carl_threshold()
    test_stack_vs_probability()
    test_imported_J_in_internal()
    test_flux_on_disk_vs_external()
    test_gui_paths()
    test_age_distribution_data_box()
    print('ALL PASSED')
