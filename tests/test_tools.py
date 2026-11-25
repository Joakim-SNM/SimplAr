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
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog, QInputDialog, QDialog, QTableWidgetItem, QPushButton
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
    assert 'Standards, routes and calibrations' in html and '<img src=' in html
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


def test_isochron_atmospheric_in_view():
    print('Isochron, atmospheric reference:')
    def iso_steps(b, xr, off=0.0):
        out = []
        for k in range(6):
            x = xr * (0.45 + 0.1 * k) + off
            y = b * (1 - x / xr)
            out.append(dict(label=chr(65 + k), status='normal', iso_nx=x, iso_nsx=x * 1e-3,
                            iso_ny=y, iso_nsy=max(y, 1e-6) * 1e-3 + 2e-8, iso_iri=0.02,
                            pct39Ar=100 / 6, r4039=11.3, err_r4039=0.004))
        return out
    for data in (iso_steps(1 / 320, 1 / 11.3),
                 [iso_steps(1 / 320, 1 / 11.3), iso_steps(1 / 320, 1 / 11.3, off=5e-4)]):
        c = pl.IsochronCanvas()
        c._showing_inverse, c._iso_zoom, c._atm_ratio = True, 'intercept', None
        c.redraw(data, J=0.0027)
        assert c.ax.get_ylim()[1] < 1 / 298.56
        c._atm_ratio = 298.56
        c.redraw(data, J=0.0027)
        assert c.ax.get_ylim()[1] > 1 / 298.56, c.ax.get_ylim()
    ok('an intercept below atmosphere: the axis reaches the reference line (one and several aliquots)')


def test_no_wheel_on_dropdowns():
    print('Mouse wheel over a dropdown:')
    from PyQt5.QtCore import Qt, QPoint, QPointF
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QComboBox, QScrollArea, QWidget, QVBoxLayout
    app = QApplication.instance()
    am.no_wheel_on_dropdowns(app)
    area, inner = QScrollArea(), QWidget()
    lay = QVBoxLayout(inner)
    cb = QComboBox()
    cb.addItems(['a', 'b', 'c'])
    lay.addWidget(cb)
    inner.setMinimumHeight(2000)
    area.setWidget(inner)
    area.resize(300, 200)
    area.show()
    app.processEvents()
    ev = QWheelEvent(QPointF(5, 5), QPointF(cb.mapToGlobal(QPoint(5, 5))), QPoint(0, 0),
                     QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    app.sendEvent(cb, ev)
    assert cb.currentIndex() == 0
    assert area.verticalScrollBar().value() > 0
    ok('the value stays; the page scrolls instead')
    from PyQt5.QtWidgets import QDoubleSpinBox
    sb = QDoubleSpinBox(); sb.setValue(2.0)
    lay.insertWidget(0, sb)
    area.verticalScrollBar().setValue(0)
    app.processEvents()
    for target in (sb, sb.lineEdit()):
        ev = QWheelEvent(QPointF(5, 5), QPointF(target.mapToGlobal(QPoint(5, 5))), QPoint(0, 0),
                         QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        app.sendEvent(target, ev)
    assert sb.value() == 2.0 and area.verticalScrollBar().value() > 0
    area.close()
    ok('a number field keeps its value too')


def test_standards_window():
    print('Standards window: monitors, calibrations, routes:')
    se = load('standards.py', 'se2')
    w = se.StandardsEditor()
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == \
        ['Monitors', 'Calibrations', 'Routes', 'Decay constants']
    mon, cal, rt, _ = w.pages
    assert [r['id'] for r in mon.rows][0] == 'FCs' and not mon.list.isRowHidden(0)
    assert 'Phillips' in mon.derived.text()
    w._commit_all()
    assert not w.dirty
    ok('FCs listed with the other monitors; browsing changes nothing')
    ids = [r['id'] for r in mon.rows]
    mon.list.setCurrentRow(ids.index('FCs'))
    acs = [i for i in range(mon.ics.rowCount()) if mon.ics.cellWidget(i, 0).currentText() == 'ACs']
    assert acs and mon.ics.cellWidget(acs[0], 1).currentText() == 'other / this'
    mon.ics.item(acs[0], 2).setText('0.0417')
    mon.list.setCurrentRow(ids.index('ACs'))
    assert mon.rows[ids.index('ACs')]['intercalibrations'][0]['R'] == 0.0417
    assert mon.ics.item(0, 2).text() == '0.0417' and w.dirty
    ok('an R stored with ACs shows and edits on the FCs page too')
    w.lib = se.load()
    w.dirty = False
    for p in w.pages:
        p._cur = None
        p.refresh_list(0)
    QInputDialog.getText = staticmethod(lambda *a, **k: ('TCs via FCs', True))
    rt._add()
    rt.mon.setCurrentText('TCs')
    assert [rt.next.itemText(i).split(':')[0] for i in range(rt.next.count())] == ['→ FCs']
    assert not rt.cal.isEnabled()                    # TCs has no calibration of its own
    rt._add_step()
    assert rt.cal.isEnabled() and rt.cal.currentText() == 'Phillips et al. (2022)'
    rt.default.setChecked(True)
    assert 'TCs = 28.4891 Ma' in rt.derived.text(), rt.derived.text()
    w._commit_all()
    lib = w.lib
    assert not se.check(lib), se.check(lib)
    assert se.default_route(lib, 'TCs')['name'] == 'TCs via FCs'
    ok('route TCs → FCs → Phillips built step by step and made the TCs default')


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


def test_combiner_loads_files():
    print('Age combiner: ages from .arar, .ararstack and .arprob files:')
    rcm = load('6_recalculator/recalculator.py', 'rc_files')
    # a stack: one row per aliquot, its plateau
    w = make_stack()
    rows, flux, notes = rcm.file_ages(save_stack(w))
    assert len(rows) == 3 and flux, (len(rows), notes)
    for row, sess in zip(rows, w._sessions):
        p = sess['plateau']
        close(row['age'], p['r']['mean'], 1e-12, f"{row['label']}: plateau age")
        close(row['int'], p['e1s_int'], 1e-12, '  internal error')
        assert row['hole'] == sess['cfg']['hole_id']
    # a single .arar: its plateau, as the Plateau & isochron tool finds it
    one = pl.PlateauWindow()
    one.cfg.update(dict(PHIL, ageFluxMode='ratio_R_std', rStd=5.70, rStdErrPct=0.02))
    one.raw_steps = steps(11.30, scatter=4e-4, seed=7)
    one.title_edit.setText('single')
    one._update_all()
    path = os.path.join(tempfile.mkdtemp(), 'single.arar')
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (path, ''))
    one.save_session()
    rows, _, _ = rcm.file_ages(path)
    close(rows[0]['age'], one.plateau['r']['mean'], 1e-9, 'single .arar: plateau age')
    close(rows[0]['ana'], one.plateau.get('e1s_ana', one.plateau['e1s_int']), 1e-9, '  analytical error')
    src = rcm._file_source(rows[0]['cfg'])
    assert src['std'] == 'FCs' and abs(src['std_age'] - 28.176) < 1e-6, src
    # an .arprob: one row per sample, its combined age
    a = pp.ProbabilityWindow()
    a._load_docs(a.samples[0], [pp.parse_arar(path)])
    a.samples[0].name = 'P'
    p2 = os.path.join(tempfile.mkdtemp(), 's.arprob')
    old = pp.dfold.get_save_file_name
    pp.dfold.get_save_file_name = lambda *a_, **k: (p2, '')
    a.save_session()
    pp.dfold.get_save_file_name = old
    rows, _, _ = rcm.file_ages(p2)
    close(rows[0]['age'], a.samples[0].wm()['wm'], 1e-12, '.arprob: the sample\'s combined age')
    # into the Combine page, recalculated onto the same calibration: unchanged
    win = rcm.RecalculatorWindow()
    tab = win.combine_tab
    old = rcm.dfold.get_open_file_names
    rcm.dfold.get_open_file_names = lambda *a_, **k: ([path], '')
    tab.load_files()
    rcm.dfold.get_open_file_names = old
    assert tab.table.text(0, 0) == 'single' and tab.table.source(0)['std'] == 'FCs'
    ok('plateau and sample ages load with their calibration')
    # the flux window: "auto" finds the hole from Position, a pick replaces it,
    # Apply recalculates and keeps the window open
    import flux_dialog as fd
    spath = save_stack(w)
    rcm.dfold.get_open_file_names = lambda *a_, **k: ([spath], '')
    tab.load_files()                       # the stack brings its disk flux
    rcm.dfold.get_open_file_names = old
    r = tab.filled_rows()[-1]
    m = tab.match(r)
    assert m is not None, tab.table.text(r, rcm.C_POS)
    key = m[0]
    holes = list(tab.disk_flux[key]['idx'])
    other = next(h for h in holes if tab.disk_flux[key]['idx'][h] != m[1])
    dlg = fd.FluxAssignDialog(tab, rcm._RecalcFluxHost(tab))
    row = tab.filled_rows().index(r)
    cb, choices = dlg._row_combos[row]
    k = next(i for i, (_l, v) in enumerate(choices)
             if isinstance(v, tuple) and (str(v[0]), str(v[1]), str(v[2])) == (key[0], key[1], other))
    cb.setCurrentIndex(k)
    dlg._apply()
    assert tab.match(r) == (key, tab.disk_flux[key]['idx'][other])
    btns = [b.text() for b in dlg.findChildren(QPushButton)]
    assert 'Apply' in btns and 'Close' in btns and not any('shown disk' in t for t in btns), btns
    assert dlg.tbl.horizontalHeaderItem(0).text() == 'Sample'
    dlg.close()
    ok('flux window: auto hole from Position, a picked hole replaces it, Apply and Close')
    # recalculated ages: one line per sample above the combined age
    tab.recalc()
    txt = tab.report.toPlainText()
    assert 'Combined age' in txt and txt.startswith('Combined age'), txt[:200]   # nothing moved
    r0 = tab.filled_rows()[0]
    src = dict(tab.table.source(r0), std_age=28.02)
    tab.table.cellWidget(r0, rcm.C_SRC).set_source(src)
    tab.recalc()
    txt = tab.report.toPlainText()
    head = txt.split('Combined age')[0]
    for d in tab.items:
        assert head.count(d['label'] + ':') == 1, (d['label'], head)
    ok('recalculated ages listed per sample before the combined age')
    # without a .jflux the combiner groups J errors by disk, like the stack
    tab.disk_flux.clear(); tab.overrides.clear(); tab.manual.clear()
    rows = tab.filled_rows()[:3]
    for r, pos in zip(rows, ('D/1', 'D/2', 'E/1')):
        tab.table.setItem(r, rcm.C_POS, QTableWidgetItem(pos))
    for r in tab.filled_rows()[3:]:
        tab.table.setItem(r, rcm.C_POS, QTableWidgetItem(''))
    tab.recalc()
    rho = tab.result['rho']
    idx = [next(i for i, d in enumerate(tab.items) if d['row'] == r) for r in rows]
    assert rho[idx[0], idx[1]] == 1.0 and rho[idx[0], idx[2]] == 0.0, rho
    ok('combiner without a .jflux: same disk shares J, another disk does not')


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


def test_arprob_as_sample():
    print('An .arprob session loaded into a sample, like a stack:')
    pp = load('5_age_distribution/age_distribution.py', 'pp_arprob')
    d = tempfile.mkdtemp()
    a = pp.ProbabilityWindow()
    for i, rstd in enumerate((5.70, 5.72, 5.75)):
        st = [dict(label=chr(65 + k), age=55.0, err2s=0.05, r4039=R, err_r4039=0.004,
                   pct39Ar=25, status='normal')
              for k, R in enumerate((11.40 + 0.01 * i, 11.41, 11.39, 11.405))]
        f = os.path.join(d, f's{i}.arar')
        json.dump(dict(title=f'S{i}', cfg=dict(PHIL, ageFluxMode='ratio_R_std', rStd=rstd,
                                               rStdErrPct=0.02 + 0.01 * i, stepSigma=1),
                       steps=st), open(f, 'w', encoding='utf-8'))
        s = a.samples[0] if i == 0 else a.add_sample(pp.Sample(f'S{i}', pp._PALETTE[i]))
        a._load_docs(s, [pp.parse_arar(f)])
    a.samples[2].manual[1] = 'out'
    path = os.path.join(d, 'three.arprob')
    old_save, old_open = pp.dfold.get_save_file_name, pp.dfold.get_open_file_names
    pp.dfold.get_save_file_name = lambda *a_, **k: (path, '')
    a.save_session()
    pp.dfold.get_open_file_names = lambda *a_, **k: ([path], '')
    ages = [list(x.ages) for x in a.samples]
    for answer in (QMessageBox.Yes, QMessageBox.No):
        b = pp.ProbabilityWindow()
        old_q = QMessageBox.question
        QMessageBox.question = staticmethod(lambda *a_, **k: answer)
        try:
            b.load_into(b.samples[0])
        finally:
            QMessageBox.question = old_q
        if answer == QMessageBox.Yes:
            assert [x.name for x in b.samples] == ['S0', 'S1', 'S2']
            for x, ref in zip(b.samples, ages):
                for u, v in zip(x.ages, ref):
                    assert abs(u - v) <= 1e-12 * abs(v), (u, v)
            assert b.samples[2].manual[1] == 'out'
            assert b.samples[1].j_cfg['rStd'] == 5.72
        else:
            assert len(b.samples) == 1 and len(b.samples[0].ages) == 12
            for u, v in zip(b.samples[0].ages, sum(ages, [])):
                assert abs(u - v) <= 1e-9 * abs(v), (u, v)
            assert b.samples[0].manual[9] == 'out'
    pp.dfold.get_save_file_name, pp.dfold.get_open_file_names = old_save, old_open
    ok('Yes: one sample each, as saved; No: all points in one sample, ages unchanged')


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
    from PyQt5.QtGui import QColor
    ad.QColorDialog.getColor = staticmethod(lambda *a, **k: QColor('#123456'))
    p._pick_color()
    assert p.s.color == '#123456' and '#123456' in p.sw.styleSheet(), p.s.color
    assert [ad.QColorDialog.customColor((k % 8) * 2 + k // 8).name() for k in range(10)] == ad._PALETTE
    ok('clicking the swatch sets the sample colour')


def test_age_distribution_normalised():
    """Normalised: every curve peaks at 100 %; the export can give each
    sample its own panel, with the pooled curve in each."""
    print('Age distribution, Normalised and one panel per sample:')
    ad = load('5_age_distribution/age_distribution.py', 'ad_norm')
    from matplotlib.figure import Figure
    import export
    win = ad.ProbabilityWindow()
    p1 = win.panels[0]
    p1.data.setPlainText('\n'.join(f'{55 + 0.02 * k}\t0.01' for k in range(-6, 7)))
    p1._apply_paste()
    win.add_sample()
    p2 = win.panels[1]
    p2.data.setPlainText('\n'.join(f'{55 + 0.05 * k}\t0.3' for k in range(-4, 5)))
    p2._apply_paste()
    win.pool_on = True
    win.ymode_sel.setCurrentIndex(3)
    spec = export.default_spec(win._export_panels(), win._export_options())
    for split, n_ax in ((False, 1), (True, 2)):
        spec['options']['split'] = split
        fig = Figure(figsize=(6, 6))
        win._export_render(fig, spec)
        assert len(fig.axes) == n_ax, len(fig.axes)
        for a in fig.axes:
            tops = [max(l.get_ydata()) for l in a.get_lines() if len(l.get_ydata()) > 50]
            assert tops and all(abs(t - 100) < 1e-6 for t in tops), tops
    ok('every curve peaks at 100 %; one panel per sample with the pooled curve')
    # both samples in one panel; a legend below the stack is one legend
    spec['options'].update(split=True, panel_0='1', panel_1='1')
    spec['legend'] = 'below'
    fig = Figure(figsize=(6, 6))
    win._export_render(fig, spec)
    assert len(fig.axes) == 1, len(fig.axes)
    spec['options'].update(panel_1='2')
    fig = Figure(figsize=(6, 6))
    win._export_render(fig, spec)
    legs = [a.get_legend() for a in fig.axes if a.get_legend() is not None]
    labels = [t.get_text() for t in legs[0].get_texts()]
    assert len(fig.axes) == 2 and len(legs) == 1 and legs[0].axes is fig.axes[-1], legs
    assert sum(l.lstrip().startswith('Pooled') for l in labels) == 1, labels
    assert labels[-1].lstrip().startswith('Pooled'), labels
    assert 'n=22/22' in labels[-1], labels[-1]          # included / all analyses
    ok('samples share a panel when asked; a legend below the stack is drawn once')


def test_plateau_calibration_dialog():
    print('Plateau & isochron, Calibration…: monitor and route:')
    w = pl.PlateauWindow()
    w.raw_steps = steps(11.30, scatter=4e-4)
    w._update_all()
    J0 = am.cfg_J(w.cfg)[0]
    t0 = w.plateau['r']['mean'] if w.plateau else w.steps_c[0]['age']
    assert 'Phillips' in w._lbl_route.text()

    def pick(monitor, index=0, typed=None):
        def run(dlg):
            dlg._mon_cb.setCurrentText(monitor)
            dlg._route_cb.setCurrentIndex(index)
            if typed:
                dlg._cust_box.setChecked(True)
                for le, v in zip((dlg._c_age, dlg._c_age_s, dlg._c_lam, dlg._c_lam_s), typed):
                    le.setText(repr(v))
            return QDialog.Accepted
        pl._CalibrationDialog.exec_ = run
        w._open_calibrate()

    pick('TCs')
    close(am.cfg_J(w.cfg)[0] / J0, 1.0112, 1e-12, 'TCs → FCs → Phillips: J × R(TCs/FCs)')
    assert w.cfg['route']['steps'][0]['to'] == 'FCs' and 'TCs' in w._lbl_route.text()
    comps = dict(am.systematic_components(55.0, w.cfg))
    assert 'R(TCs/FCs), Renne et al. (1998)' in comps
    pick('FCs')
    close(am.cfg_J(w.cfg)[0] / J0, 1.0, 1e-12, 'back to FCs: J as before')
    close(w.plateau['r']['mean'] if w.plateau else w.steps_c[0]['age'], t0, 1e-9, 'and the same age')
    pick('FCs', typed=(28.02, 0.16, 5.543e-10, 0.0))
    ing = math.expm1(5.543e-10 * 28.02e6)
    close(am.monitor_ingrowth(w.cfg) / am.cfg_J(w.cfg)[0], w.cfg['rStd'], 1e-9, 'typed age and λ: J from them')
    assert w.cfg['route']['calibration']['age'] == 28.02
    close(am.monitor_ingrowth(w.cfg), ing, 1e-15, 'ingrowth of the typed calibration')
    w._le_stdAge.setText('28.201')
    w._le_stdAgeErr.setText('0.023')
    w._le_lambda.setText(repr(5.463e-10))
    w._le_lambdaErr.setText(repr(5.4e-12))
    assert w.cfg['route']['calibration'] == 'Kuiper et al. (2008)', w.cfg['route']['calibration']
    ok('typed values that match a calibration take it; the route line follows')

    # ages without ratios in R mode: a calibration change recasts them by the
    # change alone, whatever J placeholder the session holds
    w2 = pl.PlateauWindow()
    w2.raw_steps = synth.plateau_steps()
    w2._update_all()
    a0 = [x['age'] for x in w2.steps_c]
    pick_w = w
    w = w2
    pick('FCs')
    close(max(abs(x['age'] - y) for x, y in zip(w2.steps_c, a0)), 0.0, 1e-9,
          'R mode, ages only: no change, same ages')
    pick('TCs')
    lam = 5.463e-10
    t1 = math.log1p(1.0112 * math.expm1(lam * a0[0] * 1e6)) / lam / 1e6
    close(w2.steps_c[0]['age'], t1, 1e-9, '  and TCs scales e^(λt) − 1 by R(TCs/FCs)')
    w = pick_w

    # a route built in Standards from the dialog is selected when Standards closes
    import standards as stdmod
    def build(ed):
        rt = ed.pages[2]
        QInputDialog.getText = staticmethod(lambda *a, **k: ('ACs straight to A1', True))
        rt._add()
        rt.mon.setCurrentText('ACs')
        rt.next.setCurrentIndex([k for k, c in enumerate(rt._cands) if c['to'] == 'A1'][0])
        rt._add_step()
        ed._save()
        return 0
    stdmod.StandardsEditor.exec_ = build
    seen = {}
    def run(dlg):
        dlg._open_standards()
        seen['mon'] = dlg._mon_cb.currentText()
        seen['route'] = dlg._route_cb.currentText()
        return QDialog.Accepted
    pl._CalibrationDialog.exec_ = run
    w._open_calibrate()
    assert seen == {'mon': 'ACs', 'route': 'ACs straight to A1'}, seen
    assert w.cfg['route']['steps'][0]['to'] == 'A1' and w.cfg['route']['calibration'] == 'A1 astronomical age'
    ok('route built in Standards from the dialog is picked and applied, no restart')
    pick('FCs')
    def choose(ed):                   # the route exists already: select it, close
        rt = ed.pages[2]
        rt.list.setCurrentRow([r['name'] for r in rt.rows].index('ACs straight to A1'))
        return 0
    stdmod.StandardsEditor.exec_ = choose
    pl._CalibrationDialog.exec_ = run
    w._open_calibrate()
    assert seen == {'mon': 'ACs', 'route': 'ACs straight to A1'}, seen
    assert w.cfg['monitor'] == 'ACs'
    ok('an existing route selected in Standards is picked when Standards closes')

    # a stack on an FCs .jflux, calibrated to TCs: warned, and the hole R read as TCs
    ws = make_stack()
    pl._CalibrationDialog.exec_ = lambda dlg: (dlg._mon_cb.setCurrentText('TCs'),
                                               QDialog.Accepted)[1]
    shown.clear()
    ws._open_calibrate()
    assert any('measured on FCs' in m[1][1] for m in shown), shown
    eff = ws._eff_cfg_for(0)
    close(eff['rStd'], 5.70 / 1.0112, 1e-9, 'stack hole R read as a TCs ratio')
    ok('flux on another monitor than the calibration: warned')

    w3 = pl.PlateauWindow()
    w3.raw_steps = steps(11.30, scatter=4e-4)
    w3._update_all()
    pl._CalibrationDialog.exec_ = lambda dlg: (dlg._full_cb.setChecked(True), QDialog.Accepted)[1]
    w3._open_calibrate()
    assert w3.cfg['fullSystematics'] and w3._lbl_route.text().endswith('full external')
    pl._CalibrationDialog.exec_ = lambda dlg: (dlg._full_cb.setChecked(False), QDialog.Accepted)[1]
    w3._open_calibrate()
    assert not w3.cfg['fullSystematics'] and 'full external' not in w3._lbl_route.text()
    ok('Full external set under Advanced, shown in the route line')
    stdmod.save(stdmod.load_default())
    am.reload_standards()


def test_age_distribution_calibration():
    print('Age distribution, Calibration…: one bar, samples on different monitors:')
    ad = load('5_age_distribution/age_distribution.py', 'ad_cal')
    R, sR, _F, _sF = load_ratios()
    win = ad.ProbabilityWindow()
    for sm in list(win.samples):
        win.remove_sample(sm)
    for nm in ('A', 'B'):
        smp = win.add_sample()
        smp.name = nm
        smp.input_mode = 'ratio'; smp.reject = 'none'
        smp.j_cfg.update(CFG_R)
        smp.r_vals = np.array(R[:10]); smp.r_uncs = np.array(sR[:10])
        smp.recompute()
    win._sync_calib_bar()
    assert 'Phillips' in win.route_lbl.text() and not win.mon_info.text()
    I0 = am.monitor_ingrowth(win.samples[1].j_cfg)
    wm0 = win.samples[1].wm()['wm']

    def run(dlg):
        dlg._mon_cb.setCurrentText('TCs')
        dlg._sample_cbs[0].setChecked(False)          # A stays on FCs
        return QDialog.Accepted
    ad.CalibrationDialog.exec_ = run
    win._open_calibrate()
    a, b = win.samples
    assert am.cfg_monitor(a.j_cfg)[0] == 'FCs' and am.cfg_monitor(b.j_cfg)[0] == 'TCs'
    close(am.monitor_ingrowth(b.j_cfg) / I0, 1.0112, 1e-12, 'B on TCs → FCs → Phillips: e^(λt_M) − 1 × 1.0112')
    assert b.wm()['wm'] > wm0
    assert '2 calibrations' in win.route_lbl.text() and not win.mon_info.text()
    ok('A stays on FCs, B moves to TCs; same end calibration, so no warning')
    win.pool_on = True
    pr = win.pool()
    ages = [a.wm()['wm'], b.wm()['wm']]
    C, labels = am.systematic_matrix(ages, [a.j_cfg, b.j_cfg])
    shared = [l for l in labels if (C[:, labels.index(l)] != 0).all()]
    assert set(shared) == {'A1 astronomical age', 'R(FCs/A1), Phillips et al. (2022)',
                           'λ, Min et al. (2000)'}, shared
    assert any(l.startswith('R(TCs/FCs)') for l in labels)
    ok('pooled: A1 age, R(FCs/A1) and λ shared; R(TCs/FCs) on B only')

    def run2(dlg):
        dlg._route_cb.setCurrentIndex([dlg._route_cb.itemText(i)
                                       for i in range(dlg._route_cb.count())].index('Carter et al. (2025)'))
        dlg._sample_cbs[1].setChecked(False)
        return QDialog.Accepted
    ad.CalibrationDialog.exec_ = run2
    win._open_calibrate()
    assert 'different calibrations' in win.mon_info.text()
    ok('samples ending on different calibrations: noted under the bar')
    ad.CalibrationDialog.exec_ = lambda dlg: (dlg._full_cb.setChecked(True), QDialog.Accepted)[1]
    win._open_calibrate()
    assert all(s_.j_cfg['fullSystematics'] for s_ in win.samples)
    assert win.route_lbl.text().endswith('full external')
    ok('Full external from the dialog, for every sample')


def test_route_survives_files():
    print('A route through the files: stack on TCs → FCs → Phillips, saved and read back:')
    w = make_stack()
    pl._CalibrationDialog.exec_ = lambda dlg: (dlg._mon_cb.setCurrentText('TCs'), QDialog.Accepted)[1]
    w._open_calibrate()
    w._stack_method, w._wm_n_basis = am.METHOD_WM, 'aliquots'
    r, a, i, e = w._combined_plateau()
    assert any(k.startswith('R(TCs/FCs)') for k, _ in am.systematic_components(r['mean'], w._eff_cfg_for(0)))
    path = save_stack(w)
    w2 = pl.PlateauWindow(); w2._toggle_stack_mode(True)
    QFileDialog.getOpenFileNames = staticmethod(lambda *a_, **k: ([path], ''))
    w2._stack_load()
    w2._stack_method, w2._wm_n_basis = am.METHOD_WM, 'aliquots'
    r2, a2, i2, e2 = w2._combined_plateau()
    close(r2['mean'], r['mean'], 1e-12, 'Plateau stack reloaded: age')
    close(e2, e, 1e-12, '  external 1σ')
    assert am.cfg_monitor(w2._eff_cfg_for(0))[0] == 'TCs'
    pp._STAT_METHOD = am.METHOD_WM
    win = prob_pool(path, QMessageBox.Yes)
    res = win.pool()
    close(res['wm'], r['mean'], 1e-9, 'Age distribution, one sample per aliquot: age')
    close(res['sigma_ext'], e, 1e-6, '  external 1σ (R(TCs/FCs) kept)')
    assert all(am.cfg_monitor(s_.j_cfg)[0] == 'TCs' for s_ in win._active())


def test_recalculator_and_converter_calibration():
    print('Recalculator target and Data converter source, on routes:')
    rc = load('6_recalculator/recalculator.py', 'rc_route')
    box = rc.CalibrationBox()
    # a published TCs age (TCs 28.34 Ma, S&J λ) onto FCs · Phillips, through the R
    # chosen for TCs: the same as before routes
    R, sR, ref = 1.0112, 0.001, 'Renne, P.R. et al. (1998) Chemical Geology 145, 117–152'
    old = am.recalculate(55.0, 28.34, 5.543e-10, rc.target_cfg('Phillips et al. (2022)'), R, sR,
                         'R(TCs/FCs)')
    cfg = box.route_for('TCs', R, sR, ref)
    new = am.recalculate(55.0, 28.34, 5.543e-10, cfg, am.cfg_monitor(cfg)[1], 0.0, '')
    close(new['age'], old['age'], 1e-12, 'TCs data onto FCs · Phillips: age as before')
    close(math.sqrt(sum(v * v for _, v in new['components'])),
          math.sqrt(sum(v * v for _, v in old['components'])), 1e-12, '  and its systematic error')
    # the target route TCs → FCs → Phillips: TCs data go through it as it is
    import standards as S
    box.set_route(S.default_route(am.STANDARDS, 'TCs'))
    cfg2 = box.route_for('TCs')
    close(am.cfg_monitor(cfg2)[1], 1.0112, 1e-12, 'target route on TCs used for TCs data')
    assert 'TCs' in box.name()
    ok('the Recalculator target is a route; the line shows it')
    win = rc.RecalculatorWindow()
    tab = win.recalc_tab
    tab.in_sig.setCurrentText('1σ')
    tab.inp.setCurrentCell(0, 0)
    tab.inp.paste('S1\t55.0\t0.010\t\t0.10\nS2\t55.0\t0.010\t0.030\t0.10\nS3\t55.0\t0.010\t\t\n'
                  'S4\t55.0\t\t0.060\t0.10\nS5\t55.0\t\t0.040\t0.10')
    res = {d['label']: d for d in tab.results}
    f = abs(am.recalculate(55.0, tab.src.current()['std_age'], tab.src.current()['lam'],
                           tab.tgt.cfg(), 1.0, 0.0, '')['factor'])
    want = math.hypot(0.010, am.dt_dlnR(55.0, tab.src.current()['lam']) * 0.001) * f
    close(res['S1']['int'], want, 1e-12, 'analytical + J % → internal')
    close(res['S2']['int'], 0.030 * f, 1e-12, 'a typed internal error is kept')
    assert res['S3']['int'] is None
    ok('no J %: analytical only, as before')
    jt = am.dt_dlnR(55.0, tab.src.current()['lam']) * 0.001
    close(res['S4']['ana'], math.sqrt(0.060 ** 2 - jt ** 2) * f, 1e-12, 'internal + J % → analytical')
    close(res['S4']['int'], 0.060 * f, 1e-12, '  internal kept')
    assert res['S5']['ana'] is None and tab.bad_j == 1 and 'larger than' in tab.status.text()
    ok('an internal error below the J term alone gives no analytical error, and says so')
    dc = load('1_data_converter/data_converter.py', 'dc_route')
    d = dc.ConvertDialog('x.xlsx', ['Sample', 'Age'], 'massspec')
    dc.CalibrationDialog.exec_ = lambda dlg: (dlg._mon_cb.setCurrentText('TCs'), QDialog.Accepted)[1]
    d._open_calibrate()
    assert d._mon_combo.monitor() == 'TCs' and d._route_lbl.text().startswith('TCs → FCs')
    close(d.fc_age_ma(), am.monitor_age(am.route_cfg(S.default_route(am.STANDARDS, 'TCs'))), 1e-6,
          'converter: the TCs age on the chosen calibration')


if __name__ == '__main__':
    test_launcher_and_about()
    test_every_window_opens()
    test_no_wheel_on_dropdowns()
    test_isochron_atmospheric_in_view()
    test_standards_window()
    test_data_folders()
    test_unicode_session_files()
    test_arprob_as_sample()
    test_converter_file_names()
    test_fonts()
    test_position_ident()
    test_disk_flux_round_trip()
    test_error_modes()
    test_one_J_everywhere()
    test_converter_file_from_R()
    test_wendt_carl_threshold()
    test_stack_vs_probability()
    test_combiner_loads_files()
    test_imported_J_in_internal()
    test_flux_on_disk_vs_external()
    test_gui_paths()
    test_age_distribution_data_box()
    test_age_distribution_normalised()
    test_plateau_calibration_dialog()
    test_age_distribution_calibration()
    test_route_survives_files()
    test_recalculator_and_converter_calibration()
    print('ALL PASSED')
