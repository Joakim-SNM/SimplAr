"""Every tool opens, saves and reads its files, on any operating system.

This is the test the Windows / macOS / Linux runs on GitHub lean on: file
names with characters Windows forbids, text files with σ, λ and ± in them,
the data folder, the fonts and the launcher.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_tools_open.py
"""
import importlib.util, json, os, subprocess, sys, tempfile
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
DATA = tempfile.mkdtemp()
os.environ['SIMPLAR_DATA_DIR'] = DATA
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog
from PyQt5.QtGui import QFontDatabase, QFontInfo

app = QApplication.instance() or QApplication([])
shown = []
for _n in ('information', 'warning', 'critical'):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: shown.append((_n, a[1:3])) or 0))

import arar_math as am
import data_folders as dfold
import synthetic_data as synth


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def ok(what):
    print(f'  ok  {what}')


def test_launcher_and_about():
    print('Launcher and the Methods & Reference document:')
    L = load('launcher.py', 'launcher')
    w = L.ToolLauncher()
    for label, rel in L.TOOLS:
        assert os.path.isfile(os.path.join(ROOT, *rel)), rel
    ok(f'{len(L.TOOLS)} tool scripts where the launcher looks for them')
    html = L.build_about_html(ROOT)
    assert 'Recalculator' not in html and '<img src=' in html
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
    ok('all five tool folders created')


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


if __name__ == '__main__':
    test_launcher_and_about()
    test_every_window_opens()
    test_data_folders()
    test_unicode_session_files()
    test_converter_file_names()
    test_fonts()
    assert not [m for m in shown if m[0] != 'information'], shown
    print('ALL PASSED')
