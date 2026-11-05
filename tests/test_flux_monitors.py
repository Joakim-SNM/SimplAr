"""Flux monitors: its statistics against independent values, on made-up data.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_flux_monitors.py
"""
import copy, importlib.util, math, os, sys, tempfile
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('SIMPLAR_DATA_DIR', tempfile.mkdtemp())
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox
import arar_math as am
import synthetic_data as synth

app = QApplication.instance() or QApplication([])
shown = []
for _n in ('information', 'warning', 'critical'):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: shown.append((_n, a[1:3])) or 0))

spec = importlib.util.spec_from_file_location(
    'fm', os.path.join(ROOT, '2_flux_monitors', 'flux_monitors.py'))
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)


def close(a, b, tol, what):
    ok = abs(a - b) <= tol * max(1.0, abs(b))
    print(f"  {'ok ' if ok else 'BAD'} {what:<52} {a:.10g}  vs  {b:.10g}")
    assert ok, what


def window():
    w = fm.FluxMonitorsWindow()
    return w


def test_reml_vs_metafor():
    print('REML random effects vs metafor (BCG trials, log risk ratio):')
    bcg = [(4, 119, 11, 128), (6, 300, 29, 274), (3, 228, 11, 209),
           (62, 13536, 248, 12619), (33, 5036, 47, 5761), (180, 1361, 372, 1079),
           (8, 2537, 10, 619), (505, 87886, 499, 87892), (29, 7470, 45, 7232),
           (17, 1699, 65, 1600), (186, 50448, 141, 27197), (5, 2493, 3, 2338),
           (27, 16886, 29, 17825)]
    data = [{'v': math.log((a / (a + b)) / (c / (c + d))),
             'e': math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))} for a, b, c, d in bcg]
    r = window().remlRandomEffects(data, 'v', 'e')
    close(r['tau2'], 0.3132, 2e-4, 'tau^2')
    close(r['weightedMean'], -0.7145, 1e-4, 'mu')
    close(r['se_1sigma'], 0.1798, 1e-4, 'se')


def test_weighted_mean_and_j():
    print('Without rejection or REML: the weighted mean of arar_math, J from R:')
    w = window()
    w.use_reml_var.set(False); w.use_nmad_var.set(False); w.use_error_filter_var.set(False)
    rows = [r for r in synth.monitor_rows(1)['E1'] if r['uncertainty40_39'] > 0]
    w._set_loaded_data({'E1': copy.deepcopy(rows)}, 'direct', True, 'R', 'synthetic')
    w._run_analysis('test')
    res = w.results['ratio4039Results']['E1']
    ref = am.weighted_mean([r['value40_39'] for r in rows], [r['uncertainty40_39'] for r in rows])
    close(res['weightedMean'], ref['mean'], 1e-12, 'weighted mean R')
    close(res['fe_se_1s'], ref['sem'], 1e-10, 'SE (1σ)')
    close(res['mswd'], ref['mswd'], 1e-10, 'MSWD')
    over = res['mswd'] > am.mswd_critical(len(rows))
    close(res['od_se_1s'], ref['sem'] * (math.sqrt(res['mswd']) if over else 1.0), 1e-10,
          'SE × √MSWD only above the Wendt & Carl limit')
    head, table = w._summary_rows()
    row = dict(zip(head, table[0]))
    c = am.fc_ingrowth(28.176, 5.463e-10, 0.0115)
    close(row['J'], c / row['Reported R'], 1e-12, 'J = (e^(λt) − 1)/R')
    close(row['J SE (1σ, R only)'] / row['J'], row['Reported SE (1σ)'] / row['Reported R'],
          1e-12, 'σJ/J = σR/R')


def test_total_gas():
    print('One point per crystal: ³⁹Ar-weighted total gas of the steps:')
    w = window()
    steps = [r for r in synth.monitor_rows(1)['E5'] if r['grain_id'] == 'G2']
    R, s, _ = w._pool_value(steps)
    f = np.array([r['ar39'] for r in steps]); f = f / f.sum()
    v = np.array([r['value40_39'] for r in steps]); e = np.array([r['uncertainty40_39'] for r in steps])
    close(R, float(f @ v), 1e-12, 'R = Σ fᵢRᵢ')
    close(s, float(np.sqrt(np.sum((f * e) ** 2))), 1e-12, 'σ = √Σ fᵢ²σᵢ²')


def test_window_and_export():
    print('Window: synthetic monitors and .arar files, every view, xlsx export:')
    w = window(); w.show()
    w._set_loaded_data(copy.deepcopy(synth.monitor_rows(1)), 'direct', True, 'R', 'synthetic')
    w._run_analysis('test')
    for label, key in w._VIEWS.items():
        w._view_label_var.set(label); w.crystal_view_var.set(key); w.update_plot()
    d = tempfile.mkdtemp()
    w.arar_paths = synth.write_arar_files(d)
    w.arar_opts = dict(crystal_rule='auto', group_by='position')
    w.file_source = 'arar'; w.loaded_source = None   # the file tab, not the pasted data
    w.input_file = w.arar_paths[0]; w.input_name = 'synthetic files'
    w.analyze_data()
    cats = list(w.results['ratio4039Results'])
    assert cats == ['TEST1-E3', 'TEST1-E7'], cats
    assert any(r == 'excluded in file' for r in w.manual_exclusions['TEST1-E3'].values())
    path = os.path.join(d, 'out.xlsx')
    w._write_results_xlsx(path)
    from openpyxl import load_workbook
    sheets = load_workbook(path).sheetnames
    assert sheets[:3] == ['Results', 'Points', 'Steps'], sheets
    assert not shown, shown
    print(f"  ok  {len(cats)} positions from .arar files; sheets {', '.join(sheets)}")


if __name__ == '__main__':
    test_reml_vs_metafor()
    test_weighted_mean_and_j()
    test_total_gas()
    test_window_and_export()
    print('ALL PASSED')
