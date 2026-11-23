"""Flux monitors and Flux regression: the statistics against independent values,
the .xlsx export read back by Flux regression, the position shift and the J
correlation between disks, on made-up data.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_flux.py
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
import importlib.util
import json
import math

import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox, QTableWidgetItem
from openpyxl import Workbook

import arar_math as am
import synthetic_data as synth

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


def close(a, b, tol, what):
    ok = abs(a - b) <= tol * max(1.0, abs(b))
    print(f"  {'ok ' if ok else 'BAD'} {what:<52} {a:.10g}  vs  {b:.10g}")
    assert ok, what


# --------------------------------------------------------------------------
# Flux monitors: its statistics against independent values, on made-up data.
# --------------------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    'fm', os.path.join(ROOT, '2_flux_monitors', 'flux_monitors.py'))
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)


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
    r = window().reml_random_effects(data, 'v', 'e')
    close(r['tau2'], 0.3132, 2e-4, 'tau^2')
    close(r['weighted_mean'], -0.7145, 1e-4, 'mu')
    close(r['se_1sigma'], 0.1798, 1e-4, 'se')


def test_weighted_mean_and_j():
    print('Without rejection or REML: the weighted mean of arar_math, J from R:')
    w = window()
    w.use_reml_var.set(False); w.use_nmad_var.set(False); w.use_error_filter_var.set(False)
    rows = [r for r in synth.monitor_rows(1)['E1'] if r['uncertainty40_39'] > 0]
    w._set_loaded_data({'E1': copy.deepcopy(rows)}, 'direct', True, 'R', 'synthetic')
    w._run_analysis('test')
    res = w.results['ratio4039_results']['E1']
    ref = am.weighted_mean([r['value40_39'] for r in rows], [r['uncertainty40_39'] for r in rows])
    close(res['weighted_mean'], ref['mean'], 1e-12, 'weighted mean R')
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
    assert w.crystal_info_lbl.text() == '45 crystals, 33 fusion, 12 step-heated (3 steps).', \
        w.crystal_info_lbl.text()
    for label, key in w._VIEWS.items():
        w._view_label_var.set(label); w.crystal_view_var.set(key); w.update_plot()
    d = tempfile.mkdtemp()
    w.arar_paths = synth.write_arar_files(d)
    w.arar_opts = dict(crystal_rule='auto', group_by='position')
    w.file_source = 'arar'; w.loaded_source = None   # the file tab, not the pasted data
    w.input_file = w.arar_paths[0]; w.input_name = 'synthetic files'
    w.analyze_data()
    cats = list(w.results['ratio4039_results'])
    assert cats == ['TEST1-E3', 'TEST1-E7'], cats
    assert any(r == 'excluded in file' for r in w.manual_exclusions['TEST1-E3'].values())
    path = os.path.join(d, 'out.xlsx')
    w._write_results_xlsx(path)
    from openpyxl import load_workbook
    sheets = load_workbook(path).sheetnames
    assert sheets[:3] == ['Results', 'Points', 'Steps'], sheets
    assert not shown, shown
    print(f"  ok  {len(cats)} positions from .arar files; sheets {', '.join(sheets)}")


# --------------------------------------------------------------------------
# Flux regression reads an .xlsx export of the Flux monitors tool.
# --------------------------------------------------------------------------

def test_flux_regression_import():
    """Flux regression reads an .xlsx export of the Flux monitors tool."""
    spec = importlib.util.spec_from_file_location(
        'flux_regression', os.path.join(ROOT, '3_flux_regression', 'flux_regression.py'))
    fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
    TMP = tempfile.mkdtemp()

    def close_rel(a, b, tol, what):
        assert abs(a - b) <= tol * abs(b), f'{what}: {a} vs {b}'
        ok(what)

    def write_export(path, data, ratio=True):
        """An export laid out as FluxMonitorsWindow writes it (Results sheet)."""
        wb = Workbook(); ws = wb.active; ws.title = 'Results'
        ws.cell(1, 1, 'WEIGHTED MEAN RESULTS'); ws.cell(2, 1, 'Run: test')
        vn = 'R' if ratio else 'Value'
        header = ['Category', 'Points averaged', 'n used', 'n total', 'MSWD', 'p',
                  'MSWD 95% limit (W&C)', 'Overdispersed (MSWD > W&C)', f'{vn} weighted mean',
                  'SE (1σ)', 'SE, × √MSWD if overdispersed (1σ)', f'{vn} REML mean',
                  'REML SE (1σ)', 'τ (1σ)', f'Reported {vn}', 'Reported SE (1σ)',
                  'Reported SE (2σ)', 'Method', 'MSWD rejection']
        if ratio:
            header += ['J', 'J SE (1σ, R only)', 'J SE (2σ)', 'J % (1σ)']
        for c, h in enumerate(header, 1):
            ws.cell(4, c, h)
        for r, (cat, R, s) in enumerate(data, 5):
            ws.cell(r, 1, cat)
            ws.cell(r, header.index(f'Reported {vn}') + 1, R)
            if s is not None:
                ws.cell(r, header.index('Reported SE (1σ)') + 1, s)
            if ratio:
                ws.cell(r, header.index('J') + 1, 0.0027356)    # another calibration, on purpose
        r0 = 5 + len(data) + 1
        ws.cell(r0, 1, 'ANALYSIS PARAMETERS')
        ws.cell(r0 + 1, 1, 'Calibration for J: Test (2000)  FC 28.0 ± 0.1 Ma (1σ), λ 5.5e-10 ± 0 /yr (1σ)')
        wb.save(path)

    DATA = [('C1', 5.67027, 0.0014673), ('C13', 5.70837, 0.0032229),
            ('Blank', 5.6, 0.01), ('C2', 5.67488, None)]
    PATH = os.path.join(TMP, 'export.xlsx')
    write_export(PATH, DATA)

    def cell(w, i, col):
        return float(w.disk_widgets[0].table_widget.item(i, col).text())

    print('Position labels and the Results sheet:')
    for lab, want in [('C1', ('C', 1)), ('C-01', ('C', 1)), ('C 13', ('C', 13)),
                      ('E17', ('E', 17)), ('13', ('', 13)), ('Blank', None)]:
        assert fr.split_position_label(lab) == want, (lab, fr.split_position_label(lab))
    ok('C1, C-01, C 13, E17, 13 and a name without a number')
    rows, calib = fr.read_flux_monitors_export(PATH)
    assert [r['category'] for r in rows] == ['C1', 'C13', 'Blank', 'C2'] and rows[3]['R_sigma'] is None
    assert calib.startswith('Calibration for J:')
    ok('reported R, its 1σ SE and the calibration line')

    print('R mode:')
    w = fr.FluxRegressionCalculator()
    w.mode_r_radio.click()
    shown.clear()
    w.import_flux_monitors_export(PATH)          # default 7-hole design: no hole 13
    assert 'C13' in shown[-1][1][1] and 'design' in shown[-1][1][1]
    ok('a hole the disk design does not have is reported, not imported')
    dw = w.disk_widgets[0]
    dw.disk_combo.setCurrentText('2.5mm (13 holes)')
    shown.clear()
    w.import_flux_monitors_export(PATH)
    assert len(w.disk_widgets) == 1 and dw.disk_label_edit.text() == 'C'
    ok('the empty first disk takes label C')
    close_rel(cell(w, 0, 3), 5.67027, 1e-15, 'hole 1 holds the reported R')
    close_rel(cell(w, 0, 4), 0.0014673, 1e-15, 'and its 1σ SE')
    close_rel(cell(w, 12, 3), 5.70837, 1e-15, 'C13 goes to hole 13')
    assert dw.table_widget.cellWidget(0, 2).isChecked() and dw.table_widget.item(0, 0).text() == 'C1'
    ok('ticked for the regression and named after the category')
    assert shown and 'Blank' in shown[-1][1][1] and 'C2' in shown[-1][1][1]
    ok('a category without a hole number or an SE is reported, not imported')
    std = dw.get_disk_data()['standard_data']
    assert sorted(h['hole_number'] for h in std) == [1, 13]
    close_rel(std[0]['r_error'], 0.0014673, 1e-15, 'the regression reads it back at 1σ')

    w.input_sigma_2_btn.click()
    w.import_flux_monitors_export(PATH)
    close_rel(cell(w, 0, 4), 2 * 0.0014673, 1e-15, 'with 2σ input the SE is entered doubled')

    print('J mode:')
    wj = fr.FluxRegressionCalculator()
    wj.disk_widgets[0].disk_combo.setCurrentText('2.5mm (13 holes)')
    shown.clear()
    wj.import_flux_monitors_export(PATH)
    c = wj.monitor_ingrowth()
    close_rel(cell(wj, 0, 3), c / 5.67027, 1e-12, 'J = (e^(λt) − 1)/R with this tool\'s calibration')
    close_rel(cell(wj, 0, 4) / cell(wj, 0, 3), 0.0014673 / 5.67027, 1e-12, 'σJ/J = σR/R')
    assert 'different calibration' in shown[-1][1][1]
    ok('says when the J in the export used another calibration')

    print('Disks:')
    P2 = os.path.join(TMP, 'export_D.xlsx')
    write_export(P2, [('D3', 5.66, 0.002), ('C2', 5.675, 0.002)])
    w.import_flux_monitors_export(P2)
    assert len(w.disk_widgets) == 2 and w.disk_widgets[1].disk_label_edit.text() == 'D'
    ok('a label no disk has goes to a new disk when the others hold values')
    assert float(w.disk_widgets[0].table_widget.item(1, 3).text()) == 5.675
    ok('C2 goes to the existing disk C')

    print('Not a ratio export:')
    P3 = os.path.join(TMP, 'values.xlsx')
    write_export(P3, [('C1', 1.0, 0.1)], ratio=False)
    shown.clear()
    w.import_flux_monitors_export(P3)
    assert shown and shown[-1][0] == 'critical'
    ok('refused with a message')
    print('all flux-regression import tests passed')


# --------------------------------------------------------------------------
# Flux regression: position shift (x, y only), and J correlation between
# disks.
# --------------------------------------------------------------------------

def test_flux_cross_disk():
    """Flux regression: position shift (x, y only), and J correlation
    between disks."""
    spec = importlib.util.spec_from_file_location(
        'flux_regression', os.path.join(ROOT, '3_flux_regression', 'flux_regression.py'))
    fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)

    TMP = tempfile.mkdtemp()

    print('Position shift:')
    f = np.array([fr.COORDINATE_SCALE_MIN, fr.COORDINATE_SCALE_MAX])
    assert np.allclose(fr.LATERAL_SHIFT_CM * 1e4 * f, [15, 450])
    ok('x, y (cm): 15–450 μm')
    c3 = np.array([[0.3, -0.2, 20.0], [0.0, 0.5, 0.0]])
    np.random.seed(7)
    new = fr.perturb_positions(c3)
    np.random.seed(7)
    s = 0.015 * np.random.uniform(0.1, 3.0)
    old = c3 + np.random.normal(0, s, c3.shape)
    assert np.allclose(new[:, :2], old[:, :2], rtol=0, atol=1e-15)
    ok('x, y: normal draws scaled by one factor per iteration')
    assert np.array_equal(new[:, 2], c3[:, 2])
    ok('z (disk height) is not shifted')
    np.random.seed(1)
    d = np.array([fr.perturb_positions(np.zeros((1, 2)))[0] for _ in range(20000)])
    rms = fr.LATERAL_SHIFT_CM * np.sqrt((0.1**2 + 0.1 * 3.0 + 3.0**2) / 3)   # RMS of σ ~ U(0.1, 3)·0.015 cm
    assert abs(d.std(axis=0).mean() / rms - 1.0) < 0.03
    ok(f'x, y shift RMS {d.std(axis=0).mean()*1e4:.0f} μm (about 0.26 mm)')

    print('Cross-disk J correlation:')
    fr.MONTE_CARLO_ITERATIONS = 6000
    J0, ax, by, cz = 2.7e-3, 0.01, 0.005, 0.0005
    sig = J0 * 1e-3
    MON = [1, 4, 7, 10, 13]
    w = fr.FluxRegressionCalculator()
    w.add_disk()
    for k, (dw, lab, h) in enumerate(zip(w.disk_widgets, 'CD', (0.0, 20.0))):
        dw.disk_combo.setCurrentText('2.5mm (13 holes)')
        dw.disk_label_edit.setText(lab)
        dw.height_spinbox.setValue(h)
        coords = dw.get_disk_data()['coordinates']
        for hole in MON:
            x, y = coords[hole - 1]
            t = dw.table_widget
            t.setItem(hole - 1, 0, QTableWidgetItem(f'{lab}{hole}'))
            t.setItem(hole - 1, 3, QTableWidgetItem(repr(J0 * (1 + ax * x + by * y + cz * h))))
            t.setItem(hole - 1, 4, QTableWidgetItem(repr(sig)))
            t.cellWidget(hole - 1, 2).setChecked(True)
    assert all(len(dw.get_disk_data()['standard_data']) == 5 for dw in w.disk_widgets)

    out = {}

    def export(rtype, name):
        w.regression_info = {'type': rtype, 'complex_3d': False}
        path = os.path.join(TMP, name)
        fr.dfold.get_save_file_name = lambda *a, **k: (path, '')
        np.random.seed(3)
        w.export_flux_config()
        with open(path, encoding='utf-8') as fh:
            out[rtype] = json.load(fh)
        return path

    p3 = export('3d', 'global.jflux')
    cfg = out['3d']
    assert cfg.get('fit_id') and len(cfg['cross_disk_corr']) == 1
    blk = cfg['cross_disk_corr'][0]
    assert (blk['disk_a'], blk['disk_b']) == ('C', 'D')
    assert np.array(blk['J_corr']).shape == (13, 13)
    ok('a global 3D fit through two disks stores one 13 × 13 cross block')

    # Analytic check for the unknowns: J = G b, cov(b) = (Xᵀ W X)⁻¹ (value noise only;
    # the position shift adds little at these gradients).
    std = [s for dw in w.disk_widgets for s in dw.get_disk_data()['standard_data']]
    X = np.array([[*s['coordinates_3d'], 1.0] for s in std])
    Cb = np.linalg.inv(X.T @ X / sig**2)
    G = [np.array([[*h['coordinates_3d'], 1.0] for h in dw.get_disk_data()['all_data']])
         for dw in w.disk_widgets]
    cov = G[0] @ Cb @ G[1].T
    sd0 = np.sqrt(np.einsum('ij,jk,ik->i', G[0], Cb, G[0]))
    sd1 = np.sqrt(np.einsum('ij,jk,ik->i', G[1], Cb, G[1]))
    rho_an = cov / np.outer(sd0, sd1)
    unk = [i for i in range(13) if i + 1 not in MON]
    diff = np.abs(np.array(blk['J_corr'])[np.ix_(unk, unk)] - rho_an[np.ix_(unk, unk)]).max()
    assert diff < 0.05, diff
    ok(f'cross-disk ρ between unknowns matches g_i·C·g_j (max |Δρ| = {diff:.3f})')
    big = np.abs(np.array(blk['J_corr'])[np.ix_(unk, unk)]).max()
    assert big > 0.3, big
    ok(f'and it is far from zero (max |ρ| = {big:.2f})')

    flux = am.read_jflux(p3)
    kC, kD = (cfg['irradiation'], 'C'), (cfg['irradiation'], 'D')
    assert np.allclose(flux[kD]['xcorr'][kC], np.array(blk['J_corr']).T)
    ok('read back both ways (D→C is the transpose of C→D)')
    pts = [dict(match=(kC, 1), ident='C', irr=cfg['irradiation']),
           dict(match=(kD, 5), ident='D', irr=cfg['irradiation']),
           dict(match=(kC, 2), ident='C', irr=cfg['irradiation'])]
    rho = am.j_correlation(pts, flux)
    assert abs(rho[0, 1] - blk['J_corr'][1][5]) < 1e-12 and rho[0, 1] != 0
    assert abs(rho[0, 2] - cfg['disks'][0]['J_corr'][1][2]) < 1e-12
    ok('j_correlation uses the cross block between disks and J_corr within a disk')

    for name, again in [('.arar / .ararstack', am.deserialize_disk_flux(am.serialize_disk_flux(flux)))]:
        assert abs(am.j_correlation(pts, again)[0, 1] - rho[0, 1]) < 1e-12
        ok(f'kept through the {name} registry')
    lst = [dict(irradiation=k[0], disk_id=k[1], holes=v['holes'], J=v['J'].tolist(),
                J_sigma=v['J_sigma'].tolist(), J_corr=v['J_corr'].tolist(),
                fit=v.get('fit'), xcorr=am.serialize_xcorr(v)) for k, v in flux.items()]
    assert abs(am.j_correlation(pts, am.deserialize_disk_flux(lst))[0, 1] - rho[0, 1]) < 1e-12
    ok('kept through the Age distribution session list')

    mixed = dict(flux)
    other = am.jflux_entries(dict(cfg, fit_id='another-fit'))
    mixed[kD] = other[kD]
    assert am.j_correlation(pts, mixed)[0, 1] == 0.0
    ok('ignored when disk D comes from another fit')
    old = dict(cfg); old.pop('fit_id'); old.pop('cross_disk_corr')
    assert am.j_correlation(pts, am.jflux_entries(old))[0, 1] == 0.0
    ok('older files without the block: disks independent, as before')

    export('individual', 'individual.jflux')
    assert 'cross_disk_corr' not in out['individual']
    ok('individual-disk fits store no cross block (independent disks)')
    print('all cross-disk tests passed')


def test_flux_regression_session_and_minimum():
    """A session keeps each disk's label (it sets the .jflux disk_id), and the
    simple 3D fit asks for four monitors."""
    fr = load(os.path.join('3_flux_regression', 'flux_regression.py'), 'flux_regression_s')
    print('Flux regression session and 3D minimum:')
    w = fr.FluxRegressionCalculator()
    w.add_disk()
    for dw, lab in zip(w.disk_widgets, ('E', 'F')):
        dw.disk_label_edit.setText(lab)
    path = os.path.join(tempfile.mkdtemp(), 's.json')
    fr.dfold.get_save_file_name = lambda *a, **k: (path, '')
    fr.dfold.get_open_file_name = lambda *a, **k: (path, '')
    w.save_session()
    w2 = fr.FluxRegressionCalculator()
    w2.load_session()
    assert [dw.disk_label_edit.text() for dw in w2.disk_widgets] == ['E', 'F']
    ok('disk labels E, F come back from the session')
    w2.disk_widgets[1].disk_label_edit.setText('')
    w2.disk_widgets[0].table_widget.setItem(0, 3, QTableWidgetItem('0.0027'))
    w2.disk_widgets[0].table_widget.setItem(0, 4, QTableWidgetItem('0.0000027'))
    t = w2.results_widget
    t.setRowCount(2)
    for k, dw in enumerate(w2.disk_widgets):
        dd = dict(dw.get_disk_data(), disk_number=k + 1)
        w2._populate_results_table_row(k, dd['all_data'][0], dd, None, None, None, None, fr.MODE_J)
    assert [t.item(k, 1).text() for k in range(2)] == ['E', '2'], [t.item(k, 1).text() for k in range(2)]
    ok('the results table and its export name each disk by its label (number when empty)')
    assert t.item(0, 3).text() == '0.0027000000' and t.item(0, 4).text() == '0.0000027000', \
        (t.item(0, 3).text(), t.item(0, 4).text())
    import export as ex
    from openpyxl import load_workbook
    xp = os.path.join(tempfile.mkdtemp(), 'r.xlsx')
    ex.write_xlsx(xp, [['FLUX REGRESSION RESULTS'], ['J'], [t.item(0, 3).text()]])
    c = load_workbook(xp)['Results'].cell(row=3, column=1)
    assert c.value == 0.0027 and c.number_format == '0.0000000000', (c.value, c.number_format)
    ok('J and its error to 10 decimals, kept as written in the .xlsx')
    std = [dict(coordinates_3d=(x, y, z)) for x, y, z in ((0, 0, 0), (1, 0, 0), (0, 0, 2))]
    shown.clear()
    assert not w2._validate_regression_data([{}], std, False, '3d')
    assert 'at least 4' in shown[-1][1][1]
    ok('simple 3D with three monitors is refused')


def test_jflux_r_mode_mean():
    """In R mode the .jflux R is the mean of the R draws, the value the
    results table shows (not const / mean of const/R)."""
    fr = load(os.path.join('3_flux_regression', 'flux_regression.py'), 'flux_regression_r')
    print('.jflux in R mode:')
    fr.MONTE_CARLO_ITERATIONS = 2000
    w = fr.FluxRegressionCalculator()
    w.regression_mode = fr.MODE_R
    dw = w.disk_widgets[0]
    dw.disk_combo.setCurrentText('2.5mm (13 holes)')
    dw.disk_label_edit.setText('E')
    coords = dw.get_disk_data()['coordinates']
    for hole in (1, 4, 7, 10, 13):
        x, y = coords[hole - 1]
        t = dw.table_widget
        t.setItem(hole - 1, 3, QTableWidgetItem(repr(5.74 * (1 + 0.002 * x - 0.001 * y))))
        t.setItem(hole - 1, 4, QTableWidgetItem('0.0015'))
        t.cellWidget(hole - 1, 2).setChecked(True)
    w.regression_info = {'type': 'individual', 'complex_3d': False}
    dd = dw.get_disk_data()
    path = os.path.join(tempfile.mkdtemp(), 'r.jflux')
    fr.dfold.get_save_file_name = lambda *a, **k: (path, '')
    w.export_flux_config()
    with open(path, encoding='utf-8') as fh:
        d = json.load(fh)['disks'][0]
    for i, h in enumerate(dd['all_data']):
        if d['monitor'][i]:
            continue
        tab = w.predict_j_value(*h['coordinates_3d'], specific_standards=dd['standard_data'],
                                force_regression_type='2d', n_iterations=2000)[0]
        assert abs(d['R'][i] - tab) < 1e-12, (h['hole_number'], d['R'][i], tab)
    ok('unknowns: .jflux R equals the results-table R')


def test_flux_monitors_calibration():
    print('Flux monitors, Calibration…: J on a route:')
    from PyQt5.QtWidgets import QDialog
    w = window()
    w.use_reml_var.set(False); w.use_nmad_var.set(False); w.use_error_filter_var.set(False)
    rows = [r for r in synth.monitor_rows(1)['E1'] if r['uncertainty40_39'] > 0]
    w._set_loaded_data({'E1': copy.deepcopy(rows)}, 'direct', True, 'R', 'synthetic')
    w._run_analysis('test')
    head, table = w._summary_rows()
    J0 = dict(zip(head, table[0]))['J']
    fm.CalibrationDialog.exec_ = lambda dlg: (dlg._mon_cb.setCurrentText('TCs'), QDialog.Accepted)[1]
    w._open_calibrate()
    head, table = w._summary_rows()
    row = dict(zip(head, table[0]))
    close(row['J'] / J0, 1.0112, 1e-12, 'TCs → FCs → Phillips: J × R(TCs/FCs)')
    close(row['Reported R (FCs-equivalent)'], row['Reported R'] / 1.0112, 1e-12, 'FCs-equivalent R kept')
    assert any(l.startswith('Calibration route: TCs') for l in w._parameter_lines())
    assert w.route_lbl.text().startswith('TCs → FCs')
    ok('route line and export parameter line')


def test_flux_regression_route_chain():
    """Flux regression fits the measured R as it is; the .jflux carries the
    monitor; a tool reads the measured R and applies its own route."""
    print('Flux regression: measured R, monitor label, the tools apply their route:')
    from PyQt5.QtWidgets import QDialog
    import standards as S
    fr = load(os.path.join('3_flux_regression', 'flux_regression.py'), 'flux_regression_r')
    w = fr.FluxRegressionCalculator()
    w.mode_r_radio.click()
    dw = w.disk_widgets[0]
    dw.disk_combo.setCurrentText('2.5mm (13 holes)')
    dw.disk_label_edit.setText('C')
    meas = {1: 5.7000, 4: 5.7100, 7: 5.7200, 10: 5.7150, 13: 5.7050}    # measured on TCs
    for h, v in meas.items():
        dw.table_widget.setItem(h - 1, 3, QTableWidgetItem(repr(v)))
        dw.table_widget.setItem(h - 1, 4, QTableWidgetItem(repr(v * 1e-3)))
        dw.table_widget.cellWidget(h - 1, 2).setChecked(True)
    fr.CalibrationDialog.exec_ = lambda dlg: (dlg._mon_cb.setCurrentText('TCs'), QDialog.Accepted)[1]
    w._open_calibrate()
    assert float(dw.table_widget.item(0, 3).text()) == 5.7
    tcs = am.route_cfg(S.default_route(am.STANDARDS, 'TCs'))
    close(w.monitor_ingrowth(), am.monitor_ingrowth(tcs), 1e-15,
          'values untouched; J here = (e^(λ t_TCs) − 1) / R')
    w.regression_info = {'type': '2d', 'complex_3d': False}
    path = os.path.join(tempfile.mkdtemp(), 'c.jflux')
    fr.dfold.get_save_file_name = lambda *a, **k: (path, '')
    fr.MONTE_CARLO_ITERATIONS = 2000
    np.random.seed(5)
    w.export_flux_config()
    cfgj = json.load(open(path, encoding='utf-8'))
    assert cfgj['monitor'] == 'TCs' and cfgj['monitorR'] == 1.0
    assert abs(cfgj['disks'][0]['R'][0] / 5.7 - 1) < 0.01
    ok('.jflux: the fitted measured R and the monitor TCs')
    fx = next(iter(am.read_jflux(path).values()))
    i = fx['idx']['1']
    close(am.flux_hole_measured(fx, i), cfgj['disks'][0]['R'][0], 1e-15, 'the tools read that R')
    close(am.flux_hole_J(fx, i, tcs)[0], am.monitor_ingrowth(tcs) / fx['R'][i], 1e-15,
          'a tool on TCs: J = (e^(λ t_TCs) − 1) / R')
    close(am.flux_hole_J(fx, i, tcs)[0], cfgj['disks'][0]['J'][0], 1e-12,
          '  the same J Flux regression wrote')
    fcs = am.route_cfg(S.default_route(am.STANDARDS, 'FCs'))
    assert am.monitor_mismatch_note(fcs, fx['monitor']) and not am.monitor_mismatch_note(tcs, 'TCs')
    ok('a tool on FCs is warned')


def test_flux_regression_views():
    """The planar result shows its planes and a map per disk; the 3D results show
    the stack and a map per disk, the complex one with its saddle."""
    print('Flux regression views:')
    fr = load(os.path.join('3_flux_regression', 'flux_regression.py'), 'flux_regression_v')
    fr.MONTE_CARLO_ITERATIONS = 150
    fr.MonteCarloThread.start = fr.MonteCarloThread.run
    fr.IndividualDiskMonteCarloThread.start = fr.IndividualDiskMonteCarloThread.run
    w = fr.FluxRegressionCalculator()
    w.add_disk(); w.add_disk()
    J0 = 2.7e-3
    for dw, lab, h in zip(w.disk_widgets, 'CDE', (0.0, 4.0, 8.0)):
        dw.disk_combo.setCurrentText('2.5mm (13 holes)')
        dw.disk_label_edit.setText(lab)
        dw.height_spinbox.setValue(h)
        coords = dw.get_disk_data()['coordinates']
        for hole in (1, 3, 5, 7, 9, 11):
            x, y = coords[hole - 1]
            t = dw.table_widget
            t.setItem(hole - 1, 0, QTableWidgetItem(f'{lab}{hole}'))
            t.setItem(hole - 1, 3, QTableWidgetItem(repr(J0 * (1 + 0.01 * x - 0.004 * y
                                                                  + 0.0006 * h + 0.008 * x * y))))
            t.setItem(hole - 1, 4, QTableWidgetItem(repr(J0 * 1e-3)))
            t.cellWidget(hole - 1, 2).setChecked(True)
    pw = w.plot_widget

    def pages():
        return [b.text() for b in pw._page_btns if b.isVisibleTo(pw)]

    w.calculate_individual_disk_regressions()
    assert pages() == ['Planes', 'By disk'], pages()
    pw.show_page(1)
    assert len([a for a in pw.figure.axes if a.get_title()]) == 3     # one map per disk
    ok('planar: its planes, and one map per disk')

    for complex_3d in (False, True):
        w.calculate_regression('3d', complex_3d=complex_3d)
        assert pages() == ['Stack', 'By disk'], pages()
        assert pw.figure.axes[0].name == '3d'
        pw.show_page(1)
        assert len([a for a in pw.figure.axes if a.get_title()]) == 3
        pw.figure.canvas.draw()
    ok('simple and complex 3D: the stack, and one map per disk')
    # selected cells are cleared with Delete; hole numbers and coordinates stay
    from PyQt5.QtGui import QKeyEvent
    from PyQt5.QtCore import QEvent, Qt as _Qt
    dw = w.disk_widgets[0]; t = dw.table_widget
    t.clearSelection()
    for r in (0, 2):
        for c in (1, 3, 4, 5):
            t.item(r, c).setSelected(True)
    dw.handle_key_press(QKeyEvent(QEvent.KeyPress, _Qt.Key_Delete, _Qt.NoModifier))
    assert t.item(0, 3).text() == '' and t.item(2, 4).text() == ''
    assert t.item(0, 1).text() == '1' and t.item(0, 5).text().startswith('(')
    ok('Delete clears the selected values, not the hole numbers or coordinates')
    w.close()


if __name__ == '__main__':
    test_reml_vs_metafor()
    test_weighted_mean_and_j()
    test_total_gas()
    test_window_and_export()
    test_flux_regression_import()
    test_flux_cross_disk()
    test_flux_regression_session_and_minimum()
    test_jflux_r_mode_mean()
    test_flux_monitors_calibration()
    test_flux_regression_route_chain()
    test_flux_regression_views()
    print('ALL PASSED')
