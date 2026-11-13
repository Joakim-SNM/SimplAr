"""Flux regression reads an .xlsx export of the Flux monitors tool.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_flux_regression_import.py
"""
import importlib.util, os, sys, tempfile
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('SIMPLAR_DATA_DIR', tempfile.mkdtemp())
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from openpyxl import Workbook
from PyQt5.QtWidgets import QApplication, QMessageBox

app = QApplication.instance() or QApplication([])
shown = []
for _n in ('information', 'warning', 'critical'):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: shown.append((_n, a[1:3])) or 0))

spec = importlib.util.spec_from_file_location(
    'flux_regression', os.path.join(ROOT, '3_flux_regression', 'flux_regression.py'))
fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
TMP = tempfile.mkdtemp()


def ok(what):
    print(f'  ok  {what}')


def close(a, b, tol, what):
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
close(cell(w, 0, 3), 5.67027, 1e-15, 'hole 1 holds the reported R')
close(cell(w, 0, 4), 0.0014673, 1e-15, 'and its 1σ SE')
close(cell(w, 12, 3), 5.70837, 1e-15, 'C13 goes to hole 13')
assert dw.table_widget.cellWidget(0, 2).isChecked() and dw.table_widget.item(0, 0).text() == 'C1'
ok('ticked for the regression and named after the category')
assert shown and 'Blank' in shown[-1][1][1] and 'C2' in shown[-1][1][1]
ok('a category without a hole number or an SE is reported, not imported')
std = dw.get_disk_data()['standard_data']
assert sorted(h['hole_number'] for h in std) == [1, 13]
close(std[0]['r_error'], 0.0014673, 1e-15, 'the regression reads it back at 1σ')

w.input_sigma_2_btn.click()
w.import_flux_monitors_export(PATH)
close(cell(w, 0, 4), 2 * 0.0014673, 1e-15, 'with 2σ input the SE is entered doubled')

print('J mode:')
wj = fr.FluxRegressionCalculator()
wj.disk_widgets[0].disk_combo.setCurrentText('2.5mm (13 holes)')
shown.clear()
wj.import_flux_monitors_export(PATH)
c = wj.monitor_ingrowth()
close(cell(wj, 0, 3), c / 5.67027, 1e-12, 'J = (e^(λt) − 1)/R with this tool\'s calibration')
close(cell(wj, 0, 4) / cell(wj, 0, 3), 0.0014673 / 5.67027, 1e-12, 'σJ/J = σR/R')
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
