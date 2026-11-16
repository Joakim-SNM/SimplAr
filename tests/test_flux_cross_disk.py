"""Flux regression: position shift (x, y only), and J correlation between disks.

A global 3D fit through several disks correlates holes on different disks; the
.jflux export stores that correlation and the Plateau & isochron / Age
distribution tools read it (arar_math.j_correlation).

Run:  QT_QPA_PLATFORM=offscreen python tests/test_flux_cross_disk.py
"""
import importlib.util, json, os, sys, tempfile
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('SIMPLAR_DATA_DIR', tempfile.mkdtemp())
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox, QTableWidgetItem

app = QApplication.instance() or QApplication([])
for _n in ('information', 'warning', 'critical'):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: 0))

spec = importlib.util.spec_from_file_location(
    'flux_regression', os.path.join(ROOT, '3_flux_regression', 'flux_regression.py'))
fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
import arar_math as am

TMP = tempfile.mkdtemp()


def ok(what):
    print(f'  ok  {what}')


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
ok('x, y: the same draws as before the change')
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
ok(f'and it is far from zero, which the old export assumed (max |ρ| = {big:.2f})')

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
