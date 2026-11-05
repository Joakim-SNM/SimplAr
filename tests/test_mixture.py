"""Mixture models (mixture_model.py) and the Mixture view of the Age distribution tool.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_mixture.py
"""
import csv, importlib.util, json, math, os, sys, tempfile
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('SIMPLAR_DATA_DIR', tempfile.mkdtemp())
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from scipy import optimize, stats
from scipy.special import logsumexp
import mixture_model as mm
import arar_math as am
from test_anchored_framework import load_ratios, CFG_R


def close(a, b, rel, what):
    ok = abs(a - b) <= rel * max(abs(b), 1e-300)
    print(f'  {"ok" if ok else "FAIL"}  {what:<56} {a:.10g}  vs  {b:.10g}')
    assert ok, what


def three_components(seed=1, n=60):
    rng = np.random.default_rng(seed)
    comp = rng.choice(3, n, p=[0.6, 0.3, 0.1])
    s = rng.uniform(0.04, 0.3, n)
    return np.array([55.30, 55.80, 59.3])[comp] + rng.normal(0, s), s


def test_information_matrix():
    print('Component uncertainties: information matrix = numerical Hessian')
    t, s = three_components()
    for space in ('linear', 'log'):
        f = mm.fit_fmm(t, s, 3, space)
        z, sz, ok = mm.to_z(t, s, space)

        def nll(p):
            pi = np.r_[p[:2], 1 - p[:2].sum()]
            return -logsumexp(np.log(pi)[None] + stats.norm.logpdf(
                z[:, None], p[2:][None], sz[:, None]), axis=1).sum()
        H = mm._num_hessian(nll, np.r_[f['prop'][:2], f['mu']], h=1e-5)
        E_num = np.linalg.inv(H)
        E = mm.information_cov(z, sz, f['mu'], f['prop'], f['post'][ok])
        for i in range(3):
            close(math.sqrt(E[2 + i, 2 + i]), math.sqrt(E_num[2 + i, 2 + i]), 1e-4,
                  f'{space}: σ(μ{i + 1})')
        close(math.sqrt(E[0, 0]), math.sqrt(E_num[0, 0]), 1e-4, f'{space}: σ(π1)')


def test_em_is_the_maximum():
    print('EM reaches the maximum likelihood (direct optimiser from another start)')
    t, s = three_components(seed=4)
    f = mm.fit_fmm(t, s, 3, 'linear')

    def nll(p):
        pi = np.r_[p[:2], 1 - p[:2].sum()]
        if np.any(pi <= 0):
            return 1e10
        return -logsumexp(np.log(pi)[None] + stats.norm.logpdf(
            t[:, None], p[2:][None], s[:, None]), axis=1).sum()
    r = optimize.minimize(nll, np.r_[f['prop'][:2] + [0.03, -0.02], f['mu'] + [0.01, -0.01, 0.05]],
                          method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-12, maxiter=40000))
    close(f['L'], -r.fun, 1e-8, 'log-likelihood')


def test_one_component_is_the_weighted_mean():
    print('One component = the inverse-variance weighted mean')
    R, sR, _F, _sF = load_ratios()
    (t, sig), err = am.r_to_age_cfg(R, sR, CFG_R)
    f = mm.fit_fmm(t, sig, 1, 'linear')
    wm = am.weighted_mean(t, sig)
    close(f['age'][0], wm['mean'], 1e-12, 'age')
    close(f['se_age'][0], wm['sem'], 1e-9, 'σ')
    close(f['mswd'][0], wm['mswd'], 1e-9, 'MSWD')


def test_bic_and_scales():
    print('BIC finds three clear components; linear and log agree')
    t, s = three_components(seed=2)
    kb, fits = mm.select_k(t, s, 10, 'linear')
    assert kb == 3, kb
    kl, fl = mm.select_k(t, s, 10, 'log')
    assert kl == 3, kl
    for i in range(3):
        close(fl[2]['age'][i], fits[2]['age'][i], 2e-5, f'C{i + 1} age, log vs linear')
    print('  ok  k = 3 on both scales')


def test_tiers_and_scaling():
    print('Component tiers: √MSWD only above the Wendt & Carl limit; ana <= int <= ext')
    rng = np.random.default_rng(7)
    s = rng.uniform(0.05, 0.2, 40)
    t_ok = 55.3 + rng.normal(0, s)
    t_od = 55.3 + rng.normal(0, s) + rng.normal(0, 0.25, 40)
    for t, label in ((t_ok, 'no excess scatter'), (t_od, 'excess scatter')):
        f = mm.fit_fmm(t, s, 1, 'linear')
        tr = mm.component_tiers(f, CFG_R)[0]
        over = tr['mswd'] > am.mswd_critical(tr['n_eff'])
        want = math.sqrt(tr['mswd']) if over else 1.0
        close(tr['scale'], want, 1e-12, f'{label}: scale (MSWD {tr["mswd"]:.2f})')
        close(tr['ana'], f['se_age'][0] * want, 1e-12, f'{label}: analytical σ')
        assert tr['ana'] <= tr['int'] <= tr['ext']


def test_minimum_age_model():
    print('Minimum age model')
    rng = np.random.default_rng(5)
    res = []
    for _ in range(40):
        young = rng.random(50) < 0.4
        true = np.where(young, 56.8, 56.8 + np.abs(rng.normal(0, 1.5, 50)))
        s = rng.uniform(0.05, 0.5, 50)
        f = mm.fit_mam(true + rng.normal(0, s), s, 3, 'linear')
        res.append((f['age'] - 56.8) / f['se_age'])
    res = np.array(res)
    print(f'  ok  bias {np.mean(res):+.2f} σ, |z| < 2 in {np.mean(np.abs(res) < 2):.0%} of 40 runs')
    assert abs(np.mean(res)) < 0.5 and np.mean(np.abs(res) < 2) > 0.85
    s = rng.uniform(0.05, 0.2, 20)
    t = 55.0 + rng.normal(0, s)
    f = mm.fit_mam(t, s, 3, 'linear')
    wm = am.weighted_mean(t, s)
    if f['note']:
        close(f['age'], wm['mean'], 1e-9, 'no older analyses: minimum = weighted mean')
    else:
        assert abs(f['age'] - wm['mean']) < 3 * wm['sem']
        print('  ok  no older analyses: minimum within 3σ of the weighted mean')


def test_window():
    print('Age distribution: Mixture view, exports and session')
    from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QMessageBox
    from matplotlib.figure import Figure
    import figure_export as figexp
    app = QApplication.instance() or QApplication([])  # noqa: F841 (kept alive)
    spec_ = importlib.util.spec_from_file_location(
        'ppm', os.path.join(ROOT, '5_age_distribution', 'age_distribution.py'))
    pp = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(pp)
    for n in ('information', 'warning', 'critical'):
        setattr(QMessageBox, n, staticmethod(lambda *a, **k: 0))
    R, sR, _F, _sF = load_ratios()
    w = pp.ProbabilityWindow()
    for sm in list(w.samples):
        w.remove_sample(sm)
    s0 = w.add_sample(); s0.name = 'Sample A'; s0.input_mode = 'ratio'
    s0.reject = 'none'; s0.j_cfg.update(CFG_R)
    # an old xenocryst added to the sample
    s0.r_vals = np.r_[R, R[0] * 1.07]; s0.r_uncs = np.r_[sR, sR[0]]; s0.recompute()
    # a second sample, which the mixture export must leave out
    s1 = w.add_sample(); s1.name = 'Other sample'; s1.input_mode = 'ratio'
    s1.reject = 'none'; s1.j_cfg.update(CFG_R)
    s1.r_vals = np.array(R[:10]); s1.r_uncs = np.array(sR[:10]); s1.recompute()
    w.redraw()
    w.mix_btn.setChecked(True)
    assert w.mix_on and w._plot_stack.currentIndex() == 1
    st = w._mix_state
    assert st is not None and st['fit']['k'] >= 2, 'the xenocryst should be its own component'
    assert 'MIXTURE MODEL' in w.report.toPlainText()
    print(f"  ok  view on: k = {st['fit']['k']}, youngest {st['tiers'][0]['age']:.3f} Ma")
    # figure export: all three panels
    spec = figexp.default_spec(w._mix_export_panels(), w._mix_export_options())
    for p in spec['panels']:
        p['on'] = True
    fig = Figure(figsize=figexp.figsize(spec), facecolor='white')
    w._mix_export_render(fig, spec)
    assert len(fig.axes) == 3
    # default layout as in the window: diagram left, radial plot over ΔBIC right
    comp, rad, bic = (a.get_position() for a in fig.axes)
    assert comp.x1 < rad.x0 and abs(rad.x0 - bic.x0) < 1e-9 and rad.y0 > bic.y1
    assert comp.y0 <= bic.y0 + 1e-9 and comp.y1 >= rad.y1 - 1e-9
    spec['options']['layout'] = 'stacked'
    fig = Figure(figsize=figexp.figsize(spec), facecolor='white')
    w._mix_export_render(fig, spec)
    ys = [a.get_position().y0 for a in fig.axes]
    assert ys == sorted(ys, reverse=True), 'stacked: top to bottom'
    QDialog.exec_ = lambda self: 0
    w.export_figure()
    print('  ok  figure export: window layout and stacked layout; dialog opens')
    with tempfile.TemporaryDirectory() as d:
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (os.path.join(d, 'x.csv'), ''))
        w.export_csv()
        rows = list(csv.reader(open(os.path.join(d, 'x.csv'), encoding='utf-8-sig')))
        assert any(r and r[0] == 'MIXTURE MODEL' for r in rows)
        assert not any('Other sample' in c for r in rows for c in r), 'one sample only'
        assert not any(r and r[0] in ('SAMPLE RESULTS', 'POOLED RESULT') for r in rows)
        j = next(k for k, r in enumerate(rows) if r and r[0] == 'POINT DATA')
        assert rows[j + 1][len(w._POINT_HEADER)] == 'Most likely component'
        pts = []
        for r in rows[j + 2:]:
            if not r or not any(r):
                break
            pts.append(r)
        assert len(pts) == len(s0.ages) and all(r[0] == 'Sample A' for r in pts)
        P = np.array([[float(v) for v in r[len(w._POINT_HEADER) + 1:]] for r in pts])
        assert np.allclose(P.sum(1), 1.0, atol=5e-4), 'memberships add up to one'
        print(f'  ok  mixture data export: Sample A only, components and {len(pts)} analyses')
        w.mix_btn.setChecked(False)
        w.export_csv()
        rows = list(csv.reader(open(os.path.join(d, 'x.csv'), encoding='utf-8-sig')))
        assert not any(r and r[0].startswith('MIXTURE') for r in rows)
        assert any('Other sample' in c for r in rows for c in r)
        print('  ok  with the view off: the normal export, every sample, no mixture')
        w.mix_btn.setChecked(True)
        w.mix_k.setCurrentText('2'); w.mix_space.setCurrentText('log'); w._redraw_mixture()
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (os.path.join(d, 's.arprob'), ''))
        w.save_session()
        assert json.load(open(os.path.join(d, 's.arprob'), encoding='utf-8'))['mixture']['components'] == '2'
        w2 = pp.ProbabilityWindow()
        QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (os.path.join(d, 's.arprob'), ''))
        w2.load_session()
        assert w2.mix_on and w2.mix_k.currentText() == '2' and w2.mix_space.currentText() == 'log'
        assert w2._mix_state['fit']['k'] == 2
        print('  ok  session keeps the Mixture settings')
    w.mix_btn.setChecked(False)
    assert not w.mix_on and w._plot_stack.currentIndex() == 0
    print('  ok  view off: back to the plot')


if __name__ == '__main__':
    test_information_matrix()
    test_em_is_the_maximum()
    test_one_component_is_the_weighted_mean()
    test_bic_and_scales()
    test_tiers_and_scaling()
    test_minimum_age_model()
    test_window()
    print('ALL PASSED')
