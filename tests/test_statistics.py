"""Statistics: GLS + ML combination (arar_math.combine_ages_vermeesch) against
independent references, and the mixture models with the Mixture view of the
Age distribution tool.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_statistics.py
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

import csv
import importlib.util
import json
import math

import numpy as np
from scipy import optimize, stats
from scipy.special import logsumexp

import arar_math as am
from synthetic_data import load_ratios, CFG_R


# --------------------------------------------------------------------------
# GLS + ML combination (arar_math.combine_ages_vermeesch) against
# independent references.  No data files needed.
# --------------------------------------------------------------------------

def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'BAD'} {label:44s} {got:.6g}  vs  {want:.6g}")
    assert ok, label


def test_bcg_reml():
    print('REML overdispersion vs metafor (BCG trials, log risk ratio):')
    bcg = [(4, 119, 11, 128), (6, 300, 29, 274), (3, 228, 11, 209),
           (62, 13536, 248, 12619), (33, 5036, 47, 5761), (180, 1361, 372, 1079),
           (8, 2537, 10, 619), (505, 87886, 499, 87892), (29, 7470, 45, 7232),
           (17, 1699, 65, 1600), (186, 50448, 141, 27197), (5, 2493, 3, 2338),
           (27, 16886, 29, 17825)]
    yi = np.array([math.log((a / (a + b)) / (c / (c + d))) for a, b, c, d in bcg])
    vi = np.array([1 / a - 1 / (a + b) + 1 / c - 1 / (c + d) for a, b, c, d in bcg])
    o2 = am.ml_overdispersion(yi, np.sqrt(vi), reml=True)
    w = 1 / (vi + o2)
    check('tau^2', o2, 0.3132, 5e-5)
    check('mu', (w * yi).sum() / w.sum(), -0.7145, 5e-5)
    check('se', math.sqrt(1 / w.sum()), 0.1798, 5e-5)


def _independent_ml(t, V0):
    """(mu, omega^2) maximising the normal likelihood with V = V0 + omega^2 I."""
    n = len(t)

    def nll(p):
        mu, log_om = p
        V = V0 + math.exp(2 * log_om) * np.eye(n)
        r = t - mu
        return 0.5 * (np.linalg.slogdet(V)[1] + r @ np.linalg.inv(V) @ r)

    best = None
    for lo0 in np.linspace(math.log(1e-6 * t.std() + 1e-12), math.log(3 * t.std() + 1e-12), 9):
        r = optimize.minimize(nll, [t.mean(), lo0], method='Nelder-Mead',
                              options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
        if best is None or r.fun < best.fun:
            best = r

    def nll0(mu):
        r = t - mu
        return 0.5 * (np.linalg.slogdet(V0)[1] + r @ np.linalg.inv(V0) @ r)

    b0 = optimize.minimize_scalar(nll0)
    if b0.fun <= best.fun:
        return b0.x, 0.0
    return best.x[0], math.exp(2 * best.x[1])


def test_against_independent_ml(trials=60):
    print(f'GLS + ML vs an independent implementation ({trials} random data sets):')
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(4, 25))
        a = rng.uniform(0.005, 0.08, n)                    # analytical 1σ, Ma
        pos = rng.integers(0, 3, n)                        # three irradiation positions
        p = a * rng.uniform(0.2, 1.5, n)                   # J part of the internal tier
        s = np.full(n, 0.03)                               # calibration part
        t = 55.0 + rng.normal(0, rng.choice([0, 0.01, 0.03, 0.06]), n) + rng.normal(0, a)
        si = np.sqrt(a ** 2 + p ** 2)
        se = np.sqrt(si ** 2 + s ** 2)
        res = am.combine_ages_vermeesch(t, a, si, se, position_group=list(pos),
                                        return_matrices=True)
        M = res['matrices']
        V0 = np.diag(a ** 2) + M['P'] + M['S']
        _, om2 = _independent_ml(t, V0)
        V = V0 + om2 * np.eye(n)
        Vi = np.linalg.inv(V)
        w = Vi.sum(1) / Vi.sum()
        worst = max(worst,
                    abs(res['wm'] - w @ t),
                    abs(math.sqrt(M['omega2']) - math.sqrt(om2)),
                    abs(res['sigma_ext'] - math.sqrt(w @ V @ w)))
    check('largest difference (Ma)', worst, 0.0, 1e-6)


def test_reduces_to_weighted_mean():
    print('GLS without random effects or covariance = inverse-variance WM:')
    rng = np.random.default_rng(2)
    a = rng.uniform(0.01, 0.05, 12)
    t = 55.0 + rng.normal(0, a)
    g = am.combine_ages_vermeesch(t, a, random_effects=False)
    w = am.combine_ages(t, a, scale_overdispersion=False)
    check('mean (Ma)', g['wm'], w['wm'], 1e-12)
    check('sigma (Ma)', g['sigma_ana'], w['sigma_ana'], 1e-12)


# --------------------------------------------------------------------------
# Mixture models (arar_math) and the Mixture view of the Age distribution
# tool.
# --------------------------------------------------------------------------

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
        f = am.fit_fmm(t, s, 3, space)
        z, sz, ok = am.mixture_to_z(t, s, space)

        def nll(p):
            pi = np.r_[p[:2], 1 - p[:2].sum()]
            return -logsumexp(np.log(pi)[None] + stats.norm.logpdf(
                z[:, None], p[2:][None], sz[:, None]), axis=1).sum()
        H = am._num_hessian(nll, np.r_[f['prop'][:2], f['mu']], h=1e-5)
        E_num = np.linalg.inv(H)
        E = am.information_cov(z, sz, f['mu'], f['prop'], f['post'][ok])
        for i in range(3):
            close(math.sqrt(E[2 + i, 2 + i]), math.sqrt(E_num[2 + i, 2 + i]), 1e-4,
                  f'{space}: σ(μ{i + 1})')
        close(math.sqrt(E[0, 0]), math.sqrt(E_num[0, 0]), 1e-4, f'{space}: σ(π1)')


def test_em_is_the_maximum():
    print('EM reaches the maximum likelihood (direct optimiser from another start)')
    t, s = three_components(seed=4)
    f = am.fit_fmm(t, s, 3, 'linear')

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
    f = am.fit_fmm(t, sig, 1, 'linear')
    wm = am.weighted_mean(t, sig)
    close(f['age'][0], wm['mean'], 1e-12, 'age')
    close(f['se_age'][0], wm['sem'], 1e-9, 'σ')
    close(f['mswd'][0], wm['mswd'], 1e-9, 'MSWD')


def test_bic_and_scales():
    print('BIC finds three clear components; linear and log agree')
    t, s = three_components(seed=2)
    kb, fits = am.select_k(t, s, 10, 'linear')
    assert kb == 3, kb
    kl, fl = am.select_k(t, s, 10, 'log')
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
        f = am.fit_fmm(t, s, 1, 'linear')
        tr = am.component_tiers(f, CFG_R)[0]
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
        f = am.fit_mam(true + rng.normal(0, s), s, 3, 'linear')
        res.append((f['age'] - 56.8) / f['se_age'])
    res = np.array(res)
    print(f'  ok  bias {np.mean(res):+.2f} σ, |z| < 2 in {np.mean(np.abs(res) < 2):.0%} of 40 runs')
    assert abs(np.mean(res)) < 0.5 and np.mean(np.abs(res) < 2) > 0.85
    s = rng.uniform(0.05, 0.2, 20)
    t = 55.0 + rng.normal(0, s)
    f = am.fit_mam(t, s, 3, 'linear')
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
    import export
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
    spec = export.default_spec(w._mix_export_panels(), w._mix_export_options())
    for p in spec['panels']:
        p['on'] = True
    fig = Figure(figsize=export.figsize(spec), facecolor='white')
    w._mix_export_render(fig, spec)
    assert len(fig.axes) == 3
    # default layout as in the window: diagram left, radial plot over ΔBIC right
    comp, rad, bic = (a.get_position() for a in fig.axes)
    assert comp.x1 < rad.x0 and abs(rad.x0 - bic.x0) < 1e-9 and rad.y0 > bic.y1
    assert comp.y0 <= bic.y0 + 1e-9 and comp.y1 >= rad.y1 - 1e-9
    spec['options']['layout'] = 'stacked'
    fig = Figure(figsize=export.figsize(spec), facecolor='white')
    w._mix_export_render(fig, spec)
    ys = [a.get_position().y0 for a in fig.axes]
    assert ys == sorted(ys, reverse=True), 'stacked: top to bottom'
    # the number of components in the legend heading, whatever the heading
    k = st['fit']['k']
    for head in ('model', 'name_n', 'name', 'off'):
        for on in (True, False):
            spec['options'].update(legend_title=head, legend_k=on)
            fig = Figure(figsize=export.figsize(spec), facecolor='white')
            w._mix_export_render(fig, spec)
            leg = fig.axes[0].get_legend()
            ttl = leg.get_title().get_text() if leg and leg.get_title().get_visible() else ''
            assert (f'{k} components' in ttl.lower()) == on, (head, on, ttl)
    spec['options'].update(legend_title='name_n', legend_k=False)
    fig = Figure(figsize=export.figsize(spec), facecolor='white')
    w._mix_export_render(fig, spec)
    assert fig.axes[0].get_legend().get_title().get_text() == f"Sample A  ({st['fit']['n']} analyses)"
    # every sample stacked, each with its own model and legend
    spec2 = export.default_spec(w._mix_export_panels(), w._mix_export_options())
    spec2['options'].update(stack=True, legend_title='name_n', mixk_1='1')
    spec2['letters'] = 'tr'
    fig = Figure(figsize=export.figsize(spec2), facecolor='white')
    w._mix_export_render(fig, spec2)
    export.FigureCanvasAgg(fig)
    export.finish_figure(fig, spec2)
    assert len(fig.axes) == 2, len(fig.axes)
    heads = [a.get_legend().get_title().get_text() for a in fig.axes]
    assert heads[0].startswith('Sample A') and heads[1] == 'Other sample: 1 component  (10 analyses)', heads
    assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
    assert [t.get_text() for a in fig.axes for t in a.texts] == ['A', 'B']
    # only C1 listed: one line for it, one for the rest
    spec['options'].update(legend_title='model', legend_k=True, legend_max=1)
    fig = Figure(figsize=export.figsize(spec), facecolor='white')
    w._mix_export_render(fig, spec)
    labs = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    k = st['fit']['k']
    assert len(labs) == 2 and labs[0].startswith('C1'), labs
    assert labs[1].startswith(f'+{k - 1} '), labs   # 'other' or 'outside'
    spec['options']['legend_max'] = 0
    QDialog.exec_ = lambda self: 0
    w.export_figure()
    print('  ok  figure export: window layout and stacked layout; components in legend; dialog opens')
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
    test_bcg_reml()
    test_against_independent_ml()
    test_reduces_to_weighted_mean()
    test_information_matrix()
    test_em_is_the_maximum()
    test_one_component_is_the_weighted_mean()
    test_bic_and_scales()
    test_tiers_and_scaling()
    test_minimum_age_model()
    test_window()
    print('ALL PASSED')
