"""The export dialog and both hosts render without touching the screen view.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_figure_export.py
"""
import copy, importlib.util, os, sys, tempfile
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('SIMPLAR_DATA_DIR', tempfile.mkdtemp())
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from PyQt5.QtWidgets import QApplication
from matplotlib.figure import Figure
import figure_export as figexp
from test_anchored_framework import load_ratios, CFG_R
from synthetic_data import plateau_steps

app = QApplication.instance() or QApplication([])


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def load_demo(w, title='synthetic'):
    """Put the synthetic step-heating experiment into a Plateau window."""
    w.raw_steps = plateau_steps()
    w.title_edit.setText(title)
    w._update_all()


def render(win, spec, what):
    fig = Figure(figsize=figexp.figsize(spec), facecolor='white')
    win._export_render(fig, spec)
    n = len([a for a in fig.axes])
    print(f'  ok  {what:<52} {n} axes')
    return fig


def test_plateau():
    print('Plateau: panel switches, K/Ca and content toggles')
    pl = load('4_plateau_isochron/plateau_isochron.py', 'pl')
    w = pl.PlateauWindow()
    load_demo(w)
    base = figexp.default_spec(w._export_panels(), w._export_options())
    render(w, base, 'three panels (default)')
    s = copy.deepcopy(base)
    for p in s['panels']:
        p['on'] = p['id'] in ('cak', 'age')
    s['options']['cak_mode'] = 'kca'
    f = render(w, s, 'K/Ca + age spectrum')
    labels = [a.get_ylabel() for a in f.axes]
    assert 'K/Ca' in labels, labels
    s = copy.deepcopy(base)
    for p in s['panels']:
        p['on'] = p['id'] == 'age'
    s['options']['iso_inset'] = True
    render(w, s, 'age spectrum with isochron inset')
    s = copy.deepcopy(base)
    s['options'].update(show_title=False, show_step_labels=False,
                        show_int_age=False, show_err_box=False,
                        show_bracket=False)
    f = render(w, s, 'everything optional switched off')
    assert not f.axes[0].get_title()
    ax_age = [a for a in f.axes if a.get_ylabel().startswith('Apparent')][0]
    assert not [t for t in ax_age.texts if t.get_text().strip()], \
        'content switches left text on the age panel'
    # the screen canvas must be untouched by all of that
    w._do_redraw()
    assert len(w.canvas.fig.axes) == 3, len(w.canvas.fig.axes)
    print('  ok  on-screen canvas still has its three panels')


def test_probability():
    print('Age distribution: element switches and legend placement')
    pp = load('5_age_distribution/age_distribution.py', 'pp')
    R, sR, _F, _sF = load_ratios()
    w = pp.ProbabilityWindow()
    for sm in list(w.samples):
        w.remove_sample(sm)
    for nm, sl in (('A', slice(0, 20)), ('B', slice(20, None))):
        s = w.add_sample(); s.name = nm; s.input_mode = 'ratio'; s.reject = 'none'
        s.j_cfg.update(CFG_R)
        s.r_vals = np.array(R[sl]); s.r_uncs = np.array(sR[sl]); s.recompute()
    base = figexp.default_spec(w._export_panels(), w._export_options())
    f = render(w, base, 'default')
    assert f.axes[0].get_legend() is not None
    s = copy.deepcopy(base); s['legend'] = 'off'
    f = render(w, s, 'legend off')
    assert f.axes[0].get_legend() is None
    s = copy.deepcopy(base); s['legend'] = 'outside'
    f = render(w, s, 'legend outside')
    assert f.axes[0].get_legend() is not None
    s = copy.deepcopy(base)
    s['options'].update(curve=False, fill=False, dashed=False)
    render(w, s, 'points only')
    s = copy.deepcopy(base); s['options']['points'] = False
    render(w, s, 'curves only')
    w.redraw()
    print('  ok  on-screen plot redrawn after export')


def test_dialog_and_presets():
    print('Dialog: preview, panel edits and preset round-trip')
    pl = load('4_plateau_isochron/plateau_isochron.py', 'pl')
    w = pl.PlateauWindow(); load_demo(w)
    dlg = figexp.FigureExportDialog(w, '_selftest', w._export_panels(),
                                    w._export_options(), w._export_render)
    dlg.show(); app.processEvents()
    dlg._refresh_preview()
    assert dlg.preview.pixmap() is not None and dlg.preview.pixmap().width() > 50
    dlg._panel_rows['p40'][0].setChecked(False)
    dlg._panel_rows['age'][1].setValue(3.5)
    dlg.sp_font.setValue(1.4)
    app.processEvents()
    on = {p['id']: p for p in dlg.spec['panels']}
    assert on['p40']['on'] is False and on['age']['weight'] == 3.5
    assert dlg.spec['font_scale'] == 1.4
    dlg._move(0, 1)
    assert [p['id'] for p in dlg.spec['panels']][:2] == ['cak', 'p40']
    store = figexp._load_store()
    store.setdefault('_selftest', {})['presets'] = {'T': copy.deepcopy(dlg.spec)}
    figexp._save_store(store)
    back = figexp._merge(figexp._load_store()['_selftest']['presets']['T'],
                         w._export_panels(), w._export_options())
    assert back['font_scale'] == 1.4
    assert [p['id'] for p in back['panels']][:2] == ['cak', 'p40']
    store = figexp._load_store(); store.pop('_selftest', None); figexp._save_store(store)
    print('  ok  preview, edits, ordering and presets')


def _synthetic_window(pl):
    """A plateau window holding one synthetic aliquot with isochron data."""
    import math
    lam, J = 5.463e-10, 0.00271
    Rrad = (math.exp(lam * 55.3e6) - 1) / J
    steps = []
    for k in range(8):
        f = 0.2 + 0.1 * k
        ar39, ar40r = 1.0, Rrad
        ar36 = f * 0.01
        ar40 = ar40r + 295.5 * ar36
        steps.append(dict(label=chr(65 + k), power=k + 1.0, age=55.3 + 0.02 * k,
                          err1s=0.05, pct39Ar=12.5, caK=30.0 + k, err_cak=1.0,
                          pct40=100 * ar40r / ar40, r4039=Rrad,
                          err_r4039=Rrad * 0.002, j_val=J, j_err=J * 2e-4,
                          status='normal',
                          iso_ix=ar39 / ar36, iso_isx=ar39 / ar36 * 0.004,
                          iso_iy=ar40 / ar36, iso_isy=ar40 / ar36 * 0.004,
                          iso_iri=0.5,
                          iso_nx=ar39 / ar40, iso_nsx=ar39 / ar40 * 0.004,
                          iso_ny=ar36 / ar40, iso_nsy=ar36 / ar40 * 0.004,
                          iso_nri=0.1))
    w = pl.PlateauWindow()
    w.cfg['ageFluxMode'] = 'imported'
    w.raw_steps = steps
    w.title_edit.setText('Sample 2')
    w._update_all()
    return w


def test_isochron_variants():
    print('Isochron: inverse and normal, inset, own page')
    pl = load('4_plateau_isochron/plateau_isochron.py', 'pl2')
    w = _synthetic_window(pl)
    base = figexp.default_spec(w._export_panels(), w._export_options())
    for mode in ('inverse', 'normal'):
        s = copy.deepcopy(base); s['options']['iso_type'] = mode
        fig = Figure(figsize=(6.0, 5.0), facecolor='white')
        w._export_render_isochron_only(fig, s)
        txt = [t.get_text() for t in fig.findobj(lambda o: hasattr(o, 'get_text'))]
        assert not any('No isochron data' in t for t in txt if isinstance(t, str)), mode
        # an artist escaping the axes used to make the tight bbox explode
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=60, bbox_inches='tight')
        from PIL import Image
        im = Image.open(buf)
        ratio = im.width / im.height
        assert 0.7 < ratio < 1.9, (mode, im.size)
        print(f'  ok  {mode:<8} isochron drawn, image {im.width}×{im.height}')
    s = copy.deepcopy(base); s['options']['iso_type'] = 'both'
    fig = Figure(figsize=(8.0, 4.0), facecolor='white')
    w._export_render_isochron_only(fig, s)
    assert len(fig.axes) == 2, 'both isochrons should share the page'
    assert {a.get_ylabel() for a in fig.axes} == {'\u00b3\u2076Ar/\u2074\u2070Ar',
                                                 '\u2074\u2070Ar/\u00b3\u2076Ar'}
    print('  ok  inverse and normal side by side on the isochron page')
    s = copy.deepcopy(base)
    for p in s['panels']:
        p['on'] = p['id'] == 'age'
    s['options'].update(iso_inset=True, iso_x=95, iso_y=95, iso_w=40, iso_h=42,
                        title='Renamed sample', y_min='55.15', y_max='55.60')
    # 'Main window' (default): the sample name in the main window wins over typed text
    fig = Figure(figsize=figexp.figsize(s), facecolor='white')
    w._export_render(fig, s)
    assert fig.axes[0].get_title() == w.title_edit.text(), fig.axes[0].get_title()
    s['options']['names_from'] = 'typed'
    fig = Figure(figsize=figexp.figsize(s), facecolor='white')
    w._export_render(fig, s)
    ax = fig.axes[0]
    assert ax.get_title() == 'Renamed sample', ax.get_title()
    assert abs(ax.get_ylim()[0] - 55.15) < 1e-9 and abs(ax.get_ylim()[1] - 55.60) < 1e-9
    assert len(ax.child_axes) == 1, 'inset missing'
    ins = ax.child_axes[0]
    stats = [t for t in ins.texts if 'MSWD' in t.get_text()]
    assert not stats, 'inset should not carry the statistics box'
    # asked for 95/95: it must still be pushed fully inside the panel frame
    x, y, bw, bh = ins.get_position().bounds
    host = ax.get_position().bounds
    assert x > host[0] and y > host[1]
    assert x + bw < host[0] + host[2] and y + bh < host[1] + host[3], \
        'the inset frame runs into the age-panel frame'
    assert len(ins.get_xticks()) <= 5 and len(ins.get_yticks()) <= 5
    print('  ok  inset inside the panel, few ticks, no stats box; title and range applied')
    # atmospheric reference (Lee et al. 2006) and the intercept switch
    s = copy.deepcopy(base)
    s['options'].update(iso_type='both', iso_atm=True)
    fig = Figure(figsize=(8.0, 4.0), facecolor='white')
    w._export_render_isochron_only(fig, s)
    inv = [a for a in fig.axes if a.get_ylabel().startswith('\u00b3\u2076')][0]
    nrm = [a for a in fig.axes if a.get_ylabel().startswith('\u2074\u2070')][0]
    for ax_, expect in ((inv, 1.0 / pl.ATM_40_36), (nrm, pl.ATM_40_36)):
        atm = [t for t in ax_.texts if 'atmospheric' in t.get_text()]
        assert len(atm) == 1, (ax_.get_ylabel(), len(atm))
        ys = [ln.get_ydata()[0] for ln in ax_.lines
              if ln.get_linestyle() != '-' and len(set(ln.get_ydata())) == 1]
        assert any(abs(y - expect) < 1e-9 for y in ys), (ax_.get_ylabel(), expect)
    assert 'iso_atm_value' not in {o.id for o in w._export_options()}
    print('  ok  atmospheric line at 298.56, once per diagram, no value field')
    # the trapped-argon label works with the statistics box off, and the box
    # switch must not change anything else about the diagram
    s = copy.deepcopy(base)
    s['options'].update(iso_stats=False, iso_init=True, iso_atm=True)
    fig = Figure(figsize=(6.5, 5.0), facecolor='white')
    w._export_render_isochron_only(fig, s)
    txt = [t.get_text() for t in fig.axes[0].texts]
    assert not any('MSWD' in t for t in txt), 'statistics box still drawn'
    assert any(t.startswith('\u2074\u2070Ar/\u00b3\u2076Ar\u2080 =') for t in txt), txt
    assert any('atmospheric' in t for t in txt), 'atmospheric caption missing'
    assert [t for t in txt if t in list('ABCDEFGH')], \
        'the stats-box switch should not remove the step labels'
    print('  ok  \u2074\u2070Ar/\u00b3\u2076Ar\u2080 label independent of the statistics box')
    s = copy.deepcopy(base); s['options']['iso_intercept'] = False
    fig = Figure(figsize=(6.0, 5.0), facecolor='white')
    w._export_render_isochron_only(fig, s)
    caps = [c for c in fig.axes[0].containers]
    assert not caps, 'intercept marker still drawn with the switch off'
    print('  ok  intercept marker can be switched off')
    # two-page PDF
    import tempfile
    from matplotlib.backends.backend_pdf import PdfPages
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'both.pdf')
        f1 = Figure(figsize=figexp.figsize(base), facecolor='white')
        w._export_render(f1, base)
        f2 = Figure(figsize=figexp.figsize(base), facecolor='white')
        w._export_render_isochron_only(f2, base)
        with PdfPages(path) as pdf:
            pdf.savefig(f1, bbox_inches='tight')
            pdf.savefig(f2, bbox_inches='tight')
        assert os.path.getsize(path) > 5000
    print('  ok  spectrum + isochron written as a two-page PDF')


def test_isochron_spread():
    print('Isochron spread: a lever, not a scatter statistic')
    import math
    pl = load('4_plateau_isochron/plateau_isochron.py', 'pl4')

    def window(fracs):
        lam, J = 5.463e-10, 0.00271
        Rrad = (math.exp(lam * 55.3e6) - 1) / J
        steps = []
        for k, f in enumerate(fracs):            # f = radiogenic fraction
            ar39, ar40r = 1.0, Rrad
            ar40 = ar40r / f
            ar36 = (ar40 - ar40r) / pl.ATM_40_36
            steps.append(dict(
                label=chr(65 + k), power=k + 1.0, age=55.3, err1s=0.05,
                pct39Ar=100.0 / len(fracs), caK=30.0, err_cak=1.0,
                pct40=100 * f, r4039=Rrad, err_r4039=Rrad * 0.002,
                j_val=J, j_err=J * 2e-4, status='normal',
                iso_ix=ar39 / ar36, iso_isx=ar39 / ar36 * 0.004,
                iso_iy=ar40 / ar36, iso_isy=ar40 / ar36 * 0.004, iso_iri=0.5,
                iso_nx=ar39 / ar40, iso_nsx=ar39 / ar40 * 0.004,
                iso_ny=ar36 / ar40, iso_nsy=ar36 / ar40 * 0.004, iso_nri=0.1))
        w = pl.PlateauWindow()
        w.cfg['ageFluxMode'] = 'imported'
        w.raw_steps = steps
        w._update_all()
        return w

    def measure(w, **opts):
        spec = figexp.default_spec(w._export_panels(), w._export_options())
        spec['options'].update(iso_spread=True, **opts)
        fig = Figure(figsize=(6.5, 5.0), facecolor='white')
        w._export_render_isochron_only(fig, spec)
        txt = [t.get_text() for t in fig.axes[0].texts if 'Spread' in t.get_text()]
        assert txt, 'no spread line drawn'
        return txt[0], fig

    import re

    def numbers(line):
        rng = re.search(r'(\d+)\u2013(\d+)%', line)
        lev = re.search(r'lever ([0-9.]+)', line)
        return (int(rng.group(1)), int(rng.group(2)),
                float(lev.group(1)) if lev else None)

    wide = window([0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.98, 0.99])
    narrow = window([0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.992, 0.995])
    lo_w, hi_w, lev_w = numbers(measure(wide)[0])
    lo_n, hi_n, lev_n = numbers(measure(narrow)[0])
    assert (lo_w, hi_w) == (5, 99), (lo_w, hi_w)
    assert (lo_n, hi_n) == (96, 100), (lo_n, hi_n)
    assert lev_w > 0.9 and lev_n < 0.1, (lev_w, lev_n)
    print(f'  ok  wide data: {lo_w}\u2013{hi_w}% \u2074\u2070Ar*, lever {lev_w:.2f}  ·  '
          f'clustered: {lo_n}\u2013{hi_n}%, lever {lev_n:.2f}')
    # with the statistics box off the line moves to the corner label
    line, _fig = measure(narrow, iso_stats=False)
    assert 'MSWD' not in line and 'Spread' in line
    print('  ok  spread still shown when the statistics box is off')


def test_page_switcher_and_range():
    print('Preview pages, and an age window applied before drawing')
    pl = load('4_plateau_isochron/plateau_isochron.py', 'pl3')
    w = pl.PlateauWindow()
    load_demo(w)
    dlg = figexp.FigureExportDialog(
        w, '_selftest2', w._export_panels(), w._export_options(),
        w._export_render, default_name='demo',
        extra_render=w._export_render_isochron_only,
        extra_flag='iso_file', extra_page_flag='iso_page',
        extra_suffix='isochron', extra_label='isochron')
    dlg.show(); app.processEvents()
    assert hasattr(dlg, 'page_sel') and dlg.page_sel.count() == 2
    dlg.page_sel.setCurrentIndex(1); app.processEvents()
    dlg._refresh_preview()
    assert dlg.preview.pixmap() is not None and dlg.preview.pixmap().width() > 50
    assert 'not exported yet' in dlg.note.text()          # neither flag ticked
    dlg._opt_widgets['iso_page'][0].setChecked(True); app.processEvents()
    dlg._refresh_preview()
    assert 'not exported yet' not in dlg.note.text()
    # page 2 carries its own page size
    dlg.sp_w.setValue(24.0); dlg.sp_h.setValue(12.0); app.processEvents()
    assert dlg.spec['p2_width_cm'] == 24.0 and dlg.spec['p2_height_cm'] == 12.0
    assert dlg.spec['width_cm'] != 24.0, 'page 1 size must not follow page 2'
    assert abs(figexp.figsize(dlg.spec, 1)[0] - 24.0 * figexp.CM) < 1e-9
    dlg.page_sel.setCurrentIndex(0); app.processEvents()
    assert abs(dlg.sp_w.value() - dlg.spec['width_cm']) < 1e-9
    dlg._refresh_preview()
    print('  ok  page 2 previews the isochron, has its own size, and says when '
          'it is not exported')
    assert 'iso_zoom' not in {o.id for o in w._export_options()}
    print('  ok  the isochron y range option is gone (intercept always shown)')
    # an age window must not push the plateau bracket out of the figure
    w2 = _synthetic_window(pl)
    assert w2.plateau, 'the synthetic sample should have a plateau'
    spec = figexp.default_spec(w2._export_panels(), w2._export_options())
    spec['options'].update(y_min='50', y_max='60')
    fig = Figure(figsize=figexp.figsize(spec), facecolor='white')
    w2._export_render(fig, spec)
    ax = [a for a in fig.axes if a.get_ylabel().startswith('Apparent')][0]
    lo, hi = ax.get_ylim()
    assert abs(lo - 50.0) < 1e-9 and abs(hi - 60.0) < 1e-9
    # the bracket and its label are placed in data coordinates
    data_txt = [t for t in ax.texts
                if t.get_transform() is ax.transData and 'Ma' in t.get_text()]
    assert data_txt, 'no plateau label drawn'
    assert all(lo < t.get_position()[1] < hi for t in data_txt), \
        'the plateau label fell outside the age window'
    print('  ok  the plateau label stays inside a hand-set age window')


def test_probability_names_and_ranges():
    print('Age distribution: renaming and axis ranges')
    pp = load('5_age_distribution/age_distribution.py', 'pp2')
    R, sR, _F, _sF = load_ratios()
    w = pp.ProbabilityWindow()
    for sm in list(w.samples):
        w.remove_sample(sm)
    s0 = w.add_sample(); s0.name = 'raw name'; s0.input_mode = 'ratio'
    s0.reject = 'none'; s0.j_cfg.update(CFG_R)
    s0.r_vals = np.array(R); s0.r_uncs = np.array(sR); s0.recompute()
    spec = figexp.default_spec(w._export_panels(), w._export_options())
    spec['options'].update(title='Test samples', name_0='Sample A', names_from='typed',
                           x_min='54.8', x_max='55.9')
    fig = Figure(figsize=figexp.figsize(spec), facecolor='white')
    w._export_render(fig, spec)
    ax = fig.axes[0]
    assert ax.get_title() == 'Test samples'
    assert abs(ax.get_xlim()[0] - 54.8) < 1e-9 and abs(ax.get_xlim()[1] - 55.9) < 1e-9
    leg = ax.get_legend()
    assert leg and any('Sample A' in t.get_text() for t in leg.get_texts())
    assert w.samples[0].name == 'raw name', 'the sample must keep its own name'
    print('  ok  title, renamed sample and x range applied; sample untouched')


def test_isochron_batch():
    print('Plateau: Ca/K and %40Ar* ranges, export of every isochron')
    import tempfile
    pl = load('4_plateau_isochron/plateau_isochron.py', 'pl_batch')
    # Ca/K: a far outlier is left off the scale, the rest keeps zero-based range
    lo, hi = pl._cak_ylim([0.3, 0.4, 0.5, 0.45, 0.35, 60.0], [0.05] * 6)
    assert lo == 0.0 and 0.5 < hi < 2.0, (lo, hi)
    # non-zero floor kept for values that all sit well above zero
    lo, hi = pl._cak_ylim([4.0, 4.5, 5.0, 5.5], [0.1] * 4)
    assert lo > 2.0, lo
    assert pl._p40_ylim([20, 99.5, 100.0]) == (0.0, 100.0)
    top = pl._p40_ylim([20, 99.5, 101.3])[1]
    assert 101.3 < top <= 106.0, top
    w = pl.PlateauWindow()
    load_demo(w)
    opts = w._batch_options()
    spec = figexp.default_spec(w._batch_panels(), opts)
    spec['options']['text_style'] = 'big'
    f = Figure(figsize=figexp.figsize(spec))
    w._export_render_batch(f, spec)
    assert f.axes
    with tempfile.TemporaryDirectory() as d:
        spec['fmt'] = 'png'; spec['options']['batch_one_pdf'] = False
        lines = w._export_samples_all(spec, d, lambda i, n, l='': False)
        print('  ok ', lines[:1])


if __name__ == '__main__':
    test_plateau()
    test_probability()
    test_isochron_batch()
    test_dialog_and_presets()
    test_isochron_variants()
    test_isochron_spread()
    test_page_switcher_and_range()
    test_probability_names_and_ranges()
    print('ALL PASSED')
