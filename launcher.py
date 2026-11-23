#!/usr/bin/env python3
"""SimplAr launcher: starts each tool and shows the Methods & Reference document.

Layout of this file, top to bottom:
  ToolLauncher, the launcher window, and AboutDialog
  the Methods & Reference document (build_about_html): a table of contents,
    then the sections, as one self-contained HTML string for a QTextBrowser.
    Equations are typeset with matplotlib's mathtext, rendered once to
    transparent PNGs in the user's cache folder (data_folders.cache_dir) and
    cached by content hash; without matplotlib an equation is shown as its
    LaTeX text.  Screenshots are taken from docs/img/ when that file exists.
  main
"""

import hashlib
import os
import struct
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QMessageBox, QFileDialog, QDialog, QTextBrowser
)
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QUrl
from PyQt5.QtGui import QFont, QColor, QPalette, QDesktopServices

import data_folders as dfold
import standards as stdlib


# (button label, script path relative to this folder), in workflow order
TOOLS = [
    ("Data converter",     ("1_data_converter", "data_converter.py")),
    ("Flux monitors",      ("2_flux_monitors", "flux_monitors.py")),
    ("Flux regression",    ("3_flux_regression", "flux_regression.py")),
    ("Plateau & isochron", ("4_plateau_isochron", "plateau_isochron.py")),
    ("Age distribution",   ("5_age_distribution", "age_distribution.py")),
    ("Recalculator and age combiner", ("6_recalculator", "recalculator.py")),
]

ROOT = os.path.dirname(os.path.abspath(__file__))


class ToolLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self._procs = []          # every running tool, stopped in closeEvent
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(36, 28, 36, 32)
        layout.setSpacing(10)

        heading = QLabel("SimplAr")
        heading.setAlignment(Qt.AlignCenter)
        heading.setFont(QFont("Georgia", 30))
        heading.setObjectName("heading")
        layout.addWidget(heading)

        rule = QLabel()
        rule.setObjectName("rule")
        rule.setFixedHeight(1)
        layout.addWidget(rule)
        layout.addSpacing(8)

        for label, rel in TOOLS:
            btn = QPushButton(label.replace('&', '&&'))   # '&' is not a shortcut
            btn.setFont(QFont("Georgia", 14))
            btn.setMinimumHeight(46)
            if rel[0] == "1_data_converter":
                btn.clicked.connect(self.launch_data_converter)
            else:
                btn.clicked.connect(lambda _c=False, r=rel, n=label: self.launch(r, n))
            layout.addWidget(btn)

        layout.addSpacing(12)
        low = QHBoxLayout()
        std_btn = QPushButton("Standards")
        std_btn.setFont(QFont("Georgia", 12))
        std_btn.setMinimumHeight(38)
        std_btn.setObjectName("aboutButton")
        std_btn.setToolTip("Monitors, calibrations, routes and decay constants")
        std_btn.clicked.connect(
            lambda: self.launch(("standards.py",), "the standards editor"))
        about_btn = QPushButton("About")
        about_btn.setFont(QFont("Georgia", 12))
        about_btn.setMinimumHeight(38)
        about_btn.setObjectName("aboutButton")
        about_btn.clicked.connect(self.show_about)
        low.addWidget(std_btn)
        low.addWidget(about_btn)
        layout.addLayout(low)

        # where results and sessions are saved (one subfolder per tool)
        layout.addSpacing(10)
        self.data_lbl = QLabel()
        self.data_lbl.setObjectName("dataFolder")
        self.data_lbl.setWordWrap(True)
        self.data_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.data_lbl)
        row = QHBoxLayout()
        open_btn = QPushButton("Open data folder")
        open_btn.setObjectName("smallButton")
        open_btn.clicked.connect(self.open_data_folder)
        change_btn = QPushButton("Change\u2026")
        change_btn.setObjectName("smallButton")
        change_btn.clicked.connect(self.change_data_folder)
        row.addWidget(open_btn)
        row.addWidget(change_btn)
        layout.addLayout(row)
        self._show_data_folder()

        layout.addStretch()

        self.setStyleSheet("""
            QWidget {
                background-color: #ece3d0;
            }
            QLabel#heading {
                color: #4a3b29;
            }
            QLabel#rule {
                background-color: #cdbb98;
                margin: 2px 48px;
            }
            QPushButton {
                background-color: #ddccae;
                color: #3a2e1e;
                border: 1px solid #a98e63;
                border-radius: 3px;
                padding: 8px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #d3c09c;
            }
            QPushButton:pressed {
                background-color: #c2ac84;
            }
            QPushButton#aboutButton {
                background-color: transparent;
                color: #6a5533;
                border: 1px solid #c3ad86;
                font-style: italic;
            }
            QPushButton#aboutButton:hover {
                background-color: #e3d7bd;
            }
            QPushButton#smallButton {
                padding: 4px;
                font-size: 11px;
            }
            QLabel#dataFolder {
                color: #6a5533;
                font-size: 11px;
            }
        """)

        self.resize(380, 650)

    # ── data folder ────────────────────────────────────────────────────────────
    def _show_data_folder(self):
        self.data_lbl.setText(f"Data folder:\n{dfold.data_root()}")

    def open_data_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(dfold.create_all()))

    def change_data_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose the data folder",
                                                dfold.data_root())
        if path:
            dfold.set_data_root(path)
            dfold.create_all()
            self._show_data_folder()

    # ── launching ──────────────────────────────────────────────────────────────
    def _script(self, rel):
        path = os.path.join(ROOT, *rel)
        if not os.path.exists(path):
            QMessageBox.critical(self, "File missing",
                                 f"{os.path.basename(path)} not found at:\n{path}")
            return None
        return path

    def _start(self, proc, path, args, label):
        proc.setWorkingDirectory(os.path.dirname(path))
        proc.finished.connect(lambda *_a, p=proc: self._forget(p))
        proc.start(sys.executable, args)
        self._procs.append(proc)
        if not proc.waitForStarted(3000):
            QMessageBox.critical(self, "Launch error", f"Failed to start {label}")

    def _forget(self, proc):
        if proc in self._procs:
            self._procs.remove(proc)

    def launch(self, rel, label):
        """Start one tool as its own process, in its own folder."""
        path = self._script(rel)
        if path is None:
            return
        proc = QProcess(self)

        def _closed(code, _status):
            if code != 0:
                QMessageBox.warning(self, "Closed", f"{label} closed (exit code {code}).")
        proc.finished.connect(_closed)
        self._start(proc, path, [path], label)

    def launch_data_converter(self):
        """The converter asks for its input and output here, then shows its own
        mapping dialog; it runs as a QProcess so the launcher stays responsive."""
        path = self._script(TOOLS[0][1])
        if path is None:
            return
        input_path, _ = dfold.get_open_file_name(
            self, "Pick an ArArCALC or Mass Spec export (.xlsx)", "",
            "Excel files (*.xlsx)", tool='data_converter')
        if not input_path:
            return
        output_dir = dfold.get_existing_directory(self, "Pick an output folder",
                                                  tool='data_converter')
        if not output_dir:
            return

        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'YES')
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.MergedChannels)

        def _done(code, _status, p=proc):
            output = bytes(p.readAll()).decode('utf-8', 'replace').strip()
            if code == 0:
                QMessageBox.information(self, "Done", output or "Conversion finished.")
            else:
                QMessageBox.warning(self, "Conversion error",
                                    f"Exited with code {code}.\n\n{output}")
        proc.finished.connect(_done)
        self._start(proc, path, ['-u', path, input_path, output_dir], "the data converter")

    # ── About / Methods document ─────────────────────────────────────────────────
    def show_about(self):
        dlg = AboutDialog(self)
        dlg.exec_()

    def closeEvent(self, event):
        for proc in list(self._procs):
            if proc and proc.state() != QProcess.NotRunning:
                proc.terminate()
                if not proc.waitForFinished(3000):
                    proc.kill()
        event.accept()


class AboutDialog(QDialog):
    """Full methods-and-reference document with a clickable table of contents."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SimplAr: Methods & Reference")
        self.resize(920, 760)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        # equation and screenshot images are referenced by bare filename
        browser.setSearchPaths([
            dfold.cache_dir('eqcache'),
            os.path.join(ROOT, 'docs', 'img'),
            ROOT,
        ])
        browser.setStyleSheet("QTextBrowser { background:#ffffff; color:#26211a; border:none; }")
        # Force a light palette on the browser so macOS dark/"night" mode can't
        # swap the default text color to white on top of the white background.
        about_palette = browser.palette()
        about_palette.setColor(QPalette.Base, QColor('#ffffff'))
        about_palette.setColor(QPalette.Text, QColor('#26211a'))
        browser.setPalette(about_palette)
        browser.setHtml(build_about_html(ROOT))
        # a margin on every side keeps the text clear of the scroll bar, which
        # macOS draws over the content
        browser.document().setDocumentMargin(26)
        layout.addWidget(browser)

        bar = QHBoxLayout()
        bar.setContentsMargins(12, 8, 12, 10)
        top_btn = QPushButton("↑ Back to top")
        top_btn.clicked.connect(lambda: browser.scrollToAnchor("top"))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bar.addWidget(top_btn)
        bar.addStretch()
        bar.addWidget(close_btn)
        layout.addLayout(bar)


# --------------------------------------------------------------------------
# Equation rendering: LaTeX (mathtext) to a cached transparent PNG
# --------------------------------------------------------------------------

_EQ_FONTSIZE = 15
_EQ_DPI = 200
_EQ_COLOR = '#26211a'

# matplotlib is imported lazily and only once
_MPL = {}


def _load_mpl():
    if _MPL:
        return _MPL.get('plt')
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        import matplotlib.pyplot as plt
        _MPL['plt'] = plt
    except Exception:
        _MPL['plt'] = None
    return _MPL['plt']


def _png_width_px(path):
    """Read a PNG's pixel width straight from its IHDR chunk."""
    with open(path, 'rb') as fh:
        head = fh.read(24)
    return struct.unpack('>I', head[16:20])[0]


def _render_eq(base_dir, latex):
    """Return (filename, display_width_px) for a rendered equation, or None."""
    cache = dfold.cache_dir('eqcache')
    key = hashlib.md5(
        f'cm|{_EQ_FONTSIZE}|{_EQ_DPI}|{_EQ_COLOR}|{latex}'.encode('utf-8')
    ).hexdigest()[:16]
    fname = key + '.png'
    path = os.path.join(cache, fname)
    if not os.path.exists(path):
        plt = _load_mpl()
        if plt is None:
            return None
        try:
            fig = plt.figure()
            fig.patch.set_alpha(0.0)
            fig.text(0.01, 0.5, f'${latex}$', fontsize=_EQ_FONTSIZE,
                     color=_EQ_COLOR, va='center', ha='left')
            fig.savefig(path, dpi=_EQ_DPI, transparent=True,
                        bbox_inches='tight', pad_inches=0.06)
            plt.close(fig)
        except Exception:
            return None
    try:
        disp = max(1, int(round(_png_width_px(path) / 2)))
    except Exception:
        disp = None
    return fname, disp


def _eq(base_dir, latex):
    r = _render_eq(base_dir, latex)
    if r is None:
        return ('<div style="margin:10px 0 12px 24px; font-family:Menlo,'
                'monospace; font-size:12.5px; color:#33291a;">'
                f'{latex}</div>')
    fname, disp = r
    w = f' width="{disp}"' if disp else ''
    return (f'<div style="margin:12px 0 14px 24px;">'
            f'<img src="{fname}"{w}></div>')


# --------------------------------------------------------------------------
# Figures: a screenshot from docs/img/, left out when the file is missing
# --------------------------------------------------------------------------

def _fig(base_dir, filename, caption):
    path = os.path.join(base_dir, 'docs', 'img', filename)
    if not os.path.exists(path):
        return ''
    return ('<div style="margin:14px 0 16px 0;">'
            f'<img src="{filename}" style="max-width:100%; border:1px solid #d8cdb4;">'
            '<div style="font-size:11px; color:#7a6f5c; margin-top:4px;">'
            f'{caption}</div></div>')


# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------

def _basis_text(c):
    """How the λ uncertainty enters for a calibration, in a few words."""
    basis = c.get('basis', 'k-ar')
    if basis == 'astronomical':
        return 'Astronomical', 'small (partial derivative)'
    if basis == 'joint' and c.get('covariance') is not None:
        return 'Joint with &lambda;', 'through the covariance'
    if c.get('sigma_includes_lambda'):
        return ('Joint with &lambda;' if basis == 'joint' else 'K&ndash;Ar'), 'in the age error'
    return ('Joint with &lambda;' if basis == 'joint' else 'K&ndash;Ar'), 'full, t&middot;&sigma;<sub>&lambda;</sub>/&lambda;'


def _calibration_table():
    """Every calibration in the published standards library."""
    rows = ['<table>',
            '<tr><th>Standard</th><th>Calibration</th><th>Age (Ma)</th><th>1&sigma; (Ma)</th>'
            '<th>&lambda;</th><th>Basis</th><th>&lambda; term</th></tr>']
    for c in stdlib.load_default()['calibrations']:
        basis, lam_term = _basis_text(c)
        default = ' (default)' if c.get('default') else ''
        rows.append(f"<tr><td>{c.get('standard', 'FCs')}</td>"
                    f"<td>{c['name'].split('  ')[0]}{default}</td><td>{c['age']:g}</td>"
                    f"<td>{c['age_1s']:g}</td><td>{c['decay_constant']}</td>"
                    f"<td>{basis}</td><td>{lam_term}</td></tr>")
    rows.append('</table>')
    return '\n'.join(rows)


def _lambda_table():
    """The decay constants in the published standards library."""
    rows = ['<table>', '<tr><th>Decay constant</th><th>&lambda; (yr<sup>&minus;1</sup>)</th>'
            '<th>1&sigma;</th></tr>']
    for d in stdlib.load_default()['decay_constants']:
        s = float(d.get('lambda_1s') or 0)
        rows.append(f"<tr><td>{d['name']}</td><td>{float(d['lambda']):.5g}</td>"
                    f"<td>{f'{s:.2g}' if s else 'none'}</td></tr>")
    rows.append('</table>')
    return '\n'.join(rows)


def _monitor_table():
    """The monitors and the R marked Use, from the published library."""
    rows = ['<table>', '<tr><th>Standard</th><th>R</th><th>1&sigma;</th>'
            '<th>Source</th></tr>']
    for s in stdlib.load_default()['standards']:
        if s['id'] == 'FCs':
            continue
        _, _, ic = stdlib.recommended_R(s)
        num, den = ('FCs', s['id']) if ic.get('ratio') == 'FCs/std' else (s['id'], 'FCs')
        rows.append(f"<tr><td>{s['name']}</td><td>R<sub>{num}/{den}</sub> = {float(ic['R']):.6g}</td>"
                    f"<td>{float(ic.get('R_1s') or 0):.2g}</td>"
                    f"<td>{stdlib.short_ref(ic.get('reference'))}</td></tr>")
    rows.append('</table>')
    return '\n'.join(rows)


def build_about_html(base_dir):
    E = lambda tex: _eq(base_dir, tex)
    F = lambda name, cap: _fig(base_dir, name, cap)

    style = """
    <style>
      body   { font-family: Georgia, 'Times New Roman', serif; font-size: 14.5px;
               color: #26211a; line-height: 1.66; }
      h1     { font-size: 26px; margin: 4px 0 2px 0; color: #3a2e1e; }
      h2     { font-size: 20px; margin: 30px 0 8px 0; color: #4a3b29;
               border-bottom: 2px solid #c3ad86; padding-bottom: 5px; }
      h3     { font-size: 15px; font-style: italic; margin: 18px 0 4px 0; color: #5a4a30; }
      p      { margin: 9px 0; }
      li     { margin: 5px 0; }
      code   { font-family: Menlo, Consolas, monospace; font-size: 12px;
               background: #f2efe7; padding: 0 3px; border-radius: 2px; }
      .lede  { font-size: 15px; color: #40372b; }
      .toc   { background: #f6f4ee; border: 1px solid #ddd0b4;
               border-radius: 4px; padding: 12px 10px 12px 4px; margin: 14px 0; }
      .toc a { color: #6a4a1e; text-decoration: none; }
      .toc a:hover { text-decoration: underline; }
      table  { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
      th     { background: #ece3d0; text-align: left; padding: 6px 9px;
               border: 1px solid #cdbb98; }
      td     { padding: 6px 9px; border: 1px solid #e0d6bf; vertical-align: top; }
      .back  { font-size: 11px; }
      .back a { color: #9a875f; text-decoration: none; }
    </style>
    """

    toc = """
    <a name="top"></a>
    <h1>SimplAr: Methods &amp; Reference</h1>
    <p class="lede">The methods and equations used by the six tools.</p>

    <div class="toc">
      <b>Contents</b>
      <ol>
        <li><a href="#workflow">The workflow</a></li>
        <li><a href="#idea">The age equation</a></li>
        <li><a href="#tiers">The three uncertainty tiers</a></li>
        <li><a href="#stats">Combining ages</a></li>
        <li><a href="#converter">Data converter</a></li>
        <li><a href="#wm">Flux monitors</a></li>
        <li><a href="#jcalc">Flux regression</a></li>
        <li><a href="#plateau">Plateau &amp; isochron</a></li>
        <li><a href="#prob">Age distribution</a></li>
        <li><a href="#calib">Standards, routes and calibrations</a></li>
        <li><a href="#recalc">Recalculator and age combiner</a></li>
        <li><a href="#refs">References</a></li>
      </ol>
    </div>
    """

    # ── 1 ────────────────────────────────────────────────────────────────
    workflow = """
    <a name="workflow"></a>
    <h2>1. The workflow</h2>

    <p>Each tool runs on its own, but they are normally used in this order.</p>
    <ol>
      <li><a href="#converter"><b>Data converter</b></a> turns an ArArCALC or Mass Spec
          export (.xlsx) into <code>.arar</code> files.</li>
      <li><a href="#wm"><b>Flux monitors</b></a> reduces the monitor analyses of each
          irradiation position to one R and J.</li>
      <li><a href="#jcalc"><b>Flux regression</b></a> fits J across the irradiation
          disk and predicts it, with its uncertainty, at the position of each
          unknown.</li>
      <li><a href="#plateau"><b>Plateau &amp; isochron</b></a> calculates plateau and
          isochron ages of step-heating runs, and combined ages of several aliquots.</li>
      <li><a href="#prob"><b>Age distribution</b></a> draws probability density plots,
          calculates weighted means and pooled ages of many analyses or samples, and
          fits mixture models.</li>
      <li><a href="#recalc"><b>Recalculator and age combiner</b></a> recalculates
          published ages to another calibration and combines them.</li>
    </ol>
    """

    # ── 2 ────────────────────────────────────────────────────────────────
    idea = """
    <a name="idea"></a>
    <h2>2. The age equation</h2>

    <p>R is the ratio of radiogenic to potassium-derived argon,
    <sup>40</sup>Ar*/<sup>39</sup>Ar<sub>K</sub>. With the irradiation parameter J and
    the decay constant &lambda;, the age is</p>
    """ + E(r't = \frac{1}{\lambda}\,\ln\left(1 + J\,R\right)') + """
    <p>J comes from a monitor M, a standard irradiated with the sample, with measured
    ratio R<sub>M</sub> and age t<sub>M</sub>.</p>
    """ + E(r'J = \frac{e^{\lambda t_{M}} - 1}{R_{M}}') + """
    <p>The monitor age follows from the route chosen for the monitor in Standards
    (Section&nbsp;10). A route goes from M through one or more
    measured ratios R<sub>i</sub> between standards to a standard C whose age
    t<sub>C</sub> is the calibration.</p>
    """ + E(r'e^{\lambda t_{M}} - 1 = \prod_{i} R_{i}\,\left(e^{\lambda t_{C}} - 1\right)') + """
    <p>For a Fish Canyon sanidine (FCs) monitor on an FCs calibration the route has no
    steps and the product is 1. Substituting J gives the age in terms of the
    sample-to-monitor ratio R<sub>u/M</sub>&nbsp;=&nbsp;R/R<sub>M</sub>.</p>
    """ + E(r't = \frac{1}{\lambda}\,\ln\left[\,1 + R_{u/M}\prod_{i} R_{i}\left(e^{\lambda t_{C}} - 1\right)\right]') + """
    <p>&lambda;t is small (about 0.05 at 100&nbsp;Ma), so the age is nearly
    proportional to the calibration age,
    t&nbsp;&asymp;&nbsp;R<sub>u/M</sub>&thinsp;&Pi;R<sub>i</sub>&thinsp;t<sub>C</sub>.
    How much the uncertainty of &lambda; matters depends on how t<sub>C</sub> was
    determined (Section&nbsp;3).</p>
    """

    # ── 3 ────────────────────────────────────────────────────────────────
    tiers = """
    <a name="tiers"></a>
    <h2>3. The three uncertainty tiers</h2>

    <p>Every age is reported with three uncertainties. Each adds further sources to the
    one before, and which to quote depends on what the age is compared with.</p>

    <table>
      <tr><th>Tier</th><th>Contains</th><th>Use for comparison with</th></tr>
      <tr><td><b>Analytical</b></td>
          <td>The measurement of R (counting statistics, blanks, detector).</td>
          <td>Aliquots from the same irradiation position.</td></tr>
      <tr><td><b>Internal</b></td>
          <td>Plus the J uncertainty.</td>
          <td>Ar/Ar ages on the same calibration and &lambda;.</td></tr>
      <tr><td><b>External</b></td>
          <td>Plus the route, the calibration age and the decay constant.</td>
          <td>Ar/Ar ages on other calibrations or from other laboratories, and other
              timescales.</td></tr>
    </table>

    <p>For a single age</p>
    """ + E(r'\sigma_{\mathrm{int}}^{2} = \sigma_{\mathrm{ana}}^{2} + \left(\frac{\partial t}{\partial \ln J}\,f_{J}\right)^{2} \qquad \frac{\partial t}{\partial \ln J} = \frac{1 - e^{-\lambda t}}{\lambda}') + \
    E(r'\sigma_{\mathrm{ext}}^{2} = \sigma_{\mathrm{int}}^{2} + \sum_{k} c_{k}^{2}') + """
    <p>f<sub>J</sub> is the relative uncertainty of J from the monitor measurement
    alone, and the c<sub>k</sub> are the calibration terms below.
    &part;t/&part;ln&nbsp;J is a little less than t (2.7% less at 100&nbsp;Ma), and is
    the same for a relative change in any ratio in the age equation.</p>

    <h3>Calibration terms</h3>
    <p>The external tier contains one term for each input of the route. These are
    each ratio R<sub>i</sub>, the calibration age t<sub>C</sub> and &lambda;. With
    Q&nbsp;=&nbsp;R<sub>u/M</sub>&thinsp;&Pi;R<sub>i</sub> and
    X&nbsp;=&nbsp;e<sup>&lambda;t</sup>,</p>
    """ + E(r'X = Q\left(e^{\lambda t_{C}} - 1\right) + 1') + \
    E(r'\frac{\partial t}{\partial \ln R_{i}} = \frac{1 - e^{-\lambda t}}{\lambda} \qquad \frac{\partial t}{\partial t_{C}} = \frac{Q\,e^{\lambda t_{C}}}{X} \qquad \frac{\partial t}{\partial \lambda} = -\frac{\ln X}{\lambda^{2}} + \frac{Q\,t_{C}\,e^{\lambda t_{C}}}{\lambda X}') + \
    E(r'\sum_{k} c_{k}^{2} = \sum_{i}\left(\frac{\partial t}{\partial \ln R_{i}}\frac{\sigma_{R_{i}}}{R_{i}}\right)^{2} + \left(\frac{\partial t}{\partial t_{C}}\sigma_{t_{C}}\right)^{2} + \left(\frac{\partial t}{\partial \lambda}\sigma_{\lambda}\right)^{2} + 2\,\frac{\partial t}{\partial t_{C}}\frac{\partial t}{\partial \lambda}\,\mathrm{cov}(t_{C}, \lambda)') + """
    <p>Every age on the same route shares these terms, so they do not average down
    when ages are combined (Section&nbsp;4).</p>

    <h3>How &lambda; enters</h3>
    <p>The &lambda; term depends on how the calibration age was determined. Each calibration
    in the library has a basis that sets this (Section&nbsp;10).</p>
    <table>
      <tr><th>Basis</th><th>Examples</th><th>&lambda; term</th></tr>
      <tr><td><b>Astronomical</b></td>
          <td>The A1, Melilla and Mes4 tephras, and the FCs ages derived from them</td>
          <td>t<sub>C</sub> does not depend on &lambda;, so &part;t/&part;&lambda; above
              is used. It is small. A 1% change in &lambda; moves a 100&nbsp;Ma age by
              0.025%.</td></tr>
      <tr><td><b>Joint with &lambda;</b></td>
          <td>Carter et&nbsp;al. (2025)</td>
          <td>t<sub>C</sub> and &lambda; were fitted together, and their covariance
              enters through the last term above.</td></tr>
      <tr><td><b>K&ndash;Ar</b></td>
          <td>GA1550, MMhb-1, Renne et&nbsp;al. (1998) FCs</td>
          <td>t<sub>C</sub> was calculated with &lambda; and changes with it. The
              derivative becomes &minus;t/&lambda;, so the full relative uncertainty of
              &lambda; reaches the age, 1.0% at 1&sigma; for Min et&nbsp;al.
              (2000).</td></tr>
    </table>
    <p>If a calibration's published uncertainty already contains &lambda;, the
    &lambda; term is not added again.</p>

    <h3>FCs ages derived from a tephra</h3>
    <p>The FCs ages of Kuiper et&nbsp;al. (2008), Rivera et&nbsp;al. (2011) and
    Phillips et&nbsp;al. (2022) were calculated from the astronomical age t<sub>A</sub>
    of a tephra and the measured ratio R<sub>FCs/A</sub>, with the Min et&nbsp;al.
    (2000) &lambda;. Choosing one of these FCs ages therefore gives the same result as
    the route FCs&nbsp;&rarr;&nbsp;A on the tephra's astronomical age. SimplAr
    calculates it that way, so the external tier contains R<sub>FCs/A</sub>,
    t<sub>A</sub> and &lambda;, and not the quoted FCs age error, which is built from
    the same three (Renne et&nbsp;al. 1998, eq.&nbsp;5; Kuiper et&nbsp;al. 2008).</p>
    """ + E(r't = \frac{1}{\lambda}\ln\left[1 + R_{u/\mathrm{FCs}}\,R_{\mathrm{FCs}/A}\left(e^{\lambda t_{A}} - 1\right)\right]') + """
    <table>
      <tr><th>FCs calibration</th><th>Tephra</th><th>t<sub>A</sub> (Ma, 1&sigma;)</th><th>R<sub>FCs/A</sub> (1&sigma;)</th></tr>
      <tr><td>Kuiper et al. (2008)</td><td>Melilla tephras, composite</td><td>6.500 &plusmn; 0.005</td><td>4.3664 &plusmn; 0.0009</td></tr>
      <tr><td>Rivera et al. (2011)</td><td>A1 tephra, Faneromeni</td><td>6.943 &plusmn; 0.0025</td><td>4.0813 &plusmn; 0.0013</td></tr>
      <tr><td>Phillips et al. (2022)</td><td>A1 tephra, Faneromeni</td><td>6.943 &plusmn; 0.0025</td><td>4.0819 &plusmn; 0.0007</td></tr>
      <tr><td>Phillips et al. (2022), Mes4</td><td>Mes4 tuff, Mess&acirc;dit</td><td>6.791 &plusmn; 0.005</td><td>4.1738 &plusmn; 0.00065</td></tr>
    </table>
    <p>The R<sub>FCs/A</sub> used is the one that gives each paper's FCs age with the
    Min et&nbsp;al. (2000) &lambda;. It differs from the printed value in the fifth
    figure. If the FCs age error is edited, the FCs age and the typed error are used as
    they stand, without the tephra.</p>

    <h3>Comparing with other decay systems</h3>
    <p>When an Ar/Ar age is compared with another decay system, such as U&ndash;Pb, the
    &lambda; term in the external tier is what matters. On a K&ndash;Ar calibration it is
    already the full t&middot;&sigma;<sub>&lambda;</sub>/&lambda;. On an astronomical
    calibration it stays small, and that holds against U&ndash;Pb too, where the
    comparison is between the astronomical and the uranium timescales.</p>
    <p>Renne et&nbsp;al. (2010, 2011) also calibrated the FCs age and &lambda;
    together, but their published covariances are rounded too far to be used for ages
    below about 1&nbsp;Ga. They recommend their Monte Carlo results instead, which are
    not included here. SimplAr therefore treats the FCs age and &lambda; of these two
    as independent and adds the full &lambda; term. This overstates the external error, and the
    tools say so when either is selected.</p>

    <h3>Full external (Min et al. 2000)</h3>
    <p>The External error setting of each tool offers <b>Full external</b>. It replaces
    the &lambda; term by the uncertainties of the two partial decay constants,
    &lambda;<sub>e</sub> (to <sup>40</sup>Ar) and &lambda;<sub>&beta;</sub> (to
    <sup>40</sup>Ca), propagated independently through the age equation with branching
    (Min et&nbsp;al. 2000, with the Audi et&nbsp;al. 1997 constants).</p>
    """ + E(r't = \frac{1}{\lambda}\ln\left(1 + \frac{\lambda}{\lambda_{e}}\,D\right) \qquad \lambda = \lambda_{e} + \lambda_{\beta}') + """
    <p>D is <sup>40</sup>Ar*/<sup>40</sup>K of the sample. The term is about 1.2% of the
    age at 1&sigma; at 100&nbsp;Ma (1.0% at 1&nbsp;Ga) and comes almost entirely from
    &lambda;<sub>e</sub>. It treats the decay constants as independent of the
    calibration age, so the tools warn when it is chosen with a jointly calibrated
    &lambda; such as Carter et&nbsp;al. (2025).</p>
    """

    # ── 4 ────────────────────────────────────────────────────────────────
    stats = """
    <a name="stats"></a>
    <h2>4. Combining ages</h2>

    <p>Wherever ages are combined, two methods are offered, <b>WM + √MSWD</b> and
    <b>GLS + ML</b>. Both use the same covariance. The J and calibration parts of the
    uncertainty are correlated between ages that share an irradiation position or a
    calibration, so they do not average down. The methods differ in the mean and in how
    they treat overdispersion, which is scatter larger than the analytical errors
    allow.</p>
    <table>
      <tr><th></th><th>WM + √MSWD</th><th>GLS + ML</th></tr>
      <tr><td><b>Mean</b></td>
          <td>Inverse-variance weighted mean on the analytical errors.</td>
          <td>The generalised least-squares (GLS) mean of Vermeesch (2015,
              eqs. 91&ndash;92) on the full covariance plus &omega;<sup>2</sup>.</td></tr>
      <tr><td><b>Overdispersion</b></td>
          <td>The analytical tier is multiplied by &radic;MSWD when the MSWD exceeds the
              Wendt &amp; Carl (1991) limit below.</td>
          <td>An extra variance &omega;<sup>2</sup>, estimated by maximum likelihood on
              the full covariance and added to every age.</td></tr>
    </table>
    """ + E(r'\mathrm{MSWD} > 1 + 2\sqrt{\frac{2}{n-1}}') + """
    <p>The same limit is used for plateaus, combined ages, flux monitors and flux
    regression. Isochrons are the exception, and multiply their uncertainties by
    &radic;MSWD whenever the MSWD exceeds 1 (Section&nbsp;8). Because &omega;<sup>2</sup> is fitted on
    the full covariance (analytical, J and calibration), scatter already explained by
    independent J errors is not counted twice. Vermeesch (2015, eq.&nbsp;94) adds it to
    each age's own variance, and IsoplotR fits it on the analytical errors only.</p>

    <h3>Which to use</h3>
    <p>Without overdispersion, or when the analytical errors are similar, the two give
    practically the same age and uncertainty. They differ when the ages scatter and
    their errors differ widely, as for single crystals of variable K content. A weighted
    mean then follows the few most precise ages, and multiplying its error by
    &radic;MSWD does not make up for that. GLS + ML adds &omega;<sup>2</sup> to every
    age, which evens out the weights. Its mean is closer to the true age and its
    uncertainty more realistic. With few ages, roughly fewer than ten, &omega; itself is
    poorly known and both methods tend to understate the uncertainty.</p>

    <p>For stacked step-heating aliquots, GLS + ML combines the plateau ages of the
    aliquots (n = aliquots). WM + √MSWD does the same by default, or pools every step in
    ratio space (n = steps). When the aliquots disagree beyond analytical error, use
    n = aliquots. The steps of one aliquot share that aliquot's offset, and treating them
    as independent understates the uncertainty, more so the more steps there are.</p>

    <p>The tools start from reduced ages, so the covariance contains only the J and
    calibration terms. Correlations from shared blanks, detector calibration, mass
    fractionation and interference corrections need the raw isotope data and are not
    included.</p>
    """

    # ── 5 ────────────────────────────────────────────────────────────────
    converter = """
    <a name="converter"></a>
    <h2>5. Data converter</h2>

    <p>Reads an ArArCALC summary export or a Mass Spec raw export (.xlsx) and writes
    <code>.arar</code> files for the other tools. A dialog sets the layout and the
    column of each field before anything is written. Each file stores J and its error,
    the monitor ratio R<sub>M</sub>, the calibration age, &lambda; and the irradiation
    label. The other tools therefore know the calibration the ages were calculated with
    and can recalculate them on another.</p>

    """ + F('converter_mapping.png',
            'The format and column-mapping dialog, with the export layout and the '
            'spreadsheet column of each field.') + """

    <p>Ages are not recalculated. J and R<sub>M</sub> are kept consistent through
    J&nbsp;=&nbsp;(e<sup>&lambda;t<sub>M</sub></sup>&nbsp;&minus;&nbsp;1)/R<sub>M</sub>,
    so a calibration given in either form gives the same age. The calibrations offered
    are those of Section&nbsp;10.</p>
    """

    # ── 6 ────────────────────────────────────────────────────────────────
    wm = """
    <a name="wm"></a>
    <h2>6. Flux monitors</h2>

    <p>Reduces the monitor analyses of one irradiation position to one R and J, and
    separates real scatter (flux gradients, grain-to-grain differences) from analytical
    scatter. It reads .arar, .ararpt and .ararstack files (R and its 1&sigma; error from
    every step, grouped by irradiation position), Excel/CSV tables and pasted values.
    With "Value is a <sup>40</sup>Ar*/<sup>39</sup>Ar ratio" unticked on import it is a
    weighted-mean calculator for any quantity.</p>

    """ + F('wm_main.png',
            'The analyses with 1-sigma error bars, the mean and its uncertainty band, '
            'and the MSWD and p-value in the title.') + """

    <h3>Weighted mean</h3>
    """ + E(r'w_{i} = \frac{1}{\sigma_{i}^{2}} \qquad \bar{x} = \frac{\sum w_{i} x_{i}}{\sum w_{i}} \qquad \mathrm{SE} = \sqrt{\frac{1}{\sum w_{i}}}') + \
    E(r'\mathrm{MSWD} = \frac{1}{n-1}\sum w_{i}\left(x_{i} - \bar{x}\right)^{2} \qquad p = P\left(\chi^{2}_{n-1} > \mathrm{MSWD}\,(n-1)\right)') + """
    <p>x<sub>i</sub>&nbsp;&plusmn;&nbsp;&sigma;<sub>i</sub> are the analyses. An MSWD
    near 1 means the scatter is explained by the analytical errors. Above 1 with a low
    p there is additional variance, and well below 1 the errors are
    overestimated.</p>

    <h3>Excess scatter (REML)</h3>
    <p>The excess scatter is estimated as a variance &tau;<sup>2</sup> by restricted
    maximum likelihood, which minimises</p>
    """ + E(r'\sum_{i}\ln\left(\sigma_{i}^{2} + \tau^{2}\right) + \ln W^{*} + \sum_{i} w_{i}^{*}\left(x_{i} - \hat{\mu}^{*}\right)^{2}, \qquad w_{i}^{*} = \frac{1}{\sigma_{i}^{2} + \tau^{2}}, \quad W^{*} = \sum_{i} w_{i}^{*}') + \
    E(r'\hat{\mu}^{*} = \frac{\sum_{i} w_{i}^{*} x_{i}}{W^{*}}') + """
    <p>The ln&nbsp;W* term allows for the mean being estimated from the same data, so
    &tau;<sup>2</sup> is not biased low for the few monitors of one position. With
    &tau;<sup>2</sup>&nbsp;=&nbsp;0 the result is the ordinary weighted mean. Otherwise
    &tau; is the grain-to-grain spread in units of R. This is the model of GLS + ML
    (Section&nbsp;4) applied to the monitors. The standard error is 1/&radic;W* and does
    not include the uncertainty of &tau;<sup>2</sup>.</p>
    <p>When the monitors scatter more than their errors allow, REML gives a more
    realistic uncertainty than the weighted mean with its SE multiplied by
    &radic;MSWD, which is what the tool reports with REML unticked. Without excess
    scatter the two agree.</p>

    <h3>J</h3>
    """ + E(r'J = \frac{e^{\lambda t_{M}} - 1}{R} \qquad \frac{\sigma_{J}}{J} = \frac{\sigma_{R}}{R}') + """
    <p>R is the mean monitor ratio and e<sup>&lambda;t<sub>M</sub></sup>&nbsp;&minus;&nbsp;1
    comes from the route of the monitor (Section&nbsp;2). &sigma;<sub>J</sub> contains
    the monitor measurement only. The route terms are added once, in the external tier
    of the ages (Section&nbsp;3). J is calculated from the reported R, so the J view has
    the same accepted and rejected points, MSWD and p as the R analysis.</p>

    <h3>Crystals: fusion and step heating</h3>
    <p>A crystal is identified by a grain-ID column (a blank cell continues the crystal
    above) or, in an .arar file, by the run-ID prefix (100-01A &rarr; 100-01) or by the
    laser power in run order. A step at the same or a lower power than the step before
    starts a new crystal. A crystal with one step is a fusion, one with several was
    step-heated. The points averaged can be every step, one point per crystal, or the
    single-step crystals only. One point per crystal is the total gas, the
    <sup>39</sup>Ar-weighted ratio</p>
    """ + E(r'R_{\mathrm{TG}} = \sum_{i} f_{i} R_{i}, \quad f_{i} = \frac{{}^{39}\mathrm{Ar}_{i}}{\sum {}^{39}\mathrm{Ar}} \qquad \sigma_{\mathrm{TG}} = \sqrt{\sum_{i} f_{i}^{2}\sigma_{i}^{2}}') + """
    <p>which equals what a single fusion of the crystal would have measured, so fused
    and step-heated crystals can be averaged together. Without <sup>39</sup>Ar the
    inverse-variance mean of the steps is used and labelled as such. For each
    step-heated crystal the MSWD of its steps is reported, and the steps can be grouped
    by laser power or step number to compare a weighted mean per group.</p>

    <h3>Outlier rejection</h3>
    <p>By default two filters are applied. A relative-error cut removes analyses with
    poor counting statistics, and NMAD rejection removes points more than k&nbsp;NMAD
    from the median.</p>
    """ + E(r'\mathrm{NMAD} = 1.4826 \times \mathrm{median}\,\left| x_{i} - \mathrm{median}\right|') + """
    <p>The NMAD is not inflated by several outliers at once, but it looks at the values
    only, not their errors. When the errors differ a lot it tends to remove the least
    precise points, even when they agree within error, and each rejection trims the
    scatter that &tau;<sup>2</sup> measures. The default is k&nbsp;=&nbsp;2.5, as in Age
    distribution. A larger k removes fewer good points.</p>
    <p>MSWD rejection, when ticked, removes one point at a time while the MSWD exceeds
    the Wendt &amp; Carl limit (or a set value). The candidate is the point with the
    largest externally studentized residual (Viechtbauer &amp; Cheung 2010),</p>
    """ + E(r'r_{i} = \frac{x_{i} - \hat{\mu}_{(-i)}}{\sqrt{\sigma_{i}^{2} + \hat{\tau}^{2}_{(-i)} + \mathrm{SE}^{2}_{(-i)}}} \qquad |r_{i}| > \Phi^{-1}\!\left(1 - \frac{0.025}{n}\right)') + """
    <p>where (&minus;i) marks the weighted mean, excess variance and standard error of
    the other points. The point is removed only if |r<sub>i</sub>| exceeds the 5%
    two-sided critical value corrected for n tests (2.81 at n&nbsp;=&nbsp;10, 3.02 at
    n&nbsp;=&nbsp;20). If no point does, the MSWD stays above the limit and the tool
    reports it. The scatter is then spread over the whole set and is handled by REML or
    &radic;MSWD. This rarely removes a good point and reliably finds a clear outlier,
    whereas removing points until the MSWD falls below the limit often removes good
    ones. The filters run in the order relative error, NMAD, MSWD.</p>
    """

    # ── 7 ────────────────────────────────────────────────────────────────
    jcalc = """
    <a name="jcalc"></a>
    <h2>7. Flux regression</h2>

    <p>J is measured only in the monitor positions. The tool fits a surface through the
    monitor values and predicts J, with a Monte Carlo uncertainty, at the position of
    each unknown.</p>
    <p>The monitor values can be typed or pasted into the disk tables, or imported from
    an .xlsx export of the Flux monitors tool (File &rarr; Import Flux monitors export).
    A category named like C1 goes to the disk labelled C, position 1, with its reported R
    and SE. In J mode, J is calculated from that R with the calibration set in this
    tool.</p>

    """ + F('jcalc_surface.png',
            'A fitted flux surface over the irradiation disk. The monitor positions are '
            'the control points, and unknowns are predicted from the surface.') + """

    <h3>Surfaces</h3>
    <table>
      <tr><th>Model</th><th>Surface</th><th>Parameters</th><th>Minimum monitor positions</th></tr>
      <tr><td>Planar (2D)</td><td>v = ax + by + c</td><td>3</td><td>3</td></tr>
      <tr><td>Simple 3D</td><td>v = ax + by + cz + d</td><td>4</td><td>4</td></tr>
      <tr><td>Complex 3D</td><td>v = ax + by + cz + d + e(xz) + f(yz) + g(xy) + h(xyz)</td><td>8</td><td>8 (12 or more preferred)</td></tr>
    </table>
    <p>v is the monitor value (J, or R in R mode), x and y are the position coordinates
    (cm) and z the height of the disk in the stack.</p>
    <ul>
      <li><b>Planar</b>. J changes linearly across the disk. Each disk gets its own
          plane, with nothing shared between disks.</li>
      <li><b>Simple 3D</b> adds a linear change with height, so every disk has the same
          lateral gradient on top of a gradient along the stack. It needs monitors on at
          least two levels.</li>
      <li><b>Complex 3D</b> also lets the lateral gradient change with height (xz, yz)
          and allows a saddle-shaped surface (xy, xyz). Use it only with many monitor
          positions spread over several disks. With fewer than about 12, its eight
          parameters fit the noise.</li>
    </ul>
    <p>Use the simplest model the monitors support, and a more complex one only if it
    predicts left-out monitors better (see Choosing a model below).</p>

    <h3>Monte Carlo uncertainty</h3>
    <p>Each iteration moves the monitor positions at random, adds Gaussian noise to
    each monitor value at its stated error, and refits the surface by weighted least
    squares (weights 1/&sigma;<sup>2</sup>). The mean and standard deviation of the
    predictions are the J and &sigma;<sub>J</sub> of each unknown. Monitors keep their
    measured values.</p>
    """ + E(r'\hat{J}(x,y,z) = \langle J \rangle_{\mathrm{MC}} \qquad \sigma_{J}(x,y,z) = \mathrm{std}_{\mathrm{MC}}') + """
    <p>The position shift allows for where the monitor crystals lay. A monitor value is
    the mean of n crystals, each anywhere on the floor of a hole of diameter d. A random
    point on a disc has a standard deviation of d/4 in x and in y, so the mean position
    of n crystals is uncertain by</p>
    """ + E(r'\sigma_{\mathrm{pos}} = \frac{d}{4\sqrt{n}}') + """
    <p>in each direction, 0.2&ndash;0.3&nbsp;mm for about ten crystals in the
    2.7&ndash;4.0&nbsp;mm monitor holes of the 13- and 21-hole disks. The standard
    deviation of the shift is drawn in each iteration between 15 and 450&nbsp;&mu;m
    (typically about 0.26&nbsp;mm). Disk heights are not shifted.</p>
    <p>If the monitors scatter about the surface more than their errors allow (weighted
    MSWD above the Wendt &amp; Carl limit for n&nbsp;&minus;&nbsp;k degrees of freedom,
    with k the number of parameters), every &sigma;<sub>J</sub> is multiplied by
    &radic;MSWD and the correlations between positions are kept. The MSWD of each fit is
    reported and stored in the <code>.jflux</code> file.</p>

    <h3>Choosing a model</h3>
    <p>A more flexible surface always fits the monitors better, so the models are
    compared by the leave-one-out cross-validation error, the AIC, which penalises extra
    parameters, and the reduced &chi;<sup>2</sup>, which is near 1 when the errors are
    right. J and &sigma;<sub>J</sub> are passed on as a <code>.jflux</code> file, from
    which Plateau &amp; isochron, Age distribution and the Recalculator read the J of
    each position and the correlation between positions. For a fit through several disks
    the file also holds the correlation between positions on different disks.</p>
    """

    # ── 8 ────────────────────────────────────────────────────────────────
    plateau = """
    <a name="plateau"></a>
    <h2>8. Plateau &amp; isochron</h2>

    <p>Draws the age spectrum of a step-heating run, finds the plateau, calculates its
    age with the three uncertainty tiers, fits isochrons, and combines several aliquots
    of one sample.</p>

    """ + F('plateau_spectrum.png',
            'A step-heating spectrum. Box heights are 2-sigma per step and box widths '
            'the fraction of 39Ar released. The highlighted steps form the plateau.') + """

    <h3>Plateau</h3>
    <p>A run of consecutive steps is a plateau when it has at least a set number of
    steps, a set fraction of the <sup>39</sup>Ar, and a &chi;<sup>2</sup> probability
    of the weighted mean above a set value. Steps can also be included or excluded by
    hand. The plateau age is the weighted mean of its steps on the analytical errors,
    with the uncertainty given by the error mode.</p>
    <table>
      <tr><th>Error mode</th><th>2&sigma; uncertainty</th><th>Use when</th></tr>
      <tr><td>SEM &times; 2</td><td>2&nbsp;&middot;&nbsp;SEM</td><td>no scatter correction</td></tr>
      <tr><td>&times; &radic;MSWD above the limit (default)</td><td>2&nbsp;&middot;&nbsp;SEM&nbsp;&middot;&nbsp;&radic;MSWD
          when the MSWD exceeds the Wendt &amp; Carl limit, else 2&nbsp;&middot;&nbsp;SEM</td><td>most cases</td></tr>
      <tr><td>Student's t</td><td>SEM&nbsp;&middot;&nbsp;&radic;MSWD&nbsp;&middot;&nbsp;t(n&minus;1)
          when p is below the cutoff, else 2&nbsp;&middot;&nbsp;SEM</td><td>small n, below
          about 10 steps, where t(n&minus;1) is clearly larger than 2 (4.30 at
          n&nbsp;=&nbsp;3, 2.78 at n&nbsp;=&nbsp;5, 2.26 at n&nbsp;=&nbsp;10)</td></tr>
    </table>
    """ + E(r'\sigma_{\mathrm{int}} = \sqrt{\sigma_{\mathrm{ana}}^{2} + \left(\frac{\partial t}{\partial \ln J} f_{J}\right)^{2}} \qquad \sigma_{\mathrm{ext}} = \sqrt{\sigma_{\mathrm{int}}^{2} + \sum_{k} c_{k}^{2}}') + """
    <p>&sigma;<sub>ana</sub> is the uncertainty of the plateau age, and the
    c<sub>k</sub> are those of Section&nbsp;3.</p>

    <h3>Combining aliquots</h3>
    <p>Shared uncertainties do not average down when aliquots are combined, whereas
    independent ones decrease roughly as 1/&radic;n. Whether two aliquots share J
    depends on where they were irradiated.</p>
    <table>
      <tr><th>Aliquots from</th><th>Correlation of J</th><th>Does J average down?</th></tr>
      <tr><td>The same position</td><td>&rho; = 1</td><td>No</td></tr>
      <tr><td>Different positions, same disk</td><td>0 &lt; &rho; &lt; 1, from the flux model</td><td>Partly</td></tr>
      <tr><td>Different disks, one 3D fit</td><td>from the flux model</td><td>Partly</td></tr>
      <tr><td>Different disks fitted separately, or different irradiations</td><td>&rho; = 0</td><td>Yes</td></tr>
    </table>
    <p>The middle two rows need a loaded <code>.jflux</code> file, which gives the
    correlation between positions from the flux model. Without one, aliquots on the same
    disk are treated as sharing one J (&rho; = 1), and aliquots on different disks or in
    different irradiations as independent. This rule is the same in every tool. The route
    terms are common to everything on one
    calibration, so the external tier does not average down even across disks. The
    methods are those of Section&nbsp;4.</p>
    <ul>
      <li><b>WM + √MSWD</b> works on the plateau ages of the aliquots (n = aliquots, the
          default), or normalises all steps to one J and pools them in ratio space
          (n = steps).</li>
      <li><b>GLS + ML</b> works on the plateau ages with the full covariance and
          &omega;<sup>2</sup> (n = aliquots). It is the better choice when the aliquots
          disagree beyond analytical error and their errors differ widely.</li>
    </ul>
    <p>With n = aliquots both use the same J correlation and give the same age as pooling
    the aliquots in Age distribution.</p>

    <h3>Isochrons</h3>
    <p>An isochron solves for the composition of the trapped argon instead of assuming
    it is atmospheric. For the inverse isochron (x&nbsp;=&nbsp;<sup>39</sup>Ar<sub>K</sub>/<sup>40</sup>Ar,
    y&nbsp;=&nbsp;<sup>36</sup>Ar/<sup>40</sup>Ar) with intercept a and slope b,</p>
    """ + E(r'R = -\,b/a \qquad t = \frac{1}{\lambda}\ln\left(RJ + 1\right)') + """
    <p>and for the normal isochron (x&nbsp;=&nbsp;<sup>39</sup>Ar<sub>K</sub>/<sup>36</sup>Ar,
    y&nbsp;=&nbsp;<sup>40</sup>Ar/<sup>36</sup>Ar) with slope b,</p>
    """ + E(r't = \frac{1}{\lambda}\ln\left(bJ + 1\right)') + """
    <p>Both are fitted by York (1969) regression, which allows for correlated errors on
    both axes. The uncertainties are multiplied by &radic;MSWD when the MSWD exceeds 1.
    The trapped <sup>40</sup>Ar/<sup>36</sup>Ar is compared with the atmospheric value
    298.56 (Lee et&nbsp;al. 2006), which is drawn as a reference line.</p>
    """

    # ── 9 ────────────────────────────────────────────────────────────────
    prob = """
    <a name="prob"></a>
    <h2>9. Age distribution</h2>

    <p>For a population of single-crystal or single-aliquot ages. It shows the ranked
    ages with a probability density curve, a weighted mean of each sample, pooled ages of
    several samples, and mixture and minimum-age models.</p>

    """ + F('prob_plot.png',
            'Ranked ages with 1-sigma bars and the probability density curve. Excluded '
            'points are open markers, and the dashed curve includes them.') + """

    <h3>Probability density and weighted mean</h3>
    <p>Each accepted age adds a Gaussian of unit area with its own 1&sigma;
    uncertainty, whatever the display level, and the curve is their sum. It displays the
    data and is not a fitted model. The weighted mean and MSWD use the analytical
    errors.</p>
    """ + E(r'\bar{t} = \frac{\sum w_{i} t_{i}}{\sum w_{i}}, \quad \mathrm{MSWD} = \frac{1}{n-1}\sum w_{i}\left(t_{i} - \bar{t}\right)^{2}, \quad w_{i} = \frac{1}{\sigma_{\mathrm{ana},i}^{2}}') + """
    <p>Outliers can be removed by NMAD, by iterative 2&sigma; rejection (the cut is
    widened by &radic;MSWD while the MSWD exceeds 1), or by clicking points. A dashed curve shows the distribution with the excluded points added
    back.</p>

    <h3>Pooling samples</h3>
    <p>The sample ages are combined with the methods of Section&nbsp;4. The covariance
    between the ages of samples i and j is</p>
    """ + E(r'\Sigma_{ij} = a_{i}^{2}\,\delta_{ij} + \rho^{\mathrm{pos}}_{ij}\,p_{i} p_{j} + \sum_{k} c_{ik}\,c_{jk}') + \
    E(r'a = \sigma_{\mathrm{ana}} \qquad p = \sqrt{\sigma_{\mathrm{int}}^{2} - \sigma_{\mathrm{ana}}^{2}}') + """
    <ul>
      <li>a<sub>i</sub> is the analytical uncertainty of sample i, and
          &delta;<sub>ij</sub> is 1 when i&nbsp;=&nbsp;j and 0 otherwise, so this part is
          independent between samples.</li>
      <li>p<sub>i</sub> is the J part of the internal uncertainty.
          &rho;<sup>pos</sup><sub>ij</sub> is the correlation of J between the
          positions of the two samples. It is 1 for the same position, comes from a
          loaded .jflux file for different positions of one flux model, and is 0
          otherwise.</li>
      <li>c<sub>ik</sub> is the term of calibration input k (an R of the route, the
          calibration age or &lambda;) in the age of sample i (Section&nbsp;3). The sum
          runs over the inputs the two samples share, so it is zero between samples on
          different calibrations.</li>
    </ul>
    <p>The J part therefore averages down only between positions, and the calibration
    part never does. Samples on the same calibration are grouped automatically, and the
    report lists the groups.</p>

    <h3>Mixture view</h3>
    <p><b>Mixture</b> replaces the plot with a mixture model of the sample chosen above
    it. The samples are not changed, and the button switches back. The analyses
    z<sub>j</sub>&nbsp;&plusmn;&nbsp;s<sub>j</sub> (the age, or its logarithm on the log
    scale, with the analytical uncertainty) are modelled as k components with ages
    &mu;<sub>i</sub> in proportions &pi;<sub>i</sub> (Galbraith &amp; Green 1990;
    Galbraith &amp; Laslett 1993), fitted by maximising</p>
    """ + E(r'\ln L = \sum_{j=1}^{n} \ln \sum_{i=1}^{k} \pi_{i}\,\phi(z_{j};\,\mu_{i},\,s_{j})') + """
    <p>where &phi; is the normal density. The probability that analysis j belongs to
    component i is</p>
    """ + E(r'p_{ij} = \frac{\pi_{i}\,\phi(z_{j};\,\mu_{i},\,s_{j})}{\sum_{l=1}^{k} \pi_{l}\,\phi(z_{j};\,\mu_{l},\,s_{j})}') + """
    <p>and the maximum is found by iterating, from several starting points,</p>
    """ + E(r'\pi_{i} = \frac{1}{n}\sum_{j} p_{ij}, \qquad \mu_{i} = \frac{\sum_{j} p_{ij}\,z_{j}/s_{j}^{2}}{\sum_{j} p_{ij}/s_{j}^{2}}') + """
    <p>so each component age is a weighted mean of the analyses, each weighted by its
    probability of belonging. The uncertainties of &mu;<sub>i</sub> and
    &pi;<sub>i</sub> come from the information matrix (Galbraith 2005), as in IsoplotR's
    peakfit (Vermeesch 2018). The number of components is chosen with the Bayes
    Information Criterion (Schwarz 1978),</p>
    """ + E(r'\mathrm{BIC} = -2\ln L + (2k-1)\ln n') + """
    <p>as the fewest components within 2 of the lowest BIC. A smaller difference is not
    considered evidence (Kass &amp; Raftery 1995). Up to ten components are tried, since
    each old xenocryst can form its own. A component of at least two analyses has an
    MSWD, with each analysis weighted by p<sub>ij</sub>,</p>
    """ + E(r'\mathrm{MSWD}_{i} = \frac{\sum_{j} p_{ij}(z_{j} - \mu_{i})^{2}/s_{j}^{2}}{\sum_{j} p_{ij} - 1}') + """
    <p>and when it exceeds the Wendt &amp; Carl limit for &sum;<sub>j</sub>p<sub>ij</sub>
    analyses, the analytical uncertainty of the component is multiplied by &radic;MSWD.
    J and the calibration are then added to give the internal and external tiers, as
    for the weighted mean of the sample.</p>
    <p>By default the model uses every analysis of the sample and ignores outlier
    rejection and clicked exclusions. <b>Included only</b> uses the analyses included
    on the plot. The log scale, IsoplotR's default, leaves out ages of zero or
    below.</p>
    <p>The minimum age model (Galbraith et al. 1999) fits the youngest population on its
    own, as one discrete youngest age plus a truncated spread of older analyses. It
    assumes no analysis is too young, so a young outlier (argon loss) becomes the
    minimum, and old xenocrysts widen the spread. Run it on data without them.</p>
    <p>The BIC also splits a single population whose scatter exceeds its analytical
    errors, and the number of components it finds grows with the number of analyses. A
    split is therefore a reason to look at the data, not proof of two populations. The
    radial plot (Galbraith 1988) helps here. Each analysis is plotted at its precision,
    1/&sigma;, against its deviation from the weighted mean in units of &sigma;. Its age
    is read where the line from the origin through the point meets the arc, and each
    component is such a line. The &Delta;BIC panel shows how clearly the number of
    components was chosen.</p>
    """ + F('prob_mixture.png',
            'The Mixture view. Left, the ranked analyses and the probability curve split '
            'into components, each analysis coloured by its most likely component. '
            'Right, the radial plot and the BIC of each number of components.') + """
    """

    # ── 10 ───────────────────────────────────────────────────────────────
    calib = """
    <a name="calib"></a>
    <h2>10. Standards, routes and calibrations</h2>

    <p>The standards, calibrations, routes and decay constants of all tools are kept in
    one library, opened with Standards in the launcher. The published values ship with
    SimplAr. Edits are saved separately and can be reset.</p>
    <ul>
      <li>A <b>standard</b> holds its ratios R against other standards, with their
          sources, and the ages published for it.</li>
      <li>A <b>calibration</b> is a published age of one standard, with its uncertainty,
          the &lambda; it was calculated with and its basis (Section&nbsp;3).</li>
      <li>A <b>route</b> goes from a monitor through one or more R values in the library
          to a calibration, on a chosen &lambda;.</li>
    </ul>

    <h3>Routes</h3>
    <p>The route decides how the ages of a monitor are handled downstream. It sets the
    calibration and &lambda; the ages are on, and with them which terms enter the
    external tier and how &lambda; is treated (Section&nbsp;3). TCs on the A1
    calibration, for example, goes TCs&nbsp;&rarr;&nbsp;FCs&nbsp;&rarr;&nbsp;A1. Its
    external tier contains R<sub>TCs/FCs</sub>, R<sub>FCs/A1</sub>, the astronomical age
    of A1 and a small &lambda; term. GA1550 on its K&ndash;Ar age instead carries the
    GA1550 age and the full &lambda; term.</p>
    <p>Each monitor has a default route, which can be changed or built in Standards. The
    Calibration&hellip; button of each tool shows the route in use and can set another.
    A flux file keeps the monitor it was measured on, and a tool whose route starts at a
    different monitor warns. Each R in a route adds its own term to the external tier,
    shared by every age that uses it.</p>
    <p>The table lists the R values marked Use, all 1&sigma;. R<sub>S/FCs</sub> is the
    <sup>40</sup>Ar*/<sup>39</sup>Ar<sub>K</sub> of the standard divided by that of FCs,
    and R<sub>FCs/S</sub> its inverse, each given as its paper prints it.</p>
    """ + _monitor_table() + """

    <h3>Calibrations</h3>
    <p>The tools open on the route of the calibration marked as the default. As shipped
    this is Phillips et&nbsp;al. (2022), FCs 28.176&nbsp;&plusmn;&nbsp;0.0115&nbsp;Ma
    (1&sigma;) with the Min et&nbsp;al. (2000) &lambda;, calculated through the A1
    tephra (Section&nbsp;3).</p>
    """ + _calibration_table() + """
    <p>The astronomical calibrations were calculated with the Min et&nbsp;al. (2000)
    &lambda;, and the tools pair them with it. Another &lambda; can be chosen. An FCs
    age derived from a tephra then becomes the FCs age the tephra implies at that
    &lambda;, not the published value.</p>
    """ + _lambda_table() + """

    <h3>Carter et al. (2025)</h3>
    <p>Carter's FCs age is quoted with three different 1&sigma; uncertainties.</p>
    <table>
      <tr><th>1&sigma; (Ma)</th><th>Meaning</th></tr>
      <tr><td>0.0112</td><td>Posterior precision of the fit.</td></tr>
      <tr><td>0.017</td><td>Full external systematic, including the input-data systematics. Used as the external tier here.</td></tr>
      <tr><td>0.035</td><td>Total for dating, adding the R-value and neutron-fluence terms.</td></tr>
    </table>
    """

    # ── 11 ───────────────────────────────────────────────────────────────
    recalc = """
    <a name="recalc"></a>
    <h2>11. Recalculator and age combiner</h2>
    <p>The Recalculator recasts published ages to another calibration. An age t
    published on a monitor S, with the monitor age t<sub>S</sub> and the decay constant
    &lambda;<sub>0</sub> the authors used, fixes the ratio of the sample to that
    monitor, which depends on neither constant.</p>
    """ + E(r'R_{u/S} = \frac{e^{\lambda_{0} t} - 1}{e^{\lambda_{0} t_{S}} - 1}') + """
    <p>On the target calibration the ingrowth of the monitor comes from its route
    (Section&nbsp;2), and the recalculated age is (Renne et&nbsp;al. 1998; Mercer and
    Hodges 2016)</p>
    """ + E(r"t' = \frac{1}{\lambda}\ln\left[1 + R_{u/S}\,\prod_{i} R_{i}\left(e^{\lambda t_{C}} - 1\right)\right]") + """
    <p>The analytical and internal errors are multiplied by dt&prime;/dt. The external
    error is that of the target route (Section&nbsp;3). Recalculating onto the
    calibration the ages were published on leaves them unchanged and gives the external
    error that calibration implies, for example the astronomical terms of Phillips
    et&nbsp;al. (2022) for ages published with the internal error only. Where only the
    analytical error is known, the external error contains no J term.</p>
    <p>An age given with only one of the analytical and internal errors gets the other
    from the J error of its position in a loaded .jflux, or from a J or R standard typed
    with its error. The internal error is the analytical one with the J term added, and
    the analytical error the internal one with it removed.</p>
    <p>The Combine page recalculates ages published on different calibrations to one
    calibration and combines them as in Section&nbsp;4. Ages assigned to positions of a
    loaded .jflux take the J correlation of the flux model. Without one, ages on the same
    disk share their J error, as in the other tools. The calibration terms are shared by all ages, and the R of a monitor by every age dated
    against it.</p>
    <p class="back"><a href="#top">Back to top</a></p>
    """

    # ── 12 ───────────────────────────────────────────────────────────────
    refs = """
    <a name="refs"></a>
    <h2>12. References</h2>
    <ul style="font-size:13px;">
      <li>Carter et al. (2025): jointly calibrated <sup>40</sup>K decay constant
          and Fish Canyon age. <i>Geochim. Cosmochim. Acta</i> 397, 149&ndash;163.</li>
      <li>Galbraith (1988): the radial plot. <i>Technometrics</i> 30,
          271&ndash;281.</li>
      <li>Galbraith (2005): <i>Statistics for Fission Track Analysis</i>.
          Chapman &amp; Hall/CRC.</li>
      <li>Galbraith &amp; Green (1990): component ages in a finite mixture.
          <i>Nucl. Tracks Radiat. Meas.</i> 17, 197&ndash;206.</li>
      <li>Galbraith &amp; Laslett (1993): statistical models for mixed ages.
          <i>Nucl. Tracks Radiat. Meas.</i> 21, 459&ndash;470.</li>
      <li>Galbraith et al. (1999): the minimum age model. <i>Archaeometry</i> 41,
          339&ndash;364.</li>
      <li>Kass &amp; Raftery (1995): Bayes factors. <i>J. Am. Stat. Assoc.</i> 90,
          773&ndash;795.</li>
      <li>Kuiper et al. (2008): Fish Canyon age 28.201 Ma.
          <i>Science</i> 320, 500&ndash;504.</li>
      <li>Lee et al. (2006): isotopic composition of atmospheric argon.
          <i>Geochim. Cosmochim. Acta</i> 70, 4507&ndash;4512.</li>
      <li>Mercer &amp; Hodges (2016): ArAR, recalculating K/Ar and
          <sup>40</sup>Ar/<sup>39</sup>Ar dates. <i>Chem. Geol.</i> 440, 148&ndash;163.</li>
      <li>Min et al. (2000): revised <sup>40</sup>K decay constant.
          <i>Geochim. Cosmochim. Acta</i> 64, 73&ndash;98.</li>
      <li>Niespolo et al. (2017): intercalibration and the R-value age form.
          <i>Quat. Geochronol.</i> 39, 205&ndash;213.</li>
      <li>Phillips et al. (2022): astronomically tuned Fish Canyon age.
          <i>Chem. Geol.</i> 597, 120815.</li>
      <li>Renne et al. (1998, 2010): intercalibration and joint calibration.
          <i>Chem. Geol.</i> 145, 117&ndash;152; <i>GCA</i> 74, 5349&ndash;5367.</li>
      <li>Renne et al. (2011): response to Schwarz et al., revised joint calibration.
          <i>GCA</i> 75, 5097&ndash;5100.</li>
      <li>Rivera et al. (2011): astronomically tuned Fish Canyon age.
          <i>Earth Planet. Sci. Lett.</i> 311, 420&ndash;426.</li>
      <li>Schwarz (1978): the Bayes Information Criterion. <i>Ann. Stat.</i> 6,
          461&ndash;464.</li>
      <li>Schaen et al. (2021): interpretation and reporting standards.
          <i>GSA Bull.</i> 133, 461&ndash;487.</li>
      <li>Vermeesch (2015): revised error propagation including covariances.
          <i>GCA</i> 171, 325&ndash;337.</li>
      <li>Vermeesch (2018): IsoplotR. <i>Geosci. Front.</i> 9, 1479&ndash;1493.</li>
      <li>Viechtbauer &amp; Cheung (2010): outlier and influence diagnostics for
          meta-analysis. <i>Res. Synth. Methods</i> 1, 112&ndash;125.</li>
      <li>Wendt &amp; Carl (1991): MSWD confidence limits.
          <i>Chem. Geol.</i> 86, 275&ndash;285.</li>
      <li>York (1969): least-squares fit with correlated errors on both axes.
          <i>Earth Planet. Sci. Lett.</i> 5, 320&ndash;324.</li>
    </ul>
    <p class="back"><a href="#top">Back to top</a></p>
    """

    return (style + toc + workflow + idea + tiers + stats
            + converter + wm + jcalc + plateau + prob + calib + recalc + refs)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SimplAr")
    from arar_math import no_wheel_on_dropdowns
    no_wheel_on_dropdowns(app)
    launcher = ToolLauncher()
    launcher.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()