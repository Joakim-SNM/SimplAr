"""The SimplAr 'About: Methods & Reference' document.

Kept in its own module so the launcher stays short. build_about_html(base_dir)
returns a single self-contained HTML string for a QTextBrowser: a clickable
table of contents at the top, then the sections.

Equations are typeset with matplotlib's mathtext (classic LaTeX look), rendered
once to transparent PNGs in the user's cache folder (data_folders.cache_dir)
and cached by content hash, then referenced by bare filename. The launcher adds
that folder and docs/img to the browser's search paths so both equations and
screenshots resolve. If matplotlib
is unavailable the equation falls back to its raw text so the document still
opens.

Screenshots are shown from docs/img/<name>.png when that file exists and are
left out otherwise.
"""

import os
import hashlib
import struct

import data_folders as dfold


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
      .aside { color: #4a4034; }
      .back  { font-size: 11px; }
      .back a { color: #9a875f; text-decoration: none; }
    </style>
    """

    toc = """
    <a name="top"></a>
    <h1>SimplAr: Methods &amp; Reference</h1>
    <p class="lede">A guide to the five tools: what each one is for, the order they
    are meant to be used in, and the mathematics behind them, set out first in plain
    language and then written down.</p>

    <div class="toc">
      <b>Contents</b>
      <ol>
        <li><a href="#stats">Two ways of combining ages</a></li>
        <li><a href="#idea">Working in R-values</a></li>
        <li><a href="#tiers">The three tiers of uncertainty</a></li>
        <li><a href="#workflow">The workflow</a></li>
        <li><a href="#converter">Data converter</a></li>
        <li><a href="#wm">Flux monitors</a></li>
        <li><a href="#jcalc">Flux regression</a></li>
        <li><a href="#plateau">Plateau &amp; isochron</a></li>
        <li><a href="#prob">Age distribution</a></li>
        <li><a href="#calib">Calibrations and constants</a></li>
        <li><a href="#refs">References</a></li>
      </ol>
    </div>
    """

    # ── 1 ────────────────────────────────────────────────────────────────
    stats = """
    <a name="stats"></a>
    <h2>1. Two ways of combining ages</h2>

    <p>The field agrees on the equations that turn argon measurements into ages, but
    it has never settled on a single way to combine several ages into one. These
    tools offer two, labelled <b>WM + √MSWD</b> and <b>GLS + ML</b>. It is worth being
    clear about what actually separates them, because it is narrower than the two
    names suggest.</p>

    <p>Both build the same covariance structure. Every age carries three tiers:
    analytical, then plus the neutron-flux (J) uncertainty, then plus the monitor-age
    and decay-constant uncertainty. The parts that two ages share (J for aliquots that
    sat in the same irradiation position; the monitor age and λ for aliquots on the
    same calibration) are correlated, and therefore do not average down when the ages
    are combined. Both methods encode that identically. The covariance handling is not
    what distinguishes them.</p>

    <p>What differs is only two things. The first is the mean. <b>WM + √MSWD</b> takes the
    ordinary inverse-variance weighted mean on the analytical errors. <b>GLS + ML</b>
    (Vermeesch 2015, Section 13) takes the generalised-least-squares mean on the full
    covariance, including the overdispersion term ω² described next. When the ages are
    overdispersed, ω² is added to every age, which evens out the weights, so that a few
    very precise analyses no longer decide the mean on their own. Without
    overdispersion and with a single shared J the two means are essentially identical;
    partial correlations (aliquots in different holes of one disk, with the correlation
    read from a loaded .jflux file) add a smaller difference.</p>

    <p>The second is overdispersion, the treatment of ages that scatter more than
    their stated errors allow. <b>WM + √MSWD</b> multiplies the analytical tier by
    √MSWD when the MSWD exceeds the Wendt &amp; Carl (1991) 95% limit for n, the
    same trigger in every tool. <b>GLS + ML</b> instead estimates an additive random-effects variance
    ω² by maximum likelihood and adds it to each age (the IsoplotR / Ar-Ar_Redux
    random.effects = TRUE model). ω² is fitted on the full covariance, analytical
    plus J plus calibration, so scatter that independent J errors already explain
    (ages from different holes or irradiations) is not counted a second time, and
    all three tiers are the variance of the one GLS mean that is reported.</p>

    <p>Which to use depends on the data. When the analytical errors are similar, or the
    ages are not overdispersed, the two give practically the same age and uncertainty.
    When the ages are overdispersed and their analytical errors differ widely, as they
    often do for single crystals of variable K content, they do not. In simulations with
    a tenfold range of analytical errors and a true overdispersion of one to two times
    the typical analytical error, the WM + √MSWD 95% interval contained the true age in
    about 70&ndash;80% of cases, against about 80&ndash;93% for GLS + ML, whose mean was
    also closer to the true age. Both fall short of 95% for few ages, ML more so,
    because a variance estimated from five or ten values is itself uncertain.</p>

    <p>One further choice appears only when stacking several step-heating aliquots.
    <b>GLS + ML</b> reduces each aliquot to its plateau age first and combines those
    (n = aliquots); <b>WM + √MSWD</b> does the same by default, or can pool every
    heating step in ratio space instead (n = steps). When the aliquots agree this
    makes no difference. When they
    disagree beyond analytical error, the aliquot is the correct unit: the steps within
    one aliquot all share that aliquot's single geological offset, so pooling them as
    independent evidence about the sample age understates the uncertainty, and does so
    without limit as more steps are added. The aliquot-level combination instead
    reports the between-aliquot standard error, which is the honest number.</p>

    <p>A note on scope. This is Vermeesch's combination step (his Eqs. 91 to 94), not
    his full data reduction. Unlike Ar-Ar_Redux, the tools work from already-reduced
    ages, so the covariance is built only from the J and monitor/λ terms that can be
    reconstructed downstream, not from the raw-beam correlations (shared blanks,
    detector calibration, mass fractionation, interference corrections) that require
    the original isotope data.</p>
    """

    # ── 2 ────────────────────────────────────────────────────────────────
    idea = """
    <a name="idea"></a>
    <h2>2. Working in R-values</h2>

    <p>A mass spectrometer does not measure an age. It measures how much radiogenic
    argon a sample produced relative to a monitor mineral, Fish Canyon sanidine, that
    was irradiated alongside it. That relative quantity is the
    R-value, and the age follows from it by arithmetic.</p>

    <p>The reason this matters is worth spelling out. Suppose you weigh yourself on a
    bathroom scale that reads a few percent high or low. Record only your own weight
    and you inherit the scale's error. Weigh a known reference on the same scale as
    well, and report your weight as a multiple of the reference, and the error
    cancels, because it inflates you and the reference by the same factor. The decay
    constant plays the part of the unreliable scale, and R is your weight expressed
    relative to the reference. This is why the sizeable uncertainty on the
    <sup>40</sup>K decay constant does not carry through into a comparably large
    uncertainty when one Ar/Ar age is compared with another.</p>

    <p>For an analysis with ratio R, irradiation parameter J, and decay constant
    &lambda;, the age is</p>
    """ + E(r't = \frac{1}{\lambda}\,\ln\left(1 + J\,R\right)') + """
    <p>and J is itself only the monitor's calibration written as a number,</p>
    """ + E(r'J = \frac{e^{\lambda t_{\mathrm{FCs}}} - 1}{R_{\mathrm{std}}}') + """
    <p>Substituting one into the other removes J entirely and leaves the age in terms
    of the ratio and the monitor age alone:</p>
    """ + E(r't = \frac{1}{\lambda}\,\ln\left[\,1 + R\left(e^{\lambda t_{\mathrm{FCs}}} - 1\right)\right]') + """
    <p>Because &lambda;t is small (about 0.03 for a 55&nbsp;Ma sample), this is very
    nearly t&nbsp;&asymp;&nbsp;R&thinsp;&middot;&thinsp;t<sub>FCs</sub>. The age is
    essentially the ratio multiplied by the monitor's age, with the decay constant
    almost absent. That near-cancellation is why the uncertainty on &lambda;, whatever
    its size, does not carry through into a comparably large uncertainty on an Ar/Ar
    age, so long as the comparison stays within the
    <sup>40</sup>K&ndash;<sup>40</sup>Ar system. It returns in full only when an
    Ar/Ar age is compared with a different decay system, such as U&ndash;Pb;
    Section&nbsp;3 sets out that case and the term it adds.</p>
    """

    # ── 3 ────────────────────────────────────────────────────────────────
    tiers = """
    <a name="tiers"></a>
    <h2>3. The three tiers of uncertainty</h2>

    <p>Every age these tools report carries three uncertainties, each larger than the
    last. They describe the same age; the number grows only because each tier folds
    in a further source of scatter. Which one to quote depends on what the age is
    being compared against.</p>

    <table>
      <tr><th>Tier</th><th>What it adds</th><th>Quote it against&hellip;</th></tr>
      <tr><td><b>Analytical</b></td>
          <td>The mass spectrometer alone: the R measurement, blanks, detector.</td>
          <td>Aliquots from the same run or the same irradiation position.</td></tr>
      <tr><td><b>Internal</b></td>
          <td>Plus the neutron-flux uncertainty J at your position. This is the number
              ArArCALC prints as the age error.</td>
          <td>Other Ar/Ar ages tied to the same monitor and &lambda;.</td></tr>
      <tr><td><b>External</b></td>
          <td>Plus the monitor age and, formally, the decay constant. These are
              shared systematics that repetition cannot reduce.</td>
          <td>Ar/Ar ages from other laboratories or monitors, and the astronomical
              timescale.</td></tr>
    </table>

    <p>For a single age the tiers are quadrature sums:</p>
    """ + E(r'\sigma_{\mathrm{int}}^{2} = \sigma_{\mathrm{ana}}^{2} + \left(\frac{\partial t}{\partial \ln J}\,f_{J}\right)^{2} \qquad \frac{\partial t}{\partial \ln J} = \frac{1 - e^{-\lambda t}}{\lambda}') + \
    E(r'\sigma_{\mathrm{ext}}^{2} = \sigma_{\mathrm{int}}^{2} + \sum_{k} c_{k}^{2}') + """
    <p>where f<sub>J</sub> is the relative uncertainty on J (the monitor
    measurement), and the c<sub>k</sub> are the calibration terms described below:
    for an astronomically calibrated monitor, the R<sub>FCs/A1</sub> measurement,
    the anchor tephra's orbital age and &lambda;. The derivative
    &part;t/&part;ln&nbsp;J is slightly smaller than t (by about 1.5 percent at
    55&nbsp;Ma); it is the same for a fractional change in any ratio in the age
    equation.</p>

    <p>The external tier deserves more care, because there is a right and a wrong way
    to compute it. The wrong way treats J as a fixed constant and then varies
    &lambda; on its own. But J is not independent of &lambda;: it was obtained from
    the monitor through
    J&nbsp;=&nbsp;(e<sup>&lambda;t<sub>FCs</sub></sup>&nbsp;&minus;&nbsp;1)/R<sub>std</sub>,
    so a change in &lambda; ought to move J with it. Freezing J cuts that link and
    drops the full decay-constant uncertainty onto the age, which is where the
    familiar statement that decay-constant uncertainty limits Ar/Ar to about two
    percent comes from. That two percent belongs to the Min et al. (2000) constant in
    particular; the optimized constants of Renne et al. (2011) and Carter et al.
    (2025) carry closer to a tenth of it. And the figure applies only across decay
    systems in the first place. For one Ar/Ar age against another, letting J move
    with &lambda; makes the two &lambda; terms very nearly cancel. Writing X for
    e<sup>&lambda;t</sup>,</p>
    """ + E(r'X = R\left(e^{\lambda t_{\mathrm{FCs}}} - 1\right) + 1 = e^{\lambda t}') + \
    E(r'\frac{\partial t}{\partial \lambda} = -\frac{\ln X}{\lambda^{2}} + \frac{R\,t_{\mathrm{FCs}}\,e^{\lambda t_{\mathrm{FCs}}}}{\lambda X}') + \
    E(r'\frac{\partial t}{\partial t_{\mathrm{FCs}}} = \frac{R\,e^{\lambda t_{\mathrm{FCs}}}}{X} \qquad \frac{\partial t}{\partial R} = \frac{e^{\lambda t_{\mathrm{FCs}}} - 1}{\lambda X}') + \
    E(r'\sigma_{t}^{2} = \left(\frac{\partial t}{\partial \lambda}\sigma_{\lambda}\right)^{2} + \left(\frac{\partial t}{\partial t_{\mathrm{FCs}}}\sigma_{t_{\mathrm{FCs}}}\right)^{2} + \left(\frac{\partial t}{\partial R}\sigma_{R}\right)^{2} + 2\,\frac{\partial t}{\partial \lambda}\frac{\partial t}{\partial t_{\mathrm{FCs}}}\,\mathrm{cov}(\lambda, t_{\mathrm{FCs}})') + """
    <p>The covariance term on the last line is the only difference between the two
    families of calibration the tools recognise. When the monitor age was fixed
    astronomically, by counting orbital cycles independently of &lambda; (Kuiper
    2008, Rivera 2011, Phillips 2022), the covariance is zero. For the joint Bayesian
    calibration of Carter et al. (2025), where &lambda; and the monitor age were
    determined together, it is not; the tools carry Carter's posterior value, which
    proves numerically negligible and reproduces their Table&nbsp;9 to within
    4&times;10<sup>&minus;5</sup>.</p>

    <h3>Astronomically calibrated monitors: the anchored chain</h3>
    <p>For Kuiper (2008), Rivera (2011) and Phillips (2022) the tools do not
    propagate the external from the Fish Canyon age and its error. That age was
    itself calculated from the astronomical age of an anchor tephra with the same
    equation and the Min et&nbsp;al. (2000) &lambda;, so it depends on &lambda;, and
    Renne et&nbsp;al. (1998, p.&nbsp;120) point out that their eq.&nbsp;6 is not valid
    on a standard age of that kind. The age is instead written on their two-step
    chain (eq.&nbsp;5), in which every input is measured independently:</p>
    """ + E(r't = \frac{1}{\lambda}\ln\left[1 + R_{u/\mathrm{FCs}}\,R_{\mathrm{FCs}/A}\left(e^{\lambda t_{A}} - 1\right)\right]') + """
    <p>R<sub>u/FCs</sub> is the sample-to-monitor ratio from your own irradiation;
    R<sub>FCs/A</sub> and t<sub>A</sub> are the calibration paper's intercalibration
    ratio and the anchor's orbital age. Eq.&nbsp;6 then applies to these inputs, with
    the partial derivatives given by Kuiper et&nbsp;al. (2008, SOM). Writing
    P&nbsp;=&nbsp;R<sub>u/FCs</sub>R<sub>FCs/A</sub>,</p>
    """ + E(r'\frac{\partial t}{\partial \ln R} = \frac{1 - e^{-\lambda t}}{\lambda} \qquad \frac{\partial t}{\partial t_{A}} = \frac{P\,e^{\lambda t_{A}}}{e^{\lambda t}} \qquad \frac{\partial t}{\partial \lambda} = -\frac{t}{\lambda} + \frac{P\,t_{A}\,e^{\lambda t_{A}}}{\lambda\,e^{\lambda t}}') + \
    E(r'\sigma_{\mathrm{ext}}^{2} = \sigma_{\mathrm{int}}^{2} + \left(\frac{\partial t}{\partial \ln R}\,\frac{\sigma_{R_{\mathrm{FCs}/A}}}{R_{\mathrm{FCs}/A}}\right)^{2} + \left(\frac{\partial t}{\partial t_{A}}\,\sigma_{t_{A}}\right)^{2} + \left(\frac{\partial t}{\partial \lambda}\,\sigma_{\lambda}\right)^{2}') + """
    <p>The two terms of &part;t/&part;&lambda; nearly cancel, which is why &lambda;
    contributes little. The anchors used (1&sigma;):</p>
    <table>
      <tr><th>Calibration</th><th>Anchor</th><th>t<sub>A</sub> (Ma)</th><th>R<sub>FCs/A</sub></th></tr>
      <tr><td>Kuiper et al. (2008)</td><td>Melilla tephras, composite</td><td>6.500 &plusmn; 0.005</td><td>4.3664 &plusmn; 0.0009</td></tr>
      <tr><td>Rivera et al. (2011)</td><td>A1 tephra, Faneromeni</td><td>6.943 &plusmn; 0.0025</td><td>4.0813 &plusmn; 0.0013</td></tr>
      <tr><td>Phillips et al. (2022)</td><td>A1 tephra, Faneromeni</td><td>6.943 &plusmn; 0.0025</td><td>4.0819 &plusmn; 0.0007</td></tr>
    </table>
    <p>R<sub>FCs/A</sub> is used at the value that returns each paper's published
    Fish Canyon age exactly at the Min et&nbsp;al. (2000) &lambda;; the printed values
    differ from it in the fifth figure through rounding. At that &lambda; the chain
    and the single-step equation give identical ages, and only the external differs.
    If the monitor-age error is edited away from the preset, the tools fall back to
    the single-step form above. When ages are combined, each of the three terms is
    shared by all of them and is carried through the weighted mean one term at a
    time, so none of them averages down.</p>

    <h3>When the full decay-constant term applies</h3>
    <p>Whether the decay-constant uncertainty enters an age at full strength depends
    on what the monitor age is anchored to, and the difference is large. Holding the
    measured ratio fixed against an astronomical anchor, a one percent change in
    &lambda; moves a 55&nbsp;Ma age by about 0.007 percent. Freezing J as a fixed
    number and changing &lambda; by that same one percent moves the age by very nearly
    one percent. Which case you are in is set by the derivative:</p>
    """ + E(r'\left.\frac{\partial t}{\partial \lambda}\right|_{R,\;t_{\mathrm{std}}} \approx 0 \qquad\qquad \left.\frac{\partial t}{\partial \lambda}\right|_{J} = -\frac{t}{\lambda} \;\Rightarrow\; \left(\frac{\sigma_{t}}{t}\right)_{\lambda} = \frac{\sigma_{\lambda}}{\lambda}') + """
    <p>The term does not apply, and must not be added, when the monitor is
    astronomically calibrated (Kuiper 2008, Rivera 2011, Phillips 2022). There the
    monitor age is a count of orbital cycles, independent of &lambda;, so the unknown
    ties through to the orbital timescale and &lambda; cancels to about a hundredth of
    a percent. This holds for every comparison, U&ndash;Pb included: setting such an
    age against U&ndash;Pb tests the orbital timescale against the uranium-decay
    timescale, and the <sup>40</sup>K constant is a bystander. The same applies to the
    jointly optimised calibrations (Renne 2010/2011, Carter 2025), where &lambda; and
    the monitor age were fit together against U&ndash;Pb. There &lambda; already sits
    in the external tier through its covariance with the monitor age, at roughly a
    tenth to a third of a percent, and adding a separate term would count it twice.</p>
    <p>The term applies only when the age rests on the physical <sup>40</sup>K
    radioisotopic scale: a physically measured &lambda; such as Min et al. (2000),
    with a monitor age tied to it by K&ndash;Ar rather than by astronomy or by
    U&ndash;Pb intercalibration, and you are deliberately treating the
    <sup>40</sup>Ar/<sup>39</sup>Ar age as an independent decay clock to set against a
    different decay system. Only then is the decay-constant term carried in full,</p>
    """ + E(r'\sigma_{\mathrm{full}}^{2} = \sigma_{\mathrm{ext}}^{2} + \left(t\,\frac{\sigma_{\lambda}}{\lambda}\right)^{2}') + """
    <p>with a size that follows &sigma;<sub>&lambda;</sub>/&lambda;, about two percent
    for Min et al. (2000) and smaller for the later constants. The test is mechanical:
    would the monitor age change if &lambda; changed? If it would not, &lambda; has
    cancelled and the term does not belong. If it would, it does.</p>
    """

    # ── 4 ────────────────────────────────────────────────────────────────
    workflow = """
    <a name="workflow"></a>
    <h2>4. The workflow</h2>

    <p>The five tools are independent and any one opens on its own, but they were
    built to be used in a particular order, and following it makes each tool's inputs
    fall into place.</p>

    <p>Everything begins with an export. Data leaves ArArCALC as a summary workbook or
    comes off the mass spectrometer as a raw one, and neither is in the form the
    plotting tools expect. The <a href="#converter">Data converter</a> reads the
    spreadsheet, lets you confirm which column is which, extracts the calibration
    (the J values, the monitor ratio, the irradiation label), and writes a compact
    <code>.arar</code> session file that the other tools read.</p>

    <p>The next task is to pin down the flux, since an Ar/Ar age is only as good as
    the value of J at the exact spot the unknown occupied. J is established in two
    steps. The <a href="#wm">Flux monitors</a> tool takes the many measurements of
    the Fish Canyon monitor from one hole and reduces them to a single value of R,
    with an uncertainty that separates genuine grain-to-grain scatter from analytical
    noise. The <a href="#jcalc">Flux regression</a> tool then fits a smooth surface
    through those per-hole values across the whole irradiation disk, so that J, and a
    realistic uncertainty on it, can be predicted anywhere a sample sat, including
    between the monitor holes.</p>

    <p>With J in hand the unknowns can be dated. The <a href="#plateau">Plateau
    &amp; isochron</a> tool turns a step-heating experiment into an age: it reads the
    gas-release spectrum, identifies the flat interval over which the mineral released
    argon cleanly, and computes a weighted-mean age across those steps, with isochrons
    as an independent check. When several aliquots or several samples are involved, the
    <a href="#prob">Age distribution</a> tool ranks them, draws the summed-probability
    curve, and, on request, pools them with the correlations handled properly. This is
    the stage at which the choice between the two statistical conventions of
    Section&nbsp;1 becomes a visible setting.</p>

    <p>The order is converter, flux monitors, flux regression, then plateau &amp;
    isochron and age distribution. The sections that follow take each tool in turn:
    what it does, then the mathematics it does it with.</p>
    """

    # ── 5 ────────────────────────────────────────────────────────────────
    converter = """
    <a name="converter"></a>
    <h2>5. Data converter</h2>

    <h3>What it does</h3>
    <p>The converter is the tool that makes the others possible. It reads an ArArCALC
    summary export or a raw mass-spectrometer workbook and writes the
    <code>.arar</code> session file that the Plateau &amp; isochron and Age distribution
    tools open.
    Two input layouts are recognised: the tidy per-step summary that ArArCALC
    produces, and the wider raw sheet that comes directly off the instrument. A dialog
    lets you choose the layout and confirm the column mapping before anything is
    written, so that the field the tool reads as %<sup>39</sup>Ar really is the column
    you intend.</p>

    <p>Beyond reshaping the data, the converter records the calibration. It stores the
    J value and its error, the monitor ratio R<sub>std</sub>, the reference monitor
    age, and the irradiation label in the file's configuration block, so the other
    tools know which constants an age was originally built on and can recast it to
    another calibration.</p>

    """ + F('converter_mapping.png',
            'The format and column-mapping dialog. Choose the export layout, then '
            'confirm each internal field points at the right spreadsheet column '
            'before the .arar file is written.') + """

    <h3>The mathematics</h3>
    <p>There is deliberately very little. The converter does not recompute ages; it
    carries across whatever the export contains and keeps the calibration attached to
    it. The one relationship it holds consistent is the identity between J and the
    monitor ratio, so that a calibration supplied in either form yields the same age:</p>
    """ + E(r'J = \frac{e^{\lambda t_{\mathrm{FCs}}} - 1}{R_{\mathrm{std}}}') + """
    <p>The monitor ages it offers are the calibration presets of Section&nbsp;10, each
    stored with a 1&sigma; error and a note of whether that error already includes the
    decay constant, the same bookkeeping described in Section&nbsp;3.</p>
    """

    # ── 6 ────────────────────────────────────────────────────────────────
    wm = """
    <a name="wm"></a>
    <h2>6. Flux monitors</h2>

    <h3>What it does</h3>
    <p>At a single position in the irradiation disk the Fish Canyon monitor was
    measured many times, each measurement with its own precision. This tool reduces
    those measurements to one best estimate of R for that position, and it separates
    the scatter that reflects real variation (flux gradients and grain-to-grain
    heterogeneity) from the scatter expected on analytical grounds alone. It then
    converts the mean R into the J value the regression needs.</p>
    <p>It reads the same .arar, .ararpt and .ararstack files as the Plateau and
    Age distribution tools (R and its error from every step, at 1&sigma;), grouped by
    irradiation position, as well as any Excel/CSV table or pasted values. With the
    ratio box unticked it is a plain weighted-mean calculator for any quantity.</p>

    """ + F('wm_main.png',
            'The individual acquisitions with 1-sigma error bars, the mean and its '
            'uncertainty band, and the MSWD and p-value reported in the title.') + """

    <h3>The mathematics</h3>
    <p>The estimate is an inverse-variance weighted mean: each measurement is weighted
    by the reciprocal of its variance, so the noisier ones count for less.</p>
    """ + E(r'w_{i} = \frac{1}{\sigma_{i}^{2}} \qquad \bar{x} = \frac{\sum w_{i} x_{i}}{\sum w_{i}} \qquad \mathrm{SE} = \sqrt{\frac{1}{\sum w_{i}}}') + """
    <p>Whether the scatter is real is judged by the MSWD, which compares the observed
    spread with the spread the analytical errors predict.</p>
    """ + E(r'\mathrm{MSWD} = \frac{1}{n-1}\sum w_{i}\left(x_{i} - \bar{x}\right)^{2} \qquad p = P\left(\chi^{2}_{n-1} > \mathrm{MSWD}\,(n-1)\right)') + """
    <p>A value near one means the scatter is fully accounted for by the analytical
    errors. A value above one, with a low probability, points to additional variance,
    from real differences in flux or potassium distribution across the grains. A
    value well below one suggests the analytical errors were overstated.</p>

    <p>When the excess scatter is real, the ordinary standard error understates the
    uncertainty. Rather than scale it by the square root of the MSWD, this tool
    estimates the additional population variance &tau;<sup>2</sup> directly, by
    restricted maximum likelihood (REML): &tau;<sup>2</sup> minimises</p>
    """ + E(r'\sum_{i}\ln\left(\sigma_{i}^{2} + \tau^{2}\right) + \ln W^{*} + \sum_{i} w_{i}^{*}\left(x_{i} - \hat{\mu}^{*}\right)^{2}, \qquad w_{i}^{*} = \frac{1}{\sigma_{i}^{2} + \tau^{2}}, \quad W^{*} = \sum_{i} w_{i}^{*}') + """
    <p>The ln&nbsp;W* term is what distinguishes REML from ordinary maximum
    likelihood: it allows for the mean being estimated from the same data, so
    &tau;<sup>2</sup> is not biased low for the small numbers of monitors typical of
    one irradiation position.</p>
    <p>When &tau;<sup>2</sup> is zero the result reduces to the ordinary weighted mean;
    when it is not, &tau; is reported as the true grain-to-grain spread, in the units
    of R. This is the random-effects treatment of the second convention in
    Section&nbsp;1, applied here at the monitor-calibration stage.</p>

    <p>The standard error is the plug-in value 1/&radic;W*, which does not include
    the uncertainty of &tau;<sup>2</sup> itself. Simulated with excess scatter as
    large as the analytical errors, &plusmn;1.96&nbsp;SE covers the true mean 90&ndash;92%
    of the time for 5&ndash;20 monitors (95&ndash;96% without excess scatter); the
    weighted mean with SE&nbsp;&times;&nbsp;&radic;MSWD covers 85&ndash;89%. With REML
    unticked the tool reports that weighted mean, its SE multiplied by &radic;MSWD
    when the MSWD exceeds the Wendt &amp; Carl (1991) 95% limit
    1&nbsp;+&nbsp;2&radic;(2/(n&minus;1)), the rule the Plateau &amp; isochron tool uses.</p>

    <p>Converting the mean R to J uses the monitor age and the decay constant:</p>
    """ + E(r'J = \frac{e^{\lambda t} - 1}{R} \qquad \frac{\sigma_{J}}{J} = \frac{\sigma_{R}}{R}') + """
    <p>The error on J is the monitor measurement only. The FCs age and &lambda; are
    systematic, shared by every J and every age on that calibration; folded into
    each J they would be averaged down wherever J values are combined, so they are
    added once, in the external tier of the ages (Section&nbsp;3). J is computed
    from the reported R, not averaged separately: the J view shows the same
    accepted and rejected points, MSWD and p as the R analysis.</p>

    <h3>Crystals: fusion and step heating</h3>
    <p>A crystal is identified by a grain-ID column (a blank cell continues the
    crystal above) or, in an .arar file, by the run-ID prefix (11776-13A &rarr;
    11776-13) or the laser-power sequence in run order: a step at the same or a
    lower power than the one before starts the next crystal. A crystal measured in
    one step is a fusion; one with several steps was step-heated. The points that
    are averaged can be every step, one point per crystal, the single-step
    crystals only, or the last step of each crystal. One point per crystal is the
    crystal's total gas, the &sup3;&sup9;Ar-weighted ratio</p>
    """ + E(r'R_{\mathrm{TG}} = \sum_{i} f_{i} R_{i}, \quad f_{i} = \frac{{}^{39}\mathrm{Ar}_{i}}{\sum {}^{39}\mathrm{Ar}} \qquad \sigma_{\mathrm{TG}} = \sqrt{\sum_{i} f_{i}^{2}\sigma_{i}^{2}}') + """
    <p>which is what a single fusion of the crystal would have measured, so
    step-heated and fused crystals can be averaged together. Without &sup3;&sup9;Ar
    the inverse-variance mean of the steps is used and labelled as such. For every
    step-heated crystal the tool also reports the MSWD of its steps, and the steps
    are grouped by laser power (or step number) to compare a weighted mean per
    group and test whether the groups agree.</p>

    <p>For outlier handling the defaults are a relative-error cut, which removes
    analyses with poor counting statistics, and rejection by the normalised median
    absolute deviation, which flags points lying more than k times the NMAD from
    the median:</p>
    """ + E(r'\mathrm{NMAD} = 1.4826 \times \mathrm{median}\,\left| x_{i} - \mathrm{median}\right|') + """
    <p>Being built on the median, the NMAD is not masked by several outliers at
    once. It uses the values only, not their errors, so with unequal errors it
    removes the least precise points first even when they agree within error, and
    every rejection trims the tails that &tau;<sup>2</sup> is meant to measure. On
    good data (15 monitors, errors varying threefold) it rejects at least one good
    point in about half of all sets at k&nbsp;=&nbsp;2.5 and a third at
    k&nbsp;=&nbsp;3. The default is k&nbsp;=&nbsp;2.5, as in the Age distribution
    tool.</p>

    <p>MSWD rejection, when ticked, removes one point at a time while the MSWD
    exceeds the Wendt &amp; Carl limit (or a fixed value). Each round the point most
    at odds with the others is found from its externally studentized residual,
    the distance from the weighted mean of all the other points relative to its
    own error, the other points' excess scatter &tau; and the error of their
    mean:</p>
    """ + E(r'r_{i} = \frac{x_{i} - \hat{\mu}_{(-i)}}{\sqrt{\sigma_{i}^{2} + \hat{\tau}^{2}_{(-i)} + \mathrm{SE}^{2}_{(-i)}}} \qquad |r_{i}| > \Phi^{-1}\!\left(1 - \frac{0.025}{n}\right)') + """
    <p>(the random-effects residual of Viechtbauer &amp; Cheung, 2010). The point is
    removed only if it exceeds the 5% two-sided critical value corrected for
    testing n points (2.81 at n&nbsp;=&nbsp;10, 3.02 at n&nbsp;=&nbsp;20). When no point
    does, the MSWD stays above the limit and the tool says so: the scatter is
    spread over the whole set, which REML or &radic;MSWD account for, rather than
    caused by a few outliers. Simulated on 15 points, this removes a good point in
    1% of sets without excess scatter and 15% with &tau;&nbsp;=&nbsp;&sigma;, against
    68% for removing points until the MSWD is reached, and finds a 6&sigma; outlier
    every time. The rejections run in the order relative-error filter, NMAD, MSWD,
    Chauvenet.</p>
    """

    # ── 7 ────────────────────────────────────────────────────────────────
    jcalc = """
    <a name="jcalc"></a>
    <h2>7. Flux regression</h2>

    <h3>What it does</h3>
    <p>J was measured only at the few holes that held monitors, but unknowns sat
    throughout the disk. This tool fits a smooth surface through the measured values
    so that J can be predicted at any position, and it attaches a realistic
    uncertainty to each prediction by Monte Carlo simulation rather than reporting a
    bare interpolated number.</p>

    """ + F('jcalc_surface.png',
            'A fitted flux surface over the irradiation disk. The monitor holes are '
            'the control points; unknowns are predicted from the surface with a '
            'Monte Carlo uncertainty.') + """

    <h3>The mathematics</h3>
    <p>Depending on how the disks were stacked and how many monitors are available,
    progressively richer surfaces can be fitted, from a single plane through a full
    three-dimensional form.</p>
    <table>
      <tr><th>Model</th><th>Surface</th><th>Parameters</th><th>Minimum monitors</th></tr>
      <tr><td>2D planar</td><td>f = ax + by + c</td><td>3</td><td>3</td></tr>
      <tr><td>Simple 3D</td><td>f = ax + by + cz + d</td><td>4</td><td>4</td></tr>
      <tr><td>Complex 3D</td><td>f = ax + by + cz + d + e(xz) + f(yz) + g(xy) + h(xyz)</td><td>8</td><td>8 (12+ preferred)</td></tr>
      <tr><td>Individual</td><td>an independent plane per disk</td><td>3 per disk</td><td>3 per disk</td></tr>
    </table>
    <p>Here x and y are the hole coordinates and z is the vertical position in the
    stack, so the three-dimensional models capture a flux gradient along the length of
    the irradiation container.</p>

    <p>A single least-squares fit gives a surface but no honest uncertainty at an
    arbitrary point, so the fit is repeated many times over perturbed inputs. Each
    iteration shifts the monitor positions at random and adds Gaussian noise to each
    measured J at its stated error, then refits the surface by weighted least squares
    (weights 1/&sigma;<sup>2</sup>). The mean and standard deviation of the resulting
    predictions are reported as the value and its uncertainty at each unknown; a
    monitor keeps its measured value:</p>
    """ + E(r'\hat{J}(x,y,z) = \langle J \rangle_{\mathrm{MC}} \qquad \sigma_{J}(x,y,z) = \mathrm{std}_{\mathrm{MC}}') + """
    <p>The position shift allows for where the monitor crystals sat. A monitor
    value is the mean of n crystals, each lying anywhere on the floor of a hole of
    diameter d. A point placed at random on a disc has a standard deviation of d/4 in
    each direction (its mean squared distance from the centre is d<sup>2</sup>/8,
    shared equally between x and y), so the mean position of n crystals is uncertain
    by</p>
    """ + E(r'\sigma_{\mathrm{pos}} = \frac{d}{4\sqrt{n}}') + """
    <p>in each direction: 0.2&ndash;0.3&nbsp;mm for about ten crystals in the
    2.7&ndash;4.0&nbsp;mm monitor holes of the 13- and 21-hole disks. The standard
    deviation of the shift is drawn anew in each iteration between 15 and
    450&nbsp;&mu;m (typically about 0.26&nbsp;mm). Hole coordinates are in cm.</p>
    <p>The perturbations only carry the errors the monitors are stated to have.
    When the monitors scatter about the fitted surface by more than that (a weighted
    MSWD above the Wendt &amp; Carl limit for n &minus; k degrees of freedom), every
    predicted &sigma;<sub>J</sub> is multiplied by &radic;MSWD, the same rule the
    other tools use for a weighted mean; the correlations between holes are kept.
    The MSWD of each fit is reported when the regression finishes and stored in the
    <code>.jflux</code> file.</p>
    <p>Monitors report their own measured values; unknowns take the Monte Carlo
    prediction. Because a more flexible surface always fits the control points better,
    the tool judges models by how well they predict data they did not see: leave-one-out
    cross-validation error, the AIC, which penalises extra parameters, and a reduced
    &chi;<sup>2</sup> near one as a check that the error bars are neither too large nor
    too small. The fitted J and its uncertainty can be sent to the dating tools or saved
    as a <code>.jflux</code> file, which the Plateau &amp; isochron tool reads to recover the true
    correlation between holes when combining aliquots.</p>
    """

    # ── 8 ────────────────────────────────────────────────────────────────
    plateau = """
    <a name="plateau"></a>
    <h2>8. Plateau &amp; isochron</h2>

    <h3>What it does</h3>
    <p>In a step-heating experiment the argon is released in increasingly hot steps,
    each yielding an age. If the mineral retained its argon cleanly, the middle steps
    agree, forming a plateau, and that interval gives the best age.
    The tool draws the age spectrum, identifies the plateau against criteria you set,
    computes its age with the appropriate uncertainty tiers, and provides isochrons as
    an independent test. It also combines several aliquots of one sample.</p>

    """ + F('plateau_spectrum.png',
            'A step-heating spectrum. Box heights are 2-sigma per step and box widths '
            'are the fraction of 39Ar released; the highlighted steps form the '
            'plateau, with its weighted-mean age and MSWD below.') + """

    <h3>The mathematics</h3>
    <p><i>Identifying the plateau.</i> A run of adjacent steps qualifies when it meets
    three conditions you control: a minimum number of steps, a minimum fraction of the
    total <sup>39</sup>Ar released, and a minimum probability on the &chi;<sup>2</sup>
    test of the weighted mean. Individual steps can also be forced in or excluded by
    hand.</p>

    <p><i>The plateau age.</i> The age is the weighted mean of the qualifying steps,
    weighted by analytical uncertainty, with
    the uncertainty scaled for overdispersion according to the mode you choose.</p>
    <table>
      <tr><th>Error mode</th><th>Uncertainty</th><th>Use when</th></tr>
      <tr><td>SEM &times; 2</td><td>2&nbsp;&middot;&nbsp;SEM</td><td>MSWD &asymp; 1, no scatter correction</td></tr>
      <tr><td>SEM &times; &radic;MSWD &times; 2</td><td>2&nbsp;&middot;&nbsp;SEM&nbsp;&middot;&nbsp;&radic;MSWD</td><td>most cases</td></tr>
      <tr><td>Student's t</td><td>SEM&nbsp;&middot;&nbsp;&radic;MSWD&nbsp;&middot;&nbsp;t(n&minus;1)</td><td>small n; matches ArArCALC when p &lt; 5%</td></tr>
    </table>

    <p><i>Why J enters once.</i> Every step of one aliquot shares the same J, because
    the steps come from the same position in the disk; J is therefore fully correlated
    across them and must not be allowed to average down. The internal and external
    tiers add it after the mean rather than within it:</p>
    """ + E(r'\sigma_{\mathrm{int}} = \sqrt{\sigma_{\mathrm{ana}}^{2} + \left(\frac{\partial t}{\partial \ln J} f_{J}\right)^{2}} \qquad \sigma_{\mathrm{ext}} = \sqrt{\sigma_{\mathrm{int}}^{2} + \sum_{k} c_{k}^{2}}') + """
    <p>where the c<sub>k</sub> are the calibration terms of Section&nbsp;3.</p>

    <p><i>Combining aliquots.</i> Combining aliquots is itself a weighted mean, so a
    shared systematic shifts them all together while an independent one diminishes as
    roughly the square root of their number. Whether two aliquots share J depends on
    where they sat.</p>
    <table>
      <tr><th>Provenance</th><th>Correlation of J</th><th>Does J average down?</th></tr>
      <tr><td>Same hole, repeat analyses</td><td>&rho; = 1</td><td>No</td></tr>
      <tr><td>Different holes, same disk</td><td>0 &lt; &rho; &lt; 1, from the flux model</td><td>Partly</td></tr>
      <tr><td>Different disks or irradiations</td><td>&rho; = 0</td><td>Yes, J only</td></tr>
      <tr><td>Different laboratories or calibrations</td><td>&rho; = 0; monitor and &lambda; also independent</td><td>J yes; monitor and &lambda; only if they differ</td></tr>
    </table>
    <p>Loading a <code>.jflux</code> file lets the combination use the genuine
    correlation between holes taken from the flux model, rather than assuming it is
    either zero or one. Because the decay constant and the monitor age are common to
    everything on one calibration, the external tier does not average down even across
    disks; it sets a floor on the achievable precision. The two combination methods are
    those of Section&nbsp;1: <b>WM + √MSWD</b>, on the per-aliquot
    plateau ages (n = aliquots, the default) or with the steps normalised to a single
    J and pooled in ratio space (n = steps), and <b>GLS + ML</b> (Vermeesch 2015 §13),
    which combines the per-aliquot plateau ages on the covariance with a
    maximum-likelihood dispersion term (n = aliquots) and is the right choice when
    the aliquots span more than one hole or disk, or disagree beyond analytical
    error.  On n = aliquots both use the
    same J correlation and give the same age as pooling the aliquots in the Age
    distribution tool.</p>

    <p><i>Isochrons.</i> Rather than assume the trapped argon has an atmospheric
    composition, an isochron solves for it from the spread of the steps. For the
    inverse isochron (x = <sup>39</sup>Ar<sub>K</sub>/<sup>40</sup>Ar,
    y = <sup>36</sup>Ar/<sup>40</sup>Ar) the age comes from the slope and intercept,</p>
    """ + E(r'R = -\,b/a \qquad t = \frac{1}{\lambda}\ln\left(RJ + 1\right)') + """
    <p>and for the normal isochron (x = <sup>39</sup>Ar<sub>K</sub>/<sup>36</sup>Ar,
    y = <sup>40</sup>Ar/<sup>36</sup>Ar) from the slope alone,</p>
    """ + E(r't = \frac{1}{\lambda}\ln\left(bJ + 1\right)') + """
    <p>Both are fitted by the method of York (1969), the correct weighted regression
    when both axes carry correlated errors, and both report 2&sigma; analytical
    uncertainties inflated by the square root of the MSWD where the scatter warrants
    it.</p>
    """

    # ── 9 ────────────────────────────────────────────────────────────────
    prob = """
    <a name="prob"></a>
    <h2>9. Age distribution</h2>

    <h3>What it does</h3>
    <p>When a sample yields a population of single-crystal or single-aliquot ages, the
    shape of their distribution carries information a mean alone cannot: whether the
    ages form one cluster or several. This tool ranks the ages, places a Gaussian
    of the appropriate width on each, sums them into a probability-density curve,
    computes a weighted mean for each sample, and can pool several samples with the
    correlations handled correctly.</p>

    """ + F('prob_plot.png',
            'Ranked ages with 1-sigma bars and the summed-Gaussian probability curve. '
            'Excluded points are open markers, and the dashed curve shows the '
            'distribution with them added back.') + """

    <h3>The mathematics</h3>
    <p>Each accepted age contributes a Gaussian of unit area, centred on its value with
    its own width, and the curve is their sum; it is a representation of the population
    rather than a fitted model. The weighted mean, MSWD, and probability use the same
    analytical-weighted machinery as elsewhere,</p>
    """ + E(r'\bar{t} = \frac{\sum w_{i} t_{i}}{\sum w_{i}}, \quad \mathrm{MSWD} = \frac{1}{n-1}\sum w_{i}\left(t_{i} - \bar{t}\right)^{2}, \quad w_{i} = \frac{1}{\sigma_{\mathrm{ana},i}^{2}}') + """
    <p>Pooling is where the covariance treatment matters. When samples are combined,
    only the independent parts of their uncertainties should diminish. Both methods
    assemble the same covariance matrix for this (the decomposition of Vermeesch 2015),</p>
    """ + E(r'\Sigma_{ij} = a_{i}^{2}\,\delta_{ij} + \rho^{\mathrm{pos}}_{ij}\,p_{i} p_{j} + \rho^{\mathrm{std}}_{ij}\,s_{i} s_{j}') + \
    E(r'a = \sigma_{\mathrm{ana}} \qquad p = \sqrt{\sigma_{\mathrm{int}}^{2} - \sigma_{\mathrm{ana}}^{2}} \qquad s = \sqrt{\sigma_{\mathrm{ext}}^{2} - \sigma_{\mathrm{int}}^{2}}') + """
    <p>in which a is the analytical term, p the flux term, and s the monitor-and-decay
    term. The indicator &rho;<sup>pos</sup> is one for samples that occupied the same
    irradiation position and zero otherwise, and &rho;<sup>std</sup> is one for samples
    on the same calibration. Samples that shared a position share their flux
    completely, so that part does not shrink when they are combined; samples from
    different irradiations have independent flux, which does. The monitor-and-decay
    systematic is common to every active sample and never averages down.</p>

    <p>The two statistical methods are again those of Section&nbsp;1. Position grouping
    is automatic: samples with a matching calibration fall into the same fully
    correlated group, and the pooled report states which samples grouped together.
    Outliers can be removed by NMAD or by iterative 2&sigma; rejection, or by
    clicking individual points; in every case a dashed curve shows what the excluded
    points would have contributed, so nothing is hidden from view.</p>

    <h3>Mixture view</h3>
    <p>The <b>Mixture</b> button replaces the plot with a mixture model of one sample,
    picked in the box above it; the plot and the samples are not changed, and the
    button switches back. The model asks whether the analyses come from one age
    population or several, and which analysis belongs to which. It takes the analyses,
    z<sub>j</sub>&nbsp;&plusmn;&nbsp;s<sub>j</sub> with their analytical uncertainties
    (z is the age, or its logarithm on the log scale), to come from k components with
    ages &mu;<sub>i</sub> in proportions &pi;<sub>i</sub> (Galbraith &amp; Green 1990;
    Galbraith &amp; Laslett 1993), and finds the values that maximise</p>
    """ + E(r'\ln L = \sum_{j=1}^{n} \ln \sum_{i=1}^{k} \pi_{i}\,\phi(z_{j};\,\mu_{i},\,s_{j})') + """
    <p>where &phi; is the normal density. The probability that analysis j belongs to
    component i is</p>
    """ + E(r'p_{ij} = \frac{\pi_{i}\,\phi(z_{j};\,\mu_{i},\,s_{j})}{\sum_{l=1}^{k} \pi_{l}\,\phi(z_{j};\,\mu_{l},\,s_{j})}') + """
    <p>and the maximum is reached by repeating, from several starting points,</p>
    """ + E(r'\pi_{i} = \frac{1}{n}\sum_{j} p_{ij}, \qquad \mu_{i} = \frac{\sum_{j} p_{ij}\,z_{j}/s_{j}^{2}}{\sum_{j} p_{ij}/s_{j}^{2}}') + """
    <p>so each component age is a weighted mean of the analyses, each weighted by how
    likely it is to belong. The uncertainties of &mu;<sub>i</sub> and &pi;<sub>i</sub>
    come from the information matrix (Galbraith 2005), as in IsoplotR's peakfit
    (Vermeesch 2018). The number of components is chosen with the Bayes Information
    Criterion (Schwarz 1978),</p>
    """ + E(r'\mathrm{BIC} = -2\ln L + (2k-1)\ln n') + """
    <p>taking the fewest components within 2 of the lowest BIC, since a smaller
    difference is not worth more than a bare mention (Kass &amp; Raftery 1995). Up to
    ten components are tried, because every old xenocryst takes one of its own. A
    component of at least two analyses also has an MSWD, with every analysis weighted
    by p<sub>ij</sub>,</p>
    """ + E(r'\mathrm{MSWD}_{i} = \frac{\sum_{j} p_{ij}(z_{j} - \mu_{i})^{2}/s_{j}^{2}}{\sum_{j} p_{ij} - 1}') + """
    <p>and when it exceeds the Wendt &amp; Carl limit for &sum;<sub>j</sub>p<sub>ij</sub>
    analyses, the analytical uncertainty of the component is multiplied by &radic;MSWD,
    the rule used for every weighted mean. J and the calibration are then added to each
    component age to give the internal and external tiers, as for the weighted mean of
    the sample.</p>

    <p>By default the model sees every analysis of the sample: deciding which
    analyses belong together is its job, so the outlier rejection and the clicked
    exclusions are not used. <b>Included only</b> uses the analyses included on the
    plot. The log scale, IsoplotR's default, leaves out ages of zero or below.</p>

    <p>The minimum age model (Galbraith et al. 1999) is offered for the youngest
    population on its own: one discrete youngest age plus a continuous spread of older
    analyses, truncated at it. It assumes no analysis is too young, so a single young
    outlier (argon loss) becomes the minimum, and old xenocrysts widen the spread until
    it takes in analyses that do not belong. It should be run on data without them.</p>

    <p>Two cautions apply to every result. The BIC also splits a single population whose
    scatter exceeds its analytical errors, and the number of components it finds grows
    with the number of analyses, so a split is a reason to look at the data, not proof
    of two populations. And the youngest component is not automatically the eruption
    age: a single blank-dominated analysis can form a component of its own. The radial
    plot (Galbraith 1988) helps with both. Each analysis is plotted at its precision,
    1/&sigma;, against its deviation from the weighted mean in units of &sigma;, so its
    age is read where the line from the origin through it meets the arc, and each
    component is such a line. The ΔBIC panel shows how clearly the number of components
    was chosen.</p>
    """ + F('prob_mixture.png',
            'The Mixture view. Left: the ranked analyses and the probability curve, split '
            'into the components; each analysis is coloured by the component it most '
            'likely belongs to. Right: the radial plot and the BIC of each number of '
            'components.') + """
    <p>While the view is on, both export buttons export the mixture model of the
    sample shown, and nothing else. <b>Export mixture figure</b> lays it out as in the
    window, the diagram on the left and the radial plot and &Delta;BIC on the right, or
    stacks the panels. <b>Export mixture XLSX / CSV</b> writes the component ages with
    their three tiers, proportions and MSWD, then every analysis the model used with its
    point data and its probability of belonging to each component; the constants and raw
    signals are those of that sample. The settings are saved with the session.</p>
    """

    # ── 10 ───────────────────────────────────────────────────────────────
    calib = """
    <a name="calib"></a>
    <h2>10. Calibrations and constants</h2>

    <p>Every tool defaults to the calibration of Phillips et al. (2022): a Fish Canyon
    age of 28.176&nbsp;&plusmn;&nbsp;0.0115&nbsp;Ma (1&sigma;, the published
    &plusmn;0.023&nbsp;Ma at 2&sigma; halved), paired with the decay constant it is
    anchored to, Min et&nbsp;al. (2000):
    5.463&nbsp;&plusmn;&nbsp;0.054&nbsp;&times;&nbsp;10<sup>&minus;10</sup>&nbsp;yr<sup>&minus;1</sup>.
    The monitor age is astronomically calibrated, so ages are computed on the
    anchored chain of Section&nbsp;3 (A1 tephra, 6.943&nbsp;Ma;
    R<sub>FCs/A1</sub>&nbsp;=&nbsp;4.0819), and the external tier comes from
    R<sub>FCs/A1</sub>, the A1 age and &lambda; rather than from the Fish Canyon
    age's quoted error. Any of the presets below may be chosen, or custom values
    entered.</p>

    <table>
      <tr><th>Monitor-age preset</th><th>Age (Ma)</th><th>1&sigma; (Ma)</th><th>External &lambda;</th></tr>
      <tr><td>Carter et al. (2025)</td><td>28.183</td><td>0.017</td><td>Yes</td></tr>
      <tr><td>Phillips et al. (2022)</td><td>28.176</td><td>0.0115</td><td>Yes</td></tr>
      <tr><td>Rivera et al. (2011)</td><td>28.172</td><td>0.014</td><td>Yes</td></tr>
      <tr><td>Kuiper et al. (2008)</td><td>28.201</td><td>0.023</td><td>Yes</td></tr>
      <tr><td>Renne et al. (1998)</td><td>28.02</td><td>0.14</td><td>No</td></tr>
      <tr><td>Jourdan &amp; Renne (2007)</td><td>28.03</td><td>0.08</td><td>No</td></tr>
    </table>

    <p>Two points deserve emphasis. First, the astronomically tuned monitor ages of
    Kuiper (2008), Rivera (2011) and Phillips (2022) were derived with the Min et al.
    (2000) decay constant, and the tools pair them with it. Because the ages are
    computed on the anchored chain, another &lambda; can be chosen without breaking
    the calibration: the chain then gives the Fish Canyon age that the anchor implies
    at that &lambda;, not the published 28.176&nbsp;Ma. Second, the External
    &lambda; setting only matters for calibrations without an anchor. For those, a
    jointly calibrated monitor age already contains the decay-constant uncertainty
    within its published error, so the setting prevents it from being added a second
    time, and for the Carter calibration it also activates the
    &lambda;&ndash;monitor covariance term that reproduces Carter's Table&nbsp;9.</p>

    <h3>Reading Carter's three uncertainties</h3>
    <p>Carter's Fish Canyon age appears in the literature with three different
    uncertainties. These are three different tiers, not one number quoted three ways.</p>
    <table>
      <tr><th>1&sigma; value</th><th>What it is</th></tr>
      <tr><td>0.0112 Ma</td><td>The posterior precision: how tightly the fit pins the age.</td></tr>
      <tr><td>0.017 Ma</td><td>The full external systematic, with input-data systematics folded in. This is the tool's external tier.</td></tr>
      <tr><td>0.035 Ma</td><td>The usage total when actually dating, which adds the R-value and neutron-fluence terms on top.</td></tr>
    </table>
    """

    # ── 11 ───────────────────────────────────────────────────────────────
    refs = """
    <a name="refs"></a>
    <h2>11. References</h2>
    <ul style="font-size:13px;">
      <li>Carter et al. (2025): jointly calibrated <sup>40</sup>K decay constant
          and Fish Canyon age. <i>Geochim. Cosmochim. Acta</i>.</li>
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
      <li>Min et al. (2000): revised <sup>40</sup>K decay constant.
          <i>Geochim. Cosmochim. Acta</i> 64, 73&ndash;98.</li>
      <li>Niespolo et al. (2017): intercalibration and the R-value age form.
          <i>Quat. Geochronol.</i> 39, 205&ndash;213.</li>
      <li>Phillips et al. (2022): astronomically tuned Fish Canyon age.
          <i>Geochim. Cosmochim. Acta</i>.</li>
      <li>Renne et al. (1998, 2010): intercalibration and joint calibration.
          <i>Chem. Geol.</i> 145, 117&ndash;152; <i>GCA</i> 74, 5349&ndash;5367.</li>
      <li>Rivera et al. (2011): astronomically tuned Fish Canyon age.
          <i>Earth Planet. Sci. Lett.</i> 311, 420&ndash;426.</li>
      <li>Schwarz (1978): the Bayes Information Criterion. <i>Ann. Stat.</i> 6,
          461&ndash;464.</li>
      <li>Schaen et al. (2021): interpretation and reporting standards.
          <i>GSA Bull.</i> 133, 461&ndash;487.</li>
      <li>Vermeesch (2015): revised error propagation including covariances.
          <i>GCA</i> 171, 325&ndash;337.</li>
      <li>Vermeesch (2018): IsoplotR. <i>Geosci. Front.</i> 9, 1479&ndash;1493.</li>
      <li>Wendt &amp; Carl (1991): MSWD confidence limits.
          <i>Chem. Geol.</i> 86, 275&ndash;285.</li>
      <li>York (1969): least-squares fit with correlated errors on both axes.
          <i>Earth Planet. Sci. Lett.</i> 5, 320&ndash;324.</li>
    </ul>
    <p class="back"><a href="#top">Back to top</a></p>
    """

    return (style + toc + stats + idea + tiers + workflow
            + converter + wm + jcalc + plateau + prob + calib + refs)
