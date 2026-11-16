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
    <p class="lede">The methods and equations used by the five tools.</p>

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
        <li><a href="#calib">Calibrations and constants</a></li>
        <li><a href="#refs">References</a></li>
      </ol>
    </div>
    """

    # ── 1 ────────────────────────────────────────────────────────────────
    workflow = """
    <a name="workflow"></a>
    <h2>1. The workflow</h2>

    <p>Each tool runs on its own, but they are normally used in this order:</p>
    <ol>
      <li><a href="#converter"><b>Data converter</b></a>: ArArCALC or Mass Spec
          export (.xlsx) to <code>.arar</code> files.</li>
      <li><a href="#wm"><b>Flux monitors</b></a>: the Fish Canyon monitor analyses of
          each irradiation position reduced to one R and J.</li>
      <li><a href="#jcalc"><b>Flux regression</b></a>: J fitted across the irradiation
          disk and predicted, with its uncertainty, at the position of each
          unknown.</li>
      <li><a href="#plateau"><b>Plateau &amp; isochron</b></a>: plateau and isochron
          ages of step-heating runs, and combined ages of several aliquots.</li>
      <li><a href="#prob"><b>Age distribution</b></a>: probability density plots,
          weighted means and pooled ages of many analyses or samples, and mixture
          models.</li>
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
    <p>J comes from the Fish Canyon sanidine monitor (FCs) irradiated with the sample,
    with age t<sub>FCs</sub> and measured ratio R<sub>std</sub>:</p>
    """ + E(r'J = \frac{e^{\lambda t_{\mathrm{FCs}}} - 1}{R_{\mathrm{std}}}') + """
    <p>Substituting J gives the age in terms of the sample-to-monitor ratio
    R<sub>u/FCs</sub>&nbsp;=&nbsp;R/R<sub>std</sub>:</p>
    """ + E(r't = \frac{1}{\lambda}\,\ln\left[\,1 + R_{u/\mathrm{FCs}}\left(e^{\lambda t_{\mathrm{FCs}}} - 1\right)\right]') + """
    <p>&lambda;t is small (about 0.03 at 55&nbsp;Ma), so
    t&nbsp;&asymp;&nbsp;R<sub>u/FCs</sub>&thinsp;&middot;&thinsp;t<sub>FCs</sub>: the
    age is nearly proportional to the monitor age and depends only weakly on &lambda;.
    The uncertainty of &lambda; therefore mostly cancels between Ar/Ar ages on the same
    monitor. Whether it enters in full against another decay system, such as
    U&ndash;Pb, depends on how the monitor age was calibrated
    (Section&nbsp;3).</p>
    """

    # ── 3 ────────────────────────────────────────────────────────────────
    tiers = """
    <a name="tiers"></a>
    <h2>3. The three uncertainty tiers</h2>

    <p>Every age is reported with three uncertainties. Each adds a further source to
    the one before; which to quote depends on what the age is compared with.</p>

    <table>
      <tr><th>Tier</th><th>Contains</th><th>Use for comparison with</th></tr>
      <tr><td><b>Analytical</b></td>
          <td>The measurement of R (counting statistics, blanks, detector).</td>
          <td>Aliquots from the same irradiation position.</td></tr>
      <tr><td><b>Internal</b></td>
          <td>Plus the J uncertainty. ArArCALC reports this as the age error.</td>
          <td>Ar/Ar ages on the same monitor and &lambda;.</td></tr>
      <tr><td><b>External</b></td>
          <td>Plus the monitor age and the decay constant.</td>
          <td>Ar/Ar ages on other monitors or from other laboratories, and the
              astronomical timescale.</td></tr>
    </table>

    <p>For a single age:</p>
    """ + E(r'\sigma_{\mathrm{int}}^{2} = \sigma_{\mathrm{ana}}^{2} + \left(\frac{\partial t}{\partial \ln J}\,f_{J}\right)^{2} \qquad \frac{\partial t}{\partial \ln J} = \frac{1 - e^{-\lambda t}}{\lambda}') + \
    E(r'\sigma_{\mathrm{ext}}^{2} = \sigma_{\mathrm{int}}^{2} + \sum_{k} c_{k}^{2}') + """
    <p>f<sub>J</sub> is the relative uncertainty of J (from the monitor measurement
    only) and the c<sub>k</sub> are the calibration terms below.
    &part;t/&part;ln&nbsp;J is slightly less than t (by about 1.5% at 55&nbsp;Ma) and
    is the same for a relative change in any ratio in the age equation.</p>

    <h3>Calibration terms</h3>
    <p>J itself depends on &lambda; and t<sub>FCs</sub> (Section&nbsp;2), so the
    calibration terms are taken from the substituted equation, in which J does not
    appear. With X&nbsp;=&nbsp;e<sup>&lambda;t</sup>:</p>
    """ + E(r'X = R_{u/\mathrm{FCs}}\left(e^{\lambda t_{\mathrm{FCs}}} - 1\right) + 1 = e^{\lambda t}') + \
    E(r'\frac{\partial t}{\partial \lambda} = -\frac{\ln X}{\lambda^{2}} + \frac{R_{u/\mathrm{FCs}}\,t_{\mathrm{FCs}}\,e^{\lambda t_{\mathrm{FCs}}}}{\lambda X}') + \
    E(r'\frac{\partial t}{\partial t_{\mathrm{FCs}}} = \frac{R_{u/\mathrm{FCs}}\,e^{\lambda t_{\mathrm{FCs}}}}{X} \qquad \frac{\partial t}{\partial R_{u/\mathrm{FCs}}} = \frac{e^{\lambda t_{\mathrm{FCs}}} - 1}{\lambda X}') + \
    E(r'\sigma_{t}^{2} = \left(\frac{\partial t}{\partial \lambda}\sigma_{\lambda}\right)^{2} + \left(\frac{\partial t}{\partial t_{\mathrm{FCs}}}\sigma_{t_{\mathrm{FCs}}}\right)^{2} + \left(\frac{\partial t}{\partial R_{u/\mathrm{FCs}}}\sigma_{R_{u/\mathrm{FCs}}}\right)^{2} + 2\,\frac{\partial t}{\partial \lambda}\frac{\partial t}{\partial t_{\mathrm{FCs}}}\,\mathrm{cov}(\lambda, t_{\mathrm{FCs}})') + """
    <p>The two terms of &part;t/&part;&lambda; nearly cancel. Holding J fixed instead
    would give &part;t/&part;&lambda;&nbsp;=&nbsp;&minus;t/&lambda; and put the full
    relative uncertainty of &lambda; on the age (1.0% at 1&sigma; for Min et&nbsp;al.
    2000). The covariance term is zero when the monitor age was calibrated
    astronomically (Kuiper 2008, Rivera 2011, Phillips 2022). For Carter et&nbsp;al.
    (2025), where &lambda; and the monitor age were fitted together, the tools use
    Carter's posterior covariance; it is numerically negligible and the result matches
    their Table&nbsp;9 to within 4&times;10<sup>&minus;5</sup>.</p>

    <h3>Astronomically calibrated monitors</h3>
    <p>The Fish Canyon ages of Kuiper (2008), Rivera (2011) and Phillips (2022) were
    calculated from the astronomical age of an anchor tephra with the Min et&nbsp;al.
    (2000) &lambda;. Renne et&nbsp;al. (1998, p.&nbsp;120) note that their eq.&nbsp;6
    does not hold for a standard age of this kind, so for these calibrations the tools
    use their two-step equation (eq.&nbsp;5), whose inputs are independent:</p>
    """ + E(r't = \frac{1}{\lambda}\ln\left[1 + R_{u/\mathrm{FCs}}\,R_{\mathrm{FCs}/A}\left(e^{\lambda t_{A}} - 1\right)\right]') + """
    <p>R<sub>u/FCs</sub> is the sample-to-monitor ratio from the irradiation,
    R<sub>FCs/A</sub> the published ratio of Fish Canyon to the anchor, and
    t<sub>A</sub> the anchor's astronomical age. With
    P&nbsp;=&nbsp;R<sub>u/FCs</sub>R<sub>FCs/A</sub> the partial derivatives are
    (Kuiper et&nbsp;al. 2008, SOM)</p>
    """ + E(r'\frac{\partial t}{\partial \ln R} = \frac{1 - e^{-\lambda t}}{\lambda} \qquad \frac{\partial t}{\partial t_{A}} = \frac{P\,e^{\lambda t_{A}}}{e^{\lambda t}} \qquad \frac{\partial t}{\partial \lambda} = -\frac{t}{\lambda} + \frac{P\,t_{A}\,e^{\lambda t_{A}}}{\lambda\,e^{\lambda t}}') + \
    E(r'\sigma_{\mathrm{ext}}^{2} = \sigma_{\mathrm{int}}^{2} + \left(\frac{\partial t}{\partial \ln R}\,\frac{\sigma_{R_{\mathrm{FCs}/A}}}{R_{\mathrm{FCs}/A}}\right)^{2} + \left(\frac{\partial t}{\partial t_{A}}\,\sigma_{t_{A}}\right)^{2} + \left(\frac{\partial t}{\partial \lambda}\,\sigma_{\lambda}\right)^{2}') + """
    <p>Again the two terms of &part;t/&part;&lambda; nearly cancel. The anchors
    (1&sigma;):</p>
    <table>
      <tr><th>Calibration</th><th>Anchor</th><th>t<sub>A</sub> (Ma)</th><th>R<sub>FCs/A</sub></th></tr>
      <tr><td>Kuiper et al. (2008)</td><td>Melilla tephras, composite</td><td>6.500 &plusmn; 0.005</td><td>4.3664 &plusmn; 0.0009</td></tr>
      <tr><td>Rivera et al. (2011)</td><td>A1 tephra, Faneromeni</td><td>6.943 &plusmn; 0.0025</td><td>4.0813 &plusmn; 0.0013</td></tr>
      <tr><td>Phillips et al. (2022)</td><td>A1 tephra, Faneromeni</td><td>6.943 &plusmn; 0.0025</td><td>4.0819 &plusmn; 0.0007</td></tr>
    </table>
    <p>R<sub>FCs/A</sub> is set to the value that gives each paper's published Fish
    Canyon age at the Min et&nbsp;al. (2000) &lambda;; the printed values differ from it
    in the fifth figure. At that &lambda; the two-step and single-step equations give
    the same age and differ only in the external uncertainty. If the monitor-age
    uncertainty is changed from the preset, the single-step equation is used. When ages
    are combined, each of the three calibration terms is common to all of them and is
    carried through the weighted mean separately, so none of them averages down.</p>

    <h3>The full decay-constant term</h3>
    <p>With R and the anchor fixed, a 1% change in &lambda; moves a 55&nbsp;Ma age by
    about 0.007%; with J fixed it moves it by nearly 1%:</p>
    """ + E(r'\left.\frac{\partial t}{\partial \lambda}\right|_{R,\;t_{\mathrm{std}}} \approx 0 \qquad\qquad \left.\frac{\partial t}{\partial \lambda}\right|_{J} = -\frac{t}{\lambda} \;\Rightarrow\; \left(\frac{\sigma_{t}}{t}\right)_{\lambda} = \frac{\sigma_{\lambda}}{\lambda}') + """
    <p>The full term,</p>
    """ + E(r'\sigma_{\mathrm{full}}^{2} = \sigma_{\mathrm{ext}}^{2} + \left(t\,\frac{\sigma_{\lambda}}{\lambda}\right)^{2}') + """
    <p>applies only when the age is used as an independent <sup>40</sup>K clock against
    another decay system, with a physically measured &lambda; (such as Min et&nbsp;al.
    2000) and a monitor age tied to it by K&ndash;Ar rather than by astronomy or
    U&ndash;Pb. It is not added for:</p>
    <ul>
      <li>astronomically calibrated monitors (Kuiper 2008, Rivera 2011, Phillips 2022):
          the monitor age does not depend on &lambda;, which cancels to below 0.01%.
          This also holds against U&ndash;Pb, where the comparison is between the
          astronomical and the uranium timescales;</li>
      <li>the jointly optimised calibration of Carter et&nbsp;al. (2025): &lambda; already
          enters the external tier through its covariance with the monitor age, and
          adding it again would count it twice.</li>
    </ul>
    <p>Renne et&nbsp;al. (2010, 2011) also calibrated the monitor age and &lambda;
    jointly, but the covariances they publish are rounded too far to be propagated
    for ages below about 1&nbsp;Ga; they recommend their Monte Carlo triplets instead,
    which are not included here. For these two calibrations SimplAr therefore treats
    the monitor age and &lambda; as independent and adds the full term. This
    overstates the external error, by about a factor of four at 56&nbsp;Ma, and the
    tools say so when they are selected.</p>
    <p>The term applies if the monitor age would change with &lambda;, and not
    otherwise.</p>
    """

    # ── 4 ────────────────────────────────────────────────────────────────
    stats = """
    <a name="stats"></a>
    <h2>4. Combining ages</h2>

    <p>Wherever ages are combined, two methods are offered: <b>WM + √MSWD</b> and
    <b>GLS + ML</b>. Both use the same covariance: the J and calibration parts of the
    uncertainty are correlated between ages that share an irradiation position or a
    calibration (Sections&nbsp;8 and 9), so they do not average down. The methods
    differ in the mean and in how they treat overdispersion, scatter larger than the
    analytical errors allow.</p>
    <table>
      <tr><th></th><th>WM + √MSWD</th><th>GLS + ML</th></tr>
      <tr><td><b>Mean</b></td>
          <td>Inverse-variance weighted mean on the analytical errors.</td>
          <td>The covariance-weighted mean of Vermeesch (2015, eqs. 91&ndash;92), which
              is the generalised least-squares (GLS) estimate, on the full covariance
              plus &omega;<sup>2</sup>.</td></tr>
      <tr><td><b>Overdispersion</b></td>
          <td>The analytical tier is multiplied by &radic;MSWD when the MSWD exceeds the
              Wendt &amp; Carl (1991) limit below.</td>
          <td>An additive variance &omega;<sup>2</sup>, estimated by maximum likelihood
              on the full covariance and added to every age. Vermeesch (2015, eq. 94)
              adds it to each age's own variance, and IsoplotR fits it on the
              analytical errors only.</td></tr>
    </table>
    """ + E(r'\mathrm{MSWD} > 1 + 2\sqrt{\frac{2}{n-1}}') + """
    <p>The same limit is used in every tool. Adding &omega;<sup>2</sup> to every age
    evens out the weights, so a few very precise analyses no longer decide the mean.
    &omega;<sup>2</sup> is fitted on the full covariance (analytical, J and
    calibration), so scatter already explained by independent J errors is not counted
    twice. The three tiers are then the variance of the GLS mean.</p>

    <p>With similar analytical errors, or without overdispersion, the two methods give
    practically the same age and uncertainty; with partial J correlations from a
    <code>.jflux</code> file the means differ slightly. With overdispersion and widely
    different analytical errors, as for single crystals of variable K content, they do
    not agree. In simulations with a tenfold range of analytical errors and an
    overdispersion of one to two times the typical analytical error, the 95% interval
    contained the true age in about 70&ndash;80% of cases with WM + √MSWD and
    80&ndash;93% with GLS + ML, whose mean was also closer to the true age. Coverage is
    lower for few ages, most of all for GLS + ML, because &omega;<sup>2</sup> estimated
    from five or ten ages is itself uncertain.</p>

    <p>For stacked step-heating aliquots, GLS + ML combines the plateau ages of the
    aliquots (n = aliquots). WM + √MSWD does the same by default, or pools every step
    in ratio space (n = steps). When the aliquots disagree beyond analytical error, use
    n = aliquots: the steps of one aliquot share that aliquot's offset, and treating
    them as independent understates the uncertainty, more so the more steps there
    are.</p>

    <p>The mean follows Vermeesch's combination step (his eqs. 91&ndash;94), not his
    full data reduction. The tools start from reduced ages, so the covariance contains only the J
    and monitor/&lambda; terms. Correlations from shared blanks, detector calibration,
    mass fractionation and interference corrections need the raw isotope data and are
    not included.</p>
    """

    # ── 5 ────────────────────────────────────────────────────────────────
    converter = """
    <a name="converter"></a>
    <h2>5. Data converter</h2>

    <p>Reads an ArArCALC summary export or a Mass Spec raw export (.xlsx) and writes
    <code>.arar</code> files for the other tools. A dialog sets the layout and the
    column of each field before anything is written. Each file stores the J value and
    its error, the monitor ratio R<sub>std</sub>, the monitor age and the irradiation
    label, so the other tools know the calibration the ages were calculated with and
    can recalculate them on another.</p>

    """ + F('converter_mapping.png',
            'The format and column-mapping dialog: the export layout, and the '
            'spreadsheet column of each field.') + """

    <p>Ages are not recalculated. J and R<sub>std</sub> are kept consistent through</p>
    """ + E(r'J = \frac{e^{\lambda t_{\mathrm{FCs}}} - 1}{R_{\mathrm{std}}}') + """
    <p>so a calibration given in either form gives the same age. The monitor ages
    offered are the presets of Section&nbsp;10.</p>
    """

    # ── 6 ────────────────────────────────────────────────────────────────
    wm = """
    <a name="wm"></a>
    <h2>6. Flux monitors</h2>

    <p>Reduces the Fish Canyon monitor analyses of one irradiation position to one R
    and J, and separates real scatter (flux gradients, grain-to-grain differences)
    from analytical scatter. It reads .arar, .ararpt and .ararstack files (R and its
    1&sigma; error from every step, grouped by irradiation position), Excel/CSV tables
    and pasted values. With the ratio box unticked it is a weighted-mean calculator for
    any quantity.</p>

    """ + F('wm_main.png',
            'The analyses with 1-sigma error bars, the mean and its uncertainty band, '
            'and the MSWD and p-value in the title.') + """

    <h3>Weighted mean</h3>
    """ + E(r'w_{i} = \frac{1}{\sigma_{i}^{2}} \qquad \bar{x} = \frac{\sum w_{i} x_{i}}{\sum w_{i}} \qquad \mathrm{SE} = \sqrt{\frac{1}{\sum w_{i}}}') + \
    E(r'\mathrm{MSWD} = \frac{1}{n-1}\sum w_{i}\left(x_{i} - \bar{x}\right)^{2} \qquad p = P\left(\chi^{2}_{n-1} > \mathrm{MSWD}\,(n-1)\right)') + """
    <p>x<sub>i</sub>&nbsp;&plusmn;&nbsp;&sigma;<sub>i</sub> are the analyses. An MSWD
    near 1 means the scatter is explained by the analytical errors; above 1 with a low
    p, there is additional variance; well below 1, the errors are overestimated.</p>

    <h3>Excess scatter (REML)</h3>
    <p>The excess scatter is estimated as a variance &tau;<sup>2</sup> by restricted
    maximum likelihood, which minimises</p>
    """ + E(r'\sum_{i}\ln\left(\sigma_{i}^{2} + \tau^{2}\right) + \ln W^{*} + \sum_{i} w_{i}^{*}\left(x_{i} - \hat{\mu}^{*}\right)^{2}, \qquad w_{i}^{*} = \frac{1}{\sigma_{i}^{2} + \tau^{2}}, \quad W^{*} = \sum_{i} w_{i}^{*}') + \
    E(r'\hat{\mu}^{*} = \frac{\sum_{i} w_{i}^{*} x_{i}}{W^{*}}') + """
    <p>The ln&nbsp;W* term allows for the mean being estimated from the same data, so &tau;<sup>2</sup>
    is not biased low for the few monitors of one position. With
    &tau;<sup>2</sup>&nbsp;=&nbsp;0 the result is the ordinary weighted mean; otherwise
    &tau; is the grain-to-grain spread in units of R. This is the random-effects model of
    GLS + ML (Section&nbsp;4), applied to the monitors.</p>
    <p>The standard error is 1/&radic;W* and does not include the uncertainty of
    &tau;<sup>2</sup>. In simulations with excess scatter as large as the analytical
    errors, &plusmn;1.96&nbsp;SE contained the true mean in 90&ndash;92% of cases for
    5&ndash;20 monitors (95&ndash;96% without excess scatter), against 85&ndash;89% for
    the weighted mean with SE&nbsp;&times;&nbsp;&radic;MSWD. With REML unticked the tool
    reports that weighted mean, with the SE multiplied by &radic;MSWD when the MSWD
    exceeds the Wendt &amp; Carl limit (Section&nbsp;4).</p>

    <h3>J</h3>
    """ + E(r'J = \frac{e^{\lambda t} - 1}{R} \qquad \frac{\sigma_{J}}{J} = \frac{\sigma_{R}}{R}') + """
    <p>t is the monitor age and R the mean monitor ratio. &sigma;<sub>J</sub> contains
    the monitor measurement only. The monitor age and &lambda; are common to every J on
    the calibration and would average down if included in each J, so they are added
    once, in the external tier of the ages (Section&nbsp;3). J is calculated from the
    reported R, so the J view has the same accepted and rejected points, MSWD and p as
    the R analysis.</p>

    <h3>Crystals: fusion and step heating</h3>
    <p>A crystal is identified by a grain-ID column (a blank cell continues the crystal
    above) or, in an .arar file, by the run-ID prefix (100-01A &rarr; 100-01) or by the
    laser power in run order (a step at the same or a lower power than the step before
    starts a new crystal). A crystal with one step is a fusion, one with several was
    step-heated. The points averaged can be every step, one point per crystal,
    or the single-step crystals only. One point per crystal
    is the total gas, the <sup>39</sup>Ar-weighted ratio</p>
    """ + E(r'R_{\mathrm{TG}} = \sum_{i} f_{i} R_{i}, \quad f_{i} = \frac{{}^{39}\mathrm{Ar}_{i}}{\sum {}^{39}\mathrm{Ar}} \qquad \sigma_{\mathrm{TG}} = \sqrt{\sum_{i} f_{i}^{2}\sigma_{i}^{2}}') + """
    <p>which equals what a single fusion of the crystal would have measured, so fused
    and step-heated crystals can be averaged together. Without <sup>39</sup>Ar the
    inverse-variance mean of the steps is used and labelled as such. For each
    step-heated crystal the MSWD of its steps is reported, and the steps can be grouped
    by laser power or step number to compare a weighted mean per group.</p>

    <h3>Outlier rejection</h3>
    <p>By default two filters are applied: a relative-error cut, which removes analyses
    with poor counting statistics, and NMAD rejection of points more than k&nbsp;NMAD
    from the median,</p>
    """ + E(r'\mathrm{NMAD} = 1.4826 \times \mathrm{median}\,\left| x_{i} - \mathrm{median}\right|') + """
    <p>The NMAD is not inflated by several outliers at once, but it uses the values only,
    not their errors. With unequal errors it removes the least precise points first,
    even when they agree within error, and each rejection trims the scatter that
    &tau;<sup>2</sup> measures. On 15 good monitors with errors varying threefold it
    rejects at least one good point in about half of the sets at k&nbsp;=&nbsp;2.5 and
    a third at k&nbsp;=&nbsp;3. The default is k&nbsp;=&nbsp;2.5, as in Age
    distribution.</p>
    <p>MSWD rejection, when ticked, removes one point at a time while the MSWD exceeds
    the Wendt &amp; Carl limit (or a set value). The candidate is the point with the
    largest externally studentized residual (Viechtbauer &amp; Cheung 2010),</p>
    """ + E(r'r_{i} = \frac{x_{i} - \hat{\mu}_{(-i)}}{\sqrt{\sigma_{i}^{2} + \hat{\tau}^{2}_{(-i)} + \mathrm{SE}^{2}_{(-i)}}} \qquad |r_{i}| > \Phi^{-1}\!\left(1 - \frac{0.025}{n}\right)') + """
    <p>where (&minus;i) marks the weighted mean, excess variance and standard error of
    the other points. The point is removed only if |r<sub>i</sub>| exceeds the 5%
    two-sided critical value corrected for n tests (2.81 at n&nbsp;=&nbsp;10, 3.02 at
    n&nbsp;=&nbsp;20). If no point does, the MSWD stays above the limit and the tool
    reports it: the scatter is spread over the whole set and is handled by REML or
    &radic;MSWD. In simulations with 15 points this removed a good point in 1% of sets
    without excess scatter and 15% with &tau;&nbsp;=&nbsp;&sigma; (68% when points are
    removed until the MSWD is below the limit) and found a 6&sigma; outlier every time.
    The filters run in the order relative error, NMAD, MSWD.</p>
    """

    # ── 7 ────────────────────────────────────────────────────────────────
    jcalc = """
    <a name="jcalc"></a>
    <h2>7. Flux regression</h2>

    <p>J is measured only in the monitor holes. The tool fits a surface through the
    monitor values and predicts J, with a Monte Carlo uncertainty, at the position of
    each unknown.</p>
    <p>The monitor values can be typed or pasted into the disk tables, or imported
    from an .xlsx export of the Flux monitors tool (File &rarr; Import Flux monitors
    export). A category named like C1 goes to the disk labelled C, hole 1, with its
    reported R and SE. In J mode, J is calculated from that R with the FC age and
    &lambda; set in this tool.</p>

    """ + F('jcalc_surface.png',
            'A fitted flux surface over the irradiation disk. The monitor holes are '
            'the control points; unknowns are predicted from the surface.') + """

    <h3>Surfaces</h3>
    <table>
      <tr><th>Model</th><th>Surface</th><th>Parameters</th><th>Minimum monitors</th></tr>
      <tr><td>2D planar</td><td>f = ax + by + c</td><td>3</td><td>3</td></tr>
      <tr><td>Simple 3D</td><td>f = ax + by + cz + d</td><td>4</td><td>4</td></tr>
      <tr><td>Complex 3D</td><td>f = ax + by + cz + d + e(xz) + f(yz) + g(xy) + h(xyz)</td><td>8</td><td>8 (12+ preferred)</td></tr>
      <tr><td>Individual</td><td>an independent plane per disk</td><td>3 per disk</td><td>3 per disk</td></tr>
    </table>
    <p>x and y are the hole coordinates (cm) and z the height of the disk in the
    stack.</p>
    <ul>
      <li><b>2D planar</b>: J changes linearly across the disk. For a single disk
          only.</li>
      <li><b>Simple 3D</b>: adds a linear change with height, so every disk has the same
          lateral gradient on top of a gradient along the stack. For stacked disks with
          monitors on at least two levels.</li>
      <li><b>Complex 3D</b>: also lets the lateral gradient change with height (xz, yz)
          and allows a saddle-shaped surface (xy, xyz). Only with many monitors spread
          over several disks; with fewer than about 12, its eight parameters fit the
          noise.</li>
      <li><b>Individual</b>: the 2D plane fitted to each disk separately, with nothing
          shared between disks. For several disks whose gradients are unrelated, with at
          least three monitors on each.</li>
    </ul>
    <p>Use the simplest model the monitors support, and a more complex one only if it
    predicts left-out monitors better (see Choosing a model below).</p>

    <h3>Monte Carlo uncertainty</h3>
    <p>Each iteration moves the monitor positions at random, adds Gaussian noise to
    each monitor J at its stated error, and refits the surface by weighted least
    squares (weights 1/&sigma;<sup>2</sup>). The mean and standard deviation of the
    predictions are the J and &sigma;<sub>J</sub> of each unknown; monitors keep their
    measured values:</p>
    """ + E(r'\hat{J}(x,y,z) = \langle J \rangle_{\mathrm{MC}} \qquad \sigma_{J}(x,y,z) = \mathrm{std}_{\mathrm{MC}}') + """
    <p>The position shift allows for where the monitor crystals lay. A monitor value is
    the mean of n crystals, each anywhere on the floor of a hole of diameter d. A random
    point on a disc has a standard deviation of d/4 in x and in y (its mean squared
    distance from the centre is d<sup>2</sup>/8), so the mean position of n crystals is
    uncertain by</p>
    """ + E(r'\sigma_{\mathrm{pos}} = \frac{d}{4\sqrt{n}}') + """
    <p>in each direction: 0.2&ndash;0.3&nbsp;mm for about ten crystals in the
    2.7&ndash;4.0&nbsp;mm monitor holes of the 13- and 21-hole disks. The standard
    deviation of the shift is drawn in each iteration between 15 and 450&nbsp;&mu;m
    (typically about 0.26&nbsp;mm). Disk heights are not shifted.</p>
    <p>If the monitors scatter about the surface more than their errors allow (weighted
    MSWD above the Wendt &amp; Carl limit for n&nbsp;&minus;&nbsp;k degrees of freedom,
    k the number of parameters), every &sigma;<sub>J</sub> is multiplied by
    &radic;MSWD; the correlations between holes are kept. The MSWD of each fit is
    reported and stored in the <code>.jflux</code> file.</p>

    <h3>Choosing a model</h3>
    <p>A more flexible surface always fits the monitors better, so the models are
    compared by leave-one-out cross-validation error, the AIC, which penalises extra
    parameters, and the reduced &chi;<sup>2</sup>, which is near 1 when the errors are
    right. J and &sigma;<sub>J</sub> can be sent to the dating tools or saved as a
    <code>.jflux</code> file, from which Plateau &amp; isochron and Age distribution
    read the correlation between holes. For a global fit through several disks the file
    also holds the correlation between holes on different disks.</p>
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
            'the fraction of 39Ar released; the highlighted steps form the plateau.') + """

    <h3>Plateau</h3>
    <p>A run of consecutive steps is a plateau when it has at least a set number of
    steps, a set fraction of the <sup>39</sup>Ar, and a &chi;<sup>2</sup> probability
    of the weighted mean above a set value. Steps can also be included or excluded by
    hand. The plateau age is the weighted mean of its steps on the analytical errors,
    with the uncertainty given by the error mode:</p>
    <table>
      <tr><th>Error mode</th><th>Uncertainty</th><th>Use when</th></tr>
      <tr><td>SEM &times; 2</td><td>2&nbsp;&middot;&nbsp;SEM</td><td>MSWD &asymp; 1, no scatter correction</td></tr>
      <tr><td>SEM &times; &radic;MSWD &times; 2</td><td>2&nbsp;&middot;&nbsp;SEM&nbsp;&middot;&nbsp;&radic;MSWD</td><td>most cases</td></tr>
      <tr><td>Student's t</td><td>SEM&nbsp;&middot;&nbsp;&radic;MSWD&nbsp;&middot;&nbsp;t(n&minus;1)</td><td>small n; matches ArArCALC when p &lt; 5%</td></tr>
    </table>
    <p>All steps of one aliquot share the same J, so J is added after the mean, not to
    each step, where it would average down:</p>
    """ + E(r'\sigma_{\mathrm{int}} = \sqrt{\sigma_{\mathrm{ana}}^{2} + \left(\frac{\partial t}{\partial \ln J} f_{J}\right)^{2}} \qquad \sigma_{\mathrm{ext}} = \sqrt{\sigma_{\mathrm{int}}^{2} + \sum_{k} c_{k}^{2}}') + """
    <p>with the c<sub>k</sub> of Section&nbsp;3.</p>

    <h3>Combining aliquots</h3>
    <p>Shared uncertainties do not average down when aliquots are combined; independent
    ones decrease roughly as 1/&radic;n. Whether two aliquots share J depends on where
    they were irradiated:</p>
    <table>
      <tr><th>Aliquots from</th><th>Correlation of J</th><th>Does J average down?</th></tr>
      <tr><td>The same hole</td><td>&rho; = 1</td><td>No</td></tr>
      <tr><td>Different holes, same disk</td><td>0 &lt; &rho; &lt; 1, from the flux model</td><td>Partly</td></tr>
      <tr><td>Different disks, one 3D fit</td><td>from the flux model</td><td>Partly</td></tr>
      <tr><td>Different disks fitted separately, or irradiations</td><td>&rho; = 0</td><td>Yes, J only</td></tr>
      <tr><td>Different laboratories or calibrations</td><td>&rho; = 0; monitor and &lambda; also independent</td><td>J yes; monitor and &lambda; only if they differ</td></tr>
    </table>
    <p>A loaded <code>.jflux</code> file gives the correlation between holes from the
    flux model instead of assuming 0 or 1. The monitor age and &lambda; are common to
    everything on one calibration, so the external tier does not average down even
    across disks. The methods are those of Section&nbsp;4:</p>
    <ul>
      <li><b>WM + √MSWD</b>: on the plateau ages of the aliquots (n = aliquots, the
          default), or with all steps normalised to one J and pooled in ratio space
          (n = steps);</li>
      <li><b>GLS + ML</b>: on the plateau ages with the full covariance and
          &omega;<sup>2</sup> (n = aliquots). Use it when the aliquots come from more
          than one hole or disk, or disagree beyond analytical error.</li>
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
    both axes. The uncertainties are multiplied by &radic;MSWD when the MSWD exceeds 1,
    as in ArArCALC.</p>
    """

    # ── 9 ────────────────────────────────────────────────────────────────
    prob = """
    <a name="prob"></a>
    <h2>9. Age distribution</h2>

    <p>For a population of single-crystal or single-aliquot ages: the ranked ages with
    a probability density curve, a weighted mean of each sample, pooled ages of several
    samples, and mixture and minimum-age models.</p>

    """ + F('prob_plot.png',
            'Ranked ages with 1-sigma bars and the probability density curve. Excluded '
            'points are open markers; the dashed curve includes them.') + """

    <h3>Probability density and weighted mean</h3>
    <p>Each accepted age adds a Gaussian of unit area with its own &sigma;, and the
    curve is their sum. It displays the data and is not a fitted model. The weighted
    mean and MSWD use the analytical errors:</p>
    """ + E(r'\bar{t} = \frac{\sum w_{i} t_{i}}{\sum w_{i}}, \quad \mathrm{MSWD} = \frac{1}{n-1}\sum w_{i}\left(t_{i} - \bar{t}\right)^{2}, \quad w_{i} = \frac{1}{\sigma_{\mathrm{ana},i}^{2}}') + """
    <p>Outliers can be removed by NMAD, by iterative 2&sigma; rejection, or by clicking
    points. A dashed curve shows the distribution with the excluded points added
    back.</p>

    <h3>Pooling samples</h3>
    <p>Samples are pooled on the covariance of Vermeesch (2015):</p>
    """ + E(r'\Sigma_{ij} = a_{i}^{2}\,\delta_{ij} + \rho^{\mathrm{pos}}_{ij}\,p_{i} p_{j} + \rho^{\mathrm{std}}_{ij}\,s_{i} s_{j}') + \
    E(r'a = \sigma_{\mathrm{ana}} \qquad p = \sqrt{\sigma_{\mathrm{int}}^{2} - \sigma_{\mathrm{ana}}^{2}} \qquad s = \sqrt{\sigma_{\mathrm{ext}}^{2} - \sigma_{\mathrm{int}}^{2}}') + """
    <p>a is the analytical term, p the flux term and s the monitor and decay-constant
    term. &rho;<sup>pos</sup> is 1 for samples from the same irradiation position and 0
    otherwise; &rho;<sup>std</sup> is 1 for samples on the same calibration. The flux
    term therefore averages down only between positions, and the calibration term never
    does. Samples with matching calibration are grouped automatically, and the report
    lists the groups. The methods are those of Section&nbsp;4.</p>

    <h3>Mixture view</h3>
    <p><b>Mixture</b> replaces the plot with a mixture model of the sample chosen above
    it; the samples are not changed, and the button switches back. The analyses
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
    <p>as the fewest components within 2 of the lowest BIC; a smaller difference is not
    considered evidence (Kass &amp; Raftery 1995). Up to ten components are tried,
    since each old xenocryst can form its own. A component of at least two analyses
    has an MSWD, with each analysis weighted by p<sub>ij</sub>,</p>
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
    own: one discrete youngest age plus a truncated spread of older analyses. It
    assumes no analysis is too young, so a young outlier (argon loss) becomes the
    minimum, and old xenocrysts widen the spread. Run it on data without them.</p>
    <p>The BIC also splits a single population whose scatter exceeds its analytical
    errors, and the number of components it finds grows with the number of analyses. A
    split is therefore a reason to look at the data, not proof of two populations.
    The youngest component is not necessarily the eruption age; a single
    blank-dominated analysis can form a component of its own. The radial plot
    (Galbraith 1988) helps with both: each analysis is plotted at its precision,
    1/&sigma;, against its deviation from the weighted mean in units of &sigma;. Its
    age is read where the line from the origin through the point meets the arc, and
    each component is such a line. The &Delta;BIC panel shows how clearly the number of
    components was chosen.</p>
    """ + F('prob_mixture.png',
            'The Mixture view. Left: the ranked analyses and the probability curve split '
            'into components, each analysis coloured by its most likely component. '
            'Right: the radial plot and the BIC of each number of components.') + """
    <p>While the view is on, both export buttons export the mixture model of the sample
    shown. <b>Export mixture figure</b> uses the window layout (diagram left, radial
    plot and &Delta;BIC right) or stacks the panels. <b>Export mixture XLSX / CSV</b>
    writes the component ages with their three tiers, proportions and MSWD, then every
    analysis the model used with its data and its probability for each component. The
    settings are saved with the session.</p>
    """

    # ── 10 ───────────────────────────────────────────────────────────────
    calib = """
    <a name="calib"></a>
    <h2>10. Calibrations and constants</h2>

    <p>The default calibration is Phillips et&nbsp;al. (2022): Fish Canyon
    28.176&nbsp;&plusmn;&nbsp;0.0115&nbsp;Ma (1&sigma;; published as
    &plusmn;0.023&nbsp;Ma at 2&sigma;) with the Min et&nbsp;al. (2000) decay constant,
    5.463&nbsp;&plusmn;&nbsp;0.054&nbsp;&times;&nbsp;10<sup>&minus;10</sup>&nbsp;yr<sup>&minus;1</sup>.
    Ages are calculated with the two-step equation of Section&nbsp;3 (A1 tephra,
    6.943&nbsp;Ma; R<sub>FCs/A1</sub>&nbsp;=&nbsp;4.0819), so the external tier comes
    from R<sub>FCs/A1</sub>, the A1 age and &lambda;, not from the quoted Fish Canyon
    error. The other presets, or custom values, can be chosen in each tool:</p>

    <table>
      <tr><th>Monitor-age preset</th><th>Age (Ma)</th><th>1&sigma; (Ma)</th><th>External &lambda;</th></tr>
      <tr><td>Carter et al. (2025)</td><td>28.183</td><td>0.017</td><td>Yes</td></tr>
      <tr><td>Phillips et al. (2022)</td><td>28.176</td><td>0.0115</td><td>Yes</td></tr>
      <tr><td>Rivera et al. (2011)</td><td>28.172</td><td>0.014</td><td>Yes</td></tr>
      <tr><td>Kuiper et al. (2008)</td><td>28.201</td><td>0.023</td><td>Yes</td></tr>
      <tr><td>Renne et al. (1998)</td><td>28.02</td><td>0.14</td><td>No</td></tr>
      <tr><td>Jourdan &amp; Renne (2007)</td><td>28.03</td><td>0.08</td><td>No</td></tr>
    </table>

    <p>The astronomically calibrated ages (Kuiper 2008, Rivera 2011, Phillips 2022)
    were calculated with the Min et&nbsp;al. (2000) &lambda;, and the tools pair them
    with it. Another &lambda; can be chosen; the two-step equation then gives the Fish
    Canyon age the anchor implies at that &lambda;, not the published value.</p>
    <p>The External &lambda; setting matters only for calibrations without an anchor.
    "Yes" means the published monitor-age uncertainty already contains the &lambda;
    uncertainty (a jointly calibrated age), so it is not added again. For Carter
    et&nbsp;al. (2025) it also includes the &lambda;&ndash;monitor covariance, which
    reproduces their Table&nbsp;9.</p>

    <h3>Carter et al. (2025): three uncertainties</h3>
    <p>Carter's Fish Canyon age is quoted with three different 1&sigma;
    uncertainties:</p>
    <table>
      <tr><th>1&sigma; (Ma)</th><th>Meaning</th></tr>
      <tr><td>0.0112</td><td>Posterior precision of the fit.</td></tr>
      <tr><td>0.017</td><td>Full external systematic, including the input-data systematics. Used as the external tier here.</td></tr>
      <tr><td>0.035</td><td>Total for dating, adding the R-value and neutron-fluence terms.</td></tr>
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
            + converter + wm + jcalc + plateau + prob + calib + refs)
