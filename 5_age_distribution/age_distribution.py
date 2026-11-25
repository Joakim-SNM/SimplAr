#!/usr/bin/env python3
"""
Age distribution: probability density plots, pooled ages and mixture models
for many ⁴⁰Ar/³⁹Ar ages.

  * one or many samples, each pasted, loaded (.arar/.ararpt) or imported (.xlsx/.csv);
  * input as ages (Ma) or as 40Ar*/39ArK ratios, converted with a J calibration
    per sample;
  * a summed-Gaussian probability density per sample, and a pooled curve;
  * pooling on the covariance of Vermeesch (2015): samples from one irradiation
    position share their J uncertainty, which then does not average down, and
    samples on one calibration share the FCs age and λ terms;
  * ages without J information are plotted and combined as they are;
  * a Mixture view that fits a discrete mixture or a minimum age model to the
    analyses of one sample (Galbraith & Laslett 1993; Galbraith et al. 1999);
    its figure export can also stack every sample, each with its own model;
  * figure export with the samples in separate panels, and a normalised
    y axis that brings every sample's peak to 100 %.

Layout of this file, top to bottom:
  J configuration defaults, statistics, plot helpers
  parsing (.arar / .ararpt / .ararstack / tables)
  Sample (data model) and SamplePanel (per-sample controls)
  disk-flux dialog host
  ProbabilityWindow: UI and calibration bar, loading, axes and options,
    disk flux, pooling, redraw, Mixture view, session save/load, export

"""
import sys
import os
import json
import copy
import math

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QTextEdit, QFrame,
    QScrollArea, QMessageBox, QSizePolicy, QSpinBox,
    QListWidget, QListWidgetItem,
    QDialog, QStackedWidget, QColorDialog,
)
from PyQt5.QtCore import Qt  # noqa: F401

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False

# shared math/constants live one folder up, next to the launcher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arar_math as am
from arar_math import (
    LAMBDA_DEFAULT, LAMBDA_ERR, STD_AGE_DEFAULT, STD_AGE_ERR,
    weighted_mean, j_correlation, match_flux_hole, mswd_classify,
    cfg_j_err_frac, r_to_age_cfg, dt_dlnR,
    rr_from_age, fsig, ff,
    FC_PRESETS, paired_lambda,
    external_sigma_array, calibration_class, age_error_budget,
    internal_sigma_array, systematic_matrix, cfg_ingrowth,
    external_route, calibration_group_key,
    full_systematics_warning, calibration_warning, force_light, no_wheel_on_dropdowns,
    cfg_J, scale_sem, position_ident, combine, read_jflux,
    deserialize_disk_flux, merge_disk_flux, flux_hole_R, flux_hole_rel, flux_hole_rstd,
    flux_pair_corr, serialize_xcorr,
    draw_budget_pie, file_err_div,
    METHOD_WM, METHOD_GLS, method_key,
    mono_font, to_fcs_r, from_fcs_r, cfg_monitor,
    calibration_notes, monitor_mismatch_note,
)
from flux_dialog import FluxAssignDialog, FluxHost
from standards import MonitorCombo, monitor_text, route_line
from calibration_dialog import CalibrationDialog
from arar_math import (route_cfg, route_from_cfg, route_text, set_route, ROUTE_KEYS,
                       CALIB_KEYS)
import export
import data_folders as dfold


_PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e',
            '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
_POOL_COLOR = '#000000'

# Mapping between internal ageFluxMode keys and UI display labels.
# Internal keys are used everywhere in logic/storage; display labels only in the combo box.
_JMODE_DISPLAY = {'imported':    'imported',
                  'ratio_J':     'J value',
                  'ratio_R_std': 'R (standard)',
                  'ratio_R_FCs': 'R(unk) / R(std)'}
_JMODE_INTERNAL = {v: k for k, v in _JMODE_DISPLAY.items()}


# ---------------------------------------------------------------------------
# J configuration defaults and key generation
# ---------------------------------------------------------------------------
def _default_j_cfg():
    return dict(
        ageFluxMode='imported', rStd=70.0, rStdErrPct=0.10, rStdErrMode='pct',
        jVal=0.005, jErrPct=0.10, jErrMode='pct',
        stdAgeMa=STD_AGE_DEFAULT, stdAgeErrMa=STD_AGE_ERR, stdAgeErrType='external',
        lambdaYr=LAMBDA_DEFAULT, lambdaErrYr=LAMBDA_ERR,
        rfcsErrPct=0.0, pasteTier='analytical',
    )


def _sync_j_rstd(cfg, master):
    """Keep J and R_std as two views of one calibration so switching the J-mode
    representation never changes the age:  J = (e^{lambda * t_FCs} - 1) / R_std.

    master='J' -> derive R_std from jVal;  master='R' -> derive jVal from R_std.
    Since J is proportional to 1/R_std, their relative 1-sigma errors are equal,
    so the percent uncertainty is mirrored too. (The t_FCs and lambda systematics
    are handled separately in the external tier, not here.)
    """
    em1 = cfg_ingrowth(cfg)          # anchored chain where the preset has one
    if master == 'J':
        J = cfg.get('jVal', 0.0)
        if J > 0:
            cfg['rStd'] = em1 / J
            cfg['rStdErrPct'] = cfg.get('jErrPct', 0.0)
    else:
        R = cfg.get('rStd', 0.0)
        if R > 0:
            cfg['jVal'] = em1 / R
            cfg['jErrPct'] = cfg.get('rStdErrPct', 0.0)


def _j_flux_key(cfg):
    """Hashable key identifying a J/R calibration (= an irradiation position).

    Samples with the same key share their J uncertainty fully (rho = 1) and it
    must not average down when pooling. Mirrors PlateauWindow._flux_key().
    """
    mode = cfg.get('ageFluxMode', 'imported')
    if mode == 'ratio_J':
        return ('ratio_J', round(cfg.get('jVal', 0.005), 10),
                round(cfg.get('jErrPct', 0.0), 8), cfg.get('jErrMode', 'pct'))
    if mode == 'ratio_R_std':
        return ('ratio_R_std', round(cfg.get('rStd', 70.0), 6),
                round(cfg.get('rStdErrPct', 0.10), 6), cfg.get('rStdErrMode', 'pct'),
                round(cfg.get('stdAgeMa', STD_AGE_DEFAULT), 6),
                round(cfg.get('stdAgeErrMa', STD_AGE_ERR), 8),
                cfg.get('stdAgeErrType', 'internal'),
                round(cfg.get('lambdaYr', LAMBDA_DEFAULT), 17),
                round(cfg.get('lambdaErrYr', LAMBDA_ERR), 20))
    if mode == 'ratio_R_FCs':
        # rfcsErrPct and pasteTier affect error tiers, not the position identity.
        # The position grouping has no numerical effect in this mode (p=0), so
        # the key only needs to identify the t_FCs/λ calibration used.
        return ('ratio_R_FCs', cfg_monitor(cfg)[0], round(cfg.get('stdAgeMa', STD_AGE_DEFAULT), 6),
                round(cfg.get('stdAgeErrMa', STD_AGE_ERR), 8),
                cfg.get('stdAgeErrType', 'internal'),
                round(cfg.get('lambdaYr', LAMBDA_DEFAULT), 17),
                round(cfg.get('lambdaErrYr', LAMBDA_ERR), 20))
    return ('imported',)


def _flux_J(cfg):
    """The J a flux config turns R into an age with (arar_math.cfg_J), or None."""
    return cfg_J(cfg)[0]


def _rescale_to_J(doc, J_from, J_to):
    """Put a parsed doc's R values (and their σ) on another J: R·J_from/J_to,
    the Koppers (2002) normalisation.  ln(1 + J·R)/λ is unchanged, so aliquots
    from different irradiation positions can share one sample calibration
    without their ages moving.  No-op when either J is unknown."""
    if not J_from or not J_to or not doc.get('r_vals'):
        return
    f = J_from / J_to
    if abs(f - 1.0) < 1e-12:
        return
    # keep the measured values: the export reports them with the doc's own J
    doc.setdefault('r_orig', list(doc['r_vals']))
    doc.setdefault('r_orig_unc', list(doc['r_uncs']))
    doc['r_vals'] = [r * f if r is not None else None for r in doc['r_vals']]
    doc['r_uncs'] = [u * f if u is not None else None for u in doc['r_uncs']]


def _join_names(names, limit=2):
    """Several experiment names as one: 'A + B', or 'A + 4 more' past
    *limit*, so a legend entry stays readable.  Repeats are dropped."""
    out = []
    for n in names:
        n = str(n or '').strip()
        if n and n not in out:
            out.append(n)
    if len(out) <= limit:
        return ' + '.join(out)
    return f'{out[0]} + {len(out) - 1} more'


def _fine_position(disk, hole, j_cfg, fallback):
    """Hole-level label of an irradiation position ('E/E17'), used only to
    tell whether a merged sample spans several positions and to give each its
    own J error; how strongly two positions correlate is j_correlation's job."""
    if disk and hole:
        return f'{disk}/{hole}'
    return position_ident(disk, hole, None, j_cfg) or fallback


def _doc_positions(docs):
    """Per-point (σ_J/J, irradiation position) for docs merged into one sample,
    or (None, None) when every point sits on one position.  Points on the first
    point's position get NaN: they follow the sample's own J error field."""
    jf, pos = [], []
    for d in docs:
        n = len(d['ages'])
        if d.get('pt_pos') is not None and len(d['pt_pos']) == n:
            jf += list(d['pt_jfrac'])
            pos += list(d['pt_pos'])
        else:
            p = _fine_position(d.get('disk_id'), d.get('hole_id'), d['j_cfg'],
                               str(_j_flux_key(d['j_cfg'])))
            jf += [cfg_j_err_frac(d['j_cfg'])] * n
            pos += [p] * n
    if len(set(pos)) < 2:
        return None, None
    jf = np.array([np.nan if p == pos[0] else f for p, f in zip(pos, jf)], float)
    return jf, pos


# ---------------------------------------------------------------------------
# Statistics: weighted mean, outlier rejection, MSWD
# ---------------------------------------------------------------------------


def iterative_2sigma_rejection(ages, sigma_1, max_iter=50):
    mask = np.ones(len(ages), dtype=bool)
    for _ in range(max_iter):
        if mask.sum() < 3:
            break
        r = weighted_mean(ages[mask], sigma_1[mask])
        wm, mswd = r['mean'], r['mswd']
        scale = max(1.0, math.sqrt(mswd)) if mswd == mswd else 1.0
        new = mask.copy()
        for i in np.where(mask)[0]:
            if abs(ages[i] - wm) > 2.0 * sigma_1[i] * scale:
                new[i] = False
        if np.array_equal(new, mask):
            break
        mask = new
    return mask


def nmad_rejection(ages, threshold=2.5):
    med  = np.median(ages)
    nmad = 1.4826 * np.median(np.abs(ages - med))
    if nmad == 0:
        return np.ones(len(ages), dtype=bool)
    return np.abs(ages - med) <= threshold * nmad


# ---------------------------------------------------------------------------
# Plot helpers: summed-Gaussian PDF and axis grid
# ---------------------------------------------------------------------------
def summed_gaussian_pdf(ages, sigma_1, axis):
    pdf = np.zeros_like(axis)
    for a, s in zip(ages, sigma_1):
        pdf += np.exp(-0.5 * ((axis - a) / s)**2) / (s * math.sqrt(2.0 * math.pi))
    return pdf


def _curve_axis(inc, inc_s, ds):
    """A padded x-axis for a summed-Gaussian curve. Shared by the density
    pre-pass and the draw loop so both evaluate on the identical grid.
    The curves are always drawn from the 1σ uncertainties; ds only widens
    the padding so the display-level error bars fit."""
    inc = np.asarray(inc, float)
    inc_s = np.asarray(inc_s, float)
    rng = max(inc.max() - inc.min(), 0.01)
    pad = max(6.0 * inc_s.max() * ds, 0.3 * rng, 1.0)
    lo = inc.min() - pad
    hi = inc.max() + pad
    # When one broad analysis blows up the axis range, a fixed 3000-point grid
    # under-samples tight analyses (a few-point Gaussian → triangle).  Scale
    # n_pts so the narrowest sigma always gets ≥20 points for smooth rendering.
    sig_min = max(float(inc_s.min()), 1e-9)
    n_pts = max(3000, min(int((hi - lo) / sig_min * 20) + 1, 30000))
    return np.linspace(lo, hi, n_pts)


# Active statistics method for the pool/weighted-mean: METHOD_WM (default,
# analytical weights + sqrt(MSWD)) or METHOD_GLS
# (GLS weighted mean + ML random-effects overdispersion, Vermeesch 2015 /
# Ar-Ar_Redux / IsoplotR).  Set by the window's method selector.
_STAT_METHOD = METHOD_WM


def _combine(ages, sigma_ana, sigma_int=None, sigma_ext=None, **kw):
    """The weighted-mean combine of the active statistics method."""
    return combine(_STAT_METHOD, ages, sigma_ana, sigma_int, sigma_ext, **kw)

# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def _metas_of(d, n):
    """Per-step export metadata (run time, days since irradiation, raw
    signals) of a parsed dict, padded with empty entries when absent."""
    m = d.get('metas') or []
    ms = [dict(x) for x in m] if len(m) == n else [export.step_meta({}) for _ in range(n)]
    # Each point's own J / R standard and measured 40Ar*/39ArK. Several files or
    # aliquots can be merged into one sample on the first one's J (R rescaled),
    # so the export needs these to report what each point was measured with.
    jc = d.get('j_cfg') or {}
    rv = d.get('r_orig', d.get('r_vals')) or []
    ru = d.get('r_orig_unc', d.get('r_uncs')) or []
    has_r = len(rv) == n
    try:
        show_r = bool(has_r and float(jc.get('rStd', 0) or 0) > 0
                      and jc.get('ageFluxMode') in ('ratio_J', 'ratio_R_std'))
    except (TypeError, ValueError):
        show_r = False
    fx = export.flux_cells(jc, show_r=show_r)
    pos = _pos_info(d)
    for i, x in enumerate(ms):
        x.setdefault('fx', fx)
        x.setdefault('pos', pos)
        if has_r:
            x.setdefault('r_meas', rv[i])
            x.setdefault('r_meas_unc', ru[i] if len(ru) == n else None)
    return ms


def _pos_info(d):
    """Where a parsed dict was irradiated: its disk / hole / irradiation, the
    hole the Plateau & isochron tool had it assigned to (if any) and its calibration key.
    This is what the J covariance between analyses is worked out from."""
    fh = d.get('flux_hole')
    fh = [str(x) for x in fh] if isinstance(fh, (list, tuple)) and len(fh) == 3 else None
    irr = d.get('irradiation') or (d.get('constants') or {}).get('irradiation') or ''
    jc = d.get('j_cfg') or {}
    return dict(disk=d.get('disk_id'), hole=d.get('hole_id'), irr=str(irr),
                fh=fh, key=position_ident(None, None, None, jc) or repr(_j_flux_key(jc)))


_J_CFG_KEYS = ('ageFluxMode', 'jVal', 'jErrPct', 'rStd', 'rStdErrPct', 'rStdErrMode',
               'stdAgeMa', 'stdAgeErrMa', 'stdAgeErrType', 'lambdaYr', 'lambdaErrYr',
               'fullSystematics') + tuple(am.ROUTE_KEYS)


def _parse_steps(steps, ef):
    """Per-step lists of a saved session; measured errors × ef → 1σ.

    Never drop data.  "Analysis Included = No" (status 'excluded')
    is loaded as a manual reject, present and recoverable (click the run, or
    it follows the reject mode if you set the step back to 'auto')."""
    out = dict(ages=[], unc=[], r_vals=[], r_uncs=[], manual=[], kca_vals=[],
               kca_errs=[], ar40_pct=[], labels=[], run_ids=[], metas=[])
    for s in steps:
        age, err = s.get('age'), s.get('err2s')
        if age is None or err is None:
            continue
        out['ages'].append(float(age))
        out['unc'].append(float(err) * ef)
        r, re = s.get('r4039'), s.get('err_r4039')
        out['r_vals'].append(float(r) if r is not None else None)
        out['r_uncs'].append(float(re) * ef if re is not None else None)
        cak, cak_e, ar40 = s.get('caK'), s.get('err_cak'), s.get('pct40')
        out['kca_vals'].append(float(cak) if cak is not None else float('nan'))
        out['kca_errs'].append(float(cak_e) * ef if cak_e is not None else float('nan'))
        out['ar40_pct'].append(float(ar40) if ar40 is not None else float('nan'))
        out['labels'].append(str(s.get('label', '')))
        out['run_ids'].append(str(s.get('runID', '')))
        out['metas'].append(export.step_meta(s))
        out['manual'].append('out' if s.get('status') == 'excluded' else 'auto')
    rv, ru = out['r_vals'], out['r_uncs']
    if not (len(rv) == len(out['ages']) and rv
            and all(x is not None for x in rv) and all(x is not None for x in ru)):
        out['r_vals'], out['r_uncs'] = [], []
    return out


def _file_j_cfg(cfg_src, steps):
    """The J calibration of a saved session (.arar or one stack aliquot).

    The calibration keys are copied and the steps' J values set jVal / jErrPct.
    A file on imported ages is read from its R values (this tool computes every
    age from R): with the file's R standard when it has one, the measured
    monitor ratio, which is valid at whatever calibration the tool is on; only
    without it with the stored J, which belongs to the calibration the file's
    software used and so is right only when the cfg carries that λ and FCs age
    (a converter file whose J is on Carter while its cfg names Phillips would
    come out 0.75% too old)."""
    j_cfg = _default_j_cfg()
    for k in _J_CFG_KEYS:
        if cfg_src.get(k) is not None:
            j_cfg[k] = cfg_src[k]
    jvals = [float(s['j_val']) for s in steps if s.get('j_val')]
    if jvals:
        j_cfg['jVal'] = float(np.mean(jvals))
    jerrs = [float(s['j_err']) for s in steps if s.get('j_err')]
    if jerrs and j_cfg['jVal'] > 0:
        j_cfg['jErrPct'] = float(np.mean(jerrs)) / j_cfg['jVal'] * 100.0
    if j_cfg.get('ageFluxMode', 'imported') == 'imported':
        r_std = float(cfg_src.get('rStd', 0) or 0)
        if r_std > 0 and r_std != 70.0:          # 70 = old converter placeholder
            j_cfg['ageFluxMode'] = 'ratio_R_std'
            _sync_j_rstd(j_cfg, 'R')
        elif jvals:
            j_cfg['ageFluxMode'] = 'ratio_J'
    elif j_cfg['ageFluxMode'] == 'ratio_R_std':
        _sync_j_rstd(j_cfg, 'R')
    return j_cfg


def _err_factor(cfg):
    """Factor turning a file's stored step errors into 1σ (arar_math.file_err_div,
    the rule every tool uses)."""
    return 1.0 / file_err_div(cfg)


def parse_arar(path):
    """Parse one .arar/.ararpt file -> dict(name, ages, unc, r_vals, r_uncs, j_cfg).
    Errors come back at 1σ."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    cfg_src = data.get('cfg', {})
    steps = data.get('steps', [])
    st = _parse_steps(steps, _err_factor(cfg_src))
    if not st['ages']:
        return None
    name = data.get('title', '') or os.path.splitext(os.path.basename(path))[0]
    return dict(name=name, raw_name=data.get('orig_title') or name,
                **st, j_cfg=_file_j_cfg(cfg_src, steps),
                disk_id=cfg_src.get('disk_id'), hole_id=cfg_src.get('hole_id'),
                irradiation=cfg_src.get('irradiation') or cfg_src.get('irrad'),
                flux_hole=data.get('flux_hole'),
                constants=dict(cfg_src.get('constants') or {}))


def _sample_from_dict(sd):
    """A Sample from its entry in an .arprob session."""
    s = Sample(sd.get('name', 'Sample'), sd.get('color', '#1f77b4'))
    s.raw_name = sd.get('raw_name')
    s.enabled = sd.get('enabled', True)
    s.is_plateau = sd.get('is_plateau', False)
    s.input_mode = sd.get('input_mode', 'age')
    s.reject = sd.get('reject', 'nmad')
    s.nmad_thresh = sd.get('nmad_thresh', 2.5)
    s.in_sigma = sd.get('in_sigma', 1)
    s._from_file = sd.get('from_file', False)
    s.disk_id = sd.get('disk_id')
    s.hole_id = sd.get('hole_id')
    s.irradiation = sd.get('irradiation')
    ov = sd.get('flux_override')
    s._flux_override = tuple(ov) if ov else None
    s.j_cfg = {**_default_j_cfg(), **sd.get('j_cfg', {})}
    s.ages = np.array(sd.get('ages', []), float)
    s.unc_2s = np.array(sd.get('unc_2s', []), float)
    s.r_vals = np.array(sd.get('r_vals', []), float)
    s.r_uncs = np.array(sd.get('r_uncs', []), float)
    s.kca_vals = np.array(sd.get('kca_vals', []), float)
    s.kca_errs = np.array(sd.get('kca_errs', []), float)
    s.ar40_pct = np.array(sd.get('ar40_pct', []), float)
    s.labels   = sd.get('labels', [])
    s.run_ids  = sd.get('run_ids', [])
    s.metas    = sd.get('metas', [])
    s.constants = sd.get('constants', {}) or {}
    man = sd.get('manual', [])
    s.manual = np.array(man if len(man) == len(s.ages)
                        else ['auto'] * len(s.ages), dtype=object)
    if sd.get('pt_pos') and sd.get('pt_jfrac'):
        s.pt_pos = [str(p) for p in sd['pt_pos']]
        s.pt_jfrac = np.array([np.nan if v is None else v
                               for v in sd['pt_jfrac']], float)
    # Sessions saved while an assigned hole's J sat on top of the merged
    # aliquots' file-J scaling: age them again from the measured R.
    _ov = s._flux_override
    if (_ov and (str(_ov[0]), str(_ov[1])) != ('NONE', 'NONE')
            and s.has_R() and s.use_measured_R()):
        s.recompute()
    le = sd.get('loaded_excluded', [])
    s._loaded_excluded = np.array(le, bool) if len(le) == len(s.ages) \
        else np.zeros(len(s.ages), bool)
    s.compute_auto()
    return s


def _arprob_doc(sd):
    """A saved .arprob sample as a parse_arar-like dict, to merge it into
    another sample.  Ages, R and their σ are stored at 1σ; R goes with it
    only for a sample that ages from R."""
    n = len(sd.get('ages', []))
    rv, ru = sd.get('r_vals', []) or [], sd.get('r_uncs', []) or []
    use_r = sd.get('input_mode') == 'ratio' and len(rv) == n
    d = dict(name=sd.get('name', 'Sample'), raw_name=sd.get('raw_name'),
             ages=list(sd.get('ages', [])), unc=list(sd.get('unc_2s', [])),
             r_vals=list(rv) if use_r else [],
             r_uncs=(list(ru) if len(ru) == n else [0.0] * n) if use_r else [],
             manual=(list(sd['manual']) if len(sd.get('manual') or []) == n
                     else ['auto'] * n),
             j_cfg={**_default_j_cfg(), **(sd.get('j_cfg') or {})},
             irradiation=sd.get('irradiation'), disk_id=sd.get('disk_id'),
             hole_id=sd.get('hole_id'),
             constants=sd.get('constants') or {})
    for k in ('kca_vals', 'kca_errs', 'ar40_pct', 'labels', 'run_ids', 'metas'):
        v = sd.get(k) or []
        if len(v) == n:
            d[k] = list(v)
    if sd.get('pt_pos') and sd.get('pt_jfrac') and len(sd['pt_pos']) == n:
        d['pt_pos'] = [str(x) for x in sd['pt_pos']]
        d['pt_jfrac'] = [np.nan if v is None else v for v in sd['pt_jfrac']]
    return d


def read_stack_disk_flux(path):
    """The disk-flux registry saved inside an .ararstack (may be empty)."""
    with open(path, encoding='utf-8') as f:
        state = json.load(f)
    return deserialize_disk_flux(state.get('disk_flux') or {})


def parse_ararstack(path):
    """Parse an .ararstack file → list of parse_arar-compatible dicts, one per aliquot.

    Each dict has the same keys as parse_arar returns.  Steps with
    status='excluded' (user-rejected in the Plateau & isochron tool) are loaded as
    manual rejects.  Aliquots flagged as excluded at the session level are
    skipped entirely.  When the stack stores the aliquot's plateau window
    ('pset'), the steps outside it are loaded as manual rejects too, so the
    steps used are the Plateau & isochron tool's plateau ('has_window' True).  An older
    stack without it loads every non-excluded step ('has_window' False) and
    NMAD rejection filters the non-plateau steps.
    """
    with open(path, encoding='utf-8') as f:
        state = json.load(f)
    if state.get('format') != 'ararstack_v1':
        raise RuntimeError(
            'Not a valid .ararstack file (expected ararstack_v1 format).')

    global_cfg = state.get('global_cfg', {})
    results = []
    for idx, sd in enumerate(state.get('sessions', [])):
        if sd.get('excluded', False):
            continue                          # aliquot-level exclusion
        cfg = sd.get('cfg', {})
        merged = dict(global_cfg); merged.update(cfg)   # session overrides global
        steps = sd.get('steps', [])
        pset = (sd.get('plateau') or {}).get('pset')
        if pset:
            keep = {int(i) for i in pset}
            steps = [s if i in keep else dict(s, status='excluded')
                     for i, s in enumerate(steps)]
        st = _parse_steps(steps, _err_factor(cfg))
        if not st['ages']:
            continue
        name = str(sd.get('title', '')) or f'Aliquot {idx + 1}'
        results.append(dict(
            name=name, raw_name=str(sd.get('orig_title') or '') or name,
            has_window=bool(pset),
            **st, j_cfg=_file_j_cfg(merged, steps),
            disk_id=merged.get('disk_id'), hole_id=merged.get('hole_id'),
            irradiation=merged.get('irradiation') or merged.get('irrad'),
            flux_hole=_stack_flux_hole(state, sd, idx),
            constants=dict(merged.get('constants') or {}),
        ))
    return results


def _stack_flux_hole(state, sd, idx):
    """The (irradiation, disk, hole) a Plateau stack had session *idx* assigned
    to: the session's own saved assignment, else the stack's override list.
    A session set to no hole ('none') gives ['NONE', 'NONE', 'NONE'], so it
    keeps its own J here too instead of being matched by its disk / hole;
    automatic matching gives None.  Older stacks wrote 'none' as the letters
    ['n', 'o', 'n', 'e']."""
    fh = sd.get('flux_hole')
    if not (isinstance(fh, (list, tuple)) and len(fh) == 3):
        fh = (state.get('flux_override') or {}).get(str(idx))
    if fh == 'none' or (isinstance(fh, (list, tuple)) and ''.join(map(str, fh)) == 'none'):
        return ['NONE', 'NONE', 'NONE']
    if isinstance(fh, (list, tuple)) and len(fh) == 3:
        return [str(x) for x in fh]
    return None


def parse_ararstack_aliquots(path):
    """Parse an .ararstack → one parse_arar-compatible dict per aliquot, each a
    single R-based point: the ratio-space weighted mean of that aliquot's
    plateau ⁴⁰Ar*/³⁹Ar (R_unk), on the aliquot's own calibration (its R_std, or
    the J its plateau age was computed with, and that J's precision).  No age is
    baked in, the age is computed from R, so it reproduces the Plateau & isochron tool's
    plateau age exactly and follows any later change of calibration or flux.

    Every aliquot is returned; the ones excluded at the session level (the
    ×-ticked ones in the Plateau stack) carry excluded=True.  The plateau step
    window is read from the saved plateau summary ('pset'); an older stack
    without it falls back to all non-excluded steps (re-save the stack from the
    Plateau & isochron tool to get the exact plateau window).
    """
    with open(path, encoding='utf-8') as f:
        state = json.load(f)
    if state.get('format') != 'ararstack_v1':
        raise RuntimeError(
            'Not a valid .ararstack file (expected ararstack_v1 format).')

    global_cfg = state.get('global_cfg', {})
    out = []
    for idx, sd in enumerate(state.get('sessions', [])):
        cfg = {**global_cfg, **sd.get('cfg', {})}
        _ef = _err_factor(cfg)                # → 1σ
        steps = sd.get('steps', [])
        # A stack saved by the current Plateau & isochron tool stores every aliquot's
        # plateau, None where it has none: that aliquot does not count in the
        # stack, so it comes through excluded.  Older stacks have no 'plateau'
        # key at all; there every non-excluded step is used.
        no_plateau = 'plateau' in sd and not sd['plateau']
        plat  = sd.get('plateau') or {}
        pset  = plat.get('pset')
        # plateau steps: the saved window if present, else all non-excluded
        if pset:
            sel = [steps[i] for i in pset if 0 <= i < len(steps)]
        else:
            sel = [s for s in steps if s.get('status') != 'excluded']
        R_list, Re_list = [], []
        for s in sel:
            r, re = s.get('r4039'), s.get('err_r4039')
            if r is None or float(r) <= 0:
                continue
            R_list.append(float(r))
            Re_list.append(float(re) * _ef if re else 0.0)
        if not R_list:
            continue
        if len(R_list) == 1:
            mR, sR = R_list[0], Re_list[0]
        else:
            wm = weighted_mean(np.asarray(R_list), np.asarray(Re_list))
            mR = wm['mean']
            # Scale the ratio SEM with the same rule the Plateau & isochron tool uses for
            # its analytical tier (arar_math.scale_sem with the stack's errMode),
            # so the per-aliquot 1σ, and hence the combined age and MSWD, match
            # the Plateau stack.
            sR = scale_sem(wm['sem'], wm['mswd'], wm['prob'], wm['n'],
                           cfg.get('errMode', 'sem_mswd_2s'),
                           float(cfg.get('pCut', 5.0)))
        # The J this aliquot's plateau age was computed with, and its precision:
        # saved by the Plateau & isochron tool, else rebuilt from the aliquot's own R_std (or J).
        _m = cfg.get('ageFluxMode', 'imported')
        cfg_r = dict(cfg, ageFluxMode='ratio_J' if _m == 'ratio_J' else 'ratio_R_std')
        J_i = plat.get('J') or _flux_J(cfg_r)
        jf = plat.get('jErrFrac')
        jf = jf if jf is not None else cfg_j_err_frac(cfg_r)
        # ratio_R_std calibration carried from the stack, so the age is
        # computed from R with the real R_std / FCs / λ (never R_std = 70).
        jc = _default_j_cfg()
        for k in ('rStd', 'rStdErrPct', 'rStdErrMode', 'stdAgeMa',
                  'stdAgeErrMa', 'stdAgeErrType', 'lambdaYr', 'lambdaErrYr') + tuple(am.ROUTE_KEYS):
            if cfg.get(k) is not None:
                jc[k] = cfg[k]
        jc['ageFluxMode'] = 'ratio_R_std'
        jc['rStdErrPct'] = 100.0 * jf
        jc['rStdErrMode'] = 'pct'
        if J_i:
            # R_std equivalent of this aliquot's J (differs from the stored
            # R_std only for a typed J or an applied flux-hole J)
            r_eq = cfg_ingrowth(jc) / J_i
            if abs(r_eq / jc.get('rStd', r_eq) - 1.0) > 1e-9:
                jc['rStd'] = r_eq
        _sync_j_rstd(jc, 'R')         # jVal: this J, not the default placeholder
        name = str(sd.get('title', '')) or f'Aliquot {idx + 1}'
        out.append(dict(
            name=name, raw_name=str(sd.get('orig_title') or '') or name,
            ages=[0.0], unc=[0.0],     # placeholder; recomputed from R
            r_vals=[mR], r_uncs=[sR], manual=['auto'], j_cfg=jc,
            disk_id=cfg.get('disk_id'), hole_id=cfg.get('hole_id'),
            irradiation=cfg.get('irradiation') or cfg.get('irrad'),
            kca_vals=[float('nan')], kca_errs=[float('nan')],
            ar40_pct=[float('nan')], labels=[name], run_ids=[''],
            excluded=bool(sd.get('excluded', False)) or no_plateau, J=J_i, jfrac=jf,
            flux_hole=_stack_flux_hole(state, sd, idx),
            constants=dict(cfg.get('constants') or {}),
            # irradiation position at hole level (_fine_position)
            pos=_fine_position(cfg.get('disk_id'), cfg.get('hole_id'), jc,
                               f'__sess_{idx}'),
        ))
    return out


def parse_ararstack_combined(path, aliquots=None):
    """An .ararstack → one parse_arar-compatible dict, one R point per aliquot
    (see parse_ararstack_aliquots).  A sample has one calibration, so every
    aliquot's R is put on the first aliquot's J (R·J_i/J_ref): a stack can mix
    irradiation positions (different R_std), and reading them all with the
    first R_std would shift every other position's age.  Each aliquot keeps its
    own J precision and position (pt_jfrac / pt_pos) for the internal tier.
    Excluded aliquots come through marked rejected ('out')."""
    als = parse_ararstack_aliquots(path) if aliquots is None else aliquots
    if not als:
        return None
    J_ref = als[0]['J']
    r_vals, r_uncs = [], []
    for a in als:
        f = a['J'] / J_ref if (a['J'] and J_ref) else 1.0
        r_vals.append(a['r_vals'][0] * f)
        r_uncs.append(a['r_uncs'][0] * f)
    n = len(als)
    return dict(
        name=os.path.splitext(os.path.basename(path))[0],
        raw_name=_join_names(a.get('raw_name') or a['name'] for a in als),
        ages=[0.0] * n, unc=[0.0] * n,          # placeholders; recomputed from R
        r_vals=r_vals, r_uncs=r_uncs,
        manual=['out' if a['excluded'] else 'auto' for a in als],
        j_cfg=als[0]['j_cfg'],
        disk_id=None, hole_id=None, irradiation=None,
        kca_vals=[float('nan')] * n, kca_errs=[float('nan')] * n,
        ar40_pct=[float('nan')] * n, labels=[a['name'] for a in als],
        run_ids=[''] * n,
        pt_jfrac=[a['jfrac'] for a in als], pt_pos=[a['pos'] for a in als],
        # one entry per aliquot: its own calibration, where it was irradiated
        # and its measured ratio (the sample itself works on the first J)
        metas=[dict(export.step_meta({}), pos=_pos_info(a),
                    fx=export.flux_cells(a['j_cfg'], show_r=True),
                    r_meas=a['r_vals'][0], r_meas_unc=a['r_uncs'][0])
               for a in als],
    )


def parse_table(path):
    """Import .xlsx/.csv -> (ages, unc). First two numeric columns; values verbatim."""
    if not _HAVE_PANDAS:
        raise RuntimeError('pandas/openpyxl not available for table import.')
    if path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    num = df.apply(pd.to_numeric, errors='coerce')
    cols = [c for c in num.columns if num[c].notna().sum() > 0]
    if len(cols) < 2:
        raise RuntimeError('Need at least two numeric columns (age, uncertainty).')
    a = num[cols[0]].to_numpy()
    u = num[cols[1]].to_numpy()
    m = ~(np.isnan(a) | np.isnan(u))
    return a[m].tolist(), u[m].tolist()


# --------------------------------------------------------------------------
# Sample data model
# --------------------------------------------------------------------------
class Sample:
    """One sample: ages (or ratios) + a J calibration + display state."""

    def __init__(self, name='Sample', color='#1f77b4'):
        self.name = name
        # the experiment's name as its file came in, kept through renaming
        # (None for pasted data: then it is just the current name)
        self.raw_name = None
        self.color = color
        self.enabled = True
        self.is_plateau = False          # True -> collapse steps to one WM point
        self.input_mode = 'age'          # 'age' | 'ratio'
        self.j_cfg = _default_j_cfg()
        self.ages = np.array([], float)      # always Ma (converted if ratio)
        self.unc_2s = np.array([], float)    # 1-sigma Ma (despite the name; see sig1())
        self.r_vals = np.array([], float)
        self.r_uncs = np.array([], float)    # 1-sigma
        self.kca_vals = np.array([], float)  # K/Ca per analysis (from .arar)
        self.kca_errs = np.array([], float)  # 1σ uncertainty on K/Ca (from .arar)
        self.ar40_pct = np.array([], float)  # %40Ar* per analysis (from .arar)
        self.labels   = []                   # step labels (A, B, C…) from .arar
        self.run_ids  = []                   # RunID (Filename column) from .arar
        self.metas    = []                   # per step: run time, days since irradiation, raw signals
        self.constants = {}                  # mass-spec constants block from the .arar
        self.manual = np.array([], dtype=object)   # 'auto'|'in'|'out' per point
        self._loaded_excluded = np.array([], bool) # tracks points loaded as "No" from file
        self.auto_mask = np.array([], bool)
        self.reject = 'nmad'             # 'nmad'|'2sigma'|'none'
        self.nmad_thresh = 2.5           # NMAD multiplier for outlier rejection
        self.in_sigma = 1                # how this sample's entered uncertainty is read (1 or 2 sigma); converter writes raw 1-sigma
        self._from_file = False          # True for .arar loads (already scaled to 1-sigma)
        self.irradiation = None          # identity for disk-flux matching
        self.disk_id = None
        self.hole_id = None
        self._flux_src = None            # (irr,disk) key if J came from a loaded flux
        self._flux_override = None       # manual (irr, disk, hole_id) assignment or None
        # Set only when the points come from more than one irradiation position
        # (a stack mixing disks): each point's σ_J/J (NaN = this sample's own J
        # error) and its position.  None = the whole sample is one position.
        self.pt_jfrac = None
        self.pt_pos = None
        self._app = None                 # the window, set by add_sample

    def experiment_name(self):
        """The experiment name(s) this sample's data came in under, or None
        when none was kept.  Sessions saved before the name was recorded:
        a sample built from a Plateau stack still carries each aliquot's
        name as its point labels, so those are used."""
        if self.raw_name:
            return self.raw_name
        lb = [str(x).strip() for x in self.labels]
        if lb and len(lb) == len(self.ages) and all(len(x) > 3 for x in lb):
            return _join_names(lb)
        return None

    # --- effective inclusion mask -----------------------------------------
    def compute_auto(self):
        n = len(self.ages)
        if n == 0:
            self.auto_mask = np.array([], bool)
            return
        if self.reject == 'none' or n < 3:
            self.auto_mask = np.ones(n, bool)
        elif self.reject == '2sigma':
            self.auto_mask = iterative_2sigma_rejection(self.ages, self.unc_2s)
        else:
            self.auto_mask = nmad_rejection(self.ages, self.nmad_thresh)

    def mask(self):
        n = len(self.ages)
        if n == 0:
            return np.array([], bool)
        if len(self.auto_mask) != n:
            self.compute_auto()
        m = self.auto_mask.copy()
        for i, st in enumerate(self.manual):
            if st == 'in':
                m[i] = True
            elif st == 'out':
                m[i] = False
        return m

    def set_ages(self, ages, unc):
        self.ages = np.asarray(ages, float)
        self.unc_2s = np.asarray(unc, float)
        self.manual = np.full(len(self.ages), 'auto', dtype=object)
        self.compute_auto()

    # --- age derives from R, not the other way round ----------------------
    def has_R(self):
        """True if this sample carries raw 40Ar*/39ArK ratios (R_unk)."""
        return len(self.r_vals) > 0

    def recompute(self):
        """Derive ages + uncertainties from R_unk and the current J calibration.

        When raw ratios are present the age of every step is computed from R_unk
        and J (J given directly, or from R_std / R_FCs), never from a stored age,
        so a change to R_unk, R_std, J, t_FCs or an applied disk flux can never
        leave a stale age on the plot. r_uncs and unc_2s are both stored at
        1-sigma. Safe to call repeatedly; it does not compound errors.
        Returns an error string or None.
        """
        if not self.has_R():
            return None
        if self.j_cfg.get('ageFluxMode', 'imported') == 'imported':
            return None      # 'imported' = use the current ages as-is (frozen)
        if len(self.r_uncs) != len(self.r_vals):
            self.r_uncs = np.zeros(len(self.r_vals), float)
        res, err = r_to_age_cfg(self.r_vals, self.r_uncs, self.j_cfg)
        if err:
            return err       # caller shows the message; never crash on unpack
        ages, sig1 = res
        keep = self.manual.copy() if len(self.manual) == len(self.r_vals) else None
        self.ages = np.asarray(ages, float)
        self.unc_2s = np.asarray(sig1, float)
        self.manual = keep if keep is not None else \
            np.full(len(self.ages), 'auto', dtype=object)
        self.compute_auto()
        return None

    def use_measured_R(self):
        """Put every step's measured R_unk (and its σ) back as the R values.

        Aliquots merged into one sample have their R scaled by J_file/J_first
        (_rescale_to_J), which keeps each age right only while the sample J is
        the first aliquot's file J.  A J that replaces the file J (a disk-flux
        hole) must act on the measured R, or the file-J difference between the
        aliquots is applied a second time.  Returns True when any value moved."""
        n = len(self.r_vals)
        if n == 0 or len(self.metas) != n:
            return False
        rv = np.asarray(self.r_vals, float)
        have_u = len(self.r_uncs) == n
        ru = np.asarray(self.r_uncs, float) if have_u else None
        new_r, new_u = rv.copy(), (ru.copy() if have_u else None)
        for i, m in enumerate(self.metas):
            if not isinstance(m, dict):
                continue
            if m.get('r_meas') is not None:
                new_r[i] = float(m['r_meas'])
            if have_u and m.get('r_meas_unc') is not None:
                new_u[i] = float(m['r_meas_unc'])
        same = np.allclose(new_r, rv, rtol=1e-13, atol=0.0, equal_nan=True)
        if have_u:
            same = same and np.allclose(new_u, ru, rtol=1e-13, atol=0.0, equal_nan=True)
        if same:
            return False
        self.r_vals = new_r
        if have_u:
            self.r_uncs = new_u
        return True

    def sig1(self):
        # blank-dominated steps can carry a negative stored uncertainty;
        # matplotlib refuses negative error bars, so always use the magnitude
        return np.abs(np.asarray(self.unc_2s, dtype=float))

    def flux_key(self):
        return _j_flux_key(self.j_cfg)

    def _pt(self):
        """(σ_J/J, position) per point for a multi-position sample, else None."""
        n = len(self.ages)
        if (self.pt_pos is not None and self.pt_jfrac is not None
                and len(self.pt_pos) == n and len(self.pt_jfrac) == n):
            return self.pt_jfrac, self.pt_pos
        return None

    def positions(self, idx):
        """Irradiation position of the points idx (None = the sample's own)."""
        pt = self._pt()
        if pt is None:
            return [None] * len(idx)
        return [None if np.isnan(pt[0][i]) else pt[1][i] for i in idx]

    def sig_int(self, ages, s_ana, idx):
        """Internal tier of the points idx: analytical ⊕ ∂t/∂ln J · σ_J/J, with
        each point's own J precision when the sample spans several positions."""
        pt = self._pt()
        if pt is None:
            return internal_sigma_array(ages, s_ana, self.j_cfg)
        own = cfg_j_err_frac(self.j_cfg)
        lam = self.j_cfg.get('lambdaYr', LAMBDA_DEFAULT)
        jf = np.array([own if np.isnan(pt[0][i]) else pt[0][i] for i in idx])
        fl = np.array([dt_dlnR(a, lam) for a in np.ravel(ages)]) * jf
        return np.sqrt(np.asarray(s_ana, float)**2 + fl**2)

    def wm(self):
        """Per-sample weighted mean (included points), with three tiers (1-sigma)."""
        m = self.mask()
        if m.sum() == 0:
            return None
        a = self.ages[m]
        s_ana = self.sig1()[m]
        idx = np.where(m)[0]
        # one sample = one position, unless it spans several (mixed-disk stack)
        pos = [self.flux_key() if p is None else p for p in self.positions(idx)]
        # internal = analytical + J flux (∂t/∂ln J · σ_J/J);  external adds the
        # calibration systematic, one shared component at a time
        # (arar_math.systematic_components: anchored chain for Kuiper, Rivera
        # and Phillips)
        s_int = self.sig_int(a, s_ana, idx)
        s_ext = external_sigma_array(a, s_int, self.j_cfg)
        C, _ = systematic_matrix(a, self.j_cfg)
        # Points on several holes share their J error through the .jflux
        # correlation (the Plateau & isochron tool's rule), not just "same disk".
        pcorr = None
        if self.pt_pos is not None and self._app is not None:
            pcorr = self._app._j_corr(self._app._j_point_info(self, idx))
        return _combine(a, s_ana, s_int, s_ext, position_group=pos,
                        standard_group=[0] * len(a), od_mode='threshold',
                        sys_components=C, p_corr=pcorr)

    def pool_points(self):
        """Points that enter the pooled weighted mean, as (age, ana, int, ext)
        1-sigma tuples.

        A normal (fusion) sample contributes one tuple per included analysis,
        independent aliquots. A plateau sample collapses to a single tuple, its
        plateau weighted mean (the analytical tier already √MSWD-scaled by wm()),
        so its correlated steps count as one independent aliquot, not many.
        """
        m = self.mask()
        if self.is_plateau:
            w = self.wm()
            return [] if w is None else \
                [(w['wm'], w['sigma_ana'], w['sigma_int'], w['sigma_ext'])]
        av = self.ages[m]; sa = self.sig1()[m]
        si = self.sig_int(av, sa, np.where(m)[0])
        se = external_sigma_array(av, si, self.j_cfg)
        return list(zip(av.tolist(), sa.tolist(), si.tolist(), se.tolist()))


# --------------------------------------------------------------------------
# Per-sample control panel
# --------------------------------------------------------------------------
class _PasteEdit(QTextEdit):
    """A small text box that re-applies its callback when focus leaves,
    so manually deleting/editing lines takes effect without a button press."""

    def __init__(self, on_commit):
        super().__init__()
        self._on_commit = on_commit
        self._last = ''

    def focusOutEvent(self, ev):
        super().focusOutEvent(ev)
        cur = self.toPlainText()
        if cur != self._last:
            self._last = cur
            self._on_commit()


class SamplePanel(QFrame):
    def __init__(self, sample, app, index):
        super().__init__()
        self.s = sample
        self.app = app
        self.index = index
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            'SamplePanel { background:#fafafa; border:1px solid #ddd; '
            'border-radius:6px; }')
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 8)
        v.setSpacing(5)

        # header row: colour swatch, enable, name, remove
        hdr = QHBoxLayout()
        self.sw = QPushButton()
        self.sw.setFixedSize(16, 16)
        self.sw.setCursor(Qt.PointingHandCursor)
        self.sw.setToolTip('Click to change the colour')
        self.sw.clicked.connect(self._pick_color)
        self._show_color()
        hdr.addWidget(self.sw)
        self.cb = QCheckBox()
        self.cb.setChecked(self.s.enabled)
        self.cb.toggled.connect(self._on_enable)
        hdr.addWidget(self.cb)
        self.name = QLineEdit(self.s.name)
        self.name.textChanged.connect(self._on_name)
        hdr.addWidget(self.name, 1)
        rm = QPushButton('✕')
        rm.setFixedWidth(24)
        rm.setToolTip('Remove sample')
        rm.clicked.connect(lambda: self.app.remove_sample(self.s))
        hdr.addWidget(rm)
        v.addLayout(hdr)

        # input mode + reject
        row = QHBoxLayout()
        row.addWidget(QLabel('Input:'))
        self.mode = QComboBox()
        self.mode.addItems(['Age (Ma)', 'Ratio 40/39'])
        self.mode.setCurrentIndex(0 if self.s.input_mode == 'age' else 1)
        self.mode.currentIndexChanged.connect(self._on_mode)
        row.addWidget(self.mode)
        row.addWidget(QLabel('Reject:'))
        self.rej = QComboBox()
        self.rej.addItems(['NMAD', '2-sigma', 'none'])
        self.rej.setCurrentIndex({'nmad': 0, '2sigma': 1, 'none': 2}[self.s.reject])
        self.rej.currentIndexChanged.connect(self._on_reject)
        row.addWidget(self.rej)
        row.addWidget(QLabel('thr:'))
        self.thr = QLineEdit(f'{self.s.nmad_thresh:g}')
        self.thr.setFixedWidth(36)
        self.thr.setToolTip('NMAD multiplier: exclude points with |age - median| > thr * NMAD.\n'
                            'Lower = stricter. Typical 2-3.')
        self.thr.editingFinished.connect(self._on_thresh)
        row.addWidget(self.thr)
        self.cb_plateau = QCheckBox('Collapse to 1 age')
        self.cb_plateau.setChecked(self.s.is_plateau)
        self.cb_plateau.setToolTip(
            'Combine this sample’s analyses into one weighted-mean age, drawn as a\n'
            'square, that enters the pool and the density curve as one point. Tick it\n'
            'when the analyses are the heating steps of one experiment, which share J;\n'
            'leave it off for independent fusions, where each analysis counts on its\n'
            'own. The mean carries the internal error (SEM × √MSWD when the steps are\n'
            'overdispersed) and the external error. Stacked plateau files arrive combined.')
        self.cb_plateau.toggled.connect(self._on_plateau)
        row.addWidget(self.cb_plateau)
        row.addWidget(QLabel('σ in:'))
        self.insig = QComboBox()
        self.insig.addItems(['2σ', '1σ'])
        self.insig.setCurrentIndex(0 if self.s.in_sigma == 2 else 1)
        self.insig.setToolTip("How this sample's entered uncertainty is read.\n"
                              'Changing it rescales the stored values immediately.')
        self.insig.currentIndexChanged.connect(self._on_in_sigma)
        row.addWidget(self.insig)
        row.addStretch()
        v.addLayout(row)

        # flux mode + value on one row
        jrow = QGridLayout()
        jrow.setHorizontalSpacing(6)
        jrow.addWidget(QLabel('Flux:'), 0, 0)
        self.jmode = QComboBox()
        self.jmode.addItems(list(_JMODE_DISPLAY.values()))
        self.jmode.setCurrentText(
            _JMODE_DISPLAY.get(self.s.j_cfg.get('ageFluxMode', 'imported'), 'imported'))
        self.jmode.setMinimumWidth(120)
        self.jmode.currentTextChanged.connect(self._on_jmode)
        jrow.addWidget(self.jmode, 0, 1)
        self.lbl_jval = QLabel('J:')
        jrow.addWidget(self.lbl_jval, 0, 2)
        self.jval = QLineEdit(f"{self.s.j_cfg.get('jVal', 0.005):.8g}")
        self.jval.editingFinished.connect(self._commit_j)
        jrow.addWidget(self.jval, 0, 3)
        self.lbl_jpct = QLabel('±J%:')
        jrow.addWidget(self.lbl_jpct, 0, 4)
        self.jpct = QLineEdit(f"{self.s.j_cfg.get('jErrPct', 0.10):.4g}")
        self.jpct.editingFinished.connect(self._commit_j)
        jrow.addWidget(self.jpct, 0, 5)
        # ratio_R_FCs paste-tier selector + R_FCs% error (hidden for other modes)
        self.lbl_paste_tier = QLabel('Paste σ:')
        jrow.addWidget(self.lbl_paste_tier, 1, 0)
        self.paste_tier_sel = QComboBox()
        self.paste_tier_sel.addItems(['analytical', 'internal'])
        self.paste_tier_sel.setCurrentText(self.s.j_cfg.get('pasteTier', 'analytical'))
        self.paste_tier_sel.setToolTip(
            'What tier the pasted R_unk/R_FCs uncertainties represent:\n'
            ' analytical  = measurement scatter on R_unk/R_FCs only\n'
            '               (specify ±R_FCs% below to build the internal tier)\n'
            ' internal    = pasted σ already includes the local R_FCs\n'
            '               measurement error; no additional J error added')
        self.paste_tier_sel.currentTextChanged.connect(self._on_paste_tier)
        jrow.addWidget(self.paste_tier_sel, 1, 1)
        self.lbl_rfcs_err = QLabel('±R_FCs%:')
        self.lbl_rfcs_err.setToolTip('1σ % uncertainty on the local R_FCs measurement.\n'
                                     'Propagates as the internal J-flux tier (σ_int).')
        jrow.addWidget(self.lbl_rfcs_err, 1, 2)
        self.rfcs_err = QLineEdit(f"{self.s.j_cfg.get('rfcsErrPct', 0.0):.4g}")
        self.rfcs_err.setToolTip('1σ % uncertainty on the local R_FCs measurement.\n'
                                 'Propagates as the internal J-flux tier (σ_int).')
        self.rfcs_err.editingFinished.connect(self._commit_j)
        jrow.addWidget(self.rfcs_err, 1, 3)
        v.addLayout(jrow)
        self._jwidgets = [self.jval, self.jpct]

        # paste/edit box (re-applies when focus leaves, so manual edits take effect)
        self.data = _PasteEdit(self._apply_paste)
        self.data.setFixedHeight(64)
        self.data.setFont(mono_font(9))
        self.data.setPlaceholderText('Paste: age <tab> uncertainty  (one per line) '
                                     ', edits apply when you click away')
        v.addWidget(self.data)

        # run list toolbar: sort + label
        rlhdr = QHBoxLayout()
        runlbl = QLabel('Runs (click to include / exclude):')
        runlbl.setStyleSheet('font-size:10px; color:#555;')
        rlhdr.addWidget(runlbl)
        rlhdr.addStretch()
        sort_btn = QPushButton('Sort ages')
        sort_btn.setStyleSheet('font-size:9px; padding:1px 5px;')
        sort_btn.setToolTip('Reorder runs youngest → oldest by age')
        sort_btn.clicked.connect(self._sort_by_age)
        rlhdr.addWidget(sort_btn)
        v.addLayout(rlhdr)
        self.runlist = QListWidget()
        self.runlist.setFixedHeight(84)
        self.runlist.setFont(mono_font(9))
        self.runlist.itemClicked.connect(self._on_run_click)
        self.runlist.setContextMenuPolicy(Qt.CustomContextMenu)
        self.runlist.customContextMenuRequested.connect(self._run_context_menu)
        v.addWidget(self.runlist)

        # buttons
        b = QHBoxLayout()
        for txt, fn in [('Apply paste', self._apply_paste),
                        ('Load', lambda: self.app.load_into(self.s)),
                        ('Append', lambda: self.app.append_into(self.s)),
                        ('Import', lambda: self.app.import_into(self.s)),
                        ('Clear', self._clear_data)]:
            btn = QPushButton(txt)
            btn.setStyleSheet('font-size:10px;')
            btn.clicked.connect(fn)
            b.addWidget(btn)
        v.addLayout(b)

        self.info = QLabel('')
        self.info.setStyleSheet('color:#555; font-size:10px;')
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        self._update_jvis()
        self.thr.setEnabled(self.s.reject == 'nmad')
        self.sync_data_box()
        self.refresh_info()

    # --- callbacks ---------------------------------------------------------
    def _show_color(self):
        self.sw.setStyleSheet(f'QPushButton {{ background:{self.s.color}; '
                              'border:1px solid #999; border-radius:3px; }')

    def _pick_color(self):
        from PyQt5.QtGui import QColor
        # the default sample colours as custom colours, in the order new
        # samples get them; Qt fills the custom grid column by column
        for k, col in enumerate(_PALETTE):
            QColorDialog.setCustomColor((k % 8) * 2 + k // 8, QColor(col))
        c = QColorDialog.getColor(QColor(self.s.color), self, 'Sample colour',
                                  QColorDialog.DontUseNativeDialog)
        if not c.isValid():
            return
        self.s.color = c.name()
        self._show_color()
        self.populate_runlist()
        self.app.redraw()

    def _on_enable(self, on):
        self.s.enabled = on
        self.app.redraw()

    def _on_name(self, t):
        self.s.name = t
        self.app.redraw()

    def _reset_manual_to_auto(self):
        """Clear all manual include/exclude clicks so a changed rejection rule
        (method or NMAD threshold) decides every point fresh."""
        self.s.manual = np.full(len(self.s.ages), 'auto', dtype=object)

    def _on_plateau(self, on):
        self.s.is_plateau = bool(on)
        self.app.redraw()
        self.refresh_info()

    def _on_reject(self, _):
        self.s.reject = {0: 'nmad', 1: '2sigma', 2: 'none'}[self.rej.currentIndex()]
        self.thr.setEnabled(self.s.reject == 'nmad')
        self._reset_manual_to_auto()   # new rule decides every point fresh
        self.s.compute_auto()
        self.app.redraw()
        self.refresh_info()

    def _on_thresh(self):
        try:
            t = float(self.thr.text())
            if t > 0:
                self.s.nmad_thresh = t
        except ValueError:
            self.thr.setText(f'{self.s.nmad_thresh:g}')
            return
        self._reset_manual_to_auto()   # changed NMAD threshold decides fresh
        self.s.compute_auto()
        self.app.redraw()
        self.refresh_info()

    def _on_jmode(self, mode_display):
        self.s.j_cfg['ageFluxMode'] = _JMODE_INTERNAL.get(mode_display, mode_display)
        self._update_jvis()
        self._commit_j()

    def _update_jvis(self):
        """Relabel and repopulate the value fields to match the J mode."""
        mode = _JMODE_INTERNAL.get(self.jmode.currentText(), self.jmode.currentText())
        cfg = self.s.j_cfg
        # block signals while we repopulate text
        for w in (self.jval, self.jpct):
            w.blockSignals(True)
        if mode == 'ratio_J':
            self.lbl_jval.setText('J:')
            self.lbl_jpct.setText('±J%:')
            self.jval.setText(f"{cfg.get('jVal', 0.005):.8g}")
            self.jpct.setText(f"{cfg.get('jErrPct', 0.10):.4g}")
            jr_on = True
        elif mode == 'ratio_R_std':
            mon = cfg_monitor(cfg)[0]
            self.lbl_jval.setText('R_std:')
            self.jval.setToolTip(f'Measured ⁴⁰Ar*/³⁹Ar of the monitor ({mon}), as set in the '
                                 'calibration bar.')
            self.lbl_jpct.setText('±R%:')
            self.jval.setText(f"{from_fcs_r(cfg.get('rStd', 70.0), cfg):.11g}")
            self.jpct.setText(f"{cfg.get('rStdErrPct', 0.10):.4g}")
            jr_on = True
        elif mode == 'ratio_R_FCs':
            self.lbl_jval.setText('R/R_std:')
            self.lbl_jpct.setText('(n/a)')
            self.jval.setText('(input as ratios)')
            self.jpct.setText('')
            jr_on = False
        else:  # imported
            self.lbl_jval.setText('J:')
            self.lbl_jpct.setText('±J%:')
            self.jval.setText('(ages used directly)')
            self.jpct.setText('')
            jr_on = False
        for w in (self.jval, self.jpct):
            w.setEnabled(jr_on)
            w.blockSignals(False)
        # Row 5: paste-tier + R_FCs% error, only meaningful in ratio_R_FCs mode
        is_rfcs = (mode == 'ratio_R_FCs')
        for w in (self.lbl_paste_tier, self.paste_tier_sel,
                  self.lbl_rfcs_err, self.rfcs_err):
            w.setVisible(is_rfcs)
        if is_rfcs:
            rfcs_err_on = (self.paste_tier_sel.currentText() == 'analytical')
            self.lbl_rfcs_err.setEnabled(rfcs_err_on)
            self.rfcs_err.setEnabled(rfcs_err_on)
        # mode picker tooltip
        self.jmode.setToolTip(
            'imported         : plot/combine the entered ages directly (no J error).\n'
            'J value          : ages from R via a known J value (+/-J%).\n'
            'R (standard)     : ages from R via a flux-monitor ratio R_std (+/-R%).\n'
            'R(unk) / R(std)  : R already given relative to FCs; only t_FCs needed.\n'
            '                   Use "Paste σ / ±R_FCs%" to add the internal J tier.')

    def _commit_j(self):
        def f(le, d):
            try:
                return float(le.text())
            except (ValueError, TypeError):
                return d
        # λ / t_FCs / FC-err type live in the window's calibration bar;
        # this commit only handles the sample's own J / R fields
        mode = _JMODE_INTERNAL.get(self.jmode.currentText(), self.jmode.currentText())
        if mode == 'ratio_J':
            self.s.j_cfg['jVal'] = f(self.jval, 0.005)
            self.s.j_cfg['jErrPct'] = f(self.jpct, 0.10)
            _sync_j_rstd(self.s.j_cfg, 'J')      # keep R_std consistent with J
        elif mode == 'ratio_R_std':
            # typed as the measured monitor ratio, stored FCs-equivalent
            self.s.j_cfg['rStd'] = to_fcs_r(f(self.jval, 70.0), self.s.j_cfg)
            self.s.j_cfg['rStdErrPct'] = f(self.jpct, 0.10)
            _sync_j_rstd(self.s.j_cfg, 'R')      # keep J consistent with R_std
        elif mode == 'ratio_R_FCs':
            self.s.j_cfg['pasteTier'] = self.paste_tier_sel.currentText()
            pct = f(self.rfcs_err, 0.0)
            self.s.j_cfg['rfcsErrPct'] = max(0.0, pct)
        # R-driven samples re-age on any J/R change; age-mode samples keep their
        # entered ages (J then only feeds the internal/external error tiers).
        if self.s.has_R() and self.s.input_mode == 'ratio':
            err = self.s.recompute()
            if err:
                self.info.setText(err)
        self.sync_data_box()
        self.app.redraw()
        self.refresh_info()

    def _on_paste_tier(self, _):
        """Toggle ±R_FCs% field on/off and commit."""
        self._update_jvis()
        self._commit_j()

    def _on_in_sigma(self, _):
        """Reinterpret this sample's entered uncertainty as 1 or 2 sigma."""
        new = 2 if self.insig.currentIndex() == 0 else 1
        old = self.s.in_sigma
        if new == old:
            return
        factor = old / new
        self.s.in_sigma = new
        self.s._from_file = False
        if self.s.has_R():
            self.s.r_uncs = self.s.r_uncs * factor
            self.s.recompute()                     # unc follows R, stays consistent
        else:
            self.s.unc_2s = self.s.unc_2s * factor
            self.s.compute_auto()
        self.sync_data_box()
        self.app.redraw()
        self.refresh_info()

    def _on_mode(self, _):
        want = 'age' if self.mode.currentIndex() == 0 else 'ratio'
        self.s.input_mode = want
        # Ratio view -> ages are recomputed live from R + J. Age view -> the
        # current (R-derived) ages are shown and become directly editable; J then
        # only feeds the error tiers. Both views remain available for R-samples.
        if want == 'ratio' and self.s.has_R():
            err = self.s.recompute()
            if err:
                self.info.setText(err)
        self.sync_data_box()
        self.app.redraw()
        self.refresh_info()

    def _clear_data(self):
        """Wipe all data from this sample (keeps its J settings and name)."""
        self.s.ages = np.array([], float)
        self.s.unc_2s = np.array([], float)
        self.s.r_vals = np.array([], float)
        self.s.r_uncs = np.array([], float)
        self.s.kca_vals = np.array([], float)
        self.s.kca_errs = np.array([], float)
        self.s.ar40_pct = np.array([], float)
        self.s.labels   = []
        self.s.run_ids  = []
        self.s.metas    = []
        self.s.pt_jfrac = self.s.pt_pos = None
        self.s.manual = np.array([], dtype=object)
        self.s.auto_mask = np.array([], bool)
        self.s._from_file = False
        self.s._loaded_excluded = np.array([], bool)
        self.s.input_mode = 'age'
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0)
        self.mode.blockSignals(False)
        self.data.clear()
        self.data._last = ''
        self.app.redraw()
        self.refresh_info()

    def _apply_paste(self):
        raw_text = self.data.toPlainText()
        rows = []
        for line in raw_text.splitlines():
            line = line.split('#', 1)[0]              # drop "# 55.30 Ma" annotations
            # Normalize all Unicode whitespace (incl. non-breaking spaces from PDF/spreadsheet copy)
            line = ''.join(' ' if c.isspace() else c for c in line)
            parts = line.replace(',', ' ').replace(';', ' ').split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        # Fallback: one number per line, alternating value / error (common PDF copy)
        if not rows:
            nums = []
            for line in raw_text.splitlines():
                line = line.split('#', 1)[0]
                line = ''.join(' ' if c.isspace() else c for c in line)
                parts = line.replace(',', ' ').replace(';', ' ').split()
                if len(parts) == 1:
                    try:
                        nums.append(float(parts[0]))
                    except ValueError:
                        pass
            if len(nums) >= 4 and len(nums) % 2 == 0:
                rows = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
        if not rows:
            QMessageBox.warning(self, 'Paste', 'No two-column rows found.')
            return
        col0 = np.array([p[0] for p in rows], float)
        col1 = np.array([p[1] for p in rows], float)
        new_set = len(col0) != len(self.s.ages)
        if new_set:
            self.s.pt_jfrac = self.s.pt_pos = None     # new data set
        if self.s.input_mode == 'ratio':
            # box holds R_unk <tab> sigma(R): stays R-driven, re-age from R + J
            self.s.r_vals = col0
            self.s.r_uncs = col1 / self.s.in_sigma
            self.s.kca_vals = np.array([], float)
            self.s.kca_errs = np.array([], float)
            self.s.ar40_pct = np.array([], float)
            self.s.labels   = []
            self.s.run_ids  = []
            self.s.metas    = []
            self.s._from_file = False
            err = self.s.recompute()
            if err:
                QMessageBox.warning(self, 'R->Age', err)
                return
        else:
            self.s.input_mode = 'age'
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(0)
            self.mode.blockSignals(False)
            self.s.r_vals = np.array([], float)
            self.s.r_uncs = np.array([], float)
            self.s.kca_vals = np.array([], float)
            self.s.kca_errs = np.array([], float)
            self.s.ar40_pct = np.array([], float)
            self.s.labels   = []
            self.s.run_ids  = []
            self.s.metas    = []
            self.s.pt_jfrac = self.s.pt_pos = None
            self.s._from_file = False
            # a new data set is read at the window's σ-in; an edit of the rows
            # already shown at this sample's own σ-in (sync_data_box)
            if new_set:
                self.s.in_sigma = self.app.input_sigma
            self.insig.blockSignals(True)
            self.insig.setCurrentIndex(0 if self.s.in_sigma == 2 else 1)
            self.insig.blockSignals(False)
            self.s.set_ages(col0, col1 / self.s.in_sigma)
        self.sync_data_box()
        self.app.redraw()
        self.refresh_info()

    def refresh_info(self):
        n = len(self.s.ages)
        self.populate_runlist()
        if n == 0:
            self.info.setText('No data.')
            return
        m = self.s.mask()
        r = self.s.wm()
        ds = self.app.display_sigma
        sl = f'{ds}σ'
        if r:
            self.info.setText(
                f"n={int(m.sum())}/{n} incl.   WM = {r['wm']:.4f} Ma   "
                f"MSWD={r['mswd']:.2f}  p={r.get('prob', float('nan')):.3f}\n"
                f"±{sl}: ana {r['sigma_ana']*ds:.4f} | int {r['sigma_int']*ds:.4f} | "
                f"ext {r['sigma_ext']*ds:.4f}   (key={self.s.flux_key()[0]})")
        else:
            self.info.setText(f'n={n}, none included.')

    def populate_runlist(self):
        """Refill the clickable run list, marking excluded runs."""
        from PyQt5.QtGui import QColor as _QC
        self.runlist.blockSignals(True)
        self.runlist.clear()
        n = len(self.s.ages)
        if n:
            m = self.s.mask()
            ds = self.app.display_sigma
            sig = self.s.sig1()
            for i in range(n):
                inc = bool(m[i])
                mark = ' ' if inc else 'x'
                txt = f'{mark} {i+1:>3}  {self.s.ages[i]:.4f} ±{sig[i]*ds:.4f}'
                it = QListWidgetItem(txt)
                if not inc:
                    f = mono_font(9)
                    f.setStrikeOut(True)
                    it.setFont(f)
                    it.setForeground(_QC('#b00020'))
                else:
                    it.setForeground(_QC(self.s.color))
                self.runlist.addItem(it)
        self.runlist.blockSignals(False)

    def _on_run_click(self, item):
        """Click a run to cycle auto -> exclude -> include -> auto."""
        i = self.runlist.row(item)
        if i < 0 or i >= len(self.s.ages):
            return
        eff = self.s.mask()[i]
        cur = self.s.manual[i]
        if cur in ('in', 'out'):
            self.s.manual[i] = 'auto'
        elif eff:
            self.s.manual[i] = 'out'
        else:
            self.s.manual[i] = 'in'
        self.s.compute_auto()
        self.app.redraw()
        self.refresh_info()

    def _sort_by_age(self):
        """Reorder all per-step arrays by ascending age."""
        n = len(self.s.ages)
        if n < 2:
            return
        order = np.argsort(self.s.ages)
        self.s.ages = self.s.ages[order]
        self.s.unc_2s = self.s.unc_2s[order]
        if len(self.s.r_vals) == n:
            self.s.r_vals = self.s.r_vals[order]
        if len(self.s.r_uncs) == n:
            self.s.r_uncs = self.s.r_uncs[order]
        if len(self.s.manual) == n:
            self.s.manual = self.s.manual[order]
        if len(self.s._loaded_excluded) == n:
            self.s._loaded_excluded = self.s._loaded_excluded[order]
        if len(self.s.kca_vals) == n:
            self.s.kca_vals = self.s.kca_vals[order]
        if len(self.s.kca_errs) == n:
            self.s.kca_errs = self.s.kca_errs[order]
        if len(self.s.ar40_pct) == n:
            self.s.ar40_pct = self.s.ar40_pct[order]
        if len(self.s.labels) == n:
            self.s.labels = [self.s.labels[i] for i in order]
        if len(self.s.run_ids) == n:
            self.s.run_ids = [self.s.run_ids[i] for i in order]
        if len(self.s.metas) == n:
            self.s.metas = [self.s.metas[i] for i in order]
        if self.s.pt_pos is not None and len(self.s.pt_pos) == n:
            self.s.pt_pos = [self.s.pt_pos[i] for i in order]
            self.s.pt_jfrac = np.asarray(self.s.pt_jfrac, float)[order]
        self.s.compute_auto()
        self.sync_data_box()
        self.app.redraw()
        self.refresh_info()

    def _run_context_menu(self, pos):
        """Right-click a run to permanently delete it from the sample."""
        from PyQt5.QtWidgets import QMenu
        i = self.runlist.indexAt(pos).row()
        if i < 0 or i >= len(self.s.ages):
            return
        menu = QMenu(self)
        act = menu.addAction(f'Delete run {i + 1} permanently')
        if menu.exec_(self.runlist.viewport().mapToGlobal(pos)) == act:
            self._delete_run(i)

    def _delete_run(self, i):
        """Permanently remove run i from all per-step arrays."""
        n = len(self.s.ages)
        if i < 0 or i >= n:
            return
        keep = np.ones(n, bool)
        keep[i] = False
        self.s.ages = self.s.ages[keep]
        self.s.unc_2s = self.s.unc_2s[keep]
        if len(self.s.r_vals) == n:
            self.s.r_vals = self.s.r_vals[keep]
        if len(self.s.r_uncs) == n:
            self.s.r_uncs = self.s.r_uncs[keep]
        if len(self.s.manual) == n:
            self.s.manual = self.s.manual[keep]
        if len(self.s._loaded_excluded) == n:
            self.s._loaded_excluded = self.s._loaded_excluded[keep]
        if len(self.s.kca_vals) == n:
            self.s.kca_vals = self.s.kca_vals[keep]
        if len(self.s.kca_errs) == n:
            self.s.kca_errs = self.s.kca_errs[keep]
        if len(self.s.ar40_pct) == n:
            self.s.ar40_pct = self.s.ar40_pct[keep]
        if len(self.s.labels) == n:
            self.s.labels = [self.s.labels[i] for i in range(n) if keep[i]]
        if len(self.s.run_ids) == n:
            self.s.run_ids = [self.s.run_ids[i] for i in range(n) if keep[i]]
        if len(self.s.metas) == n:
            self.s.metas = [self.s.metas[i] for i in range(n) if keep[i]]
        if self.s.pt_pos is not None and len(self.s.pt_pos) == n:
            self.s.pt_pos = [self.s.pt_pos[i] for i in range(n) if keep[i]]
            self.s.pt_jfrac = np.asarray(self.s.pt_jfrac, float)[keep]
        self.s.compute_auto()
        self.sync_data_box()
        self.app.redraw()
        self.refresh_info()

    def sync_data_box(self):
        """Mirror the model into the data box and remember the text, so a
        focus-out never misfires _apply_paste on text we set ourselves.
        Uncertainties are written at this sample's σ-in, the level
        _apply_paste reads them back at, so editing one line leaves the
        others unchanged."""
        k = self.s.in_sigma
        if self.s.has_R() and self.s.input_mode == 'ratio':
            # editable R_unk <tab> sigma(R); derived age annotated
            lines = [f'{r:.8g}\t{ru * k:.4g}\t# {a:.4f} Ma' for r, ru, a in
                     zip(self.s.r_vals, self.s.r_uncs, self.s.ages)]
            self.data.setText('\n'.join(lines))
        else:
            self.data.setText('\n'.join(f'{a}\t{float(u * k)}' for a, u in
                                        zip(self.s.ages, self.s.unc_2s)))
        self.data._last = self.data.toPlainText()


# --------------------------------------------------------------------------
# Disk flux dialog
# --------------------------------------------------------------------------
class _ProbFluxHost(FluxHost):
    """Adapter: drives the shared flux dialog for this tool's samples."""
    title = 'Load disk flux'
    item_noun = 'Sample'

    def __init__(self, app):
        self.app = app

    def disk_flux(self):
        return self.app._disk_flux

    def load(self, dialog):
        paths, _ = dfold.get_open_file_names(
            dialog, 'Import .jflux', '', 'Flux config (*.jflux);;All (*)',
            tool='flux_regression')
        if not paths:
            return
        if self.app._import_jflux_files(paths, error_parent=dialog):
            self.app._apply_flux_to_samples()

    def preferred_flux(self):
        return self.app._preferred_flux

    def items(self):
        return self.app.samples

    def item_label(self, s):
        return s.name

    def item_disk_hole(self, s):
        return (s.disk_id, s.hole_id)

    def get_override(self, row, s):
        ov = getattr(s, '_flux_override', None)
        if ov is None:
            return 'auto'
        if (str(ov[0]), str(ov[1])) == ('NONE', 'NONE'):
            return 'none'
        return tuple(ov)

    def set_override(self, row, s, val):
        if val == 'auto':
            s._flux_override = None
        elif val == 'none':
            s._flux_override = ('NONE', 'NONE', 'NONE')
        else:
            s._flux_override = val

    def match(self, row, s):
        return self.app._flux_match(s)

    def hole_R(self, key, idx):
        return self.app._flux_hole_R(key, idx)

    def on_row_change(self, row, s):
        # a hole picked in the table takes effect at once, J and ρ together
        self.app._apply_flux_to_samples()

    def apply(self):
        self.app._apply_flux_to_samples()

    def after_delete(self):
        self.app._apply_flux_to_samples()


class ProbabilityWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Age distribution')
        self.resize(1280, 760)
        self.samples = []
        self.panels = []
        self.display_sigma = 2
        self.age_dec = 3              # decimal places for ages/uncertainties in the legend
        self.input_sigma = 1          # interpret incoming (paste/import/load) uncertainty as 1 or 2 sigma; converter writes 1-sigma
        self.pool_on = False
        self.same_fcs = True   # every sample is on the calibration bar's FCs age
        self.corr_j = False
        self._disk_flux = {}   # (irradiation, disk_id) -> {hole_id: idx, 'J','J_sigma','J_corr'}
        self._preferred_flux = None   # (irr, disk) the user most recently loaded/chose; wins ties
        self._pool_report = ''
        self._click_pts = []          # (sample, point_index, age, rank) for click-toggle
        self.xlim_mode = 'auto'       # 'auto' | 'manual'
        self.x_lo = None
        self.x_hi = None
        self.ylim_mode = 'auto'
        self.y_max = None
        self.y_mode = 'rank'          # 'rank' | 'relative' | 'probability' | 'normalised'
        self._ylim_saved = {}         # per-mode storage: {mode: (ylim_mode, y_max)}
        self.mix_on = False           # Mixture view shown in place of the plot
        self._mix_cache = {}          # (data, scale, model) -> fit(s)
        self._mix_state = None        # what the Mixture view shows now (for the exports)
        self._mix_want = None         # sample to show, from a loaded session
        self._build()
        # start with one empty sample
        self.add_sample()
        self.redraw()

    # --- UI -----------------------------------------------------------------
    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # left: plot + toolbar + report
        left = QVBoxLayout()
        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self._main_toolbar = NavigationToolbar(self.canvas, self)
        left.addWidget(self._main_toolbar, 0)

        # axis / scale controls
        axrow = QHBoxLayout()
        axrow.setSpacing(4)
        axrow.addWidget(QLabel('X:'))
        self.x_lo_ed = QLineEdit()
        self.x_lo_ed.setPlaceholderText('min')
        self.x_lo_ed.setFixedWidth(60)
        axrow.addWidget(self.x_lo_ed)
        self.x_hi_ed = QLineEdit()
        self.x_hi_ed.setPlaceholderText('max')
        self.x_hi_ed.setFixedWidth(60)
        axrow.addWidget(self.x_hi_ed)
        axrow.addWidget(QLabel('Y max:'))
        self.y_max_ed = QLineEdit()
        self.y_max_ed.setPlaceholderText('auto')
        self.y_max_ed.setFixedWidth(50)
        axrow.addWidget(self.y_max_ed)
        for txt, fn, tip in [
            ('Apply', self._apply_axes, 'Apply the X/Y limits typed above'),
            ('Auto', self._auto_axes, 'Reset to automatic limits'),
            ('Fit incl.', self._fit_included, 'Fit X-range to the included data'),
            ('Show all', self._show_all, 'Fit X-range to all data (incl. excluded)')]:
            b = QPushButton(txt)
            b.setStyleSheet('font-size:10px; padding:2px 6px;')
            b.setToolTip(tip)
            b.clicked.connect(fn)
            axrow.addWidget(b)
        axrow.addSpacing(10)
        axrow.addWidget(QLabel('Y axis:'))
        self.ymode_sel = QComboBox()
        self.ymode_sel.addItems(['Rank', 'Relative prob.', 'Probability', 'Normalised'])
        self.ymode_sel.setCurrentIndex(0)
        self.ymode_sel.setMaximumWidth(150)
        self.ymode_sel.setToolTip(
            'Rank: the probability curve of each sample (peak = 1) with the analyses '
            'at their rank (rank / n), so the top analysis sits at the curve peak.\n'
            'Relative prob.: the summed-Gaussian curves scaled so the tallest peak = 1, '
            'with the analyses as a rug along the base, for comparing shapes.\n'
            'Probability: each curve peaks at the p-value of its MSWD × 100 '
            '(p = 0.88 peaks at 88, p = 0.05 hardly rises).\n'
            'Normalised: every curve, the pooled one included, peaks at 100 %, with '
            'the analyses of each sample spread by rank from 0 to 100 %, so data '
            'sets of different size and precision can be compared.')
        self.ymode_sel.currentIndexChanged.connect(self._on_ymode)
        axrow.addWidget(self.ymode_sel)
        axrow.addStretch()
        left.addLayout(axrow)

        # the plot area holds the plot and the Mixture view (Mixture button)
        self._plot_stack = QStackedWidget()
        self._plot_stack.addWidget(self.canvas)
        self._plot_stack.addWidget(self._build_mixture_page())
        left.addWidget(self._plot_stack, 1)
        rep_row = QHBoxLayout()
        rep_row.addWidget(QLabel('Combination report'))
        rep_row.addStretch()
        self.btn_breakdown = QPushButton('ⓘ  Error breakdown')
        self.btn_breakdown.setToolTip(
            'Pie chart of where the age uncertainty comes from '
            '(analytical, J flux, FCs age, decay constant).')
        self.btn_breakdown.clicked.connect(self._show_error_breakdown)
        rep_row.addWidget(self.btn_breakdown)
        left.addLayout(rep_row)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setFixedHeight(150)
        self.report.setFont(mono_font(9))
        left.addWidget(self.report)
        root.addLayout(left, 3)

        # right: controls
        right = QVBoxLayout()
        bar = QHBoxLayout()
        add = QPushButton('+ Add sample')
        add.clicked.connect(lambda: (self.add_sample(), self.redraw()))
        bar.addWidget(add)
        self.pool_btn = QPushButton('Pool: off')
        self.pool_btn.setCheckable(True)
        self.pool_btn.toggled.connect(self._on_pool)
        bar.addWidget(self.pool_btn)
        self.mix_btn = QPushButton('Mixture: off')
        self.mix_btn.setCheckable(True)
        self.mix_btn.setToolTip(
            'Show a mixture model of one sample in place of the plot: the age\n'
            'components its analyses fall into, with their ages, proportions and\n'
            'a radial plot (Galbraith & Laslett 1993). The plot and the samples\n'
            'are not changed; switch off to go back.')
        self.mix_btn.toggled.connect(self._on_mixture)
        bar.addWidget(self.mix_btn)
        right.addLayout(bar)

        opt = QHBoxLayout()
        self.cb_corr = QCheckBox('Force corr. J')
        self.cb_corr.setToolTip('Treat every sample as sharing one irradiation position '
                                '(J fully correlated). Off = group by J-flux key.')
        self.cb_corr.toggled.connect(self._on_corr)
        opt.addWidget(self.cb_corr)
        opt.addWidget(QLabel('Display:'))
        self.sig_sel = QComboBox()
        self.sig_sel.addItems(['1σ', '2σ'])
        self.sig_sel.setCurrentIndex(1)
        self.sig_sel.currentIndexChanged.connect(self._on_sigma)
        opt.addWidget(self.sig_sel)
        opt.addSpacing(10)
        opt.addWidget(QLabel('Decimal:'))
        self.dec_sel = QSpinBox()
        self.dec_sel.setRange(0, 6)
        self.dec_sel.setValue(self.age_dec)
        self.dec_sel.setToolTip('Decimal places for the ages and uncertainties in the '
                                'legend (and image exports).')
        self.dec_sel.valueChanged.connect(self._on_decimals)
        opt.addWidget(self.dec_sel)
        opt.addSpacing(10)
        opt.addWidget(QLabel('Import σ:'))
        self.in_sig_sel = QComboBox()
        self.in_sig_sel.addItems(['1σ', '2σ'])
        self.in_sig_sel.setCurrentIndex(0)   # converter writes raw 1-sigma
        self.in_sig_sel.setToolTip('σ level of incoming data (pasted, imported and loaded '
                                   '.arar/.ararpt).\nThe numbers are read as they are, at this '
                                   'level.\nThe converter writes 1σ, so the default is 1σ.\n'
                                   'Each sample has its own σ in selector, which can be changed '
                                   'after loading.')
        self.in_sig_sel.currentIndexChanged.connect(self._on_input_sigma)
        opt.addWidget(self.in_sig_sel)
        opt.addSpacing(10)
        opt.addWidget(QLabel('Stats:'))
        self.method_sel = QComboBox()
        self.method_sel.addItems(['WM + √MSWD', 'GLS + ML'])
        self.method_sel.setCurrentIndex(0)
        self.method_sel.setToolTip(
            'How the aliquot/sample ages are combined. Both options build the same\n'
            'covariance matrix (shared J-flux and FCs/λ, which do not average down);\n'
            'they differ only in the mean estimator and the overdispersion model.\n'
            'WM + √MSWD: inverse-variance weighted mean, analytical tier scaled by\n'
            '√MSWD when overdispersed.\n'
            'GLS + ML: generalised-least-squares mean on the covariance + additive\n'
            'maximum-likelihood random-effects overdispersion ω (Vermeesch 2015 §13 /\n'
            'IsoplotR random.effects=TRUE). Note: operates on reduced-data covariances\n'
            '(J, FCs, λ), not the full raw-beam propagation of Ar-Ar_Redux.\n'
            'Preferable when the ages scatter beyond their errors and those errors\n'
            'differ widely: ω evens out the weights.')
        self.method_sel.currentIndexChanged.connect(self._on_stat_method)
        opt.addWidget(self.method_sel)
        opt.addStretch()
        right.addLayout(opt)

        # calibration bar: one calibration for the whole session (two rows)
        cal1 = QHBoxLayout()
        cal1.addWidget(QLabel('Monitor:'))
        self.mon_sel = MonitorCombo()
        self.mon_sel.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.mon_sel.setMinimumContentsLength(6)
        self.mon_sel.activated.connect(self._on_calib_field)
        cal1.addWidget(self.mon_sel)
        cal1.addWidget(QLabel('Calib:'))
        cal = cal1
        self.calib_sel = QComboBox()
        self.calib_sel.addItems([r[0] for r in FC_PRESETS] + ['Custom…'])
        self.calib_sel.setToolTip(
            'FC monitor calibration, applied to all samples. Picking a preset\n'
            'sets t_FCs, its error, the FC err type and the decay constant the\n'
            'calibration is anchored to (e.g. Phillips 2022 → Min et al. 2000 λ).\n'
            'R_std is treated as the measured quantity: J is re-derived and all\n'
            'ratio samples are recalculated. Hand-edited fields show as Custom.')
        self.calib_sel.activated.connect(self._on_calib_preset)
        self.calib_sel.setMinimumWidth(150)
        self.calib_sel.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        cal.addWidget(self.calib_sel, 1)
        cal2 = QHBoxLayout()
        cal = cal2
        cal.addWidget(QLabel('t_FCs:'))
        self.cal_tfc = QLineEdit(f'{STD_AGE_DEFAULT:.5g}')
        self.cal_tfc.setFixedWidth(70)
        cal.addWidget(self.cal_tfc)
        cal.addWidget(QLabel('±'))
        self.cal_tfce = QLineEdit(f'{STD_AGE_ERR:.4g}')
        self.cal_tfce.setFixedWidth(60)
        cal.addWidget(self.cal_tfce)
        cal1.addWidget(QLabel('External:'))
        self.cal_external = QComboBox()
        self.cal_external.addItems(['Intercalibrated external', 'Full external'])
        self.cal_external.setToolTip(
            'Which external uncertainty to report:\n'
            ' Intercalibrated external (default): the monitor age with the decay\n'
            '          constant it was calibrated with. For Kuiper, Rivera and\n'
            '          Phillips it comes from R(FCs/A), the anchor age and λ.\n'
            '          This is also the error to use against U-Pb ages.\n'
            ' Full external (Min et al., 2000): the uncertainty of the two partial\n'
            '          ⁴⁰K decay constants (about 1.2% of the age, 1σ) in place of the λ term,\n'
            '          for an independent test of the ⁴⁰K clock.')
        self.cal_external.activated.connect(self._on_external)
        cal1.addWidget(self.cal_external)
        # stdAgeErrType is not a user control; it comes from the preset
        # (astronomical/Carter = 'external'); λ handling is automatic (R-value).
        self._cal_std_err_type = 'external'
        cal.addWidget(QLabel('λ:'))
        self.cal_lam = QLineEdit(f'{LAMBDA_DEFAULT:.6g}')
        self.cal_lam.setFixedWidth(95)
        cal.addWidget(self.cal_lam)
        cal.addWidget(QLabel('±'))
        self.cal_lame = QLineEdit(f'{LAMBDA_ERR:.4g}')
        self.cal_lame.setFixedWidth(75)
        cal.addWidget(self.cal_lame)
        cal.addStretch()
        for w in (self.cal_tfc, self.cal_tfce, self.cal_lam, self.cal_lame):
            w.editingFinished.connect(self._on_calib_field)
        # the preset fields are kept (sessions and the default sample read
        # them) but hidden: the calibration is set with Calibration…
        _old = QWidget()
        _ol = QVBoxLayout(_old)
        _ol.setContentsMargins(0, 0, 0, 0)
        _ol.addLayout(cal1)
        _ol.addLayout(cal2)
        _old.setVisible(False)
        right.addWidget(_old)
        cal1.removeWidget(self.cal_external)
        bar = QHBoxLayout()
        self.route_lbl = QLabel('')
        self.route_lbl.setWordWrap(True)
        self.route_lbl.setStyleSheet('color:#555; font-size:10px;')
        b_cal = QPushButton('Calibration…')
        b_cal.setToolTip('Monitor and calibration of the samples. Under Advanced: typed values, '
                         'a route built in Standards, and which samples it goes to.')
        b_cal.clicked.connect(self._open_calibrate)
        bar.addWidget(b_cal)
        self.cal_external.setVisible(False)      # set in Calibration… (Advanced)
        bar.addStretch()
        right.addLayout(bar)
        right.addWidget(self.route_lbl)
        self.mon_info = QLabel('')
        self.mon_info.setStyleSheet('color:#555; font-size:10px;')
        self.mon_info.setVisible(False)
        right.addWidget(self.mon_info)

        # scrollable sample list
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumWidth(360)
        holder = QWidget()
        self.panel_layout = QVBoxLayout(holder)
        self.panel_layout.setSpacing(8)
        self.panel_layout.addStretch()
        self.scroll.setWidget(holder)
        right.addWidget(self.scroll, 1)

        # bottom: session/file row then export row
        def _io_btn(txt, fn):
            b = QPushButton(txt)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(fn)
            return b

        io1 = QHBoxLayout()
        io1.setSpacing(4)
        for txt, fn in [('Save session', self.save_session),
                        ('Load session', self.load_session),
                        ('Load disk flux…', self.load_disk_flux)]:
            io1.addWidget(_io_btn(txt, fn))
        right.addLayout(io1)

        io2 = QHBoxLayout()
        io2.setSpacing(4)
        for txt, fn in [('Load stack…', self.load_stack),
                        ('Export figure…', self.export_figure),
                        ('Export XLSX / CSV', self.export_csv)]:
            io2.addWidget(_io_btn(txt, fn))
        # the two export buttons, relabelled while the Mixture view is on
        self._btn_export_fig, self._btn_export_data = (
            io2.itemAt(i).widget() for i in (1, 2))
        right.addLayout(io2)

        # cap the control column so a wide window goes to the plot, not the forms
        right_w = QWidget()
        right_w.setLayout(right)
        right_w.setMaximumWidth(660)
        root.addWidget(right_w, 0)

    # --- sample management --------------------------------------------------
    def _calib_bar_values(self):
        """Read the calibration bar. Returns a dict or None if a field is bad."""
        try:
            vals = dict(
                stdAgeMa=float(self.cal_tfc.text()),
                stdAgeErrMa=float(self.cal_tfce.text()),
                stdAgeErrType=self._cal_std_err_type,
                lambdaYr=float(self.cal_lam.text()),
                lambdaErrYr=float(self.cal_lame.text()),
                fullSystematics=(self.cal_external.currentIndex() == 1),
            )
        except (TypeError, ValueError):
            return None
        if vals['stdAgeMa'] <= 0 or vals['lambdaYr'] <= 0:
            return None
        vals.update(self.mon_sel.keys())
        return vals

    def _calib_bar_match(self):
        """FC preset index matching the bar values, or the Custom index."""
        vals = self._calib_bar_values()
        if vals is None:
            return len(FC_PRESETS)
        for i, (name, age, err, ext) in enumerate(FC_PRESETS):
            pl = paired_lambda(name)
            if pl is None:
                continue
            if (abs(vals['stdAgeMa'] - age) < 5e-4
                    and abs(vals['stdAgeErrMa'] - err) < 5e-5
                    and abs(vals['lambdaYr'] - pl[1]) < 1e-14
                    and vals['stdAgeErrType'] == ('external' if ext else 'internal')):
                return i
        return len(FC_PRESETS)

    def _on_calib_preset(self, idx):
        if idx >= len(FC_PRESETS):
            return                      # Custom: leave the fields alone
        name, age, err, ext = FC_PRESETS[idx]
        self.cal_tfc.setText(f'{age:.5g}')
        self.cal_tfce.setText(f'{err:.4g}')
        self._cal_std_err_type = 'external' if ext else 'internal'
        pl = paired_lambda(name)
        if pl is not None:
            self.cal_lam.setText(f'{pl[1]:.6g}')
            self.cal_lame.setText(f'{pl[2]:.4g}')
        self._apply_calibration()

    def _on_calib_field(self, *_):
        self.calib_sel.setCurrentIndex(self._calib_bar_match())
        self._apply_calibration()

    def _apply_calibration(self):
        """Write the bar's constants into every sample and recalculate.

        R_std is the measured monitor ratio and is calibration-independent,
        so J is re-derived from it under the new constants (the equivalent
        of J2 = J1*(e^(l2*t2)-1)/(e^(l1*t1)-1)) and ratio samples re-age.
        Holding J fixed instead would change R_std and shift the ages.
        """
        vals = self._calib_bar_values()
        if vals is None:
            return
        idx = self._calib_bar_match()
        vals['calibClass'] = (calibration_class(FC_PRESETS[idx][0])
                              if idx < len(FC_PRESETS) else 'other')
        # samples whose data came with another monitor than the bar's
        mism = [s.name for s in self.samples
                if len(s.ages) and cfg_monitor(s.j_cfg)[0] != vals['monitor']]
        notes = calibration_notes(vals)
        if mism:
            notes.append(monitor_mismatch_note(vals, cfg_monitor(
                next(s for s in self.samples if s.name == mism[0]).j_cfg)[0],
                'the data of ' + ', '.join(mism[:3]) + (' …' if len(mism) > 3 else '')))
        self._calib_notes = notes
        for s in self.samples:
            s.j_cfg.update(vals)
            if s.j_cfg.get('ageFluxMode') in ('ratio_J', 'ratio_R_std'):
                _sync_j_rstd(s.j_cfg, 'R')
            if s.has_R() and s.input_mode == 'ratio':
                s.recompute()
        txt = '\n'.join([t for t in [monitor_text(vals)] + notes if t])
        self.mon_info.setText(txt)
        self.mon_info.setWordWrap(True)
        self.mon_info.setVisible(bool(txt))
        for p in self.panels:
            p._update_jvis()
            p.sync_data_box()
            p.refresh_info()
        self.redraw()

    def _on_external(self, *_):
        full = self.cal_external.currentIndex() == 1
        for s in self.samples:
            s.j_cfg['fullSystematics'] = full
            if s.has_R() and s.input_mode == 'ratio':
                s.recompute()
        for p in self.panels:
            p.refresh_info()
        self._show_routes()
        self.redraw()

    def _open_calibrate(self):
        """Calibration… : the shared dialog, applied to the ticked samples."""
        if not self.samples:
            return
        lines = [(s.name, True, route_text(route_from_cfg(s.j_cfg))) for s in self.samples]
        dlg = CalibrationDialog(self.samples[0].j_cfg, parent=self, age_source=False,
                                stored=False, samples=lines if len(lines) > 1 else None,
                                full_external=self.cal_external.currentIndex() == 1)
        if dlg.exec_() != QDialog.Accepted:
            return
        p = dlg.get_params()
        targets = p['targets'] or [True] * len(self.samples)
        rkeys = p['route_keys']
        cal = dict(stdAgeMa=p['fc2_ma'], stdAgeErrMa=p['fc2_err_ma'],
                   stdAgeErrType='external' if p['fc2_external'] else 'internal',
                   lambdaYr=p['lam2'], lambdaErrYr=p['lam2_err'])
        notes = []
        for s, on in zip(self.samples, targets):
            if not on:
                # the others keep theirs, spelled out as a route, so shared
                # terms carry the same names when pooled
                set_route(s.j_cfg, route_cfg(route_from_cfg(s.j_cfg)))
                continue
            prod_old = cfg_monitor(s.j_cfg)[1]
            s.j_cfg.update(cal)
            set_route(s.j_cfg, rkeys)
            if s.j_cfg.get('rStd'):
                s.j_cfg['rStd'] = s.j_cfg['rStd'] * prod_old / cfg_monitor(s.j_cfg)[1]
            if s.j_cfg.get('ageFluxMode') in ('ratio_J', 'ratio_R_std'):
                _sync_j_rstd(s.j_cfg, 'R')
            if s.has_R() and s.input_mode == 'ratio':
                s.recompute()
            m = self._flux_match(s)
            if m is not None:
                n = monitor_mismatch_note(s.j_cfg, self._disk_flux[m[0]].get('monitor', 'FCs'),
                                          f'the disk flux of {s.name}')
                if n:
                    notes.append(n)
        if notes:
            QMessageBox.warning(self, 'Calibration', '\n\n'.join(notes))
        self._apply_flux_to_samples()
        if bool(p.get('full_external')) != (self.cal_external.currentIndex() == 1):
            self.cal_external.setCurrentIndex(1 if p.get('full_external') else 0)
            self._on_external()
        self._sync_calib_bar()
        for pnl in self.panels:
            pnl._update_jvis()
            pnl.sync_data_box()
            pnl.refresh_info()
        self.redraw()

    def _show_routes(self):
        """The calibration line: one route, or each sample's when they differ,
        with a note when samples end on different calibrations."""
        if not hasattr(self, 'route_lbl') or not self.samples:
            return
        rs = [(s.name, route_from_cfg(s.j_cfg)) for s in self.samples]
        lines = [route_line(r) for _, r in rs]
        full = ' · full external' if self.cal_external.currentIndex() == 1 else ''
        if len(set(lines)) == 1:
            self.route_lbl.setText(lines[0] + full)
            self.route_lbl.setToolTip(route_text(rs[0][1]))
        else:
            self.route_lbl.setText(f'{len(set(lines))} calibrations (hover for each sample){full}')
            self.route_lbl.setToolTip('\n'.join(f'{n}: {l}' for (n, _), l in zip(rs, lines)))
        ends = {(str(r.get('calibration')), str(r.get('lambda'))) for _, r in rs}
        note = ('The samples end on different calibrations; a pooled age mixes them. '
                'Calibration… can put them on one.' if len(ends) > 1 else '')
        txt = '\n'.join(t for t in [note] + list(getattr(self, '_calib_notes', []) or []) if t)
        self.mon_info.setWordWrap(True)
        self.mon_info.setText(txt)
        self.mon_info.setVisible(bool(txt))

    def _sync_calib_bar(self):
        """Reflect the samples' calibration in the bar (after loads).

        Uniform across samples -> show it and match the preset; mixed ->
        show the first sample's values with the preset set to Custom.
        """
        if not self.samples:
            return
        cfgs = [s.j_cfg for s in self.samples]
        c0 = cfgs[0]
        keys = ('stdAgeMa', 'stdAgeErrMa', 'stdAgeErrType', 'lambdaYr', 'lambdaErrYr', 'monitor')
        uniform = all(all(c.get(k) == c0.get(k) for k in keys) for c in cfgs[1:])
        self.mon_sel.set_monitor(cfg_monitor(c0)[0])
        txt = '\n'.join([t for t in [monitor_text(c0)] + calibration_notes(c0) if t])
        self.mon_info.setWordWrap(True)
        self.mon_info.setText(txt)
        self.mon_info.setVisible(bool(txt))
        self.cal_tfc.setText(f"{c0.get('stdAgeMa', STD_AGE_DEFAULT):.5g}")
        self.cal_tfce.setText(f"{c0.get('stdAgeErrMa', STD_AGE_ERR):.4g}")
        self._cal_std_err_type = c0.get('stdAgeErrType', 'external')
        self.cal_lam.setText(f"{c0.get('lambdaYr', LAMBDA_DEFAULT):.6g}")
        self.cal_lam.setCursorPosition(0)
        self.cal_lame.setText(f"{c0.get('lambdaErrYr', LAMBDA_ERR):.4g}")
        self.cal_external.blockSignals(True)
        self.cal_external.setCurrentIndex(1 if c0.get('fullSystematics') else 0)
        self.cal_external.blockSignals(False)
        self.cal_lame.setCursorPosition(0)
        self.calib_sel.setCurrentIndex(
            self._calib_bar_match() if uniform else len(FC_PRESETS))
        self._show_routes()

    def add_sample(self, sample=None):
        if sample is None:
            used = {s.color for s in self.samples}
            color = next((c for c in _PALETTE if c not in used),
                         _PALETTE[len(self.samples) % len(_PALETTE)])
            sample = Sample(name=f'Sample {len(self.samples)+1}', color=color)
            vals = self._calib_bar_values()
            if self.samples:            # the calibration of the first sample
                c0 = self.samples[0].j_cfg
                sample.j_cfg.update({k: copy.deepcopy(c0[k]) for k in
                                     CALIB_KEYS + ROUTE_KEYS + ('fullSystematics',) if k in c0})
            elif vals:
                idx = self._calib_bar_match()
                vals['calibClass'] = (calibration_class(FC_PRESETS[idx][0])
                                      if idx < len(FC_PRESETS) else 'other')
                sample.j_cfg.update(vals)
        self.samples.append(sample)
        sample._app = self
        panel = SamplePanel(sample, self, len(self.samples) - 1)
        self.panels.append(panel)
        self.panel_layout.insertWidget(self.panel_layout.count() - 1, panel)
        return sample

    def remove_sample(self, sample):
        if sample not in self.samples:
            return
        i = self.samples.index(sample)
        self.samples.pop(i)
        p = self.panels.pop(i)
        p.setParent(None)
        p.deleteLater()
        self.redraw()

    # --- file ops -----------------------------------------------------------
    def load_into(self, sample):
        paths, _ = dfold.get_open_file_names(
            self, 'Load .arar / .ararpt / .ararstack / .arprob', '',
            'Ar-Ar (*.arar *.ararpt *.ararstack *.arprob);;All (*)')
        if not paths:
            return
        # One Age distribution session: like a stack, its samples load as
        # separate samples or as points of this one.
        if len(paths) == 1 and paths[0].lower().endswith('.arprob'):
            try:
                sds, flux = self._read_arprob(paths[0])
            except Exception as e:
                QMessageBox.warning(self, 'Load', f'{os.path.basename(paths[0])}: {e}')
                return
            merge_disk_flux(self._disk_flux, flux)
            if not sds:
                QMessageBox.information(self, 'Load',
                                        f'{os.path.basename(paths[0])} holds no samples.')
                return
            if len(sds) > 1:
                ans = QMessageBox.question(
                    self, 'Load session',
                    f'{os.path.basename(paths[0])} holds {len(sds)} samples.\n\n'
                    'Load each as a separate sample?\n\n'
                    'Yes: one sample each, as saved (calibration, exclusions, '
                    'ticked or not).\n\n'
                    'No: all their points in this sample, with their R values put '
                    'on the first sample\'s J (each keeps its own J error).',
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.Yes)
                if ans == QMessageBox.Cancel:
                    return
                if ans == QMessageBox.No:
                    self._load_docs(sample, [_arprob_doc(sd) for sd in sds])
                    return
            self._load_saved_samples(sample, sds)
            return
        # One stack with several aliquots: offer one sample per aliquot, so each
        # keeps its own calibration and the pool combines them.
        if len(paths) == 1 and paths[0].lower().endswith('.ararstack'):
            try:
                als = parse_ararstack_aliquots(paths[0])
            except Exception as e:
                QMessageBox.warning(self, 'Load', f'{os.path.basename(paths[0])}: {e}')
                return
            self._merge_stack_flux(paths[0])
            if len(als) > 1:
                n_ex = sum(a['excluded'] for a in als)
                ans = QMessageBox.question(
                    self, 'Load stack',
                    f'{os.path.basename(paths[0])} holds {len(als)} aliquots'
                    + (f' ({n_ex} excluded in the Plateau stack)' if n_ex else '')
                    + '.\n\nLoad each aliquot as a separate sample?\n\n'
                    'Yes: one sample per aliquot, each on its own R_std / J, and '
                    'Pool switched on to combine them. Excluded aliquots load '
                    'unticked.\n\n'
                    'No: the aliquots that count in the Plateau stack, in this '
                    'sample as one plateau age each, with their R values put on '
                    'the first aliquot\'s J (each keeps its own J error). '
                    'Excluded aliquots are left out.',
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.Yes)
                if ans == QMessageBox.Cancel:
                    return
                if ans == QMessageBox.Yes:
                    self._load_aliquots_as_samples(sample, als)
                    return
                # one plateau age per aliquot that counts (n = those); the
                # ×-ticked ones stay out, so nothing off to the side of the plot
                als_in = [a for a in als if not a['excluded']] or als
                d = parse_ararstack_combined(paths[0], als_in)
                if d is not None:
                    self._load_docs(sample, [d])
                return
        # Expand every selected file into parse dicts.  A .ararstack collapses to
        # one point per aliquot = that aliquot's combined (plateau) age: all
        # aliquots load, and the ×-ticked (session-excluded) ones come through
        # marked rejected so they show on the plot without entering the stats.
        # A .arar/.ararpt still loads its individual steps.
        docs = []
        for p in paths:
            try:
                if p.lower().endswith('.ararstack'):
                    d = parse_ararstack_combined(p)
                    self._merge_stack_flux(p)
                    if d is not None:
                        docs.append(d)
                elif p.lower().endswith('.arprob'):
                    sds, flux = self._read_arprob(p)
                    merge_disk_flux(self._disk_flux, flux)
                    docs += [_arprob_doc(sd) for sd in sds if sd.get('ages')]
                else:
                    d = parse_arar(p)
                    if d is not None:
                        docs.append(d)
            except Exception as e:
                QMessageBox.warning(self, 'Load', f'{os.path.basename(p)}: {e}')
        if not docs:
            return
        self._load_docs(sample, docs)

    @staticmethod
    def _read_arprob(path):
        """(saved sample dicts with points, disk flux) of an .arprob session."""
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'samples' not in data:
            raise ValueError('not an Age distribution session')
        sds = [sd for sd in data.get('samples', []) if sd.get('ages')]
        return sds, deserialize_disk_flux(data.get('disk_flux', []))

    def _load_saved_samples(self, sample, sds):
        """The samples of an .arprob session: the first fills `sample` (which
        keeps its colour), the rest are added after it, each as saved."""
        used = {s.color for s in self.samples if s is not sample}
        for k, sd in enumerate(sds):
            new = _sample_from_dict(sd)
            if k == 0:
                color = sample.color
                sample.__dict__.update(new.__dict__)
                sample.color = color
                self._rebuild_panel(sample)
            else:
                if new.color in used:
                    new.color = next((c for c in _PALETTE if c not in used),
                                     _PALETTE[len(self.samples) % len(_PALETTE)])
                self.add_sample(new)
            used.add(sample.color if k == 0 else new.color)
        if self._disk_flux:
            self._apply_flux_to_samples()
        self.redraw()

    def _load_aliquots_as_samples(self, sample, als):
        """One sample per stack aliquot: the first fills `sample`, the rest are
        added after it.  Each is a single R point on its own calibration;
        aliquots excluded in the Plateau stack load unticked."""
        used = {s.color for s in self.samples if s is not sample}
        for k, a in enumerate(als):
            if k == 0:
                s = sample
            else:
                color = next((c for c in _PALETTE if c not in used),
                             _PALETTE[len(self.samples) % len(_PALETTE)])
                s = self.add_sample(Sample(name=a['name'], color=color))
            used.add(s.color)
            s.enabled = not a['excluded']
            self._load_docs(s, [a], redraw=False)
        if not self.pool_on:
            self.pool_btn.setChecked(True)      # redraws
        else:
            self.redraw()

    def _load_docs(self, sample, docs, redraw=True, reject='none', name=None):
        """Fill `sample` from parsed file dicts (parse_arar, parse_ararstack,
        parse_ararstack_combined / parse_ararstack_aliquots).  reject is the
        sample's outlier rejection ('none': only "No" steps start excluded);
        name replaces the joined file names."""
        # The merged sample uses the first file's calibration; files from other
        # irradiation positions keep their own ages via their own J.
        J_ref = _flux_J(docs[0]['j_cfg'])
        for d in docs[1:]:
            _rescale_to_J(d, _flux_J(d['j_cfg']), J_ref)
        names, all_ages, all_uncs, all_rv, all_ru, all_man = [], [], [], [], [], []
        all_kca, all_kca_e, all_ar40, all_labels, all_run_ids = [], [], [], [], []
        all_metas, all_consts = [], {}
        j_cfg = None
        r_ok = True
        for d in docs:
            names.append(d['name'])
            all_ages += d['ages']
            all_uncs += d['unc']        # already 1σ from parse_arar
            all_man += d.get('manual', ['auto'] * len(d['ages']))
            if len(d['r_vals']) == len(d['ages']):
                all_rv += d['r_vals']
                all_ru += d['r_uncs']   # already 1σ from parse_arar
            else:
                r_ok = False
            kca_d = d.get('kca_vals', [])
            kca_ed = d.get('kca_errs', [])
            ar40_d = d.get('ar40_pct', [])
            lbl_d  = d.get('labels', [])
            rid_d  = d.get('run_ids', [])
            n_d = len(d['ages'])
            all_kca      += kca_d  if len(kca_d)  == n_d else [float('nan')] * n_d
            all_kca_e    += kca_ed if len(kca_ed) == n_d else [float('nan')] * n_d
            all_ar40     += ar40_d if len(ar40_d) == n_d else [float('nan')] * n_d
            all_labels   += lbl_d  if len(lbl_d)  == n_d else [''] * n_d
            all_run_ids  += rid_d  if len(rid_d)  == n_d else [''] * n_d
            all_metas    += _metas_of(d, n_d)
            for _k, _v in (d.get('constants') or {}).items():
                all_consts.setdefault(_k, _v)
            if j_cfg is None:
                j_cfg = d['j_cfg']
                sample.irradiation = d.get('irradiation')
                sample.disk_id = d.get('disk_id')
                sample.hole_id = d.get('hole_id')
        if not all_ages:
            return
        sample.name = name or ' + '.join(names)
        sample.raw_name = _join_names(d.get('raw_name') or d['name'] for d in docs)
        sample.j_cfg = j_cfg or _default_j_cfg()
        sample.pt_jfrac, sample.pt_pos = _doc_positions(docs)
        # a file saved with its flux hole assigned (Plateau & isochron tool) keeps that hole
        _fh = docs[0].get('flux_hole') if len(docs) == 1 else None
        if isinstance(_fh, (list, tuple)) and len(_fh) == 3:
            sample._flux_override = tuple(str(x) for x in _fh)
        sample.in_sigma = 1   # parse_arar always normalises to 1σ
        sample.r_vals = np.array(all_rv, float) if (r_ok and all_rv) else np.array([], float)
        sample.r_uncs = np.array(all_ru, float) if (r_ok and all_ru) else np.array([], float)
        n_all = len(all_ages)
        sample.kca_vals = np.array(all_kca,   float) if len(all_kca)   == n_all else np.array([], float)
        sample.kca_errs = np.array(all_kca_e, float) if len(all_kca_e) == n_all else np.array([], float)
        sample.ar40_pct = np.array(all_ar40,  float) if len(all_ar40)  == n_all else np.array([], float)
        sample.labels   = all_labels   if len(all_labels)   == n_all else []
        sample.run_ids  = all_run_ids  if len(all_run_ids)  == n_all else []
        sample.metas    = all_metas    if len(all_metas)    == n_all else []
        sample.constants = all_consts
        sample._from_file = True   # raw .arar values, read at the import-sigma level
        sample.reject = reject
        if sample.has_R():
            # R-centric: ignore the stored age column entirely; the age of every
            # step is computed from R_unk and the J calibration (J / R_std).
            sample.input_mode = 'ratio'
            err = sample.recompute()
            if err:
                sample.input_mode = 'age'
                sample.r_vals = np.array([], float)
                sample.r_uncs = np.array([], float)
                sample.set_ages(all_ages, all_uncs)
                QMessageBox.warning(self, 'Load', f'R->Age: {err}; showing stored ages.')
        else:
            sample.input_mode = 'age'
            sample.set_ages(all_ages, all_uncs)
        # restore the per-step include/exclude flags ("No" -> manual reject)
        if len(all_man) == len(sample.ages):
            sample.manual = np.array(all_man, dtype=object)
            sample._loaded_excluded = np.array([m == 'out' for m in all_man], bool)
            sample.compute_auto()
        # on a hole of the loaded disk flux: that hole's J, not the file's
        if self._disk_flux:
            self._apply_flux_to_samples()
        self._rebuild_panel(sample)
        if redraw:
            self.redraw()

    def load_stack(self):
        """Load an .ararstack file.

        Shows a small options dialog first.  By default each aliquot becomes
        its own Sample.  With 'Load as single sample' checked all steps are
        merged into one Sample so the weighted mean / MSWD operates on the
        full step population.  The steps are each aliquot's saved plateau
        window; stacks saved without windows fall back to NMAD rejection.
        """
        path, _ = dfold.get_open_file_name(
            self, 'Load .ararstack', '',
            'Ar-Ar Stack (*.ararstack);;All (*)')
        if not path:
            return
        try:
            aliquots = parse_ararstack(path)
        except Exception as e:
            QMessageBox.warning(self, 'Load Stack', str(e))
            return
        if not aliquots:
            QMessageBox.information(self, 'Load Stack',
                                    'No aliquots found in file.')
            return

        # ── Options dialog ────────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle('Load Stack')
        dv = QVBoxLayout(dlg)
        dv.addWidget(QLabel(
            f'{len(aliquots)} aliquot(s) found in:\n{os.path.basename(path)}'))
        dv.addSpacing(6)
        cb_single = QCheckBox('Load as single sample')
        cb_single.setToolTip(
            'Merge all aliquots into one Sample.\n'
            'The weighted mean (and NMAD rejection, for stacks without saved\n'
            'plateau windows) then apply to the entire step population rather\n'
            'than aliquot-by-aliquot.')
        dv.addWidget(cb_single)
        dv.addSpacing(8)
        bh = QHBoxLayout()
        bh.addStretch()
        ok_btn     = QPushButton('Load');   ok_btn.setDefault(True)
        cancel_btn = QPushButton('Cancel')
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        bh.addWidget(ok_btn); bh.addWidget(cancel_btn)
        dv.addLayout(bh)
        dlg.setMinimumWidth(320)
        if dlg.exec_() != QDialog.Accepted:
            return

        single = cb_single.isChecked()
        # The saved plateau windows select the steps (as "Load" does); NMAD
        # rejection only for older stacks that do not store them.
        reject = 'none' if all(d.get('has_window') for d in aliquots) else 'nmad'
        self._merge_stack_flux(path)
        used_colors = {s.color for s in self.samples}

        def _color():
            c = next((c for c in _PALETTE if c not in used_colors),
                     _PALETTE[len(self.samples) % len(_PALETTE)])
            used_colors.add(c)
            return c

        if single:
            # one sample, one calibration: _load_docs puts the other positions'
            # R on the first aliquot's J, so their ages do not move
            sample = self.add_sample(Sample(name=os.path.splitext(os.path.basename(path))[0],
                                            color=_color()))
            self._load_docs(sample, aliquots, redraw=False, reject=reject,
                            name=sample.name)
            msg = (f'Loaded {len(aliquots)} aliquots merged into one sample\n'
                   f'({len(sample.ages)} steps total) from:\n{os.path.basename(path)}')
        else:
            for d in aliquots:
                sample = self.add_sample(Sample(name=d['name'], color=_color()))
                sample.is_plateau = True   # a stacked plateau = one aliquot → one pooled point (n = aliquots)
                self._load_docs(sample, [d], redraw=False,
                                reject='none' if d.get('has_window') else 'nmad')
            msg = (f'Loaded {len(aliquots)} aliquot(s) as separate samples from:\n'
                   f'{os.path.basename(path)}')

        self.redraw()
        QMessageBox.information(self, 'Load Stack', msg)

    def append_into(self, sample):
        paths, _ = dfold.get_open_file_names(
            self, 'Append .arar / .ararpt', '',
            'Ar-Ar (*.arar *.ararpt);;All (*)')
        if not paths:
            return
        if len(sample.ages) == 0:
            # nothing to append to: load the files, so their R values are kept
            docs = []
            for p in paths:
                try:
                    d = parse_arar(p)
                except Exception as e:
                    QMessageBox.warning(self, 'Append', f'{os.path.basename(p)}: {e}')
                    continue
                if d is not None:
                    docs.append(d)
            if docs:
                self._load_docs(sample, docs, reject='none')
            return
        new_a, new_u, new_rv, new_ru, new_man = [], [], [], [], []
        new_kca, new_kca_e, new_ar40, new_labels, new_run_ids = [], [], [], [], []
        new_metas = []
        new_names = []
        app_r = True
        for p in paths:
            try:
                d = parse_arar(p)
            except Exception as e:
                QMessageBox.warning(self, 'Append', f'{os.path.basename(p)}: {e}')
                continue
            if d is None:
                continue
            new_names.append(d.get('raw_name') or d['name'])
            new_a += d['ages']
            new_u += d['unc']
            new_man += d.get('manual', ['auto'] * len(d['ages']))
            if len(d['r_vals']) == len(d['ages']):
                new_rv += d['r_vals']
                new_ru += d['r_uncs']
            else:
                app_r = False
            kca_d = d.get('kca_vals', [])
            kca_ed = d.get('kca_errs', [])
            ar40_d = d.get('ar40_pct', [])
            lbl_d  = d.get('labels', [])
            rid_d  = d.get('run_ids', [])
            n_d = len(d['ages'])
            new_kca      += kca_d  if len(kca_d)  == n_d else [float('nan')] * n_d
            new_kca_e    += kca_ed if len(kca_ed) == n_d else [float('nan')] * n_d
            new_ar40     += ar40_d if len(ar40_d) == n_d else [float('nan')] * n_d
            new_labels   += lbl_d  if len(lbl_d)  == n_d else [''] * n_d
            new_run_ids  += rid_d  if len(rid_d)  == n_d else [''] * n_d
            new_metas    += _metas_of(d, n_d)
            for _k, _v in (d.get('constants') or {}).items():
                sample.constants.setdefault(_k, _v)
        if not new_a:
            return
        old_man = sample.manual.tolist() if len(sample.manual) == len(sample.ages) \
            else ['auto'] * len(sample.ages)
        # parse_arar already normalises to 1σ; no further conversion needed
        old_r_ok = len(sample.r_vals) == len(sample.ages) and len(sample.ages) > 0
        keep_r = old_r_ok and app_r
        n_old_ages = len(sample.ages)
        a = sample.ages.tolist() + new_a
        u = sample.unc_2s.tolist() + new_u
        if keep_r:
            sample.r_vals = np.array(sample.r_vals.tolist() + new_rv, float)
            sample.r_uncs = np.array(sample.r_uncs.tolist() + new_ru, float)
        else:
            sample.r_vals = np.array([], float)
            sample.r_uncs = np.array([], float)
            if old_r_ok or not app_r:
                QMessageBox.information(
                    self, 'R mode disabled',
                    'Appended data mixes files with and without raw ratios; '
                    'R mode is off for this sample. Ages combine normally.')
        old_kca_ok = len(sample.kca_vals) == len(sample.ages) and len(sample.ages) > 0
        if old_kca_ok and len(new_kca) == len(new_a):
            sample.kca_vals = np.concatenate([sample.kca_vals, np.array(new_kca,   float)])
            sample.kca_errs = np.concatenate([sample.kca_errs, np.array(new_kca_e, float)])
            sample.ar40_pct = np.concatenate([sample.ar40_pct, np.array(new_ar40,  float)])
        else:
            sample.kca_vals = np.array([], float)
            sample.kca_errs = np.array([], float)
            sample.ar40_pct = np.array([], float)
        old_lbl_ok = len(sample.labels) == len(sample.ages) and len(sample.ages) > 0
        if old_lbl_ok and len(new_labels) == len(new_a):
            sample.labels = sample.labels + new_labels
        else:
            sample.labels = []
        old_rid_ok = len(sample.run_ids) == len(sample.ages) and len(sample.ages) > 0
        if old_rid_ok and len(new_run_ids) == len(new_a):
            sample.run_ids = sample.run_ids + new_run_ids
        else:
            sample.run_ids = []
        if old_rid_ok and len(sample.metas) == len(sample.ages) and len(new_metas) == len(new_a):
            sample.metas = sample.metas + new_metas
        else:
            sample.metas = []
        sample._from_file = True
        if new_names:
            sample.raw_name = _join_names(
                [sample.experiment_name() or sample.name] + new_names)
        sample.reject = 'none'     # appended data: only "No" steps start excluded
        if keep_r and len(sample.r_vals):
            sample.input_mode = 'ratio'
            err = sample.recompute()
            if err:
                sample.input_mode = 'age'
                sample.r_vals = np.array([], float)
                sample.r_uncs = np.array([], float)
                sample.set_ages(a, u)
        else:
            sample.input_mode = 'age'
            sample.set_ages(a, u)
        if sample.pt_pos is not None and len(sample.pt_pos) == n_old_ages:
            ref = next((p for p, j in zip(sample.pt_pos, sample.pt_jfrac)
                        if np.isnan(j)), sample.pt_pos[0])
            sample.pt_pos = list(sample.pt_pos) + [ref] * len(new_a)
            sample.pt_jfrac = np.concatenate([sample.pt_jfrac, np.full(len(new_a), np.nan)])
        combined_man = old_man + new_man
        if len(combined_man) == len(sample.ages):
            sample.manual = np.array(combined_man, dtype=object)
            old_excl = sample._loaded_excluded
            new_excl = np.array([m == 'out' for m in new_man], bool)
            if len(old_excl) == n_old_ages:
                sample._loaded_excluded = np.concatenate([old_excl, new_excl])
            else:
                sample._loaded_excluded = np.concatenate(
                    [np.zeros(n_old_ages, bool), new_excl])
            sample.compute_auto()
        self._rebuild_panel(sample)
        self.redraw()

    def import_into(self, sample):
        path, _ = dfold.get_open_file_name(
            self, 'Import table', '', 'Spreadsheet (*.xlsx *.csv);;All (*)')
        if not path:
            return
        try:
            ages, uncs = parse_table(path)
        except Exception as e:
            QMessageBox.warning(self, 'Import', str(e))
            return
        sample.r_vals = np.array([], float)
        sample.r_uncs = np.array([], float)
        sample.kca_vals = np.array([], float)
        sample.kca_errs = np.array([], float)
        sample.ar40_pct = np.array([], float)
        sample.labels   = []
        sample.run_ids  = []
        sample.metas    = []
        sample.constants = {}
        sample.pt_jfrac = sample.pt_pos = None
        sample.input_mode = 'age'
        sample._from_file = False
        sample.in_sigma = self.input_sigma
        uncs = [u / self.input_sigma for u in uncs]
        sample.set_ages(ages, uncs)
        self._rebuild_panel(sample)
        self.redraw()

    def _rebuild_panel(self, sample):
        i = self.samples.index(sample)
        old = self.panels[i]
        new = SamplePanel(sample, self, i)
        self.panel_layout.replaceWidget(old, new)
        old.setParent(None)
        old.deleteLater()
        self.panels[i] = new

    # --- toolbar callbacks --------------------------------------------------
    def _on_pool(self, on):
        self.pool_on = on
        self.pool_btn.setText(f'Pool: {"on" if on else "off"}')
        self.redraw()

    def _on_corr(self, on):
        self.corr_j = on
        self.redraw()

    # --- axis / scale controls ---------------------------------------------
    def _apply_axes(self):
        def fv(le):
            try:
                return float(le.text())
            except ValueError:
                return None
        lo, hi, ym = fv(self.x_lo_ed), fv(self.x_hi_ed), fv(self.y_max_ed)
        if lo is not None and hi is not None and hi > lo:
            self.xlim_mode, self.x_lo, self.x_hi = 'manual', lo, hi
        if ym is not None and ym > 0:
            self.ylim_mode, self.y_max = 'manual', ym
        self.redraw()

    def _auto_axes(self):
        self.xlim_mode = 'auto'
        self.ylim_mode = 'auto'
        self.x_lo_ed.clear()
        self.x_hi_ed.clear()
        self.y_max_ed.clear()
        self.redraw()

    def _data_range(self, included_only):
        los, his = [], []
        for s in self._active_all():
            m = s.mask() if included_only else np.ones(len(s.ages), bool)
            if m.sum() == 0:
                continue
            a = s.ages[m]
            sig = s.sig1()[m] * self.display_sigma
            los.append((a - sig).min())
            his.append((a + sig).max())
        if not los:
            return None
        pad = 0.05 * (max(his) - min(los) + 0.01)
        return min(los) - pad, max(his) + pad

    def _fit_included(self):
        rng = self._data_range(True)
        if rng:
            self.xlim_mode, self.x_lo, self.x_hi = 'manual', rng[0], rng[1]
            self.x_lo_ed.setText(f'{rng[0]:.3f}')
            self.x_hi_ed.setText(f'{rng[1]:.3f}')
            self.redraw()

    def _show_all(self):
        rng = self._data_range(False)
        if rng:
            self.xlim_mode, self.x_lo, self.x_hi = 'manual', rng[0], rng[1]
            self.x_lo_ed.setText(f'{rng[0]:.3f}')
            self.x_hi_ed.setText(f'{rng[1]:.3f}')
            self.redraw()

    def _on_sigma(self, _):
        self.display_sigma = self.sig_sel.currentIndex() + 1
        for p in self.panels:
            p.refresh_info()
        self.redraw()

    def _on_decimals(self, val):
        self.age_dec = int(val)
        self.redraw()

    def _on_stat_method(self, _):
        """Select the pooling statistics method (module-level so Sample.wm sees it)."""
        global _STAT_METHOD
        _STAT_METHOD = METHOD_GLS if self.method_sel.currentIndex() == 1 else METHOD_WM
        self.redraw()

    def _on_ymode(self, _):
        """Switch Y-axis presentation. Drawing only, statistics are unchanged."""
        self._ylim_saved[self.y_mode] = (self.ylim_mode, self.y_max)
        self.y_mode = {0: 'rank', 1: 'relative', 2: 'probability',
                       3: 'normalised'}.get(self.ymode_sel.currentIndex(), 'rank')
        self.ylim_mode, self.y_max = self._ylim_saved.get(self.y_mode, ('auto', None))
        if self.y_max is not None:
            self.y_max_ed.setText(f'{self.y_max:.3f}')
        else:
            self.y_max_ed.clear()
        self.redraw()

    def _on_input_sigma(self, _):
        """Set the sigma-level assumed for incoming data (paste/import/load).

        Combo order is ['1σ','2σ']. Retroactive rescaling is per-sample (each
        panel has its own sigma-in), so this only changes what new data assumes.
        """
        self.input_sigma = 1 if self.in_sig_sel.currentIndex() == 0 else 2

    def _on_canvas_click(self, event):
        """Click a data point to cycle its state: auto -> exclude -> include -> auto."""
        if event.inaxes is not self.ax or event.xdata is None or not self._click_pts:
            return
        # nearest point in display (pixel) space
        trans = self.ax.transData.transform
        click = np.array([event.x, event.y])
        best, best_d = None, 1e9
        for (s, i, age, rank) in self._click_pts:
            px = trans((age, rank))
            d = math.hypot(px[0] - click[0], px[1] - click[1])
            if d < best_d:
                best_d, best = d, (s, i)
        if best is None or best_d > 14:   # px radius
            return
        s, i = best
        cur = s.manual[i]
        eff = s.mask()[i]
        if cur in ('in', 'out'):
            s.manual[i] = 'auto'
        elif eff:
            s.manual[i] = 'out'
        else:
            s.manual[i] = 'in'
        s.compute_auto()
        self.redraw()
        for p in self.panels:
            if p.s is s:
                p.refresh_info()

    # --- combination --------------------------------------------------------
    def load_disk_flux(self):
        """Open the shared disk-flux window: import .jflux files, view disk
        designs and J values, and assign samples to (disk, hole), auto by
        identity with manual override."""
        dlg = FluxAssignDialog(self, _ProbFluxHost(self))
        dlg.exec_()
        self.redraw()

    def _import_jflux_files(self, paths, error_parent=None):
        """Read .jflux files into the registry (arar_math.read_jflux, which
        keeps the flux's own λ / FC so a hole is applied as an R value).
        Returns the number of disks loaded."""
        parent = error_parent or self
        loaded = 0
        for p in paths:
            if not p:   # macOS native dialog can return [''] on some Qt versions
                continue
            try:
                new = read_jflux(p)
            except Exception as e:
                QMessageBox.warning(parent, 'Load disk flux', f'{os.path.basename(p)}: {e}')
                continue
            self._disk_flux.update(new)
            loaded += len(new)
            _mons = {fx.get('monitor', 'FCs') for fx in new.values()}
            _vals = self._calib_bar_values()
            if _vals:
                _mn = monitor_mismatch_note(_vals, _mons.pop() if len(_mons) == 1 else 'FCs',
                                            f'the flux in {os.path.basename(p)}')
                if _mn:
                    QMessageBox.information(parent, 'Load disk flux', _mn)
            if new:
                self._preferred_flux = list(new)[-1]   # newest load wins ties on re-match
        return loaded

    def _merge_stack_flux(self, path):
        """Add the disk flux saved inside an .ararstack to the registry (disks
        already loaded are kept), so the J correlation between its aliquots is
        the one the Plateau stack used."""
        try:
            return merge_disk_flux(self._disk_flux, read_stack_disk_flux(path))
        except Exception:
            return 0

    def _apply_flux_to_samples(self):
        """For each sample matched to a loaded disk-flux hole, set its J and J%
        from the flux model (single source of truth = the regression).

        Samples with ratios are re-aged from the stored R with the hole's J in
        either input mode, so age and uncertainty move together and pool and
        append agree.  A sample with ages only keeps its ages; the J then moves
        the internal and external tiers only.
        """
        applied = 0
        for s in self.samples:
            m = self._flux_match(s)
            if m is None:
                # taken off its hole: back to the J it had before the flux
                base = getattr(s, '_flux_base_jcfg', None)
                if s._flux_src is not None and base is not None:
                    s.j_cfg = dict(base)
                    s._flux_base_jcfg = None
                    s._flux_src = None
                    if s.has_R():
                        s.recompute()
                continue
            if getattr(s, '_flux_base_jcfg', None) is None:
                s._flux_base_jcfg = dict(s.j_cfg)
            key, i = m
            fx = self._disk_flux[key]
            J = float(fx['J'][i])               # legacy fallback only
            rel = flux_hole_rel(fx, i) or 0.0
            # Apply the flux as an R value, not a raw J (arar_math.flux_hole_R:
            # the R stored in the .jflux, else (e^{λ_f·t_f} − 1)/J from the
            # flux's own calibration), then set ratio_R_std so the age is
            # computed from R with the sample's current λ / FC.  Changing the
            # calibration, or applying a flux from another calibration, then
            # re-derives the age instead of jumping it.
            r_std = flux_hole_rstd(fx, i, s.j_cfg)
            if r_std is not None and r_std > 0:
                s.j_cfg['ageFluxMode'] = 'ratio_R_std'
                s.j_cfg['rStd'] = r_std
                s.j_cfg['rStdErrMode'] = 'pct'
                s.j_cfg['rStdErrPct'] = rel * 100.0
                _sync_j_rstd(s.j_cfg, 'R')   # keep jVal in sync (for display) at the current calib
            else:
                # Legacy flux with no stored calibration → fall back to raw J.
                s.j_cfg['ageFluxMode'] = 'ratio_J'
                s.j_cfg['jVal'] = J
                s.j_cfg['jErrPct'] = rel * 100.0 if J else s.j_cfg.get('jErrPct', 0.1)
                _sync_j_rstd(s.j_cfg, 'J')
            s._flux_src = key
            # Disk flux is the single source of truth for J: re-age from the
            # (J-independent) ratios so age and uncertainty both update, in any
            # input mode.
            if s.has_R():
                s.input_mode = 'ratio'
                # The hole's J replaces the file J of every step, so it acts on the
                # measured R.  (A mixed-position sample keeps its file-J scaling
                # unless the hole was assigned by hand.)
                if s.pt_pos is None or getattr(s, '_flux_override', None) is not None:
                    s.use_measured_R()
                s.recompute()
            applied += 1
        for p in self.panels:
            # rebuild panels so the J fields reflect the applied values
            self._rebuild_panel(p.s)
        return applied

    def _flux_match(self, s):
        """Find the loaded disk-flux entry for sample s. Returns (key, idx) or None.

        A manual override (irr, disk, hole) takes precedence; otherwise match by
        disk + hole, with irradiation only to disambiguate. Handles hole_id like
        'E11' (disk-prefixed) vs flux holes like '11'.
        """
        if getattr(s, '_flux_override', None) is not None:
            irr, disk, hole = s._flux_override
            if (str(irr), str(disk)) == ('NONE', 'NONE'):
                return None              # explicitly set to "none"
            key = (str(irr), str(disk))
            fx = self._disk_flux.get(key)
            if fx is not None and str(hole) in fx['idx']:
                return (key, fx['idx'][str(hole)])
            return None
        return self._flux_match_ids(s.disk_id, s.hole_id, s.irradiation)

    def _flux_match_point(self, pos):
        """Loaded-flux hole of one point from its position info (see
        _pos_info): the hole the Plateau & isochron tool had it assigned to when that was
        saved, else matched by disk / hole like a sample."""
        fh = pos.get('fh')
        if fh:
            key = (str(fh[0]), str(fh[1]))
            fx = self._disk_flux.get(key)
            if fx is not None and str(fh[2]) in fx['idx']:
                return (key, fx['idx'][str(fh[2])])
            return None
        return self._flux_match_ids(pos.get('disk'), pos.get('hole'),
                                    pos.get('irr') or None)

    def _flux_match_ids(self, disk_id, hole_id, irradiation):
        """Auto-match a (disk, hole, irradiation) to a loaded disk-flux hole.
        Returns (key, idx) or None.  The same rule as the Plateau & isochron tool
        (arar_math.match_flux_hole); the user's preferred disk, if one is set,
        decides a tie first.  A tie it cannot settle is left unmatched."""
        return match_flux_hole(self._disk_flux, disk_id, hole_id, irradiation,
                               getattr(self, '_preferred_flux', None))

    def _j_point_info(self, s, idx):
        """For j_correlation: where each of sample s's points idx was irradiated.

        A sample made of several irradiation positions (a stack mixing holes or
        disks) uses each point's own position; any other sample, or one with a
        flux assigned to it, is a single position (the sample's own hole)."""
        consts = getattr(s, 'constants', None) or {}
        s_irr = str(s.irradiation or consts.get('irradiation') or '')
        s_info = dict(match=self._flux_match(s),
                      ident=(position_ident(s.disk_id, s.hole_id, None, s.j_cfg)
                             or repr(s.flux_key())),
                      irr=s_irr)
        metas = s.metas if len(s.metas) == len(s.ages) else None
        multi = (metas is not None and s.pt_pos is not None
                 and s._flux_src is None and getattr(s, '_flux_override', None) is None)
        out = []
        for i in idx:
            pos = metas[i].get('pos') if multi else None
            if not pos:
                out.append(s_info)
                continue
            out.append(dict(match=self._flux_match_point(pos),
                            ident=pos.get('disk') or pos.get('hole') or pos.get('key'),
                            irr=str(pos.get('irr') or '')))
        return out

    def _j_corr(self, infos, force=False):
        """J-flux correlation matrix of the points described by *infos*: the
        rule shared with the Plateau & isochron tool (arar_math.j_correlation).  Points
        with identical position info are one group, so only the distinct groups
        are compared."""
        groups, gi = {}, []
        for d in infos:
            k = (d['match'], d['ident'], d['irr'])
            if k not in groups:
                groups[k] = len(groups)
            gi.append(groups[k])
        pts = [dict(match=k[0], ident=k[1], irr=k[2]) for k in groups]
        ru = j_correlation(pts, self._disk_flux, force=force)
        gi = np.asarray(gi, int)
        return ru[np.ix_(gi, gi)]

    def _flux_hole_R(self, key, i):
        """Monitor R a loaded disk-flux hole encodes (arar_math.flux_hole_R)."""
        fx = self._disk_flux.get(key)
        return None if fx is None else flux_hole_R(fx, i)

    def _flux_desc_R(self, key, i):
        """A hole's monitor R for reports (the calibration-independent value the
        hole is applied with; its stored J belongs to the flux's calibration)."""
        R = self._flux_hole_R(key, i)
        return f'{R:.6g}' if R else '—'

    def _active(self):
        return [s for s in self.samples if s.enabled and len(s.ages) > 0
                and s.mask().sum() > 0]

    def _show_error_breakdown(self):
        """Pie chart of where the reported age uncertainty comes from."""
        act = self._active()
        if not act:
            QMessageBox.information(self, 'Error breakdown',
                                    'No active samples to break down.')
            return
        # pooled budget if pooling, else the single active sample
        if self.pool_on and len(act) > 1:
            r = self.pool()
            cfg = act[0].j_cfg
            title = f'Pooled  ({len(act)} samples, n = {r["n"]})'
        else:
            s = act[0]
            r = s.wm()
            cfg = s.j_cfg
            title = s.name
        if r is None:
            return
        budget = age_error_budget(r['sigma_ana'], r['sigma_int'],
                                  r['sigma_ext'], r['wm'], cfg)
        ds = self.display_sigma

        dlg = QDialog(self)
        dlg.setWindowTitle('Where the age uncertainty comes from')
        dlg.setMinimumSize(560, 460)
        lay = QVBoxLayout(dlg)
        fig = Figure(figsize=(5.4, 4.2))
        canvas = FigureCanvas(fig)
        lay.addWidget(canvas, 1)
        ax = fig.add_subplot(111)
        draw_budget_pie(ax, budget, ds, f'±{ds}σ')
        route = external_route(cfg)
        ax.set_title(f'{title}\nWM = {r["wm"]:.4f} Ma   ·   external via {route}',
                     fontsize=9)
        fig.tight_layout()
        canvas.draw()
        dlg.exec_()

    def pool(self):
        """Combine all included ages (covariance engine).

        Builds a per-point J-flux correlation matrix with arar_math.j_correlation,
        the same rule as the Plateau & isochron tool: same hole -> 1; same loaded disk ->
        the exported cross-hole correlation; otherwise same position -> 1, else 0
        (or all 1 if Force corr. J).
        """
        act = self._active()
        if len(act) < 1:
            return None
        ages, a, si, se = [], [], [], []
        pcfg = []                    # per point: its sample's calibration
        infos = []                   # per point: where it was irradiated
        for s in act:
            # one tuple per fusion analysis; a single collapsed tuple per plateau
            pts = s.pool_points()
            pidx = ([0] * len(pts) if s.is_plateau else list(np.where(s.mask())[0]))
            for (ag, sa, sint, sext) in pts:
                ages.append(ag); a.append(sa)
                si.append(sint); se.append(sext)
                pcfg.append(s.j_cfg)
            infos += self._j_point_info(s, pidx)
        # J-flux correlation: the rule shared with the Plateau & isochron tool
        # (arar_math.j_correlation): same loaded disk -> the .jflux cross-hole
        # correlation; one on a loaded hole and one not -> 0; neither on a
        # loaded hole and same position -> 1; else 0 (all 1 if "Force corr. J")
        rho = self._j_corr(infos, force=self.corr_j)
        # samples on the same calibration share every systematic component;
        # samples on different calibrations share none of them
        keys = [calibration_group_key(c) for c in pcfg]
        std = [keys.index(k) for k in keys]
        C, _ = systematic_matrix(ages, pcfg)
        return _combine(ages, a, si, se, standard_group=std,
                        od_mode='threshold', p_corr=rho, sys_components=C)

    # --- drawing ------------------------------------------------------------
    @staticmethod
    def _legend_ages(r, tier, ds, af):
        """The age part of a legend entry: '  WM  ±2σ ana/int/ext' (all
        three, as in the window) or '  WM ± one tier'."""
        if tier in ('ana', 'int', 'ext'):
            return f"  {r['wm']:{af}} ± {r['sigma_' + tier] * ds:{af}}"
        return (f"  {r['wm']:{af}}  ±{ds}σ "
                f"{r['sigma_ana']*ds:{af}}/{r['sigma_int']*ds:{af}}/"
                f"{r['sigma_ext']*ds:{af}}")

    def _ex(self, key):
        """Is this figure element switched on?  True unless the export dialog
        has turned it off for the figure it is currently rendering."""
        return getattr(self, '_export_show', {}).get(key, True)

    def redraw(self):
        if hasattr(self, 'calib_sel'):
            self._sync_calib_bar()
        self.ax.clear()
        self._click_pts = []
        act = self._active_all()        # active = enabled with data (incl. fully-excluded)
        ds = self.display_sigma
        _ps      = getattr(self, '_pub_scale',    1.0)  # font size scale
        _ps_line = getattr(self, '_pub_lw_scale', 1.0)  # curve/vline linewidth scale
        _ps_ms   = getattr(self, '_pub_ms_scale', 1.0)  # marker & error-bar scale

        if not act:
            self.ax.set_xlabel('Age (Ma)')
            self.ax.set_ylabel(
                'Relative probability' if self.y_mode == 'relative' else
                'Probability (%)' if self.y_mode == 'probability' else
                'Normalised probability (%)' if self.y_mode == 'normalised' else 'Rank')
            self.ax.text(0.5, 0.5, 'No data.\nPaste, Load or Import on the right.',
                         transform=self.ax.transAxes, ha='center', va='center',
                         color='#888', style='italic')
            self.canvas.draw_idle()
            self.report.setPlainText('No active samples.')
            return

        # ── global rank when pooling, else per-sample rank ───────────────────
        pooling = self.pool_on and len(act) > 1
        if pooling:
            allages = np.concatenate([s.ages for s in act])
            order = np.argsort(allages)
            grank = np.empty(len(allages))
            grank[order] = np.arange(1, len(allages) + 1)
            ranks, off = [], 0
            for s in act:
                ranks.append(grank[off:off + len(s.ages)])
                off += len(s.ages)
        else:
            ranks = []
            for s in act:
                rk = np.empty(len(s.ages))
                rk[np.argsort(s.ages)] = np.arange(1, len(s.ages) + 1)
                ranks.append(rk)

        # density_mode: 'relative' only: global-max normalisation + baseline rug.
        # prob_mode:    curve peaks at p × 100; points rank-normalised to curve height.
        # rank_mode:    curve peaks at n_in; points at integer rank (1…n) for max spread.
        # norm_mode:    as prob_mode, but every curve (the pooled one too) peaks
        #               at 100 % instead of at p × 100
        density_mode = (self.y_mode == 'relative')
        norm_mode    = (self.y_mode == 'normalised')
        prob_mode    = (self.y_mode == 'probability') or norm_mode
        rank_mode    = (self.y_mode == 'rank')
        # export, panels by sample: draw only this panel's samples (the pool
        # still uses all of them)
        _only = getattr(self, '_only_sample', None)

        Gmax = 1.0
        if density_mode:
            pk = []
            for s in act:
                m = s.mask(); n_in = int(m.sum())
                if n_in >= 1:
                    inc = s.ages[m]; inc_s = s.sig1()[m]
                    axd = _curve_axis(inc, inc_s, ds)
                    p = summed_gaussian_pdf(inc, inc_s, axd).max() / n_in
                    if p > 0:
                        pk.append(p)
            if pooling:
                ia, isg = [], []
                for s in act:
                    m = s.mask(); ia.append(s.ages[m]); isg.append(s.sig1()[m])
                ia = np.concatenate(ia) if ia else np.array([])
                isg = np.concatenate(isg) if isg else np.array([])
                if len(ia):
                    axd = _curve_axis(ia, isg, ds)
                    env = np.zeros_like(axd)
                    for s in act:
                        m = s.mask(); n_in = int(m.sum())
                        if n_in >= 1:
                            env += summed_gaussian_pdf(s.ages[m], s.sig1()[m], axd) / n_in
                    if env.max() > 0:
                        pk.append(env.max())
            Gmax = max(pk) if pk else 1.0

        # Pre-pass: autoscale limits for prob and rank modes
        _y_auto_max = 0.0          # used by prob_mode: max(p_val × 100) across samples
        _rank_y_max = 1            # used by rank_mode: max(n_in) across samples
        if prob_mode:
            for s in act:
                ri = s.wm()
                if ri:
                    _y_auto_max = max(_y_auto_max,
                                      100.0 if norm_mode else ri['prob'] * 100)
            if pooling:
                _pre_pr = self.pool()
                if _pre_pr:
                    _y_auto_max = max(_y_auto_max,
                                      100.0 if norm_mode else _pre_pr['prob'] * 100)
            _y_auto_max = max(_y_auto_max, 1.0)
        elif rank_mode:
            _rank_y_max = max((len(s.ages) for s in act), default=1)

        legend = []
        _af = f'.{int(self.age_dec)}f'   # decimal places for ages/uncertainties
        # export dialog's 'Ages in legend' (the window always shows all three)
        _lg_tier = getattr(self, '_export_show', {}).get('legend_ages', 'all')
        # Pad the names to one measured width so the age block starts in the
        # same column in the normal legend font (export.pad_names): every
        # legend, with or without ages, then uses the same font.
        def _tn(name, limit=77):
            return name if len(name) <= limit else name[:limit] + '…'
        _names = [_tn(s.name) for s in act]
        if pooling:
            _names.append('Pooled')
        _padded = dict(zip(_names, export.pad_names(_names, 8 * _ps, gap=0)))
        for k, (s, rk) in enumerate(zip(act, ranks)):
            if _only is not None and not any(s is o for o in _only):
                continue
            m = s.mask()
            n_in = int(m.sum())
            sig = s.sig1()
            r_info = s.wm()
            p_val = 1.0 if norm_mode else (r_info['prob'] if r_info else 0.0)
            _fmt = 's' if s.is_plateau else 'o'      # plateau steps drawn as squares
            # The density curve is a picture of the raw analyses, so it is drawn
            # from the points and is identical whether or not the sample is a
            # plateau. Only the pool (pool_points) collapses a plateau to one point.
            if n_in >= 1:
                inc = s.ages[m]
                inc_s = sig[m]
                axis = _curve_axis(inc, inc_s, ds)
                pdf = summed_gaussian_pdf(inc, inc_s, axis)
                if pdf.max() > 0:
                    if density_mode:
                        scaled = pdf / n_in / Gmax              # relative: global-max = 1
                    elif prob_mode:
                        scaled = pdf / pdf.max() * p_val * 100  # probability: peak = p × 100
                    else:
                        scaled = pdf / pdf.max() * n_in         # rank: peak = n_in
                    if self._ex('fill'):
                        self.ax.fill_between(axis, scaled, color=s.color,
                                             alpha=0.18, zorder=1)
                    if self._ex('curve'):
                        self.ax.plot(axis, scaled, color=s.color,
                                     lw=1.8 * _ps_line, zorder=6)   # over the points
                # dashed "would-be" outline including the excluded points
                if n_in < len(s.ages):
                    full = summed_gaussian_pdf(s.ages, sig, axis)
                    if full.max() > 0:
                        if density_mode:
                            fsc = full / len(s.ages) / Gmax
                        elif prob_mode:
                            # mirror rel-prob: dashed always shorter by n_in/n_total ratio
                            fsc = full / full.max() * p_val * 100 * n_in / len(s.ages)
                        else:                                    # rank: peak = n_all
                            fsc = full / full.max() * len(s.ages)
                        if self._ex('dashed'):
                            self.ax.plot(axis, fsc, color=s.color, lw=1.2 * _ps_line,
                                         ls='--', alpha=0.45, zorder=1.5)
                # WM marker line
                r = s.wm()
                if r:
                    band = r['sigma_int'] * ds
                    if self._ex('wm'):
                        self.ax.axvline(r['wm'], color=s.color, lw=1.4 * _ps_line,
                                        ls='--', zorder=4)
                        self.ax.axvspan(r['wm'] - band, r['wm'] + band, color=s.color,
                                        alpha=0.08, zorder=1)
            # individual analyses
            _eb = self.ax.errorbar if self._ex('points') else (lambda *a, **k: None)
            if density_mode:
                # relative mode: rug near the bottom, one stripe per sample
                y_top = 1.0
                base_y = (0.03 + 0.035 * k) * y_top
                for i in range(len(s.ages)):
                    self._click_pts.append((s, i, s.ages[i], base_y))
                    if m[i]:
                        _eb(s.ages[i], base_y, xerr=sig[i] * ds, fmt=_fmt,
                                         color=s.color, ms=4 * _ps_ms, lw=0.9 * _ps_ms,
                                         capsize=2 * _ps_ms, zorder=5)
                    else:
                        _eb(s.ages[i], base_y, xerr=sig[i] * ds, fmt=_fmt,
                                         mfc='white', mec=s.color, mew=1.3 * _ps_ms,
                                         ecolor=s.color, ms=5 * _ps_ms, lw=0.8 * _ps_ms,
                                         capsize=2 * _ps_ms, alpha=0.8, zorder=5)
            elif prob_mode:
                # probability mode: always per-sample rank so pooling only adds the
                # pooled curve without repositioning any individual analysis points
                _rk_ps = np.empty(len(s.ages))
                _rk_ps[np.argsort(s.ages)] = np.arange(1, len(s.ages) + 1)
                norm_n = max(len(s.ages), 1)
                for i in range(len(s.ages)):
                    y_i = _rk_ps[i] / norm_n * p_val * 100
                    self._click_pts.append((s, i, s.ages[i], y_i))
                    if m[i]:
                        _eb(s.ages[i], y_i, xerr=sig[i] * ds, fmt=_fmt,
                                         color=s.color, ms=5 * _ps_ms, lw=1.1 * _ps_ms,
                                         capsize=3 * _ps_ms, zorder=5)
                    else:
                        _eb(s.ages[i], y_i, xerr=sig[i] * ds, fmt=_fmt,
                                         mfc='white', mec=s.color, mew=1.5 * _ps_ms,
                                         ecolor=s.color, ms=6 * _ps_ms, lw=1.0 * _ps_ms,
                                         capsize=3 * _ps_ms, alpha=0.8, zorder=5)
            else:
                # rank mode: integer rank steps so 1000 tight analyses spread over 1…n
                for i in range(len(s.ages)):
                    y_i = rk[i]
                    self._click_pts.append((s, i, s.ages[i], y_i))
                    if m[i]:
                        _eb(s.ages[i], y_i, xerr=sig[i] * ds, fmt=_fmt,
                                         color=s.color, ms=5 * _ps_ms, lw=1.1 * _ps_ms,
                                         capsize=3 * _ps_ms, zorder=5)
                    else:
                        _eb(s.ages[i], y_i, xerr=sig[i] * ds, fmt=_fmt,
                                         mfc='white', mec=s.color, mew=1.5 * _ps_ms,
                                         ecolor=s.color, ms=6 * _ps_ms, lw=1.0 * _ps_ms,
                                         capsize=3 * _ps_ms, alpha=0.8, zorder=5)
            lab = _padded[_tn(s.name)] if _lg_tier != 'off' else _tn(s.name)
            r = s.wm()
            if r and _lg_tier != 'off':
                lab += self._legend_ages(r, _lg_tier, ds, _af)
                # second line: MSWD and p (one sample), then n (pooled: only on
                # the pooled entry), then ω
                _l2 = []
                if len(act) == 1:
                    _l2.append(f"MSWD={r['mswd']:.2f}"
                               f"  p={r.get('prob', float('nan')):.2f}")
                if not pooling and self._ex('legend_n'):
                    _l2.append(f"n={int(s.mask().sum())}/{len(s.ages)}")
                if len(act) == 1 and 'overdispersion' in r:
                    _l2.append(f"ω={r['overdispersion']*ds:{_af}}")
                if _l2:
                    lab += "\n" + "  ".join(_l2)
            legend.append(Line2D([0], [0], color=s.color, lw=2, label=lab))

        # ── pooled curve ─────────────────────────────────────────────────────
        # In density mode the pooled curve is the sum (envelope) of the per-sample
        # unit-area curves, so it visibly combines them and sits at/above each
        # component rather than as a count-weighted average that dips below them.
        rep = []
        if pooling:
            pr = self.pool()
            inc_ages, inc_sig = [], []
            for s in act:
                m = s.mask()
                inc_ages.append(s.ages[m])
                inc_sig.append(s.sig1()[m])
            inc_ages = np.concatenate(inc_ages)
            inc_sig = np.concatenate(inc_sig)
            if len(inc_ages):
                axis = _curve_axis(inc_ages, inc_sig, ds)
                if density_mode:
                    # relative: sum of per-sample unit-area envelopes, Gmax-normalised
                    env = np.zeros_like(axis)
                    for s in act:
                        m = s.mask(); n_in = int(m.sum())
                        if n_in >= 1:
                            env += summed_gaussian_pdf(
                                s.ages[m], s.sig1()[m], axis) / n_in
                    scaled = env / Gmax if env.max() > 0 else None
                elif prob_mode:
                    pdf = summed_gaussian_pdf(inc_ages, inc_sig, axis)
                    pool_p = 1.0 if norm_mode else (pr['prob'] if pr else 0.0)
                    scaled = pdf / pdf.max() * pool_p * 100 if pdf.max() > 0 else None
                else:
                    # rank: curve peaks at total included n
                    total_n = sum(int(s.mask().sum()) for s in act)
                    pdf = summed_gaussian_pdf(inc_ages, inc_sig, axis)
                    scaled = pdf / pdf.max() * total_n if pdf.max() > 0 else None
                if scaled is not None and np.max(scaled) > 0:
                    if self._ex('fill'):
                        self.ax.fill_between(axis, scaled, color=_POOL_COLOR,
                                             alpha=0.12, zorder=0.8)
                    if self._ex('curve'):
                        self.ax.plot(axis, scaled, color=_POOL_COLOR,
                                     lw=2.0 * _ps_line, zorder=6.5)
            if pr:
                band = pr['sigma_int'] * ds
                if self._ex('wm'):
                    self.ax.axvline(pr['wm'], color=_POOL_COLOR, lw=1.6 * _ps_line,
                                    ls='--', zorder=4)
                    self.ax.axvspan(pr['wm'] - band, pr['wm'] + band,
                                    color=_POOL_COLOR, alpha=0.07, zorder=0.7)
                # the pooled entry follows the samples unless the export
                # asks otherwise ('Pooled age in legend')
                _pl_tier = getattr(self, '_export_show', {}).get('pool_legend', 'same')
                _pl_tier = (_lg_tier if _pl_tier == 'same' else
                            'off' if _pl_tier == 'name' else _pl_tier)
                _plab = 'Pooled'
                if _pl_tier != 'off':
                    if _lg_tier != 'off':          # line up with the samples
                        _plab = _padded['Pooled']
                        _pa = self._legend_ages(pr, _pl_tier, ds, _af)
                        if _pl_tier != _lg_tier and _pl_tier in ('ana', 'int', 'ext'):
                            # not the tier the legend heading names
                            _pa += (f" ({ds}σ " + dict(ana='analytical', int='internal',
                                                      ext='external')[_pl_tier] + ")")
                    elif _pl_tier == 'all':
                        _pa = self._legend_ages(pr, 'all', ds, _af)
                    else:                          # names-only samples: say which σ
                        _pa = (f"  {pr['wm']:{_af}} ± {pr['sigma_' + _pl_tier] * ds:{_af}} Ma"
                               f" ({ds}σ {dict(ana='analytical', int='internal', ext='external')[_pl_tier]})")
                    _plab += (_pa + "\n"
                              f"MSWD={pr['mswd']:.2f}"
                              f"  p={pr.get('prob', float('nan')):.2f}"
                              + (f"  n={pr['n']}/"
                                 + str(sum(1 if sm.is_plateau else len(sm.ages)
                                           for sm in act))
                                 if self._ex('legend_n') else "")
                              + (f"  ω={pr['overdispersion']*ds:{_af}}"
                                 if 'overdispersion' in pr else ""))
                legend.append(Line2D([0], [0], color=_POOL_COLOR, lw=2.2 * _ps_line,
                                     ls='-', label=_plab))
                rep = self._pool_text(pr, act)

        self.ax.set_xlabel('Age (Ma)', fontsize=11 * _ps)
        if prob_mode:
            self.ax.set_ylabel('Normalised probability (%)' if norm_mode else
                               'Probability (%)', fontsize=11 * _ps)
            if self.ylim_mode == 'manual' and self.y_max:
                self.ax.set_ylim(0, self.y_max)
            else:
                self.ax.set_ylim(0, _y_auto_max * 1.10)
        elif density_mode:
            self.ax.set_ylabel('Relative probability', fontsize=11 * _ps)
            if self.ylim_mode == 'manual' and self.y_max:
                self.ax.set_ylim(0, self.y_max)
            else:
                self.ax.set_ylim(0, 1.15)
        else:
            self.ax.set_ylabel('Rank', fontsize=11 * _ps)
            if self.ylim_mode == 'manual' and self.y_max:
                self.ax.set_ylim(0, self.y_max)
            else:
                self.ax.set_ylim(0, _rank_y_max * 1.10)
        if self.xlim_mode == 'manual' and self.x_lo is not None:
            self.ax.set_xlim(self.x_lo, self.x_hi)
        self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        self.ax.tick_params(which='both', direction='in', top=True, right=True,
                            labelsize=10 * _ps)
        if legend:
            # the normal font: the ages line up through the measured padding
            # above (export.pad_names), not a typewriter font
            _prop = {'size': 8 * _ps}
            # one tier chosen in the export: say which in the legend heading
            _tn_ = {'ana': 'analytical', 'int': 'internal',
                    'ext': 'external'}.get(_lg_tier)
            _lt = dict(title=f'Weighted mean ± {ds}σ ({_tn_})',
                       title_fontsize=8 * _ps) if _tn_ else {}
            if getattr(self, '_pub_scale', 1.0) > 1.0:
                # publication export: legend above the axes so it never overlaps the data
                self.ax.legend(handles=legend, loc='lower left',
                               bbox_to_anchor=(0, 1.01), borderaxespad=0,
                               framealpha=0.95, prop=_prop, **_lt).set_zorder(20)
            else:
                self.ax.legend(handles=legend, loc='upper right',
                               framealpha=0.92, prop=_prop, **_lt).set_zorder(20)
        self.fig.tight_layout()
        if not getattr(self, '_offscreen', False):
            self.canvas.draw_idle()

        if not rep:
            # not pooling -> show a detailed per-sample report (ana/int/ext)
            blocks = [self._sample_report_text(s) for s in act]
            rep = []
            for b in blocks:
                rep += b + ['']
            if not rep:
                rep = ['Enable Pool to combine all active samples.',
                       'Tip: click a point on the plot to include/exclude it.']
        if not getattr(self, '_offscreen', False):
            self.report.setPlainText('\n'.join(rep))
        for p in self.panels:
            p.refresh_info()
        if self.mix_on and not getattr(self, '_offscreen', False):
            self._redraw_mixture()

    # --- Mixture view -------------------------------------------------------
    _MIX_MAM = 'Minimum age model'

    def _build_mixture_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(QLabel('Sample:'))
        self.mix_sample = QComboBox()
        self.mix_sample.setMinimumWidth(140)
        self.mix_sample.activated.connect(lambda _: self._redraw_mixture())
        row.addWidget(self.mix_sample)
        row.addSpacing(8)
        row.addWidget(QLabel('Components:'))
        self.mix_k = QComboBox()
        self.mix_k.addItems(['Auto (BIC)'] + [str(k) for k in range(1, 11)] + [self._MIX_MAM])
        self.mix_k.setToolTip(
            'Auto (BIC): the number of components with the lowest Bayes Information\n'
            'Criterion, taking the fewest components within 2 of the lowest.\n'
            '1-10: that many components.\n'
            'Minimum age model: the youngest age plus a continuous spread of older\n'
            'analyses (Galbraith et al. 1999). It assumes no analysis is too young:\n'
            'a single young outlier (Ar loss) becomes the minimum, and old\n'
            'xenocrysts widen the spread. Use it on data without them.')
        self.mix_k.activated.connect(lambda _: self._redraw_mixture())
        row.addWidget(self.mix_k)
        row.addSpacing(8)
        row.addWidget(QLabel('Data:'))
        self.mix_data = QComboBox()
        self.mix_data.addItems(['All analyses', 'Included only'])
        self.mix_data.setToolTip(
            'All analyses: the model decides which analyses belong together, so\n'
            'the outlier rejection and the click exclusions are not used.\n'
            'Included only: the analyses included on the plot.')
        self.mix_data.activated.connect(lambda _: self._redraw_mixture())
        row.addWidget(self.mix_data)
        row.addSpacing(8)
        row.addWidget(QLabel('Scale:'))
        self.mix_space = QComboBox()
        self.mix_space.addItems(['linear', 'log'])
        self.mix_space.setToolTip(
            'linear: the model works on the ages (keeps zero and negative ages).\n'
            'log: on ln(age), as IsoplotR does by default; ages <= 0 are left out.')
        self.mix_space.activated.connect(lambda _: self._redraw_mixture())
        row.addWidget(self.mix_space)
        row.addStretch()
        v.addLayout(row)
        self.mix_fig = Figure(figsize=(7, 6), dpi=100)
        self.mix_canvas = FigureCanvas(self.mix_fig)
        self.mix_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(NavigationToolbar(self.mix_canvas, self), 0)
        v.addWidget(self.mix_canvas, 1)
        return page

    def _show_mixture(self, on):
        """Switch the plot area and the button without redrawing."""
        self.mix_on = bool(on)
        self.mix_btn.blockSignals(True)
        self.mix_btn.setChecked(self.mix_on)
        self.mix_btn.blockSignals(False)
        self.mix_btn.setText(f'Mixture: {"on" if self.mix_on else "off"}')
        self._plot_stack.setCurrentIndex(1 if self.mix_on else 0)
        self._main_toolbar.setVisible(not self.mix_on)
        # with the view on, both exports are of the mixture model of the sample shown
        self._btn_export_fig.setText('Export mixture figure…' if self.mix_on else 'Export figure…')
        self._btn_export_data.setText('Export mixture XLSX / CSV' if self.mix_on
                                      else 'Export XLSX / CSV')

    def _on_mixture(self, on):
        self._show_mixture(on)
        self.redraw()

    def _mix_sample_list(self):
        """Fill the sample box from the active samples; the sample shown."""
        act = self._active_all()
        names = [s.name for s in act]
        cur = self._mix_want or self.mix_sample.currentText()
        self._mix_want = None
        self.mix_sample.blockSignals(True)
        self.mix_sample.clear()
        self.mix_sample.addItems(names)
        if cur in names:
            self.mix_sample.setCurrentIndex(names.index(cur))
        self.mix_sample.blockSignals(False)
        i = self.mix_sample.currentIndex()
        return act[i] if 0 <= i < len(act) else None

    def _mix_fit(self, t, s, kchoice, space):
        """(fit, all BIC fits or None, BIC choice of k or None), cached."""
        data = (tuple(np.round(t, 12)), tuple(np.round(s, 12)), space)
        if len(self._mix_cache) > 60:
            self._mix_cache.clear()
        if kchoice == self._MIX_MAM:
            key = data + ('mam',)
            if key not in self._mix_cache:
                self._mix_cache[key] = am.fit_mam(t, s, 3, space)
            return self._mix_cache[key], None, None
        key = data + ('bic',)
        if key not in self._mix_cache:
            self._mix_cache[key] = am.select_k(t, s, 10, space)
        kb, fits = self._mix_cache[key]
        k = kb if kchoice.startswith('Auto') else int(kchoice)
        f = next((x for x in fits if x['k'] == k), None)
        if f is None:
            fkey = data + ('k', k)
            if fkey not in self._mix_cache:
                self._mix_cache[fkey] = am.fit_fmm(t, s, k, space)
            f = self._mix_cache[fkey]
        return f, fits, kb

    def _mix_message(self, text):
        self.mix_fig.clear()
        ax = self.mix_fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, text, ha='center', va='center', color='#888',
                style='italic', transform=ax.transAxes)
        self.mix_canvas.draw_idle()
        self._mix_state = None

    def _redraw_mixture(self):
        s = self._mix_sample_list()
        if s is None:
            self._mix_message('No data.\nPaste, Load or Import on the right.')
            return
        all_data = self.mix_data.currentIndex() == 0
        use = np.ones(len(s.ages), bool) if all_data else s.mask()
        t, sig = s.ages[use], s.sig1()[use]
        kchoice, space = self.mix_k.currentText(), self.mix_space.currentText()
        if len(t) < 3:
            self._mix_message('At least 3 analyses are needed.')
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            fit, fits, kb = self._mix_fit(t, sig, kchoice, space)
        finally:
            QApplication.restoreOverrideCursor()
        if fit is None:
            self._mix_message('Too few analyses for this many components.')
            return
        tiers = am.component_tiers(fit, s.j_cfg)
        if self.xlim_mode == 'manual' and self.x_lo is not None:
            xl = (self.x_lo, self.x_hi)
        else:
            xl = am.mixture_xlim(t, sig, fit, self.display_sigma)
        self._mix_state = dict(sample=s, idx=np.where(use)[0], t=t, sig=sig, fit=fit,
                               fits=fits, kb=kb, tiers=tiers, xlim=xl, space=space,
                               all_data=all_data, kchoice=kchoice)
        self.mix_fig.clear()
        gs = self.mix_fig.add_gridspec(2, 2, width_ratios=[1.45, 1],
                                       height_ratios=[1, 0.3], hspace=0.42, wspace=0.2)
        ax = self.mix_fig.add_subplot(gs[:, 0])
        axr = self.mix_fig.add_subplot(gs[0, 1] if fits else gs[:, 1])
        axb = self.mix_fig.add_subplot(gs[1, 1]) if fits else None
        self.mix_fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.1)
        am.draw_components(ax, t, sig, fit, xl, tiers, ds=self.display_sigma,
                             dec=int(self.age_dec), title=self._mix_title())
        am.draw_radial(axr, t, sig, fit, xl, space)
        if axb is not None:
            am.draw_bic(axb, fits, fit['k'], kb)
        self.mix_canvas.draw_idle()
        self.report.setPlainText('\n'.join(self._mixture_report()))

    def _mix_title(self, mode='model', with_k=True, s=None, fit=None):
        st = self._mix_state
        s, fit = s or st['sample'], fit or st['fit']
        what = (f"{fit['k']} component{'s' if fit['k'] > 1 else ''}"
                if fit['model'] == 'fmm' else 'minimum age model')
        if mode == 'name':
            return f'{s.name}: {what}' if with_k else s.name
        if mode == 'name_n':
            n = f"({fit['n']} analyses)"
            return f'{s.name}: {what}  {n}' if with_k else f'{s.name}  {n}'
        if mode == 'off':
            return what[0].upper() + what[1:] if with_k else None
        if not with_k:
            return f"{s.name}  ({fit['n']} analyses, {st['space']})"
        return f"{s.name}: {what}  ({fit['n']} analyses, {st['space']})"

    def _mixture_report(self):
        st = self._mix_state
        s, fit, tiers = st['sample'], st['fit'], st['tiers']
        ds, d = self.display_sigma, int(self.age_dec)
        L = ['-' * 64,
             f"MIXTURE MODEL: {s.name}   ({fit['n']} analyses, "
             f"{'all' if st['all_data'] else 'included only'}, {st['space']} scale)",
             '-' * 64]
        if fit['model'] == 'fmm':
            how = 'chosen by BIC' if st['kchoice'].startswith('Auto') else 'set by hand'
            L.append(f"  Discrete mixture, k = {fit['k']} ({how}; BIC choice k = {st['kb']})")
            bmin = min(f['bic'] for f in st['fits'])
            L.append('  ΔBIC  ' + '   '.join(f"k={f['k']}: {f['bic'] - bmin:.1f}" for f in st['fits']))
            L.append(f'  Components, ±{ds}σ analytical / internal / external:')
            for i, tr in enumerate(tiers):
                L.append(f"  C{i + 1}  {tr['age']:10.{d}f}  ±{tr['ana']*ds:.{d}f}/{tr['int']*ds:.{d}f}"
                         f"/{tr['ext']*ds:.{d}f}   {100*fit['prop'][i]:3.0f} ± "
                         f"{100*ds*fit['se_prop'][i]:.0f} %   n = {fit['n_assigned'][i]:<3d}"
                         f"  MSWD {tr['mswd']:.2f}"
                         + (f"  (×√MSWD = {tr['scale']:.2f})" if tr['scale'] > 1 else ''))
            L.append('  n = analyses most likely in the component. MSWD of the analyses weighted')
            L.append('  by how likely they belong to it; above the Wendt & Carl limit the')
            L.append('  analytical uncertainty is multiplied by √MSWD.')
        else:
            tr = tiers[0]
            L.append('  Minimum age model (3 parameters, Galbraith et al. 1999)')
            L.append(f"  Minimum age = {tr['age']:.{d}f}  ±{ds}σ {tr['ana']*ds:.{d}f}/"
                     f"{tr['int']*ds:.{d}f}/{tr['ext']*ds:.{d}f}"
                     + (f"   (×√MSWD = {tr['scale']:.2f})" if tr['scale'] > 1 else ''))
            L.append(f"  Proportion in the minimum = {100*fit['prop']:.0f} %   spread of the "
                     f"older analyses σ = {fit['sigma']:.3g}"
                     f"{' (log units)' if st['space'] == 'log' else ' Ma'}"
                     f"   MSWD of the minimum {tr['mswd']:.2f}")
            if fit.get('note'):
                L.append('  ' + fit['note'])
        nl = int((~fit['used']).sum())
        if nl:
            L.append(f'  {nl} analyses with age <= 0 left out (log scale).')
        L.append('  The model uses the analytical uncertainties; J and the calibration are')
        L.append('  added to each component age afterwards, as for a weighted mean.')
        return L

    def _mix_settings(self):
        return dict(on=self.mix_on, sample=self.mix_sample.currentText(),
                    components=self.mix_k.currentText(),
                    data=self.mix_data.currentText(), scale=self.mix_space.currentText())

    def _apply_mix_settings(self, d):
        """Mixture settings of a loaded session (none: the view is off)."""
        d = d or {}
        for box, key in ((self.mix_k, 'components'), (self.mix_data, 'data'),
                         (self.mix_space, 'scale')):
            if d.get(key) is not None and box.findText(d[key]) >= 0:
                box.setCurrentIndex(box.findText(d[key]))
        self._mix_want = d.get('sample')
        self._show_mixture(d.get('on', False))

    # --- Mixture view: figure export -----------------------------------------
    def _export_mixture_figure(self):
        if self._mix_state is None:
            QMessageBox.information(self, 'Export figure', 'The Mixture view shows no model yet.')
            return
        dlg = export.FigureExportDialog(
            self, 'ageprob_mixture', self._mix_export_panels(), self._mix_export_options(),
            self._mix_export_render,
            default_name=(self._mix_state['sample'].name or 'mixture').replace(' ', '_') + '_mixture',
            snapshot=self._mix_export_snapshot, title='Export figure: Mixture view',
            defaults=dict(width_cm=24.0, height_cm=14.0))
        dlg.exec_()
        self.redraw()

    def _mix_export_panels(self):
        return [export.Panel('components', 'Components (ranked ages and curve)', fixed=True,
                             weight=1.0),
                export.Panel('radial', 'Radial plot', on=True, weight=0.8),
                export.Panel('bic', 'ΔBIC per number of components', on=True, weight=0.3)]

    def _mix_export_options(self):
        return [
            export.Option('layout', 'Layout', 'choice', wide=True,
                          values=[('window', 'As in the window: diagram left, the others right'),
                                  ('stacked', 'Stacked, top to bottom')],
                          tip='As in the window: the components diagram fills the left side and '
                              'the other panels are stacked on the right, in the order and '
                              'heights set above.'),
            export.Option('curve', 'Probability curve', 'bool', True),
            export.Option('fill', 'Shading by component', 'bool', True),
            export.Option('points', 'Individual analyses', 'bool', True),
            export.Option('bands', 'Component ages (line and band)', 'bool', True),
            export.Option('legend_ages', 'Ages in legend', 'choice',
                          values=[('all', 'Age ± analytical / internal / external'),
                                  ('int', 'Age ± internal'),
                                  ('ana', 'Age ± analytical'),
                                  ('ext', 'Age ± external'),
                                  ('off', 'Component names only')],
                          tip='What follows each component in the legend, at the 1σ/2σ '
                              'level and decimals set in the window.'),
            export.Option('legend_n', 'Proportion and n in legend', 'bool', True),
            export.Option('legend_max', 'Components in legend', 'int', default=0,
                          lo=0, hi=10, placeholder='All',
                          tip='Lists only the first components in the plot (youngest '
                              'first), each with its colour; the others are summed up in '
                              'one line. All lists every component.'),
            export.Option('legend_title', 'Legend heading', 'choice', group='Names',
                          values=[('model', 'Sample, model and number of analyses'),
                                  ('name_n', 'Sample and number of analyses'),
                                  ('name', 'Sample name'),
                                  ('off', 'None')]),
            export.Option('legend_k', 'Number of components in legend heading', 'bool', True,
                          group='Names',
                          tip='Adds "3 components" (or "minimum age model") to the legend '
                              'heading, whichever heading is chosen above.'),
        ] + self._mix_stack_options() + [
            export.Option('x_min', 'Age from', 'text', group='Axis ranges', placeholder='auto'),
            export.Option('x_max', 'to', 'text', group='Axis ranges', placeholder='auto'),
        ]

    def _mix_stack_options(self):
        """Every sample in its own panel, one above the other (2 or more samples)."""
        act = self._active_all()
        if len(act) < 2:
            return []
        kv = [('window', 'As in the window')] + [
            (self.mix_k.itemText(i), self.mix_k.itemText(i)) for i in range(self.mix_k.count())]
        opts = [export.Option(
            'stack', 'Every sample, one above the other', 'bool', False, group='Samples',
            tip='A components diagram for each sample, each with its own mixture model '
                'and legend, on one age axis. The radial plot and ΔBIC are left out; '
                'Data and Scale are as in the window.')]
        for k, sm in enumerate(act):
            opts.append(export.Option(
                f'mixk_{k}', sm.name, 'choice', group='Samples', values=kv,
                depends=('stack', True),
                tip='Components of the mixture model of this sample.',
                ident=sm.experiment_name() or sm.name))
        return opts

    def _mix_export_stacked(self, fig, spec):
        """Each sample's components diagram, stacked on one age axis."""
        st, opts = self._mix_state, spec.get('options', {})
        rows = []
        for k, sm in enumerate(self._active_all()):
            use = np.ones(len(sm.ages), bool) if st['all_data'] else sm.mask()
            t, sig = sm.ages[use], sm.sig1()[use]
            kc = opts.get(f'mixk_{k}', 'window')
            if len(t) < 3:
                continue
            fit = self._mix_fit(t, sig, st['kchoice'] if kc in (None, 'window') else kc,
                                st['space'])[0]
            if fit is not None:
                rows.append((sm, t, sig, fit, am.component_tiers(fit, sm.j_cfg)))
        if not rows:
            fig.text(0.5, 0.5, 'No sample has enough analyses', ha='center')
            return
        x0, x1 = self._opt_float(spec, 'x_min'), self._opt_float(spec, 'x_max')
        lims = [am.mixture_xlim(t, sig, fit, self.display_sigma) for _, t, sig, fit, _ in rows]
        xl = (min(l[0] for l in lims) if x0 is None else x0,
              max(l[1] for l in lims) if x1 is None else x1)
        axes = [fig.add_subplot(len(rows), 1, 1)]
        axes += [fig.add_subplot(len(rows), 1, k + 1, sharex=axes[0])
                 for k in range(1, len(rows))]
        self._fixed_frame(fig, spec)
        mx = opts.get('legend_max')
        for ax, (sm, t, sig, fit, tiers) in zip(axes, rows):
            title = self._mix_title(opts.get('legend_title', 'model'),
                                    opts.get('legend_k', True), s=sm, fit=fit)
            am.draw_components(
                ax, t, sig, fit, xl, tiers, ds=self.display_sigma, dec=int(self.age_dec),
                title=title, show=opts, tiers_mode=opts.get('legend_ages', 'all'),
                with_n=opts.get('legend_n', True),
                max_rows=(int(mx) if str(mx).isdigit() else 0) or None).remove()
            # inside each panel: above it is the panel before
            ax.legend(loc='upper left', framealpha=0.92, prop={'size': 8}, title=title,
                      title_fontproperties={'size': 9, 'weight': 'bold'},
                      alignment='left').set_zorder(20)
            export.place_legend(ax, spec)
        for a in axes[:-1]:                 # one age axis, at the bottom
            a.set_xlabel('')
            a.tick_params(labelbottom=False)
        for a in axes:
            a.set_ylabel('')
        fig.supylabel('Rank', fontsize=axes[0].xaxis.label.get_fontsize())
        export.scale_text(fig, spec.get('font_scale', 1.0),
                          spec.get('line_scale', 1.0), spec.get('marker_scale', 1.0))
        fig.subplots_adjust(hspace=0.08)

    def _mix_export_render(self, fig, spec):
        st = self._mix_state
        opts = spec.get('options', {})
        fig.clear()
        if opts.get('stack') and len(self._active_all()) > 1:
            self._mix_export_stacked(fig, spec)
            return
        pans = [(p, w) for p, w in export.active_panels(spec)
                if not (p == 'bic' and not st['fits'])] or [('components', 1.0)]
        x0, x1 = self._opt_float(spec, 'x_min'), self._opt_float(spec, 'x_max')
        xl = (st['xlim'][0] if x0 is None else x0, st['xlim'][1] if x1 is None else x1)
        others = [(p, w) for p, w in pans if p != 'components']
        if opts.get('layout', 'window') == 'window' and others:
            # as in the window: the diagram on the left, the others stacked
            # on the right in the panel order and heights of the dialog
            outer = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
            right = outer[0, 1].subgridspec(len(others), 1, hspace=0.45,
                                            height_ratios=[w for _, w in others])
            axes = ([('components', fig.add_subplot(outer[0, 0]))]
                    + [(pid, fig.add_subplot(right[i])) for i, (pid, _) in enumerate(others)])
        else:
            gs = fig.add_gridspec(len(pans), 1, height_ratios=[w for _, w in pans])
            axes = [(pid, fig.add_subplot(gs[i])) for i, (pid, _) in enumerate(pans)]
        self._fixed_frame(fig, spec)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        for pid, ax in axes:
            if pid == 'components':
                leg = am.draw_components(
                    ax, st['t'], st['sig'], st['fit'], xl, st['tiers'],
                    ds=self.display_sigma, dec=int(self.age_dec),
                    title=self._mix_title(opts.get('legend_title', 'model'),
                                    opts.get('legend_k', True)),
                    show=opts, tiers_mode=opts.get('legend_ages', 'all'),
                    with_n=opts.get('legend_n', True),
                    max_rows=(int(opts['legend_max'])
                              if str(opts.get('legend_max', '')).isdigit() else 0) or None)
                if pid != axes[0][0]:
                    # not the top panel: no room above it, so inside
                    leg.remove()
                    ax.legend(loc='upper left', framealpha=0.92, prop={'size': 8},
                              title=self._mix_title(opts.get('legend_title', 'model'),
                                    opts.get('legend_k', True)),
                              title_fontproperties={'size': 9, 'weight': 'bold'},
                              alignment='left').set_zorder(20)
                export.place_legend(ax, spec)
            elif pid == 'radial':
                am.draw_radial(ax, st['t'], st['sig'], st['fit'], xl, st['space'])
            elif pid == 'bic':
                am.draw_bic(ax, st['fits'], st['fit']['k'], st['kb'])
        export.scale_text(fig, spec.get('font_scale', 1.0),
                          spec.get('line_scale', 1.0), spec.get('marker_scale', 1.0))

    def _mix_export_snapshot(self, spec):
        lo, hi = self._mix_state['xlim']
        x0, x1 = self._opt_float(spec, 'x_min'), self._opt_float(spec, 'x_max')
        return {'x_min': f'{lo if x0 is None else x0:.4g}',
                'x_max': f'{hi if x1 is None else x1:.4g}'}

    # --- Mixture view: data export --------------------------------------------
    def _write_mixture_rows(self, w, ds):
        st = self._mix_state
        s, fit, tiers = st['sample'], st['fit'], st['tiers']
        sl = f'{ds}σ'
        w.writerow(['MIXTURE MODEL'])
        w.writerow(['Sample', s.name])
        if fit['model'] == 'fmm':
            how = 'chosen by BIC' if st['kchoice'].startswith('Auto') else 'set by hand'
            w.writerow(['Model', 'Discrete mixture (Galbraith & Laslett 1993)',
                        f"k = {fit['k']} ({how})", f"BIC choice k = {st['kb']}"])
        else:
            w.writerow(['Model', 'Minimum age model, 3 parameters (Galbraith et al. 1999)'])
        w.writerow(['Data', 'All analyses' if st['all_data'] else 'Included only',
                    f"n = {fit['n']}", f"{st['space']} scale"])
        w.writerow(['Component', 'Age (Ma)', f'±{sl} Ana (Ma)', f'±{sl} Int (Ma)',
                    f'±{sl} Ext (Ma)', 'Proportion', f'±{sl} Proportion', 'n assigned',
                    'n effective', 'MSWD', '√MSWD applied'])
        for i, tr in enumerate(tiers):
            if fit['model'] == 'fmm':
                name, pr, spr, na = (f'C{i + 1}', fit['prop'][i], fit['se_prop'][i],
                                     fit['n_assigned'][i])
            else:
                name, pr, spr = 'Minimum', fit['prop'], fit['se_prop']
                na = int(np.sum(np.nan_to_num(fit['post_min']) >= 0.5))
            w.writerow([name, f"{tr['age']:.8f}", f"{tr['ana']*ds:.8f}", f"{tr['int']*ds:.8f}",
                        f"{tr['ext']*ds:.8f}", f'{pr:.4f}',
                        '' if spr != spr else f'{spr*ds:.4f}', na, f"{tr['n_eff']:.2f}",
                        '' if tr['mswd'] != tr['mswd'] else f"{tr['mswd']:.4f}",
                        f"{tr['scale']:.4f}" if tr['scale'] > 1 else 'no'])
        if fit['model'] == 'fmm':
            bmin = min(f['bic'] for f in st['fits'])
            w.writerow(['k', *[f['k'] for f in st['fits']]])
            w.writerow(['ΔBIC', *[f"{f['bic'] - bmin:.3f}" for f in st['fits']]])
        else:
            w.writerow(['Spread of the older analyses (σ)', f"{fit['sigma']:.6g}",
                        'log units' if st['space'] == 'log' else 'Ma'])
        w.writerow([])

    def _export_mixture_data(self):
        """The mixture model of the sample shown in the Mixture view, and
        nothing else: its component ages, then each analysis the model used
        with its point data and its membership of every component.  The
        constants and raw signals are those of this sample only."""
        import csv
        st = self._mix_state
        s, fit = st['sample'], st['fit']
        name = (s.name or 'sample').replace(' ', '_') + '_mixture.xlsx'
        path, _flt = dfold.get_save_file_name(self, 'Export mixture model', name,
                                                 export.save_filter())
        if not path:
            return
        path = export.ensure_ext(path, _flt)
        ds = self.display_sigma
        idx = list(st['idx'])
        n = len(s.ages)
        ms = list(s.metas) if len(s.metas) == n else [export.step_meta({}) for _ in range(n)]
        lbls = s.labels if len(s.labels) == n else [''] * n
        rids = s.run_ids if len(s.run_ids) == n else [''] * n
        _sink = export.TableSink(path, results_note='The mixture model of the sample shown: its '
                               'components, then each analysis with its membership.')
        with _sink as fh:
            w = csv.writer(fh)
            w.writerow([f'# Mixture model of {s.name} only, as shown in the Mixture view'])
            w.writerow([])
            export.write_constants(w, groups=export.constants_groups(
                [(s.j_cfg, s.constants, [ms[i] for i in idx])]))
            _sink.set_raw_items([(s.name, [{'label': lbls[i], 'runID': rids[i]} for i in idx],
                                  [ms[i] for i in idx], s.constants)])
            self._write_mixture_rows(w, ds)
            comp, _prob, P = am.mixture_membership(fit)
            cols = ([f'P(C{i + 1})' for i in range(P.shape[1])] if fit['model'] == 'fmm'
                    else ['P(minimum)', 'P(older)'])
            w.writerow(['POINT DATA'])
            w.writerow([*self._POINT_HEADER, 'Most likely component', *cols])
            rows = self._point_rows(s)
            for j, i in enumerate(idx):
                if comp[j] < 0:
                    most = 'not used (age <= 0, log scale)'
                elif fit['model'] == 'fmm':
                    most = f'C{comp[j] + 1}'
                else:
                    most = 'Minimum' if comp[j] == 0 else 'Older'
                w.writerow([*rows[i], most, *['' if v != v else f'{v:.4f}' for v in P[j]]])
            w.writerow([])

    def _sample_report_text(self, s):
        """Detailed single-sample report for the bottom box."""
        ds = self.display_sigma
        r = s.wm()
        if r is None:
            return [f'{s.name}: no included data.']
        m = s.mask()
        L = ['-' * 58,
             f'SAMPLE: {s.name}',
             '-' * 58,
             f"  Weighted mean = {r['wm']:.5f} Ma   (n = {int(m.sum())}/{len(s.ages)})",
             f"  ±{ds}σ analytical = {r['sigma_ana']*ds:.5f} Ma",
             f"  ±{ds}σ internal   = {r['sigma_int']*ds:.5f} Ma   (+ J flux)",
             f"  ±{ds}σ external   = {r['sigma_ext']*ds:.5f} Ma   (+ FCs/λ)",
             f"  MSWD = {r['mswd']:.3f}   p = {r.get('prob', float('nan')):.3f}"
             f"   ({mswd_classify(r['mswd'], int(m.sum())) or '-'})"]
        if 'overdispersion' in r or r.get('overdispersed'):
            if 'overdispersion' in r:        # Vermeesch ML random-effects
                L.append(f"  ML random-effects ω = "
                         f"±{r['overdispersion']*ds:.5f} Ma ({ds}σ)")
            else:                            # √MSWD scaling
                L.append(f"  overdispersed → analytical × √MSWD = {r['mswd_scale']:.3f}")
        c = s.j_cfg
        mode = c.get('ageFluxMode', 'imported')
        L.append(f"  J mode: {mode}   {self._flux_desc(c)}")
        L.append(f"  t_FCs = {c.get('stdAgeMa', '?')} ± {c.get('stdAgeErrMa', '?')} Ma "
                 f"({c.get('stdAgeErrType', 'external')})")
        fm = self._flux_match(s)
        if fm is not None:
            key, i = fm
            fx = self._disk_flux[key]
            hole_id = next((h for h, ix in fx['idx'].items() if ix == i), str(i))
            L.append(f"  disk flux: irr \"{key[0]}\" disk \"{key[1]}\" hole {hole_id} "
                     f"(R={self._flux_desc_R(key, i)})")
        elif s.disk_id or s.hole_id:
            L.append(f"  identity: disk={s.disk_id} hole={s.hole_id} "
                     f"(no loaded disk-flux match)")
        return L

    def _active_all(self):
        """Enabled samples that have data (including ones with all points excluded)."""
        return [s for s in self.samples if s.enabled and len(s.ages) > 0]

    def _position_groups(self, act):
        """The samples' J position groups as the pool actually treats them:
        samples linked by a non-zero J correlation (the same j_correlation ρ
        pool() builds) form one group.  Returns [(members, description)]."""
        if not act:
            return []
        infos = [self._j_point_info(s, [int(np.where(s.mask())[0][0])]
                                    if s.mask().any() else [0])[0] for s in act]
        rho = self._j_corr(infos, force=self.corr_j)
        n = len(act)
        grp = list(range(n))

        def root(i):
            while grp[i] != i:
                i = grp[i]
            return i
        for i in range(n):
            for j in range(i + 1, n):
                if abs(rho[i, j]) > 1e-12:
                    grp[root(j)] = root(i)
        out = {}
        for i in range(n):
            out.setdefault(root(i), []).append(i)
        res = []
        all_r_fcs = all(s.j_cfg.get('ageFluxMode') == 'ratio_R_FCs' for s in act)
        for idx in out.values():
            members = [act[i] for i in idx]
            if all_r_fcs:
                desc = 'R_unk/R_FCs input: no J term shared between samples'
            elif len(idx) == 1:
                desc = 'independent'
            else:
                r = [rho[i, j] for k, i in enumerate(idx) for j in idx[k + 1:]]
                if all(abs(x - 1.0) < 1e-9 for x in r):
                    desc = 'shared J (ρ = 1), no averaging-down'
                else:
                    desc = f'correlated J, ρ = {min(r):.2f} to {max(r):.2f}'
                if any(self._flux_match(m) for m in members):
                    desc += ', from disk flux'
            res.append((members, desc))
        return res

    def _pool_text(self, pr, act):
        ds = self.display_sigma
        method_label = ('GLS + ML random-effects (after Vermeesch 2015 §13)'
                        if _STAT_METHOD == METHOD_GLS
                        else 'WM + √MSWD')
        _fs_warn = ''
        ext_route = external_route(act[0].j_cfg)
        if act[0].j_cfg.get('fullSystematics'):
            _fs_warn = full_systematics_warning(act[0].j_cfg)
        bar = '═' * 60
        L = [bar,
             f"  POOLED AGE     {pr['wm']:.4f} ± {pr['sigma_ext']*ds:.4f} Ma"
             f"   ({ds}σ, external)",
             bar,
             f"  {method_label}",
             f"  n = {pr['n']} analyses in {len(act)} samples"
             f"      MSWD = {pr['mswd']:.2f}   p = {pr.get('prob', float('nan')):.2f}"
             f"   ({mswd_classify(pr['mswd'], pr['n']) or '-'})",
             '',
             f"  Uncertainty budget ({ds}σ)          (Error breakdown for the pie chart)",
             f"    analytical    ± {pr['sigma_ana']*ds:.4f} Ma     mass-spec scatter",
             f"    internal      ± {pr['sigma_int']*ds:.4f} Ma     + J / neutron flux",
             f"    external      ± {pr['sigma_ext']*ds:.4f} Ma     + FCs age & decay const"
             f"  [{ext_route}]"]
        if _fs_warn:
            L.append(f"    Note: {_fs_warn}")
        _cal_warn = calibration_warning(act[0].j_cfg)
        if _cal_warn:
            L.append(f"    Note: {_cal_warn}")
        if 'overdispersion' in pr:           # Vermeesch: always report ω
            zeta = pr['overdispersion'] * ds
            tail = '  (no excess scatter)' if zeta == 0 else ''
            L.append(f"    overdispersion  ω = ± {zeta:.4f} Ma{tail}")
        elif pr['overdispersed']:            # √MSWD scaling
            L.append(f"    overdispersed → analytical × √MSWD = {pr['mswd_scale']:.3f}")

        # ── J position groups, as the pool's ρ has them ──────────────────────
        L += ['', '  J position groups' + ('  (Force corr. J: all ρ = 1)' if self.corr_j else '')]
        for members, shared in self._position_groups(act):
            jdesc = self._flux_desc(members[0].j_cfg)
            L.append(f"    {jdesc}   [{shared}]")
            L.append(f"        {', '.join(m.name for m in members)}")
        n_cal = len({calibration_group_key(s.j_cfg) for s in act})
        L.append('    FCs age / λ systematic: '
                 + ('one shared value (external tier)' if n_cal == 1 else
                    f'{n_cal} calibrations, each shared within its own samples'))

        # ── loaded disk-flux status ──────────────────────────────────────────
        if self._disk_flux:
            matched = [s for s in act if self._flux_match(s) is not None]
            if matched:
                L += ['', f'  Disk flux, {len(matched)}/{len(act)} samples matched '
                          '(J correlation between holes from the .jflux)']
                for s in matched:
                    key, i = self._flux_match(s)
                    hole_id = next((h for h, ix in self._disk_flux[key]['idx'].items()
                                    if ix == i), f'#{i}')
                    L.append(f'      {s.name}   disk {key[1]} hole {hole_id}'
                             f'   R = {self._flux_desc_R(key, i)}')
                if len(matched) >= 2:
                    L += ['', '  Cross-hole J correlation  ρ']
                    for ii in range(len(matched)):
                        for jj in range(ii + 1, len(matched)):
                            a, b = matched[ii], matched[jj]
                            ma, mb = self._flux_match(a), self._flux_match(b)
                            rho = flux_pair_corr(self._disk_flux, ma, mb)
                            if ma[0] == mb[0]:
                                tag = 'same hole' if ma[1] == mb[1] else 'same disk'
                            elif rho != 0.0:
                                tag = 'different disk, same fit'
                            else:
                                tag = 'different disk → independent'
                            L.append(f'      {a.name} ↔ {b.name}:  ρ = {rho:+.3f}   {tag}')
            else:
                L += ['', '  Disk flux: loaded, but no active sample matched by disk+hole '
                          '(check disk label / hole numbers).']
        L.append(bar)
        return L

    @staticmethod
    def _flux_desc(cfg):
        m = cfg.get('ageFluxMode', 'imported')
        if m == 'ratio_J':
            return f"J={cfg.get('jVal', 0):.8g} ±{cfg.get('jErrPct', 0):.4g}%"
        if m == 'ratio_R_std':
            return f"R_std={cfg.get('rStd', 0):.6g} ±{cfg.get('rStdErrPct', 0):.4g}%"
        if m == 'ratio_R_FCs':
            return f"t_FCs={cfg.get('stdAgeMa', 0):.5g}"
        return 'ages'

    # --- session + export ---------------------------------------------------
    def save_session(self):
        path, _ = dfold.get_save_file_name(self, 'Save session', 'session.arprob',
                                              'Age-Prob (*.arprob);;All (*)')
        if not path:
            return
        disk_flux_list = []
        for (irr, disk), fx in self._disk_flux.items():
            disk_flux_list.append(dict(
                irradiation=irr, disk_id=disk,
                holes=fx['holes'],
                hole_xy=fx['hole_xy'].tolist(),
                design=fx.get('design', ''),
                J=fx['J'].tolist(),
                J_sigma=fx['J_sigma'].tolist(),
                J_corr=fx['J_corr'].tolist(),
                R=(fx['R'].tolist() if fx.get('R') is not None else None),
                R_sigma=(fx['R_sigma'].tolist() if fx.get('R_sigma') is not None else None),
                lamFlux=fx.get('lamFlux'), fcFlux=fx.get('fcFlux'),
                fcFluxErr=fx.get('fcFluxErr'),
                fit=fx.get('fit'), xcorr=serialize_xcorr(fx),
            ))
        data = dict(
            version=3,
            display_sigma=self.display_sigma,
            age_dec=self.age_dec,
            input_sigma=self.input_sigma,
            pool_on=self.pool_on,
            same_fcs=self.same_fcs,
            corr_j=self.corr_j,
            stat_method=_STAT_METHOD,
            xlim_mode=self.xlim_mode,
            x_lo=self.x_lo,
            x_hi=self.x_hi,
            ylim_mode=self.ylim_mode,
            y_max=self.y_max,
            y_mode=self.y_mode,
            preferred_flux=list(self._preferred_flux) if self._preferred_flux else None,
            disk_flux=disk_flux_list,
            mixture=self._mix_settings(),
            samples=[dict(
                name=s.name, raw_name=s.raw_name, color=s.color, enabled=s.enabled,
                is_plateau=s.is_plateau,
                input_mode=s.input_mode, reject=s.reject, j_cfg=s.j_cfg,
                from_file=s._from_file, nmad_thresh=s.nmad_thresh,
                in_sigma=s.in_sigma,
                disk_id=s.disk_id, hole_id=s.hole_id, irradiation=s.irradiation,
                flux_override=(list(s._flux_override)
                               if getattr(s, '_flux_override', None) else None),
                ages=s.ages.tolist(), unc_2s=s.unc_2s.tolist(),
                r_vals=s.r_vals.tolist(), r_uncs=s.r_uncs.tolist(),
                kca_vals=s.kca_vals.tolist(), kca_errs=s.kca_errs.tolist(),
                ar40_pct=s.ar40_pct.tolist(), labels=s.labels, run_ids=s.run_ids,
                metas=s.metas, constants=s.constants,
                manual=s.manual.tolist(),
                pt_pos=s.pt_pos,
                pt_jfrac=(None if s.pt_jfrac is None else
                          [None if np.isnan(v) else float(v) for v in s.pt_jfrac]),
                loaded_excluded=(s._loaded_excluded.tolist()
                                 if len(s._loaded_excluded) == len(s.ages) else []))
                for s in self.samples])
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_session(self):
        path, _ = dfold.get_open_file_name(self, 'Load session', '',
                                              'Age-Prob (*.arprob);;All (*)')
        if not path:
            return
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for p in list(self.panels):
            p.setParent(None)
            p.deleteLater()
        self.panels.clear()
        self.samples.clear()
        self.display_sigma = data.get('display_sigma', 2)
        self.age_dec = int(data.get('age_dec', 3))
        if hasattr(self, 'dec_sel'):
            self.dec_sel.blockSignals(True)
            self.dec_sel.setValue(self.age_dec)
            self.dec_sel.blockSignals(False)
        self.input_sigma = data.get('input_sigma', 1)
        self.same_fcs = data.get('same_fcs', True)
        self.corr_j = data.get('corr_j', False)
        self.pool_on = data.get('pool_on', False)

        global _STAT_METHOD
        _STAT_METHOD = method_key(data.get('stat_method', METHOD_WM))

        self.xlim_mode = data.get('xlim_mode', 'auto')
        self.x_lo = data.get('x_lo')
        self.x_hi = data.get('x_hi')
        self.ylim_mode = data.get('ylim_mode', 'auto')
        self.y_max = data.get('y_max')
        self.y_mode = data.get('y_mode', 'rank')

        self._disk_flux = deserialize_disk_flux(data.get('disk_flux', []))
        pf = data.get('preferred_flux')
        self._preferred_flux = tuple(pf) if pf else None

        for sd in data.get('samples', []):
            s = _sample_from_dict(sd)
            self.add_sample(s)
        # A sample on a hole of the loaded disk flux takes that hole's J, as in
        # the flux dialog (the J saved with it can be an older one).
        if self._disk_flux:
            self._apply_flux_to_samples()

        self.cb_corr.setChecked(self.corr_j)
        self.sig_sel.setCurrentIndex(self.display_sigma - 1)
        self.in_sig_sel.setCurrentIndex(0 if self.input_sigma == 1 else 1)
        self.method_sel.setCurrentIndex(1 if _STAT_METHOD == METHOD_GLS else 0)
        self.ymode_sel.setCurrentIndex(
            {'rank': 0, 'relative': 1, 'probability': 2,
             'normalised': 3}.get(self.y_mode, 0))

        self.pool_btn.blockSignals(True)
        self.pool_btn.setChecked(self.pool_on)
        self.pool_btn.setText(f'Pool: {"on" if self.pool_on else "off"}')
        self.pool_btn.blockSignals(False)

        if self.xlim_mode == 'manual' and self.x_lo is not None and self.x_hi is not None:
            self.x_lo_ed.setText(f'{self.x_lo:.3f}')
            self.x_hi_ed.setText(f'{self.x_hi:.3f}')
        else:
            self.x_lo_ed.clear()
            self.x_hi_ed.clear()
        if self.ylim_mode == 'manual' and self.y_max is not None:
            self.y_max_ed.setText(f'{self.y_max:.3f}')
        else:
            self.y_max_ed.clear()
        self._mix_cache.clear()
        self._apply_mix_settings(data.get('mixture'))

        self.redraw()

    # ── Figure export (shared dialog: layout, sizes, text, legend, preview) ──
    def _export_panels(self):
        return [export.Panel('plot', 'Age–probability plot', fixed=True, weight=1.0)]

    def _export_options(self):
        act = self._active_all()
        opts = [
            export.Option('curve', 'Probability curves', 'bool', True),
            export.Option('fill', 'Shading under the curves', 'bool', True),
            export.Option('dashed', 'Dashed curve (excluded)', 'bool', True),
            export.Option('points', 'Individual analyses', 'bool', True),
            export.Option('wm', 'Weighted-mean line', 'bool', True),
            export.Option('split', 'Samples in separate panels', 'bool', False,
                          tip='The samples in panels stacked on one age axis, one '
                              'sample each unless set otherwise under Panels, '
                              'with the pooled curve in every panel when Pool is '
                              'on. Best with the Y axis on Normalised.'),
            export.Option('legend_ages', 'Ages in legend', 'choice',
                          values=[('all', 'WM ± analytical / internal / external'),
                                  ('int', 'WM ± internal'),
                                  ('ana', 'WM ± analytical'),
                                  ('ext', 'WM ± external'),
                                  ('off', 'Names only')],
                          tip='What follows each name in the legend, at the 1σ/2σ '
                              'level and decimals set in the window. Internal = '
                              'analytical + J. MSWD and p stay with the ages.'),
            export.Option('legend_n', 'n in legend', 'bool', True,
                          tip='Included analyses out of all, e.g. n = 46/56: after each sample '
                              'when Pool is off, on the pooled entry only when '
                              'Pool is on.'),
            export.Option('pool_legend', 'Pooled age in legend', 'choice',
                          values=[('same', 'As the samples'),
                                  ('int', 'Pooled age ± internal'),
                                  ('ana', 'Pooled age ± analytical'),
                                  ('ext', 'Pooled age ± external'),
                                  ('all', 'Pooled age ± all three'),
                                  ('name', 'Name only')],
                          tip='The pooled entry on its own, e.g. names only for '
                              'the samples but the pooled age with its error. '
                              'Only shown when Pool is on.'),
            # Names: the main window's, the experiment names as loaded (before
            # any renaming), or typed.  The fields are always editable and show
            # the chosen names; pick 'Typed below' to put edits in the figure.
            export.Option(
                'names_from', 'Names in the figure', 'choice', group='Names',
                wide=True,
                values=[('window', 'Main window'),
                        ('raw', 'Experiment names (as loaded, before renaming)'),
                        ('typed', 'Typed below')],
                tip='Main window: the sample names as they are now. Experiment '
                    'names: the names the files came with, even if renamed since. '
                    'The fields below show the chosen names; edit them, then '
                    'pick "Typed below" to use your edits.'),
            export.Option('title', 'Title', 'text', group='Names', wide=True,
                          placeholder='no title'),
        ]
        # only the samples that are drawn (ticked, with data) get a field
        for k, sm in enumerate(act):
            exp = sm.experiment_name()
            opts.append(export.Option(
                f'name_{k}', f'Sample {k + 1}' + ('' if exp else ' (no exp. name)'),
                'text', group='Names', wide=True, default=sm.name,
                tip=('' if exp else
                     'No experiment name was saved with this sample (it was '
                     'loaded before experiment names were kept). Load its '
                     'file again to get it; until then its current name is used.'),
                follows=('names_from', {'window': sm.name, 'raw': exp or sm.name}),
                ident=exp or sm.name))
        if len(act) > 1:
            for k, sm in enumerate(act):
                opts.append(export.Option(
                    f'panel_{k}', sm.name, 'choice', group='Panels',
                    values=[(str(i + 1), f'Panel {i + 1}') for i in range(len(act))],
                    default=str(k + 1), depends=('split', True),
                    tip='Samples given the same panel are drawn together in it, '
                        'from the top panel down.',
                    ident=sm.experiment_name() or sm.name))
        opts += [
            export.Option('x_min', 'Age from', 'text', group='Axis ranges',
                          placeholder='auto'),
            export.Option('x_max', 'to', 'text', group='Axis ranges',
                          placeholder='auto'),
            export.Option('y_min', 'Y from', 'text', group='Axis ranges',
                          placeholder='auto'),
            export.Option('y_max', 'to', 'text', group='Axis ranges',
                          placeholder='auto'),
        ]
        return opts

    @staticmethod
    def _opt_float(spec, key):
        """A numeric export option, or None when the field is blank."""
        try:
            txt = str(spec['options'].get(key, '')).strip().replace(',', '.')
            return float(txt) if txt else None
        except (TypeError, ValueError):
            return None

    def _export_render(self, fig, spec):
        """Draw the plot into *fig* exactly as the dialog asks."""
        opts = spec.get('options', {})
        fig.clear()
        act = self._active_all()
        split = bool(opts.get('split')) and len(act) > 1
        groups = [None]
        if split:
            pnum = []
            for k in range(len(act)):
                try:
                    pnum.append(int(opts.get(f'panel_{k}', k + 1)))
                except (TypeError, ValueError):
                    pnum.append(k + 1)
            groups = [[sm for sm, q in zip(act, pnum) if q == g]
                      for g in sorted(set(pnum))]
            axes = [fig.add_subplot(len(groups), 1, 1)]
            axes += [fig.add_subplot(len(groups), 1, k + 1, sharex=axes[0])
                     for k in range(1, len(groups))]
        else:
            axes = [fig.add_subplot(111)]
        ax = axes[0]
        old_fig, old_ax = self.fig, self.ax
        old_clicks = self._click_pts
        old_names = [sm.name for sm in act]
        src = opts.get('names_from', 'window')
        for k, sm in enumerate(act):                  # renamed for this figure
            nm = (sm.experiment_name() if src == 'raw' else
                  opts.get(f'name_{k}') if src == 'typed' else None)
            if nm:
                sm.name = nm
        self.fig = fig
        self._offscreen = True
        self._export_show = dict(opts)
        try:
            for k, a in enumerate(axes):
                self.ax = a
                self._only_sample = groups[k]
                self.redraw()
        finally:
            self.fig, self.ax = old_fig, old_ax
            self._click_pts = old_clicks
            self._offscreen = False
            self._export_show = {}
            self._only_sample = None
            for sm, nm in zip(act, old_names):
                sm.name = nm
        for a in axes[:-1]:                 # one age axis, at the bottom
            a.set_xlabel('')
            a.tick_params(labelbottom=False)
        if split:                           # and one y label for the stack
            ylab = ax.get_ylabel()
            for a in axes:
                a.set_ylabel('')
            fig.supylabel(ylab, fontsize=ax.xaxis.label.get_fontsize())
        if opts.get('title'):
            ax.set_title(opts['title'], fontsize=11, fontweight='bold')
        x0, x1 = self._opt_float(spec, 'x_min'), self._opt_float(spec, 'x_max')
        if split and x0 is None and x1 is None:
            # the same age range in every panel: the widest of them
            lims = [a.get_xlim() for a in axes]
            ax.set_xlim(min(l[0] for l in lims), max(l[1] for l in lims))
        if x0 is not None or x1 is not None:
            cur = ax.get_xlim()
            ax.set_xlim(cur[0] if x0 is None else x0,
                        cur[1] if x1 is None else x1)
        y0, y1 = self._opt_float(spec, 'y_min'), self._opt_float(spec, 'y_max')
        for a in axes:
            if y0 is not None or y1 is not None:
                cur = a.get_ylim()
                a.set_ylim(cur[0] if y0 is None else y0,
                           cur[1] if y1 is None else y1)
        if len(axes) > 1 and spec.get('legend') in ('below', 'above', 'outside'):
            # outside the panels: one legend for the stack, each entry once
            self._merge_legends(axes, axes[-1] if spec.get('legend') == 'below'
                                else axes[0])
            export.place_legend(axes[-1] if spec.get('legend') == 'below'
                                else axes[0], spec)
        else:
            for a in axes:
                export.place_legend(a, spec)
        export.scale_text(fig, spec.get('font_scale', 1.0),
                          spec.get('line_scale', 1.0), spec.get('marker_scale', 1.0))
        self._fixed_frame(fig, spec, has_title=bool(opts.get('title')))
        if split:
            fig.subplots_adjust(hspace=0.08)

    @staticmethod
    def _merge_legends(axes, target):
        """Move the legends of *axes* into one on *target*, keeping each
        entry once (the pooled entry is in every panel)."""
        handles, labels, first = [], [], None
        for a in axes:
            lg = a.get_legend()
            if lg is None:
                continue
            first = first or lg
            hs = list(getattr(lg, 'legend_handles', None)
                      or getattr(lg, 'legendHandles', []))
            for h, t in zip(hs, lg.get_texts()):
                if t.get_text() not in labels:
                    handles.append(h)
                    labels.append(t.get_text())
        if first is None:
            return
        # the pooled entry last, after every sample
        order = sorted(range(len(labels)),
                       key=lambda i: labels[i].lstrip().startswith('Pooled'))
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
        kw = {}
        if first.get_texts():
            kw['prop'] = first.get_texts()[0].get_fontproperties()
        if first.get_title().get_text():
            kw['title'] = first.get_title().get_text()
            kw['title_fontsize'] = first.get_title().get_fontsize()
        for a in axes:
            if a.get_legend() is not None:
                a.get_legend().remove()
        target.legend(handles=handles, labels=labels, **kw)

    @staticmethod
    def _fixed_frame(fig, spec, has_title=False):
        """Place the plot frame from the page size, text size and title only.

        tight_layout() fitted the frame around whatever surrounded it, so a
        longer legend (below or beside the plot) or wider tick labels shrank
        the diagram.  These margins (points, grown with the text size) depend
        on the settings alone, so two figures with the same settings get
        frames of exactly the same size.  A legend outside the frame, or a
        label that needs more room, still goes in the file: the export crops
        to the drawing, so it extends the file instead of squeezing the plot."""
        fs = float(spec.get('font_scale', 1.0) or 1.0)
        w_pt, h_pt = (float(v) * 72.0 for v in fig.get_size_inches())
        left, bottom = 20.0 + 26.0 * fs, 19.0 + 21.0 * fs
        right = 10.0 + 7.0 * fs
        top = (16.0 + 10.0 * fs) if has_title else (8.0 + 6.0 * fs)
        l, r = left / w_pt, 1.0 - right / w_pt
        b, t = bottom / h_pt, 1.0 - top / h_pt
        if r - l < 0.2 or t - b < 0.2:          # a tiny page: keep some plot
            l, r, b, t = 0.15, 0.97, 0.15, 0.92
        fig.subplots_adjust(left=l, right=r, bottom=b, top=t)

    def export_figure(self):
        """Open the shared export dialog (panels, page size, text, legend)."""
        if self.mix_on:
            return self._export_mixture_figure()
        act = self._active_all()
        name = 'ageprob'
        if len(act) == 1:
            name = (act[0].name or 'ageprob').replace(' ', '_')
        dlg = export.FigureExportDialog(
            self, 'ageprob', self._export_panels(), self._export_options(),
            self._export_render, default_name=name,
            snapshot=self._export_snapshot)
        dlg.exec_()
        self.redraw()

    def _export_snapshot(self, spec):
        """Axis ranges the figure is drawn with now, for a preset to keep:
        a blank 'auto' range would otherwise follow whatever the window
        shows when the preset is loaded later."""
        fig = Figure(figsize=export.figsize(spec))
        self._export_render(fig, spec)
        ax = fig.axes[0]
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        return {'x_min': f'{x0:.4g}', 'x_max': f'{x1:.4g}',
                'y_min': f'{y0:.4g}', 'y_max': f'{y1:.4g}'}

    _POINT_HEADER = ['Sample', 'RunID', 'Run', 'Run date/time',
                     'Days since irradiation', *export.FLUX_HEADERS,
                     'Age (Ma)',
                     '±1σ Ana (Ma)', '±1σ Int (Ma)',
                     '40Ar*/39ArK', '±1σ',
                     'K/Ca', '±1σ', '%40Ar*',
                     'R_unk/R_std',
                     '±1σ Ana', '±1σ Int',
                     'Status']

    def _point_rows(self, s):
        """The POINT DATA rows of one sample, one per analysis in run order
        (the columns of _POINT_HEADER).  Analytical and internal only: the
        external tier belongs to the final (sample, pooled, component) ages."""
        rows = []
        m = s.mask()
        auto = s.auto_mask if len(s.auto_mask) == len(s.ages) else m
        has_r = len(s.r_vals) == len(s.ages) and len(s.ages) > 0
        has_kca   = len(s.kca_vals) == len(s.ages) and len(s.ages) > 0
        has_kca_e = len(s.kca_errs) == len(s.ages) and len(s.ages) > 0
        has_ar40  = len(s.ar40_pct) == len(s.ages) and len(s.ages) > 0
        has_rid   = len(s.run_ids)  == len(s.ages) and len(s.ages) > 0
        mode = s.j_cfg.get('ageFluxMode', 'imported')
        r_std = s.j_cfg.get('rStd', 0.0) if mode in ('ratio_J', 'ratio_R_std') else None
        lam = s.j_cfg.get('lambdaYr', LAMBDA_DEFAULT)
        em1 = cfg_ingrowth(s.j_cfg, lam)
        _ages_all = np.asarray(s.ages, float)
        _sana_all = s.sig1()
        _sint_all = s.sig_int(_ages_all, _sana_all, np.arange(len(s.ages)))
        for i in range(len(s.ages)):
            man = s.manual[i] if i < len(s.manual) else 'auto'
            if man == 'in':
                status = 'forced-in'
            elif man == 'out':
                status = 'forced-out'
            else:
                status = 'auto-in' if auto[i] else f'auto-out ({s.reject})'
            age_i = s.ages[i]
            sig1_ana = _sana_all[i]
            sig1_int = _sint_all[i]
            # ratio_R_FCs stores R_unk/R_FCs in r_vals, not 40Ar*/39ArK
            rv = ff(s.r_vals[i], 8) if has_r and mode != 'ratio_R_FCs' else ''
            ru = ff(s.r_uncs[i], 8) if has_r and mode != 'ratio_R_FCs' else ''
            _cak = s.kca_vals[i] if has_kca else float('nan')
            if has_kca and not math.isnan(_cak) and _cak != 0.0:
                kca_str = ff(1.0 / _cak, 1)
                _cak_e = s.kca_errs[i] if has_kca_e else float('nan')
                kca_e_str = ('' if math.isnan(_cak_e)
                             else ff(_cak_e / (_cak * _cak), 1))
            else:
                kca_str = ''
                kca_e_str = ''
            ar40_str = ('' if not has_ar40 or math.isnan(s.ar40_pct[i])
                        else f'{s.ar40_pct[i]:.1f}')
            if has_r and mode == 'ratio_R_FCs':
                ratio = s.r_vals[i]
                rrat = ff(ratio, 8)
                dr_dt = lam * (1.0 + em1 * ratio) * 1e6 / em1
                rrat_ana = ff(sig1_ana * dr_dt, 8)
                rrat_int = ff(sig1_int * dr_dt, 8)
            elif has_r and r_std and r_std > 0:
                ratio = s.r_vals[i] / r_std
                _m0 = s.metas[i] if len(s.metas) == len(s.ages) else {}
                if s.pt_pos is not None and s._flux_src is None and _m0.get('fx'):
                    # own measured ratio over own monitor ratio
                    try:
                        _rs0 = float(_m0['fx'][2])
                        if _rs0 > 0 and _m0.get('r_meas') is not None:
                            ratio = float(_m0['r_meas']) / _rs0
                    except (TypeError, ValueError):
                        pass
                rrat = ff(ratio, 8)
                dr_dt = lam * (1.0 + em1 * ratio) * 1e6 / em1
                rrat_ana = ff(sig1_ana * dr_dt, 8)
                rrat_int = ff(sig1_int * dr_dt, 8)
            else:
                rrat = rrat_ana = rrat_int = ''
            rid = s.run_ids[i]  if has_rid else ''
            _mt = (s.metas[i] if len(s.metas) == len(s.ages)
                   else export.step_meta({}))
            # A sample merged from several irradiation positions keeps each
            # point's own J, R standard and measured 40Ar*/39ArK (the sample
            # works on the first position's J, with R rescaled to it). A
            # disk-flux assignment overrides this: it sets one R for the sample.
            if (s.pt_pos is not None and s._flux_src is None
                    and _mt.get('fx') is not None):
                _flx = _mt['fx']
                if has_r and mode != 'ratio_R_FCs' and _mt.get('r_meas') is not None:
                    rv = ff(_mt['r_meas'], 8)
                    ru = ff(_mt['r_meas_unc'], 8) if _mt.get('r_meas_unc') is not None else ''
            else:
                _flx = export.flux_cells(s.j_cfg, show_r=bool(has_r and r_std and r_std > 0))
            rows.append([s.name, rid, i + 1, _mt.get('run_dt', ''),
                         export.fmt_days(_mt), *_flx,
                         f'{age_i:.8f}',
                         f'{sig1_ana:.8f}', f'{sig1_int:.8f}',
                         rv, ru,
                         kca_str, kca_e_str, ar40_str,
                         rrat, rrat_ana, rrat_int,
                         status])
        return rows

    def export_csv(self):
        """Write all per-point data plus weighted-mean results and calibration.
        With the Mixture view on: the mixture model of the sample shown only."""
        if self.mix_on and self._mix_state is not None:
            return self._export_mixture_data()
        import csv
        path, _flt = dfold.get_save_file_name(self, 'Export data', 'ageprob_data.xlsx',
                                                 export.save_filter())
        if not path:
            return
        path = export.ensure_ext(path, _flt)
        ds = self.display_sigma
        sig_lbl = f'{ds}σ'
        method_label = ('GLS + ML random-effects (after Vermeesch 2015 §13)'
                        if _STAT_METHOD == METHOD_GLS
                        else 'WM + √MSWD')
        _sink = export.TableSink(path, results_note='The analyses of each sample, the sample results and the pooled result.')
        with _sink as fh:
            w = csv.writer(fh)
            w.writerow([f'# Statistics: {method_label}  |  Covariance: shared J + FCs/λ (Vermeesch 2015 §13, both methods)'])
            w.writerow([])

            # ---- constants: one box for the whole export ----------------------
            _raw_items, _entries = [], []
            for s in self.samples:
                if len(s.ages) == 0:
                    continue
                _ms = (list(s.metas) if len(s.metas) == len(s.ages)
                       else [export.step_meta({}) for _ in s.ages])
                _lbls = s.labels if len(s.labels) == len(s.ages) else [''] * len(s.ages)
                _rids = s.run_ids if len(s.run_ids) == len(s.ages) else [''] * len(s.ages)
                _raw_items.append((s.name,
                                   [{'label': l, 'runID': r} for l, r in zip(_lbls, _rids)],
                                   _ms, s.constants))
                _entries.append((s.j_cfg, s.constants, _ms))
            if _entries:
                export.write_constants(w, groups=export.constants_groups(_entries))
            _sink.set_raw_items(_raw_items)

            # ---- per-point data ------------------------------------------------
            w.writerow(['POINT DATA'])
            w.writerow(self._POINT_HEADER)
            for s in self.samples:
                if len(s.ages) == 0:
                    continue
                for row in self._point_rows(s):
                    w.writerow(row)
            w.writerow([])

            # ---- per-sample results + calibration -----------------------------
            w.writerow(['SAMPLE RESULTS'])
            w.writerow(['Sample', 'n incl', 'n total', 'WM Age (Ma)',
                        f'±{sig_lbl} Ana (Ma)', f'±{sig_lbl} Int (Ma)',
                        f'±{sig_lbl} Ext (Ma)',
                        'R_unk/R_std', f'±{sig_lbl} Ana R', f'±{sig_lbl} Int R', f'±{sig_lbl} Ext R',
                        'MSWD', 'p',
                        'Reject', 'NMAD thr', 'Age source',
                        'J', '±1σ J', '±1σ J (%)',
                        'R standard', '±1σ R standard', '±1σ R standard (%)',
                        'FC age (Ma)', '±1σ FC age (Ma)', 'FC err type',
                        'σ in', 'Paste tier', 'J calibration'])
            for s in self.samples:
                r = s.wm()
                if r is None:
                    continue
                c = s.j_cfg
                mode = c.get('ageFluxMode', 'imported')
                if mode in ('ratio_J', 'ratio_R_std'):
                    j_val = c.get('jVal', '')
                    j_pct = c.get('jErrPct', '')
                    j_err = fsig(j_val * j_pct / 100.0, 6) if (j_val != '' and j_pct != '') else ''
                    j_val = fsig(j_val, 8) if j_val != '' else ''
                    j_pct = f"{j_pct:.5f}" if j_pct != '' else ''
                    r_val = c.get('rStd', '')
                    r_pct = c.get('rStdErrPct', '')
                    r_err = fsig(r_val * r_pct / 100.0, 6) if (r_val != '' and r_pct != '') else ''
                    r_val = fsig(r_val, 8) if r_val != '' else ''
                    r_pct = f"{r_pct:.5f}" if r_pct != '' else ''
                    paste_tier = ''
                elif mode == 'ratio_R_FCs':
                    j_val = j_err = ''
                    j_pct = f"{c.get('rfcsErrPct', 0.0):.5f}"
                    r_val = r_err = r_pct = ''
                    paste_tier = c.get('pasteTier', 'analytical')
                else:
                    j_val = j_err = j_pct = ''
                    r_val = r_err = r_pct = ''
                    paste_tier = ''
                _rr, _drdt = rr_from_age(r['wm'], c)
                _rr_str  = ff(_rr, 8)
                _rra_str = ff(_drdt * r['sigma_ana'] * ds, 8) if _drdt is not None else ''
                _rri_str = ff(_drdt * r['sigma_int'] * ds, 8) if _drdt is not None else ''
                _rre_str = ff(_drdt * r['sigma_ext'] * ds, 8) if _drdt is not None else ''
                w.writerow([
                    s.name, int(s.mask().sum()), len(s.ages), f"{r['wm']:.8f}",
                    f"{r['sigma_ana']*ds:.8f}", f"{r['sigma_int']*ds:.8f}",
                    f"{r['sigma_ext']*ds:.8f}",
                    _rr_str, _rra_str, _rri_str, _rre_str,
                    f"{r['mswd']:.4f}",
                    f"{r.get('prob', float('nan')):.4f}",
                    s.reject, s.nmad_thresh, _JMODE_DISPLAY.get(mode, mode),
                    j_val, j_err, j_pct,
                    r_val, r_err, r_pct,
                    c.get('stdAgeMa', ''), c.get('stdAgeErrMa', ''),
                    c.get('stdAgeErrType', ''), s.in_sigma, paste_tier,
                    self._flux_desc(c)])
            w.writerow([])

            # ---- pooled result -------------------------------------------------
            if self.pool_on:
                pr = self.pool()
                if pr:
                    w.writerow(['POOLED RESULT'])
                    w.writerow(['WM Age (Ma)',
                                f'±{sig_lbl} Ana (Ma)', f'±{sig_lbl} Int (Ma)',
                                f'±{sig_lbl} Ext (Ma)',
                                'R_unk/R_std', f'±{sig_lbl} Ana R', f'±{sig_lbl} Int R', f'±{sig_lbl} Ext R',
                                'MSWD', 'p', 'n',
                                'Overdispersed', 'MSWD scale',
                                'N position groups', 'Force corr. J'])
                    act = self._active()
                    _groups = self._position_groups(act)
                    ngrp = len(_groups)
                    _pool_cfg = act[0].j_cfg if act else {}
                    _rr_pr, _drdt_pr = rr_from_age(pr['wm'], _pool_cfg)
                    _rr_pool_str     = ff(_rr_pr, 8)
                    _rr_pool_ana_str = ff(_drdt_pr * pr['sigma_ana'] * ds, 8) if _drdt_pr else ''
                    _rr_pool_int_str = ff(_drdt_pr * pr['sigma_int'] * ds, 8) if _drdt_pr else ''
                    _rr_pool_ext_str = ff(_drdt_pr * pr['sigma_ext'] * ds, 8) if _drdt_pr else ''
                    w.writerow([
                        f"{pr['wm']:.8f}", f"{pr['sigma_ana']*ds:.8f}",
                        f"{pr['sigma_int']*ds:.8f}", f"{pr['sigma_ext']*ds:.8f}",
                        _rr_pool_str, _rr_pool_ana_str, _rr_pool_int_str, _rr_pool_ext_str,
                        f"{pr['mswd']:.4f}", f"{pr.get('prob', float('nan')):.4f}",
                        pr['n'], 'yes' if pr['overdispersed'] else 'no',
                        f"{pr['mswd_scale']:.4f}", ngrp, 'yes' if self.corr_j else 'no'])
                    w.writerow([])
                    w.writerow(['Pool method', method_label])
                    if 'overdispersion' in pr:
                        w.writerow(['Overdispersion',
                                    f"ML random-effects ω = ±{pr['overdispersion']*ds:.5f} Ma ({ds}σ)"])
                    elif pr['overdispersed']:
                        w.writerow(['Overdispersion',
                                    f"WM √MSWD scale = {pr['mswd_scale']:.3f}"])
                    # position groups exactly as the pool's ρ has them
                    w.writerow(['Position groups', str(ngrp),
                                'all correlated, ρ=1 (Force corr. J)' if self.corr_j
                                else 'from the J correlation used in the pool'])
                    for g_idx, (gmembers, gshared) in enumerate(_groups, 1):
                        w.writerow([f'Position group {g_idx}',
                                    _JMODE_DISPLAY.get(gmembers[0].j_cfg.get('ageFluxMode', 'imported'), ''),
                                    self._flux_desc(gmembers[0].j_cfg), gshared,
                                    ', '.join(_m.name for _m in gmembers)])
                    _ncal = len({calibration_group_key(_s.j_cfg) for _s in act})
                    w.writerow(['Standard',
                                'Shared FCs/λ systematic (external tier)' if _ncal == 1 else
                                f'{_ncal} calibrations, each shared within its own samples'])
                    if self._disk_flux:
                        dm = [_s for _s in act if self._flux_match(_s) is not None]
                        for _s in dm:
                            _km, _im = self._flux_match(_s)
                            _hole = next(
                                (h for h, ix in self._disk_flux[_km]['idx'].items()
                                 if ix == _im), f'#{_im}')
                            w.writerow(['Disk flux', _s.name,
                                        f'irr "{_km[0]}"', f'disk "{_km[1]}"',
                                        f'hole {_hole}',
                                        f'R={self._flux_desc_R(_km, _im)}'])


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Age distribution")
    dfold.use_tool('age_distribution')
    force_light(app)
    no_wheel_on_dropdowns(app)
    win = ProbabilityWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()