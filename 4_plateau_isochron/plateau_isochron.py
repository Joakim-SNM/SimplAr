#!/usr/bin/env python3
"""
Plateau & isochron: step-heating age spectra, plateau and isochron ages, and
the combination of several aliquots (stack mode).

Paste tab-separated data directly from Excel, or import an .xlsx / .csv file.
Column headers are auto-detected from common naming conventions.

Uncertainty convention
-----------------------
All uncertainties are stored and propagated internally at 1σ (step keys
'err1s', 'err_r4039', 'err_cak', 'j_err', and the isochron errors are all 1σ;
plateau-result tiers are 'e1s_ana'/'e1s_int'/'e1s_ext', also 1σ).  The 1σ-vs-2σ
choice is applied only at the point of display/export via the "1σ / 2σ" toggle
(self._rb_disp_1s / _rb_disp_2s); imported values are converted to 1σ once, on
input, and never bounce between 1σ and 2σ thereafter.

On disk (.arar / .ararpt / .ararstack) the three measured errors are written at
2σ under the legacy key 'err2s' (with cfg['errScale'] == 2) so files stay
readable by older versions of the tool and by the ArArCALC converters; the 2σ↔1σ
conversion happens only at the I/O boundary (_load_session_data / _steps_to_disk).

Layout of this file, top to bottom
-----------------------------------
  plateau finder, Ca/K axis, cumulative %39Ar, column auto-detection
  dialogs: column mapping, age recalculation, quick calibration
  session persistence (.arar / .ararpt / .ararstack)
  canvases: spectrum (PlateauCanvas), overlay, isochron (IsochronCanvas)
  disk-flux hosts and the combination-statistics dialog (_StatsDialog)
  PlateauWindow: UI, data lifecycle, table, summary, plot, import, session
    save/load, aliquot carousel, stack/isochron modes, figure and data export,
    flux groups, stacking method and disk flux, combined plateau, Paste+,
    .ararpt loading, stack save/load, stack CSV export
  paste-aliquot dialog, entry point
"""

import sys
import os
import re
import json
import math
import copy

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QPushButton, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QTabWidget,
    QGridLayout,
    QFrame, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QButtonGroup, QRadioButton, QGroupBox, QDialog, QDialogButtonBox,
    QMessageBox, QAction, QSizePolicy, QScrollArea,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QStackedWidget,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

import pandas as pd

# shared math/constants live one folder up, next to the launcher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arar_math import (
    chi2_cdf, t_crit, LAMBDA_DEFAULT, LAMBDA_ERR, STD_AGE_DEFAULT,
    STD_AGE_ERR, LAMBDA_PRESETS, FC_PRESETS, weighted_mean,
    combine_ages_vermeesch, mswd_classify, mswd_critical, york_regression,
    cfg_j_err_frac, r_to_age, rr_from_age, fsig, ff,
    paired_lambda_index, paired_lambda, DEFAULT_STD_AGE_ERR_TYPE,
    external_sigma_array, age_error_budget,
    full_systematics_warning, calibration_warning, force_light, no_wheel_on_dropdowns, calibration_group_key,
    internal_sigma_array, systematic_matrix, dt_dlnR, fc_ingrowth, j_correlation, match_flux_hole, flux_hole_matches,
    external_route, DEFAULT_FC_NAME, DEFAULT_LAMBDA_NAME,
    cfg_J, scale_sem, position_ident, read_jflux, serialize_disk_flux,
    deserialize_disk_flux, merge_disk_flux, flux_hole_R, flux_hole_rel, flux_hole_J,
    isochron_age, draw_budget_pie, combine, file_age_calibration, file_err_div,
    METHOD_WM, METHOD_GLS, method_key,
)
from flux_dialog import FluxAssignDialog, FluxHost
from calibration_dialog import CalibrationDialog
from standards import MonitorCombo, monitor_text, route_line
from arar_math import (route_cfg, route_from_cfg, route_text, route_options, custom_route,
                       monitor_ingrowth, monitor_age, ROUTE_KEYS, set_route, STANDARD_IDS,
                       reload_standards, sync_route, STANDARDS, flux_hole_rstd)
from arar_math import (MONITOR_KEYS, to_fcs_r, from_fcs_r, cfg_monitor, calibration_notes,
                       monitor_mismatch_note)
import export
import data_folders as dfold


# Human-readable labels for ageFluxMode values (dropdown + CSV + summary).
_AGEMODE_LABEL = {
    'imported':    'Imported ages',
    'ratio_J':     'R mode (insert J)',
    'ratio_R_std': 'R mode',
}

# recalculation presets: the shared tables plus a Custom row for the dialogs
_RECALC_LAMBDA_OPTS = LAMBDA_PRESETS + [('Custom…', None, 0.0)]
_RECALC_FC_OPTS     = FC_PRESETS + [('Custom…', None, 0.0, False)]


def _lam_idx(prefix):
    """Index of the first λ preset whose name starts with prefix."""
    return next(i for i, r in enumerate(_RECALC_LAMBDA_OPTS) if r[0].startswith(prefix))


def _fc_idx(prefix):
    """Index of the first FC preset whose name starts with prefix."""
    return next(i for i, r in enumerate(_RECALC_FC_OPTS) if r[0].startswith(prefix))


def _alpha_label(i):
    """Step label for index i: A … Z, AA, AB, …"""
    s = ''
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


# --------------------------------------------------------------------------
# Plateau finder
# --------------------------------------------------------------------------

def find_plateau(steps, cfg):
    """
    Find the largest-by-%39Ar contiguous window of non-excluded steps that
    passes the probability cutoff.

    cfg keys
    --------
    minPct   float   minimum %39Ar in plateau (default 50)
    minN     int     minimum number of steps (default 3)
    pCut     float   probability cutoff in % (default 5)
    errMode  str     'sem_mswd_2s' (default) | 'sem_mswd' | 'sem' (arar_math.scale_sem)
    """
    p_cut      = cfg.get('pCut',  5.0)
    mode       = cfg.get('ageFluxMode', 'imported')
    use_r      = (mode != 'imported')   # ratio-space WM for all R_unk modes

    # J that turns the plateau's mean R into Ma: the one the step ages were
    # computed with.  In R-standard mode, or with a disk-flux J applied, that is
    # the cfg's J; in J mode without a flux every step is aged with its own
    # j_val (_compute_steps_with_age_mode), which can differ from the typed J.
    _J_rd, _lam_rd = None, None
    _J_aged, _J_rel = _aged_J(cfg, steps)
    if use_r:
        _lam_rd = cfg.get('lambdaYr', LAMBDA_DEFAULT)
        _J_rd = _J_aged

    def _r_wm_to_ma(r_raw):
        """Convert a weighted_mean result on ⁴⁰Ar/³⁹Ar R values → age in Ma."""
        if _J_rd is None or _J_rd <= 0 or r_raw['mean'] <= 0:
            return r_raw
        R, sR  = r_raw['mean'], r_raw['sem']
        age_ma = np.log(_J_rd * R + 1.0) / _lam_rd * 1e-6
        sem_ma = _J_rd * sR / (_lam_rd * (_J_rd * R + 1.0)) * 1e-6
        return dict(r_raw, mean=age_ma, sem=sem_ma)

    def _compute_tiers(r, _em, p_cut):
        """Return (s1_ana, s1_int, s1_ext): analytical, internal, external; all 1σ.

        Tier meanings:
          s1_ana: pure weighted-mean SEM, possibly MSWD-scaled (no J contribution)
          s1_int: s1_ana ⊕ ∂t/∂ln J · σ_J/J (monitor / J precision)
          s1_ext: s1_int ⊕ calibration systematic (systematic_components)

        The analytical tier follows the error mode (arar_math.scale_sem).  In
        'sem_mswd' an overdispersed plateau (p below the cutoff) is expanded by
        √MSWD and the 95% Student-t factor t(n−1), stored at the 1σ scale
        (that expansion ÷ 2) so the displayed 2σ is the Isoplot 95% interval.
        """
        s1_ana = scale_sem(r['sem'], r['mswd'], r['prob'], r['n'], _em, p_cut)
        # + monitor / J precision (internal), ∂t/∂ln J · σ_J/J.  Imported ages
        # carry the J error stored with the steps.
        if use_r:
            s1_int = float(internal_sigma_array([r['mean']], [s1_ana], cfg)[0])
        else:
            s1_int = math.hypot(s1_ana, dt_dlnR(r['mean'], cfg.get(
                'lambdaYr', LAMBDA_DEFAULT)) * _J_rel)
        # External: internal ⊕ the calibration systematic
        # (arar_math.systematic_components; anchored chain for Kuiper, Rivera
        # and Phillips).
        s1_ext = float(external_sigma_array(np.array([r['mean']]),
                                            np.array([s1_int]), cfg)[0])
        return float(s1_ana), float(s1_int), float(s1_ext)

    _em = cfg.get('errMode', 'sem_mswd_2s')

    # ── Forced-inclusion override ─────────────────────────────────────────────
    forced_ois = [i for i, s in enumerate(steps)
                  if s.get('status', 'normal') == 'forced']
    if forced_ois:
        win = [steps[i] for i in forced_ois]
        if (use_r and all(s.get('r4039', 0.0) > 0 and s.get('err_r4039', 0.0) > 0
                          for s in win)):
            r = _r_wm_to_ma(weighted_mean([s['r4039']     for s in win],
                                          [s['err_r4039'] for s in win]))
        else:
            r = weighted_mean([s['age'] for s in win], [s['err1s'] for s in win])
        if r is None:
            return None
        pct39 = sum(s['pct39Ar'] for s in win)
        s1_ana, s1_int, s1_ext = _compute_tiers(r, _em, p_cut)
        return dict(ois=forced_ois, pct39=pct39, r=r, n=r['n'],
                    e1s_ana=s1_ana, e1s_int=s1_int, e1s_ext=s1_ext, forced=True,
                    J=_J_rd, J_rel=_J_rel)

    min_pct = cfg.get('minPct', 50.0)
    min_n   = int(cfg.get('minN',  3))

    active = [(i, s) for i, s in enumerate(steps)
              if s.get('status', 'normal') not in ('excluded', 'forced')]
    # R-direct: skip steps that have no valid ratio (e.g. age-only imports)
    if use_r:
        active = [(i, s) for i, s in active
                  if s.get('r4039', 0.0) > 0 and s.get('err_r4039', 0.0) > 0]

    if len(active) < min_n:
        return None

    best = None

    for a in range(len(active)):
        for b in range(a + min_n - 1, len(active)):
            win = active[a:b + 1]

            # Contiguity: between adjacent window members only excluded steps allowed
            ok = True
            for k in range(1, len(win)):
                oi_prev = win[k - 1][0]
                oi_cur  = win[k][0]
                gap = steps[oi_prev + 1: oi_cur]
                if any(g.get('status', 'normal') != 'excluded' for g in gap):
                    ok = False
                    break
            if not ok:
                continue

            pct39 = sum(s['pct39Ar'] for _, s in win)
            if pct39 < min_pct:
                continue

            if use_r:
                r_raw = weighted_mean([s['r4039']     for _, s in win],
                                      [s['err_r4039'] for _, s in win])
                if r_raw is None or r_raw['prob'] * 100.0 < p_cut:
                    continue
                r = _r_wm_to_ma(r_raw)
            else:
                r = weighted_mean([s['age']   for _, s in win],
                                  [s['err1s'] for _, s in win])
                if r is None or r['prob'] * 100.0 < p_cut:
                    continue

            if (best is None
                    or pct39 > best['pct39']
                    or (pct39 == best['pct39'] and len(win) > best['n'])):
                best = dict(ois=[i for i, _ in win],
                            pct39=pct39,
                            r=r,
                            n=len(win))

    if best is None:
        return None

    r = best['r']
    s1_ana, s1_int, s1_ext = _compute_tiers(r, _em, p_cut)
    best['e1s_ana'] = s1_ana
    best['e1s_int'] = s1_int
    best['e1s_ext'] = s1_ext
    best['J'] = _J_rd
    best['J_rel'] = _J_rel
    return best


# --------------------------------------------------------------------------
# Ca/K axis helper
# --------------------------------------------------------------------------

def _kca_steps(steps):
    """The steps with the ratio panel switched to K/Ca = 1/(Ca/K).

    A step with Ca/K <= 0 has no K/Ca and is not drawn.  The error box is the
    1/x image of the Ca/K interval, so it is asymmetric and runs off the top of
    the panel when the Ca/K interval reaches zero; the original Ca/K and its
    error are kept in '_kca' for that.  'err_cak' holds the linear error
    sigma/(Ca/K)^2, used only to scale the axis."""
    out = []
    for s0 in steps:
        s1 = dict(s0)
        c = s0.get('caK', 0.0)
        e = s0.get('err_cak', 0.0) or 0.0
        if c is not None and math.isfinite(c) and c > 0:
            s1['caK'] = 1.0 / c
            s1['err_cak'] = e / (c * c)
            s1['_kca'] = (c, e)
        else:
            s1['caK'] = float('nan')
            s1['err_cak'] = 0.0
        out.append(s1)
    return out


def _box_bounds(s, val_key, err_key, scale, fixed_half, top):
    """Lower and upper edge of a step's box in a spectrum panel; None when the
    step has no value there (K/Ca of a step with Ca/K <= 0)."""
    val = s[val_key]
    if val is None or not math.isfinite(val):
        return None
    if val_key == 'caK' and '_kca' in s:
        c, e = s['_kca']
        ek = e * scale
        if ek <= 0:
            return val - fixed_half, val + fixed_half
        return 1.0 / (c + ek), (1.0 / (c - ek) if c > ek else top)
    ev = (s.get(err_key, 0.0) if err_key else 0.0) * scale
    eh = ev if ev > 0 else fixed_half
    return val - eh, val + eh


def _cak_ylim(cak_vals, err_cak_vals=None, padding=0.15, gas_pct=None):
    """Ca/K y-axis range that keeps the steps in view.

    Every step counts, plateau or not.  Only a step far outside the rest
    (beyond 3 x IQR) is left off the scale, so one very high Ca/K step does
    not flatten the others.  Error bars widen the range, but no single error
    can stretch it past the spread of the step values.  The axis starts at
    zero unless all steps sit well above it.  gas_pct (the % 39Ar of each step)
    lets a step holding most of the gas stay in view however far out it is;
    only the thin steps are ever trimmed.
    """
    vals = np.asarray([v for v in (cak_vals or []) if v is not None], dtype=float)
    gas = (np.asarray([g if g is not None else 0.0 for v, g in zip(cak_vals, gas_pct)
                       if v is not None], dtype=float)
           if gas_pct is not None and cak_vals else None)
    if err_cak_vals is not None and cak_vals:
        errs = np.asarray([e if e is not None else 0.0 for v, e in
                           zip(cak_vals, err_cak_vals) if v is not None], dtype=float)
    else:
        errs = np.zeros(len(vals))
    good = np.isfinite(vals)
    vals, errs = vals[good], errs[good]
    if gas is not None:
        gas = gas[good]
    if len(vals) == 0:
        return (0.0, 10.0)
    errs = np.where(np.isfinite(errs) & (errs > 0), errs, 0.0)
    n = len(vals)

    keep = np.ones(n, dtype=bool)
    if n >= 4:
        # a thin step (under 5 % of the gas) far outside the rest (3 x IQR) is
        # left off the scale so it cannot flatten the others; a step holding
        # more gas than that always stays in view
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = max(q3 - q1, 0.25 * abs(float(np.median(vals))))
        far = (vals < q1 - 3.0 * iqr) | (vals > q3 + 3.0 * iqr)
        thin = (gas < 5.0) if gas is not None else np.ones(n, dtype=bool)
        keep = ~(far & thin)
        if keep.sum() < 2:
            keep[:] = True
    v, e = vals[keep], errs[keep]

    med = float(np.median(v))
    spread = float(v.max() - v.min())
    # error bars widen the range, but no single one may run past the spread
    # of the step values themselves
    e = np.minimum(e, max(spread, 0.15 * abs(med)))
    lo = float((v - e).min())
    hi = float((v + e).max())
    d = hi - lo

    # Minimum span: 15% of the median value, with a 2-unit floor for Ca/K-sized
    # numbers only (K/Ca is ~0.03, where a 2-unit floor would flatten the panel)
    min_span = max(abs(med) * 0.15, 2.0 if abs(med) >= 1.0 else abs(med) * 0.15)
    if min_span <= 0:
        min_span = 2.0 if hi <= 0 else 0.15 * abs(hi)
    if d < min_span:
        mid = (lo + hi) / 2.0
        lo  = mid - min_span / 2.0
        hi  = mid + min_span / 2.0
        d   = min_span

    if lo > 0 and lo <= 0.5 * hi:
        lo = 0.0
        d = hi
    elif lo < 0:
        # a step (or its error) below zero is real: the low-Ca samples scatter
        # about zero.  Keep it in view, but never more than the span above 0.
        lo = max(lo, -hi)
        d = hi - lo
        return (lo - d * padding, hi + d * padding)
    return (max(0.0, lo - d * padding), hi + d * padding)


# Top of the %40Ar* axis set in the zoom row; None = automatic (_p40_ylim).
_P40_MAX = None


def _p40_ylim(pct_vals):
    """%40Ar* axis: 0 to the top set in the zoom row, else 0-100, higher when
    a step is above 100 %.  Steps above a set top are cut off at it."""
    if _P40_MAX:
        return (0.0, float(_P40_MAX))
    v = [x for x in pct_vals if x is not None and np.isfinite(x)]
    top = max(v) if v else 100.0
    if top <= 100.0:
        return (0.0, 100.0)
    return (0.0, 5.0 * float(np.ceil((top + 1.0) / 5.0)))


# --------------------------------------------------------------------------
# Cumulative %39Ar
# --------------------------------------------------------------------------

def add_cumulative(steps):
    total = sum(s['pct39Ar'] for s in steps) or 100.0
    scale = 100.0 / total
    c = 0.0
    out = []
    for s in steps:
        cs = c
        c += s['pct39Ar'] * scale
        out.append({**s, 'cs': cs, 'ce': c})
    return out


# --------------------------------------------------------------------------
# Column auto-detection
# --------------------------------------------------------------------------

# Regex patterns tried against lower-cased header strings.
# More specific patterns are listed first to win on priority.
_COL_PATTERNS = {
    'label':    [r'^step$', r'^label$', r'^name$', r'^id$', r'^#$', r'^no\.?$', r'step'],
    'power':    [r'power', r'pwr', r'watt', r'laser.*w', r'temp', r'°c', r'\bw\b'],
    'age':      [r'apparent.?age', r'\bage\s*\(ma\)', r'\bage\b'],
    'err':    [r'[±±].*[12]', r'[12][sσ]\b', r'[12]se\b', r'\berr', r'uncert', r'±'],
    'pct39ar':  [r'%.*39', r'39.*%', r'39ar.*\(k\)', r'39ar\b', r'ar39\b', r'cumul.*39', r'frac.*39'],
    'cak':      [r'ca\s*/\s*k', r'\bcak\b', r'ca_k', r'k\s*/\s*ca', r'\bkca\b', r'k_ca'],
    'pct40ar':  [r'%.*40.*\*', r'40ar\*', r'radiog', r'40ar.*\(r\)', r'%.*40', r'40ar'],
    'r4039':    [r'40ar\*\s*/\s*39ar', r'40\s*ar\s*\*\s*/\s*39', r'40\s*\(r\)\s*/\s*39\s*\(k\)', r'40.*39.*ratio', r'r\s*40/39', r'40/39\b', r'\br4039\b', r'ratio.*40.*39'],
    'err_r4039':[r'err.*r?\s*40.*39', r'2s.*r?\s*40.*39', r'40.*39.*err', r'±.*40.*39', r'40/39.*err'],
    'err_cak':  [r'err.*ca.*k', r'err.*cak', r'2s.*ca.*k', r'ca.*k.*err', r'2s.*ca/k', r'ca/k.*2s'],
    'j_val':    [r'\bj[- _]?val\b', r'\bj[- _]?factor\b', r'^j$', r'\bj\s*='],
    'j_err':    [r'err.*\bj\b', r'\bj\b.*err', r'1[sσ].*\bj\b', r'\bj\b.*1[sσ]', r'±.*\bj\b'],
    # Isochron columns, specific enough not to clash with the ratio/age columns above
    'iso_nri':  [r'normal\s+isochron\s+ri', r'normal.*isochron.*ri\b'],
    'iso_iri':  [r'inverse\s+isochron\s+ri', r'inverse.*isochron.*ri\b'],
    'iso_nx':   [r'ar39.*\(k\).*ar40|39\s*\(k\)\s*/\s*40'],       # 39(k)/40: normal X
    'iso_ny':   [r'ar36.*\(a\).*ar40|36\s*\(a\)\s*/\s*40'],       # 36(a)/40: normal Y
    'iso_ix':   [r'ar39.*\(k\).*ar36|39\s*\(k\)\s*/\s*36'],       # 39(k)/36: inverse X
    'iso_iy':   [r'ar40.*\(a\+r\).*ar36|40.*\(a\+r\)\s*/\s*36'], # 40(a+r)/36: inverse Y
}

COLUMN_LABELS = {
    'label':    'Label / Step',
    'power':    'Power / Temp',
    'age':      'Age (Ma)',
    'err':    '± Error (Ma)',
    'pct39ar':  '%³⁹Ar',
    'cak':      'Ca/K',
    'err_cak':  '± Error (Ca/K)',
    'pct40ar':  '%⁴⁰Ar*',
    'r4039':    '⁴⁰Ar/³⁹Ar Ratio',
    'err_r4039':'± Error (ratio)',
    'j_val':    'J value',
    'j_err':    '±1σ (J)',
    'iso_nri':  'Normal Isochron RI',
    'iso_iri':  'Inverse Isochron RI',
    'iso_nx':   '³⁹Ar(k)/⁴⁰Ar (norm. X)',
    'iso_nsx':  '±1σ (norm. X)',
    'iso_ny':   '³⁶Ar/⁴⁰Ar (norm. Y)',
    'iso_nsy':  '±1σ (norm. Y)',
    'iso_ix':   '³⁹Ar(k)/³⁶Ar (inv. X)',
    'iso_isx':  '±1σ (inv. X)',
    'iso_iy':   '⁴⁰Ar/³⁶Ar (inv. Y)',
    'iso_isy':  '±1σ (inv. Y)',
}


def detect_columns(headers):
    detected = {}
    scores   = {}

    for ci, raw in enumerate(headers):
        h = str(raw).lower().strip()
        for key, patterns in _COL_PATTERNS.items():
            for rank, pat in enumerate(patterns):
                if re.search(pat, h):
                    # Higher rank = higher priority (lower rank index = better match)
                    score = len(patterns) - rank
                    if key not in scores or score > scores[key][1]:
                        scores[key] = (ci, score)
                    break   # first matching pattern for this key

    for key, (ci, _) in scores.items():
        detected[key] = ci

    _ERR_RE = re.compile(r'[12][sσ]\b|±|\berr\b|uncert')

    def _next_col_claim(base_key, err_key):
        if base_key not in detected:
            return
        next_ci = detected[base_key] + 1
        if next_ci >= len(headers):
            return
        if _ERR_RE.search(str(headers[next_ci]).lower().strip()):
            detected[err_key] = next_ci
            if detected.get('err') == next_ci and err_key != 'err':
                del detected['err']

    _next_col_claim('r4039', 'err_r4039')
    _next_col_claim('cak', 'err_cak')
    _next_col_claim('j_val', 'j_err')
    _next_col_claim('iso_nx', 'iso_nsx')
    _next_col_claim('iso_ny', 'iso_nsy')
    _next_col_claim('iso_ix', 'iso_isx')
    _next_col_claim('iso_iy', 'iso_isy')

    # Age error: search forward from age column whenever the error column is missing
    # (-1 < any valid ci) or landed before the age column (grabbed a ratio 2s).
    if 'age' in detected:
        age_ci = detected['age']
        if detected.get('err', -1) < age_ci:
            claimed = set(detected.values())
            for ci in range(age_ci + 1, len(headers)):
                if ci in claimed:
                    continue
                h = str(headers[ci]).lower().strip()
                for pat in _COL_PATTERNS['err']:
                    if re.search(pat, h):
                        detected['err'] = ci
                        break
                else:
                    continue
                break
            # Fallback: if still undetected, assume the column immediately
            # after Age is the error (common convention: Age | ±err | %39Ar).
            if 'err' not in detected:
                next_ci = age_ci + 1
                if next_ci < len(headers) and next_ci not in set(detected.values()):
                    detected['err'] = next_ci

    return detected


def integrated_age(steps_c, lam):
    """Total-gas (integrated) age: the age the experiment would give if all
    of its gas had been released in one step, from the ³⁹Ar-weighted mean
    ⁴⁰Ar*/³⁹Ar of every step, R_tot = Σ fᵢRᵢ / Σ fᵢ (fᵢ = %³⁹Ar).

    Written in ages, J·Rᵢ = e^{λtᵢ} − 1, so it needs no J and holds in every
    age mode (also Imported ages):
        t = ln(1 + Σ fᵢ(e^{λtᵢ} − 1) / Σ fᵢ) / λ
    with the analytical 1σ from the step errors.  Every step counts, also the
    ones excluded from the plateau.  Returns dict(mean, sem, n) or None."""
    st = [s for s in steps_c if s.get('pct39Ar', 0.0) > 0 and s.get('age') is not None]
    if not st or not lam or lam <= 0:
        return None
    f = np.array([s['pct39Ar'] for s in st], float)
    t = np.array([s['age'] for s in st], float) * 1e6
    e = np.array([s.get('err1s', 0.0) for s in st], float) * 1e6
    f = f / f.sum()
    g = np.exp(lam * t)
    X = float((f * (g - 1.0)).sum())
    if X <= -1.0:
        return None
    mean = math.log1p(X) / lam
    sem = math.sqrt(float(((f * g * e) ** 2).sum())) / (1.0 + X)
    return dict(mean=mean / 1e6, sem=sem / 1e6, n=len(st))


def _recalc_ages_in_steps(steps, lam1, fc1_ma, j1_default, lam2, fc2_ma,
                           fc2_err_ma=0.0, ing1=None, ing2=None):
    """Convert step ages from (lam1, fc1_ma) to (lam2, fc2_ma).

    Uses the per-step r4039 ratio when available; otherwise back-calculates R
    from the stored age + J.  All uncertainties (err1s, err_r4039) are 1σ.

    Only the age and its analytical uncertainty are converted; err1s stays
    analytical.  The FC-age and λ errors are shared by every step, and err1s
    is the weight in the plateau and integrated means: a systematic that
    grows with R put into it would change the weights, and so the mean age,
    with every change of calibration.  The external tier is added once, to
    the plateau age (find_plateau._compute_tiers → external_sigma_array).

    fc2_err_ma only selects the anchored chain for the new FC age (fc_ingrowth);
    it does not enter err1s.  ing1 and ing2, when given, are the monitor
    ingrowths e^{λ t_M} − 1 of the old and new calibration (arar_math.
    monitor_ingrowth), which include the route from the monitor; they replace
    the two FC ingrowths.
    """
    _MAY = 1e6
    # monitor ingrowth of the old and new calibrations (anchored chain where
    # the calibration has an orbital anchor), so J2 = J1 · denom2 / denom1
    denom1 = ing1 if ing1 is not None else fc_ingrowth(fc1_ma, lam1)
    denom2 = (ing2 if ing2 is not None
              else fc_ingrowth(fc2_ma, lam2, fc2_err_ma if fc2_err_ma else None))
    if denom1 <= 0:
        return steps
    fc_ratio  = denom2 / denom1
    for s in steps:
        j1 = s.get('j_val') or j1_default
        if not j1:
            continue
        j2 = j1 * fc_ratio
        R  = s.get('r4039', 0.0)
        sR = s.get('err_r4039', 0.0)          # already 1σ
        r_measured = R > 0
        if not r_measured:
            age_yr  = s['age'] * _MAY
            err1_yr = s['err1s'] * _MAY        # already 1σ (analytical)
            R  = (math.exp(lam1 * age_yr) - 1.0) / j1
            sR = err1_yr * lam1 * math.exp(lam1 * age_yr) / j1 if err1_yr > 0 else 0.0
        new_age_yr = math.log(1.0 + j2 * R) / lam2
        C = 1.0 + j2 * R
        # Analytical uncertainty only, converted to the new J; the FC-age and
        # λ errors are the external tier, added at plateau level (docstring).
        new_err_total = j2 * sR / (lam2 * C) if sR > 0 else 0.0
        s['age']   = new_age_yr / _MAY
        s['err1s'] = new_err_total / _MAY     # 1σ, analytical only
        s['j_val'] = j2                             # update J to new calibration
        if s.get('j_err', 0.0) > 0:                # relative J error is preserved
            s['j_err'] = s['j_err'] * fc_ratio
        if not r_measured:
            # R was back-calculated; store a self-consistent R for ratio_J mode.
            # err_r4039 is kept at 1σ, matching new_err_total and err1s.
            exp_t2 = math.exp(lam2 * new_age_yr)
            s['r4039']     = (exp_t2 - 1.0) / j2
            s['err_r4039'] = lam2 * exp_t2 / j2 * new_err_total if new_age_yr > 0 else 0.0
    return steps


# --------------------------------------------------------------------------
# Column mapping dialog
# --------------------------------------------------------------------------

class ColumnMappingDialog(QDialog):

    def __init__(self, headers, detected, parent=None, default_sigma=2,
                 j_default=None, j_err_default=None):
        super().__init__(parent)
        self._j_default     = j_default
        self._j_err_default = j_err_default
        self.setWindowTitle("Column Mapping: Review Auto-Detection")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)

        info = QLabel(
            "Auto-detected column assignments are shown below.\n"
            "Change any drop-down if the detection is wrong.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; font-size: 10px; margin-bottom: 6px;")
        layout.addWidget(info)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self._combos      = {}
        self._auto_label  = False
        self._btn_inv_cak = None

        def _disp(raw, idx):
            s = str(raw).strip()
            if not s or re.match(r'Unnamed:\s*\d+', s, re.IGNORECASE):
                return f'(col {idx})'
            return s
        none_plus = ['(none)'] + [_disp(h, i) for i, h in enumerate(headers)]

        for key, label in COLUMN_LABELS.items():
            cb = QComboBox()
            cb.addItems(none_plus)
            if key in detected:
                cb.setCurrentIndex(detected[key] + 1)   # +1 because "(none)" is index 0

            if key == 'label':
                row_w = QWidget()
                row_l = QHBoxLayout(row_w)
                row_l.setContentsMargins(0, 0, 0, 0)
                row_l.setSpacing(4)
                row_l.addWidget(cb, stretch=1)
                btn_al = QPushButton("Auto-label")
                btn_al.setFixedWidth(82)
                btn_al.setToolTip("Auto-generate A, B, C… labels, ignores the label column")
                def _on_auto_label(_, _cb=cb):
                    _cb.setCurrentIndex(0)
                    self._auto_label = True
                btn_al.clicked.connect(_on_auto_label)
                row_l.addWidget(btn_al)
                form.addRow(label + ':', row_w)
            elif key == 'cak':
                row_w = QWidget()
                row_l = QHBoxLayout(row_w)
                row_l.setContentsMargins(0, 0, 0, 0)
                row_l.setSpacing(4)
                row_l.addWidget(cb, stretch=1)
                self._btn_inv_cak = QPushButton("K/Ca → invert")
                self._btn_inv_cak.setCheckable(True)
                self._btn_inv_cak.setFixedWidth(95)
                self._btn_inv_cak.setToolTip(
                    "Column contains K/Ca, converts to Ca/K = 1/(K/Ca) on import")
                row_l.addWidget(self._btn_inv_cak)
                form.addRow(label + ':', row_w)
            else:
                form.addRow(label + ':', cb)

            self._combos[key] = cb

        layout.addLayout(form)

        sep_sig = QFrame()
        sep_sig.setFrameShape(QFrame.HLine)
        sep_sig.setStyleSheet("color: #ddd; margin-top: 6px;")
        layout.addWidget(sep_sig)

        sig_row = QHBoxLayout()
        sig_lbl = QLabel("Error columns are:")
        sig_lbl.setStyleSheet("font-size: 10px;")
        sig_row.addWidget(sig_lbl)
        self._rb_1s = QRadioButton("1σ")
        self._rb_2s = QRadioButton("2σ")
        self._rb_1s.setStyleSheet("font-size: 10px;")
        self._rb_2s.setStyleSheet("font-size: 10px;")
        self._sig_group = QButtonGroup(self)
        self._sig_group.addButton(self._rb_1s, 1)
        self._sig_group.addButton(self._rb_2s, 2)
        (self._rb_1s if default_sigma == 1 else self._rb_2s).setChecked(True)
        sig_row.addWidget(self._rb_1s)
        sig_row.addWidget(self._rb_2s)
        sig_row.addStretch()
        layout.addLayout(sig_row)

        self._recalc_params = None
        recalc_row = QHBoxLayout()
        self._btn_recalc = QPushButton("Recalculate ages…")
        self._btn_recalc.setToolTip(
            "Convert ages to a different decay constant / flux monitor standard")
        self._btn_recalc.clicked.connect(self._open_recalc)
        self._lbl_recalc = QLabel("")
        self._lbl_recalc.setStyleSheet("font-size: 9px; color: #1565c0;")
        recalc_row.addWidget(self._btn_recalc)
        recalc_row.addWidget(self._lbl_recalc, stretch=1)
        layout.addLayout(recalc_row)

        # ── Compute-from-R section ───────────────────────────────────────────
        # Visible only when both R (r4039) and J (j_val) columns are detected.
        self._cb_compute_from_r   = None
        self._compute_lam_le      = None
        self._compute_lam_err_val = LAMBDA_ERR
        self._compute_fc_sp       = None
        self._compute_fc_err_sp   = None
        self._compute_fc_ext      = True
        if 'r4039' in detected and 'j_val' in detected:
            sep_r = QFrame()
            sep_r.setFrameShape(QFrame.HLine)
            sep_r.setStyleSheet("color: #ddd; margin-top: 4px;")
            layout.addWidget(sep_r)

            self._cb_compute_from_r = QCheckBox(
                "Compute ages from R and J  (R mode with J; sets λ, FC age and R_std)")
            self._cb_compute_from_r.setChecked(True)
            self._cb_compute_from_r.setToolTip(
                "After pasting, the age source is R mode (insert J), with\n"
                "R_std = (exp(λ·t_FC) − 1) / J from the λ and FC age set below.\n"
                "The ages follow when the calibration is changed later.")
            layout.addWidget(self._cb_compute_from_r)

            _cfr_form = QFormLayout()
            _cfr_form.setLabelAlignment(Qt.AlignRight)
            _cfr_form.setContentsMargins(20, 2, 0, 2)
            _cfr_form.setHorizontalSpacing(6)

            # ── λ row (all presets + Custom) ──────────────────────────────────
            _lam_opts = _RECALC_LAMBDA_OPTS   # includes Custom at [-1]
            _compute_lam_cb = QComboBox()
            _compute_lam_cb.addItems([x[0] for x in _lam_opts])
            _default_lam_idx = _lam_idx(DEFAULT_LAMBDA_NAME)
            _compute_lam_cb.setCurrentIndex(_default_lam_idx)

            self._compute_lam_le = QLineEdit()
            self._compute_lam_le.setMaximumWidth(120)
            self._compute_lam_le.setPlaceholderText("e.g. 5.543e-10")
            self._compute_lam_le.setReadOnly(True)
            self._compute_lam_le.setText(f"{_lam_opts[_default_lam_idx][1]:.6e}")
            self._compute_lam_err_val = _lam_opts[_default_lam_idx][2]

            def _on_compute_lam(idx, _le=self._compute_lam_le,
                                _opts=_lam_opts):
                row = _opts[idx]
                if row[1] is None:        # Custom
                    _le.setReadOnly(False)
                    _le.clear()
                    _le.setFocus()
                    self._compute_lam_err_val = 0.0
                else:
                    _le.setReadOnly(True)
                    _le.setText(f"{row[1]:.6e}")
                    self._compute_lam_err_val = row[2]
            _compute_lam_cb.currentIndexChanged.connect(_on_compute_lam)

            _lam_row_w = QWidget()
            _lam_row_l = QHBoxLayout(_lam_row_w)
            _lam_row_l.setContentsMargins(0, 0, 0, 0)
            _lam_row_l.setSpacing(4)
            _lam_row_l.addWidget(_compute_lam_cb)
            _lam_row_l.addWidget(self._compute_lam_le)
            _lam_row_l.addWidget(QLabel("yr⁻¹"))
            _lam_row_l.addStretch()
            _cfr_form.addRow("λ:", _lam_row_w)

            # ── FC monitor row (all presets + Custom, short labels) ───────────
            def _fc_short(full):
                # Return author-year portion before " — ", or full label
                if ' — ' in full:
                    return full.split(' — ')[0]
                return full

            _fc_opts = _RECALC_FC_OPTS             # includes Custom at [-1]
            _compute_fc_cb = QComboBox()
            _compute_fc_cb.addItems([_fc_short(x[0]) for x in _fc_opts])
            _default_fc_idx = _fc_idx(DEFAULT_FC_NAME)
            _compute_fc_cb.setCurrentIndex(_default_fc_idx)
            _fcinit = _fc_opts[_default_fc_idx]

            self._compute_fc_sp = QDoubleSpinBox()
            self._compute_fc_sp.setDecimals(3)
            self._compute_fc_sp.setRange(0.001, 99999.0)
            self._compute_fc_sp.setValue(_fcinit[1])
            self._compute_fc_sp.setEnabled(False)
            self._compute_fc_sp.setMaximumWidth(90)

            self._compute_fc_err_sp = QDoubleSpinBox()
            self._compute_fc_err_sp.setDecimals(4)
            self._compute_fc_err_sp.setRange(0.0, 99.0)
            self._compute_fc_err_sp.setValue(_fcinit[2])
            self._compute_fc_err_sp.setEnabled(False)
            self._compute_fc_err_sp.setMaximumWidth(75)
            self._compute_fc_ext = _fcinit[3]

            def _on_compute_fc(idx, _sp=self._compute_fc_sp,
                               _err=self._compute_fc_err_sp,
                               _opts=_fc_opts):
                row = _opts[idx]
                is_custom = (row[1] is None)
                _sp.setEnabled(is_custom)
                _err.setEnabled(is_custom)
                if not is_custom:
                    _sp.setValue(row[1])
                    _err.setValue(row[2])
                    self._compute_fc_ext = row[3]
                    # follow the λ this calibration is anchored to
                    li = paired_lambda_index(row[0])
                    if li is not None:
                        _compute_lam_cb.setCurrentIndex(li)
                else:
                    self._compute_fc_ext = False
            _compute_fc_cb.currentIndexChanged.connect(_on_compute_fc)

            _fc_row_w = QWidget()
            _fc_row_l = QHBoxLayout(_fc_row_w)
            _fc_row_l.setContentsMargins(0, 0, 0, 0)
            _fc_row_l.setSpacing(4)
            _fc_row_l.addWidget(_compute_fc_cb)
            _fc_row_l.addWidget(self._compute_fc_sp)
            _fc_row_l.addWidget(QLabel("±"))
            _fc_row_l.addWidget(self._compute_fc_err_sp)
            _fc_row_l.addWidget(QLabel("Ma"))
            _fc_row_l.addStretch()
            _cfr_form.addRow("FC monitor:", _fc_row_w)

            layout.addLayout(_cfr_form)

            def _toggle_compute_rows(checked,
                                     _lam=_compute_lam_cb, _fc=_compute_fc_cb):
                _lam.setEnabled(checked)
                _fc.setEnabled(checked)
                if checked:
                    _on_compute_lam(_lam.currentIndex())
                    _on_compute_fc(_fc.currentIndex())
                else:
                    if self._compute_lam_le:
                        self._compute_lam_le.setReadOnly(True)
                    if self._compute_fc_sp:
                        self._compute_fc_sp.setEnabled(False)
                    if self._compute_fc_err_sp:
                        self._compute_fc_err_sp.setEnabled(False)
            self._cb_compute_from_r.toggled.connect(_toggle_compute_rows)

            _note = QLabel("R_std auto-derived from J and FC age · ages update live")
            _note.setStyleSheet("font-size: 9px; color: #888; margin-left: 20px;")
            layout.addWidget(_note)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_mapping(self):
        m = {}
        for key, cb in self._combos.items():
            idx = cb.currentIndex()
            if idx > 0:
                m[key] = idx - 1
        return m

    @property
    def auto_label(self):
        return self._auto_label

    @property
    def err_sigma(self):
        return self._sig_group.checkedId()  # 1 or 2

    @property
    def invert_cak(self):
        return self._btn_inv_cak is not None and self._btn_inv_cak.isChecked()

    @property
    def recalc_params(self):
        return self._recalc_params

    @property
    def compute_from_r(self):
        return (self._cb_compute_from_r is not None and
                self._cb_compute_from_r.isChecked())

    @property
    def compute_lambda(self):
        if self._compute_lam_le is not None:
            try:
                return float(self._compute_lam_le.text())
            except ValueError:
                pass
        return LAMBDA_DEFAULT

    @property
    def compute_lam_err(self):
        return self._compute_lam_err_val

    @property
    def compute_fc_ma(self):
        return self._compute_fc_sp.value() if self._compute_fc_sp is not None else STD_AGE_DEFAULT

    @property
    def compute_fc_err_ma(self):
        return self._compute_fc_err_sp.value() if self._compute_fc_err_sp is not None else STD_AGE_ERR

    @property
    def compute_fc_external(self):
        return self._compute_fc_ext

    def _open_recalc(self):
        mapping = self.get_mapping()
        has_ratio = 'r4039' in mapping
        has_j     = 'j_val' in mapping
        dlg = AgeRecalcDialog(has_ratio=has_ratio, has_j=has_j,
                              j_default=self._j_default,
                              j_err_default=self._j_err_default, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._recalc_params = dlg.get_params()
            p = self._recalc_params
            orig = f"{p['lam1']*1e10:.4f}×10⁻¹⁰, FC {p['fc1_ma']:.3f} Ma"
            tgt  = f"{p['lam2']*1e10:.4f}×10⁻¹⁰, FC {p['fc2_ma']:.3f} Ma"
            self._lbl_recalc.setText(f"✓  {orig}  →  {tgt}")
        else:
            self._recalc_params = None
            self._lbl_recalc.setText("")


# --------------------------------------------------------------------------
# Age recalculation dialog
# --------------------------------------------------------------------------

class AgeRecalcDialog(QDialog):

    def __init__(self, has_ratio=False, has_j=False, j_default=None,
                 j_err_default=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recalculate Ages: Decay Constant / FC Conversion")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)

        def _make_param_group(title, lam_default_idx, fc_default_idx, show_err_ext=False):
            box = QWidget()
            bl  = QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(f"<b>{title}</b>")
            bl.addWidget(lbl)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignRight)

            lam_cb = QComboBox()
            lam_cb.addItems([x[0] for x in _RECALC_LAMBDA_OPTS])
            lam_cb.setCurrentIndex(lam_default_idx)
            lam_le = QLineEdit()
            lam_le.setPlaceholderText("e.g. 5.543e-10")
            lam_le.setEnabled(lam_default_idx == len(_RECALC_LAMBDA_OPTS) - 1)
            lam_le.setText(f"{_RECALC_LAMBDA_OPTS[lam_default_idx][1] or 5.543e-10:.6e}")

            def _lam_changed(idx, _le=lam_le):
                val = _RECALC_LAMBDA_OPTS[idx][1]
                if val is None:
                    _le.setEnabled(True)
                    _le.setFocus()
                else:
                    _le.setEnabled(False)
                    _le.setText(f"{val:.6e}")
            lam_cb.currentIndexChanged.connect(_lam_changed)

            fc_cb = QComboBox()
            fc_cb.addItems([x[0] for x in _RECALC_FC_OPTS])
            fc_cb.setCurrentIndex(fc_default_idx)
            fc_sp = QDoubleSpinBox()
            fc_sp.setDecimals(3)
            fc_sp.setRange(0.001, 99999.0)
            _fc_init = _RECALC_FC_OPTS[fc_default_idx]
            fc_sp.setValue(_fc_init[1] or 28.201)
            _custom_fc_idx = len(_RECALC_FC_OPTS) - 1
            fc_sp.setEnabled(fc_default_idx == _custom_fc_idx)

            fc_err_sp = None
            fc_ext_cb = None
            if show_err_ext:
                fc_err_sp = QDoubleSpinBox()
                fc_err_sp.setDecimals(4)
                fc_err_sp.setRange(0.0, 99.0)
                fc_err_sp.setSuffix(" Ma")
                fc_err_sp.setValue(_fc_init[2])
                fc_err_sp.setEnabled(fc_default_idx == _custom_fc_idx)

                fc_ext_cb = QCheckBox("External calibration  (FC age encodes λ uncertainty)")
                fc_ext_cb.setChecked(_fc_init[3])
                fc_ext_cb.setEnabled(fc_default_idx == _custom_fc_idx)
                fc_ext_cb.setToolTip(
                    "On for calibrations that fix how λ enters: the astronomically\n"
                    "anchored ages (Kuiper 2008, Rivera 2011, Phillips 2022) and the\n"
                    "joint fit of Carter et al. (2025), which comes with its own λ\n"
                    "covariance.\n"
                    "Off when the FC age does not carry λ (e.g. Renne 2010/2011 used\n"
                    "as a single value), so the full σ_λ is added.\n\n"
                    "It only sets how the calibration enters the external tier of\n"
                    "the plateau age; the step errors stay analytical."
                )

            def _fc_changed(idx, _sp=fc_sp, _err=fc_err_sp, _ext=fc_ext_cb,
                            _lam_cb=lam_cb):
                row = _RECALC_FC_OPTS[idx]
                is_custom = (idx == _custom_fc_idx)
                if is_custom:
                    _sp.setEnabled(True); _sp.setFocus()
                    if _err is not None: _err.setEnabled(True)
                    if _ext is not None: _ext.setEnabled(True)
                else:
                    _sp.setEnabled(False); _sp.setValue(row[1])
                    if _err is not None: _err.setEnabled(False); _err.setValue(row[2])
                    if _ext is not None: _ext.setEnabled(False); _ext.setChecked(row[3])
                    # follow the λ this calibration is anchored to
                    li = paired_lambda_index(row[0])
                    if li is not None:
                        _lam_cb.setCurrentIndex(li)
            fc_cb.currentIndexChanged.connect(_fc_changed)

            form.addRow("Decay constant:", lam_cb)
            form.addRow("λ (yr⁻¹):",      lam_le)
            form.addRow("Flux monitor:",   fc_cb)
            form.addRow("FC age (Ma):",    fc_sp)
            if show_err_ext:
                form.addRow("FC age ±1σ:", fc_err_sp)
                form.addRow("",            fc_ext_cb)
            bl.addLayout(form)
            return box, lam_cb, lam_le, fc_cb, fc_sp, fc_err_sp, fc_ext_cb

        orig_box, self._o_lam_cb, self._o_lam_le, self._o_fc_cb, self._o_fc_sp, _, _ = \
            _make_param_group("Original parameters  (used to compute the published ages)",
                              _lam_idx('Min'), _fc_idx('Kuiper'))
        tgt_box, self._t_lam_cb, self._t_lam_le, self._t_fc_cb, self._t_fc_sp, \
            self._t_fc_err_sp, self._t_fc_ext_cb = \
            _make_param_group("Target parameters  (convert to)",
                              _lam_idx(DEFAULT_LAMBDA_NAME), _fc_idx(DEFAULT_FC_NAME),
                              show_err_ext=True)

        layout.addWidget(orig_box)

        # J value, always editable; auto-populated from data when available
        self._j_row = QWidget()
        j_lay = QFormLayout(self._j_row)
        j_lay.setLabelAlignment(Qt.AlignRight)
        self._j_le    = QLineEdit()
        self._j_le.setPlaceholderText("e.g. 7.234e-3")
        if j_default is not None:
            self._j_le.setText(f"{j_default:.10g}")
        else:
            self._j_le.setText("1e-3")
        self._j_err_le = QLineEdit(
            f"{j_err_default:.6e}" if j_err_default is not None else "1.6e-7"
        )
        j_unit = QLabel("(1σ)")
        j_unit.setStyleSheet("font-size: 9px; color: #666;")
        j_val_row = QWidget()
        jvr = QHBoxLayout(j_val_row)
        jvr.setContentsMargins(0,0,0,0)
        jvr.addWidget(self._j_le, stretch=2)
        jvr.addWidget(QLabel("±"))
        jvr.addWidget(self._j_err_le, stretch=2)
        jvr.addWidget(j_unit)
        if has_j:
            note_txt = "J column mapped, per-step values used; edit here to override all steps."
        elif has_ratio:
            note_txt = "⁴⁰Ar*/³⁹Ar column mapped, J used for error propagation and back-check."
        else:
            note_txt = "Required: no ratio or J column mapped."
        j_note = QLabel(note_txt)
        j_note.setStyleSheet("font-size: 9px; color: #888;")
        j_note.setWordWrap(True)
        j_lay.addRow("J value (1σ):", j_val_row)
        j_lay.addRow("", j_note)
        layout.addWidget(self._j_row)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ddd;")
        layout.addWidget(sep)
        layout.addWidget(tgt_box)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _parse_lambda(self, le):
        try:
            v = float(le.text().strip())
            if v > 0:
                return v
        except ValueError:
            pass
        return None

    def _validate_and_accept(self):
        if self._parse_lambda(self._o_lam_le) is None:
            QMessageBox.warning(self, "Invalid", "Original λ value is not valid.")
            return
        if self._parse_lambda(self._t_lam_le) is None:
            QMessageBox.warning(self, "Invalid", "Target λ value is not valid.")
            return
        self.accept()

    def get_params(self):
        j1 = 0.0; j1_err = 0.0
        try: j1 = float(self._j_le.text())
        except ValueError: pass
        try: j1_err = float(self._j_err_le.text())
        except ValueError: pass
        lam2_err = _RECALC_LAMBDA_OPTS[self._t_lam_cb.currentIndex()][2]
        return dict(
            lam1         = self._parse_lambda(self._o_lam_le),
            fc1_ma       = self._o_fc_sp.value(),
            j1           = j1,
            j1_err       = j1_err,
            lam2         = self._parse_lambda(self._t_lam_le),
            lam2_err     = lam2_err,
            fc2_ma       = self._t_fc_sp.value(),
            fc2_err_ma   = self._t_fc_err_sp.value() if self._t_fc_err_sp else 0.0,
            fc2_external = self._t_fc_ext_cb.isChecked() if self._t_fc_ext_cb else True,
        )


# --------------------------------------------------------------------------
# Quick calibration dialog
# --------------------------------------------------------------------------

def _calib_origin(cfg, orig):
    """Where the constants of the stored ages and J come from, in words."""
    keys = ('stdAgeMa', 'lambdaYr', 'monitor')
    same = bool(orig) and all(orig.get(k) == cfg.get(k) for k in keys)
    if orig and not same:
        return "Set by the last Calibration…; Revert to original goes back to the file's own."
    if (cfg.get('constants') or {}).get('file_std_age'):
        return ("Read from the file: the monitor age and λ of the mass-spectrometer export, "
                "as the Data converter found them.")
    return ("Read from the file or session. Change them only if the data were calculated "
            "with other constants, e.g. ages and a J copied from an older study.")


def _recast_j(cfg):
    """The J the stored ages of steps without a J of their own were calculated
    with: the flux J of an R mode, else the typed J."""
    J, _ = cfg_J(cfg)
    return J if J else (cfg.get('jVal', 0.0) or 0.0)


def _match_calibration_name(lam, fc_ma, fc_err_ma):
    """Name of the FC preset matching these constants (with its anchored λ),
    or None if they do not correspond to a known calibration."""
    for name, age, err, ext in FC_PRESETS:
        pl = paired_lambda(name)
        if (pl is not None and abs(fc_ma - age) < 5e-4
                and abs(fc_err_ma - err) < 5e-5 and abs(lam - pl[1]) < 1e-14):
            return name
    return None


_CalibrationDialog = CalibrationDialog        # shared with Age distribution


# --------------------------------------------------------------------------
# Session persistence
# --------------------------------------------------------------------------

_SESSION_CFG_DEFAULTS = dict(
    minPct=50.0, minN=3, pCut=5.0, errMode='sem_mswd_2s',
    zoomMode='auto', yMin=50.0, yMax=58.0, ageDec=2,
    # Age source / flux calibration, single combined key:
    #   'imported'    : use stored age/err1s directly (no J propagation)
    #   'ratio_J'     : compute age from r4039 using a J value (ratio-space WM)
    #   'ratio_R_std' : compute age from r4039 via R standard (ratio-space WM)
    ageFluxMode='ratio_R_std',
    jVal=1.0e-3,                  # J value (when ageFluxMode='ratio_J')
    jErrPct=0.0,                  # J error % 1σ (when ageFluxMode='ratio_J')
    rStd=70.0,                    # R standard 40Ar/39Ar (when ageFluxMode='ratio_R_std')
    rStdErrPct=0.10,              # R_std 1σ error, interpretation set by rStdErrMode
    rStdErrMode='pct',            # 'pct' → value is % relative; 'abs' → absolute 1σ
    stdAgeMa=STD_AGE_DEFAULT,    # flux monitor age (Ma)
    stdAgeErrMa=STD_AGE_ERR,     # 1σ error on monitor age (Ma)
    # 'internal' (no λ) or 'external' (λ already carried by the monitor age).
    # Follows the default calibration: astronomically anchored monitors such as
    # Phillips (2022) are 'external', so σ_λ is not added on top a second time.
    stdAgeErrType=DEFAULT_STD_AGE_ERR_TYPE,
    lambdaYr=LAMBDA_DEFAULT,     # ⁴⁰K decay constant (yr⁻¹)
    lambdaErrYr=LAMBDA_ERR,      # 1σ uncertainty on λ (yr⁻¹)
    fullSystematics=False,       # False → intercalibrated external (default);
                                 # True  → Min et al. (2000) full external (physical λ)
)


def _session_err_div(cfg, input_1s_checked):
    """Divisor that brings on-disk measured errors to 1σ.

    errScale=2 marks files written by this tool (errors stored at 2σ).
    Converter files declare stepSigma=1 (errors already 1σ); honour that
    flag the same way the Age distribution tool does.  For older files with
    neither marker, fall back to the "Input errors are 1σ" checkbox.
    """
    return file_err_div(cfg, 1.0 if input_1s_checked else 2.0)


def _load_session_data(data, err_div=2.0):
    """Deserialize an on-disk session into in-memory step dicts.

    On disk the three *measured* uncertainties (age error, R error, Ca/K error)
    are stored at 2σ for backward compatibility with older sessions and with the
    ArArCALC converter exports.  In memory everything is kept at 1σ, so they are
    divided by `err_div` here (2.0 for the standard 2σ-on-disk format; the caller
    passes 1.0 for legacy files that stored raw 1σ values).  j_err and the
    isochron errors are 1σ on disk *and* in memory, so they are never scaled.
    """
    cfg = {**_SESSION_CFG_DEFAULTS, **data.get('cfg', {})}
    # Migrate old ageMode + fluxMode → ageFluxMode
    if 'ageFluxMode' not in data.get('cfg', {}):
        old_age   = cfg.get('ageMode',  'imported')
        old_flux  = cfg.get('fluxMode', 'J')
        if old_age == 'from_ratio' or old_flux == 'R_std':
            # If R_std was ever configured, the user intended ratio mode
            cfg['ageFluxMode'] = 'ratio_R_std' if old_flux == 'R_std' else 'ratio_J'
        elif old_age == 'imported' and old_flux == 'J':
            # Legacy files with no explicit age mode: default to ratio_R_std so
            # ratio data in the file is used automatically (user can switch to
            # 'Imported ages' if the file contains no ratio columns).
            cfg['ageFluxMode'] = 'ratio_R_std'
        else:
            cfg['ageFluxMode'] = 'imported'
    # files written with the former mode name 'ratio_R_direct'
    if cfg.get('ageFluxMode') == 'ratio_R_direct':
        cfg['ageFluxMode'] = 'ratio_R_std'
    # Old converter exports wrote rStd=0 as a placeholder; zero causes J→∞
    if cfg.get('rStd', 0.0) <= 0.0:
        cfg['rStd'] = _SESSION_CFG_DEFAULTS['rStd']
    def _f(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _fopt(v):
        """Float or None, preserves None for optional isochron fields."""
        if v is None:
            return None
        try:
            f = float(v)
            return None if f != f else f   # NaN → None
        except (TypeError, ValueError):
            return None

    steps = []
    for s in data.get('steps', []):
        # A step may arrive in one of two conventions:
        #   • on-disk dict: has 'err2s' (2σ) → divide by err_div to get 1σ memory
        #   • in-memory dict: has 'err1s' (already 1σ, e.g. from the paste path)
        # The same divisor applies to the R and Ca/K errors.
        if 'err1s' in s and 'err2s' not in s:
            _ediv = 1.0
            _e1s  = _f(s.get('err1s', 0.0))
        else:
            _ediv = err_div
            _e1s  = _f(s.get('err2s', 0.0)) / err_div
        steps.append({
            'label':     str(s.get('label', '?')),
            'power':     _f(s.get('power', 0.0)),
            'age':       _f(s.get('age', 0.0)),
            'err1s':     _e1s,
            'pct39Ar':   _f(s.get('pct39Ar', 0.0)),
            'caK':       _f(s.get('caK', 0.0)),
            'pct40':     _f(s.get('pct40', 0.0)),
            'r4039':     _f(s.get('r4039', 0.0)),
            'err_r4039': _f(s.get('err_r4039', 0.0)) / _ediv,
            'err_cak':   _f(s.get('err_cak', 0.0)) / _ediv,
            'j_val':     _f(s.get('j_val', 0.0)),
            'j_err':     _f(s.get('j_err', 0.0)),
            'status':    s.get('status', 'normal'),
            'runID':     str(s.get('runID', '')),
            # Run time and raw signals (export only), written by the converter
            'run_dt':    str(s.get('run_dt') or ''),
            'days_irr':  _fopt(s.get('days_irr')),
            'raw':       dict(s['raw']) if isinstance(s.get('raw'), dict) else {},
            # Isochron fields, optional, preserved as float or None
            'iso_nx':    _fopt(s.get('iso_nx')),
            'iso_nsx':   _fopt(s.get('iso_nsx')),
            'iso_ny':    _fopt(s.get('iso_ny')),
            'iso_nsy':   _fopt(s.get('iso_nsy')),
            'iso_nri':   _f(s.get('iso_nri', 0.0)),
            'iso_ix':    _fopt(s.get('iso_ix')),
            'iso_isx':   _fopt(s.get('iso_isx')),
            'iso_iy':    _fopt(s.get('iso_iy')),
            'iso_isy':   _fopt(s.get('iso_isy')),
            'iso_iri':   _f(s.get('iso_iri', 0.0)),
        })
    # Files from older converter versions labelled their (never recast)
    # step ages with the tool default calibration instead of the file's.  On
    # Imported ages the ages are the file's, so they take its label.  In the R
    # modes cfg's calibration is the one chosen for computing from R, so it is
    # kept and the stored ages and per-step J are brought onto it instead.
    _fix = file_age_calibration(cfg, steps)
    if _fix:
        if cfg.get('ageFluxMode') == 'imported':
            cfg.update(_fix)
        else:
            _recalc_ages_in_steps(steps, _fix['lambdaYr'], _fix['stdAgeMa'], 0.0,
                                  cfg.get('lambdaYr', LAMBDA_DEFAULT),
                                  cfg.get('stdAgeMa', STD_AGE_DEFAULT),
                                  fc2_err_ma=cfg.get('stdAgeErrMa', STD_AGE_ERR))
    # orig_title: the experiment's name as it first came in, kept through any
    # renaming so the export can still use it
    return {'title': data.get('title', ''), 'cfg': cfg, 'raw_steps': steps,
            'orig_title': data.get('orig_title') or data.get('title', '')}


def _steps_to_disk(steps):
    """Serialize in-memory step dicts (1σ, 'err1s' key) to the on-disk format.

    On disk the three measured uncertainties are written at 2σ under the legacy
    key names ('err2s', plus 2σ 'err_r4039'/'err_cak') and the session cfg carries
    errScale=2.  This keeps every file produced here readable by older versions of
    the tool and by the ArArCALC → .ararpt converters, which all expect 2σ.
    j_err and the isochron errors are 1σ on disk too, so they pass through as-is.
    """
    out = []
    for s in steps:
        d = dict(s)
        d['err2s'] = d.pop('err1s', 0.0) * 2.0
        for k in ('err_r4039', 'err_cak'):
            if k in d:
                d[k] = d[k] * 2.0
        out.append(d)
    return out


def _fo_rr(s, cfg):
    """Step export cells [ratio_str, err_str]: R_unk/R_std and its error, in
    R mode and on Imported ages with the file's own R standard; blank
    otherwise.  Shared by the single and the stack CSV export."""
    if cfg.get('ageFluxMode', 'imported') not in ('ratio_R_std', 'imported'):
        return ['', '']
    r = s.get('r4039') or 0.0
    rStd = cfg.get('rStd') or 0.0
    if (cfg.get('ageFluxMode', 'imported') == 'imported'
            and rStd == _SESSION_CFG_DEFAULTS['rStd']):
        return ['', '']   # placeholder R_std, not the file's
    if r <= 0 or rStd <= 0:
        return ['', '']
    ratio = r / rStd
    sr = s.get('err_r4039') or 0.0
    rStdErrVal = cfg.get('rStdErrPct', 0.0) or 0.0
    if cfg.get('rStdErrMode', 'pct') == 'pct':
        srStd_frac = rStdErrVal / 100.0
    else:
        srStd_frac = rStdErrVal / rStd
    err = ratio * math.sqrt((sr / r) ** 2 + srStd_frac ** 2)
    return [ff(ratio, 8), ff(err, 8)]


def _flux_override_from_disk(v):
    """A stack session's saved hole assignment: (irr, disk, hole), 'none', or
    None for anything else.  Older stacks wrote 'none' as the letters
    ['n', 'o', 'n', 'e']."""
    if v == 'none' or (isinstance(v, (list, tuple)) and ''.join(map(str, v)) == 'none'):
        return 'none'
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return tuple(v)
    return None


def _aged_J(cfg, steps):
    """(J, relative 1σ J) the steps' ages are computed with: the cfg's J
    (arar_math.cfg_J), except in J mode without a disk-flux J, where every
    step is aged with its own j_val (_compute_steps_with_age_mode), and on
    imported ages, where it is the J and σ_J stored with the steps (σ_J from
    the cfg's J error when the steps carry none).  (None, 0.0) when there is
    no J.  Used for the plateau age, its internal tier, the saved plateau J
    and the isochron, so all of them sit on the same J."""
    if cfg.get('ageFluxMode', 'imported') == 'imported':
        jv = [(float(s['j_val']), float(s.get('j_err', 0.0) or 0.0))
              for s in steps if s.get('j_val', 0.0) > 0]
        if not jv:
            return None, 0.0
        rels = [e / j for j, e in jv if e > 0]
        rel = (float(np.median(rels)) if rels
               else float(cfg.get('jErrPct', 0.0) or 0.0) / 100.0)
        return float(np.median([j for j, _ in jv])), rel
    J, rel = cfg_J(cfg)
    if cfg.get('ageFluxMode') == 'ratio_J' and not cfg.get('jForce'):
        jv = [s['j_val'] for s in steps
              if s.get('r4039', 0.0) > 0 and s.get('j_val', 0.0) > 0]
        if jv:
            J = float(np.median(jv))
            rel = float(cfg_j_err_frac(cfg))
    return (J, rel) if J else (None, 0.0)


def isochron_tiers(steps, fit, J, lam, inverse, cfg, sJ_age=None):
    """Isochron age with its three 1σ tiers, for the on-screen isochron (single
    and stacked) and both data exports.

    Analytical: the York fit (arar_math.isochron_age, √MSWD when MSWD > 1).
    Internal: ⊕ the J term, sJ_age (Ma) when given (a stack with each
    aliquot's own J error), else ∂t/∂ln J · σ_J/J with σ_J the steps' median
    j_err.  External: ⊕ the calibration systematic (systematic_components), as
    for the plateau.  Returns isochron_age's dict plus e1s_ana/int/ext (None
    when there is no age)."""
    out = isochron_age(fit, J, lam, inverse)
    out.update(e1s_ana=None, e1s_int=None, e1s_ext=None)
    age, s_ana = out['age_ma'], out['s_ana']
    if age is None or s_ana is None:
        return out
    if sJ_age is None:
        j_errs = [s.get('j_err', 0.0) for s in steps
                  if s.get('j_err', 0.0) > 0 and s.get('status', 'normal') != 'excluded']
        sJ_age = dt_dlnR(age, lam) * (float(np.median(j_errs)) / J if j_errs else 0.0)
    s_int = math.sqrt(s_ana**2 + sJ_age**2)
    s_ext = (float(external_sigma_array(np.array([age]), np.array([s_int]), cfg)[0])
             if cfg else s_int)
    out.update(e1s_ana=s_ana, e1s_int=s_int, e1s_ext=s_ext)
    return out


def _iso_scale(fit):
    """√MSWD when the isochron MSWD exceeds 1, else 1: the factor applied to
    the isochron age and trapped ratio (arar_math.isochron_age), and so to the
    band and intercept bar drawn with them."""
    return math.sqrt(fit['mswd']) if fit['mswd'] > 1.0 else 1.0


def _compute_steps_with_age_mode(raw_steps, cfg):
    mode = cfg.get('ageFluxMode', 'imported')
    if mode == 'imported':
        return raw_steps
    lam = cfg.get('lambdaYr', LAMBDA_DEFAULT)
    J = cfg_J(cfg)[0]
    if J is None:
        if mode != 'ratio_J':
            return raw_steps          # no usable R standard
        J = 0.0                       # J mode: the steps may carry their own J
    # j_err stores the flux measurement precision only (the internal tier), the
    # same fraction the plateau's internal tier uses.  The calibration
    # systematics are added separately in the external tier via
    # systematic_components; they must not be bundled here or they get
    # double-counted when the isochron adds the two contributions in quadrature.
    _j_int_frac = cfg_j_err_frac(cfg)
    out = []
    for s in raw_steps:
        sc = dict(s)
        r  = s.get('r4039',     0.0)
        er = s.get('err_r4039', 0.0)
        if r > 0:
            # In ratio_J mode, per-step j_val overrides the global J (supports
            # aliquots from different irradiation positions). In ratio_R_std mode
            # the global J recomputed from the current rStd/rStdErrPct is always
            # used so that changing those parameters actually affects the ages.
            # When jForce is set (a disk-flux hole's J has been applied), the
            # global J wins over any per-step j_val.
            if mode == 'ratio_J' and not cfg.get('jForce'):
                step_j = s.get('j_val', 0.0)
                use_j  = step_j if step_j > 0 else J
            else:
                use_j = J
            age_ma, err1s_ma = r_to_age(r, er, use_j, 0.0, lam)  # J error excluded from MSWD
            sc['age']   = age_ma
            sc['err1s'] = err1s_ma
            # Keep j_val consistent with the current calibration so that downstream
            # code (e.g. _wm_session_j) reads the correct J, not a stale value
            # baked into the original file.
            sc['j_val'] = use_j
            sc['j_err'] = use_j * _j_int_frac       # 1σ, flux precision only
        out.append(sc)
    return out


def _plateau_summary(p, eff):
    """The plateau of an aliquot as saved in an .ararstack: its age and 1σ
    tiers, the plateau step window ('pset'), and the J its mean R was turned
    into an age with, so a reader that rebuilds ages from R (the probability
    plot) uses exactly this plateau.  None when there is no plateau."""
    if not (p and p.get('r')):
        return None
    out = {
        'age':     p['r'].get('mean'),
        'e1s_ana': p.get('e1s_ana', p.get('e1s_int')),
        'e1s_int': p.get('e1s_int'),
        'e1s_ext': p.get('e1s_ext'),
        'mswd':    p['r'].get('mswd'),
        'prob':    p['r'].get('prob'),
        'n':       p['r'].get('n'),
        'pset':    sorted(int(i) for i in p.get('ois', [])),
    }
    _J0, _rel = cfg_J(eff)
    _J = p.get('J') or _J0      # the J find_plateau aged this plateau with
    if _J:
        out['J'] = float(_J)
        out['jErrFrac'] = float(_rel if _J0 else cfg_j_err_frac(eff))
    elif p.get('J_rel'):
        # imported ages: the J error stored with the steps (no J is saved, so a
        # reader rebuilds the age from the R standard, not from the lab's J)
        out['jErrFrac'] = float(p['J_rel'])
    return out


def _show_r_standard(cfg):
    """False when the R standard of a cfg is the placeholder of an Imported-ages
    session (see _fo_rr), so the export leaves the column blank; else None, the
    default of export.flux_cells."""
    if (cfg.get('ageFluxMode', 'imported') == 'imported'
            and (cfg.get('rStd') or 0.0) == _SESSION_CFG_DEFAULTS['rStd']):
        return False
    return None


def _export_j_cells(cfg):
    """(J, 1σ J) in force for a cfg whose J is not read from the steps, else None.

    In R-standard mode every step is aged with the J derived from the R
    standard, and with a disk-flux J applied (jForce) with that hole's J.
    Steps without a usable 40Ar*/39ArK keep a stale J from the file, so the
    exported J column comes from the cfg instead of the steps."""
    mode = cfg.get('ageFluxMode', 'imported')
    if mode == 'ratio_R_std' or (mode == 'ratio_J' and cfg.get('jForce')):
        J, rel = cfg_J(cfg)
        if J:
            return J, J * rel
    return None


def _check_rstd_age_consistency(raw_steps, cfg, tol_ma=0.05, tol_pct=0.1):
    """Compare ages re-computed from r4039 against the imported ages in raw_steps.

    Only meaningful for ratio_R_std sessions, where both the r4039 ratios and
    the 'age' values in the file were produced by the same converter run.  If the
    constants (R_std, FC age, λ) are read back correctly the two should agree to
    better than rounding error.  A discrepancy > tol_ma (Ma) or > tol_pct (%)
    on the per-step mean indicates a calibration mismatch, wrong constants loaded.

    Returns (ok, summary_str).  ok=True means ages are consistent.
    """
    if cfg.get('ageFluxMode') != 'ratio_R_std':
        return True, ''
    # Files saved by this tool (errScale=2) keep the converter's original step
    # ages while R_std / λ / FC may since have been changed on purpose (a disk
    # flux, Calibration…), so a difference is expected there, not an error.
    if cfg.get('errScale') == 2:
        return True, ''

    computed = _compute_steps_with_age_mode(raw_steps, cfg)

    diffs, ref_ages = [], []
    for orig, comp in zip(raw_steps, computed):
        imp_age  = orig.get('age', 0.0)
        comp_age = comp.get('age', 0.0)
        r        = orig.get('r4039', 0.0)
        if r > 0 and imp_age > 0 and comp_age > 0:
            diffs.append(abs(comp_age - imp_age))
            ref_ages.append(imp_age)

    if not diffs:
        return True, ''

    mean_abs  = sum(diffs) / len(diffs)
    max_abs   = max(diffs)
    median_age = sorted(ref_ages)[len(ref_ages) // 2]
    mean_rel   = mean_abs / median_age * 100.0 if median_age > 0 else 0.0

    if mean_abs <= tol_ma and mean_rel <= tol_pct:
        return True, ''

    return False, (
        f"Mean step difference:    {mean_abs:.3f} Ma  ({mean_rel:.3f} %)\n"
        f"Maximum step difference: {max_abs:.3f} Ma\n"
        f"Steps compared:          {len(diffs)}\n\n"
        f"Check that R_std, FC age, and λ in this file match the values\n"
        f"used by the converter that produced the imported ages."
    )


# --------------------------------------------------------------------------
# Spectrum canvas (three panels)
# --------------------------------------------------------------------------

def _bracket_y_in_axes(ax, yr, data_top, n_lines, fontsize=8.5,
                       gap_frac=0.04, pad_frac=0.008, at=None):
    """Y (data coords) for a plateau bracket so the bracket and the label that
    hangs above it both stay inside the age panel.

    The bracket normally sits a little above the tallest plateau step box.  A
    step with a large uncertainty can push that box past the y-limit, so the
    label's height is reserved (points -> data units via the axes' pixel height)
    and the bracket clamped below it; tall boxes are cut off by the normal axes
    clipping instead.

    at: an age (Ma) chosen in the export dialog; the bracket goes there,
    within the same floor and ceiling.
    """
    lo, hi = yr
    span = hi - lo
    if span <= 0:
        return hi
    px = dpi = 0.0
    try:
        px  = float(ax.bbox.height)
        dpi = float(ax.figure.dpi)
    except Exception:
        pass
    if px > 0 and dpi > 0:
        # 1.45 = line height / font size, leaving a little inter-line air.
        lbl_data = n_lines * fontsize * 1.45 * dpi / 72.0 * span / px
    else:
        lbl_data = span * 0.06 * n_lines      # no usable geometry yet
    ceiling = hi - lbl_data - span * (pad_frac + 0.012)
    floor   = lo + span * 0.05
    want = data_top + span * gap_frac if at is None else float(at)
    return max(min(want, ceiling), floor)


# --------------------------------------------------------------------------
# Panel layout shared by the screen canvases and the export dialog
# --------------------------------------------------------------------------

# Default stack: % 40Ar*, Ca/K, apparent-age spectrum.
DEFAULT_PANEL_LAYOUT = [('p40', 1.0), ('cak', 1.0), ('age', 2.8)]

# Panels the drawing code writes to but the user has switched off go here, so
# the drawing code never has to know which panels are on.
_SCRATCH_FIG = Figure()


def _scratch_ax():
    _SCRATCH_FIG.clear()
    return _SCRATCH_FIG.add_subplot(111)


def build_spectrum_axes(fig, layout=None, ids=('p40', 'cak', 'age', 'iso'),
                        gaps=None):
    """Create the panels in *layout* (a list of (panel id, height weight)).

    Returns {id: axes} for every id in *ids*: the requested panels live in
    *fig*, in that order and with those height ratios; the rest get a
    throwaway axes on a scratch figure, so the drawing code can keep writing to
    every panel without knowing which are switched on.  The isochron panel does
    not share the x axis with the spectrum panels.
    """
    fig.clear()
    layout = [(pid, float(w)) for pid, w in (layout or DEFAULT_PANEL_LAYOUT)
              if pid in ids]
    if not layout:
        layout = [('age', 1.0)]
    # Consecutive spectrum panels are packed tight (they share the x axis); an
    # isochron panel is a separate block with room for its own axis labels.
    blocks, cur = [], []
    for pid, w in layout:
        if pid == 'iso':
            if cur:
                blocks.append(('spec', cur)); cur = []
            blocks.append(('iso', [(pid, w)]))
        else:
            cur.append((pid, w))
    if cur:
        blocks.append(('spec', cur))
    gaps = gaps or {}
    place = gaps.get('place', 'below')
    if place == 'right' and any(pid == 'iso' for pid, _ in layout) \
            and any(pid != 'iso' for pid, _ in layout):
        # spectrum panels in the left column, the isochrons in the right one
        spec_items = [(pid, w) for pid, w in layout if pid != 'iso']
        outer = gridspec.GridSpec(
            1, 2, figure=fig,
            width_ratios=[1.0, float(gaps.get('iso_width', 0.85))],
            wspace=float(gaps.get('side', 0.40)), left=0.10, right=0.97,
            top=0.92, bottom=0.09)
        inner = gridspec.GridSpecFromSubplotSpec(
            len(spec_items), 1, subplot_spec=outer[0],
            height_ratios=[w for _, w in spec_items],
            hspace=float(gaps.get('spec', 0.06)))
        axes, share, spectrum = {}, None, []
        for k, (pid, _w) in enumerate(spec_items):
            ax = fig.add_subplot(inner[k], sharex=share)
            share = share or ax
            spectrum.append(ax)
            axes[pid] = ax
        axes['iso'] = fig.add_subplot(outer[1])
        for ax in spectrum[:-1]:
            ax.tick_params(labelbottom=False)
        for pid in ids:
            axes.setdefault(pid, _scratch_ax())
        axes['_visible'] = [pid for pid, _ in layout]
        axes['_spectrum'] = spectrum
        return axes
    if place == 'above':
        blocks.sort(key=lambda b: 0 if b[0] == 'iso' else 1)
    outer = gridspec.GridSpec(
        len(blocks), 1, figure=fig,
        height_ratios=[sum(w for _, w in items) for _, items in blocks],
        hspace=float(gaps.get('iso', 0.30)), left=0.10, right=0.97,
        top=0.92, bottom=0.09,
    )
    axes, share, spectrum = {}, None, []
    for bi, (kind, items) in enumerate(blocks):
        if kind == 'iso':
            axes['iso'] = fig.add_subplot(outer[bi])
            continue
        inner = gridspec.GridSpecFromSubplotSpec(
            len(items), 1, subplot_spec=outer[bi],
            height_ratios=[w for _, w in items],
            hspace=float(gaps.get('spec', 0.06)))
        for k, (pid, _w) in enumerate(items):
            ax = fig.add_subplot(inner[k], sharex=share)
            share = share or ax
            spectrum.append(ax)
            axes[pid] = ax
    for ax in spectrum[:-1]:
        ax.tick_params(labelbottom=False)
    for pid in ids:
        axes.setdefault(pid, _scratch_ax())
    axes['_visible'] = [pid for pid, _ in layout]
    axes['_spectrum'] = spectrum
    return axes


def finish_spectrum_axes(axes, xlabel='Cumulative %\u00b3\u2079Ar Released',
                         fontsize=9):
    """Put the x-axis label and tick labels on the lowest spectrum panel only."""
    spectrum = axes.get('_spectrum') or []
    for ax in spectrum:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
    if spectrum:
        spectrum[-1].tick_params(labelbottom=True)
        spectrum[-1].set_xlabel(xlabel, fontsize=fontsize)


class PlateauCanvas(FigureCanvas):
    """Embeddable three-panel (% ⁴⁰Ar*, Ca/K, Apparent Age) spectrum widget."""

    def __init__(self, click_cb=None, parent=None):
        self.fig = Figure(figsize=(8, 9))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self._click_cb  = click_cb
        self._steps_c   = []     # kept for hit-testing
        self._build_axes()
        self.mpl_connect('button_press_event', self._on_click)

    def _on_click(self, event):
        if event.inaxes is not self.ax_age or self._click_cb is None:
            return
        for ti, s in enumerate(self._steps_c):
            if s['cs'] <= event.xdata <= s['ce']:
                eh = s.get('err1s', 0.0) * 2.0
                if s['age'] - eh <= event.ydata <= s['age'] + eh:
                    self._click_cb(ti)
                    return

    # ── Axes construction ─────────────────────────────────────────────────────
    def _build_axes(self):
        # _panel_layout is set by the export dialog; on screen it is None and
        # the default three-panel stack is used.
        self._axes = build_spectrum_axes(self.fig,
                                         getattr(self, '_panel_layout', None),
                                         gaps=getattr(self, '_layout_gaps', None))
        self.ax_p40 = self._axes['p40']
        self.ax_cak = self._axes['cak']
        self.ax_age = self._axes['age']
        self.ax_iso = self._axes['iso']

    # ── Full redraw ───────────────────────────────────────────────────────────
    def redraw(self, steps_c, plateau, pset, age_range, sample_title='',
               display_1sigma=False, age_dec=2):
        self._steps_c = steps_c or []
        self._build_axes()

        if not steps_c:
            self.ax_age.text(0.5, 0.5, 'No data loaded',
                             transform=self.ax_age.transAxes,
                             ha='center', va='center', fontsize=12, color='#aaa')
            self.draw()
            return

        yr_p40 = _p40_ylim([s['pct40'] for s in steps_c])
        yr_cak = _cak_ylim(
            [s['caK']           for s in steps_c],
            [s.get('err_cak', 0.0) for s in steps_c],
            gas_pct=[s.get('pct39Ar') for s in steps_c]) if steps_c else (0.0, 10.0)

        sig_scale = 1.0 if display_1sigma else 2.0
        self._draw_boxes(self.ax_p40, steps_c, 'pct40', None,      yr_p40, pset)
        self._draw_boxes(self.ax_cak, steps_c, 'caK',   'err_cak', yr_cak, pset,
                         sig_scale=sig_scale)
        self._draw_boxes(self.ax_age, steps_c, 'age',   'err1s',   age_range, pset,
                         sig_scale=sig_scale)

        self._ext_hi = None
        if plateau and getattr(self, '_show_bracket', True):
            self._draw_bracket(self.ax_age, steps_c, plateau, pset, age_range,
                               display_1sigma=display_1sigma, age_dec=age_dec)
        if getattr(self, '_show_int_age', True):
            self._draw_int_age(self.ax_age, steps_c, age_range,
                               display_1sigma=display_1sigma, age_dec=age_dec)

        yr_span = age_range[1] - age_range[0]
        for s in (steps_c if getattr(self, '_show_step_labels', True) else []):
            cx  = (s['cs'] + s['ce']) / 2.0
            bot = s['age'] - s.get('err1s', 0.0) * sig_scale
            self.ax_age.text(
                cx, bot - yr_span * 0.012,
                s['label'], ha='center', va='top', fontsize=6.5,
                color='#aaa', clip_on=True, zorder=3,
            )

        self.ax_p40.set_ylabel('%\u2074\u2070Ar*',     fontsize=9)
        self.ax_cak.set_ylabel(getattr(self, '_cak_label', 'Ca/K'), fontsize=9)
        self.ax_age.set_ylabel('Apparent Age (Ma)',     fontsize=9)
        self.ax_age.set_xlabel('Cumulative %\u00b3\u2079Ar Released', fontsize=9)

        self.ax_p40.set_ylim(yr_p40)
        self.ax_cak.set_ylim(yr_cak)
        self.ax_age.set_ylim(age_range[0], getattr(self, '_ext_hi', None)
                             or age_range[1])
        for ax in (self.ax_p40, self.ax_cak, self.ax_age):
            ax.set_xlim(0.0, 100.0)
            ax.tick_params(axis='both', labelsize=8)
            ax.yaxis.grid(True, linewidth=0.5, color='#f0f0f0', zorder=0)
            ax.spines['top'].set(linewidth=0.5,  color='#ccc')
            ax.spines['right'].set(linewidth=0.5, color='#ccc')
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)

        self.ax_p40.yaxis.set_major_locator(ticker.MultipleLocator(25))

        if sample_title and getattr(self, '_show_title', True):
            top = (self._axes.get('_spectrum') or [self.ax_p40])[0]
            top.set_title(sample_title, fontsize=10, fontweight='bold', pad=4)

        finish_spectrum_axes(self._axes)
        if not getattr(self, '_offscreen', False):
            self.draw_idle()

    # ── Per-panel drawing ─────────────────────────────────────────────────────
    def _draw_boxes(self, ax, steps_c, val_key, err_key, yr, pset, sig_scale=1.0):
        yr_span = yr[1] - yr[0]
        fixed_half = yr_span * 0.017   # half-height for non-age panels

        top = yr[1] + yr_span        # an open K/Ca box runs off the panel
        for i, s in enumerate(steps_c):
            b = _box_bounds(s, val_key, err_key, sig_scale, fixed_half, top)
            if b is None:
                continue
            lo_i, hi_i = b
            status = s.get('status', 'normal')
            exc    = status == 'excluded'
            forced = status == 'forced'
            color  = '#c0392b' if exc else ('#1565c0' if forced else '#111111')
            face   = '#fdecea' if exc else ('#e8f0fe' if forced else '#f5f5f5')

            x  = s['cs']
            bw = max(s['ce'] - s['cs'], 0.0)
            in_plat = err_key and (i in pset)
            lw = 1.6 if in_plat else 1.0
            ax.add_patch(Rectangle(
                (x, lo_i), bw, hi_i - lo_i,
                facecolor=face, edgecolor=color, linewidth=lw, zorder=2,
            ))

            nx = steps_c[i + 1] if i + 1 < len(steps_c) else None
            nb = (_box_bounds(nx, val_key, err_key, sig_scale, fixed_half, top)
                  if nx else None)
            if nb:
                cx   = s['ce']          # shared x boundary

                gap_lo = min(hi_i, nb[1])
                gap_hi = max(lo_i, nb[0])
                if gap_lo < gap_hi:
                    nx_st = nx.get('status', 'normal')
                    if exc or nx_st == 'excluded':
                        lc = '#c0392b'
                    elif forced or nx_st == 'forced':
                        lc = '#1565c0'
                    else:
                        lc = '#111111'
                    ax.plot([cx, cx], [gap_lo, gap_hi], color=lc, linewidth=1.0, zorder=2)

    def _draw_bracket(self, ax, steps_c, plateau, pset, age_range,
                      display_1sigma=False, age_dec=2):
        plat_steps = [s for i, s in enumerate(steps_c) if i in pset]
        if not plat_steps:
            return
        x1 = plat_steps[0]['cs']
        x2 = plat_steps[-1]['ce']
        xm = (x1 + x2) / 2.0

        yr_span   = age_range[1] - age_range[0]
        _sc       = 1.0 if display_1sigma else 2.0
        max_top   = max(s['age'] + s['err1s'] * _sc for s in plat_steps)
        # Big text: the label goes to the top of the panel, the bracket just
        # under it.  If a tall step would cross the bracket, the age axis is
        # extended upward so the label still fits above every step.
        k_txt = float(getattr(self, '_text_k', 1.0) or 1.0)
        at = getattr(self, '_bracket_at', None)
        top = bool(getattr(self, '_label_top', False))
        self._ext_hi = None
        bracket_y = None
        if top and at is None and not getattr(self, '_ymax_locked', False):
            try:
                px = float(ax.bbox.height); dpi = float(ax.figure.dpi)
                lo, hi = age_range
                span = hi - lo
                fl = 8.5 * k_txt * 1.45 * dpi / 72.0 / px     # one text line
                f = fl + 0.02
                # the age sits above the line, the stats line below it
                want = max_top + span * (0.02 + fl)
                ceiling = hi - span * f
                if want > ceiling and f < 0.5:
                    self._ext_hi = lo + (want - lo) / (1.0 - f)
                    bracket_y = want
                else:
                    # just clear of the tallest plateau step (room for the
                    # statistics line under the bracket), not up at the top
                    bracket_y = want
            except Exception:
                bracket_y = None
        if bracket_y is None:
            if at is None and top:
                at = age_range[1]
            bracket_y = _bracket_y_in_axes(
                ax, age_range, max_top, n_lines=2 if top else 1,
                fontsize=8.5 * (k_txt if top else 1.0), at=at)
        tick_h    = yr_span * 0.012

        manual = plateau.get('forced', False)
        bracket_color = '#1565c0' if manual else '#111'

        ax.plot([x1, x1, x2, x2],
                [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
                color=bracket_color, linewidth=1.5, zorder=5, solid_capstyle='butt')

        r       = plateau['r']
        e1s_int = plateau['e1s_int']   # internal tier (+J_Rstd); shown in the bracket label
        scale   = 1.0 if display_1sigma else 2.0
        sig_lbl = '1σ' if display_1sigma else '2σ'
        e_show  = e1s_int * scale
        _af = f".{age_dec}f"
        lbl = (f"{r['mean']:{_af}} \u00b1 {e_show:{_af}} Ma ({sig_lbl})   "
               f"({plateau['pct39']:.1f}% \u00b3\u2079Ar,  "
               f"MSWD\u202f=\u202f{r['mswd']:.2f},  "
               f"p\u202f=\u202f{r['prob']:.2f})"
               + ("  [manual]" if manual else ""))
        if top:
            # Big text: the age above the line, the statistics below it
            _head, _tail = lbl.split('   ', 1)
            ax.text(xm, bracket_y + yr_span * 0.008, _head,
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                    color=bracket_color, clip_on=True, zorder=6)
            ax.text(xm, bracket_y - yr_span * 0.008, _tail,
                    ha='center', va='top', fontsize=8.5, fontweight='bold',
                    color=bracket_color, clip_on=True, zorder=6)
        else:
            ax.text(xm, bracket_y + yr_span * 0.008, lbl,
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                    color=bracket_color, clip_on=True, zorder=6)

        # Analytical and full-external errors in bottom-left
        if not getattr(self, '_show_err_box', True):
            return
        e1s_ana = plateau.get('e1s_ana', e1s_int) * scale
        e1s_ext = plateau.get('e1s_ext', e1s_int) * scale
        ax.text(0.02, 0.02,
                f"Analytical: \u00b1{e1s_ana:{_af}} Ma ({sig_lbl})\n"
                f"External: \u00b1{e1s_ext:{_af}} Ma ({sig_lbl})",
                ha='left', va='bottom', fontsize=8, color='#555',
                transform=ax.transAxes, clip_on=True, zorder=5)

    def _draw_int_age(self, ax, steps_c, age_range, display_1sigma=False,
                      age_dec=2):
        r = integrated_age(steps_c, getattr(self, '_lam', None) or LAMBDA_DEFAULT)
        if r is None:
            return
        ia      = r['mean']
        scale   = 1.0 if display_1sigma else 2.0
        sig_lbl = '1σ' if display_1sigma else '2σ'
        e_show  = r['sem'] * scale
        if age_range[0] <= ia <= age_range[1]:
            ax.axhline(ia, color='#bbb', linewidth=0.8,
                       linestyle=(0, (5, 3)), zorder=1)
        _af2 = f'.{age_dec}f'
        ax.text(0.99, 0.02,
                f"Total-gas age:  {ia:{_af2}} \u00b1 {e_show:{_af2}} Ma ({sig_lbl})",
                ha='right', va='bottom', fontsize=8, color='#666',
                transform=ax.transAxes, clip_on=True, zorder=5)

    # ── Export helper ─────────────────────────────────────────────────────────
    def save(self, path):
        dpi = 300 if path.lower().endswith('.png') else 150
        self.fig.savefig(path, dpi=dpi, bbox_inches='tight',
                         facecolor='white', edgecolor='none')


# --------------------------------------------------------------------------
# Overlay canvas (stacked aliquots)
# --------------------------------------------------------------------------

SAMPLE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def _draw_overlay_figure(fig, sessions, age_range, global_cfg=None, combined_e1s=None,
                         combined_e1s_ana=None, combined_e1s_ext=None,
                         combined_mean=None, combined_r_dict=None,
                         display_1sigma=False, age_dec=2,
                         panel_layout=None, axes_out=None, cak_label='Ca/K',
                         show_legend=True, title='', show_bracket=True,
                         show_err_box=True, legend_title='', legend_mono=False,
                         bracket_at=None, gaps=None, text_k=1.0,
                         label_top=False):
    """Draw the three-panel overlay spectrum into *fig*.

    Pure function, no Qt, no FigureCanvas, no draw() calls.
    Works on any matplotlib Figure regardless of backend.
    combined_e1s / _ana / _ext are 1σ; the display scale is applied here.
    combined_mean: age from the active combining method (WM + √MSWD or GLS + ML); if None,
                      falls back to the simple per-step weighted mean.
    combined_e1s: internal tier (+J_Rstd), shown in the combined bracket label
    combined_e1s_ana: analytical tier (no J), shown as 'Analytical' in bottom-left
    combined_e1s_ext: external tier
    combined_r_dict: full result dict from _combined_plateau(); used to show the
                      correct n/MSWD/p/ω for whichever method is active.
    bracket_at: age (Ma) for the combined-age line, None = automatic.
    """
    axes = build_spectrum_axes(fig, panel_layout, gaps=gaps)
    if axes_out is not None:
        axes_out.update(axes)
    ax_p40, ax_cak, ax_age = axes['p40'], axes['cak'], axes['age']

    if not sessions:
        ax_age.text(0.5, 0.5, 'No data loaded',
                    transform=ax_age.transAxes,
                    ha='center', va='center', fontsize=12, color='#aaa')
        return ax_p40, ax_cak, ax_age

    # Only count non-excluded sessions for y-limits
    active_sessions = [sess for sess in sessions if not sess.get('excluded', False)]
    all_act = [s for sess in active_sessions
               for s in sess.get('steps_c', [])]
    yr_p40  = _p40_ylim([s['pct40'] for s in all_act])
    yr_cak  = _cak_ylim(
        [s['caK']              for s in all_act],
        [s.get('err_cak', 0.0) for s in all_act],
        gas_pct=[s.get('pct39Ar') for s in all_act]) if all_act else (0.0, 10.0)

    def _draw_boxes(ax, steps_c, val_key, err_key, yr, color, scale=1.0):
        yr_span    = yr[1] - yr[0]
        fixed_half = yr_span * 0.017
        top        = yr[1] + yr_span
        for i, s in enumerate(steps_c):
            b = _box_bounds(s, val_key, err_key, scale, fixed_half, top)
            if b is None:
                continue
            lo_i, hi_i = b
            status = s.get('status', 'normal')
            exc    = (status == 'excluded')
            ec = '#aaaaaa' if exc else color
            lw = 0.7       if exc else 1.2
            ls = '--'      if exc else '-'
            bw = max(s['ce'] - s['cs'], 0.0)
            ax.add_patch(Rectangle(
                (s['cs'], lo_i), bw, hi_i - lo_i,
                facecolor='none', edgecolor=ec,
                linewidth=lw, linestyle=ls, zorder=2,
            ))
            nx = steps_c[i + 1] if i + 1 < len(steps_c) else None
            nb = (_box_bounds(nx, val_key, err_key, scale, fixed_half, top)
                  if nx else None)
            if nb:
                gap_lo = min(hi_i, nb[1])
                gap_hi = max(lo_i, nb[0])
                if gap_lo < gap_hi:
                    nx_exc = nx.get('status', 'normal') == 'excluded'
                    lc = '#aaaaaa' if (exc or nx_exc) else color
                    ax.plot([s['ce'], s['ce']], [gap_lo, gap_hi],
                            color=lc, linewidth=lw * 0.8, zorder=2)

    _ov_scale = 1.0 if display_1sigma else 2.0
    legend_handles = []
    for si, sess in enumerate(sessions):
        # the export passes its own colours (one per drawn aliquot)
        color   = sess.get('color') or SAMPLE_COLORS[si % len(SAMPLE_COLORS)]
        steps_c = sess.get('steps_c', [])
        # Skip excluded aliquots entirely, not drawn on the diagram
        if sess.get('excluded', False):
            continue
        _draw_boxes(ax_p40, steps_c, 'pct40',   None,      yr_p40,    color)
        _draw_boxes(ax_cak, steps_c, 'caK',     'err_cak', yr_cak,    color,
                    scale=_ov_scale)
        _draw_boxes(ax_age, steps_c, 'age',     'err1s',   age_range, color,
                    scale=_ov_scale)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2,
                                     label=(sess.get('legend_label')
                                            or sess.get('title', f'#{si+1}'))))

    # Only include non-excluded sessions in the combined plateau bracket
    all_plat = [s for sess in active_sessions
                for i, s in enumerate(sess.get('steps_c', []))
                if i in sess.get('pset', set())]
    if all_plat:
        r = weighted_mean([s['age']   for s in all_plat],
                          [s['err1s'] for s in all_plat])
        if r is not None:
            if combined_e1s is not None:
                e1s = combined_e1s
            else:
                cfg = global_cfg or {}
                e1s = scale_sem(r['sem'], r['mswd'], r['prob'], r['n'],
                                cfg.get('errMode', 'sem_mswd_2s'), cfg.get('pCut', 5.0))
            yr_span = age_range[1] - age_range[0]
            bx1     = min(s['cs'] for s in all_plat)
            bx2     = max(s['ce'] for s in all_plat)
            max_top = max(s['age'] + s['err1s'] * _ov_scale for s in all_plat)
            _scale   = _ov_scale
            _sig_lbl = '1σ' if display_1sigma else '2σ'
            _e_show  = e1s * _scale
            _af = f'.{age_dec}f'
            _age_show = combined_mean if combined_mean is not None else r['mean']
            # Use n/MSWD/p/ω from the active combining method when available.
            if combined_r_dict is not None:
                _cd      = combined_r_dict
                _method  = _cd.get('method', '')
                _n_val   = _cd.get('n', r['n'])
                _mswd_v  = _cd.get('mswd', r['mswd'])
                _prob_v  = _cd.get('prob', r['prob'])
                if 'aliquots' in _method:
                    _n_str = f"n={_n_val} aliquots"
                    _od    = _cd.get('overdispersion', 0.0)
                    _od_str = (f", ω={_od * _scale:.2f} Ma"
                               if ('GLS + ML' in _method and _cd.get('overdispersed'))
                               else "")
                else:
                    _n_str  = f"n={_n_val} steps"
                    _od_str = ""
                lbl = (f"Combined: {_age_show:{_af}} ± {_e_show:{_af}} Ma ({_sig_lbl})\n"
                       f"{_n_str}, MSWD={_mswd_v:.2f}, p={_prob_v:.2f}{_od_str}")
            else:
                lbl = (f"Combined: {_age_show:{_af}} ± {_e_show:{_af}} Ma ({_sig_lbl})\n"
                       f"n={r['n']}, MSWD={r['mswd']:.2f}, p={r['prob']:.2f}")
            # The bracket is placed only now that the label exists: its height
            # (two lines) sets how far down the bracket has to sit for the
            # whole summary to stay inside the panel.
            _at = bracket_at
            if label_top and _at is None:
                # Big text: just above the tallest step, with a text line of
                # room under the bracket for the statistics
                try:
                    _fl = (8.5 * (text_k or 1.0) * 1.45 * float(fig.dpi) / 72.0
                           / float(ax_age.bbox.height))
                    _at = max_top + yr_span * (0.02 + _fl)
                except Exception:
                    _at = age_range[1]
            by = _bracket_y_in_axes(ax_age, age_range, max_top,
                                    n_lines=1 if label_top else lbl.count('\n') + 1,
                                    fontsize=8.5 * (text_k or 1.0),
                                    at=_at)
            th = yr_span * 0.012
            if show_bracket:
                ax_age.plot([bx1, bx1, bx2, bx2],
                            [by - th, by, by, by - th],
                            color='#111', linewidth=1.8, zorder=5,
                            solid_capstyle='butt')
            # Center the summary on the axes (not the bracket midpoint) so the
            # longer label with ω never runs past the left or right border.
            # x in axes fraction, y in data coords.
            from matplotlib.transforms import blended_transform_factory
            _lbl_tr = blended_transform_factory(ax_age.transAxes, ax_age.transData)
            if show_bracket:
                _parts = lbl.split('\n', 1)
                if label_top and len(_parts) == 2:
                    # Big text: combined age above the line, statistics below
                    ax_age.text(0.5, by + yr_span * 0.008, _parts[0],
                                ha='center', va='bottom', fontsize=8.5,
                                fontweight='bold', color='#111',
                                transform=_lbl_tr, clip_on=True, zorder=6)
                    ax_age.text(0.5, by - yr_span * 0.008, _parts[1],
                                ha='center', va='top', fontsize=8.5,
                                fontweight='bold', color='#111',
                                transform=_lbl_tr, clip_on=True, zorder=6)
                else:
                    ax_age.text(0.5, by + yr_span * 0.008, lbl,
                                ha='center', va='bottom', fontsize=8.5,
                                fontweight='bold', color='#111',
                                transform=_lbl_tr, clip_on=True, zorder=6)
            _e1si = (combined_e1s_ana if combined_e1s_ana is not None else e1s) * _scale
            _e1se = (combined_e1s_ext if combined_e1s_ext is not None else e1s) * _scale
            if show_err_box:
                ax_age.text(0.02, 0.02,
                            f"Analytical: ±{_e1si:{_af}} Ma ({_sig_lbl})\n"
                            f"External: ±{_e1se:{_af}} Ma ({_sig_lbl})",
                            ha='left', va='bottom', fontsize=8, color='#555',
                            transform=ax_age.transAxes, clip_on=True, zorder=5)

    if legend_handles and show_legend:
        # monospace when ages are listed, so they line up in one column
        _lprop = ({'family': ['Courier New', 'DejaVu Sans Mono', 'monospace'],
                   'size': 7.5} if legend_mono else {'size': 7.5})
        ax_age.legend(handles=legend_handles, loc='lower right',
                      prop=_lprop, framealpha=0.88,
                      title=legend_title or None, title_fontsize=7.5).set_zorder(20)

    ax_p40.set_ylabel('%⁴⁰Ar*',               fontsize=9)
    ax_cak.set_ylabel(cak_label,               fontsize=9)
    ax_age.set_ylabel('Apparent Age (Ma)',      fontsize=9)
    ax_age.set_xlabel('Cumulative %³⁹Ar Released', fontsize=9)

    ax_p40.set_ylim(yr_p40)
    ax_cak.set_ylim(yr_cak)
    ax_age.set_ylim(age_range)
    for ax in (ax_p40, ax_cak, ax_age):
        ax.set_xlim(0.0, 100.0)
        ax.tick_params(axis='both', labelsize=8)
        ax.yaxis.grid(True, linewidth=0.5, color='#f0f0f0', zorder=0)
        ax.spines['top'].set(linewidth=0.5,  color='#ccc')
        ax.spines['right'].set(linewidth=0.5, color='#ccc')
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
    ax_p40.yaxis.set_major_locator(ticker.MultipleLocator(25))
    if title:
        (axes.get('_spectrum') or [ax_p40])[0].set_title(
            title, fontsize=10, fontweight='bold', pad=4)
    finish_spectrum_axes(axes)
    return ax_p40, ax_cak, ax_age


# Atmospheric ⁴⁰Ar/³⁶Ar, Lee et al. (2006).  Drawn as the reference line on the
# isochron diagrams: an intercept on it means no excess argon.
ATM_40_36 = 298.56


class IsochronCanvas(FigureCanvas):
    """Single-panel isochron diagram: normal or inverse, toggled by the user."""

    # Palette for per-aliquot colours in stack mode
    _PALETTE = ['#2980b9', '#e74c3c', '#27ae60', '#8e44ad',
                '#d35400', '#16a085', '#c0392b', '#2c3e50']

    def _tier_text(self, ana, int_, ext, age, wrapped_gap, short=False):
        """(headline error, its %-text, tag, tier line or None) for the stats box,
        following the export's error-tier choice."""
        tier = getattr(self, '_err_tier', 'int')
        val = {'ana': ana, 'int': int_, 'ext': ext}.get(tier, int_)
        show_line = getattr(self, '_err_line', True)
        tag = '' if show_line else {'ana': ', ana.', 'int': ', int.',
                                    'ext': ', ext.'}.get(tier, '')
        line = None
        if show_line and ana is not None:
            g = wrapped_gap
            n = ('Ana', 'Int', 'Ext') if short else ('Analytical', 'Internal', 'External')
            line = (f'{n[0]}: \u00b1{ana:.3f}{g}{n[1]}: \u00b1{int_:.3f}{g}'
                    f'{n[2]}: \u00b1{ext:.3f} Ma' if short else
                    f'{n[0]}: \u00b1{ana:.3f} Ma{g}{n[1]}: \u00b1{int_:.3f} Ma{g}'
                    f'{n[2]}: \u00b1{ext:.3f} Ma')
        return val, tag, line

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 9))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self._showing_inverse = True   # default: show inverse isochron
        self._iso_zoom = 'data'        # 'data' | 'intercept'
        self._build_axes()

    def _build_axes(self):
        # _target_ax is set by the export dialog when the isochron is drawn as
        # one panel of a bigger figure; on screen it is None.
        tgt = getattr(self, '_target_ax', None)
        if tgt is not None:
            tgt.clear()
            self.ax = tgt
            return
        self.fig.clear()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(
            left=0.13, right=0.97, top=0.93, bottom=0.09)

    def _apply_user_lims(self, ax):
        """Axis ranges typed in the export dialog (None = keep automatic).
        Applied before the regression line, which runs edge to edge."""
        lim = getattr(self, '_user_lims', None) or {}
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        nx0, nx1 = lim.get('x0'), lim.get('x1')
        ny0, ny1 = lim.get('y0'), lim.get('y1')
        nx = (x0 if nx0 is None else nx0, x1 if nx1 is None else nx1)
        ny = (y0 if ny0 is None else ny0, y1 if ny1 is None else ny1)
        if nx[1] > nx[0]:
            ax.set_xlim(*nx)
        if ny[1] > ny[0]:
            ax.set_ylim(*ny)

    def _stats_xy(self, inverse):
        """(x, y, ha, va) in axes fraction for the statistics box: the corner
        chosen in the export dialog, else lower left (inverse) / right (normal)."""
        pos = getattr(self, '_stats_pos', 'auto') or 'auto'
        if pos == 'auto':
            pos = 'll' if inverse else 'lr'
        left, upper = pos[1] == 'l', pos[0] == 'u'
        return (0.03 if left else 0.97, 0.97 if upper else 0.03,
                'left' if left else 'right', 'top' if upper else 'bottom')

    @staticmethod
    def _on_y_axis(ax, y):
        """True when the intercept (0, y) sits on the visible y-axis: it is then
        drawn unclipped, so the marker is not cut in half by the axis edge."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return min(x0, x1) <= 0.0 <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1)

    def _labels_on(self):
        return (not getattr(self, '_compact', False)
                and getattr(self, '_show_step_labels', True))

    def _style_ax(self, ax):
        ax.tick_params(axis='both', labelsize=8)
        ax.yaxis.grid(True, linewidth=0.4, color='#f0f0f0', zorder=0)
        ax.xaxis.grid(True, linewidth=0.4, color='#f0f0f0', zorder=0)
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)
        ax.spines['top'].set(linewidth=0.5, color='#ccc')
        ax.spines['right'].set(linewidth=0.5, color='#ccc')

    @staticmethod
    def _error_ellipse(ax, cx, cy, sx, sy, rho, color, alpha=0.8, n_std=1.0):
        """Draw a 1σ error ellipse using eigendecomposition of the 2×2 covariance."""
        from matplotlib.patches import Ellipse as _Ellipse
        cov = np.array([[sx**2, rho*sx*sy],
                        [rho*sx*sy, sy**2]])
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0.0)
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        width  = 2.0 * n_std * np.sqrt(vals[1])
        height = 2.0 * n_std * np.sqrt(vals[0])
        el = _Ellipse((cx, cy), width, height, angle=angle,
                      edgecolor=color, facecolor='none',
                      linewidth=0.9, alpha=alpha, zorder=4)
        ax.add_patch(el)

    @staticmethod
    def _confidence_band(ax, xfit, b, a, cov_ab, slope_1s, intercept_1s, color, scale=1.0):
        """Draw the ±2σ confidence envelope around the regression line, times
        *scale* (√MSWD when MSWD > 1, as the isochron age and trapped ratio)."""
        yfit = a + b * xfit
        # Variance of predicted y: var(y) = var(a) + x²·var(b) + 2x·cov(a,b)
        var_y = (intercept_1s**2
                 + xfit**2 * slope_1s**2
                 + 2.0 * xfit * cov_ab)
        var_y = np.maximum(var_y, 0.0)
        dy2 = 2.0 * scale * np.sqrt(var_y)
        ax.fill_between(xfit, yfit - dy2, yfit + dy2,
                        alpha=0.12, color=color, zorder=2)
        ax.plot(xfit, yfit - dy2, '--', color=color, linewidth=0.7,
                alpha=0.6, zorder=3)
        ax.plot(xfit, yfit + dy2, '--', color=color, linewidth=0.7,
                alpha=0.6, zorder=3)

    @staticmethod
    def spread_stats(xs, ys, fit, inverse=True):
        """How far the regressed points spread along the mixing line.

        A measure of coverage, not of scatter: how much of the atmospheric →
        radiogenic range the steps cover, which sets how well the intercept
        (the trapped ratio) is constrained.  Steps that are
        all ~100% radiogenic sit in one corner and the intercept is an
        extrapolation; steps spanning a wide range of ⁴⁰Ar* pin both ends.

        Returned: the radiogenic yield of each point implied by the diagram
        itself (inverse: 1 − y·⁴⁰Ar/³⁶Ar_atm; normal: 1 − ⁴⁰Ar/³⁶Ar_atm/y) as
        min/max in per cent, and, for the inverse diagram, the covered fraction
        of the atmosphere-to-radiogenic lever, x_range / x_intercept.  None
        when there is nothing to measure.
        """
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        if xs.size < 2:
            return None
        with np.errstate(divide='ignore', invalid='ignore'):
            rad = (1.0 - ys * ATM_40_36) if inverse else (1.0 - ATM_40_36 / ys)
        rad = rad[np.isfinite(rad)]
        out = {'rad_min': float(np.min(rad)) * 100.0 if rad.size else None,
               'rad_max': float(np.max(rad)) * 100.0 if rad.size else None,
               'lever': None}
        if inverse and fit and fit.get('slope'):
            # x-intercept of the fit = the fully radiogenic end member
            x_int = -fit['intercept'] / fit['slope'] if fit['slope'] else None
            if x_int and x_int > 0:
                out['lever'] = float((xs.max() - xs.min()) / x_int)
        return out

    @staticmethod
    def spread_text(sp):
        """One line describing the spread, or '' when it cannot be measured."""
        if not sp or sp.get('rad_min') is None:
            return ''
        txt = (f"Spread: \u2074\u2070Ar* {sp['rad_min']:.0f}\u2013"
               f"{sp['rad_max']:.0f}%")
        if sp.get('lever') is not None:
            txt += f",  lever {sp['lever']:.2f}"
        return txt

    def _draw_init_label(self, ax, init_ratio, init_2s, inverse=True,
                         extra_lines=()):
        """One-line label with the trapped ⁴⁰Ar/³⁶Ar from the y-intercept.

        Independent of the statistics box, so an inset (or a plot with the box
        switched off) can still carry the initial ratio.  Atmospheric argon is
        298.56 (Lee et al. 2006); a value within error of it means the trapped
        component is atmospheric.
        """
        lines = []
        if getattr(self, '_show_init_label', False) and init_ratio is not None:
            t = f'\u2074\u2070Ar/\u00b3\u2076Ar\u2080 = {init_ratio:.1f}'
            if init_2s:
                t += f' \u00b1 {init_2s:.1f} (2\u03c3)'
            lines.append(t)
        lines += [t for t in extra_lines if t]
        if not lines:
            return
        txt = '\n'.join(lines)
        fs = 6.0 if getattr(self, '_compact', False) else 8.0
        # the data run top-left to bottom-right on an inverse isochron and the
        # other way on a normal one, so the free corner differs
        x, ha = (0.97, 'right') if inverse else (0.03, 'left')
        ax.text(x, 0.97, txt, transform=ax.transAxes, ha=ha, va='top',
                fontsize=fs, color='#444', zorder=6, clip_on=True,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#ccc', alpha=0.85))

    def _show_atmospheric(self, ax, inverse=True):
        """With the atmospheric reference on, raise the top of the y axis to
        it when it lies above (an intercept below atmosphere would otherwise
        leave the line off the plot)."""
        atm = getattr(self, '_atm_ratio', None)
        if not atm or atm <= 0:
            return
        y = (1.0 / float(atm)) if inverse else float(atm)
        lo, hi = ax.get_ylim()
        if y >= hi:
            ax.set_ylim(lo, y + 0.06 * (y - lo))

    def _draw_atmospheric(self, ax, inverse=True):
        """Mark atmospheric argon on the y axis, when a value is set.

        _atm_ratio is the atmospheric ⁴⁰Ar/³⁶Ar (295.5 Nier 1950, 298.56 Lee
        et al. 2006).  An inverse isochron carries ³⁶Ar/⁴⁰Ar, so the reference
        plots at 1/atm; a normal isochron plots it directly.  A sample with no
        excess argon has its intercept on this line.
        """
        atm = getattr(self, '_atm_ratio', None)
        if not atm or atm <= 0:
            return
        atm = float(atm)
        y = (1.0 / atm) if inverse else atm
        lo, hi = ax.get_ylim()
        if not (lo <= y <= hi):
            return
        ax.axhline(y, color='#777777', linewidth=0.8,
                   linestyle=(0, (4, 3)), zorder=2)
        ax.plot([ax.get_xlim()[0]], [y], marker='o', markersize=4,
                markerfacecolor='none', markeredgecolor='#777777',
                zorder=6, clip_on=True)
        if not getattr(self, '_compact', False):
            # keep clear of the ⁴⁰Ar/³⁶Ar₀ label, which takes the other corner
            # a line near the top of the panel gets its caption underneath, so
            # the corner labels stay readable
            high = (y - lo) / (hi - lo) > 0.75 if hi > lo else False
            # caption always at the right-hand end of the line
            x_at, dx, ha = ax.get_xlim()[1], -4, 'right'
            dy, va = (-3, 'top') if high else (3, 'bottom')
            ax.annotate(f'atmospheric \u2074\u2070Ar/\u00b3\u2076Ar = {atm:g}',
                        xy=(x_at, y), xytext=(dx, dy),
                        textcoords='offset points', ha=ha, va=va,
                        fontsize=7, color='#777777', zorder=6, clip_on=True)

    def _draw_one_panel(self, ax, steps, x_key, sx_key, y_key, sy_key, ri_key,
                        xlabel, ylabel, title, color='#2980b9',
                        normal_isochron=True, J=None, lam=LAMBDA_DEFAULT,
                        zoom='auto', cfg=None):
        """Draw data, York regression, age annotation on one subplot."""

        incl = [(s[x_key], s[sx_key], s[y_key], s[sy_key],
                 s.get(ri_key, 0.0) or 0.0)
                for s in steps
                if s.get(x_key) is not None and s.get(sx_key) is not None
                and s.get(y_key) is not None and s.get(sy_key) is not None
                and s[x_key] > 0 and s[y_key] > 0
                and s.get('status', 'normal') != 'excluded']
        if not incl:
            ax.text(0.5, 0.5, 'No isochron data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='#aaa')
            ax.set_title(title, fontsize=9, pad=4)
            return

        xs, sxs, ys, sys_, ris = map(np.array, zip(*incl))

        # York regression (included steps only)
        fit = york_regression(xs, sxs, ys, sys_, ris)

        # ── Draw included steps: ellipses + centre dots + labels ─────────────
        for xi, sxi, yi, syi, ri in incl:
            self._error_ellipse(ax, xi, yi, sxi, syi, ri, color)
        ax.scatter(xs, ys, s=20, color=color, zorder=5, clip_on=True, linewidths=0)
        lbl_steps = [s for s in steps
                     if s.get(x_key) is not None and s.get(sx_key) is not None
                     and s.get(y_key) is not None and s.get(sy_key) is not None
                     and s[x_key] > 0 and s[y_key] > 0
                     and s.get('status', 'normal') != 'excluded']
        for s, xi, yi in zip(lbl_steps if self._labels_on() else [], xs, ys):
            ax.annotate(s.get('label', ''), xy=(xi, yi),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=8.5, ha='left', va='bottom',
                        color='#222', fontweight='bold',
                        clip_on=True, zorder=6)

        # ── Draw excluded steps: red, not used in regression ─────────────────
        _excl_col = '#e74c3c'
        excl = [s for s in steps
                if s.get(x_key) is not None and s.get(sx_key) is not None
                and s.get(y_key) is not None and s.get(sy_key) is not None
                and s[x_key] > 0 and s[y_key] > 0
                and s.get('status', 'normal') == 'excluded']
        if excl:
            for s in excl:
                self._error_ellipse(ax, s[x_key], s[y_key],
                                    s[sx_key], s[sy_key],
                                    s.get(ri_key, 0.0) or 0.0,
                                    _excl_col, alpha=0.5)
            exc_xs = np.array([s[x_key] for s in excl])
            exc_ys = np.array([s[y_key] for s in excl])
            ax.scatter(exc_xs, exc_ys, s=20, color=_excl_col,
                       zorder=5, clip_on=True, linewidths=0, alpha=0.7)
            for s in (excl if self._labels_on() else []):
                ax.annotate(s.get('label', ''), xy=(s[x_key], s[y_key]),
                            xytext=(4, 4), textcoords='offset points',
                            fontsize=8.5, ha='left', va='bottom',
                            color=_excl_col, fontweight='bold',
                            clip_on=True, zorder=6)

        # ── Set axis limits before drawing the regression line ─────────────
        iso_zoom = getattr(self, '_iso_zoom', 'data')
        if zoom == 'plateau':
            xm = (xs.max() - xs.min()) * 0.12 or xs.max() * 0.05
            ym = (ys.max() - ys.min()) * 0.12 or ys.max() * 0.05
            y_top = (ys + sys_).max() + ym
            if iso_zoom == 'intercept' and fit is not None:
                y_int = fit['intercept'] + 2 * fit['intercept_1s'] * _iso_scale(fit)
                y_top = max(y_top, y_int + ym)
            ax.set_xlim(0.0, (xs + sxs).max() + xm)
            ax.set_ylim(0.0, y_top)
        else:
            ax.autoscale_view()
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_xlim(0.0, x1)
            if iso_zoom == 'intercept' and fit is not None:
                y_int = fit['intercept'] + 2 * fit['intercept_1s'] * _iso_scale(fit)
                margin = (y1 - 0) * 0.08 or y_int * 0.08
                y1 = max(y1, y_int + margin)
            ax.set_ylim(0.0, y1)
        self._show_atmospheric(ax, inverse=normal_isochron)
        self._apply_user_lims(ax)
        ax.autoscale(enable=False)   # freeze so regression line won't expand limits

        # ── Regression line from one axis edge to the other ──────────────────
        if fit is not None:
            b, a = fit['slope'], fit['intercept']
            x0, x1 = ax.get_xlim()
            xfit = np.linspace(x0, x1, 300)
            # regression line, band and intercept are grey, as in the stacked
            # isochron, whatever the colour of the points
            ax.plot(xfit, a + b * xfit, '-', color='#222',
                    linewidth=1.2, zorder=3, clip_on=True)
            # band and intercept bar at the level of the reported numbers:
            # 2σ, times √MSWD when MSWD > 1 (arar_math.isochron_age)
            _mf = _iso_scale(fit)
            self._confidence_band(ax, xfit, b, a,
                                  fit['cov_ab'], fit['slope_1s'],
                                  fit['intercept_1s'], '#333', scale=_mf)

            # Initial ⁴⁰Ar/³⁶Ar₀ from Y-intercept (no J required)
            sa_int = fit['intercept_1s']
            if normal_isochron:     # inverse panel: Y = ³⁶Ar/⁴⁰Ar
                if a > 0:
                    init_ratio    = 1.0 / a
                    init_ratio_2s = 2.0 * sa_int / (a ** 2)
                else:
                    init_ratio = init_ratio_2s = None
            else:                   # normal panel:  Y = ⁴⁰Ar/³⁶Ar
                init_ratio    = a
                init_ratio_2s = 2.0 * sa_int

            # Y-intercept marker at (0, a): shows initial ratio on the plot
            if init_ratio is not None and getattr(self, '_show_intercept', True):
                ax.errorbar([0], [a], yerr=[[2.0 * _mf * sa_int], [2.0 * _mf * sa_int]],
                            fmt='D', color='#333', markersize=5,
                            capsize=3, elinewidth=1.0, zorder=7,
                            clip_on=not self._on_y_axis(ax, a))

            # Age and three tiers from the shared isochron rule (isochron_tiers)
            _it = isochron_tiers(steps, fit, J, lam, normal_isochron, cfg)
            age_ma = _it['age_ma']
            if _it['init_ratio'] is not None:
                init_ratio_2s = 2.0 * _it['init_ratio_1s']
            age_2s = age_2s_ana = age_2s_int = age_2s_ext = None
            if _it['e1s_ana'] is not None:
                age_2s = age_2s_ana = 2.0 * _it['e1s_ana']
                age_2s_int = 2.0 * _it['e1s_int']
                age_2s_ext = 2.0 * _it['e1s_ext']

            # Annotation box
            lines = []
            if age_ma is not None:
                # Primary displayed error = internal (regression + J), matching ArArCALC
                _v, _tag, _ = self._tier_text(age_2s_ana, age_2s_int, age_2s_ext,
                                              age_ma, '  ')
                pct = (_v / age_ma * 100.0) if age_ma > 0 else 0.0
                lines.append(f'Age = {age_ma:.2f} ± {_v:.2f} Ma (2σ, {pct:.1f}%{_tag})')
            if init_ratio is not None and not getattr(self, '_show_init_label', False):
                lines.append(
                    f'⁴⁰Ar/³⁶Ar₀ = {init_ratio:.1f} ± {init_ratio_2s:.1f} (2σ)')
            lines.append(
                f'MSWD = {fit["mswd"]:.2f},  p = {fit["prob"]:.3f}  '
                f'(n = {fit["n"]})')
            _spread_line = ''
            if getattr(self, '_show_spread', False):
                _spread_line = self.spread_text(
                    self.spread_stats(xs, ys, fit, inverse=normal_isochron))
                if _spread_line and getattr(self, '_show_stats', True) \
                        and not getattr(self, '_compact', False):
                    lines.append(_spread_line)
                    _spread_line = ''
            self._pending_spread = _spread_line
            if age_2s is not None:
                _gap = '\n' if getattr(self, '_stats_stacked', False) else '  '
                _, _, _tl = self._tier_text(age_2s_ana, age_2s_int, age_2s_ext,
                                            age_ma, _gap)
                if _tl:
                    lines.append(_tl)
            if lines and getattr(self, '_show_stats', True) \
                    and not getattr(self, '_compact', False):
                # default corner: inverse → lower left, normal → lower right
                _sx, _sy, _ha, _va = self._stats_xy(normal_isochron)
                ax.text(_sx, _sy, '\n'.join(lines),
                        transform=ax.transAxes, ha=_ha, va=_va,
                        fontsize=getattr(self, '_stats_fs', 7.5),
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='#ccc',
                                  alpha=0.85),
                        zorder=6)

        self._draw_atmospheric(ax, inverse=normal_isochron)
        self._draw_init_label(ax, locals().get('init_ratio'),
                              locals().get('init_ratio_2s'),
                              inverse=normal_isochron,
                              extra_lines=(getattr(self, '_pending_spread', ''),))
        self._pending_spread = ''
        _fs = 6.0 if getattr(self, '_compact', False) else 8.0
        ax.set_xlabel(xlabel, fontsize=_fs)
        ax.set_ylabel(ylabel, fontsize=_fs)
        ax.set_title(title, fontsize=_fs + 1.0, pad=3)
        self._style_ax(ax)
        if getattr(self, '_compact', False):
            ax.tick_params(axis='both', labelsize=5.5)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

    @staticmethod
    def _iso_age_ma(fit, J, lam, inverse):
        """Isochron age (Ma) of a York fit, or None."""
        if fit is None:
            return None
        b, a = fit['slope'], fit['intercept']
        X = (-b / a * J) if inverse else (b * J)
        if (inverse and a <= 0) or X <= 0:
            return None
        return math.log(1.0 + X) / lam * 1e-6

    def _group_j_sigma(self, xs, sxs, ys, sys_, ris, gis, J, lam, inverse, age_ma):
        """1σ (Ma) J contribution of a stacked isochron with each aliquot's
        own J error, or None when they are not known.

        Every aliquot's ³⁹Ar was put on J_ref with J_ref/J_i, so a change in
        J_i moves only that aliquot's points (x ∝ 1/J_i).  g_i = ∂t/∂ln J_i ·
        σ_Ji/J_i is found by refitting with that shift, and σ² = gᵀρg with ρ
        the stack's hole / disk J correlation.  All aliquots on one J gives
        ∂t/∂ln J · σ_J/J, the single-J value."""
        rel = getattr(self, '_group_jrel', None)
        rho = getattr(self, '_group_rho', None)
        if not rel or rho is None or age_ma is None:
            return None
        n = len(rel)
        rho = np.asarray(rho, float)
        if rho.shape != (n, n) or (gis.size and gis.max() >= n):
            return None
        d = 1e-5                      # step in ln J_i, central difference
        g = np.zeros(n)
        for gi in range(n):
            m = gis == gi
            if not m.any() or not rel[gi]:
                continue
            t = []
            for k in (math.exp(-d), math.exp(d)):     # J_i·e^{±d}: x·e^{∓d}
                t.append(self._iso_age_ma(
                    york_regression(np.where(m, xs * k, xs), np.where(m, sxs * k, sxs),
                                    ys, sys_, ris), J, lam, inverse))
            if None in t:
                return None
            g[gi] = (t[0] - t[1]) / (2.0 * d) * rel[gi]
        return float(math.sqrt(max(float(g @ rho @ g), 0.0)))

    def redraw(self, steps_list, J=None,
               lam=LAMBDA_DEFAULT, title='', colors=None, zoom='auto', cfg=None):
        """Redraw one isochron panel (normal or inverse) based on _showing_inverse.

        steps_list : list-of-list (one list per aliquot) or flat list of steps.
        """
        self._build_axes()

        # Normalise to list-of-lists
        if steps_list and not isinstance(steps_list[0], list):
            steps_list = [steps_list]

        if colors is None:
            colors = self._PALETTE

        all_steps = [s for grp in steps_list for s in grp]
        n_alq = len(steps_list)
        show_inv = getattr(self, '_showing_inverse', True)

        # Select panel parameters
        if show_inv:
            # Inverse isochron: X=39/40, Y=36/40, ρ=iso_iri (~0.02)
            x_k, sx_k = 'iso_nx', 'iso_nsx'
            y_k, sy_k = 'iso_ny', 'iso_nsy'
            ri_k  = 'iso_iri'
            xl, yl = '³⁹Ar(k)/⁴⁰Ar', '³⁶Ar/⁴⁰Ar'
            ttl   = 'Inverse Isochron'
            is_norm_flag = True    # normal_isochron param: True = inverse panel
        else:
            # Normal isochron: X=39/36, Y=40/36, ρ=iso_nri (~0.99)
            x_k, sx_k = 'iso_ix', 'iso_isx'
            y_k, sy_k = 'iso_iy', 'iso_isy'
            ri_k  = 'iso_nri'
            xl, yl = '³⁹Ar(k)/³⁶Ar', '⁴⁰Ar/³⁶Ar'
            ttl   = 'Normal Isochron'
            is_norm_flag = False

        if n_alq == 1:
            self._draw_one_panel(
                self.ax, all_steps,
                x_k, sx_k, y_k, sy_k, ri_k,
                xl, yl, ttl,
                color=colors[0], normal_isochron=is_norm_flag,
                J=J, lam=lam, zoom=zoom, cfg=cfg)
        else:
            # Multi-aliquot: colour ellipses + dots + labels per aliquot,
            # single combined York fit with external error annotation
            for gi, grp in enumerate(steps_list):
                col = colors[gi % len(colors)]
                gxs, gys = [], []
                for s in grp:
                    if (s.get(x_k) is None or s.get(y_k) is None
                            or s[x_k] <= 0 or s[y_k] <= 0
                            or s.get('status', 'normal') == 'excluded'):
                        continue
                    self._error_ellipse(self.ax, s[x_k], s[y_k],
                                        s.get(sx_k) or 0,
                                        s.get(sy_k) or 0,
                                        s.get(ri_k) or 0, col)
                    gxs.append(s[x_k]); gys.append(s[y_k])
                if gxs:
                    self.ax.scatter(gxs, gys, s=20, color=col,
                                    zorder=5, clip_on=True, linewidths=0)
                for s in (grp if self._labels_on() else []):
                    if (s.get(x_k) is None or s.get(y_k) is None
                            or s[x_k] <= 0 or s[y_k] <= 0
                            or s.get('status', 'normal') == 'excluded'):
                        continue
                    self.ax.annotate(s.get('label', ''),
                                     xy=(s[x_k], s[y_k]),
                                     xytext=(4, 4), textcoords='offset points',
                                     fontsize=8.5, ha='left', va='bottom',
                                     color='#222', fontweight='bold',
                                     clip_on=True, zorder=6)

            # Steps outside the plateaus, grey as in the stacked age spectra
            # (red would read as one of the aliquot colours)
            _excl_col = '#9e9e9e'
            for s in all_steps:
                if (s.get(x_k) is None or s.get(sx_k) is None
                        or s.get(y_k) is None or s.get(sy_k) is None
                        or s[x_k] <= 0 or s[y_k] <= 0
                        or s.get('status', 'normal') != 'excluded'):
                    continue
                self._error_ellipse(self.ax, s[x_k], s[y_k],
                                    s[sx_k], s[sy_k],
                                    s.get(ri_k, 0.0) or 0.0,
                                    _excl_col, alpha=0.5)
                self.ax.scatter([s[x_k]], [s[y_k]], s=20, color=_excl_col,
                                zorder=5, clip_on=True, linewidths=0, alpha=0.7)
                if self._labels_on():
                    self.ax.annotate(s.get('label', ''), xy=(s[x_k], s[y_k]),
                                     xytext=(4, 4), textcoords='offset points',
                                     fontsize=8.5, ha='left', va='bottom',
                                     color=_excl_col, fontweight='bold',
                                     clip_on=True, zorder=6)

            pts = [(s[x_k], s[sx_k], s[y_k], s[sy_k],
                    s.get(ri_k, 0.0) or 0.0, gi)
                   for gi, grp in enumerate(steps_list) for s in grp
                   if s.get(x_k) is not None and s.get(sx_k) is not None
                   and s.get(y_k) is not None and s.get(sy_k) is not None
                   and s[x_k] > 0 and s[y_k] > 0
                   and s.get('status', 'normal') != 'excluded']
            if pts:
                xs2, sxs2, ys2, sys2_, ris2, gis2 = map(np.array, zip(*pts))
                fit2 = york_regression(xs2, sxs2, ys2, sys2_, ris2)

                # Set axis limits before drawing the regression line
                _inc = [s for s in all_steps
                        if s.get(x_k) is not None and s.get(y_k) is not None
                        and s[x_k] > 0 and s[y_k] > 0
                        and s.get('status', 'normal') != 'excluded']
                incl_xs = np.array([s[x_k] for s in _inc])
                incl_ys = np.array([s[y_k] for s in _inc])
                incl_sxs = np.array([s.get(sx_k, 0) or 0 for s in _inc])
                incl_sys = np.array([s.get(sy_k, 0) or 0 for s in _inc])
                if zoom == 'plateau' and len(incl_xs) > 0:
                    xm = (incl_xs.max() - incl_xs.min()) * 0.12 or incl_xs.max() * 0.05
                    ym = (incl_ys.max() - incl_ys.min()) * 0.12 or incl_ys.max() * 0.05
                    y_top = (incl_ys + incl_sys).max() + ym
                    if getattr(self, '_iso_zoom', 'data') == 'intercept' and fit2 is not None:
                        y_int2 = fit2['intercept'] + 2 * fit2['intercept_1s'] * _iso_scale(fit2)
                        y_top = max(y_top, y_int2 + ym)
                    self.ax.set_xlim(0.0, (incl_xs + incl_sxs).max() + xm)
                    self.ax.set_ylim(0.0, y_top)
                else:
                    # data extent plus error bars, but one wildly uncertain
                    # step must not stretch the axes: its error counts at most
                    # half the spread of the included centres
                    capx = max(float(np.ptp(incl_xs)), 0.05 * float(incl_xs.max()))
                    capy = max(float(np.ptp(incl_ys)), 0.05 * float(incl_ys.max()))
                    _tx = incl_xs + np.minimum(incl_sxs, 0.5 * capx)
                    _ty = incl_ys + np.minimum(incl_sys, 0.5 * capy)
                    x1, y1 = float(_tx.max()), float(_ty.max())
                    # a few far outliers (bad steps) would squash everything
                    # else into a corner: they run off the edge instead
                    if len(_tx) >= 8:
                        q = float(np.percentile(_tx, 90))
                        if x1 > 2.5 * q:
                            x1 = 1.3 * q
                        q = float(np.percentile(_ty, 90))
                        if y1 > 2.5 * q:
                            y1 = 1.3 * q
                    x1 *= 1.05
                    y1 *= 1.06
                    self.ax.set_xlim(0.0, x1)
                    if getattr(self, '_iso_zoom', 'data') == 'intercept' and fit2 is not None:
                        y_int2 = fit2['intercept'] + 2 * fit2['intercept_1s'] * _iso_scale(fit2)
                        y1 = max(y1, y_int2 + y1 * 0.08)
                    self.ax.set_ylim(0.0, y1)
                self._show_atmospheric(self.ax, inverse=show_inv)
                self._apply_user_lims(self.ax)
                self.ax.autoscale(enable=False)

                if fit2 is not None:
                    b2, a2 = fit2['slope'], fit2['intercept']
                    x20, x21 = self.ax.get_xlim()
                    xfit2 = np.linspace(x20, x21, 300)
                    self.ax.plot(xfit2, a2 + b2 * xfit2, '-k',
                                 linewidth=1.2, zorder=3, clip_on=True)
                    _mf2 = _iso_scale(fit2)
                    self._confidence_band(self.ax, xfit2, b2, a2,
                                          fit2['cov_ab'], fit2['slope_1s'],
                                          fit2['intercept_1s'], '#333', scale=_mf2)

                    # Y-intercept marker and initial ratio
                    sa2_int = fit2['intercept_1s']
                    if show_inv and a2 > 0:
                        init2_ratio    = 1.0 / a2
                        init2_ratio_2s = 2.0 * sa2_int / (a2 ** 2)
                    elif not show_inv:
                        init2_ratio    = a2
                        init2_ratio_2s = 2.0 * sa2_int
                    else:
                        init2_ratio = init2_ratio_2s = None
                    if fit2['mswd'] > 1.0 and init2_ratio_2s is not None:
                        init2_ratio_2s *= np.sqrt(fit2['mswd'])
                    if (init2_ratio is not None
                            and getattr(self, '_show_intercept', True)):
                        self.ax.errorbar([0], [a2], yerr=[[2.0 * _mf2 * sa2_int],
                                                          [2.0 * _mf2 * sa2_int]],
                                         fmt='D', color='#333', markersize=5,
                                         capsize=3, elinewidth=1.0, zorder=7,
                                         clip_on=not self._on_y_axis(self.ax, a2))

                    lines2 = []
                    _gap = '\n' if getattr(self, '_stats_stacked', False) else '  '
                    # Shared isochron rule; the J term carries each aliquot's
                    # own J error with the stack's hole / disk ρ when known
                    _ia = isochron_age(fit2, J, lam, show_inv)
                    sJ2 = (self._group_j_sigma(xs2, sxs2, ys2, sys2_, ris2, gis2,
                                               J, lam, show_inv, _ia['age_ma'])
                           if _ia['age_ma'] is not None else None)
                    _it2 = isochron_tiers(all_steps, fit2, J, lam, show_inv, cfg,
                                          sJ_age=sJ2)
                    age2_ma = _it2['age_ma']
                    age2_sig = age2_sig_ana = age2_sig_int = age2_sig_ext = None
                    if _it2['e1s_ana'] is not None:
                        age2_sig = age2_sig_ana = 2.0 * _it2['e1s_ana']
                        age2_sig_int = 2.0 * _it2['e1s_int']
                        age2_sig_ext = 2.0 * _it2['e1s_ext']

                    if age2_ma is not None and age2_sig_int is not None:
                        _v2, _tag2, _ = self._tier_text(age2_sig_ana, age2_sig_int,
                                                        age2_sig_ext, age2_ma, _gap)
                        pct2 = (_v2 / age2_ma * 100.0) if age2_ma > 0 else 0.0
                        lines2.append(f'Age = {age2_ma:.2f} ± {_v2:.2f} Ma (2σ, {pct2:.1f}%{_tag2})')
                    if init2_ratio is not None and not getattr(self, '_show_init_label', False):
                        lines2.append(f'⁴⁰Ar/³⁶Ar₀ = {init2_ratio:.1f} ± {init2_ratio_2s:.1f} (2σ)')
                    lines2.append(
                        f'MSWD = {fit2["mswd"]:.2f},  p = {fit2["prob"]:.3f}'
                        f'  (n = {fit2["n"]})')
                    _spread2 = ''
                    if getattr(self, '_show_spread', False):
                        _spread2 = self.spread_text(
                            self.spread_stats(xs2, ys2, fit2, inverse=show_inv))
                        if _spread2 and getattr(self, '_show_stats', True) \
                                and not getattr(self, '_compact', False):
                            lines2.append(_spread2)
                            _spread2 = ''
                    self._pending_spread = _spread2
                    if age2_sig is not None:
                        _, _, _tl2 = self._tier_text(age2_sig_ana, age2_sig_int,
                                                     age2_sig_ext, age2_ma, _gap, short=True)
                        if _tl2:
                            lines2.append(_tl2)
                    if (getattr(self, '_compact', False)
                            or not getattr(self, '_show_stats', True)):
                        lines2 = []
                    if lines2:
                        _sx, _sy, _ha, _va = self._stats_xy(show_inv)
                        self.ax.text(_sx, _sy, '\n'.join(lines2),
                                     transform=self.ax.transAxes,
                                     ha=_ha, va=_va,
                                     fontsize=getattr(self, '_stats_fs', 7.5),
                                     bbox=dict(boxstyle='round,pad=0.3',
                                               facecolor='white', edgecolor='#ccc',
                                               alpha=0.85), zorder=6)

            names = getattr(self, '_legend_names', None)
            if names:
                hs = [Line2D([0], [0], marker='o', ls='none', ms=5,
                             color=colors[gi % len(colors)],
                             label=names[gi] if gi < len(names) else f'#{gi + 1}')
                      for gi in range(len(steps_list))]
                self.ax.legend(handles=hs, loc='best', fontsize=7,
                               framealpha=0.9).set_zorder(20)
            self._draw_atmospheric(self.ax, inverse=show_inv)
            self._draw_init_label(self.ax, locals().get('init2_ratio'),
                                  locals().get('init2_ratio_2s'),
                                  inverse=show_inv,
                                  extra_lines=(getattr(self, '_pending_spread', ''),))
            self._pending_spread = ''
            _fs2 = 6.0 if getattr(self, '_compact', False) else 8.0
            self.ax.set_xlabel(xl, fontsize=_fs2)
            self.ax.set_ylabel(yl, fontsize=_fs2)
            self.ax.set_title(ttl, fontsize=_fs2 + 1.0, pad=3)
            self._style_ax(self.ax)
            if getattr(self, '_compact', False):
                self.ax.tick_params(axis='both', labelsize=5.5)
                self.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                self.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

        if title:
            if getattr(self, '_target_ax', None) is None:
                self.fig.suptitle(title, fontsize=10, y=0.98)
            else:
                self.ax.set_title(title, fontsize=9, pad=4)

        if not getattr(self, '_offscreen', False):
            self.draw_idle()


class OverlayCanvas(FigureCanvas):
    """Three-panel overlay spectrum: all aliquots share the same axes."""

    def __init__(self, click_cb=None, parent=None, dblclick_cb=None):
        self.fig = Figure(figsize=(9, 9))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self._click_cb = click_cb
        self._dblclick_cb = dblclick_cb
        self._err_k = 2.0        # error bars as drawn (1σ or 2σ), for hit-testing
        self._sessions = []
        self._build_axes()
        self.mpl_connect('button_press_event', self._on_click)

    def _build_axes(self):
        # _panel_layout is set by the export dialog; on screen it is None and
        # the default three-panel stack is used.
        self._axes = build_spectrum_axes(self.fig,
                                         getattr(self, '_panel_layout', None),
                                         gaps=getattr(self, '_layout_gaps', None))
        self.ax_p40 = self._axes['p40']
        self.ax_cak = self._axes['cak']
        self.ax_age = self._axes['age']
        self.ax_iso = self._axes['iso']

    def _hit(self, event):
        """(session, step) of the step box under the mouse in the age panel,
        or None.  Where the boxes of several aliquots overlap, the one whose
        age is nearest the click; thin boxes get a few pixels of slack."""
        if event.inaxes is not self.ax_age or event.xdata is None:
            return None
        inv = self.ax_age.transData.inverted()
        slack = abs(inv.transform((0, 0))[1] - inv.transform((0, 4))[1])
        best = None
        for si, sess in enumerate(self._sessions):
            for ti, s in enumerate(sess.get('steps_c', [])):
                if s['cs'] <= event.xdata <= s['ce']:
                    eh = max(s.get('err1s', 0.0) * self._err_k, slack)
                    d = abs(event.ydata - s['age'])
                    if d <= eh and (best is None or d < best[0]):
                        best = (d, si, ti)
        return best[1:] if best else None

    def _on_click(self, event):
        cb = self._dblclick_cb if event.dblclick else self._click_cb
        if cb is None:
            return
        hit = self._hit(event)
        if hit is not None:
            cb(*hit)

    def redraw(self, sessions, age_range, global_cfg=None, combined_e1s=None,
               combined_e1s_ana=None, combined_e1s_ext=None,
               combined_mean=None, combined_r_dict=None,
               display_1sigma=False, age_dec=2, cak_label='Ca/K',
               legend_title=''):
        self._sessions = sessions
        self._err_k = 1.0 if display_1sigma else 2.0
        self.ax_p40, self.ax_cak, self.ax_age = _draw_overlay_figure(
            self.fig, sessions, age_range,
            global_cfg=global_cfg, combined_e1s=combined_e1s,
            combined_e1s_ana=combined_e1s_ana, combined_e1s_ext=combined_e1s_ext,
            combined_mean=combined_mean, combined_r_dict=combined_r_dict,
            display_1sigma=display_1sigma, age_dec=age_dec, cak_label=cak_label,
            legend_title=legend_title,
        )
        if not getattr(self, '_offscreen', False):
            self.draw_idle()

    def save(self, path):
        dpi = 300 if path.lower().endswith('.png') else 150
        self.fig.savefig(path, dpi=dpi, bbox_inches='tight',
                         facecolor='white', edgecolor='none')


class _StackFluxHost(FluxHost):
    """Adapter: shared flux dialog for the plateau stack's sessions."""
    title = 'Disk flux: stack'
    item_noun = 'Session'

    def __init__(self, win):
        self.win = win

    def disk_flux(self):
        return self.win._disk_flux

    def load(self, dialog):
        self.win._load_disk_flux()

    def items(self):
        return self.win._sessions

    def item_label(self, sess):
        try:
            i = self.win._sessions.index(sess)
        except ValueError:
            i = -1
        return sess.get('title') or (f'#{i + 1}' if i >= 0 else '#?')

    def item_disk_hole(self, sess):
        cfg = sess.get('cfg', {})
        return (cfg.get('disk_id'), cfg.get('hole_id'))

    def get_override(self, row, sess):
        return self.win._flux_override.get(row, 'auto')

    def set_override(self, row, sess, val):
        if val == 'auto':
            self.win._flux_override.pop(row, None)
        else:
            self.win._flux_override[row] = val

    def match(self, row, sess):
        return self.win._flux_match_session(sess, row)

    def on_row_change(self, row, sess):
        self.win._bake_flux_into_session(row)

    def hole_R(self, key, idx):
        return self.win._flux_hole_R(self.win._disk_flux[key], idx)

    def apply(self):
        # Bake every session's matched hole R into its own cfg before regrouping:
        # the flux groups are keyed on the calibration, so this is what splits
        # sessions sitting on different holes into separate groups instead of
        # leaving them all on the first group's R standard.
        for si in range(len(self.win._sessions)):
            self.win._bake_flux_into_session(si)
        self.win._update_flux_groups()
        self.win._recompute_all_sessions()


def _copy_table_selection(t):
    """Copy the selected cells of a QTableWidget to the clipboard as TSV."""
    rngs = t.selectedRanges()
    if not rngs:
        it = t.currentItem()
        if it:
            QApplication.clipboard().setText(it.text())
        return
    rmin = min(r.topRow() for r in rngs);    rmax = max(r.bottomRow() for r in rngs)
    cmin = min(r.leftColumn() for r in rngs); cmax = max(r.rightColumn() for r in rngs)
    lines = []
    for r in range(rmin, rmax + 1):
        row = []
        for c in range(cmin, cmax + 1):
            it = t.item(r, c)
            row.append(it.text() if (it and it.isSelected()) else '')
        lines.append('\t'.join(row))
    QApplication.clipboard().setText('\n'.join(lines))


def _enable_table_copy(t):
    """Make a QTableWidget's cells freely selectable and Ctrl/Cmd-C copyable,
    and clear any initial selection so it doesn't open fully highlighted."""
    from PyQt5.QtWidgets import QShortcut
    from PyQt5.QtGui import QKeySequence
    t.setSelectionMode(QAbstractItemView.ContiguousSelection)
    t.setSelectionBehavior(QAbstractItemView.SelectItems)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    sc = QShortcut(QKeySequence.Copy, t)
    sc.activated.connect(lambda tt=t: _copy_table_selection(tt))
    t.clearSelection()
    t.setCurrentCell(-1, -1)


def _fit_table_height(t, max_rows=None):
    """Size a QTableWidget to show its rows without an internal scrollbar
    (or up to max_rows, then scroll)."""
    t.resizeRowsToContents()
    n = t.rowCount()
    shown = n if max_rows is None else min(n, max_rows)
    h = t.horizontalHeader().height() + 2 * t.frameWidth()
    for r in range(shown):
        h += t.rowHeight(r)
    if max_rows is not None and n > max_rows:
        h += 18  # room for the scrollbar
        t.setMinimumHeight(h); t.setMaximumHeight(h)
    else:
        t.setFixedHeight(h + 2)


def _wm_label(r):
    """Column label of a WM + √MSWD result: pooled steps in ratio space
    (n = steps) or the per-aliquot plateau ages (n = aliquots)."""
    m = (r or {}).get('method', '')
    if m == 'wm_ratio':
        return 'WM + √MSWD (ratio space, n = steps)'
    if 'aliquots' in m:
        return 'WM + √MSWD (n = aliquots)'
    return 'WM + √MSWD'


class _StatsDialog(QDialog):
    """Read-only audit of a combination, comparing WM + √MSWD and GLS + ML side by
    side. mode='stack' runs the production _combined_plateau() both ways over the
    aliquots; mode='single' compares the current plateau (WM + √MSWD tiers) against a
    GLS + ML treatment of the same plateau steps. Both
    offer a tab-separated block of the results (Name / Age / 2σ ana / 2σ int /
    2σ ext / J / ±J%).
    """

    def __init__(self, win, mode='stack'):
        super().__init__(win)
        self.win = win
        self.mode = mode
        self.setWindowTitle("Combined age statistics"
                            if mode == 'stack' else "Plateau statistics")
        self.resize(760, 700)
        root = QVBoxLayout(self)
        root.setSpacing(6)

        gathered = self._gather_stack() if mode == 'stack' else self._gather_single()
        if gathered is None:
            root.addWidget(QLabel("No plateau is defined yet, nothing to summarise."))
            b = QPushButton("Close"); b.clicked.connect(self.accept)
            root.addWidget(b)
            return
        res, rowsd, paste_rows = gathered
        self._res = res

        rk, ak, ik, ek = res[METHOD_WM]
        rv, av, iv, ev = res[METHOD_GLS]

        # ── helpers ──────────────────────────────────────────────────────────
        def cell(txt, bold=False, align=Qt.AlignLeft | Qt.AlignVCenter):
            it = QTableWidgetItem(str(txt))
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            it.setTextAlignment(align)
            if bold:
                f = it.font(); f.setBold(True); it.setFont(f)
            return it

        def rcell(txt, bold=False):
            return cell(txt, bold=bold, align=Qt.AlignRight | Qt.AlignVCenter)

        def fmt(v, nd=4):
            return "—" if v is None else f"{v:.{nd}f}"

        def section_lbl(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "font-weight:600; font-size:12px; color:#222; "
                "padding-top:8px; padding-bottom:1px;")
            return lbl

        nk  = rk.get('n', 0) if rk else 0
        nv  = rv.get('n', 0) if rv else 0
        msk = rk.get('mswd', 0.0) if rk else 0.0
        msv = rv.get('mswd', 0.0) if rv else 0.0
        od     = rv.get('overdispersed', False) if rv else False
        om2    = rv.get('overdispersion', 0.0) if rv else 0.0

        # The scaling the WM tier actually carries: its analytical tier over
        # the raw SEM (√MSWD above the 95% limit, Student-t in sem_mswd)
        _raw = rk.get('sem_raw') if rk else None
        kmag = (ak / _raw) if (ak and _raw) else 1.0
        if kmag > 1.0 + 1e-9:
            kscatter = f"σₐₙₐ × {kmag:.3f}  (MSWD = {msk:.3f})"
        else:
            kscatter = f"none  (MSWD = {msk:.3f}, within the 95% limit)"
        if od:
            vscatter = f"ω = {om2 * 2:.4f} Ma  (2σ, ML)"
        else:
            vscatter = (f"ω = 0  (no excess,  p = {rv['prob']:.3f})"
                        if rv else "—")

        nk_lbl = f"{nk} steps"   if (mode == 'stack' and nk != nv) else str(nk)
        nv_lbl = f"{nv} aliquots" if (mode == 'stack' and nk != nv) else str(nv)

        # ── Section 1: Combined age ───────────────────────────────────────────
        root.addWidget(section_lbl("Combined age"))
        def fmt2(v):
            """Format a 1σ value multiplied to 2σ for display."""
            return "—" if v is None else f"{v * 2:.4f}"
        result_rows = [
            ("Age (Ma)",            fmt(rk and rk['mean']), fmt(rv and rv['mean'])),
            ("± Analytical (2σ)",
                                    fmt2(ak),               fmt2(av)),
            ("± Internal (2σ, ana + J)",
                                    fmt2(ik),               fmt2(iv)),
            ("± External (2σ, int + FC)",
                                    fmt2(ek),               fmt2(ev)),
            ("MSWD",                fmt(msk, 3),            fmt(msv, 3)),
            ("p (fit)",             fmt(rk and rk['prob'], 3), fmt(rv and rv['prob'], 3)),
            ("N",                   nk_lbl,                 nv_lbl),
            ("Excess dispersion",   kscatter,               vscatter),
        ]
        rt = QTableWidget(len(result_rows), 3)
        rt.setHorizontalHeaderLabels(
            ["", _wm_label(rk), "GLS + ML"])
        rt.verticalHeader().setVisible(False)
        rt.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        rt.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        rt.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        for r, (a, b, c) in enumerate(result_rows):
            rt.setItem(r, 0, cell(a, bold=True))
            rt.setItem(r, 1, rcell(b))
            rt.setItem(r, 2, rcell(c))
        _enable_table_copy(rt)
        _fit_table_height(rt)
        root.addWidget(rt)

        # Brief method note
        if mode == 'stack' and nk != nv:
            note = (f"WM pools all {nk} steps in ratio space (n = steps); GLS + ML "
                    f"combines {nv} plateau ages (n = aliquots). Both use the "
                    f"shared J / FCs-λ covariance.")
        else:
            note = ("Both build the same shared-J / FCs-λ covariance. WM scales σₐₙₐ by "
                    "√MSWD when overdispersed; GLS + ML adds ML overdispersion ω instead.")
        note_lbl = QLabel(note)
        note_lbl.setStyleSheet("color:#666; font-size:11px; padding:2px 2px 0 2px;")
        note_lbl.setWordWrap(True)
        root.addWidget(note_lbl)

        # ── Section 2: Constants & flux ───────────────────────────────────────
        root.addWidget(section_lbl("Constants & flux"))
        if mode == 'stack' and self.win._sessions:
            _cfg = (self.win._eff_cfg_for(0)
                    if hasattr(self.win, '_eff_cfg_for') else self.win._global_cfg)
        else:
            _cfg = (self.win._effective_single_cfg()
                    if hasattr(self.win, '_effective_single_cfg') else dict(self.win.cfg))
        self._cfg_used = _cfg

        lam     = _cfg.get('lambdaYr',    LAMBDA_DEFAULT)
        lam_err = _cfg.get('lambdaErrYr', LAMBDA_ERR)
        fc_ma   = _cfg.get('stdAgeMa',    STD_AGE_DEFAULT)
        fc_err  = _cfg.get('stdAgeErrMa', STD_AGE_ERR)
        fc_type = _cfg.get('stdAgeErrType', 'internal')
        r_std   = _cfg.get('rStd',        70.0)

        const_rows = [
            ("λ  (yr⁻¹)",            f"{lam:.6e}"),
            ("σλ  (2σ,  yr⁻¹)", f"{lam_err * 2:.4e}"),
            ("FC age  (Ma)",                          f"{fc_ma:.4f}"),
            (f"σ_FC  (2σ,  Ma,  {fc_type})", f"{fc_err * 2:.4f}"),
            ("⁴⁰Ar/³⁹Ar Rₛₜᵈ", f"{r_std:.6g}"),
        ]
        ct = QTableWidget(len(const_rows), 2)
        ct.setHorizontalHeaderLabels(["Parameter", "Value"])
        ct.verticalHeader().setVisible(False)
        ct.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        ct.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for r, (a, b) in enumerate(const_rows):
            ct.setItem(r, 0, cell(a, bold=True))
            ct.setItem(r, 1, cell(b))
        _enable_table_copy(ct)
        _fit_table_height(ct)
        root.addWidget(ct)

        # ── Section 3: Per-aliquot / per-step inputs ─────────────────────────
        root.addWidget(section_lbl(
            "Per-aliquot plateau ages  (inputs to the combination):" if mode == 'stack'
            else "Plateau steps  (inputs to this plateau):"))
        has_j = any(d.get('J', 0) > 0 for d in rowsd)
        hdrs  = ['#', 'Label', 'Age (Ma)', '±2σ ana',
                 '±2σ int', '±2σ ext']
        if has_j:
            hdrs += ['J', '±J (%)']
        hdrs += ['grp']
        at = QTableWidget(len(rowsd), len(hdrs))
        at.setHorizontalHeaderLabels(hdrs)
        at.verticalHeader().setVisible(False)
        at.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for r, d in enumerate(rowsd):
            vals = [str(d['n']), d['label'],
                    f"{d['age']:.4f}", f"{d['ana']:.4f}",
                    (f"{d['int']:.4f}" if d['int'] is not None else "—"),
                    (f"{d['ext']:.4f}" if d['ext'] is not None else "—")]
            if has_j:
                jv = d.get('J', 0.0); jp = d.get('Jp', 0.0)
                vals += [f"{jv:.7g}" if jv > 0 else "—",
                         f"{jp:.3f}"  if jv > 0 else "—"]
            vals += [d['grp']]
            for c, v in enumerate(vals):
                it = cell(v) if c in (0, 1, len(hdrs) - 1) else rcell(v)
                at.setItem(r, c, it)
        _enable_table_copy(at)
        _fit_table_height(at, max_rows=6)
        root.addWidget(at)

        # ── Section 4: Excel / clipboard ─────────────────────────────────────
        root.addWidget(section_lbl(
            "Copy to Excel  (tab-separated, headers included):"))
        self._paste = QTextEdit()
        self._paste.setReadOnly(True)
        self._paste.setAcceptRichText(False)
        self._paste.setStyleSheet(
            "font-family:Menlo,Consolas,monospace; font-size:11px;")
        self._paste.setPlainText(
            self._excel_text(rowsd, rk, ak, ik, ek, rv, av, iv, ev,
                             lam, fc_ma, fc_err, fc_type, lam_err, r_std))
        self._paste.setMaximumHeight(110)
        root.addWidget(self._paste)

        bb = QHBoxLayout()
        cp = QPushButton("Copy to clipboard")
        cp.setToolTip("Copies the tab-separated Excel block above")
        cp.clicked.connect(
            lambda: QApplication.clipboard().setText(self._paste.toPlainText()))
        bb.addWidget(cp)
        cp2 = QPushButton("Copy rows")
        cp2.setToolTip(
            "Tab-separated rows for a spreadsheet\n"
            "(no headers; errors at 2σ; Name / Age / ±2σ ana / "
            "±2σ int / ±2σ ext / J / ±J%)")
        cp2.clicked.connect(
            lambda: QApplication.clipboard().setText(self._paste_text(paste_rows)))
        bb.addWidget(cp2)
        bb.addStretch()
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        close.setDefault(True)
        bb.addWidget(close)
        root.addLayout(bb)
        close.setFocus()

    # ---- shared helpers ----
    def _sess_J(self, si, sess):
        """(J, ±J %) the aliquot's plateau ages are computed with."""
        J, rel = _aged_J(self.win._eff_cfg_for(si), sess.get('steps_c', []))
        return (J, rel * 100.0) if J else (0.0, 0.0)

    def _paste_text(self, rowsd):
        lines = []
        for d in rowsd:
            line = (f"{d['label']}\t{d['age']:.6f}\t{d['ana']:.6f}"
                    f"\t{d['int']:.6f}\t{d['ext']:.6f}")
            if d.get('J', 0) > 0:
                line += f"\t{d['J']:.7g}\t{d['Jp']:.4g}"
            lines.append(line)
        return "\n".join(lines)

    def _excel_text(self, rowsd, rk, ak, ik, ek, rv, av, iv, ev,
                    lam, fc_ma, fc_err, fc_type, lam_err, r_std):
        """Tab-separated block with column headers, suitable for direct paste into Excel."""
        has_j = any(d.get('J', 0) > 0 for d in rowsd)
        hdrs = ['Label', 'Age (Ma)', '±2σ ana', '±2σ int', '±2σ ext']
        if has_j:
            hdrs += ['J', '±J (%)']
        hdrs += ['lambda (yr-1)', 'FC age (Ma)',
                 f'sigma_FC (2s, {fc_type})', 'R_std']
        lines = ['\t'.join(hdrs)]
        for d in rowsd:
            row = [d['label'], f"{d['age']:.6f}", f"{d['ana']:.6f}",
                   (f"{d['int']:.6f}" if d['int'] is not None else ''),
                   (f"{d['ext']:.6f}" if d['ext'] is not None else '')]
            if has_j:
                jv = d.get('J', 0.0); jp = d.get('Jp', 0.0)
                row += [f"{jv:.7g}" if jv > 0 else '', f"{jp:.4g}" if jv > 0 else '']
            row += [f"{lam:.6e}", f"{fc_ma:.6f}", f"{fc_err * 2:.6f}", f"{r_std:.6g}"]
            lines.append('\t'.join(row))

        # Blank separator then summary
        def fv(v): return '' if v is None else f'{v:.6f}'
        def fv2(v): return '' if v is None else f'{v * 2:.6f}'   # 1σ → 2σ
        lines += ['',
                  '\t'.join([''] + [_wm_label(rk), 'GLS + ML']),
                  '\t'.join(['Age (Ma)',
                             fv(rk['mean'] if rk else None),
                             fv(rv['mean'] if rv else None)]),
                  '\t'.join(['±2σ analytical', fv2(ak), fv2(av)]),
                  '\t'.join(['±2σ internal',   fv2(ik), fv2(iv)]),
                  '\t'.join(['±2σ external',   fv2(ek), fv2(ev)]),
                  '\t'.join(['MSWD',
                             f"{rk['mswd']:.4f}" if rk else '',
                             f"{rv['mswd']:.4f}" if rv else '']),
                  '\t'.join(['p',
                             f"{rk['prob']:.4f}" if rk else '',
                             f"{rv['prob']:.4f}" if rv else '']),
                  '\t'.join(['lambda (yr-1)', f"{lam:.6e}", '']),
                  '\t'.join(['sigma_lambda (2s)', f"{lam_err * 2:.4e}", '']),
                  '\t'.join(['FC age (Ma)', f"{fc_ma:.6f}", '']),
                  '\t'.join([f'sigma_FC (2s, {fc_type})', f"{fc_err * 2:.6f}", '']),
                  '\t'.join(['R_std', f"{r_std:.6g}", '']),
                  ]
        return '\n'.join(lines)

    # ---- stack mode ----
    def _gather_stack(self):
        res = {}
        saved = self.win._stack_method
        try:
            for m in (METHOD_WM, METHOD_GLS):
                self.win._stack_method = m
                try:
                    res[m] = self.win._combined_plateau()
                except Exception:
                    res[m] = (None, None, None, None)
        finally:
            self.win._stack_method = saved
        if res[METHOD_WM][0] is None and res[METHOD_GLS][0] is None:
            return None
        rowsd = []
        for si, sess in enumerate(self.win._sessions):
            p = sess.get('plateau')
            if not p or not p.get('r'):
                continue
            ana = p.get('e1s_ana'); int_ = p.get('e1s_int', ana); ext = p.get('e1s_ext', int_)
            if ana is None:
                continue
            label = str(sess.get('title', f'#{si+1}')).replace('\t', ' ').replace('\n', ' ')
            grp = str(sess.get('j_grp') or sess.get('cfg', {}).get('disk_id') or '')
            jv, jp = self._sess_J(si, sess)
            # _StatsDialog reports at 2σ; internal tiers are 1σ → ×2 here.
            rowsd.append(dict(n=si + 1, label=label, age=float(p['r']['mean']),
                              ana=float(ana) * 2.0, int=float(int_) * 2.0,
                              ext=float(ext) * 2.0,
                              J=jv, Jp=jp, grp=grp))
        return res, rowsd, rowsd

    # ---- single mode ----
    def _single_J(self):
        """(J, ±J %) the plateau ages are computed with."""
        w = self.win
        J, rel = _aged_J(w._effective_single_cfg(), w.steps_c or [])
        return (J, rel * 100.0) if J else (0.0, 0.0)

    def _gather_single(self):
        w = self.win
        p = getattr(w, 'plateau', None)
        if not p or not p.get('r'):
            return None
        ois = p.get('ois', [])
        ages = [float(w.steps_c[i]['age']) for i in ois]
        s1 = [float(w.steps_c[i]['err1s']) for i in ois]        # analytical 1σ
        rk = dict(mean=p['r']['mean'], mswd=p['r'].get('mswd', 0.0),
                  prob=p['r'].get('prob', 1.0), n=p['r'].get('n', len(ages)),
                  sem_raw=p['r'].get('sem'))
        # res holds 1σ tiers, like the stack mode (the table doubles them for
        # its 2σ columns); the copied rows are written at 2σ below
        ana_k = p['e1s_ana']; int_k = p.get('e1s_int', ana_k); ext_k = p.get('e1s_ext', int_k)
        # the shared J and standard/λ terms on top of the analytical tier
        jterm = math.sqrt(max(int_k ** 2 - ana_k ** 2, 0.0))
        sterm = math.sqrt(max(ext_k ** 2 - int_k ** 2, 0.0))
        rv = ana_v = int_v = ext_v = None
        if len(ages) >= 1:
            rv_ = combine_ages_vermeesch(ages, s1, s1, s1, random_effects=True)
            ana_v = rv_['sigma_ana']
            int_v = math.sqrt(ana_v ** 2 + jterm ** 2)
            ext_v = math.sqrt(int_v ** 2 + sterm ** 2)
            rv = dict(mean=rv_['wm'], mswd=rv_['mswd'], prob=rv_['prob'],
                      n=rv_['n'], overdispersed=rv_['overdispersed'],
                      overdispersion=rv_['overdispersion'], n_flux_matched=0)
        res = {METHOD_WM: (rk, ana_k, int_k, ext_k),
               METHOD_GLS: (rv, ana_v, int_v, ext_v)}
        jv, jp = self._single_J()
        rowsd = []
        for k, i in enumerate(ois):
            s = w.steps_c[i]
            rowsd.append(dict(n=k + 1, label=str(s.get('label', f'step {k+1}')),
                              age=float(s['age']), ana=float(s['err1s']) * 2.0,
                              int=None, ext=None, J=0.0, Jp=0.0, grp=''))
        sample = (w.cfg.get('sampleName') or 'plateau')
        paste_rows = [dict(label=str(sample).replace('\t', ' '), age=rk['mean'],
                           ana=ana_k * 2.0, int=int_k * 2.0, ext=ext_k * 2.0,
                           J=jv, Jp=jp)]
        return res, rowsd, paste_rows


class _SingleFluxHost(FluxHost):
    """Adapter: shared flux dialog for a single plateau (one assignable row)."""
    title = 'Disk flux: plateau'
    item_noun = 'Plateau'
    caption = ("Pick the hole this plateau sits in; its J value and \u03c3_J drive "
               "the plateau age. \u201cauto\u201d uses the file's disk_id / hole_id; "
               "\u201cnone\u201d keeps the typed Age-source values.")

    def __init__(self, win):
        self.win = win
        cur = self.win._sp_flux_hole
        self._ov = tuple(cur) if isinstance(cur, tuple) else 'auto'

    def disk_flux(self):
        return self.win._disk_flux

    def load(self, dialog):
        self.win._load_disk_flux()

    def items(self):
        return [self.win]

    def item_label(self, _item):
        return 'This plateau'

    def item_disk_hole(self, _item):
        cfg = self.win.cfg
        return (cfg.get('disk_id'), cfg.get('hole_id'))

    def get_override(self, row, _item):
        return self._ov

    def set_override(self, row, _item, val):
        self._ov = val

    def _resolve(self):
        ov = self._ov
        if ov == 'none':
            return None
        if ov == 'auto':
            cfg = self.win.cfg
            m = self.win._flux_auto_match(cfg.get('disk_id'), cfg.get('hole_id'),
                                          cfg.get('irradiation'))
            if m is None:
                return None
            (irr, disk), idx = m
            fx = self.win._disk_flux[(str(irr), str(disk))]
            hole = next((h for h, ix in fx['idx'].items() if ix == idx), None)
            return (irr, disk, hole) if hole is not None else None
        return ov

    def match(self, row, _item):
        fh = self._resolve()
        if fh is None:
            return None
        key = (str(fh[0]), str(fh[1]))
        fx = self.win._disk_flux.get(key)
        if fx is not None and str(fh[2]) in fx['idx']:
            return key, fx['idx'][str(fh[2])]
        return None

    def hole_R(self, key, idx):
        return self.win._flux_hole_R(self.win._disk_flux[key], idx)

    def apply(self):
        self.win._sp_flux_hole = self._resolve()
        self.win._bake_flux_into_cfg(self.win.cfg, self.win._sp_flux_hole,
                                     self.win._sp_flux_base)
        self.win._restore_cfg_to_widgets(self.win.cfg)
        if self.win.raw_steps:
            self.win._update_all()


class _CarouselFluxHost(FluxHost):
    """Adapter: shared flux dialog for the carousel's aliquots."""
    title = 'Disk flux: all aliquots'
    item_noun = 'Aliquot'

    def __init__(self, win):
        self.win = win
        self._ov = {}

    def on_open(self):
        self.win._carousel_save_current()
        self._ov = {}
        for r, slot in enumerate(self.win._carousel):
            fh = slot.get('flux_hole')
            self._ov[r] = tuple(fh) if isinstance(fh, tuple) else 'auto'

    def disk_flux(self):
        return self.win._disk_flux

    def load(self, dialog):
        self.win._load_disk_flux()

    def items(self):
        return self.win._carousel

    def item_label(self, slot):
        try:
            i = self.win._carousel.index(slot)
        except ValueError:
            i = -1
        return slot.get('title') or (f'Aliquot {i + 1}' if i >= 0 else 'Aliquot')

    def item_disk_hole(self, slot):
        cfg = slot.get('cfg', {})
        return (cfg.get('disk_id'), cfg.get('hole_id'))

    def _resolve(self, slot, ov):
        if ov == 'none':
            return None
        if ov == 'auto':
            return self.win._slot_flux_auto(slot)
        return ov

    def get_override(self, row, slot):
        return self._ov.get(row, 'auto')

    def set_override(self, row, slot, val):
        self._ov[row] = val
        slot['flux_hole'] = self._resolve(slot, val)

    def on_row_change(self, row, slot):
        self.win._bake_flux_into_slot(slot)

    def match(self, row, slot):
        fh = self._resolve(slot, self._ov.get(row, 'auto'))
        if fh is None:
            return None
        key = (str(fh[0]), str(fh[1]))
        fx = self.win._disk_flux.get(key)
        if fx is not None and str(fh[2]) in fx['idx']:
            return key, fx['idx'][str(fh[2])]
        return None

    def hole_R(self, key, idx):
        return self.win._flux_hole_R(self.win._disk_flux[key], idx)

    def apply(self):
        for r, slot in enumerate(self.win._carousel):
            slot['flux_hole'] = self._resolve(slot, self._ov.get(r, 'auto'))
            self.win._bake_flux_into_slot(slot)
        ci = self.win._carousel_idx
        if 0 <= ci < len(self.win._carousel):
            self.win._carousel_load_slot(ci)
            self.win._carousel_refresh_ui()
        self.win._refresh_sp_flux_state()


class PlateauWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plateau & isochron")
        self.setGeometry(80, 80, 1380, 820)

        self.raw_steps = []
        self.steps_c   = []
        self.plateau   = None
        self.pset      = set()

        self.cfg = dict(_SESSION_CFG_DEFAULTS)

        # Stack-mode state
        self._stack_mode    = False
        self._isochron_mode = False
        self._sessions      = []
        self._global_cfg    = dict(_SESSION_CFG_DEFAULTS)
        self._flux_groups   = []
        # (irradiation, disk_id) -> flux entry (.jflux).  Normal mode (single plateau
        # + carousel) and stack mode each keep their own registry; self._disk_flux
        # (property below) is whichever belongs to the mode that is on.
        self._disk_flux_single = {}
        self._disk_flux_stack  = {}
        self._flux_override = {}          # session_idx -> (irr, disk, hole) manual assignment
        self._flux_no_ratio = set()        # titles of flux-matched sessions lacking r4039
        self._sp_flux_hole  = None         # single-plateau: (irr, disk, hole) flux override, or None
        self._sp_flux_base  = {}           # snapshot of pre-flux calibration (for 'none' restore)
        self._stack_method  = METHOD_WM   # METHOD_WM (WM + √MSWD) or METHOD_GLS (GLS + ML)
        self._wm_n_basis = 'aliquots' # WM+√MSWD basis: 'aliquots' (per-plateau age then
                                          # WM, matches prob. plot) or 'steps' (pool step-ratios)

        # Single-mode aliquot carousel
        self._carousel     = []   # list of {title, raw_steps, cfg, orig_raw_steps, orig_calib_cfg, plateau}
        self._carousel_idx = -1   # -1 = carousel inactive (single file)

        self._build_ui()

    # ─── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        # Timer must exist before any valueChanged signals fire during setup
        self._redraw_timer = QTimer()
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(30)
        self._redraw_timer.timeout.connect(self._do_redraw)

        # ── Menu bar ──────────────────────────────────────────────────────────
        mb = self.menuBar()

        fm = mb.addMenu('File')
        fm.addAction(self._act('Import File…',         self._import_dispatch, 'Ctrl+O'))
        fm.addAction(self._act('Paste from Clipboard', self._paste_dispatch,  'Ctrl+V'))
        fm.addSeparator()
        fm.addAction(self._act('Save…',                self._save_dispatch,   'Ctrl+S'))
        fm.addAction(self._act('Load…',                self._load_dispatch))

        vm = mb.addMenu('View')
        vm.addAction(self._act('Error breakdown (pie)…', self._show_error_breakdown))

        em = mb.addMenu('Export')
        em.addAction(self._act('Export Figure…  (layout + preview)', self.export_figure,
                               'Ctrl+E'))
        em.addAction(self._act('Export All Sessions as PDF…',    self.export_all_pdf))
        em.addAction(self._act('Export All Samples…  (spectrum + isochrons)', self.export_all_samples))
        em.addSeparator()
        em.addAction(self._act('Export Data (XLSX / CSV)',              self.export_csv))

        # ── Central widget ────────────────────────────────────────────────────
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QHBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: plot area ───────────────────────────────────────────────────
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 4, 6)
        lv.setSpacing(4)

        # Title row
        th = QHBoxLayout()
        th.addWidget(QLabel("Sample:"))
        self.title_edit = QLineEdit()
        self.title_edit.setFont(QFont('Arial', 11, QFont.Bold))
        self.title_edit.setStyleSheet(
            'QLineEdit { background: #ffffff; color: #1a1a1a; '
            'border: 1px solid #b0b0b0; border-radius: 3px; padding: 1px 4px; }')
        self.title_edit.textChanged.connect(self._queue_redraw)
        self.title_edit.textChanged.connect(self._carousel_sync_title)
        th.addWidget(self.title_edit, 1)
        lv.addLayout(th)

        # ── Aliquot carousel strip (hidden until 2+ aliquots loaded) ─────────
        _car_w = QWidget()
        _car_w.setVisible(False)
        _cl = QHBoxLayout(_car_w)
        _cl.setContentsMargins(0, 2, 0, 2)
        _cl.setSpacing(4)
        _btn_prev = QPushButton("◀")
        _btn_prev.setFixedWidth(26); _btn_prev.setFixedHeight(22)
        _btn_prev.setToolTip("Previous aliquot")
        self._carousel_combo = QComboBox()
        self._carousel_combo.setFixedHeight(22)
        self._carousel_combo.setToolTip("Select aliquot")
        _btn_next = QPushButton("▶")
        _btn_next.setFixedWidth(26); _btn_next.setFixedHeight(22)
        _btn_next.setToolTip("Next aliquot")
        self._carousel_count_lbl = QLabel("")
        self._carousel_count_lbl.setFixedWidth(40)
        self._carousel_count_lbl.setAlignment(Qt.AlignCenter)
        _btn_add = QPushButton("+ Add")
        _btn_add.setFixedHeight(22)
        _btn_add.setToolTip("Load additional .arar file into carousel")
        _btn_remove = QPushButton("✕")
        _btn_remove.setFixedWidth(22); _btn_remove.setFixedHeight(22)
        _btn_remove.setToolTip("Remove current aliquot from carousel")
        _cl.addWidget(_btn_prev)
        _cl.addWidget(self._carousel_combo, 1)
        _cl.addWidget(_btn_next)
        _cl.addWidget(self._carousel_count_lbl)
        _cl.addWidget(_btn_add)
        _cl.addWidget(_btn_remove)
        lv.addWidget(_car_w)
        self._carousel_widget = _car_w
        _btn_prev.clicked.connect(lambda: self._carousel_go(-1))
        _btn_next.clicked.connect(lambda: self._carousel_go(+1))
        self._carousel_combo.currentIndexChanged.connect(self._carousel_on_combo)
        _btn_add.clicked.connect(self._carousel_open_add)
        _btn_remove.clicked.connect(self._carousel_remove_current)

        self._canvas_stack = QStackedWidget()
        self.canvas = PlateauCanvas(click_cb=self._on_canvas_step_clicked, parent=left)
        self.overlay_canvas = OverlayCanvas(
            click_cb=self._on_overlay_step_clicked, parent=left,
            dblclick_cb=self._stack_toggle_step)
        self.isochron_canvas = IsochronCanvas(parent=left)
        self._canvas_stack.addWidget(self.canvas)           # index 0
        self._canvas_stack.addWidget(self.overlay_canvas)   # index 1
        self._canvas_stack.addWidget(self.isochron_canvas)  # index 2
        lv.addWidget(self._canvas_stack, 1)

        # Isochron toggle row, only visible when in isochron mode
        self._iso_toggle_widget = QWidget()
        _itl = QHBoxLayout(self._iso_toggle_widget)
        _itl.setContentsMargins(0, 0, 0, 0)
        _itl.addStretch()
        self._btn_iso_panel = QPushButton("Switch to Normal Isochron")
        self._btn_iso_panel.setFixedHeight(24)
        self._btn_iso_panel.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 14px; }")
        self._btn_iso_panel.clicked.connect(self._toggle_iso_panel)
        _itl.addWidget(self._btn_iso_panel)
        _itl.addSpacing(12)
        self._btn_iso_zoom = QPushButton("Show Intercept")
        self._btn_iso_zoom.setFixedHeight(24)
        self._btn_iso_zoom.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 14px; }")
        self._btn_iso_zoom.clicked.connect(self._toggle_iso_zoom)
        _itl.addWidget(self._btn_iso_zoom)
        _itl.addSpacing(16)
        # Axis ranges of the on-screen isochron, one set per diagram (inverse /
        # normal have different units).  Blank = automatic, as in the export.
        self._iso_axis_lims = {'inverse': {}, 'normal': {}}
        self._iso_lim_edits = {}
        for axis in ('x', 'y'):
            _itl.addWidget(QLabel(f"{axis.upper()}:"))
            for end, tip in (('0', 'from'), ('1', 'to')):
                le = QLineEdit()
                le.setPlaceholderText('auto')
                le.setFixedWidth(74)
                le.setToolTip(f"{axis.upper()} axis {tip} (blank = automatic)")
                le.editingFinished.connect(self._on_iso_lim_edit)
                self._iso_lim_edits[axis + end] = le
                _itl.addWidget(le)
            _itl.addSpacing(6)
        self._btn_iso_lim_auto = QPushButton("Auto")
        self._btn_iso_lim_auto.setFixedHeight(24)
        self._btn_iso_lim_auto.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 10px; }")
        self._btn_iso_lim_auto.setToolTip("Clear the typed ranges of this diagram")
        self._btn_iso_lim_auto.clicked.connect(self._on_iso_lim_auto)
        _itl.addWidget(self._btn_iso_lim_auto)
        _itl.addStretch()
        self._iso_toggle_widget.setVisible(False)
        lv.addWidget(self._iso_toggle_widget)

        zr = QHBoxLayout()
        self._lbl_zoom = QLabel("Zoom:")
        zr.addWidget(self._lbl_zoom)
        self._zoom_grp = QButtonGroup(self)
        self._rb_plateau = None
        # the age-spectrum axis controls, hidden while the isochron is shown
        self._spectrum_zoom_widgets = [self._lbl_zoom]
        for i, (v, l) in enumerate([('auto', 'Auto'), ('plateau', 'Plateau'),
                                     ('manual', 'Manual')]):
            rb = QRadioButton(l)
            rb.setProperty('zoomMode', v)
            self._zoom_grp.addButton(rb, i)
            zr.addWidget(rb)
            self._spectrum_zoom_widgets.append(rb)
            if v == 'auto':
                rb.setChecked(True)
            if v == 'plateau':
                self._rb_plateau = rb
        self._zoom_grp.buttonClicked.connect(self._on_zoom_btn)

        self._lbl_ymin = QLabel("Y min:")
        self._sp_ymin  = QDoubleSpinBox()
        self._lbl_ymax = QLabel("Y max:")
        self._sp_ymax  = QDoubleSpinBox()
        for sp in (self._sp_ymin, self._sp_ymax):
            sp.setRange(0, 99999); sp.setDecimals(1); sp.setSingleStep(0.5)
            sp.setSuffix(' Ma');   sp.setFixedWidth(82)
            sp.valueChanged.connect(self._on_zoom_spin)
        self._sp_ymin.setValue(self.cfg['yMin'])
        self._sp_ymax.setValue(self.cfg['yMax'])
        for w in (self._lbl_ymin, self._sp_ymin, self._lbl_ymax, self._sp_ymax):
            zr.addWidget(w)
        zr.addSpacing(10)
        self._lbl_p40max = QLabel("%\u2074\u2070Ar* max:")
        zr.addWidget(self._lbl_p40max)
        self._sp_p40max = QDoubleSpinBox()
        self._sp_p40max.setRange(0, 1000); self._sp_p40max.setDecimals(0)
        self._sp_p40max.setSingleStep(10); self._sp_p40max.setSuffix(' %')
        self._sp_p40max.setSpecialValueText('Auto')
        self._sp_p40max.setFixedWidth(72)
        self._sp_p40max.setToolTip(
            "Top of the %\u2074\u2070Ar* panel, in the window and in exports.\n"
            "Auto: 0-100 %, higher when a step is above 100 %.\n"
            "Steps above the value you set are cut off at the top.")
        self._sp_p40max.valueChanged.connect(self._on_p40max)
        zr.addWidget(self._sp_p40max)
        self._spectrum_zoom_widgets += [self._lbl_p40max, self._sp_p40max]
        zr.addStretch()
        # Age-breakdown info box, far right of the zoom row.  Opens a panel with
        # the 2σ confidence limit, the error-scaling story and a variance pie.
        self._btn_age_info = QPushButton("Age breakdown  ⓘ")
        self._btn_age_info.setFixedHeight(24)
        self._btn_age_info.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 12px; }")
        self._btn_age_info.setToolTip(
            "Statistical breakdown of the plateau (or combined) age:\n"
            "2σ confidence limit, MSWD / error magnification, the three\n"
            "error tiers, and a pie chart of the variance sources.")
        self._btn_age_info.clicked.connect(self._show_age_breakdown)
        zr.addWidget(self._btn_age_info)
        self._set_zoom_widgets_visible(False)
        lv.addLayout(zr)

        root.addWidget(left, 3)

        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setFrameShadow(QFrame.Sunken)
        root.addWidget(div)

        # ── Right: sidebar ────────────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(360)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 8, 8, 6)
        rv.setSpacing(4)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        #, Data tab,
        dt = QWidget()
        dv = QVBoxLayout(dt)
        dv.setContentsMargins(0, 4, 0, 0)
        dv.setSpacing(3)

        # Two columns, not three: the sidebar is a fixed 360 px, so three
        # side-by-side buttons left ~110 px each and every label was clipped
        # ("aste+ Add Aliquo", "eset All Selection"). Stack mode swaps in the
        # longest labels of all, so the row has to hold those.
        bh = QGridLayout()
        bh.setHorizontalSpacing(4)
        bh.setVerticalSpacing(3)
        self._btn_paste  = QPushButton("Paste Clipboard")
        self._btn_import = QPushButton("Import File…")
        self._btn_reset  = QPushButton("Reset All Selections")
        self._btn_reset.setToolTip(
            "Clear manual exclusions and forced steps on every loaded plateau "
            "(carousel aliquots and stack sessions included), then re-run auto "
            "detection. Calibration and disk-flux assignments are left alone.")
        self._btn_paste.clicked.connect(self._paste_dispatch)
        self._btn_import.clicked.connect(self._import_dispatch)
        self._btn_reset.clicked.connect(self._reset_dispatch)
        bh.addWidget(self._btn_paste,  0, 0)
        bh.addWidget(self._btn_import, 0, 1)
        bh.addWidget(self._btn_reset,  1, 0, 1, 2)
        bh.setColumnStretch(0, 1)
        bh.setColumnStretch(1, 1)
        dv.addLayout(bh)

        sig_row = QHBoxLayout()
        sig_row.setContentsMargins(2, 0, 0, 0)
        self._cb_1sigma = QCheckBox("Input errors are 1σ")
        self._cb_1sigma.setChecked(True)   # converter writes raw 1σ; default to 1σ
        self._cb_1sigma.setToolTip(
            "Affects session file loading only. "
            "When pasting clipboard data, the σ level is chosen per-paste in the "
            "column-mapping dialog.")
        self._cb_1sigma.setStyleSheet("font-size: 10px; color: #888;")
        sig_row.addWidget(self._cb_1sigma)
        sig_row.addStretch()
        dv.addLayout(sig_row)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet("color: #ddd;"); dv.addWidget(sep1)

        bh2 = QHBoxLayout()
        self._btn_save = QPushButton("Save Session…")
        self._btn_load = QPushButton("Load Session(s)…")
        self._btn_save.clicked.connect(self._save_dispatch)
        self._btn_load.clicked.connect(self._load_dispatch)
        bh2.addWidget(self._btn_save)
        bh2.addWidget(self._btn_load)
        dv.addLayout(bh2)

        # Mode toggles, Stack and Isochron (mutually exclusive)
        mode_row = QHBoxLayout()
        self._btn_stack_mode = QPushButton("Stack Mode")
        self._btn_stack_mode.setCheckable(True)
        self._btn_stack_mode.setToolTip(
            "Switch to multi-aliquot overlay mode.\n"
            "Paste or load .ararpt files to add aliquots.")
        self._btn_stack_mode.toggled.connect(self._toggle_stack_mode)
        mode_row.addWidget(self._btn_stack_mode)

        self._btn_isochron_mode = QPushButton("Isochron")
        self._btn_isochron_mode.setCheckable(True)
        self._btn_isochron_mode.setToolTip(
            "Show normal and inverse isochron diagrams.\n"
            "Uses data from whichever mode is currently active.")
        self._btn_isochron_mode.toggled.connect(self._toggle_isochron_mode)
        mode_row.addWidget(self._btn_isochron_mode)
        dv.addLayout(mode_row)

        sep_mode = QFrame(); sep_mode.setFrameShape(QFrame.HLine)
        sep_mode.setStyleSheet("color: #ddd;"); dv.addWidget(sep_mode)

        # QStackedWidget: page 0 = normal controls, page 1 = stack controls
        self._mode_stack = QStackedWidget()

        # Page 0: single-aliquot controls
        normal_page = QWidget()
        np_lay = QVBoxLayout(normal_page)
        np_lay.setContentsMargins(0, 0, 0, 0)
        np_lay.setSpacing(3)

        self._btn_pick = QPushButton("Pick Plateau")
        self._btn_pick.setCheckable(True)
        self._btn_pick.setToolTip(
            "When active, click steps to add/remove them from the plateau directly")
        self._btn_pick.toggled.connect(self._on_pick_plateau_toggled)
        _np_btnrow = QHBoxLayout()
        _np_btnrow.addWidget(self._btn_pick, 1)
        self._btn_single_stats = QPushButton("Stats\u2026")
        self._btn_single_stats.setToolTip(
            "Show this plateau's combination statistics, WM + \u221aMSWD and "
            "GLS + ML side by side, how each is scaled, and a row ready to "
            "copy.")
        self._btn_single_stats.clicked.connect(self._open_single_stats)
        _np_btnrow.addWidget(self._btn_single_stats)
        np_lay.addLayout(_np_btnrow)

        self._hint = QLabel(
            "Click step: toggle excluded (red) / normal  ·  Use Pick mode to force plateau steps (blue)")
        self._hint.setStyleSheet("color: #888; font-size: 10px;")
        np_lay.addWidget(self._hint)

        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(
            ['#', 'Pwr', 'Age', '±2σ', '%³⁹Ar',
             'Ca/K', '%⁴⁰Ar*', '●', '⁴⁰/³⁹ R', '±err R'])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self._on_cell_clicked)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setStretchLastSection(False)
        np_lay.addWidget(self.table, 1)

        self._mode_stack.addWidget(normal_page)

        # Page 1: stack controls
        stack_page = QWidget()
        sp_lay = QVBoxLayout(stack_page)
        sp_lay.setContentsMargins(0, 0, 0, 0)
        sp_lay.setSpacing(3)

        sp_lay.addWidget(QLabel(
            "Aliquots (double-click name to rename):"))

        self.sample_tree = QTreeWidget()
        self.sample_tree.setColumnCount(5)
        self.sample_tree.setHeaderLabels(
            ['#', 'Sample / Label', 'Age ±2σ (Ma)', '✕', 'J grp'])
        self.sample_tree.setToolTip(
            '✕ column: check the box to exclude an aliquot from the combined plateau age.\n'
            'The aliquot is still displayed on the plot but grayed out.\n\n'
            'J grp: without a loaded disk flux, sessions with the same label are\n'
            'taken to share J (ρ = 1), so their J error does not average down.\n'
            'Filled in from the disk letter of the irradiation label\n'
            '(e.g. "9E19-25" → "9E"); type the same label on sessions from one disk.')
        _sth = self.sample_tree.header()
        _sth.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        _sth.setSectionResizeMode(1, QHeaderView.Stretch)
        _sth.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        _sth.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        _sth.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.sample_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sample_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.sample_tree.itemChanged.connect(self._on_tree_name_edited)
        self.sample_tree.itemClicked.connect(self._on_tree_item_clicked)
        self.sample_tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        sp_lay.addWidget(self.sample_tree, 1)
        _stack_hint = QLabel("Double-click a step (in the plot or the list): "
                             "exclude it (red), or reset it")
        _stack_hint.setStyleSheet("color: #888; font-size: 10px;")
        sp_lay.addWidget(_stack_hint)

        brow = QHBoxLayout()
        for lbl, fn in [('Remove', self._stack_remove_selected),
                        ('↑', self._stack_move_up),
                        ('↓', self._stack_move_down)]:
            b = QPushButton(lbl); b.clicked.connect(fn); brow.addWidget(b)
        brow.addStretch()
        self._btn_copy_ages = QPushButton("Copy Ages")
        self._btn_copy_ages.setToolTip(
            "Copy plateau ages to the clipboard, tab-separated:\n"
            "label \u2502 age \u2502 \u00b12\u03c3 analytical \u2502 \u00b12\u03c3 internal (+J).\n"
            "For a weighted mean in IsoplotR use the analytical column: it treats\n"
            "the errors as independent, so a J error shared by aliquots on one\n"
            "hole would average down.  Add the J error once to the combined age.")
        self._btn_copy_ages.clicked.connect(self._stack_copy_ages)
        brow.addWidget(self._btn_copy_ages)
        self._btn_stack_stats = QPushButton("Stats\u2026")
        self._btn_stack_stats.setToolTip(
            "Show the full combination statistics, WM + \u221aMSWD and GLS + ML "
            "side by side, how each is scaled, and a table ready to copy.")
        self._btn_stack_stats.clicked.connect(self._open_stack_stats)
        brow.addWidget(self._btn_stack_stats)
        sp_lay.addLayout(brow)

        self._lbl_stack_errors = QLabel("")
        self._lbl_stack_errors.setWordWrap(True)
        self._lbl_stack_errors.setStyleSheet(
            "font-size:10px; color:#555; padding:2px;")
        sp_lay.addWidget(self._lbl_stack_errors)

        self._mode_stack.addWidget(stack_page)
        dv.addWidget(self._mode_stack, 1)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #ddd;"); dv.addWidget(sep2)

        bh3 = QHBoxLayout()
        btn_eimg = QPushButton("Export figure…")
        btn_eimg.setToolTip("Choose which panels go in the figure, how tall each one is, "
                            "the page size, text size and legend, with a live preview.")
        btn_ecsv = QPushButton("Save XLSX / CSV")
        btn_ecsv.setToolTip("The step data and plateau of the open sample, or of all "
                            "loaded samples in one file (asked first).")
        for b, fn in [(btn_eimg, self.export_figure),
                      (btn_ecsv, self.export_csv)]:
            b.clicked.connect(fn)
            b.setStyleSheet("font-size:10px;")
            bh3.addWidget(b)
        dv.addLayout(bh3)

        btn_eall = QPushButton("Save All Sessions as PDF…")
        btn_eall.clicked.connect(self.export_all_pdf)
        btn_eall.setStyleSheet("font-size:10px;")
        dv.addWidget(btn_eall)
        btn_eiso = QPushButton("Export All Samples…")
        btn_eiso.setToolTip("One page per sample: the age spectrum with the inverse and "
                            "normal isochron underneath. Choose panels, size and text.")
        btn_eiso.clicked.connect(self.export_all_samples)
        btn_eiso.setStyleSheet("font-size:10px;")
        dv.addWidget(btn_eiso)

        tabs.addTab(dt, "Data")

        #, Settings tab,
        st = QWidget()
        sv = QVBoxLayout(st)
        sv.setContentsMargins(6, 6, 6, 6)
        sv.setSpacing(2)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        def _hd(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight:bold; font-size:11px; margin-top:8px; color:#333;")
            return lbl

        form.addRow(_hd("Plateau Definition"))
        self._sp_minPct = QDoubleSpinBox()
        self._sp_minPct.setRange(0,100); self._sp_minPct.setSingleStep(1); self._sp_minPct.setSuffix(' %')
        self._sp_minPct.setValue(self.cfg['minPct']); self._sp_minPct.valueChanged.connect(self._on_cfg)
        form.addRow("Min. %³⁹Ar:", self._sp_minPct)

        self._sp_minN = QSpinBox()
        self._sp_minN.setRange(2, 50); self._sp_minN.setValue(self.cfg['minN'])
        self._sp_minN.valueChanged.connect(self._on_cfg)
        form.addRow("Min. steps:", self._sp_minN)

        self._sp_pCut = QDoubleSpinBox()
        self._sp_pCut.setRange(0,100); self._sp_pCut.setSingleStep(1); self._sp_pCut.setSuffix(' %')
        self._sp_pCut.setValue(self.cfg['pCut']); self._sp_pCut.valueChanged.connect(self._on_cfg)
        form.addRow("Min. p-value (≥):", self._sp_pCut)

        form.addRow(_hd("Error Calculation"))
        self._cb_errMode = QComboBox()
        self._cb_errMode.addItem("SEM\u00b7\u221aMSWD\u00b7t(n\u22121) if p < cutoff, else SEM\u00d72", 'sem_mswd')
        self._cb_errMode.addItem("SEM\u00d72, \u00d7\u221aMSWD if MSWD > 95% limit(n)", 'sem_mswd_2s')
        self._cb_errMode.addItem("SEM\u00d72", 'sem')
        for _i in range(self._cb_errMode.count()):
            if self._cb_errMode.itemData(_i) == self.cfg.get('errMode', 'sem_mswd_2s'):
                self._cb_errMode.setCurrentIndex(_i)
                break
        self._cb_errMode.currentIndexChanged.connect(self._on_cfg)
        self._lbl_errMode = QLabel("Error method:")
        form.addRow(self._lbl_errMode, self._cb_errMode)

        _disp_w = QWidget()
        _disp_l = QHBoxLayout(_disp_w)
        _disp_l.setContentsMargins(0, 0, 0, 0)
        _disp_l.setSpacing(12)
        self._rb_disp_2s = QRadioButton("2σ")
        self._rb_disp_1s = QRadioButton("1σ")
        self._rb_disp_2s.setChecked(True)
        _disp_bg = QButtonGroup(_disp_w)
        _disp_bg.addButton(self._rb_disp_2s)
        _disp_bg.addButton(self._rb_disp_1s)
        self._rb_disp_2s.toggled.connect(self._queue_redraw)
        _disp_l.addWidget(self._rb_disp_2s)
        _disp_l.addWidget(self._rb_disp_1s)
        _disp_l.addStretch()
        form.addRow("Display errors as:", _disp_w)
        _disp_w = QWidget()
        _disp_l = QHBoxLayout(_disp_w)
        _disp_l.setContentsMargins(0, 0, 0, 0)
        _disp_l.setSpacing(12)
        self._sp_ageDec = QSpinBox()
        self._sp_ageDec.setRange(0, 4)
        self._sp_ageDec.setValue(self.cfg.get('ageDec', 2))
        self._sp_ageDec.setFixedWidth(60)
        self._sp_ageDec.setToolTip("Decimal places for age and error values in table, plot, and summary")
        self._sp_ageDec.valueChanged.connect(self._on_cfg)
        _disp_l.addWidget(self._sp_ageDec)
        _disp_l.addStretch()
        form.addRow("Decimals:", _disp_w)
        self._cb_ratio = QComboBox()
        self._cb_ratio.addItem("Ca/K", 'cak')
        self._cb_ratio.addItem("K/Ca", 'kca')
        self._cb_ratio.setToolTip("Ratio panel and table column: Ca/K, or K/Ca = 1/(Ca/K). "
                                  "Steps with Ca/K ≤ 0 have no K/Ca and are left out.")
        self._cb_ratio.currentIndexChanged.connect(self._queue_redraw)
        _ratio_w = QWidget()
        _ratio_l = QHBoxLayout(_ratio_w)
        _ratio_l.setContentsMargins(0, 0, 0, 0)
        _ratio_l.addWidget(self._cb_ratio)
        _ratio_l.addStretch()
        form.addRow("Ratio:", _ratio_w)

        self._lbl_flux_hd = _hd("Age & Flux Calibration")
        form.addRow(self._lbl_flux_hd)
        self._cb_ageFluxMode = QComboBox()
        self._cb_ageFluxMode.addItem("Imported ages",     'imported')
        self._cb_ageFluxMode.addItem("R mode",            'ratio_R_std')
        self._cb_ageFluxMode.addItem("R mode (insert J)", 'ratio_J')
        for _i in range(self._cb_ageFluxMode.count()):
            if self._cb_ageFluxMode.itemData(_i) == self.cfg.get(
                    'ageFluxMode', 'ratio_R_std'):
                self._cb_ageFluxMode.setCurrentIndex(_i)
        self._cb_ageFluxMode.currentIndexChanged.connect(self._on_cfg)
        self._lbl_age_src = QLabel("Age source:")
        form.addRow(self._lbl_age_src, self._cb_ageFluxMode)
        # The age source is chosen in the Calibration… dialog; the hidden combo
        # holds the setting.
        self._lbl_age_src.setVisible(False)
        self._cb_ageFluxMode.setVisible(False)

        # Disk flux + Calibration buttons, sit directly under Age source,
        # identical layout to the row in the stack-mode flux section.
        self._btn_sp_flux = QPushButton("Disk flux…")
        self._btn_sp_flux.setToolTip(
            "Load a .jflux, view the disk, and pick this plateau's hole to adopt "
            "its J value and σ_J.")
        self._btn_sp_flux.clicked.connect(self._open_single_flux)
        self._btn_calibrate = QPushButton("Calibration…")
        self._btn_calibrate.setToolTip(
            "Choose the age source (R mode / R mode with J / imported ages) and\n"
            "recalculate loaded step ages to a different decay constant / FC\n"
            "standard. Applies to every aliquot when 'Apply to all' is ticked.")
        self._btn_calibrate.clicked.connect(self._open_calibrate)
        self._lbl_sp_flux_state = QLabel("(none)")
        self._lbl_sp_flux_state.setStyleSheet("color:#555;")
        self._lbl_sp_flux = QLabel("Disk flux J:")
        _sp_flux_w = QWidget()
        _sp_flux_h = QHBoxLayout(_sp_flux_w)
        _sp_flux_h.setContentsMargins(0, 0, 0, 0)
        _sp_flux_h.addWidget(self._btn_sp_flux)
        _sp_flux_h.addWidget(self._btn_calibrate)
        _sp_flux_h.addStretch()
        self._sp_flux_w = _sp_flux_w
        form.addRow(self._sp_flux_w)
        self._lbl_sp_flux_state.setWordWrap(True)
        form.addRow(self._lbl_sp_flux, self._lbl_sp_flux_state)
        self._refresh_sp_flux_state()

        self._lbl_calib_state = QLabel("")
        self._lbl_calib_state.setStyleSheet(
            "color: #1a7a1a; font-size:9px; font-style:italic; padding-top:1px;")
        self._lbl_calib_state.setWordWrap(True)
        self._lbl_calib_state.setVisible(False)
        form.addRow("", self._lbl_calib_state)
        self._lbl_route = QLabel("")
        self._lbl_route.setWordWrap(True)
        self._lbl_route.setStyleSheet("color:#555; font-size:9px;")
        self._lbl_route.setToolTip("The calibration of this sample: the monitor, the R values from "
                                   "Standards it goes through, and the calibration it ends on. "
                                   "Change it with Calibration….")
        self._lbl_route_hdr = QLabel("Calibration:")
        form.addRow(self._lbl_route_hdr, self._lbl_route)

        self._flux_box = QFrame()
        self._flux_box.setFrameStyle(QFrame.StyledPanel)
        self._flux_box.setStyleSheet(
            "QFrame{background:#f9f9f9;border:1px solid #ccc;border-radius:4px;margin-top:4px;}")
        _fb = QFormLayout(self._flux_box)
        _fb.setContentsMargins(6,6,6,6); _fb.setSpacing(3)
        _fb.setLabelAlignment(Qt.AlignRight)

        self._le_jVal = QLineEdit(repr(self.cfg['jVal']))
        self._le_jVal.setPlaceholderText("e.g. 0.00271559")
        self._le_jVal.textChanged.connect(self._on_cfg)
        self._lbl_jVal = QLabel("J value:")
        _fb.addRow(self._lbl_jVal, self._le_jVal)

        self._le_jErrPct = QLineEdit(repr(self.cfg.get('jErrPct', 0.0)))
        self._le_jErrPct.setPlaceholderText("% 1σ, e.g. 0.20")
        self._le_jErrPct.textChanged.connect(self._on_cfg)
        self._lbl_jErrPct = QLabel("J error (% 1σ):")
        _fb.addRow(self._lbl_jErrPct, self._le_jErrPct)

        self._le_rStd = QLineEdit(f"{from_fcs_r(self.cfg['rStd'], self.cfg):.11g}")
        self._le_rStd.setPlaceholderText("e.g. 70.12345")
        self._le_rStd.textChanged.connect(self._on_cfg)
        self._lbl_rStd = QLabel("\u2074\u2070Ar/\u00b3\u2079Ar R standard:")
        _fb.addRow(self._lbl_rStd, self._le_rStd)

        self._le_rStdErr = QLineEdit(f"{self.cfg['rStdErrPct']:.8g}")
        self._le_rStdErr.textChanged.connect(self._on_cfg)
        self._cb_rStdErrMode = QComboBox()
        self._cb_rStdErrMode.addItem("% 1\u03c3",  'pct')
        self._cb_rStdErrMode.addItem("abs 1\u03c3", 'abs')
        self._cb_rStdErrMode.setMaximumWidth(72)
        self._cb_rStdErrMode.currentIndexChanged.connect(self._on_cfg)
        _rStdErr_w = QWidget()
        _rStdErr_l = QHBoxLayout(_rStdErr_w)
        _rStdErr_l.setContentsMargins(0, 0, 0, 0)
        _rStdErr_l.setSpacing(4)
        _rStdErr_l.addWidget(self._le_rStdErr)
        _rStdErr_l.addWidget(self._cb_rStdErrMode)
        self._rStdErr_w = _rStdErr_w
        self._lbl_rStdErr = QLabel("R std error:")
        _fb.addRow(self._lbl_rStdErr, self._rStdErr_w)

        self._le_stdAge = QLineEdit(repr(self.cfg['stdAgeMa']))
        self._le_stdAge.setPlaceholderText("Ma, e.g. 28.176")
        self._le_stdAge.textChanged.connect(self._on_cfg)
        self._lbl_stdAge = QLabel("Monitor age (Ma):")
        _fb.addRow(self._lbl_stdAge, self._le_stdAge)

        self._le_stdAgeErr = QLineEdit(repr(self.cfg['stdAgeErrMa']))
        self._le_stdAgeErr.setPlaceholderText("Ma 1\u03c3, e.g. 0.0115")
        self._le_stdAgeErr.textChanged.connect(self._on_cfg)
        self._lbl_stdAgeErr = QLabel("Monitor age error (Ma 1\u03c3):")
        _fb.addRow(self._lbl_stdAgeErr, self._le_stdAgeErr)

        # Whether the monitor-age error already includes λ ('external') or not
        # ('internal').  It follows the calibration preset (its external flag)
        # and is not a user control.
        self._std_age_err_type = ('external' if self.cfg.get('stdAgeErrType', 'internal')
                                  == 'external' else 'internal')

        self._le_lambda = QLineEdit(f"{self.cfg['lambdaYr']:.6e}")
        self._le_lambda.textChanged.connect(self._on_cfg)
        _fb.addRow("\u03bb (\u2074\u2070K, yr\u207b\u00b9):", self._le_lambda)

        self._le_lambdaErr = QLineEdit(f"{self.cfg['lambdaErrYr']:.6e}")
        self._le_lambdaErr.textChanged.connect(self._on_cfg)
        _fb.addRow("\u03c3\u03bb (yr\u207b\u00b9):", self._le_lambdaErr)

        # External-error box: intercalibrated (default) vs Min full external
        self._cb_externalMode = QComboBox()
        self._cb_externalMode.addItems(['Intercalibrated external', 'Full external'])
        self._cb_externalMode.setToolTip(
            "Which external uncertainty to report:\n"
            " Intercalibrated external (default): the monitor age with the decay\n"
            "          constant it was calibrated with. For Kuiper, Rivera and\n"
            "          Phillips it comes from R(FCs/A), the anchor age and λ.\n"
            "          This is also the error to use against U-Pb ages.\n"
            " Full external (Min et al., 2000): the uncertainty of the two partial\n"
            "          ⁴⁰K decay constants (about 1.2% of the age, 1σ) in place of the λ term,\n"
            "          for an independent test of the ⁴⁰K clock.")
        self._cb_externalMode.setCurrentIndex(1 if self.cfg.get('fullSystematics') else 0)
        self._cb_externalMode.activated.connect(self._on_cfg)

        form.addRow(self._flux_box)
        # The External-error selector lives on `form` (not inside _flux_box, which
        # is hidden in stack mode) so it is visible and identical in both modes.
        # set in Calibration… (Advanced); kept here, hidden, for the settings
        self._lbl_externalMode = QLabel("External error:")
        form.addRow(self._lbl_externalMode, self._cb_externalMode)
        self._lbl_externalMode.setVisible(False)
        self._cb_externalMode.setVisible(False)

        # "Apply to all", only visible when carousel has 2+ aliquots
        self._btn_calib_apply_all = QPushButton("Apply calibration to all aliquots")
        self._btn_calib_apply_all.setToolTip(
            "Give every aliquot this calibration (route, monitor age, λ). Each keeps its own flux.")
        self._btn_calib_apply_all.setVisible(False)
        self._btn_calib_apply_all.clicked.connect(self._carousel_apply_calib_to_all)
        form.addRow("", self._btn_calib_apply_all)

        # "Apply flux to all", assign the loaded disk flux to every aliquot by
        # identity (disk_id / hole_id), mirroring the Age distribution tool's
        # auto-match-all.  Only visible when the carousel has 2+ aliquots.
        self._btn_flux_apply_all = QPushButton("Apply flux to all aliquots")
        self._btn_flux_apply_all.setToolTip(
            "Assign the loaded disk flux to every aliquot in the carousel, matched "
            "by each aliquot's disk_id / hole_id (same auto-match as the single "
            "plateau). Each aliquot's plateau age is recomputed from its hole's J.")
        self._btn_flux_apply_all.setVisible(False)
        self._btn_flux_apply_all.clicked.connect(self._carousel_apply_flux_to_all)
        form.addRow("", self._btn_flux_apply_all)

        sv.addLayout(form)

        # Stack flux-groups section (visible only when stack mode is on)
        self._stack_flux_section = QWidget()
        _sfs_lay = QVBoxLayout(self._stack_flux_section)
        _sfs_lay.setContentsMargins(0, 6, 0, 0)
        _sfs_lay.setSpacing(4)
        _lbl_fg = QLabel("Flux Calibration Groups")
        _lbl_fg.setStyleSheet(
            "font-weight:bold; font-size:11px; margin-top:4px; color:#333;")
        _sfs_lay.addWidget(_lbl_fg)

        # ── Combination method toggle (WM + √MSWD vs GLS + ML) ──────
        _mrow = QHBoxLayout(); _mrow.setSpacing(8)
        _mrow.addWidget(QLabel("Combine:"))
        self._rb_wm = QRadioButton("WM + \u221aMSWD")
        self._rb_wm.setToolTip(
            "Weighted mean with analytical weights; the analytical tier is scaled\n"
            "by \u221aMSWD when overdispersed. Basis: n = steps or n = aliquots (next box).")
        self._rb_gls = QRadioButton("GLS + ML")
        self._rb_gls.setToolTip(
            "Generalised least squares + ML overdispersion (Vermeesch 2015 \u00a713):\n"
            "combine per-session plateau ages using the shared-J / FCs-\u03bb covariance,\n"
            "with the J correlation between holes from a loaded .jflux (\u03c1 between\n"
            "0 and 1).\n"
            "Both options use the same covariance; this one changes the estimator.\n"
            "Preferable when the aliquots scatter beyond their errors and those\n"
            "errors differ widely: ω evens out the weights.")
        self._rb_wm.setChecked(True)
        self._rb_wm.toggled.connect(self._on_stack_method)
        _mrow.addWidget(self._rb_wm)
        _mrow.addWidget(self._rb_gls)
        _mrow.addStretch()
        _sfs_lay.addLayout(_mrow)
        _mrow = QHBoxLayout(); _mrow.setSpacing(8)
        _mrow.addWidget(QLabel("WM basis:"))
        # WM+√MSWD basis: pool step-ratios (n=steps) or per-plateau ages (n=aliquots)
        self._cb_wm_n = QComboBox()
        self._cb_wm_n.addItem("n = steps",    'steps')
        self._cb_wm_n.addItem("n = aliquots", 'aliquots')
        self._cb_wm_n.setToolTip(
            "Basis for WM + √MSWD (ignored by GLS + ML, which is always per-aliquot):\n"
            "  n = steps, pool every plateau step's ratio into one weighted mean\n"
            "     (steps normalised to one J\u2081 in ratio space; each aliquot's J\n"
            "     error is carried once, by its share of the mean).\n"
            "  n = aliquots, collapse each plateau to its age first, then weighted-\n"
            "     mean the aliquot ages. Matches the Age distribution tool's pooled\n"
            "     result (same combine engine), so loading the aliquots there agrees.")
        self._cb_wm_n.setCurrentIndex(
            max(self._cb_wm_n.findData(self._wm_n_basis), 0))
        self._cb_wm_n.currentIndexChanged.connect(self._on_wm_n)
        _mrow.addWidget(self._cb_wm_n)
        _mrow.addStretch()
        _sfs_lay.addLayout(_mrow)

        _brow = QHBoxLayout(); _brow.setSpacing(6)
        self._btn_disk_flux = QPushButton("Disk flux\u2026")
        self._btn_disk_flux.setToolTip(
            "Load .jflux configs, view the disk layout, assign sessions to holes, "
            "and optionally apply the flux J, all in one window.")
        self._btn_disk_flux.clicked.connect(self._open_stack_flux)
        _brow.addWidget(self._btn_disk_flux)
        self._btn_calib_stack = QPushButton("Calibration\u2026")
        self._btn_calib_stack.setToolTip(
            "Recalculate all loaded step ages to a different decay constant / FC standard.")
        self._btn_calib_stack.clicked.connect(self._open_calibrate)
        _brow.addWidget(self._btn_calib_stack)
        _brow.addStretch()
        _sfs_lay.addLayout(_brow)
        self._lbl_flux_status = QLabel("No disk flux loaded, GLS + ML falls back "
                                       "to binary disk grouping.")
        self._lbl_flux_status.setStyleSheet("color:#888; font-size:9px; font-style:italic;")
        self._lbl_flux_status.setWordWrap(True)
        _sfs_lay.addWidget(self._lbl_flux_status)
        _fg_container = QWidget()
        self._s_flux_groups_layout = QVBoxLayout(_fg_container)
        self._s_flux_groups_layout.setSpacing(6)
        self._s_flux_groups_layout.setContentsMargins(0, 0, 2, 0)
        _fg_scroll = QScrollArea()
        _fg_scroll.setWidgetResizable(True)
        _fg_scroll.setWidget(_fg_container)
        _fg_scroll.setFrameShape(QFrame.NoFrame)
        _sfs_lay.addWidget(_fg_scroll, 1)
        self._stack_flux_section.setVisible(False)
        sv.addWidget(self._stack_flux_section, 1)

        tabs.addTab(st, "Settings")
        self._update_flux_ui()

        rv.addWidget(tabs, 1)

        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet(
            "font-size:10px; color:#555; padding:4px 2px 2px 2px;")
        rv.addWidget(self._lbl_status)
        root.addWidget(right)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _act(label, slot, shortcut=None):
        a = QAction(label)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(shortcut)
        return a

    def _show_error_breakdown(self):
        """Pie chart of where the plateau (or combined) age uncertainty comes from."""
        data = self._age_breakdown_data()
        if data is None:
            QMessageBox.information(self, 'Error breakdown',
                                    'No plateau found to break down.')
            return
        age, e_ana, e_int, e_ext, _r, cfg, title = data

        _, sig_scale, sig_lbl = self._disp()
        dlg = QDialog(self)
        dlg.setWindowTitle('Where the age uncertainty comes from')
        dlg.setMinimumSize(560, 470)
        lay = QVBoxLayout(dlg)
        fig = Figure(figsize=(5.4, 4.2))
        canvas = FigureCanvas(fig)
        lay.addWidget(canvas, 1)
        ax = fig.add_subplot(111)
        draw_budget_pie(ax, age_error_budget(e_ana, e_int, e_ext, age, cfg),
                        sig_scale, sig_lbl)
        route = external_route(cfg)
        ax.set_title(f'{title}   ({age:.3f} Ma)\nexternal via {route}', fontsize=9)
        fig.tight_layout()
        canvas.draw()
        note = QLabel(
            'Slices are shares of the external variance (they add to the '
            'external error). After combining many steps/aliquots the analytical '
            'slice shrinks, so the shared calibration systematic tends to dominate; '
            'with an astronomically anchored calibration the decay-constant slice '
            'is small.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#666; font-size:10px;')
        lay.addWidget(note)
        dlg.exec_()

    def _age_breakdown_data(self):
        """Gather (age, e_ana, e_int, e_ext [1σ], r_stats, cfg, title) for the
        current plateau, combined in stack mode, single otherwise. Returns None
        when there is nothing to break down."""
        if self._stack_mode:
            r, e_ana, e_int, e_ext = self._combined_plateau()
            if not r:
                return None
            used = [si for si, ss in enumerate(self._sessions)
                    if not ss.get('excluded') and ss.get('plateau')]
            cfg = self._eff_cfg_for(used[0]) if used else self._global_cfg
            title = f'Combined plateau  ({len(used)} aliquots)'
            return r['mean'], e_ana, e_int, e_ext, r, cfg, title
        p = self.plateau
        if not p:
            return None
        e_ana = p.get('e1s_ana', p['e1s_int'])
        e_int = p.get('e1s_int', e_ana)
        e_ext = p.get('e1s_ext', e_int)
        return (p['r']['mean'], e_ana, e_int, e_ext, p['r'],
                self._effective_single_cfg(), 'Plateau age')

    def _show_age_breakdown(self):
        """Info panel: statistical breakdown of the plateau (or combined) age,
        2σ confidence limit, MSWD / error-magnification story, the three error
        tiers, and a pie chart of the variance sources."""
        data = self._age_breakdown_data()
        if data is None:
            QMessageBox.information(
                self, 'Age breakdown',
                'No plateau found to break down, load data or adjust settings.')
            return
        age, e_ana, e_int, e_ext, r, cfg, title = data
        _age_dec = max(cfg.get('ageDec', 3), 3)
        _af = f'.{_age_dec}f'

        # raw (unscaled) SEM: a combined stack reports its scaled tier as 'sem'
        sem    = float(r.get('sem_raw', r.get('sem', 0.0)) or 0.0)
        mswd   = float(r.get('mswd', 0.0) or 0.0)
        n      = int(r.get('n', 0) or 0)
        prob   = float(r.get('prob', 1.0) or 0.0)
        p_cut  = float(cfg.get('pCut', 5.0))
        errMode = cfg.get('errMode', 'sem_mswd_2s')
        method = str(r.get('method', ''))

        # Error magnification = analytical tier ÷ raw SEM (the factor actually
        # applied to the weighted-mean SEM): √MSWD and any Student-t expansion.
        mag = (e_ana / sem) if sem > 0 else 1.0

        # Describe how the analytical error was scaled.
        if method.startswith('GLS + ML'):
            om = float(r.get('overdispersion', 0.0) or 0.0)
            scale_txt = (f"GLS + ML: overdispersion ω = {om:.4f} Ma (1σ) added to each "
                         "aliquot's analytical variance" if om > 0 else
                         "GLS + ML: no overdispersion between aliquots (ω = 0)")
        elif errMode == 'sem_mswd':
            if prob * 100.0 < p_cut and mswd > 0:
                tfac = t_crit(n - 1) if n > 1 else 2.0
                scale_txt = (f"overdispersed (p = {prob*100:.1f}% &lt; {p_cut:.0f}%): "
                             f"SEM × √MSWD × t₉₅(n−1)/2 "
                             f"= × {math.sqrt(mswd):.2f} × {tfac:.2f}/2")
            else:
                scale_txt = (f"within the cutoff (p = {prob*100:.1f}% ≥ {p_cut:.0f}%): "
                             f"no expansion, SEM used as it is")
        elif errMode == 'sem_mswd_2s':
            crit = mswd_critical(n) if n else float('inf')
            if n and n > 1 and mswd > crit:
                scale_txt = (f"overdispersed (MSWD {mswd:.2f} &gt; 95% limit {crit:.2f} "
                             f"at n={n}): SEM × √MSWD = × {math.sqrt(mswd):.2f}")
            else:
                scale_txt = (f"within the limit (MSWD {mswd:.2f} ≤ 95% limit "
                             f"{crit:.2f} at n={n}): SEM used as it is")
        else:
            scale_txt = "plain SEM (MSWD scaling off)"

        cls_txt = mswd_classify(mswd, n) if n else 'n/a'

        # External route label (mirrors the pie popup).
        route = external_route(cfg)

        # ── Build the dialog ─────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle('Age breakdown')
        dlg.setMinimumSize(720, 430)
        outer = QVBoxLayout(dlg)
        row = QHBoxLayout()
        outer.addLayout(row, 1)

        # Left: statistics text
        def _pm(e):
            return f"{e:{_af}}"
        stats_html = f"""
        <div style='font-family:sans-serif; font-size:12px; color:#222;'>
          <div style='font-size:15px; font-weight:bold; margin-bottom:2px;'>{title}</div>
          <div style='font-size:20px; font-weight:bold; color:#1565c0; margin-bottom:1px;'>
            {age:{_af}} ± {_pm(e_ext*2.0)} Ma</div>
          <div style='color:#666; margin-bottom:10px;'>2σ external confidence limit</div>

          <table cellspacing='0' cellpadding='3'>
            <tr><td colspan='3' style='font-weight:bold; color:#444;
                border-bottom:1px solid #ccc;'>Error tiers</td></tr>
            <tr><td>&nbsp;</td><td style='color:#888;'>±1σ</td><td style='color:#888;'>±2σ</td></tr>
            <tr><td>Analytical (mass-spec)</td>
                <td align='right'>{_pm(e_ana)}</td><td align='right'>{_pm(e_ana*2)}</td></tr>
            <tr><td>Internal (+ J / flux)</td>
                <td align='right'>{_pm(e_int)}</td><td align='right'>{_pm(e_int*2)}</td></tr>
            <tr><td style='font-weight:bold;'>External (+ FCs age / λ)</td>
                <td align='right' style='font-weight:bold;'>{_pm(e_ext)}</td>
                <td align='right' style='font-weight:bold;'>{_pm(e_ext*2)}</td></tr>
          </table>
          <div style='color:#888; font-size:11px; margin:2px 0 10px 0;'>
            External route: {route}</div>

          <table cellspacing='0' cellpadding='3'>
            <tr><td colspan='2' style='font-weight:bold; color:#444;
                border-bottom:1px solid #ccc;'>Scatter &amp; scaling</td></tr>
            <tr><td>Analyses combined (n)</td><td align='right'>{n}</td></tr>
            <tr><td>MSWD</td><td align='right'>{mswd:.2f} &nbsp;({cls_txt})</td></tr>
            <tr><td>Fit probability</td><td align='right'>{prob*100:.1f}%</td></tr>
            <tr><td>Raw SEM (1σ)</td><td align='right'>{_pm(sem)} Ma</td></tr>
            <tr><td style='font-weight:bold;'>Error magnification</td>
                <td align='right' style='font-weight:bold;'>× {mag:.2f}</td></tr>
          </table>
          <div style='color:#666; font-size:11px; margin-top:4px;'>
            Scaling: {scale_txt}.</div>
        </div>"""
        lbl = QLabel(stats_html)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(lbl, 3)

        # Right: variance pie (shares of the external variance)
        fig = Figure(figsize=(3.6, 3.6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        draw_budget_pie(ax, age_error_budget(e_ana, e_int, e_ext, age, cfg),
                        2.0, '2σ', centre_fmt=_af, legend_values=False)
        ax.set_title('External-variance shares', fontsize=9)
        fig.tight_layout()
        canvas.draw()
        row.addWidget(canvas, 2)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        outer.addWidget(bb)
        dlg.exec_()

    def _set_zoom_widgets_visible(self, vis):
        vis = vis and not getattr(self, '_isochron_mode', False)
        for w in (self._lbl_ymin, self._sp_ymin, self._lbl_ymax, self._sp_ymax):
            w.setVisible(vis)

    def _iso_kind(self):
        return 'inverse' if self.isochron_canvas._showing_inverse else 'normal'

    def _iso_lims_to_fields(self):
        """Show the typed ranges of the diagram on screen and apply them."""
        lim = self._iso_axis_lims[self._iso_kind()]
        for key, le in self._iso_lim_edits.items():
            v = lim.get(key)
            le.blockSignals(True)
            le.setText('' if v is None else f"{v:.6g}")
            le.blockSignals(False)
        self.isochron_canvas._user_lims = dict(lim) or None

    def _on_iso_lim_edit(self):
        lim = {}
        for key, le in self._iso_lim_edits.items():
            try:
                lim[key] = float(le.text().replace(',', '.'))
            except ValueError:
                le.setText('')
        self._iso_axis_lims[self._iso_kind()] = lim
        self.isochron_canvas._user_lims = dict(lim) or None
        self._do_redraw_isochron()

    def _on_iso_lim_auto(self):
        self._iso_axis_lims[self._iso_kind()] = {}
        self._iso_lims_to_fields()
        self._do_redraw_isochron()

    def _queue_redraw(self):
        self._redraw_timer.start()

    # ── Data lifecycle ────────────────────────────────────────────────────────
    def _sp_auto_match(self):
        """Auto-detect the single plateau's flux hole from self.cfg (disk_id /
        hole_id / irradiation), the same way the stack and the Age distribution tool do.
        Sets self._sp_flux_hole if a unique match is found. Only fills in when
        nothing is chosen yet, so it never overrides a manual pick."""
        if self._sp_flux_hole is not None or not self._disk_flux:
            return
        cfg = self.cfg
        m = self._flux_auto_match(cfg.get('disk_id'), cfg.get('hole_id'),
                                  cfg.get('irradiation'))
        if m is not None:
            (irr, disk), idx = m
            fx = self._disk_flux[(str(irr), str(disk))]
            hole = next((h for h, ix in fx['idx'].items() if ix == idx), None)
            if hole is not None:
                self._sp_flux_hole = (irr, disk, hole)

    def _refresh_sp_flux_state(self):
        """Update the label next to the single-plateau "Disk flux…" button."""
        lbl = getattr(self, '_lbl_sp_flux_state', None)
        if lbl is None:
            return
        fh = self._sp_flux_hole
        if fh is None:
            lbl.setText("(none)")
        else:
            irr, disk, hole = fh
            lbl.setText(f"{disk}/{hole}  ({irr})")

    def _open_calibrate(self):
        """Recalibrate already-loaded step ages to a new decay constant / FC standard.

        Works in both single and stack mode.  Old constants are read from cfg;
        _recalc_ages_in_steps is applied in-place to raw_steps / session raw_steps.
        """
        if self._stack_mode:
            if not self._sessions:
                QMessageBox.information(self, "Calibration", "No sessions loaded.")
                return
            # Read old constants from the stored cfg (flux-group overrides
            # included, the disk-flux hole overlay not), as the panel shows them.
            eff0        = self._stored_cfg_for(0)
            old_lam     = eff0.get('lambdaYr',    LAMBDA_DEFAULT)
            old_lam_err = eff0.get('lambdaErrYr', LAMBDA_ERR)
            old_fc      = eff0.get('stdAgeMa',    STD_AGE_DEFAULT)
            old_fc_err  = eff0.get('stdAgeErrMa', STD_AGE_ERR)
        else:
            if not self.raw_steps:
                QMessageBox.information(self, "Calibration", "No steps loaded.")
                return
            old_lam     = self.cfg.get('lambdaYr',    LAMBDA_DEFAULT)
            old_lam_err = self.cfg.get('lambdaErrYr', LAMBDA_ERR)
            old_fc      = self.cfg.get('stdAgeMa',    STD_AGE_DEFAULT)
            old_fc_err  = self.cfg.get('stdAgeErrMa', STD_AGE_ERR)

        # "Revert to original" is available whenever we have a saved snapshot.
        can_revert = (self._stack_mode
                      and any('orig_raw_steps' in s for s in self._sessions)
                      ) or (not self._stack_mode
                            and getattr(self, '_orig_raw_steps', None) is not None)
        if self._stack_mode:
            # The stored age source, not the effective one: a session on a
            # loaded hole is aged from the hole's R whatever its stored source.
            old_mode = eff0.get('ageFluxMode', 'ratio_R_std')
            # Stack mode always applies to every aliquot (per-group settings
            # are in the group panels), so the "Apply to all" box is not shown.
            n_al = 1
        else:
            old_mode = self.cfg.get('ageFluxMode', 'ratio_R_std')
            n_al = len(self._carousel) if len(self._carousel) >= 2 else 1
        old_cfg = self._stored_cfg_for(0) if self._stack_mode else dict(self.cfg)
        orig = (self._sessions[0].get('orig_calib_cfg', {}) if self._stack_mode
                else getattr(self, '_orig_calib_cfg', None) or {})
        dlg = _CalibrationDialog(old_cfg, parent=self, can_revert=can_revert,
                                 old_mode=old_mode, n_aliquots=n_al,
                                 origin=_calib_origin(old_cfg, orig),
                                 full_external=bool(self.cfg.get('fullSystematics')))
        result = dlg.exec_()
        if result == QDialog.Rejected:
            return
        if result == _CalibrationDialog.REVERT:
            self._revert_calibration()
            return

        p        = dlg.get_params()
        lam2     = p['lam2']
        lam2_err = p['lam2_err']
        fc2_ma   = p['fc2_ma']
        fc2_err  = p['fc2_err_ma']
        fc2_ext  = p['fc2_external']
        new_mode = p.get('age_mode', old_mode)
        rkeys    = p['route_keys']
        _ext_str = 'external' if fc2_ext else 'internal'
        cal_upd  = dict(lambdaYr=lam2, lambdaErrYr=lam2_err, stdAgeMa=fc2_ma,
                        stdAgeErrMa=fc2_err, stdAgeErrType=_ext_str)

        def _put(cfg):
            """New calibration into cfg; the measured monitor ratio is kept."""
            prod_old = cfg_monitor(cfg)[1]
            cfg.update(cal_upd)
            set_route(cfg, rkeys)
            if cfg.get('rStd'):
                cfg['rStd'] = cfg['rStd'] * prod_old / cfg_monitor(cfg)[1]

        status_txt = f"✓  {route_text(rkeys)},  {_AGEMODE_LABEL.get(new_mode, new_mode)}"

        if self._stack_mode:
            # The stored step ages and per-step J are recast from each
            # session's own calibration, as in single mode, so Imported ages
            # and a file J follow the new calibration (R modes recompute from
            # R anyway).  The age source is written only when it was changed
            # in the dialog; otherwise every group keeps its own.
            mode_upd = {} if new_mode == old_mode else dict(ageFluxMode=new_mode)
            for si, sess in enumerate(self._sessions):
                base   = self._stored_cfg_for(si)
                steps  = sess.get('raw_steps', [])
                if steps:
                    ing1 = (p['ing1'] if si == 0 else
                            monitor_ingrowth(base, base.get('lambdaYr', old_lam)))
                    _recalc_ages_in_steps(
                        steps, base.get('lambdaYr', old_lam),
                        base.get('stdAgeMa', old_fc),
                        _recast_j(self._eff_cfg_for(si)),
                        lam2, fc2_ma, fc2_err_ma=fc2_err,
                        ing1=ing1, ing2=p['ing2'])
                c = sess.setdefault('cfg', {})
                _put(c)
                c.update(mode_upd)

            # Update each flux group's flux_cfg and its panel widgets.
            # Updating the panel text fires textChanged → _on_flux_group_cfg,
            # which writes back to flux_cfg and triggers _stack_recompute for
            # ratio modes, so we block signals and do one explicit recompute.
            for gi, group in enumerate(self._flux_groups):
                _put(group['flux_cfg'])
                group['flux_cfg'].update(mode_upd)
                panel = group.get('panel')
                if panel is not None:
                    _cbm = getattr(panel, 'cb_mode', None)
                    if _cbm is not None and mode_upd:
                        for _i in range(_cbm.count()):
                            if _cbm.itemData(_i) == new_mode:
                                _cbm.blockSignals(True)
                                _cbm.setCurrentIndex(_i)
                                _cbm.blockSignals(False)
                                break
                    for w, txt in [
                        (panel.le_lambda,    f"{lam2:.8g}"),
                        (panel.le_lambdaErr, f"{lam2_err:.8g}"),
                        (panel.le_stdAge,    f"{fc2_ma:.8g}"),
                        (panel.le_stdAgeErr, f"{fc2_err:.8g}"),
                    ]:
                        w.blockSignals(True); w.setText(txt); w.blockSignals(False)
                    panel.std_age_err_type = _ext_str

            # Update global cfg fallback
            _put(self._global_cfg)
            self._global_cfg.update(mode_upd)
            for group in self._flux_groups:
                panel = group.get('panel')
                if panel is not None and getattr(panel, 'le_rStd', None) is not None:
                    v = group['flux_cfg'].get('rStd')
                    if v:
                        panel.le_rStd.blockSignals(True)
                        panel.le_rStd.setText(f"{from_fcs_r(v, group['flux_cfg']):.11g}")
                        panel.le_rStd.blockSignals(False)
            self._recompute_all_sessions()
        else:
            j1 = _recast_j(self._effective_single_cfg())
            _recalc_ages_in_steps(
                self.raw_steps, p['lam1'], old_fc, j1,
                lam2, fc2_ma, fc2_err_ma=fc2_err,
                ing1=p['ing1'], ing2=p['ing2'])
            rp = dict(lam2=lam2, lam2_err=lam2_err,
                      fc2_ma=fc2_ma, fc2_err_ma=fc2_err,
                      fc2_external=fc2_ext,
                      j1=j1, j1_err=0.0)
            prod_old = cfg_monitor(self.cfg)[1]
            self._sync_settings_after_recalc(rp, self.raw_steps)
            set_route(self.cfg, rkeys)
            if self.cfg.get('rStd'):
                self.cfg['rStd'] = self.cfg['rStd'] * prod_old / cfg_monitor(self.cfg)[1]
            self._le_rStd.blockSignals(True)
            self._le_rStd.setText(f"{from_fcs_r(self.cfg.get('rStd', 70.0), self.cfg):.11g}")
            self._le_rStd.blockSignals(False)
            if new_mode != self.cfg.get('ageFluxMode'):
                for _i in range(self._cb_ageFluxMode.count()):
                    if self._cb_ageFluxMode.itemData(_i) == new_mode:
                        # signals on: _on_cfg stores the mode and redraws
                        self._cb_ageFluxMode.setCurrentIndex(_i)
                        break
            self._update_all()
            if p.get('apply_all') and len(self._carousel) >= 2:
                self._carousel_save_current()
                self._carousel_recalibrate_others(p, old_mode)

        self._warn_flux_monitor(rkeys)
        if bool(p.get('full_external')) != bool(self.cfg.get('fullSystematics')):
            self._cb_externalMode.setCurrentIndex(1 if p.get('full_external') else 0)
            self._on_cfg()
            _w = full_systematics_warning(self.cfg) if p.get('full_external') else ''
            if _w:
                QMessageBox.information(self, "Full external", _w)

        if self._stack_mode:
            # Show status at the bottom of each flux-group panel (inside the scroll area)
            for group in self._flux_groups:
                plbl = getattr(group.get('panel'), 'lbl_calib_status', None)
                if plbl is not None:
                    plbl.setText(status_txt)
                    plbl.setVisible(True)
        else:
            # the route line under Calibration… shows the new calibration
            self._show_route_line()

    def _revert_calibration(self):
        """Restore ages and calibration constants to their state at file load time.

        Works for both single and stack mode.  The snapshots (orig_raw_steps /
        orig_calib_cfg) are taken once when a file is loaded; subsequent calls to
        Calibration… do not overwrite them, so this always reverts to the original
        file: not just one step back.
        """
        if self._stack_mode:
            reverted = []
            for si, sess in enumerate(self._sessions):
                if 'orig_raw_steps' not in sess:
                    continue
                # Restore raw_steps and cfg
                sess['raw_steps'] = copy.deepcopy(sess['orig_raw_steps'])
                orig_cfg = sess.get('orig_calib_cfg', {})
                sess.setdefault('cfg', {}).update(orig_cfg)
                reverted.append(sess.get('title', f'#{si+1}'))

            # Push restored constants back into flux groups + panels
            if reverted and self._sessions:
                orig_cfg = self._sessions[0].get('orig_calib_cfg', {})
                lam2     = orig_cfg.get('lambdaYr',    LAMBDA_DEFAULT)
                lam2_err = orig_cfg.get('lambdaErrYr', LAMBDA_ERR)
                fc2_ma   = orig_cfg.get('stdAgeMa',    STD_AGE_DEFAULT)
                fc2_err  = orig_cfg.get('stdAgeErrMa', STD_AGE_ERR)
                fc2_ext  = orig_cfg.get('stdAgeErrType', 'external') == 'external'
                _ext_str = 'external' if fc2_ext else 'internal'

                for gi, group in enumerate(self._flux_groups):
                    group['flux_cfg'].update(
                        lambdaYr=lam2, lambdaErrYr=lam2_err,
                        stdAgeMa=fc2_ma, stdAgeErrMa=fc2_err,
                        stdAgeErrType=_ext_str)
                    panel = group.get('panel')
                    if panel is not None:
                        for w, txt in [
                            (panel.le_lambda,    f"{lam2:.8g}"),
                            (panel.le_lambdaErr, f"{lam2_err:.8g}"),
                            (panel.le_stdAge,    f"{fc2_ma:.8g}"),
                            (panel.le_stdAgeErr, f"{fc2_err:.8g}"),
                        ]:
                            w.blockSignals(True); w.setText(txt); w.blockSignals(False)
                        panel.std_age_err_type = 'external' if fc2_ext else 'internal'
                        lbl = getattr(panel, 'lbl_calib_status', None)
                        if lbl is not None:
                            lbl.setVisible(False)

                self._global_cfg.update(
                    lambdaYr=lam2, lambdaErrYr=lam2_err,
                    stdAgeMa=fc2_ma, stdAgeErrMa=fc2_err,
                    stdAgeErrType=_ext_str)
                self._recompute_all_sessions()

            # Clear the stack calib status label
            lbl = getattr(self, '_lbl_calib_state', None)
            if lbl is not None:
                lbl.setVisible(False)
        else:
            orig_steps = getattr(self, '_orig_raw_steps', None)
            orig_cfg   = getattr(self, '_orig_calib_cfg', {})
            if orig_steps is None:
                QMessageBox.information(self, "Revert",
                    "No original snapshot found, load a file first.")
                return
            self.raw_steps = copy.deepcopy(orig_steps)
            self.cfg.update(orig_cfg)

            # Push restored constants back into the UI fields
            field_map = [
                (self._le_lambda,    'lambdaYr',    LAMBDA_DEFAULT),
                (self._le_lambdaErr, 'lambdaErrYr', LAMBDA_ERR),
                (self._le_stdAge,    'stdAgeMa',    STD_AGE_DEFAULT),
                (self._le_stdAgeErr, 'stdAgeErrMa', STD_AGE_ERR),
            ]
            for le, key, default in field_map:
                le.blockSignals(True)
                v = self.cfg.get(key, default)
                le.setText(f'{v:.8g}' if isinstance(v, float) else str(v))
                le.blockSignals(False)
            is_ext = self.cfg.get('stdAgeErrType', 'internal') == 'external'
            self._std_age_err_type = 'external' if is_ext else 'internal'
            self._sync_ext_mode()

            self._update_flux_ui()   # refresh flux-box visibility if ageFluxMode was restored
            self._update_all()

            # Clear the single calib status label
            lbl = getattr(self, '_lbl_calib_state', None)
            if lbl is not None:
                lbl.setVisible(False)

    def _open_single_flux(self):
        """Open the shared disk-flux dialog.  With 2+ aliquots in the carousel,
        assign the flux to every aliquot at once (like the Age distribution tool);
        otherwise a single-row assignment for the current plateau."""
        if len(self._carousel) >= 2 and not self._stack_mode:
            host = _CarouselFluxHost(self)
        else:
            host = _SingleFluxHost(self)
        FluxAssignDialog(self, host).exec_()
        self._refresh_sp_flux_state()

    # Calibration keys the flux overlay writes / restores.
    _FLUX_BAKE_KEYS = ('ageFluxMode', 'rStd', 'rStdErrPct', 'rStdErrMode',
                       'jVal', 'jErrPct', 'jForce')

    def _cfg_with_flux(self, base_cfg, flux_hole):
        """Return a copy of ``base_cfg`` with ``flux_hole``'s monitor ratio R
        applied as the R standard (R mode).

        ``flux_hole`` is an (irr, disk, hole) tuple or None. The hole's R (the
        calibration-independent monitor ratio, preferring the value stored in the
        .jflux) becomes rStd with its relative σ_R, as in the Age distribution
        tool, so the R-standard field shows the flux and the age is computed
        from r4039 / R. A flux without R (older files) gives a re-derived J
        instead. None → unchanged. Used both for the live computation and to
        write the assignment into a cfg (_bake_flux_into_cfg)."""
        cfg = dict(base_cfg)
        if flux_hole is not None:
            irr, disk, hole = flux_hole
            fx = self._disk_flux.get((str(irr), str(disk)))
            if fx is not None and str(hole) in fx['idx']:
                i = fx['idx'][str(hole)]
                R = flux_hole_rstd(fx, i, cfg)
                if R and R > 0:
                    rel = flux_hole_rel(fx, i)
                    relpct = (rel * 100.0 if rel is not None
                              else cfg.get('rStdErrPct', 0.0) or 0.0)
                    cfg['ageFluxMode'] = 'ratio_R_std'
                    cfg['rStd']        = R
                    cfg['rStdErrPct']  = relpct
                    cfg['rStdErrMode'] = 'pct'
                else:
                    # Legacy flux without a stored/derivable R → re-derived J.
                    Jf, Jsf = self._flux_hole_j(fx, i, cfg)
                    if Jf and Jf > 0:
                        cfg['ageFluxMode'] = 'ratio_J'
                        cfg['jVal']    = Jf
                        cfg['jErrPct'] = (Jsf / Jf * 100.0) if Jsf else 0.0
                        cfg['jForce']  = True
        return cfg

    def _bake_flux_into_cfg(self, cfg, flux_hole, base_holder):
        """Write ``flux_hole``'s R into ``cfg`` in place so the assignment is
        visible (R-standard field) and persists. ``base_holder`` is a dict that
        keeps a one-time snapshot of the pre-flux calibration under
        'flux_base_cfg', restored verbatim when ``flux_hole`` is None."""
        if flux_hole is not None:
            if 'flux_base_cfg' not in base_holder:
                base_holder['flux_base_cfg'] = {k: cfg.get(k)
                                                for k in self._FLUX_BAKE_KEYS}
            eff = self._cfg_with_flux(cfg, flux_hole)
            for k in self._FLUX_BAKE_KEYS:
                if k in eff:
                    cfg[k] = eff[k]
        else:
            base = base_holder.pop('flux_base_cfg', None)
            if base is not None:
                for k, v in base.items():
                    if v is None:
                        cfg.pop(k, None)
                    else:
                        cfg[k] = v

    def _bake_flux_into_slot(self, slot):
        """Bake slot['flux_hole'] into slot['cfg'] (snapshot stored on the slot)."""
        cfg = slot.get('cfg')
        if cfg is not None:
            self._bake_flux_into_cfg(cfg, slot.get('flux_hole'), slot)

    def _stack_flux_hole(self, si):
        """Resolve stack session ``si``'s assignment to an (irr, disk, hole)
        tuple, or None.  Mirrors _SingleFluxHost._resolve / _slot_flux_auto so
        the stack answers "which hole is this session on?" exactly the way the
        single-plateau and carousel modes do."""
        if not self._disk_flux or not (0 <= si < len(self._sessions)):
            return None
        ov = self._flux_override.get(si, 'auto')
        if ov == 'none':
            return None
        if ov != 'auto':
            irr, disk, hole = ov
            fx = self._disk_flux.get((str(irr), str(disk)))
            if fx is not None and str(hole) in fx['idx']:
                return (irr, disk, hole)
            return None
        cfg = self._sessions[si].get('cfg', {})
        m = self._flux_auto_match(cfg.get('disk_id'), cfg.get('hole_id'),
                                  cfg.get('irradiation'))
        if m is None:
            return None
        (irr, disk), idx = m
        fx = self._disk_flux[(str(irr), str(disk))]
        hole = next((h for h, ix in fx['idx'].items() if ix == idx), None)
        return (irr, disk, hole) if hole is not None else None

    def _bake_flux_into_session(self, si):
        """Write session ``si``'s matched hole R into its cfg as the R standard
        (as the carousel does), so the assignment reaches the ages and the flux
        groups, which are keyed on the calibration, put sessions on different
        holes in different groups."""
        if not (0 <= si < len(self._sessions)):
            return
        sess = self._sessions[si]
        cfg  = sess.get('cfg')
        if cfg is not None:
            self._bake_flux_into_cfg(cfg, self._stack_flux_hole(si), sess)

    def _effective_single_cfg(self):
        """self.cfg with the selected single-plateau flux hole's R overlaid.

        Selecting a hole adopts that hole's monitor ratio R (and σ_R) as the R
        standard for the single plateau; '(none)' leaves the typed values."""
        return self._cfg_with_flux(self.cfg, getattr(self, '_sp_flux_hole', None))

    def _update_all(self):
        self._show_route_line()
        eff          = self._effective_single_cfg()
        steps        = _compute_steps_with_age_mode(self.raw_steps, eff)
        self.steps_c = add_cumulative(steps)
        self.plateau = find_plateau(self.steps_c, eff)
        self.pset    = set(self.plateau['ois']) if self.plateau else set()
        # Cache plateau into active carousel slot so it's preserved on switch
        if 0 <= self._carousel_idx < len(self._carousel):
            self._carousel[self._carousel_idx]['plateau'] = self.plateau
        self._refresh_table()
        self._refresh_summary()
        self._queue_redraw()

    # ── Display-scale helper ────────────────────────────────────────────────────
    def _window_sessions(self, kca, sig_scale, age_dec):
        """The stack's sessions as drawn in the window: each legend entry
        carries the aliquot's plateau age ± internal error at the display
        level, and the ratio panel shows K/Ca when asked."""
        _af = f".{age_dec}f"
        names = [sess.get('title', '') for sess in self._sessions
                 if not sess.get('excluded', False)]
        pad = dict(zip(names, export.pad_names(names, 7.5)))
        out = []
        for sess in self._sessions:
            s2 = dict(sess)
            name = sess.get('title', '')
            p = sess.get('plateau')
            if p and not sess.get('excluded', False):
                e = (p.get('e1s_int') or 0.0) * sig_scale
                s2['legend_label'] = (f"{pad.get(name, name)}"
                                      f"{p['r']['mean']:{_af}} ± {e:{_af}} Ma")
            if kca:
                s2['steps_c'] = _kca_steps(sess.get('steps_c', []))
            out.append(s2)
        return out

    def _kca_view(self):
        """True when the window shows K/Ca instead of Ca/K."""
        return hasattr(self, '_cb_ratio') and self._cb_ratio.currentData() == 'kca'

    def _disp(self):
        """Current display setting as (display_1sigma, sig_scale, sig_lbl).

        Errors are stored at 1σ; multiply by sig_scale (1.0 for 1σ, 2.0 for 2σ)
        wherever a value is shown or exported.
        """
        one = hasattr(self, '_rb_disp_1s') and self._rb_disp_1s.isChecked()
        return one, (1.0 if one else 2.0), ('1σ' if one else '2σ')

    # ── Table ─────────────────────────────────────────────────────────────────
    def _refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.steps_c))

        # Per-step errors are stored at 1σ; show them at the selected display scale
        # so the table matches the spectrum plot.
        display_1sigma, sig_scale, sig_lbl = self._disp()
        _age_dec = self.cfg.get('ageDec', 2)
        _af = f'.{_age_dec}f'
        self.table.setHorizontalHeaderLabels(
            ['#', 'Pwr', 'Age', f'±{sig_lbl}', '%³⁹Ar',
             'K/Ca' if self._kca_view() else 'Ca/K',
             '%⁴⁰Ar*', '●', '⁴⁰/³⁹ R', f'±R ({sig_lbl})'])

        for i, s in enumerate(self.steps_c):
            status = s.get('status', 'normal')
            exc    = status == 'excluded'
            forced = status == 'forced'
            ip     = (i in self.pset)

            if exc:
                sym = '\u2715'
            elif forced:
                sym = '\u25c6'
            elif ip:
                sym = '\u25cf'
            else:
                sym = '\u25cb'

            r4039     = s.get('r4039',     0.0)
            err_r4039 = s.get('err_r4039', 0.0)
            cells = [
                s['label'],
                f"{s['power']:.2f}",
                f"{s['age']:{_af}}",
                f"{s['err1s'] * sig_scale:{_af}}",
                f"{s['pct39Ar']:.1f}",
                ((f"{1.0 / s['caK']:.3g}" if s['caK'] > 0 else '—')
                 if self._kca_view() else f"{s['caK']:.1f}"),
                f"{s['pct40']:.1f}",
                sym,
                f"{r4039:.4f}"     if r4039     > 0 else '—',
                f"±{err_r4039 * sig_scale:.4f}" if err_r4039 > 0 else '—',
            ]
            for j, v in enumerate(cells):
                item = QTableWidgetItem(v)
                item.setTextAlignment(
                    Qt.AlignCenter if j in (0, 1, 7)
                    else Qt.AlignRight | Qt.AlignVCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                if exc:
                    item.setForeground(QColor('#c0392b'))
                    item.setBackground(QColor('#fff0f0'))
                elif forced:
                    item.setForeground(QColor('#1565c0'))
                    item.setBackground(QColor('#e3f2fd'))
                    if j == 0:
                        f = item.font(); f.setBold(True); item.setFont(f)
                elif ip:
                    item.setForeground(QColor('#047857'))
                    item.setBackground(QColor('#f0fdf4'))
                    if j == 0:
                        f = item.font(); f.setBold(True); item.setFont(f)

                self.table.setItem(i, j, item)

        self.table.blockSignals(False)

    # ── Summary panel ─────────────────────────────────────────────────────────
    def _refresh_summary(self):
        display_1sigma, sig_scale, sig_lbl = self._disp()
        _age_dec = self.cfg.get('ageDec', 2)
        _af = f'.{_age_dec}f'

        ir = integrated_age(self.steps_c, self.cfg.get('lambdaYr', LAMBDA_DEFAULT))

        parts = []
        if ir:
            e_show = ir['sem'] * sig_scale          # sem is 1σ
            parts.append(
                f"Total-gas age: {ir['mean']:{_af}} \u00b1 {e_show:{_af}} Ma ({sig_lbl})")
        if not self.plateau and self.raw_steps:
            parts.append("No plateau found, adjust settings or force steps manually")

        # Age source mode
        mode_lbl = _AGEMODE_LABEL.get(self.cfg.get('ageFluxMode', 'imported'), 'imported')
        parts.append(f"Age source: {mode_lbl}")

        p = self.plateau
        if p:
            e1s_ana = p.get('e1s_ana', p['e1s_int'])
            e1s_ext = p.get('e1s_ext', p['e1s_int'])
            parts.append(f"Analytical: \u00b1{e1s_ana * sig_scale:{_af}} Ma ({sig_lbl})")
            parts.append(f"External: \u00b1{e1s_ext * sig_scale:{_af}} Ma ({sig_lbl})")

        self._lbl_status.setText("\n".join(parts))

    # ── Plot ──────────────────────────────────────────────────────────────────

    def _do_redraw(self):
        display_1sigma, sig_scale, sig_lbl = self._disp()
        if self._isochron_mode:
            self._do_redraw_isochron()
        elif self._stack_mode:
            yr = self._compute_age_range()
            for sp, v in [(self._sp_ymin, yr[0]), (self._sp_ymax, yr[1])]:
                sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)
            c_r_dict, c_e1s_ana, c_e1s_int, c_e1s_ext = self._combined_plateau()
            c_mean = c_r_dict['mean'] if c_r_dict else None
            _age_dec = self._global_cfg.get('ageDec', 2)
            kca = self._kca_view()
            self.overlay_canvas.redraw(
                self._window_sessions(kca, sig_scale, _age_dec), yr,
                global_cfg=self._global_cfg, combined_e1s=c_e1s_int,
                combined_e1s_ana=c_e1s_ana, combined_e1s_ext=c_e1s_ext,
                combined_mean=c_mean, combined_r_dict=c_r_dict,
                display_1sigma=display_1sigma, age_dec=_age_dec,
                cak_label='K/Ca' if kca else 'Ca/K',
                legend_title=f"Plateau age ± {sig_lbl} (internal)")
            if c_e1s_int is not None:
                _af = f'.{_age_dec}f'
                self._lbl_stack_errors.setText(
                    f"Analytical: ±{c_e1s_ana * sig_scale:{_af}} Ma ({sig_lbl})\n"
                    f"External: ±{c_e1s_ext * sig_scale:{_af}} Ma ({sig_lbl})")
            else:
                self._lbl_stack_errors.setText("")
            self._lbl_status.setText("")
            self._stack_refresh_table()
            self._refresh_flux_status()
        else:
            age_range = self._compute_age_range()
            self.canvas._lam = self.cfg.get('lambdaYr', LAMBDA_DEFAULT)
            kca = self._kca_view()
            self.canvas._cak_label = 'K/Ca' if kca else 'Ca/K'
            self.canvas.redraw(
                _kca_steps(self.steps_c) if kca else self.steps_c,
                self.plateau, self.pset,
                age_range, self.title_edit.text(),
                display_1sigma=display_1sigma, age_dec=self.cfg.get('ageDec', 2))
            self._refresh_table()
            self._refresh_summary()

    def _do_redraw_isochron(self):
        """Redraw the isochron canvas from whatever data is currently loaded."""
        steps_list, J, lam, title, colors, zoom = self._isochron_inputs()
        self.isochron_canvas._group_jrel = self._iso_jrel
        self.isochron_canvas._group_rho = self._iso_rho
        self.isochron_canvas.redraw(
            steps_list, J=J, lam=lam, title=title, colors=colors, zoom=zoom,
            cfg=getattr(self, '_iso_cfg', None) or self.cfg)

    @staticmethod
    def _iso_J(eff, steps):
        """J for the isochron age: the J the plateau ages are computed with
        (_aged_J), else, for imported ages, the median J stored with the steps."""
        J = _aged_J(eff, steps)[0]
        if J:
            return J
        jv = [s['j_val'] for s in steps
              if s.get('j_val', 0) > 0 and s.get('status', 'normal') != 'excluded']
        return float(np.median(jv)) if jv else None

    @staticmethod
    def _tag_isochron_steps(raw, pset, scale=1.0, j_val=None, j_err=None):
        """Copies of the steps for the isochron: steps outside the plateau
        tagged 'excluded' (drawn red, kept out of the fit); with scale != 1 the
        ³⁹Ar in both ratios is put on another J (³⁹Ar/⁴⁰Ar and ³⁹Ar/³⁶Ar times
        J_ref/J_i, and their errors), and the J used is written to the steps."""
        out = []
        for i, s in enumerate(raw):
            s2 = dict(s)
            if pset and i not in pset:
                s2['status'] = 'excluded'
            if scale != 1.0:
                for k in ('iso_nx', 'iso_nsx', 'iso_ix', 'iso_isx'):
                    if s2.get(k) is not None:
                        s2[k] = s2[k] * scale
            if j_val:
                s2['j_val'] = j_val
                s2['j_err'] = j_err or 0.0
            out.append(s2)
        return out

    def _isochron_inputs(self):
        """(steps_list, J, lam, title, colors, zoom) for the isochron panel.

        Shared by the on-screen isochron and the export dialog so the two can
        never drift apart.

        J is the one the plateau ages are computed with (R_std, typed J or an
        assigned flux hole), not the j_val stored in the file, which goes stale
        after a recalibration or a flux assignment.  In stack mode the aliquots
        excluded from the combined age (×) are left out, and every aliquot's
        ³⁹Ar is put on one J, J_ref, the best-monitored aliquot's: aliquots
        from different irradiation positions do not share a ³⁹Ar scale, and
        one York fit needs them to.  As in the n = steps combination, each
        aliquot's own J error is carried in the internal tier, once per
        aliquot, correlated by hole / disk (self._iso_jrel / self._iso_rho).
        self._iso_cfg / self._iso_si record the calibration used and the
        session index of each group (for colours and legend names)."""
        cfg = self.cfg
        lam = cfg.get('lambdaYr', LAMBDA_DEFAULT)
        J = None
        self._iso_cfg = cfg
        self._iso_si = None
        self._iso_jrel = None
        self._iso_rho = None

        if self._stack_mode:
            used = [si for si, sess in enumerate(self._sessions)
                    if not sess.get('excluded', False)]
            Js = {si: _aged_J(self._eff_cfg_for(si), self._sessions[si].get('steps_c', []))
                  for si in used}
            known = [si for si in used if Js[si][0]]
            ref = min(known, key=lambda si: Js[si][1]) if known else None
            if ref is not None:
                J, rel = Js[ref]
                self._iso_cfg = self._eff_cfg_for(ref)
                lam = self._iso_cfg.get('lambdaYr', LAMBDA_DEFAULT)
            steps_list = []
            for si in used:
                sess = self._sessions[si]
                Ji = Js[si][0]
                scale = (J / Ji) if (J and Ji) else 1.0
                steps_list.append(self._tag_isochron_steps(
                    sess.get('raw_steps', []), sess.get('pset', set()), scale,
                    j_val=J, j_err=(J * rel if J else None)))
            self._iso_si = used
            if J and used and all(Js[si][0] for si in used):
                self._iso_jrel = [Js[si][1] for si in used]
                self._iso_rho = self._build_stack_pcorr(used)
            # the same colour as the aliquot's age spectrum
            colors = [SAMPLE_COLORS[si % len(SAMPLE_COLORS)] for si in used]
            title = 'Stacked Isochron'
        else:
            # Pass all steps; tag non-plateau as 'excluded' for display.
            J, rel = _aged_J(self._effective_single_cfg(), self.steps_c or [])
            tagged = self._tag_isochron_steps(
                self.raw_steps or [], self.pset or set(),
                j_val=J, j_err=(J * rel if J else None))
            steps_list = [tagged] if tagged else []
            colors = [IsochronCanvas._PALETTE[0]]
            title = self.title_edit.text()

        if not J:
            # imported ages: fall back to the J stored with the steps
            all_flat = [s for grp in steps_list for s in grp]
            j_vals = [s['j_val'] for s in all_flat if s.get('j_val', 0.0) > 0]
            J = float(np.median(j_vals)) if j_vals else (
                cfg.get('jVal', 0.0) or None
                if cfg.get('ageFluxMode') == 'ratio_J' else None)

        zoom = cfg.get('zoomMode', 'auto')
        return steps_list, J, lam, title, colors, zoom

    def _compute_age_range(self):
        if self._stack_mode:
            if self.cfg.get('zoomMode', 'auto') == 'manual':
                return (self._sp_ymin.value(), self._sp_ymax.value())
            edges = []
            # excluded (×) aliquots are not drawn, so they don't set the range
            shown = ([sess for sess in self._sessions if not sess.get('excluded', False)]
                     or self._sessions)
            for sess in shown:
                for s in sess.get('steps_c', []):
                    edges += [s['age'] - 2.0 * s['err1s'], s['age'] + 2.0 * s['err1s']]
            if not edges:
                return (50.0, 60.0)
            lo, hi = min(edges), max(edges)
            d = (hi - lo) or 1.0
            return (lo - d * 0.12, hi + d * 0.12)

        mode = self.cfg.get('zoomMode', 'auto')
        if mode == 'manual':
            return (float(self.cfg.get('yMin', self._sp_ymin.value())),
                    float(self.cfg.get('yMax', self._sp_ymax.value())))
        if mode == 'plateau' and self.plateau:
            plat_steps = [self.steps_c[i] for i in self.plateau['ois']
                          if i < len(self.steps_c)]
            if plat_steps:
                edges = ([s['age'] - 2.0 * s['err1s'] for s in plat_steps] +
                         [s['age'] + 2.0 * s['err1s'] for s in plat_steps])
                lo, hi = min(edges), max(edges)
                d = (hi - lo) or 1.0
                return (lo - d * 0.5, hi + d * 0.5)
        if not self.steps_c:
            return (50.0, 60.0)
        edges = ([s['age'] - 2.0 * s['err1s'] for s in self.steps_c] +
                 [s['age'] + 2.0 * s['err1s'] for s in self.steps_c])
        lo, hi = min(edges), max(edges)
        d = (hi - lo) or 1.0
        return (lo - d * 0.14, hi + d * 0.14)

    # ─── Window close ────────────────────────────────────────────────────────
    def closeEvent(self, event):
        # Explicitly clear matplotlib figures before Qt tears down the widgets.
        # This prevents the FigureCanvasQTAgg destructor from accessing already-
        # deleted Qt objects, which causes SIGABRT (exit code 6) on macOS.
        for canvas in (self.canvas, self.overlay_canvas, self.isochron_canvas):
            try:
                canvas.fig.clf()
            except Exception:
                pass
        super().closeEvent(event)

    # ─── Events ───────────────────────────────────────────────────────────────
    def _on_canvas_step_clicked(self, ti):
        """Called when the user clicks a step bar directly on the spectrum plot
        (single-session mode only). Same logic as _on_cell_clicked."""
        if ti < len(self.raw_steps):
            self._on_cell_clicked(ti, 0)

    def _on_cell_clicked(self, row, _col):
        if row < len(self.raw_steps):
            s = self.raw_steps[row]
            if self._btn_pick.isChecked():
                # Pick mode: toggle forced (blue) in/out of plateau
                s['status'] = 'normal' if s.get('status') == 'forced' else 'forced'
            else:
                # Normal mode: toggle excluded (red) / normal only
                s['status'] = ('normal'
                               if s.get('status', 'normal') == 'excluded'
                               else 'excluded')
            self._update_all()

    def _on_pick_plateau_toggled(self, checked):
        if checked:
            self._hint.setText("Pick mode: click steps to toggle them in/out of the plateau (blue)")
            self._btn_pick.setStyleSheet(
                "background-color: #1565c0; color: white; font-weight: bold;")
        else:
            self._hint.setText(
                "Click step: toggle excluded (red) / normal  ·  Use Pick mode to force plateau steps (blue)")
            self._btn_pick.setStyleSheet("")

    def _on_p40max(self, v):
        global _P40_MAX
        _P40_MAX = float(v) if v > 0 else None
        self._global_cfg['p40Max'] = float(v)
        self._queue_redraw()

    def _set_p40max(self, v):
        """Show a saved %40Ar* top in the zoom row and apply it."""
        sp = getattr(self, '_sp_p40max', None)
        if sp is not None and v is not None:
            sp.setValue(float(v))

    def _on_zoom_spin(self, *_):
        # the manual Y range belongs to the sample being viewed
        self.cfg['yMin'] = self._sp_ymin.value()
        self._cfg_yMax_sync()
        self._queue_redraw()

    def _cfg_yMax_sync(self):
        self.cfg['yMax'] = self._sp_ymax.value()

    def _on_zoom_btn(self, btn):
        new_mode = btn.property('zoomMode')
        if new_mode == 'manual' and self.cfg.get('zoomMode') != 'manual' \
                and not self._stack_mode:
            # start the manual range from what is shown now, not from a stale one
            try:
                lo, hi = self._compute_age_range()
                for sp, v in ((self._sp_ymin, lo), (self._sp_ymax, hi)):
                    sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)
                self.cfg['yMin'], self.cfg['yMax'] = lo, hi
            except Exception:
                pass
        self.cfg['zoomMode'] = new_mode
        self._set_zoom_widgets_visible(self.cfg['zoomMode'] == 'manual')
        self._queue_redraw()

    def _update_table_ratio_cols(self):
        """Show/hide the ⁴⁰/³⁹ R and ±err R columns based on the current age mode."""
        show = self.cfg.get('ageFluxMode', 'imported') != 'imported'
        for col in (8, 9):
            self.table.setColumnHidden(col, not show)

    def _update_flux_ui(self):
        self._update_table_ratio_cols()
        if self._stack_mode:
            return  # flux section always hidden in stack mode
        mode = self.cfg.get('ageFluxMode', 'imported')
        is_ratio = mode != 'imported'
        is_j = is_ratio and (mode == 'ratio_J')
        # The box frame stays visible in all single-mode states so λ and FC fields
        # remain accessible for error-tier reference even in imported mode.
        # Row visibility is controlled per field below.
        for w in (self._lbl_jVal, self._le_jVal, self._lbl_jErrPct, self._le_jErrPct):
            w.setVisible(is_j)
        for w in (self._lbl_rStd, self._le_rStd, self._lbl_rStdErr, self._rStdErr_w):
            w.setVisible(is_ratio and not is_j)
        # FC age rows: visible in imported + ratio_R_std; hidden in ratio_J,
        # where the typed J sets the ages.  The FC age and its error still set
        # the external tier there (Calibration… changes them).
        for w in (self._lbl_stdAge, self._le_stdAge,
                  self._lbl_stdAgeErr, self._le_stdAgeErr):
            w.setVisible(not is_j)
        # FC age ± type row stays hidden (auto-driven; replaced by External error)

    def _sync_ext_mode(self):
        """Reflect self.cfg['fullSystematics'] in the External-error combobox
        without re-triggering _on_cfg. Called wherever cfg is loaded/switched."""
        cb = getattr(self, '_cb_externalMode', None)
        if cb is None:
            return
        cb.blockSignals(True)
        cb.setCurrentIndex(1 if self.cfg.get('fullSystematics') else 0)
        cb.blockSignals(False)

    def _on_cfg(self):
        # When the user manually edits a J/R/mode field, detach from the
        # disk-flux hole so their typed value takes effect.
        sndr = self.sender()
        _flux_fields = (self._le_jVal, self._le_jErrPct,
                        self._le_rStd, self._le_rStdErr,
                        self._cb_ageFluxMode, self._cb_rStdErrMode)
        if (sndr in _flux_fields
                and getattr(self, '_sp_flux_hole', None) is not None):
            self._sp_flux_hole = None
            # The typed values are now the base: drop the pre-flux snapshot, or
            # the next flux Apply with no hole restores it and reverts the
            # age source / R_std the user just set (e.g. back to Imported ages).
            self._sp_flux_base.pop('flux_base_cfg', None)
            if 0 <= self._carousel_idx < len(self._carousel):
                self._carousel[self._carousel_idx].pop('flux_base_cfg', None)
            self._refresh_sp_flux_state()

        # When rStdErrMode combobox changes, convert the displayed value so
        # the number stays numerically equivalent (e.g. 0.035 % → 0.000245 abs).
        if sndr is self._cb_rStdErrMode:
            old_mode = self.cfg.get('rStdErrMode', 'pct')
            new_mode = self._cb_rStdErrMode.currentData()
            if old_mode != new_mode:
                try:
                    v = float(self._le_rStdErr.text())
                    r = float(self._le_rStd.text() or '70.0')
                    if r > 0:
                        cv = (v / 100.0 * r) if new_mode == 'abs' else (v / r * 100.0)
                        self._le_rStdErr.blockSignals(True)
                        self._le_rStdErr.setText(f'{cv:.6g}')
                        self._le_rStdErr.blockSignals(False)
                except (ValueError, ZeroDivisionError):
                    pass

        def _f(le, default=0.0):
            try:    return float(le.text())
            except (ValueError, TypeError): return default
        self.cfg.update(dict(
            minPct        = self._sp_minPct.value(),
            minN          = self._sp_minN.value(),
            pCut          = self._sp_pCut.value(),
            ageDec        = self._sp_ageDec.value(),
            errMode       = self._cb_errMode.currentData(),
            ageFluxMode   = self._cb_ageFluxMode.currentData(),
            jVal          = _f(self._le_jVal,      1e-3),
            jErrPct       = _f(self._le_jErrPct,   0.0),
            rStd          = to_fcs_r(_f(self._le_rStd, 70.0), self.cfg),
            rStdErrPct    = _f(self._le_rStdErr,   0.10),
            rStdErrMode   = self._cb_rStdErrMode.currentData(),
            stdAgeMa      = _f(self._le_stdAge,     STD_AGE_DEFAULT),
            stdAgeErrMa   = _f(self._le_stdAgeErr,  STD_AGE_ERR),
            stdAgeErrType = self._std_age_err_type,
            lambdaYr      = _f(self._le_lambda,     LAMBDA_DEFAULT),
            lambdaErrYr   = _f(self._le_lambdaErr,  LAMBDA_ERR),
            fullSystematics = (self._cb_externalMode.currentIndex() == 1),
        ))
        # In ratio_J mode the user controls J explicitly: the typed J must win
        # over any per-step j_val values baked into the imported file. jForce
        # is cleared for other modes so per-step values are preserved there.
        if self.cfg.get('ageFluxMode') == 'ratio_J':
            self.cfg['jForce'] = True
        else:
            self.cfg.pop('jForce', None)
        sync_route(self.cfg)
        self._show_route_line()
        # Coherence check: warn once when Full external is selected over a
        # jointly calibrated λ (e.g. Carter), which is self-contradictory.
        if sndr is getattr(self, '_cb_externalMode', None) and self.cfg.get('fullSystematics'):
            _w = full_systematics_warning(self.cfg)
            if _w:
                QMessageBox.information(self, "Full external", _w)
        # Warn once each time the calibration becomes one of Renne et al.
        _cw = calibration_warning(self.cfg)
        if _cw and _cw != getattr(self, '_calib_warned', ''):
            QMessageBox.information(self, "Calibration", _cw)
        self._calib_warned = _cw
        _cn = ' '.join(calibration_notes(self.cfg))
        if _cn and _cn != getattr(self, '_calib_noted', ''):
            QMessageBox.information(self, "Calibration", _cn)
        self._calib_noted = _cn
        self._update_flux_ui()
        # Sync plateau detection keys + display to _global_cfg so stack mode uses same values
        for _k in ('minPct', 'minN', 'pCut', 'errMode', 'ageDec', 'fullSystematics'):
            self._global_cfg[_k] = self.cfg[_k]
        # Keep active carousel slot's cfg in sync
        if 0 <= self._carousel_idx < len(self._carousel):
            self._carousel[self._carousel_idx]['cfg'] = self.cfg.copy()
        # Decimal places are a display setting, not a per-aliquot property: push
        # it to every carousel slot and stack session so the whole window reads
        # at one precision instead of changing as you page through the carousel.
        _dec = self.cfg.get('ageDec', 2)
        for _slot in self._carousel:
            if isinstance(_slot.get('cfg'), dict):
                _slot['cfg']['ageDec'] = _dec
        for _sess in self._sessions:
            if isinstance(_sess.get('cfg'), dict):
                _sess['cfg']['ageDec'] = _dec

        if self._stack_mode:
            for sess in self._sessions:
                self._stack_recompute(sess)
            self._stack_refresh_table()
            self._queue_redraw()
        elif self.raw_steps:
            self._update_all()
        elif self.steps_c:
            self.plateau = find_plateau(self.steps_c, self.cfg)
            self.pset    = set(self.plateau['ois']) if self.plateau else set()
            self._refresh_table()
            self._refresh_summary()
            self._queue_redraw()

    # ── Import ────────────────────────────────────────────────────────────────
    def import_file(self):
        path, _ = dfold.get_open_file_name(
            self, "Import Data", "",
            "Excel / CSV (*.xlsx *.csv);;All files (*)")
        if not path:
            return
        try:
            df = (pd.read_csv(path)
                  if path.lower().endswith('.csv')
                  else pd.read_excel(path))
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return
        self._process_df(df, os.path.basename(path))

    def paste_data(self):
        text = QApplication.clipboard().text().strip()
        if not text:
            QMessageBox.warning(self, "Clipboard Empty",
                                "Nothing found in clipboard.")
            return
        import io
        try:
            df = pd.read_csv(io.StringIO(text), sep='\t', header=None, dtype=str)
            try:
                float(str(df.iloc[0, 1]))
                has_header = False
            except (ValueError, IndexError):
                has_header = True

            if has_header:
                df.columns = df.iloc[0].astype(str).str.strip()
                df = df.iloc[1:].reset_index(drop=True)
                self._process_df(df, 'clipboard')
            else:
                self._import_positional(df)
        except Exception as exc:
            QMessageBox.critical(self, "Paste Error", str(exc))

    def _sync_j_from_steps(self, steps):
        """Update jVal / jErrPct in the settings tab from per-step J data (no recalc needed)."""
        _jv = [s.get('j_val', 0.0) for s in steps if s.get('j_val', 0.0) > 0]
        if not _jv:
            return
        _j_med = float(np.median(_jv))
        self.cfg['jVal'] = _j_med
        self._le_jVal.blockSignals(True)
        self._le_jVal.setText(f"{_j_med:.10g}")
        self._le_jVal.blockSignals(False)
        _je = [s.get('j_err', 0.0) for s in steps if s.get('j_err', 0.0) > 0]
        if _je:
            _pct = float(np.median(_je)) / _j_med * 100.0
            self.cfg['jErrPct'] = _pct
            self._le_jErrPct.blockSignals(True)
            self._le_jErrPct.setText(f"{_pct:.6g}")
            self._le_jErrPct.blockSignals(False)

    def _warn_flux_monitor(self, cfg):
        """Warn when a disk flux in use was measured on another monitor than
        the one cfg's route starts from (its R values are then read as that
        monitor's; arar_math.monitor_mismatch_note)."""
        holes = ([self._stack_flux_hole(si) for si in range(len(self._sessions))]
                 if self._stack_mode else
                 [self._sp_flux_hole] + [sl.get('flux_hole') for sl in self._carousel])
        mons = {}
        for h in holes:
            fx = self._disk_flux.get((str(h[0]), str(h[1]))) if h else None
            if fx is not None:
                mons.setdefault(fx.get('monitor', 'FCs'), set()).add(str(h[1]))
        notes = [monitor_mismatch_note(cfg, m, f"the disk flux of disk {', '.join(sorted(d))}")
                 for m, d in mons.items()]
        notes = [n for n in notes if n]
        if notes:
            QMessageBox.warning(self, "Calibration", '\n\n'.join(notes))

    def _show_route_line(self):
        lbl = getattr(self, '_lbl_route', None)
        if lbl is not None:
            try:
                r = route_from_cfg(self.cfg)
                lbl.setText(route_line(r) + (' · full external'
                                             if self.cfg.get('fullSystematics') else ''))
                lbl.setToolTip(route_text(r))
            except (ValueError, KeyError, TypeError):
                lbl.setText('')

    def _sync_settings_after_recalc(self, rp, steps):
        """Update settings-tab widgets to reflect the new calibration after recalculation."""
        # J value and J error % (overrides _sync_j_from_steps result with recalc-aware values)
        _j2v  = [s.get('j_val', 0.0) for s in steps if s.get('j_val', 0.0) > 0]
        _j2e  = [s.get('j_err', 0.0) for s in steps if s.get('j_err', 0.0) > 0]
        if _j2v:
            _j2 = float(np.median(_j2v))
            self.cfg['jVal'] = _j2
            self._le_jVal.blockSignals(True)
            self._le_jVal.setText(f"{_j2:.10g}")
            self._le_jVal.blockSignals(False)
            # jErrPct: prefer per-step j_err; fall back to dialog's absolute j_err entry
            if _j2e:
                _pct = float(np.median(_j2e)) / _j2 * 100.0
            elif rp.get('j1_err', 0.0) > 0 and rp.get('j1', 0.0) > 0:
                _pct = rp['j1_err'] / rp['j1'] * 100.0
            else:
                _pct = 0.0
            self.cfg['jErrPct'] = _pct
            self._le_jErrPct.blockSignals(True)
            self._le_jErrPct.setText(f"{_pct:.6g}")
            self._le_jErrPct.blockSignals(False)
        _pairs = [
            (self._le_lambda,    'lambdaYr',    rp['lam2'],
             f"{rp['lam2']:.6e}"),
            (self._le_lambdaErr, 'lambdaErrYr', rp.get('lam2_err', 0.0),
             f"{rp.get('lam2_err', 0.0):.6e}"),
            (self._le_stdAge,    'stdAgeMa',    rp['fc2_ma'],
             f"{rp['fc2_ma']:.4f}"),
            (self._le_stdAgeErr, 'stdAgeErrMa', rp.get('fc2_err_ma', 0.0),
             f"{rp.get('fc2_err_ma', 0.0):.4f}"),
        ]
        for _w, _key, _val, _txt in _pairs:
            self.cfg[_key] = _val
            _w.blockSignals(True)
            _w.setText(_txt)
            _w.blockSignals(False)
        is_ext = rp.get('fc2_external', True)
        self.cfg['stdAgeErrType'] = 'external' if is_ext else 'internal'
        self._std_age_err_type = self.cfg['stdAgeErrType']
        self._sync_ext_mode()

    def _finalize_parsed_steps(self, steps, dlg):
        """Apply column-dialog post-processing to freshly parsed steps.

        Used by _process_df (file import and clipboard paste): optional relabel
        and Ca/K inversion, the 2σ→1σ conversion, J sync, optional
        decay-constant recalibration, and the optional "compute ages from R
        and J" mode switch.  Steps are modified in place.
        """
        if dlg.auto_label:
            for i, s in enumerate(steps):
                s['label'] = _alpha_label(i)
        if dlg.invert_cak:
            for s in steps:
                kca = s.get('caK', 0.0)
                if kca > 0:
                    s['caK'] = 1.0 / kca
                    ekca = s.get('err_cak', 0.0)
                    if ekca > 0:
                        s['err_cak'] = ekca / (kca * kca)
        if dlg.err_sigma == 2:
            # Pasted values are 2σ; memory keeps everything at 1σ.
            for s in steps:
                for k in ('err1s', 'err_r4039', 'err_cak'):
                    s[k] = s.get(k, 0.0) * 0.5
        # Auto-update J settings from pasted data (so ratio_J mode is ready to use)
        self._sync_j_from_steps(steps)
        rp = dlg.recalc_params
        if rp:
            _recalc_ages_in_steps(steps,
                                  rp['lam1'], rp['fc1_ma'], rp['j1'],
                                  rp['lam2'], rp['fc2_ma'],
                                  fc2_err_ma=rp.get('fc2_err_ma', 0.0))
            self._sync_settings_after_recalc(rp, steps)
        if dlg.compute_from_r:
            self._apply_compute_from_r(steps, dlg)

    def _apply_compute_from_r(self, steps, dlg):
        """Switch to ratio_J mode using the dialog's λ / FC inputs (the dialog's
        "Compute ages from R and J" option).  R_std is back-computed from the
        median J so the existing R ratios reproduce the chosen ages."""
        _cfr_lam     = dlg.compute_lambda
        _cfr_lam_err = dlg.compute_lam_err
        _cfr_fc_ma   = dlg.compute_fc_ma
        _cfr_fc_err  = dlg.compute_fc_err_ma
        _cfr_fc_ext  = dlg.compute_fc_external
        _j_med_list  = [s.get('j_val', 0.0) for s in steps if s.get('j_val', 0.0) > 0]
        _j_med = float(np.median(_j_med_list)) if _j_med_list else self.cfg.get('jVal', 1e-3)
        _cfr_r_std = (fc_ingrowth(_cfr_fc_ma, _cfr_lam, _cfr_fc_err) / _j_med
                      if _j_med > 0 else 70.0)
        self.cfg.update({
            'ageFluxMode':   'ratio_J',
            'lambdaYr':      _cfr_lam,
            'lambdaErrYr':   _cfr_lam_err,
            'stdAgeMa':      _cfr_fc_ma,
            'stdAgeErrMa':   _cfr_fc_err,
            'stdAgeErrType': 'external' if _cfr_fc_ext else 'internal',
            'rStd':          _cfr_r_std,
        })
        for _i in range(self._cb_ageFluxMode.count()):
            if self._cb_ageFluxMode.itemData(_i) == 'ratio_J':
                self._cb_ageFluxMode.blockSignals(True)
                self._cb_ageFluxMode.setCurrentIndex(_i)
                self._cb_ageFluxMode.blockSignals(False)
                break
        for _le, _txt in [
            (self._le_lambda,    f"{_cfr_lam:.6e}"),
            (self._le_lambdaErr, f"{_cfr_lam_err:.6e}"),
            (self._le_stdAge,    f"{_cfr_fc_ma:.4f}"),
            (self._le_stdAgeErr, f"{_cfr_fc_err:.4f}"),
            (self._le_rStd,      f"{from_fcs_r(_cfr_r_std, self.cfg):.6g}"),
        ]:
            _le.blockSignals(True)
            _le.setText(_txt)
            _le.blockSignals(False)
        self._std_age_err_type = 'external' if _cfr_fc_ext else 'internal'
        self._update_flux_ui()

    def _process_df(self, df, source=''):
        headers  = list(df.columns)
        detected = detect_columns(headers)
        # Pre-read representative J / J-error values to pre-fill the recalc dialog
        def _col_median(col_key):
            if col_key not in detected:
                return None
            ci, vlist = detected[col_key], []
            for _, _row in df.iterrows():
                try:
                    v = float(str(_row.iloc[ci]).strip())
                    if v > 0:
                        vlist.append(v)
                except Exception:
                    pass
            return float(np.median(vlist)) if vlist else None
        _j_default     = _col_median('j_val')
        _j_err_default = _col_median('j_err')
        dlg = ColumnMappingDialog(headers, detected, self, default_sigma=2,
                                  j_default=_j_default, j_err_default=_j_err_default)
        if dlg.exec_() != QDialog.Accepted:
            return
        mapping = dlg.get_mapping()

        required = ['age', 'pct39ar']
        missing  = [COLUMN_LABELS[k] for k in required if k not in mapping]
        if missing:
            QMessageBox.critical(self, "Missing Columns",
                                 "Required columns not mapped:\n  " + '\n  '.join(missing))
            return

        steps = []
        for _, row in df.iterrows():
            try:
                def _get(key, default=0.0):
                    if key not in mapping:
                        return default
                    try:
                        val = row.iloc[mapping[key]]
                        if isinstance(val, str):
                            val = val.strip().rstrip('%').replace(',', '.')
                        return float(val)
                    except (ValueError, TypeError):
                        return default

                s = dict(
                    label   = str(row.iloc[mapping['label']]).strip()
                              if 'label' in mapping else _alpha_label(len(steps)),
                    power   = _get('power', float(len(steps) + 1)),
                    age     = _get('age'),
                    err1s   = _get('err'),
                    pct39Ar = _get('pct39ar'),
                    caK     = _get('cak'),
                    pct40   = _get('pct40ar'),
                    r4039     = _get('r4039'),
                    err_r4039 = _get('err_r4039'),
                    err_cak   = _get('err_cak'),
                    j_val     = _get('j_val'),
                    j_err     = _get('j_err'),
                    # Isochron data (optional)
                    iso_nx  = _get('iso_nx',  None),
                    iso_nsx = _get('iso_nsx', None),
                    iso_ny  = _get('iso_ny',  None),
                    iso_nsy = _get('iso_nsy', None),
                    iso_nri = _get('iso_nri', 0.0),
                    iso_ix  = _get('iso_ix',  None),
                    iso_isx = _get('iso_isx', None),
                    iso_iy  = _get('iso_iy',  None),
                    iso_isy = _get('iso_isy', None),
                    iso_iri = _get('iso_iri', 0.0),
                    status    = 'normal',
                )
                # Replace 0.0 with None for optional isochron fields when not mapped
                for _ik in ('iso_nx','iso_nsx','iso_ny','iso_nsy',
                            'iso_ix','iso_isx','iso_iy','iso_isy'):
                    if _ik not in mapping:
                        s[_ik] = None
                if s['pct39Ar'] > 0 and (s['age'] > 0 or s['r4039'] > 0):
                    steps.append(s)
            except (ValueError, TypeError):
                continue

        if not steps:
            QMessageBox.warning(self, "No Valid Data",
                                "No valid rows found in the imported data.")
            return
        self._finalize_parsed_steps(steps, dlg)
        self.raw_steps = steps
        if source and source != 'clipboard':
            self.title_edit.setText(os.path.splitext(source)[0])
        self._raw_title = (os.path.splitext(source)[0]
                           if source and source != 'clipboard' else None)
        self._update_all()
        QMessageBox.information(self, "Import OK",
                                f"Loaded {len(steps)} steps from {source or 'data'}.")

    def _import_positional(self, df):
        """Headerless paste: show column-mapping dialog with generic column names."""
        df = df.copy()
        df.columns = [f'col {i}' for i in range(len(df.columns))]
        self._process_df(df, 'clipboard')

    def _serialize_disk_flux(self):
        """The loaded disk-flux registry as a JSON-safe dict (arrays → lists)."""
        return serialize_disk_flux(self._disk_flux)

    def _restore_disk_flux(self, data):
        """Add the disks of a serialized registry (keeps existing keys)."""
        merge_disk_flux(self._disk_flux, deserialize_disk_flux(data or {}))

    def save_session(self):
        # ── Carousel mode: save all slots as .ararstack ──────────────────────
        if len(self._carousel) >= 2:
            self._carousel_save_current()   # flush active window state to slot
            path, _ = dfold.get_save_file_name(
                self, f"Save All {len(self._carousel)} Sessions",
                self.title_edit.text().replace(' ', '_') or 'sessions',
                "Ar-Ar Stack (*.ararstack);;All files (*)")
            if not path:
                return
            try:
                sess_list = []
                for slot in self._carousel:
                    # Bake the slot's assigned disk-flux J + σ_J into the saved cfg
                    # so the assignment (and its error) round-trips on reload.
                    cfg_s = self._cfg_with_flux(slot['cfg'], slot.get('flux_hole'))
                    cfg_s['errScale'] = 2
                    fh = slot.get('flux_hole')
                    entry = {
                        'title': slot['title'],
                        'orig_title': slot.get('orig_title') or slot['title'],
                        'cfg':   cfg_s,
                        'steps': _steps_to_disk(slot['raw_steps']),
                        'flux_hole': list(fh) if fh is not None else None,
                    }
                    # The plateau window, as the stack save stores it: without it
                    # the Age distribution tool cannot tell which steps are the plateau.
                    _pl = slot.get('plateau')
                    if _pl is None and slot.get('raw_steps'):
                        # a slot never opened has no plateau yet: detect it the
                        # way opening it would
                        _pl = find_plateau(add_cumulative(_compute_steps_with_age_mode(
                            slot['raw_steps'], cfg_s)), cfg_s)
                    _ps = _plateau_summary(_pl, cfg_s)
                    if _ps is not None:
                        entry['plateau'] = _ps
                    sess_list.append(entry)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'format':        'ararstack_v1',
                        'global_cfg':    self._global_cfg,
                        'stack_method':  self._stack_method,
                        'apply_flux_j':  True,
                        'sessions':      sess_list,
                        # Persist the loaded disk-flux registry so the assignment
                        # metadata (disk layout, holes, cross-hole covariance) and
                        # each slot's hole survive a reload, same as the stack save.
                        'disk_flux':     self._serialize_disk_flux(),
                        'flux_override': {},
                    }, f, indent=2)
                QMessageBox.information(
                    self, "Saved",
                    f"Saved {len(sess_list)} sessions to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))
            return

        # ── Single mode ───────────────────────────────────────────────────────
        if not self.raw_steps:
            QMessageBox.warning(self, "No Data", "No data to save.")
            return
        path, _ = dfold.get_save_file_name(
            self, "Save Session",
            self.title_edit.text().replace(' ', '_') or 'session',
            "Ar-Ar Plateau (*.arar *.ararpt);;All files (*)")
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                # Bake any assigned disk-flux J + σ_J into the saved cfg so the
                # flux-driven age (and its error) round-trips on reload.
                cfg_to_save = self._effective_single_cfg()
                cfg_to_save['errScale'] = 2   # errors stored as 2σ
                _fh = getattr(self, '_sp_flux_hole', None)
                out = {'version': 1,
                       'title': self.title_edit.text(),
                       'orig_title': self._raw_single_title(),
                       'cfg': cfg_to_save,
                       'steps': _steps_to_disk(self.raw_steps)}
                # Persist the flux assignment + registry (optional keys; older
                # readers ignore them) so the hole is editable again on reload.
                if _fh is not None:
                    out['flux_hole'] = list(_fh)
                if self._disk_flux:
                    out['disk_flux'] = self._serialize_disk_flux()
                json.dump(out, f, indent=2)
            QMessageBox.information(self, "Saved", f"Session saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def load_session(self):
        paths, _ = dfold.get_open_file_names(
            self, "Load Session(s)", "",
            "All Ar-Ar files (*.arar *.ararpt *.ararstack);;"
            "Ar-Ar Plateau (*.arar *.ararpt);;"
            "Ar-Ar Stack / Carousel (*.ararstack);;"
            "All files (*)")
        if not paths:
            return

        # .ararstack → load all sessions into carousel (no stack mode needed)
        stack_paths = [p for p in paths if p.lower().endswith('.ararstack')]
        if stack_paths:
            if len(paths) > 1:
                QMessageBox.warning(
                    self, 'Mixed selection',
                    'Select either one .ararstack file or individual .arar/.ararpt files.')
                return
            self._load_ararstack_into_carousel(stack_paths[0])
            return

        # Parse every selected file; collect valid slot dicts
        slots = []
        for path in paths:
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                # See _session_err_div: errScale=2 files store 2σ, converter
                # files declare stepSigma=1, and older files follow the checkbox.
                legacy_1s = (_session_err_div(data.get('cfg', {}),
                                              self._cb_1sigma.isChecked()) == 1.0)
                sess = _load_session_data(
                    data, err_div=(1.0 if legacy_1s else 2.0))
                raw_steps = sess['raw_steps']
                orig_cfg  = {k: sess['cfg'][k] for k in (
                    'lambdaYr', 'lambdaErrYr', 'stdAgeMa', 'stdAgeErrMa',
                    'stdAgeErrType', 'ageFluxMode',
                ) if k in sess['cfg']}
                title = (sess['title']
                         or os.path.splitext(os.path.basename(path))[0])
                # Restore any saved disk-flux registry + hole assignment.
                self._restore_disk_flux(data.get('disk_flux', {}))
                _fh = data.get('flux_hole')
                slots.append({
                    'title':          title,
                    'orig_title':     sess.get('orig_title') or title,
                    'raw_steps':      raw_steps,
                    'cfg':            sess['cfg'],
                    'orig_raw_steps': copy.deepcopy(raw_steps),
                    'orig_calib_cfg': orig_cfg,
                    'plateau':        None,
                    'flux_hole':      tuple(_fh) if _fh else None,
                    '_path':          path,
                })
            except Exception as e:
                QMessageBox.critical(self, "Load Error",
                                     f"{os.path.basename(path)}:\n{e}")

        if not slots:
            return

        # Reset carousel state regardless of how many files were loaded
        self._carousel.clear()
        self._carousel_idx = -1

        if len(slots) == 1:
            # Single file, classic behaviour, carousel stays inactive
            slot = slots[0]
            self.raw_steps         = slot['raw_steps']
            self._orig_raw_steps   = slot['orig_raw_steps']
            self._orig_calib_cfg   = slot['orig_calib_cfg']
            self.cfg.update(slot['cfg'])
            self.title_edit.setText(slot['title'])
            self._raw_title = slot.get('orig_title') or slot['title']
            self._restore_cfg_to_widgets(self.cfg)
            self._update_flux_ui()
            self._sp_flux_hole = slot.get('flux_hole')
            self._sp_auto_match()
            self._refresh_sp_flux_state()
            self._update_all()
            ok, info = _check_rstd_age_consistency(self.raw_steps, self.cfg)
            if not ok:
                QMessageBox.warning(
                    self, "Age Mismatch: ratio_R_std",
                    f"The ages recomputed from the 40Ar/39Ar ratios do not match\n"
                    f"the imported ages stored in this file.\n\n"
                    f"{info}\n\n"
                    f"The ages shown are computed from the ratios with these\n"
                    f"constants. Correct them with Calibration…, or choose\n"
                    f"Imported ages there to use the ages stored in the file."
                )
        else:
            # Multiple files, activate carousel; each aliquot keeps its own cfg/J
            for slot in slots:
                slot.pop('_path', None)
                self._carousel.append(slot)
            self._carousel_load_slot(0)
            self._carousel_refresh_ui()

    # ── Aliquot carousel ─────────────────────────────────────────────────────
    def _restore_cfg_to_widgets(self, cfg):
        """Apply cfg dict to all settings widgets (no recalc, no signals fired)."""
        for sp in (self._sp_minPct, self._sp_minN, self._sp_pCut, self._sp_ageDec):
            sp.blockSignals(True)
        self._sp_minPct.setValue(cfg.get('minPct', 50.0))
        self._sp_minN.setValue(int(cfg.get('minN', 3)))
        self._sp_pCut.setValue(cfg.get('pCut', 5.0))
        self._sp_ageDec.setValue(int(cfg.get('ageDec', 2)))
        for sp in (self._sp_minPct, self._sp_minN, self._sp_pCut, self._sp_ageDec):
            sp.blockSignals(False)
        for le, key, default in [
            (self._le_jVal,      'jVal',        1e-3),
            (self._le_jErrPct,   'jErrPct',     0.0),
            (self._le_rStdErr,   'rStdErrPct',  0.10),
            (self._le_stdAge,    'stdAgeMa',    STD_AGE_DEFAULT),
            (self._le_stdAgeErr, 'stdAgeErrMa', STD_AGE_ERR),
            (self._le_lambda,    'lambdaYr',    LAMBDA_DEFAULT),
            (self._le_lambdaErr, 'lambdaErrYr', LAMBDA_ERR),
        ]:
            le.blockSignals(True)
            v = cfg.get(key, default)
            le.setText(f'{v:.8g}' if isinstance(v, float) else str(v))
            le.blockSignals(False)
        self._le_rStd.blockSignals(True)
        self._le_rStd.setText(f"{from_fcs_r(float(cfg.get('rStd', 70.0) or 70.0), cfg):.11g}")
        self._le_rStd.blockSignals(False)
        for cb, key in [(self._cb_errMode,     'errMode'),
                        (self._cb_ageFluxMode, 'ageFluxMode'),
                        (self._cb_rStdErrMode, 'rStdErrMode')]:
            val = cfg.get(key)
            cb.blockSignals(True)
            for i in range(cb.count()):
                if cb.itemData(i) == val:
                    cb.setCurrentIndex(i); break
            cb.blockSignals(False)
        is_ext = cfg.get('stdAgeErrType', 'internal') == 'external'
        self._std_age_err_type = 'external' if is_ext else 'internal'
        self._sync_ext_mode()
        # zoom: each sample keeps its own Auto / Plateau / Manual and Y range
        zm = cfg.get('zoomMode', 'auto')
        for rb in self._zoom_grp.buttons():
            rb.blockSignals(True)
            rb.setChecked(rb.property('zoomMode') == zm)
            rb.blockSignals(False)
        for sp, k, d in ((self._sp_ymin, 'yMin', 50.0), (self._sp_ymax, 'yMax', 58.0)):
            sp.blockSignals(True)
            sp.setValue(float(cfg.get(k, d)))
            sp.blockSignals(False)
        self._set_zoom_widgets_visible(zm == 'manual')

    def _carousel_slot_from_current(self):
        """Snapshot current window state as a carousel slot dict."""
        return {
            'title':          self.title_edit.text(),
            'orig_title':     self._raw_single_title(),
            'raw_steps':      copy.deepcopy(self.raw_steps),
            'cfg':            self.cfg.copy(),
            'orig_raw_steps': copy.deepcopy(
                getattr(self, '_orig_raw_steps', None) or self.raw_steps),
            'orig_calib_cfg': dict(getattr(self, '_orig_calib_cfg', None) or {}),
            'plateau':        getattr(self, 'plateau', None),
            'flux_hole':      getattr(self, '_sp_flux_hole', None),
        }

    def _carousel_save_current(self):
        """Write current window state back into the active carousel slot."""
        if 0 <= self._carousel_idx < len(self._carousel):
            old = self._carousel[self._carousel_idx]
            new = self._carousel_slot_from_current()
            if 'flux_base_cfg' in old:            # keep the pre-flux snapshot
                new['flux_base_cfg'] = old['flux_base_cfg']
            self._carousel[self._carousel_idx] = new

    # Calibration keys that are per-aliquot and may differ across carousel slots
    _CALIB_KEYS = ('ageFluxMode', 'jVal', 'jErrPct',
                   'rStd', 'rStdErrPct', 'rStdErrMode',
                   'stdAgeMa', 'stdAgeErrMa', 'stdAgeErrType',
                   'lambdaYr', 'lambdaErrYr')

    def _carousel_load_slot(self, idx):
        """Restore window from carousel slot[idx] and recompute."""
        # Capture old calibration for diff display
        old_calib = {k: self.cfg.get(k) for k in self._CALIB_KEYS}

        slot = self._carousel[idx]
        self._carousel_idx = idx
        self.raw_steps         = copy.deepcopy(slot['raw_steps'])
        self._orig_raw_steps   = copy.deepcopy(slot['orig_raw_steps'])
        self._orig_calib_cfg   = dict(slot['orig_calib_cfg'])
        # Keys that describe one aliquot only: drop the previous slot's value
        # when this slot has none, so a J-mode jForce or another aliquot's
        # disk / hole id is not carried over (the hole id feeds the auto-match).
        for k in ('jForce', 'disk_id', 'hole_id', 'irradiation'):
            if k not in slot['cfg']:
                self.cfg.pop(k, None)
        self.cfg.update(slot['cfg'])
        self.title_edit.blockSignals(True)
        self.title_edit.setText(slot['title'])
        self._raw_title = slot.get('orig_title') or slot['title']
        self.title_edit.blockSignals(False)
        self._restore_cfg_to_widgets(self.cfg)
        self._update_flux_ui()
        self._sp_flux_hole = slot.get('flux_hole')
        self._sp_auto_match()          # fills in only if the slot had none stored
        self._refresh_sp_flux_state()
        self._update_all()

        # Show calibration diff in hint label if anything changed
        self._carousel_show_calib_diff(old_calib, self.cfg)

    def _carousel_refresh_ui(self):
        """Sync carousel strip widgets to the current carousel state."""
        n = len(self._carousel)
        # Carousel is single-mode only, never show it while stack mode is active
        active = n >= 2 and not self._stack_mode
        self._carousel_widget.setVisible(active)
        if hasattr(self, '_btn_calib_apply_all'):
            self._btn_calib_apply_all.setVisible(active)
        if hasattr(self, '_btn_flux_apply_all'):
            self._btn_flux_apply_all.setVisible(active)
        if hasattr(self, '_btn_save') and not self._stack_mode:
            self._btn_save.setText(
                f"Save {n} Sessions…" if active else "Save Session…")
        if not active:
            return
        self._carousel_combo.blockSignals(True)
        self._carousel_combo.clear()
        for i, slot in enumerate(self._carousel):
            self._carousel_combo.addItem(slot['title'] or f'Aliquot {i + 1}')
        self._carousel_combo.setCurrentIndex(self._carousel_idx)
        self._carousel_combo.blockSignals(False)
        self._carousel_count_lbl.setText(f'{self._carousel_idx + 1}/{n}')

    def _carousel_show_calib_diff(self, old_cfg, new_cfg):
        """Update the hint label with a one-line diff of changed calibration values."""
        _labels = {
            'ageFluxMode':   'source',
            'jVal':          'J',
            'jErrPct':       'J err%',
            'rStd':          'R_std',
            'rStdErrPct':    'R_std err',
            'stdAgeMa':      'monitor Ma',
            'stdAgeErrMa':   'monitor err',
            'lambdaYr':      'λ',
        }
        parts = []
        for k, lbl in _labels.items():
            ov, nv = old_cfg.get(k), new_cfg.get(k)
            if ov == nv:
                continue
            if isinstance(ov, float) and isinstance(nv, float):
                parts.append(f'{lbl}: {ov:.6g} → {nv:.6g}')
            elif ov != nv:
                parts.append(f'{lbl}: {ov} → {nv}')
        if parts and hasattr(self, '_hint'):
            self._hint.setText('Calibration changed, ' + '  |  '.join(parts))
        elif hasattr(self, '_hint'):
            self._hint.setText(
                'Click step: toggle excluded (red) / normal  ·  Use Pick mode to force plateau steps (blue)')

    def _carousel_recalibrate_others(self, p, old_mode):
        """Calibration… with "Apply to all": recast every other carousel slot
        from its own calibration to the new one, as the visible slot was.

        Only the calibration (λ, FC age and their errors) is shared.  Each
        slot keeps its own J, R standard and errors, and its own age source
        unless the age source was changed in the dialog."""
        new_mode = p.get('age_mode', old_mode)
        ext = 'external' if p['fc2_external'] else 'internal'
        for i, slot in enumerate(self._carousel):
            if i == self._carousel_idx:
                continue
            cfg = slot.setdefault('cfg', {})
            steps = slot.get('raw_steps') or []
            if steps:
                _recalc_ages_in_steps(
                    steps, cfg.get('lambdaYr', LAMBDA_DEFAULT),
                    cfg.get('stdAgeMa', STD_AGE_DEFAULT),
                    _recast_j(self._cfg_with_flux(cfg, slot.get('flux_hole'))),
                    p['lam2'], p['fc2_ma'], fc2_err_ma=p['fc2_err_ma'],
                    ing1=monitor_ingrowth(cfg, cfg.get('lambdaYr', LAMBDA_DEFAULT)),
                    ing2=p.get('ing2'))
                js = [s['j_val'] for s in steps if s.get('j_val', 0.0) > 0]
                if js:
                    cfg['jVal'] = float(np.median(js))
            prod_old = cfg_monitor(cfg)[1]
            cfg.update(lambdaYr=p['lam2'], lambdaErrYr=p['lam2_err'],
                       stdAgeMa=p['fc2_ma'], stdAgeErrMa=p['fc2_err_ma'],
                       stdAgeErrType=ext)
            if p.get('route_keys'):
                set_route(cfg, p['route_keys'])
                if cfg.get('rStd'):
                    cfg['rStd'] = cfg['rStd'] * prod_old / cfg_monitor(cfg)[1]
            if new_mode != old_mode:
                cfg['ageFluxMode'] = new_mode
                if new_mode != 'ratio_J':
                    cfg.pop('jForce', None)
            if steps:
                eff = self._cfg_with_flux(cfg, slot.get('flux_hole'))
                slot['plateau'] = find_plateau(add_cumulative(
                    _compute_steps_with_age_mode(steps, eff)), eff)

    def _carousel_apply_calib_to_all(self, quiet=False):
        """Give every other carousel aliquot the open aliquot's calibration:
        the route, monitor age and λ with their errors.  The flux is not
        touched: each aliquot keeps its own J or R standard, its disk-flux
        hole and its age source, and its ages are recast as the Calibration…
        dialog's "Apply to all" does."""
        n = len(self._carousel)
        if n < 2:
            return
        self._carousel_save_current()
        c = self.cfg
        lam = c.get('lambdaYr', LAMBDA_DEFAULT)
        p = dict(lam2=lam, lam2_err=c.get('lambdaErrYr', 0.0),
                 fc2_ma=c.get('stdAgeMa', STD_AGE_DEFAULT),
                 fc2_err_ma=c.get('stdAgeErrMa', 0.0),
                 fc2_external=c.get('stdAgeErrType') == 'external',
                 route_keys={k: c[k] for k in ROUTE_KEYS if k in c},
                 ing2=monitor_ingrowth(c, lam))
        self._carousel_recalibrate_others(p, c.get('ageFluxMode'))
        if quiet:
            return
        QMessageBox.information(
            self, 'Applied',
            f'Calibration applied to all {n} aliquots. Each keeps its own flux.')

    def _slot_flux_auto(self, slot):
        """Resolve a carousel slot to (irr, disk, hole) by identity, or None.
        Uses the same disk_id / hole_id / irradiation auto-match as the single
        plateau and the Age distribution tool."""
        cfg = slot.get('cfg', {})
        m = self._flux_auto_match(cfg.get('disk_id'), cfg.get('hole_id'),
                                  cfg.get('irradiation'))
        if m is None:
            return None
        (irr, disk), idx = m
        fx = self._disk_flux.get((str(irr), str(disk)))
        if fx is None:
            return None
        hole = next((h for h, ix in fx['idx'].items() if ix == idx), None)
        return (irr, disk, hole) if hole is not None else None

    # the typed flux of an aliquot: how J is given, its value and error
    # (the R standard is copied as the measured monitor ratio; each aliquot
    # keeps its own calibration and route)
    _TYPED_FLUX_KEYS = ('ageFluxMode', 'jVal', 'jErrPct', 'jForce',
                        'rStd', 'rStdErrPct', 'rStdErrMode')

    def _slot_refresh_plateau(self, slot):
        """Recompute a slot's plateau after its cfg changed, so a slot that is
        not opened again is still saved and stacked with its new ages."""
        steps = slot.get('raw_steps')
        if steps:
            eff = self._cfg_with_flux(slot['cfg'], slot.get('flux_hole'))
            slot['plateau'] = find_plateau(add_cumulative(
                _compute_steps_with_age_mode(steps, eff)), eff)

    def _carousel_apply_flux_to_all(self):
        """Give every carousel aliquot the open aliquot's flux.

        If the open aliquot takes its J from a hole of the loaded disk flux,
        every aliquot is matched to its own hole by identity (disk_id /
        hole_id), as the Age distribution tool's 'Auto-match all' does.  If it
        uses a typed J or R standard, that value and its error are copied to
        every aliquot, and any disk-flux assignment of theirs is removed."""
        n = len(self._carousel)
        if n < 2:
            return
        if getattr(self, '_sp_flux_hole', None) is None:
            self._carousel_save_current()
            flux = {k: self.cfg[k] for k in self._TYPED_FLUX_KEYS if k in self.cfg}
            prod = cfg_monitor(self.cfg)[1]
            for i, slot in enumerate(self._carousel):
                if i == self._carousel_idx:
                    continue
                self._bake_flux_into_cfg(slot['cfg'], None, slot)  # undo a disk-flux bake
                slot['flux_hole'] = None
                for k in self._TYPED_FLUX_KEYS:
                    if k in flux:
                        slot['cfg'][k] = flux[k]
                    else:
                        slot['cfg'].pop(k, None)
                if slot['cfg'].get('rStd'):
                    slot['cfg']['rStd'] = flux['rStd'] * prod / cfg_monitor(slot['cfg'])[1]
                self._slot_refresh_plateau(slot)
            self._carousel_load_slot(self._carousel_idx)
            self._carousel_refresh_ui()
            if self.cfg.get('ageFluxMode') == 'imported':
                what = 'the imported J of each aliquot'
            elif self.cfg.get('ageFluxMode') == 'ratio_J':
                what = f"J = {self.cfg.get('jVal', 0):.8g} ± {self.cfg.get('jErrPct', 0):.4g} %"
            else:
                unit = ' %' if self.cfg.get('rStdErrMode', 'pct') == 'pct' else ''
                what = (f"R standard = {self.cfg.get('rStd', 0):.11g} "
                        f"± {self.cfg.get('rStdErrPct', 0):.4g}{unit} (1σ)")
            QMessageBox.information(self, 'Applied',
                                    f'Typed flux applied to all {n} aliquots: {what}.')
            return
        if not self._disk_flux:
            QMessageBox.warning(
                self, 'No disk flux',
                'Load a disk flux first (the single-plateau "Disk flux…" button, '
                '"Load .jflux…").')
            return
        # Flush live state into the active slot before reassigning them all.
        self._carousel_save_current()
        n_ok = n_miss = 0
        misses = []
        for slot in self._carousel:
            fh = self._slot_flux_auto(slot)
            slot['flux_hole'] = fh
            if fh is not None:
                n_ok += 1
            else:
                n_miss += 1
                cfg = slot.get('cfg', {})
                misses.append(slot.get('title')
                              or f"{cfg.get('disk_id') or '?'}/{cfg.get('hole_id') or '?'}")
        # Reload the current slot so the view + widgets reflect its assignment.
        self._carousel_load_slot(self._carousel_idx)
        self._carousel_refresh_ui()
        msg = f'Disk flux assigned to {n_ok} of {n} aliquots by identity.'
        if n_miss:
            msg += (f'\n\n{n_miss} aliquot(s) had no matching hole on the loaded '
                    'disk (kept their typed J):\n  • ' + '\n  • '.join(misses))
        QMessageBox.information(self, 'Applied', msg)

    def _carousel_sync_title(self, text):
        """Keep the active carousel slot's title in sync with the title bar."""
        if 0 <= self._carousel_idx < len(self._carousel):
            self._carousel[self._carousel_idx]['title'] = text
            self._carousel_combo.blockSignals(True)
            self._carousel_combo.setItemText(
                self._carousel_idx,
                text or f'Aliquot {self._carousel_idx + 1}')
            self._carousel_combo.blockSignals(False)

    def _carousel_go(self, delta):
        """Navigate ±1 through the carousel, wrapping around."""
        n = len(self._carousel)
        if n < 2:
            return
        self._carousel_save_current()
        self._carousel_load_slot((self._carousel_idx + delta) % n)
        self._carousel_refresh_ui()

    def _carousel_on_combo(self, combo_idx):
        """Called when the user picks an entry from the carousel dropdown."""
        if combo_idx < 0 or combo_idx == self._carousel_idx:
            return
        self._carousel_save_current()
        self._carousel_load_slot(combo_idx)
        self._carousel_refresh_ui()

    def _carousel_open_add(self):
        """Load one or more .arar/.ararpt files and append them to the carousel."""
        paths, _ = dfold.get_open_file_names(
            self, "Add Aliquot(s)", "",
            "Ar-Ar Plateau (*.arar *.ararpt);;All files (*)")
        if not paths:
            return
        # First call: push current state as slot 0 so it's preserved
        if self._carousel_idx < 0:
            self._carousel.append(self._carousel_slot_from_current())
            self._carousel_idx = 0
        else:
            self._carousel_save_current()
        for path in paths:
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                legacy_1s = (_session_err_div(data.get('cfg', {}),
                                              self._cb_1sigma.isChecked()) == 1.0)
                sess = _load_session_data(
                    data, err_div=(1.0 if legacy_1s else 2.0))
                raw_steps = sess['raw_steps']
                orig_cfg  = {k: sess['cfg'][k] for k in (
                    'lambdaYr', 'lambdaErrYr', 'stdAgeMa', 'stdAgeErrMa',
                    'stdAgeErrType', 'ageFluxMode',
                ) if k in sess['cfg']}
                title = (sess['title']
                         or os.path.splitext(os.path.basename(path))[0])
                # the file's disk-flux registry and hole, as Load does
                self._restore_disk_flux(data.get('disk_flux', {}))
                _fh = data.get('flux_hole')
                self._carousel.append({
                    'title':          title,
                    'orig_title':     sess.get('orig_title') or title,
                    'raw_steps':      raw_steps,
                    'cfg':            sess['cfg'],
                    'orig_raw_steps': copy.deepcopy(raw_steps),
                    'orig_calib_cfg': orig_cfg,
                    'plateau':        None,
                    'flux_hole':      tuple(_fh) if _fh else None,
                })
            except Exception as e:
                QMessageBox.critical(self, "Load Error",
                                     f"{os.path.basename(path)}:\n{e}")
        if len(self._carousel) >= 2:
            self._carousel_load_slot(len(self._carousel) - 1)
            self._carousel_refresh_ui()

    def _carousel_remove_current(self):
        """Remove the active slot; deactivate strip when only one aliquot remains."""
        n = len(self._carousel)
        if n == 0 or self._carousel_idx < 0:
            return
        idx = self._carousel_idx
        self._carousel.pop(idx)
        n -= 1
        if n == 0:
            self._carousel_idx = -1
            self._carousel_refresh_ui()
            return
        if n == 1:
            # Only one slot left, deactivate carousel, keep the data
            self._carousel_idx = -1
            slot = self._carousel[0]
            self._carousel.clear()
            self.raw_steps         = copy.deepcopy(slot['raw_steps'])
            self._orig_raw_steps   = copy.deepcopy(slot['orig_raw_steps'])
            self._orig_calib_cfg   = dict(slot['orig_calib_cfg'])
            self.cfg.update(slot['cfg'])
            self.title_edit.blockSignals(True)
            self.title_edit.setText(slot['title'])
            self._raw_title = slot.get('orig_title') or slot['title']
            self.title_edit.blockSignals(False)
            self._restore_cfg_to_widgets(self.cfg)
            self._update_flux_ui()
            self._sp_flux_hole = None
            self._sp_auto_match()
            self._refresh_sp_flux_state()
            self._update_all()
            self._carousel_refresh_ui()
        else:
            self._carousel_load_slot(min(idx, n - 1))
            self._carousel_refresh_ui()

    def _load_ararstack_into_carousel(self, path):
        """Load all sessions from an .ararstack file into the single-mode carousel."""
        try:
            with open(path, encoding='utf-8') as f:
                state = json.load(f)
            if state.get('format') != 'ararstack_v1':
                QMessageBox.warning(self, 'Unknown Format',
                                    'Not a valid stack/carousel file.')
                return
            sessions_data = state.get('sessions', [])
            if not sessions_data:
                QMessageBox.warning(self, 'Empty File',
                                    'No sessions found in the file.')
                return

            # Restore the disk-flux registry so assignments are fully editable again.
            self._restore_disk_flux(state.get('disk_flux', {}))

            slots = []
            for i, sd in enumerate(sessions_data):
                _sc = sd.get('cfg', {})
                legacy_1s = (_session_err_div(_sc, self._cb_1sigma.isChecked())
                             == 1.0)
                sess = _load_session_data(
                    {'title': sd.get('title', ''),
                     'cfg':   _sc,
                     'steps': sd.get('steps', [])},
                    err_div=(1.0 if legacy_1s else 2.0))
                orig_cfg = {k: sess['cfg'][k] for k in (
                    'lambdaYr', 'lambdaErrYr', 'stdAgeMa', 'stdAgeErrMa',
                    'stdAgeErrType', 'ageFluxMode',
                ) if k in sess['cfg']}
                _fh = sd.get('flux_hole')
                slots.append({
                    'title':          sd.get('title', '') or f'Session {i + 1}',
                    'orig_title':     (sd.get('orig_title') or sd.get('title', '')
                                       or f'Session {i + 1}'),
                    'raw_steps':      sess['raw_steps'],
                    'cfg':            sess['cfg'],
                    'orig_raw_steps': copy.deepcopy(sess['raw_steps']),
                    'orig_calib_cfg': orig_cfg,
                    'plateau':        None,
                    'flux_hole':      tuple(_fh) if _fh else None,
                })

            self._carousel.clear()
            self._carousel_idx = -1

            if len(slots) == 1:
                # Single session, load without carousel
                slot = slots[0]
                self.raw_steps         = slot['raw_steps']
                self._orig_raw_steps   = slot['orig_raw_steps']
                self._orig_calib_cfg   = slot['orig_calib_cfg']
                self.cfg.update(slot['cfg'])
                self.title_edit.setText(slot['title'])
                self._raw_title = slot.get('orig_title') or slot['title']
                self._restore_cfg_to_widgets(self.cfg)
                self._update_flux_ui()
                self._sp_flux_hole = slot.get('flux_hole')
                self._sp_auto_match()
                self._refresh_sp_flux_state()
                self._update_all()
            else:
                for slot in slots:
                    self._carousel.append(slot)
                self._carousel_load_slot(0)
                self._carousel_refresh_ui()

        except Exception as e:
            QMessageBox.critical(self, 'Load Error', str(e))

    # ── Data / selection helpers ──────────────────────────────────────────────
    def _reset_selections(self):
        """Clear manual step selections on every loaded plateau, then re-run
        auto detection.

        Steps live in three places depending on the mode: the live window
        (self.raw_steps), the carousel slots and the stack sessions; all are
        reset, not only the aliquot shown.  'forced' is cleared too: a manual
        plateau is a selection, and find_plateau uses the forced steps instead
        of searching while any are set.
        """
        holders = []
        if self.raw_steps:
            holders.append(self.raw_steps)
        for slot in self._carousel:
            rs = slot.get('raw_steps')
            if rs and rs is not self.raw_steps:
                holders.append(rs)
        for sess in self._sessions:
            rs = sess.get('raw_steps')
            if rs and rs is not self.raw_steps:
                holders.append(rs)
        if not holders:
            return

        n_cleared = 0
        for steps in holders:
            for st in steps:
                if st.get('status', 'normal') != 'normal':
                    n_cleared += 1
                st['status'] = 'normal'

        # Recompute the carousel slots that are not the visible one, so their
        # cached plateau is not left describing the pre-reset selection.
        for i, slot in enumerate(self._carousel):
            if i == self._carousel_idx or not slot.get('raw_steps'):
                continue
            eff = self._cfg_with_flux(slot.get('cfg', {}), slot.get('flux_hole'))
            steps_c = add_cumulative(
                _compute_steps_with_age_mode(slot['raw_steps'], eff))
            slot['plateau'] = find_plateau(steps_c, eff)

        if self._stack_mode:
            for sess in self._sessions:
                self._stack_recompute(sess)
            self._stack_refresh_table()
            self._queue_redraw()
        elif self.raw_steps:
            self._update_all()
        self._carousel_refresh_ui()

        if hasattr(self, '_hint'):
            self._hint.setText(
                f"Reset {n_cleared} manual step selection(s) across "
                f"{len(holders)} plateau(s); auto detection re-run.")

    # ── Stack / isochron mode toggles ────────────────────────────────────────
    def _toggle_isochron_mode(self, enabled):
        self._isochron_mode = enabled
        self._iso_toggle_widget.setVisible(enabled)
        # The zoom row sets the age-spectrum axes; the isochron has its own
        # X / Y fields in the isochron row.
        for w in self._spectrum_zoom_widgets:
            w.setVisible(not enabled)
        self._set_zoom_widgets_visible(self.cfg.get('zoomMode') == 'manual')
        if enabled:
            # Reset to inverse (default) whenever isochron mode is opened
            self.isochron_canvas._showing_inverse = True
            self.isochron_canvas._iso_zoom = 'data'
            self._btn_iso_panel.setText("Switch to Normal Isochron")
            self._btn_iso_zoom.setText("Show Intercept")
            self._iso_lims_to_fields()
        # Isochron coexists with stack or single mode, uses whichever is active.
        self._canvas_stack.setCurrentIndex(
            2 if enabled else (1 if self._stack_mode else 0))
        self._queue_redraw()

    def _toggle_iso_panel(self):
        """Flip between normal and inverse isochron panels."""
        inv = not self.isochron_canvas._showing_inverse
        self.isochron_canvas._showing_inverse = inv
        self._btn_iso_panel.setText(
            "Switch to Normal Isochron" if inv else "Switch to Inverse Isochron")
        self._iso_lims_to_fields()
        self._do_redraw_isochron()

    def _toggle_iso_zoom(self):
        """Toggle between data-focused and intercept-showing zoom."""
        show_int = self.isochron_canvas._iso_zoom == 'data'
        self.isochron_canvas._iso_zoom = 'intercept' if show_int else 'data'
        self._btn_iso_zoom.setText("Zoom to Data" if show_int else "Show Intercept")
        self._do_redraw_isochron()

    @property
    def _disk_flux(self):
        return self._disk_flux_stack if self._stack_mode else self._disk_flux_single

    @_disk_flux.setter
    def _disk_flux(self, value):
        if self._stack_mode:
            self._disk_flux_stack = value
        else:
            self._disk_flux_single = value

    def _sync_flux_state_to_mode(self):
        """After a mode switch: the flux labels show the registry of the mode
        that is now on, not the one that was just left."""
        lbl = getattr(self, '_lbl_flux_status', None)
        if lbl is not None and self._stack_mode:
            if self._disk_flux:
                self._refresh_flux_status()
            else:
                lbl.setText("No disk flux loaded, GLS + ML falls back "
                            "to binary disk grouping.")
                lbl.setStyleSheet("color:#888; font-size:9px; font-style:italic;")
                lbl.setToolTip('')
        self._refresh_sp_flux_state()

    def _toggle_stack_mode(self, enabled):
        self._stack_mode = enabled
        # Isochron mode stays active; it will now draw from stack or single as appropriate.
        idx = 1 if enabled else 0
        self._canvas_stack.setCurrentIndex(
            2 if self._isochron_mode else idx)
        self._mode_stack.setCurrentIndex(idx)

        # Sidebar button labels
        self._btn_save.setText("Save Stack…"    if enabled else "Save Session…")
        self._btn_load.setText("Load Stack…"    if enabled else "Load Session(s)…")
        self._btn_paste.setText("Paste+ Add Aliquot" if enabled else "Paste Clipboard")
        self._btn_import.setText("Load .ararpt file(s)…" if enabled else "Import File…")
        # Reset works on every loaded plateau, in stack mode as in single mode
        self._btn_reset.setVisible(True)

        # "Plateau" zoom is nonsensical for multiple spectra
        if self._rb_plateau is not None:
            self._rb_plateau.setVisible(not enabled)
            if enabled and self._rb_plateau.isChecked():
                self._zoom_grp.buttons()[0].setChecked(True)
                self._on_zoom_btn(self._zoom_grp.buttons()[0])

        # Settings tab: hide single-aliquot flux section, show stack flux groups
        for w in (self._lbl_flux_hd,
                  self._lbl_sp_flux, self._sp_flux_w, self._lbl_sp_flux_state,
                  self._lbl_calib_state,
                  self._lbl_route_hdr, self._lbl_route, self._flux_box):
            w.setVisible(not enabled)
        self._stack_flux_section.setVisible(enabled)
        if enabled:
            self._rebuild_flux_ui()
        self._sync_flux_state_to_mode()
        # Keep error-method greyed out in sync with the current combining method
        self._on_stack_method()

        # Hide the carousel strip when entering stack mode, the carousel is
        # single-mode only, and leaving it visible while _stack_mode=True causes
        # combo switches to call _update_all() / _queue_redraw() in a mixed state.
        if hasattr(self, '_carousel_widget'):
            self._carousel_widget.setVisible(
                (not enabled) and len(self._carousel) >= 2)
        if hasattr(self, '_btn_calib_apply_all'):
            self._btn_calib_apply_all.setVisible(
                (not enabled) and len(self._carousel) >= 2)

        self._queue_redraw()

    # ── Mode-dispatch helpers ─────────────────────────────────────────────────
    def _paste_dispatch(self):
        if self._stack_mode:
            self._stack_paste_add()
        else:
            self.paste_data()

    def _import_dispatch(self):
        if self._stack_mode:
            self._stack_load_files()
        else:
            self.import_file()

    def _save_dispatch(self):
        if self._stack_mode:
            self._stack_save()
        else:
            self.save_session()

    def _load_dispatch(self):
        if self._stack_mode:
            self._stack_load()
        else:
            self.load_session()

    def _reset_dispatch(self):
        self._reset_selections()

    # ── Export ────────────────────────────────────────────────────────────────
    # Panels and switches offered by the export dialog.  Declared here, drawn
    # by _export_render below; export.py knows nothing about them.
    def _export_panels(self):
        return [
            export.Panel('p40', '% \u2074\u2070Ar*',      on=True,  weight=1.0),
            export.Panel('cak', 'Ca/K  (or K/Ca)',          on=True,  weight=1.0),
            export.Panel('age', 'Age spectrum',             on=True,  weight=2.8),
        ] + ([export.Panel('iso', 'Isochrons (under the spectrum)', on=True,
                           weight=2.6)] if self._stack_mode else [])

    @staticmethod
    def _layout_gaps(opts):
        """Spacing between the panels, from the Layout options."""
        def f(k, d):
            try:
                return float(opts.get(k, d))
            except (TypeError, ValueError):
                return d
        pl = opts.get('iso_place', 'below')
        return {'iso': f('gap_iso', 0.40), 'spec': f('gap_spec', 0.06),
                'place': pl if pl in ('below', 'above', 'right') else 'below',
                'iso_width': f('iso_width', 0.85), 'side': f('gap_side', 0.40)}

    @staticmethod
    def _layout_options():
        L = 'Layout (spacing)'
        return [
            export.Option('iso_place', 'Isochrons are', 'choice', group=L,
                          values=[('below', 'Below the spectrum'),
                                  ('above', 'Above the spectrum'),
                                  ('right', 'To the right (stacked, tall)')],
                          tip='To the right puts the inverse isochron over the '
                              'normal one in a column beside the spectrum; use '
                              'a wide page for it.'),
            export.Option('iso_width', 'Right column width', 'float', default=0.85,
                          lo=0.3, hi=2.0, step=0.05, group=L,
                          depends=('iso_place', 'right'),
                          tip='Width of the isochron column relative to the '
                              'spectrum.'),
            export.Option('gap_side', 'Space beside the spectrum', 'float',
                          default=0.40, lo=0.1, hi=1.0, step=0.05, group=L,
                          depends=('iso_place', 'right')),
            export.Option('gap_spec', 'Between spectrum panels', 'float', default=0.06,
                          lo=0.0, hi=0.5, step=0.02, group=L,
                          tip='Space between the % 40Ar*, Ca/K and age panels, '
                              'as a fraction of a panel height.'),
            export.Option('gap_iso', 'Above the isochrons', 'float', default=0.40,
                          lo=0.1, hi=1.5, step=0.05, group=L,
                          tip='Space between the spectrum (with its axis title) '
                              'and the isochrons.'),
            export.Option('iso_pair_gap', 'Between the two isochrons', 'float',
                          default=0.18, lo=0.05, hi=0.5, step=0.01, group=L,
                          tip='Space between the inverse and the normal '
                              'isochron, as a fraction of their combined width.'),
        ]

    @staticmethod
    def _label_on_top(spec, opts):
        """Big text, or any enlarged text (Text x of 1.25 or more): the plateau
        / combined age label goes to the top of the panel, the line under it,
        so it never sits on top of the steps."""
        try:
            return (opts.get('text_style') == 'big'
                    or float(spec.get('font_scale', 1.0) or 1.0) >= 1.25)
        except Exception:
            return False

    def _export_options(self):
        """Switches offered by the export dialog, grouped into boxes."""
        yr = self._compute_age_range()
        opts = [
            export.Option('cak_mode', 'Ratio panel', 'choice',
                          default='kca' if self._kca_view() else 'cak',
                          values=[('cak', 'Ca/K'), ('kca', 'K/Ca')],
                          tip='K/Ca plots 1/(Ca/K); the error box is the image of '
                              'the Ca/K error, and steps with Ca/K ≤ 0 are left out.'),
            export.Option('show_title', 'Title', 'bool', True),
            export.Option('show_err_box', 'Error box', 'bool', True,
                          tip='The analytical / external note in the corner.'),
        ]
        if self._stack_mode:
            opts += [
                export.Option('show_bracket', 'Combined age label', 'bool', True),
                export.Option('bracket_at', 'Line at (Ma)', 'text', placeholder='auto',
                              depends=('show_bracket', True),
                              tip='Age (Ma) the plateau line sits at; type a lower value to bring the line and its label down. Blank = automatic, just above the tallest plateau step. The label always stays inside the panel.'),
                export.Option('show_legend', 'Aliquot legend', 'bool', True),
                export.Option('legend_ages', 'Ages in legend', 'choice',
                              values=[('off', 'Names only'),
                                      ('all', 'Plateau age ± analytical / internal / external'),
                                      ('int', 'Plateau age ± internal'),
                                      ('ana', 'Plateau age ± analytical'),
                                      ('ext', 'Plateau age ± external')],
                              depends=('show_legend', True),
                              tip='Each aliquot\'s plateau age next to its name, '
                                  'at the 1σ/2σ level and decimals shown in the '
                                  'window. Internal = analytical + J.'),
            ]
        else:
            opts += [
                export.Option('show_bracket', 'Plateau bracket', 'bool', True),
                export.Option('bracket_at', 'Line at (Ma)', 'text', placeholder='auto',
                              depends=('show_bracket', True),
                              tip='Age (Ma) the plateau line sits at; type a lower value to bring the line and its label down. Blank = automatic, just above the tallest plateau step. The label always stays inside the panel.'),
                export.Option('show_step_labels', 'Step labels', 'bool', True),
                export.Option('show_int_age', 'Total-gas age', 'bool', True),
            ]
        # names: the figure title, plus one field per aliquot in stack mode
        # Names: the main window's, the experiment names as loaded (before any
        # renaming), or typed.  The fields are always editable and show the
        # chosen names; pick 'Typed below' to put edits in the figure.
        opts.append(export.Option(
            'names_from', 'Names in the figure', 'choice', group='Names', wide=True,
            values=[('window', 'Main window'),
                    ('raw', 'Experiment names (as loaded, before renaming)'),
                    ('typed', 'Typed below')],
            tip=('Main window: the names as they are now'
                 + (' in the stack table' if self._stack_mode else '')
                 + '. Experiment names: the names the files came with, even '
                   'if renamed since. The fields below show the chosen names; '
                   'edit them, then pick "Typed below" to use your edits.')))
        if self._stack_mode:
            opts.append(export.Option(
                'title', 'Title', 'text', group='Names', wide=True,
                default='Stacked aliquots', placeholder='figure title'))
            # excluded (×) aliquots are not drawn, so they get no field
            for k, sess in enumerate(self._sessions):
                if sess.get('excluded', False):
                    continue
                cur = sess.get('title', f'Aliquot {k + 1}')
                opts.append(export.Option(
                    f'name_{k}', f'Aliquot {k + 1}', 'text', group='Names', wide=True,
                    default=cur,
                    follows=('names_from', {'window': cur,
                                            'raw': sess.get('orig_title') or cur}),
                    ident=sess.get('orig_title') or cur))
        else:
            cur = self.title_edit.text()
            opts.append(export.Option(
                'title', 'Title', 'text', group='Names', wide=True, default=cur,
                placeholder='figure title',
                follows=('names_from', {'window': cur,
                                        'raw': self._raw_single_title()})))
        # axis ranges: blank means "as on screen"
        opts += [
            export.Option('x_min', 'X from', 'text', group='Axis ranges',
                          default='0', placeholder='auto'),
            export.Option('x_max', 'to', 'text', group='Axis ranges',
                          default='100', placeholder='auto'),
            export.Option('y_min', 'Age from', 'text', group='Axis ranges',
                          default=f'{yr[0]:.3f}', placeholder='auto'),
            export.Option('y_max', 'to', 'text', group='Axis ranges',
                          default=f'{yr[1]:.3f}', placeholder='auto'),
            export.Option('cak_min', 'Ratio from', 'text', group='Axis ranges',
                          placeholder='auto'),
            export.Option('cak_max', 'to', 'text', group='Axis ranges',
                          placeholder='auto'),
            export.Option('p40_min', '%\u2074\u2070Ar* from', 'text',
                          group='Axis ranges', placeholder='auto'),
            export.Option('p40_max', 'to', 'text', group='Axis ranges',
                          default=(f'{_P40_MAX:g}' if _P40_MAX else ''),
                          placeholder='auto',
                          tip='Steps above this are cut off at the top of the '
                              'panel. Blank: 0-100 %, higher when a step is '
                              'above 100 %.'),
        ]
        # page 1: the isochron as a small window inside the age panel
        I1 = 'Isochron window in the age panel'
        opts += [
            export.Option('iso_inset', 'Show isochron window', 'bool', False,
                          group=I1,
                          tip='A small isochron inside the age spectrum. '
                              'Independent of page 2.'),
            export.Option('iso_inset_type', 'Diagram', 'choice', group=I1,
                          values=[('inverse', 'Inverse'), ('normal', 'Normal')],
                          depends=('iso_inset', True)),
            export.Option('iso_x', 'x (%)', 'float', default=58.0,
                          lo=0.0, hi=100.0, step=1.0, group=I1,
                          depends=('iso_inset', True)),
            export.Option('iso_y', 'y (%)', 'float', default=6.0,
                          lo=0.0, hi=100.0, step=1.0, group=I1,
                          depends=('iso_inset', True)),
            export.Option('iso_w', 'Width (%)', 'float', default=38.0,
                          lo=10.0, hi=95.0, step=1.0, group=I1,
                          depends=('iso_inset', True)),
            export.Option('iso_h', 'Height (%)', 'float', default=38.0,
                          lo=10.0, hi=95.0, step=1.0, group=I1,
                          depends=('iso_inset', True)),
            export.Option('iso_inset_intercept', 'Intercept marker', 'bool', True,
                          group=I1, depends=('iso_inset', True),
                          tip='The fitted y-intercept (trapped \u2074\u2070Ar/'
                              '\u00b3\u2076Ar) with its error bar, drawn on the '
                              'y-axis.'),
            export.Option('iso_inset_init', '\u2074\u2070Ar/\u00b3\u2076Ar\u2080 label',
                          'bool', False, group=I1, depends=('iso_inset', True)),
            export.Option('iso_inset_atm', 'Atmospheric reference', 'bool', False,
                          group=I1, depends=('iso_inset', True), wide=True),
            export.Option('iso_ins_x0', 'x from', 'text', group=I1,
                          placeholder='auto', depends=('iso_inset', True),
                          tip='Axis ranges of the window only; blank = automatic '
                              '(always reaching the y-intercept).'),
            export.Option('iso_ins_x1', 'to', 'text', group=I1,
                          placeholder='auto', depends=('iso_inset', True)),
            export.Option('iso_ins_y0', 'y from', 'text', group=I1,
                          placeholder='auto', depends=('iso_inset', True)),
            export.Option('iso_ins_y1', 'to', 'text', group=I1,
                          placeholder='auto', depends=('iso_inset', True)),
        ]
        opts += self._layout_options()
        # the isochron controls sit on the page-2 tab in both modes; in stack
        # mode the panel under the spectrum reads the same settings
        iso = self._isochron_page_options(page=1)
        if self._stack_mode:
            k = next((i for i, o in enumerate(iso) if o.id == 'iso_heading_text'),
                     len(iso) - 1) + 1
            iso.insert(k, export.Option(
                'iso_legend', 'Aliquot legend on this page', 'bool', False,
                group='Isochron', page=1,
                tip='Off: the legend on page 1 serves, the colours are the '
                    'same. Tick it for an isochron used on its own.'))
        opts += iso
        return opts

    def _isochron_page_options(self, page=1):
        """Controls for the isochron's own page (page 2 of the export, or
        the only page when the window is in isochron mode)."""
        G, GI, GN = 'Isochron', 'Inverse isochron axes', 'Normal isochron axes'
        o = [
            export.Option('iso_type', 'Diagram', 'choice', group=G, page=page,
                          values=[('inverse', 'Inverse  (\u00b3\u2076Ar/\u2074\u2070Ar vs \u00b3\u2079Ar/\u2074\u2070Ar)'),
                                  ('normal',  'Normal  (\u2074\u2070Ar/\u00b3\u2076Ar vs \u00b3\u2079Ar/\u00b3\u2076Ar)'),
                                  ('both',    'Both, side by side')]),
            export.Option('iso_heading', 'Heading', 'choice', group=G, page=page,
                          values=[('sample', 'Sample name'),
                                  ('type', 'Diagram name (Inverse / Normal isochron)'),
                                  ('custom', 'Custom text'),
                                  ('none', 'None')],
                          tip='With both diagrams, each keeps its diagram name '
                              'and the sample name / custom text goes above them.'),
            export.Option('iso_heading_text', 'Heading text', 'text', group=G,
                          page=page, wide=True, depends=('iso_heading', 'custom')),
            export.Option('iso_stats', 'Stats box', 'bool', True, group=G, page=page),
            export.Option('iso_stats_pos', 'Stats corner', 'choice', group=G,
                          page=page, depends=('iso_stats', True),
                          values=[('auto', 'Automatic'), ('ul', 'Upper left'),
                                  ('ur', 'Upper right'), ('ll', 'Lower left'),
                                  ('lr', 'Lower right')]),
            export.Option('iso_err_tier', 'Age error shown', 'choice', group=G,
                          page=page, depends=('iso_stats', True),
                          values=[('int', 'Internal (analytical + J)'),
                                  ('ana', 'Analytical only'),
                                  ('ext', 'External (+ calibration)')]),
            export.Option('iso_err_line', 'Analytical / internal / external line',
                          'bool', True, group=G, page=page,
                          depends=('iso_stats', True),
                          tip='Untick to show only the chosen error in the age '
                              'line, tagged int., ana. or ext.'),
            export.Option('iso_labels', 'Step labels', 'bool', True, group=G,
                          page=page),
            export.Option('iso_intercept', 'Intercept marker', 'bool', True,
                          group=G, page=page,
                          tip='The fitted y-intercept with its error bar: the '
                              'initial \u2074\u2070Ar/\u00b3\u2076Ar of the sample.'),
            export.Option('iso_init', '\u2074\u2070Ar/\u00b3\u2076Ar\u2080 label',
                          'bool', False, group=G, page=page,
                          tip='The trapped \u2074\u2070Ar/\u00b3\u2076Ar from the '
                              'y-intercept as a one-line label; works with the '
                              'statistics box off.'),
            export.Option('iso_atm', 'Atmospheric reference', 'bool', False,
                          group=G, page=page,
                          tip='Dashed line at the atmospheric \u2074\u2070Ar/'
                              '\u00b3\u2076Ar of Lee et al. (2006), 298.56; an '
                              'intercept on it means no excess argon.'),
            export.Option('iso_spread', 'Spread of the points', 'bool', False,
                          group=G, page=page,
                          tip='How much of the atmospheric-to-radiogenic range '
                              'the steps cover. A wide spread pins the trapped '
                              'ratio; a narrow one makes the intercept an '
                              'extrapolation.'),
        ]
        for grp, pre, xl, yl in ((GI, 'iso_inv', '\u00b3\u2079Ar/\u2074\u2070Ar', '\u00b3\u2076Ar/\u2074\u2070Ar'),
                                 (GN, 'iso_nor', '\u00b3\u2079Ar/\u00b3\u2076Ar', '\u2074\u2070Ar/\u00b3\u2076Ar')):
            o += [
                export.Option(f'{pre}_x0', f'{xl} from', 'text', group=grp,
                              page=page, placeholder='auto',
                              tip='Blank = automatic (always reaching the '
                                  'y-intercept). Applies to this diagram on '
                                  'its own page; the window in the age panel '
                                  'stays automatic.'),
                export.Option(f'{pre}_x1', 'to', 'text', group=grp, page=page,
                              placeholder='auto'),
                export.Option(f'{pre}_y0', f'{yl} from', 'text', group=grp,
                              page=page, placeholder='auto'),
                export.Option(f'{pre}_y1', 'to', 'text', group=grp, page=page,
                              placeholder='auto'),
            ]
        if page == 1:
            o += [
                export.Option('iso_page', 'As page 2 of the PDF', 'bool', False,
                              group='Export this page', page=page),
                export.Option('iso_file', 'As a separate file', 'bool', False,
                              group='Export this page', page=page),
            ]
        return o

    @staticmethod
    def _opt_float(spec, key):
        """A numeric export option, or None when the field is blank."""
        try:
            txt = str(spec['options'].get(key, '')).strip().replace(',', '.')
            return float(txt) if txt else None
        except (TypeError, ValueError):
            return None

    def _export_steps(self, spec):
        """Steps for the export, with Ca/K inverted to K/Ca when asked and the
        steps outside the plateau drawn red when asked.

        K/Ca = 1/(Ca/K), so the error transforms as sigma/(Ca/K)^2."""
        steps = self.steps_c
        if spec['options'].get('red_outside') and self.pset:
            steps = [dict(s0, status='excluded')
                     if (j not in self.pset and s0.get('status', 'normal') == 'normal')
                     else s0 for j, s0 in enumerate(steps)]
        if spec['options'].get('cak_mode') != 'kca':
            return steps
        return _kca_steps(steps)

    def _export_sessions(self, spec):
        """Stack sessions for the export: renamed aliquots, and Ca/K inverted
        to K/Ca when asked."""
        kca = spec['options'].get('cak_mode') == 'kca'
        tier = spec['options'].get('legend_ages', 'off')
        _, sig_scale, _ = self._disp()
        _af = f".{self._global_cfg.get('ageDec', 2)}f"
        # pad the names of the aliquots in the legend to one measured width,
        # so the ages start in the same column in the normal legend font
        _shown = [self._export_name(spec['options'], k, sess)
                  for k, sess in enumerate(self._sessions)
                  if not sess.get('excluded', False)]
        _pad = dict(zip(_shown, export.pad_names(_shown, 7.5)))
        out = []
        shown = 0
        for k, sess in enumerate(self._sessions):
            s2 = dict(sess)
            s2['title'] = self._export_name(spec['options'], k, sess)
            if not sess.get('excluded', False):
                # in a figure every drawn aliquot gets its own colour; by
                # session number, the excluded ones used up colours and the
                # legend showed the same colour twice
                s2['color'] = SAMPLE_COLORS[shown % len(SAMPLE_COLORS)]
                shown += 1
            if tier in ('all', 'int', 'ana', 'ext'):
                p = sess.get('plateau')
                if p and tier == 'all':
                    e = [(p.get(f'e1s_{t}') or 0.0) * sig_scale for t in ('ana', 'int', 'ext')]
                    s2['legend_label'] = (f"{_pad.get(s2.get('title', ''), s2.get('title', ''))}"
                                          f"{p['r']['mean']:{_af}} ± "
                                          f"{e[0]:{_af}}/{e[1]:{_af}}/{e[2]:{_af}} Ma")
                elif p:
                    e1s = p.get(f'e1s_{tier}', p.get('e1s_int', 0.0))
                    s2['legend_label'] = (f"{_pad.get(s2.get('title', ''), s2.get('title', ''))}"
                                          f"{p['r']['mean']:{_af}} ± "
                                          f"{e1s * sig_scale:{_af}} Ma")
                else:
                    s2['legend_label'] = (f"{_pad.get(s2.get('title', ''), s2.get('title', ''))}"
                                          "(no plateau)")
            if kca:
                s2['steps_c'] = _kca_steps(sess.get('steps_c', []))
            out.append(s2)
        return out

    def _export_name(self, opts, k, sess):
        """Aliquot k's name in the export: the stack table's, the experiment
        name it was loaded with, or the one typed in the dialog."""
        cur = sess.get('title', f'Aliquot {k + 1}')
        src = opts.get('names_from', 'window')
        if src == 'raw':
            return sess.get('orig_title') or cur
        if src == 'typed':
            return opts.get(f'name_{k}') or cur
        return cur

    def _raw_single_title(self):
        """The single-mode sample's name as its file came in (before any
        renaming in the title field)."""
        return getattr(self, '_raw_title', None) or self.title_edit.text()

    def _legend_ages_title(self, opts):
        """Legend heading when plateau ages are listed in it, e.g.
        'Plateau age ± 2σ (internal)'; empty for names only."""
        tier = {'int': 'internal', 'ana': 'analytical', 'ext': 'external',
                'all': 'analytical / internal / external'}.get(opts.get('legend_ages', 'off'))
        if not tier:
            return ''
        return f"Plateau age ± {self._disp()[2]} ({tier})"

    def _iso_lims(self, spec, kind):
        """Axis ranges typed for the inverse / normal diagram or the window in
        the age panel ('inset'); None = automatic."""
        pre = {'inverse': 'iso_inv', 'normal': 'iso_nor'}.get(kind, 'iso_ins')
        return {k: self._opt_float(spec, f'{pre}_{k}') for k in ('x0', 'x1', 'y0', 'y1')}

    def _draw_isochron_into(self, ax, spec, title=None, compact=False, kind=None):
        """Draw the current isochron into an existing axes.

        kind: 'inverse' / 'normal'; default from the page's Diagram choice.
        The compact window in the age panel has its own ranges, intercept
        marker and label (page 1); the full diagram reads page 2's."""
        if not hasattr(self, '_exp_iso'):
            self._exp_iso = IsochronCanvas()
            self._exp_iso._offscreen = True
        steps_list, J, lam, ttl, colors, zoom = self._isochron_inputs()
        if self._stack_mode and getattr(self, '_iso_si', None):
            colors = [SAMPLE_COLORS[k % len(SAMPLE_COLORS)]
                      for k in range(len(self._iso_si))]
        opts = spec.get('options', {})
        canvas = self._exp_iso
        if kind is None:
            kind = 'normal' if opts.get('iso_type', 'inverse') == 'normal' else 'inverse'
        canvas._showing_inverse = kind != 'normal'
        canvas._user_lims = self._iso_lims(spec, 'inset' if compact else kind)
        canvas._stats_pos = opts.get('iso_stats_pos', 'auto')
        canvas._stats_stacked = (opts.get('text_style') == 'big'
                                 or bool(opts.get('_stack_stats')))
        canvas._show_step_labels = bool(opts.get('iso_labels', True))
        canvas._err_tier = opts.get('iso_err_tier', 'int')
        canvas._err_line = bool(opts.get('iso_err_line', True))
        # always show the y-intercept: it carries the initial ⁴⁰Ar/³⁶Ar
        canvas._iso_zoom = 'intercept'
        canvas._compact = bool(compact)          # inset styling
        canvas._show_stats = bool(opts.get('iso_stats', True)) and not compact
        if compact:
            canvas._show_intercept = bool(opts.get('iso_inset_intercept', True))
            canvas._show_init_label = bool(opts.get('iso_inset_init', False))
            canvas._atm_ratio = ATM_40_36 if opts.get('iso_inset_atm') else None
            canvas._show_spread = False
        else:
            canvas._show_intercept = bool(opts.get('iso_intercept', True))
            canvas._show_init_label = bool(opts.get('iso_init'))
            canvas._atm_ratio = ATM_40_36 if opts.get('iso_atm') else None
            canvas._show_spread = bool(opts.get('iso_spread'))
        # stacked isochron: an aliquot legend (colour → name, as renamed on
        # page 1) so the page's legend placement has something to place
        canvas._legend_names = (
            [self._export_name(opts, k, self._sessions[k])
             for k in (getattr(self, '_iso_si', None) or [])]
            if (self._stack_mode and not compact) else None)
        canvas._group_jrel = self._iso_jrel
        canvas._group_rho = self._iso_rho
        canvas._target_ax = ax
        try:
            canvas.redraw(steps_list, J=J, lam=lam,
                          title=(ttl if title is None else title),
                          colors=colors, zoom=zoom,
                          cfg=getattr(self, '_iso_cfg', None) or self.cfg)
            if not compact:
                self._tidy_iso_ticks(ax, big=(opts.get('text_style') == 'big'))
        finally:
            canvas._target_ax = None
            canvas._compact = False
            canvas._show_stats = True
            canvas._atm_ratio = None
            canvas._show_init_label = False
            canvas._show_spread = False
            canvas._user_lims = None
            canvas._stats_pos = 'auto'
            canvas._stats_stacked = False
            canvas._err_tier = 'int'
            canvas._err_line = True
            canvas._show_step_labels = True
            canvas._legend_names = None

    @staticmethod
    def _tidy_iso_ticks(ax, big=False):
        """Few, compact tick labels: no 1e6 offset, 20k / 1.4M for large values."""
        from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter

        def compact(v, _pos=None):
            a = abs(v)
            if a >= 1e6:
                t = f'{v / 1e6:.3g}M'
            elif a >= 1e4:
                t = f'{v / 1e3:.3g}k'
            else:
                return f'{v:g}'
            return t
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            axis.set_major_locator(MaxNLocator(nbins=(4 if big else 5), prune=None))
            if max(abs(lim[0]), abs(lim[1])) >= 1e4:
                axis.set_major_formatter(FuncFormatter(compact))
            else:
                sf = ScalarFormatter(useOffset=False)
                sf.set_scientific(False)
                axis.set_major_formatter(sf)

    def _draw_isochron_pair(self, fig, ax, spec):
        """Replace the isochron panel *ax* by the inverse and the normal
        isochron side by side, in the same place."""
        pos = ax.get_position()
        ax.remove()
        big = spec['options'].get('text_style') == 'big'
        try:
            g0 = float(spec['options'].get('iso_pair_gap', 0.18))
        except (TypeError, ValueError):
            g0 = 0.18
        fw, fh = fig.get_size_inches()
        stacked = pos.height * fh > pos.width * fw          # a tall column
        if not hasattr(self, '_exp_iso'):
            self._exp_iso = IsochronCanvas()
            self._exp_iso._offscreen = True
        self._exp_iso._stats_fs = 6.0
        spec = dict(spec, options=dict(spec['options'], _stack_stats=True))
        try:
            if stacked:
                gap = pos.height * max(g0 + (0.04 if big else 0.0), 0.22)
                h = (pos.height - gap) / 2.0
                rects = [[pos.x0, pos.y0 + h + gap, pos.width, h],   # inverse on top
                         [pos.x0, pos.y0, pos.width, h]]
            else:
                gap = pos.width * (g0 + (0.04 if big else 0.0))
                w = (pos.width - gap) / 2.0
                rects = [[pos.x0 + k * (w + gap), pos.y0, w, pos.height]
                         for k in range(2)]
            for rect, kind in zip(rects, ('inverse', 'normal')):
                a = fig.add_axes(rect)
                self._draw_isochron_into(a, spec, title='', kind=kind)
        finally:
            self._exp_iso._stats_fs = 7.5

    def _export_render(self, fig, spec):
        """Draw the whole export figure. Called by the dialog and on save."""
        opts = spec['options']
        layout = [(pid, w) for pid, w in export.active_panels(spec)]
        cak_label = 'K/Ca' if opts.get('cak_mode') == 'kca' else 'Ca/K'
        title = opts.get('title', '') if opts.get('show_title', True) else ''
        if opts.get('show_title', True) and not self._stack_mode:
            src = opts.get('names_from', 'window')
            if src == 'window':
                title = self.title_edit.text()
            elif src == 'raw':
                title = self._raw_single_title()
            else:
                title = opts.get('title') or self.title_edit.text()
        # the age window is applied before drawing: the plateau bracket and its
        # label are positioned inside whatever range is in force
        auto = self._compute_age_range()
        y0, y1 = self._opt_float(spec, 'y_min'), self._opt_float(spec, 'y_max')
        age_range = (auto[0] if y0 is None else y0, auto[1] if y1 is None else y1)
        if age_range[1] <= age_range[0]:
            age_range = auto
        axes = {}
        if self._stack_mode:
            c_r_dict, c_ana, c_int, c_ext = self._combined_plateau()
            _draw_overlay_figure(
                fig, self._export_sessions(spec), age_range,
                title=title,
                show_bracket=bool(opts.get('show_bracket', True)),
                text_k=float(spec.get('font_scale', 1.0) or 1.0),
                label_top=self._label_on_top(spec, opts),
                bracket_at=self._opt_float(spec, 'bracket_at'),
                show_err_box=bool(opts.get('show_err_box', True)),
                global_cfg=self._global_cfg, combined_e1s=c_int,
                combined_e1s_ana=c_ana, combined_e1s_ext=c_ext,
                combined_mean=(c_r_dict['mean'] if c_r_dict else None),
                combined_r_dict=c_r_dict,
                display_1sigma=self._sigma_is_1s(),
                age_dec=self._global_cfg.get('ageDec', 2),
                panel_layout=layout, axes_out=axes, cak_label=cak_label,
                show_legend=bool(opts.get('show_legend', True)),
                legend_title=self._legend_ages_title(opts),
                legend_mono=False, gaps=self._layout_gaps(opts))
        else:
            if not hasattr(self, '_exp_canvas'):
                self._exp_canvas = PlateauCanvas()
            c = self._exp_canvas
            c.fig = fig
            c._offscreen = True
            c._panel_layout = layout
            c._layout_gaps = self._layout_gaps(opts)
            c._cak_label = cak_label
            c._show_title = bool(opts.get('show_title', True))
            c._show_step_labels = bool(opts.get('show_step_labels', True))
            c._show_int_age = bool(opts.get('show_int_age', True))
            c._show_err_box = bool(opts.get('show_err_box', True))
            c._show_bracket = bool(opts.get('show_bracket', True))
            c._bracket_at = self._opt_float(spec, 'bracket_at')
            c._ymax_locked = y1 is not None
            c._label_top = (self._label_on_top(spec, opts)
                            and c._bracket_at is None)
            c._text_k = float(spec.get('font_scale', 1.0) or 1.0)
            c._lam = self.cfg.get('lambdaYr', LAMBDA_DEFAULT)
            c.redraw(self._export_steps(spec), self.plateau, self.pset,
                     age_range, title,
                     display_1sigma=self._sigma_is_1s(),
                     age_dec=self.cfg.get('ageDec', 2))
            axes = c._axes
        # x window, and the ratio panel's own range (blank = as drawn)
        x0, x1 = self._opt_float(spec, 'x_min'), self._opt_float(spec, 'x_max')
        for ax in (axes.get('_spectrum') or []):
            if x0 is not None or x1 is not None:
                cur = ax.get_xlim()
                ax.set_xlim(cur[0] if x0 is None else x0,
                            cur[1] if x1 is None else x1)
        for pid, k_lo, k_hi in (('cak', 'cak_min', 'cak_max'),
                                ('p40', 'p40_min', 'p40_max')):
            ax_r = axes.get(pid)
            if ax_r in fig.axes:
                lo = self._opt_float(spec, k_lo)
                hi = self._opt_float(spec, k_hi)
                if lo is not None or hi is not None:
                    cur = ax_r.get_ylim()
                    ax_r.set_ylim(cur[0] if lo is None else lo,
                                  cur[1] if hi is None else hi)
        # short panels: thin out the y ticks so the numbers of neighbouring
        # panels cannot collide at the shared border
        for pid, w_ in layout:
            ax = axes.get(pid)
            if pid in ('p40', 'cak') and ax in fig.axes and w_ < 1.2:
                ax.yaxis.set_major_locator(
                    ticker.MaxNLocator(nbins=3, prune='both'))
        # isochron: as its own panel, as an inset, or both
        iso_axes = []
        if axes.get('iso') in fig.axes:
            # the panel carries its own 'Inverse/Normal Isochron' heading, so
            # the sample title is not repeated here
            before = set(fig.axes)
            if opts.get('iso_type') == 'both':
                self._draw_isochron_pair(fig, axes['iso'], spec)
            else:
                self._draw_isochron_into(axes['iso'], spec, title='')
            iso_axes = [a for a in fig.axes if a not in before] or [axes['iso']]
            if self._stack_mode:
                # the spectrum already carries the aliquot legend: one legend
                # for the page
                for a in iso_axes:
                    if a.get_legend() is not None:
                        a.get_legend().remove()
        ins = None
        if opts.get('iso_inset') and axes.get('age') in fig.axes:
            # keep a margin so the inset frame never sits on the panel frame
            m = 0.02
            w_i = max(0.10, min(1.0 - 2 * m, float(opts.get('iso_w', 38.0)) / 100.0))
            h_i = max(0.10, min(1.0 - 2 * m, float(opts.get('iso_h', 38.0)) / 100.0))
            x_i = min(max(float(opts.get('iso_x', 58.0)) / 100.0, m), 1.0 - m - w_i)
            y_i = min(max(float(opts.get('iso_y', 6.0)) / 100.0, m), 1.0 - m - h_i)
            ins = axes['age'].inset_axes([x_i, y_i, w_i, h_i])
            self._draw_isochron_into(ins, spec, title='', compact=True,
                                     kind=opts.get('iso_inset_type', 'inverse'))
            ins.set_title('')
            ins.set_facecolor('white')
            ins.patch.set_alpha(1.0)
            ins.set_zorder(8)
            for sp_ in ins.spines.values():
                sp_.set(linewidth=0.8, color='#888')
            ins.tick_params(labelsize=6)
            ins.xaxis.label.set_size(6.5)
            ins.yaxis.label.set_size(6.5)
        leg_spec = spec
        if opts.get('iso_inset') and spec.get('legend', 'auto') == 'auto':
            # the aliquot legend is drawn bottom-right, where the inset sits
            leg_spec = dict(spec, legend='inside-ul')
        if self._stack_mode and iso_axes and axes.get('age') in fig.axes:
            # the isochrons sit under the spectrum, so the shared legend goes
            # below everything, centred, instead of under the spectrum only
            if leg_spec.get('legend', 'auto') == 'auto':
                leg_spec = dict(leg_spec, legend='below')
            export.place_legend(axes['age'], leg_spec,
                                span=True, keep_cols=True)
            leg_spec = dict(leg_spec, legend='auto')
        for pid in ('age', 'p40', 'cak'):
            ax = axes.get(pid)
            if ax in fig.axes:
                export.place_legend(ax, leg_spec)
        export.scale_text(fig, spec.get('font_scale', 1.0),
                      spec.get('line_scale', 1.0), spec.get('marker_scale', 1.0))
        if ins is not None:
            # last, once the text has its final size
            self._keep_inset_inside(fig, ins, axes['age'])

    @staticmethod
    def _keep_inset_inside(fig, ins, parent, margin=0.02):
        """Move, and only if it must, shrink the isochron window so that its
        tick labels and axis titles stay inside the age panel, not just its
        frame.  The position and size set in the dialog are kept wherever
        they already fit."""
        from matplotlib.transforms import Bbox, TransformedBbox

        class _FracLocator:                    # bounds in parent axes fraction
            def __init__(self, b):
                self.b = b

            def __call__(self, ax, renderer):
                return TransformedBbox(Bbox.from_bounds(*self.b),
                                       parent.transAxes
                                       + ax.figure.transFigure.inverted())
        try:
            get = getattr(fig.canvas, 'get_renderer', None)
            if get is None:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                get = FigureCanvasAgg(fig).get_renderer
            r = get()
            for _ in range(4):                 # tick labels can change with size
                T = ins.get_tightbbox(r)       # frame + labels (applies locator)
                F = ins.get_window_extent(r)   # frame only
                P = parent.get_window_extent(r)
                if T is None or P.width <= 0 or P.height <= 0:
                    return
                l, b = max(F.x0 - T.x0, 0.0), max(F.y0 - T.y0, 0.0)
                rt, t = max(T.x1 - F.x1, 0.0), max(T.y1 - F.y1, 0.0)
                mx, my = margin * P.width, margin * P.height
                w = min(F.width, P.width - 2 * mx - l - rt)
                h = min(F.height, P.height - 2 * my - b - t)
                if w <= 0.1 * P.width or h <= 0.1 * P.height:
                    return                     # no sensible room: leave it
                x0 = min(max(F.x0, P.x0 + mx + l), P.x1 - mx - rt - w)
                y0 = min(max(F.y0, P.y0 + my + b), P.y1 - my - t - h)
                if (abs(x0 - F.x0) < 0.5 and abs(y0 - F.y0) < 0.5
                        and abs(w - F.width) < 0.5 and abs(h - F.height) < 0.5):
                    return
                ins.set_axes_locator(_FracLocator(
                    [(x0 - P.x0) / P.width, (y0 - P.y0) / P.height,
                     w / P.width, h / P.height]))
        except Exception:
            pass                               # never block the export

    def _export_render_isochron_only(self, fig, spec):
        """The isochron page: the chosen diagram, or both side by side."""
        fig.clear()
        opts = spec.get('options', {})
        if self._stack_mode and opts.get('iso_legend') is False:
            # same colours as the aliquot legend on page 1: one legend serves
            spec = dict(spec, legend='off')
        w_in, h_in = fig.get_size_inches()
        heading = opts.get('iso_heading', 'sample')
        text = (self._isochron_inputs()[3] if heading == 'sample' else
                (opts.get('iso_heading_text') or '').strip() if heading == 'custom'
                else '')
        if opts.get('iso_type') == 'both':
            rows, cols = ((1, 2) if w_in >= h_in else (2, 1))
            big = opts.get('text_style') == 'big'
            pair_spec = dict(spec, options=dict(opts, _stack_stats=True))
            lspec = dict(spec)
            if lspec.get('legend', 'auto') == 'auto':
                lspec['legend'] = 'below'
            axs = []
            if not hasattr(self, '_exp_iso'):
                self._exp_iso = IsochronCanvas()
                self._exp_iso._offscreen = True
            for k, kind in enumerate(('inverse', 'normal')):
                ax = fig.add_subplot(rows, cols, k + 1)
                axs.append(ax)
                self._exp_iso._stats_fs = 7.0
                try:
                    self._draw_isochron_into(ax, pair_spec, title='', kind=kind)
                finally:
                    self._exp_iso._stats_fs = 7.5
                if heading == 'none':
                    ax.set_title('')
                if k == 1 and ax.get_legend() is not None:
                    ax.get_legend().remove()       # one shared legend (below)
            if lspec.get('legend') != 'off' and axs[0].get_legend() is not None:
                if lspec['legend'] in ('below', 'above'):
                    export.place_legend(axs[0], lspec, span=True)
                else:
                    export.place_legend(axs[0], lspec)
            else:
                export.place_legend(axs[0], lspec)
            if text:
                fig.suptitle(text, fontsize=10, fontweight='bold')
            fig.subplots_adjust(left=0.09, right=0.985,
                                top=(0.88 if text else 0.92), bottom=0.13,
                                wspace=(0.34 if big else 0.28), hspace=0.36)
        else:
            ax = fig.add_subplot(111)
            # title '' keeps the diagram's own 'Inverse/Normal Isochron' heading
            self._draw_isochron_into(ax, spec, title=('' if heading == 'type' else text))
            if heading in ('none', 'custom') and not text:
                ax.set_title('')
            export.place_legend(ax, spec)
        export.scale_text(fig, spec.get('font_scale', 1.0),
                          spec.get('line_scale', 1.0), spec.get('marker_scale', 1.0))

    def _sigma_is_1s(self):
        """True when the window is currently displaying 1σ."""
        try:
            return bool(self._disp()[0])
        except Exception:
            return False

    def export_figure(self):
        """Open the shared export dialog: panels, sizes, text, legend, preview."""
        if self._isochron_mode:
            name = (self.title_edit.text().replace(' ', '_') or 'plateau') + '_isochron'
            dlg = export.FigureExportDialog(
                self, 'plateau_isochron', [export.Panel('iso', 'Isochron', fixed=True)],
                self._isochron_page_options(page=0),
                self._export_render_isochron_only, default_name=name)
        else:
            base = 'stacked_overlay' if self._stack_mode else \
                (self.title_edit.text().replace(' ', '_') or 'plateau')
            dlg = export.FigureExportDialog(
                self, 'plateau_stack' if self._stack_mode else 'plateau',
                self._export_panels(), self._export_options(),
                self._export_render, default_name=base,
                defaults=(dict(width_cm=18.0, height_cm=25.0)
                          if self._stack_mode else None),
                extra_render=self._export_render_isochron_only,
                extra_flag='iso_file', extra_page_flag='iso_page',
                extra_suffix='isochron', extra_label='isochron')
        dlg.exec_()
        self._queue_redraw()

    # ── Export every sample ──────────────────────────────────────────────────
    def _batch_panels(self):
        return [
            export.Panel('p40', '% \u2074\u2070Ar*',   on=True, weight=0.8),
            export.Panel('cak', 'Ca/K  (or K/Ca)',        on=True, weight=0.8),
            export.Panel('age', 'Age spectrum',           on=True, weight=2.8),
            export.Panel('iso', 'Isochrons',              on=True, weight=2.6),
        ]

    def _batch_options(self):
        """Switches of the 'Export All Samples' dialog: what the spectrum shows,
        the isochron switches, which samples, one PDF or one file each, and the
        text size.  Axis ranges are left out: every sample uses its own."""
        n = len(self._stack_slots()) if self._stack_mode else len(self._carousel)
        iso = [o for o in self._isochron_page_options(page=0)
               if not o.id.startswith(('iso_inv_', 'iso_nor_', 'iso_heading'))]
        for k, o in enumerate(iso):
            if o.id == 'iso_type':
                iso[k] = export.Option(
                    'iso_type', 'Diagrams', 'choice', group=o.group,
                    values=[('both', 'Inverse and normal, side by side'),
                            ('inverse', 'Inverse only'),
                            ('normal', 'Normal only')])
        out = []
        if n >= 2:
            out.append(export.Option(
                'samples', 'Samples', 'choice', group='Samples and output',
                values=[('all', f'All {n} samples'),
                        ('current', 'Current sample only')]
                if not self._stack_mode else [('all', f'All {n} sessions')],
                tip='The preview always shows the sample that is open now '
                    '(in stack mode, the first session).'))
        out += [
            export.Option('batch_one_pdf', 'One PDF, one page per sample', 'bool',
                          True, group='Samples and output',
                          tip='Untick to write one file per sample into a folder. '
                              'PNG, SVG, EPS and TIFF always go that way.'),
            export.Option('text_style', 'Text', 'choice', group='Text size',
                          values=[('displayed', 'As displayed'),
                                  ('big', 'Big text, for small figures in print')],
                          tip='Big text enlarges the numbers, letters, lines and '
                              'points by the factor below, on top of the Text × '
                              'and Lines × above, so they stay readable when '
                              'the figure is reduced to fit a page.'),
            export.Option('big_factor', 'Big text ×', 'float', default=1.6,
                          lo=1.0, hi=3.0, step=0.1, group='Text size',
                          depends=('text_style', 'big')),
            export.Option('cak_mode', 'Ratio panel', 'choice',
                          default='kca' if self._kca_view() else 'cak',
                          values=[('cak', 'Ca/K'), ('kca', 'K/Ca')]),
            export.Option('show_title', 'Title', 'bool', True),
            export.Option('show_err_box', 'Error box', 'bool', True),
            export.Option('show_bracket', 'Plateau bracket', 'bool', True),
            export.Option('show_step_labels', 'Step labels', 'bool', True),
            export.Option('show_int_age', 'Total-gas age', 'bool', True),
            export.Option('red_outside', 'Steps outside the plateau in red', 'bool',
                          True),
        ]
        return out + self._layout_options() + iso

    @staticmethod
    def _iso_batch_spec(spec):
        """The spec one page of the batch is drawn with: with Big text, the
        larger text, lines and markers."""
        out = dict(spec)
        out['options'] = dict(spec['options'])
        if out['options'].get('text_style') == 'big':
            f = float(out['options'].get('big_factor', 1.6))
            g = 1.0 + 0.5 * (f - 1.0)
            out['font_scale'] = spec.get('font_scale', 1.0) * f
            out['line_scale'] = spec.get('line_scale', 1.0) * g
            out['marker_scale'] = spec.get('marker_scale', 1.0) * g
        return out

    def _stack_slots(self):
        """The shown stack sessions as carousel-style slots (k, slot)."""
        return [(k, dict(title=sess.get('title') or f'Session {k + 1}',
                         raw_steps=sess['raw_steps'], cfg=sess['cfg'],
                         flux_hole=None))     # already baked into sess['cfg']
                for k, sess in enumerate(self._sessions)
                if not sess.get('excluded', False)]

    def _single_state(self):
        return dict(raw_steps=copy.deepcopy(self.raw_steps), cfg=self.cfg.copy(),
                    steps_c=list(self.steps_c), plateau=self.plateau,
                    pset=set(self.pset), flux=getattr(self, '_sp_flux_hole', None),
                    title=self.title_edit.text(), stack=self._stack_mode)

    def _restore_single_state(self, st):
        self.raw_steps = st['raw_steps']
        self.cfg = st['cfg']
        self.steps_c = st['steps_c']
        self.plateau = st['plateau']
        self.pset = st['pset']
        self._sp_flux_hole = st['flux']
        self._stack_mode = st['stack']
        self.title_edit.blockSignals(True)
        self.title_edit.setText(st['title'])
        self.title_edit.blockSignals(False)

    def _export_render_batch(self, fig, spec):
        """Preview of the batch dialog: the open sample (in stack mode the
        first session), as one page."""
        if not self._stack_mode:
            self._export_render(fig, self._iso_batch_spec(spec))
            return
        slots = self._stack_slots()
        if not slots:
            return
        st = self._single_state()
        try:
            self._stack_mode = False
            self._iso_load_slot(slots[0][1], st['cfg'])
            self._export_render(fig, self._iso_batch_spec(spec))
        finally:
            self._restore_single_state(st)

    def _iso_load_slot(self, slot, base_cfg):
        """Make a carousel slot the loaded sample, as Save All Sessions does,
        without touching the widgets."""
        self.raw_steps = copy.deepcopy(slot['raw_steps'])
        self.cfg = base_cfg.copy()
        self.cfg.update(slot['cfg'])
        self._sp_flux_hole = slot.get('flux_hole')
        eff = self._effective_single_cfg()
        self.steps_c = add_cumulative(
            _compute_steps_with_age_mode(self.raw_steps, eff))
        self.plateau = find_plateau(self.steps_c, eff)
        self.pset = set(self.plateau['ois']) if self.plateau else set()
        self.title_edit.blockSignals(True)
        self.title_edit.setText(slot.get('title', ''))
        self.title_edit.blockSignals(False)

    def _export_samples_all(self, spec, path, progress):
        """Write one page per chosen sample (spectrum + isochrons, see the
        dialog) to *path*: a PDF, or a folder with one file per sample.
        Returns the lines to show the user."""
        from matplotlib.backends.backend_pdf import PdfPages
        opts = spec['options']
        fmt = spec.get('fmt', 'pdf')
        one_pdf = fmt == 'pdf' and bool(opts.get('batch_one_pdf', True))

        self._carousel_save_current()
        stack_slots = self._stack_slots() if self._stack_mode else None
        if stack_slots is not None:
            slots = stack_slots
        elif self._carousel:
            slots = list(enumerate(self._carousel))
            if opts.get('samples') == 'current' and 0 <= self._carousel_idx < len(slots):
                slots = [slots[self._carousel_idx]]
        else:
            slots = [(0, self._carousel_slot_from_current())]

        saved = self._single_state()
        written, no_iso, used, pages = [], [], set(), 0
        pdf = None
        try:
            self._stack_mode = False     # each session is drawn as a single sample
            with matplotlib.rc_context(export.EDITABLE_TEXT):
                if one_pdf:
                    pdf = PdfPages(path)
                for si, (k, slot) in enumerate(slots):
                    title = slot.get('title') or f'Sample {k + 1}'
                    if progress(si, len(slots), f'{title}   ({si + 1} of {len(slots)})'):
                        break
                    self._iso_load_slot(slot, saved['cfg'])
                    s2 = self._iso_batch_spec(spec)
                    if not any(s.get('iso_nx') is not None or s.get('iso_ix') is not None
                               for s in self.raw_steps):
                        no_iso.append(title)
                        s2['panels'] = [dict(p, on=False) if p['id'] == 'iso' else p
                                        for p in spec['panels']]
                    fig = Figure(figsize=export.figsize(s2), facecolor='white')
                    self._export_render(fig, s2)
                    if pdf is not None:
                        export.settle_legends(fig)
                        pdf.savefig(fig, dpi=int(s2.get('dpi', 300)),
                                    facecolor='white', bbox_inches='tight',
                                    pad_inches=0.05)
                        pages += 1
                    else:
                        base = re.sub(r'[^\w.+-]+', '_', title).strip('_') \
                            or f'sample_{k + 1}'
                        name, n_ = base, 1
                        while name in used:
                            n_ += 1
                            name = f'{base}_{n_}'
                        used.add(name)
                        fp = os.path.join(path, f'{name}.{fmt}')
                        export.save_figure(fig, fp, s2)
                        written.append(fp)
                progress(len(slots), len(slots), '')
        finally:
            if pdf is not None:
                # the fonts are written when the file is closed, so the
                # editable-text setting must still be on then
                with matplotlib.rc_context(export.EDITABLE_TEXT):
                    pdf.close()
            self._restore_single_state(saved)
        lines = []
        if no_iso:
            lines.append('No isochron data, spectrum only: ' + ', '.join(no_iso))
        if pdf is not None and pages:
            lines.append(f'{path}   ({pages} page{"s" if pages != 1 else ""})')
        return lines + written

    def export_all_samples(self):
        """Dialog for exporting every loaded sample: one page each, with the
        age spectrum and the isochrons."""
        if self._stack_mode:
            if not self._stack_slots():
                QMessageBox.information(self, "Export All Samples",
                                        "No sessions loaded.")
                return
        elif not self._carousel and not self.raw_steps:
            QMessageBox.information(self, "Export All Samples", "No sessions loaded.")
            return
        dlg = export.FigureExportDialog(
            self, 'plateau_all_samples', self._batch_panels(),
            self._batch_options(), self._export_render_batch,
            default_name='all_samples', batch=self._export_samples_all,
            batch_pdf_flag='batch_one_pdf', title='Export all samples',
            defaults=dict(width_cm=21.0, height_cm=24.0))
        dlg.exec_()
        self._queue_redraw()

    def export_all_pdf(self):
        """Export every loaded aliquot (carousel) as one page in a multi-page PDF.

        Iterates self._carousel without touching the UI, renders each slot's
        spectrum directly via canvas.redraw(), and saves to PdfPages.  The
        currently displayed slot is restored exactly as found.
        """
        from matplotlib.backends.backend_pdf import PdfPages

        n = len(self._carousel)
        if self._stack_mode or self._isochron_mode:
            QMessageBox.information(
                self, "Export All",
                "Export All Sessions is only available in single-session (carousel) mode.")
            return
        if n == 0 and not self.raw_steps:
            QMessageBox.information(self, "Export All", "No sessions loaded.")
            return

        # Build the list of slots to export:
        # carousel active → all carousel slots; single session → wrap current data
        if n >= 1:
            slots_to_export = self._carousel
            first_title = (self._carousel[0].get('title', '') or 'plateaus').replace(' ', '_')
        else:
            slots_to_export = [{'title': self.title_edit.text(),
                                 'raw_steps': self.raw_steps,
                                 'cfg': self.cfg.copy(),
                                 'plateau': self.plateau}]
            first_title = (self.title_edit.text() or 'plateau').replace(' ', '_')

        # ── Ask for save path ────────────────────────────────────────────────
        path, _ = dfold.get_save_file_name(
            self, "Export All Sessions as PDF",
            first_title + '_all.pdf', 'PDF (*.pdf)')
        if not path:
            return

        # ── Snapshot current state (to restore after export) ─────────────────
        saved_raw       = copy.deepcopy(self.raw_steps)
        saved_cfg       = self.cfg.copy()
        saved_steps_c   = list(self.steps_c)
        saved_plateau   = self.plateau
        saved_pset      = set(self.pset)
        saved_flux_hole = getattr(self, '_sp_flux_hole', None)
        saved_title     = self.title_edit.text()
        saved_idx       = self._carousel_idx

        display_1sigma, _, _ = self._disp()
        n_exp    = len(slots_to_export)
        _success = False

        try:
            with PdfPages(path) as pdf:
                for i, slot in enumerate(slots_to_export):
                    # Load slot data into instance variables (no Qt widget updates)
                    self.raw_steps     = copy.deepcopy(slot['raw_steps'])
                    self.cfg           = saved_cfg.copy()   # keep shared UI settings…
                    self.cfg.update(slot['cfg'])            # …then overlay per-slot cfg
                    self._sp_flux_hole = slot.get('flux_hole')  # per-slot flux assignment

                    eff          = self._effective_single_cfg()
                    steps        = _compute_steps_with_age_mode(self.raw_steps, eff)
                    self.steps_c = add_cumulative(steps)
                    self.plateau = find_plateau(self.steps_c, eff)
                    self.pset    = set(self.plateau['ois']) if self.plateau else set()
                    title        = slot.get('title', '') or f'Aliquot {i + 1}'
                    age_range    = self._compute_age_range()
                    age_dec      = self.cfg.get('ageDec', 2)

                    # Tag non-plateau steps as excluded so they render red in the PDF
                    steps_pdf = []
                    for j, s in enumerate(self.steps_c):
                        s2 = dict(s)
                        if self.pset and j not in self.pset and s2.get('status', 'normal') == 'normal':
                            s2['status'] = 'excluded'
                        steps_pdf.append(s2)

                    self.canvas._lam = self.cfg.get('lambdaYr', LAMBDA_DEFAULT)
                    self.canvas.redraw(
                        steps_pdf, self.plateau, self.pset,
                        age_range, title,
                        display_1sigma=display_1sigma, age_dec=age_dec)
                    pdf.savefig(self.canvas.fig, facecolor='white')

            _success = True

        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

        finally:
            # ── Restore original state unconditionally ───────────────────────
            self.raw_steps         = saved_raw
            self.cfg               = saved_cfg
            self.steps_c           = saved_steps_c
            self.plateau           = saved_plateau
            self.pset              = saved_pset
            self._sp_flux_hole     = saved_flux_hole
            self._carousel_idx     = saved_idx
            self.title_edit.blockSignals(True)
            self.title_edit.setText(saved_title)
            self.title_edit.blockSignals(False)
            # Redraw the canvas with the original slot's content
            try:
                age_range = self._compute_age_range()
                self.canvas.redraw(
                    self.steps_c, self.plateau, self.pset,
                    age_range, saved_title,
                    display_1sigma=display_1sigma,
                    age_dec=saved_cfg.get('ageDec', 2))
            except Exception:
                pass

        if _success:
            QMessageBox.information(
                self, "Export OK",
                f"Saved {n_exp} page{'s' if n_exp != 1 else ''} to:\n{path}")

    # ─── Isochron age helper ──────────────────────────────────────────────────
    def _calc_isochron_result(self, steps, is_inverse, J, lam, cfg=None):
        """York-regression isochron age from step data (isochron_tiers, the
        rule the on-screen isochron uses).

        is_inverse=True  → Inverse isochron (39K/40Ar vs 36Ar/40Ar)
        is_inverse=False → Normal  isochron (39K/36Ar vs 40Ar/36Ar)
        cfg is the calibration for the external tier (default: the window's).

        Returns dict with age_ma, e1s_ana/int/ext, init_ratio, init_ratio_1s,
        mswd, n, prob: or None if regression fails / data absent.
        """
        if is_inverse:
            x_k, sx_k, y_k, sy_k, ri_k = ('iso_nx', 'iso_nsx',
                                            'iso_ny', 'iso_nsy', 'iso_iri')
        else:
            x_k, sx_k, y_k, sy_k, ri_k = ('iso_ix', 'iso_isx',
                                            'iso_iy', 'iso_isy', 'iso_nri')

        pts = [(s[x_k], s[sx_k], s[y_k], s[sy_k], s.get(ri_k, 0.0) or 0.0)
               for s in steps
               if (s.get(x_k) is not None and s.get(sx_k) is not None
                   and s.get(y_k) is not None and s.get(sy_k) is not None
                   and s[x_k] > 0 and s[y_k] > 0
                   and s.get('status', 'normal') != 'excluded')]
        if not pts or J is None or J <= 0:
            return None

        xs, sxs, ys, sys_, ris = map(np.array, zip(*pts))
        fit = york_regression(xs, sxs, ys, sys_, ris)
        if fit is None:
            return None
        it = isochron_tiers(steps, fit, J, lam, is_inverse,
                            cfg if cfg is not None else self.cfg)
        if it['e1s_ana'] is None:
            return None
        return {
            'age_ma':        it['age_ma'],
            'e1s_ana':       it['e1s_ana'],
            'e1s_int':       it['e1s_int'],
            'e1s_ext':       it['e1s_ext'],
            'init_ratio':    it['init_ratio'],
            'init_ratio_1s': it['init_ratio_1s'],
            'mswd':          fit['mswd'],
            'n':             fit['n'],
            'prob':          fit['prob'],
        }

    def _stacked_isochron_result(self, is_inverse):
        """The stacked isochron of the Isochron view in stack mode: one York
        fit to the plateau steps of every aliquot in the stack, each aliquot's
        ³⁹Ar on J_ref and the J term with each aliquot's own J error, correlated
        by hole / disk.  Same inputs (_isochron_inputs) and the same steps as
        IsochronCanvas.redraw, so the export gives the numbers on the screen.

        Returns _calc_isochron_result's dict plus 'n_aliquots' and 'cfg' (the
        calibration of the J_ref aliquot), or None."""
        steps_list, J, lam, _t, _c, _z = self._isochron_inputs()
        cfg = getattr(self, '_iso_cfg', None) or self.cfg
        if is_inverse:
            x_k, sx_k, y_k, sy_k, ri_k = ('iso_nx', 'iso_nsx',
                                            'iso_ny', 'iso_nsy', 'iso_iri')
        else:
            x_k, sx_k, y_k, sy_k, ri_k = ('iso_ix', 'iso_isx',
                                            'iso_iy', 'iso_isy', 'iso_nri')
        pts = [(s[x_k], s[sx_k], s[y_k], s[sy_k], s.get(ri_k, 0.0) or 0.0, gi)
               for gi, grp in enumerate(steps_list) for s in grp
               if (s.get(x_k) is not None and s.get(sx_k) is not None
                   and s.get(y_k) is not None and s.get(sy_k) is not None
                   and s[x_k] > 0 and s[y_k] > 0
                   and s.get('status', 'normal') != 'excluded')]
        if not pts or J is None or J <= 0:
            return None
        xs, sxs, ys, sys_, ris, gis = map(np.array, zip(*pts))
        fit = york_regression(xs, sxs, ys, sys_, ris)
        if fit is None:
            return None
        age = isochron_age(fit, J, lam, is_inverse)['age_ma']
        canvas = self.isochron_canvas
        canvas._group_jrel, canvas._group_rho = self._iso_jrel, self._iso_rho
        sJ = (canvas._group_j_sigma(xs, sxs, ys, sys_, ris, gis, J, lam,
                                    is_inverse, age)
              if age is not None else None)
        it = isochron_tiers([s for grp in steps_list for s in grp], fit, J, lam,
                            is_inverse, cfg, sJ_age=sJ)
        if it['e1s_ana'] is None:
            return None
        return {
            'age_ma':        it['age_ma'],
            'e1s_ana':       it['e1s_ana'],
            'e1s_int':       it['e1s_int'],
            'e1s_ext':       it['e1s_ext'],
            'init_ratio':    it['init_ratio'],
            'init_ratio_1s': it['init_ratio_1s'],
            'mswd':          fit['mswd'],
            'n':             fit['n'],
            'prob':          fit['prob'],
            'n_aliquots':    len(set(gis.tolist())),
            'cfg':           cfg,
        }

    def export_csv(self):
        if self._stack_mode:
            self._stack_export_csv()
            return
        if not self.steps_c:
            QMessageBox.warning(self, "No Data", "No data to export.")
            return
        default = self.title_edit.text().replace(' ', '_') or 'plateau'
        slots = None
        if len(self._carousel) > 1:
            box = QMessageBox(self)
            box.setWindowTitle("Export Data")
            box.setText(f"Export all {len(self._carousel)} samples, or only the open one?")
            b_all = box.addButton("All samples", QMessageBox.AcceptRole)
            b_one = box.addButton("This sample", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(b_all)
            box.exec_()
            if box.clickedButton() is b_all:
                self._carousel_save_current()
                slots = list(enumerate(self._carousel))
                default = 'all_samples'
            elif box.clickedButton() is not b_one:
                return
        path, _flt = dfold.get_save_file_name(
            self, "Export Data", default + '.xlsx', export.save_filter())
        if not path:
            return
        path = export.ensure_ext(path, _flt)

        try:
            import csv
            _sink = export.TableSink(path)
            with _sink as f:
                w = csv.writer(f)
                if slots is None:
                    _entries = [(self.cfg, self.cfg.get('constants'),
                                 [export.step_meta(s) for s in self.steps_c])]
                else:
                    _entries = [({**self.cfg, **sl['cfg']},
                                 sl['cfg'].get('constants'),
                                 [export.step_meta(s) for s in sl['raw_steps']])
                                for _k, sl in slots]
                export.write_constants(w, groups=export.constants_groups(_entries),
                                     atm=ATM_40_36)
                _raw_items = []
                if slots is None:
                    self._csv_write_sample(w)
                    _raw_items.append((self.title_edit.text() or '', self.steps_c,
                                       [export.step_meta(s) for s in self.steps_c],
                                       self.cfg.get('constants')))
                else:
                    _saved = self._single_state()
                    try:
                        for _k, _slot in slots:
                            self._iso_load_slot(_slot, _saved['cfg'])
                            self._csv_write_sample(w)
                            w.writerow([])      # blank row between samples
                            _raw_items.append((self.title_edit.text() or f'Sample {_k + 1}',
                                               self.steps_c,
                                               [export.step_meta(s) for s in self.steps_c],
                                               (_slot.get('cfg') or {}).get('constants')))
                    finally:
                        self._restore_single_state(_saved)
                _sink.set_raw_items(_raw_items)
            QMessageBox.information(self, "Export OK", f"Saved:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _csv_write_sample(self, w):
        """Step table, plateau, integrated age and isochron results of the
        loaded sample, written to csv writer *w*."""
        display_1sigma, sig_scale, sig_lbl = self._disp()
        plat_title = self.title_edit.text() or ''
        plat = self.plateau
        ir = integrated_age(self.steps_c, self.cfg.get('lambdaYr', LAMBDA_DEFAULT))
        _metas = [export.step_meta(s) for s in self.steps_c]
        # The R standard / J in force: the typed values with the assigned
        # disk-flux hole's R overlaid, exactly what the ages were computed from.
        _cfg_e = self._effective_single_cfg()
        _jc = _export_j_cells(_cfg_e)

        def _fo(v):    return fsig(v, 8)   # 8 sig figs for ratio values
        def _foe(v):   return fsig(v, 6)   # 6 sig figs for ratio errors

        _mode_lbl = _AGEMODE_LABEL.get(_cfg_e.get('ageFluxMode', 'imported'), 'imported')
        w.writerow([f'STEP DATA, {plat_title}  [Age source: {_mode_lbl}]'])
        w.writerow(['RunID', 'Label', 'Power', 'Run date/time',
                    'Days since irradiation', *export.FLUX_HEADERS,
                    'Age (Ma)', '±1σ Ana (Ma)',
                    '%39Ar', 'K/Ca', '±1σ', '%40Ar*',
                    '40Ar*/39ArK', '±1σ',
                    'R_unk/R_std', '±1σ',
                    '39Ar(k)/36Ar', '±1σ', '40Ar/36Ar', '±1σ', 'RI(norm)',
                    '39Ar(k)/40Ar', '±1σ', '36Ar/40Ar', '±1σ', 'RI(inv)',
                    'In plateau', 'Status'])
        for i, s in enumerate(self.steps_c):
            cak = s['caK']
            if cak and cak > 0:
                kca_str   = fsig(1.0 / cak, 5)
                kca_e_str = (fsig(s['err_cak'] / (cak * cak), 3)
                             if s.get('err_cak') is not None else '')
            else:
                kca_str = kca_e_str = ''
            w.writerow([
                s.get('runID', ''),
                s['label'],
                f"{s['power']:.3f}",
                s.get('run_dt', ''), export.fmt_days(_metas[i]),
                *export.flux_cells(_cfg_e, *(_jc or (s.get('j_val'), s.get('j_err'))),
                                 show_r=_show_r_standard(_cfg_e)),
                f"{s['age']:.8f}",
                f"{s['err1s']:.8f}",
                f"{s['pct39Ar']:.3f}",
                kca_str,
                kca_e_str,
                f"{s['pct40']:.1f}",
                ff(s.get('r4039'), 8),   ff(s.get('err_r4039'), 8),
                *_fo_rr(s, _cfg_e),
                _fo(s.get('iso_ix')),  _foe(s.get('iso_isx')),
                _fo(s.get('iso_iy')),  _foe(s.get('iso_isy')),
                _fo(s.get('iso_nri')),
                _fo(s.get('iso_nx')),  _foe(s.get('iso_nsx')),
                _fo(s.get('iso_ny')),  _foe(s.get('iso_nsy')),
                _fo(s.get('iso_iri')),
                'yes' if i in self.pset else 'no',
                s.get('status', 'normal'),
            ])
        w.writerow([])

        # ── Plateau results ───────────────────────────────────────────
        w.writerow(['PLATEAU RESULTS'])
        w.writerow(['Sample',
                    'Plateau age (Ma)',
                    f'±{sig_lbl} Ana (Ma)', f'±{sig_lbl} Int (Ma)', f'±{sig_lbl} Ext (Ma)',
                    'R_unk/R_std', f'±{sig_lbl} Ana R', f'±{sig_lbl} Int R', f'±{sig_lbl} Ext R',
                    '%39Ar', 'MSWD', 'p', 'n steps', 'Manual'])
        if plat:
            r = plat['r']
            _rr_p, _drdt_p = rr_from_age(r['mean'], _cfg_e)
            w.writerow([plat_title,
                        f"{r['mean']:.8f}",
                        f"{plat.get('e1s_ana', plat['e1s_int']) * sig_scale:.8f}",
                        f"{plat['e1s_int'] * sig_scale:.8f}",
                        f"{plat.get('e1s_ext', plat['e1s_int']) * sig_scale:.8f}",
                        ff(_rr_p, 8),
                        ff(_drdt_p * plat.get('e1s_ana', plat['e1s_int']) * sig_scale, 8) if _drdt_p else '',
                        ff(_drdt_p * plat['e1s_int'] * sig_scale, 8) if _drdt_p else '',
                        ff(_drdt_p * plat.get('e1s_ext', plat['e1s_int']) * sig_scale, 8) if _drdt_p else '',
                        f"{plat['pct39']:.2f}",
                        f"{r['mswd']:.3f}",
                        f"{r['prob']:.4f}",
                        plat['n'],
                        'yes' if plat.get('forced') else 'no'])
        else:
            w.writerow([plat_title] + [''] * 13)
        w.writerow([])
        w.writerow(['Total-gas age (Ma)', f'±{sig_lbl} (Ma)',
                    'R_unk/R_std', f'±{sig_lbl} R',
                    'MSWD', 'p', 'n steps'])
        if ir:
            _rr_i, _drdt_i = rr_from_age(ir['mean'], _cfg_e)
            w.writerow([f"{ir['mean']:.8f}",
                        f"{ir['sem'] * sig_scale:.8f}",
                        ff(_rr_i, 8),
                        ff(_drdt_i * ir['sem'] * sig_scale, 8) if _drdt_i else '',
                        '', '',                 # a total-gas age has no MSWD
                        ir.get('n', '')])
        w.writerow([])

        # ── Isochron results ──────────────────────────────────────────
        # The isochron on screen: plateau steps only, on the plateau's J
        _iso_steps = self._tag_isochron_steps(self.steps_c, self.pset or set())
        _J   = self._iso_J(_cfg_e, self.steps_c)
        _lam = _cfg_e.get('lambdaYr',   LAMBDA_DEFAULT)
        norm_r = self._calc_isochron_result(_iso_steps, False, _J, _lam, _cfg_e)
        inv_r  = self._calc_isochron_result(_iso_steps, True,  _J, _lam, _cfg_e)
        w.writerow(['ISOCHRON RESULTS'])
        w.writerow(['Type', 'Age (Ma)',
                    f'±{sig_lbl} Ana (Ma)', f'±{sig_lbl} Int (Ma)', f'±{sig_lbl} Ext (Ma)',
                    'R_unk/R_std', f'±{sig_lbl} Ana R', f'±{sig_lbl} Int R', f'±{sig_lbl} Ext R',
                    'Initial 40Ar/36Ar', f'±{sig_lbl}', 'MSWD', 'p', 'n'])
        for label, res in [('Normal', norm_r), ('Inverse', inv_r)]:
            if res:
                ir40_36 = (f"{res['init_ratio']:.2f}" if res['init_ratio'] is not None else '')
                ir40_36_e = (f"{res['init_ratio_1s'] * sig_scale:.2f}" if res['init_ratio_1s'] is not None else '')
                _rr_iso, _drdt_iso = rr_from_age(res['age_ma'], _cfg_e)
                w.writerow([label,
                            f"{res['age_ma']:.8f}",
                            f"{res['e1s_ana'] * sig_scale:.8f}",
                            f"{res['e1s_int'] * sig_scale:.8f}",
                            f"{res['e1s_ext'] * sig_scale:.8f}",
                            ff(_rr_iso, 8),
                            ff(_drdt_iso * res['e1s_ana'] * sig_scale, 8) if _drdt_iso else '',
                            ff(_drdt_iso * res['e1s_int'] * sig_scale, 8) if _drdt_iso else '',
                            ff(_drdt_iso * res['e1s_ext'] * sig_scale, 8) if _drdt_iso else '',
                            ir40_36, ir40_36_e,
                            f"{res['mswd']:.3f}",
                            f"{res['prob']:.4f}",
                            res['n']])
            else:
                w.writerow([label] + [''] * 13)

    # ══════════════════════════════════════════════════════════════════════════
    # Stack-mode methods (multi-aliquot overlay)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Flux-group helpers ────────────────────────────────────────────────────
    _FLUX_KEYS = ('ageFluxMode', 'jVal', 'jErrPct', 'jForce',
                  'rStd', 'rStdErrPct', 'rStdErrMode',
                  'stdAgeMa', 'stdAgeErrMa', 'stdAgeErrType',
                  'lambdaYr', 'lambdaErrYr') + tuple(ROUTE_KEYS)

    @staticmethod
    def _flux_key(cfg):
        # aliquots on different routes (monitors) never share a flux group
        return PlateauWindow._flux_key_vals(cfg) + (route_line(route_from_cfg(cfg)),)

    @staticmethod
    def _flux_key_vals(cfg):
        mode = cfg.get('ageFluxMode', 'imported')
        if mode == 'ratio_J':
            return ('ratio_J', round(cfg.get('jVal', 0), 8),
                    round(cfg.get('jErrPct', 0), 6))
        elif mode == 'ratio_R_std':
            return ('ratio_R_std',
                    round(cfg.get('rStd', 70), 6),
                    round(cfg.get('rStdErrPct', 0.10), 6),
                    cfg.get('rStdErrMode', 'pct'),
                    round(cfg.get('stdAgeMa', STD_AGE_DEFAULT), 5),
                    round(cfg.get('stdAgeErrMa', STD_AGE_ERR), 7),
                    round(cfg.get('lambdaYr', LAMBDA_DEFAULT), 17),
                    round(cfg.get('lambdaErrYr', LAMBDA_ERR), 20))
        # 'imported' mode: sessions are grouped by their numerical R_std value
        # (or J value if R_std is absent). This ensures sessions with different
        # flux calibrations are correctly identified as separate flux groups,
        # even when the mode is 'imported'. Without this, every imported session
        # would hash to the same key ('imported',) regardless of J value.
        r_std = round(cfg.get('rStd', 0.0), 6)
        j_val = round(cfg.get('jVal', 0.0), 8)
        if r_std > 0:
            return ('imported', r_std, round(cfg.get('rStdErrPct', 0.0), 6))
        elif j_val > 0:
            return ('imported_J', j_val, round(cfg.get('jErrPct', 0.0), 6))
        return ('imported',)

    @classmethod
    def _extract_flux_cfg(cls, cfg):
        out = {k: cfg.get(k, _SESSION_CFG_DEFAULTS.get(k)) for k in cls._FLUX_KEYS}
        # route keys only when the session has them (an empty stdCov would
        # set a joint covariance to zero)
        return {k: v for k, v in out.items() if k not in ROUTE_KEYS or v is not None}

    # ── Stacking method (WM + √MSWD / GLS + ML) + disk-flux import ─────────────
    def _open_single_stats(self):
        """Open the statistics dialog for the current single plateau."""
        if not getattr(self, 'plateau', None):
            QMessageBox.information(self, "Stats", "No plateau is defined yet.")
            return
        dlg = _StatsDialog(self, mode='single')
        dlg.exec_()

    def _stack_copy_ages(self):
        """Copy each aliquot's plateau age and 2σ errors to the clipboard.

        Format: tab-separated rows of  label ⟨tab⟩ age ⟨tab⟩ ±2σ_ana ⟨tab⟩ ±2σ_int
        so a single Ctrl-V into Excel lands them in four consecutive columns.
        The analytical error is the one to combine with outside this tool (the
        J part of the internal error is shared between aliquots on one hole).
        Aliquots that have no plateau are skipped.
        """
        if not self._sessions:
            QMessageBox.information(self, 'Copy Ages',
                                    'No sessions in stack.')
            return

        rows = []
        skipped = 0
        for sess in self._sessions:
            p = sess.get('plateau')
            if p is None:
                skipped += 1
                continue
            title = sess.get('title', '')
            age   = p['r']['mean']
            e_ana = p.get('e1s_ana', p.get('e1s_int', 0.0))
            e_int = p.get('e1s_int', e_ana)
            rows.append(f"{title}\t{age:.4f}\t{e_ana * 2:.4f}\t{e_int * 2:.4f}")

        if not rows:
            QMessageBox.information(self, 'Copy Ages',
                                    'No plateau ages available to copy.')
            return

        QApplication.clipboard().setText('\n'.join(rows))
        msg = (f'Copied {len(rows)} plateau age(s) to clipboard '
               '(columns: label, age, 2σ analytical, 2σ internal).')
        if skipped:
            msg += f'\n({skipped} aliquot(s) skipped, no plateau defined.)'
        QMessageBox.information(self, 'Copy Ages', msg)

    def _open_stack_stats(self):
        """Open the read-only statistics dialog comparing the two combination
        methods and offering the results as tab-separated rows."""
        if not self._sessions:
            QMessageBox.information(self, "Stats", "No aliquots in the stack yet.")
            return
        dlg = _StatsDialog(self, mode="stack")
        dlg.exec_()

    def _on_wm_n(self, _idx=None):
        """Switch the WM+√MSWD combine basis (n=steps vs n=aliquots) and refresh."""
        self._wm_n_basis = self._cb_wm_n.currentData() or 'steps'
        if self._sessions and hasattr(self, '_stack_refresh_table'):
            self._stack_refresh_table()
        self._queue_redraw()

    def _on_stack_method(self, _checked=None):
        self._stack_method = METHOD_GLS if self._rb_gls.isChecked() else METHOD_WM
        is_v = (self._stack_method == METHOD_GLS)
        # The n-basis selector only affects WM + √MSWD.
        if hasattr(self, '_cb_wm_n'):
            self._cb_wm_n.setEnabled(not is_v)

        if is_v:
            # Save the current (WM + √MSWD) errMode so it is restored when switching back.
            self._wm_errMode_saved = self._global_cfg.get('errMode', 'sem_mswd_2s')
            # GLS + ML uses sem_mswd_2s here: each aliquot's plateau error is
            # scaled by √MSWD before the GLS weighting, and ω takes up only the
            # remaining scatter between aliquots (maximum likelihood).
            _forced = 'sem_mswd_2s'
            self._global_cfg['errMode'] = _forced
            self.cfg['errMode']         = _forced
            for _i in range(self._cb_errMode.count()):
                if self._cb_errMode.itemData(_i) == _forced:
                    self._cb_errMode.blockSignals(True)
                    self._cb_errMode.setCurrentIndex(_i)
                    self._cb_errMode.blockSignals(False)
                    break
        else:
            # Restore the errMode chosen for WM + √MSWD (the current one if
            # GLS + ML was never selected).
            _em = (getattr(self, '_wm_errMode_saved', None)
                   or self._global_cfg.get('errMode', 'sem_mswd_2s'))
            self._global_cfg['errMode'] = _em
            self.cfg['errMode']         = _em
            for _i in range(self._cb_errMode.count()):
                if self._cb_errMode.itemData(_i) == _em:
                    self._cb_errMode.blockSignals(True)
                    self._cb_errMode.setCurrentIndex(_i)
                    self._cb_errMode.blockSignals(False)
                    break

        for w in (self._cb_errMode, self._lbl_errMode):
            w.setEnabled(not is_v)
        _tip = ("Fixed to SEM·√MSWD×2 for GLS + ML.\n"
                "Each aliquot's within-plateau MSWD scales its GLS weight;\n"
                "ω then captures additional between-aliquot scatter by ML."
                if is_v else
                "WM + √MSWD: scales the SEM of the combined mean.\n"
                "sem_mswd applies √MSWD·t(n−1) when p < cutoff, else SEM×2;\n"
                "sem_mswd_2s applies √MSWD when the MSWD is above the\n"
                "Wendt & Carl 95% limit for n.")
        self._cb_errMode.setToolTip(_tip)

        # Recompute per-session plateau tiers with the new errMode, then redraw.
        if self._sessions:
            for sess in self._sessions:
                self._stack_recompute(sess)
            self._stack_refresh_table()
        self._queue_redraw()

    def _load_disk_flux(self):
        paths, _ = dfold.get_open_file_names(
            self, "Load disk flux (.jflux)", "", "Flux config (*.jflux);;All (*)",
            tool='flux_regression')
        if not paths:
            return
        loaded = 0
        for p in paths:
            # arar_math.read_jflux keeps the flux's own λ / FC monitor age (and
            # R where the file has it), so the hole is applied as an R value and
            # re-derived for whatever calibration the plateau is on.
            try:
                new = read_jflux(p)
            except Exception as e:
                QMessageBox.warning(self, "Load disk flux",
                                    f"{os.path.basename(p)}: {e}")
                continue
            self._disk_flux.update(new)
            loaded += len(new)
            _mons = {fx.get('monitor', 'FCs') for fx in new.values()}
            _mn = monitor_mismatch_note(self.cfg, _mons.pop() if len(_mons) == 1 else 'FCs',
                                        f'the flux in {os.path.basename(p)}')
            if _mn:
                QMessageBox.information(self, "Load disk flux", _mn)
        if loaded:
            self._lbl_flux_status.setText(
                f"{loaded} disk(s) loaded: "
                + ", ".join(f'{k[1]} ({k[0]})' for k in self._disk_flux)
                + ". Sessions auto-matched by disk_id / hole_id.")
            self._lbl_flux_status.setStyleSheet(
                "color:#2e7d32; font-size:9px;")
            self._sp_auto_match()
            self._refresh_sp_flux_state()
            if self.raw_steps:
                self._update_all()
            self._queue_redraw()

    def _open_stack_flux(self):
        """Open the shared disk-flux dialog (load / layout+rings / assign)
        for the stack, one window instead of a row of buttons."""
        FluxAssignDialog(self, _StackFluxHost(self)).exec_()

    def _recompute_all_sessions(self):
        self._flux_no_ratio = set()
        for sess in self._sessions:
            self._stack_recompute(sess)
        # Show the matched flux R / J in the group panels, so the panels show
        # the values the ages are computed with.
        self._refresh_flux_group_display()
        if hasattr(self, '_stack_refresh_table'):
            self._stack_refresh_table()
        self._queue_redraw()
        if self._flux_no_ratio:
            names = '\n  • '.join(sorted(self._flux_no_ratio))
            QMessageBox.warning(
                self, "Flux J not applied",
                "These sessions match a disk-flux hole but carry no ⁴⁰Ar/³⁹Ar "
                "ratio data, so their ages cannot be recomputed from the flux J "
                "and remain on their imported values:\n\n  • " + names +
                "\n\nRe-import them with the per-step ratio (r4039) column to let "
                "the flux J drive the ages.")
            self._flux_no_ratio = set()

    def _flux_auto_match(self, disk, hole, irr=None):
        """Core auto-match of a sample's (disk_id, hole_id, irradiation) to a
        loaded disk-flux hole.  Returns ((irr, disk_id), hole_idx) or None.

        Matching is by hole, mirroring the Age distribution tool: the sample's own
        disk-letter prefix is stripped from hole_id ('E17'->'17') and, as a
        fallback, the trailing digits are used, so a hole matches even when the
        sample's disk label ('E') differs from the flux's disk label ('E inner').
        The disk name and irradiation only disambiguate when several loaded holes
        share the same number, they never block a match. Shared by the stack,
        the single plateau, and the carousel so all behave alike."""
        return match_flux_hole(self._disk_flux, disk, hole, irr)

    def _flux_hole_matches(self, disk, hole):
        """Every loaded disk-flux hole a sample's hole_id could be, as
        [((irr, disk_id), hole_idx), ...], before any disambiguation."""
        return flux_hole_matches(self._disk_flux, disk, hole)

    def _flux_match_problems(self):
        """Stack aliquots a loaded disk flux is not matched to a hole, as
        (warn, info): warn = [(name, reason)] for ones that should match but
        don't (the hole is on several loaded disks, e.g. 17 on 'E inner' and
        'E outer'; the disk / irradiation is a loaded one but the hole isn't
        found; an assigned hole that is gone); info = {position: count} for
        aliquots from positions the loaded flux doesn't cover at all.
        Aliquots set to 'no hole' in Disk flux… are left out (a choice)."""
        warn, info = [], {}
        if not self._disk_flux:
            return warn, info
        disks = [str(k[1]).upper() for k in self._disk_flux]
        irrs = {str(k[0]).upper() for k in self._disk_flux}
        for si, sess in enumerate(self._sessions):
            if sess.get('excluded', False):
                continue
            ov = self._flux_override.get(si, 'auto')
            if ov == 'none' or self._flux_match_session(sess, si) is not None:
                continue
            name = sess.get('title') or f'#{si + 1}'
            cfg = sess.get('cfg', {})
            hole = cfg.get('hole_id')
            disk = str(cfg.get('disk_id') or '').strip()
            irr = str(cfg.get('irradiation') or cfg.get('irrad') or '').strip()
            m = (self._flux_hole_matches(cfg.get('disk_id'), hole)
                 if hole not in (None, '') else [])
            # the aliquot's own disk / irradiation is one the flux covers
            ours = (bool(irr) and irr.upper() in irrs) or (bool(disk) and any(
                d == disk.upper() or d.startswith(disk.upper() + ' ')
                or disk.upper().startswith(d) for d in disks))
            if ov != 'auto':
                warn.append((name, 'its assigned hole is not in the loaded flux'))
            elif len(m) > 1:
                warn.append((name, f"hole {hole} is on " + ' and '.join(
                    sorted({str(k[1]) for k, _ in m})) + ', pick one'))
            elif ours:
                warn.append((name, 'no hole id' if hole in (None, '')
                             else f'hole {hole} not found on disk {disk or "?"}'))
            else:
                pos = disk or str(hole or '') or 'no position'
                info[pos] = info.get(pos, 0) + 1
        return warn, info

    def _refresh_flux_status(self):
        """The stack's flux line: which disks are loaded and, in orange, every
        aliquot that should take a hole from them but doesn't, so an
        unassigned hole (the aliquot silently keeping its own R_std / J)
        cannot slip through."""
        lbl = getattr(self, '_lbl_flux_status', None)
        if lbl is None or not self._disk_flux:
            return
        text = (f"{len(self._disk_flux)} disk(s) loaded: "
                + ', '.join(f'{k[1]} ({k[0]})' for k in self._disk_flux) + '.')
        warn, info = self._flux_match_problems()
        if info:
            n = sum(info.values())
            text += (f" {n} aliquot(s) from positions not in the loaded flux ("
                     + ', '.join(sorted(info)) + ') keep their own J.')
        if not warn:
            if not info:
                text += ' Every aliquot is matched to a hole.'
            lbl.setText(text)
            lbl.setStyleSheet("color:#2e7d32; font-size:9px;")
            lbl.setToolTip('')
            return
        keep = ' They keep their own R_std / J.'
        shown = '; '.join(f'{n} ({w})' for n, w in warn[:3])
        more = f'; +{len(warn) - 3} more' if len(warn) > 3 else ''
        lbl.setText(text + f"\n\u26a0 {len(warn)} aliquot(s) not matched to a hole: "
                    + shown + more + '.' + keep + ' Assign them in Disk flux\u2026')
        lbl.setStyleSheet("color:#d35400; font-size:9px;")
        lbl.setToolTip('\n'.join(f'{n}: {w}' for n, w in warn))

    def _flux_match_session(self, sess, si=None):
        """Return ((irr,disk_id), hole_idx) for a session, or None.

        A hole picked in Disk flux… wins; otherwise auto-matches by
        disk_id + hole_id from the session cfg.

        ``si`` is the session's index in ``self._sessions``.  Pass it whenever
        it is known: ``_flux_override`` is keyed by index, and falling back to
        ``list.index(sess)`` resolves by dict equality, so two sessions with
        identical content (the same .arar loaded twice, a duplicated aliquot)
        both resolve to the first one's index and silently inherit its hole.
        """
        if not self._disk_flux:
            return None
        # manual override first (lets the flux be used even when the .arar file
        # carries no disk_id / hole_id).  Sentinels: 'none' = force no match;
        # absent / 'auto' = fall through to automatic matching below.
        if si is None:
            si = next((i for i, s in enumerate(self._sessions) if s is sess), -1)
        ov = self._flux_override.get(si, 'auto') if si >= 0 else 'auto'
        if ov == 'none':
            return None
        if ov != 'auto':
            irr, disk, hole = ov
            key = (str(irr), str(disk))
            fx = self._disk_flux.get(key)
            if fx is not None and str(hole) in fx['idx']:
                return key, fx['idx'][str(hole)]
            return None
        cfg = sess.get('cfg', {})
        return self._flux_auto_match(cfg.get('disk_id'), cfg.get('hole_id'),
                                     cfg.get('irradiation'))

    def _stack_systematics(self, used, ages):
        """(standard groups, n × k systematic components) for aliquot ages.

        Aliquots on the same calibration share every component of
        arar_math.systematic_components (for an anchored calibration: the
        R(FCs/A) measurement, the anchor's orbital age and λ) and are combined
        one component at a time; aliquots on different calibrations share none.
        """
        cfgs = [self._eff_cfg_for(si) for si in used]
        keys = [calibration_group_key(c) for c in cfgs]
        grp = [keys.index(k) for k in keys]
        C, _ = systematic_matrix(ages, cfgs)
        return grp, C

    def _build_stack_pcorr(self, si_list):
        """NxN J-flux correlation matrix over the given session indices.

        Priority: loaded-flux match -> the .jflux ρ between the holes; else the position
        identity (arar_math.position_ident: disk, hole, typed J group, then the
        J itself), same -> 1, different -> 0.  The rule itself is
        arar_math.j_correlation, shared with the Age distribution tool so both
        give the same covariance.
        """
        pts = []
        for si in si_list:
            sess = self._sessions[si]
            cfg = sess.get('cfg', {})
            pts.append(dict(
                match=self._flux_match_session(sess, si),
                ident=(position_ident(cfg.get('disk_id'), cfg.get('hole_id'),
                                      sess.get('j_grp'), self._eff_cfg_for(si))
                       or f'__sess_{si}'),
                irr=(cfg.get('irradiation')
                     or (cfg.get('constants') or {}).get('irradiation') or '')))
        return j_correlation(pts, self._disk_flux)

    # what a disk-flux hole encodes: arar_math.flux_hole_R / flux_hole_J
    _flux_hole_R = staticmethod(flux_hole_R)

    def _flux_hole_j(self, fx, hidx, cfg):
        """(J, σ_J) of a disk-flux hole re-derived for cfg's calibration
        (arar_math.flux_hole_J: the hole's R turned into J with cfg's ingrowth)."""
        return flux_hole_J(fx, hidx, cfg)

    def _flux_overlay(self, sess, cfg=None, si=None):
        """The disk-flux override for one session, or None.

        A session on a loaded hole (picked in the Hole column, or matched
        automatically) is aged from that hole's
        monitor ratio R and σ_R (R mode, R standard = hole R), over every
        cfg-/group-level R or J, also on Imported ages, as in the Age
        distribution tool.  This is the same overlay the single plateau and the carousel
        use (_cfg_with_flux).  A legacy flux without R gives the hole's J,
        re-derived for ``cfg``'s calibration.  A session on no loaded hole
        keeps its own R / J, and j_correlation makes it independent of the
        ones on the disk.  Returns a dict ready to overlay onto a cfg, or None.
        """
        if not self._disk_flux:
            return None
        if si is None:
            si = next((i for i, s in enumerate(self._sessions) if s is sess), -1)
        hole = self._stack_flux_hole(si)
        if hole is None:
            return None
        ref = dict(cfg if cfg is not None else self._global_cfg)
        ref.pop('jForce', None)
        eff = self._cfg_with_flux(ref, hole)
        if eff.get('jForce'):                      # legacy flux without R
            return dict(ageFluxMode='ratio_J', jVal=eff['jVal'],
                        jErrPct=eff['jErrPct'], jForce=True)
        if eff.get('ageFluxMode') != 'ratio_R_std' or not eff.get('rStd'):
            return None
        return dict(ageFluxMode='ratio_R_std', rStd=eff['rStd'],
                    rStdErrPct=eff['rStdErrPct'],
                    rStdErrMode=eff.get('rStdErrMode', 'pct'), jForce=False)

    def _stored_cfg_for(self, si):
        """Global cfg overlaid with session si's flux group (or, outside any
        group, its own flux keys): the settings the user chose, without the
        disk-flux hole overlay."""
        eff = dict(self._global_cfg)
        # the route is the aliquot's own: none from the window-wide settings
        for k in ('route', 'fcAnchor', 'calibClass', 'stdCov'):
            eff.pop(k, None)
        for g in self._flux_groups:
            if si in g['indices']:
                eff.update(g['flux_cfg'])
                break
        else:
            if 0 <= si < len(self._sessions):
                eff.update(self._extract_flux_cfg(self._sessions[si].get('cfg', {})))
        return eff

    def _eff_cfg_for(self, si):
        """Effective cfg for session si (global cfg overlaid with its flux group,
        then with the R of the disk-flux hole it is on, if any).

        The per-session recompute, the n = steps combination, the GLS + ML
        tiers and the Stats dialog all read their J through this method, so
        the disk flux is applied the same way in each.
        """
        eff = self._stored_cfg_for(si)
        if 0 <= si < len(self._sessions):
            ov = self._flux_overlay(self._sessions[si], eff, si)
            if ov:
                eff.update(ov)
        return eff

    def _update_flux_groups(self):
        groups_by_key = {}
        for i, sess in enumerate(self._sessions):
            if sess.get('excluded', False):
                continue          # not in the stack: no flux group, not listed
            k = self._flux_key(sess['cfg'])
            if k not in groups_by_key:
                groups_by_key[k] = {
                    'flux_cfg': self._extract_flux_cfg(sess['cfg']),
                    'indices': [], 'titles': [], 'panel': None,
                }
            groups_by_key[k]['indices'].append(i)
            groups_by_key[k]['titles'].append(sess.get('title', f'#{i+1}'))

        old_by_key = {self._flux_key(g['flux_cfg']): g['flux_cfg']
                      for g in self._flux_groups}
        self._flux_groups = []
        for k, g in groups_by_key.items():
            if k in old_by_key:
                g['flux_cfg'] = old_by_key[k]
            self._flux_groups.append(g)

        self._rebuild_flux_ui()
        self._refresh_flux_group_display()
        for sess in self._sessions:
            self._stack_recompute(sess)
        self._stack_refresh_table()
        self._queue_redraw()

    def _rebuild_flux_ui(self):
        while self._s_flux_groups_layout.count():
            item = self._s_flux_groups_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not self._flux_groups:
            lbl = QLabel("Load samples to configure flux settings.")
            lbl.setStyleSheet("color:#888; font-size:10px; font-style:italic;")
            self._s_flux_groups_layout.addWidget(lbl)
            return

        if len(self._flux_groups) == 1:
            n = len(self._flux_groups[0]['indices'])
            note = QLabel(f"{n} sample{'s' if n != 1 else ''} with identical flux")
            note.setStyleSheet(
                "color:#27ae60; font-size:10px; font-style:italic;")
            self._s_flux_groups_layout.addWidget(note)

        for gi, group in enumerate(self._flux_groups):
            panel = self._make_flux_group_frame(gi, group)
            self._flux_groups[gi]['panel'] = panel
            self._s_flux_groups_layout.addWidget(panel)
        self._s_flux_groups_layout.addStretch()

    def _make_flux_group_frame(self, gi, group):
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame{background:#f9f9f9;border:1px solid #ccc;border-radius:4px;}")
        fv = QVBoxLayout(frame)
        fv.setContentsMargins(8, 6, 8, 6); fv.setSpacing(4)

        titles = group['titles']
        if len(self._flux_groups) == 1:
            hdr_txt = "All samples"
        else:
            shown = ', '.join(titles[:3])
            if len(titles) > 3:
                shown += f' +{len(titles)-3} more'
            hdr_txt = f"Group {gi+1}: {shown}"
        hdr = QLabel(hdr_txt)
        hdr.setStyleSheet("font-weight:bold; font-size:10px;")
        hdr.setWordWrap(True)
        fv.addWidget(hdr)

        fb = QFormLayout(); fb.setSpacing(3)

        cb_mode = QComboBox()
        cb_mode.addItem("Imported ages",     'imported')
        cb_mode.addItem("R mode",            'ratio_R_std')
        cb_mode.addItem("R mode (insert J)", 'ratio_J')
        mode = group['flux_cfg'].get('ageFluxMode', 'imported')
        for _i in range(cb_mode.count()):
            if cb_mode.itemData(_i) == mode:
                cb_mode.setCurrentIndex(_i); break
        fb.addRow("Age source:", cb_mode)
        frame.cb_mode = cb_mode

        le_jVal = QLineEdit(f"{group['flux_cfg'].get('jVal', 1e-3):.8g}")
        lbl_jVal = QLabel("J value:")
        fb.addRow(lbl_jVal, le_jVal)
        frame.le_jVal = le_jVal; frame.lbl_jVal = lbl_jVal

        le_jErrPct = QLineEdit(f"{group['flux_cfg'].get('jErrPct', 0.0):.8g}")
        le_jErrPct.setPlaceholderText("% 1σ")
        lbl_jErrPct = QLabel("J error (% 1σ):")
        fb.addRow(lbl_jErrPct, le_jErrPct)
        frame.le_jErrPct = le_jErrPct; frame.lbl_jErrPct = lbl_jErrPct

        le_rStd = QLineEdit(f"{from_fcs_r(group['flux_cfg'].get('rStd', 70.0), group['flux_cfg']):.11g}")
        le_rStd.setPlaceholderText("e.g. 5.753")
        lbl_rStd = QLabel("⁴⁰Ar/³⁹Ar R standard:")
        fb.addRow(lbl_rStd, le_rStd)
        frame.le_rStd = le_rStd; frame.lbl_rStd = lbl_rStd

        le_rStdErr = QLineEdit(f"{group['flux_cfg'].get('rStdErrPct', 0.10):.8g}")
        cb_rErrMode = QComboBox()
        cb_rErrMode.addItem("% 1σ", 'pct')
        cb_rErrMode.addItem("abs 1σ", 'abs')
        cb_rErrMode.setMaximumWidth(72)
        for _i in range(cb_rErrMode.count()):
            if cb_rErrMode.itemData(_i) == group['flux_cfg'].get(
                    'rStdErrMode', 'pct'):
                cb_rErrMode.setCurrentIndex(_i); break
        rErr_w = QWidget()
        rErr_l = QHBoxLayout(rErr_w)
        rErr_l.setContentsMargins(0, 0, 0, 0); rErr_l.setSpacing(4)
        rErr_l.addWidget(le_rStdErr); rErr_l.addWidget(cb_rErrMode)
        lbl_rStdErr = QLabel("R std error:")
        fb.addRow(lbl_rStdErr, rErr_w)
        frame.le_rStdErr = le_rStdErr; frame.cb_rErrMode = cb_rErrMode
        frame.lbl_rStdErr = lbl_rStdErr; frame.rErr_w = rErr_w

        le_stdAge = QLineEdit(
            f"{group['flux_cfg'].get('stdAgeMa', STD_AGE_DEFAULT):.8g}")
        le_stdAge.setPlaceholderText("Ma")
        lbl_stdAge = QLabel("Monitor age (Ma):")
        fb.addRow(lbl_stdAge, le_stdAge)
        frame.le_stdAge = le_stdAge; frame.lbl_stdAge = lbl_stdAge

        le_stdAgeErr = QLineEdit(
            f"{group['flux_cfg'].get('stdAgeErrMa', STD_AGE_ERR):.8g}")
        le_stdAgeErr.setPlaceholderText("Ma 1σ")
        lbl_stdAgeErr = QLabel("Monitor age ±1σ (Ma):")
        fb.addRow(lbl_stdAgeErr, le_stdAgeErr)
        frame.le_stdAgeErr = le_stdAgeErr; frame.lbl_stdAgeErr = lbl_stdAgeErr

        # monitor-age error type of this group; follows the calibration preset
        frame.std_age_err_type = ('external' if group['flux_cfg'].get(
            'stdAgeErrType', 'internal') == 'external' else 'internal')

        le_lambda = QLineEdit(
            f"{group['flux_cfg'].get('lambdaYr', LAMBDA_DEFAULT):.8g}")
        lbl_lambda = QLabel("λ (⁴⁰K, yr⁻¹):")
        fb.addRow(lbl_lambda, le_lambda)
        frame.le_lambda = le_lambda; frame.lbl_lambda = lbl_lambda

        le_lambdaErr = QLineEdit(
            f"{group['flux_cfg'].get('lambdaErrYr', LAMBDA_ERR):.8g}")
        lbl_lambdaErr = QLabel("σλ (yr⁻¹):")
        fb.addRow(lbl_lambdaErr, le_lambdaErr)
        frame.le_lambdaErr = le_lambdaErr; frame.lbl_lambdaErr = lbl_lambdaErr

        # Status label, shown after a Calibration… conversion
        lbl_calib_status = QLabel("")
        lbl_calib_status.setStyleSheet(
            "color: #1a7a1a; font-size: 9px; font-style: italic; padding-top: 2px;")
        lbl_calib_status.setWordWrap(True)
        lbl_calib_status.setVisible(False)
        fv.addLayout(fb)
        fv.addWidget(lbl_calib_status)
        frame.lbl_calib_status = lbl_calib_status

        _upd = lambda _=None, _gi=gi: self._on_flux_group_cfg(_gi)
        cb_mode.currentIndexChanged.connect(_upd)

        # When rErrMode toggles, convert the displayed error value before saving
        def _on_rErrMode_changed(_idx):
            fc = self._flux_groups[gi].get('flux_cfg', {})
            old_mode = fc.get('rStdErrMode', 'pct')
            new_mode = frame.cb_rErrMode.currentData()
            if old_mode != new_mode:
                try:
                    v = float(frame.le_rStdErr.text())
                    r = float(frame.le_rStd.text() or '70.0')
                    if r > 0:
                        cv = (v / 100.0 * r) if new_mode == 'abs' else (v / r * 100.0)
                        frame.le_rStdErr.blockSignals(True)
                        frame.le_rStdErr.setText(f'{cv:.6g}')
                        frame.le_rStdErr.blockSignals(False)
                except (ValueError, ZeroDivisionError):
                    pass
            self._on_flux_group_cfg(gi)
        cb_rErrMode.currentIndexChanged.connect(_on_rErrMode_changed)
        for le in (le_jVal, le_jErrPct, le_rStd, le_rStdErr,
                   le_stdAge, le_stdAgeErr, le_lambda, le_lambdaErr):
            le.textChanged.connect(_upd)

        self._update_flux_group_visibility(frame)
        return frame

    def _update_flux_group_visibility(self, frame):
        mode     = frame.cb_mode.currentData()
        is_ratio = mode != 'imported'
        is_j     = (mode == 'ratio_J')
        for w in (frame.lbl_jVal, frame.le_jVal,
                  frame.lbl_jErrPct, frame.le_jErrPct):
            w.setVisible(is_ratio and is_j)
        for w in (frame.lbl_rStd, frame.le_rStd,
                  frame.lbl_rStdErr, frame.rErr_w,
                  frame.lbl_stdAge, frame.le_stdAge,
                  frame.lbl_stdAgeErr, frame.le_stdAgeErr):
            w.setVisible(is_ratio and not is_j)
        for w in (frame.lbl_lambda, frame.le_lambda,
                  frame.lbl_lambdaErr, frame.le_lambdaErr):
            w.setVisible(is_ratio)   # λ needed for all R_unk modes

    def _group_flux_overlays(self, g):
        """Disk-flux overlays of a group's sessions (None for a session on no
        loaded hole), each for that session's own calibration."""
        return [self._flux_overlay(self._sessions[si], self._stored_cfg_for(si), si)
                if 0 <= si < len(self._sessions) else None
                for si in g['indices']]

    def _refresh_flux_group_display(self):
        """Make each group panel show the R standard (or J) actually in force.

        A session on a loaded hole is aged from that hole's R, so when every
        session of a group is on one, the panel shows R mode with that R
        (locked, mode box disabled) rather than the stored cfg values, which
        would contradict the result.  Otherwise the editable cfg values are
        shown.
        """
        for g in self._flux_groups:
            panel = g.get('panel')
            if panel is None:
                continue
            ovs = self._group_flux_overlays(g)
            locked = len(ovs) > 0 and all(ov is not None for ov in ovs)
            widgets = (panel.le_jVal, panel.le_jErrPct, panel.le_rStd,
                       panel.le_rStdErr, panel.cb_rErrMode, panel.cb_mode)
            for w in widgets:
                w.blockSignals(True)
            fc = g['flux_cfg']
            mode = ovs[0]['ageFluxMode'] if locked else fc.get('ageFluxMode', 'imported')
            for _i in range(panel.cb_mode.count()):
                if panel.cb_mode.itemData(_i) == mode:
                    panel.cb_mode.setCurrentIndex(_i); break
            if locked:
                if mode == 'ratio_R_std':
                    vals = {(round(ov['rStd'], 12), round(ov['rStdErrPct'], 12)) for ov in ovs}
                    fields = (panel.le_rStd, panel.le_rStdErr)
                    if len(vals) == 1:
                        panel.le_rStd.setText(f"{from_fcs_r(ovs[0]['rStd'], fc):.11g}")
                        panel.le_rStdErr.setText(f"{ovs[0]['rStdErrPct']:.8g}")
                        for _i in range(panel.cb_rErrMode.count()):
                            if panel.cb_rErrMode.itemData(_i) == 'pct':
                                panel.cb_rErrMode.setCurrentIndex(_i); break
                        tip = "Locked to the R of the disk-flux hole these sessions are on."
                    else:
                        panel.le_rStd.setText("(per-hole flux)")
                        panel.le_rStdErr.setText("(per-hole flux)")
                        tip = "This group spans several flux holes, R is applied per session."
                else:                                   # legacy flux: hole J
                    vals = {round(ov['jVal'], 12) for ov in ovs}
                    fields = (panel.le_jVal, panel.le_jErrPct)
                    if len(vals) == 1:
                        panel.le_jVal.setText(f"{ovs[0]['jVal']:.8g}")
                        panel.le_jErrPct.setText(f"{ovs[0]['jErrPct']:.8g}")
                        tip = "Locked to the J of the disk-flux hole these sessions are on."
                    else:
                        panel.le_jVal.setText("(per-hole flux)")
                        tip = "This group spans several flux holes, J is applied per session."
                for le in fields:
                    le.setReadOnly(True); le.setToolTip(tip)
                    le.setStyleSheet("background:#eef7ee;")
                panel.cb_rErrMode.setEnabled(False)
                panel.cb_mode.setEnabled(False)
                panel.cb_mode.setToolTip(
                    "These sessions are on a loaded disk-flux hole and are aged "
                    "from its R.\nUnassign the hole in Disk flux… to choose another source.")
            else:
                panel.le_jVal.setText(f"{fc.get('jVal', 1e-3):.8g}")
                panel.le_jErrPct.setText(f"{fc.get('jErrPct', 0.0):.8g}")
                panel.le_rStd.setText(f"{from_fcs_r(fc.get('rStd', 70.0), fc):.11g}")
                panel.le_rStdErr.setText(f"{fc.get('rStdErrPct', 0.10):.8g}")
                for _i in range(panel.cb_rErrMode.count()):
                    if panel.cb_rErrMode.itemData(_i) == fc.get('rStdErrMode', 'pct'):
                        panel.cb_rErrMode.setCurrentIndex(_i); break
                for le in (panel.le_jVal, panel.le_jErrPct,
                           panel.le_rStd, panel.le_rStdErr):
                    le.setReadOnly(False); le.setToolTip(""); le.setStyleSheet("")
                panel.cb_rErrMode.setEnabled(True)
                panel.cb_mode.setEnabled(True)
                panel.cb_mode.setToolTip("")
            for w in widgets:
                w.blockSignals(False)
            self._update_flux_group_visibility(panel)

    def _on_flux_group_cfg(self, gi):
        if gi >= len(self._flux_groups):
            return
        group = self._flux_groups[gi]
        panel = group.get('panel')
        if panel is None:
            return
        def _f(le, default=0.0):
            try:    return float(le.text())
            except (ValueError, TypeError): return default
        # While all the group's sessions are on a loaded hole, the mode, R and
        # J fields show that hole's values (see _refresh_flux_group_display)
        # and must not be read back from the display-only panel: that would
        # overwrite the stored cfg with the flux values and lose them when the
        # hole is unassigned. Keep the stored values.
        ovs = self._group_flux_overlays(group)
        flux_locked = len(ovs) > 0 and all(ov is not None for ov in ovs)
        fc = group['flux_cfg']
        _manual_mode = panel.cb_mode.currentData() if not flux_locked else fc.get('ageFluxMode', 'imported')
        group['flux_cfg'].update(dict(
            ageFluxMode   = (fc.get('ageFluxMode', 'imported') if flux_locked
                             else _manual_mode),
            jVal          = (fc.get('jVal', 1e-3) if flux_locked
                             else _f(panel.le_jVal, 1e-3)),
            jErrPct       = (fc.get('jErrPct', 0.0) if flux_locked
                             else _f(panel.le_jErrPct, 0.0)),
            # User explicitly typed a J → override per-step j_val values baked
            # into the session file so the group J actually takes effect.
            jForce        = (bool(fc.get('jForce')) if flux_locked
                             else _manual_mode == 'ratio_J'),
            rStd          = (fc.get('rStd', 70.0) if flux_locked
                             else to_fcs_r(_f(panel.le_rStd, 70.0), fc)),
            rStdErrPct    = (fc.get('rStdErrPct', 0.10) if flux_locked
                             else _f(panel.le_rStdErr, 0.10)),
            rStdErrMode   = (fc.get('rStdErrMode', 'pct') if flux_locked
                             else panel.cb_rErrMode.currentData()),
            stdAgeMa      = _f(panel.le_stdAge,     STD_AGE_DEFAULT),
            stdAgeErrMa   = _f(panel.le_stdAgeErr,  STD_AGE_ERR),
            stdAgeErrType = panel.std_age_err_type,
            lambdaYr      = _f(panel.le_lambda,     LAMBDA_DEFAULT),
            lambdaErrYr   = _f(panel.le_lambdaErr,  LAMBDA_ERR),
        ))
        sync_route(group['flux_cfg'])
        # The groups are rebuilt from each session's own cfg on every regroup
        # (flux dialog Apply, adding or removing aliquots), so an edit that lives
        # only in group['flux_cfg'] is thrown away and the panel falls back to
        # whatever the file was loaded with (e.g. converter files: Imported ages).
        # Write the edit through to the member sessions so it survives.
        fc = group['flux_cfg']
        for si in group['indices']:
            if 0 <= si < len(self._sessions):
                sess = self._sessions[si]
                sess.setdefault('cfg', {}).update(
                    {k: fc[k] for k in self._FLUX_KEYS if k in fc})
                # Un-assigning a hole later restores this snapshot; it should
                # restore the age source chosen here, not the file's original.
                base = sess.get('flux_base_cfg')
                if base is not None and not flux_locked:
                    base['ageFluxMode'] = fc.get('ageFluxMode')
        self._update_flux_group_visibility(panel)
        for si in group['indices']:
            if si < len(self._sessions):
                self._stack_recompute(self._sessions[si])
        self._stack_refresh_table()
        self._queue_redraw()
        # Edited to the same flux as another group: merge them.
        _k = self._flux_key(group['flux_cfg'])
        if any(self._flux_key(g['flux_cfg']) == _k
               for g in self._flux_groups if g is not group):
            QTimer.singleShot(0, self._update_flux_groups)

    # ── Recompute / refresh ───────────────────────────────────────────────────
    def _stack_recompute(self, sess):
        si = self._sessions.index(sess) if sess in self._sessions else -1
        if si >= 0:
            eff_cfg = self._eff_cfg_for(si)
        else:
            eff_cfg = dict(self._global_cfg)
            eff_cfg.update(self._extract_flux_cfg(sess.get('cfg', {})))
        # When this session is on a loaded hole, the hole's R is already overlaid
        # by _eff_cfg_for. The ages can only follow it
        # if the session carries raw ⁴⁰Ar/³⁹Ar ratios to re-age from. If it does
        # not (e.g. an age-only summary export), flag it so the user is warned
        # rather than left looking at an unchanged age.
        if self._flux_overlay(sess, None, si) is not None:
            has_ratio = any(s.get('r4039', 0.0) > 0 for s in sess.get('raw_steps', []))
            if not has_ratio:
                self._flux_no_ratio.add(sess.get('title', f'#{si+1}'))
        steps = _compute_steps_with_age_mode(sess['raw_steps'], eff_cfg)
        sess['steps_c'] = add_cumulative(steps)
        sess['plateau'] = find_plateau(sess['steps_c'], eff_cfg)
        sess['pset']    = (set(sess['plateau']['ois'])
                           if sess['plateau'] else set())

    def _stack_refresh_table(self):
        self.sample_tree.blockSignals(True)
        display_1sigma, sig_scale, sig_lbl = self._disp()
        _age_dec = self._global_cfg.get('ageDec', 2)
        _af = f'.{_age_dec}f'
        self.sample_tree.setHeaderLabels(
            ['#', 'Sample / Label', f'Age ±{sig_lbl} (Ma)', '✕', 'J grp'])
        expanded = {
            self.sample_tree.topLevelItem(i).data(0, Qt.UserRole)
            for i in range(self.sample_tree.topLevelItemCount())
            if self.sample_tree.topLevelItem(i).isExpanded()
        }
        self.sample_tree.clear()
        for i, sess in enumerate(self._sessions):
            color = SAMPLE_COLORS[i % len(SAMPLE_COLORS)]
            p = sess.get('plateau')
            age_txt = (f"{p['r']['mean']:{_af}} ± {p['e1s_int'] * sig_scale:{_af}}"
                       + (' [m]' if p.get('forced') else '')) if p else '—'
            # J grp: priority: disk_id (from irrad label) → hole_id → j_grp
            # → R_std value → J value → blank
            sess_cfg = sess.get('cfg', {})
            r_std    = sess_cfg.get('rStd', 0.0)
            j_val    = sess_cfg.get('jVal', 0.0)
            j_grp = (sess_cfg.get('disk_id')
                     or sess_cfg.get('hole_id')
                     or sess.get('j_grp')
                     or (f'R={round(r_std, 4)}' if r_std > 0 else '')
                     or (f'J={round(j_val, 6)}' if j_val > 0 else '')
                     or '')
            is_excl = sess.get('excluded', False)
            top = QTreeWidgetItem(
                ['', sess.get('title', f'#{i+1}'), age_txt, '', j_grp])
            top.setData(0, Qt.UserRole, i)
            top.setText(0, str(i + 1))
            # Col 3: checkable exclusion box, checked = excluded from combined calc
            top.setCheckState(3, Qt.Checked if is_excl else Qt.Unchecked)
            top.setToolTip(3, "Check to exclude this aliquot from the combined plateau age")
            # Greyed-out foreground when excluded
            _fg_color = '#999999' if is_excl else color
            top.setForeground(0, QColor(_fg_color))
            top.setForeground(1, QColor(_fg_color))
            # Column 1 (name) and column 4 (J grp) are editable; col 3 checkable; others not
            top.setFlags(
                (top.flags() | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
                & ~Qt.ItemIsDropEnabled)
            for j, s in enumerate(sess.get('steps_c', [])):
                status  = s.get('status', 'normal')
                in_plat = j in sess.get('pset', set())
                sym = ({'excluded': '✕', 'forced': '◆'}.get(status,
                        '●' if in_plat else '○'))
                # Fold age ± error into col 2; col 3 left blank for steps
                child = QTreeWidgetItem([
                    sym, s['label'],
                    f"{s['age']:{_af}} ±{s['err1s'] * sig_scale:{_af}}", '', '',
                ])
                child.setData(0, Qt.UserRole, (i, j))
                child.setFlags(
                    child.flags() & ~(Qt.ItemIsEditable | Qt.ItemIsDropEnabled
                                      | Qt.ItemIsUserCheckable))
                fc = ('#c0392b' if status == 'excluded'
                      else '#1565c0' if status == 'forced'
                      else color if in_plat else '#555555')
                for c in range(5):
                    child.setForeground(c, QColor(fc))
                top.addChild(child)
            self.sample_tree.addTopLevelItem(top)
            if i in expanded:
                top.setExpanded(True)
        self.sample_tree.blockSignals(False)

    def _wm_session_j(self, si, steps_si):
        """Resolve (eff_cfg, J, relative-σ_J) for one session in the n = steps
        stack: the J its step ages were computed with (_aged_J: the R
        standard's J, an applied disk-flux hole's J, the steps' own J in J
        mode, or on imported ages the J and σ_J stored with the steps) and the
        same σ_J/J its plateau's internal tier carries.  Without any J the
        cfg's jVal is used, for the normalisation only.
        """
        eff = self._eff_cfg_for(si)
        Jv, rel = _aged_J(eff, steps_si)
        if not Jv:
            jv = [float(s['j_val']) for s in steps_si if s.get('j_val', 0) > 0]
            Jv = float(np.median(jv)) if jv else float(eff.get('jVal', 0) or 0)
            rel = float(cfg_j_err_frac(eff))
        return eff, Jv, rel

    # ── Combined plateau ──────────────────────────────────────────────────────
    def _combined_plateau(self):
        """
        Combine the plateaus of the stack's sessions into one age.

        n = aliquots, and GLS + ML: each plateau is collapsed to its age with
        its three tiers, and the ages are combined by arar_math.combine (the
        weighted mean with analytical weights and √MSWD, or GLS + ML), with the
        J correlation between aliquots from the shared ρ rule.

        n = steps: ratio-space weighted mean (Koppers 2002, Eq. 9 + Section 3.4)
        --------------------------------------------------------------------
        When combining plateau ages from multiple experiments with different J
        values, ArArCALC does not take a weighted mean of individual step ages.
        Instead (Koppers 2002, p. 615):

          1. Normalize all steps to one common J value (J1: the first session's J):
               F_norm = r4039_i * (Ji / J1)
             Error propagation (Koppers Eqs. 1-2):
               s2_Fnorm = (Ji/J1)^2 * [s2_r + r^2 * (s2_J1/J1^2 + s2_Ji/Ji^2)]
             (here σ(F_norm) keeps the ratio error only; the J errors are
             carried in step 4, see below)

          2. Weighted mean of all normalized ratios (Eq. 9):
               F_bar = sum(Ei*Fi) / sum(Ei)    with Ei = 1/s2_Fnorm
               SEM   = sqrt(1 / sum(Ei))

          3. Convert mean ratio to age (Eq. 17):
               t = ln(1 + J1 * F_bar) / lambda

          4. Propagate the J uncertainty (Eq. 18):
               s2_T = (J1/(C*lam))^2 * s2_F + (F_bar/(C*lam))^2 * s2_J1
                     + (ln(C)/lam^2)^2 * s2_lam
             where C = 1 + J1*F_bar

        Eq. 18 carries J1's error alone, which is exact when every step shares
        one J.  For aliquots on different J's this tool instead carries each
        aliquot's own J error, once per aliquot (all its steps share it),
        weighted by the aliquot's share of F_bar and correlated between
        aliquots by the same hole / disk ρ as the n = aliquots path.  That
        reduces to Eq. 18 for a single J and does not depend on the choice
        of J1.

        Fallback (no raw ratio data):
        If r4039 is not available, falls back to age-space weighted mean with
        disk-group-aware J propagation.
        """
        by_si = {}
        for si, sess in enumerate(self._sessions):
            if sess.get('excluded', False):
                continue
            for i, s in enumerate(sess.get('steps_c', [])):
                if i in sess.get('pset', set()):
                    by_si.setdefault(si, []).append(s)
        if not by_si:
            return None, None, None, None

        si_list = sorted(by_si.keys())

        # ── n = ALIQUOTS: GLS + ML, or WM + √MSWD ─────────────────────────────
        # Collapse each plateau to its age + 1σ tiers (ana/int/ext) and combine
        # the aliquot ages with the shared engine (arar_math.combine), the same
        # engine and inputs as the Age distribution tool's pool(), so loading
        # these aliquots there gives the same age.  The J error is shared through
        # the hole / disk correlation ρ (p_corr), the calibration systematic one
        # component at a time within a calibration.
        # The per-aliquot analytical tier is already errMode-scaled: within-
        # plateau scatter (MSWD > 1 is common: grain heterogeneity, recoil) stays
        # with its aliquot, so the GLS ω or the √MSWD here only answer for the
        # scatter between aliquots.
        if self._stack_method == METHOD_GLS or self._wm_n_basis == 'aliquots':
            ages, s_ana, s_int, s_ext, used = [], [], [], [], []
            for si in si_list:
                p = self._sessions[si].get('plateau')
                if not p:
                    continue
                e_ana = p.get('e1s_ana')
                if e_ana is None or e_ana <= 0:
                    continue
                ages.append(float(p['r']['mean']))
                s_ana.append(e_ana)
                s_int.append(p.get('e1s_int', e_ana))
                s_ext.append(p.get('e1s_ext', p.get('e1s_int', e_ana)))
                used.append(si)
            if len(ages) >= 1:
                rho = self._build_stack_pcorr(used)
                std_grp, sysC = self._stack_systematics(used, ages)
                is_v = self._stack_method == METHOD_GLS
                res = combine(self._stack_method,
                              ages, s_ana, s_int, s_ext,
                              standard_group=std_grp, p_corr=rho,
                              od_mode='threshold', sys_components=sysC)
                n_match = sum(1 for si in used
                              if self._flux_match_session(self._sessions[si], si))
                r_dict = dict(
                    mean=res['wm'], sem=res['sigma_ana'], mswd=res['mswd'],
                    prob=res['prob'], n=res['n'],
                    sem_raw=res['sigma_ana'] / res['mswd_scale'],
                    n_flux_matched=n_match, overdispersed=res['overdispersed'],
                    mswd_scale=res['mswd_scale'])
                if is_v:
                    od_txt = (f", overdispersion ±{res['overdispersion']*2:.4f} Ma (2σ)"
                              if res['overdispersed'] else "")
                    r_dict.update(
                        method='GLS + ML (n = aliquots' + od_txt + ')',
                        overdispersion=res['overdispersion'])
                else:
                    r_dict['method'] = 'n = aliquots'
                return (r_dict, res['sigma_ana'],
                        res['sigma_int'], res['sigma_ext'])
            # nothing usable → fall through to the step-ratio path

        # ── Check if all sessions have raw ratio data ─────────────────────────
        all_have_ratios = all(
            all(s.get('r4039', 0) > 0 and s.get('err_r4039', 0) > 0
                for s in steps)
            for steps in by_si.values()
        )

        # ── Select J1: the session with the smallest relative J error ─────────
        # J1 is only the normalisation reference: the age (J1·F_bar = Σ ŵ·R·J_i)
        # and, with each aliquot's own J error carried below, its uncertainty
        # are the same whichever aliquot is J1.
        best_si    = si_list[0]
        best_j_rel = float('inf')
        sess_J     = {}   # si -> (eff_cfg, J, relative σ_J)
        for si in si_list:
            eff, Jv, rel = self._wm_session_j(si, by_si[si])
            sess_J[si] = (eff, Jv, rel)
            if rel < best_j_rel:
                best_j_rel = rel
                best_si    = si

        eff_cfg1 = sess_J[best_si][0]
        J1       = sess_J[best_si][1]
        lam     = eff_cfg1.get('lambdaYr', LAMBDA_DEFAULT)
        # Ratios on different J's can only be put on one J when the λ and
        # monitor age are shared; a stack mixing calibrations goes age-space.
        one_calib = len({calibration_group_key(sess_J[si][0]) for si in si_list}) == 1

        if all_have_ratios and J1 > 0 and one_calib:
            # ── Ratio-space weighted mean (n = steps) ───────────────────────────
            F_list  = []   # normalized ratios
            sF_list = []   # 1σ on normalized ratios (analytical)
            F_si    = []   # aliquot (session) of each normalized ratio

            for si in si_list:
                _eff_si, Ji, _rel_si = sess_J[si]
                steps = by_si[si]
                for s in steps:
                    F_si.append(si)
                    r_val  = float(s['r4039'])
                    sr     = float(s['err_r4039'])   # already 1σ

                    # Normalize: F_norm = r * (Ji/J1)
                    norm   = Ji / J1 if J1 > 0 else 1.0
                    F_norm = r_val * norm

                    # σ(F_norm_i) is the ratio measurement error only: the J
                    # errors do not enter the weights or the analytical tier;
                    # they are carried once per aliquot in the internal tier
                    # below.
                    sF     = norm * sr   # ratio precision only

                    F_list.append(F_norm)
                    sF_list.append(max(sF, 1e-30))

            F_arr  = np.array(F_list)
            sF_arr = np.array(sF_list)
            n      = len(F_arr)

            if n == 0:
                return None, None, None, None

            # Eq. 9: weighted mean of normalized ratios
            E       = 1.0 / sF_arr**2
            F_bar   = float(np.sum(E * F_arr) / np.sum(E))
            sF_bar  = float(np.sqrt(1.0 / np.sum(E)))    # analytical SEM (1σ)

            # MSWD on ratios (Eq. 14)
            chi2_v  = float(np.sum(E * (F_arr - F_bar)**2))
            mswd    = chi2_v / (n - 1) if n > 1 else 0.0
            prob    = float(1.0 - chi2_cdf(chi2_v, n - 1)) if n > 1 else 1.0

            # Error mode scaling, the rule find_plateau uses
            sF_scaled = scale_sem(sF_bar, mswd, prob, n,
                                  self._global_cfg.get('errMode', 'sem_mswd_2s'),
                                  self._global_cfg.get('pCut', 5.0))

            # Eq. 17: mean ratio → age
            C      = 1.0 + J1 * F_bar
            age_yr = math.log(C) / lam
            age_ma = age_yr / 1e6

            # Eq. 18: error propagation
            dT_dF = J1 / (C * lam)

            # Analytical 1σ: ratio-SEM only, scaled
            sT_ana_ma = abs(dT_dF) * sF_scaled / 1e6

            # J uncertainty, each aliquot's own, carried once per aliquot.
            # J1·F_bar = Σ_k ŵ_k·R_k·J_i(k), so ∂(J1·F_bar)/∂ln J_i is aliquot
            # i's share of it, g_i = J1·Σ_{k∈i} ŵ_k·F_norm,k.  Every step of one
            # aliquot shares that aliquot's J error, so it enters once through
            # g_i (adding σ_J to each step instead would shrink it by
            # √n_steps); aliquots on one hole / disk are correlated through the
            # same ρ as the n = aliquots path.  σ²_J(t) = gᵀ·diag(δ)·ρ·diag(δ)·g
            # / (λC)², δ_i = σ_Ji/J_i.  With every aliquot on one J this is
            # exactly Eq. 18's (∂T/∂J1·σ_J1)², the Koppers (2002) value, and
            # unlike σ_J1 alone it does not depend on which aliquot is J1.
            wF    = (E / np.sum(E)) * F_arr
            F_si  = np.asarray(F_si)
            gJ    = np.array([J1 * float(np.sum(wF[F_si == si])) * sess_J[si][2]
                              for si in si_list])
            rho_J = self._build_stack_pcorr(si_list)
            var_J_yr = float(gJ @ rho_J @ gJ) / (C * lam)**2

            # Internal 1σ: scaled ratio SEM ⊕ the J term (both 1σ).  The
            # analytical error is the first term of Eq. 18 and is the one
            # multiplied by √MSWD, so the scaled SEM (sF_scaled) enters the
            # internal and external tiers; the J (and λ) terms are not
            # MSWD-scaled.  The unscaled sF_bar here could give internal <
            # analytical.
            sT_int_ma = math.sqrt((dT_dF * sF_scaled)**2 + var_J_yr) / 1e6

            # External 1σ: internal ⊕ the calibration systematic
            # (systematic_components).  Mirrors _compute_tiers.
            sT_ext_ma = float(external_sigma_array(np.array([age_ma]),
                                                   np.array([sT_int_ma]),
                                                   eff_cfg1)[0])

            r_dict = dict(
                mean   = age_ma,
                sem    = sT_ana_ma,
                sem_raw = abs(dT_dF) * sF_bar / 1e6,
                mswd   = mswd,
                prob   = prob,
                n      = n,
                F_bar  = F_bar,
                sF_bar = sF_bar,
                J1     = J1,
                method = 'wm_ratio',
            )
            return r_dict, float(sT_ana_ma), float(sT_int_ma), float(sT_ext_ma)

        # ── FALLBACK: age-space weighted mean of the plateau steps ────────────
        # Same structure as the ratio path: each aliquot's J error enters once,
        # through its share of the mean, correlated by the stack's hole / disk ρ;
        # the calibration systematic one component at a time per calibration.
        si_w, all_steps = [], []
        for si in si_list:
            for s in by_si[si]:
                if s['err1s'] > 0:
                    si_w.append(si)
                    all_steps.append(s)
        r = weighted_mean([s['age']   for s in all_steps],
                          [s['err1s'] for s in all_steps])
        if r is None:
            return None, None, None, None
        e1s_ana = scale_sem(r['sem'], r['mswd'], r['prob'], r['n'],
                            self._global_cfg.get('errMode', 'sem_mswd_2s'),
                            self._global_cfg.get('pCut', 5.0))
        w = np.array([1.0 / s['err1s']**2 for s in all_steps])
        w /= w.sum()
        si_w = np.asarray(si_w)
        effs = {si: sess_J[si][0] for si in si_list}
        g = np.array([sum(w[k] * dt_dlnR(all_steps[k]['age'],
                                         effs[si].get('lambdaYr', LAMBDA_DEFAULT))
                          for k in np.where(si_w == si)[0]) * sess_J[si][2]
                      for si in si_list])
        var_j = float(g @ self._build_stack_pcorr(si_list) @ g)
        e1s_int = math.sqrt(e1s_ana**2 + var_j)
        cfgs_f = [effs[si] for si in si_w]
        C, _ = systematic_matrix([s['age'] for s in all_steps], cfgs_f)
        keys = [calibration_group_key(c) for c in cfgs_f]
        var_sys = 0.0
        for kk in set(keys):
            m = np.array([k == kk for k in keys])
            var_sys += float(np.sum((w[m] @ C[m])**2))
        e1s_ext = math.sqrt(e1s_int**2 + var_sys)

        r['method'] = 'age_space_fallback'
        return r, float(e1s_ana), float(e1s_int), float(e1s_ext)

    # ── Tree interactions ─────────────────────────────────────────────────────

    def _on_tree_name_edited(self, item, column):
        idx = item.data(0, Qt.UserRole)
        if not isinstance(idx, int) or idx < 0 or idx >= len(self._sessions):
            return
        if column == 1:
            self._sessions[idx]['title'] = item.text(1)
        elif column == 3:
            # Exclusion checkbox toggled for an aliquot (top-level) row
            excl = (item.checkState(3) == Qt.Checked)
            self._sessions[idx]['excluded'] = excl
            self._queue_redraw()
            QTimer.singleShot(0, self._update_flux_groups)
        elif column == 4:
            # J grp edited, write back into session cfg disk_id
            new_grp = item.text(4).strip()
            self._sessions[idx].setdefault('cfg', {})['disk_id'] = new_grp
            self._sessions[idx]['j_grp'] = new_grp
            # Trigger combined plateau recalculation
            self._queue_redraw()

    def _on_tree_item_clicked(self, item, _column):
        # A single click only selects (for Remove / ↑ / ↓); steps are
        # excluded or reset by double-click (_stack_toggle_step).
        pass

    def _on_overlay_step_clicked(self, si, ti):
        # Single click on the plot does nothing in stack mode (double-click
        # toggles the step), so a stray click cannot change a plateau.
        pass

    def _on_tree_item_double_clicked(self, item, _column):
        """Double-click on a step row of the stack list: exclude / reset it.
        Aliquot rows keep their own double-click (rename)."""
        d = item.data(0, Qt.UserRole)
        if isinstance(d, tuple) and len(d) == 2:
            # deferred: the toggle rebuilds the tree, which would delete the
            # item while Qt is still delivering its signal
            QTimer.singleShot(0, lambda si=d[0], ti=d[1]: self._stack_toggle_step(si, ti))

    def _stack_toggle_step(self, si, ti):
        """Exclude step ti of stack session si (red), or reset an excluded or
        forced step to normal, then recompute that aliquot's plateau and the
        combined age."""
        if not (0 <= si < len(self._sessions)):
            return
        sess = self._sessions[si]
        steps = sess.get('raw_steps', [])
        if not (0 <= ti < len(steps)):
            return
        st = steps[ti]
        was = st.get('status', 'normal')
        st['status'] = 'excluded' if was == 'normal' else 'normal'
        # keep the tree's expanded state on the aliquot that was edited
        self._stack_recompute(sess)
        self._stack_refresh_table()
        top = self.sample_tree.topLevelItem(si)
        if top is not None:
            top.setExpanded(True)
        self._queue_redraw()

    def _reindex_flux_override(self, old_sessions):
        """Carry the per-index hole assignments along after the session list
        was reordered or shortened, so a hole stays with its aliquot."""
        pos = {id(s): i for i, s in enumerate(old_sessions)}
        old = self._flux_override
        self._flux_override = {ni: old[pos[id(s)]]
                               for ni, s in enumerate(self._sessions)
                               if pos.get(id(s)) in old}

    def _stack_remove_selected(self):
        indices = sorted(
            {item.data(0, Qt.UserRole)
             for item in self.sample_tree.selectedItems()
             if isinstance(item.data(0, Qt.UserRole), int)},
            reverse=True)
        before = list(self._sessions)
        for idx in indices:
            self._sessions.pop(idx)
        self._reindex_flux_override(before)
        self._update_flux_groups()

    def _stack_move_up(self):
        indices = sorted(
            {item.data(0, Qt.UserRole)
             for item in self.sample_tree.selectedItems()
             if isinstance(item.data(0, Qt.UserRole), int)})
        before = list(self._sessions)
        for idx in indices:
            if idx > 0:
                self._sessions[idx], self._sessions[idx - 1] = \
                    self._sessions[idx - 1], self._sessions[idx]
        self._reindex_flux_override(before)
        self._update_flux_groups()

    def _stack_move_down(self):
        indices = sorted(
            {item.data(0, Qt.UserRole)
             for item in self.sample_tree.selectedItems()
             if isinstance(item.data(0, Qt.UserRole), int)},
            reverse=True)
        before = list(self._sessions)
        for idx in indices:
            if idx < len(self._sessions) - 1:
                self._sessions[idx], self._sessions[idx + 1] = \
                    self._sessions[idx + 1], self._sessions[idx]
        self._reindex_flux_override(before)
        self._update_flux_groups()

    # ── Add session ───────────────────────────────────────────────────────────
    def _stack_add_session(self, sess):
        self._sessions.append(sess)
        self._update_flux_groups()

    # ── Paste+ ────────────────────────────────────────────────────────────────
    def _stack_paste_add(self):
        text = QApplication.clipboard().text().strip()
        if not text:
            QMessageBox.warning(self, "Clipboard Empty",
                                "Nothing in clipboard.")
            return
        import io
        try:
            df = pd.read_csv(io.StringIO(text), sep='\t',
                             header=None, dtype=str)
            try:
                float(str(df.iloc[0, 1]))
                has_header = False
            except (ValueError, IndexError):
                has_header = True
            if has_header:
                df.columns = df.iloc[0].astype(str).str.strip()
                df = df.iloc[1:].reset_index(drop=True)

            headers  = list(df.columns)
            detected = detect_columns(headers)
            def _stk_col_median(col_key):
                if col_key not in detected:
                    return None
                ci, vlist = detected[col_key], []
                for _, _row_stk in df.iterrows():
                    try:
                        v = float(str(_row_stk.iloc[ci]).strip())
                        if v > 0:
                            vlist.append(v)
                    except Exception:
                        pass
                return float(np.median(vlist)) if vlist else None
            _j_default_stk     = _stk_col_median('j_val')
            _j_err_default_stk = _stk_col_median('j_err')
            default_sigma = 1 if self._cb_1sigma.isChecked() else 2
            dlg = ColumnMappingDialog(headers, detected, self, default_sigma=default_sigma,
                                      j_default=_j_default_stk,
                                      j_err_default=_j_err_default_stk)
            if dlg.exec_() != QDialog.Accepted:
                return
            mapping = dlg.get_mapping()

            has_ratio = 'r4039' in mapping
            if 'pct39ar' not in mapping:
                QMessageBox.critical(self, "Missing Column",
                                     "Column %³⁹Ar (cumulative) is required.")
                return

            steps = []
            for _, row in df.iterrows():
                try:
                    def _get(key, default=0.0, _row=row, _mapping=mapping):
                        if key not in _mapping:
                            return default
                        try:
                            val = _row.iloc[_mapping[key]]
                            if isinstance(val, str):
                                val = val.strip().rstrip('%').replace(',', '.')
                            return float(val)
                        except (ValueError, TypeError):
                            return default

                    s = dict(
                        label     = str(row.iloc[mapping['label']]).strip()
                                    if 'label' in mapping
                                    else _alpha_label(len(steps)),
                        power     = _get('power', float(len(steps) + 1)),
                        age       = _get('age'),
                        err1s     = _get('err'),
                        pct39Ar   = _get('pct39ar'),
                        caK       = _get('cak'),
                        pct40     = _get('pct40ar'),
                        r4039     = _get('r4039'),
                        err_r4039 = _get('err_r4039'),
                        err_cak   = _get('err_cak'),
                        j_val     = _get('j_val'),
                        j_err     = _get('j_err'),
                        iso_nx  = _get('iso_nx',  None),
                        iso_nsx = _get('iso_nsx', None),
                        iso_ny  = _get('iso_ny',  None),
                        iso_nsy = _get('iso_nsy', None),
                        iso_nri = _get('iso_nri', 0.0),
                        iso_ix  = _get('iso_ix',  None),
                        iso_isx = _get('iso_isx', None),
                        iso_iy  = _get('iso_iy',  None),
                        iso_isy = _get('iso_isy', None),
                        iso_iri = _get('iso_iri', 0.0),
                        status    = 'normal',
                    )
                    for _ik in ('iso_nx','iso_nsx','iso_ny','iso_nsy',
                                'iso_ix','iso_isx','iso_iy','iso_isy'):
                        if _ik not in mapping:
                            s[_ik] = None
                    if s['pct39Ar'] > 0 and (s['age'] > 0 or s['r4039'] > 0):
                        steps.append(s)
                except (ValueError, TypeError):
                    continue

            if dlg.auto_label:
                for i, s in enumerate(steps):
                    s['label'] = _alpha_label(i)
            if dlg.invert_cak:
                for s in steps:
                    kca = s.get('caK', 0.0)
                    if kca > 0:
                        s['caK'] = 1.0 / kca
                        ekca = s.get('err_cak', 0.0)
                        if ekca > 0:
                            s['err_cak'] = ekca / (kca * kca)
            if not steps:
                QMessageBox.warning(self, "No Valid Data",
                                    "No valid rows found in the pasted data.")
                return
            if dlg.err_sigma == 2:
                # Pasted values are 2σ; memory keeps everything at 1σ.
                for s in steps:
                    for k in ('err1s', 'err_r4039', 'err_cak'):
                        if s.get(k, 0.0) > 0:
                            s[k] *= 0.5
            rp = dlg.recalc_params
            if rp:
                _recalc_ages_in_steps(steps,
                                      rp['lam1'], rp['fc1_ma'], rp['j1'],
                                      rp['lam2'], rp['fc2_ma'],
                                      fc2_err_ma=rp.get('fc2_err_ma', 0.0))

            dlg2 = _PasteAliquotDialog(
                default_name=f"Aliquot {len(self._sessions) + 1}",
                has_ratio=has_ratio, parent=self)
            if dlg2.exec_() != QDialog.Accepted:
                return
            title    = dlg2.name()
            sess_cfg = dict(_SESSION_CFG_DEFAULTS)
            if rp:
                sess_cfg['lambdaYr']    = rp['lam2']
                sess_cfg['lambdaErrYr'] = rp.get('lam2_err', 0.0)
                sess_cfg['stdAgeMa']    = rp['fc2_ma']
                sess_cfg['stdAgeErrMa'] = rp.get('fc2_err_ma', 0.0)
                sess_cfg['stdAgeErrType'] = 'external' if rp.get('fc2_external', True) else 'internal'
            if dlg2.use_j():
                sess_cfg['ageFluxMode'] = 'ratio_J'
                _j2v_stk = [s.get('j_val', 0.0) for s in steps if s.get('j_val', 0.0) > 0]
                sess_cfg['jVal'] = float(np.median(_j2v_stk)) if _j2v_stk else dlg2.j_value()

            sess = _load_session_data(
                {'title': title, 'cfg': sess_cfg, 'steps': steps})
            self._stack_add_session(sess)

        except Exception as e:
            QMessageBox.critical(self, "Paste Error", str(e))

    # ── Load .ararpt / .arar files ────────────────────────────────────────────
    def _stack_load_files(self, paths=None):
        if paths is None:
            paths, _ = dfold.get_open_file_names(
                self, 'Load Sessions', '',
                'Ar-Ar Plateau (*.arar *.ararpt);;All files (*)')
        if not paths:
            return

        # Collect all sessions first, then rebuild the UI once.
        # Calling _update_flux_groups() per file causes O(n²) recomputes and
        # n repeated Qt widget create/delete cycles which crashes on large batches.
        deferred_warnings = []
        loaded_any = False
        for path in paths:
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                legacy_1s = (_session_err_div(data.get('cfg', {}),
                                              self._cb_1sigma.isChecked()) == 1.0)
                sess = _load_session_data(
                    data, err_div=(1.0 if legacy_1s else 2.0))
                sess['path'] = path
                fname = os.path.splitext(os.path.basename(path))[0]
                # the title saved in the file (a name the user typed) wins over
                # the file name, which is only the fallback
                sess['title'] = (data.get('title') or '').strip() or fname
                sess['orig_title'] = data.get('orig_title') or sess['title']
                # Snapshot for "Revert to original"
                sess['orig_raw_steps'] = copy.deepcopy(sess['raw_steps'])
                sess['orig_calib_cfg'] = {k: sess['cfg'][k] for k in (
                    'lambdaYr', 'lambdaErrYr', 'stdAgeMa', 'stdAgeErrMa',
                    'stdAgeErrType', 'ageFluxMode'
                ) if k in sess['cfg']}
                # A normal-mode save carries the disk-flux registry and the hole
                # the plateau was assigned to; bring both along so the stack
                # keeps that flux instead of starting with none.
                self._restore_disk_flux(data.get('disk_flux', {}))
                _fh = data.get('flux_hole')
                if _fh:
                    self._flux_override[len(self._sessions)] = tuple(_fh)
                # Append directly, skip per-file _update_flux_groups
                self._sessions.append(sess)
                loaded_any = True
                # Queue age-consistency check for after the UI update
                ok, info = _check_rstd_age_consistency(
                    sess['raw_steps'], sess['cfg'])
                if not ok:
                    deferred_warnings.append((fname, info))
            except Exception as e:
                QMessageBox.warning(self, 'Load Error',
                                    f'{os.path.basename(path)}:\n{e}')

        # Single UI rebuild + recompute for all sessions at once
        if loaded_any:
            if self._disk_flux:
                self._lbl_flux_status.setText(
                    f"{len(self._disk_flux)} disk(s) loaded: "
                    + ', '.join(f'{k[1]} ({k[0]})' for k in self._disk_flux))
                self._lbl_flux_status.setStyleSheet("color:#2e7d32; font-size:9px;")
            self._update_flux_groups()

        # Show age-mismatch warnings after UI is stable
        for fname, info in deferred_warnings:
            QMessageBox.warning(
                self, f"Age Mismatch: {fname}",
                f"In '{fname}':\n"
                f"The ages recomputed from the 40Ar/39Ar ratios do not\n"
                f"match the imported ages stored in the file.\n\n"
                f"{info}\n\n"
                f"Use Calibration… to recalculate to consistent constants."
            )

    # ── Stack save / load ─────────────────────────────────────────────────────
    def _stack_save(self):
        if not self._sessions:
            QMessageBox.warning(self, 'No Data', 'No aliquots loaded.')
            return
        path, _ = dfold.get_save_file_name(
            self, 'Save Stack', 'stack_session.ararstack',
            'Ar-Ar Stack (*.ararstack);;All files (*)')
        if not path:
            return
        try:
            sessions_data = []
            # Sync flux group cfg into each session cfg so it round-trips correctly
            for g in self._flux_groups:
                for si in g['indices']:
                    if 0 <= si < len(self._sessions):
                        self._sessions[si]['cfg'].update(g['flux_cfg'])
            for si, sess in enumerate(self._sessions):
                cfg = dict(sess['cfg']); cfg['errScale'] = 2
                # Save the plateau age + tiers (1σ) and the plateau step window
                # (pset): the steps outside the plateau are stored with status
                # 'normal', so a reader (the Age distribution tool) needs the
                # window to use this plateau.  The J its mean R was turned into
                # an age with (group R_std, typed J or a flux-hole J) is saved
                # too, so a reader that rebuilds ages from R can put aliquots
                # from different positions on one calibration without shifting
                # any of them.
                plat_summary = _plateau_summary(sess.get('plateau'),
                                                self._eff_cfg_for(si))
                sessions_data.append({
                    'title':    sess.get('title', ''),
                    'orig_title': sess.get('orig_title') or sess.get('title', ''),
                    'cfg':      cfg,
                    'steps':    _steps_to_disk(sess['raw_steps']),
                    'excluded': sess.get('excluded', False),
                    'plateau':  plat_summary,
                })
            disk_flux_data = serialize_disk_flux(self._disk_flux)
            # flux_override keys are ints (session indices), store as strings;
            # a hole as [irr, disk, hole], a session set to no hole as 'none'
            flux_override_data = {str(k): (v if v == 'none' else list(v))
                                  for k, v in self._flux_override.items()}
            state = {
                'format':         'ararstack_v1',
                'global_cfg':     self._global_cfg,
                'sessions':       sessions_data,
                'disk_flux':      disk_flux_data,
                'flux_override':  flux_override_data,
                'apply_flux_j':   True,
                'stack_method':   self._stack_method,
                'wm_n_basis': self._wm_n_basis,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            QMessageBox.information(self, 'Saved', f'Stack saved:\n{path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save Error', str(e))

    def _stack_load(self):
        paths, _ = dfold.get_open_file_names(
            self, 'Load Stack / Sessions', '',
            'All Ar-Ar files (*.ararstack *.arar *.ararpt);;'
            'Stack session (*.ararstack);;'
            'Individual sessions (*.arar *.ararpt);;'
            'All files (*)')
        if not paths:
            return
        # Route: .ararstack → full stack replace; .arar/.ararpt → append sessions
        stack_paths = [p for p in paths if p.lower().endswith('.ararstack')]
        sess_paths  = [p for p in paths if not p.lower().endswith('.ararstack')]
        if stack_paths and sess_paths:
            QMessageBox.warning(
                self, 'Mixed selection',
                'Select either one .ararstack file or individual .arar/.ararpt files, not both.')
            return
        if sess_paths:
            self._stack_load_files(paths=sess_paths)
            return
        path = stack_paths[0]
        try:
            with open(path, encoding='utf-8') as f:
                state = json.load(f)
            if state.get('format') != 'ararstack_v1':
                QMessageBox.warning(self, 'Unknown Format',
                                    'Not a valid stack session file.')
                return
            # Always append, never replace the existing stack.
            existing_n = len(self._sessions)
            is_fresh   = (existing_n == 0)

            # Only restore global settings / UI when loading into an empty stack.
            if is_fresh:
                self._global_cfg.update(state.get('global_cfg', {}))
                self._set_p40max(self._global_cfg.get('p40Max', 0.0))
                for _k in ('minPct', 'minN', 'pCut', 'errMode', 'ageDec'):
                    self.cfg[_k] = self._global_cfg.get(_k, _SESSION_CFG_DEFAULTS.get(_k))
                for sp, key in [(self._sp_minPct,  'minPct'),
                                (self._sp_minN,    'minN'),
                                (self._sp_pCut,    'pCut'),
                                (self._sp_ageDec,  'ageDec')]:
                    sp.blockSignals(True)
                    sp.setValue(self._global_cfg.get(key, self.cfg.get(key, 2)))
                    sp.blockSignals(False)
                for _i in range(self._cb_errMode.count()):
                    if self._cb_errMode.itemData(_i) == self._global_cfg.get('errMode'):
                        self._cb_errMode.blockSignals(True)
                        self._cb_errMode.setCurrentIndex(_i)
                        self._cb_errMode.blockSignals(False)
                        break

            # Append sessions from file
            for sd in state.get('sessions', []):
                _sc = sd.get('cfg', {})
                legacy_1s = (_session_err_div(_sc, self._cb_1sigma.isChecked())
                             == 1.0)
                sess = _load_session_data({
                    'title': sd.get('title', ''),
                    'orig_title': sd.get('orig_title'),
                    'cfg':   _sc,
                    'steps': sd.get('steps', []),
                }, err_div=(1.0 if legacy_1s else 2.0))
                # Snapshot for "Revert to original" (same as in _stack_load_files)
                sess['orig_raw_steps'] = copy.deepcopy(sess['raw_steps'])
                sess['orig_calib_cfg'] = {k: sess['cfg'][k] for k in (
                    'lambdaYr', 'lambdaErrYr', 'stdAgeMa', 'stdAgeErrMa',
                    'stdAgeErrType', 'ageFluxMode'
                ) if k in sess['cfg']}
                # Restore aliquot-level exclusion (the × checkbox in the table)
                sess['excluded'] = sd.get('excluded', False)
                # Carousel saves (normal mode, 2+ aliquots) store each aliquot's
                # hole on the session; an explicit flux_override below wins.
                _fh = sd.get('flux_hole')
                if _fh:
                    self._flux_override[len(self._sessions)] = tuple(_fh)
                self._sessions.append(sess)

            # Merge disk flux, add new disks without overwriting existing ones
            self._restore_disk_flux(state.get('disk_flux', {}))

            # Merge manual hole assignments, offset indices by existing session count
            for k, v in state.get('flux_override', {}).items():
                ov = _flux_override_from_disk(v)
                if ov is not None:
                    self._flux_override[int(k) + existing_n] = ov

            # Only restore combine method / apply-flux-J when loading into empty stack
            if is_fresh:
                saved_method = method_key(state.get('stack_method', METHOD_WM))
                self._stack_method = saved_method
                rb = self._rb_gls if saved_method == METHOD_GLS else self._rb_wm
                rb.blockSignals(True); rb.setChecked(True); rb.blockSignals(False)
                # GLS + ML fixes the error method (see _on_stack_method)
                for _w in (self._cb_errMode, self._lbl_errMode):
                    _w.setEnabled(saved_method != METHOD_GLS)
                self._wm_n_basis = state.get('wm_n_basis',
                                             state.get('koppers_n_basis', 'aliquots'))
                if hasattr(self, '_cb_wm_n'):
                    _ki = self._cb_wm_n.findData(self._wm_n_basis)
                    self._cb_wm_n.blockSignals(True)
                    self._cb_wm_n.setCurrentIndex(max(_ki, 0))
                    self._cb_wm_n.setEnabled(saved_method != METHOD_GLS)
                    self._cb_wm_n.blockSignals(False)

            if self._disk_flux:
                self._lbl_flux_status.setText(
                    f"{len(self._disk_flux)} disk(s) loaded: "
                    + ', '.join(f'{k[1]} ({k[0]})' for k in self._disk_flux))
                self._lbl_flux_status.setStyleSheet("color:#2e7d32; font-size:9px;")
            self._update_flux_groups()
            n_added = len(self._sessions) - existing_n
            total   = len(self._sessions)
            msg = f"Added {n_added} aliquot(s) from:\n{path}"
            if existing_n:
                msg += f"\n(Stack total: {total} aliquots)"
            QMessageBox.information(self, 'Loaded', msg)
        except Exception as e:
            QMessageBox.critical(self, 'Load Error', str(e))

    # ── Stack CSV export ──────────────────────────────────────────────────────
    def _stack_export_csv(self):
        if not self._sessions:
            QMessageBox.warning(self, 'No Data', 'No aliquots loaded.')
            return
        path, _flt = dfold.get_save_file_name(
            self, 'Export Data', 'stacked_data.xlsx', export.save_filter())
        if not path:
            return
        path = export.ensure_ext(path, _flt)
        try:
            import csv
            display_1sigma, sig_scale, sig_lbl = self._disp()
            # only the aliquots that are in the stack are exported
            _inc = [(si, sess) for si, sess in enumerate(self._sessions)
                    if not sess.get('excluded', False)]
            if not _inc:
                QMessageBox.warning(self, 'No Data', 'No aliquots are included in the stack.')
                return
            _sink = export.TableSink(path)
            with _sink as f:
                w = csv.writer(f)

                # ── Constants (one block for everything), raw sheet ───────────
                _raw_items, _entries = [], []
                for si, sess in _inc:
                    _st = sess.get('steps_c', [])
                    _ms = [export.step_meta(x) for x in _st]
                    _entries.append((self._eff_cfg_for(si),
                                     (sess.get('cfg') or {}).get('constants'), _ms))
                    _raw_items.append((sess.get('title', f'#{si+1}'), _st, _ms,
                                       (sess.get('cfg') or {}).get('constants')))
                export.write_constants(w, groups=export.constants_groups(_entries),
                                     atm=ATM_40_36)
                _sink.set_raw_items(_raw_items)

                # ── Per-aliquot step data ─────────────────────────────────────
                def _fo(v):    return fsig(v, 8)   # 8 sig figs for ratio values
                def _foe(v):   return fsig(v, 6)   # 6 sig figs for ratio errors

                for si, sess in _inc:
                    title = sess.get('title', f'#{si+1}')
                    eff_cfg_s = self._eff_cfg_for(si)
                    _jc_s = _export_j_cells(eff_cfg_s)
                    _rv_s = _re_s = None
                    _fh_s = self._stack_flux_hole(si)
                    if _fh_s is not None and eff_cfg_s.get('ageFluxMode') == 'ratio_J':
                        # on a loaded hole: the hole supplies J and its R
                        _fx = self._disk_flux.get((str(_fh_s[0]), str(_fh_s[1])))
                        if _fx is not None and str(_fh_s[2]) in _fx['idx']:
                            _hi = _fx['idx'][str(_fh_s[2])]
                            _rv_s = self._flux_hole_R(_fx, _hi)
                            try:
                                if _rv_s and _fx.get('R_sigma') is not None and _fx.get('R') is not None:
                                    _re_s = float(_fx['R_sigma'][_hi])
                                elif _rv_s:
                                    _re_s = _rv_s * float(_fx['J_sigma'][_hi]) / float(_fx['J'][_hi])
                            except Exception:
                                _re_s = None
                    _cfg_rr = eff_cfg_s
                    if _rv_s and _re_s is not None:
                        _cfg_rr = dict(eff_cfg_s, ageFluxMode='ratio_R_std', rStd=_rv_s,
                                       rStdErrPct=_re_s / _rv_s * 100.0, rStdErrMode='pct')
                    _mode_lbl_s = _AGEMODE_LABEL.get(
                        eff_cfg_s.get('ageFluxMode', 'imported'), 'imported')
                    w.writerow([f'STEP DATA, {title}  [Age source: {_mode_lbl_s}]'])
                    w.writerow(['RunID', 'Label', 'Power', 'Run date/time',
                                'Days since irradiation', *export.FLUX_HEADERS,
                                'Age (Ma)', '±1σ Ana (Ma)',
                                '%39Ar', 'K/Ca', '±1σ', '%40Ar*',
                                '40Ar*/39ArK', '±1σ',
                                'R_unk/R_std', '±1σ',
                                '39Ar(k)/36Ar', '±1σ', '40Ar/36Ar', '±1σ', 'RI(norm)',
                                '39Ar(k)/40Ar', '±1σ', '36Ar/40Ar', '±1σ', 'RI(inv)',
                                'In plateau', 'Status'])
                    for i, s in enumerate(sess.get('steps_c', [])):
                        cak = s['caK']
                        if cak and cak > 0:
                            kca_str   = fsig(1.0 / cak, 5)
                            kca_e_str = (fsig(s['err_cak'] / (cak * cak), 3)
                                         if s.get('err_cak') is not None else '')
                        else:
                            kca_str = kca_e_str = ''
                        w.writerow([
                            s.get('runID', ''),
                            s['label'], f"{s['power']:.3f}",
                            s.get('run_dt', ''), export.fmt_days(export.step_meta(s)),
                            *export.flux_cells(eff_cfg_s, *(_jc_s or (s.get('j_val'), s.get('j_err'))),
                                             show_r=_show_r_standard(eff_cfg_s),
                                             r_val=_rv_s, r_err=_re_s),
                            f"{s['age']:.8f}", f"{s['err1s']:.8f}",
                            f"{s['pct39Ar']:.3f}",
                            kca_str, kca_e_str,
                            f"{s['pct40']:.1f}",
                            ff(s.get('r4039'), 8),   ff(s.get('err_r4039'), 8),
                            *_fo_rr(s, _cfg_rr),
                            _fo(s.get('iso_ix')),  _foe(s.get('iso_isx')),
                            _fo(s.get('iso_iy')),  _foe(s.get('iso_isy')),
                            _fo(s.get('iso_nri')),
                            _fo(s.get('iso_nx')),  _foe(s.get('iso_nsx')),
                            _fo(s.get('iso_ny')),  _foe(s.get('iso_nsy')),
                            _fo(s.get('iso_iri')),
                            'yes' if i in sess.get('pset', set()) else 'no',
                            s.get('status', 'normal'),
                        ])
                    w.writerow([])

                # ── Per-aliquot plateau summary ───────────────────────────────
                w.writerow(['PLATEAU RESULTS'])
                w.writerow(['Sample', 'Plateau age (Ma)',
                            f'±{sig_lbl} Ana (Ma)', f'±{sig_lbl} Int (Ma)', f'±{sig_lbl} Ext (Ma)',
                            'R_unk/R_std', f'±{sig_lbl} Ana R', f'±{sig_lbl} Int R', f'±{sig_lbl} Ext R',
                            '%39Ar', 'MSWD', 'p', 'n steps', 'J', '±J%', 'disk', 'hole',
                            'In stack'])
                for si, sess in _inc:
                    title   = sess.get('title', f'#{si+1}')
                    p       = sess.get('plateau')
                    cfg     = sess.get('cfg', {})
                    eff_cfg = self._eff_cfg_for(si)
                    # the J the plateau age is computed with, not the jVal
                    # stored in the cfg (stale after a recalibration or flux)
                    _jv, _jrel = _aged_J(eff_cfg, sess.get('steps_c', []))
                    j_val   = fsig(_jv, 8) if _jv else ''
                    j_err   = f"{_jrel * 100.0:.5f}" if _jv else ''
                    # the hole the age actually uses (a loaded disk-flux hole,
                    # picked or matched), else the disk / hole named in the file
                    _fh = self._stack_flux_hole(si)
                    disk_id = str(_fh[1]) if _fh else cfg.get('disk_id', '')
                    hole_id = str(_fh[2]) if _fh else cfg.get('hole_id', '')
                    if p:
                        _rr_p, _drdt_p = rr_from_age(p['r']['mean'], eff_cfg)
                        w.writerow([title,
                                    f"{p['r']['mean']:.8f}",
                                    f"{p.get('e1s_ana', p['e1s_int']) * sig_scale:.8f}",
                                    f"{p['e1s_int'] * sig_scale:.8f}",
                                    f"{p.get('e1s_ext', p['e1s_int']) * sig_scale:.8f}",
                                    ff(_rr_p, 8),
                                    ff(_drdt_p * p.get('e1s_ana', p['e1s_int']) * sig_scale, 8) if _drdt_p else '',
                                    ff(_drdt_p * p['e1s_int'] * sig_scale, 8) if _drdt_p else '',
                                    ff(_drdt_p * p.get('e1s_ext', p['e1s_int']) * sig_scale, 8) if _drdt_p else '',
                                    f"{p['pct39']:.2f}",
                                    f"{p['r']['mswd']:.4f}",
                                    f"{p['r']['prob']:.4f}",
                                    p['n'], j_val, j_err, disk_id, hole_id,
                                    'no' if sess.get('excluded', False) else 'yes'])
                    else:
                        w.writerow([title, 'No plateau'] + [''] * 11 +
                                   [j_val, j_err, disk_id, hole_id, 'no'])
                w.writerow([])

                # ── Per-aliquot isochron results ──────────────────────────────
                w.writerow(['ISOCHRON RESULTS'])
                w.writerow(['Sample', 'Type', 'Age (Ma)',
                            f'±{sig_lbl} Ana (Ma)', f'±{sig_lbl} Int (Ma)', f'±{sig_lbl} Ext (Ma)',
                            'R_unk/R_std', f'±{sig_lbl} Ana R', f'±{sig_lbl} Int R', f'±{sig_lbl} Ext R',
                            'Initial 40Ar/36Ar', f'±{sig_lbl}', 'MSWD', 'p', 'n'])
                for si, sess in _inc:
                    title  = sess.get('title', f'#{si+1}')
                    # the aliquot's plateau steps on the J its plateau age uses
                    # (not the J stored in the file, which predates any
                    # recalibration or flux assignment)
                    cfg_s  = self._eff_cfg_for(si)
                    steps_s = self._tag_isochron_steps(sess.get('steps_c', []),
                                                       sess.get('pset', set()))
                    _J_s   = self._iso_J(cfg_s, sess.get('steps_c', []))
                    _lam_s = cfg_s.get('lambdaYr',    LAMBDA_DEFAULT)
                    for label, is_inv in [('Normal', False), ('Inverse', True)]:
                        res = self._calc_isochron_result(steps_s, is_inv, _J_s, _lam_s,
                                                         cfg_s)
                        if res:
                            ir40 = (f"{res['init_ratio']:.2f}"
                                    if res['init_ratio'] is not None else '')
                            ir40e = (f"{res['init_ratio_1s'] * sig_scale:.2f}"
                                     if res['init_ratio_1s'] is not None else '')
                            _rr_iso, _drdt_iso = rr_from_age(res['age_ma'], cfg_s)
                            w.writerow([title, label,
                                        f"{res['age_ma']:.8f}",
                                        f"{res['e1s_ana'] * sig_scale:.8f}",
                                        f"{res['e1s_int'] * sig_scale:.8f}",
                                        f"{res['e1s_ext'] * sig_scale:.8f}",
                                        ff(_rr_iso, 8),
                                        ff(_drdt_iso * res['e1s_ana'] * sig_scale, 8) if _drdt_iso else '',
                                        ff(_drdt_iso * res['e1s_int'] * sig_scale, 8) if _drdt_iso else '',
                                        ff(_drdt_iso * res['e1s_ext'] * sig_scale, 8) if _drdt_iso else '',
                                        ir40, ir40e,
                                        f"{res['mswd']:.3f}",
                                        f"{res['prob']:.4f}",
                                        res['n']])
                        else:
                            w.writerow([title, label] + [''] * 13)
                w.writerow([])

                # ── Stacked isochron (the Isochron view in stack mode) ────────
                w.writerow(['STACKED ISOCHRON RESULTS'])
                w.writerow(['Sample', 'Type', 'Age (Ma)',
                            f'±{sig_lbl} Ana (Ma)', f'±{sig_lbl} Int (Ma)', f'±{sig_lbl} Ext (Ma)',
                            'R_unk/R_std', f'±{sig_lbl} Ana R', f'±{sig_lbl} Int R', f'±{sig_lbl} Ext R',
                            'Initial 40Ar/36Ar', f'±{sig_lbl}', 'MSWD', 'p', 'n'])
                for label, is_inv in [('Normal', False), ('Inverse', True)]:
                    res = self._stacked_isochron_result(is_inv)
                    if res:
                        _rr_st, _drdt_st = rr_from_age(res['age_ma'], res['cfg'])
                        w.writerow([f"Stack ({res['n_aliquots']} aliquots)", label,
                                    f"{res['age_ma']:.8f}",
                                    f"{res['e1s_ana'] * sig_scale:.8f}",
                                    f"{res['e1s_int'] * sig_scale:.8f}",
                                    f"{res['e1s_ext'] * sig_scale:.8f}",
                                    ff(_rr_st, 8),
                                    ff(_drdt_st * res['e1s_ana'] * sig_scale, 8) if _drdt_st else '',
                                    ff(_drdt_st * res['e1s_int'] * sig_scale, 8) if _drdt_st else '',
                                    ff(_drdt_st * res['e1s_ext'] * sig_scale, 8) if _drdt_st else '',
                                    (f"{res['init_ratio']:.2f}"
                                     if res['init_ratio'] is not None else ''),
                                    (f"{res['init_ratio_1s'] * sig_scale:.2f}"
                                     if res['init_ratio_1s'] is not None else ''),
                                    f"{res['mswd']:.3f}",
                                    f"{res['prob']:.4f}",
                                    res['n']])
                    else:
                        w.writerow(['Stack', label] + [''] * 13)
                w.writerow([])

                # ── Combined plateau results (both methods) ───────────────────
                _saved_method = self._stack_method
                _both = {}
                for _m in (METHOD_GLS, METHOD_WM):
                    self._stack_method = _m
                    try:
                        _both[_m] = self._combined_plateau()
                    except Exception:
                        _both[_m] = (None, None, None, None)
                self._stack_method = _saved_method

                w.writerow(['COMBINED PLATEAU RESULTS'])
                w.writerow(['Method', 'Combined plateau age (Ma)',
                            f'±{sig_lbl} Ana (Ma)', f'±{sig_lbl} Int (Ma)', f'±{sig_lbl} Ext (Ma)',
                            'R_unk/R_std', f'±{sig_lbl} Ana R', f'±{sig_lbl} Int R', f'±{sig_lbl} Ext R',
                            'n', 'MSWD', 'p', f'ω ({sig_lbl} Ma)', 'notes'])
                _comb_cfg = self._eff_cfg_for(0) if self._sessions else {}

                # GLS + ML row: ω carries the excess scatter, so MSWD and p are
                # left blank here (the Stats dialog shows them).
                _vr, _va, _vi, _vx = _both[METHOD_GLS]
                if _vr and _vi is not None:
                    _od = _vr.get('overdispersion', 0.0)
                    _od_str = f"{_od * sig_scale:.4f}" if _vr.get('overdispersed') else '0'
                    _rr_v, _drdt_v = rr_from_age(_vr['mean'], _comb_cfg)
                    w.writerow(['GLS + ML random-effects (Vermeesch 2015 §13)',
                                f"{_vr['mean']:.8f}",
                                f"{_va * sig_scale:.8f}",
                                f"{_vi * sig_scale:.8f}",
                                f"{_vx * sig_scale:.8f}",
                                ff(_rr_v, 8),
                                ff(_drdt_v * _va * sig_scale, 8) if _drdt_v else '',
                                ff(_drdt_v * _vi * sig_scale, 8) if _drdt_v else '',
                                ff(_drdt_v * _vx * sig_scale, 8) if _drdt_v else '',
                                f"{_vr['n']} aliquots",
                                '', '', _od_str, ''])

                # WM + √MSWD row, MSWD and p are the primary test statistics
                _kr, _ka, _ki, _kx = _both[METHOD_WM]
                if _kr and _ki is not None:
                    _k_method = _kr.get('method', '')
                    _k_note = (f"J₁={_kr.get('J1', 0.0):.6g}, F_bar={_kr.get('F_bar', 0.0):.8g}"
                               if _k_method == 'wm_ratio' else _k_method)
                    _rr_k, _drdt_k = rr_from_age(_kr['mean'], _comb_cfg)
                    w.writerow([_wm_label(_kr),
                                f"{_kr['mean']:.8f}",
                                f"{_ka * sig_scale:.8f}",
                                f"{_ki * sig_scale:.8f}",
                                f"{_kx * sig_scale:.8f}",
                                ff(_rr_k, 8),
                                ff(_drdt_k * _ka * sig_scale, 8) if _drdt_k else '',
                                ff(_drdt_k * _ki * sig_scale, 8) if _drdt_k else '',
                                ff(_drdt_k * _kx * sig_scale, 8) if _drdt_k else '',
                                f"{_kr['n']} " + ('aliquots' if 'aliquots' in _k_method else 'steps'),
                                f"{_kr['mswd']:.4f}", f"{_kr['prob']:.4f}",
                                '', _k_note])

                w.writerow([])

                # ── Vermeesch covariance matrices ─────────────────────────────
                # Reconstruct inputs for the Vermeesch call to get matrices.
                _v_ages, _v_sa, _v_si, _v_se, _v_used = [], [], [], [], []
                for _si2 in sorted({
                        _si3: [s for i3, s in enumerate(sess.get('steps_c', []))
                               if i3 in sess.get('pset', set())]
                        for _si3, sess in enumerate(self._sessions)
                        if sess.get('plateau') and not sess.get('excluded', False)}.keys()):
                    _p2 = self._sessions[_si2].get('plateau')
                    if not _p2: continue
                    _ea = _p2.get('e1s_ana')
                    if _ea is None or _ea <= 0: continue
                    _v_ages.append(float(_p2['r']['mean']))
                    _v_sa.append(_ea)
                    _v_si.append(_p2.get('e1s_int', _ea))
                    _v_se.append(_p2.get('e1s_ext', _p2.get('e1s_int', _ea)))
                    _v_used.append(_si2)

                if len(_v_ages) >= 2:
                    _rho2  = self._build_stack_pcorr(_v_used)
                    _sg2, _C2 = self._stack_systematics(_v_used, _v_ages)
                    _mres  = combine_ages_vermeesch(
                        _v_ages, _v_sa, _v_si, _v_se,
                        standard_group=_sg2, p_corr=_rho2,
                        random_effects=True, return_matrices=True,
                        sys_components=_C2)
                    _mx    = _mres.get('matrices', {})
                    _lbls  = [self._sessions[_si2].get('title', f'#{_si2+1}')
                              for _si2 in _v_used]

                    def _write_matrix(name, M, fmt='.6g'):
                        if name:
                            w.writerow([name] + _lbls)
                        else:
                            w.writerow([''] + _lbls)
                        for ri2, row in enumerate(M):
                            w.writerow([_lbls[ri2]]
                                       + [f'{v:{fmt}}' for v in row])
                        w.writerow([])

                    w.writerow(['GLS + ML COVARIANCE MATRICES'])
                    w.writerow(['V = A + P + S  |  all matrix cells in Ma² (variance/covariance units)'])
                    w.writerow([f'ω (overdispersion, 1σ) = {math.sqrt(_mx.get("omega2", 0.0)):.6g} Ma',
                                f'ω² = {_mx.get("omega2", 0.0):.6g} Ma²'])
                    w.writerow([])
                    w.writerow(['Aliquot inputs (all 1σ, Ma)', 'Age (Ma)',
                                'σ_ana', 'σ_int', 'σ_ext',
                                'p = √(σ_int²−σ_ana²)  J-precision',
                                's = √(σ_ext²−σ_int²)  monitor+λ'])
                    _p_vec = _mx.get('p', [])
                    _s_vec = _mx.get('s', [])
                    for _ii, _lbl in enumerate(_lbls):
                        w.writerow([_lbl,
                                    f'{_v_ages[_ii]:.6f}',
                                    f'{_v_sa[_ii]:.6g}',
                                    f'{_v_si[_ii]:.6g}',
                                    f'{_v_se[_ii]:.6g}',
                                    f'{float(_p_vec[_ii]):.6g}' if len(_p_vec) > _ii else '',
                                    f'{float(_s_vec[_ii]):.6g}' if len(_s_vec) > _ii else ''])
                    w.writerow([])
                    for _mname, _mkey, _mdesc in [
                            ('A  (Ma²)', 'A',
                             'Diagonal: σ_ana_i² + ω², analytical variance + overdispersion. '
                             'Off-diagonal = 0 (mass-spec errors are independent between aliquots).'),
                            ('P  (Ma²)', 'P',
                             'P_ij = p_i × p_j × ρ_ij, J/R_std covariance. '
                             'ρ=1 same irradiation hole, 0<ρ<1 from .jflux, 0 if independent.'),
                            ('S  (Ma²)', 'S',
                             'S_ij = s_i × s_j if same monitor standard, else 0. '
                             'Monitor age and λ systematics are perfectly correlated within a calibration group.'),
                            ('V = A+P+S  (Ma²)', 'V',
                             'Covariance matrix of the external tier. GLS mean = (1\'V⁻¹1)⁻¹ · 1\'V⁻¹t  (Vermeesch Eq. 92).')]:
                        if _mkey in _mx:
                            w.writerow([_mname, _mdesc])
                            _write_matrix('', _mx[_mkey])

                w.writerow([])
            QMessageBox.information(self, 'Export OK', f'Saved:\n{path}')
        except Exception as e:
            QMessageBox.critical(self, 'Export Error', str(e))


# --------------------------------------------------------------------------
# Paste-aliquot dialog
# --------------------------------------------------------------------------

class _PasteAliquotDialog(QDialog):

    def __init__(self, default_name='', has_ratio=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Aliquot")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._name_edit = QLineEdit(default_name)
        form.addRow("Aliquot name:", self._name_edit)
        layout.addLayout(form)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        hdr = QLabel("Flux / Age calibration (optional)")
        hdr.setStyleSheet("font-weight:bold; font-size:10px; color:#444;")
        layout.addWidget(hdr)

        self._cb_use_j = QCheckBox(
            "Compute ages from ⁴⁰Ar/³⁹Ar ratio using J value")
        self._cb_use_j.setChecked(has_ratio)
        layout.addWidget(self._cb_use_j)

        jrow = QHBoxLayout()
        jrow.addWidget(QLabel("J value:"))
        self._le_j = QLineEdit(repr(_SESSION_CFG_DEFAULTS['jVal']))
        self._le_j.setPlaceholderText("e.g. 0.00271559")
        jrow.addWidget(self._le_j)
        layout.addLayout(jrow)

        self._cb_use_j.stateChanged.connect(
            lambda: self._le_j.setEnabled(self._cb_use_j.isChecked()))
        self._le_j.setEnabled(has_ratio)

        layout.addWidget(QLabel(
            "<small style='color:#888;'>Leave unchecked to use imported ages as-is.</small>",
            self))
        layout.itemAt(layout.count() - 1).widget().setTextFormat(Qt.RichText)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def name(self):
        return self._name_edit.text().strip() or self._name_edit.placeholderText()

    def use_j(self):
        return self._cb_use_j.isChecked()

    def j_value(self):
        try:    return float(self._le_j.text())
        except (ValueError, TypeError): return _SESSION_CFG_DEFAULTS['jVal']


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Plateau & isochron")
    dfold.use_tool('plateau_isochron')
    force_light(app)   # shared light palette, same as the other tools
    no_wheel_on_dropdowns(app)
    win = PlateauWindow()
    win.show()
    ret = app.exec_()
    del win   # destroy before sys.exit() so Qt cleans up before Python GC runs
    sys.exit(ret)


if __name__ == '__main__':
    main()