"""Exports: the data tables (.xlsx / .csv) and the figure-export dialog.

Data export
-----------
One place for what the exports have in common, so they cannot drift apart:

* the CONSTANTS block that opens every export (atmospheric argon, decay
  constants, FC age, J, irradiation time ...);
* the time of each analysis, as days since the irradiation;
* the raw-signal table, written as a second sheet of an .xlsx (and appended as
  a section to a .csv);
* TableSink, a drop-in replacement for ``open(path, 'w')`` around a csv.writer
  that writes either a .csv or a multi-sheet .xlsx.

The raw values come from the mass-spec file through the converter, which stores
them in each step's ``raw`` dict and the run time in ``run_dt`` / ``days_irr``.
Files converted before raw signals were stored have blank cells here.

Figure export
-------------
One dialog, used by every tool that draws a figure: pick which panels go in,
how tall each one is, the page size, the text size, where the legend sits, and
whatever extra switches the host tool declares (Ca/K vs K/Ca, isochron layout,
...).  A live preview shows exactly what will be written to disk.

A host wires itself in with three things:

    panels   list of Panel(...)        the stackable panels, top to bottom
    options  list of Option(...)       extra switches, rendered as a form
    render   callable(fig, spec)       draws the whole figure into *fig*

`spec` is a plain dict (see DEFAULT_SPEC) carrying the panel list, the page
size, the scaling factors, the legend mode and the host's own option values,
so a host can read it without importing anything from here.  The dialog
knows nothing about geochronology; the drawing stays in the tool.

Presets are stored in figure_export_presets.json in the user's configuration
folder (data_folders.config_path).  Each figure type keeps its own last-used
layout; the named presets are shared within a tool (PRESET_GROUPS), so one
saved for the stack is also offered for a single sample.
"""
import copy
import csv
import datetime
import io
import json
import math
import os
import re

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMessageBox, QProgressDialog, QPushButton,
    QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib
import matplotlib.lines as _mlines
import matplotlib.text as _mtext

from arar_math import (cfg_J, cfg_monitor, monitor_age, calibration_notes,
                       edited_standards_note)
import data_folders as dfold


# ===========================================================================
# DATA EXPORT
# ===========================================================================

# ---------------------------------------------------------------------------
# step metadata
# ---------------------------------------------------------------------------

def step_meta(step):
    """The export-only fields of one on-disk or in-memory step dict."""
    days = step.get('days_irr')
    try:
        days = None if days is None else float(days)
        if days is not None and not math.isfinite(days):
            days = None
    except (TypeError, ValueError):
        days = None
    raw = step.get('raw')
    return {'run_dt': str(step.get('run_dt') or ''),
            'days_irr': days,
            'raw': dict(raw) if isinstance(raw, dict) else {}}


def fmt_days(meta):
    d = (meta or {}).get('days_irr')
    return '' if d is None else f'{d:.4f}'


def _parse_dt(s):
    try:
        return datetime.datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def irradiation_estimate(metas):
    """Median of (run time - days since irradiation) over the steps: the
    irradiation reference time implied by the decay corrections, as ISO text,
    or '' when the steps carry no times."""
    vals = []
    for m in metas:
        dt = _parse_dt((m or {}).get('run_dt'))
        d = (m or {}).get('days_irr')
        if dt is not None and d is not None:
            vals.append(dt - datetime.timedelta(days=d))
    if not vals:
        return ''
    vals.sort()
    # one value per irradiation: cluster estimates that lie within a day
    clusters, cur = [], [vals[0]]
    for v in vals[1:]:
        if (v - cur[-1]).total_seconds() > 86400:
            clusters.append(cur)
            cur = []
        cur.append(v)
    clusters.append(cur)
    return '; '.join(c[len(c) // 2].strftime('%Y-%m-%d %H:%M') for c in clusters)


# ---------------------------------------------------------------------------
# constants block
# ---------------------------------------------------------------------------

def _g(v, sig=8):
    if v is None or v == '':
        return ''
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return ''
    if f == 0:
        return '0'
    return f'{f:.{sig}g}'


def constants_rows(cfg, constants=None, metas=(), atm=None, extra=()):
    """Rows [name, value, note] describing every constant behind the numbers.

    cfg        the effective cfg of the data (λ, FC, J, R standard ...)
    constants  the 'constants' block the converter stored from the mass-spec
               file (atmospheric argon, partial decay constants, correction
               factors); may be empty
    metas      step_meta() dicts, for the irradiation time estimate
    atm        atmospheric 40Ar/36Ar used by the tools when the file has none
    extra      more [name, value, note] rows to append
    """
    c = constants or {}
    rows = []

    def add(name, value, note=''):
        rows.append([name, value, note])

    # the ratio the ages were actually calculated with: the source file's when
    # it has one, else the tools' own
    _air = c.get('air_40_36') if c.get('air_40_36') not in (None, '') else atm
    if _air not in (None, ''):
        add('Atmospheric 40Ar/36Ar', _g(_air, 8), '')
        if c.get('air_40_36') not in (None, '') and c.get('air_40_36_err') not in (None, ''):
            add('  ± (1σ)', _g(c.get('air_40_36_err')), '')
    if c.get('air_40_38') not in (None, ''):
        add('Atmospheric 40Ar/38Ar', _g(c.get('air_40_38')), '')

    add('FC age (Ma)', _g(cfg.get('stdAgeMa'), 8), 'Fish Canyon sanidine (FCs) calibration')
    add('  ± (Ma, 1σ)', _g(cfg.get('stdAgeErrMa'), 6),
        'external' if cfg.get('stdAgeErrType') == 'external' else 'internal')
    for _n in calibration_notes(cfg):
        add('  Note', _n, '')
    _ed = edited_standards_note()
    if _ed:
        add('Standards', _ed, '')
    mon, R_m, s_m = cfg_monitor(cfg)
    if mon != 'FCs':
        add('Flux monitor', mon, 'R standard values are FCs-equivalent: measured R / R(monitor/FCs)')
        add(f'  R({mon}/FCs)', _g(R_m, 8), f'± {_g(s_m, 4)} (1σ)')
        try:
            add(f'  {mon} age (Ma)', _g(monitor_age(cfg), 8), 'on this calibration')
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    add('40K decay constant λ (1/yr)', _g(cfg.get('lambdaYr'), 8), 'total (ε + β)')
    add('  ± (1/yr, 1σ)', _g(cfg.get('lambdaErrYr'), 6), '')
    # The partial constants come from the source file (the software that made
    # it). List them only when they add up to the total λ the ages here were
    # calculated with; otherwise they are the source software's own values,
    # play no part in these numbers and would contradict the λ row above.
    if c.get('lam_eps') not in (None, '') and c.get('lam_beta') not in (None, ''):
        try:
            _le, _lb = float(c.get('lam_eps')), float(c.get('lam_beta'))
            _lt = float(cfg.get('lambdaYr'))
            _match = abs(_le + _lb - _lt) <= 1e-3 * _lt
        except (TypeError, ValueError):
            _match = False
        if _match:
            add('  λ ε (1/yr)', _g(_le), '')
            add('  λ β- (1/yr)', _g(_lb), '')
    if c.get('k40_abund') not in (None, ''):
        # fraction of K; the mass-spec file gives it in percent (0.01167)
        _k = float(c.get('k40_abund'))
        add('40K abundance (fraction of K)', _g(_k / 100.0 if _k >= 1e-3 else _k), '')
    if c.get('lam_ar37') not in (None, ''):
        add('37Ar decay constant (1/day)', _g(c.get('lam_ar37')), '')
    if c.get('lam_ar39') not in (None, ''):
        add('39Ar decay constant (1/day)', _g(c.get('lam_ar39')), '')
    if c.get('lam_cl36') not in (None, ''):
        add('36Cl decay constant (1/day)', _g(c.get('lam_cl36')), '')

    est = irradiation_estimate(metas)
    if est:
        add('Irradiation reference time', est,
            'Implied by the 37Ar decay correction of each step (one value per '
            'irradiation); the days below are counted from it')
    for key, name in (('ca_39_37', 'Ca: 39Ar/37Ar'), ('ca_38_37', 'Ca: 38Ar/37Ar'),
                      ('ca_36_37', 'Ca: 36Ar/37Ar'), ('k_38_39', 'K: 38Ar/39Ar'),
                      ('k_40_39', 'K: 40Ar/39Ar'), ('k_37_39', 'K: 37Ar/39Ar'),
                      ('p36cl_38cl', '36Cl/38Cl production ratio')):
        if c.get(key) not in (None, ''):
            add('Interference: ' + name, _g(c.get(key)), '')
            if c.get(key + '_err') not in (None, ''):
                add('  ± (1σ)', _g(c.get(key + '_err')), '')
    if c.get('mdf') not in (None, ''):
        add('Mass discrimination (MDF)', _g(c.get('mdf'), 9),
            '± ' + _g(c.get('mdf_err')) if c.get('mdf_err') not in (None, '') else '')
    _labs = labs_of(metas, c)
    if _labs:
        add('Data source', ' + '.join(LAB_NAMES[x] for x in _labs),
            'Each source has its own "Raw signals" sheet with its own columns and units')
    if 'arcalc' in _labs or c.get('signal_unit'):
        _fu = unit_text(c.get('signal_unit'))
        add('Signal unit (ArArCALC)', ARARCALC_UNIT or _fu or 'not stated',
            ('set manually; the file\'s "Sensitivity Unit" reads mol/' + _fu.replace('V', 'volt')
             if ARARCALC_UNIT and _fu and _fu != ARARCALC_UNIT
             else 'as stated by the ArArCALC file ("Sensitivity Unit")'))
    if 'massspec' in _labs:
        add('Signal unit (Mass Spec)', MASSSPEC_UNIT or 'not stated in file',
            'set manually; the Mass Spec file does not state a unit' if MASSSPEC_UNIT
            else 'the Mass Spec file does not state a unit')
    if c.get('sensitivity') not in (None, ''):
        add('Sensitivity (mol per signal unit)', _g(c.get('sensitivity'), 6), '')
    if c.get('__varies__'):
        _nm = {'lam_eps': 'λ ε', 'lam_beta': 'λ β-', 'k40_abund': '40K abundance',
               'lam_ar37': '37Ar decay constant', 'lam_ar39': '39Ar decay constant'}
        add('Not constant within this irradiation',
            ', '.join(_nm.get(k, str(k)) for k in c['__varies__']),
            'these source-file values differ between aliquots and are left out')
    for r in extra:
        rows.append(list(r))
    return rows


def group_label(constants):
    c = constants or {}
    return str(c.get('irradiation') or c.get('irrad_label') or 'Irradiation')


def constants_groups(entries):
    """Collect (cfg, constants, metas) entries by irradiation.

    entries: iterable of (cfg, constants, metas). Entries of one irradiation
    share one column in the constants box; the metas of that irradiation are
    pooled for its irradiation-time estimate. Returns a list of
    (label, cfg, constants, metas), in order of first appearance.
    """
    out, varies = {}, {}
    for cfg, consts, metas in entries:
        key = group_label(consts)
        if key not in out:
            out[key] = [cfg, dict(consts or {}), list(metas)]
            varies[key] = set()
        else:
            out[key][2].extend(metas)
            for k, v in (consts or {}).items():
                if k not in out[key][1]:
                    out[key][1][k] = v
                elif (not _same(out[key][1][k], v) and isinstance(v, (int, float))
                      and not str(k).startswith('file_') and k not in _DESCRIPTIVE):
                    varies[key].add(k)
    for key, bad in varies.items():
        for k in bad:
            out[key][1].pop(k, None)
        if bad:
            out[key][1]['__varies__'] = sorted(bad)
    return [(k, v[0], v[1], v[2]) for k, v in out.items()]


_DESCRIPTIVE = {'mass_spec_system', 'mass_spec_file_version'}


def _same(a, b):
    try:
        a, b = float(a), float(b)
        return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-300)
    except (TypeError, ValueError):
        return a == b


def constants_matrix(groups, atm=None):
    """Rows [name, value-per-group..., note] for the constants box.

    Rows are matched by name and, for repeated names ("  ± (1σ)"), by
    occurrence, so every group's values stay on the right line.
    """
    cols = []
    for _l, cfg, consts, metas in groups:
        seen, d = {}, {}
        for name, v, note in constants_rows(cfg, consts, metas, atm=atm):
            n = seen.get(name, 0)
            seen[name] = n + 1
            d[(name, n)] = (v, note)
        cols.append(d)
    keys = []
    for d in cols:
        for k in d:
            if k not in keys:
                keys.append(k)
    out = []
    for k in keys:
        note = next((d[k][1] for d in cols if k in d and d[k][1]), '')
        out.append([k[0]] + [d[k][0] if k in d else '' for d in cols] + [note])
    return out


def write_constants(w, cfg=None, constants=None, metas=(), atm=None, extra=(),
                    title='CONSTANTS', groups=None):
    """Write the constants box to a csv writer, followed by a blank row.

    One value column per irradiation: pass *groups* (from constants_groups),
    or a single cfg / constants / metas for the one-irradiation case.
    """
    if groups is None:
        groups = [(group_label(constants), cfg, constants, list(metas))]
    w.writerow([title])
    w.writerow(['Constant'] + [g[0] for g in groups] + ['Note'])
    for r in constants_matrix(groups, atm=atm):
        w.writerow(r)
    for r in extra:
        w.writerow(list(r))
    w.writerow([])


# ---------------------------------------------------------------------------
# raw-signal tables: one sheet per lab, with that lab's own columns
# ---------------------------------------------------------------------------
#
# The two source formats store the signals differently, so they are not mixed
# in one table:
#
#   ArArCALC   the blank-corrected intercept of every isotope with its
#              uncertainty, in the file's unit or ARARCALC_UNIT (37Ar and 39Ar
#              also decay-corrected; MDF is a separate column, not applied);
#   Mass Spec  the baseline/blank-corrected signal of every isotope with its
#              uncertainty, the discrimination factors and the argon moles.
#              The file states no signal unit.
# 37Ar and 39Ar are decay-corrected in both, so the columns can be divided
# directly (40/39, 40/36, ...); the decay factors stay in the sheet.
#
# Every header carries the unit, or says that none is stated.

_ISO = (36, 37, 38, 39, 40)
LAB_NAMES = {'arcalc': 'ArArCALC', 'massspec': 'Mass Spec', '': 'Unknown source'}

_UNIT_TXT = {'volt': 'V', 'volts': 'V', 'v': 'V', 'fa': 'fA', 'pa': 'pA',
             'na': 'nA', 'a': 'A', 'amp': 'A', 'amps': 'A', 'mol': 'mol',
             'count': 'counts', 'counts': 'counts', 'cps': 'cps'}


# Signal units written in the exports.  Empty means "as the source file
# states": the ArArCALC "Sensitivity Unit" (mol/volt -> V), and none for Mass
# Spec, which does not state one.  Set a unit here (e.g. 'fA') to override it
# for a lab whose files are labelled differently from what they hold.
ARARCALC_UNIT = ''
MASSSPEC_UNIT = ''


def unit_text(u):
    """Display form of a unit string taken from a source file ('volt' -> 'V')."""
    if u in (None, ''):
        return ''
    u = str(u).strip()
    return _UNIT_TXT.get(u.lower(), u)


def detect_lab(raw, constants=None):
    """'arcalc' or 'massspec' for one step's raw dict, '' when not known.

    ArArCALC gives the intercept and the blank of each isotope; the Mass Spec
    file gives not-decay-corrected signals, discrimination/IC factors and
    moles. The constants block is the fallback for bare signals."""
    raw = raw or {}
    ks = set(raw)
    if any(k.endswith(('_bkgd', '_bsln')) for k in ks):
        return 'arcalc'
    if any(k.endswith(('_nd', '_disc', '_ic', '_mol')) for k in ks):
        return 'massspec'
    c = constants or {}
    if c.get('signal_unit') or c.get('age_equation'):
        return 'arcalc'
    if c.get('mass_spec_system') not in (None, '') or c.get('mass_spec_file_version') not in (None, ''):
        return 'massspec'
    return ''


def labs_of(metas, constants=None):
    """Distinct lab keys of a list of step_meta() dicts, in order, ignoring ''."""
    out = []
    for m in metas or []:
        lab = detect_lab((m or {}).get('raw'), constants)
        if lab and lab not in out:
            out.append(lab)
    return out


def _v(raw, key):
    x = raw.get(key)
    return None if x in (None, '') else x


def _arc_signal(r, n, err):
    """ArArCALC signal: intercept minus blank (as the file gives them), and for
    37Ar and 39Ar multiplied by the step's decay factor (taken as exact).
    Blank when the decay factor is missing, so a value is never mislabelled."""
    x = _v(r, f'ar{n}_err' if err else f'ar{n}')
    if x is None or n not in (37, 39):
        return x
    try:
        return float(x) * float(_v(r, f'decay{n}'))
    except (TypeError, ValueError):
        return None


def _arcalc_columns(u):
    """(header, getter, note) for the ArArCALC sheet. u: unit text or ''."""
    U = f'[{u}]' if u else '[unit not stated in file]'
    cols = []
    for n in _ISO:
        dc = n in (37, 39)
        cols += [
            (f'{n}Ar {U}', lambda r, n=n: _arc_signal(r, n, False),
             f'{n}Ar: intercept minus blank' + (', multiplied by the %dAr decay factor' % n if dc else
                                                 ' (no decay correction applies)')),
            (f'{n}Ar ±1σ {U}', lambda r, n=n: _arc_signal(r, n, True),
             'Intercept and blank uncertainties combined in quadrature'
             + ('; scaled by the decay factor' if dc else '')),
        ]
    cols += [
        ('37Ar decay factor', lambda r: _v(r, 'decay37'),
         'Factor that corrects 37Ar for decay since irradiation (dimensionless)'),
        ('39Ar decay factor', lambda r: _v(r, 'decay39'),
         'Factor that corrects 39Ar for decay since irradiation (dimensionless)'),
        ('Mass discrimination (MDF)', lambda r: _v(r, 'mdf'),
         'Mass-discrimination factor of the aliquot (dimensionless). Not applied to the signals above: '
         'ArArCALC applies it to its ratios'),
        (f'Sensitivity [mol per {u}]' if u else 'Sensitivity [mol per signal unit]',
         lambda r: _v(r, 'sens'), 'Moles of argon per unit of signal, as stated in the file'),
    ]
    return cols


def _molratio(r):
    """mol per signal unit implied by the file: 40Ar moles / 40Ar signal."""
    try:
        m, s = float(r.get('ar40_mol')), float(r.get('ar40'))
        if math.isfinite(m) and math.isfinite(s) and s != 0 and m / s > 0:
            return m / s
    except (TypeError, ValueError):
        pass
    return None


def _ic_if_used(r, n):
    v = _v(r, f'ar{n}_ic')
    try:
        return None if v is None or float(v) == 1.0 else v
    except (TypeError, ValueError):
        return v


def _massspec_columns(u=''):
    U = f'[{u}]' if u else '[file units]'
    cols = []
    for n in _ISO:
        # the plain ArN_ column of the mass-spec file is already decay-corrected
        # for 37Ar and 39Ar (it equals ArN_notDecayCor x the decay factor)
        dc = n in (37, 39)
        cols += [(f'{n}Ar {U}', lambda r, n=n: _v(r, f'ar{n}'),
                  f'Ar{n}_: {n}Ar after baseline and blank' + (', decay-corrected' if dc else '')),
                 (f'{n}Ar ±1σ {U}', lambda r, n=n: _v(r, f'ar{n}_err'),
                  'its uncertainty')]
    for n in _ISO:
        cols.append((f'{n}Ar mass-discrimination factor', lambda r, n=n: _v(r, f'ar{n}_disc'),
                     f'Ar{n}_Disc (dimensionless)'))
    for n in _ISO:
        cols.append((f'{n}Ar IC factor', lambda r, n=n: _ic_if_used(r, n),
                     f'Ar{n}_ICFactor (dimensionless); left out when every value is 1'))
    cols += [
        ('39Ar [mol]', lambda r: _v(r, 'ar39_mol'), 'Ar39_Moles'),
        ('40Ar [mol]', lambda r: _v(r, 'ar40_mol'), 'Ar40_Moles'),
        (f'mol per {u or "signal unit"} (40Ar mol ÷ 40Ar signal, approx.)', _molratio,
         'Computed here, not read from the file: what one signal unit is worth in moles, as a check on the unit.'),
        ('37Ar decay factor', lambda r: _v(r, 'decay37'), '37_Decay (dimensionless)'),
        ('39Ar decay factor', lambda r: _v(r, 'decay39'), '39_Decay (dimensionless)'),
    ]
    return cols


def _legacy_columns():
    cols = []
    for n in _ISO:
        cols.append((f'{n}Ar signal', lambda r, n=n: _v(r, f'ar{n}'),
                     f'{n}Ar signal, unit not known'))
    cols += [('37Ar decay factor', lambda r: _v(r, 'decay37'), ''),
             ('39Ar decay factor', lambda r: _v(r, 'decay39'), ''),
             ('Mass discrimination (MDF)', lambda r: _v(r, 'mdf'), ''),
             ('Sensitivity [mol per signal unit]', lambda r: _v(r, 'sens'), '')]
    return cols


ID_HEADERS = ['Sample', 'Step', 'RunID', 'Run date/time', 'Days since irradiation']


def raw_tables(items, include_empty=False):
    """The raw-signal sheets, one per lab (and signal unit).

    items: iterable of (sample title, steps, metas[, constants]) where *steps*
    are the step dicts (label, runID ...), *metas* the matching step_meta()
    dicts and *constants* the sample's constants block (used for the unit the
    ArArCALC file states). Returns a list of dicts
    {sheet, lab, unit, header, rows, notes}; a group's columns that nobody has a
    value for are left out, and sources with no raw signals give no sheet."""
    groups = {}
    order = []
    for it in items:
        title, steps, metas = it[0], it[1], it[2]
        consts = it[3] if len(it) > 3 else None
        c = consts or {}
        for s, m in zip(steps, metas):
            raw = (m or {}).get('raw') or {}
            if not raw and not include_empty:
                continue
            lab = detect_lab(raw, c)
            unit = ((ARARCALC_UNIT or unit_text(c.get('signal_unit'))) if lab == 'arcalc'
                    else MASSSPEC_UNIT if lab == 'massspec' else '')
            key = (lab, unit)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((title, s, m))
    out = []
    for key in order:
        lab, unit = key
        cols = (_arcalc_columns(unit) if lab == 'arcalc'
                else _massspec_columns(MASSSPEC_UNIT) if lab == 'massspec' else _legacy_columns())
        recs = groups[key]
        used = [c for c in cols
                if include_empty or any(c[1](m.get('raw') or {}) is not None for _t, _s, m in recs)]
        header = ID_HEADERS + [c[0] for c in used]
        rows = []
        for title, s, m in recs:
            rid = s.get('runID') or s.get('run_id') or ''
            row = [title, s.get('label', ''), rid, m.get('run_dt', ''), fmt_days(m)]
            for _h, fn, _n in used:
                v = fn(m.get('raw') or {})
                row.append('' if v is None else _g(v, 9))
            rows.append(row)
        out.append({'lab': lab, 'unit': unit, 'header': header, 'rows': rows,
                    'notes': [(c[0], c[2]) for c in used if c[2]]})
    # sheet names: lab; add the unit when one lab has several
    labs_n = {}
    for g in out:
        labs_n[g['lab']] = labs_n.get(g['lab'], 0) + 1
    for g in out:
        nm = 'Raw signals - ' + LAB_NAMES.get(g['lab'], 'Unknown source')
        if labs_n[g['lab']] > 1 and g['unit']:
            nm += f" ({g['unit']})"
        g['sheet'] = re.sub(r'[\[\]:*?/\\]', '_', nm)[:31]
    return out


# ---------------------------------------------------------------------------
# notes sheet
# ---------------------------------------------------------------------------

RESULTS_NOTE = 'The step table, plateau and isochron results of each sample.'


def notes_rows(raws=None, results_note=None):
    """Rows of the Notes sheet. raws: the raw_tables() groups written;
    results_note: what the Results sheet holds (RESULTS_NOTE when not given)."""
    raws = raws or []
    rows = [
        ['ABOUT THIS FILE'],
        ['Sheet "Results"', results_note or RESULTS_NOTE],
        ['Sheet "Constants"', 'Every constant behind the numbers (atmospheric argon, decay constants, '
                              'FC age, interferences, irradiation time, data source and signal unit).'],
    ]
    for g in raws:
        rows.append([f'Sheet "{g["sheet"]}"',
                     'The measured argon signals of the steps from ' + LAB_NAMES.get(g['lab'], 'an unknown source')
                     + (f' (unit: {g["unit"]})' if g['unit'] else '')
                     + ', exactly as that source stores them. One row per step; "Sample", "Step" and '
                       '"RunID" match the Results sheet.'])
    rows += [
        [],
        ['CONVENTIONS'],
        ['Days since irradiation', 'Time from the irradiation reference time (see Constants) to the '
                                   'analysis, in days. It is derived from the 37Ar decay correction '
                                   'applied by the source software: days = ln(37Ar decay factor) / λ37. '
                                   'The same value follows from the 39Ar decay factor.'],
        ['Run date/time', 'Date and hour of the analysis as written in the source file '
                          '(lab clock; only differences are meaningful here).'],
    ]
    if raws:
        rows += [
            ['Signal units', 'The unit is in every header of the raw sheets: the one the source file '
                             'states, or the one set for that source (see Constants). No unit '
                             'conversion is made.'],
            ['Signals', 'Every signal column in the raw sheets is blank-corrected (baseline and blank '
                        'subtracted). 37Ar and 39Ar are in addition decay-corrected; 36Ar, '
                        '38Ar and 40Ar need no decay correction. Mass Spec: the file\'s own corrected signals, which '
                        'reproduce the ratio columns of the file. ArArCALC: intercept minus blank, with 37Ar and '
                        '39Ar multiplied by their decay factors. Mass discrimination (MDF) and the interference '
                        '(Ca, K, Cl) corrections are not applied to the ArArCALC signals; ArArCALC applies MDF to '
                        'its ratios and adds a 36Ar collector calibration that its export does not contain, so a '
                        'ratio of these signals is not the same as ArArCALC\'s corrected ratio.'],
        ]
    rows += [
        ['FC age, decay constant', 'Used to calculate the ages when the Age source is R mode. Imported ages '
                                   'were calculated by the source software with its own constants.'],
        ['R standard', 'The monitor 40Ar*/39ArK taken from the source file. Where the ArArCALC export '
                       'gives none (0) it is recomputed from the file\'s J, FC age and decay constant as '
                       '(e^(λt) - 1) / J.'],
        ['Uncertainties', 'Errors in the step table are 1σ unless the heading says otherwise.'],
        ['Empty cell', 'The source file did not contain that value.'],
        [],
    ]
    for g in raws:
        rows.append([f'COLUMNS OF "{g["sheet"]}"'.upper()])
        rows.append(['Column', 'Meaning'])
        for head, note in g['notes']:
            rows.append([head, note])
        rows.append([])
    return rows


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

_NUM = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]\d+)?$')
_SECTIONS = ('CONSTANTS', 'STEP DATA', 'PLATEAU RESULTS', 'ISOCHRON RESULTS',
             'FLUX REGRESSION RESULTS', 'COMBINED PLATEAU RESULTS',
             'STACKED ISOCHRON RESULTS', 'GLS + ML COVARIANCE MATRICES',
             'SAMPLE RESULTS', 'POINT DATA', 'POOLED RESULT', 'RAW SIGNALS',
             'RECALCULATED AGES', 'COMBINED AGE',
             'ABOUT THIS FILE', 'CONVENTIONS', 'COLUMNS OF', 'MIXTURE')


def _cell(v):
    if isinstance(v, str):
        s = v.strip()
        if s and _NUM.match(s) and not (len(s) > 1 and s[0] == '0' and s[1].isdigit()):
            try:
                f = float(s)
                if math.isfinite(f):
                    return f
            except ValueError:
                pass
    return v


def _write_sheet(ws, rows, bold_font, widths_cap=44):
    prev_title = False
    widths = {}
    for r, row in enumerate(rows, start=1):
        is_title = bool(row) and isinstance(row[0], str) and \
            any(row[0].startswith(t) for t in _SECTIONS)
        for c, v in enumerate(row, start=1):
            cell = ws.cell(row=r, column=c, value=_cell(v))
            if is_title or prev_title:
                cell.font = bold_font
            n = len(str(v)) if v not in (None, '') else 0
            widths[c] = max(widths.get(c, 8), min(n + 2, widths_cap))
        prev_title = is_title
    for c, wd in widths.items():
        ws.column_dimensions[ws.cell(row=1, column=c).column_letter].width = wd


def split_constants(rows):
    """(constants rows, remaining rows): the CONSTANTS block (title row up to
    its closing blank row) is taken out of the results rows."""
    for i, r in enumerate(rows):
        if r and r[0] == 'CONSTANTS':
            j = i
            while j < len(rows) and any(str(c) != '' for c in rows[j]):
                j += 1
            rest = rows[:i] + rows[j + 1:]
            return rows[i:j], rest
    return [], rows


def write_xlsx(path, results_rows, raws=None, notes=None, constants=None):
    """results_rows: rows of the Results sheet; constants: rows of the
    Constants sheet; raws: raw_tables() groups (one sheet each); notes: rows."""
    from openpyxl import Workbook
    from openpyxl.styles import Font
    bold = Font(bold=True)
    wb = Workbook()
    ws = wb.active
    ws.title = 'Results'
    _write_sheet(ws, results_rows, bold)
    if constants:
        wc = wb.create_sheet('Constants')
        _write_sheet(wc, constants, bold)
    for g in raws or []:
        w2 = wb.create_sheet(g['sheet'])
        _write_sheet(w2, [['RAW SIGNALS - ' + LAB_NAMES.get(g['lab'], 'Unknown source').upper()
                           + (f' [{g["unit"]}]' if g['unit'] else '')], g['header']] + g['rows'], bold)
        w2.freeze_panes = 'F3'
    if notes:
        ws3 = wb.create_sheet('Notes')
        _write_sheet(ws3, notes, bold)
    wb.save(path)


class TableSink:
    """Context manager used where the code had ``open(path, 'w', ...)``.

        sink = TableSink(path)
        with sink as f:
            w = csv.writer(f)
            ...
            sink.set_raw_items(items)   # before leaving the block

    A .xlsx path gets a Results sheet (what was written, without the constants
    block), a Constants sheet, one raw-signal sheet per lab and a Notes sheet;
    any other path is a plain .csv, with the constants first and the raw
    tables appended as sections.
    """

    def __init__(self, path, results_note=None):
        self.path = path
        self.results_note = results_note
        self.is_xlsx = str(path).lower().endswith('.xlsx')
        self.raws = []
        self.raw_missing = False
        self._f = None

    def set_raw_items(self, items):
        # Files saved before the raw signals were stored have nothing to show:
        # leave the sheets out rather than export tables of empty cells.
        self.raws = raw_tables(items)
        self.raw_missing = not self.raws

    def __enter__(self):
        if self.is_xlsx:
            self._f = io.StringIO(newline='')
        else:
            self._f = open(self.path, 'w', newline='', encoding='utf-8-sig')
        return self._f

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None:
                return False
            if self.is_xlsx:
                self._f.seek(0)
                rows = list(csv.reader(self._f))
                consts, rows = split_constants(rows)
                _notes = notes_rows(self.raws, self.results_note)
                if self.raw_missing:
                    _notes.append(['No raw-signal sheet',
                                   'The loaded data carry no raw signals: typed or pasted values, '
                                   'or files converted before raw signals were stored in them.'])
                write_xlsx(self.path, rows, self.raws, _notes, consts)
            elif self.raws:
                w = csv.writer(self._f)
                for g in self.raws:
                    w.writerow([])
                    w.writerow(['RAW SIGNALS - ' + LAB_NAMES.get(g['lab'], 'Unknown source').upper()
                                + (f' [{g["unit"]}]' if g['unit'] else '')])
                    w.writerow(g['header'])
                    for r in g['rows']:
                        w.writerow(r)
        finally:
            self._f.close()
        return False


def save_filter():
    return 'Excel workbook (*.xlsx);;CSV (*.csv);;All files (*)'


def ensure_ext(path, selected_filter=''):
    """Add .xlsx / .csv when the dialog returned a name without one."""
    if os.path.splitext(path)[1]:
        return path
    return path + ('.csv' if 'csv' in (selected_filter or '').lower() else '.xlsx')


FLUX_HEADERS = ['J', '±1σ J', 'R standard', '±1σ R standard']


def flux_cells(cfg, j_val=None, j_err=None, show_r=None, r_val=None, r_err=None):
    """[J, ±1σ J, R standard, ±1σ R standard] for one step / sample row.

    j_val / j_err (1σ, absolute) come from the step when it has them, else
    from the cfg: the J its ages are computed with (arar_math.cfg_J, so in
    R-standard mode the J of the R standard at the current calibration, never
    a stored jVal that went stale), else the stored jVal.  R standard is left
    blank in ratio_J mode, where it is unused.
    """
    def _num(v):
        try:
            v = float(v)
            return v if math.isfinite(v) and v > 0 else None
        except (TypeError, ValueError):
            return None
    j = _num(j_val)
    je = _num(j_err)
    if j is None:
        jc, rel = cfg_J(cfg)
        if jc:
            j = jc
            if je is None and rel > 0:
                je = jc * rel
        elif cfg.get('ageFluxMode') != 'ratio_R_FCs':    # R_unk/R_FCs uses no J
            j = _num(cfg.get('jVal'))
    if je is None and j is not None and _num(cfg.get('jErrPct')):
        je = j * float(cfg['jErrPct']) / 100.0
    r = e = None
    if show_r is None:
        show_r = cfg.get('ageFluxMode') != 'ratio_J'
    if _num(r_val) is not None:
        # R given explicitly (the matched disk-flux hole's monitor ratio)
        r, e = _num(r_val), _num(r_err)
    elif show_r:
        r = _num(cfg.get('rStd'))
        pct = _num(cfg.get('rStdErrPct'))
        if r is not None and pct is not None:
            e = pct if cfg.get('rStdErrMode', 'pct') != 'pct' else r * pct / 100.0
    return [_g(j, 8), _g(je, 6), _g(r, 8), _g(e, 6)]


# ===========================================================================
# FIGURE EXPORT
# ===========================================================================


CM = 1.0 / 2.54          # cm → inch
CM_PER_IN = 2.54

LEGEND_MODES = [
    ('auto',     'As drawn'),
    ('inside',   'Inside, upper right'),
    ('inside-ul', 'Inside, upper left'),
    ('inside-lr', 'Inside, lower right'),
    ('inside-ll', 'Inside, lower left'),
    ('outside',  'Outside, right'),
    ('above',    'Above the plot'),
    ('below',    'Below the plot'),
    ('off',      'No legend'),
]

DEFAULT_SPEC = dict(
    width_cm=17.0,
    height_cm=19.0,
    dpi=300,
    fmt='pdf',
    font_scale=1.0,
    line_scale=1.0,
    marker_scale=1.0,
    legend='auto',
    legend_cols=1,
    transparent=False,
    p2_width_cm=None,   # second page (e.g. the isochron): its own size,
    p2_height_cm=None,  # None = same as page 1
    p2_font_scale=None,     # ... and its own text / line / marker scaling
    p2_line_scale=None,     # and legend; None = same as page 1
    p2_marker_scale=None,
    p2_legend=None,
    p2_legend_cols=None,
    panels=[],          # [{'id': str, 'on': bool, 'weight': float}, ...]
    options={},         # host-declared switches, by option id
)


class Panel:
    """One stackable panel offered to the user.

    id      key the host recognises in spec['panels']
    label   shown in the dialog
    on      default state
    weight  default height weight (relative, like gridspec height_ratios)
    fixed   True = always included, the checkbox is disabled
    """

    def __init__(self, id, label, on=True, weight=1.0, fixed=False):
        self.id, self.label, self.on, self.weight, self.fixed = \
            id, label, on, weight, fixed


class Option:
    """One host-declared switch.

    kind 'bool'   -> checkbox,      default True/False
    kind 'choice' -> combo box,     values = [(value, label), ...]
    kind 'float'  -> spin box,      lo/hi/step
    kind 'text'   -> line edit,     free text (titles, axis limits, names)
    group: the box it appears in ('Contents' by default); groups are shown in
    the order the options declare them.
    depends: (option_id, value) — the row is only enabled when that option has
    that value (or, for a bool, is ticked).
    page: 0 = shown on the page-1 tab, 1 = on the page-2 tab (only used when
    the host supplies a second figure).
    follows: (option_id, {value: field value, ...}) — while that option has
    one of those values, the field shows the matching value: when the dialog
    opens, when a preset is applied and when that option is switched.  Any
    other value of that option (e.g. 'typed below') leaves the field as it is,
    so its text can be edited first and then put to use; a quick switch back
    and forth does not lose what was typed.  Used for the name fields, which
    show the main-window or the experiment names until 'typed' is chosen.
    ident: what the field belongs to (e.g. a sample's experiment name), for
    fields that come one per sample/aliquot.  Presets and the last-used
    settings store values against it, so a name goes back to the same sample
    even when the samples are in another order or some are not shown.
    """

    def __init__(self, id, label, kind='bool', default=None, values=None,
                 lo=0.0, hi=100.0, step=1.0, depends=None, tip='',
                 group='Contents', placeholder='', width=0, wide=False, page=0,
                 follows=None, ident=None):
        self.id, self.label, self.kind = id, label, kind
        self.page = page
        self.follows = follows
        self.ident = ident
        self.values = values or []
        self.lo, self.hi, self.step = lo, hi, step
        self.depends, self.tip = depends, tip
        self.group, self.placeholder, self.width = group, placeholder, width
        self.wide = wide
        if default is None:
            default = self.values[0][0] if self.values else (
                '' if kind == 'text' else False)
        self.default = default


# --------------------------------------------------------------------------
# Preset storage
# --------------------------------------------------------------------------

def _preset_path():
    return dfold.config_path('figure_export_presets.json')


def _load_store():
    try:
        with open(_preset_path(), encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_store(store):
    """Write the store through a temporary file, so a crash mid-write cannot
    leave a half-written file (which would read back as empty)."""
    path = _preset_path()
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(store, f, indent=1)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def _update_store(change):
    """Read the file afresh, apply change(store) and write it back, so a
    preset saved meanwhile by another tool or dialog is not overwritten by the
    copy this dialog read when it opened.  Returns (store, ok)."""
    store = _load_store()
    change(store)
    return store, _save_store(store)


# Figure types whose presets are shared: a preset saved for one of them is
# offered in all, and applies the settings they have in common (page size,
# text, legend, panels ...).  The first key holds the shared list.
PRESET_GROUPS = (
    ('plateau', 'plateau_stack', 'plateau_all_samples', 'plateau_isochron'),
    ('ageprob', 'ageprob_mixture'),
)
KIND_LABELS = {'plateau': 'single', 'plateau_stack': 'stack',
               'plateau_all_samples': 'all samples', 'plateau_isochron': 'isochron',
               'ageprob': 'distribution', 'ageprob_mixture': 'mixture'}


def _preset_group(host_key):
    for g in PRESET_GROUPS:
        if host_key in g:
            return g
    return (host_key,)


def _gather_presets(store, group):
    """Move the presets of every key in *group* into the first one, tagging
    each with the figure type it was saved from.  Returns True if anything
    moved."""
    shared = store.setdefault(group[0], {}).setdefault('presets', {})
    moved = False
    for key in group:
        for name, pres in list(store.get(key, {}).get('presets', {}).items()):
            pres.setdefault('kind', key)
            if key == group[0]:
                continue
            new, k = name, 1
            while new in shared:
                k += 1
                new = f"{name} ({KIND_LABELS.get(key, key)}{'' if k == 2 else f' {k}'})"
            shared[new] = pres
            del store[key]['presets'][name]
            moved = True
        if key != group[0] and key in store and not store[key].get('presets'):
            store[key].pop('presets', None)
    return moved


def default_spec(panels, options):
    """A fresh spec for a host's panel/option declaration."""
    spec = copy.deepcopy(DEFAULT_SPEC)
    spec['panels'] = [dict(id=p.id, on=p.on, weight=p.weight) for p in panels]
    spec['options'] = {o.id: o.default for o in options}
    return spec


def _merge(spec, panels, options):
    """Bring a stored spec up to date with the host's current declaration."""
    base = default_spec(panels, options)
    out = copy.deepcopy(base)
    for k, v in (spec or {}).items():
        if k in ('panels', 'options'):
            continue
        if k in out:
            out[k] = v
    stored = {p.get('id'): p for p in (spec or {}).get('panels', [])}
    for p in out['panels']:
        s = stored.get(p['id'])
        if s:
            p['on'] = bool(s.get('on', p['on']))
            p['weight'] = float(s.get('weight', p['weight']))
    order = [p.get('id') for p in (spec or {}).get('panels', [])]
    out['panels'].sort(key=lambda p: order.index(p['id'])
                       if p['id'] in order else 99)
    stored = (spec or {}).get('options', {})
    for k, v in stored.items():
        if k in out['options']:
            out['options'][k] = v
    # settings saved before the names choice replaced the tickbox
    if ('names_from' in out['options'] and 'names_from' not in stored
            and 'names_from_window' in stored):
        out['options']['names_from'] = ('window' if stored['names_from_window']
                                        else 'typed')
    # per-sample fields go back to the sample they were saved for, not to
    # whichever sample now has the same position
    idents = (spec or {}).get('idents') or {}
    if idents:
        by_ident = {idents[k]: stored[k] for k in idents if k in stored}
        for o in options:
            if getattr(o, 'ident', None) is not None:
                out['options'][o.id] = by_ident.get(o.ident, o.default)
    for o in options:
        fl = getattr(o, 'follows', None)
        if fl and out['options'].get(fl[0]) in fl[1]:
            out['options'][o.id] = fl[1][out['options'][fl[0]]]
    for p in out['panels']:
        decl = next((d for d in panels if d.id == p['id']), None)
        if decl is not None and decl.fixed:
            p['on'] = True
    return out


# --------------------------------------------------------------------------
# Helpers a host can use inside its render(fig, spec)
# --------------------------------------------------------------------------

def active_panels(spec):
    """[(id, weight), ...] for the panels that are switched on, in order."""
    return [(p['id'], float(p['weight'])) for p in spec.get('panels', [])
            if p.get('on')]


def scale_text(fig, font_scale=1.0, line_scale=1.0, marker_scale=1.0):
    """Multiply every text size, line width and marker size already drawn.

    Applied after the figure is drawn, so a host does not have to thread a
    font size through its drawing code.
    """
    if font_scale != 1.0:
        for t in fig.findobj(_mtext.Text):
            try:
                t.set_fontsize(t.get_fontsize() * font_scale)
            except Exception:
                pass
    if line_scale != 1.0 or marker_scale != 1.0:
        for ln in fig.findobj(_mlines.Line2D):
            try:
                if line_scale != 1.0:
                    ln.set_linewidth(ln.get_linewidth() * line_scale)
                if marker_scale != 1.0 and ln.get_marker() not in (None, 'None', ''):
                    ln.set_markersize(ln.get_markersize() * marker_scale)
            except Exception:
                pass
    if line_scale != 1.0:
        for ax in fig.axes:
            for sp in ax.spines.values():
                sp.set_linewidth(sp.get_linewidth() * line_scale)
            ax.tick_params(width=0.8 * line_scale, length=3.5 * line_scale)
    fit_texts(fig)


def fit_texts(fig, min_scale=0.55):
    """Keep every label that sits inside a panel inside that panel.

    Large text (Big text, Text x) can make a plateau label or a statistics
    box wider than the panel it is drawn in, and it would be cut off at the frame.
    A label is reduced in size to fit its panel; only one that would need to go
    below about 3/4 of its size is first split in two at its triple-space gap,
    then reduced (to no less than *min_scale* of its size); one
    that merely crosses an edge is moved back inside.
    """
    from matplotlib.text import Annotation
    renderer = getattr(fig.canvas, 'get_renderer', None)
    if renderer is None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        renderer = FigureCanvasAgg(fig).get_renderer
    renderer = renderer()
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        box = ax.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        pad_x, pad_y = 0.012 * box.width, 0.012 * box.height
        for t in list(ax.texts):
            if not t.get_visible() or not t.get_text().strip():
                continue
            try:
                bb = t.get_window_extent(renderer)
                if bb.width < 0.08 * box.width and bb.height < 0.3 * box.height:
                    continue                       # a small step label etc.
                if (bb.width * 0.75 > box.width - 2 * pad_x
                        and '   ' in t.get_text()
                        and '\n' not in t.get_text()):
                    t.set_text(t.get_text().replace('   ', '\n', 1))
                    bb = t.get_window_extent(renderer)
                k = min((box.width - 2 * pad_x) / max(bb.width, 1.0),
                        (box.height - 2 * pad_y) / max(bb.height, 1.0))
                if k < 1.0:
                    size0 = t.get_fontsize()
                    t.set_fontsize(max(size0 * k * 0.98, size0 * min_scale))
                    bb = t.get_window_extent(renderer)
                dx = dy = 0.0
                if bb.x0 < box.x0 + pad_x:
                    dx = box.x0 + pad_x - bb.x0
                elif bb.x1 > box.x1 - pad_x:
                    dx = box.x1 - pad_x - bb.x1
                if bb.y0 < box.y0 + pad_y:
                    dy = box.y0 + pad_y - bb.y0
                elif bb.y1 > box.y1 - pad_y:
                    dy = box.y1 - pad_y - bb.y1
                if (dx or dy) and not isinstance(t, Annotation):
                    tr = t.get_transform()
                    px, py = tr.transform(t.get_position())
                    t.set_position(tr.inverted().transform((px + dx, py + dy)))
            except Exception:
                pass


_LEGEND_LOC = {
    'inside':    dict(loc='upper right'),
    'inside-ul': dict(loc='upper left'),
    'inside-lr': dict(loc='lower right'),
    'inside-ll': dict(loc='lower left'),
    'outside':   dict(loc='upper left', bbox_to_anchor=(1.01, 1.0)),
    'above':     dict(loc='lower left', bbox_to_anchor=(0.0, 1.01)),
    'below':     dict(loc='upper left', bbox_to_anchor=(0.0, -0.08)),
}


def place_legend(ax, spec, **kw):
    """Re-place (or remove) the legend of *ax* according to the spec.

    Call it after the host has drawn its own legend; 'As drawn' leaves it
    alone.  Extra keyword arguments are passed to ax.legend().
    """
    mode = spec.get('legend', 'auto')
    if mode == 'auto':
        return ax.get_legend()
    leg = ax.get_legend()
    zorder = leg.get_zorder() if leg is not None else 20
    handles, labels = ax.get_legend_handles_labels()
    if leg is not None and not handles:
        # a legend built from proxy handles: read them back off the legend
        # (the attribute was renamed in matplotlib 3.9)
        handles = list(getattr(leg, 'legend_handles', None)
                       or getattr(leg, 'legendHandles', []))
        labels = [t.get_text() for t in leg.get_texts()]
    if mode == 'off':
        if leg is not None:
            leg.remove()
        return None
    if not handles:
        return None
    span = bool(kw.pop('span', False))
    keep_cols = bool(kw.pop('keep_cols', False))
    opts = dict(_LEGEND_LOC.get(mode, {'loc': 'upper right'}))
    opts['ncol'] = max(1, int(spec.get('legend_cols', 1)))
    if span and mode in ('below', 'above'):
        # one legend shared by several axes: centred under/over all of them
        opts['loc'] = 'upper center' if mode == 'below' else 'lower center'
        if int(spec.get('legend_cols', 1)) <= 1:
            if keep_cols and leg is not None:
                have = int(getattr(leg, '_ncols', getattr(leg, '_ncol', 1)) or 1)
                # a legend under the whole page has the width for more columns
                opts['ncol'] = max(have, min(3, -(-len(handles) // 8)))
            else:
                opts['ncol'] = max(1, min(4, -(-len(handles) // 5)))
    if leg is not None:
        prop = leg.get_texts()[0].get_fontproperties() if leg.get_texts() else None
        if prop is not None:
            opts.setdefault('prop', prop)
        opts.setdefault('framealpha', leg.get_frame().get_alpha() or 0.9)
        ttl = leg.get_title()
        if ttl.get_text():
            opts.setdefault('title', ttl.get_text())
            opts.setdefault('title_fontsize', ttl.get_fontsize())
        leg.remove()
    opts.update(kw)
    new = ax.legend(handles=handles, labels=labels, **opts)
    new.set_zorder(max(zorder, 20))      # above every data artist
    new._figexp_mode = mode              # settle_legends() finishes the placement
    new._figexp_span = span
    return new


def pad_names(names, size, gap=2.0, prop=None):
    """Pad *names* so that whatever follows them starts at the same x in
    the normal (proportional) legend font.

    Lets a legend list ages in one column without switching to a typewriter
    font: each name is measured at *size* points and topped up with spaces
    and hair spaces (U+200A, a tenth of an em) to the widest name plus *gap*
    spaces.  Digits are equal-width in the default font, so the numbers line
    up too.  Text scaled later (scale_text) keeps the alignment."""
    from matplotlib.textpath import TextToPath
    from matplotlib.font_manager import FontProperties
    fp = (prop.copy() if prop is not None else FontProperties())
    fp.set_size(size)
    t2p = TextToPath()

    def width(t):
        return t2p.get_text_width_height_descent(t, fp, ismath=False)[0] if t else 0.0
    sp, hs = width(' '), width('\u200a')
    ws = [width(n) for n in names]
    target = max(ws, default=0.0) + gap * sp
    out = []
    for n, w in zip(names, ws):
        d = max(target - w, 0.0)
        k = int(d // sp) if sp > 0 else 0
        h = int(round((d - k * sp) / hs)) if hs > 0 else 0
        out.append(n + ' ' * k + '\u200a' * h)
    return out


def settle_legends(fig):
    """Put legends placed above / below / right of the plot clear of every
    title, axis label and tick label in the figure.

    Those only have their final size once the host has drawn and scaled the
    text, so this runs just before the figure is drawn for the preview or
    saved; a fixed offset ran into the x-axis title as soon as the text grew.
    """
    legs = [(ax, ax.get_legend()) for ax in fig.axes]
    legs = [(ax, lg) for ax, lg in legs
            if lg is not None and getattr(lg, '_figexp_mode', None)
            in ('below', 'above', 'outside')]
    if not legs:
        return
    renderer = getattr(fig.canvas, 'get_renderer', None)
    if renderer is None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        renderer = FigureCanvasAgg(fig).get_renderer
    renderer = renderer()
    # extent of everything except these legends, in display units
    for _, lg in legs:
        lg.set_in_layout(False)
    try:
        boxes = [ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible()]
    finally:
        for _, lg in legs:
            lg.set_in_layout(True)
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    pad = 3.0 * fig.dpi / 72.0
    to_fig = fig.transFigure.inverted()
    for ax, lg in legs:
        pos = ax.get_window_extent(renderer)
        mode = lg._figexp_mode
        if getattr(lg, '_figexp_span', False) and mode in ('below', 'above'):
            wins = [a.get_window_extent(renderer) for a in fig.axes if a.get_visible()]
            cx = 0.5 * (min(b.x0 for b in wins) + max(b.x1 for b in wins))
            xy = (cx, (y0 - pad) if mode == 'below' else (y1 + pad))
        elif mode == 'below':
            xy = (pos.x0, y0 - pad)
        elif mode == 'above':
            xy = (pos.x0, y1 + pad)
        else:
            xy = (x1 + pad, pos.y1)
        lg.set_bbox_to_anchor(tuple(to_fig.transform(xy)), transform=fig.transFigure)


def page_size_cm(spec, page=0):
    """(width, height) in cm of page 1 (page=0) or page 2 (page=1)."""
    w = spec.get('width_cm', 17.0)
    h = spec.get('height_cm', 19.0)
    if page == 1:
        w = spec.get('p2_width_cm') or w
        h = spec.get('p2_height_cm') or h
    return float(w), float(h)


def figsize(spec, page=0):
    """(width, height) in inches for that page's size."""
    w, h = page_size_cm(spec, page)
    return max(2.0, w) * CM, max(2.0, h) * CM


# settings page 2 can override; None in the spec means "same as page 1"
_P2_KEYS = ('font_scale', 'line_scale', 'marker_scale', 'legend', 'legend_cols')


def page_spec(spec, page):
    """The spec as the renderer of *page* sees it: for page 2, its own text /
    line / marker scaling and legend replace page 1's where they are set."""
    if page != 1:
        return spec
    out = dict(spec)
    for k in _P2_KEYS:
        v = spec.get('p2_' + k)
        if v is not None:
            out[k] = v
    return out


# PDF/EPS text as embedded TrueType rather than Type 3, so the text stays
# text (selectable, editable) when a figure is opened in Illustrator,
# Inkscape or Affinity to be placed in a multi-panel layout.
EDITABLE_TEXT = {'pdf.fonttype': 42, 'ps.fonttype': 42}


def save_figure(fig, path, spec):
    """Write *fig* using the spec's dpi / transparency."""
    settle_legends(fig)
    with matplotlib.rc_context(EDITABLE_TEXT):
        fig.savefig(path, dpi=int(spec.get('dpi', 300)),
                    facecolor='none' if spec.get('transparent') else 'white',
                    edgecolor='none', transparent=bool(spec.get('transparent')),
                    bbox_inches='tight', pad_inches=0.05)


# --------------------------------------------------------------------------
# The dialog
# --------------------------------------------------------------------------

class FigureExportDialog(QDialog):
    """Layout + preview + save, shared by all the tools.

    host_key       key under which presets are stored ('plateau', 'ageprob', ...)
    panels         list of Panel
    options        list of Option
    render         callable(fig, spec); draws the figure
    default_name   suggested file name, without extension
    extra_render   optional callable(fig, spec) for a second figure (e.g. the
                   isochron on its own).  With the option named by extra_flag
                   on, it is written as <name>_<extra_suffix>.<ext>; with the
                   option named by extra_page_flag on it becomes page 2 of the
                   same PDF (PDF only).
    batch          optional callable(spec, path, progress) that writes many
                   figures in one go (e.g. every sample).  With it the dialog's
                   button becomes 'Export all…' and *render* only drives the
                   preview.  *path* is a PDF file when the option named by
                   batch_pdf_flag is on and the format is PDF, otherwise a
                   folder.  progress(i, n, label) updates the progress bar and
                   returns True when the user pressed Cancel.  Returns the
                   list of files written.
    defaults       spec values (page size, text scale ...) used instead of
                   DEFAULT_SPEC the first time, and on Reset.
    """

    FORMATS = [('pdf', 'PDF (vector)'), ('png', 'PNG'), ('svg', 'SVG (vector)'),
               ('eps', 'EPS (vector)'), ('tif', 'TIFF')]

    def __init__(self, parent, host_key, panels, options, render,
                 default_name='figure', extra_render=None, extra_flag=None,
                 extra_suffix='extra', extra_page_flag=None,
                 extra_label='Second figure', snapshot=None,
                 batch=None, batch_pdf_flag=None, defaults=None, title=None):
        super().__init__(parent)
        self.setWindowTitle(title or 'Export figure')
        self.setMinimumSize(1150, 720)
        self._key = host_key
        self._panels_decl = list(panels)
        self._options_decl = list(options)
        self._render = render
        self._extra_render = extra_render
        self._extra_flag = extra_flag
        self._extra_page_flag = extra_page_flag
        self._extra_suffix = extra_suffix
        self._extra_label = extra_label
        self._batch = batch
        self._batch_pdf_flag = batch_pdf_flag
        self._defaults = dict(defaults or {})
        # snapshot(spec) -> {option: value}: what blank 'auto' fields (axis
        # ranges) currently come out as, so a saved preset reproduces the
        # figure exactly instead of following whatever is on screen later
        self._snapshot = snapshot
        self._page = 0
        self._name = default_name or 'figure'
        self._store = _load_store()
        self._group = _preset_group(host_key)
        self._pkey = self._group[0]
        if _gather_presets(self._store, self._group):
            self._store, _ = _update_store(
                lambda st: _gather_presets(st, self._group))
        self._building = True

        host = self._store.get(host_key, {})
        self.spec = _merge(host.get('last'), panels, options)
        if host.get('last') is None:
            self.spec.update(self._defaults)

        self._build_ui()
        self._spec_to_ui()
        self._building = False

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._refresh_preview)
        self._queue()

    # ── construction ─────────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)

        # left: the controls, in a scroll area so small screens still work
        side = QWidget()
        col = QVBoxLayout(side)
        col.setContentsMargins(0, 0, 0, 0)

        self._opt_widgets = {}
        if self._extra_render is None:
            col.addWidget(self._panels_group())
            col.addWidget(self._page_group())
            for g in self._option_groups():
                col.addWidget(g)
            col.addStretch(1)
        else:
            # one tab per page: page 1's panels and switches, page 2's own
            # switches; the size / text / legend box above follows the tab
            col.addWidget(self._page_group())
            self.tabs = QTabWidget()
            for pg, title in ((0, 'Page 1'), (1, f'Page 2 — {self._extra_label}')):
                tw = QWidget()
                tl = QVBoxLayout(tw)
                tl.setContentsMargins(4, 4, 4, 4)
                if pg == 0:
                    tl.addWidget(self._panels_group())
                for g in self._option_groups(page=pg):
                    tl.addWidget(g)
                tl.addStretch(1)
                self.tabs.addTab(tw, title)
            self.tabs.currentChanged.connect(self._on_tab)
            col.addWidget(self.tabs, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # wide enough that nothing is ever cut off horizontally, whatever
        # labels the host declares
        scroll.setFixedWidth(min(760, side.sizeHint().width() + 26))
        root.addWidget(scroll)

        # right: preview + presets + buttons.  The preview is a rendered image
        # rather than a live canvas, so the page proportions are exact.
        right = QVBoxLayout()
        # page switcher: only meaningful when the host supplies a second figure
        if self._extra_render is not None:
            prow = QHBoxLayout()
            prow.addWidget(QLabel('Preview:'))
            self.page_sel = QComboBox()
            self.page_sel.addItem('Page 1', 0)
            self.page_sel.addItem(f'Page 2 — {self._extra_label}', 1)
            self.page_sel.currentIndexChanged.connect(self._on_page_sel)
            prow.addWidget(self.page_sel)
            prow.addStretch(1)
            right.addLayout(prow)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(520, 520)
        self.preview.setStyleSheet('background:#f4f4f4; border:1px solid #ddd;')
        right.addWidget(self.preview, 1)

        self.note = QLabel('')
        self.note.setStyleSheet('color:#888; font-size:11px;')
        right.addWidget(self.note)

        pres = QHBoxLayout()
        pres.addWidget(QLabel('Preset:'))
        self.preset_sel = QComboBox()
        self.preset_sel.setMinimumWidth(180)
        pres.addWidget(self.preset_sel, 1)
        b_load = QPushButton('Load')
        b_save = QPushButton('Save as…')
        b_del = QPushButton('Delete')
        for b in (b_load, b_save, b_del):
            pres.addWidget(b)
        b_load.clicked.connect(self._load_preset)
        b_save.clicked.connect(self._save_preset)
        b_del.clicked.connect(self._delete_preset)
        right.addLayout(pres)
        self._refresh_presets()

        bb = QDialogButtonBox()
        self.b_export = bb.addButton(
            'Export all…' if self._batch is not None else 'Save figure…',
            QDialogButtonBox.AcceptRole)
        b_reset = bb.addButton('Reset', QDialogButtonBox.ResetRole)
        bb.addButton('Close', QDialogButtonBox.RejectRole)
        self.b_export.clicked.connect(self._export)
        b_reset.clicked.connect(self._reset)
        bb.rejected.connect(self.reject)
        right.addWidget(bb)
        root.addLayout(right, 1)

    def _panels_group(self):
        g = QGroupBox('Panels  (top to bottom)')
        lay = QVBoxLayout(g)
        self._panel_rows = {}
        self._rows_box = QVBoxLayout()
        self._rows_box.setSpacing(2)
        lay.addLayout(self._rows_box)
        hint = QLabel('Height is relative: a panel with 2 is twice as tall '
                      'as one with 1.')
        hint.setWordWrap(True)
        hint.setStyleSheet('color:#888; font-size:11px;')
        lay.addWidget(hint)
        self._rebuild_panel_rows()
        # one fixed panel: nothing to switch, size or reorder, so no box
        if len(self._panels_decl) <= 1:
            g.setHidden(True)
        return g

    def _rebuild_panel_rows(self):
        while self._rows_box.count():
            it = self._rows_box.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)
        self._panel_rows = {}
        for i, p in enumerate(self.spec['panels']):
            decl = next((d for d in self._panels_decl if d.id == p['id']), None)
            if decl is None:
                continue
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            chk = QCheckBox(decl.label)
            chk.setChecked(bool(p['on']))
            chk.setEnabled(not decl.fixed)
            chk.stateChanged.connect(self._ui_to_spec)
            h.addWidget(chk, 1)
            h.addWidget(QLabel('height'))
            sp = QDoubleSpinBox()
            sp.setRange(0.1, 20.0)
            sp.setSingleStep(0.1)
            sp.setDecimals(1)
            sp.setValue(float(p['weight']))
            sp.setFixedWidth(60)
            sp.valueChanged.connect(self._ui_to_spec)
            h.addWidget(sp)
            up = QPushButton('↑')
            dn = QPushButton('↓')
            for b in (up, dn):
                b.setFixedWidth(26)
            up.setEnabled(i > 0)
            dn.setEnabled(i < len(self.spec['panels']) - 1)
            up.clicked.connect(lambda _c, k=i: self._move(k, -1))
            dn.clicked.connect(lambda _c, k=i: self._move(k, +1))
            h.addWidget(up)
            h.addWidget(dn)
            self._rows_box.addWidget(row)
            self._panel_rows[p['id']] = (chk, sp)

    def _page_group(self):
        """Page size, format and text/legend scaling, two controls per row."""
        g = QGroupBox('Page, text and legend')
        self._page_box = g
        grid = QGridLayout(g)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(3)
        grid.setContentsMargins(8, 6, 8, 6)

        self.sp_w = QDoubleSpinBox(); self.sp_w.setRange(3.0, 60.0)
        self.sp_w.setSuffix(' cm'); self.sp_w.setDecimals(1)
        self.sp_h = QDoubleSpinBox(); self.sp_h.setRange(3.0, 60.0)
        self.sp_h.setSuffix(' cm'); self.sp_h.setDecimals(1)
        self.sp_dpi = QSpinBox(); self.sp_dpi.setRange(72, 1200)
        self.sp_dpi.setSingleStep(50)
        self.cb_fmt = QComboBox()
        for v, lbl in self.FORMATS:
            self.cb_fmt.addItem(lbl, v)
        self.chk_transparent = QCheckBox('Transparent background')
        self.sp_font = QDoubleSpinBox(); self.sp_font.setRange(0.4, 4.0)
        self.sp_font.setSingleStep(0.05); self.sp_font.setDecimals(2)
        self.sp_font.setToolTip('Multiplies every text size in the figure. '
                                '1.0 = the size used on screen.')
        self.sp_line = QDoubleSpinBox(); self.sp_line.setRange(0.3, 5.0)
        self.sp_line.setSingleStep(0.1); self.sp_line.setDecimals(2)
        self.sp_mark = QDoubleSpinBox(); self.sp_mark.setRange(0.3, 5.0)
        self.sp_mark.setSingleStep(0.1); self.sp_mark.setDecimals(2)
        self.cb_leg = QComboBox()
        for v, lbl in LEGEND_MODES:
            self.cb_leg.addItem(lbl, v)
        self.sp_legcol = QSpinBox(); self.sp_legcol.setRange(1, 6)

        for _w in (self.sp_w, self.sp_h, self.sp_dpi, self.sp_font,
                   self.sp_line, self.sp_mark, self.sp_legcol):
            _w.setFixedWidth(96)
        for _w in (self.cb_fmt, self.cb_leg):
            _w.setFixedWidth(150)
        self.chk_transparent.setText('Transparent')
        self.lbl_page = QLabel('Size of:')
        self.lbl_page_val = QLabel('page 1')
        rows = [('Width', self.sp_w, 'Height', self.sp_h),
                ('Resolution', self.sp_dpi, 'Format', self.cb_fmt),
                ('Text ×', self.sp_font, 'Lines ×', self.sp_line),
                ('Markers ×', self.sp_mark, 'Legend cols', self.sp_legcol),
                ('Legend', self.cb_leg, '', self.chk_transparent)]
        for r, (l1, w1, l2, w2) in enumerate(rows):
            grid.addWidget(QLabel(l1 + ':' if l1 else ''), r, 0)
            grid.addWidget(w1, r, 1)
            grid.addWidget(QLabel(l2 + ':' if l2 else ''), r, 2)
            grid.addWidget(w2, r, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        for w in (self.sp_w, self.sp_h, self.sp_font, self.sp_line,
                  self.sp_mark, self.sp_legcol, self.sp_dpi):
            w.valueChanged.connect(self._ui_to_spec)
        self.cb_fmt.currentIndexChanged.connect(self._ui_to_spec)
        self.cb_leg.currentIndexChanged.connect(self._ui_to_spec)
        self.chk_transparent.stateChanged.connect(self._ui_to_spec)
        self._page_group_title()
        return g

    def _option_groups(self, page=None):
        """One group box per option group, two controls per row (only the
        options of *page* when given)."""
        decl = [o for o in self._options_decl
                if page is None or getattr(o, 'page', 0) == page]
        order = []
        for o in decl:
            if o.group not in order:
                order.append(o.group)
        boxes = []
        for name in order:
            box = QGroupBox(name)
            grid = QGridLayout(box)
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(3)
            grid.setContentsMargins(8, 6, 8, 6)
            row = 0
            cell = 0
            for o in [x for x in decl if x.group == name]:
                if o.kind == 'bool':
                    w = QCheckBox(o.label)
                    w.stateChanged.connect(self._ui_to_spec)
                    lbl = None
                elif o.kind == 'choice':
                    w = QComboBox()
                    for v, t in o.values:
                        w.addItem(t, v)
                    w.currentIndexChanged.connect(self._ui_to_spec)
                    lbl = QLabel(o.label + ':')
                elif o.kind == 'text':
                    w = QLineEdit()
                    if o.placeholder:
                        w.setPlaceholderText(o.placeholder)
                    w.textChanged.connect(self._ui_to_spec)
                    lbl = QLabel(o.label + ':')
                else:
                    w = QDoubleSpinBox()
                    w.setRange(o.lo, o.hi)
                    w.setSingleStep(o.step)
                    w.valueChanged.connect(self._ui_to_spec)
                    lbl = QLabel(o.label + ':')
                if o.tip:
                    w.setToolTip(o.tip)
                    if lbl is not None:
                        lbl.setToolTip(o.tip)
                if o.width:
                    w.setFixedWidth(o.width)
                elif o.kind in ('float', 'text') and not o.wide:
                    w.setFixedWidth(96)
                # explicitly wide fields, and combos with long entries, get a
                # row to themselves; everything else pairs up two per row
                wide = o.wide or (o.kind == 'choice'
                                  and max(len(t) for _v, t in o.values) > 22)
                if wide:
                    if cell % 2:
                        row += 1
                        cell = 0
                    if lbl is not None:
                        grid.addWidget(lbl, row, 0)
                        grid.addWidget(w, row, 1, 1, 3)
                    else:
                        grid.addWidget(w, row, 0, 1, 4)
                    row += 1
                    cell = 0
                else:
                    c = cell % 2
                    if lbl is None:             # checkbox spans label + field
                        grid.addWidget(w, row, c * 2, 1, 2)
                    else:
                        grid.addWidget(lbl, row, c * 2)
                        grid.addWidget(w, row, c * 2 + 1)
                    cell += 1
                    if cell == 2:
                        cell = 0
                        row += 1
                self._opt_widgets[o.id] = (w, lbl) if lbl is not None else (w,)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            boxes.append(box)
        return boxes

    # ── state <-> widgets ────────────────────────────────────────────────
    def _spec_to_ui(self):
        self._building = True
        s = self.spec
        w_cm, h_cm = page_size_cm(s, self._page)
        self.sp_w.setValue(w_cm)
        self.sp_h.setValue(h_cm)
        self.sp_dpi.setValue(int(s['dpi']))
        i = self.cb_fmt.findData(s['fmt'])
        self.cb_fmt.setCurrentIndex(max(0, i))
        self.chk_transparent.setChecked(bool(s.get('transparent')))
        ps = page_spec(s, self._page)
        self.sp_font.setValue(ps['font_scale'])
        self.sp_line.setValue(ps['line_scale'])
        self.sp_mark.setValue(ps['marker_scale'])
        i = self.cb_leg.findData(ps.get('legend', 'auto'))
        self.cb_leg.setCurrentIndex(max(0, i))
        self.sp_legcol.setValue(int(ps.get('legend_cols', 1)))
        self._rebuild_panel_rows()
        for o in self._options_decl:
            w = self._opt_widgets[o.id][0]
            v = s['options'].get(o.id, o.default)
            if o.kind == 'bool':
                w.setChecked(bool(v))
            elif o.kind == 'choice':
                j = w.findData(v)
                w.setCurrentIndex(max(0, j))
            elif o.kind == 'text':
                w.setText('' if v is None else str(v))
            else:
                w.setValue(float(v))
        self._building = False
        self._sync_option_rows()

    def _ui_to_spec(self, *_a):
        if self._building:
            return
        s = self.spec
        if self._page == 1:
            s['p2_width_cm'] = self.sp_w.value()
            s['p2_height_cm'] = self.sp_h.value()
        else:
            s['width_cm'] = self.sp_w.value()
            s['height_cm'] = self.sp_h.value()
        s['dpi'] = self.sp_dpi.value()
        s['fmt'] = self.cb_fmt.currentData()
        s['transparent'] = self.chk_transparent.isChecked()
        pre = 'p2_' if self._page == 1 else ''
        s[pre + 'font_scale'] = self.sp_font.value()
        s[pre + 'line_scale'] = self.sp_line.value()
        s[pre + 'marker_scale'] = self.sp_mark.value()
        s[pre + 'legend'] = self.cb_leg.currentData()
        s[pre + 'legend_cols'] = self.sp_legcol.value()
        for p in s['panels']:
            row = self._panel_rows.get(p['id'])
            if row:
                p['on'] = row[0].isChecked()
                p['weight'] = row[1].value()
        before = dict(s['options'])
        for o in self._options_decl:
            w = self._opt_widgets[o.id][0]
            if o.kind == 'bool':
                s['options'][o.id] = w.isChecked()
            elif o.kind == 'choice':
                s['options'][o.id] = w.currentData()
            elif o.kind == 'text':
                s['options'][o.id] = w.text()
            else:
                s['options'][o.id] = w.value()
        self._apply_follows(before)
        self._sync_option_rows()
        self._queue()

    def _apply_follows(self, before):
        """Refill the fields that follow an option which was just switched
        (see Option.follows).  Leaving the free value stashes the fields;
        coming back to it untouched restores them."""
        opts = self.spec['options']
        stash = self.__dict__.setdefault('_follow_stash', {})
        for o in self._options_decl:
            fl = getattr(o, 'follows', None)
            if not fl or o.kind != 'text':
                continue
            src, table = fl
            old, new = before.get(src), opts.get(src)
            if old == new:
                continue
            cur = opts.get(o.id, '')
            if new in table:
                if old not in table:
                    stash[o.id] = cur            # what was typed, kept for later
                val = table[new]
            elif old in table and cur == table[old] and o.id in stash:
                val = stash.pop(o.id)            # back to typed, nothing edited
            else:
                continue
            opts[o.id] = val
            w = self._opt_widgets[o.id][0]
            self._building = True
            try:
                w.setText('' if val is None else str(val))
            finally:
                self._building = False

    def _sync_option_rows(self):
        """Grey out options whose 'depends' condition is not met."""
        for o in self._options_decl:
            if not o.depends:
                continue
            dep_id, dep_val = o.depends
            cur = self.spec['options'].get(dep_id)
            ok = bool(cur) if dep_val is True else (cur == dep_val)
            for w in self._opt_widgets[o.id]:
                w.setEnabled(ok)

    def _move(self, idx, step):
        ps = self.spec['panels']
        j = idx + step
        if 0 <= j < len(ps):
            ps[idx], ps[j] = ps[j], ps[idx]
            self._rebuild_panel_rows()
            self._queue()

    def _on_tab(self, idx):
        if getattr(self, 'page_sel', None) is not None \
                and self.page_sel.currentIndex() != idx:
            self.page_sel.setCurrentIndex(idx)      # -> _on_page_sel -> _on_page
        else:
            self._on_page(idx)

    def _on_page_sel(self, idx):
        if getattr(self, 'tabs', None) is not None and self.tabs.currentIndex() != idx:
            self.tabs.setCurrentIndex(idx)          # -> _on_tab
        self._on_page(idx)

    def _on_page(self, idx):
        idx = int(idx)
        if idx == self._page:
            return
        self._page = idx
        if self._page == 1 and not self.spec.get('p2_width_cm'):
            # page 2 starts out the same size as page 1, then goes its own way
            self.spec['p2_width_cm'] = self.spec['width_cm']
            self.spec['p2_height_cm'] = self.spec['height_cm']
        self._building = True
        ps = page_spec(self.spec, self._page)
        w, h = page_size_cm(self.spec, self._page)
        self.sp_w.setValue(w)
        self.sp_h.setValue(h)
        self.sp_font.setValue(ps['font_scale'])
        self.sp_line.setValue(ps['line_scale'])
        self.sp_mark.setValue(ps['marker_scale'])
        self.cb_leg.setCurrentIndex(max(0, self.cb_leg.findData(ps.get('legend', 'auto'))))
        self.sp_legcol.setValue(int(ps.get('legend_cols', 1)))
        self._building = False
        self._page_group_title()
        self._queue()

    def _page_group_title(self):
        if getattr(self, '_page_box', None) is None:
            return
        self._page_box.setTitle(
            'Page, text and legend' if self._extra_render is None else
            f'Page {self._page + 1}: size, text and legend')

    # ── preview ──────────────────────────────────────────────────────────
    def _queue(self):
        self._timer.start(140)

    def _refresh_preview(self):
        w_in, h_in = figsize(self.spec, self._page)
        box = self.preview.size()
        # render at screen resolution, then scale the image to the label
        dpi = max(30.0, min(160.0,
                            min(box.width() / max(w_in, 0.1),
                                box.height() / max(h_in, 0.1))))
        fig = Figure(figsize=(w_in, h_in), dpi=dpi, facecolor='white')
        second = self._page == 1 and self._extra_render is not None
        try:
            if second:
                self._extra_render(fig, page_spec(self.spec, 1))
            else:
                self._render(fig, self.spec)
            extra_note = ''
            if second:
                opts = self.spec.get('options', {})
                on = ((self._extra_page_flag and opts.get(self._extra_page_flag))
                      or (self._extra_flag and opts.get(self._extra_flag)))
                names = [f'"{o.label}"' for o in self._options_decl
                         if o.id in (self._extra_page_flag, self._extra_flag)]
                extra_note = ('' if on else
                              '   ·   not exported yet: tick '
                              + (' or '.join(names) or 'it on its tab'))
            ok = True
        except Exception as e:                      # never kill the dialog
            fig.clear()
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, f'Preview failed:\n{e}', ha='center', va='center',
                    fontsize=9, color='#c0392b', wrap=True)
            self.note.setText('')
            ok = False
        # go through the same save path as the export, so the preview shows the
        # real margins (bbox_inches='tight') rather than the raw canvas
        FigureCanvasAgg(fig)
        settle_legends(fig)
        if ok:
            self.note.setText(self._size_note(fig, w_in, h_in) + extra_note)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, facecolor='white',
                    bbox_inches='tight', pad_inches=0.05)
        img = QImage.fromData(buf.getvalue(), 'PNG')
        pm = QPixmap.fromImage(img)
        # the tight bounding box can come out larger than the space we have;
        # scale it down so the whole page is always visible
        if pm.width() > box.width() or pm.height() > box.height():
            pm = pm.scaled(box.width(), box.height(),
                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pm)

    def _size_note(self, fig, w_in, h_in):
        """Page, plot-area and file size for the note under the preview.

        The plot area (the frame of the panels) depends on the settings only;
        the file is cropped to the drawing, so a legend outside the plot or a
        long label makes the file larger without shrinking the plot."""
        dpi = int(self.spec.get('dpi', 300))
        pw, ph = page_size_cm(self.spec, self._page)
        txt = f'page {pw:.1f} × {ph:.1f} cm'
        try:
            r = fig.canvas.get_renderer()
            boxes = [a.get_position() for a in fig.axes if a.get_visible()]
            if boxes:
                x0 = min(b.x0 for b in boxes); x1 = max(b.x1 for b in boxes)
                y0 = min(b.y0 for b in boxes); y1 = max(b.y1 for b in boxes)
                txt += (f'  ·  plot area {(x1 - x0) * w_in * CM_PER_IN:.1f} × '
                        f'{(y1 - y0) * h_in * CM_PER_IN:.1f} cm')
            tb = fig.get_tightbbox(r)
            fw, fh = tb.width + 0.1, tb.height + 0.1      # pad_inches 0.05
            txt += (f'  ·  file {fw * CM_PER_IN:.1f} × {fh * CM_PER_IN:.1f} cm, '
                    f'{int(round(fw * dpi))} × {int(round(fh * dpi))} px '
                    f'at {dpi} dpi')
        except Exception:
            txt += (f' at {dpi} dpi  ·  {int(round(w_in * dpi))} × '
                    f'{int(round(h_in * dpi))} px')
        return txt

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._queue()

    # ── presets ──────────────────────────────────────────────────────────
    def _presets(self):
        return self._store.get(self._pkey, {}).get('presets', {})

    def _refresh_presets(self, select=None):
        """List the shared presets; one saved from another figure type is
        marked with that type."""
        self.preset_sel.clear()
        for name, pres in sorted(self._presets().items()):
            kind = pres.get('kind', self._pkey)
            shown = (name if kind == self._key
                     else f"{name}   [{KIND_LABELS.get(kind, kind)}]")
            self.preset_sel.addItem(shown, name)
        if select is not None:
            i = self.preset_sel.findData(select)
            if i >= 0:
                self.preset_sel.setCurrentIndex(i)

    def _current_preset(self):
        return self.preset_sel.currentData() or ''

    def _load_preset(self):
        name = self._current_preset()
        pres = self._presets().get(name)
        if not pres:
            return
        self.spec = _merge(pres, self._panels_decl, self._options_decl)
        self._spec_to_ui()
        self._queue()

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, 'Save preset', 'Preset name:',
                                        text=self._current_preset())
        if not ok or not name.strip():
            return
        pres = self._spec_to_store()
        pres['kind'] = self._key
        if self._snapshot is not None:
            try:
                fill = self._snapshot(copy.deepcopy(self.spec)) or {}
            except Exception:
                fill = {}
            for k, v in fill.items():
                if k in pres['options'] and not str(pres['options'][k]).strip():
                    pres['options'][k] = v
        self._store, ok = _update_store(
            lambda st: st.setdefault(self._pkey, {}).setdefault('presets', {})
            .__setitem__(name.strip(), pres))
        if not ok:
            QMessageBox.warning(self, 'Presets',
                                'Could not write the preset file.')
            return
        self._refresh_presets(select=name.strip())

    def _delete_preset(self):
        name = self._current_preset()
        if name in self._presets():
            self._store, _ = _update_store(
                lambda st: st.get(self._pkey, {}).get('presets', {}).pop(name, None))
            self._refresh_presets()

    def _reset(self):
        self.spec = default_spec(self._panels_decl, self._options_decl)
        self.spec.update(self._defaults)
        self._spec_to_ui()
        self._queue()

    def _spec_to_store(self):
        """The spec as saved: everything shown in the dialog (page, text,
        legend, panels, titles, names, axis ranges ...) plus, for the
        per-sample fields, which sample each value belongs to."""
        out = copy.deepcopy(self.spec)
        out['idents'] = {o.id: o.ident for o in self._options_decl
                         if getattr(o, 'ident', None) is not None}
        return out

    def _remember(self):
        last = self._spec_to_store()
        self._store, _ = _update_store(
            lambda st: st.setdefault(self._key, {}).__setitem__('last', last))

    # ── saving ───────────────────────────────────────────────────────────
    def _export_batch(self):
        fmt = self.spec['fmt']
        opts = self.spec.get('options', {})
        one_pdf = bool(fmt == 'pdf' and self._batch_pdf_flag
                       and opts.get(self._batch_pdf_flag))
        if one_pdf:
            path, _ = dfold.get_save_file_name(
                self, 'Save figures', f'{self._name}.pdf', 'PDF (*.pdf)')
            if path and not os.path.splitext(path)[1]:
                path += '.pdf'
        else:
            path = dfold.get_existing_directory(
                self, 'Save the figures in this folder')
        if not path:
            return
        prog = QProgressDialog('Exporting…', 'Cancel', 0, 1, self)
        prog.setWindowTitle('Export')
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)

        def progress(i, n, label=''):
            prog.setMaximum(max(n, 1))
            prog.setValue(i)
            prog.setLabelText(label)
            QApplication.processEvents()
            return prog.wasCanceled()
        try:
            written = self._batch(copy.deepcopy(self.spec), path, progress)
        except Exception as e:
            prog.close()
            QMessageBox.critical(self, 'Export error', str(e))
            return
        prog.close()
        self._remember()
        shown = written[:14] + ([f'… and {len(written) - 14} more'] if len(written) > 14 else [])
        QMessageBox.information(
            self, 'Saved',
            ('Saved:\n' + '\n'.join(shown)) if written else 'Nothing was written.')

    def _export(self):
        if self._batch is not None:
            return self._export_batch()
        fmt = self.spec['fmt']
        opts = self.spec.get('options', {})
        want_page = bool(self._extra_page_flag and opts.get(self._extra_page_flag))
        want_file = bool(self._extra_flag and opts.get(self._extra_flag))
        path, _ = dfold.get_save_file_name(
            self, 'Save figure', f'{self._name}.{fmt}',
            f'{fmt.upper()} (*.{fmt});;All files (*)')
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += '.' + fmt
        note = ''
        try:
            fig = Figure(figsize=figsize(self.spec), facecolor='white')
            self._render(fig, self.spec)
            fig2 = None
            if self._extra_render is not None and (want_page or want_file):
                fig2 = Figure(figsize=figsize(self.spec, 1), facecolor='white')
                self._extra_render(fig2, page_spec(self.spec, 1))
            written = []
            if want_page and fig2 is not None and fmt == 'pdf':
                from matplotlib.backends.backend_pdf import PdfPages
                with matplotlib.rc_context(EDITABLE_TEXT), PdfPages(path) as pdf:
                    for f in (fig, fig2):
                        settle_legends(f)
                        pdf.savefig(f, dpi=int(self.spec.get('dpi', 300)),
                                    facecolor='white', bbox_inches='tight',
                                    pad_inches=0.05)
                written.append(path + '   (2 pages)')
            else:
                if want_page and fmt != 'pdf':
                    note = ('\nA second page needs the PDF format; '
                            f'the {self._extra_label.lower()} was written as its own '
                            'file instead.')
                    want_file = True
                save_figure(fig, path, self.spec)
                written.append(path)
            if want_file and fig2 is not None and not (want_page and fmt == 'pdf'):
                base, ext = os.path.splitext(path)
                p2 = f'{base}_{self._extra_suffix}{ext}'
                save_figure(fig2, p2, self.spec)
                written.append(p2)
        except Exception as e:
            QMessageBox.critical(self, 'Export error', str(e))
            return
        self._remember()
        QMessageBox.information(self, 'Saved',
                                'Saved:\n' + '\n'.join(written) + note)

    def reject(self):
        self._remember()
        super().reject()
