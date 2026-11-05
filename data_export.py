"""Shared data export for the Plateau & isochron and Age distribution tools.

One place for what both tables need, so the two exports cannot drift apart:

* the CONSTANTS block that opens every export (atmospheric argon, decay
  constants, FC age, J, irradiation time ...);
* the time of each analysis, as days since the irradiation;
* the raw-signal table, written as a second sheet of an .xlsx (and appended as
  a section to a .csv);
* TableSink, a drop-in replacement for ``open(path, 'w')`` around a csv.writer
  that writes either a .csv or a multi-sheet .xlsx.

The raw values come from the mass-spec file through the converter, which stores
them in each step's ``raw`` dict and the run time in ``run_dt`` / ``days_irr``.
Files converted before this existed simply have blank cells here.
"""
import csv
import datetime
import io
import math
import os
import re

from arar_math import cfg_J


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

    add('FC age (Ma)', _g(cfg.get('stdAgeMa'), 8), 'Flux monitor (Fish Canyon sanidine)')
    add('  ± (Ma, 1σ)', _g(cfg.get('stdAgeErrMa'), 6),
        'external' if cfg.get('stdAgeErrType') == 'external' else 'internal')
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
         'Mass-discrimination factor of the aliquot (dimensionless). NOT applied to the signals above: '
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

def notes_rows(raws=None):
    """Rows of the Notes sheet. raws: the raw_tables() groups written."""
    raws = raws or []
    rows = [
        ['ABOUT THIS FILE'],
        ['Sheet "Results"', 'The step table, plateau and isochron results of each sample.'],
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
        ['Signal units', 'The unit is in every header: fA for ArArCALC and V for Mass Spec, both set '
                         'manually (see Constants). No unit conversion is made.'],
        ['Signals', 'Every signal column in the raw sheets is blank-corrected (baseline and blank '
                    'subtracted). 37Ar and 39Ar are in addition decay-corrected; 36Ar, '
                    '38Ar and 40Ar need no decay correction. Mass Spec: the file\'s own corrected signals, which '
                    'reproduce the ratio columns of the file. ArArCALC: intercept minus blank, with 37Ar and '
                    '39Ar multiplied by their decay factors. Mass discrimination (MDF) and the interference '
                    '(Ca, K, Cl) corrections are NOT applied to the ArArCALC signals; ArArCALC applies MDF to '
                    'its ratios and adds a 36Ar collector calibration that its export does not contain, so a '
                    'ratio of these signals is not the same as ArArCALC\'s corrected ratio.'],
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
             'STACKED ISOCHRON RESULTS',
             'SAMPLE RESULTS', 'POINT DATA', 'POOLED RESULT', 'RAW SIGNALS',
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

    def __init__(self, path):
        self.path = path
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
                _notes = notes_rows(self.raws)
                if self.raw_missing:
                    _notes.append(['No raw-signal sheet',
                                   'The loaded files were saved before raw signals were stored '
                                   'in them. Re-convert the source data and reload to include them.'])
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
