#!/usr/bin/env python3
"""The standards library: ⁴⁰K decay constants, calibrations, routes and
neutron-fluence monitors, with its editor and the Monitor dropdown of the tools.

The published values ship in standards_default.json next to this file.  Edits
made in the Standards editor are saved to standards.json in the user's
configuration folder (SimplAr/standards.json), and only the entries that differ
from the defaults are stored there, so a later SimplAr version can still add or
correct defaults.  An entry the user deleted is stored as {"name": ...,
"deleted": true}.  The environment variable SIMPLAR_STANDARDS_FILE points the
user file elsewhere; the tests set it to an empty location so that they always
run on the published values.

Four sections, each a list of dicts keyed by "name" (standards by "id"):

  decay_constants  total ⁴⁰K decay constant (yr⁻¹) and its 1σ
  calibrations     an age of one standard (Fish Canyon sanidine, FCs, unless
                   'standard' says otherwise) with the decay constant it pairs
                   with, how it was determined (basis 'astronomical', 'joint'
                   or 'k-ar'), the astronomical anchor when there is one, and
                   cov(λ, t_FCs) for a joint calibration
  routes           saved routes: a monitor, the R steps between standards,
                   the calibration they end at and the decay constant
  standards        a monitor with its R values relative to FCs or another
                   standard ('other'), each stored the way its paper prints it
                   ('ratio': 'std/FCs', the ⁴⁰Ar*/³⁹Ar_K of the standard over
                   that of the other, or 'FCs/std') and the ages published
                   for it

All uncertainties are 1σ.  arar_math builds its preset tables from the
library.

Layout of this file, top to bottom:
  the library (load, merge with the user's edits, save, check)
  MonitorCombo, the Monitor dropdown in front of every calibration dropdown
  StandardsEditor, the Standards window of the launcher (python standards.py
  opens it on its own)
"""

import copy
import json
import math
import os
import re
import sys

from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QWidget, QListWidget, QListWidgetItem, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QComboBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QGroupBox, QSplitter, QAbstractItemView,
    QInputDialog, QScrollArea, QFrame,
)
from PyQt5.QtCore import Qt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, 'standards_default.json')
SECTIONS = ('decay_constants', 'calibrations', 'standards', 'routes')
BASES = ('astronomical', 'joint', 'k-ar')


def key_of(section):
    return 'id' if section == 'standards' else 'name'


def user_path():
    """Path of the user's edits (it may not exist)."""
    env = os.environ.get('SIMPLAR_STANDARDS_FILE')
    if env:
        return env
    base = None
    try:
        from PyQt5.QtCore import QStandardPaths
        base = QStandardPaths.writableLocation(QStandardPaths.GenericConfigLocation)
    except Exception:
        pass
    base = base or os.path.join(os.path.expanduser('~'), '.config')
    return os.path.join(base, 'SimplAr', 'standards.json')


def load_default():
    with open(DEFAULT_PATH, encoding='utf-8') as f:
        return json.load(f)


def _load_user():
    p = user_path()
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def merge(default, user):
    """default with the user's entries laid over it, section by section."""
    lib = copy.deepcopy(default)
    for sec in SECTIONS:
        k = key_of(sec)
        rows = lib.get(sec, [])
        index = {r[k]: i for i, r in enumerate(rows)}
        for u in user.get(sec, []):
            name = u.get(k)
            if name is None:
                continue
            if u.get('deleted'):
                if name in index:
                    rows[index[name]] = None
                continue
            entry = copy.deepcopy(u)
            entry['edited'] = name in index
            entry['user'] = name not in index
            if name in index:
                rows[index[name]] = entry
            else:
                index[name] = len(rows)
                rows.append(entry)
        lib[sec] = [r for r in rows if r is not None]
    return lib


def load():
    """The library in use: the defaults with the user's edits applied."""
    return merge(load_default(), _load_user())


def _clean(entry):
    return {k: v for k, v in entry.items() if k not in ('edited', 'user')}


def diff(lib, default=None):
    """The user file content that turns default into lib."""
    default = load_default() if default is None else default
    out = {'format': default.get('format', '')}
    for sec in SECTIONS:
        k = key_of(sec)
        d_rows = {r[k]: r for r in default.get(sec, [])}
        l_rows = {r[k]: _clean(r) for r in lib.get(sec, [])}
        rows = []
        for name, r in l_rows.items():
            if name not in d_rows or r != d_rows[name]:
                rows.append(r)
        for name in d_rows:
            if name not in l_rows:
                rows.append({k: name, 'deleted': True})
        out[sec] = rows
    return out


def save(lib, base=None):
    """Write the user's edits; removes the user file when nothing differs.
    base: the published values the editor was opened with.  An entry still
    equal to them was not edited, even if the published file changed since."""
    p = user_path()
    d = diff(lib)
    if base is not None:
        for sec in SECTIONS:
            k = key_of(sec)
            b_rows = {r[k]: r for r in base.get(sec, [])}
            d[sec] = [r for r in d[sec] if r.get('deleted') or r != b_rows.get(r[k])]
    if not any(d[sec] for sec in SECTIONS):
        if os.path.isfile(p):
            os.remove(p)
        return p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    os.replace(tmp, p)
    return p


def reset():
    """Forget every edit."""
    p = user_path()
    if os.path.isfile(p):
        os.remove(p)


def find(lib, section, name):
    k = key_of(section)
    return next((r for r in lib.get(section, []) if r.get(k) == name), None)


RATIOS = ('std/FCs', 'FCs/std')


def ic_other(ic):
    """The standard an intercalibration is measured against (FCs when not given)."""
    return ic.get('other') or 'FCs'


def ic_printed(ic, holder):
    """(numerator, denominator, R, 1σ) of an intercalibration as its paper
    prints it.  'ratio' is 'std/FCs' (holder / other) or 'FCs/std' (other /
    holder), whatever the other standard is."""
    R, s = float(ic['R']), float(ic.get('R_1s') or 0.0)
    other = ic_other(ic)
    if ic.get('ratio') == 'FCs/std':
        return other, holder, R, s
    return holder, other, R, s


def intercalibrations(lib):
    """Every R in the library as (holder id, ic, numerator, denominator, R, 1σ)."""
    out = []
    for st in lib.get('standards', []):
        for ic in st.get('intercalibrations') or []:
            try:
                row = (st['id'], ic) + ic_printed(ic, st['id'])
            except (KeyError, TypeError, ValueError):
                continue
            if row[2] != row[3] and row[4] > 0:      # FCs/FCs = 1 is not a step
                out.append(row)
    return out


def r_std_fcs(ic):
    """(R, 1σ) of an intercalibration as standard / FCs, whichever way the
    paper printed it (ic['ratio'] 'std/FCs', the default, or 'FCs/std')."""
    R, s = float(ic['R']), float(ic.get('R_1s') or 0.0)
    if ic.get('ratio') == 'FCs/std':
        return 1.0 / R, s / R**2
    return R, s


def pm_text(v, s, sig=6):
    """'0.041707 ± 0.000011': the error with as many decimals as the value."""
    vt = f'{v:.{sig}g}'
    if 'e' in vt:
        return f'{v:.{sig}g} ± {s:.2g}'
    d = len(vt.split('.')[1]) if '.' in vt else 0
    return f'{vt} ± {s:.{d}f}'


def r_text(ic, std_id='std'):
    """'R(FCs/A1) = 4.0819 ± 0.0007' as printed."""
    num, den = ('FCs', std_id) if ic.get('ratio') == 'FCs/std' else (std_id, 'FCs')
    return f"R({num}/{den}) = {pm_text(float(ic['R']), float(ic.get('R_1s') or 0))}"


def short_ref(ref):
    """'Renne, P.R., Swisher, C.C., ... (1998) Chem. Geol.' -> 'Renne et al. (1998)'."""
    ref = str(ref or '').strip()
    m = re.search(r'\((\d{4})\)', ref)
    year = m.group(1) if m else ''
    head = ref.split('(')[0].strip().rstrip(',;')
    if not head:
        return ref
    if 'et al' in head:
        names = [head.split(',')[0].strip()]
        etal = True
    else:
        parts = [p.strip() for p in head.split(',') if p.strip()]
        names = parts[0::2]
        etal = len(names) > 2
    if etal:
        txt = f'{names[0]} et al.'
    elif len(names) == 2:
        txt = f'{names[0]} & {names[1]}'
    else:
        txt = names[0]
    return f'{txt} ({year})' if year else txt


def recommended_R(std):
    """(R standard/FCs, its 1σ, entry) of a standard's recommended
    intercalibration: the one marked recommended, else the first."""
    ics = [x for x in std.get('intercalibrations') or [] if ic_other(x) == 'FCs']
    if not ics:
        return None
    ic = next((x for x in ics if x.get('recommended')), ics[0])
    R, s = r_std_fcs(ic)
    return R, s, ic


def check(lib):
    """List of problems that would stop the tools from using the library."""
    errs = []
    lam_names = {d.get('name') for d in lib.get('decay_constants', [])}
    for sec in SECTIONS:
        k = key_of(sec)
        seen = set()
        for r in lib.get(sec, []):
            name = r.get(k)
            if not name:
                errs.append(f'{sec}: an entry has no {k}')
            elif name in seen:
                errs.append(f'{sec}: "{name}" appears twice')
            seen.add(name)
    for d in lib.get('decay_constants', []):
        if not (float(d.get('lambda') or 0) > 0):
            errs.append(f'Decay constant "{d.get("name")}": λ must be positive')
    std_ids = {st.get('id') for st in lib.get('standards', [])}
    for c in lib.get('calibrations', []):
        n = c.get('name')
        if not (float(c.get('age') or 0) > 0):
            errs.append(f'Calibration "{n}": the age must be positive')
        if c.get('standard', 'FCs') not in std_ids:
            errs.append(f'Calibration "{n}": standard "{c.get("standard")}" is not in the library')
        if c.get('decay_constant') not in lam_names:
            errs.append(f'Calibration "{n}": decay constant "{c.get("decay_constant")}" is not in the library')
        if c.get('basis') not in BASES:
            errs.append(f'Calibration "{n}": basis must be one of {", ".join(BASES)}')
        a = c.get('anchor')
        if a and not (float(a.get('age') or 0) > 0 and float(a.get('R_fcs_anchor') or 0) > 0):
            errs.append(f'Calibration "{n}": the anchor needs an age and R(FCs/anchor)')
    for s in lib.get('standards', []):
        for ic in s.get('intercalibrations') or []:
            if not (float(ic.get('R') or 0) > 0):
                errs.append(f'Standard "{s.get("id")}": every R must be positive')
            if ic.get('ratio', 'std/FCs') not in RATIOS:
                errs.append(f'Standard "{s.get("id")}": R must be std/FCs or FCs/std')
            if ic_other(ic) not in std_ids:
                errs.append(f'Standard "{s.get("id")}": R against "{ic_other(ic)}", '
                            'which is not another standard in the library')
    for r in lib.get('routes', []):
        for e in route_problems(r, lib):
            errs.append(f'Route "{r.get("name")}": {e}')
    return errs


# --------------------------------------------------------------------------
# Routes
#
# A route ties the monitor of an irradiation to a calibration through R
# values from the library:
#
#   {'name': ..., 'monitor': 'TCs',
#    'steps': [{'from': 'TCs', 'to': 'FCs', 'num': 'TCs', 'den': 'FCs',
#               'R': 1.0112, 'R_1s': 0.001, 'source': 'Renne et al. (1998)'}],
#    'calibration': name of a calibration of the last standard,
#    'lambda': name of a decay constant,
#    'default': True}            (the route the tools use for this monitor)
#
# Each step holds the R as its paper prints it (num/den), so a file that
# stores the route can be read without the library.  The age arithmetic is in
# arar_math ("Calibration routes").
# --------------------------------------------------------------------------

def step_from_ic(holder, ic, frm):
    """A route step leaving standard frm through intercalibration ic."""
    num, den, R, s = ic_printed(ic, holder)
    to = den if frm == num else num
    return {'from': frm, 'to': to, 'num': num, 'den': den, 'R': R, 'R_1s': s,
            'source': short_ref(ic.get('reference', '')) or ic.get('reference', '')}


def steps_from(lib, std_id, visited=()):
    """Every step the library supports out of std_id, to standards not in visited."""
    out = []
    for holder, ic, num, den, R, s in intercalibrations(lib):
        if std_id not in (num, den):
            continue
        step = step_from_ic(holder, ic, std_id)
        if step['to'] not in visited:
            out.append(step)
    return out


def calibrations_of(lib, std_id):
    """The calibrations whose age is that of std_id."""
    return [c for c in lib.get('calibrations', []) if c.get('standard', 'FCs') == std_id]


def _same_step(a, b):
    return (a.get('num') == b.get('num') and a.get('den') == b.get('den')
            and abs(float(a.get('R', 0)) - float(b.get('R', 0))) <= 1e-12 * abs(float(b.get('R', 1)))
            and abs(float(a.get('R_1s', 0)) - float(b.get('R_1s', 0))) <= 1e-12 * max(1.0, abs(float(b.get('R_1s', 0)))))


def route_problems(route, lib):
    """What is wrong with a route, as a list of sentences (empty when usable)."""
    errs = []
    std_ids = {st.get('id') for st in lib.get('standards', [])}
    at = route.get('monitor')
    if at not in std_ids:
        return [f'monitor "{at}" is not in the library']
    seen = [at]
    for i, st in enumerate(route.get('steps') or [], 1):
        if st.get('from') != at:
            errs.append(f'step {i} starts at {st.get("from")}, not at {at}')
            break
        if not any(_same_step(st, c) for c in steps_from(lib, at)):
            errs.append(f'step {i}, {st.get("num")}/{st.get("den")} = {st.get("R")}, '
                        'is not an R in the library')
        at = st.get('to')
        if at in seen:
            errs.append(f'step {i} returns to {at}')
            break
        seen.append(at)
    cal = find(lib, 'calibrations', route.get('calibration'))
    if cal is None:
        errs.append(f'calibration "{route.get("calibration")}" is not in the library')
    elif cal.get('standard', 'FCs') != at:
        errs.append(f'the route ends at {at}, but "{cal["name"]}" is a calibration of '
                    f'{cal.get("standard", "FCs")}')
    if find(lib, 'decay_constants', route.get('lambda')) is None:
        errs.append(f'decay constant "{route.get("lambda")}" is not in the library')
    return errs


def default_calibration(lib, std_id='FCs'):
    """The calibration the tools open on for a standard: the one marked default
    when it is of std_id, else the first of std_id."""
    cals = calibrations_of(lib, std_id)
    return next((c for c in cals if c.get('default')), cals[0] if cals else None)


def shortest_route(lib, monitor, target='FCs'):
    """Steps from monitor to target through the R values marked Use (the first
    R of a pair when none is marked), fewest steps first; None if there is no
    such path."""
    use = {}
    for holder, ic, num, den, R, s in intercalibrations(lib):
        key = frozenset((num, den))
        if key not in use or (ic.get('recommended') and not use[key][1].get('recommended')):
            use[key] = (holder, ic)
    frontier, seen = [(monitor, [])], {monitor}
    while frontier:
        nxt = []
        for at, steps in frontier:
            if at == target:
                return steps
            for key, (holder, ic) in use.items():
                if at in key:
                    st = step_from_ic(holder, ic, at)
                    if st['to'] not in seen:
                        seen.add(st['to'])
                        nxt.append((st['to'], steps + [st]))
        frontier = nxt
    return None


def route_line(route):
    """'TCs → FCs (Renne et al. (1998)) → Phillips et al. (2022) · λ Min et al. (2000)'."""
    parts = [route.get('monitor', '')]
    for st in route.get('steps') or []:
        parts.append(f"{st.get('to')} ({st.get('source') or 'no source'})")
    cal, lam = route.get('calibration'), route.get('lambda')
    cal = cal.get('name', 'custom') if isinstance(cal, dict) else str(cal).split('  ')[0]
    lam = lam.get('name', 'custom') if isinstance(lam, dict) else lam
    return ' → '.join(parts) + f' · {cal} · λ {lam}'


def default_route(lib, monitor):
    """The route the tools use for a monitor: the one marked default in the
    library; else, for a standard with an astronomical age of its own (A1,
    Mes4), that age; else the shortest path to FCs on the default FCs
    calibration (so a K–Ar standard such as GA1550 goes through FCs, not
    through its own K–Ar age)."""
    mine = [r for r in lib.get('routes', []) if r.get('monitor') == monitor]
    r = next((x for x in mine if x.get('default')), None)
    if r is not None:
        return copy.deepcopy(r)
    fcs_cal = default_calibration(lib, 'FCs')
    own = next((c for c in calibrations_of(lib, monitor) if c.get('basis') == 'astronomical'), None)
    if own is not None and monitor != 'FCs':
        cal, steps = own, []
    else:
        steps = shortest_route(lib, monitor, 'FCs')
        if steps is None or fcs_cal is None:
            return None
        cal = fcs_cal
    return {'name': f'{monitor}: default', 'monitor': monitor, 'steps': steps,
            'calibration': cal['name'], 'lambda': cal.get('decay_constant'), 'auto': True}


# --------------------------------------------------------------------------
# Monitor dropdown
#
# The Monitor dropdown that goes in front of every calibration dropdown.
#
# A calibration is a Fish Canyon (FCs) calibration; the monitor says which
# standard was irradiated with the samples.  For a monitor other than FCs its age
# follows from the calibration through the monitor's R relative to FCs in the
# standards library (the R marked "Use" there), and that R's uncertainty is added
# to the external tier.  See arar_math, "Monitors other than Fish Canyon
# sanidine".
# --------------------------------------------------------------------------

class MonitorCombo(QComboBox):
    """Monitor picker; currentData() is the monitor id."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from arar_math import monitor_choices
        for mid, label in monitor_choices():
            self.addItem(label, mid)
        self.setToolTip(
            'The standard irradiated with the samples. Its age follows from the\n'
            'Fish Canyon calibration through its R relative to FCs (Standards…),\n'
            'and the uncertainty of that R is added to the external error.')
        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)

    def monitor(self):
        return self.currentData() or 'FCs'

    def set_monitor(self, mid):
        i = self.findData(mid or 'FCs')
        self.blockSignals(True)
        self.setCurrentIndex(i if i >= 0 else 0)
        self.blockSignals(False)

    def keys(self):
        """cfg keys (monitor, monitorR, monitorRErr) of the current monitor."""
        from arar_math import monitor_keys
        return monitor_keys(self.monitor())


def monitor_text(cfg):
    """'ACs = 1.18384 Ma (R(ACs/FCs) = 0.041707 ± 0.000011, Niespolo et al. 2017)',
    or '' for FCs."""
    from arar_math import library_standard, monitor_age
    mid = (cfg or {}).get('monitor') or 'FCs'
    if mid == 'FCs':
        return ''
    try:
        age = monitor_age(cfg)
    except (ValueError, ZeroDivisionError, OverflowError):
        return ''
    s = library_standard(mid)
    rec = recommended_R(s) if s else None
    src = ''
    if rec:
        ic = rec[2]
        ref = str(ic.get('reference') or '')
        src = ', ' + ref.split(',')[0].strip() + (' et al.' if ref.count(',') > 2 else '') \
            + (' ' + ref[ref.find('(') + 1:ref.find(')')] if '(' in ref else '')
        src = f" ({r_text(ic, mid)}{src})"
    return f'{mid} = {age:.6g} Ma{src}'


# --------------------------------------------------------------------------
# Standards window
#
# The Standards window: view and edit the library above.
#
# Three pages: the Fish Canyon calibrations (the ages of FCs), the other
# monitors with their R values relative to FCs and their published ages, and the
# ⁴⁰K decay constants.  Every value is 1σ.  Edits are kept in the window until Save; Save
# writes only what differs from the published values to the user's
# configuration folder, so "Restore published" and "Reset all" can always go
# back.  The tools read the library when they start.
# --------------------------------------------------------------------------

def _f(text):
    """Float from a field, or None when empty or not a number."""
    try:
        s = str(text).strip().replace(',', '.')
        return float(s) if s else None
    except ValueError:
        return None


def _same(a, b):
    """Equal, with floats compared to 1e-9 relative (form fields round them)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
            not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-30)
    return a == b


def _g(v, sig=12):
    if v is None or v == '':
        return ''
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if 1e-7 <= abs(x) < 1e-3:
        return f'{x:.{sig + 3}f}'.rstrip('0').rstrip('.')
    return f'{x:.{sig}g}'


def _note(text):
    from arar_math import note_label      # arar_math imports this module
    return note_label(text, 11)


def _flip(ratio):
    return 'std/FCs' if ratio == 'FCs/std' else 'FCs/std'


def _ic_back(x, sid):
    """A table row shown on sid's page, as stored with its holder (x['other'])."""
    d = {'R': x['R'], 'R_1s': x.get('R_1s') or 0.0, 'ratio': _flip(x.get('ratio')),
         'reference': x.get('reference', ''), 'doi': x.get('doi', ''), 'notes': x.get('notes', '')}
    if sid != 'FCs':
        d['other'] = sid
    if x.get('recommended'):
        d['recommended'] = True
    return d


def _ic_norm(ic):
    """An intercalibration with defaults filled in, for comparison."""
    g = lambda v: float(f'{float(v or 0.0):.10g}')
    return (ic_other(ic), ic.get('ratio') or 'std/FCs', g(ic['R']),
            g(ic.get('R_1s')), bool(ic.get('recommended')),
            ic.get('reference') or '', ic.get('doi') or '', ic.get('notes') or '')


class _Table(QTableWidget):
    """A small editable table of dicts with fixed columns."""

    def __init__(self, columns, parent=None):
        super().__init__(0, len(columns), parent)
        self.columns = columns            # [(key, header, kind)], kind 'num' | 'text' | 'lam'
        self.setHorizontalHeaderLabels([c[1] for c in columns])
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        h = self.horizontalHeader()
        for i, (_, _, kind) in enumerate(columns):
            h.setSectionResizeMode(i, QHeaderView.Stretch if kind == 'text'
                                   else QHeaderView.ResizeToContents)
        self.lam_names = []
        self.std_names = []

    def set_rows(self, rows):
        self.setRowCount(0)
        for r in rows:
            self.add_row(r)

    def add_row(self, r=None):
        r = r or {}
        i = self.rowCount()
        self.insertRow(i)
        for j, (key, _, kind) in enumerate(self.columns):
            if kind == 'lam':
                cb = QComboBox()
                cb.addItems(['(not needed)'] + self.lam_names)
                v = r.get(key)
                cb.setCurrentIndex(self.lam_names.index(v) + 1 if v in self.lam_names else 0)
                self.setCellWidget(i, j, cb)
            elif kind == 'ratio':
                cb = QComboBox()
                cb.addItems(['this / other', 'other / this'])
                cb.setCurrentIndex(1 if r.get(key) == 'FCs/std' else 0)
                self.setCellWidget(i, j, cb)
            elif kind == 'std':
                cb = QComboBox()
                cb.addItems(self.std_names)
                v = r.get(key) or 'FCs'
                cb.setCurrentIndex(max(0, cb.findText(v)))
                self.setCellWidget(i, j, cb)
            elif kind == 'check':
                it = QTableWidgetItem()
                it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                it.setCheckState(Qt.Checked if r.get(key) else Qt.Unchecked)
                self.setItem(i, j, it)
            else:
                v = r.get(key)
                it = QTableWidgetItem(_g(v) if kind == 'num' else (v or ''))
                if kind == 'text' and v:
                    it.setToolTip(str(v))
                self.setItem(i, j, it)

    def remove_selected(self):
        rows = sorted({i.row() for i in self.selectedIndexes()}, reverse=True)
        for r in rows:
            self.removeRow(r)

    def rows(self):
        out = []
        for i in range(self.rowCount()):
            r = {}
            for j, (key, _, kind) in enumerate(self.columns):
                if kind == 'lam':
                    cb = self.cellWidget(i, j)
                    r[key] = None if cb.currentIndex() == 0 else cb.currentText()
                elif kind == 'ratio':
                    r[key] = 'FCs/std' if self.cellWidget(i, j).currentIndex() == 1 else 'std/FCs'
                elif kind == 'std':
                    r[key] = self.cellWidget(i, j).currentText()
                elif kind == 'check':
                    r[key] = self.item(i, j).checkState() == Qt.Checked
                else:
                    it = self.item(i, j)
                    t = it.text().strip() if it else ''
                    r[key] = _f(t) if kind == 'num' else t
            out.append(r)
        return out


class _Page(QWidget):
    """One section of the library: a list on the left, a form on the right."""

    section = ''
    noun = ''

    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self._cur = None
        lay = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        lay.addWidget(split)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_select)
        ll.addWidget(self.list)
        row = QHBoxLayout()
        for txt, fn in [('Add', self._add), ('Duplicate', self._duplicate),
                        ('Delete', self._delete)]:
            b = QPushButton(txt)
            b.clicked.connect(fn)
            row.addWidget(b)
        ll.addLayout(row)
        self.btn_restore = QPushButton('Restore published values')
        self.btn_restore.clicked.connect(self._restore)
        ll.addWidget(self.btn_restore)
        split.addWidget(left)
        self.form = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.form)
        split.addWidget(scroll)
        split.setStretchFactor(1, 3)
        split.setSizes([300, 860])
        self.build_form(self.form)

    # --- library access --------------------------------------------------
    @property
    def rows(self):
        return self.editor.lib[self.section]

    def key(self):
        return key_of(self.section)

    def label(self, r):
        return self.list_text(r) + (' (added)' if r.get('user') else '')

    def list_text(self, r):
        return r.get(self.key(), '')

    def hidden(self, r):
        """Entries kept in the library but not listed on this page."""
        return False

    def refresh_list(self, select=None):
        self.list.blockSignals(True)
        self.list.clear()
        for k, r in enumerate(self.rows):
            self.list.addItem(QListWidgetItem(self.label(r)))
            self.list.setRowHidden(k, self.hidden(r))
        self.list.blockSignals(False)
        shown = [k for k, r in enumerate(self.rows) if not self.hidden(r)]
        if shown:
            i = select if select is not None else max(0, min(self.list.count() - 1, self._cur or 0))
            if i not in shown:
                i = min(shown, key=lambda k: (k < i, abs(k - i)))
            self._cur = None
            self.list.setCurrentRow(i)
            self._on_select(i)
        else:
            self._cur = None
            self.load({})

    def _on_select(self, i):
        if self._cur is not None and 0 <= self._cur < len(self.rows):
            self.commit()
        self._cur = i if 0 <= i < len(self.rows) else None
        if self._cur is not None:
            r = self.rows[self._cur]
            self.load(r)
            self.btn_restore.setEnabled(bool(r.get('edited')))

    def commit(self):
        """Copy the form into the selected entry."""
        if self._cur is None or not (0 <= self._cur < len(self.rows)):
            return True
        old = self.rows[self._cur]
        new = self.save(copy.deepcopy(old))
        new.pop('edited', None)
        new.pop('user', None)
        published = find(self.editor.published, self.section, new.get(self.key()))
        if published is not None and _same(new, published):
            new = copy.deepcopy(published)
        elif _same(new, _clean(old)):
            new = copy.deepcopy(_clean(old))
        edited = published is not None and new != published
        new['user'] = published is None
        new['edited'] = edited
        if _clean(new) != _clean(old):
            self.rows[self._cur] = new
            self.editor.dirty = True
            item = self.list.item(self._cur)
            if item is not None:
                item.setText(self.label(new))
            self.btn_restore.setEnabled(bool(new.get('edited')))
        return True

    def _unique(self, base):
        names = {r.get(self.key()) for r in self.rows}
        name, n = base, 2
        while name in names:
            name, n = f'{base} ({n})', n + 1
        return name

    def _add(self):
        self.commit()
        name, ok = QInputDialog.getText(self, f'New {self.noun}', f'Name of the new {self.noun}:')
        if not ok or not name.strip():
            return
        r = self.blank()
        r[self.key()] = self._unique(name.strip())
        r['user'] = True
        self.rows.append(r)
        self.editor.dirty = True
        self.refresh_list(len(self.rows) - 1)

    def _duplicate(self):
        if self._cur is None:
            return
        self.commit()
        r = copy.deepcopy(self.rows[self._cur])
        r[self.key()] = self._unique(r.get(self.key(), '') + ' copy')
        r.pop('default', None)
        r['user'], r['edited'] = True, False
        self.rows.append(r)
        self.editor.dirty = True
        self.refresh_list(len(self.rows) - 1)

    def _delete(self):
        if self._cur is None:
            return
        r = self.rows[self._cur]
        name = r.get(self.key())
        extra = ('' if r.get('user') else
                 '\n\nIt is one of the published entries; "Reset all" brings it back.')
        if QMessageBox.question(self, 'Delete', f'Delete "{name}"?{extra}') != QMessageBox.Yes:
            return
        del self.rows[self._cur]
        self._cur = None
        self.editor.dirty = True
        self.refresh_list()

    def _restore(self):
        if self._cur is None:
            return
        r = self.rows[self._cur]
        pub = find(self.editor.published, self.section, r.get(self.key()))
        if pub is None:
            return
        i = self._cur
        self.rows[i] = copy.deepcopy(pub)
        self._cur = None
        self.editor.dirty = True
        self.refresh_list(i)

    # --- to implement ------------------------------------------------------
    def build_form(self, w):
        raise NotImplementedError

    def blank(self):
        return {}

    def load(self, r):
        raise NotImplementedError

    def save(self, r):
        raise NotImplementedError


def _ref_fields(grid, row):
    """Reference, DOI and notes fields, added to grid from row; returns them."""
    ref, doi, notes = QLineEdit(), QLineEdit(), QPlainTextEdit()
    notes.setFixedHeight(60)
    grid.addWidget(QLabel('Reference'), row, 0)
    grid.addWidget(ref, row, 1, 1, 3)
    grid.addWidget(QLabel('DOI'), row + 1, 0)
    grid.addWidget(doi, row + 1, 1, 1, 3)
    grid.addWidget(QLabel('Notes'), row + 2, 0, Qt.AlignTop)
    grid.addWidget(notes, row + 2, 1, 1, 3)
    return ref, doi, notes


class DecayPage(_Page):
    section = 'decay_constants'
    noun = 'decay constant'

    def build_form(self, w):
        g = QGridLayout(w)
        self.name, self.lam, self.lam_s = QLineEdit(), QLineEdit(), QLineEdit()
        g.addWidget(QLabel('Name'), 0, 0)
        g.addWidget(self.name, 0, 1, 1, 3)
        g.addWidget(QLabel('λ total (yr⁻¹)'), 1, 0)
        g.addWidget(self.lam, 1, 1)
        g.addWidget(QLabel('±1σ (yr⁻¹)'), 1, 2)
        g.addWidget(self.lam_s, 1, 3)
        self.ref, self.doi, self.notes = _ref_fields(g, 2)
        g.addWidget(_note('The total decay constant of ⁴⁰K. A calibration names the '
                          'decay constant it was derived with on the Calibrations page.'), 5, 0, 1, 4)
        g.setRowStretch(6, 1)

    def blank(self):
        return {'lambda': 5.463e-10, 'lambda_1s': 0.0, 'reference': '', 'doi': '', 'notes': ''}

    def load(self, r):
        self.name.setText(r.get('name', ''))
        self.name.setReadOnly(not r.get('user', False))
        self.lam.setText(_g(r.get('lambda')))
        self.lam_s.setText(_g(r.get('lambda_1s')))
        self.ref.setText(r.get('reference', ''))
        self.ref.setCursorPosition(0)
        self.doi.setText(r.get('doi', ''))
        self.notes.setPlainText(r.get('notes', ''))

    def save(self, r):
        r['name'] = self.name.text().strip() or r.get('name', '')
        r['lambda'] = _f(self.lam.text())
        r['lambda_1s'] = _f(self.lam_s.text()) or 0.0
        r['reference'] = self.ref.text().strip()
        r['doi'] = self.doi.text().strip()
        r['notes'] = self.notes.toPlainText().strip()
        return r


class CalibrationPage(_Page):
    section = 'calibrations'
    noun = 'calibration'

    def list_text(self, r):
        return f"{r.get('standard', 'FCs')} — {r.get('name', '')}"

    def build_form(self, w):
        g = QGridLayout(w)
        self.name, self.age, self.age_s, self.cov = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.lam = QComboBox()
        self.basis = QComboBox()
        for key, txt in [('astronomical', 'Astronomical anchor'),
                         ('joint', 'Joint calibration with λ'),
                         ('k-ar', 'K–Ar standard or intercalibration')]:
            self.basis.addItem(txt, key)
        self.basis.currentIndexChanged.connect(lambda _i: self._basis_changed(self.basis.currentData()))
        self.incl = QCheckBox('The ± already includes the uncertainty of λ')
        self.default = QCheckBox('Default calibration of all tools')
        self.std = QComboBox()
        self.std.currentTextChanged.connect(lambda t: self.default.setEnabled(t == 'FCs'))
        g.addWidget(QLabel('Name'), 0, 0)
        g.addWidget(self.name, 0, 1)
        g.addWidget(QLabel('Standard'), 0, 2)
        g.addWidget(self.std, 0, 3)
        g.addWidget(QLabel('Age (Ma)'), 1, 0)
        g.addWidget(self.age, 1, 1)
        g.addWidget(QLabel('±1σ (Ma)'), 1, 2)
        g.addWidget(self.age_s, 1, 3)
        g.addWidget(QLabel('Decay constant'), 2, 0)
        g.addWidget(self.lam, 2, 1, 1, 3)
        g.addWidget(QLabel('Determined by'), 3, 0)
        g.addWidget(self.basis, 3, 1)
        g.addWidget(self.incl, 4, 1, 1, 3)
        g.addWidget(QLabel('cov(λ, t)'), 5, 0)
        g.addWidget(self.cov, 5, 1)
        g.addWidget(_note('yr⁻¹·yr; joint calibrations only. Leave empty when none is '
                          'published: the age and λ are then treated as independent.'), 5, 2, 1, 2)
        self.anchor_box = QGroupBox('Astronomical anchor')
        ag = QGridLayout(self.anchor_box)
        self.a_name, self.a_age, self.a_age_s = QLineEdit(), QLineEdit(), QLineEdit()
        self.a_R, self.a_R_s = QLineEdit(), QLineEdit()
        self.a_std = QComboBox()
        ag.addWidget(QLabel('Anchor'), 0, 0)
        ag.addWidget(self.a_name, 0, 1)
        ag.addWidget(QLabel('Standard'), 0, 2)
        ag.addWidget(self.a_std, 0, 3)
        ag.addWidget(QLabel('Astronomical age (Ma)'), 1, 0)
        ag.addWidget(self.a_age, 1, 1)
        ag.addWidget(QLabel('±1σ'), 1, 2)
        ag.addWidget(self.a_age_s, 1, 3)
        ag.addWidget(QLabel('R (standard / anchor)'), 2, 0)
        ag.addWidget(self.a_R, 2, 1)
        ag.addWidget(QLabel('±1σ'), 2, 2)
        ag.addWidget(self.a_R_s, 2, 3)
        ag.addWidget(_note('The external uncertainty of an age on this calibration is '
                           'propagated from the anchor age, R and λ.'),
                     3, 0, 1, 4)
        g.addWidget(self.anchor_box, 6, 0, 1, 4)
        g.addWidget(self.default, 7, 1, 1, 3)
        self.ref, self.doi, self.notes = _ref_fields(g, 8)
        self.derived = _note('')
        g.addWidget(self.derived, 11, 0, 1, 4)
        g.setRowStretch(12, 1)

    def _basis_changed(self, b):
        self.anchor_box.setEnabled(b == 'astronomical')
        self.cov.setEnabled(b == 'joint')

    def blank(self):
        lam = self.editor.lib['decay_constants'][0]['name'] if self.editor.lib['decay_constants'] else ''
        return {'standard': 'FCs', 'age': 28.0, 'age_1s': 0.0, 'decay_constant': lam,
                'basis': 'k-ar', 'sigma_includes_lambda': False,
                'reference': '', 'doi': '', 'notes': ''}

    def load(self, r):
        self.lam.blockSignals(True)
        self.lam.clear()
        self.lam.addItems([d['name'] for d in self.editor.lib['decay_constants']])
        i = self.lam.findText(r.get('decay_constant', ''))
        self.lam.setCurrentIndex(max(i, 0))
        self.lam.blockSignals(False)
        self.std.blockSignals(True)
        self.std.clear()
        self.std.addItems([st['id'] for st in self.editor.lib['standards']])
        self.std.setCurrentIndex(max(0, self.std.findText(r.get('standard', 'FCs'))))
        self.std.blockSignals(False)
        self.default.setEnabled(self.std.currentText() == 'FCs')
        self.name.setText(r.get('name', ''))
        self.name.setReadOnly(not r.get('user', False))
        self.age.setText(_g(r.get('age')))
        self.age_s.setText(_g(r.get('age_1s')))
        self.basis.setCurrentIndex(max(0, self.basis.findData(r.get('basis', 'k-ar'))))
        self._basis_changed(self.basis.currentData())
        self.incl.setChecked(bool(r.get('sigma_includes_lambda')))
        self.cov.setText(_g(r.get('covariance')))
        a = r.get('anchor') or {}
        self.a_std.clear()
        self.a_std.addItem('(none)', None)
        for st in self.editor.lib['standards']:
            self.a_std.addItem(st['id'], st['id'])
        self.a_std.setCurrentIndex(max(0, self.a_std.findData(a.get('standard'))))
        self.a_name.setText(a.get('name', ''))
        self.a_age.setText(_g(a.get('age')))
        self.a_age_s.setText(_g(a.get('age_1s')))
        self.a_R.setText(_g(a.get('R_fcs_anchor')))
        self.a_R_s.setText(_g(a.get('R_1s')))
        self.default.setChecked(bool(r.get('default')))
        self.ref.setText(r.get('reference', ''))
        self.ref.setCursorPosition(0)
        self.doi.setText(r.get('doi', ''))
        self.notes.setPlainText(r.get('notes', ''))
        self.derived.setText(self._derived(r))

    def _derived(self, r):
        a = r.get('anchor') or {}
        lam = next((d.get('lambda') for d in self.editor.lib['decay_constants']
                    if d.get('name') == r.get('decay_constant')), None)
        try:
            if r.get('basis') == 'astronomical' and a and lam:
                t = math.log1p(float(a['R_fcs_anchor']) * math.expm1(lam * float(a['age']) * 1e6)) / lam / 1e6
                return (f"From the anchor: {r.get('standard', 'FCs')} = {t:.4f} Ma with this decay constant. "
                        'The printed R is usually rounded, so this can differ from the '
                        'published FCs age in the last digit; SimplAr uses the R that '
                        'reproduces the published age.')
        except (KeyError, TypeError, ValueError):
            pass
        return ''

    def save(self, r):
        r['name'] = self.name.text().strip() or r.get('name', '')
        r['standard'] = self.std.currentText() or 'FCs'
        r['age'] = _f(self.age.text())
        r['age_1s'] = _f(self.age_s.text()) or 0.0
        r['decay_constant'] = self.lam.currentText()
        r['basis'] = self.basis.currentData()
        r['sigma_includes_lambda'] = self.incl.isChecked()
        cov = _f(self.cov.text())
        if r['basis'] == 'joint' and cov is not None:
            r['covariance'] = cov
        else:
            r.pop('covariance', None)
        if r['basis'] == 'astronomical' and self.a_age.text().strip():
            r['anchor'] = {'name': self.a_name.text().strip(), 'age': _f(self.a_age.text()),
                           'standard': self.a_std.currentData(),
                           'age_1s': _f(self.a_age_s.text()) or 0.0,
                           'R_fcs_anchor': _f(self.a_R.text()),
                           'R_1s': _f(self.a_R_s.text()) or 0.0}
        else:
            r.pop('anchor', None)
        if self.default.isChecked() and r['standard'] == 'FCs':
            r['default'] = True
            for other in self.rows:
                if other is not self.rows[self._cur] and other.get('default'):
                    other.pop('default')
                    other['edited'] = True
        else:
            r.pop('default', None)
        r['reference'] = self.ref.text().strip()
        r['doi'] = self.doi.text().strip()
        r['notes'] = self.notes.toPlainText().strip()
        return r


class StandardPage(_Page):
    section = 'standards'
    noun = 'monitor'

    def list_text(self, r):
        return f"{r.get('id', '')} — {r.get('name', '')}"


    def build_form(self, w):
        v = QVBoxLayout(w)
        g = QGridLayout()
        self.sid, self.name, self.mineral, self.loc = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.notes = QPlainTextEdit()
        self.notes.setFixedHeight(46)
        g.addWidget(QLabel('Short name'), 0, 0)
        g.addWidget(self.sid, 0, 1)
        g.addWidget(QLabel('Mineral'), 0, 2)
        g.addWidget(self.mineral, 0, 3)
        g.addWidget(QLabel('Name'), 1, 0)
        g.addWidget(self.name, 1, 1, 1, 3)
        g.addWidget(QLabel('Locality'), 2, 0)
        g.addWidget(self.loc, 2, 1, 1, 3)
        g.addWidget(QLabel('Notes'), 3, 0, Qt.AlignTop)
        g.addWidget(self.notes, 3, 1, 1, 3)
        v.addLayout(g)

        box = QGroupBox('R relative to other standards (⁴⁰Ar*/³⁹Ar_K), as published')
        bl = QVBoxLayout(box)
        self.ics = _Table([('other', 'Other', 'std'), ('ratio', 'Ratio', 'ratio'),
                           ('R', 'R', 'num'), ('R_1s', '±1σ', 'num'),
                           ('recommended', 'Use', 'check'), ('reference', 'Reference', 'text'),
                           ('doi', 'DOI', 'text'), ('notes', 'Notes', 'text')])
        self.ics.itemChanged.connect(self._one_recommended)
        self.ics.setMinimumHeight(220)
        bl.addWidget(self.ics)
        bl.addLayout(self._row_buttons(self.ics))
        bl.addWidget(_note('Use marks the R that default routes take, one for each pair of '
                           'standards. Each R shows on the pages of both its standards; an edit on '
                           'either page changes it on both. A value of your own lab can be added '
                           'as a row and used in a route like a published one.'))
        v.addWidget(box, 1)

        box2 = QGroupBox('Published ages')
        bl2 = QVBoxLayout(box2)
        self.ages = _Table([('age', 'Age (Ma)', 'num'), ('age_1s', '±1σ', 'num'),
                            ('decay_constant', 'Decay constant used', 'lam'),
                            ('reference', 'Reference', 'text'), ('doi', 'DOI', 'text'),
                            ('notes', 'Notes', 'text')])
        self.ages.setMinimumHeight(120)
        bl2.addWidget(self.ages)
        bl2.addLayout(self._row_buttons(self.ages))
        bl2.addWidget(_note('The ages authors have used for this monitor. The Recalculator and age '
                            'combiner offers them as the monitor age of a published dataset.'))
        self._cals_lbl = QLabel('Calibrations of this standard (kept under Calibrations; '
                                'double-click to open there):')
        self._cals_lbl.setWordWrap(True)
        self.cals = QListWidget()
        self.cals.setMaximumHeight(110)
        self.cals.itemDoubleClicked.connect(self._go_to_calibration)
        bl2.addWidget(self._cals_lbl)
        bl2.addWidget(self.cals)
        v.addWidget(box2, 1)
        self.derived = _note('')
        v.addWidget(self.derived)

    def _go_to_calibration(self, item):
        name = self._cal_names[self.cals.row(item)]
        ed = self.editor
        page = next(p for p in ed.pages if p.section == 'calibrations')
        ed.tabs.setCurrentWidget(page)
        names = [r.get('name') for r in page.rows]
        if name in names:
            page.list.setCurrentRow(names.index(name))

    def _mark_holder(self, i, holder):
        """A row stored with another standard: its Other is that standard."""
        self.ics.item(i, 4).setData(Qt.UserRole, holder)
        if holder:
            self.ics.cellWidget(i, 0).setEnabled(False)
            self.ics.cellWidget(i, 0).setToolTip(f'Stored with {holder}')

    def commit(self):
        """Also write back the rows stored with other standards."""
        sid = getattr(self, '_sid', None)
        ok = super().commit()
        out, self._foreign_out = getattr(self, '_foreign_out', {}), {}
        if not sid or self._cur is None:
            return ok
        for holder, xs in out.items():
            k = next((i for i, st in enumerate(self.rows) if st.get('id') == holder), None)
            if k is None:
                continue
            old = self.rows[k]
            new_ics = [_ic_back(x, sid) for x in xs]
            mine = [ic for ic in old.get('intercalibrations') or [] if ic_other(ic) == sid]
            if len(mine) == len(new_ics) and all(_ic_norm(a) == _ic_norm(b) for a, b in zip(mine, new_ics)):
                continue
            it = iter(new_ics)
            ics = []
            for ic in old.get('intercalibrations') or []:
                if ic_other(ic) != sid:
                    ics.append(ic)
                else:
                    nxt = next(it, None)
                    if nxt is not None:
                        ics.append(nxt)
            ics += list(it)
            new = copy.deepcopy(_clean(old))
            new['intercalibrations'] = ics
            published = find(self.editor.published, self.section, holder)
            if published is not None and _same(new, published):
                new = copy.deepcopy(published)
            new['user'] = published is None
            new['edited'] = published is not None and new != published
            self.rows[k] = new
            self.editor.dirty = True
        return ok

    def _row_buttons(self, table):
        row = QHBoxLayout()
        add, rem = QPushButton('Add row'), QPushButton('Remove row')
        add.clicked.connect(lambda: table.add_row())
        rem.clicked.connect(table.remove_selected)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch()
        return row

    def _one_recommended(self, item):
        if item.column() != 4 or item.checkState() != Qt.Checked:
            return
        other = self.ics.cellWidget(item.row(), 0).currentText()
        self.ics.blockSignals(True)
        for i in range(self.ics.rowCount()):
            if i != item.row() and self.ics.cellWidget(i, 0).currentText() == other:
                self.ics.item(i, 4).setCheckState(Qt.Unchecked)
        self.ics.blockSignals(False)

    def blank(self):
        return {'name': '', 'mineral': '', 'locality': '', 'notes': '',
                'intercalibrations': [], 'published_ages': []}

    def load(self, r):
        self.sid.setText(r.get('id', ''))
        self.sid.setReadOnly(not r.get('user', False))
        self.name.setText(r.get('name', ''))
        self.mineral.setText(r.get('mineral', ''))
        self.loc.setText(r.get('locality', ''))
        self.loc.setCursorPosition(0)
        self.notes.setPlainText(r.get('notes', ''))
        self.ages.lam_names = [d['name'] for d in self.editor.lib['decay_constants']]
        sid = r.get('id', '')
        self.ics.std_names = [st['id'] for st in self.editor.lib['standards'] if st['id'] != sid]
        ics = r.get('intercalibrations') or []
        self._sid = sid
        self._self_ics = [x for x in ics if ic_other(x) == sid]
        rows = [x for x in ics if ic_other(x) != sid]
        holders = [None] * len(rows)
        for st in self.editor.lib['standards']:
            if st.get('id') == sid:
                continue
            for ic in st.get('intercalibrations') or []:
                if ic_other(ic) == sid:
                    rows.append(dict(ic, other=st['id'], ratio=_flip(ic.get('ratio'))))
                    holders.append(st['id'])
        self._holders_shown = {h for h in holders if h}
        self.ics.blockSignals(True)
        self.ics.set_rows(rows)
        for i, h in enumerate(holders):
            self._mark_holder(i, h)
        self.ics.blockSignals(False)
        self.ages.set_rows(r.get('published_ages') or [])
        self.cals.clear()
        self._cal_names = []
        for c in calibrations_of(self.editor.lib, sid):
            self.cals.addItem(f"{pm_text(float(c['age']), float(c.get('age_1s') or 0))} Ma  ·  "
                              f"{c['name']}  ·  {c.get('decay_constant') or 'no λ'}")
            self._cal_names.append(c['name'])
        self.cals.setVisible(bool(self._cal_names))
        self._cals_lbl.setVisible(bool(self._cal_names))
        self.derived.setText(self._derived(r))

    def _derived(self, r):
        if not r.get('id'):
            return ''
        route = default_route(self.editor.lib, r['id'])
        if route is None:
            return 'No route to a calibration: add an R against a standard that has one.'
        try:
            import arar_math as am
            t = am.monitor_age(am.route_cfg(route, self.editor.lib))
        except Exception:
            return ''
        return f"Default route: {route_line(route)}. On it, {r['id']} is {t:.6g} Ma."

    def save(self, r):
        r['id'] = self.sid.text().strip() or r.get('id', '')
        r['name'] = self.name.text().strip()
        r['mineral'] = self.mineral.text().strip()
        r['locality'] = self.loc.text().strip()
        r['notes'] = self.notes.toPlainText().strip()
        ics = []
        self._foreign_out = {h: [] for h in getattr(self, '_holders_shown', set())}
        for i, x in enumerate(self.ics.rows()):
            if x.get('R') is None:
                continue
            holder = self.ics.item(i, 4).data(Qt.UserRole)
            if holder:
                self._foreign_out.setdefault(holder, []).append(x)
                continue
            d = {'R': x['R'], 'R_1s': x.get('R_1s') or 0.0, 'ratio': x.get('ratio', 'std/FCs'),
                 'reference': x.get('reference', ''),
                 'doi': x.get('doi', ''), 'notes': x.get('notes', '')}
            if (x.get('other') or 'FCs') != 'FCs':
                d['other'] = x['other']
            if x.get('recommended'):
                d['recommended'] = True
            ics.append(d)
        r['intercalibrations'] = getattr(self, '_self_ics', []) + ics
        r['published_ages'] = [
            {'age': x['age'], 'age_1s': x.get('age_1s') or 0.0,
             'decay_constant': x.get('decay_constant'), 'reference': x.get('reference', ''),
             'doi': x.get('doi', ''), 'notes': x.get('notes', '')}
            for x in self.ages.rows() if x.get('age') is not None]
        return r


class RoutePage(_Page):
    """Routes: the monitor, steps through R values in the library, and a
    calibration of the standard the steps end at."""
    section = 'routes'
    noun = 'route'

    def label(self, r):
        tag = '  (default)' if r.get('default') else ''
        return f"{r.get('monitor', '')} — {r.get('name', '')}{tag}"

    def build_form(self, w):
        g = QGridLayout(w)
        self.name, self.mon, self.cal, self.lam = QLineEdit(), QComboBox(), QComboBox(), QComboBox()
        self.default = QCheckBox('Default route for this monitor in all tools')
        self.steps = QListWidget()
        self.steps.setMaximumHeight(130)
        self.next = QComboBox()
        b_add, b_rem = QPushButton('Add step'), QPushButton('Remove last step')
        b_add.clicked.connect(self._add_step)
        b_rem.clicked.connect(self._remove_step)
        self.mon.currentTextChanged.connect(self._monitor_changed)
        self.cal.currentIndexChanged.connect(self._cal_changed)
        self.lam.currentIndexChanged.connect(lambda _i: self._show())
        g.addWidget(QLabel('Name'), 0, 0)
        g.addWidget(self.name, 0, 1, 1, 3)
        g.addWidget(QLabel('Monitor'), 1, 0)
        g.addWidget(self.mon, 1, 1)
        g.addWidget(QLabel('Steps'), 2, 0, Qt.AlignTop)
        g.addWidget(self.steps, 2, 1, 1, 3)
        g.addWidget(QLabel('Next step'), 3, 0)
        g.addWidget(self.next, 3, 1)
        g.addWidget(b_add, 3, 2)
        g.addWidget(b_rem, 3, 3)
        g.addWidget(_note('Only R values in the library are offered: one measured between the '
                          'standard the route has reached and another it has not visited yet. '
                          'Add your own R under Monitors to use it here.'), 4, 1, 1, 3)
        g.addWidget(QLabel('Calibration'), 5, 0)
        g.addWidget(self.cal, 5, 1, 1, 3)
        g.addWidget(QLabel('Decay constant'), 6, 0)
        g.addWidget(self.lam, 6, 1, 1, 3)
        g.addWidget(self.default, 7, 1, 1, 3)
        self.derived = QLabel('')
        self.derived.setWordWrap(True)
        self.derived.setTextInteractionFlags(Qt.TextSelectableByMouse)
        g.addWidget(self.derived, 8, 0, 1, 4)
        g.setRowStretch(9, 1)
        self._steps = []

    # --- the route being edited ------------------------------------------
    def _end(self):
        return self._steps[-1]['to'] if self._steps else self.mon.currentText()

    def _refill(self, cal_name=None, lam_name=None):
        lib = self.editor.lib
        self.steps.clear()
        for st in self._steps:
            self.steps.addItem(f"{st['from']} → {st['to']}:  R({st['num']}/{st['den']}) = "
                               f"{pm_text(st['R'], st.get('R_1s') or 0.0)}, "
                               f"{st.get('source') or 'no source'}")
        visited = [self.mon.currentText()] + [st['to'] for st in self._steps]
        self._cands = steps_from(lib, self._end(), visited)
        self.next.clear()
        for st in self._cands:
            self.next.addItem(f"→ {st['to']}:  R({st['num']}/{st['den']}) = "
                              f"{pm_text(st['R'], st.get('R_1s') or 0.0)}, "
                              f"{st.get('source') or 'no source'}")
        self.next.setEnabled(bool(self._cands))
        cals = calibrations_of(lib, self._end())
        self.cal.blockSignals(True)
        self.cal.clear()
        for c in cals:
            self.cal.addItem(c['name'])
        if not cals:
            self.cal.addItem(f'(no calibration of {self._end()}: add a step)')
        i = self.cal.findText(cal_name) if cal_name else -1
        if i < 0:
            dc = default_calibration(lib, self._end())
            i = self.cal.findText(dc['name']) if dc else 0
        self.cal.setCurrentIndex(max(0, i))
        self.cal.setEnabled(bool(cals))
        self.cal.blockSignals(False)
        if lam_name:
            self.lam.setCurrentIndex(max(0, self.lam.findText(lam_name)))
        else:
            self._cal_changed()
        self._show()

    def _monitor_changed(self, _t):
        if self._steps and self._steps[0]['from'] != self.mon.currentText():
            self._steps = []
        self._refill()

    def _cal_changed(self, *_):
        c = find(self.editor.lib, 'calibrations', self.cal.currentText())
        if c is not None:
            self.lam.blockSignals(True)
            self.lam.setCurrentIndex(max(0, self.lam.findText(c.get('decay_constant', ''))))
            self.lam.blockSignals(False)
        self._show()

    def _add_step(self):
        i = self.next.currentIndex()
        if 0 <= i < len(self._cands):
            self._steps.append(copy.deepcopy(self._cands[i]))
            self._refill()

    def _remove_step(self):
        if self._steps:
            self._steps.pop()
            self._refill()

    def _route(self):
        return {'name': self.name.text().strip(), 'monitor': self.mon.currentText(),
                'steps': copy.deepcopy(self._steps), 'calibration': self.cal.currentText(),
                'lambda': self.lam.currentText()}

    def _show(self):
        r = self._route()
        errs = route_problems(r, self.editor.lib)
        if errs:
            self.derived.setText('Not usable yet: ' + '; '.join(errs) + '.')
            return
        try:
            import arar_math as am
            c = am.route_cfg(r, self.editor.lib)
            t = am.monitor_age(c)
        except Exception as e:
            self.derived.setText(f'Not usable: {e}')
            return
        prod = ' · '.join(f"{st['R']:g}" if st['num'] == st['from'] else f"1/{st['R']:g}"
                          for st in self._steps)
        end = self._end()
        eq = (f"e^(λ t_{r['monitor']}) − 1 = {prod + ' · ' if prod else ''}(e^(λ t_{end}) − 1)"
              if self._steps else '')
        self.derived.setText(f"{route_line(r)}\n{eq}\n{r['monitor']} = {t:.6g} Ma on this route."
                             .replace('\n\n', '\n'))

    # --- _Page -----------------------------------------------------------
    def blank(self):
        cal = default_calibration(self.editor.lib, 'FCs') or {}
        return {'monitor': 'FCs', 'steps': [], 'calibration': cal.get('name', ''),
                'lambda': cal.get('decay_constant', '')}

    def load(self, r):
        lib = self.editor.lib
        self.name.setText(r.get('name', ''))
        self.name.setReadOnly(not r.get('user', False) and bool(r))
        self.mon.blockSignals(True)
        self.mon.clear()
        self.mon.addItems([st['id'] for st in lib['standards']])
        self.mon.setCurrentIndex(max(0, self.mon.findText(r.get('monitor', 'FCs'))))
        self.mon.blockSignals(False)
        self.lam.blockSignals(True)
        self.lam.clear()
        self.lam.addItems([d['name'] for d in lib['decay_constants']])
        self.lam.blockSignals(False)
        self.default.setChecked(bool(r.get('default')))
        self._steps = copy.deepcopy(r.get('steps') or [])
        self._refill(r.get('calibration'), r.get('lambda'))

    def save(self, r):
        r.update(self._route())
        r['name'] = r['name'] or r.get('name', '')
        if self.default.isChecked():
            r['default'] = True
            for other in self.rows:
                if (other is not self.rows[self._cur] and other.get('default')
                        and other.get('monitor') == r['monitor']):
                    other.pop('default')
        else:
            r.pop('default', None)
        return r


class StandardsEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Standards')
        self.resize(1180, 860)
        self.published = load_default()
        self.lib = load()
        self.dirty = False
        self.saved = False
        v = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.pages = [StandardPage(self), CalibrationPage(self), RoutePage(self), DecayPage(self)]
        for p, t in zip(self.pages, ['Monitors', 'Calibrations', 'Routes', 'Decay constants']):
            self.tabs.addTab(p, t)
        self.tabs.currentChanged.connect(self._tab_changed)
        v.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        self.path_note = _note(f'Your edits are saved in {user_path()}. '
                               'The Calibration… dialogs of the tools read them each time they open.')
        row.addWidget(self.path_note, 1)
        b_reset = QPushButton('Reset all…')
        b_reset.clicked.connect(self._reset)
        b_save = QPushButton('Save')
        b_save.setDefault(True)
        b_save.clicked.connect(self._save)
        b_close = QPushButton('Close')
        b_close.clicked.connect(self.close)
        for b in (b_reset, b_save, b_close):
            row.addWidget(b)
        v.addLayout(row)
        for p in self.pages:
            p.refresh_list(0)

    def _tab_changed(self, _i):
        for p in self.pages:
            p.commit()
        cur = self.tabs.currentWidget()
        if cur._cur is not None:
            cur.load(cur.rows[cur._cur])

    def _commit_all(self):
        for p in self.pages:
            p.commit()

    def _save(self):
        self._commit_all()
        errs = check(self.lib)
        if errs:
            QMessageBox.warning(self, 'Not saved', 'Please correct these first:\n\n• ' + '\n• '.join(errs))
            return False
        try:
            save(self.lib, self.published)
        except OSError as e:
            QMessageBox.warning(self, 'Not saved', f'Could not write the file:\n{e}')
            return False
        self.dirty = False
        self.saved = True
        self.published = load_default()
        self.lib = load()
        for pg in self.pages:
            pg.refresh_list(pg.list.currentRow() if pg.list.currentRow() >= 0 else 0)
        QMessageBox.information(self, 'Saved', 'Calibration available in tools')
        return True

    def _reset(self):
        if QMessageBox.question(self, 'Reset all',
                                'Discard every edit and go back to the published values?') != QMessageBox.Yes:
            return
        reset()
        self.lib = load()
        self.dirty = False
        self.saved = True
        for p in self.pages:
            p._cur = None
            p.refresh_list(0)

    def closeEvent(self, e):
        self._commit_all()
        if self.dirty:
            r = QMessageBox.question(self, 'Unsaved changes', 'Save the changes before closing?',
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Cancel or (r == QMessageBox.Save and not self._save()):
                e.ignore()
                return
        e.accept()


def main():
    from arar_math import force_light
    app = QApplication(sys.argv)
    app.setApplicationName('Standards')
    force_light(app)
    from arar_math import no_wheel_on_dropdowns, report_errors
    no_wheel_on_dropdowns(app)
    report_errors()
    dlg = StandardsEditor()
    dlg.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
