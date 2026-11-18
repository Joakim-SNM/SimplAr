#!/usr/bin/env python3
"""The standards library: Fish Canyon calibrations, ⁴⁰K decay constants and
neutron-fluence monitors, with its editor and the Monitor dropdown of the tools.

The published values ship in standards_default.json next to this file.  Edits
made in the Standards editor are saved to standards.json in the user's
configuration folder (SimplAr/standards.json), and only the entries that differ
from the defaults are stored there, so a later SimplAr version can still add or
correct defaults.  An entry the user deleted is stored as {"name": ...,
"deleted": true}.  The environment variable SIMPLAR_STANDARDS_FILE points the
user file elsewhere; the tests set it to an empty location so that they always
run on the published values.

Three sections, each a list of dicts keyed by "name" (standards by "id"):

  decay_constants  total ⁴⁰K decay constant (yr⁻¹) and its 1σ
  calibrations     an age of Fish Canyon sanidine (FCs) with the decay
                   constant it pairs with, how it was determined (basis
                   'astronomical', 'joint' or 'k-ar'), the astronomical anchor
                   when there is one, and cov(λ, t_FCs) for a joint calibration
  standards        a monitor with its R values relative to FCs, each stored
                   the way its paper prints it ('ratio': 'std/FCs', the
                   ⁴⁰Ar*/³⁹Ar_K of the standard over that of FCs, or 'FCs/std')
                   and the ages published for it

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
    QInputDialog,
)
from PyQt5.QtCore import Qt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, 'standards_default.json')
SECTIONS = ('decay_constants', 'calibrations', 'standards')
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


def save(lib):
    """Write the user's edits; removes the user file when nothing differs."""
    p = user_path()
    d = diff(lib)
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
    ics = std.get('intercalibrations') or []
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
    for c in lib.get('calibrations', []):
        n = c.get('name')
        if not (float(c.get('age') or 0) > 0):
            errs.append(f'Calibration "{n}": the FCs age must be positive')
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
    return errs


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
# Three pages: the monitors with their R values relative to Fish Canyon sanidine
# and their published ages, the Fish Canyon calibrations, and the ⁴⁰K decay
# constants.  Every value is 1σ.  Edits are kept in the window until Save; Save
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
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet('color:#555; font-size:11px;')
    return lab


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
                cb.addItems(['standard / FCs', 'FCs / standard'])
                cb.setCurrentIndex(1 if r.get(key) == 'FCs/std' else 0)
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
        lay = QHBoxLayout(self)
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
        split.addWidget(self.form)
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
        tag = ' (added)' if r.get('user') else (' (edited)' if r.get('edited') else '')
        return self.list_text(r) + tag

    def list_text(self, r):
        return r.get(self.key(), '')

    def refresh_list(self, select=None):
        self.list.blockSignals(True)
        self.list.clear()
        for r in self.rows:
            self.list.addItem(QListWidgetItem(self.label(r)))
        self.list.blockSignals(False)
        if self.rows:
            i = select if select is not None else max(0, min(self.list.count() - 1, self._cur or 0))
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
        g.addWidget(QLabel('Name'), 0, 0)
        g.addWidget(self.name, 0, 1, 1, 3)
        g.addWidget(QLabel('FCs age (Ma)'), 1, 0)
        g.addWidget(self.age, 1, 1)
        g.addWidget(QLabel('±1σ (Ma)'), 1, 2)
        g.addWidget(self.age_s, 1, 3)
        g.addWidget(QLabel('Decay constant'), 2, 0)
        g.addWidget(self.lam, 2, 1, 1, 3)
        g.addWidget(QLabel('Determined by'), 3, 0)
        g.addWidget(self.basis, 3, 1)
        g.addWidget(self.incl, 4, 1, 1, 3)
        g.addWidget(QLabel('cov(λ, t_FCs)'), 5, 0)
        g.addWidget(self.cov, 5, 1)
        g.addWidget(_note('yr⁻¹·yr; joint calibrations only. Leave empty when none is '
                          'published: the FCs age and λ are then treated as independent.'), 5, 2, 1, 2)
        self.anchor_box = QGroupBox('Astronomical anchor')
        ag = QGridLayout(self.anchor_box)
        self.a_name, self.a_age, self.a_age_s = QLineEdit(), QLineEdit(), QLineEdit()
        self.a_R, self.a_R_s = QLineEdit(), QLineEdit()
        self.a_std = QComboBox()
        ag.addWidget(QLabel('Anchor'), 0, 0)
        ag.addWidget(self.a_name, 0, 1)
        ag.addWidget(QLabel('Monitor'), 0, 2)
        ag.addWidget(self.a_std, 0, 3)
        ag.addWidget(QLabel('Astronomical age (Ma)'), 1, 0)
        ag.addWidget(self.a_age, 1, 1)
        ag.addWidget(QLabel('±1σ'), 1, 2)
        ag.addWidget(self.a_age_s, 1, 3)
        ag.addWidget(QLabel('R (FCs / anchor)'), 2, 0)
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
                return (f'From the anchor: FCs = {t:.4f} Ma with this decay constant. '
                        'The printed R is usually rounded, so this can differ from the '
                        'published FCs age in the last digit; SimplAr uses the R that '
                        'reproduces the published age.')
        except (KeyError, TypeError, ValueError):
            pass
        return ''

    def save(self, r):
        r['name'] = self.name.text().strip() or r.get('name', '')
        r['standard'] = 'FCs'
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
        if self.default.isChecked():
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

        box = QGroupBox('R relative to Fish Canyon sanidine (⁴⁰Ar*/³⁹Ar_K), as published')
        bl = QVBoxLayout(box)
        self.ics = _Table([('ratio', 'Ratio', 'ratio'), ('R', 'R', 'num'), ('R_1s', '±1σ', 'num'),
                           ('recommended', 'Use', 'check'), ('reference', 'Reference', 'text'),
                           ('doi', 'DOI', 'text'), ('notes', 'Notes', 'text')])
        self.ics.itemChanged.connect(self._one_recommended)
        self.ics.setMinimumHeight(120)
        bl.addWidget(self.ics)
        bl.addLayout(self._row_buttons(self.ics))
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
        v.addWidget(box2, 1)
        self.derived = _note('')
        v.addWidget(self.derived)

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
        if item.column() != 3 or item.checkState() != Qt.Checked:
            return
        self.ics.blockSignals(True)
        for i in range(self.ics.rowCount()):
            if i != item.row():
                self.ics.item(i, 3).setCheckState(Qt.Unchecked)
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
        self.ics.blockSignals(True)
        self.ics.set_rows(r.get('intercalibrations') or [])
        self.ics.blockSignals(False)
        self.ages.set_rows(r.get('published_ages') or [])
        self.derived.setText(self._derived(r))

    def _derived(self, r):
        rec = recommended_R(r) if r else None
        cal = next((c for c in self.editor.lib['calibrations'] if c.get('default')),
                   (self.editor.lib['calibrations'] or [None])[0])
        if not rec or not cal:
            return ''
        lam = next((d.get('lambda') for d in self.editor.lib['decay_constants']
                    if d.get('name') == cal.get('decay_constant')), None)
        try:
            t = math.log1p(rec[0] * math.expm1(lam * float(cal['age']) * 1e6)) / lam / 1e6
        except (TypeError, ValueError):
            return ''
        return (f"With the R marked Use, this monitor is {t:.6g} Ma on {cal['name']} "
                f"(FCs {cal['age']} Ma).")

    def save(self, r):
        r['id'] = self.sid.text().strip() or r.get('id', '')
        r['name'] = self.name.text().strip()
        r['mineral'] = self.mineral.text().strip()
        r['locality'] = self.loc.text().strip()
        r['notes'] = self.notes.toPlainText().strip()
        ics = []
        for x in self.ics.rows():
            if x.get('R') is None:
                continue
            d = {'R': x['R'], 'R_1s': x.get('R_1s') or 0.0, 'ratio': x.get('ratio', 'std/FCs'),
                 'reference': x.get('reference', ''),
                 'doi': x.get('doi', ''), 'notes': x.get('notes', '')}
            if x.get('recommended'):
                d['recommended'] = True
            ics.append(d)
        r['intercalibrations'] = ics
        r['published_ages'] = [
            {'age': x['age'], 'age_1s': x.get('age_1s') or 0.0,
             'decay_constant': x.get('decay_constant'), 'reference': x.get('reference', ''),
             'doi': x.get('doi', ''), 'notes': x.get('notes', '')}
            for x in self.ages.rows() if x.get('age') is not None]
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
        v.addWidget(_note('Every uncertainty is 1σ. R is the ⁴⁰Ar*/³⁹Ar_K of a monitor '
                          'divided by that of Fish Canyon sanidine (FCs) in the same '
                          'irradiation, so it depends on no age or decay constant. Edited '
                          'entries are marked; the published values stay available.'))
        self.tabs = QTabWidget()
        self.pages = [StandardPage(self), CalibrationPage(self), DecayPage(self)]
        for p, t in zip(self.pages, ['Monitors', 'Fish Canyon calibrations', 'Decay constants']):
            self.tabs.addTab(p, t)
        self.tabs.currentChanged.connect(self._tab_changed)
        v.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        self.path_note = _note(f'Your edits are saved in {user_path()}. '
                               'Tools read the standards when they start.')
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
            save(self.lib)
        except OSError as e:
            QMessageBox.warning(self, 'Not saved', f'Could not write the file:\n{e}')
            return False
        self.dirty = False
        self.saved = True
        self.lib = load()
        for pg in self.pages:
            pg.refresh_list(pg.list.currentRow() if pg.list.currentRow() >= 0 else 0)
        QMessageBox.information(self, 'Saved', 'Saved. Tools that are open use the new values '
                                'after they are restarted; the Recalculator and age combiner '
                                'updates at once.')
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
    dlg = StandardsEditor()
    dlg.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
