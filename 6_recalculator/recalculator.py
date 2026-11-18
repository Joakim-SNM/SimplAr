#!/usr/bin/env python3
"""Recalculator and age combiner: recast published ⁴⁰Ar/³⁹Ar ages to another
calibration, and combine ages that were published on different calibrations.

Recalculate: one dataset at a time.  Pick the monitor the authors used, the age
they gave it and their decay constant, then the target calibration; paste the
ages.  The sample's ratio to the monitor is recovered from the published age and
the age is recalculated on the target, through the monitor's R relative to Fish
Canyon sanidine when the monitor is not FCs (arar_math.recalculate; Renne et al.
1998, Mercer and Hodges 2016).  The analytical and internal errors are carried
over; the external error is that of the target calibration, propagated as in the
other SimplAr tools.  Recalculating onto the calibration the ages were published
on gives the external error that calibration implies, e.g. the astronomical
terms for Phillips et al. (2022).

Combine: ages from several datasets, each with its own source calibration, are
recalculated to one target and combined with WM + √MSWD or GLS + ML, with the J
correlation between positions (typed position labels or a loaded .jflux) and
the calibration terms shared between the ages.

Monitors, calibrations and decay constants come from the standards library
(Standards… button).
"""

import json
import math
import os
import sys

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QLabel, QComboBox, QCheckBox, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QMessageBox, QAbstractItemView,
    QDialog, QDialogButtonBox, QLineEdit, QSpinBox, QPlainTextEdit, QSplitter,
    QFormLayout,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arar_math as am
from arar_math import (
    FC_PRESETS, LAMBDA_PRESETS, STANDARDS, paired_lambda, calibration_class,
    recalculate, external_route, calibration_warning, combine, j_correlation,
    read_jflux, serialize_disk_flux, deserialize_disk_flux, match_flux_hole,
    flux_hole_R, flux_hole_rel, dt_dlnR, force_light,
    DEFAULT_FC_NAME, MIN_DECAY_DEFAULT, METHOD_WM, METHOD_GLS, edited_standards_note,
)
import data_folders as dfold
import export
import standards as stdlib
from flux_dialog import FluxAssignDialog, FluxHost


# ---------------------------------------------------------------------------
# Library helpers
# ---------------------------------------------------------------------------

def lam_value(name):
    return next((l for n, l, _ in LAMBDA_PRESETS if n == name), None)


def source_presets():
    """Every (monitor, monitor age, decay constant) combination the library
    knows, as (label, source dict).  FCs ages are the calibrations; other
    monitors their published ages.  Ages without a decay constant (astronomical
    anchors) are left out; use a custom source for them."""
    out = []
    for c in STANDARDS['calibrations']:
        lam = lam_value(c['decay_constant'])
        if lam is None:
            continue
        out.append((f"FCs {c['age']:g} Ma · {c['name'].split('  ')[0]}",
                    dict(std='FCs', std_age=float(c['age']), lam_name=c['decay_constant'],
                         lam=lam, cal=c['name'])))
    for s in STANDARDS['standards']:
        if s['id'] == 'FCs':
            continue
        for pa in s.get('published_ages') or []:
            lam = lam_value(pa.get('decay_constant'))
            if lam is None or not pa.get('age'):
                continue
            out.append((f"{s['id']} {pa['age']:g} Ma · {stdlib.short_ref(pa.get('reference'))}",
                        dict(std=s['id'], std_age=float(pa['age']),
                             lam_name=pa['decay_constant'], lam=lam)))
    return out


def fmt_lam(lam, s=None):
    """'5.463 ± 0.054 × 10⁻¹⁰ yr⁻¹'."""
    m = lam * 1e10
    if s is None:
        return f'{m:.5g} × 10⁻¹⁰ yr⁻¹'
    return f'{stdlib.pm_text(m, s * 1e10, 5)} × 10⁻¹⁰ yr⁻¹'


def source_text(src):
    if not src:
        return ''
    return (f"{src['std']} at {src['std_age']:g} Ma, λ = {fmt_lam(src['lam'])}"
            + (f", {src['lam_name']}" if src.get('lam_name') else ''))


def target_cfg(cal_name, full=False):
    """Calibration dict (arar_math cfg keys) of a calibration preset."""
    name, age, err, ext = next(r for r in FC_PRESETS if r[0] == cal_name)
    pl = paired_lambda(name)
    return dict(stdAgeMa=age, stdAgeErrMa=err,
                stdAgeErrType='external' if ext else 'internal',
                lambdaYr=pl[1], lambdaErrYr=pl[2], calibClass=calibration_class(name),
                fullSystematics=bool(full), minDecaySet=MIN_DECAY_DEFAULT)


def r_choices(std_id):
    """[(label, R, R_1s, reference)] of a monitor's intercalibrations, the
    recommended one first."""
    s = am.library_standard(std_id)
    if s is None or std_id == 'FCs':
        return [('FCs (R = 1)', 1.0, 0.0, '')]
    ics = list(s.get('intercalibrations') or [])
    ics.sort(key=lambda x: not x.get('recommended'))
    out = []
    for x in ics:
        R, Rs = stdlib.r_std_fcs(x)
        num, den = ('FCs', std_id) if x.get('ratio') == 'FCs/std' else (std_id, 'FCs')
        label = (f"R({num}/{den}) = {stdlib.pm_text(float(x['R']), float(x.get('R_1s') or 0))}"
                 f" · {stdlib.short_ref(x.get('reference'))}" + (' · recommended' if x.get('recommended') else ''))
        out.append((label, R, Rs, x.get('reference', '')))
    return out


def _num(text):
    try:
        s = str(text).strip().replace(',', '.').replace('−', '-')
        return float(s) if s else None
    except ValueError:
        return None


def _fmt(v, dec):
    return '' if v is None or not math.isfinite(v) else f'{v:.{dec}f}'


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

def _note(text):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet('color:#555; font-size:11px;')
    return lab


class PasteTable(QTableWidget):
    """Editable table that takes tab-separated text from Excel (Ctrl+V) and
    copies the selection back (Ctrl+C).  Combo cells are left alone."""

    def __init__(self, headers, rows=12, parent=None):
        super().__init__(rows, len(headers), parent)
        self.setHorizontalHeaderLabels(headers)
        self.verticalHeader().setDefaultSectionSize(22)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.text_cols = len(headers)

    def keyPressEvent(self, e):
        if e.matches(QKeySequence.Paste):
            self.paste()
        elif e.matches(QKeySequence.Copy):
            self.copy()
        elif e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and not self.state() == self.EditingState:
            for it in self.selectedItems():
                it.setText('')
        else:
            super().keyPressEvent(e)

    def copy(self):
        idx = self.selectedIndexes()
        if not idx:
            return
        rows = sorted({i.row() for i in idx})
        cols = sorted({i.column() for i in idx})
        lines = []
        for r in rows:
            cells = []
            for c in cols:
                w = self.cellWidget(r, c)
                if isinstance(w, QComboBox):
                    cells.append(w.currentText())
                else:
                    it = self.item(r, c)
                    cells.append(it.text() if it else '')
            lines.append('\t'.join(cells))
        QApplication.clipboard().setText('\n'.join(lines))

    def paste(self, text=None):
        text = QApplication.clipboard().text() if text is None else text
        lines = [ln for ln in text.replace('\r\n', '\n').replace('\r', '\n').split('\n')]
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return
        r0 = max(self.currentRow(), 0)
        c0 = max(self.currentColumn(), 0)
        self.blockSignals(True)
        for i, ln in enumerate(lines):
            r = r0 + i
            if r >= self.rowCount():
                self.insertRow(r)
                self.row_added(r)
            for j, cell in enumerate(ln.split('\t')):
                c = c0 + j
                if c >= self.text_cols or self.cellWidget(r, c) is not None:
                    continue
                self.setItem(r, c, QTableWidgetItem(cell.strip()))
        self.blockSignals(False)
        self.itemChanged.emit(self.item(r0, c0) or QTableWidgetItem())

    def row_added(self, r):
        """Hook for subclasses that put widgets in new rows."""

    def text(self, r, c):
        it = self.item(r, c)
        return it.text().strip() if it else ''


class CalibrationBox(QGroupBox):
    """Target calibration: a Fish Canyon calibration with its paired λ."""

    def __init__(self, title='Recalculate to', parent=None):
        super().__init__(title, parent)
        g = QGridLayout(self)
        self.cal = QComboBox()
        self.cal.addItems([r[0] for r in FC_PRESETS])
        i = self.cal.findText(DEFAULT_FC_NAME)
        self.cal.setCurrentIndex(max(i, 0))
        self.full = QCheckBox('Full decay-constant uncertainty (Min et al., 2000)')
        self.full.setToolTip('Replace the λ term of the external error by the uncertainty of the '
                             'two partial decay constants of Min et al. (2000), for an independent '
                             'test of the ⁴⁰K clock.')
        self.info = _note('')
        g.addWidget(QLabel('Calibration'), 0, 0)
        g.addWidget(self.cal, 0, 1)
        g.addWidget(self.full, 1, 1)
        g.addWidget(self.info, 2, 0, 1, 2)
        self.cal.currentIndexChanged.connect(self._info)
        self.full.toggled.connect(self._info)
        self._info()

    def name(self):
        return self.cal.currentText()

    def cfg(self):
        return target_cfg(self.name(), self.full.isChecked())

    def _info(self, *_):
        name, age, err, _ = next(r for r in FC_PRESETS if r[0] == self.name())
        pl = paired_lambda(name)
        txt = (f'FCs = {age:g} ± {err:g} Ma, λ = {fmt_lam(pl[1], pl[2])}, {pl[0]} (1σ). '
               f'External error: {external_route(self.cfg())}.')
        if edited_standards_note():
            txt += ' ' + edited_standards_note()
        self.info.setText(txt)

    def set_name(self, name):
        i = self.cal.findText(name)
        if i >= 0:
            self.cal.setCurrentIndex(i)


class SourceDialog(QDialog):
    """A custom source calibration: monitor, its age and the decay constant."""

    def __init__(self, parent=None, src=None):
        super().__init__(parent)
        self.setWindowTitle('Source calibration')
        f = QFormLayout(self)
        self.std = QComboBox()
        for s in STANDARDS['standards']:
            self.std.addItem(f"{s['id']} — {s['name']}", s['id'])
        self.age = QLineEdit()
        self.lam = QComboBox()
        self.lam.addItems([n for n, _, _ in LAMBDA_PRESETS] + ['Other value…'])
        self.lam_val = QLineEdit()
        self.lam_val.setPlaceholderText('yr⁻¹')
        self.lam.currentIndexChanged.connect(
            lambda i: self.lam_val.setEnabled(i == len(LAMBDA_PRESETS)))
        f.addRow('Monitor', self.std)
        f.addRow('Age the authors used (Ma)', self.age)
        f.addRow('Decay constant they used', self.lam)
        f.addRow('', self.lam_val)
        f.addRow(_note('The monitor age and λ as printed in the publication. They only need '
                       'to reproduce the published ages; whether they were right does not matter.'))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self.lam_val.setEnabled(False)
        if src:
            i = self.std.findData(src['std'])
            self.std.setCurrentIndex(max(i, 0))
            self.age.setText(f"{src['std_age']:g}")
            j = self.lam.findText(src.get('lam_name') or '')
            if j >= 0:
                self.lam.setCurrentIndex(j)
            else:
                self.lam.setCurrentIndex(len(LAMBDA_PRESETS))
                self.lam_val.setText(f"{src['lam']:.6g}")
        self.result = None

    def _ok(self):
        age = _num(self.age.text())
        if self.lam.currentIndex() < len(LAMBDA_PRESETS):
            lam_name, lam = LAMBDA_PRESETS[self.lam.currentIndex()][0], LAMBDA_PRESETS[self.lam.currentIndex()][1]
        else:
            lam_name, lam = '', _num(self.lam_val.text())
        if not age or age <= 0 or not lam or lam <= 0:
            QMessageBox.warning(self, 'Source calibration', 'Enter a positive age and decay constant.')
            return
        self.result = dict(std=self.std.currentData(), std_age=age, lam_name=lam_name, lam=lam)
        self.accept()


class SourceCombo(QComboBox):
    """Source calibrations of the library, plus custom ones."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.custom = []
        self.reload()

    def reload(self, keep=None):
        keep = keep or self.current()
        self.blockSignals(True)
        self.clear()
        for label, src in source_presets() + self.custom:
            self.addItem(label, src)
        self.addItem('Other…', None)
        self.blockSignals(False)
        self.set_source(keep)

    def current(self):
        return self.currentData()

    def set_source(self, src):
        if not src:
            i = next((i for i in range(self.count())
                      if (self.itemData(i) or {}).get('cal') == DEFAULT_FC_NAME), 0)
            self.setCurrentIndex(i)
            return
        for i in range(self.count()):
            d = self.itemData(i)
            if d and d['std'] == src['std'] and abs(d['std_age'] - src['std_age']) < 1e-9 \
                    and abs(d['lam'] - src['lam']) < 1e-20:
                self.setCurrentIndex(i)
                return
        self.add_custom(src)

    def add_custom(self, src):
        label = f"{src['std']} {src['std_age']:g} Ma · λ {src['lam']:.6g}"
        self.custom.append((label, dict(src)))
        self.blockSignals(True)
        self.insertItem(self.count() - 1, label, dict(src))
        self.blockSignals(False)
        self.setCurrentIndex(self.count() - 2)


def ask_custom_source(combo, parent):
    """Handle the 'Other…' entry of a SourceCombo; returns True when a source
    was chosen."""
    if combo.currentData() is not None:
        return True
    prev = getattr(combo, '_last', None)
    dlg = SourceDialog(parent, prev)
    if dlg.exec_() == QDialog.Accepted and dlg.result:
        combo.add_custom(dlg.result)
        return True
    combo.blockSignals(True)
    combo.set_source(prev)
    combo.blockSignals(False)
    return False


# ---------------------------------------------------------------------------
# Recalculate tab
# ---------------------------------------------------------------------------

IN_COLS = ['Sample', 'Age', '± analytical', '± internal']
OUT_COLS = ['Sample', 'Age', '± analytical', '± internal', '± external', 'Change (%)']


class RecalcTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.results = []
        v = QVBoxLayout(self)

        top = QHBoxLayout()
        src_box = QGroupBox('Published on')
        sg = QGridLayout(src_box)
        self.src = SourceCombo()
        self.src.setToolTip('The monitor the authors used, the age they gave it and their '
                            'decay constant. "Other…" for a combination not in the list.')
        self.src_info = _note('')
        self.r_combo = QComboBox()
        self.r_label = QLabel('R to FCs')
        sg.addWidget(QLabel('Monitor and constants'), 0, 0)
        sg.addWidget(self.src, 0, 1)
        sg.addWidget(self.r_label, 1, 0)
        sg.addWidget(self.r_combo, 1, 1)
        sg.addWidget(self.src_info, 2, 0, 1, 2)
        top.addWidget(src_box, 1)
        self.tgt = CalibrationBox('Recalculate to')
        top.addWidget(self.tgt, 1)
        v.addLayout(top)

        opts = QHBoxLayout()
        self.unit = QComboBox()
        self.unit.addItems(['Ma', 'ka'])
        self.in_sig = QComboBox()
        self.in_sig.addItems(['1σ', '2σ'])
        self.in_sig.setCurrentIndex(1)
        self.out_sig = QComboBox()
        self.out_sig.addItems(['1σ', '2σ'])
        self.out_sig.setCurrentIndex(1)
        self.dec = QSpinBox()
        self.dec.setRange(0, 8)
        self.dec.setValue(4)
        for lab, w in [('Ages in', self.unit), ('Pasted ± are', self.in_sig),
                       ('Show ± as', self.out_sig), ('Decimals', self.dec)]:
            opts.addWidget(QLabel(lab))
            opts.addWidget(w)
            opts.addSpacing(10)
        opts.addStretch()
        for txt, fn in [('Paste', lambda: self.inp.paste()), ('Clear', self.clear),
                        ('Export…', self.export)]:
            b = QPushButton(txt)
            b.clicked.connect(fn)
            opts.addWidget(b)
        v.addLayout(opts)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel('Published ages'))
        self.inp = PasteTable(IN_COLS, 20)
        ll.addWidget(self.inp)
        ll.addWidget(_note('Paste from Excel with Ctrl+V (Cmd+V). ± internal includes the J '
                           'uncertainty; leave it empty if only the analytical error is known.'))
        split.addWidget(left)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel('Recalculated ages'))
        self.out = QTableWidget(0, len(OUT_COLS))
        self.out.setHorizontalHeaderLabels(OUT_COLS)
        self.out.verticalHeader().setVisible(False)
        self.out.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.out.horizontalHeader().setStretchLastSection(True)
        rl.addWidget(self.out)
        self.status = _note('')
        rl.addWidget(self.status)
        split.addWidget(right)
        split.setSizes([480, 640])
        v.addWidget(split, 1)

        self.src.currentIndexChanged.connect(self._src_changed)
        self.r_combo.currentIndexChanged.connect(self.recalc)
        self.tgt.cal.currentIndexChanged.connect(self.recalc)
        self.tgt.full.toggled.connect(self.recalc)
        for w in (self.unit, self.in_sig, self.out_sig):
            w.currentIndexChanged.connect(self.recalc)
        self.dec.valueChanged.connect(self.recalc)
        self.inp.itemChanged.connect(self.recalc)
        self._src_changed()

    # --- source -------------------------------------------------------------
    def _src_changed(self, *_):
        if not ask_custom_source(self.src, self):
            return
        src = self.src.current()
        self.src._last = src
        self.r_combo.blockSignals(True)
        self.r_combo.clear()
        for label, R, s, ref in r_choices(src['std']):
            self.r_combo.addItem(label, (R, s, ref))
        self.r_combo.blockSignals(False)
        other = src['std'] != 'FCs'
        self.r_combo.setVisible(other)
        self.r_label.setVisible(other)
        std = am.library_standard(src['std'])
        txt = source_text(src)
        if std:
            txt = f"{std['name']}. " + txt
        if src.get('cal'):
            self.tgt.set_name(src['cal'])
        self.src_info.setText(txt + '.')
        self.recalc()

    def R(self):
        src = self.src.current()
        if src is None or src['std'] == 'FCs':
            return 1.0, 0.0, ''
        return self.r_combo.currentData() or (1.0, 0.0, '')

    # --- data ---------------------------------------------------------------
    def rows(self):
        """[(label, age_Ma, ana_1s_Ma, int_1s_Ma or None)] of the filled rows."""
        k = 1e-3 if self.unit.currentText() == 'ka' else 1.0
        div = 2.0 if self.in_sig.currentText() == '2σ' else 1.0
        out = []
        for r in range(self.inp.rowCount()):
            age = _num(self.inp.text(r, 1))
            if age is None:
                continue
            a = _num(self.inp.text(r, 2))
            i = _num(self.inp.text(r, 3))
            out.append((self.inp.text(r, 0) or f'Row {r + 1}', age * k,
                        None if a is None else abs(a) * k / div,
                        None if i is None else abs(i) * k / div))
        return out

    def recalc(self, *_):
        src = self.src.current()
        if src is None:
            return
        cfg = self.tgt.cfg()
        R, R_1s, ref = self.R()
        label = f"R({src['std']}/FCs) {stdlib.short_ref(ref)}" if src['std'] != 'FCs' else ''
        k = 1e-3 if self.unit.currentText() == 'ka' else 1.0
        mult = 2.0 if self.out_sig.currentText() == '2σ' else 1.0
        dec = self.dec.value()
        self.results = []
        errs = 0
        for lab, age, a, i in self.rows():
            try:
                res = recalculate(age, src['std_age'], src['lam'], cfg, R, R_1s, label)
            except (ValueError, OverflowError):
                errs += 1
                continue
            f = abs(res['factor'])
            sa = None if a is None else f * a
            si = None if i is None else f * i
            sys_ = math.sqrt(sum(c * c for _, c in res['components']))
            base = si if si is not None else sa
            se = None if base is None else math.hypot(base, sys_)
            self.results.append(dict(label=lab, age_in=age, age=res['age'], ana=sa, int=si,
                                     ext=se, sys=sys_, change=(res['age'] / age - 1) * 100,
                                     components=res['components']))
        self.out.setRowCount(len(self.results))
        for r, d in enumerate(self.results):
            vals = [d['label'], _fmt(d['age'] / k, dec),
                    _fmt(None if d['ana'] is None else d['ana'] * mult / k, dec),
                    _fmt(None if d['int'] is None else d['int'] * mult / k, dec),
                    _fmt(None if d['ext'] is None else d['ext'] * mult / k, dec),
                    f"{d['change']:+.3f}"]
            for c, t in enumerate(vals):
                it = QTableWidgetItem(t)
                if c:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.out.setItem(r, c, it)
        self.out.resizeColumnsToContents()
        notes = []
        if self.results and all(d['int'] is None for d in self.results):
            notes.append('No ± internal was given, so the external error contains no J '
                         'uncertainty.')
        if errs:
            notes.append(f'{errs} row(s) could not be recalculated.')
        w = calibration_warning(cfg)
        if w:
            notes.append(w)
        if src['std'] == 'FCs' and src.get('cal') == self.tgt.name() and not self.tgt.full.isChecked():
            notes.append('Same calibration: the ages are unchanged and the external error is the '
                         'one this calibration implies.')
        self.status.setText(' '.join(notes))

    def clear(self):
        self.inp.clearContents()
        self.recalc()

    # --- export / session ---------------------------------------------------
    def export_rows(self):
        src = self.src.current()
        cfg = self.tgt.cfg()
        R, R_1s, ref = self.R()
        k_unit = self.unit.currentText()
        k = 1e-3 if k_unit == 'ka' else 1.0
        mult = 2.0 if self.out_sig.currentText() == '2σ' else 1.0
        s_lab = self.out_sig.currentText()
        consts = [['CONSTANTS'],
                  ['Published on', source_text(src)]]
        if src['std'] != 'FCs':
            consts.append([f"R({src['std']}/FCs)", f'{R:.8g}', f'± {R_1s:.3g} (1σ)', ref])
        consts += [['Recalculated to', self.tgt.name()],
                   ['FCs age (Ma)', f"{cfg['stdAgeMa']:g}", f"± {cfg['stdAgeErrMa']:g} (1σ)"],
                   ['Decay constant (yr⁻¹)', f"{cfg['lambdaYr']:.6g}", f"± {cfg['lambdaErrYr']:.3g} (1σ)"],
                   ['External error', external_route(cfg)], []]
        head = ['Sample', f'Age published ({k_unit})', f'Age ({k_unit})',
                f'±{s_lab} analytical', f'±{s_lab} internal', f'±{s_lab} external', 'Change (%)']
        rows = [['RECALCULATED AGES'], head]
        dec = self.dec.value()
        for d in self.results:
            rows.append([d['label'], _fmt(d['age_in'] / k, dec), _fmt(d['age'] / k, dec),
                         _fmt(None if d['ana'] is None else d['ana'] * mult / k, dec),
                         _fmt(None if d['int'] is None else d['int'] * mult / k, dec),
                         _fmt(None if d['ext'] is None else d['ext'] * mult / k, dec),
                         f"{d['change']:.4f}"])
        return consts, rows

    def export(self):
        if not self.results:
            QMessageBox.information(self, 'Export', 'There is nothing to export yet.')
            return
        self.win.export_tables(*self.export_rows(), 'recalculated_ages')

    def state(self):
        return dict(source=self.src.current(), R=self.r_combo.currentIndex(),
                    target=self.tgt.name(), full=self.tgt.full.isChecked(),
                    unit=self.unit.currentText(), in_sig=self.in_sig.currentText(),
                    out_sig=self.out_sig.currentText(), dec=self.dec.value(),
                    table=[[self.inp.text(r, c) for c in range(len(IN_COLS))]
                           for r in range(self.inp.rowCount())])

    def set_state(self, st):
        self.blockSignals(True)
        self.src.set_source(st.get('source'))
        self.src._last = self.src.current()
        self.tgt.set_name(st.get('target', DEFAULT_FC_NAME))
        self.tgt.full.setChecked(bool(st.get('full')))
        for w, key in [(self.unit, 'unit'), (self.in_sig, 'in_sig'), (self.out_sig, 'out_sig')]:
            i = w.findText(st.get(key, ''))
            if i >= 0:
                w.setCurrentIndex(i)
        self.dec.setValue(int(st.get('dec', 4)))
        tab = st.get('table') or []
        self.inp.blockSignals(True)
        self.inp.clearContents()
        self.inp.setRowCount(max(20, len(tab)))
        for r, row in enumerate(tab):
            for c, t in enumerate(row[:len(IN_COLS)]):
                self.inp.setItem(r, c, QTableWidgetItem(t))
        self.inp.blockSignals(False)
        self.blockSignals(False)
        self._src_changed()
        if 0 <= int(st.get('R', 0)) < self.r_combo.count():
            self.r_combo.setCurrentIndex(int(st.get('R', 0)))
        self.recalc()


# ---------------------------------------------------------------------------
# Combine tab
# ---------------------------------------------------------------------------

C_IN = ['Sample', 'Age', '± analytical', '± internal', 'Published on', 'Position']
C_SRC, C_POS = 4, 5


class CombineTable(PasteTable):
    def __init__(self, tab, rows=10):
        super().__init__(C_IN, rows)
        self.tab = tab
        self.text_cols = C_SRC          # pasted text fills Sample to ± internal
        for r in range(rows):
            self.row_added(r)
        self.setColumnWidth(C_SRC, 260)

    def row_added(self, r):
        cb = SourceCombo()
        cb._last = cb.current()
        cb.currentIndexChanged.connect(lambda _i, c=cb: self._src_changed(c))
        self.setCellWidget(r, C_SRC, cb)

    def _src_changed(self, cb):
        if ask_custom_source(cb, self):
            cb._last = cb.current()
        self.tab.sources_changed()

    def source(self, r):
        w = self.cellWidget(r, C_SRC)
        return w.current() if w else None


class _RecalcFluxHost(FluxHost):
    """Adapter for the shared flux dialog: one item per filled row."""
    title = 'J correlation'
    item_noun = 'Sample'
    manual = True
    caption = ('Assign each age to the hole it was irradiated in to take the J correlation '
               'from the .jflux and, where ± internal is empty, the J error of that hole. '
               'Without a .jflux, type the J or the R standard of the position and its 1σ '
               'under Manual: where ± internal is empty, its relative error gives the J term. '
               'A hole wins over a manual value. Ages without a hole are correlated when their '
               'Position labels are the same, or, without a label, when their manual J or R '
               'and its error are the same.')

    def __init__(self, tab):
        self.tab = tab

    def disk_flux(self):
        return self.tab.disk_flux

    def load(self, dialog):
        paths, _ = dfold.get_open_file_names(dialog, 'Import .jflux', '',
                                             'Flux config (*.jflux);;All (*)',
                                             tool='flux_regression')
        for p in paths or []:
            if not p:
                continue
            try:
                self.tab.disk_flux.update(read_jflux(p))
            except Exception as e:
                QMessageBox.warning(dialog, 'J correlation', f'{os.path.basename(p)}: {e}')

    def items(self):
        return self.tab.filled_rows()

    def item_label(self, r):
        return self.tab.table.text(r, 0) or f'Row {r + 1}'

    def item_disk_hole(self, r):
        return self.tab.disk_hole(r)

    def get_override(self, row, r):
        return self.tab.overrides.get(r, 'auto')

    def set_override(self, row, r, val):
        if val == 'auto':
            self.tab.overrides.pop(r, None)
        else:
            self.tab.overrides[r] = val

    def match(self, row, r):
        return self.tab.match(r)

    def get_manual(self, row, r):
        return self.tab.manual.get(r)

    def set_manual(self, row, r, val):
        if val is None:
            self.tab.manual.pop(r, None)
        else:
            self.tab.manual[r] = tuple(val)

    def hole_R(self, key, idx):
        fx = self.tab.disk_flux.get(key)
        return None if fx is None else flux_hole_R(fx, idx)

    def apply(self):
        self.tab.recalc()


class CombineTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.disk_flux = {}
        self.overrides = {}
        self.manual = {}            # row -> ('J' or 'R', value, 1σ) typed in the J dialog
        self.items = []
        self.result = None
        v = QVBoxLayout(self)

        top = QHBoxLayout()
        self.tgt = CalibrationBox('Combine on')
        top.addWidget(self.tgt, 1)
        mbox = QGroupBox('Combining')
        mg = QGridLayout(mbox)
        self.method = QComboBox()
        self.method.addItems(['WM + √MSWD', 'GLS + ML'])
        self.method.setToolTip('WM + √MSWD: weighted by the analytical errors, the error scaled '
                               'by √MSWD above the Wendt and Carl (1991) limit.\nGLS + ML: '
                               'generalised least squares on the full covariance, with '
                               'overdispersion fitted by maximum likelihood (Vermeesch 2015).')
        self.r_form = QFormLayout()
        mg.addWidget(QLabel('Method'), 0, 0)
        mg.addWidget(self.method, 0, 1)
        mg.addLayout(self.r_form, 1, 0, 1, 2)
        b_j = QPushButton('J correlation…')
        b_j.clicked.connect(self.open_flux)
        mg.addWidget(b_j, 2, 1)
        top.addWidget(mbox, 1)
        v.addLayout(top)

        opts = QHBoxLayout()
        self.unit = QComboBox()
        self.unit.addItems(['Ma', 'ka'])
        self.in_sig = QComboBox()
        self.in_sig.addItems(['1σ', '2σ'])
        self.in_sig.setCurrentIndex(1)
        self.out_sig = QComboBox()
        self.out_sig.addItems(['1σ', '2σ'])
        self.out_sig.setCurrentIndex(1)
        self.dec = QSpinBox()
        self.dec.setRange(0, 8)
        self.dec.setValue(4)
        for lab, w in [('Ages in', self.unit), ('Pasted ± are', self.in_sig),
                       ('Show ± as', self.out_sig), ('Decimals', self.dec)]:
            opts.addWidget(QLabel(lab))
            opts.addWidget(w)
            opts.addSpacing(10)
        opts.addStretch()
        for txt, fn in [('Paste', lambda: self.table.paste()), ('Add row', self.add_row),
                        ('Remove row', self.remove_rows), ('Clear', self.clear),
                        ('Export…', self.export)]:
            b = QPushButton(txt)
            b.clicked.connect(fn)
            opts.addWidget(b)
        v.addLayout(opts)

        split = QSplitter(Qt.Vertical)
        self.table = CombineTable(self, 10)
        split.addWidget(self.table)
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        self.out = QTableWidget(0, 5)
        self.out.setHorizontalHeaderLabels(['Sample', 'Age', '± analytical', '± internal', '± external'])
        self.out.verticalHeader().setVisible(False)
        self.out.setEditTriggers(QAbstractItemView.NoEditTriggers)
        bl.addWidget(self.out, 1)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        bl.addWidget(self.report, 1)
        split.addWidget(bottom)
        split.setSizes([330, 260])
        v.addWidget(split, 1)
        v.addWidget(_note('Each row is recalculated from the calibration it was published on to the '
                          'one above, then the ages are combined. Ages with the same Position label '
                          '(e.g. disk/hole) share their J error; the calibration terms are shared '
                          'by all ages, and the R of a monitor by every age on that monitor.'))

        self.method.currentIndexChanged.connect(self.recalc)
        self.tgt.cal.currentIndexChanged.connect(self.recalc)
        self.tgt.full.toggled.connect(self.recalc)
        for w in (self.unit, self.in_sig, self.out_sig):
            w.currentIndexChanged.connect(self.recalc)
        self.dec.valueChanged.connect(self.recalc)
        self.table.itemChanged.connect(self.recalc)
        self._r_combos = {}
        self.sources_changed()

    # --- rows ---------------------------------------------------------------
    def add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.row_added(r)

    def remove_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self.overrides = {}
        self.manual = {}
        self.sources_changed()

    def clear(self):
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            for c in range(C_SRC):
                self.table.setItem(r, c, QTableWidgetItem(''))
            self.table.setItem(r, C_POS, QTableWidgetItem(''))
        self.table.blockSignals(False)
        self.overrides = {}
        self.manual = {}
        self.recalc()

    def filled_rows(self):
        return [r for r in range(self.table.rowCount()) if _num(self.table.text(r, 1)) is not None]

    def disk_hole(self, r):
        pos = self.table.text(r, C_POS)
        if '/' in pos:
            d, h = pos.split('/', 1)
            return d.strip() or None, h.strip() or None
        return None, (pos or None)

    def match(self, r):
        ov = self.overrides.get(r)
        if ov == 'none':
            return None
        if isinstance(ov, tuple):
            key = (str(ov[0]), str(ov[1]))
            fx = self.disk_flux.get(key)
            if fx is not None and str(ov[2]) in fx['idx']:
                return key, fx['idx'][str(ov[2])]
            return None
        if not self.disk_flux:
            return None
        d, h = self.disk_hole(r)
        return match_flux_hole(self.disk_flux, d, h, None, None)

    def open_flux(self):
        if not self.filled_rows():
            QMessageBox.information(self, 'J correlation', 'Add some ages first.')
            return
        FluxAssignDialog(self, _RecalcFluxHost(self)).exec_()
        self.recalc()

    # --- monitors in use: one R choice each -------------------------------
    def sources_changed(self):
        stds = []
        for r in range(self.table.rowCount()):
            s = self.table.source(r)
            if s and s['std'] != 'FCs' and s['std'] not in stds:
                stds.append(s['std'])
        old = {k: cb.currentIndex() for k, cb in self._r_combos.items()}
        while self.r_form.rowCount():
            self.r_form.removeRow(0)
        self._r_combos = {}
        for sid in stds:
            cb = QComboBox()
            for label, R, s, ref in r_choices(sid):
                cb.addItem(label, (R, s, ref))
            if sid in old and old[sid] < cb.count():
                cb.setCurrentIndex(old[sid])
            cb.currentIndexChanged.connect(self.recalc)
            self.r_form.addRow(f'R({sid}/FCs)', cb)
            self._r_combos[sid] = cb
        self.recalc()

    def R(self, sid):
        if sid == 'FCs':
            return 1.0, 0.0, ''
        cb = self._r_combos.get(sid)
        if cb is None:
            ch = r_choices(sid)
            return ch[0][1], ch[0][2], ch[0][3]
        return cb.currentData()

    # --- calculation --------------------------------------------------------
    def recalc(self, *_):
        cfg = self.tgt.cfg()
        k = 1e-3 if self.unit.currentText() == 'ka' else 1.0
        div = 2.0 if self.in_sig.currentText() == '2σ' else 1.0
        mult = 2.0 if self.out_sig.currentText() == '2σ' else 1.0
        dec = self.dec.value()
        self.items = []
        skipped = []
        for r in self.filled_rows():
            src = self.table.source(r)
            age = _num(self.table.text(r, 1))
            a = _num(self.table.text(r, 2))
            i = _num(self.table.text(r, 3))
            lab = self.table.text(r, 0) or f'Row {r + 1}'
            if src is None or (a is None or a <= 0) and i is None:
                skipped.append(lab)
                continue
            R, R_1s, ref = self.R(src['std'])
            label = f"R({src['std']}/FCs) {stdlib.short_ref(ref)}"
            try:
                res = recalculate(age * k, src['std_age'], src['lam'], cfg, R, R_1s, label)
            except (ValueError, OverflowError):
                skipped.append(lab)
                continue
            f = abs(res['factor'])
            # J error of the assigned .jflux hole, else of the J or R typed
            # for the row
            m = self.match(r)
            rel = flux_hole_rel(self.disk_flux[m[0]], m[1]) if m else None
            man = self.manual.get(r)
            if not rel and man and man[1] > 0:
                rel = man[2] / man[1]
            p_j = dt_dlnR(res['age'], cfg['lambdaYr']) * rel if rel else None
            j_from_flux = False
            ana_from_int = False
            if a is None or a <= 0:
                # only ± internal given: analytical = internal without the J term
                si = f * i * k / div
                if p_j is None or p_j >= si:
                    skipped.append(lab)
                    continue
                sa = math.sqrt(si * si - p_j * p_j)
                ana_from_int = 'jflux' if m else 'manual'
            else:
                sa = f * a * k / div
                if i is not None:
                    si = max(f * i * k / div, sa)
                elif p_j is not None:
                    si = math.hypot(sa, p_j)
                    j_from_flux = 'jflux' if m else 'manual'
                else:
                    si = sa
            sys_ = math.sqrt(sum(c * c for _, c in res['components']))
            self.items.append(dict(row=r, label=lab, age=res['age'], ana=sa, int=si,
                                   ext=math.hypot(si, sys_), components=res['components'],
                                   no_int=(i is None and not j_from_flux),
                                   j_flux=j_from_flux, ana_from_int=ana_from_int, src=src))
        self.out.setRowCount(len(self.items))
        for rr, d in enumerate(self.items):
            for c, t in enumerate([d['label'], _fmt(d['age'] / k, dec), _fmt(d['ana'] * mult / k, dec),
                                   _fmt(d['int'] * mult / k, dec), _fmt(d['ext'] * mult / k, dec)]):
                it = QTableWidgetItem(t)
                if c:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.out.setItem(rr, c, it)
        self.out.resizeColumnsToContents()
        self.result = self._combine(cfg) if len(self.items) >= 2 else None
        self.report.setPlainText(self._report(cfg, skipped))

    def _combine(self, cfg):
        n = len(self.items)
        ages = np.array([d['age'] for d in self.items])
        a = np.array([d['ana'] for d in self.items])
        si = np.array([d['int'] for d in self.items])
        se = np.array([d['ext'] for d in self.items])
        labels = []
        for d in self.items:
            for lbl, _ in d['components']:
                if lbl not in labels:
                    labels.append(lbl)
        C = np.zeros((n, len(labels)))
        for i, d in enumerate(self.items):
            for lbl, val in d['components']:
                C[i, labels.index(lbl)] += val
        pts = []
        for d in self.items:
            pos = self.table.text(d['row'], C_POS)
            man = self.manual.get(d['row'])
            ident = (pos if pos else
                     ('manual', man[0], round(man[1], 12), round(man[2], 12)) if man else
                     ('row', d['row']))
            pts.append(dict(match=self.match(d['row']), ident=ident, irr=''))
        rho = j_correlation(pts, self.disk_flux)
        method = METHOD_GLS if self.method.currentIndex() == 1 else METHOD_WM
        res = combine(method, ages, a, si, se, standard_group=[0] * n,
                      sys_components=C, p_corr=rho)
        res['method'] = method
        res['rho'] = rho
        return res

    def _j_groups(self, rho):
        """One line per group of ages whose J errors are correlated."""
        n = len(rho)
        seen, out = set(), []
        for i in range(n):
            if i in seen:
                continue
            grp, todo = [], [i]
            while todo:
                k = todo.pop()
                if k in seen:
                    continue
                seen.add(k)
                grp.append(k)
                todo += [m for m in range(n) if m not in seen and rho[k][m] != 0]
            if len(grp) < 2:
                continue
            grp.sort()
            names = ', '.join(self.items[k]['label'] for k in grp)
            off = [rho[a][b] for a in grp for b in grp if a < b]
            if all(abs(r - 1) < 1e-9 for r in off):
                out.append(f'Same J for {names}.')
            else:
                nz = [r for r in off if r != 0]
                rng = (f'{min(nz):.2f}' if max(nz) - min(nz) < 0.005 else
                       f'{min(nz):.2f}–{max(nz):.2f}')
                out.append(f'Correlated J for {names} (ρ = {rng}).')
        return out or ['J errors independent.']

    def _report(self, cfg, skipped):
        k_unit = self.unit.currentText()
        k = 1e-3 if k_unit == 'ka' else 1.0
        mult = 2.0 if self.out_sig.currentText() == '2σ' else 1.0
        s_lab = self.out_sig.currentText()
        dec = self.dec.value()
        lines = []
        res = self.result
        if res is None:
            lines.append('Add at least two ages to combine.')
        else:
            lines += [f"Combined age: {res['wm'] / k:.{dec}f} {k_unit}",
                      f"± {res['sigma_ana'] * mult / k:.{dec}f} analytical, "
                      f"± {res['sigma_int'] * mult / k:.{dec}f} internal, "
                      f"± {res['sigma_ext'] * mult / k:.{dec}f} external ({s_lab})",
                      f"n = {res['n']}, MSWD = {res['mswd']:.2f}, p = {res['prob']:.3f}"]
            if res['method'] == METHOD_WM:
                lines.append('WM + √MSWD' + (f"; analytical error × √MSWD ({res['mswd_scale']:.2f})"
                                              if res.get('overdispersed') else ''))
            else:
                od = res.get('overdispersion', 0.0)
                lines.append('GLS + ML' + (f'; overdispersion {od * mult / k:.{dec}f} {k_unit} ({s_lab})'
                                           if od > 0 else '; no overdispersion'))
            lines += self._j_groups(res['rho'])
        lines.append('')
        lines.append(f'Calibration: {self.tgt.name()}. External: {external_route(cfg)}.')
        used = {}
        for d in self.items:
            used.setdefault(d['src']['std'], 0)
            used[d['src']['std']] += 1
        for sid in used:
            if sid != 'FCs':
                R, s, ref = self.R(sid)
                lines.append(f'R({sid}/FCs) = {stdlib.pm_text(R, s)} (1σ), {stdlib.short_ref(ref)}.')
        if any(d.get('j_flux') == 'jflux' for d in self.items):
            lines.append('Rows without ± internal but with a .jflux hole: internal = analytical '
                         'with the J error of that hole.')
        if any(d.get('j_flux') == 'manual' for d in self.items):
            lines.append('Rows without ± internal but with a manual J or R: internal = analytical '
                         'with its relative error.')
        if any(d.get('ana_from_int') for d in self.items):
            lines.append('Rows without ± analytical: analytical = internal without the J error '
                         'of the .jflux hole or the manual J or R.')
        if any(d['no_int'] for d in self.items):
            lines.append('Rows without ± internal and without a J error use the analytical '
                         'error there (no J term).')
        if skipped:
            lines.append('Not used (missing ± or source, or ± internal without a J error '
                         'larger than it): ' + ', '.join(skipped) + '.')
        w = calibration_warning(cfg)
        if w:
            lines.append(w)
        return '\n'.join(lines)

    # --- export / session ---------------------------------------------------
    def export(self):
        if not self.items:
            QMessageBox.information(self, 'Export', 'There is nothing to export yet.')
            return
        cfg = self.tgt.cfg()
        k_unit = self.unit.currentText()
        k = 1e-3 if k_unit == 'ka' else 1.0
        mult = 2.0 if self.out_sig.currentText() == '2σ' else 1.0
        s_lab = self.out_sig.currentText()
        dec = self.dec.value()
        consts = [['CONSTANTS'],
                  ['Combined on', self.tgt.name()],
                  ['FCs age (Ma)', f"{cfg['stdAgeMa']:g}", f"± {cfg['stdAgeErrMa']:g} (1σ)"],
                  ['Decay constant (yr⁻¹)', f"{cfg['lambdaYr']:.6g}", f"± {cfg['lambdaErrYr']:.3g} (1σ)"],
                  ['External error', external_route(cfg)]]
        for sid in sorted({d['src']['std'] for d in self.items} - {'FCs'}):
            R, s, ref = self.R(sid)
            consts.append([f'R({sid}/FCs)', f'{R:.8g}', f'± {s:.3g} (1σ)', ref])
        consts.append([])
        rows = [['RECALCULATED AGES'],
                ['Sample', 'Published on', 'Position', 'J or R (1σ)', f'Age ({k_unit})', f'±{s_lab} analytical',
                 f'±{s_lab} internal', f'±{s_lab} external']]
        for d in self.items:
            man = self.manual.get(d['row'])
            m = self.match(d['row'])
            jtxt = (f"{m[0][1]}/{self.disk_flux[m[0]]['holes'][m[1]]} (.jflux)" if m else
                    f"{man[0]} {man[1]:.8g} ± {man[2]:.3g}" if man else '')
            rows.append([d['label'], source_text(d['src']), self.table.text(d['row'], C_POS), jtxt,
                         _fmt(d['age'] / k, dec), _fmt(d['ana'] * mult / k, dec),
                         _fmt(d['int'] * mult / k, dec), _fmt(d['ext'] * mult / k, dec)])
        res = self.result
        if res is not None:
            rows += [[], ['COMBINED AGE'],
                     ['Method', 'n', f'Age ({k_unit})', f'±{s_lab} analytical', f'±{s_lab} internal',
                      f'±{s_lab} external', 'MSWD', 'p'],
                     ['WM + √MSWD' if res['method'] == METHOD_WM else 'GLS + ML', res['n'],
                      _fmt(res['wm'] / k, dec), _fmt(res['sigma_ana'] * mult / k, dec),
                      _fmt(res['sigma_int'] * mult / k, dec), _fmt(res['sigma_ext'] * mult / k, dec),
                      f"{res['mswd']:.3f}", f"{res['prob']:.4f}"]]
        self.win.export_tables(consts, rows, 'combined_ages')

    def state(self):
        rows = []
        for r in range(self.table.rowCount()):
            rows.append(dict(cells=[self.table.text(r, c) for c in range(C_SRC)],
                             source=self.table.source(r), pos=self.table.text(r, C_POS),
                             override=self.overrides.get(r),
                             manual=list(self.manual[r]) if r in self.manual else None))
        return dict(target=self.tgt.name(), full=self.tgt.full.isChecked(),
                    method=self.method.currentIndex(), unit=self.unit.currentText(),
                    in_sig=self.in_sig.currentText(), out_sig=self.out_sig.currentText(),
                    dec=self.dec.value(), rows=rows,
                    disk_flux=serialize_disk_flux(self.disk_flux),
                    R={sid: cb.currentIndex() for sid, cb in self._r_combos.items()})

    def set_state(self, st):
        self.table.blockSignals(True)
        rows = st.get('rows') or []
        self.table.setRowCount(0)
        self.overrides = {}
        self.manual = {}
        self.disk_flux.clear()
        self.disk_flux.update(deserialize_disk_flux(st.get('disk_flux') or {}))
        for r, row in enumerate(rows):
            self.table.insertRow(r)
            self.table.row_added(r)
            for c, t in enumerate((row.get('cells') or [])[:C_SRC]):
                self.table.setItem(r, c, QTableWidgetItem(t))
            self.table.setItem(r, C_POS, QTableWidgetItem(row.get('pos') or ''))
            cb = self.table.cellWidget(r, C_SRC)
            cb.blockSignals(True)
            cb.set_source(row.get('source'))
            cb._last = cb.current()
            cb.blockSignals(False)
            ov = row.get('override')
            if ov:
                self.overrides[r] = tuple(ov) if isinstance(ov, list) else ov
            if row.get('manual'):
                self.manual[r] = tuple(row['manual'])
        while self.table.rowCount() < 10:
            self.add_row()
        self.table.blockSignals(False)
        self.tgt.set_name(st.get('target', DEFAULT_FC_NAME))
        self.tgt.full.setChecked(bool(st.get('full')))
        self.method.setCurrentIndex(int(st.get('method', 0)))
        for w, key in [(self.unit, 'unit'), (self.in_sig, 'in_sig'), (self.out_sig, 'out_sig')]:
            i = w.findText(st.get(key, ''))
            if i >= 0:
                w.setCurrentIndex(i)
        self.dec.setValue(int(st.get('dec', 4)))
        self.sources_changed()
        for sid, i in (st.get('R') or {}).items():
            cb = self._r_combos.get(sid)
            if cb is not None and 0 <= i < cb.count():
                cb.setCurrentIndex(i)
        self.recalc()


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

class RecalculatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Recalculator and age combiner')
        self.resize(1180, 760)
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        for txt, fn in [('Open session…', self.load_session), ('Save session…', self.save_session),
                        ('Standards…', self.open_standards)]:
            b = QPushButton(txt)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch()
        v.addLayout(bar)
        self.tabs = QTabWidget()
        self.recalc_tab = RecalcTab(self)
        self.combine_tab = CombineTab(self)
        self.tabs.addTab(self.recalc_tab, 'Recalculate')
        self.tabs.addTab(self.combine_tab, 'Combine')
        v.addWidget(self.tabs, 1)

    def export_tables(self, consts, rows, name):
        path, flt = dfold.get_save_file_name(self, 'Export', dfold.safe_filename(name) + '.xlsx',
                                             export.save_filter())
        if not path:
            return
        path = export.ensure_ext(path, flt)
        if edited_standards_note():
            consts = consts + [['Standards', edited_standards_note()]]
        notes = [['ABOUT THIS FILE'],
                 ['Sheet "Results"', 'The ages recalculated to the calibration in the Constants sheet'
                                     + (', and their combined age' if name == 'combined_ages' else '') + '.'],
                 ['Sheet "Constants"', 'The source and target calibrations and the R values used.'],
                 [],
                 ['CONVENTIONS'],
                 ['Recalculation', 'Each published age is turned back into its ratio to the monitor '
                                   'with the monitor age and decay constant it was published with, '
                                   'and recalculated on the target calibration through the R of the '
                                   'monitor relative to Fish Canyon sanidine.'],
                 ['± analytical, ± internal', 'The published errors, scaled by dt′/dt. The internal '
                                             'error includes J; it is blank where none was given.'],
                 ['± external', 'The internal error (or the analytical one where no internal was '
                                'given) with the target calibration\'s systematic terms and, for '
                                'a monitor other than FCs, the uncertainty of its R.'],
                 ['Change (%)', 'Recalculated age relative to the published age.'],
                 []]
        try:
            if path.lower().endswith('.xlsx'):
                export.write_xlsx(path, rows, notes=notes, constants=[c for c in consts if c])
            else:
                import csv
                with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                    w = csv.writer(f)
                    for r in consts + rows:
                        w.writerow(r)
        except Exception as e:
            QMessageBox.warning(self, 'Export', f'Could not write the file:\n{e}')
            return
        QMessageBox.information(self, 'Export', f'Saved:\n{path}')

    def open_standards(self):
        from standards import StandardsEditor
        dlg = StandardsEditor(self)
        dlg.exec_()
        if dlg.saved:
            am.reload_standards()
            rs = self.recalc_tab.state()
            cs = self.combine_tab.state()
            self.recalc_tab.src.reload()
            for r in range(self.combine_tab.table.rowCount()):
                self.combine_tab.table.cellWidget(r, C_SRC).reload()
            for box in (self.recalc_tab.tgt, self.combine_tab.tgt):
                name = box.name()
                box.cal.blockSignals(True)
                box.cal.clear()
                box.cal.addItems([r[0] for r in FC_PRESETS])
                box.cal.blockSignals(False)
                box.set_name(name)
                box._info()
            self.recalc_tab.set_state(rs)
            self.combine_tab.set_state(cs)

    def save_session(self):
        path, _ = dfold.get_save_file_name(self, 'Save session', 'recalculator.arrc',
                                           'Recalculator session (*.arrc);;All (*)')
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += '.arrc'
        st = dict(format='SimplAr recalculator 1', recalculate=self.recalc_tab.state(),
                  combine=self.combine_tab.state())
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(st, f, indent=1, ensure_ascii=False)
        except OSError as e:
            QMessageBox.warning(self, 'Save session', str(e))

    def load_session(self):
        path, _ = dfold.get_open_file_name(self, 'Open session', '',
                                           'Recalculator session (*.arrc);;All (*)')
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                st = json.load(f)
            self.recalc_tab.set_state(st.get('recalculate') or {})
            self.combine_tab.set_state(st.get('combine') or {})
        except (OSError, ValueError, KeyError, TypeError) as e:
            QMessageBox.warning(self, 'Open session', f'Could not read the session:\n{e}')


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Recalculator and age combiner')
    dfold.use_tool('recalculator')
    force_light(app)
    win = RecalculatorWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
