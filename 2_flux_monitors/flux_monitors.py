#!/usr/bin/env python3
"""
Flux monitors: reduces the monitor analyses from each irradiation position to
one weighted-mean R (and J), with outlier rejection and REML overdispersion.
Also works as a plain weighted-mean calculator for any value.

Layout of this file, top to bottom:
  reading .arar / .ararpt / .ararstack session files
  option values and the widgets bound to them
  DirectInputPanel (the Direct Input tab)
  FluxMonitorsWindow: file import, manual exclusion tab, the analysis
    pipeline, crystals and step groups, NMAD outlier rejection, statistics
    (random effects, p-values), plots, session files, export
"""

import json
import math
import os
import re
import sys
import textwrap
from collections import OrderedDict, Counter

import pandas as pd
from scipy.stats import chi2 as chi2_dist
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QRadioButton, QButtonGroup, QComboBox, QDoubleSpinBox, QPlainTextEdit,
    QTabWidget, QSplitter, QScrollArea, QTreeWidget, QTreeWidgetItem,
    QMessageBox, QInputDialog, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

# shared math/constants live one folder up, next to the launcher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arar_math import (
    FC_PRESETS, FC_PAIRED_LAMBDA, LAMBDA_PRESETS, fc_ingrowth, file_err_div,
    mswd_critical, DEFAULT_FC_NAME, STD_AGE_DEFAULT, STD_AGE_ERR,
    LAMBDA_DEFAULT, LAMBDA_ERR, force_light, no_wheel_on_dropdowns, mono_font, monitor_keys,
    calibration_notes, edited_standards_note, ml_overdispersion,
)
from standards import MonitorCombo, monitor_text, route_line
from calibration_dialog import CalibrationDialog
from arar_math import (route_cfg, route_from_cfg, route_text, monitor_ingrowth, STANDARD_IDS,
                       note_label, window_shortcuts, release_figures, report_errors)
import data_folders as dfold
import export


def _finite(x):
    """True for a real, finite number (None, NaN, inf and text are False)."""
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _uniquify_ids(rows):
    """Make every row's sample_id unique within its category (a Sample column
    repeated on every step would otherwise tie all those steps to one id)."""
    used = set()
    for r in rows:
        base = str(r.get('sample_id') or 'row')
        sid, k = base, 1
        while sid in used:
            k += 1
            sid = f"{base}#{k}"
        r['sample_id'] = sid
        used.add(sid)
    return rows


# ── Ar-Ar session files (.arar / .ararpt / .ararstack) ─────────────────────
# The same files the Plateau & isochron and Age distribution tools read.  Each step gives
# one row: R = r4039 with its error brought to 1σ by arar_math.file_err_div.

ARAR_CRYSTAL_RULES = OrderedDict([
    ('auto', 'Automatic: by run-ID prefix (e.g. 11776-13A → 11776-13), otherwise '
             'a new crystal starts whenever the laser power does not increase'),
    ('file', 'Each file (or stack aliquot) is one crystal / aliquot'),
    ('step', 'Every step is a separate crystal (single-crystal fusions)'),
])

_RUN_SUFFIX = re.compile(r'^(.*\d)([A-Za-z]+)$')


def _arar_sessions(path):
    """[(title, cfg, steps, session_excluded, plateau_window)] for one file."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    stem = os.path.splitext(os.path.basename(path))[0]
    if data.get('format') == 'ararstack_v1':
        g = data.get('global_cfg') or {}
        out = []
        for i, sd in enumerate(data.get('sessions') or []):
            cfg = {**g, **(sd.get('cfg') or {})}
            pset = (sd.get('plateau') or {}).get('pset')
            out.append((str(sd.get('title') or f"{stem} #{i + 1}"), cfg,
                        sd.get('steps') or [], bool(sd.get('excluded', False)),
                        {int(k) for k in pset} if pset else None))
        return out
    if 'steps' not in data:
        raise ValueError('not an .arar/.ararpt session (no steps)')
    return [(str(data.get('title') or stem), data.get('cfg') or {},
             data.get('steps') or [], False, None)]


def _experiment_id(title, stem):
    """Short experiment name: '121561 · 26G08776 · 9D12-25' → '26G08776',
    '11776-01 (QL-OSU-56F)' → '11776-01'."""
    parts = [p.strip() for p in str(title).split('·') if p.strip()]
    if len(parts) >= 2:
        return parts[1]
    head = str(title).split(' (')[0].strip()
    return head or stem


def _run_number(step):
    """Trailing run number of a step's run ID (tie-break within a minute)."""
    m = re.search(r'(\d+)\D*$', str(step.get('runID') or step.get('run_id') or ''))
    return int(m.group(1)) if m else 0


def _assign_crystals(steps, exp_id, rule):
    """Crystal id for every step of one session, steps given in run order
    (see ARAR_CRYSTAL_RULES)."""
    n = len(steps)
    labels = [str(s.get('label') or i + 1) for i, s in enumerate(steps)]
    if rule == 'file':
        return [exp_id] * n
    if rule == 'step':
        return [f"{exp_id}-{labels[i]}" for i in range(n)]
    # auto, 1: run IDs that end in a step letter (Mass Spec / OSU: 11776-13DDE)
    rids = [str(s.get('run_id') or s.get('runID') or '').strip() for s in steps]
    m = [_RUN_SUFFIX.match(r) for r in rids]
    if n and all(m):
        return [x.group(1) for x in m]
    # auto, 2: laser-power sequence.  Step heating raises the power; a step at
    # the same or a lower power than the one before starts the next crystal.
    ids, start, prev = [], None, None
    for i, s in enumerate(steps):
        p = float(s['power']) if _finite(s.get('power')) else None
        if i == 0 or p is None or prev is None or p <= prev:
            start = labels[i]
        ids.append(f"{exp_id}-{start}")
        prev = p
    if len(set(ids)) == 1:
        ids = [exp_id] * n
    return ids


def read_arar_rows(path, crystal_rule='auto', group_by='position'):
    """Rows for this tool from one .arar/.ararpt/.ararstack.

    Returns (OrderedDict{category: [rows]}, info).  Category is the irradiation
    position (irrad_label, else hole, else disk) or the session title.  Steps
    excluded in the file (Analysis Included = No, an excluded stack aliquot, or
    outside a saved plateau window) come back with 'file_reason' set: they are
    loaded as manual exclusions, never dropped."""
    stem = os.path.splitext(os.path.basename(path))[0]
    cats = OrderedDict()
    info = dict(sessions=0, steps=0, skipped=0, calibrations=set())
    for title, cfg, steps, sess_excl, pset in _arar_sessions(path):
        info['sessions'] += 1
        ef = 1.0 / file_err_div(cfg)
        consts = cfg.get('constants') or {}
        exp_id = _experiment_id(title, stem)
        pos = str(consts.get('irrad_label') or cfg.get('hole_id')
                  or cfg.get('disk_id') or '').strip()
        cat = pos if (group_by == 'position' and pos) else title
        # Work in run order: an export can list a step out of sequence (e.g.
        # the 30 W fusion step of a step-heated crystal written first), and
        # the power-sequence rule needs the real order.
        order = list(range(len(steps)))
        if steps and all(s.get('run_dt') for s in steps):
            order.sort(key=lambda i: (str(steps[i]['run_dt']), _run_number(steps[i]), i))
        crystals = _assign_crystals([steps[i] for i in order], exp_id, crystal_rule)
        if _finite(cfg.get('stdAgeMa')) and _finite(cfg.get('lambdaYr')):
            info['calibrations'].add((float(cfg['stdAgeMa']), float(cfg['lambdaYr'])))
        rows = []
        for k, i in enumerate(order):
            s = steps[i]
            r, er = s.get('r4039'), s.get('err_r4039')
            if not (_finite(r) and _finite(er)):
                info['skipped'] += 1
                continue
            sid = (str(s.get('run_id') or s.get('runID') or '').strip()
                   or f"{exp_id}-{s.get('label', i + 1)}")
            if sess_excl:
                reason = 'aliquot excluded in stack'
            elif s.get('status') == 'excluded':
                reason = 'excluded in file'
            elif pset is not None and i not in pset:
                reason = 'outside saved plateau window'
            else:
                reason = None
            r, er1 = float(r), float(er) * ef
            rows.append({
                'sample_id':         sid,
                'value40_39':       r,
                'uncertainty40_39': er1,
                'value':            r,
                'uncertainty':      er1,
                'measured_j':        None,
                'measured_j_error':   None,
                'imported_j':        None,   # J comes from R only, never the file's J
                'imported_j_error':   None,
                'power':            float(s['power']) if _finite(s.get('power')) else None,
                'ar39':             float(s['pct39Ar']) if _finite(s.get('pct39Ar')) else None,
                'grain_id':         crystals[k],
                'label':            str(s.get('label', '')),
                'source':           os.path.basename(path),
                'file_reason':      reason,
            })
            info['steps'] += 1
        cats.setdefault(cat, []).extend(rows)
    return cats, info


# ── Option values and the widgets bound to them ─────────────────────────────
# Every option is a Value that the analysis code reads with .get() and the
# window shows in a widget; changing either updates the other, and every
# .set() calls the Value's callbacks (re-analysis, hints, ...).

class Value:
    def __init__(self, value=None):
        self._v = value
        self._callbacks = []

    def get(self):
        return self._v

    def set(self, v):
        self._v = v
        for cb in list(self._callbacks):
            try:
                cb()
            except RuntimeError as e:
                # the widget this callback showed the value in has been closed
                if 'has been deleted' not in str(e):
                    raise
                self._callbacks.remove(cb)

    def on_change(self, cb):
        """Call cb after every set()."""
        self._callbacks.append(cb)


def _quietly(widget, fn):
    """Run fn with the widget's signals blocked (code updates, not the user)."""
    widget.blockSignals(True)
    try:
        fn()
    finally:
        widget.blockSignals(False)


def bind_line_edit(value, le):
    le.setText('' if value.get() is None else str(value.get()))
    le.textChanged.connect(value.set)

    def show():
        t = '' if value.get() is None else str(value.get())
        if le.text() != t:
            _quietly(le, lambda: le.setText(t))
    value.on_change(show)
    return le


def bind_label(value, lbl):
    lbl.setText(str(value.get() or ''))
    value.on_change(lambda: lbl.setText(str(value.get() or '')))
    return lbl


def bind_check(value, cb, command=None):
    cb.setChecked(bool(value.get()))
    cb.toggled.connect(lambda on: value.set(bool(on)))
    if command is not None:
        cb.clicked.connect(lambda *_: command())

    def show():
        if cb.isChecked() != bool(value.get()):
            _quietly(cb, lambda: cb.setChecked(bool(value.get())))
    value.on_change(show)
    return cb


def bind_radios(value, pairs, command=None):
    """pairs: [(QRadioButton, key)]; the checked button's key is the value."""
    group = QButtonGroup(pairs[0][0])
    for rb, key in pairs:
        group.addButton(rb)
        rb.setChecked(value.get() == key)
        rb.toggled.connect(lambda on, k=key: on and value.set(k))
        if command is not None:
            rb.clicked.connect(lambda *_: command())

    def show():
        for rb, key in pairs:
            if key == value.get() and not rb.isChecked():
                _quietly(rb, lambda rb=rb: rb.setChecked(True))
    value.on_change(show)
    return group


def bind_combo(value, combo, on_select=None):
    """A read-only combo box whose text is the value; on_select runs when the
    user picks an entry (also the one already shown)."""
    if value.get():
        combo.setCurrentText(str(value.get()))
    combo.currentTextChanged.connect(value.set)

    def show():
        if combo.currentText() != (value.get() or ''):
            _quietly(combo, lambda: combo.setCurrentText(str(value.get() or '')))
    value.on_change(show)
    if on_select is not None:
        def picked(_i):
            if value.get() != combo.currentText():
                value.set(combo.currentText())
            on_select()
        combo.activated.connect(picked)
    return combo


def set_combo_items(combo, items, value):
    """Replace a combo box's entries without touching its value."""
    def fill():
        combo.clear()
        combo.addItems([str(i) for i in items])
        combo.setCurrentIndex(combo.findText(str(value.get())) if value.get() in items else -1)
    _quietly(combo, fill)


def bind_spin(value, spin):
    spin.setValue(float(value.get()))
    spin.valueChanged.connect(lambda v: value.set(float(v)))

    def show():
        if spin.value() != value.get():
            _quietly(spin, lambda: spin.setValue(float(value.get())))
    value.on_change(show)
    return spin


def _spin(lo, hi, step, decimals=2, width=70):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setFixedWidth(width)
    return s


def _note(text, color='gray', size=8, italic=False):
    """A small wrapped explanatory label."""
    return note_label(text, size + 3, color, italic)


def _row(*items, stretch=True):
    """A horizontal row of widgets (an int adds that much spacing)."""
    lay = QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    for it in items:
        if isinstance(it, int):
            lay.addSpacing(it)
        else:
            lay.addWidget(it)
    if stretch:
        lay.addStretch()
    return lay


def _ok_cancel(dialog, ok_text="OK", on_ok=None, extra=None):
    """Buttons row for a dialog: [extra ...] stretch [Cancel] [ok_text]."""
    lay = QHBoxLayout()
    for b in (extra or []):
        lay.addWidget(b)
    lay.addStretch()
    cancel = QPushButton("Cancel")
    cancel.clicked.connect(dialog.reject)
    ok = QPushButton(ok_text)
    ok.setDefault(True)
    ok.clicked.connect(on_ok or dialog.accept)
    lay.addWidget(cancel)
    lay.addWidget(ok)
    return lay


class DirectInputPanel(QWidget):
    """Direct data entry via paste-from-spreadsheet."""

    def __init__(self, calculator, parent=None):
        super().__init__(parent)
        self.calculator       = calculator
        self.groups           = []   # list of (name_value, text_edit, group_box)
        self.uncertainty_type = Value("2sigma")
        # Column config
        self.col_value_label  = Value("Value")
        self.col_unc_label    = Value("Uncertainty")
        self.col_id_var       = Value(False)
        self.col_id_label     = Value("ID")
        self.col_extra_var    = Value(False)
        self.col_extra_label  = Value("Power")
        # Values are ⁴⁰Ar*/³⁹Ar ratios → J is derived; otherwise any quantity.
        self.is_ratio_var     = Value(False)
        self._hint_labels     = []   # hint label widgets, refreshed on config change
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(30)
        self._refresh_timer.timeout.connect(self._refresh_counts)
        self.setup_ui()

    def setup_ui(self):
        outer = QVBoxLayout(self)

        # ── Column configuration ──────────────────────────────────────────────
        cfg = QGroupBox("Columns")
        cl = QVBoxLayout(cfg)
        cl.addLayout(_row(QLabel("Value label:"),
                          bind_line_edit(self.col_value_label, QLineEdit()), 20,
                          QLabel("Uncertainty label:"),
                          bind_line_edit(self.col_unc_label, QLineEdit())))
        cl.addLayout(_row(bind_check(self.col_id_var, QCheckBox("ID column:"),
                                     self._update_hints),
                          bind_line_edit(self.col_id_label, QLineEdit()), 20,
                          bind_check(self.col_extra_var, QCheckBox("Extra column:"),
                                     self._update_hints),
                          bind_line_edit(self.col_extra_label, QLineEdit())))
        cl.addWidget(bind_check(self.is_ratio_var, QCheckBox(
            "Values are ⁴⁰Ar*/³⁹Ar ratios (R): also show J from the calibration in the "
            "File import tab")))
        cl.addWidget(_note('Checked columns appear in that order before Value and Uncertainty. '
                           'ID = crystal; a blank ID continues the crystal above. '
                           'Extra = laser power (or temperature) of the step.'))
        cl.addWidget(_note('Outlier rejection, statistical model, crystal and calibration '
                           'options are on the File import tab and apply here too. Changing '
                           'one there re-runs this analysis at once (exclusions kept).',
                           color='#1f4e79'))
        outer.addWidget(cfg)

        # keep the column hints in step with the labels
        for v in (self.col_value_label, self.col_unc_label,
                  self.col_id_label, self.col_extra_label):
            v.on_change(lambda: QTimer.singleShot(20, self._update_hints))

        # ── Groups (scrollable) ───────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        holder = QWidget()
        self._groups_layout = QVBoxLayout(holder)
        self._groups_layout.addStretch()
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)

        # ── Bottom bar ────────────────────────────────────────────────────────
        add_btn = QPushButton("+ Add group")
        add_btn.clicked.connect(lambda: self.add_group())
        ex_btn = QPushButton("Load example")
        ex_btn.clicked.connect(self.load_example)
        rb2, rb1 = QRadioButton("±2σ"), QRadioButton("±1σ")
        bind_radios(self.uncertainty_type, [(rb2, "2sigma"), (rb1, "1sigma")],
                    command=self._update_hints)
        self.analyze_button = QPushButton("Analyse")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.analyze_direct_data)
        self.status_label = QLabel("Add a group and paste data")
        self.status_label.setStyleSheet("color: gray;")
        outer.addLayout(_row(add_btn, 12, ex_btn, 20, QLabel("Pasted errors are:"),
                             rb2, rb1, 16, self.analyze_button, 8, self.status_label))

        self.add_group("C1")

    def _get_col_hint(self):
        """Build the tab-separated column header shown above each text area."""
        cols = []
        if self.col_id_var.get():
            cols.append(self.col_id_label.get() or "ID")
        if self.col_extra_var.get():
            cols.append(self.col_extra_label.get() or "Extra")
        cols.append(self.col_value_label.get() or "Value")
        sig = "±2σ" if self.uncertainty_type.get() == "2sigma" else "±1σ"
        cols.append(f"{self.col_unc_label.get() or 'Uncertainty'} ({sig})")
        return "\t".join(cols)

    def _update_hints(self, *_):
        """Refresh all group column hint labels when the column config changes."""
        hint = self._get_col_hint()
        for lbl in self._hint_labels:
            lbl.setText(hint.replace("\t", "    "))
        self._refresh()

    def add_group(self, name=""):
        idx   = len(self.groups) + 1
        gname = name or f"C{idx}"

        box = QGroupBox()
        bl = QVBoxLayout(box)
        name_var = Value(gname)
        count_lbl = QLabel("")
        count_lbl.setStyleSheet("color: gray;")
        remove_btn = QPushButton("Remove")
        hdr = _row(QLabel("Group:"), bind_line_edit(name_var, QLineEdit()), 10, count_lbl)
        hdr.addWidget(remove_btn)
        bl.addLayout(hdr)

        hint_lbl = QLabel(self._get_col_hint().replace("\t", "    "))
        hint_lbl.setFont(mono_font(9))
        hint_lbl.setStyleSheet("color: #888888; background: #F5F5F5;")
        bl.addWidget(hint_lbl)
        self._hint_labels.append(hint_lbl)

        txt = QPlainTextEdit()
        txt.setFont(mono_font(10))
        txt.setLineWrapMode(QPlainTextEdit.NoWrap)
        txt.setFixedHeight(txt.fontMetrics().lineSpacing() * 7 + 12)
        txt._count_lbl = count_lbl
        txt.textChanged.connect(self._refresh_timer.start)
        bl.addWidget(txt)

        def remove(b=box, nv=name_var, hl=hint_lbl):
            self.groups = [(n, t, g) for n, t, g in self.groups if n is not nv]
            if hl in self._hint_labels:
                self._hint_labels.remove(hl)
            b.setParent(None)
            b.deleteLater()
            self._refresh()
        remove_btn.clicked.connect(lambda: remove())

        self._groups_layout.insertWidget(self._groups_layout.count() - 1, box)
        self.groups.append((name_var, txt, box))
        self._refresh()

    def _parse_line(self, line):
        """
        Parse one tab-, comma- or space-separated line using the active column
        config.  Column order: [ID]  [Extra]  Value  Uncertainty  ([] = optional).
        Cells are positional, so a blank ID cell (pasted from a spreadsheet)
        stays blank instead of shifting the value into the ID column.
        Returns (grain_id, extra_val, value, uncertainty) or None if invalid.
        """
        if not line.strip():
            return None
        if '\t' in line:
            parts = [p.strip() for p in line.split('\t')]
        elif ',' in line:
            parts = [p.strip() for p in line.split(',')]
        else:
            parts = line.split()
        while parts and parts[-1] == '':
            parts.pop()
        has_id    = self.col_id_var.get()
        has_extra = self.col_extra_var.get()
        min_cols  = 2 + (1 if has_id else 0) + (1 if has_extra else 0)
        if len(parts) < min_cols:
            return None
        idx       = 0
        grain_id  = None
        extra_val = None
        try:
            if has_id:
                grain_id = parts[idx] or None; idx += 1
            if has_extra:
                try:
                    extra_val = float(parts[idx])
                except ValueError:
                    extra_val = None
                idx += 1
            value       = float(parts[idx]); idx += 1
            uncertainty = float(parts[idx])
        except (ValueError, IndexError):
            return None
        return (grain_id, extra_val, value, uncertainty)

    def _count_valid(self, text_widget):
        count = 0
        for line in text_widget.toPlainText().splitlines():
            if self._parse_line(line) is not None:
                count += 1
        return count

    def _update_count_label(self, text_widget, label):
        n = self._count_valid(text_widget)
        label.setText(f"{n} point{'s' if n != 1 else ''}" if n else "")

    def _refresh_counts(self):
        for _, t, _ in self.groups:
            self._update_count_label(t, t._count_lbl)
        self._refresh()

    def _refresh(self):
        total = sum(self._count_valid(t) for _, t, _ in self.groups)
        if total >= 2:
            self.analyze_button.setEnabled(True)
            g = len([1 for _, t, _ in self.groups if self._count_valid(t) > 0])
            self.status_label.setText(f"{total} valid points in {g} group{'s' if g != 1 else ''}")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.analyze_button.setEnabled(False)
            self.status_label.setText("Need at least 2 valid points")
            self.status_label.setStyleSheet("color: gray;")

    def get_valid_data(self):
        """Build category_data dicts from the text areas (errors stored at 1σ)."""
        do_halve    = self.uncertainty_type.get() == "2sigma"
        category_data = {}

        for i, (name_var, text_widget, _) in enumerate(self.groups):
            category = name_var.get().strip() or f"Group{i+1}"
            rows = []
            pt_idx = 0
            current_grain = None
            for line in text_widget.toPlainText().splitlines():
                parsed = self._parse_line(line)
                if parsed is None:
                    continue
                grain_id, extra_val, value, uncertainty = parsed
                pt_idx += 1
                if do_halve:
                    uncertainty /= 2.0  # store as 1σ

                sid = f"{category}_pt{pt_idx}"
                if self.col_id_var.get():
                    if grain_id:
                        current_grain = grain_id
                    grain = current_grain or sid     # blank ID: same crystal as above
                else:
                    grain = sid                      # no ID column: every row its own
                rows.append({
                    'sample_id':         sid,
                    'value40_39':       value,
                    'uncertainty40_39': uncertainty,
                    'value':            value,
                    'uncertainty':      uncertainty,
                    'measured_j':        None,
                    'measured_j_error':   None,
                    'imported_j':        None,
                    'imported_j_error':   None,
                    'power':            extra_val,   # step label (laser power / T)
                    'grain_id':         grain,
                    'ar39':             None,
                    'label':            str(pt_idx),
                })
            if rows:
                category_data[category] = _uniquify_ids(rows)

        return category_data

    def _clear_groups(self):
        for _, _, box in self.groups:
            box.setParent(None)
            box.deleteLater()
        self._hint_labels.clear()
        self.groups = []

    _CONFIG = ('col_value_label', 'col_unc_label', 'col_id_var', 'col_id_label',
               'col_extra_var', 'col_extra_label', 'is_ratio_var', 'uncertainty_type')

    def state(self):
        """The pasted groups and the column settings, for a session file."""
        return dict(groups=[[n.get(), t.toPlainText()] for n, t, _ in self.groups],
                    **{k: getattr(self, k).get() for k in self._CONFIG})

    def set_state(self, st):
        for k in self._CONFIG:
            if k in st:
                getattr(self, k).set(st[k])
        self._clear_groups()
        for name, text in st.get('groups') or [['C1', '']]:
            self.add_group(name)
            self.groups[-1][1].setPlainText(text)
        self._update_hints()
        self._refresh_counts()

    def load_example(self):
        # Ar-Ar example: Grain ID + Power % + Value + Uncertainty
        # Sets column config to match before loading.
        self.col_id_var.set(True)
        self.col_id_label.set("Grain ID")
        self.col_extra_var.set(True)
        self.col_extra_label.set("Power")
        self.col_value_label.set("40/39 Ratio")
        self.col_unc_label.set("Uncertainty")
        self.is_ratio_var.set(True)

        # C1: two step-heated crystals (G1, G2: 2 → 5 → 30 W) and three
        # single-step fusions; C2: fusions only.  Blank ID = same crystal.
        example = {
            "C1": [
                ("G1", "2",  "5.6682", "0.0040"),
                ("",   "5",  "5.6695", "0.0022"),
                ("",   "30", "5.6671", "0.0031"),
                ("G2", "2",  "5.6710", "0.0028"),
                ("",   "5",  "5.6658", "0.0045"),
                ("",   "30", "5.6700", "0.0033"),
                ("G3", "30", "5.6690", "0.0029"),
                ("G4", "30", "5.6702", "0.0031"),
                ("G5", "30", "5.6679", "0.0027"),
            ],
            "C2": [
                ("G6", "30", "5.6720", "0.0038"),
                ("G7", "30", "5.6698", "0.0025"),
                ("G8", "30", "5.6735", "0.0042"),
                ("G9", "30", "5.6712", "0.0033"),
            ],
        }
        self._clear_groups()
        for gname, rows in example.items():
            self.add_group(gname)
            _, txt, _ = self.groups[-1]
            txt.setPlainText("\n".join(f"{g}\t{p}\t{v}\t{e}" for g, p, v, e in rows))

        self._update_hints()
        self._refresh_counts()

    def analyze_direct_data(self):
        category_data = self.get_valid_data()
        if not category_data:
            QMessageBox.critical(self, "Error", "No valid data to analyse")
            return

        valid = {cat: data for cat, data in category_data.items() if len(data) >= 2}
        if not valid:
            QMessageBox.critical(self, "Error", "Each group needs at least 2 valid points")
            return

        main_calc = self.calculator
        label = self.col_value_label.get().strip() or "Value"

        if not main_calc._options_valid(show=True):
            return
        try:
            # Same pipeline as file import: crystal basis, rejection, mean, J.
            # The same pasted data again (e.g. after changing an option) keeps
            # its manual exclusions; new or edited data starts clean.
            same = (main_calc.loaded_source == 'direct'
                    and bool(self.is_ratio_var.get()) == main_calc.value_is_ratio
                    and main_calc._data_signature(valid)
                    == main_calc._data_signature(main_calc.raw_category_data))
            if same:
                main_calc.value_label = "40/39 Ratio" if main_calc.value_is_ratio else label
                main_calc._run_analysis('=== DIRECT INPUT RE-ANALYSIS (exclusions kept) ===')
            else:
                main_calc._set_loaded_data(valid, source='direct',
                                           value_is_ratio=self.is_ratio_var.get(),
                                           value_label=label,
                                           input_name=('Direct input, errors pasted as ±2σ '
                                                       '(halved to 1σ)'
                                                       if self.uncertainty_type.get() == '2sigma'
                                                       else 'Direct input, errors pasted as ±1σ'))
                main_calc._run_analysis('=== DIRECT INPUT ANALYSIS ===')
            main_calc.log('\nDirect input analysis complete.')
            n_pts = sum(len(v) for v in valid.values())
            main_calc._set_current_data(f"Direct input: {n_pts} points in {len(valid)} groups")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analysing data: {str(e)}")


class FluxMonitorsWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flux monitors")
        self.resize(1600, 800)

        self.input_file  = None
        self.results     = None
        self.plot_data   = None
        self.loaded_df      = None
        self.loaded_col_map = {}
        self.import_sigma_var = Value('2sigma')   # errors in an imported table
        self.import_ratio_var = Value(True)       # imported value is R
        # What is loaded: 'table' (CSV/xlsx), 'arar' (session files), 'direct'
        self.loaded_source  = None
        self.file_source    = None   # what the File Import tab holds ('table'/'arar')
        self.session_source = None   # where the data of a loaded session came from
        self.arar_paths     = []
        self.arar_opts      = {}
        self.input_name     = None
        self.file_value_is_ratio = True
        self.file_value_label    = "40/39 Ratio"
        # one re-analysis shortly after a burst of option changes
        self._rerun_timer = QTimer(self)
        self._rerun_timer.setSingleShot(True)
        self._rerun_timer.setInterval(600)
        self._rerun_timer.timeout.connect(self._rerun_from_options)
        # True when the values are ⁴⁰Ar*/³⁹Ar ratios, so a J can be derived.
        # Otherwise the tool is a plain weighted-mean calculator for any value.
        self.value_is_ratio = True
        self.value_label    = "40/39 Ratio"
        # Rows actually analysed per category (steps, or one row per crystal),
        # and the crystal / step-group diagnostics of the last run.
        self.analysis_rows    = {}
        self.crystal_stats    = {}
        self.step_group_stats = {}
        # set while the export dialog draws the plot into its own figure
        self._export_fig  = None
        self._export_ax   = None
        self._export_opts = {}

        self.setup_gui()

    # ---------------------------------------------------------------------
    # File import: load, detect, select, build
    # ---------------------------------------------------------------------

    def repair_symbols(self, text):
        repairs = {'Â±': '±', 'â±': '±', '±': '±'}
        s = str(text)
        for bad, good in repairs.items():
            s = s.replace(bad, good)
        return ' '.join(s.split())

    def _load_file_df(self, path):
        """
        Load a CSV or Excel file to a DataFrame.
        For CSVs, tries multiple encodings and scans the first several
        rows to find the real header row (handles files with metadata
        rows before the column headers). Returns (df, error_string).
        """
        try:
            if path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
                df.columns = [self.repair_symbols(str(c)).strip() for c in df.columns]
                df = df.dropna(how='all')
                return df, None

            # Try encodings until one works
            raw_df, used_enc = None, 'utf-8-sig'
            for enc in ['utf-8-sig', 'utf-8', 'cp1252', 'iso-8859-1']:
                try:
                    try:
                        raw_df = pd.read_csv(path, encoding=enc, on_bad_lines='skip')
                    except TypeError:
                        raw_df = pd.read_csv(path, encoding=enc, error_bad_lines=False)
                    used_enc = enc
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception:
                    continue

            if raw_df is None:
                return None, "Could not decode file"

            # If very few columns came back, the real headers might be on a later row
            # (e.g. files with metadata preamble before the actual column names)
            best_df = raw_df
            if len(raw_df.columns) <= 3:
                for hrow in range(1, 10):
                    try:
                        try:
                            candidate = pd.read_csv(path, encoding=used_enc,
                                                    header=hrow, on_bad_lines='skip')
                        except TypeError:
                            candidate = pd.read_csv(path, encoding=used_enc,
                                                    header=hrow, error_bad_lines=False)
                        if len(candidate.columns) > len(best_df.columns):
                            best_df = candidate
                    except Exception:
                        continue

            best_df.columns = [self.repair_symbols(str(c)).strip() for c in best_df.columns]
            return best_df, None

        except Exception as e:
            return None, str(e)

    def _detect_columns(self, df):
        """
        Score every column in df against four roles.
        Returns dict: {'value': (best_col, score), 'error': ...,
                       'category': ..., 'sample_id': ...}
        """
        patterns = {
            'value': [
                (r'40\s*(?:ar)?\s*/\s*39',   95),
                (r'40.*mean.*r\b',            90),
                (r'40.*39',                   75),
                (r'\br\s*value\b',            80),
                (r'\bj\s*value\b',            75),
                (r'^r$',                      80),
                (r'^j$',                      70),
                (r'reml\s*mean|reml.*mean',   85),
                (r'ratio',                    60),
                (r'^value$',                  50),
                (r'mean\b',                   40),
            ],
            'error': [
                (r'reml\s*se|reml.*err',      90),
                (r'fixed.*se|fe.*se',         80),
                (r'2s.*(?:ratio|r\b)',        85),
                (r'(?:err|unc|sigma|se\b|std).*(?:ratio|r\b)', 85),
                (r'^\s*[±+]?\s*2s?\s*$',     75),
                (r'2\s*sigma',                75),
                (r'\berr(?:or)?\b',           55),
                (r'\bunc(?:ert)?\b',          55),
                (r'\bse\b|\bstd\b|\bsigma\b', 50),
            ],
            'category': [
                (r'categ',                    90),
                (r'\bgroup\b',                85),
                (r'\bdisk\b|\bdisc\b',        80),
                (r'\bposition\b|\bpos\b',     75),
                (r'\brun\b|\bbatch\b|\bset\b',65),
                (r'\btype\b|\bclass\b',       50),
            ],
            'sample_id': [
                (r'sample\s*[_\-]?\s*id',     90),
                (r'\bsample\b',               70),
                (r'\bid\b',                   60),
                (r'\bname\b',                 50),
                (r'\blabel\b',               45),
            ],
            'pct_gas': [
                (r'%\s*gas|gas\s*%',           95),
                (r'\bgas\b',                   70),
                (r'\bpower\b|\bpwr|laser',     65),
                (r'power\s*step|step',         60),
                (r'\btemp',                    50),
                (r'percent',                   40),
                (r'unnamed.*1\b|^unnamed: 1$', 35),
            ],
            # ³⁹Ar released in the step (any unit, or % of the total): lets a
            # step-heated crystal be pooled to its ³⁹Ar-weighted total gas.
            # Never a 40/39 column, an error column or a cumulative %.
            'ar39': [
                (r'^(?!.*(?:40|err|\bsd\b|sigma|σ|unc|cum|%\s*1)).*'
                 r'(?:%\s*39\s*ar|39\s*ar.*%|pct\s*39|%\s*ar\s*39)', 90),
                (r'^(?!.*(?:40|err|\bsd\b|sigma|σ|unc|cum|%\s*1)).*'
                 r'(?:\b39\s*ar|\bar\s*39|ar39)', 75),
            ],
            'grain_id': [
                (r'grain[_\-\s]?id',           95),
                (r'\bgrain\b',                 80),
                (r'\bcrystal\b',               75),
                (r'additional[_\-\s]?id',      70),
                (r'unnamed.*22|^unnamed: 22$', 35),
            ],
        }
        result = {}
        for field, pats in patterns.items():
            best_col, best_score = None, 0
            for col in df.columns:
                c = col.lower().strip()
                for pat, score in pats:
                    if re.search(pat, c) and score > best_score:
                        best_score = score
                        best_col   = col
                        break
            result[field] = (best_col, best_score)

        cols = list(df.columns)

        # ── Next-column inference ─────────────────────────────────────────────
        # When a value column is found, the immediately following column is
        # very likely its uncertainty (common in instrument output).
        def _next_col(detected_col):
            """Return the column immediately after detected_col, or None."""
            if detected_col and detected_col in cols:
                idx = cols.index(detected_col)
                if idx + 1 < len(cols):
                    return cols[idx + 1]
            return None

        # If error not found confidently, try the column after value
        if result.get('error', (None, 0))[1] < 60:
            candidate = _next_col(result.get('value', (None, 0))[0])
            if candidate and candidate != result.get('value', (None, 0))[0]:
                result['error'] = (candidate, 55)

        # Deduplication: each column may fill at most one role.
        # Process roles in descending score order so the best match wins.
        claimed = {}
        for role, (col, score) in sorted(result.items(), key=lambda x: -x[1][1]):
            if col is None:
                continue
            if col not in claimed:
                claimed[col] = role
            else:
                result[role] = (None, 0)  # yield to the higher-scoring role

        return result

    def _show_column_selector(self, df, detected):
        """
        Fallback dialog: three required dropdowns (value, error, category)
        and one optional (sample_id). Returns col_map dict or None if cancelled.
        """
        cols = [str(c) for c in df.columns]
        dlg = QDialog(self)
        dlg.setWindowTitle("Select columns")
        lay = QVBoxLayout(dlg)
        lay.addWidget(_note("Could not automatically identify all required columns.\n"
                            "Please select them from your file's headers below.",
                            color='black', size=9))
        fields = [
            ('value',     'Value column (R, J, ratio…)  *', True),
            ('error',     'Uncertainty / error column  *',   True),
            ('category',  'Group column (none = one group)', False),
            ('sample_id', 'Sample ID column',                False),
            ('grain_id',  'Grain / crystal ID column',       False),
            ('pct_gas',   'Step: laser power / temperature', False),
            ('ar39',      '³⁹Ar released in the step',       False),
        ]
        grid = QGridLayout()
        boxes = {}
        for i, (key, label, req) in enumerate(fields):
            choices = ([] if req else ['(none)']) + cols
            cur = detected.get(key, (None, 0))[0] or ('(none)' if not req else cols[0] if cols else '')
            cb = QComboBox()
            cb.addItems(choices)
            cb.setCurrentText(str(cur))
            grid.addWidget(QLabel(label), i, 0)
            grid.addWidget(cb, i, 1)
            boxes[key] = cb
        lay.addLayout(grid)
        result = [None]

        def ok():
            m = {k: b.currentText() for k, b in boxes.items()}
            missing = [label for key, label, req in fields
                       if req and (not m.get(key) or m[key] == '(none)')]
            if missing:
                QMessageBox.critical(dlg, "Missing", "Please select: " + ", ".join(missing))
                return
            result[0] = {k: (v if v != '(none)' else None) for k, v in m.items()}
            dlg.accept()

        lay.addLayout(_ok_cancel(dlg, "OK", ok))
        dlg.exec_()
        dlg.deleteLater()
        return result[0]

    def _build_category_data_from_df(self, df, col_map):
        """
        Convert a DataFrame + column mapping to the category_data dict.
        col_map: {'value': col, 'error': col, 'category': col_or_None,
                  'sample_id', 'grain_id', 'pct_gas', 'ar39': col_or_None}
        No category column puts every row in one group.  All errors are
        stored as 1σ internally.
        """
        val_col   = col_map['value']
        err_col   = col_map['error']
        cat_col   = col_map.get('category')
        sid_col   = col_map.get('sample_id')
        gid_col   = col_map.get('grain_id')
        gas_col   = col_map.get('pct_gas')
        ar39_col  = col_map.get('ar39')
        one_group = os.path.splitext(os.path.basename(self.input_file or 'All'))[0]

        # σ conversion: divide by 2 if input is 2σ
        sigma_mode = getattr(self, 'import_sigma_var', None)
        do_halve   = sigma_mode and sigma_mode.get() == '2sigma'

        category_data = {}
        current_grain = {}
        has_grain_col = bool(gid_col) or any(c in df.columns
                                             for c in ('Additional ID', 'grain_id'))

        def _num(row, col):
            try:
                v = float(row[col]) if col and pd.notna(row.get(col)) else None
            except (ValueError, TypeError, KeyError):
                v = None
            return v if _finite(v) else None

        for idx, row in df.iterrows():
            try:
                value = float(row[val_col])
                error = float(row[err_col])
            except (ValueError, TypeError, KeyError):
                continue
            if not (_finite(value) and _finite(error)):
                continue
            if cat_col:
                try:
                    category = str(row[cat_col]).strip()
                    if not category or category.lower() == 'nan':
                        continue
                except (KeyError, TypeError):
                    continue
            else:
                category = one_group

            if do_halve:
                error = error / 2.0

            if sid_col and pd.notna(row.get(sid_col, None)):
                sample_id = str(row[sid_col]).strip() or f"row_{idx}"
            else:
                sample_id = f"row_{idx}"

            if category not in category_data:
                category_data[category] = []
                current_grain[category] = None

            power = _num(row, gas_col)
            ar39  = _num(row, ar39_col)

            # Grain ID: from dedicated column, or propagate forward from previous row
            grain_id = None
            if gid_col:
                try:
                    v = row.get(gid_col)
                    if v is not None and pd.notna(v) and str(v).strip() not in ('', 'nan'):
                        grain_id = str(v).strip()
                except Exception:
                    pass
            if not grain_id:
                # Try legacy column names
                for gcol in ['Additional ID', 'grain_id']:
                    try:
                        v = row.get(gcol)
                        if v is not None and pd.notna(v) and str(v).strip() not in ('', 'nan'):
                            grain_id = str(v).strip()
                            break
                    except Exception:
                        pass
            if grain_id:
                current_grain[category] = grain_id
            elif has_grain_col and current_grain.get(category):
                grain_id = current_grain[category]  # blank cell: same crystal as above
            else:
                grain_id = sample_id                # no crystal info: row is its own

            category_data[category].append({
                'sample_id':          sample_id,
                'value40_39':        value,
                'uncertainty40_39':  error,
                'value':             value,
                'uncertainty':       error,
                'measured_j':         None,
                'measured_j_error':    None,
                'imported_j':         None,
                'imported_j_error':    None,
                'power':             power,
                'grain_id':          grain_id,
                'ar39':              ar39,
                'label':             str(idx),
            })

        for cat in category_data:
            _uniquify_ids(category_data[cat])
        return category_data

    def browse_raw_file(self):
        path, _ = dfold.get_open_file_name(
            self, "Select Data File", "",
            "All supported (*.csv *.xlsx);;CSV (*.csv);;Excel (*.xlsx);;"
            "All files (*)")
        if not path:
            return

        self.raw_file_var.set(path)
        name = os.path.basename(path)
        self.format_info_var.set(f"Reading {name}…")
        self._format_info_lbl.repaint()

        # ── All files: load to DataFrame, detect columns ─────────────────────
        df, err = self._load_file_df(path)
        if err:
            QMessageBox.critical(self, "Load error", f"Could not read file:\n{err}")
            self.format_info_var.set("No file loaded")
            return

        detected = self._detect_columns(df)
        required = ['value', 'error', 'category']
        missing  = [f for f in required if detected.get(f, (None, 0))[1] < 60]

        if missing:
            col_map = self._show_column_selector(df, detected)
            if col_map is None:
                self.format_info_var.set("Import cancelled")
                return
        else:
            col_map = {k: v[0] for k, v in detected.items()}

        self._store_and_confirm(df, col_map, path)

    def _store_and_confirm(self, df, col_map, path):
        """Store column mapping and show confirmation dialog with Change option."""
        name = os.path.basename(path)

        final_map = self._confirm_columns_dialog(df, col_map, name)
        if final_map is None:
            self.format_info_var.set("Import cancelled")
            return

        self.loaded_df      = df
        self.loaded_col_map = final_map
        self.input_file     = path
        self.input_name     = name
        self.loaded_source  = self.file_source = 'table'
        self.value_is_ratio = self.file_value_is_ratio = bool(self.import_ratio_var.get())
        self.value_label    = self.file_value_label = (
            (final_map.get('value') or 'Value') if not self.value_is_ratio else "40/39 Ratio")
        self._reset_overrides()        # exclusions belong to the previous file
        self.analyze_btn.setEnabled(True)

        val  = final_map.get('value')    or '?'
        err_c= final_map.get('error')    or '?'
        cat  = final_map.get('category') or '(one group)'
        self.format_info_var.set(f"Ready: {val} ± {err_c}  |  grouped by {cat}")
        self.log(f"File loaded: {name}  |  {val} ± {err_c}  by {cat}"
                 + ("" if self.value_is_ratio else "  |  plain values (no J)"))

    def _confirm_columns_dialog(self, df, col_map, filename):
        """
        Show detected columns, allow σ input selection and column override.
        Returns final col_map or None if cancelled.
        """
        result = [None]
        dlg = QDialog(self)
        dlg.setWindowTitle("Confirm column mapping")
        lay = QVBoxLayout(dlg)

        title = QLabel(filename)
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lay.addWidget(title)
        lay.addWidget(QLabel("<b>Columns detected:</b>"))
        grid = QGridLayout()
        for i, (key, label) in enumerate([('value', 'Value (R / J)'), ('error', 'Uncertainty'),
                                          ('category', 'Category'), ('sample_id', 'Sample ID'),
                                          ('grain_id', 'Grain / Crystal ID'),
                                          ('pct_gas', 'Step (power)'), ('ar39', '³⁹Ar')]):
            val = col_map.get(key) or '—'
            v = QLabel(str(val))
            v.setFont(mono_font(9))
            v.setStyleSheet("color: black;" if val != '—' else "color: gray;")
            grid.addWidget(QLabel(f"  {label}:"), i, 0)
            grid.addWidget(v, i, 1)
        lay.addLayout(grid)

        # ── Input σ selector ───────────────────────────────────────────
        rb2, rb1 = QRadioButton("±2σ"), QRadioButton("±1σ")
        bind_radios(self.import_sigma_var, [(rb2, '2sigma'), (rb1, '1sigma')])
        lay.addLayout(_row(QLabel("Errors in file:"), 10, rb2, rb1))

        # ── What the value is: a ⁴⁰Ar*/³⁹Ar ratio (J derived) or anything else
        vname = str(col_map.get('value') or '').lower()
        looks_ratio = bool(re.search(r'40.*39|ratio|^r$|\br\s*value', vname))
        self.import_ratio_var = Value(looks_ratio)
        lay.addWidget(bind_check(self.import_ratio_var, QCheckBox(
            "Value is a ⁴⁰Ar*/³⁹Ar ratio (R): derive J from the calibration")))
        lay.addWidget(_note("Untick for any other quantity (J values, ages, earlier means): "
                            "plain weighted mean, no J."))

        def accept():
            result[0] = col_map
            dlg.accept()

        def change():
            dlg.reject()
            new_map = self._show_column_selector(
                df, {k: (v, 100) for k, v in col_map.items() if v})
            if new_map:
                result[0] = new_map

        change_btn = QPushButton("Change columns…")
        change_btn.clicked.connect(change)
        lay.addLayout(_ok_cancel(dlg, "Accept", accept, extra=[change_btn]))
        dlg.exec_()
        dlg.deleteLater()
        return result[0]

    def setup_gui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Horizontal)
        main.addWidget(splitter)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Flux monitors")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        left_lay.addWidget(title)
        b_save = QPushButton("Save session…")
        b_save.setToolTip("The data, the options, the calibration and the manual "
                          "exclusions, in one file.")
        b_save.clicked.connect(self.save_session)
        b_load = QPushButton("Load session…")
        b_load.clicked.connect(self.load_session)
        left_lay.addLayout(_row(b_save, b_load))
        window_shortcuts(self, save=self.save_session, load=self.load_session,
                         export_figure=self.export_figure, export_data=self.export_results)

        # Mode tabs: File Import (scrollable) and Direct Input
        mode_tabs = QTabWidget()
        left_lay.addWidget(mode_tabs, 1)
        file_scroll = QScrollArea()
        file_scroll.setWidgetResizable(True)
        file_scroll.setFrameShape(QFrame.NoFrame)
        file_page = QWidget()
        file_scroll.setWidget(file_page)
        mode_tabs.addTab(file_scroll, "File import")
        self.setup_file_import_mode(file_page)

        self.direct_input = DirectInputPanel(self)
        mode_tabs.addTab(self.direct_input, "Direct input")

        # Results display
        results_box = QGroupBox("Results")
        rb_lay = QVBoxLayout(results_box)
        splitter.addWidget(left)
        splitter.addWidget(results_box)
        splitter.setSizes([640, 960])

        # State for manual exclusions: {category: {sample_id: reason}}
        self.manual_exclusions = {}
        # Force-included points: {category: set(sample_id)}, kept despite auto-rejection
        self.manual_inclusions = {}
        # Raw unfiltered data per category, preserved for exclusion table
        self.raw_category_data = {}

        tabs = QTabWidget()
        rb_lay.addWidget(tabs)

        # Results tab
        self.results_text = QPlainTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(mono_font(9))
        tabs.addTab(self.results_text, "Analysis results")

        # Plots tab
        plots_tab = QWidget()
        pl = QVBoxLayout(plots_tab)
        self.category_var = Value("")
        self.category_combo = bind_combo(self.category_var, QComboBox(),
                                         self._on_plot_category_change)
        self.category_combo.setMinimumWidth(120)
        self.data_type_var = Value("40/39 Ratios")
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["40/39 Ratios", "Calculated J"])
        bind_combo(self.data_type_var, self.data_type_combo, self.update_plot)

        # View: the analysed points, the steps of each crystal, the steps
        # grouped by power / step number, or every category's mean side by side.
        self._VIEWS = OrderedDict([("All points", 'all'), ("By crystal", 'crystal'),
                                   ("By step group", 'power'), ("All categories", 'groups')])
        self.crystal_view_var = Value('all')
        self._view_label_var  = Value("All points")
        view_combo = QComboBox()
        view_combo.addItems(list(self._VIEWS))

        def _on_view():
            self.crystal_view_var.set(self._VIEWS.get(self._view_label_var.get(), 'all'))
            self.update_plot()
        bind_combo(self._view_label_var, view_combo, _on_view)

        self.zoom_mode_var = Value("adaptive")
        rb_ad, rb_full = QRadioButton("Adaptive"), QRadioButton("Full range")
        bind_radios(self.zoom_mode_var, [(rb_ad, "adaptive"), (rb_full, "full")],
                    command=self.update_plot)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(lambda: self.update_plot())
        pl.addLayout(_row(QLabel("Category:"), self.category_combo, 15,
                          QLabel("Data type:"), self.data_type_combo, 10,
                          QLabel("View:"), view_combo, 10, rb_ad, rb_full, 10, refresh))

        self.plot_frame = QWidget()
        self._plot_layout = QVBoxLayout(self.plot_frame)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.plot_frame, 1)
        tabs.addTab(plots_tab, "Plots")

        # Manual Exclusion tab
        exclusion_tab = QWidget()
        tabs.addTab(exclusion_tab, "Manual exclusion")
        self._setup_exclusion_tab(exclusion_tab)

    # ── plot area ──────────────────────────────────────────────────────────────
    def _clear_plot(self):
        if self._export_fig is not None:        # drawing for the export dialog
            return
        while self._plot_layout.count():
            w = self._plot_layout.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _plot_message(self, text, size=12):
        if self._export_fig is not None:
            return
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"font-size: {size + 3}px;")
        self._plot_layout.addWidget(lbl)

    @staticmethod
    def _wrapped_suptitle(fig, head, note, fontsize=9):
        """Title plus a note wrapped to the figure width, rewrapped when the
        plot is resized; the layout leaves room for however many lines it takes."""
        st = fig.suptitle(head, fontsize=fontsize)

        def rewrap(_event=None):
            width_px = fig.get_figwidth() * fig.dpi
            char_px = 0.6 * st.get_fontsize() * fig.dpi / 72
            st.set_text(head + '\n' + textwrap.fill(note, max(40, int(0.95 * width_px / char_px))))
        rewrap()
        fig._rewrap = rewrap
        try:
            fig.set_layout_engine('constrained')
        except AttributeError:          # matplotlib < 3.6
            fig.tight_layout(rect=[0, 0, 1, 0.88])

    def _new_figure(self, figsize, dpi=100):
        """The figure a plot is drawn on: a new one for the window, or the
        export dialog's (cleared) while it renders."""
        if self._export_fig is not None:
            self._export_fig.clear()
            self._export_fig._rewrap = None
            return self._export_fig
        return Figure(figsize=figsize, dpi=dpi)

    def _grid_cols(self, n, most):
        """Columns for a grid of n panels: up to *most*, and on an export page
        no more than fit at about 5 cm a column."""
        cols = min(n, most)
        if self._export_fig is not None:
            cols = min(cols, max(1, int(self._export_fig.get_figwidth() / 2.0)))
        return cols

    def _show_figure(self, fig):
        if fig is self._export_fig:
            return None
        canvas = FigureCanvas(fig)
        if hasattr(fig, '_rewrap'):
            canvas.mpl_connect('resize_event', fig._rewrap)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # zoom and pan, as on the plots of the other tools
        self._toolbar = NavigationToolbar2QT(canvas, self.plot_frame)
        self._plot_layout.addWidget(self._toolbar)
        self._plot_layout.addWidget(canvas)
        canvas.draw()
        return canvas

    # ── Figure export: the dialog shared with the other tools ────────────
    def _plot_shown(self):
        """True when the Plots tab shows a plot (not a message)."""
        return any(isinstance(self._plot_layout.itemAt(i).widget(), FigureCanvas)
                   for i in range(self._plot_layout.count()))

    def _categories(self):
        return sorted((self.results or {}).get('ratio4039_results', {}),
                      key=self._natural_sort_key)

    def _category_options(self, depends=None):
        """A tick box per category: those drawn (in panels, or one page each)."""
        return [export.Option(f'cat_{k}', cat, 'bool', True, group='Categories',
                              depends=depends, ident=cat)
                for k, cat in enumerate(self._categories())]

    def _chosen_categories(self, opts):
        return [cat for k, cat in enumerate(self._categories())
                if opts.get(f'cat_{k}', True)]

    def _export_options(self, batch=False):
        """The switches of the export dialog.  batch: 'Export all categories',
        one page per category in the view chosen there."""
        view = self.crystal_view_var.get()
        cats = self._categories()
        opts = []
        if batch:
            opts.append(export.Option(
                'view', 'View', 'choice',
                default=view if view in ('all', 'crystal', 'power') else 'all',
                values=[('all', 'All points'), ('crystal', 'By crystal'),
                        ('power', 'By step group')]))
        opts.append(export.Option(
            'title', 'Title', 'choice',
            values=[('full', 'As in the window'), ('short', 'First line only'),
                    ('none', 'No title')],
            tip='As in the window: the group, n, MSWD and p, and for the '
                'crystal and step-group views the line explaining the '
                'symbols. First line only: the group and data type, and '
                'the name of each crystal or step group over its panel. With '
                'several categories in panels, each panel keeps its category '
                'and, unless No title, its mean and SE.'))
        if batch or view == 'all':
            opts.append(export.Option('ids', 'Crystal IDs next to the points', 'bool', True,
                                      depends=('view', 'all') if batch else None))
        if not batch and view == 'all' and len(cats) > 1:
            cur = self.category_var.get()
            opts += [
                export.Option(
                    'categories', 'Categories', 'choice', group='Categories',
                    values=[('shown', f'{cur} only'),
                            ('panels', 'Several, one panel each')],
                    tip='One panel per category ticked below, stacked as the samples '
                        'of the Age distribution tool: one point-number axis, one '
                        'value axis label. Panel letters label the panels.'),
                export.Option('columns', 'Columns', 'int', 1, lo=1, hi=6,
                              group='Categories', depends=('categories', 'panels'),
                              tip='1: the panels in one stack, as in the Age distribution '
                                  'tool. More columns suit many categories.'),
                export.Option('same_range', 'Same value range in every panel', 'bool',
                              False, group='Categories', depends=('categories', 'panels'),
                              tip='The positions compared on one scale (e.g. a flux '
                                  'gradient), rather than each panel fitted to its points.'),
            ] + self._category_options(depends=('categories', 'panels'))
        elif batch:
            opts += self._category_options()
            opts.append(export.Option(
                'batch_one_pdf', 'One PDF, one page per category', 'bool', True,
                group='Output',
                tip='Untick to write one file per category into a folder. PNG, SVG, '
                    'EPS and TIFF always go that way.'))
        axis = 'X' if view == 'groups' and not batch else 'Y'
        opts += [
            export.Option('v_min', f'{axis} from', 'text', group='Axis ranges',
                          placeholder='auto'),
            export.Option('v_max', 'to', 'text', group='Axis ranges',
                          placeholder='auto'),
        ]
        return opts

    def _draw_panels(self, fig, spec, data_type):
        """The 'all points' plot of each ticked category in its own panel, laid
        out as the samples of the Age distribution tool: panels stacked (in
        columns if asked), one point-number axis at the bottom, one label for
        the value axis.  What every panel shares (the data type and the points
        averaged) is the figure's title.  Each panel's title has its category,
        mean and SE ('First line only': no MSWD, p; 'No title': the category
        alone), so the legends say only what the symbols are and can be merged
        into one."""
        opts = spec.get('options', {})
        title = opts.get('title', 'full')
        cats = self._chosen_categories(opts) or [self.category_var.get()]
        n = len(cats)
        try:
            ncol = max(1, min(int(opts.get('columns', 1)), n))
        except (TypeError, ValueError):
            ncol = 1
        nrow = math.ceil(n / ncol)
        # characters of a title line that fit over one panel
        fs = float(spec.get('font_scale', 1.0) or 1.0)
        width = max(30, int(0.85 * fig.get_figwidth() * 72 / ncol / (0.55 * 9 * fs)))
        axes, common = [], ''
        for k, cat in enumerate(cats):
            ax = fig.add_subplot(nrow, ncol, k + 1, sharex=axes[0] if axes else None)
            axes.append(ax)
            self._export_ax = ax
            try:
                self.create_outlier_plot(cat, data_type)
            finally:
                self._export_ax = None
            hs, labels = ax.get_legend_handles_labels()
            stats = [lab for lab in labels if ' = ' in lab]       # the mean and its SE
            for h, lab in zip(hs, labels):
                h.set_label(re.sub(r' \(n=\d+\)', '', lab.split(' = ')[0]))
            if ax.get_legend() is not None:
                ax.legend(fontsize=8)
            head, _, rest = ax.get_title().partition('\n')
            common = common or head.split(': ', 1)[-1]
            lines = [cat]
            if title != 'none':                 # the category alone with no title
                lines += [self._wrap_items(t, width, ', ') for t in stats]
            if title == 'full':
                lines += [self._wrap_items(t, width) for t in rest.split('\n') if t]
            ax.set_title('\n'.join(lines), fontsize=9)
        if common and title != 'none':
            fig.suptitle(common, fontsize=10)
        ylab = axes[0].get_ylabel()
        for k, ax in enumerate(axes):
            ax.set_ylabel('')
            if k + ncol < n:                    # a panel below: no x axis labels
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)
        fig._panel_ylabel = fig.supylabel(ylab, fontsize=axes[0].xaxis.label.get_fontsize())
        if opts.get('same_range'):
            lims = [ax.get_ylim() for ax in axes]
            for ax in axes:
                ax.set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))
        fig._panel_grid = (nrow, ncol)

    @staticmethod
    def _wrap_items(line, width, sep=',  '):
        """A line such as '(18/23 points,  MSWD = 0.515,  p = 0.95)' wrapped
        between its items (split at *sep*), never inside one."""
        out, cur = [], ''
        for it in line.split(sep):
            nxt = f'{cur}{sep}{it}' if cur else it
            if cur and len(nxt) > width:
                out.append(cur + sep.rstrip())
                cur = it
            else:
                cur = nxt
        return '\n'.join(out + [cur])

    @staticmethod
    def _panel_frame(fig, spec):
        """Margins and gaps of the panels: the frame of a single plot
        (export.fixed_frame), with room above each row for the panel titles,
        between the columns for the value labels and on the left for the one
        axis label.  Measured on the text as drawn, so it holds at any text
        size; no layout engine, so the legends are placed as in the other
        tools' exports."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        nrow, ncol = fig._panel_grid
        axes = fig.axes
        export.fixed_frame(fig, spec)
        r = FigureCanvasAgg(fig).get_renderer()
        pt = 72.0 / fig.dpi
        W, H = (v * 72.0 for v in fig.get_size_inches())
        title_h = max([a.title.get_window_extent(r).height * pt for a in axes
                       if a.get_title()] or [0.0])
        tick_w = max((a.bbox.x0 - a.yaxis.get_tightbbox(r).x0) * pt for a in axes)
        st = fig._suptitle
        sup_h = st.get_window_extent(r).height * pt if st is not None and st.get_text() else 0.0
        ylab = getattr(fig, '_panel_ylabel', None)
        ylab_w = ylab.get_window_extent(r).width * pt if ylab is not None else 0.0
        p = fig.subplotpars
        left = (3.0 + ylab_w + 4.0 + tick_w + 2.0) / W
        top = 1.0 - (4.0 + sup_h + 4.0 + title_h + 4.0) / H
        gap_h, gap_w = title_h + 8.0, tick_w + 10.0
        aw = ((p.right - left) * W - (ncol - 1) * gap_w) / ncol
        ah = ((top - p.bottom) * H - (nrow - 1) * gap_h) / nrow
        if aw > 20 and ah > 20:
            fig.subplots_adjust(left=left, top=top, wspace=gap_w / aw, hspace=gap_h / ah)
        if ylab is not None:
            ylab.set_x(3.0 / W)

    def _export_render(self, fig, spec, category=None):
        """Draw the plot shown on the Plots tab into *fig* as the dialog asks;
        with *category*, that category (a page of 'Export all categories')."""
        opts = spec.get('options', {})
        view = opts.get('view') or self.crystal_view_var.get()
        data_type = self.data_type_var.get()
        if category is None:
            category = self.category_var.get()
            chosen = self._chosen_categories(opts)
            if 'view' in opts and chosen and category not in chosen:
                category = chosen[0]            # the preview of a batch
        keep = (self._pick_pts, self._pick_ax, getattr(self, '_pick_category', None))
        self._export_fig, self._export_opts = fig, dict(opts)
        try:
            if view == 'groups':
                self._plot_all_groups(data_type)
            elif view == 'all' and opts.get('categories') == 'panels':
                self._draw_panels(fig, spec, data_type)
            elif category:
                self._draw_category(view, category, data_type)
        finally:
            self._export_fig, self._export_opts = None, {}
            self._pick_pts, self._pick_ax, self._pick_category = keep
        groups = view == 'groups'
        axes = fig.axes
        title = opts.get('title', 'full')
        if title != 'full' and not getattr(fig, '_panel_grid', None):
            # the title over the plot: the figure's over a grid of panels, where
            # each panel keeps the first line of its own (the crystal or group)
            st = fig._suptitle if fig._suptitle is not None and fig._suptitle.get_text() else None
            for t in ([st] if st is not None else [a.title for a in axes]):
                t.set_text('' if title == 'none' else t.get_text().split('\n')[0])
            for t in ([a.title for a in axes] if st is not None else []):
                t.set_text(t.get_text().split('\n')[0])
            fig._rewrap = None                # keep the note out
        lo, hi = export.opt_float(spec, 'v_min'), export.opt_float(spec, 'v_max')
        if lo is not None or hi is not None:
            for a in axes:
                get, put = (a.get_xlim, a.set_xlim) if groups else (a.get_ylim, a.set_ylim)
                cur = get()
                put(cur[0] if lo is None else lo, cur[1] if hi is None else hi)
        panels = getattr(fig, '_panel_grid', None)
        mode = spec.get('legend', 'auto')
        if panels and len(axes) > 1 and mode in ('below', 'above', 'outside'):
            # outside the panels: one legend for the stack, each symbol once
            target = axes[-1] if mode == 'below' else axes[0]
            export.merge_legends(axes, target)
            export.place_legend(target, spec, span=mode != 'outside')
        else:
            for a in axes:
                export.place_legend(a, spec)
        export.scale_text(fig, spec.get('font_scale', 1.0),
                          spec.get('line_scale', 1.0), spec.get('marker_scale', 1.0))
        if callable(getattr(fig, '_rewrap', None)):
            fig._rewrap()                     # the note wrapped at its scaled size
        if panels:
            self._panel_frame(fig, spec)
        # one plot: the frame the other tools' exports use (the grids of
        # crystals and step groups keep their constrained layout)
        elif len(axes) == 1 and getattr(fig, 'get_layout_engine', lambda: None)() is None:
            export.fixed_frame(fig, spec, has_title=title != 'none')

    def export_figure(self):
        """Export figure…: the plot on the Plots tab, or several categories in
        panels, with page size, text, legend and presets set in the dialog the
        other tools use."""
        if not self._plot_shown():
            QMessageBox.information(self, "Export figure",
                                    "No plot to export: analyse the data first.")
            return
        view = self.crystal_view_var.get()
        name = ('all_categories' if view == 'groups'
                else (self.category_var.get() or 'flux_monitors'))
        name = dfold.safe_filename(name, 'flux_monitors')
        dlg = export.FigureExportDialog(
            self, 'flux_monitors', [export.Panel('plot', 'Plot', fixed=True)],
            self._export_options(), self._export_render,
            default_name=name, defaults=dict(width_cm=17.0, height_cm=11.0))
        dlg.exec_()

    def export_all_categories(self):
        """Export all categories…: one page per category (all points, by crystal
        or by step group), as one PDF or one file each."""
        if not self._categories():
            QMessageBox.information(self, "Export all categories",
                                    "No results to export: analyse the data first.")
            return
        dlg = export.FigureExportDialog(
            self, 'flux_monitors_all', [export.Panel('plot', 'Plot', fixed=True)],
            self._export_options(batch=True), self._export_render,
            default_name='flux_monitors_all', batch=self._export_categories_all,
            batch_pdf_flag='batch_one_pdf', title='Export all categories',
            defaults=dict(width_cm=17.0, height_cm=11.0))
        dlg.exec_()

    def _export_categories_all(self, spec, path, progress):
        """Write the page of every ticked category to *path* (a PDF, or a
        folder with one file each).  Returns the lines to show the user."""
        opts = spec['options']
        one_pdf = spec.get('fmt', 'pdf') == 'pdf' and bool(opts.get('batch_one_pdf', True))

        def page(cat):
            def draw():
                fig = Figure(figsize=export.figsize(spec), facecolor='white')
                self._export_render(fig, spec, category=cat)
                return (fig, spec) if fig.axes else None
            return draw

        items = [(cat, page(cat)) for cat in self._chosen_categories(opts)]
        if not items:
            return ['No category ticked: nothing written.']
        written, pages, empty = export.write_pages(path, spec, items, progress, one_pdf)
        lines = []
        if empty:
            lines.append('Nothing to draw in this view, left out: ' + ', '.join(empty))
        if one_pdf and pages:
            lines.append(f'{path}   ({pages} page{"s" if pages != 1 else ""})')
        return lines + written

    # ---------------------------------------------------------------------
    # Manual exclusion tab
    # ---------------------------------------------------------------------

    _EXCL_COLUMNS = [("#", 35), ("Crystal", 110), ("Include", 55), ("Sample ID", 140),
                     ("Step", 80), ("40/39 Ratio", 110), ("40/39 Error (1σ)", 125),
                     ("J Value", 100), ("J Error (1σ)", 110), ("Rel. Error %", 90),
                     ("Reason for exclusion", 200)]
    _EXCL_COLOURS = {"excluded": ("#ffe0e0", "#aa0000"), "included": ("#ffffff", "#000000"),
                     "forced":   ("#e2f6e2", "#0a7a0a"), "auto":     ("#f3f3f3", "#777777")}

    def _setup_exclusion_tab(self, parent):
        """Build the Manual Exclusion tab UI."""
        lay = QVBoxLayout(parent)

        # ── Top controls ────────────────────────────────────────────────────
        self.excl_cat_var = Value("")
        self.excl_cat_combo = bind_combo(self.excl_cat_var, QComboBox(),
                                         self._on_excl_category_change)
        self.excl_cat_combo.setMinimumWidth(110)
        b_apply = QPushButton("Apply && recalculate")
        b_apply.clicked.connect(self._apply_exclusions)
        b_clear = QPushButton("Clear all exclusions for category")
        b_clear.clicked.connect(self._clear_exclusions)
        b_all = QPushButton("Clear all categories")
        b_all.clicked.connect(self._clear_all_exclusions)
        lay.addLayout(_row(QLabel("Category:"), self.excl_cat_combo, 12, b_apply, 6,
                           b_clear, 6, b_all))

        # ── Info label ───────────────────────────────────────────────────────
        lay.addWidget(_note(
            "Double-click a row to cycle default → excluded → force-included; right-click "
            "(or Enter) an excluded row to give a reason; then 'Apply & recalculate'. "
            "Clicking a point on the plot does the same and recalculates at once. "
            "The rows are what is being averaged: steps, or crystals in 'one point per "
            "crystal' mode. A step excluded in 'every step' mode also stays out of its "
            "crystal's total gas.", size=9))

        # ── Table ────────────────────────────────────────────────────────────
        self.excl_tree = QTreeWidget()
        self.excl_tree.setRootIsDecorated(False)
        self.excl_tree.setColumnCount(len(self._EXCL_COLUMNS))
        self.excl_tree.setHeaderLabels([h for h, _ in self._EXCL_COLUMNS])
        for i, (_, w) in enumerate(self._EXCL_COLUMNS):
            self.excl_tree.setColumnWidth(i, w)
        self.excl_tree.itemDoubleClicked.connect(self._toggle_exclusion_row)
        self.excl_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.excl_tree.customContextMenuRequested.connect(
            lambda pos: self._edit_reason(self.excl_tree.itemAt(pos)))
        self.excl_tree.installEventFilter(self)
        lay.addWidget(self.excl_tree, 1)

        # Summary label at bottom
        self.excl_summary_var = Value("Run the analysis to fill this table.")
        summary = bind_label(self.excl_summary_var, QLabel())
        summary.setStyleSheet("font-style: italic;")
        lay.addWidget(summary)

    def eventFilter(self, obj, event):
        """Enter on a row of the exclusion table edits its reason."""
        from PyQt5.QtCore import QEvent
        if (obj is getattr(self, 'excl_tree', None) and event.type() == QEvent.KeyPress
                and event.key() in (Qt.Key_Return, Qt.Key_Enter)):
            self._edit_reason(self.excl_tree.currentItem())
            return True
        return super().eventFilter(obj, event)

    def _populate_exclusion_table(self):
        """Fill the exclusion table with data for the selected category."""
        self.excl_tree.clear()

        # Update error column headers to match current display-sigma choice
        sigma_label = "2σ" if self._pt_err_mult() == 2.0 else "1σ"
        vname = "40/39" if self.value_is_ratio else (self.value_label or "Value")
        head = self.excl_tree.headerItem()
        head.setText(5, vname)
        head.setText(6, f"{vname} Error (±{sigma_label})")
        head.setText(8, f"J Error (±{sigma_label})")

        category = self.excl_cat_var.get()
        if not category or category not in self.raw_category_data:
            self.excl_summary_var.set("No data for this category.")
            return

        excl_ids = self.manual_exclusions.get(category, {})
        incl_ids = self.manual_inclusions.get(category, set())
        # The rows being averaged (steps or crystals), as in the plot.
        data = self.analysis_rows.get(category) or self.raw_category_data[category]
        # Automatic rejections of the last run, so they are visible here too.
        auto, derived = {}, {}
        try:
            for p in self.results['ratio4039_results'][category]['plot_data']['all_points']:
                if p.get('status') in ('high_error', 'outlier', 'invalid'):
                    auto[p['sample_id']] = p['status']
                elif p.get('status') == 'manual':
                    derived[p['sample_id']] = p.get('reason', '')   # e.g. all steps excluded
        except (KeyError, TypeError):
            pass
        auto_txt = {'high_error': 'auto: relative error above limit',
                    'outlier': 'auto: outlier rejection',
                    'invalid': 'auto: unusable (no error, R ≤ 0 or ³⁹Ar ≤ 0)'}

        def _e(v, mult=1.0):
            return f"{v * mult:.6e}" if _finite(v) else ""

        n_excl = n_incl = n_auto = 0
        mult = self._pt_err_mult()
        items = []
        for row_num, item in enumerate(data, start=1):
            sid      = item['sample_id']
            v4039    = item.get('value40_39')
            e4039    = item.get('uncertainty40_39')
            vj       = item.get('measured_j') if self.value_is_ratio else None
            ej       = item.get('measured_j_error') if self.value_is_ratio else None
            rel_err  = (e4039 / abs(v4039) * 100) if (_finite(v4039) and _finite(e4039)
                                                       and v4039 != 0) else float('nan')
            if sid in excl_ids:
                tag, mark = "excluded", "✗"
                reason = excl_ids.get(sid, "")
                n_excl += 1
            elif sid in incl_ids:
                tag, mark = "forced", "✚"
                reason = "force-included"
                n_incl += 1
            elif sid in derived:
                tag, mark, reason = "excluded", "✗", derived[sid]
                n_excl += 1
            elif sid in auto:
                tag, mark, reason = "auto", "⊘", auto_txt[auto[sid]]
                n_auto += 1
            else:
                tag, mark, reason = "included", "✓", ""

            values = (row_num, item.get('grain_id') or '', mark, sid, self._step_text(item),
                      _e(v4039), _e(e4039, mult), _e(vj), _e(ej, mult),
                      f"{rel_err:.3f}%" if _finite(rel_err) else "", reason)
            it = QTreeWidgetItem([str(v) for v in values])
            it.setData(0, Qt.UserRole, sid)
            bg, fg = self._EXCL_COLOURS[tag]
            for c in range(len(values)):
                it.setBackground(c, QColor(bg))
                it.setForeground(c, QColor(fg))
                if c in (0, 2, 4):
                    it.setTextAlignment(c, Qt.AlignCenter)
            items.append(it)
        self.excl_tree.addTopLevelItems(items)
        for c in range(self.excl_tree.columnCount()):
            self.excl_tree.resizeColumnToContents(c)

        n_total = len(data)
        what = "crystals" if self.crystal_mode_var.get() == 'crystal' else "rows"
        self.excl_summary_var.set(
            f"Category {category}: {n_total} {what}  |  {n_excl} excluded  |  "
            f"{n_auto} auto-rejected  |  {n_incl} force-included  |  "
            f"{n_total - n_excl - n_auto} used"
        )

    def _step_text(self, item):
        """Short step description for tables: power (or label), or the pool."""
        if item.get('step_type') == 'pooled':
            return f"Σ {item.get('n_steps', '?')}/{item.get('crystal_n_steps', '?')} steps"
        p = item.get('power')
        if _finite(p):
            return f"{float(p):g}"
        return str(item.get('label') or '')

    def _cycle_point_state(self, category, sid):
        """Cycle one point through: default → excluded → force-included → default.
        A point is only ever in one of the two override sets."""
        excl = self.manual_exclusions.setdefault(category, {})
        incl = self.manual_inclusions.setdefault(category, set())
        if sid in excl:
            del excl[sid]
            incl.add(sid)          # excluded → force-included
        elif sid in incl:
            incl.discard(sid)      # force-included → default (auto)
        else:
            excl[sid] = ""         # default → excluded

    def _toggle_exclusion_row(self, item, _column=0):
        """Double-click cycles a row: default → excluded → force-included → default."""
        if item is None:
            return
        self._cycle_point_state(self.excl_cat_var.get(), item.data(0, Qt.UserRole))
        self._populate_exclusion_table()

    def _edit_reason(self, item):
        """Ask for the exclusion reason of an excluded row."""
        if item is None:
            return
        row = item.data(0, Qt.UserRole)
        category = self.excl_cat_var.get()
        excl = self.manual_exclusions.get(category, {})
        if row not in excl:
            return   # only edit reasons for excluded rows
        text, ok = QInputDialog.getText(self, "Edit Exclusion Reason",
                                        f"Reason for excluding  {row}:",
                                        QLineEdit.Normal, excl.get(row, ""))
        if ok:
            self.manual_exclusions[category][row] = text.strip()
            self._populate_exclusion_table()

    def _refresh_calculated_j(self):
        """Set each row's J to the J calculated from its own 40/39 ratio with
        the current calibration (FC age + λ from the Calibration dropdown).
        A row without a positive ratio gets no J: a J read from a file is the
        J that file's software assigned, not a monitor measurement.
        Recomputed on every analysis so J follows the chosen calibration."""
        if not getattr(self, 'raw_category_data', None):
            return
        for rows in self.raw_category_data.values():
            for row in rows:
                r  = row.get('value40_39')
                er = row.get('uncertainty40_39')
                if self.value_is_ratio and _finite(r) and r > 0 and _finite(er):
                    jc = self.j_from_r(r, er)
                    row['measured_j']      = jc['j_value']
                    row['measured_j_error'] = jc['j_error']
                else:
                    row['measured_j'] = row['measured_j_error'] = None

    # ---------------------------------------------------------------------
    # One analysis pipeline (Analyse, Apply exclusions, plot clicks,
    # crystal-mode changes and Direct Input all run this)
    # ---------------------------------------------------------------------

    CRYSTAL_MODES = OrderedDict([
        ('steps',       'Every step is a point'),
        ('crystal',     'One point per crystal (³⁹Ar-weighted total gas)'),
        ('fusion_only', 'Single-step (fusion) crystals only'),
    ])

    def _reset_overrides(self):
        """Forget manual exclusions / force-inclusions (new data loaded)."""
        self.manual_exclusions = {}
        self.manual_inclusions = {}

    def _set_loaded_data(self, category_data, source, value_is_ratio,
                         value_label, input_name):
        self.raw_category_data = category_data
        self.loaded_source     = source
        self.value_is_ratio    = bool(value_is_ratio)
        self.value_label       = "40/39 Ratio" if value_is_ratio else (value_label or "Value")
        self.input_name        = input_name
        self._reset_overrides()

    def _show_crystals(self):
        """One line on the crystals in the loaded data."""
        lbl = getattr(self, 'crystal_info_lbl', None)
        if lbl is None:
            return
        fus = heat = 0
        steps = []
        for rows in (self.raw_category_data or {}).values():
            seen = {}
            for t in self._parse_crystal_structure(rows):
                seen[t['grain_id']] = t['crystal_n_steps']
            for n in seen.values():
                if n == 1:
                    fus += 1
                else:
                    heat += 1
                    steps.append(n)
        if not fus and not heat:
            lbl.setText("No data analysed yet.")
            return
        txt = f"{fus + heat} crystals, {fus} fusion, {heat} step-heated"
        if steps:
            lo, hi = min(steps), max(steps)
            txt += f" ({lo} steps)" if lo == hi else f" ({lo}–{hi} steps)"
        lbl.setText(txt + ".")

    def _rerun_if_loaded(self):
        """Crystal basis or step grouping changed: redo the last analysis."""
        if self.raw_category_data and self.results is not None:
            if self._options_valid(show=False):
                self._run_analysis("=== RE-ANALYSIS ===")

    # Option fields whose change re-runs the current analysis.
    def _option_vars(self):
        return (self.use_error_filter_var, self.error_threshold_var,
                self.use_mswd_var, self.mswd_threshold_var, self.mswd_rule_var,
                self.mswd_discordant_var,
                self.use_nmad_var, self.nmad_k_var,
                self.use_reml_var, self.std_age_var, self.std_age_err_var,
                self.lambda_var, self.lambda_err_var)

    def _options_valid(self, show=False):
        """True when every numeric option field holds a number (a field being
        typed in can briefly hold '' or '2.')."""
        checks = [("Relative error threshold", self.error_threshold_var),
                  ("NMAD k threshold", self.nmad_k_var),
                  ("MSWD threshold", self.mswd_threshold_var),
                  ("Standard age", self.std_age_var),
                  ("Standard age error", self.std_age_err_var),
                  ("λ", self.lambda_var), ("λ error", self.lambda_err_var)]
        for name, var in checks:
            try:
                float(var.get())
            except (ValueError, TypeError):
                if show:
                    QMessageBox.critical(self, "Option not a number",
                                         f"'{name}' does not hold a number. Correct it and "
                                         f"analyse again.")
                return False
        return True

    def _schedule_rerun(self, *_):
        """An option changed: re-run the current analysis shortly after (one
        run for several quick changes, e.g. a calibration preset or typing)."""
        if not self.raw_category_data or self.results is None:
            return
        self._rerun_timer.start()

    def _rerun_from_options(self):
        if not self._options_valid(show=False):
            return              # wait until the field being edited holds a number
        try:
            self._run_analysis("=== RE-ANALYSIS (options changed; exclusions kept) ===")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analysing data: {str(e)}")

    def _set_current_data(self, text):
        """Show what 'Analyse data' will re-run, and allow it."""
        self.format_info_var.set(f"Current data: {text}.")
        self.analyze_btn.setEnabled(True)

    @staticmethod
    def _data_signature(data):
        """What identifies a data set (IDs, values, errors, crystals, steps),
        to tell 'the same data again' from new data."""
        if not data:
            return None
        return tuple((cat, tuple((r.get('sample_id'), r.get('value40_39'),
                                  r.get('uncertainty40_39'), r.get('grain_id'),
                                  r.get('power'), r.get('ar39')) for r in rows))
                     for cat, rows in sorted(data.items()))

    def _apply_exclusions(self):
        """Rerun analysis respecting current manual exclusions, refresh plot."""
        if not self.raw_category_data:
            QMessageBox.warning(self, "No data", "Run the analysis before applying exclusions.")
            return
        self._run_analysis("=== RE-ANALYSIS WITH MANUAL EXCLUSIONS ===")
        self.log("\nRe-analysis with manual exclusions complete.")

    def _run_analysis(self, title):
        """raw steps → crystal basis → rejection + weighted mean of R → J."""
        self.results_text.clear()
        self.log(title)
        if self.input_name:
            self.log(f"Data: {self.input_name}")
        if self.value_is_ratio:
            self._refresh_calculated_j()
            self.log(f"J = (e^(λt) − 1)/R with FC {self.std_age_var.get()} Ma, "
                     f"λ = {self.lambda_var.get()} /yr ({self.calib_var.get()}); "
                     f"J errors are from R only (FC age and λ are systematic).")
        mode = self.crystal_mode_var.get()
        self.log(f"Points averaged: {self.CRYSTAL_MODES.get(mode, mode)}")
        self._show_crystals()

        ratio_res, j_res = OrderedDict(), OrderedDict()
        self.analysis_rows, self.crystal_stats, self.step_group_stats = {}, {}, {}
        cats = sorted(self.raw_category_data, key=self._natural_sort_key)
        for category in cats:
            self.log(f"\nAnalysing category {category}...")
            tagged = self._parse_crystal_structure(self.raw_category_data[category])
            rows, derived_excl = self._build_analysis_rows(category, tagged)
            self.analysis_rows[category] = rows
            excl = dict(self.manual_exclusions.get(category, {}))
            excl.update(derived_excl)
            incl = set(self.manual_inclusions.get(category, set()))
            row_ids = {r['sample_id'] for r in rows}
            active = [r for r in rows if r['sample_id'] not in excl]
            n_forced = len(incl & row_ids)
            if len(active) != len(rows) or n_forced:
                self.log(f"  {len(rows) - len(active)} manually excluded, "
                         f"{n_forced} force-included → {len(active)} used")
            ratio_res[category] = self.analyze_category(
                category, active, 'value40_39', 'uncertainty40_39',
                manual_exclusions=excl, all_data=rows, manual_inclusions=incl)
            if self.value_is_ratio:
                j_res[category] = self._j_result_from_r(ratio_res[category])
            if any(t['crystal_n_steps'] > 1 for t in tagged):
                self.crystal_stats[category]    = self._crystal_summary(category, tagged)
                self.step_group_stats[category] = self._step_group_summary(category, tagged)
                if mode == 'steps':
                    n_sh = len({t['grain_id'] for t in tagged if t['crystal_n_steps'] > 1})
                    self.log(f"  Note: {n_sh} step-heated crystal(s) enter with every step. "
                             f"Steps of one crystal are not independent samples of the "
                             f"crystal population (they weight that crystal by its number "
                             f"of steps and mix within- and between-crystal scatter in the "
                             f"MSWD); 'One point per crystal' averages crystals instead.")

        self.results = {'ratio4039_results': ratio_res, 'measured_j_results': j_res}
        self.plot_data = {}
        for category in cats:
            if self.value_is_ratio:
                self.plot_data[category] = {'40_39_ratios': ratio_res[category]['plot_data'],
                                            'measured_j':   j_res[category]['plot_data']}
            else:
                self.plot_data[category] = {self.value_label: ratio_res[category]['plot_data']}

        types = ["40/39 Ratios", "Calculated J"] if self.value_is_ratio else [self.value_label]
        set_combo_items(self.data_type_combo, types, self.data_type_var)
        if self.data_type_var.get() not in types:
            self.data_type_var.set(types[0])
        set_combo_items(self.category_combo, cats, self.category_var)
        set_combo_items(self.excl_cat_combo, cats, self.excl_cat_var)
        if cats and self.category_var.get() not in cats:
            self.category_var.set(cats[0])
        if cats and self.excl_cat_var.get() not in cats:
            self.excl_cat_var.set(self.category_var.get())
        self.update_plot()
        self._populate_exclusion_table()
        self.export_btn.setEnabled(True)
        self.export_fig_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)

    def _clear_exclusions(self):
        """Clear manual exclusions and force-inclusions for the current category."""
        category = self.excl_cat_var.get()
        if category in self.manual_exclusions:
            self.manual_exclusions[category] = {}
        if category in self.manual_inclusions:
            self.manual_inclusions[category] = set()
        self._populate_exclusion_table()

    def _clear_all_exclusions(self):
        """Clear all manual exclusions and force-inclusions across all categories."""
        self.manual_exclusions = {}
        self.manual_inclusions = {}
        self._populate_exclusion_table()

    # ---------------------------------------------------------------------
    # Crystals: structure, pooling, per-crystal and per-step diagnostics
    # ---------------------------------------------------------------------

    def _parse_crystal_structure(self, rows):
        """Copy the rows and tag each with its crystal.

        grain_id        crystal ID set by the loader (grain ID column with
                        blanks filled from above; .arar run-ID prefix or laser
                        power sequence); a row without one is its own crystal
        step_index      1, 2, … in file order within the crystal
        crystal_n_steps number of steps of that crystal
        step_type       'fusion' for a crystal measured in one step,
                        'incremental' for every step of a step-heated crystal
        """
        tagged = []
        for item in rows:
            t = dict(item)
            gid = str(t.get('grain_id') or '').strip()
            t['grain_id'] = gid if gid and gid.lower() not in ('nan', 'none') \
                else str(t['sample_id'])
            tagged.append(t)
        counts = Counter(t['grain_id'] for t in tagged)
        seen = Counter()
        for t in tagged:
            g = t['grain_id']
            seen[g] += 1
            t['step_index']      = seen[g]
            t['crystal_n_steps'] = counts[g]
            t['step_type']       = 'fusion' if counts[g] == 1 else 'incremental'
        return tagged

    def _build_analysis_rows(self, category, tagged):
        """The rows that are averaged under the chosen crystal mode, and any
        exclusions that follow from it ({sample_id: reason})."""
        mode = self.crystal_mode_var.get()
        if mode == 'fusion_only':
            return [t for t in tagged if t['step_type'] == 'fusion'], {}
        if mode == 'crystal':
            return self._pool_crystal_steps(category, tagged)
        return tagged, {}

    def _valid_row(self, t, value_key='value40_39', uncertainty_key='uncertainty40_39'):
        """A row that can be weighted: finite value, positive finite error.
        For ⁴⁰Ar*/³⁹Ar ratios also R > 0 and, where known, ³⁹Ar > 0: a
        blank-level step (³⁹Ar ≈ 0) can give R = −567 ± 17, whose small
        relative error passes the relative-error filter although the ratio
        is meaningless."""
        v, e = t.get(value_key), t.get(uncertainty_key)
        if not (_finite(v) and _finite(e) and float(e) > 0):
            return False
        if self.value_is_ratio and value_key == 'value40_39':
            if float(v) <= 0:
                return False
            a = t.get('ar39')
            if _finite(a) and float(a) <= 0:
                return False
        return True

    def _pool_value(self, steps):
        """(R, σR at 1σ, method) for the steps of one crystal.

        With ³⁹Ar for every step: the total-gas ratio R = Σ fᵢRᵢ,
        fᵢ = ³⁹Arᵢ / Σ³⁹Ar (blank-level steps with negative ³⁹Ar weigh 0),
        σ = √(Σ fᵢ²σᵢ²) with independent step errors and fᵢ taken as exact.
        That is what a total fusion of the crystal would have measured, so
        it can be averaged with single-step fusions.
        Without ³⁹Ar: the inverse-variance mean of the steps, its error × √MSWD
        when the MSWD exceeds the Wendt & Carl limit.  That weights steps by
        precision, not by gas, so it is not a total-gas value."""
        vals = [float(s['value40_39']) for s in steps]
        sigs = [float(s['uncertainty40_39']) for s in steps]
        a39  = [s.get('ar39') for s in steps]
        if all(_finite(a) for a in a39) and sum(max(float(a), 0.0) for a in a39) > 0:
            tot = sum(max(float(a), 0.0) for a in a39)
            f = [max(float(a), 0.0) / tot for a in a39]
            R = sum(fi * v for fi, v in zip(f, vals))
            s = math.sqrt(sum((fi * e) ** 2 for fi, e in zip(f, sigs)))
            return R, s, "³⁹Ar-weighted total gas"
        st = self._wm_stats(vals, sigs, reml=False)
        return (st['mean'], st['se_od'],
                "inverse-variance mean of steps (no ³⁹Ar: not a total-gas value)")

    def _pool_crystal_steps(self, category, tagged):
        """One row per crystal.  A single-step crystal passes through as it is
        (same ID, so its exclusions carry over).  A step-heated crystal becomes
        '<crystal>_pooled' from its steps that are not manually excluded; when
        every step is excluded the crystal is shown, excluded."""
        excl = self.manual_exclusions.get(category, {})
        crystals = OrderedDict()
        for t in tagged:
            crystals.setdefault(t['grain_id'], []).append(t)
        out, derived = [], {}
        for gid, steps in crystals.items():
            if len(steps) == 1:
                out.append(steps[0])
                continue
            good = [s for s in steps if self._valid_row(s)]
            use = [s for s in good if s['sample_id'] not in excl]
            all_out = not use
            if all_out:
                use = good
            if not use:
                continue
            R, s_r, method = self._pool_value(use)
            sid = f"{gid}_pooled"
            row = dict(steps[-1])
            row.update({
                'sample_id': sid, 'value40_39': R, 'uncertainty40_39': s_r,
                'value': R, 'uncertainty': s_r, 'step_type': 'pooled',
                'n_steps': len(use), 'crystal_n_steps': len(steps),
                'step_index': None, 'power': None, 'label': 'Σ',
                'pool_method': method, 'file_reason': None,
                'ar39': (sum(max(float(s['ar39']), 0.0) for s in use)
                         if all(_finite(s.get('ar39')) for s in use) else None),
            })
            if self.value_is_ratio and R > 0:
                jc = self.j_from_r(R, s_r)
                row['measured_j'], row['measured_j_error'] = jc['j_value'], jc['j_error']
            else:
                row['measured_j'] = row['measured_j_error'] = None
            if all_out:
                derived[sid] = 'all steps excluded'
            self.log(f"    Crystal {gid}: {len(use)}/{len(steps)} steps → "
                     f"{self._fmtv(R)} ± {s_r:.3g} (1σ)  [{method}]")
            out.append(row)
        return out, derived

    def _usable_steps(self, category, tagged):
        """Steps for the per-step comparison: valid, not manually excluded and,
        when the relative-error filter is on, within its limit.  No outlier
        rejection, so the comparison sees the steps as measured."""
        excl = self.manual_exclusions.get(category, {})
        lim = self.error_threshold_var.get() if self.use_error_filter_var.get() else None
        out = []
        for t in tagged:
            if t['sample_id'] in excl or not self._valid_row(t):
                continue
            v, e = float(t['value40_39']), float(t['uncertainty40_39'])
            if lim is not None and (v == 0 or e / abs(v) * 100.0 > lim):
                continue
            out.append(t)
        return out

    def _step_group(self, t):
        """(sort key, label) of a step for the per-step comparison."""
        if t['crystal_n_steps'] == 1:
            return (0, 0.0), '1-step crystals'
        if self.step_group_var.get() == 'power' and _finite(t.get('power')):
            p = float(t['power'])
            return (1, p), f"power {p:g}"
        return (2, float(t['step_index'])), f"step {t['step_index']}"

    def _wm_stats(self, vals, sigs, reml=None):
        """Weighted mean, SE, √MSWD-scaled SE, MSWD, p and (optionally) REML
        of one set; 'best_*' is what the tool reports for it (REML when the
        REML box is ticked, else WM with SE × √MSWD when the MSWD exceeds the
        Wendt & Carl (1991) 95% limit for n, the rule the other tools use)."""
        nan = float('nan')
        n = len(vals)
        if n == 0:
            return None
        w = [1.0 / s ** 2 for s in sigs]
        W = sum(w)
        m = sum(wi * v for wi, v in zip(w, vals)) / W
        se = math.sqrt(1.0 / W)
        if n > 1:
            mswd = sum(wi * (v - m) ** 2 for wi, v in zip(w, vals)) / (n - 1)
            p = self.mswd_p_value(mswd, n)
        else:
            mswd = p = nan
        wc = mswd_critical(n)
        over = n > 1 and _finite(mswd) and mswd > wc
        se_od = se * math.sqrt(mswd) if over else se
        out = dict(n=n, mean=m, se=se, se_od=se_od, mswd=mswd, p=p, wc_limit=wc,
                   overdispersed=over, reml_mean=nan, reml_se=nan, tau=nan)
        use_reml = self.use_reml_var.get() if reml is None else reml
        if use_reml and n >= 2:
            r = self.reml_random_effects([{'v': v, 'e': s} for v, s in zip(vals, sigs)], 'v', 'e')
            if r:
                out.update(reml_mean=r['weighted_mean'], reml_se=r['se_1sigma'], tau=r['tau'])
        if use_reml and _finite(out['reml_mean']):
            out.update(best_mean=out['reml_mean'], best_se=out['reml_se'], method='REML')
        else:
            out.update(best_mean=m, best_se=se_od,
                       method='WM + √MSWD (W&C)' if over else 'WM')
        return out

    def _fmtv(self, v):
        if not _finite(v):
            return "n/a"
        v = float(v)
        return f"{v:.4e}" if (v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5)) else f"{v:.6f}"

    def _crystal_summary(self, category, tagged):
        """Per step-heated crystal: its total gas (what 'one point per crystal'
        averages) next to the inverse-variance mean of its steps and their
        MSWD, i.e. whether the steps of that crystal agree."""
        excl = self.manual_exclusions.get(category, {})
        crystals = OrderedDict()
        for t in tagged:
            crystals.setdefault(t['grain_id'], []).append(t)
        out = OrderedDict()
        for gid, steps in crystals.items():
            if len(steps) < 2:
                continue
            good = [s for s in steps if self._valid_row(s) and s['sample_id'] not in excl]
            if not good:
                continue
            R, s_r, method = self._pool_value(good)
            st = self._wm_stats([float(s['value40_39']) for s in good],
                                [float(s['uncertainty40_39']) for s in good], reml=False)
            out[gid] = dict(n_used=len(good), n_steps=len(steps), tg=R, tg_se=s_r,
                            method=method, wm=st['mean'], wm_se=st['se'],
                            mswd=st['mswd'], p=st['p'])
        if out:
            self.log(f"  Step-heated crystals ({len(out)}): steps used, total gas ±1σ, "
                     f"mean of steps ±1σ, MSWD of steps, p")
            for gid, c in out.items():
                self.log(f"    {gid:<22} {c['n_used']:>2}/{c['n_steps']:<2}  "
                         f"{self._fmtv(c['tg'])} ± {c['tg_se']:.2g}   "
                         f"{self._fmtv(c['wm'])} ± {c['wm_se']:.2g}   "
                         f"MSWD {c['mswd']:.2f}  p {c['p']:.3f}")
        return out

    def _step_group_summary(self, category, tagged):
        """Weighted mean per step group (laser power or step number; 1-step
        crystals form their own group) and a test of whether the group means
        agree."""
        steps = self._usable_steps(category, tagged)
        groups = OrderedDict()
        for t in sorted(steps, key=lambda t: self._step_group(t)[0]):
            groups.setdefault(self._step_group(t)[1], []).append(t)
        res = OrderedDict()
        for lab, items in groups.items():
            st = self._wm_stats([float(t['value40_39']) for t in items],
                                [float(t['uncertainty40_39']) for t in items])
            st['n_crystals'] = len({t['grain_id'] for t in items})
            st['ids'] = [t['sample_id'] for t in items]
            res[lab] = st
        between = None
        if len(res) >= 2:
            between = self._wm_stats([s['best_mean'] for s in res.values()],
                                     [s['best_se'] for s in res.values()], reml=False)
        by = 'laser power' if self.step_group_var.get() == 'power' else 'step number'
        self.log(f"  Step groups by {by} (manual exclusions and the relative-error filter "
                 f"applied, no outlier rejection):")
        for lab, s in res.items():
            mswd = f"{s['mswd']:.2f}" if _finite(s['mswd']) else "  – "
            p = f"{s['p']:.3f}" if _finite(s['p']) else "  – "
            self.log(f"    {lab:<16} n={s['n']:<3} {self._fmtv(s['best_mean'])} ± "
                     f"{s['best_se']:.2g} (1σ, {s['method']})   MSWD {mswd}  p {p}")
        if between:
            self.log(f"    Between groups: MSWD = {between['mswd']:.2f}, p = {between['p']:.3f} "
                     f"({len(res)} group means; groups that share crystals are not "
                     f"independent, so a crystal effect common to its steps makes this "
                     f"test conservative)")
        return {'groups': res, 'between': between, 'by': by}

    def _monitor_cfg(self):
        """Calibration and monitor keys of the current settings."""
        try:
            cfg = dict(stdAgeMa=float(self.std_age_var.get()),
                       stdAgeErrMa=float(self.std_age_err_var.get()),
                       lambdaYr=float(self.lambda_var.get()),
                       lambdaErrYr=float(self.lambda_err_var.get()))
        except (ValueError, AttributeError):
            cfg = dict(stdAgeMa=STD_AGE_DEFAULT, stdAgeErrMa=STD_AGE_ERR,
                       lambdaYr=LAMBDA_DEFAULT, lambdaErrYr=LAMBDA_ERR)
        mc = getattr(self, 'monitor_combo', None)
        cfg.update(mc.keys() if mc is not None else monitor_keys('FCs'))
        if getattr(self, '_route', None):
            try:
                return route_cfg(self._route)
            except (ValueError, KeyError, TypeError):
                pass
        return cfg

    def _show_route(self):
        lbl = getattr(self, 'route_lbl', None)
        if lbl is not None:
            r = route_from_cfg(self._monitor_cfg())
            lbl.setText(route_line(r))
            lbl.setToolTip(route_text(r))

    def _open_calibrate(self):
        """Calibration… : the dialog shared with the other tools."""
        dlg = CalibrationDialog(self._monitor_cfg(), parent=self, age_source=False, stored=False)
        if dlg.exec_() != QDialog.Accepted:
            return
        rk = dlg.get_params()['route_keys']
        self._route = rk['route']
        cfg = route_cfg(self._route)
        self._applying_preset = True
        name = next((n for n, *_ in FC_PRESETS if n == rk['route'].get('calibration')), 'Custom')
        self.calib_var.set(name)
        self.std_age_var.set(f"{cfg['stdAgeMa']:g}")
        self.std_age_err_var.set(f"{cfg['stdAgeErrMa']:g}")
        self.lambda_var.set(f"{cfg['lambdaYr']:.6g}")
        self.lambda_err_var.set(f"{cfg['lambdaErrYr']:.6g}")
        self._applying_preset = False
        self.monitor_combo.set_monitor(cfg['monitor'] if cfg['monitor'] in STANDARD_IDS else 'FCs')
        self._show_route()
        if self.results:
            self._run_analysis('calibration')

    def _monitor_R(self):
        """R of the monitor relative to FCs (1 for FCs)."""
        return self._monitor_cfg()['monitorR']

    def _ingrowth(self):
        """e^{λ t_M} − 1 of the monitor on the current calibration, so that
        J = this / R for a measured monitor ratio R: R_M/FCs (e^{λ t_FCs} − 1),
        anchored where the preset has a chain (arar_math.fc_ingrowth)."""
        try:
            lambda_val = float(self.lambda_var.get())
            fc_age      = float(self.std_age_var.get())
            fc_age_error = float(self.std_age_err_var.get())
        except (ValueError, AttributeError):
            lambda_val, fc_age, fc_age_error = LAMBDA_DEFAULT, STD_AGE_DEFAULT, STD_AGE_ERR
        if getattr(self, '_route', None):
            return monitor_ingrowth(self._monitor_cfg())
        return fc_ingrowth(fc_age, lambda_val, fc_age_error) * self._monitor_R()

    def _j_result_from_r(self, res):
        """The J result as the image of the R result under J = (e^{λt} − 1)/R.

        J is not a second measurement: it is a monotone function of R, so it
        has the same accepted / rejected points, MSWD and p as the R analysis,
        and its mean is J of the mean R (first-order errors, σJ/J = σR/R).
        Running a separate rejection on J could keep a different set."""
        c = self._ingrowth()
        nan = float('nan')

        def J(r):
            return c / r if (_finite(r) and r > 0) else nan

        def s_j(s, r):
            return c * s / r ** 2 if (_finite(s) and _finite(r) and r > 0) else nan

        pd0 = res.get('plot_data') or {}
        pts = []
        for p in pd0.get('all_points', []):
            q = dict(p)
            q['value'] = J(p.get('value'))
            q['uncertainty'] = s_j(p.get('uncertainty'), p.get('value'))
            if _finite(q['value']):
                pts.append(q)
        g = lambda k: pd0.get(k, res.get(k, nan))
        wm, rm = g('weighted_mean'), g('reml_mean')
        new = dict(pd0)
        new.update(
            all_points=pts,
            kept_points=[q for q in pts if q.get('status') in ('kept', 'included')],
            removed_points=[q for q in pts if q.get('status') not in ('kept', 'included')],
            weighted_mean=J(wm), standard_error=s_j(g('standard_error'), wm),
            fe_se_1s=s_j(g('fe_se_1s'), wm), od_se_1s=s_j(g('od_se_1s'), wm),
            reml_mean=J(rm), reml_se_1s=s_j(g('reml_se_1s'), rm),
            reml_tau=s_j(g('reml_tau'), rm))
        new['reml_tau2'] = new['reml_tau'] ** 2 if _finite(new['reml_tau']) else nan
        out = dict(res)
        for k in ('weighted_mean', 'standard_error', 'fe_se_1s', 'od_se_1s',
                  'reml_mean', 'reml_se_1s', 'reml_tau', 'reml_tau2'):
            out[k] = new[k]
        out['plot_data'] = new
        return out

    # ---------------------------------------------------------------------
    # NMAD outlier rejection
    # ---------------------------------------------------------------------

    def apply_nmad(self, data, value_key='value', uncertainty_key='uncertainty', k=2.5):
        """
        Normalised Median Absolute Deviation outlier rejection.

        NMAD = 1.4826 × median(|xᵢ − median(x)|)
        Flags points where |xᵢ − median(x)| > k × NMAD.

        Unlike MSWD-based methods the median is resistant to masking by
        multiple simultaneous outliers.  It uses the values only, not their
        errors, so with unequal errors it removes the least precise points
        first even when they agree within error.  Simulated on good data
        (n = 15, errors varying ×3, no excess scatter) it rejects at least one
        good point in 70 % of sets at k = 2, 49 % at k = 2.5, 32 % at k = 3.
        Default k = 2.5, as in the Age distribution tool.
        """
        if len(data) < 3:
            return {'final_data': data, 'removed_items': []}

        values = [item[value_key] for item in data]
        med    = sorted(values)[len(values) // 2] if len(values) % 2 else (
                 sorted(values)[len(values)//2 - 1] + sorted(values)[len(values)//2]) / 2.0
        abs_devs = [abs(v - med) for v in values]
        mad_sorted = sorted(abs_devs)
        n = len(mad_sorted)
        mad = mad_sorted[n//2] if n % 2 else (mad_sorted[n//2-1] + mad_sorted[n//2]) / 2.0
        nmad = 1.4826 * mad

        if nmad == 0:
            return {'final_data': data, 'removed_items': []}

        threshold = k * nmad
        kept, removed = [], []
        for item in data:
            if abs(item[value_key] - med) > threshold:
                self.log(f"  NMAD: removing {item.get('sample_id','')} "
                         f"(|{item[value_key]:.4e} − {med:.4e}| = {abs(item[value_key]-med):.4e} "
                         f"> {threshold:.4e} = {k}×NMAD)")
                removed.append(item)
            else:
                kept.append(item)

        self.log(f"NMAD result: {len(kept)} kept, {len(removed)} removed "
                 f"(NMAD={nmad:.4e}, threshold={threshold:.4e})")
        return {'final_data': kept, 'removed_items': removed}

    def setup_file_import_mode(self, page):
        lay = QVBoxLayout(page)

        # Import section
        imp = QGroupBox("Step 1: Import data")
        il = QVBoxLayout(imp)
        il.addWidget(QLabel("Data file:"))
        self.raw_file_var = Value("")
        file_le = bind_line_edit(self.raw_file_var, QLineEdit())
        file_le.setReadOnly(True)
        il.addWidget(file_le)
        b_arar = QPushButton("Load .arar / .ararstack…")
        b_arar.setToolTip("Session files from the Data converter or Plateau & isochron "
                          "(.arar, .ararpt, .ararstack)")
        b_arar.clicked.connect(self.browse_arar_files)
        b_tab = QPushButton("Import Excel/CSV…")
        b_tab.clicked.connect(self.browse_raw_file)
        il.addLayout(_row(b_arar, 8, b_tab))
        self.format_info_var = Value("No file loaded")
        self._format_info_lbl = bind_label(self.format_info_var, QLabel())
        self._format_info_lbl.setWordWrap(True)
        self._format_info_lbl.setStyleSheet("color: blue;")
        il.addWidget(self._format_info_lbl)
        lay.addWidget(imp)

        # Options
        opts = QGroupBox("Analysis options")
        ol = QVBoxLayout(opts)
        lay.addWidget(opts)

        # Quality filters
        quality = QGroupBox("Data quality")
        ql = QVBoxLayout(quality)
        self.use_error_filter_var = Value(True)
        ql.addWidget(bind_check(self.use_error_filter_var,
                                QCheckBox("Apply relative error filter")))
        self.error_threshold_var = Value(6.0)
        ql.addLayout(_row(20, QLabel("Threshold:"),
                          bind_spin(self.error_threshold_var, _spin(0.1, 100.0, 0.5)),
                          QLabel("% (1σ)")))
        ol.addWidget(quality)

        # Outlier rejection
        outl = QGroupBox("Outlier rejection")
        orl = QVBoxLayout(outl)
        self.use_nmad_var = Value(True)
        cb_nmad = bind_check(self.use_nmad_var, QCheckBox("NMAD outlier rejection"))
        cb_nmad.setToolTip("Removes a point if |x − median| > k × NMAD.")
        orl.addWidget(cb_nmad)
        self.nmad_k_var = Value(2.5)
        orl.addLayout(_row(20, QLabel("k threshold:"),
                           bind_spin(self.nmad_k_var, _spin(0.5, 20.0, 0.5))))
        self.use_mswd_var = Value(False)
        cb_mswd = bind_check(self.use_mswd_var,
                             QCheckBox("MSWD outlier rejection (one point at a time)"))
        cb_mswd.setToolTip("Each round removes the point most at odds with the mean of the "
                           "others,\nuntil the MSWD is within the limit.")
        orl.addWidget(cb_mswd)
        self.mswd_rule_var = Value('wc')
        rb_wc, rb_fixed = QRadioButton("Wendt && Carl limit for n"), QRadioButton("fixed:")
        bind_radios(self.mswd_rule_var, [(rb_wc, 'wc'), (rb_fixed, 'fixed')])
        self.mswd_threshold_var = Value(2.0)
        orl.addLayout(_row(20, QLabel("Stop at MSWD ≤"), rb_wc, 8, rb_fixed,
                           bind_spin(self.mswd_threshold_var, _spin(1.0, 100.0, 0.5))))
        self.mswd_discordant_var = Value(True)
        cb_disc = bind_check(self.mswd_discordant_var, QCheckBox(
            "Only remove a point that is discordant with the others (5 %, Bonferroni for n)"))
        cb_disc.setToolTip(
            "A point goes only if it is a significant outlier (5 %, Bonferroni for n),\n"
            "judged against the others' errors and excess scatter (externally studentized\n"
            "residual, Viechtbauer & Cheung 2010). Scatter that remains is left to\n"
            "REML / √MSWD. Unticked, points go until the MSWD is reached, which trims\n"
            "real scatter.")
        orl.addLayout(_row(20, cb_disc))
        orl.addLayout(_row(44, _note("Removes only significant outliers."), stretch=False))
        ol.addWidget(outl)

        # Statistical Models
        stats = QGroupBox("Statistical models")
        sl = QVBoxLayout(stats)
        self.use_reml_var = Value(True)
        cb_reml = bind_check(self.use_reml_var, QCheckBox("REML random-effects"))
        cb_reml.setToolTip(
            "Estimates the excess scatter τ² by restricted maximum likelihood and reports\n"
            "the random-effects mean and its SE. Unticked: weighted mean, SE × √MSWD when\n"
            "the MSWD exceeds the Wendt & Carl (1991) limit 1 + 2√(2/(n−1)).")
        sl.addWidget(cb_reml)
        sl.addWidget(_note("Better at assessing scatter. Unticked: inverse-variance weighted "
                           "mean. The weighted mean, MSWD and p are always calculated."))
        ol.addWidget(stats)

        # Display σ
        disp = QGroupBox("Reported uncertainties")
        dl = QVBoxLayout(disp)
        self.display_sigma_var = Value('1sigma')

        def _redisplay():
            self.update_plot()
            self._populate_exclusion_table()
        rb_1s, rb_2s = QRadioButton("±1σ"), QRadioButton("±2σ")
        bind_radios(self.display_sigma_var, [(rb_1s, '1sigma'), (rb_2s, '2sigma')],
                    command=_redisplay)
        dl.addLayout(_row(QLabel("Report SE as:"), 10, rb_1s, rb_2s))
        ol.addWidget(disp)

        # Crystals: which points are averaged, and how steps are compared
        cryst = QGroupBox("Crystals (fusion and step heating)")
        cyl = QVBoxLayout(cryst)
        cyl.addWidget(QLabel("Points averaged:"))
        self.crystal_mode_var = Value('steps')
        pairs = []
        for key, text in self.CRYSTAL_MODES.items():
            rb = QRadioButton(text)
            pairs.append((rb, key))
            cyl.addLayout(_row(12, rb))
        bind_radios(self.crystal_mode_var, pairs, command=self._rerun_if_loaded)
        self.step_group_var = Value('power')
        rb_pw, rb_st = QRadioButton("Laser power"), QRadioButton("Step number")
        bind_radios(self.step_group_var, [(rb_pw, 'power'), (rb_st, 'step')],
                    command=self._rerun_if_loaded)
        cyl.addLayout(_row(QLabel("Compare steps by:"), rb_pw, 8, rb_st))
        cryst.setToolTip(
            "A crystal is the grain ID (CSV / Direct input; a blank cell continues the\n"
            "crystal above) or, for .arar files, the run-ID prefix or the laser-power\n"
            "sequence. One step = fusion; several = step-heated. 'One point per crystal'\n"
            "uses each step-heated crystal's ³⁹Ar-weighted total-gas R. Per-crystal and\n"
            "per-step-group statistics are in the results, the export and the Plots tab.")
        cyl.addWidget(_note("What are my crystals up to?"))
        self.crystal_info_lbl = _note("No data analysed yet.")
        cyl.addWidget(self.crystal_info_lbl)
        ol.addWidget(cryst)

        # Standard age / decay constant settings
        std = QGroupBox("Calibration (for J)")
        _stv = QVBoxLayout(std)
        _cal_row = QHBoxLayout()
        _b_cal = QPushButton("Calibration…")
        _b_cal.setToolTip("Monitor and calibration J is calculated on: a route from Standards,\n"
                          "or typed values under Advanced.")
        _b_cal.clicked.connect(self._open_calibrate)
        _cal_row.addWidget(_b_cal)
        _cal_row.addStretch()
        _stv.addLayout(_cal_row)
        self.route_lbl = _note('')
        _stv.addWidget(self.route_lbl)
        # the preset fields are kept, hidden: they hold the values the rest of
        # the tool and its exports read
        _oldw = QWidget()
        _oldw.setVisible(False)
        _stv.addWidget(_oldw)
        stl = QGridLayout(_oldw)
        self._route = None
        self.std_age_var      = Value(f"{STD_AGE_DEFAULT:g}")
        self.std_age_err_var  = Value(f"{STD_AGE_ERR:g}")
        self.lambda_var       = Value(f"{LAMBDA_DEFAULT:.6g}")
        self.lambda_err_var   = Value(f"{LAMBDA_ERR:.6g}")
        self.calib_var        = Value("")
        self._applying_preset = False

        # Calibration preset dropdown: names + paired λ come from arar_math.
        _calib_names = [r[0] for r in FC_PRESETS]

        def _apply_calib_preset(*_):
            name = self.calib_var.get()
            if name == 'Custom':
                return
            fc = next((r for r in FC_PRESETS if r[0] == name), None)
            if fc is None:
                return
            lam = next((r for r in LAMBDA_PRESETS
                        if r[0] == FC_PAIRED_LAMBDA.get(name)), None)
            self._applying_preset = True
            self.std_age_var.set(f"{fc[1]:g}")
            self.std_age_err_var.set(f"{fc[2]:g}")
            if lam is not None:
                self.lambda_var.set(f"{lam[1]:g}")
                self.lambda_err_var.set(f"{lam[2]:g}")
            self._applying_preset = False

        def _on_std_edit(*_):
            # A manual edit to any field switches the dropdown to "Custom".
            if self._applying_preset:
                return
            if self.calib_var.get() != 'Custom':
                self.calib_var.set('Custom')

        self.monitor_combo = MonitorCombo()
        stl.addWidget(QLabel("Monitor:"), 0, 0)
        stl.addWidget(self.monitor_combo, 0, 1, 1, 2)
        self.calib_combo = QComboBox()
        self.calib_combo.addItems(_calib_names + ['Custom'])
        bind_combo(self.calib_var, self.calib_combo, _apply_calib_preset)
        stl.addWidget(QLabel("Calibration:"), 1, 0)
        stl.addWidget(self.calib_combo, 1, 1, 1, 2)
        self.monitor_info = _note('')
        stl.addWidget(self.monitor_info, 7, 0, 1, 3)

        def _show_monitor(*_):
            txt = '\n'.join(t for t in [monitor_text(self._monitor_cfg())]
                            + calibration_notes(self._monitor_cfg()) if t)
            self.monitor_info.setText(txt)
            self.monitor_info.setVisible(bool(txt))
        self.monitor_combo.currentIndexChanged.connect(_show_monitor)
        for i, (label, var, unit) in enumerate([
                ("Standard age (Ma):",       self.std_age_var,     ""),
                ("Standard age error (Ma):", self.std_age_err_var, "1σ"),
                ("λ (yr⁻¹):",                self.lambda_var,      ""),
                ("λ error (yr⁻¹):",          self.lambda_err_var,  "1σ")], start=2):
            stl.addWidget(QLabel(label), i, 0)
            stl.addWidget(bind_line_edit(var, QLineEdit()), i, 1)
            stl.addWidget(_note(unit), i, 2)

        for _v in (self.std_age_var, self.std_age_err_var,
                   self.lambda_var, self.lambda_err_var):
            _v.on_change(_on_std_edit)
            _v.on_change(_show_monitor)

        # the suite-wide default calibration (arar_math.DEFAULT_FC_NAME)
        self.calib_var.set(DEFAULT_FC_NAME)
        _apply_calib_preset()
        stl.addWidget(_note("The monitor is the standard measured here; with one other than "
                            "FCs, J uses its age on the calibration (Standards…) and the "
                            "export adds its R as FCs-equivalent. Pick a calibration, or edit "
                            "a field to switch to Custom."), 6, 0, 1, 3)
        _show_monitor()
        self._show_route()
        ol.addWidget(std)

        self.analyze_btn = QPushButton("Analyse data")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze_data)
        lay.addWidget(self.analyze_btn)
        self.export_fig_btn = QPushButton("Export figure…")
        self.export_fig_btn.setToolTip("The plot on the Plots tab, with page size, text "
                                       "size, legend and presets, and a live preview.")
        self.export_fig_btn.setEnabled(False)
        self.export_fig_btn.clicked.connect(self.export_figure)
        self.export_all_btn = QPushButton("Export all categories…")
        self.export_all_btn.setToolTip("One page per category (all points, by crystal or "
                                       "by step group), as one PDF or one file each.")
        self.export_all_btn.setEnabled(False)
        self.export_all_btn.clicked.connect(self.export_all_categories)
        self.export_btn = QPushButton("Export XLSX / CSV…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        lay.addLayout(_row(self.export_fig_btn, self.export_all_btn, self.export_btn,
                           stretch=False))
        lay.addWidget(_note("Changing an option re-runs the analysis; exclusions are kept."))
        lay.addStretch()

        # Re-run the current analysis whenever an option changes.
        for _v in self._option_vars():
            _v.on_change(self._schedule_rerun)

    def log(self, message):
        self.results_text.appendPlainText(str(message))
        bar = self.results_text.verticalScrollBar()
        bar.setValue(bar.maximum())
        self.results_text.repaint()

    def analyze_category(self, category_name, category_data, value_key='value', uncertainty_key='uncertainty',
                        manual_exclusions=None, all_data=None, manual_inclusions=None):
        # Total points including any manually excluded ones
        total_points = len(all_data) if all_data is not None else len(category_data)

        _nan = float('nan')
        _empty = {'all_points': [], 'kept_points': [], 'removed_points': [],
                  'weighted_mean': _nan, 'standard_error': _nan, 'mswd': _nan}

        if not category_data and not all_data:
            return {
                'category': category_name, 'initial_points': total_points, 'final_points': 0,
                'weighted_mean': _nan, 'standard_error': _nan,
                'percent_error_2sigma': _nan, 'percent_error_1sigma': _nan,
                'mswd': _nan, 'plot_data': _empty
            }

        # A row without a finite value and a positive, finite error cannot be
        # weighted (1/σ² would fail); for ratios R ≤ 0 or ³⁹Ar ≤ 0 is not a
        # usable R either (see _valid_row).  Shown as rejected, not averaged.
        invalid_ids = {it['sample_id'] for it in category_data
                      if not self._valid_row(it, value_key, uncertainty_key)}
        if invalid_ids:
            self.log(f"{len(invalid_ids)} point(s) without a usable value/uncertainty "
                     f"(or R ≤ 0, ³⁹Ar ≤ 0) left out")
            category_data = [it for it in category_data if it['sample_id'] not in invalid_ids]

        original_data = category_data.copy()

        data_with_rel_err = []
        apply_error_filter = self.use_error_filter_var.get()
        error_limit = self.error_threshold_var.get()

        for item in category_data:
            denom = abs(item[value_key])
            relative_error = (item[uncertainty_key] / denom * 100) if denom != 0 else float('inf')
            new_item = item.copy()
            new_item['relative_error'] = relative_error
            data_with_rel_err.append(new_item)

        # Force-included points bypass every auto-rejection step. Split them out so
        # the error filter / NMAD / MSWD only ever act on the rest, then
        # add them back before the final weighted mean.
        incl_ids = set(manual_inclusions or set())
        protected  = [it for it in data_with_rel_err if it['sample_id'] in incl_ids]
        candidates = [it for it in data_with_rel_err if it['sample_id'] not in incl_ids]

        if apply_error_filter:
            filtered_data = [item for item in candidates if item['relative_error'] <= error_limit]
            self.log(f"After error filter (>{error_limit}%): {len(filtered_data)}/{len(candidates)} points"
                     + (f"  (+{len(protected)} force-included)" if protected else ""))
        else:
            filtered_data = candidates.copy()
            self.log(f"Error filter disabled: {len(filtered_data)} points"
                     + (f"  (+{len(protected)} force-included)" if protected else ""))

        def calculate_weighted_mean_mswd(data):
            if len(data) == 0:
                return {'weighted_mean': float('nan'), 'standard_error': float('nan'), 'mswd': float('nan'), 'n': 0}

            weights = [1 / (item[uncertainty_key] ** 2) for item in data]
            total_weight = sum(weights)
            weighted_mean = sum(item[value_key] * weights[i] for i, item in enumerate(data)) / total_weight

            squared_deviations = [(item[value_key] - weighted_mean) ** 2 * weights[i] for i, item in enumerate(data)]
            mswd = 0 if len(data) <= 1 else sum(squared_deviations) / (len(data) - 1)
            standard_error = math.sqrt(1 / total_weight)

            return {'weighted_mean': weighted_mean, 'standard_error': standard_error, 'mswd': mswd, 'n': len(data)}

        current_data = filtered_data.copy()
        all_removed_items = []

        mswd_note = ""
        # Robust screen first (not masked by several outliers at once), then the
        # MSWD test on what remains.
        if self.use_nmad_var.get() and len(current_data) > 2:
            k = self.nmad_k_var.get()
            self.log(f"Applying NMAD rejection (k={k})...")
            nmad_results = self.apply_nmad(current_data, value_key, uncertainty_key, k=k)
            current_data = nmad_results['final_data']
            all_removed_items.extend(nmad_results['removed_items'])

        if self.use_mswd_var.get() and len(current_data) + len(protected) > 3:
            current_data, removed, mswd_note = self.sequential_mswd(
                current_data, value_key, uncertainty_key, protected=protected)
            all_removed_items.extend(removed)

        # Force-included points rejoin the kept set, whatever the algorithms decided.
        current_data = protected + current_data

        final_results = calculate_weighted_mean_mswd(current_data)

        if math.isnan(final_results['weighted_mean']) or final_results['weighted_mean'] == 0:
            final_percent_error_2sigma = float('nan')
            final_percent_error_1sigma = float('nan')
        else:
            final_percent_error_1sigma = (final_results['standard_error'] / final_results['weighted_mean']) * 100
            final_percent_error_2sigma = final_percent_error_1sigma * 2.0

        fe_se_1s = final_results['standard_error']

        mswd_val = final_results['mswd']
        n_final  = final_results['n']
        # √MSWD scaling only for significant overdispersion: MSWD above the
        # Wendt & Carl (1991) 95% limit for n, as in the Plateau & isochron tool.
        wc_limit = mswd_critical(n_final)
        overdispersed = _finite(mswd_val) and mswd_val > wc_limit
        od_se_1s = fe_se_1s * math.sqrt(mswd_val) if overdispersed else fe_se_1s

        use_reml = getattr(self, 'use_reml_var', None)
        run_reml = use_reml.get() if use_reml is not None else True

        if run_reml:
            reml_result = self.reml_random_effects(current_data, value_key, uncertainty_key)
        else:
            reml_result = None

        if reml_result:
            reml_mean  = reml_result['weighted_mean']
            reml_se_1s = reml_result['se_1sigma']
            reml_tau   = reml_result['tau']
            reml_tau2  = reml_result['tau2']
        else:
            reml_mean  = final_results['weighted_mean']
            reml_se_1s = fe_se_1s
            reml_tau   = 0.0
            reml_tau2  = 0.0

        p_value = self.mswd_p_value(mswd_val, n_final)

        plot_data = {
            'all_points': [], 'kept_points': [], 'removed_points': [],
            'weighted_mean':   final_results['weighted_mean'],
            'standard_error':  final_results['standard_error'],
            'mswd':           mswd_val,
            'wc_limit':       wc_limit,
            'overdispersed':  overdispersed,
            'mswd_note':      mswd_note,
            'fe_se_1s':       fe_se_1s,
            'od_se_1s':       od_se_1s,
            'reml_mean':      reml_mean,
            'reml_se_1s':     reml_se_1s,
            'reml_tau':       reml_tau,
            'reml_tau2':      reml_tau2,
            'run_reml':       run_reml,
            'p_value':        p_value,
            'n_final':        n_final,
        }

        high_error_ids = [item['sample_id'] for item in data_with_rel_err
                       if apply_error_filter and item['relative_error'] > error_limit]
        outlier_ids   = [item['sample_id'] for item in all_removed_items]
        manual_ids    = set((manual_exclusions or {}).keys())

        source_data = all_data if all_data is not None else original_data

        for item in source_data:
            if not _finite(item.get(value_key)):
                continue
            unc = item.get(uncertainty_key)
            unc = float(unc) if _finite(unc) else float('nan')
            point_data = {
                'sample_id':    item['sample_id'],
                'value':       item[value_key],
                'uncertainty': unc,
                'relative_error': (unc / abs(item[value_key]) * 100) if item[value_key] != 0 else float('inf'),
                'power':       item.get('power'),
                'label':       item.get('label'),
                'ar39':        item.get('ar39'),
                'grain_id':    item.get('grain_id'),
                'step_type':   item.get('step_type'),
                'step_index':  item.get('step_index'),
                'n_steps':     item.get('n_steps'),
                'crystal_n_steps': item.get('crystal_n_steps'),
                'pool_method': item.get('pool_method'),
                'source':      item.get('source'),
            }

            if item['sample_id'] in invalid_ids:
                point_data['status'] = 'invalid'
                plot_data['removed_points'].append(point_data)
            elif item['sample_id'] in manual_ids:
                reason = (manual_exclusions or {}).get(item['sample_id'], "")
                point_data['status'] = 'manual'
                point_data['reason'] = reason
                plot_data['removed_points'].append(point_data)
            elif item['sample_id'] in incl_ids:
                point_data['status'] = 'included'   # force-kept by the user
                plot_data['kept_points'].append(point_data)
            elif item['sample_id'] in high_error_ids:
                point_data['status'] = 'high_error'
                plot_data['removed_points'].append(point_data)
            elif item['sample_id'] in outlier_ids:
                point_data['status'] = 'outlier'
                plot_data['removed_points'].append(point_data)
            else:
                point_data['status'] = 'kept'
                plot_data['kept_points'].append(point_data)

            plot_data['all_points'].append(point_data)

        n_manual = len(manual_ids & {it['sample_id'] for it in source_data})
        self.log(f"Final: {len(current_data)}/{len(category_data)} points, "
                f"Mean: {final_results['weighted_mean']:.6e}, MSWD: {final_results['mswd']:.3f}"
                + (f", {n_manual} manually excluded" if n_manual else ""))

        return {
            'category': category_name, 'initial_points': total_points, 'final_points': len(current_data),
            'weighted_mean': final_results['weighted_mean'], 'standard_error': final_results['standard_error'],
            'percent_error_2sigma': final_percent_error_2sigma, 'percent_error_1sigma': final_percent_error_1sigma,
            'mswd': final_results['mswd'], 'plot_data': plot_data,
            'fe_se_1s':   fe_se_1s,
            'od_se_1s':   od_se_1s,
            'reml_mean':  reml_mean,
            'reml_se_1s': reml_se_1s,
            'reml_tau':   reml_tau,
            'reml_tau2':  reml_tau2,
            'p_value':    p_value,
            'wc_limit':   wc_limit,
            'overdispersed': overdispersed,
        }

    def sequential_mswd(self, data, value_key, uncertainty_key, protected=()):
        """MSWD-based outlier rejection, one point at a time.

        Each round, while the MSWD of the current set exceeds the threshold
        (the Wendt & Carl 1991 limit for the current n, or a fixed value):

          • every candidate is compared with the weighted mean of all the
            other points (its externally studentized residual);
          • with the discordance test (default), the residual is
                rᵢ = (xᵢ − μ̂₍₋ᵢ₎) / √(σᵢ² + τ̂²₍₋ᵢ₎ + SE²₍₋ᵢ₎)
            with μ̂, τ̂² and SE from a REML fit to the others (the random-
            effects residual of Viechtbauer & Cheung 2010).  The largest is
            removed only if |rᵢ| exceeds the two-sided 5 % critical value
            Bonferroni-corrected for n points, Φ⁻¹(1 − 0.025/n) (2.81 at n = 10,
            3.02 at n = 20).  Otherwise the round stops: the remaining scatter
            is overdispersion spread over the whole set, not a few outliers,
            and it is carried by the REML / √MSWD uncertainty instead;
          • without the test, the point with the largest fixed-effect residual
            (xᵢ − μ̂₍₋ᵢ₎)/√(σᵢ² + SE²₍₋ᵢ₎) is removed: its square is exactly
            the drop in χ² when it is left out.  This continues until the
            MSWD is reached, and trims genuine scatter when there is some.

        Force-included points count in the MSWD and in the means but are
        never removed.  At least three points are kept.
        """
        from statistics import NormalDist
        cur = list(data)
        prot = list(protected)
        removed = []
        use_wc = self.mswd_rule_var.get() == 'wc'
        fixed = float(self.mswd_threshold_var.get())
        test = bool(self.mswd_discordant_var.get())
        rule = "Wendt & Carl limit" if use_wc else f"fixed {fixed:g}"
        self.log(f"MSWD rejection (stop at MSWD ≤ {rule}; "
                 f"{'discordance test 5 % Bonferroni' if test else 'no discordance test'}):")

        def wm(vals, sigs):
            w = [1.0 / s ** 2 for s in sigs]
            W = sum(w)
            m = sum(wi * v for wi, v in zip(w, vals)) / W
            chi2 = sum(wi * (v - m) ** 2 for wi, v in zip(w, vals))
            return m, W, chi2

        note = ""          # one line for the plot title / export
        while True:
            pool = prot + cur
            n = len(pool)
            vals = [float(it[value_key]) for it in pool]
            sigs = [float(it[uncertainty_key]) for it in pool]
            _, _, chi2 = wm(vals, sigs)
            mswd = chi2 / (n - 1) if n > 1 else 0.0
            thr = mswd_critical(n) if use_wc else fixed
            if mswd <= thr:
                self.log(f"  MSWD {mswd:.3f} ≤ {thr:.3f} (n = {n}): done, "
                         f"{len(removed)} removed")
                note = (f"MSWD rejection: {len(removed)} removed, MSWD {mswd:.2f} ≤ {thr:.2f}"
                        if removed else "")
                break
            if len(cur) == 0 or n <= 3:
                self.log(f"  MSWD {mswd:.3f} > {thr:.3f} but no point can be removed "
                         f"(n = {n}); stopped")
                note = f"MSWD rejection stopped at {n} points (minimum 3)"
                break
            best_k, best_r = None, -1.0
            for k in range(len(prot), n):          # candidates only
                ov = vals[:k] + vals[k + 1:]
                os_ = sigs[:k] + sigs[k + 1:]
                if test:
                    rr = self.reml_random_effects(
                        [{'v': v, 'e': s} for v, s in zip(ov, os_)], 'v', 'e')
                    mu, se, tau2 = rr['weighted_mean'], rr['se_1sigma'], rr['tau2']
                    den = math.sqrt(sigs[k] ** 2 + tau2 + se ** 2)
                else:
                    mu, W, _ = wm(ov, os_)
                    den = math.sqrt(sigs[k] ** 2 + 1.0 / W)
                r = abs(vals[k] - mu) / den
                if r > best_r:
                    best_k, best_r = k, r
            item = pool[best_k]
            if test:
                crit = NormalDist().inv_cdf(1.0 - 0.025 / n)
                if best_r <= crit:
                    self.log(f"  MSWD {mswd:.3f} > {thr:.3f} (n = {n}), but no point is "
                             f"discordant with the others: largest studentized residual "
                             f"{best_r:.2f} ({item.get('sample_id', '')}) ≤ {crit:.2f}. The "
                             f"remaining scatter is overdispersion, not outliers; it is "
                             f"carried by the REML / √MSWD uncertainty.")
                    note = ("MSWD rejection: " + (f"{len(removed)} removed, then " if removed else "")
                            + f"no point discordant (max |r| {best_r:.2f} ≤ {crit:.2f}): "
                            f"the scatter is overdispersion, not outliers")
                    break
                why = f"|r| = {best_r:.2f} > {crit:.2f}"
            else:
                why = f"|r| = {best_r:.2f}, χ² drop {best_r ** 2:.2f}"
            cur.remove(item)
            removed.append(item)
            self.log(f"  removing {item.get('sample_id', '')} ({why}; MSWD was {mswd:.3f} "
                     f"> {thr:.3f}, n = {n})")
        return cur, removed, note

    # ---------------------------------------------------------------------
    # Statistics: random-effects model and p-value
    # ---------------------------------------------------------------------

    def reml_random_effects(self, data, value_key, uncertainty_key):
        """
        REML (restricted maximum likelihood) random-effects model.

        y_i ~ N(μ, σ_i² + τ²).  τ² maximises the restricted log-likelihood

            ℓ_R(τ²) = −½ [ Σ log(σ_i² + τ²) + log W + Σ w_i (y_i − μ̂)² ]

        with w_i = 1/(σ_i² + τ²), W = Σ w_i, μ̂ = Σ w_i y_i / W.  The log W
        term is what distinguishes REML from ML: it accounts for μ being
        estimated from the same data, so τ² is not biased low for small n
        (n < 20 is typical for Fish Canyon disk positions).  With equal σ it
        gives τ² = s² − σ² (s² the sample variance), where ML gives
        (n−1)/n · s² − σ².

        τ² comes from arar_math.ml_overdispersion with reml=True, the same
        likelihood fit the other tools use for overdispersion (with ML there).

        The SE is the usual plug-in √(1/Σw*), which does not include the
        uncertainty of τ² itself.

        Returns 1σ uncertainties throughout.
        """
        n = len(data)
        if n < 2:
            return None

        sigmas_in = [float(item[uncertainty_key]) for item in data]
        values_in = [float(item[value_key]) for item in data]
        tau2 = ml_overdispersion(values_in, sigmas_in, reml=True)
        tau = math.sqrt(tau2)
        w   = [1.0 / (s**2 + tau2) for s in sigmas_in]
        W   = sum(w)
        mu  = sum(w[i] * values_in[i] for i in range(n)) / W
        se  = math.sqrt(1.0 / W)

        return {
            'weighted_mean': mu,
            'se_1sigma':    se,
            'tau2':         tau2,
            'tau':          tau,
            'n':            n
        }

    def mswd_p_value(self, mswd, n):
        """
        Upper-tail p-value for MSWD under H0: no overdispersion.
        Uses chi-squared distribution with (n-1) degrees of freedom.
        P = P(χ²_{n-1} > MSWD × (n-1)). Small p (< 0.05) means the
        observed scatter significantly exceeds what the stated errors predict.
        """
        if n <= 1 or math.isnan(mswd):
            return float('nan')
        df   = n - 1
        chi2 = mswd * df
        return self._chi2_sf(chi2, df)

    def _chi2_sf(self, x, df):
        """Survival function P(χ² > x) for chi-squared distribution."""
        return float(chi2_dist.sf(x, df))

    def j_from_r(self, r_value, r_error):
        """J = (e^{λ t_FCs} − 1) / R for a monitor ratio R, and its 1σ error.

        The ingrowth factor comes from arar_math.fc_ingrowth (the anchored chain
        R_FCs/A · (e^{λ t_A} − 1) for Kuiper, Rivera and Phillips), so this J is
        identical to the J every other tool computes from the same R.

        The error is the monitor measurement only, σ_J/J = σ_R/R.  The FCs
        calibration and λ are systematics shared by every J and every age on
        that calibration; folding them into each J would let a weighted mean
        average them down.  They belong in the age's external tier."""
        j_value = self._ingrowth() / r_value
        j_error = abs(j_value) * abs(r_error / r_value)
        return {'j_value': j_value, 'j_error': j_error}

    def _pt_err_mult(self):
        """Multiplier to convert stored 1σ to the display σ (1σ or 2σ)."""
        disp = getattr(self, 'display_sigma_var', None)
        return 2.0 if (disp and disp.get() == '2sigma') else 1.0

    @staticmethod
    def _natural_sort_key(text):
        """Sort key that orders strings with embedded numbers numerically."""
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split('([0-9]+)', str(text))]

    def create_outlier_plot(self, category, data_type):
        self._clear_plot()

        # Reset the click-to-toggle registry for this (re)draw.
        self._pick_pts = []
        self._pick_ax = None
        self._pick_category = category

        if not self.plot_data or category not in self.plot_data:
            self._plot_message("No plot data available")
            return

        cat_data = self.plot_data[category]
        if data_type == "40/39 Ratios" and '40_39_ratios' in cat_data:
            plot_data = cat_data['40_39_ratios']
            ylabel = "40/39 Ratio"
        elif data_type == "Calculated J" and 'measured_j' in cat_data:
            plot_data = cat_data['measured_j']
            ylabel = "Calculated J"
        elif data_type in cat_data:
            plot_data = cat_data[data_type]
            ylabel = data_type
        else:
            self._plot_message("No data for selected type")
            return

        if self._export_ax is not None:         # one panel of an export
            ax = self._export_ax
            fig = ax.figure
        else:
            fig = self._new_figure((10, 6))
            ax  = fig.add_subplot(111)

        nan = float('nan')
        all_points     = plot_data.get('all_points', [])
        kept_points    = plot_data.get('kept_points', [])
        weighted_mean  = plot_data.get('weighted_mean', nan)
        standard_error = plot_data.get('standard_error', nan)
        mswd           = plot_data.get('mswd', nan)

        fe_se_1s   = plot_data.get('fe_se_1s',   standard_error)
        od_se_1s   = plot_data.get('od_se_1s',   fe_se_1s)
        reml_mean  = plot_data.get('reml_mean',  weighted_mean)
        reml_se_1s = plot_data.get('reml_se_1s', fe_se_1s)
        reml_tau   = plot_data.get('reml_tau',   0.0)
        run_reml   = plot_data.get('run_reml',   False)
        p_value    = plot_data.get('p_value',    nan)

        mult = self._pt_err_mult()
        sig  = "2σ" if mult == 2.0 else "1σ"

        if not all_points:
            ax.text(0.5, 0.5, 'No data points', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{category}: {data_type}")
        else:
            x_of = {p['sample_id']: i + 1 for i, p in enumerate(all_points)}

            # Register every plotted point for click-to-toggle (x, value, sample_id).
            self._pick_ax = ax
            self._pick_category = category
            self._pick_pts = [(x_of[p['sample_id']], p['value'], p['sample_id'])
                              for p in all_points]

            # Circle = single-step (fusion) crystal, triangle = a step of a
            # step-heated crystal, square = a crystal's pooled total gas.
            MK = {'fusion': 'o', 'incremental': '^', 'pooled': 's'}
            has_crystals = any(p.get('step_type') in ('incremental', 'pooled')
                               for p in all_points)
            kind = {'o': 'Fusion (1-step crystals)' if has_crystals else 'Analyses',
                    '^': 'Incremental steps', 's': 'Crystal total gas'}

            def _marker_for(point):
                return MK.get(point.get('step_type'), 'o')

            def _grain_label(point):
                """Crystal ID, unless the point is its own crystal (no ID given)."""
                gid = str(point.get('grain_id') or '').strip()
                return gid if gid and gid != str(point.get('sample_id')) else ''

            def _draw(pts, color, edge, label, size=5, z=3, ann_color='dimgray',
                      per_kind=False):
                first = True
                for fmt in ('o', '^', 's'):
                    sel = [p for p in pts if _marker_for(p) == fmt]
                    if not sel:
                        continue
                    xs   = [x_of[p['sample_id']] for p in sel]
                    ys   = [p['value'] for p in sel]
                    yerr = [p['uncertainty'] * mult if _finite(p['uncertainty']) else 0.0
                            for p in sel]
                    if per_kind:
                        lbl = f"{kind[fmt]} (n={len(sel)}) [±{sig}]"
                    else:
                        lbl = f"{label} (n={len(pts)})" if first else None
                    ax.errorbar(xs, ys, yerr=yerr, fmt=fmt, color=color, capsize=3,
                                markersize=size if fmt == 'o' else size + 1,
                                markeredgecolor=edge, markeredgewidth=1,
                                label=lbl, zorder=z)
                    first = False
                    for x, p in zip(xs, sel):
                        gid = _grain_label(p) if self._export_opts.get('ids', True) else ''
                        if gid:
                            ax.annotate(gid, (x, p['value']), fontsize=6, ha='left',
                                        va='center', xytext=(5, 0),
                                        textcoords='offset points', color=ann_color)

            by_status = lambda *st: [p for p in all_points if p.get('status') in st]
            _draw(by_status('manual'), 'orange', 'darkorange', 'Excluded',
                  ann_color='darkorange')
            _draw(by_status('high_error', 'outlier', 'invalid'), 'red', 'darkred',
                  'Rejected', size=4, ann_color='darkred')
            _draw(by_status('included'), '#1f77b4', 'navy', 'Force-included',
                  size=6, z=4, ann_color='navy')
            _draw(by_status('kept'), 'green', 'darkgreen', None, per_kind=True)

            if _finite(weighted_mean):

                def fmt_val(v):
                    return f"{v:.3e}" if (abs(v) < 0.001 or abs(v) > 10000) else f"{v:.6f}"

                def pct(se, mean):
                    return f"{se / abs(mean) * 100:.4f}%" if mean != 0 else "N/A"

                x_span = [0.5, len(all_points) + 0.5]

                def band(m, se, color, ls, alpha, label):
                    ax.fill_between(x_span, m - mult * se, m + mult * se,
                                    alpha=alpha, color=color, label=label)
                    ax.axhline(y=m + mult * se, color=color, linestyle=ls, alpha=0.7, linewidth=1.2)
                    ax.axhline(y=m - mult * se, color=color, linestyle=ls, alpha=0.7, linewidth=1.2)

                # ── Mean line ─────────────────────────────────────────────
                if run_reml:
                    ax.axhline(y=reml_mean, color='black', linewidth=2,
                               label=f'REML mean = {fmt_val(reml_mean)}')
                else:
                    ax.axhline(y=weighted_mean, color='black', linewidth=2,
                               label=f'Weighted mean = {fmt_val(weighted_mean)}')

                if run_reml:
                    tau_str = f", τ = {reml_tau:.3e}" if reml_tau > 0 else ", τ = 0 (= weighted mean)"
                    band(reml_mean, reml_se_1s, 'blue', '-.', 0.20,
                         f'REML ±{sig} SE = {fmt_val(mult * reml_se_1s)} '
                         f'({pct(mult * reml_se_1s, reml_mean)}){tau_str}')
                else:
                    band(weighted_mean, fe_se_1s, 'gray', '--', 0.15,
                         f'±{sig} SE = {fmt_val(mult * fe_se_1s)} '
                         f'({pct(mult * fe_se_1s, weighted_mean)})')
                    if plot_data.get('overdispersed'):
                        band(weighted_mean, od_se_1s, 'orange', ':', 0.15,
                             f'±{sig} SE × √MSWD (MSWD > W&C) = {fmt_val(mult * od_se_1s)} '
                             f'({pct(mult * od_se_1s, weighted_mean)})')

            ax.set_xlabel('Point number (file order)')
            ax.set_ylabel(ylabel)

            # Title: n/N, MSWD, p-value
            final_n   = len(kept_points)
            initial_n = len(all_points)
            # Overdispersion judged by the Wendt & Carl (1991) limit for n.
            wc = plot_data.get('wc_limit', float('nan'))
            p_str = f",  p = {p_value:.4f}" if _finite(p_value) else ""
            if _finite(wc):
                sig_str = (f",  above W&C limit {wc:.2f}" if plot_data.get('overdispersed')
                           else f",  within W&C limit {wc:.2f}")
            else:
                sig_str = ""
            mswd_str = f"{mswd:.3f}" if _finite(mswd) else "n/a"
            basis = self.CRYSTAL_MODES.get(self.crystal_mode_var.get(), '')
            note = plot_data.get('mswd_note') or ''
            ax.set_title(f"{category}: {data_type}   [{basis}]\n"
                         f"({final_n}/{initial_n} points,  MSWD = {mswd_str}{p_str}{sig_str})"
                         + (f"\n{note}" if note else ""),
                         fontsize=11 if note else 12)

            if self.zoom_mode_var.get() == "adaptive" and kept_points:
                kv = [(p['value'], p['uncertainty'] * mult) for p in kept_points
                      if _finite(p['value']) and _finite(p['uncertainty'])]
                if kv:
                    y_lo = min(v - e for v, e in kv)
                    y_hi = max(v + e for v, e in kv)
                    pad  = (y_hi - y_lo) * 0.25 or abs(y_lo) * 0.005 or 1.0
                    ax.set_ylim(y_lo - pad, y_hi + pad)

            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        canvas = self._show_figure(fig)
        # Click a point to cycle it default → exclude → force-include → default.
        if canvas is not None:
            canvas.mpl_connect('button_press_event', self._on_plot_click)

    def _on_plot_click(self, event):
        """Click the nearest data point on the outlier plot to cycle its state."""
        ax  = getattr(self, '_pick_ax', None)
        pts = getattr(self, '_pick_pts', None)
        if ax is None or not pts or event.inaxes is not ax or event.xdata is None:
            return
        if getattr(self, '_toolbar', None) is not None and self._toolbar.mode:
            return                        # zooming or panning, not picking
        trans = ax.transData.transform
        best, best_d = None, 1e9
        for (x, y, sid) in pts:
            px, py = trans((x, y))
            d = math.hypot(px - event.x, py - event.y)
            if d < best_d:
                best_d, best = d, sid
        if best is None or best_d > 16:   # pixel radius
            return
        self._cycle_point_state(self._pick_category, best)
        self._apply_exclusions()          # rerun + redraw plot + refresh table

    def _on_plot_category_change(self, event=None):
        """Keep exclusion tab in sync when plot category changes."""
        cat = self.category_var.get()
        if hasattr(self, 'excl_cat_combo') and cat:
            self.excl_cat_var.set(cat)
            self._populate_exclusion_table()
        self.update_plot()

    def _on_excl_category_change(self, event=None):
        """Keep plot tab in sync when exclusion category changes."""
        cat = self.excl_cat_var.get()
        if hasattr(self, 'category_combo') and cat:
            self.category_var.set(cat)
            self.update_plot()
        self._populate_exclusion_table()

    def _plot_all_groups(self, data_type):
        """Overview: every group's own weighted mean ± SE on one axis, to compare
        the groups. No mean or statistics across the groups are computed."""
        self._clear_plot()
        # This view has no clickable points.
        self._pick_pts = []
        self._pick_ax = None

        results = getattr(self, 'results', None)
        if not results:
            self._plot_message("No results yet: run the analysis first.")
            return

        if data_type == "Calculated J":
            res_map = results.get('measured_j_results', {})
            xlabel  = "Calculated J"
        else:
            res_map = results.get('ratio4039_results', {})
            xlabel  = "40/39 Ratio" if self.value_is_ratio else data_type

        mult      = self._pt_err_mult()
        sig_lbl   = "2σ" if mult == 2.0 else "1σ"
        reml_on   = getattr(self, 'use_reml_var', None)
        use_reml  = reml_on.get() if reml_on is not None else True

        rows = []   # (group, mean, se_display, n, mswd)
        for cat, r in res_map.items():
            wm = r.get('weighted_mean')
            if wm is None or (isinstance(wm, float) and math.isnan(wm)):
                continue
            reml_mean = r.get('reml_mean')
            if use_reml and reml_mean is not None and not math.isnan(reml_mean):
                m, se1 = reml_mean, r.get('reml_se_1s', r.get('standard_error'))
            else:
                m, se1 = wm, r.get('od_se_1s', r.get('standard_error'))
            pdat = r.get('plot_data', {})
            rows.append((cat, m, (se1 or 0.0) * mult,
                         pdat.get('n_final'), r.get('mswd', float('nan'))))

        if not rows:
            self._plot_message("No group means to show.")
            return

        rows.sort(key=lambda t: t[1])   # order groups by their mean value
        tier_lbl = "REML" if use_reml else "WM, SE × √MSWD above the W&C limit"

        fig = self._new_figure((9, max(3.0, 0.45 * len(rows) + 1.6)))
        ax  = fig.add_subplot(111)
        ys  = list(range(len(rows)))
        for y, (cat, m, se, n, mswd) in zip(ys, rows):
            ax.errorbar(m, y, xerr=se, fmt='o', color='#1f77b4', capsize=4,
                        markersize=6, markeredgecolor='navy', markeredgewidth=1.0)
            note = []
            if n is not None:
                note.append(f"n={n}")
            if mswd is not None and not math.isnan(mswd):
                note.append(f"MSWD={mswd:.2f}")
            if note:
                ax.annotate("   ".join(note), (m, y), fontsize=7, color='dimgray',
                            xytext=(8, 7), textcoords='offset points')

        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel(f"{xlabel}   (error bars ±{sig_lbl}, {tier_lbl})")
        ax.set_title(f"All categories: {data_type}   ({len(rows)} categories)")
        ax.grid(True, axis='x', alpha=0.3)
        fig.tight_layout()

        self._show_figure(fig)

    def update_plot(self, event=None):
        data_type = self.data_type_var.get()
        crystal_view = getattr(self, 'crystal_view_var', None)
        view_mode    = crystal_view.get() if crystal_view else 'all'

        # "All groups" spans every category, so it doesn't need one selected.
        if view_mode == 'groups':
            self._plot_all_groups(data_type or '40/39 Ratios')
            return

        category = self.category_var.get()
        if not category or not data_type:
            return
        self._draw_category(view_mode, category, data_type)

    def _draw_category(self, view, category, data_type):
        """One category in the 'all' (points), 'crystal' or 'power' view."""
        if view == 'crystal' and self.raw_category_data.get(category):
            self._plot_by_crystal(category, data_type)
        elif view == 'power' and self.raw_category_data.get(category):
            self._plot_by_power_step(category, data_type)
        else:
            self.create_outlier_plot(category, data_type)

    # ---------------------------------------------------------------------
    # Crystal-view plots
    # ---------------------------------------------------------------------

    def _view_transform(self, data_type):
        """(f, ylabel) for the crystal views: f maps (R, σR) to what is
        plotted, J = (e^{λt} − 1)/R for 'Calculated J', R itself otherwise."""
        if self.value_is_ratio and data_type == "Calculated J":
            c = self._ingrowth()

            def f(v, s):
                if not (_finite(v) and float(v) > 0):
                    return float('nan'), float('nan')
                j = c / float(v)
                return j, (j * float(s) / float(v) if _finite(s) else float('nan'))
            return f, "Calculated J"
        return (lambda v, s: (v, s)), ("40/39 Ratio" if self.value_is_ratio else data_type)

    def _no_view(self, text):
        self._clear_plot()
        self._pick_pts = []
        self._pick_ax = None
        self._plot_message(text, size=11)

    def _category_reference(self, category, f):
        """The category's reported mean (REML or WM), in plotted units."""
        try:
            r = self.results['ratio4039_results'][category]
        except (KeyError, TypeError):
            return float('nan')
        m = r.get('reml_mean') if r.get('plot_data', {}).get('run_reml') else r.get('weighted_mean')
        return f(m, 0.0)[0] if _finite(m) else float('nan')

    def _plot_by_crystal(self, category, data_type):
        """One panel per step-heated crystal: its steps (hollow grey =
        excluded), the crystal's total gas ± SE (what 'One point per crystal'
        averages) and the category mean for reference.  Single-step crystals
        share one last panel."""
        raw = self.raw_category_data.get(category, [])
        if not raw:
            self._no_view("No data yet: run the analysis first.")
            return
        tagged = self._parse_crystal_structure(raw)
        crystals = OrderedDict()
        for t in tagged:
            crystals.setdefault(t['grain_id'], []).append(t)
        multi = [(g, s) for g, s in crystals.items() if len(s) > 1]
        singles = [s[0] for s in crystals.values() if len(s) == 1]
        if not multi:
            self._no_view(f"{category}: every crystal is a single-step (fusion) analysis, so "
                          f"there are no steps to show per crystal. Use 'All points'.")
            return

        self._clear_plot()
        self._pick_pts = []
        self._pick_ax = None

        f, ylabel = self._view_transform(data_type)
        mult = self._pt_err_mult()
        sig = "2σ" if mult == 2.0 else "1σ"
        excl = self.manual_exclusions.get(category, {})
        cstats = self.crystal_stats.get(category) or {}
        MAXP = 30
        panels = multi[:MAXP] + ([('1-step crystals', singles)] if singles else [])

        lim = self.error_threshold_var.get() if self.use_error_filter_var.get() else None
        full = self.zoom_mode_var.get() == 'full'

        def in_view(t):
            if not self._valid_row(t):
                return False
            if full:
                return True
            if t['sample_id'] in excl:
                return False
            v, e = float(t['value40_39']), float(t['uncertainty40_39'])
            return lim is None or (v != 0 and e / abs(v) * 100.0 <= lim)

        ys = []
        for _, steps in panels:
            for t in steps:
                if in_view(t):
                    y, s = f(t['value40_39'], t['uncertainty40_39'])
                    if _finite(y) and _finite(s):
                        ys += [y - mult * s, y + mult * s]
        y_lim = None
        if ys:
            lo, hi = min(ys), max(ys)
            pad = (hi - lo) * 0.12 or abs(lo) * 0.005 or 1.0
            y_lim = (lo - pad, hi + pad)
        ref = self._category_reference(category, f)

        n = len(panels)
        ncols = self._grid_cols(n, 5)
        nrows = math.ceil(n / ncols)
        fig = self._new_figure((max(10, ncols * 2.8), max(4, nrows * 3.0)), dpi=90)
        for idx, (gid, steps) in enumerate(panels):
            ax = fig.add_subplot(nrows, ncols, idx + 1)
            for i, t in enumerate(steps, start=1):
                if not self._valid_row(t):
                    continue
                y, s = f(t['value40_39'], t['uncertainty40_39'])
                if not _finite(y):
                    continue
                out = t['sample_id'] in excl
                ax.errorbar(i, y, yerr=mult * s if _finite(s) else 0.0, fmt='o',
                            color='#b0b0b0' if out else '#1f77b4',
                            mfc='white' if out else '#1f77b4', capsize=2, markersize=4)
            if gid == '1-step crystals':
                ax.set_title(f"1-step crystals (n={len(steps)})", fontsize=8)
                ticks = []
            else:
                c = cstats.get(gid)
                if c:
                    y, s = f(c['tg'], c['tg_se'])
                    if _finite(y):
                        ax.axhline(y, color='black', linewidth=1.3)
                        if _finite(s):
                            ax.axhspan(y - mult * s, y + mult * s, color='#ff7f0e', alpha=0.25)
                    mswd = f"{c['mswd']:.2f}" if _finite(c['mswd']) else "–"
                    ax.set_title(f"{gid}\n{c['n_used']}/{c['n_steps']} steps, MSWD {mswd}",
                                 fontsize=8)
                else:
                    ax.set_title(f"{gid}\n(all steps excluded)", fontsize=8)
                ticks = [self._step_text(t) for t in steps]
            if _finite(ref):
                ax.axhline(ref, color='green', linestyle='--', linewidth=0.9)
            ax.set_xticks(range(1, len(steps) + 1))
            ax.set_xticklabels(ticks if len(steps) <= 14 else [], fontsize=6, rotation=45)
            if y_lim:
                ax.set_ylim(y_lim)
            ax.set_ylabel(ylabel if idx % ncols == 0 else "", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        more = (f"  (first {MAXP} of {len(multi)} step-heated crystals)"
                if len(multi) > MAXP else "")
        self._wrapped_suptitle(
            fig, f"{category}: {ylabel} by crystal{more}",
            f"black line + orange band: the crystal's total gas ±{sig} (its point in "
            f"'one point per crystal'); dashed green: category mean; "
            f"hollow grey: excluded steps; x labels: laser power")
        self._show_figure(fig)

    def _plot_by_power_step(self, category, data_type):
        """One panel per step group (laser power or step number; 1-step
        crystals form their own group) with the group mean ± SE, and a last
        panel comparing the group means with the between-group MSWD and p."""
        stats = self.step_group_stats.get(category)
        if not stats or not stats.get('groups'):
            self._no_view(f"{category}: no step-heated crystals, so there are no step groups "
                          f"to compare. Step groups need crystals measured in more than one step.")
            return
        self._clear_plot()
        self._pick_pts = []
        self._pick_ax = None

        f, ylabel = self._view_transform(data_type)
        mult = self._pt_err_mult()
        sig = "2σ" if mult == 2.0 else "1σ"
        tagged = self._parse_crystal_structure(self.raw_category_data.get(category, []))
        by_id = {t['sample_id']: t for t in tagged}
        groups = stats['groups']
        n_g = len(groups)
        npan = n_g + (1 if n_g >= 2 else 0)
        ncols = self._grid_cols(npan, 6)
        nrows = math.ceil(npan / ncols)
        fig = self._new_figure((max(10, ncols * 2.6), max(4.5, nrows * 3.4)), dpi=90)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        ys = []
        for st in groups.values():
            for sid in st['ids']:
                t = by_id.get(sid)
                if t:
                    y, s = f(t['value40_39'], t['uncertainty40_39'])
                    if _finite(y) and _finite(s):
                        ys += [y - mult * s, y + mult * s]
        y_lim = None
        if ys:
            lo, hi = min(ys), max(ys)
            pad = (hi - lo) * 0.12 or abs(lo) * 0.005 or 1.0
            y_lim = (lo - pad, hi + pad)

        for gi, (lab, st) in enumerate(groups.items()):
            ax = fig.add_subplot(nrows, ncols, gi + 1)
            col = colors[gi % len(colors)]
            for k, sid in enumerate(st['ids'], start=1):
                t = by_id.get(sid)
                if not t:
                    continue
                y, s = f(t['value40_39'], t['uncertainty40_39'])
                ax.errorbar(k, y, yerr=mult * s if _finite(s) else 0.0, fmt='o',
                            color=col, capsize=2, markersize=4)
            m, se = f(st['best_mean'], st['best_se'])
            if _finite(m):
                ax.axhline(m, color='black', linewidth=1.3)
                if _finite(se):
                    ax.axhspan(m - mult * se, m + mult * se, color=col, alpha=0.2)
            mswd = f"{st['mswd']:.2f}" if _finite(st['mswd']) else "–"
            ax.set_title(f"{lab}  (n={st['n']})\nMSWD {mswd}", fontsize=8)
            ax.set_xticks([])
            ax.set_xlabel("crystals", fontsize=7)
            if y_lim:
                ax.set_ylim(y_lim)
            ax.set_ylabel(ylabel if gi % ncols == 0 else "", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        b = stats.get('between')
        if n_g >= 2:
            ax = fig.add_subplot(nrows, ncols, npan)
            for gi, (lab, st) in enumerate(groups.items()):
                m, se = f(st['best_mean'], st['best_se'])
                if _finite(m):
                    ax.errorbar(gi + 1, m, yerr=mult * se if _finite(se) else 0.0, fmt='s',
                                color=colors[gi % len(colors)], capsize=3, markersize=5)
            if b:
                m, _ = f(b['mean'], b['se'])
                if _finite(m):
                    ax.axhline(m, color='black', linestyle='--', linewidth=0.9)
            ax.set_xticks(range(1, n_g + 1))
            ax.set_xticklabels(list(groups), fontsize=6, rotation=45)
            btxt = (f"MSWD {b['mswd']:.2f}, p {b['p']:.3f}" if b and _finite(b['mswd'])
                    else "")
            ax.set_title(f"Group means (±{sig})\n{btxt}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        self._wrapped_suptitle(
            fig, f"{category}: {ylabel} by step group ({stats['by']})",
            f"black line + band: group mean ±{sig}; steps after manual exclusions "
            f"and the relative-error filter, no outlier rejection")
        self._show_figure(fig)

    # ---------------------------------------------------------------------
    # Ar-Ar session files (.arar / .ararpt / .ararstack)
    # ---------------------------------------------------------------------

    def _read_arar_files(self, paths, crystal_rule='auto', group_by='position'):
        merged = OrderedDict()
        info_all = dict(sessions=0, steps=0, skipped=0, calibrations=set())
        for p in paths:
            cats, info = read_arar_rows(p, crystal_rule, group_by)
            for k in ('sessions', 'steps', 'skipped'):
                info_all[k] += info[k]
            info_all['calibrations'] |= info['calibrations']
            for c, rows in cats.items():
                merged.setdefault(c, []).extend(rows)
        for c in merged:
            _uniquify_ids(merged[c])
        return merged, info_all

    def _seed_file_exclusions(self, data):
        """Steps a file excludes come in as manual exclusions (visible and
        reversible), never silently dropped."""
        n = 0
        for cat, rows in data.items():
            for r in rows:
                if r.get('file_reason'):
                    self.manual_exclusions.setdefault(cat, {})[r['sample_id']] = r['file_reason']
                    n += 1
        return n

    def _arar_options_dialog(self, paths):
        """Grouping and crystal rule for the chosen files, with a live preview.
        Returns dict(crystal_rule, group_by) or None if cancelled."""
        result = [None]
        dlg = QDialog(self)
        dlg.setWindowTitle("Ar-Ar import")
        lay = QVBoxLayout(dlg)

        names = [os.path.basename(p) for p in paths]
        title = QLabel(names[0] if len(names) == 1 else f"{len(names)} files")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lay.addWidget(title)
        lay.addWidget(_note("R = ⁴⁰Ar*/³⁹ArK of every step, errors brought to 1σ as the "
                            "Plateau & isochron and Age distribution tools read them."))

        group_var = Value('position')
        rule_var = Value('auto')
        gbox = QGroupBox("One category per")
        gl = QVBoxLayout(gbox)
        rb_pos = QRadioButton("Irradiation position (irradiation label, else hole)")
        rb_file = QRadioButton("File / experiment")
        gl.addWidget(rb_pos)
        gl.addWidget(rb_file)
        bind_radios(group_var, [(rb_pos, 'position'), (rb_file, 'file')])
        lay.addWidget(gbox)
        rbox = QGroupBox("Crystals within a file")
        rl = QVBoxLayout(rbox)
        pairs = []
        for key, text in ARAR_CRYSTAL_RULES.items():
            rb = QRadioButton(text)
            rl.addWidget(rb)
            pairs.append((rb, key))
        bind_radios(rule_var, pairs)
        lay.addWidget(rbox)

        preview = QPlainTextEdit()
        preview.setReadOnly(True)
        preview.setFont(mono_font(9))
        preview.setLineWrapMode(QPlainTextEdit.NoWrap)
        preview.setMinimumSize(640, 180)
        lay.addWidget(preview, 1)

        def refresh(*_):
            lines = []
            try:
                data, info = self._read_arar_files(paths, rule_var.get(), group_var.get())
                lines.append(f"{'category':<24}{'steps':>6}{'crystals':>10}"
                             f"{'step-heated':>13}{'excl. in file':>15}")
                for cat, rows in data.items():
                    cnt = Counter(r['grain_id'] for r in rows)
                    lines.append(f"{cat[:23]:<24}{len(rows):>6}{len(cnt):>10}"
                                 f"{sum(1 for v in cnt.values() if v > 1):>13}"
                                 f"{sum(1 for r in rows if r.get('file_reason')):>15}")
                if info['skipped']:
                    lines.append(f"\n{info['skipped']} step(s) without R or its error skipped.")
            except Exception as e:
                lines.append(f"Could not read the files:\n{e}")
            preview.setPlainText("\n".join(lines))

        group_var.on_change(refresh)
        rule_var.on_change(refresh)
        refresh()

        def ok():
            result[0] = dict(crystal_rule=rule_var.get(), group_by=group_var.get())
            dlg.accept()

        lay.addLayout(_ok_cancel(dlg, "Load", ok))
        dlg.exec_()
        dlg.deleteLater()
        return result[0]

    def browse_arar_files(self):
        paths, _ = dfold.get_open_file_names(
            self, "Select Ar-Ar session files", "",
            "Ar-Ar sessions (*.arar *.ararpt *.ararstack);;All files (*)")
        if not paths:
            return
        paths = list(paths)
        opts = self._arar_options_dialog(paths)
        if opts is None:
            self.format_info_var.set("Import cancelled")
            return
        try:
            data, info = self._read_arar_files(paths, **opts)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Could not read the Ar-Ar files:\n{e}")
            return
        if not data:
            QMessageBox.critical(self, "Load error", "No steps with R and its error in these files.")
            return

        self.arar_paths, self.arar_opts = paths, opts
        self.file_source = 'arar'
        self.loaded_source = 'arar'
        self.loaded_df, self.loaded_col_map = None, {}
        self.input_file = paths[0]
        self.input_name = (os.path.basename(paths[0]) if len(paths) == 1
                           else f"{len(paths)} Ar-Ar files")
        self.value_is_ratio = self.file_value_is_ratio = True
        self.value_label = self.file_value_label = "40/39 Ratio"
        self._reset_overrides()
        n_excl = self._seed_file_exclusions(data)

        self.raw_file_var.set("; ".join(os.path.basename(p) for p in paths))
        n_steps = sum(len(v) for v in data.values())
        n_cryst = sum(len({r['grain_id'] for r in v}) for v in data.values())
        self.format_info_var.set(f"Ready: {n_steps} steps, {n_cryst} crystals, "
                                 f"{len(data)} categories ({n_excl} steps excluded in the files)")
        self.log(f"Loaded {self.input_name}: {n_steps} steps, {n_cryst} crystals, "
                 f"{len(data)} categories; {n_excl} steps excluded in the files come in as "
                 f"manual exclusions (Manual exclusion tab).")
        if info['calibrations']:
            cal = ", ".join(f"FC {t:g} Ma / λ {l:.4e}" for t, l in sorted(info['calibrations']))
            self.log(f"Calibration stored in the files: {cal}. J here uses the Standard Age & "
                     f"Decay Constant panel ({self.calib_var.get()}); R does not depend on it.")
        self.analyze_btn.setEnabled(True)

    def analyze_data(self):
        """Analyse the current data with the current options: the file(s) last
        loaded, or the Direct Input data when that was analysed last (manual
        exclusions are kept either way)."""
        if not self._options_valid(show=True):
            return
        if self.loaded_source in ('direct', 'session') and self.raw_category_data:
            what = 'Direct input' if self.loaded_source == 'direct' else 'session'
            try:
                self._run_analysis(f'=== RE-ANALYSIS of the {what} data (exclusions kept) ===')
                self.log('\nAnalysis complete.')
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error analysing data: {str(e)}")
            return

        source = getattr(self, 'file_source', None)
        if source not in ('table', 'arar'):
            QMessageBox.critical(self, "Error", "Please import a file first")
            return

        try:
            self.analyze_btn.setEnabled(False)

            if source == 'arar':
                category_data, _info = self._read_arar_files(self.arar_paths, **self.arar_opts)
            else:
                if self.loaded_df is None or not self.loaded_col_map:
                    QMessageBox.critical(self, "Error", "No data loaded. Please import a file first.")
                    return
                category_data = self._build_category_data_from_df(
                    self.loaded_df, self.loaded_col_map)
            if not category_data:
                QMessageBox.critical(self, "Error", "No valid data found. Check column mapping.")
                return

            if self.loaded_source != source:
                # back from Direct Input: its exclusions do not belong to this file
                self._reset_overrides()
                if source == 'arar':
                    self._seed_file_exclusions(category_data)
            self.loaded_source  = source
            self.value_is_ratio = self.file_value_is_ratio
            self.value_label    = self.file_value_label
            self.input_name     = (os.path.basename(self.input_file) if source == 'table'
                                   else self.input_name)
            self.raw_category_data = category_data
            self._run_analysis('=== WEIGHTED MEAN ANALYSIS ===')
            self.log('\nAnalysis complete.')

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analysing data: {str(e)}")
        finally:
            self.analyze_btn.setEnabled(True)

    def closeEvent(self, event):
        release_figures(self)       # before Qt deletes the canvases (macOS)
        super().closeEvent(event)

    # ---------------------------------------------------------------------
    # Sessions
    # ---------------------------------------------------------------------

    _SESSION_FILTER = 'Flux monitors session (*.json);;All (*)'
    _SESSION_FORMAT = 'SimplAr flux monitors 1'

    def _session_vars(self):
        """The options a session keeps, by name."""
        return dict(
            error_filter=self.use_error_filter_var, error_limit=self.error_threshold_var,
            nmad=self.use_nmad_var, nmad_k=self.nmad_k_var,
            mswd=self.use_mswd_var, mswd_rule=self.mswd_rule_var,
            mswd_limit=self.mswd_threshold_var, mswd_discordant=self.mswd_discordant_var,
            reml=self.use_reml_var, display_sigma=self.display_sigma_var,
            crystal_mode=self.crystal_mode_var, step_groups=self.step_group_var,
            import_sigma=self.import_sigma_var)

    def save_session(self):
        """Save session…: the data as analysed, the options, the calibration and
        the manual exclusions, so the analysis can be reopened as it is."""
        if not self.raw_category_data or not self.results:
            QMessageBox.information(self, "Save session",
                                    "Nothing to save yet: analyse the data first.")
            return
        path, _ = dfold.get_save_file_name(self, "Save session", "flux_monitors_session.json",
                                           self._SESSION_FILTER)
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += '.json'
        src = (self.session_source if self.loaded_source == 'session' else self.loaded_source)
        st = dict(
            format=self._SESSION_FORMAT,
            data=self.raw_category_data,
            source=src, input_name=self.input_name,
            value_is_ratio=self.value_is_ratio, value_label=self.value_label,
            arar_files=[os.path.basename(p) for p in self.arar_paths] if src == 'arar' else [],
            arar_opts=self.arar_opts if src == 'arar' else {},
            exclusions=self.manual_exclusions,
            inclusions={c: sorted(v) for c, v in self.manual_inclusions.items()},
            options={k: v.get() for k, v in self._session_vars().items()},
            calibration=dict(name=self.calib_var.get(), std_age=self.std_age_var.get(),
                             std_age_err=self.std_age_err_var.get(),
                             lam=self.lambda_var.get(), lam_err=self.lambda_err_var.get(),
                             monitor=self.monitor_combo.keys()['monitor'], route=self._route),
            view=dict(category=self.category_var.get(), data_type=self.data_type_var.get(),
                      view=self._view_label_var.get(), zoom=self.zoom_mode_var.get()),
            direct_input=self.direct_input.state() if src == 'direct' else None)

        def plain(v):                     # numpy numbers from a pandas table
            return v.item() if hasattr(v, 'item') else str(v)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(st, f, indent=1, ensure_ascii=False, default=plain)
        except OSError as e:
            QMessageBox.warning(self, "Save session", f"Could not write the file:\n{e}")
            return
        self.log(f"Session saved: {path}")

    def load_session(self):
        """Load session…: reopen a saved analysis with its options, calibration
        and manual exclusions."""
        path, _ = dfold.get_open_file_name(self, "Load session", "", self._SESSION_FILTER)
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                st = json.load(f)
            if st.get('format') != self._SESSION_FORMAT or not st.get('data'):
                raise ValueError('not a Flux monitors session')
            data = {str(c): list(rows) for c, rows in st['data'].items()}
        except (OSError, ValueError, AttributeError, TypeError) as e:
            QMessageBox.warning(self, "Load session", f"Could not read the session:\n{e}")
            return
        opts = st.get('options') or {}
        for k, var in self._session_vars().items():
            if k in opts:
                var.set(opts[k])
        cal = st.get('calibration') or {}
        self._applying_preset = True          # the fields as saved, not a preset's
        try:
            if cal.get('name'):
                self.calib_var.set(cal['name'])
            for k, var in (('std_age', self.std_age_var), ('std_age_err', self.std_age_err_var),
                           ('lam', self.lambda_var), ('lam_err', self.lambda_err_var)):
                if k in cal:
                    var.set(cal[k])
        finally:
            self._applying_preset = False
        self.monitor_combo.set_monitor(cal.get('monitor') or 'FCs')
        self._route = cal.get('route')
        self._show_route()
        if st.get('direct_input'):
            self.direct_input.set_state(st['direct_input'])

        self.file_source = None               # the file tab's file is not this data
        self.session_source = st.get('source')
        self.arar_paths = list(st.get('arar_files') or [])
        self.arar_opts = dict(st.get('arar_opts') or {})
        self._set_loaded_data(data, 'direct' if st.get('source') == 'direct' else 'session',
                              st.get('value_is_ratio', True), st.get('value_label'),
                              st.get('input_name'))
        self.manual_exclusions = {c: dict(v) for c, v in (st.get('exclusions') or {}).items()}
        self.manual_inclusions = {c: set(v) for c, v in (st.get('inclusions') or {}).items()}
        view = st.get('view') or {}
        self.data_type_var.set(view.get('data_type') or self.data_type_var.get())
        self.category_var.set(view.get('category') or '')
        if view.get('view') in self._VIEWS:
            self._view_label_var.set(view['view'])
            self.crystal_view_var.set(self._VIEWS[view['view']])
        self.zoom_mode_var.set(view.get('zoom') or 'adaptive')
        self._rerun_timer.stop()
        if not self._options_valid(show=True):
            return
        try:
            self._run_analysis(f"=== SESSION: {os.path.basename(path)} ===")
        except Exception as e:
            QMessageBox.critical(self, "Load session", f"Error analysing the data: {e}")
            return
        self._rerun_timer.stop()
        n = sum(len(v) for v in data.values())
        self._set_current_data(f"session {os.path.basename(path)}, {n} points in "
                               f"{len(data)} groups")

    # ---------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------

    _STATUS_LABEL = {
        'kept':       'Included',
        'included':   'Force-included',
        'high_error': 'Rejected: relative error above limit',
        'outlier':    'Rejected: outlier',
        'invalid':    'Rejected: unusable (no error, R ≤ 0 or ³⁹Ar ≤ 0)',
        'manual':     'Excluded: manual',
    }

    def export_results(self):
        """Export XLSX / CSV…: the workbook has the results, the constants and
        notes laid out as in the other tools' exports, and a sheet each for the
        points, steps, crystals and step groups; a CSV holds the constants, the
        results table and the analysis settings."""
        if not self.results:
            QMessageBox.warning(self, "Export", "No results to export: analyse the data first.")
            return
        path, flt = dfold.get_save_file_name(self, "Export results", "flux_monitors.xlsx",
                                             export.save_filter())
        if not path:
            return
        path = export.ensure_ext(path, flt)
        try:
            if path.lower().endswith('.xlsx'):
                self._write_results_xlsx(path)
            else:
                self._write_results_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Could not write the file:\n{e}")
            return
        QMessageBox.information(self, "Export", f"Saved:\n{path}")
        self.log(f'Exported: {path}')

    def _summary_rows(self):
        """(header, rows) of the Results sheet, one row per category."""
        nan = float('nan')
        R = self.results['ratio4039_results']
        cats = sorted(R, key=self._natural_sort_key)
        ratio = self.value_is_ratio
        vn = "R" if ratio else (self.value_label or "Value")
        basis = self.CRYSTAL_MODES.get(self.crystal_mode_var.get(), '')
        c = self._ingrowth() if ratio else None
        header = ['Category', 'Points averaged', 'n used', 'n total', 'MSWD', 'p',
                  'MSWD 95% limit (W&C)', 'Overdispersed (MSWD > W&C)',
                  f'{vn} weighted mean', 'SE (1σ)', 'SE, × √MSWD if overdispersed (1σ)',
                  f'{vn} REML mean', 'REML SE (1σ)', 'τ (1σ)',
                  f'Reported {vn}', 'Reported SE (1σ)', 'Reported SE (2σ)', 'Method',
                  'MSWD rejection']
        mon = self._monitor_cfg()
        # an FCs-equivalent R only where the route goes through FCs
        fcs_eq = (ratio and mon['monitor'] != 'FCs'
                  and (mon.get('route') or {}).get('standard', 'FCs') == 'FCs')
        if ratio:
            header += ['J', 'J SE (1σ, R only)', 'J SE (2σ)', 'J % (1σ)']
        if fcs_eq:
            header += ['Reported R (FCs-equivalent)', 'Reported SE (FCs-equivalent, 1σ)']
        rows = []
        for cat in cats:
            r = R[cat]
            pdat = r.get('plot_data') or {}
            n = r.get('final_points', 0)
            mswd = r.get('mswd', nan)
            run_reml = bool(pdat.get('run_reml'))
            wm, se, od = r.get('weighted_mean', nan), r.get('fe_se_1s', nan), r.get('od_se_1s', nan)
            rm, rse, tau = ((r.get('reml_mean', nan), r.get('reml_se_1s', nan), r.get('reml_tau', nan))
                            if run_reml else (nan, nan, nan))
            if run_reml and _finite(rm):
                rep, rep_se, method = rm, rse, 'REML random effects'
            else:
                rep, rep_se = wm, od
                method = 'WM + √MSWD (W&C)' if pdat.get('overdispersed') else 'WM'
            wc = mswd_critical(n) if n > 1 else nan
            row = [cat, basis, n, r.get('initial_points'), mswd, r.get('p_value', nan), wc,
                   'yes' if pdat.get('overdispersed') else ('no' if n > 1 else ''),
                   wm, se, od, rm, rse, tau, rep, rep_se,
                   2.0 * rep_se if _finite(rep_se) else nan, method,
                   pdat.get('mswd_note', '')]
            if ratio:
                if _finite(rep) and rep > 0 and _finite(rep_se):
                    J = c / rep
                    s_j = J * rep_se / rep
                    row += [J, s_j, 2.0 * s_j, 100.0 * s_j / J]
                else:
                    row += [nan, nan, nan, nan]
            if fcs_eq:
                k = mon['monitorR']
                row += [rep / k if _finite(rep) else nan, rep_se / k if _finite(rep_se) else nan]
            rows.append(row)
        return header, rows

    def _parameter_lines(self):
        crit = ([f"Data: {self.input_name or os.path.basename(self.input_file or '') or 'Unknown'}"])
        src = (getattr(self, 'session_source', None) if self.loaded_source == 'session'
               else self.loaded_source)
        if src == 'arar':
            crit.append(f"Ar-Ar files: {', '.join(os.path.basename(p) for p in self.arar_paths)}")
            crit.append(f"Categories: {'irradiation position' if self.arar_opts.get('group_by') == 'position' else 'file'};"
                        f" crystals: {ARAR_CRYSTAL_RULES.get(self.arar_opts.get('crystal_rule'), '')}")
            crit.append("Step errors read at 1σ; steps excluded in the files loaded as "
                        "manual exclusions")
        elif src == 'table':
            sm = getattr(self, 'import_sigma_var', None)
            crit.append(f"Errors in file: {'±2σ (halved)' if sm and sm.get() == '2sigma' else '±1σ'}")
        if self.value_is_ratio:
            crit.append(f"Calibration for J: {self.calib_var.get()}  FC {self.std_age_var.get()} "
                        f"± {self.std_age_err_var.get()} Ma (1σ), λ {self.lambda_var.get()} "
                        f"± {self.lambda_err_var.get()} /yr (1σ)")
            mon = self._monitor_cfg()
            if mon.get('route'):
                crit.append(f"Calibration route: {route_text(mon)}")
            crit += [f"Note: {n}" for n in calibration_notes(mon)]
            if edited_standards_note():
                crit.append(edited_standards_note())
            if mon['monitor'] != 'FCs' and not mon.get('route'):
                crit.append(f"Monitor: {mon['monitor']}, R({mon['monitor']}/FCs) = "
                            f"{mon['monitorR']:.8g} ± {mon['monitorRErr']:.3g} (1σ); "
                            f"{monitor_text(mon)}. R (FCs-equivalent) = measured R / "
                            f"R({mon['monitor']}/FCs)")
        crit += [
            f"Points averaged: {self.CRYSTAL_MODES.get(self.crystal_mode_var.get(), '')}",
            f"Step groups by: {'laser power' if self.step_group_var.get() == 'power' else 'step number'}",
            f"Statistical model: {'REML random effects' if self.use_reml_var.get() else 'weighted mean, SE × √MSWD when MSWD exceeds the Wendt & Carl limit'}",
            f"Relative error filter: {'> ' + str(self.error_threshold_var.get()) + '% (1σ) rejected' if self.use_error_filter_var.get() else 'off'}",
            f"NMAD rejection: {'k = ' + str(self.nmad_k_var.get()) if self.use_nmad_var.get() else 'off'}",
            ("MSWD rejection: one point at a time until MSWD ≤ "
             + ('the Wendt & Carl limit' if self.mswd_rule_var.get() == 'wc'
                else str(self.mswd_threshold_var.get()))
             + ('; a point removed only if discordant with the others (random-effects '
                'studentized residual, 5 % Bonferroni)' if self.mswd_discordant_var.get()
                else '; no discordance test')
             if self.use_mswd_var.get() else "MSWD rejection: off"),
            "Order of rejection: relative-error filter, NMAD, MSWD.",
        ]
        return crit

    def _notes_rows(self, sheets=()):
        """Rows of the Notes sheet: what each sheet holds, then how the
        numbers are reported."""
        about = {
            'Results': 'One row per category: the weighted mean and REML mean of its points, '
                       'the value reported' + (' and its J' if self.value_is_ratio else '')
                       + '. Below the table, how the data were analysed.',
            'Constants': 'The calibration J is calculated on.',
            'Points': 'Every point averaged in each category, with its status and the reason '
                      'for any rejection or exclusion.',
            'Steps': 'Every measured step, when the points averaged are not the steps.',
            'Crystals': 'The total gas of each step-heated crystal and the statistics of its steps.',
            'Step groups': 'The steps grouped by laser power or step number, and the comparison '
                           'of the group means.',
        }
        rows = [['ABOUT THIS FILE']]
        rows += [[f'Sheet "{sh}"', about[sh]] for sh in sheets if sh in about]
        rows += [
            [],
            ['CONVENTIONS'],
            ['Uncertainties', 'SE values are 1σ unless the column says 2σ.'],
            ['Reported', 'The REML random-effects mean and SE when REML is ticked, else the '
                         'weighted mean with SE × √MSWD when the MSWD exceeds the Wendt & Carl '
                         'limit.'],
            ['MSWD 95% limit', 'Wendt & Carl (1991), 1 + 2√(2/(n−1)); above it the scatter is '
                               'significant overdispersion and only then is the SE multiplied '
                               'by √MSWD.'],
        ]
        if self.value_is_ratio:
            rows.append(['J', 'J = (e^(λt) − 1)/R of the reported R. Its error is from R only: the '
                              'FC age and λ errors are shared by every J on this calibration and '
                              'belong in the external tier of the ages, not here.'])
        if self.crystal_stats or self.crystal_mode_var.get() == 'crystal':
            rows.append(['Total gas', 'Of a step-heated crystal: R = Σ fᵢRᵢ, fᵢ = ³⁹Ar fraction, '
                                      'σ = √Σ fᵢ²σᵢ²; without ³⁹Ar the inverse-variance mean of '
                                      'its steps.'])
        return rows

    def _constants_rows(self):
        """The constants block shared with the other tools' exports (R mode)."""
        if not self.value_is_ratio:
            return []
        return [['CONSTANTS'], ['Constant', 'Value', 'Note']] + \
            export.constants_rows(self._monitor_cfg())

    def _write_results_csv(self, path):
        import csv
        header, rows = self._summary_rows()
        with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)              # constants first, as in the other tools
            if self.value_is_ratio:
                w.writerows(self._constants_rows())
                w.writerow([])
            w.writerow(['FLUX MONITOR RESULTS'])
            w.writerow(header)
            for r in rows:
                w.writerow(['' if isinstance(v, float) and not math.isfinite(v) else v
                            for v in r])
            w.writerow([])
            w.writerow(['ANALYSIS PARAMETERS'])
            for line in self._parameter_lines():
                w.writerow([line])

    def _write_results_xlsx(self, path):
        from openpyxl import Workbook
        from openpyxl.styles import Font

        nan = float('nan')
        bold = Font(bold=True)

        def table(ws, start, header, rows, widths=None):
            for ci, h in enumerate(header, 1):
                ws.cell(row=start, column=ci, value=h).font = bold
            for ri, row in enumerate(rows):
                for ci, v in enumerate(row, 1):
                    if isinstance(v, float) and not math.isfinite(v):
                        v = None
                    ws.cell(row=start + 1 + ri, column=ci, value=v)
            for ci in range(1, len(header) + 1):
                w = (widths or {}).get(ci, 16)
                ws.column_dimensions[ws.cell(row=start, column=ci).column_letter].width = w
            return start + 1 + len(rows)

        wb = Workbook()
        ws = wb.active
        ws.title = "Results"
        ws.cell(row=1, column=1, value="FLUX MONITOR RESULTS").font = bold
        ws.cell(row=2, column=1, value=f"Run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        header, rows = self._summary_rows()
        end = table(ws, 4, header, rows, widths={1: 18, 2: 30, 8: 14, 11: 20, 18: 20, 19: 50})
        r0 = end + 1
        ws.cell(row=r0, column=1, value="ANALYSIS PARAMETERS").font = bold
        for i, line in enumerate(self._parameter_lines(), start=1):
            ws.cell(row=r0 + i, column=1, value=line)
        ws.freeze_panes = "B5"
        if self.value_is_ratio:
            export.write_sheet(wb.create_sheet("Constants"), self._constants_rows(), bold)

        # Points: what was averaged in each category, with its status
        ratio = self.value_is_ratio
        vn = "R" if ratio else (self.value_label or "Value")
        c = self._ingrowth() if ratio else None
        ph = ['Category', 'Crystal', 'Point ID', 'Type', 'Step', '³⁹Ar', vn, f'{vn} σ (1σ)',
              'Rel. error (%)']
        if ratio:
            ph += ['J', 'J σ (1σ)']
        ph += ['Status', 'Reason']
        prow = []
        tname = {'fusion': 'fusion (1 step)', 'incremental': 'step', 'pooled': 'crystal total gas'}
        for cat in sorted(self.results['ratio4039_results'], key=self._natural_sort_key):
            pdat = self.results['ratio4039_results'][cat].get('plot_data') or {}
            for p in pdat.get('all_points', []):
                st = p.get('status', 'kept')
                step = (f"Σ {p.get('n_steps')}/{p.get('crystal_n_steps')} steps"
                        if p.get('step_type') == 'pooled' else
                        (f"{float(p['power']):g}" if _finite(p.get('power')) else (p.get('label') or '')))
                row = [cat, p.get('grain_id') or '', p['sample_id'],
                       tname.get(p.get('step_type'), ''), step,
                       float(p['ar39']) if _finite(p.get('ar39')) else None,
                       float(p['value']), p.get('uncertainty', nan), p.get('relative_error', nan)]
                if ratio:
                    v = float(p['value'])
                    if v > 0 and _finite(p.get('uncertainty')):
                        J = c / v
                        row += [J, J * p['uncertainty'] / v]
                    else:
                        row += [nan, nan]
                row += [self._STATUS_LABEL.get(st, st), p.get('reason', '') or (p.get('pool_method') or '')]
                prow.append(row)
        ws2 = wb.create_sheet("Points")
        end = table(ws2, 1, ph, prow, widths={1: 16, 2: 20, 3: 18, 4: 16, 12: 30, 13: 36})
        ws2.freeze_panes = "A2"
        ws2.auto_filter.ref = f"A1:{ws2.cell(row=1, column=len(ph)).column_letter}{max(end - 1, 1)}"

        # Steps: every measured step, when the averaged points are not the steps
        if self.crystal_mode_var.get() != 'steps' or self.crystal_stats:
            sh = ['Category', 'Crystal', 'Step #', 'Steps in crystal', 'Step ID', 'Step', '³⁹Ar',
                  vn, f'{vn} σ (1σ)', 'Manual exclusion']
            srow = []
            for cat in sorted(self.raw_category_data, key=self._natural_sort_key):
                excl = self.manual_exclusions.get(cat, {})
                for t in self._parse_crystal_structure(self.raw_category_data[cat]):
                    reason = excl.get(t['sample_id'])
                    srow.append([cat, t['grain_id'], t['step_index'], t['crystal_n_steps'],
                                 t['sample_id'], self._step_text(t),
                                 float(t['ar39']) if _finite(t.get('ar39')) else None,
                                 t.get('value40_39'), t.get('uncertainty40_39'),
                                 ('excluded' + (f": {reason}" if reason else "")) if reason is not None else ''])
            ws3 = wb.create_sheet("Steps")
            end = table(ws3, 1, sh, srow, widths={1: 16, 2: 20, 5: 18, 10: 30})
            ws3.freeze_panes = "A2"

        if self.crystal_stats:
            ch = ['Category', 'Crystal', 'Steps used', 'Steps', f'Total gas {vn}', 'σ (1σ)',
                  'Method', f'Mean of steps {vn}', 'SE (1σ)', 'MSWD of steps', 'p']
            crow = []
            for cat in sorted(self.crystal_stats, key=self._natural_sort_key):
                for gid, s in self.crystal_stats[cat].items():
                    crow.append([cat, gid, s['n_used'], s['n_steps'], s['tg'], s['tg_se'],
                                 s['method'], s['wm'], s['wm_se'], s['mswd'], s['p']])
            ws4 = wb.create_sheet("Crystals")
            table(ws4, 1, ch, crow, widths={1: 16, 2: 20, 7: 34})
            ws4.freeze_panes = "A2"

        if self.step_group_stats:
            gh = ['Category', 'Grouped by', 'Group', 'n steps', 'n crystals', f'{vn} weighted mean',
                  'SE (1σ)', 'MSWD', 'p', 'REML mean', 'REML SE (1σ)', 'τ (1σ)',
                  f'Reported {vn}', 'Reported SE (1σ)', 'Method']
            grow = []
            for cat in sorted(self.step_group_stats, key=self._natural_sort_key):
                sg = self.step_group_stats[cat]
                for lab, s in sg['groups'].items():
                    grow.append([cat, sg['by'], lab, s['n'], s['n_crystals'], s['mean'], s['se'],
                                 s['mswd'], s['p'], s['reml_mean'], s['reml_se'], s['tau'],
                                 s['best_mean'], s['best_se'], s['method']])
                b = sg.get('between')
                if b:
                    grow.append([cat, sg['by'], 'BETWEEN GROUPS', len(sg['groups']), None,
                                 b['mean'], b['se'], b['mswd'], b['p'], None, None, None,
                                 None, None, 'WM of group means'])
            ws5 = wb.create_sheet("Step groups")
            table(ws5, 1, gh, grow, widths={1: 16, 2: 14, 3: 18, 15: 22})
            ws5.freeze_panes = "A2"

        export.write_sheet(wb.create_sheet("Notes"), self._notes_rows(wb.sheetnames), bold)
        wb.save(path)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Flux monitors")
    dfold.use_tool('flux_monitors')
    force_light(app)
    no_wheel_on_dropdowns(app)
    report_errors()
    win = FluxMonitorsWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
