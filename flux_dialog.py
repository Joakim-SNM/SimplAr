"""Shared disk-flux assignment dialog of the Plateau & isochron, Age distribution and
Recalculator and age combiner tools.

One window: load .jflux, view the disk (holes coloured by R or J, matched holes
ringed red), and assign each item (sample / session / aliquot) to a hole, either
automatically by identity, by a per-row override, or by selecting a disk and
applying it to the ticked rows. Every tool-specific behaviour lives in a small
host adapter (see FluxHost), so the tools share one window.

The dialog talks to its host through canonical override values:
  'auto'                → match by the item's own disk_id / hole_id
  'none'                → force no flux for this item
  (irr, disk, hole)     → pin to a specific loaded hole
The adapter translates those to and from whatever the host actually stores.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QComboBox,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QMessageBox, QApplication, QLineEdit,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from arar_math import render_disk, force_light, hole_candidates, mono_font


def flux_choices(disk_flux):
    """Dropdown choices: auto, none, then every loaded (disk, hole)."""
    out = [('auto (by identity)', 'auto'), ('none (independent)', 'none')]
    for (irr, disk), fx in disk_flux.items():
        for h in fx['holes']:
            out.append((f'{disk}/{h}  ({irr})', (irr, disk, h)))
    return out


def hole_in_disk(disk_id, hole_id, fx):
    """The hole on the selected disk matching this item's hole_id, resolved
    with arar_math.hole_candidates, the same rule as the automatic match."""
    for h in hole_candidates(disk_id, hole_id):
        if h in fx['idx']:
            return h
    return None


class FluxHost:
    """Adapter interface between a tool and the shared dialog. Subclass per tool
    and override the pieces that differ. Static labels are plain attributes;
    everything that reads or writes tool state is a method.
    """

    title = 'Load disk flux'
    item_noun = 'Item'          # column header + message wording ('Sample' etc.)
    caption = ''                # optional explanatory line above the table
    manual = False              # True: rows can be given a J or R typed by hand

    # --- registry / loading ------------------------------------------------
    def disk_flux(self):
        """The (irr, disk) -> flux-entry registry dict."""
        raise NotImplementedError

    def load(self, dialog):
        """Open a file chooser, parse .jflux files, and run any post-load work.
        The dialog refreshes its pickers and rows afterwards."""
        raise NotImplementedError

    def preferred_flux(self):
        """(irr, disk) to navigate to after a load, or None."""
        return None

    def on_open(self):
        """Called once when the dialog opens, before the rows are built."""

    def after_delete(self):
        """Called after a disk is deleted from the registry."""

    # --- items -------------------------------------------------------------
    def items(self):
        """The list of assignable items (samples / sessions / aliquots)."""
        raise NotImplementedError

    def item_label(self, item):
        return str(item)

    def item_disk_hole(self, item):
        """(disk_id, hole_id) as read from the item's own metadata."""
        return (None, None)

    def get_override(self, row, item):
        """Current assignment as 'auto' / 'none' / (irr, disk, hole)."""
        return 'auto'

    def set_override(self, row, item, val):
        """Store the assignment (val is 'auto' / 'none' / (irr, disk, hole))."""

    def on_row_change(self, row, item):
        """Optional per-row side effect after set_override (e.g. bake R)."""

    def match(self, row, item):
        """Resolve the item's current assignment to (key, idx), or None."""
        return None

    def hole_R(self, key, idx):
        """Monitor R (or J) the hole encodes, for the 'Matched R' cell."""
        return None

    # --- manual flux (hosts with manual = True) ------------------------------
    def get_manual(self, row, item):
        """('J' or 'R', value, 1σ) typed for this row, or None."""
        return None

    def set_manual(self, row, item, val):
        """Store ('J' or 'R', value, 1σ), or None to clear it."""

    # --- apply -------------------------------------------------------------
    def apply(self):
        """Re-age / recompute everything from the current assignments."""


class FluxAssignDialog(QDialog):
    """The shared window. All behaviour is delegated to `host` (a FluxHost)."""

    def __init__(self, parent, host):
        super().__init__(parent)
        self.host = host
        self.setWindowTitle(host.title)
        self.resize(1180, 640)
        force_light(QApplication.instance())
        host.on_open()

        root = QVBoxLayout(self)

        # top bar: import / delete / pickers
        top = QHBoxLayout()
        load_btn = QPushButton('Load .jflux…')
        load_btn.clicked.connect(self._load)
        top.addWidget(load_btn)
        del_btn = QPushButton('Delete disk')
        del_btn.clicked.connect(self._delete_flux)
        top.addWidget(del_btn)
        top.addSpacing(14)
        top.addWidget(QLabel('Irradiation:'))
        self.irr_combo = QComboBox()
        self.irr_combo.currentIndexChanged.connect(self._on_irr)
        top.addWidget(self.irr_combo)
        top.addWidget(QLabel('Disk:'))
        self.disk_combo = QComboBox()
        self.disk_combo.currentIndexChanged.connect(self._on_disk)
        top.addWidget(self.disk_combo)
        top.addStretch()
        root.addLayout(top)

        if host.caption:
            cap = QLabel(host.caption)
            cap.setWordWrap(True)
            cap.setStyleSheet('color:#555;')
            root.addWidget(cap)

        # main split: left disk view, right assignment table
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(4, 4), facecolor='white')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        lv.addWidget(self.canvas, 3)
        self.hole_tbl = QTableWidget(0, 3)
        self.hole_tbl.setHorizontalHeaderLabels(['Hole', 'R', '±R%'])
        self.hole_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hole_tbl.setFont(mono_font(9))
        lv.addWidget(self.hole_tbl, 2)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        how = QLabel(
            f'Pick each {host.item_noun.lower()}\'s hole in the "Hole" column, or tick '
            '"Use shown disk" for the rows that sit on the disk shown on the left. '
            'Apply & Close applies both. "auto" matches by the disk/hole in the file. '
            'A row on a hole is aged with that hole\'s J, correlated with the other '
            'holes through the flux. A row marked "not matched" (or set to "none") '
            'keeps its own R_std / J and is independent of the rows on the disk.')
        how.setWordWrap(True)
        rv.addWidget(how)
        heads = ['Use\nshown disk', host.item_noun, 'disk/hole\nin file', 'Hole', 'Matched R']
        if host.manual:
            heads += ['Manual', 'J or R', '±1σ']
        self.tbl = QTableWidget(0, len(heads))
        self.tbl.setHorizontalHeaderLabels(heads)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl.setFont(mono_font(9))
        rv.addWidget(self.tbl)
        split.addWidget(right)
        split.setSizes([400, 780])
        root.addWidget(split, 1)

        # bottom buttons
        bb = QHBoxLayout()
        auto_btn = QPushButton('Auto-match all')
        auto_btn.clicked.connect(self._auto_all)
        bb.addWidget(auto_btn)
        none_btn = QPushButton('No flux for all (independent)')
        none_btn.setToolTip('Unselect every hole: all rows are set to "none (independent)" '
                            'and keep their own R_std / J.')
        none_btn.clicked.connect(self._none_all)
        bb.addWidget(none_btn)
        applysel_btn = QPushButton('Assign ticked rows to shown disk')
        applysel_btn.clicked.connect(self._apply_selected_to_checked)
        bb.addWidget(applysel_btn)
        bb.addStretch()
        ok = QPushButton('Apply && Close')
        ok.clicked.connect(self._apply_close)
        bb.addWidget(ok)
        cancel = QPushButton('Close')
        cancel.setToolTip('Close, keeping the holes picked in the table (they apply as\n'
                          'you pick them).  Ticked rows are only assigned by Apply & Close.')
        cancel.clicked.connect(self.reject)
        bb.addWidget(cancel)
        root.addLayout(bb)

        self._row_combos = []
        self._row_checks = []
        self._refresh_pickers()
        self._populate()

    # ---- pickers / disk view ----------------------------------------------
    def _refresh_pickers(self):
        self.irr_combo.blockSignals(True)
        self.irr_combo.clear()
        self.irr_combo.addItems(sorted({k[0] for k in self.host.disk_flux()}))
        self.irr_combo.blockSignals(False)
        self._on_irr()

    def _on_irr(self):
        irr = self.irr_combo.currentText()
        self.disk_combo.blockSignals(True)
        self.disk_combo.clear()
        self.disk_combo.addItems(
            sorted({k[1] for k in self.host.disk_flux() if k[0] == irr}))
        self.disk_combo.blockSignals(False)
        self._on_disk()

    def _cur_key(self):
        return (self.irr_combo.currentText(), self.disk_combo.currentText())

    def _matched_idx_on(self, key):
        out = set()
        for row, item in enumerate(self.host.items()):
            m = self.host.match(row, item)
            if m is not None and m[0] == key:
                out.add(m[1])
        return out

    def _on_disk(self):
        key = self._cur_key()
        self.ax = render_disk(self.fig, self.hole_tbl,
                              self.host.disk_flux().get(key), key,
                              self._matched_idx_on(key))
        self.canvas.draw_idle()

    # ---- assignment table --------------------------------------------------
    def _ro(self, txt):
        it = QTableWidgetItem(txt)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        return it

    def _populate(self):
        choices = flux_choices(self.host.disk_flux())
        items = self.host.items()
        self.tbl.setRowCount(len(items))
        self._row_combos = []
        self._row_checks = []
        for row, item in enumerate(items):
            disk_id, hole_id = self.host.item_disk_hole(item)
            ident = f'{disk_id or "?"}/{hole_id or "?"}'
            self.tbl.setItem(row, 1, self._ro(self.host.item_label(item)))
            self.tbl.setItem(row, 2, self._ro(ident))
            cb = QComboBox()
            for label, _v in choices:
                cb.addItem(label)
            cb.setCurrentIndex(self._choice_index(choices,
                                                  self.host.get_override(row, item)))
            cb.currentIndexChanged.connect(lambda _i, r=row: self._on_assign(r))
            self._row_combos.append((cb, choices))
            self.tbl.setCellWidget(row, 3, cb)
            chk = QCheckBox()
            self._row_checks.append(chk)
            self.tbl.setCellWidget(row, 0, chk)
            if self.host.manual:
                self._manual_cells(row, item)
            self._update_match_cell(row)
        self.tbl.resizeColumnsToContents()

    def _manual_cells(self, row, item):
        """Kind, value and 1σ editors of a row's hand-typed J or R."""
        cur = self.host.get_manual(row, item)
        kind = QComboBox()
        kind.addItems(['—', 'J', 'R'])
        kind.setCurrentIndex({'J': 1, 'R': 2}.get(cur[0], 0) if cur else 0)
        val = QLineEdit('' if not cur else f'{cur[1]:.8g}')
        err = QLineEdit('' if not cur else f'{cur[2]:.8g}')
        for w in (val, err):
            w.setFixedWidth(90)
            w.editingFinished.connect(lambda r=row: self._on_manual(r))
        kind.currentIndexChanged.connect(lambda _i, r=row: self._on_manual(r))
        self.tbl.setCellWidget(row, 5, kind)
        self.tbl.setCellWidget(row, 6, val)
        self.tbl.setCellWidget(row, 7, err)

    def _on_manual(self, row):
        item = self.host.items()[row]
        kind = self.tbl.cellWidget(row, 5).currentText()

        def num(c):
            try:
                return float(self.tbl.cellWidget(row, c).text().replace(',', '.'))
            except ValueError:
                return None
        v, e = num(6), num(7)
        if kind in ('J', 'R') and v and v > 0 and e is not None and e >= 0:
            self.host.set_manual(row, item, (kind, v, e))
        else:
            self.host.set_manual(row, item, None)
        self.host.on_row_change(row, item)
        self._update_match_cell(row)

    @staticmethod
    def _choice_index(choices, ov):
        """Row index of the dropdown entry matching the stored override."""
        for k, (_label, v) in enumerate(choices):
            if v == ov:
                return k
            if (isinstance(v, tuple) and isinstance(ov, tuple)
                    and tuple(map(str, v)) == tuple(map(str, ov))):
                return k
        return 0

    def _update_match_cell(self, row):
        item = self.host.items()[row]
        m = self.host.match(row, item)
        if m is not None:
            key, i = m
            R = self.host.hole_R(key, i)
            fx = self.host.disk_flux().get(key)
            if R is not None:
                txt = f'{R:.6g}'
            elif fx is not None:
                txt = f'{fx["J"][i]:.8g}'
            else:
                txt = '—'
            self.tbl.setItem(row, 4, self._ro(txt))
        elif self.host.manual and self.host.get_manual(row, item):
            kind, v, e = self.host.get_manual(row, item)
            self.tbl.setItem(row, 4, self._ro(f'manual {kind}'))
        else:
            it = self._ro('not matched')
            it.setForeground(QColor('#c0392b'))
            it.setToolTip('No flux hole: this row keeps its own R_std / J and is '
                          'independent of the rows on the disk. Pick a '
                          'hole in the "Hole" column, or tick "Use shown disk" and '
                          'press Apply & Close.')
            self.tbl.setItem(row, 4, it)

    def _on_assign(self, row):
        cb, choices = self._row_combos[row]
        _label, val = choices[cb.currentIndex()]
        item = self.host.items()[row]
        self.host.set_override(row, item, val)
        self.host.on_row_change(row, item)
        self._update_match_cell(row)
        self._on_disk()

    def _auto_all(self):
        items = self.host.items()
        for row, (cb, _c) in enumerate(self._row_combos):
            cb.blockSignals(True)
            cb.setCurrentIndex(0)
            cb.blockSignals(False)
            item = items[row]
            self.host.set_override(row, item, 'auto')
            self.host.on_row_change(row, item)
            self._update_match_cell(row)
        self._on_disk()

    def _none_all(self):
        items = self.host.items()
        for row, (cb, choices) in enumerate(self._row_combos):
            k = next((i for i, (_l, v) in enumerate(choices) if v == 'none'), 1)
            cb.blockSignals(True)
            cb.setCurrentIndex(k)
            cb.blockSignals(False)
            item = items[row]
            self.host.set_override(row, item, 'none')
            self.host.on_row_change(row, item)
            self._update_match_cell(row)
        self._on_disk()

    def _assign_ticked(self):
        """Assign every ticked row to its hole on the disk shown on the left.
        Returns (n_ticked, n_assigned, labels of ticked rows with no such hole),
        or None when no disk is loaded."""
        key = self._cur_key()
        fx = self.host.disk_flux().get(key)
        if fx is None:
            return None
        irr, disk = key
        n_ok = n_checked = 0
        misses = []
        for row, item in enumerate(self.host.items()):
            if not self._row_checks[row].isChecked():
                continue
            n_checked += 1
            disk_id, hole_id = self.host.item_disk_hole(item)
            hole = hole_in_disk(disk_id, hole_id, fx)
            if hole is None:
                misses.append(self.host.item_label(item))
                continue
            self.host.set_override(row, item, (irr, disk, hole))
            self.host.on_row_change(row, item)
            n_ok += 1
        return n_checked, n_ok, misses

    def _miss_text(self, misses):
        irr, disk = self._cur_key()
        noun = self.host.item_noun.lower()
        return (f'{len(misses)} ticked {noun}(s) have no matching hole on {disk} '
                f'({irr}):\n  • ' + '\n  • '.join(misses))

    def _apply_selected_to_checked(self):
        res = self._assign_ticked()
        if res is None:
            QMessageBox.warning(self, 'Apply selected', 'Select a loaded disk flux first.')
            return
        n_checked, n_ok, misses = res
        n_miss = len(misses)
        irr, disk = self._cur_key()
        if n_checked == 0:
            QMessageBox.information(
                self, 'Apply selected',
                f'Tick the {self.host.item_noun.lower()}s that belong to this disk first.')
            return
        self.host.apply()
        self._populate()
        msg = f'Assigned {n_ok} {self.host.item_noun.lower()}(s) to {disk}  ({irr}).'
        if n_miss:
            msg += '\n' + self._miss_text(misses)
        QMessageBox.information(self, 'Apply selected', msg)

    # ---- load / delete / close --------------------------------------------
    def _load(self):
        self.host.load(self)
        self._refresh_pickers()
        pref = self.host.preferred_flux()
        if pref:
            i = self.irr_combo.findText(pref[0])
            if i >= 0:
                self.irr_combo.setCurrentIndex(i)
                j = self.disk_combo.findText(pref[1])
                if j >= 0:
                    self.disk_combo.setCurrentIndex(j)
        self._populate()

    def _delete_flux(self):
        key = self._cur_key()
        if not key[0]:
            return
        ans = QMessageBox.question(
            self, 'Delete disk flux',
            f'Remove disk "{key[1]}" ({key[0]}) from the flux registry?',
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self.host.disk_flux().pop(key, None)
        self.host.after_delete()
        self._refresh_pickers()
        self._populate()

    def reject(self):
        # Hole picks take effect as they are made (_on_assign), so closing
        # finishes them like Apply does (the stack regroups), rather than
        # leaving a half-applied state behind.
        self.host.apply()
        super().reject()

    def _apply_close(self):
        # Ticked rows go to the disk shown, as with "Assign ticked rows".
        res = self._assign_ticked()
        misses = res[2] if res else []
        self.host.apply()
        if misses:
            self._populate()
            self._on_disk()
            QMessageBox.warning(self, 'Apply', self._miss_text(misses)
                                + '\n\nThe other rows are applied. Pick a hole for '
                                'these in the "Hole" column, or choose another disk.')
            return
        self.accept()
