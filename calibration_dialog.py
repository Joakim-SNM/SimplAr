"""The Calibration dialog shared by the tools.

Monitor and calibration on top (the routes of the standards library); under
Advanced a typed monitor age and λ, the constants the stored ages and J were
calculated with, which samples the calibration goes to, and a way into
Standards to build another route.  The tool applies what get_params returns.
"""

import copy

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QGroupBox, QComboBox,
                             QCheckBox, QLabel, QLineEdit, QPushButton, QWidget,
                             QDialogButtonBox)

from arar_math import (route_cfg, route_from_cfg, route_text, route_options, custom_route,
                       monitor_ingrowth, monitor_age, ROUTE_KEYS, STANDARD_IDS, STANDARDS,
                       reload_standards, cfg_monitor, calibration_notes,
                       LAMBDA_DEFAULT, LAMBDA_ERR)


def _note(text):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet('color:#555; font-size:10px;')
    return lab


class CalibrationDialog(QDialog):
    """Age source and calibration of the loaded data.

    The calibration is a route from the standards library: the monitor and a
    calibration it can reach (routes saved in Standards first).  Advanced
    holds what most people never need: a typed monitor age and λ, the
    constants the loaded data were calculated with, and a link to Standards
    to build another route.  On accept the caller recasts the stored ages and
    J from the old monitor ingrowth to the new one.

    REVERT = 42, result code for the "Revert to original" button.
    """

    REVERT = 42

    def __init__(self, old_cfg, parent=None, can_revert=False, old_mode='ratio_R_std',
                 n_aliquots=1, origin='', age_source=True, stored=True, samples=None,
                 full_external=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.setMinimumWidth(560)
        reload_standards()            # routes saved in Standards since the tool started
        self._old = dict(old_cfg)
        layout = QVBoxLayout(self)

        src_box = QGroupBox("Age source")
        src_lay = QFormLayout(src_box)
        src_lay.setLabelAlignment(Qt.AlignRight)
        self._mode_cb = QComboBox()
        self._mode_cb.addItem("R mode", 'ratio_R_std')
        self._mode_cb.addItem("R mode (insert J)", 'ratio_J')
        self._mode_cb.addItem("Imported ages", 'imported')
        for _i in range(self._mode_cb.count()):
            if self._mode_cb.itemData(_i) == old_mode:
                self._mode_cb.setCurrentIndex(_i)
        self._mode_cb.setToolTip(
            "R mode: ages from the ⁴⁰Ar*/³⁹ArK ratios and the R standard.\n"
            "R mode (insert J): the same, with a J value you type or pick.\n"
            "Imported ages: use the ages stored in the file as they are.")
        src_lay.addRow("Age source:", self._mode_cb)
        self._apply_all_cb = QCheckBox(f"Apply to all {n_aliquots} aliquots")
        self._apply_all_cb.setChecked(True)
        self._apply_all_cb.setVisible(n_aliquots >= 2)
        src_lay.addRow("", self._apply_all_cb)
        layout.addWidget(src_box)
        src_box.setVisible(age_source)

        cal_box = QGroupBox("Calibration")
        cal_lay = QFormLayout(cal_box)
        cal_lay.setLabelAlignment(Qt.AlignRight)
        old_route = route_from_cfg(self._old)
        self._mon_cb = QComboBox()
        self._mon_cb.addItems(STANDARD_IDS)
        self._mon_cb.setCurrentIndex(max(0, self._mon_cb.findText(old_route.get('monitor', 'FCs'))))
        self._route_cb = QComboBox()
        self._route_cb.setToolTip("The calibrations this monitor can reach through the R values "
                                  "in Standards. Routes saved in Standards come first.")
        self._route_lbl = QLabel('')
        self._route_lbl.setWordWrap(True)
        self._route_lbl.setStyleSheet('color:#555; font-size:10px;')
        cal_lay.addRow("Monitor:", self._mon_cb)
        cal_lay.addRow("Calibration:", self._route_cb)
        cal_lay.addRow("", self._route_lbl)
        layout.addWidget(cal_box)

        # Advanced: hidden until asked for
        self._adv_btn = QPushButton("Advanced ▸")
        self._adv_btn.setFlat(True)
        self._adv_btn.setStyleSheet('text-align:left; color:#1565c0;')
        self._adv_btn.clicked.connect(self._toggle_advanced)
        layout.addWidget(self._adv_btn)
        self._adv = QWidget()
        adv = QVBoxLayout(self._adv)
        adv.setContentsMargins(0, 0, 0, 0)

        self._full_cb = QCheckBox("Full external error (Min et al. 2000)")
        self._full_cb.setToolTip(
            "Report the external error with the uncertainty of the two partial ⁴⁰K decay\n"
            "constants of Min et al. (2000), about 1.2% of the age (1σ), in place of the λ\n"
            "term of the calibration: for comparison with U–Pb or another chronometer.\n"
            "Off: the intercalibrated external of the calibration above, for comparing\n"
            "⁴⁰Ar/³⁹Ar ages with each other or with astronomically tuned ages.\n"
            "The ages do not change.")
        self._full_cb.setChecked(bool(full_external))
        self._full_cb.setVisible(full_external is not None)
        adv.addWidget(self._full_cb)

        cust_box = QGroupBox("Typed monitor age and λ  (instead of the calibration above)")
        cust_box.setCheckable(True)
        cust_box.setChecked(False)
        cust_lay = QFormLayout(cust_box)
        self._cust_box = cust_box
        self._c_age, self._c_age_s = QLineEdit(), QLineEdit()
        self._c_lam, self._c_lam_s = QLineEdit(f"{LAMBDA_DEFAULT:.6e}"), QLineEdit(f"{LAMBDA_ERR:.4e}")
        self._c_ext = QCheckBox("The ± of the age already includes the uncertainty of λ")
        cust_lay.addRow("Monitor age (Ma):", self._c_age)
        cust_lay.addRow("± 1σ (Ma):", self._c_age_s)
        cust_lay.addRow("λ (yr⁻¹):", self._c_lam)
        cust_lay.addRow("λ ± 1σ:", self._c_lam_s)
        cust_lay.addRow("", self._c_ext)
        adv.addWidget(cust_box)

        old_box = QGroupBox("Constants of the stored ages and J")
        old_lay = QFormLayout(old_box)
        self._o_line = QLabel(route_text(old_route) + (f"\n{origin}" if origin else ''))
        self._o_line.setWordWrap(True)
        self._o_line.setStyleSheet('color:#555; font-size:10px;')
        self._o_age = QLineEdit(f"{monitor_age(self._old):.8g}")
        self._o_lam = QLineEdit(f"{float(self._old.get('lambdaYr', LAMBDA_DEFAULT)):.6e}")
        self._o_texts = (self._o_age.text(), self._o_lam.text())
        for w in (self._o_age, self._o_lam):
            w.setToolTip("Change these for data published with other constants, e.g. ages and\n"
                         "a J from an older study. The stored ages and J are recast from them.")
        old_lay.addRow("", self._o_line)
        old_lay.addRow(f"{old_route.get('monitor', 'FCs')} age (Ma):", self._o_age)
        old_lay.addRow("λ (yr⁻¹):", self._o_lam)
        adv.addWidget(old_box)
        old_box.setVisible(stored)

        # which samples the calibration goes to (Age distribution)
        self._sample_cbs = []
        if samples:
            to_box = QGroupBox("Apply to")
            to_lay = QVBoxLayout(to_box)
            for name, on, line in samples:
                cb = QCheckBox(name)
                cb.setChecked(on)
                cb.setToolTip(line)
                to_lay.addWidget(cb)
                self._sample_cbs.append(cb)
            to_lay.addWidget(_note("Untick the samples that keep their own calibration, e.g. "
                                   "samples dated against another monitor."))
            adv.addWidget(to_box)

        b_std = QPushButton("Build a route in Standards…")
        b_std.setToolTip("Open Standards on the Routes page. Routes saved there are offered\n"
                         "under Calibration when you come back.")
        b_std.clicked.connect(self._open_standards)
        adv.addWidget(b_std)
        self._adv.setVisible(False)
        layout.addWidget(self._adv)

        self._mon_cb.currentTextChanged.connect(lambda _t: self._fill_routes())
        self._route_cb.currentIndexChanged.connect(self._show_route)
        cust_box.toggled.connect(self._show_route)
        for w in (self._c_age, self._c_age_s, self._c_lam, self._c_lam_s):
            w.textChanged.connect(self._show_route)
        self._fill_routes(old_route)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        if can_revert:
            btn_revert = QPushButton("Revert to original")
            btn_revert.setToolTip(
                "Undo all calibrations: restore the ages and constants\n"
                "exactly as they were when the file was first loaded.")
            btn_revert.clicked.connect(lambda: self.done(self.REVERT))
            bb.addButton(btn_revert, QDialogButtonBox.ResetRole)
        layout.addWidget(bb)

    # --- widgets ---------------------------------------------------------
    def _toggle_advanced(self):
        on = not self._adv.isVisible()
        self._adv.setVisible(on)
        self._adv_btn.setText("Advanced ▾" if on else "Advanced ▸")
        self.adjustSize()

    def _open_standards(self):
        from standards import StandardsEditor
        dlg = StandardsEditor(self)
        dlg.tabs.setCurrentIndex(2)
        dlg.exec_()
        rt = dlg.pages[2]
        name = (rt.rows[rt._cur].get('name') if rt._cur is not None
                and 0 <= rt._cur < len(rt.rows) else None)
        reload_standards()
        # take the route selected on the Routes page when Standards closed
        # (once saved, so it is in the library)
        hit = [r for r in STANDARDS.get('routes', []) if name and r.get('name') == name]
        if hit:
            r = hit[0]
            self._cust_box.setChecked(False)
            self._mon_cb.blockSignals(True)
            self._mon_cb.setCurrentIndex(max(0, self._mon_cb.findText(r.get('monitor', 'FCs'))))
            self._mon_cb.blockSignals(False)
            self._fill_routes(r)
        else:
            self._fill_routes()

    def _fill_routes(self, keep=None):
        self._opts = route_options(self._mon_cb.currentText())
        self._route_cb.blockSignals(True)
        self._route_cb.clear()
        self._route_cb.addItems([lbl for lbl, _ in self._opts])
        i = 0
        if keep is not None:
            for k, (_, r) in enumerate(self._opts):
                if keep.get('name') and r.get('name') == keep.get('name'):
                    i = k
                    break
                if (r.get('calibration') == keep.get('calibration')
                        and [st.get('to') for st in r.get('steps') or []]
                        == [st.get('to') for st in keep.get('steps') or []]):
                    i = k
                    break
            else:
                if isinstance(keep.get('calibration'), dict):     # typed values
                    c, lam = keep['calibration'], keep.get('lambda') or {}
                    self._cust_box.setChecked(True)
                    self._c_age.setText(f"{c.get('age', '')}")
                    self._c_age_s.setText(f"{c.get('age_1s', '')}")
                    if isinstance(lam, dict):
                        self._c_lam.setText(f"{lam.get('lambda', LAMBDA_DEFAULT):.6e}")
                        self._c_lam_s.setText(f"{lam.get('lambda_1s', 0.0):.4e}")
                    self._c_ext.setChecked(bool(c.get('sigma_includes_lambda')))
        self._route_cb.setCurrentIndex(i)
        self._route_cb.blockSignals(False)
        self._route_cb.setEnabled(bool(self._opts))
        self._show_route()

    def route(self):
        """The chosen route, or None when the typed values are not numbers."""
        if self._cust_box.isChecked():
            try:
                return custom_route(self._mon_cb.currentText(), float(self._c_age.text()),
                                    float(self._c_age_s.text() or 0), float(self._c_lam.text()),
                                    float(self._c_lam_s.text() or 0), self._c_ext.isChecked())
            except ValueError:
                return None
        i = self._route_cb.currentIndex()
        return copy.deepcopy(self._opts[i][1]) if 0 <= i < len(self._opts) else None

    def _show_route(self, *_):
        r = self.route()
        if r is None:
            self._route_lbl.setText('Type the monitor age and λ as numbers.'
                                    if self._cust_box.isChecked() else
                                    'No calibration can be reached from this monitor.')
            return
        try:
            c = route_cfg(r)
            txt = f"{route_text(r)}\n{r['monitor']} = {monitor_age(c):.6g} Ma"
            txt = '\n'.join([txt] + calibration_notes(c))
        except (ValueError, KeyError, TypeError) as e:
            txt = str(e)
        self._route_lbl.setText(txt)

    # --- result ----------------------------------------------------------
    def get_params(self):
        r = self.route()
        keys = route_cfg(r) if r is not None else route_cfg(route_from_cfg(self._old))
        old = dict(self._old)
        if (self._o_age.text(), self._o_lam.text()) != self._o_texts:
            try:
                old = route_cfg(custom_route(route_from_cfg(old)['monitor'],
                                             float(self._o_age.text()), 0.0,
                                             float(self._o_lam.text()), 0.0))
            except ValueError:
                pass
        return dict(
            lam2=keys['lambdaYr'], lam2_err=keys['lambdaErrYr'],
            fc2_ma=keys['stdAgeMa'], fc2_err_ma=keys['stdAgeErrMa'],
            fc2_external=keys['stdAgeErrType'] == 'external',
            route_keys={k: keys[k] for k in ROUTE_KEYS if k in keys},
            ing1=monitor_ingrowth(old, float(old.get('lambdaYr', LAMBDA_DEFAULT))),
            lam1=float(old.get('lambdaYr', LAMBDA_DEFAULT)),
            ing2=monitor_ingrowth(keys),
            prod1=cfg_monitor(self._old)[1], prod2=cfg_monitor(keys)[1],
            age_mode=self._mode_cb.currentData(),
            apply_all=bool(self._apply_all_cb.isChecked()),
            targets=[cb.isChecked() for cb in self._sample_cbs],
            full_external=self._full_cb.isChecked(),
        )
