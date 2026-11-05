#!/usr/bin/env python3
"""
Data converter: turns an ArArCALC summary export or a Mass Spec raw export
(.xlsx) into .arar session files for the other SimplAr tools.

Opens a PyQt5 dialog for format selection and column mapping verification.
Reads a 'Filename' column (if present) and stores it as 'runID' in each step.
"""

import argparse
import json
import math
import os
import re
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas openpyxl")

from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QRadioButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialogButtonBox, QLabel, QAbstractItemView, QComboBox, QLineEdit,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QDoubleValidator


# shared constants live one folder up, next to the launcher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arar_math import (
    LAMBDA_DEFAULT as _LAMBDA_DEFAULT, LAMBDA_ERR as _LAMBDA_ERR,
    STD_AGE_DEFAULT as _STD_AGE_DEFAULT, STD_AGE_ERR as _STD_AGE_ERR,
    calibration_for as _calibration_for, FC_PRESETS, DEFAULT_FC_NAME,
)
import data_folders as dfold

_HEADER_ROW = 0

# The FC age the Mass Spec software derived J with, offered from the shared
# presets (1σ), the suite default first.  It is stored in the .arar cfg so the
# other tools know the original calibration when recasting to another one.
def _fc_label(name, age):
    return name if '—' in name else f"{name} — {age:g} Ma"


_FC_REFERENCES = (
    [(_fc_label(n, a), a, e) for n, a, e, _ in FC_PRESETS if n == DEFAULT_FC_NAME]
    + [(_fc_label(n, a), a, e) for n, a, e, _ in FC_PRESETS if n != DEFAULT_FC_NAME]
    + [("Custom…", _STD_AGE_DEFAULT, _STD_AGE_ERR)]
)


# ── Column map definitions ────────────────────────────────────────────────────
# Each entry: (internal_key, default_excel_column, display_role)
# The first N_KEY_* entries are "key" (required); the rest are optional.

ARARCALC_COLS = [
    ("sample",    "Sample Name",           "Sample (grouping)"),
    ("age",       "Analysis Age",          "Age (Ma)"),
    ("age_sd",    "Analysis Age SD",       "Age 1σ error"),
    ("ar39k",     "Analysis Ar39(k)",      "³⁹Ar(k): %39Ar basis"),
    ("j_val",     "J-Value",               "J value"),
    ("j_sd",      "J-Value SD",            "J error"),
    ("r4039",     "Analysis Ar40/39",      "⁴⁰Ar/³⁹Ar ratio"),
    ("r4039_sd",  "Analysis Ar40/39 SD",   "⁴⁰Ar/³⁹Ar error"),
    ("pct40",     "Analysis Ar40(r)",      "%⁴⁰Ar*"),
    ("kca",       "Analysis K/Ca",         "K/Ca"),
    ("kca_sd",    "Analysis K/Ca SD",      "K/Ca error"),
    ("filename",  "Filename",              "RunID"),
    # ── optional ──
    ("experiment","Experiment",            "Experiment label"),
    ("irrad",     "Irradiation Label",     "Irradiation label"),
    ("temp",      "Analysis Temperature",  "Step temperature"),
    ("included",  "Analysis Included",     "Include/exclude flag"),
    ("std_r",     "Standard Ar40/39",      "Standard ⁴⁰Ar/³⁹Ar"),
    ("std_r_sd",  "Standard Ar40/39 SD",   "Standard ⁴⁰Ar/³⁹Ar error"),
]
_ARAR_N_KEY = 12   # rows 0–11 are key fields

MASSSPEC_COLS = [
    ("sample",   "Sample",               "Sample (grouping)"),
    ("age",      "Age",                  "Age (Ma)"),
    ("age_er",   "Age_Er",               "Age 1σ error"),
    ("ar39mol",  "Ar39_Moles",           "³⁹Ar moles, %39Ar basis"),
    ("j_val",    "J",                    "J value"),
    ("j_er",     "J_Er",                 "J error"),
    ("r4039",    "Ar40Rad_Over_Ar39",     "⁴⁰Ar*/³⁹Ar ratio (radiogenic)"),
    ("r4039_er", "Ar40Rad_Over_Ar39_Er", "⁴⁰Ar*/³⁹Ar error"),
    ("pct40",    "PctAr40Rad",           "%⁴⁰Ar*"),
    ("cak",      "Ca_Over_K",            "Ca/K"),
    ("cak_er",   "Ca_Over_K_Er",         "Ca/K error"),
    ("filename", "Filename",             "RunID"),
    # ── optional ──
    ("irrad",    "Irrad",                "Irradiation label"),
    ("pwr",      "Pwr_Requested",        "Step power"),
    ("run_id",   "Run_ID",               "Run ID"),
]
_MASS_N_KEY = 12


# ── Helpers ───────────────────────────────────────────────────────────────────

def _alpha_label(i: int) -> str:
    """0→'A', 25→'Z', 26→'AA', …"""
    s = ""; i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _parse_irrad_label(irrad: str) -> dict:
    s = str(irrad).strip()
    m = re.match(r'^\d*([A-Za-z]+)(\d+)', s)
    if m:
        dl = m.group(1).upper()
        return dict(disk_id=dl, hole_id=f'{dl}{m.group(2)}')
    return dict(disk_id=s, hole_id=s)


def _sf(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _ratio_sd(num, num_sd, den, den_sd) -> float:
    if num == 0 or den == 0:
        return 0.0
    return abs(num / den) * math.sqrt((num_sd / num) ** 2 + (den_sd / den) ** 2)


def _gc(row, col, default=0.0):
    """Get a column value; returns default if col is falsy or not in row."""
    if not col or col not in row.index:
        return default
    return _sf(row[col], default)


def _gco(row, col):
    """Get optional column; returns None if missing or zero."""
    v = _gc(row, col)
    return v if v != 0.0 else None


def _run_id(row, col):
    """Read the Filename/RunID column; returns empty string if absent."""
    if not col or col not in row.index:
        return ""
    v = row[col]
    return "" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v)


# ── Format detection ──────────────────────────────────────────────────────────

def detect_format(df: pd.DataFrame) -> str:
    """Returns 'ararcalc', 'massspec', or '' if unrecognised."""
    if "Sample Name" in df.columns and "Analysis Age" in df.columns:
        return "ararcalc"
    if all(c in df.columns for c in ("Sample", "Age", "Age_Er", "Ar39_Moles")):
        return "massspec"
    return ""


# ── ArArCALC export: run time, signals and constants ─────────────────────────
# ArArCALC writes the 37Ar/39Ar decay constants per HOUR and 36Cl per YEAR; the
# export module wants per day.
_ARC_CONST_SOURCES = [
    # "Air Ar40/36" is the measured air shot (it changes from run to run);
    # the atmospheric ratio actually adopted is the "Constant" column.
    ("air_40_36", "Air Ar40/36 Constant", 1.0), ("air_40_36_err", "Air Ar40/36 Constant SD", 1.0),
    ("lam_eps", "Lambda EC", 1.0), ("lam_beta", "Lambda Beta", 1.0),
    ("k40_abund", "40K/K Abundance Ratio", 1.0),
    ("lam_ar37", "Lambda 37Ar", 24.0), ("lam_ar39", "Lambda 39Ar", 24.0),
    ("lam_cl36", "Lambda 36Cl", 1.0 / 365.25),   # file: per year
    ("ca_39_37", "Irradiation Constant 39/37(ca)", 1.0),
    ("ca_38_37", "Irradiation Constant 38/37(ca)", 1.0),
    ("ca_36_37", "Irradiation Constant 36/37(ca)", 1.0),
    ("k_40_39", "Irradiation Constant 40/39(k)", 1.0),
    ("k_38_39", "Irradiation Constant 38/39(k)", 1.0),
    ("file_std_age", "Standard Age", 1.0), ("file_std_age_err", "Standard Age SD", 1.0),
    ("file_lambda_total", "Lambda Total", 1.0),
]


def _arc_run_datetime(row):
    """ISO 'YYYY-MM-DDTHH:MM' from the Date and Time columns, or ''."""
    if "Date" not in row.index:
        return ""
    try:
        ts = pd.Timestamp(row["Date"])
        if pd.isna(ts):
            return ""
        t = row["Time"] if "Time" in row.index else None
        if t is not None and not (isinstance(t, float) and math.isnan(t)):
            tt = pd.Timedelta(str(t)) if not hasattr(t, "hour") else \
                 pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
            ts = ts.normalize() + tt
        return ts.strftime("%Y-%m-%dT%H:%M")
    except Exception:
        return ""


def _arc_raw_block(row):
    """Signals as ArArCALC reports them: Intercept (before the blank), Blank,
    and their difference (the blank-corrected signal)."""
    out = {}
    for n in (36, 37, 38, 39, 40):
        i, ie = _opt_num(row, f"Intercept Ar{n}"), _opt_num(row, f"Intercept Ar{n} SD")
        b, be = _opt_num(row, f"Blank Ar{n}"), _opt_num(row, f"Blank Ar{n} SD")
        if i is not None:
            out[f"ar{n}_bsln"] = i
            if ie is not None:
                out[f"ar{n}_bsln_err"] = ie
        if b is not None:
            out[f"ar{n}_bkgd"] = b
            if be is not None:
                out[f"ar{n}_bkgd_err"] = be
        if i is not None and b is not None:
            out[f"ar{n}"] = i - b
            if ie is not None and be is not None:
                out[f"ar{n}_err"] = math.hypot(ie, be)
    for key, col in (("decay37", "Ar37 Decay Correction"),
                     ("decay39", "Ar39 Decay Correction"),
                     ("mdf", "MDF"), ("sens", "Sensitivity")):
        v = _opt_num(row, col)
        if v is not None:
            out[key] = v
    return out


def _arc_constants_block(grp):
    out = {}
    for _, row in grp.iterrows():
        for key, col, mult in _ARC_CONST_SOURCES:
            if key in out:
                continue
            v = _opt_num(row, col)
            if v is not None:
                out[key] = v * mult
    first = grp.iloc[0]
    for key, col in (("irrad_label", "Irradiation Label"), ("irradiation", "Irradiation"),
                     ("material", "Material"), ("signal_unit", "Sensitivity Unit"),
                     ("age_equation", "Age Equation Set"), ("standard_name", "Standard Name")):
        if col in grp.columns:
            v = first[col]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[key] = v if isinstance(v, (int, float)) else str(v)
    if "Sensitivity Unit" in grp.columns:
        u = str(first["Sensitivity Unit"]).lower()
        if "/" in u:
            out["signal_unit"] = u.split("/")[-1].strip()   # 'mol/volt' -> 'volt'
    return out


# ── FORMAT 1: ArArCALC summary ───────────────────────────────────────────────

def _convert_ararcalc_summary(df: pd.DataFrame, output_dir: str,
                               cm: dict) -> tuple:
    sample_col  = cm.get("sample",    "Sample Name")
    exp_cn      = cm.get("experiment","Experiment")
    irrad_cn    = cm.get("irrad",     "Irradiation Label")
    exp_col     = exp_cn   if exp_cn   in df.columns else None
    irrad_col   = irrad_cn if irrad_cn in df.columns else None
    group_keys  = [k for k in [sample_col, exp_col, irrad_col] if k]

    df = df[df[sample_col].notna() &
            (df[sample_col].astype(str).str.strip() != "nan")].copy().reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    written, warnings = [], []

    for keys, grp in df.groupby(group_keys, sort=False):
        grp  = grp.reset_index(drop=True)
        keys = keys if isinstance(keys, tuple) else (keys,)
        km   = dict(zip(group_keys, keys))
        sample_name = km.get(sample_col, "")
        experiment  = km.get(exp_col,   "") if exp_col   else ""
        irrad       = km.get(irrad_col, "") if irrad_col else ""

        ar39_col  = cm.get("ar39k", "Analysis Ar39(k)")
        ar39_vals = grp[ar39_col].apply(_sf) if ar39_col in grp.columns \
                    else pd.Series([1.0] * len(grp))
        total_ar39 = ar39_vals.sum()
        if total_ar39 <= 0:
            warnings.append(f"{sample_name}: Ar39(k) sums to zero, skipped")
            continue
        pct39 = ar39_vals / total_ar39 * 100.0

        steps = []
        for i, row in grp.iterrows():
            kca    = _gc(row, cm.get("kca",    "Analysis K/Ca"))
            kca_sd = _gc(row, cm.get("kca_sd", "Analysis K/Ca SD"))
            cak     = 1.0 / kca         if kca != 0 else 0.0
            err_cak = kca_sd / kca ** 2 if kca != 0 else 0.0

            pct40 = min(max(_gc(row, cm.get("pct40", "Analysis Ar40(r)")), 0.0), 100.0)

            inc_raw = row.get(cm.get("included", "Analysis Included"))
            if inc_raw is not None:
                status = "normal" if str(inc_raw).strip().lower() \
                         in ("true", "1", "yes", "y", "included") else "excluded"
            else:
                status = "normal"

            steps.append(dict(
                label     = _alpha_label(len(steps)),
                power     = _gc(row, cm.get("temp", "Analysis Temperature"),
                                float(len(steps) + 1)),
                age       = _gc(row, cm.get("age",    "Analysis Age")),
                err2s     = _gc(row, cm.get("age_sd", "Analysis Age SD")),
                pct39Ar   = float(pct39[i]),
                caK       = cak,
                err_cak   = err_cak,
                pct40     = pct40,
                r4039     = _gc(row, cm.get("r4039",    "Analysis Ar40/39")),
                err_r4039 = _gc(row, cm.get("r4039_sd", "Analysis Ar40/39 SD")),
                j_val     = _gc(row, cm.get("j_val", "J-Value")),
                j_err     = _gc(row, cm.get("j_sd",  "J-Value SD")),
                iso_ix    = _gco(row, "Ar39(k)/Ar36(a)"),
                iso_isx   = _gco(row, "Ar39(k)/Ar36(a) SD"),
                iso_iy    = _gco(row, "Ar40(a+r)/Ar36(a)"),
                iso_isy   = _gco(row, "Ar40(a+r)/Ar36(a) SD"),
                iso_iri   = _gc(row,  "Inverse Isochron RI", 0.0),
                iso_nx    = _gco(row, "Ar39(k)/Ar40(a+r)"),
                iso_nsx   = _gco(row, "Ar39(k)/Ar40(a+r) SD"),
                iso_ny    = _gco(row, "Ar36(a)/Ar40(a+r)"),
                iso_nsy   = _gco(row, "Ar36(a)/Ar40(a+r) SD"),
                iso_nri   = _gc(row,  "Normal Isochron RI", 0.0),
                status    = status,
                runID     = _run_id(row, cm.get("filename", "Filename")),
            ))
            _dt = _arc_run_datetime(row)
            if _dt:
                steps[-1]["run_dt"] = _dt
            _dd = _opt_num(row, "Decay Days")
            if _dd is not None:
                steps[-1]["days_irr"] = round(_dd, 6)
            _rb = _arc_raw_block(row)
            if _rb:
                steps[-1]["raw"] = _rb

        if not steps:
            warnings.append(f"{sample_name}: no valid steps, skipped")
            continue

        j_val     = steps[0]["j_val"]
        j_err     = steps[0]["j_err"]
        j_err_pct = (j_err / j_val * 100.0) if j_val > 0 else 0.0

        r_std = r_std_err_pct = 0.0
        src   = cm.get("std_r",    "Standard Ar40/39")
        src_e = cm.get("std_r_sd", "Standard Ar40/39 SD")
        if src in grp.columns:
            sv = grp[src].apply(_sf); sv = sv[sv > 0]
            if len(sv): r_std = float(sv.mean())
        if src_e in grp.columns and r_std > 0:
            se = grp[src_e].apply(_sf); se = se[se > 0]
            if len(se): r_std_err_pct = float(se.mean()) / r_std * 100.0

        # The calibration the file's ages and J were calculated with: its own
        # Standard Age and Lambda Total columns (tool default when absent).
        _ta = grp["Standard Age"].apply(_sf) if "Standard Age" in grp.columns else None
        _la = grp["Lambda Total"].apply(_sf) if "Lambda Total" in grp.columns else None
        _te = grp["Standard Age SD"].apply(_sf) if "Standard Age SD" in grp.columns else None
        _t = float(_ta[_ta > 0].median()) if _ta is not None and (_ta > 0).any() else _STD_AGE_DEFAULT
        _l = float(_la[_la > 0].median()) if _la is not None and (_la > 0).any() else _LAMBDA_DEFAULT
        _e = float(_te[_te > 0].median()) if _te is not None and (_te > 0).any() else None
        file_calib = _calibration_for(_t, _l, _e)

        # ArArCALC leaves "Standard Ar40/39" at 0 for some irradiations. The
        # monitor ratio is then recomputed from the file's own J, FC age and
        # decay constant, (e^(lambda*t) - 1)/J, which reproduces the column
        # wherever the file does give it.
        if r_std <= 0 and j_val > 0:
            r_std = (math.exp(_l * _t * 1e6) - 1.0) / j_val
            r_std_err_pct = j_err_pct

        irr = _parse_irrad_label(irrad) if irrad else dict(disk_id="", hole_id="")
        cfg = dict(
            minPct=50.0, minN=3, pCut=5.0, errMode="sem_mswd_2s",
            zoomMode="auto", yMin=50.0, yMax=70.0, ageFluxMode="imported",
            jVal=j_val, jErrPct=round(j_err_pct, 6),
            rStd=round(r_std, 6) if r_std > 0 else 70.0,
            rStdErrPct=round(r_std_err_pct, 6) if r_std_err_pct > 0 else 0.10,
            # labelled with the file's calibration, which the imported ages,
            # J and R standard are on (Calibration… converts from it)
            **file_calib,
            disk_id=irr["disk_id"], hole_id=irr["hole_id"],
            stepSigma=1,
        )
        _consts = _arc_constants_block(grp)
        if _consts:
            cfg["constants"] = _consts

        parts = [str(sample_name)]
        if experiment: parts.append(str(experiment))
        if irrad:      parts.append(str(irrad))
        title   = "  ·  ".join(parts)
        session = dict(version=2, title=title, cfg=cfg, steps=steps)

        def _safe(s): return dfold.safe_filename(str(s).replace(" ", "_").replace("/", "-"))
        name_parts = [_safe(sample_name)]
        if experiment: name_parts.append(_safe(experiment))
        if irrad:      name_parts.append(_safe(irrad))
        out_path = os.path.join(output_dir, "_".join(name_parts) + ".arar")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)
        written.append(out_path)

    return written, warnings


# ── FORMAT 2: Mass-spec raw ──────────────────────────────────────────────────

_RID_PREFIX = re.compile(r'^(\d+-\d+)')


def _run_id_prefix(val: str) -> str | None:
    """Extract experiment prefix from a Run_ID like '11776-01A' → '11776-01'.
    Returns None if the value doesn't match the expected pattern."""
    m = _RID_PREFIX.match(str(val))
    return m.group(1) if m else None


# ── Run time, raw signals and constants (for the data export) ────────────────

_RAW_SOURCES = []
for _n in (36, 37, 38, 39, 40):
    _RAW_SOURCES += [
        (f"ar{_n}",          f"Ar{_n}_"),            (f"ar{_n}_err",      f"Ar{_n}_Er"),
        (f"ar{_n}_nd",       f"Ar{_n}_notDecayCor"), (f"ar{_n}_nd_err",   f"Ar{_n}_Er_notDecayCor"),
        (f"ar{_n}_dc",       f"Ar{_n}_DecayCor"),    (f"ar{_n}_dc_err",   f"Ar{_n}_DecayCorEr"),
        (f"ar{_n}_bsln",     f"Ar{_n}_BslnCorOnly"), (f"ar{_n}_bsln_err", f"Ar{_n}_Er_BslnCorOnly"),
        (f"ar{_n}_bkgd",     f"Ar{_n}_Bkgd"),        (f"ar{_n}_bkgd_err", f"Ar{_n}_BkgdEr"),
        (f"ar{_n}_disc",     f"Ar{_n}_Disc"),        (f"ar{_n}_disc_err", f"Ar{_n}_DiscEr"),
        (f"ar{_n}_ic",       f"Ar{_n}_ICFactor"),
    ]
_RAW_SOURCES += [
    ("ar39_mol", "Ar39_Moles"), ("ar40_mol", "Ar40_Moles"),
    ("decay37", "37_Decay"),    ("decay39", "39_Decay"),
    ("pwr_req", "Pwr_Requested"), ("pwr_ach", "Pwr_Achieved"),
    ("heat_dur", "Tot_Dur_Heating"),
]

# constants that are the same for every step of an experiment
_CONST_SOURCES = [
    ("air_40_36", "Air_40_Over_36"), ("air_40_36_err", "Air_40_Over_36_Er"),
    ("air_40_38", "Air_40_Over_38"), ("air_40_38_err", "Air_40_Over_38_Er"),
    ("lam_eps", "Lambda_40K_epsilon"), ("lam_beta", "Lambda_40K_Beta"),
    ("lam_ar37", "Lambda_Ar37"), ("lam_ar39", "Lambda_Ar39"),
    ("lam_cl36", "Lambda_36Cl"), ("k40_abund", "40K_abundance"),
    ("ca_39_37", "Ca_39_Over_37"), ("ca_39_37_err", "Ca_39_Over_37_Er"),
    ("ca_38_37", "Ca_38_Over_37"), ("ca_38_37_err", "Ca_38_Over_37_Er"),
    ("ca_36_37", "Ca_36_Over_37"), ("ca_36_37_err", "Ca_36_Over_37_Er"),
    ("k_38_39", "K_38_Over_39"),   ("k_38_39_err", "K_38_Over_39_Er"),
    ("k_40_39", "K_40_Over_39"),   ("k_40_39_err", "K_40_Over_39_Er"),
    ("k_37_39", "K_37_Over_39"),   ("k_37_39_err", "K_37_Over_39_Er"),
    ("p36cl_38cl", "P36Cl_Over_38Cl"),
]


def _opt_num(row, col):
    """A number from the row, or None when the column is missing/empty."""
    if not col or col not in row.index:
        return None
    try:
        v = float(row[col])
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _run_datetime(row):
    """ISO 'YYYY-MM-DDTHH:MM' from Run_Date (+ Run_Hour, decimal hours), or ''."""
    if "Run_Date" not in row.index:
        return ""
    d = row["Run_Date"]
    try:
        if d is None or (isinstance(d, float) and math.isnan(d)):
            return ""
        ts = pd.Timestamp(d)
        if pd.isna(ts):
            return ""
        hr = _opt_num(row, "Run_Hour") or 0.0
        ts = ts.normalize() + pd.Timedelta(hours=hr)
        return ts.strftime("%Y-%m-%dT%H:%M")
    except Exception:
        return ""


def _days_since_irradiation(row):
    """Days from the irradiation reference to the analysis, from the decay
    correction the mass-spec software applied: ln(37Ar decay factor)/λ37
    (λ37 in 1/day); falls back to the 39Ar factor."""
    for fac, lam in (("37_Decay", "Lambda_Ar37"), ("39_Decay", "Lambda_Ar39")):
        f, l = _opt_num(row, fac), _opt_num(row, lam)
        if f and l and f > 1.0 and l > 0:
            return math.log(f) / l
    return None


def _raw_block(row):
    out = {}
    for key, col in _RAW_SOURCES:
        v = _opt_num(row, col)
        if v is not None:
            out[key] = v
    return out


def _constants_block(grp):
    """Constants from the first row that has them, plus the labels."""
    out = {}
    for _, row in grp.iterrows():
        for key, col in _CONST_SOURCES:
            if key in out:
                continue
            v = _opt_num(row, col)
            if v is not None:
                out[key] = v
    first = grp.iloc[0]
    for key, col in (("irrad_label", "Irrad"), ("material", "Material"),
                     ("mass_spec_system", "System_"), ("mass_spec_file_version", "Version")):
        if col in grp.columns:
            v = first[col]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[key] = v if isinstance(v, (int, float)) else str(v)
    return out



def _fix_run_dates(df, irrad_col=None):
    """Repair day/month swaps in Run_Date.

    The mass-spec file stores dates as text 'MM/DD/YYYY'. A spreadsheet in a
    DD/MM locale turns the ambiguous ones (day <= 12) into real dates with day
    and month exchanged, while the others stay as text, so one file holds both.
    The decay factors give the true time since irradiation, so a swapped date
    shows up as an irradiation time that disagrees with the text-dated rows.
    Each ambiguous date is kept or swapped, whichever agrees better with those.
    """
    if "Run_Date" not in df.columns or "37_Decay" not in df.columns:
        return df
    out = df.copy()
    info = []                                   # (index, ts, certain, ref)
    for idx, row in out.iterrows():
        d = row["Run_Date"]
        certain = isinstance(d, str)
        try:
            ts = pd.Timestamp(d)
        except Exception:
            ts = pd.NaT
        days = _days_since_irradiation(row)
        if pd.isna(ts) or days is None:
            info.append((idx, None, certain, None))
            continue
        hr = _opt_num(row, "Run_Hour") or 0.0
        ts = ts.normalize()
        ref = ts + pd.Timedelta(hours=hr) - pd.Timedelta(days=days)
        info.append((idx, ts, certain, ref))
    keys = (out[irrad_col].astype(str) if irrad_col else pd.Series("", index=out.index))
    fixed = {}
    for key in keys.unique():
        rows = [t for t in info if keys[t[0]] == key and t[1] is not None]
        sure = [t[3] for t in rows if t[2]]
        if not sure:
            continue
        med = pd.Series(sure).sort_values().iloc[len(sure) // 2]
        for idx, ts, certain, ref in rows:
            if certain or ts.day > 12:
                continue
            try:
                sw = ts.replace(month=ts.day, day=ts.month)
            except ValueError:
                continue
            ref_sw = ref + (sw - ts)
            if abs(ref_sw - med) < abs(ref - med) and abs(ref - med) > pd.Timedelta(days=1):
                fixed[idx] = sw
    if fixed:
        col = out["Run_Date"].astype(object)
        for idx, sw in fixed.items():
            col[idx] = sw
        out["Run_Date"] = col
    return out


def _convert_mass_spec_raw(df: pd.DataFrame, output_dir: str,
                            cm: dict,
                            fc_age_ma: float = _STD_AGE_DEFAULT,
                            fc_age_err_ma: float = _STD_AGE_ERR) -> tuple:
    sample_col = cm.get("sample", "Sample")
    rid_cn     = cm.get("run_id", "Run_ID")
    irrad_cn   = cm.get("irrad",  "Irrad")
    irrad_col  = irrad_cn if irrad_cn in df.columns else None
    df = _fix_run_dates(df, irrad_col)

    # Group by Run_ID prefix (e.g. "11776-01") when available; this correctly
    # splits separate crystals/aliquots whose Sample column is a shared lab number.
    df = df.copy()
    if rid_cn in df.columns:
        df['_exp'] = df[rid_cn].apply(_run_id_prefix)
        df = df[df['_exp'].notna()].reset_index(drop=True)
        group_col = '_exp'
    else:
        df = df[df[sample_col].notna() &
                (df[sample_col].astype(str).str.strip() != "nan")].reset_index(drop=True)
        group_col = sample_col

    group_keys = [group_col, irrad_col] if irrad_col else [group_col]

    os.makedirs(output_dir, exist_ok=True)
    written, warnings = [], []

    for keys, grp in df.groupby(group_keys, sort=False):
        grp = grp.reset_index(drop=True)
        sample_name = keys[0] if isinstance(keys, tuple) else keys
        irrad       = keys[1] if isinstance(keys, tuple) and irrad_col else ""

        ar39_col  = cm.get("ar39mol", "Ar39_Moles")
        ar39_vals = grp[ar39_col].apply(_sf) if ar39_col in grp.columns \
                    else pd.Series([1.0] * len(grp))
        total_ar39 = ar39_vals.sum()
        if total_ar39 <= 0:
            warnings.append(f"{sample_name}: Ar39_Moles sums to zero, skipped")
            continue
        pct39 = ar39_vals / total_ar39 * 100.0

        steps = []
        for i, row in grp.iterrows():
            # Inverse isochron from decay-corrected signals
            ar39d  = _gc(row, "Ar39_DecayCor")
            ar39de = _gc(row, "Ar39_DecayCorEr")
            ar40d  = _gc(row, "Ar40_DecayCor")
            ar40de = _gc(row, "Ar40_DecayCorEr")
            ar36d  = _gc(row, "Ar36_DecayCor")
            ar36de = _gc(row, "Ar36_DecayCorEr")

            if ar36d != 0:
                iso_ix  = ar39d / ar36d
                iso_isx = _ratio_sd(ar39d, ar39de, ar36d, ar36de)
                iso_iy  = ar40d / ar36d
                iso_isy = _ratio_sd(ar40d, ar40de, ar36d, ar36de)
            else:
                iso_ix = iso_isx = iso_iy = iso_isy = None

            nx_val = _gc(row, "Isoch_39_Over_40", float("nan"))
            ny_val = _gc(row, "Isoch_36_Over_40", float("nan"))
            pnx    = _gc(row, "Pct_i39_Over_40_Er", 0.0)
            pny    = _gc(row, "Pct_i36_Over_40_Er", 0.0)
            iso_nx  = nx_val if math.isfinite(nx_val) and nx_val != 0 else None
            iso_nsx = abs(nx_val) * abs(pnx) / 100.0 if iso_nx else None
            iso_ny  = ny_val if math.isfinite(ny_val) and ny_val != 0 else None
            iso_nsy = abs(ny_val) * abs(pny) / 100.0 if iso_ny else None

            # No decay-corrected signals in the file: build the 39/36 vs 40/36
            # diagram from the isochron columns (36 there is the atmospheric
            # ³⁶Ar, after the Ca and Cl corrections).
            if iso_ix is None and iso_nx and iso_ny and iso_ny > 0:
                p_ix    = _gc(row, "Pct_i39_Over_36_Er", 0.0)
                iso_ix  = iso_nx / iso_ny
                iso_isx = abs(iso_ix) * abs(p_ix) / 100.0
                iso_iy  = 1.0 / iso_ny
                iso_isy = iso_iy * abs(pny) / 100.0

            rid_col = cm.get("run_id", "Run_ID")

            step = dict(
                label     = _alpha_label(len(steps)),
                power     = _gc(row, cm.get("pwr", "Pwr_Requested"), float(len(steps) + 1)),
                age       = _gc(row, cm.get("age",    "Age")),
                err2s     = _gc(row, cm.get("age_er", "Age_Er")),
                pct39Ar   = float(pct39[i]),
                caK       = _gc(row, cm.get("cak",    "Ca_Over_K")),
                err_cak   = _gc(row, cm.get("cak_er", "Ca_Over_K_Er")),
                pct40     = _gc(row, cm.get("pct40",  "PctAr40Rad")),
                r4039     = _gc(row, cm.get("r4039",    "Ar40Rad_Over_Ar39")),
                err_r4039 = _gc(row, cm.get("r4039_er", "Ar40Rad_Over_Ar39_Er")),
                j_val     = _gc(row, cm.get("j_val", "J")),
                j_err     = _gc(row, cm.get("j_er",  "J_Er")),
                iso_ix=iso_ix, iso_isx=iso_isx, iso_iy=iso_iy, iso_isy=iso_isy,
                iso_iri   = _gc(row, "Correl_36_Over_39", 0.0),
                iso_nx=iso_nx, iso_nsx=iso_nsx, iso_ny=iso_ny, iso_nsy=iso_nsy,
                iso_nri   = _gc(row, "Correl_40_Over_39", 0.0),
                status    = "normal",
                runID     = _run_id(row, cm.get("filename", "Filename")),
            )
            _dt = _run_datetime(row)
            if _dt:
                step["run_dt"] = _dt
            _dd = _days_since_irradiation(row)
            if _dd is not None:
                step["days_irr"] = round(_dd, 6)
            _rawb = _raw_block(row)
            if _rawb:
                step["raw"] = _rawb
            rid = _run_id(row, rid_col)
            if rid:
                step["run_id"] = rid
            steps.append(step)

        if not steps:
            warnings.append(f"{sample_name}: no valid steps, skipped")
            continue

        # one J per sample: the mean over its steps (they normally share it)
        j_steps = [(s["j_val"], s["j_err"]) for s in steps if s.get("j_val", 0) > 0]
        if j_steps:
            j_val     = math.fsum(j for j, _ in j_steps) / len(j_steps)
            j_err_pct = 100.0 * math.fsum(e / j for j, e in j_steps) / len(j_steps)
        else:
            j_val, j_err_pct = steps[0]["j_val"], 0.0

        # Read λ from the data (Lambda_40K_epsilon + Lambda_40K_Beta per step).
        # This is stored per-step by the mass spec software and must match what
        # was used to compute the exported ages; using the wrong λ shifts all
        # ages systematically (e.g. 5.46e-10 vs 5.5042e-10 → ~0.4 Ma at 60 Ma).
        def _lam_from_row(row):
            eps  = _gc(row, "Lambda_40K_epsilon", 0.0)
            beta = _gc(row, "Lambda_40K_Beta",    0.0)
            return eps + beta if (eps + beta) > 0 else 0.0

        lam_vals = [_lam_from_row(grp.iloc[i]) for i in range(len(grp))]
        lam_vals = [v for v in lam_vals if v > 0]
        lam_used = float(sum(lam_vals) / len(lam_vals)) if lam_vals else _LAMBDA_DEFAULT

        # λ error: combine epsilon and beta errors in quadrature (mean across steps)
        def _lam_err_from_row(row):
            ee = _gc(row, "Lambda_40K_epsilon_Er", 0.0)
            be = _gc(row, "Lambda_40K_Beta_Er",    0.0)
            return math.sqrt(ee**2 + be**2) if (ee + be) > 0 else 0.0

        lam_err_vals = [_lam_err_from_row(grp.iloc[i]) for i in range(len(grp))]
        lam_err_vals = [v for v in lam_err_vals if v > 0]
        lam_err_used = float(sum(lam_err_vals) / len(lam_err_vals)) \
                       if lam_err_vals else _LAMBDA_ERR

        # rStd must be computed with the same (λ, FC age) that the mass spec used
        # when deriving J, so that ratio_R_std mode recovers that J exactly.
        # The other tools can then recast to any target calibration using
        # lambdaYr + stdAgeMa as the "old" constants.
        # Plain e^(λ t) − 1 is deliberate: this undoes the mass spectrometer's own
        # J, which it computed that way, to recover the measured monitor ratio.
        # From here on the other tools apply the anchored ingrowth
        # (arar_math.fc_ingrowth), identical to this at the calibration's own λ.
        r_std = (math.exp(lam_used * fc_age_ma * 1e6) - 1.0) / j_val \
                if j_val > 0 else 70.0

        irr = _parse_irrad_label(irrad) if irrad else dict(disk_id="", hole_id="")
        cfg = dict(
            minPct=50.0, minN=3, pCut=5.0, errMode="sem_mswd_2s",
            zoomMode="auto", yMin=50.0, yMax=70.0, ageFluxMode="imported",
            jVal=j_val, jErrPct=round(j_err_pct, 6),
            rStd=round(r_std, 6), rStdErrPct=round(j_err_pct, 6),
            stdAgeMa=fc_age_ma, stdAgeErrMa=fc_age_err_ma,
            lambdaYr=round(lam_used, 15), lambdaErrYr=round(lam_err_used, 18),
            disk_id=irr["disk_id"], hole_id=irr["hole_id"],
            stepSigma=1,
        )
        _consts = _constants_block(grp)
        if _consts:
            cfg["constants"] = _consts

        title   = f"{sample_name} ({irrad})" if irrad else str(sample_name)
        session = dict(version=2, title=title, cfg=cfg, steps=steps)

        safe_s   = dfold.safe_filename(str(sample_name).replace(" ", "_").replace("/", "-"))
        safe_i   = (dfold.safe_filename(str(irrad).replace(" ", "_").replace("/", "-"))
                    if irrad else "noIrrad")
        out_path = os.path.join(output_dir, f"{safe_s}_{safe_i}.arar")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)
        written.append(out_path)

    return written, warnings


def convert(input_path: str, output_dir: str, fmt: str, col_map: dict,
            fc_age_ma: float = _STD_AGE_DEFAULT,
            fc_age_err_ma: float = _STD_AGE_ERR) -> tuple:
    df = pd.read_excel(input_path, header=_HEADER_ROW, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    if fmt == "ararcalc":
        return _convert_ararcalc_summary(df, output_dir, col_map)
    else:
        return _convert_mass_spec_raw(df, output_dir, col_map, fc_age_ma, fc_age_err_ma)


# ── GUI dialog ────────────────────────────────────────────────────────────────

class ConvertDialog(QDialog):
    _GREEN = QColor("#2e7d32")
    _RED   = QColor("#c62828")
    _GREY  = QColor("#888888")

    def __init__(self, input_path: str, df_cols: list, detected_fmt: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data converter")
        self.setMinimumWidth(520)

        self._df_cols    = set(df_cols)
        self.fmt_result  = None
        self.col_result  = None

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 14, 16, 14)

        # File label
        lbl = QLabel(f"<b>File:</b> {os.path.basename(input_path)}")
        root.addWidget(lbl)

        # ── Format selector ──────────────────────────────────────────────────
        fg = QGroupBox("Import format")
        fl = QVBoxLayout(fg)
        self._rb_arar = QRadioButton("ArArCALC summary export")
        self._rb_mass = QRadioButton("Mass Spec raw export")
        fl.addWidget(self._rb_arar)
        fl.addWidget(self._rb_mass)

        if detected_fmt == "massspec":
            self._rb_mass.setChecked(True)
        else:
            self._rb_arar.setChecked(True)   # default / ararcalc

        det_label = {
            "ararcalc": "  (auto-detected: ArArCALC)",
            "massspec":  "  (auto-detected: Mass Spec)",
            "":          "  (format not recognised, please select)",
        }.get(detected_fmt, "")
        if det_label:
            fl.addWidget(QLabel(det_label))

        self._rb_arar.toggled.connect(self._refresh_table)
        self._rb_mass.toggled.connect(self._refresh_table)
        root.addWidget(fg)

        # ── FC age reference (mass spec only) ────────────────────────────────
        self._fc_group = QGroupBox("FC sanidine age used to derive J")
        fc_layout = QVBoxLayout(self._fc_group)

        self._fc_combo = QComboBox()
        for label, _, _ in _FC_REFERENCES:
            self._fc_combo.addItem(label)
        # opens on the suite default; the combo fills the editable fields below
        self._fc_combo.setCurrentIndex(0)
        fc_layout.addWidget(self._fc_combo)

        fields_row = QHBoxLayout()
        fields_row.addWidget(QLabel("Age (Ma):"))
        self._fc_age_edit = QLineEdit()
        self._fc_age_edit.setValidator(QDoubleValidator(0, 100, 6))
        self._fc_age_edit.setFixedWidth(80)
        fields_row.addWidget(self._fc_age_edit)
        fields_row.addWidget(QLabel("±1σ (Ma):"))
        self._fc_err_edit = QLineEdit()
        self._fc_err_edit.setValidator(QDoubleValidator(0, 10, 6))
        self._fc_err_edit.setFixedWidth(70)
        fields_row.addWidget(self._fc_err_edit)
        fields_row.addStretch()
        fc_layout.addLayout(fields_row)

        note = QLabel("Written to the .arar file so the other tools know the\n"
                      "original calibration when recasting to another one.")
        note.setStyleSheet("color: #666; font-size: 10px;")
        fc_layout.addWidget(note)
        root.addWidget(self._fc_group)

        self._fc_combo.currentIndexChanged.connect(self._on_fc_ref_changed)
        self._on_fc_ref_changed(0)   # populate fields with first entry

        # ── Column mapping table ─────────────────────────────────────────────
        cg = QGroupBox("Column mapping: edit any cell to override")
        cl = QVBoxLayout(cg)

        hint = QLabel("* required columns   (grey rows are optional)")
        hint.setStyleSheet("color: #666; font-size: 10px;")
        cl.addWidget(hint)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Role", "Excel column", ""])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.itemChanged.connect(self._on_cell_changed)
        cl.addWidget(self._table)
        root.addWidget(cg)

        # ── Buttons ──────────────────────────────────────────────────────────
        bbox = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        bbox.button(QDialogButtonBox.Ok).setText("Convert")
        bbox.accepted.connect(self._accept)
        bbox.rejected.connect(self.reject)
        root.addWidget(bbox)

        self._refresh_table()

    def _effective_fmt(self) -> str:
        return "ararcalc" if self._rb_arar.isChecked() else "massspec"

    def _on_fc_ref_changed(self, idx: int):
        _, age, err = _FC_REFERENCES[idx]
        self._fc_age_edit.setText(str(age))
        self._fc_err_edit.setText(str(err))

    def fc_age_ma(self) -> float:
        try:    return float(self._fc_age_edit.text())
        except ValueError: return _STD_AGE_DEFAULT

    def fc_age_err_ma(self) -> float:
        try:    return float(self._fc_err_edit.text())
        except ValueError: return _STD_AGE_ERR

    def _refresh_table(self):
        fmt  = self._effective_fmt()
        self._fc_group.setVisible(fmt == "massspec")
        cols = ARARCALC_COLS if fmt == "ararcalc" else MASSSPEC_COLS
        nkey = _ARAR_N_KEY  if fmt == "ararcalc" else _MASS_N_KEY

        self._table.blockSignals(True)
        self._table.setRowCount(len(cols))

        for r, (key, default, desc) in enumerate(cols):
            is_key = r < nkey

            # Role cell (read-only)
            role_txt  = (desc + "  *") if is_key else desc
            role_item = QTableWidgetItem(role_txt)
            role_item.setFlags(Qt.ItemIsEnabled)
            if not is_key:
                role_item.setForeground(self._GREY)
            self._table.setItem(r, 0, role_item)

            # Column name cell (editable); store internal key in UserRole
            col_item = QTableWidgetItem(default)
            col_item.setData(Qt.UserRole, key)
            if not is_key:
                col_item.setForeground(self._GREY)
            self._table.setItem(r, 1, col_item)

            # Status indicator
            self._update_status(r, default)

        self._table.blockSignals(False)

    def _update_status(self, row: int, col_name: str):
        found = col_name in self._df_cols
        item  = QTableWidgetItem("✓" if found else "✗")
        item.setForeground(self._GREEN if found else self._RED)
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(Qt.ItemIsEnabled)
        self._table.setItem(row, 2, item)

    def _on_cell_changed(self, item: QTableWidgetItem):
        if item.column() == 1:
            self._update_status(item.row(), item.text().strip())

    def _accept(self):
        fmt = self._effective_fmt()
        col_map = {}
        for r in range(self._table.rowCount()):
            ci = self._table.item(r, 1)
            if ci:
                col_map[ci.data(Qt.UserRole)] = ci.text().strip()
        self.fmt_result      = fmt
        self.col_result      = col_map
        self.fc_age_result   = self.fc_age_ma()
        self.fc_age_err_result = self.fc_age_err_ma()
        self.accept()


# ── CLI helpers ───────────────────────────────────────────────────────────────

def list_groups(input_path: str) -> None:
    df  = pd.read_excel(input_path, header=_HEADER_ROW, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    fmt = detect_format(df)
    sc  = "Sample Name" if fmt == "ararcalc" else "Sample"
    ic  = "Irradiation Label" if fmt == "ararcalc" else "Irrad"
    df  = df[df[sc].notna() & (df[sc].astype(str).str.strip() != "nan")]
    has = ic in df.columns
    gk  = [sc, ic] if has else [sc]
    print(f"{'Sample':<20}  {'Irrad':<15}  Steps")
    print("-" * 45)
    for keys, grp in df.groupby(gk, sort=False):
        s = keys[0] if isinstance(keys, tuple) else keys
        i = keys[1] if isinstance(keys, tuple) and has else "-"
        print(f"  {str(s):<18}  {str(i):<13}  {len(grp)}")
    print(f"\nTotal: {df[sc].count()} step(s)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert an ArArCALC or Mass Spec export (.xlsx) to .arar session files."
    )
    parser.add_argument("input",  nargs="?", default=None,
                        help="Input .xlsx (omit to open file picker)")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output folder (default: pick one, starting in the "
                             "Data converter folder)")
    parser.add_argument("--list", dest="list_only", action="store_true",
                        help="List all sample groups and exit")
    args = parser.parse_args()

    if args.list_only:
        if not args.input or not os.path.isfile(args.input):
            sys.exit("Provide a valid input file with --list.")
        list_groups(args.input)
        return

    # ── read file (or pick it) ────────────────────────────────────────────────
    _app = QApplication.instance() or QApplication(sys.argv)   # keep a reference
    dfold.use_tool('data_converter')

    input_path = args.input
    if input_path and not os.path.isfile(input_path):
        sys.exit(f"File not found: {input_path}")

    if not input_path:
        input_path, _ = dfold.get_open_file_name(
            None, "Open an ArArCALC or Mass Spec export", "", "Excel files (*.xlsx *.xls)")
        if not input_path:
            sys.exit(0)

    output_dir = args.output
    if not output_dir:
        output_dir = dfold.get_existing_directory(None, "Select output folder")
        if not output_dir:
            sys.exit(0)

    df = pd.read_excel(input_path, header=_HEADER_ROW, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    detected = detect_format(df)

    # ── show dialog ───────────────────────────────────────────────────────────
    dlg = ConvertDialog(input_path, list(df.columns), detected)
    if dlg.exec_() != QDialog.Accepted:
        sys.exit(0)

    # ── convert ───────────────────────────────────────────────────────────────
    written, warnings = convert(input_path, output_dir, dlg.fmt_result, dlg.col_result,
                                fc_age_ma=dlg.fc_age_result,
                                fc_age_err_ma=dlg.fc_age_err_result)

    lines = []
    for w in warnings:
        lines.append(f"WARNING: {w}")
    lines.append(f"Converted {len(written)} file(s) to:\n{output_dir}")
    if warnings:
        lines.append(f"({len(warnings)} skipped, see warnings above)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
