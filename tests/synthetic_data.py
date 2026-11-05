"""Made-up data for the tests.  Nothing here is a real measurement: the values
are drawn from fixed random seeds around round numbers, so every run produces
the same data and the tests can be shared publicly."""
import json
import os

import numpy as np

R_MONITOR = 5.67          # a Fish Canyon-like ⁴⁰Ar*/³⁹Ar for a ~55 Ma unknown's disk
J_TRUE = 0.0027


def monitor_rows(seed=1, positions=('E1', 'E5', 'E9')):
    """{position: [rows]} in the row format of the Flux monitors tool: per
    position four step-heated crystals (three laser-power steps each), eight
    fusions, one outlier, one point with a large error and one unusable point.
    Errors are 1σ."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, pos in enumerate(positions):
        centre = R_MONITOR * (1 + 0.002 * (k - 1))         # a small flux gradient
        tau = centre * 0.0004                               # grain-to-grain scatter
        rows = []

        def add(gid, label, power, ar39, value, err):
            sid = f"{pos}-{gid}-{label}"
            rows.append({'sampleId': sid, 'value40_39': value, 'uncertainty40_39': err,
                         'value': value, 'uncertainty': err, 'measuredJ': None,
                         'measuredJError': None, 'importedJ': None, 'importedJError': None,
                         'power': power, 'grain_id': gid, 'ar39': ar39, 'label': label})
        for c in range(4):
            g = centre + rng.normal(0, tau)
            for s, (p, f) in enumerate(((2.0, 0.2), (5.0, 0.5), (30.0, 0.3))):
                e = centre * rng.uniform(0.0006, 0.004)
                add(f"G{c + 1}", "ABC"[s], p, f, float(g + rng.normal(0, e)), float(e))
        for c in range(8):
            e = centre * rng.uniform(0.0004, 0.006)
            add(f"F{c + 1}", "A", 30.0, 1.0,
                float(centre + rng.normal(0, tau) + rng.normal(0, e)), float(e))
        add("X1", "A", 30.0, 1.0, float(centre * 1.02), float(centre * 0.001))     # outlier
        add("X2", "A", 30.0, 1.0, float(centre), float(centre * 0.09))             # 9 % error
        add("X3", "A", 30.0, 1.0, float(centre), 0.0)                              # unusable
        out[pos] = rows
    return out


def write_arar_files(folder, seed=2):
    """Two .arar session files of step-heated monitor crystals (run IDs with a
    step letter, so crystals are found from the run ID), plus one step marked
    excluded.  Returns the paths."""
    rng = np.random.default_rng(seed)
    paths = []
    for n, (irr, pos) in enumerate((('TEST1', 'E3'), ('TEST1', 'E7'))):
        steps = []
        for c in range(5):
            g = R_MONITOR * (1 + 0.001 * n) + rng.normal(0, 0.002)
            for s, (p, f) in enumerate(((2.0, 15.0), (6.0, 55.0), (30.0, 30.0))):
                e = R_MONITOR * rng.uniform(0.0008, 0.003)
                r = float(g + rng.normal(0, e))
                steps.append(dict(label="ABC"[s], power=p, pct39Ar=f, r4039=r,
                                  err_r4039=float(2 * e), age=55.0, err2s=0.1,
                                  status='excluded' if (c, s) == (2, 0) else 'normal',
                                  runID=f"{90000 + 10 * n + c}-{c + 1:02d}{'ABC'[s]}"))
        cfg = dict(errScale=2, stdAgeMa=28.176, lambdaYr=5.463e-10, hole_id=pos,
                   disk_id=pos[0], constants=dict(irrad_label=f"{irr}-{pos}"))
        path = os.path.join(folder, f"monitor_{pos}.arar")
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(dict(version=2, title=f"{irr} {pos}", cfg=cfg, steps=steps), fh, indent=1)
        paths.append(path)
    return paths


# ── single-crystal ratios of one unknown ───────────────────────────────────

RSTD = 5.7          # monitor ⁴⁰Ar*/³⁹Ar of the made-up irradiation
FLUX_PCT = 0.02     # its relative 1σ error, in %


def ratio_set(n=43, seed=3):
    """n made-up single-crystal analyses of one ~55 Ma sample:
    R = (⁴⁰Ar*/³⁹Ar)_unknown / (⁴⁰Ar*/³⁹Ar)_monitor with 1σ errors (no excess
    scatter), and the same analyses as unknown ratios F = R · RSTD.
    Returns (R, sR, F, sF)."""
    rng = np.random.default_rng(seed)
    sR = 2.0 * rng.uniform(1.5e-4, 7e-4, n)
    R = 2.0 + rng.normal(0, sR)
    return R, sR, R * RSTD, sR * RSTD


# ── a step-heating experiment ──────────────────────────────────────────────

def plateau_steps(seed=4):
    """24 made-up heating steps with a ~54.7 Ma plateau: two low-temperature
    steps lose argon, the last step is young, the rest agree within error.
    Errors are 1σ ('err1s')."""
    rng = np.random.default_rng(seed)
    n = 24
    pct = rng.uniform(2.0, 6.0, n)
    pct = 100.0 * pct / pct.sum()
    out = []
    for k in range(n):
        e = float(rng.uniform(0.09, 0.40))
        age = 54.7 + rng.normal(0, e)
        if k in (0, 1):
            age -= 0.9
        if k == n - 1:
            age -= 1.2
        label = chr(65 + k) if k < 26 else f"A{chr(65 + k - 26)}"
        out.append(dict(label=label, power=round(2.8 + 0.5 * k, 2), age=float(age),
                        err1s=e, pct39Ar=float(pct[k]), caK=float(33.0 + 0.5 * k),
                        pct40=float(min(99.0, 20.0 + 3.0 * k)), status='normal'))
    return out
