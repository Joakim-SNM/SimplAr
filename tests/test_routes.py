"""Calibration routes checked without the route code: each age equation is
written out directly, every input is moved by its 1σ, and the change in the age
is compared with the error terms the tools compute.  Also the reading of a disk
flux on a route, and the shared terms of a pooled age.

Run:  python tests/test_routes.py
"""
import copy
import math
import os
import sys
import tempfile

os.environ.setdefault('SIMPLAR_STANDARDS_FILE', os.path.join(tempfile.mkdtemp(), 'standards.json'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import arar_math as am
import standards as S

lib = copy.deepcopy(am.STANDARDS)
acs = next(s for s in lib['standards'] if s['id'] == 'ACs')
acs['intercalibrations'].append(dict(R=0.17045, R_1s=0.00005, other='A1', ratio='std/FCs',
                                     reference='Lab (2026)'))
assert not S.check(lib)
LAM = {d['name']: (d['lambda'], d['lambda_1s']) for d in lib['decay_constants']}
CAL = {c['name']: c for c in lib['calibrations']}
bad = []

def step(frm, to, R=None, src=None):
    c = [x for x in S.steps_from(lib, frm) if x['to'] == to
         and (R is None or x['R'] == R) and (src is None or src in x['source'])]
    assert len(c) == 1, (frm, to, R, src, c)
    return c[0]

def chk(what, a, b, tol):
    ok = abs(a - b) <= tol * max(1e-12, abs(b))
    print(f"  {'ok ' if ok else 'BAD'} {what:<58} {a: .10g}  {b: .10g}")
    if not ok:
        bad.append(what)

def direct_I(factors, t_end, lam):
    """e^{λ t_M} − 1 = Π factors · (e^{λ t_end} − 1)."""
    p = 1.0
    for f in factors:
        p *= f
    return p * math.expm1(lam * t_end * 1e6)

def age(x, I, lam):
    return math.log1p(x * I) / lam / 1e6

cases = []
# (label, route, factors as (value, sigma, sign, label-prefix), end: dict)
def route(mon, steps, cal, lamname):
    return dict(name='', monitor=mon, steps=steps, calibration=cal, **{'lambda': lamname})

tcs = step('TCs', 'FCs')
cases.append(('TCs → FCs → Phillips (anchored)', route('TCs', [tcs], 'Phillips et al. (2022)', 'Min et al. (2000)')))
cases.append(('ACs → A1 lab R → A1 astronomical', route('ACs', [step('ACs', 'A1', src='Lab')], 'A1 astronomical age', 'Min et al. (2000)')))
cases.append(('FCs → A1 via A1/FCs (inverted) → A1', route('FCs', [step('FCs', 'A1', R=0.244686)], 'A1 astronomical age', 'Min et al. (2000)')))
cases.append(('ACs → Mes4 → Mes4 astronomical', route('ACs', [step('ACs', 'Mes4')], 'Mes4 astronomical age', 'Min et al. (2000)')))
cases.append(('FCs → GA1550 → K–Ar', route('FCs', [step('FCs', 'GA1550', R=3.5957)], 'GA1550 K–Ar age', 'Steiger & Jäger (1977)')))
cases.append(('TCs → FCs → Carter (joint)', route('TCs', [tcs], 'Carter et al. (2025)', 'Carter et al. (2025)')))
cases.append(('GA1550 → FCs → ACs → A1 (3 steps)', route('GA1550', [step('GA1550', 'FCs', R=3.5958), step('FCs', 'ACs', R=0.041707), step('ACs', 'A1', src='Lab')], 'A1 astronomical age', 'Min et al. (2000)')))

for label, r in cases:
    print(label)
    errs = S.route_problems(r, lib)
    assert not errs, errs
    cfg = am.route_cfg(r, lib)
    cal = CAL[r['calibration']]
    lam, slam = LAM[r['lambda']]
    # factors of the steps, from the printed values
    facs = [(st['R'] if st['num'] == st['from'] else 1 / st['R']) for st in r['steps']]
    anc = cal.get('anchor') if cal.get('basis') == 'astronomical' else None
    if anc:
        # published anchored calibration: R implied by its age at its own λ
        lam0 = LAM[cal['decay_constant']][0]
        Rc = math.expm1(lam0 * cal['age'] * 1e6) / math.expm1(lam0 * anc['age'] * 1e6)
        end_facs, t_end, s_end = [Rc], anc['age'], anc['age_1s']
    else:
        end_facs, t_end, s_end = [], cal['age'], cal['age_1s']
    I = direct_I(facs + end_facs, t_end, lam)
    chk('monitor ingrowth', am.monitor_ingrowth(cfg), I, 1e-13)
    chk('monitor age', am.monitor_age(cfg), math.log1p(I) / lam / 1e6, 1e-12)
    # a sample at about 55 Ma (1.2 Ma for young monitors is also fine)
    x = math.expm1(lam * 55e6) / I
    t0 = age(x, I, lam)
    comps = dict(am.systematic_components(t0, cfg))
    # each step: perturb its printed R by +σ
    for k, st in enumerate(r['steps']):
        f2 = list(facs)
        Rp = st['R'] + st['R_1s']
        f2[k] = Rp if st['num'] == st['from'] else 1 / Rp
        num = age(x, direct_I(f2 + end_facs, t_end, lam), lam) - t0
        lbl = f"R({st['num']}/{st['den']}), {st['source']}"
        chk(f'  step {lbl}', comps[lbl], num, 2e-3)
    # end age
    if anc:
        num = age(x, direct_I(facs + [Rc + anc['R_1s']], t_end, lam), lam) - t0
        lbl = next(k for k in comps if k.startswith(f"R({cal['standard']}/"))
        chk(f'  {lbl}', comps[lbl], num, 2e-3)
    num = age(x, direct_I(facs + end_facs, t_end + s_end, lam), lam) - t0
    lbl = next(k for k in comps if k.endswith('astronomical age') or k.startswith(f"{cal['standard']} age"))
    if cal.get('covariance') is not None:
        # joint: compare the total of the end-age and λ terms with gᵀΣg
        g_t = (age(x, direct_I(facs, t_end + 1e-4, lam), lam) - t0) / 1e-4
        g_l = (age(x, direct_I(facs, t_end, lam * (1 + 1e-7)), lam * (1 + 1e-7)) - t0) / (lam * 1e-7)
        cov = cal['covariance']
        var = (g_t * s_end)**2 + (g_l * slam)**2 + 2 * g_t * g_l * cov * 1e-6
        tot = math.hypot(comps[lbl], comps[f"λ, {r['lambda']}"])
        chk('  end age and λ together (with covariance)', tot, math.sqrt(var), 2e-3)
    else:
        chk(f'  {lbl}', comps[lbl], num, 2e-3)
        dl = slam
        if cal.get('basis') == 'k-ar':
            # a K–Ar age scales as 1/λ: λ t_end stays fixed; the ± excludes λ
            num = age(x, direct_I(facs + end_facs, t_end * lam / (lam + dl), lam + dl), lam + dl) - t0
        else:
            num = age(x, direct_I(facs + end_facs, t_end, lam + dl), lam + dl) - t0
        if dl > 0:
            chk(f"  λ, {r['lambda']}", comps[f"λ, {r['lambda']}"], num, 5e-3)
    # nothing else in the budget
    n_exp = len(r['steps']) + (3 if anc else 2)
    if len(comps) != n_exp:
        bad.append(f'{label}: {len(comps)} terms, expected {n_exp}: {list(comps)}')
        print('  BAD number of terms', list(comps))

# --- flux: read as a ratio of the route's monitor ---------------------------
print('Flux holes')
fx = am.disk_flux_entry(['1'], [0.0027], [2.7e-6], [[1.0]], (), '', [5.7], [0.0057],
                        5.463e-10, 28.176, 0.0115, None)
for fmon, fR in (('FCs', 1.0), ('TCs', 1.0112)):
    fx['monitor'], fx['monitorR'] = fmon, fR
    meas = 5.7 * fR                         # the ratio measured on the flux's own monitor
    for label, r in cases[:3]:
        cfg = am.route_cfg(r, lib)
        J, sJ = am.flux_hole_J(fx, 0, cfg)
        chk(f'  {fmon} flux, {label[:28]}: J = I_M / F_meas', J, am.monitor_ingrowth(cfg) / meas, 1e-13)
        rs = am.flux_hole_rstd(fx, 0, cfg)
        c2 = dict(cfg, ageFluxMode='ratio_R_std', rStd=rs, rStdErrPct=0.1)
        chk('    R standard gives the same J', am.cfg_J(c2)[0], J, 1e-13)
        chk('    relative σJ kept', sJ / J, 2.7e-6 / 0.0027, 1e-12)
        mm = am.monitor_mismatch_note(cfg, fmon)
        if (fmon != r['monitor']) != bool(mm):
            bad.append(f'mismatch note {fmon} vs {r["monitor"]}')
print('  ok  mismatch notes exactly when the monitors differ' if not any('mismatch' in b for b in bad) else '')

# --- pooled: shared terms combine as one parameter --------------------------
print('Pooled systematic error (two ages, fixed weights)')
rA = route('FCs', [], 'Phillips et al. (2022)', 'Min et al. (2000)')
rB = cases[0][1]
cA, cB = am.route_cfg(rA, lib), am.route_cfg(rB, lib)
lam = 5.463e-10
IA, IB = am.monitor_ingrowth(cA), am.monitor_ingrowth(cB)
xA, xB = math.expm1(lam * 55.0e6) / IA, math.expm1(lam * 55.3e6) / IB
w = [0.6, 0.4]
anc = CAL['Phillips et al. (2022)']['anchor']; lam0 = lam
Rc = math.expm1(lam0 * 28.176e6) / math.expm1(lam0 * anc['age'] * 1e6)
def wm(p):
    tA = age(xA, direct_I([p['Rc']], p['tA'], p['lam']), p['lam'])
    tB = age(xB, direct_I([p['RT'], p['Rc']], p['tA'], p['lam']), p['lam'])
    return w[0] * tA + w[1] * tB
p0 = dict(Rc=Rc, tA=anc['age'], lam=lam, RT=1.0112)
sig = dict(Rc=anc['R_1s'], tA=anc['age_1s'], lam=5.4e-12, RT=0.001)
var_num = sum((wm(dict(p0, **{k: p0[k] + s})) - wm(p0))**2 for k, s in sig.items())
ages = [age(xA, IA, lam), age(xB, IB, lam)]
C, labels = am.systematic_matrix(ages, [cA, cB])
var_mat = sum((w[0] * C[0, k] + w[1] * C[1, k])**2 for k in range(C.shape[1]))
chk('  shared terms add linearly, R(TCs/FCs) on B only', math.sqrt(var_mat), math.sqrt(var_num), 3e-3)
same = am.calibration_group_key(cA) == am.calibration_group_key(cB)
print(f"  {'ok ' if same else 'BAD'} both in one calibration group (so the terms are shared)")
if not same:
    bad.append('group key')

# --- full external replaces only λ ------------------------------------------
cF = dict(cB, fullSystematics=True)
cf = dict(am.systematic_components(55.0, cF))
assert not any(k.startswith('λ, ') for k in cf) and 'Decay constant (Min full external)' in cf
chk('full external: Min term', cf['Decay constant (Min full external)'],
    am.min_full_external(55.0)['decay_ma'], 1e-12)
print('\nPROBLEMS:' if bad else '\nALL PASSED', *bad, sep='\n  ')
assert not bad, bad
