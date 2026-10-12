"""Assert that the interim report's JSON-backed numbers match the summaries.

The decks are built by script so they cannot drift from the logs (``deck/README.md``).  A
prose report cannot be built that way, so this is the equivalent guard: every number in
``frontogenesis_M3_interim.md`` that has a JSON behind it is re-read from
``../data/m3_closure_summary.json`` (task 6) and ``../data/m3_figs_summary.json`` (task 7),
formatted the way the report formats it, and required to appear verbatim in the text.
That is JXP's answer to M3-Q24 and M3-Q23 (a), made mechanical.

It also enforces the prompt's "Do not" list on the report:

* the measured-vs-predicted slopes (``slopes_not_quoted``) must **not** appear, and
* the report must say, in words, that they are withheld.

The M0-M2 numbers have no JSON behind them -- the task log entries are their only source --
so they are listed at the end as *unchecked*, which is what the report's numbers appendix
says about them too.

Exit status 0 if the report is consistent, 1 if not.

Run:  <frontogenesis env python> check_report_numbers.py
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / 'data'
REPORT = HERE / 'frontogenesis_M3_interim.md'

S = json.loads((DATA / 'm3_closure_summary.json').read_text())
F = json.loads((DATA / 'm3_figs_summary.json').read_text())
L_ALL = ('0', '2', '4', '8')


def f3(x):
    return f'{x:.3f}'


def checks():
    """(label, [acceptable renderings]) for every JSON-backed number."""
    out = []
    per = {L: S['per_L'][L]['closure'] for L in L_ALL}

    d = S['declared']['criterion1']
    out.append(('criterion 1 residual tolerance', [f"{d['resid_ratio_max']:.2f}"]))
    out.append(('criterion 1 explained tolerance', [f"{d['explained_min']:.2f}"]))
    out.append(('criterion 1 slope tolerance', [f"{d['resid_slope_tol']:.2f}"]))
    c2 = S['declared']['criterion2']
    out.append(('criterion 2 slope band low', [f"{c2['slope'][0]:.2f}"]))
    out.append(('criterion 2 slope band high', [f"{c2['slope'][1]:.2f}"]))
    out.append(('criterion 2 correlation floor', [f"{c2['corr_min']:.2f}"]))
    out.append(('M1 baseline', [f"{S['declared']['baseline']:.3f}"]))
    out.append(('M1 baseline CI low', [f"{S['declared']['baseline_ci'][0]:.3f}"]))
    out.append(('M1 baseline CI high', [f"{S['declared']['baseline_ci'][1]:.3f}"]))

    for L in L_ALL:
        r = per[L]['residual']
        out.append((f'L={L} residual / measured', [f3(r['over_measured'])]))
        out.append((f'L={L} explained fraction',
                    [f3(r['explained']), f'{abs(r["explained"]):.3f}']))
        out.append((f'L={L} residual slope on 2F', [f3(r['slope_on_2F'])]))
        out.append((f'L={L} catch-all residual', [f3(per[L]['catchall']['over_measured'])]))
        t = per[L]['terms']
        for name in ('two_F', 'vertical', 'surface_flux'):
            out.append((f'L={L} {name} / measured', [f3(t[name]['over_measured'])]))
        out.append((f'L={L} subfilter / 2F', [f3(S['per_L'][L]['subfilter']['over_two_F'])]))
        e = per[L]['measured_coherence']
        out.append((f'L={L} rms(Eulerian - semi-Lagrangian) / measured',
                    [f3(e['rms_euler_minus_semilag']),
                     f"{e['rms_euler_minus_semilag']:.2f}"]))
        eu = S['per_L'][L]['euler']['front']
        out.append((f'L={L} criterion 2 slope', [f3(eu['ols'])]))
        out.append((f'L={L} criterion 2 correlation', [f3(eu['corr'])]))
        g = S['per_L'][L]['fig2b']
        out.append((f'L={L} partial corr residual-lap2b given hbl',
                    [f3(g['partial_res_lap_given_kpp'])]))
        out.append((f'L={L} partial corr residual-hbl given lap2b',
                    [f3(g['partial_res_kpp_given_lap'])]))
        k = per[L]['surface_flux_kpp']
        out.append((f'L={L} residual with the KPP depth scale',
                    [f3(k['residual_if_kpp'])]))

    out.append(('front pixels pooled', ['1,866,803']))
    out.append(('pairs', ['71']))
    return out


def main():
    if not REPORT.exists():
        print(f'MISSING: {REPORT}')
        return 1
    text = REPORT.read_text()
    flat = re.sub(r'[−–—]', '-', text)          # minus / en / em dash

    missing = []
    for label, renderings in checks():
        if not any(r in flat for r in renderings):
            missing.append((label, renderings))

    # the "Do not" list, made mechanical
    # A rounded slope can collide with a legitimate number (0.831 is both the L = 8 slope
    # and the L = 0 per-pair median residual), so flag a 3-dp match only when the line is
    # also talking about a slope; an unrounded 4-dp match is damning on its own.
    forbidden = []
    slope_words = re.compile(r'slope|regress|efficien', re.I)
    for L in L_ALL:
        v = S['per_L'][L]['closure']['measured_on_2F']['ols']
        for r, needs_context in ((f'{v:.4f}', False), (f'{v:.3f}', True),
                                 (f'{v:.2f}', True)):
            pat = re.compile(rf'(?<![\d.]){re.escape(r)}(?![\d])')
            for line in flat.splitlines():
                if pat.search(line) and (not needs_context or slope_words.search(line)):
                    forbidden.append((f'L={L} measured-on-2F slope (NOT QUOTED)', r))
                    break
    said = any(p in flat.lower() for p in
               ('not reported', 'not quoted', 'are withheld', 'is withheld'))

    figs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
    bad_figs = [f for f in figs if not (HERE / f).exists()]

    print(f'report      {REPORT.relative_to(HERE.parent)}  '
          f'{len(text.split())} words, {len(text.splitlines())} lines')
    print(f'checked     {len(checks())} JSON-backed numbers')
    print(f'figures     {len(figs)} referenced, {len(figs) - len(bad_figs)} present')

    ok = True
    if missing:
        ok = False
        print(f'\nMISSING from the report ({len(missing)}):')
        for label, r in missing:
            print(f'  - {label}: expected one of {r}')
    if forbidden:
        ok = False
        print(f'\nFORBIDDEN (the "Do not" list; M3-Q24 (a)):')
        for label, r in forbidden:
            print(f'  - {label} appears as {r!r}')
    if not said:
        ok = False
        print('\nFORBIDDEN: the report never says the slopes are withheld.')
    if bad_figs:
        ok = False
        print(f'\nMISSING figures ({len(bad_figs)}):')
        for f in bad_figs:
            print(f'  - {f}')

    print('\nUnchecked by this script (log entries are their only source): the M0 stencil '
          'ratio,\nthe M1 sub-filter fractions, the M2 store counts and the V3 temporal '
          'stability band.\nSee the report\'s numbers appendix, which marks them.')
    print('\nOK' if ok else '\nFAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
