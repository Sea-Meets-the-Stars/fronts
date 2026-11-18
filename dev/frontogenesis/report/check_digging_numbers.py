"""Assert that the digging note's JSON-backed numbers match the digging summaries.

``check_report_numbers.py`` guards the interim report; this is its sibling for
``frontogenesis_M3_digging.md`` (M3-digging task 7; M3D-Q4 (a): a separate checker, the
interim's frozen).  Every number in the note that has a JSON behind it is re-read from the
digging summaries in ``../data/`` -- the controls, the case re-seeds, the term tests, the
tracer, the second tile, the parcels -- and from the product's ``m3_closure_summary.json``
for the comparison columns, formatted the way the note formats it, and required to appear
verbatim in the text.

The "Do not" list, made mechanical, as for the interim:

* the product's four measured-vs-predicted slopes (``measured_on_2F.ols``) must **not**
  appear in a sentence about a slope;
* the tracer's slopes may appear only in a sentence that also says "tracer" or
  "calibration";
* the note must say, in words, that the product's slopes are withheld.

Exit status 0 if the note is consistent, 1 if not.

Run:  <frontogenesis env python> check_digging_numbers.py
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / 'data'
NOTE = HERE / 'frontogenesis_M3_digging.md'
L_ALL = ('0', '2', '4', '8')

P = json.loads((DATA / 'm3_closure_summary.json').read_text())
C = json.loads((DATA / 'm3_controls_summary.json').read_text())
K = json.loads((DATA / 'm3_case_controls.json').read_text())
T = json.loads((DATA / 'm3_term_tests_summary.json').read_text())
R = json.loads((DATA / 'm3_tracer_summary.json').read_text())
T2 = json.loads((DATA / 'm3_tile2_summary.json').read_text())
PA = json.loads((DATA / 'm3_parcels_summary.json').read_text())


def f3(x):
    return f'{x:.3f}'


def f2(x):
    return f'{x:.2f}'


def signed(x, nd=3):
    return f'{x:+.{nd}f}'


def checks():
    out = []
    # --- the product's columns -------------------------------------------------
    for L in L_ALL:
        c = P['per_L'][L]['closure']
        out.append((f'product L={L} residual', [f3(c['residual']['over_measured'])]))
        out.append((f'product L={L} catch-all', [f3(c['catchall']['over_measured'])]))
        out.append((f'product L={L} Eulerian gap', [f3(c['measured_coherence']['rms_euler_minus_semilag'])]))
        out.append((f'product L={L} residual slope on 2F', [signed(c['residual']['slope_on_2F'])]))
        out.append((f'product L={L} explained', [signed(c['residual']['explained'])]))
    out.append(('declared tolerance', [f2(P['declared']['criterion1']['resid_ratio_max'])]))
    # --- the controls ------------------------------------------------------------
    for L in L_ALL:
        d = C['per_L'][L]
        r, ctl, g = d['real'], d['controls'], d['gaps']['resid_over_meas']
        pm, rl = ctl['permutation']['dist'], ctl['roll']['dist']
        out.append((f'controls L={L} real residual', [f3(r['resid_over_meas'])]))
        out.append((f'controls L={L} permutation median', [f3(pm['resid_over_meas']['median'])]))
        out.append((f'controls L={L} permutation p5', [f3(pm['resid_over_meas']['p5'])]))
        out.append((f'controls L={L} 24-h lag', [f3(ctl['lag24']['resid_over_meas'])]))
        out.append((f'controls L={L} roll median', [f3(rl['resid_over_meas']['median'])]))
        out.append((f'controls L={L} roll p5', [f3(rl['resid_over_meas']['p5'])]))
        out.append((f'controls L={L} independent reference', [f3(r['independent_reference'])]))
        out.append((f'controls L={L} gap in permutation sd', [f'{abs(g["permutation"]["in_control_sd"]):.0f}']))
        out.append((f'controls L={L} gap in real half-widths', [f'{abs(g["permutation"]["in_real_halfwidth"]):.1f}']))
        out.append((f'controls L={L} explained real', [signed(r['explained'])]))
        out.append((f'controls L={L} explained permutation', [signed(pm['explained']['median'])]))
        out.append((f'controls L={L} residual slope real', [signed(r['resid_on_2F'])]))
        out.append((f'controls L={L} residual slope permutation', [signed(pm['resid_on_2F']['median'])]))
        for sel in ('G_prev', 'persist', 'sigma_mag'):
            out.append((f'selection {sel} L={L}', [f3(d['selections'][sel]['resid_over_meas'])]))
        out.append((f'blocks L={L} 3h width ratio', [f2(d['blocks']['3h']['width_over_1h']['resid_over_meas'])]))
        out.append((f'blocks L={L} 24h width ratio', [f2(d['blocks']['24h']['width_over_1h']['resid_over_meas'])]))
    out.append(('selection p80 L=0', [f3(C['per_L']['0']['selections']['p80']['resid_over_meas'])]))
    out.append(('selection p95 L=0', [f3(C['per_L']['0']['selections']['p95']['resid_over_meas'])]))
    out.append(('skill passed everywhere', ['every scale'] if all(C['skill_summary'].values()) else ['NOT AT EVERY SCALE']))
    # --- the case re-seeded --------------------------------------------------------
    for name in ('seed_minus_12h', 'seed_persist'):
        v = K['runs'][name]
        out.append((f'case {name} change factor', [f2(v['change_factor'])]))
        out.append((f'case {name} control change factor', [f2(v['control_change_factor'])]))
        out.append((f'case {name} 2F over observed', [signed(v['predicted_over_observed_change'], 2)]))
        out.append((f'case {name} budget over observed', [signed(v['budget_over_observed_change'], 2)]))
    o = K['original']
    out.append(('case original change factor', [f2(o['change_factor'])]))
    out.append(('case original control', [f2(o['control_change_factor'])]))
    out.append(('case original budget over observed', [f2(o['budget_over_observed_change'])]))
    # --- the term tests ------------------------------------------------------------
    sp = T['split']['per_L']
    ident = max(max(sp[L]['identity']['split_max_rel_rms'].values()) for L in L_ALL)
    out.append(('split identity max', [f'{ident:.1e}']))
    for L in L_ALL:
        d = sp[L]['depth']['top']
        out.append((f'split L={L} nonsolar / S', [f3(d['components']['nonsolar']['rms_over_S'])]))
        f = d['fit3']
        for c in ('nonsolar', 'shortwave', 'freshwater'):
            out.append((f'split L={L} multiplier {c}', [signed(f['multipliers'][c]['value'], 2)]))
        out.append((f'split L={L} rms fit over I', [f3(f['rms_over_I']['fit'])]))
        assert not d['cancellation']['passed'], f'cancellation passed at L={L}: the note says it does not'
    vs = T['vertical']['summary']
    out.append(('V cell2/top rms min', [f2(vs['cell2_over_top_rms']['min'])]))
    out.append(('V cell2/top rms max', [f2(vs['cell2_over_top_rms']['max'])]))
    out.append(('V corr min', [signed(vs['cell2_on_top_corr']['min'])]))
    out.append(('V corr max', [signed(vs['cell2_on_top_corr']['max'])]))
    out.append(('V same sign', [f"{vs['same_sign']} of {vs['n_rows']}"]))
    # --- the tracer ------------------------------------------------------------------
    for L in L_ALL:
        lin, pc = R['columns']['linear']['per_L'][L], R['columns']['piecewise']['per_L'][L]
        out.append((f'tracer L={L} linear residual', [f3(lin['resid_over_meas'])]))
        out.append((f'tracer L={L} piecewise residual', [f3(pc['resid_over_meas'])]))
        out.append((f'tracer L={L} Eulerian gap', [f3(lin['euler_gap'])]))
        out.append((f'tracer L={L} residual slope', [signed(lin['resid_on_2F'])]))
    out.append(('tracer L=0 measured on 2F (calibration)', [f3(R['columns']['linear']['per_L']['0']['meas_on_2F']['ols'])]))
    ie = max(abs(v['difference']) for v in R['interpolation_effect'].values())
    out.append(('tracer interpolation effect', [f'{ie:.4f}']))
    fp = R['columns']['linear']['checks']['checks_2_to_4']['front_population']
    out.append(('tracer G p90 ratio 24 h', [f3(fp['24']['ratio']['90'])]))
    out.append(('tracer G p90 ratio 48 h', [f3(fp['48']['ratio']['90'])]))
    for L in ('2', '4', '8'):
        ex = P['per_L'][L]['closure']['catchall']['over_measured'] - R['columns']['linear']['per_L'][L]['resid_over_meas']
        out.append((f'model excess over the floor L={L}', [f3(ex)]))
    th = R['declared']['reading']['thresholds']
    out.append(('tracer thresholds', [f2(th['hi']), f2(th['lo'])]))
    # --- the second tile ---------------------------------------------------------------
    qa = T2['orientation_qa']
    out.append(('tile 329 QA slope', [f'{qa["slope"]:.4f}']))
    for L in L_ALL:
        row = T2['compare']['per_L'][L]
        t = row['tile']
        out.append((f'tile 329 L={L} catch-all', [f3(t['catchall_over_meas'])]))
        out.append((f'tile 329 L={L} explained', [signed(t['explained'])]))
        out.append((f'tile 329 L={L} residual slope', [signed(t['resid_on_2F'])]))
        out.append((f'tile 329 L={L} Eulerian gap', [f3(t['euler_gap'])]))
        out.append((f'tile 329 L={L} G p90 ratio', [f2(row['G_p90_ratio_tile_over_product'])]))
    out.append(('tile 329 n_front ratio', [f'{T2["compare"]["per_L"]["0"]["n_front_ratio"]:.1f}']))
    # --- the parcels -------------------------------------------------------------------
    sd = PA['seed']
    out.append(('parcels speed', [f2(sd['speed_ms']['median'])]))
    out.append(('parcels cross-front speed', [f3(sd['cross_ms']['median'])]))
    out.append(('parcels along-front speed', [f3(sd['along_ms']['median'])]))
    out.append(('parcels front width km', [f'{sd["front_width_km"]:.1f}']))
    h = PA['hours_from_seed']
    n16 = h.index(16.0)
    out.append(('parcels on front at +16 h', [f2(PA['selected']['on_front'][n16])]))
    bm = PA['box_mean']
    out.append(('box-mean speed', [f2(bm['mean_speed_ms'])]))
    out.append(('box-mean rms about the mean', [f2(bm['rms_about_mean_ms'])]))
    return out


def main():
    if not NOTE.exists():
        print(f'MISSING: {NOTE}')
        return 1
    text = NOTE.read_text()
    flat = re.sub(r'[−–—]', '-', text)
    missing = [(lab, r) for lab, r in checks() if not any(x in flat for x in r)]

    forbidden = []
    slope_words = re.compile(r'slope|efficien', re.I)
    tracer_words = re.compile(r'tracer|calibration', re.I)
    for L in L_ALL:
        v = P['per_L'][L]['closure']['measured_on_2F']['ols']
        for r, needs_context in ((f'{v:.4f}', False), (f'{v:.3f}', True), (f'{v:.2f}', True)):
            pat = re.compile(rf'(?<![\d.]){re.escape(r)}(?![\d])')
            for line in flat.splitlines():
                if pat.search(line) and (not needs_context or slope_words.search(line)):
                    forbidden.append((f'product L={L} measured-on-2F slope (WITHHELD)', r))
                    break
    for vel in ('linear', 'piecewise'):
        for L in L_ALL:
            v = R['columns'][vel]['per_L'][L]['meas_on_2F']['ols']
            pat = re.compile(rf'(?<![\d.]){re.escape(f"{v:.3f}")}(?![\d])')
            for line in flat.splitlines():
                if pat.search(line) and slope_words.search(line) and not tracer_words.search(line):
                    forbidden.append((f'tracer {vel} L={L} slope without "tracer"/"calibration"', f'{v:.3f}'))
                    break
    said = any(p in flat.lower() for p in ('withheld', 'not quoted', 'not reported'))
    figs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
    bad_figs = [f for f in figs if not (HERE / f).exists()]

    print(f'note        {NOTE.relative_to(HERE.parent)}  {len(text.split())} words, {len(text.splitlines())} lines')
    print(f'checked     {len(checks())} JSON-backed numbers')
    print(f'figures     {len(figs)} referenced, {len(figs) - len(bad_figs)} present')
    ok = True
    if missing:
        ok = False
        print(f'\nMISSING from the note ({len(missing)}):')
        for lab, r in missing:
            print(f'  - {lab}: expected one of {r}')
    if forbidden:
        ok = False
        print('\nFORBIDDEN:')
        for lab, r in forbidden:
            print(f'  - {lab} appears as {r!r}')
    if not said:
        ok = False
        print('\nFORBIDDEN: the note never says the product\'s slopes are withheld.')
    if bad_figs:
        ok = False
        print(f'\nMISSING figures: {bad_figs}')
    print('\nOK' if ok else '\nFAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
