"""Assert that ``frontogenesis_report.md`` (M5, prompt 6 task 2) matches the summaries.

The sibling of ``check_report_numbers.py`` (the interim, frozen) and
``check_digging_numbers.py`` (the digging note).  It re-uses their lists of
JSON-backed numbers -- the M5 report grows from both documents -- drops, by
name, the ones the M5 report deliberately does not carry, and adds the M4
numbers from the five object summaries.  Every remaining number is re-read
from ``../data/``, formatted the way the report prints it, and required to
appear verbatim.  No data store is opened (M3-Q23 (a)).

The "Do not" list of prompt 6, made mechanical:

* the four measured-vs-predicted slopes (and their other estimators) may
  appear **only in Appendix C** (M5-Q1 (b)), and the body must say they are
  withheld;
* Appendix C must say, in words, that they are not an efficiency;
* any line quoting M4's hourly correlation / amplitude ratio (0.951 / 1.008)
  must say it is not an efficiency;
* any line saying criterion 4 is met or passes must also say it failed as
  pre-registered;
* "damping" may appear only in a line that negates it;
* the front matter (authors, dated status, circulation) and the three
  glossary tables (with units) must be present, and every figure must exist
  and be one that ``py/make_figs.py --report`` redraws.

Exit status 0 if the report is consistent, 1 if not.

Run:  <frontogenesis env python> check_frontogenesis_report.py
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / 'data'
REPORT = HERE / 'frontogenesis_report.md'
sys.path.insert(0, str(HERE))

import check_report_numbers as interim                      # noqa: E402
import check_digging_numbers as digging                     # noqa: E402

S = interim.S
L_ALL = ('0', '2', '4', '8')

#: interim numbers the M5 report does not carry (the trim M3-Q19 / M5-Q2 (a) asked for);
#: they remain in the interim report, which is not rewritten
INTERIM_DROPPED = {
    'case: hours each way', 'case: window start', 'case: window end',
    'case: displacement median cells', 'case: displacement p90 cells',
    'case: displacement max cells', 'case: net displacement cells',
    'case: spread start cells', 'case: spread end cells', 'case: G peak',
    'case: G peak hour', 'case: trajectory p99 return error',
    'case: rate check median ratio', 'case: subfilter / measured',
    'case sflux: S / measured, forward day', 'case sflux: nonsolar part over S',
    'case sflux: shortwave part over S', 'case sflux: freshwater part over S',
    'case sflux: fitted factor on the imbalance',
    'case sflux: with S on h_bl over observed change',
    'case sflux: most frontolytic hour', 'case sflux: least frontolytic hour',
}

#: digging numbers the M5 report carries (the note has the rest)
DIGGING_KEPT_PREFIX = (
    'product ', 'declared tolerance', 'tracer L=', 'model excess over the floor L=4',
    'model excess over the floor L=8', 'tracer interpolation effect', 'tile 329 L=',
    'parcels ', 'selection G_prev L=0', 'selection persist L=0', 'split L=0 nonsolar',
    'case seed_minus_12h budget over observed',
)
DIGGING_KEPT_RE = (
    r'controls L=\d real residual', r'controls L=\d permutation median',
    r'controls L=\d gap in permutation sd', r'split L=[08] rms fit over I',
)
DIGGING_DROPPED = {                       # under a kept prefix, but not carried
    'tile 329 L=0 explained', 'tile 329 L=2 explained', 'tile 329 L=4 explained',
    'tile 329 L=8 explained', 'tile 329 L=0 residual slope', 'tile 329 L=2 residual slope',
    'tile 329 L=4 residual slope', 'tile 329 L=8 residual slope',
    'tile 329 L=0 Eulerian gap', 'tile 329 L=2 Eulerian gap', 'tile 329 L=4 Eulerian gap',
    'tile 329 L=8 Eulerian gap', 'tile 329 L=0 G p90 ratio', 'tile 329 L=2 G p90 ratio',
    'tile 329 L=4 G p90 ratio', 'tile 329 L=8 G p90 ratio',
    'tracer L=0 piecewise residual', 'tracer L=2 piecewise residual',
    'tracer L=4 piecewise residual', 'tracer L=8 piecewise residual',
    'tracer L=0 Eulerian gap', 'tracer L=2 Eulerian gap', 'tracer L=4 Eulerian gap',
    'tracer L=8 Eulerian gap', 'tracer L=4 residual slope', 'tracer L=8 residual slope',
    'product L=2 explained', 'product L=4 explained', 'product L=8 explained',
    'product L=0 explained',               # printed as +0.344 by the interim's rule instead
    'box-mean speed', 'box-mean rms about the mean',
    'tracer L=0 measured on 2F (calibration)',   # required in Appendix C instead
}


def _load(name):
    return json.loads((DATA / name).read_text())


def f2(x):
    return f'{x:.2f}'


def f3(x):
    return f'{x:.3f}'


def sci(x, nd=2):
    m, e = f'{x:.{nd}e}'.split('e')
    return f'{m} \\times 10^{{{int(e)}}}'


def m4_checks():
    """The M4 numbers of §9 (and §1), from the five object summaries."""
    out = []
    fd, ts = _load('m4_find_done.json'), _load('m4_track_summary.json')
    rc, q, st = _load('m4_recon.json'), _load('m4_quality.json'), _load('m4_strength.json')
    for s in (fd, ts, q, st):
        assert s['status'] == 'ok', 'an M4 product is not ok'
    # --- finding
    out.append(('m4 find: fronts per hour', [f"{fd['n_fronts']['min']}–{fd['n_fronts']['max']}"]))
    out.append(('m4 find: median fronts', [f"median {fd['n_fronts']['median']:.0f}"]))
    out.append(('m4 find: pixels per hour', [f"{fd['n_pixels']['min']:,}–{fd['n_pixels']['max']:,}"]))
    out.append(('m4 find: overlap with the p90 pool',
                [f"{fd['frac_on_p90']['min']:.2f}–{fd['frac_on_p90']['max']:.2f}"]))
    # --- tracking
    tf, tp = ts['tracks_flow'], ts['tracks_plain']
    sizes = [t['anchor_size'] for t in tf]
    out.append(('m4 track: anchor sizes', [f'{min(sizes)}–{max(sizes)} pixels']))
    assert all(t['n_found'] == 71 and t['n_gaps_in_span'] == 0 for t in tf)
    out.append(('m4 track: all ten found at every step', ['71 of 71 hours']))
    out.append(('m4 track: span', [f"{ts['min_span_h_flow']:.0f} hours"]))
    n_changed = sum(1 for c in ts['comparison'] if c['different'] or c['only_a'] or c['only_b'])
    out.append(('m4 track: tracks changed by the flow', [f'on {n_changed} of {ts["n_tracks"]} tracks']))
    out.append(('m4 track: links chosen differently', [f"{ts['n_links_different']} differ"]))
    out.append(('m4 track: links only the flow walk made', [f"made {ts['n_links_only_flow']} links"]))
    assert ts['n_links_only_plain'] == 0
    lost = sorted((t['n_found'] for t in tp if t['n_found'] < 71), reverse=True)
    out.append(('m4 track: the centroid tracker\'s two short tracks',
                [f'found at {lost[0]} and {lost[1]} of 71 hours']))
    fl = ts['flow']
    out.append(('m4 flow: displacement', [f"{fl['disp_median_cells']['median']:.2f} cells per hour"]))
    out.append(('m4 flow: IoU advected', [f3(fl['iou_advected']['median'])]))
    out.append(('m4 flow: IoU persistence', [f3(fl['iou_persist']['median'])]))
    out.append(('m4 flow: beats persistence', [f"{fl['n_transitions_advected_beats_persistence']} of 70"]))
    out.append(('m4 flow: beats the wrong sign', [f"{fl['n_transitions_advected_beats_wrong_sign']} of 70"]))
    # --- quality
    qs, rows = q['flow']['summary'], q['flow']['rows']
    out.append(('m4 quality: centroid median', [f2(qs['centroid_diff_cells']['median'])]))
    out.append(('m4 quality: centroid p90', [f"{qs['centroid_diff_cells']['p90']:.1f}"]))
    out.append(('m4 quality: unflagged', [f"{qs['centroid_diff_no_flag']['n']} links",
                                           ]))
    out.append(('m4 quality: unflagged median', [f2(qs['centroid_diff_no_flag']['median'])]))
    out.append(('m4 quality: flagged median', [f"{qs['centroid_diff_flagged']['median']:.1f}"]))
    p90 = [r['mask_dist_p90'] for r in rows]
    out.append(('m4 quality: within one cell', [f"{100 * sum(p <= 1 for p in p90) / len(p90):.0f} %"]))
    out.append(('m4 quality: within two cells', [f"{100 * sum(p <= 2 for p in p90) / len(p90):.0f} %"]))
    out.append(('m4 quality: majority overlap', [f"{100 * qs['frac_chosen_is_majority']:.1f} %"]))
    out.append(('m4 quality: flags', [f"{qs['n_split_or_merge']} of {qs['n_links']} links"]))
    out.append(('m4 quality: splits and merges', [f"{qs['n_split']} splits and {qs['n_merge']} merges"]))
    # --- reconciliation
    tol = rc['variants']['band1_centred']['summary']['tolerance']
    out.append(('m4 recon: tolerance', [f"{tol['corr_min']:.1f} and {tol['rms_ratio_max']:.1f}"]))
    for name in ('skeleton_forward', 'skeleton_centred', 'band1_forward', 'band1_centred'):
        s = rc['variants'][name]['summary']
        mv = s['matched_vs_store']
        out.append((f'm4 recon {name}: links', [f"| {s['n_links']} |"]))
        out.append((f'm4 recon {name}: corr', [f3(mv['corr'])]))
        out.append((f'm4 recon {name}: rms ratio', [f3(mv['rms_ratio'])]))
        out.append((f'm4 recon {name}: tracks passing', [f"{s['n_tracks_passing']} of 10"]))
    assert not rc['variants']['skeleton_forward']['summary']['passes']
    assert rc['variants']['band1_centred']['summary']['passes']
    se = [v['summary']['set_evolution_over_matched'] for v in rc['variants'].values()]
    out.append(('m4 recon: set-evolution range', [f'{min(se):.1f}–{max(se):.1f}']))
    tm = rc['variants']['band1_centred']['summary']['two_F_vs_matched']
    out.append(('m4 recon: 2F vs matched corr (not an efficiency)', [f3(tm['corr'])]))
    out.append(('m4 recon: 2F vs matched amplitude (not an efficiency)', [f3(tm['amp_ratio'])]))
    # --- strength
    po = st['summary']['pooled']
    for key, lab in (('frac_frontogenetic_matched', 'matched'), ('frac_frontogenetic_2F', '2F'),
                     ('sign_agreement', 'sign agreement')):
        v = po[key]
        out.append((f'm4 strength: {lab}', [f"{v['value']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]"]))
    out.append(('m4 strength: links', [f"{po['n_links']} links"]))
    f9, f8 = st['fig09'], st['fig08']
    out.append(('m4 strength: fronts with positive integral', [f"{f9['n_fronts_dG_pred_positive']} of 10"]))
    out.append(('m4 strength: fronts with positive material change',
                [f"{f9['n_fronts_dG_matched_positive']} of 10"]))
    ratios = [f['dG_matched'] / f['dG_pred_2F'] for f in st['summary']['per_front']]
    for i, r in enumerate(ratios):
        out.append((f'm4 strength: front {i} ratio (not an efficiency)', [f2(r)]))
    assert all(f['lifetime_h'] == 70.0 for f in st['summary']['per_front'])
    w = sorted(s['width_km'] for k in st['series'] for s in st['series'][k]
               if s.get('width_km') is not None and s['width_km'] == s['width_km'])
    d = sorted(s['delta_b'] for k in st['series'] for s in st['series'][k]
               if s.get('delta_b') is not None and s['delta_b'] == s['delta_b'])
    med = lambda a: (a[(len(a) - 1) // 2] + a[len(a) // 2]) / 2          # noqa: E731
    out.append(('m4 strength: width median', [f'{med(w):.1f} km']))
    out.append(('m4 strength: delta b median', [sci(med(d))]))
    out.append(('m4 fig08: anchor size', [f'{max(sizes)}-pixel']))
    out.append(('m4 fig08: material end', [sci(f8['G_matched_end'])]))
    out.append(('m4 fig08: predicted end', [sci(f8['G_pred_end'])]))
    out.append(('m4 fig08: anchor', [sci(f8['G_label_anchor'])]))
    out.append(('m4 fig08: budget end', [sci(f8['G_budget_end'])]))
    out.append(('m4 fig08: 2F positive', [f"{100 * f8['frac_2F_positive']:.0f} %"]))
    out.append(('m4 fig08: matched positive', [f"{100 * f8['frac_matched_positive']:.0f} %"]))
    out.append(('m4 fig08: delta b', [sci(f8['delta_b_median'], 1)]))
    out.append(('m4 fig08: width', [f"{f8['width_km_median']:.1f} km"]))
    return out


def appendix_c_checks():
    """The withheld slopes, required in Appendix C (M5-Q1 (b))."""
    out = []
    R = _load('m3_tracer_summary.json')
    for L in L_ALL:
        q = S['per_L'][L]['slopes_not_quoted']
        assert S['per_L'][L]['slopes_quotable'] is False
        est = dict(q['primary']['estimates'], chain=q['variants']['chain'])
        for name in ('ols', 'trimmed', 'tls', 'chain'):
            v = est[name]
            out.append((f'appendix C L={L} {name}',
                        [f"{v['value']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"]))
        out.append((f'appendix C L={L} tracer calibration',
                    [f3(R['columns']['linear']['per_L'][L]['meas_on_2F']['ols'])]))
    v = S['per_L']['0']['slopes_not_quoted']['primary_3h']
    out.append(('appendix C 3-h interval at L=0', [f"[{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"]))
    return out


def body_checks():
    out = [c for c in interim.checks() if c[0] not in INTERIM_DROPPED]
    for lab, r in digging.checks():
        keep = lab.startswith(DIGGING_KEPT_PREFIX) or any(re.fullmatch(p, lab) for p in DIGGING_KEPT_RE)
        if keep and lab not in DIGGING_DROPPED:
            out.append((lab, r))
    for L in L_ALL:
        out.append((f'L={L} residual over 2F', [f2(S['per_L'][L]['closure']['residual']['over_two_F'])]))
    return out + m4_checks()


def slope_renderings():
    """Every withheld slope value, with whether a bare match is damning."""
    out = []
    for L in L_ALL:
        q = S['per_L'][L]['slopes_not_quoted']
        est = dict(q['primary']['estimates'], chain=q['variants']['chain'])
        for name, v in est.items():
            x = v['value']
            out += [(f'L={L} {name}', f'{x:.4f}', False), (f'L={L} {name}', f'{x:.3f}', True)]
        x = est['ols']['value']
        out.append((f'L={L} ols', f'{x:.2f}', True))
    return out


def main():
    if not REPORT.exists():
        print(f'MISSING: {REPORT}')
        return 1
    text = REPORT.read_text()
    flat = re.sub(r'[−]', '-', text)                         # the minus sign only
    flat_all = re.sub(r'[−–—]', '-', text)
    head, sep, app_c = flat_all.partition('## Appendix C')
    problems = []

    # numbers: the body's anywhere before Appendix C; Appendix C's inside it
    for lab, rr in body_checks():
        rr2 = [re.sub(r'[−–—]', '-', r) for r in rr]
        if not any(r in head for r in rr2):
            problems.append(f'MISSING (body) {lab}: expected one of {rr}')
    if not sep:
        problems.append('MISSING: Appendix C')
    for lab, rr in appendix_c_checks():
        if not any(r in app_c for r in rr):
            problems.append(f'MISSING (Appendix C) {lab}: expected one of {rr}')

    # the withheld slopes: nowhere before Appendix C
    slope_words = re.compile(r'slope|efficien', re.I)
    seen = set()
    for lab, r, needs_context in slope_renderings():
        pat = re.compile(rf'(?<![\d.]){re.escape(r)}(?![\d])')
        for line in head.splitlines():
            if pat.search(line) and (not needs_context or slope_words.search(line)):
                if (lab, r) not in seen:
                    seen.add((lab, r))
                    problems.append(f'FORBIDDEN outside Appendix C: {lab} slope appears as {r!r}')
                break
    if not any(p in head.lower() for p in ('are withheld', 'withheld from the body')):
        problems.append('FORBIDDEN: the body never says the slopes are withheld')
    if not re.search(r'none of the numbers in this appendix is a frontogenesis efficiency', app_c):
        problems.append('FORBIDDEN: Appendix C does not open with its caveat')
    if 'not met as written' not in app_c:
        problems.append('MISSING: Appendix C does not judge the second null condition')

    # the M4 rules and the damping rule, line by line
    neg = re.compile(r'\b(no|not|none|never|neither|nor|withdrawn|without)\b', re.I)
    for line in flat_all.splitlines():
        if re.search(r'(?<![\d.])0\.951(?!\d)', line) and re.search(r'(?<![\d.])1\.008(?!\d)', line) \
                and 'not an efficiency' not in line:
            problems.append(f'FORBIDDEN: 0.951 / 1.008 without "not an efficiency": {line[:80]}...')
        if re.search(r'criterion 4', line, re.I) and re.search(r'\b(met|pass(es|ed)?)\b', line) \
                and 'pre-registered' not in line:
            problems.append(f'FORBIDDEN: criterion 4 "met" without "pre-registered": {line[:80]}...')
        if 'damping' in line.lower() and not neg.search(line):
            problems.append(f'FORBIDDEN: "damping" in a line that does not negate it: {line[:80]}...')
        if re.search(r'efficienc', line, re.I) and not neg.search(line):
            problems.append(f'FORBIDDEN: "efficiency" in a line that does not negate it: {line[:80]}...')

    # front matter, glossary, required statements
    for lab, pat in (('authors line', r'^\*\*Authors:\*\* J\. X\. Prochaska, with Claude'),
                     ('dated status line', r'^\*\*Status:\*\* .*\d{1,2} \w+ 2026'),
                     ('circulation note', r'^\*\*Circulation:\*\* .*not for citation'),
                     ('glossary: physics with units', r'^\| term \| meaning \| units \|'),
                     ('glossary: method', r'^\*\*Method\*\*'),
                     ('glossary: data and code', r'^\*\*Data and code\*\*'),
                     ('the tolerance and its date', r'0\.50.*fixed 7 October 2026'),
                     ('the null-result section', r'^## 11\. What would have made this a null result'),
                     ('the limitations section', r'^## 10\. Limitations')):
        if not re.search(pat, text, re.M):
            problems.append(f'MISSING: {lab}')

    # figures: present, and redrawn by make_figs.py --report
    figs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
    for f in figs:
        p = (HERE / f)
        if not p.exists():
            problems.append(f'MISSING figure: {f}')
        if not (f.startswith('figs/') or re.match(r'\.\./figs/(fig|V)', f)):
            problems.append(f'figure not redrawn by make_figs.py --report: {f}')

    n_body, n_c = len(body_checks()), len(appendix_c_checks())
    print(f'report      report/{REPORT.name}  '
          f'{len(text.split())} words, {len(text.splitlines())} lines')
    print(f'checked     {n_body} JSON-backed numbers in the body, {n_c} in Appendix C')
    print(f'figures     {len(figs)} referenced')
    if problems:
        print(f'\n{len(problems)} problem(s):')
        for p in problems:
            print('  - ' + p)
        print('\nFAILED')
        return 1
    print('\nUnchecked by this script (log entries are their only source): the rows marked '
          '"log only"\nin the report\'s Appendix B.')
    print('\nOK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
