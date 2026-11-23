"""Track analysis, "Matched parcels" task 1 (2026-11-18): one tracked front, step by step.

For every step ``t`` of a flow-informed track this module shows two pixel sets and three
rates:

* the pixels of the **chosen label** at ``t`` and the pixels of the **previous label advected
  by the flow** to ``t`` (``tracking.Flow.advect`` on the bool mask -- task 2's prediction);
* the hourly rates on the **matched parcels** of step ``t`` exactly as M4 built them
  (``tracking.matched_link(centred=True, band=1)``, M4-Q3 (a)): ``2F``, the stored
  ``DGDt_semilag`` and the matched rate, here also kept **per parcel**.

Nothing in ``tracking.py`` is changed; :func:`parcel_link` repeats ``matched_link``'s centred
construction parcel by parcel and :func:`analyse_track` asserts that its means equal
``matched_link``'s row.

    cd dev/frontogenesis/py
    PY=/home/xavier/miniconda3/envs/frontogenesis/bin/python
    $PY matched_parcels.py                          # track 0 = label 81 at pair 35
    $PY matched_parcels.py --track 3                # another of the ten M4 tracks
    $PY matched_parcels.py --label 100              # the same, by anchor label
    $PY matched_parcels.py --label 57 --anchor-step 20   # not an M4 track: followed on the fly

Outputs (``NAME = matched_parcels_track_<label>``, with ``_pair<step>`` appended when the
anchor is not the middle pair): ``tracked/NAME.md``, ``tracked/figs/NAME_*.png`` and
``data/NAME.json`` (every number in the markdown).
"""
import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tracking as tr                                           # noqa: E402
from fronts import front_tracking as ft                         # noqa: E402

DATA_DIR = tr.DATA_DIR
OUT_DIR = HERE.parent / 'tracked'
BAND = tr.BAND                  # 1: M4-Q3 (a)
DT = tr.DT
UTC_OFFSET = -8.0               # local solar time = UTC - 8 h (figures._lst); pair s is at s + 0.5 h UTC
UNIT = 1e-19                    # rates are printed in units of 1e-19 s^-5
PANELS_PER_PAGE = 8
STRUCT = np.ones((3, 3), bool)


# ---------------------------------------------------------------------------
# the track
# ---------------------------------------------------------------------------
def get_track(labels, flow, *, track=None, label=None, anchor_step=None, tracks_path=None,
              log=print):
    """``(Track, index or None, source)``.  ``track`` is an index into the flow-informed tracks
    of ``m4_tracks.json``; otherwise the stored track with this anchor ``label`` (and
    ``anchor_step``, default the middle pair) is used, and when there is none the front is
    followed here with ``tracking.follow_flow`` and M4's settings."""
    from m4_track import TRACKS_PATH
    path = Path(tracks_path or TRACKS_PATH)
    stored = tr.tracks_from_json(json.load(open(path))['flow']) if path.exists() else []
    if track is not None:
        return stored[int(track)], int(track), f'{path.name} flow track {int(track)}'
    if label is None:
        return stored[0], 0, f'{path.name} flow track 0'
    labels_at, keys = tr.labels_at_factory(labels)
    step = len(keys) // 2 if anchor_step is None else int(anchor_step)
    for k, t in enumerate(stored):
        if t.anchor.label == int(label) and t.anchor.step == step:
            return t, k, f'{path.name} flow track {k}'
    if not (labels_at(step) == int(label)).any():
        raise ValueError(f'label {label} does not exist at pair {step}')
    log(f'label {label} at pair {step} is not an M4 track: following it now (follow_flow)')
    anchor = ft.anchor_at(labels_at(step), step, int(label))
    t = tr.follow_flow(labels_at, keys, anchor, flow, km_per_px=flow.km_per_px,
                       weights=dict(tr.FLOW_WEIGHTS))
    t.anchor_size = int((labels_at(step) == int(label)).sum())
    return t, None, 'followed here: tracking.follow_flow, FLOW_WEIGHTS, flow km_per_px'


def out_name(track, n_steps):
    name = f'matched_parcels_track_{track.anchor.label}'
    return name if track.anchor.step == n_steps // 2 else f'{name}_pair{track.anchor.step}'


# ---------------------------------------------------------------------------
# the two pixel sets
# ---------------------------------------------------------------------------
def pixel_sets(track, step, labels, flow):
    """The chosen label at ``step`` (``chosen``) and the label of the previous found step in
    TIME advected to ``step`` by the flow (``advected``; ``None`` at the track's first step).
    Counts: ``n_both`` (same cell), ``n_chosen_only``, ``n_advected_only``, and the same two
    with one cell of tolerance (``n_gained`` = chosen pixels farther than one cell from every
    advected pixel, ``n_lost`` the reverse).  ``split`` / ``merge`` follow task 4's test
    (``SPLIT_MIN_PIXELS``) but in time order: a split is the advected previous label
    overlapping >= 2 labels at ``step``; a merge is the chosen label advected back
    overlapping >= 2 labels at the previous step."""
    keys = list(labels)
    C = labels[keys[step]] == track.labels[step]
    earlier = [p for p in track.steps() if p < step]
    out = dict(step=int(step), time=keys[step], label=int(track.labels[step]), chosen=C,
               npix=int(C.sum()), advected=None, prev_step=None, prev_label=None)
    if not earlier:
        return out
    p = max(earlier)
    P = labels[keys[p]] == track.labels[p]
    A = tr._advect_chain(flow, P, p, step)
    fwd = tr._overlap_labels(A, labels[keys[step]])
    bwd = tr._overlap_labels(tr._advect_chain(flow, C, step, p), labels[keys[p]])
    out.update(advected=A, prev_step=int(p), prev_label=int(track.labels[p]),
               prev_npix=int(P.sum()), n_advected=int(A.sum()),
               n_both=int((C & A).sum()), n_chosen_only=int((C & ~A).sum()),
               n_advected_only=int((A & ~C).sum()),
               n_gained=int((C & ~ndimage.binary_dilation(A, STRUCT)).sum()),
               n_lost=int((A & ~ndimage.binary_dilation(C, STRUCT)).sum()),
               iou_dilated=tr.iou_dilated(A, C),
               split=len(fwd) >= 2, merge=len(bwd) >= 2,
               split_labels=[a for a, _ in fwd], merge_labels=[a for a, _ in bwd])
    return out


# ---------------------------------------------------------------------------
# the matched parcels of one step, parcel by parcel
# ---------------------------------------------------------------------------
def parcel_link(step, label, labels, flow, fields, *, band=BAND, dt=DT):
    """``tracking.matched_link(centred=True, band=band)`` kept per parcel.  The parcels are the
    cells of the label at ``step`` dilated by ``band`` cells; each is carried forward to pair
    ``step + 1`` and backward to pair ``step - 1``.  Returns a dict of arrays over the parcels
    -- ``j``, ``i`` (seed cell), ``x_plus``, ``x_minus`` (positions, ``(n, 2) = (j, i)``),
    ``G_minus``, ``G_0``, ``G_plus``, ``rate`` (``(G_plus - G_minus) / 2 dt``), ``store``
    (``DGDt_semilag`` of pair ``step`` at the seed cell), ``two_F``, ``alive`` -- and the
    three means ``r_matched``, ``r_store``, ``two_F_mean``.  ``None`` at the first and last
    pair, where there is no centred difference."""
    keys = list(labels)
    if step == 0 or step + 1 >= len(keys):
        return None
    S = labels[keys[step]] == int(label)
    if band:
        S = ndimage.binary_dilation(S, STRUCT, iterations=int(band))
    jj, ii = np.nonzero(S)
    pos = np.stack([jj, ii], axis=1).astype('float64')
    x_plus = tr.forward_positions(pos, *flow.displacement(step, +1))
    x_minus = tr.forward_positions(pos, *flow.displacement(step - 1, -1))
    g0 = fields.get('G', step)[jj, ii]
    gp = tr.sample_at(fields.get('G', step + 1), x_plus)
    gm = tr.sample_at(fields.get('G', step - 1), x_minus)
    d0 = fields.get('DGDt_semilag', step)[jj, ii]
    d1 = tr.sample_at(fields.get('DGDt_semilag', step + 1), x_plus)    # matched_link's `alive`
    f0 = fields.get('two_F', step)[jj, ii]
    alive = np.isfinite(g0) & np.isfinite(gp) & np.isfinite(d0) & np.isfinite(d1) & np.isfinite(gm)
    out = dict(step=int(step), label=int(label), band=int(band), j=jj, i=ii, seed=S,
               x_plus=x_plus, x_minus=x_minus, G_minus=gm, G_0=g0, G_plus=gp,
               rate=(gp - gm) / (2 * dt), store=d0, two_F=f0, alive=alive,
               n_parcels=int(S.sum()), n_alive=int(alive.sum()))
    ridge = (labels[keys[step]] == int(label))[jj, ii]       # the label's own pixels; the rest are flanks
    # task 3: the surface-flux term (1 m primary; KPPhbl sensitivity), the vertical term and
    # the boundary-layer depth, read at the seed cells of this pair exactly as 2F is
    for key, var in EXTRA_FIELDS.items():
        out[key] = fields.get(var, step)[jj, ii] if var in fields.ds else None
    out.update(ridge=ridge, n_ridge=int(ridge.sum()), n_two_F=int((alive & np.isfinite(f0)).sum()))
    if alive.any():
        ok = alive & np.isfinite(f0)
        for nm, x, y in (('corr_store_two_F', d0, f0), ('corr_rate_store', out['rate'], d0),
                         ('corr_rate_two_F', out['rate'], f0)):
            out[nm] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else np.nan
        for nm, x in (('rate', out['rate']), ('store', d0), ('two_F', f0)):
            for part, m in (('ridge', ok & ridge), ('flank', ok & ~ridge)):
                out[f'{nm}_{part}'] = float(x[m].mean()) if m.any() else np.nan
        for key in EXTRA_FIELDS:
            x = out[key]
            if x is None:
                continue
            m = ok & np.isfinite(x)
            out[f'{key}_mean'] = float(x[m].mean()) if m.any() else np.nan
            out[f'n_{key}'] = int(m.sum())
            if key.startswith('S_'):
                out[f'{key}_ridge'] = float(x[m & ridge].mean()) if (m & ridge).any() else np.nan
                out[f'{key}_flank'] = float(x[m & ~ridge].mean()) if (m & ~ridge).any() else np.nan
                out[f'{key}_frac_pos'] = float(np.mean(x[m] > 0)) if m.any() else np.nan
                out[f'corr_{key}_imbalance'] = (float(np.corrcoef(x[m], (d0 - f0)[m])[0, 1])
                                                if m.sum() > 2 else np.nan)
        if out['hbl'] is not None:
            out['hbl_median'] = float(np.nanmedian(out['hbl'][ok]))
        # 2F task 1: the chain-rule (continuum-formula) 2F at the same seed cells
        if CHAIN_VAR in fields.ds:
            fc = fields.get(CHAIN_VAR, step)[jj, ii]
            m = ok & np.isfinite(fc)
            out.update(two_F_chain=fc, n_two_F_chain=int(m.sum()))
            if m.any():
                out.update(two_F_chain_mean=float(fc[m].mean()), two_F_discrete_on_chain=float(f0[m].mean()),
                           two_F_chain_ridge=float(fc[m & ridge].mean()) if (m & ridge).any() else np.nan,
                           two_F_chain_flank=float(fc[m & ~ridge].mean()) if (m & ~ridge).any() else np.nan,
                           two_F_chain_rms=float(np.sqrt(np.mean(fc[m] ** 2))),
                           two_F_rms_parcels=float(np.sqrt(np.mean(f0[m] ** 2))))
                for nm, x in (('corr_chain_two_F', f0), ('corr_store_chain', d0), ('corr_rate_chain', out['rate'])):
                    out[nm] = float(np.corrcoef(fc[m], x[m])[0, 1]) if m.sum() > 2 else np.nan
        out.update(r_matched=(gp[alive].mean() - gm[alive].mean()) / (2 * dt),
                   r_store=float(d0[alive].mean()),
                   two_F_mean=float(f0[ok].mean()) if ok.any() else np.nan,
                   G_mean_minus=float(gm[alive].mean()), G_mean_0=float(g0[alive].mean()),
                   G_mean_plus=float(gp[alive].mean()),
                   disp_plus_median=float(np.median(np.hypot(*(x_plus - pos)[alive].T))),
                   disp_minus_median=float(np.median(np.hypot(*(x_minus - pos)[alive].T))),
                   frac_parcels_rate_pos=float(np.mean(out['rate'][alive] > 0)))
    else:
        out.update(r_matched=np.nan, r_store=np.nan, two_F_mean=np.nan)
    return out


#: 2F task 1: the chain-rule form of 2F in the derived store, and its per-step scalars
CHAIN_VAR = 'two_F_chain'
CHAIN_SCALARS = ('two_F_chain_mean', 'n_two_F_chain', 'two_F_discrete_on_chain', 'two_F_chain_ridge',
                 'two_F_chain_flank', 'two_F_chain_rms', 'two_F_rms_parcels', 'corr_chain_two_F',
                 'corr_store_chain', 'corr_rate_chain')

#: task 3: per-parcel key -> variable of the derived store
EXTRA_FIELDS = dict(S_1m='surface_flux', S_kpp='surface_flux_kpp', V='vertical', hbl='KPPhbl')
SCALARS_EXTRA = tuple(f'{k}_mean' for k in EXTRA_FIELDS) + tuple(f'n_{k}' for k in EXTRA_FIELDS) + tuple(
    f'{k}_{q}' for k in ('S_1m', 'S_kpp') for q in ('ridge', 'flank', 'frac_pos')) + (
    'corr_S_1m_imbalance', 'corr_S_kpp_imbalance', 'hbl_median') + CHAIN_SCALARS

SCALARS = ('n_parcels', 'n_alive', 'r_matched', 'r_store', 'two_F_mean', 'G_mean_minus',
           'G_mean_0', 'G_mean_plus', 'disp_plus_median', 'disp_minus_median',
           'frac_parcels_rate_pos', 'n_ridge', 'n_two_F', 'corr_store_two_F', 'corr_rate_store',
           'corr_rate_two_F', 'rate_ridge', 'rate_flank', 'store_ridge', 'store_flank',
           'two_F_ridge', 'two_F_flank')


def analyse_track(track, labels, flow, fields, *, band=BAND, check=True, log=None):
    """One row per found step: the pixel-set counts of :func:`pixel_sets` and the rates of
    :func:`parcel_link` (``two_F``, ``r_store``, ``r_matched`` in s^-5).  With ``check`` the
    three means are asserted equal to ``tracking.matched_link``'s."""
    rows = []
    for s in track.steps():
        ps = pixel_sets(track, s, labels, flow)
        row = {k: v for k, v in ps.items() if k not in ('chosen', 'advected')}
        pl = parcel_link(s, track.labels[s], labels, flow, fields, band=band)
        if pl is not None:
            row.update({k: pl.get(k, np.nan) for k in SCALARS})
            row.update({k: pl[k] for k in SCALARS_EXTRA if k in pl})
            row['lst'] = float((s + 0.5 + UTC_OFFSET) % 24)
            if check:
                ref = tr.matched_link(s, track.labels[s], track.labels.get(s + 1), labels, flow,
                                      fields, centred=True, band=band)
                for mine, theirs in (('r_matched', 'r_matched'), ('r_store', 'r_store'),
                                     ('two_F_mean', 'two_F')):
                    if not np.isclose(row[mine], ref[theirs], rtol=1e-9, atol=1e-30, equal_nan=True):
                        raise AssertionError(f'step {s}: {mine} {row[mine]} != matched_link '
                                             f'{theirs} {ref[theirs]}')
                row.update({k: ref[k] for k in ('subfilter', 'vertical', 'surface_flux', 'residual')})
        rows.append(row)
        if log is not None and s % 10 == 0:
            log(f'  step {s}: label {row["label"]}, {row["npix"]} px')
    df = pd.DataFrame(rows).rename(columns={'two_F_mean': 'two_F'})
    df['anchor_step'], df['anchor_label'] = track.anchor.step, track.anchor.label
    return df


def _rms(x):
    return float(np.sqrt(np.nanmean(np.asarray(x, 'float64') ** 2)))


def _trapz(d, col, dt=DT):
    """Trapezoid integral of an hourly rate over the steps that have one [s^-4]."""
    return float(np.trapezoid(d[col].to_numpy('float64'), d['step'].to_numpy('float64') * dt))


SET_JUMP = 0.25                 # |change of log npix| between consecutive steps that counts as a jump


def set_change_summary(d, cols):
    """Does a rate's hour-to-hour change follow the change of the pixel set?  Per series: the
    correlation of ``|change of the rate|`` with ``|change of log npix|`` over consecutive
    steps, and the median ``|change|`` on the links where the label size jumps by more than
    ``SET_JUMP`` in the log against the other links."""
    d = d.sort_values('step')
    ok = np.diff(d['step'].to_numpy()) == 1
    dl = np.abs(np.diff(np.log(d['npix'].to_numpy('float64'))))[ok]
    big = dl > SET_JUMP
    out = dict(n_links=int(ok.sum()), n_jumps=int(big.sum()), jump_threshold=SET_JUMP, series={})
    for name, col in cols.items():
        if col not in d or d[col].isna().all():
            continue
        dr = np.abs(np.diff(d[col].to_numpy('float64')))[ok]
        out['series'][name] = dict(
            corr=float(np.corrcoef(dr, dl)[0, 1]) if dr.size > 2 and dl.std() > 0 and dr.std() > 0 else np.nan,
            median_change_jump=float(np.median(dr[big])) if big.any() else np.nan,
            median_change_other=float(np.median(dr[~big])) if (~big).any() else np.nan)
    return out


def surface_flux_summary(d):
    """Task 3: the surface-flux term (both depth scales) and the vertical term on the matched
    parcels, against the two measured rates.  ``None`` when the store has no such fields.
    The ``multiplier`` is the report's §7.2 test: the weight on the term that minimises the
    r.m.s. of (measured - 2F - V - weight * S); +1 for a correct term."""
    if 'S_1m_mean' not in d or d['S_1m_mean'].isna().all():
        return None
    has_kpp = 'S_kpp_mean' in d and d['S_kpp_mean'].notna().any()
    S1, V = d['S_1m_mean'], d['V_mean']
    Sk = d['S_kpp_mean'] if has_kpp else S1 * np.nan
    cols = dict(two_F=d['two_F'], r_store=d['r_store'], r_matched=d['r_matched'], vertical=V, S_1m=S1, S_kpp=Sk)
    night = (d['lst'] < 6) | (d['lst'] >= 18)
    out = dict(
        same_parcels=bool((d['n_S_1m'] == d['n_alive']).all() and (not has_kpp or (d['n_S_kpp'] == d['n_alive']).all())),
        terms={k: dict(mean=float(v.mean()), rms=_rms(v), frac_pos=float(np.mean(v > 0))) for k, v in cols.items()},
        residual_rms={m: dict(two_F=_rms(d[m] - d['two_F']), two_F_V=_rms(d[m] - d['two_F'] - V),
                              full_1m=_rms(d[m] - d['two_F'] - V - S1), full_kpp=_rms(d[m] - d['two_F'] - V - Sk))
                      for m in ('r_store', 'r_matched')},
        projection={k: {m: dict(corr=float(np.corrcoef(S, d[m] - d['two_F'] - V)[0, 1]),
                                multiplier=float((S * (d[m] - d['two_F'] - V)).sum() / (S ** 2).sum()))
                        for m in ('r_store', 'r_matched')} for k, S in (('S_1m', S1), ('S_kpp', Sk))},
        corr_S_1m_S_kpp=float(np.corrcoef(S1, Sk)[0, 1]) if has_kpp else np.nan,
        corr_S_1m_two_F=float(np.corrcoef(S1, d['two_F'])[0, 1]),
        S_1m_night=float(S1[night].mean()), S_1m_day=float(S1[~night].mean()),
        S_kpp_night=float(Sk[night].mean()), S_kpp_day=float(Sk[~night].mean()),
        S_1m_ridge=float(d['S_1m_ridge'].mean()), S_1m_flank=float(d['S_1m_flank'].mean()),
        S_kpp_ridge=float(d['S_kpp_ridge'].mean()) if has_kpp else np.nan,
        S_kpp_flank=float(d['S_kpp_flank'].mean()) if has_kpp else np.nan,
        frac_parcels_pos_median=float(d['S_1m_frac_pos'].median()),
        per_parcel_corr_S_imbalance_median=float(d['corr_S_1m_imbalance'].median()),
        per_parcel_corr_S_kpp_imbalance_median=float(d['corr_S_kpp_imbalance'].median()) if has_kpp else np.nan,
        hbl_median=float(d['hbl_median'].median()) if 'hbl_median' in d else np.nan,
        hbl_min=float(d['hbl_mean'].min()) if 'hbl_mean' in d else np.nan,
        hbl_max=float(d['hbl_mean'].max()) if 'hbl_mean' in d else np.nan,
        integral=dict(S_1m=_trapz(d, 'S_1m_mean'), S_kpp=_trapz(d, 'S_kpp_mean') if has_kpp else np.nan,
                      V=_trapz(d, 'V_mean')))
    return out


def chain_summary(d):
    """2F task 1: the chain-rule ``2F`` against the stored (discretely consistent) ``2F`` on the
    matched parcels, and each against the two measured rates.  ``None`` without the field.
    ``mean_ratio`` is discrete over chain -- two forms of the prediction, not measured over
    predicted."""
    if 'two_F_chain_mean' not in d or d['two_F_chain_mean'].isna().all():
        return None
    c, f = d['two_F_chain_mean'], d['two_F']
    out = dict(
        same_parcels=bool((d['n_two_F_chain'] == d['n_two_F']).all()),
        mean=dict(chain=float(c.mean()), discrete=float(f.mean())), rms=dict(chain=_rms(c), discrete=_rms(f)),
        frac_hours_pos=dict(chain=float(np.mean(c > 0)), discrete=float(np.mean(f > 0))),
        mean_ratio=float(f.mean() / c.mean()), rms_ratio=_rms(f) / _rms(c),
        corr=float(np.corrcoef(c, f)[0, 1]), rms_diff=_rms(c - f),
        frac_hours_abs_chain_larger=float(np.mean(np.abs(c) > np.abs(f))),
        per_parcel_corr_median=float(d['corr_chain_two_F'].median()),
        per_parcel_rms_ratio_median=float((d['two_F_rms_parcels'] / d['two_F_chain_rms']).median()),
        ridge=dict(chain=float(d['two_F_chain_ridge'].mean()), discrete=float(d['two_F_ridge'].mean())),
        flank=dict(chain=float(d['two_F_chain_flank'].mean()), discrete=float(d['two_F_flank'].mean())),
        integral=dict(chain=_trapz(d, 'two_F_chain_mean'), discrete=_trapz(d, 'two_F')),
        against={})
    for m in ('r_store', 'r_matched'):
        out['against'][m] = {k: dict(corr=float(np.corrcoef(d[m], x)[0, 1]), rms_diff=_rms(d[m] - x),
                                     mean_diff=float((d[m] - x).mean()))
                             for k, x in (('chain', c), ('discrete', f))}
    out['per_parcel_corr_store'] = dict(chain=float(d['corr_store_chain'].median()),
                                        discrete=float(d['corr_store_two_F'].median()))
    out['per_parcel_corr_matched'] = dict(chain=float(d['corr_rate_chain'].median()),
                                          discrete=float(d['corr_rate_two_F'].median()))
    return out


def track_summary(df):
    """The track's numbers the markdown quotes."""
    d = df.dropna(subset=['r_matched'])
    st = tr._stats
    link = df.dropna(subset=['prev_step'])
    return dict(
        n_steps=int(len(df)), n_rates=int(len(d)),
        npix=dict(min=int(df['npix'].min()), median=float(df['npix'].median()),
                  max=int(df['npix'].max())),
        n_parcels_median=float(d['n_parcels'].median()), n_lost=int((d['n_parcels'] - d['n_alive']).sum()),
        disp_plus_median=float(d['disp_plus_median'].median()),
        frac_hours_pos=dict(two_F=float(np.mean(d['two_F'] > 0)), r_store=float(np.mean(d['r_store'] > 0)),
                            r_matched=float(np.mean(d['r_matched'] > 0))),
        mean_rate=dict(two_F=float(d['two_F'].mean()), r_store=float(d['r_store'].mean()),
                       r_matched=float(d['r_matched'].mean())),
        matched_vs_store=st(d['r_matched'], d['r_store']),
        matched_vs_two_F=st(d['r_matched'], d['two_F']),
        store_vs_two_F=st(d['r_store'], d['two_F']),
        same_parcels=dict(
            n_steps=int(len(d)), parcels_total=int(d['n_parcels'].sum()),
            in_matched_and_stored=int(d['n_alive'].sum()), in_two_F=int(d['n_two_F'].sum()),
            identical_at_every_step=bool(((d['n_parcels'] == d['n_alive']) & (d['n_alive'] == d['n_two_F'])).all()),
            steps_with_dropped_parcels=[int(v) for v in d.loc[d['n_parcels'] != d['n_two_F'], 'step']]),
        rms=dict(two_F=_rms(d['two_F']), r_store=_rms(d['r_store']), r_matched=_rms(d['r_matched']),
                 store_minus_two_F=_rms(d['r_store'] - d['two_F']),
                 matched_minus_store=_rms(d['r_matched'] - d['r_store']),
                 matched_minus_two_F=_rms(d['r_matched'] - d['two_F'])),
        frac_hours=dict(store_gt_two_F=float(np.mean(d['r_store'] > d['two_F'])),
                        matched_lt_store=float(np.mean(d['r_matched'] < d['r_store'])),
                        matched_lt_two_F=float(np.mean(d['r_matched'] < d['two_F']))),
        integral=dict(hours=float(d['step'].max() - d['step'].min()),
                      two_F=_trapz(d, 'two_F'), r_store=_trapz(d, 'r_store'), r_matched=_trapz(d, 'r_matched'),
                      G_band_first=float(d['G_mean_0'].iloc[0]), G_band_last=float(d['G_mean_0'].iloc[-1]),
                      dG_band=float(d['G_mean_0'].iloc[-1] - d['G_mean_0'].iloc[0])),
        per_parcel_corr_median=dict(store_two_F=float(d['corr_store_two_F'].median()),
                                    rate_store=float(d['corr_rate_store'].median()),
                                    rate_two_F=float(d['corr_rate_two_F'].median())),
        ridge_fraction_median=float((d['n_ridge'] / d['n_parcels']).median()),
        ridge_flank_mean={k: float(d[k].mean()) for k in ('rate_ridge', 'rate_flank', 'store_ridge',
                                                           'store_flank', 'two_F_ridge', 'two_F_flank')},
        budget_mean={k: float(d[k].mean()) for k in ('subfilter', 'vertical', 'surface_flux', 'residual')
                     if k in d},
        surface_flux=surface_flux_summary(d),
        two_F_chain=chain_summary(d),
        set_change=set_change_summary(d, dict(two_F='two_F', r_store='r_store', r_matched='r_matched',
                                              S_1m='S_1m_mean', S_kpp='S_kpp_mean')),
        n_links=int(len(link)), n_split=int(link['split'].sum()), n_merge=int(link['merge'].sum()),
        n_split_or_merge=int((link['split'] | link['merge']).sum()),
        both_frac_median=float((link['n_both'] / link['npix']).median()),
        gained_frac_median=float((link['n_gained'] / link['npix']).median()),
        lost_frac_median=float((link['n_lost'] / link['n_advected']).median()),
        iou_dilated_median=float(link['iou_dilated'].median()),
        steps_gain_gt_25pct=[int(s) for s in link.loc[link['n_gained'] > 0.25 * link['npix'], 'step']],
        steps_loss_gt_25pct=[int(s) for s in link.loc[link['n_lost'] > 0.25 * link['n_advected'], 'step']])


def check_against_m4(df, index, path=None):
    """Largest absolute differences from ``m4_strength.json``'s series for a stored track
    (the numbers behind report Fig. 11), relative to the r.m.s. of each rate."""
    path = Path(path or DATA_DIR / 'm4_strength.json')
    if index is None or not path.exists():
        return None
    ref = pd.DataFrame(json.load(open(path))['series'][str(index)]).set_index('step')
    mine = df.set_index('step')
    out = dict(source=path.name, track=int(index))
    for c in ('r_matched', 'r_store', 'two_F'):
        a, b = mine[c].to_numpy(float), ref[c].reindex(mine.index).to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        out[c] = dict(n=int(ok.sum()), n_nan_mismatch=int((np.isfinite(a) != np.isfinite(b)).sum()),
                      max_abs_diff_over_rms=float(np.max(np.abs(a[ok] - b[ok])) / np.sqrt(np.mean(b[ok] ** 2))))
    out['labels_equal'] = bool((mine['label'] == ref['label'].reindex(mine.index)).all())
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return matplotlib, plt


COL_BOTH, COL_CHOSEN, COL_ADV, COL_OTHER = '#000000', '#1f5fa8', '#e07b00', '#9a9a9a'
COL_S1M, COL_SKPP = '#c8102e', '#e07b00'                           # the surface-flux term, two depths
COL_2F, COL_STORE, COL_MATCHED = '#1f5fa8', '#5b2a86', 'black'      # as in Figure 8


def track_crop(track, labels, flow, pad=8):
    """A fixed crop ``(slice j, slice i)`` holding the chosen label at every step."""
    keys = list(labels)
    jj, ii = [], []
    for s in track.steps():
        j, i = np.nonzero(labels[keys[s]] == track.labels[s])
        jj += [j.min(), j.max()]
        ii += [i.min(), i.max()]
    nj, ni = labels[keys[0]].shape
    return (slice(max(min(jj) - pad, 0), min(max(jj) + pad + 1, nj)),
            slice(max(min(ii) - pad, 0), min(max(ii) + pad + 1, ni)))


def _fmt(v):
    return 'n/a' if v is None or not np.isfinite(v) else f'{v / UNIT:+.2f}'


def _draw_sets(ax, ps, labels, fields, crop, vlim, band=BAND):
    """One panel, north up (``j`` runs east, ``i`` runs south): log10 G in light grey, other
    fronts mid-grey, then the three pixel classes and the outline of the parcel band."""
    matplotlib, _ = _mpl()
    sj, si = crop
    ext = (sj.start - 0.5, sj.stop - 0.5, si.stop - 0.5, si.start - 0.5)
    G = fields.get('G', ps['step'])[sj, si]
    with np.errstate(divide='ignore', invalid='ignore'):
        ax.imshow(np.log10(G).T, cmap='Greys', vmin=vlim[0], vmax=vlim[0] + 2.2 * (vlim[1] - vlim[0]),
                  extent=ext, interpolation='nearest', aspect='equal')
    C = ps['chosen'][sj, si]
    A = ps['advected'][sj, si] if ps['advected'] is not None else np.zeros_like(C)
    other = (labels[ps['time']][sj, si] > 0) & ~C & ~A
    rgba = np.zeros(C.shape + (4,))
    for m, col in ((other, COL_OTHER), (A & ~C, COL_ADV), (C & ~A, COL_CHOSEN), (C & A, COL_BOTH)):
        rgba[m] = matplotlib.colors.to_rgba(col)
    ax.imshow(np.transpose(rgba, (1, 0, 2)), extent=ext, interpolation='nearest', aspect='equal')
    if band:
        B = ndimage.binary_dilation(C, STRUCT, iterations=int(band))
        J, I = np.mgrid[sj, si]
        ax.contour(J, I, B.astype(float), levels=[0.5], colors=['#2a9d8f'], linewidths=0.5)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.tick_params(labelsize=6)


def fig_sets(track, df, labels, flow, fields, steps, path, *, band=BAND, title=''):
    """A page of panels, one per step of ``steps``: the chosen label, the advected previous
    label and the three rates of the step's matched parcels."""
    matplotlib, plt = _mpl()
    crop = track_crop(track, labels, flow)
    nj, ni = crop[0].stop - crop[0].start, crop[1].stop - crop[1].start
    ncol = 2 if nj >= ni else 4
    nrow = int(np.ceil(len(steps) / ncol))
    pw = 15.0 / ncol
    ph = pw * ni / nj + 0.75
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.0, ph * nrow + 1.3), squeeze=False)
    g = fields.get('G', track.anchor.step)[crop]
    vlim = np.percentile(np.log10(g[np.isfinite(g) & (g > 0)]), [5, 99.5])
    rows = df.set_index('step')
    for ax, s in zip(axes.ravel(), steps):
        ps = pixel_sets(track, s, labels, flow)
        _draw_sets(ax, ps, labels, fields, crop, vlim, band)
        r = rows.loc[s]
        flag = ''.join(f'  {k.upper()}' for k in ('split', 'merge') if ps.get(k))
        anchor = '  (anchor)' if s == track.anchor.step else ''
        head = (f'pair {s}  {ps["time"][:13].replace("T", " ")}:30 UTC   label {ps["label"]}, '
                f'{ps["npix"]} px{anchor}{flag}')
        if ps['advected'] is not None:
            sets = (f'previous label {ps["prev_label"]} advected: {ps["n_advected"]} px; same cell '
                    f'{ps["n_both"]}; > 1 cell away: {ps["n_gained"]} chosen, {ps["n_lost"]} advected')
        else:
            sets = 'first step of the track: no previous label'
        if np.isfinite(r.get('r_matched', np.nan)):
            rate = (f'2F {_fmt(r["two_F"])}   stored DG/Dt {_fmt(r["r_store"])}   matched '
                    f'{_fmt(r["r_matched"])}')
            if np.isfinite(r.get('S_1m_mean', np.nan)):
                rate += f'   S(1 m) {_fmt(r["S_1m_mean"])}   S(h$_{{bl}}$) {_fmt(r.get("S_kpp_mean", np.nan))}'
            rate += f'   [1e-19 s$^{{-5}}$, {int(r["n_parcels"])} parcels]'
        else:
            rate = 'no centred rate at the first / last pair'
        ax.set_title(f'{head}\n{sets}\n{rate}', fontsize=7.5, loc='left')
    for ax in axes.ravel()[len(steps):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('j (east →)', fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('i (south ↓)', fontsize=7)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(color=COL_BOTH, label='chosen label and advected previous label (same cell)'),
                        Patch(color=COL_CHOSEN, label='chosen label only'),
                        Patch(color=COL_ADV, label='advected previous label only'),
                        Patch(color=COL_OTHER, label='other fronts at this pair'),
                        Line2D([], [], color='#2a9d8f', lw=1, label=f'band {band}: the seeds of the matched parcels')],
               loc='upper center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1 - 0.55 / (ph * nrow + 1.3)))
    fig.suptitle(title, fontsize=11, y=1 - 0.12 / (ph * nrow + 1.3))
    fig.text(0.5, 0.004, 'L = 0 cells   |   mask: mask_analysis (the fronts are found on it)   |   config D   |   '
             f'band {band}.  Grey shading: log10 G at the pair.  Rates are means over the matched parcels '
             '(seeded on the band of the chosen label); no ratio of them is an efficiency.',
             ha='center', va='bottom', fontsize=7, color='0.3')
    top = 1 - 1.0 / (ph * nrow + 1.3)
    fig.tight_layout(rect=(0, 0.3 / (ph * nrow + 1.3), 1, top), h_pad=0.8)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def fig_series(track, df, path, *, band=BAND):
    """Along the track: the three rates of the matched parcels, and the sizes of the two
    pixel sets with the split and merge links marked."""
    _, plt = _mpl()
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.08))
    d = df.sort_values('step')
    a.axhline(0, color='k', lw=0.5)
    a.plot(d['step'], d['r_matched'] / UNIT, '-', color=COL_MATCHED, lw=1.8, label='matched rate')
    a.plot(d['step'], d['two_F'] / UNIT, '-', color=COL_2F, lw=1.8, label='2F')
    a.plot(d['step'], d['r_store'] / UNIT, ':', color=COL_STORE, lw=1.8, label='stored DG/Dt (DGDt_semilag)')
    a.set_ylabel('mean over the matched parcels [1e-19 s$^{-5}$]')
    a.legend(fontsize=8, frameon=False, ncol=3, loc='upper left')
    a.set_title(f'Track of label {track.anchor.label} (anchor pair {track.anchor.step}): hourly rates on the '
                f'matched parcels (band {band}, centred)', fontsize=10, loc='left')
    b.plot(d['step'], d['npix'], '-', color=COL_CHOSEN, lw=1.8, label='chosen label')
    b.plot(d['step'], d['n_advected'], '-', color=COL_ADV, lw=1.4, label='previous label advected')
    b.plot(d['step'], d['n_both'], '-', color=COL_BOTH, lw=1.2, label='in both (same cell)')
    b.plot(d['step'], d['n_gained'], '--', color=COL_CHOSEN, lw=1.0, label='chosen, > 1 cell from advected')
    b.plot(d['step'], d['n_lost'], '--', color=COL_ADV, lw=1.0, label='advected, > 1 cell from chosen')
    b.set_ylabel('pixels')
    b.set_xlabel('pair (step)')
    b.legend(fontsize=8, frameon=False, ncol=3, loc='upper left')
    b.set_ylim(0, 1.35 * float(d['npix'].max()))
    for ax in (a, b):
        ax.axvline(track.anchor.step, color='k', lw=0.6)
        lo, hi = ax.get_ylim()
        sp, mg = d.loc[d['split'] == True, 'step'], d.loc[d['merge'] == True, 'step']    # noqa: E712
        ax.plot(sp, [lo + 0.03 * (hi - lo)] * len(sp), '^', color='k', ms=5, ls='none')
        ax.plot(mg, [lo + 0.03 * (hi - lo)] * len(mg), 'v', color='#c8102e', ms=5, ls='none')
        ax.set_ylim(lo, hi)
    fig.text(0.5, 0.005, f'L = 0 cells   |   mask: mask_analysis   |   config D   |   band {band}.   ▲ split, ▼ merge '
             '(in time order, >= 5 overlapping pixels per label); vertical line: the anchor. No ratio of the rates is an efficiency.',
             ha='center', va='bottom', fontsize=7, color='0.3')
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.10)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def fig_budget(track, df, path, *, band=BAND):
    """Task 3: the measured rates against the predicted terms, the surface-flux term in its two
    forms against the imbalances, and the boundary-layer depth, along the track."""
    _, plt = _mpl()
    d = df.dropna(subset=['r_matched']).sort_values('step')
    x = d['step'].to_numpy()
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.2, 1.2, 0.6], hspace=0.08))
    pred = d['two_F'] + d['V_mean'] + d['S_kpp_mean']
    a.axhline(0, color='k', lw=0.5)
    a.plot(x, d['r_matched'] / UNIT, '-', color=COL_MATCHED, lw=1.8, label='matched rate')
    a.plot(x, d['r_store'] / UNIT, ':', color=COL_STORE, lw=1.8, label='stored DG/Dt')
    a.plot(x, d['two_F'] / UNIT, '-', color=COL_2F, lw=1.8, label='2F')
    a.plot(x, pred / UNIT, '--', color=COL_SKPP, lw=1.4, label='2F + V + S ($h_{bl}$)')
    a.set_ylabel('[1e-19 s$^{-5}$]')
    a.legend(fontsize=8, frameon=False, ncol=4, loc='upper left')
    a.set_title(f'Track of label {track.anchor.label} (anchor pair {track.anchor.step}): the surface-flux term on the '
                f'matched parcels (band {band})', fontsize=10, loc='left')
    b.axhline(0, color='k', lw=0.5)
    b.plot(x, (d['r_store'] - d['two_F'] - d['V_mean']) / UNIT, ':', color=COL_STORE, lw=1.8,
           label='imbalance: stored − 2F − V')
    b.plot(x, (d['r_matched'] - d['two_F'] - d['V_mean']) / UNIT, '-', color=COL_MATCHED, lw=1.5,
           label='imbalance: matched − 2F − V')
    b.plot(x, d['S_kpp_mean'] / UNIT, '--', color=COL_SKPP, lw=1.6, label='S, flux spread over $h_{bl}$')
    b.plot(x, d['V_mean'] / UNIT, '-', color='0.6', lw=1.0, label='vertical term V')
    b.set_ylabel('[1e-19 s$^{-5}$]')
    b2 = b.twinx()
    b2.plot(x, d['S_1m_mean'] / UNIT, '-', color=COL_S1M, lw=1.8, label='S, 1 m top cell (right axis)')
    b2.set_ylabel('S, 1 m  [1e-19 s$^{-5}$]', color=COL_S1M)
    b2.tick_params(axis='y', colors=COL_S1M)
    lo = min(0.0, float((d['S_1m_mean'] / UNIT).min()))
    b2.set_ylim(lo * 1.1, 1.25 * float((d['S_1m_mean'] / UNIT).max()))
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, ncol=3, loc='upper left')
    yl = b.get_ylim()
    b.set_ylim(yl[0], yl[1] + 0.35 * (yl[1] - yl[0]))
    c.plot(x, d['hbl_mean'], '-', color='k', lw=1.4)
    c.set_ylabel('$h_{bl}$ on the parcels [m]')
    c.set_xlabel('pair (step)')
    c.set_ylim(0, None)
    night = ((d['lst'] < 6) | (d['lst'] >= 18)).to_numpy()
    for ax in (a, b, c):
        for xi, n in zip(x, night):
            if n:
                ax.axvspan(xi - 0.5, xi + 0.5, color='0.92', lw=0, zorder=0)
        ax.axvline(track.anchor.step, color='k', lw=0.6)
    fig.text(0.5, 0.004, f'L = 0 cells   |   mask: mask_analysis   |   config D   |   band {band}.   Means over the matched '
             'parcels. Grey: local night (18–06 LST = UTC − 8 h). The fluxes are 6-hourly forcing interpolated by the model. '
             'No ratio of these terms is an efficiency.', ha='center', va='bottom', fontsize=7, color='0.3')
    fig.subplots_adjust(left=0.07, right=0.93, top=0.96, bottom=0.07)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def fig_parcels(track, labels, flow, fields, steps, path, *, band=BAND, pad=4):
    """Per-parcel maps at a few steps: one row per step, the three per-parcel quantities whose
    means are the three rates, on a shared diverging scale."""
    _, plt = _mpl()
    keys = list(labels)
    links = [parcel_link(s, track.labels[s], labels, flow, fields, band=band) for s in steps]
    links = [pl for pl in links if pl is not None]
    crops = []
    for pl in links:
        crops.append((slice(pl['j'].min() - pad, pl['j'].max() + pad + 1),
                      slice(pl['i'].min() - pad, pl['i'].max() + pad + 1)))
    w = max(c[0].stop - c[0].start for c in crops)
    hs = [(c[1].stop - c[1].start) / w * 5.0 + 0.55 for c in crops]
    cols = [('rate', 'ΔG / 2Δt per parcel', 'r_matched'), ('store', 'stored DG/Dt', 'r_store'),
            ('two_F', '2F', 'two_F_mean')]
    if links[0].get('S_1m') is not None:
        cols.append(('S_1m', 'S (1 m)', 'S_1m_mean'))
    hs = [h * 3 / len(cols) for h in hs]
    fig, axes = plt.subplots(len(links), len(cols), figsize=(15.5, sum(hs) + 1.0), squeeze=False,
                             gridspec_kw=dict(height_ratios=hs))
    lim = np.nanpercentile(np.abs(np.concatenate(
        [np.concatenate([pl[k] for k, _, _ in cols]) for pl in links])), 95) / UNIT
    for row, pl, (sj, si) in zip(axes, links, crops):
        ext = (sj.start - 0.5, sj.stop - 0.5, si.stop - 0.5, si.start - 0.5)
        for ax, (key, name, mkey) in zip(row, cols):
            mean = pl[mkey]
            M = np.full((sj.stop - sj.start, si.stop - si.start), np.nan)
            v = np.where(pl['alive'], pl[key], np.nan)
            M[pl['j'] - sj.start, pl['i'] - si.start] = v / UNIT
            ax.set_facecolor('0.93')
            im = ax.imshow(M.T, cmap='RdBu_r', vmin=-lim, vmax=lim, extent=ext, interpolation='nearest',
                           aspect='equal')
            cj, ci = np.nonzero(labels[keys[pl['step']]] == pl['label'])
            ax.plot(cj, ci, '.', color='k', ms=1.0, ls='none')
            ax.set_title(f'pair {pl["step"]}: {name};  mean {_fmt(mean)}', fontsize=8, loc='left')
            ax.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label('[1e-19 s$^{-5}$] (clipped at the 95th percentile of |value|)', fontsize=8)
    fig.text(0.5, 0.004, f'L = 0 cells   |   mask: mask_analysis   |   config D   |   band {band}.   Each coloured cell is one '
             'matched parcel, drawn at its seed cell; black dots: the label\'s own (skeleton) pixels. '
             'The value quoted for a step is the mean of the map.',
             ha='center', va='bottom', fontsize=7, color='0.3')
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return str(path), [pl['step'] for pl in links]


# ---------------------------------------------------------------------------
# the markdown
# ---------------------------------------------------------------------------
def _f(x, nd=2):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:+.{nd}f}'


def _slide_example(track, df, data_dir=None):
    """The worked example of ``slides_dGdt.py`` (``data/slides_dGdt_<pair>.json``) when it is
    a pair of this track with the same label; ``None`` otherwise."""
    for f in sorted(Path(data_dir or DATA_DIR).glob('slides_dGdt_*.json')):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        s = d.get('pair')
        if s in track.labels and int(track.labels[s]) == int(d.get('label', -1)):
            row = df[df['step'] == s]
            if len(row) and np.isclose(float(row['r_store'].iloc[0]), d['check']['band_mean'], rtol=1e-6):
                return dict(d, row=row.iloc[0], json=f.name)
    return None


def semilag_detail(track, df, *, data_dir=None, slides_rel='../slides'):
    """Markdown lines for the end of §1.1: the semi-Lagrangian ``DG/Dt`` step by step (written
    from ``semilag.py``), and -- when ``slides_dGdt.py`` has drawn a pair of this track -- the
    worked example of ``slides/dGdt_method.png`` with its numbers."""
    L = []
    w = L.append
    w('#### The semi-Lagrangian `DG/Dt`, step by step')
    w('')
    w('*(added 2026-11-21, "Semi-Lagrangian dG/dt" task 1; written from `py/semilag.py`.)*')
    w('')
    w('**What is being measured.** The rate of change of $G$ following the horizontal surface flow,')
    w('')
    w(r'$$\frac{D_h G}{Dt} = \frac{\partial G}{\partial t} + \mathbf{u}\cdot\nabla_h G .$$')
    w('')
    w('The two terms on the right are each much larger than their sum on a front that moves '
      '(the Eulerian form is kept in the store as `DGDt_euler`, a cross-check only). The '
      'semi-Lagrangian step avoids forming them: it takes one difference of $G$ along the path of the water, '
      'between where the water was at $h_t$ and where it is at $h_{t+1}$. The path is chosen to **end** on a cell '
      'centre, so the value at $h_{t+1}$ needs no interpolation and only the value at $h_t$ does. '
      'The result is a field on the grid, one value per cell, attached to the cell where the water arrives.')
    w('')
    w('**Step 1 — velocity at the cell centres, in cells per hour.** The model velocities sit on the cell faces '
      '(`U` on the faces along the grid index $i$, `V` along $j$). They are averaged in time to the midpoint of the '
      'hour and in space to the cell centre, and divided by the local centre-to-centre spacing '
      '(`semilag.centre_velocities`, `departure_index`):')
    w('')
    w(r'$$u_c(j,i) = \tfrac12\left[U^{m}(j,\,i_g{=}i) + U^{m}(j,\,i_g{=}i{+}1)\right], \qquad '
      r'\hat u = \frac{u_c\,\Delta t}{\Delta x_C}, \qquad \hat v = \frac{v_c\,\Delta t}{\Delta y_C},$$')
    w('')
    w(r'with $U^{m} = \tfrac12[U(h_t) + U(h_{t+1})]$ and $v_c$ likewise from `V`. Everything is done in index space '
      r'along the model axes; nothing is rotated to east / north at this stage. $(\hat u, \hat v)$ is the '
      r'displacement in cells over the hour if the velocity were uniform.')
    w('')
    w(r'**Step 2 — the departure point.** For the cell centre $\mathbf{x} = (j, i)$ the displacement '
      r'$\mathbf{d} = (d_j, d_i)$ of the water arriving there is found by the midpoint rule, iterated three times:')
    w('')
    w(r'$$\mathbf{d}^{(0)} = \hat{\mathbf{u}}(\mathbf{x}), \qquad '
      r'\mathbf{d}^{(n+1)} = \hat{\mathbf{u}}\!\left(\mathbf{x} - \tfrac12\mathbf{d}^{(n)}\right),\quad n = 0, 1, 2, '
      r'\qquad \mathbf{x}_d = \mathbf{x} - \mathbf{d}^{(3)} .$$')
    w('')
    w(r'The velocity is evaluated half-way along the path, at the middle of the hour, which makes the step second '
      r'order in $\Delta t$. $\hat{\mathbf{u}}$ at the fractional position is interpolated with the cubic of step 3. '
      r'$\mathbf{x}_d$ is a fractional grid position: where the water that is at the cell centre at $h_{t+1}$ was at $h_t$.')
    w('')
    w(r'**Step 3 — the interpolation.** A field $f$ at a fractional position $\mathbf{p} = (p_j, p_i)$ is the '
      r'tensor-product cubic Lagrange interpolant on the 4 × 4 nearest nodes (`semilag.interp_to_departure`). '
      r'With $j_0 = \lfloor p_j \rfloor$, $\tau_j = p_j - j_0$ and the same along $i$,')
    w('')
    w(r'$$\tilde f(\mathbf{p}) = \sum_{m=-1}^{2}\ \sum_{n=-1}^{2} \ell_m(\tau_j)\,\ell_n(\tau_i)\, f(j_0 + m,\ i_0 + n), '
      r'\qquad \ell_k(\tau) = \prod_{\substack{q=-1 \\ q \ne k}}^{2} \frac{\tau - q}{k - q} .$$')
    w('')
    w('It reproduces any cubic exactly and returns the node value at a whole-cell position. If any of the 16 nodes '
      'is land or outside the tile the result is NaN; nothing is filled.')
    w('')
    w(r'**Step 4 — $G$ at the departure point.** $b(h_t)$ is interpolated at five points: $\mathbf{x}_d$ and its four '
      r'neighbours one cell away along each axis, $\mathbf{x}_d \pm \mathbf{e}_i$ and $\mathbf{x}_d \pm \mathbf{e}_j$. '
      r'The **same** displacement $\mathbf{d}(\mathbf{x})$ is used for all five, so the stencil is the arrival '
      r"cell's own stencil moved rigidly to $\mathbf{x}_d$. Write the five values $b_0, b_E, b_W$ (along $i$) "
      r'and $b_N, b_S$ (along $j$). Then the same gradient stencil $\mathsf{L}$ as everywhere else is applied '
      r'(`semilag.grad_b_at_departure`):')
    w('')
    w(r'$$g_i = \tfrac12\left[\frac{b_0 - b_W}{\Delta x_C^{-}} + \frac{b_E - b_0}{\Delta x_C^{+}}\right], \qquad '
      r'g_j = \tfrac12\left[\frac{b_0 - b_S}{\Delta y_C^{-}} + \frac{b_N - b_0}{\Delta y_C^{+}}\right],$$')
    w('')
    w(r'$$b_x = g_i \cos\theta - g_j \sin\theta, \qquad b_y = g_i \sin\theta + g_j \cos\theta, \qquad '
      r'G(\mathbf{x}_d, h_t) = b_x^2 + b_y^2 .$$')
    w('')
    w(r'$\Delta x_C^{\mp}$ are the spacings to the lower and upper neighbour and $\theta$ the grid angle (`CS`, `SN`), '
      r'all taken at the arrival cell; they change by 0.02 % per cell. With $\mathbf{d} = 0$ this is bit for bit the '
      r'ordinary $G$ of the cell.')
    w('')
    w(r'**Step 5 — $G$ at the arrival cell,** $G(\mathbf{x}, h_{t+1}) = |\mathsf{L}\, b(h_{t+1})|^2$, the ordinary '
      r'gradient of the hour-end buoyancy at the cell. No interpolation.')
    w('')
    w('**Step 6 — the rate.**')
    w('')
    w(r'$$\left.\frac{D_h G}{Dt}\right|_t(\mathbf{x}) = \frac{G(\mathbf{x}, h_{t+1}) - G(\mathbf{x}_d, h_t)}{\Delta t}, '
      r'\qquad \Delta t = 3600\ \mathrm{s}.$$')
    w('')
    w('It is stored at the arrival cell and labelled with the midpoint time of the pair. It can be written as a '
      'local change at the cell plus a change along the path at the first hour, which add to it exactly:')
    w('')
    w(r'$$\frac{G(\mathbf{x}, h_{t+1}) - G(\mathbf{x}_d, h_t)}{\Delta t} = '
      r'\underbrace{\frac{G(\mathbf{x}, h_{t+1}) - G(\mathbf{x}, h_t)}{\Delta t}}_{\text{local}} + '
      r'\underbrace{\frac{G(\mathbf{x}, h_t) - G(\mathbf{x}_d, h_t)}{\Delta t}}_{\text{advective}} .$$')
    w('')
    w('**Three choices in this construction, and why.**')
    w('')
    w(r'- **Buoyancy is interpolated, never $G$.** $G$ has a sharp maximum on a front, and interpolating across a '
      r'maximum cuts it: bilinear interpolation of $G$ is low by about $\Delta x^2\, G_{xx}/8$ there. A value at '
      r'$h_t$ that is too low reads as strengthening that did not happen. On a test front 1.5 cells wide, moved '
      r'half a cell, the error of $G$ at the maximum is −4.8 % with linear interpolation of $b$, −0.49 % with the '
      r'cubic used here and −0.09 % with a quintic (the store keeps the quintic version as `DGDt_semilag_o5`).')
    w(r'- **The stencil is moved rigidly.** The alternative — carry the whole field $b(h_t)$ to the arrival grid '
      r'with each cell\'s own displacement, then differentiate — gives '
      r'$\nabla\!\left[b_t(\mathbf{x}_d(\mathbf{x}))\right] = (\mathsf{I} - \nabla\mathbf{d})^{\mathsf T}\,\nabla b_t$. '
      r'The factor $\nabla\mathbf{d}$ is the strain of the flow over the hour, which is the frontogenesis itself. '
      r'For water that keeps its buoyancy that carried field **is** $b(h_{t+1})$, so the difference would be zero '
      r'when the true rate is $2F$. Holding $\mathbf{d}$ fixed across the stencil keeps the strain out of the '
      r'value at $h_t$, so that it appears in the difference.')
    w(r'- **This is why it can be compared with `2F`.** If buoyancy is carried unchanged, '
      r'$b(\mathbf{x}, h_{t+1}) = b(\mathbf{x}_d(\mathbf{x}), h_t)$, the two gradients differ only because '
      r'neighbouring cells have different displacements. Expanding that difference to first order in $\Delta t$ '
      r'gives the commutator form of `2F` written above, on the same stencil. Anything else that changes $b$ along '
      r'the path — vertical exchange, surface fluxes, mixing, the model\'s own numerical diffusion — and the '
      r'error of an hourly step, appears as a difference between the stored `DG/Dt` and `2F`.')
    w('')
    w('**What limits it.**')
    w('')
    w('- *The hourly step.* The path is built from two velocity snapshots an hour apart. The Eulerian and the '
      'semi-Lagrangian estimates of the same rate differ by 0.672 of its r.m.s. on the pooled front pixels at this '
      'scale (report §7.1); the interpolation order accounts for 0.108.')
    w('- *Ridge cells.* The value at $h_t$ is interpolated and the value at $h_{t+1}$ is read at a cell. On the crest '
      'of a ridge one cell wide the interpolated value is slightly low, which biases the rate positive: +0.6 to +1 % '
      'of $G$ per hour on a synthetic ridge whose true rate is zero (report §9.3, §10). §4 below shows the stored '
      "rate is much larger on the label's own pixels than on the flanks; how much of that is this bias is not established.")
    w('- *Time levels.* It uses hour-end buoyancy; `G` and `2F` use the midpoint mean. All three are centred on the '
      'midpoint of the pair.')
    ex = _slide_example(track, df, data_dir)
    if ex is not None:
        s, m, sp, ck = ex['pair'], ex['method'], ex['split'], ex['check']
        pc, row = m['parcel'], ex['row']
        t0, t1 = (x[-5:] for x in ex['times'])
        u = UNIT
        w('')
        w(f'**A worked example: pair {s} of this track** ({ex["times"][0][:10]}, {t0} → {t1} UTC), drawn by '
          f'`py/slides_dGdt.py`. The tracker chose label {ex["label"]} at this pair: {ck["n_skeleton"]} pixels, '
          f'{ck["n_band"]:,} parcels after the one-cell dilation. Every field in the figure was recomputed from the raw '
          f'store with the functions above and agrees with the stored `DGDt_semilag` to '
          f'{ck["max_err_over_rms"]:.0e} of its r.m.s. on the parcels.')
        w('')
        w(f'![The semi-Lagrangian DG/Dt of pair {s}]({slides_rel}/dGdt_method.png)')
        w('')
        w(f'**Figure (from the slides): `slides/dGdt_method.png`.** Unlike the other figures of this document it is '
          f'drawn north-up in longitude and latitude.')
        w('')
        w(f'- **(a)** The front on $\\log_{{10}} G$ at the midpoint ({ex["time_mid"][-5:]} UTC): the label\'s pixels '
          f'(red), the parcels (cyan outline) and a {ex["window"]["size"]} × {ex["window"]["size"]}-cell window (orange) '
          f'used for the lower row.')
        w(f'- **(b)** Steps 2 to 4 for one parcel, at true scale. The background and contours are $b$ at {t0}; the '
          f'grey lines are cell edges. The white dot is the arrival cell centre $\\mathbf{{x}}$ (grid cell j {pc["j"]}, '
          f'i {pc["i"]}; black square). The red dot is its departure point $\\mathbf{{x}}_d$: '
          f'$\\mathbf{{d}} = (d_i, d_j) = ({pc["di"]:+.3f}, {pc["dj"]:+.3f})$ cells, {pc["disp_cells"]:.2f} of a cell '
          f'in length. The four open circles are the stencil neighbours of step 4, one cell from $\\mathbf{{x}}_d$; '
          f'$b$ at {t0} is interpolated at these five points.')
        w(f'  - For this parcel $G(\\mathbf{{x}}, \\text{{{t1}}}) = {pc["G_x_h1"] / 1e-13:.3f}$ and '
          f'$G(\\mathbf{{x}}_d, \\text{{{t0}}}) = {pc["G_xd_h0"] / 1e-13:.3f}$ (× 10⁻¹³ s⁻⁴), so '
          f'$D_hG/Dt = ({pc["G_x_h1"] / 1e-13:.3f} - {pc["G_xd_h0"] / 1e-13:.3f}) \\times 10^{{-13}} / 3600 = '
          f'{pc["rate"] / u:+.1f}$ × 10⁻¹⁹ s⁻⁵: a {100 * (pc["G_x_h1"] / pc["G_xd_h0"] - 1):.0f} % rise of $G$ in the hour.')
        w(f'- **(d), (e), (f)** The same three numbers for every parcel in the window: $G$ at the departure points at '
          f'{t0} (step 4), $G$ at the arrival cells at {t1} (step 5), and their difference over one hour (step 6). '
          f'The two $G$ maps are nearly identical to the eye; the rate is their small difference. The window holds '
          f'{m["window_n"]} parcels with a mean of {m["window_mean"]:+.1f}.')
        w(f'- **(c)** The rate at every parcel of the front. The mean over the {ck["n_band"]:,} parcels is '
          f'**{ck["band_mean"] / u:+.2f}** × 10⁻¹⁹ s⁻⁵. This is $r_{{\\rm store}}$ of pair {s}: the "stored" entry of '
          f'that pair in the table of §7 and one point of the stored curve in Figure 1. Individual parcels reach '
          f'several tens in either sign (r.m.s. {sp["total"]["band_rms"]:.1f}), so the band mean is a small '
          f'remainder of values of both signs.')
        w('')
        gb = ex['G_band_mean']
        w(f'Displacements on this band are a fraction of a cell: median {ex["disp_cells"]["p50"]:.2f}, 90th percentile '
          f'{ex["disp_cells"]["p90"]:.2f}, largest {ex["disp_cells"]["max"]:.2f} cells. In band means, '
          f'$G(\\mathbf{{x}}_d, h_t) = {gb["Gd_h0"] / 1e-14:.3f}$, $G(\\mathbf{{x}}, h_t) = {gb["G_h0"] / 1e-14:.3f}$ and '
          f'$G(\\mathbf{{x}}, h_{{t+1}}) = {gb["G_h1"] / 1e-14:.3f}$ (× 10⁻¹⁴ s⁻⁴). The local part is therefore '
          f'{sp["local"]["band_mean"]:+.2f} and the advective part {sp["advect"]["band_mean"]:+.2f}, which add to '
          f'{sp["total"]["band_mean"]:+.2f}. Parcel by parcel the two parts are larger than their sum (r.m.s. '
          f'{sp["local"]["band_rms"]:.1f} and {sp["advect"]["band_rms"]:.1f} against {sp["total"]["band_rms"]:.1f}) '
          f'and largely cancel; `slides/dGdt_split.png` shows them. For comparison, at this pair the matched rate is '
          f'{row["r_matched"] / u:+.2f} and `2F` is {row["two_F"] / u:+.2f}.')
    w('')
    return L


def _slide_2F_example(track, df, data_dir=None):
    """The worked example of ``slides_2F.py`` (``data/slides_2F_<pair>.json``) when it is a
    pair of this track with the same label and the same band-mean ``2F``; else ``None``."""
    for f in sorted(Path(data_dir or DATA_DIR).glob('slides_2F_*.json')):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        s = d.get('pair')
        if s in track.labels and int(track.labels[s]) == int(d.get('label', -1)):
            row = df[df['step'] == s]
            if len(row) and np.isclose(float(row['two_F'].iloc[0]), d['check']['band_mean_two_F'], rtol=1e-6):
                return dict(d, row=row.iloc[0])
    return None


def two_F_detail(track, df, summ, *, data_dir=None, slides_rel='../slides'):
    """Markdown lines of §1.3: how ``2F`` is calculated -- the continuum expression, its direct
    evaluation on the grid (the chain-rule form, ``two_F_chain``) and the discretely consistent
    form that is stored as ``two_F`` and used in every comparison (written from
    ``operators.frontogenesis``), with the two forms compared on this track's parcels."""
    L = []
    w = L.append
    u = UNIT
    w('### 1.3 How `2F` is calculated: the continuum expression and its two forms on the grid')
    w('')
    w('*(added 2026-11-23, "2F" task 1; written from `py/operators.py`, `frontogenesis` and '
      '`_frontogenesis_discrete`.)*')
    w('')
    w('There is one continuum expression for $F$ and there are two ways of evaluating it on the model grid. '
      'The derived store holds both: `two_F_chain` and `two_F`. **Every `2F` in this document, and in the report, '
      'is `two_F`, the discretely consistent form.** In short:')
    w('')
    w('| | what it is | exact for | in the store |')
    w('|---|---|---|---|')
    w('| continuum | the formula, for smooth fields | water that keeps its buoyancy and moves horizontally | — |')
    w('| chain-rule form | the formula with every derivative replaced by a grid difference | nothing on a finite grid; '
      'error of order $(\\Delta x/\\ell)^2$ for a front of width $\\ell$ | `two_F_chain` |')
    w('| discretely consistent form | the rate of change of the **gridded** $G = \\lvert\\mathsf{L} b\\rvert^2$ under '
      'advection | the quantity the stored `DG/Dt` measures, as $\\Delta t \\to 0$ | `two_F` (used here) |')
    w('')
    w('So the continuum expression is exact as mathematics, but it cannot be evaluated exactly from gridded fields. '
      'Of the two gridded forms, the chain-rule form is the more direct transcription and the discretely consistent '
      'form is the one that matches what the measured side computes. Neither is "the exact one"; they answer '
      'slightly different questions, and on fronts one or two cells wide the difference is not small.')
    w('')
    w('#### The continuum expression')
    w('')
    w('If buoyancy is carried by the horizontal flow without change, $D_h b/Dt = \\partial_t b + \\mathbf{u}\\cdot\\nabla b = 0$. '
      'Take the gradient of that equation. The gradient does not commute with advection, and the extra term is the '
      'velocity gradient acting on the buoyancy gradient:')
    w('')
    w(r'$$\frac{D_h}{Dt}\frac{\partial b}{\partial x_k} = -\,\frac{\partial \mathbf{u}}{\partial x_k}\cdot\nabla b, '
      r'\qquad k \in \{x, y\}.$$')
    w('')
    w(r'Multiply by $2\,\partial b/\partial x_k$ and sum over $k$, with $G = b_x^2 + b_y^2$:')
    w('')
    w(r'$$\frac{D_h G}{Dt} = -2\,(\nabla b)^{\mathsf T}(\nabla\mathbf{u})\,\nabla b = 2F, \qquad '
      r'F = -\left(u_x\, b_x^2 + (u_y + v_x)\, b_x b_y + v_y\, b_y^2\right).$$')
    w('')
    w(r'The same quantity in terms of the divergence $\delta = u_x + v_y$, the strain magnitude $|\sigma|$ and the '
      r'angle $\theta$ between $\nabla b$ and the compressional axis of the strain:')
    w('')
    w(r'$$2F = -\,\delta\, G + |\sigma|\, G \cos 2\theta .$$')
    w('')
    w('Convergence ($\\delta < 0$) sharpens the front; strain sharpens it when the buoyancy gradient lies within '
      '45° of the compressional axis. The only assumptions are that $b$ is conserved along the horizontal flow and '
      'that the fields are smooth. Vorticity does not appear: rotating the gradient does not change its length.')
    w('')
    w('#### Form 1 — the chain-rule form (`two_F_chain`)')
    w('')
    w('The formula for $F$ is evaluated with each derivative replaced by its grid estimate at the cell centre '
      '(`operators.frontogenesis(form=\'chain\')`):')
    w('')
    w('- $b_x, b_y$ from the gradient stencil $\\mathsf{L}$ of `operators.grad_b`: differences of $b^m$ to the '
      'two faces of the cell, averaged back to the centre (a centred difference over two cells), rotated to east / north.')
    w('- $u_x, u_y, v_x, v_y$ from `operators.jacobian`: the face velocities $\\mathbf{u}^m$ are averaged to the '
      'cell centres and rotated to east / north, then differenced and averaged back in the same way.')
    w('- the three products are formed at the cell and summed; the stored field is $2F$.')
    w('')
    w('Each derivative is second-order accurate, which is very good for a feature many cells wide. The weakness is '
      'specific to this use. The measured side does not hold the true $|\\nabla b|^2$; it holds the gridded '
      '$G = |\\mathsf{L} b|^2$. A two-cell difference underestimates the gradient at the centre of a narrow front, '
      'so the gridded $G$ is smaller than the true one, and its rate of change is smaller too. The chain-rule '
      'form does not know this: it applies the product rule of calculus, which a finite-difference stencil does not '
      'obey. It therefore predicts a larger rate than the gridded $G$ can show, even for a perfectly advected '
      'field with no physics missing.')
    w('')
    w('#### Form 2 — the discretely consistent form (`two_F`, the one used)')
    w('')
    w('The idea is to repeat the continuum derivation with the stencil $\\mathsf{L}$ in place of the derivative, '
      'keeping the commutator instead of applying the product rule. $\\mathsf{L}$ is linear, so under '
      '$\\partial_t b = -\\mathbf{u}\\cdot\\nabla b$,')
    w('')
    w(r'$$\frac{D_h}{Dt}\left(\mathsf{L}_k b\right) = -\,\mathsf{L}_k\!\left(\mathbf{u}\cdot\nabla b\right) + '
      r'\mathbf{u}\cdot\nabla\!\left(\mathsf{L}_k b\right) \equiv -\left[\mathsf{L}_k,\ \mathbf{u}\cdot\nabla\right] b, '
      r'\qquad \frac{D_h}{Dt}\left|\mathsf{L} b\right|^2 = -2\sum_{k}\left(\mathsf{L}_k b\right)'
      r'\left[\mathsf{L}_k,\ \mathbf{u}\cdot\nabla\right] b = 2F_{\rm disc}.$$')
    w('')
    w(r'In the continuum the commutator is $(\partial_k \mathbf{u})\cdot\nabla b$ and the chain-rule form is '
      r'recovered. For the two-cell stencil, with $\mathsf{L}_k f = [f(\mathbf{x}+\mathbf{e}_k) - f(\mathbf{x}-\mathbf{e}_k)]/2h_k$, '
      r'it is exactly')
    w('')
    w(r'$$\left[\mathsf{L}_k,\ \mathbf{u}\cdot\nabla\right] b = \frac{1}{2}\left[\frac{\mathbf{u}(\mathbf{x}+\mathbf{e}_k) - '
      r'\mathbf{u}(\mathbf{x})}{h_k^{+}}\cdot\nabla b(\mathbf{x}+\mathbf{e}_k) + \frac{\mathbf{u}(\mathbf{x}) - '
      r'\mathbf{u}(\mathbf{x}-\mathbf{e}_k)}{h_k^{-}}\cdot\nabla b(\mathbf{x}-\mathbf{e}_k)\right].$$')
    w('')
    w('Read it as the chain-rule term with two changes: the velocity gradient is the pair of **one-sided** '
      'differences to the two neighbours, and each is multiplied by the buoyancy gradient **at that neighbour**, '
      'not at the cell itself. Across a narrow front the gradient at the neighbours is weaker than at the centre, '
      'which is how this form comes out smaller. As coded (`_frontogenesis_discrete`), on the midpoint fields '
      '$b^m$, $\\mathbf{u}^m$:')
    w('')
    w('1. $g = (g_i, g_j) = \\mathsf{L} b^m$ along the two model axes (`grad_b`, then the rotation undone). '
      '$F$ does not depend on the choice of axes, and the stencil is naturally written along the grid.')
    w('2. $\\mathbf{u}_c = (u_c, v_c)$: the face velocities averaged to the cell centres, along the same axes. '
      'This is the same centred velocity the semi-Lagrangian departure point of §1.1 is built from.')
    w('3. The gradient at the neighbours is taken at fourth order, each component along its own axis: '
      '$g^{(4)}_i = \\left[4 g_i - \\tfrac12\\left(g_i(i{+}1) + g_i(i{-}1)\\right)\\right]/3$, and likewise $g^{(4)}_j$ along $j$. '
      'This is also the node derivative of the cubic interpolant the semi-Lagrangian step uses.')
    w('4. For each axis $k$, with $h_k^{\\pm}$ the spacings to the upper and lower neighbour,')
    w('')
    w(r'$$C_k = \frac12\left[\frac{\left(\mathbf{u}_c(\mathbf{x}{+}\mathbf{e}_k) - \mathbf{u}_c(\mathbf{x})\right)\cdot '
      r'g^{(4)}(\mathbf{x}{+}\mathbf{e}_k)}{h_k^{+}} + \frac{\left(\mathbf{u}_c(\mathbf{x}) - \mathbf{u}_c(\mathbf{x}{-}\mathbf{e}_k)\right)\cdot '
      r'g^{(4)}(\mathbf{x}{-}\mathbf{e}_k)}{h_k^{-}}\right], \qquad 2F_{\rm disc} = -2\left(g_i\, C_i + g_j\, C_j\right).$$')
    w('')
    w('**How the two forms differ.**')
    w('')
    w(r'- They are equal when $b$ varies linearly across the stencil, and they converge as the front becomes wide '
      r'compared with the grid.')
    w(r'- At the centre of a front with a tanh profile of width $\ell$, the chain-rule form is larger than the '
      r'rate of the gridded $G$ by the factor $1 + \tfrac23(\Delta x/\ell)^2$. The discrete form over the chain-rule '
      r'form is then 0.86 for $\ell = 2\Delta x$ and 0.96 for $\ell = 4\Delta x$. The discrete form is correct to '
      r'order $(\Delta x/\ell)^4$.')
    w('- The form was not chosen in advance. The project\'s first validation test advected a tracer with the '
      'analysis\'s own semi-Lagrangian step, so that the truth was known, and regressed the measured rate on the '
      'chain-rule `2F`: the result on the real tile was 0.758 where 1 was required. With the discretely consistent '
      'form and a cubic departure velocity the same test gives 0.981 [0.970, 0.994] (report §4, "the discrete '
      'null"). That is a property of the operators on a synthetic truth, not a result about the model\'s fronts.')
    w('- What the discrete form gives up: it predicts the rate of the gridded $G$, which on a front one or two cells '
      'wide is not the rate of the true $|\\nabla b|^2$. The comparison in this document is between two gridded '
      'quantities, which is what makes it a fair one.')
    w('')
    c = summ.get('two_F_chain')
    if c is not None:
        ag = c['against']
        w(f'**The two forms on this track** (means over the same parcels at every step: '
          f'{"confirmed" if c["same_parcels"] else "not at every step"}; × 10⁻¹⁹ s⁻⁵; '
          f'{summ["n_rates"]} steps). The ratio of the two forms compares two predictions and is not a measured '
          f'over predicted ratio.')
        w('')
        w('| | chain-rule form | discrete form (`two_F`) |')
        w('|---|---|---|')
        w(f'| mean over the track | {_f(c["mean"]["chain"] / u)} | {_f(c["mean"]["discrete"] / u)} |')
        w(f'| r.m.s. of the hourly band means | {c["rms"]["chain"] / u:.2f} | {c["rms"]["discrete"] / u:.2f} |')
        w(f'| fraction of hours positive | {c["frac_hours_pos"]["chain"]:.2f} | {c["frac_hours_pos"]["discrete"]:.2f} |')
        w(f'| mean on the ridge (the label\'s own pixels) | {_f(c["ridge"]["chain"] / u)} | {_f(c["ridge"]["discrete"] / u)} |')
        w(f'| mean on the flanks | {_f(c["flank"]["chain"] / u)} | {_f(c["flank"]["discrete"] / u)} |')
        w(f'| integral over the track (10⁻¹⁴ s⁻⁴) | {_f(c["integral"]["chain"] / 1e-14)} | {_f(c["integral"]["discrete"] / 1e-14)} |')
        w(f'| against the stored `DG/Dt`: correlation of hourly means | {ag["r_store"]["chain"]["corr"]:.3f} | {ag["r_store"]["discrete"]["corr"]:.3f} |')
        w(f'| against the stored `DG/Dt`: r.m.s. difference | {ag["r_store"]["chain"]["rms_diff"] / u:.2f} | {ag["r_store"]["discrete"]["rms_diff"] / u:.2f} |')
        w(f'| against the stored `DG/Dt`: mean difference (stored − form) | {_f(ag["r_store"]["chain"]["mean_diff"] / u)} | {_f(ag["r_store"]["discrete"]["mean_diff"] / u)} |')
        w(f'| against the matched rate: correlation of hourly means | {ag["r_matched"]["chain"]["corr"]:.3f} | {ag["r_matched"]["discrete"]["corr"]:.3f} |')
        w(f'| against the matched rate: r.m.s. difference | {ag["r_matched"]["chain"]["rms_diff"] / u:.2f} | {ag["r_matched"]["discrete"]["rms_diff"] / u:.2f} |')
        w(f'| against the matched rate: mean difference (matched − form) | {_f(ag["r_matched"]["chain"]["mean_diff"] / u)} | {_f(ag["r_matched"]["discrete"]["mean_diff"] / u)} |')
        w(f'| per parcel within a step, correlation with the stored `DG/Dt` (median) | {c["per_parcel_corr_store"]["chain"]:.2f} | {c["per_parcel_corr_store"]["discrete"]:.2f} |')
        w(f'| per parcel within a step, correlation with the matched rate (median) | {c["per_parcel_corr_matched"]["chain"]:.2f} | {c["per_parcel_corr_matched"]["discrete"]:.2f} |')
        w('')
        w(f'- The hourly band means of the two forms correlate at {c["corr"]:.3f}; the discrete form is '
          f'{c["rms_ratio"]:.2f} of the chain-rule form in r.m.s. and {c["mean_ratio"]:.2f} in the mean. '
          f'The chain-rule form is the larger in magnitude at {c["frac_hours_abs_chain_larger"]:.0%} of hours.')
        w(f'- Parcel by parcel within a step they correlate at a median {c["per_parcel_corr_median"]:.2f}, with the '
          f'discrete form a median {c["per_parcel_rms_ratio_median"]:.2f} of the chain-rule form in r.m.s. '
          f'They are the same pattern at different amplitude, not two different fields.')
        w('- The rows "against the stored `DG/Dt`" and "against the matched rate" show which form sits closer to each '
          'measured rate on this track. They are descriptive: the budget did not close, and no row is an efficiency.')
        w('')
    ex = _slide_2F_example(track, df, data_dir)
    if ex is not None:
        s, m, sp, ck, dec = ex['pair'], ex['method'], ex['split'], ex['check'], ex['decomposition']
        pc, row = m['parcel'], ex['row']
        w(f'**A worked example: pair {s} of this track** (label {ex["label"]}, {ck["n_band"]:,} parcels, midpoint '
          f'{ex["time_mid"][-5:]} UTC), the same hour, front, window and parcel as the `DG/Dt` figure of §1.1; drawn by '
          f'`py/slides_2F.py`, which recomputes both forms from the raw store and agrees with the stored fields to '
          f'{max(ck["two_F_max_err_over_rms"], ck["two_F_chain_max_err_over_rms"]):.0e} of their r.m.s.')
        w('')
        w(f'![2F of pair {s}]({slides_rel}/2F_method.png)')
        w('')
        w('**Figure (from the slides): `slides/2F_method.png`**, north-up. The picture is explained with the '
          'continuum expression (convergence plus aligned strain); the numbers in (c) and (f) are the stored '
          'discrete form. (a) the front and the window; (b) one parcel: contours of $b^m$, the direction of '
          '$\\nabla b$ (red), the compressional axis of the strain (black) and the flow relative to the parcel '
          '(purple); (c) `2F` at every parcel; (d) the relative flow in the window, converging on the front; '
          '(e) the alignment $\\cos 2\\theta$; (f) `2F` in the window.')
        w('')
        w(f'- **The parcel of panel (b)** (j {pc["j"]}, i {pc["i"]}): $G = {pc["G"] / 1e-13:.3f}$ × 10⁻¹³ s⁻⁴, '
          f'$\\delta = {pc["delta"] / 1e-5:+.2f}$ × 10⁻⁵ s⁻¹ ({"convergent" if pc["delta"] < 0 else "divergent"}), $|\\sigma| = {pc["sigma"] / 1e-5:.2f}$ × 10⁻⁵ s⁻¹, '
          f'$\\theta = {pc["theta_deg"]:.0f}°$. Convergence gives $-\\delta G = {pc["conv"] / u:+.1f}$ and strain '
          f'$|\\sigma| G \\cos 2\\theta = {pc["strain"] / u:+.1f}$, so the chain-rule form is {pc["two_F_chain"] / u:+.1f}. '
          f'The discrete form at the same cell is **{pc["two_F"] / u:+.1f}**, {pc["two_F"] / pc["two_F_chain"]:.2f} of it.')
        w(f'- **The band.** The chain-rule form averages {sp["two_F_chain_mean"]:+.2f} ({sp["conv"]["band_mean"]:+.2f} from '
          f'convergence and {sp["strain"]["band_mean"]:+.2f} from strain; `slides/2F_split.png`); the discrete form '
          f'averages **{sp["two_F_discrete_mean"]:+.2f}**, {sp["ratio"]:.2f} of it, and the two correlate at '
          f'{sp["corr"]:.2f} over the parcels. {m["frac_theta_lt_45"]:.0%} of the parcels have $\\theta < 45°$.')
        w(f'- **Against the measured rates at this pair:** stored `DG/Dt` {row["r_store"] / u:+.2f}, matched rate '
          f'{row["r_matched"] / u:+.2f}. One pair shows how the numbers are formed; it is not evidence for either form.')
        w('')
    return L


def write_markdown(path, name, track, source, df, summ, m4, pages, series_png, parcels_png,
                   detail_steps, keys, *, band=BAND, budget_png=None):
    lab, a = track.anchor.label, track.anchor.step
    st = track.steps()
    s = summ
    L = []
    w = L.append
    w(f'# Matched parcels along the track of label {lab} (anchor pair {a})')
    w('')
    w(f'*Generated {_dt.date.today().isoformat()} by `py/matched_parcels.py` (track analysis, "Matched parcels" '
      f'task 1). Regenerate with `python matched_parcels.py --label {lab} --anchor-step {a}`. '
      f'Every number here is in `data/{name}.json`.*')
    w('')
    w(f'**Setting.** LLC4320 tile 330, `L = 0`, `mask_analysis`, finder config D, flow-informed track '
      f'({source}). The track is found at {len(st)} pairs, {st[0]}–{st[-1]} '
      f'({keys[st[0]][:13].replace("T", " ")}:30 to {keys[st[-1]][:13].replace("T", " ")}:30 UTC); the label has '
      f'{s["npix"]["min"]} / {s["npix"]["median"]:.0f} / {s["npix"]["max"]} pixels (min / median / max). '
      f'Rates are on the band-{band}, centred construction (M4-Q3 (a)) and are quoted in units of '
      f'10⁻¹⁹ s⁻⁵. The M3 budget did not close, so **no ratio or slope of these rates is an efficiency**.')
    w('')
    w('## 1. How the hourly rates of the matched parcels are calculated')
    w('')
    w('### 1.1 The fields of one pair')
    w('')
    w('Pair `t` joins the model hours $h_t$ and $h_{t+1}$ ($\\Delta t = 3600$ s). Every field of the derived store '
      '(`tile330_derived_L0.zarr`) is on the cell centres and is assigned to the midpoint time $t + \\tfrac12$ h. '
      'Buoyancy and velocity at the midpoint are the two-hour means,')
    w('')
    w('$$b^{m} = \\tfrac12\\left[b(h_t) + b(h_{t+1})\\right], \\qquad \\mathbf{u}^{m} = \\tfrac12\\left[\\mathbf{u}(h_t) + \\mathbf{u}(h_{t+1})\\right].$$')
    w('')
    w('**Front strength** (`G`). With $\\mathsf{L} = (\\mathsf{L}_x, \\mathsf{L}_y)$ the centred gradient stencil of '
      '`operators.grad_b` (a difference over two cells),')
    w('')
    w('$$G_t(\\mathbf{x}) = \\left|\\mathsf{L}\\, b^{m}\\right|^2 = b_x^2 + b_y^2 \\quad [\\mathrm{s^{-4}}].$$')
    w('')
    w('This is the field the fronts were found on and the field the matched rate samples.')
    w('')
    w('**`2F`** (`two_F`), the kinematic prediction. In the continuum')
    w('')
    w('$$F = -\\left(u_x\\, b_x^2 + (u_y + v_x)\\, b_x b_y + v_y\\, b_y^2\\right), \\qquad \\frac{D_h G}{Dt} = 2F \\ \\text{for adiabatic horizontal flow}.$$')
    w('')
    w('The stored field is twice the *discretely consistent* form (`operators.frontogenesis(form=\'discrete\')`), '
      'evaluated on the midpoint fields $b^{m}$, $\\mathbf{u}^{m}$:')
    w('')
    w('$$2F_t(\\mathbf{x}) = -2 \\sum_{k \\in \\{x,\\,y\\}} \\left(\\mathsf{L}_k b^{m}\\right) \\left[\\mathsf{L}_k,\\ \\mathbf{u}^{m}\\!\\cdot\\!\\nabla\\right] b^{m},$$')
    w('')
    w('$$\\left[\\mathsf{L}_k,\\ \\mathbf{u}\\!\\cdot\\!\\nabla\\right] b = \\frac{\\left(\\mathbf{u}(\\mathbf{x}+\\mathbf{e}_k) - \\mathbf{u}(\\mathbf{x})\\right)\\!\\cdot\\!\\nabla b(\\mathbf{x}+\\mathbf{e}_k) + \\left(\\mathbf{u}(\\mathbf{x}) - \\mathbf{u}(\\mathbf{x}-\\mathbf{e}_k)\\right)\\!\\cdot\\!\\nabla b(\\mathbf{x}-\\mathbf{e}_k)}{2 h_k}.$$')
    w('')
    w('The bracket is the commutator of the gradient stencil with advection; $h_k$ is the grid spacing and the '
      'neighbour gradients are fourth order. It is what the semi-Lagrangian difference below tends to under exact '
      'advection on this grid. It equals the continuum expression for a linear $b$; at the centre of a front of '
      'width $\\ell$ the continuum expression is larger by a factor $1 + \\tfrac23 (\\Delta x/\\ell)^2$. `2F` is '
      'an instantaneous quantity: it uses one time level (the midpoint) and no time difference. §1.3 derives both '
      'forms and compares them on this track.')
    w('')
    w('**Stored `DG/Dt`** (`DGDt_semilag`), the measured tendency. One semi-Lagrangian step over the hour of the pair '
      '(`semilag.measured_DGDt`):')
    w('')
    w('$$\\left.\\frac{D_h G}{Dt}\\right|_t(\\mathbf{x}) = \\frac{G\\!\\left(\\mathbf{x},\\, h_{t+1}\\right) - G\\!\\left(\\mathbf{x}_d,\\, h_t\\right)}{\\Delta t}, \\qquad \\mathbf{x}_d = \\mathbf{x} - \\mathbf{d}(\\mathbf{x}), \\qquad \\mathbf{d} \\leftarrow \\Delta t\\ \\mathbf{u}^{m}\\!\\left(\\mathbf{x} - \\tfrac12\\mathbf{d}\\right).$$')
    w('')
    w('- $G(\\mathbf{x}, h_{t+1}) = |\\mathsf{L}\\, b(h_{t+1})|^2$ at the cell: the hour-end buoyancy, not the midpoint mean.')
    w('- $\\mathbf{x}_d$ is the departure point of the parcel that **arrives at the cell at $h_{t+1}$**; the displacement '
      '$\\mathbf{d}$ is found by three iterations of the midpoint rule with the velocity interpolated at cubic order.')
    w('- $G(\\mathbf{x}_d, h_t)$ is formed by interpolating $b(h_t)$ (cubic) onto the five-point stencil centred at '
      '$\\mathbf{x}_d$ and applying the same $\\mathsf{L}$. Buoyancy is interpolated, never $G$.')
    w('')
    L.extend(semilag_detail(track, df))
    w('### 1.2 The parcels of a step and the three means')
    w('')
    w(f'At each pair `t` the tracker has chosen one label. `tracking.matched_link(centred=True, band={band})` then does '
      f'the following, and `matched_parcels.parcel_link` repeats it parcel by parcel (the means agree to round-off at '
      f'every step; asserted in the run).')
    w('')
    w(f'1. **Seeds.** The chosen label at `t` — a one-pixel-wide skeleton — is dilated by {band} cell '
      f'(3 × 3, 8-connected). Every cell $\\mathbf{{x}}_p$ of this band is one parcel at its cell centre: the set '
      f'$P_t$, of $N_t$ parcels. Median: {s["n_parcels_median"]:.0f} parcels per step, for a label of '
      f'{s["npix"]["median"]:.0f} pixels; the label\'s own pixels (the ridge) are a median '
      f'{s["ridge_fraction_median"]:.0%} of the band and its flanks the rest.')
    w('2. **Trajectories.** Each parcel is carried forward to pair `t + 1` and backward to pair `t − 1` with the '
      'model\'s surface flow (`tile330_flow.zarr`: the same midpoint-rule displacement as above, over one hour, from '
      '`U`, `V` at the integer hour between the two pairs). `forward_positions` solves '
      '$\\mathbf{x} = \\mathbf{x}_p + \\mathbf{d}(\\mathbf{x})$ for the arrival points $\\mathbf{x}_p^{+}$ (in pair '
      f'`t + 1`) and $\\mathbf{{x}}_p^{{-}}$ (in pair `t − 1`). The median displacement is '
      f'{s["disp_plus_median"]:.2f} cells per hour on this track.')
    w('3. **The three rates** are means over the same $N_t$ parcels:')
    w('')
    w('$$r_{\\rm matched}(t) = \\frac{1}{2\\Delta t}\\left[\\frac{1}{N_t}\\sum_{p \\in P_t} G_{t+1}\\!\\left(\\mathbf{x}_p^{+}\\right) - \\frac{1}{N_t}\\sum_{p \\in P_t} G_{t-1}\\!\\left(\\mathbf{x}_p^{-}\\right)\\right],$$')
    w('')
    w('$$r_{\\rm store}(t) = \\frac{1}{N_t}\\sum_{p \\in P_t} \\left.\\frac{D_h G}{Dt}\\right|_t\\!(\\mathbf{x}_p), \\qquad \\overline{2F}(t) = \\frac{1}{N_t}\\sum_{p \\in P_t} 2F_t(\\mathbf{x}_p).$$')
    w('')
    w('   $G_{t\\pm1}$ are sampled at the fractional positions at cubic order; the stored tendency and `2F` are read at '
      'the seed cells, with no interpolation. Both samples of the matched rate are interpolated and the selection at '
      '`t` is symmetric in time, so the rate is centred on pair `t`.')
    w(f'4. **Parcels kept.** A parcel counts only if every sample is finite. The first and last pairs have no centred '
      f'difference, so {s["n_rates"]} of {s["n_steps"]} steps have rates.')
    w('')
    w('**The previous label does not enter any rate.** The parcels of step `t` are seeded on the label chosen at '
      '`t`; the label at `t − 1` is used only by the tracker, to choose the label at `t`. The parcel population is '
      'therefore re-seeded every hour. The figures of §6 show how the chosen label compares with the previous label '
      'carried by the flow, which is what decides how much the population changes from one hour to the next.')
    w('')
    L.extend(two_F_detail(track, df, s))
    sp = s['same_parcels']
    w('## 2. Are the three rates calculated on the same set of parcels in each pair?')
    w('')
    if sp['identical_at_every_step']:
        w(f'**Confirmed, with two qualifications.** Within each pair the three rates are means over one and the same '
          f'set of seed cells $P_t$. Counted in the run: {sp["parcels_total"]:,} parcels over the {sp["n_steps"]} steps; '
          f'{sp["in_matched_and_stored"]:,} enter the matched and the stored rate and {sp["in_two_F"]:,} enter `2F` — '
          f'the same number, at every step. No parcel was dropped.')
    else:
        w(f'**Nearly.** Within each pair the three rates are means over the seed cells $P_t$, but not every parcel is '
          f'finite in every field: of {sp["parcels_total"]:,} parcels over {sp["n_steps"]} steps, '
          f'{sp["in_matched_and_stored"]:,} enter the matched and the stored rate and {sp["in_two_F"]:,} enter `2F`. '
          f'The sets differ at pairs {", ".join(map(str, sp["steps_with_dropped_parcels"]))}.')
    w('')
    w('In the code, one mask (`alive`: `G` finite at all three times and the stored tendency finite) selects the '
      'parcels of the matched and the stored rate; `2F` uses that mask and also requires `2F` finite.')
    w('')
    w('The qualifications are about what the same seed cell means for each rate:')
    w('')
    w('| rate | read at | built from | time span |')
    w('|---|---|---|---|')
    w('| `2F` | the seed cell, pair `t` | midpoint fields $b^m$, $\\mathbf{u}^m$ | an instant, $t + \\tfrac12$ h |')
    w('| stored `DG/Dt` | the seed cell, pair `t` | $b(h_t)$, $b(h_{t+1})$, $\\mathbf{u}^m$ | one hour, $h_t \\to h_{t+1}$ |')
    if s.get('surface_flux') is not None:
        w('| surface-flux term $S$ (§5) | the seed cell, pair `t` | midpoint fluxes, $\\Theta$, $\\mathcal{S}$, $b^m$ | an instant, $t + \\tfrac12$ h |')
    w('| matched | $\\mathbf{x}_p^{-}$ in pair `t − 1`, $\\mathbf{x}_p^{+}$ in pair `t + 1` | midpoint $G_{t-1}$, $G_{t+1}$ | two hours, $t - \\tfrac12 \\to t + \\tfrac32$ h |')
    w('')
    w(f'1. **A half-hour offset in the stored tendency.** The matched parcel sits on the seed cell at the midpoint time. '
      f'The stored tendency at that cell belongs to the parcel that arrives there at the *end* of the hour, which at '
      f'the midpoint is half a displacement upstream: a median {0.5 * s["disp_plus_median"]:.2f} cells here. The '
      f'parcels are the same to within that distance, not identically.')
    w('2. **Different time windows and time levels.** The matched rate differences midpoint-mean `G` fields two hours '
      'apart; the stored tendency differences hour-end `G` fields one hour apart; `2F` is instantaneous. All three are '
      'centred on $t + \\tfrac12$ h.')
    w('')
    w('Between pairs the sets are *not* the same: $P_t$ and $P_{t+1}$ are seeded on different labels (§6).')
    w('')
    w('## 3. The track at a glance')
    w('')
    w(f'![]({series_png})')
    w('')
    w(f'**Figure 1.** Top: the three hourly rates on the matched parcels. Bottom: pixel counts of the chosen label, '
      f'of the previous label advected to the same pair, of their same-cell overlap, and of the pixels of each set '
      f'farther than one cell from the other. ▲ split, ▼ merge.')
    w('')
    mv, m2, s2 = s['matched_vs_store'], s['matched_vs_two_F'], s['store_vs_two_F']
    fp, mr = s['frac_hours_pos'], s['mean_rate']
    w('| over the track | `2F` | stored `DG/Dt` | matched rate |')
    w('|---|---|---|---|')
    w(f'| mean [10⁻¹⁹ s⁻⁵] | {_f(mr["two_F"] / UNIT)} | {_f(mr["r_store"] / UNIT)} | {_f(mr["r_matched"] / UNIT)} |')
    w(f'| fraction of hours positive | {fp["two_F"]:.2f} | {fp["r_store"]:.2f} | {fp["r_matched"]:.2f} |')
    w('')
    w(f'Hour by hour ({mv["n"]} steps): matched against stored, correlation {mv["corr"]:.3f}, r.m.s. difference '
      f'{mv["rms_ratio"]:.3f} of the stored r.m.s.; matched against `2F`, correlation {m2["corr"]:.3f}, sign agreement '
      f'{m2["sign_agree"]:.2f}; stored against `2F`, correlation {s2["corr"]:.3f}. These are consistency checks, not '
      f'efficiencies.')
    w('')
    w(f'Pixel sets over {s["n_links"]} links: the median fraction of the chosen label\'s pixels that coincide '
      f'cell-for-cell with the advected previous label is {s["both_frac_median"]:.2f}; the median dilated IoU is '
      f'{s["iou_dilated_median"]:.2f}. More than one cell from the advected previous label lie a median '
      f'{s["gained_frac_median"]:.2f} of the chosen pixels; {len(s["steps_gain_gt_25pct"])} links have more than a '
      f'quarter of the chosen label new in this sense (pairs {", ".join(map(str, s["steps_gain_gt_25pct"])) or "none"}) '
      f'and {len(s["steps_loss_gt_25pct"])} have more than a quarter of the advected label unmatched '
      f'(pairs {", ".join(map(str, s["steps_loss_gt_25pct"])) or "none"}). Splits: {s["n_split"]}; merges: '
      f'{s["n_merge"]}; links with either: {s["n_split_or_merge"]}.')
    w('')
    if m4 is not None:
        worst = max(m4[c]['max_abs_diff_over_rms'] for c in ('r_matched', 'r_store', 'two_F'))
        w(f'**Check against M4.** The three rates equal the series in `data/{m4["source"]}` (the numbers behind the '
          f'report\'s Figure 11) {"exactly" if worst == 0 else f"to {worst:.1e} of their r.m.s."} at every step; the labels are '
          f'{"identical" if m4["labels_equal"] else "NOT identical"}.')
        w('')
    w(f'![]({parcels_png})')
    w('')
    w(f'**Figure 2.** The matched parcels of pairs {", ".join(map(str, detail_steps))}, one row per pair, each '
      f'parcel drawn at its seed cell. Left: the per-parcel centred difference `[G(t+1, x⁺) − G(t−1, x⁻)] / 2Δt`; middle: the stored tendency; '
      f'then `2F`{" and the surface-flux term at 1 m (§5)" if budget_png else ""}. The rate quoted for a step is the mean of the map.')
    w('')
    rm, fh, it, pc, rf, bm = (s['rms'], s['frac_hours'], s['integral'], s['per_parcel_corr_median'],
                              s['ridge_flank_mean'], s['budget_mean'])
    u, g = UNIT, 1e-14
    w('## 4. `2F` against the stored `DG/Dt`: what the agreement does and does not say')
    w('')
    w('**Do the two track one another through the window?** The numbers, hour by hour on the band means '
      f'({s2["n"]} steps):')
    w('')
    w('| pair of rates | correlation | r.m.s. difference [10⁻¹⁹ s⁻⁵] | sign agreement | mean difference [10⁻¹⁹ s⁻⁵] |')
    w('|---|---|---|---|---|')
    w(f'| stored − `2F` | {s2["corr"]:.3f} | {rm["store_minus_two_F"] / u:.2f} | {s2["sign_agree"]:.2f} | '
      f'{_f((mr["r_store"] - mr["two_F"]) / u)} |')
    w(f'| matched − stored | {mv["corr"]:.3f} | {rm["matched_minus_store"] / u:.2f} | {mv["sign_agree"]:.2f} | '
      f'{_f((mr["r_matched"] - mr["r_store"]) / u)} |')
    w(f'| matched − `2F` | {m2["corr"]:.3f} | {rm["matched_minus_two_F"] / u:.2f} | {m2["sign_agree"]:.2f} | '
      f'{_f((mr["r_matched"] - mr["two_F"]) / u)} |')
    w('')
    w(f'For scale, the r.m.s. of the hourly rates is {rm["two_F"] / u:.2f} (`2F`), {rm["r_store"] / u:.2f} (stored) and '
      f'{rm["r_matched"] / u:.2f} (matched). So on this track the stored tendency and `2F` agree in the hour-to-hour '
      f'shape, in sign on {s2["sign_agree"]:.0%} of hours and in the mean to {abs(mr["r_store"] - mr["two_F"]) / u:.2f}; '
      f'the stored tendency is the larger on {fh["store_gt_two_F"]:.0%} of hours. Of the three pairs of rates, '
      f'stored and `2F` are the {"closest" if rm["store_minus_two_F"] <= min(rm["matched_minus_store"], rm["matched_minus_two_F"]) else "not the closest"} '
      f'in r.m.s.')
    w('')
    w('**Does this imply that the front strengthens or weakens at the rate predicted by `2F`?** Not on its own. '
      'What the agreement establishes is narrow: averaged over the band of that hour\'s parcels, the one-hour '
      'semi-Lagrangian tendency of `G` has the sign, timing and amplitude of the kinematic prediction. Four things '
      'stand between that and a statement about the front.')
    w('')
    w(f'1. **The agreement is between band means, not between parcels.** Parcel by parcel within a step, the stored '
      f'tendency and `2F` correlate at a median {pc["store_two_F"]:.2f}; the matched rate and the stored tendency, '
      f'two measurements of the same thing, at {pc["rate_store"]:.2f}; the matched rate and `2F` at '
      f'{pc["rate_two_F"]:.2f}. Split by position in the band (means over the track, 10⁻¹⁹ s⁻⁵):')
    w('')
    w('   | | `2F` | stored `DG/Dt` | matched |')
    w('   |---|---|---|---|')
    w(f'   | ridge (the label\'s own pixels, {s["ridge_fraction_median"]:.0%} of the band) | {_f(rf["two_F_ridge"] / u)} | '
      f'{_f(rf["store_ridge"] / u)} | {_f(rf["rate_ridge"] / u)} |')
    w(f'   | flanks | {_f(rf["two_F_flank"] / u)} | {_f(rf["store_flank"] / u)} | {_f(rf["rate_flank"] / u)} |')
    w(f'   | whole band | {_f(mr["two_F"] / u)} | {_f(mr["r_store"] / u)} | {_f(mr["r_matched"] / u)} |')
    w('')
    w('   The band means of `2F` and the stored tendency are close, but they are made up differently: `2F` is '
      'spread over ridge and flanks, while the stored tendency is concentrated on the ridge. The report (§9.3) '
      'showed that the stored construction has a positive registration bias on one-cell ridges, which a band '
      'cancels to first order; how much of the ridge value here is that bias is not established.')
    w(f'2. **The other measurement of the same rate gives a different level.** The matched rate is a second estimate '
      f'of the Lagrangian rate of the same parcels. It follows the same shape (correlation {mv["corr"]:.3f} with the '
      f'stored tendency) but its mean is {_f(mr["r_matched"] / u)} against {_f(mr["r_store"] / u)}, it is below the '
      f'stored tendency on {fh["matched_lt_store"]:.0%} of hours and it is positive on {fp["r_matched"]:.0%} of hours '
      f'against {fp["r_store"]:.0%}. The two measurements differ from each other '
      f'({rm["matched_minus_store"] / u:.2f} r.m.s.) by {"more" if rm["matched_minus_store"] > rm["store_minus_two_F"] else "less"} '
      f'than the stored one differs from `2F` ({rm["store_minus_two_F"] / u:.2f}). Whether the parcels\' `G` changes '
      f'*at the rate* `2F` therefore depends on which measurement is taken; which is nearer the truth is not settled.')
    w(f'3. **Integrated over the window the three disagree.** Summed over the {it["hours"]:.0f} hours between the '
      f'first and last step with a rate (trapezoid rule, 10⁻¹⁴ s⁻⁴): `2F` {_f(it["two_F"] / g)}, stored '
      f'{_f(it["r_store"] / g)}, matched {_f(it["r_matched"] / g)}. Over the same hours the mean `G` of the band of '
      f'the label as found went from {it["G_band_first"] / g:.2f} to {it["G_band_last"] / g:.2f}, a change of '
      f'{_f(it["dG_band"] / g)}. None of these integrals is the history of one set of water: the parcels are '
      f're-seeded on a new label every hour (§6), and water followed for a day leaves the front (report §5.3). '
      f'None of their ratios is an efficiency.')
    if 'surface_flux' in bm:
        w(f'4. **The budget is not closed on these parcels.** On the same parcels the measured surface-flux term '
          f'averages {_f(bm["surface_flux"] / u)}, the vertical term {_f(bm["vertical"] / u)} and the residual '
          f'{_f(bm["residual"] / u)} (10⁻¹⁹ s⁻⁵). The stored tendency is close to `2F` alone, not to `2F` plus the '
          f'other measured terms, so the near-equality is not a closed budget (report §7.2). §5 takes the '
          f'surface-flux term up in both of its forms.')
    w('')
    w('**In short.** Hour by hour, the mean tendency of `G` on this front\'s parcels rises and falls with `2F`, '
      'and by the stored measurement it does so at about the predicted amplitude. That is a consistency check on '
      'band-mean rates. It does not show that the front, as an object, strengthens at the rate `2F`, and by the '
      'matched measurement even its water does not, in the mean. One front, one 72-hour window.')
    w('')
    sf = s.get('surface_flux')
    if sf is not None:
        w('## 5. The surface-flux term on the matched parcels')
        w('')
        w('**Can it be included?** Yes. The derived store holds the surface-flux term of the five-term budget for '
          'every pair, on the same grid and at the same midpoint time as `2F`, so it is read at the seed cells of '
          'the matched parcels exactly as `2F` is. Nothing new was computed from the model output.')
        w('')
        w('### 5.1 What the term is')
        w('')
        w('The budget of the report (§2.3) at `L = 0`, where the sub-filter term is identically zero, is')
        w('')
        w('$$\\frac{D_h G}{Dt} = 2F + V + S + R,$$')
        w('')
        w('with $V$ the vertical term, $S$ the surface-flux term and $R$ the residual. The model forces its top cell '
          'with the net heat flux $Q_{\\rm net}$, the shortwave flux $Q_{\\rm sw}$ and the freshwater flux $F_w$ '
          '(positive into the ocean). Within a layer of thickness $h$ (report §2.4),')
        w('')
        w('$$\\left.\\frac{\\partial\\Theta}{\\partial t}\\right|_{\\rm sfc} = \\frac{(Q_{\\rm net} - Q_{\\rm sw}) + f_{\\rm sw}(h)\\,Q_{\\rm sw}}{\\rho_c\\, c_p\\, h}, \\qquad \\left.\\frac{\\partial\\mathcal{S}}{\\partial t}\\right|_{\\rm sfc} = -\\frac{\\mathcal{S}\\,F_w}{\\rho_c\\, h},$$')
        w('')
        w('$$B_{\\rm sfc} = \\frac{g}{\\rho_0}\\left(-\\rho\\,\\alpha\\left.\\frac{\\partial\\Theta}{\\partial t}\\right|_{\\rm sfc} + \\rho\\,\\beta\\left.\\frac{\\partial\\mathcal{S}}{\\partial t}\\right|_{\\rm sfc}\\right), \\qquad S_t(\\mathbf{x}) = 2\\,\\nabla_h b^{m} \\cdot \\nabla_h B_{\\rm sfc}, \\qquad \\overline{S}(t) = \\frac{1}{N_t}\\sum_{p \\in P_t} S_t(\\mathbf{x}_p).$$')
        w('')
        w('The fluxes, $\\Theta$ and $\\mathcal{S}$ are midpoint means of the two hours; the gradients use the same '
          'stencil $\\mathsf{L}$ as `G` and `2F`. $S > 0$ sharpens the front and $S < 0$ weakens it. Two depth '
          'scales are stored, as in the report:')
        w('')
        w('- **`surface_flux`, $h = \\Delta z = 1$ m** (the top cell). The primary form of the report: the term the '
          'top cell obeys instant by instant.')
        w('- **`surface_flux_kpp`, $h = h_{\\rm bl}$** (the KPP boundary-layer depth at each cell, floored at 1 m). '
          'The declared sensitivity of report §7.3: the hourly-mean tendency once the mixing scheme has spread the '
          f'flux through the boundary layer. On these parcels $h_{{\\rm bl}}$ has a median of {sf["hbl_median"]:.1f} m '
          f'(step means {sf["hbl_min"]:.1f}–{sf["hbl_max"]:.1f} m).')
        w('')
        w(f'Like `2F`, the term is instantaneous at $t + \\tfrac12$ h and is read at the seed cell with no '
          f'interpolation. It is finite on {"every parcel at every step" if sf["same_parcels"] else "NOT every parcel"}, '
          f'so its mean is over the same $N_t$ parcels as the three rates of §2. The surface fluxes are six-hourly '
          f'forcing interpolated by the model, so the shortwave cycle is a triangle, not resolved insolation.')
        w('')
        w('### 5.2 What it gives on this track')
        w('')
        w(f'![]({budget_png})')
        w('')
        w('**Figure 3.** Top: the measured rates (stored, matched) against `2F` alone and against the sum of the '
          'predicted terms with the boundary-layer form of $S$. Middle: the surface-flux term in its two forms, with '
          'the two imbalances it would have to explain (stored − `2F` − $V$ and matched − `2F` − $V$); the 1 m form is '
          'on the right-hand axis. Bottom: the boundary-layer depth on the parcels. Grey bands: local night '
          '(18–06 local solar time, UTC − 8 h).')
        w('')
        w('| over the track [10⁻¹⁹ s⁻⁵] | mean | r.m.s. | fraction of hours positive |')
        w('|---|---|---|---|')
        for nm, key in (('`2F`', 'two_F'), ('stored `DG/Dt`', 'r_store'), ('matched rate', 'r_matched'),
                        ('vertical term $V$', 'vertical'), ('$S$, 1 m', 'S_1m'), ('$S$, $h_{\\rm bl}$', 'S_kpp')):
            t_ = sf['terms'][key]
            w(f'| {nm} | {_f(t_["mean"] / UNIT)} | {t_["rms"] / UNIT:.2f} | {t_["frac_pos"]:.2f} |')
        w('')
        w('The residual left on the band means, as the r.m.s. over the track of measured − predicted:')
        w('')
        w('| predicted side | against stored `DG/Dt` | against the matched rate |')
        w('|---|---|---|')
        for nm, key in (('`2F` alone', 'two_F'), ('`2F` + $V$', 'two_F_V'), ('`2F` + $V$ + $S$ (1 m)', 'full_1m'),
                        ('`2F` + $V$ + $S$ ($h_{\\rm bl}$)', 'full_kpp')):
            w(f'| {nm} | {sf["residual_rms"]["r_store"][key] / UNIT:.2f} | {sf["residual_rms"]["r_matched"][key] / UNIT:.2f} |')
        w('')
        t1, tk = sf['terms']['S_1m'], sf['terms']['S_kpp']
        rs, rmm = sf['residual_rms']['r_store'], sf['residual_rms']['r_matched']
        ps = sf['projection']
        w(f'1. **The 1 m form is large and one-signed here.** It averages {_f(t1["mean"] / UNIT)} and is positive on '
          f'{t1["frac_pos"]:.0%} of hours, day and night '
          f'(means by local time: night {_f(sf["S_1m_night"] / UNIT)}, day {_f(sf["S_1m_day"] / UNIT)}); '
          f'{sf["frac_parcels_pos_median"]:.0%} of the parcels are positive in the median step. Its r.m.s., '
          f'{t1["rms"] / UNIT:.2f}, is {t1["rms"] / sf["terms"]["r_store"]["rms"]:.1f} times that of the stored tendency. '
          f'It is larger on the ridge ({_f(sf["S_1m_ridge"] / UNIT)}) than on the flanks ({_f(sf["S_1m_flank"] / UNIT)}).')
        w(f'2. **Adding it at unit weight makes the budget of these parcels worse.** The residual against the stored '
          f'tendency goes from {rs["two_F_V"] / UNIT:.2f} to {rs["full_1m"] / UNIT:.2f}, and against the matched rate '
          f'from {rmm["two_F_V"] / UNIT:.2f} to {rmm["full_1m"] / UNIT:.2f}. The measured tendency shows no sign of a '
          f'forcing of this size: this is the report\'s finding (§5.2, §7.2) on one front.')
        w(f'3. **It does not project onto what is missing.** Hour by hour, the 1 m term correlates with the imbalance '
          f'stored − `2F` − $V$ at {ps["S_1m"]["r_store"]["corr"]:+.2f} and the multiplier on it that would minimise the '
          f'residual is {ps["S_1m"]["r_store"]["multiplier"]:+.3f} (against the matched imbalance: '
          f'{ps["S_1m"]["r_matched"]["corr"]:+.2f} and {ps["S_1m"]["r_matched"]["multiplier"]:+.3f}); a correct term would '
          f'give +1. Parcel by parcel within a step the correlation with stored − `2F` has a median of '
          f'{sf["per_parcel_corr_S_imbalance_median"]:+.2f}. The pooled value in the report is a multiplier of +0.048 '
          f'at `L = 0`.')
        w(f'4. **The boundary-layer form is small and changes sign.** It averages {_f(tk["mean"] / UNIT)} with an r.m.s. '
          f'of {tk["rms"] / UNIT:.2f} — {tk["rms"] / t1["rms"]:.3f} of the 1 m form — and is positive on '
          f'{tk["frac_pos"]:.0%} of hours. Adding it leaves the residual where `2F` + $V$ left it '
          f'({rs["full_kpp"] / UNIT:.2f} against {rs["two_F_V"] / UNIT:.2f} for the stored tendency; '
          f'{rmm["full_kpp"] / UNIT:.2f} against {rmm["two_F_V"] / UNIT:.2f} for the matched rate). Its correlation with '
          f'the stored imbalance is {ps["S_kpp"]["r_store"]["corr"]:+.2f} and with the matched imbalance '
          f'{ps["S_kpp"]["r_matched"]["corr"]:+.2f}. The two forms are not copies of one another: their hourly band '
          f'means correlate at {sf["corr_S_1m_S_kpp"]:+.2f}, because $h_{{\\rm bl}}$ varies in space and through the day.')
        w(f'5. **The vertical term is negligible** (r.m.s. {sf["terms"]["vertical"]["rms"] / UNIT:.2f}).')
        sc = s['set_change']
        if sc['n_jumps'] and 'S_1m' in sc['series']:
            q = sc['series']
            w(f'6. **The 1 m term follows the pixel set; the three rates do not.** This bears on the concern of this '
              f'analysis. Over the {sc["n_links"]} consecutive steps, the size of the hour-to-hour change of each band '
              f'mean is compared with the size of the change of the label (`|Δ log npix|`); a jump is a change of more '
              f'than {sc["jump_threshold"]:.2f} in the log ({sc["n_jumps"]} links):')
            w('')
            w('   | band mean | correlation of its hourly change with the change of label size | median change at a jump | median change elsewhere |')
            w('   |---|---|---|---|')
            for nm, key in (('`2F`', 'two_F'), ('stored `DG/Dt`', 'r_store'), ('matched rate', 'r_matched'),
                            ('$S$, 1 m', 'S_1m'), ('$S$, $h_{\\rm bl}$', 'S_kpp')):
                if key in q:
                    w(f'   | {nm} | {q[key]["corr"]:+.2f} | {q[key]["median_change_jump"] / UNIT:.2f} | '
                      f'{q[key]["median_change_other"] / UNIT:.2f} |')
            w('')
            w('   The steps in the red curve of Figure 3 are the hours at which the tracker moves to a shorter or a '
              'longer label: the 1 m term has a different mean on different parts of the front, so its band mean '
              'changes when the set does (Figure 2, right-hand column, shows where on the front it is concentrated). '
              'The hourly changes of `2F` and of the two measured rates are no larger at a jump than elsewhere.')
        w('')
        w(f'Integrated over the {it["hours"]:.0f} hours (trapezoid, 10⁻¹⁴ s⁻⁴): $S$ at 1 m {_f(sf["integral"]["S_1m"] / 1e-14)}, '
          f'$S$ over $h_{{\\rm bl}}$ {_f(sf["integral"]["S_kpp"] / 1e-14)}, against `2F` {_f(it["two_F"] / 1e-14)}, stored '
          f'{_f(it["r_store"] / 1e-14)} and matched {_f(it["r_matched"] / 1e-14)}. The 1 m integral is what carries the '
          f'four-term curve off the axes in the report\'s Figure 11.')
        w('')
        w('**Reading.** On this front the surface-flux term does not account for the difference between the measured '
          'rates and `2F`, in either form: at 1 m it is far too large and steady, and over the boundary layer it is '
          'too small to matter. That is the same result as the pooled budget of the report, and it does not reopen '
          'it: the depth scale is a declared sensitivity, not a measurement. It also means the gap between the '
          'matched rate and `2F` (§4, point 2) is not explained by the surface flux as stored.')
        w('')
    w('## 6. The chosen label and the advected previous label, pair by pair')
    w('')
    w('North is up (`j` runs east, `i` runs south); the crop is the same in every panel. Black: a cell in both '
      'the chosen label and the previous label advected by the flow. Blue: chosen label only. Orange: advected '
      'previous label only. Grey: other fronts at that pair. The thin green outline is the band on which the '
      'matched parcels are seeded. Each panel\'s heading gives the three rates of that pair\'s matched parcels. '
      '"Previous" is the previous pair in time (`t − 1`) at every step, including those before the anchor, where '
      'the tracker itself walked backwards from `t + 1`.')
    w('')
    for k, (png, steps) in enumerate(pages):
        w(f'![]({png})')
        w('')
        w(f'**Figure {(4 if budget_png else 3) + k}.** Pairs {steps[0]}–{steps[-1]}.')
        w('')
    w('## 7. Table')
    w('')
    w('Rates in 10⁻¹⁹ s⁻⁵. "same cell": pixels in both sets; "gained" / "lost": chosen / advected pixels more '
      'than one cell from the other set. Flag: S = split, M = merge. The last two columns are the surface-flux '
      'term of §5 at the two depth scales.')
    w('')
    w('| pair | time (UTC) | label | px | prev label | advected px | same cell | gained | lost | flag | parcels | `2F` | stored `DG/Dt` | matched | $S$ (1 m) | $S$ ($h_{\\rm bl}$) |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for _, r in df.sort_values('step').iterrows():
        has = pd.notna(r['prev_step'])
        flag = ('S' if has and r['split'] else '') + ('M' if has and r['merge'] else '')
        i_ = (lambda v: '—' if pd.isna(v) else str(int(v)))
        r_ = (lambda v: '—' if pd.isna(v) else _f(v / UNIT))
        w(f'| {int(r["step"])}{" (anchor)" if r["step"] == a else ""} | {r["time"][5:13].replace("T", " ")}:30 | '
          f'{int(r["label"])} | {int(r["npix"])} | {i_(r["prev_label"])} | {i_(r.get("n_advected"))} | '
          f'{i_(r.get("n_both"))} | {i_(r.get("n_gained"))} | {i_(r.get("n_lost"))} | {flag} | {i_(r.get("n_parcels"))} | '
          f'{r_(r.get("two_F"))} | {r_(r.get("r_store"))} | {r_(r.get("r_matched"))} | {r_(r.get("S_1m_mean"))} | '
          f'{r_(r.get("S_kpp_mean"))} |')
    w('')
    Path(path).write_text('\n'.join(L))
    return str(path)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def default_detail_steps(track, df):
    """The anchor and the two links (one each side, when there are) where the chosen label
    has the most pixels more than one cell from the advected previous label."""
    d = df.dropna(subset=['r_matched', 'prev_step'])
    out = [track.anchor.step]
    for side in (d[d['step'] < track.anchor.step], d[d['step'] > track.anchor.step]):
        if len(side):
            out.append(int(side.loc[side['n_gained'].idxmax(), 'step']))
    return sorted(set(out))


def run(track=None, label=None, anchor_step=None, detail_steps=None, out_dir=None, data_dir=None,
        log=print):
    import budget
    from m4_track import FLOW_PATH
    out_dir, data_dir = Path(out_dir or OUT_DIR), Path(data_dir or DATA_DIR)
    flow = tr.Flow(store=FLOW_PATH)
    labels = tr.open_fronts_series(tr.fronts_path(0))
    keys = list(labels)
    fields = tr.DerivedFields(budget.derived_path(0))
    trk, index, source = get_track(labels, flow, track=track, label=label, anchor_step=anchor_step, log=log)
    name = out_name(trk, len(keys))
    log(f'{name}: {source}; {len(trk.steps())} steps')
    df = analyse_track(trk, labels, flow, fields, log=log)
    summ = track_summary(df)
    m4 = check_against_m4(df, index)
    fig_dir = out_dir / 'figs'
    fig_dir.mkdir(parents=True, exist_ok=True)
    series_png = fig_series(trk, df, fig_dir / f'{name}_series.png')
    budget_png = (fig_budget(trk, df, fig_dir / f'{name}_budget.png')
                  if summ.get('surface_flux') is not None and 'S_kpp_mean' in df else None)
    detail = sorted(detail_steps) if detail_steps else default_detail_steps(trk, df)
    parcels_png, detail = fig_parcels(trk, labels, flow, fields, detail, fig_dir / f'{name}_parcels.png')
    st = trk.steps()
    pages = []
    for k in range(0, len(st), PANELS_PER_PAGE):
        steps = st[k:k + PANELS_PER_PAGE]
        png = fig_dir / f'{name}_sets_{steps[0]:02d}_{steps[-1]:02d}.png'
        fig_sets(trk, df, labels, flow, fields, steps, png,
                 title=f'Track of label {trk.anchor.label} (anchor pair {trk.anchor.step}): chosen label and '
                       f'advected previous label, pairs {steps[0]}–{steps[-1]}')
        pages.append((f'figs/{png.name}', steps))
        log(f'  drew {png.name}')
    md = write_markdown(out_dir / f'{name}.md', name, trk, source, df, summ, m4, pages,
                        f'figs/{Path(series_png).name}', f'figs/{Path(parcels_png).name}', detail, keys,
                        budget_png=f'figs/{Path(budget_png).name}' if budget_png else None)
    out = dict(name=name, source=source, track_index=index, anchor=dict(label=trk.anchor.label, step=trk.anchor.step),
               params=dict(L=0, config='D', band=BAND, centred=True, dt=DT, mask='mask_analysis',
                           split_min_pixels=tr.SPLIT_MIN_PIXELS, reference='previous found pair in time'),
               summary=summ, check_m4=m4, detail_steps=detail, rows=df.to_dict('records'),
               figures=[series_png, parcels_png] + ([budget_png] if budget_png else []) + [str(out_dir / p) for p, _ in pages], markdown=md,
               created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'))
    tr.dump_json(out, data_dir / f'{name}.json')
    log(f'wrote {md} and {data_dir / (name + ".json")}')
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--track', type=int, default=None, help='index into the M4 flow-informed tracks (0-9)')
    ap.add_argument('--label', type=int, default=None, help='anchor label (default: track 0, label 81)')
    ap.add_argument('--anchor-step', type=int, default=None, help='anchor pair (default: the middle pair, 35)')
    ap.add_argument('--detail-steps', type=int, nargs='*', default=None,
                    help='pairs for the per-parcel figure (default: the anchor and the largest gain each side)')
    args = ap.parse_args(argv)
    run(track=args.track, label=args.label, anchor_step=args.anchor_step, detail_steps=args.detail_steps)
    return 0


if __name__ == '__main__':
    sys.exit(main())
