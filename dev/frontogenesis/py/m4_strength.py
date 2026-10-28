"""M4 task 5 runner: front strength along the ten flow-informed tracks (primary: the label
mean of G through ``colocate_fronts_with_properties`` on the band; secondary: the cross-front
buoyancy step and width through ``curtains``), the population statistics with the
fronts-and-hours bootstrap, and Figures 8 and 9.

    cd dev/frontogenesis/py && /home/xavier/miniconda3/envs/frontogenesis/bin/python m4_strength.py [--case 0]

Outputs: ``data/m4_strength.json`` (every track's series, the population summary, the
figures' numbers), ``figs/fig08_front_case.png``, ``figs/fig09_population.png``,
``data/m4_strength.log``.
"""
import argparse
import datetime as dt
import json
import pathlib
import sys
import time
import traceback

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import budget                                                       # noqa: E402
import tracking as tr                                               # noqa: E402
import figures_m4 as f4                                             # noqa: E402
from m4_find import Log                                             # noqa: E402
from m4_track import FLOW_PATH, TRACKS_PATH                         # noqa: E402

DATA_DIR = tr.DATA_DIR
LOG_PATH = DATA_DIR / 'm4_strength.log'
OUT_PATH = DATA_DIR / 'm4_strength.json'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--case', type=int, default=0, help='the track Figure 8 shows (index, largest first)')
    args = ap.parse_args(argv)
    say = Log(LOG_PATH)
    started, t_start = dt.datetime.now(dt.timezone.utc), time.time()
    say(f'start: band {tr.BAND} (M4-Q3 (a)), centred rates; transects every {tr.TRANSECT_EVERY} px, '
        f'half-width {tr.TRANSECT_HALF_WIDTH}; bootstrap {tr.N_BOOT} draws over fronts and {tr.BOOT_BLOCK_H}-h blocks')
    status, error, out = 'ok', None, {}
    try:
        flow = tr.Flow(store=FLOW_PATH)
        labels = tr.open_fronts_series(tr.fronts_path(0))
        tracks = tr.tracks_from_json(json.load(open(TRACKS_PATH))['flow'])
        fields = tr.DerivedFields(budget.derived_path(0))
        series = {}
        for k, trk in enumerate(tracks):
            t0 = time.time()
            df = tr.front_strength_series(trk, labels, flow, fields, matched_pixels=True)
            series[k] = df
            d = df.dropna(subset=['r_matched'])
            say(f'track {k} (label {trk.anchor.label}): {len(df)} steps, npix {int(df["npix"].median())} median; '
                f'G_label {df["G_label"].iloc[0]:.2e} -> anchor {df.loc[df["step"] == trk.anchor.step, "G_label"].iloc[0]:.2e} '
                f'-> {df["G_label"].iloc[-1]:.2e}; material end {df["G_matched"].iloc[-1]:.2e}, predicted end '
                f'{df["G_pred"].iloc[-1]:.2e}, Phase-2 end {df["G_store"].iloc[-1]:.2e}; frontogenetic hours matched '
                f'{(d["r_matched"] > 0).mean():.2f} / 2F {(d["two_F"] > 0).mean():.2f}; Δb median {df["delta_b"].median():.2e}, '
                f'width median {df["width_km"].median():.1f} km ({int(df["n_transects"].median())} transects); '
                f'{time.time() - t0:.0f} s')
        summ = tr.population_summary(series)
        out['summary'] = summ
        p = summ['pooled']
        say(f'POOLED {p["n_links"]} links: frontogenetic fraction by the matched rate '
            f'{p["frac_frontogenetic_matched"]["value"]:.3f} [{p["frac_frontogenetic_matched"]["lo"]:.3f}, '
            f'{p["frac_frontogenetic_matched"]["hi"]:.3f}], by 2F {p["frac_frontogenetic_2F"]["value"]:.3f} '
            f'[{p["frac_frontogenetic_2F"]["lo"]:.3f}, {p["frac_frontogenetic_2F"]["hi"]:.3f}], by the stored rate '
            f'{p["frac_frontogenetic_store"]["value"]:.3f}; sign agreement {p["sign_agreement"]["value"]:.3f} '
            f'[{p["sign_agreement"]["lo"]:.3f}, {p["sign_agreement"]["hi"]:.3f}]; lifetimes '
            f'{summ["lifetime_h"]["min"]:.0f}-{summ["lifetime_h"]["max"]:.0f} h')
        out['fig08'] = f4.fig08_front_case(tracks[args.case], series[args.case], labels, fields)
        say(f'figure 8: {out["fig08"]["path"]}')
        out['fig09'] = f4.fig09_population(summ, series)
        say(f'figure 9: {out["fig09"]["path"]}')
        out['series'] = {str(k): df.to_dict('records') for k, df in series.items()}
    except Exception as e:                                   # noqa: BLE001
        status, error = 'error', traceback.format_exc()
        say(f'ERROR: {e!r}')
    out.update(status=status, started=started.isoformat(timespec='seconds'),
               finished=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
               wall_s=round(time.time() - t_start, 1), error=error,
               params=dict(band=tr.BAND, centred=True, transect_half_width=tr.TRANSECT_HALF_WIDTH,
                           transect_every=tr.TRANSECT_EVERY, n_boot=tr.N_BOOT, block_h=tr.BOOT_BLOCK_H,
                           case_track=args.case, tracks=str(TRACKS_PATH), flow=str(FLOW_PATH),
                           derived=str(budget.derived_path(0))))
    tr.dump_json(out, OUT_PATH)
    say(f'done: {status}, {time.time() - t_start:.0f} s; {OUT_PATH.name} written')
    say.close()
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
