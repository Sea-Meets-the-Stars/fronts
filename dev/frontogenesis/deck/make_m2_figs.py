"""Figures for the M2 acceptance deck.

Every number here is quoted from the M2 task logs in
``claude_prompts/frontogenesis_prompts.md`` ("Execution prompt 3", tasks 1-6 and the M2-Q7
entry, 2026-10-03 .. 2026-10-04), from ``frontogenesis_prompt_3.md``, or is read from the
small summary / done-files the tasks left in ``../data/`` (JSON only).  Nothing is
recomputed from the stores, no zarr is opened and no network is touched.

Outputs, all in ``figs_m2/``:

  crop    panel (c) of ``../figs/m2_q7_edge_margin.png`` (the raw Theta profile across the
          day-3 front) at its native 200 dpi
  simple  large-font re-plots sized to the width at which the deck places them:
            m2_fig_osn_walltime     per-hour wall time of the 72-hour OSN pull
                                    (data/m2_pull_done_run1.json)
            m2_fig_chunk_walltime   per-hour wall time of the chunk pull, with the two
                                    timeouts, two slow reads and the sleep stall marked
                                    (data/m2_chunk_pull_done_run1.json + the task-5 log)
            m2_fig_qa_series        Eta tile mean (the tide), KPPhbl tile mean (the diurnal
                                    cycle), hourly displacement envelope
                                    (data/m2_qa_hours.json, data/m2_qa_pairs.json)
            m2_fig_stability        V3 llc slope on all 71 pairs, with CI, trimmed slope,
                                    gate and baseline bands (data/m2_v3_stability.json)
            m2_fig_flux_diurnal     raw chunk-store oceQsw / oceQnet tile means over 07-03,
                                    upward-positive, 6-hourly kinks (data/m2_chunk_recon.json)

Run:  <frontogenesis env python> make_m2_figs.py
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).parent
OUT = HERE / "figs_m2"
OUT.mkdir(exist_ok=True)
FIGS = HERE.parent / "figs"
DATA = HERE.parent / "data"

MIDNIGHT, DEEP, TEAL = "#21295C", "#065A82", "#1C7293"
AMBER, MUTED, LIGHT, GREEN = "#A8501B", "#62737F", "#F2F6F8", "#2C6E49"
RED = "#B03A2E"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 20,
    "axes.titlesize": 21, "axes.labelsize": 20, "legend.fontsize": 18,
    "xtick.labelsize": 19, "ytick.labelsize": 19,
    "axes.edgecolor": MUTED, "axes.labelcolor": MIDNIGHT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": MIDNIGHT,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)


def load(name):
    p = DATA / name
    return json.load(open(p)) if p.exists() else None


# ---------------------------------------------------------------- crop of the real figure
# (left, top, right, bottom) in native pixels of figs/m2_q7_edge_margin.png (3400 x 1040).
# Panels (a) and (b) are not cropped: (a) is carried by the table on the slide and (b) is a
# log-scale cloud of 24 series that does not read at slide size.
CROPS = {
    "m2_q7_edge_margin.png": [("m2_crop_q7c.png", (2500, 0, 3400, 1040))],
}
for src, items in CROPS.items():
    p = FIGS / src
    if p.exists():
        for dst, box in items:
            Image.open(p).crop(box).save(OUT / dst, optimize=True)

# ---------------------------------------------------------------- task 2: OSN per-hour wall
# Task-2 log: total 1636.9 s = 27.3 min; median 22.4 s, mean 22.7, range 20.7-33.1 s;
# slowest 2012-07-03 10:00 (33.1 s).  0 retries, 0 failures, 0 repairs.
done = load("m2_pull_done_run1.json")
if done:
    walls = np.array(list(done["report"]["wall_s"].values()))
    assert len(walls) == 72
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.2)
    ax.bar(np.arange(72), walls, 0.8, color=DEEP)
    ax.axhline(22.4, color=AMBER, ls="--", lw=1.8)
    ax.text(50, 36.5, "median 22.4 s (dashed)", color=AMBER, fontsize=14, ha="center")
    i = int(np.argmax(walls))
    ax.annotate(f"{walls[i]:.1f} s\n07-03 10:00", (i, walls[i]), (i - 14, walls[i] + 2),
                fontsize=14, color=MIDNIGHT, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_ylim(0, 42); ax.set_xlim(-1, 72)
    ax.set_xticks([0, 24, 48, 71]); ax.tick_params(labelsize=15)
    ax.set_xlabel("hour of the window", fontsize=16)
    ax.set_ylabel("wall time per hour [s]", fontsize=16)
    ax.set_title("OSN: 72 hours in 27.3 min, no retries", loc="left", fontweight="bold",
                 fontsize=17)
    save(fig, "m2_fig_osn_walltime.png")

# ---------------------------------------------------------------- task 5: chunk per-hour wall
# Task-5 log: 25,305 s = 7.03 h; median 299.2 s, range 297.2-1886.9 s; two FSTimeoutErrors
# on W (20120702T09, 860 s; 20120703T02, 730 s), two slow reads at 0.12 MB/s (20120702T17
# Salt, 621 s; 20120702T20 Theta, 683 s), one 28-min idle-sleep stall (20120704T10 Salt,
# 1887 s; caffeinate afterwards).  The hour indices below are those timestamps' positions.
done = load("m2_chunk_pull_done_run1.json")
if done:
    ws = done["report"]["wall_s"]
    keys = list(ws.keys()); walls = np.array([ws[k] for k in keys])
    assert len(walls) == 72
    events = {  # index -> (label, colour), from the task-5 log
        keys.index("2012-07-02 09:00:00"): ("timeout (W)", AMBER),
        keys.index("2012-07-03 02:00:00"): ("timeout (W)", AMBER),
        keys.index("2012-07-02 17:00:00"): ("slow read", MUTED),
        keys.index("2012-07-02 20:00:00"): ("slow read", MUTED),
        keys.index("2012-07-04 10:00:00"): ("28-min stall\n(idle sleep)", RED),
    }
    assert set(events) == set(np.argsort(walls)[-5:]), "the five slow hours are the logged ones"
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    fig.subplots_adjust(left=0.19, right=0.98, top=0.86, bottom=0.2)
    cols = [events.get(i, ("", DEEP))[1] for i in range(72)]
    ax.bar(np.arange(72), walls / 60, 0.8, color=cols)
    ax.axhline(299.2 / 60, color=AMBER, ls="--", lw=1.5)
    ax.text(30, 6.0, "median 299 s", color=AMBER, fontsize=15)
    for i, (lab, c) in events.items():
        if lab.startswith("timeout"):
            ax.text(i, walls[i] / 60 + 0.8, "timeout", ha="right" if i < 20 else "left",
                    fontsize=13, color=AMBER)
        elif lab.startswith("28"):
            ax.text(i - 1, walls[i] / 60 - 4, "28-min\nstall", ha="right", fontsize=14,
                    color=RED)
    ax.text(18.5, 18.5, "slow\nreads", ha="center", fontsize=13, color=MUTED)
    ax.set_ylim(0, 34); ax.set_xlim(-1, 72)
    ax.set_xticks([0, 24, 48, 71]); ax.tick_params(labelsize=15)
    ax.set_xlabel("hour of the window", fontsize=16)
    ax.set_ylabel("wall time per hour [min]", fontsize=16)
    ax.set_title("Chunk store: 72 hours in 7.03 h", loc="left", fontweight="bold",
                 fontsize=17)
    save(fig, "m2_fig_chunk_walltime.png")

# ---------------------------------------------------------------- task 3: QA series
# Task-3 log: Eta tile-mean range 2.009 m, M2 0.619 + K1 0.494 m; KPPhbl 24-h amplitude
# 6.4 m, max ~01 h local solar; displacement over 71 pairs: ocean median 0.27-0.44, p99
# 1.05-1.38, max 4.05 (pair 59, Gulf of California); mask_analysis max 2.27 (pair 45).
hours, pairs = load("m2_qa_hours.json"), load("m2_qa_pairs.json")
if hours and pairs:
    hk = sorted(hours, key=int); t = np.arange(len(hk))
    eta = np.array([hours[k]["Eta"]["mean"] for k in hk])
    hbl = np.array([hours[k]["KPPhbl"]["mean"] for k in hk])
    pk = sorted(pairs, key=int); tp = np.arange(len(pk)) + 0.5
    med = np.array([pairs[k]["ocean"]["median"] for k in pk])
    p99 = np.array([pairs[k]["ocean"]["p99"] for k in pk])
    mx = np.array([pairs[k]["ocean"]["max"] for k in pk])
    mxa = np.array([pairs[k]["analysis"]["max"] for k in pk])
    fig, axs = plt.subplots(1, 3, figsize=(11.3, 3.3))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.22, wspace=0.42)
    for a in axs:
        a.tick_params(labelsize=15); a.set_xlim(0, 72); a.set_xticks([0, 24, 48, 72])
        for d in (24, 48):
            a.axvline(d, color=MUTED, lw=0.8, ls=":")
        a.set_xlabel("hours since 07-02 00 UTC", fontsize=15)
    ax = axs[0]
    ax.plot(t, eta, color=DEEP, lw=2.5)
    ax.set_ylabel("Eta tile mean [m]", fontsize=15)
    ax.set_title("Tide: 2.0 m range", loc="left", fontweight="bold", fontsize=16)
    ax = axs[1]
    ax.plot(t, hbl, color=TEAL, lw=2.5)
    ax.set_ylabel("KPPhbl tile mean [m]", fontsize=15)
    ax.set_title("KPPhbl: 6.4 m diurnal", loc="left", fontweight="bold", fontsize=16)
    ax = axs[2]
    ax.plot(tp, mx, color=RED, lw=2.2, label="max, ocean")
    ax.plot(tp, mxa, color=RED, lw=2.0, ls="--", label="max, analysis")
    ax.plot(tp, p99, color=DEEP, lw=2.2, label="p99, ocean")
    ax.plot(tp, med, color=MIDNIGHT, lw=2.2, label="median, ocean")
    ax.set_ylim(0, 6.0); ax.set_yticks([0, 2, 4])
    ax.set_ylabel("displacement [cells]", fontsize=15)
    ax.set_title("Displacement, 71 pairs", loc="left", fontweight="bold", fontsize=16)
    ax.legend(frameon=False, fontsize=12, loc="upper left", ncol=2, columnspacing=0.8,
              handlelength=1.6, handletextpad=0.5)
    save(fig, "m2_fig_qa_series.png")

# ---------------------------------------------------------------- task 3 / M2-Q3: stability
# Task-3 log: all 71 pairs 0.902-0.997, mean 0.972 std 0.020, 64/71 pass; trimmed (top 1 %
# |2F| dropped) 0.971-1.002, mean 0.987 std 0.005; failures 36 and 62-67.
v3 = load("m2_v3_stability.json")
if v3:
    tab = sorted(v3["_table"], key=lambda r: r["t0"])
    t0 = np.array([r["t0"] for r in tab]); sl = np.array([r["slope"] for r in tab])
    lo = np.array([r["ci"][0] for r in tab]); hi = np.array([r["ci"][1] for r in tab])
    gate = np.array([r["gate"] for r in tab])
    trim = np.array([v3["_robust"][str(k)]["ols_trim1"] for k in t0])
    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.3)
    ax.axhspan(0.95, 1.05, color="#DCE8F2", zorder=0, label="gate 1 ± 0.05")
    ax.axhspan(0.970, 0.994, color="#C9DCC4", zorder=0, alpha=0.8,
               label="M1 baseline 0.981 [0.970, 0.994]")
    ax.fill_between(t0, lo, hi, color=MUTED, alpha=0.25, lw=0, label="95 % CI (block bootstrap)")
    ax.plot(t0, sl, color=MIDNIGHT, lw=2.2, label="OLS slope, the gate")
    ax.plot(t0, trim, color=TEAL, lw=2.0, ls="--", label="top 1 % |2F| trimmed")
    ax.plot(t0[~gate], sl[~gate], "o", color=RED, ms=8, label="fails the gate (7 of 71)")
    ax.set_ylim(0.80, 1.06); ax.set_xlim(-1, 71)
    ax.set_xticks([0, 24, 48, 70]); ax.tick_params(labelsize=15)
    for d in (24, 48):
        ax.axvline(d, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("first hour of the pair (hours since 07-02 00 UTC)", fontsize=16)
    ax.set_ylabel("V3 llc slope", fontsize=16)
    ax.set_title("V3 llc slope on all 71 pairs: 0.972 ± 0.020", loc="left",
                 fontweight="bold", fontsize=16)
    ax.legend(frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.45, -0.24),
              ncol=2, columnspacing=1.0, handletextpad=0.5, handlelength=1.6)
    save(fig, "m2_fig_stability.png")

# ---------------------------------------------------------------- task 4: raw flux diurnal
# Task-4 log item 8: the source attrs say "+=down" but oceQsw <= 0 everywhere (-589 W/m2 tile
# mean at 21 UTC = 13 LST), oceQnet +115 at night / -454 at noon; piecewise linear with
# kinks every 6 h at 03/09/15/21 UTC.  Tile means of 07-03 as m2_chunk_recon.py stored them.
rec = load("m2_chunk_recon.json")
if rec and "flux_diurnal" in rec:
    rows = rec["flux_diurnal"]["rows"]
    hh = sorted(rows, key=int); x = np.array([int(h) for h in hh])
    qsw = np.array([rows[h]["oceQsw"]["mean"] for h in hh])
    qnet = np.array([rows[h]["oceQnet"]["mean"] for h in hh])
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.84, bottom=0.2)
    for k in (3, 9, 15, 21):
        ax.axvline(k, color=MUTED, lw=1.0, ls=":")
    ax.axhline(0, color=MIDNIGHT, lw=1.0)
    ax.plot(x, qsw, "o-", color=AMBER, lw=2.2, ms=5, label="oceQsw")
    ax.plot(x, qnet, "s-", color=DEEP, lw=2.2, ms=5, label="oceQnet")
    ax.set_xlim(0, 23); ax.set_xticks([0, 6, 12, 18, 23]); ax.tick_params(labelsize=15)
    ax.set_ylim(-650, 200)
    ax.set_xlabel("UTC hour of 2012-07-03 (13 LST = 21 UTC)", fontsize=15)
    ax.set_ylabel("source tile mean [W m$^{-2}$]", fontsize=15)
    ax.set_title("Raw fluxes: upward-positive, 6-h kinks", loc="left",
                 fontweight="bold", fontsize=16)
    ax.legend(frameon=False, fontsize=14, loc="lower left")
    save(fig, "m2_fig_flux_diurnal.png")

print("wrote:")
for f in sorted(OUT.glob("*.png")):
    im = Image.open(f)
    print(f"  {f.relative_to(OUT.parent)}  {im.size[0]}x{im.size[1]}  ({f.stat().st_size/1024:.0f} KB)")
