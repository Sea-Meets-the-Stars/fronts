"""Figures for the M1 acceptance deck.

Every number here is quoted from the M1 task logs in
``claude_prompts/frontogenesis_prompts.md`` ("Execution prompt 2", tasks 1-7, 6b, 7a,
2026-09-28 .. 2026-09-30) and from ``frontogenesis_prompt_2.md``; nothing is recomputed
and no network or data store is touched.

Two kinds of output, all in ``figs_m1/``:

  crops   panels cut from the real V-figures in ``../figs/`` (V1a, V3d, V3b-c, V5a, V6a),
          at their native 200 dpi so the axis labels stay legible on a slide
  simple  large-font (20 pt+) re-plots of logged numbers where the real figure is too
          dense for a projected slide:
            m1_fig_operators     Jacobian vs flux-form strain regression, hour 0
            m1_fig_coarsegrain   budget closure with / without the subfilter term; term vs L
            m1_fig_gates         V1 error vs front width; V4 bias vs front width
            m1_fig_v3_changes    every change tried for the V3 gate (LLC variant)
            m1_fig_v3b           V3b: slope per truth with CI, LLC variant

Figures are sized in inches to the width at which the deck places them, so a 20 pt
label in the PNG is 20 pt on the slide.

Run:  <frontogenesis env python> make_m1_figs.py
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).parent
OUT = HERE / "figs_m1"
OUT.mkdir(exist_ok=True)
FIGS = HERE.parent / "figs"

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


# ---------------------------------------------------------------- crops of the real figures
# (left, top, right, bottom) in native pixels; the panel letters refer to each figure's own
# labelling.  The bounding boxes were read off the PNGs by eye.
CROPS = {
    "V1_cartesian_deformation.png": ("m1_crop_V1a.png", (0, 170, 1490, 1190)),
    "V3_discrete_null.png":         ("m1_crop_V3d.png", (1390, 1468, 2482, 2490)),
    "V3b_fv_null.png":              ("m1_crop_V3b_c.png", (1400, 180, 2520, 1320)),
    "V5_interp_half_cell.png":      ("m1_crop_V5a.png", (0, 160, 1340, 1240)),
    "V6_land_halo_tile330.png":     ("m1_crop_V6a.png", (100, 150, 1170, 1190)),
}
for src, (dst, box) in CROPS.items():
    p = FIGS / src
    if p.exists():
        im = Image.open(p).crop(box)
        if dst == "m1_crop_V3d.png":
            # panel (f)'s y-tick labels ("strain: ...", "LLC: ...") intrude at the right
            # edge of panel (d); paint them out (two small patches outside the axes).
            from PIL import ImageDraw
            d = ImageDraw.Draw(im)
            d.rectangle((1062, 30, im.width, 80), fill="white")
            d.rectangle((1062, 290, im.width, 345), fill="white")
        im.save(OUT / dst, optimize=True)

# ---------------------------------------------------------------- task 2: operators
# Task-2 log: Jacobian-derived strain regressed on the flux-form strain over mask_analysis
# (n 262,925): delta 0.853, sigma_n 0.854, sigma_s 1.000, |sigma| 0.925; and the
# gradb2 / grad_b2 interior-median ratio 0.9106.
fig, ax = plt.subplots(figsize=(6.2, 3.9))
labels = ["delta", "sigma_n", "sigma_s", "|sigma|", "gradb2 /\ngrad_b2"]
vals = [0.853, 0.854, 1.000, 0.925, 0.9106]
cols = [AMBER, AMBER, DEEP, TEAL, MUTED]
x = np.arange(len(vals))
ax.bar(x, vals, 0.6, color=cols)
ax.axhline(1.0, color=MIDNIGHT, lw=1.2, ls="--")
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=17, color=MIDNIGHT)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=16)
ax.set_ylim(0.7, 1.08); ax.set_ylabel("ratio to the flux form", fontsize=17)
ax.set_title("Jacobian vs flux-form strain, hour 0", loc="left", fontweight="bold",
             fontsize=18)
fig.tight_layout()
save(fig, "m1_fig_operators.png")

# ---------------------------------------------------------------- task 4: coarsegrain
# Task-4 log closure table (dx = 900 m, L = 8, dt = 3600): rms residual / rms measured,
# shear 0.47 without T, 0.041 with T, 0.041 flux form only; divergent 0.58 / 0.090 / 0.39.
# Real hour: rms(term)/rms(Fbar) 0.31 / 0.50 / 0.70 at L = 2 / 4 / 8; flux-only 2.2x.
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.1), width_ratios=[1.5, 1])
groups = ["shear flow", "divergent flow"]
without = [0.47, 0.58]; with_t = [0.041, 0.090]; flux_only = [0.041, 0.39]
x = np.arange(2); w = 0.22
ax.bar(x - w, without, w, color=MUTED, label="no subfilter term")
ax.bar(x, flux_only, w, color=AMBER, label="flux form only (no tau_delta)")
ax.bar(x + w, with_t, w, color=DEEP, label="with the term + tau_delta")
for xs, vs in ((x - w, without), (x, flux_only), (x + w, with_t)):
    for xi, v in zip(xs, vs):
        ax.text(xi, v + 0.015, f"{v:.2f}" if v >= 0.1 else f"{v:.3f}",
                ha="center", fontsize=15, color=MIDNIGHT)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=17)
ax.set_xlim(-0.55, 1.55)
ax.set_ylim(0, 0.95); ax.set_ylabel("residual / measured  (rms)", fontsize=17)
ax.set_title("Closure of the Gbar budget, 900 m, L = 8", loc="left",
             fontweight="bold", fontsize=18)
ax.legend(frameon=False, loc="upper left", fontsize=14, ncol=1,
          bbox_to_anchor=(0.0, 1.0))
Ls = [2, 4, 8]; ratio = [0.31, 0.50, 0.70]
ax2.bar(np.arange(3), ratio, 0.55, color=TEAL)
for xi, v in zip(np.arange(3), ratio):
    ax2.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=17, color=MIDNIGHT)
ax2.set_xticks(np.arange(3)); ax2.set_xticklabels([f"L = {L}" for L in Ls], fontsize=17)
ax2.set_ylim(0, 0.85); ax2.set_ylabel("rms term / rms Fbar", fontsize=17)
ax2.set_title("Hour 0: the term vs Fbar", loc="left", fontweight="bold", fontsize=18)
fig.tight_layout(w_pad=2.5)
save(fig, "m1_fig_coarsegrain.png")

# ---------------------------------------------------------------- task 5: V1 and V4 vs width
# V1 (task 5): 8-h chain max error 0.78 / 1.57 / 3.09 / 4.43 / 7.86 % at 8 / 6 / 4 / 3 / 2 dx;
# semi-Lagrangian step alone (order 3) 0.010 / 0.026 / 0.085 / 0.174 / 0.354 %.
# V4 (task 5): rms bias at the real-hour displacement distribution, % of G per hour,
# sigma_G = 1 / 1.5 / 2 / 3 / 4 / 6 cells: order 3 1.02 / 0.28 / 0.099 / 0.021 / 0.007 /
# 0.001; order 5 0.38 / 0.060 / 0.013 / 0.001; order 1 4.6 / 2.3 / 1.35 / 0.62 / 0.35 / 0.16.
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 4.3))
for a in (ax, ax2):
    a.tick_params(labelsize=15)
ell = [2, 3, 4, 6, 8]
chain = [7.86, 4.43, 3.09, 1.57, 0.78]
alone = [0.354, 0.174, 0.085, 0.026, 0.010]
ax.plot(ell, chain, "o-", color=AMBER, lw=2.5, ms=8)
ax.plot(ell, alone, "s-", color=DEEP, lw=2.5, ms=8)
ax.axhline(1.0, color=RED, ls="--", lw=1.5)
ax.text(2.0, 1.25, "1% gate", color=RED, fontsize=14, ha="left")
ax.text(4.3, 4.5, "G vs exp(2at)\nover 8 h", color=AMBER, fontsize=14, ha="left")
ax.text(3.2, 0.09, "semi-Lagrangian\nstep alone", color=DEEP, fontsize=14, ha="left")
ax.set_yscale("log"); ax.set_xticks(ell); ax.set_ylim(0.006, 15)
ax.set_xlabel("front width  ell / dx", fontsize=16)
ax.set_ylabel("max error  [%]", fontsize=16)
ax.set_title("V1 vs front width", loc="left", fontweight="bold", fontsize=17)
sg = [1, 1.5, 2, 3, 4, 6]
o1 = [4.6, 2.3, 1.35, 0.62, 0.35, 0.16]
o3 = [1.02, 0.28, 0.099, 0.021, 0.007, 0.001]
o5 = [0.38, 0.060, 0.013, 0.001]
ax2.plot(sg, o1, "o-", color=AMBER, lw=2.5, ms=8)
ax2.plot(sg, o3, "s-", color=DEEP, lw=2.5, ms=8)
ax2.plot(sg[:4], o5, "D-", color=GREEN, lw=2.5, ms=8)
ax2.text(3.2, 0.9, "order 1", color=AMBER, fontsize=14)
ax2.text(3.2, 0.045, "order 3\n(default)", color=DEEP, fontsize=14)
ax2.text(1.1, 0.012, "order 5", color=GREEN, fontsize=14)
ax2.set_yscale("log"); ax2.set_xticks([1, 2, 3, 4, 6]); ax2.set_ylim(5e-4, 8)
ax2.set_xlabel("front width  sigma_G  [cells]", fontsize=16)
ax2.set_ylabel("bias  [% of G per hour]", fontsize=16)
ax2.set_title("V4 vs front width", loc="left", fontweight="bold", fontsize=17)
fig.tight_layout(w_pad=2.0)
save(fig, "m1_fig_gates.png")

# ---------------------------------------------------------------- task 6: V3, every change tried
# Task-6 table (OLS on the pre-declared front pixels), LLC variant unless marked.
fig, ax = plt.subplots(figsize=(7.2, 4.0))
fig.subplots_adjust(left=0.40, right=0.97, top=0.88, bottom=0.22)
rows = [
    ("chain F, bilinear u  (1st try)", 0.7579, RED),
    ("chain F, cubic u", 0.7914, RED),
    ("discrete o2, bilinear u", 0.9824, MUTED),
    ("discrete o4, bilinear u", 0.9382, MUTED),
    ("discrete o2, cubic u", 1.0267, MUTED),
    ("discrete o4, cubic u  (adopted)", 0.9806, GREEN),
]
y = np.arange(len(rows))[::-1]
ax.axvspan(0.95, 1.05, color="#DCE8F2", zorder=0)
ax.axvline(1.0, color=MIDNIGHT, lw=1.2)
for yi, (lab, v, c) in zip(y, rows):
    ax.barh(yi, v - 0.7, left=0.7, height=0.62, color=c)
    ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=16, color=MIDNIGHT)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=15)
ax.tick_params(axis="x", labelsize=15)
ax.set_xlim(0.7, 1.12); ax.set_xlabel("OLS slope on front pixels  (LLC)", fontsize=16)
ax.set_title("Every change tried; gate 1 ± 0.05", loc="left", fontweight="bold",
             fontsize=17)
save(fig, "m1_fig_v3_changes.png")

# ---------------------------------------------------------------- task 6b: V3b slopes per truth
# Task-6b table, LLC variant, slope [2.5%, 97.5%] by 32-cell block bootstrap.
fig, ax = plt.subplots(figsize=(7.0, 4.6))
fig.subplots_adjust(left=0.36, right=0.97, top=0.90, bottom=0.36)
truths = ["semi-Lagrangian (V3)", "FV centred (stencil)", "FV OS7 (7th order)",
          "FV OS7MP-like", "FV DST3 (3rd order)"]
disc = [(0.981, 0.970, 0.994), (0.974, 0.930, 1.026), (0.983, 0.958, 1.018),
        (0.975, 0.954, 1.003), (0.845, 0.784, 0.918)]
chain_ = [(0.791, 0.753, 0.815), (0.811, 0.768, 0.849), (0.803, 0.778, 0.823),
          (0.790, 0.753, 0.814), (0.666, 0.622, 0.725)]
y = np.arange(len(truths))[::-1]
ax.axvspan(0.954, 1.003, color="#DCE8F2", zorder=0, label="recorded bias band 0.954-1.003")
ax.axvline(0.981, color=MIDNIGHT, lw=1.2, ls="--", label="V3 baseline 0.981")
ax.axvline(1 / 0.85, color=RED, lw=1.5, ls=":", label="1/0.85: if the attenuation acted")
for yi, (m, lo, hi) in zip(y, disc):
    ax.errorbar(m, yi + 0.15, xerr=[[m - lo], [hi - m]], fmt="o", color=DEEP, ms=9,
                capsize=4, lw=2)
for yi, (m, lo, hi) in zip(y, chain_):
    ax.errorbar(m, yi - 0.15, xerr=[[m - lo], [hi - m]], fmt="s", color=AMBER, ms=8,
                capsize=4, lw=2, mfc="white")
ax.plot([], [], "o", color=DEEP, ms=9, label="form = discrete")
ax.plot([], [], "s", color=AMBER, ms=8, mfc="white", label="form = chain")
ax.set_yticks(y); ax.set_yticklabels(truths, fontsize=15)
ax.tick_params(axis="x", labelsize=15)
ax.set_xlim(0.6, 1.25); ax.set_xlabel("OLS slope on front pixels  (LLC, hour 0)", fontsize=16)
ax.legend(frameon=False, fontsize=13, loc="upper center", bbox_to_anchor=(0.35, -0.28),
          ncol=2, columnspacing=1.0, handletextpad=0.5)
ax.set_title("V3b: slope per advection truth", loc="left", fontweight="bold", fontsize=17)
save(fig, "m1_fig_v3b.png")

print("wrote:")
for f in sorted(OUT.glob("*.png")):
    im = Image.open(f)
    print(f"  {f.relative_to(OUT.parent)}  {im.size[0]}x{im.size[1]}  ({f.stat().st_size/1024:.0f} KB)")
