"""Figures for the M0 acceptance deck.

Every number here is quoted from the M0 task logs in
``claude_prompts/frontogenesis_prompts.md`` (entries of 2026-09-27/28); nothing is
recomputed and no network or data store is touched.  The point of these panels is to
make three M0 findings legible at a glance:

  m0_fig1  implicit numerical diffusion vs the kinematic frontogenesis rate, by scale
  m0_fig2  hourly displacement -- why the scheme is semi-Lagrangian
  m0_fig3  the two overturned planning claims, side by side

Run:  <frontogenesis env python> make_m0_figs.py
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).parent / "figs_m0"
OUT.mkdir(exist_ok=True)

MIDNIGHT, DEEP, TEAL = "#21295C", "#065A82", "#1C7293"
AMBER, MUTED, LIGHT = "#A8501B", "#62737F", "#F2F6F8"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": MIDNIGHT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": MIDNIGHT,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})

# ---------------------------------------------------------------- fig 1
# Q5: OS7MP implicit diffusivity kappa/(|u| dx) by scale, and the resulting
# damping rate of G (= 2 kappa k^2) at the tile's median / p90 / p99 speeds.
# Kinematic reference: 2F/G ~ 2|sigma|, with |sigma| median 1.9e-5 s^-1 (Q4).
scales = ["2dx\n3.6 km", "4dx\n7.2 km", "10 km\n(front)", "20 km"]
rate_med = np.array([2.0e-4, 1.2e-5, 1.3e-6, 1.0e-8])   # s^-1, from e-folding times
rate_p99 = np.array([7.0e-4, 4.1e-5, 4.4e-6, 3.0e-8])
kinematic = 2 * 1.9e-5                                   # 2|sigma| median

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.9), width_ratios=[1.25, 1])
x = np.arange(len(scales))
ax.bar(x - 0.19, rate_med, 0.36, color=DEEP, label="median speed (0.19 m/s)")
ax.bar(x + 0.19, rate_p99, 0.36, color=TEAL, label="p99 speed (0.64 m/s)")
ax.axhline(kinematic, color=AMBER, lw=2.0, ls="--",
           label=r"kinematic $2|\sigma|$ = 3.8e-5 s$^{-1}$")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(scales)
ax.set_ylabel(r"damping rate of $G$   (s$^{-1}$)")
ax.set_title("Implicit numerical diffusion (OS7MP) vs the signal",
             fontsize=12, fontweight="bold", loc="left", color=MIDNIGHT)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.annotate("numerics win", xy=(0, rate_med[0]), xytext=(0, 2.4e-3),
            ha="center", fontsize=9, color=AMBER, fontweight="bold")
ax.annotate("3-10% of signal", xy=(2, rate_p99[2]), xytext=(2.15, 2.0e-5),
            ha="center", fontsize=9, color=DEEP)

surv = np.array([0, 4, 71, 100])       # % of G surviving 72 h, median speed
ax2.bar(np.arange(4), surv, 0.55, color=[AMBER, AMBER, DEEP, TEAL])
ax2.set_xticks(np.arange(4)); ax2.set_xticklabels(scales)
ax2.set_ylabel("% of $G$ surviving 72 h"); ax2.set_ylim(0, 108)
ax2.set_title("At the median speed", fontsize=12, fontweight="bold",
              loc="left", color=MIDNIGHT)
for xi, v in zip(np.arange(4), surv):
    ax2.text(xi, v + 3, f"{v}%", ha="center", fontsize=10, fontweight="bold",
             color=MIDNIGHT)
fig.text(0.008, -0.02,
         "Where the MP limiter engages (1-2 cell fronts) the local diffusivity rises toward "
         r"$|u|\,dx/2$ — at ~1.5-cell fronts the numerics are the whole story.",
         fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(OUT / "m0_fig1_numerical_diffusion.png", bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- fig 2
# Q3: hourly displacement in cells, ocean vs front pixels.
fig, ax = plt.subplots(figsize=(6.6, 3.9))
labels = ["median", "p90", "p99", "max"]
ocean = [0.37, 0.75, 1.28, 3.5]
front = [0.54, 1.10, 1.64, 2.5]
x = np.arange(len(labels))
ax.bar(x - 0.19, ocean, 0.36, color=DEEP, label="all ocean")
ax.bar(x + 0.19, front, 0.36, color=TEAL, label=r"front pixels ($G>p_{90}$)")
ax.axhline(1.0, color=AMBER, ls="--", lw=1.6)
ax.text(3.45, 1.06, "1 cell", color=AMBER, fontsize=9, ha="right", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("hourly displacement  (grid cells)")
ax.set_title("Why the scheme is semi-Lagrangian", fontsize=12, fontweight="bold",
             loc="left", color=MIDNIGHT)
ax.legend(frameon=False, fontsize=9)
ax.text(0.02, 0.86,
        "Planning §5.3 predicted 0.2-0.4 typical and >1.5 in the strong-front tail.\n"
        "Measured: 0.37 median, 1.64 at the front p99.  Both confirmed.",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="top")
fig.tight_layout()
fig.savefig(OUT / "m0_fig2_displacement.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- fig 3
# The two overturned claims, as a before/after card pair.
fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4))
cards = [
    ("Land fill value",
     "PLANNING §5.5 SAID",
     "MITgcm stores land as 0, so b(0,0) is\nfinite and poisons every ocean cell\nadjacent to coast.",
     "M0 MEASURED",
     "Land is NaN — cell-for-cell equal to\nhFacC / hFacW / hFacS, 0 mismatches\nin 518,400 cells.",
     "No coastal ribbon exists. The stencil\npart of the halo is automatic."),
    ("Surface vertical velocity",
     "PLANNING §2.2 SAID",
     "w vanishes at z = 0, so the tilting\nterm drops out identically.",
     "M0 MEASURED",
     "W(k_l=0) = dEta/dt.  corr 0.9936,\nslope 1.037 against a centred ±1 h\nfree-surface rate.",
     "Conclusion survives, premise does not:\nit follows from w = D(eta)/Dt, not w = 0."),
]
for ax, (title, h1, t1, h2, t2, verdict) in zip(axes, cards):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=LIGHT, edgecolor="none"))
    ax.text(0.05, 0.90, title, fontsize=12, fontweight="bold", color=MIDNIGHT,
            transform=ax.transAxes)
    ax.text(0.05, 0.775, h1, fontsize=8.5, fontweight="bold", color=MUTED,
            transform=ax.transAxes)
    ax.text(0.05, 0.60, t1, fontsize=9.5, color=MIDNIGHT, transform=ax.transAxes,
            va="top", linespacing=1.5)
    ax.text(0.05, 0.435, h2, fontsize=8.5, fontweight="bold", color=AMBER,
            transform=ax.transAxes)
    ax.text(0.05, 0.26, t2, fontsize=9.5, color=MIDNIGHT, transform=ax.transAxes,
            va="top", linespacing=1.5)
    ax.text(0.05, 0.115, verdict, fontsize=9, color=DEEP, style="italic",
            transform=ax.transAxes, va="top", linespacing=1.4)
fig.tight_layout()
fig.savefig(OUT / "m0_fig3_overturned.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- downscale the QA plot
from PIL import Image
src = pathlib.Path(__file__).parents[1] / "figs" / "m0_qa_tile330_20120702T00.png"
if src.exists():
    im = Image.open(src)
    im.thumbnail((1700, 1700), Image.LANCZOS)
    im.save(OUT / "m0_qa_small.png", optimize=True)

print("wrote:")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.relative_to(OUT.parent)}  ({f.stat().st_size/1024:.0f} KB)")
