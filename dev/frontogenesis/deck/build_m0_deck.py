"""Build the M0 acceptance deck.

Content is quoted from the M0 task logs in
``claude_prompts/frontogenesis_prompts.md`` (entries of 2026-09-27/28).  Figures come
from ``make_m0_figs.py`` (run it first) and from ``../figs/``.

Run:  <frontogenesis env python> make_m0_figs.py && <...> build_m0_deck.py
Deps: python-pptx (pip install python-pptx into the frontogenesis env).
"""
import pathlib
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = pathlib.Path(__file__).parent
FIGS = HERE / "figs_m0"

MIDNIGHT, DEEP, TEAL = C(0x21,0x29,0x5C), C(0x06,0x5A,0x82), C(0x1C,0x72,0x93)
MINT, WHITE, INK = C(0x8F,0xBF,0xD0), C(0xFF,0xFF,0xFF), C(0x1B,0x2A,0x33)
MUTED, LIGHT, CARD = C(0x62,0x73,0x7F), C(0xF2,0xF6,0xF8), C(0xE7,0xEF,0xF3)
AMBER, AMBERBG = C(0xA8,0x50,0x1B), C(0xFA,0xEF,0xE4)
GREEN = C(0x2C,0x6E,0x49)
HEAD, BODY, MONO = "Cambria", "Calibri", "Courier New"

prs = Presentation(); prs.slide_width, prs.slide_height = I(13.333), I(7.5)
BLANK = prs.slide_layouts[6]


def slide(bg=LIGHT):
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    r.fill.solid(); r.fill.fore_color.rgb = bg
    r.line.fill.background(); r.shadow.inherit = False
    return s


def txt(s, x, y, w, h, runs, size=14, color=INK, font=BODY, bold=False,
        align=PP_ALIGN.LEFT, space=5, line=None, italic=False):
    tb = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(runs, str):
        runs = [(runs, {})]
    for i, item in enumerate(runs):
        t, o = item if isinstance(item, tuple) else (item, {})
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = o.get("align", align)
        p.space_after = Pt(o.get("space", space))
        if line or o.get("line"):
            p.line_spacing = o.get("line", line)
        if o.get("bullet"):
            t = "•   " + t
        r = p.add_run(); r.text = t; f = r.font
        f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
        f.name = o.get("font", font); f.italic = o.get("italic", italic)
        f.color.rgb = o.get("color", color)
    return tb


def card(s, x, y, w, h, fill=CARD):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(x), I(y), I(w), I(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.fill.background(); sh.adjustments[0] = 0.07; sh.shadow.inherit = False
    return sh


def circle(s, x, y, d, label, fill=DEEP, fg=WHITE, size=13):
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL, I(x), I(y), I(d), I(d))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.fill.background(); sh.shadow.inherit = False
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = label
    r.font.size = Pt(size); r.font.bold = True; r.font.name = BODY
    r.font.color.rgb = fg
    return sh


def title(s, text, sub=None, dark=False, tag=None):
    y = 0.50
    if tag:
        txt(s, 0.85, y, 11.7, 0.3, tag, size=10.5, bold=True,
            color=MINT if dark else TEAL, space=0)
        y += 0.34
    txt(s, 0.85, y, 11.7, 0.75, text, size=30, bold=True, font=HEAD,
        color=WHITE if dark else MIDNIGHT, space=0)
    if sub:
        txt(s, 0.85, y + 0.78, 11.7, 0.42, sub, size=13,
            color=MINT if dark else MUTED, space=0)


def pic(s, name, x, y, w):
    p = FIGS / name
    if p.exists():
        s.shapes.add_picture(str(p), I(x), I(y), width=I(w))
    return p.exists()


# ------------------------------------------------------------------ 1 TITLE
s = slide(MIDNIGHT)
for d, cx, cy, col in ((4.4, 10.2, 3.1, DEEP), (3.0, 11.2, 4.6, TEAL), (1.8, 10.4, 5.7, MINT)):
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, I(cx - d / 2), I(cy - d / 2), I(d), I(d))
    o.fill.background(); o.line.color.rgb = col; o.line.width = Pt(1.4)
    o.shadow.inherit = False
txt(s, 0.95, 1.7, 8.4, 0.35, "MILESTONE M0  ·  ACCESS AND RECONNAISSANCE",
    size=11.5, color=MINT, bold=True, space=0)
txt(s, 0.95, 2.3, 8.6, 1.9,
    [("Frontogenesis in the", {}), ("LLC4320 California Current", {})],
    size=36, bold=True, font=HEAD, color=WHITE, space=4, line=1.05)
txt(s, 0.95, 4.35, 8.2, 0.9,
    "Prove we can read the data, and settle every open empirical question "
    "before any physics is written.",
    size=14, color=MINT, line=1.3)
card(s, 0.95, 5.35, 8.2, 0.85, DEEP)
txt(s, 1.3, 5.6, 7.6, 0.4,
    "M0 complete — all acceptance criteria PASS, two planning claims overturned",
    size=13.5, bold=True, color=WHITE, space=0)
txt(s, 0.95, 6.6, 8.2, 0.5,
    "J. Xavier Prochaska  ·  tile 330, face 10  ·  2026-09-27/29",
    size=11.5, color=MUTED, space=0)

# ------------------------------------------------------------------ 2 CONTENTS
s = slide()
title(s, "Contents", "Six tasks, three findings worth their own slide")
rows = [
    ("1", "Environment", "A dedicated py3.13 env; dbof from a worktree", DEEP),
    ("2", "First contact with the data", "osn_tiles.py; one hour from both OSN stores", DEEP),
    ("3", "Five empirical questions", "Land fill, surface W, spacing, rotation, advection scheme", DEEP),
    ("⚑", "What M0 overturned", "Two planning claims contradicted by measurement", AMBER),
    ("⚑", "Numerical diffusion, quantified", "The leading rival explanation for any slope < 1", AMBER),
    ("⚑", "Hourly displacement", "Confirms the semi-Lagrangian choice", TEAL),
    ("4", "tile330_grid.zarr and two hours", "The static grid store and a two-hour raw product", DEEP),
    ("5", "QA plot", "Stencil rims, ribbon test, tile-edge crop test", DEEP),
    ("6", "This deck", "M0 acceptance summary", DEEP),
]
for i, (n, h, sub, col) in enumerate(rows):
    y = 1.85 + i * 0.58
    circle(s, 0.9, y + 0.02, 0.38, n, col, size=12)
    txt(s, 1.5, y, 5.3, 0.32, h, size=14, bold=True, color=MIDNIGHT, space=0)
    txt(s, 6.9, y + 0.03, 5.5, 0.32, sub, size=11.5, color=MUTED, space=0)

# ------------------------------------------------------------------ 3 TASK 1
s = slide()
title(s, "Environment", "A dedicated env, and one pin in the brief that was wrong", tag="TASK 1")
card(s, 0.85, 2.0, 5.7, 2.5)
txt(s, 1.2, 2.3, 5.1, 0.3, "ENV `frontogenesis`", size=11, bold=True, color=DEEP, space=0)
txt(s, 1.2, 2.75, 5.1, 1.6,
    [("Python 3.13.15  ·  xarray 2026.7.0  ·  dask 2026.8.0", {}),
     ("zarr 3.4.0  ·  xgcm 0.10.1  ·  scikit-fmm 2025.6.23", {}),
     ("s3fs 2026.9.0  ·  torch 2.13.0 (untouched)", {})],
    size=12, color=INK, space=5, line=1.25)
txt(s, 1.2, 4.05, 5.1, 0.3,
    "dbof installed --no-deps from a git worktree; fronts checkout undisturbed.",
    size=10.5, color=MUTED, space=0)
card(s, 6.75, 2.0, 5.7, 2.5, AMBERBG)
circle(s, 7.1, 2.28, 0.42, "!", AMBER, size=15)
txt(s, 7.75, 2.38, 4.4, 0.3, "THE BRIEF'S `xgcm<0.10` PIN WAS WRONG",
    size=11, bold=True, color=AMBER, space=0)
txt(s, 7.1, 2.95, 5.0, 1.5,
    "`set_xgcm_grid` passes `padding='fill'`, which exists only in xgcm >= 0.10 — and "
    "dbof's own pyproject requires it. The reason given for the pin was true of xgcm but "
    "backwards for dbof.",
    size=12, color=INK, line=1.28)
txt(s, 7.1, 4.05, 5.0, 0.3, "Upgraded to 0.10.1; brief corrected 2026-09-28.",
    size=10.5, bold=True, color=AMBER, space=0)
card(s, 0.85, 4.8, 11.6, 1.5, MIDNIGHT)
txt(s, 1.25, 5.1, 10.8, 0.95,
    [("Python 3.14 was checked and rejected.", {"bold": True, "color": WHITE}),
     ("conda-forge has scikit-fmm for 3.14, but PyPI has no wheel, xgcm 0.9 depends on the "
      "unmaintained `future`, and `fronts` drags timm 0.3.2, PyQt6, pyvista and healpy. "
      "Not worth debugging for this project.", {"color": MINT, "size": 12})],
    size=13, space=4, line=1.25)

# ------------------------------------------------------------------ 4 TASK 2
s = slide()
title(s, "First contact with the data", "One hour, both OSN stores, anonymously", tag="TASK 2")
stats = [("45 s", "grid + two stores"), ("24.9 MB", "grid tile in memory"),
         ("0.6884", "ocean fraction"), ("1022976", "OSN iteration at t0")]
for i, (big, lab) in enumerate(stats):
    x = 0.85 + i * 2.95
    card(s, x, 1.95, 2.7, 1.25, CARD)
    txt(s, x + 0.15, 2.15, 2.4, 0.5, big, size=21, bold=True, font=HEAD,
        color=DEEP, align=PP_ALIGN.CENTER, space=0)
    txt(s, x + 0.15, 2.72, 2.4, 0.3, lab, size=10.5, color=MUTED,
        align=PP_ALIGN.CENTER, space=0)
txt(s, 0.85, 3.45, 5.7, 0.3, "WRITTEN", size=11, bold=True, color=DEEP, space=0)
txt(s, 0.85, 3.85, 5.7, 2.3,
    [("`py/osn_tiles.py` — tile_spec, load_grid, build_xgcm, load_hour, load_wind_hour.", {"bullet": True}),
     ("Every pull checks the store's own decoded clock against the requested timestamp and raises on mismatch.", {"bullet": True}),
     ("Box lon −127.99..−113.01, lat 26.66..38.27 (planning said 38.20).", {"bullet": True})],
    size=12, color=INK, space=8, line=1.25)
txt(s, 6.75, 3.45, 5.7, 0.3, "TRAPS CONFIRMED", size=11, bold=True, color=DEEP, space=0)
txt(s, 6.75, 3.85, 5.7, 2.3,
    [("Comodo attrs survive `process_llc4320_grid` on real OSN data — the §2.1 trap does not fire here (index coords keep their attrs).", {"bullet": True}),
     ("Both OSN stores are keyed by the same iteration; the timestamp round trip is exact.", {"bullet": True}),
     ("`drF = 1.0`, `Z = −0.5` are in the OSN gridfile after all — the brief said they were not.", {"bullet": True})],
    size=12, color=INK, space=8, line=1.25)

# ------------------------------------------------------------------ 5 TASK 3
s = slide()
title(s, "Five empirical questions", "Each was an assumption; each is now a number", tag="TASK 3")
qs = [
    ("Q1", "Land fill value", "NaN, not 0 — cell-for-cell equal to hFacC / hFacW / hFacS, 0 mismatches in 518,400 cells.", AMBER),
    ("Q2", "Surface W", "W(k_l=0) = dEta/dt, not ~0. corr 0.9936, slope 1.037.", AMBER),
    ("Q3", "Grid spacing", "1.71 × 1.85 km at 37N (not 1.8-2.3). Face 10 is rotated 90°: i is meridional, j zonal.", TEAL),
    ("Q4", "Rotation terms", "Identically zero — SN = −1, CS = 0 everywhere. Metric term 0.1% median.", DEEP),
    ("Q5", "Advection scheme", "OS7MP (tempAdvScheme = 7), no explicit horizontal diffusion, linear free surface.", DEEP),
]
for i, (n, h, body, col) in enumerate(qs):
    y = 1.95 + i * 0.95
    circle(s, 0.9, y + 0.08, 0.44, n, col, size=12)
    txt(s, 1.55, y, 2.6, 0.3, h, size=13.5, bold=True, color=MIDNIGHT, space=0)
    txt(s, 4.3, y + 0.02, 8.1, 0.75, body, size=12, color=INK, space=0, line=1.25)
txt(s, 0.85, 6.75, 11.6, 0.35,
    "Q1 and Q2 contradict the planning doc. Q3 and Q4 confirm it, Q4 more strongly than claimed.",
    size=11.5, italic=True, color=MUTED, align=PP_ALIGN.CENTER, space=0)

# ------------------------------------------------------------------ 6 OVERTURNED
s = slide()
title(s, "What M0 overturned", "The most valuable output of a reconnaissance milestone",
      tag="FINDING")
pic(s, "m0_fig3_overturned.png", 0.85, 2.05, 11.6)
card(s, 0.85, 5.55, 11.6, 1.3, MIDNIGHT)
txt(s, 1.25, 5.85, 10.8, 0.75,
    [("Neither breaks the study.", {"bold": True, "color": WHITE}),
     ("Land being NaN makes the halo cheaper, not harder. And the surface budget still follows "
      "— from the kinematic condition w = D(eta)/Dt rather than from w = 0, so the conclusion "
      "survives even though the sentence does not.", {"color": MINT, "size": 12})],
    size=13, space=4, line=1.25)

# ------------------------------------------------------------------ 7 NUMERICS
s = slide()
title(s, "Numerical diffusion, quantified", "The leading rival explanation for any slope below 1",
      tag="FINDING")
pic(s, "m0_fig1_numerical_diffusion.png", 0.85, 1.95, 11.6)
txt(s, 0.85, 6.35, 11.6, 0.75,
    [("Planning §2.3 guessed κ ~ 6-60 m²/s and 0.1-1 f. That is the 4dx number, and it holds there "
      "— but it does not transfer to 10 km, where the same scheme is an order of magnitude gentler.",
      {"color": INK}),
     ("Consequence: the discriminator in Figure 2b is needed at front scales, not merely prudent.",
      {"color": AMBER, "bold": True})],
    size=11.5, space=3, line=1.3)

# ------------------------------------------------------------------ 8 DISPLACEMENT
s = slide()
title(s, "Hourly displacement", "Why the measured side is semi-Lagrangian", tag="FINDING")
pic(s, "m0_fig2_displacement.png", 0.85, 2.0, 6.5)
card(s, 7.7, 2.0, 4.75, 3.9, CARD)
txt(s, 8.05, 2.35, 4.1, 0.3, "WHAT THIS SETTLES", size=11, bold=True, color=DEEP, space=0)
txt(s, 8.05, 2.85, 4.1, 2.8,
    [("Planning §5.3 predicted 0.2-0.4 cells typically and >1.5 in the strong-front tail. Measured: 0.37 median, 1.64 at the front p99. Both confirmed.", {"bullet": True}),
     ("30% of ocean cells exceed half a cell; 3.4% exceed one. An Eulerian split would cancel two large terms exactly where the signal lives.", {"bullet": True}),
     ("`follow(km_per_px=2.3)` is 25% too large for this tile — should be ~1.8 (M4).", {"bullet": True})],
    size=11.5, color=INK, space=9, line=1.25)
txt(s, 0.85, 6.15, 6.5, 0.4,
    "Speeds at cell centres: median 0.19, p90 0.38, p99 0.64, max 1.74 m/s.",
    size=11, color=MUTED, space=0)

# ------------------------------------------------------------------ 9 TASK 4
s = slide()
title(s, "tile330_grid.zarr and two hours", "The static store everything downstream reuses",
      tag="TASK 4")
card(s, 0.85, 1.95, 5.7, 2.15, CARD)
txt(s, 1.2, 2.25, 5.1, 0.3, "tile330_grid.zarr", size=12.5, bold=True, font=MONO,
    color=DEEP, space=0)
txt(s, 1.2, 2.7, 5.1, 1.3,
    "1.8 MB on disk (29.1 MB in memory). Twelve §3.1 variables plus hFacW, hFacS and the "
    "0-d drF, Z, Zl. Orientation and measured spacing stored as attrs.",
    size=12, color=INK, line=1.28)
card(s, 6.75, 1.95, 5.7, 2.15, CARD)
txt(s, 7.1, 2.25, 5.1, 0.3, "tile330_raw_20120702T00_2h.zarr", size=12.5, bold=True,
    font=MONO, color=DEEP, space=0)
txt(s, 7.1, 2.7, 5.1, 1.3,
    "21 MB on disk. Two consecutive hours from both OSN stores — M1's gate 3 needs a "
    "midpoint velocity, which one hour cannot give.",
    size=12, color=INK, line=1.28)
txt(s, 0.85, 4.35, 11.6, 0.3, "DECISIONS WORTH KNOWING", size=11, bold=True,
    color=DEEP, space=0)
txt(s, 0.85, 4.8, 11.6, 1.9,
    [("The face dim is dropped on disk and kept as a scalar coord, so `expand_dims('face')` restores the layout the dbof operators expect.", {"bullet": True}),
     ("One (1, 720, 720) chunk per hour per variable — the append unit for M2's `pull_series`.", {"bullet": True}),
     ("`write_grid` asserts SN = −1 and |CS| < 1e-6 before writing the orientation attr, so the string cannot outlive a change of tile.", {"bullet": True}),
     ("`data/.gitignore` added: the repo ignores *.nc and *.png but not *.zarr.", {"bullet": True})],
    size=12, color=INK, space=7, line=1.22)

# ------------------------------------------------------------------ 10 TASK 5
s = slide()
title(s, "QA plot", "Stencil rims, the ribbon test, and the tile-edge crop test", tag="TASK 5")
pic(s, "m0_qa_small.png", 0.85, 1.9, 8.1)
card(s, 9.25, 1.9, 3.2, 4.6, CARD)
txt(s, 9.55, 2.2, 2.6, 0.3, "WHAT IT SHOWS", size=11, bold=True, color=DEEP, space=0)
txt(s, 9.55, 2.65, 2.65, 3.6,
    [("G's NaN rim is exactly 1 cell along the coast; the Jacobian's is 2 — the brief's \"~3\" was one cell conservative.", {"bullet": True}),
     ("No gradient ribbon. Median G rises smoothly over ~10 cells: that is the coastal upwelling front, not an artefact.", {"bullet": True}),
     ("A b(0,0) ribbon would be ~1e7× the interior — and the tile's own j = 0 edge, where xgcm pads with 0, shows exactly that.", {"bullet": True}),
     ("All four tile edges are invalid and finite, not NaN.", {"bullet": True})],
    size=10.5, color=INK, space=7, line=1.2)
txt(s, 0.85, 6.6, 8.1, 0.35,
    "figs/m0_qa_tile330_20120702T00.png  ·  py/m0_qa_plot.py, py/m0_qa_checks.py",
    size=10.5, font=MONO, color=MUTED, space=0)

# ------------------------------------------------------------------ 11 TASK 6
s = slide()
title(s, "This deck", "Reproducible from the logged numbers alone", tag="TASK 6")
card(s, 0.85, 2.0, 11.6, 1.6, CARD)
txt(s, 1.25, 2.3, 10.8, 1.1,
    [("Every number in these slides is quoted from the M0 task logs in `frontogenesis_prompts.md`.", {"bold": True}),
     ("Nothing is recomputed, and no network or data store is touched — so the deck can be "
      "rebuilt anywhere, and it cannot silently drift from the log it summarises.", {"color": MUTED, "size": 12})],
    size=13, color=INK, space=4, line=1.25)
files = [
    ("make_m0_figs.py", "generates the three deck figures and downscales the QA plot"),
    ("build_m0_deck.py", "builds this .pptx"),
    ("figs_m0/", "the four images used here"),
    ("README.md", "the work log for this task"),
]
txt(s, 0.85, 3.9, 11.6, 0.3, "IN `dev/frontogenesis/deck/`", size=11, bold=True,
    color=DEEP, space=0)
for i, (f, d) in enumerate(files):
    y = 4.35 + i * 0.52
    txt(s, 1.1, y, 3.4, 0.3, f, size=12, font=MONO, bold=True, color=MIDNIGHT, space=0)
    txt(s, 4.8, y + 0.02, 7.6, 0.3, d, size=11.5, color=MUTED, space=0)
txt(s, 0.85, 6.6, 11.6, 0.35,
    "Requires python-pptx in the frontogenesis env; figures need only matplotlib and PIL.",
    size=10.5, color=MUTED, space=0)

# ------------------------------------------------------------------ 12 ACCEPTANCE
s = slide(MIDNIGHT)
title(s, "M0 acceptance", "Every criterion audited in the task-5 log entry", dark=True)
crit = [
    "Two consecutive hours load end to end, from both OSN stores, in a reproducible env",
    "tile330_grid.zarr written, with hFacW/hFacS and the 0-d drF/Z/Zl",
    "All five questions answered with numbers",
    "All four §2 traps confirmed against real data",
    "QA plot written; stencil rim present, gradient ribbon absent",
]
for i, c in enumerate(crit):
    y = 1.95 + i * 0.62
    circle(s, 0.9, y + 0.02, 0.36, "✓", GREEN, size=13)
    txt(s, 1.5, y, 10.9, 0.4, c, size=13, color=WHITE, space=0, line=1.2)
card(s, 0.85, 5.3, 11.6, 1.55, DEEP)
txt(s, 1.25, 5.6, 10.8, 1.0,
    [("Next: M1 — operators and validation. A hard gate.", {"bold": True, "color": WHITE, "size": 15}),
     ("Its discrete null test must return slope = 1 ± 0.05 before any real data is touched; "
      "C-grid interpolation attenuation alone can bias the headline slope 0.7-1.4.",
      {"color": MINT, "size": 12})],
    space=4, line=1.25)

out = HERE / "Frontogenesis_M0_Acceptance.pptx"
prs.save(out)
print(f"saved {out.name}  ({out.stat().st_size/1024:.0f} KB, {len(prs.slides._sldIdLst)} slides)")
