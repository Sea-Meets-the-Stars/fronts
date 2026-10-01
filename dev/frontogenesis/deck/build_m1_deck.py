"""Build the M1 acceptance deck.

Content is quoted from the M1 task logs in ``claude_prompts/frontogenesis_prompts.md``
("Execution prompt 2", tasks 1-7, 6b, 7a; 2026-09-28 .. 2026-09-30) and from the Status,
criteria and Q&A of ``claude_prompts/frontogenesis_prompt_2.md``.  Figures come from
``make_m1_figs.py`` (run it first).  Nothing is recomputed.

House rule for this deck: **no text smaller than 20 pt** -- every run, including tags,
captions and footers.  ``check_m1_deck.py`` verifies that after the build.

Style (palette, helpers) is copied from ``build_m0_deck.py`` rather than imported, since
importing it would run the M0 build.

Run:  <frontogenesis env python> make_m1_figs.py && <...> build_m1_deck.py
Deps: python-pptx (already in the frontogenesis env since M0).
"""
import pathlib
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = pathlib.Path(__file__).parent
FIGS = HERE / "figs_m1"

MIDNIGHT, DEEP, TEAL = C(0x21,0x29,0x5C), C(0x06,0x5A,0x82), C(0x1C,0x72,0x93)
MINT, WHITE, INK = C(0x8F,0xBF,0xD0), C(0xFF,0xFF,0xFF), C(0x1B,0x2A,0x33)
MUTED, LIGHT, CARD = C(0x62,0x73,0x7F), C(0xF2,0xF6,0xF8), C(0xE7,0xEF,0xF3)
AMBER, AMBERBG = C(0xA8,0x50,0x1B), C(0xFA,0xEF,0xE4)
GREEN = C(0x2C,0x6E,0x49)
HEAD, BODY, MONO = "Cambria", "Calibri", "Courier New"
MIN_PT = 20          # the house rule

prs = Presentation(); prs.slide_width, prs.slide_height = I(13.333), I(7.5)
BLANK = prs.slide_layouts[6]


def slide(bg=LIGHT):
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    r.fill.solid(); r.fill.fore_color.rgb = bg
    r.line.fill.background(); r.shadow.inherit = False
    return s


def _run(p, t, size, bold, font, italic, color):
    r = p.add_run(); r.text = t; f = r.font
    f.size = Pt(max(size, MIN_PT)); f.bold = bold; f.name = font
    f.italic = italic; f.color.rgb = color
    return r


def txt(s, x, y, w, h, runs, size=20, color=INK, font=BODY, bold=False,
        align=PP_ALIGN.LEFT, space=4, line=1.1, italic=False):
    """Text box.  ``runs`` is a str, or a list of paragraphs; a paragraph is a str or a
    (text, opts) tuple, and ``text`` may itself be a list of (str, opts) runs."""
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
        p.line_spacing = o.get("line", line)
        pieces = t if isinstance(t, list) else [(t, {})]
        for k, (pt, po) in enumerate(pieces):
            if k == 0 and o.get("bullet"):
                pt = "•  " + pt
            oo = {**o, **po}
            _run(p, pt, oo.get("size", size), oo.get("bold", bold), oo.get("font", font),
                 oo.get("italic", italic), oo.get("color", color))
    return tb


def card(s, x, y, w, h, fill=CARD):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(x), I(y), I(w), I(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.fill.background(); sh.adjustments[0] = 0.07; sh.shadow.inherit = False
    return sh


def circle(s, x, y, d, label, fill=DEEP, fg=WHITE, size=20):
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL, I(x), I(y), I(d), I(d))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.fill.background(); sh.shadow.inherit = False
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    _run(p, label, size, True, BODY, False, fg)
    return sh


def title(s, text, sub=None, dark=False, tag=None):
    y = 0.38
    if tag:
        txt(s, 0.85, y, 11.7, 0.36, tag, size=20, bold=True,
            color=MINT if dark else TEAL, space=0)
        y += 0.36
    txt(s, 0.85, y, 11.7, 0.65, text, size=30, bold=True, font=HEAD,
        color=WHITE if dark else MIDNIGHT, space=0)
    if sub:
        txt(s, 0.85, y + 0.66, 11.7, 0.42, sub, size=20,
            color=MINT if dark else MUTED, space=0)


def pic(s, name, x, y, w=None, h=None):
    p = FIGS / name
    if p.exists():
        kw = {"width": I(w)} if w else {"height": I(h)}
        s.shapes.add_picture(str(p), I(x), I(y), **kw)
    return p.exists()


def bullets(s, x, y, w, h, items, size=20, space=8, color=INK):
    return txt(s, x, y, w, h, [(t, {"bullet": True}) for t in items],
               size=size, color=color, space=space)


# ------------------------------------------------------------------ 1 TITLE
s = slide(MIDNIGHT)
for d, cx, cy, col in ((4.4, 10.2, 3.1, DEEP), (3.0, 11.2, 4.6, TEAL), (1.8, 10.4, 5.7, MINT)):
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, I(cx - d / 2), I(cy - d / 2), I(d), I(d))
    o.fill.background(); o.line.color.rgb = col; o.line.width = Pt(1.4)
    o.shadow.inherit = False
txt(s, 0.95, 1.3, 8.4, 0.4, "MILESTONE M1  ·  OPERATORS AND VALIDATION  ·  A HARD GATE",
    size=20, color=MINT, bold=True, space=0)
txt(s, 0.95, 1.95, 8.6, 1.9,
    [("Frontogenesis in the", {}), ("LLC4320 California Current", {})],
    size=38, bold=True, font=HEAD, color=WHITE, space=4, line=1.05)
txt(s, 0.95, 4.05, 8.2, 1.0,
    "Operators that are known correct, not assumed correct — and a figure for "
    "every methodological choice.",
    size=20, color=MINT, line=1.2)
card(s, 0.95, 5.2, 8.2, 0.95, DEEP)
txt(s, 1.3, 5.42, 7.6, 0.5,
    "M1 closed 2026-09-30 — all seven criteria PASS; gate V3 at 1.004 / 0.981",
    size=21, bold=True, color=WHITE, space=0)
txt(s, 0.95, 6.5, 8.2, 0.5,
    "J. Xavier Prochaska  ·  tile 330, face 10  ·  2026-09-28 / 30",
    size=20, color=MUTED, space=0)

# ------------------------------------------------------------------ 2 CONTENTS
s = slide()
title(s, "Contents", "Nine task sessions, then the audit and what it leaves behind")
left = [
    ("1", "Masking, tile330_masks.nc and V6", DEEP),
    ("2", "operators.py and the regression oracle", DEEP),
    ("3", "semilag.py and V5", DEEP),
    ("4", "coarsegrain.py", DEEP),
    ("5", "Gates V1, V2 and V4", DEEP),
    ("6", "Gate V3 — the discrete null", AMBER),
    ("6b", "V3b — the finite-volume null", AMBER),
]
right = [
    ("7a", "test_nan_finding.py", DEEP),
    ("7", "Decisions applied and the audit", DEEP),
    ("✓", "M1 acceptance", GREEN),
    ("⚑", "Findings for the writeup", TEAL),
    ("⚑", "Decisions and open issues", TEAL),
    ("8", "This deck", DEEP),
]
for col, x in ((left, 0.9), (right, 7.0)):
    for i, (n, h, c) in enumerate(col):
        y = 1.85 + i * 0.68
        circle(s, x, y, 0.46, n, c, size=20)
        txt(s, x + 0.65, y + 0.06, 5.3, 0.4, h, size=20, bold=True, color=MIDNIGHT, space=0)

# ------------------------------------------------------------------ 3 TASK 1
s = slide()
title(s, "Masking and tile330_masks.nc",
      "Halo = 7 × the measured dxC = 12.57 km; the 100 km cut removes the Gulf, no polygon",
      tag="TASK 1  ·  V6")
pic(s, "m1_crop_V6a.png", 0.85, 1.9, w=4.9)
bullets(s, 6.1, 1.95, 6.4, 4.8, [
    "518,400 cells → ocean 356,877 → halo 341,960 → offshore ≥ 100 km 273,431 → "
    "analysis 262,925 (73.7% of the ocean): one connected component.",
    "Gulf of California is its own ocean component (9,891 cells), max coast distance "
    "73.9 km — 0 cells survive the cut.",
    "The tile-edge rim is finite, not NaN: ~1e6x on the low edges, ~0.5x on the high. "
    "edge_cells = 7 covers it (minimum 2).",
    "17 tests. ocean_mask == isfinite(Theta) cell for cell at both hours; both "
    "halo_mask defects guarded.",
])
txt(s, 0.85, 6.85, 11.6, 0.4, "figs/V6_land_halo_tile330.png, panel (a)",
    size=20, font=MONO, color=MUTED, space=0)

# ------------------------------------------------------------------ 4 TASK 2
s = slide()
title(s, "operators.py and the regression oracle",
      "One shared operator for both sides; the oracle checks the wiring bit for bit",
      tag="TASK 2  ·  CRITERION 7")
pic(s, "m1_fig_operators.png", 0.85, 1.95, w=5.6)
txt(s, 0.85, 5.3, 5.6, 1.4,
    "The interpolated Jacobian is 0.85x the flux-form strain on the analysis mask "
    "(M0's 0.80 included the coast). The 0.911x gradb2 ratio is confirmed.",
    size=20, color=MUTED, line=1.15)
bullets(s, 6.8, 1.95, 5.7, 4.9, [
    "Criterion 7: unfiltered F vs frontogenesis_tendency — max |dF| = 0.0 on all "
    "352,673 finite cells, both hours.",
    "lowpass: a top-hat of half-width L/2; NaN propagates, never renormalised. F is "
    "finite on all 262,925 analysis cells at every L.",
    "Planning §2.4 had the strain-term sign wrong: F = −½δG + ½|σ|G cos 2θ, θ from "
    "the compressional axis.",
    "21 tests; the dims guard turns the 4-D broadcast into a ValueError before dbof.",
])

# ------------------------------------------------------------------ 5 TASK 3
s = slide()
title(s, "semilag.py and V5",
      "Interpolate b (order ≥ 3) onto the departure stencil — never G",
      tag="TASK 3  ·  V5, THE FIGURE LAUREN ASKED FOR")
pic(s, "m1_crop_V5a.png", 0.85, 1.9, w=5.5)
bullets(s, 6.7, 1.95, 5.8, 4.9, [
    "Half-cell shift, 1.5-cell front, bias at the maximum: bilinear G −4.94%, "
    "bilinear b −4.99%, cubic b −0.54%, quintic b −0.10% (prediction −5.56%).",
    "The trap: shifting b and calling gradb2 measures DG/Dt − 2F ≈ 0. The stencil "
    "is built at the departure point instead.",
    "Local Lagrange, not map_coordinates: a filled NaN leaks 46 / 12 / 3.3% at "
    "1 / 2 / 3 nodes through the spline prefilter.",
    "Real hour: order 1 fabricates +5.2% of G per hour on front pixels. 15 tests.",
])
txt(s, 0.85, 6.5, 5.5, 0.4, "figs/V5_interp_half_cell.png, panel (a)",
    size=20, font=MONO, color=MUTED, space=0)

# ------------------------------------------------------------------ 6 TASK 4
s = slide()
title(s, "coarsegrain.py",
      "The subfilter term closes the coarse-grained budget — with its divergent part τ_δ",
      tag="TASK 4")
pic(s, "m1_fig_coarsegrain.png", 1.1, 1.9, w=10.8)
bullets(s, 0.85, 5.55, 11.6, 1.6, [
    "Germano holds to 4e-16 for the composite filter; τ is exactly 0 at L = 0 and "
    "O(L²) after — it does not become the numerical-diffusion term.",
    "Returned in F units (M3's subfilter field = 2 × term). The flux form without "
    "τ_δ overstates it 2.2x on hour 0. 10 tests; task 6's discrete F improved the "
    "900 m residuals to 0.037 / 0.088.",
], space=6)

# ------------------------------------------------------------------ 7 TASK 5
s = slide()
title(s, "Gates V1, V2 and V4",
      "Two gates pass by a wide margin; V4 is the permanent error bar on every slope",
      tag="TASK 5  ·  CRITERIA 1, 2, 4")
cards = [
    ("V1 — PASS, 0.78%", "8 chained hours at ell = 8 dx; the step alone < 0.36%."),
    ("V2 — PASS, 0.077%", "b_x max; b_y 0.041%; metric alone 0.012%; swapped 87%."),
    ("V4 — 0.28 to 1.0%", "of G per hour, order 3, 1.5- to 1-cell front. Order 5: 0.06%."),
]
for i, (h, b) in enumerate(cards):
    x = 0.85 + i * 3.95
    card(s, x, 1.9, 3.7, 1.85)
    txt(s, x + 0.2, 2.02, 3.3, 0.4, h, size=21, bold=True, color=DEEP, space=0)
    txt(s, x + 0.2, 2.47, 3.3, 1.2, b, size=20, color=INK, line=1.08, space=0)
pic(s, "m1_crop_V1a.png", 0.85, 4.0, h=3.1)
pic(s, "m1_fig_gates.png", 6.6, 3.95, h=3.15)

# ------------------------------------------------------------------ 8 TASK 6
s = slide()
title(s, "Gate V3 — the discrete null",
      "First attempt FAIL 0.950 / 0.758 → PASS 1.004 [0.995, 1.017] / 0.981 [0.970, 0.994]",
      tag="TASK 6  ·  THE HARD GATE, CRITERION 3")
pic(s, "m1_crop_V3d.png", 0.85, 1.9, w=4.5)
txt(s, 0.85, 6.6, 4.5, 0.5, "V3 panel (d): LLC, n 26,293, corr 0.982",
    size=20, font=MONO, color=MUTED, space=0)
pic(s, "m1_fig_v3_changes.png", 5.7, 1.9, w=6.9)
txt(s, 5.7, 5.2, 6.8, 1.9,
    [("The chain-rule product fails on the grid by (2/3)(dx/ell)². The consistent "
      "F = −Σ (L_k b)[L_k, u·∇] b is now the default (form='discrete'); the departure "
      "velocity is cubic.", {}),
     ("Both sides of this null see the same centred velocity, so the 0.85x Jacobian "
      "attenuation cannot appear here. 70 tests, 0 skipped.", {"color": MUTED})],
    size=20, line=1.08, space=5)

# ------------------------------------------------------------------ 9 TASK 6b
s = slide()
title(s, "V3b — the finite-volume null",
      "A recorded bias, not a gate: 0.975 [0.954, 1.003] on the real hour, "
      "−0.006 ± 0.025 from V3's 0.981",
      tag="TASK 6b  ·  M1-Q2, OPTION (a)")
pic(s, "m1_crop_V3b_c.png", 0.85, 1.9, w=4.3)
txt(s, 0.85, 6.35, 4.5, 0.8, "V3b panel (c): OS7MP-like truth, discrete F",
    size=20, font=MONO, color=MUTED, space=0)
pic(s, "m1_fig_v3b.png", 5.6, 1.9, w=6.9)
txt(s, 5.6, 6.5, 6.9, 0.8,
    "fvadvect.py: flux-form C-grid advection with MITgcm face transports; "
    "centred, DST3, OS7 and OS7MP-like.",
    size=20, color=MUTED, line=1.08, space=0)

# ------------------------------------------------------------------ 10 TASK 7a
s = slide()
title(s, "test_nan_finding.py",
      "fronts_from_gradb2 under config D on NaN land: 14 tests, 11 pass, 3 strict xfails",
      tag="TASK 7a  ·  CRITERION 6")
card(s, 0.85, 1.9, 5.7, 5.0)
txt(s, 1.15, 2.1, 5.1, 0.4, "WHAT WORKS", size=20, bold=True, color=GREEN, space=0)
bullets(s, 1.15, 2.6, 5.1, 4.2, [
    "A NaN land block: 0 front pixels on NaN; fronts ≥ 10 cells from the coast "
    "equal the land-free field pixel for pixel.",
    "Real hour 0: 11,836 front pixels in 7 s, 0 on NaN, 878 (7.4%) inside the halo.",
    "The three threshold modes agree bit for bit; no coastal bias in the window.",
], space=7)
card(s, 6.75, 1.9, 5.7, 5.0, AMBERBG)
txt(s, 7.05, 2.1, 5.1, 0.4, "WHAT BREAKS  (xfail, strict)", size=20, bold=True,
    color=AMBER, space=0)
bullets(s, 7.05, 2.6, 5.1, 4.2, [
    "thresh_mode 'pool' with the default n_workers=None → TypeError: config D does "
    "not run as written.",
    "remove_small_holes fills an enclosed NaN island: 6 front pixels on land.",
    "despur on an empty skeleton → ValueError in skan.",
    "M4 recipe: pass NaN as-is (never fill), explicit n_workers, despur off, then "
    "fronts &= isfinite(gradb2).",
], space=7)

# ------------------------------------------------------------------ 11 TASK 7
s = slide()
title(s, "Decisions applied and the M1 audit",
      "M1-Q1..Q8 written into the docs on 2026-09-30; every criterion audited; M1 closed",
      tag="TASK 7  ·  THE AUDIT")
txt(s, 0.85, 1.95, 5.6, 0.4, "APPLIED TO THE DOCS", size=20, bold=True, color=DEEP, space=0)
bullets(s, 0.85, 2.45, 5.6, 4.5, [
    "Prompt 2: criteria 1, 3, 4, 5, 7 restated; Q&A header.",
    "Prompt 4 (M3) and 6 (M5): both forms of F, the baseline and its bands, order 5 "
    "as a sensitivity, Figure 2 at 0.981.",
    "Coding §2.5, §4.3, §4.4, §4.9, §4.10, §6 M1 and M3; planning §2.4 (sign final), "
    "§6 tests 1 and 3, §7.",
    "V3b figure: clipped titles and the legend fixed; every number identical.",
], space=7)
txt(s, 6.9, 1.95, 5.6, 0.4, "THE AUDIT", size=20, bold=True, color=DEEP, space=0)
bullets(s, 6.9, 2.45, 5.6, 4.5, [
    "Suite: 84 passed, 3 xfailed in 98 s; offline 66 passed, 18 deselected.",
    "Seven PNGs in figs/: five tracked (9be37dc), V3 and V3b untracked and shown; "
    "none ignored.",
    "Criteria 1-7 each PASS with the numbers of the next slide; every 'Discharges' "
    "line honoured, nothing claimed twice.",
    "Open issues carried to M2-M4: the fronts bugs, prompt 5's tile_find path, the "
    "NaN recipe, M3's requirements, the V3b band.",
], space=7)

# ------------------------------------------------------------------ 12 ACCEPTANCE
s = slide(MIDNIGHT)
title(s, "M1 acceptance", "All seven criteria, as audited in the task-7 log entry", dark=True)
crit = [
    ("V1 Cartesian deformation", "0.78% at ell = 8 dx over 8 h  (< 1%)"),
    ("V2 native-grid metric", "b_x max 0.077%, b_y 0.041%  (< 1%)"),
    ("V3 discrete null", "1.004 [0.995, 1.017] strain, 0.981 [0.970, 0.994] llc  (1 ± 0.05); "
                         "V3b recorded 0.975 [0.954, 1.003]"),
    ("V4 interpolation bias", "recorded: 0.28-1.0% of G per hour at order 3"),
    ("PNGs V1-V6 (+ V3b)", "all seven in figs/ and in git status; V6 shows the edge rim"),
    ("Tests", "84 passed, 3 strict xfailed, incl. test_nan_finding.py"),
    ("Regression oracle", "form='chain' bit-for-bit frontogenesis_tendency, max |dF| = 0"),
]
y = 1.8
for h, body in crit:
    circle(s, 0.9, y + 0.03, 0.42, "✓", GREEN, size=20)
    txt(s, 1.55, y, 10.9, 0.7,
        [([(h + "  ", {"bold": True, "color": WHITE}), (body, {"color": MINT})], {})],
        size=20, space=0, line=1.05)
    y += 0.95 if h.startswith("V3 ") else 0.6
card(s, 0.85, 6.35, 11.6, 0.8, DEEP)
txt(s, 1.25, 6.55, 10.8, 0.45,
    "M1 closed 2026-09-30.  Next: M2 pulls the 72-hour series; M3 computes the budget.",
    size=20, bold=True, color=WHITE, space=0)

# ------------------------------------------------------------------ 13 FINDINGS
s = slide()
title(s, "Findings for the writeup", "Five things the discretisation taught us", tag="FINDINGS")
finds = [
    ("Discrete F.", "The chain-rule product −(∇b)ᵀ(∇u)(∇b) fails on the grid by "
     "(2/3)(dx/ell)²; the consistent form passes V3 and is ~0.79x the chain F on real "
     "front pixels — about the size of the effect being measured."),
    ("Stencil at the departure point.", "Shifting b and differentiating measures the "
     "residual DG/Dt − 2F (≈ 0 under pure strain); the departure stencil measures DG/Dt."),
    ("The divergent subfilter correction τ_δ.", "Without it the flux form leaves 38-44% "
     "of the budget unclosed on a divergent flow and overstates the term 2.2x on hour 0."),
    ("V3b closes the 0.85x question.", "A flux-form OS7MP-like truth shifts the slope by "
     "−0.006 ± 0.025: the Jacobian attenuation is the resolved G's consistent view of the "
     "strain, not a bias. Shortfall per width: −2 / −4 / −11% at 2 / 1.5 / 1 dx."),
    ("Planning §2.4 sign.", "F = −½δG + ½|σ|G cos 2θ with θ from the compressional axis "
     "(plus, not minus). Final, M1-Q3."),
]
txt(s, 0.85, 1.9, 11.6, 5.3,
    [([(h + " ", {"bold": True, "color": DEEP}), (b, {})], {"space": 9}) for h, b in finds],
    size=20, line=1.05)

# ------------------------------------------------------------------ 14 DECISIONS / OPEN
s = slide()
title(s, "Decisions and open issues",
      "What shapes M3 and M4, and the fronts fixes JXP chose to make on this branch",
      tag="M1-Q1 .. Q15")
txt(s, 0.85, 1.95, 5.6, 0.4, "SHAPES M3 / M4", size=20, bold=True, color=DEEP, space=0)
bullets(s, 0.85, 2.45, 5.6, 4.6, [
    "Q1: both forms of F; discrete primary, chain as a stated systematic.",
    "Q2 / Q4: Figure 2 at 0.981 with its band; systematic 0.954-1.003; no upward correction.",
    "Q5 / Q6: V1 at 8 dx; V4 bar 0.28-1.0% of G/h at order 3; order 5 as a sensitivity.",
    "Subtract the numerics shortfall first: −2 / −4 / −11% at 2 / 1.5 / 1 dx.",
    "Pass τ_δ to subfilter_term; split the ratio estimator by the sign of 2F.",
], space=6)
card(s, 6.75, 1.9, 5.75, 5.2, AMBERBG)
txt(s, 7.05, 2.05, 5.2, 0.8,
    [([("NEXT: fronts FIXES — not yet applied. ", {"bold": True, "color": AMBER}),
       ("Q9: on this frontogenesis branch; JXP tells Lauren.", {"color": AMBER})], {})],
    size=20, space=0, line=1.05)
bullets(s, 7.05, 2.95, 5.2, 4.1, [
    "Q10: n_workers=None → os.cpu_count(); document the __main__ guard.",
    "Q11: fronts &= isfinite(gradb2) after cropping and dilation.",
    "Q12: despur returns early on an empty skeleton.",
    "Q13 / Q14: max_size = min_size − 1; silence the All-NaN warnings.",
    "Q15: prompt 5 → fronts_from_gradb2 + the NaN recipe; two docstrings.",
], space=5)

# ------------------------------------------------------------------ 15 TASK 8
s = slide()
title(s, "This deck", "Reproducible from the logged numbers alone; nothing below 20 pt",
      tag="TASK 8")
card(s, 0.85, 1.9, 11.6, 1.5, CARD)
txt(s, 1.25, 2.1, 10.8, 1.2,
    [("Every number here is quoted from the M1 task logs in frontogenesis_prompts.md "
      "and from frontogenesis_prompt_2.md.", {"bold": True}),
     ("Nothing is recomputed and no data store is touched.", {"color": MUTED})],
    size=20, space=4, line=1.05)
files = [
    ("make_m1_figs.py", "five panel crops from ../figs/ and five large-font re-plots"),
    ("build_m1_deck.py", "builds this .pptx"),
    ("check_m1_deck.py", "verifies every run is ≥ 20 pt and the text-box geometry"),
    ("figs_m1/", "the ten images used here"),
    ("README.md", "the work log for this task"),
]
txt(s, 0.85, 3.65, 11.6, 0.4, "IN dev/frontogenesis/deck/", size=20, bold=True,
    color=DEEP, space=0)
for i, (f, d) in enumerate(files):
    y = 4.15 + i * 0.52
    txt(s, 1.1, y, 3.6, 0.4, f, size=20, font=MONO, bold=True, color=MIDNIGHT, space=0)
    txt(s, 4.8, y + 0.02, 7.6, 0.4, d, size=20, color=MUTED, space=0)
txt(s, 0.85, 6.85, 11.6, 0.4,
    "Rendered with LibreOffice and inspected page by page before hand-over.",
    size=20, color=MUTED, space=0)

out = HERE / "Frontogenesis_M1_Acceptance.pptx"
prs.save(out)
print(f"saved {out.name}  ({out.stat().st_size/1024:.0f} KB, {len(prs.slides._sldIdLst)} slides)")
