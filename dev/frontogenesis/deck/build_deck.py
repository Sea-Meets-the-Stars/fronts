"""Build the planning deck, ``Frontogenesis_Planning.pptx`` -- version 2.

v1 (2026-09-19): 13 slides summarising ``frontogenesis_planning.md`` as of planning prompt 6.
v2 (2026-10-02, planning prompt 9):
  * no text smaller than 20 pt -- every run, including tags and footers (``MIN_PT``);
  * a glossary (two slides) of the primary terms;
  * a slide on how a front is tracked from one hour to the next (planning §5.7, Q14);
  * a version number on the title slide and in every footer, bumped to v2;
  * statements that the planning doc has corrected since v1 brought into line with it
    (listed on the last slide).

Content is quoted from ``../frontogenesis_planning.md``; nothing is recomputed.
``check_m1_deck.py Frontogenesis_Planning.pptx`` verifies the 20 pt floor after the build.

Run:  <frontogenesis env python> build_deck.py
Deps: python-pptx.
"""
import pathlib
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.dml import MSO_LINE

HERE = pathlib.Path(__file__).parent
VERSION, VDATE = "v2", "2 October 2026"

MIDNIGHT, DEEP, TEAL = C(0x21,0x29,0x5C), C(0x06,0x5A,0x82), C(0x1C,0x72,0x93)
MINT, WHITE, INK = C(0x8F,0xBF,0xD0), C(0xFF,0xFF,0xFF), C(0x1B,0x2A,0x33)
MUTED, LIGHT, CARD = C(0x62,0x73,0x7F), C(0xF2,0xF6,0xF8), C(0xE7,0xEF,0xF3)
AMBER, AMBERBG = C(0xA8,0x50,0x1B), C(0xFA,0xEF,0xE4)
GREY = C(0xA9,0xB4,0xBC)
HEAD, BODY, MONO = "Cambria", "Calibri", "Courier New"
MIN_PT = 20          # the v2 house rule

prs = Presentation(); prs.slide_width, prs.slide_height = I(13.333), I(7.5)
BLANK = prs.slide_layouts[6]
_n = [0]


def footer(s, dark):
    _n[0] += 1
    col = MUTED
    txt(s, 0.85, 6.98, 8.0, 0.35, f"Frontogenesis planning  ·  {VERSION}", size=20,
        color=col, space=0)
    txt(s, 10.45, 6.98, 2.0, 0.35, str(_n[0]), size=20, color=col, space=0,
        align=PP_ALIGN.RIGHT)


def slide(bg=LIGHT, foot=True):
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    r.fill.solid(); r.fill.fore_color.rgb = bg
    r.line.fill.background(); r.shadow.inherit = False
    if foot:
        footer(s, bg == MIDNIGHT)
    else:
        _n[0] += 1
    return s


def _run(p, t, size, bold, font, italic, color):
    r = p.add_run(); r.text = t; f = r.font
    f.size = Pt(max(size, MIN_PT)); f.bold = bold; f.name = font
    f.italic = italic; f.color.rgb = color
    return r


def txt(s, x, y, w, h, runs, size=20, color=INK, font=BODY, bold=False,
        align=PP_ALIGN.LEFT, space=4, line=1.05, italic=False):
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
    sh.line.fill.background(); sh.adjustments[0] = 0.06; sh.shadow.inherit = False
    return sh


def circle(s, x, y, d, label, fill=DEEP, fg=WHITE, size=20):
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL, I(x), I(y), I(d), I(d))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.fill.background(); sh.shadow.inherit = False
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = False
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    if label:
        _run(p, label, size, True, BODY, False, fg)
    return sh


def title(s, text, sub=None, dark=False):
    txt(s, 0.85, 0.42, 11.7, 0.65, text, size=32, bold=True, font=HEAD,
        color=WHITE if dark else MIDNIGHT, space=0)
    if sub:
        txt(s, 0.85, 1.12, 11.7, 0.42, sub, size=20,
            color=MINT if dark else MUTED, space=0)


def blob(s, cx, cy, w, h, rot, fill=None, line=None, dash=False, lw=2.25):
    """An elongated front-like ellipse centred at (cx, cy), rotated ``rot`` degrees."""
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL, I(cx - w / 2), I(cy - h / 2), I(w), I(h))
    sh.rotation = rot; sh.shadow.inherit = False
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(lw)
        if dash:
            sh.line.dash_style = MSO_LINE.DASH
    return sh


# ---------------------------------------------------------------- TITLE
s = slide(MIDNIGHT, foot=False)
for d, cx, cy, col in ((4.6, 9.9, 3.0, DEEP), (3.2, 11.0, 4.5, TEAL), (1.9, 10.2, 5.6, MINT)):
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, I(cx - d/2), I(cy - d/2), I(d), I(d))
    o.fill.background(); o.line.color.rgb = col; o.line.width = Pt(1.5); o.shadow.inherit = False
txt(s, 0.95, 1.35, 8.6, 0.4, "LLC4320  ·  CALIFORNIA CURRENT  ·  SURFACE",
    size=20, color=MINT, bold=True, space=0)
txt(s, 0.95, 1.95, 8.6, 2.0,
    [("Frontogenesis in the", {}), ("LLC4320 California Current", {})],
    size=40, bold=True, font=HEAD, color=WHITE, space=4, line=1.05)
txt(s, 0.95, 3.95, 8.0, 1.2,
    "Measured front strengthening against the frontogenesis rate computed from the "
    "model's own velocity and buoyancy fields.",
    size=20, color=MINT, line=1.15)
card(s, 0.95, 5.35, 1.0, 0.55, TEAL)
txt(s, 0.95, 5.43, 1.0, 0.4, VERSION, size=22, bold=True, color=WHITE,
    align=PP_ALIGN.CENTER, space=0)
txt(s, 2.15, 5.43, 7.0, 0.4, f"Planning summary  ·  {VDATE}", size=22, bold=True,
    color=WHITE, space=0)
txt(s, 0.95, 6.2, 8.6, 0.4, "J. Xavier Prochaska  ·  HIINet", size=20, color=MINT, space=0)

# ---------------------------------------------------------------- QUESTION
s = slide()
title(s, "The question", "Two numbers that should agree — and the reasons they will not")
for x, sym, lab, col, body, mono in (
        (0.85, "M", "MEASURED", DEEP,
         "How much fronts actually sharpen, hour to hour, following the flow.",
         "DG/Dt  from snapshots"),
        (6.85, "P", "PREDICTED", TEAL,
         "How much the strain field says they should sharpen, instantaneously.",
         "2F  from u, v and b")):
    card(s, x, 1.85, 5.6, 2.85, CARD)
    circle(s, x + 0.35, 2.12, 0.55, sym, col)
    txt(s, x + 1.1, 2.22, 4.2, 0.4, lab, size=20, bold=True, color=col, space=0)
    txt(s, x + 0.35, 2.9, 4.95, 1.1, body, size=20, color=INK, line=1.1)
    txt(s, x + 0.35, 4.05, 4.95, 0.4, mono, size=20, font=MONO, color=MUTED, space=0)
card(s, 0.85, 5.0, 11.6, 1.7, MIDNIGHT)
txt(s, 1.3, 5.25, 10.7, 1.3,
    [("These are not two guesses at one number.", {"bold": True, "color": WHITE}),
     ("They are the two sides of an exact budget — and the residual between them is the "
      "physical result.", {"color": MINT})],
    size=21, space=4, line=1.1)

# ---------------------------------------------------------------- BUDGET
s = slide()
title(s, "One budget, three terms", "The surface buoyancy-gradient budget, stated exactly")
card(s, 0.85, 1.8, 11.6, 0.95, MIDNIGHT)
txt(s, 1.1, 2.08, 11.1, 0.45,
    "D/Dt (½G)  =  F  −  b_z (w_x b_x + w_y b_y)  +  ∇b · ∇B",
    size=22, font=MONO, color=WHITE, space=0, align=PP_ALIGN.CENTER)
rows = [
    ("F", "KINEMATIC", "From u, v and b, with one shared operator on the native grid.", DEEP, CARD),
    ("T", "TILTING", "Cancels at the free surface: there w = Dη/Dt (the kinematic condition).", TEAL, CARD),
    ("B", "DIABATIC", "Air-sea fluxes and mixing — plus whatever the numerics do.", AMBER, AMBERBG),
]
for i, (sym, head, body, col, bg) in enumerate(rows):
    y = 3.0 + i * 1.05
    card(s, 0.85, y, 11.6, 0.9, bg)
    circle(s, 1.1, y + 0.18, 0.55, sym, col)
    txt(s, 1.9, y + 0.26, 2.0, 0.4, head, size=20, bold=True, color=col, space=0)
    txt(s, 3.9, y + 0.26, 8.4, 0.4, body, size=20, color=INK, space=0)
txt(s, 0.85, 6.25, 11.6, 0.45,
    "Convention: F = ½ DG/Dt, so every comparison is 2F against the measured tendency.",
    size=20, bold=True, color=MIDNIGHT, align=PP_ALIGN.CENTER, space=0)

# ---------------------------------------------------------------- SURFACE-ONLY
s = slide()
title(s, "Why surface-only is enough", "and the one place the discretisation takes it back")
for x, sym, lab, col, bg, body, punch in (
        (0.85, "✓", "IN THE CONTINUUM", DEEP, CARD,
         "At the free surface w = Dη/Dt, so the along-surface budget loses the tilting "
         "term exactly. No kinematic term is missing.",
         "Surface data is sufficient."),
        (6.85, "!", "IN THE DATA", AMBER, AMBERBG,
         "k = 0 is a fixed 1 m cell, not the surface. Its budget carries the flux through "
         "the cell base, where w does not cancel.",
         "~30% of F by day, ~0 at night.")):
    card(s, x, 1.8, 5.6, 3.5, bg)
    circle(s, x + 0.35, 2.05, 0.55, sym, col, size=22)
    txt(s, x + 1.1, 2.15, 4.3, 0.4, lab, size=20, bold=True, color=col, space=0)
    txt(s, x + 0.35, 2.85, 4.95, 1.8, body, size=20, color=INK, line=1.1)
    txt(s, x + 0.35, 4.7, 4.95, 0.4, punch, size=20, bold=True, color=col, space=0)
card(s, 0.85, 5.55, 11.6, 1.15, MIDNIGHT)
txt(s, 1.25, 5.72, 10.8, 0.9,
    [("Now measured, not bounded: ", {"bold": True, "color": WHITE}),
     ("hourly full-depth chunks give b_z and the cell-base W at every step (Q13).",
      {"color": MINT})],
    size=20, line=1.1, space=0)

# ---------------------------------------------------------------- EXISTS / NEW
s = slide()
title(s, "Half the study is already built", "The predicted side exists, and is built well")
have = ["F on the native grid", "Metric-correct gradients, Jacobian",
        "Front detection (NaN-safe)", "Front tracking: follow()",
        "Per-front property statistics", "Front cross-sections (curtains)"]
new = ["Measured DG/Dt (semi-Lagrangian)", "Coarse-graining, subfilter flux τ",
       "Time series concatenated to zarr", "Discrete null-test harness",
       "Flow-informed front tracking", "Vertical + surface-flux terms"]
for x, lab, items, bg, fg, hc in ((0.85, "ALREADY EXISTS", have, CARD, INK, DEEP),
                                   (6.85, "THE NEW WORK", new, MIDNIGHT, WHITE, MINT)):
    card(s, x, 1.8, 5.6, 4.35, bg)
    txt(s, x + 0.35, 2.05, 4.95, 0.4, lab, size=20, bold=True, color=hc, space=0)
    txt(s, x + 0.35, 2.65, 5.0, 3.4, [(t, {"bullet": True}) for t in items],
        size=20, color=fg, space=9, line=1.0)
txt(s, 0.85, 6.35, 11.6, 0.4,
    "Nobody has yet measured how fast the fronts actually sharpen. That is the gap.",
    size=20, italic=True, color=MUTED, align=PP_ALIGN.CENTER, space=0)

# ---------------------------------------------------------------- DATA
s = slide(MIDNIGHT)
title(s, "The data", "Tile 330 (face 10), 2–4 July 2012, 72 consecutive hours", dark=True)
stats = [("720²", "native cells"), ("72", "hourly steps"), ("1.7–2.1 km", "grid spacing"),
         ("0.9 GB", "surface fields")]
for i, (big, lab) in enumerate(stats):
    x = 0.85 + i * 2.95
    card(s, x, 1.8, 2.7, 1.45, DEEP)
    txt(s, x + 0.1, 1.98, 2.5, 0.6, big, size=28, bold=True, font=HEAD,
        color=WHITE, align=PP_ALIGN.CENTER, space=0)
    txt(s, x + 0.1, 2.68, 2.5, 0.4, lab, size=20, color=MINT,
        align=PP_ALIGN.CENTER, space=0)
txt(s, 0.85, 3.55, 11.6, 0.4, "THREE SOURCES", size=20, bold=True, color=MINT, space=0)
srcs = [("OSN surface", "Theta, Salt, U, V (+ W, Eta) — the primary fields"),
        ("OSN llc_wind", "KPPhbl (boundary-layer depth) and wind stress"),
        ("Hourly chunks", "k = 0–2 Theta, Salt, W; oceQnet, oceQsw, oceFWflx")]
for i, (h, b) in enumerate(srcs):
    y = 4.1 + i * 0.62
    txt(s, 0.85, y, 2.9, 0.4, h, size=20, bold=True, color=WHITE, space=0)
    txt(s, 3.75, y, 8.7, 0.4, b, size=20, color=MINT, space=0)
txt(s, 0.85, 6.05, 11.6, 0.75,
    "The chunks add budget terms, not primary fields — and OSN vs chunks is a free cross-check.", size=20, italic=True, color=MINT, line=1.05, space=0)

# ---------------------------------------------------------------- METHOD
s = slide()
title(s, "Method", "Five commitments that make the comparison mean something")
items = [
    ("One operator, both sides", "Same stencil, same filter for both — or we manufacture a mismatch."),
    ("Semi-Lagrangian, not Eulerian", "One difference along the trajectory; cubic b at the departure point."),
    ("Identical filtering of b, u and v", "Swept over L = 0, 2, 4, 8 cells, with the subfilter flux τ computed."),
    ("Land masked before differencing", "OSN land is NaN; a 7-cell halo and a tile-edge margin on top."),
    ("Statistics ≥ 100 km offshore", "Also reported by distance, so what we excluded stays visible."),
]
for i, (h, b) in enumerate(items):
    y = 1.8 + i * 0.95
    circle(s, 0.88, y + 0.12, 0.55, str(i + 1), DEEP if i % 2 == 0 else TEAL)
    txt(s, 1.7, y, 10.8, 0.4, h, size=21, bold=True, color=MIDNIGHT, space=0)
    txt(s, 1.7, y + 0.42, 10.8, 0.4, b, size=20, color=MUTED, space=0)

# ---------------------------------------------------------------- HARD PART
s = slide()
title(s, "The residual is not purely diabatic",
      "The correction that most changes how we read the result")
card(s, 0.85, 1.8, 11.6, 0.95, AMBERBG)
txt(s, 1.2, 2.06, 10.9, 0.5,
    "A slope of 0.5–0.8 at the grid scale is explicable by the model's numerics alone.",
    size=21, bold=True, color=AMBER, align=PP_ALIGN.CENTER, space=0)
blocks = [
    ("Numerical diffusion", "No explicit diffusion; OS7MP's implicit damping is ~0.1–0.5 f at 4Δx."),
    ("Finite-cell vertical term", "~30% of F by day — now measured from the hourly chunks."),
    ("Discrete operator bias", "Chain rule + C-grid interpolation: slope 0.7–1.4, no physics."),
]
for i, (h, b) in enumerate(blocks):
    y = 3.0 + i * 0.92
    circle(s, 0.88, y + 0.1, 0.5, "", AMBER if i == 0 else (TEAL if i == 1 else DEEP))
    txt(s, 1.65, y, 10.8, 0.4, h, size=21, bold=True, color=MIDNIGHT, space=0)
    txt(s, 1.65, y + 0.42, 10.8, 0.4, b, size=20, color=INK, space=0)
card(s, 0.85, 5.85, 11.6, 0.9, MIDNIGHT)
txt(s, 1.2, 5.95, 10.9, 0.75,
    "With the vertical and surface-flux terms measured, the residual reduces to numerical "
    "diffusion + interior KPP.", size=20, color=WHITE, align=PP_ALIGN.CENTER, line=1.0, space=0)

# ---------------------------------------------------------------- GATES
s = slide()
title(s, "Four gates before any science", "Operators must be known correct, not assumed correct")
gates = [
    ("1", "Cartesian scheme", "Pure deformation: G grows exactly as exp(2αt).", DEEP, CARD),
    ("2", "Native-grid metrics", "An analytic field of XC, YC with known gradients.", TEAL, CARD),
    ("3", "Discrete null", "Our exact operators on a synthetic tracer: slope = 1 ± 0.05.", AMBER, AMBERBG),
    ("4", "Interpolation bias", "Uniform zero-strain flow: the truth is 0, so out comes the bias.", MIDNIGHT, CARD),
]
for i, (n, h, b, col, bg) in enumerate(gates):
    x = 0.85 + (i % 2) * 5.95
    y = 1.8 + (i // 2) * 2.15
    card(s, x, y, 5.6, 1.95, bg)
    circle(s, x + 0.33, y + 0.25, 0.55, n, col)
    txt(s, x + 1.08, y + 0.33, 4.3, 0.4, h, size=21, bold=True, color=MIDNIGHT, space=0)
    txt(s, x + 0.33, y + 0.98, 5.0, 0.85, b, size=20, color=INK, line=1.05)
txt(s, 0.85, 6.15, 11.6, 0.65,
    "Gate 3 protects the headline: every slope is quoted against its baseline, never against 1.",
    size=20, italic=True, color=MUTED, align=PP_ALIGN.CENTER, line=1.0, space=0)

# ---------------------------------------------------------------- FIGURES
s = slide()
title(s, "What we will show", "Ten main figures and six validation figures; these six carry the argument")
figs = [
    ("Maps side by side", "measured beside 2F, shared scale, plus residual"),
    ("Joint PDF", "measured vs 2F, null baseline drawn on it"),
    ("Numerical vs diabatic", "residual vs ∇⁴b and vs KPPhbl (2b)"),
    ("Slope vs filter scale", "and the τ sweep shown panel by panel (3b)"),
    ("Strain alignment", "∇b vs the compressional axis"),
    ("Term budget", "2F, vertical, surface flux, τ, residual"),
]
for i, (h, b) in enumerate(figs):
    y = 1.75 + i * 0.75
    circle(s, 0.88, y, 0.55, str(i + 1), [DEEP, TEAL, AMBER, DEEP, TEAL, AMBER][i])
    txt(s, 1.7, y + 0.08, 3.9, 0.4, h, size=21, bold=True, color=MIDNIGHT, space=0)
    txt(s, 5.6, y + 0.09, 6.85, 0.4, b, size=20, color=MUTED, space=0)
txt(s, 0.85, 6.4, 11.6, 0.35,
    "Then the fronts: one tracked front, hour by hour, and the 10 largest as a population.",
    size=20, italic=True, color=MUTED, align=PP_ALIGN.CENTER, space=0)

# ---------------------------------------------------------------- TRACKING (new in v2)
s = slide()
title(s, "Tracking a front from one hour to the next",
      "Flow-informed follow() — planning §5.7, decision Q14")
# -- the cartoon: front A at t, its mask advected by the flow, and two candidates at t+1
card(s, 0.85, 1.8, 5.5, 3.7, CARD)
blob(s, 2.25, 2.95, 2.3, 0.36, -18, fill=MINT)
txt(s, 1.05, 2.0, 2.6, 0.4, "front A at t", size=20, bold=True, color=TEAL, space=0)
arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, I(3.15), I(3.2), I(0.95), I(0.36))
arr.rotation = 35; arr.fill.solid(); arr.fill.fore_color.rgb = MUTED
arr.line.fill.background(); arr.shadow.inherit = False
txt(s, 4.15, 2.75, 2.1, 0.4, "u, v", size=20, bold=True, font=MONO, color=MUTED, space=0)
blob(s, 4.45, 4.05, 2.15, 0.34, -14, fill=DEEP)                 # the true continuation
blob(s, 4.35, 4.0, 2.35, 0.5, -18, line=AMBER, dash=True)       # flow-predicted mask
blob(s, 1.95, 4.15, 1.7, 0.3, -18, fill=GREY)                   # stationary neighbour
txt(s, 3.25, 4.5, 3.0, 0.4, "A at t+1: link", size=20, bold=True, color=DEEP, space=0)
txt(s, 1.05, 4.5, 2.2, 0.4, "B: nearer", size=20, bold=True, color=MUTED, space=0)
txt(s, 1.05, 4.95, 5.2, 0.4, "dashed: A's mask moved by the flow", size=20, color=AMBER,
    space=0)
# -- the steps
steps = [
    "Find and label fronts at t and at t+1, independently.",
    "Advect A's boolean mask with u, v (semilag); threshold at 0.5.",
    "Score each candidate: follow()'s position, overlap, length, area and "
    "orientation, plus IoU with the predicted mask.",
    "Best score ≤ 2.5 is the link; none passes → a gap, not a guess.",
]
y = 1.8
for i, t in enumerate(steps):
    circle(s, 6.65, y + 0.02, 0.45, str(i + 1), DEEP if i % 2 == 0 else TEAL)
    h = 0.72 if len(t) < 75 else 1.05
    txt(s, 7.3, y, 5.15, h, t, size=20, color=INK, line=1.0, space=0)
    y += h + 0.12
card(s, 0.85, 5.65, 11.6, 1.2, MIDNIGHT)
txt(s, 1.2, 5.75, 10.9, 1.0,
    [("Why: ", {"bold": True, "color": WHITE}),
     ("a wrong link breaks the material derivative, so the budget is compared on the "
      "advected pixel set. Overlap with two fronts flags a split or merge.",
      {"color": MINT})],
    size=20, line=1.0, space=0)

# ---------------------------------------------------------------- MILESTONES
s = slide(MIDNIGHT)
title(s, "Milestones", "Each has its own execution prompt document", dark=True)
ms = [("M0", "Access", False), ("M1", "Operators", True), ("M2", "Data pull", False),
      ("M3", "Budget", True), ("M4", "Fronts", False), ("M5", "Figures", False)]
for i, (m, lab, gate) in enumerate(ms):
    x = 0.85 + i * 2.0
    card(s, x, 1.85, 1.8, 1.45, AMBER if gate else DEEP)
    txt(s, x + 0.05, 2.02, 1.7, 0.5, m, size=26, bold=True, font=HEAD,
        color=WHITE, align=PP_ALIGN.CENTER, space=0)
    txt(s, x + 0.05, 2.65, 1.7, 0.4, lab, size=20,
        color=WHITE if gate else MINT, align=PP_ALIGN.CENTER, space=0)
    if gate:
        txt(s, x + 0.05, 3.42, 1.7, 0.35, "GATE", size=20, bold=True,
            color=AMBER, align=PP_ALIGN.CENTER, space=0)
for x, h, b in ((0.85, "M1 — operators",
                 "Passes only when the discrete null gives slope = 1 ± 0.05. On failure we "
                 "fix the operators, not the interpretation."),
                (6.85, "M3 — budget",
                 "Exit is budget closure, not a slope: no efficiency is quoted until the "
                 "terms balance to a stated tolerance.")):
    txt(s, x, 4.0, 5.6, 2.2,
        [(h, {"bold": True, "color": WHITE, "space": 4}),
         (b, {"color": MINT, "line": 1.05})], size=20)
txt(s, 0.85, 6.4, 11.6, 0.4,
    "M1 and M2 can run in parallel — the data pull needs no physics.",
    size=20, italic=True, color=MINT, align=PP_ALIGN.CENTER, space=0)

# ---------------------------------------------------------------- RISKS
s = slide()
title(s, "The honest risks", "and the one decision that was open in v1")
card(s, 0.85, 1.75, 11.6, 1.05, CARD)
circle(s, 1.15, 2.0, 0.55, "✓", DEEP, size=22)
txt(s, 1.95, 1.9, 10.3, 0.8,
    [("Resolved: which branches to work from (Q15). ", {"bold": True, "color": DEEP}),
     ("A four-step merge sequence; the work is not blocked on it.", {"color": INK})],
    size=20, line=1.0, space=0)
risks = [
    ("Numerical diffusion mimics diabatic damping", "same order as the strain"),
    ("Discrete operators bias the slope", "0.7–1.4, no physics"),
    ("Semi-Lagrangian interpolation bias", "25–80% of the signal"),
    ("72 h cannot separate K1, M2 and inertial", "3, 5.8 and 3.6 cycles"),
    ("Tracking links to the wrong front", "flow-informed follow()"),
]
txt(s, 0.85, 3.05, 11.6, 0.4, "TOP RISKS", size=20, bold=True, color=DEEP, space=0)
for i, (h, b) in enumerate(risks):
    y = 3.6 + i * 0.62
    circle(s, 0.9, y + 0.07, 0.28, "", TEAL if i % 2 else DEEP)
    txt(s, 1.4, y, 6.9, 0.4, h, size=20, bold=True, color=INK, space=0)
    txt(s, 8.4, y, 4.05, 0.4, b, size=20, color=MUTED, space=0)

# ---------------------------------------------------------------- NULL
s = slide(MIDNIGHT)
title(s, "What would make this a null result",
      "Stated now, so we recognise it rather than rationalise around it", dark=True)
nulls = ["The budget does not close at any filter scale.",
         "The slope is indistinguishable from the discrete-null baseline.",
         "The residual tracks ∇⁴b (numerics), not KPPhbl or the diurnal cycle.",
         "The discrete null test cannot be made to pass at 1 ± 0.05."]
for i, t in enumerate(nulls):
    y = 1.85 + i * 0.85
    circle(s, 0.88, y, 0.55, str(i + 1), DEEP)
    txt(s, 1.7, y + 0.08, 10.8, 0.45, t, size=21, color=WHITE, space=0)
card(s, 0.85, 5.4, 11.6, 1.35, DEEP)
txt(s, 1.25, 5.55, 10.8, 1.1,
    "Then the conclusion is methodological, the limiting factor is named — and that beats "
    "a slope of 0.6 presented as an efficiency.",
    size=21, bold=True, color=WHITE, align=PP_ALIGN.CENTER, line=1.05, space=0)

# ---------------------------------------------------------------- GLOSSARY (new in v2)
def glossary(s, entries):
    txt(s, 0.85, 1.75, 11.65, 5.1,
        [([(term + "  ", {"bold": True, "color": DEEP}), (defn, {})], {"space": 6})
         for term, defn in entries],
        size=20, line=1.0)


s = slide()
title(s, "Glossary — the physics", "Notation as on the slides")
glossary(s, [
    ("Front, frontogenesis", "a narrow band of strong ∇b; frontogenesis is its sharpening (G rising)."),
    ("b", "buoyancy, g σ₀/ρ₀ from surface Theta and Salt (JMD95, the model's own EOS)."),
    ("G", "front strength, |∇b|² = b_x² + b_y²."),
    ("F", "kinematic frontogenesis rate from u, v and b; F = ½ DG/Dt with no mixing."),
    ("DG/Dt", "the measured tendency: the change in G following a parcel over one hour."),
    ("Residual", "measured DG/Dt − 2F: numerics, mixing and forcing together."),
    ("Tilting term", "−b_z (w_x b_x + w_y b_y); zero at the surface, back in the 1 m top cell."),
    ("Diabatic, B", "buoyancy sources: air-sea heat and freshwater fluxes, KPP mixing."),
    ("Strain, θ", "the deformation of the flow; θ is the angle of ∇b to its compressional axis."),
])

s = slide()
title(s, "Glossary — the method and the data")
glossary(s, [
    ("Semi-Lagrangian", "difference G along a parcel, from its departure point an hour earlier."),
    ("Front pixels", "cells with G above a set percentile at the midpoint time, inside the mask."),
    ("Filter scale L", "b, u and v smoothed identically over L cells (0, 2, 4, 8)."),
    ("Subfilter flux τ", "the buoyancy flux carried by scales below L; computed, not assumed."),
    ("Discrete null", "a synthetic test with known slope 1; its fitted value is the baseline."),
    ("OS7MP", "LLC4320's tracer advection; its implicit diffusion is the rival signal."),
    ("KPP, KPPhbl", "the model's mixing scheme and its boundary-layer depth."),
    ("follow(), IoU", "the repo's front tracker; IoU = overlap ÷ union of two masks."),
    ("Advected pixel set", "a front's pixels at t carried by the flow to t+1: same water, both hours."),
    ("Tile 330, OSN, chunks", "our 720² window; the public surface store; the full-depth store."),
])

# ---------------------------------------------------------------- VERSIONS (new in v2)
s = slide()
title(s, "Version history")
card(s, 0.85, 1.45, 11.6, 1.1, CARD)
txt(s, 1.2, 1.6, 11.0, 0.85,
    [("v1  ·  19 Sep 2026", {"bold": True, "color": DEEP, "space": 2}),
     ("13 slides, from the planning doc as of planning prompt 6.", {})],
    size=20, space=0)
card(s, 0.85, 2.75, 11.6, 4.05, CARD)
txt(s, 1.2, 2.9, 11.0, 3.8,
    [("v2  ·  2 Oct 2026", {"bold": True, "color": DEEP, "space": 2}),
     ("No text below 20 pt; a glossary; a tracking slide; a version number.", {"space": 10}),
     ("Also brought into line with the planning doc as corrected since v1:",
      {"bold": True, "color": MIDNIGHT, "space": 4}),
     ("surface w is Dη/Dt, not zero — the tilting term still cancels", {"bullet": True}),
     ("the top-cell vertical term is measured (hourly chunks), not bounded", {"bullet": True}),
     ("OSN land is NaN, not zero; surface fluxes come from the chunks", {"bullet": True}),
     ("numerical damping ~0.1–0.5 f at 4Δx (was 0.1–1 f)", {"bullet": True}),
     ("the branch decision is resolved (Q15)", {"bullet": True})],
    size=20, space=3, line=1.0)

out = HERE / "Frontogenesis_Planning.pptx"
prs.save(out)
print(f"saved {out.name}  ({out.stat().st_size/1024:.0f} KB, {len(prs.slides._sldIdLst)} slides)")
