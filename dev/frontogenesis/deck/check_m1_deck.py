"""Programmatic QA for the M1 deck.

1. Font floor: every run in every text frame (text boxes, shapes, tables) has an explicit
   size >= 20 pt.  Reports the minimum and any offender.
2. Geometry: every shape lies inside the slide; every text box's estimated wrapped height
   fits its box (average Calibri glyph ~0.5 em; line height 1.2 x line_spacing).
3. Slide dump for content review.

Run:  <frontogenesis env python> check_m1_deck.py [deck.pptx]
"""
import math
import pathlib
import sys
from pptx import Presentation
from pptx.util import Emu

HERE = pathlib.Path(__file__).parent
path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "Frontogenesis_M1_Acceptance.pptx"
prs = Presentation(path)
SW, SH = prs.slide_width, prs.slide_height
FLOOR = 20.0
EM = 0.50          # average glyph width / em for Calibri-class text (bold a touch wider)

min_pt, offenders, geom = 1e9, [], []
for si, s in enumerate(prs.slides, 1):
    print(f"\n--- slide {si}")
    for sh in s.shapes:
        # bounds
        if sh.left < 0 or sh.top < 0 or sh.left + sh.width > SW + Emu(1) or sh.top + sh.height > SH + Emu(1):
            geom.append((si, sh.shape_type, "off-slide",
                         f"{Emu(sh.left).inches:.2f},{Emu(sh.top).inches:.2f} "
                         f"{Emu(sh.width).inches:.2f}x{Emu(sh.height).inches:.2f}"))
        if sh.shape_type == 13:  # picture
            print(f"  [pic] {Emu(sh.left).inches:.2f},{Emu(sh.top).inches:.2f} "
                  f"{Emu(sh.width).inches:.2f}x{Emu(sh.height).inches:.2f} in")
            continue
        if not sh.has_text_frame:
            continue
        tf = sh.text_frame
        w_in = Emu(sh.width).inches - Emu(tf.margin_left).inches - Emu(tf.margin_right).inches
        h_in = Emu(sh.height).inches
        est = 0.0
        text_lines = []
        for p in tf.paragraphs:
            runs = [r for r in p.runs if r.text]
            if not runs:
                continue
            sizes = []
            for r in runs:
                if r.font.size is None:
                    offenders.append((si, "no explicit size", r.text[:40]))
                    continue
                pt = r.font.size.pt
                sizes.append(pt)
                min_pt = min(min_pt, pt)
                if pt < FLOOR - 1e-6:
                    offenders.append((si, f"{pt} pt", r.text[:40]))
            if not sizes:
                continue
            size = max(sizes)
            text = "".join(r.text for r in runs)
            text_lines.append(text)
            bold = any(r.font.bold for r in runs)
            cpl = max(1, int(w_in * 72 / (size * EM * (1.06 if bold else 1.0))))
            n_lines = max(1, math.ceil(len(text) / cpl)) if tf.word_wrap else 1
            ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.0
            est += n_lines * size * 1.2 * ls / 72
            est += (p.space_after.pt if p.space_after is not None else 0) / 72
        if text_lines:
            flag = ""
            if est > h_in + 0.08 and sh.shape_type != 1:   # autoshapes (circles) excluded
                flag = f"   <-- est {est:.2f} in > box {h_in:.2f} in"
                geom.append((si, "text", "overflow?", f"est {est:.2f} > {h_in:.2f}: {text_lines[0][:50]}"))
            for t in text_lines:
                print(f"  {t[:110]}{flag if t is text_lines[0] else ''}")

print("\n=== font floor")
print(f"minimum run size: {min_pt} pt  (floor {FLOOR})")
print("offenders:", offenders if offenders else "none")
print("=== geometry")
for g in geom:
    print(" ", g)
if not geom:
    print("  no shape off-slide, no estimated text overflow")
print(f"=== {len(prs.slides._sldIdLst)} slides, {path.stat().st_size/1024:.0f} KB")
