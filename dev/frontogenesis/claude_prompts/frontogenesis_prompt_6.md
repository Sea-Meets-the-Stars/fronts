# Frontogenesis execution prompt 6 — M5: Figures and report

**Milestone:** M5 (`frontogenesis_coding.md` §6).
**Prerequisites:** M3 passed; M4 complete.
**Goal:** one command regenerates every figure from the stores, and a report that a reader can
check rather than take on trust.

---

## Tasks

### 1. `figures.py` — one function per figure

`fig01_maps` ... `fig10_term_budget`, plus `figV1..figV6` (or delegate those to `validate.py`,
which already writes them). **The figures themselves were produced in M1 (V1-V6), M3 (1-7, 10)
and M4 (8, 9); M5's job is consolidation** — uniform captions, one-command regeneration, and the
report. Every figure:

- regenerable from the zarrs by **one command**, with no manual steps;
- stating its filter scale `L` and its mask in the caption — a figure whose mask is ambiguous is
  not checkable;
- written to `dev/frontogenesis/figs/`.

Main figures 1, 2, 2b, 3, 3b, 4, 5, 6, 7, 8, 9, 10 and validation V1-V6 are specified in
planning §7. The two that carry the argument:

- **Figure 2** with the M1 discrete-null slope **drawn as a baseline line**.
- **Figure 2b**, the numerical-vs-diabatic discriminator, without which Figure 2 has no physical
  interpretation.

### 2. `frontogenesis_report.md`

Structure it so the load-bearing numbers are findable, not buried:

1. **What we measured**, with the closure residual and its tolerance stated up front.
2. **The budget**, term by term: `2F`, vertical, surface flux, subfilter `tau`, residual — with
   their relative sizes.
3. **The slope**, every estimator, with feature-level bootstrap intervals, **quoted against the
   M1 discrete-null baseline** rather than against 1.
4. **Numerical vs diabatic** — what Figure 2b actually shows. This is the section where the
   headline claim is either earned or withdrawn.
5. **Per-front results**, including whether flow-informed tracking changed the linking.
6. **Limitations**, stated plainly and without softening:
   - 72 h is 3 diurnal, 3.6 inertial and 5.8 M2 cycles — **those bands are not separable**, so
     Figure 6 is *indicative, not conclusive*. This is the strongest argument for extending to
     the full 504-hour series.
   - The `>= 100 km` offshore cut excludes the inner upwelling zone where CC frontogenesis is
     most vigorous, so the headline number describes the **offshore regime**, not the CCS.
   - LLC4320 tides are reported over-energetic; reversible tidal strain pushes the slope toward
     1 and can mask damping.
   - One region, one season, one 72-hour window.
7. **What would have made this a null result** (planning §12) — and whether any of those
   conditions was in fact met.

### 3. Reproducibility

A `README` in `dev/frontogenesis/` giving the env spec, the exact commands for M0-M5 in order,
and the expected wall time of each. Someone should be able to reproduce the whole study from the
two docs plus this README.

---

## Acceptance criteria

- Every figure regenerable by one command from the stores.
- Every figure caption states its `L` and mask.
- The report states the closure tolerance, the null-test baseline, and the §12 null-result
  criteria **explicitly**.
- Limitations section written without hedging or softening.

## Do not

- Do not present a slope without its baseline.
- Do not describe the residual as "diabatic" — with the Q13 terms measured it is
  **numerical diffusion + interior KPP**, and if the chunk terms were unavailable it is a
  catch-all. Say which.
- Do not claim the result generalises beyond one region, one season and 72 hours.

## Log

Figure inventory with paths; the headline numbers as reported; and an explicit statement of
which planning-doc claims the results **confirmed** and which they **overturned**. That last
item matters most: this project has already overturned four of its own claims during planning
(the residual being purely diabatic, the unfiltered limit isolating it, interpolation error
being negligible, and heat fluxes being unavailable) and five more in M0 against real data
(land stored as 0 — it is NaN; `W[k_l=0] ~ 0` — it is `dEta/dt`; 1.8-2.3 km spacing — it is
1.7-2.1 km on a 90-degree-rotated face; rotation terms ~0.1% — they are identically zero; a
single `kappa_num` — it is strongly scale-dependent), and the writeup should be honest about
which survived contact with data.
