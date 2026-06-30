# Pull Requests

This file will hold prompts to perform reviews of pull requests.

All of these will be at:
https://github.com/Sea-Meets-the-Stars/fronts/pull/##

## PR 22

1. Review PR 22 and post to GitHub.  Be sure to include comments on specific lines.  If you have any questions, put them under the Clarifications section. Log your work.

### Clarifications

1. **Merge dependency.** The PR body says this requires the `tile_fields` branch in the preprocessing repo, which is not yet merged to `main`. Is merging PR 22 blocked on that branch landing first, or is it fine to merge the fronts side ahead of it (the new code degrades to a clear `ImportError`/`SystemExit` without it)?
2. **`--clim` default semantics.** When `--field-transform`/`--field-clip` are overridden, should the pinned style `clim` be honored or dropped in favor of percentiles? `fronts_viz_3d` drops it; `fronts_viz_curtain` keeps it (see the inline comment on `fronts_viz_curtain.py:544`). Which is the intended behavior — I'll align both to match.
3. **Tests.** Are unit tests for `field_styles.py` and the figure assemblers in scope for this PR, or deferred to a follow-up?

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-06-30 (Reviewed PR 22 — dual-field 3-D coloring + 2-D curtains)

Reviewed [PR 22](https://github.com/Sea-Meets-the-Stars/fronts/pull/22) ("Plot any field in 3D or in 2D along/across front curtains", +3111/-75, 12 files) and posted a `COMMENTED` review with 6 inline comments ([review](https://github.com/Sea-Meets-the-Stars/fronts/pull/22#pullrequestreview-4600886638)).

**What the PR does:** generalizes the isopycnal 3-D viz so any field can color the σ0 iso-surfaces (geometry stays density-driven; the color field rides along via VTK's `contour` filter interpolating every point array onto the extracted surfaces — no resampling code). Adds `fronts/viz/field_styles.py` (per-variable transform/clip/cmap/title/center registry), `fronts/viz/curtains.py` + `fronts/scripts/fronts_viz_curtain.py` (2-D vertical "curtain" cross-sections: main-axis, along-front offsets, perpendicular transect), and refactors `dev/mld/density_utils.py` (`load_tile` generalized from `load_density_tile`; `check_tiles_consistent`; robust `tile_mapping` import via installed `dbof` → `LLC4320_PREPROC_SRC` env var → clear `ImportError`).

**Findings posted (all non-blocking):**
- `fronts_viz_curtain.py:544` — likely real bug: `default_clim` returns the pinned style `clim` even when `--field-transform`/`--field-clip` are overridden, unlike `fronts_viz_3d.py` which guards with `style_clim_ok`. A `symlog` override on Ri would get a `log10`-space colorbar. (strongest finding)
- `density_utils.py:222` — `check_tiles_consistent` raises `SystemExit` from a shared util (sibling `load_tile` uses `KeyError`); suggested `ValueError` + CLI translation.
- `curtains.py:480` — `trim_offset_loops` is worst-case ~O(L³) (restarts the O(L²) scan after each excision); fine for typical lengths, suggested an iteration note/cap.
- `curtains.py:1053` — `figure_perpendicular` computes `path_metrics(perp_path, …)` whose result is unused (dead work + misleading `XC/YC` args); tie to the documented km-twin-axis follow-up.
- `field_styles.py:125` — trailing whitespace on the `okubo_weiss` entry.
- `test_curtains.py:6` — no tests for `field_styles.py` (`apply_transform`/`default_clim`) or the figure assemblers.

**What I learned / verified:** confirmed top-level `dbof` resolves to the installed preprocessing repo (no collision with `fronts.dbof`); `mixed_layer_clim` is NaN-safe (`nanpercentile`), so feeding it the NaN-laden transformed field is OK. The dual-field path is fully backward compatible — without `--field-tile` the 3-D scene is unchanged. 3 questions raised under Clarifications (merge dependency on the unmerged `tile_fields` preprocessing branch; intended `--clim`-vs-override semantics; whether the missing tests are in scope).