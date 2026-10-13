""" Slope estimators, binned conditional means and the space-time block
bootstrap (M3 task 5; coding §4.8, planning §11).

Planning §11's argument is that **several distinct effects push the headline
slope below 1 with no physics in them at all** -- errors-in-variables
attenuation, selection on the outcome, correlated errors between axes, and
an effective sample size of *tens of independent patches* rather than 10^7
pixels.  This module is the machinery that keeps those four out of the
answer:

* **every estimator, not one.**  OLS (with intercept -- the gate's
  definition, as V3 declared it), orthogonal/TLS, the bisector, a trimmed
  OLS, and the ratio.  OLS attenuates by ``var(X)/(var(X)+var(eta))`` and
  TLS does not; quoting them together is how the attenuation becomes
  visible instead of being reported as damping.
* **the ratio split by sign.**  ``sum(y)/sum(x)`` is ill-conditioned when
  the pool holds both signs of ``2F`` -- 0.85 on V3's synthetic pool where
  every other estimator gave 1.004 (M1 task 6, flag 5) -- so
  :func:`ratio_estimator` reports ``pos`` and ``neg`` separately and labels
  ``all`` as the unreliable one.
* **binned conditional means, split by sign.**  Diabatic damping is
  asymmetric and a single slope averages over the asymmetry that carries
  the physics.
* **bootstrap over contiguous space-time blocks, never over pixels.**  At
  this milestone a block is a 32 x 32-cell square **in one hour**
  (:func:`block_ids`); :func:`space_time_block_ids` makes it 3 hours deep so
  the hour-to-hour correlation M2 task 3 measured (chi^2/dof 1.8 across
  pairs against the one-hour bootstrap) widens the interval instead of
  hiding in it.  Frontal features replace blocks in M4 --
  :func:`feature_bootstrap` is the §4.8 name for the same call.
* **every slope against the M1 baseline, never against 1** (coding §8).
  :func:`slope_report` divides by :data:`BASELINE` and carries the three
  systematic bands beside it, so "relative to the discrete null" is the
  default output rather than a reminder in prose.

**Reuse, not a fork.**  ``validate.slope_estimators``,
``validate.block_bootstrap_ols`` and ``validate._block_ids`` already
implement the OLS / orthogonal / GM / ratio set and the 32-cell bootstrap
(M1 task 6), and ``m2_baseline_stability.robust`` the trimmed statistic.
This module generalises them -- any estimator, time blocks, the split by
sign -- and ``tests/test_stats.py`` asserts the equivalences.  ``validate.py``
is **not edited**: it is M1's, and closed.

Numpy and pandas only; nothing here opens a store.
"""

import numpy as np
import pandas as pd

BLOCK_CELLS = 32                  # the Phase-2 block: 32 x 32 cells in one hour (planning §11)
N_BOOT = 1000
TRIM_PCT = 1.0                    # M2 task 3's trimmed statistic drops the top 1% of |x|

#: The M1 discrete-null slope every M3 slope is quoted against -- **never 1**
#: (M1-Q4; coding §8).  V3 on the real tile, order 3, discrete form.
BASELINE = 0.981
BASELINE_CI = (0.970, 0.994)      # its own block-bootstrap interval (M1 task 6)
#: V3b, the finite-volume null: a *systematic* band to carry beside the CI,
#: not an error bar (M1 task 6b -- the advection scheme the model actually ran
#: is not our semi-Lagrangian step).
V3B_BAND = (0.954, 1.003)
#: M2 task 3: mean and sd of the null slope over the 71 pairs of this window
#: -- the temporal systematic, wider than any single pair's CI.
TEMPORAL = (0.972, 0.020)


# ---------------------------------------------------------------------------
# the estimators
# ---------------------------------------------------------------------------
def _finite(x, y, *rest):
    """The cells where ``x`` and ``y`` are both finite, as float64 1-D."""
    x = np.asarray(x, dtype='float64').ravel()
    y = np.asarray(y, dtype='float64').ravel()
    if x.shape != y.shape:
        raise ValueError(f'x and y have shapes {x.shape} and {y.shape}')
    ok = np.isfinite(x) & np.isfinite(y)
    out = [x[ok], y[ok]]
    for r in rest:
        r = np.asarray(r).ravel()
        if r.shape != ok.shape:
            raise ValueError(f'labels shape {r.shape} != data shape {ok.shape}')
        out.append(r[ok])
    return tuple(out)


def _moments(x, y):
    # written exactly as ``validate.slope_estimators`` writes it (np.sum of the
    # centred products, not a dot product), so the OLS and TLS slopes are
    # **bit-identical** to M1's and not merely close: the equivalence test can
    # then be an equality, which is what makes "reuse, do not fork" checkable
    xm, ym = x.mean(), y.mean()
    return (xm, ym, float(np.sum((x - xm) ** 2)), float(np.sum((y - ym) ** 2)),
            float(np.sum((x - xm) * (y - ym))))


#: Column order of the per-block sufficient statistics the fast bootstrap path
#: uses: ``(n, sum x, sum y, sum x^2, sum y^2, sum xy)``.  Every estimator that
#: is a function of these alone can be bootstrapped without ever materialising
#: a resampled sample -- the same trick ``validate.block_bootstrap_ols`` uses,
#: and the reason a 1.9 M-pixel pool with 37,000 blocks is seconds not hours.
SUFFICIENT = ('n', 'sx', 'sy', 'sxx', 'syy', 'sxy')


def _central(T):
    """Centred moments ``(sxx, syy, sxy)`` from the raw totals ``T``
    (rows of :data:`SUFFICIENT`), for one replicate or many."""
    n, sx, sy, sxx, syy, sxy = (T[..., k] for k in range(6))
    return sxx - sx ** 2 / n, syy - sy ** 2 / n, sxy - sx * sy / n


def _moment_form(fn):
    """Mark ``fn`` as computable from :data:`SUFFICIENT` alone, and attach the
    totals -> value map the bootstrap's fast path calls."""
    def deco(est):
        est.moment_form = fn
        return est
    return deco


def _ols_T(T):
    cxx, _, cxy = _central(T)
    return cxy / cxx


def _tls_T(T):
    cxx, cyy, cxy = _central(T)
    return ((cyy - cxx) + np.sqrt((cyy - cxx) ** 2 + 4.0 * cxy ** 2)) / (2.0 * cxy)


def _bisector_T(T):
    cxx, cyy, cxy = _central(T)
    b1, b2 = cxy / cxx, cyy / cxy
    return (b1 * b2 - 1.0 + np.sqrt((1 + b1 ** 2) * (1 + b2 ** 2))) / (b1 + b2)


def _mean_y_T(T):
    return T[..., 2] / T[..., 0]


def mean_y(x, y) -> float:
    """``mean(y)`` as an estimator -- what
    :func:`binned_conditional_mean` bootstraps inside each bin."""
    x, y = _finite(x, y)
    return float(y.mean())


mean_y.moment_form = _mean_y_T


@_moment_form(_ols_T)
def slope_ols(x, y) -> float:
    """OLS slope of ``y`` on ``x``, **with intercept** -- the gate's
    definition (V3; ``validate.slope_estimators()['ols']``)."""
    x, y = _finite(x, y)
    _, _, sxx, _, sxy = _moments(x, y)
    return float(sxy / sxx)


@_moment_form(_tls_T)
def slope_tls(x, y) -> float:
    """Total least squares (orthogonal regression) with equal error
    variances -- meaningful here only because both axes carry ``s^-5``.
    Unlike OLS it is **not** attenuated by noise on ``x`` (planning §11),
    which is what makes the pair worth quoting together."""
    x, y = _finite(x, y)
    _, _, sxx, syy, sxy = _moments(x, y)
    return float(((syy - sxx) + np.sqrt((syy - sxx) ** 2 + 4.0 * sxy ** 2)) / (2.0 * sxy))


@_moment_form(_bisector_T)
def slope_bisector(x, y) -> float:
    """The bisector of the two OLS lines (Isobe et al. 1990's OLS bisector):
    ``(b1 b2 - 1 + sqrt((1 + b1^2)(1 + b2^2))) / (b1 + b2)`` with ``b1`` the
    slope of ``y`` on ``x`` and ``b2`` the inverse slope of ``x`` on ``y``.
    Lies between them by construction; reported because the two OLS
    directions bracket the truth when both axes carry error."""
    x, y = _finite(x, y)
    _, _, sxx, syy, sxy = _moments(x, y)
    b1, b2 = sxy / sxx, syy / sxy
    return float((b1 * b2 - 1.0 + np.sqrt((1 + b1 ** 2) * (1 + b2 ** 2))) / (b1 + b2))


def slope_trimmed(x, y, pct: float = TRIM_PCT) -> float:
    """OLS with the top ``pct`` % of ``|x|`` dropped -- M2 task 3's
    statistic (0.987 +- 0.005 over the 71 pairs against 0.972 +- 0.020
    raw).  ``G`` is kurtotic, so a handful of pixels carry much of
    ``sum (x - xbar)^2``; dropping them costs almost no sample and most of
    the leverage."""
    x, y = _finite(x, y)
    if not 0.0 <= pct < 100.0:
        raise ValueError(f'pct must be in [0, 100), got {pct}')
    if pct == 0.0:
        return slope_ols(x, y)
    keep = np.abs(x) < np.percentile(np.abs(x), 100.0 - pct)
    if keep.sum() < 3:
        return float('nan')
    return slope_ols(x[keep], y[keep])


def ratio_estimator(x, y, split_sign: bool = True) -> dict:
    """``sum(y) / sum(x)``, **split by the sign of ``x``** by default.

    The unsplit ratio is ill-conditioned whenever the pool holds both signs
    of ``2F``: the denominator is a difference of large numbers, and on V3's
    synthetic pool it read 0.85 where OLS, TLS, GM and the inverse all gave
    1.004 (M1 task 6, flag 5).  ``all`` is returned, and flagged, because
    planning §11 asks for it; ``pos`` and ``neg`` are the ones to quote.
    """
    x, y = _finite(x, y)
    out = dict(all=float(y.sum() / x.sum()), n=int(x.size),
               cancellation=float(np.abs(x).sum() / np.abs(x.sum())),
               note='"all" is ill-conditioned on a two-signed pool (M1 task 6, flag 5); '
                    'quote pos and neg')
    if split_sign:
        for name, m in (('pos', x > 0), ('neg', x < 0)):
            out[name] = float(y[m].sum() / x[m].sum()) if m.sum() else float('nan')
            out[f'n_{name}'] = int(m.sum())
    return out


#: every estimator :func:`slope_report` runs, by name
ESTIMATORS = dict(ols=slope_ols, tls=slope_tls, bisector=slope_bisector, trimmed=slope_trimmed)


# ---------------------------------------------------------------------------
# the blocks
# ---------------------------------------------------------------------------
def block_ids(shape, offset: int = 0, B: int = BLOCK_CELLS) -> np.ndarray:
    """Contiguous ``B x B``-cell block labels on a ``(nj, ni)`` field --
    the same construction as ``validate._block_ids`` (M1 task 6), so the two
    bootstraps resample the same blocks."""
    nj, ni = shape
    jj, ii = np.meshgrid(np.arange(nj) // B, np.arange(ni) // B, indexing='ij')
    return offset + jj * (ni // B + 1) + ii


def n_blocks(shape, B: int = BLOCK_CELLS) -> int:
    """How many distinct ids :func:`block_ids` can produce on ``shape`` --
    the stride between pairs, so ids never collide across hours."""
    nj, ni = shape
    return int((nj // B + 1) * (ni // B + 1) + 1)


def space_time_block_ids(shape, pair: int, B: int = BLOCK_CELLS, hours: int = 1) -> np.ndarray:
    """Block labels for one pair of a series, unique across pairs.

    ``hours = 1`` (the M3 block) gives every pair its own blocks;
    ``hours = 3`` makes the block **three hours deep**, so three consecutive
    pairs share a label and the bootstrap can no longer treat them as
    independent.  M2 task 3 found chi^2/dof 1.8 for the pair-to-pair scatter
    against the one-hour interval -- i.e. there is correlation the one-hour
    block does not see -- so the deeper block is how much of that is
    recovered, reported beside the one-hour number rather than instead of it.
    """
    if hours < 1:
        raise ValueError(f'hours must be >= 1, got {hours}')
    return block_ids(shape, offset=(int(pair) // int(hours)) * n_blocks(shape, B), B=B)


# ---------------------------------------------------------------------------
# the bootstrap
# ---------------------------------------------------------------------------
def _block_sums(x, y, inv, nb):
    """Per-block ``(n, sx, sy, sxx, syy, sxy)``."""
    S = np.empty((nb, 6), dtype='float64')
    for k, col in enumerate((np.ones_like(x), x, y, x * x, y * y, x * y)):
        S[:, k] = np.bincount(inv, weights=col, minlength=nb)
    return S


def _resample_index(order, starts, lens, chosen):
    """Pixel indices of the blocks in ``chosen`` (a block id per draw, with
    multiplicity), fully vectorised -- no Python loop over blocks, which is
    what makes the general path usable at 37,000 blocks."""
    take = lens[chosen]
    out = np.cumsum(take) - take                    # where each block lands
    off = np.arange(int(take.sum())) - np.repeat(out, take)
    return order[np.repeat(starts[chosen], take) + off]


def block_bootstrap(x, y, labels, estimator=slope_ols, n: int = N_BOOT, seed: int = 0,
                    ci: tuple = (2.5, 97.5)) -> dict:
    """Resample **whole blocks** with replacement and re-estimate.

    ``labels`` gives each cell its block (:func:`block_ids` /
    :func:`space_time_block_ids`); blocks are drawn with replacement by a
    multinomial over the block count -- **the same draw
    ``validate.block_bootstrap_ols`` makes for a given seed**, so with
    ``estimator=slope_ols`` the two agree to float round-off rather than
    only within bootstrap noise.

    Two paths, same answer.  An estimator carrying a ``moment_form``
    (:func:`slope_ols`, :func:`slope_tls`, :func:`slope_bisector`,
    :func:`mean_y`) is a function of the per-block sufficient statistics
    alone, so the replicates are one matrix product and the sample is never
    materialised -- seconds on the 1.9 M-pixel, 37,000-block M3 pool.  Any
    other callable ``(x, y) -> float`` (``slope_trimmed``, whose trim
    threshold depends on the resample, is the one that matters) takes the
    general path, which builds each replicate's index array; that is
    correct but ~100x slower, so pass a smaller ``n`` for it when the pool
    is large.

    Returns ``{'value', 'ci', 'lo', 'hi', 'se', 'n_blocks', 'n_boot', 'n'}``.
    Pixels are never resampled: on a field this autocorrelated a pixel
    bootstrap would give an interval smaller by the square root of the block
    size, which is the whole of planning §11's "effective sample size" point.
    """
    x, y, lab = _finite(x, y, labels)
    ids, inv = np.unique(lab, return_inverse=True)
    nb = ids.size
    if nb < 2:
        return dict(value=float(estimator(x, y)), ci=[float('nan')] * 2, lo=float('nan'),
                    hi=float('nan'), se=float('nan'), n_blocks=int(nb), n_boot=0, n=int(x.size),
                    note='fewer than two blocks: no interval')
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=int(n))
    form = getattr(estimator, 'moment_form', None)
    if form is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            vals = np.asarray(form(counts.astype('float64') @ _block_sums(x, y, inv, nb)),
                              dtype='float64')
        path = 'sufficient statistics'
    else:
        order = np.argsort(inv, kind='stable')      # cells grouped by block, once
        starts = np.searchsorted(inv[order], np.arange(nb))
        lens = np.searchsorted(inv[order], np.arange(nb), side='right') - starts
        vals = np.empty(int(n), dtype='float64')
        for r in range(int(n)):
            idx = _resample_index(order, starts, lens, np.repeat(np.arange(nb), counts[r]))
            vals[r] = estimator(x[idx], y[idx])
        path = 'resampled sample'
    lo, hi = (float(v) for v in np.nanpercentile(vals, ci))
    return dict(value=float(estimator(x, y)), ci=[lo, hi], lo=lo, hi=hi,
                se=float(np.nanstd(vals)), n_blocks=int(nb), n_boot=int(n), n=int(x.size),
                block_cells=int(BLOCK_CELLS), path=path)


def feature_bootstrap(x, y, labels, estimator=slope_ols, n: int = N_BOOT, seed: int = 0) -> dict:
    """Coding §4.8's name for :func:`block_bootstrap`.  The *labels* are what
    changes by milestone, not the arithmetic: contiguous space-time blocks in
    M3 (no front objects exist yet), frontal features in M4 (planning §11)."""
    return block_bootstrap(x, y, labels, estimator=estimator, n=n, seed=seed)


# ---------------------------------------------------------------------------
# binned conditional means
# ---------------------------------------------------------------------------
def binned_conditional_mean(x, y, bins=12, split_sign: bool = True, labels=None,
                            n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """``E[Y|X]`` per bin, **separately for ``X > 0`` and ``X < 0``**.

    ``bins``: an int (that many quantile bins **within each sign**, so the
    bins are populated rather than evenly spaced over a kurtotic axis) or an
    explicit sequence of edges.  With ``labels`` the per-bin mean also gets a
    block-bootstrap band; without, ``lo``/``hi`` are NaN and ``se`` is the
    plain standard error, which **understates** it on an autocorrelated
    field and is labelled as such.

    Planning §11 prefers this to a slope: diabatic damping is asymmetric,
    and one number averages over the asymmetry that carries the physics.
    """
    x, y, lab = _finite(x, y, labels if labels is not None else np.zeros(np.size(x)))
    groups = (('pos', x > 0), ('neg', x < 0)) if split_sign else (('all', np.ones(x.size, bool)),)
    rows = []
    for sign, m in groups:
        if m.sum() < 2:
            continue
        xs, ys, ls = x[m], y[m], lab[m]
        edges = (np.quantile(xs, np.linspace(0, 1, int(bins) + 1)) if np.isscalar(bins)
                 else np.asarray(bins, dtype='float64'))
        edges = np.unique(edges)
        k = np.clip(np.digitize(xs, edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            sel = k == b
            if not sel.any():
                continue
            row = dict(sign=sign, bin=b, x_lo=float(edges[b]), x_hi=float(edges[b + 1]),
                       x_mean=float(xs[sel].mean()), y_mean=float(ys[sel].mean()),
                       n=int(sel.sum()),
                       se_iid=float(ys[sel].std(ddof=1) / np.sqrt(sel.sum()))
                       if sel.sum() > 1 else float('nan'))
            if labels is not None:
                bb = block_bootstrap(xs[sel], ys[sel], ls[sel], estimator=mean_y,
                                     n=n_boot, seed=seed)
                row.update(lo=bb['lo'], hi=bb['hi'], se=bb['se'], n_blocks=bb['n_blocks'])
            else:
                row.update(lo=float('nan'), hi=float('nan'), se=row['se_iid'], n_blocks=0)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs['se_note'] = ('se_iid treats pixels as independent and UNDERSTATES the spread on '
                            'this field; pass labels for the block-bootstrap band')
    return out


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def slope_report(x, y, labels, *, baseline: float = BASELINE, baseline_ci=BASELINE_CI,
                 v3b_band=V3B_BAND, temporal=TEMPORAL, n_boot: int = N_BOOT,
                 n_boot_general: int = 200, seed: int = 0, estimators=None,
                 trim_pct: float = TRIM_PCT) -> dict:
    """Every estimator, its block-bootstrap interval, and each slope
    **relative to the M1 discrete-null baseline** -- coding §8's "never
    against 1", made the default output.

    ``n_boot_general`` is the replicate count for estimators without a
    ``moment_form`` -- ``slope_trimmed`` is the only one here, and on the M3
    pool it costs ~100x what the other three cost together (a percentile over
    1.9 M points per replicate), so it is bootstrapped less deeply by default.
    Its interval is correspondingly coarser; the count is in the result.

    ``relative[name]`` is ``slope / baseline``; the three systematic bands
    (the baseline's own CI, V3b's finite-volume band, and the temporal
    spread of the null across this window) are carried **separately** and
    are not folded into the interval, because they are systematics and the
    bootstrap is a statistical error.  ``verdict`` says only whether the
    OLS interval lies below the baseline's -- which is what planning §11
    calls evidence of damping; a slope merely below 1 is not.
    """
    if baseline is None or not np.isfinite(baseline) or baseline == 0:
        raise ValueError(f'baseline must be a finite non-zero slope, got {baseline!r}')
    est = dict(ESTIMATORS if estimators is None else estimators)
    x, y, lab = _finite(x, y, labels)
    out = dict(n=int(x.size), n_blocks=int(np.unique(lab).size),
               baseline=float(baseline), baseline_ci=list(baseline_ci),
               v3b_band=list(v3b_band), temporal=dict(mean=temporal[0], sd=temporal[1]),
               baseline_source='M1-Q4 / M1 task 6: the V3 discrete-null slope on the real tile',
               estimates={}, relative={})
    for name, fn in est.items():
        if name == 'trimmed':
            def f(a, b, _fn=fn):                 # the trim threshold is resample-dependent
                return _fn(a, b, trim_pct)
        else:
            f = fn
        nb = n_boot if getattr(f, 'moment_form', None) is not None else n_boot_general
        bb = block_bootstrap(x, y, lab, estimator=f, n=nb, seed=seed)
        out['estimates'][name] = bb
        out['relative'][name] = dict(value=bb['value'] / baseline,
                                     ci=[bb['lo'] / baseline, bb['hi'] / baseline])
    out['ratio'] = ratio_estimator(x, y)
    o = out['estimates']['ols']
    out['verdict'] = dict(
        below_baseline=bool(o['hi'] < baseline_ci[0]),
        above_baseline=bool(o['lo'] > baseline_ci[1]),
        consistent=bool(o['lo'] <= baseline_ci[1] and o['hi'] >= baseline_ci[0]),
        rule='damping is the OLS interval lying below the BASELINE interval, never below 1 '
             '(planning §11); the V3b and temporal bands are systematics beside it')
    return out


def format_report(rep) -> str:
    """The report as a table, baseline-relative column included."""
    w = '-' * 72
    out = [w, f'slope report: n = {rep["n"]:,} in {rep["n_blocks"]} blocks   '
              f'baseline {rep["baseline"]:.3f} {tuple(rep["baseline_ci"])}', w,
           f'  {"estimator":<10} {"slope":>8} {"95% CI":>20} {"/baseline":>10} {"CI/baseline":>20}']
    for name, e in rep['estimates'].items():
        r = rep['relative'][name]
        out.append(f'  {name:<10} {e["value"]:8.4f} [{e["lo"]:8.4f},{e["hi"]:8.4f}] '
                   f'{r["value"]:10.4f} [{r["ci"][0]:8.4f},{r["ci"][1]:8.4f}]')
    q = rep['ratio']
    out += [f'  {"ratio":<10} all {q["all"]:8.4f} (ill-conditioned)  '
            f'pos {q["pos"]:8.4f} (n {q["n_pos"]:,})  neg {q["neg"]:8.4f} (n {q["n_neg"]:,})',
            f'  systematics: V3b {tuple(rep["v3b_band"])}, '
            f'temporal {rep["temporal"]["mean"]:.3f} +- {rep["temporal"]["sd"]:.3f}',
            f'  verdict: ' + ('BELOW the baseline (damping)' if rep['verdict']['below_baseline']
                              else 'ABOVE the baseline' if rep['verdict']['above_baseline']
                              else 'consistent with the baseline'), w]
    return '\n'.join(out)
