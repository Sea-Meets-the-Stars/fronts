"""Post-processing calculations that characterise front type.

These functions take pre-computed gradient fields saved by the upstream
LLC4320 pipeline (gradtheta2, gradsalt2, gradrho2, ...) and return derived
oceanographic quantities.  No xgcm Grid or xarray accessor is needed —
plain NumPy arrays in, NumPy arrays out.

Typical workflow:
  1. The upstream pipeline computes squared gradient fields from native
     LLC4320 output and saves them as NetCDF, e.g.:
         LLC4320_2012-11-09T12_00_00_gradtheta2_v1.nc
  2. Load those fields (via xarray or colocation._load_property_file).
  3. Pass the arrays here — either full 2-D grids or 1-D per-front vectors
     from a colocate_fronts_with_properties() DataFrame.

Gridded example:
    gt2 = xr.open_dataset('...gradtheta2...nc')['gradtheta2'].values
    gs2 = xr.open_dataset('...gradsalt2...nc')['gradsalt2'].values
    gr2 = xr.open_dataset('...gradrho2...nc')['gradrho2'].values
    tu_h = turner_angle(gt2, gs2, gr2)

Per-front example:
    tu_h = turner_angle(
        df['gradtheta2_median'].values,
        df['gradsalt2_median'].values,
        df['gradrho2_median'].values,
    )
    df['tu_h_deg'] = tu_h
"""

import numpy as np



def turner_angle(
    gradtheta2,
    gradsalt2,
    gradrho2,
) -> np.ndarray:
    """Compute the horizontal Turner Angle from pre-computed gradient fields.

    Tu_h = arctan( ∇ρ·(α∇T + β∇S) / ∇ρ·(α∇T − β∇S) )

    Under the linear EOS ∇ρ = ρ₀(−α∇T + β∇S), the dot products reduce to
    expressions involving only squared gradient magnitudes:

        Numerator   = ρ₀(β²|∇S|² − α²|∇T|²)   [cross terms cancel]
        Denominator = −|∇ρ|²/ρ₀                 [always ≤ 0]

    Sign convention:
        Tu > 0    temperature-dominated  (α|∇T| > β|∇S|)
        Tu < 0    salinity-dominated     (β|∇S| > α|∇T|)
        Tu → ±90° compensating T/S gradients, |∇ρ| → 0
        Tu = 0    β²|∇S|² = α²|∇T|²

    Parameters
    ----------
    gradtheta2 : array-like
        |∇T|² — squared potential-temperature gradient (°C² m⁻²).
        May be 1-D (one value per front) or 2-D (full gridded field).
    gradsalt2 : array-like
        |∇S|² — squared salinity gradient (PSU² m⁻²).
    gradrho2 : array-like
        |∇ρ|² — squared density gradient ((kg m⁻³)² m⁻²).

    Returns
    -------
    numpy.ndarray
        Turner angle in degrees, same shape as the inputs.
        Pixels where gradrho2 == 0 (land / fill values) are NaN.
    """
    # ── Default physical constants (linear EOS) ──────────────────────────────
    ALPHA = 2.0e-4   # thermal  expansion coefficient  (°C⁻¹)
    BETA  = 7.4e-4   # haline contraction coefficient (PSU⁻¹)
    RHO0  = 1025.0   # reference density              (kg m⁻³)


    gradtheta2 = np.asarray(gradtheta2, dtype=np.float64)
    gradsalt2  = np.asarray(gradsalt2,  dtype=np.float64)
    gradrho2   = np.asarray(gradrho2,   dtype=np.float64)


    # ── Numerator: ρ₀(β²|∇S|² − α²|∇T|²) ────────────────────────────────────
    # T–S cross terms cancel exactly under the linear EOS.
    numer = RHO0 * (BETA**2 * gradsalt2 - ALPHA**2 * gradtheta2)

    # ── Denominator: −|∇ρ|²/ρ₀ (always ≤ 0) ─────────────────────────────────
    # Explicitly NaN where |∇ρ| = 0 to avoid a zero-division warning; those
    # pixels are already excluded by ocean_mask, but being explicit is safer.
    denom = np.where(gradrho2 > 0, -gradrho2 / RHO0, np.nan)

    # ── Turner angle ──────────────────────────────────────────────────────────
    tu_h = np.degrees(np.arctan(numer / denom))

    return tu_h
