"""Run 2: Feature engineering with reduced feature set.

Excluded per instructions: velocities (U,V,W,ug,vg), Eta, Okubo-Weiss.
Excluded by analysis: standalone Coriolis f (pure latitude proxy).

Correlation check showed all 8 features have |r| < 0.31, so no
de-correlation needed beyond log-transform + standardization.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
group_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'group_fronts', 'v1')
data_file = os.path.join(group_path, 'front_properties_20121109T12_00_00_v1_bin_A.parquet')
df = pd.read_parquet(data_file)
print(f"Loaded {len(df)} fronts")

abs_f = np.abs(df['coriolis_f_median']).clip(lower=1e-7)

features = pd.DataFrame({
    'flabel': df['flabel'],
    'npix': df['npix'],
    'strain_over_f': df['strain_mag_median'] / abs_f,
    'divergence_over_f': df['divergence_median'] / abs_f,
    'vorticity_over_f': df['relative_vorticity_median'] / abs_f,
    'log_gradb2': np.log10(df['gradb2_median'].clip(lower=1e-30)),
    'log_strain_mag': np.log10(df['strain_mag_median'].clip(lower=1e-30)),
    'frontogenesis': df['frontogenesis_tendency_median'],
    'Theta': df['Theta_median'],
    'Salt': df['Salt_median'],
})

feature_cols = ['strain_over_f', 'divergence_over_f', 'vorticity_over_f',
                'log_gradb2', 'log_strain_mag', 'frontogenesis',
                'Theta', 'Salt']

features = features.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
print(f"{len(features)} fronts after cleaning (removed {len(df) - len(features)})")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features[feature_cols].values)
print(f"Feature matrix: {X_scaled.shape}")

# Save
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(outdir, exist_ok=True)

features.to_parquet(os.path.join(outdir, 'features_r2.parquet'), index=False)
np.save(os.path.join(outdir, 'X_scaled_r2.npy'), X_scaled)
pd.Series(feature_cols).to_json(os.path.join(outdir, 'feature_cols_r2.json'))
scaler_df = pd.DataFrame({'feature': feature_cols, 'mean': scaler.mean_, 'scale': scaler.scale_})
scaler_df.to_csv(os.path.join(outdir, 'scaler_params_r2.csv'), index=False)
print(f"Saved to {outdir}")
