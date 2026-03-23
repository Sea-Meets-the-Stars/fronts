"""Phase 2: Feature selection and engineering for front property clustering."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
group_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'group_fronts', 'v1')
data_file = os.path.join(group_path, 'front_properties_20121109T12_00_00_v1_bin_A.parquet')
df = pd.read_parquet(data_file)
print(f"Loaded {len(df)} fronts")

# --- Feature selection ---
# Use median values; avoid redundant gradient fields (keep gradb2 as primary).
# Normalize dynamical quantities by |f| where appropriate.
# Drop geostrophic velocities (highly correlated with U, V).
# Drop individual gradient fields except gradb2 (they're all highly correlated).

features = pd.DataFrame({'flabel': df['flabel'], 'npix': df['npix']})

# Dynamical quantities normalized by |f|
abs_f = np.abs(df['coriolis_f_median'])
# Avoid division by zero at equator
abs_f = abs_f.clip(lower=1e-7)

features['strain_over_f'] = df['strain_mag_median'] / abs_f
features['divergence_over_f'] = df['divergence_median'] / abs_f
features['vorticity_over_f'] = df['relative_vorticity_median'] / abs_f  # = Rossby number
features['OW_over_f2'] = df['okubo_weiss_median'] / (abs_f ** 2)

# Log-transformed positive-definite fields
features['log_gradb2'] = np.log10(df['gradb2_median'].clip(lower=1e-30))
features['log_strain_mag'] = np.log10(df['strain_mag_median'].clip(lower=1e-30))

# Frontogenesis (can be negative, use raw)
features['frontogenesis'] = df['frontogenesis_tendency_median']

# Surface state
features['Theta'] = df['Theta_median']
features['Salt'] = df['Salt_median']
features['Eta'] = df['Eta_median']

# Velocity magnitudes
features['speed'] = np.sqrt(df['U_median']**2 + df['V_median']**2)
features['W'] = df['W_median']

# Coriolis (proxy for latitude)
features['coriolis_f'] = df['coriolis_f_median']

# --- Clean up ---
feature_cols = [c for c in features.columns if c not in ('flabel', 'npix')]
features = features.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
print(f"{len(features)} fronts after dropping NaN/Inf (removed {len(df) - len(features)})")

# --- Standardize ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features[feature_cols].values)
print(f"Feature matrix shape: {X_scaled.shape}")
print(f"Features: {feature_cols}")

# --- Save ---
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(outdir, exist_ok=True)

# Save features table
features.to_parquet(os.path.join(outdir, 'features.parquet'), index=False)

# Save scaled matrix and metadata
np.save(os.path.join(outdir, 'X_scaled.npy'), X_scaled)
pd.Series(feature_cols).to_json(os.path.join(outdir, 'feature_cols.json'))

# Save scaler params
scaler_df = pd.DataFrame({'feature': feature_cols,
                           'mean': scaler.mean_,
                           'scale': scaler.scale_})
scaler_df.to_csv(os.path.join(outdir, 'scaler_params.csv'), index=False)

print(f"Saved to {outdir}")
