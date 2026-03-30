"""Run 3: Feature engineering with gradient decomposition.

Changes from Run 2:
- Removed: log_strain_mag (redundant with strain_over_f), Theta (latitude proxy)
- Added: gradsalt2/gradtheta2 ratio (haline vs thermal front indicator)
- Added: log(npix) (front length; conditional on not dominating UMAP)
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

# Gradient ratio: haline / thermal
# Clip to avoid division by zero
gradtheta2 = df['gradtheta2_median'].clip(lower=1e-30)
grad_ratio = df['gradsalt2_median'] / gradtheta2

features = pd.DataFrame({
    'flabel': df['flabel'],
    'npix': df['npix'],
    'strain_over_f': df['strain_mag_median'] / abs_f,
    'divergence_over_f': df['divergence_median'] / abs_f,
    'vorticity_over_f': df['relative_vorticity_median'] / abs_f,
    'log_gradb2': np.log10(df['gradb2_median'].clip(lower=1e-30)),
    'frontogenesis': df['frontogenesis_tendency_median'],
    'Salt': df['Salt_median'],
    'log_grad_ratio': np.log10(grad_ratio.clip(lower=1e-30)),
    'log_npix': np.log10(df['npix'].clip(lower=1)),
})

# Feature columns: include log_npix for initial assessment
feature_cols_with_npix = ['strain_over_f', 'divergence_over_f', 'vorticity_over_f',
                          'log_gradb2', 'frontogenesis', 'Salt',
                          'log_grad_ratio', 'log_npix']

feature_cols_no_npix = ['strain_over_f', 'divergence_over_f', 'vorticity_over_f',
                        'log_gradb2', 'frontogenesis', 'Salt',
                        'log_grad_ratio']

features = features.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols_with_npix)
print(f"{len(features)} fronts after cleaning (removed {len(df) - len(features)})")

# Save both versions
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(outdir, exist_ok=True)

for suffix, cols in [('r3_with_npix', feature_cols_with_npix),
                     ('r3', feature_cols_no_npix)]:
    scaler = StandardScaler()
    X = scaler.fit_transform(features[cols].values)
    features.to_parquet(os.path.join(outdir, f'features_{suffix}.parquet'), index=False)
    np.save(os.path.join(outdir, f'X_scaled_{suffix}.npy'), X)
    pd.Series(cols).to_json(os.path.join(outdir, f'feature_cols_{suffix}.json'))
    print(f"  {suffix}: {X.shape} features={cols}")

print(f"Saved to {outdir}")
