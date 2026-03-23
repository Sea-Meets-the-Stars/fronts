"""Generate distribution histograms for key median front properties."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
data_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'group_fronts', 'v1', 'front_properties_20121109T12_00_00_v1_bin_A.parquet')
df = pd.read_parquet(data_file)

# Key median properties to plot
key_cols = {
    'divergence_median': r'Divergence (s$^{-1}$)',
    'vorticity_median': r'Vorticity (s$^{-1}$)',
    'strain_mag_median': r'Strain magnitude (s$^{-1}$)',
    'gradb2_median': r'$|\nabla b|^2$ (s$^{-4}$)',
    'frontogenesis_tendency_median': r'Frontogenesis tendency',
    'OW_median': 'Okubo-Weiss',
    'SSTK_median': 'SST (K)',
    'SSH_median': 'SSH (m)',
    'W_median': r'W (m s$^{-1}$)',
    'coriolis_f_median': 'Coriolis f',
    'U_median': r'U (m s$^{-1}$)',
    'V_median': r'V (m s$^{-1}$)',
}

# Only use columns that exist
plot_cols = {k: v for k, v in key_cols.items() if k in df.columns}

ncols = 3
nrows = (len(plot_cols) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
axes = axes.flatten()

for i, (col, label) in enumerate(plot_cols.items()):
    ax = axes[i]
    vals = df[col].dropna().values
    # Clip to 1st-99th percentile for better visualization
    vmin, vmax = np.percentile(vals, [1, 99])
    clipped = vals[(vals >= vmin) & (vals <= vmax)]
    ax.hist(clipped, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(col.replace('_median', ''), fontsize=12)

# Hide extra axes
for j in range(len(plot_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distributions of Median Front Properties (1st-99th percentile)', fontsize=14, y=1.01)
plt.tight_layout()

outfile = '/home/xavier/Projects/overleaf/Front_properties/distributions.png'
fig.savefig(outfile, dpi=150, bbox_inches='tight')
print(f'Saved to {outfile}')
plt.close()
