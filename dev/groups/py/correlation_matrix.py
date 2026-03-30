"""Generate correlation matrix heatmap for median front properties."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'group_fronts', 'v1', 'front_properties_20121109T12_00_00_v1_bin_A.parquet')
df = pd.read_parquet(data_file)

# Select median columns + npix
median_cols = [c for c in df.columns if c.endswith('_median')]
subset = df[median_cols].copy()

# Shorten column names for display
subset.columns = [c.replace('_median', '') for c in subset.columns]

# Compute correlation
corr = subset.corr()

# Plot
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(corr, cmap='RdBu_r', vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5, ax=ax,
            xticklabels=True, yticklabels=True)
ax.set_title('Correlation Matrix of Median Front Properties', fontsize=16)
ax.tick_params(axis='both', labelsize=8)
plt.tight_layout()

outfile = '/home/xavier/Projects/overleaf/Front_properties/correlation_matrix.png'
fig.savefig(outfile, dpi=150)
print(f'Saved to {outfile}')
plt.close()
