"""Phase 3a: PCA analysis of front properties."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load
datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
X_scaled = np.load(os.path.join(datadir, 'X_scaled.npy'))
feature_cols = json.load(open(os.path.join(datadir, 'feature_cols.json')))
# feature_cols is a dict with string keys
feature_cols = [feature_cols[str(i)] for i in range(len(feature_cols))]

from sklearn.decomposition import PCA

# Full PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratios:")
cumvar = 0
for i, (ev, cev) in enumerate(zip(pca.explained_variance_ratio_,
                                    np.cumsum(pca.explained_variance_ratio_))):
    print(f"  PC{i+1}: {ev:.4f}  (cumulative: {cev:.4f})")

# --- Figure 1: Scree plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_, color='steelblue')
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
ax1.set_title('PCA Scree Plot', fontsize=14)

ax2.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), 'o-', color='steelblue')
ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90%')
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax2.set_title('Cumulative Variance', fontsize=14)
ax2.legend()

plt.tight_layout()
outfile = '/home/xavier/Projects/overleaf/Front_properties/pca_scree.png'
fig.savefig(outfile, dpi=150)
print(f'Saved {outfile}')
plt.close()

# --- Figure 2: Loadings heatmap ---
n_show = min(8, len(feature_cols))
loadings = pd.DataFrame(pca.components_[:n_show].T,
                         index=feature_cols,
                         columns=[f'PC{i+1}' for i in range(n_show)])

fig, ax = plt.subplots(figsize=(10, 8))
import seaborn as sns
sns.heatmap(loadings, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
            ax=ax, linewidths=0.5)
ax.set_title('PCA Loadings', fontsize=14)
plt.tight_layout()
outfile = '/home/xavier/Projects/overleaf/Front_properties/pca_loadings.png'
fig.savefig(outfile, dpi=150)
print(f'Saved {outfile}')
plt.close()

# Save PCA results
np.save(os.path.join(datadir, 'X_pca.npy'), X_pca)
np.save(os.path.join(datadir, 'pca_components.npy'), pca.components_)
np.save(os.path.join(datadir, 'pca_explained_variance.npy'), pca.explained_variance_ratio_)
print("PCA done")
