"""Run 2: PCA analysis with reduced feature set."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
X_scaled = np.load(os.path.join(datadir, 'X_scaled_r2.npy'))
feature_cols = json.load(open(os.path.join(datadir, 'feature_cols_r2.json')))
feature_cols = [feature_cols[str(i)] for i in range(len(feature_cols))]

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratios:")
for i, (ev, cev) in enumerate(zip(pca.explained_variance_ratio_,
                                    np.cumsum(pca.explained_variance_ratio_))):
    print(f"  PC{i+1}: {ev:.4f}  (cumulative: {cev:.4f})")

# Scree plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_, color='steelblue')
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
ax1.set_title('PCA Scree Plot (Run 2)', fontsize=14)

ax2.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), 'o-', color='steelblue')
ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90%')
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax2.set_title('Cumulative Variance (Run 2)', fontsize=14)
ax2.legend()
plt.tight_layout()
fig.savefig('/home/xavier/Projects/overleaf/Front_properties/pca_scree_r2.png', dpi=150)
print('Saved pca_scree_r2.png')
plt.close()

# Loadings
loadings = pd.DataFrame(pca.components_.T, index=feature_cols,
                         columns=[f'PC{i+1}' for i in range(len(feature_cols))])
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(loadings, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
            ax=ax, linewidths=0.5)
ax.set_title('PCA Loadings (Run 2)', fontsize=14)
plt.tight_layout()
fig.savefig('/home/xavier/Projects/overleaf/Front_properties/pca_loadings_r2.png', dpi=150)
print('Saved pca_loadings_r2.png')
plt.close()

np.save(os.path.join(datadir, 'X_pca_r2.npy'), X_pca)
np.save(os.path.join(datadir, 'pca_explained_variance_r2.npy'), pca.explained_variance_ratio_)
print("PCA done")
