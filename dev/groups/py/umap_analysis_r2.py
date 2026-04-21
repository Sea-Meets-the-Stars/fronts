"""Run 2: UMAP with reduced feature set."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
X_scaled = np.load(os.path.join(datadir, 'X_scaled_r2.npy'))
features = pd.read_parquet(os.path.join(datadir, 'features_r2.parquet'))
feature_cols = json.load(open(os.path.join(datadir, 'feature_cols_r2.json')))
feature_cols = [feature_cols[str(i)] for i in range(len(feature_cols))]

print(f"Running UMAP on {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features...")
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                     metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X_scaled)
print(f"UMAP embedding shape: {embedding.shape}")

np.save(os.path.join(datadir, 'umap_embedding_r2.npy'), embedding)

# Scatter
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(embedding[:, 0], embedding[:, 1], s=0.3, alpha=0.2, c='k')
ax.set_xlabel('UMAP 1', fontsize=13)
ax.set_ylabel('UMAP 2', fontsize=13)
ax.set_title('UMAP of Front Properties (Run 2, 8 features)', fontsize=15)
plt.tight_layout()
fig.savefig('/home/xavier/Projects/overleaf/Front_properties/umap_scatter_r2.png', dpi=150)
print('Saved umap_scatter_r2.png')
plt.close()

# Density
fig, ax = plt.subplots(figsize=(10, 10))
hb = ax.hexbin(embedding[:, 0], embedding[:, 1], gridsize=120, cmap='inferno', mincnt=1)
ax.set_xlabel('UMAP 1', fontsize=13)
ax.set_ylabel('UMAP 2', fontsize=13)
ax.set_title('UMAP Density (Run 2)', fontsize=15)
plt.colorbar(hb, ax=ax, label='Count')
plt.tight_layout()
fig.savefig('/home/xavier/Projects/overleaf/Front_properties/umap_density_r2.png', dpi=150)
print('Saved umap_density_r2.png')
plt.close()

# Colored by features
ncols = 3
nrows = (len(feature_cols) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    ax = axes[i]
    vals = features[col].values
    vmin, vmax = np.nanpercentile(vals, [5, 95])
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], s=0.3, alpha=0.2,
                    c=vals, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xlabel('UMAP 1', fontsize=10)
    ax.set_ylabel('UMAP 2', fontsize=10)
    ax.set_title(col, fontsize=11)
    plt.colorbar(sc, ax=ax, shrink=0.8)
for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('UMAP colored by input features (Run 2)', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig('/home/xavier/Projects/overleaf/Front_properties/umap_by_feature_r2.png', dpi=150, bbox_inches='tight')
print('Saved umap_by_feature_r2.png')
plt.close()
print("UMAP done")
