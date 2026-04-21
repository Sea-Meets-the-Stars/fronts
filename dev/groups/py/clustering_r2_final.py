"""Run 2 final: HDBSCAN on tight UMAP embedding (min_dist=0.01).

Optimal parameters from scan: min_cluster_size=200, min_samples=10.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.preprocessing import StandardScaler

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
embedding = np.load(os.path.join(datadir, 'umap_embedding_r2_tight.npy'))
features = pd.read_parquet(os.path.join(datadir, 'features_r2.parquet'))
feature_cols = json.load(open(os.path.join(datadir, 'feature_cols_r2.json')))
feature_cols = [feature_cols[str(i)] for i in range(len(feature_cols))]

print(f"Clustering {len(embedding)} points (tight UMAP)...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=10,
                             cluster_selection_method='eom')
labels = clusterer.fit_predict(embedding)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Found {n_clusters} clusters, {n_noise} noise ({n_noise/len(labels)*100:.1f}%)")

features['cluster'] = labels
features.to_parquet(os.path.join(datadir, 'features_clustered_r2.parquet'), index=False)
np.save(os.path.join(datadir, 'cluster_labels_r2.npy'), labels)

overleaf = '/home/xavier/Projects/overleaf/Front_properties'

# --- UMAP density (tight) ---
fig, ax = plt.subplots(figsize=(10, 10))
hb = ax.hexbin(embedding[:, 0], embedding[:, 1], gridsize=150, cmap='inferno', mincnt=1)
ax.set_xlabel('UMAP 1', fontsize=13)
ax.set_ylabel('UMAP 2', fontsize=13)
ax.set_title('UMAP Density (Run 2, min_dist=0.01)', fontsize=15)
plt.colorbar(hb, ax=ax, label='Count')
plt.tight_layout()
fig.savefig(os.path.join(overleaf, 'umap_density_r2.png'), dpi=150)
print('Saved umap_density_r2.png')
plt.close()

# --- Cluster map ---
fig, ax = plt.subplots(figsize=(12, 10))
noise_mask = labels == -1
if noise_mask.any():
    ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
               s=0.3, alpha=0.1, c='lightgrey', label='Noise')

unique_labels = sorted(set(labels) - {-1})
cmap = matplotlib.colormaps.get_cmap('tab10')
for i, lab in enumerate(unique_labels):
    mask = labels == lab
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               s=0.5, alpha=0.3, c=[cmap(i % 10)], label=f'C{lab} (n={mask.sum()})')

ax.set_xlabel('UMAP 1', fontsize=13)
ax.set_ylabel('UMAP 2', fontsize=13)
ax.set_title(f'HDBSCAN Clusters (Run 2, {n_clusters} clusters)', fontsize=15)
ax.legend(markerscale=10, fontsize=9, loc='best')
plt.tight_layout()
fig.savefig(os.path.join(overleaf, 'hdbscan_clusters_r2.png'), dpi=150)
print('Saved hdbscan_clusters_r2.png')
plt.close()

# --- Colored by each feature ---
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
plt.suptitle('UMAP colored by features (Run 2, tight)', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(overleaf, 'umap_by_feature_r2.png'), dpi=150, bbox_inches='tight')
print('Saved umap_by_feature_r2.png')
plt.close()

# --- Cluster profiles ---
cluster_profiles = features[features['cluster'] >= 0].groupby('cluster')[feature_cols].median()
print("\nCluster profiles (median values):")
print(cluster_profiles.round(6).to_string())

profile_scaled = pd.DataFrame(
    StandardScaler().fit_transform(cluster_profiles),
    index=cluster_profiles.index, columns=cluster_profiles.columns)

fig, ax = plt.subplots(figsize=(12, max(4, n_clusters * 1.0)))
sns.heatmap(profile_scaled, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
            ax=ax, linewidths=0.5)
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)
ax.set_title('Cluster Profiles (Run 2, standardized medians)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(overleaf, 'cluster_profiles_r2.png'), dpi=150)
print('Saved cluster_profiles_r2.png')
plt.close()

print("\nCluster sizes:")
for lab in sorted(set(labels)):
    name = "Noise" if lab == -1 else f"Cluster {lab}"
    print(f"  {name}: {(labels == lab).sum()}")
print("\nDone")
