"""Phase 4: HDBSCAN clustering on UMAP embedding."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hdbscan

# Load
datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
embedding = np.load(os.path.join(datadir, 'umap_embedding.npy'))
features = pd.read_parquet(os.path.join(datadir, 'features.parquet'))
feature_cols = json.load(open(os.path.join(datadir, 'feature_cols.json')))
feature_cols = [feature_cols[str(i)] for i in range(len(feature_cols))]

print(f"Clustering {len(embedding)} points...")

# HDBSCAN on UMAP embedding
clusterer = hdbscan.HDBSCAN(min_cluster_size=500, min_samples=50,
                             cluster_selection_method='eom')
labels = clusterer.fit_predict(embedding)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")

# Save labels
features['cluster'] = labels
features.to_parquet(os.path.join(datadir, 'features_clustered.parquet'), index=False)
np.save(os.path.join(datadir, 'cluster_labels.npy'), labels)

# --- Figure 1: Clusters on UMAP ---
fig, ax = plt.subplots(figsize=(12, 10))

# Plot noise in grey
noise_mask = labels == -1
ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
           s=0.3, alpha=0.1, c='lightgrey', label='Noise')

# Plot clusters
unique_labels = sorted(set(labels) - {-1})
cmap = plt.cm.get_cmap('tab20', len(unique_labels))
for i, lab in enumerate(unique_labels):
    mask = labels == lab
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               s=0.5, alpha=0.3, c=[cmap(i)], label=f'Cluster {lab} (n={mask.sum()})')

ax.set_xlabel('UMAP 1', fontsize=13)
ax.set_ylabel('UMAP 2', fontsize=13)
ax.set_title(f'HDBSCAN Clusters ({n_clusters} clusters)', fontsize=15)
ax.legend(markerscale=10, fontsize=9, loc='best', ncol=2)
plt.tight_layout()
outfile = '/home/xavier/Projects/overleaf/Front_properties/hdbscan_clusters.png'
fig.savefig(outfile, dpi=150)
print(f'Saved {outfile}')
plt.close()

# --- Figure 2: Cluster profiles (median properties per cluster) ---
cluster_profiles = features[features['cluster'] >= 0].groupby('cluster')[feature_cols].median()
print("\nCluster profiles (median values):")
print(cluster_profiles.to_string())

# Normalize profiles for heatmap visualization
from sklearn.preprocessing import StandardScaler
profile_scaled = pd.DataFrame(
    StandardScaler().fit_transform(cluster_profiles),
    index=cluster_profiles.index,
    columns=cluster_profiles.columns
)

import seaborn as sns
fig, ax = plt.subplots(figsize=(14, max(6, n_clusters * 0.8)))
sns.heatmap(profile_scaled, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
            ax=ax, linewidths=0.5, xticklabels=True, yticklabels=True)
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)
ax.set_title('Cluster Profiles (standardized median values)', fontsize=14)
plt.tight_layout()
outfile = '/home/xavier/Projects/overleaf/Front_properties/cluster_profiles.png'
fig.savefig(outfile, dpi=150)
print(f'Saved {outfile}')
plt.close()

# --- Print cluster sizes ---
print("\nCluster sizes:")
for lab in sorted(set(labels)):
    name = "Noise" if lab == -1 else f"Cluster {lab}"
    print(f"  {name}: {(labels == lab).sum()}")

print("\nClustering done")
