"""Run 2: HDBSCAN clustering on reduced-feature UMAP."""
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
embedding = np.load(os.path.join(datadir, 'umap_embedding_r2.npy'))
features = pd.read_parquet(os.path.join(datadir, 'features_r2.parquet'))
feature_cols = json.load(open(os.path.join(datadir, 'feature_cols_r2.json')))
feature_cols = [feature_cols[str(i)] for i in range(len(feature_cols))]

print(f"Clustering {len(embedding)} points...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=500, min_samples=50,
                             cluster_selection_method='eom')
labels = clusterer.fit_predict(embedding)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")

features['cluster'] = labels
features.to_parquet(os.path.join(datadir, 'features_clustered_r2.parquet'), index=False)
np.save(os.path.join(datadir, 'cluster_labels_r2.npy'), labels)

# Cluster map
fig, ax = plt.subplots(figsize=(12, 10))
noise_mask = labels == -1
ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
           s=0.3, alpha=0.1, c='lightgrey', label='Noise')

unique_labels = sorted(set(labels) - {-1})
cmap = matplotlib.colormaps.get_cmap('tab20')
for i, lab in enumerate(unique_labels):
    mask = labels == lab
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               s=0.5, alpha=0.3, c=[cmap(i % 20)], label=f'C{lab} (n={mask.sum()})')

ax.set_xlabel('UMAP 1', fontsize=13)
ax.set_ylabel('UMAP 2', fontsize=13)
ax.set_title(f'HDBSCAN Clusters (Run 2, {n_clusters} clusters)', fontsize=15)
ax.legend(markerscale=10, fontsize=8, loc='best', ncol=2)
plt.tight_layout()
fig.savefig('/home/xavier/Projects/overleaf/Front_properties/hdbscan_clusters_r2.png', dpi=150)
print('Saved hdbscan_clusters_r2.png')
plt.close()

# Cluster profiles heatmap
cluster_profiles = features[features['cluster'] >= 0].groupby('cluster')[feature_cols].median()
print("\nCluster profiles (median values):")
print(cluster_profiles.round(6).to_string())

profile_scaled = pd.DataFrame(
    StandardScaler().fit_transform(cluster_profiles),
    index=cluster_profiles.index, columns=cluster_profiles.columns)

fig, ax = plt.subplots(figsize=(12, max(6, n_clusters * 0.8)))
sns.heatmap(profile_scaled, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
            ax=ax, linewidths=0.5)
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)
ax.set_title('Cluster Profiles (Run 2, standardized medians)', fontsize=14)
plt.tight_layout()
fig.savefig('/home/xavier/Projects/overleaf/Front_properties/cluster_profiles_r2.png', dpi=150)
print('Saved cluster_profiles_r2.png')
plt.close()

# Also generate a correlation matrix for the 8 features
corr = features[feature_cols].corr()
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(corr, cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.5, ax=ax, annot=True, fmt='.2f')
ax.set_title('Feature Correlation Matrix (Run 2)', fontsize=14)
plt.tight_layout()
fig.savefig('/home/xavier/Projects/overleaf/Front_properties/correlation_matrix_r2.png', dpi=150)
print('Saved correlation_matrix_r2.png')
plt.close()

print("\nCluster sizes:")
for lab in sorted(set(labels)):
    name = "Noise" if lab == -1 else f"Cluster {lab}"
    print(f"  {name}: {(labels == lab).sum()}")

print("\nClustering done")
