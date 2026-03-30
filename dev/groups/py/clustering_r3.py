"""Run 3: HDBSCAN clustering + sub-clustering of dominant group.

Uses the with-npix UMAP embedding. Reuses Run 2 parameter tuning
(min_cluster_size=200, min_samples=10) as baseline.
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
overleaf = '/home/xavier/Projects/overleaf/Front_properties'

embedding = np.load(os.path.join(datadir, 'umap_embedding_r3_with_npix.npy'))
features = pd.read_parquet(os.path.join(datadir, 'features_r3_with_npix.parquet'))
feature_cols = json.load(open(os.path.join(datadir, 'feature_cols_r3_with_npix.json')))
feature_cols = [feature_cols[str(i)] for i in range(len(feature_cols))]

# ---- Phase 1: Full clustering ----
print(f"Clustering {len(embedding)} points...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=10,
                             cluster_selection_method='eom')
labels = clusterer.fit_predict(embedding)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Found {n_clusters} clusters, {n_noise} noise ({n_noise/len(labels)*100:.1f}%)")

features['cluster'] = labels

# Cluster sizes
print("\nCluster sizes:")
for lab in sorted(set(labels)):
    name = "Noise" if lab == -1 else f"Cluster {lab}"
    print(f"  {name}: {(labels == lab).sum()}")

# ---- Cluster map ----
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
ax.set_title(f'HDBSCAN Clusters (Run 3, {n_clusters} clusters)', fontsize=15)
ax.legend(markerscale=10, fontsize=8, loc='best', ncol=2)
plt.tight_layout()
fig.savefig(os.path.join(overleaf, 'hdbscan_clusters_r3.png'), dpi=150)
print('Saved hdbscan_clusters_r3.png')
plt.close()

# ---- Cluster profiles ----
cluster_profiles = features[features['cluster'] >= 0].groupby('cluster')[feature_cols].median()
print("\nCluster profiles (median):")
print(cluster_profiles.round(6).to_string())

profile_scaled = pd.DataFrame(
    StandardScaler().fit_transform(cluster_profiles),
    index=cluster_profiles.index, columns=cluster_profiles.columns)

fig, ax = plt.subplots(figsize=(12, max(4, n_clusters * 0.8)))
sns.heatmap(profile_scaled, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
            ax=ax, linewidths=0.5)
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)
ax.set_title('Cluster Profiles (Run 3, standardized medians)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(overleaf, 'cluster_profiles_r3.png'), dpi=150)
print('Saved cluster_profiles_r3.png')
plt.close()

# ---- Correlation matrix ----
corr = features[feature_cols].corr()
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(corr, cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.5, ax=ax, annot=True, fmt='.2f')
ax.set_title('Feature Correlation Matrix (Run 3)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(overleaf, 'correlation_matrix_r3.png'), dpi=150)
print('Saved correlation_matrix_r3.png')
plt.close()

# ---- Phase 2: Sub-clustering the dominant cluster ----
dominant_label = labels[labels >= 0]
dominant_label = pd.Series(dominant_label).value_counts().index[0]
dominant_mask = labels == dominant_label
dominant_frac = dominant_mask.sum() / len(labels) * 100
print(f"\nDominant cluster: {dominant_label} ({dominant_mask.sum()} fronts, {dominant_frac:.1f}%)")

if dominant_frac > 50:
    print("Sub-clustering dominant cluster...")
    import umap as umap_mod

    X_dom = features.loc[dominant_mask, feature_cols].values
    scaler_dom = StandardScaler()
    X_dom_scaled = scaler_dom.fit_transform(X_dom)

    reducer_dom = umap_mod.UMAP(n_neighbors=30, min_dist=0.01, n_components=2,
                                 metric='euclidean', random_state=42)
    emb_dom = reducer_dom.fit_transform(X_dom_scaled)

    # Scan HDBSCAN params for sub-clustering
    best_nc = 0
    best_params = None
    best_labels = None
    for mcs in [200, 500, 1000]:
        for ms in [10, 25, 50]:
            cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                  cluster_selection_method='eom')
            sub_labels = cl.fit_predict(emb_dom)
            nc = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
            nn = (sub_labels == -1).sum()
            noise_pct = nn / len(sub_labels) * 100
            print(f"  mcs={mcs}, ms={ms}: {nc} sub-clusters, {noise_pct:.1f}% noise")
            if nc > best_nc and noise_pct < 20:
                best_nc = nc
                best_params = (mcs, ms)
                best_labels = sub_labels

    if best_labels is not None:
        print(f"\nBest sub-clustering: {best_nc} sub-clusters (mcs={best_params[0]}, ms={best_params[1]})")

        # Sub-cluster map
        fig, ax = plt.subplots(figsize=(12, 10))
        noise_mask_sub = best_labels == -1
        if noise_mask_sub.any():
            ax.scatter(emb_dom[noise_mask_sub, 0], emb_dom[noise_mask_sub, 1],
                       s=0.3, alpha=0.1, c='lightgrey', label='Noise')
        sub_unique = sorted(set(best_labels) - {-1})
        cmap_sub = matplotlib.colormaps.get_cmap('tab20')
        for i, lab in enumerate(sub_unique):
            mask = best_labels == lab
            ax.scatter(emb_dom[mask, 0], emb_dom[mask, 1],
                       s=0.5, alpha=0.3, c=[cmap_sub(i % 20)],
                       label=f'Sub-C{lab} (n={mask.sum()})')
        ax.set_xlabel('UMAP 1', fontsize=13)
        ax.set_ylabel('UMAP 2', fontsize=13)
        ax.set_title(f'Sub-clusters within C{dominant_label} ({best_nc} sub-clusters)', fontsize=15)
        ax.legend(markerscale=10, fontsize=7, loc='best', ncol=3)
        plt.tight_layout()
        fig.savefig(os.path.join(overleaf, 'subclusters_r3.png'), dpi=150)
        print('Saved subclusters_r3.png')
        plt.close()

        # Sub-cluster profiles
        dom_features = features.loc[dominant_mask].copy()
        dom_features['subcluster'] = best_labels
        sub_profiles = dom_features[dom_features['subcluster'] >= 0].groupby('subcluster')[feature_cols].median()
        print("\nSub-cluster profiles (median):")
        print(sub_profiles.round(6).to_string())

        sub_scaled = pd.DataFrame(
            StandardScaler().fit_transform(sub_profiles),
            index=sub_profiles.index, columns=sub_profiles.columns)
        fig, ax = plt.subplots(figsize=(14, max(4, best_nc * 0.6)))
        sns.heatmap(sub_scaled, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
                    ax=ax, linewidths=0.5)
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Sub-cluster', fontsize=12)
        ax.set_title(f'Sub-cluster Profiles within C{dominant_label} (Run 3)', fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(overleaf, 'subcluster_profiles_r3.png'), dpi=150)
        print('Saved subcluster_profiles_r3.png')
        plt.close()

        # Save sub-clustering results
        np.save(os.path.join(datadir, 'subcluster_labels_r3.npy'), best_labels)
        np.save(os.path.join(datadir, 'subcluster_umap_r3.npy'), emb_dom)

# Save final results
features.to_parquet(os.path.join(datadir, 'features_clustered_r3.parquet'), index=False)
np.save(os.path.join(datadir, 'cluster_labels_r3.npy'), labels)
print("\nDone")
