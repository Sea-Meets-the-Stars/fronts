"""Run 4: UMAP + HDBSCAN on C5 from Run 3, dropping Salt and log_grad_ratio.

Remaining features (6, purely dynamical + geometry):
  strain_over_f, divergence_over_f, vorticity_over_f,
  log_gradb2, frontogenesis, log_npix
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
overleaf = '/home/xavier/Projects/overleaf/Front_properties'

# Load Run 3 clustered data
features = pd.read_parquet(os.path.join(datadir, 'features_clustered_r3.parquet'))
labels_r3 = np.load(os.path.join(datadir, 'cluster_labels_r3.npy'))

# Select C5
c5_mask = labels_r3 == 5
c5 = features.loc[c5_mask].copy().reset_index(drop=True)
print(f"C5: {len(c5)} fronts")

# Dynamical features only (drop Salt, log_grad_ratio)
feat_cols = ['strain_over_f', 'divergence_over_f', 'vorticity_over_f',
             'log_gradb2', 'frontogenesis', 'log_npix']

scaler = StandardScaler()
X = scaler.fit_transform(c5[feat_cols].values)
print(f"Feature matrix: {X.shape}")

# UMAP
print("Running UMAP on C5...")
reducer = umap.UMAP(n_neighbors=30, min_dist=0.01, n_components=2,
                     metric='euclidean', random_state=42)
emb = reducer.fit_transform(X)
np.save(os.path.join(datadir, 'umap_embedding_r4.npy'), emb)

# --- Density plot ---
fig, ax = plt.subplots(figsize=(10, 10))
hb = ax.hexbin(emb[:, 0], emb[:, 1], gridsize=150, cmap='inferno', mincnt=1)
ax.set_xlabel('UMAP 1', fontsize=13)
ax.set_ylabel('UMAP 2', fontsize=13)
ax.set_title('UMAP Density of C5 (Run 4, dynamical features only)', fontsize=14)
plt.colorbar(hb, ax=ax, label='Count')
plt.tight_layout()
fig.savefig(os.path.join(overleaf, 'umap_density_r4.png'), dpi=150)
print('Saved umap_density_r4.png')
plt.close()

# --- Color by features ---
ncols_plot = 3
nrows_plot = 2
fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(20, 12))
axes = axes.flatten()
for i, col in enumerate(feat_cols):
    ax = axes[i]
    vals = c5[col].values
    vmin, vmax = np.nanpercentile(vals, [5, 95])
    sc = ax.scatter(emb[:, 0], emb[:, 1], s=0.3, alpha=0.2,
                    c=vals, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(col, fontsize=11)
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    plt.colorbar(sc, ax=ax, shrink=0.8)
plt.suptitle('C5 UMAP colored by dynamical features (Run 4)', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(overleaf, 'umap_by_feature_r4.png'), dpi=150, bbox_inches='tight')
print('Saved umap_by_feature_r4.png')
plt.close()

# --- HDBSCAN parameter scan ---
print("\nHDBSCAN parameter scan:")
best_nc = 0
best_params = None
best_labels = None
for mcs in [200, 500, 1000, 2000]:
    for ms in [10, 25, 50]:
        cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                              cluster_selection_method='eom')
        sub_labels = cl.fit_predict(emb)
        nc = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
        nn = (sub_labels == -1).sum()
        noise_pct = nn / len(sub_labels) * 100
        print(f"  mcs={mcs:5d}, ms={ms:3d}: {nc:3d} clusters, {noise_pct:5.1f}% noise")
        if nc > best_nc and noise_pct < 20:
            best_nc = nc
            best_params = (mcs, ms)
            best_labels = sub_labels

print(f"\nBest: {best_nc} clusters (mcs={best_params[0]}, ms={best_params[1]})")

# --- Cluster map ---
fig, ax = plt.subplots(figsize=(12, 10))
noise_mask = best_labels == -1
if noise_mask.any():
    ax.scatter(emb[noise_mask, 0], emb[noise_mask, 1],
               s=0.3, alpha=0.1, c='lightgrey', label='Noise')
unique_labels = sorted(set(best_labels) - {-1})
cmap = matplotlib.colormaps.get_cmap('tab20')
for i, lab in enumerate(unique_labels):
    mask = best_labels == lab
    ax.scatter(emb[mask, 0], emb[mask, 1],
               s=0.5, alpha=0.3, c=[cmap(i % 20)],
               label=f'C5.{lab} (n={mask.sum()})')
ax.set_xlabel('UMAP 1', fontsize=13)
ax.set_ylabel('UMAP 2', fontsize=13)
ax.set_title(f'Sub-clusters within C5 (Run 4, {best_nc} clusters)', fontsize=15)
ax.legend(markerscale=10, fontsize=7, loc='best', ncol=3)
plt.tight_layout()
fig.savefig(os.path.join(overleaf, 'hdbscan_clusters_r4.png'), dpi=150)
print('Saved hdbscan_clusters_r4.png')
plt.close()

# --- Cluster profiles ---
c5['subcluster'] = best_labels
sub_profiles = c5[c5['subcluster'] >= 0].groupby('subcluster')[feat_cols].median()
print("\nSub-cluster profiles (median):")
print(sub_profiles.round(6).to_string())

# Also show Salt and log_grad_ratio (excluded from clustering but informative)
extra_cols = ['Salt', 'log_grad_ratio']
sub_extra = c5[c5['subcluster'] >= 0].groupby('subcluster')[extra_cols].median()
print("\nExcluded features by sub-cluster (for interpretation):")
print(sub_extra.round(4).to_string())

# Standardized heatmap of clustering features
sub_scaled = pd.DataFrame(
    StandardScaler().fit_transform(sub_profiles),
    index=sub_profiles.index, columns=sub_profiles.columns)
fig, ax = plt.subplots(figsize=(12, max(4, best_nc * 0.6)))
sns.heatmap(sub_scaled, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
            ax=ax, linewidths=0.5)
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Sub-cluster', fontsize=12)
ax.set_title('Sub-cluster Profiles within C5 (Run 4, dynamical only)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(overleaf, 'subcluster_profiles_r4.png'), dpi=150)
print('Saved subcluster_profiles_r4.png')
plt.close()

# Cluster sizes
print("\nCluster sizes:")
for lab in sorted(set(best_labels)):
    name = "Noise" if lab == -1 else f"C5.{lab}"
    print(f"  {name}: {(best_labels == lab).sum()}")

# Save
np.save(os.path.join(datadir, 'subcluster_labels_r4.npy'), best_labels)
np.save(os.path.join(datadir, 'subcluster_umap_r4.npy'), emb)
c5.to_parquet(os.path.join(datadir, 'c5_clustered_r4.parquet'), index=False)
print("\nDone")
