"""Run 3: UMAP without npix for comparison."""
import os
import numpy as np
import umap

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

X_no = np.load(os.path.join(datadir, 'X_scaled_r3.npy'))
print(f'Running UMAP without npix ({X_no.shape})...')
reducer = umap.UMAP(n_neighbors=30, min_dist=0.01, n_components=2,
                     metric='euclidean', random_state=42)
emb_no = reducer.fit_transform(X_no)
np.save(os.path.join(datadir, 'umap_embedding_r3.npy'), emb_no)
print('Done')
