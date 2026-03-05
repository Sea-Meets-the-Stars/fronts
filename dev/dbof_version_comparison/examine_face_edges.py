from matplotlib import pyplot as plt
import numpy as np
import xarray

from skimage.restoration import inpaint as sk_inpaint
from scipy.interpolate import RBFInterpolator


from fronts.llc import io as llc_io

from IPython import embed

def view_face4():
    d = np.load('face4.npy')

    # Set the edges to 1e-50
    d[0, :] = 1e-50
    d[:, 0] = 1e-50

    plt.imshow(np.log10(d), origin='lower')
    plt.colorbar()
    plt.show()

def load_gradb2():
    # Before
    timestamp = '2012-11-09T12_00_00'
    version = '1'

    print("Loading")
    gradb2_file = llc_io.derived_filename(timestamp, 'gradb2', version=version)
    gradb2 = xarray.open_dataset(gradb2_file)['gradb2'].values
    return gradb2

def examine_inpainting():

    gradb2 = load_gradb2()
    orig = gradb2.copy()

    # Original
    plt.imshow(gradb2[7750:9250, 8000:10000], origin='lower')
    plt.colorbar()
    plt.show()

    # After
    mask = gradb2 == -999.
    valid = ~mask

    print("Inpainting...")
    tmp = gradb2.copy()
    isnan = np.isnan(tmp)
    tmp[isnan] = np.median(tmp[~isnan])
    mask = np.uint8(mask)
    gradb2 = sk_inpaint.inpaint_biharmonic(tmp, mask, channel_axis=None)
    gradb2[isnan] = np.nan

    print("Plotting...")
    plt.imshow(np.log10(gradb2[7750:9250, 8000:10000]), origin='lower')#, vmin=1e-16, vmax=1e-14)
    plt.colorbar()
    plt.show()

    # One more round
    print("More inpainting...")
    mask = (gradb2 < 1e-20) & np.isfinite(gradb2)
    print(f"{np.sum(mask)} bad pixels remain")
    tmp = gradb2.copy()
    isnan = np.isnan(tmp)
    tmp[isnan] = np.median(tmp[~isnan])
    mask = np.uint8(mask)
    gradb2 = sk_inpaint.inpaint_biharmonic(tmp, mask, channel_axis=None)

    print("Plotting...")
    plt.imshow(np.log10(gradb2[7750:9250, 8000:10000]), origin='lower')#, vmin=1e-16, vmax=1e-14)
    plt.colorbar()
    plt.show()

    embed(header='52 of examine_inpainting.py')
    #plt.imshow(np.log10(gradb2), origin='lower')#, vmin=1e-16, vmax=1e-14)
    #plt.colorbar()
    #plt.show()

def examine_RBF():
    gradb2 = load_gradb2()

    # After
    mask = gradb2 == -999.

    # RBF
    rows, cols = np.indices(gradb2.shape)
    valid = ~mask
    valid_pts = np.column_stack([rows[valid], cols[valid]])
    bad_pts   = np.column_stack([rows[mask], cols[mask]])

    rbf = RBFInterpolator(valid_pts, gradb2[valid],
                        kernel='linear',
                        neighbors=32)   # only use 32 nearest valid pixels
    data_filled = gradb2.copy()
    data_filled[mask] = rbf(bad_pts)

    print("Plotting...")
    plt.imshow(np.log10(data_filled[7750:9250, 8000:10000]), origin='lower')#, vmin=1e-16, vmax=1e-14)
    plt.colorbar()
    plt.show()

# Command line
if __name__ == '__main__':
    # view_face4()
    examine_inpainting()
    #examine_RBF()