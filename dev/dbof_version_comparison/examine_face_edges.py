from matplotlib import pyplot as plt
import numpy as np
import xarray

from skimage.restoration import inpaint as sk_inpaint
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


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

def inpaint_biharm(gradb2, mask=None):
    print("Inpainting with biharmonic...")
    if mask is None:
        mask = gradb2 == -999.
    tmp = gradb2.copy()
    # Deal with NaN
    isnan = np.isnan(tmp)
    tmp[isnan] = np.median(tmp[~isnan])
    mask = np.uint8(mask)
    # Inpatin
    inpainted = sk_inpaint.inpaint_biharmonic(tmp, mask, channel_axis=None)
    # Fill in the original values
    inpainted[isnan] = np.nan
    #
    return inpainted

def inpaint_regular(gradb2, mask=None):
    """Inpaint bad pixels using RegularGridInterpolator.

    Fills bad pixels by nearest-neighbor first to provide a
    complete grid, then applies linear interpolation to smooth
    the filled values using surrounding valid data.
    """
    print("Inpainting with RegularGridInterpolator...")
    if mask is None:
        mask = gradb2 == -999.

    data = gradb2.copy()

    # Fill bad pixels with nearest valid neighbor so the
    # interpolator has a complete grid to work with
    full_mask = mask | np.isnan(data)
    nn_indices = distance_transform_edt(
        full_mask, return_distances=False, return_indices=True)
    filled = data.copy()
    filled[full_mask] = data[tuple(nn_indices)][full_mask]

    # Build the interpolator on the filled grid
    rows = np.arange(data.shape[0], dtype=float)
    cols = np.arange(data.shape[1], dtype=float)
    interp = RegularGridInterpolator(
        (rows, cols), filled, method='linear',
        bounds_error=False, fill_value=None)

    # Evaluate at bad pixel locations
    bad_rows, bad_cols = np.where(mask)
    if bad_rows.size > 0:
        pts = np.column_stack([bad_rows.astype(float),
                               bad_cols.astype(float)])
        data[mask] = interp(pts)

    return data


def inpaint_RBF(gradb2, mask=None):
    print("Inpainting with RBF...")
    rows, cols = np.indices(gradb2.shape)
    if mask is None:
        mask = gradb2 == -999.
    #valid = ~mask
    valid_pts = np.column_stack([rows[~mask], cols[~mask]])
    bad_pts   = np.column_stack([rows[mask], cols[mask]])
    # Deal with NaN
    #tmp = gradb2.copy()
    #isnan = np.isnan(tmp)
    #tmp[isnan] = np.median(tmp[~isnan])
    # Inpaint
    rbf = RBFInterpolator(valid_pts, gradb2[~mask],
                        kernel='linear',
                        neighbors=32)   # only use 32 nearest valid pixels
    #
    data_filled = gradb2.copy()
    data_filled[mask] = rbf(bad_pts)
    return data_filled

def examine_inpainting():

    gradb2 = load_gradb2()

    # Original
    #print("Plotting original...")
    #plt.imshow(gradb2[7750:9250, 8000:10000], origin='lower')
    #plt.colorbar()
    #plt.show()

    # Inpaint with biharmonic
    gradb2 = inpaint_biharm(gradb2)

    print("Plotting...")
    plt.imshow(np.log10(gradb2[7750:9250, 8000:10000]), origin='lower')#, vmin=1e-16, vmax=1e-14)
    plt.colorbar()
    plt.show()

    # Cut down for memory
    gradb2 = gradb2[7750:9250, 8000:10000]
    print("More inpainting...")
    mask = (gradb2 < 1e-20) & np.isfinite(gradb2)
    print(f"{np.sum(mask)} bad pixels remain")

    # Regular
    gradb2 = inpaint_regular(gradb2, mask)

    # RBF
    #gradb2 = inpaint_RBF(gradb2, mask)

    #print(f"{np.sum(mask)} bad pixels remain")
    #tmp = gradb2.copy()
    #isnan = np.isnan(tmp)
    #tmp[isnan] = np.median(tmp[~isnan])
    #mask = np.uint8(mask)
    #gradb2 = sk_inpaint.inpaint_biharmonic(tmp, mask, channel_axis=None)

    print("Plotting new one...")
    #plt.imshow(np.log10(gradb2[7750:9250, 8000:10000]), origin='lower')#, vmin=1e-16, vmax=1e-14)
    plt.imshow(np.log10(gradb2), origin='lower')#, vmin=1e-16, vmax=1e-14)
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