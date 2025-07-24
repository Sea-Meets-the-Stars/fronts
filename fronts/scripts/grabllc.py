""" Script to grab/plot data from LLC """

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Grab/plot LLC data')
    parser.add_argument("table_file", type=str, help="File+path to Preproc file")
    parser.add_argument("UID", type=int, help="Evaluation table filename (.csv)")
    parser.add_argument("--fields", type=str, default="SST,SSS,Divb2", help="Evaluation table filename (.csv)")
    parser.add_argument("--fig_file", type=str, help="Filename for figure")
    parser.add_argument("--data_file", type=str, help="Filename for figure")
    parser.add_argument("--field_size", type=int, default=64, help="Filename for figure")
    parser.add_argument("--fixed_km", type=float, default=144., help="Filename for figure")
    parser.add_argument("-s","--show", default=False, action="store_true", help="Show pre-processed image?")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec

    from skimage.transform import resize_local_mean

    import pandas

    from fronts.llc import io as llc_io
    from fronts.llc import extract as llc_extract
    from fronts.plotting import images as pimages

    # Load the table
    print("Loading the table..")
    df = pandas.read_parquet(pargs.table_file)
    df_idx = np.where(df.UID == pargs.UID)[0][0]
    idf = df.iloc[df_idx]

    # Fields
    fields = pargs.fields.split(",")
    nfields = len(fields)

    # Grab the data
    udate = idf.datetime
    filename = llc_io.grab_llc_datafile(udate, local=True)
    print(f"Loading the LLC data from {filename}")
    ds = llc_io.load_llc_ds(filename, local=True)

    # Deal with fixed
    print("Loading the coords...")
    coords_ds = llc_io.load_coords()
    R_earth = 6371. # km
    circum = 2 * np.pi* R_earth
    km_deg = circum / 360.
    dlat_km = (coords_ds.lat.data[idf.row+1,idf.col]-
               coords_ds.lat.data[idf.row,idf.col]) * km_deg

    print("Loading the images...")
    images = []
    for field in fields:
        full = llc_extract.field_from_ds(ds, field)
        # Cut out
        dr = int(np.round(pargs.fixed_km / dlat_km))
        dc = dr
        cutout = full[idf.row:idf.row+dr, idf.col:idf.col+dc]
        cutout = resize_local_mean(cutout, (pargs.field_size, pargs.field_size))
        # Resize
        images.append(cutout)
        
    # Plot?
    if pargs.show or pargs.fig_file is not None:
        # Plot
        fig = plt.figure(figsize=(10, 3))
        #pal, cm = plotting.load_palette()
        plt.clf()
        gs = gridspec.GridSpec(1, nfields)

        for ss, image in enumerate(images):
            ax = plt.subplot(gs[ss])
            pimages.show_image(image, ax=ax, clbl=fields[ss])
            

        plt.tight_layout()

        if pargs.fig_file is not None:
            plt.savefig(pargs.fig_file, dpi=300)
            print(f'Wrote: {pargs.fig_file}')
        if pargs.show:
            plt.show()

    # Save?
    if pargs.data_file is not None:
        idict = {}
        for field in fields:
            idict[field] = images.pop(0)
        np.savez(pargs.data_file, **idict)
        print(f'Wrote: {pargs.data_file}')