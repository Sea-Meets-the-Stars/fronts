""" Module to generate a prototype for the SST-SSH U-Net with
LLC4320 data """

from importlib import reload

import os
import numpy as np

import pandas
import xarray
import json
import h5py

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from fronts.llc import table
from fronts.llc import extract
from fronts.preproc import process
from fronts.tables import catalog
from fronts import io as fronts_io
from fronts.po import fronts
from fronts.plotting import images
from fronts.train import tables as train_tables

from IPython import embed

local_out_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts')
super_tbl_file = os.path.join(local_out_path, 'LLC4320_SST144_SSS40_super.parquet')
super_preproc_file = os.path.join(local_out_path, 'LLC4320_SST144_SSS40_super.h5')

def generate_super_table(debug=False, resol=0.5, plot=False,
               minmax_lat=(-90.,57.), field_size=(64,64),
               local:bool=True):
    """ Get the show started by sampling uniformly
    in space and and time

    Args:
        tbl_file (str): _description_
        debug (bool, optional): _description_. Defaults to True.
        resol (float, optional): 
            Typical separation of images in deg
        plot (bool, optional): Plot the spatial distribution?
        max_lat (float, optional): Restrict on latitude
    """

    if debug:
        tbl_file = os.path.join(local_out_path, 'LLC4320_SST144_SSS40_super_test.parquet')
    else:
        tbl_file = super_tbl_file

    # Begin 
    llc_table = table.uniform_coords(resol=resol, minmax_lat=minmax_lat,
                               field_size=field_size)#, outfile=tbl_file)
    # Plot
    if plot:
        table.plot_extraction(llc_table, s=1, resol=resol)

    # Temporal sampling
    if debug:
        # Extract 6 days across the full range;  ends of months
        dti = pandas.date_range('2011-09-13', periods=6, freq='2M')
    else:
        # Extract 24 days across the full range;  ends of months; every 2 weeks
        dti = pandas.date_range('2011-09-13', periods=24, freq='2W')

    # Add days
    llc_table = table.add_days(llc_table, dti, outfile=tbl_file, to_s3=False)

    # Add UIDs
    _ = table.add_uid(llc_table, outfile=tbl_file, to_s3=False)

    print(f"Wrote: {tbl_file} with {len(llc_table)} unique cutouts.")
    print("All done with init")

    if debug:
        embed(header='71 of llc4320_sst_ssh_proto.py')

def preproc_super(extract_file:str, debug:bool=False):

    outfile = super_preproc_file 

    # Load the table
    llc_table = pandas.read_parquet(super_tbl_file)

    # Debug?
    if debug:
        llc_table = llc_table.iloc[:100].copy()
        outfile = os.path.join(local_out_path, 'LLC4320_SST144_SSS40_super_test.h5')

    # Extract dict
    # Load JSON
    #with open(preproc_file, 'r') as infile:
    #    preproc_dict = json.load(infile)


    #extract_dict = {'fields': ['SSS'],
    #extract_dict = {'fields': ['Divb2'],
    extract_dict = {'fields': ['SST','SSS','Divb2'],
             'field_size': 64,
             'pdicts': 
                 {
                     'SST': 
                        {
                        'fixed_km': 144.,
                        'field_size': 64,
                        "quality_thresh": 2,
                        "nrepeat": 1,
                        "downscale": False,
                        "inpaint": False,
                        "median": False,
                        "only_inpaint": False
                        }
                    ,
                     'SSS': 
                        {
                        'fixed_km': 144.,
                        'field_size': 64,
                        'smooth_km': 40., 
                        "quality_thresh": 2,
                        "nrepeat": 1,
                        "downscale": False,
                        "inpaint": False,
                        "de_mean": False,
                        "median": False,
                        "only_inpaint": False
                        }
                    ,
                     'Divb2': 
                        {
                        'fixed_km': 144.,
                        'field_size': 64,
                        'dx': 144./64,
                        }
                    ,
                 }
             }

    # Prep LLC Table
    llc_table = process.prep_table_for_preproc(
        llc_table, extract_file, field_size=
        (extract_dict['field_size'], extract_dict['field_size']))


    # Open HDF5 file
    f = h5py.File(outfile, 'w')
    for field in extract_dict['fields']:
        # Preprocess
        llc_table, success, pp_fields, meta = extract.preproc_field(
            llc_table, field, extract_dict['pdicts'][field],
            fixed_km=extract_dict['pdicts'][field]['fixed_km'],
            n_cores=10, dlocal=True,
            test_failures=False,
            test_process=False, override_RAM=True)

        # Write data
        pp_fields = np.array(pp_fields).astype(np.float32)
        f.create_dataset(field, data=pp_fields)

        # Add meta
        for key in meta.keys():
            llc_table[field+key] = meta[key]

        # Deal with failures in Table
        if np.any(~success):
            print(f"Failed to preprocess some {field} fields")
            fail = np.where(~success)[0]
            llc_table.loc[fail,'pp_type'] = -999
        # Good
        good = success & (llc_table['pp_type'] != -999)
        llc_table.loc[np.where(good)[0],'pp_type'] = 0
        #
        del pp_fields

    # Close
    f.close()    

    # Write table
    assert catalog.vet_main_table(llc_table)
    if not debug:
        fronts_io.write_main_table(llc_table, super_tbl_file, to_s3=False)
    else:
        embed(header='preproc_super 118')
        tbl_file = os.path.join(local_out_path, 'blah')
        fronts_io.write_main_table(llc_table, tbl_file, to_s3=False)

def gen_trainvalid(trainfile_config:str, outroot:str, debug:bool=False):
    """
    Generates training and validation datasets from a given configuration file.
    Args:
        trainfile_config (str): Path to the JSON configuration file for training.
        outroot (str): Root name for the output files.
        debug (bool, optional): If True, processes only a small subset of the data for debugging purposes. Defaults to False.
    Raises:
        ValueError: If the dataset specified in the configuration file is not 'LLC4320'.
        ValueError: If the format specified in the configuration file is not 'unet3d'.
    The function performs the following steps:
        1. Loads the LLC4320 table from a parquet file.
        2. Filters the table to include only rows where 'pp_type' is 0.
        3. Loads the configuration dictionary from the specified JSON file.
        4. Validates that the dataset specified in the configuration is 'LLC4320'.
        5. Generates training, validation, and test tables using the configuration.
        6. Processes each table (train, valid, test) to generate HDF5 and parquet files.
        7. For each table, loops through 'inputs' and 'targets' to preprocess fields and store them in the HDF5 file.
        8. Writes the processed data to HDF5 and parquet files.
        9. Validates and writes the main table to a parquet file.
    Note:
        The function assumes the existence of several external modules and functions such as `train_tables.gen_tvt`, 
        `extract.preproc_field`, `catalog.vet_main_table`, and `fronts_io.write_main_table`.
    """

    # Load
    llc_table = pandas.read_parquet(super_tbl_file)

    # Cut down
    llc_table = llc_table[llc_table.pp_type == 0].copy()
    
    with open(trainfile_config, 'r') as infile:
        config_dict = json.load(infile)

    # Datset
    if config_dict['dataset'] != 'LLC4320':
        raise ValueError("Only LLC4320 supported")

    # Generate the tables
    #embed(header='194 of llc4320_sst_ssh_proto.py')
    #reload(train_tables)
    train_tbl, valid_tbl, test_tbl = train_tables.gen_tvt(
        llc_table, config_dict)

    field_size = config_dict['field_size']

    # Process me
    for froot, tbl in zip(['train', 'valid', 'test'], 
                          [train_tbl, valid_tbl, test_tbl]):

        # Open HDF5 file
        h5_outfile = os.path.join(local_out_path, 'Training_Sets', f'{outroot}_{froot}{config_dict['name']}.h5')
        tbl_outfile = os.path.join(local_out_path, 'Training_Sets', f'{outroot}_{froot}{config_dict['name']}.parquet')
        f = h5py.File(h5_outfile, 'w')

        # Debug?
        if debug:
            tbl = tbl.iloc[:5].copy()

        # Loop on Inputs and Targets
        for ftype in ['inputs', 'targets']:
            if debug and ftype == 'inputs':
                continue

            # Data array
            ntype = len(config_dict[ftype].keys())
            if config_dict['format'] == 'unet3d':
                darray = np.zeros((len(tbl), ntype, 1, field_size, field_size), dtype=np.float32)
            else:
                raise ValueError("Bad format")
            
            for nchannel, field in enumerate(config_dict[ftype].keys()):
                print(f"Working on {field} for {ftype}")
                # Make sure we have the right field_size
                config_dict[ftype][field]['field_size'] = field_size

                # Preprocess
                tbl, success, pp_fields, meta = extract.preproc_field(
                    tbl, field, config_dict[ftype][field],
                    fixed_km=config_dict[ftype][field]['fixed_km'],
                    n_cores=10, dlocal=True,
                    test_failures=False,
                    test_process=False, override_RAM=True)

                if np.any(~success):
                    embed(header='Not all successful!')

                # Threshold or percentile?
                if ftype == 'targets' and ('threshold' in config_dict[ftype][field].keys() or
                  'precentile' in config_dict[ftype][field].keys()):
                    segments = np.ones_like(pp_fields, dtype=bool)
                    if 'threshold' in config_dict[ftype][field].keys():
                        segments &= pp_fields > config_dict[ftype][field]['threshold']
                    if 'percentile' in config_dict[ftype][field].keys():
                        per = np.percentile(pp_fields, config_dict[ftype][field]['percentile'],
                                            axis=(1,2))
                        for kk in range(pp_fields.shape[0]):
                            segments[kk] &= pp_fields[kk] > per[kk]
                    pp_fields = segments.astype(np.float32)

                embed(header='287 of proto')
                # Fill in
                darray[:,nchannel,0,:,:] = pp_fields
            # Write inputs
            dset = f.create_dataset(ftype, data=darray)
            dset.attrs['fields'] = list(config_dict[ftype].keys())
            dset.attrs['UID'] = tbl.UID.values.astype(str).astype('S')

        # Close it
        f.close()
        print(f"Wrote: {h5_outfile}")

        # Write the Table too
        assert catalog.vet_main_table(tbl)
        fronts_io.write_main_table(tbl, tbl_outfile, to_s3=False)
        print(f"Wrote: {tbl_outfile}")

# #######################################################33
def gallery(data_file:str=None, tbl_file:str=None,
            outfile:str='fig_gallery.png'):
    if data_file is None:
        data_file = super_preproc_file
    if tbl_file is None:
        tbl_file = super_tbl_file

    # Load the table
    print(f"Loading table: {tbl_file}")
    tbl = pandas.read_parquet(tbl_file)
    # Cut down to good ones
    keep = tbl.pp_type == 0
    front_tbl = tbl[keep].copy()

    srt = np.argsort(front_tbl['Divb2mu'].values)

    # Images
    f = h5py.File(data_file, 'r')

    # Figure
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(5,4)

    for row, perc in enumerate([1,5,50,95,99]):
        print("Working on row: ", row)
        ii = srt[int(perc/100*len(srt))]
        idx = front_tbl.index[ii]

        ax0 = plt.subplot(gs[row,0])
        ax1 = plt.subplot(gs[row,1])
        ax2 = plt.subplot(gs[row,2])
        ax3 = plt.subplot(gs[row,3])

        # Get the data
        ssta = f['SST'][idx]
        sss = f['SSS'][idx]
        Divb2 = f['Divb2'][idx]

        # Plot the 3 easy ones
        images.show_image(ssta, clbl='SSTa (deg C)', ax=ax0)
        images.show_image(sss, clbl='SSS (psu)', cm='viridis', ax=ax1)

        vmnx = (Divb2.min(), Divb2.max())
        images.show_image(Divb2, clbl=r'$\nabla b^2$', cm='Greys', 
                          ax=ax2, vmnx=vmnx)

        # Calculate divb2 from the smoothed sss
        sst = ssta + front_tbl['SSTmu'].values[ii]
        smooth_Divb2 = fronts.calc_gradb(
            sst, sss, dx=144./64)
        images.show_image(smooth_Divb2, clbl=r'Smooth $\nabla b^2$', 
                          cm='Greys', ax=ax3, vmnx=vmnx)


    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print("Wrote: ", outfile)

# #######################################################33
def main(flg:str):
    flg= int(flg)

    # Generate the LLC Table
    if flg == 1:
        generate_super_table()#debug=True, plot=True)

    # Generate the Super Preproc File
    if flg == 2:
        #preproc_super('dummy_file.json', debug=True)
        json_file = 'llc4320_sst144_sss40_extract.json'
        preproc_super(json_file)

    # Generate the Training, Validation, Test files
    if flg == 3:
        # A: 
        #   Inputs = Div SST, SST, SSS 
        #   Targets = Divb2 > 1e-14 + >=90%
        #json_file = 'llc4320_sst144_sss40_tvfileA.json'
        # B: 
        #   Inputs = Div SST^2, SSS, SST 
        #   Targets = Divb2 
        # C:
        #   Inputs = Div SST^2, SSS, SST 
        #   Targets = Divb2 normalized by <b>
        json_file = 'llc4320_sst144_sss40_tvfileC.json'
        gen_trainvalid(json_file, 'LLC4320_SST144_SSS40', debug=True)

    # Examine a set of images
    if flg == 10:
        gallery()

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:

        flg = 1 # Generate super table
        flg = 2 # Preproc super table

        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate (with noise)
        #flg += 2 ** 3  # 8 -- Evaluate (without noise)
    else:
        flg = sys.argv[1]

    main(flg)