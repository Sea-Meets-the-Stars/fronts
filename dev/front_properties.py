
import os
import numpy as np

import h5py
import pandas

from wrangler.ogcm import utils as wr_ogcm_utils

from fronts.finding import dev as finding_dev
from fronts.finding import run as finding_run
from fronts.finding import params as finding_params
from fronts.dbof import io as dbof_io
from fronts.dbof import utils as dbof_utils
from fronts.properties import measure as fprop_measure
from fronts.properties import views as fprop_views

from IPython import embed
    
b_front_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'Training_Sets', 
                     'LLC4320_SST144_SSS40_fronts.h5')
dbof_dev_file = '../fronts/runs/dbof/dev/llc4320_dbof_dev.json'


def test_fprop_cutout(idx:int=500):

    # Load up
    cutouts, tbl = finding_dev.load_test_data()
    Divb2 = cutouts['targets'][:, 0, 0, ...]
    with h5py.File(b_front_file, 'r') as f:
        fronts = f['fronts'][:]

    ncutouts = Divb2.shape[0]

    # Fake lat, lon
    latlons = np.random.uniform(size=(2,ncutouts))

    # Generate lat, lon images
    lat_cutouts, lon_cutouts = wr_ogcm_utils.latlons_for_cutouts(
        latlons, Divb2.shape[1], 2.25)

    # Front properties for one
    fprop_dict = fprop_measure.fprops_in_fields(
        fronts[idx], ['avg_lat', 'avg_lon', 'avg_Divb2'],
        [lat_cutouts[idx], lon_cutouts[idx], Divb2[idx]])

    embed(header='157 of test_fprop cutout')

def test_views(UID:int=132796874601355073):
    dbof_tbl = dbof_io.load_main_table(dbof_dev_file)

    # Grab em
    fields = ['Divb2', 'Fs']
    field_data = dbof_utils.grab_fields(dbof_dev_file, fields, UID)

    # Show em
    fprop_views.show_fields(field_data, 'test_views.png')

if __name__ == "__main__":

    # Test measuring front properties for a series of cutouts
    #test_fprop_cutout()


    # Views
    test_views()