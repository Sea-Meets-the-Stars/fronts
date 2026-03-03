""" Data model related to front properties """

import numpy as np

import pandas

# Front properties
fprop_dmodel = {
    'UID': dict(dtype=(int, np.integer),
                help='UID of the front'),
    'UID_cutout': dict(dtype=(int, np.integer),
                help='UID of the cutout'),
    'datetime': dict(dtype=pandas.Timestamp,
                help='Timestamp of the cutout'),
    'Npix': dict(dtype=(int, np.integer),
                help='Number of pixels in the front'),
    'Divb2_mean': dict(dtype=(float,np.floating),
                help='Average Divb2 of the front'),
    'length': dict(dtype=(float,np.floating),
                help='Length of the front (km)'),
    'lat': dict(dtype=(float,np.floating),
                help='Average latitude of the front (deg)'),
    'lon': dict(dtype=(float,np.floating),
                help='Average longitude of the front (deg)'),
    'PV_max': dict(dtype=(float,np.floating),
                help='Maximum PV of the front (units)'),
    'PV_mean': dict(dtype=(float,np.floating),
                help='Maximum PV of the front (units)'),
}