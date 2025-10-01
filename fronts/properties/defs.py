""" Data model related to front properties """

import numpy as np

import pandas

# Front proprties
fprop_dmodel = {
    'UID': dict(dtype=(int, np.integer),
                help='UID of the cutout'),
    'datetime': dict(dtype=pandas.Timestamp,
                help='Timestamp of the cutout'),
    'flabel': dict(dtype=(int, np.integer),
                help='Front label in the cutout (1, 2,, ...). 0=not in a front'),
    'Npix': dict(dtype=(int, np.integer),
                help='Number of pixels in the front'),
    'avg_Divb2': dict(dtype=(float,np.floating),
                help='Average Divb2 of the front'),
    'avg_lat': dict(dtype=(float,np.floating),
                help='Average latitude of the front (deg)'),
    'avg_lon': dict(dtype=(float,np.floating),
                help='Average longitude of the front (deg)'),
}