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
    'avg_Divb2': dict(dtype=(float,np.floating),
                help='Average Divb2 of the front'),
    'avg_lat': dict(dtype=(float,np.floating),
                help='Average latitude of the front (deg)'),
    'avg_lon': dict(dtype=(float,np.floating),
                help='Average longitude of the front (deg)'),
    'front_id': dict(dtype=str,
                help='Unique front ID in TIME_LAT_LON format'),
    'time': dict(dtype=str,
                help='Timestamp of the front (ISO 8601 format)'),
    'npix': dict(dtype=(int, np.integer),
                help='Number of pixels in the front'),
    'centroid_lat': dict(dtype=(float, np.floating),
                help='Centroid latitude of the front (deg)'),
    'centroid_lon': dict(dtype=(float, np.floating),
                help='Centroid longitude of the front (deg)'),
    'length_km': dict(dtype=(float, np.floating),
                help='Length of the front along its skeleton (km)'),
    'orientation': dict(dtype=(float, np.floating),
                help='Front orientation angle from North (deg); 0=N-S, 90=E-W'),
    'lat_min': dict(dtype=(float, np.floating),
                help='Minimum latitude of the front bounding box (deg)'),
    'lat_max': dict(dtype=(float, np.floating),
                help='Maximum latitude of the front bounding box (deg)'),
    'lon_min': dict(dtype=(float, np.floating),
                help='Minimum longitude of the front bounding box (deg)'),
    'lon_max': dict(dtype=(float, np.floating),
                help='Maximum longitude of the front bounding box (deg)'),
    'mean_curvature': dict(dtype=(float, np.floating),
                help='Mean absolute curvature of the front (km⁻¹)'),
    'curvature_direction': dict(dtype=(float, np.floating),
                help='Mean signed curvature; positive=clockwise, negative=counterclockwise'),
}
