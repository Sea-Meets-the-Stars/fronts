""" Define data model for DBOF tables, building on wrangler.tbl_dmodel"""

import os
import numpy as np
import pandas

dbof_path = os.getenv('DBOF_PATH')

from wrangler import defs as wr_defs

tbl_dmodel = wr_defs.tbl_dmodel.copy()
tbl_dmodel['required'] = ('lat', 'lon', 'datetime', 'row', 'col', 'UID')

# Add more!
tbl_dmodel.update({
    'Divb2': dict(dtype=np.bool,
                help='Was Divb2 (div of buoyancy, squared) successfully extracted?'),
    'Fs': dict(dtype=np.bool,
                help='Was Fs (frontogenesis tendency) successfully extracted?'),
    'U': dict(dtype=np.bool,
                help='Was U successfully extracted?'),
    'V': dict(dtype=np.bool,
                help='Was V successfully extracted?'),
    'W': dict(dtype=np.bool,
                help='Was W successfully extracted?'),
    'SSH': dict(dtype=np.bool,
                help='Was SSH successfully extracted?'),
    'SSS': dict(dtype=np.bool,
                help='Was SSS successfully extracted?'),
    'SSSs': dict(dtype=np.bool,
                help='Was SSS (smoothed SSS) successfully extracted?'),
    'SSTK': dict(dtype=np.bool,
                help='Was SSTK successfully extracted?'),
})

fields_dmodel = {
       "U": {
            "desc": "U in m/s at full resolution, modulo resizing",
            "units": "m/s",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False
            },
        },
        "V": {
            "desc": "V in m/s at full resolution, modulo resizing",
            "units": "m/s",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False
            },
        },
        "SSTK": {
            "desc": "SST in Kelvin at full resolution, modulo resizing",
            "units": "K",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False
            },
        },
        "SSH": {
            "desc": "SSH in ??, native resolution",
            "units": "??",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False
            },
        },
        "SSS": {
            "desc": "SSS in psu, native resolution",
            "units": "psu",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False
            },
        },
        "SSSs": {
            "desc": "SSS in psu, smoothed to sattelite resolution",
            "units": "psu",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "smooth_km": 40.0, 
                "median": False,
                "de_mean": False
            },
        },
        "Fs": {
            "desc": "Frontogenesis tendency in 1/s^2, native resolution",
            "units": "1/s^2 (maybe)",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False,
                "dx": 2.25
            },
        },
        "Divb2": {
            "desc": "|Div b|^2 in 1/s^2, native resolution",
            "units": "1/s^2 (maybe)",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False,
                "dx": 2.25
            }
        }
    }
        