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
    'b': dict(dtype=np.bool,
                help='Was b (buoyancy) successfully extracted?'),
    'Fs': dict(dtype=np.bool,
                help='Was Fs (frontogenesis tendency) successfully extracted?'),
    'OW': dict(dtype=np.bool,
                help='Was OW (Okubo-Weiss) successfully extracted?'),
    'strain_rate': dict(dtype=np.bool,
                help='Was strain_rate successfully extracted?'),
    'divergence': dict(dtype=np.bool,
                help='Was divergence successfully extracted?'),
    'U': dict(dtype=np.bool,
                help='Was U successfully extracted?'),
    'V': dict(dtype=np.bool,
                help='Was V successfully extracted?'),
    'W': dict(dtype=np.bool,
                help='Was W successfully extracted?'),
    'SSH': dict(dtype=np.bool,
                help='Was SSH successfully extracted?'),
    'SSHs': dict(dtype=np.bool,
                help='Was SSHs (smoothed SSH) successfully extracted?'),
    'SSS': dict(dtype=np.bool,
                help='Was SSS successfully extracted?'),
    'SSSs': dict(dtype=np.bool,
                help='Was SSS (smoothed SSS) successfully extracted?'),
    'DivSST2': dict(dtype=np.bool,
                help='Was DivSST2 (gradient of SST) successfully extracted?'),
    'SSTK': dict(dtype=np.bool,
                help='Was SSTK successfully extracted?'),
})

# Meta
meta_dmodel = wr_defs.meta_dmodel.copy()

# Fields
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
        "DivSST2": {
            "desc": "Gradient of SST in Kelvin at full resolution, modulo resizing",
            "units": "(K/km)^2",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False,
                "dx": 2.25  # km
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
        "SSHs": {
            "desc": "SSHs smoothed by 15km to match MEASUREs data product",
            "units": "??",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "smooth_km": 15.0, 
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
        "divergence": {
            "desc": "Divergence, native resolution.",
            "units": "1/s^2",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False,
                "dx": 2.25  # km
            },
        },
        "strain_rate": {
            "desc": "Strain rate, native resolution.",
            "units": "1/s^2",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False,
                "dx": 2.25  # km
            },
        },
        "OW": {
            "desc": "Okubo-Weiss in 1/s^2, native resolution",
            "units": "1/s^2",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False,
                "dx": 2.25  # km
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
        "b": {
            "desc": "buoyancy, native resolution",
            "units": "unitless?",
            "pdict": {
                "resize": True,
                "downscale": False,
                "inpaint": False,
                "median": False,
                "de_mean": False
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
        