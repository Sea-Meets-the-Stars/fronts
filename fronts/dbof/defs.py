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
    'SSH': dict(dtype=np.bool,
                help='Was SSH successfully extracted?'),
    'SSS': dict(dtype=np.bool,
                help='Was SSS successfully extracted?'),
    'SSSs': dict(dtype=np.bool,
                help='Was SSS (smoothed SSS) successfully extracted?'),
    'SSTK': dict(dtype=np.bool,
                help='Was SSTK successfully extracted?'),
})