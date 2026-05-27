# Code used to generate the tiles for the MLD investigation

import sys
import os
import logging

# locals
from mld_defs import MLD_DEFS

from dbof.tiles import tile_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)

def main(flg):
    flg = int(flg)

    # Generate tiles
    if flg == 1:
        timestamp = '2012-11-09 12:00:00'
        output_path = os.path.join(os.getenv('OS_OGCM'), 'LLC',
            'Fronts', 'V4', '20121109_120000', 'tiles')
        # Generate folder
        os.makedirs(output_path, exist_ok=True)

        properties = ['density', 'temperature']
        regions = ['tropical_pacific', 'gs_central', 
                   'gs_nc_1', 'gs_nc_2', 'gs_north_atlantic']

        # Loop me
        for region in regions:
            for prop in properties:
                tile_utils.run(
                    i_rect=MLD_DEFS[region]['i_j'][0],
                    j_rect=MLD_DEFS[region]['i_j'][1],
                    timestamp=timestamp,
                    property=prop,
                    output=output_path)

# Command line
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 1
    else:
        flg = sys.argv[1]

    main(flg)