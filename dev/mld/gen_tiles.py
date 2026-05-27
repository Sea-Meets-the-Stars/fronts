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
        output = os.path.join(os.getenv('OS_OGCM'), 'LLC',
            'Fronts', 'V4', '20121109_120000', 'tiles')

        properties = ['density', 'temperature']
        regions = ['tropical_pacific']

        # Loop me
        for region in regions:
            for prop in properties:
                tile_utils.run(
                    i_rect=MLD_DEFS[region]['i_j'][0],
                    j_rect=MLD_DEFS[region]['i_j'][1],
                    timestamp=timestamp,
                    property=prop,
                    output=output)

# Command line
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 1
    else:
        flg = sys.argv[1]

    main(flg)