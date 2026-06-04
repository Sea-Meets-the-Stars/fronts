# Code used to generate the tiles for the MLD investigation

import sys
import os
import logging

from pathlib import Path

# locals
from mld_defs import MLD_DEFS
import plot_top_N_density_profiles

from dbof.tiles import tile_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)

def gen_tiles(timestamp:str='2012-11-09 12:00:00',
        output_path:str=os.path.join(os.getenv('OS_OGCM'), 'LLC',
        'Fronts', 'V4', '20121109_120000', 'tiles')) -> None:

    # Generate folder
    os.makedirs(output_path, exist_ok=True)

    properties = ['density', 'temperature']
    regions = ['tropical_pacific', 'gs_central', 
                'gs_nc_1', 'gs_nc_2', 'gs_north_atlantic', 
                'kuroshio', 'california_current']

    # Loop me
    for region in regions:
        print('--------------------------------')
        print(f"Processing region: {region}")
        print('--------------------------------')
        for prop in properties:
            tile_utils.run(
                i_rect=MLD_DEFS[region]['i_j'][0],
                j_rect=MLD_DEFS[region]['i_j'][1],
                timestamp=timestamp,
                property=prop,
                output=output_path)

def gen_profiles(output_path:str='Figures/'):

    # Generate folder
    os.makedirs(output_path, exist_ok=True)

    # Loop me
    regions = ['tropical_pacific', 'gs_nc_1', 'california_current', 
               'gs_north_atlantic', 'kuroshio'] 

    for region in regions:
        print('--------------------------------')
        print(f"Processing region: {region}")
        print('--------------------------------')
        plot_top_N_density_profiles.run(
            density_tile=Path(MLD_DEFS[region]['density_tile']),
            gradb2_path=Path(MLD_DEFS[region]['gradb2']),
            labels_path=Path(MLD_DEFS[region]['labels']),
            front_index_path=Path(MLD_DEFS[region]['front_index']),
            front_properties_path=Path(MLD_DEFS[region]['front_properties']),
            N=10,
            outdir=Path(output_path),
            top_fronts_csv=None,
            strength_col='gradb2_p90',
            region_name=region,
            i_rect_range=(MLD_DEFS[region]['i_range'][0], MLD_DEFS[region]['i_range'][1]),
            j_rect_range=(MLD_DEFS[region]['j_range'][0], MLD_DEFS[region]['j_range'][1]),
            theta_path=Path(MLD_DEFS[region]['theta_tile']),
        )
        print('--------------------------------')

def main(flg):
    flg = int(flg)

    # Generate tiles
    if flg == 1:
        gen_tiles()

    # Generate tiles
    if flg == 2:
        gen_profiles()


# Command line
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 1
    else:
        flg = sys.argv[1]

    main(flg)