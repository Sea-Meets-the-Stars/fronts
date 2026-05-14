# Workflow to explore hyper-parameters for front finding
##  e.g. threshold, window size, etc.

import os
import sys

from fronts.llc import io as llc_io

from fronts.preproc.gradb2 import generate_gradb2
from fronts.finding.run import find_gradb2_fronts
from fronts.properties.run import group_fronts
from fronts.properties.run import generate_properties
from fronts.properties.run import colocate_fronts

from IPython import embed


# #######################################################
def main(flg: str):
    flg = int(flg)

    version = '3'
    timestamp   = '2012-11-09T12_00_00'
    config  = 'D'
    config_file = './testing_global_v3.yaml'
    run_id      = 'global_20121109_120000_v3'

    # All products land under: PATH / V{version} / YYYYMMDD_HHMMSS /
    # e.g. $OS_OGCM/LLC/Fronts/V3/20121109_120000/
    llc_io.set_fronts_path(os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts'))

    # Generate gradb2 as zarr → netcdf
    if flg == 1:
        generate_gradb2(timestamp, config_file, version=version, run_id=run_id,
            create_zarr=False)

    # Find fronts -- binary pixels
    if flg == 2:
        find_gradb2_fronts(timestamp, config, version)

    # Group fronts (label + geometric properties)
    if flg == 3:
        group_fronts(timestamp, config, version)

    # Generate physical property fields → individual per-property nc files
    if flg == 4:
        property_names = [
            'relative_vorticity', 'divergence', 'strain_mag',
            'frontogenesis_tendency', 'okubo_weiss','coriolis_f',
            'Eta','gradeta2','gradrho2','gradtheta2','gradsalt2',
            'rossby_number','Salt','strain_n','strain_s','Theta',
            'ug','vg','U','V','W','frontogenesis_geo','frontogenesis_ageo',
            'turner_angle',
            'oceTAUX','oceTAUY','SIarea',
        ]
        generate_properties(timestamp, config_file, version,
                            property_names=property_names, run_id=run_id,
                            create_zarr=False)

    # Co-locate fronts with physical properties
    if flg == 5:
        property_names = [
            'relative_vorticity', 'divergence', 'strain_mag',
            'frontogenesis_tendency', 'okubo_weiss','coriolis_f',
            'Eta','gradb2','gradeta2','gradrho2','gradtheta2','gradsalt2',
            'rossby_number','Salt','strain_n','strain_s','Theta',
            'ug','vg','U','V','W','frontogenesis_geo','frontogenesis_ageo',
            'turner_angle',
            'oceTAUX','oceTAUY','SIarea',
        ]
        colocate_fronts(timestamp, config, version,
                            property_names=property_names)


# Command line execution
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 0
        pass
    else:
        flg = sys.argv[1]

    main(flg)
