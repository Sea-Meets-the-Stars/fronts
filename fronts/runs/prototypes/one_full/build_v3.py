# Workflow to explore hyper-parameters for front finding
##  e.g. threshold, window size, etc.

import os
import sys

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

    # Versioned directory layout:
    #   $OS_OGCM/LLC/Fronts/v3/derived/
    #   $OS_OGCM/LLC/Fronts/v3/outputs/
    #   $OS_OGCM/LLC/Fronts/v3/group_fronts/
    fronts_root  = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', f'v{version}')
    derived_dir  = os.path.join(fronts_root, 'derived')
    outputs_dir  = os.path.join(fronts_root, 'outputs')
    group_dir    = os.path.join(fronts_root, 'group_fronts')

    # Generate gradb2 as zarr → netcdf
    if flg == 1:
        generate_gradb2(timestamp, config_file, version=version, run_id=run_id,
            create_zarr=False, path=derived_dir)

    # Find fronts -- binary pixels
    if flg == 2:
        find_gradb2_fronts(timestamp, config, version,
            derived_path=derived_dir, output_path=outputs_dir)

    # Group fronts (label + geometric properties)
    if flg == 3:
        group_fronts(timestamp, config, version,
            binary_path=outputs_dir, output_dir=group_dir)

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
                            create_zarr=False, derived_path=derived_dir)

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
                            property_names=property_names,
                            property_dir=derived_dir,
                            binary_path=outputs_dir,
                            group_dir=group_dir)


# Command line execution
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 0
        pass
    else:
        flg = sys.argv[1]

    main(flg)
