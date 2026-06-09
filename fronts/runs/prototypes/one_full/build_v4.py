# Full front-finding + co-location workflow, DEPTH pipeline (v4).
#
# Data production is fully delegated to the preprocessing repo's batch driver
# (dbof.run_all_subsets): it generates every active-subset Zarr store and
# exports each (suffix-expanded) channel to NetCDF in one pass, with depth-
# suffix expansion, optional ice masking, and per-subset/per-channel error
# isolation.  fronts then only finds, groups, and co-locates.
#
# Path alignment
# --------------
# run_all_subsets writes:
#     {netcdf_base}/{run_id}/{date_prefix}/LLC4320_{date}_{channel}_{run_id}.nc
# fronts reads from:
#     {OS_OGCM}/LLC/Fronts/V{version}/{date_prefix}/LLC4320_{ts}_{field}_V{version}.nc
# These coincide when:
#     netcdf_base = {OS_OGCM}/LLC/Fronts   and   run_id = 'V{version}'
# So run_v4_depth.yaml MUST set run.run_id = 'V4' for version = '4'.
#
# Steps (pass the step number on the command line):
#   1  generate + export ALL active subsets via run_all_subsets
#   2  find binary fronts in the (surface) gradb2 field
#   3  group fronts (label + geometric properties)
#   4  co-locate fronts with the per-property fields

import os
import sys

import yaml

from fronts.llc import io as llc_io

from fronts.finding.run import find_gradb2_fronts
from fronts.properties.run import group_fronts
from fronts.properties.run import colocate_fronts
from fronts.properties.run import expand_property_roots
from fronts.properties.run import generate_global_dataset

# Use the producer's own date helpers so derived timestamps match the exported
# NetCDF filenames exactly (date -> prefix -> 'YYYY-MM-DDTHH_MM_SS').
from dbof.global_dataset_creation.iterations import (
    date_to_run_id, prefix_to_filename_date,
)

from IPython import embed


# Property ROOTS to co-locate -- every channel across all DEPTH subsets.
# expand_property_roots() turns each root into its configured channels:
#   * "suffixed" roots (depth subsets w/ a depth_suffixes key) -> one channel
#     per active suffix, e.g. N2 -> N2_sfc, N2_z25m, N2_mld, N2_mld_mean
#   * "bare" roots (extra_channels, native model fields, and all surface-only
#     subsets) pass through unchanged, e.g. coriolis_f, oceTAUX, SIarea
# Grouping below mirrors dbof.global_dataset_creation.subset_definitions
# (DEPTH).  Every subset listed here must also be in active_subsets in the YAML
# so run_all_subsets actually produces the files.
PROPERTY_ROOTS = [
    # stratification          (suffixed: N2;  bare extras: mixed_layer_depth, ml_heat_content)
    'N2', 'mixed_layer_depth', 'ml_heat_content',
    # vertical_shear          (suffixed)
    'vertical_shear', 'Ri',
    # mixing_parameters       (suffixed)
    'Fr', 'Ro', 'Bu',
    # ertel_pv                (suffixed)
    'ertel_pv', 'ertel_pv_vertical', 'ertel_pv_tilt',
    # buoyancy_fluxes         (suffixed)
    'uB', 'vB', 'wB',
    # surface_wind            (surface-only -> all bare)
    'oceTAUX', 'oceTAUY', 'oceQnet',
    'wind_stress_curl', 'ekman_pumping', 'u_ekman', 'v_ekman',
    # energetics              (suffixed)
    'KE',
    # frontal_structure       (suffixed)
    'gradb2', 'gradtheta2', 'gradsalt2', 'gradrho2', 'gradeta2', 'turner_angle',
    # kinematic               (suffixed;  bare extra: coriolis_f)
    'relative_vorticity', 'strain_n', 'strain_s', 'strain_mag', 'divergence',
    'rossby_number', 'okubo_weiss', 'coriolis_f',
    # frontogenesis           (suffixed)
    'frontogenesis_tendency', 'frontogenesis_geo', 'frontogenesis_ageo',
    'ug', 'vg',
    # native_fields           (suffixed)
    'Theta', 'Salt', 'Eta', 'U', 'V', 'W',
    # icearea                 (surface-only -> bare)
    'SIarea',
]

# Front-finding uses the SURFACE gradb2 field.
GRADB2_FIELD = 'gradb2_sfc'

# Set True to NaN-mask ice-covered points during the NetCDF export.
ICE_MASK = False


# #######################################################
def main(flg: str):
    flg = int(flg)

    # config (front-finding label) selects fronts/finding/configs/
    # finding_config_{config}.yaml -- a different concern from the generation
    # config below, so it stays an explicit knob here.
    config      = 'D'
    config_file = './run_v4_depth.yaml'

    # Derive run_id + timestamps from the generation config (single source of
    # truth) so they cannot drift from the config.  run_id is the run tag used
    # verbatim everywhere (it can be anything, e.g. 'Vtest', 'vtest', ...).
    with open(config_file) as fh:
        cfg = yaml.safe_load(fh) or {}
    run_id = cfg['run']['run_id']
    timestamps = [prefix_to_filename_date(date_to_run_id(d))
                  for d in cfg['data']['date_iterations']]

    # All products land under: PATH / {run_id} / YYYYMMDD_HHMMSS /
    # e.g. $OS_OGCM/LLC/Fronts/Vtest/20121109_120000/  -- matching the
    # output of dbof.run_all_subsets ({netcdf_base}/{run_id}/{date_prefix}/).
    fronts_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts')
    llc_io.set_fronts_path(fronts_path)

    # Property roots -> fully-suffixed channel names from the config.
    property_names = expand_property_roots(PROPERTY_ROOTS, config_file)

    # Generate + export all active subsets (Zarr + per-channel NetCDF) for all
    # dates.  run_id, pipeline, subsets, dates, and depth_suffixes all come
    # from the config.  Existing stores/files are skipped (pass clobber=True
    # to force a rebuild).
    if flg == 1:
        generate_global_dataset(config_file, fronts_path, ice_mask=ICE_MASK)
        return

    # Per-timestamp steps (run over every date in the config).
    # Not needed for flg == 1 since generate_global_dataset() already loops over dates.
    for timestamp in timestamps:
        # Find fronts -- binary pixels
        if flg == 2:
            find_gradb2_fronts(timestamp, config, run_id,
                               gradb2_field=GRADB2_FIELD)

        # Group fronts (label + geometric properties)
        if flg == 3:
            group_fronts(timestamp, config, run_id)

        # Co-locate fronts with physical properties
        if flg == 4:
            colocate_fronts(timestamp, config, run_id,
                            property_names=property_names)


# Command line execution
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 0
        pass
    else:
        flg = sys.argv[1]

    main(flg)
