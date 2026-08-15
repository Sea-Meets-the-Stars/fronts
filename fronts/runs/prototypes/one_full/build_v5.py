# Full front-finding + co-location workflow (v5).
#
# Same four steps as v4, but pipeline-agnostic and split so that finding fronts
# no longer pays for fields it doesn't use.
#
#   1  GENERATE gradb2 only
#        build the frontal-structure zarr store, export ONE channel (gradb2)
#   2  FIND     binary fronts in that gradb2 field
#   3  GROUP    label the fronts + geometric properties
#   4  COLOCATE generate/export every remaining subset, then co-locate
#
# Steps 1-3 are all you need for a front-binary map; step 4 is the only step
# that touches the other ~140 channels.  v4 generated all of them in step 1.
#
# What v5 fixes (all six were verified against the current preprocessing repo;
# see prompts/fronts_build.md):
#   1. gradb2 is 'gradb2' on SURF/OSN and 'gradb2_sfc' on DEPTH -- resolved from
#      the config, not hardcoded.
#   2. PROPERTY_ROOTS is derived from subset_definitions, so the driver runs on
#      any pipeline instead of raising on 15 DEPTH-only roots.
#   3. Derived roots also pick up channels added upstream (R_ib, Wstar,
#      rossby_number) that v4's hand-written list silently dropped.
#   4. Step 1 exports 1 NetCDF instead of the whole subset (8 on SURF, 21 on
#      DEPTH).
#   5. Ice masking is a per-step toggle read from the config.
#   6. Exports are per-timestamp, so a config with >1 date no longer trips the
#      "output-filename needs a single date" guard in zarr_to_netcdf.
#
# Path alignment
# --------------
# run_id is used verbatim as the run tag -- no 'V' prefix, no separate version.
# Producer and consumer therefore agree automatically:
#     {netcdf_base}/{run_id}/{date_prefix}/LLC4320_{date}_{channel}_{run_id}.nc
# as long as netcdf_base == {OS_OGCM}/LLC/Fronts, which is what we set below.
#
# Usage
# -----
#     python build_v5.py 1 [config.yaml]     # gradb2 only
#     python build_v5.py 2 [config.yaml]     # find
#     python build_v5.py 3 [config.yaml]     # group
#     python build_v5.py 4 [config.yaml]     # everything else + co-locate

import os
import sys

from fronts.llc import io as llc_io

from fronts.finding.run import find_gradb2_fronts
from fronts.properties.run import (
    all_property_roots,
    channel_for_root,
    colocate_fronts,
    expand_property_roots,
    export_channels,
    generate_global_dataset,
    group_fronts,
    read_build_config,
    subset_for_channel,
)

DEFAULT_CONFIG = './run_v5_100_timesteps.yaml'


def main(flg, config_file: str = DEFAULT_CONFIG):
    flg = int(flg)

    # ---- Everything run-specific comes from the YAML -----------------------
    cfg        = read_build_config(config_file)
    run_id     = cfg['run_id']            # used verbatim as the run tag
    timestamps = cfg['timestamps']        # 'YYYY-MM-DDTHH_MM_SS'
    find_cfg   = cfg['finding_config']    # fronts/finding/configs/finding_config_{X}.yaml

    # gradb2's real channel name under THIS pipeline.
    gradb2_channel = channel_for_root(config_file, cfg['gradb2_root'],
                                      depth_suffix=cfg['finding_suffix'])
    gradb2_subset  = subset_for_channel(config_file, gradb2_channel)

    # All products land under {OS_OGCM}/LLC/Fronts/{run_id}/{date_prefix}/,
    # which is exactly where run_all_subsets writes.
    fronts_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts')
    llc_io.set_fronts_path(fronts_path)

    print(f"pipeline={cfg['pipeline']}  run_id={run_id}  "
          f"dates={len(timestamps)}  gradb2={gradb2_channel} "
          f"(subset={gradb2_subset})  finding_config={find_cfg}")

    # =======================================================================
    # STEP 1 -- gradb2 ONLY
    # =======================================================================
    # Build just the frontal-structure zarr store (--generate-only), then
    # export the single gradb2 channel.  Nothing else is produced: steps 2-3
    # need no other field, and a front-binary map is a valid end product on
    # its own.
    if flg == 1:
        subsets = [gradb2_subset]
        if cfg['ice_mask_find']:
            # The mask is read from icearea.zarr at export time, so it has to
            # exist for the same run_id + date.
            subsets.append('icearea')

        generate_global_dataset(config_file, fronts_path,
                                subsets=subsets,
                                generate_only=True)

        for timestamp in timestamps:
            print(f"[{timestamp}]")
            export_channels(config_file, timestamp, [gradb2_channel],
                            version=run_id,
                            ice_mask=cfg['ice_mask_find'])
        return

    # =======================================================================
    # STEP 4 -- everything else, then co-locate
    # =======================================================================
    # The full generate + export pass, for every subset in active_subsets.
    # Existing stores and .nc files are skipped, so the frontal_structure work
    # from step 1 is reused rather than redone.
    if flg == 4:
        generate_global_dataset(config_file, fronts_path,
                                ice_mask=cfg['ice_mask_props'])

        property_names = expand_property_roots(
            all_property_roots(config_file, exclude=cfg['exclude_roots']),
            config_file)
        print(f"Co-locating {len(property_names)} channels")
    else:
        property_names = None

    # ---- Per-timestamp steps ----------------------------------------------
    for timestamp in timestamps:
        print(f"[{timestamp}]")

        # STEP 2 -- binary fronts from gradb2
        if flg == 2:
            find_gradb2_fronts(timestamp, find_cfg, run_id,
                               gradb2_field=gradb2_channel)

        # STEP 3 -- label + geometric properties
        if flg == 3:
            group_fronts(timestamp, find_cfg, run_id)

        # STEP 4 -- co-locate fronts with the property fields.
        # skip_missing=True: co-locate whatever exists rather than dying on a
        # channel that failed to export.
        if flg == 4:
            colocate_fronts(timestamp, find_cfg, run_id,
                            property_names=property_names,
                            percentiles=cfg['percentiles'],
                            skip_missing=True,
                            clobber=True)


# Command line execution
if __name__ == '__main__':
    flg = sys.argv[1] if len(sys.argv) > 1 else 0
    cfg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CONFIG
    main(flg, cfg)
