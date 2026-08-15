# Front finding and co-location for LLC4320.
#
#   1  gradb2      export the gradb2 channel from the frontal-structure store
#   2  find        threshold gradb2 -> binary front map
#   3  group       label the fronts + geometric properties
#   4  colocate    build the remaining subsets, then co-locate
#   5  push        copy the front products back to S3
#
# Steps 1-3 are self-contained: a front-binary map costs one NetCDF.  Step 4 is
# the only step that needs the other fields.
#
# Step 1 reads a zarr store that already exists.  Build one first with the
# preprocessing repo:
#     run-all-subsets --config <cfg> --netcdf-base <dir> \
#         --subsets frontal_structure --generate-only
#
# Everything run-specific -- pipeline, run_id, dates, subsets, ice masking --
# comes from the YAML config.  See prompts/fronts_build.md.
#
# Products are organised by the build that made them; filenames keep the source
# run_id, so a file always names the dataset it came from:
#     $OS_OGCM/LLC/Fronts/V5/{pipeline}/{date_prefix}/
#         LLC4320_{ts}_{channel}_{run_id}.nc
#
# Usage
# -----
#     python build_v5.py <step> [config.yaml]

import os
import sys

from fronts.llc import io as llc_io
from fronts.llc import meta as llc_meta
from fronts.llc import publish as llc_publish

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

#: Matches this script's name.  Everything build_v5 makes lands under
#: $OS_OGCM/LLC/Fronts/V5/{pipeline}/, whatever dataset it was made from --
#: the source is recorded in the filenames and the .meta descriptor instead.
BUILD_VERSION = 'V5'


def main(flg, config_file: str = DEFAULT_CONFIG):
    flg = int(flg)

    cfg        = read_build_config(config_file, build_version=BUILD_VERSION)
    run_id     = cfg['run_id']            # the source dataset tag
    timestamps = cfg['timestamps']        # 'YYYY-MM-DDTHH_MM_SS'
    find_cfg   = cfg['finding_config']    # fronts/finding/configs/finding_config_{X}.yaml

    # gradb2's channel name and owning subset under this pipeline.
    gradb2_channel = channel_for_root(config_file, cfg['gradb2_root'],
                                      depth_suffix=cfg['finding_suffix'])
    gradb2_subset  = subset_for_channel(config_file, gradb2_channel)

    llc_io.set_fronts_path(os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts'))
    llc_io.set_run_layout(cfg['run_dir'], file_tag=run_id)

    print(f"pipeline={cfg['pipeline']}  run_id={run_id}  "
          f"dates={len(timestamps)}  gradb2={gradb2_channel} "
          f"(subset={gradb2_subset})  finding_config={find_cfg}")
    print(f"products -> {llc_io.run_root(run_id)}")

    # =======================================================================
    # STEP 1 -- gradb2
    # =======================================================================
    # Export the one channel steps 2-3 read.  The subset holds 8 channels on
    # SURF and 21 on DEPTH; the rest wait for step 4.
    if flg == 1:
        llc_meta.write_run_meta(cfg, config_file,
                                extra={'gradb2_channel': gradb2_channel,
                                       'gradb2_subset': gradb2_subset})
        for timestamp in timestamps:
            print(f"[{timestamp}]")
            export_channels(config_file, timestamp, [gradb2_channel],
                            version=run_id,
                            ice_mask=cfg['ice_mask_find'])
        return

    # =======================================================================
    # STEP 5 -- publish
    # =======================================================================
    # Copy each timestamp's front products into a Fronts/ folder next to the
    # zarr stores they were derived from.
    if flg == 5:
        llc_publish.push_run(config_file, timestamps, version=run_id)
        return

    # =======================================================================
    # STEP 4 -- remaining subsets, then co-location
    # =======================================================================
    # Build any zarr stores that are missing, then export every channel the
    # active subsets produce.  The export runs through export_channels() -- the
    # same path step 1 uses -- so every product lands in this build's directory.
    if flg == 4:
        generate_global_dataset(config_file, llc_io.run_root(run_id),
                                generate_only=True)

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

        # STEP 4 -- export the property fields, then co-locate.
        # skip_missing=True: co-locate whatever exists rather than dying on a
        # channel that failed to export.
        if flg == 4:
            export_channels(config_file, timestamp, property_names,
                            version=run_id,
                            ice_mask=cfg['ice_mask_props'])
            colocate_fronts(timestamp, find_cfg, run_id,
                            property_names=property_names,
                            percentiles=cfg['percentiles'],
                            skip_missing=True,
                            clobber=True)


# Command line execution
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 0
        pass
    else:
        flg = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CONFIG

    main(flg, config_file)
