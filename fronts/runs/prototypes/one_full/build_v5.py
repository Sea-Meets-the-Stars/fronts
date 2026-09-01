# Front finding and co-location for LLC4320.
#
#   1  gradb2      build the frontal-structure store, export the gradb2 channel
#   2  find        threshold gradb2 -> binary front map
#   3  group       label the fronts + geometric properties
#   4  colocate    build the remaining subsets, co-locate straight from zarr
#   5  push        copy the front products back to S3
#
# Steps 1-3 are self-contained: a front-binary map costs one NetCDF.  Step 4 is
# the only step that needs the other fields.
#
# Steps 1 and 4 hand store-building to dbof.run_all_subsets, which decides per
# subset and date: a store that is already complete is skipped, so re-running is
# a few S3 metadata reads.  The export is always done here, because
# run_all_subsets has no way to export a single channel.
#
# Everything run-specific -- pipeline, run_id, dates, subsets, ice masking --
# comes from the YAML config.  See prompts/fronts_build.md.
#
# Set build.tile_find in the config to run steps 1-3 on ONE 720x720 tile
# instead of the global rect grid.  Step 1 then computes gradb2 straight from
# the raw surface tracers for that tile (no global zarr is built, and none
# exists at hourly cadence anyway); steps 2 and 3 are unchanged, because the
# finding algorithms are plain 2D array operations with no grid assumptions.
# Products go to a tile-named subdirectory so they cannot collide with the
# global run's files, which share the same date folder and filenames.
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
from fronts.preproc.gradb2 import generate_tile_gradb2
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

    tile_find = cfg['tile_find']

    # binary_filename ignores the finding-config label, so a tile run and a
    # global run with the same run_id would write the same filenames into the
    # same date folder.  Give the tile its own leaf.
    run_dir = cfg['run_dir']
    if tile_find:
        run_dir = os.path.join(run_dir, tile_find.get('name', 'tile'))

    llc_io.set_fronts_path(os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts'))
    llc_io.set_run_layout(run_dir, file_tag=run_id)

    print(f"pipeline={cfg['pipeline']}  run_id={run_id}  "
          f"dates={len(timestamps)}  gradb2={gradb2_channel} "
          f"(subset={gradb2_subset})  finding_config={find_cfg}"
          + (f"  tile={tile_find}" if tile_find else ''))
    print(f"products -> {llc_io.run_root(run_id)}")

    # =======================================================================
    # STEP 1 -- gradb2
    # =======================================================================
    # Build the store that owns gradb2, then export that one channel.  The
    # subset holds 8 channels on SURF and 21 on DEPTH; the rest wait for step 4.
    if flg == 1:
        # --- one tile: compute gradb2 directly, no global store ------------
        if tile_find:
            generate_tile_gradb2(
                cfg['date_iterations'], timestamps, tile_find,
                version=run_id, field=gradb2_channel,
                clobber=cfg['clobber'])
            llc_meta.write_run_meta(cfg, config_file,
                                    extra={'gradb2_channel': gradb2_channel,
                                           'gradb2_subset': gradb2_subset,
                                           'tile_find': tile_find})
            return

        subsets = [gradb2_subset]
        if cfg['ice_mask_find']:
            # The mask is read from icearea.zarr at export time, so it has to
            # exist for the same run_id and date.
            subsets.append('icearea')

        generate_global_dataset(config_file, llc_io.run_root(run_id),
                                subsets=subsets, generate_only=True)

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
    # Build any zarr stores that are missing, then co-locate straight from
    # them: each field is read one at a time and never written to disk.
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
            # prop_algorithms.group_fronts wants lat/lon on the same grid as
            # the binary map.  The global default is the 12960x17280 coords
            # file; the tile's own XC/YC ride along in its gradb2 NetCDF.
            coords_file = (
                llc_io.derived_filename(timestamp, gradb2_channel,
                                        version=run_id)
                if tile_find else None)
            group_fronts(timestamp, find_cfg, run_id,
                         coords_file=coords_file)

        # STEP 4 -- co-locate fronts with the property fields.
        # skip_missing=True: co-locate whatever the stores hold rather than
        # dying on a channel that was never generated.
        if flg == 4:
            colocate_fronts(timestamp, find_cfg, run_id,
                            property_names=property_names,
                            percentiles=cfg['percentiles'],
                            skip_missing=True,
                            clobber=True,
                            config_file=config_file,
                            source=cfg['colocate_source'],
                            ice_mask=cfg['ice_mask_props'])


# Command line execution
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 0
        pass
    else:
        flg = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CONFIG

    main(flg, config_file)
