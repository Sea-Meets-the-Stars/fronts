# Front finding and co-location for LLC4320.
#
#   1  gradb2      build the frontal-structure zarr store, export gradb2
#   2  find        threshold gradb2 -> binary front map
#   3  group       label the fronts + geometric properties
#   4  colocate    build the remaining subsets, then co-locate
#
# Steps 1-3 are self-contained: a front-binary map costs one subset and one
# NetCDF.  Step 4 is the only step that needs the other fields.
#
# Data production is delegated to the preprocessing repo (dbof.run_all_subsets);
# this driver finds, groups and co-locates.  Everything run-specific -- pipeline,
# run_id, dates, subsets, ice masking -- comes from the YAML config.
# See prompts/fronts_build.md.
#
# run_id is the run tag, used verbatim, so producer and consumer paths agree:
#     $OS_OGCM/LLC/Fronts/{run_id}/{date_prefix}/LLC4320_{ts}_{channel}_{run_id}.nc
#
# Usage
# -----
#     python build_v5.py <step> [config.yaml] [--ndates N | --date DATE ...]
#
# --ndates / --date narrow the run to a few timesteps without touching the
# config: a reduced copy is written to a temp file and used for every stage, so
# the smoke test and the full run share one source of truth.

import argparse
import os

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
    write_date_subset_config,
)

DEFAULT_CONFIG = './run_v5_100_timesteps.yaml'


def main(flg, config_file: str = DEFAULT_CONFIG,
         dates: list = None, ndates: int = None):
    flg = int(flg)

    if dates or ndates:
        config_file = write_date_subset_config(config_file, dates=dates,
                                               ndates=ndates)

    cfg        = read_build_config(config_file)
    run_id     = cfg['run_id']            # the run tag, used verbatim
    timestamps = cfg['timestamps']        # 'YYYY-MM-DDTHH_MM_SS'
    find_cfg   = cfg['finding_config']    # fronts/finding/configs/finding_config_{X}.yaml

    # gradb2's channel name and owning subset under this pipeline.
    gradb2_channel = channel_for_root(config_file, cfg['gradb2_root'],
                                      depth_suffix=cfg['finding_suffix'])
    gradb2_subset  = subset_for_channel(config_file, gradb2_channel)

    fronts_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts')
    llc_io.set_fronts_path(fronts_path)

    print(f"pipeline={cfg['pipeline']}  run_id={run_id}  "
          f"dates={len(timestamps)}  gradb2={gradb2_channel} "
          f"(subset={gradb2_subset})  finding_config={find_cfg}")

    # =======================================================================
    # STEP 1 -- gradb2
    # =======================================================================
    # Build only the subset that owns gradb2, and export only that channel.
    # Steps 2-3 read nothing else, and a front-binary map is a valid end
    # product on its own.
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
    # STEP 4 -- remaining subsets, then co-location
    # =======================================================================
    # Full generate + export pass over every subset in active_subsets.
    # Existing stores and .nc files are skipped, so step 1's work is reused.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', type=int, choices=[1, 2, 3, 4],
                   help='1 gradb2, 2 find, 3 group, 4 colocate')
    p.add_argument('config', nargs='?', default=DEFAULT_CONFIG,
                   help=f'run YAML (default: {DEFAULT_CONFIG})')
    p.add_argument('--date', dest='dates', action='append', metavar='DATE',
                   help="Run only this date, e.g. '2011-12-04 00:00:00'.  "
                        "Repeatable.")
    p.add_argument('--ndates', type=int, metavar='N',
                   help='Run only the first N dates in the config.')
    a = p.parse_args()
    main(a.step, a.config, dates=a.dates, ndates=a.ndates)
