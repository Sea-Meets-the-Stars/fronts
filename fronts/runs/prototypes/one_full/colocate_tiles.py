# Co-locate LLC4320 fronts with properties computed on individual tiles.
#
# Consumes the fronts build_v5 already made and writes into the same tree, so
# run build_v5 steps 1-3 first.  Each property is computed on the fly for one
# 720x720 tile at the surface; nothing global is read or written.
#
# Per tile and timestamp, in
#     $OS_OGCM/LLC/Fronts/V5/{pipeline}/{date_prefix}/tile{idx:03d}/
#         front_properties_{TS}_{run_id}_bfronts.parquet
#         fields/tile{idx}_{ts}_{property}.nc     (cache, reused on re-run)
#
# Property names come from dbof.tiles.field_registry, not subset_definitions.
# See prompts/fronts_build.md.
#
# Usage
# -----
#     python colocate_tiles.py [config.yaml]

import os
import sys

from fronts.llc import io as llc_io
from fronts.llc import tiles as llc_tiles
from fronts.properties.run import colocate_tile, read_build_config

DEFAULT_CONFIG = './run_v5_tiles_monterey.yaml'

#: Reads and writes build_v5's tree -- the fronts are build_v5's fronts.
BUILD_VERSION = 'V5'


def main(config_file: str = DEFAULT_CONFIG):
    cfg      = read_build_config(config_file, build_version=BUILD_VERSION)
    run_id   = cfg['run_id']
    find_cfg = cfg['finding_config']
    props    = cfg['tile_properties']

    llc_io.set_fronts_path(os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts'))
    llc_io.set_run_layout(cfg['run_dir'], file_tag=run_id)

    print(f"pipeline={cfg['pipeline']}  run_id={run_id}  "
          f"dates={len(cfg['timestamps'])}  tiles={len(cfg['tiles'])}  "
          f"properties={len(props)}  finding_config={find_cfg}")
    print(f"products -> {llc_io.run_root(run_id)}")

    for loc in cfg['tiles']:
        where = {k: loc[k] for k in ('lon', 'lat', 'i_rect', 'j_rect') if k in loc}
        tile = llc_tiles.tile_for(**where)
        print(f"\n=== {loc.get('name', 'tile')}: index {tile.tile_idx}, "
              f"face {tile.face_idx}, rect j={tile.rect_j_slice.start} "
              f"i={tile.rect_i_slice.start} ===")

        for timestamp in cfg['timestamps']:
            print(f"[{timestamp}]")
            colocate_tile(timestamp, find_cfg, run_id,
                          property_names=props, tile=tile,
                          percentiles=cfg['percentiles'])


# Command line execution
if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG

    main(config_file)
