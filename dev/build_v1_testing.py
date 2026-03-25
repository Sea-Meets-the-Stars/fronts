"""Test run of the build_v1 pipeline — all inputs and outputs go to TESTING_DIR."""

import os
import sys

import yaml
import numpy as np
import xarray
import sys

from dbof.cli import generate_global
from dbof.cli import zarr_to_netcdf
import dbof.dataset_creation.config as dbof_config

from fronts.preproc import inpaint_edges
from fronts.llc import io as llc_io
from fronts.finding import algorithms as finding_algorithms
from fronts.finding import config as find_config
from fronts.finding import io as finding_io
from fronts.properties import algorithms as prop_algorithms
from fronts.properties import io as properties_io


# ---------------------------------------------------------------------------
# Settings — only thing to change between runs
# ---------------------------------------------------------------------------

TIMESTAMP   = '2012-11-09T12_00_00'
VERSION     = '1'
CONFIG_FILE = './testing_global_v1.yaml'
RUN_ID      = 'global_20260324_010000'
CONFIGS     = ['A', 'B', 'C']

TESTING_DIR = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts', 'testing')

PROPERTY_NAMES = [
    'relative_vorticity', 'divergence', 'strain_mag',
    'frontogenesis_tendency', 'okubo_weiss',
]


# ---------------------------------------------------------------------------
# Pipeline steps — same logic as build_v1.py, all paths point to TESTING_DIR
# ---------------------------------------------------------------------------

def generate_gradb2(clobber=False):
    out_file = llc_io.derived_filename(TIMESTAMP, 'gradb2', version=VERSION,
                                       path=TESTING_DIR)
    if os.path.isfile(out_file) and not clobber:
        print(f"gradb2 file exists and clobber is False. Returning")
        return
    generate_global.main(CONFIG_FILE, subset='frontal_structure', run_id=RUN_ID)
    cfg = dbof_config.load_config(CONFIG_FILE)
    with open(CONFIG_FILE) as fh:
        raw = yaml.safe_load(fh) or {}
    dataset_name = (raw.get('subsets', {}).get('frontal_structure', {}).get('dataset_name')
                    or cfg.output.dataset_name)
    os.makedirs(TESTING_DIR, exist_ok=True)
    zarr_to_netcdf.main(
        TESTING_DIR,
        output_filename=os.path.basename(out_file),
        mode='snapshots',
        run_id=RUN_ID or cfg.run.run_id,
        s3_endpoint=cfg.output.s3_endpoint,
        bucket=cfg.output.bucket,
        channels=['gradb2'],
        dates=cfg.data.date_iterations,
        dataset_name=dataset_name,
        folder=cfg.output.folder)


def find_fronts(config, inpaint=False, clobber=False):
    bfile = llc_io.derived_filename(TIMESTAMP, f'v{VERSION}_bin_{config}',
                                    version=VERSION, path=TESTING_DIR)
    # Use finding_io to get the standard filename, but redirect to TESTING_DIR
    bfile = finding_io.binary_filename(TIMESTAMP, config, VERSION, path=TESTING_DIR)
    if os.path.isfile(bfile) and not clobber:
        print(f"Binary front field {bfile} exists and clobber is False. Returning")
        return

    gradb2_file = llc_io.derived_filename(TIMESTAMP, 'gradb2', version=VERSION,
                                           path=TESTING_DIR)
    print(f"Loading gradb2 from: {gradb2_file}")
    gradb2 = xarray.open_dataset(gradb2_file)['gradb2'].values

    if inpaint:
        print("Inpainting...")
        gradb2 = inpaint_edges.inpaint(gradb2, method='biharmonic',
                                        second_pass='regular',
                                        second_threshold=1e-20)

    config_file = find_config.config_filename(config)
    cdict = find_config.load(config_file)
    bparam = cdict['binary']
    bparam['n_workers'] = 10
    bparam['verbose'] = True

    fronts = finding_algorithms.fronts_from_gradb2(gradb2, **bparam)
    finding_io.save_binary_fronts(fronts, TIMESTAMP, config, VERSION, path=TESTING_DIR)


def group_fronts(config, n_workers=None, skip_curvature=False):
    fronts_file = finding_io.binary_filename(TIMESTAMP, config, VERSION, path=TESTING_DIR)
    coords_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Fronts','coords','LLC_coords_lat_lon.nc')

    fronts_binary = np.load(fronts_file)
    ds = xarray.open_dataset(coords_file)
    lat = ds['lat'].values if 'lat' in ds else ds['YC'].values
    lon = ds['lon'].values if 'lon' in ds else ds['XC'].values
    ds.close()

    prop_algorithms.group_fronts(
        fronts_binary, lat, lon,
        fronts_file=fronts_file,
        output_dir=TESTING_DIR,
        n_workers=n_workers,
        skip_curvature=skip_curvature,
    )


def generate_properties(clobber=False):
    with open(CONFIG_FILE) as fh:
        raw = yaml.safe_load(fh) or {}

    channel_to_subset = {}
    for subset_name, sub_cfg in (raw.get('subsets') or {}).items():
        for ch in sub_cfg.get('compute_features_channels', []):
            ch = ch.strip()
            if ch:
                channel_to_subset[ch] = subset_name
        for ch in sub_cfg.get('model_data_feature_channels', []):
            ch = ch.strip()
            if ch:
                channel_to_subset[ch] = subset_name

    unknown = [p for p in PROPERTY_NAMES if p not in channel_to_subset]
    if unknown:
        raise ValueError(f"Properties not found in config: {unknown}")

    subset_to_channels = {}
    for prop in PROPERTY_NAMES:
        subset_to_channels.setdefault(channel_to_subset[prop], []).append(prop)

    for subset, channels in subset_to_channels.items():
        missing = [ch for ch in channels
                   if not os.path.isfile(
                       llc_io.derived_filename(TIMESTAMP, ch, version=VERSION,
                                               path=TESTING_DIR))]
        if not missing and not clobber:
            print(f"All {len(channels)} file(s) for subset '{subset}' exist. Skipping.")
            continue

        to_generate = channels if clobber else missing
        generate_global.main(CONFIG_FILE, subset=subset, run_id=RUN_ID)

        cfg = dbof_config.load_config(CONFIG_FILE)
        with open(CONFIG_FILE) as fh:
            raw = yaml.safe_load(fh) or {}
        dataset_name = (raw.get('subsets', {}).get(subset, {}).get('dataset_name')
                        or cfg.output.dataset_name)
        os.makedirs(TESTING_DIR, exist_ok=True)
        for channel in to_generate:
            out_file = llc_io.derived_filename(TIMESTAMP, channel, version=VERSION,
                                               path=TESTING_DIR)
            zarr_to_netcdf.main(
                TESTING_DIR,
                output_filename=os.path.basename(out_file),
                mode='snapshots',
                run_id=RUN_ID or cfg.run.run_id,
                s3_endpoint=cfg.output.s3_endpoint,
                bucket=cfg.output.bucket,
                channels=[channel],
                dates=cfg.data.date_iterations,
                dataset_name=dataset_name,
                folder=cfg.output.folder)


def colocate_fronts(config, clobber=False):
    fronts_file = finding_io.binary_filename(TIMESTAMP, config, VERSION, path=TESTING_DIR)
    time_str = TIMESTAMP.replace('_', ':')
    run_tag  = f'v{VERSION}_bin_{config}'
    out_file = properties_io.get_global_front_output_path(
        TESTING_DIR, time_str, 'properties', run_tag)
    if os.path.isfile(out_file) and not clobber:
        print(f"Properties file {out_file} exists and clobber is False. Returning")
        return

    missing = [name for name in PROPERTY_NAMES
               if not os.path.isfile(
                   llc_io.derived_filename(TIMESTAMP, name, version=VERSION,
                                           path=TESTING_DIR))]
    if missing:
        raise FileNotFoundError(
            f"Missing property file(s): {missing}. Run generate_properties() first.")

    labeled_file = properties_io.get_global_front_output_path(
        TESTING_DIR, time_str, 'label_map',run_tag)
    labeled = np.load(labeled_file)

    prop_algorithms.colocate_fronts(
        labeled=labeled,
        property_names=PROPERTY_NAMES,
        property_dir=TESTING_DIR,
        fronts_file=fronts_file,
        output_dir=TESTING_DIR,
        dilation_radius= 1,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(flg: str):
    flg = int(flg)

    if flg == 1:
        generate_gradb2()

    if flg == 2:
        for config in CONFIGS:
            find_fronts(config)

    if flg == 3:
        for config in CONFIGS:
            group_fronts(config)

    if flg == 4:
        generate_properties()

    if flg == 5:
        for config in CONFIGS:
            colocate_fronts(config)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 0)
