# Script to generate the tiles for the fronts N analysis

## Tropical Pacific

RUN_TP=true
GEN_FIGS=false

## Tiles

if $RUN_TP; then
    # Density
    python ../../../llc4320-native-grid-preprocessing/dev/tiles/generate_tile.py \
        --i 9800 --j 9000 --timestamp '2012-11-09 12:00:00' \
        --output $OS_OGCM/LLC/Fronts/V3/20121109_120000/tiles
fi

### Figures
if $RUN_TP && $GEN_FIGS; then
    python /home/xavier/Oceanography/python/fronts/dev/rho_and_N/plot_top_N_density_profiles.py \
        --density-tile     $OS_OGCM/LLC/Fronts/V3/20121109_120000/density_tile301_20121109T12.nc \
        --gradb2           $OS_OGCM/LLC/Fronts/V3/20121109_120000/LLC4320_2012-11-09T12_00_00_gradb2_v3.nc \
        --labels           $OS_OGCM/LLC/Fronts/V3/20121109_120000/labeled_fronts_global_20121109T12_00_00_v3_bin_D.npy \
        --front-index      $OS_OGCM/LLC/Fronts/V3/20121109_120000/front_index_20121109T12_00_00_v3_bin_D.parquet \
        --front-properties $OS_OGCM/LLC/Fronts/V3/20121109_120000/front_properties_20121109T12_00_00_v3_bin_D.parquet \
        --N 10 \
        --i-rect-range 9600 9950 \
        --j-rect-range 8950 9200 \
        --outdir .
fi
