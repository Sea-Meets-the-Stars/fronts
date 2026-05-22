# Script to generate the tiles for the fronts N analysis

## Tropical Pacific

RUN_GS=true # Gulf Stream
RUN_TP=true # Tropical Pacific
GEN_FIGS=true

## Tropical Pacific

### Figures
if $RUN_TP && $GEN_FIGS; then
    python /home/xavier/Oceanography/python/fronts/dev/rho_and_N/plot_isopycnals.py \
        --density-tile     $OS_OGCM/LLC/Fronts/V3/20121109_120000/tiles/density_tile301_20121109T12.nc \
        --labels           $OS_OGCM/LLC/Fronts/V3/20121109_120000/labeled_fronts_global_20121109T12_00_00_v3_bin_D.npy \
        --front-index      $OS_OGCM/LLC/Fronts/V3/20121109_120000/front_index_20121109T12_00_00_v3_bin_D.parquet \
        --sigma0           22.6 \
        --outdir $OS_OGCM/LLC/Fronts/V3/20121109_120000/tiles
fi
