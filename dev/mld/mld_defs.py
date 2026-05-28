# Define locations and the like for MLD investigation

import os

MLD_DEFS = {}

# Tropical Pacific
tropical_pacific = {
    'name': 'tropical_pacific',
    'i_j': (9800, 9000),
    'i_range': (9600, 9950),
    'j_range': (8950, 9200),
    'tile': 301
}

# Gulf Stream central
gs_central = {
    'name': 'gs_central',
    'i_j': (16170, 9950),
    'i_range': (16200, 16450),
    'j_range': (9850, 10050),
    'tile': 334
}

# GS North Carolina 1
gs_north_carolina_1 = {
    'name': 'gs_nc_1',
    'i_j': (15600, 9800),
    'i_range': (15525, 15725),
    'j_range': (9700, 9900),
    'tile': 333
}

# GS North Carolina 2
gs_north_carolina_2 = {
    'name': 'gs_nc_2',
    'i_j': (15760, 10060),
    'i_range': (),
    'j_range': (),
    'tile': 333
}

# GS North Atlantic
gs_north_atlantic = {
    'name': 'gs_north_atlantic',
    'i_j': (16590, 10350),
    'i_range': (16550, 16750),
    'j_range': (10180, 10380),
    'tile': 359
}

# Kuroshio
kuroshio = {
    'name': 'kuroshio',
    'i_j': (8880, 9870),
    'i_range': (8700, 8900),
    'j_range': (9700, 9900),
    'tile': 324
}

# California Current
california_current = {
    'name': 'california_current',
    'i_j': (13170, 9950),
    'i_range': (13000, 13250),
    'j_range': (9800, 10000),
    'tile': 330
}

# Slurp em
MLD_DEFS['tropical_pacific'] = tropical_pacific
MLD_DEFS['gs_central'] = gs_central
MLD_DEFS['gs_nc_1'] = gs_north_carolina_1
MLD_DEFS['gs_nc_2'] = gs_north_carolina_2
MLD_DEFS['gs_north_atlantic'] = gs_north_atlantic
MLD_DEFS['kuroshio'] = kuroshio
MLD_DEFS['california_current'] = california_current

for region in MLD_DEFS.keys():
    MLD_DEFS[region]['density_tile'] = os.path.join(os.getenv('OS_OGCM'), 'LLC',
        'Fronts', 'V4', '20121109_120000', 'tiles', 
        f'density_tile{MLD_DEFS[region]['tile']}_20121109T12.nc')
    MLD_DEFS[region]['theta_tile'] = os.path.join(os.getenv('OS_OGCM'), 'LLC',
        'Fronts', 'V4', '20121109_120000', 'tiles', 
        f'theta_tile{MLD_DEFS[region]['tile']}_20121109T12.nc')
    MLD_DEFS[region]['gradb2'] = os.path.join(os.getenv('OS_OGCM'), 'LLC',
        'Fronts', 'V3', '20121109_120000', 'LLC4320_2012-11-09T12_00_00_gradb2_v3.nc')
    MLD_DEFS[region]['labels'] = os.path.join(os.getenv('OS_OGCM'), 'LLC',
        'Fronts', 'V3', '20121109_120000', 'labeled_fronts_global_20121109T12_00_00_v3_bin_D.npy')
    MLD_DEFS[region]['front_index'] = os.path.join(os.getenv('OS_OGCM'), 'LLC',
        'Fronts', 'V3', '20121109_120000', 'front_index_20121109T12_00_00_v3_bin_D.parquet')
    MLD_DEFS[region]['front_properties'] = os.path.join(os.getenv('OS_OGCM'), 'LLC',
        'Fronts', 'V3', '20121109_120000', 'front_properties_20121109T12_00_00_v3_bin_D.parquet')
