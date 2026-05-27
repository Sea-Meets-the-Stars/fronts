# Define locations and the like for MLD investigation


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
    'i_range': (),
    'j_range': (),
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
    'i_range': (),
    'j_range': (),
    'tile': 359
}

# Kuroshio
kuroshio = {
    'name': 'kuroshio',
    'i_j': (8880, 9870),
    'i_range': (),
    'j_range': (),
    'tile': 324
}

# California Current
california_current = {
    'name': 'california_current',
    'i_j': (13170, 9950),
    'i_range': (),
    'j_range': (),
    'tile': 999
}

# Slurp em
MLD_DEFS['tropical_pacific'] = tropical_pacific
MLD_DEFS['gs_central'] = gs_central
MLD_DEFS['gs_nc_1'] = gs_north_carolina_1
MLD_DEFS['gs_nc_2'] = gs_north_carolina_2
MLD_DEFS['gs_north_atlantic'] = gs_north_atlantic
MLD_DEFS['kuroshio'] = kuroshio
MLD_DEFS['california_current'] = california_current