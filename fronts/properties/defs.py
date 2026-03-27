""" Data model related to front properties """

import numpy as np

import pandas

# Front properties
fprop_dmodel = {
    'UID': dict(dtype=(int, np.integer),
                help='UID of the front'),
    'UID_cutout': dict(dtype=(int, np.integer),
                help='UID of the cutout'),
    'datetime': dict(dtype=pandas.Timestamp,
                help='Timestamp of the cutout'),
    'Npix': dict(dtype=(int, np.integer),
                help='Number of pixels in the front'),
    'flabel': dict(dtype=(int, np.integer),
                help='Front label in the cutout (1, 2,, ...). 0=not in a front'),
    'avg_Divb2': dict(dtype=(float,np.floating),
                help='Average Divb2 of the front'),
    'length': dict(dtype=(float,np.floating),
                help='Length of the front (km)'),
    'lat': dict(dtype=(float,np.floating),
                help='Average latitude of the front (deg)'),
    'lon': dict(dtype=(float,np.floating),
                help='Average longitude of the front (deg)'),
    'PV_max': dict(dtype=(float,np.floating),
                help='Maximum PV of the front (units)'),
    'PV_mean': dict(dtype=(float,np.floating),
                help='Maximum PV of the front (units)'),
    'front_id': dict(dtype=str,
                help='Unique front ID in TIME_LAT_LON format'),
    'time': dict(dtype=str,
                help='Timestamp of the front (ISO 8601 format)'),
    'npix': dict(dtype=(int, np.integer),
                help='Number of pixels in the front'),
    'centroid_lat': dict(dtype=(float, np.floating),
                help='Centroid latitude of the front (deg)'),
    'centroid_lon': dict(dtype=(float, np.floating),
                help='Centroid longitude of the front (deg)'),
    'length_km': dict(dtype=(float, np.floating),
                help='Length of the front along its skeleton (km)'),
    'orientation': dict(dtype=(float, np.floating),
                help='Front orientation angle from North (deg); 0=N-S, 90=E-W'),
    'lat_min': dict(dtype=(float, np.floating),
                help='Minimum latitude of the front bounding box (deg)'),
    'lat_max': dict(dtype=(float, np.floating),
                help='Maximum latitude of the front bounding box (deg)'),
    'lon_min': dict(dtype=(float, np.floating),
                help='Minimum longitude of the front bounding box (deg)'),
    'lon_max': dict(dtype=(float, np.floating),
                help='Maximum longitude of the front bounding box (deg)'),
    'mean_curvature': dict(dtype=(float, np.floating),
                help='Mean absolute curvature of the front (km⁻¹)'),
    'curvature_direction': dict(dtype=(float, np.floating),
                help='Mean signed curvature; positive=clockwise, negative=counterclockwise'),
    'label': dict(dtype=(int, np.integer),
            help='Integer front label in the labeled array'),
    'name': dict(dtype=str,
                help='Unique front ID string (TIME_LAT_LON format), same as front_id'),
    'y0': dict(dtype=(int, np.integer),
                help='Bounding box row start index (latitude axis)'),
    'y1': dict(dtype=(int, np.integer),
                help='Bounding box row end index (latitude axis)'),
    'x0': dict(dtype=(int, np.integer),
                help='Bounding box column start index (longitude axis)'),
    'x1': dict(dtype=(int, np.integer),
                help='Bounding box column end index (longitude axis)'),
    'num_branches': dict(dtype=(int, np.integer),
                help='Number of branch/junction points in the front skeleton'),
}

# Oceanographic property field definitions
# Each entry provides units, a short equation, a description, and the subset
# group as defined in testing_global_v1.yaml.
ocean_field_defs = {
    # ------------------------------------------------------------------
    # Native LLC4320 fields
    # ------------------------------------------------------------------
    'Theta': dict(
        units='°C',
        equation=None,
        description='Potential temperature (LLC4320 native field)',
        subset='native_fields',
    ),
    'Salt': dict(
        units='PSU',
        equation=None,
        description='Salinity (LLC4320 native field)',
        subset='native_fields',
    ),
    'Eta': dict(
        units='m',
        equation=None,
        description='Sea surface height (LLC4320 native field)',
        subset='native_fields',
    ),
    'U': dict(
        units='m/s',
        equation=None,
        description='Zonal velocity (LLC4320 native field)',
        subset='native_fields',
    ),
    'V': dict(
        units='m/s',
        equation=None,
        description='Meridional velocity (LLC4320 native field)',
        subset='native_fields',
    ),
    'W': dict(
        units='m/s',
        equation=None,
        description='Vertical velocity (LLC4320 native field)',
        subset='native_fields',
    ),
    # ------------------------------------------------------------------
    # Frontal structure fields
    # ------------------------------------------------------------------
    'gradb2': dict(
        units='s\u207b\u2074',          # s⁻⁴
        equation='|∇b|² = (∂b/∂x)² + (∂b/∂y)²',
        description='Squared surface buoyancy gradient magnitude',
        subset='frontal_structure',
    ),
    'gradsalt2': dict(
        units='(PSU/m)²',
        equation='|∇S|² = (∂S/∂x)² + (∂S/∂y)²',
        description='Squared salinity gradient magnitude',
        subset='frontal_structure',
    ),
    'gradtheta2': dict(
        units='(K/m)²',
        equation='|∇θ|² = (∂θ/∂x)² + (∂θ/∂y)²',
        description='Squared temperature gradient magnitude',
        subset='frontal_structure',
    ),
    'gradeta2': dict(
        units='(m/m)²',
        equation='|∇η|² = (∂η/∂x)² + (∂η/∂y)²',
        description='Squared SSH gradient magnitude',
        subset='frontal_structure',
    ),
    # ------------------------------------------------------------------
    # Kinematic fields
    # ------------------------------------------------------------------
    'relative_vorticity': dict(
        units='s\u207b\u00b9',          # s⁻¹
        equation='ω = ∂v/∂x − ∂u/∂y',
        description='Relative vorticity',
        subset='kinematic',
    ),
    'strain_n': dict(
        units='s\u207b\u00b9',          # s⁻¹
        equation='σ_n = ∂u/∂x − ∂v/∂y',
        description='Normal (stretching) strain',
        subset='kinematic',
    ),
    'strain_s': dict(
        units='s\u207b\u00b9',          # s⁻¹
        equation='σ_s = ∂u/∂y + ∂v/∂x',
        description='Shear strain',
        subset='kinematic',
    ),
    'strain_mag': dict(
        units='s\u207b\u00b9',          # s⁻¹
        equation='|σ| = √(σ_n² + σ_s²)',
        description='Strain magnitude',
        subset='kinematic',
    ),
    'divergence': dict(
        units='s\u207b\u00b9',          # s⁻¹
        equation='δ = ∂u/∂x + ∂v/∂y',
        description='Horizontal velocity divergence',
        subset='kinematic',
    ),
    'coriolis_f': dict(
        units='s\u207b\u00b9',          # s⁻¹
        equation='f = 2Ω sin(φ)',
        description='Coriolis parameter',
        subset='kinematic',
    ),
    'rossby_number': dict(
        units='dimensionless',
        equation='Ro = ω/f',
        description='Rossby number',
        subset='kinematic',
    ),
    'okubo_weiss': dict(
        units='s\u207b\u00b2',          # s⁻²
        equation='OW = σ_n² + σ_s² − ω²',
        description='Okubo-Weiss parameter',
        subset='kinematic',
    ),
    # ------------------------------------------------------------------
    # Frontogenesis fields
    # ------------------------------------------------------------------
    'frontogenesis_tendency': dict(
        units='s\u207b\u2075',          # s⁻⁵
        equation='F = −(∂u/∂x · b_x² + (∂u/∂y + ∂v/∂x) · b_x b_y + ∂v/∂y · b_y²)',
        description='Kinematic frontogenesis tendency',
        subset='frontogenesis',
    ),
    'ug': dict(
        units='m/s',
        equation='u_g = −(g/f) ∂η/∂y',
        description='Geostrophic zonal velocity',
        subset='frontogenesis',
    ),
    'vg': dict(
        units='m/s',
        equation='v_g = (g/f) ∂η/∂x',
        description='Geostrophic meridional velocity',
        subset='frontogenesis',
    ),
    'frontogenesis_geo': dict(
        units='s\u207b\u2075',          # s⁻⁵
        equation='F(u_g, v_g)',
        description='Geostrophic frontogenesis tendency',
        subset='frontogenesis',
    ),
    'frontogenesis_ageo': dict(
        units='s\u207b\u2075',          # s⁻⁵
        equation='F − F_geo',
        description='Ageostrophic frontogenesis tendency',
        subset='frontogenesis',
    ),
}
