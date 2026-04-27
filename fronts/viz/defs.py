"""Shared definitions for oceanographic visualization."""

# Colormap choices for oceanographic fields
cmaps = dict(
    Theta='RdYlBu_r',       # SST: warm reds to cool blues
    Salt='YlGnBu',          # Salinity: yellow-green-blue
    density='cividis',       # Density: colorblind-friendly sequential
    gradb='gray_r',          # Buoyancy gradient: dark = strong
    divergence='RdBu_r',     # Divergence: red/blue symmetric
    strain_mag='Reds',       # Strain magnitude
    # JPDF colormaps (one per property panel)
    jpdf_strain='Oranges',
    jpdf_divergence='Purples',
    jpdf_vorticity='Greens',
    jpdf_frontogenesis='Reds',
)

# Labels for axis / colorbar annotation
labels = dict(
    Theta=r'$\Theta$  ($^\circ$C)',
    Salt='Salinity  (PSU)',
    density=r'$\rho - 1025$  (kg m$^{-3}$)',
    gradb=r'$|\nabla b|$  (s$^{-2}$)',
    strain_mag_f=r'$|\mathbf{S}| / f$',
    divergence_f=r'$\delta / f$',
    relative_vorticity_f=r'$\zeta / f$',
    frontogenesis_tendency=r'$F_s$  (s$^{-3}$)',
)

# Reference density for offset display
RHO_REF = 1025.0
