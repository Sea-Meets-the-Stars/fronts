"""Shared definitions for oceanographic visualization."""

# Colormap choices for oceanographic fields
cmaps = dict(
    Theta='RdYlBu_r',       # SST: warm reds to cool blues
    Salt='YlGnBu',          # Salinity: yellow-green-blue
    density='cividis',       # Density: colorblind-friendly sequential
    gradb='gray_r',          # Buoyancy gradient: dark = strong
    divergence='RdBu_r',     # Divergence: red/blue symmetric
    strain_mag='RdBu_r',     # Strain magnitude
)

# Labels for axis / colorbar annotation
labels = dict(
    Theta=r'$\Theta$  ($^\circ$C)',
    Salt='Salinity  (PSU)',
    density=r'$\rho - 1025$  (kg m$^{-3}$)',
    gradb=r'$|\nabla b|$  (s$^{-2}$)',
)

# Reference density for offset display
RHO_REF = 1025.0
