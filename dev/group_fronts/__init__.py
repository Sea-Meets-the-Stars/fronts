"""
Front Grouping and Characterization Module

This module provides tools for grouping connected front pixels into individual fronts
and characterizing their geometric and physical properties.
"""

from . import label
from . import geometry
from . import io


__version__ = '0.2.0'
__author__ = 'Lauren Hoffman'
__email__ = 'lhoffma2@ucsc.edu'

__all__ = [
    'label',
    'geometry',
    'io',
    'fields',
    'dimensionless',
    'dynamics',
]
