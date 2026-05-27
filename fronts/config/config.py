# Frozen-dataclass configuration for fronts front-finding and properties.
#
# Modeled on llc4320-native-grid-preprocessing/src/dbof/dataset_creation/config.py.
# The YAML the loader expects is the new format (see
# fronts/runs/prototypes/one_full/testing_global_v3b.yaml) where the fronts
# configuration lives under a top-level ``fronts:`` key alongside dbof's own
# ``run:``, ``data:``, ``output:``, ... sections.

import os
from dataclasses import dataclass
from typing import List, Optional

import yaml


# Front-finding (the YAML ``binary:`` sub-section) parameters
@dataclass(frozen=True)
class FindingConfig:
    # Required
    window: int
    threshold: float
    thresh_mode: str
    thin: bool
    sharpen: bool
    despur: bool

    # Optional -- defaults chosen to be inert (no spur-removal, no dilation,
    # 8-connectivity which is what every checked-in YAML uses)
    Lspur: Optional[int] = None
    dilate: bool = False
    min_size: int = 0
    connectivity: int = 2

# Front-property (the YAML ``properties:`` sub-section) parameters
@dataclass(frozen=True)
class PropertiesConfig:
    stats: List[str]
    percentiles: List[int]
    min_npix: int
    nan_policy: str
    dilation_radius: int


# Top-level container
@dataclass(frozen=True)
class FrontsConfig:
    finding: FindingConfig
    properties: PropertiesConfig


def load_config(cfg_input: str or dict) -> FrontsConfig:
    """Load a :class:`FrontsConfig` from a YAML file.

    Expects the YAML to contain a top-level ``fronts:`` section with 
    ``binary``, and ``properties`` sub-sections. Unknown keys in any sub-section
    raise :class:`TypeError` (dataclass ``__init__`` rejecting unexpected
    kwargs); missing required keys raise :class:`TypeError` for the same reason.

    Parameters
    ----------
    cfg_input : str or dict
        Path to the YAML configuration file or the dictionary itself.

    Returns
    -------
    FrontsConfig
    """
    # Load the YAML
    if isinstance(cfg_input, str):
        with open(cfg_input, 'r') as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = cfg_input

    # The new format nests the fronts config under a top-level ``fronts:`` key
    fronts_section = raw.get('fronts')
    if fronts_section is None:
        raise ValueError(
            f"Config file '{cfg_input}' is missing the required top-level "
            f"'fronts:' section"
        )

    # Build the dataclasses; unknown/missing keys raise TypeError automatically
    return FrontsConfig(
        finding=FindingConfig(**fronts_section['finding']),
        properties=PropertiesConfig(**fronts_section['properties']),
    )
