# Frozen-dataclass configuration for fronts front-finding and properties.
#
# Modeled on llc4320-native-grid-preprocessing/src/dbof/dataset_creation/config.py.
# The YAML the loader expects is the new format (see
# fronts/runs/prototypes/one_full/testing_global_v3b.yaml) where the fronts
# configuration lives under a top-level ``fronts:`` key alongside dbof's own
# ``run:``, ``data:``, ``output:``, ... sections.

import os
from dataclasses import dataclass
from importlib import resources
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
    label: str
    finding: FindingConfig
    properties: PropertiesConfig


def config_filename(config_label: str, path: str = None) -> str:
    """Build the full path to a finding configuration YAML file.

    Parameters
    ----------
    config_label : str
        Short label identifying the config (e.g. 'A'). The resulting filename is
        ``finding_config_{config_label}.yaml``.
    path : str, optional
        Directory containing the config file. Defaults to
        ``fronts/finding/configs/`` inside the installed package.

    Returns
    -------
    str
        Full path to the configuration file.
    """
    if path is None:
        path = os.path.join(resources.files('fronts'), 'finding', 'configs')
    base = f'finding_config_{config_label}.yaml'
    return os.path.join(path, base)


def load_config(path: str) -> FrontsConfig:
    """Load a :class:`FrontsConfig` from a YAML file.

    Expects the YAML to contain a top-level ``fronts:`` section with ``label``,
    ``binary``, and ``properties`` sub-sections. Unknown keys in any sub-section
    raise :class:`TypeError` (dataclass ``__init__`` rejecting unexpected
    kwargs); missing required keys raise :class:`TypeError` for the same reason.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    FrontsConfig
    """
    # Load the YAML
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}

    # The new format nests the fronts config under a top-level ``fronts:`` key
    fronts_section = raw.get('fronts')
    if fronts_section is None:
        raise ValueError(
            f"Config file '{path}' is missing the required top-level "
            f"'fronts:' section"
        )

    # Build the dataclasses; unknown/missing keys raise TypeError automatically
    return FrontsConfig(
        label=fronts_section['label'],
        finding=FindingConfig(**fronts_section['binary']),
        properties=PropertiesConfig(**fronts_section['properties']),
    )
