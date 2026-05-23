import dataclasses
import os
from importlib import resources

import pytest
import yaml

from fronts import config as fronts_config
from fronts.config.config import (
    FindingConfig,
    FrontsConfig,
    PropertiesConfig,
    config_filename,
    load_config,
)


# Real reference file in the new format
V3B_YAML = os.path.join(
    resources.files('fronts'),
    'runs', 'prototypes', 'one_full', 'testing_global_v3b.yaml',
)


class TestConfigFilename:

    def test_default_path(self):
        fname = config_filename('A')
        assert fname.endswith('finding_config_A.yaml')
        assert 'finding' in fname and 'configs' in fname

    def test_custom_path(self, tmp_path):
        fname = config_filename('B', path=str(tmp_path))
        assert fname == os.path.join(str(tmp_path), 'finding_config_B.yaml')


class TestLoadV3B:
    """Loader against the real new-format YAML (testing_global_v3b.yaml)."""

    def test_returns_frontsconfig(self):
        cfg = load_config(V3B_YAML)
        assert isinstance(cfg, FrontsConfig)
        assert isinstance(cfg.finding, FindingConfig)
        assert isinstance(cfg.properties, PropertiesConfig)

    def test_label_and_finding(self):
        cfg = load_config(V3B_YAML)
        assert cfg.label == 'D'
        assert cfg.finding.window == 64
        assert cfg.finding.threshold == 85
        assert cfg.finding.thresh_mode == 'pool'
        assert cfg.finding.thin is False
        assert cfg.finding.sharpen is True
        assert cfg.finding.despur is True
        assert cfg.finding.Lspur == 10
        assert cfg.finding.dilate is False
        assert cfg.finding.min_size == 7
        assert cfg.finding.connectivity == 2

    def test_properties(self):
        cfg = load_config(V3B_YAML)
        assert cfg.properties.stats == ['mean', 'std', 'median']
        assert cfg.properties.percentiles == [10, 90]
        assert cfg.properties.min_npix == 5
        assert cfg.properties.nan_policy == 'omit'
        assert cfg.properties.dilation_radius == 2


class TestFrozen:
    """All three dataclasses must be immutable."""

    def test_findingconfig_frozen(self):
        fc = FindingConfig(window=8, threshold=1.0, thresh_mode='pool',
                           thin=True, sharpen=False, despur=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fc.window = 16

    def test_propertiesconfig_frozen(self):
        pc = PropertiesConfig(stats=['mean'], percentiles=[50], min_npix=1,
                              nan_policy='omit', dilation_radius=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            pc.min_npix = 2

    def test_frontsconfig_frozen(self):
        fc = FindingConfig(window=8, threshold=1.0, thresh_mode='pool',
                           thin=True, sharpen=False, despur=False)
        pc = PropertiesConfig(stats=['mean'], percentiles=[50], min_npix=1,
                              nan_policy='omit', dilation_radius=0)
        top = FrontsConfig(label='X', finding=fc, properties=pc)
        with pytest.raises(dataclasses.FrozenInstanceError):
            top.label = 'Y'


class TestLoadErrors:

    def _write(self, tmp_path, payload):
        p = tmp_path / 'cfg.yaml'
        p.write_text(yaml.dump(payload))
        return str(p)

    def _minimal(self):
        # Smallest payload the loader will accept
        return {
            'fronts': {
                'label': 'T',
                'binary': {
                    'window': 32,
                    'threshold': 80,
                    'thresh_mode': 'pool',
                    'thin': True,
                    'sharpen': False,
                    'despur': False,
                },
                'properties': {
                    'stats': ['mean'],
                    'percentiles': [50],
                    'min_npix': 1,
                    'nan_policy': 'omit',
                    'dilation_radius': 0,
                },
            },
        }

    def test_minimal_payload_loads(self, tmp_path):
        cfg = load_config(self._write(tmp_path, self._minimal()))
        assert cfg.label == 'T'
        assert cfg.finding.window == 32
        assert cfg.properties.min_npix == 1

    def test_missing_fronts_section(self, tmp_path):
        with pytest.raises(ValueError, match="'fronts:' section"):
            load_config(self._write(tmp_path, {'other': 1}))

    def test_missing_required_binary_field(self, tmp_path):
        payload = self._minimal()
        del payload['fronts']['binary']['threshold']
        with pytest.raises(TypeError):
            load_config(self._write(tmp_path, payload))

    def test_missing_required_property_field(self, tmp_path):
        payload = self._minimal()
        del payload['fronts']['properties']['min_npix']
        with pytest.raises(TypeError):
            load_config(self._write(tmp_path, payload))

    def test_unknown_binary_field(self, tmp_path):
        payload = self._minimal()
        payload['fronts']['binary']['bogus'] = 1
        with pytest.raises(TypeError):
            load_config(self._write(tmp_path, payload))

    def test_unknown_property_field(self, tmp_path):
        payload = self._minimal()
        payload['fronts']['properties']['bogus'] = 1
        with pytest.raises(TypeError):
            load_config(self._write(tmp_path, payload))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config('/no/such/file.yaml')


class TestPackageReExports:
    """Public surface lives on `fronts.config`."""

    def test_reexports(self):
        assert fronts_config.FrontsConfig is FrontsConfig
        assert fronts_config.FindingConfig is FindingConfig
        assert fronts_config.PropertiesConfig is PropertiesConfig
        assert fronts_config.load_config is load_config
        assert fronts_config.config_filename is config_filename
