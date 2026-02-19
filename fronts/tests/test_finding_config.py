import os
import tempfile

import pytest
import yaml

from fronts.finding import config


class TestConfigFilename:

    def test_default_path(self):
        fname = config.config_filename('A')
        assert fname.endswith('finding_config_A.yaml')
        assert 'finding' in fname and 'configs' in fname

    def test_custom_path(self, tmp_path):
        fname = config.config_filename('B', path=str(tmp_path))
        assert fname == os.path.join(str(tmp_path), 'finding_config_B.yaml')


class TestLoad:

    def test_load_config_A(self):
        """Load the real finding_config_A.yaml and check structure."""
        config_file = config.config_filename('A')
        cfg = config.load(config_file)

        assert cfg['label'] == 'A'
        assert isinstance(cfg['binary'], dict)
        assert cfg['binary']['window'] == 64
        assert cfg['binary']['threshold'] == 90
        assert cfg['binary']['thresh_mode'] == 'pool'
        assert cfg['binary']['thin'] is True

    def test_required_fields_present(self, tmp_path):
        """All required fields present should load without error."""
        data = {
            'label': 'test',
            'binary': {
                'window': 32,
                'threshold': 0.8,
                'thresh_mode': 'generic',
                'thin': False,
            },
        }
        cfg_file = tmp_path / 'test.yaml'
        cfg_file.write_text(yaml.dump(data))
        cfg = config.load(str(cfg_file))
        assert cfg['label'] == 'test'
        assert cfg['binary']['window'] == 32

    def test_missing_required_field(self, tmp_path):
        """Omitting a required field should raise ValueError."""
        data = {
            'label': 'test',
            'binary': {
                'window': 32,
                # missing threshold, thresh_mode, thin
            },
        }
        cfg_file = tmp_path / 'bad.yaml'
        cfg_file.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match='Missing required fields'):
            config.load(str(cfg_file))

    def test_missing_label(self, tmp_path):
        """Omitting the top-level label should raise ValueError."""
        data = {
            'binary': {
                'window': 32,
                'threshold': 0.8,
                'thresh_mode': 'generic',
                'thin': True,
            },
        }
        cfg_file = tmp_path / 'nolabel.yaml'
        cfg_file.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match='Missing required fields'):
            config.load(str(cfg_file))

    def test_unknown_top_level_field(self, tmp_path):
        """An unknown top-level field should raise ValueError."""
        data = {
            'label': 'test',
            'binary': {
                'window': 32,
                'threshold': 0.8,
                'thresh_mode': 'generic',
                'thin': True,
            },
            'bogus': 42,
        }
        cfg_file = tmp_path / 'unknown.yaml'
        cfg_file.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match='Unknown field'):
            config.load(str(cfg_file))

    def test_unknown_nested_field(self, tmp_path):
        """An unknown field inside a group should raise ValueError."""
        data = {
            'label': 'test',
            'binary': {
                'window': 32,
                'threshold': 0.8,
                'thresh_mode': 'generic',
                'thin': True,
                'fake_param': 99,
            },
        }
        cfg_file = tmp_path / 'unknown_nested.yaml'
        cfg_file.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="Unknown field: 'binary.fake_param'"):
            config.load(str(cfg_file))

    def test_wrong_dtype_top_level(self, tmp_path):
        """Wrong type for a top-level leaf field should raise ValueError."""
        data = {
            'label': 123,  # should be str
            'binary': {
                'window': 32,
                'threshold': 0.8,
                'thresh_mode': 'generic',
                'thin': True,
            },
        }
        cfg_file = tmp_path / 'badtype.yaml'
        cfg_file.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="invalid type"):
            config.load(str(cfg_file))

    def test_wrong_dtype_nested(self, tmp_path):
        """Wrong type for a nested field should raise ValueError."""
        data = {
            'label': 'test',
            'binary': {
                'window': 'big',  # should be int
                'threshold': 0.8,
                'thresh_mode': 'generic',
                'thin': True,
            },
        }
        cfg_file = tmp_path / 'badtype_nested.yaml'
        cfg_file.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="invalid type"):
            config.load(str(cfg_file))

    def test_optional_fields(self, tmp_path):
        """Optional fields should be accepted when present."""
        data = {
            'label': 'full',
            'binary': {
                'window': 64,
                'threshold': 90.0,
                'thresh_mode': 'pool',
                'thin': True,
                'dilate': False,
                'min_size': 7,
                'connectivity': 2,
            },
        }
        cfg_file = tmp_path / 'full.yaml'
        cfg_file.write_text(yaml.dump(data))
        cfg = config.load(str(cfg_file))
        assert cfg['binary']['dilate'] is False
        assert cfg['binary']['min_size'] == 7
        assert cfg['binary']['connectivity'] == 2

    def test_file_not_found(self):
        """A nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            config.load('/no/such/file.yaml')
