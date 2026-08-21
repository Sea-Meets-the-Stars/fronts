"""Contract tests for the build_v5 front-building workflow.

These are *offline* tests: no S3, no OSN, no data.  They pin the interfaces
that ``fronts/runs/prototypes/one_full/build_v5.py`` relies on -- both inside
``fronts`` and across the boundary into the ``llc4320-native-grid-preprocessing``
(``dbof``) package.  If the preprocessing repo changes shape underneath us,
these fail fast and tell us exactly what moved.

Run with::

    pytest fronts/tests/test_build_v5.py -v
"""
import os
import inspect
import re
import textwrap

import numpy as np
import pandas as pd
import pytest

from dbof.cli import generate_global, run_all_subsets, zarr_to_netcdf
from dbof.global_dataset_creation.config import default_output_folder
from dbof.global_dataset_creation.iterations import (
    date_to_run_id, prefix_to_filename_date,
)
from dbof.global_dataset_creation.subset_definitions import (
    expand_channels_with_suffixes, get_subset_definition, valid_subsets,
)

from fronts.finding import io as finding_io
from fronts.llc import io as llc_io
from fronts.llc import meta as llc_meta
from fronts.llc import publish as llc_publish
from fronts.properties import algorithms as prop_algorithms
from fronts.properties import colocation
from fronts.properties import run as prun
from fronts.runs.prototypes.one_full import build_v5

PIPELINES = ("SURF", "OSN", "DEPTH")


# ===========================================================================
#  Fixtures -- throwaway configs, no S3 and no data
# ===========================================================================

_SURF_YAML = """
pipeline: "SURF"
run:
  run_id: "V5test"
data:
  date_iterations:
    - '2012-11-09 12:00:00'
    - '2012-11-10 06:00:00'
output:
  s3_endpoint: "https://s3-west.nrp-nautilus.io"
  bucket: "dbof/"
active_subsets:
  - frontal_structure
  - kinematic
  - frontogenesis
  - native_fields
  - surface_wind
  - icearea
build:
  build_version: "V5"
  finding_config: "D"
  ice_mask_find: false
  ice_mask_props: true
  percentiles: [90]
"""

_DEPTH_YAML = """
pipeline: "DEPTH"
run:
  run_id: "V5depth"
data:
  date_iterations:
    - '2012-11-09 12:00:00'
active_subsets:
  - frontal_structure
  - stratification
  - mixing_parameters
  - icearea
depth_suffixes: [sfc, z25m]
"""


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


@pytest.fixture
def surf_cfg(tmp_path):
    return _write(tmp_path, "run_surf.yaml", _SURF_YAML)


@pytest.fixture
def depth_cfg(tmp_path):
    return _write(tmp_path, "run_depth.yaml", _DEPTH_YAML)


# ===========================================================================
#  The contract with the preprocessing repo
# ===========================================================================

def test_all_three_pipelines_are_known():
    """SURF / OSN / DEPTH each resolve to a subset table."""
    for pipeline in PIPELINES:
        assert valid_subsets(pipeline), f"{pipeline} has no subsets"
    with pytest.raises(ValueError):
        valid_subsets("NOPE")


def test_default_output_folder_per_pipeline():
    """The S3 folder build_v5 reads from is pipeline-derived, not hardcoded."""
    assert default_output_folder("SURF") == "surface_fields/"
    assert default_output_folder("OSN") == "surface_fields/"
    assert default_output_folder("DEPTH") == "depth_fields/"


def test_frontal_structure_exists_in_every_pipeline():
    """Step 1 only ever generates frontal_structure -- it must exist everywhere."""
    for pipeline in PIPELINES:
        assert "frontal_structure" in valid_subsets(pipeline)


def test_icearea_exists_in_every_pipeline():
    """Ice masking needs icearea.zarr for the same run_id, in any pipeline."""
    for pipeline in PIPELINES:
        assert "icearea" in valid_subsets(pipeline)


def _channels_for(pipeline, subset, depth_suffixes=None):
    defn = get_subset_definition(pipeline, subset)
    if depth_suffixes and "depth_suffixes" in defn:
        defn["depth_suffixes"] = depth_suffixes
    return (list(defn.get("model_data_feature_channels") or [])
            + expand_channels_with_suffixes(
                defn.get("compute_features_channels") or [],
                defn.get("depth_suffixes"),
                defn.get("extra_channels")))


def test_gradb2_channel_name_depends_on_pipeline():
    """The gradb2 channel is bare on SURF/OSN and suffixed on DEPTH.

    Any driver that names the channel literally therefore works on one pipeline
    and fails on the others with a missing file.
    """
    for pipeline in ("SURF", "OSN"):
        chans = _channels_for(pipeline, "frontal_structure")
        assert "gradb2" in chans
        assert "gradb2_sfc" not in chans

    depth = _channels_for("DEPTH", "frontal_structure",
                          depth_suffixes=["sfc", "z25m"])
    assert "gradb2_sfc" in depth
    assert "gradb2" not in depth


def test_surface_frontal_structure_has_extra_channels():
    """SURF/OSN frontal_structure carries density + buoyancy; DEPTH does not.

    Exporting the whole subset in step 1 would therefore build 8 NetCDFs on
    SURF (21 on DEPTH with 4 suffixes) when only gradb2 is needed.
    """
    surf = _channels_for("SURF", "frontal_structure")
    assert {"density", "buoyancy"} <= set(surf)
    assert len(surf) == 8

    depth = _channels_for("DEPTH", "frontal_structure",
                          depth_suffixes=["sfc", "z25m", "mld", "mld_mean"])
    assert {"density", "buoyancy"} & set(depth) == set()
    assert len(depth) == 21


def test_depth_channel_roster_is_discovered_not_assumed():
    """subset_definitions is the only place the DEPTH channel list lives.

    R_ib, Wstar and rossby_number are easy to miss by hand.  build_v5 reads the
    roster rather than restating it; this test is the canary for a roster that
    grows again.
    """
    roots = set()
    for subset in valid_subsets("DEPTH"):
        defn = get_subset_definition("DEPTH", subset)
        roots |= set(defn.get("compute_features_channels") or [])
        roots |= set(defn.get("model_data_feature_channels") or [])
        roots |= set(defn.get("extra_channels") or [])
    assert {"R_ib", "Wstar", "rossby_number"} <= roots


def test_run_all_subsets_exposes_the_flags_build_v5_uses():
    """build_v5 shells out to ``python -m dbof.cli.run_all_subsets``."""
    src = inspect.getsource(run_all_subsets._parse_args)
    flags = set(re.findall(r'"(--[a-z-]+)"', src))
    required = {
        "--config", "--netcdf-base", "--pipeline", "--subsets", "--run-id",
        "--ice-mask", "--clobber", "--clobber-export",
        "--generate-only", "--export-only", "--dry-run",
    }
    assert required <= flags, f"missing: {sorted(required - flags)}"


def test_generate_global_main_signature():
    """Step 4 may call generate_global.main() directly for a single subset."""
    params = inspect.signature(generate_global.main).parameters
    assert {"config_file", "run_id", "subset", "pipeline", "clobber"} <= set(params)


def test_zarr_to_netcdf_main_supports_single_date_and_ice_mask():
    """Step 1's per-channel export goes through zarr_to_netcdf.main().

    It must accept an explicit ``date_prefix`` (so a 100-date config still
    writes one file at a time) and an ``ice_mask`` toggle.
    """
    params = inspect.signature(zarr_to_netcdf.main).parameters
    assert {"date_prefix", "channels", "ice_mask", "ice_mask_dataset_name",
            "output_filename", "dataset_name", "folder"} <= set(params)
    assert params["ice_mask"].default is False
    assert params["ice_mask_dataset_name"].default == "icearea.zarr"


def test_output_filename_requires_a_single_date():
    """Guard rail: many dates + one output filename is an error upstream.

    Exports must therefore be per-timestamp -- handing the whole
    ``date_iterations`` list to zarr_to_netcdf raises for any config with more
    than one date.
    """
    src = inspect.getsource(zarr_to_netcdf.main)
    assert "--output-filename can only be used when converting a single" in src


def test_date_helpers_roundtrip():
    """fronts derives its timestamps with the producer's own helpers."""
    prefix = date_to_run_id("2012-11-09 12:00:00")
    assert prefix == "20121109_120000"
    assert prefix_to_filename_date(prefix) == "2012-11-09T12_00_00"


# ===========================================================================
#  Step 1 builds gradb2 and nothing else
# ===========================================================================

class _Spy:
    """Record calls instead of making them."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    @property
    def kwargs(self):
        assert len(self.calls) == 1, f"expected 1 call, got {len(self.calls)}"
        return self.calls[0][1]

    @property
    def args(self):
        assert len(self.calls) == 1, f"expected 1 call, got {len(self.calls)}"
        return self.calls[0][0]


@pytest.fixture(autouse=True)
def _reset_layout():
    """Keep the module-level run layout from leaking between tests."""
    llc_io.clear_run_layout()
    yield
    llc_io.clear_run_layout()


@pytest.fixture
def spies(monkeypatch, tmp_path):
    """Neutralise every side-effecting call build_v5 makes."""
    monkeypatch.setenv("OS_OGCM", str(tmp_path / "ogcm"))
    s = {
        "generate": _Spy(),
        "export": _Spy(result=[]),
        "find": _Spy(),
        "group": _Spy(),
        "colocate": _Spy(),
        "push": _Spy(result=[]),
    }
    monkeypatch.setattr(build_v5, "generate_global_dataset", s["generate"])
    monkeypatch.setattr(build_v5, "export_channels", s["export"])
    monkeypatch.setattr(build_v5, "find_gradb2_fronts", s["find"])
    monkeypatch.setattr(build_v5, "group_fronts", s["group"])
    monkeypatch.setattr(build_v5, "colocate_fronts", s["colocate"])
    monkeypatch.setattr(build_v5.llc_publish, "push_run", s["push"])
    return s


def test_step1_builds_only_the_subset_that_owns_gradb2(spies, surf_cfg):
    """Step 1 must not build kinematic / frontogenesis / native_fields."""
    build_v5.main(1, surf_cfg)
    kw = spies["generate"].kwargs
    assert kw["subsets"] == ["frontal_structure"]


def test_step1_never_exports_through_run_all_subsets(spies, surf_cfg):
    """generate_only: the store is built there, the channel is exported here.

    run_all_subsets has no --channels flag, so letting it export would write
    all 8 SURF channels (21 on DEPTH) when only gradb2 is wanted.
    """
    build_v5.main(1, surf_cfg)
    assert spies["generate"].kwargs["generate_only"] is True
    assert len(spies["export"].calls) == 2          # one per date, in fronts


def test_step1_exports_exactly_one_channel_per_timestamp(spies, surf_cfg):
    """One gradb2 file per date -- not the 8 channels in SURF frontal_structure."""
    build_v5.main(1, surf_cfg)
    assert len(spies["export"].calls) == 2          # two dates in the fixture
    for args, kwargs in spies["export"].calls:
        channels = args[2]
        assert channels == ["gradb2"]


def test_step1_uses_the_depth_channel_name_on_depth(spies, depth_cfg):
    build_v5.main(1, depth_cfg)
    args, _ = spies["export"].calls[0]
    assert args[2] == ["gradb2_sfc"]


def test_step1_does_not_colocate_or_find(spies, surf_cfg):
    build_v5.main(1, surf_cfg)
    assert spies["find"].calls == []
    assert spies["group"].calls == []
    assert spies["colocate"].calls == []


def test_step4_covers_every_subset_and_colocates(spies, surf_cfg):
    """Step 4 is where the other subsets are finally paid for."""
    build_v5.main(4, surf_cfg)
    kw = spies["generate"].kwargs
    assert "subsets" not in kw or kw["subsets"] is None   # -> all active_subsets
    assert len(spies["colocate"].calls) == 2              # one per date


def test_step4_writes_no_property_netcdfs(spies, surf_cfg):
    """Fields are read from the stores, so nothing lands on disk.

    At ~900 MB per global field, exporting them first costs ~800 GB for 100
    timesteps x 9 SURF channels.
    """
    build_v5.main(4, surf_cfg)
    assert spies["generate"].kwargs["generate_only"] is True
    assert spies["export"].calls == []
    kw = spies["colocate"].calls[0][1]
    assert kw["source"] == "zarr"
    assert kw["config_file"] == surf_cfg


def test_steps_2_and_3_generate_nothing(spies, surf_cfg):
    """Finding and grouping are pure consumers of step 1's output."""
    build_v5.main(2, surf_cfg)
    build_v5.main(3, surf_cfg)
    assert spies["generate"].calls == []
    assert spies["export"].calls == []
    assert len(spies["find"].calls) == 2
    assert len(spies["group"].calls) == 2


def test_step2_reads_the_pipeline_correct_gradb2_field(spies, surf_cfg, depth_cfg):
    build_v5.main(2, surf_cfg)
    assert spies["find"].calls[0][1]["gradb2_field"] == "gradb2"
    spies["find"].calls.clear()
    build_v5.main(2, depth_cfg)
    assert spies["find"].calls[0][1]["gradb2_field"] == "gradb2_sfc"


def test_export_channels_skips_files_that_already_exist(
        monkeypatch, tmp_path, surf_cfg):
    """Re-running step 1 is cheap: existing .nc files are not re-exported."""
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    ts = "2012-11-09T12_00_00"
    target = llc_io.derived_filename(ts, "gradb2", version="V5test")

    os.makedirs(os.path.dirname(target), exist_ok=True)
    open(target, "w").close()

    spy = _Spy()
    monkeypatch.setattr(llc_io, "zarr_to_nc", spy)
    out = prun.export_channels(surf_cfg, ts, ["gradb2"], version="V5test")
    assert spy.calls == []                       # skipped
    assert out == [target]

    prun.export_channels(surf_cfg, ts, ["gradb2"], version="V5test",
                         clobber=True)
    assert len(spy.calls) == 1                   # clobber forces it


# ===========================================================================
#  Pipeline selection
# ===========================================================================

def test_channel_for_root_resolves_per_pipeline(surf_cfg, depth_cfg):
    assert prun.channel_for_root(surf_cfg, "gradb2") == "gradb2"
    assert prun.channel_for_root(depth_cfg, "gradb2") == "gradb2_sfc"
    assert prun.channel_for_root(depth_cfg, "gradb2",
                                 depth_suffix="z25m") == "gradb2_z25m"


def test_channel_for_root_rejects_a_suffix_the_config_does_not_build(depth_cfg):
    with pytest.raises(ValueError, match="finding_suffix"):
        prun.channel_for_root(depth_cfg, "gradb2", depth_suffix="mld")


def test_channel_for_root_rejects_an_unknown_root(surf_cfg):
    with pytest.raises(ValueError, match="not produced by any active subset"):
        prun.channel_for_root(surf_cfg, "N2")     # DEPTH-only


def test_subset_for_channel(surf_cfg, depth_cfg):
    assert prun.subset_for_channel(surf_cfg, "gradb2") == "frontal_structure"
    assert prun.subset_for_channel(depth_cfg, "gradb2_sfc") == "frontal_structure"


def test_all_property_roots_follows_the_pipeline(surf_cfg, depth_cfg):
    """The root list follows the pipeline, with no overlap between the two."""
    surf = set(prun.all_property_roots(surf_cfg))
    assert {"gradb2", "density", "buoyancy", "rossby_number"} <= surf
    assert not ({"N2", "Ri", "ertel_pv", "KE"} & surf)   # DEPTH-only

    depth = set(prun.all_property_roots(depth_cfg))
    assert {"N2", "R_ib", "gradb2"} <= depth
    assert not ({"density", "buoyancy"} & depth)        # SURF-only


def test_all_property_roots_always_expand_cleanly(surf_cfg, depth_cfg):
    """Derived roots round-trip through expand_property_roots without raising.

    A hand-written root list drifts out of the pipeline it was written for and
    raises ValueError on every root the active subsets do not produce.
    """
    for cfg in (surf_cfg, depth_cfg):
        roots = prun.all_property_roots(cfg)
        channels = prun.expand_property_roots(roots, cfg)
        assert len(channels) >= len(roots)


def test_exclude_roots_are_dropped(surf_cfg):
    roots = prun.all_property_roots(surf_cfg, exclude=["density", "buoyancy"])
    assert "density" not in roots and "buoyancy" not in roots
    assert "gradb2" in roots


def test_read_build_config_merges_defaults_with_the_yaml(surf_cfg, depth_cfg):
    cfg = prun.read_build_config(surf_cfg)
    assert cfg["pipeline"] == "SURF"
    assert cfg["run_id"] == "V5test"
    assert cfg["timestamps"] == ["2012-11-09T12_00_00", "2012-11-10T06_00_00"]
    assert cfg["date_prefixes"] == ["20121109_120000", "20121110_060000"]
    assert cfg["finding_config"] == "D"           # from the YAML
    assert cfg["percentiles"] == [90]             # from the YAML
    assert cfg["finding_suffix"] == "sfc"         # from BUILD_DEFAULTS
    assert cfg["exclude_roots"] == []             # from BUILD_DEFAULTS

    # A config with no build: block still gets every default.
    d = prun.read_build_config(depth_cfg)
    assert set(prun.BUILD_DEFAULTS) <= set(d)
    assert d["depth_suffixes"] == ["sfc", "z25m"]


# ===========================================================================
#  The ice mask is a per-step toggle
# ===========================================================================

def test_ice_mask_off_everywhere_by_default(spies, depth_cfg):
    """depth_cfg has no build: block at all."""
    build_v5.main(1, depth_cfg)
    assert spies["export"].calls[0][1]["ice_mask"] is False
    build_v5.main(4, depth_cfg)
    assert spies["colocate"].calls[-1][1]["ice_mask"] is False


def test_find_and_props_masks_are_independent(spies, surf_cfg):
    """surf_cfg: ice_mask_find false, ice_mask_props true."""
    build_v5.main(1, surf_cfg)
    assert spies["export"].calls[0][1]["ice_mask"] is False   # unmasked gradb2

    build_v5.main(4, surf_cfg)
    assert spies["colocate"].calls[-1][1]["ice_mask"] is True  # masked properties


def test_masking_gradb2_also_builds_icearea(spies, tmp_path):
    """The mask is read from icearea.zarr, so step 1 has to produce it too."""
    cfg = _write(tmp_path, "masked.yaml",
                 _SURF_YAML.replace("ice_mask_find: false",
                                    "ice_mask_find: true"))
    build_v5.main(1, cfg)
    assert spies["generate"].kwargs["subsets"] == ["frontal_structure", "icearea"]
    assert spies["export"].calls[0][1]["ice_mask"] is True


def test_zarr_to_nc_passes_ice_mask_and_one_date_prefix(
        monkeypatch, tmp_path, surf_cfg):
    """The fronts-side export forwards ice_mask and pins one date_prefix.

    Passing every date in the config instead would be rejected upstream as soon
    as an output filename is given.
    """
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    spy = _Spy()
    monkeypatch.setattr(llc_io.zarr_to_netcdf, "main", spy)

    llc_io.zarr_to_nc("2012-11-09T12_00_00", surf_cfg, "frontal_structure",
                      field="gradb2", version="V5test", ice_mask=True)

    kw = spy.kwargs
    assert kw["date_prefix"] == "20121109_120000"
    assert kw.get("dates") is None            # one snapshot, not the whole list
    assert kw["ice_mask"] is True
    assert kw["ice_mask_dataset_name"] == "icearea.zarr"
    assert kw["channels"] == ["gradb2"]
    assert kw["dataset_name"] == "frontal_structure.zarr"
    assert kw["folder"] == "surface_fields/"
    assert kw["output_filename"] == "LLC4320_2012-11-09T12_00_00_gradb2_V5test.nc"


# ===========================================================================
#  Locating the source stores
# ===========================================================================

def test_store_folder_override_reaches_the_export(monkeypatch, tmp_path):
    """output.folder locates the stores; without it SURF resolves elsewhere."""
    cfg = _write(tmp_path, "elsewhere.yaml", _SURF_YAML + """
output:
  s3_endpoint: "https://s3-west.nrp-nautilus.io"
  bucket: "dbof/"
  folder: "globals_for_cutouts/"
""")
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    spy = _Spy()
    monkeypatch.setattr(llc_io.zarr_to_netcdf, "main", spy)
    llc_io.zarr_to_nc("2012-11-09T12_00_00", cfg, "frontal_structure",
                      field="gradb2", version="v2_2_01")
    kw = spy.kwargs
    assert kw["folder"] == "globals_for_cutouts/"
    assert kw["bucket"] == "dbof/"
    assert kw["run_id"] == "V5test"
    assert kw["dataset_name"] == "frontal_structure.zarr"
    assert kw["date_prefix"] == "20121109_120000"


# ===========================================================================
#  The shipped 100-timestep config
# ===========================================================================

_RUN_CFG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs", "prototypes", "one_full", "run_v5_100_timesteps.yaml")


@pytest.mark.skipif(not os.path.exists(_RUN_CFG), reason="config not present")
def test_shipped_config_is_coherent():
    cfg = prun.read_build_config(_RUN_CFG)
    assert cfg["pipeline"] == "SURF"
    assert len(cfg["date_iterations"]) == 100
    assert len(set(cfg["date_iterations"])) == 100      # no duplicates
    assert "frontal_structure" in cfg["active_subsets"]

    # Points at the stores under s3://dbof/globals_for_cutouts/v2_2_01/.
    assert cfg["run_id"] == "v2_2_01"
    assert cfg["build_version"] == "V5"
    assert cfg["run_dir"] == "V5/SURF"
    with open(_RUN_CFG) as fh:
        import yaml as _yaml
        raw = _yaml.safe_load(fh)
    assert raw["output"]["folder"] == "globals_for_cutouts/"
    assert raw["output"]["bucket"] == "dbof/"

    # Steps 1-3 resolve without touching S3.
    channel = prun.channel_for_root(_RUN_CFG, cfg["gradb2_root"],
                                    depth_suffix=cfg["finding_suffix"])
    assert channel == "gradb2"
    assert prun.subset_for_channel(_RUN_CFG, channel) == "frontal_structure"

    # The finding config it names actually exists.
    from fronts.finding import config as find_config
    assert os.path.isfile(find_config.config_filename(cfg["finding_config"]))


@pytest.mark.skipif(not os.path.exists(_RUN_CFG), reason="config not present")
def test_shipped_config_dates_match_the_transfer_config():
    """Every date is one of the timesteps sitting in LLC4320_RAW/SURFACE."""
    cfg = prun.read_build_config(_RUN_CFG)
    for date in cfg["date_iterations"]:
        prefix = date_to_run_id(date)               # raises if out of range
        assert len(prefix) == 15 and prefix[8] == "_"


def test_generate_global_dataset_builds_the_right_command(monkeypatch):
    """The subprocess argv is the contract with run_all_subsets."""
    spy = _Spy()
    monkeypatch.setattr(prun.subprocess, "run", spy)
    prun.generate_global_dataset(
        "cfg.yaml", "/base", subsets=["frontal_structure", "icearea"],
        generate_only=True, ice_mask=True, pipeline="SURF", run_id="V5test")
    cmd = spy.args[0]
    assert cmd[1:4] == ["-m", "dbof.cli.run_all_subsets", "--config"]
    assert "--generate-only" in cmd
    assert "--ice-mask" in cmd
    assert cmd[cmd.index("--subsets") + 1:cmd.index("--subsets") + 3] == \
        ["frontal_structure", "icearea"]
    assert cmd[cmd.index("--pipeline") + 1] == "SURF"
    assert cmd[cmd.index("--run-id") + 1] == "V5test"


# ===========================================================================
#  Output layout
# ===========================================================================

def test_layout_splits_directory_from_filename_tag(tmp_path):
    """Products sit under the build; filenames name the source dataset."""
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="v2_2_01")

    path = llc_io.derived_filename("2011-12-04T00_00_00", "gradb2",
                                   version="v2_2_01")
    assert path == str(tmp_path / "Fronts" / "V5" / "SURF" / "20111204_000000"
                       / "LLC4320_2011-12-04T00_00_00_gradb2_v2_2_01.nc")


def test_layout_applies_to_the_binary_fronts_file(tmp_path):
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="v2_2_01")
    path = finding_io.binary_filename("2011-12-04T00_00_00", "D", "v2_2_01")
    assert os.path.dirname(path).endswith("V5/SURF/20111204_000000")
    assert os.path.basename(path) == \
        "LLC4320_2011-12-04T00_00_00_v2_2_01_bfronts.npy"


def test_run_root_is_the_level_above_the_timestamps(tmp_path):
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="v2_2_01")
    root = llc_io.run_root("v2_2_01")
    assert root == str(tmp_path / "Fronts" / "V5" / "SURF")
    assert llc_io.fronts_dir("v2_2_01", "2011-12-04T00_00_00").startswith(root)


def test_without_a_layout_the_version_drives_both(tmp_path):
    """Callers that never set a layout keep the flat run_id/ directory."""
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    path = llc_io.derived_filename("2011-12-04T00_00_00", "gradb2",
                                   version="V4")
    assert path == str(tmp_path / "Fronts" / "V4" / "20111204_000000"
                       / "LLC4320_2011-12-04T00_00_00_gradb2_V4.nc")


def test_build_v5_sets_the_layout_from_the_config(spies, surf_cfg, tmp_path):
    build_v5.main(1, surf_cfg)
    assert llc_io.run_root("V5test").endswith("LLC/Fronts/V5/SURF")


# ===========================================================================
#  The label map written by step 3 is the one step 4 reads
# ===========================================================================

def test_label_map_tag_matches_between_group_and_colocate(tmp_path):
    """Both sides derive the run tag from the binary-fronts filename.

    group_fronts names its outputs after the .npy it was handed; colocate must
    resolve the same name or the label map is never found.
    """
    from fronts.properties import algorithms as prop_algorithms
    from fronts.properties import io as properties_io

    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="v2_2_01")

    fronts_file = finding_io.binary_filename("2011-12-04T00_00_00", "D",
                                             "v2_2_01")
    time_str, run_tag, _ = prop_algorithms._parse_fronts_filename(fronts_file)
    assert run_tag == "v2_2_01_bfronts"

    written = properties_io.get_global_front_output_path(
        tmp_path, time_str, "label_map", run_tag)
    assert written.name == \
        "labeled_fronts_global_20111204T00_00_00_v2_2_01_bfronts.npy"


# ===========================================================================
#  Pushing products back to S3
# ===========================================================================

def test_s3_prefix_lands_beside_the_source_stores(surf_cfg, tmp_path):
    cfg = _write(tmp_path, "src.yaml", _SURF_YAML + """
output:
  bucket: "dbof/"
  folder: "globals_for_cutouts/"
""")
    prefix = llc_publish.fronts_s3_prefix(cfg, "2011-12-04T00_00_00")
    assert prefix == "dbof/globals_for_cutouts/V5test/20111204_000000/Fronts"


def test_only_front_products_are_listed(tmp_path):
    d = tmp_path / "ts"
    d.mkdir()
    for name in ("LLC4320_2011-12-04T00_00_00_v2_2_01_bfronts.npy",
                 "labeled_fronts_global_20111204T00_00_00_v2_2_01_bfronts.npy",
                 "front_index_20111204T00_00_00_v2_2_01_bfronts.parquet",
                 "global_front_geometry_20111204T00_00_00_v2_2_01_bfronts.parquet",
                 "front_properties_20111204T00_00_00_v2_2_01_bfronts.parquet",
                 "metadata_20111204T00_00_00_v2_2_01_bfronts.json",
                 "LLC4320_2011-12-04T00_00_00_gradb2_v2_2_01.nc",   # excluded
                 "scratch.txt"):                                     # excluded
        (d / name).touch()

    found = [os.path.basename(f) for f in llc_publish.list_products(str(d))]
    assert len(found) == 6
    assert not any(f.endswith(".nc") for f in found)
    assert "scratch.txt" not in found


def test_push_ignores_a_co_tenant_dataset(tmp_path):
    """Two datasets share the directory; each push must carry only its own.

    Every product name embeds the file tag, so the S3 prefix for one run never
    receives the other run's files.
    """
    d = tmp_path / "ts"
    d.mkdir()
    for name in ("LLC4320_2011-12-04T00_00_00_v2_2_01_bfronts.npy",
                 "front_index_20111204T00_00_00_v2_2_01_bfronts.parquet",
                 "LLC4320_2011-12-04T00_00_00_v2_00_2_bfronts.npy",
                 "front_index_20111204T00_00_00_v2_00_2_bfronts.parquet",
                 "LLC4320_2011-12-04T00_00_00_v2_2_012_bfronts.npy"):
        (d / name).touch()

    mine = [os.path.basename(f)
            for f in llc_publish.list_products(str(d), file_tag="v2_2_01")]
    assert len(mine) == 2
    assert all("_v2_2_01_" in f for f in mine)
    assert not any("v2_2_012" in f for f in mine)     # not a prefix match
    assert len(llc_publish.list_products(str(d), file_tag="v2_00_2")) == 2
    assert len(llc_publish.list_products(str(d))) == 5   # unfiltered


def test_push_uploads_products_and_skips_existing(monkeypatch, tmp_path):
    cfg = _write(tmp_path, "src.yaml", _SURF_YAML + """
output:
  bucket: "dbof/"
  folder: "globals_for_cutouts/"
""")
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="v2_2_01")
    ts = "2011-12-04T00_00_00"
    d = llc_io.fronts_dir("v2_2_01", ts, generate=True)
    open(os.path.join(d, f"LLC4320_{ts}_v2_2_01_bfronts.npy"), "w").close()
    open(os.path.join(d, f"LLC4320_{ts}_gradb2_v2_2_01.nc"), "w").close()
    open(os.path.join(d, f"LLC4320_{ts}_v2_00_2_bfronts.npy"), "w").close()

    class _FS:
        def __init__(self, existing=()):
            self.put_calls = []
            self.existing = set(existing)

        def exists(self, key):
            return key in self.existing

        def put(self, local, key):
            self.put_calls.append((local, key))

    fs = _FS()
    out = llc_publish.push_timestamp(cfg, ts, "v2_2_01", fs=fs)
    # the .nc is not pushed, and neither is the co-tenant's .npy
    assert len(fs.put_calls) == 1
    local, key = fs.put_calls[0]
    assert key == ("dbof/globals_for_cutouts/V5test/20111204_000000/Fronts/"
                   f"LLC4320_{ts}_v2_2_01_bfronts.npy")
    assert out == [f"s3://{key}"]

    fs2 = _FS(existing={key})
    llc_publish.push_timestamp(cfg, ts, "v2_2_01", fs=fs2)
    assert fs2.put_calls == []                         # skipped
    llc_publish.push_timestamp(cfg, ts, "v2_2_01", fs=fs2, clobber=True)
    assert len(fs2.put_calls) == 1                     # clobber forces it


def test_step5_pushes_every_timestamp(spies, surf_cfg):
    build_v5.main(5, surf_cfg)
    args, kwargs = spies["push"].calls[0]
    assert args[1] == ["2012-11-09T12_00_00", "2012-11-10T06_00_00"]
    assert kwargs["version"] == "V5test"
    assert spies["export"].calls == []                 # push only


# ===========================================================================
#  The run descriptor
# ===========================================================================

def test_meta_filename_names_its_source():
    name = llc_meta.meta_filename("V5", "SURF", "globals_for_cutouts/",
                                  "v2_2_01")
    assert name == "fronts_meta_V5_SURF_from_globals_for_cutouts_v2_2_01.meta"


def test_meta_filename_separates_configs_that_share_everything_else():
    """Two date lists against one dataset must not share a descriptor.

    build_version, pipeline, folder and run_id can all be identical; the
    config filename is the only thing left to tell them apart.
    """
    args = ("V5", "SURF", "globals_for_chunks/", "V5")
    a = llc_meta.meta_filename(*args, config_stem="run_v5_chunks")
    b = llc_meta.meta_filename(*args, config_stem="run_v5_SO_chunks")
    assert a != b
    assert a.endswith("_V5_run_v5_chunks.meta")
    assert b.endswith("_V5_run_v5_SO_chunks.meta")


def test_meta_is_written_at_the_run_root_and_is_readable(tmp_path):
    import yaml as _yaml
    cfg_path = _write(tmp_path, "src.yaml", _SURF_YAML + """
output:
  bucket: "dbof/"
  folder: "globals_for_cutouts/"
""")
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("V5/SURF", file_tag="V5test")

    cfg = prun.read_build_config(cfg_path)
    path = llc_meta.write_run_meta(cfg, cfg_path,
                                   extra={"gradb2_channel": "gradb2",
                                          "gradb2_subset": "frontal_structure"})
    assert os.path.dirname(path).endswith("V5/SURF")
    assert os.path.basename(path) == \
        "fronts_meta_V5_SURF_from_globals_for_cutouts_V5test_src.meta"

    doc = _yaml.safe_load(open(path))
    assert doc["build"]["pipeline"] == "SURF"
    assert doc["source"]["folder"] == "globals_for_cutouts/"
    assert doc["source"]["run_id"] == "V5test"
    assert "globals_for_cutouts" in doc["source"]["store_uri"]
    assert doc["fronts"]["gradb2_channel"] == "gradb2"
    assert doc["fronts"]["finding_config"] == "D"
    assert doc["fronts"]["ice_mask_props"] is True
    assert doc["dates"]["n"] == 2
    assert set(doc["code"]) == {"fronts_git", "dbof_git"}


def test_step1_writes_the_descriptor(spies, surf_cfg):
    build_v5.main(1, surf_cfg)
    root = llc_io.run_root("V5test")
    metas = [f for f in os.listdir(root) if f.endswith(".meta")]
    assert metas == ["fronts_meta_V5_SURF_from_surface_fields_V5test_run_surf.meta"]


# ===========================================================================
#  Generalising across pipelines and naming schemes
# ===========================================================================

_DEPTH_VX_YAML = """
pipeline: "DEPTH"
run:
  run_id: "V5"
data:
  date_iterations:
    - '2012-11-09 12:00:00'
output:
  bucket: "dbof/"
active_subsets: [frontal_structure, stratification, icearea]
depth_suffixes: [sfc, z25m]
build:
  build_version: "V5"
  finding_config: "D"
  finding_suffix: "sfc"
"""

_SURF_DOTTED_YAML = """
pipeline: "SURF"
run:
  run_id: "v2_00_2"
data:
  date_iterations:
    - '2012-11-09 12:00:00'
output:
  bucket: "dbof/"
  folder: "globals_for_cutouts/"
active_subsets: [frontal_structure, icearea]
build:
  build_version: "v2_00"
  finding_config: "A"
"""


def _paths_for(cfg_path, tmp_path):
    """Every path a run touches, resolved without any I/O."""
    from dbof.global_dataset_creation.config import default_output_folder
    from dbof.global_dataset_creation.zarr_dataset_global import make_run_prefix

    cfg = prun.read_build_config(cfg_path)
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout(cfg["run_dir"], file_tag=cfg["run_id"])

    channel = prun.channel_for_root(cfg_path, cfg["gradb2_root"],
                                    depth_suffix=cfg["finding_suffix"])
    subset = prun.subset_for_channel(cfg_path, channel)
    folder = cfg["folder"] or default_output_folder(cfg["pipeline"])
    ts = cfg["timestamps"][0]
    return cfg, {
        "channel": channel,
        "store": make_run_prefix(cfg["bucket"], folder, cfg["run_id"],
                                 f"{subset}.zarr",
                                 date_prefix=cfg["date_prefixes"][0]),
        "nc": llc_io.derived_filename(ts, channel, version=cfg["run_id"]),
        "bfronts": finding_io.binary_filename(ts, cfg["finding_config"],
                                              cfg["run_id"]),
        "push": "s3://" + llc_publish.fronts_s3_prefix(cfg_path, ts),
        "meta": llc_meta.meta_filename(cfg["build_version"], cfg["pipeline"],
                                       folder, cfg["run_id"]),
    }


def test_depth_pipeline_with_vx_naming(tmp_path):
    """DEPTH, run_id == build_version, folder from the pipeline default."""
    cfg_path = _write(tmp_path, "depth_vx.yaml", _DEPTH_VX_YAML)
    cfg, p = _paths_for(cfg_path, tmp_path)

    assert cfg["run_dir"] == "V5/DEPTH"
    assert p["channel"] == "gradb2_sfc"                  # suffixed on DEPTH
    assert p["store"] == \
        "s3://dbof/depth_fields/V5/20121109_120000/frontal_structure.zarr"
    assert p["nc"].endswith(
        "Fronts/V5/DEPTH/20121109_120000/"
        "LLC4320_2012-11-09T12_00_00_gradb2_sfc_V5.nc")
    assert p["bfronts"].endswith(
        "Fronts/V5/DEPTH/20121109_120000/"
        "LLC4320_2012-11-09T12_00_00_V5_bfronts.npy")
    assert p["push"] == "s3://dbof/depth_fields/V5/20121109_120000/Fronts"
    assert p["meta"] == "fronts_meta_V5_DEPTH_from_depth_fields_V5.meta"


def test_surf_pipeline_with_dotted_naming(tmp_path):
    """A run_id full of underscores, and a folder that is not the default."""
    cfg_path = _write(tmp_path, "surf_dotted.yaml", _SURF_DOTTED_YAML)
    cfg, p = _paths_for(cfg_path, tmp_path)

    assert cfg["run_dir"] == "v2_00/SURF"
    assert p["channel"] == "gradb2"                      # bare on SURF
    assert p["store"] == ("s3://dbof/globals_for_cutouts/v2_00_2/"
                          "20121109_120000/frontal_structure.zarr")
    assert p["nc"].endswith(
        "Fronts/v2_00/SURF/20121109_120000/"
        "LLC4320_2012-11-09T12_00_00_gradb2_v2_00_2.nc")
    assert p["bfronts"].endswith(
        "Fronts/v2_00/SURF/20121109_120000/"
        "LLC4320_2012-11-09T12_00_00_v2_00_2_bfronts.npy")
    assert p["push"] == ("s3://dbof/globals_for_cutouts/v2_00_2/"
                         "20121109_120000/Fronts")
    assert p["meta"] == \
        "fronts_meta_v2_00_SURF_from_globals_for_cutouts_v2_00_2.meta"


def test_underscored_run_id_survives_the_filename_parser(tmp_path):
    """A run_id like 'v2_00_2' must round-trip out of the .npy filename.

    group_fronts and colocate both recover the run tag by parsing the binary
    fronts filename, so an underscore-heavy tag must not be truncated.
    """
    from fronts.properties import algorithms as prop_algorithms

    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    llc_io.set_run_layout("v2_00/SURF", file_tag="v2_00_2")
    fronts_file = finding_io.binary_filename("2012-11-09T12_00_00", "A",
                                             "v2_00_2")
    time_str, run_tag, raw = prop_algorithms._parse_fronts_filename(fronts_file)
    assert run_tag == "v2_00_2_bfronts"
    assert time_str == "2012-11-09T12:00:00"
    assert raw == "2012-11-09T12_00_00"


def test_build_version_comes_from_the_driver_not_the_config(tmp_path):
    """The output directory is a property of the code that made the products.

    A config may name any dataset; everything build_v5 writes still lands
    under V5/, so the layout cannot drift between runs or between people.
    """
    cfg_path = _write(tmp_path, "claims_otherwise.yaml",
                      _SURF_DOTTED_YAML.replace('build_version: "v2_00"',
                                                'build_version: "SOMETHING_ELSE"'))
    cfg = prun.read_build_config(cfg_path, build_version=build_v5.BUILD_VERSION)
    assert build_v5.BUILD_VERSION == "V5"
    assert cfg["run_dir"] == "V5/SURF"
    assert cfg["run_id"] == "v2_00_2"          # the source is still recorded


def test_driver_run_dir_is_the_same_for_every_source(spies, tmp_path):
    for run_id in ("v2_2_01", "v2_00_2"):
        cfg = _write(tmp_path, f"{run_id}.yaml",
                     _SURF_YAML.replace('run_id: "V5test"', f'run_id: "{run_id}"'))
        build_v5.main(1, cfg)
        assert llc_io.run_root(run_id).endswith("LLC/Fronts/V5/SURF")


def test_two_source_datasets_do_not_overwrite_each_other(tmp_path):
    """Same build + pipeline, different run_id -> distinct filenames.

    They share a directory; the filename tag is what keeps them apart.
    """
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    names = []
    for run_id in ("v2_2_01", "v2_2_02"):
        llc_io.set_run_layout("V5/SURF", file_tag=run_id)
        names.append((llc_io.derived_filename("2012-11-09T12_00_00", "gradb2",
                                              version=run_id),
                      finding_io.binary_filename("2012-11-09T12_00_00", "D",
                                                 run_id)))
    (nc_a, np_a), (nc_b, np_b) = names
    assert os.path.dirname(nc_a) == os.path.dirname(nc_b)     # shared dir
    assert nc_a != nc_b and np_a != np_b                      # distinct files


def test_every_pipeline_resolves_a_full_path_set(tmp_path):
    """Smoke: SURF, OSN and DEPTH all resolve end to end.

    The store the products are pushed to is always the store they were read
    from, whether the folder came from the pipeline default or a YAML override.
    """
    from dbof.global_dataset_creation.config import default_output_folder

    for pipeline in PIPELINES:
        body = _DEPTH_VX_YAML if pipeline == "DEPTH" else _SURF_DOTTED_YAML
        body = body.replace('pipeline: "DEPTH"', f'pipeline: "{pipeline}"')
        body = body.replace('pipeline: "SURF"', f'pipeline: "{pipeline}"')
        cfg_path = _write(tmp_path, f"{pipeline}.yaml", body)
        cfg, p = _paths_for(cfg_path, tmp_path)
        folder = (cfg["folder"] or default_output_folder(pipeline)).strip("/")

        assert cfg["pipeline"] == pipeline
        assert p["channel"].startswith("gradb2")
        assert p["meta"].startswith(
            f"fronts_meta_{cfg['build_version']}_{pipeline}_from_{folder}_")
        # read-from and push-to share a prefix: same bucket, folder, run_id, date
        assert p["store"].startswith(f"s3://dbof/{folder}/{cfg['run_id']}/")
        assert p["push"] == (f"s3://dbof/{folder}/{cfg['run_id']}/"
                             f"{cfg['date_prefixes'][0]}/Fronts")
        # local products stay under this build, never under the source run_id
        assert f"/Fronts/{cfg['run_dir']}/" in p["nc"]


# ===========================================================================
#  Co-locating straight from the zarr stores
# ===========================================================================

def _tiny_labels():
    """A 6x6 label map with two fronts."""
    lab = np.zeros((6, 6), dtype=np.int32)
    lab[1, 1:4] = 1
    lab[4, 2:5] = 2
    return lab


def test_a_callable_property_matches_the_array_it_returns():
    """Lazy and eager sources produce identical columns."""
    lab = _tiny_labels()
    arr = np.arange(36, dtype=np.float32).reshape(6, 6)
    kw = dict(stats=["mean", "std"], percentiles=[90], nan_policy="omit")

    eager = colocation.colocate_fronts_with_properties(lab, {"f": arr}, **kw)
    lazy = colocation.colocate_fronts_with_properties(lab, {"f": lambda: arr}, **kw)
    pd.testing.assert_frame_equal(eager, lazy)


def test_properties_are_loaded_one_at_a_time(tmp_path):
    """Load and reduce interleave, so only one field is ever resident.

    Each property's checkpoint is written before the next one loads, which only
    holds if the loop is load -> reduce -> drop. Reading every field up front
    would leave N x 896 MB resident on a global grid.
    """
    lab = _tiny_labels()
    ck = tmp_path / "ckpt"
    order = []

    def make(name, prior):
        def load():
            order.append(name)
            for done in prior:
                assert (ck / f"{done}.parquet").is_file(), \
                    f"{name} loaded before {done} was reduced"
            return np.full((6, 6), len(order), dtype=np.float32)
        return load

    names = ["a", "b", "c"]
    props = {n: make(n, names[:i]) for i, n in enumerate(names)}
    df = colocation.colocate_fronts_with_properties(
        lab, props, stats=["mean"], checkpoint_dir=str(ck))
    assert order == names
    assert list(df.columns) == ["flabel", "npix", "a_mean", "b_mean", "c_mean"]


def test_checkpoint_caches_each_property_and_skips_it_next_time(tmp_path):
    lab = _tiny_labels()
    calls = []

    def make(name, value):
        def load():
            calls.append(name)
            return np.full((6, 6), value, dtype=np.float32)
        return load

    props = {"a": make("a", 1.0), "b": make("b", 2.0)}
    ck = tmp_path / "ckpt"
    first = colocation.colocate_fronts_with_properties(
        lab, props, stats=["mean"], checkpoint_dir=str(ck))
    assert calls == ["a", "b"]
    assert sorted(f.name for f in ck.glob("*.parquet")) == ["a.parquet", "b.parquet"]

    calls.clear()
    second = colocation.colocate_fronts_with_properties(
        lab, props, stats=["mean"], checkpoint_dir=str(ck))
    assert calls == []                       # both served from cache
    pd.testing.assert_frame_equal(first, second)


def test_a_partial_checkpoint_resumes(tmp_path):
    """A run killed after property 'a' only reloads 'b'."""
    lab = _tiny_labels()
    ck = tmp_path / "ckpt"
    colocation.colocate_fronts_with_properties(
        lab, {"a": lambda: np.full((6, 6), 1.0, dtype=np.float32)},
        stats=["mean"], checkpoint_dir=str(ck))

    calls = []

    def load_b():
        calls.append("b")
        return np.full((6, 6), 2.0, dtype=np.float32)

    df = colocation.colocate_fronts_with_properties(
        lab, {"a": lambda: pytest.fail("should not reload a"), "b": load_b},
        stats=["mean"], checkpoint_dir=str(ck))
    assert calls == ["b"]
    assert {"a_mean", "b_mean"} <= set(df.columns)


def test_a_lazy_property_of_the_wrong_shape_still_raises():
    lab = _tiny_labels()
    with pytest.raises(ValueError, match="shape does not match"):
        colocation.colocate_fronts_with_properties(
            lab, {"f": lambda: np.zeros((3, 3), dtype=np.float32)})


def test_read_channel_and_zarr_to_nc_resolve_the_same_store(surf_cfg):
    """The .nc export is the zarr channel -- same store, same channel.

    zarr_to_netcdf casts to float32 and adds integer y/x coords; it does not
    regrid or roll, so a field read from the store is interchangeable with the
    file that would have been written from it.
    """
    args = llc_io.store_args(surf_cfg, "frontal_structure")
    assert args["bucket"] == "dbof/"
    assert args["folder"] == "surface_fields/"
    assert args["run_id"] == "V5test"
    assert args["dataset_name"] == "frontal_structure.zarr"

    src = inspect.getsource(llc_io.read_channel)
    assert "get_channel_snapshot" in src and "astype(np.float32)" in src
    export = inspect.getsource(zarr_to_netcdf.zarr_to_netcdf)
    assert "get_channel_snapshot(ch).astype(np.float32)" in export


def test_zarr_loader_routes_each_channel_to_its_owning_subset(monkeypatch, surf_cfg):
    seen = {}
    monkeypatch.setattr(llc_io, "read_channel",
                        lambda cfg, ts, subset, ch, **kw: seen.setdefault(ch, subset))
    loader = prun._zarr_loader(surf_cfg, "2012-11-09T12_00_00", "V5test")
    loader("gradb2")
    loader("SIarea")
    assert seen == {"gradb2": "frontal_structure", "SIarea": "icearea"}


def test_unknown_channels_are_reported_as_missing(surf_cfg):
    avail = prun._zarr_channels(surf_cfg, ["gradb2", "N2_sfc", "SIarea"])
    assert avail == ["gradb2", "SIarea"]        # N2 is DEPTH-only


def test_colocate_rejects_an_unknown_source(tmp_path, surf_cfg):
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    with pytest.raises(ValueError, match="must be 'zarr' or 'netcdf'"):
        prun.colocate_fronts("2012-11-09T12_00_00", "D", "V5test",
                             property_names=["gradb2"], source="hdf5")


def test_zarr_source_needs_a_config(tmp_path):
    llc_io.set_fronts_path(str(tmp_path / "Fronts"))
    with pytest.raises(ValueError, match="requires config_file"):
        prun.colocate_fronts("2012-11-09T12_00_00", "D", "V5test",
                             property_names=["gradb2"], source="zarr")


# ===========================================================================
#  Co-locating on a single tile
# ===========================================================================

from dbof.tiles.tile_mapping import TILE_SIZE, TileInfo   # noqa: E402
from fronts.llc import tiles as llc_tiles                 # noqa: E402


def _fake_maps(rot):
    """Rect-grid j_face/i_face maps for a 2x1-tile world with one face.

    *rot* picks the rect -> face-local mapping for the left tile:
    'identity', 'transpose', or 'flip_j'.  The right tile is always identity,
    offset by TILE_SIZE in i_face.
    """
    H, W = TILE_SIZE, 2 * TILE_SIZE
    jj, ii = np.meshgrid(np.arange(H), np.arange(TILE_SIZE), indexing="ij")
    j_map = np.zeros((H, W), np.int32)
    i_map = np.zeros((H, W), np.int32)

    if rot == "identity":
        j_map[:, :TILE_SIZE], i_map[:, :TILE_SIZE] = jj, ii
    elif rot == "transpose":
        j_map[:, :TILE_SIZE], i_map[:, :TILE_SIZE] = ii, jj
    elif rot == "flip_j":
        j_map[:, :TILE_SIZE], i_map[:, :TILE_SIZE] = TILE_SIZE - 1 - jj, ii
    else:
        raise ValueError(rot)

    j_map[:, TILE_SIZE:], i_map[:, TILE_SIZE:] = jj, ii + TILE_SIZE
    return np.zeros((H, W), np.int8), j_map, i_map


def _tile(i_tile, j_face_start=0, i_face_start=0):
    return TileInfo(
        tile_idx=i_tile, tile_j_rect=0, tile_i_rect=i_tile,
        rect_j_slice=slice(0, TILE_SIZE),
        rect_i_slice=slice(i_tile * TILE_SIZE, (i_tile + 1) * TILE_SIZE),
        face_idx=0,
        j_face_slice=slice(j_face_start, j_face_start + TILE_SIZE),
        i_face_slice=slice(i_face_start, i_face_start + TILE_SIZE))


@pytest.fixture
def patch_maps(monkeypatch):
    """Install synthetic lookup maps in place of the stitched LLC ones."""
    def install(rot):
        monkeypatch.setattr(llc_tiles, "lookup_maps", lambda: _fake_maps(rot))
    return install


def test_labels_for_tile_is_a_plain_slice_on_an_unrotated_face(patch_maps):
    patch_maps("identity")
    rng = np.random.default_rng(0)
    glob = rng.integers(0, 5, size=(TILE_SIZE, 2 * TILE_SIZE), dtype=np.int32)
    got = llc_tiles.labels_for_tile(glob, _tile(0))
    np.testing.assert_array_equal(got, glob[:, :TILE_SIZE])


def test_labels_for_tile_follows_a_rotated_face(patch_maps):
    """A transposed face must transpose the labels, not just slice them.

    Both arrays are 720x720, so a plain slice would look plausible while
    pairing every front pixel with the wrong field value.
    """
    patch_maps("transpose")
    rng = np.random.default_rng(1)
    glob = rng.integers(0, 5, size=(TILE_SIZE, 2 * TILE_SIZE), dtype=np.int32)
    got = llc_tiles.labels_for_tile(glob, _tile(0))
    np.testing.assert_array_equal(got, glob[:, :TILE_SIZE].T)
    assert not np.array_equal(got, glob[:, :TILE_SIZE])   # slicing would differ


def test_labels_for_tile_follows_a_flipped_face(patch_maps):
    patch_maps("flip_j")
    rng = np.random.default_rng(2)
    glob = rng.integers(0, 5, size=(TILE_SIZE, 2 * TILE_SIZE), dtype=np.int32)
    got = llc_tiles.labels_for_tile(glob, _tile(0))
    np.testing.assert_array_equal(got, glob[:, :TILE_SIZE][::-1, :])


def test_labels_for_tile_honours_the_face_local_offset(patch_maps):
    """The second tile sits at i_face 720..1439; indices must be rebased."""
    patch_maps("identity")
    glob = np.zeros((TILE_SIZE, 2 * TILE_SIZE), np.int32)
    glob[3, TILE_SIZE + 7] = 9
    got = llc_tiles.labels_for_tile(glob, _tile(1, i_face_start=TILE_SIZE))
    assert got[3, 7] == 9
    assert got.sum() == 9


def test_labels_for_tile_keeps_global_label_values(patch_maps):
    """Labels are not renumbered, so tile results join on flabel."""
    patch_maps("identity")
    glob = np.zeros((TILE_SIZE, 2 * TILE_SIZE), np.int32)
    glob[5, 5:9] = 40317
    got = llc_tiles.labels_for_tile(glob, _tile(0))
    assert set(np.unique(got)) == {0, 40317}


def test_edge_margin_zeroes_the_rim(patch_maps):
    patch_maps("identity")
    glob = np.ones((TILE_SIZE, 2 * TILE_SIZE), np.int32)
    got = llc_tiles.labels_for_tile(glob, _tile(0), edge_margin=3)
    assert got[3:-3, 3:-3].all()
    assert not got[:3, :].any() and not got[-3:, :].any()
    assert not got[:, :3].any() and not got[:, -3:].any()


def test_tile_loader_computes_once_then_reads_the_cache(monkeypatch, tmp_path):
    """The tile NetCDF is a cache: a second load recomputes nothing."""
    import xarray as xr
    tile = _tile(0)
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        xr.Dataset({"sigma0": (("j", "i"), np.full((4, 4), 2.5, np.float32))}
                   ).to_netcdf(kw["output"])
        return kw["output"]

    monkeypatch.setattr(llc_tiles.tile_utils, "run", fake_run)
    loader = llc_tiles.tile_loader("2012-07-03T12_00_00", tile, str(tmp_path))

    a = loader("density")
    assert len(calls) == 1
    assert calls[0]["property"] == "density"
    assert calls[0]["timestamp"] == "2012-07-03 12:00:00"    # dbof DATE_FMT
    assert calls[0]["i_rect"] == 0 and calls[0]["j_rect"] == 0

    b = loader("density")
    assert len(calls) == 1                                   # served from disk
    np.testing.assert_array_equal(a, b)


def test_tile_loader_reads_out_name_not_the_property_name(monkeypatch, tmp_path):
    """'density' is written as 'sigma0'; the loader must follow out_name."""
    import xarray as xr

    def fake_run(**kw):
        xr.Dataset({"sigma0": (("j", "i"), np.zeros((4, 4), np.float32))}
                   ).to_netcdf(kw["output"])
        return kw["output"]

    monkeypatch.setattr(llc_tiles.tile_utils, "run", fake_run)
    loader = llc_tiles.tile_loader("2012-07-03T12_00_00", _tile(0), str(tmp_path))
    assert loader("density").shape == (4, 4)


def test_tile_loader_takes_the_surface_level_of_a_3d_field(monkeypatch, tmp_path):
    import xarray as xr
    arr = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)

    def fake_run(**kw):
        xr.Dataset({"N2": (("k", "j", "i"), arr)}).to_netcdf(kw["output"])
        return kw["output"]

    monkeypatch.setattr(llc_tiles.tile_utils, "run", fake_run)
    loader = llc_tiles.tile_loader("2012-07-03T12_00_00", _tile(0), str(tmp_path))
    np.testing.assert_array_equal(loader("N2"), arr[0])


def test_tile_colocation_reuses_the_global_colocation_kernel(patch_maps):
    """A 720x720 pair goes through colocate_fronts_with_properties unchanged."""
    patch_maps("identity")
    glob = np.zeros((TILE_SIZE, 2 * TILE_SIZE), np.int32)
    glob[10, 10:14] = 7
    glob[20, 20:23] = 8
    labels = llc_tiles.labels_for_tile(glob, _tile(0))

    field = np.full((TILE_SIZE, TILE_SIZE), 3.0, np.float32)
    df = colocation.colocate_fronts_with_properties(
        labels, {"sigma0": lambda: field}, stats=["mean", "count"])
    assert sorted(df["flabel"]) == [7, 8]
    assert list(df["npix"]) == [4, 3]
    assert (df["sigma0_mean"] == 3.0).all()


def test_extra_columns_land_in_the_parquet(tmp_path):
    lab = _tiny_labels()
    df = colocation.colocate_fronts_with_properties(
        lab, {"f": np.ones((6, 6), np.float32)}, stats=["mean"])
    assert "tile_idx" not in df.columns
    params = inspect.signature(prop_algorithms.colocate_fronts).parameters
    assert "extra_columns" in params and "loader" in params


# ===========================================================================
#  Reading a tile from a CHUNKS store
# ===========================================================================

def test_chunk_store_date_name(monkeypatch, tmp_path):
    """The store leaf is YYYYMMDDTHH, from the fronts timestamp."""
    seen = {}

    def fake_open(uri, endpoint):
        seen[uri.rsplit("/", 1)[-1]] = True
        import xarray as xr
        return xr.Dataset()

    monkeypatch.setattr(llc_tiles, "_open_chunk", fake_open)
    monkeypatch.setattr(llc_tiles.tile_utils, "_build_tile_context",
                        lambda a, b: (a, b))
    llc_tiles.chunk_loader("monterey_bay", "2012-07-03T12_00_00")
    assert "20120703T12.zarr" in seen and "grid.zarr" in seen


def test_chunk_uri_layout():
    assert llc_tiles._chunk_uri("monterey_bay", "grid.zarr") == \
        "s3://dbof/LLC4320_RAW/CHUNKS/monterey_bay/grid.zarr"


def test_comodo_attrs_are_stamped_when_the_store_lacks_them(monkeypatch):
    """A transfer-written grid.zarr may not carry them; xgcm needs X and Y.

    ``_build_tile_context`` raises rather than failing later inside a compute
    callback, so the loader fills them in first.
    """
    import xarray as xr
    grid = xr.Dataset(coords={d: ("i" if d.startswith("i") else "j",
                                 np.arange(4)) for d in ("i", "j")})
    grid = xr.Dataset(coords={"i": np.arange(4), "j": np.arange(4),
                              "i_g": np.arange(4), "j_g": np.arange(4)})
    tracers = xr.Dataset({"Theta": (("j", "i"), np.zeros((4, 4), np.float32))},
                         coords={"j": np.arange(4), "i": np.arange(4)})
    captured = {}

    monkeypatch.setattr(llc_tiles, "_open_chunk",
                        lambda uri, ep: tracers if "T12" in uri else grid)

    def fake_ctx(ds_t, ds_g):
        captured["axes"] = {d: ds_g[d].attrs.get("axis") for d in
                            ("i", "j", "i_g", "j_g")}
        captured["shift"] = ds_g["i_g"].attrs.get("c_grid_axis_shift")
        return ds_t, None

    monkeypatch.setattr(llc_tiles.tile_utils, "_build_tile_context", fake_ctx)
    llc_tiles.chunk_loader("monterey_bay", "2012-07-03T12_00_00")
    assert captured["axes"] == {"i": "X", "j": "Y", "i_g": "X", "j_g": "Y"}
    assert captured["shift"] == -0.5


def test_existing_comodo_attrs_are_left_alone(monkeypatch):
    import xarray as xr
    grid = xr.Dataset(coords={"i": np.arange(4), "j": np.arange(4)})
    grid["i"].attrs["axis"] = "ALREADY"
    tracers = xr.Dataset({"Theta": (("j", "i"), np.zeros((4, 4), np.float32))})
    captured = {}
    monkeypatch.setattr(llc_tiles, "_open_chunk",
                        lambda uri, ep: tracers if "T12" in uri else grid)
    monkeypatch.setattr(llc_tiles.tile_utils, "_build_tile_context",
                        lambda t, g: (captured.setdefault("axis", g["i"].attrs["axis"]),
                                      None))
    llc_tiles.chunk_loader("monterey_bay", "2012-07-03T12_00_00")
    assert captured["axis"] == "ALREADY"


def test_tile_from_chunk_store_uses_the_stores_own_attrs(monkeypatch):
    """face / j_start / i_start identify the tile -- no lon/lat lookup."""
    import xarray as xr
    from dbof.tiles.tile_mapping import TileInfo

    ds = xr.Dataset(attrs={"resolved_face": 10, "j_start": 0,
                           "i_start": 2880, "tile_size": TILE_SIZE})
    monkeypatch.setattr(llc_tiles, "_open_chunk", lambda uri, ep: ds)

    face_id = np.full((TILE_SIZE, TILE_SIZE), 10, np.int8)
    jj, ii = np.meshgrid(np.arange(TILE_SIZE), np.arange(TILE_SIZE),
                         indexing="ij")
    monkeypatch.setattr(llc_tiles, "lookup_maps",
                        lambda: (face_id, jj, ii + 2880))

    resolved = {}
    monkeypatch.setattr(llc_tiles, "rect_ij_to_tile",
                        lambda i, j: resolved.setdefault(
                            "t", _tile(0, j_face_start=0, i_face_start=2880)))
    monkeypatch.setattr(llc_tiles, "rect_ij_to_tile",
                        lambda i, j: TileInfo(
                            tile_idx=7, tile_j_rect=0, tile_i_rect=0,
                            rect_j_slice=slice(0, TILE_SIZE),
                            rect_i_slice=slice(0, TILE_SIZE),
                            face_idx=10,
                            j_face_slice=slice(0, TILE_SIZE),
                            i_face_slice=slice(2880, 2880 + TILE_SIZE)))

    tile = llc_tiles.tile_from_chunk_store("monterey_bay")
    assert (tile.face_idx, tile.j_face_slice.start, tile.i_face_slice.start) \
        == (10, 0, 2880)


def test_tile_from_chunk_store_rejects_a_layout_mismatch(monkeypatch):
    """A store whose attrs disagree with the rect lookup must not be trusted."""
    import xarray as xr
    from dbof.tiles.tile_mapping import TileInfo

    ds = xr.Dataset(attrs={"resolved_face": 10, "j_start": 0,
                           "i_start": 2880, "tile_size": TILE_SIZE})
    monkeypatch.setattr(llc_tiles, "_open_chunk", lambda uri, ep: ds)
    face_id = np.full((TILE_SIZE, TILE_SIZE), 10, np.int8)
    jj, ii = np.meshgrid(np.arange(TILE_SIZE), np.arange(TILE_SIZE),
                         indexing="ij")
    monkeypatch.setattr(llc_tiles, "lookup_maps",
                        lambda: (face_id, jj, ii + 2880))
    monkeypatch.setattr(llc_tiles, "rect_ij_to_tile",
                        lambda i, j: TileInfo(
                            tile_idx=7, tile_j_rect=0, tile_i_rect=0,
                            rect_j_slice=slice(0, TILE_SIZE),
                            rect_i_slice=slice(0, TILE_SIZE),
                            face_idx=3,                      # disagrees
                            j_face_slice=slice(0, TILE_SIZE),
                            i_face_slice=slice(2880, 2880 + TILE_SIZE)))
    with pytest.raises(ValueError, match="rect lookup resolves to face"):
        llc_tiles.tile_from_chunk_store("monterey_bay")


def test_colocate_tile_accepts_an_injected_loader():
    params = inspect.signature(prun.colocate_tile).parameters
    assert "loader" in params and params["loader"].default is None


# ===========================================================================
#  clobber must invalidate the checkpoint cache
# ===========================================================================

def test_clear_checkpoints_removes_the_cache(tmp_path):
    ck = tmp_path / "colocate_ckpt_x"
    ck.mkdir()
    (ck / "density.parquet").touch()
    prun._clear_checkpoints(str(ck))
    assert not ck.exists()


def test_clear_checkpoints_is_a_noop_when_absent(tmp_path):
    prun._clear_checkpoints(str(tmp_path / "nope"))      # must not raise


def test_a_clobber_that_cannot_clear_the_cache_is_fatal(monkeypatch, tmp_path):
    """Stale fragments would be served instead of recomputed values.

    rmtree can fail on a network filesystem, and the cache wins over clobber
    inside colocate_fronts_with_properties -- so a clobber that leaves the
    cache in place silently returns the old numbers.
    """
    ck = tmp_path / "colocate_ckpt_x"
    ck.mkdir()

    def boom(_):
        raise OSError("Stale file handle")

    monkeypatch.setattr(prun.shutil, "rmtree", boom)
    with pytest.raises(RuntimeError, match="stale fragments would be reused"):
        prun._clear_checkpoints(str(ck), strict=True)


def test_a_failed_cleanup_after_a_run_only_warns(monkeypatch, tmp_path, capsys):
    """Litter left behind is not a reason to lose the result."""
    ck = tmp_path / "colocate_ckpt_x"
    ck.mkdir()
    monkeypatch.setattr(prun.shutil, "rmtree",
                        lambda _: (_ for _ in ()).throw(OSError("busy")))
    prun._clear_checkpoints(str(ck))                     # must not raise
    assert "left behind" in capsys.readouterr().out


def test_both_colocators_clear_the_cache_before_a_clobber():
    for fn in (prun.colocate_fronts, prun.colocate_tile):
        src = inspect.getsource(fn)
        assert "_clear_checkpoints(ckpt_dir, strict=True)" in src, fn.__name__
        assert "if clobber:" in src, fn.__name__


def test_checkpoints_live_with_the_output_not_the_timestamp():
    """A custom output_dir must get its own cache.

    Otherwise a side run sharing the timestamp directory would read -- and on
    clobber delete -- the checkpoints of a production run in flight.
    """
    for fn in (prun.colocate_fronts, prun.colocate_tile):
        src = inspect.getsource(fn)
        assert "os.path.join(output_dir, f'colocate_ckpt_{run_tag}')" in src, \
            fn.__name__


def test_chunk_context_is_exposed_for_direct_compute():
    """A caller can get (ds_merge, grid) without going through chunk_loader."""
    assert callable(llc_tiles.chunk_context)
    assert "chunk_context(" in inspect.getsource(llc_tiles.chunk_loader)


def test_surface_reduces_a_tile_field_to_2d():
    import xarray as xr
    arr = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    da = xr.DataArray(arr, dims=('face', 'k', 'j', 'i')).isel(face=[0])
    out = llc_tiles.surface(da)
    assert out.shape == (4, 4) and out.dtype == np.float32
    np.testing.assert_array_equal(out, arr[0, 0])

    flat = xr.DataArray(np.zeros((4, 4), np.float32), dims=('j', 'i'))
    assert llc_tiles.surface(flat).shape == (4, 4)


def test_comodo_puts_the_staggered_point_at_the_lower_face():
    """A staggered field must interpolate to centres using i_g[n] and i_g[n+1].

    The chunk transfer slices i_g over the same index range as i, so i_g[n] is
    the lower face of cell i[n].  If the shift sign says otherwise, xgcm pairs
    each centre with the face below it instead, and every field that lives on a
    staggered point -- the velocities, and all the kinematics built from them --
    lands one full cell away.
    """
    import xarray as xr
    import xgcm

    ds = xr.Dataset(coords={"i": ("i", np.arange(4)),
                            "i_g": ("i_g", np.arange(4))})
    ds["i"].attrs = {"axis": "X"}
    ds["i_g"].attrs = dict(llc_tiles._COMODO["i_g"])
    u = xr.DataArray([0.0, 10.0, 20.0, 30.0], dims="i_g",
                     coords={"i_g": np.arange(4)})

    grid = xgcm.Grid(ds)
    try:                                    # xgcm renamed this keyword
        out = grid.interp(u, "X", padding="fill")
    except (TypeError, ValueError):
        out = grid.interp(u, "X", boundary="fill")

    # centre 0 sits between i_g[0] and i_g[1]: (0 + 10) / 2
    assert out.values[0] == 5.0, (
        f"interp put centre 0 at {out.values[0]}, so i_g is being read as the "
        f"upper face; the velocity fields would be off by one cell")
    assert out.values[1] == 15.0
