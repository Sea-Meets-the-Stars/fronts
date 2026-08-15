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

import pytest

from dbof.cli import generate_global, run_all_subsets, zarr_to_netcdf
from dbof.global_dataset_creation.config import default_output_folder
from dbof.global_dataset_creation.iterations import (
    date_to_run_id, prefix_to_filename_date,
)
from dbof.global_dataset_creation.subset_definitions import (
    expand_channels_with_suffixes, get_subset_definition, valid_subsets,
)

from fronts.llc import io as llc_io
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
    }
    monkeypatch.setattr(build_v5, "generate_global_dataset", s["generate"])
    monkeypatch.setattr(build_v5, "export_channels", s["export"])
    monkeypatch.setattr(build_v5, "find_gradb2_fronts", s["find"])
    monkeypatch.setattr(build_v5, "group_fronts", s["group"])
    monkeypatch.setattr(build_v5, "colocate_fronts", s["colocate"])
    return s


def test_step1_builds_only_the_subset_that_owns_gradb2(spies, surf_cfg):
    """Step 1 must not generate kinematic / frontogenesis / native_fields."""
    build_v5.main(1, surf_cfg)
    kw = spies["generate"].kwargs
    assert kw["subsets"] == ["frontal_structure"]
    assert kw["generate_only"] is True


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


def test_step4_generates_every_subset_and_colocates(spies, surf_cfg):
    """Step 4 is where the other subsets are finally paid for."""
    build_v5.main(4, surf_cfg)
    kw = spies["generate"].kwargs
    assert "subsets" not in kw or kw["subsets"] is None   # -> all active_subsets
    assert not kw.get("generate_only")
    assert len(spies["colocate"].calls) == 2              # one per date


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
    spies["generate"].calls.clear()
    build_v5.main(4, depth_cfg)
    assert spies["generate"].kwargs["ice_mask"] is False


def test_find_and_props_masks_are_independent(spies, surf_cfg):
    """surf_cfg: ice_mask_find false, ice_mask_props true."""
    build_v5.main(1, surf_cfg)
    assert spies["export"].calls[0][1]["ice_mask"] is False   # unmasked gradb2
    assert spies["generate"].kwargs["subsets"] == ["frontal_structure"]

    spies["generate"].calls.clear()
    build_v5.main(4, surf_cfg)
    assert spies["generate"].kwargs["ice_mask"] is True        # masked properties


def test_masking_gradb2_also_builds_icearea(spies, tmp_path):
    """The mask is read from icearea.zarr, so step 1 has to produce it."""
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
