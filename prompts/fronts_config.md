# Config

## Goals

This file will guide the configuration code for fronts.  The configuration for parameters for front finding and properties will be stored in YAML files.  The code will load the YAML files and return a frozen class with the parameters.  This models the config.py module in the llc4320-native-grid-preprocessing repository.

## Code

Here are guidelines for writing code: 

- Use Python
- If you need to run Python code, use the "ocean14" environment of conda.
- Place imports at the top of the file.
- Add inline comments to explain the effort
- Reuse existing code when possible, throughout this fronts repository
- Use exisisting I/O methods when possible, throughout this fronts repository.  These are in a number of io.py modules.
- Use methods, not classes

## Refactor

The code currently uses a dict datamodel located in fronts/finding/config.py as finding_dmodel.  This is used to validate the YAML files.  Please refactor the code to use a frozen class with the parameters.  This models the config.py module in the llc4320-native-grid-preprocessing repository.  Here are additional specifics:

- The top-level class should be named FrontsConfig
- There should be one subclass for front finding parameters
- There should be one subclass for front properties parameters
- All classes should have a frozen=True attribute
- Place the classes in fronts/config/config.py

### Plan

Refactor the dict-based `finding_dmodel` validator into a frozen-dataclass config
modeled on
[llc4320-native-grid-preprocessing/src/dbof/dataset_creation/config.py](../../llc4320-native-grid-preprocessing/src/dbof/dataset_creation/config.py).

**1. New module: `fronts/config/config.py`**

Three `@dataclass(frozen=True)` classes:

- `FindingConfig` — mirrors today's `binary:` YAML section.
  - Required: `window: int`, `threshold: float`, `thresh_mode: str`, `thin: bool`,
    `sharpen: bool`, `despur: bool`.
  - Optional (defaults match current YAMLs): `Lspur: Optional[int] = None`,
    `dilate: bool = False`, `min_size: int = 0`, `connectivity: int = 2`.
- `PropertiesConfig` — mirrors today's `properties:` YAML section.
  - Required: `stats: List[str]`, `percentiles: List[int]`, `min_npix: int`,
    `nan_policy: str`, `dilation_radius: int`.
- `FrontsConfig` — top-level container.
  - `label: str` (top-level metadata, used to build output filenames).
  - `finding: FindingConfig`
  - `properties: Optional[PropertiesConfig] = None` (some current YAMLs omit it —
    see Clarifications).

Loader:

```python
def load_config(path: str) -> FrontsConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return FrontsConfig(
        label=raw["label"],
        finding=FindingConfig(**raw.get("binary", {})),
        properties=(PropertiesConfig(**raw["properties"])
                    if "properties" in raw else None),
    )
```

Type validation comes "for free" from dataclass `__init__` (unknown keys → `TypeError`),
matching the llc4320 pattern.  Add `__post_init__` range checks where it adds value
(e.g. `window > 0`, `0 <= threshold <= 100` or whatever the current implicit contract
is — keep these conservative so we don't break existing YAMLs).

Also move `config_filename(config_label, path=None)` into this module unchanged — it
is independent of the dmodel and is used by every call site.

**2. Package init: `fronts/config/__init__.py`**

Re-export the public surface so callers can do
`from fronts import config as fronts_config`:

```python
from fronts.config.config import (
    FrontsConfig, FindingConfig, PropertiesConfig,
    load_config, config_filename,
)
```

**3. Update call sites**

Replace `from fronts.finding import config as find_config` and the
`cdict['binary']['window']`-style dict access with attribute access on the
dataclass.  Files:

- [fronts/finding/run.py](../fronts/finding/run.py) (line 15, 45-46)
- [fronts/properties/run.py](../fronts/properties/run.py) — `colocate_fronts` currently
  references an undefined `cdict` (lines 87-91); this refactor is the right place
  to wire a real `FrontsConfig` (or just `PropertiesConfig`) parameter through.
- [fronts/runs/prototypes/finding/build_v1.py](../fronts/runs/prototypes/finding/build_v1.py) (lines 16, 84-85)
- [fronts/runs/prototypes/finding/explore_hyper.py](../fronts/runs/prototypes/finding/explore_hyper.py) (lines 9, 59-60, 92-93, 126-127)
- [dev/build_v1_testing.py](../dev/build_v1_testing.py) (lines 22, 101-102)

**4. Tests**

Rewrite [fronts/tests/test_finding_config.py](../fronts/tests/test_finding_config.py)
against the new API:

- `test_config_filename_*` → keep, point at new module.
- `test_load_config_A` → assert `cfg.label == 'A'`, `cfg.finding.window == 64`, etc.
- `test_missing_required_field` → assert `TypeError` (from dataclass) instead of
  `ValueError`.
- `test_unknown_*_field` → assert `TypeError` (unexpected kwarg).
- `test_wrong_dtype_*` → keep, possibly relaxed since dataclasses don't type-check
  by default; add explicit `isinstance` checks in `__post_init__` if we want the
  current strictness.

**5. Old module: `fronts/finding/config.py`**

Delete it once call sites are migrated (no backwards-compat shim per the project's
"no compatibility hacks" guideline — see Clarifications to confirm).

**6. Order of work**

1. Add `fronts/config/config.py` + `__init__.py` with the new classes and loader.
2. Add new tests; run `pytest fronts/tests/test_finding_config.py` until green.
3. Migrate call sites one file at a time, running its smoke path where possible.
4. Delete `fronts/finding/config.py`.
5. Spot-check by loading each of `finding_config_{A,B,C,D,Z}.yaml` and
   `runs/prototypes/one_full/testing_global_v3b.yaml` (the latter has a `fronts:`
   sub-section — see Clarifications).

### Clarifications

1. **Existing YAMLs without a `properties:` section.** `finding_config_A.yaml`,
   `B.yaml`, `Z.yaml` (and presumably `C`, `D`) only define `label` + `binary`, but
   today's `finding_dmodel['required']` lists the properties keys
   (`stats`, `percentiles`, `min_npix`, `nan_policy`, `dilation_radius`) as required —
   which means `load()` *should* be raising on these files but evidently isn't being
   exercised that way. **Question:** should `PropertiesConfig` be optional on
   `FrontsConfig` (my current plan), or do we want to update every YAML to include
   a properties block and keep it required?

Properties are no longer optional.  But you do not need to modify those YAML files.

2. **Nested YAML key `fronts:` in `testing_global_v3b.yaml`.** That file embeds the
   front config under a top-level `fronts:` key (alongside dbof's `run`, `data`,
   `output`, etc.). Should `load_config(path)` understand both shapes (root-level
   *and* `fronts:` sub-section), or should the caller pull out the `fronts:` dict
   and pass it in some other way (e.g. a `from_dict()` constructor)?

The testing_global_v3b.yaml file is the new format.  The loader will need to be able to parse it.

3. **Subclass naming.** I've proposed `FindingConfig` (for the `binary:` block) and
   `PropertiesConfig`. Alternatives: `BinaryConfig`/`PropertiesConfig` (matches YAML
   keys exactly) or `FrontFindingConfig`/`FrontPropertiesConfig` (more explicit).
   Preference?

FindingsConfig and PropertiesConfig are good.

4. **Old `fronts/finding/config.py`.** Delete outright after migration, or keep a
   one-line shim that re-exports `config_filename` for backwards-compat with
   downstream notebooks?

Do not delete. I will deal with it.

5. **`__post_init__` validation.** The current dmodel validates dtypes (and would
   reject e.g. `window: 'big'`). Dataclasses don't enforce types at runtime.
   Should I add `isinstance` checks in `__post_init__` to preserve today's behavior,
   or accept the looser contract (consistent with the llc4320 reference, which only
   validates value ranges, not types)?

We will not enforce types.

6. **`colocate_fronts` in `properties/run.py`.** It references an undefined `cdict`
   (lines 87-91) — looks like dead code or a half-finished refactor. Is wiring a
   real `FrontsConfig` (or just `PropertiesConfig`) parameter through here in scope
   for this refactor, or should I leave that bug alone?

Leave that bug alone.  I will fix it.

## Prompts

### Develop

1. Read this doc.  Plan the refactoring described in the Refactor section and write your plan under the Plan section.  If you have any questions, add them under the Clarifications section.

### Perform

1. Read this doc.  Perform the refactoring described in the Refactor section.  Be sure to read my answers in the Clarifications section.