# CLAUDE.md

Guidance for AI agents working in this repo. Also read [.github/copilot-instructions.md](.github/copilot-instructions.md) for project overview, package layout, and code patterns. This file focuses on the non-obvious mechanics of building, testing, and contributing.

## What this repo is

**RMG-Py** = Reaction Mechanism Generator (chemical kinetics mechanism generator). Two installable packages: `rmgpy/` (mechanism generation) and `arkane/` (statistical mechanics from QM calculations). Heavy use of Cython for performance. Depends on a sibling repo **RMG-database** for thermo/kinetics data.

Entry points:
- `rmg.py <input.py>` — runs RMG on a Python-style input file (see `examples/rmg/minimal/input.py`)
- `Arkane.py <input.py>` — runs Arkane
- Both are also installed as console scripts (`rmg.py`, `Arkane.py`) via `setup.py`. They thin-wrap `rmgpy.__main__:main` / `arkane.__main__:main`.
- Scripts under `scripts/` are also installed (e.g. `simulate.py`, `diffModels.py`, `mergeModels.py`, `rmg2to3.py`).

## Setup

The conda env is the only supported install path. Python is pinned `>=3.9,<3.12`.

```bash
conda env create --file environment.yml   # creates env named `rmg_env`
conda activate rmg_env
git clone https://github.com/ReactionMechanismGenerator/RMG-database ../RMG-database
make install
```

`make install` runs `python utilities.py check-pydas` (which writes [rmgpy/solver/settings.pxi](rmgpy/solver/settings.pxi) — see Cython section), then `pip install --no-build-isolation -vv -e .`, then touches a `.installed` sentinel. Subsequent `make` invocations skip reinstall unless the sentinel is missing.

**Claude Code on the web**: the SessionStart hook [.claude/hooks/session-start.sh](.claude/hooks/session-start.sh) (registered in [.claude/settings.json](.claude/settings.json)) performs these steps automatically in remote sessions: it installs Miniforge to `/opt/miniforge` (and points conda at the sandbox proxy's CA bundle), creates `rmg_env`, clones RMG-database to `../RMG-database`, runs `make install` (or `make build` if already installed), installs `pytest-xdist` and `py-spy`, and puts `rmg_env` on the `PATH`. Every step is skipped when already done. The first run takes ~20 minutes (the environment download is several GB); later sessions reuse the cached container. If you change `environment.yml`, the existing environment is not updated automatically; run `conda env update --file environment.yml --name rmg_env` (or remove `/opt/miniforge/envs/rmg_env` to have the hook recreate it). To run it by hand: `CLAUDE_CODE_REMOTE=true .claude/hooks/session-start.sh`.

**Always keep [environment.yml](environment.yml) and [.conda/meta.yaml](.conda/meta.yaml) in sync** — both define runtime deps and CI builds from `meta.yaml` for the conda package.

Optional pieces:
- `./install_rms.sh` — installs ReactionMechanismSimulator (Julia-based reactor backend). Required for `rms*` reactor types in input files. Honors `RMS_INSTALLER={continuous,standard,developer}` and `RMS_BRANCH` (default `for_rmg`).
- `make q2dtor` — clones Q2DTor into `external/` for 2D rotor calculations in Arkane.

## Build / Cython

Cython modules are listed explicitly in `setup.py` `ext_modules`. **Some `.py` files are cythonized** (not just `.pyx`): e.g. `rmgpy/molecule/molecule.py`, `group.py`, `atomtype.py`, `rmgpy/species.py`, `rmgpy/reaction.py`, `rmgpy/quantity.py`, `rmgpy/constants.py`. If you edit one of these, **rebuild** — the `.so` is what gets imported, not the `.py`.

Workflow:
- `make build` — incremental in-place `setup.py build_ext --inplace`. Fast. Use this after editing `.pyx`/`.pxd`/cythonized `.py`.
- `make` (default `all`) — checks deps, ensures `.installed` sentinel, then `make build`. Safe go-to.
- `make clean` — removes `.so`, `.pyc`, generated `.c`, `build/`, and `.installed`. Also `pip uninstall`s the package.
- `make decython` — deletes most `.so` files (keeps `_statmech.so`, `quantity.so`, and `rmgpy/solver/*.so`) so pure Python is loaded for debugging. **Pure Python mode is not reliably tested**; expect breakage.

Cython conventions in this repo:
- Compile language level is Python 3.
- Pair every public `cdef class` / `cpdef` method with a `.pxd` declaration.
- New extension files **must be added to `ext_modules` in `setup.py`** or they will silently not be built.
- The DASPK/DASSL solver is selected at compile time via `rmgpy/solver/settings.pxi` (auto-written by `utilities.py check-pydas` from whatever PyDAS variant is installed). Do not commit changes to `settings.pxi`.
- macOS-specific: `setup.py` deduplicates `-Wl,-rpath` flags from sysconfig before invoking Cython, to work around an LC_RPATH issue with conda-forge's Python on darwin. Don't remove that block.

## Tests

Configured in [pytest.ini](pytest.ini): `testpaths = test`, `python_files = *Test.py`, `python_classes = *Test Test*`. Tests live under `test/` mirroring `rmgpy/` and `arkane/`.

Default pytest flags include `-s -vv --keep-duplicates` and coverage (`--cov=arkane --cov=rmgpy --cov-report html`). `test/regression/` is excluded.

Markers (from `pytest.ini`):
- `@pytest.mark.functional` — slower functional tests
- `@pytest.mark.database` — tests that require RMG-database to be cloned and loaded
- Unmarked = unit tests

Make targets:
```bash
make test            # unit tests only (excludes functional, database)
make test-functional
make test-database
make test-all        # everything
```

Run a subset directly (use `python -m pytest`, as the Makefile does: a bare `pytest` puts `test/` first on `sys.path`, where `test/rmgpy/` shadows the real `rmgpy` package and every import fails with `No module named 'rmgpy.molecule.graph'` etc.):
```bash
python -m pytest test/rmgpy/molecule/atomtypeTest.py
python -m pytest -k "test_pattern"
python -m pytest -m "functional"
```

`pytest-xdist` (`-n auto`) is supported but **incompatible with RMS/Julia** — only use when RMS is not installed. Some test classes rely on their methods running in order within one process (e.g. `TestEnlarge.test_enlarge_1_...` to `_4_...` in `modelTest.py`, `TestTreeGeneration` in `familyTest.py`, `TestMain` in `mainTest.py`), so they can fail under `-n`; rerun such failures serially before assuming a regression.

Two more pitfalls when running tests (or RMG/Arkane jobs) in parallel:
- **Set `OPENBLAS_NUM_THREADS=1`.** Each process's OpenBLAS starts one busy-waiting thread per core, and with several processes the machine is heavily oversubscribed. The Arkane pressure-dependence examples (many eigendecompositions of small matrices) then slow down by 10x or more: `examples/arkane/networks/CH2NH2_mse` takes ~50 s alone but took 9+ minutes under `pytest -n 4`, which makes `test_arkane_examples` appear to hang.
- **Tests that write output files into the working directory** (e.g. Cantera YAML conversions writing `chem-gas.yaml` / `chem_annotated.yaml` into the repo root) can collide under `pytest -n`; rerun such failures serially. (Arkane's symmetry calculations used to share `./scratch` in the same way; they now use a private temporary directory unless a scratch directory is given explicitly.)

`test/conftest.py` forces `multiprocessing.set_start_method('fork')` and silences OpenBabel error logging. Be aware of the `fork` start method when adding tests that touch multiprocessing.

### Regression tests

Separate from pytest. Each `test/regression/<name>/` has an `input.py`. CI runs `python rmg.py test/regression/<name>/input.py` and diffs core/edge models against artifacts produced on `main`. Locally you can reproduce a single one:
```bash
python rmg.py test/regression/superminimal/input.py
python scripts/checkModels.py ...   # (see .github/workflows/CI.yml for arg shape)
```
Adding a new regression test means editing the **two lists** in [.github/workflows/CI.yml](.github/workflows/CI.yml) (Execution + Comparison steps); the first PR will fail CI until baseline artifacts exist on `main`.

The `Makefile` also has `eg0`-`eg10` targets that copy example inputs into `testing/<name>/` and run `rmg.py` — useful for ad-hoc end-to-end smoke testing (`eg0` is fastest).

**RMG runs are not bit-reproducible between processes unless `PYTHONHASHSEED` is fixed**: some iteration orders depend on string hashing, so two runs of the same code and input can differ in edge reactions and kinetics. When checking that a change does not alter the generated model (e.g. for performance work), run both versions with `PYTHONHASHSEED=0` and compare `chemkin/chem.inp` / `chem_edge.inp` (and `scripts/checkModels.py`). Even then, the order of third-body collider efficiencies in the Chemkin file can differ, because `write_kinetics_entry` in `rmgpy/chemkin.pyx` sorts them by `id()` (memory address).

**Database cache**: if `RMG_DATABASE_CACHE` is set to a directory, `RMG.load_database()` pickles the prepared database there and reuses it in later jobs with the same settings ([rmgpy/rmg/database_cache.py](rmgpy/rmg/database_cache.py)), saving 20–30 s per job. The key covers the database files, the rmgpy `.py`/`.so` files and the input settings, so a rebuild invalidates it. When benchmarking database loading or timing whole runs, check whether the variable is set, since a warm cache hides the load cost. Anything stored in the prepared database must pickle **exactly**: a lossy `__reduce__` (e.g. one converting units, as quantities used to) makes cached runs differ from uncached ones. Several database classes (`ThermoDatabase`, `KineticsDatabase`, `SolvationDatabase`, ...) pickle an explicit list of attributes in `__reduce__`/`__setstate__`: when you add an attribute that is used after loading (or is set by `RMG.load_database()`), add it there too, or cached runs silently use its default (binding energies were lost this way). To check, load the database as a job does, round-trip it through `pickle`, and compare the attributes of each sub-database, and compare a run with a warm cache against one without.

**Shadow reactions in reaction generation**: `KineticsFamily._generate_reactions()` returns `ShadowReaction` placeholders (see [rmgpy/data/kinetics/common.py](rmgpy/data/kinetics/common.py)) for template mappings related to an earlier one by a symmetry of the reactant, when called with `compress_symmetric=True` (by `generate_reactions_from_families`, `calculate_degeneracy` and `add_reverse_attribute`). Only `find_degenerate_reactions()` handles them, and it removes them. If you change product generation, the checks in `_generate_reactions`, or `find_degenerate_reactions`, keep the two paths equivalent: set `rmgpy.data.kinetics.family.SYMMETRY_COMPRESSION = False` to compare against plain generation (`test/rmgpy/data/kinetics/symmetryCompressionTest.py` does this). Side effects on the shared reactant molecules (atom order from VF2 sorting, leftover atom labels) are observable downstream and must stay the same. The same switch also turns off `_ProductConnectivityFilter`, which skips mappings whose products cannot match the `products` requested from `_generate_reactions()` (it must only reject mappings that the final `same_species_lists(..., strict=False)` check would reject).

### Profiling

`py-spy record --native -f raw -o profile.txt -- python rmg.py input.py` profiles a run including the compiled Cython modules, without rebuilding them with profiling enabled (`pip install py-spy`). It can also attach to a running job with `-p <pid>`. Keep in mind that database loading is a fixed cost of roughly 20–30 s, which dominates short runs such as the regression tests; use a larger input to see how model generation scales.

## Linting / formatting / typing

There is **no configured linter, formatter, or type checker** in this repo (no `pyproject.toml`, `ruff.toml`, `.flake8`, `mypy.ini`, or `pre-commit` config). The only style guidance is "follow PEP 8 for new code, but don't churn existing code just for style." Don't run `black`/`ruff format`/`isort` over the tree as part of unrelated changes — diffs balloon and reviews stall.

## Database integration

RMG looks up `database.directory` in this order:
1. `database.load(path=...)` arg in code
2. `rmgrc` in cwd
3. `~/.rmg/rmgrc`
4. `rmgpy/rmgrc` (alongside the package)
5. Default: `../RMG-database/input` relative to RMG-Py source

Template: [rmgpy/rmgrc_template](rmgpy/rmgrc_template). Copy it (don't edit in place — it's overwritten on install). In CI, the database is checked out at the branch named in `RMG_DATABASE_BRANCH` (env var in [.github/workflows/CI.yml](.github/workflows/CI.yml)); change that line if your PR depends on an unmerged database branch.

## Conventions

- **Python API uses `snake_case`**.
- **Input file DSL keeps `camelCase`** (`thermoLibraries`, `simpleReactor`, `terminationConversion`, ...) for backward compatibility. When adding a new input keyword, follow camelCase and update [documentation/source/users/rmg/input.rst](documentation/source/users/rmg/input.rst).
- All source files require the MIT license header (template lives in [LICENSE.txt](LICENSE.txt); `python utilities.py update-headers` re-applies it across `.py`/`.pyx`/`.pxd` in `rmgpy/`, `scripts/`, and the root).
- Use `logging` not `print`.
- Don't reach for `__init__.py`-as-namespace imports across cython modules; use `cimport rmgpy.constants as constants` etc.
- Git commit messages should include a short summary (one line), followed by a blank line, then a more detailed description that explains the motivation and rationale for the change, so that a human code reviewer can understand without the context of the conversation.

## Documentation

Sphinx docs in `documentation/source/`. Build with `make documentation` (calls `make -C documentation html`). Output: `documentation/build/html/index.html`. Built with the standard `rmg_env` (no separate doc env).

When changing things, also update:
- **Input file syntax / new options** → [documentation/source/users/rmg/input.rst](documentation/source/users/rmg/input.rst) (this is treated as required by reviewers).
- **New public API** → ensure docstrings exist; add module to a toctree under `documentation/source/reference/` if the module is new. API docs are auto-generated via `sphinx.ext.autodoc`.
- **New user-facing feature** → mention in `documentation/source/users/rmg/features.rst` or a sibling `.rst`.
- **Behavior change** → relevant section of the user guide (`users/rmg/` or `users/arkane/`).

You should keep this documentation (CLAUDE.md) up to date as needed.
Always make changes to this file in a separate commit for clarity, and explain in detail to the user why changes were necessary.

The `gh-pages` branch hosts the live site; CI publishes on push to `main`.

## CI

- [.github/workflows/CI.yml](.github/workflows/CI.yml): build + tests on Linux (ubuntu-latest, all Python versions, with and without RMS) and macOS (latest Python only). Linux runs `make test-all`; other matrix entries run only unit tests. Regression job runs separately on `ubuntu-latest`.
- [.github/workflows/conda_build.yml](.github/workflows/conda_build.yml): builds the conda package from [.conda/meta.yaml](.conda/meta.yaml).
- [.github/workflows/docs.yml](.github/workflows/docs.yml): Sphinx build + publish to `gh-pages`.

## Quick gotchas

- **Edits to `.pyx`/`.pxd`/cythonized `.py` won't take effect until you rebuild** (`make build`). Mysterious unchanged behavior is almost always a stale `.so`.
- **Don't rebuild while an RMG job is running from the same checkout.** `make build` overwrites the in-place `.so` files that the running process has loaded, which can crash it.
- **`.so` files persist across branch switches.** When chasing a weird bug after a checkout, `make clean && make` before debugging.
- **Don't use `--no-verify` or skip Cython rebuilds** to make a commit go through; the underlying issue will resurface in CI.
- **Functional/database tests need RMG-database checked out** at a compatible branch in `../RMG-database`.
- **RMS reactor types in input files require Julia** — without `install_rms.sh` they'll fail at runtime, not import.
