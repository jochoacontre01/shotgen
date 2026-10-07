# AGENTS.md — Agent Guidelines & Instructions for `shotgen`

This document provides essential instructions, architectural rules, coding standards, and operational commands for AI coding agents interacting with the `shotgen` codebase.

---

## 1. Repository & Package Architecture

The repository is structured as a **two-package monorepo** designed to isolate conflicting native C/CUDA runtimes:

```text
shotgen/
├── AGENTS.md                          # Agent instructions (this file)
├── pyproject.toml                     # Monorepo workspace configuration
├── README.md                          # Repository overview
├── USER_GUIDE.md                      # Comprehensive architecture and user guide
├── packages/
│   ├── shotgen-core/                  # CPU / PyTorch / Deepwave / Kirchhoff / I/O
│   │   ├── pyproject.toml
│   │   └── src/shotgen/
│   │       ├── cli.py                 # Core CLI entrypoint
│   │       ├── geometry/              # Grid geometry and coordinate-index mappers
│   │       ├── io/                    # SEG-Y and HDF5 I/O routines (SegyIO, video generators)
│   │       ├── migration/             # Kirchhoff PSDM & PyTorch/Deepwave RTM
│   │       ├── models.py              # Geological velocity models (GeoModel)
│   │       ├── pylogger.py            # Unified logging utilities
│   │       ├── runner.py              # Subprocess launcher for isolated GPU runs
│   │       ├── sampleshot.py          # PyLops forward modeling & LoadShotRecord
│   │       └── utils.py               # Signal processing and validation helpers
│   └── shotgen-gpu/                   # Devito / OpenACC / NVHPC Stencils
│       ├── pyproject.toml
│       └── src/shotgen_gpu/
│           ├── cli.py                 # GPU CLI entrypoint (`shotgen-gpu`)
│           ├── config.py              # Device affinity & PureNvidiaCompiler overrides
│           └── engine/                # Devito acoustic wave solvers & operators
├── tests/                             # Pytest test suite
├── examples/                          # Reference scripts and workflows
├── assets/                            # Sample geological and SEG-Y input files
├── data/                              # Generated shot records and HDF5 storage
└── shotgen/                           # [DEPRECATED / LEGACY] Monolithic source tree
```

### Critical Architecture Rules for Agents
- **STRICTLY MODIFY `packages/` ONLY**:
  - All modern source code lives under `packages/shotgen-core/src/shotgen/` and `packages/shotgen-gpu/src/shotgen_gpu/`.
  - **NEVER** edit files in the root `shotgen/` folder. The root `shotgen/` directory is legacy/deprecated and slated for removal.
- **Physical Process & Runtime Isolation**:
  - `shotgen-core` and `shotgen-gpu` must **NEVER** be imported together in the same Python process.
  - Doing so creates symbol interposition conflicts between GNU OpenMP (`libgomp.so`) and NVIDIA OpenACC/OpenMP (`libnvomp.so`), leading to `RTLD_GLOBAL` deadlock and CUDA driver crashes.
  - When GPU forward modeling or Devito execution is needed from `shotgen-core`, dispatch it via the subprocess runner: `shotgen.runner.run_gpu_simulation_dict(...)` or the `shotgen-gpu` CLI.

---

## 2. Environment & Installation

The primary Python environment is the Conda `msc` environment (Python >= 3.11, default 3.12).

### Environment Activation & Editable Installation
Always ensure both packages are installed in editable mode:

```bash
# Activate the msc conda environment
conda activate msc

# From the repository root, install both packages in editable mode
pip install -e packages/shotgen-core -e packages/shotgen-gpu
```

### Verifying Environment
```bash
python -c "import shotgen; print('shotgen-core:', shotgen.__file__)"
python -c "import shotgen_gpu; print('shotgen-gpu:', shotgen_gpu.__file__)"
```

---

## 3. Testing & Verification Rules

### Golden Rule for Pytest
**Always run pytest targeting the `tests/` directory explicitly:**

```bash
# Correct:
pytest tests/
pytest tests/ -v -k "test_snr_config_parsing"

# INCORRECT:
pytest          # DO NOT run bare pytest — it discovers examples/ which require specific external data paths!
```

### Simulation & Hardware Constraints
- **Default verification runs**: Keep grid dimensions small during tests to avoid long runtimes and memory exhaustion (e.g., `nx <= 60`, `nz <= 40`, `time_samples <= 200`, `sources <= 2`, `receivers <= 6`).
- **GPU testing**: GPU tests and Devito/OpenACC executions may be run if NVIDIA hardware and drivers are detected, but keep shot counts (1–2 shots) and simulation duration minimal for quick smoke/regression checks.
- **Pytest cache**: On some WSL / mounted environments, write permissions to `.pytest_cache` might produce harmless warnings (`PytestCacheWarning`); pass `-o cache_dir=/tmp/.pytest_cache` if a clean cache is needed.

---

## 4. Coding Standards & Conventions

### 1. Logging (No Bare `print` statements)
Use the centralized `pylogger` module rather than `print()`:
```python
from shotgen.pylogger import get_logger, log_execution_time

logger = get_logger(__name__)

# Usage:
logger.info("Starting wavefield simulation...")
logger.warning("Velocity model requires smoothing.")
logger.error("Simulation grid out of bounds.")
```

### 2. Matplotlib & Headless Rendering
Prevent interactive window hangs in CLI or CI environments by ensuring safe backend configuration:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

### 3. Git Commits & Branch Naming
Follow Conventional Commits and standard branch naming:
- **Commit format**: `<type>(<scope>): <short description>`
  - Types: `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`
  - Examples:
    - `feat(migration): add random boundary support for RTM`
    - `fix(io): handle truncated trace headers in SegyIO`
    - `chore(deps): update devito boundary operator flags`
- **Branch naming**:
  - `feat/<feature-name>`
  - `fix/<bug-name>`
  - `chore/<chore-name>`

### 4. Multiprocessing & Multicore Resource Utilization
Prioritize multiprocessing over multithreading for CPU-heavy modeling and traveltime calculations:
- **Bypass the Python GIL**: Libraries like `skfmm` (Fast Marching Method) do not release the Python GIL during C execution. Always prioritize `multiprocessing` (using Linux-native `fork` context) over `threading` to saturate available CPU cores.
- **Worker Process Bounding**: Cap process worker pools (e.g., `min(8, os.cpu_count())`) to prevent memory over-allocation from worker C-heap buffers.
- **Mandatory `__main__` Guards**: All CLI scripts and entrypoints invoking multiprocessing or batch migration **must** guard execution inside `if __name__ == "__main__":` to prevent child processes from recursively re-executing workflows.
- **Active-Window Caching & In-Place Stacking**: For Kirchhoff PSDM, always use active-window lifecycle caching (evicting coordinates after their last required shot) and stream shot summation directly in-place (`output += shot_img`) to maintain a bounded memory footprint (~3-4 GB).

### 5. Migration Execution Logging & Run Logs
All migration entrypoints and workflows (including `examples/kirchhoff_migration.py` and `examples/rtm_from_data.py`) must persist execution logs directly in the target dataset directory:
- **Log Location & Naming Convention**:
  - The log file must reside in the active dataset directory (`<dataset_dir>/`).
  - Naming template: `<script_name>_%Y%m%d_%H%M%S.log` (e.g., `kirchhoff_migration_20261006_224500.log` or `rtm_from_data_20261006_224500.log`), where `%Y%m%d_%H%M%S` records the start date and time of the run.
- **Centralized Logging Setup**:
  - Initialize file logging immediately upon resolving the dataset path using `shotgen.pylogger.setup_file_logging(dataset_dir, script_name, logger=logger)`.
  - This attaches a standard `logging.FileHandler` to the logger hierarchy so all submodule operations (e.g., `shotgen.migration.kirchhoff`, `shotgen.io.segy`, `shotgen.models`) stream to both the console (via `TqdmLoggingHandler`) and the dataset log file simultaneously.
- **Strict Prohibition of Bare `print()`**:
  - All status updates, timing metrics, and array diagnostics must pass through `logger.info()`, `logger.warning()`, or `logger.error()` instead of bare `print()` statements to ensure they are captured in the log file.
- **Read-Only / Unwritable Directory Fallback**:
  - If the dataset directory is read-only or lacks write permissions, file logging must automatically fall back to `/tmp/<dataset_name>_<script_name>_%Y%m%d_%H%M%S.log` and issue a warning, preventing permission crashes during execution on shared or protected volumes.

---

## 5. Data Management & Safety

- **Asset Protection**: Do **not** overwrite or delete baseline test assets in `assets/` (e.g. `complex_graben.sgy`, `assets.zip`).
- **Generated Data in `data/`**: Synthetic shot records and simulation artifacts are stored in `data/` or temporary folders. Always use `tmp_path` fixtures in tests.
- **Never Commit Binary Artifacts**:
  - Do not commit `.h5`, `.sgy`, `.segy`, `.npy`, `.npz`, `.zip`, `.tar`, or large output `.png` files to git.
  - Verify `git status` before finishing any task to confirm no large files were staged.

---

## 6. Quick Reference: CLI Commands

### Run GPU Simulation via CLI (`shotgen-gpu`)
```bash
shotgen-gpu --config examples/test_config.json --output-dir data/sim_output
```

### Run Kirchhoff Pre-Stack Depth Migration (PSDM) Example
```bash
python examples/kirchhoff_migration.py
```

### Run Subprocess GPU Simulation Example
```bash
python examples/run_born_simulation.py
```
