# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-07-28

### Overview & Architecture
This release restructures `shotgen` into a clean monorepo architecture, splitting core modeling/processing routines from GPU acceleration backends. It fixes severe symbol collisions (such as `libgomp` crashes) and PyTorch / Devito GPU execution deadlocks, enabling robust OpenACC GPU offloading on NVIDIA GPUs alongside flexible CPU simulation.

### Added
- **Monorepo Structure**: Split into `packages/shotgen-core` (core modeling, geometry, IO, migration) and `packages/shotgen-gpu` (GPU execution engines and compilers).
- **Explicit `device` Selection**: Added a unified `device="cpu"` / `device="gpu"` parameter across modeling and migration APIs (`ShotRecord`, `ReverseTimeMigration`, `KirchhoffMigration`).
- **NVIDIA Pure OpenACC JIT Compiler**: `PureNvidiaCompiler` in `shotgen-gpu` strips out `-mp` (OpenMP) and `-gpu=pinned` flags when calling `nvc++`, preventing GNU `libgomp` symbol collisions during Devito JIT compilation.
- **Lazy Symbol & Dependency Isolation**: Deferred PyTorch and Deepwave imports until GPU routines are explicitly called, avoiding premature initialization of conflicting CUDA/GOMP runtimes.

### Changed
- **Package Layout**: Relocated core code from `shotgen/` to `packages/shotgen-core/src/shotgen/` and GPU utilities to `packages/shotgen-gpu/src/shotgen_gpu/`.
- **Test Suite Updates**: Updated test fixtures to dynamically detect asset locations (e.g., `marine_overthrust_3d.segy`) and safely skip tests when optional benchmark datasets are missing. Explicitly specified `device="cpu"` in migration unit tests.

### Fixed
- **GPU Deadlock on Main Thread**: Resolved OpenACC IPC process deadlocks by executing GPU forward and migration tasks directly on the main thread rather than in child process pools.
- **OpenACC Compilation Crash**: Fixed Devito compiler flags for OpenACC targets using `nvc++` under Linux / WSL2 environments.

---

## Usage Examples

### 1. Acoustic Shot Simulation (CPU or GPU)
```python
import numpy as np
from shotgen import ShotRecord

# Velocity model dimensions
nx, nz = 200, 150
vp = 2000.0 * np.ones((nx, nz))
vp[:, 50:] = 2500.0  # 2-layer model

# Initialize ShotRecord with explicit device selection ("cpu" or "gpu")
shot_rec = ShotRecord(
    nx=nx,
    nz=nz,
    dx=10.0,
    dz=10.0,
    n_sources=1,
    n_receivers=100,
    f0=25.0,
    device="cpu",  # Use "gpu" for OpenACC acceleration on NVIDIA GPUs
    engine="devito"
)

shot_rec.set_model(vp)
data = shot_rec.run(ms=500)
print("Simulated shot data shape:", data.shape)
```

### 2. Reverse Time Migration (RTM)
```python
from shotgen import ReverseTimeMigration

# Initialize RTMsolver on CPU
rtm = ReverseTimeMigration(
    dataset_dir="data/sigsbee_dataset",
    nbl=20,
    space_order=4,
    device="cpu"
)

# Run migration for a specific shot
image = rtm.migrate_shot(shot_idx=0)
print("Migrated image shape:", image.shape)
```
