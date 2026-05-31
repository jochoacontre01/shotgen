# Changelog

All notable changes to the `shotgen` project are documented in this file.

## [Unreleased]

### Added
- **RTM from Data Example**: Added [examples/rtm_from_data.py](file:///mnt/3E0C05A50C055973/Documentos/MUN_MSc/100thesisresearchproject/codes/shotgen/examples/rtm_from_data.py) to demonstrate running a CPU-based Reverse Time Migration (RTM) on existing acquisition datasets (such as the Sigsbee dataset `data/commonshot-shot_750nx_275nz_48rec_6src_100hz_10goffset_45soffset_sigsbee.h5`).

### Changed
- **ReverseTimeMigration Integration**: Refactored `ReverseTimeMigration` in [migration.py](file:///mnt/3E0C05A50C055973/Documentos/MUN_MSc/100thesisresearchproject/codes/shotgen/shotgen/migration.py) to support both the simulation RTM mode and data-driven mode (whether the receiver arrays are fixed or change dynamically per shot) natively. It automatically handles time-discretization resampling and data type/shape casting.
- **Documentation**: Enhanced the [KirchhoffMigration](file:///mnt/3E0C05A50C055973/Documentos/MUN_MSc/100thesisresearchproject/codes/shotgen/shotgen/migration.py#L279) class docstring in [migration.py](file:///mnt/3E0C05A50C055973/Documentos/MUN_MSc/100thesisresearchproject/codes/shotgen/shotgen/migration.py) with comprehensive `numpydoc`-compliant documentation. It specifies that input arrays typically originate from a `ShotRecord` (or `LoadShotRecord`) instance.

---

## Design Rationale: Native `ReverseTimeMigration` Data Integration

To run Reverse Time Migration using real or pre-simulated datasets, `ReverseTimeMigration` was updated to solve three critical technical problems:

### 1. Varying/Moving Receiver Geometries
* **Problem**: The standard `ReverseTimeMigration` class assumes that the receiver array is fixed in place for all source activations. In contrast, datasets like Sigsbee use a moving template geometry where the 48 receivers shift dynamically with each of the 6 sources (represented by a 3D coordinate array of shape `(6, 48, 2)`).
* **Solution**: The class stores the symbolic `residual` `PointSource` as `self.residual_source`. During each shot iteration in the `run` loop, the physical coordinates are updated dynamically for both the forward receiver geometry and the backward imaging residual source:
  ```python
  self.geometry.rec.coordinates.data[:, :] = current_recs
  self.residual_source.coordinates.data[:, :] = current_recs
  ```

### 2. Time-Step (`dt` and `nt`) Mismatches
* **Problem**: Devito automatically computes the critical time step size (`critical_dt`) based on the stability limits of the velocity model. Because the smoothed velocity model (`v0`) differs from the true velocity model (`vp`), this produces different step counts (e.g., 2115 steps vs the dataset's 2262 steps). Devito throws a `ValueError` or `InvalidArgument` because the time grid dimensions of the forward wavefield `u`, backward wavefield `v`, and dataset `residual` are incompatible.
* **Solution**: The class resamples Devito's `AcquisitionGeometry` to match the exact time-step size of the dataset (`dt_data`) via `self.geometry.resample(dt)`. Furthermore, the resampled time step size is explicitly passed to both the forward solver and the imaging operator:
  ```python
  # Solver
  self.solver.forward(vp=self.model0.vp, save=True, dt=self.geometry.dt)
  # Operator
  operator(u=u0, v=v, vp=self.model0.vp, dt=self.geometry.dt, residual=self.shots[i])
  ```

### 3. Shot Data Dimensions and Numeric Types
* **Problem**: HDF5 files store shot data in shape `(n_sources, n_receivers, nt)` using double-precision (`float64`). Devito's time-injection operator requires single-precision (`float32`) and a transposed axis structure of `(n_sources, nt, n_receivers)`.
* **Solution**: The constructor automatically transposes the shot records and casts them to `float32` before launching the migration:
  ```python
  self.shots = np.transpose(shots, (0, 2, 1)).astype(np.float32)
  ```

