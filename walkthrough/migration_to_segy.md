# Full Migration to SEGY Completed

The `shotgen` library has been fully migrated to use your customized SEGY folder structure! 

## What was implemented

1. **New Folder-Based Save Logic**
   The `ShotRecord.save_shot(name)` method now creates a **folder** instead of a single `.h5` file. Inside this folder, it saves:
   - `traces.segy`: The primary seismic trace data with full headers (CDP, Offsets, coordinates) using `SegyIO.write`.
   - `velocity_model.segy`: The 2D velocity model array saved as traces (time-axis acts as depth) using the new `SegyIO.write_model`.
   - `smooth_velocity.segy`: The smoothed 2D background velocity model.
   - `metadata.h5`: A supplementary HDF5 file storing numpy 1D arrays and scalars (e.g., `time`, `wavelet`, and `f0`).

2. **New Folder-Based Load Logic**
   The `LoadShotRecord(path)` method now takes the **folder path** and reads the split files effortlessly behind the scenes. It uses `np.reshape` on the flat `traces.segy` data to perfectly recreate the `(n_sources, n_receivers, n_time)` matrix that PyLops and Devito expect.

3. **Example Scripts Updated**
   All examples that previously hard-coded `.h5` file extensions were updated to point to directory paths (suffixed with `_dataset`). The following files were patched seamlessly:
   - `examples/generate_record.py`
   - `examples/rtm_from_data.py`
   - `examples/reverse_time_migration.py`
   - `examples/kirchhoff_migration.py`
   - `examples/cmp_stacking.py`
   - `examples/test_plot_shotrecord.py`

## Migration Algorithms Native Integration

The migration algorithms have been refactored to natively ingest the new SEGY folder structure, decoupling them from intermediate `LoadShotRecord` objects. 

1. **`dataset_dir` API**
   `KirchhoffMigration`, `ReverseTimeMigration`, and `ReverseTimeMigrationGPU` now accept a `dataset_dir` keyword argument. When provided, they automatically read `traces.segy` and `smooth_velocity.segy` (falling back to `velocity_model.segy` if not found).
2. **Dynamic Trace Reshaping**
   They reconstruct the correct multidimensional shapes for `shots`, `sources`, and `receivers` seamlessly on the fly by deducing unique trace indices from the SEGY headers.
3. **Strict Validation for `f0`**
   For RTM, the algorithm attempts to extract `f0` from `metadata.h5`. Per your request, if `f0` is absent and not provided explicitly, the system strictly raises a `ValueError` rather than defaulting, guaranteeing algorithmic safety.

## Verification

A comprehensive save/load test script ([examples/test_save_load.py](file:///mnt/3E0C05A50C055973/Documentos/MUN_MSc/100thesisresearchproject/codes/shotgen/examples/test_save_load.py)) was executed. It verified that:
1. The folder creates properly.
2. The `traces.segy` successfully loads and reshapes its matrix.
3. The `velocity_model.segy` is read identically matching its source via `np.allclose`.

Additionally, the native migration logic was validated via a dedicated test suite ([tests/test_migration_segy.py](file:///mnt/3E0C05A50C055973/Documentos/MUN_MSc/100thesisresearchproject/codes/shotgen/tests/test_migration_segy.py)).
- **`pytest` Results:**
```
============================= test session starts ==============================
collected 4 items
tests/test_migration_segy.py ....                                        [100%]
======================== 4 passed, 1 warning in 20.02s =========================
```
The tests confirm that both Kirchhoff and RTM load datasets correctly, process coordinate reshapes reliably, and correctly trap edge-cases like missing structural folders or omitted `f0` fields.

```
Running simulation...
Saving to test_dataset...
Saved simulation files to folder test_dataset
Loading from test_dataset...
Test passed! Directory saving and loading is working perfectly.
```
