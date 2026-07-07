# Dimension Tracking & Plotting

I have updated the pipeline to seamlessly handle and persist physical dimensions, ensuring that your plotting axes always stay true to physical scales (meters) instead of arbitrary grid cell counts.

## Key Changes
1. **SEGY Persistence**: `SegyIO` now correctly reads and writes the model dimension spacings (`dx` and `dz`). The dataset writer embeds these sizes directly into the SEGY header.
2. **Dataset Loaders**: The `LoadShotRecord` class and `ReverseTimeMigration` have been upgraded to unpack `(velocity_model, dx, dz)` dynamically when loading existing data, meaning `RTM` automatically adopts the scale of the dataset it's migrating.
3. **Axes Visualization**: Migrated figures and `ShotRecord` visualizations now calculate extents effectively. Instead of defaulting to index counts, `plt.imshow` receives an exact `extent` mapping out the physical meters of the data (dynamically reading the loaded `dx` and `dz`).

> [!NOTE]
> I kept the standard unit in **meters (m)** across all plot generation scripts for consistency.

## Example Updates
I updated `generate_record.py` to use `dx=5, dz=5` for the `vp` dataset (reflecting the `[::5, ::5]` downsampling of the original 1-meter Marmousi dataset) and passed it to the dataset initialization.

I updated `rtm_from_data.py` to extract `dx_spacing` and `dz_spacing` dynamically from the instantiated `ReverseTimeMigration` class, feeding it into the `extent` of the plotted image.

Background tests confirm everything generates and displays correctly via the `timg` interface!
