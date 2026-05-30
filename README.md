
# Shotgen

A Python-based framework for simulating **2D acoustic wave propagation** and generating synthetic **shot records** for seismic applications. This project leverages `Devito`, `PyLops`, and `DeepWave` to perform high-performance wavefield modeling using the Born approximation.

## Overview

**Shotgen** provides tools for:
- Creating synthetic geological models with layered structures, velocity variations, and fault representations
- Simulating 2D acoustic wave propagation in heterogeneous media
- Generating synthetic seismic shot records for forward modeling and inversion studies
- Performing Reverse-Time Migration (RTM) for seismic imaging
- Processing and visualizing seismic data

## Installation

### Prerequisites

- Python >= 3.12.3
- A virtual environment (recommended)

### Setup Instructions

1. **Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Editable Install from the project root:**

```bash
pip install -e .
```

This command installs `shotgen` in editable mode, allowing changes to the source code to be reflected immediately without re-installation.

3. **Verify installation:**

```bash
python -c "import shotgen; print('Shotgen installed successfully')"
```

---

## Key Classes and Functions

### `GeoModel` (sampleshot.py)
A geological model generator for creating synthetic velocity structures:

- **`__init__(nx, nz, v_base=4500)`** — Initialize a velocity model with dimensions (nx, nz) and base velocity
- **`layered()`** — Generate a layered model with stratigraphy, velocity gradients, and fault structures
- **`_create_layer_interface(base_depth, amplitude, wavelength, phase, slope)`** — Helper method to create wavy layer interfaces

### `ReverseTimeMigration` (migration.py)
Implements reverse-time migration imaging using Devito:

- **`__init__(...)`** — Initialize RTM with velocity model, acquisition geometry, and imaging parameters
- **`_create_model()`** — Create Devito Model objects for forward and smoothed velocity models
- **`_create_geometry()`** — Define source and receiver geometries
- **`_setup_solver()`** — Configure the acoustic wave solver

### `LoadShot` (sampleshot.py)
Utility class for loading and managing seismic shot records from HDF5 files

---

## Project Structure

```text
shotgen/
├── pyproject.toml                      # Package metadata and dependencies
├── README.md                           # Project documentation
├── assets/                             # Sample SEG-Y data files
│   └── complex_graben.sgy
├── data/                               # Generated shot records (HDF5 format)
├── examples/                           # Example scripts
│   ├── cmp_stacking.py                # Common Mid-Point stacking example
│   ├── generate_record.py             # Shot record generation example
│   ├── reverse_time_migration.py      # RTM imaging example
│   └── test_plot_shotrecord.py        # Visualization example
└── shotgen/                            # Main package
    ├── sampleshot.py                  # Geological models and shot generation
    └── migration.py                   # Reverse-Time Migration implementation
```

---

## Usage Examples

### Example 1: Create a Layered Geological Model

```python
from shotgen.sampleshot import GeoModel

# Initialize model with 600 x-samples and 250 z-samples
model = GeoModel(nx=600, nz=250, v_base=4500)

# Generate layered structure with faults
model.layered()

# Access velocity field
velocity = model.vel
```

### Example 2: Perform Reverse-Time Migration

```python
from shotgen.migration import ReverseTimeMigration
import numpy as np

# RTM configuration
rtm = ReverseTimeMigration(
    vp=velocity_model,
    n_sources=6,
    n_receivers=36,
    origin=(0., 0.),
    spacing=(25., 10.),
    nbl=40,
    t0=0.,
    tn=0.240,
    f0=100.,
    smooth_sigma=5.0
)

# Perform migration (see examples/ for full implementation)
```

### Example 3: Generate and Visualize Shot Records

See `examples/generate_record.py` and `examples/test_plot_shotrecord.py` for complete examples of:
- Generating synthetic shot records
- Plotting and visualizing seismic data
- Reading/writing HDF5 shot records

---

## Dependencies

Core dependencies (specified in `pyproject.toml`):
- **devito** — High-performance finite-difference framework
- **pylops** — Linear operators and inversion
- **deepwave** — Deep learning for seismic modeling
- **numpy** — Numerical computing
- **scipy** — Scientific computing utilities
- **matplotlib** — Visualization
- **h5py** — HDF5 file handling
- **scikit-image** — Image processing
- **scienceplots** — Publication-quality plotting
- **joblib** — Parallel processing
- **segyio** — SEG-Y seismic file I/O
- **tqdm** — Progress bars

---

## Citation & Author

**Author:** Jesus Ochoa (Memorial University of Newfoundland)  
**Email:** jochoacontre@mun.ca  
**Version:** 1.0.0  
**Status:** Alpha

---

## License

See LICENSE file for details (if applicable).

## References

This project builds upon:
- [Devito Project](https://www.devitoproject.org/) — Finite-difference modeling
- [PyLops](https://pylops.readthedocs.io/) — Linear operators and seismic inversion
- [DeepWave](https://deepwave.readthedocs.io/) — Deep learning for seismic modeling

For additional references on wave propagation and seismic imaging, see the example notebooks and academic literature on acoustic wave equations and RTM.
