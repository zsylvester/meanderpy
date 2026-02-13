# CLAUDE.md - Meanderpy Project Guide

## Project Overview

Meanderpy is a Python library implementing a numerical model of meandering rivers based on the Howard & Knutson (1984) kinematic approach. It simulates channel migration, cutoff formation, and builds 3D stratigraphic models from centerline evolution.

**Author**: Zoltan Sylvester (zoltan.sylvester@beg.utexas.edu)
**Version**: 0.2.0
**License**: Apache 2.0

## Quick Commands

```bash
# Install in development mode
pip install -e .

# Run Jupyter notebooks
jupyter lab meanderpy/meanderpy.ipynb

# Create conda environment
conda env create -f environment.yml
```

## Project Structure

```
meanderpy/
├── meanderpy/
│   ├── __init__.py          # Package init
│   ├── meanderpy.py         # Core module (~1750 lines)
│   └── *.ipynb              # Example notebooks
├── setup.py                 # Package configuration
└── environment.yml          # Conda environment
```

## Core Architecture

### Main Classes (meanderpy.py)

- **`Channel`** - Single channel centerline (x, y, z coords, W width, D depth)
- **`Cutoff`** - Oxbow lake/abandoned loop representation
- **`ChannelBelt`** - Collection of channels over time; main simulation container
  - `migrate()` - Core simulation method
  - `plot()` - Visualization (strat/morph/age views)
- **`ChannelBelt3D`** - 3D stratigraphic model container

### Key Functions

- `generate_initial_channel()` - Create initial sinuous centerline
- `compute_curvature(x, y)` - Calculate centerline curvature
- `compute_migration_rate()` - **Numba-optimized** Howard-Knutson migration
- `migrate_one_step()` - Single iteration lateral migration
- `build_3d_model()` - Create 3D stratigraphic model
- `save_3d_chb_to_hdf5()` / `read_3d_chb_from_hdf5()` - HDF5 I/O

## Dependencies

numpy, scipy, matplotlib, numba, scikit-image, pillow, tqdm, h5py

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `W` | Channel width (m) | 50-200 |
| `D` | Channel depth (m) | W/40 |
| `deltas` | Point spacing (m) | W/4 to W/2 |
| `kl` | Migration rate constant | ~1.9e-6 m/s |
| `kv` | Vertical erosion rate | ~1e-11 m/s |
| `dt` | Time step | 0.1 years (in seconds) |
| `crdist` | Cutoff threshold | 2-3 × W |
| `Cfs` | Chezy friction | ~0.01 |
| `cfl_factor` | CFL stability factor | 0.5 (default) |

## Typical Usage Pattern

```python
import meanderpy

# 1. Generate initial channel
ch = meanderpy.generate_initial_channel(W, D, Sl, deltas, pad, n_bends)

# 2. Create channel belt
chb = meanderpy.ChannelBelt(channels=[ch], cutoffs=[], cl_times=[0.0], cutoff_times=[])

# 3. Run migration (this is the main simulation)
chb.migrate(nit, saved_ts, deltas, pad, crdist, depths, Cfs, kl, kv, dt, dens,
            t1=500, t2=700, t3=1200, aggr_factor=2e-9)

# 4. Visualize
fig = chb.plot('strat', pb_age=20, ob_age=60, end_time=chb.cl_times[-1])

# 5. Build 3D model (optional, computationally expensive)
chb_3d = meanderpy.build_3d_model(chb, model_type='fluvial', h_mud=0.4, ...)
```

## Code Conventions

- Variable naming: `chb` (ChannelBelt), `ch` (Channel), `x,y,z` (coordinates)
- Time in seconds internally; years via `365*24*60*60.0`
- NumPy-style docstrings
- Howard-Knutson constants: omega=-1.0, gamma=2.5

## Numerical Stability (CFL Condition)

The migration functions implement a CFL-style stability check to prevent numerical instability when time steps or migration rates are too large.

**How it works:**
- Migration displacement per time step is clamped to `cfl_factor * deltas`
- Default `cfl_factor = 0.5` (displacement limited to half the point spacing)
- If violated, a warning is issued and displacement is clamped while preserving direction

**Parameters:**
- `cfl_factor` in `migrate()`: Set to `None` to disable clamping (not recommended)
- Stability requires: `|R1| * dt < deltas`

**If you see CFL warnings:**
1. Reduce `dt` (time step)
2. Reduce `kl` (migration rate constant)
3. Increase `deltas` (point spacing) - but this reduces resolution

## Performance Notes

- `compute_migration_rate()` is Numba JIT-compiled - this is the computational bottleneck
- 3D model building is decoupled from migration for efficiency
- CFL clamping adds minimal overhead but prevents simulation blowup

## File Formats

- **HDF5**: 3D model storage (`save_3d_chb_to_hdf5`)
- **Eclipse**: Reservoir model export (`write_eclipse_grid`)

## Testing

Run tests with the meanderpy conda environment:

```bash
source /path/to/conda/etc/profile.d/conda.sh && conda activate meanderpy
python tests/test_meanderpy.py
# Or with pytest if installed:
pytest tests/test_meanderpy.py -v
```

Test coverage includes:
- `compute_curvature()` - straight line, circle, sign changes
- `compute_derivatives()` - shapes, arc length
- `generate_initial_channel()` - object creation, attributes
- `resample_centerline()` - uniform spacing
- `migrate_one_step()` - coordinate changes, CFL warning, CFL clamping
- `find_cutoffs()` / `cut_off_cutoffs()` - cutoff detection
- `get_channel_banks()` - bank coordinates
- `Channel`, `Cutoff`, `ChannelBelt` classes

Additional validation via Jupyter notebooks in `meanderpy/` directory.

## Git Workflow

- Main branch: `master`
- Remote: https://github.com/zsylvester/meanderpy
