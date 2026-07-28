import numpy as np


def map_coordinate_to_index(pos, spacing, origin, max_cells):
    """
    Convert a physical coordinate (in meters) to an exact integer cell index
    using a rounded nearest-neighbor calculation.

    Parameters
    ----------
    pos : float or ndarray
        Physical position(s) in meters.
    spacing : float
        Grid spacing (dx or dz) in meters.
    origin : float
        Origin coordinate in meters.
    max_cells : int
        Maximum number of cells (nx or nz).

    Returns
    -------
    int or ndarray
        Nearest integer cell index/indices.
    """
    pos_arr = np.atleast_1d(pos)
    max_physical = origin + (max_cells - 1) * spacing
    if np.any(pos_arr < origin) or np.any(pos_arr > max_physical):
        raise ValueError(
            f"Physical position {pos} is out of bounds [{origin}, {max_physical}] "
            f"for grid spacing {spacing} and origin {origin}."
        )
    indices = np.floor((pos_arr - origin) / spacing + 0.5).astype(int)
    if np.isscalar(pos):
        return int(indices[0])
    return indices if len(indices) > 1 else int(indices[0])


__all__ = ["map_coordinate_to_index"]
