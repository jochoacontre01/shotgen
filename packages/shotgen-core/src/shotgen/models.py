import numpy as np


class GeoModel:

    def __init__(self, nx, nz, v_base=4500):
        self.nx = nx
        self.nz = nz
        self.X, self.Z = np.meshgrid(np.arange(self.nx), np.arange(self.nz), indexing='ij')
        self.vel = v_base * np.ones((self.nx, self.nz))

    def _create_layer_interface(self, base_depth, amplitude, wavelength, phase=0, slope=0):
        """Helper to create wavy layer interfaces"""
        return (base_depth +
                slope * (self.X - self.nx / 2) +
                amplitude * np.sin(2 * np.pi * self.X / wavelength + phase))

    def layered(self):
        """
        Generates a 'Compactive Layered' model.
        Features: Mostly horizontal stratigraphy with varying bed thicknesses (thick & thin)
        and normal faults to test frequency/resolution.
        """
        print("Generating Layered Resolution model...")

        for i in range(self.nz):
            self.vel[:, i] = 2000 + (i / self.nz) * 2500

        beds = [
            (0.10, 0.15, 2100),
            (0.30, 0.01, 3500),
            (0.35, 0.08, 2400),
            (0.44, 0.015, 1800),
            (0.47, 0.015, 1800),
            (0.55, 0.20, 3200),
            (0.78, 0.005, 4500)
        ]

        wobble = 0.005 * self.nz * np.sin(2 * np.pi * self.X / self.nx)

        for top_pct, thick_pct, v in beds:
            z_top = (top_pct * self.nz) + wobble
            z_bot = ((top_pct + thick_pct) * self.nz) + wobble
            mask = (self.Z >= z_top) & (self.Z < z_bot)
            self.vel[mask] = v

        f1_x = 0.35 * self.nx
        f1_angle = 1.6
        f1_throw = int(0.06 * self.nz)

        for ix in range(self.nx):
            z_fault = int(f1_angle * (ix - f1_x))
            if 0 <= z_fault < self.nz:
                col = self.vel[ix, :].copy()
                shifted = np.roll(col[z_fault:], f1_throw)
                col[z_fault:] = shifted
                col[z_fault:z_fault + f1_throw] = col[z_fault - 1] if z_fault > 0 else 1500
                self.vel[ix, :] = col

        f2_x = 0.75 * self.nx
        f2_angle = 1.3
        f2_throw = int(0.12 * self.nz)

        for ix in range(self.nx):
            z_fault = int(f2_angle * (ix - f2_x) + 0.3 * self.nz)
            if 0 <= z_fault < self.nz:
                col = self.vel[ix, :].copy()
                shifted = np.roll(col[z_fault:], f2_throw)
                col[z_fault:] = shifted
                col[z_fault:z_fault + f2_throw] = col[z_fault - 1] if z_fault > 0 else 1500
                self.vel[ix, :] = col

        return self.vel

    def foothills(self):
        print("Generating Complex Foothills (Fold-and-Thrust) model...")
        for i in range(self.nz):
            self.vel[:, i] = 2200 + (i / self.nz) * 2000

        z1 = self._create_layer_interface(0.3 * self.nz, 0.08 * self.nz, 0.4 * self.nx)
        z2 = self._create_layer_interface(0.5 * self.nz, 0.12 * self.nz, 0.35 * self.nx, phase=1.0)
        z3 = self._create_layer_interface(0.7 * self.nz, 0.05 * self.nz, 0.5 * self.nx)

        self.vel[self.Z >= z1] = 2800
        self.vel[self.Z >= z2] = 3600
        self.vel[self.Z >= z3] = 4800

        t1_x0 = 0.1 * self.nx
        t1_slope = 0.8
        t1_displacement = int(0.15 * self.nz)

        for ix in range(self.nx):
            z_fault = int(t1_slope * (ix - t1_x0) + 0.2 * self.nz)
            if 0 <= z_fault < self.nz:
                col = self.vel[ix, :].copy()
                shifted = np.roll(col[:z_fault], -t1_displacement)
                col[:z_fault] = shifted
                self.vel[ix, :] = col

        return self.vel


__all__ = ["GeoModel", "resample_velocity_model"]


def resample_velocity_model(vel: np.ndarray, dx_orig: float, dz_orig: float, target_dx: float = None, target_dz: float = None):
    """
    Resamples a 2D velocity model matrix (nx, nz) to target spatial spacings target_dx, target_dz
    while preserving total physical extent L_x and L_z.

    Parameters
    ----------
    vel : np.ndarray
        2D velocity matrix of shape (nx_orig, nz_orig).
    dx_orig : float
        Original grid spacing along X axis in meters.
    dz_orig : float
        Original grid spacing along Z axis in meters.
    target_dx : float, optional
        Target grid spacing along X axis in meters.
    target_dz : float, optional
        Target grid spacing along Z axis in meters.

    Returns
    -------
    vel_resampled : np.ndarray
        Resampled 2D velocity matrix of shape (nx_new, nz_new).
    nx_new : int
        New number of grid points along X axis.
    nz_new : int
        New number of grid points along Z axis.
    dx_out : float
        Effective output dx grid spacing.
    dz_out : float
        Effective output dz grid spacing.
    """
    nx_orig, nz_orig = vel.shape
    dx_out = float(target_dx) if target_dx is not None else float(dx_orig)
    dz_out = float(target_dz) if target_dz is not None else float(dz_orig)

    if np.isclose(dx_out, dx_orig) and np.isclose(dz_out, dz_orig):
        return vel, nx_orig, nz_orig, dx_out, dz_out

    L_x = (nx_orig - 1) * float(dx_orig)
    L_z = (nz_orig - 1) * float(dz_orig)

    nx_new = int(np.round(L_x / dx_out)) + 1
    nz_new = int(np.round(L_z / dz_out)) + 1

    x_orig = np.linspace(0.0, L_x, nx_orig)
    z_orig = np.linspace(0.0, L_z, nz_orig)

    x_new = np.linspace(0.0, L_x, nx_new)
    z_new = np.linspace(0.0, L_z, nz_new)

    from scipy.interpolate import RegularGridInterpolator
    interpolator = RegularGridInterpolator((x_orig, z_orig), vel, method="linear", bounds_error=False, fill_value=None)
    X_new, Z_new = np.meshgrid(x_new, z_new, indexing="ij")
    pts = np.vstack([X_new.ravel(), Z_new.ravel()]).T

    vel_resampled = interpolator(pts).reshape((nx_new, nz_new)).astype(vel.dtype)
    return vel_resampled, nx_new, nz_new, dx_out, dz_out

