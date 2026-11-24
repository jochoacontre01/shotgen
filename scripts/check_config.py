#!/usr/bin/env python3
"""
Pre-flight sanity checker for shotgen JSON/YAML simulation configuration files.

Inspects source/receiver reach (latest receiver X), maximum number of sources
that fit in the velocity model without padding, spatial bounds, numerical
dispersion/stability (PPW, CFL, Nyquist), two-way traveltime vs recording
duration, survey geometry (spread, offset, CMP fold, spatial aliasing),
estimated memory/disk footprint, and resolved compact output directory path.

Usage:
    python scripts/check_config.py examples/configs/test_config.json
    python scripts/check_config.py examples/configs/ --fast
    python scripts/check_config.py cfg1.json cfg2.yaml --json --strict
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from shotgen.cli import _load_velocity_model_core
from shotgen.utils import resolve_simulation_output_dir

SUPPORTED_EXTS = {".json", ".yaml", ".yml"}
MAX_PATH_LIMIT = 260
TYPICAL_ARTIFACT_SUFFIX_LEN = 36  # e.g. /20261005_223356_rtm_migrated.npy


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str


@dataclass
class ConfigReport:
    config_path: str
    task: str
    model_source: str
    velocity_source: str
    gather: str

    # Grid & velocity
    nx: int
    nz: int
    dx: float
    dz: float
    origin: tuple[float, float]
    model_min_x: float
    model_max_x: float
    model_min_z: float
    model_max_z: float
    v_min: float
    v_max: float
    v_mean: float

    # Core source/receiver reach & capacity
    n_sources: int
    n_receivers: int
    shot_offset: float
    group_offset: float
    src_origin: tuple[float, float]
    rec_origin: tuple[float, float]
    first_src_x: float
    last_src_x: float
    first_rec_x: float
    latest_rec_x: float
    max_reach_x: float
    margin_x_m: float
    overflow_x_m: float
    overflow_x_cells: int
    max_sources_fit: int | None
    max_sources_by_rec: int | None
    max_sources_by_src: int | None

    # Survey geometry
    spread_length_m: float
    min_offset_m: float
    max_offset_m: float
    nominal_cmp_fold: float | None
    cmp_bin_size_m: float | None
    spatial_aliasing_limit_m: float

    # Dispersion & stability
    f0_hz: float
    f_max_hz: float
    fd_order: int
    n_damping: int
    lambda_min_m: float
    ppw: float
    dt_crit_ms: float
    ms: float
    fs_ms: float | None
    nt_internal: int
    nt_out: int
    f_nyquist_hz: float | None

    # Traveltime
    twt_vertical_ms: float
    twt_max_offset_ms: float

    # Footprint
    padded_nx: int
    padded_nz: int
    total_traces: int
    float_bytes: int
    est_shot_ram_mb: float
    est_wavefield_ram_mb: float
    est_segy_mb: float
    est_h5_mb: float

    # Output path
    output_dir: str
    output_dir_abs_len: int
    est_artifact_path_len: int

    # Checks & overall status
    checks: list[CheckResult] = field(default_factory=list)
    overall_status: str = "PASS"


def _parse_val(val: Any, default: Any, val_type=float) -> Any:
    if val is None or (isinstance(val, str) and val.strip().lower() in ("none", "null")):
        return default
    return val_type(val)


def _normalize_gather(raw: str) -> str:
    val = str(raw).strip().lower()
    if val in ("cs", "common shot", "common_shot"):
        return "common shot"
    if val in ("cmp", "common midpoint", "common_midpoint"):
        return "common midpoint"
    return val


def _cfl_factor_2d(fd_order: int) -> float:
    """Approximate 2D acoustic FD CFL stability factor."""
    if fd_order <= 2:
        return 1.0 / math.sqrt(2.0)  # ~0.707
    if fd_order <= 4:
        return math.sqrt(3.0 / 8.0)  # ~0.612
    if fd_order <= 6:
        return 0.56
    return 0.52


def _apply_max_size_crop(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    origin: tuple[float, float],
    max_size: Any,
    vel: np.ndarray | None = None,
) -> tuple[int, int, np.ndarray | None]:
    """Replicate shotgen's max_size cropping logic."""
    if max_size is None:
        return nx, nz, vel

    max_x, max_z = float(max_size[0]), float(max_size[1])
    model_len_x = nx * dx
    model_len_z = nz * dz
    if max_z > model_len_z and max_x <= model_len_z and max_z <= model_len_x:
        max_x, max_z = max_z, max_x

    x_start = max(0, min(int(np.round(origin[0] / dx)), nx))
    x_end = max(x_start, min(x_start + int(np.round(max_x / dx)), nx))
    z_start = max(0, min(int(np.round(origin[1] / dz)), nz))
    z_end = max(z_start, min(z_start + int(np.round(max_z / dz)), nz))

    if x_end > x_start and z_end > z_start:
        if vel is not None:
            vel = vel[x_start:x_end, z_start:z_end]
            return vel.shape[0], vel.shape[1], vel
        return x_end - x_start, z_end - z_start, None

    return nx, nz, vel


def load_config_dict(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a JSON/YAML object, got {type(data).__name__}.")
    return data


def discover_config_files(paths: list[str], recursive: bool = False) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            iterator = p.rglob("*") if recursive else p.glob("*")
            dir_files = sorted(f for f in iterator if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS)
            found.extend(dir_files)
        else:
            raise FileNotFoundError(f"Path not found: {raw}")
    return found


def _compute_geometry(
    cfg: dict[str, Any],
    gather: str,
    n_sources: int,
    n_receivers: int,
    src_origin: tuple[float, float],
    rec_origin: tuple[float, float],
    shot_offset: float,
    group_offset: float,
    model_max_x: float,
) -> dict[str, Any]:
    """Compute source/receiver coordinates, reach, offsets, and max fitting sources."""
    explicit_sources = cfg.get("sources")
    explicit_receivers = cfg.get("receivers")

    if explicit_sources is not None:
        src_arr = np.asarray(explicit_sources, dtype=np.float64)
        src_x = src_arr[..., 0]
        src_z = src_arr[..., 1]
        n_sources = src_arr.shape[0]
    else:
        src_x = src_origin[0] + np.arange(n_sources, dtype=np.float64) * shot_offset
        src_z = np.full(n_sources, src_origin[1], dtype=np.float64)

    if explicit_receivers is not None:
        rec_arr = np.asarray(explicit_receivers, dtype=np.float64)
        rec_x = rec_arr[..., 0]
        rec_z = rec_arr[..., 1]
        n_receivers = rec_arr.shape[1] if rec_arr.ndim == 3 else rec_arr.shape[0]
    elif gather == "common shot":
        # Matches ShotRecord._set_common_shot():
        # for si in range(n_sources):
        #     rec_x = linspace(0, (n_receivers - 1) * group_offset, n_receivers) + (si + 1) * shot_offset + rec_origin[0]
        rec_rel = np.arange(n_receivers, dtype=np.float64) * group_offset
        shot_shifts = (np.arange(n_sources, dtype=np.float64) + 1.0) * shot_offset
        rec_x = rec_origin[0] + shot_shifts[:, None] + rec_rel[None, :]
        rec_z = np.full_like(rec_x, rec_origin[1])
    else:
        rec_x = rec_origin[0] + np.arange(n_receivers, dtype=np.float64) * group_offset
        rec_z = np.full(n_receivers, rec_origin[1], dtype=np.float64)

    first_src_x = float(np.min(src_x)) if src_x.size else float(src_origin[0])
    last_src_x = float(np.max(src_x)) if src_x.size else float(src_origin[0])
    first_rec_x = float(np.min(rec_x)) if rec_x.size else float(rec_origin[0])
    latest_rec_x = float(np.max(rec_x)) if rec_x.size else float(rec_origin[0])
    min_src_z = float(np.min(src_z)) if src_z.size else float(src_origin[1])
    max_src_z = float(np.max(src_z)) if src_z.size else float(src_origin[1])
    min_rec_z = float(np.min(rec_z)) if rec_z.size else float(rec_origin[1])
    max_rec_z = float(np.max(rec_z)) if rec_z.size else float(rec_origin[1])

    # Offsets
    if rec_x.ndim == 2 and src_x.size == rec_x.shape[0]:
        offsets = np.abs(rec_x - src_x[:, None])
    elif rec_x.size and src_x.size:
        offsets = np.abs(rec_x[None, :] - src_x[:, None])
    else:
        offsets = np.array([0.0])

    min_offset_m = float(np.min(offsets))
    max_offset_m = float(np.max(offsets))
    spread_length_m = float(abs((n_receivers - 1) * group_offset)) if explicit_receivers is None else float(
        np.max(rec_x[0]) - np.min(rec_x[0]) if rec_x.ndim == 2 else np.max(rec_x) - np.min(rec_x)
    )

    # Max sources that fit in model without exceeding model_max_x
    max_sources_fit: int | None = None
    max_sources_by_rec: int | None = None
    max_sources_by_src: int | None = None

    if explicit_sources is None and shot_offset > 0:
        n_src_limit = math.floor((model_max_x - src_origin[0]) / shot_offset) + 1
        max_sources_by_src = max(0, n_src_limit)

        if gather == "common shot" and explicit_receivers is None:
            rec_span = max(0.0, (n_receivers - 1) * group_offset)
            n_rec_limit = math.floor((model_max_x - rec_origin[0] - rec_span) / shot_offset)
            max_sources_by_rec = max(0, n_rec_limit)
            max_sources_fit = max(0, min(max_sources_by_src, max_sources_by_rec))
        else:
            max_sources_fit = max_sources_by_src

    # Nominal CMP fold (applicable to standard roll-along common shot)
    nominal_cmp_fold: float | None = None
    cmp_bin_size_m: float | None = None
    if gather == "common shot" and shot_offset > 0 and group_offset > 0:
        nominal_cmp_fold = (n_receivers * abs(group_offset)) / (2.0 * abs(shot_offset))
        cmp_bin_size_m = abs(group_offset) / 2.0

    return {
        "n_sources": int(n_sources),
        "n_receivers": int(n_receivers),
        "first_src_x": first_src_x,
        "last_src_x": last_src_x,
        "first_rec_x": first_rec_x,
        "latest_rec_x": latest_rec_x,
        "min_src_z": min_src_z,
        "max_src_z": max_src_z,
        "min_rec_z": min_rec_z,
        "max_rec_z": max_rec_z,
        "min_offset_m": min_offset_m,
        "max_offset_m": max_offset_m,
        "spread_length_m": spread_length_m,
        "max_sources_fit": max_sources_fit,
        "max_sources_by_rec": max_sources_by_rec,
        "max_sources_by_src": max_sources_by_src,
        "nominal_cmp_fold": nominal_cmp_fold,
        "cmp_bin_size_m": cmp_bin_size_m,
    }


def analyze_config(config_path: Path, fast: bool = False) -> ConfigReport:
    cfg = load_config_dict(config_path)
    checks: list[CheckResult] = []

    task = str(cfg.get("task", "simulation")).strip().lower()
    if task != "simulation":
        checks.append(
            CheckResult(
                "task",
                "WARN",
                f"Config task is '{task}' (not 'simulation'); geometry checks assume simulation parameters.",
            )
        )

    origin_raw = cfg.get("origin", [0.0, 0.0])
    origin = (float(origin_raw[0]), float(origin_raw[1]))
    src_origin_raw = cfg.get("src_origin", [0.0, 0.0])
    src_origin = (float(src_origin_raw[0]), float(src_origin_raw[1]))
    rec_origin_raw = cfg.get("rec_origin", [0.0, 0.0])
    rec_origin = (float(rec_origin_raw[0]), float(rec_origin_raw[1]))

    vel_spec = cfg.get("velocity_model") or cfg.get("vel_file") or cfg.get("vel_model") or cfg.get("model")
    model_source = str(vel_spec) if vel_spec is not None else "synthetic (fallback)"

    # Load or infer model dimensions & velocity stats
    vel = None
    if not fast:
        try:
            vel, nx, nz, dx, dz = _load_velocity_model_core(cfg)
            nx, nz, vel = _apply_max_size_crop(nx, nz, dx, dz, origin, cfg.get("max_size"), vel=vel)
            v_min = float(np.min(vel))
            v_max = float(np.max(vel))
            v_mean = float(np.mean(vel))
            velocity_source = "loaded velocity model"
        except FileNotFoundError as exc:
            checks.append(CheckResult("velocity_model", "FAIL", f"{exc} (falling back to config nx/nz/v_base)"))
            fast = True

    if fast:
        dx = _parse_val(cfg.get("dx"), 10.0, float)
        dz = _parse_val(cfg.get("dz"), 10.0, float)
        nx = _parse_val(cfg.get("nx"), 100, int)
        nz = _parse_val(cfg.get("nz"), 100, int)
        nx, nz, _ = _apply_max_size_crop(nx, nz, dx, dz, origin, cfg.get("max_size"), vel=None)
        v_base = _parse_val(cfg.get("v_base"), 2500.0, float)
        v_min = v_base
        v_max = v_base
        v_mean = v_base
        velocity_source = "config v_base (fast mode)"

    model_min_x = origin[0]
    model_max_x = origin[0] + nx * dx
    model_min_z = origin[1]
    model_max_z = origin[1] + nz * dz

    n_sources = _parse_val(cfg.get("n_sources"), 2, int)
    n_receivers = _parse_val(cfg.get("n_receivers"), 10, int)
    f0 = _parse_val(cfg.get("f0"), 25.0, float)
    ms = _parse_val(cfg.get("ms", cfg.get("ntime")), 300.0, float)
    fd_order = _parse_val(cfg.get("fd_order"), 4, int)
    n_damping = _parse_val(cfg.get("n_damping"), 40, int)
    group_offset = _parse_val(cfg.get("group_offset"), 1.0, float)
    shot_offset = _parse_val(cfg.get("shot_offset"), 1.0, float)
    gather = _normalize_gather(cfg.get("gather", "cmp"))
    fs_ms = _parse_val(cfg.get("fs_ms"), None, float)
    float_type_str = str(cfg.get("float_type", "float32")).lower()
    float_bytes = 8 if "64" in float_type_str else 4

    geom = _compute_geometry(
        cfg=cfg,
        gather=gather,
        n_sources=n_sources,
        n_receivers=n_receivers,
        src_origin=src_origin,
        rec_origin=rec_origin,
        shot_offset=shot_offset,
        group_offset=group_offset,
        model_max_x=model_max_x,
    )
    n_sources = geom["n_sources"]
    n_receivers = geom["n_receivers"]

    max_reach_x = max(geom["last_src_x"], geom["latest_rec_x"])
    margin_x_m = model_max_x - max_reach_x
    overflow_x_m = max(0.0, -margin_x_m)
    overflow_x_cells = max(0, int(math.ceil((max_reach_x - origin[0]) / dx)) - nx) if dx > 0 else 0

    # --- Spatial bounds & padding checks ---
    if geom["first_src_x"] < model_min_x or geom["first_rec_x"] < model_min_x:
        checks.append(
            CheckResult(
                "x_min_bounds",
                "FAIL",
                f"Coordinates extend before model origin X ({model_min_x:.2f} m): "
                f"min_src_x={geom['first_src_x']:.2f} m, min_rec_x={geom['first_rec_x']:.2f} m.",
            )
        )
    else:
        checks.append(CheckResult("x_min_bounds", "PASS", f"All X coordinates >= model origin ({model_min_x:.2f} m)."))

    if geom["last_src_x"] > model_max_x:
        checks.append(
            CheckResult(
                "source_x_bounds",
                "FAIL",
                f"Last source X ({geom['last_src_x']:.2f} m) exceeds model X extent ({model_max_x:.2f} m) "
                f"by {geom['last_src_x'] - model_max_x:.2f} m (ShotRecord will raise ValueError).",
            )
        )
    else:
        checks.append(
            CheckResult(
                "source_x_bounds",
                "PASS",
                f"Sources within model X extent ({geom['last_src_x']:.2f} <= {model_max_x:.2f} m).",
            )
        )

    if geom["latest_rec_x"] > model_max_x:
        checks.append(
            CheckResult(
                "receiver_x_bounds",
                "WARN" if gather == "common shot" else "FAIL",
                f"Latest receiver X ({geom['latest_rec_x']:.2f} m) exceeds model X extent ({model_max_x:.2f} m) "
                f"by {geom['latest_rec_x'] - model_max_x:.2f} m ({overflow_x_cells} cells; triggers edge-padding).",
            )
        )
    else:
        checks.append(
            CheckResult(
                "receiver_x_bounds",
                "PASS",
                f"Latest receiver X ({geom['latest_rec_x']:.2f} m) fits inside model ({model_max_x:.2f} m; "
                f"margin {model_max_x - geom['latest_rec_x']:.2f} m).",
            )
        )

    z_ok = (
        model_min_z <= geom["min_src_z"] <= model_max_z
        and model_min_z <= geom["max_src_z"] <= model_max_z
        and model_min_z <= geom["min_rec_z"] <= model_max_z
        and model_min_z <= geom["max_rec_z"] <= model_max_z
    )
    if not z_ok:
        checks.append(
            CheckResult(
                "z_bounds",
                "FAIL",
                f"Source/receiver Z outside [{model_min_z:.2f}, {model_max_z:.2f}] m "
                f"(src_z=[{geom['min_src_z']:.2f}, {geom['max_src_z']:.2f}], "
                f"rec_z=[{geom['min_rec_z']:.2f}, {geom['max_rec_z']:.2f}]).",
            )
        )
    else:
        checks.append(CheckResult("z_bounds", "PASS", f"All Z coordinates within [{model_min_z:.2f}, {model_max_z:.2f}] m."))

    if n_damping < 20:
        checks.append(
            CheckResult(
                "damping_boundary",
                "WARN",
                f"n_damping={n_damping} cells ({n_damping * dx:.1f} m) is small (< 20); risk of boundary reflections.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "damping_boundary",
                "PASS",
                f"n_damping={n_damping} cells ({n_damping * dx:.1f} m X, {n_damping * dz:.1f} m Z).",
            )
        )

    # --- Dispersion & numerical stability ---
    f_max_hz = 2.5 * f0
    lambda_min_m = v_min / f_max_hz if f_max_hz > 0 else float("inf")
    ppw = lambda_min_m / max(dx, dz) if max(dx, dz) > 0 else float("inf")

    ppw_warn = 10.0 if fd_order <= 2 else 5.0
    ppw_fail = 5.0 if fd_order <= 2 else 3.0
    if ppw < ppw_fail:
        checks.append(
            CheckResult(
                "dispersion_ppw",
                "FAIL",
                f"Severe numerical dispersion risk: {ppw:.2f} PPW at v_min={v_min:.1f} m/s, "
                f"f_max={f_max_hz:.1f} Hz (recommended >= {ppw_warn:.0f} for fd_order={fd_order}).",
            )
        )
    elif ppw < ppw_warn:
        checks.append(
            CheckResult(
                "dispersion_ppw",
                "WARN",
                f"Low grid sampling: {ppw:.2f} PPW at v_min={v_min:.1f} m/s, "
                f"f_max={f_max_hz:.1f} Hz (recommended >= {ppw_warn:.0f} for fd_order={fd_order}).",
            )
        )
    else:
        checks.append(
            CheckResult(
                "dispersion_ppw",
                "PASS",
                f"Grid dispersion OK: {ppw:.2f} PPW (min wavelength {lambda_min_m:.2f} m, fd_order={fd_order}).",
            )
        )

    cfl_fac = _cfl_factor_2d(fd_order)
    dt_crit_ms = (cfl_fac * min(dx, dz) / v_max) * 1000.0 if v_max > 0 else 0.0
    nt_internal = int(math.ceil(ms / dt_crit_ms)) + 1 if dt_crit_ms > 0 else 1

    f_nyquist_hz: float | None = None
    if fs_ms is not None and fs_ms > 0:
        nt_out = int(round(ms / fs_ms)) + 1
        f_nyquist_hz = 500.0 / fs_ms
        if f_nyquist_hz < 1.5 * f0:
            checks.append(
                CheckResult(
                    "temporal_nyquist",
                    "FAIL",
                    f"Temporal aliasing: fs_ms={fs_ms} ms gives Nyquist {f_nyquist_hz:.1f} Hz < 1.5*f0 ({1.5 * f0:.1f} Hz).",
                )
            )
        elif f_nyquist_hz < f_max_hz:
            checks.append(
                CheckResult(
                    "temporal_nyquist",
                    "WARN",
                    f"Tight temporal sampling: Nyquist {f_nyquist_hz:.1f} Hz < f_max=2.5*f0 ({f_max_hz:.1f} Hz).",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "temporal_nyquist",
                    "PASS",
                    f"Temporal sampling OK: fs_ms={fs_ms} ms -> Nyquist {f_nyquist_hz:.1f} Hz >= {f_max_hz:.1f} Hz.",
                )
            )
    else:
        nt_out = nt_internal
        checks.append(
            CheckResult(
                "temporal_nyquist",
                "PASS",
                f"Native solver dt (~{dt_crit_ms:.4f} ms, ~{nt_internal} steps); no output resampling.",
            )
        )

    # --- Spatial aliasing check ---
    spatial_aliasing_limit_m = lambda_min_m / 2.0
    if abs(group_offset) > spatial_aliasing_limit_m and n_receivers > 1:
        checks.append(
            CheckResult(
                "spatial_aliasing",
                "WARN",
                f"group_offset ({abs(group_offset):.2f} m) > lambda_min/2 ({spatial_aliasing_limit_m:.2f} m); "
                f"steep arrivals at {f_max_hz:.1f} Hz may be spatially aliased.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "spatial_aliasing",
                "PASS",
                f"Receiver spacing ({abs(group_offset):.2f} m) <= lambda_min/2 ({spatial_aliasing_limit_m:.2f} m).",
            )
        )

    # --- Two-way traveltime vs recording duration ---
    depth_m = nz * dz
    twt_vertical_ms = (2.0 * depth_m / v_mean) * 1000.0 if v_mean > 0 else 0.0
    twt_max_offset_ms = (2.0 * math.hypot(geom["max_offset_m"] / 2.0, depth_m) / v_mean) * 1000.0 if v_mean > 0 else 0.0
    if ms < twt_vertical_ms:
        checks.append(
            CheckResult(
                "recording_twt",
                "WARN",
                f"Recording duration ({ms:.1f} ms) < vertical TWT to model bottom ({twt_vertical_ms:.1f} ms); "
                f"deep reflectors will be truncated.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "recording_twt",
                "PASS",
                f"Recording duration ({ms:.1f} ms) >= vertical TWT to bottom ({twt_vertical_ms:.1f} ms; "
                f"max-offset TWT ~{twt_max_offset_ms:.1f} ms).",
            )
        )

    # --- Memory & disk footprint ---
    effective_nx = max(nx, nx + overflow_x_cells)
    padded_nx = effective_nx + 2 * n_damping
    padded_nz = nz + 2 * n_damping
    total_traces = n_sources * n_receivers
    shot_ram_bytes = total_traces * nt_out * float_bytes
    wavefield_ram_bytes = 3 * padded_nx * padded_nz * float_bytes
    segy_bytes = 3600 + total_traces * (240 + nt_out * 4)
    h5_bytes = shot_ram_bytes + (effective_nx * nz * float_bytes)

    est_shot_ram_mb = shot_ram_bytes / (1024.0 ** 2)
    est_wavefield_ram_mb = wavefield_ram_bytes / (1024.0 ** 2)
    est_segy_mb = segy_bytes / (1024.0 ** 2)
    est_h5_mb = h5_bytes / (1024.0 ** 2)

    # --- Output directory & MAX_PATH preview ---
    cfg_for_dir = dict(cfg)
    cfg_for_dir.update(
        {
            "nx": nx,
            "nz": nz,
            "dx": dx,
            "dz": dz,
            "n_sources": n_sources,
            "n_receivers": n_receivers,
            "f0": f0,
            "ms": ms,
            "fs_ms": fs_ms,
            "group_offset": group_offset,
            "shot_offset": shot_offset,
            "gather": gather,
        }
    )
    output_dir = resolve_simulation_output_dir(cfg_for_dir, default_base_dir="data")
    output_dir_abs = output_dir.resolve()
    output_dir_abs_len = len(str(output_dir_abs))
    est_artifact_path_len = output_dir_abs_len + TYPICAL_ARTIFACT_SUFFIX_LEN

    if est_artifact_path_len > MAX_PATH_LIMIT:
        checks.append(
            CheckResult(
                "output_path",
                "FAIL",
                f"Estimated artifact path length ({est_artifact_path_len} chars) exceeds MAX_PATH ({MAX_PATH_LIMIT}).",
            )
        )
    elif est_artifact_path_len > 240:
        checks.append(
            CheckResult(
                "output_path",
                "WARN",
                f"Estimated artifact path length ({est_artifact_path_len} chars) is close to MAX_PATH ({MAX_PATH_LIMIT}).",
            )
        )
    else:
        checks.append(
            CheckResult(
                "output_path",
                "PASS",
                f"Resolved output dir '{output_dir}' (est. artifact path {est_artifact_path_len}/{MAX_PATH_LIMIT} chars).",
            )
        )

    statuses = {c.status for c in checks}
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    return ConfigReport(
        config_path=str(config_path),
        task=task,
        model_source=model_source,
        velocity_source=velocity_source,
        gather=gather,
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        origin=origin,
        model_min_x=model_min_x,
        model_max_x=model_max_x,
        model_min_z=model_min_z,
        model_max_z=model_max_z,
        v_min=v_min,
        v_max=v_max,
        v_mean=v_mean,
        n_sources=n_sources,
        n_receivers=n_receivers,
        shot_offset=shot_offset,
        group_offset=group_offset,
        src_origin=src_origin,
        rec_origin=rec_origin,
        first_src_x=geom["first_src_x"],
        last_src_x=geom["last_src_x"],
        first_rec_x=geom["first_rec_x"],
        latest_rec_x=geom["latest_rec_x"],
        max_reach_x=max_reach_x,
        margin_x_m=margin_x_m,
        overflow_x_m=overflow_x_m,
        overflow_x_cells=overflow_x_cells,
        max_sources_fit=geom["max_sources_fit"],
        max_sources_by_rec=geom["max_sources_by_rec"],
        max_sources_by_src=geom["max_sources_by_src"],
        spread_length_m=geom["spread_length_m"],
        min_offset_m=geom["min_offset_m"],
        max_offset_m=geom["max_offset_m"],
        nominal_cmp_fold=geom["nominal_cmp_fold"],
        cmp_bin_size_m=geom["cmp_bin_size_m"],
        spatial_aliasing_limit_m=spatial_aliasing_limit_m,
        f0_hz=f0,
        f_max_hz=f_max_hz,
        fd_order=fd_order,
        n_damping=n_damping,
        lambda_min_m=lambda_min_m,
        ppw=ppw,
        dt_crit_ms=dt_crit_ms,
        ms=ms,
        fs_ms=fs_ms,
        nt_internal=nt_internal,
        nt_out=nt_out,
        f_nyquist_hz=f_nyquist_hz,
        twt_vertical_ms=twt_vertical_ms,
        twt_max_offset_ms=twt_max_offset_ms,
        padded_nx=padded_nx,
        padded_nz=padded_nz,
        total_traces=total_traces,
        float_bytes=float_bytes,
        est_shot_ram_mb=est_shot_ram_mb,
        est_wavefield_ram_mb=est_wavefield_ram_mb,
        est_segy_mb=est_segy_mb,
        est_h5_mb=est_h5_mb,
        output_dir=str(output_dir),
        output_dir_abs_len=output_dir_abs_len,
        est_artifact_path_len=est_artifact_path_len,
        checks=checks,
        overall_status=overall,
    )


def format_report(r: ConfigReport) -> str:
    sep = "=" * 78
    lines = [
        sep,
        f"CONFIG: {r.config_path}  [{r.overall_status}]",
        sep,
        f"  Task / Gather       : {r.task} / {r.gather}",
        f"  Velocity Model      : {r.model_source} ({r.velocity_source})",
        f"  Grid (nx x nz)      : {r.nx} x {r.nz}  (dx={r.dx:g} m, dz={r.dz:g} m)",
        f"  Physical Extent     : X=[{r.model_min_x:.1f}, {r.model_max_x:.1f}] m, "
        f"Z=[{r.model_min_z:.1f}, {r.model_max_z:.1f}] m",
        f"  Velocity [m/s]      : min={r.v_min:.1f}, mean={r.v_mean:.1f}, max={r.v_max:.1f}",
        "",
        "  [Acquisition Reach & Capacity]",
        f"  Sources             : {r.n_sources}  (origin={r.src_origin}, shot_offset={r.shot_offset:g} m)",
        f"  Receivers / Shot    : {r.n_receivers}  (origin={r.rec_origin}, group_offset={r.group_offset:g} m)",
        f"  Source X Range      : [{r.first_src_x:.2f}, {r.last_src_x:.2f}] m",
        f"  Receiver X Range    : [{r.first_rec_x:.2f}, {r.latest_rec_x:.2f}] m  <-- latest receiver X",
        (
            f"  X Margin / Overflow : OVERFLOW by {r.overflow_x_m:.2f} m ({r.overflow_x_cells} cells)"
            if r.overflow_x_m > 0
            else f"  X Margin / Overflow : +{r.margin_x_m:.2f} m remaining inside model"
        ),
        (
            f"  Max Fitting Sources : {r.max_sources_fit}  "
            + (
                f"(rec-limited={r.max_sources_by_rec}, src-limited={r.max_sources_by_src})"
                if r.max_sources_by_rec is not None and r.max_sources_by_src is not None
                else ""
            )
            if r.max_sources_fit is not None
            else "  Max Fitting Sources : N/A (explicit sources or shot_offset <= 0)"
        ),
        "",
        "  [Survey Geometry]",
        f"  Spread Length       : {r.spread_length_m:.2f} m",
        f"  Offset Range        : [{r.min_offset_m:.2f}, {r.max_offset_m:.2f}] m",
        (
            f"  Nominal CMP Fold    : {r.nominal_cmp_fold:.1f}  (CMP bin={r.cmp_bin_size_m:.2f} m)"
            if r.nominal_cmp_fold is not None
            else "  Nominal CMP Fold    : N/A"
        ),
        f"  Spatial Alias Limit : {r.spatial_aliasing_limit_m:.2f} m  (group_offset={abs(r.group_offset):.2f} m)",
        "",
        "  [Numerical Stability & Traveltime]",
        f"  Wavelet / FD Order  : f0={r.f0_hz:g} Hz (f_max~{r.f_max_hz:g} Hz), fd_order={r.fd_order}, n_damping={r.n_damping}",
        f"  Dispersion (PPW)    : {r.ppw:.2f} pts/wavelength  (lambda_min={r.lambda_min_m:.2f} m)",
        f"  CFL dt_crit         : {r.dt_crit_ms:.4f} ms  (~{r.nt_internal} internal steps for {r.ms:g} ms)",
        (
            f"  Output Sampling     : fs_ms={r.fs_ms:g} ms ({r.nt_out} samples, Nyquist={r.f_nyquist_hz:.1f} Hz)"
            if r.fs_ms is not None
            else f"  Output Sampling     : native dt (~{r.nt_out} samples)"
        ),
        f"  Est. TWT to Bottom  : vertical={r.twt_vertical_ms:.1f} ms, max-offset={r.twt_max_offset_ms:.1f} ms (ms={r.ms:g} ms)",
        "",
        "  [Footprint & Output Path]",
        f"  Padded Grid         : {r.padded_nx} x {r.padded_nz}  |  Total Traces: {r.total_traces:,}",
        f"  Est. Memory (RAM)   : shots={r.est_shot_ram_mb:.2f} MB, wavefield={r.est_wavefield_ram_mb:.2f} MB",
        f"  Est. Disk Output    : traces.segy~{r.est_segy_mb:.2f} MB, results.h5~{r.est_h5_mb:.2f} MB",
        f"  Output Directory    : {r.output_dir}  (est. artifact path {r.est_artifact_path_len}/{MAX_PATH_LIMIT} chars)",
        "",
        "  [Sanity Checks]",
    ]
    for chk in r.checks:
        lines.append(f"  [{chk.status:4s}] {chk.name:18s}: {chk.message}")
    return "\n".join(lines)


def format_summary_table(reports: list[ConfigReport]) -> str:
    headers = ["Status", "Config", "Model X (m)", "Latest Rec X", "Margin/Over", "Srcs", "MaxSrcs", "PPW"]
    rows = []
    for r in reports:
        margin_str = f"-{r.overflow_x_m:.1f}m" if r.overflow_x_m > 0 else f"+{r.margin_x_m:.1f}m"
        max_src_str = str(r.max_sources_fit) if r.max_sources_fit is not None else "N/A"
        rows.append(
            [
                r.overall_status,
                Path(r.config_path).name,
                f"{r.model_max_x:.1f}",
                f"{r.latest_rec_x:.1f}",
                margin_str,
                str(r.n_sources),
                max_src_str,
                f"{r.ppw:.1f}",
            ]
        )

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = "-" * (sum(widths) + 3 * (len(headers) - 1))
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    body_lines = [" | ".join(row[i].ljust(widths[i]) for i, row in enumerate(rows)) for row in rows]
    return "\n".join(["", "SUMMARY", sep, header_line, sep, *body_lines, sep])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect shotgen JSON/YAML config files for receiver reach, max fitting sources, and sanity checks."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSON/YAML configuration files or directories containing them.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively scan directories for config files.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip loading velocity model files and use config nx/nz/dx/dz/v_base directly.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit machine-readable JSON output instead of formatted text.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any config reports a FAIL check.",
    )
    args = parser.parse_args(argv)

    try:
        config_files = discover_config_files(args.paths, recursive=args.recursive)
    except FileNotFoundError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2

    if not config_files:
        sys.stderr.write("Error: No JSON/YAML configuration files found.\n")
        return 2

    reports: list[ConfigReport] = []
    for cfg_file in config_files:
        try:
            reports.append(analyze_config(cfg_file, fast=args.fast))
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"Error parsing '{cfg_file}': {exc}\n")
            return 2

    if args.as_json:
        payload = [asdict(r) for r in reports]
        sys.stdout.write(json.dumps(payload if len(payload) > 1 else payload[0], indent=2) + "\n")
    else:
        for r in reports:
            sys.stdout.write(format_report(r) + "\n\n")
        if len(reports) > 1:
            sys.stdout.write(format_summary_table(reports) + "\n")

    if args.strict and any(r.overall_status == "FAIL" for r in reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
