"""GPU runtime and VRAM estimation and calibration for shotgen-gpu."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from sympy import finite_diff_weights as fd_w

from shotgen_gpu.config import configure_devito_device, detect_device

CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.json"

# Generic fallback parameters for uncalibrated NVIDIA GPUs (conservative mid-range baseline)
DEFAULT_FALLBACK_PROFILE: dict[str, Any] = {
    "simulation": {
        "jit_compile_s": 4.5,
        "per_shot_overhead_s": 0.35,
        "gcells_per_s_order4": 0.35,
        "order_exponent": 0.85,
        "float64_factor": 2.0,
        "save_wavefield_multiplier": 1.65,
    },
    "rtm": {
        "jit_compile_s": 5.5,
        "per_shot_overhead_s": 0.25,
        "gcells_per_s_order4": 0.30,
        "order_exponent": 0.85,
        "float64_factor": 2.0,
        "save_wavefield_multiplier": 1.35,
    },
}

# Conservative disk I/O throughput (MB/s) for SEG-Y/HDF5 writes (accounting for WSL2/NTFS overhead)
DEFAULT_IO_MB_PER_S = 150.0
# Base CUDA / OpenACC context memory overhead (MiB)
CUDA_CONTEXT_OVERHEAD_MIB = 400.0


@dataclass
class GPUInfo:
    name: str
    vram_mib: float
    compute_cap: str
    is_gpu: bool


@dataclass
class WorkloadSpec:
    task: str
    nx: int
    nz: int
    dx: float
    dz: float
    nbl: int
    fd_order: int
    n_sources: int
    n_receivers: int
    ms: float
    dt_ms: float
    nt: int
    max_vp_m_s: float
    dtype_name: str
    itemsize: int
    save_wavefield: bool
    fs_ms: float | None = None
    snr: float | None = None
    model_label: str = "synthetic"

    @property
    def padded_nx(self) -> int:
        return self.nx + 2 * self.nbl

    @property
    def padded_nz(self) -> int:
        return self.nz + 2 * self.nbl

    @property
    def padded_cells(self) -> int:
        return self.padded_nx * self.padded_nz

    @property
    def total_cell_steps(self) -> int:
        return self.padded_cells * self.nt * self.n_sources


@dataclass
class RuntimeEstimate:
    gpu: GPUInfo
    workload: WorkloadSpec
    calibration_source: str
    calibrated_at: str | None
    jit_cold_s: float
    jit_warm_s: float
    setup_overhead_s: float
    compute_s: float
    io_s: float
    total_warm_s: float
    total_cold_s: float
    per_shot_s: float
    uncertainty_frac: float
    estimated_vram_mib: float
    vram_warning: str | None
    effective_gcells_per_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu": asdict(self.gpu),
            "workload": {
                **asdict(self.workload),
                "padded_nx": self.workload.padded_nx,
                "padded_nz": self.workload.padded_nz,
                "padded_cells": self.workload.padded_cells,
                "total_cell_steps": self.workload.total_cell_steps,
            },
            "calibration_source": self.calibration_source,
            "calibrated_at": self.calibrated_at,
            "jit_cold_s": self.jit_cold_s,
            "jit_warm_s": self.jit_warm_s,
            "setup_overhead_s": self.setup_overhead_s,
            "compute_s": self.compute_s,
            "io_s": self.io_s,
            "total_warm_s": self.total_warm_s,
            "total_cold_s": self.total_cold_s,
            "per_shot_s": self.per_shot_s,
            "uncertainty_frac": self.uncertainty_frac,
            "estimated_vram_mib": self.estimated_vram_mib,
            "vram_warning": self.vram_warning,
            "effective_gcells_per_s": self.effective_gcells_per_s,
        }


def detect_gpu_info(device: str = "auto") -> GPUInfo:
    """Query connected NVIDIA GPU name, VRAM (MiB), and compute capability via nvidia-smi."""
    if device is not None and device.lower() == "cpu":
        return GPUInfo(name="CPU (explicit)", vram_mib=0.0, compute_cap="N/A", is_gpu=False)

    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                first_line = res.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first_line.split(",")]
                if len(parts) >= 3:
                    name = parts[0]
                    vram_mib = float(parts[1])
                    compute_cap = parts[2]
                    return GPUInfo(
                        name=name,
                        vram_mib=vram_mib,
                        compute_cap=compute_cap,
                        is_gpu=True,
                    )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    detected = detect_device()
    if detected == "cuda":
        return GPUInfo(name="NVIDIA GPU (unknown model)", vram_mib=0.0, compute_cap="unknown", is_gpu=True)
    return GPUInfo(name="CPU (no NVIDIA GPU detected)", vram_mib=0.0, compute_cap="N/A", is_gpu=False)


def compute_cfl_dt_nt(
    max_vp_m_s: float,
    dx: float,
    dz: float,
    space_order: int,
    ms: float,
    dtype=np.float32,
    dt_scale: float = 1.0,
) -> tuple[float, int]:
    """
    Compute exact Devito CFL time step (in ms) and number of time steps `nt`.

    Matches `examples.seismic.Model.critical_dt` and `examples.seismic.TimeAxis`.
    """
    if max_vp_m_s <= 0.0:
        raise ValueError(f"max_vp_m_s must be positive, got {max_vp_m_s}")
    min_spacing = min(float(dx), float(dz))
    if min_spacing <= 0.0:
        raise ValueError(f"Grid spacing must be positive, got dx={dx}, dz={dz}")

    # 2D acoustic 2nd-order in time CFL coefficient (matches Devito Model._cfl_coeff)
    coeffs = fd_w(2, range(-int(space_order), int(space_order) + 1), 0)[-1][-1]
    coeff_sum = sum(abs(float(c)) for c in coeffs)
    cfl_coeff = math.sqrt(4.0 / (2.0 * coeff_sum))

    # Devito scales vp to km/s while spacing is in meters -> dt is in ms
    max_vp_km_s = float(max_vp_m_s) * 1e-3
    dt_raw = cfl_coeff * min_spacing / max_vp_km_s
    np_dtype = np.dtype(dtype).type
    dt_ms = float(np_dtype(f"{dt_scale * dt_raw:.3e}"))

    # Matches Devito TimeAxis(start=0.0, stop=ms, step=dt_ms)
    nt = math.ceil((float(ms) + dt_ms) / dt_ms)
    return dt_ms, nt


def load_calibration_db(calibration_path: Path = CALIBRATION_FILE) -> dict[str, Any]:
    """Load the in-repo calibration JSON database if it exists."""
    if calibration_path.is_file():
        try:
            with open(calibration_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("gpus"), dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {"schema_version": 1, "gpus": {}}


def save_calibration_db(db: dict[str, Any], calibration_path: Path = CALIBRATION_FILE) -> None:
    """Persist the calibration database to calibration.json."""
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, sort_keys=True)
        f.write("\n")


def get_task_calibration(
    gpu: GPUInfo,
    task: str,
    calibration_path: Path = CALIBRATION_FILE,
) -> tuple[dict[str, float], str, str | None, float]:
    """
    Look up calibration parameters for the given GPU and task.

    Returns (params, source_label, calibrated_at, uncertainty_frac).
    """
    db = load_calibration_db(calibration_path)
    gpus = db.get("gpus", {})

    entry = gpus.get(gpu.name)
    if entry is None:
        # Case-insensitive match fallback
        gpu_lower = gpu.name.lower()
        for key, val in gpus.items():
            if key.lower() == gpu_lower:
                entry = val
                break

    if entry and isinstance(entry.get(task), dict):
        merged = {**DEFAULT_FALLBACK_PROFILE[task], **entry[task]}
        return (
            merged,
            f"calibrated ({calibration_path.name})",
            entry.get("calibrated_at"),
            0.20,
        )

    return (
        dict(DEFAULT_FALLBACK_PROFILE[task]),
        "fallback (uncalibrated — run `shotgen-gpu --calibrate`)",
        None,
        0.40,
    )


def _parse_val(val: Any, default: Any, val_type=float) -> Any:
    if val is None or (isinstance(val, str) and val.strip().lower() in ("none", "null")):
        return default
    return val_type(val)


def extract_simulation_workload(cfg: dict[str, Any]) -> WorkloadSpec:
    """Extract exact grid, CFL, and acquisition dimensions for a `simulation` config."""
    from shotgen_gpu.cli import _load_velocity_model

    vel, nx, nz, dx, dz = _load_velocity_model(cfg)

    n_sources = _parse_val(cfg.get("n_sources"), 2, int)
    n_receivers = _parse_val(cfg.get("n_receivers"), 10, int)
    ms = _parse_val(cfg.get("ms", cfg.get("ntime")), 300.0, float)
    fd_order = _parse_val(cfg.get("fd_order"), 4, int)
    n_damping = _parse_val(cfg.get("n_damping"), 40, int)
    smooth = _parse_val(cfg.get("smooth"), 5.0, float)
    origin = tuple(cfg.get("origin", [0.0, 0.0]))
    snr = _parse_val(cfg.get("snr"), None, float)
    fs_ms = _parse_val(cfg.get("fs_ms"), None, float)
    save_wavefield = bool(cfg.get("save_wavefield", False))

    float_type_str = str(cfg.get("float_type", "float32")).lower()
    float_type = np.float64 if "64" in float_type_str else np.float32

    max_size = cfg.get("max_size")
    if max_size is not None:
        max_x, max_z = float(max_size[0]), float(max_size[1])
        model_len_x = vel.shape[0] * dx
        model_len_z = vel.shape[1] * dz
        if max_z > model_len_z and max_x <= model_len_z and max_z <= model_len_x:
            max_x, max_z = max_z, max_x
        x_start = max(0, min(int(np.round(origin[0] / dx)), vel.shape[0]))
        x_end = max(x_start, min(x_start + int(np.round(max_x / dx)), vel.shape[0]))
        z_start = max(0, min(int(np.round(origin[1] / dz)), vel.shape[1]))
        z_end = max(z_start, min(z_start + int(np.round(max_z / dz)), vel.shape[1]))
        if x_end > x_start and z_end > z_start:
            vel = vel[x_start:x_end, z_start:z_end]
            nx = vel.shape[0]
            nz = vel.shape[1]

    if "sources" in cfg and cfg["sources"] is not None:
        n_sources = len(cfg["sources"])
    if "receivers" in cfg and cfg["receivers"] is not None:
        recs = np.asarray(cfg["receivers"])
        n_receivers = recs.shape[1] if recs.ndim == 3 else len(recs)

    # AcousticWaveSolverWrapper passes smoothed v0 into AcousticWave2D
    v0 = gaussian_filter(vel, sigma=smooth) if smooth > 0 else vel
    max_vp = float(np.max(v0))

    # AcousticWave2D creates Model with dtype=np.float32 internally
    dt_ms, nt = compute_cfl_dt_nt(
        max_vp_m_s=max_vp,
        dx=dx,
        dz=dz,
        space_order=fd_order,
        ms=ms,
        dtype=np.float32,
    )

    model_label = str(
        cfg.get("velocity_model")
        or cfg.get("vel_file")
        or cfg.get("vel_model")
        or cfg.get("model")
        or "synthetic"
    )

    return WorkloadSpec(
        task="simulation",
        nx=int(nx),
        nz=int(nz),
        dx=float(dx),
        dz=float(dz),
        nbl=int(n_damping),
        fd_order=int(fd_order),
        n_sources=int(n_sources),
        n_receivers=int(n_receivers),
        ms=float(ms),
        dt_ms=dt_ms,
        nt=nt,
        max_vp_m_s=max_vp,
        dtype_name=np.dtype(float_type).name,
        itemsize=np.dtype(float_type).itemsize,
        save_wavefield=save_wavefield,
        fs_ms=fs_ms,
        snr=snr,
        model_label=model_label,
    )


def extract_rtm_workload(cfg: dict[str, Any]) -> WorkloadSpec:
    """Extract exact grid, CFL, and acquisition dimensions for an `rtm` config."""
    dataset_dir = cfg.get("dataset_dir")
    nbl = _parse_val(cfg.get("nbl", cfg.get("n_damping")), 40, int)
    space_order = _parse_val(cfg.get("space_order", cfg.get("fd_order")), 4, int)
    save_wavefield = bool(cfg.get("save_wavefield", False))
    float_type_str = str(cfg.get("float_type", "float32")).lower()
    float_type = np.float64 if "64" in float_type_str else np.float32

    if dataset_dir is not None:
        from shotgen_gpu.engine.rtm import load_dataset_dir_gpu

        vp, sources, receivers, _shots, time_arr, _f0, dx, dz = load_dataset_dir_gpu(
            dataset_dir, require_f0=False, verbose=False
        )
        nx, nz = vp.shape
        n_sources = int(sources.shape[0])
        n_receivers = int(receivers.shape[1] if receivers.ndim == 3 else receivers.shape[0])
        is_seconds = (len(time_arr) > 1 and (time_arr[1] - time_arr[0]) < 0.1) or (time_arr[-1] < 100.0)
        t0 = float(time_arr[0] * 1000.0 if is_seconds else time_arr[0])
        tn = float(time_arr[-1] * 1000.0 if is_seconds else time_arr[-1])
        ms = tn - t0
        max_vp = float(np.max(vp))
        model_label = str(Path(dataset_dir).name)
    else:
        from shotgen_gpu.cli import _load_velocity_model

        vel, nx, nz, dx, dz = _load_velocity_model(cfg)
        n_sources = _parse_val(cfg.get("n_sources"), 2, int)
        n_receivers = _parse_val(cfg.get("n_receivers"), 10, int)
        ms = _parse_val(cfg.get("ms", cfg.get("tn")), 300.0, float)
        max_vp = float(np.max(vel))
        model_label = str(
            cfg.get("velocity_model")
            or cfg.get("vel_file")
            or cfg.get("vel_model")
            or cfg.get("model")
            or "synthetic"
        )

    dt_ms, nt = compute_cfl_dt_nt(
        max_vp_m_s=max_vp,
        dx=dx,
        dz=dz,
        space_order=space_order,
        ms=ms,
        dtype=float_type,
    )

    return WorkloadSpec(
        task="rtm",
        nx=int(nx),
        nz=int(nz),
        dx=float(dx),
        dz=float(dz),
        nbl=int(nbl),
        fd_order=int(space_order),
        n_sources=int(n_sources),
        n_receivers=int(n_receivers),
        ms=float(ms),
        dt_ms=dt_ms,
        nt=nt,
        max_vp_m_s=max_vp,
        dtype_name=np.dtype(float_type).name,
        itemsize=np.dtype(float_type).itemsize,
        save_wavefield=save_wavefield,
        model_label=model_label,
    )


def estimate_vram_mib(workload: WorkloadSpec) -> float:
    """
    Estimate peak GPU VRAM usage in MiB for the wavefield propagation kernel.

    Accounts for Devito padded grids (including halo padding), wavefield time buffers,
    receiver trace buffers, optional saved wavefields, and CUDA/OpenACC context overhead.
    """
    halo_nx = workload.padded_nx + 2 * workload.fd_order
    halo_nz = workload.padded_nz + 2 * workload.fd_order
    grid_bytes = halo_nx * halo_nz * workload.itemsize

    # Static fields (vp, damp, dm/image ~3) + two 2nd-order TimeFunctions (2 * 3 slices = 6)
    active_grids = 9
    total_bytes = active_grids * grid_bytes

    if workload.save_wavefield:
        # Full time-history wavefield saved on device/host
        total_bytes += workload.nt * grid_bytes

    # Receiver traces buffer for a single shot (float32/float64)
    total_bytes += workload.n_receivers * workload.nt * workload.itemsize

    return CUDA_CONTEXT_OVERHEAD_MIB + (total_bytes / (1024.0 * 1024.0))


def estimate_runtime(
    cfg: dict[str, Any],
    calibration_path: Path = CALIBRATION_FILE,
) -> RuntimeEstimate:
    """Compute a full GPU runtime and VRAM estimate for a shotgen-gpu config dictionary."""
    task = str(cfg.get("task", "simulation")).lower()
    if task not in ("simulation", "rtm"):
        raise ValueError(f"Unsupported task '{task}' for runtime estimation (expected 'simulation' or 'rtm').")

    device_pref = cfg.get("device", "auto")
    gpu = detect_gpu_info(device_pref)

    workload = extract_simulation_workload(cfg) if task == "simulation" else extract_rtm_workload(cfg)
    params, cal_source, calibrated_at, uncertainty_frac = get_task_calibration(
        gpu, task, calibration_path=calibration_path
    )

    # Effective throughput adjusted for FD space order and precision
    base_gcells_per_s = max(float(params["gcells_per_s_order4"]), 1e-4)
    order_exp = float(params.get("order_exponent", 0.85))
    order_scale = ((workload.fd_order + 2.0) / 6.0) ** order_exp
    dtype_scale = float(params.get("float64_factor", 2.0)) if workload.itemsize == 8 else 1.0
    save_wav_scale = (
        float(params.get("save_wavefield_multiplier", 1.5)) if workload.save_wavefield else 1.0
    )

    effective_gcells_per_s = base_gcells_per_s / (order_scale * dtype_scale * save_wav_scale)

    # Phase times
    jit_cold_s = float(params["jit_compile_s"])
    jit_warm_s = min(1.0, jit_cold_s * 0.2)
    setup_overhead_s = workload.n_sources * float(params["per_shot_overhead_s"])
    compute_s = (workload.total_cell_steps / 1e9) / effective_gcells_per_s

    # I/O & post-processing estimate
    if task == "simulation":
        if workload.fs_ms is not None and workload.fs_ms > 0:
            nt_out = round(workload.ms / workload.fs_ms) + 1
        else:
            nt_out = workload.nt
        trace_bytes = workload.n_sources * workload.n_receivers * nt_out * 4
        vel_bytes = 2 * workload.nx * workload.nz * 4
        io_s = (trace_bytes + vel_bytes) / (DEFAULT_IO_MB_PER_S * 1024.0 * 1024.0)
        if workload.fs_ms is not None and workload.fs_ms > 0:
            # Linear resampling pass over traces
            io_s += (workload.n_sources * workload.n_receivers * workload.nt) / 2.5e7
    else:
        image_bytes = workload.nx * workload.nz * workload.itemsize
        io_s = image_bytes / (DEFAULT_IO_MB_PER_S * 1024.0 * 1024.0)

    total_warm_s = jit_warm_s + setup_overhead_s + compute_s + io_s
    total_cold_s = jit_cold_s + setup_overhead_s + compute_s + io_s
    per_shot_s = (setup_overhead_s + compute_s) / max(workload.n_sources, 1)

    est_vram_mib = estimate_vram_mib(workload)
    vram_warning: str | None = None
    if gpu.is_gpu and gpu.vram_mib > 0:
        ratio = est_vram_mib / gpu.vram_mib
        if ratio >= 1.0:
            vram_warning = (
                f"CRITICAL: Estimated peak VRAM ({est_vram_mib:,.0f} MiB) exceeds "
                f"{gpu.name} capacity ({gpu.vram_mib:,.0f} MiB) — likely OOM!"
            )
        elif ratio >= 0.85:
            vram_warning = (
                f"WARNING: Estimated peak VRAM ({est_vram_mib:,.0f} MiB) is {ratio * 100:.0f}% "
                f"of {gpu.name} capacity ({gpu.vram_mib:,.0f} MiB)."
            )

    return RuntimeEstimate(
        gpu=gpu,
        workload=workload,
        calibration_source=cal_source,
        calibrated_at=calibrated_at,
        jit_cold_s=jit_cold_s,
        jit_warm_s=jit_warm_s,
        setup_overhead_s=setup_overhead_s,
        compute_s=compute_s,
        io_s=io_s,
        total_warm_s=total_warm_s,
        total_cold_s=total_cold_s,
        per_shot_s=per_shot_s,
        uncertainty_frac=uncertainty_frac,
        estimated_vram_mib=est_vram_mib,
        vram_warning=vram_warning,
        effective_gcells_per_s=effective_gcells_per_s,
    )


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-friendly string."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    if seconds < 3600.0:
        mins = int(seconds // 60)
        rem = seconds % 60
        return f"{mins}m {rem:04.1f}s"
    hours = int(seconds // 3600)
    rem_mins = int((seconds % 3600) // 60)
    rem_s = round(seconds % 60)
    return f"{hours}h {rem_mins:02d}m {rem_s:02d}s"


def format_estimate_report(est: RuntimeEstimate) -> str:
    """Format a RuntimeEstimate into a detailed human-readable terminal report."""
    w = est.workload
    g = est.gpu

    low_s = est.total_warm_s * (1.0 - est.uncertainty_frac)
    high_s = est.total_warm_s * (1.0 + est.uncertainty_frac)
    pct = round(est.uncertainty_frac * 100)

    vram_str = (
        f"{est.estimated_vram_mib:,.0f} MiB / {g.vram_mib:,.0f} MiB "
        f"({100.0 * est.estimated_vram_mib / g.vram_mib:.1f}%)"
        if g.is_gpu and g.vram_mib > 0
        else f"{est.estimated_vram_mib:,.0f} MiB"
    )
    gpu_str = (
        f"{g.name} (CC {g.compute_cap}, {g.vram_mib:,.0f} MiB VRAM)"
        if g.is_gpu and g.vram_mib > 0
        else g.name
    )
    cal_str = est.calibration_source
    if est.calibrated_at:
        cal_str += f" @ {est.calibrated_at}"

    lines = [
        "=" * 72,
        f"  shotgen-gpu Runtime Estimate ({w.task.upper()})",
        "=" * 72,
        f"  Hardware       : {gpu_str}",
        f"  Calibration    : {cal_str}",
        "-" * 72,
        f"  Model          : {w.model_label}",
        f"  Domain Grid    : {w.nx} x {w.nz}  (dx={w.dx:g} m, dz={w.dz:g} m)",
        f"  Padded Grid    : {w.padded_nx} x {w.padded_nz}  (nbl={w.nbl}, {w.padded_cells:,} cells)",
        f"  Discretization : fd_order={w.fd_order}, dtype={w.dtype_name}, max_vp={w.max_vp_m_s:,.1f} m/s",
        f"  Time Stepping  : ms={w.ms:g} ms, CFL dt={w.dt_ms:.4f} ms -> nt={w.nt:,} steps",
        f"  Acquisition    : {w.n_sources} shots x {w.n_receivers} receivers (save_wavefield={w.save_wavefield})",
        f"  Total Work     : {w.total_cell_steps / 1e9:,.3f} Gcell-steps @ {est.effective_gcells_per_s:.3f} Gcells/s",
        "-" * 72,
        "  Phase Breakdown:",
        f"    JIT Compile (warm / cold) : {format_duration(est.jit_warm_s)} / {format_duration(est.jit_cold_s)}",
        f"    Per-shot Setup Overhead   : {format_duration(est.setup_overhead_s)} ({est.setup_overhead_s / max(w.n_sources, 1):.3f} s/shot)",
        f"    GPU Stencil Compute       : {format_duration(est.compute_s)} ({est.compute_s / max(w.n_sources, 1):.3f} s/shot)",
        f"    I/O & Post-processing     : {format_duration(est.io_s)}",
        "-" * 72,
        f"  Estimated Runtime (warm)    : {format_duration(est.total_warm_s)}  (+/-{pct}%: {format_duration(low_s)} - {format_duration(high_s)})",
        f"  Estimated Runtime (cold JIT): {format_duration(est.total_cold_s)}",
        f"  Avg Time per Shot           : {format_duration(est.per_shot_s)}/shot",
        f"  Estimated Peak VRAM         : {vram_str}",
    ]
    if est.vram_warning:
        lines.append("-" * 72)
        lines.append(f"  {est.vram_warning}")
    lines.append("=" * 72)
    return "\n".join(lines)


def _make_synthetic_vel(nx: int, nz: int) -> np.ndarray:
    vel = np.full((nx, nz), 2500.0, dtype=np.float32)
    vel[:, nz // 2 :] = 3200.0
    return vel


def _benchmark_simulation(device: str, verbose: bool = False) -> dict[str, float]:
    """Run a compact multi-size AcousticWaveSolverWrapper benchmark to fit simulation timing params."""
    from shotgen_gpu.engine.solver import AcousticWaveSolverWrapper

    def _run_case(nx: int, nz: int, nbl: int, fd_order: int, n_sources: int, ms: float) -> tuple[float, int]:
        vel = _make_synthetic_vel(nx, nz)
        dx = dz = 10.0
        sources = np.column_stack(
            (np.linspace(100.0, (nx - 10) * dx, n_sources), np.full(n_sources, 20.0))
        ).astype(np.float32)
        receivers = np.column_stack(
            (np.linspace(100.0, (nx - 10) * dx, 12), np.full(12, 20.0))
        ).astype(np.float32)

        solver = AcousticWaveSolverWrapper(
            nx=nx,
            nz=nz,
            dx=dx,
            dz=dz,
            vel=vel,
            sources=sources,
            receivers=receivers,
            f0=15.0,
            fd_order=fd_order,
            n_damping=nbl,
            smooth=2.0,
            device=device,
            float_type=np.float32,
            verbose=verbose,
        )
        t0 = time.perf_counter()
        solver.run(ms=ms, save_wavefield=False)
        elapsed = time.perf_counter() - t0

        v0 = gaussian_filter(vel, sigma=2.0)
        _, nt = compute_cfl_dt_nt(float(np.max(v0)), dx, dz, fd_order, ms, dtype=np.float32)
        padded_cells = (nx + 2 * nbl) * (nz + 2 * nbl)
        return elapsed, padded_cells * nt

    # 1. Cold/warmup single shot (order=4, small) + steady-state 2 shots (pins down overhead)
    t_warm1, w_small = _run_case(80, 80, nbl=20, fd_order=4, n_sources=1, ms=50.0)
    t_small2, _ = _run_case(80, 80, nbl=20, fd_order=4, n_sources=2, ms=50.0)
    per_shot_small = t_small2 / 2.0
    jit_est = max(0.5, t_warm1 - per_shot_small)

    # 2. Larger grid (order=4): 1-shot warmup, then 2 steady-state shots to isolate GPU throughput
    t_med_warm, w_med = _run_case(800, 800, nbl=40, fd_order=4, n_sources=1, ms=1500.0)
    t_med2, _ = _run_case(800, 800, nbl=40, fd_order=4, n_sources=2, ms=1500.0)
    per_shot_med = t_med2 / 2.0

    # Affine fit: per_shot_time = overhead + (work_gcells / throughput)
    dw_gcells = max((w_med - w_small) / 1e9, 1e-6)
    dt_per_shot = max(per_shot_med - per_shot_small, 1e-4)
    gcells_per_s_o4 = dw_gcells / dt_per_shot
    overhead_s = max(0.01, per_shot_small - (w_small / 1e9) / gcells_per_s_o4)

    # 3. Higher-order run (order=8): 1-shot warmup, then 2 steady-state shots
    t_o8_1, w_o8 = _run_case(800, 800, nbl=40, fd_order=8, n_sources=1, ms=1500.0)
    t_o8_2, _ = _run_case(800, 800, nbl=40, fd_order=8, n_sources=2, ms=1500.0)
    per_shot_o8 = t_o8_2 / 2.0
    compute_o8 = max(per_shot_o8 - overhead_s, 1e-4)
    gcells_per_s_o8 = (w_o8 / 1e9) / compute_o8

    # Fit order_exponent: gcells_o8 = gcells_o4 / ((8+2)/(4+2))**order_exp
    ratio = max(gcells_per_s_o4 / max(gcells_per_s_o8, 1e-6), 1.0)
    order_exp = float(np.clip(math.log(ratio) / math.log(10.0 / 6.0), 0.2, 2.5))
    jit_est = max(jit_est, max(0.5, t_med_warm - per_shot_med), max(0.5, t_o8_1 - per_shot_o8))

    return {
        "jit_compile_s": round(jit_est, 3),
        "per_shot_overhead_s": round(overhead_s, 4),
        "gcells_per_s_order4": round(gcells_per_s_o4, 4),
        "order_exponent": round(order_exp, 3),
        "float64_factor": 2.0,
        "save_wavefield_multiplier": 1.65,
    }


def _benchmark_rtm(device: str, verbose: bool = False) -> dict[str, float]:
    """Run a compact multi-size ReverseTimeMigration benchmark to fit RTM timing params."""
    import matplotlib
    matplotlib.use("Agg")
    from shotgen_gpu.engine.rtm import ReverseTimeMigration

    def _run_case(nx: int, nz: int, nbl: int, space_order: int, n_sources: int, ms: float) -> tuple[float, int]:
        vel = _make_synthetic_vel(nx, nz)
        dx = dz = 10.0
        _dt_ms, nt = compute_cfl_dt_nt(float(np.max(vel)), dx, dz, space_order, ms, dtype=np.float32)
        sources = np.column_stack(
            (np.linspace(100.0, (nx - 10) * dx, n_sources), np.full(n_sources, 20.0))
        ).astype(np.float32)
        receivers = np.column_stack(
            (np.linspace(100.0, (nx - 10) * dx, 12), np.full(12, 20.0))
        ).astype(np.float32)
        shots = np.zeros((n_sources, 12, nt), dtype=np.float32)
        time_arr = np.linspace(0.0, ms / 1000.0, nt, dtype=np.float32)

        rtm = ReverseTimeMigration(
            vp=vel,
            sources=sources,
            receivers=receivers,
            shots=shots,
            time=time_arr,
            spacing=(dx, dz),
            nbl=nbl,
            smooth_sigma=2.0,
            f0=15.0,
            space_order=space_order,
            dtype=np.float32,
            device=device,
            verbose=verbose,
        )
        t0 = time.perf_counter()
        rtm.run(save_wavefield=False, verbose=False)
        elapsed = time.perf_counter() - t0

        padded_cells = (nx + 2 * nbl) * (nz + 2 * nbl)
        return elapsed, padded_cells * rtm.geometry.nt

    t_warm1, w_small = _run_case(80, 80, nbl=20, space_order=4, n_sources=1, ms=50.0)
    t_small2, _ = _run_case(80, 80, nbl=20, space_order=4, n_sources=2, ms=50.0)
    per_shot_small = t_small2 / 2.0
    jit_est = max(0.5, t_warm1 - per_shot_small)

    t_med_warm, w_med = _run_case(800, 800, nbl=40, space_order=4, n_sources=1, ms=1500.0)
    t_med2, _ = _run_case(800, 800, nbl=40, space_order=4, n_sources=2, ms=1500.0)
    per_shot_med = t_med2 / 2.0

    dw_gcells = max((w_med - w_small) / 1e9, 1e-6)
    dt_per_shot = max(per_shot_med - per_shot_small, 1e-4)
    gcells_per_s_o4 = dw_gcells / dt_per_shot
    overhead_s = max(0.01, per_shot_small - (w_small / 1e9) / gcells_per_s_o4)

    t_o8_1, w_o8 = _run_case(800, 800, nbl=40, space_order=8, n_sources=1, ms=1500.0)
    t_o8_2, _ = _run_case(800, 800, nbl=40, space_order=8, n_sources=2, ms=1500.0)
    per_shot_o8 = t_o8_2 / 2.0
    compute_o8 = max(per_shot_o8 - overhead_s, 1e-4)
    gcells_per_s_o8 = (w_o8 / 1e9) / compute_o8

    ratio = max(gcells_per_s_o4 / max(gcells_per_s_o8, 1e-6), 1.0)
    order_exp = float(np.clip(math.log(ratio) / math.log(10.0 / 6.0), 0.2, 2.5))
    jit_est = max(jit_est, max(0.5, t_med_warm - per_shot_med), max(0.5, t_o8_1 - per_shot_o8))

    return {
        "jit_compile_s": round(jit_est, 3),
        "per_shot_overhead_s": round(overhead_s, 4),
        "gcells_per_s_order4": round(gcells_per_s_o4, 4),
        "order_exponent": round(order_exp, 3),
        "float64_factor": 2.0,
        "save_wavefield_multiplier": 1.35,
    }


def run_calibration(
    device: str = "auto",
    verbose: bool = False,
    calibration_path: Path = CALIBRATION_FILE,
) -> dict[str, Any]:
    """Run standalone synthetic GPU calibration for both simulation and RTM and update calibration.json."""
    configured_device = configure_devito_device(device, verbose=verbose)
    gpu = detect_gpu_info(configured_device)

    print(f"[shotgen-gpu calibrate] Device: {configured_device.upper()} | Hardware: {gpu.name}")
    print("[shotgen-gpu calibrate] Running synthetic forward simulation benchmarks...")
    sim_params = _benchmark_simulation(configured_device, verbose=verbose)
    print(
        f"[shotgen-gpu calibrate] Simulation fitted: "
        f"{sim_params['gcells_per_s_order4']:.3f} Gcells/s (order 4), "
        f"order_exp={sim_params['order_exponent']:.2f}, "
        f"overhead={sim_params['per_shot_overhead_s']:.3f} s/shot, "
        f"JIT~{sim_params['jit_compile_s']:.2f} s"
    )

    print("[shotgen-gpu calibrate] Running synthetic RTM benchmarks...")
    rtm_params = _benchmark_rtm(configured_device, verbose=verbose)
    print(
        f"[shotgen-gpu calibrate] RTM fitted:        "
        f"{rtm_params['gcells_per_s_order4']:.3f} Gcells/s (order 4), "
        f"order_exp={rtm_params['order_exponent']:.2f}, "
        f"overhead={rtm_params['per_shot_overhead_s']:.3f} s/shot, "
        f"JIT~{rtm_params['jit_compile_s']:.2f} s"
    )

    db = load_calibration_db(calibration_path)
    entry = {
        "vram_mib": gpu.vram_mib,
        "compute_cap": gpu.compute_cap,
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        "simulation": sim_params,
        "rtm": rtm_params,
    }
    db["gpus"][gpu.name] = entry
    save_calibration_db(db, calibration_path)
    print(f"[shotgen-gpu calibrate] Saved calibration profile for '{gpu.name}' to {calibration_path}")
    return entry
