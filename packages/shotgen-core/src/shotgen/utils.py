from pathlib import Path


def _format_num_for_filename(val) -> str:
    """
    Format a number for inclusion in filenames.
    If None, returns 'None'.
    If float is convertible to int without precision loss (e.g. 100.0 or 12.0),
    convert to int string ('100', '12').
    If it has decimal places (e.g. 12.5), replace period '.' with underscore '_'.
    """
    if val is None:
        return "None"
    if isinstance(val, (int, float)):
        # If float equals integer value, return integer representation
        if float(val).is_integer():
            return str(int(val))
        s = str(val)
        return s.replace(".", "_")
    # For strings or other types
    s = str(val).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s.replace(".", "_")
    except ValueError:
        return s.replace(".", "_")


def _extract_model_name(cfg: dict) -> str:
    vel_spec = cfg.get("velocity_model") or cfg.get("vel_file") or cfg.get("vel_model") or cfg.get("model")
    if vel_spec is None:
        return "synthetic"
    if isinstance(vel_spec, (str, Path)):
        p = Path(str(vel_spec).strip())
        name = p.stem.lower()
        clean = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        clean = clean.strip("_")
        return clean if clean else "synthetic"
    return "synthetic"


def generate_simulation_dir_name(cfg: dict, base_dir: str = "data") -> Path:
    """
    Generates a unique simulation output directory name based on parameters from a JSON/YAML config.

    Format order:
    model, gather, nx, nz, dx, dz, n_sources, n_receivers, ms, fs, f0, group_offset, shot_offset, snr

    Example:
    data/marmousi_commonshot_100nx_100nz_10dx_10dz_2src_10rec_300ms_fs_6ms_25Hz_1groupoffset_1shotoffset_noiseless
    """
    model = _extract_model_name(cfg)
    raw_gather = str(cfg.get("gather", "common shot"))
    gather_clean = raw_gather.replace(" ", "")

    nx = _format_num_for_filename(cfg.get("nx", 100))
    nz = _format_num_for_filename(cfg.get("nz", 100))
    dx = _format_num_for_filename(cfg.get("dx", 10.0))
    dz = _format_num_for_filename(cfg.get("dz", 10.0))
    n_sources = _format_num_for_filename(cfg.get("n_sources", 2))
    n_receivers = _format_num_for_filename(cfg.get("n_receivers", 10))
    ms = _format_num_for_filename(cfg.get("ms", cfg.get("ntime", 300.0)))

    fs_val = cfg.get("fs_ms")
    if fs_val is None:
        fs_val = cfg.get("fs")
    if fs_val is not None:
        fs_token = f"fs_{_format_num_for_filename(fs_val)}ms"
    else:
        fs_token = "fs_native"

    f0 = _format_num_for_filename(cfg.get("f0", 25.0))
    group_offset = _format_num_for_filename(cfg.get("group_offset", 1.0))
    shot_offset = _format_num_for_filename(cfg.get("shot_offset", 1.0))

    snr_val = cfg.get("snr")
    if snr_val is None or (isinstance(snr_val, str) and snr_val.strip().lower() in ("none", "null")):
        snr_token = "noiseless"
    else:
        snr_token = f"{_format_num_for_filename(snr_val)}snr"

    dir_name = (
        f"{model}_{gather_clean}_{nx}nx_{nz}nz_{dx}dx_{dz}dz_{n_sources}src_{n_receivers}rec_"
        f"{ms}ms_{fs_token}_{f0}Hz_{group_offset}groupoffset_{shot_offset}shotoffset_{snr_token}"
    )

    if base_dir:
        return Path(base_dir) / dir_name
    return Path(dir_name)


def resolve_simulation_output_dir(cfg: dict, default_base_dir: str = "data") -> Path:
    """
    Resolves the simulation output directory.
    If 'output_dir' is omitted: returns generate_simulation_dir_name(cfg, base_dir=default_base_dir).
    If 'output_dir' is a parent container (e.g. 'data', 'data/', or ending with a slash):
    appends the generated directory name.
    If 'output_dir' specifies a custom distinct leaf folder, returns that folder directly.
    """
    if cfg.get("output_dir"):
        raw_out = str(cfg["output_dir"]).strip()
        out_p = Path(raw_out)
        if raw_out.endswith(("/", "\\")) or out_p.name in ("data", "datasets", "output", "sim_output", "simulations"):
            return out_p / generate_simulation_dir_name(cfg, base_dir=None)
        return out_p
    return generate_simulation_dir_name(cfg, base_dir=default_base_dir)


def setup_matplotlib_style():
    """
    Standardize matplotlib figure formatting across the codebase:
    - scienceplots font style (['science', 'no-latex'])
    - figure size: (10, 5)
    - dpi: 150
    - no grid
    - external ticks only on all four spines
    """
    import matplotlib.pyplot as plt

    try:
        plt.style.use(["science", "no-latex"])
    except (OSError, ValueError):
        pass

    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.grid"] = False
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True

