import hashlib
import json
import re
from pathlib import Path

MAX_SIM_DIR_NAME_LEN = 64
MAX_MODEL_SLUG_LEN = 18


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
        if float(val).is_integer():
            return str(int(val))
        s = str(val)
        return s.replace(".", "_")
    s = str(val).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s.replace(".", "_")
    except ValueError:
        return s.replace(".", "_")


def _raw_model_stem(cfg: dict) -> str:
    vel_spec = cfg.get("velocity_model") or cfg.get("vel_file") or cfg.get("vel_model") or cfg.get("model")
    if vel_spec is None:
        return "synthetic"
    if isinstance(vel_spec, (str, Path)):
        p = Path(str(vel_spec).strip())
        name = p.stem.lower()
        clean = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        return clean if clean else "synthetic"
    return "synthetic"


def canonicalize_model_name(raw_name: str | Path | None) -> str:
    """
    Compress a velocity model name or file path into a short, human-readable slug
    (<= MAX_MODEL_SLUG_LEN characters) while preserving dataset identity.
    """
    if raw_name is None:
        return "synth"
    stem = Path(str(raw_name).strip()).stem.lower()
    clean = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if not clean or clean == "synthetic":
        return "synth"

    # Known benchmark & project velocity models
    if re.search(r"ca_?30009", clean):
        if "moho" in clean and "disc" in clean:
            return "ca30009_moho_disc"
        if "moho" in clean:
            return "ca30009_moho"
        return "ca30009"
    if "marmousi" in clean:
        return "marmousi"
    if "bpsalt" in clean or "bp_salt" in clean or "bp2004" in clean or ("vel_z6" in clean and "exact" in clean):
        return "bpsalt"
    if "sigsbee2a" in clean:
        return "sigsbee2a"
    if "sigsbee" in clean:
        return "sigsbee"
    if "overthrust" in clean:
        if "2d" in clean:
            return "overthrust2d"
        if "3d" in clean:
            return "overthrust3d"
        return "overthrust"
    if "r3405" in clean:
        return "r3405"
    if clean in ("fpb", "fbp"):
        return "fpb"
    if "simple_sinusoidal_fault" in clean:
        return "sin_fault"
    if "simple_sinusoidal" in clean:
        return "sinusoidal"

    m = re.match(r"^wavy_inclined_layer_(\d+)$", clean)
    if m:
        return f"wavy_inc{m.group(1)}"
    m = re.match(r"^wavy_layer_(\d+)$", clean)
    if m:
        return f"wavy{m.group(1)}"
    m = re.match(r"^faulted_layer_(\d+)$", clean)
    if m:
        return f"fault{m.group(1)}"
    if clean in ("three_layer", "three_layers"):
        return "3layer"
    if clean in ("two_layer", "two_layers"):
        return "2layer"
    if "complex_graben" in clean:
        return "graben"

    # Strip boilerplate prefixes/suffixes for arbitrary custom models
    slug = re.sub(r"^(vp_|vel_)", "", clean)
    slug = re.sub(r"(_stratigraphy_meters|_stratigraphy|_migvel|_meters|_model|_vel)$", "", slug)
    slug = slug.strip("_") or clean
    if len(slug) > MAX_MODEL_SLUG_LEN:
        slug = slug[:MAX_MODEL_SLUG_LEN].rstrip("_")
    return slug or "synth"


def _extract_model_name(cfg: dict) -> str:
    vel_spec = cfg.get("velocity_model") or cfg.get("vel_file") or cfg.get("vel_model") or cfg.get("model")
    if vel_spec is None or not isinstance(vel_spec, (str, Path)):
        return "synth"
    return canonicalize_model_name(vel_spec)


def _format_gather_token(raw_gather: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "", str(raw_gather).lower())
    if clean in ("commonshot", "cs"):
        return "cs"
    if clean in ("commonmidpoint", "cmp"):
        return "cmp"
    return clean[:6] if clean else "cs"


def _compute_sim_hash(cfg: dict) -> str:
    fs_val = cfg.get("fs_ms")
    if fs_val is None:
        fs_val = cfg.get("fs")
    fs_norm = _format_num_for_filename(fs_val) if fs_val is not None else "native"

    snr_val = cfg.get("snr")
    if snr_val is None or (isinstance(snr_val, str) and snr_val.strip().lower() in ("none", "null")):
        snr_norm = "none"
    else:
        snr_norm = _format_num_for_filename(snr_val)

    payload = {
        "model": _raw_model_stem(cfg),
        "gather": _format_gather_token(cfg.get("gather", "common shot")),
        "nx": _format_num_for_filename(cfg.get("nx", 100)),
        "nz": _format_num_for_filename(cfg.get("nz", 100)),
        "dx": _format_num_for_filename(cfg.get("dx", 10.0)),
        "dz": _format_num_for_filename(cfg.get("dz", 10.0)),
        "n_sources": _format_num_for_filename(cfg.get("n_sources", 2)),
        "n_receivers": _format_num_for_filename(cfg.get("n_receivers", 10)),
        "ms": _format_num_for_filename(cfg.get("ms", cfg.get("ntime", 300.0))),
        "fs_ms": fs_norm,
        "f0": _format_num_for_filename(cfg.get("f0", 25.0)),
        "group_offset": _format_num_for_filename(cfg.get("group_offset", 1.0)),
        "shot_offset": _format_num_for_filename(cfg.get("shot_offset", 1.0)),
        "snr": snr_norm,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def generate_simulation_dir_name(cfg: dict, base_dir: str | None = "data") -> Path:
    """
    Generates a compact, deterministic, unique simulation output directory name
    guaranteed to stay <= MAX_SIM_DIR_NAME_LEN (64) characters so full OS paths
    remain well under the 260-character Windows MAX_PATH limit.

    Format:
        <model>_<gather>_<nsrc>s_<nrec>r_<f0>hz_<hash8>

    Example:
        data/marmousi_cs_2s_10r_25hz_a1b2c3d4
    """
    model = _extract_model_name(cfg)
    gather = _format_gather_token(cfg.get("gather", "common shot"))
    n_sources = _format_num_for_filename(cfg.get("n_sources", 2))
    n_receivers = _format_num_for_filename(cfg.get("n_receivers", 10))
    f0 = _format_num_for_filename(cfg.get("f0", 25.0))
    hash8 = _compute_sim_hash(cfg)

    suffix = f"{gather}_{n_sources}s_{n_receivers}r_{f0}hz_{hash8}"
    max_model_len = max(4, MAX_SIM_DIR_NAME_LEN - len(suffix) - 1)
    if len(model) > max_model_len:
        model = model[:max_model_len].rstrip("_") or "sim"

    dir_name = f"{model}_{suffix}"

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

