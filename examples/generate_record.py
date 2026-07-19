import argparse
import ast
import pathlib
from time import perf_counter
import numpy as np
import yaml
import warnings

from shotgen.sampleshot import ShotRecord, load_overthrust, load_bpsalt, load_marmousi
from shotgen.io import generate_video

import scienceplots
import matplotlib.pyplot as plt
plt.style.use(["science", "no-latex"])

def parse_tuple(value):
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in ("none", "null"):
            return None

    if not isinstance(value, str):
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
            return tuple(float(x) for x in value)
        raise argparse.ArgumentTypeError(f"Expected a 2-tuple of numbers or None, got {type(value)}")
    
    s = value.strip()
    # Try ast.literal_eval
    try:
        val = ast.literal_eval(s)
        if val is None:
            return None
        if isinstance(val, (tuple, list)):
            if len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
                return tuple(float(x) for x in val)
    except Exception:
        pass
    
    # Try manual split by comma
    try:
        parts = s.replace('(', '').replace(')', '').replace('[', '').replace(']', '').split(',')
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
    except Exception:
        pass
        
    raise argparse.ArgumentTypeError(
        f"Invalid tuple value: '{value}'. Expected a tuple of two numbers or None, e.g., '(10000,2000)' or 'None'"
    )

# Parameter expected types for validation (excluding nx, nz, dx, dz)
PARAM_TYPES = {
    "meters_per_cell": float,
    "float_type": None,  # Special validation
    "decimate": int,
    "n_sources": int,
    "n_receivers": int,
    "f0": float,
    "src_origin": tuple,
    "rec_origin": tuple,
    "origin": tuple,
    "max_size": tuple,
    "group_offset": float,
    "shot_offset": float,
    "gather": str,
    "smooth": int,
    "snr": float,
    "fd_order": int,
    "n_damping": int,
    "engine": str,
    "ntime": float,
}


def validate_type(name, val):
    if isinstance(val, str):
        val_clean = val.strip().lower()
        if val_clean in ("none", "null"):
            val = None

    if val is None:
        if name in ("snr", "f0", "src_origin", "rec_origin", "origin", "float_type", "max_size"):
            return None
        raise TypeError(f"Parameter '{name}' cannot be None.")
        
    expected_type = PARAM_TYPES.get(name)
    if expected_type is None:
        if name == "float_type":
            if isinstance(val, str):
                val_clean = val.strip().lower()
                if val_clean in ("float32", "np.float32"):
                    return np.float32
                elif val_clean in ("float64", "np.float64"):
                    return np.float64
                else:
                    raise TypeError(f"Invalid value for 'float_type': '{val}'. Expected 'float32' or 'float64'.")
            elif val in (np.float32, np.float64):
                return val
            else:
                raise TypeError(f"Invalid value for 'float_type': '{val}'. Expected np.float32 or np.float64.")
        return val

    if expected_type is int:
        if isinstance(val, bool):
            raise TypeError(f"Parameter '{name}' must be of type int, got boolean {val}.")
        if isinstance(val, (int, float)):
            if int(val) == val:
                return int(val)
        if isinstance(val, str):
            try:
                f_val = float(val)
                if int(f_val) == f_val:
                    return int(f_val)
            except ValueError:
                pass
        raise TypeError(f"Parameter '{name}' must be of type int, got {type(val).__name__} ({val}).")
        
    elif expected_type is float:
        if isinstance(val, bool):
            raise TypeError(f"Parameter '{name}' must be of type float, got boolean {val}.")
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                pass
        raise TypeError(f"Parameter '{name}' must be of type float, got {type(val).__name__} ({val}).")
        
    elif expected_type is tuple:
        try:
            return parse_tuple(val)
        except argparse.ArgumentTypeError as e:
            raise TypeError(f"Parameter '{name}' failed validation: {str(e)}")
            
    elif expected_type is str:
        if isinstance(val, str):
            return val
        raise TypeError(f"Parameter '{name}' must be of type str, got {type(val).__name__} ({val}).")
        
    return val

def main():
    parser = argparse.ArgumentParser(description="Generate shot records with customizable geometry and modeling parameters.")
    
    # Execution / workflow control flags
    parser.add_argument("-c", "--cli", action="store_true", help="Setup runtime for non-gui interface")
    parser.add_argument("-r", "--run", action="store_true", help="Proceed with modeling after showing model geometry.")
    parser.add_argument("-H", "--high-quality", action="store_true", help="Saves and transfers the images to termux to display it in native Android system")
    parser.add_argument("-n", "--no-show", action="store_true", help="Do not display any figure during the simulation.")
    parser.add_argument("-y", "--yes", action="store_true", help="Save simulation files to disk")
    parser.add_argument("-v", "--video", type=int, default=None, help="Save wavefield video with frame decimation factor (save_each).")
    
    # Config file
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML configuration file.")
    
    # Optional simulation parameters (using default=argparse.SUPPRESS)
    parser.add_argument("--n-sources", "--n_sources", dest="n_sources", type=int, default=argparse.SUPPRESS, help="Total number of sources (int)")
    parser.add_argument("--n-receivers", "--n_receivers", dest="n_receivers", type=int, default=argparse.SUPPRESS, help="Total number of receivers (int)")
    parser.add_argument("--f0", type=float, default=argparse.SUPPRESS, help="Central frequency of the wavelet in Hz (float)")
    parser.add_argument("--ntime", dest="ntime", type=float, default=argparse.SUPPRESS, help="Total simulation duration in miliseconds (float)")
    parser.add_argument("--src-origin", "--src_origin", dest="src_origin", type=parse_tuple, default=argparse.SUPPRESS, help="First point of the first source as tuple (e.g. '(0,2)')")
    parser.add_argument("--rec-origin", "--rec_origin", dest="rec_origin", type=parse_tuple, default=argparse.SUPPRESS, help="First point of the first receiver as tuple (e.g. '(0,2)')")
    parser.add_argument("--origin", type=parse_tuple, default=argparse.SUPPRESS, help="Grid physical origin coordinate as tuple (e.g. '(0,0)')")
    parser.add_argument("--max-size", "--max_size", dest="max_size", type=parse_tuple, default=argparse.SUPPRESS, help="Maximum model size in meters as tuple (e.g. '(10000,2000)') or None (default: None)")
    parser.add_argument("--group-offset", "--group_offset", dest="group_offset", type=float, default=argparse.SUPPRESS, help="Offset/spacing between receivers (float)")
    parser.add_argument("--shot-offset", "--shot_offset", dest="shot_offset", type=float, default=argparse.SUPPRESS, help="Offset/spacing between shots (float)")
    parser.add_argument("--gather", type=str, default=argparse.SUPPRESS, help="Type of shot gather, e.g. 'common shot' or 'common midpoint' (str)")
    parser.add_argument("--smooth", type=int, default=argparse.SUPPRESS, help="Gaussian smoothing filter parameter for background velocity (int)")
    parser.add_argument("--snr", type=float, default=argparse.SUPPRESS, help="Signal-to-noise ratio for noise addition (float)")
    parser.add_argument("--fd-order", "--fd_order", dest="fd_order", type=int, default=argparse.SUPPRESS, help="Order of the Finite Differences equation (int)")
    parser.add_argument("--n-damping", "--n_damping", dest="n_damping", type=int, default=argparse.SUPPRESS, help="Number of cells in the damping border (int)")
    parser.add_argument("--engine", type=str, default=argparse.SUPPRESS, help="Born modeling computation engine, e.g., 'pylops' (str)")
    
    # Maintain support for existing other arguments
    parser.add_argument("--decimate", type=int, default=argparse.SUPPRESS, help="Decimation factor for the velocity model (int)")
    parser.add_argument("--meters-per-cell", "--meters_per_cell", dest="meters_per_cell", type=float, default=argparse.SUPPRESS, help="Grid physical meters per cell conversion scale (float)")
    parser.add_argument("--float-type", "--float_type", dest="float_type", default=argparse.SUPPRESS, help="Numeric float type (e.g. 'float32' or 'float64')")


    args = parser.parse_args()

    # 1. Parse YAML config if provided
    yaml_params = {}
    if args.config is not None:
        try:
            with open(args.config, "r") as f:
                content = yaml.safe_load(f)
                if content is not None:
                    if not isinstance(content, dict):
                        raise TypeError("YAML configuration must be a dictionary mapping parameter names to values.")
                    # Normalize hyphenated keys to underscores
                    for k, v in content.items():
                        yaml_params[k.replace("-", "_")] = v
        except Exception as e:
            print(f"Error loading YAML config: {e}")
            raise

    # 2. Extract CLI parameters (only those explicitly passed)
    cli_params = {}
    for key, value in vars(args).items():
        if key in PARAM_TYPES:
            cli_params[key] = value

    # 3. Merge: CLI overrides YAML
    final_params = {}
    for k, v in yaml_params.items():
        if k in PARAM_TYPES:
            final_params[k] = v
    for k, v in cli_params.items():
        final_params[k] = v

    # 4. Validate types
    validated_params = {}
    for k, v in final_params.items():
        try:
            validated_params[k] = validate_type(k, v)
        except TypeError as e:
            raise TypeError(f"Validation error for parameter '{k}': {e}")

    # Load overthrust dataset
    vp, metadata = load_marmousi()
    
    # dx and dz must be processed directly from the metadata (cannot be modified by the user)
    dx = metadata["dx"]
    dz = metadata["dz"]

    # Crop model if max_size is specified (not None) and model is larger than specified limit
    max_size = validated_params.get("max_size", None)
    if max_size is not None:
        # Extract origin in meters (defaults to 0,0)
        origin = validated_params.get("origin", (0.0, 0.0))
        
        # Calculate indices based on origin and max_size limits
        x_start = int(origin[0] / dx)
        x_end = x_start + int(max_size[0] / dx)
        
        z_start = int(origin[1] / dz)
        z_end = z_start + int(max_size[1] / dz)
        
        # Clamp to bounds (if size is smaller, it naturally clamps to max bound, leaving it as is)
        x_start_clamp = max(0, min(x_start, vp.shape[0]))
        x_end_clamp = max(0, min(x_end, vp.shape[0]))
        z_start_clamp = max(0, min(z_start, vp.shape[1]))
        z_end_clamp = max(0, min(z_end, vp.shape[1]))
        
        if x_end_clamp > x_start_clamp and z_end_clamp > z_start_clamp:
            vp = vp[x_start_clamp:x_end_clamp, z_start_clamp:z_end_clamp]

    # Decimate model if decimate is specified (defaults to 1)
    decimate = validated_params.get("decimate", 1)
    if decimate < 1:
        raise ValueError("Decimation factor must be a positive integer.")
    if decimate > 1:
        vp = vp[::decimate, ::decimate]
        dx = dx * decimate
        dz = dz * decimate

    # nx and nz must be inferred directly from the resulting vp array
    nx = vp.shape[0]
    nz = vp.shape[1]


    # Required positional/keyword args for ShotRecord
    sr_args = {
        "nx": nx,
        "nz": nz,
        "dx": dx,
        "dz": dz,
        "n_sources": validated_params.get("n_sources", 10),
        "n_receivers": validated_params.get("n_receivers", 24),
    }

    # Optional parameters to pass ONLY if explicitly provided in final_params
    optional_keys = [
        "f0", "src_origin", "rec_origin", "origin", "group_offset",
        "shot_offset", "gather", "smooth", "snr", "fd_order", "n_damping",
        "engine", "meters_per_cell", "float_type"
    ]
    for key in optional_keys:
        if key in validated_params:
            sr_args[key] = validated_params[key]

    print(f"Instantiating ShotRecord with size: ({sr_args['nx']}, {sr_args['nz']})")
    print(sr_args)
    if getattr(args, "video", None) is not None:
        sr_args["engine"] = "devito"
    shot = ShotRecord(**sr_args)
    shot.set_model(vp)
    
    if not args.no_show:
        shot.show_model(cmap="turbo", cli=args.cli, hq=args.high_quality)

    if args.run:
        start = perf_counter()
        run_kwargs = {}
        if getattr(args, "video", None) is not None:
            run_kwargs["save_wavefield"] = True
            run_kwargs["save_each"] = args.video
        data = shot.run(validated_params.get("ntime", 1000.0), **run_kwargs)
        end = perf_counter()

        print(f"Simulation ended after {end-start:.6f} seconds")
        
        if getattr(args, "video", None) is not None:
            generate_video(shot)
        if not args.no_show:
            shot.show_shot(cmap="grey", cli=args.cli, hq=args.high_quality)
        
        snr_val = f"{shot.snr:.1f}" if shot.snr is not None else "None"
        filename = f"data/{shot.gather.replace(' ', '')}-shot_{shot.nx}nx_{shot.nz}nz_{shot.tn}ms_{shot.dx}dx_{shot.dz}dz_{shot.n_receivers}rec_{shot.n_sources}src_{shot.f0}hz_{shot.group_offset:.0f}goffset_{shot.shot_offset:.0f}soffset_{snr_val}snr"

        if args.yes:
            shot.save_shot(filename)
        else:
            save = input("Save simulation? (y/n): ")
            if save.lower() == "y":
                shot.save_shot(filename)

if __name__ == "__main__":
    main()
