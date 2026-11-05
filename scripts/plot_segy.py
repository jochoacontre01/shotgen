import argparse
import os
import subprocess
import time

import matplotlib.pyplot as plt
import segyio
from shotgen.utils import setup_matplotlib_style

setup_matplotlib_style()

parser = argparse.ArgumentParser()
parser.add_argument("file", help="File path")
parser.add_argument("--n_traces", type=int, help="Number of traces to extract")
parser.add_argument("--cli", action="store_true", help="Shows the image in cli")

args = parser.parse_args()

file = args.file if args.file is not None else None
if file is None:
    raise ValueError(
        "The file name should not be empty"
    )

with segyio.open(file, ignore_geometry=True) as f:
    # 1. Extract File Attributes    
    raw_dt = f.bin[segyio.BinField.Interval]
    if raw_dt == 0 and len(f.header) > 0:
        raw_dt = f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
    dt_us = (raw_dt % 65536) if raw_dt != 0 else segyio.tools.dt(f)
    sample_rate_ms = dt_us / 1000.0  # Convert microseconds to milliseconds
    num_traces = len(f.trace)
    num_samples = len(f.samples)
    sample_times = f.samples  # Array of sample timestamps 
    
    traces = f.trace.raw[:args.n_traces]

    dx = None
    if num_traces > 1:
        scalar = f.header[0][segyio.TraceField.SourceGroupScalar]
        mult = 1.0 / abs(scalar) if scalar < 0 else (scalar if scalar > 0 else 1.0)
        cdp_x = f.attributes(segyio.TraceField.CDP_X)[:]
        if len(cdp_x) > 1 and cdp_x[1] != cdp_x[0]:
            dx = abs(cdp_x[1] - cdp_x[0]) * mult
        else:
            gx = f.attributes(segyio.TraceField.GroupX)[:]
            if len(gx) > 1 and gx[1] != gx[0]:
                dx = abs(gx[1] - gx[0]) * mult

print("Sample rate (ms): ", sample_rate_ms)
print("Total number of traces: ", num_traces)
print("Samples per trace:", num_samples)
print("dx (m): ", dx)
print("\n")

extent = (
    0, traces.shape[0], num_samples*sample_rate_ms, 0
)
# norm = TwoSlopeNorm(0.0, np.amin(traces), np.amax(traces))
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
im = ax.imshow(traces.T, cmap="seismic", aspect="auto", extent=extent)
ax.set_xlabel("Traces")
ax.set_ylabel("TWT (ms)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
if args.cli:
    plt.savefig("segy_img.png", dpi=150)
    plt.close(fig)
    subprocess.run(["chafa", "segy_img.png"], check=False)
    time.sleep(0.5)
    os.remove("segy_img.png")
else:
    plt.show()
