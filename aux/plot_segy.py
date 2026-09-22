import segyio
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="File path")
parser.add_argument("--n_traces", type=int, help="Number of traces to extract")

args = parser.parse_args()

file = args.f if args.f is not None else None
if file is None:
    raise ValueError(
        "The file name should not be empty"
    )

with segyio.open(file, ignore_geometry=True) as f:
    # 1. Extract File Attributes    
    sample_rate_ms = segyio.tools.dt(f) / 1000.0  # Convert microseconds to milliseconds
    num_traces = len(f.trace)
    num_samples = len(f.samples)
    sample_times = f.samples  # Array of sample timestamps 
    
    traces = f.trace.raw[:args.n_traces]

print("Sample rate (ms): ", sample_rate_ms)
print("Total number of traces: ", num_traces)
print("Samples per trace:", num_samples)

extent = (
    0, args.n_traces, num_samples*sample_rate_ms, 0
)
norm = TwoSlopeNorm(0.0, np.amin(traces), np.amax(traces))
plt.imshow(traces.T, cmap="seismic", aspect="auto", extent=extent, norm=norm)
plt.xlabel("Traces")
plt.ylabel("TWT (ms")
plt.colorbar()
plt.show()
