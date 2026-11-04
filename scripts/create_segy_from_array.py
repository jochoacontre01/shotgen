import shutil
import segyio
import numpy as np

input_file = 'data/simple_sinusoidal_fault_commonshot_2000nx_1000nz_1dx_1dz_25src_96rec_1550ms_fs_6ms_25Hz_5groupoffset_25shotoffset_noiseless/smooth_velocity.segy'
output_file = 'data/simple_sinusoidal_fault_commonshot_2000nx_1000nz_1dx_1dz_25src_96rec_1550ms_fs_6ms_25Hz_5groupoffset_25shotoffset_noiseless/smooth_velocity.segy'

# 1. Copy the original SEG-Y file to preserve metadata and headers
# shutil.copyfile(input_file, output_file)

# 2. Open the new file in read-write mode
with segyio.open(output_file, 'r+', ignore_geometry=True) as dest:
    # Example: multiply all trace values by 2.0
        # Get the length of a single trace
    num_samples = len(dest.samples)
    
    # Create an array of 2000s matching the trace length
    constant_trace = np.full(num_samples, 2000.0, dtype=np.float32)
    
    # Safely assign it to every trace in the file
    for i in range(dest.tracecount):
        dest.trace[i] = constant_trace

