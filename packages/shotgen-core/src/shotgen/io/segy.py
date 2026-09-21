import os
import time
import struct
import numpy as np
import segyio

from shotgen.pylogger import get_logger, log_execution_time

logger = get_logger("shotgen.io.segy")


class SegyIO:
    """
    A class to handle creating, writing, and reading SEGY files for seismic shot records.
    """

    @staticmethod
    def write(filepath, shot_record):
        """
        Write a ShotRecord object to a SEGY file.
        
        Parameters
        ----------
        filepath : str
            The path where the SEGY file will be saved.
        shot_record : ShotRecord
            The shot record instance containing data and geometry.
        """
        if shot_record.shot_run is None:
            raise ValueError("ShotRecord does not contain simulated data. Please run it first.")

        n_sources, n_receivers, n_time = shot_record.shot_run.shape
        total_traces = n_sources * n_receivers
        
        logger.info(f"Writing SEGY file to: {filepath} ({total_traces:,} traces, {n_time:,} samples/trace)")
        dt_s = shot_record.time_vector[1] - shot_record.time_vector[0]
        dt_us = int(np.round(dt_s * 1e6))
        
        spec = segyio.spec()
        spec.sorting = 1
        spec.format = 5
        spec.samples = shot_record.time_vector * 1000
        spec.tracecount = total_traces
        
        cmp_x_all = []
        for src_idx in range(n_sources):
            sx = shot_record.sources[src_idx, 0]
            for rec_idx in range(n_receivers):
                if shot_record.recs.ndim == 3:
                    rx = shot_record.recs[src_idx, rec_idx, 0]
                else:
                    rx = shot_record.recs[rec_idx, 0]
                cmp_x = (sx + rx) / 2
                cmp_x_all.append(cmp_x)
                
        unique_cmp_x = np.unique(np.round(cmp_x_all, 3))
        cmp_x_to_cdp = {val: idx + 1 for idx, val in enumerate(unique_cmp_x)}
        
        with segyio.create(filepath, spec) as f:
            f.bin[segyio.BinField.Interval] = dt_us
            f.bin[segyio.BinField.Samples] = n_time
            
            trace_idx = 0
            
            for src_idx in range(n_sources):
                sx = shot_record.sources[src_idx, 0]
                sy = shot_record.sources[src_idx, 1]
                
                for rec_idx in range(n_receivers):
                    if shot_record.recs.ndim == 3:
                        rx = shot_record.recs[src_idx, rec_idx, 0]
                        ry = shot_record.recs[src_idx, rec_idx, 1]
                    else:
                        rx = shot_record.recs[rec_idx, 0]
                        ry = shot_record.recs[rec_idx, 1]
                        
                    cmp_x = (sx + rx) / 2
                    cdp_number = cmp_x_to_cdp[np.round(cmp_x, 3)]
                    offset = np.sqrt((sx - rx)**2 + (sy - ry)**2)
                    
                    f.header[trace_idx] = {
                        segyio.TraceField.FieldRecord: src_idx + 1,
                        segyio.TraceField.TraceNumber: trace_idx + 1,
                        segyio.TraceField.EnergySourcePoint: src_idx + 1,
                        segyio.TraceField.CDP: cdp_number,
                        segyio.TraceField.CDP_TRACE: 1,
                        segyio.TraceField.SourceGroupScalar: -1000,
                        segyio.TraceField.SourceX: int(sx * 1000),
                        segyio.TraceField.SourceY: int(sy * 1000),
                        segyio.TraceField.GroupX: int(rx * 1000),
                        segyio.TraceField.GroupY: int(ry * 1000),
                        segyio.TraceField.CDP_X: int(cmp_x * 1000),
                        segyio.TraceField.CDP_Y: int(sy * 1000),
                        segyio.TraceField.offset: int(offset * 1000),
                        segyio.TraceField.TRACE_SAMPLE_COUNT: n_time,
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_us,
                    }
                    f.trace[trace_idx] = shot_record.shot_run[src_idx, rec_idx, :].astype(np.float32)
                    trace_idx += 1
        logger.info(f"Successfully written {total_traces:,} traces to {filepath}")

    @staticmethod
    def read(filepath, verbose=True):
        """
        Read a SEGY file and return its geometry and traces.
        Uses ultra-fast memory mapping with seamless fallback to vectorized segyio attributes.
        """
        t0 = time.perf_counter()
        size_mb = os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0.0
        if verbose:
            logger.info(f"Reading seismic traces from: {filepath} ({size_mb:.2f} MB)...")

        try:
            logger.debug(f"Attempting memory-mapped IO for file: {filepath}")
            res = SegyIO._read_mmap(filepath)
            method = "memory-mapped I/O"
        except Exception as e:
            logger.debug(f"Memory mapping failed ({e}); falling back to vectorized segyio.")
            res = SegyIO._read_vectorized_segyio(filepath)
            method = "vectorized segyio"

        n_traces, n_samples = res["data"].shape
        t_elapsed = time.perf_counter() - t0
        msg = f"Loaded {n_traces:,} traces x {n_samples:,} samples ({res['dt']*1000:.2f} ms dt) via {method} in {t_elapsed:.3f} s."
        if verbose:
            logger.info(msg)
        else:
            logger.debug(msg)

        return res

    @staticmethod
    def _read_mmap(filepath):
        file_size = os.path.getsize(filepath)
        if file_size < 3600:
            raise ValueError("File too small to be a valid SEGY file.")

        with open(filepath, "rb") as f:
            f.seek(3200)
            bin_hdr = f.read(400)
            if len(bin_hdr) < 400:
                raise ValueError("Incomplete binary header.")

            dt_us = struct.unpack(">h", bin_hdr[16:18])[0]
            ns = struct.unpack(">h", bin_hdr[20:22])[0]
            format_code = struct.unpack(">h", bin_hdr[24:26])[0]
            endian_prefix = ">"

            if ns <= 0 or ns > 100000 or format_code not in (1, 5):
                dt_us = struct.unpack("<h", bin_hdr[16:18])[0]
                ns = struct.unpack("<h", bin_hdr[20:22])[0]
                format_code = struct.unpack("<h", bin_hdr[24:26])[0]
                endian_prefix = "<"

        if ns <= 0 or format_code not in (1, 5):
            raise ValueError(f"Unsupported or invalid SEGY header: ns={ns}, format={format_code}")

        trace_stride = 240 + ns * 4
        if (file_size - 3600) % trace_stride != 0 or file_size < 3600 + trace_stride:
            raise ValueError("Trace stride mismatch or truncated file.")

        n_traces = (file_size - 3600) // trace_stride
        float_dtype = f"{endian_prefix}f4" if format_code == 5 else f"{endian_prefix}u4"
        trace_dtype = np.dtype([
            ("hdr", np.uint8, 240),
            ("data", float_dtype, ns)
        ])

        mmap_data = np.memmap(filepath, dtype=trace_dtype, mode="r", offset=3600, shape=(n_traces,))
        
        if format_code == 5:
            data = np.ascontiguousarray(mmap_data["data"], dtype=np.float32)
        else:
            raw_u4 = np.ascontiguousarray(mmap_data["data"])
            sign = np.where((raw_u4 >> 31) & 0x01, -1.0, 1.0).astype(np.float32)
            exp = (((raw_u4 >> 24) & 0x7F).astype(np.int32) - 64)
            mant = (raw_u4 & 0x00FFFFFF).astype(np.float32) / 16777216.0
            data = (sign * mant * (16.0 ** exp)).astype(np.float32)

        hdr_bytes = mmap_data["hdr"]
        scalars = np.frombuffer(hdr_bytes[:, 70:72].tobytes(), dtype=f"{endian_prefix}i2")
        sx = np.frombuffer(hdr_bytes[:, 72:76].tobytes(), dtype=f"{endian_prefix}i4")
        sy = np.frombuffer(hdr_bytes[:, 76:80].tobytes(), dtype=f"{endian_prefix}i4")
        rx = np.frombuffer(hdr_bytes[:, 80:84].tobytes(), dtype=f"{endian_prefix}i4")
        ry = np.frombuffer(hdr_bytes[:, 84:88].tobytes(), dtype=f"{endian_prefix}i4")
        cdps = np.frombuffer(hdr_bytes[:, 20:24].tobytes(), dtype=f"{endian_prefix}i4")
        raw_offsets = np.frombuffer(hdr_bytes[:, 36:40].tobytes(), dtype=f"{endian_prefix}i4")

        mults = np.ones(n_traces, dtype=np.float64)
        neg_mask = scalars < 0
        pos_mask = scalars > 0
        mults[neg_mask] = 1.0 / np.abs(scalars[neg_mask])
        mults[pos_mask] = scalars[pos_mask].astype(np.float64)

        sources = np.column_stack((sx * mults, sy * mults))
        receivers = np.column_stack((rx * mults, ry * mults))

        offsets = np.where(
            raw_offsets != 0,
            raw_offsets * mults,
            np.sqrt((sources[:, 0] - receivers[:, 0])**2 + (sources[:, 1] - receivers[:, 1])**2)
        )

        dt = dt_us / 1e6 if dt_us > 0 else 0.001
        time_vector = np.arange(ns, dtype=np.float64) * dt

        return {
            "data": data,
            "sources": sources,
            "receivers": receivers,
            "cdp": cdps,
            "offset": offsets,
            "dt": dt,
            "time": time_vector
        }

    @staticmethod
    def _read_vectorized_segyio(filepath):
        with segyio.open(filepath, "r", ignore_geometry=True, strict=False) as f:
            n_traces = len(f.trace)
            data = f.trace.raw[:]

            scalars = f.attributes(segyio.TraceField.SourceGroupScalar)[:]
            mults = np.ones(n_traces, dtype=np.float64)
            neg_mask = scalars < 0
            pos_mask = scalars > 0
            mults[neg_mask] = 1.0 / np.abs(scalars[neg_mask])
            mults[pos_mask] = scalars[pos_mask].astype(np.float64)

            sx = f.attributes(segyio.TraceField.SourceX)[:]
            sy = f.attributes(segyio.TraceField.SourceY)[:]
            rx = f.attributes(segyio.TraceField.GroupX)[:]
            ry = f.attributes(segyio.TraceField.GroupY)[:]
            cdps = f.attributes(segyio.TraceField.CDP)[:]
            raw_offsets = f.attributes(segyio.TraceField.offset)[:]

            sources = np.column_stack((sx * mults, sy * mults))
            receivers = np.column_stack((rx * mults, ry * mults))

            offsets = np.where(
                raw_offsets != 0,
                raw_offsets * mults,
                np.sqrt((sources[:, 0] - receivers[:, 0])**2 + (sources[:, 1] - receivers[:, 1])**2)
            )

            dt = f.bin[segyio.BinField.Interval] / 1e6
            time_vector = f.samples / 1000.0

        return {
            "data": data,
            "sources": sources,
            "receivers": receivers,
            "cdp": cdps,
            "offset": offsets,
            "dt": dt,
            "time": time_vector
        }

    @staticmethod
    def write_model(filepath, model, dx, dz):
        """
        Write a 2D velocity model array to a SEGY file.
        """
        nx, nz = model.shape
        spec = segyio.spec()
        spec.sorting = 1
        spec.format = 5
        spec.samples = np.arange(nz) * dz
        spec.tracecount = nx
        
        with segyio.create(filepath, spec) as f:
            f.bin[segyio.BinField.Interval] = int(dz * 1000)
            f.bin[segyio.BinField.Samples] = nz
            
            for i in range(nx):
                f.trace[i] = model[i, :].astype(np.float32)
                f.header[i] = {
                    segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                    segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                    segyio.TraceField.CDP: i + 1,
                    segyio.TraceField.CDP_X: int(np.round(i * dx * 1000)),
                    segyio.TraceField.SourceGroupScalar: -1000,
                }

    @staticmethod
    def read_model(filepath, target_dx=None, target_dz=None, verbose=True):
        """
        Read a 2D velocity model array and dimensions from a SEGY file.
        Optionally resamples the model to target_dx and target_dz while preserving physical extent.
        """
        t0 = time.perf_counter()
        if verbose:
            size_mb = os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0.0
            print(f"[SegyIO] Reading velocity model from: {filepath} ({size_mb:.2f} MB)...")

        try:
            data, dx, dz = SegyIO._read_model_mmap(filepath)
            method = "memory-mapped I/O"
        except Exception:
            data, dx, dz = SegyIO._read_model_vectorized(filepath)
            method = "vectorized segyio"

        if target_dx is not None or target_dz is not None:
            from shotgen.models import resample_velocity_model
            data, nx_new, nz_new, dx, dz = resample_velocity_model(data, dx, dz, target_dx, target_dz)

        if verbose:
            t_elapsed = time.perf_counter() - t0
            print(f"[SegyIO] Loaded velocity model grid shape {data.shape[0]}x{data.shape[1]} (dx={dx:.1f}m, dz={dz:.1f}m) via {method} in {t_elapsed:.3f} s.")

        return data, dx, dz


    @staticmethod
    def _read_model_mmap(filepath):
        file_size = os.path.getsize(filepath)
        if file_size < 3600:
            raise ValueError("File too small.")

        with open(filepath, "rb") as f:
            f.seek(3200)
            bin_hdr = f.read(400)
            dz_ms = struct.unpack(">h", bin_hdr[16:18])[0]
            ns = struct.unpack(">h", bin_hdr[20:22])[0]
            fmt = struct.unpack(">h", bin_hdr[24:26])[0]
            endian_prefix = ">"
            if ns <= 0 or ns > 100000 or fmt not in (1, 5):
                dz_ms = struct.unpack("<h", bin_hdr[16:18])[0]
                ns = struct.unpack("<h", bin_hdr[20:22])[0]
                fmt = struct.unpack("<h", bin_hdr[24:26])[0]
                endian_prefix = "<"

        dz = dz_ms / 1000.0 if dz_ms > 0 else 1.0
        trace_stride = 240 + ns * 4
        nx = (file_size - 3600) // trace_stride
        if nx <= 0 or (file_size - 3600) % trace_stride != 0:
            raise ValueError("Invalid trace stride or length.")

        float_dtype = f"{endian_prefix}f4" if fmt == 5 else f"{endian_prefix}u4"
        trace_dtype = np.dtype([
            ("hdr", np.uint8, 240),
            ("data", float_dtype, ns)
        ])

        mmap_data = np.memmap(filepath, dtype=trace_dtype, mode="r", offset=3600, shape=(nx,))
        if fmt == 5:
            data = np.ascontiguousarray(mmap_data["data"], dtype=np.float32)
        else:
            raw_u4 = np.ascontiguousarray(mmap_data["data"])
            sign = np.where((raw_u4 >> 31) & 0x01, -1.0, 1.0).astype(np.float32)
            exp = (((raw_u4 >> 24) & 0x7F).astype(np.int32) - 64)
            mant = (raw_u4 & 0x00FFFFFF).astype(np.float32) / 16777216.0
            data = (sign * mant * (16.0 ** exp)).astype(np.float32)

        hdr_bytes = mmap_data["hdr"]
        cdp_x = np.frombuffer(hdr_bytes[:, 180:184].tobytes(), dtype=f"{endian_prefix}i4")
        if nx > 1 and cdp_x[1] != cdp_x[0]:
            dx = (cdp_x[1] - cdp_x[0]) / 1000.0
        else:
            dx = 1.0

        if dx <= 0.0: dx = 1.0
        if dz <= 0.0: dz = 1.0

        return data, dx, dz

    @staticmethod
    def _read_model_vectorized(filepath):
        with segyio.open(filepath, "r", ignore_geometry=True, strict=False) as f:
            data = f.trace.raw[:]
            dz = f.bin[segyio.BinField.Interval] / 1000.0
            nx = len(f.trace)
            if nx > 1:
                cdp_x = f.attributes(segyio.TraceField.CDP_X)[:]
                dx = (cdp_x[1] - cdp_x[0]) / 1000.0
            else:
                dx = 1.0

            if dx <= 0.0: dx = 1.0
            if dz <= 0.0: dz = 1.0

        return data, dx, dz


