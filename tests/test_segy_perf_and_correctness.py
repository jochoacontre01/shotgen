import os
import time
import pytest
import numpy as np
import segyio
from shotgen.io import SegyIO
from shotgen.sampleshot import ShotRecord


def test_segy_mmap_and_vectorized_equality(tmp_path):
    """Verifies that mmap and vectorized segyio reading paths return identical results."""
    nx, nz = 30, 20
    dx, dz = 10.0, 10.0
    
    shot_rec = ShotRecord(
        nx=nx, nz=nz, dx=dx, dz=dz,
        n_receivers=6, n_sources=3,
        gather="common shot",
        origin=(0, 0),
        engine="pylops"
    )
    dummy_vel = np.ones((nx, nz)) * 2200.0
    shot_rec.set_model(dummy_vel)
    shot_rec.run(ms=15)
    
    segy_file = os.path.join(tmp_path, "test_traces.segy")
    model_file = os.path.join(tmp_path, "test_model.segy")
    
    SegyIO.write(segy_file, shot_rec)
    SegyIO.write_model(model_file, dummy_vel, dx, dz)
    
    # 1. Test traces reading equality
    data_mmap = SegyIO._read_mmap(segy_file)
    data_vec = SegyIO._read_vectorized_segyio(segy_file)
    data_public = SegyIO.read(segy_file)
    
    assert np.allclose(data_mmap["data"], data_vec["data"])
    assert np.allclose(data_mmap["sources"], data_vec["sources"])
    assert np.allclose(data_mmap["receivers"], data_vec["receivers"])
    assert np.array_equal(data_mmap["cdp"], data_vec["cdp"])
    assert np.allclose(data_mmap["offset"], data_vec["offset"])
    assert np.isclose(data_mmap["dt"], data_vec["dt"])
    assert np.allclose(data_mmap["time"], data_vec["time"])

    assert np.allclose(data_public["data"], data_vec["data"])

    # 2. Test model reading equality
    mod_mmap, dx_m, dz_m = SegyIO._read_model_mmap(model_file)
    mod_vec, dx_v, dz_v = SegyIO._read_model_vectorized(model_file)
    mod_public, dx_p, dz_p = SegyIO.read_model(model_file)

    assert np.allclose(mod_mmap, mod_vec)
    assert np.isclose(dx_m, dx_v)
    assert np.isclose(dz_m, dz_v)

    assert np.allclose(mod_public, mod_vec)
    assert np.isclose(dx_p, dx_v)
    assert np.isclose(dz_p, dz_v)


def test_segy_large_file_perf(tmp_path):
    """Benchmarks reading performance on a dataset with 20,000 traces."""
    n_traces = 20000
    n_samples = 500
    filename = os.path.join(tmp_path, "large_bench.segy")
    
    spec = segyio.spec()
    spec.sorting = 1
    spec.format = 5
    spec.samples = np.arange(n_samples, dtype=np.float32)
    spec.tracecount = n_traces
    
    with segyio.create(filename, spec) as f:
        f.bin[segyio.BinField.Interval] = 1000
        f.bin[segyio.BinField.Samples] = n_samples
        dummy_trace = np.sin(np.linspace(0, 10, n_samples, dtype=np.float32))
        for i in range(n_traces):
            f.trace[i] = dummy_trace
            f.header[i] = {
                segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                segyio.TraceField.SourceGroupScalar: -1000,
                segyio.TraceField.SourceX: 5000 + i * 5,
                segyio.TraceField.SourceY: 1000,
                segyio.TraceField.GroupX: 5010 + i * 5,
                segyio.TraceField.GroupY: 1000,
                segyio.TraceField.CDP: i + 1,
                segyio.TraceField.offset: 10,
            }
            
    t0 = time.perf_counter()
    res = SegyIO.read(filename)
    t_read = time.perf_counter() - t0
    
    assert res["data"].shape == (n_traces, n_samples)
    assert res["sources"].shape == (n_traces, 2)
    assert res["receivers"].shape == (n_traces, 2)
    assert len(res["cdp"]) == n_traces
    assert len(res["offset"]) == n_traces
    
    # 20,000 traces should take under 0.5s with optimized reader
    assert t_read < 1.0, f"Read took too long: {t_read:.3f}s"
