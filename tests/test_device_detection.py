import pytest
import os
from devito import configuration
from shotgen import ShotRecord, detect_device, configure_devito_device


def test_detect_device_return_type():
    device = detect_device()
    assert device in ("cpu", "cuda")


def test_configure_devito_device_cpu():
    dev = configure_devito_device("cpu")
    assert dev == "cpu"
    assert os.environ["DEVITO_PLATFORM"] == "intel64"
    assert os.environ["DEVITO_COMPILER"] == "custom"
    assert os.environ["DEVITO_LANGUAGE"] == "C"
    assert str(configuration["platform"]) == "intel64"
    assert configuration["language"] == "C"


def test_configure_devito_device_cuda():
    import shutil
    has_gpu_compiler = bool(shutil.which("nvc++") or shutil.which("nvc"))
    if has_gpu_compiler:
        dev = configure_devito_device("cuda")
        assert dev == "cuda"
        assert os.environ["DEVITO_PLATFORM"] == "nvidiaX"
        assert os.environ["DEVITO_COMPILER"] in ("nvc++", "nvc")
        assert os.environ["DEVITO_LANGUAGE"] == "openacc"
        assert str(configuration["platform"]) == "nvidiaX"
        assert configuration["language"] == "openacc"
    else:
        with pytest.warns(UserWarning, match="Devito GPU compiler"):
            dev = configure_devito_device("cuda")
        assert dev == "cpu"
        assert str(configuration["platform"]) == "intel64"
    
    # Restore CPU config
    configure_devito_device("cpu")


def test_shotrecord_device_auto():
    sr = ShotRecord(
        nx=50, nz=50, dx=10.0, dz=10.0,
        n_sources=2, n_receivers=5,
        device="auto"
    )
    assert sr.device in ("cpu", "cuda")


def test_shotrecord_device_cpu():
    sr = ShotRecord(
        nx=50, nz=50, dx=10.0, dz=10.0,
        n_sources=2, n_receivers=5,
        device="cpu"
    )
    assert sr.device == "cpu"


def test_shotrecord_device_cuda_warning():
    import shutil
    has_gpu_compiler = bool(shutil.which("nvc++") or shutil.which("nvc"))
    if has_gpu_compiler:
        sr = ShotRecord(
            nx=50, nz=50, dx=10.0, dz=10.0,
            n_sources=2, n_receivers=5,
            device="cuda"
        )
        assert sr.device == "cuda"
    else:
        with pytest.warns(UserWarning):
            sr = ShotRecord(
                nx=50, nz=50, dx=10.0, dz=10.0,
                n_sources=2, n_receivers=5,
                device="cuda"
            )
        assert sr.device == "cpu"
    # Restore CPU config
    configure_devito_device("cpu")


def test_shotrecord_invalid_device():
    with pytest.raises(ValueError, match="Invalid device"):
        ShotRecord(
            nx=50, nz=50, dx=10.0, dz=10.0,
            n_sources=2, n_receivers=5,
            device="tpu"
        )
