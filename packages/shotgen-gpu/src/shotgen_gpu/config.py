import os
import sys
import types
import importlib.machinery
import shutil
import ctypes
import subprocess
import warnings

def _disable_torch_import():
    """Prevent PyTorch / libgomp symbol interposition in Devito GPU process memory."""
    if "torch" not in sys.modules:
        class DummyTensor: pass
        mock_torch = types.ModuleType("torch")
        mock_torch.Tensor = DummyTensor
        mock_torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
        sys.modules["torch"] = mock_torch

        dummy_torch_op = types.ModuleType("pylops.torchoperator")
        dummy_torch_op.__all__ = []
        sys.modules["pylops.torchoperator"] = dummy_torch_op

_disable_torch_import()

# Default environment variables for OpenACC GPU execution
os.environ["ACC_DEVICE_TYPE"] = "nvidia"
os.environ["ACC_DEVICE_NUM"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NVCOMPILER_ACC_TIME"] = "0"
os.environ["OMP_TARGET_OFFLOAD"] = "DISABLED"
os.environ["NVCOMPILER_ACC_NOTIFY"] = "0"

try:
    from devito import configuration
    import devito.arch.compiler as dac

    configuration["log-level"] = "WARNING"

    class PureNvidiaCompiler(dac.NvidiaCompiler):
        """
        Subclasses NvidiaCompiler to strip out OpenMP flags (-mp, -fopenmp) and -gpu=pinned,
        returning clean OpenACC flags to avoid triggering system GNU libgomp linkage in WSL2.
        """
        def __init_finalize__(self, **kwargs):
            self.cflags = [
                "-O3",
                "-acc",
                "-gpu=cc89",
                "-fPIC",
                "-shared"
            ]
            self.ldflags = [
                "-shared"
            ]
            self.openmp = []

    dac.compiler_registry['nvc++'] = PureNvidiaCompiler
    dac.compiler_registry['nvc'] = PureNvidiaCompiler
    dac.compiler_registry['nvidia'] = PureNvidiaCompiler
    dac.compiler_registry['nvidiaX'] = PureNvidiaCompiler
    dac.compiler_registry['custom'] = PureNvidiaCompiler
except ImportError:
    pass


def detect_device():
    """
    Detect whether CUDA GPU hardware and a suitable Devito GPU compiler (nvc++/nvc) are available.
    """
    has_gpu_compiler = bool(shutil.which("nvc++") or shutil.which("nvc"))
    if not has_gpu_compiler:
        return "cpu"

    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if res.returncode == 0:
                return "cuda"
        except Exception:
            pass

    if _check_cuda_lib():
        return "cuda"

    return "cpu"


def _check_cuda_lib():
    """Helper to check if libcuda runtime library can be loaded via ctypes."""
    for libname in ["libcuda.so", "libcuda.dylib", "nvcuda.dll", "libcudart.so"]:
        try:
            ctypes.CDLL(libname)
            return True
        except Exception:
            pass
    return False


def configure_devito_device(device="auto", platform=None, compiler=None, language=None, verbose=False):
    """
    Configures Devito environment variables and runtime settings for CUDA GPU or CPU.
    """
    if verbose:
        os.environ["NVCOMPILER_ACC_TIME"] = "1"
    else:
        os.environ["NVCOMPILER_ACC_TIME"] = "0"

    if device == "auto" or device is None:
        device = detect_device()

    device = device.lower()

    if device in ("cuda", "gpu"):
        gpu_compiler = compiler
        if not gpu_compiler:
            if shutil.which("nvc++"):
                gpu_compiler = "nvc++"
            elif shutil.which("nvc"):
                gpu_compiler = "nvc"

        if not gpu_compiler:
            warnings.warn(
                "NVIDIA GPU hardware was detected, but Devito GPU compiler (nvc++/nvc) is not available. "
                "Falling back to CPU execution.",
                category=UserWarning
            )
            return configure_devito_device("cpu", verbose=verbose)

        target_platform = platform if platform else "nvidiaX"
        target_compiler = gpu_compiler
        target_language = language if language else "openacc"

        os.environ["DEVITO_ARCH"] = target_compiler
        os.environ["DEVITO_PLATFORM"] = target_platform
        os.environ["DEVITO_COMPILER"] = target_compiler
        os.environ["DEVITO_LANGUAGE"] = target_language
        os.environ["CC"] = target_compiler
        os.environ["CFLAGS"] = "-O3 -acc -gpu=cc89 -fPIC -shared"

        try:
            from devito import configuration
            configuration["platform"] = target_platform
            configuration["compiler"] = target_compiler
            configuration["language"] = target_language
        except Exception as e:
            warnings.warn(f"Failed to set Devito GPU configuration ({e}). Falling back to CPU.", category=UserWarning)
            return configure_devito_device("cpu", verbose=verbose)
        device = "cuda"
    else:
        target_platform = platform if platform else "intel64"
        target_compiler = compiler if compiler else "custom"
        target_language = language if language else "C"

        os.environ["DEVITO_PLATFORM"] = target_platform
        os.environ["DEVITO_COMPILER"] = target_compiler
        os.environ["DEVITO_LANGUAGE"] = target_language
        os.environ["CFLAGS"] = "-O3 -fPIC -shared"

        try:
            from devito import configuration
            configuration["platform"] = target_platform
            configuration["compiler"] = target_compiler
            configuration["language"] = target_language
        except Exception as e:
            warnings.warn(f"Failed to set Devito CPU configuration: {e}")
        device = "cpu"

    return device
