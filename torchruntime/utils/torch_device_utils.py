import sys
from typing import Union

CPU = "cpu"
CUDA = "cuda"
XPU = "xpu"
MPS = "mps"
DIRECTML = "directml"
MTIA = "mtia"

SUPPORTED_BACKENDS = (CPU, CUDA, XPU, MPS, DIRECTML, MTIA)

if sys.version_info < (3, 10):
    # polyfill for callable static methods
    class CallableStaticMethod(staticmethod):
        def __call__(self, *args, **kwargs):
            return self.__func__(*args, **kwargs)

    # Patch the built-in staticmethod with CallableStaticMethod
    import builtins

    builtins.staticmethod = CallableStaticMethod


def _is_directml_platform_available():
    import torch

    try:
        import torch_directml

        torch.directml = torch_directml

        return torch.directml.is_available()
    except ImportError:
        pass

    return False


def get_installed_torch_platform():
    import torch
    import torch.backends
    from platform import system as os_name

    if torch.cuda.is_available():
        return CUDA, torch.cuda
    if hasattr(torch, XPU) and torch.xpu.is_available():
        return XPU, torch.xpu

    # DirectML is a useful fallback on Windows, but users can have torch-directml installed
    # alongside a working CUDA/ROCm torch build. Prefer the native torch backend when available.
    if _is_directml_platform_available():
        return DIRECTML, torch.directml
    if os_name() == "Darwin":
        if hasattr(torch, MPS):
            return MPS, torch.mps
        if hasattr(torch.backends, MPS) and torch.backends.mps.is_available():
            return MPS, torch.backends.mps
    if hasattr(torch, MTIA) and torch.mtia.is_available():
        return MTIA, torch.mtia

    return CPU, torch.cpu


def get_device_count() -> int:
    torch_platform_name, torch_platform = get_installed_torch_platform()

    if not hasattr(torch_platform, "device_count") or torch_platform_name == "cpu":
        return 1

    return torch_platform.device_count()


def get_device_name(device) -> str:
    "Expects a torch.device as the argument"

    torch_platform_name, torch_platform = get_installed_torch_platform()
    if torch_platform_name not in (XPU, CUDA, DIRECTML):
        return f"{torch_platform_name}:{device.index}"

    if torch_platform_name == DIRECTML:
        return torch_platform.device_name(device.index)

    return torch_platform.get_device_name(device.index)


def get_device(device: Union[int, str]):
    import torch

    if isinstance(device, str):
        if ":" in device:
            torch_platform_name, device_index = device.split(":")
            device_index = int(device_index)
        else:
            torch_platform_name, device_index = device, 0
    else:
        torch_platform_name, _ = get_installed_torch_platform()
        device_index = device

    if torch_platform_name == DIRECTML and _is_directml_platform_available():
        return torch.directml.device(device_index)

    return torch.device(torch_platform_name, device_index)
