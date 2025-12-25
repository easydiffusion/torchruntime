import sys
import platform

from packaging.requirements import Requirement
from packaging.version import Version

from .gpu_db import get_nvidia_arch, get_amd_gfx_info
from .consts import AMD, INTEL, NVIDIA, CONTACT_LINK

os_name = platform.system()
arch = platform.machine().lower()
py_version = sys.version_info

_CUDA_12_8_MIN_VERSIONS = {
    "torch": Version("2.7.0"),
    "torchaudio": Version("2.7.0"),
    "torchvision": Version("0.22.0"),
}


def _packages_require_cuda_12_4(packages):
    if not packages:
        return False

    for package in packages:
        try:
            requirement = Requirement(package)
        except Exception:
            continue

        name = requirement.name.lower().replace("_", "-")
        threshold = _CUDA_12_8_MIN_VERSIONS.get(name)
        if not threshold or not requirement.specifier:
            continue

        test_versions = [
            threshold,
            Version(f"{threshold.major}.{threshold.minor}.{threshold.micro + 1}"),
            Version(f"{threshold.major}.{threshold.minor}.{threshold.micro + 2}"),
            Version(f"{threshold.major}.{threshold.minor}.{threshold.micro + 3}"),
            Version(f"{threshold.major}.{threshold.minor + 1}.0"),
            Version(f"{threshold.major + 1}.0.0"),
        ]

        allows_threshold_or_higher = any(
            requirement.specifier.contains(str(version), prereleases=True) for version in test_versions
        )
        if not allows_threshold_or_higher:
            return True

    return False


def get_torch_platform(gpu_infos, packages=[]):
    """
    Determine the appropriate PyTorch platform to use based on the system architecture, OS, and GPU information.

    Args:
        gpu_infos (list of `torchruntime.device_db.GPU` instances)
        packages (list of str): Optional list of torch/torchvision/torchaudio requirement strings.

    Returns:
        str: A string representing the platform to use. Possible values:
            - "cpu": No discrete GPUs or unsupported configuration.
            - "cuXXX": NVIDIA CUDA version (e.g., "cu124").
            - "rocmXXX": AMD ROCm version (e.g., "rocm6.2").
            - "directml": DirectML for AMD or Intel GPUs on Windows.
            - "ipex": Intel Extension for PyTorch for Linux.
            - "xpu": Intel's backend for PyTorch.

    Raises:
        NotImplementedError: For unsupported architectures, OS-GPU combinations, or multiple GPU vendors.
        Warning: Outputs warnings for deprecated Python versions or fallback configurations.
    """

    VALID_ARCHS = {
        "Windows": {"amd64"},
        "Linux": {"x86_64", "aarch64"},
        "Darwin": {"x86_64", "arm64"},
    }

    if arch not in VALID_ARCHS[os_name]:
        raise NotImplementedError(
            f"torch is not currently available for {os_name} on {arch} architecture! If this is no longer true, please contact torchruntime at {CONTACT_LINK}"
        )

    if len(gpu_infos) == 0:
        return "cpu"

    discrete_devices, integrated_devices = [], []
    for device in gpu_infos:
        if device.is_discrete:
            discrete_devices.append(device)
        else:
            integrated_devices.append(device)

    if discrete_devices:
        return _get_platform_for_discrete(discrete_devices, packages=packages)

    return _get_platform_for_integrated(integrated_devices)


def _get_platform_for_discrete(gpu_infos, packages=None):
    vendor_ids = set(gpu.vendor_id for gpu in gpu_infos)

    if len(vendor_ids) > 1:
        if NVIDIA in vendor_ids:  # temp hack to pick NVIDIA over everything else, pending a better fix
            gpu_infos = [gpu for gpu in gpu_infos if gpu.vendor_id == NVIDIA]
            vendor_ids = set(gpu.vendor_id for gpu in gpu_infos)
        else:
            device_names = list(gpu.vendor_name + " " + gpu.device_name for gpu in gpu_infos)
            raise NotImplementedError(
                f"torchruntime does not currently support multiple graphics card manufacturers on the same computer: {device_names}! Please contact torchruntime at {CONTACT_LINK} with details about your hardware."
            )

    vendor_id = vendor_ids.pop()
    if vendor_id == AMD:
        if os_name == "Windows":
            return "directml"
        elif os_name == "Linux":
            device_names = set(gpu.device_name for gpu in gpu_infos)
            if any(device_name.startswith("Navi 4") for device_name in device_names):
                if py_version < (3, 9):
                    raise NotImplementedError(
                        f"Torch does not support Navi 4x series of GPUs on Python 3.8. Please switch to a newer Python version to use the latest version of torch!"
                    )
                return "rocm6.4"
            if any(device_name.startswith("Navi") for device_name in device_names) and any(
                device_name.startswith("Vega 2") for device_name in device_names
            ):  # lowest-common denominator is rocm5.7, which works with both Navi and Vega 20
                return "rocm5.7"
            if any(
                device_name.startswith("Navi 3") or device_name.startswith("Navi 2") for device_name in device_names
            ):
                if py_version < (3, 9):
                    print(
                        "[WARNING] Support for Python 3.8 was dropped in ROCm 6.2. torchruntime will default to using ROCm 6.1 instead, but consider switching to a newer Python version to use the latest ROCm!"
                    )
                    return "rocm6.1"
                return "rocm6.2"
            if any(device_name.startswith("Vega 2") for device_name in device_names):
                return "rocm5.7"
            if any(device_name.startswith("Navi 1") for device_name in device_names):
                return "rocm5.2"
            if any(device_name.startswith("Vega 1") for device_name in device_names):
                return "rocm5.2"
            if any(device_name.startswith("Ellesmere") for device_name in device_names):
                return "rocm4.2"

            print(
                f"[WARNING] Unsupported AMD graphics card: {device_names}. If this is a recent graphics card (less than 8 years old), please contact torchruntime at {CONTACT_LINK} with details about your hardware."
            )
            return "cpu"
        elif os_name == "Darwin":
            return "mps"
    elif vendor_id == NVIDIA:
        if os_name in ("Windows", "Linux"):
            device_names = set(gpu.device_name for gpu in gpu_infos)
            arch_version = get_nvidia_arch(device_names)
            if py_version < (3, 9) and arch_version == 12:
                raise NotImplementedError(
                    f"Torch does not support NVIDIA 50xx series of GPUs on Python 3.8. Please switch to a newer Python version to use the latest version of torch!"
                )

            # https://github.com/pytorch/pytorch/blob/0b6ea0b959f65d53ea8a34c1fa1c46446dfe3603/.ci/manywheel/build_cuda.sh#L54
            if arch_version == 3.7:
                return "cu118"
            if (arch_version > 3.7 and arch_version < 7.5) or py_version < (3, 9):
                return "cu124"

            if _packages_require_cuda_12_4(packages):
                return "cu124"

            return "cu128"
        elif os_name == "Darwin":
            raise NotImplementedError(
                f"torchruntime does not currently support NVIDIA graphics cards on Macs! Please contact torchruntime at {CONTACT_LINK}"
            )
    elif vendor_id == INTEL:
        if os_name == "Windows":
            if py_version < (3, 9):
                print(
                    "[WARNING] Support for Python 3.8 was dropped in torch 2.5, which supports a higher-performance 'xpu' backend for Intel. torchruntime will default to using 'directml' instead, but consider switching to a newer Python version to use the latest 'xpu' backend for Intel!"
                )
                return "directml"
            return "xpu"
        elif os_name == "Linux":
            if py_version < (3, 9):
                print(
                    "[WARNING] Support for Python 3.8 was dropped in torch 2.5, which supports a higher-performance 'xpu' backend for Intel. torchruntime will default to using 'intel-extension-for-pytorch' instead, but consider switching to a newer Python version to use the latest 'xpu' backend for Intel!"
                )
                return "ipex"
            return "xpu"
        else:
            raise NotImplementedError(
                f"torchruntime does not currently support Intel graphics cards on Macs! Please contact torchruntime at {CONTACT_LINK}"
            )

    print(f"Unrecognized vendor: {gpu_infos}")

    return "cpu"


def _get_platform_for_integrated(gpu_infos):
    gpu = gpu_infos[0]

    if os_name == "Windows":
        return "directml"
    elif os_name == "Linux":
        if gpu.vendor_id == AMD:
            family_name, gfx_id, hsa_version = get_amd_gfx_info(gpu.device_id)
            if gfx_id.startswith("gfx11") or gfx_id.startswith("gfx103"):
                if py_version < (3, 9):
                    print(
                        "[WARNING] Support for Python 3.8 was dropped in ROCm 6.2. torchruntime will default to using ROCm 6.1 instead, but consider switching to a newer Python version to use the latest ROCm!"
                    )
                    return "rocm6.1"
                return "rocm6.2"
            elif gfx_id.startswith("gfx102") or gfx_id.startswith("gfx101") or gfx_id.startswith("gfx90"):
                return "rocm5.5"
            elif gfx_id.startswith("gfx8"):
                return "rocm4.2"

            print(f"[WARNING] Unsupported AMD APU: {gpu}! Please contact torchruntime at {CONTACT_LINK}")
        elif gpu.vendor_id == INTEL:
            if py_version < (3, 9):
                print(
                    "[WARNING] Support for Python 3.8 was dropped in torch 2.5, which supports a higher-performance 'xpu' backend for Intel. torchruntime will default to using 'intel-extension-for-pytorch' instead, but consider switching to a newer Python version to use the latest 'xpu' backend for Intel!"
                )
                return "ipex"
            return "xpu"
    elif os_name == "Darwin":
        print(
            f"[WARNING] torchruntime does not currently support integrated graphics cards on Macs! Please contact torchruntime at {CONTACT_LINK}"
        )

    return "cpu"
