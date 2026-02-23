import re
import sys
import platform
import subprocess

from .consts import CONTACT_LINK
from .device_db import get_gpus
from .platform_detection import get_torch_platform

os_name = platform.system()

PIP_PREFIX = [sys.executable, "-m", "pip", "install"]
CUDA_REGEX = re.compile(r"^(nightly/)?cu\d+$")
ROCM_REGEX = re.compile(r"^(nightly/)?rocm\d+\.\d+$")
ROCM_VERSION_REGEX = re.compile(r"^(?:nightly/)?rocm(?P<major>\d+)\.(?P<minor>\d+)$")


def get_install_commands(torch_platform, packages):
    """
    Generates pip installation commands for PyTorch and related packages based on the specified platform.

    Args:
        torch_platform (str): Target platform for PyTorch. Must be one of:
            - "cpu"
            - "cuXXX" (e.g., "cu112", "cu126")
            - "rocmXXX" (e.g., "rocm4.2", "rocm6.2")
            - "xpu"
            - "directml"
            - "ipex"
        packages (list of str): List of package names (and optionally versions in pip format). Examples:
            - ["torch", "torchvision"]
            - ["torch>=2.0", "torchaudio==0.16.0"]

    Returns:
        list of list of str: Each sublist contains a pip install command (excluding the `pip install` prefix).
            Examples:
            - [["torch", "--index-url", "https://foo.com/whl"]]
            - [["torch-directml"], ["torch", "torchvision"]]

    Raises:
        ValueError: If an unsupported platform is provided.

    Notes:
        - For "directml", the "torch-directml" package is returned as part of the installation commands.
        - For "ipex", the "intel-extension-for-pytorch" package is returned as part of the installation commands.
        - For Windows CUDA, the function also installs "triton-windows" (for torch.compile and Triton kernels).
        - For Linux ROCm 6.x, the function also installs "pytorch-triton-rocm".
        - For Linux XPU, the function also installs "pytorch-triton-xpu".
    """
    if not packages:
        packages = ["torch", "torchaudio", "torchvision"]

    if torch_platform == "cpu":
        return [packages]

    if CUDA_REGEX.match(torch_platform) or ROCM_REGEX.match(torch_platform):
        index_url = f"https://download.pytorch.org/whl/{torch_platform}"
        cmds = [packages + ["--index-url", index_url]]

        if os_name == "Windows" and CUDA_REGEX.match(torch_platform):
            cmds.append(["triton-windows"])

        if os_name == "Linux" and ROCM_REGEX.match(torch_platform):
            match = ROCM_VERSION_REGEX.match(torch_platform)
            if match and int(match.group("major")) >= 6:
                cmds.append(["pytorch-triton-rocm", "--index-url", "https://download.pytorch.org/whl"])

        return cmds

    if torch_platform == "xpu":
        index_url = f"https://download.pytorch.org/whl/{torch_platform}"

        cmds = [packages + ["--index-url", index_url]]
        if os_name == "Linux":
            cmds.append(["pytorch-triton-xpu", "--index-url", "https://download.pytorch.org/whl"])
        return cmds

    if torch_platform == "directml":
        return [["torch-directml"], packages]

    if torch_platform == "ipex":
        return [packages, ["intel-extension-for-pytorch"]]

    raise ValueError(f"Unsupported platform: {torch_platform}")


def get_pip_commands(cmds, use_uv=False):
    assert not any(cmd is None for cmd in cmds)
    if use_uv:
        pip_prefix = ["uv", "pip", "install"]
    else:
        pip_prefix = [sys.executable, "-m", "pip", "install"]
    return [pip_prefix + cmd for cmd in cmds]


def run_commands(cmds):
    for cmd in cmds:
        print("> ", cmd)
        subprocess.run(cmd)


def install(packages=[], use_uv=False):
    """
    packages: a list of strings with package names (and optionally their versions in pip-format). e.g. ["torch", "torchvision"] or ["torch>=2.0", "torchaudio==0.16.0"]. Defaults to ["torch", "torchvision", "torchaudio"].
    use_uv: bool, whether to use uv for installation. Defaults to False.
    """

    gpu_infos = get_gpus()
    torch_platform = get_torch_platform(gpu_infos, packages=packages)
    cmds = get_install_commands(torch_platform, packages)
    cmds = get_pip_commands(cmds, use_uv=use_uv)
    run_commands(cmds)
