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

_CUDA_12_8_PLATFORM = "cu128"
_CUDA_12_4_PLATFORM = "cu124"
_CUDA_12_8_MIN_VERSIONS = {
    "torch": (2, 7, 0),
    "torchaudio": (2, 7, 0),
    "torchvision": (0, 22, 0),
}


def _parse_version_segments(text):
    text = text.strip().split("+", 1)[0]
    segments = []
    for part in text.split("."):
        m = re.match(r"^(\d+)", part)
        if not m:
            break
        segments.append(int(m.group(1)))
    return segments


def _as_version_tuple(version_segments):
    padded = list(version_segments[:3])
    while len(padded) < 3:
        padded.append(0)
    return tuple(padded)


def _version_lt(a, b):
    return _as_version_tuple(a) < _as_version_tuple(b)


def _version_le(a, b):
    return _as_version_tuple(a) <= _as_version_tuple(b)


def _get_requirement_name_and_specifier(requirement):
    req = requirement.strip()
    if not req or req.startswith("-") or "@" in req:
        return None, None

    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^\]]+\])?", req)
    if not match:
        return None, None

    name = match.group(1).lower().replace("_", "-")
    spec = req[match.end() :].split(";", 1)[0].strip()
    return name, spec


def _upper_bound_for_specifier(specifier):
    """
    Returns (upper_bound_segments, is_inclusive) for specifiers that impose an upper bound,
    or (None, None) if there is no upper bound.
    """

    s = specifier.strip()

    if s.startswith("=="):
        value = s[2:].strip()
        if "*" in value:
            prefix = value.split("*", 1)[0].rstrip(".")
            prefix_segments = _parse_version_segments(prefix)
            if not prefix_segments:
                return None, None
            upper = list(prefix_segments)
            upper[-1] += 1
            upper.append(0)
            return upper, False

        return _parse_version_segments(value), True

    if s.startswith("<="):
        return _parse_version_segments(s[2:].strip()), True

    if s.startswith("<"):
        return _parse_version_segments(s[1:].strip()), False

    if s.startswith("~="):
        value_segments = _parse_version_segments(s[2:].strip())
        if len(value_segments) < 2:
            return None, None
        upper = list(value_segments[:-1])
        upper[-1] += 1
        upper.append(0)
        return upper, False

    return None, None


def _packages_require_cuda_12_4(packages):
    """
    True if the requested torch package versions cannot be satisfied by the CUDA 12.8 wheel index.

    This happens when a package is pinned (or capped) below the first version that has CUDA 12.8 wheels.
    """

    if not packages:
        return False

    for package in packages:
        name, spec = _get_requirement_name_and_specifier(package)
        if not name or name not in _CUDA_12_8_MIN_VERSIONS or not spec:
            continue

        threshold = _CUDA_12_8_MIN_VERSIONS[name]
        for raw in spec.split(","):
            upper, inclusive = _upper_bound_for_specifier(raw)
            if not upper:
                continue

            if inclusive:
                if _version_lt(upper, threshold):
                    return True
            else:
                if _version_le(upper, threshold):
                    return True

    return False


def _adjust_cuda_platform_for_requested_packages(torch_platform, packages):
    if torch_platform == _CUDA_12_8_PLATFORM and _packages_require_cuda_12_4(packages):
        return _CUDA_12_4_PLATFORM
    return torch_platform


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
        - For "xpu" on Windows, if torchvision or torchaudio are included, the function switches to nightly builds.
        - For "directml", the "torch-directml" package is returned as part of the installation commands.
        - For "ipex", the "intel-extension-for-pytorch" package is returned as part of the installation commands.
    """
    if not packages:
        packages = ["torch", "torchaudio", "torchvision"]

    if torch_platform == "cpu":
        return [packages]

    if CUDA_REGEX.match(torch_platform) or ROCM_REGEX.match(torch_platform):
        index_url = f"https://download.pytorch.org/whl/{torch_platform}"
        return [packages + ["--index-url", index_url]]

    if torch_platform == "xpu":
        if os_name == "Windows" and ("torchvision" in packages or "torchaudio" in packages):
            print(
                f"[WARNING] The preview build of 'xpu' on Windows currently only supports torch, not torchvision/torchaudio. "
                f"torchruntime will instead use the nightly build, to get the 'xpu' version of torchaudio and torchvision as well. "
                f"Please contact torchruntime if this is no longer accurate: {CONTACT_LINK}"
            )
            index_url = f"https://download.pytorch.org/whl/nightly/{torch_platform}"
        else:
            index_url = f"https://download.pytorch.org/whl/test/{torch_platform}"

        return [packages + ["--index-url", index_url]]

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
    torch_platform = get_torch_platform(gpu_infos)
    torch_platform = _adjust_cuda_platform_for_requested_packages(torch_platform, packages)
    cmds = get_install_commands(torch_platform, packages)
    cmds = get_pip_commands(cmds, use_uv=use_uv)
    run_commands(cmds)
