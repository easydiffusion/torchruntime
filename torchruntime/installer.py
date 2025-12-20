import re
import sys
import platform
import subprocess

from .consts import CONTACT_LINK
from .device_db import get_gpus
from .gpu_db import get_nvidia_arch
from .platform_detection import get_torch_platform

os_name = platform.system()

PIP_PREFIX = [sys.executable, "-m", "pip", "install"]
CUDA_REGEX = re.compile(r"^(nightly/)?cu\d+$")
ROCM_REGEX = re.compile(r"^(nightly/)?rocm\d+\.\d+$")
REQ_SPEC_REGEX = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(?P<op>==|>=|<=|~=|!=|<|>)\s*(?P<version>[^,;\s]+)"
)
MAJOR_MINOR_REGEX = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)")
TORCH_2_7 = (2, 7)


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


def _parse_major_minor(version: str):
    match = MAJOR_MINOR_REGEX.match(version)
    if not match:
        return None
    return int(match.group("major")), int(match.group("minor"))


def _is_major_minor_gte(left, right):
    return left[0] > right[0] or (left[0] == right[0] and left[1] >= right[1])


def _is_major_minor_lt(left, right):
    return left[0] < right[0] or (left[0] == right[0] and left[1] < right[1])


def _cuda_platform_has_prefix(torch_platform: str):
    return torch_platform.startswith("nightly/")


def _cuda_platform_with_prefix(torch_platform: str, cuda_platform: str):
    if _cuda_platform_has_prefix(torch_platform):
        return f"nightly/{cuda_platform}"
    return cuda_platform


def _get_cuda_platform_for_pytorch_packages(packages):
    """
    Infer a CUDA platform (cu124 vs cu128) from user-specified PyTorch package versions.

    This is needed because PyTorch 2.7.x is published under cu128 wheels, and older
    releases (<=2.6) are published under cu124 wheels. When the requested versions
    are pinned, the installer must select the matching index URL or pip will fail
    with "No matching distribution found".

    Returns:
        "cu124" | "cu128" | None
    """

    if not packages:
        return None

    desired_cuda = None

    for raw_req in packages:
        if not raw_req:
            continue

        req = str(raw_req).strip()
        if not req or req.startswith("-"):
            continue

        match = REQ_SPEC_REGEX.match(req)
        if not match:
            continue

        name = match.group("name").lower()
        op = match.group("op")
        version = match.group("version")

        major_minor = _parse_major_minor(version)
        if not major_minor:
            continue

        # Map torchvision's versioning scheme to the matching torch major/minor.
        if name == "torchvision":
            tv_major, tv_minor = major_minor
            if tv_major != 0:
                continue
            torch_major_minor = (2, max(0, tv_minor - 15))
        elif name in ("torch", "torchaudio"):
            torch_major_minor = major_minor
        else:
            continue

        required_cuda = None
        if op == "==":
            required_cuda = "cu128" if _is_major_minor_gte(torch_major_minor, TORCH_2_7) else "cu124"
        elif op in (">=", ">", "~="):
            if _is_major_minor_gte(torch_major_minor, TORCH_2_7):
                required_cuda = "cu128"
        elif op in ("<", "<="):
            if _is_major_minor_lt(torch_major_minor, TORCH_2_7):
                required_cuda = "cu124"

        if required_cuda is None:
            continue

        if desired_cuda is None:
            desired_cuda = required_cuda
        elif desired_cuda != required_cuda:
            # Conflicting version pins, leave platform unchanged and let pip resolve/fail.
            return None

    return desired_cuda


def _maybe_override_nvidia_cuda_platform(torch_platform, packages, gpu_infos):
    """
    Adjust cu124/cu128 index selection based on pinned torch/torchvision/torchaudio versions.
    """
    if not torch_platform or not CUDA_REGEX.match(torch_platform):
        return torch_platform

    desired_cuda = _get_cuda_platform_for_pytorch_packages(packages)
    if desired_cuda not in ("cu124", "cu128"):
        return torch_platform

    current_cuda = torch_platform.split("/", 1)[-1]
    if current_cuda == desired_cuda:
        return torch_platform

    # Do not demote Blackwell GPUs from cu128 -> cu124; older torch versions won't support them anyway.
    if current_cuda == "cu128" and desired_cuda == "cu124":
        device_names = set(gpu.device_name for gpu in (gpu_infos or []))
        arch_version = get_nvidia_arch(device_names) if device_names else 0
        if arch_version == 12:
            return torch_platform

    return _cuda_platform_with_prefix(torch_platform, desired_cuda)


def install(packages=[], use_uv=False):
    """
    packages: a list of strings with package names (and optionally their versions in pip-format). e.g. ["torch", "torchvision"] or ["torch>=2.0", "torchaudio==0.16.0"]. Defaults to ["torch", "torchvision", "torchaudio"].
    use_uv: bool, whether to use uv for installation. Defaults to False.
    """

    gpu_infos = get_gpus()
    torch_platform = get_torch_platform(gpu_infos)
    torch_platform = _maybe_override_nvidia_cuda_platform(torch_platform, packages, gpu_infos)
    cmds = get_install_commands(torch_platform, packages)
    cmds = get_pip_commands(cmds, use_uv=use_uv)
    run_commands(cmds)
