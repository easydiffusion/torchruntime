import sys
import pytest
from unittest.mock import patch
from torchruntime.installer import get_install_commands, get_pip_commands, run_commands


def test_empty_args():
    packages = []
    result = get_install_commands("cpu", packages)
    assert result == [["torch", "torchaudio", "torchvision"]]


def test_cpu_platform():
    packages = ["torch", "torchvision"]
    result = get_install_commands("cpu", packages)
    assert result == [packages]


def test_cuda_platform():
    packages = ["torch", "torchvision"]
    result = get_install_commands("cu112", packages)
    expected_url = "https://download.pytorch.org/whl/cu112"
    assert result == [packages + ["--index-url", expected_url]]


def test_cuda_nightly_platform():
    packages = ["torch", "torchvision"]
    result = get_install_commands("nightly/cu112", packages)
    expected_url = "https://download.pytorch.org/whl/nightly/cu112"
    assert result == [packages + ["--index-url", expected_url]]


def test_rocm_platform():
    packages = ["torch", "torchvision"]
    result = get_install_commands("rocm4.2", packages)
    expected_url = "https://download.pytorch.org/whl/rocm4.2"
    assert result == [packages + ["--index-url", expected_url]]


def test_xpu_platform_windows_with_torch_only(monkeypatch):
    monkeypatch.setattr("torchruntime.installer.os_name", "Windows")
    packages = ["torch"]
    result = get_install_commands("xpu", packages)
    expected_url = "https://download.pytorch.org/whl/test/xpu"
    assert result == [packages + ["--index-url", expected_url]]


def test_xpu_platform_windows_with_torchvision(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.installer.os_name", "Windows")
    packages = ["torch", "torchvision"]
    result = get_install_commands("xpu", packages)
    expected_url = "https://download.pytorch.org/whl/nightly/xpu"
    assert result == [packages + ["--index-url", expected_url]]
    captured = capsys.readouterr()
    assert "[WARNING]" in captured.out


def test_xpu_platform_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.installer.os_name", "Linux")
    packages = ["torch", "torchvision"]
    result = get_install_commands("xpu", packages)
    expected_url = "https://download.pytorch.org/whl/test/xpu"
    assert result == [packages + ["--index-url", expected_url]]


def test_directml_platform():
    packages = ["torch", "torchvision"]
    result = get_install_commands("directml", packages)
    assert result == [["torch-directml"], packages]


def test_ipex_platform():
    packages = ["torch", "torchvision"]
    result = get_install_commands("ipex", packages)
    assert result == [packages, ["intel-extension-for-pytorch"]]


def test_unsupported_platform():
    packages = ["torch", "torchvision"]
    with pytest.raises(ValueError, match="Unsupported platform: unknown"):
        get_install_commands("unknown", packages)


def test_get_pip_commands_valid():
    cmds = [["package1"], ["package2", "--upgrade"]]
    expected = [
        [sys.executable, "-m", "pip", "install", "package1"],
        [sys.executable, "-m", "pip", "install", "package2", "--upgrade"],
    ]

    result = get_pip_commands(cmds)
    assert result == expected


def test_get_pip_commands_with_uv():
    cmds = [["package1"], ["package2", "--upgrade"]]
    expected = [
        ["uv", "pip", "install", "package1"],
        ["uv", "pip", "install", "package2", "--upgrade"],
    ]

    result = get_pip_commands(cmds, use_uv=True)
    assert result == expected


def test_get_pip_commands_none_input():
    cmds = [["package1"], None]
    with pytest.raises(AssertionError):
        get_pip_commands(cmds)


# Test suite for run_commands
def test_run_commands():
    cmds = [
        [sys.executable, "-m", "pip", "install", "package1"],
        [sys.executable, "-m", "pip", "install", "package2", "--upgrade"],
    ]

    with patch("subprocess.run") as mock_run:
        run_commands(cmds)

        # Ensure subprocess.run was called for each command
        assert mock_run.call_count == len(cmds)

        # Check that subprocess.run was called with the correct arguments
        mock_run.assert_any_call(cmds[0])
        mock_run.assert_any_call(cmds[1])
