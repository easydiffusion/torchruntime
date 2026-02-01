import pytest
from torchruntime.device_db import GPU
from torchruntime.platform_detection import AMD, INTEL, get_torch_platform, py_version


def test_preview_rocm_6_4_selection(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "Navi 41", True)]

    if py_version < (3, 9):
        pytest.skip("Navi 4 requires Python 3.9+")

    # Default: preview=False -> cpu
    assert get_torch_platform(gpu_infos) == "cpu"
    assert get_torch_platform(gpu_infos, preview=False) == "cpu"

    # preview=True -> rocm6.4
    assert get_torch_platform(gpu_infos, preview=True) == "rocm6.4"


def test_eol_rocm_5_2_selection(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "Navi 10", True)]

    assert get_torch_platform(gpu_infos) == "rocm5.2"

    with pytest.raises(ValueError, match="considered End-of-Life"):
        get_torch_platform(gpu_infos, unsupported=False)


def test_eol_rocm42_selection(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    # Ellesmere (e.g. RX 580)
    gpu_infos = [GPU(AMD, "AMD", "67df", "Ellesmere [Radeon RX 470/480/570/570X/580/580X/590]", True)]

    # Default: unsupported=True -> rocm4.2
    assert get_torch_platform(gpu_infos) == "rocm4.2"

    # unsupported=False -> raises ValueError
    with pytest.raises(ValueError, match="considered End-of-Life"):
        get_torch_platform(gpu_infos, unsupported=False)


def test_eol_directml_selection(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "Radeon", True)]

    assert get_torch_platform(gpu_infos) == "directml"

    with pytest.raises(ValueError, match="considered End-of-Life"):
        get_torch_platform(gpu_infos, unsupported=False)


def test_eol_ipex_selection(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    monkeypatch.setattr("torchruntime.platform_detection.py_version", (3, 8))
    gpu_infos = [GPU(INTEL, "Intel", 0x1234, "Iris", True)]

    assert get_torch_platform(gpu_infos) == "ipex"

    # unsupported=False -> raises ValueError
    with pytest.raises(ValueError, match="considered End-of-Life"):
        get_torch_platform(gpu_infos, unsupported=False)
