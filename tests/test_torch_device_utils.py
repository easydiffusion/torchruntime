import sys
import types


def _make_fake_torch(*, cuda_available: bool):
    torch = types.ModuleType("torch")
    torch.__path__ = []  # allow importing torch.backends

    class _FakeCUDA:
        @staticmethod
        def is_available():
            return cuda_available

    torch.cuda = _FakeCUDA()
    torch.cpu = object()

    backends = types.ModuleType("torch.backends")
    backends.__path__ = []
    torch.backends = backends

    return torch, backends


def test_get_installed_torch_platform_prefers_cuda_over_directml(monkeypatch):
    # Regression test for Windows environments where users have both a working CUDA/ROCm torch
    # backend AND torch-directml installed. In that scenario we should prefer torch.cuda over
    # DirectML to avoid mis-detecting the active backend.
    fake_torch, fake_backends = _make_fake_torch(cuda_available=True)

    fake_torch_directml = types.ModuleType("torch_directml")
    fake_torch_directml.is_available = lambda: True

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.backends", fake_backends)
    monkeypatch.setitem(sys.modules, "torch_directml", fake_torch_directml)

    from torchruntime.utils.torch_device_utils import get_installed_torch_platform

    torch_platform_name, torch_platform = get_installed_torch_platform()
    assert torch_platform_name == "cuda"
    assert torch_platform is fake_torch.cuda


def test_get_installed_torch_platform_uses_directml_when_cuda_unavailable(monkeypatch):
    fake_torch, fake_backends = _make_fake_torch(cuda_available=False)

    fake_torch_directml = types.ModuleType("torch_directml")
    fake_torch_directml.is_available = lambda: True

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.backends", fake_backends)
    monkeypatch.setitem(sys.modules, "torch_directml", fake_torch_directml)

    from torchruntime.utils.torch_device_utils import get_installed_torch_platform

    torch_platform_name, _ = get_installed_torch_platform()
    assert torch_platform_name == "directml"
