import importlib.util
import platform
import time

from ..torch_device_utils import get_device, get_device_count, get_device_name, get_installed_torch_platform


def test(subcommand):
    from ...configuration import configure

    configure()

    test_fn = globals().get(f"test_{subcommand}")
    if not test_fn or not callable(test_fn):
        raise RuntimeError(f"Unknown test sub-command: {subcommand}")

    test_fn()


def test_all():
    for fn in (test_import, test_devices, test_compile, test_math, test_functions):
        fn()
        print("")


def test_import():
    print("--- IMPORT TEST ---")
    import torch

    print(f"Torch version: {torch.__version__}")

    print("--- / IMPORT TEST ---")


def test_devices():
    print("--- DEVICE TEST ---")

    print("Installed torch platform:", get_installed_torch_platform()[0])
    print("Device count:", get_device_count())
    for i in range(get_device_count()):
        device = get_device(i)
        print(f"Torch device ({i}):", device)
        device_name = get_device_name(device)
        print(f"Device name ({i}):", device_name)

    print("--- / DEVICE TEST ---")


def test_math():
    print("--- MATH TEST ---")

    import torch

    def simple_sum(device):
        print("  ", "Simple math:")
        x = torch.tensor([0, 1, 2], device=device)
        print("    ", "x:", x)
        x_new = x + 10
        print("    ", "x + 10:", x_new)
        expected_x = torch.tensor([10, 11, 12], device=device)

        try:
            assert torch.equal(x_new, expected_x), f"{x_new} != {expected_x}"
        except Exception as e:
            print("    ", f"Simple sum: FAILED ({e})")

    def norm(device):
        print("  ", "Norm:")
        N_ITERS = 10
        x = torch.randn((10, 2048, 2048, 3), device=device)
        print("    ", f"Size of x: {x.numel() * x.element_size() / 1024**2} Mb", "on", x.device)

        x.norm()
        t = time.time()
        for i in range(N_ITERS):
            y = x.norm()
        print("    ", f"Norm ({y}), took {1000 * (time.time() - t) / N_ITERS:0.1f} ms")

    def run(device):
        print("On torch device:", device)
        simple_sum(device)
        norm(device)

    device = get_device("cpu")
    run(device)

    for i in range(get_device_count()):
        device = get_device(i)
        run(device)

    print("--- / MATH TEST ---")


def test_functions():
    print("--- FUNCTIONAL TEST ---")

    from .torch_regression_tests import PyTorchRegressionTest

    for i in range(get_device_count()):
        device = get_device(i)
        print("On torch device:", device)
        t = PyTorchRegressionTest(device, prefix="  ")
        t.run_all_tests()

    print("--- / FUNCTIONAL TEST ---")


def test_compile():
    print("--- COMPILE TEST ---")

    try:
        import torch
    except ImportError:
        print("torch.compile: SKIPPED (torch not installed)")
        print("--- / COMPILE TEST ---")
        return

    if not hasattr(torch, "compile"):
        print("torch.compile: SKIPPED (requires torch>=2.0)")
        print("--- / COMPILE TEST ---")
        return

    torch_platform_name, _ = get_installed_torch_platform()
    if torch_platform_name not in ("cuda", "xpu"):
        print(f"torch.compile: SKIPPED (unsupported backend: {torch_platform_name})")
        print("--- / COMPILE TEST ---")
        return

    if importlib.util.find_spec("triton") is None:
        print("triton: NOT INSTALLED")
    else:
        print("triton: installed")

    device = get_device(0)
    print("On torch device:", device)

    def f(x):
        return x * 2 + 1

    try:
        compiled_f = torch.compile(f)
        x = torch.randn((1024,), device=device)
        y = compiled_f(x)
        expected = f(x)
        if not torch.allclose(y, expected):
            print("torch.compile: FAILED (output mismatch)")
        else:
            if torch_platform_name == "cuda":
                torch.cuda.synchronize()
            if torch_platform_name == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
                torch.xpu.synchronize()
            print("torch.compile: PASSED")
    except Exception as e:
        print(f"torch.compile: FAILED ({type(e).__name__}: {e})")

        hint = None
        os_name = platform.system()
        if torch_platform_name == "cuda" and os_name == "Windows":
            hint = "pip install triton-windows  (or: python -m torchruntime install)"
        elif torch_platform_name == "cuda" and os_name == "Linux":
            if getattr(torch.version, "hip", None):
                hint = (
                    "pip install pytorch-triton-rocm --index-url https://download.pytorch.org/whl  "
                    "(or: python -m torchruntime install)"
                )
        elif torch_platform_name == "xpu" and os_name == "Linux":
            hint = (
                "pip install pytorch-triton-xpu --index-url https://download.pytorch.org/whl  "
                "(or: python -m torchruntime install)"
            )

        if hint:
            print("If this failed due to Triton, try:")
            print("  ", hint)

    print("--- / COMPILE TEST ---")
