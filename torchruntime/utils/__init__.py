from .torch_device_utils import (
    get_installed_torch_platform,
    get_device_count,
    get_device_name,
    get_device,
    SUPPORTED_BACKENDS,
)


def info(preview=False, unsupported=True):
    from torchruntime.device_db import get_gpus
    from torchruntime.platform_detection import get_torch_platform
    from torchruntime.configuration import configure

    print("--- GPUs ---")
    gpu_infos = get_gpus()
    if gpu_infos:
        for i, gpu in enumerate(gpu_infos):
            print(f"{i}. {gpu}")
    else:
        print("No GPUs found!")

    print("")

    print("--- RECOMMENDED TORCH PLATFORM ---")
    torch_platform = get_torch_platform(gpu_infos, preview=preview, unsupported=unsupported)
    print(torch_platform)

    print("")

    print("--- CONFIGURATION ---")
    configure()
