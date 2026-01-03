# API
Import using:
```py
import torchruntime
```

## Install torch packages
You can use the command line:
`python -m torchruntime install <optional list of package names and versions>`

Or you can use the library:
```py
torchruntime.install(["torch", "torchvision<0.20"])
```

Optional flags:
- `preview=True` to allow preview builds (e.g. ROCm 6.4, nightly builds, XPU test index)
- `unsupported=False` to forbid EOL/unsupported builds (errors instead of installing older builds)

On Windows CUDA, Linux ROCm (6.x+), and Linux XPU, this also installs the appropriate Triton package to enable `torch.compile` (`triton-windows`, `pytorch-triton-rocm`, or `pytorch-triton-xpu`).

## Test torch
Run:
`python -m torchruntime test`

To specifically verify `torch.compile` / Triton:
`python -m torchruntime test compile`

## Get device info
You can use the device database built into `torchruntime` for your projects:
```py
from torchruntime.device_db import get_gpus

gpus = get_gpus()  # Returns a list of `torchruntime.device_db.GPU` instances containing the fields: vendor_id, vendor_name, device_id, device_name, is_discrete
```

**Important:** This API could break in the future, so if you're writing a program using this, please open a new Issue on this repo and let me know what you're trying to do.

## Get torch platform (given any GPU)
This will return the recommended torch platform to use for the PC. It will analyze the GPUs and OS on the PC, and suggest the most-performant version of torch for that.

*Note: this is different from the installed torch platfrom (see the next paragraph)!*

E.g. `cu124` or `rocm6.1` or `directml` or `ipex` or `xpu` or `cpu`.

```py
from torchruntime.platform_detection import get_torch_platform

torch_platform = get_torch_platform(gpus)  # use `torchruntime.device_db.get_gpus()` to get a list of recognized GPUs
```

## Get installed torch platform
This will return the installed torch platform, if any. E.g. "cuda", "mps", "cpu" etc.

```py
from torchruntime.utils import get_installed_torch_platform

torch_platform = get_installed_torch_platform()[0]
```

## Get torch device
This will return an instance of `torch.device` for the given device index, from the currently installed torch platform.

```py
from torchruntime.utils import get_device

device1 = get_device(0)
device2 = get_device("cuda:1")
```
