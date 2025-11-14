# torchruntime
[![Discord Server](https://img.shields.io/discord/1014774730907209781?label=Discord)](https://discord.com/invite/u9yhsFmEkB)

**torchruntime** is a lightweight package for automatically installing the appropriate variant of PyTorch on a user's computer, based on their OS, and GPU manufacturer and GPU model.

This package is used by [Easy Diffusion](https://github.com/easydiffusion/easydiffusion), but you're welcome to use it as well. It's useful for developers who make PyTorch-based apps that target users with NVIDIA, AMD and Intel graphics cards (as well as CPU-only usage), on Windows, Mac and Linux.

* **Platforms:** `cpu`, `cuda`, `rocm`, `xpu`, `directml`, `ipex`
* **Operating systems:** `Windows`, `Linux`, `Mac` (Apple Silicon and Intel)
* **GPU Manufacturers:** `NVIDIA`, `AMD`, `Intel`
* **GPU Types:** Dedicated and Integrated

### Why?
It lets you treat PyTorch as a single dependency (like it should be), and lets you assume that each user will get the most-performant variant of PyTorch suitable for their computer's OS and hardware.

It deals with the complexity of the variety of torch builds and configurations required for CUDA, AMD (ROCm, DirectML), Intel (xpu/DirectML/ipex), and CPU-only.

**Compatibility table**: [Click here](#compatibility-table) to see the supported graphics cards and operating systems.

# Installation
Supports Windows, Linux, and Mac.

`pip install torchruntime`

## Usage
### Step 1. Install the appropriate variant of PyTorch
*This command should be run on the user's computer, or while creating platform-specific builds:*

`python -m torchruntime install`

This will install `torch`, `torchvision`, and `torchaudio`, and will decide the variant based on the user's OS, GPU manufacturer and GPU model number. See [customizing packages](#customizing-packages) for more options.

**Tip:** You can also add the `--uv` flag to install packages using [uv](https://docs.astral.sh/uv/) (instead of `pip`). For e.g. `python -m torchruntime install --uv`

### Step 2. Configure torch
This should be run inside your program, to initialize the required environment variables (if any) for the variant of torch being used.

```py
import torchruntime

torchruntime.configure()
```

### (Optional) Step 3. Test torch
Run `python -m torchruntime test` to run a set of tests to check whether the installed version of torch is working correctly.

## Customizing packages
By default, `python -m torchruntime install` will install the latest available `torch`, `torchvision` and `torchaudio` suitable on the user's platform.

You can customize the packages to install by including their names:
* For e.g. to install only `torch` and `torchvision`, you can run `python -m torchruntime install torch torchvision`
* To install specific versions (in pip format), you can run `python -m torchruntime install "torch>2.0" "torchvision==0.20"`

Supported torch packages: `torch`, `torchvision`, `torchaudio`, `torchao`.

**Note:** If you specify package versions, please keep in mind that the version may not be available to *all* the users on *all* the torch platforms. For e.g. a user with Python 3.8 would not be able to install torch 2.5 (or higher), because torch 2.5 dropped support for Python 3.8.

So in general, it's better to avoid specifying a version unless it really matters to you (or you know what you're doing). Instead, please allow `torchruntime` to pick the latest-possible version for the user.

## Versioning scheme
`torchruntime` uses [semantic versioning](https://semver.org/). Versions will follow the `major.minor.patch` pattern, e.g. `1.20.3`.

- `major` version change: for breaking code changes, e.g. API changes. E.g. `1.0` to `2.0`.
- `minor` version change: for automatic PCI database updates, e.g. support for new graphics cards. E.g. `1.1` to `1.2`.
- `minor` version change: for non-breaking code changes, e.g. backwards-compatible new functionality, routine maintenance, refactoring. E.g. `1.1` to `1.2`.
- `patch` version change: for backward compatible bug fixes. E.g. `1.1.1` to `1.1.2`.

It is recommended that you rely on the minor version, for e.g. use `torchruntime ~= 1` in `requirements.txt` (change this to the current major version), which will install versions like `1.21.0`, `1.22.2` etc but not `2.0.0`.

# Compatibility table
The list of platforms on which `torchruntime` can install a working variant of PyTorch.

**Note:** *This list is based on user feedback (since I don't have all the cards). Please let me know if your card is supported (or not) by opening a pull request or issue or messaging on [Discord](https://discord.com/invite/u9yhsFmEkB) (with supporting logs).*

### CPU-only

| OS  | Supported?| Notes  |
|---|---|---|
| Windows  | ✅ Yes  | x86_64  |
| Linux  | ✅ Yes  | x86_64 and aarch64  |
| Mac (M1/M2/M3/M4)  | ✅ Yes  | arm64. `mps` backend  |
| Mac (Intel)  | ✅ Yes  | x86_64. Stopped after `torch 2.2.2`  |

### NVIDIA

| Series  | Supported? | OS | Notes  |
|---|---|---|---|
| 50xx  | ✅ Yes  | Win/Linux  | Uses CUDA 12.8  |
| 40xx  | ✅ Yes  | Win/Linux  | Uses CUDA 12.8  |
| 30xx  | ✅ Yes  | Win/Linux  | Uses CUDA 12.8  |
| 20xx  | ✅ Yes  | Win/Linux  | Uses CUDA 12.8  |
| 16xx  | ✅ Yes  | Win/Linux  | Uses CUDA 12.8. Requires full-precision for image generation  |
| 10xx  | ✅ Yes  | Win/Linux  | Uses CUDA 12.4  |
| 7xx  | ✅ Yes  | Win/Linux  | Uses CUDA 11.8 |

Datacenter: Supports all 2xx and 1xx series GPUs after Kepler (e.g. H200, B200, H100 etc).

**Note:** Torch dropped support for Python 3.8 from torch >= 2.5. torchruntime falls back to CUDA 12.4, if python 3.8 is being used.

### AMD

#### Discrete GPUs

| Series  | Supported? | OS   | Notes  |
|---|---|---|---|
| 9xxx  | ✅ Yes  | Win/Linux    | Navi4/RDNA4 (gfx120x). ROCm 6.4 on Linux. DirectML on Windows  |
| 7xxx  | ✅ Yes  | Win/Linux    | Navi3/RDNA3 (gfx110x). ROCm 6.2 on Linux. DirectML on Windows  |
| 6xxx  | ✅ Yes  | Win/Linux    | Navi2/RDNA2 (gfx103x). ROCm 6.2 on Linux. DirectML on Windows  |
| 6xxx on Intel Mac  | ✅ Yes  | Intel Mac  | gfx103x. 'mps' backend |
| 5xxx  | ✅ Yes  | Win/Linux    | Navi1/RDNA1 (gfx101x). Full-precision required. DirectML on Windows. Linux only supports upto ROCm 5.2. Waiting for [this](https://github.com/pytorch/pytorch/issues/132570#issuecomment-2313071756) for ROCm 6.2 support.  |
| 5xxx on Intel Mac  | ❓ Untested (WIP)  | Intel Mac  | gfx101x. Implemented but need testers, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |
| Radeon VII  | ✅ Yes  | Win/[Linux](https://discord.com/channels/1014774730907209781/1329021732794667068/1329048324405465108)  | Vega 20 gfx906. Need testers for Windows, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |
| Radeon RX Vega 56  | ✅ Yes  | [Win](https://discord.com/channels/1014774730907209781/1329021732794667068/1331203312137273375)/[Linux](https://discord.com/channels/1014774730907209781/1329021732794667068/1329261488300363776)  | Vega 10 gfx900. ROCm 5.2 on Linux. DirectML on Windows |
| 4xx/5xx/Polaris  | ⚠️ Partial  | Win  | gfx80x. Works with DirectML on Windows ([notes](https://discord.com/channels/1014774730907209781/1329021732794667068/1331199652451713044), [4GB bug](https://github.com/microsoft/DirectML/issues/579#issuecomment-2178963936)). Did not work with ROCm5.7 with custom-compiled PyTorch 1.13 on Linux ([notes](https://discord.com/channels/1014774730907209781/1329021732794667068/1331486882479210602)). |

#### Integrated GPUs (APU)

| Series  | Supported? | OS   | Notes  |
|---|---|---|---|
| Radeon 840M/860M/880M/890M/8040S/8050S/8060S (Strix/Strix Halo/Krackan) | ⚠️ Partial  | Win/Linux    | gfx115x/RDNA3.5. Works with DirectML on Windows. Need testers for Linux, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |
| Radeon 740M/760M/780M (Phoenix/Hawk Point)  | ⚠️ Partial  | Win/Linux    | gfx1103/RDNA3. Works with [DirectML on Windows](https://discord.com/channels/1014774730907209781/1324044688751333472/1332016666346913915). Need testers for Linux, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |
| Radeon 610M/660M/680M (Rembrandt/Raphael/Mendocino/VanGogh/GraniteRidge)  | ❓ Untested (WIP)  | Win/Linux    | gfx103x/RDNA2. Need testers for Windows and Linux, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |
| Ryzen 5000 series ([Cezanne](https://www.cpu-infos.net/amd/amd-cpus-by-core-name/Cezanne/)/[Lucienne](https://www.cpu-infos.net/amd/amd-cpus-by-core-name/Lucienne/)) | ❓ Untested (WIP)  | Win/Linux    | gfx90c/GCN5.1. Need testers for Windows and Linux, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |
| Ryzen 4000 series ([Renoir](https://www.cpu-infos.net/amd/amd-cpus-by-core-name/Renoir/)) | ❓ Untested (WIP)  | Win/Linux    | gfx90c/GCN5.1. Need testers for Windows and Linux, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |
| Ryzen 3000 series ([Picasso](https://www.cpu-infos.net/amd/amd-cpus-by-core-name/Picasso/))  | ❓ Untested (WIP)  | Win/Linux    | gfx903/GCN5. Need testers for Windows and Linux, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |
| Ryzen 2000 series ([Raven Ridge](https://www.cpu-infos.net/amd/amd-cpus-by-core-name/Raven%20Ridge/))  | ❓ Untested (WIP)  | Win/Linux    | gfx902/GCN5. Need testers for Windows and Linux, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |

### Apple

| Series  | Supported? |Notes  |
|---|---|---|
| M1/M2/M3/M4  | ✅ Yes  | 'mps' backend  |
| AMD 6xxx on Intel Mac  | ✅ Yes  | Intel Mac  | gfx103x. 'mps' backend |
| AMD 5xxx on Intel Mac  | ❓ Untested (WIP)  | Intel Mac  | gfx101x. Implemented but need testers, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |

### Intel

| Series  | Supported? | OS | Notes  |
|---|---|---|---|
| Arc  | ❓ Untested (WIP)  | Win/Linux  | Implemented but need testers, please message on [Discord](https://discord.com/invite/u9yhsFmEkB). Backends: 'xpu' or DirectML or [ipex](https://github.com/intel/intel-extension-for-pytorch) |
| Integrated Iris/HD/UHD  | ❓ Untested (WIP)  | Win/Linux  | Implemented but need testers, please message on [Discord](https://discord.com/invite/u9yhsFmEkB). Backends: 'xpu' or DirectML or [ipex](https://github.com/intel/intel-extension-for-pytorch) |

# API
See [API](API.md) for a complete list of module functions.

# FAQ
## Why can't I just run 'pip install torch'?
`pip install torch` installs the CPU-only version of torch, so it won't utilize your GPU's capabilities.

## Why can't I just install torch-for-ROCm directly to support AMD?
Different models of AMD cards require different LLVM targets, and sometimes different ROCm versions. And ROCm currently doesn't work on Windows, so AMD on Windows is best served (currently) with DirectML.

And plenty of AMD cards work with ROCm (even when they aren't in the official list of supported cards). Information about these cards (for e.g. the LLVM target to use) is pretty scattered.

`torchruntime` deals with this complexity for your convenience.

# Contributing
📢 I'm looking for contributions in these specific areas:
- More testing on consumer AMD GPUs.
- More support for older AMD GPUs. Explore: Compile and host PyTorch wheels and rocm (on GitHub) for older AMD gpus (e.g. 580/590/Polaris) with the required patches.
- Intel GPUs.
- Testing on professional AMD GPUs (e.g. the Instinct series).
- An easy-to-run benchmark script (that people can run to check the level of compatibility on their platform).
- Improve [the logic](tests/test_configure.py) for supporting multiple AMD GPUs with different ROCm compatibility. At present, it just picks the latest GPU, which means it doesn't support running workloads on multiple AMD GPUs in parallel.

Please message on the [Discord community](https://discord.com/invite/u9yhsFmEkB) if you have AMD or Intel GPUs, and would like to help with testing or adding support for them! Thanks!

# Credits
* Code contributors on [Easy Diffusion](https://github.com/easydiffusion/easydiffusion).
* Users on [Easy Diffusion's Discord](https://discord.com/invite/u9yhsFmEkB) who've helped with testing on various GPUs.
* [PCI Database](https://raw.githubusercontent.com/pciutils/pciids/refs/heads/master/pci.ids) automatically generated from the PCI ID Database at http://pci-ids.ucw.cz

# More resources
* [AMD Discrete GPU Names](https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units) and [AMD Integrated GPU Names](https://en.wikipedia.org/wiki/AMD_APU)
* [AMD HSA architectures for APUs](https://www.techpowerup.com/forums/threads/amd-graphics-ip.243974/), and also the [official ROCm list](https://github.com/ROCm/clr/blob/4b443f813335e40bf0a2b0686c311a19164ce30f/rocclr/device/pal/paldevice.cpp#L85-L117) (check the latest).
* [AMD GPU LLVM Architectures](https://web.archive.org/web/20241228163540/https://llvm.org/docs/AMDGPUUsage.html#processors)
* [Status of ROCm support for AMD Navi 1](https://github.com/ROCm/ROCm/issues/2527)
* [Torch support for ROCm 6.2 on AMD Navi 1](https://github.com/pytorch/pytorch/issues/132570#issuecomment-2313071756)
* [ROCmLibs-for-gfx1103-AMD780M-APU](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU)
* [Pre-compiled torch for AMD gfx803 (and steps to compile)](https://github.com/tsl0922/pytorch-gfx803)
* [Another guide for compiling torch with rocm 6.2 for gfx803](https://github.com/robertrosenbusch/gfx803_rocm62_pt24)
* [ROCm's device list (in their code)](https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/pal/paldevice.cpp#L85)
