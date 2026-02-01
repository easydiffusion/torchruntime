CONTACT_LINK = "https://github.com/easydiffusion/torchruntime"

AMD = "1002"
NVIDIA = "10de"
INTEL = "8086"

POLICIES = {
    "stable": (False, False),  # preview=False, unsupported=False
    "compat": (False, True),   # preview=False, unsupported=True (Default)
    "preview": (True, True),   # preview=True, unsupported=True
    "nightly": (True, True),   # alias for preview
}
