from ..consts import POLICIES


def parse_policy_args(args):
    """
    Parses arguments for policy and flags.
    Returns (preview, unsupported, cleaned_args)

    Supports both `--policy NAME` and `--policy=NAME`.

    Logic:
    1. Determine base configuration from the LAST provided --policy argument (or default 'compat').
    2. Apply explicit flags (--preview, --no-unsupported) which ALWAYS override the policy.
    3. Remove policy and flags from args to produce cleaned_args.
    """
    # Default: compat
    preview, unsupported = POLICIES["compat"]
    
    # 1. Scan for the last policy to set the baseline
    last_policy_name = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--policy":
            if i + 1 < len(args):
                last_policy_name = args[i+1]
                i += 2
            else:
                # We will catch this error in the second pass or we can raise it now.
                # Raising now is safer.
                raise ValueError("--policy requires an argument")
        elif arg.startswith("--policy="):
            last_policy_name = arg.split("=", 1)[1]
            if not last_policy_name:
                raise ValueError("--policy requires an argument")
            i += 1
        else:
            i += 1

    if last_policy_name:
        if last_policy_name in POLICIES:
            preview, unsupported = POLICIES[last_policy_name]
        else:
            raise ValueError(f"Unknown policy: {last_policy_name}")

    # 2. Apply flags and build cleaned_args
    cleaned_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--policy":
            # Skip policy and its value (already processed)
            i += 2
            continue
        elif arg.startswith("--policy="):
            i += 1
            continue
        elif arg == "--preview":
            preview = True
            i += 1
        elif arg == "--no-unsupported":
            unsupported = False
            i += 1
        else:
            cleaned_args.append(arg)
            i += 1
            
    return preview, unsupported, cleaned_args
