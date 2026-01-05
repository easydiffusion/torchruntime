from .installer import install
from .utils.torch_test import test
from .utils import info
from .utils.args import parse_policy_args


def print_usage(entry_command: str):
    """Print usage information with examples."""
    usage = """
Usage: {entry_command} <command> [arguments]

Commands:
    install             Install PyTorch packages
    test [subcommand]   Run tests (subcommands: all, import, devices, compile, math, functions)
    --help             Show this help message

Examples:
    {entry_command} install
    {entry_command} install --uv
    {entry_command} install --preview
    {entry_command} install --no-unsupported
    {entry_command} install --policy stable
    {entry_command} install torch==2.2.0 torchvision==0.17.0
    {entry_command} install --uv torch>=2.0.0 torchaudio
    {entry_command} install torch==2.1.* torchvision>=0.16.0 torchaudio==2.1.0

    {entry_command} test          # Runs all tests (import, devices, compile, math, functions)
    {entry_command} test all      # Same as above
    {entry_command} test import  # Test only import
    {entry_command} test devices  # Test only devices
    {entry_command} test compile  # Test torch.compile (Triton)
    {entry_command} test math     # Test only math
    {entry_command} test functions # Test only functions

    {entry_command} info          # Prints the list of connected graphics cards, and the recommended torch platform

If no packages are specified, the latest available versions
of torch, torchaudio and torchvision will be installed.

Options:
    --uv               Use uv instead of pip for installation
    --preview          Allow preview builds (e.g. ROCm 6.4)
    --no-unsupported   Forbid EOL/unsupported builds (e.g. DirectML / IPEX / Torch 1.x)
    --policy <name>    Set configuration policy (stable, compat, preview|nightly). Default: compat

Version specification formats (follows pip format):
    package==2.1.0     Exact version
    package>=2.0.0     Minimum version
    package<=2.2.0     Maximum version
    package~=2.1.0     Compatible release
    package==2.1.*     Any 2.1.x version
    package            Latest version
    """.format(
        entry_command=entry_command
    )
    print(usage.strip())


def main():
    import sys

    if len(sys.argv) < 2 or sys.argv[1] in ["--help", "-h"]:
        entry_path = sys.argv[0]
        cli = "python -m torchruntime" if "__main__.py" in entry_path else "torchruntime"
        print_usage(cli)
        return

    command = sys.argv[1]

    if command == "install":
        args = sys.argv[2:] if len(sys.argv) > 2 else []
        try:
            preview, unsupported, cleaned_args = parse_policy_args(args)
        except ValueError as e:
            print(f"Error: {e}")
            return

        use_uv = "--uv" in cleaned_args
        # Remove --uv from package list
        package_versions = [arg for arg in cleaned_args if arg != "--uv"]
        install(package_versions, use_uv=use_uv, preview=preview, unsupported=unsupported)
    elif command == "test":
        subcommand = sys.argv[2] if len(sys.argv) > 2 else "all"
        test(subcommand)
    elif command == "info":
        args = sys.argv[2:] if len(sys.argv) > 2 else []
        try:
            preview, unsupported, _ = parse_policy_args(args)
        except ValueError as e:
            print(f"Error: {e}")
            return
        from .utils import info
        info(preview=preview, unsupported=unsupported)
    else:
        print(f"Unknown command: {command}")
        entry_path = sys.argv[0]
        cli = "python -m torchruntime" if "__main__.py" in entry_path else "torchruntime"
        print_usage(cli)


if __name__ == "__main__":
    main()
