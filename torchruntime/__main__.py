from .installer import install
from .utils.torch_test import test
from .utils import info


def print_usage(entry_command: str):
    """Print usage information with examples."""
    usage = """
Usage: {entry_command} <command> [arguments]

Commands:
    install             Install PyTorch packages
    test [subcommand]   Run tests (subcommands: all, devices, math, functions)
    --help             Show this help message

Examples:
    {entry_command} install
    {entry_command} install --uv
    {entry_command} install torch==2.2.0 torchvision==0.17.0
    {entry_command} install --uv torch>=2.0.0 torchaudio
    {entry_command} install torch==2.1.* torchvision>=0.16.0 torchaudio==2.1.0

    {entry_command} test          # Runs all tests (import, devices, math, functions)
    {entry_command} test all      # Same as above
    {entry_command} test import  # Test only import
    {entry_command} test devices  # Test only devices
    {entry_command} test math     # Test only math
    {entry_command} test functions # Test only functions

    {entry_command} info          # Prints the list of connected graphics cards, and the recommended torch platform

If no packages are specified, the latest available versions
of torch, torchaudio and torchvision will be installed.

Options:
    --uv               Use uv instead of pip for installation

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
        use_uv = "--uv" in args
        # Remove --uv from args to get package list
        package_versions = [arg for arg in args if arg != "--uv"] if args else None
        install(package_versions, use_uv=use_uv)
    elif command == "test":
        subcommand = sys.argv[2] if len(sys.argv) > 2 else "all"
        test(subcommand)
    elif command == "info":
        info()
    else:
        print(f"Unknown command: {command}")
        entry_path = sys.argv[0]
        cli = "python -m torchruntime" if "__main__.py" in entry_path else "torchruntime"
        print_usage(cli)


if __name__ == "__main__":
    main()
