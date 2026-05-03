from pathlib import Path

from tools import tool


@tool
def list_files(path_str: str = ".") -> str:
    """List files and directories in the given path."""

    path = Path(path_str)

    if not path.exists():
        return f"Path not found: {path}"

    if not path.is_dir():
        return f"Not a directory: {path}"

    items: list[str] = []
    for item in sorted(path.iterdir()):
        if item.name.startswith("."):
            continue

        if item.is_dir():
            items.append(f"{item.name}/")
        else:
            items.append(f"{item.name}")

    # Sort directories first
    dirs = [f for f in items if f.endswith("/")]
    files = [f for f in items if not f.endswith("/")]

    result = f"Directory: {path}\n"
    for entry in dirs + files:
        result += f"  {entry}\n"

    return result


# I will call this script to list the files in a directory.
# I want the first argument to be the path to the directory.

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run list_files.py <path>")
        sys.exit(1)

    path: str = sys.argv[1]
    result: str = list_files.func(path_str=path)
    print(result)
