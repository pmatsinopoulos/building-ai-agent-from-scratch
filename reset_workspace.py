import shutil
from pathlib import Path


def reset_workspace(workspace_directory: Path, cache_directory: Path) -> None:
    """Restore the workspace to its initial state."""

    shutil.rmtree(path=workspace_directory, ignore_errors=True)
    shutil.copytree(src=cache_directory / "2023/validation", dst=workspace_directory)
    print(f"Workspace reset: {workspace_directory}")
    print(f"Workspace reset: {workspace_directory}")
