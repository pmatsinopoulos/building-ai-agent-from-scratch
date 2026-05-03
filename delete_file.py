from tools import tool


@tool
def delete_file(file_path: str) -> str:
    """Delete a file. This action cannot be undone."""

    # Only returns message instead of actual deletion (for demo)
    return f"File {file_path} has been deleted."
