import zipfile
from pathlib import Path

from tools import tool


@tool
def unzip_file(zip_path: str, extract_to: str | None = None) -> str:
    """Extract a zip file to the specified directory."""

    zip_file = Path(zip_path)

    if not zip_file.exists():
        return f"File not found: {zip_file}"

    # Default extraction path: create folder with zip filename
    if extract_to is None:
        extract_path = zip_file.parent / zip_file.stem
    else:
        extract_path = Path(extract_to)

    extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        file_list = zip_ref.namelist()
        zip_ref.extractall(extract_path)

    # Format results
    result = f"Extracted {len(file_list)} files to {extract_path}/\n\n"
    result += "Contents:\n"

    for f in file_list[:20]:
        result += f"  - {f}\n"
    if len(file_list) > 20:
        result += f"  ...and {len(file_list) - 20} more files\n"

    return result


# I will call this script to unzip a file.
# I want the first argument to be the path to the zip file
# and the 2nd argument, if provided, the path to extract to.
# If no 2nd argument, use the filename of the zip file as the extraction path.
# If the zip file is not found, return a message.
# If the extraction is successful, return a message with the number of files extracted and the path.
# If the extraction is not successful, return a message with the error.

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run unzip_file.py <zip_file> [extract_to]")
        sys.exit(1)

    zip_file = sys.argv[1]
    extract_to = sys.argv[2] if len(sys.argv) > 2 else None
    result = unzip_file.func(zip_path=zip_file, extract_to=extract_to)
    print(result)
