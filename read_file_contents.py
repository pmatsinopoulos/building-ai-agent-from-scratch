from pathlib import Path
from typing import cast

import pandas as pd

from tools import tool

TEXT_EXTENSIONS = [
    ".txt",
    ".py",
    ".json",
    ".md",
    ".html",
    ".css",
    ".xml",
    ".yaml",
    ".yml",
    ".log",
    ".sh",
]
SPREADSHEET_EXTENSIONS = [".xlsx", ".xls", ".csv"]


@tool
def read_file_contents(file_path: str, start_line: int = 1, end_line: int = -1) -> str:
    """Read file contents. Supports txt, py, json, md, csv, xlsx."""

    path = Path(file_path)

    if not path.exists():
        return f"File not found: {file_path}"

    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return _read_text_file(file_path, start_line, end_line)
    elif ext == ".csv":
        return _read_csv_file(file_path)
    elif ext in SPREADSHEET_EXTENSIONS:
        return _read_excel_file(file_path)
    else:
        return _read_text_file(file_path, start_line, end_line)


def _read_text_file(file_path: str, start_line: int = 1, end_line: int = -1) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Adjust line numbers (1-indexed to 0-indexed)
    start_idx = max(0, start_line - 1)
    end_idx = len(lines) if end_line == -1 else min(end_line, len(lines))

    selected_lines = lines[start_idx:end_idx]

    result: list[str] = []
    for i, line in enumerate(selected_lines, start=start_line):
        result.append(f"{i:4d} | {line.rstrip()}")

    return "\n".join(result)


def _read_csv_file(file_path: str) -> str:
    df = pd.read_csv(file_path)  # pyright: ignore[reportUnknownMemberType]
    return cast(  # pyright: ignore[reportUnnecessaryCast]
        str, df.to_markdown(index=False)  # pyright: ignore[reportUnknownMemberType]
    )


def _read_excel_file(file_path: str) -> str:
    df = pd.read_excel(file_path)  # pyright: ignore[reportUnknownMemberType]
    return cast(  # pyright: ignore[reportUnnecessaryCast]
        str, df.to_markdown(index=False)  # pyright: ignore[reportUnknownMemberType]
    )
