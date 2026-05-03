from pathlib import Path
from typing import Any, cast
from reset_workspace import reset_workspace
from datasets import load_dataset
from huggingface_hub import snapshot_download

from agent import Agent
from list_files import list_files
from llm_client import LlmClient
from read_file_contents import read_file_contents
from read_media_file import read_media_file
from tools import search_web
from unzip_file import unzip_file

if __name__ == "__main__":
    dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")

    # Download attached files
    current_working_directory = Path.cwd()
    sub_folder_for_download = "2023/validation"
    downloads_folder = "downloads"
    dataset_download_folder = current_working_directory / downloads_folder / sub_folder_for_download
    project_root = current_working_directory
    cache_directory = project_root / "gaia_cache"

    snapshot_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        allow_patterns="2023/validation/*",
        local_dir=cache_directory,
    )

    reset_workspace(workspace_directory=dataset_download_folder, cache_directory=cache_directory)

    # Finding problems with file attachments

    problems = cast(list[dict[str, Any]], list(dataset))
    problems_with_files = [p for p in problems if p.get("file_name")]
    problems_with_zip_files = [
        p for p in problems_with_files if str(p["file_name"]).endswith(".zip")
    ]

    print(f"Total problems: {len(dataset)}")
    print(f"Total problems with files: {len(problems_with_files)}")
    print(f"Total problems with zip files: {len(problems_with_zip_files)}")

    problem = problems_with_zip_files[0]
    print(f"Question: {problem['Question'][:100]}...")
    print(f"File name: {problem['file_name']}")

    file_path: Path = dataset_download_folder / str(problem["file_name"])
    print(f"File exists: {file_path.exists()}")

    tools = [
        search_web,
        unzip_file,
        list_files,
        read_file_contents,
        read_media_file,
    ]

    agent = Agent(
        name="structured_search_agent",
        llm_client=LlmClient(model="gpt-5"),
        tools=tools,
        instructions=[
            "You are a helpful assistant that can search the web and "
            "explore files to answer questions."
        ],
        max_steps=20,
    )
