import asyncio
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset

from agent import Agent, AgentResult
from list_files import list_files
from llm_client import LlmClient
from read_file_contents import read_file_contents
from read_media_file import read_media_file
from reset_workspace import reset_workspace
from tools import search_web
from unzip_file import unzip_file


async def solve_gaia_gzip_file_problems() -> AgentResult:
    dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")

    current_working_directory = Path.cwd()
    sub_folder_for_download = "2023/validation"
    downloads_folder = "downloads"
    dataset_download_folder = current_working_directory / downloads_folder / sub_folder_for_download
    project_root = current_working_directory
    cache_directory = project_root / "gaia_cache"

    reset_workspace(workspace_directory=dataset_download_folder, cache_directory=cache_directory)

    problems = cast(list[dict[str, Any]], list(dataset))
    problems_with_files = [p for p in problems if p.get("file_name")]
    problems_with_zip_files = [
        p for p in problems_with_files if str(p["file_name"]).endswith(".zip")
    ]

    problem = problems_with_zip_files[1]  # we know that there are at least 2.

    file_path = dataset_download_folder / problem["file_name"]

    # Construct prompt including file location
    prompt = f"""{problem['Question']}

    The attached file is located at: {file_path}
    """
    print(prompt)

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

    result = await agent.run(prompt)

    return result


if __name__ == "__main__":
    result = asyncio.run(solve_gaia_gzip_file_problems())
    print(result.model_dump_json(indent=2))
