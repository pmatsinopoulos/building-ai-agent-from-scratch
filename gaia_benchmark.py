import asyncio
import os
from typing import Any, cast

from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset

from agent import Agent, AgentResult
from gaia import create_gaia_agent, GaiaOutput
from mcp_utils import load_mcp_tools
from tools import FunctionTool, calculator

tavily_connection = {
    "command": "npx",
    "args": ["-y", "tavily-mcp@latest"],
    "env": {
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
    }
}

SEMAPHORE = asyncio.Semaphore(3)

models = [
    "gpt-5",
]


async def solve_problem(agent: Agent, question: str) -> AgentResult:
    async with SEMAPHORE:
        return await agent.run(question)


async def evaluate_gaia_single(
    problem: dict[str, Any],
    model: str,
    tools: list[FunctionTool]
) -> dict[str, Any]:
    """Evaluate a single GAIA problem with a given model."""
    agent = create_gaia_agent(model=model, tools=tools)
    question = problem["Question"]
    expected_answer = problem.get("Final answer", "")

    result = await solve_problem(agent, question)

    agent_answer = ""
    if isinstance(result.output, GaiaOutput):
        agent_answer = result.output.final_answer
    elif isinstance(result.output, str):
        agent_answer = result.output

    is_correct = agent_answer.strip().lower() == expected_answer.strip().lower()

    return {
        "model": model,
        "question": question,
        "expected_answer": expected_answer,
        "agent_answer": agent_answer,
        "is_correct": is_correct,
    }


async def run_experiment(
    problems: list[dict[str, Any]],
    models: list[str],
    tools: list[FunctionTool]
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate all models on all problems."""
    tasks = [
        evaluate_gaia_single(problem, model, tools)
        for problem in problems
        for model in models
    ]

    all_results: list[dict[str, Any]] = await tqdm_asyncio.gather(*tasks)  # pyright: ignore[reportUnknownMemberType]

    results: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    for result in all_results:
        results[result["model"]].append(result)

    return results


async def main() -> None:
    level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
    print(f"Number of Level 1 problems: {len(level1_problems)}")

    mcp_tools = await load_mcp_tools(tavily_connection)
    tools = [calculator] + mcp_tools

    results = await run_experiment(
        problems=cast(list[dict[str, Any]], level1_problems),
        models=models,
        tools=tools,
    )

    for model, model_results in results.items():
        correct = sum(1 for r in model_results if r["is_correct"])
        total = len(model_results)
        print(f"\n{model}: {correct}/{total} ({100 * correct / total:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
