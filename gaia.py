from pydantic import BaseModel

from agent import Agent
from llm_client import LlmClient
from tools import FunctionTool

class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""

gaia_prompt = """You are a general AI assistant. I will ask you a question.

First, determine if you can solve this problem with your current capabilities and set "is_solvable" accordingly.

If you can solve it, set "is_solvable" to true and provide your answer in "final_answer".

If you cannot solve it, set "is_solvable" to false and explain why in "unsolvable_reason".

Your final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.

If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.

If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.

If you are asked for a comma separated list, apply the above rules depending on whether the element is a number or a string.
"""

def create_gaia_agent(model: str, tools: list[FunctionTool]) -> Agent:
    return Agent(
        name="gaia_agent",
        llm_client=LlmClient(model=model),
        tools=tools,
        instructions=[gaia_prompt],
        output_type=GaiaOutput,
        max_steps=15,
    )
