import asyncio
from agent import Agent, AgentResult
from llm_client import LlmClient
from tools import calculator, search_web


async def main() -> None:
    agent = Agent(
        name="search_agent",
        llm_client=LlmClient(
            model="gpt-5-mini"
        ),
        tools=[calculator, search_web],
        instructions=["You are a helpful assistant"],
        max_steps=10,
    )
    result: AgentResult = await agent.run(user_input="Can you tell me how much is 1234 * 5678?")
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
