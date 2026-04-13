import asyncio
from agent import Agent
from llm_client import LlmClient
from tools import calculator, search_web

async def main():
    agent = Agent(
        model=LlmClient(model="gpt-5-mini"),
        tools=[calculator, search_web],
        instructions="You are a helpful assistant"
    )
    result = await agent.run("What is the result of the multiplication 1234 * 5678?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
