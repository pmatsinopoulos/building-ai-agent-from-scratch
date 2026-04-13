import asyncio
import os

from content_types import Message
# from agent import Agent
# from llm_client import LlmClient
# from tools import calculator, search_web

# async def main():
#     agent = Agent(
#         model=LlmClient(model="gpt-5-mini"),
#         tools=[calculator, search_web],
#         instructions="You are a helpful assistant"
#     )
#     result = await agent.run("What is the result of the multiplication 1234 * 5678?")
#     print(result)


# if __name__ == "__main__":
#     asyncio.run(main())

# Create client
from llm_client import LlmClient, LlmRequest


async def main():
    client = LlmClient(model="gpt-5-mini")

    request = LlmRequest(
        instructions=["You are a helpful assistant"],
        contents=[
            Message(
                role="user",
                content="What is the result of 2 + 2?"
            )
        ],
    )

    response = await client.generate(request)

    if response.error_message:
        print(f"Error: {response.error_message}")

    for item in response.content:
        if isinstance(item, Message):
            print(item.content)

if __name__ == "__main__":
    os.environ["LITELLM_LOG"] = "DEBUG"
    asyncio.run(main())
