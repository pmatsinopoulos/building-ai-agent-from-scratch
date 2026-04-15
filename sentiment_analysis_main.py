import asyncio
from typing import cast
from agent import Agent, AgentResult
from llm_client import LlmClient
from sentiment_analysis import SentimentAnalysis

async def main() -> None:
    agent = Agent(
        name="search_agent",
        llm_client=LlmClient(
            model="gpt-5-mini"
        ),
        tools=[],
        instructions=["You are a helpful assistant that can analyze the sentiment of a given text."],
        max_steps=10,
        output_type=SentimentAnalysis,
    )
    result: AgentResult = await agent.run(user_input="This product exceeded my expectations! Highly recommended!")
    sentiment_analysis: SentimentAnalysis = cast(SentimentAnalysis, result.output)
    print(sentiment_analysis.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
