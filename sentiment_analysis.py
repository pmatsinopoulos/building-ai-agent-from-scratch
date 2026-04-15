from typing import Literal
from pydantic import BaseModel

class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: list[str]
