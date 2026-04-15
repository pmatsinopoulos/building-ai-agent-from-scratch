from typing import Any, Literal, Sequence, Union
from pydantic import BaseModel, Field
import uuid
from datetime import datetime as dt


class Message(BaseModel):
    """A text message in the conversation."""
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str

class ToolCall(BaseModel):
    """LLM's request to call a tool."""
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict[str, Any]

class ToolResult(BaseModel):
    """Result from tool execution."""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    contents: list[Any] # list of Content Items to send back to the LLM

ContentItem = Union[Message, ToolCall, ToolResult]

class Event(BaseModel):
    """A recorded occurrence during agent execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: dt.now().timestamp())
    author: str # "user" or agent name
    contents: Sequence[ContentItem] = Field(default_factory=lambda: list[ContentItem]())
