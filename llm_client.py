import json
from typing import Any, Optional, cast
from pydantic import BaseModel, Field
from litellm import ModelResponse, acompletion
from litellm.types.utils import Usage

from content_types import ContentItem, Message, ToolCall
from tools import FunctionTool


class LlmRequest(BaseModel):
    """Request object for LLM calls."""
    model_config = {"arbitrary_types_allowed": True}

    """Holds system prompt fragments. Rather than a single monolithic prompt,
    we allow multiple instruction strings that get combined. This flexibility
    proves useful when instructions come from different sources: base agent instructions,
    task-specific guidance, or dynamically generated context."""
    instructions: list[str] | None = Field(default_factory=list)

    """Contains the conversation history as ContentItem objects. Messages from
    users and assistants, ToolCalls the LLM requested, and ToolResults from
    executions. This is the core context the LLM uses to understand the current situation."""
    contents: list[ContentItem] = Field(default_factory=list)

    """Lists available tools as BaseTool instances. LlmClient will extract its
    definitions when building the API request."""
    tools: list[FunctionTool] = Field(default_factory=list)

    """Controls how the LLM selects tools. "auto" lets the LLM decide freely.
    "required" forces the tool usage, which becomes important for structured output."""
    tool_choice: Optional[str] = None


class LlmResponse(BaseModel):
    """Response object from LLM calls."""

    """Contains what the LLM produced represented as ContentItem objects. A response
    might include Message with text, one or more ToolCalls requesting tool execution,
    or both."""
    contents: list[ContentItem] = Field(default_factory=list)

    """Contains failures."""
    error_message: Optional[str] = None

    """Tracks token consumption."""
    usage_metadata: dict[str, Any] = Field(default_factory=dict)


class LlmClient:
    """Client for LLM API calls using LiteLLM."""

    def __init__(
        self,
        model: str,
        **config: Any
    ):
        self.model = model
        self.config: dict[str, Any] = config

    async def generate(
        self,
        request: LlmRequest
    ) -> LlmResponse:
        """Generate a response from the LLM."""
        try:
            messages = self._build_messages(request)
            tools = [tool.tool_definition for tool in request.tools] if request.tools else None
            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                **({"tool_choice": request.tool_choice} if request.tool_choice else {}),
                **self.config
            )
            assert isinstance(response, ModelResponse)
            return self._parse_response(response)
        except Exception as e:
            return LlmResponse(
                error_message=str(e)
            )

    def _build_messages(self, request: LlmRequest) -> list[dict[str, Any]]:
        """Convert LlmRequest to API message format."""
        messages: list[dict[str, Any]] = []

        for instruction in request.instructions or []:
            messages.append({
                "role": "system",
                "content": instruction
            })

        for item in request.contents:
            if isinstance(item, Message):
                """Messages become standard message objects."""

                messages.append({
                    "role": item.role,
                    "content": item.content
                })
            elif isinstance(item, ToolCall):
                """ToolCalls get appended to the preceding assistant messages's
                tool_calls array, since our agent stores them consecutively from
                the same LLM response. If a ToolCall appears without a preceding
                assistant message, we create one with null content."""
                tool_call_dict = {
                    "id": item.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": json.dumps(item.arguments)
                    }
                }
                # Append to previous assistant message if exists
                if messages and messages[-1]["role"] == "assistant":
                    messages[-1].setdefault("tool_calls", []).append(tool_call_dict)
                else:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call_dict]
                    })
            else: # if isinstance(item, ToolResult):
                """ToolResults become tool-role messages linked back to their
                originating call via tool_call_id."""
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.tool_call_id,
                    "content": str(item.contents[0]) if item.contents else ""
                })

        return messages

    def _parse_response(self, response: ModelResponse) -> LlmResponse:
        """Convert API response to LlmResponse object."""
        choice = response.choices[0]
        content_items: list[ContentItem] = []

        if choice.message.content:
            content_items.append(
                Message(
                    role="assistant",
                    content=choice.message.content
                )
            )

        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                content_items.append(
                    ToolCall(
                        tool_call_id = tool_call.id,
                        name = tool_call.function.name or "",
                        arguments = json.loads(tool_call.function.arguments)
                    )
                )

        usage = cast(Usage, getattr(response, "usage"))
        return LlmResponse(
            contents = content_items,
            usage_metadata = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            }
        )
