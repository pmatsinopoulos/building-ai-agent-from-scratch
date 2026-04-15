from content_types import Event, Message, ToolCall, ToolResult
from llm_client import LlmClient, LlmRequest, LlmResponse
from tools import FunctionTool
from pydantic import BaseModel, Field
from execution_context import ExecutionContext


class AgentResult(BaseModel):
    """Result of an agent execution."""
    output: str | BaseModel | None = Field(default=None)# from Pydantic
    context: ExecutionContext = Field(default_factory=ExecutionContext)


class Agent:
    def __init__(
        self,
        name: str,
        llm_client: LlmClient, # The LlmClient instance that handles LLM communication. Example: client = LlmClient(model="gpt-5-mini").
        tools: list[FunctionTool] | None= None, # The tools that the Agent can use
        instructions: list[str] | None=None, # System prompt that defines the agent's behaviour.
        max_steps: int =10, # Safety limit to prevent infinite loops.
    ):
        self.name = name
        self.llm_client = llm_client
        self.instructions = instructions
        self.max_steps = max_steps
        self.tools = self._setup_tools(tools or [])

    def _setup_tools(self, tools: list[FunctionTool]) -> list[FunctionTool]:
        return tools

    async def run(
        self,
        user_input: str,
        context: ExecutionContext | None=None,
    ) -> AgentResult:
        # Create or reuse context
        if context is None:
            context = ExecutionContext()

        # Everything is added as Event in order to help troubleshooting
        event = Event(
            execution_id=context.execution_id, # we keep all events under the same execution id
            author="user",
            contents=[
                Message(
                    role="user",
                    content=user_input,
                )
            ]
        )
        context.add_event(event) # all the events are added to the context

        while not context.final_result and context.current_step < self.max_steps:
            await self._step(context)

            last_event = context.events[-1]
            # last_event will tell us
            # - whether it is a final response (completion detection)
            # - if it is, then we can extract the final result from this last event (result extraction)
            if self._is_final_response(last_event):
                context.final_result = self._extract_final_result(last_event)

        return AgentResult(
            output=context.final_result,
            context=context,
        )

    def _is_final_response(self, event: Event) -> bool:
        """Check if the event contains a final response.
        In order for an Event to be a final response it shouldn't contain
        ToolCalls neither ToolResults."""
        has_tool_calls = any(isinstance(content_item, ToolCall) for content_item in event.contents)
        has_tool_results = any(isinstance(content_item, ToolResult) for content_item in event.contents)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> str | None:
        """Event doesn't contain ToolCall, neither ToolResult.
        We return the content of the first Message type
        with role "assistant".
        """
        for item in event.contents:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content

        return None

    async def _step(self, context: ExecutionContext) -> None:
        # Prepare what to send to the LLM
        llm_request: LlmRequest = self._prepare_llm_request(context)

        # Get LLM's decision
        llm_response: LlmResponse = await self._think(llm_request)

        # Record LLM response as an Event
        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            contents=llm_response.contents,
        )
        context.add_event(response_event)

        # Execute tools if the LLM requested any
        tool_calls = [content_item for content_item in llm_response.contents if isinstance(content_item, ToolCall)]
        if tool_calls:
            tool_results = await self._act(context, tool_calls)
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                contents=tool_results,
            )
            context.add_event(tool_event)

        context.increment_step()

    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        """Prepare the LlmRequest based on the context."""

        # event.contents is a List of ContentItem objects
        flat_contents = [content_item for event in context.events for content_item in event.contents]

        return LlmRequest(
            instructions=self.instructions, # all requests contain the same instructions
            contents=flat_contents, # this changes per step, hence each request grows the +contents+
            tools=self.tools, # all requests contain the same tools
            tool_choice="auto" if self.tools else None, # all requests contain the same tool choice
        )

    async def _think(self, llm_request: LlmRequest) -> LlmResponse:
        return await self.llm_client.generate(llm_request)

    async def _act(
        self,
        context: ExecutionContext,
        tool_calls: list[ToolCall],
    ) -> list[ToolResult]:
        """Execute the tools that the Llm requested."""

        # a dictionary of all the tools that the Agent has been initialized with
        # and has access to. The tools are indexed by their name.
        tools_dict = {tool.name: tool for tool in self.tools}
        results: list[ToolResult] = []

        for tool_call in tool_calls:
            if tool_call.name not in tools_dict:
                raise ValueError(f"Tool {tool_call.name} not found in the available tools.")

            tool = tools_dict[tool_call.name]

            try:
                output = await tool.execute(context=context, **tool_call.arguments)
                results.append(
                    ToolResult(
                        tool_call_id=tool_call.tool_call_id,
                        name=tool_call.name,
                        status="success",
                        contents=[output],
                    )
                )
            except Exception as e:
                results.append(
                    ToolResult(
                        tool_call_id=tool_call.tool_call_id,
                        name=tool_call.name,
                        status="error",
                        contents=[str(e)]
                    )
                )

        return results
