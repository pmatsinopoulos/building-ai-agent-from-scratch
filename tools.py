from abc import ABC, abstractmethod
import inspect, os
from typing import Any, Callable, Literal, cast, overload

from execution_context import ExecutionContext
from function_to_tool_utils import format_tool_definition, function_to_input_schema
from tavily import TavilyClient

class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(
        self,
        name: str | None= None,
        description: str | None = None,
        tool_definition: dict[str, Any] | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition or self._generate_tool_definition()

    @property
    def tool_definition(self) -> dict[str, Any] | None:
        return self._tool_definition

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _generate_tool_definition(self) -> dict[str, Any]:
        pass

    async def __call__(self, context: ExecutionContext, **kwargs: Any) -> Any:
        return await self.execute(context, **kwargs)


class FunctionTool(BaseTool):
    """Wraps a Python function as a BaseTool.

    Any Python function can be used as a Tool. If the function
    takes as argument the context/ExecutionContext when we will
    pass it. Otherwise, we will not.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        tool_definition: dict[str, Any] | None = None,
    ):
        self.func = func
        name = name or func.__name__
        description = description or (func.__doc__ or "").strip()

        super().__init__(
            name=name,
            description=description,
            tool_definition=tool_definition
        )

        self.needs_context = 'context' in inspect.signature(func).parameters

    async def execute(
        self,
        context: ExecutionContext,
        **kwargs: Any
    ) -> Any:
        if self.needs_context:
            result =self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)

        if inspect.iscoroutine(result):
            return await result

        return result

    def _generate_tool_definition(self) -> dict[str, Any]:
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)


@overload
def tool(func: Callable[..., Any], /) -> FunctionTool: ...

@overload
def tool(*, name: str | None = None, description: str | None = None) -> Callable[[Callable[..., Any]], FunctionTool]: ...

def tool(
    func: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """Decorator to turn a function into a FunctionTool.

    Can be used bare or with optional overrides:

        @tool
        def calculator(expression: str) -> float: ...

        @tool(name="web_search", description="Search the internet")
        def search_web(query: str) -> str: ...
    """
    if func is not None:
        return FunctionTool(func)

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func, name=name, description=description)

    return decorator

@tool
def calculator(expression: str) -> float:
    """Calculate the result of a mathematical expression."""
    return cast(float, eval(expression))

@tool
def search_web(
    query: str,
    max_results: int = 2,
    topic: Literal["general", "news", "finance"] = "general",
    time_range: Literal["day", "week", "month", "year"] | None = None,
    country: str | None = None) -> list[dict[str, Any]] | str:
    """Search the web using Tavily API.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        topic: Search topic - 'general', 'news', or 'finance'
        time_range: Time range filter - 'day', 'week', 'month', or 'year'
        country: Country filter (e.g., 'US', 'UK', 'CA', 'AU', 'NZ')
    """
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        kwargs: dict[str, Any] = dict(query=query, max_results=max_results, topic=topic)
        if time_range is not None:
            kwargs["time_range"] = time_range
        if country is not None:
            kwargs["country"] = country
        response = cast(dict[str, Any], tavily_client.search(**kwargs))  # pyright: ignore[reportUnknownMemberType]
        return cast(list[dict[str, Any]], response.get("results", []))
    except Exception as e:
        return f"Search error: {str(e)}"
