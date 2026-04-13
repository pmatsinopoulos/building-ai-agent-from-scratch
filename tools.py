from abc import ABC, abstractmethod
import inspect
from typing import Any, Callable

from execution_context import ExecutionContext
from function_to_tool_utils import format_tool_definition, function_to_input_schema


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
        name = name or func.__name__
        description = description or (func.__doc__ or "").strip()

        super().__init__(
            name=name,
            description=description,
            tool_definition=tool_definition
        )

        self.func = func
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
