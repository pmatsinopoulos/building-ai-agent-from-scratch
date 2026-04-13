from abc import ABC, abstractmethod
from typing import Any

from execution_context import ExecutionContext


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
        self._tool_definition = tool_definition

    @property
    def tool_definition(self) -> dict[str, Any] | None:
        return self._tool_definition

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs: Any) -> Any:
        pass

    async def __call__(self, context: ExecutionContext, **kwargs: Any) -> Any:
        return await self.execute(context, **kwargs)
