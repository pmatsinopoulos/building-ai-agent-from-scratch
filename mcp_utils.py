from typing import Any

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from tools import FunctionTool


async def load_mcp_tools(connection: dict[str, Any]) -> list[FunctionTool]:
    """Load tools from an MCP server and converts them to BaseTool objects."""
    tools: list[FunctionTool] = []

    async with stdio_client(StdioServerParameters(**connection)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            for mcp_tool in mcp_tools.tools:
                func_tool = _create_mcp_tool(mcp_tool, connection)
                tools.append(func_tool)

    return tools


def _create_mcp_tool(mcp_tool: Tool, connection: dict[str, Any]) -> FunctionTool:
    """Create a FunctionTool from an MCP tool."""

    async def call_mcp(**kwargs: Any) -> Any:
        async with stdio_client(StdioServerParameters(**connection)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result: CallToolResult = await session.call_tool(mcp_tool.name, kwargs)
                return _extract_text_content(result)

    tool_definition = {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        }
    }

    return FunctionTool(
        func=call_mcp,
        name=mcp_tool.name,
        description=mcp_tool.description,
        tool_definition=tool_definition
    )

def _extract_text_content(result: CallToolResult) -> str:
    """Extract concatenated text from an MCP CallToolResult."""
    return "\n".join(
        block.text for block in result.content if block.type == "text"
    )
