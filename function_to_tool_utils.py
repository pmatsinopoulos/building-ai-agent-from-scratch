import inspect
from typing import Any, Callable

from pydantic import BaseModel


def function_to_input_schema(func: Callable[..., Any]) -> dict[str, Any]:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        annotation = param.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            parameters[param.name] = annotation.model_json_schema()
        else:
            param_type = type_map.get(annotation, "string") if isinstance(annotation, type) else "string"
            parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default is inspect.Parameter.empty
    ]

    return {
            "type": "object",
            "properties": parameters,
            "required": required,
        }

def format_tool_definition(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    }
