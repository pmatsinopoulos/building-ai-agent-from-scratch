"""Human-in-the-loop (HITL) patterns for agent tool execution.

This exercise expands the basic approval callback into a richer set of
human-in-the-loop interactions. The LLM proposes a tool call, but a human
remains in control of:

  1. APPROVE  - run the tool as proposed
  2. DENY     - block the tool, optionally feeding a reason back to the LLM
  3. MODIFY   - edit the proposed arguments before execution
  4. REPLACE  - skip the tool entirely and inject a manual result
  5. ALLOW-ALL (this session) - whitelist a tool for the rest of the run
  6. REVIEW (after execution) - inspect/redact a tool's result before it
                                 is sent back to the LLM

Each pattern is implemented in this file via the agent's
``before_tool_callbacks`` and ``after_tool_callbacks`` hooks.
"""

import asyncio
import json
from typing import Any

from agent import Agent, AgentResult
from content_types import ToolCall, ToolResult
from delete_file import delete_file
from execution_context import ExecutionContext
from llm_client import LlmClient
from list_files import list_files
from read_file_contents import read_file_contents
from send_email import send_email

# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------
# Tools that mutate the world or have side effects require explicit approval.
DANGEROUS_TOOLS = {"delete_file", "send_email", "execute_sql"}

# Tools whose *output* may contain sensitive information the human may want
# to review or redact before it is fed back into the LLM context.
SENSITIVE_OUTPUT_TOOLS = {"read_file_contents"}

# Key used to persist session-level decisions inside ``context.state``.
SESSION_ALLOWLIST_KEY = "hitl_session_allowlist"


# ---------------------------------------------------------------------------
# Small CLI helpers
# ---------------------------------------------------------------------------
def _prompt(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or (default or "")


def _print_banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _print_tool_call(tool_call: ToolCall) -> None:
    print(f"Tool:      {tool_call.name}")
    print(f"Arguments: {json.dumps(tool_call.arguments, indent=2)}")


def _edit_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Let the human edit each argument value via a simple line-by-line prompt.

    Pressing ENTER keeps the current value. Values are parsed as JSON when
    possible so the user can type ``42``, ``true``, or ``["a","b"]`` directly.
    """
    edited: dict[str, Any] = {}
    print("\nEdit arguments (ENTER to keep current value):")
    for key, value in arguments.items():
        raw = _prompt(f"  {key}", default=json.dumps(value))
        try:
            edited[key] = json.loads(raw)
        except json.JSONDecodeError:
            edited[key] = raw
    return edited


# ---------------------------------------------------------------------------
# Before-tool callback: gate dangerous tools through a human
# ---------------------------------------------------------------------------
def human_approval_callback(
    context: ExecutionContext,
    tool_call: ToolCall,
) -> str | None:
    """Interactive HITL gate executed *before* a tool runs.

    Returning ``None`` lets the tool execute normally. Returning a string
    short-circuits execution: the string becomes the tool's result and is
    fed back to the LLM (so the model can react to denial / replacement).
    """

    # Skip the human entirely for safe tools.
    if tool_call.name not in DANGEROUS_TOOLS:
        return None

    # Honour any session-level "always allow" decisions.
    allowlist: set[str] = context.state.setdefault(SESSION_ALLOWLIST_KEY, set())
    if tool_call.name in allowlist:
        print(f"\n[HITL] Auto-approved (session allowlist): {tool_call.name}")
        return None

    _print_banner(f"DANGEROUS TOOL REQUEST: {tool_call.name}")
    _print_tool_call(tool_call)

    print("\nChoose an action:")
    print("  [a] approve       - run as-is")
    print("  [w] allow-all     - approve and whitelist for this session")
    print("  [d] deny          - block, send reason back to the LLM")
    print("  [m] modify        - edit arguments before running")
    print("  [r] replace       - skip execution, return manual result")

    choice = _prompt("Your choice", default="d").lower()

    if choice in ("a", "approve"):
        print("[HITL] Approved.\n")
        return None

    if choice in ("w", "allow-all"):
        allowlist.add(tool_call.name)
        print(f"[HITL] Approved and whitelisted '{tool_call.name}' for this session.\n")
        return None

    if choice in ("m", "modify"):
        # Mutating ``tool_call.arguments`` in place causes the agent to execute
        # the tool with the *edited* arguments, because _act() reads
        # ``tool_call.arguments`` after the before-callbacks return.
        tool_call.arguments = _edit_arguments(tool_call.arguments)
        print(f"[HITL] Arguments updated. Executing {tool_call.name}...\n")
        return None

    if choice in ("r", "replace"):
        manual_result = _prompt("Enter the result to return to the LLM")
        print("[HITL] Execution skipped, manual result injected.\n")
        return f"[human-provided result] {manual_result}"

    # Default / "d" / anything else => deny with a reason.
    reason = _prompt("Reason (sent back to the LLM)", default="not allowed by user")
    print(f"[HITL] Denied: {reason}\n")
    return f"User denied execution of {tool_call.name}. Reason: {reason}"


# ---------------------------------------------------------------------------
# After-tool callback: review/redact sensitive output before the LLM sees it
# ---------------------------------------------------------------------------
def human_review_callback(
    context: ExecutionContext,
    tool_result: ToolResult,
) -> ToolResult | None:
    """Interactive HITL review executed *after* a sensitive tool runs.

    Returning ``None`` keeps the tool result as-is. Returning a new
    ``ToolResult`` overrides what the LLM will see in its next step.
    """

    if tool_result.name not in SENSITIVE_OUTPUT_TOOLS:
        return None
    if tool_result.status != "success":
        return None

    _print_banner(f"REVIEW OUTPUT: {tool_result.name}")
    for item in tool_result.contents:
        print(item)

    choice = _prompt(
        "\n[k]eep / [e]dit / [b]lock the output before sending it to the LLM",
        default="k",
    ).lower()

    if choice in ("k", "keep"):
        return None

    if choice in ("b", "block"):
        return tool_result.model_copy(
            update={
                "contents": ["[output redacted by human reviewer]"],
            }
        )

    # edit
    print("Enter replacement output (end with an empty line):")
    lines: list[str] = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return tool_result.model_copy(update={"contents": ["\n".join(lines)]})


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    agent = Agent(
        name="hitl_file_agent",
        llm_client=LlmClient(model="gpt-5"),
        tools=[list_files, read_file_contents, delete_file, send_email],
        instructions=[
            "You are a helpful assistant that can explore files, send emails, "
            "and delete files.",
            "A human supervisor may approve, deny, modify, or replace any "
            "dangerous action you propose - adapt to their feedback.",
        ],
        max_steps=20,
        before_tool_callbacks=[human_approval_callback],
        after_tool_callbacks=[human_review_callback],
    )

    user_input = (
        "List the files in the current directory, then send an email to "
        "alice@example.com summarising what you found, and finally delete "
        "the file 'old_notes.txt'."
    )

    result: AgentResult = await agent.run(user_input=user_input)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
