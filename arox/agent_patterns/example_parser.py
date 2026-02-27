import uuid

import yaml
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


def parse_example_yaml(yaml_content: str) -> list[ModelMessage]:
    data = yaml.safe_load(yaml_content)
    messages = []

    # Track tool call IDs to match with results
    last_tool_calls = []

    for entry in data:
        role = entry.get("role")
        if role == "user":
            parts = []
            if "content" in entry:
                parts.append(UserPromptPart(content=entry["content"]))
            if "tool_results" in entry:
                for res in entry["tool_results"]:
                    # Find matching tool call by tool name
                    # This is a bit heuristic if there are multiple calls to same tool
                    tool_name = res["tool"]
                    matching_tc = None
                    for tc in last_tool_calls:
                        if tc.tool_name == tool_name:
                            matching_tc = tc
                            last_tool_calls.remove(tc)
                            break

                    if matching_tc:
                        parts.append(
                            ToolReturnPart(
                                tool_name=tool_name,
                                content=res["content"],
                                tool_call_id=matching_tc.tool_call_id,
                            )
                        )
            messages.append(ModelRequest(parts=parts))

        elif role == "assistant":
            parts = []
            if "content" in entry:
                parts.append(TextPart(content=entry["content"]))
            if "tool_calls" in entry:
                last_tool_calls = []  # Reset for this response
                for tc in entry["tool_calls"]:
                    tc_part = ToolCallPart(
                        tool_name=tc["tool"],
                        args=tc.get("parameters", {}),
                        tool_call_id=f"call_{uuid.uuid4().hex[:8]}",
                    )
                    parts.append(tc_part)
                    last_tool_calls.append(tc_part)
            messages.append(ModelResponse(parts=parts))

    return messages
