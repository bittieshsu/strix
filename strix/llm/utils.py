import html
import re
from typing import Any


STRIX_API_BASE = "https://models.strix.ai/api/v1"

STRIX_PROVIDER_PREFIXES: dict[str, str] = {
    "claude-": "anthropic",
    "gpt-": "openai",
    "gemini-": "gemini",
}


def is_strix_model(model_name: str | None) -> bool:
    """Check if model uses strix/ prefix."""
    return bool(model_name and model_name.startswith("strix/"))


def get_strix_api_base(model_name: str | None) -> str | None:
    """Return Strix API base URL if using strix/ model, None otherwise."""
    if is_strix_model(model_name):
        return STRIX_API_BASE
    return None


def get_litellm_model_name(model_name: str | None) -> str | None:
    """Convert strix/ prefixed model to litellm-compatible provider/model format.

    Maps strix/ models to their corresponding litellm provider:
    - strix/claude-* -> anthropic/claude-*
    - strix/gpt-* -> openai/gpt-*
    - strix/gemini-* -> gemini/gemini-*
    - Other models -> openai/<model> (routed via Strix API)
    """
    if not model_name:
        return model_name
    if not model_name.startswith("strix/"):
        return model_name

    base_model = model_name[6:]

    for prefix, provider in STRIX_PROVIDER_PREFIXES.items():
        if base_model.startswith(prefix):
            return f"{provider}/{base_model}"

    return f"openai/{base_model}"


def _truncate_to_first_function(content: str) -> str:
    if not content:
        return content

    function_starts = [match.start() for match in re.finditer(r"<function=", content)]

    if len(function_starts) >= 2:
        second_function_start = function_starts[1]

        return content[:second_function_start].rstrip()

    return content


def parse_tool_invocations(content: str) -> list[dict[str, Any]] | None:
    content = fix_incomplete_tool_call(content)

    tool_invocations: list[dict[str, Any]] = []

    fn_regex_pattern = r"<function=([^>]+)>\n?(.*?)</function>"
    fn_param_regex_pattern = r"<parameter=([^>]+)>(.*?)</parameter>"

    fn_matches = re.finditer(fn_regex_pattern, content, re.DOTALL)

    for fn_match in fn_matches:
        fn_name = fn_match.group(1)
        fn_body = fn_match.group(2)

        param_matches = re.finditer(fn_param_regex_pattern, fn_body, re.DOTALL)

        args = {}
        for param_match in param_matches:
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()

            param_value = html.unescape(param_value)
            args[param_name] = param_value

        tool_invocations.append({"toolName": fn_name, "args": args})

    return tool_invocations if tool_invocations else None


def fix_incomplete_tool_call(content: str) -> str:
    """Fix incomplete tool calls by adding missing </function> tag."""
    if (
        "<function=" in content
        and content.count("<function=") == 1
        and "</function>" not in content
    ):
        content = content.rstrip()
        content = content + "function>" if content.endswith("</") else content + "\n</function>"
    return content


def format_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    xml_parts = [f"<function={tool_name}>"]

    for key, value in args.items():
        xml_parts.append(f"<parameter={key}>{value}</parameter>")

    xml_parts.append("</function>")

    return "\n".join(xml_parts)


def clean_content(content: str) -> str:
    if not content:
        return ""

    content = fix_incomplete_tool_call(content)

    tool_pattern = r"<function=[^>]+>.*?</function>"
    cleaned = re.sub(tool_pattern, "", content, flags=re.DOTALL)

    incomplete_tool_pattern = r"<function=[^>]+>.*$"
    cleaned = re.sub(incomplete_tool_pattern, "", cleaned, flags=re.DOTALL)

    partial_tag_pattern = r"<f(?:u(?:n(?:c(?:t(?:i(?:o(?:n(?:=(?:[^>]*)?)?)?)?)?)?)?)?)?$"
    cleaned = re.sub(partial_tag_pattern, "", cleaned)

    hidden_xml_patterns = [
        r"<inter_agent_message>.*?</inter_agent_message>",
        r"<agent_completion_report>.*?</agent_completion_report>",
    ]
    for pattern in hidden_xml_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)

    return cleaned.strip()
