from typing import Optional

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext

from config import MODEL_NAME
from tools import generate_newsletter

def newsletter_tool(tool_context: Optional[ToolContext] = None) -> str:
    """Wrapper around the newsletter generation function.

    Converts raw research text from the session state into a polished markdown
    newsletter. Returns the result as a plain-text string so the model can
    output it directly without unwrapping.

    Args:
        tool_context: The ADK ToolContext containing the session state.

    Returns:
        The generated markdown newsletter.
    """
    research = ""
    if tool_context and tool_context.state:
        research = tool_context.state.get("last_research", "")

    return generate_newsletter(research, tool_context)


content_agent = Agent(
    name="content_agent",
    model=MODEL_NAME,
    description="Converts research into a markdown newsletter.",
    instruction="""
You are a professional newsletter editor.

Your ONLY responsibility is to transform research into a polished markdown
newsletter.

Always call newsletter_tool.

The tool returns the finished markdown newsletter as plain text. Your final
answer must be EXACTLY that markdown text, copied verbatim, with no extra
commentary, no code fences, and no additional wrapping before or after it.

Use this structure (already enforced by the tool, so just pass it through):

# Title

## Hook

## Core Body

## Actionable Takeaways

## Sources

Do NOT perform web searches.

Only use the provided research.
""",
    tools=[newsletter_tool],
)