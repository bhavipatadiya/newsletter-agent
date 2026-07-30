from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from tools import generate_newsletter
from config import MODEL_NAME


def newsletter_tool(
    newsletter_text: str = "",
    research_text: str = "",
    tool_context: ToolContext = None
) -> str:
    """
    Convert research into a markdown newsletter.

    Returns the newsletter as a plain-text (markdown) string, not a dict,
    so there is nothing for the model to "unwrap" - it can just pass the
    markdown straight through as its final answer.
    """
    research = research_text
    if not research and tool_context and tool_context.state:
        research = tool_context.state.get("last_research", "")
    
    if newsletter_text and tool_context and tool_context.state:
        tool_context.state["last_newsletter"] = newsletter_text

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