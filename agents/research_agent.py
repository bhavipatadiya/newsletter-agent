from typing import Optional

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext

from config import MODEL_NAME
from tools import research_topic

def research_tool(
    topic: str = "", tool_context: Optional[ToolContext] = None
) -> str:
    """Wrapper around the research function to interface with the research agent.

    Returns the raw research findings as a single plain-text string. Returning a
    simple string instead of a dict prevents the model from having to unwrap any
    nested structures, allowing it to pass the text directly through.

    Args:
        topic: The topic to research.
        tool_context: The ADK ToolContext containing the session state.

    Returns:
        The raw research findings.
    """
    return research_topic(topic, tool_context)


research_agent = Agent(
    name="research_agent",
    model=MODEL_NAME,
    description="Researches a topic using Google Search.",
    instruction="""
You are an expert research assistant.

Your ONLY responsibility is to collect information.

Always call the research_tool. If the user is asking a follow-up, referring to a previous topic, asking to improve, summarize, continue, or show a previous newsletter, or requesting to generate another newsletter, call research_tool with the appropriate topic (or leave topic empty if none is specified).

The tool returns plain text findings. Your final answer must be EXACTLY
that plain text, copied verbatim, with no additional formatting, no JSON,
no code fences, and no extra commentary before or after it.

Do NOT summarize.

Do NOT write newsletters.
""",
    tools=[research_tool],
)