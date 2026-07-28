from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from tools import research_topic
from config import MODEL_NAME


def research_tool(topic: str = "", tool_context: ToolContext = None) -> str:
    """
    Wrapper around the research function.

    Returns the raw research findings as a single plain-text string
    (not a dict) - there's nothing to unwrap this way, so the model
    can only pass the text straight through as its final answer.
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