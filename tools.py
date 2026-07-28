from google import genai
from google.genai import types
from google.adk.tools.tool_context import ToolContext
from config import GOOGLE_API_KEY, MODEL_NAME
from prompts import RESEARCH_PROMPT, CONTENT_PROMPT

client = genai.Client(api_key=GOOGLE_API_KEY)

def log_newsletter(topic: str, tool_context: ToolContext) -> dict:
    history = tool_context.state.get("newsletter_history", [])
    history.append(topic)
    tool_context.state["newsletter_history"] = history
    
    return {
        "status": "logged",
        "topic": topic,
        "total_newsletters": len(history)
    }

def research_topic(topic: str = "", tool_context: ToolContext = None) -> str:
    """
    Research a topic using Gemini + Google Search.

    Returns raw research.
    """
    user_msg = ""
    if tool_context and tool_context.user_content and tool_context.user_content.parts:
        user_msg = tool_context.user_content.parts[0].text or ""
    
    last_topic = tool_context.state.get("last_topic") if tool_context else None
    last_research = tool_context.state.get("last_research") if tool_context else None
    last_newsletter = tool_context.state.get("last_newsletter") if tool_context else None
    newsletter_history = tool_context.state.get("newsletter_history", []) if tool_context else []

    resolved_topic = topic
    if not resolved_topic and last_topic:
        resolved_topic = last_topic

    if not resolved_topic:
        return "Please provide a topic to research."

    if tool_context:
        log_newsletter(resolved_topic, tool_context)

    session_context = ""
    if tool_context and tool_context.state:
        session_context = f"""
Previous Session State:
- Last Topic Researched: {last_topic}
- Last Research: {last_research}
- Last Newsletter: {last_newsletter}
- Newsletter History: {newsletter_history}
"""

    prompt = f"""
{RESEARCH_PROMPT}

{session_context}

Topic to research:
{resolved_topic}

User Message:
{user_msg}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    google_search=types.GoogleSearch()
                )
            ]
        ),
    )

    research_res = response.text
    if tool_context:
        tool_context.state["last_topic"] = resolved_topic
        tool_context.state["last_research"] = research_res
        
    return research_res

def generate_newsletter(research_text: str = "", tool_context: ToolContext = None) -> str:
    """
    Convert research into a markdown newsletter.
    """
    last_newsletter = tool_context.state.get("last_newsletter") if tool_context else None
    user_msg = ""
    if tool_context and tool_context.user_content and tool_context.user_content.parts:
        user_msg = tool_context.user_content.parts[0].text or ""

    session_context = ""
    if last_newsletter:
        session_context = f"\nLast Generated Newsletter:\n{last_newsletter}\n"

    prompt = f"""
{CONTENT_PROMPT}

{session_context}

Research:
{research_text}

User Feedback/Request:
{user_msg}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    newsletter_res = response.text
    if tool_context:
        tool_context.state["last_newsletter"] = newsletter_res

    return newsletter_res
