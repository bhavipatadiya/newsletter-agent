from google import genai
from google.genai import types
from google.adk.tools.tool_context import ToolContext
from config import GOOGLE_API_KEY, MODEL_NAME
from prompts import RESEARCH_PROMPT, CONTENT_PROMPT
import time

client = genai.Client(api_key=GOOGLE_API_KEY)

def generate_content_with_retry(contents, config=None, tools=None):
    """
    Calls the Gemini API with automatic retries (exponential backoff)
    for transient errors (503, 429) and falls back to a different model
    if the primary model remains unavailable.
    """
    models_to_try = [MODEL_NAME, "gemini-2.5-flash", "gemini-1.5-flash"]
    seen = set()
    models = []
    for m in models_to_try:
        if m not in seen:
            seen.add(m)
            models.append(m)
    last_err = None
    for model in models:
        for attempt in range(3):
            try:
                # Prepare call config
                call_config = config or types.GenerateContentConfig()
                if tools:
                    call_config.tools = tools
                
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=call_config,
                )
                return response
            except Exception as e:
                last_err = e
                err_msg = str(e)
                # Check for 503 Unavailable or 429 Rate Limit
                if any(x in err_msg for x in ["503", "429", "UNAVAILABLE", "ResourceExhausted", "RESOURCE_EXHAUSTED", "limit"]):
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-transient error, break attempt loop to try next model
                    break
    raise last_err

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

    response = generate_content_with_retry(
        contents=prompt,
        tools=[
            types.Tool(
                google_search=types.GoogleSearch()
            )
        ]
    ),

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

    response = generate_content_with_retry(
        contents=prompt
    )

    newsletter_res = response.text
    if tool_context:
        tool_context.state["last_newsletter"] = newsletter_res

    return newsletter_res
