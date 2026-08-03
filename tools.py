import logging
import time
from typing import Any, List, Optional

from google import genai
from google.genai import types
from google.adk.tools.tool_context import ToolContext

from config import GOOGLE_API_KEY, MODEL_NAME
from prompts import CONTENT_PROMPT, RESEARCH_PROMPT

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_RETRY_DELAY = 2
RETRY_ERRORS = (
    "429",
    "503",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "ResourceExhausted",
    "limit",
)

client = genai.Client(api_key=GOOGLE_API_KEY)

def is_retryable_error(error: Exception) -> bool:
    """Checks if the given exception is a transient error that can be retried.

    Args:
        error: The exception to analyze.

    Returns:
        True if the error is retryable, False otherwise.
    """
    code = getattr(error, "code", None)
    if code in (429, 503):
        return True

    err_msg = str(error)
    return any(code in err_msg for code in RETRY_ERRORS)

def _attempt_generate_content(
    contents: Any,
    config: Optional[types.GenerateContentConfig] = None,
    tools: Optional[List[types.Tool]] = None,
) -> types.GenerateContentResponse:
    """Performs a single attempt to generate content using the GenAI client.

    Args:
        contents: The contents to send to the Gemini API.
        config: Optional content generation config.
        tools: Optional list of tools to use.

    Returns:
        The response from the Gemini API.
    """
    call_config = config or types.GenerateContentConfig()
    if tools:
        call_config.tools = tools

    return client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=call_config,
    )

def generate_content_with_retry(
    contents: Any,
    config: Optional[types.GenerateContentConfig] = None,
    tools: Optional[List[types.Tool]] = None,
) -> types.GenerateContentResponse:
    """Calls the Gemini API with automatic retries (exponential backoff) for transient errors.

    Args:
        contents: The input content for the model.
        config: Configuration options for generation.
        tools: List of tools available to the model.

    Returns:
        The response from the model.

    Raises:
        Exception: The last encountered exception if all retry attempts fail.
    """
    last_err = Exception("Unknown API error occurred")
    for attempt in range(MAX_RETRIES):
        try:
            return _attempt_generate_content(contents, config, tools)
        except Exception as e:
            last_err = e
            if not is_retryable_error(e):
                logger.error("Non-retryable error encountered: %s", e)
                break

            wait_time = (attempt + 1) * BASE_RETRY_DELAY
            logger.warning(
                "Transient error (attempt %d/%d). Retrying in %d seconds. Error: %s",
                attempt + 1,
                MAX_RETRIES,
                wait_time,
                e,
            )
            time.sleep(wait_time)

    raise last_err

def get_user_message(tool_context: Optional[ToolContext]) -> str:
    """Extracts the first user message from the tool context.

    Args:
        tool_context: The ADK ToolContext.

    Returns:
        The user's message as a string, or an empty string if not found.
    """
    if not tool_context or not tool_context.user_content or not tool_context.user_content.parts:
        return ""
    return tool_context.user_content.parts[0].text or ""

def get_session_value(
    tool_context: Optional[ToolContext], key: str, default: Any = None
) -> Any:
    """Retrieves a value from the session state safely.

    Args:
        tool_context: The ADK ToolContext.
        key: The state key to retrieve.
        default: The default value to return if key does not exist.

    Returns:
        The value from session state, or the default value.
    """
    if not tool_context or not tool_context.state:
        return default
    return tool_context.state.get(key, default)

def save_session_value(
    tool_context: Optional[ToolContext], key: str, value: Any
) -> None:
    """Saves a value to the session state safely.

    Args:
        tool_context: The ADK ToolContext.
        key: The state key to save.
        value: The value to save.
    """
    if tool_context and tool_context.state is not None:
        tool_context.state[key] = value

def log_newsletter(topic: str, tool_context: ToolContext) -> dict:
    """Logs the researched topic into the session newsletter history.

    Args:
        topic: The topic that was researched.
        tool_context: The ADK ToolContext containing the session state.

    Returns:
        A dictionary with the logging status, topic, and total count.
    """
    history = get_session_value(tool_context, "newsletter_history", [])
    updated_history = list(history)
    updated_history.append(topic)
    save_session_value(tool_context, "newsletter_history", updated_history)

    return {
        "status": "logged",
        "topic": topic,
        "total_newsletters": len(updated_history),
    }

def build_research_prompt(
    resolved_topic: str,
    user_msg: str,
    has_session: bool,
    last_topic: Optional[str],
    last_research: Optional[str],
    last_newsletter: Optional[str],
    newsletter_history: list,
) -> str:
    """Constructs the structured prompt for topic research.

    Args:
        resolved_topic: The topic to research.
        user_msg: The latest message or instruction from the user.
        has_session: Whether session state exists.
        last_topic: The last topic that was researched.
        last_research: The results of the last research attempt.
        last_newsletter: The content of the last generated newsletter.
        newsletter_history: History of all newsletters in this session.

    Returns:
        The fully formatted prompt string.
    """
    session_context = ""
    if has_session:
        session_context = f"""
Previous Session State:
- Last Topic Researched: {last_topic}
- Last Research: {last_research}
- Last Newsletter: {last_newsletter}
- Newsletter History: {newsletter_history}
"""
    return f"""{RESEARCH_PROMPT}

{session_context}

Topic to research:
{resolved_topic}

User Message:
{user_msg}
"""


def research_topic(
    topic: str = "", tool_context: Optional[ToolContext] = None
) -> str:
    """Researches a topic using Gemini and Google Search.

    Args:
        topic: The specific topic to research.
        tool_context: The ADK ToolContext containing the session state.

    Returns:
        The raw research findings.
    """
    user_msg = get_user_message(tool_context)
    last_topic = get_session_value(tool_context, "last_topic")
    last_research = get_session_value(tool_context, "last_research")
    last_newsletter = get_session_value(tool_context, "last_newsletter")
    newsletter_history = get_session_value(tool_context, "newsletter_history", [])

    resolved_topic = topic or last_topic
    if not resolved_topic:
        return "Please provide a topic to research."

    if tool_context:
        log_newsletter(resolved_topic, tool_context)

    has_session = tool_context is not None and tool_context.state is not None
    prompt = build_research_prompt(
        resolved_topic=resolved_topic,
        user_msg=user_msg,
        has_session=has_session,
        last_topic=last_topic,
        last_research=last_research,
        last_newsletter=last_newsletter,
        newsletter_history=newsletter_history,
    )

    google_search_tool = types.Tool(google_search=types.GoogleSearch())
    response = generate_content_with_retry(
        contents=prompt, tools=[google_search_tool]
    )
    research_res = response.text or ""

    save_session_value(tool_context, "last_topic", resolved_topic)
    save_session_value(tool_context, "last_research", research_res)

    return research_res


def build_newsletter_prompt(
    research_text: str,
    user_msg: str,
    last_newsletter: Optional[str],
) -> str:
    """Constructs the prompt for generating the newsletter draft.

    Args:
        research_text: The source research text.
        user_msg: Feedback or edit requests from the user.
        last_newsletter: The previous generated newsletter draft.

    Returns:
        The fully formatted prompt string.
    """
    session_context = ""
    if last_newsletter:
        session_context = f"\nLast Generated Newsletter:\n{last_newsletter}\n"

    return f"""{CONTENT_PROMPT}

{session_context}

Research:
{research_text}

User Feedback/Request:
{user_msg}
"""


def generate_newsletter(
    research_text: str = "", tool_context: Optional[ToolContext] = None
) -> str:
    """Converts the compiled research text into a markdown newsletter.

    Args:
        research_text: The compiled research text.
        tool_context: The ADK ToolContext containing the session state.

    Returns:
        The generated markdown newsletter content.
    """
    last_newsletter = get_session_value(tool_context, "last_newsletter")
    user_msg = get_user_message(tool_context)

    prompt = build_newsletter_prompt(
        research_text=research_text,
        user_msg=user_msg,
        last_newsletter=last_newsletter,
    )

    response = generate_content_with_retry(contents=prompt)
    newsletter_res = response.text or ""

    save_session_value(tool_context, "last_newsletter", newsletter_res)

    return newsletter_res