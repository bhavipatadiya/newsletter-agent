import datetime
import os
import sys
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import ConfigDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.content_agent import content_agent
from agents.research_agent import research_agent

class NewsletterPipelineAgent(BaseAgent):
    """Custom orchestrator agent for generating newsletters.

    Coordinates research_agent and content_agent. Implements a human-in-the-loop
    approval workflow, draft versioning, draft editing, and approval commands.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    research_agent: LlmAgent
    content_agent: LlmAgent

    def __init__(
        self,
        name: str,
        research_agent: LlmAgent,
        content_agent: LlmAgent,
    ):
        """Initializes the NewsletterPipelineAgent with its dependencies.

        Args:
            name: Name of the orchestrator agent.
            research_agent: Agent responsible for collecting research.
            content_agent: Agent responsible for draft copy writing.
        """
        super().__init__(
            name=name,
            research_agent=research_agent,
            content_agent=content_agent,
            sub_agents=[research_agent, content_agent],
        )

    def _get_user_message(self, ctx: InvocationContext) -> str:
        """Extracts the user message text from the invocation context."""
        if ctx.user_content and ctx.user_content.parts:
            return ctx.user_content.parts[0].text or ""
        return ""

    def _is_approved_already(self, ctx: InvocationContext) -> bool:
        """Checks if the newsletter has already been approved."""
        return ctx.session.state.get("approval_status", "Pending") == "Approved"

    def _should_approve(self, user_msg: str, current_draft: str) -> bool:
        """Determines if the user wants to approve the current draft."""
        APPROVAL_COMMANDS = {
            "approve",
            "publish",
            "looks good",
            "done",
            "finalize",
            "finish",
            "yes publish",
            "approved",
        }
        clean_msg = user_msg.strip().lower().rstrip(".!?")
        return bool(clean_msg in APPROVAL_COMMANDS and current_draft)

    def _needs_new_research(self, user_msg: str, draft_version: int) -> bool:
        """Determines if the pipeline should trigger a new research step."""
        RESEARCH_REQUESTS = {
            "update the research",
            "latest news",
            "latest statistics",
            "refresh data",
        }
        user_msg_lower = user_msg.lower()
        has_research_request = any(
            req in user_msg_lower for req in RESEARCH_REQUESTS
        )
        return draft_version == 0 or has_research_request

    def _build_already_approved_event(self, draft: str, version: int) -> Event:
        """Builds the event returned when the newsletter has already been approved."""
        return Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            f"{draft}\n\n---\n### Final Newsletter Summary\n"
                            f"- **Version Number**: Version {version}\n"
                            f"- **Status**: Already Approved"
                        )
                    ),
                ],
            ),
        )

    def _approve_draft(
        self, ctx: InvocationContext, draft: str, version: int
    ) -> Event:
        """Marks the current draft as approved and saves status to session state."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ctx.session.state["approval_status"] = "Approved"

        approval_text = f"""{draft}

---
### Final Newsletter Summary
- **Version Number**: Version {version}
- **Approval Timestamp**: {timestamp}
- **Status**: Approved
"""
        return Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=approval_text)]
            ),
            actions=EventActions(state_delta={"approval_status": "Approved"}),
        )

    def _build_empty_research_event(self) -> Event:
        """Builds the event returned when research yields no content."""
        return Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text="The research step did not return any usable content."
                    )
                ],
            ),
        )

    def _save_new_draft(
        self,
        ctx: InvocationContext,
        draft_text: str,
        new_version: int,
        user_msg: str,
        draft_history: list,
    ) -> Event:
        """Saves a new newsletter draft version to the session state."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        version_entry = {
            "version": new_version,
            "timestamp": timestamp,
            "user_instruction": user_msg,
            "draft": draft_text,
        }
        draft_history.append(version_entry)

        state_updates = {
            "current_draft": draft_text,
            "draft_version": new_version,
            "approval_status": "Pending",
            "last_user_instruction": user_msg,
            "draft_history": draft_history,
        }
        for k, v in state_updates.items():
            ctx.session.state[k] = v

        return Event(
            author=self.name,
            actions=EventActions(state_delta=state_updates),
        )

    def _build_draft_ready_event(self, version: int) -> Event:
        """Builds the event notifying the user that a new draft is ready for review."""
        return Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            f"\n\n**The newsletter is currently a draft (Version {version}).**\n"
                            "Please review it. You can provide editing feedback (e.g., 'Shorten it', "
                            "'Use bullet points', 'Rewrite the introduction') or type **Approve** "
                            "if you are satisfied."
                        )
                    ),
                ],
            ),
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Runs the orchestrator agent pipeline asynchronously.

        Orchestrates research first, and then content compilation while checking
        for user approval or research updates.
        """
        user_msg = self._get_user_message(ctx)
        current_draft = ctx.session.state.get("current_draft", "")
        draft_version = ctx.session.state.get("draft_version", 0)
        draft_history = list(ctx.session.state.get("draft_history", []))

        if self._is_approved_already(ctx):
            yield self._build_already_approved_event(current_draft, draft_version)
            return

        if self._should_approve(user_msg, current_draft):
            yield self._approve_draft(ctx, current_draft, draft_version)
            return

        research_text = ""
        if self._needs_new_research(user_msg, draft_version):
            async for event in self.research_agent.run_async(ctx):
                yield event
                if event.is_final_response() and event.content and event.content.parts:
                    research_text = event.content.parts[0].text or ""

            if not research_text.strip():
                yield self._build_empty_research_event()
                return
        else:
            research_text = ctx.session.state.get("last_research", "")

        ctx.session.state["last_research"] = research_text
        if current_draft:
            ctx.session.state["last_newsletter"] = current_draft

        draft_text = ""
        async for event in self.content_agent.run_async(ctx):
            yield event
            if event.is_final_response() and event.content and event.content.parts:
                draft_text = event.content.parts[0].text or ""

        if not draft_text.strip():
            draft_text = current_draft or "Error generating newsletter draft."

        new_version = draft_version + 1
        yield self._save_new_draft(
            ctx, draft_text, new_version, user_msg, draft_history
        )
        yield self._build_draft_ready_event(new_version)


root_agent = NewsletterPipelineAgent(
    name="newsletter_pipeline",
    research_agent=research_agent,
    content_agent=content_agent,
)