import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import AsyncGenerator

from pydantic import ConfigDict

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from agents.research_agent import research_agent
from agents.content_agent import content_agent


class NewsletterPipelineAgent(BaseAgent):
    """
    Custom orchestrator: runs research_agent first, checks whether it
    produced usable output, and only then runs content_agent. If research
    comes back empty, the pipeline stops early with a clear message instead
    of handing an empty string to the content agent.
    
    Extended to support a complete Human-in-the-Loop Approval Workflow with
    draft versioning, draft editing, and approval commands.
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
        super().__init__(
            name=name,
            research_agent=research_agent,
            content_agent=content_agent,
            sub_agents=[research_agent, content_agent],
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        user_msg = ""
        if ctx.user_content and ctx.user_content.parts:
            user_msg = ctx.user_content.parts[0].text or ""

        approval_status = ctx.session.state.get("approval_status", "Pending")
        current_draft = ctx.session.state.get("current_draft", "")
        draft_version = ctx.session.state.get("draft_version", 0)
        draft_history = list(ctx.session.state.get("draft_history", []))

        if approval_status == "Approved":
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=f"{current_draft}\n\n---\n### Final Newsletter Summary\n- **Version Number**: Version {draft_version}\n- **Status**: Already Approved"
                        )
                    ]
                )
            )
            return

        APPROVAL_COMMANDS = {
            "approve", "publish", "looks good", "done", "finalize", "finish", "yes publish", "approved"
        }
        clean_msg = user_msg.strip().lower().rstrip(".!?")
        if clean_msg in APPROVAL_COMMANDS and current_draft:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            ctx.session.state["approval_status"] = "Approved"
            
            approval_text = f"""{current_draft}

---
### Final Newsletter Summary
- **Version Number**: Version {draft_version}
- **Approval Timestamp**: {timestamp}
- **Status**: Approved
"""
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=approval_text)]
                ),
                actions=EventActions(
                    state_delta={
                        "approval_status": "Approved"
                    }
                )
            )
            return

        RESEARCH_REQUESTS = {
            "update the research", "latest news", "latest statistics", "refresh data"
        }
        user_msg_lower = user_msg.lower()
        needs_new_research = any(req in user_msg_lower for req in RESEARCH_REQUESTS)

        research_text = ""
        if draft_version == 0 or needs_new_research:
            async for event in self.research_agent.run_async(ctx):
                yield event

                if event.is_final_response() and event.content and event.content.parts:
                    research_text = event.content.parts[0].text or ""

            if not research_text.strip():
                yield Event(
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
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        version_entry = {
            "version": new_version,
            "timestamp": timestamp,
            "user_instruction": user_msg,
            "draft": draft_text
        }
        draft_history.append(version_entry)

        state_updates = {
            "current_draft": draft_text,
            "draft_version": new_version,
            "approval_status": "Pending",
            "last_user_instruction": user_msg,
            "draft_history": draft_history
        }
        for k, v in state_updates.items():
            ctx.session.state[k] = v

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_updates)
        )

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=f"\n\n**The newsletter is currently a draft (Version {new_version}).**\n"
                             "Please review it. You can provide editing feedback (e.g., 'Shorten it', 'Use bullet points', 'Rewrite the introduction') or type **Approve** if you are satisfied."
                    )
                ]
            )
        )


root_agent = NewsletterPipelineAgent(
    name="newsletter_pipeline",
    research_agent=research_agent,
    content_agent=content_agent,
)