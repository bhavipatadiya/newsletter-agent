import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import AsyncGenerator

from pydantic import ConfigDict

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from agents.research_agent import research_agent
from agents.content_agent import content_agent


class NewsletterPipelineAgent(BaseAgent):
    """
    Custom orchestrator: runs research_agent first, checks whether it
    produced usable output, and only then runs content_agent. If research
    comes back empty, the pipeline stops early with a clear message instead
    of handing an empty string to the content agent.
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

        research_text = ""

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

        async for event in self.content_agent.run_async(ctx):
            yield event


root_agent = NewsletterPipelineAgent(
    name="newsletter_pipeline",
    research_agent=research_agent,
    content_agent=content_agent,
)