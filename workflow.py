from agent import NewsletterPipelineAgent
from agents.research_agent import research_agent
from agents.content_agent import content_agent

newsletter_workflow = NewsletterPipelineAgent(
    name="newsletter_workflow",
    research_agent=research_agent,
    content_agent=content_agent,
)