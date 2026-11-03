import os
from dotenv import load_dotenv
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.query_agent import create_query_agent
from agents.knowledge_agent import create_knowledge_agent
from agents.resolution_agent import create_resolution_agent
from agents.log_agent import create_log_agent

load_dotenv()


def get_model_client(with_tools: bool = False) -> OpenAIChatCompletionClient:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing! Please configure it in your .env file.")

    kwargs = {
        "model": model_name,
        "api_key": api_key,
        "temperature": 0.1,
    }
    # Only set parallel_tool_calls on clients bound to agents that actually have tools
    if with_tools:
        kwargs["parallel_tool_calls"] = False

    if base_url:
        kwargs["base_url"] = base_url

    return OpenAIChatCompletionClient(**kwargs)


def build_support_team() -> RoundRobinGroupChat:
    """Creates the 4-agent sequential customer support team: Query -> Knowledge -> Resolution -> Log."""
    # Standard client for non-tool agents (Query & Resolution)
    text_client = get_model_client(with_tools=False)
    # Tool client with parallel_tool_calls=False for tool-using agents (Knowledge & Log)
    tool_client = get_model_client(with_tools=True)

    query_agent = create_query_agent(text_client)
    knowledge_agent = create_knowledge_agent(tool_client)
    resolution_agent = create_resolution_agent(text_client)
    log_agent = create_log_agent(tool_client)

    # Stops immediately when log_ticket returns TICKET_LOGGED_COMPLETE
    termination = TextMentionTermination("TICKET_LOGGED_COMPLETE") | MaxMessageTermination(max_messages=8)

    team = RoundRobinGroupChat(
        participants=[query_agent, knowledge_agent, resolution_agent, log_agent],
        termination_condition=termination,
    )
    return team