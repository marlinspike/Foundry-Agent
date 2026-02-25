"""
foundry_tools.py
----------------
Thin async wrappers that invoke *existing* Azure AI Foundry agents
(AF and Niceify) via the azure-ai-agents SDK.

These functions are called by the OrchestratorExecutor in orchestrator.py
to delegate specialised work to your cloud-hosted agents.

SDK note
--------
The v2 azure-ai-foundry stack separates the agents runtime into its own
package: azure-ai-agents (not azure-ai-projects).
The client is azure.ai.agents.aio.AgentsClient, endpoint is the
Foundry project endpoint.
"""

from __future__ import annotations

import os
import logging
import functools

from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import (
    AgentThreadCreationOptions,
    MessageRole,
    ThreadMessageOptions,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.policies import AzureKeyCredentialPolicy
from azure.identity.aio import DefaultAzureCredential

logger = logging.getLogger(__name__)

from agent_framework import ai_function

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: AgentsClient | None = None
_credential: DefaultAzureCredential | None = None


async def _get_client() -> AgentsClient:
    """Return (or lazily create) the shared AgentsClient."""
    global _client, _credential
    if _client is None:
        endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
        if os.environ.get("FOUNDRY_USE_KEY_AUTH", "false").lower() == "true":
            key_cred = AzureKeyCredential(os.environ["FOUNDRY_API_KEY"])
            _client = AgentsClient(
                endpoint=endpoint,
                credential=key_cred,
                authentication_policy=AzureKeyCredentialPolicy(key_cred, "api-key"),
            )
        else:
            _credential = DefaultAzureCredential()
            _client = AgentsClient(endpoint=endpoint, credential=_credential)
    return _client


async def close_client() -> None:
    """Close the shared client and credential.  Call this on shutdown."""
    global _client, _credential
    if _client:
        await _client.close()
        _client = None
    if _credential:
        await _credential.close()
        _credential = None


# ---------------------------------------------------------------------------
# Agent-ID resolution (cached in-process)
# ---------------------------------------------------------------------------

_agent_id_cache: dict[str, str] = {}


async def _resolve_agent_id(client: AgentsClient, agent_name: str) -> str:
    """
    Find the ID of an agent by its display name.
    Results are cached for the lifetime of the process to avoid repeated
    list-agents calls.
    """
    if agent_name in _agent_id_cache:
        return _agent_id_cache[agent_name]

    # list_agents() returns AsyncItemPaged[Agent] — iterate with async for.
    async for agent in client.list_agents():
        if agent.name == agent_name:
            _agent_id_cache[agent_name] = agent.id
            logger.debug("Resolved agent '%s' -> id=%s", agent_name, agent.id)
            return agent.id

    raise ValueError(
        f"Agent '{agent_name}' was not found in your Foundry project. "
        "Check the agent name in .env and make sure it is deployed."
    )


# ---------------------------------------------------------------------------
# Core invocation helper
# ---------------------------------------------------------------------------

async def invoke_foundry_agent(agent_name: str, user_input: str) -> str:
    """
    Run an existing Foundry agent with a single-turn user message and return
    the assistant's text reply.

    Parameters
    ----------
    agent_name:  Display name of the agent as it appears in Azure AI Foundry.
    user_input:  The message to send to the agent.

    Returns
    -------
    The agent's text reply, or an error string if something goes wrong.
    """
    try:
        client = await _get_client()
        agent_id = await _resolve_agent_id(client, agent_name)

        # create_thread_and_process_run creates a thread, posts the message,
        # runs the agent, and blocks until the run reaches a terminal state.
        run = await client.create_thread_and_process_run(
            agent_id=agent_id,
            thread=AgentThreadCreationOptions(
                messages=[
                    ThreadMessageOptions(
                        role=MessageRole.USER,
                        content=user_input,
                    )
                ]
            ),
        )

        if run.status == "failed":
            err = getattr(run, "last_error", None)
            err_msg = err.message if err else "unknown error"
            logger.error("Agent '%s' run failed: %s", agent_name, err_msg)
            return f"[{agent_name} agent error: {err_msg}]"

        # Convenience method returns the last assistant text message directly.
        text_msg = await client.messages.get_last_message_text_by_role(
            thread_id=run.thread_id,
            role=MessageRole.AGENT,
        )
        if text_msg:
            return text_msg.text.value

        return f"[{agent_name} returned no text response]"

    except ValueError as exc:
        logger.error(str(exc))
        return f"[Configuration error: {exc}]"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error calling agent '%s'", agent_name)
        return f"[Unexpected error calling {agent_name}: {exc}]"


# ---------------------------------------------------------------------------
# Named convenience wrappers (used by the orchestrator as "tools")
# ---------------------------------------------------------------------------

def foundry_tool(env_var: str, default_name: str):
    """Decorator to create a tool that invokes a Foundry agent."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(text: str) -> str:
            agent_name = os.environ.get(env_var, default_name)
            logger.info("[Tool] %s -> '%s'", func.__name__, text[:80])
            result = await invoke_foundry_agent(agent_name, text)
            logger.info("[Tool] %s result (%d chars)", func.__name__, len(result))
            return result
        return ai_function(wrapper)
    return decorator

@foundry_tool("AF_AGENT_NAME", "AF")
async def call_af(question: str) -> str:
    """
    Ask the AF specialist agent about Air Force planes.

    Use this for any question about specific aircraft (F-22, B-2, F-15 …),
    specifications, capabilities, history, or comparisons.
    """
    pass

@foundry_tool("NICEIFY_AGENT_NAME", "Niceify")
async def call_niceify(text: str) -> str:
    """
    Pass text through the Niceify agent to give it a positive spin.

    Use this when the user explicitly asks for positive reframing, or when
    a previous agent response contains notably negative/sad content.
    """
    pass
