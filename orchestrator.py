"""
orchestrator.py
---------------
Agent Framework workflow that routes user prompts to the right specialists.

Routing
───────
  The OrchestratorDirectAgent acts as a router. It is given tools that wrap
  the other specialist agents (AF, Niceify, Weather, Dad Joke). The LLM
  natively decides which tool to call based on the user's prompt.

Workflow Shape
──────────────
  WorkflowBuilder
    └─ OrchestratorExecutor
         └─ OrchestratorDirectAgent
              ├─ calls foundry_tools.call_af
              ├─ calls foundry_tools.call_niceify
              ├─ calls orchestrator.call_weather_agent
              └─ calls orchestrator.call_dad_joke_agent
"""

import logging
import os
import json
import yaml
import importlib
import functools
from contextlib import asynccontextmanager, AsyncExitStack
from typing import AsyncIterator
from uuid import uuid4

from agent_framework import (
    ChatMessage,
    Role,
    ai_function,
)
from agent_framework.azure import AzureAIClient, AzureOpenAIChatClient
from azure.ai.projects.aio import AIProjectClient as AsyncAIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.policies import AzureKeyCredentialPolicy
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential
from typing_extensions import Never

from foundry_tools import call_af, call_niceify

logger = logging.getLogger(__name__)

_global_agents = {}

@ai_function
async def delegate_to_agent(agent_name: str, query: str) -> str:
    """
    Delegate a query to a specialist agent.
    Use this tool to route the user's request to the appropriate agent.
    Provide the exact agent_name (e.g., 'WeatherAgent', 'DadJokeAgent', 'AF', 'Niceify').
    """
    # 1. Check local agents
    if agent := _global_agents.get(agent_name):
        response = await agent.run([ChatMessage(role=Role.USER, text=query)])
        return response.text
        
    # 2. Check Foundry agents
    from foundry_tools import invoke_foundry_agent
    try:
        # If it's a Foundry agent, we might need to resolve its env var name or just pass the display name.
        # invoke_foundry_agent expects the display name (e.g., "AF" or "Niceify").
        # We'll check if there's an env var override, otherwise use the name directly.
        env_var_map = {"AF": "AF_AGENT_NAME", "Niceify": "NICEIFY_AGENT_NAME"}
        actual_name = os.environ.get(env_var_map.get(agent_name, ""), agent_name)
        return await invoke_foundry_agent(actual_name, query)
    except Exception as e:
        return f"Agent '{agent_name}' not available. Error: {e}"

# ---------------------------------------------------------------------------
# Workflow factory — async context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _build_client(provider: str):
    """Factory to build the appropriate client based on the provider."""
    use_key_auth = os.environ.get("FOUNDRY_USE_KEY_AUTH", "false").lower() == "true"

    if provider == "foundry":
        endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
        model = os.environ.get("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
        if use_key_auth:
            key_cred = AzureKeyCredential(os.environ["FOUNDRY_API_KEY"])
            yield AzureAIClient(
                project_client=AsyncAIProjectClient(
                    endpoint=endpoint,
                    credential=key_cred,
                    authentication_policy=AzureKeyCredentialPolicy(key_cred, "api-key"),
                ),
                model_deployment_name=model,
            )
        else:
            async with DefaultAzureCredential() as credential:
                yield AzureAIClient(
                    project_endpoint=endpoint,
                    model_deployment_name=model,
                    credential=credential,
                )

    elif provider == "azure_openai":
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
        if use_key_auth:
            yield AzureOpenAIChatClient(
                endpoint=endpoint,
                deployment_name=deployment,
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
            )
        else:
            credential = SyncDefaultAzureCredential()
            try:
                yield AzureOpenAIChatClient(
                    endpoint=endpoint,
                    deployment_name=deployment,
                    credential=credential,
                )
            finally:
                credential.close()

    elif provider == "openai":
        from agent_framework.openai import OpenAIChatClient
        yield OpenAIChatClient(
            model_id=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
        )
    else:
        raise ValueError(f"Unknown MODEL_PROVIDER '{provider}'. Choose one of: foundry, azure_openai, openai")

@asynccontextmanager
async def build_orchestrator() -> "AsyncIterator[OrchestratorWorkflow]":
    """
    Async context manager that creates, yields, and cleans up the workflow.
    """
    provider = os.environ.get("MODEL_PROVIDER", "foundry").lower()
    
    async with _build_client(provider) as client:
        stack, agents = await _load_agents_dynamically(client, "agents.yaml")
        try:
            yield OrchestratorWorkflow(agents.get("OrchestratorDirectAgent"))
        finally:
            await stack.aclose()

async def _load_agents_dynamically(client, config_path="agents.yaml"):
    """
    Dynamically load agents from a YAML configuration file.
    Returns an AsyncExitStack and a dictionary of loaded agents.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    stack = AsyncExitStack()
    agents = {}
    
    for name, details in config.get("agents", {}).items():
        tools = []
        for tool_path in details.get("tools", []):
            module_name, func_name = tool_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            tools.append(getattr(module, func_name))
            
        agent_ctx = client.create_agent(
            name=name,
            instructions=details.get("instructions", ""),
            tools=tools if tools else None
        )
        
        agents[name] = await stack.enter_async_context(agent_ctx)
        _global_agents[name] = agents[name]
        
    return stack, agents

# ---------------------------------------------------------------------------
# OrchestratorWorkflow — public facade
# ---------------------------------------------------------------------------

class OrchestratorWorkflow:
    """
    Thin facade over the Agent Framework Workflow.
    """

    def __init__(self, direct_agent) -> None:
        self._direct_agent = direct_agent

    async def run(self, user_text: str, history: list[dict] | None = None) -> str:
        messages = _build_messages(user_text, history)
        response = await self._direct_agent.run(messages)
        return response.text

    async def run_stream(self, user_text: str, history: list[dict] | None = None) -> AsyncIterator[tuple[str, str]]:
        """
        Async generator that yields (event_type, text) tuples.
        """
        messages = _build_messages(user_text, history)
        
        # Yield a status event to mimic the old OrchestratorExecutor behavior
        yield ("status", "Thinking…")
        
        try:
            async for update in self._direct_agent.run_stream(messages):
                if update.text:
                    yield ("answer", update.text)
        except Exception as e:
            yield ("error", f"Execution failed: {str(e)}")


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_messages(user_text: str, history: list[dict] | None) -> list[ChatMessage]:
    """Convert history + new user text into ChatMessage list."""
    messages: list[ChatMessage] = []
    if history:
        for entry in history:
            role = Role.USER if entry["role"] == "user" else Role.ASSISTANT
            messages.append(ChatMessage(role=role, text=entry["content"]))
    messages.append(ChatMessage(role=Role.USER, text=user_text))
    return messages
