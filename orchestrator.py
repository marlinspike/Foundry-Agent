"""
orchestrator.py
---------------
Agent Framework workflow that routes user prompts to the right specialists.

Routing
───────
  The OrchestratorDirectAgent acts as a router. It is given tools that wrap
  the other specialist agents. The LLM natively decides which tool to call 
  based on the user's prompt.

Workflow Shape
──────────────
  WorkflowBuilder
    └─ OrchestratorExecutor
         └─ OrchestratorDirectAgent
              ├─ calls delegate_to_agent
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

logger = logging.getLogger(__name__)

_global_agents = {}
_foundry_agent_map = {}

@ai_function
async def delegate_to_agent(agent_name: str, query: str) -> str:
    """
    Delegate a query to a specialist agent.
    Use this tool to route the user's request to the appropriate agent.
    Provide the exact agent_name as listed in your instructions.
    """
    # 1. Check local agents
    if agent := _global_agents.get(agent_name):
        response = await agent.run([ChatMessage(role=Role.USER, text=query)])
        return response.text
        
    # 2. Check Foundry agents
    from foundry_tools import invoke_foundry_agent
    
    # Only attempt to call Foundry if it's explicitly mapped in our config
    if agent_name in _foundry_agent_map:
        try:
            env_var_name = _foundry_agent_map.get(agent_name)
            actual_name = os.environ.get(env_var_name, agent_name) if env_var_name else agent_name
            return await invoke_foundry_agent(actual_name, query)
        except Exception as e:
            return f"Agent '{agent_name}' not available. Error: {e}"
            
    # 3. If it's neither a local agent nor a mapped Foundry agent, return an error
    return f"Error: Agent '{agent_name}' is not a recognized specialist."

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
        stack, agents, orch_name = await _load_agents_dynamically(client, "agents.yaml")
        try:
            yield OrchestratorWorkflow(agents.get(orch_name))
        finally:
            await stack.aclose()

async def _load_agents_dynamically(client, config_path="agents.yaml"):
    """Dynamically load agents from a YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    stack = AsyncExitStack()
    agents = {}
    agent_list_lines = []
    
    async def _create(name: str, details: dict):
        tools = [
            getattr(importlib.import_module(m), f)
            for m, f in (t.rsplit(".", 1) for t in details.get("tools", []))
        ]
        agent_ctx = client.create_agent(
            name=name,
            instructions=details.get("instructions", ""),
            tools=tools or None
        )
        agents[name] = _global_agents[name] = await stack.enter_async_context(agent_ctx)
        
    for name, details in config.get("local_agents", {}).items():
        desc = details.get("description", "No description provided.")
        agent_list_lines.append(f"    • {name} - {desc}")
        await _create(name, details)
        
    # Load Foundry agent mappings
    for name, details in config.get("foundry_agents", {}).items():
        desc = details.get("description", "No description provided.")
        agent_list_lines.append(f"    • {name} - {desc}")
        if env_var := details.get("env_var"):
            _foundry_agent_map[name] = env_var
            
    agent_list_str = "\n".join(agent_list_lines)
        
    orch = config.get("orchestrator", {})
    orch_name = orch.get("name", "OrchestratorDirectAgent")
    
    # Inject AGENT_LIST into orchestrator instructions
    instructions = orch.get("instructions", "")
    if "{{AGENT_LIST}}" in instructions:
        orch["instructions"] = instructions.replace("{{AGENT_LIST}}", agent_list_str)
        
    await _create(orch_name, orch)
        
    return stack, agents, orch_name

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
