"""
agent/orchestrator.py
---------------
Agent Framework workflow that routes user prompts to the right specialists.

Routing
───────
  The OrchestratorDirectAgent acts as a router. It is given tools that wrap
  the other specialist agents. The LLM natively decides which tool to call
  based on the user's prompt.

Tools exposed to the orchestrator
──────────────────────────────────
  • delegate_to_agent               – single-agent routing
  • delegate_to_multiple_agents     – parallel fan-out to N agents

Workflow Shape
──────────────
  WorkflowBuilder
    └─ OrchestratorExecutor
         └─ OrchestratorDirectAgent
              ├─ calls delegate_to_agent
              └─ calls delegate_to_multiple_agents
"""

import asyncio
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
from agent.pipeline import Pipeline, load_pipelines
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
_pipeline_registry: dict[str, Pipeline] = {}

async def _invoke_agent(agent_name: str, query: str) -> str:
    """Core coroutine: invoke a single named agent with a query. Shared by both delegation tools."""
    # 1. Check local agents
    if agent := _global_agents.get(agent_name):
        response = await agent.run([ChatMessage(role=Role.USER, text=query)])
        return response.text

    # 2. Check Foundry agents (only if explicitly mapped in config)
    from agent.foundry_tools import invoke_foundry_agent

    if agent_name in _foundry_agent_map:
        try:
            env_var_name = _foundry_agent_map.get(agent_name)
            actual_name = os.environ.get(env_var_name, agent_name) if env_var_name else agent_name
            return await invoke_foundry_agent(actual_name, query)
        except Exception as e:
            return f"Agent '{agent_name}' not available. Error: {e}"

    # 3. Neither local nor Foundry
    return f"Error: Agent '{agent_name}' is not a recognized specialist."


@ai_function
async def delegate_to_agent(agent_name: str, query: str) -> str:
    """
    Delegate a query to a single specialist agent.
    Use this tool when the user's request maps clearly to one specialist.
    Provide the exact agent_name as listed in your instructions.
    """
    return await _invoke_agent(agent_name, query)


@ai_function
async def run_pipeline(pipeline_name: str, initial_input: str) -> str:
    """
    Execute a named agent pipeline defined in agents.yaml.

    The pipeline runs its steps in order; the output of each step is fed as
    input to the next.  Use this when the user's request maps to a named
    pipeline rather than a single specialist agent.

    Provide the exact pipeline_name as listed in your instructions.
    """
    pipeline = _pipeline_registry.get(pipeline_name)
    if not pipeline:
        available = ", ".join(_pipeline_registry.keys()) or "none"
        return (
            f"Error: Pipeline '{pipeline_name}' not found. "
            f"Available pipelines: {available}"
        )

    if not pipeline.steps:
        return f"Error: Pipeline '{pipeline_name}' has no steps configured."

    current = initial_input
    for i, step in enumerate(pipeline.steps):
        logger.info(
            "[Pipeline %s] step %d/%d -> %s",
            pipeline_name, i + 1, len(pipeline.steps), step.agent,
        )
        current = await _invoke_agent(step.agent, current)
        # Abort early if a step signals an error so the caller is informed
        # exactly which step in the chain failed.
        if current.startswith("Error:"):
            return (
                f"Pipeline '{pipeline_name}' failed at step {i + 1} "
                f"({step.agent}): {current}"
            )

    return current


@ai_function
async def delegate_to_multiple_agents(agent_queries: list[dict]) -> str:
    """
    Delegate queries to multiple specialist agents IN PARALLEL and return their combined results.

    Use this tool when the user's request spans several domains and can be answered
    faster or more completely by consulting multiple agents simultaneously.

    agent_queries must be a JSON array of objects, each containing:
      - "agent_name" (string): exact agent name as listed in your instructions
      - "query"      (string): the specific question or task for that agent

    Example input:
      [{"agent_name": "WeatherAgent", "query": "What is the weather in Seattle?"},
       {"agent_name": "DadJokeAgent",  "query": "Tell me a weather-related dad joke"}]

    Returns a single string with each agent's response clearly labeled.
    Synthesise all responses into a coherent final answer for the user.
    """
    if not agent_queries:
        return "Error: agent_queries list is empty."

    missing = [
        i for i, item in enumerate(agent_queries)
        if "agent_name" not in item or "query" not in item
    ]
    if missing:
        return f"Error: items at index {missing} are missing 'agent_name' or 'query'."

    tasks = [_invoke_agent(item["agent_name"], item["query"]) for item in agent_queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    parts = []
    for item, result in zip(agent_queries, results):
        name = item["agent_name"]
        if isinstance(result, Exception):
            parts.append(f"[{name}]: Error — {result}")
        else:
            parts.append(f"[{name}]: {result}")

    return "\n\n".join(parts)

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
    agent_section_lines: list[str] = []
    pipeline_section_lines: list[str] = []

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
        agent_section_lines.append(f"    • {name} - {desc}")
        await _create(name, details)

    # Load Foundry agent mappings
    for name, details in config.get("foundry_agents", {}).items():
        desc = details.get("description", "No description provided.")
        agent_section_lines.append(f"    • {name} - {desc}")
        if env_var := details.get("env_var"):
            _foundry_agent_map[name] = env_var

    # Load declarative pipelines
    pipelines = load_pipelines(config)
    _pipeline_registry.clear()
    _pipeline_registry.update(pipelines)
    for name, pipeline in pipelines.items():
        pipeline_section_lines.append(
            f"    • {name} - {pipeline.description}  [Steps: {pipeline.step_summary}]"
        )

    # Build the combined AGENT_LIST block injected into the orchestrator prompt
    agent_list_parts = [
        "Specialist Agents (use `delegate_to_agent` or `delegate_to_multiple_agents`):",
        *agent_section_lines,
    ]
    if pipeline_section_lines:
        agent_list_parts += [
            "",
            "Named Pipelines (ordered agent chains — use `run_pipeline`):",
            *pipeline_section_lines,
        ]
    agent_list_str = "\n".join(agent_list_parts)
        
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
