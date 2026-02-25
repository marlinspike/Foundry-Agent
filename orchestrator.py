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
    AgentRunResponseUpdate,
    AgentRunUpdateEvent,
    ChatMessage,
    Executor,
    Role,
    TextContent,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    handler,
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

def local_agent_tool(agent_name: str):
    """Decorator to create a tool that invokes a dynamically loaded local agent."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(query: str) -> str:
            agent = _global_agents.get(agent_name)
            if agent:
                response = await agent.run([ChatMessage(role=Role.USER, text=query)])
                return response.text
            return f"{agent_name} not available."
        return ai_function(wrapper)
    return decorator

@local_agent_tool("WeatherAgent")
async def call_weather_agent(query: str) -> str:
    """Use this tool when the user asks about the weather."""
    pass

@local_agent_tool("DadJokeAgent")
async def call_dad_joke_agent(query: str) -> str:
    """Use this tool when the user asks for a joke, pun, or something funny."""
    pass

# ---------------------------------------------------------------------------
# Orchestrator Executor
# ---------------------------------------------------------------------------

class OrchestratorExecutor(Executor):
    """
    Single-node workflow executor that handles routing and execution.

    Parameters
    ----------
    agents:         A dictionary of loaded agents.
    """

    def __init__(self, agents: dict, id: str = "Orchestrator") -> None:
        self._agents = agents
        self._direct_agent = agents.get("OrchestratorDirectAgent")
        super().__init__(id=id)

    # ── main handler ─────────────────────────────────────────────────────────

    @handler
    async def handle(self, messages: list[ChatMessage], ctx: WorkflowContext[Never, str]) -> None:
        """Receive the conversation, route it, and yield the final answer."""

        await self._emit_status(ctx, "Thinking…")
        response = await self._direct_agent.run(messages)
        final = response.text

        await ctx.yield_output(final)

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _emit_status(self, ctx: WorkflowContext, status_text: str) -> None:
        """Emit a lightweight status event so the CLI can show progress."""
        logger.debug("[Orchestrator status] %s", status_text)
        await ctx.add_event(
            AgentRunUpdateEvent(
                self.id,
                data=AgentRunResponseUpdate(
                    contents=[TextContent(text=f"⟳ {status_text}")],
                    role=Role.ASSISTANT,
                    response_id=str(uuid4()),
                ),
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def load_agents_dynamically(client, config_path="agents.yaml"):
    """
    Dynamically load agents from a YAML configuration file.
    Returns an AsyncExitStack and a dictionary of loaded agents.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    stack = AsyncExitStack()
    agents = {}
    
    for name, details in config.get("agents", {}).items():
        # 1. Dynamically resolve tool functions from strings (e.g., "local_agent.get_weather")
        tools = []
        for tool_path in details.get("tools", []):
            module_name, func_name = tool_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            tools.append(getattr(module, func_name))
            
        # 2. Create the agent context manager
        agent_ctx = client.create_agent(
            name=name,
            instructions=details.get("instructions", ""),
            tools=tools if tools else None
        )
        
        # 3. Enter the context and store the yielded agent
        agents[name] = await stack.enter_async_context(agent_ctx)
        _global_agents[name] = agents[name]
        
    return stack, agents


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
        stack, agents = await load_agents_dynamically(client, "agents.yaml")
        try:
            yield _make_workflow(agents)
        finally:
            await stack.aclose()


def _make_workflow(agents: dict) -> "OrchestratorWorkflow":
    """Build the AgentFramework workflow and wrap it in OrchestratorWorkflow."""
    workflow = (
        WorkflowBuilder()
        .register_executor(
            lambda: OrchestratorExecutor(agents),
            name="Orchestrator",
        )
        .set_start_executor("Orchestrator")
        .build()
    )
    return OrchestratorWorkflow(workflow)


# ---------------------------------------------------------------------------
# OrchestratorWorkflow — public facade
# ---------------------------------------------------------------------------

class OrchestratorWorkflow:
    """
    Thin facade over the Agent Framework Workflow.

    Methods
    -------
    run(user_text, history)   → str
        Process a user message and return the final answer.

    run_stream(user_text, history)  → AsyncIterator[str]
        Stream status events and then the final answer.
    """

    def __init__(self, workflow) -> None:
        self._workflow = workflow

    async def run(self, user_text: str, history: list[dict] | None = None) -> str:
        messages = _build_messages(user_text, history)
        events = await self._workflow.run(messages)
        outputs = events.get_outputs()
        return outputs[0] if outputs else "[No output]"

    async def run_stream(self, user_text: str, history: list[dict] | None = None) -> AsyncIterator[tuple[str, str]]:
        """
        Async generator that yields (event_type, text) tuples.

        event_type is one of:
            "status"  — progress indicator (e.g. "Calling AF agent…")
            "answer"  — the final answer
            "error"   — something went wrong
        """
        messages = _build_messages(user_text, history)
        from agent_framework import WorkflowOutputEvent, ExecutorFailedEvent, WorkflowFailedEvent

        async for event in self._workflow.run_stream(messages):
            if isinstance(event, AgentRunUpdateEvent):
                text = _extract_text(event.data)
                if text:
                    yield ("status", text)
            elif isinstance(event, WorkflowOutputEvent):
                yield ("answer", str(event.data))
            elif isinstance(event, (ExecutorFailedEvent, WorkflowFailedEvent)):
                details = event.details
                yield ("error", f"{details.error_type}: {details.message}")


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


def _extract_text(update: AgentRunResponseUpdate) -> str:
    for content in update.contents:
        if isinstance(content, TextContent):
            return content.text
    return ""
