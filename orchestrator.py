"""
orchestrator.py
---------------
Agent Framework workflow that routes user prompts to the right specialists.

Routing Mode  (set ROUTING_MODE in .env)
─────────────────────────────────────────
  ROUTING_MODE=keyword   (default)
      Fast, zero-cost intent detection using hard-coded keyword lists.
      Resilient to model failures; misses synonyms and paraphrases.
      See the "KEYWORD-BASED ROUTING" section below.

  ROUTING_MODE=model
      One extra LLM call per turn classifies intent from a plain-English
      description of each agent.  Handles synonyms, context, and novel
      phrasings automatically.  No keyword lists to maintain.
      See the "MODEL-BASED ROUTING" section below.

Routes (both modes produce the same four outcomes)
────────────────────────────────────────────────────
  aircraft  → call_af  [→ call_niceify if auto_niceify + negative]
  niceify   → call_niceify
  both      → call_af then call_niceify
  direct    → answer directly via ChatAgent

Workflow Shape
──────────────
  WorkflowBuilder
    └─ OrchestratorExecutor   (single node, handles all routing internally)
         ├─ calls foundry_tools.call_af
         ├─ calls foundry_tools.call_niceify
         └─ calls direct ChatAgent for general answers
"""

import logging
import os
from contextlib import asynccontextmanager
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

# ---------------------------------------------------------------------------
# Orchestrator system prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_INSTRUCTIONS = """\
You are an intelligent assistant and orchestrator for a multi-agent system.

You have access to two specialist agents whose results are provided to you:

• AF Agent   – Air Force aircraft specialist (F-22, F-35, B-2, etc.)
• Niceify    – Transforms negative or sad content into a positive reframing

When answering questions directly (no specialist needed), be clear and concise.
When you receive specialist agent results, synthesise them into a final reply
for the user—do not just echo the raw output verbatim.
"""

# ===========================================================================
# ROUTING: KEYWORD-BASED  (ROUTING_MODE=keyword)
# ===========================================================================
# Used when ROUTING_MODE is unset or set to "keyword".
#
# Pro:  Zero latency, no extra LLM call, deterministic.
# Con:  Must manually add new keywords; misses synonyms and paraphrases
#       (e.g. "Raptor" for F-22, "Warthog" for A-10 won't match).
#
# To extend routing coverage, add keywords to the frozensets below.
# ---------------------------------------------------------------------------

_AIRCRAFT_KEYWORDS = frozenset([
    "f-22", "f-35", "f-15", "f-16", "f/a-18", "fa-18", "f-117",
    "b-2", "b-21", "b-52", "b-1",
    "sr-71", "u-2", "a-10", "c-17", "c-130", "kc-135", "kc-46",
    "fighter", "bomber", "stealth", "jet aircraft", "military aircraft",
    "air force plane", "usaf", "mach ", "supersonic", "hypersonic",
    "dogfight", "radar cross", "thrust vector", "airspeed", "altitude ceiling",
    "avionics", "payload capacity", "wingspan", "combat radius",
    "what plane", "which plane", "tell me about the", "compare the",
])

_NICEIFY_KEYWORDS = frozenset([
    "make it positive", "silver lining", "reframe", "be positive",
    "positive spin", "cheer me up", "look on the bright side",
    "make this better", "put a positive", "optimistic take",
    "good news version",
])

_WEATHER_KEYWORDS = frozenset([
    "weather", "temperature", "forecast", "rain", "sunny", "cloudy",
    "how hot", "how cold", "is it raining"
])

# Used by auto_niceify post-processing (both routing modes).
_NEGATIVE_KEYWORDS = frozenset([
    "crash", "fail", "disaster", "accident", "tragic", "fatal",
    "unfortunate", "horrible", "terrible", "awful", "catastrophic",
    "grounded", "shot down",
])


def _keyword_route(text: str) -> str:
    """
    Classify intent using keyword matching.

    Returns one of: "aircraft", "niceify", "both", "weather", "direct".
    """
    lower = text.lower()
    aircraft = sum(1 for kw in _AIRCRAFT_KEYWORDS if kw in lower) > 0
    niceify = any(kw in lower for kw in _NICEIFY_KEYWORDS)
    weather = any(kw in lower for kw in _WEATHER_KEYWORDS)

    if aircraft and niceify:
        return "both"
    if aircraft:
        return "aircraft"
    if niceify:
        return "niceify"
    if weather:
        return "weather"
    return "direct"


def _seems_negative(text: str) -> bool:
    """Used by auto_niceify to detect negative sentiment in AF responses."""
    lower = text.lower()
    return sum(1 for kw in _NEGATIVE_KEYWORDS if kw in lower) >= 2


# ===========================================================================
# ROUTING: MODEL-BASED  (ROUTING_MODE=model)
# ===========================================================================
# Used when ROUTING_MODE=model.
#
# Pro:  Handles synonyms, context, and novel phrasings automatically.
#       No keyword lists to maintain; just update the descriptions below.
# Con:  One extra LLM call per turn adds latency and token cost.
#
# To change how the model routes, edit _ROUTER_PROMPT below — specifically
# the plain-English descriptions of each route.
# ---------------------------------------------------------------------------

_ROUTER_PROMPT = """\
You are a routing classifier for a multi-agent assistant.
Given the user message below, return ONLY a JSON object with a single key "route".

Choose exactly one value:
  "aircraft" — the user is asking about US military aircraft: specifications,
               performance, history, comparisons, or named aircraft/designations
               (e.g. F-22, B-2, A-10, Raptor, Warthog, stealth bomber, etc.).
  "niceify"  — the user explicitly wants negative, sad, or discouraging content
               reframed in a positive or uplifting way.
  "both"     — the user wants aircraft information AND a positive/uplifting spin
               on the answer.
  "weather"  — the user is asking about the weather, temperature, or forecast.
  "direct"   — everything else (general knowledge, greetings, unrelated topics).

Respond with ONLY valid JSON, e.g.: {{"route": "aircraft"}}

User message: {user_text}
"""


# ---------------------------------------------------------------------------
# Orchestrator Executor
# ---------------------------------------------------------------------------

class OrchestratorExecutor(Executor):
    """
    Single-node workflow executor that handles routing and execution.

    Parameters
    ----------
    direct_agent:   A ChatAgent used for general (non-specialist) questions.
    weather_agent:  A local ChatAgent used for weather questions.
    auto_niceify:   If True, AF responses with negative sentiment are
                    automatically piped through the Niceify agent.
    """

    def __init__(self, direct_agent, weather_agent, auto_niceify: bool = False, id: str = "Orchestrator") -> None:
        self._direct_agent = direct_agent
        self._weather_agent = weather_agent
        self.auto_niceify = auto_niceify
        # Read ROUTING_MODE once at construction time.
        # Set ROUTING_MODE=model in .env to enable model-based routing.
        # Any other value (or unset) falls back to keyword-based routing.
        raw_mode = os.environ.get("ROUTING_MODE", "keyword").lower().strip()
        if raw_mode not in ("keyword", "model"):
            logger.warning(
                "Unknown ROUTING_MODE=%r — falling back to 'keyword'.", raw_mode
            )
            raw_mode = "keyword"
        self._routing_mode = raw_mode
        logger.debug("[Orchestrator] routing_mode=%s", self._routing_mode)
        super().__init__(id=id)

    # ── main handler ─────────────────────────────────────────────────────────

    @handler
    async def handle(self, messages: list[ChatMessage], ctx: WorkflowContext[Never, str]) -> None:
        """Receive the conversation, route it, and yield the final answer."""

        # Extract the last user message for intent detection.
        user_text = _last_user_text(messages)

        # ── routing decision ─────────────────────────────────────────────────
        # Dispatch to whichever routing strategy was selected at startup.
        # Both strategies return the same four canonical strings:
        #   "aircraft" | "niceify" | "both" | "direct"
        if self._routing_mode == "model":
            await self._emit_status(ctx, "Classifying intent…")
            route = await self._model_route(user_text)
        else:
            route = _keyword_route(user_text)

        logger.debug("[Orchestrator] route=%s (mode=%s)", route, self._routing_mode)

        # ── execute the chosen route ──────────────────────────────────────────
        if route == "both":
            # User wants aircraft info AND a positive spin.
            await self._emit_status(ctx, "Calling AF agent…")
            af_result = await call_af(user_text)
            await self._emit_status(ctx, "Running through Niceify agent…")
            final = await call_niceify(af_result)

        elif route == "aircraft":
            # Pure aircraft query.
            await self._emit_status(ctx, "Calling AF agent…")
            af_result = await call_af(user_text)
            if self.auto_niceify and _seems_negative(af_result):
                await self._emit_status(ctx, "Auto-niceifying negative content…")
                final = await call_niceify(af_result)
            else:
                final = af_result

        elif route == "niceify":
            # User explicitly wants positive reframing.
            await self._emit_status(ctx, "Calling Niceify agent…")
            final = await call_niceify(user_text)

        elif route == "weather":
            # User wants weather info.
            await self._emit_status(ctx, "Calling local Weather agent…")
            final = await self._weather_answer(messages)

        else:  # "direct"
            # General question — answer directly with the model.
            await self._emit_status(ctx, "Thinking…")
            final = await self._direct_answer(messages)

        await ctx.yield_output(final)

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _direct_answer(self, messages: list[ChatMessage]) -> str:
        """Run the conversation through the direct ChatAgent."""
        response = await self._direct_agent.run(messages)
        return response.text

    async def _weather_answer(self, messages: list[ChatMessage]) -> str:
        """Run the conversation through the local WeatherAgent."""
        response = await self._weather_agent.run(messages)
        return response.text

    # ── model-based routing (ROUTING_MODE=model) ──────────────────────────────

    async def _model_route(self, user_text: str) -> str:
        """
        Ask the model to classify the user's intent.

        Returns one of: "aircraft", "niceify", "both", "weather", "direct".
        Falls back to "direct" if the response cannot be parsed.
        """
        import json

        prompt = _ROUTER_PROMPT.format(user_text=user_text)
        routing_messages = [ChatMessage(role=Role.USER, text=prompt)]
        try:
            response = await self._direct_agent.run(routing_messages)
            # Strip markdown fences the model may add (```json ... ```)
            raw = response.text.strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            data = json.loads(raw)
            route = data.get("route", "direct").lower().strip()
            if route not in ("aircraft", "niceify", "both", "weather", "direct"):
                logger.warning(
                    "[Orchestrator] model returned unknown route %r — using 'direct'",
                    route,
                )
                return "direct"
            return route
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Orchestrator] model routing failed (%s) — falling back to 'direct'",
                exc,
            )
            return "direct"

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

def _last_user_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == Role.USER:
            if msg.contents:
                return msg.contents[-1].text
    return ""


# ---------------------------------------------------------------------------
# Workflow factory — async context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def build_orchestrator(auto_niceify: bool = False) -> "AsyncIterator[OrchestratorWorkflow]":
    """
    Async context manager that creates, yields, and cleans up the workflow.

    Usage
    -----
    async with build_orchestrator(auto_niceify=False) as orch:
        result = await orch.run(user_text)

    The workflow uses one of three model providers controlled by MODEL_PROVIDER
    in .env:
        foundry       (default) — AzureAIClient with Foundry project endpoint
        azure_openai             — AzureOpenAIChatClient with AOAI endpoint
        openai                   — direct OpenAI API via openai.AsyncOpenAI
    """
    provider = os.environ.get("MODEL_PROVIDER", "foundry").lower()

    if provider == "foundry":
        async with _build_foundry_orchestrator(auto_niceify) as orch:
            yield orch

    elif provider == "azure_openai":
        async with _build_azure_openai_orchestrator(auto_niceify) as orch:
            yield orch

    elif provider == "openai":
        async with _build_openai_orchestrator(auto_niceify) as orch:
            yield orch

    else:
        raise ValueError(
            f"Unknown MODEL_PROVIDER '{provider}'. "
            "Choose one of: foundry, azure_openai, openai"
        )


# ── provider-specific factories ───────────────────────────────────────────────

@asynccontextmanager
async def _build_foundry_orchestrator(auto_niceify: bool):
    from local_agent import LOCAL_AGENT_NAME, LOCAL_AGENT_INSTRUCTIONS, LOCAL_AGENT_TOOLS
    endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
    model = os.environ.get("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
    use_key_auth = os.environ.get("FOUNDRY_USE_KEY_AUTH", "false").lower() == "true"

    if use_key_auth:
        api_key = os.environ["FOUNDRY_API_KEY"]
        key_cred = AzureKeyCredential(api_key)
        auth_policy = AzureKeyCredentialPolicy(key_cred, "api-key")
        project_client = AsyncAIProjectClient(
            endpoint=endpoint,
            credential=key_cred,  # type: ignore[arg-type] — non-None check; auth_policy takes over
            authentication_policy=auth_policy,
        )
        async with (
            AzureAIClient(
                project_client=project_client,
                model_deployment_name=model,
            ).create_agent(
                name="OrchestratorDirectAgent",
                instructions=ORCHESTRATOR_INSTRUCTIONS,
            ) as direct_agent,
            AzureAIClient(
                project_client=project_client,
                model_deployment_name=model,
            ).create_agent(
                name=LOCAL_AGENT_NAME,
                instructions=LOCAL_AGENT_INSTRUCTIONS,
                tools=LOCAL_AGENT_TOOLS,
            ) as weather_agent
        ):
            yield _make_workflow(direct_agent, weather_agent, auto_niceify)
    else:
        async with DefaultAzureCredential() as credential:
            async with (
                AzureAIClient(
                    project_endpoint=endpoint,
                    model_deployment_name=model,
                    credential=credential,
                ).create_agent(
                    name="OrchestratorDirectAgent",
                    instructions=ORCHESTRATOR_INSTRUCTIONS,
                ) as direct_agent,
                AzureAIClient(
                    project_endpoint=endpoint,
                    model_deployment_name=model,
                    credential=credential,
                ).create_agent(
                    name=LOCAL_AGENT_NAME,
                    instructions=LOCAL_AGENT_INSTRUCTIONS,
                    tools=LOCAL_AGENT_TOOLS,
                ) as weather_agent
            ):
                yield _make_workflow(direct_agent, weather_agent, auto_niceify)


@asynccontextmanager
async def _build_azure_openai_orchestrator(auto_niceify: bool):
    from local_agent import LOCAL_AGENT_NAME, LOCAL_AGENT_INSTRUCTIONS, LOCAL_AGENT_TOOLS
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
    use_key_auth = os.environ.get("FOUNDRY_USE_KEY_AUTH", "false").lower() == "true"

    if use_key_auth:
        # Pass api_key directly — AzureOpenAIChatClient uses it without any credential.
        api_key = os.environ["AZURE_OPENAI_API_KEY"]
        direct_agent = (
            AzureOpenAIChatClient(
                endpoint=endpoint,
                deployment_name=deployment,
                api_key=api_key,
            ).create_agent(
                name="OrchestratorDirectAgent",
                instructions=ORCHESTRATOR_INSTRUCTIONS,
            )
        )
        weather_agent = (
            AzureOpenAIChatClient(
                endpoint=endpoint,
                deployment_name=deployment,
                api_key=api_key,
            ).create_agent(
                name=LOCAL_AGENT_NAME,
                instructions=LOCAL_AGENT_INSTRUCTIONS,
                tools=LOCAL_AGENT_TOOLS,
            )
        )
        yield _make_workflow(direct_agent, weather_agent, auto_niceify)
    else:
        # AzureOpenAIChatClient calls get_entra_auth_token synchronously,
        # so it requires a sync TokenCredential (azure.identity, not .aio).
        credential = SyncDefaultAzureCredential()
        direct_agent = (
            AzureOpenAIChatClient(
                endpoint=endpoint,
                deployment_name=deployment,
                credential=credential,
            ).create_agent(
                name="OrchestratorDirectAgent",
                instructions=ORCHESTRATOR_INSTRUCTIONS,
            )
        )
        weather_agent = (
            AzureOpenAIChatClient(
                endpoint=endpoint,
                deployment_name=deployment,
                credential=credential,
            ).create_agent(
                name=LOCAL_AGENT_NAME,
                instructions=LOCAL_AGENT_INSTRUCTIONS,
                tools=LOCAL_AGENT_TOOLS,
            )
        )
        try:
            yield _make_workflow(direct_agent, weather_agent, auto_niceify)
        finally:
            credential.close()


@asynccontextmanager
async def _build_openai_orchestrator(auto_niceify: bool):
    from agent_framework.openai import OpenAIChatClient
    from local_agent import LOCAL_AGENT_NAME, LOCAL_AGENT_INSTRUCTIONS, LOCAL_AGENT_TOOLS

    model_name = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    api_key = os.environ["OPENAI_API_KEY"]

    client = OpenAIChatClient(
        model_id=model_name,
        api_key=api_key,
    )
    
    direct_agent = client.create_agent(
        name="OrchestratorDirectAgent",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
    )
    
    weather_agent = client.create_agent(
        name=LOCAL_AGENT_NAME,
        instructions=LOCAL_AGENT_INSTRUCTIONS,
        tools=LOCAL_AGENT_TOOLS,
    )
    
    yield _make_workflow(direct_agent, weather_agent, auto_niceify)


def _make_workflow(direct_agent, weather_agent, auto_niceify: bool) -> "OrchestratorWorkflow":
    """Build the AgentFramework workflow and wrap it in OrchestratorWorkflow."""
    workflow = (
        WorkflowBuilder()
        .register_executor(
            lambda: OrchestratorExecutor(direct_agent, weather_agent, auto_niceify),
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
