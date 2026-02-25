"""
tests/test_pipeline.py
----------------------
Unit tests for the declarative pipeline engine (Todo #4).

Unit tests patch _global_agents and _pipeline_registry directly — no LLM or
network calls required.
"""

import asyncio
import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.orchestrator as orch_module
from agent.orchestrator import run_pipeline
from agent.pipeline import Pipeline, PipelineStep, load_pipelines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(text: str):
    """Return a mock agent whose .run() resolves to a response with .text == text."""
    response = MagicMock()
    response.text = text
    agent = AsyncMock()
    agent.run = AsyncMock(return_value=response)
    return agent


def _call_tool(tool_fn, *args, **kwargs):
    """
    Transparently call an @ai_function-decorated function.
    Falls back to .func if the decorator wraps the coroutine.
    """
    fn = getattr(tool_fn, "func", tool_fn)
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Unit: load_pipelines
# ---------------------------------------------------------------------------


def test_load_pipelines_basic():
    """load_pipelines correctly parses a minimal pipelines config."""
    config = {
        "pipelines": {
            "NiceJoke": {
                "description": "Dad joke then niceified",
                "steps": [
                    {"agent": "DadJokeAgent"},
                    {"agent": "Niceify"},
                ],
            }
        }
    }
    pipelines = load_pipelines(config)

    assert "NiceJoke" in pipelines
    p = pipelines["NiceJoke"]
    assert p.name == "NiceJoke"
    assert p.description == "Dad joke then niceified"
    assert len(p.steps) == 2
    assert p.steps[0].agent == "DadJokeAgent"
    assert p.steps[1].agent == "Niceify"


def test_load_pipelines_empty_section():
    """load_pipelines returns an empty dict when the key is absent."""
    assert load_pipelines({}) == {}
    assert load_pipelines({"pipelines": {}}) == {}


def test_load_pipelines_multiple():
    """load_pipelines handles multiple pipelines independently."""
    config = {
        "pipelines": {
            "PipeA": {"description": "A", "steps": [{"agent": "AgentX"}]},
            "PipeB": {"description": "B", "steps": [{"agent": "AgentY"}, {"agent": "AgentZ"}]},
        }
    }
    pipelines = load_pipelines(config)
    assert len(pipelines) == 2
    assert pipelines["PipeA"].steps[0].agent == "AgentX"
    assert len(pipelines["PipeB"].steps) == 2


# ---------------------------------------------------------------------------
# Unit: Pipeline.step_summary
# ---------------------------------------------------------------------------


def test_step_summary_single():
    p = Pipeline(name="P", description="", steps=[PipelineStep(agent="A")])
    assert p.step_summary == "A"


def test_step_summary_chain():
    p = Pipeline(
        name="P",
        description="",
        steps=[PipelineStep(agent="A"), PipelineStep(agent="B"), PipelineStep(agent="C")],
    )
    assert p.step_summary == "A → B → C"


def test_step_summary_empty():
    p = Pipeline(name="P", description="", steps=[])
    assert p.step_summary == ""


# ---------------------------------------------------------------------------
# Unit: run_pipeline — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_two_steps():
    """Output of step 1 feeds as input to step 2; final output is returned."""
    # Step 1: DadJokeAgent receives the user input and returns a joke.
    # Step 2: Niceify receives the joke and returns a niceified version.
    mock_agents = {
        "DadJokeAgent": _make_agent("Why did the scarecrow win an award? Because he was outstanding in his field!"),
        "Niceify": _make_agent("What a wonderful, uplifting joke about dedication!"),
    }
    pipeline = Pipeline(
        name="NiceJoke",
        description="",
        steps=[PipelineStep(agent="DadJokeAgent"), PipelineStep(agent="Niceify")],
    )

    with patch.dict(orch_module._global_agents, mock_agents, clear=True), \
         patch.dict(orch_module._foundry_agent_map, {}, clear=True), \
         patch.dict(orch_module._pipeline_registry, {"NiceJoke": pipeline}, clear=True):
        result = await _call_tool(run_pipeline, "NiceJoke", "Tell me a farm joke")

    assert "uplifting" in result
    # Verify step 2 received step 1's output as its input
    mock_agents["Niceify"].run.assert_called_once()
    call_messages = mock_agents["Niceify"].run.call_args[0][0]
    assert any("outstanding in his field" in m.text for m in call_messages)


@pytest.mark.asyncio
async def test_run_pipeline_single_step():
    """A single-step pipeline returns the agent's output directly."""
    mock_agents = {"AgentA": _make_agent("result from A")}
    pipeline = Pipeline(name="Solo", description="", steps=[PipelineStep(agent="AgentA")])

    with patch.dict(orch_module._global_agents, mock_agents, clear=True), \
         patch.dict(orch_module._foundry_agent_map, {}, clear=True), \
         patch.dict(orch_module._pipeline_registry, {"Solo": pipeline}, clear=True):
        result = await _call_tool(run_pipeline, "Solo", "input text")

    assert result == "result from A"


# ---------------------------------------------------------------------------
# Unit: run_pipeline — error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_unknown_name():
    """run_pipeline returns a descriptive error for an unknown pipeline name."""
    with patch.dict(orch_module._pipeline_registry, {}, clear=True):
        result = await _call_tool(run_pipeline, "GhostPipeline", "hello")

    assert "GhostPipeline" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_run_pipeline_no_steps():
    """run_pipeline returns an error if a pipeline has no steps."""
    pipeline = Pipeline(name="Empty", description="", steps=[])

    with patch.dict(orch_module._pipeline_registry, {"Empty": pipeline}, clear=True):
        result = await _call_tool(run_pipeline, "Empty", "hello")

    assert "no steps" in result


@pytest.mark.asyncio
async def test_run_pipeline_step_error_aborts():
    """If a step returns an Error string, the pipeline aborts and reports the step."""
    mock_agents = {
        "BadAgent": _make_agent("Error: something went wrong"),
        "GoodAgent": _make_agent("should not be reached"),
    }
    pipeline = Pipeline(
        name="FaultyPipe",
        description="",
        steps=[PipelineStep(agent="BadAgent"), PipelineStep(agent="GoodAgent")],
    )

    with patch.dict(orch_module._global_agents, mock_agents, clear=True), \
         patch.dict(orch_module._foundry_agent_map, {}, clear=True), \
         patch.dict(orch_module._pipeline_registry, {"FaultyPipe": pipeline}, clear=True):
        result = await _call_tool(run_pipeline, "FaultyPipe", "trigger failure")

    assert "failed at step 1" in result
    assert "BadAgent" in result
    # GoodAgent must never have been invoked
    mock_agents["GoodAgent"].run.assert_not_called()


@pytest.mark.asyncio
async def test_run_pipeline_unknown_agent_aborts():
    """An unrecognised agent name inside a pipeline aborts with a clear message."""
    pipeline = Pipeline(
        name="MissingPipe",
        description="",
        steps=[PipelineStep(agent="GhostAgent")],
    )

    with patch.dict(orch_module._global_agents, {}, clear=True), \
         patch.dict(orch_module._foundry_agent_map, {}, clear=True), \
         patch.dict(orch_module._pipeline_registry, {"MissingPipe": pipeline}, clear=True):
        result = await _call_tool(run_pipeline, "MissingPipe", "hello")

    # _invoke_agent returns "Error: Agent 'GhostAgent' is not a recognized specialist."
    # run_pipeline should surface that and include the step number
    assert "failed at step 1" in result
    assert "GhostAgent" in result


# ---------------------------------------------------------------------------
# Integration-style unit: NiceJokeLocal (local-only pipeline)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nice_joke_local_pipeline():
    """
    NiceJoke pipeline uses local agents only (DadJokeAgent + Uplifting).
    Verifies the full two-step chain works end-to-end without any Foundry agents.
    """
    mock_agents = {
        "DadJokeAgent": _make_agent("Why don't scientists trust atoms? Because they make up everything!"),
        "Uplifting": _make_agent("What a wonderfully curious and thoughtful joke about the amazing world of science!"),
    }
    pipeline = Pipeline(
        name="NiceJoke",
        description="",
        steps=[PipelineStep(agent="DadJokeAgent"), PipelineStep(agent="Uplifting")],
    )

    with patch.dict(orch_module._global_agents, mock_agents, clear=True), \
         patch.dict(orch_module._foundry_agent_map, {}, clear=True), \
         patch.dict(orch_module._pipeline_registry, {"NiceJoke": pipeline}, clear=True):
        result = await _call_tool(run_pipeline, "NiceJoke", "Tell me a science joke")

    assert "wonderfully curious" in result
    # Confirm Uplifting received DadJokeAgent's output as its input
    uplifting_call_messages = mock_agents["Uplifting"].run.call_args[0][0]
    assert any("atoms" in m.text for m in uplifting_call_messages)


@pytest.mark.asyncio
async def test_knock_knock_nice_local_pipeline():
    """
    KnockKnockNice pipeline uses local agents only (KnockKnockJokeAgent + Uplifting).
    """
    mock_agents = {
        "KnockKnockJokeAgent": _make_agent("Knock knock. Who's there? Lettuce. Lettuce who? Lettuce in, it's cold out here!"),
        "Uplifting": _make_agent("What a delightfully playful knock-knock joke that brings warmth and laughter!"),
    }
    pipeline = Pipeline(
        name="KnockKnockNice",
        description="",
        steps=[PipelineStep(agent="KnockKnockJokeAgent"), PipelineStep(agent="Uplifting")],
    )

    with patch.dict(orch_module._global_agents, mock_agents, clear=True), \
         patch.dict(orch_module._foundry_agent_map, {}, clear=True), \
         patch.dict(orch_module._pipeline_registry, {"KnockKnockNice": pipeline}, clear=True):
        result = await _call_tool(run_pipeline, "KnockKnockNice", "Tell me a door joke")

    assert "delightfully playful" in result
    uplifting_call_messages = mock_agents["Uplifting"].run.call_args[0][0]
    assert any("Lettuce" in m.text for m in uplifting_call_messages)
