"""
tests/test_parallel_delegation.py
-----------------------------------
Unit + integration tests for the parallel delegation feature (Task #2).

Unit tests mock _global_agents directly — no LLM / network calls required.
The integration test spins up a real orchestrator and requires env vars.
"""

import asyncio
import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import orchestrator as orch_module
from orchestrator import _invoke_agent, delegate_to_multiple_agents


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
# Unit: _invoke_agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invoke_agent_local_agent():
    """_invoke_agent dispatches to a local agent and returns its text."""
    with patch.dict(orch_module._global_agents, {"MockAgent": _make_agent("pong")}, clear=True):
        result = await _invoke_agent("MockAgent", "ping")
    assert result == "pong"


@pytest.mark.asyncio
async def test_invoke_agent_unknown():
    """_invoke_agent returns a descriptive error for unrecognised names."""
    with patch.dict(orch_module._global_agents, {}, clear=True), \
         patch.dict(orch_module._foundry_agent_map, {}, clear=True):
        result = await _invoke_agent("GhostAgent", "hello")
    assert "GhostAgent" in result
    assert "not a recognized specialist" in result


# ---------------------------------------------------------------------------
# Unit: delegate_to_multiple_agents — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parallel_two_agents():
    """Fan-out to two agents; both results appear in output, labelled."""
    mock_agents = {
        "JokeAgent": _make_agent("Why did the chicken cross the road?"),
        "WeatherAgent": _make_agent("Sunny, 72°F"),
    }
    with patch.dict(orch_module._global_agents, mock_agents, clear=True), \
         patch.dict(orch_module._foundry_agent_map, {}, clear=True):
        result = await _call_tool(
            delegate_to_multiple_agents,
            [
                {"agent_name": "JokeAgent",    "query": "Tell me a joke"},
                {"agent_name": "WeatherAgent", "query": "Weather in NYC?"},
            ],
        )

    assert "[JokeAgent]" in result
    assert "chicken" in result
    assert "[WeatherAgent]" in result
    assert "Sunny" in result


@pytest.mark.asyncio
async def test_parallel_results_are_concurrent():
    """Agents run concurrently: total wall time ≈ one agent's latency, not N×."""
    import time

    async def _slow_run(messages):
        await asyncio.sleep(0.1)
        r = MagicMock()
        r.text = "done"
        return r

    mock_agents = {}
    for name in ("A", "B", "C"):
        agent = AsyncMock()
        agent.run = _slow_run
        mock_agents[name] = agent

    with patch.dict(orch_module._global_agents, mock_agents, clear=True), \
         patch.dict(orch_module._foundry_agent_map, {}, clear=True):
        start = time.monotonic()
        result = await _call_tool(
            delegate_to_multiple_agents,
            [{"agent_name": n, "query": "ping"} for n in ("A", "B", "C")],
        )
        elapsed = time.monotonic() - start

    # Sequential would take ~0.3 s; parallel should finish in ~0.1 s (+ overhead)
    assert elapsed < 0.25, f"Expected parallel execution but took {elapsed:.2f}s"
    assert all(f"[{n}]" in result for n in ("A", "B", "C"))


# ---------------------------------------------------------------------------
# Unit: delegate_to_multiple_agents — edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parallel_empty_list():
    """Empty input returns a clear error message."""
    result = await _call_tool(delegate_to_multiple_agents, [])
    assert "empty" in result.lower()


@pytest.mark.asyncio
async def test_parallel_missing_keys():
    """Items without required keys are rejected with an informative error."""
    result = await _call_tool(
        delegate_to_multiple_agents,
        [{"agent_name": "X"}, {"query": "hi"}],  # both missing one key each
    )
    assert "missing" in result.lower() or "index" in result.lower()


@pytest.mark.asyncio
async def test_parallel_partial_failure():
    """If one agent raises, its slot shows an error; others still succeed."""
    good_agent = _make_agent("I'm fine")
    bad_agent = AsyncMock()
    bad_agent.run = AsyncMock(side_effect=RuntimeError("boom"))

    mock_agents = {"GoodAgent": good_agent, "BadAgent": bad_agent}
    with patch.dict(orch_module._global_agents, mock_agents, clear=True), \
         patch.dict(orch_module._foundry_agent_map, {}, clear=True):
        result = await _call_tool(
            delegate_to_multiple_agents,
            [
                {"agent_name": "GoodAgent", "query": "hello"},
                {"agent_name": "BadAgent",  "query": "explode"},
            ],
        )

    assert "[GoodAgent]" in result
    assert "I'm fine" in result
    assert "[BadAgent]" in result
    assert "Error" in result


# ---------------------------------------------------------------------------
# Integration: orchestrator routes multi-domain prompt through both tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.integration
async def test_orchestrator_multi_domain_routing():
    """
    Integration test: a prompt that spans two joke genres should trigger
    delegate_to_multiple_agents (or two single calls) and return a non-empty answer.
    Requires real env vars; skipped automatically if FOUNDRY_PROJECT_ENDPOINT is unset.
    """
    from dotenv import load_dotenv
    load_dotenv(override=True)

    if not os.environ.get("FOUNDRY_PROJECT_ENDPOINT") and \
       not os.environ.get("AZURE_OPENAI_ENDPOINT") and \
       not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("No LLM credentials found; skipping integration test.")

    from orchestrator import build_orchestrator

    async with build_orchestrator() as orch:
        response = await orch.run(
            user_text="Tell me a dad joke AND a knock-knock joke at the same time."
        )

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
