"""
agent/pipeline.py
-----------
Declarative agent-chaining / pipeline engine.

Pipelines are defined in agents.yaml under the `pipelines` key:

    pipelines:
      NiceJoke:
        description: "Generates a dad joke then niceifies it"
        steps:
          - agent: DadJokeAgent
          - agent: Niceify

Each pipeline is a named, ordered sequence of PipelineSteps.  The output of
each step is fed as the input to the next step; the initial input comes from
the caller.  This module only contains the *data model* and *loader*; the
actual async execution lives in orchestrator.py alongside ``_invoke_agent``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PipelineStep:
    """A single step in a pipeline, naming the agent to invoke."""

    agent: str
    """The agent name, matching a key in local_agents or foundry_agents."""

    # Reserved for future extensions:
    #   prompt_template: str | None = None  – wrap previous output in a richer prompt
    #   condition: str | None = None        – only run if a condition expression is truthy


@dataclass
class Pipeline:
    """A named, ordered sequence of agent invocations."""

    name: str
    description: str
    steps: list[PipelineStep] = field(default_factory=list)

    @property
    def step_summary(self) -> str:
        """Human-readable chain, e.g. 'DadJokeAgent → Niceify'."""
        return " → ".join(s.agent for s in self.steps)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_pipelines(config: dict[str, Any]) -> dict[str, Pipeline]:
    """
    Parse the ``pipelines`` section of an agents.yaml config dict into a
    mapping of pipeline-name → Pipeline.

    Example YAML (already parsed to dict)::

        pipelines:
          NiceJoke:
            description: "Dad joke → niceified"
            steps:
              - agent: DadJokeAgent
              - agent: Niceify

    Parameters
    ----------
    config:
        The full parsed agents.yaml dict.

    Returns
    -------
    dict[str, Pipeline]
        An empty dict if the ``pipelines`` key is absent.
    """
    result: dict[str, Pipeline] = {}
    for name, defn in config.get("pipelines", {}).items():
        raw_steps = defn.get("steps", [])
        steps = [PipelineStep(agent=s["agent"]) for s in raw_steps]
        result[name] = Pipeline(
            name=name,
            description=defn.get("description", ""),
            steps=steps,
        )
    return result
