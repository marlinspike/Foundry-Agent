# Architectural Decisions & Coding Guidelines for Foundry-Agent

This document outlines the core architectural decisions and patterns used in the `Foundry-Agent` project. Any AI coding assistant (Claude, Copilot, etc.) should adhere to these principles when modifying or extending the codebase.

## 1. Hybrid Multi-Agent Orchestration
**Decision:** The system uses a central "Router" pattern (`OrchestratorDirectAgent`) that delegates tasks to a mix of local and cloud-hosted specialist agents.
*   **Mechanism:** The orchestrator is given three routing tools:
    *   `delegate_to_agent(agent_name, query)` — routes a request to a **single** specialist.
    *   `delegate_to_multiple_agents(agent_queries)` — fans out to **multiple** specialists **concurrently** via `asyncio.gather`, returning labeled results for the orchestrator to synthesise.
    *   `run_pipeline(pipeline_name, initial_input)` — executes a named **ordered agent chain** where the output of each step becomes the input to the next.
*   **Why:** Using three well-defined tools keeps the orchestrator's schema simple and composable. Single-domain → `delegate_to_agent`. Multi-domain at once → `delegate_to_multiple_agents`. Ordered chain with data passing → `run_pipeline`.
*   **Rule:** Do not add individual specialist tools directly to the `OrchestratorDirectAgent`. Instead, update its system instructions in `agents.yaml` to make it aware of new specialist agents it can route to via the delegation tools.
*   **Strict Routing Rule:** The orchestrator must act *only* as a router. Its system prompt in `agents.yaml` must explicitly forbid it from answering general knowledge questions itself. If a request falls outside the scope of its known specialist agents and pipelines, it must return a polite refusal message.
*   **Tool Selection Rule:** The orchestrator's prompt specifies: use `delegate_to_agent` for one specialist; use `delegate_to_multiple_agents` for concurrent multi-domain queries; use `run_pipeline` for named pipeline invocations — do **not** call `delegate_to_agent` repeatedly in sequence for multi-domain queries or for chained pipeline steps.

## 2. Configuration-Driven Local Agents (Pure-Text Agents)
**Decision:** Local agents are defined declaratively in `agents.yaml` rather than hardcoded in Python. This enables a "No-Code" or "Pure-Text" agent architecture.
*   **Mechanism:** `agent/orchestrator.py` reads `agents.yaml` on startup, dynamically imports the specified tool functions (if any), and instantiates the agents using `agent_framework`.
*   **Why:** 
    *   **Innovation:** It allows for the creation of entirely new, fully functional agents (like a `DadJokeAgent`) without writing a single line of Python code, provided they don't need external API tools.
    *   **Separation of Concerns:** It cleanly separates the *behavior* (the prompt in YAML) from the *execution* (the tool function in Python).
    *   **Accessibility:** Allows non-developers (prompt engineers, product managers) to tweak agent prompts, add new generative agents, and modify routing logic just by editing a text file.
*   **Rule:** When adding a new local agent, define its `instructions` and `tools` in `agents.yaml`. If it requires custom logic, implement the tool in a separate Python file under `agent/` (e.g., `agent/local_agent.py`) and reference it via its module path. If it is purely generative, omit the `tools` array entirely.

## 3. Parallel Delegation via `asyncio.gather`
**Decision:** Multi-domain user requests are handled by fanning out to multiple agents concurrently, not sequentially.
*   **Mechanism:**
    1.  The orchestrator LLM calls `delegate_to_multiple_agents(agent_queries)` with a list of `{agent_name, query}` objects.
    2.  `agent/orchestrator.py` builds a list of `_invoke_agent` coroutines — one per entry — and runs them with `asyncio.gather(..., return_exceptions=True)`.
    3.  Each result (or exception) is labeled `[AgentName]: …` and the combined string is returned to the orchestrator for final synthesis.
*   **Why `return_exceptions=True`:** A failure in one agent does not cancel the others. Partial results are still returned and labeled with the error, so the orchestrator can compose a useful answer from whichever agents succeeded.
*   **Shared Core:** Both `delegate_to_agent` and `delegate_to_multiple_agents` delegate to a shared `_invoke_agent(agent_name, query)` coroutine. This is the single source of truth for local-vs-Foundry dispatch logic, avoiding duplication.
*   **Rule:** All agent dispatch logic belongs in `_invoke_agent`. Do not replicate the local/Foundry lookup pattern inside any new tool function.

## 4. Declarative Agent Pipelines
**Decision:** Ordered, multi-step agent chains are defined declaratively in `agents.yaml` rather than in Python.

> **⚠️ STATIC / DECLARE-BEFORE-USE — THIS IS DIFFERENT FROM AGENT ROUTING.**
> Agents are discovered dynamically and routed to at inference time with no restart required (within a running process). Pipelines are fundamentally different: the entire registry is built **once** at process startup from `agents.yaml` and is **immutable** for the lifetime of the server. A pipeline that is not present in `agents.yaml` when the process starts **does not exist** — it cannot be added, modified, or removed at runtime. End users cannot compose pipelines through the chat interface. This is an intentional design boundary: pipeline authorship is a developer/operator task performed in config, not a user-facing capability.

*   **Mechanism:**
    1.  A `pipelines` block in `agents.yaml` lists named pipelines, each with an ordered `steps` array of agent names.
    2.  `agent/pipeline.py` provides the pure data model (`PipelineStep`, `Pipeline`) and the `load_pipelines(config)` parser — no execution logic.
    3.  `_load_agents_dynamically` in `agent/orchestrator.py` calls `load_pipelines`, populates `_pipeline_registry`, and appends a "Named Pipelines" section (with step summaries) to the `{{AGENT_LIST}}` injection.
    4.  The `run_pipeline` `@ai_function` iterates steps, calling `_invoke_agent` for each and threading the output of step *N* as the input to step *N+1*. Any step that returns an `Error:` prefix aborts the chain immediately with a diagnostic message identifying the step.
*   **Why:**
    *   **Zero-code extensibility:** Adding a new pipeline requires only a YAML edit — no Python changes whatsoever.
    *   **Separation of concerns:** `agent/pipeline.py` knows only about structure; `agent/orchestrator.py` knows only about execution. This mirrors the existing split between `agents.yaml` (what) and `agent/orchestrator.py` (how).
    *   **Fail-fast semantics:** Aborting on the first step error prevents silent data corruption deeper in the chain and gives the user an actionable error at the right level of abstraction.
*   **Rule:** Pipeline execution must always go through `_invoke_agent` for each step. Do not bypass the local/Foundry dispatch by calling agents directly from `run_pipeline`.
*   **Rule:** `agent/pipeline.py` must remain side-effect-free and import-free of `agent/orchestrator.py`. It is a pure data/parsing module.

## 5. Separation of Cloud Agent Invocation
**Decision:** Interactions with Azure AI Foundry hosted agents are isolated in `agent/foundry_tools.py`.
*   **Mechanism:** Uses the `azure-ai-agents` SDK to resolve agent IDs (with in-memory caching), create threads, submit messages, and poll for completion.
*   **Why:** Cloud agents operate asynchronously and require stateful thread management on the server side. Isolating this keeps the core orchestrator logic clean.
*   **Rule:** If you need to interact with a new Azure AI Foundry agent, you generally do not need to write new Python code. The `delegate_to_agent` function automatically falls back to searching Foundry for the `agent_name` if it isn't found locally.

## 6. Streaming by Default
**Decision:** The orchestrator streams responses back to the user interface.
*   **Mechanism:** `OrchestratorWorkflow.run_stream` yields tuples of `(event_type, text)`. The CLI (`main.py`) consumes this async generator and prints chunks to `stdout` as they arrive.
*   **Why:** Multi-agent systems can have high latency (especially when routing to cloud agents). Streaming provides immediate feedback to the user.
*   **Rule:** Do not revert to blocking `run()` calls in the main execution path. Ensure any modifications to `agent/orchestrator.py` or `main.py` preserve the `async for` streaming loop.

## 7. Provider Agnosticism for the Orchestrator
**Decision:** The core orchestrator LLM can be powered by different providers.
*   **Mechanism:** `_build_client` in `agent/orchestrator.py` checks the `MODEL_PROVIDER` environment variable to instantiate `AzureAIClient`, `AzureOpenAIChatClient`, or `OpenAIChatClient`.
*   **Why:** Allows developers to test the orchestrator logic locally using cheaper/faster models (like standard OpenAI) before deploying to Azure AI Foundry.
*   **Rule:** Do not hardcode Azure-specific client logic outside of the `_build_client` factory in `agent/orchestrator.py`.

## Summary Checklist for Adding a New Agent

**To add a Local Agent:**
1. Write the tool function in Python (e.g., `@ai_function def my_tool(): ...`) in a file under `agent/`.
2. Add the agent definition to `agents.yaml` (name, instructions, tool path using the `agent.` module prefix).
3. The `{{AGENT_LIST}}` placeholder in the orchestrator's instructions is populated automatically at startup — no manual prompt editing required.

**To add a Cloud (Foundry) Agent:**
1. Deploy the agent in Azure AI Foundry.
2. Add the agent to the `foundry_agents` section of `agents.yaml` with a `description` and `env_var`.
3. The `{{AGENT_LIST}}` placeholder is populated automatically — again, no manual prompt editing required.

**To add a Pipeline:**
1. Add a new entry to the `pipelines` section of `agents.yaml`.
2. **Restart the process.** Unlike adding an agent (which takes effect within the running process via `_load_agents_dynamically`), a pipeline cannot be hot-loaded. The registry is built once at startup.
3. No Python code required.

**Routing is automatic.** The orchestrator decides at inference time whether to use `delegate_to_agent` (single domain), `delegate_to_multiple_agents` (concurrent multi-domain), or `run_pipeline` (ordered chain) based on the routing rules in its system prompt. No code changes are needed when adding new agents or pipelines.