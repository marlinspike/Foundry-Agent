# Architectural Decisions & Coding Guidelines for Foundry-Agent

This document outlines the core architectural decisions and patterns used in the `Foundry-Agent` project. Any AI coding assistant (Claude, Copilot, etc.) should adhere to these principles when modifying or extending the codebase.

## 1. Hybrid Multi-Agent Orchestration
**Decision:** The system uses a central "Router" pattern (`OrchestratorDirectAgent`) that delegates tasks to a mix of local and cloud-hosted specialist agents.
*   **Mechanism:** The orchestrator is given two routing tools:
    *   `delegate_to_agent(agent_name, query)` — routes a request to a **single** specialist.
    *   `delegate_to_multiple_agents(agent_queries)` — fans out to **multiple** specialists **concurrently** via `asyncio.gather`, returning labeled results for the orchestrator to synthesise.
*   **Why:** Using two well-defined tools keeps the orchestrator's schema simple while enabling parallel execution for multi-domain requests. Instead of calling one agent repeatedly in sequence, the orchestrator can fan out to N agents in a single tool call and return a richer, faster answer.
*   **Rule:** Do not add individual specialist tools directly to the `OrchestratorDirectAgent`. Instead, update its system instructions in `agents.yaml` to make it aware of new specialist agents it can route to via the delegation tools.
*   **Strict Routing Rule:** The orchestrator must act *only* as a router. Its system prompt in `agents.yaml` must explicitly forbid it from answering general knowledge questions itself. If a request falls outside the scope of its known specialist agents, it must return a polite refusal message.
*   **Tool Selection Rule:** The orchestrator's prompt specifies: use `delegate_to_agent` when the request maps to one specialist; use `delegate_to_multiple_agents` when the request spans multiple domains — do **not** call `delegate_to_agent` repeatedly in sequence for multi-domain queries.

## 2. Configuration-Driven Local Agents (Pure-Text Agents)
**Decision:** Local agents are defined declaratively in `agents.yaml` rather than hardcoded in Python. This enables a "No-Code" or "Pure-Text" agent architecture.
*   **Mechanism:** `orchestrator.py` reads `agents.yaml` on startup, dynamically imports the specified tool functions (if any), and instantiates the agents using `agent_framework`.
*   **Why:** 
    *   **Innovation:** It allows for the creation of entirely new, fully functional agents (like a `DadJokeAgent`) without writing a single line of Python code, provided they don't need external API tools.
    *   **Separation of Concerns:** It cleanly separates the *behavior* (the prompt in YAML) from the *execution* (the tool function in Python).
    *   **Accessibility:** Allows non-developers (prompt engineers, product managers) to tweak agent prompts, add new generative agents, and modify routing logic just by editing a text file.
*   **Rule:** When adding a new local agent, define its `instructions` and `tools` in `agents.yaml`. If it requires custom logic, implement the tool in a separate Python file (e.g., `local_agent.py`) and reference it via its module path. If it is purely generative, omit the `tools` array entirely.

## 3. Parallel Delegation via `asyncio.gather`
**Decision:** Multi-domain user requests are handled by fanning out to multiple agents concurrently, not sequentially.
*   **Mechanism:**
    1.  The orchestrator LLM calls `delegate_to_multiple_agents(agent_queries)` with a list of `{agent_name, query}` objects.
    2.  `orchestrator.py` builds a list of `_invoke_agent` coroutines — one per entry — and runs them with `asyncio.gather(..., return_exceptions=True)`.
    3.  Each result (or exception) is labeled `[AgentName]: …` and the combined string is returned to the orchestrator for final synthesis.
*   **Why `return_exceptions=True`:** A failure in one agent does not cancel the others. Partial results are still returned and labeled with the error, so the orchestrator can compose a useful answer from whichever agents succeeded.
*   **Shared Core:** Both `delegate_to_agent` and `delegate_to_multiple_agents` delegate to a shared `_invoke_agent(agent_name, query)` coroutine. This is the single source of truth for local-vs-Foundry dispatch logic, avoiding duplication.
*   **Rule:** All agent dispatch logic belongs in `_invoke_agent`. Do not replicate the local/Foundry lookup pattern inside any new tool function.

## 4. Separation of Cloud Agent Invocation
**Decision:** Interactions with Azure AI Foundry hosted agents are isolated in `foundry_tools.py`.
*   **Mechanism:** Uses the `azure-ai-agents` SDK to resolve agent IDs (with in-memory caching), create threads, submit messages, and poll for completion.
*   **Why:** Cloud agents operate asynchronously and require stateful thread management on the server side. Isolating this keeps the core orchestrator logic clean.
*   **Rule:** If you need to interact with a new Azure AI Foundry agent, you generally do not need to write new Python code. The `delegate_to_agent` function automatically falls back to searching Foundry for the `agent_name` if it isn't found locally.

## 5. Streaming by Default
**Decision:** The orchestrator streams responses back to the user interface.
*   **Mechanism:** `OrchestratorWorkflow.run_stream` yields tuples of `(event_type, text)`. The CLI (`main.py`) consumes this async generator and prints chunks to `stdout` as they arrive.
*   **Why:** Multi-agent systems can have high latency (especially when routing to cloud agents). Streaming provides immediate feedback to the user.
*   **Rule:** Do not revert to blocking `run()` calls in the main execution path. Ensure any modifications to `orchestrator.py` or `main.py` preserve the `async for` streaming loop.

## 6. Provider Agnosticism for the Orchestrator
**Decision:** The core orchestrator LLM can be powered by different providers.
*   **Mechanism:** `_build_client` in `orchestrator.py` checks the `MODEL_PROVIDER` environment variable to instantiate `AzureAIClient`, `AzureOpenAIChatClient`, or `OpenAIChatClient`.
*   **Why:** Allows developers to test the orchestrator logic locally using cheaper/faster models (like standard OpenAI) before deploying to Azure AI Foundry.
*   **Rule:** Do not hardcode Azure-specific client logic outside of the `_build_client` factory.

## Summary Checklist for Adding a New Agent

**To add a Local Agent:**
1. Write the tool function in Python (e.g., `@ai_function def my_tool(): ...`).
2. Add the agent definition to `agents.yaml` (name, instructions, tool path).
3. The `{{AGENT_LIST}}` placeholder in the orchestrator's instructions is populated automatically at startup — no manual prompt editing required.

**To add a Cloud (Foundry) Agent:**
1. Deploy the agent in Azure AI Foundry.
2. Add the agent to the `foundry_agents` section of `agents.yaml` with a `description` and `env_var`.
3. The `{{AGENT_LIST}}` placeholder is populated automatically — again, no manual prompt editing required.

**Routing is automatic.** The orchestrator decides at inference time whether to use `delegate_to_agent` (single domain) or `delegate_to_multiple_agents` (multi-domain) based on the routing rules in its system prompt. No code changes are needed when adding new agents.