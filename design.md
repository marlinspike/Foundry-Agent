# Architectural Decisions & Coding Guidelines for Foundry-Agent

This document outlines the core architectural decisions and patterns used in the `Foundry-Agent` project. Any AI coding assistant (Claude, Copilot, etc.) should adhere to these principles when modifying or extending the codebase.

## 1. Hybrid Multi-Agent Orchestration
**Decision:** The system uses a central "Router" pattern (`OrchestratorDirectAgent`) that delegates tasks to a mix of local and cloud-hosted specialist agents.
*   **Mechanism:** The orchestrator is given a single, powerful tool: `delegate_to_agent(agent_name, query)`. 
*   **Why:** This simplifies the orchestrator's prompt and tool schema. Instead of giving the orchestrator 50 different tools for 50 different agents, it only needs to know *how to route* by name.
*   **Rule:** Do not add individual specialist tools directly to the `OrchestratorDirectAgent`. Instead, update its system instructions in `agents.yaml` to make it aware of new specialist agents it can route to via `delegate_to_agent`.
*   **Strict Routing Rule:** The orchestrator must act *only* as a router. Its system prompt in `agents.yaml` must explicitly forbid it from answering general knowledge questions itself. If a request falls outside the scope of its known specialist agents, it must return a polite refusal message.

## 2. Configuration-Driven Local Agents (Pure-Text Agents)
**Decision:** Local agents are defined declaratively in `agents.yaml` rather than hardcoded in Python. This enables a "No-Code" or "Pure-Text" agent architecture.
*   **Mechanism:** `orchestrator.py` reads `agents.yaml` on startup, dynamically imports the specified tool functions (if any), and instantiates the agents using `agent_framework`.
*   **Why:** 
    *   **Innovation:** It allows for the creation of entirely new, fully functional agents (like a `DadJokeAgent`) without writing a single line of Python code, provided they don't need external API tools.
    *   **Separation of Concerns:** It cleanly separates the *behavior* (the prompt in YAML) from the *execution* (the tool function in Python).
    *   **Accessibility:** Allows non-developers (prompt engineers, product managers) to tweak agent prompts, add new generative agents, and modify routing logic just by editing a text file.
*   **Rule:** When adding a new local agent, define its `instructions` and `tools` in `agents.yaml`. If it requires custom logic, implement the tool in a separate Python file (e.g., `local_agent.py`) and reference it via its module path. If it is purely generative, omit the `tools` array entirely.

## 3. Separation of Cloud Agent Invocation
**Decision:** Interactions with Azure AI Foundry hosted agents are isolated in `foundry_tools.py`.
*   **Mechanism:** Uses the `azure-ai-agents` SDK to resolve agent IDs (with in-memory caching), create threads, submit messages, and poll for completion.
*   **Why:** Cloud agents operate asynchronously and require stateful thread management on the server side. Isolating this keeps the core orchestrator logic clean.
*   **Rule:** If you need to interact with a new Azure AI Foundry agent, you generally do not need to write new Python code. The `delegate_to_agent` function automatically falls back to searching Foundry for the `agent_name` if it isn't found locally.

## 4. Streaming by Default
**Decision:** The orchestrator streams responses back to the user interface.
*   **Mechanism:** `OrchestratorWorkflow.run_stream` yields tuples of `(event_type, text)`. The CLI (`main.py`) consumes this async generator and prints chunks to `stdout` as they arrive.
*   **Why:** Multi-agent systems can have high latency (especially when routing to cloud agents). Streaming provides immediate feedback to the user.
*   **Rule:** Do not revert to blocking `run()` calls in the main execution path. Ensure any modifications to `orchestrator.py` or `main.py` preserve the `async for` streaming loop.

## 5. Provider Agnosticism for the Orchestrator
**Decision:** The core orchestrator LLM can be powered by different providers.
*   **Mechanism:** `_build_client` in `orchestrator.py` checks the `MODEL_PROVIDER` environment variable to instantiate `AzureAIClient`, `AzureOpenAIChatClient`, or `OpenAIChatClient`.
*   **Why:** Allows developers to test the orchestrator logic locally using cheaper/faster models (like standard OpenAI) before deploying to Azure AI Foundry.
*   **Rule:** Do not hardcode Azure-specific client logic outside of the `_build_client` factory.

## Summary Checklist for Adding a New Agent

**To add a Local Agent:**
1. Write the tool function in Python (e.g., `@ai_function def my_tool(): ...`).
2. Add the agent definition to `agents.yaml` (name, instructions, tool path).
3. Update `OrchestratorDirectAgent` instructions in `agents.yaml` so it knows when to route to this new agent.

**To add a Cloud (Foundry) Agent:**
1. Deploy the agent in Azure AI Foundry.
2. Update `OrchestratorDirectAgent` instructions in `agents.yaml` so it knows the exact display name of the Foundry agent.
3. Add an environment variable mapping in the `foundry_agents` section of `agents.yaml` to map the routing name to the environment variable containing the actual Foundry display name.