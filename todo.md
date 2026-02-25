# Foundry-Agent Innovation Tasks

This document tracks the planned architectural enhancements and innovations for the Foundry-Agent project.

## 1. Dynamic Agent Discovery (Auto-Prompt Injection)
**Status:** Completed

**Synopsis:** 
Currently, the orchestrator's instructions in `agents.yaml` manually list the available agents. If someone adds a new agent, they must remember to update the orchestrator's prompt. We will dynamically inject the list of available agents and their descriptions into the orchestrator's system prompt at runtime. In `_load_agents_dynamically`, we can iterate through `local_agents` and `foundry_agents`, build a formatted string of their names and descriptions, and inject it into a placeholder (e.g., `{{AGENT_LIST}}`) in the orchestrator's instructions. This makes adding new agents truly "zero-touch" for the routing logic.

## 2. Parallel Delegation (Multi-Agent Collaboration)
**Status:** Completed

**Synopsis:** 
Currently, the `delegate_to_agent` tool takes a single `agent_name`. If a user asks a complex question, the orchestrator has to pick one or call them sequentially. We will enhance the routing to support parallel execution. We can modify the tool to accept a list of agents: `delegate_to_multiple_agents(agent_queries: list[dict])`. The orchestrator could then fan-out requests to multiple agents concurrently using `asyncio.gather`, and synthesize their combined responses much faster.

## 3. Semantic Routing (For Scale)
**Status:** Not Started

**Synopsis:** 
Currently, the orchestrator relies on the LLM reading the list of agents in its prompt to decide where to route. This works well for 5-10 agents but breaks down or gets expensive if you have 50+ agents. We will implement a semantic router. When loading `agents.yaml`, we generate embeddings for each agent's `description`. When a user prompt comes in, we embed the prompt, find the top-K most semantically similar agents using cosine similarity, and *only* provide those top-K agents to the orchestrator LLM to choose from.

## 4. Agent Chaining / Pipelines in YAML
**Status:** Not Started

**Synopsis:** 
Currently, agents operate independently. The orchestrator calls Agent A, gets the result, and replies to the user. We will allow defining declarative pipelines in `agents.yaml`. Add a `pipelines` section to the YAML where the output of one agent is automatically piped into another. For example, a user could call a `NiceJoke` pipeline that automatically routes the prompt to `DadJokeAgent`, takes the output, and pipes it directly into the `Niceify` Foundry agent before returning to the user.

## 5. REST API / WebSocket Interface
**Status:** Completed

**Synopsis:** 
Currently, the system is a CLI application (`main.py`). We will expose the orchestrator as a service. Add a lightweight `api.py` using FastAPI. We can expose a `/chat` endpoint for standard requests and a WebSocket endpoint to support the streaming responses (`async for event, text in workflow.run_stream(...)`) directly to a web frontend.

## 6. Shared Memory / Context Object
**Status:** Not Started

**Synopsis:** 
Currently, context is maintained purely through the conversation history passed back and forth. We will give agents a shared "scratchpad". Create a `Context` object that is passed along with the `delegate_to_agent` call. A specialist agent could write a fact to the context (e.g., `context.set("user_location", "Seattle")`), which the orchestrator and other agents can read in subsequent turns without it needing to be explicitly stated in the chat history.
