"""
server.py
---------
Agent-server entry point for the AI Toolkit Agent Inspector.

This exposes the orchestrator via the azure-ai-agentserver protocol,
which registers the /binder-requests endpoint (port 8000 by default).

Usage:
  python server.py                  # run normally
  agentdev run server.py --verbose  # with AI Toolkit Agent Inspector
"""

import asyncio
import os

from dotenv import load_dotenv

# Load .env before any project imports so env vars are set
load_dotenv(override=True)

from azure.ai.agentserver.agentframework import from_agent_framework
from agent.orchestrator import build_orchestrator


async def main() -> None:
    async with build_orchestrator() as orch:
        # orch._direct_agent is the raw agent-framework agent object
        await from_agent_framework(orch._direct_agent).run_async()


if __name__ == "__main__":
    asyncio.run(main())
