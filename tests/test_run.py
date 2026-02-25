import asyncio
import logging
import sys
import os
from dotenv import load_dotenv

# Add parent directory to path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(override=True)

from agent.orchestrator import build_orchestrator, _global_agents
from agent_framework import ChatMessage, Role

logging.basicConfig(level=logging.DEBUG)

async def main():
    async with build_orchestrator() as orch:
        agent = _global_agents["DadJokeAgent"]
        print("Calling DadJokeAgent concurrently...")
        results = await asyncio.gather(
            agent.run("Tell me a joke about a cat."),
            agent.run("Tell me a joke about a dog.")
        )
        print(results)

if __name__ == "__main__":
    asyncio.run(main())
