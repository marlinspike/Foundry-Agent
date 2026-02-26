import pytest
import asyncio
import sys
import os
from dotenv import load_dotenv

# Add parent directory to path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(override=True)

from agent.orchestrator import build_orchestrator

@pytest.mark.asyncio
@pytest.mark.integration
async def test_orchestrator_basic_routing():
    """Test that the orchestrator can route a simple request to an agent."""
    async with build_orchestrator() as orch:
        # We ask for a dad joke, which should route to the DadJokeAgent
        response = await orch.run(user_text="Tell me a dad joke")
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # The response should ideally contain a joke, but we just check it's not empty

@pytest.mark.asyncio
@pytest.mark.integration
async def test_orchestrator_parallel_routing():
    """Test that the orchestrator can handle parallel requests."""
    async with build_orchestrator() as orch:
        # Run two requests in parallel
        task1 = orch.run(user_text="Tell me a dad joke")
        task2 = orch.run(user_text="Tell me a knock-knock joke")
        
        results = await asyncio.gather(task1, task2)
        
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)
        assert all(len(r) > 0 for r in results)
