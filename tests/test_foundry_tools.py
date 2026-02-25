import pytest
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock

# Add parent directory to path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foundry_tools import _resolve_agent_id, invoke_foundry_agent

class AsyncIteratorMock:
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.items):
            item = self.items[self.index]
            self.index += 1
            return item
        raise StopAsyncIteration

@pytest.mark.asyncio
async def test_resolve_agent_id():
    """Test that _resolve_agent_id correctly finds an agent ID."""
    # Mock the client and its list_agents method
    mock_client = MagicMock()
    
    # Create mock agents
    mock_agent1 = MagicMock()
    mock_agent1.name = "Agent1"
    mock_agent1.id = "id-1"
    
    mock_agent2 = MagicMock()
    mock_agent2.name = "Agent2"
    mock_agent2.id = "id-2"
    
    # Mock the async iterator returned by list_agents
    mock_client.list_agents.return_value = AsyncIteratorMock([mock_agent1, mock_agent2])
    
    # Test resolving an existing agent
    agent_id = await _resolve_agent_id(mock_client, "Agent2")
    assert agent_id == "id-2"
    
    # Test resolving a non-existent agent
    mock_client.list_agents.return_value = AsyncIteratorMock([mock_agent1, mock_agent2])
    with pytest.raises(ValueError, match="not found in your Foundry project"):
        await _resolve_agent_id(mock_client, "Agent3")

@pytest.mark.asyncio
@patch("foundry_tools._get_client")
@patch("foundry_tools._resolve_agent_id")
async def test_invoke_foundry_agent(mock_resolve, mock_get_client):
    """Test that invoke_foundry_agent correctly interacts with the client."""
    # Setup mocks
    mock_client = AsyncMock()
    mock_get_client.return_value = mock_client
    mock_resolve.return_value = "mock-agent-id"
    
    # Mock run creation
    mock_run = MagicMock()
    mock_run.status = "completed"
    mock_run.thread_id = "mock-thread-id"
    mock_client.create_thread_and_process_run.return_value = mock_run
    
    # Mock message retrieval
    mock_text_msg = MagicMock()
    mock_text_msg.text.value = "This is a mock response from the Foundry agent."
    mock_client.messages.get_last_message_text_by_role.return_value = mock_text_msg
    
    # Call the function
    response = await invoke_foundry_agent("MockAgent", "Hello")
    
    # Verify the response
    assert response == "This is a mock response from the Foundry agent."
    
    # Verify the client was called correctly
    mock_resolve.assert_called_once_with(mock_client, "MockAgent")
    mock_client.create_thread_and_process_run.assert_called_once()
    mock_client.messages.get_last_message_text_by_role.assert_called_once()
