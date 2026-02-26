import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add parent directory to path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app

# We use a single TestClient instance in a fixture to ensure the lifespan 
# (which initializes the orchestrator) is only triggered once per test session,
# or we can just use it directly in tests. Using it in a context manager 
# triggers the startup/shutdown events.

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    """Test the REST /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Foundry-Agent API"

@pytest.mark.integration
def test_rest_chat_endpoint(client):
    """Test the REST /chat endpoint."""
    payload = {
        "content": "Tell me a dad joke"
    }
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0

@pytest.mark.integration
def test_websocket_chat_endpoint(client):
    """Test the WebSocket /ws/chat endpoint."""
    with client.websocket_connect("/ws/chat") as websocket:
        # Send a request
        websocket.send_json({
            "content": "Tell me a dad joke"
        })
        
        # Receive the streamed response
        messages = []
        while True:
            data = websocket.receive_json()
            messages.append(data)
            if data.get("type") == "done":
                break
                
        assert len(messages) > 0
        
        # Verify we got the expected event types
        event_types = [msg.get("type") for msg in messages]
        assert "done" in event_types
        # We should get at least some answer tokens or a status update
        assert "answer" in event_types or "status" in event_types

def test_rest_chat_endpoint_missing_field(client):
    """Test the REST /chat endpoint with invalid payload."""
    payload = {
        # Missing 'content'
        "history": []
    }
    response = client.post("/chat", json=payload)
    
    # FastAPI should return 422 Unprocessable Entity for validation errors
    assert response.status_code == 422
