"""
local_agent.py
--------------
A sample local agent that uses a tool to get the weather.
This demonstrates how to mix local agents (using agent_framework)
with cloud-hosted Foundry agents.
"""

from agent_framework import ai_function

@ai_function
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # Mock implementation for demonstration purposes
    return f"The weather in {location} is sunny and 72°F."
