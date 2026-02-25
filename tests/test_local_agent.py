import pytest
import sys
import os

# Add parent directory to path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_agent import get_weather

def test_get_weather():
    """Test the get_weather tool."""
    # The get_weather function is decorated with @ai_function, 
    # but we can still call it directly or access its underlying function.
    # If it's an AIFunction object, we might need to call it differently,
    # but usually they are callable.
    
    # Let's check if it's callable directly
    try:
        result = get_weather("Seattle")
        assert "Seattle" in result
        assert "sunny" in result
    except TypeError:
        # If it's an AIFunction object that requires specific invocation
        # we might need to access the original function
        if hasattr(get_weather, "func"):
            result = get_weather.func("Seattle")
            assert "Seattle" in result
            assert "sunny" in result
        else:
            pytest.fail("Could not call get_weather")
