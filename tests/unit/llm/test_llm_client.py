"""Unit tests for LLM client."""

import pytest
from typing import Dict, Any

from CadSeqProc.enhanced_geometry.llm_client import LLMClient

pytestmark = pytest.mark.asyncio

async def test_analyze_text(llm_client: LLMClient, simple_cube_prompt: str):
    """Test basic text analysis."""
    result = await llm_client.analyze_text(simple_cube_prompt)
    assert isinstance(result, dict)
    assert "dimensions" in result
    assert "manufacturing" in result
    assert "features" in result

async def test_analyze_request_with_context(llm_client: LLMClient, complex_shape_prompt: str):
    """Test analysis with conversation context."""
    # First request
    result1 = await llm_client.analyze_request("I want to design a container")
    assert isinstance(result1, dict)
    
    # Second request should include context from first
    result2 = await llm_client.analyze_request(complex_shape_prompt)
    assert isinstance(result2, dict)
    assert "dimensions" in result2
    assert "manufacturing" in result2
    assert "features" in result2

async def test_generate_recommendations(llm_client: LLMClient):
    """Test manufacturing recommendations generation."""
    geometry_data = {
        "type": "cylinder",
        "dimensions": {
            "height": 150,
            "diameter": 80,
            "wall_thickness": 2,
            "unit": "mm"
        }
    }
    
    result = await llm_client.generate_recommendations(geometry_data)
    assert isinstance(result, dict)
    assert "best_process" in result
    assert "material_suggestions" in result
    assert "considerations" in result
    assert "constraints" in result

def test_validate_response(llm_client: LLMClient):
    """Test response validation."""
    # Valid response
    valid_response = {
        "dimensions": {"width": 10, "height": 10, "depth": 10},
        "manufacturing": {"process": "3D printing"},
        "features": [{"type": "cube"}]
    }
    assert llm_client.validate_response(valid_response) is True
    
    # Invalid response - missing required field
    invalid_response = {
        "dimensions": {"width": 10, "height": 10, "depth": 10},
        "manufacturing": {"process": "3D printing"}
    }
    assert llm_client.validate_response(invalid_response) is False

@pytest.mark.parametrize("prompt,expected_dimensions", [
    ("Create a cube with 5cm sides", {"width": 50, "height": 50, "depth": 50, "unit": "mm"}),
    ("Make a thin plate 10cm x 20cm x 1mm", {"width": 100, "height": 200, "depth": 1, "unit": "mm"})
])
async def test_dimension_parsing(llm_client: LLMClient, prompt: str, expected_dimensions: Dict[str, Any]):
    """Test dimension parsing from various prompts."""
    result = await llm_client.analyze_text(prompt)
    assert "dimensions" in result
    dimensions = result["dimensions"]
    for key, value in expected_dimensions.items():
        assert dimensions[key] == value

@pytest.mark.parametrize("prompt,expected_process", [
    ("Create a cube for 3D printing", "3D printing"),
    ("Machine a metal bracket", "CNC milling")
])
async def test_manufacturing_process_detection(llm_client: LLMClient, prompt: str, expected_process: str):
    """Test manufacturing process detection from prompts."""
    result = await llm_client.analyze_text(prompt)
    assert "manufacturing" in result
    assert result["manufacturing"]["process"].lower() == expected_process.lower()

async def test_error_handling(llm_client: LLMClient):
    """Test error handling in LLM client."""
    # Test with empty prompt
    result = await llm_client.analyze_text("")
    assert "error" in result
    
    # Test with invalid prompt
    result = await llm_client.analyze_text("@#$%^")
    assert "error" in result
    
    # Test with extremely long prompt
    long_prompt = "word " * 1000
    result = await llm_client.analyze_text(long_prompt)
    assert "error" in result 