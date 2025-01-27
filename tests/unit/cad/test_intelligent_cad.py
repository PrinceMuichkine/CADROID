"""Unit tests for intelligent CAD system."""

import pytest
from typing import Dict, Any

from CadSeqProc.enhanced_geometry.intelligent_cad import IntelligentCAD
from CadSeqProc.enhanced_geometry.base import GeometricEntity

pytestmark = pytest.mark.asyncio

async def test_analyze_description(cad_system: IntelligentCAD, simple_cube_prompt: str):
    """Test basic geometry analysis."""
    result = await cad_system.analyze_description(simple_cube_prompt)
    assert result["status"] == "success"
    assert "geometry" in result
    assert "parameters" in result
    assert "validation" in result

async def test_complex_geometry(cad_system: IntelligentCAD, complex_shape_prompt: str):
    """Test complex geometry generation."""
    result = await cad_system.analyze_description(complex_shape_prompt)
    assert result["status"] == "success"
    assert "geometry" in result
    
    # Check specific features
    features = result["parameters"].get("features", [])
    feature_types = {f["type"] for f in features}
    assert "cylinder" in feature_types
    assert any("hole" in ft for ft in feature_types)

async def test_parametric_design(cad_system: IntelligentCAD, parametric_shape_prompt: str):
    """Test parametric design capabilities."""
    result = await cad_system.analyze_description(parametric_shape_prompt)
    assert result["status"] == "success"
    
    # Check parametric features
    params = result["parameters"]
    assert "adjustable_parameters" in params
    adjustable = params["adjustable_parameters"]
    assert any(p["name"] == "arm_length" for p in adjustable)

async def test_organic_shape(cad_system: IntelligentCAD, organic_shape_prompt: str):
    """Test organic shape generation."""
    result = await cad_system.analyze_description(organic_shape_prompt)
    assert result["status"] == "success"
    
    # Check organic features
    features = result["parameters"].get("features", [])
    assert any(f["type"] == "organic_surface" for f in features)

async def test_pattern_recognition(cad_system: IntelligentCAD):
    """Test pattern recognition in geometry."""
    # Create a geometry with patterns
    result = await cad_system.analyze_description(
        "Create a plate with a 4x4 grid of holes"
    )
    assert result["status"] == "success"
    
    # Analyze patterns
    patterns = cad_system._analyze_patterns(result["geometry"])
    assert len(patterns) > 0
    assert any(p.pattern_type == "grid" for p in patterns)

async def test_manufacturing_validation(cad_system: IntelligentCAD):
    """Test manufacturing validation."""
    # Create a geometry that might have manufacturing issues
    result = await cad_system.analyze_description(
        "Create a cube with 0.1mm thin walls"
    )
    assert result["status"] == "success"
    
    # Check validation results
    validation = result["validation"]
    assert not validation["valid"]  # Should fail due to thin walls
    assert any("wall thickness" in issue.lower() for issue in validation["issues"])

@pytest.mark.parametrize("input_params", [
    {
        "type": "cube",
        "dimensions": {"width": 10, "height": 10, "depth": 10}
    },
    {
        "type": "cylinder",
        "dimensions": {"height": 20, "diameter": 10}
    }
])
def test_parameter_mapping(cad_system: IntelligentCAD, input_params: Dict[str, Any]):
    """Test parameter mapping for different shapes."""
    cad_params = cad_system._map_llm_to_parameters({"features": [input_params]})
    assert isinstance(cad_params, dict)
    assert f"{input_params['type']}_width" in cad_params or f"{input_params['type']}_diameter" in cad_params

async def test_optimization(cad_system: IntelligentCAD):
    """Test geometry optimization."""
    # Create a geometry that needs optimization
    result = await cad_system.analyze_description(
        "Create a part with steep overhangs"
    )
    assert result["status"] == "success"
    
    # Get initial validation
    initial_validation = result["validation"]
    
    # Optimize geometry
    optimized = await cad_system.optimize_for_manufacturing(
        result["geometry"],
        initial_validation["issues"]
    )
    
    # Check optimized geometry
    final_validation = await cad_system.validate_design(optimized)
    assert final_validation["valid"] or len(final_validation["issues"]) < len(initial_validation["issues"])

async def test_error_handling(cad_system: IntelligentCAD):
    """Test error handling in CAD system."""
    # Test with invalid input
    result = await cad_system.analyze_description("")
    assert result["status"] == "error"
    
    # Test with nonsensical input
    result = await cad_system.analyze_description("make something impossible")
    assert result["status"] == "error"
    
    # Test with invalid parameters
    result = await cad_system.analyze_description("create a cube with -10cm sides")
    assert result["status"] == "error" 