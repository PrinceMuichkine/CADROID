"""Unit tests for manufacturing analyzer."""

import pytest
from typing import Dict, Any

from CadSeqProc.manufacturing.manufacturing_analyzer import (
    ManufacturingAnalyzer,
    ManufacturingConstraint,
    ManufacturingProcess,
    MaterialProperties
)

pytestmark = pytest.mark.asyncio

@pytest.fixture
def analyzer(llm_client) -> ManufacturingAnalyzer:
    """Create manufacturing analyzer for testing."""
    return ManufacturingAnalyzer(llm_client)

@pytest.fixture
def test_geometry(cad_system) -> Any:
    """Create test geometry."""
    return cad_system.shape_factory.create_cube(10)

async def test_basic_analysis(analyzer: ManufacturingAnalyzer, test_geometry):
    """Test basic manufacturability analysis."""
    result = await analyzer.analyze_manufacturability(test_geometry, "3d_printing_fdm")
    assert result["status"] == "success"
    assert "score" in result
    assert "violations" in result
    assert "recommendations" in result

@pytest.mark.parametrize("process_name,expected_fields", [
    ("3d_printing_fdm", ["min_wall_thickness", "max_overhang"]),
    ("cnc_milling", ["min_internal_radius", "max_depth_to_width"])
])
async def test_process_specific_analysis(
    analyzer: ManufacturingAnalyzer,
    test_geometry,
    process_name: str,
    expected_fields: list
):
    """Test analysis for different manufacturing processes."""
    result = await analyzer.analyze_manufacturability(test_geometry, process_name)
    assert result["status"] == "success"
    
    # Check process-specific constraints
    constraints = result["analyses"][process_name]["constraints"]
    constraint_types = [c["type"] for c in constraints]
    for field in expected_fields:
        assert field in constraint_types

async def test_thin_wall_detection(analyzer: ManufacturingAnalyzer, cad_system):
    """Test detection of thin walls."""
    # Create geometry with thin walls
    thin_geometry = cad_system.shape_factory.create_cube(10, wall_thickness=0.1)
    result = await analyzer.analyze_manufacturability(thin_geometry, "3d_printing_fdm")
    
    # Should detect thin wall violation
    violations = result["violations"]
    assert any(v["type"] == "min_wall_thickness" for v in violations)
    assert result["score"] < 100

async def test_overhang_detection(analyzer: ManufacturingAnalyzer, cad_system):
    """Test detection of steep overhangs."""
    # Create geometry with steep overhang
    overhang_geometry = cad_system.shape_factory.create_overhang(angle=60)
    result = await analyzer.analyze_manufacturability(overhang_geometry, "3d_printing_fdm")
    
    # Should detect overhang violation
    violations = result["violations"]
    assert any(v["type"] == "max_overhang" for v in violations)
    assert result["score"] < 100

def test_material_suggestions(analyzer: ManufacturingAnalyzer, test_geometry):
    """Test material suggestion system."""
    suggestions = analyzer.suggest_material(
        test_geometry,
        "3d_printing_fdm",
        {
            "min_strength": 40.0,
            "max_temp": 100.0,
            "max_cost": 30.0
        }
    )
    
    assert len(suggestions) > 0
    for suggestion in suggestions:
        assert "material" in suggestion
        assert "score" in suggestion
        assert "reasons" in suggestion

@pytest.mark.parametrize("dimension,expected_time", [
    ((10, 10, 10), 1000),  # 1000mm³ should take X time
    ((20, 20, 20), 8000)   # 8000mm³ should take ~8X time
])
def test_manufacturing_time_estimation(
    analyzer: ManufacturingAnalyzer,
    cad_system,
    dimension: tuple,
    expected_time: float
):
    """Test manufacturing time estimation."""
    geometry = cad_system.shape_factory.create_cube(*dimension)
    time = analyzer._estimate_manufacturing_time(
        geometry,
        analyzer.processes["3d_printing_fdm"],
        volume=dimension[0] * dimension[1] * dimension[2],
        surface_area=6 * dimension[0] * dimension[1]
    )
    assert abs(time - expected_time) / expected_time < 0.2  # Within 20% accuracy

def test_cost_analysis(analyzer: ManufacturingAnalyzer, test_geometry):
    """Test manufacturing cost analysis."""
    cost_analysis = analyzer._analyze_cost(
        test_geometry,
        analyzer.processes["3d_printing_fdm"]
    )
    
    assert "material_cost" in cost_analysis
    assert "time_cost" in cost_analysis
    assert "setup_cost" in cost_analysis
    assert "total_cost" in cost_analysis
    assert cost_analysis["total_cost"] == sum([
        cost_analysis["material_cost"],
        cost_analysis["time_cost"],
        cost_analysis["setup_cost"]
    ])

async def test_error_handling(analyzer: ManufacturingAnalyzer, test_geometry):
    """Test error handling in manufacturing analyzer."""
    # Test with invalid process
    result = await analyzer.analyze_manufacturability(test_geometry, "invalid_process")
    assert result["status"] == "error"
    assert "message" in result
    
    # Test with invalid geometry
    result = await analyzer.analyze_manufacturability(None, "3d_printing_fdm")
    assert result["status"] == "error"
    assert "message" in result

def test_constraint_validation(analyzer: ManufacturingAnalyzer, test_geometry):
    """Test manufacturing constraint validation."""
    # Create test constraint
    constraint = ManufacturingConstraint(
        constraint_type="min_wall_thickness",
        value=0.8,
        unit="mm",
        description="Minimum wall thickness",
        severity="error"
    )
    
    # Test constraint check
    violation = analyzer._check_single_constraint(test_geometry, constraint)
    assert isinstance(violation, dict) or violation is None 