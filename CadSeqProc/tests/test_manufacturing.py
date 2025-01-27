"""Tests for manufacturing analysis module."""

import unittest
from typing import Dict, Any
import numpy as np  # type: ignore

from CadSeqProc.base import Point, GeometricEntity
from CadSeqProc.manufacturing.manufacturing_analyzer import (
    ManufacturingAnalyzer,
    ManufacturingConstraint,
    ManufacturingProcess,
    MaterialProperties
)

class MockGeometry(GeometricEntity):
    """Mock geometry for testing."""
    
    def __init__(self, volume: float = 1000.0, surface_area: float = 100.0):
        self.test_volume = volume
        self.test_surface_area = surface_area
        self.center = Point(0, 0, 0)
        self.bounds = (
            Point(-1, -1, -1),
            Point(1, 1, 1)
        )

class TestManufacturingAnalyzer(unittest.TestCase):
    """Test cases for manufacturing analysis."""
    
    def setUp(self):
        """Set up test cases."""
        self.analyzer = ManufacturingAnalyzer()
        self.test_geometry = MockGeometry()
        
    def test_process_initialization(self):
        """Test manufacturing process initialization."""
        processes = self.analyzer.processes
        
        # Check FDM process
        self.assertIn("3d_printing_fdm", processes)
        fdm = processes["3d_printing_fdm"]
        self.assertEqual(fdm.name, "FDM 3D Printing")
        self.assertEqual(fdm.type, "additive")
        self.assertIn("PLA", fdm.materials)
        
        # Check CNC process
        self.assertIn("cnc_milling", processes)
        cnc = processes["cnc_milling"]
        self.assertEqual(cnc.name, "CNC Milling")
        self.assertEqual(cnc.type, "subtractive")
        self.assertIn("aluminum", cnc.materials)
        
    def test_material_initialization(self):
        """Test material properties initialization."""
        materials = self.analyzer.materials
        
        # Check PLA properties
        self.assertIn("PLA", materials)
        pla = materials["PLA"]
        self.assertEqual(pla.type, "plastic")
        self.assertGreater(pla.tensile_strength, 0)
        
        # Check aluminum properties
        self.assertIn("aluminum_6061", materials)
        alu = materials["aluminum_6061"]
        self.assertEqual(alu.type, "metal")
        self.assertGreater(alu.yield_strength, 0)
        
    def test_manufacturability_analysis(self):
        """Test manufacturability analysis."""
        result = self.analyzer.analyze_manufacturability(
            self.test_geometry, "3d_printing_fdm")
        
        # Check result structure
        self.assertIn("process", result)
        self.assertIn("manufacturability_score", result)
        self.assertIn("constraint_violations", result)
        self.assertIn("cost_analysis", result)
        self.assertIn("recommendations", result)
        
        # Check score is within valid range
        self.assertGreaterEqual(result["manufacturability_score"], 0)
        self.assertLessEqual(result["manufacturability_score"], 100)
        
    def test_constraint_checking(self):
        """Test manufacturing constraint checking."""
        process = self.analyzer.processes["3d_printing_fdm"]
        violations = self.analyzer._check_constraints(
            self.test_geometry, process)
        
        # Should be a list
        self.assertIsInstance(violations, list)
        
        # Each violation should have required fields
        for violation in violations:
            self.assertIn("type", violation)
            self.assertIn("severity", violation)
            self.assertIn("message", violation)
            
    def test_cost_analysis(self):
        """Test manufacturing cost analysis."""
        process = self.analyzer.processes["3d_printing_fdm"]
        cost_analysis = self.analyzer._analyze_cost(
            self.test_geometry, process)
        
        # Check cost breakdown
        self.assertIn("material_cost", cost_analysis)
        self.assertIn("time_cost", cost_analysis)
        self.assertIn("setup_cost", cost_analysis)
        self.assertIn("total_cost", cost_analysis)
        
        # Total cost should be sum of components
        expected_total = (
            cost_analysis["material_cost"] +
            cost_analysis["time_cost"] +
            cost_analysis["setup_cost"]
        )
        self.assertAlmostEqual(cost_analysis["total_cost"], expected_total)
        
    def test_material_suggestions(self):
        """Test material suggestion system."""
        requirements = {
            "min_strength": 40.0,
            "max_temp": 100.0,
            "max_cost": 30.0
        }
        
        suggestions = self.analyzer.suggest_material(
            self.test_geometry, "3d_printing_fdm", requirements)
        
        # Should return a list of suggestions
        self.assertIsInstance(suggestions, list)
        
        # Each suggestion should have required fields
        for suggestion in suggestions:
            self.assertIn("material", suggestion)
            self.assertIn("score", suggestion)
            self.assertIn("reasons", suggestion)
            
            # Score should be within valid range
            self.assertGreaterEqual(suggestion["score"], 0)
            self.assertLessEqual(suggestion["score"], 100)
            
    def test_recommendation_generation(self):
        """Test manufacturing recommendation generation."""
        process = self.analyzer.processes["3d_printing_fdm"]
        violations = [
            {
                "type": "min_wall_thickness",
                "severity": "error",
                "message": "Wall too thin"
            }
        ]
        
        recommendations = self.analyzer._generate_recommendations(
            self.test_geometry, process, violations)
        
        # Should return a list of recommendations
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Each recommendation should be a string
        for recommendation in recommendations:
            self.assertIsInstance(recommendation, str)
            
    def test_material_score_calculation(self):
        """Test material scoring system."""
        material = self.analyzer.materials["PLA"]
        
        # Test with matching requirements
        matching_requirements = {
            "min_strength": 40.0,  # Below PLA strength
            "max_temp": 200.0,     # Above PLA melting point
            "max_cost": 30.0       # Above PLA cost
        }
        matching_score = self.analyzer._calculate_material_score(
            material, matching_requirements)
        self.assertEqual(matching_score, 100.0)
        
        # Test with non-matching requirements
        non_matching_requirements = {
            "min_strength": 100.0,  # Above PLA strength
            "max_temp": 50.0,       # Below PLA melting point
            "max_cost": 20.0        # Below PLA cost
        }
        non_matching_score = self.analyzer._calculate_material_score(
            material, non_matching_requirements)
        self.assertLess(non_matching_score, 100.0)

if __name__ == "__main__":
    unittest.main() 