"""Tests for the pattern recognition module."""

import unittest
import numpy as np  # type: ignore
from typing import List, Dict, Any
from CadSeqProc.enhanced_geometry.pattern_recognition import (
    PatternRecognizer, PatternFeature, DesignPattern
)
from CadSeqProc.base import GeometricEntity, Point

class MockGeometricEntity(GeometricEntity):
    """Mock geometric entity for testing."""
    
    def __init__(self, center: Point, dimensions: Dict[str, float]):
        self.center = center
        self.dimensions = dimensions
        self.bounds = (
            Point(center.x - 1, center.y - 1, center.z - 1),
            Point(center.x + 1, center.y + 1, center.z + 1)
        )

class TestPatternRecognition(unittest.TestCase):
    """Test cases for pattern recognition functionality."""
    
    def setUp(self):
        """Set up test cases."""
        self.recognizer = PatternRecognizer()
        
        # Create test geometries
        self.linear_array = self._create_linear_array()
        self.circular_array = self._create_circular_array()
        
    def _create_linear_array(self) -> List[MockGeometricEntity]:
        """Create a linear array of mock entities."""
        entities = []
        for i in range(5):  # 5 entities in a line
            center = Point(i * 2.0, 0.0, 0.0)  # Spaced 2 units apart
            dimensions = {"width": 1.0, "height": 1.0, "depth": 1.0}
            entities.append(MockGeometricEntity(center, dimensions))
        return entities
        
    def _create_circular_array(self) -> List[MockGeometricEntity]:
        """Create a circular array of mock entities."""
        entities = []
        radius = 5.0
        num_entities = 8
        for i in range(num_entities):
            angle = (2 * np.pi * i) / num_entities
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            center = Point(x, y, 0.0)
            dimensions = {"width": 1.0, "height": 1.0, "depth": 1.0}
            entities.append(MockGeometricEntity(center, dimensions))
        return entities
        
    def test_linear_array_detection(self):
        """Test detection of linear array patterns."""
        patterns = self.recognizer.analyze_geometry(self.linear_array[0])
        
        # Should find one linear array pattern
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        self.assertEqual(pattern.features[0].pattern_type, "linear_array")
        
        # Check pattern properties
        feature = pattern.features[0]
        self.assertEqual(len(feature.instances), 4)  # 4 instances after base
        self.assertAlmostEqual(feature.parameters["spacing"], 2.0)  # 2 units spacing
        self.assertGreater(feature.confidence, 0.8)  # High confidence
        
    def test_circular_array_detection(self):
        """Test detection of circular array patterns."""
        patterns = self.recognizer.analyze_geometry(self.circular_array[0])
        
        # Should find one circular array pattern
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        self.assertEqual(pattern.features[0].pattern_type, "circular_array")
        
        # Check pattern properties
        feature = pattern.features[0]
        self.assertEqual(len(feature.instances), 7)  # 7 instances after base
        self.assertAlmostEqual(feature.parameters["radius"], 5.0)  # 5 units radius
        self.assertGreater(feature.confidence, 0.8)  # High confidence
        
    def test_pattern_similarity(self):
        """Test feature similarity comparison."""
        entity1 = MockGeometricEntity(
            Point(0.0, 0.0, 0.0),
            {"width": 1.0, "height": 1.0, "depth": 1.0}
        )
        entity2 = MockGeometricEntity(
            Point(2.0, 0.0, 0.0),
            {"width": 1.1, "height": 0.9, "depth": 1.0}
        )
        
        similarity = self.recognizer._compare_features(entity1, entity2)
        self.assertGreater(similarity, 0.8)  # Should be similar
        
    def test_pattern_relationships(self):
        """Test analysis of pattern relationships."""
        patterns = self.recognizer.analyze_geometry(self.linear_array[0])
        pattern = patterns[0]
        
        relationships = pattern.relationships
        self.assertGreater(len(relationships), 0)
        
        # Check spacing relationship
        spacing_rel = next(r for r in relationships if r["type"] == "spacing")
        self.assertAlmostEqual(spacing_rel["value"], 2.0)
        
    def test_manufacturing_notes(self):
        """Test generation of manufacturing notes."""
        patterns = self.recognizer.analyze_geometry(self.linear_array[0])
        pattern = patterns[0]
        
        self.assertIsNotNone(pattern.manufacturing_notes)
        self.assertIn("note", pattern.manufacturing_notes)
        
    def test_reuse_suggestions(self):
        """Test generation of reuse suggestions."""
        patterns = self.recognizer.analyze_geometry(self.linear_array[0])
        pattern = patterns[0]
        
        self.assertIsNotNone(pattern.reuse_suggestions)
        self.assertGreater(len(pattern.reuse_suggestions), 0)
        
if __name__ == '__main__':
    unittest.main() 