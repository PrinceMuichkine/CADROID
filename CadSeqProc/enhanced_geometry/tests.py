"""
Test module for enhanced geometry system.
"""

import unittest
import numpy as np
from typing import List
from .base import Point, GeometricEntity
from .nurbs import NURBSCurve, NURBSSurface
from .organic import OrganicSurface, TwistDeformation
from .factory import OrganicShapeFactory
from .integration import GeometryAdapter, ShapeGenerator, ModelIntegration

class TestPoint(unittest.TestCase):
    def test_point_creation(self):
        p = Point(1.0, 2.0, 3.0)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 2.0)
        self.assertEqual(p.z, 3.0)
    
    def test_point_array_conversion(self):
        p = Point(1.0, 2.0, 3.0)
        arr = p.to_array()
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))
        
        p2 = Point.from_array(arr)
        self.assertEqual(p.x, p2.x)
        self.assertEqual(p.y, p2.y)
        self.assertEqual(p.z, p2.z)

class TestNURBS(unittest.TestCase):
    def test_curve_creation(self):
        points = [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(2.0, 0.0, 0.0)
        ]
        curve = NURBSCurve(points)
        
        # Test point sampling
        samples = curve.sample_points(5)
        self.assertEqual(len(samples), 5)
        
        # Test first and last points match control points
        np.testing.assert_array_almost_equal(
            samples[0].to_array(),
            points[0].to_array()
        )
        np.testing.assert_array_almost_equal(
            samples[-1].to_array(),
            points[-1].to_array()
        )
    
    def test_surface_creation(self):
        points = [
            [Point(0,0,0), Point(0,1,0)],
            [Point(1,0,0), Point(1,1,1)]
        ]
        surface = NURBSSurface(points)
        
        # Test point sampling
        samples = surface.sample_points(3)
        self.assertEqual(len(samples), 9)  # 3x3 grid
        
        # Test corners match control points
        np.testing.assert_array_almost_equal(
            samples[0].to_array(),
            points[0][0].to_array()
        )
        np.testing.assert_array_almost_equal(
            samples[-1].to_array(),
            points[-1][-1].to_array()
        )

class TestOrganicShape(unittest.TestCase):
    def test_deformation(self):
        # Create a simple surface
        points = [
            [Point(0,0,0), Point(0,1,0)],
            [Point(1,0,0), Point(1,1,0)]
        ]
        surface = NURBSSurface(points)
        organic = OrganicSurface([surface])
        
        # Apply twist deformation
        organic.apply_deformation('twist', {
            'axis': [0, 0, 1],
            'angle': np.pi/4,
            'center': [0.5, 0.5, 0]
        })
        
        # Sample points and verify they're not all in z=0 plane
        samples = organic.sample_points(5)
        z_coords = [p.z for p in samples]
        self.assertTrue(any(z != 0 for z in z_coords))

class TestFactory(unittest.TestCase):
    def test_flower_creation(self):
        factory = OrganicShapeFactory()
        flower = factory.create_flower(
            n_petals=5,
            petal_length=2.0,
            petal_width=1.0,
            center_radius=1.0,
            center_height=0.5
        )
        
        # Verify flower has correct number of surfaces
        # (1 center + n_petals surfaces)
        self.assertEqual(len(flower.control_surfaces), 6)
    
    def test_leaf_creation(self):
        factory = OrganicShapeFactory()
        leaf = factory.create_leaf(
            length=3.0,
            width=1.5,
            curve_factor=0.2
        )
        
        # Verify leaf has one surface
        self.assertEqual(len(leaf.control_surfaces), 1)
        
        # Sample points and verify dimensions
        samples = leaf.sample_points(10)
        xs = [p.x for p in samples]
        ys = [p.y for p in samples]
        
        self.assertLess(max(xs), 3.1)  # Length
        self.assertLess(max(ys), 0.8)  # Half width

class TestIntegration(unittest.TestCase):
    def test_shape_generation(self):
        integration = ModelIntegration()
        
        # Test flower generation
        result = integration.process_text_input("Create a flower with 5 petals")
        self.assertEqual(result['metadata']['generation_status'], 'success')
        self.assertEqual(result['cad_sequence']['type'], 'organic')
        
        # Test validation
        self.assertTrue(
            integration.validate_sequence(result['cad_sequence'])
        )
    
    def test_geometry_adapter(self):
        # Create a simple curve
        points = [Point(0,0,0), Point(1,1,0)]
        curve = NURBSCurve(points)
        
        # Convert to sequence and back
        adapter = GeometryAdapter()
        sequence = adapter.to_cad_sequence(curve)
        curve2 = adapter.from_cad_sequence(sequence)
        
        # Verify points match
        np.testing.assert_array_almost_equal(
            curve.control_points[0].to_array(),
            curve2.control_points[0].to_array()
        )

def run_tests():
    unittest.main() 