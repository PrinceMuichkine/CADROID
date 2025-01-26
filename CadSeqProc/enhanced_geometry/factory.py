"""
Factory class for generating organic shapes from basic primitives.
Supports both geometric and organic shape generation with advanced patterns.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from .base import Point, GeometricEntity
from .nurbs import NURBSCurve, NURBSSurface
from .organic import OrganicSurface
from .parametric import FlowerPetal, PatternGenerator, OrganicPatternFactory

class OrganicShapeFactory:
    """Factory for creating organic shapes with advanced patterns."""
    
    @staticmethod
    def create_petal(
        length: float,
        width: float,
        curve_factor: float = 0.3,
        twist_angle: Optional[float] = None
    ) -> OrganicSurface:
        """Generate a petal shape with optional twist."""
        # Create base petal curve
        petal = FlowerPetal(length, width, curve_factor)
        curve = petal.to_nurbs()
        
        # Create surface by extruding curve
        control_points = []
        for t in np.linspace(0, 1, 10):
            points = curve.sample_points(20)
            # Add thickness variation
            thickness = 0.1 * (1 - t)  # Taper towards tip
            offset = np.array([0, 0, thickness])
            control_points.append([Point(p.x, p.y, p.z + offset[2]) for p in points])
        
        surface = NURBSSurface.from_points(control_points)
        
        # Apply twist if specified
        if twist_angle is not None:
            return OrganicSurface([surface], {
                'twist': {
                    'axis': [0, 0, 1],
                    'angle': twist_angle,
                    'center': [0, 0, 0]
                }
            })
        
        return OrganicSurface([surface])
    
    @staticmethod
    def create_flower(
        num_petals: int,
        petal_length: float,
        petal_width: float,
        center_radius: float = 0.2,
        petal_curve_factor: float = 0.3
    ) -> List[OrganicSurface]:
        """Create a flower with multiple petals."""
        # Create center disk
        center_points = []
        for r in np.linspace(0, center_radius, 5):
            circle_points = []
            for theta in np.linspace(0, 2*np.pi, 20):
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = 0.1 * (1 - r/center_radius)  # Slight dome shape
                circle_points.append(Point(x, y, z))
            center_points.append(circle_points)
        
        center_surface = NURBSSurface.from_points(center_points)
        center = OrganicSurface([center_surface])
        
        # Create and arrange petals
        petals = []
        for i in range(num_petals):
            angle = 2 * np.pi * i / num_petals
            # Add variation to petal parameters
            length_var = petal_length * (1 + 0.1 * np.sin(3*angle))
            width_var = petal_width * (1 + 0.1 * np.cos(2*angle))
            
            petal = OrganicShapeFactory.create_petal(
                length_var,
                width_var,
                petal_curve_factor,
                twist_angle=0.2 * np.sin(angle)
            )
            
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, center_radius * c],
                [s, c, 0, center_radius * s],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            petals.append(petal.transform(transform))
        
        return [center] + petals
    
    @staticmethod
    def create_leaf(
        length: float,
        width: float,
        curve_factor: float = 0.3,
        vein_depth: float = 0.05
    ) -> OrganicSurface:
        """Create a leaf shape with veins."""
        # Create base leaf curve
        leaf = FlowerPetal(length, width, curve_factor, harmonics=2)
        curve = leaf.to_nurbs()
        
        # Create surface with vein pattern
        control_points = []
        for t in np.linspace(0, 1, 10):
            points = curve.sample_points(20)
            # Add vein pattern
            vein = vein_depth * np.sin(5 * np.pi * t) * np.exp(-2*t)
            offset = np.array([0, 0, vein])
            control_points.append([Point(p.x, p.y, p.z + offset[2]) for p in points])
        
        surface = NURBSSurface.from_points(control_points)
        return OrganicSurface([surface])
    
    @staticmethod
    def create_tree(
        trunk_height: float = 2.0,
        trunk_radius: float = 0.2,
        num_branches: int = 5,
        leaf_size: float = 0.5
    ) -> List[OrganicSurface]:
        """Create a simple tree with trunk and leaves."""
        # Create trunk
        trunk_points = []
        for h in np.linspace(0, trunk_height, 10):
            circle_points = []
            radius = trunk_radius * (1 - 0.3 * h/trunk_height)  # Taper trunk
            for theta in np.linspace(0, 2*np.pi, 20):
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                circle_points.append(Point(x, y, h))
            trunk_points.append(circle_points)
        
        trunk_surface = NURBSSurface.from_points(trunk_points)
        trunk = OrganicSurface([trunk_surface])
        
        # Add branches with leaves
        branches = []
        for i in range(num_branches):
            height = trunk_height * (0.3 + 0.7 * i/num_branches)
            angle = 2 * np.pi * i / num_branches
            
            leaf = OrganicShapeFactory.create_leaf(
                leaf_size,
                leaf_size * 0.4,
                curve_factor=0.2
            )
            
            # Position leaf
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, trunk_radius * 2 * c],
                [s, c, 0, trunk_radius * 2 * s],
                [0, 0, 1, height],
                [0, 0, 0, 1]
            ])
            
            branches.append(leaf.transform(transform))
        
        return [trunk] + branches
    
    @staticmethod
    def create_vine(
        points: List[Point],
        thickness: float,
        n_leaves: int,
        leaf_size: float
    ) -> OrganicSurface:
        """
        Create a vine with leaves along a curve.
        
        Args:
            points: Control points for vine curve
            thickness: Thickness of vine
            n_leaves: Number of leaves
            leaf_size: Size of leaves
        """
        surfaces = []
        
        # Create vine curve
        vine_curve = NURBSCurve(points)
        
        # Create vine surface
        n_points_u = len(points)
        n_points_v = 8
        vine_points = []
        
        for i in range(n_points_u):
            u = i / (n_points_u - 1)
            row = []
            for j in range(n_points_v):
                v = j / (n_points_v - 1)
                theta = 2 * np.pi * v
                
                # Get point on curve
                p = vine_curve.evaluate(u)
                
                # Create circle around curve point
                r = thickness
                x = p.x + r * np.cos(theta)
                y = p.y + r * np.sin(theta)
                z = p.z
                
                row.append(Point(x, y, z))
            vine_points.append(row)
        
        surfaces.append(NURBSSurface(vine_points))
        
        # Add leaves along vine
        for i in range(n_leaves):
            u = i / (n_leaves - 1)
            p = vine_curve.evaluate(u)
            
            leaf = OrganicShapeFactory.create_leaf(
                leaf_size,
                leaf_size * 0.4,
                curve_factor=0.2
            )
            
            # Create transformation to position leaf
            # This would need proper orientation based on curve tangent
            transform = np.array([
                [1,     0,     0,  p.x],
                [0,     1,     0,  p.y],
                [0,     0,     1,  p.z],
                [0,     0,     0,  1]
            ])
            
            transformed_leaf = leaf.transform(transform)
            surfaces.extend(transformed_leaf.control_surfaces)
        
        return OrganicSurface(surfaces) 