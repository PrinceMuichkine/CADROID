"""
Factory class for generating organic shapes from basic primitives.
"""

import numpy as np
from typing import List, Tuple, Optional
from ..geometry.nurbs import NURBSCurve, NURBSSurface
from ..geometry.organic import OrganicSurface
from ..sequence.transformation.deform import TwistDeformation, BendDeformation, TaperDeformation

class OrganicShapeFactory:
    """Factory for creating organic shapes."""
    
    @staticmethod
    def create_petal(length: float = 1.0,
                    width: float = 0.3,
                    curve_factor: float = 0.3,
                    twist_angle: float = 0.0) -> OrganicSurface:
        """Create a petal shape."""
        # Create control points for petal surface
        num_u, num_v = 5, 3
        control_points = []
        
        for i in range(num_u):
            row = []
            u = i / (num_u - 1)
            
            # Base curve follows a quadratic shape
            base_x = length * u
            base_y = 0
            base_z = curve_factor * u * (1 - u)
            
            for j in range(num_v):
                v = j / (num_v - 1) - 0.5
                # Width follows a cubic falloff
                width_factor = 1 - (2*v)**2
                
                x = base_x
                y = width * v * width_factor
                z = base_z * width_factor
                
                row.append((x, y, z))
            control_points.append(row)
        
        surface = NURBSSurface(control_points)
        organic_surface = OrganicSurface([surface])
        
        # Add twist if specified
        if twist_angle != 0:
            organic_surface.add_deformation(
                TwistDeformation(twist_angle, axis='x')
            )
        
        return organic_surface
    
    @staticmethod
    def create_leaf(length: float = 1.0,
                   width: float = 0.3,
                   curve_factor: float = 0.2,
                   vein_depth: float = 0.05) -> OrganicSurface:
        """Create a leaf shape."""
        num_u, num_v = 7, 5
        control_points = []
        
        for i in range(num_u):
            row = []
            u = i / (num_u - 1)
            
            # Base curve with upward curve
            base_x = length * u
            base_y = 0
            base_z = curve_factor * np.sin(np.pi * u)
            
            for j in range(num_v):
                v = j / (num_v - 1) - 0.5
                # Width follows an elliptical shape
                width_factor = np.sqrt(1 - (2*v)**2)
                # Add vein pattern
                vein_z = vein_depth * np.cos(np.pi * v)
                
                x = base_x
                y = width * v * width_factor * (1 - u**0.5)  # Taper towards tip
                z = base_z + vein_z * width_factor
                
                row.append((x, y, z))
            control_points.append(row)
        
        surface = NURBSSurface(control_points)
        return OrganicSurface([surface])
    
    @staticmethod
    def create_flower(num_petals: int = 5,
                     petal_length: float = 1.0,
                     petal_width: float = 0.3,
                     center_radius: float = 0.2) -> List[OrganicSurface]:
        """Create a flower with multiple petals."""
        shapes = []
        
        # Create center
        center_points = []
        num_u, num_v = 5, 8
        for i in range(num_u):
            row = []
            u = i / (num_u - 1)
            radius = center_radius * (1 - u**2)
            height = center_radius * 0.5 * u
            
            for j in range(num_v):
                angle = 2 * np.pi * j / (num_v - 1)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = height
                row.append((x, y, z))
            center_points.append(row)
        
        center_surface = NURBSSurface(center_points)
        shapes.append(OrganicSurface([center_surface]))
        
        # Create and position petals
        for i in range(num_petals):
            angle = 2 * np.pi * i / num_petals
            petal = OrganicShapeFactory.create_petal(
                length=petal_length,
                width=petal_width,
                curve_factor=0.3,
                twist_angle=0.2
            )
            
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, center_radius * c],
                [s, c, 0, center_radius * s],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            shapes.append(petal.transform(transform))
        
        return shapes
    
    @staticmethod
    def create_tree(trunk_height: float = 2.0,
                   trunk_radius: float = 0.2,
                   num_branches: int = 5,
                   leaf_size: float = 0.5) -> List[OrganicSurface]:
        """Create a simple tree structure."""
        shapes = []
        
        # Create trunk
        trunk_points = []
        num_u, num_v = 5, 8
        for i in range(num_u):
            row = []
            u = i / (num_u - 1)
            radius = trunk_radius * (1 - 0.3 * u)  # Slight taper
            height = trunk_height * u
            
            for j in range(num_v):
                angle = 2 * np.pi * j / (num_v - 1)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = height
                row.append((x, y, z))
            trunk_points.append(row)
        
        trunk_surface = NURBSSurface(trunk_points)
        shapes.append(OrganicSurface([trunk_surface]))
        
        # Add branches with leaves
        for i in range(num_branches):
            height_factor = 0.3 + 0.7 * i / (num_branches - 1)
            angle = 2 * np.pi * i / num_branches
            
            # Create leaf
            leaf = OrganicShapeFactory.create_leaf(
                length=leaf_size,
                width=leaf_size * 0.4,
                curve_factor=0.2
            )
            
            # Position leaf
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, trunk_radius * 2 * c],
                [s, c, 0, trunk_radius * 2 * s],
                [0, 0, 1, trunk_height * height_factor],
                [0, 0, 0, 1]
            ])
            
            shapes.append(leaf.transform(transform))
        
        return shapes
    
    @staticmethod
    def create_vine(control_points: List[Tuple[float, float, float]],
                   thickness: float = 0.1,
                   num_leaves: int = 5,
                   leaf_size: float = 0.3) -> List[OrganicSurface]:
        """Create a vine with leaves."""
        shapes = []
        
        # Create vine curve
        vine_curve = NURBSCurve.from_points(control_points)
        
        # Create vine surface
        vine_points = []
        num_u, num_v = len(control_points), 8
        
        for i in range(num_u):
            row = []
            u = i / (num_u - 1)
            point = vine_curve.evaluate(u)
            
            for j in range(num_v):
                angle = 2 * np.pi * j / (num_v - 1)
                x = point[0] + thickness * np.cos(angle)
                y = point[1] + thickness * np.sin(angle)
                z = point[2]
                row.append((x, y, z))
            vine_points.append(row)
        
        vine_surface = NURBSSurface(vine_points)
        shapes.append(OrganicSurface([vine_surface]))
        
        # Add leaves along the vine
        for i in range(num_leaves):
            t = i / (num_leaves - 1)
            point = vine_curve.evaluate(t)
            
            # Create leaf
            leaf = OrganicShapeFactory.create_leaf(
                length=leaf_size,
                width=leaf_size * 0.4,
                curve_factor=0.2
            )
            
            # Calculate orientation based on curve tangent
            delta = 0.01
            next_point = vine_curve.evaluate(min(t + delta, 1.0))
            tangent = np.array(next_point) - np.array(point)
            tangent /= np.linalg.norm(tangent)
            
            # Create transformation matrix
            up = np.array([0, 0, 1])
            right = np.cross(tangent, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, tangent)
            
            transform = np.array([
                [right[0], tangent[0], up[0], point[0]],
                [right[1], tangent[1], up[1], point[1]],
                [right[2], tangent[2], up[2], point[2]],
                [0, 0, 0, 1]
            ])
            
            shapes.append(leaf.transform(transform))
        
        return shapes 