"""
Factory for creating geometric and organic shapes.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from .base import BaseGeometry, Point, BoundingBox
from .nurbs import NURBSCurve, NURBSSurface
from .organic import OrganicSurface
from ..utility.logger import setup_logger

logger = setup_logger(__name__)

class OrganicShapeFactory:
    """Factory for creating organic shapes."""
    
    def create_from_params(self, params: Dict[str, Any]) -> BaseGeometry:
        """Create shape from parameters."""
        try:
            # Extract basic parameters
            shape_type = params.get('shape_type', 'generic')
            scale = params.get('scale_factor', 1.0)
            detail_level = params.get('detail_level', 0.5)
            
            if shape_type == 'flower':
                return self._create_flower(
                    num_petals=params.get('num_petals', 5),
                    petal_length=0.5 * scale,
                    petal_width=0.2 * scale,
                    detail_level=detail_level
                )
            elif shape_type == 'leaf':
                return self._create_leaf(
                    length=1.0 * scale,
                    width=0.5 * scale,
                    detail_level=detail_level
                )
            else:
                return self._create_generic_organic(
                    scale=scale,
                    detail_level=detail_level
                )
                
        except Exception as e:
            logger.error(f"Error creating organic shape: {str(e)}")
            # Return a simple fallback shape
            return self._create_generic_organic(scale=1.0, detail_level=0.3)
    
    def _create_flower(self, num_petals: int, petal_length: float,
                      petal_width: float, detail_level: float) -> OrganicSurface:
        """Create a flower shape."""
        # Create center
        center_radius = petal_length * 0.2
        center = self._create_center(radius=center_radius, detail_level=detail_level)
        
        # Create petals
        petals = []
        for i in range(num_petals):
            angle = (2 * np.pi * i) / num_petals
            petal = self._create_petal(
                length=petal_length,
                width=petal_width,
                angle=angle,
                detail_level=detail_level
            )
            petals.append(petal)
        
        # Combine shapes
        return OrganicSurface.combine([center] + petals)
    
    def _create_leaf(self, length: float, width: float,
                    detail_level: float) -> OrganicSurface:
        """Create a leaf shape."""
        # Create main surface
        control_points = self._generate_leaf_points(length, width)
        surface = NURBSSurface.from_points(control_points)
        
        # Add veins
        veins = self._create_leaf_veins(length, width, detail_level)
        
        # Combine and add organic deformation
        combined = OrganicSurface.from_nurbs(surface)
        for vein in veins:
            combined.add_feature(vein)
        
        return combined
    
    def _create_generic_organic(self, scale: float,
                              detail_level: float) -> OrganicSurface:
        """Create a generic organic shape."""
        # Create base sphere
        radius = 0.5 * scale
        sphere = self._create_sphere(radius)
        
        # Add organic deformation
        surface = OrganicSurface.from_nurbs(sphere)
        surface.add_random_deformation(intensity=detail_level)
        
        return surface
    
    def _create_center(self, radius: float, detail_level: float) -> OrganicSurface:
        """Create flower center."""
        # Create base sphere
        sphere = self._create_sphere(radius)
        
        # Add organic texture
        surface = OrganicSurface.from_nurbs(sphere)
        surface.add_bumps(
            count=int(20 * detail_level),
            height=radius * 0.1,
            radius=radius * 0.1
        )
        
        return surface
    
    def _create_petal(self, length: float, width: float,
                     angle: float, detail_level: float) -> OrganicSurface:
        """Create a single petal."""
        # Create base curve
        control_points = [
            [0, 0, 0],
            [length * 0.3, width * 0.5, 0],
            [length * 0.7, width * 0.5, 0],
            [length, 0, 0]
        ]
        curve = NURBSCurve.from_points(control_points)
        
        # Create surface by sweeping
        surface = curve.sweep(width)
        
        # Add organic deformation
        organic = OrganicSurface.from_nurbs(surface)
        organic.add_random_deformation(intensity=detail_level * 0.3)
        
        # Rotate to position
        organic.rotate(angle)
        
        return organic
    
    def _create_sphere(self, radius: float) -> NURBSSurface:
        """Create a NURBS sphere."""
        # Create control points for sphere
        u_count, v_count = 10, 10
        control_points = []
        
        for i in range(u_count):
            u = (i / (u_count - 1)) * 2 * np.pi
            row = []
            for j in range(v_count):
                v = (j / (v_count - 1)) * np.pi
                x = radius * np.sin(v) * np.cos(u)
                y = radius * np.sin(v) * np.sin(u)
                z = radius * np.cos(v)
                row.append([x, y, z])
            control_points.append(row)
        
        return NURBSSurface.from_points(control_points)
    
    def _generate_leaf_points(self, length: float,
                            width: float) -> List[List[List[float]]]:
        """Generate control points for a leaf shape."""
        # Create control point grid
        u_count, v_count = 5, 3
        control_points = []
        
        for i in range(u_count):
            u = i / (u_count - 1)
            row = []
            for j in range(v_count):
                v = j / (v_count - 1) - 0.5
                
                # Create leaf shape
                x = length * u
                y = width * v * (1 - u) * (1 - u)
                z = 0.0
                
                row.append([x, y, z])
            control_points.append(row)
        
        return control_points
    
    def _create_leaf_veins(self, length: float, width: float,
                          detail_level: float) -> List[NURBSCurve]:
        """Create leaf vein curves."""
        veins = []
        
        # Main vein
        main_vein = NURBSCurve.from_points([
            [0, 0, 0],
            [length * 0.3, 0, 0],
            [length * 0.7, 0, 0],
            [length, 0, 0]
        ])
        veins.append(main_vein)
        
        # Side veins
        num_side_veins = int(5 * detail_level)
        for i in range(num_side_veins):
            t = (i + 1) / (num_side_veins + 1)
            start = main_vein.point_at(t)
            
            # Create side vein on both sides
            for side in [-1, 1]:
                end = [
                    start[0] + length * 0.2,
                    side * width * 0.4 * (1 - t),
                    0
                ]
                vein = NURBSCurve.from_points([start, end])
                veins.append(vein)
        
        return veins 