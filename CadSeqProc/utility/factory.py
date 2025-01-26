"""
Factory utility for creating complex geometric shapes.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from CadSeqProc.enhanced_geometry.base import BaseGeometry
from CadSeqProc.enhanced_geometry.nurbs import NURBSCurve, NURBSSurface
from CadSeqProc.enhanced_geometry.organic import OrganicSurface
from CadSeqProc.utility.shape_factory import (
    create_circle, create_sphere, create_cylinder,
    create_cone, create_torus, create_box
)
from CadSeqProc.utility.logger import setup_logger

logger = setup_logger(__name__)

class ShapeFactory:
    """Factory for creating complex geometric shapes."""
    
    @staticmethod
    def create_shape(shape_type: str, params: Dict[str, Any]) -> BaseGeometry:
        """Create shape from type and parameters."""
        try:
            if shape_type == 'organic':
                return ShapeFactory._create_organic_shape(params)
            elif shape_type == 'mechanical':
                return ShapeFactory._create_mechanical_shape(params)
            elif shape_type == 'architectural':
                return ShapeFactory._create_architectural_shape(params)
            else:
                return ShapeFactory._create_basic_shape(shape_type, params)
        except Exception as e:
            logger.error(f"Error creating shape: {str(e)}")
            # Return simple fallback shape
            return create_sphere(1.0)
    
    @staticmethod
    def _create_organic_shape(params: Dict[str, Any]) -> BaseGeometry:
        """Create organic shape from parameters."""
        shape_type = params.get('type', 'generic')
        scale = params.get('scale', 1.0)
        
        if shape_type == 'flower':
            return ShapeFactory._create_flower(
                num_petals=params.get('num_petals', 5),
                petal_length=0.5 * scale,
                petal_width=0.2 * scale,
                center_radius=0.2 * scale
            )
        elif shape_type == 'leaf':
            return ShapeFactory._create_leaf(
                length=1.0 * scale,
                width=0.5 * scale,
                vein_depth=0.05 * scale
            )
        else:
            # Create generic organic shape
            base = create_sphere(0.5 * scale)
            organic = OrganicSurface(base)
            organic.add_random_deformation(0.3)
            return organic
    
    @staticmethod
    def _create_mechanical_shape(params: Dict[str, Any]) -> BaseGeometry:
        """Create mechanical shape from parameters."""
        shape_type = params.get('type', 'generic')
        scale = params.get('scale', 1.0)
        
        if shape_type == 'gear':
            return ShapeFactory._create_gear(
                outer_radius=0.5 * scale,
                inner_radius=0.3 * scale,
                thickness=0.2 * scale,
                num_teeth=params.get('num_teeth', 12)
            )
        elif shape_type == 'bolt':
            return ShapeFactory._create_bolt(
                head_radius=0.3 * scale,
                shaft_radius=0.15 * scale,
                length=1.0 * scale,
                thread_pitch=0.1 * scale
            )
        else:
            # Create generic mechanical shape
            return create_cylinder(0.5 * scale, 1.0 * scale)
    
    @staticmethod
    def _create_architectural_shape(params: Dict[str, Any]) -> BaseGeometry:
        """Create architectural shape from parameters."""
        shape_type = params.get('type', 'generic')
        scale = params.get('scale', 1.0)
        
        if shape_type == 'column':
            return ShapeFactory._create_column(
                height=2.0 * scale,
                radius=0.3 * scale,
                capital_height=0.4 * scale,
                base_height=0.3 * scale
            )
        elif shape_type == 'arch':
            return ShapeFactory._create_arch(
                width=2.0 * scale,
                height=1.5 * scale,
                depth=0.5 * scale,
                thickness=0.2 * scale
            )
        else:
            # Create generic architectural shape
            return create_box(1.0 * scale, 2.0 * scale, 1.0 * scale)
    
    @staticmethod
    def _create_basic_shape(shape_type: str, params: Dict[str, Any]) -> BaseGeometry:
        """Create basic shape from type and parameters."""
        scale = params.get('scale', 1.0)
        center = params.get('center', [0.0, 0.0, 0.0])
        
        if shape_type == 'sphere':
            return create_sphere(0.5 * scale, center)
        elif shape_type == 'cylinder':
            return create_cylinder(0.5 * scale, 1.0 * scale, center)
        elif shape_type == 'cone':
            return create_cone(0.5 * scale, 1.0 * scale, center)
        elif shape_type == 'torus':
            return create_torus(0.5 * scale, 0.2 * scale, center)
        elif shape_type == 'box':
            return create_box(1.0 * scale, 1.0 * scale, 1.0 * scale, center)
        else:
            return create_sphere(0.5 * scale, center)
    
    @staticmethod
    def _create_flower(num_petals: int, petal_length: float,
                      petal_width: float, center_radius: float) -> BaseGeometry:
        """Create flower shape."""
        # Create center
        center = create_sphere(center_radius)
        center_organic = OrganicSurface(center)
        center_organic.add_bumps(20, center_radius * 0.1, center_radius * 0.1)
        
        # Create petals
        petals = []
        for i in range(num_petals):
            angle = (2 * np.pi * i) / num_petals
            
            # Create petal curve
            control_points = [
                [0, 0, 0],
                [petal_length * 0.3, petal_width * 0.5, 0],
                [petal_length * 0.7, petal_width * 0.5, 0],
                [petal_length, 0, 0]
            ]
            curve = NURBSCurve(control_points)
            
            # Create petal surface
            surface = curve.sweep(petal_width)
            petal = OrganicSurface(surface)
            petal.add_random_deformation(0.2)
            petal.rotate(angle)
            
            petals.append(petal)
        
        # Combine shapes
        return OrganicSurface.combine([center_organic] + petals)
    
    @staticmethod
    def _create_leaf(length: float, width: float, vein_depth: float) -> BaseGeometry:
        """Create leaf shape."""
        # Create main surface control points
        points = []
        for i in range(5):
            u = i / 4
            row = []
            for j in range(3):
                v = j / 2 - 0.5
                
                # Create leaf shape
                x = length * u
                y = width * v * (1 - u) * (1 - u)
                z = vein_depth * np.sin(np.pi * u) * (1 - abs(v))
                
                row.append([x, y, z])
            points.append(row)
        
        # Create surface
        surface = NURBSSurface(points)
        leaf = OrganicSurface(surface)
        
        # Add random deformation
        leaf.add_random_deformation(0.1)
        
        return leaf
    
    @staticmethod
    def _create_gear(outer_radius: float, inner_radius: float,
                    thickness: float, num_teeth: int) -> BaseGeometry:
        """Create gear shape."""
        # Create base cylinder
        base = create_cylinder(inner_radius, thickness)
        
        # Create teeth
        teeth_points = []
        for i in range(num_teeth):
            angle = (2 * np.pi * i) / num_teeth
            
            # Create tooth profile
            tooth_points = [
                [inner_radius * np.cos(angle), inner_radius * np.sin(angle), 0],
                [outer_radius * np.cos(angle - 0.1), outer_radius * np.sin(angle - 0.1), 0],
                [outer_radius * np.cos(angle), outer_radius * np.sin(angle), 0],
                [outer_radius * np.cos(angle + 0.1), outer_radius * np.sin(angle + 0.1), 0],
                [inner_radius * np.cos(angle), inner_radius * np.sin(angle), 0]
            ]
            
            # Extrude tooth
            for z in [0, thickness]:
                row = []
                for point in tooth_points:
                    row.append([point[0], point[1], z])
                teeth_points.append(row)
        
        teeth = NURBSSurface(teeth_points)
        return teeth
    
    @staticmethod
    def _create_bolt(head_radius: float, shaft_radius: float,
                    length: float, thread_pitch: float) -> BaseGeometry:
        """Create bolt shape."""
        # Create head
        head = create_cylinder(head_radius, head_radius * 0.8)
        
        # Create shaft
        shaft = create_cylinder(shaft_radius, length)
        
        # Create thread helix
        helix_points = []
        turns = int(length / thread_pitch)
        points_per_turn = 20
        
        for i in range(turns * points_per_turn):
            t = i / points_per_turn
            angle = 2 * np.pi * t
            x = shaft_radius * np.cos(angle)
            y = shaft_radius * np.sin(angle)
            z = t * thread_pitch
            helix_points.append([x, y, z])
        
        thread = NURBSCurve(helix_points)
        
        # TODO: Combine shapes properly
        return shaft
    
    @staticmethod
    def _create_column(height: float, radius: float,
                      capital_height: float, base_height: float) -> BaseGeometry:
        """Create column shape."""
        # Create base
        base = create_cylinder(radius * 1.2, base_height)
        
        # Create shaft
        shaft = create_cylinder(radius, height - capital_height - base_height)
        
        # Create capital
        capital = create_cylinder(radius * 1.3, capital_height)
        
        # TODO: Combine shapes properly
        return shaft
    
    @staticmethod
    def _create_arch(width: float, height: float,
                    depth: float, thickness: float) -> BaseGeometry:
        """Create arch shape."""
        # Create arch curve
        curve_points = []
        for t in np.linspace(0, np.pi, 20):
            x = width/2 * np.cos(t)
            y = height * np.sin(t)
            curve_points.append([x, y, 0])
        
        curve = NURBSCurve(curve_points)
        
        # Create arch surface
        arch = curve.sweep(depth)
        
        # TODO: Add thickness
        return arch 