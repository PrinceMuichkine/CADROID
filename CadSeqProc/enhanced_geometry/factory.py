"""
Factory for creating enhanced geometry objects.
"""

from typing import List, Dict, Any, Optional, Union, cast, Tuple, Sequence
import numpy as np
from numpy.typing import NDArray
from ..utility.logger import CLGLogger
from .base import NURBSEntity, BaseGeometry, Point
from .nurbs import NURBSCurve, NURBSSurface
from .organic import OrganicSurface

# Initialize logger with module name
logger = CLGLogger(__name__).configure_logger()

# Type alias for 3D point coordinates
Point = Tuple[float, float, float]

def convert_points_to_float_lists(points: Sequence[Point]) -> List[List[float]]:
    """Convert sequence of point tuples to list of float lists."""
    # Convert points to numpy array
    points_array = np.array(points, dtype=np.float64)
    # Convert back to nested list and cast to expected type
    return cast(List[List[float]], points_array.tolist())

class GeometryFactory:
    """Factory for creating geometric entities."""
    
    @staticmethod
    def create_curve_from_points(points: Sequence[Point], degree: int = 3) -> NURBSCurve:
        """Create a NURBS curve from points.
        
        Args:
            points: List of control points as (x,y,z) tuples
            degree: Degree of the curve
        Returns:
            NURBS curve
        """
        float_points = convert_points_to_float_lists(points)
        return NURBSCurve.from_points(float_points, degree=degree)
    
    @staticmethod
    def create_surface_from_points(points: Sequence[Sequence[Point]], degree_u: int = 3, degree_v: int = 3) -> NURBSSurface:
        """Create a NURBS surface from points.
        
        Args:
            points: Grid of control points as (x,y,z) tuples
            degree_u: Degree in u direction
            degree_v: Degree in v direction
        Returns:
            NURBS surface
        """
        # Convert each row of points
        float_points = [convert_points_to_float_lists(row) for row in points]
        return NURBSSurface.from_points(float_points, degree_u=degree_u, degree_v=degree_v)

class OrganicShapeFactory:
    """Factory for creating organic and parametric shapes."""
    
    def __init__(self) -> None:
        """Initialize the shape factory."""
        self.logger = logger
    
    def create_from_params(self, parameters: Dict[str, Any]) -> BaseGeometry:
        """Create geometry from parameters."""
        try:
            # Extract basic dimensions
            dimensions = parameters.get("dimensions", {})
            if not dimensions:
                # If no dimensions provided, try to get them from features
                if features := parameters.get("features", []):
                    for feature in features:
                        if feature.get("type") == "cube" and "dimensions" in feature:
                            dimensions = feature["dimensions"]
                            break
            
            # Create base geometry
            geometry = BaseGeometry(dimensions)
            
            # Set features
            geometry.features = parameters.get("features", [])
            
            # Apply manufacturing constraints
            if manufacturing := parameters.get("manufacturing", {}):
                if constraints := manufacturing.get("constraints", []):
                    for constraint in constraints:
                        if constraint["type"] == "min_wall_thickness":
                            geometry = geometry.thicken_walls(constraint["value"])
                        elif constraint["type"] == "max_overhang":
                            geometry = geometry.reduce_overhangs(constraint["value"])
            
            return geometry
            
        except Exception as e:
            self.logger.error(f"Error creating geometry: {str(e)}")
            # Return a default cube if something goes wrong
            return BaseGeometry({
                "width": 10.0,
                "height": 10.0,
                "depth": 10.0,
                "unit": "mm"
            })
    
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

    @staticmethod
    def create_surface(points: Sequence[Sequence[Point]], degree_u: int = 3, degree_v: int = 3) -> OrganicSurface:
        """Create an organic surface from points.
        
        Args:
            points: Grid of control points as (x,y,z) tuples
            degree_u: Degree in u direction
            degree_v: Degree in v direction
        Returns:
            Organic surface
        """
        # Convert each row of points
        float_points = [convert_points_to_float_lists(row) for row in points]
        # Create NURBS surface first
        nurbs_surface = NURBSSurface.from_points(float_points, degree_u=degree_u, degree_v=degree_v)
        # Create organic surface from NURBS surface
        return OrganicSurface(nurbs_surface) 