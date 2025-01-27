"""
Base classes and interfaces for enhanced geometry system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, cast, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class Point:
    """3D point representation."""
    x: float
    y: float
    z: float

class BoundingBox:
    """Axis-aligned bounding box."""
    
    def __init__(self, min_point: Point, max_point: Point):
        self.min_point = min_point
        self.max_point = max_point
    
    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Get box dimensions (width, height, depth)."""
        return (
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z
        )
    
    @property
    def volume(self) -> float:
        """Calculate box volume."""
        w, h, d = self.dimensions
        return w * h * d
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside box."""
        return (
            self.min_point.x <= point.x <= self.max_point.x and
            self.min_point.y <= point.y <= self.max_point.y and
            self.min_point.z <= point.z <= self.max_point.z
        )

class BaseGeometry:
    """Base class for all geometric entities."""
    
    def __init__(self, dimensions: Dict[str, Any]) -> None:
        """Initialize geometry with dimensions."""
        self.dimensions = dimensions
        self.type = "base"
        self.features: List[Dict[str, Any]] = []
        self.transformations: List[Dict[str, Any]] = []
    
    def analyze_thickness(self) -> float:
        """Analyze minimum wall thickness."""
        # For basic shapes, use the smallest dimension
        dims = [
            self.dimensions.get("width", float('inf')),
            self.dimensions.get("height", float('inf')),
            self.dimensions.get("depth", float('inf')),
            self.dimensions.get("thickness", float('inf'))
        ]
        return min(d for d in dims if d > 0)
    
    def analyze_overhangs(self) -> float:
        """Analyze maximum overhang angle."""
        # Base implementation assumes no overhangs
        return 0.0
    
    def analyze_stress_points(self) -> List[Point]:
        """Analyze potential stress points."""
        # Base implementation assumes no stress points
        return []
    
    def thicken_walls(self, min_thickness: float = 0.1) -> 'BaseGeometry':
        """Thicken walls to meet minimum thickness requirement."""
        new_dims = self.dimensions.copy()
        for key in ["width", "height", "depth", "thickness"]:
            if key in new_dims and new_dims[key] < min_thickness:
                new_dims[key] = min_thickness
        return BaseGeometry(new_dims)
    
    def reduce_overhangs(self, max_angle: float = 45.0) -> 'BaseGeometry':
        """Reduce overhangs to be within maximum angle."""
        # Base implementation returns self as there are no overhangs
        return self
    
    def reinforce_weak_points(self) -> 'BaseGeometry':
        """Reinforce areas with potential structural weakness."""
        # Base implementation returns self as there are no weak points
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert geometry to dictionary representation."""
        return {
            "type": self.type,
            "dimensions": self.dimensions,
            "features": self.features,
            "transformations": self.transformations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseGeometry':
        """Create geometry from dictionary representation."""
        geometry = cls(data["dimensions"])
        geometry.type = data["type"]
        geometry.features = data["features"]
        geometry.transformations = data["transformations"]
        return geometry

class GeometricEntity(BaseGeometry):
    """Base class for geometric entities with common functionality."""
    
    def __init__(self):
        super().__init__({})
        self._bounding_box: Optional[BoundingBox] = None
        self._volume: Optional[float] = None
        self._surface_area: Optional[float] = None
    
    @property
    def bounding_box(self) -> BoundingBox:
        """Get entity's bounding box."""
        if self._bounding_box is None:
            self._compute_bounding_box()
        return cast(BoundingBox, self._bounding_box)
    
    @property
    def volume(self) -> float:
        """Get entity's volume."""
        if self._volume is None:
            self._compute_volume()
        return cast(float, self._volume)
    
    @property
    def surface_area(self) -> float:
        """Get entity's surface area."""
        if self._surface_area is None:
            self._compute_surface_area()
        return cast(float, self._surface_area)
    
    @abstractmethod
    def _compute_bounding_box(self) -> None:
        """Compute bounding box."""
        pass
    
    @abstractmethod
    def _compute_volume(self) -> None:
        """Compute volume."""
        pass
    
    @abstractmethod
    def _compute_surface_area(self) -> None:
        """Compute surface area."""
        pass

class NURBSEntity(GeometricEntity):
    """Base class for NURBS representations."""
    
    def __init__(self, 
                 control_points: List[Point],
                 weights: Optional[List[float]] = None,
                 knots: Optional[List[float]] = None,
                 degree: int = 3):
        super().__init__()
        self.control_points = control_points
        self.weights = weights if weights is not None else [1.0] * len(control_points)
        self.knots = knots
        self.degree = degree
        self._validate()
    
    def _validate(self):
        """Validate NURBS parameters."""
        if len(self.control_points) < self.degree + 1:
            raise ValueError("Not enough control points for specified degree")
        if len(self.weights) != len(self.control_points):
            raise ValueError("Number of weights must match number of control points")
        if self.knots is not None and len(self.knots) != len(self.control_points) + self.degree + 1:
            raise ValueError("Invalid number of knots")

    def evaluate(self, u: float) -> Point:
        """Evaluate NURBS at parameter u."""
        # Basic implementation - will be enhanced
        if not (0 <= u <= 1):
            raise ValueError("Parameter u must be in [0,1]")
        # TODO: Implement proper NURBS evaluation
        return self.control_points[0]  # Placeholder

    def _compute_bounding_box(self) -> None:
        """Compute bounding box."""
        # Default implementation using control points
        points = np.array([p.to_array() for p in self.control_points])
        min_point = Point(*np.min(points, axis=0))
        max_point = Point(*np.max(points, axis=0))
        self._bounding_box = BoundingBox(min_point, max_point)
    
    def _compute_volume(self) -> None:
        """Compute volume."""
        # Default implementation - zero volume for curves
        self._volume = 0.0
    
    def _compute_surface_area(self) -> None:
        """Compute surface area."""
        # Default implementation - zero area for curves
        self._surface_area = 0.0

class OrganicShape(GeometricEntity):
    """Base class for organic shapes."""
    
    def __init__(self, 
                 control_surfaces: List[NURBSEntity],
                 deformation_params: Optional[dict] = None):
        super().__init__()
        self.control_surfaces = control_surfaces
        self.deformation_params = deformation_params or {}
    
    def _compute_bounding_box(self) -> None:
        """Compute bounding box from control surfaces."""
        if not self.control_surfaces:
            raise ValueError("No control surfaces defined")
        
        # Get bounding boxes of all control surfaces
        boxes = [surface.bounding_box for surface in self.control_surfaces]
        
        # Find min and max points across all boxes
        min_x = min(box.min_point.x for box in boxes)
        min_y = min(box.min_point.y for box in boxes)
        min_z = min(box.min_point.z for box in boxes)
        max_x = max(box.max_point.x for box in boxes)
        max_y = max(box.max_point.y for box in boxes)
        max_z = max(box.max_point.z for box in boxes)
        
        self._bounding_box = BoundingBox(
            Point(min_x, min_y, min_z),
            Point(max_x, max_y, max_z)
        )
    
    def _compute_volume(self) -> None:
        """Compute approximate volume from control surfaces."""
        total_volume = 0.0
        for surface in self.control_surfaces:
            total_volume += surface.volume
        self._volume = total_volume
    
    def _compute_surface_area(self) -> None:
        """Compute approximate surface area from control surfaces."""
        total_area = 0.0
        for surface in self.control_surfaces:
            total_area += surface.surface_area
        self._surface_area = total_area

    def apply_deformation(self, deformation_type: str, params: dict) -> 'OrganicShape':
        """Apply deformation to shape."""
        # TODO: Implement deformation types (twist, bend, etc.)
        return self

    def to_nurbs(self) -> NURBSEntity:
        """Convert to NURBS representation."""
        # TODO: Implement conversion to NURBS
        return self.control_surfaces[0]

class ParametricEntity(GeometricEntity):
    """Base class for parametric entities."""
    
    def __init__(self, parameters: dict, constraints: Optional[dict] = None):
        self.parameters = parameters
        self.constraints = constraints or {}
        self._validate_constraints()
    
    def _validate_constraints(self):
        """Validate parameter constraints."""
        for param, value in self.parameters.items():
            if param in self.constraints:
                constraint = self.constraints[param]
                if 'min' in constraint and value < constraint['min']:
                    raise ValueError(f"Parameter {param} below minimum value")
                if 'max' in constraint and value > constraint['max']:
                    raise ValueError(f"Parameter {param} above maximum value")
    
    def update_parameter(self, param: str, value: float):
        """Update parameter value with validation."""
        self.parameters[param] = value
        self._validate_constraints()

class GeometryFactory:
    """Factory for creating geometric entities."""
    
    @staticmethod
    def create_nurbs_curve(control_points: List[Point], 
                          weights: Optional[List[float]] = None,
                          degree: int = 3) -> NURBSEntity:
        """Create a NURBS curve."""
        return NURBSEntity(control_points, weights, degree=degree)
    
    @staticmethod
    def create_organic_shape(control_points: List[List[Point]],
                           deformation_params: Optional[dict] = None) -> OrganicShape:
        """Create an organic shape."""
        control_surfaces = [
            NURBSEntity(points) for points in control_points
        ]
        return OrganicShape(control_surfaces, deformation_params) 