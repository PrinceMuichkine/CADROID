"""
Base classes and interfaces for enhanced geometry system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, cast
import numpy as np

class Point:
    """3D point representation."""
    
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Point':
        """Create point from numpy array."""
        return cls(arr[0], arr[1], arr[2])
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate distance to another point."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

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

class BaseGeometry(ABC):
    """Abstract base class for all geometric entities."""
    
    @abstractmethod
    def analyze_thickness(self) -> float:
        """Analyze minimum wall thickness."""
        pass
    
    @abstractmethod
    def analyze_overhangs(self) -> float:
        """Analyze maximum overhang angle."""
        pass
    
    @abstractmethod
    def analyze_stress_points(self) -> List[Tuple[float, float, float]]:
        """Analyze potential stress points."""
        pass
    
    @abstractmethod
    def thicken_walls(self, min_thickness: float) -> 'BaseGeometry':
        """Thicken walls to meet minimum thickness."""
        pass
    
    @abstractmethod
    def reduce_overhangs(self, max_angle: float) -> 'BaseGeometry':
        """Reduce overhangs to meet maximum angle."""
        pass
    
    @abstractmethod
    def reinforce_weak_points(self) -> 'BaseGeometry':
        """Reinforce identified weak points."""
        pass

class GeometricEntity(BaseGeometry):
    """Base class for geometric entities with common functionality."""
    
    def __init__(self):
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
    
    def analyze_thickness(self) -> float:
        """Default thickness analysis."""
        # Implement basic thickness analysis
        return float('inf')
    
    def analyze_overhangs(self) -> float:
        """Default overhang analysis."""
        # Implement basic overhang analysis
        return 0.0
    
    def analyze_stress_points(self) -> List[Tuple[float, float, float]]:
        """Default stress point analysis."""
        # Implement basic stress analysis
        return []
    
    def thicken_walls(self, min_thickness: float) -> 'GeometricEntity':
        """Default wall thickening."""
        # Implement basic wall thickening
        return self
    
    def reduce_overhangs(self, max_angle: float) -> 'GeometricEntity':
        """Default overhang reduction."""
        # Implement basic overhang reduction
        return self
    
    def reinforce_weak_points(self) -> 'GeometricEntity':
        """Default weak point reinforcement."""
        # Implement basic reinforcement
        return self

class NURBSEntity(GeometricEntity):
    """Base class for NURBS representations."""
    
    def __init__(self, 
                 control_points: List[Point],
                 weights: Optional[List[float]] = None,
                 knots: Optional[List[float]] = None,
                 degree: int = 3):
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

class OrganicShape(GeometricEntity):
    """Base class for organic shapes."""
    
    def __init__(self, 
                 control_surfaces: List[NURBSEntity],
                 deformation_params: Optional[dict] = None):
        self.control_surfaces = control_surfaces
        self.deformation_params = deformation_params or {}
    
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