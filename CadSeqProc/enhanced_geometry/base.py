"""
Enhanced geometry system for CAD generation.
Supports both geometric primitives and organic shapes.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Point':
        return cls(float(arr[0]), float(arr[1]), float(arr[2]) if len(arr) > 2 else 0.0)

class GeometricEntity(ABC):
    """Base class for all geometric entities."""
    
    @abstractmethod
    def to_nurbs(self) -> 'NURBSEntity':
        """Convert entity to NURBS representation."""
        pass
    
    @abstractmethod
    def sample_points(self, n_points: int) -> List[Point]:
        """Sample points along the entity."""
        pass
    
    @abstractmethod
    def get_bbox(self) -> Tuple[Point, Point]:
        """Get bounding box (min_point, max_point)."""
        pass
    
    @abstractmethod
    def transform(self, matrix: np.ndarray) -> 'GeometricEntity':
        """Apply transformation matrix."""
        pass

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