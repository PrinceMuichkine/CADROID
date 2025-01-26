"""
NURBS (Non-Uniform Rational B-Spline) implementation.
Provides advanced curve and surface manipulation capabilities.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, cast
from .base import Point, NURBSEntity, GeometricEntity, BaseGeometry, BoundingBox

def find_span(n: int, p: int, u: float, U: List[float]) -> int:
    """
    Find the knot span index.
    
    Args:
        n: Number of control points - 1
        p: Degree of curve
        u: Parameter value
        U: Knot vector
    
    Returns:
        Knot span index
    """
    if u == U[n + 1]:
        return n
    
    low = p
    high = n + 1
    mid = (low + high) // 2
    
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    
    return mid

def basis_funs(i: int, u: float, p: int, U: List[float]) -> List[float]:
    """
    Compute the nonzero basis functions.
    
    Args:
        i: Knot span index
        u: Parameter value
        p: Degree of curve
        U: Knot vector
    
    Returns:
        List of basis function values
    """
    N = [0.0] * (p + 1)
    left = [0.0] * (p + 1)
    right = [0.0] * (p + 1)
    
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        N[j] = saved
    
    return N

class NURBSCurve(BaseGeometry):
    """NURBS curve implementation."""
    
    def __init__(self, control_points: List[List[float]], degree: int = 3,
                 weights: Optional[List[float]] = None):
        """Initialize NURBS curve."""
        super().__init__()
        self.control_points = np.array(control_points)
        self.degree = degree
        self.weights = np.ones(len(control_points)) if weights is None else np.array(weights)
        self._knot_vector = self._generate_knot_vector()
    
    @classmethod
    def from_points(cls, points: List[List[float]], degree: int = 3) -> 'NURBSCurve':
        """Create curve from points."""
        return cls(points, degree)
    
    def point_at(self, t: float) -> List[float]:
        """Evaluate curve at parameter t."""
        # Implement De Boor's algorithm
        n = len(self.control_points) - 1
        p = self.degree
        
        # Find knot span
        span = self._find_span(t)
        
        # Calculate basis functions
        N = self._basis_functions(span, t)
        
        # Calculate point
        point = np.zeros(3)
        w_sum = 0.0
        
        for i in range(p + 1):
            weight = self.weights[span - p + i]
            point += N[i] * weight * self.control_points[span - p + i]
            w_sum += N[i] * weight
        
        return (point / w_sum).tolist()
    
    def sweep(self, width: float) -> 'NURBSSurface':
        """Create surface by sweeping curve."""
        # Create surface control points by sweeping curve
        surface_points = []
        for t in np.linspace(0, 1, 10):
            curve_point = self.point_at(t)
            row = []
            for s in np.linspace(-width/2, width/2, 5):
                # Create points perpendicular to curve
                point = [
                    curve_point[0],
                    curve_point[1] + s,
                    curve_point[2]
                ]
                row.append(point)
            surface_points.append(row)
        
        return NURBSSurface(surface_points)
    
    def _generate_knot_vector(self) -> np.ndarray:
        """Generate uniform knot vector."""
        n = len(self.control_points) - 1
        p = self.degree
        m = n + p + 1
        
        knots = np.zeros(m + 1)
        for i in range(m + 1):
            if i <= p:
                knots[i] = 0
            elif i >= m - p:
                knots[i] = 1
            else:
                knots[i] = (i - p) / (m - 2 * p)
        
        return knots
    
    def _find_span(self, t: float) -> int:
        """Find knot span index."""
        n = len(self.control_points) - 1
        p = self.degree
        
        if t >= 1.0:
            return n
        
        low = p
        high = n + 1
        mid = (low + high) // 2
        
        while t < self._knot_vector[mid] or t >= self._knot_vector[mid + 1]:
            if t < self._knot_vector[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
        
        return mid
    
    def _basis_functions(self, span: int, t: float) -> np.ndarray:
        """Calculate basis functions."""
        p = self.degree
        N = np.zeros(p + 1)
        left = np.zeros(p + 1)
        right = np.zeros(p + 1)
        
        N[0] = 1.0
        for j in range(1, p + 1):
            left[j] = t - self._knot_vector[span + 1 - j]
            right[j] = self._knot_vector[span + j] - t
            saved = 0.0
            
            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            
            N[j] = saved
        
        return N
    
    def analyze_thickness(self) -> float:
        """Analyze curve thickness (always 0 for curves)."""
        return 0.0
    
    def analyze_overhangs(self) -> float:
        """Analyze curve overhangs (not applicable for curves)."""
        return 0.0
    
    def analyze_stress_points(self) -> List[Tuple[float, float, float]]:
        """Analyze curve stress points (not applicable for curves)."""
        return []
    
    def thicken_walls(self, min_thickness: float) -> 'BaseGeometry':
        """Thicken curve (not applicable)."""
        return self
    
    def reduce_overhangs(self, max_angle: float) -> 'BaseGeometry':
        """Reduce curve overhangs (not applicable)."""
        return self
    
    def reinforce_weak_points(self) -> 'BaseGeometry':
        """Reinforce curve weak points (not applicable)."""
        return self

class NURBSSurface(BaseGeometry):
    """NURBS surface implementation."""
    
    def __init__(self, control_points: List[List[List[float]]], 
                 degree_u: int = 3, degree_v: int = 3,
                 weights: Optional[List[List[float]]] = None):
        """Initialize NURBS surface."""
        super().__init__()
        self.control_points = np.array(control_points)
        self.degree_u = degree_u
        self.degree_v = degree_v
        
        if weights is None:
            self.weights = np.ones((
                self.control_points.shape[0],
                self.control_points.shape[1]
            ))
        else:
            self.weights = np.array(weights)
        
        self._knot_vector_u = self._generate_knot_vector(
            self.control_points.shape[0] - 1,
            degree_u
        )
        self._knot_vector_v = self._generate_knot_vector(
            self.control_points.shape[1] - 1,
            degree_v
        )
    
    @classmethod
    def from_points(cls, points: List[List[List[float]]],
                   degree_u: int = 3, degree_v: int = 3) -> 'NURBSSurface':
        """Create surface from points."""
        return cls(points, degree_u, degree_v)
    
    def point_at(self, u: float, v: float) -> List[float]:
        """Evaluate surface at parameters (u,v)."""
        # Find spans
        span_u = self._find_span(u, self._knot_vector_u, self.control_points.shape[0] - 1, self.degree_u)
        span_v = self._find_span(v, self._knot_vector_v, self.control_points.shape[1] - 1, self.degree_v)
        
        # Calculate basis functions
        Nu = self._basis_functions(span_u, u, self._knot_vector_u, self.degree_u)
        Nv = self._basis_functions(span_v, v, self._knot_vector_v, self.degree_v)
        
        # Calculate point
        point = np.zeros(3)
        w_sum = 0.0
        
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                weight = self.weights[span_u - self.degree_u + i, span_v - self.degree_v + j]
                point += (Nu[i] * Nv[j] * weight * 
                         self.control_points[span_u - self.degree_u + i,
                                          span_v - self.degree_v + j])
                w_sum += Nu[i] * Nv[j] * weight
        
        return (point / w_sum).tolist()
    
    def _generate_knot_vector(self, n: int, p: int) -> np.ndarray:
        """Generate uniform knot vector."""
        m = n + p + 1
        knots = np.zeros(m + 1)
        
        for i in range(m + 1):
            if i <= p:
                knots[i] = 0
            elif i >= m - p:
                knots[i] = 1
            else:
                knots[i] = (i - p) / (m - 2 * p)
        
        return knots
    
    def _find_span(self, t: float, knot_vector: np.ndarray,
                  n: int, p: int) -> int:
        """Find knot span index."""
        if t >= 1.0:
            return n
        
        low = p
        high = n + 1
        mid = (low + high) // 2
        
        while t < knot_vector[mid] or t >= knot_vector[mid + 1]:
            if t < knot_vector[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
        
        return mid
    
    def _basis_functions(self, span: int, t: float,
                        knot_vector: np.ndarray, p: int) -> np.ndarray:
        """Calculate basis functions."""
        N = np.zeros(p + 1)
        left = np.zeros(p + 1)
        right = np.zeros(p + 1)
        
        N[0] = 1.0
        for j in range(1, p + 1):
            left[j] = t - knot_vector[span + 1 - j]
            right[j] = knot_vector[span + j] - t
            saved = 0.0
            
            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            
            N[j] = saved
        
        return N
    
    def analyze_thickness(self) -> float:
        """Analyze surface thickness."""
        # Simple thickness analysis based on bounding box
        bbox = self.bounding_box
        return min(bbox.dimensions)
    
    def analyze_overhangs(self) -> float:
        """Analyze surface overhangs."""
        # Simple overhang analysis
        max_angle = 0.0
        
        # Sample surface normals
        for u in np.linspace(0, 1, 10):
            for v in np.linspace(0, 1, 10):
                normal = self._compute_normal(u, v)
                angle = np.arccos(np.dot(normal, [0, 0, 1])) * 180 / np.pi
                max_angle = max(max_angle, angle)
        
        return max_angle
    
    def analyze_stress_points(self) -> List[Tuple[float, float, float]]:
        """Analyze surface stress points."""
        stress_points = []
        
        # Simple stress analysis based on curvature
        for u in np.linspace(0, 1, 10):
            for v in np.linspace(0, 1, 10):
                curvature = self._compute_curvature(u, v)
                if curvature > 1.0:  # Threshold for high curvature
                    point = self.point_at(u, v)
                    stress_points.append(tuple(point))  # type: ignore
        
        return stress_points
    
    def thicken_walls(self, min_thickness: float) -> 'BaseGeometry':
        """Thicken surface walls."""
        # Simple uniform thickening
        thickened_points = []
        
        for i in range(self.control_points.shape[0]):
            row = []
            for j in range(self.control_points.shape[1]):
                point = self.control_points[i, j]
                normal = self._compute_normal(i/(self.control_points.shape[0]-1),
                                           j/(self.control_points.shape[1]-1))
                offset = normal * min_thickness
                row.append((point + offset).tolist())
            thickened_points.append(row)
        
        return NURBSSurface(thickened_points, self.degree_u, self.degree_v)
    
    def reduce_overhangs(self, max_angle: float) -> 'BaseGeometry':
        """Reduce surface overhangs."""
        # Simple overhang reduction by rotating surface
        if self.analyze_overhangs() > max_angle:
            # Rotate surface to reduce overhangs
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(np.pi/6), -np.sin(np.pi/6)],
                [0, np.sin(np.pi/6), np.cos(np.pi/6)]
            ])
            
            rotated_points = []
            for i in range(self.control_points.shape[0]):
                row = []
                for j in range(self.control_points.shape[1]):
                    point = np.dot(rotation_matrix, self.control_points[i, j])
                    row.append(point.tolist())
                rotated_points.append(row)
            
            return NURBSSurface(rotated_points, self.degree_u, self.degree_v)
        
        return self
    
    def reinforce_weak_points(self) -> 'BaseGeometry':
        """Reinforce surface weak points."""
        # Simple reinforcement by thickening high-stress regions
        stress_points = self.analyze_stress_points()
        if not stress_points:
            return self
        
        # Add thickness near stress points
        reinforced_points = []
        for i in range(self.control_points.shape[0]):
            row = []
            for j in range(self.control_points.shape[1]):
                point = self.control_points[i, j]
                
                # Check if near stress point
                near_stress = False
                for stress_point in stress_points:
                    if np.linalg.norm(point - np.array(stress_point)) < 0.5:
                        near_stress = True
                        break
                
                if near_stress:
                    normal = self._compute_normal(i/(self.control_points.shape[0]-1),
                                               j/(self.control_points.shape[1]-1))
                    point = point + normal * 0.1  # Add thickness
                
                row.append(point.tolist())
            reinforced_points.append(row)
        
        return NURBSSurface(reinforced_points, self.degree_u, self.degree_v)
    
    def _compute_normal(self, u: float, v: float) -> np.ndarray:
        """Compute surface normal at point."""
        # Approximate normal using central differences
        delta = 0.01
        
        p1 = np.array(self.point_at(u + delta, v))
        p2 = np.array(self.point_at(u - delta, v))
        p3 = np.array(self.point_at(u, v + delta))
        p4 = np.array(self.point_at(u, v - delta))
        
        du = (p1 - p2) / (2 * delta)
        dv = (p3 - p4) / (2 * delta)
        
        normal = np.cross(du, dv)
        return normal / np.linalg.norm(normal)
    
    def _compute_curvature(self, u: float, v: float) -> float:
        """Compute surface curvature at point."""
        # Approximate curvature using second derivatives
        delta = 0.01
        
        p0 = np.array(self.point_at(u, v))
        p1 = np.array(self.point_at(u + delta, v))
        p2 = np.array(self.point_at(u - delta, v))
        p3 = np.array(self.point_at(u, v + delta))
        p4 = np.array(self.point_at(u, v - delta))
        
        # Second derivatives
        duu = (p1 - 2*p0 + p2) / (delta * delta)
        dvv = (p3 - 2*p0 + p4) / (delta * delta)
        
        # Mean curvature (simplified)
        return float(np.linalg.norm(duu + dvv) / 2) 