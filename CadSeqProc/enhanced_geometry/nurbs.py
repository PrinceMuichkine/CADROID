"""
NURBS (Non-Uniform Rational B-Spline) implementation.
Provides advanced curve and surface manipulation capabilities.
"""

import numpy as np
from typing import List, Tuple, Optional
from .base import Point, NURBSEntity, GeometricEntity

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

class NURBSCurve(NURBSEntity):
    """NURBS curve implementation."""
    
    def __init__(self,
                 control_points: List[Point],
                 weights: Optional[List[float]] = None,
                 knots: Optional[List[float]] = None,
                 degree: int = 3):
        super().__init__(control_points, weights, knots, degree)
        
        if knots is None:
            # Generate uniform knot vector
            n = len(control_points) - 1
            self.knots = [0.0] * (degree + 1) + \
                        [i / (n - degree + 1) for i in range(1, n - degree + 1)] + \
                        [1.0] * (degree + 1)
    
    def evaluate(self, u: float) -> Point:
        """
        Evaluate the NURBS curve at parameter u.
        
        Args:
            u: Parameter value in [0,1]
            
        Returns:
            Point on curve
        """
        if not (0 <= u <= 1):
            raise ValueError("Parameter u must be in [0,1]")
        
        n = len(self.control_points) - 1
        p = self.degree
        
        span = find_span(n, p, u, self.knots)
        basis = basis_funs(span, u, p, self.knots)
        
        Cw = np.zeros(3)
        w = 0.0
        
        for i in range(p + 1):
            Cw += self.control_points[span - p + i].to_array() * \
                  self.weights[span - p + i] * basis[i]
            w += self.weights[span - p + i] * basis[i]
        
        return Point.from_array(Cw / w)
    
    def sample_points(self, n_points: int) -> List[Point]:
        """Sample points along the curve."""
        if n_points < 2:
            raise ValueError("n_points must be at least 2")
        
        points = []
        for i in range(n_points):
            u = i / (n_points - 1)
            points.append(self.evaluate(u))
        
        return points
    
    def get_bbox(self) -> Tuple[Point, Point]:
        """Get bounding box of curve."""
        points = self.sample_points(100)  # Sample enough points for good approximation
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        zs = [p.z for p in points]
        
        return (
            Point(min(xs), min(ys), min(zs)),
            Point(max(xs), max(ys), max(zs))
        )
    
    def transform(self, matrix: np.ndarray) -> 'NURBSCurve':
        """Apply transformation matrix to curve."""
        if matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")
        
        new_points = []
        for p in self.control_points:
            p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
            transformed = matrix @ p_homogeneous
            new_points.append(Point(
                transformed[0] / transformed[3],
                transformed[1] / transformed[3],
                transformed[2] / transformed[3]
            ))
        
        return NURBSCurve(
            new_points,
            weights=self.weights.copy(),
            knots=self.knots.copy(),
            degree=self.degree
        )
    
    def to_nurbs(self) -> 'NURBSEntity':
        """Already a NURBS entity."""
        return self

class NURBSSurface(NURBSEntity):
    """NURBS surface implementation."""
    
    def __init__(self,
                 control_points: List[List[Point]],
                 weights: Optional[List[List[float]]] = None,
                 u_knots: Optional[List[float]] = None,
                 v_knots: Optional[List[float]] = None,
                 u_degree: int = 3,
                 v_degree: int = 3):
        self.u_degree = u_degree
        self.v_degree = v_degree
        self.control_points_2d = control_points
        self.control_points = [p for row in control_points for p in row]
        
        if weights is None:
            weights = [[1.0] * len(row) for row in control_points]
        self.weights_2d = weights
        self.weights = [w for row in weights for w in row]
        
        self.u_knots = u_knots
        self.v_knots = v_knots
        self._validate()
        
        if u_knots is None or v_knots is None:
            self._generate_knot_vectors()
    
    def _generate_knot_vectors(self):
        """Generate uniform knot vectors if not provided."""
        if self.u_knots is None:
            n = len(self.control_points_2d) - 1
            self.u_knots = [0.0] * (self.u_degree + 1) + \
                          [i / (n - self.u_degree + 1) for i in range(1, n - self.u_degree + 1)] + \
                          [1.0] * (self.u_degree + 1)
        
        if self.v_knots is None:
            m = len(self.control_points_2d[0]) - 1
            self.v_knots = [0.0] * (self.v_degree + 1) + \
                          [i / (m - self.v_degree + 1) for i in range(1, m - self.v_degree + 1)] + \
                          [1.0] * (self.v_degree + 1)
    
    def evaluate(self, u: float, v: float) -> Point:
        """
        Evaluate the NURBS surface at parameters (u,v).
        
        Args:
            u: Parameter in u direction [0,1]
            v: Parameter in v direction [0,1]
            
        Returns:
            Point on surface
        """
        if not (0 <= u <= 1 and 0 <= v <= 1):
            raise ValueError("Parameters must be in [0,1]")
        
        n = len(self.control_points_2d) - 1
        m = len(self.control_points_2d[0]) - 1
        
        u_span = find_span(n, self.u_degree, u, self.u_knots)
        v_span = find_span(m, self.v_degree, v, self.v_knots)
        
        u_basis = basis_funs(u_span, u, self.u_degree, self.u_knots)
        v_basis = basis_funs(v_span, v, self.v_degree, self.v_knots)
        
        Sw = np.zeros(3)
        w = 0.0
        
        for i in range(self.u_degree + 1):
            for j in range(self.v_degree + 1):
                temp = u_basis[i] * v_basis[j] * \
                       self.weights_2d[u_span - self.u_degree + i][v_span - self.v_degree + j]
                Sw += self.control_points_2d[u_span - self.u_degree + i][v_span - self.v_degree + j].to_array() * temp
                w += temp
        
        return Point.from_array(Sw / w)
    
    def sample_points(self, n_points: int) -> List[Point]:
        """Sample points on surface grid."""
        if n_points < 2:
            raise ValueError("n_points must be at least 2")
        
        points = []
        for i in range(n_points):
            for j in range(n_points):
                u = i / (n_points - 1)
                v = j / (n_points - 1)
                points.append(self.evaluate(u, v))
        
        return points 