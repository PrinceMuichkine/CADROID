"""
NURBS (Non-Uniform Rational B-Spline) implementation for curve and surface manipulation.
"""

import numpy as np
from typing import List, Tuple, Optional
from .curve import Curve

def find_span(n: int, p: int, u: float, U: List[float]) -> int:
    """Find the knot span index."""
    if u >= U[n]:
        return n
    if u <= U[p]:
        return p
    
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
    """Compute the nonzero basis functions."""
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

class NURBSCurve(Curve):
    """NURBS curve implementation."""
    
    def __init__(self, 
                 control_points: List[Tuple[float, float, float]],
                 weights: Optional[List[float]] = None,
                 knots: Optional[List[float]] = None,
                 degree: int = 3):
        super().__init__()
        self.control_points = control_points
        self.weights = weights if weights is not None else [1.0] * len(control_points)
        self.degree = degree
        
        if knots is None:
            # Generate uniform knot vector
            n = len(control_points) - 1
            m = n + degree + 1
            self.knots = [0.0] * (degree + 1) + \
                        list(np.linspace(0, 1, m - 2*degree)) + \
                        [1.0] * (degree + 1)
        else:
            self.knots = knots
    
    def evaluate(self, u: float) -> Tuple[float, float, float]:
        """Evaluate the NURBS curve at parameter u."""
        n = len(self.control_points) - 1
        p = self.degree
        
        span = find_span(n, p, u, self.knots)
        N = basis_funs(span, u, p, self.knots)
        
        x = y = z = w = 0.0
        for i in range(p + 1):
            j = span - p + i
            weight = self.weights[j]
            point = self.control_points[j]
            
            factor = N[i] * weight
            x += factor * point[0]
            y += factor * point[1]
            z += factor * point[2]
            w += factor
        
        return (x/w, y/w, z/w)
    
    @classmethod
    def from_points(cls, points: List[Tuple[float, float, float]], degree: int = 3) -> 'NURBSCurve':
        """Create a NURBS curve interpolating the given points."""
        return cls(points, degree=degree)

class NURBSSurface:
    """NURBS surface implementation."""
    
    def __init__(self,
                 control_points: List[List[Tuple[float, float, float]]],
                 weights: Optional[List[List[float]]] = None,
                 u_knots: Optional[List[float]] = None,
                 v_knots: Optional[List[float]] = None,
                 degree_u: int = 3,
                 degree_v: int = 3):
        self.control_points = control_points
        self.degree_u = degree_u
        self.degree_v = degree_v
        
        nu = len(control_points) - 1
        nv = len(control_points[0]) - 1
        
        if weights is None:
            self.weights = [[1.0] * (nv + 1) for _ in range(nu + 1)]
        else:
            self.weights = weights
        
        if u_knots is None:
            mu = nu + degree_u + 1
            self.u_knots = [0.0] * (degree_u + 1) + \
                          list(np.linspace(0, 1, mu - 2*degree_u)) + \
                          [1.0] * (degree_u + 1)
        else:
            self.u_knots = u_knots
            
        if v_knots is None:
            mv = nv + degree_v + 1
            self.v_knots = [0.0] * (degree_v + 1) + \
                          list(np.linspace(0, 1, mv - 2*degree_v)) + \
                          [1.0] * (degree_v + 1)
        else:
            self.v_knots = v_knots
    
    def evaluate(self, u: float, v: float) -> Tuple[float, float, float]:
        """Evaluate the NURBS surface at parameters (u,v)."""
        nu = len(self.control_points) - 1
        nv = len(self.control_points[0]) - 1
        
        u_span = find_span(nu, self.degree_u, u, self.u_knots)
        v_span = find_span(nv, self.degree_v, v, self.v_knots)
        
        Nu = basis_funs(u_span, u, self.degree_u, self.u_knots)
        Nv = basis_funs(v_span, v, self.degree_v, self.v_knots)
        
        x = y = z = w = 0.0
        
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                ui = u_span - self.degree_u + i
                vj = v_span - self.degree_v + j
                
                weight = self.weights[ui][vj]
                point = self.control_points[ui][vj]
                
                factor = Nu[i] * Nv[j] * weight
                x += factor * point[0]
                y += factor * point[1]
                z += factor * point[2]
                w += factor
        
        return (x/w, y/w, z/w)
    
    @classmethod
    def from_points(cls, points: List[List[Tuple[float, float, float]]], 
                   degree_u: int = 3, degree_v: int = 3) -> 'NURBSSurface':
        """Create a NURBS surface interpolating the given points."""
        return cls(points, degree_u=degree_u, degree_v=degree_v) 