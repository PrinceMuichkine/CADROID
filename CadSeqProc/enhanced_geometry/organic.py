"""
Organic shape implementation with advanced deformation capabilities.
Supports natural forms and complex surface manipulations.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .base import Point, OrganicShape, GeometricEntity
from .nurbs import NURBSSurface, NURBSCurve

class DeformationOp:
    """Base class for deformation operations."""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self._validate_params()
    
    def _validate_params(self):
        """Validate deformation parameters."""
        pass
    
    def apply(self, points: List[Point]) -> List[Point]:
        """Apply deformation to points."""
        raise NotImplementedError

class TwistDeformation(DeformationOp):
    """Twist deformation around an axis."""
    
    def _validate_params(self):
        required = {'axis', 'angle', 'center'}
        if not all(k in self.params for k in required):
            raise ValueError(f"Missing required parameters: {required - set(self.params.keys())}")
    
    def apply(self, points: List[Point]) -> List[Point]:
        axis = np.array(self.params['axis'])
        angle = self.params['angle']
        center = np.array(self.params['center'])
        
        # Normalize axis
        axis = axis / np.linalg.norm(axis)
        
        new_points = []
        for p in points:
            # Convert to numpy array
            p_array = np.array([p.x, p.y, p.z])
            
            # Calculate distance from axis
            v = p_array - center
            dist = np.linalg.norm(np.cross(v, axis))
            
            # Calculate twist angle based on distance
            theta = angle * dist
            
            # Create rotation matrix
            c = np.cos(theta)
            s = np.sin(theta)
            t = 1 - c
            x, y, z = axis
            
            R = np.array([
                [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
                [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
                [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
            ])
            
            # Apply rotation
            new_p = center + R @ (p_array - center)
            new_points.append(Point(new_p[0], new_p[1], new_p[2]))
        
        return new_points

class BendDeformation(DeformationOp):
    """Bend deformation along a curve."""
    
    def _validate_params(self):
        required = {'curve', 'angle', 'axis'}
        if not all(k in self.params for k in required):
            raise ValueError(f"Missing required parameters: {required - set(self.params.keys())}")
    
    def apply(self, points: List[Point]) -> List[Point]:
        curve = self.params['curve']
        angle = self.params['angle']
        axis = np.array(self.params['axis'])
        
        # Normalize axis
        axis = axis / np.linalg.norm(axis)
        
        new_points = []
        for p in points:
            # Project point onto bending plane
            p_array = np.array([p.x, p.y, p.z])
            proj = p_array - np.dot(p_array, axis) * axis
            
            # Calculate distance along curve
            t = curve.project_point(proj)
            
            # Calculate bend angle at this point
            theta = angle * t
            
            # Create rotation matrix
            c = np.cos(theta)
            s = np.sin(theta)
            t = 1 - c
            x, y, z = axis
            
            R = np.array([
                [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
                [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
                [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
            ])
            
            # Apply rotation
            new_p = R @ p_array
            new_points.append(Point(new_p[0], new_p[1], new_p[2]))
        
        return new_points

class TaperDeformation(DeformationOp):
    """Taper deformation along an axis."""
    
    def _validate_params(self):
        required = {'axis', 'start_scale', 'end_scale'}
        if not all(k in self.params for k in required):
            raise ValueError(f"Missing required parameters: {required - set(self.params.keys())}")
    
    def apply(self, points: List[Point]) -> List[Point]:
        axis = np.array(self.params['axis'])
        start_scale = self.params['start_scale']
        end_scale = self.params['end_scale']
        
        # Normalize axis
        axis = axis / np.linalg.norm(axis)
        
        new_points = []
        for p in points:
            p_array = np.array([p.x, p.y, p.z])
            
            # Calculate distance along axis
            t = np.dot(p_array, axis)
            
            # Calculate scale factor
            scale = start_scale + (end_scale - start_scale) * t
            
            # Apply scaling perpendicular to axis
            proj = np.dot(p_array, axis) * axis
            perp = p_array - proj
            new_p = proj + scale * perp
            
            new_points.append(Point(new_p[0], new_p[1], new_p[2]))
        
        return new_points

class OrganicSurface(OrganicShape):
    """Implementation of organic surface with deformation support."""
    
    def __init__(self,
                 control_surfaces: List[NURBSSurface],
                 deformation_params: Optional[Dict[str, Any]] = None):
        super().__init__(control_surfaces, deformation_params)
        self.deformations: List[DeformationOp] = []
        
        if deformation_params:
            self._setup_deformations()
    
    def _setup_deformations(self):
        """Set up deformation operations from parameters."""
        deformation_types = {
            'twist': TwistDeformation,
            'bend': BendDeformation,
            'taper': TaperDeformation
        }
        
        for def_type, params in self.deformation_params.items():
            if def_type in deformation_types:
                self.deformations.append(deformation_types[def_type](params))
    
    def apply_deformation(self, deformation_type: str, params: dict) -> 'OrganicSurface':
        """Apply a new deformation to the surface."""
        deformation_types = {
            'twist': TwistDeformation,
            'bend': BendDeformation,
            'taper': TaperDeformation
        }
        
        if deformation_type not in deformation_types:
            raise ValueError(f"Unknown deformation type: {deformation_type}")
        
        deformation = deformation_types[deformation_type](params)
        self.deformations.append(deformation)
        
        return self
    
    def sample_points(self, n_points: int) -> List[Point]:
        """Sample points on the organic surface."""
        # First sample points from all control surfaces
        points = []
        for surface in self.control_surfaces:
            points.extend(surface.sample_points(n_points))
        
        # Apply all deformations in sequence
        for deformation in self.deformations:
            points = deformation.apply(points)
        
        return points
    
    def get_bbox(self) -> Tuple[Point, Point]:
        """Get bounding box after all deformations."""
        points = self.sample_points(20)  # Sample enough points for good approximation
        
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        zs = [p.z for p in points]
        
        return (
            Point(min(xs), min(ys), min(zs)),
            Point(max(xs), max(ys), max(zs))
        )
    
    def transform(self, matrix: np.ndarray) -> 'OrganicSurface':
        """Apply transformation matrix to surface."""
        if matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")
        
        new_surfaces = []
        for surface in self.control_surfaces:
            # Transform control points of each surface
            new_points = []
            for row in surface.control_points_2d:
                new_row = []
                for p in row:
                    p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                    transformed = matrix @ p_homogeneous
                    new_row.append(Point(
                        transformed[0] / transformed[3],
                        transformed[1] / transformed[3],
                        transformed[2] / transformed[3]
                    ))
                new_points.append(new_row)
            
            new_surfaces.append(NURBSSurface(
                new_points,
                weights=surface.weights_2d,
                u_knots=surface.u_knots,
                v_knots=surface.v_knots,
                u_degree=surface.u_degree,
                v_degree=surface.v_degree
            ))
        
        return OrganicSurface(
            new_surfaces,
            deformation_params=self.deformation_params
        )
    
    def to_nurbs(self) -> NURBSSurface:
        """Convert to NURBS representation."""
        # For now, return the first surface after applying deformations
        points = self.sample_points(20)
        
        # Create a grid of points
        grid_size = int(np.sqrt(len(points)))
        points_grid = [points[i:i + grid_size] for i in range(0, len(points), grid_size)]
        
        return NURBSSurface(points_grid) 