"""
Organic surface implementation for complex shape generation.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .nurbs import NURBSCurve, NURBSSurface
from ..sequence.transformation.deform import DeformationOp

class OrganicSurface:
    """Surface with organic deformation capabilities."""
    
    def __init__(self,
                 control_surfaces: List[NURBSSurface],
                 deformations: Optional[List[DeformationOp]] = None):
        self.control_surfaces = control_surfaces
        self.deformations = deformations or []
    
    def add_deformation(self, deform: DeformationOp):
        """Add a deformation operation."""
        self.deformations.append(deform)
    
    def clear_deformations(self):
        """Remove all deformations."""
        self.deformations = []
    
    def sample_points(self, 
                     num_u: int = 20,
                     num_v: int = 20) -> List[List[Tuple[float, float, float]]]:
        """Sample points from the surface with deformations applied."""
        points = []
        
        # Sample points from each control surface
        for surface in self.control_surfaces:
            surface_points = []
            for u in np.linspace(0, 1, num_u):
                row = []
                for v in np.linspace(0, 1, num_v):
                    point = surface.evaluate(u, v)
                    
                    # Apply deformations in sequence
                    current_point = point
                    for deform in self.deformations:
                        current_point = deform.apply([current_point])[0]
                    
                    row.append(current_point)
                surface_points.append(row)
            points.append(surface_points)
        
        return points
    
    def get_bounding_box(self) -> Tuple[Tuple[float, float, float], 
                                      Tuple[float, float, float]]:
        """Get the bounding box of the surface."""
        points = self.sample_points()
        all_points = [p for surface in points for row in surface for p in row]
        
        min_x = min(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        min_z = min(p[2] for p in all_points)
        max_x = max(p[0] for p in all_points)
        max_y = max(p[1] for p in all_points)
        max_z = max(p[2] for p in all_points)
        
        return ((min_x, min_y, min_z), (max_x, max_y, max_z))
    
    def transform(self, 
                 matrix: np.ndarray) -> 'OrganicSurface':
        """Apply transformation matrix to the surface."""
        transformed_surfaces = []
        for surface in self.control_surfaces:
            # Transform control points
            new_points = []
            for row in surface.control_points:
                new_row = []
                for point in row:
                    p_homogeneous = np.array([*point, 1.0])
                    transformed = matrix @ p_homogeneous
                    new_row.append(tuple(transformed[:3] / transformed[3]))
                new_points.append(new_row)
            
            # Create new surface with transformed points
            transformed_surface = NURBSSurface(
                new_points,
                weights=surface.weights,
                u_knots=surface.u_knots,
                v_knots=surface.v_knots,
                degree_u=surface.degree_u,
                degree_v=surface.degree_v
            )
            transformed_surfaces.append(transformed_surface)
        
        return OrganicSurface(transformed_surfaces, self.deformations.copy())
    
    def to_nurbs(self) -> List[NURBSSurface]:
        """Convert to NURBS representation."""
        return self.control_surfaces 