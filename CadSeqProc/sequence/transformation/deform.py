"""
Deformation operations for geometric transformations.
"""

import numpy as np
from typing import List, Tuple, Optional
from ...geometry.nurbs import NURBSCurve, NURBSSurface

class DeformationOp:
    """Base class for deformation operations."""
    
    def validate_params(self) -> bool:
        """Validate deformation parameters."""
        raise NotImplementedError
    
    def apply(self, points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Apply deformation to points."""
        raise NotImplementedError

class TwistDeformation(DeformationOp):
    """Twist points around an axis."""
    
    def __init__(self, 
                 angle: float,
                 axis: str = 'z',
                 center: Optional[Tuple[float, float, float]] = None):
        self.angle = angle
        self.axis = axis.lower()
        self.center = center or (0.0, 0.0, 0.0)
    
    def validate_params(self) -> bool:
        return self.axis in ['x', 'y', 'z']
    
    def apply(self, points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        if not self.validate_params():
            raise ValueError(f"Invalid axis: {self.axis}")
        
        result = []
        for point in points:
            # Convert to local coordinates
            local = np.array(point) - np.array(self.center)
            
            # Calculate twist angle based on position
            if self.axis == 'z':
                height = local[2]
                twist = self.angle * height
                c, s = np.cos(twist), np.sin(twist)
                x = local[0] * c - local[1] * s
                y = local[0] * s + local[1] * c
                z = local[2]
            elif self.axis == 'y':
                height = local[1]
                twist = self.angle * height
                c, s = np.cos(twist), np.sin(twist)
                x = local[0] * c - local[2] * s
                y = local[1]
                z = local[0] * s + local[2] * c
            else:  # x axis
                height = local[0]
                twist = self.angle * height
                c, s = np.cos(twist), np.sin(twist)
                x = local[0]
                y = local[1] * c - local[2] * s
                z = local[1] * s + local[2] * c
            
            # Convert back to global coordinates
            result.append(tuple(np.array([x, y, z]) + np.array(self.center)))
        
        return result

class BendDeformation(DeformationOp):
    """Bend points along a curve."""
    
    def __init__(self,
                 curve: NURBSCurve,
                 bend_factor: float = 1.0,
                 up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)):
        self.curve = curve
        self.bend_factor = bend_factor
        self.up_vector = np.array(up_vector)
        self.up_vector /= np.linalg.norm(self.up_vector)
    
    def validate_params(self) -> bool:
        return self.bend_factor > 0
    
    def apply(self, points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        if not self.validate_params():
            raise ValueError("Invalid bend factor")
        
        result = []
        for point in points:
            # Project point onto curve
            min_dist = float('inf')
            min_param = 0.0
            for t in np.linspace(0, 1, 100):
                curve_point = self.curve.evaluate(t)
                dist = np.linalg.norm(np.array(point) - np.array(curve_point))
                if dist < min_dist:
                    min_dist = dist
                    min_param = t
            
            # Calculate bend
            curve_point = np.array(self.curve.evaluate(min_param))
            bend_amount = min_dist * self.bend_factor
            
            # Apply bend
            bent_point = curve_point + self.up_vector * bend_amount
            result.append(tuple(bent_point))
        
        return result

class TaperDeformation(DeformationOp):
    """Taper points along an axis."""
    
    def __init__(self,
                 start_scale: float = 1.0,
                 end_scale: float = 0.5,
                 axis: str = 'z',
                 center: Optional[Tuple[float, float, float]] = None):
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.axis = axis.lower()
        self.center = center or (0.0, 0.0, 0.0)
    
    def validate_params(self) -> bool:
        return (self.axis in ['x', 'y', 'z'] and 
                self.start_scale > 0 and 
                self.end_scale > 0)
    
    def apply(self, points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        if not self.validate_params():
            raise ValueError("Invalid parameters")
        
        result = []
        for point in points:
            # Convert to local coordinates
            local = np.array(point) - np.array(self.center)
            
            # Calculate scale factor based on position
            if self.axis == 'z':
                t = (local[2] - self.center[2]) / (max(p[2] for p in points) - self.center[2])
            elif self.axis == 'y':
                t = (local[1] - self.center[1]) / (max(p[1] for p in points) - self.center[1])
            else:  # x axis
                t = (local[0] - self.center[0]) / (max(p[0] for p in points) - self.center[0])
            
            scale = self.start_scale + t * (self.end_scale - self.start_scale)
            
            # Apply scale
            if self.axis == 'z':
                x = local[0] * scale
                y = local[1] * scale
                z = local[2]
            elif self.axis == 'y':
                x = local[0] * scale
                y = local[1]
                z = local[2] * scale
            else:  # x axis
                x = local[0]
                y = local[1] * scale
                z = local[2] * scale
            
            # Convert back to global coordinates
            result.append(tuple(np.array([x, y, z]) + np.array(self.center)))
        
        return result 