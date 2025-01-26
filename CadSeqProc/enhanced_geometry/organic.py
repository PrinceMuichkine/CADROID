"""
Organic shape implementation with natural deformations.
"""

from typing import List, Dict, Any, Tuple, Optional, Union, cast
import numpy as np
from .base import BaseGeometry, Point, BoundingBox
from .nurbs import NURBSCurve, NURBSSurface

class OrganicSurface(BaseGeometry):
    """Organic surface with natural deformations."""
    
    def __init__(self, base_surface: NURBSSurface):
        """Initialize organic surface."""
        super().__init__()
        self.base_surface = base_surface
        self.deformations: List[Dict[str, Any]] = []
        self.features: List[Union[NURBSCurve, NURBSSurface]] = []
    
    @classmethod
    def from_nurbs(cls, surface: NURBSSurface) -> 'OrganicSurface':
        """Create organic surface from NURBS surface."""
        return cls(surface)
    
    @classmethod
    def combine(cls, surfaces: List['OrganicSurface']) -> 'OrganicSurface':
        """Combine multiple organic surfaces."""
        # Create base surface from first surface
        combined = cls(surfaces[0].base_surface)
        
        # Add deformations and features from all surfaces
        for surface in surfaces:
            combined.deformations.extend(surface.deformations)
            combined.features.extend(surface.features)
        
        return combined
    
    def add_random_deformation(self, intensity: float = 0.5) -> None:
        """Add random organic deformation."""
        self.deformations.append({
            'type': 'random',
            'intensity': intensity,
            'seed': np.random.randint(1000)
        })
    
    def add_bumps(self, count: int, height: float, radius: float) -> None:
        """Add organic bumps to surface."""
        self.deformations.append({
            'type': 'bumps',
            'count': count,
            'height': height,
            'radius': radius
        })
    
    def add_feature(self, feature: Union[NURBSCurve, NURBSSurface]) -> None:
        """Add geometric feature to surface."""
        self.features.append(feature)
    
    def rotate(self, angle: float) -> None:
        """Rotate surface around Z axis."""
        c = np.cos(angle)
        s = np.sin(angle)
        rotation = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to control points
        points = self.base_surface.control_points
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                points[i, j] = np.dot(rotation, points[i, j])
    
    def point_at(self, u: float, v: float) -> List[float]:
        """Evaluate surface at parameters (u,v) with deformations."""
        # Get base point
        point = np.array(self.base_surface.point_at(u, v))
        
        # Apply deformations
        for deform in self.deformations:
            if deform['type'] == 'random':
                point += self._apply_random_deform(u, v, deform)
            elif deform['type'] == 'bumps':
                point += self._apply_bumps(u, v, deform)
        
        return point.tolist()
    
    def _apply_random_deform(self, u: float, v: float,
                           deform: Dict[str, Any]) -> np.ndarray:
        """Apply random deformation."""
        np.random.seed(deform['seed'])
        
        # Generate Perlin-like noise
        freq = 5.0
        x = u * freq
        y = v * freq
        
        noise = 0.0
        amplitude = deform['intensity']
        for i in range(4):  # Octaves
            noise += amplitude * self._noise2d(x, y)
            x *= 2
            y *= 2
            amplitude *= 0.5
        
        # Get surface normal
        normal = self.base_surface._compute_normal(u, v)
        return normal * noise
    
    def _apply_bumps(self, u: float, v: float,
                    deform: Dict[str, Any]) -> np.ndarray:
        """Apply bump deformation."""
        # Generate random bump centers
        np.random.seed(0)  # For reproducibility
        centers = []
        for _ in range(deform['count']):
            centers.append((
                np.random.random(),  # u coordinate
                np.random.random()   # v coordinate
            ))
        
        # Calculate bump influence
        total = np.zeros(3)
        for cu, cv in centers:
            dist = np.sqrt((u - cu)**2 + (v - cv)**2)
            if dist < deform['radius']:
                factor = (1 - dist/deform['radius'])**2
                normal = self.base_surface._compute_normal(u, v)
                total += normal * factor * deform['height']
        
        return total
    
    def _noise2d(self, x: float, y: float) -> float:
        """Simple 2D noise function."""
        # Hash coordinates to pseudo-random gradient
        n = int(x + y * 57)
        n = (n << 13) ^ n
        rand = (n * (n * n * 15731 + 789221) + 1376312589)
        return 1.0 - float(rand & 0x7fffffff) / 1073741824.0
    
    def analyze_thickness(self) -> float:
        """Analyze surface thickness."""
        # Use base surface thickness as starting point
        thickness = self.base_surface.analyze_thickness()
        
        # Adjust for deformations
        for deform in self.deformations:
            if deform['type'] == 'random':
                thickness *= (1.0 - 0.2 * deform['intensity'])
            elif deform['type'] == 'bumps':
                thickness = min(thickness, deform['height'])
        
        return thickness
    
    def analyze_overhangs(self) -> float:
        """Analyze surface overhangs."""
        max_angle = self.base_surface.analyze_overhangs()
        
        # Sample deformed surface
        for u in np.linspace(0, 1, 10):
            for v in np.linspace(0, 1, 10):
                point = np.array(self.point_at(u, v))
                normal = self._compute_normal(u, v)
                angle = np.arccos(np.dot(normal, [0, 0, 1])) * 180 / np.pi
                max_angle = max(max_angle, angle)
        
        return max_angle
    
    def analyze_stress_points(self) -> List[Tuple[float, float, float]]:
        """Analyze surface stress points."""
        stress_points = []
        
        # Check base surface stress points
        base_points = self.base_surface.analyze_stress_points()
        stress_points.extend(base_points)
        
        # Check deformation stress points
        for u in np.linspace(0, 1, 10):
            for v in np.linspace(0, 1, 10):
                # Calculate curvature of deformed surface
                curvature = self._compute_curvature(u, v)
                if curvature > 1.0:  # High curvature threshold
                    point = self.point_at(u, v)
                    stress_points.append(tuple(point))  # type: ignore
        
        return stress_points
    
    def thicken_walls(self, min_thickness: float) -> 'BaseGeometry':
        """Thicken surface walls."""
        # Thicken base surface
        thickened_base = self.base_surface.thicken_walls(min_thickness)
        if not isinstance(thickened_base, NURBSSurface):
            return self
        
        # Create new organic surface with thickened base
        thickened = OrganicSurface(thickened_base)
        thickened.deformations = self.deformations.copy()
        thickened.features = self.features.copy()
        
        return thickened
    
    def reduce_overhangs(self, max_angle: float) -> 'BaseGeometry':
        """Reduce surface overhangs."""
        if self.analyze_overhangs() <= max_angle:
            return self
        
        # Reduce base surface overhangs
        reduced_base = self.base_surface.reduce_overhangs(max_angle)
        if not isinstance(reduced_base, NURBSSurface):
            return self
        
        # Create new organic surface with reduced overhangs
        reduced = OrganicSurface(reduced_base)
        
        # Reduce deformation intensities
        for deform in self.deformations:
            if deform['type'] == 'random':
                deform['intensity'] *= 0.7
            elif deform['type'] == 'bumps':
                deform['height'] *= 0.7
        reduced.deformations = self.deformations
        reduced.features = self.features
        
        return reduced
    
    def reinforce_weak_points(self) -> 'BaseGeometry':
        """Reinforce surface weak points."""
        # Reinforce base surface
        reinforced_base = self.base_surface.reinforce_weak_points()
        if not isinstance(reinforced_base, NURBSSurface):
            return self
        
        # Create new organic surface with reinforced base
        reinforced = OrganicSurface(reinforced_base)
        
        # Reduce deformation intensities near stress points
        stress_points = self.analyze_stress_points()
        if stress_points:
            for deform in self.deformations:
                if deform['type'] == 'random':
                    deform['intensity'] *= 0.8
                elif deform['type'] == 'bumps':
                    deform['height'] *= 0.8
        
        reinforced.deformations = self.deformations
        reinforced.features = self.features
        
        return reinforced
    
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