"""
Parametric curve support for organic shape generation.
Provides mathematical functions for generating complex curves and patterns.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Union, Dict, Any, cast
from .base import Point, GeometricEntity, BoundingBox
from .nurbs import NURBSCurve
from abc import ABC, abstractmethod
import math

def convert_points_to_float_lists(points: List[Point]) -> List[List[float]]:
    """Convert a list of Points to a list of float lists."""
    return [[float(x) for x in point.to_array()] for point in points]

def convert_point_to_float_list(point: Point) -> List[float]:
    """Convert a single Point to a list of floats."""
    return [float(x) for x in point.to_array()]

class ParametricCurve(GeometricEntity):
    """Base class for parametric curves."""
    
    def __init__(self):
        super().__init__()
        self._points: List[Point] = []
        
    def sample_points(self, num_points: int = 100) -> List[Point]:
        """Sample points along the curve."""
        points = []
        for i in range(num_points):
            t = float(i) / float(num_points - 1)
            points.append(self.evaluate(t))
        return points

    @abstractmethod
    def evaluate(self, t: float) -> Point:
        """Evaluate curve at parameter t."""
        pass

    def to_nurbs(self) -> NURBSCurve:
        """Convert to NURBS representation."""
        points = self.sample_points()
        float_points = convert_points_to_float_lists(points)
        return NURBSCurve.from_points(float_points)

    def _compute_bounding_box(self) -> None:
        """Compute bounding box."""
        points = self.sample_points()
        points_array = np.array([p.to_array() for p in points])
        min_point = Point(*np.min(points_array, axis=0))
        max_point = Point(*np.max(points_array, axis=0))
        self._bounding_box = BoundingBox(min_point, max_point)
    
    def _compute_volume(self) -> None:
        """Compute volume (zero for curves)."""
        self._volume = 0.0
    
    def _compute_surface_area(self) -> None:
        """Compute surface area (zero for curves)."""
        self._surface_area = 0.0

    @staticmethod
    def create_circle(center: List[float], radius: float) -> NURBSCurve:
        """Create a NURBS circle."""
        from math import pi, cos, sin
        points = []
        for i in range(8):
            angle = 2 * pi * float(i) / 8.0
            x = center[0] + radius * cos(angle)
            y = center[1] + radius * sin(angle)
            z = center[2]
            points.append([float(x), float(y), float(z)])
        return NURBSCurve.from_points(points)

    def create_parametric_curves(self) -> List['ParametricCurve']:
        """Create a list of parametric curves."""
        nurbs = self.to_nurbs()
        return [cast(ParametricCurve, nurbs)]

class Spiral(ParametricCurve):
    """Parametric spiral curve."""
    
    def __init__(self, radius: float, height: float, turns: float):
        super().__init__()
        self.radius = float(radius)
        self.height = float(height)
        self.turns = float(turns)
        
    def evaluate(self, t: float) -> Point:
        from math import pi, cos, sin
        angle = 2 * pi * t * self.turns
        x = self.radius * cos(angle) * t
        y = self.radius * sin(angle) * t
        z = self.height * t
        return Point(float(x), float(y), float(z))

class FlowerPetal(ParametricCurve):
    """Parametric flower petal curve."""
    
    def __init__(self, radius: float, petals: float):
        super().__init__()
        self.radius = float(radius)
        self.petals = float(petals)
        
    def evaluate(self, t: float) -> Point:
        angle = 2 * np.pi * t
        r = self.radius * np.sin(self.petals * angle)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return Point(float(x), float(y), 0.0)

class Helix(ParametricCurve):
    """Parametric helix curve."""
    
    def __init__(self, radius: float, height: float, turns: float):
        super().__init__()
        self.radius = float(radius)
        self.height = float(height)
        self.turns = float(turns)
        
    def evaluate(self, t: float) -> Point:
        from math import pi, cos, sin
        angle = 2 * pi * t * self.turns
        x = self.radius * cos(angle)
        y = self.radius * sin(angle)
        z = self.height * t
        return Point(float(x), float(y), float(z))

class Lissajous(ParametricCurve):
    """Parametric Lissajous curve."""
    
    def __init__(self, a: float, b: float, delta: float):
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.delta = float(delta)
        
    def evaluate(self, t: float) -> Point:
        from math import pi, cos, sin
        x = cos(self.a * t * 2 * pi)
        y = sin(self.b * t * 2 * pi + self.delta)
        return Point(float(x), float(y), 0.0)

class SuperShape(ParametricCurve):
    """Superformula-based curve for complex organic shapes."""
    
    def __init__(self, a: float = 1.0, b: float = 1.0, m1: float = 7.0,
                 m2: float = 3.0, n1: float = 0.2, n2: float = 1.7, n3: float = 1.7):
        super().__init__()
        self.params = (a, b, m1, m2, n1, n2, n3)
        
    def evaluate(self, t: float) -> Point:
        a, b, m1, m2, n1, n2, n3 = self.params
        phi = t * 2 * np.pi
        
        # Superformula
        part1 = (1/a) * np.abs(np.cos(m1*phi/4))**n2
        part2 = (1/b) * np.abs(np.sin(m2*phi/4))**n3
        r = float((part1 + part2)**(-1/n1))
        
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = 0.0
        return Point(float(x), float(y), float(z))

class BezierCurve(ParametricCurve):
    """Bézier curve with variable control points."""
    
    def __init__(self, control_points: List[Point]):
        super().__init__()
        self.control_points = control_points
        
    def evaluate(self, t: float) -> Point:
        n = len(self.control_points) - 1
        point = np.zeros(3)
        for i, p in enumerate(self.control_points):
            # Calculate binomial coefficient using multiplicative formula
            coeff = 1.0
            for j in range(i):
                coeff *= (n - j) / (j + 1)
            coeff *= (1-t)**(n-i) * t**i
            point += float(coeff) * np.array([p.x, p.y, p.z])
        return Point(float(point[0]), float(point[1]), float(point[2]))

class RoseCurve(ParametricCurve):
    """Rose curve (rhodonea) for petal-like patterns."""
    
    def __init__(self,
                 radius: float = 1.0,
                 n: int = 3,
                 d: int = 1,
                 height_factor: float = 0.2):
        """
        Args:
            radius: Base radius
            n, d: Determines number of petals (n/d petals if n odd, 2n/d if n even)
            height_factor: Controls 3D variation
        """
        def rose_curve(t: float) -> Tuple[float, float, float]:
            k = n / d
            r = radius * np.cos(k * t)
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = height_factor * np.sin(k * t)
            return (x, y, z)
        
        super().__init__()
        self.curve_func = rose_curve

class EpicycloidCurve(ParametricCurve):
    """Epicycloid curve for complex geometric patterns."""
    
    def __init__(self,
                 R: float = 1.0,  # Fixed circle radius
                 r: float = 0.3,  # Moving circle radius
                 d: float = 0.5): # Distance from center
        def epicycloid_curve(t: float) -> Tuple[float, float, float]:
            x = (R+r) * np.cos(t) - d * np.cos((R+r)*t/r)
            y = (R+r) * np.sin(t) - d * np.sin((R+r)*t/r)
            z = 0.1 * np.sin(5*t)  # Add slight 3D variation
            return (x, y, z)
        
        super().__init__()
        self.curve_func = epicycloid_curve

class FractalCurve(ParametricCurve):
    """Base class for fractal-based curves."""
    
    def __init__(self,
                 iterations: int = 3,
                 scale: float = 1.0):
        self.iterations = iterations
        self.scale = scale
        super().__init__()
        self.curve_func = self._fractal_curve

    def _fractal_curve(self, t: float) -> Tuple[float, float, float]:
        raise NotImplementedError

class DragonCurve(FractalCurve):
    """Dragon curve implementation."""
    
    def _fractal_curve(self, t: float) -> Tuple[float, float, float]:
        angle = 0.0
        x, y = 0.0, 0.0
        for i in range(self.iterations):
            angle += t * np.pi/4
            factor = 0.7**i
            x += factor * np.cos(angle)
            y += factor * np.sin(angle)
        return (self.scale * x, self.scale * y, 0.0)

class HypocycloidCurve(ParametricCurve):
    """Parametric hypocycloid curve."""
    
    def __init__(self, R: float, r: float):
        super().__init__()
        self.R = float(R)
        self.r = float(r)
        
    def evaluate(self, t: float) -> Point:
        from math import pi, cos, sin
        theta = 2 * pi * t
        x = (self.R - self.r) * cos(theta) + self.r * cos((self.R - self.r) * theta / self.r)
        y = (self.R - self.r) * sin(theta) - self.r * sin((self.R - self.r) * theta / self.r)
        return Point(float(x), float(y), 0.0)

class TorusKnotCurve(ParametricCurve):
    """Parametric torus knot curve."""
    
    def __init__(self, p: float, q: float, radius: float):
        super().__init__()
        self.p = float(p)
        self.q = float(q)
        self.radius = float(radius)
        
    def evaluate(self, t: float) -> Point:
        from math import pi, cos, sin
        theta = 2 * pi * t
        r = self.radius * (2 + cos(self.q * theta))
        x = r * cos(self.p * theta)
        y = r * sin(self.p * theta)
        z = self.radius * sin(self.q * theta)
        return Point(float(x), float(y), float(z))

class HermiteSpline(ParametricCurve):
    """Parametric Hermite spline curve."""
    
    def __init__(self, points: List[Point], tangents: List[Point]):
        super().__init__()
        if len(points) < 2 or len(points) != len(tangents):
            raise ValueError("Need at least 2 points and matching number of tangents")
        self.points = points
        self.tangents = tangents
        
    def evaluate(self, t: float) -> Point:
        # Hermite basis functions
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        
        p0 = self.points[0].to_array()
        p1 = self.points[1].to_array()
        m0 = self.tangents[0].to_array()
        m1 = self.tangents[1].to_array()
        
        result = h00*p0 + h10*m0 + h01*p1 + h11*m1
        return Point(float(result[0]), float(result[1]), float(result[2]))

    def to_nurbs(self) -> NURBSCurve:
        """Convert to NURBS representation."""
        points = self.sample_points()
        float_points = convert_points_to_float_lists(points)
        return NURBSCurve.from_points(float_points)

class CombinedPattern:
    """Factory for creating combined parametric patterns."""
    
    @staticmethod
    def create_pattern(pattern_type: str, size: float = 1.0, complexity: float = 1.0) -> ParametricCurve:
        """Create a parametric pattern of the specified type."""
        if pattern_type == 'torus_knot':
            p = float(2 + complexity * 2)
            q = float(3 + complexity * 2)
            return TorusKnotCurve(p=p, q=q, radius=size)
        elif pattern_type == 'hypocycloid':
            return HypocycloidCurve(R=size, r=size*0.3)
        elif pattern_type == 'hermite':
            p0 = Point(0, 0, 0)
            p1 = Point(size, 0, 0)
            t0 = Point(0, size*complexity, 0)
            t1 = Point(0, -size*complexity, 0)
            return HermiteSpline([p0, p1], [t0, t1])
        else:
            return FlowerPetal(radius=size, petals=0.3*complexity)

class PatternGenerator:
    """Generate patterns of curves with transformations."""
    
    @staticmethod
    def circular_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        radius: float,
        z_offset: float = 0.0,
        scale_factor: Optional[float] = None,
        rotation_offset: float = 0.0
    ) -> List[ParametricCurve]:
        """Create a circular pattern of curves."""
        patterns = []
        for i in range(num_copies):
            angle = 2 * np.pi * i / num_copies + rotation_offset
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, radius * c],
                [s, c, 0, radius * s],
                [0, 0, 1, z_offset],
                [0, 0, 0, 1]
            ])
            
            if scale_factor is not None:
                scale = 1.0 + (scale_factor - 1.0) * i / num_copies
                transform[:3, :3] *= scale
            
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_point = Point(
                    float(transformed[0] / transformed[3]),
                    float(transformed[1] / transformed[3]),
                    float(transformed[2] / transformed[3])
                )
                new_points.append(new_point)
            
            float_points = convert_points_to_float_lists(new_points)
            nurbs_curve = NURBSCurve.from_points(float_points)
            patterns.append(cast(ParametricCurve, nurbs_curve))
            
        return patterns
    
    @staticmethod
    def spiral_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        start_radius: float,
        end_radius: float,
        height: float = 0.0,
        rotation_offset: float = 0.0
    ) -> List[ParametricCurve]:
        patterns = []
        for i in range(num_copies):
            t = i / (num_copies - 1)
            angle = 2 * np.pi * t * 3 + rotation_offset
            radius = start_radius + (end_radius - start_radius) * t
            
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, radius * c],
                [s, c, 0, radius * s],
                [0, 0, 1, height * t],
                [0, 0, 0, 1]
            ])
            
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_point = Point(
                    float(transformed[0] / transformed[3]),
                    float(transformed[1] / transformed[3]),
                    float(transformed[2] / transformed[3])
                )
                new_points.append(new_point)
            
            float_points = convert_points_to_float_lists(new_points)
            nurbs_curve = NURBSCurve.from_points(float_points)
            patterns.append(cast(ParametricCurve, nurbs_curve))
        
        return patterns
    
    @staticmethod
    def fibonacci_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        max_radius: float,
        scale_factor: float = 0.95
    ) -> List[ParametricCurve]:
        """Create a Fibonacci spiral pattern of curves."""
        patterns: List[ParametricCurve] = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(num_copies):
            angle = i * golden_angle
            radius = max_radius * np.sqrt(i / num_copies)
            scale = scale_factor ** i
            
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c*scale, -s*scale, 0, radius * c],
                [s*scale, c*scale, 0, radius * s],
                [0, 0, scale, 0],
                [0, 0, 0, 1]
            ])
            
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_point = Point(
                    float(transformed[0] / transformed[3]),
                    float(transformed[1] / transformed[3]),
                    float(transformed[2] / transformed[3])
                )
                new_points.append(new_point)
            
            float_points = convert_points_to_float_lists(new_points)
            nurbs_curve = NURBSCurve.from_points(float_points)
            patterns.append(cast(ParametricCurve, nurbs_curve))
        
        return patterns

    @staticmethod
    def fractal_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        scale_range: Tuple[float, float] = (0.2, 1.0),
        rotation_base: float = np.pi/3
    ) -> List[ParametricCurve]:
        """Create a fractal-like pattern of curves."""
        patterns: List[ParametricCurve] = []
        for i in range(num_copies):
            scale = scale_range[0] + (scale_range[1] - scale_range[0]) * (i/num_copies)
            angle = rotation_base * i
            
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c*scale, -s*scale, 0, scale * np.cos(i*np.pi/4)],
                [s*scale, c*scale, 0, scale * np.sin(i*np.pi/4)],
                [0, 0, scale, 0.1 * scale * np.sin(i*np.pi/3)],
                [0, 0, 0, 1]
            ])
            
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_point = Point(
                    float(transformed[0] / transformed[3]),
                    float(transformed[1] / transformed[3]),
                    float(transformed[2] / transformed[3])
                )
                new_points.append(new_point)
            
            float_points = convert_points_to_float_lists(new_points)
            nurbs_curve = NURBSCurve.from_points(float_points)
            patterns.append(cast(ParametricCurve, nurbs_curve))
        
        return patterns
    
    @staticmethod
    def radial_wave_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        radius: float,
        wave_freq: float = 3.0,
        wave_amp: float = 0.2
    ) -> List[ParametricCurve]:
        """Create a pattern with wave-like radial variation."""
        patterns: List[ParametricCurve] = []
        for i in range(num_copies):
            angle = 2 * np.pi * i / num_copies
            # Add wave variation to radius
            r = radius * (1 + wave_amp * np.sin(wave_freq * angle))
            
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, r * c],
                [s, c, 0, r * s],
                [0, 0, 1, 0.1 * np.sin(wave_freq * angle)],
                [0, 0, 0, 1]
            ])
            
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_point = Point(
                    float(transformed[0] / transformed[3]),
                    float(transformed[1] / transformed[3]),
                    float(transformed[2] / transformed[3])
                )
                new_points.append(new_point)
            
            float_points = convert_points_to_float_lists(new_points)
            nurbs_curve = NURBSCurve.from_points(float_points)
            patterns.append(cast(ParametricCurve, nurbs_curve))
        
        return patterns

class OrganicPatternFactory:
    """Factory for creating organic patterns."""
    
    @staticmethod
    def create_flower(
        num_petals: int,
        petal_length: float,
        petal_width: float,
        curve_factor: float = 0.3,
        center_radius: float = 0.2,
        pattern_type: str = 'circular',
        center_type: str = 'spiral'
    ) -> List[GeometricEntity]:
        """Create a flower pattern with petals."""
        # Create base petal
        base_petal = FlowerPetal(
            radius=petal_length,
            petals=petal_width
        )
        
        # Generate petal pattern based on type
        petals: List[ParametricCurve]
        if pattern_type == 'fibonacci':
            petals = PatternGenerator.fibonacci_pattern(
                base_petal,
                num_petals,
                max_radius=petal_length * 0.2,
                scale_factor=0.97
            )
        elif pattern_type == 'spiral':
            petals = PatternGenerator.spiral_pattern(
                base_petal,
                num_petals,
                start_radius=center_radius,
                end_radius=center_radius * 2,
                height=petal_length * 0.1
            )
        else:  # circular
            petals = PatternGenerator.circular_pattern(
                base_petal,
                num_petals,
                radius=center_radius,
                rotation_offset=np.random.random() * np.pi/6
            )
        
        # Create center based on type
        center: Optional[GeometricEntity] = None
        if center_type == 'spiral':
            spiral = Spiral(
                radius=center_radius * 0.2,
                height=center_radius * 0.2,
                turns=3
            )
            center = cast(GeometricEntity, spiral.to_nurbs())
        elif center_type == 'lissajous':
            lissajous = Lissajous(
                a=center_radius,
                b=center_radius,
                delta=np.random.random() * np.pi/2
            )
            center = cast(GeometricEntity, lissajous.to_nurbs())
        else:  # simple circle
            points = []
            for i in range(8):
                angle = 2 * np.pi * i / 8.0
                x = center_radius * np.cos(angle)
                y = center_radius * np.sin(angle)
                points.append([float(x), float(y), 0.0])
            center = cast(GeometricEntity, NURBSCurve.from_points(points))
        
        result: List[GeometricEntity] = []
        if center is not None:
            result.append(center)
        result.extend(cast(List[GeometricEntity], petals))
        return result

class PatternAnalyzer:
    """AI-driven pattern analysis and selection."""
    
    @staticmethod
    def analyze_shape_requirements(description: str) -> Dict[str, Any]:
        """Analyze text description to determine optimal pattern parameters."""
        requirements: Dict[str, Any] = {
            'pattern_type': 'circular',  # default
            'complexity': 1.0,
            'regularity': 1.0,
            'dimensionality': '2D',
            'symmetry': True,
            'curve_types': [],  # Initialize as empty list
            'combination_mode': None
        }
        
        curve_types: List[str] = []
        
        # Natural/organic patterns
        if any(word in description.lower() for word in 
               ['natural', 'organic', 'flowing', 'random', 'irregular']):
            requirements.update({
                'pattern_type': 'fibonacci',
                'regularity': 0.7,
                'complexity': 1.2
            })
            curve_types.extend(['rose', 'bezier'])
        
        # Geometric/regular patterns
        if any(word in description.lower() for word in 
               ['geometric', 'regular', 'symmetric', 'even']):
            requirements.update({
                'pattern_type': 'circular',
                'regularity': 1.0,
                'complexity': 0.8
            })
            curve_types.extend(['epicycloid', 'hypocycloid'])
        
        # Complex/intricate patterns
        if any(word in description.lower() for word in 
               ['complex', 'intricate', 'detailed', 'ornate']):
            requirements.update({
                'pattern_type': 'fractal',
                'complexity': 1.5,
                'regularity': 0.8
            })
            curve_types.extend(['torus_knot', 'supershape'])
        
        # 3D variations
        if any(word in description.lower() for word in 
               ['3d', 'dimensional', 'depth', 'layered']):
            requirements['dimensionality'] = '3D'
            curve_types.append('torus_knot')
        
        requirements['curve_types'] = curve_types
        return requirements
    
    @staticmethod
    def get_optimal_pattern(requirements: Dict[str, Any]) -> str:
        """Determine the optimal pattern type based on requirements."""
        pattern_scores = {
            'circular': 0,
            'spiral': 0,
            'fibonacci': 0,
            'fractal': 0,
            'radial_wave': 0
        }
        
        # Score patterns based on requirements
        if requirements['regularity'] > 0.9:
            pattern_scores['circular'] += 2
        if requirements['complexity'] > 1.2:
            pattern_scores['fractal'] += 2
            pattern_scores['fibonacci'] += 1
        if requirements['dimensionality'] == '3D':
            pattern_scores['spiral'] += 1
            pattern_scores['radial_wave'] += 1
        
        # Consider curve types in scoring
        if 'torus_knot' in requirements['curve_types']:
            pattern_scores['spiral'] += 1
        if 'hypocycloid' in requirements['curve_types']:
            pattern_scores['circular'] += 1
        
        # Return pattern with highest score
        return max(pattern_scores.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def create_curve_from_type(
        curve_type: str,
        size: float = 1.0,
        complexity: float = 1.0
    ) -> ParametricCurve:
        """Create a curve instance based on type and parameters."""
        if curve_type == 'torus_knot':
            p = int(2 + complexity * 2)
            q = int(3 + complexity * 2)
            return TorusKnotCurve(p=p, q=q, radius=size)
        elif curve_type == 'hypocycloid':
            return HypocycloidCurve(R=size, r=size*0.3)
        elif curve_type == 'hermite':
            # Create a smooth curve with controlled complexity
            p0 = Point(0, 0, 0)
            p1 = Point(size, 0, 0)
            t0 = Point(0, size*complexity, 0)
            t1 = Point(0, -size*complexity, 0)
            return HermiteSpline([p0, p1], [t0, t1])
        # ... handle other curve types ...
        else:
            return FlowerPetal(radius=size, petals=0.3*complexity)

def from_points(points: List[Point]) -> List[NURBSCurve]:
    """Convert points to NURBS curves."""
    float_points = convert_points_to_float_lists(points)
    return [NURBSCurve.from_points(float_points)]

def create_circle(center: Point, radius: float) -> List[NURBSCurve]:
    """Create a circle as NURBS curves."""
    center_coords = convert_point_to_float_list(center)
    points = []
    for i in range(8):
        angle = 2 * np.pi * i / 8.0
        x = center.x + radius * np.cos(angle)
        y = center.y + radius * np.sin(angle)
        z = center.z
        points.append([float(x), float(y), float(z)])
    return [NURBSCurve.from_points(points)]

def create_parametric_curve(t: float, x_func: Callable[[float], float], 
                          y_func: Callable[[float], float], 
                          z_func: Callable[[float], float]) -> Point:
    """Create a parametric curve point."""
    from math import sin, cos, pi  # Import math functions explicitly
    return Point(x_func(t), y_func(t), z_func(t))

# Update variable types from int to float where needed
def create_shape_parameters() -> Dict[str, float]:
    """Create shape parameters with proper float types."""
    params = {
        'width': 1.0,
        'height': 1.0,
        'depth': 1.0,
        'radius': 1.0
    }
    return params 