"""
Parametric curve support for organic shape generation.
Provides mathematical functions for generating complex curves and patterns.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Union, Dict, Any
from .base import Point, GeometricEntity
from .nurbs import NURBSCurve

class ParametricCurve(GeometricEntity):
    """Base class for parametric curves."""
    
    def __init__(self, 
                 curve_func: Callable[[float], Tuple[float, float, float]],
                 t_range: Tuple[float, float] = (0, 2*np.pi),
                 num_samples: int = 100):
        self.curve_func = curve_func
        self.t_range = t_range
        self.num_samples = num_samples
        
    def sample_points(self) -> List[Point]:
        """Sample points along the curve."""
        t_vals = np.linspace(self.t_range[0], self.t_range[1], self.num_samples)
        return [Point(*self.curve_func(t)) for t in t_vals]
    
    def to_nurbs(self) -> NURBSCurve:
        """Convert to NURBS representation."""
        points = self.sample_points()
        return NURBSCurve.from_points(points)

class Spiral(ParametricCurve):
    """Logarithmic or Archimedean spiral curve."""
    
    def __init__(self,
                 a: float = 1.0,
                 b: float = 0.2,
                 height: float = 0.0,
                 num_turns: float = 2.0,
                 spiral_type: str = 'logarithmic'):
        """
        Args:
            a: Base radius
            b: Growth rate
            height: Total height change
            num_turns: Number of complete turns
            spiral_type: 'logarithmic' or 'archimedean'
        """
        t_range = (0, 2*np.pi * num_turns)
        
        def spiral_curve(t: float) -> Tuple[float, float, float]:
            if spiral_type == 'logarithmic':
                r = a * np.exp(b * t)
            else:  # archimedean
                r = a + b * t
            
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = height * t / (2*np.pi * num_turns)
            return (x, y, z)
        
        super().__init__(spiral_curve, t_range)

class FlowerPetal(ParametricCurve):
    """Parametric curve for flower petal shapes."""
    
    def __init__(self,
                 length: float = 1.0,
                 width: float = 0.3,
                 curve_factor: float = 0.3,
                 harmonics: int = 3,
                 asymmetry: float = 0.0):
        def petal_curve(t: float) -> Tuple[float, float, float]:
            # Add asymmetry to create more natural looking petals
            asym = 1.0 + asymmetry * np.sin(2*t)
            x = length * np.cos(t) * (1 + curve_factor * np.sin(harmonics*t)) * asym
            y = width * np.sin(t) * (1 + curve_factor * np.sin(harmonics*t))
            z = 0.1 * np.sin(t) * (1 + 0.5 * np.sin(2*t))  # More complex 3D curvature
            return (x, y, z)
            
        super().__init__(petal_curve)

class Helix(ParametricCurve):
    """Helical curve with variable radius."""
    
    def __init__(self,
                 radius: float = 1.0,
                 pitch: float = 1.0,
                 num_turns: float = 3.0,
                 taper: float = 0.0):
        t_range = (0, 2*np.pi * num_turns)
        
        def helix_curve(t: float) -> Tuple[float, float, float]:
            # Apply taper to radius
            r = radius * (1.0 - taper * t/(2*np.pi * num_turns))
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = pitch * t/(2*np.pi)
            return (x, y, z)
        
        super().__init__(helix_curve, t_range)

class Lissajous(ParametricCurve):
    """Lissajous curve for complex symmetric patterns."""
    
    def __init__(self,
                 a: float = 1.0,
                 b: float = 1.0,
                 freq_x: int = 3,
                 freq_y: int = 2,
                 phase: float = np.pi/2):
        def lissajous_curve(t: float) -> Tuple[float, float, float]:
            x = a * np.sin(freq_x * t)
            y = b * np.sin(freq_y * t + phase)
            z = 0.0
            return (x, y, z)
        
        super().__init__(lissajous_curve)

class SuperShape(ParametricCurve):
    """Superformula-based curve for complex organic shapes."""
    
    def __init__(self,
                 a: float = 1.0,
                 b: float = 1.0,
                 m1: float = 7.0,
                 m2: float = 3.0,
                 n1: float = 0.2,
                 n2: float = 1.7,
                 n3: float = 1.7):
        def supershape_curve(t: float) -> Tuple[float, float, float]:
            phi = t
            
            # Superformula
            part1 = (1/a) * np.abs(np.cos(m1*phi/4))**n2
            part2 = (1/b) * np.abs(np.sin(m2*phi/4))**n3
            r = (part1 + part2)**(-1/n1)
            
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = 0.0
            return (x, y, z)
        
        super().__init__(supershape_curve)

class BezierCurve(ParametricCurve):
    """Bézier curve with variable control points."""
    
    def __init__(self, control_points: List[Point]):
        def bezier_curve(t: float) -> Tuple[float, float, float]:
            n = len(control_points) - 1
            point = np.zeros(3)
            for i, p in enumerate(control_points):
                # Bernstein polynomial
                coeff = np.math.comb(n, i) * (1-t)**(n-i) * t**i
                point += coeff * np.array([p.x, p.y, p.z])
            return tuple(point)
        
        super().__init__(bezier_curve, t_range=(0, 1))

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
        
        super().__init__(rose_curve)

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
        
        super().__init__(epicycloid_curve)

class FractalCurve(ParametricCurve):
    """Base class for fractal-based curves."""
    
    def __init__(self,
                 iterations: int = 3,
                 scale: float = 1.0):
        self.iterations = iterations
        self.scale = scale
        super().__init__(self._fractal_curve)
    
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
    """Hypocycloid curve for star-like patterns."""
    
    def __init__(self,
                 R: float = 1.0,    # Fixed circle radius
                 r: float = 0.3,    # Moving circle radius
                 d: float = 0.5,    # Distance from center
                 height_var: float = 0.1):
        def hypocycloid_curve(t: float) -> Tuple[float, float, float]:
            x = (R-r) * np.cos(t) + d * np.cos((R-r)*t/r)
            y = (R-r) * np.sin(t) - d * np.sin((R-r)*t/r)
            z = height_var * np.sin((R/r)*t)  # Add 3D variation
            return (x, y, z)
        
        super().__init__(hypocycloid_curve)

class TorusKnotCurve(ParametricCurve):
    """Torus knot curve for complex 3D patterns."""
    
    def __init__(self,
                 p: int = 2,        # Number of winds around torus
                 q: int = 3,        # Number of winds through torus
                 R: float = 1.0,    # Major radius
                 r: float = 0.3):   # Minor radius
        def torus_knot_curve(t: float) -> Tuple[float, float, float]:
            # Parametric equations for torus knot
            pt = p * t
            qt = q * t
            x = R * (2 + np.cos(qt)) * np.cos(pt) / 3
            y = R * (2 + np.cos(qt)) * np.sin(pt) / 3
            z = R * np.sin(qt) / 3
            return (x, y, z)
        
        super().__init__(torus_knot_curve)

class HermiteSpline(ParametricCurve):
    """Hermite spline for smooth interpolation."""
    
    def __init__(self,
                 p0: Point,         # Start point
                 p1: Point,         # End point
                 t0: Point,         # Start tangent
                 t1: Point):        # End tangent
        def hermite_curve(t: float) -> Tuple[float, float, float]:
            # Hermite basis functions
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2
            
            # Interpolate each component
            x = h00*p0.x + h10*t0.x + h01*p1.x + h11*t1.x
            y = h00*p0.y + h10*t0.y + h01*p1.y + h11*t1.y
            z = h00*p0.z + h10*t0.z + h01*p1.z + h11*t1.z
            return (x, y, z)
        
        super().__init__(hermite_curve, t_range=(0, 1))

class CombinedPattern:
    """Support for combining multiple patterns."""
    
    def __init__(self,
                 base_patterns: List[List[ParametricCurve]],
                 weights: Optional[List[float]] = None):
        self.patterns = base_patterns
        self.weights = weights or [1.0] * len(base_patterns)
        
    def blend(self, t: float = 0.5) -> List[ParametricCurve]:
        """Blend between patterns based on weights and parameter t."""
        result = []
        total_weight = sum(self.weights)
        normalized_weights = [w/total_weight for w in self.weights]
        
        # Find maximum number of curves in any pattern
        max_curves = max(len(pattern) for pattern in self.patterns)
        
        for i in range(max_curves):
            points = []
            total_points = 0
            
            # Collect points from each pattern
            for pattern, weight in zip(self.patterns, normalized_weights):
                if i < len(pattern):
                    curve_points = pattern[i].sample_points()
                    points.append((curve_points, weight))
                    total_points = max(total_points, len(curve_points))
            
            # Blend points
            blended_points = []
            for j in range(total_points):
                x, y, z = 0, 0, 0
                total_w = 0
                
                for curve_points, weight in points:
                    if j < len(curve_points):
                        p = curve_points[j]
                        w = weight * (1 - abs(2*t - 1))  # Smooth transition
                        x += p.x * w
                        y += p.y * w
                        z += p.z * w
                        total_w += w
                
                if total_w > 0:
                    blended_points.append(Point(x/total_w, y/total_w, z/total_w))
            
            if blended_points:
                result.append(NURBSCurve.from_points(blended_points))
        
        return result

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
            
            # Apply scale if specified
            if scale_factor is not None:
                scale = 1.0 + (scale_factor - 1.0) * i / num_copies
                transform[:3, :3] *= scale
            
            # Create new curve with transformed points
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_points.append(Point(
                    transformed[0] / transformed[3],
                    transformed[1] / transformed[3],
                    transformed[2] / transformed[3]
                ))
            
            patterns.append(NURBSCurve.from_points(new_points))
            
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
        """Create a spiral pattern of curves."""
        patterns = []
        for i in range(num_copies):
            t = i / (num_copies - 1)
            angle = 2 * np.pi * t * 3 + rotation_offset  # 3 turns
            radius = start_radius + (end_radius - start_radius) * t
            
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, radius * c],
                [s, c, 0, radius * s],
                [0, 0, 1, height * t],
                [0, 0, 0, 1]
            ])
            
            # Create new curve
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_points.append(Point(
                    transformed[0] / transformed[3],
                    transformed[1] / transformed[3],
                    transformed[2] / transformed[3]
                ))
            
            patterns.append(NURBSCurve.from_points(new_points))
        
        return patterns
    
    @staticmethod
    def fibonacci_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        max_radius: float,
        scale_factor: float = 0.95
    ) -> List[ParametricCurve]:
        """Create a Fibonacci spiral pattern of curves."""
        patterns = []
        golden_angle = np.pi * (3 - np.sqrt(5))  # ≈ 137.5 degrees
        
        for i in range(num_copies):
            angle = i * golden_angle
            # Radius grows as square root of i
            radius = max_radius * np.sqrt(i / num_copies)
            scale = scale_factor ** i
            
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c*scale, -s*scale, 0, radius * c],
                [s*scale, c*scale, 0, radius * s],
                [0, 0, scale, 0],
                [0, 0, 0, 1]
            ])
            
            # Create new curve
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_points.append(Point(
                    transformed[0] / transformed[3],
                    transformed[1] / transformed[3],
                    transformed[2] / transformed[3]
                ))
            
            patterns.append(NURBSCurve.from_points(new_points))
        
        return patterns

    @staticmethod
    def fractal_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        scale_range: Tuple[float, float] = (0.2, 1.0),
        rotation_base: float = np.pi/3
    ) -> List[ParametricCurve]:
        """Create a fractal-like pattern of curves."""
        patterns = []
        for i in range(num_copies):
            scale = scale_range[0] + (scale_range[1] - scale_range[0]) * (i/num_copies)
            angle = rotation_base * i
            
            # Create transformation matrix with fractal properties
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c*scale, -s*scale, 0, scale * np.cos(i*np.pi/4)],
                [s*scale, c*scale, 0, scale * np.sin(i*np.pi/4)],
                [0, 0, scale, 0.1 * scale * np.sin(i*np.pi/3)],
                [0, 0, 0, 1]
            ])
            
            # Create new curve
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_points.append(Point(
                    transformed[0] / transformed[3],
                    transformed[1] / transformed[3],
                    transformed[2] / transformed[3]
                ))
            
            patterns.append(NURBSCurve.from_points(new_points))
        
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
        patterns = []
        for i in range(num_copies):
            angle = 2 * np.pi * i / num_copies
            # Add wave variation to radius
            r = radius * (1 + wave_amp * np.sin(wave_freq * angle))
            
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, r * c],
                [s, c, 0, r * s],
                [0, 0, 1, 0.1 * np.sin(wave_freq * angle)],
                [0, 0, 0, 1]
            ])
            
            # Create new curve
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_points.append(Point(
                    transformed[0] / transformed[3],
                    transformed[1] / transformed[3],
                    transformed[2] / transformed[3]
                ))
            
            patterns.append(NURBSCurve.from_points(new_points))
        
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
        # Create base petal with random asymmetry
        base_petal = FlowerPetal(
            length=petal_length,
            width=petal_width,
            curve_factor=curve_factor,
            asymmetry=0.1 * np.random.random()
        )
        
        # Generate petal pattern based on type
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
                rotation_offset=np.random.random() * np.pi/6  # Slight random rotation
            )
        
        # Create center based on type
        if center_type == 'spiral':
            center = Spiral(
                a=center_radius * 0.2,
                b=0.1,
                height=center_radius * 0.2,
                num_turns=3
            ).to_nurbs()
        elif center_type == 'lissajous':
            center = Lissajous(
                a=center_radius,
                b=center_radius,
                freq_x=3,
                freq_y=4
            ).to_nurbs()
        else:  # simple circle
            center = NURBSCurve.create_circle(center_radius)
        
        return [center] + petals 

class PatternAnalyzer:
    """AI-driven pattern analysis and selection."""
    
    @staticmethod
    def analyze_shape_requirements(description: str) -> Dict[str, Any]:
        """Analyze text description to determine optimal pattern parameters."""
        requirements = {
            'pattern_type': 'circular',  # default
            'complexity': 1.0,
            'regularity': 1.0,
            'dimensionality': '2D',
            'symmetry': True,
            'curve_types': [],
            'combination_mode': None
        }
        
        # Natural/organic patterns
        if any(word in description.lower() for word in 
               ['natural', 'organic', 'flowing', 'random', 'irregular']):
            requirements.update({
                'pattern_type': 'fibonacci',
                'regularity': 0.7,
                'complexity': 1.2,
                'curve_types': ['rose', 'bezier']
            })
        
        # Geometric/regular patterns
        if any(word in description.lower() for word in 
               ['geometric', 'regular', 'symmetric', 'even']):
            requirements.update({
                'pattern_type': 'circular',
                'regularity': 1.0,
                'complexity': 0.8,
                'curve_types': ['epicycloid', 'hypocycloid']
            })
        
        # Complex/intricate patterns
        if any(word in description.lower() for word in 
               ['complex', 'intricate', 'detailed', 'ornate']):
            requirements.update({
                'pattern_type': 'fractal',
                'complexity': 1.5,
                'regularity': 0.8,
                'curve_types': ['torus_knot', 'supershape']
            })
        
        # Spiral patterns
        if any(word in description.lower() for word in 
               ['spiral', 'swirl', 'twist', 'coil']):
            requirements.update({
                'pattern_type': 'spiral',
                'complexity': 1.2,
                'regularity': 0.9,
                'curve_types': ['spiral', 'helix']
            })
        
        # Wave patterns
        if any(word in description.lower() for word in 
               ['wave', 'ripple', 'undulating']):
            requirements.update({
                'pattern_type': 'radial_wave',
                'complexity': 1.1,
                'regularity': 0.85,
                'curve_types': ['lissajous', 'hermite']
            })
        
        # 3D variations
        if any(word in description.lower() for word in 
               ['3d', 'dimensional', 'depth', 'layered']):
            requirements['dimensionality'] = '3D'
            requirements['curve_types'].append('torus_knot')
        
        # Pattern combinations
        if any(word in description.lower() for word in 
               ['mixed', 'combined', 'blend', 'hybrid']):
            requirements['combination_mode'] = 'blend'
        elif any(word in description.lower() for word in 
                ['layered', 'stacked', 'overlaid']):
            requirements['combination_mode'] = 'layer'
        
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
            return TorusKnotCurve(p=p, q=q, R=size, r=size*0.3)
        elif curve_type == 'hypocycloid':
            return HypocycloidCurve(R=size, r=size*0.3, height_var=0.1*complexity)
        elif curve_type == 'hermite':
            # Create a smooth curve with controlled complexity
            p0 = Point(0, 0, 0)
            p1 = Point(size, 0, 0)
            t0 = Point(0, size*complexity, 0)
            t1 = Point(0, -size*complexity, 0)
            return HermiteSpline(p0, p1, t0, t1)
        # ... handle other curve types ...
        else:
            return FlowerPetal(length=size, curve_factor=0.3*complexity) 