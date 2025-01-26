"""
Integration module to connect enhanced geometry system with existing CAD model.
Provides conversion and adaptation layers.
"""

import numpy as np
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from .base import Point, GeometricEntity
from .nurbs import NURBSCurve, NURBSSurface
from .organic import OrganicSurface
from .factory import OrganicShapeFactory
from .parametric import (
    ParametricCurve, FlowerPetal, PatternGenerator,
    RoseCurve, BezierCurve, EpicycloidCurve,
    PatternAnalyzer, CombinedPattern
)

class TextParser:
    """Advanced text parsing for shape parameters."""
    
    @staticmethod
    def extract_number(text: str, default: float) -> float:
        """Extract first number from text or return default."""
        if match := re.search(r'(\d*\.?\d+)', text):
            return float(match.group(1))
        return default
    
    @staticmethod
    def extract_size_modifier(text: str) -> float:
        """Extract size modifier from descriptive text."""
        size_modifiers = {
            'tiny': 0.5,
            'small': 0.7,
            'medium': 1.0,
            'large': 1.5,
            'huge': 2.0,
            'giant': 2.5
        }
        
        for word, modifier in size_modifiers.items():
            if word in text.lower():
                return modifier
        return 1.0
    
    @staticmethod
    def extract_pattern_type(text: str) -> str:
        """Determine pattern type from description using AI analysis."""
        # Use AI-driven pattern analysis
        requirements = PatternAnalyzer.analyze_shape_requirements(text)
        return PatternAnalyzer.get_optimal_pattern(requirements)
    
    @staticmethod
    def extract_curve_type(text: str) -> str:
        """Determine the best curve type for the shape."""
        if any(word in text.lower() for word in ['rose', 'flower', 'petal']):
            return 'rose'
        elif any(word in text.lower() for word in ['spiral', 'coil']):
            return 'spiral'
        elif any(word in text.lower() for word in ['complex', 'ornate']):
            return 'epicycloid'
        elif any(word in text.lower() for word in ['smooth', 'curved']):
            return 'bezier'
        return 'default'
    
    @staticmethod
    def extract_shape_complexity(text: str) -> Dict[str, float]:
        """Analyze shape complexity requirements."""
        complexity = {
            'detail_level': 1.0,  # Base detail level
            'symmetry': 1.0,      # Perfect symmetry
            'variation': 0.0,     # No random variation
            'layers': 1           # Single layer
        }
        
        # Adjust detail level
        if any(word in text.lower() for word in ['detailed', 'complex', 'intricate']):
            complexity['detail_level'] = 1.5
        elif any(word in text.lower() for word in ['simple', 'basic']):
            complexity['detail_level'] = 0.7
        
        # Adjust symmetry
        if any(word in text.lower() for word in ['irregular', 'natural', 'organic']):
            complexity['symmetry'] = 0.8
        elif any(word in text.lower() for word in ['perfect', 'exact']):
            complexity['symmetry'] = 1.0
        
        # Adjust variation
        if any(word in text.lower() for word in ['random', 'varied', 'diverse']):
            complexity['variation'] = 0.3
        
        # Adjust layers
        if 'layered' in text.lower():
            complexity['layers'] = 2
        elif 'multi-layered' in text.lower():
            complexity['layers'] = 3
        
        return complexity
    
    @staticmethod
    def extract_center_type(text: str) -> str:
        """Determine center type from description."""
        if any(word in text.lower() for word in ['spiral', 'swirl']):
            return 'spiral'
        elif any(word in text.lower() for word in ['complex', 'intricate']):
            return 'lissajous'
        return 'circle'
    
    @staticmethod
    def extract_color_hints(text: str) -> List[str]:
        """Extract color-related words from text."""
        color_words = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'white', 'black']
        return [word for word in color_words if word in text.lower()]

class GeometryAdapter:
    """Adapter to convert between enhanced geometry and CAD model formats."""
    
    @staticmethod
    def to_cad_sequence(
        entities: List[Union[GeometricEntity, OrganicSurface]]
    ) -> Dict[str, Any]:
        """Convert geometric entities to CAD sequence format."""
        sequence = {
            'type': 'composite',
            'operations': []
        }
        
        for entity in entities:
            if isinstance(entity, NURBSCurve):
                points = entity.sample_points(20)
                sequence['operations'].append({
                    'type': 'curve',
                    'points': [[p.x, p.y, p.z] for p in points],
                    'closed': False
                })
            elif isinstance(entity, NURBSSurface):
                points_2d = entity.sample_points_grid(20, 20)
                sequence['operations'].append({
                    'type': 'surface',
                    'points': [[[p.x, p.y, p.z] for p in row] for row in points_2d],
                    'closed_u': True,
                    'closed_v': True
                })
            elif isinstance(entity, OrganicSurface):
                for surface in entity.control_surfaces:
                    points_2d = surface.sample_points_grid(20, 20)
                    sequence['operations'].append({
                        'type': 'surface',
                        'points': [[[p.x, p.y, p.z] for p in row] for row in points_2d],
                        'closed_u': True,
                        'closed_v': True
                    })
        
        return sequence
    
    @staticmethod
    def from_cad_sequence(sequence: Dict[str, Any]) -> List[GeometricEntity]:
        """Convert CAD sequence back to geometric entities."""
        entities = []
        
        for op in sequence.get('operations', []):
            if op['type'] == 'curve':
                points = [Point(*p) for p in op['points']]
                entities.append(NURBSCurve.from_points(points))
            elif op['type'] == 'surface':
                points_2d = [[Point(*p) for p in row] for row in op['points']]
                entities.append(NURBSSurface.from_points(points_2d))
        
        return entities

class ShapeGenerator:
    """High-level interface for generating shapes from text descriptions."""
    
    @staticmethod
    def parse_flower_description(text: str) -> Dict[str, Any]:
        """Parse text description for flower parameters."""
        parser = TextParser()
        size_modifier = parser.extract_size_modifier(text)
        pattern_type = parser.extract_pattern_type(text)
        curve_type = parser.extract_curve_type(text)
        complexity = parser.extract_shape_complexity(text)
        
        # Base parameters
        params = {
            'num_petals': 5,
            'petal_length': 1.0 * size_modifier,
            'petal_width': 0.3 * size_modifier,
            'center_radius': 0.2 * size_modifier,
            'petal_curve_factor': 0.3,
            'pattern_type': pattern_type,
            'center_type': 'spiral' if complexity['detail_level'] > 1.2 else 'circle',
            'complexity': complexity
        }
        
        # Extract number of petals
        if match := re.search(r'(\d+)\s*petals?', text.lower()):
            params['num_petals'] = int(match.group(1))
        
        # Adjust curve factor based on descriptive words
        if any(word in text.lower() for word in ['wavy', 'curly', 'curved']):
            params['petal_curve_factor'] *= 1.5
        elif any(word in text.lower() for word in ['straight', 'flat']):
            params['petal_curve_factor'] *= 0.5
        
        # Add randomness based on complexity
        if complexity['variation'] > 0:
            params['petal_length'] *= (1 + 0.2 * np.random.random() - 0.1)
            params['petal_width'] *= (1 + 0.2 * np.random.random() - 0.1)
            params['petal_curve_factor'] *= (1 + 0.3 * np.random.random() - 0.15)
        
        return params
    
    @staticmethod
    def parse_tree_description(text: str) -> Dict[str, Any]:
        """Parse text description for tree parameters."""
        parser = TextParser()
        size_modifier = parser.extract_size_modifier(text)
        
        params = {
            'trunk_height': 2.0 * size_modifier,
            'trunk_radius': 0.2 * size_modifier,
            'num_branches': 5,
            'leaf_size': 0.5 * size_modifier
        }
        
        # Adjust trunk height
        if 'tall' in text.lower():
            params['trunk_height'] *= 1.5
        elif 'short' in text.lower():
            params['trunk_height'] *= 0.7
        
        # Adjust trunk thickness
        if any(word in text.lower() for word in ['thick', 'wide', 'broad']):
            params['trunk_radius'] *= 1.5
        elif any(word in text.lower() for word in ['thin', 'narrow', 'slender']):
            params['trunk_radius'] *= 0.7
        
        # Adjust number of branches
        if 'many branches' in text.lower():
            params['num_branches'] = 8
        elif 'few branches' in text.lower():
            params['num_branches'] = 3
        
        # Adjust leaf size
        if 'large leaves' in text.lower():
            params['leaf_size'] *= 1.5
        elif 'small leaves' in text.lower():
            params['leaf_size'] *= 0.7
        
        return params
    
    @staticmethod
    def create_flower_from_text(text: str) -> List[OrganicSurface]:
        """Generate a flower based on text description."""
        params = ShapeGenerator.parse_flower_description(text)
        requirements = PatternAnalyzer.analyze_shape_requirements(text)
        
        # Create curves based on requirements
        curves = []
        for curve_type in requirements['curve_types']:
            curve = PatternAnalyzer.create_curve_from_type(
                curve_type,
                size=params['petal_length'],
                complexity=requirements['complexity']
            )
            curves.append(curve)
        
        # Use default curve if none specified
        if not curves:
            curves = [FlowerPetal(
                length=params['petal_length'],
                width=params['petal_width'],
                curve_factor=params['petal_curve_factor']
            )]
        
        # Generate patterns for each curve
        patterns = []
        for curve in curves:
            if requirements['pattern_type'] == 'fractal':
                pattern = PatternGenerator.fractal_pattern(
                    curve,
                    params['num_petals'],
                    scale_range=(0.3, 1.0)
                )
            elif requirements['pattern_type'] == 'radial_wave':
                pattern = PatternGenerator.radial_wave_pattern(
                    curve,
                    params['num_petals'],
                    radius=params['center_radius'] * 2
                )
            else:
                pattern = PatternGenerator.circular_pattern(
                    curve,
                    params['num_petals'],
                    radius=params['center_radius']
                )
            patterns.append(pattern)
        
        # Combine patterns if specified
        if requirements['combination_mode'] == 'blend':
            combined = CombinedPattern(patterns)
            final_pattern = combined.blend(0.5)
        else:
            # Use the most complex pattern
            pattern_complexity = [len(p) for p in patterns]
            final_pattern = patterns[pattern_complexity.index(max(pattern_complexity))]
        
        # Create surfaces
        surfaces = []
        for curve in final_pattern:
            control_points = []
            for t in np.linspace(0, 1, 10):
                points = curve.sample_points(20)
                thickness = 0.1 * (1 - t)
                offset = np.array([0, 0, thickness])
                control_points.append([Point(p.x, p.y, p.z + offset[2]) for p in points])
            surfaces.append(NURBSSurface.from_points(control_points))
        
        return [OrganicSurface([surface]) for surface in surfaces]
    
    @staticmethod
    def create_tree_from_text(text: str) -> List[OrganicSurface]:
        """Generate a tree based on text description."""
        params = ShapeGenerator.parse_tree_description(text)
        return OrganicShapeFactory.create_tree(**params)

class ModelIntegration:
    """Integration with the main CAD model."""
    
    def __init__(self):
        self.adapter = GeometryAdapter()
        self.generator = ShapeGenerator()
    
    def process_text_input(self, text: str) -> Dict[str, Any]:
        """
        Process text input to generate CAD model.
        
        Args:
            text: Text description of desired shape
            
        Returns:
            Dictionary containing:
            - cad_sequence: CAD sequence for the shape
            - metadata: Additional information about the generation
        """
        try:
            # Determine shape type from text
            if any(word in text.lower() for word in ['flower', 'petal', 'bloom', 'sunflower', 'daisy']):
                entities = self.generator.create_flower_from_text(text)
            elif any(word in text.lower() for word in ['tree', 'branch', 'leaf', 'plant']):
                entities = self.generator.create_tree_from_text(text)
            else:
                raise ValueError("Unsupported shape type in text description")
            
            cad_sequence = self.adapter.to_cad_sequence(entities)
            
            # Add metadata about the generation
            metadata = {
                'input_text': text,
                'generation_status': 'success',
                'shape_type': cad_sequence['type'],
                'parameters': {
                    'color_hints': TextParser.extract_color_hints(text),
                    'pattern_type': TextParser.extract_pattern_type(text),
                    'size_modifier': TextParser.extract_size_modifier(text)
                }
            }
            
            return {
                'cad_sequence': cad_sequence,
                'metadata': metadata
            }
        except Exception as e:
            return {
                'cad_sequence': None,
                'metadata': {
                    'input_text': text,
                    'generation_status': 'error',
                    'error_message': str(e)
                }
            }
    
    def validate_sequence(self, sequence: Dict[str, Any]) -> bool:
        """
        Validate CAD sequence before processing.
        
        Args:
            sequence: CAD sequence to validate
            
        Returns:
            True if sequence is valid
        """
        if not isinstance(sequence, dict):
            return False
        
        if 'type' not in sequence or sequence['type'] != 'composite':
            return False
        
        if 'operations' not in sequence or not isinstance(sequence['operations'], list):
            return False
        
        for op in sequence['operations']:
            if 'type' not in op:
                return False
            
            if op['type'] not in ['curve', 'surface']:
                return False
            
            if op['type'] == 'curve':
                if 'points' not in op or not isinstance(op['points'], list):
                    return False
            elif op['type'] == 'surface':
                if 'points' not in op or not isinstance(op['points'], list):
                    return False
                if not all(isinstance(row, list) for row in op['points']):
                    return False
        
        return True 