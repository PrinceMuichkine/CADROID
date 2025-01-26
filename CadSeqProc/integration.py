"""
Integration module to connect enhanced geometry system with existing CAD model.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from .geometry.nurbs import NURBSCurve, NURBSSurface
from .geometry.organic import OrganicSurface
from .utility.shape_factory import OrganicShapeFactory
from .sequence.transformation.deform import TwistDeformation, BendDeformation, TaperDeformation

class GeometryAdapter:
    """Adapter to convert between enhanced geometry and CAD model formats."""
    
    @staticmethod
    def to_cad_sequence(
        entities: List[Union[NURBSCurve, NURBSSurface, OrganicSurface]]
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
                    'points': [[p[0], p[1], p[2]] for p in points],
                    'closed': False
                })
            elif isinstance(entity, NURBSSurface):
                points_2d = entity.sample_points(20, 20)
                sequence['operations'].append({
                    'type': 'surface',
                    'points': [[[p[0], p[1], p[2]] for p in row] for row in points_2d],
                    'closed_u': True,
                    'closed_v': True
                })
            elif isinstance(entity, OrganicSurface):
                for surface in entity.control_surfaces:
                    points_2d = surface.sample_points(20, 20)
                    sequence['operations'].append({
                        'type': 'surface',
                        'points': [[[p[0], p[1], p[2]] for p in row] for row in points_2d],
                        'closed_u': True,
                        'closed_v': True
                    })
        
        return sequence
    
    @staticmethod
    def from_cad_sequence(sequence: Dict[str, Any]) -> List[Union[NURBSCurve, NURBSSurface]]:
        """Convert CAD sequence back to geometric entities."""
        entities = []
        
        for op in sequence.get('operations', []):
            if op['type'] == 'curve':
                points = [(p[0], p[1], p[2]) for p in op['points']]
                entities.append(NURBSCurve.from_points(points))
            elif op['type'] == 'surface':
                points_2d = [[(p[0], p[1], p[2]) for p in row] for row in op['points']]
                entities.append(NURBSSurface.from_points(points_2d))
        
        return entities

class ShapeGenerator:
    """High-level interface for generating shapes from text descriptions."""
    
    def __init__(self):
        self.factory = OrganicShapeFactory()
    
    def parse_flower_description(self, text: str) -> Dict[str, Any]:
        """Parse text description for flower parameters."""
        params = {
            'num_petals': 5,
            'petal_length': 1.0,
            'petal_width': 0.3,
            'center_radius': 0.2
        }
        
        # Extract number of petals
        if 'many petals' in text.lower():
            params['num_petals'] = 8
        elif 'few petals' in text.lower():
            params['num_petals'] = 3
        
        # Adjust size
        if any(word in text.lower() for word in ['large', 'big']):
            params['petal_length'] *= 1.5
            params['petal_width'] *= 1.5
            params['center_radius'] *= 1.5
        elif any(word in text.lower() for word in ['small', 'tiny']):
            params['petal_length'] *= 0.7
            params['petal_width'] *= 0.7
            params['center_radius'] *= 0.7
        
        return params
    
    def create_from_text(self, text: str) -> Dict[str, Any]:
        """Generate shapes based on text description."""
        try:
            shapes = []
            
            if any(word in text.lower() for word in ['flower', 'petal', 'bloom']):
                params = self.parse_flower_description(text)
                shapes.extend(self.factory.create_flower(**params))
            elif any(word in text.lower() for word in ['leaf', 'foliage']):
                shapes.append(self.factory.create_leaf())
            elif any(word in text.lower() for word in ['tree', 'plant']):
                shapes.extend(self.factory.create_tree())
            elif any(word in text.lower() for word in ['vine', 'creeper']):
                # Create a simple curved path for the vine
                control_points = [
                    (0, 0, 0),
                    (0.5, 0.5, 0.2),
                    (1.0, 0, 0.5),
                    (1.5, -0.5, 0.3),
                    (2.0, 0, 0)
                ]
                shapes.extend(self.factory.create_vine(control_points))
            else:
                raise ValueError("Unsupported shape type in description")
            
            return {
                'status': 'success',
                'shapes': shapes
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

class ModelIntegration:
    """Integration with the main CAD model."""
    
    def __init__(self):
        self.adapter = GeometryAdapter()
        self.generator = ShapeGenerator()
    
    def process_text_input(self, text: str) -> Dict[str, Any]:
        """Process text input to generate CAD model."""
        try:
            # Generate shapes from text
            result = self.generator.create_from_text(text)
            
            if result['status'] == 'error':
                return {
                    'status': 'error',
                    'message': result['message']
                }
            
            # Convert shapes to CAD sequence
            sequence = self.adapter.to_cad_sequence(result['shapes'])
            
            return {
                'status': 'success',
                'cad_sequence': sequence,
                'metadata': {
                    'input_text': text,
                    'num_shapes': len(result['shapes'])
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def validate_sequence(self, sequence: Dict[str, Any]) -> bool:
        """Validate CAD sequence before processing."""
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