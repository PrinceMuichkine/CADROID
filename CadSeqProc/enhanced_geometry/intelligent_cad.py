"""
Intelligent CAD system that bridges LLM output with CAD parameters.
"""

from typing import Dict, Any, List, Optional, Union
from .base import BaseGeometry
from .factory import OrganicShapeFactory
from .llm_client import LLMClient
from ..utility.logger import setup_logger

logger = setup_logger(__name__)

class IntelligentCAD:
    """Main class for intelligent CAD operations."""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.shape_factory = OrganicShapeFactory()
        self.context = {}
        
    def _map_llm_to_parameters(self, llm_output: dict) -> dict:
        """Convert LLM output to CAD parameters"""
        param_map = {
            'density': {
                'petals': ('num_petals', lambda x: 34 + int(21 * x)),
                'holes': ('hole_count', lambda x: 1 + int(4 * x))
            },
            'size': {
                'small': ('scale_factor', 0.5),
                'large': ('scale_factor', 1.5)
            },
            'complexity': {
                'simple': ('detail_level', 0.3),
                'complex': ('detail_level', 0.8)
            }
        }
        
        cad_params = {}
        for category, values in llm_output.get('properties', {}).items():
            for prop, intensity in values.items():
                if mapping := param_map.get(category, {}).get(prop):
                    param_name, converter = mapping
                    if callable(converter):
                        cad_params[param_name] = converter(intensity)
                    else:
                        cad_params[param_name] = converter
        
        # Add geometric constraints
        cad_params.setdefault('min_wall_thickness', 0.1)
        cad_params.setdefault('max_overhang_angle', 45)
        
        return cad_params
    
    def process_design_request(self, text_input: str) -> Dict[str, Any]:
        """Process a design request from text input."""
        try:
            # Get LLM analysis
            llm_response = self.llm_client.analyze_request(text_input)
            
            # Map to CAD parameters
            cad_params = self._map_llm_to_parameters(llm_response)
            
            # Generate geometry
            geometry = self.shape_factory.create_from_params(cad_params)
            
            # Validate manufacturability
            validation_result = self.validate_design(geometry)
            
            if not validation_result['valid']:
                # Attempt to fix issues
                geometry = self.optimize_for_manufacturing(geometry, validation_result['issues'])
                validation_result = self.validate_design(geometry)
            
            return {
                'status': 'success',
                'geometry': geometry,
                'parameters': cad_params,
                'validation': validation_result
            }
            
        except Exception as e:
            logger.error(f"Error processing design request: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def validate_design(self, geometry: BaseGeometry) -> Dict[str, Any]:
        """Validate design for manufacturability."""
        issues = []
        
        # Check minimum wall thickness
        min_thickness = geometry.analyze_thickness()
        if min_thickness < 0.1:
            issues.append(f"Minimum wall thickness {min_thickness:.2f}mm is below 0.1mm")
        
        # Check overhangs
        max_overhang = geometry.analyze_overhangs()
        if max_overhang > 45:
            issues.append(f"Maximum overhang angle {max_overhang:.1f}° exceeds 45°")
        
        # Check structural integrity
        stress_points = geometry.analyze_stress_points()
        if stress_points:
            issues.append(f"Found {len(stress_points)} potential stress points")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def optimize_for_manufacturing(self, geometry: BaseGeometry, issues: List[str]) -> BaseGeometry:
        """Optimize geometry to resolve manufacturing issues."""
        optimized = geometry
        
        for issue in issues:
            if 'wall thickness' in issue.lower():
                optimized = optimized.thicken_walls(min_thickness=0.1)
            elif 'overhang' in issue.lower():
                optimized = optimized.reduce_overhangs(max_angle=45)
            elif 'stress points' in issue.lower():
                optimized = optimized.reinforce_weak_points()
        
        return optimized 