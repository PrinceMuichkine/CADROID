"""
Intelligent CAD system that bridges LLM output with CAD parameters.
"""

from typing import Dict, Any, List, Optional, Union
from .base import BaseGeometry
from .factory import OrganicShapeFactory
from .llm_client import LLMClient
from ..utility.logger import CLGLogger

# Initialize logger
logger = CLGLogger("intelligent_cad").configure_logger()

class IntelligentCAD:
    """Main class for intelligent CAD operations."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize IntelligentCAD system.
        
        Args:
            llm_client: LLM client for text analysis (optional)
        """
        self.llm_client = llm_client or LLMClient()
        self.shape_factory = OrganicShapeFactory()
        self.context: Dict[str, Any] = {
            'design_history': [],
            'patterns': {},
            'optimizations': {}
        }
    
    def _map_llm_to_parameters(self, llm_output: Dict[str, Any]) -> Dict[str, Any]:
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
        
        # Map basic properties
        if properties := llm_output.get('properties', {}):
            for category, values in properties.items():
                for prop, intensity in values.items():
                    if mapping := param_map.get(category, {}).get(prop):
                        param_name, converter = mapping
                        if callable(converter):
                            cad_params[param_name] = converter(intensity)
                        else:
                            cad_params[param_name] = converter
        
        # Add manufacturing constraints
        if manufacturing := llm_output.get('manufacturing', {}):
            if constraints := manufacturing.get('constraints', []):
                for constraint in constraints:
                    if 'type' in constraint and 'value' in constraint:
                        cad_params[constraint['type']] = constraint['value']
        
        # Add feature parameters
        if features := llm_output.get('features', []):
            for feature in features:
                if 'type' in feature and 'dimensions' in feature:
                    for dim_name, value in feature['dimensions'].items():
                        if isinstance(value, (int, float)):
                            cad_params[f"{feature['type']}_{dim_name}"] = value
        
        # Add pattern parameters
        if patterns := llm_output.get('patterns', []):
            for pattern in patterns:
                if 'type' in pattern:
                    pattern_type = pattern['type']
                    if 'count' in pattern:
                        cad_params[f"{pattern_type}_count"] = pattern['count']
                    if 'spacing' in pattern:
                        cad_params[f"{pattern_type}_spacing"] = pattern['spacing']
        
        return cad_params
    
    def process_design_request(self, text_input: str) -> Dict[str, Any]:
        """Process a design request from text input."""
        try:
            # Get LLM analysis
            llm_response = self.llm_client.analyze_request(text_input)
            
            # Update context with patterns and optimizations
            if patterns := llm_response.get('patterns', []):
                for pattern in patterns:
                    pattern_type = pattern.get('type')
                    if pattern_type:
                        self.context['patterns'][pattern_type] = pattern
            
            if optimizations := llm_response.get('optimizations', []):
                for opt in optimizations:
                    opt_type = opt.get('type')
                    if opt_type:
                        self.context['optimizations'][opt_type] = opt
            
            # Map to CAD parameters
            cad_params = self._map_llm_to_parameters(llm_response)
            
            # Generate geometry
            geometry = self.shape_factory.create_from_params(cad_params)
            
            # Apply patterns from context
            geometry = self._apply_patterns(geometry)
            
            # Validate manufacturability
            validation_result = self.validate_design(geometry)
            
            if not validation_result['valid']:
                # Attempt to fix issues
                geometry = self.optimize_for_manufacturing(geometry, validation_result['issues'])
                validation_result = self.validate_design(geometry)
            
            # Store in design history
            self.context['design_history'].append({
                'input': text_input,
                'parameters': cad_params,
                'validation': validation_result
            })
            
            return {
                'status': 'success',
                'geometry': geometry,
                'parameters': cad_params,
                'validation': validation_result,
                'patterns': llm_response.get('patterns', []),
                'optimizations': llm_response.get('optimizations', []),
                'references': llm_response.get('references', [])
            }
            
        except Exception as e:
            logger.error(f"Error processing design request: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _apply_patterns(self, geometry: BaseGeometry) -> BaseGeometry:
        """Apply design patterns to geometry."""
        for pattern_type, pattern in self.context['patterns'].items():
            if pattern_type == 'linear_array':
                geometry = self._apply_linear_array(geometry, pattern)
            elif pattern_type == 'circular_array':
                geometry = self._apply_circular_array(geometry, pattern)
            elif pattern_type == 'mirror':
                geometry = self._apply_mirror(geometry, pattern)
        return geometry
    
    def _apply_linear_array(self, geometry: BaseGeometry, pattern: Dict[str, Any]) -> BaseGeometry:
        """Apply linear array pattern."""
        # Implementation will depend on specific geometry system
        logger.info(f"Applying linear array pattern: {pattern}")
        return geometry
    
    def _apply_circular_array(self, geometry: BaseGeometry, pattern: Dict[str, Any]) -> BaseGeometry:
        """Apply circular array pattern."""
        # Implementation will depend on specific geometry system
        logger.info(f"Applying circular array pattern: {pattern}")
        return geometry
    
    def _apply_mirror(self, geometry: BaseGeometry, pattern: Dict[str, Any]) -> BaseGeometry:
        """Apply mirror pattern."""
        # Implementation will depend on specific geometry system
        logger.info(f"Applying mirror pattern: {pattern}")
        return geometry
    
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
        
        # Apply relevant optimizations from context
        for opt_type, optimization in self.context['optimizations'].items():
            if opt_type == 'material_reduction' and any('wall thickness' in issue.lower() for issue in issues):
                optimized = optimized.thicken_walls(min_thickness=0.1)
            elif opt_type == 'support_reduction' and any('overhang' in issue.lower() for issue in issues):
                optimized = optimized.reduce_overhangs(max_angle=45)
            elif opt_type == 'strength_improvement' and any('stress points' in issue.lower() for issue in issues):
                optimized = optimized.reinforce_weak_points()
        
        return optimized

    async def analyze_description(self, text: str) -> Dict[str, Any]:
        """Analyze a text description to extract CAD parameters."""
        try:
            # Get LLM analysis
            llm_response = await self.llm_client.analyze_request(text)
            
            if llm_response["status"] != "success":
                return llm_response
            
            # Generate geometry
            geometry = self.shape_factory.create_from_params(llm_response["parameters"])
            
            # Apply patterns from context
            geometry = await self._apply_patterns(geometry)
            
            # Validate manufacturability
            validation_result = await self.validate_design(geometry)
            
            if not validation_result['valid']:
                # Attempt to fix issues
                geometry = await self.optimize_for_manufacturing(geometry, validation_result['issues'])
                validation_result = await self.validate_design(geometry)
            
            return {
                "status": "success",
                "geometry": geometry,
                "parameters": llm_response["parameters"],
                "validation": validation_result
            }
            
        except Exception as e:
            logger.error(f"Error analyzing description: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def _apply_patterns(self, geometry: BaseGeometry) -> BaseGeometry:
        """Apply design patterns to geometry."""
        for pattern_type, pattern in self.context['patterns'].items():
            if pattern_type == 'linear_array':
                geometry = await self._apply_linear_array(geometry, pattern)
            elif pattern_type == 'circular_array':
                geometry = await self._apply_circular_array(geometry, pattern)
            elif pattern_type == 'mirror':
                geometry = await self._apply_mirror(geometry, pattern)
        return geometry
    
    async def _apply_linear_array(self, geometry: BaseGeometry, pattern: Dict[str, Any]) -> BaseGeometry:
        """Apply linear array pattern."""
        logger.info(f"Applying linear array pattern: {pattern}")
        return geometry
    
    async def _apply_circular_array(self, geometry: BaseGeometry, pattern: Dict[str, Any]) -> BaseGeometry:
        """Apply circular array pattern."""
        logger.info(f"Applying circular array pattern: {pattern}")
        return geometry
    
    async def _apply_mirror(self, geometry: BaseGeometry, pattern: Dict[str, Any]) -> BaseGeometry:
        """Apply mirror pattern."""
        logger.info(f"Applying mirror pattern: {pattern}")
        return geometry
    
    async def validate_design(self, geometry: BaseGeometry) -> Dict[str, Any]:
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
    
    async def optimize_for_manufacturing(self, geometry: BaseGeometry, issues: List[str]) -> BaseGeometry:
        """Optimize geometry to resolve manufacturing issues."""
        optimized = geometry
        
        # Apply relevant optimizations from context
        for opt_type, optimization in self.context['optimizations'].items():
            if opt_type == 'material_reduction' and any('wall thickness' in issue.lower() for issue in issues):
                optimized = optimized.thicken_walls(min_thickness=0.1)
            elif opt_type == 'support_reduction' and any('overhang' in issue.lower() for issue in issues):
                optimized = optimized.reduce_overhangs(max_angle=45)
            elif opt_type == 'strength_improvement' and any('stress points' in issue.lower() for issue in issues):
                optimized = optimized.reinforce_weak_points()
        
        return optimized 