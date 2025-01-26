"""Design pattern recognition and analysis module for CAD models."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np  # type: ignore
from .base import GeometricEntity, Point
from .nurbs import NURBSCurve, NURBSSurface

@dataclass
class PatternFeature:
    """Represents a recognized pattern feature in the design."""
    pattern_type: str  # e.g., "linear_array", "circular_array", "symmetry"
    base_feature: GeometricEntity
    instances: List[GeometricEntity]
    parameters: Dict[str, Any]
    confidence: float  # Recognition confidence score

@dataclass
class DesignPattern:
    """Represents a higher-level design pattern."""
    name: str
    description: str
    features: List[PatternFeature]
    relationships: List[Dict[str, Any]]
    manufacturing_notes: Optional[Dict[str, Any]] = None
    reuse_suggestions: Optional[List[Dict[str, Any]]] = None

class PatternRecognizer:
    """Core class for recognizing and analyzing design patterns."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.pattern_library = self._initialize_pattern_library()
        
    def _initialize_pattern_library(self) -> Dict[str, Any]:
        """Initialize the library of known patterns and their characteristics."""
        return {
            "linear_array": {
                "detector": self._detect_linear_array,
                "validator": self._validate_linear_array,
                "parameters": ["spacing", "count", "direction"]
            },
            "circular_array": {
                "detector": self._detect_circular_array,
                "validator": self._validate_circular_array,
                "parameters": ["radius", "count", "angle"]
            },
            "symmetry": {
                "detector": self._detect_symmetry,
                "validator": self._validate_symmetry,
                "parameters": ["plane", "elements"]
            },
            "repetitive_feature": {
                "detector": self._detect_repetitive_feature,
                "validator": self._validate_repetitive_feature,
                "parameters": ["feature_type", "instances", "variations"]
            }
        }

    def analyze_geometry(self, geometry: GeometricEntity) -> List[DesignPattern]:
        """Analyze geometry to identify design patterns."""
        patterns = []
        
        # Decompose geometry into basic features
        features = self._decompose_geometry(geometry)
        
        # Analyze each pattern type
        for pattern_type, pattern_info in self.pattern_library.items():
            detector = pattern_info["detector"]
            validator = pattern_info["validator"]
            
            # Detect potential patterns
            potential_patterns = detector(features)
            
            # Validate and refine patterns
            for pattern in potential_patterns:
                if validator(pattern):
                    patterns.append(self._create_design_pattern(pattern))
        
        return patterns

    def _decompose_geometry(self, geometry: GeometricEntity) -> List[GeometricEntity]:
        """Decompose complex geometry into basic features."""
        features = []
        
        if hasattr(geometry, "sub_entities"):
            features.extend(geometry.sub_entities)
        else:
            features.append(geometry)
            
        return features

    def _detect_linear_array(self, features: List[GeometricEntity]) -> List[PatternFeature]:
        """Detect linear array patterns in features."""
        patterns: List[PatternFeature] = []
        for i, base_feature in enumerate(features):
            similar_features = self._find_similar_features(base_feature, features[i+1:])
            if len(similar_features) >= 2:  # Minimum 3 instances for a pattern
                spacing = self._calculate_linear_spacing(base_feature, similar_features)
                if spacing is not None:
                    patterns.append(PatternFeature(
                        pattern_type="linear_array",
                        base_feature=base_feature,
                        instances=similar_features,
                        parameters={"spacing": spacing},
                        confidence=0.9
                    ))
        return patterns

    def _detect_circular_array(self, features: List[GeometricEntity]) -> List[PatternFeature]:
        """Detect circular array patterns in features."""
        patterns: List[PatternFeature] = []
        for i, base_feature in enumerate(features):
            similar_features = self._find_similar_features(base_feature, features[i+1:])
            if len(similar_features) >= 2:
                center, radius = self._calculate_circular_parameters(base_feature, similar_features)
                if center is not None and radius is not None:
                    patterns.append(PatternFeature(
                        pattern_type="circular_array",
                        base_feature=base_feature,
                        instances=similar_features,
                        parameters={"center": center, "radius": radius},
                        confidence=0.85
                    ))
        return patterns

    def _detect_symmetry(self, features: List[GeometricEntity]) -> List[PatternFeature]:
        """Detect symmetry patterns in features."""
        patterns: List[PatternFeature] = []
        # Implementation for symmetry detection
        return patterns

    def _detect_repetitive_feature(self, features: List[GeometricEntity]) -> List[PatternFeature]:
        """Detect repetitive features that may not follow a strict pattern."""
        patterns: List[PatternFeature] = []
        # Implementation for repetitive feature detection
        return patterns

    def _find_similar_features(self, base: GeometricEntity, 
                             candidates: List[GeometricEntity]) -> List[GeometricEntity]:
        """Find features similar to the base feature."""
        similar = []
        for candidate in candidates:
            if self._compare_features(base, candidate) > 0.9:  # Similarity threshold
                similar.append(candidate)
        return similar

    def _compare_features(self, feature1: GeometricEntity, 
                         feature2: GeometricEntity) -> float:
        """Compare two features and return similarity score."""
        # Basic implementation - should be enhanced based on specific requirements
        if type(feature1) != type(feature2):
            return 0.0
            
        # Compare basic properties
        similarity = 1.0
        
        # Compare dimensions if available
        if hasattr(feature1, "dimensions") and hasattr(feature2, "dimensions"):
            dim_similarity = self._compare_dimensions(
                feature1.dimensions, feature2.dimensions)
            similarity *= dim_similarity
            
        return similarity

    def _compare_dimensions(self, dim1: Dict[str, float], dim2: Dict[str, float]) -> float:
        """Compare dimensions of two features and return similarity score."""
        if not dim1 or not dim2:
            return 0.0
            
        # Get common dimension keys
        common_dims = set(dim1.keys()) & set(dim2.keys())
        if not common_dims:
            return 0.0
            
        # Compare each dimension
        similarities = []
        for dim in common_dims:
            val1, val2 = dim1[dim], dim2[dim]
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            elif val1 == 0 or val2 == 0:
                similarities.append(0.0)
            else:
                ratio = min(val1, val2) / max(val1, val2)
                similarities.append(ratio)
                
        return sum(similarities) / len(similarities)

    def _calculate_linear_spacing(self, base: GeometricEntity, 
                                instances: List[GeometricEntity]) -> Optional[float]:
        """Calculate spacing for linear array pattern."""
        if not instances:
            return None
            
        spacings = []
        base_center = self._get_center(base)
        
        for instance in instances:
            instance_center = self._get_center(instance)
            spacing = np.linalg.norm(instance_center - base_center)
            spacings.append(spacing)
            
        # Check if spacings are consistent
        if self._are_spacings_consistent(spacings):
            return np.mean(spacings)
        return None

    def _calculate_circular_parameters(self, base: GeometricEntity,
                                    instances: List[GeometricEntity]) -> Tuple[Optional[Point], Optional[float]]:
        """Calculate center and radius for circular array pattern."""
        # Implementation for circular parameter calculation
        return None, None

    def _get_center(self, entity: GeometricEntity) -> np.ndarray:
        """Get the center point of a geometric entity."""
        if hasattr(entity, "center"):
            return np.array(entity.center)
        elif hasattr(entity, "bounds"):
            min_bound, max_bound = entity.bounds
            return (np.array(min_bound) + np.array(max_bound)) / 2
        return np.zeros(3)

    def _are_spacings_consistent(self, spacings: List[float], tolerance: float = 0.01) -> bool:
        """Check if spacings are consistent within tolerance."""
        if not spacings:
            return False
        mean_spacing = np.mean(spacings)
        return all(abs(s - mean_spacing) <= tolerance * mean_spacing for s in spacings)

    def _calculate_distance(self, entity1: GeometricEntity, entity2: GeometricEntity) -> float:
        """Calculate the distance between two geometric entities."""
        center1 = self._get_center(entity1)
        center2 = self._get_center(entity2)
        return float(np.linalg.norm(center2 - center1))

    def _create_design_pattern(self, pattern_feature: PatternFeature) -> DesignPattern:
        """Create a DesignPattern from a PatternFeature with additional analysis."""
        return DesignPattern(
            name=f"{pattern_feature.pattern_type}_pattern",
            description=self._generate_pattern_description(pattern_feature),
            features=[pattern_feature],
            relationships=self._analyze_pattern_relationships(pattern_feature),
            manufacturing_notes=self._generate_manufacturing_notes(pattern_feature),
            reuse_suggestions=self._generate_reuse_suggestions(pattern_feature)
        )

    def _generate_pattern_description(self, pattern: PatternFeature) -> str:
        """Generate human-readable description of the pattern."""
        if self.llm_client:
            # Use LLM for rich description
            return self.llm_client.generate_pattern_description(pattern)
        
        # Fallback basic description
        return f"{pattern.pattern_type} with {len(pattern.instances)} instances"

    def _analyze_pattern_relationships(self, pattern: PatternFeature) -> List[Dict[str, Any]]:
        """Analyze relationships between pattern elements."""
        relationships: List[Dict[str, Any]] = []
        
        # Analyze spatial relationships
        if pattern.pattern_type == "linear_array":
            relationships.append({
                "type": "spacing",
                "value": pattern.parameters.get("spacing"),
                "unit": "mm"
            })
        elif pattern.pattern_type == "circular_array":
            relationships.append({
                "type": "radius",
                "value": pattern.parameters.get("radius"),
                "unit": "mm"
            })
            
        # Analyze feature relationships
        for i, instance in enumerate(pattern.instances):
            relationships.append({
                "type": "instance",
                "index": i,
                "base_distance": self._calculate_distance(pattern.base_feature, instance)
            })
            
        return relationships

    def _generate_manufacturing_notes(self, pattern: PatternFeature) -> Dict[str, Any]:
        """Generate manufacturing considerations for the pattern."""
        if self.llm_client:
            return self.llm_client.generate_manufacturing_notes(pattern)
        return {"note": "Standard manufacturing process recommended"}

    def _generate_reuse_suggestions(self, pattern: PatternFeature) -> List[Dict[str, Any]]:
        """Generate suggestions for pattern reuse in other contexts."""
        if self.llm_client:
            return self.llm_client.generate_reuse_suggestions(pattern)
        return [{"suggestion": "Pattern can be reused in similar contexts"}]

    def _validate_linear_array(self, pattern: PatternFeature) -> bool:
        """Validate detected linear array pattern."""
        if len(pattern.instances) < 2:
            return False
        # Add more validation logic
        return True

    def _validate_circular_array(self, pattern: PatternFeature) -> bool:
        """Validate detected circular array pattern."""
        if len(pattern.instances) < 3:
            return False
        # Add more validation logic
        return True

    def _validate_symmetry(self, pattern: PatternFeature) -> bool:
        """Validate detected symmetry pattern."""
        # Implementation for symmetry validation
        return True

    def _validate_repetitive_feature(self, pattern: PatternFeature) -> bool:
        """Validate detected repetitive feature pattern."""
        # Implementation for repetitive feature validation
        return True 
