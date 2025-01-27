"""Manufacturing analysis and optimization module."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import numpy as np  # type: ignore
from ..enhanced_geometry.base import GeometricEntity, Point
from ..enhanced_geometry.pattern_recognition import DesignPattern

@dataclass
class ManufacturingConstraint:
    """Represents a manufacturing constraint."""
    constraint_type: str  # e.g., "min_wall_thickness", "max_overhang"
    value: float
    unit: str
    description: str
    severity: str  # "error", "warning", "info"

@dataclass
class ManufacturingProcess:
    """Represents a manufacturing process with its capabilities."""
    name: str
    type: str  # "additive", "subtractive", "forming"
    materials: List[str]
    min_feature_size: float
    max_feature_size: float
    tolerances: Dict[str, float]
    surface_finish: float
    cost_factors: Dict[str, float]
    constraints: List[ManufacturingConstraint]

@dataclass
class MaterialProperties:
    """Material properties relevant for manufacturing."""
    name: str
    type: str  # "metal", "plastic", "composite"
    density: float
    yield_strength: float
    tensile_strength: float
    thermal_properties: Dict[str, float]
    cost_per_unit: float
    unit: str

class ManufacturingAnalyzer:
    """Analyzes and validates manufacturing aspects of CAD models."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.processes = self._initialize_processes()
        self.materials = self._initialize_materials()
        
    def _initialize_processes(self) -> Dict[str, ManufacturingProcess]:
        """Initialize supported manufacturing processes."""
        return {
            "3d_printing_fdm": ManufacturingProcess(
                name="FDM 3D Printing",
                type="additive",
                materials=["PLA", "ABS", "PETG"],
                min_feature_size=0.4,
                max_feature_size=1000.0,
                tolerances={"xy": 0.2, "z": 0.1},
                surface_finish=0.2,
                cost_factors={
                    "material": 1.0,
                    "time": 0.5,
                    "setup": 0.1
                },
                constraints=[
                    ManufacturingConstraint(
                        constraint_type="min_wall_thickness",
                        value=0.8,
                        unit="mm",
                        description="Minimum wall thickness for structural integrity",
                        severity="error"
                    ),
                    ManufacturingConstraint(
                        constraint_type="max_overhang",
                        value=45.0,
                        unit="degrees",
                        description="Maximum overhang angle without supports",
                        severity="warning"
                    )
                ]
            ),
            "cnc_milling": ManufacturingProcess(
                name="CNC Milling",
                type="subtractive",
                materials=["aluminum", "steel", "brass"],
                min_feature_size=0.1,
                max_feature_size=2000.0,
                tolerances={"xy": 0.05, "z": 0.05},
                surface_finish=0.05,
                cost_factors={
                    "material": 2.0,
                    "time": 1.0,
                    "setup": 0.5
                },
                constraints=[
                    ManufacturingConstraint(
                        constraint_type="min_internal_radius",
                        value=1.0,
                        unit="mm",
                        description="Minimum internal radius for tool access",
                        severity="error"
                    ),
                    ManufacturingConstraint(
                        constraint_type="max_depth_to_width",
                        value=4.0,
                        unit="ratio",
                        description="Maximum depth to width ratio for stability",
                        severity="warning"
                    )
                ]
            )
        }
        
    def _initialize_materials(self) -> Dict[str, MaterialProperties]:
        """Initialize supported materials."""
        return {
            "PLA": MaterialProperties(
                name="PLA",
                type="plastic",
                density=1.24,
                yield_strength=50.0,
                tensile_strength=60.0,
                thermal_properties={
                    "glass_transition": 60,
                    "melting_point": 180
                },
                cost_per_unit=25.0,
                unit="kg"
            ),
            "aluminum_6061": MaterialProperties(
                name="Aluminum 6061",
                type="metal",
                density=2.7,
                yield_strength=276.0,
                tensile_strength=310.0,
                thermal_properties={
                    "thermal_conductivity": 167,
                    "melting_point": 660
                },
                cost_per_unit=5.0,
                unit="kg"
            )
        }

    def analyze_manufacturability(self, geometry: GeometricEntity, 
                                process_name: str) -> Dict[str, Any]:
        """Analyze manufacturability of a design for a specific process."""
        process = self.processes[process_name]
        
        # Analyze basic geometric constraints
        constraint_violations = self._check_constraints(geometry, process)
        
        # Analyze cost factors
        cost_analysis = self._analyze_cost(geometry, process)
        
        # Generate manufacturing recommendations
        recommendations = self._generate_recommendations(
            geometry, process, constraint_violations)
        
        return {
            "process": process.name,
            "manufacturability_score": self._calculate_score(constraint_violations),
            "constraint_violations": constraint_violations,
            "cost_analysis": cost_analysis,
            "recommendations": recommendations
        }

    def _check_constraints(self, geometry: GeometricEntity, 
                         process: ManufacturingProcess) -> List[Dict[str, Any]]:
        """Check manufacturing constraints for the geometry."""
        violations = []
        
        # Check each constraint
        for constraint in process.constraints:
            if violation := self._check_single_constraint(geometry, constraint):
                violations.append(violation)
        
        return violations

    def _check_single_constraint(self, geometry: GeometricEntity,
                               constraint: ManufacturingConstraint) -> Optional[Dict[str, Any]]:
        """Check a single manufacturing constraint."""
        if constraint.constraint_type == "min_wall_thickness":
            min_thickness = self._analyze_wall_thickness(geometry)
            if min_thickness < constraint.value:
                return {
                    "type": constraint.constraint_type,
                    "severity": constraint.severity,
                    "message": f"Wall thickness {min_thickness:.2f}mm is below minimum {constraint.value}mm",
                    "locations": self._find_thin_walls(geometry, constraint.value)
                }
        
        elif constraint.constraint_type == "max_overhang":
            max_angle = self._analyze_overhang_angles(geometry)
            if max_angle > constraint.value:
                return {
                    "type": constraint.constraint_type,
                    "severity": constraint.severity,
                    "message": f"Overhang angle {max_angle:.1f}° exceeds maximum {constraint.value}°",
                    "locations": self._find_steep_overhangs(geometry, constraint.value)
                }
        
        return None

    def _analyze_wall_thickness(self, geometry: GeometricEntity) -> float:
        """Analyze minimum wall thickness in the geometry."""
        # Implementation would use actual CAD kernel analysis
        return 1.0  # Placeholder

    def _analyze_overhang_angles(self, geometry: GeometricEntity) -> float:
        """Analyze maximum overhang angles in the geometry."""
        # Implementation would use actual CAD kernel analysis
        return 30.0  # Placeholder

    def _find_thin_walls(self, geometry: GeometricEntity, 
                        min_thickness: float) -> List[Dict[str, Any]]:
        """Find locations of walls thinner than minimum."""
        # Implementation would use actual CAD kernel analysis
        return []  # Placeholder

    def _find_steep_overhangs(self, geometry: GeometricEntity,
                             max_angle: float) -> List[Dict[str, Any]]:
        """Find locations of overhangs exceeding maximum angle."""
        # Implementation would use actual CAD kernel analysis
        return []  # Placeholder

    def _analyze_cost(self, geometry: GeometricEntity,
                     process: ManufacturingProcess) -> Dict[str, Any]:
        """Analyze manufacturing costs."""
        volume = self._calculate_volume(geometry)
        surface_area = self._calculate_surface_area(geometry)
        
        # Calculate material cost
        material_cost = volume * process.cost_factors["material"]
        
        # Estimate manufacturing time
        estimated_time = self._estimate_manufacturing_time(
            geometry, process, volume, surface_area)
        
        # Calculate time-based costs
        time_cost = estimated_time * process.cost_factors["time"]
        
        # Calculate setup costs
        setup_cost = process.cost_factors["setup"]
        
        return {
            "material_cost": material_cost,
            "time_cost": time_cost,
            "setup_cost": setup_cost,
            "total_cost": material_cost + time_cost + setup_cost,
            "breakdown": {
                "material": material_cost,
                "time": time_cost,
                "setup": setup_cost
            },
            "estimated_time": estimated_time
        }

    def _calculate_volume(self, geometry: GeometricEntity) -> float:
        """Calculate volume of the geometry."""
        # Implementation would use actual CAD kernel calculation
        return 1000.0  # Placeholder

    def _calculate_surface_area(self, geometry: GeometricEntity) -> float:
        """Calculate surface area of the geometry."""
        # Implementation would use actual CAD kernel calculation
        return 100.0  # Placeholder

    def _estimate_manufacturing_time(self, geometry: GeometricEntity,
                                  process: ManufacturingProcess,
                                  volume: float,
                                  surface_area: float) -> float:
        """Estimate manufacturing time based on geometry and process."""
        if process.type == "additive":
            # Estimate 3D printing time
            return volume * 0.1  # Simplified estimate
        elif process.type == "subtractive":
            # Estimate machining time
            return surface_area * 0.2  # Simplified estimate
        return 0.0

    def _calculate_score(self, violations: List[Dict[str, Any]]) -> float:
        """Calculate overall manufacturability score."""
        base_score = 100.0
        
        for violation in violations:
            if violation["severity"] == "error":
                base_score -= 20.0
            elif violation["severity"] == "warning":
                base_score -= 5.0
                
        return max(0.0, base_score)

    def _generate_recommendations(self, geometry: GeometricEntity,
                               process: ManufacturingProcess,
                               violations: List[Dict[str, Any]]) -> List[str]:
        """Generate manufacturing recommendations."""
        recommendations = []
        
        if self.llm_client:
            # Use LLM for rich recommendations
            prompt = self._create_recommendation_prompt(geometry, process, violations)
            recommendations = self.llm_client.generate_recommendations(prompt)
        else:
            # Generate basic recommendations based on violations
            for violation in violations:
                if violation["severity"] == "error":
                    recommendations.append(
                        f"Fix {violation['type']}: {violation['message']}")
                else:
                    recommendations.append(
                        f"Consider addressing {violation['type']}: {violation['message']}")
        
        return recommendations

    def _create_recommendation_prompt(self, geometry: GeometricEntity,
                                   process: ManufacturingProcess,
                                   violations: List[Dict[str, Any]]) -> str:
        """Create prompt for LLM to generate recommendations."""
        return f"""
        Analyze this manufacturing scenario and provide recommendations:
        
        Process: {process.name}
        Process Type: {process.type}
        Materials: {', '.join(process.materials)}
        
        Violations:
        {json.dumps(violations, indent=2)}
        
        Provide specific recommendations to:
        1. Address each violation
        2. Optimize for manufacturing
        3. Reduce costs
        4. Improve quality
        """

    def suggest_material(self, geometry: GeometricEntity,
                       process_name: str,
                       requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest suitable materials based on requirements."""
        process = self.processes[process_name]
        suggestions = []
        
        for material_name in process.materials:
            if material := self.materials.get(material_name):
                score = self._calculate_material_score(
                    material, requirements)
                if score > 0:
                    suggestions.append({
                        "material": material,
                        "score": score,
                        "reasons": self._generate_material_reasons(
                            material, requirements)
                    })
        
        return sorted(suggestions, key=lambda x: x["score"], reverse=True)

    def _calculate_material_score(self, material: MaterialProperties,
                               requirements: Dict[str, Any]) -> float:
        """Calculate how well a material meets requirements."""
        score = 100.0
        
        # Check strength requirements
        if "min_strength" in requirements:
            if material.tensile_strength < requirements["min_strength"]:
                score -= 30.0
                
        # Check thermal requirements
        if "max_temp" in requirements:
            if material.thermal_properties.get("melting_point", 0) < requirements["max_temp"]:
                score -= 30.0
                
        # Check cost requirements
        if "max_cost" in requirements:
            if material.cost_per_unit > requirements["max_cost"]:
                score -= 20.0
                
        return max(0.0, score)

    def _generate_material_reasons(self, material: MaterialProperties,
                                requirements: Dict[str, Any]) -> List[str]:
        """Generate reasons for material suggestion."""
        reasons = []
        
        if "min_strength" in requirements:
            if material.tensile_strength >= requirements["min_strength"]:
                reasons.append(
                    f"Meets strength requirement: {material.tensile_strength} MPa")
            else:
                reasons.append(
                    f"Below strength requirement: {material.tensile_strength} MPa")
                
        if "max_temp" in requirements:
            melting_point = material.thermal_properties.get("melting_point", 0)
            if melting_point >= requirements["max_temp"]:
                reasons.append(
                    f"Meets temperature requirement: {melting_point}°C")
            else:
                reasons.append(
                    f"Below temperature requirement: {melting_point}°C")
                
        if "max_cost" in requirements:
            if material.cost_per_unit <= requirements["max_cost"]:
                reasons.append(
                    f"Within cost requirement: {material.cost_per_unit}/{material.unit}")
            else:
                reasons.append(
                    f"Exceeds cost requirement: {material.cost_per_unit}/{material.unit}")
                
        return reasons 