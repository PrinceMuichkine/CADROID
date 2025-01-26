""" Intelligent CAD module leveraging LLM capabilities for enhanced generation. """

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from .base import Point, GeometricEntity
from .nurbs import NURBSCurve, NURBSSurface
from .organic import OrganicSurface
from .factory import OrganicShapeFactory
import json
from .enhanced_geometry.pattern_recognition import PatternRecognizer, DesignPattern

@dataclass
class PartMetadata:
    """Metadata about a part including manufacturing and assembly info."""
    name: str
    description: str
    material: Optional[str] = None
    manufacturing_process: Optional[str] = None
    tolerances: Dict[str, float] = field(default_factory=dict)
    assembly_constraints: List[Dict] = field(default_factory=list)

@dataclass
class CADContext:
    """Track conversation context and CAD state."""
    current_geometry: Optional[GeometricEntity] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "dimensions": {},  # Store dimensional parameters
        "features": {},    # Store feature parameters
        "constraints": {}, # Store geometric constraints
        "patterns": {}     # Store pattern information
    })
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    undo_stack: List[Dict[str, Any]] = field(default_factory=list)
    redo_stack: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.parameters is None:
            self.parameters = {
                "dimensions": {},  # Store dimensional parameters
                "features": {},    # Store feature parameters
                "constraints": {}, # Store geometric constraints
                "patterns": {}     # Store pattern information
            }
        if self.modifications is None:
            self.modifications = []
        if self.undo_stack is None:
            self.undo_stack = []
        if self.redo_stack is None:
            self.redo_stack = []

    def update_parameter(self, category: str, name: str, value: Any):
        """Update a parameter value and track the change."""
        if category not in self.parameters:
            self.parameters[category] = {}
            
        old_value = self.parameters[category].get(name)
        self.parameters[category][name] = value
            
        # Track change for undo/redo
        self.undo_stack.append({
            "type": "parameter_update",
            "category": category,
            "name": name,
            "old_value": old_value,
            "new_value": value
        })
        self.redo_stack.clear()  # Clear redo stack on new change

    def undo(self) -> Optional[Dict[str, Any]]:
        """Undo last change and return the change info."""
        if not self.undo_stack:
            return None
            
        change = self.undo_stack.pop()
        self.redo_stack.append(change)
            
        if change["type"] == "parameter_update":
            self.parameters[change["category"]][change["name"]] = change["old_value"]
            
        return change

    def redo(self) -> Optional[Dict[str, Any]]:
        """Redo last undone change and return the change info."""
        if not self.redo_stack:
            return None
            
        change = self.redo_stack.pop()
        self.undo_stack.append(change)
            
        if change["type"] == "parameter_update":
            self.parameters[change["category"]][change["name"]] = change["new_value"]
            
        return change

class IntelligentCAD:
    """CAD system enhanced with LLM capabilities."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.feature_recognizer = FeatureRecognizer(llm_client)
        self.assembly_planner = AssemblyPlanner(llm_client)
        self.manufacturing_validator = ManufacturingValidator(llm_client)
        self.pattern_recognizer = PatternRecognizer(llm_client)
        self.context = CADContext()

    def analyze_description(self, description: str) -> Dict[str, Any]:
        """
        Use LLM to analyze part description and extract key information.
        """
        # First, check if this is an organic shape request
        organic_analysis = self._analyze_organic_shape(description)
        if organic_analysis.get("is_organic"):
            return self._handle_organic_shape(organic_analysis, description)
            
        # Otherwise, proceed with normal geometric analysis
        prompt = f"""
        Analyze this CAD part/assembly description: "{description}"
            
        Extract and provide:
        1. Key geometric parameters and their values
        2. Material requirements
        3. Manufacturing constraints
        4. Assembly relationships
        5. Critical features and their purposes
        6. Suggested tolerances
            
        Format as structured JSON with clear parameter hierarchies.
        Include engineering rationale for key decisions.
        """
            
        analysis = self.llm_client.analyze(prompt)
        return self._validate_analysis(analysis)

    def _analyze_organic_shape(self, description: str) -> Dict[str, Any]:
        """Analyze if the request is for an organic shape and how to decompose it."""
        prompt = f"""
        Analyze if this describes an organic shape: "{description}"
            
        If it is organic, provide:
        1. Basic geometric primitives that can approximate the shape
        2. Mathematical curves/surfaces needed (e.g., Bezier, NURBS)
        3. Decomposition into simpler parts
        4. Key dimensions and proportions
        5. Symmetry patterns
            
        Return as JSON:
        {{
            "is_organic": boolean,
            "decomposition": [
                {{
                    "part": "part_name",
                    "primitive_type": "curve|surface|solid",
                    "parameters": {{...}},
                    "mathematical_description": "...",
                    "relative_position": {{...}}
                }}
            ],
            "symmetry": {{
                "type": "radial|bilateral|none",
                "count": number  # for radial symmetry
            }},
            "key_dimensions": {{...}}
        }}
        """
            
        return self.llm_client.analyze(prompt)

    def _handle_organic_shape(self, analysis: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Generate CAD description for organic shape using geometric approximation."""
        # Convert organic shape analysis into geometric operations
        prompt = f"""
        Convert this organic shape analysis into a CAD description:
        {json.dumps(analysis, indent=2)}
            
        Original description: "{description}"
            
        Generate a complete CAD description that:
        1. Uses basic geometric primitives to approximate organic forms
        2. Preserves key proportions and relationships
        3. Implements symmetry patterns
        4. Creates smooth transitions between parts
            
        Format as a single descriptive sentence suitable for Text2CAD model input.
        Focus on geometric terms that the model understands.
        """
            
        geometric_description = self.llm_client.generate(prompt)
            
        return {
            "description": geometric_description,
            "organic_analysis": analysis,
            "geometric_parameters": self._extract_parameters(analysis)
        }

    def _extract_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geometric parameters from organic shape analysis."""
        parameters: Dict[str, Dict[str, Any]] = {
            "dimensions": {},
            "features": {},
            "constraints": {},
            "patterns": {}
        }
            
        # Extract dimensions
        if "key_dimensions" in analysis:
            parameters["dimensions"].update(analysis["key_dimensions"])
            
        # Extract symmetry patterns
        if "symmetry" in analysis:
            parameters["patterns"]["symmetry"] = analysis["symmetry"]
            
        # Extract part features
        if "decomposition" in analysis:
            for part in analysis["decomposition"]:
                parameters["features"][part["part"]] = {
                    "type": part["primitive_type"],
                    "parameters": part["parameters"]
                }
            
        return parameters

    def apply_modifications(self, current_geometry: GeometricEntity, modifications: List[Dict]) -> str:
        """
        Instead of modifying geometry directly, generate a new text description
        that incorporates the modifications.
        """
        # Get current features
        features = self.feature_recognizer.analyze_features(current_geometry)
            
        # Create prompt to generate new description
        prompt = f"""
        Current CAD model features:
        {json.dumps(features, indent=2)}
            
        Requested modifications:
        {json.dumps(modifications, indent=2)}
            
        Generate a complete text description that preserves existing features
        and incorporates the requested modifications. Format as a single
        descriptive sentence suitable for Text2CAD model input.
        """
            
        new_description = self.llm_client.generate(prompt)
        return new_description

    def execute_sequence(self, sequence: List[Dict]) -> str:
        """
        Convert operation sequence to text description for Text2CAD.
        """
        prompt = f"""
        Convert this CAD operation sequence to a natural language description:
        {json.dumps(sequence, indent=2)}
            
        Generate a single descriptive sentence that captures all geometric
        features and relationships. Format suitable for Text2CAD model input.
        """
            
        description = self.llm_client.generate(prompt)
        return description

    def _validate_analysis(self, analysis: Dict) -> Dict:
        """Validate and normalize analyzed parameters."""
        required_fields = ["geometric_parameters", "features", "constraints"]
        for field in required_fields:
            if field not in analysis:
                analysis[field] = {}
        return analysis

    def generate_part_sequence(self, metadata: PartMetadata) -> List[Dict]:
        """
        Generate CAD sequence for part creation.
            
        Args:
            metadata: Part metadata from analysis
            
        Returns:
            List of CAD operations to create part
        """
        prompt = f"""
        Given this part specification:
        Name: {metadata.name}
        Description: {metadata.description}
        Material: {metadata.material}
        Manufacturing: {metadata.manufacturing_process}
            
        Generate a detailed CAD sequence considering:
        1. Optimal construction order
        2. Manufacturing constraints
        3. Feature relationships
        4. Required tolerances
            
        Format as sequence of geometric operations.
        """
            
        sequence = self.llm_client.generate(prompt)
        return self._validate_sequence(sequence)

    def _validate_sequence(self, sequence: List[Dict]) -> List[Dict]:
        """Validate generated CAD sequence."""
        # Implement validation logic
        return sequence

    def suggest_improvements(self, part: GeometricEntity, context: str) -> List[Dict]:
        """
        Suggest improvements for part design.
            
        Args:
            part: Current part geometry
            context: Usage context and requirements
            
        Returns:
            List of suggested improvements
        """
        prompt = f"""
        Analyze this part design in context: "{context}"
            
        Suggest improvements considering:
        1. Structural integrity
        2. Manufacturing efficiency
        3. Assembly optimization
        4. Material usage
        5. Cost reduction
            
        Provide specific geometric modifications.
        """
            
        suggestions = self.llm_client.analyze(prompt)
        return self._validate_suggestions(suggestions)

    def _validate_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Validate improvement suggestions."""
        # Implement validation logic
        return suggestions

    def analyze_design_patterns(self, geometry: GeometricEntity) -> List[DesignPattern]:
        """Analyze geometry for design patterns and optimization opportunities."""
        patterns = self.pattern_recognizer.analyze_geometry(geometry)
        
        # Update context with pattern information
        self.context.parameters["patterns"].update({
            pattern.name: {
                "type": pattern.features[0].pattern_type,
                "instances": len(pattern.features[0].instances),
                "confidence": pattern.features[0].confidence
            } for pattern in patterns
        })
        
        return patterns

    def suggest_pattern_optimizations(self, patterns: List[DesignPattern]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on recognized patterns."""
        suggestions = []
        
        for pattern in patterns:
            # Manufacturing optimization
            if pattern.manufacturing_notes:
                suggestions.append({
                    "type": "manufacturing",
                    "pattern": pattern.name,
                    "suggestions": pattern.manufacturing_notes
                })
                
            # Reuse opportunities
            if pattern.reuse_suggestions:
                suggestions.append({
                    "type": "reuse",
                    "pattern": pattern.name,
                    "suggestions": pattern.reuse_suggestions
                })
                
            # Performance optimization
            if pattern.features[0].pattern_type in ["linear_array", "circular_array"]:
                suggestions.append({
                    "type": "performance",
                    "pattern": pattern.name,
                    "suggestion": "Consider using native CAD pattern features for better performance"
                })
        
        return suggestions

    def apply_pattern_optimization(self, geometry: GeometricEntity, 
                                 optimization: Dict[str, Any]) -> GeometricEntity:
        """Apply a pattern-based optimization to the geometry."""
        if optimization["type"] == "manufacturing":
            # Apply manufacturing optimizations
            return self._apply_manufacturing_optimization(geometry, optimization)
        elif optimization["type"] == "reuse":
            # Apply pattern reuse
            return self._apply_pattern_reuse(geometry, optimization)
        elif optimization["type"] == "performance":
            # Apply performance optimizations
            return self._apply_performance_optimization(geometry, optimization)
        
        return geometry

    def _apply_manufacturing_optimization(self, geometry: GeometricEntity, 
                                       optimization: Dict[str, Any]) -> GeometricEntity:
        """Apply manufacturing-specific optimizations."""
        # Implementation for manufacturing optimization
        return geometry

    def _apply_pattern_reuse(self, geometry: GeometricEntity, 
                            optimization: Dict[str, Any]) -> GeometricEntity:
        """Apply pattern reuse optimizations."""
        # Implementation for pattern reuse
        return geometry

    def _apply_performance_optimization(self, geometry: GeometricEntity, 
                                      optimization: Dict[str, Any]) -> GeometricEntity:
        """Apply performance optimizations."""
        # Implementation for performance optimization
        return geometry

class FeatureRecognizer:
    """Recognize and classify CAD features using LLM."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def analyze_features(self, geometry: GeometricEntity) -> Dict[str, Any]:
        """
        Analyze geometric features and their purposes.
            
        Args:
            geometry: Part geometry to analyze
            
        Returns:
            Dictionary of features and their classifications
        """
        # Convert geometry to analyzable format
        geo_desc = self._geometry_to_description(geometry)
            
        prompt = f"""
        Analyze these geometric features:
        {geo_desc}
            
        Identify and classify:
        1. Primary features (holes, bosses, etc.)
        2. Manufacturing features
        3. Assembly features
        4. Structural features
            
        Explain the purpose of each feature.
        """
            
        return self.llm_client.analyze(prompt)

    def _geometry_to_description(self, geometry: GeometricEntity) -> str:
        """Convert geometry to textual description for LLM."""
        # Implement conversion logic
        return "Geometric description"

class AssemblyPlanner:
    """Plan and validate assembly sequences using LLM."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan_assembly(self, parts: List[GeometricEntity],
                     constraints: List[Dict]) -> Dict[str, Any]:
        """
        Plan assembly sequence and validate constraints.
            
        Args:
            parts: List of parts to assemble
            constraints: Assembly constraints
            
        Returns:
            Assembly plan and validation results
        """
        # Convert parts and constraints to analyzable format
        assembly_desc = self._create_assembly_description(parts, constraints)
            
        prompt = f"""
        Plan assembly sequence for:
        {assembly_desc}
            
        Provide:
        1. Optimal assembly order
        2. Required fixtures/tooling
        3. Critical alignments
        4. Tolerance stack-up analysis
        5. Potential interference issues
            
        Consider manufacturing and assembly constraints.
        """
            
        return self.llm_client.analyze(prompt)

    def _create_assembly_description(self, parts: List[GeometricEntity],
                                  constraints: List[Dict]) -> str:
        """Create assembly description for LLM."""
        # Implement conversion logic
        return "Assembly description"

class ManufacturingValidator:
    """Validate manufacturing feasibility using LLM."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def validate_design(self, part: GeometricEntity,
                      process: str) -> Dict[str, Any]:
        """
        Validate design for manufacturing process.
            
        Args:
            part: Part geometry
            process: Manufacturing process
            
        Returns:
            Validation results and suggestions
        """
        # Convert part to analyzable format
        design_desc = self._create_design_description(part)
            
        prompt = f"""
        Validate this design for {process} manufacturing:
        {design_desc}
            
        Check for:
        1. Process-specific constraints
        2. Feature manufacturability
        3. Required tolerances
        4. Cost implications
        5. Potential optimizations
            
        Suggest specific improvements.
        """
            
        return self.llm_client.analyze(prompt)

    def _create_design_description(self, part: GeometricEntity) -> str:
        """Create design description for LLM."""
        # Implement conversion logic
        return "Design description"

class ConversationalCAD:
    """Handles multi-turn CAD generation with context."""
    
    def __init__(self, llm_client, intelligent_cad):
        self.llm = llm_client
        self.cad = intelligent_cad
        self.context = CADContext()
        self.current_description = None

    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process a user request in context of previous interactions."""
        # Add request to history
        self.context.history.append({
            "user_input": user_input,
            "timestamp": np.datetime64('now')
        })
            
        # First, check if this is a command (undo/redo)
        if user_input.lower().strip() == "undo":
            return self._handle_undo()
        elif user_input.lower().strip() == "redo":
            return self._handle_redo()
            
        # Then check if it's a parameter modification
        param_update = self._check_parameter_update(user_input)
        if param_update:
            return self._handle_parameter_update(param_update)
            
        # Finally, check if it's a geometric modification
        is_modification = self._is_modification_request(user_input)
        if is_modification and self.current_description:
            return self._handle_modification(user_input)
        else:
            return self._handle_new_request(user_input)

    def _check_parameter_update(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Check if the request is to update parameters."""
        prompt = f"""
        Analyze if this request modifies parameters:
        "{user_input}"
            
        Current parameters:
        {json.dumps(self.context.parameters, indent=2)}
            
        If this is a parameter update, return JSON:
        {{
            "is_parameter_update": true,
            "updates": [
                {{
                    "category": "dimensions|features|constraints|patterns",
                    "name": "parameter_name",
                    "value": new_value,
                    "unit": "unit_if_applicable"
                }}
            ]
        }}
        Otherwise return: {{"is_parameter_update": false}}
        """
            
        analysis = self.llm.analyze(prompt)
        if analysis.get("is_parameter_update"):
            return analysis
        return None

    def _handle_parameter_update(self, update_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle parameter update request."""
        try:
            for update in update_info["updates"]:
                self.context.update_parameter(
                    update["category"],
                    update["name"],
                    update["value"]
                )
                
            # Generate new description with updated parameters
            prompt = f"""
            Current model: "{self.current_description}"
            Updated parameters:
            {json.dumps(update_info["updates"], indent=2)}
                
            Generate a new complete description that:
            1. Incorporates the updated parameters
            2. Preserves other existing features
            3. Is formatted as a single clear sentence
            4. Is suitable for Text2CAD model input
            """
                
            new_description = self.llm.generate(prompt)
            self.current_description = new_description
                
            return {
                "text": new_description,
                "success": True,
                "message": "Parameters updated successfully",
                "updates": update_info["updates"]
            }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error updating parameters: {str(e)}"
            }

    def _handle_undo(self) -> Dict[str, Any]:
        """Handle undo request."""
        change = self.context.undo()
        if not change:
            return {
                "success": False,
                "message": "Nothing to undo"
            }
            
        # Generate new description based on undone change
        prompt = f"""
        Current model: "{self.current_description}"
        Undone change: {json.dumps(change, indent=2)}
            
        Generate a new complete description that:
        1. Reflects the undone change
        2. Preserves other features
        3. Is formatted as a single clear sentence
        4. Is suitable for Text2CAD model input
        """
            
        try:
            new_description = self.llm.generate(prompt)
            self.current_description = new_description
                
            return {
                "text": new_description,
                "success": True,
                "message": f"Undid {change['type']}"
            }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error applying undo: {str(e)}"
            }

    def _handle_redo(self) -> Dict[str, Any]:
        """Handle redo request."""
        change = self.context.redo()
        if not change:
            return {
                "success": False,
                "message": "Nothing to redo"
            }
            
        # Generate new description based on redone change
        prompt = f"""
        Current model: "{self.current_description}"
        Redone change: {json.dumps(change, indent=2)}
            
        Generate a new complete description that:
        1. Reflects the redone change
        2. Preserves other features
        3. Is formatted as a single clear sentence
        4. Is suitable for Text2CAD model input
        """
            
        try:
            new_description = self.llm.generate(prompt)
            self.current_description = new_description
                
            return {
                "text": new_description,
                "success": True,
                "message": f"Redid {change['type']}"
            }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error applying redo: {str(e)}"
            }

    def _is_modification_request(self, user_input: str) -> bool:
        """Determine if request is a modification to existing geometry."""
        if not self.current_description:
            return False
            
        # Use LLM to analyze if request is a modification
        prompt = f"""
        Previous CAD model description: "{self.current_description}"
        New request: "{user_input}"
            
        Analyze if this is a modification request by checking:
        1. If it references the existing model
        2. If it suggests changes to existing features
        3. If it adds/removes/modifies parts
            
        Return JSON: {{"is_modification": boolean, "modification_type": string}}
        """
            
        analysis = self.llm.analyze(prompt)
        return analysis.get("is_modification", False)

    def _handle_modification(self, user_input: str) -> Dict[str, Any]:
        """Handle modification by generating new complete description."""
        prompt = f"""
        Current CAD model: "{self.current_description}"
        Requested change: "{user_input}"
            
        Generate a new complete description that:
        1. Preserves existing features
        2. Incorporates requested changes
        3. Is formatted as a single clear sentence
        4. Is suitable for Text2CAD model input
            
        Return the new description only.
        """
            
        try:
            # Generate new complete description
            new_description = self.llm.generate(prompt)
                
            # Update context
            self.current_description = new_description
            self.context.modifications.append({
                "request": user_input,
                "new_description": new_description
            })
                
            return {
                "text": new_description,
                "success": True,
                "message": "Description updated successfully"
            }
                
        except Exception as e:
            return {
                "text": self.current_description,
                "success": False,
                "message": f"Error updating description: {str(e)}"
            }

    def _handle_new_request(self, user_input: str) -> Dict[str, Any]:
        """Handle request for new geometry."""
        # Reset context for new request
        self.context = CADContext()
        self.current_description = user_input
            
        try:
            # Analyze and validate the description
            metadata = self.cad.analyze_description(user_input)
                
            return {
                "text": user_input,
                "success": True,
                "message": "New description processed successfully"
            }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing description: {str(e)}"
            } 