"""Core pipeline module for orchestrating CAD operations."""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from PIL import Image

from .intelligent_cad import IntelligentCAD
from .llm_client import LLMClient
from .enhanced_geometry.pattern_recognition import PatternRecognizer, DesignPattern
from .manufacturing.manufacturing_analyzer import ManufacturingAnalyzer
from .base import GeometricEntity

@dataclass
class PipelineConfig:
    """Configuration for the CAD pipeline."""
    model_type: str = "claude"
    cache_dir: str = "./App/cache"
    output_dir: str = "./App/output"
    debug: bool = False
    manufacturing_settings: Dict[str, Any] = field(default_factory=dict)
    pattern_recognition_settings: Dict[str, Any] = field(default_factory=dict)

class CADPipeline:
    """Main pipeline for CAD operations."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.llm_client = LLMClient(model_type=self.config.model_type)
        self.cad_system = IntelligentCAD(self.llm_client)
        self.pattern_recognizer = PatternRecognizer(self.llm_client)
        self.manufacturing_analyzer = ManufacturingAnalyzer(self.llm_client)
        
        # Ensure directories exist
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def process(self, input_data: Union[str, Dict, Path]) -> Dict[str, Any]:
        """Process input through the complete CAD pipeline.
        
        Args:
            input_data: Can be:
                - Text description of desired CAD model
                - Dictionary with model specifications
                - Path to input file (image, CAD file, etc.)
                
        Returns:
            Dictionary containing:
                - Generated CAD model
                - Analysis results
                - Manufacturing recommendations
                - Pattern recognition results
        """
        try:
            # 1. Input Processing
            processed_input = self._process_input(input_data)
            
            # 2. Generate Initial CAD Model
            model_data = self._generate_cad_model(processed_input)
            
            # 3. Pattern Recognition
            patterns = self._analyze_patterns(model_data["geometry"])
            
            # 4. Manufacturing Analysis
            manufacturing_data = self._analyze_manufacturing(
                model_data["geometry"], patterns)
            
            # 5. Optimization Suggestions
            optimizations = self._generate_optimizations(
                model_data["geometry"], patterns, manufacturing_data)
            
            # 6. Final Processing
            result = self._process_results(
                model_data, patterns, manufacturing_data, optimizations)
            
            return result
            
        except Exception as e:
            if self.config.debug:
                raise
            return {
                "success": False,
                "error": str(e),
                "stage": "pipeline_processing"
            }

    def _process_input(self, input_data: Union[str, Dict, Path]) -> Dict[str, Any]:
        """Process and validate input data."""
        if isinstance(input_data, str):
            return {"type": "text", "content": input_data}
        elif isinstance(input_data, dict):
            return {"type": "specs", "content": input_data}
        elif isinstance(input_data, Path):
            return {"type": "file", "content": self._load_file(input_data)}
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and process input file."""
        extension = file_path.suffix.lower()
        if extension in ['.jpg', '.png', '.jpeg']:
            return self._load_image(file_path)
        elif extension in ['.step', '.stp']:
            return self._load_cad_file(file_path)
        elif extension == '.json':
            return self._load_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def _load_image(self, file_path: Path) -> Dict[str, Any]:
        """Load and process image file."""
        try:
            image = Image.open(file_path)
            # Convert to format expected by system
            return {
                "type": "image",
                "format": image.format.lower(),
                "size": image.size,
                "data": np.array(image),
                "path": str(file_path)
            }
        except Exception as e:
            raise ValueError(f"Error loading image {file_path}: {str(e)}")

    def _load_cad_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and process CAD file."""
        try:
            # Use CAD system to load geometry
            geometry = self.cad_system.import_geometry(file_path)
            return {
                "type": "cad",
                "geometry": geometry,
                "path": str(file_path)
            }
        except Exception as e:
            raise ValueError(f"Error loading CAD file {file_path}: {str(e)}")

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load and process JSON file."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            return {
                "type": "json",
                "content": data,
                "path": str(file_path)
            }
        except Exception as e:
            raise ValueError(f"Error loading JSON file {file_path}: {str(e)}")

    def _generate_cad_model(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CAD model from processed input."""
        if processed_input["type"] == "text":
            # Use LLM to analyze description
            analysis = self.cad_system.analyze_description(
                processed_input["content"])
            
            # Generate geometry
            geometry = self.cad_system.generate_part_sequence(
                analysis["metadata"])
                
        elif processed_input["type"] == "specs":
            # Direct generation from specifications
            geometry = self.cad_system.execute_sequence(
                processed_input["content"])
                
        elif processed_input["type"] == "file":
            # Process existing geometry
            geometry = processed_input["content"]["geometry"]
            
        return {
            "geometry": geometry,
            "metadata": analysis if "analysis" in locals() else None
        }

    def _analyze_patterns(self, geometry: GeometricEntity) -> List[DesignPattern]:
        """Analyze geometry for patterns."""
        return self.pattern_recognizer.analyze_geometry(geometry)

    def _analyze_manufacturing(self, geometry: GeometricEntity,
                             patterns: List[DesignPattern]) -> Dict[str, Any]:
        """Perform manufacturing analysis."""
        # Analyze for different manufacturing processes
        analyses = {}
        for process in ["3d_printing_fdm", "cnc_milling"]:
            analyses[process] = self.manufacturing_analyzer.analyze_manufacturability(
                geometry, process)
            
        # Get material suggestions
        material_suggestions = self.manufacturing_analyzer.suggest_material(
            geometry, "3d_printing_fdm", {
                "min_strength": 40.0,
                "max_temp": 100.0,
                "max_cost": 30.0
            })
            
        return {
            "analyses": analyses,
            "material_suggestions": material_suggestions
        }

    def _generate_optimizations(self, geometry: GeometricEntity,
                              patterns: List[DesignPattern],
                              manufacturing_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions."""
        # Get pattern-based optimizations
        pattern_opts = self.cad_system.suggest_pattern_optimizations(patterns)
        
        # Get manufacturing optimizations
        mfg_opts = []
        for process, analysis in manufacturing_data["analyses"].items():
            if "recommendations" in analysis:
                mfg_opts.extend([
                    {"type": "manufacturing", "process": process, "suggestion": rec}
                    for rec in analysis["recommendations"]
                ])
                
        return pattern_opts + mfg_opts

    def _process_results(self, model_data: Dict[str, Any],
                        patterns: List[DesignPattern],
                        manufacturing_data: Dict[str, Any],
                        optimizations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and format final results."""
        return {
            "success": True,
            "geometry": model_data["geometry"],
            "metadata": model_data.get("metadata"),
            "patterns": [
                {
                    "name": p.name,
                    "description": p.description,
                    "type": p.features[0].pattern_type,
                    "confidence": p.features[0].confidence
                }
                for p in patterns
            ],
            "manufacturing": {
                "best_process": self._determine_best_process(manufacturing_data),
                "material_suggestions": manufacturing_data["material_suggestions"][:3],
                "analyses": manufacturing_data["analyses"]
            },
            "optimizations": optimizations
        }

    def _determine_best_process(self, manufacturing_data: Dict[str, Any]) -> str:
        """Determine the best manufacturing process based on analysis."""
        best_score = -1
        best_process = None
        
        for process, analysis in manufacturing_data["analyses"].items():
            score = analysis.get("manufacturability_score", 0)
            if score > best_score:
                best_score = score
                best_process = process
                
        return best_process

    def save_results(self, results: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """Save pipeline results to file."""
        if output_path is None:
            output_path = Path(self.config.output_dir) / "results.json"
            
        # Save geometry
        if "geometry" in results:
            geo_path = output_path.parent / "model.step"
            self.cad_system.export_geometry(results["geometry"], geo_path)
            results["geometry"] = str(geo_path)
            
        # Save complete results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return output_path 
