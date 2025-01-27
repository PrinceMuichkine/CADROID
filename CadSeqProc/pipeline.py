"""Core pipeline module for orchestrating CAD operations."""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from PIL import Image

from .enhanced_geometry.intelligent_cad import IntelligentCAD
from .enhanced_geometry.llm_client import LLMClient
from .enhanced_geometry.pattern_recognition import PatternRecognizer, DesignPattern
from .manufacturing.manufacturing_analyzer import ManufacturingAnalyzer
from .enhanced_geometry.base import Point, GeometricEntity
from .utility.logger import CLGLogger

# Initialize logger
logger = CLGLogger(__name__).configure_logger()

@dataclass
class PipelineConfig:
    """Configuration for the CAD pipeline."""
    model_type: str = "claude"
    cache_dir: str = "./App/cache"
    output_dir: str = "./App/output"
    debug: bool = False
    timeout: int = 30
    manufacturing_settings: Dict[str, Any] = field(default_factory=dict)
    pattern_recognition_settings: Dict[str, Any] = field(default_factory=dict)

class CADPipeline:
    """Main pipeline for CAD operations."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize pipeline with configuration."""
        self.config = config or PipelineConfig()
        self.llm_client = LLMClient(
            model_type=self.config.model_type,
            timeout=self.config.timeout
        )
        self.cad_system = IntelligentCAD(self.llm_client)
        self.pattern_recognizer = PatternRecognizer(self.llm_client)
        self.manufacturing_analyzer = ManufacturingAnalyzer(self.llm_client)
        self.cache_dir = Path(self.config.cache_dir)
        self.output_dir = Path(self.config.output_dir)
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def process(self, text_input: str) -> Dict[str, Any]:
        """Process a text request through the pipeline."""
        try:
            # Analyze text and generate initial model
            result = await self.cad_system.analyze_description(text_input)
            
            if result["status"] != "success":
                return {
                    "success": False,
                    "error": result.get("message", "Unknown error occurred")
                }
            
            # Extract parameters
            parameters = result["parameters"]
            
            # Generate the CAD model
            model_data = self._generate_cad_model(parameters)
            
            # Save the model
            output_path = self.save_results(model_data)
            
            return {
                "success": True,
                "model_path": str(output_path),
                "parameters": parameters,
                "analysis": {
                    "manufacturing": parameters.get("manufacturing", {}),
                    "features": parameters.get("features", []),
                    "validation": result.get("validation", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {str(e)}")
            return {
                "success": False,
                "error": str(e)
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

    def _generate_cad_model(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CAD model from parameters."""
        try:
            # Extract dimensions
            dimensions = parameters.get("dimensions", {})
            
            # Create a simple model based on parameters
            model_data = {
                "type": "solid",
                "operations": [
                    {
                        "type": "cube",
                        "parameters": {
                            "width": dimensions.get("width", 10.0),
                            "height": dimensions.get("height", 10.0),
                            "depth": dimensions.get("depth", 10.0),
                            "unit": dimensions.get("unit", "mm")
                        }
                    }
                ]
            }
            
            # Apply manufacturing constraints
            if manufacturing := parameters.get("manufacturing", {}):
                if constraints := manufacturing.get("constraints", []):
                    for constraint in constraints:
                        if constraint["type"] == "min_wall_thickness":
                            model_data["wall_thickness"] = constraint["value"]
                        elif constraint["type"] == "max_overhang":
                            model_data["max_overhang"] = constraint["value"]
            
            return model_data
            
        except Exception as e:
            logger.error(f"Error generating CAD model: {str(e)}")
            raise

    def _analyze_patterns(self, geometry: GeometricEntity) -> List[DesignPattern]:
        """Analyze geometry for patterns."""
        return self.pattern_recognizer.analyze_geometry(geometry)

    async def _analyze_manufacturing(self, geometry: GeometricEntity,
                             patterns: List[DesignPattern]) -> Dict[str, Any]:
        """Perform manufacturing analysis."""
        # Analyze for different manufacturing processes
        analyses = {}
        for process in ["3d_printing_fdm", "cnc_milling"]:
            analyses[process] = await self.manufacturing_analyzer.analyze_manufacturability(
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

    def save_results(self, model_data: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """Save the generated model."""
        try:
            if output_path is None:
                output_path = self.output_dir / "output.stl"
            
            # Save model data as JSON for now
            with open(output_path.with_suffix(".json"), "w") as f:
                json.dump(model_data, f, indent=2)
            
            # TODO: Implement actual STL generation
            # For now, create a dummy STL file
            with open(output_path, "w") as f:
                f.write("solid cube\n")
                f.write("  facet normal 0 0 0\n")
                f.write("    outer loop\n")
                f.write("      vertex 0 0 0\n")
                f.write("      vertex 1 0 0\n")
                f.write("      vertex 1 1 0\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
                f.write("endsolid cube\n")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def analyze_model(self, model_path: Path) -> Dict[str, Any]:
        """Analyze an existing CAD model."""
        try:
            # Load the model
            geometry = self.cad_system.load_model(str(model_path))
            
            # Analyze patterns
            pattern_result = self.cad_system.analyze_patterns(geometry)
            
            # Analyze manufacturability
            manufacturing_analysis = self.manufacturing_analyzer.analyze(
                geometry,
                pattern_result.get("parameters", {})
            )
            
            # Generate optimization suggestions
            optimizations = self.cad_system.suggest_optimizations(
                geometry,
                pattern_result,
                manufacturing_analysis
            )
            
            return {
                "success": True,
                "patterns": pattern_result.get("patterns", []),
                "manufacturing": manufacturing_analysis,
                "optimizations": optimizations
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 
