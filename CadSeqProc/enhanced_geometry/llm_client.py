"""
LLM client for handling interactions with Claude 3.5.
"""

import os
from typing import Dict, Any, Optional
from ..utility.logger import setup_logger

logger = setup_logger(__name__)

class LLMClient:
    """Client for interacting with Claude 3.5."""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
    
    def analyze_request(self, text: str) -> Dict[str, Any]:
        """Analyze a design request using Claude 3.5."""
        try:
            # Format prompt for design analysis
            prompt = f"""
            Analyze this CAD design request: "{text}"
            
            Extract and provide:
            1. Shape properties and their intensities (0-1 scale)
            2. Material suggestions
            3. Manufacturing considerations
            4. Key features and dimensions
            
            Format response as JSON with this structure:
            {{
                "properties": {{
                    "density": {{"petals": 0.8, "holes": 0.3}},
                    "size": {{"small": 0.2, "large": 0.8}},
                    "complexity": {{"simple": 0.3, "complex": 0.7}}
                }},
                "material": {{
                    "primary": "PLA",
                    "alternatives": ["ABS", "PETG"]
                }},
                "manufacturing": {{
                    "process": "3D printing",
                    "considerations": [
                        "Support structures needed",
                        "Layer adhesion critical"
                    ]
                }},
                "features": [
                    {{
                        "type": "hole",
                        "purpose": "mounting",
                        "count": 4
                    }}
                ]
            }}
            """
            
            # TODO: Implement actual Claude 3.5 API call
            # For now, return mock response for development
            mock_response = {
                "properties": {
                    "density": {"petals": 0.8, "holes": 0.3},
                    "size": {"small": 0.2, "large": 0.8},
                    "complexity": {"simple": 0.3, "complex": 0.7}
                },
                "material": {
                    "primary": "PLA",
                    "alternatives": ["ABS", "PETG"]
                },
                "manufacturing": {
                    "process": "3D printing",
                    "considerations": [
                        "Support structures needed",
                        "Layer adhesion critical"
                    ]
                },
                "features": [
                    {
                        "type": "hole",
                        "purpose": "mounting",
                        "count": 4
                    }
                ]
            }
            
            return mock_response
            
        except Exception as e:
            logger.error(f"Error analyzing request: {str(e)}")
            return {
                "error": str(e),
                "properties": {},
                "material": {},
                "manufacturing": {},
                "features": []
            }
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate the structure of an LLM response."""
        required_fields = ["properties", "material", "manufacturing", "features"]
        return all(field in response for field in required_fields)
    
    def extract_parameters(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract CAD parameters from LLM response."""
        parameters = {}
        
        # Extract property-based parameters
        if "properties" in response:
            for category, values in response["properties"].items():
                for prop, intensity in values.items():
                    param_name = f"{category}_{prop}"
                    parameters[param_name] = intensity
        
        # Extract feature-based parameters
        if "features" in response:
            for feature in response["features"]:
                if "type" in feature and "count" in feature:
                    parameters[f"{feature['type']}_count"] = feature["count"]
        
        return parameters 