"""
LLM client module for Claude 3.5 integration.
Handles communication and response parsing.
"""

import json
from typing import Dict, Any, List, Optional
import anthropic
from dataclasses import dataclass

@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    api_key: str
    model: str = "claude-3-sonnet-20240229"
    temperature: float = 0.7
    max_tokens: int = 4096

class LLMClient:
    """Client for interacting with Claude 3.5."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)
        self.context = []
    
    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Send analysis prompt to Claude and parse response.
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            Parsed response as dictionary
        """
        # Add engineering context
        enriched_prompt = self._enrich_prompt(prompt)
        
        try:
            response = self._call_api(enriched_prompt)
            return self._parse_response(response)
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return {"error": str(e)}
    
    def generate(self, prompt: str) -> str:
        """
        Generate CAD descriptions or operations.
        
        Args:
            prompt: Generation prompt
            
        Returns:
            Generated text
        """
        # Add CAD-specific context
        enriched_prompt = self._enrich_prompt(prompt, context_type="cad")
        
        try:
            response = self._call_api(enriched_prompt)
            return response.strip()
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            return f"Error: {str(e)}"
    
    def _enrich_prompt(self, prompt: str, context_type: str = "analysis") -> str:
        """Add context and examples to prompt."""
        if context_type == "analysis":
            context = """
            You are an expert CAD engineer with deep knowledge of:
            - Mechanical engineering principles
            - Manufacturing processes
            - Assembly design
            - GD&T (Geometric Dimensioning and Tolerancing)
            - Material properties
            
            Analyze the following request considering all engineering aspects.
            Provide detailed, structured responses with clear rationale.
            """
        else:  # CAD generation
            context = """
            You are an expert CAD system generating precise geometric operations.
            Consider:
            - Construction sequence
            - Feature dependencies
            - Parametric relationships
            - Manufacturing constraints
            
            Generate detailed, executable CAD operations.
            """
        
        return f"{context}\n\n{prompt}"
    
    def _call_api(self, prompt: str) -> str:
        """Make API call to Claude."""
        try:
            message = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse analysis response into structured format."""
        try:
            # First try to parse as direct JSON
            try:
                return json.loads(response)
            except:
                # If not direct JSON, try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise ValueError("No JSON found in response")
        except Exception as e:
            return {"error": f"Failed to parse response: {str(e)}"}

class CADPromptGenerator:
    """Generate structured prompts for CAD operations."""
    
    @staticmethod
    def geometric_analysis(description: str) -> str:
        """Generate prompt for geometric analysis."""
        return f"""
        Analyze this geometric description:
        {description}
        
        Extract:
        1. Primary geometric features
        2. Dimensional parameters
        3. Spatial relationships
        4. Symmetry patterns
        5. Construction hierarchy
        
        Format response as JSON with:
        {{
            "features": [
                {{
                    "type": "feature_type",
                    "parameters": {{...}},
                    "relationships": [...]
                }}
            ],
            "parameters": {{
                "name": {{
                    "value": number,
                    "unit": "unit",
                    "constraints": {{...}}
                }}
            }},
            "construction_sequence": [...]
        }}
        """
    
    @staticmethod
    def manufacturing_analysis(geometry: Dict) -> str:
        """Generate prompt for manufacturing analysis."""
        return f"""
        Analyze manufacturing requirements for:
        {json.dumps(geometry, indent=2)}
        
        Consider:
        1. Material selection
        2. Manufacturing processes
        3. Feature manufacturability
        4. Required tolerances
        5. Cost implications
        
        Format response as JSON with:
        {{
            "material_options": [
                {{
                    "material": "name",
                    "properties": {{...}},
                    "suitability_score": number,
                    "rationale": "..."
                }}
            ],
            "manufacturing": {{
                "primary_process": "...",
                "secondary_processes": [...],
                "critical_features": [
                    {{
                        "feature": "name",
                        "challenges": [...],
                        "solutions": [...]
                    }}
                ]
            }},
            "tolerances": {{...}},
            "cost_analysis": {{...}}
        }}
        """
    
    @staticmethod
    def assembly_analysis(parts: List[Dict], constraints: List[Dict]) -> str:
        """Generate prompt for assembly analysis."""
        return f"""
        Analyze assembly requirements for:
        Parts: {json.dumps(parts, indent=2)}
        Constraints: {json.dumps(constraints, indent=2)}
        
        Determine:
        1. Assembly sequence
        2. Mating relationships
        3. Tolerance stack-up
        4. Required fixtures
        5. Potential issues
        
        Format response as JSON with:
        {{
            "assembly_sequence": [
                {{
                    "step": number,
                    "parts": [...],
                    "operations": [...],
                    "critical_notes": [...]
                }}
            ],
            "mating_features": [
                {{
                    "part1": "name",
                    "part2": "name",
                    "type": "mate_type",
                    "parameters": {{...}}
                }}
            ],
            "tolerance_analysis": {{...}},
            "fixtures": [...],
            "issues": [
                {{
                    "type": "issue_type",
                    "description": "...",
                    "severity": number,
                    "solutions": [...]
                }}
            ]
        }}
        """

class ResponseValidator:
    """Validate and normalize LLM responses."""
    
    @staticmethod
    def validate_geometric_analysis(response: Dict) -> Dict:
        """Validate geometric analysis response."""
        required_fields = {
            "features",
            "parameters",
            "construction_sequence"
        }
        
        if not all(field in response for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - set(response.keys())}")
        
        # Validate feature structure
        for feature in response["features"]:
            if not all(k in feature for k in ["type", "parameters"]):
                raise ValueError(f"Invalid feature structure: {feature}")
        
        return response
    
    @staticmethod
    def validate_manufacturing_analysis(response: Dict) -> Dict:
        """Validate manufacturing analysis response."""
        required_fields = {
            "material_options",
            "manufacturing",
            "tolerances",
            "cost_analysis"
        }
        
        if not all(field in response for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - set(response.keys())}")
        
        # Validate manufacturing structure
        mfg = response["manufacturing"]
        if not all(k in mfg for k in ["primary_process", "secondary_processes"]):
            raise ValueError(f"Invalid manufacturing structure: {mfg}")
        
        return response
    
    @staticmethod
    def validate_assembly_analysis(response: Dict) -> Dict:
        """Validate assembly analysis response."""
        required_fields = {
            "assembly_sequence",
            "mating_features",
            "tolerance_analysis",
            "fixtures",
            "issues"
        }
        
        if not all(field in response for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - set(response.keys())}")
        
        # Validate assembly sequence
        for step in response["assembly_sequence"]:
            if not all(k in step for k in ["step", "parts", "operations"]):
                raise ValueError(f"Invalid assembly step: {step}")
        
        return response 