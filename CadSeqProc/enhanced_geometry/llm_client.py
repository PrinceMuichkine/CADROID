"""
LLM client for enhanced geometry system.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from ..utility.logger import CLGLogger, setup_logger
import anthropic
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ContentBlock, TextBlock

# Initialize logger with module name
logger = CLGLogger(__name__).configure_logger()

class LLMClient:
    """Client for interacting with Claude 3.5."""
    
    def __init__(self, model_type: str = "claude", timeout: int = 30) -> None:
        """Initialize LLM client.
        
        Args:
            model_type: Type of model to use (default: "claude")
            timeout: Timeout in seconds for API calls (default: 30)
        """
        self.logger = setup_logger(__name__)
        self.logger.debug(f"Initializing LLMClient with model_type={model_type}")
        self.model_type = model_type
        self.timeout = timeout
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            self.logger.warning("ANTHROPIC_API_KEY not found in environment variables")
        else:
            self.logger.debug("ANTHROPIC_API_KEY found")
            self.client = Anthropic(api_key=self.api_key)
        
        self.conversation_history: List[Dict[str, str]] = []
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text input for CAD generation.
        
        This is a wrapper around analyze_request that provides a simpler interface
        for basic text analysis without full conversation context.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing the analysis results
        """
        return await self.analyze_request(text)
    
    async def analyze_request(self, text: str) -> Dict[str, Any]:
        """Analyze a design request using Claude 3.5."""
        try:
            self.logger.debug(f"Analyzing request: {text}")
            
            # Add request to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Format prompt with context from conversation history
            context = self._build_context()
            self.logger.debug(f"Built context: {context}")
            
            prompt = f"""
            Based on the following conversation history and current request, analyze the CAD design requirements.
            Provide a structured analysis in JSON format with EXACTLY this structure:
            {{
                "dimensions": {{
                    "width": 10,
                    "height": 10,
                    "depth": 10,
                    "unit": "mm"
                }},
                "manufacturing": {{
                    "process": "3D printing",
                    "considerations": [
                        "Support structures needed",
                        "Layer adhesion critical"
                    ],
                    "constraints": [
                        {{"type": "min_wall_thickness", "value": 0.8, "unit": "mm"}},
                        {{"type": "max_overhang", "value": 45, "unit": "degrees"}}
                    ]
                }},
                "features": [
                    {{
                        "type": "cube",
                        "purpose": "main body",
                        "dimensions": {{
                            "width": 10,
                            "height": 10,
                            "depth": 10,
                            "unit": "mm"
                        }}
                    }}
                ]
            }}
            
            Context:
            {context}
            
            Current request: {text}
            
            Ensure your response is valid JSON and contains ALL the required fields.
            Adjust the values based on the current request but maintain this exact structure.
            Do not include any explanatory text, just return the JSON object.
            """
            
            # Call API with timeout
            try:
                response = await asyncio.wait_for(
                    self._call_api(prompt),
                    timeout=self.timeout
                )
                return response
            except asyncio.TimeoutError:
                self.logger.error(f"API call timed out after {self.timeout} seconds")
                return {"error": "Request timed out"}
            
        except Exception as e:
            self.logger.error(f"Error analyzing request: {str(e)}")
            return {"error": str(e)}
    
    async def generate_recommendations(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Generate manufacturing recommendations for a given geometry.
        
        Args:
            geometry: Dictionary containing geometry information
            
        Returns:
            Dict containing manufacturing recommendations
        """
        try:
            prompt = f"""
            Analyze the following CAD geometry and provide manufacturing recommendations.
            Return the analysis in JSON format with EXACTLY this structure:
            {{
                "best_process": "3D printing",
                "material_suggestions": [
                    {{"material": "PLA", "score": 0.9}},
                    {{"material": "PETG", "score": 0.8}}
                ],
                "considerations": [
                    "Support structures needed",
                    "Layer adhesion critical"
                ],
                "constraints": [
                    {{"type": "min_wall_thickness", "value": 0.8, "unit": "mm"}},
                    {{"type": "max_overhang", "value": 45, "unit": "degrees"}}
                ]
            }}
            
            Geometry:
            {json.dumps(geometry, indent=2)}
            
            Ensure your response is valid JSON and contains ALL the required fields.
            Do not include any explanatory text, just return the JSON object.
            """
            
            response = await asyncio.wait_for(
                self._call_api(prompt),
                timeout=self.timeout
            )
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {"error": str(e)}
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate that a response contains all required fields."""
        required_fields = ["dimensions", "manufacturing", "features"]
        return all(field in response for field in required_fields)
    
    def _build_context(self) -> str:
        """Build context string from conversation history."""
        context = ""
        for message in self.conversation_history:
            role = message["role"].capitalize()
            content = message["content"]
            context += f"{role}: {content}\n"
        return context
    
    async def _call_api(self, prompt: str) -> Dict[str, Any]:
        """Call the Claude API with the given prompt."""
        try:
            if not self.api_key:
                self.logger.warning("No API key available, using mock response")
                return self._get_mock_response()
            
            # Create message synchronously since Anthropic's client is not async
            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract JSON from response
            try:
                response_text = message.content[0].text
                # Find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
                else:
                    self.logger.error("No JSON found in response")
                    return {"error": "No JSON found in response"}
            except (json.JSONDecodeError, IndexError) as e:
                self.logger.error(f"Error parsing API response: {str(e)}")
                return {"error": "Invalid response format"}
            
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _get_mock_response(self) -> Dict[str, Any]:
        """Get a mock response for testing without API access."""
        return {
            "dimensions": {
                "width": 10,
                "height": 10,
                "depth": 10,
                "unit": "mm"
            },
            "manufacturing": {
                "process": "3D printing",
                "considerations": [
                    "Support structures needed",
                    "Layer adhesion critical"
                ],
                "constraints": [
                    {"type": "min_wall_thickness", "value": 0.8, "unit": "mm"},
                    {"type": "max_overhang", "value": 45, "unit": "degrees"}
                ]
            },
            "features": [
                {
                    "type": "cube",
                    "purpose": "main body",
                    "dimensions": {
                        "width": 10,
                        "height": 10,
                        "depth": 10,
                        "unit": "mm"
                    }
                }
            ]
        } 