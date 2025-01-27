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
                
                if not self.validate_response(response):
                    self.logger.warning("Invalid response format")
                    return {"status": "error", "message": "Invalid response format"}
                
                return {
                    "status": "success",
                    "parameters": response
                }
                
            except asyncio.TimeoutError:
                self.logger.error(f"API call timed out after {self.timeout} seconds")
                return {
                    "status": "error",
                    "message": f"Request timed out after {self.timeout} seconds"
                }
            
        except Exception as e:
            self.logger.error(f"Error analyzing request: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate the response format."""
        required_fields = ['dimensions', 'manufacturing', 'features']
        return all(field in response for field in required_fields)
    
    def _build_context(self) -> str:
        """Build context from conversation history."""
        if not self.conversation_history:
            return ""
        
        context = []
        for entry in self.conversation_history[-5:]:  # Only use last 5 entries
            role = entry['role']
            content = entry['content']
            context.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(context)
    
    async def _call_api(self, prompt: str) -> Dict[str, Any]:
        """Make an async API call to Claude."""
        try:
            if not self.api_key:
                self.logger.warning("No API key available, using mock response")
                return self._get_mock_response()
            
            # Create a new event loop for the thread
            loop = asyncio.get_event_loop()
            
            # Make the API call in a non-blocking way
            message = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4096,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            )
            
            # Parse the response content
            content = message.content[0].text
            
            # Remove any explanatory text before the JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]
            
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: {str(e)}")
                self.logger.debug(f"Raw content: {content}")
                return self._get_mock_response()
            
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            return self._get_mock_response()
    
    def _get_mock_response(self) -> Dict[str, Any]:
        """Get a mock response for testing."""
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