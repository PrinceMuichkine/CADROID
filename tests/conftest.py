"""Test configuration and common fixtures."""

import os
import sys
import pytest
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CadSeqProc.pipeline import CADPipeline, PipelineConfig
from CadSeqProc.enhanced_geometry.intelligent_cad import IntelligentCAD
from CadSeqProc.enhanced_geometry.llm_client import LLMClient

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]

@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_config() -> PipelineConfig:
    """Create test configuration."""
    return PipelineConfig(
        debug=True,
        cache_dir="./tests/cache",
        output_dir="./tests/output"
    )

@pytest.fixture
def llm_client() -> LLMClient:
    """Create LLM client for testing."""
    return LLMClient(model_type="claude")

@pytest.fixture
def cad_system(llm_client) -> IntelligentCAD:
    """Create CAD system for testing."""
    return IntelligentCAD(llm_client)

@pytest.fixture
def pipeline(test_config) -> CADPipeline:
    """Create pipeline for testing."""
    return CADPipeline(test_config)

# Test data
@pytest.fixture
def simple_cube_prompt() -> str:
    """Simple cube test prompt."""
    return "Create a simple cube with 10cm sides"

@pytest.fixture
def complex_shape_prompt() -> str:
    """Complex shape test prompt."""
    return """Create a cylindrical container with:
    - 15cm height
    - 8cm diameter
    - 2mm wall thickness
    - Threaded lid
    - Four evenly spaced mounting holes near the base"""

@pytest.fixture
def parametric_shape_prompt() -> str:
    """Parametric shape test prompt."""
    return """Design an adjustable bracket with:
    - Base plate: 10cm x 5cm
    - Mounting holes: 4 x M4
    - Adjustable arm length: 15-25cm
    - Load capacity: 5kg"""

@pytest.fixture
def organic_shape_prompt() -> str:
    """Organic shape test prompt."""
    return """Create an ergonomic handle with:
    - Curved grip surface
    - Finger indentations
    - Mounting point at base
    - Overall length: 12cm""" 