"""Integration tests for the CAD pipeline."""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from CadSeqProc.pipeline import CADPipeline, PipelineConfig
from CadSeqProc.enhanced_geometry.intelligent_cad import IntelligentCAD
from CadSeqProc.enhanced_geometry.llm_client import LLMClient

class TestPipeline:
    """Test suite for the CAD pipeline."""
    
    def __init__(self):
        """Initialize test environment."""
        self.config = PipelineConfig(
            debug=True,
            cache_dir="./tests/integration/cache",
            output_dir="./tests/integration/output"
        )
        self.pipeline = CADPipeline(self.config)
        
    async def test_llm_client(self) -> Dict[str, Any]:
        """Test LLM client functionality."""
        print("\n🔍 Testing LLM Client...")
        try:
            result = await self.pipeline.llm_client.analyze_text(
                "Create a simple cube with 10cm sides"
            )
            print("✅ LLM Client test passed")
            return {"success": True, "result": result}
        except Exception as e:
            print(f"❌ LLM Client test failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def test_intelligent_cad(self) -> Dict[str, Any]:
        """Test intelligent CAD system."""
        print("\n🔍 Testing Intelligent CAD...")
        try:
            result = await self.pipeline.cad_system.analyze_description(
                "Create a simple cube with 10cm sides"
            )
            print("✅ Intelligent CAD test passed")
            return {"success": True, "result": result}
        except Exception as e:
            print(f"❌ Intelligent CAD test failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def test_pattern_recognition(self) -> Dict[str, Any]:
        """Test pattern recognition system."""
        print("\n🔍 Testing Pattern Recognition...")
        try:
            # First create a simple geometry
            cad_result = await self.pipeline.cad_system.analyze_description(
                "Create a simple cube with 10cm sides"
            )
            if cad_result["status"] != "success":
                raise Exception("Failed to create test geometry")
                
            patterns = self.pipeline._analyze_patterns(cad_result["geometry"])
            print("✅ Pattern Recognition test passed")
            return {"success": True, "patterns": patterns}
        except Exception as e:
            print(f"❌ Pattern Recognition test failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def test_manufacturing_analysis(self) -> Dict[str, Any]:
        """Test manufacturing analysis system."""
        print("\n🔍 Testing Manufacturing Analysis...")
        try:
            # First create a simple geometry
            cad_result = await self.pipeline.cad_system.analyze_description(
                "Create a simple cube with 10cm sides"
            )
            if cad_result["status"] != "success":
                raise Exception("Failed to create test geometry")
                
            patterns = self.pipeline._analyze_patterns(cad_result["geometry"])
            analysis = await self.pipeline._analyze_manufacturing(
                cad_result["geometry"],
                patterns
            )
            print("✅ Manufacturing Analysis test passed")
            return {"success": True, "analysis": analysis}
        except Exception as e:
            print(f"❌ Manufacturing Analysis test failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def test_full_pipeline(self) -> Dict[str, Any]:
        """Test the complete pipeline."""
        print("\n🔍 Testing Full Pipeline...")
        try:
            result = await self.pipeline.process(
                "Create a simple cube with 10cm sides"
            )
            print("✅ Full Pipeline test passed")
            return {"success": True, "result": result}
        except Exception as e:
            print(f"❌ Full Pipeline test failed: {str(e)}")
            return {"success": False, "error": str(e)}

async def run_tests():
    """Run all tests."""
    print("🚀 Starting CADroid Pipeline Tests...")
    
    # Initialize test suite
    test_suite = TestPipeline()
    
    # Run tests
    results = {
        "llm_client": await test_suite.test_llm_client(),
        "intelligent_cad": await test_suite.test_intelligent_cad(),
        "pattern_recognition": await test_suite.test_pattern_recognition(),
        "manufacturing_analysis": await test_suite.test_manufacturing_analysis(),
        "full_pipeline": await test_suite.test_full_pipeline()
    }
    
    # Print summary
    print("\n📊 Test Summary:")
    for component, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"{status} {component}: {'Passed' if result['success'] else 'Failed'}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_tests()) 