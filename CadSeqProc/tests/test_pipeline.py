"""Tests for the CAD pipeline module."""

import unittest
from pathlib import Path
from typing import Dict, Any
import json
import tempfile
import shutil

from CadSeqProc.pipeline import CADPipeline, PipelineConfig
from CadSeqProc.base import GeometricEntity, Point

class MockGeometry(GeometricEntity):
    """Mock geometry for testing."""
    def __init__(self):
        self.center = Point(0, 0, 0)
        self.bounds = (Point(-1, -1, -1), Point(1, 1, 1))

class TestCADPipeline(unittest.TestCase):
    """Test cases for CAD pipeline functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for testing
        self.test_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.test_dir / "cache"
        self.output_dir = self.test_dir / "output"
        
        # Configure pipeline
        self.config = PipelineConfig(
            cache_dir=str(self.cache_dir),
            output_dir=str(self.output_dir),
            debug=True
        )
        self.pipeline = CADPipeline(self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        
    def test_text_input_processing(self):
        """Test processing text description input."""
        input_text = "Create a rectangular bracket with four mounting holes"
        result = self.pipeline.process(input_text)
        
        self.assertTrue(result["success"])
        self.assertIn("geometry", result)
        self.assertIn("patterns", result)
        self.assertIn("manufacturing", result)
        
    def test_dict_input_processing(self):
        """Test processing dictionary specification input."""
        input_specs = {
            "type": "bracket",
            "dimensions": {
                "length": 100,
                "width": 50,
                "thickness": 5
            },
            "features": [
                {
                    "type": "hole",
                    "diameter": 8,
                    "positions": [
                        [-40, -20, 0],
                        [40, -20, 0],
                        [-40, 20, 0],
                        [40, 20, 0]
                    ]
                }
            ]
        }
        
        result = self.pipeline.process(input_specs)
        self.assertTrue(result["success"])
        
    def test_file_input_processing(self):
        """Test processing file input."""
        # Create test files
        test_files = self._create_test_files()
        
        for file_path in test_files:
            result = self.pipeline.process(file_path)
            self.assertTrue(result["success"])
            
    def test_error_handling(self):
        """Test error handling in pipeline."""
        # Test with invalid input
        invalid_input = 123  # Invalid input type
        result = self.pipeline.process(invalid_input)
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("stage", result)
        
    def test_result_saving(self):
        """Test saving pipeline results."""
        # Process simple input
        result = self.pipeline.process("Create a simple cube")
        
        # Save results
        output_path = self.pipeline.save_results(result)
        
        # Verify saved files
        self.assertTrue(output_path.exists())
        self.assertTrue((output_path.parent / "model.step").exists())
        
        # Verify content
        with open(output_path) as f:
            saved_data = json.load(f)
            self.assertEqual(saved_data["success"], result["success"])
            
    def _create_test_files(self) -> list[Path]:
        """Create test files for file input testing."""
        files = []
        
        # Create test JSON
        json_path = self.test_dir / "test.json"
        with open(json_path, 'w') as f:
            json.dump({
                "type": "cube",
                "size": 10
            }, f)
        files.append(json_path)
        
        # Create test image (if PIL is available)
        try:
            from PIL import Image
            import numpy as np
            
            image_path = self.test_dir / "test.png"
            image = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
            image.save(image_path)
            files.append(image_path)
        except ImportError:
            pass
            
        return files
        
if __name__ == '__main__':
    unittest.main() 