"""Demo script showing usage of the CAD pipeline."""

import os
from pathlib import Path
from pprint import pprint

from CadSeqProc.pipeline import CADPipeline, PipelineConfig

def demo_text_input():
    """Demonstrate pipeline usage with text input."""
    print("\n=== Text Input Demo ===")
    
    # Configure pipeline
    config = PipelineConfig(
        model_type="claude",
        debug=True
    )
    pipeline = CADPipeline(config)
    
    # Process text description
    description = """
    Create a mounting bracket with:
    - Base plate 100x50mm
    - 4 mounting holes (8mm diameter) at corners
    - Reinforcement ribs along length
    - Material suitable for 3D printing
    """
    
    print("Processing description...")
    result = pipeline.process(description)
    
    # Print results
    print("\nResults:")
    print(f"Success: {result['success']}")
    
    if result["success"]:
        print("\nDetected Patterns:")
        for pattern in result["patterns"]:
            print(f"- {pattern['name']}: {pattern['description']}")
            print(f"  Type: {pattern['type']}")
            print(f"  Confidence: {pattern['confidence']:.2f}")
        
        print("\nManufacturing Analysis:")
        mfg = result["manufacturing"]
        print(f"Best Process: {mfg['best_process']}")
        print("\nTop Material Suggestions:")
        for mat in mfg["material_suggestions"]:
            print(f"- {mat['material'].name}")
            print(f"  Score: {mat['score']:.1f}")
            print(f"  Reasons:")
            for reason in mat["reasons"]:
                print(f"    * {reason}")
        
        print("\nOptimization Suggestions:")
        for opt in result["optimizations"]:
            print(f"- [{opt['type']}] {opt['suggestion']}")
        
        # Save results
        output_path = pipeline.save_results(result)
        print(f"\nResults saved to: {output_path}")

def demo_file_input():
    """Demonstrate pipeline usage with file input."""
    print("\n=== File Input Demo ===")
    
    config = PipelineConfig(debug=True)
    pipeline = CADPipeline(config)
    
    # Create example specification file
    specs = {
        "type": "assembly",
        "components": [
            {
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
                    },
                    {
                        "type": "fillet",
                        "radius": 5,
                        "edges": "all"
                    }
                ]
            }
        ]
    }
    
    # Save specifications to file
    specs_file = Path("example_specs.json")
    import json
    with open(specs_file, 'w') as f:
        json.dump(specs, f, indent=2)
    
    print(f"Processing specification file: {specs_file}")
    result = pipeline.process(specs_file)
    
    if result["success"]:
        print("\nProcessing successful!")
        print(f"Generated {len(result['patterns'])} patterns")
        print(f"Found {len(result['optimizations'])} optimization opportunities")
        
        # Save results
        output_path = pipeline.save_results(result)
        print(f"\nResults saved to: {output_path}")
    else:
        print(f"Error: {result['error']}")
    
    # Cleanup
    specs_file.unlink()

def main():
    """Run pipeline demos."""
    print("CAD Pipeline Demonstration")
    print("=" * 50)
    
    # Text input demo
    demo_text_input()
    
    # File input demo
    demo_file_input()
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main() 