"""Demo script for pattern recognition in CAD models."""

import os
import sys
from typing import List, Dict, Any
import numpy as np  # type: ignore

from CadSeqProc.intelligent_cad import IntelligentCAD
from CadSeqProc.llm_client import LLMClient
from CadSeqProc.base import GeometricEntity, Point
from CadSeqProc.enhanced_geometry.pattern_recognition import DesignPattern

def load_example_model() -> GeometricEntity:
    """Load or create an example CAD model with patterns."""
    # Create a simple model with patterns for demonstration
    from CadSeqProc.geometry.circle import Circle
    from CadSeqProc.geometry.line import Line
    
    # Create a base plate
    plate = Circle(center=Point(0, 0, 0), radius=50)
    
    # Add linear array of holes
    holes = []
    for x in range(-30, 31, 10):
        hole = Circle(center=Point(x, 0, 0), radius=2)
        holes.append(hole)
    
    # Add circular array of mounting points
    mounts = []
    radius = 40
    for i in range(8):
        angle = (2 * np.pi * i) / 8
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        mount = Circle(center=Point(x, y, 0), radius=3)
        mounts.append(mount)
    
    # Add some reinforcement features
    reinforcements = []
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        x1 = 20 * np.cos(angle)
        y1 = 20 * np.sin(angle)
        x2 = 35 * np.cos(angle)
        y2 = 35 * np.sin(angle)
        reinforcement = Line(
            start=Point(x1, y1, 0),
            end=Point(x2, y2, 0)
        )
        reinforcements.append(reinforcement)
    
    # Combine all features
    class CombinedGeometry(GeometricEntity):
        def __init__(self, base, holes, mounts, reinforcements):
            self.base = base
            self.holes = holes
            self.mounts = mounts
            self.reinforcements = reinforcements
            self.sub_entities = [base] + holes + mounts + reinforcements
    
    return CombinedGeometry(plate, holes, mounts, reinforcements)

def analyze_patterns(cad: IntelligentCAD, model: GeometricEntity) -> None:
    """Analyze and print patterns found in the model."""
    print("\n=== Pattern Analysis ===")
    
    # Detect patterns
    patterns = cad.analyze_design_patterns(model)
    
    # Print detected patterns
    for i, pattern in enumerate(patterns, 1):
        print(f"\nPattern {i}: {pattern.name}")
        print(f"Description: {pattern.description}")
        
        # Print pattern details
        feature = pattern.features[0]
        print(f"Type: {feature.pattern_type}")
        print(f"Instances: {len(feature.instances)}")
        print(f"Confidence: {feature.confidence:.2f}")
        
        # Print parameters
        print("Parameters:")
        for key, value in feature.parameters.items():
            print(f"  {key}: {value}")
        
        # Print manufacturing notes if available
        if pattern.manufacturing_notes:
            print("\nManufacturing Notes:")
            for key, value in pattern.manufacturing_notes.items():
                print(f"  {key}: {value}")
        
        # Print reuse suggestions if available
        if pattern.reuse_suggestions:
            print("\nReuse Suggestions:")
            for suggestion in pattern.reuse_suggestions:
                print(f"  - {suggestion['suggestion']}")

def suggest_optimizations(cad: IntelligentCAD, patterns: List[DesignPattern]) -> None:
    """Generate and print optimization suggestions."""
    print("\n=== Optimization Suggestions ===")
    
    suggestions = cad.suggest_pattern_optimizations(patterns)
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\nSuggestion {i}:")
        print(f"Type: {suggestion['type']}")
        print(f"Pattern: {suggestion['pattern']}")
        
        if suggestion['type'] == 'manufacturing':
            print("Manufacturing suggestions:")
            for key, value in suggestion['suggestions'].items():
                print(f"  {key}: {value}")
        elif suggestion['type'] == 'reuse':
            print("Reuse suggestions:")
            for s in suggestion['suggestions']:
                print(f"  - {s['suggestion']}")
        elif suggestion['type'] == 'performance':
            print(f"Performance suggestion: {suggestion['suggestion']}")

def main():
    """Main demo function."""
    # Initialize CAD system
    llm_client = LLMClient()  # Configure with your API key if needed
    cad = IntelligentCAD(llm_client)
    
    # Load example model
    print("Loading example model...")
    model = load_example_model()
    
    # Analyze patterns
    print("Analyzing patterns...")
    patterns = cad.analyze_design_patterns(model)
    analyze_patterns(cad, model)
    
    # Generate optimization suggestions
    print("\nGenerating optimization suggestions...")
    suggest_optimizations(cad, patterns)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main() 