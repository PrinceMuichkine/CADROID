"""Demo script for manufacturing analysis capabilities."""

import os
import sys
from typing import Dict, Any
import json

from CadSeqProc.base import Point, GeometricEntity
from CadSeqProc.geometry.circle import Circle
from CadSeqProc.geometry.line import Line
from CadSeqProc.manufacturing.manufacturing_analyzer import ManufacturingAnalyzer

def create_example_part() -> GeometricEntity:
    """Create an example part for manufacturing analysis."""
    # Create a simple bracket with features that might cause manufacturing issues
    class Bracket(GeometricEntity):
        def __init__(self):
            # Base plate
            self.base = Circle(center=Point(0, 0, 0), radius=50)
            
            # Mounting holes (potentially too small for CNC)
            self.holes = [
                Circle(center=Point(x, y, 0), radius=0.4)
                for x, y in [(-40, -40), (40, -40), (-40, 40), (40, 40)]
            ]
            
            # Thin walls (potentially problematic for 3D printing)
            self.walls = [
                Line(
                    start=Point(-45, y, 0),
                    end=Point(45, y, 0),
                    thickness=0.5  # Too thin
                )
                for y in [-30, 0, 30]
            ]
            
            # Steep overhangs (problematic for 3D printing)
            self.supports = [
                Line(
                    start=Point(x, -20, 0),
                    end=Point(x, -20, 30),
                    angle=75  # Too steep
                )
                for x in [-20, 0, 20]
            ]
            
            # Combine all features
            self.sub_entities = [self.base] + self.holes + self.walls + self.supports
            
    return Bracket()

def analyze_for_3d_printing(analyzer: ManufacturingAnalyzer, part: GeometricEntity) -> None:
    """Analyze part for 3D printing manufacturing."""
    print("\n=== 3D Printing Analysis ===")
    
    # Analyze manufacturability
    result = analyzer.analyze_manufacturability(part, "3d_printing_fdm")
    
    # Print overall score
    print(f"\nManufacturability Score: {result['manufacturability_score']:.1f}/100")
    
    # Print violations
    if result["constraint_violations"]:
        print("\nManufacturing Issues:")
        for violation in result["constraint_violations"]:
            severity_marker = "❌" if violation["severity"] == "error" else "⚠️"
            print(f"{severity_marker} {violation['message']}")
    else:
        print("\n✅ No manufacturing issues found")
    
    # Print cost analysis
    cost = result["cost_analysis"]
    print("\nCost Analysis:")
    print(f"Material Cost: ${cost['material_cost']:.2f}")
    print(f"Time Cost: ${cost['time_cost']:.2f}")
    print(f"Setup Cost: ${cost['setup_cost']:.2f}")
    print(f"Total Cost: ${cost['total_cost']:.2f}")
    print(f"Estimated Time: {cost['estimated_time']:.1f} hours")
    
    # Print recommendations
    if result["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"{i}. {rec}")

def analyze_for_cnc(analyzer: ManufacturingAnalyzer, part: GeometricEntity) -> None:
    """Analyze part for CNC manufacturing."""
    print("\n=== CNC Machining Analysis ===")
    
    # Analyze manufacturability
    result = analyzer.analyze_manufacturability(part, "cnc_milling")
    
    # Print overall score
    print(f"\nManufacturability Score: {result['manufacturability_score']:.1f}/100")
    
    # Print violations
    if result["constraint_violations"]:
        print("\nManufacturing Issues:")
        for violation in result["constraint_violations"]:
            severity_marker = "❌" if violation["severity"] == "error" else "⚠️"
            print(f"{severity_marker} {violation['message']}")
    else:
        print("\n✅ No manufacturing issues found")
    
    # Print cost analysis
    cost = result["cost_analysis"]
    print("\nCost Analysis:")
    print(f"Material Cost: ${cost['material_cost']:.2f}")
    print(f"Time Cost: ${cost['time_cost']:.2f}")
    print(f"Setup Cost: ${cost['setup_cost']:.2f}")
    print(f"Total Cost: ${cost['total_cost']:.2f}")
    print(f"Estimated Time: {cost['estimated_time']:.1f} hours")
    
    # Print recommendations
    if result["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"{i}. {rec}")

def suggest_materials(analyzer: ManufacturingAnalyzer, part: GeometricEntity) -> None:
    """Suggest materials for the part."""
    print("\n=== Material Suggestions ===")
    
    # Define requirements
    requirements = {
        "min_strength": 45.0,  # MPa
        "max_temp": 100.0,     # °C
        "max_cost": 30.0       # $/kg
    }
    
    print("\nRequirements:")
    print(f"Minimum Strength: {requirements['min_strength']} MPa")
    print(f"Maximum Temperature: {requirements['max_temp']}°C")
    print(f"Maximum Cost: ${requirements['max_cost']}/kg")
    
    # Get suggestions for 3D printing
    print("\nFor 3D Printing:")
    suggestions = analyzer.suggest_material(part, "3d_printing_fdm", requirements)
    
    for i, suggestion in enumerate(suggestions, 1):
        material = suggestion["material"]
        print(f"\nOption {i}: {material.name}")
        print(f"Score: {suggestion['score']:.1f}/100")
        print("Reasons:")
        for reason in suggestion["reasons"]:
            print(f"  - {reason}")
    
    # Get suggestions for CNC
    print("\nFor CNC Machining:")
    suggestions = analyzer.suggest_material(part, "cnc_milling", requirements)
    
    for i, suggestion in enumerate(suggestions, 1):
        material = suggestion["material"]
        print(f"\nOption {i}: {material.name}")
        print(f"Score: {suggestion['score']:.1f}/100")
        print("Reasons:")
        for reason in suggestion["reasons"]:
            print(f"  - {reason}")

def main():
    """Main demo function."""
    # Initialize analyzer
    analyzer = ManufacturingAnalyzer()
    
    # Create example part
    print("Creating example part...")
    part = create_example_part()
    
    # Analyze for different manufacturing processes
    analyze_for_3d_printing(analyzer, part)
    analyze_for_cnc(analyzer, part)
    
    # Get material suggestions
    suggest_materials(analyzer, part)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main() 