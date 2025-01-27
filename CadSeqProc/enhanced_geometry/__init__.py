"""
Enhanced geometry module for CAD generation.
Provides support for both geometric primitives and organic shapes.
"""

from .base import Point, GeometricEntity
from .nurbs import NURBSCurve, NURBSSurface
from .organic import OrganicSurface
from ..sequence.transformation.deform import (
    TwistDeformation,
    BendDeformation,
    TaperDeformation
)
from .factory import OrganicShapeFactory
from .integration import (
    GeometryAdapter,
    ShapeGenerator,
    ModelIntegration
)

__all__ = [
    # Base classes
    'Point',
    'GeometricEntity',
    
    # NURBS implementations
    'NURBSCurve',
    'NURBSSurface',
    
    # Organic shape support
    'OrganicSurface',
    'TwistDeformation',
    'BendDeformation',
    'TaperDeformation',
    
    # Factory
    'OrganicShapeFactory',
    
    # Integration
    'GeometryAdapter',
    'ShapeGenerator',
    'ModelIntegration'
]

__version__ = '0.1.0' 