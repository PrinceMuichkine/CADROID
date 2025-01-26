# ---------------------------------------------------------------------------- #
#                               Global Variables                               #
# ---------------------------------------------------------------------------- #


N_BIT=8

END_TOKEN=["PADDING", "START", "END_SKETCH",
                "END_FACE", "END_LOOP", "END_CURVE", "END_EXTRUSION"]

END_PAD=7
BOOLEAN_PAD=4

MAX_CAD_SEQUENCE_LENGTH=272

SKETCH_TOKEN = ["PADDING", "START", "END_SKETCH",
                "END_FACE", "END_LOOP", "END_CURVE", "CURVE"]
EXTRUSION_TOKEN = ["PADDING", "START", "END_EXTRUDE_SKETCH"]

CURVE_TYPE=["Line","Arc","Circle"]

EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
                      "CutFeatureOperation", "IntersectFeatureOperation"]


NORM_FACTOR=0.75
EXTRUDE_R=1
SKETCH_R=1

PRECISION = 1e-5
eps = 1e-7


MAX_SKETCH_SEQ_LENGTH = 150
MAX_EXTRUSION = 10
ONE_EXT_SEQ_LENGTH = 10  # Without including start/stop and pad token ((e1,e2),ox,oy,oz,theta,phi,gamma,b,s,END_EXTRUSION) -> 10
VEC_TYPE=2 # Different types of vector representation (Keep only 2)


CAD_CLASS_INFO = {
    'one_hot_size': END_PAD+BOOLEAN_PAD+2**N_BIT,
    'index_size': MAX_EXTRUSION+1, # +1 for padding
    'flag_size': ONE_EXT_SEQ_LENGTH+2 # +2 for sketch and padding
}


"""
Macro utility for recording and replaying operations.
"""

from typing import List, Dict, Any, Optional, Callable
import json
import time
from .logger import setup_logger

logger = setup_logger(__name__)

class MacroRecorder:
    """Record and replay sequences of operations."""
    
    def __init__(self):
        """Initialize macro recorder."""
        self.operations: List[Dict[str, Any]] = []
        self.is_recording = False
        self.handlers: Dict[str, Callable] = {}
    
    def start_recording(self) -> None:
        """Start recording operations."""
        self.operations = []
        self.is_recording = True
        logger.info("Started recording macro")
    
    def stop_recording(self) -> None:
        """Stop recording operations."""
        self.is_recording = False
        logger.info(f"Stopped recording macro with {len(self.operations)} operations")
    
    def record_operation(self, operation_type: str, **params: Any) -> None:
        """Record an operation."""
        if not self.is_recording:
            return
        
        operation = {
            'type': operation_type,
            'params': params,
            'timestamp': time.time()
        }
        
        self.operations.append(operation)
        logger.debug(f"Recorded operation: {operation_type}")
    
    def register_handler(self, operation_type: str,
                        handler: Callable[..., Any]) -> None:
        """Register handler for operation type."""
        self.handlers[operation_type] = handler
        logger.debug(f"Registered handler for: {operation_type}")
    
    def replay(self, speed: float = 1.0) -> None:
        """Replay recorded operations."""
        if not self.operations:
            logger.warning("No operations to replay")
            return
        
        logger.info(f"Replaying {len(self.operations)} operations")
        
        last_time = self.operations[0]['timestamp']
        for operation in self.operations:
            # Calculate delay
            if speed > 0:
                delay = (operation['timestamp'] - last_time) / speed
                if delay > 0:
                    time.sleep(delay)
            
            # Execute operation
            try:
                self._execute_operation(operation)
            except Exception as e:
                logger.error(f"Error replaying operation: {str(e)}")
                raise
            
            last_time = operation['timestamp']
    
    def _execute_operation(self, operation: Dict[str, Any]) -> None:
        """Execute a single operation."""
        operation_type = operation['type']
        params = operation['params']
        
        if operation_type not in self.handlers:
            raise ValueError(f"No handler registered for: {operation_type}")
        
        handler = self.handlers[operation_type]
        handler(**params)
        logger.debug(f"Executed operation: {operation_type}")
    
    def save_to_file(self, filename: str) -> None:
        """Save recorded operations to file."""
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'operations': self.operations,
                    'version': '1.0'
                }, f, indent=2)
            logger.info(f"Saved macro to: {filename}")
        except Exception as e:
            logger.error(f"Error saving macro: {str(e)}")
            raise
    
    def load_from_file(self, filename: str) -> None:
        """Load operations from file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            if 'version' not in data or data['version'] != '1.0':
                raise ValueError("Invalid macro file version")
                
            self.operations = data['operations']
            logger.info(f"Loaded {len(self.operations)} operations from: {filename}")
        except Exception as e:
            logger.error(f"Error loading macro: {str(e)}")
            raise
    
    def clear(self) -> None:
        """Clear recorded operations."""
        self.operations = []
        logger.debug("Cleared macro operations")
    
    def get_operation_count(self) -> int:
        """Get number of recorded operations."""
        return len(self.operations)
    
    def get_total_time(self) -> float:
        """Get total time of recorded operations."""
        if not self.operations:
            return 0.0
        
        start_time = self.operations[0]['timestamp']
        end_time = self.operations[-1]['timestamp']
        return end_time - start_time
    
    def get_operation_types(self) -> List[str]:
        """Get list of unique operation types."""
        return list(set(op['type'] for op in self.operations))

class MacroBuilder:
    """Builder for creating macros programmatically."""
    
    def __init__(self):
        """Initialize macro builder."""
        self.recorder = MacroRecorder()
        self.recorder.start_recording()
    
    def add_operation(self, operation_type: str, **params: Any) -> 'MacroBuilder':
        """Add operation to macro."""
        self.recorder.record_operation(operation_type, **params)
        return self
    
    def add_delay(self, seconds: float) -> 'MacroBuilder':
        """Add delay between operations."""
        if self.recorder.operations:
            last_time = self.recorder.operations[-1]['timestamp']
            self.recorder.operations[-1]['timestamp'] = last_time + seconds
        return self
    
    def build(self) -> MacroRecorder:
        """Build and return macro recorder."""
        self.recorder.stop_recording()
        return self.recorder

