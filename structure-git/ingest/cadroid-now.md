Directory structure:
└── princemuichkine-cadroid/
    ├── README.md
    ├── LICENSE
    ├── environment.yml
    ├── py.typed
    ├── .env.sample
    ├── App/
    │   └── app.py
    ├── CadSeqProc/
    │   ├── README.md
    │   ├── cad_sequence.py
    │   ├── eda.py
    │   ├── integration.py
    │   ├── intelligent_cad.py
    │   ├── json2step.py
    │   ├── json2stl_skt3d.py
    │   ├── json2vec.py
    │   ├── llm_client.py
    │   ├── merge_vlm_minimal.py
    │   ├── minimal_cad_json.py
    │   ├── split_json.py
    │   ├── test_recon_step.py
    │   ├── test_recon_stl.py
    │   ├── OCCUtils/
    │   │   ├── Common.py
    │   │   ├── Construct.py
    │   │   ├── Image.py
    │   │   ├── Iteration.py
    │   │   ├── Topology.py
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── edge.py
    │   │   ├── face.py
    │   │   ├── shell.py
    │   │   ├── solid.py
    │   │   ├── types_lut.py
    │   │   ├── vertex.py
    │   │   └── wire.py
    │   ├── enhanced_geometry/
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── factory.py
    │   │   ├── integration.py
    │   │   ├── intelligent_cad.py
    │   │   ├── llm_client.py
    │   │   ├── nurbs.py
    │   │   ├── organic.py
    │   │   ├── parametric.py
    │   │   ├── pattern_recognition.py
    │   │   └── tests.py
    │   ├── examples/
    │   │   └── pattern_recognition_demo.py
    │   ├── geometry/
    │   │   ├── arc.py
    │   │   ├── circle.py
    │   │   ├── curve.py
    │   │   ├── line.py
    │   │   ├── nurbs.py
    │   │   └── organic.py
    │   ├── sequence/
    │   │   ├── sketch/
    │   │   │   ├── coord_system.py
    │   │   │   ├── face.py
    │   │   │   ├── loop.py
    │   │   │   └── sketchsequence.py
    │   │   └── transformation/
    │   │       ├── deform.py
    │   │       └── extrude_sequence.py
    │   ├── tests/
    │   │   └── test_pattern_recognition.py
    │   └── utility/
    │       ├── decorator.py
    │       ├── factory.py
    │       ├── logger.py
    │       ├── macro.py
    │       ├── shape_factory.py
    │       └── utils.py
    ├── Cad_VLM/
    │   ├── test.py
    │   ├── test_user_input.py
    │   ├── train.py
    │   ├── config/
    │   │   ├── inference.yaml
    │   │   ├── inference_user_input.yaml
    │   │   └── trainer.yaml
    │   ├── dataprep/
    │   │   └── t2c_dataset.py
    │   └── models/
    │       ├── decoder.py
    │       ├── loss.py
    │       ├── metrics.py
    │       ├── text2cad.py
    │       ├── utils.py
    │       └── layers/
    │           ├── __init__.py
    │           ├── adaptive_layer.py
    │           ├── attention.py
    │           ├── decorator.py
    │           ├── embedder.py
    │           ├── functional.py
    │           ├── layer_utils.py
    │           ├── text_embed.py
    │           └── utils_decode.py
    ├── Evaluation/
    │   └── eval_seq.py
    └── structure-git/
        ├── cad3dify.md
        ├── cadroid-now.md
        └── text-to-cad.md

================================================
File: README.md
================================================
# CADroid

An advanced text-to-CAD engine that generates precise 3D CAD models from natural language descriptions.

## Overview

CADroid is built on top of Text2CAD and extends its capabilities to provide:
- Enhanced text understanding for technical specifications
- Improved CAD generation with parametric design support
- Advanced quality validation and manufacturability checks
- Interactive parameter adjustment and real-time preview

## Features

### Core Functionality
- Text-to-CAD generation using state-of-the-art language models
- Support for complex CAD operations (extrude, boolean operations)
- Precise dimensional control
- Web interface using Gradio

### Enhanced Text Understanding
- [ ] Advanced prompt engineering for technical specifications
- [ ] Support for engineering terminology and standards
- [ ] Dimensional constraint parsing
- [ ] Material specification handling

### Improved CAD Generation
- [ ] Expanded primitive shape library
- [ ] Parametric design support
- [ ] Advanced constraint handling
- [ ] Enhanced precision control
- [ ] Assembly operations

### Quality & Validation
- [ ] Real-time geometry validation
- [ ] Manufacturability analysis
- [ ] Dimension verification
- [ ] Material compatibility checks
- [ ] Cost estimation

## Installation

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate cadroid

# Install additional dependencies
pip install -r requirements.txt
```

## Usage

### Web Interface
```bash
cd App
python app.py
```

### Command Line
```bash
# Generate CAD model from text
python generate.py --prompt "A rectangular prism with a cylindrical hole through the center"

# Generate with specific parameters
python generate.py --prompt "A 10cm x 5cm x 2cm rectangular plate with four 5mm diameter holes" --precision high
```

## Project Structure
```
cadroid/
├── App/                    # Web interface
├── CadSeqProc/            # CAD processing core
├── models/                # ML models
├── utils/                 # Utility functions
└── tests/                 # Test suite
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on [Text2CAD](https://github.com/SadilKhan/text2cad)
- Uses OpenCascade via pythonOCC
- Gradio for web interface

## Roadmap

### Phase 1: Core Enhancement
- [ ] Implement advanced prompt engineering
- [ ] Add support for technical specifications
- [ ] Improve primitive shape generation

### Phase 2: Advanced Features
- [ ] Add parametric design support
- [ ] Implement constraint system
- [ ] Add assembly operations

### Phase 3: Quality & Validation
- [ ] Add geometry validation
- [ ] Implement manufacturability checks
- [ ] Add cost estimation

### Phase 4: User Experience
- [ ] Add interactive parameter adjustment
- [ ] Implement real-time preview
- [ ] Add export to multiple CAD formats 

================================================
File: LICENSE
================================================
MIT License

Copyright (c) 2024 @princemuichkine | psychoroid.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

================================================
File: environment.yml
================================================
name: cadroid
channels:
    - conda-forge
    - pytorch
    - defaults

dependencies:
    - python=3.9
    - numpy
    - pytorch::pytorch
    - pytorch::torchvision
    - pythonocc-core=7.7.2
    - pip
    - pip:
        - tqdm
        - prettytable
        - torchinfo==1.8.0
        - plyfile==0.9
        - trimesh==4.1.8
        - tensorboard
        - pyvista
        - rich==13.7.1
        - CythonGSL
        - loguru
        - torchdata==0.7.1
        - litdata
        - transformers
        - nltk
        - ipykernel
        - ipywidgets
        - rich
        - pillow
        - accelerate
        - python-dotenv
        - gradio
        - open3d

================================================
File: .env.sample
================================================
# Anthropic
ANTHROPIC_API_KEY='your-key-baby'

# Model Config
MODEL_TEMPERATURE=0.7
MODEL_MAX_TOKENS=4096

# Output Configuration
OUTPUT_DIR=output
CACHE_DIR=cache 

================================================
File: App/app.py
================================================
import os, sys
import dotenv

# Load environment variables
dotenv.load_dotenv()

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from Cad_VLM.models.text2cad import Text2CAD
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT
from CadSeqProc.cad_sequence import CADSequence
from Cad_VLM.models.utils import get_device, get_device_str
from CadSeqProc.enhanced_geometry.intelligent_cad import IntelligentCAD, ConversationalCAD
from CadSeqProc.enhanced_geometry.llm_client import LLMClient, LLMConfig
import gradio as gr
import yaml
import torch

# Global state for conversation
conversation_state = None

def initialize_conversation():
    global conversation_state
    if conversation_state is None:
        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
        llm_config = LLMConfig(
            api_key=api_key,
            model="claude-3-sonnet-20240229",
            temperature=0.7,  # Lower for more precise CAD descriptions
            max_tokens=4096
        )
        llm_client = LLMClient(llm_config)
        intelligent_cad = IntelligentCAD(llm_client)
        conversation_state = ConversationalCAD(llm_client, intelligent_cad)

def load_model(config, device):
    try:
        # -------------------------------- Load Model -------------------------------- #
        cad_config = config["cad_decoder"]
        cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
        text2cad = Text2CAD(text_config=config["text_encoder"], cad_config=cad_config).to(
            device
        )

        if config["test"]["checkpoint_path"] is not None:
            checkpoint_file = config["test"]["checkpoint_path"]
            
            if not os.path.exists(checkpoint_file):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
                
            # Load checkpoint with device-agnostic code
            checkpoint = torch.load(checkpoint_file, map_location=device)
            pretrained_dict = {}
            for key, value in checkpoint["model_state_dict"].items():
                if key.split(".")[0] == "module":
                    pretrained_dict[".".join(key.split(".")[1:])] = value
                else:
                    pretrained_dict[key] = value

            text2cad.load_state_dict(pretrained_dict, strict=False)
        text2cad.eval()
        return text2cad
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def test_model(model, text, config, device):
    if not isinstance(text, list):
        text = [text]
    
    try:
        # Move model to device if not already there
        model = model.to(device)
        
        # Use device-agnostic code
        pred_cad_seq_dict = model.test_decode(
            texts=text,
            maxlen=MAX_CAD_SEQUENCE_LENGTH,
            nucleus_prob=config["test"]["nucleus_prob"],
            topk_index=1,
            device=device
        )
        
        # Move tensors to CPU before numpy conversion
        cad_vec = pred_cad_seq_dict["cad_vec"][0].detach().cpu().numpy()
        
        pred_cad = CADSequence.from_vec(
            cad_vec,
            bit=N_BIT,
            post_processing=True,
        ).create_mesh()

        return pred_cad.mesh, pred_cad
    except Exception as e:
        print(f"Error in test_model: {str(e)}")
        return None, None  # Return tuple to match expected unpacking

def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

# Set up device using our utility function
device = get_device()
print(f"Using {get_device_str()} device")

config_path = "../Cad_VLM/config/inference_user_input.yaml"
config = parse_config_file(config_path)
model = load_model(config, device)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_cad_model_from_text(text, chat_history):
    global model, config, conversation_state
    
    # Initialize conversation if needed
    initialize_conversation()
    
    # Process the request through conversational CAD
    result = conversation_state.process_request(text)
    
    if result["success"]:
        # Use the generated/modified text description
        description = result["text"]
        
        # Convert to mesh using Text2CAD
        mesh, *extra = test_model(model=model, text=description, config=config, device=device)
        if mesh is not None:
            output_path = os.path.join(OUTPUT_DIR, "output.stl")
            mesh.export(output_path)
            
            # Update chat history with more detailed information
            message = f"Generated CAD model from: {description}"
            if "updates" in result:
                # Add parameter update information
                updates = result["updates"]
                message += "\nUpdated parameters:"
                for update in updates:
                    message += f"\n- {update['name']}: {update['value']}"
            
            chat_history.append((text, message))
            
            return output_path, chat_history
    
    # Handle error case
    chat_history.append((text, f"Error: {result['message']}"))
    return None, chat_history

examples = [
    "A ring.",
    "A rectangular prism.",
    "A 3D star shape with 5 points.",
    "A cylindrical object with a cylindrical hole in the center.",
    "A rectangular metal plate with four holes along its length.",
    "Make the holes larger",
    "Add another hole",
    "undo",
    "redo"
]

title = "Text2CAD: Generating Sequential CAD Designs from Text Prompts"
description = """
Generate 3D CAD models from text descriptions. This demo runs on CPU/MPS for Mac compatibility.

Features:
- Create basic shapes and complex assemblies
- Modify existing models through conversation
- Update parameters (dimensions, features, etc.)
- Undo/redo support

Commands:
- "undo": Revert last change
- "redo": Reapply undone change
- Modify parameters: "make the holes larger", "increase the width to 10mm"
- Add features: "add another hole", "create a pattern of 5 holes"

<div style="display: flex; justify-content: center; gap: 10px; align-items: center;">
<a href="https://arxiv.org/abs/2409.17106">
  <img src="https://img.shields.io/badge/Arxiv-3498db?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=2c3e50&borderRadius=10" alt="Arxiv" />
</a>
<a href="https://sadilkhan.github.io/text2cad-project/">
  <img src="https://img.shields.io/badge/Project-2ecc71?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=27ae60&borderRadius=10" alt="Project" />
</a>
</div>
"""

def check_requirements():
    required_paths = [
        "../Cad_VLM/config/inference_user_input.yaml",
        # Add path to checkpoint file here
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

check_requirements()

# Create the Gradio interface with chat and history display
demo = gr.Interface(
    fn=generate_cad_model_from_text,
    inputs=[
        gr.Textbox(label="Text", placeholder="Enter a text description or modification (e.g., 'make it larger', 'undo')"),
        gr.State([])  # For chat history
    ],
    outputs=[
        gr.Model3D(clear_color=[0.678, 0.847, 0.902, 1.0], label="3D CAD Model"),
        gr.Chatbot(label="Conversation History")  # Added chat history display
    ],
    examples=examples,
    title=title,
    description=description,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=True)


================================================
File: CadSeqProc/README.md
================================================

This folder contains codes for processing the DeepCAD Jsons.


### 1. Json to Vector Representation

```python
import json
from cad_sequence import CADSequence
from utility.macro import MAX_CAD_SEQ_LEN, N_BIT

json_path="./data/cad_json/0000/00000007.json"
with open(json_path,"r") as f:
    json_data=json.load(f)

cad_seq, cad_vec, flag_vec, index_vec=CADSequence.json_to_vec(data=json_data,bit=N_BIT,padding=True,max_cad_seq_len=MAX_CAD_SEQ_LEN)
```

### 2. Generate Mesh/Brep/Point Cloud from Json/Vec

_NOTE: Use `CADSequence.from_vec(vec, denumericalize=True)` to create the class if your input is a vector representation._

_WARNING: Do not generate the CAD Model while the class contains quantized parameters. Use the method `.denumericalize(bit=N_BIT)` during that case._

```python
import json
from cad_sequence import CADSequence
from utility.macro import MAX_CAD_SEQ_LEN, N_BIT

json_path="./data/cad_json/0000/00000007.json"
with open(json_path,"r") as f:
    json_data=json.load(f)

cad_seq=CADSequence.json_to_NormalizedCAD(data=json_data, bit=N_BIT)

# <!-- ------------------------ Generate Brep or Mesh ------------------------ -->
# NOTE: It will be saved in os.path.join(output_dir,filename+".step")

brep=cad_seq.save_stp(filename=filename, output_dir=output_dir, type="step") # type="stl" for mesh

# <!-- ------------------------ Generate Point Cloud ------------------------- -->
# NOTE: filename without .ply
brep=cad_seq.save_points(filename=filename, output_dir=output_dir, n_points=8192, pointype="uniform")

```


### 3. To Check the parameters of the CAD Model

```python
print(cad_seq)
```

```python
# Output
CAD Sequence:

    - Sketch:
       - CoordinateSystem:
            - Rotation Matrix [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            - Translation [0. 0. 0.]
       - Face:
          - Loop: Start Point: [0.0, 0.0], Direction: collinear
              - Circle: center([0.3765 0.3765]),             radius(0.3765), pt1 [0.37647059 0.75      ]


    - ExtrudeSequence: (profile_uids: [['FQgWGf8WhgalpUy', 'JGC']], extent_one: 0.1015625, extent_two: 0.0, boolean: 0, sketch_size: 0.75) Euler Angles [0. 0. 0.]

```

================================================
File: CadSeqProc/eda.py
================================================
import os, sys

sys.path.append(os.path.dirname(__file__))
sys.path.append("..")

import json
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from CadSeqProc.utility.utils import get_files_scan
from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.decorator import measure_performance
from tqdm import tqdm
import argparse

clg_logger = CLGLogger().configure_logger().logger


# ---------------------------------------------------------------------------- #
#                 Exploratory Data Analysis for DeepCAD Dataset                #
# ---------------------------------------------------------------------------- #

# NOTE: This code is used to perform exploratory data analysis on the DeepCAD dataset. 
# It returds a dataframe with number of sketches, faces, loops, curves and extrusions 
# for each CAD object in the dataset.

def process_one(json_path):

    with open(json_path, "r") as f:
        data = json.load(f)

    uid = "/".join(json_path.split("/")[-2:]).strip(".json")

    try:
        cad_obj = CADSequence.from_dict(data)
    except Exception as e:
        clg_logger.info(f"Json: {uid}. Error: {e}")
        return None
    new_df = pd.DataFrame(
        {
            "uid": uid,
            "sketch": [len(cad_obj.sketch_seq)],
            "face": [len(cad_obj.all_faces)],
            "loop": [len(cad_obj.all_loops)],
            "curve": [len(cad_obj.all_curves)],
            "extrusion": [len(cad_obj.extrude_seq)],
        }
    )

    return new_df

@measure_performance
def process_json(args):
    all_json_files = get_files_scan(args.json_dir, max_workers=args.max_workers)

    df = pd.DataFrame()

    # ----------------------- Process the json Sequentially ---------------------- #
    # for json_path in tqdm(all_json_files):
    #     new_df = process_one(json_path)
    #     if new_df is not None:
    #         df = pd.concat([df, new_df], ignore_index=True)

    # ---------------- Process the Json using Parallel processing ---------------- #
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_one, json_path)
            for json_path in tqdm(all_json_files, desc="Submitting Tasks")
        ]
        for future in tqdm(as_completed(futures), desc="Processing Files"):
            val = future.result()  # complexity is number of curves
            if val is not None:
                df = pd.concat([df, val], ignore_index=True)

    # ----------------------- Save the results ---------------------- #
    df.to_csv(os.path.join(args.output_dir, "analysis.csv"), index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis")
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    process_json(args)


if __name__ == "__main__":
    main()


================================================
File: CadSeqProc/integration.py
================================================
"""
Integration module to connect enhanced geometry system with existing CAD model.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from .geometry.nurbs import NURBSCurve, NURBSSurface
from .geometry.organic import OrganicSurface
from .utility.shape_factory import OrganicShapeFactory
from .sequence.transformation.deform import TwistDeformation, BendDeformation, TaperDeformation

class GeometryAdapter:
    """Adapter to convert between enhanced geometry and CAD model formats."""
    
    @staticmethod
    def to_cad_sequence(
        entities: List[Union[NURBSCurve, NURBSSurface, OrganicSurface]]
    ) -> Dict[str, Any]:
        """Convert geometric entities to CAD sequence format."""
        sequence = {
            'type': 'composite',
            'operations': []
        }
        
        for entity in entities:
            if isinstance(entity, NURBSCurve):
                points = entity.sample_points(20)
                sequence['operations'].append({
                    'type': 'curve',
                    'points': [[p[0], p[1], p[2]] for p in points],
                    'closed': False
                })
            elif isinstance(entity, NURBSSurface):
                points_2d = entity.sample_points(20, 20)
                sequence['operations'].append({
                    'type': 'surface',
                    'points': [[[p[0], p[1], p[2]] for p in row] for row in points_2d],
                    'closed_u': True,
                    'closed_v': True
                })
            elif isinstance(entity, OrganicSurface):
                for surface in entity.control_surfaces:
                    points_2d = surface.sample_points(20, 20)
                    sequence['operations'].append({
                        'type': 'surface',
                        'points': [[[p[0], p[1], p[2]] for p in row] for row in points_2d],
                        'closed_u': True,
                        'closed_v': True
                    })
        
        return sequence
    
    @staticmethod
    def from_cad_sequence(sequence: Dict[str, Any]) -> List[Union[NURBSCurve, NURBSSurface]]:
        """Convert CAD sequence back to geometric entities."""
        entities = []
        
        for op in sequence.get('operations', []):
            if op['type'] == 'curve':
                points = [(p[0], p[1], p[2]) for p in op['points']]
                entities.append(NURBSCurve.from_points(points))
            elif op['type'] == 'surface':
                points_2d = [[(p[0], p[1], p[2]) for p in row] for row in op['points']]
                entities.append(NURBSSurface.from_points(points_2d))
        
        return entities

class ShapeGenerator:
    """High-level interface for generating shapes from text descriptions."""
    
    def __init__(self):
        self.factory = OrganicShapeFactory()
    
    def parse_flower_description(self, text: str) -> Dict[str, Any]:
        """Parse text description for flower parameters."""
        params = {
            'num_petals': 5,
            'petal_length': 1.0,
            'petal_width': 0.3,
            'center_radius': 0.2
        }
        
        # Extract number of petals
        if 'many petals' in text.lower():
            params['num_petals'] = 8
        elif 'few petals' in text.lower():
            params['num_petals'] = 3
        
        # Adjust size
        if any(word in text.lower() for word in ['large', 'big']):
            params['petal_length'] *= 1.5
            params['petal_width'] *= 1.5
            params['center_radius'] *= 1.5
        elif any(word in text.lower() for word in ['small', 'tiny']):
            params['petal_length'] *= 0.7
            params['petal_width'] *= 0.7
            params['center_radius'] *= 0.7
        
        return params
    
    def create_from_text(self, text: str) -> Dict[str, Any]:
        """Generate shapes based on text description."""
        try:
            shapes = []
            
            if any(word in text.lower() for word in ['flower', 'petal', 'bloom']):
                params = self.parse_flower_description(text)
                shapes.extend(self.factory.create_flower(**params))
            elif any(word in text.lower() for word in ['leaf', 'foliage']):
                shapes.append(self.factory.create_leaf())
            elif any(word in text.lower() for word in ['tree', 'plant']):
                shapes.extend(self.factory.create_tree())
            elif any(word in text.lower() for word in ['vine', 'creeper']):
                # Create a simple curved path for the vine
                control_points = [
                    (0, 0, 0),
                    (0.5, 0.5, 0.2),
                    (1.0, 0, 0.5),
                    (1.5, -0.5, 0.3),
                    (2.0, 0, 0)
                ]
                shapes.extend(self.factory.create_vine(control_points))
            else:
                raise ValueError("Unsupported shape type in description")
            
            return {
                'status': 'success',
                'shapes': shapes
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

class ModelIntegration:
    """Integration with the main CAD model."""
    
    def __init__(self):
        self.adapter = GeometryAdapter()
        self.generator = ShapeGenerator()
    
    def process_text_input(self, text: str) -> Dict[str, Any]:
        """Process text input to generate CAD model."""
        try:
            # Generate shapes from text
            result = self.generator.create_from_text(text)
            
            if result['status'] == 'error':
                return {
                    'status': 'error',
                    'message': result['message']
                }
            
            # Convert shapes to CAD sequence
            sequence = self.adapter.to_cad_sequence(result['shapes'])
            
            return {
                'status': 'success',
                'cad_sequence': sequence,
                'metadata': {
                    'input_text': text,
                    'num_shapes': len(result['shapes'])
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def validate_sequence(self, sequence: Dict[str, Any]) -> bool:
        """Validate CAD sequence before processing."""
        if not isinstance(sequence, dict):
            return False
        
        if 'type' not in sequence or sequence['type'] != 'composite':
            return False
        
        if 'operations' not in sequence or not isinstance(sequence['operations'], list):
            return False
        
        for op in sequence['operations']:
            if 'type' not in op:
                return False
            
            if op['type'] not in ['curve', 'surface']:
                return False
            
            if op['type'] == 'curve':
                if 'points' not in op or not isinstance(op['points'], list):
                    return False
            elif op['type'] == 'surface':
                if 'points' not in op or not isinstance(op['points'], list):
                    return False
                if not all(isinstance(row, list) for row in op['points']):
                    return False
        
        return True 

================================================
File: CadSeqProc/intelligent_cad.py
================================================
""" Intelligent CAD module leveraging LLM capabilities for enhanced generation. """

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from .base import Point, GeometricEntity
from .nurbs import NURBSCurve, NURBSSurface
from .organic import OrganicSurface
from .factory import OrganicShapeFactory
import json
from .enhanced_geometry.pattern_recognition import PatternRecognizer, DesignPattern

@dataclass
class PartMetadata:
    """Metadata about a part including manufacturing and assembly info."""
    name: str
    description: str
    material: Optional[str] = None
    manufacturing_process: Optional[str] = None
    tolerances: Dict[str, float] = field(default_factory=dict)
    assembly_constraints: List[Dict] = field(default_factory=list)

@dataclass
class CADContext:
    """Track conversation context and CAD state."""
    current_geometry: Optional[GeometricEntity] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "dimensions": {},  # Store dimensional parameters
        "features": {},    # Store feature parameters
        "constraints": {}, # Store geometric constraints
        "patterns": {}     # Store pattern information
    })
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    undo_stack: List[Dict[str, Any]] = field(default_factory=list)
    redo_stack: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.parameters is None:
            self.parameters = {
                "dimensions": {},  # Store dimensional parameters
                "features": {},    # Store feature parameters
                "constraints": {}, # Store geometric constraints
                "patterns": {}     # Store pattern information
            }
        if self.modifications is None:
            self.modifications = []
        if self.undo_stack is None:
            self.undo_stack = []
        if self.redo_stack is None:
            self.redo_stack = []

    def update_parameter(self, category: str, name: str, value: Any):
        """Update a parameter value and track the change."""
        if category not in self.parameters:
            self.parameters[category] = {}
            
        old_value = self.parameters[category].get(name)
        self.parameters[category][name] = value
            
        # Track change for undo/redo
        self.undo_stack.append({
            "type": "parameter_update",
            "category": category,
            "name": name,
            "old_value": old_value,
            "new_value": value
        })
        self.redo_stack.clear()  # Clear redo stack on new change

    def undo(self) -> Optional[Dict[str, Any]]:
        """Undo last change and return the change info."""
        if not self.undo_stack:
            return None
            
        change = self.undo_stack.pop()
        self.redo_stack.append(change)
            
        if change["type"] == "parameter_update":
            self.parameters[change["category"]][change["name"]] = change["old_value"]
            
        return change

    def redo(self) -> Optional[Dict[str, Any]]:
        """Redo last undone change and return the change info."""
        if not self.redo_stack:
            return None
            
        change = self.redo_stack.pop()
        self.undo_stack.append(change)
            
        if change["type"] == "parameter_update":
            self.parameters[change["category"]][change["name"]] = change["new_value"]
            
        return change

class IntelligentCAD:
    """CAD system enhanced with LLM capabilities."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.feature_recognizer = FeatureRecognizer(llm_client)
        self.assembly_planner = AssemblyPlanner(llm_client)
        self.manufacturing_validator = ManufacturingValidator(llm_client)
        self.pattern_recognizer = PatternRecognizer(llm_client)
        self.context = CADContext()

    def analyze_description(self, description: str) -> Dict[str, Any]:
        """
        Use LLM to analyze part description and extract key information.
        """
        # First, check if this is an organic shape request
        organic_analysis = self._analyze_organic_shape(description)
        if organic_analysis.get("is_organic"):
            return self._handle_organic_shape(organic_analysis, description)
            
        # Otherwise, proceed with normal geometric analysis
        prompt = f"""
        Analyze this CAD part/assembly description: "{description}"
            
        Extract and provide:
        1. Key geometric parameters and their values
        2. Material requirements
        3. Manufacturing constraints
        4. Assembly relationships
        5. Critical features and their purposes
        6. Suggested tolerances
            
        Format as structured JSON with clear parameter hierarchies.
        Include engineering rationale for key decisions.
        """
            
        analysis = self.llm_client.analyze(prompt)
        return self._validate_analysis(analysis)

    def _analyze_organic_shape(self, description: str) -> Dict[str, Any]:
        """Analyze if the request is for an organic shape and how to decompose it."""
        prompt = f"""
        Analyze if this describes an organic shape: "{description}"
            
        If it is organic, provide:
        1. Basic geometric primitives that can approximate the shape
        2. Mathematical curves/surfaces needed (e.g., Bezier, NURBS)
        3. Decomposition into simpler parts
        4. Key dimensions and proportions
        5. Symmetry patterns
            
        Return as JSON:
        {{
            "is_organic": boolean,
            "decomposition": [
                {{
                    "part": "part_name",
                    "primitive_type": "curve|surface|solid",
                    "parameters": {{...}},
                    "mathematical_description": "...",
                    "relative_position": {{...}}
                }}
            ],
            "symmetry": {{
                "type": "radial|bilateral|none",
                "count": number  # for radial symmetry
            }},
            "key_dimensions": {{...}}
        }}
        """
            
        return self.llm_client.analyze(prompt)

    def _handle_organic_shape(self, analysis: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Generate CAD description for organic shape using geometric approximation."""
        # Convert organic shape analysis into geometric operations
        prompt = f"""
        Convert this organic shape analysis into a CAD description:
        {json.dumps(analysis, indent=2)}
            
        Original description: "{description}"
            
        Generate a complete CAD description that:
        1. Uses basic geometric primitives to approximate organic forms
        2. Preserves key proportions and relationships
        3. Implements symmetry patterns
        4. Creates smooth transitions between parts
            
        Format as a single descriptive sentence suitable for Text2CAD model input.
        Focus on geometric terms that the model understands.
        """
            
        geometric_description = self.llm_client.generate(prompt)
            
        return {
            "description": geometric_description,
            "organic_analysis": analysis,
            "geometric_parameters": self._extract_parameters(analysis)
        }

    def _extract_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geometric parameters from organic shape analysis."""
        parameters: Dict[str, Dict[str, Any]] = {
            "dimensions": {},
            "features": {},
            "constraints": {},
            "patterns": {}
        }
            
        # Extract dimensions
        if "key_dimensions" in analysis:
            parameters["dimensions"].update(analysis["key_dimensions"])
            
        # Extract symmetry patterns
        if "symmetry" in analysis:
            parameters["patterns"]["symmetry"] = analysis["symmetry"]
            
        # Extract part features
        if "decomposition" in analysis:
            for part in analysis["decomposition"]:
                parameters["features"][part["part"]] = {
                    "type": part["primitive_type"],
                    "parameters": part["parameters"]
                }
            
        return parameters

    def apply_modifications(self, current_geometry: GeometricEntity, modifications: List[Dict]) -> str:
        """
        Instead of modifying geometry directly, generate a new text description
        that incorporates the modifications.
        """
        # Get current features
        features = self.feature_recognizer.analyze_features(current_geometry)
            
        # Create prompt to generate new description
        prompt = f"""
        Current CAD model features:
        {json.dumps(features, indent=2)}
            
        Requested modifications:
        {json.dumps(modifications, indent=2)}
            
        Generate a complete text description that preserves existing features
        and incorporates the requested modifications. Format as a single
        descriptive sentence suitable for Text2CAD model input.
        """
            
        new_description = self.llm_client.generate(prompt)
        return new_description

    def execute_sequence(self, sequence: List[Dict]) -> str:
        """
        Convert operation sequence to text description for Text2CAD.
        """
        prompt = f"""
        Convert this CAD operation sequence to a natural language description:
        {json.dumps(sequence, indent=2)}
            
        Generate a single descriptive sentence that captures all geometric
        features and relationships. Format suitable for Text2CAD model input.
        """
            
        description = self.llm_client.generate(prompt)
        return description

    def _validate_analysis(self, analysis: Dict) -> Dict:
        """Validate and normalize analyzed parameters."""
        required_fields = ["geometric_parameters", "features", "constraints"]
        for field in required_fields:
            if field not in analysis:
                analysis[field] = {}
        return analysis

    def generate_part_sequence(self, metadata: PartMetadata) -> List[Dict]:
        """
        Generate CAD sequence for part creation.
            
        Args:
            metadata: Part metadata from analysis
            
        Returns:
            List of CAD operations to create part
        """
        prompt = f"""
        Given this part specification:
        Name: {metadata.name}
        Description: {metadata.description}
        Material: {metadata.material}
        Manufacturing: {metadata.manufacturing_process}
            
        Generate a detailed CAD sequence considering:
        1. Optimal construction order
        2. Manufacturing constraints
        3. Feature relationships
        4. Required tolerances
            
        Format as sequence of geometric operations.
        """
            
        sequence = self.llm_client.generate(prompt)
        return self._validate_sequence(sequence)

    def _validate_sequence(self, sequence: List[Dict]) -> List[Dict]:
        """Validate generated CAD sequence."""
        # Implement validation logic
        return sequence

    def suggest_improvements(self, part: GeometricEntity, context: str) -> List[Dict]:
        """
        Suggest improvements for part design.
            
        Args:
            part: Current part geometry
            context: Usage context and requirements
            
        Returns:
            List of suggested improvements
        """
        prompt = f"""
        Analyze this part design in context: "{context}"
            
        Suggest improvements considering:
        1. Structural integrity
        2. Manufacturing efficiency
        3. Assembly optimization
        4. Material usage
        5. Cost reduction
            
        Provide specific geometric modifications.
        """
            
        suggestions = self.llm_client.analyze(prompt)
        return self._validate_suggestions(suggestions)

    def _validate_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Validate improvement suggestions."""
        # Implement validation logic
        return suggestions

    def analyze_design_patterns(self, geometry: GeometricEntity) -> List[DesignPattern]:
        """Analyze geometry for design patterns and optimization opportunities."""
        patterns = self.pattern_recognizer.analyze_geometry(geometry)
        
        # Update context with pattern information
        self.context.parameters["patterns"].update({
            pattern.name: {
                "type": pattern.features[0].pattern_type,
                "instances": len(pattern.features[0].instances),
                "confidence": pattern.features[0].confidence
            } for pattern in patterns
        })
        
        return patterns

    def suggest_pattern_optimizations(self, patterns: List[DesignPattern]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on recognized patterns."""
        suggestions = []
        
        for pattern in patterns:
            # Manufacturing optimization
            if pattern.manufacturing_notes:
                suggestions.append({
                    "type": "manufacturing",
                    "pattern": pattern.name,
                    "suggestions": pattern.manufacturing_notes
                })
                
            # Reuse opportunities
            if pattern.reuse_suggestions:
                suggestions.append({
                    "type": "reuse",
                    "pattern": pattern.name,
                    "suggestions": pattern.reuse_suggestions
                })
                
            # Performance optimization
            if pattern.features[0].pattern_type in ["linear_array", "circular_array"]:
                suggestions.append({
                    "type": "performance",
                    "pattern": pattern.name,
                    "suggestion": "Consider using native CAD pattern features for better performance"
                })
        
        return suggestions

    def apply_pattern_optimization(self, geometry: GeometricEntity, 
                                 optimization: Dict[str, Any]) -> GeometricEntity:
        """Apply a pattern-based optimization to the geometry."""
        if optimization["type"] == "manufacturing":
            # Apply manufacturing optimizations
            return self._apply_manufacturing_optimization(geometry, optimization)
        elif optimization["type"] == "reuse":
            # Apply pattern reuse
            return self._apply_pattern_reuse(geometry, optimization)
        elif optimization["type"] == "performance":
            # Apply performance optimizations
            return self._apply_performance_optimization(geometry, optimization)
        
        return geometry

    def _apply_manufacturing_optimization(self, geometry: GeometricEntity, 
                                       optimization: Dict[str, Any]) -> GeometricEntity:
        """Apply manufacturing-specific optimizations."""
        # Implementation for manufacturing optimization
        return geometry

    def _apply_pattern_reuse(self, geometry: GeometricEntity, 
                            optimization: Dict[str, Any]) -> GeometricEntity:
        """Apply pattern reuse optimizations."""
        # Implementation for pattern reuse
        return geometry

    def _apply_performance_optimization(self, geometry: GeometricEntity, 
                                      optimization: Dict[str, Any]) -> GeometricEntity:
        """Apply performance optimizations."""
        # Implementation for performance optimization
        return geometry

class FeatureRecognizer:
    """Recognize and classify CAD features using LLM."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def analyze_features(self, geometry: GeometricEntity) -> Dict[str, Any]:
        """
        Analyze geometric features and their purposes.
            
        Args:
            geometry: Part geometry to analyze
            
        Returns:
            Dictionary of features and their classifications
        """
        # Convert geometry to analyzable format
        geo_desc = self._geometry_to_description(geometry)
            
        prompt = f"""
        Analyze these geometric features:
        {geo_desc}
            
        Identify and classify:
        1. Primary features (holes, bosses, etc.)
        2. Manufacturing features
        3. Assembly features
        4. Structural features
            
        Explain the purpose of each feature.
        """
            
        return self.llm_client.analyze(prompt)

    def _geometry_to_description(self, geometry: GeometricEntity) -> str:
        """Convert geometry to textual description for LLM."""
        # Implement conversion logic
        return "Geometric description"

class AssemblyPlanner:
    """Plan and validate assembly sequences using LLM."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan_assembly(self, parts: List[GeometricEntity],
                     constraints: List[Dict]) -> Dict[str, Any]:
        """
        Plan assembly sequence and validate constraints.
            
        Args:
            parts: List of parts to assemble
            constraints: Assembly constraints
            
        Returns:
            Assembly plan and validation results
        """
        # Convert parts and constraints to analyzable format
        assembly_desc = self._create_assembly_description(parts, constraints)
            
        prompt = f"""
        Plan assembly sequence for:
        {assembly_desc}
            
        Provide:
        1. Optimal assembly order
        2. Required fixtures/tooling
        3. Critical alignments
        4. Tolerance stack-up analysis
        5. Potential interference issues
            
        Consider manufacturing and assembly constraints.
        """
            
        return self.llm_client.analyze(prompt)

    def _create_assembly_description(self, parts: List[GeometricEntity],
                                  constraints: List[Dict]) -> str:
        """Create assembly description for LLM."""
        # Implement conversion logic
        return "Assembly description"

class ManufacturingValidator:
    """Validate manufacturing feasibility using LLM."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def validate_design(self, part: GeometricEntity,
                      process: str) -> Dict[str, Any]:
        """
        Validate design for manufacturing process.
            
        Args:
            part: Part geometry
            process: Manufacturing process
            
        Returns:
            Validation results and suggestions
        """
        # Convert part to analyzable format
        design_desc = self._create_design_description(part)
            
        prompt = f"""
        Validate this design for {process} manufacturing:
        {design_desc}
            
        Check for:
        1. Process-specific constraints
        2. Feature manufacturability
        3. Required tolerances
        4. Cost implications
        5. Potential optimizations
            
        Suggest specific improvements.
        """
            
        return self.llm_client.analyze(prompt)

    def _create_design_description(self, part: GeometricEntity) -> str:
        """Create design description for LLM."""
        # Implement conversion logic
        return "Design description"

class ConversationalCAD:
    """Handles multi-turn CAD generation with context."""
    
    def __init__(self, llm_client, intelligent_cad):
        self.llm = llm_client
        self.cad = intelligent_cad
        self.context = CADContext()
        self.current_description = None

    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process a user request in context of previous interactions."""
        # Add request to history
        self.context.history.append({
            "user_input": user_input,
            "timestamp": np.datetime64('now')
        })
            
        # First, check if this is a command (undo/redo)
        if user_input.lower().strip() == "undo":
            return self._handle_undo()
        elif user_input.lower().strip() == "redo":
            return self._handle_redo()
            
        # Then check if it's a parameter modification
        param_update = self._check_parameter_update(user_input)
        if param_update:
            return self._handle_parameter_update(param_update)
            
        # Finally, check if it's a geometric modification
        is_modification = self._is_modification_request(user_input)
        if is_modification and self.current_description:
            return self._handle_modification(user_input)
        else:
            return self._handle_new_request(user_input)

    def _check_parameter_update(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Check if the request is to update parameters."""
        prompt = f"""
        Analyze if this request modifies parameters:
        "{user_input}"
            
        Current parameters:
        {json.dumps(self.context.parameters, indent=2)}
            
        If this is a parameter update, return JSON:
        {{
            "is_parameter_update": true,
            "updates": [
                {{
                    "category": "dimensions|features|constraints|patterns",
                    "name": "parameter_name",
                    "value": new_value,
                    "unit": "unit_if_applicable"
                }}
            ]
        }}
        Otherwise return: {{"is_parameter_update": false}}
        """
            
        analysis = self.llm.analyze(prompt)
        if analysis.get("is_parameter_update"):
            return analysis
        return None

    def _handle_parameter_update(self, update_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle parameter update request."""
        try:
            for update in update_info["updates"]:
                self.context.update_parameter(
                    update["category"],
                    update["name"],
                    update["value"]
                )
                
            # Generate new description with updated parameters
            prompt = f"""
            Current model: "{self.current_description}"
            Updated parameters:
            {json.dumps(update_info["updates"], indent=2)}
                
            Generate a new complete description that:
            1. Incorporates the updated parameters
            2. Preserves other existing features
            3. Is formatted as a single clear sentence
            4. Is suitable for Text2CAD model input
            """
                
            new_description = self.llm.generate(prompt)
            self.current_description = new_description
                
            return {
                "text": new_description,
                "success": True,
                "message": "Parameters updated successfully",
                "updates": update_info["updates"]
            }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error updating parameters: {str(e)}"
            }

    def _handle_undo(self) -> Dict[str, Any]:
        """Handle undo request."""
        change = self.context.undo()
        if not change:
            return {
                "success": False,
                "message": "Nothing to undo"
            }
            
        # Generate new description based on undone change
        prompt = f"""
        Current model: "{self.current_description}"
        Undone change: {json.dumps(change, indent=2)}
            
        Generate a new complete description that:
        1. Reflects the undone change
        2. Preserves other features
        3. Is formatted as a single clear sentence
        4. Is suitable for Text2CAD model input
        """
            
        try:
            new_description = self.llm.generate(prompt)
            self.current_description = new_description
                
            return {
                "text": new_description,
                "success": True,
                "message": f"Undid {change['type']}"
            }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error applying undo: {str(e)}"
            }

    def _handle_redo(self) -> Dict[str, Any]:
        """Handle redo request."""
        change = self.context.redo()
        if not change:
            return {
                "success": False,
                "message": "Nothing to redo"
            }
            
        # Generate new description based on redone change
        prompt = f"""
        Current model: "{self.current_description}"
        Redone change: {json.dumps(change, indent=2)}
            
        Generate a new complete description that:
        1. Reflects the redone change
        2. Preserves other features
        3. Is formatted as a single clear sentence
        4. Is suitable for Text2CAD model input
        """
            
        try:
            new_description = self.llm.generate(prompt)
            self.current_description = new_description
                
            return {
                "text": new_description,
                "success": True,
                "message": f"Redid {change['type']}"
            }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error applying redo: {str(e)}"
            }

    def _is_modification_request(self, user_input: str) -> bool:
        """Determine if request is a modification to existing geometry."""
        if not self.current_description:
            return False
            
        # Use LLM to analyze if request is a modification
        prompt = f"""
        Previous CAD model description: "{self.current_description}"
        New request: "{user_input}"
            
        Analyze if this is a modification request by checking:
        1. If it references the existing model
        2. If it suggests changes to existing features
        3. If it adds/removes/modifies parts
            
        Return JSON: {{"is_modification": boolean, "modification_type": string}}
        """
            
        analysis = self.llm.analyze(prompt)
        return analysis.get("is_modification", False)

    def _handle_modification(self, user_input: str) -> Dict[str, Any]:
        """Handle modification by generating new complete description."""
        prompt = f"""
        Current CAD model: "{self.current_description}"
        Requested change: "{user_input}"
            
        Generate a new complete description that:
        1. Preserves existing features
        2. Incorporates requested changes
        3. Is formatted as a single clear sentence
        4. Is suitable for Text2CAD model input
            
        Return the new description only.
        """
            
        try:
            # Generate new complete description
            new_description = self.llm.generate(prompt)
                
            # Update context
            self.current_description = new_description
            self.context.modifications.append({
                "request": user_input,
                "new_description": new_description
            })
                
            return {
                "text": new_description,
                "success": True,
                "message": "Description updated successfully"
            }
                
        except Exception as e:
            return {
                "text": self.current_description,
                "success": False,
                "message": f"Error updating description: {str(e)}"
            }

    def _handle_new_request(self, user_input: str) -> Dict[str, Any]:
        """Handle request for new geometry."""
        # Reset context for new request
        self.context = CADContext()
        self.current_description = user_input
            
        try:
            # Analyze and validate the description
            metadata = self.cad.analyze_description(user_input)
                
            return {
                "text": user_input,
                "success": True,
                "message": "New description processed successfully"
            }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing description: {str(e)}"
            } 

================================================
File: CadSeqProc/json2step.py
================================================
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")

# Adding Python Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

from tqdm import tqdm
from CadSeqProc.utility.decorator import measure_performance
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import get_files_scan
import argparse
import multiprocessing
import json
from CadSeqProc.cad_sequence import CADSequence
import warnings
import gc

warnings.filterwarnings("ignore")

multiprocessing.set_start_method("forkserver", force=True)
clglogger = CLGLogger().configure_logger().logger

# ---------------------------------------------------------------------------- #
#                           DeepCAD Json to Brep/Mesh                          #
# ---------------------------------------------------------------------------- #


@measure_performance
def main():
    """
    Parse Json into sketch and extrusion sequence tokens
    """
    parser = argparse.ArgumentParser(
        description="Creating Sketch and Extrusion Sequence"
    )
    parser.add_argument(
        "-p", "--input_dir", help="Input Directory for DeepCAD Json dataset", type=str
    )
    parser.add_argument(
        "--split_json", help="Input Directory for DeepCAD split json", type=str
    )
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepcad",
        choices=["deepcad", "fusion360", "cad_parser"],
    )
    parser.add_argument("--bit", type=int, default=N_BIT)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--save_type", type=str, default="step")
    args = parser.parse_args()

    # print(args)
    clglogger.info(f"Running Task with {args.max_workers} workers.")

    if args.dataset == "deepcad":
        process_deepcad(args, clglogger)
    elif args.dataset == "fusion360":
        process_fusion360(args, clglogger)

    # all_json_files = sorted(get_files_scan(args.input_dir, max_workers=args.max_workers))


def process_fusion360(args, clglogger):
    """
    Processes the Fusion360 dataset

    Args:
        args (argparse.Namespace): The arguments.
    """
    all_files = get_files_scan(args.input_dir, max_workers=args.max_workers)
    all_json_files = [
        file
        for file in all_files
        if file.endswith(".json") and file.split("/")[-2] == "json"
    ]
    clglogger.info(
        f"Preprocessing {len(all_json_files)} Fusion360 dataset using Method 1."
    )
    process_all_jsons(all_json_files, args, clglogger)
    clglogger.success(f"Task Complete")


def process_deepcad(args, clglogger):
    """
    Processes the DeepCAD dataset

    Args:
        args (argparse.Namespace): The arguments.
    """
    with open(args.split_json, "r") as f:
        data = json.load(f)

    all_json_files = (
        data["train"][:82000]
        + data["train"][84000:]
        + data["test"]
        + data["validation"]
    )

    # --------------------------------- Method 1 --------------------------------- #
    process_all_jsons(all_json_files, args, clglogger)

    extra_json_files = [
        os.path.join(args.input_dir, uid + ".json")
        for uid in data["train"][82000:84000]
    ]

    # --------------------------------- Method 2 --------------------------------- #
    clglogger.info(f"Preprocessing {len(extra_json_files)} using Method 2")
    for json_path in tqdm(all_json_files):
        try:
            process_json(json_path, args)
        except:
            pass

    clglogger.success(f"Task Complete")


def process_all_jsons(all_json_files, args, clglogger):
    """
    Processes all the JSON files in the list and saves the CAD models

    Args:
        all_json_files (list): A list of JSON files.
    """
    # Create a ProcessPoolExecutor
    executor = ThreadPoolExecutor(max_workers=args.max_workers)

    # Submit tasks to the executor
    futures = [
        executor.submit(process_json, json_path, args)
        for json_path in tqdm(all_json_files, desc="Submitting Tasks")
    ]

    # Wait for the tasks to complete
    for future in tqdm(as_completed(futures), desc="Processing Files"):
        future.result()
    
    clglogger.success(f"Method 1 Complete")


def process_json(json_path, args):
    """
    Processes a JSON file and saves the whole CAD model as well as intermediate ones

    Args:
        json_path (str): The path to the JSON file.
    """
    try:
        if args.dataset == "deepcad":
            uid = "/".join(json_path.strip(".json").split("/")[-2:])  # 0003/00003121

        elif args.dataset == "fusion360":
            uid = "/".join(json_path.split("/")[-4:-2])
        name = uid.split("/")[-1]  # 00003121

        # Open the JSON file.
        with open(json_path, "r") as f:
            data = json.load(f)

        # cad_seq = CADSequence.json_to_NormalizedCAD(data=data, bit=args.bit)
        cad_seq = CADSequence.from_dict(all_stat=data)

        # ------------------------- Save the final cad Model ------------------------- #
        cad_seq.save_stp(
            filename=name + "_final",
            output_folder=os.path.join(args.output_dir, uid, args.save_type),
            type=args.save_type,
        )

        # ------------------------ Save the intermediate models ----------------------- #
        num_intermediate_model = len(cad_seq.sketch_seq)
        if num_intermediate_model > 1:
            # -------------------------------- Separate -------------------------------- #

            for i in range(num_intermediate_model):
                new_cad_seq = CADSequence(
                    sketch_seq=[cad_seq.sketch_seq[i]],
                    extrude_seq=[cad_seq.extrude_seq[i]],
                )

                # Make the operation as NewBodyOperation to create a solid body
                new_cad_seq.extrude_seq[0].metadata["boolean"] = 0
                new_cad_seq.save_stp(
                    filename=name + f"_intermediate_{i+1}",
                    output_folder=os.path.join(args.output_dir, uid, args.save_type),
                    type=args.save_type,
                )
                del new_cad_seq
        
        gc.collect()
       
    except Exception as e:
        pass
        clglogger.error(f"Problem processing {json_path}. Error: {e}")


if __name__ == "__main__":
    main()


================================================
File: CadSeqProc/json2stl_skt3d.py
================================================
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from cad_sequence import CADSequence
import argparse
import json
import open3d as o3d
import torch
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
  # ---------------------------------------------------------------------------- #
  #            DeepCAD Original Json or Vec to Mesh + 3D Sketch Points           #
  # ---------------------------------------------------------------------------- #

clglogger=CLGLogger().configure_logger(verbose=True).logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--input_dir", help="Input Directory for DeepCAD Json dataset", type=str
    )
    parser.add_argument("--split_json", help="Train-test-validation split", type=str)
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("--num_sketch_points", type=int, default=30000)
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args()
    
    # print(args)
    clglogger.info(f"Running Task with {args.max_workers} workers.")

    # all_json_uids = sorted(get_files_scan(args.input_dir, max_workers=args.max_workers))

    with open(args.split_json, "r") as f:
        data = json.load(f)

    all_json_uids = (
        data["train"][:82000]
        + data["train"][84000:]
        + data["test"]
        + data["validation"]
    ) # Possible corruption in data causing ProcessPool to fail
    
    all_json_uids=sorted(all_json_uids)

    clglogger.info(f"Found {len(all_json_uids)} files.")
    clglogger.info(f"Saving Sequence in {args.output_dir}.")

    # --------------------------------- Method 1 --------------------------------- #
    executor = ProcessPoolExecutor(max_workers=args.max_workers)
    # Submit tasks to the executor
    futures = [
        executor.submit(
            process_one, os.path.join(args.input_dir, json_uid + ".json"), args
        )
        for json_uid in tqdm(all_json_uids, desc="Submitting Tasks")
    ]

    for future in tqdm(as_completed(futures), desc="Processing Files"):
        future.result()

    # --------------------------------- Method 2 --------------------------------- #
    for json_uid in tqdm(data["train"][:82000] + data["train"][84000:]):
        try:
            process_one(os.path.join(args.input_dir, json_uid + ".json"), args)
        except:
            pass
    

def process_one(file_path,args):
    
    try:
        file_type = file_path.split(".")[-1]
        if file_type == "json":
            uid = "/".join(file_path.split("/")[-2:]).split(".")[0]
            name = uid.split("/")[-1]
            with open(file_path, "r") as f:
                json_data = json.load(f)

            cad_seq = CADSequence.json_to_NormalizedCAD(
                data=json_data, bit=8
            )
        else:
            uid = "/".join(file_path.split("/")[-4:-2])
            name = uid.split("/")[-1]
            cad_vec = torch.load(file_path)['vec']["cad_vec"].numpy()
            cad_seq = CADSequence.from_vec(cad_vec, post_processing=True)

        # Generate Mesh and 3D sketch Points
        cad_seq.create_mesh().sample_sketch_points3D(n_points=args.num_sketch_points, color=True)
        
        # ---------------------------------- Output ---------------------------------- #
        output_dir=os.path.join(args.output_dir, uid, "mesh_skt_3d")
        
        # ----------------------------------- Mesh ----------------------------------- #
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save Mesh using trimesh
        cad_seq.mesh.export(os.path.join(output_dir, f"{name}_mesh.stl"), file_type="stl")

        # -------------------------------- Point Cloud ------------------------------- #
        # Save 3D Sketch Points in open3d
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(cad_seq.sketch_points3D)
        # Set colors for each point
        colors = o3d.utility.Vector3dVector(cad_seq.sketch_points3D_color)
        point_cloud.colors = colors
        
        o3d.io.write_point_cloud(
            os.path.join(output_dir, f"{name}_skt_3d_color.ply"), point_cloud
        )
        # clglogger.success(f"Saved in {output_dir}.")
    except Exception as e:
        clglogger.error(f"Error in {file_path}: {e}")
        # print(traceback.print_exc())


if __name__ == "__main__":
    main()

================================================
File: CadSeqProc/json2vec.py
================================================
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")

# Adding Python Path
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
import pandas as pd
import time
import traceback
from rich import print
from tqdm import tqdm
from CadSeqProc.utility.decorator import measure_performance
from CadSeqProc.utility.logger import CLGLogger
import torch
from loguru import logger
import numpy as np
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import (
    generate_attention_mask,
    ensure_dir,
    hash_map,
    get_files_scan,
)
from cad_sequence import CADSequence
import argparse
import multiprocessing
import json
from CadSeqProc.cad_sequence import CADSequence
import warnings
import shutil

warnings.filterwarnings("ignore")

multiprocessing.set_start_method("forkserver", force=True)

clglogger = CLGLogger().configure_logger().logger

unique_model_hash = []


# ---------------------------------------------------------------------------- #
#                       DeepCAD Json to CAD-SIGNet Vector                      #
# ---------------------------------------------------------------------------- #

# This code is used to convert the DeepCAD Json dataset to CAD-SIGNet vector representation.
# Required for Training the Text2CAD Transformer

@measure_performance
def main():
    """
    Parse Json into sketch and extrusion sequence tokens
    """
    parser = argparse.ArgumentParser(
        description="Creating Sketch and Extrusion Sequence"
    )
    parser.add_argument(
        "-p", "--input_dir", help="Input Directory for DeepCAD Json dataset", type=str
    )
    parser.add_argument("--split_json", help="Train-test-validation split", type=str, default="")
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepcad",
        choices=["deepcad", "fusion360", "cad_parser"],
    )
    parser.add_argument("--bit", type=int, default=N_BIT)
    parser.add_argument(
        "--max_cad_seq_len",
        type=int,
        default=MAX_CAD_SEQUENCE_LENGTH,
        help="Maximum length of cad sequence",
    )
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument(
        "--padding",
        action="store_true",
        help="Add padding in the vector for same token length",
    )
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    # print(args)
    if args.verbose:
        clglogger.info(f"Running Task with {args.max_workers} workers.")

    if args.dataset=="deepcad":
        process_deepcad(args, clglogger)
    
    elif args.dataset=="fusion360":
        process_fusion360(args, clglogger)
    
    elif args.dataset=="cad_parser":
        #process_cad_parser(args, clglogger)
        pass
    
    else:
        raise ValueError(f"Invalid dataset name {args.dataset}.")



def process_deepcad(args, clglogger):
    # Get Json Path
    with open(args.split_json, "r") as f:
        data = json.load(f)

    uidlist = (
        data["train"][:82000]
        + data["train"][84000:]
        + data["test"]
        + data["validation"]
    )  # Possible corruption in data causing ProcessPool to fail

    all_json_files = [os.path.join(args.input_dir, uid + ".json") for uid in uidlist]

    # ------------------------------------------- Method 1 ------------------------------------------- #

    process_all_jsons(all_json_files, args, clglogger)

    # ------------------------------------------- Method 2 ------------------------------------------- #

    # NOTE: METHOD 2 - Sometimes the processpoolexecutor or threadpoolexecutor will fail.
    # As a final choice, run the following then
    extra_json_files = [
        os.path.join(args.input_dir, uid + ".json")
        for uid in data["train"][82000:84000]
    ]

    if args.verbose:
        clglogger.info(f"Preprocessing {len(extra_json_files)} using Method 2")
    for json_path in tqdm(extra_json_files):
        try:
            process_json(json_path, args)
        except:
            pass

    clglogger.success(f"Task Complete")



def process_fusion360(args, clglogger):
    all_files = get_files_scan(args.input_dir,max_workers=args.max_workers)
    all_json_files = [file for file in all_files if file.endswith(".json")]
    process_all_jsons(all_json_files, args, clglogger)



def process_all_jsons(all_json_files, args, clglogger):
    DUPLICATE_MODEL = 0
    executor = ProcessPoolExecutor(max_workers=args.max_workers)
    duplicate_uid = []

    if args.verbose:
        clglogger.info(f"Found {len(all_json_files)} files.")
        clglogger.info(f"Saving Sequence in {args.output_dir}")
        if args.deduplicate:
            clglogger.warning(f"Deduplicate is on. Some duplicate models won't be saved.")

    # NOTE: METHOD 1 - Run the following for faster data processing
    # If it fails, resume from the next iteration

    # Submit tasks to the executor
    futures = [
        executor.submit(process_json, json_path, args)
        for json_path in tqdm(all_json_files, desc="Submitting Tasks")
    ]
    unique_uid = dict()
    for future in tqdm(as_completed(futures), desc="Processing Files", total=len(futures)):
        val, uid, complexity = future.result()  # complexity is number of curves
        DUPLICATE_MODEL += val
        if val == 1:
            duplicate_uid.append(uid)
        if val == 0:
            unique_uid[uid] = complexity

    # sorted_uid = sorted(unique_uid.keys(), key=lambda k: unique_uid[k]) # Sorted according to the number of curves

    complexity_df = pd.DataFrame(
        {"uid": unique_uid.keys(), "complexity": unique_uid.values()}
    )
    complexity_df.to_csv(f"complexity.csv", index=False)

    with open(f"duplicate_uid.txt", "w") as f:
        for item in duplicate_uid:
            f.write(str(item) + "\n")

    # with open(f"sorted_uid_train_test_val.json", "w") as f:
    #     json.dump({args.subset: sorted_uid}, f)

    if args.verbose:
        clglogger.info(f"Total Number of Models {len(all_json_files)}")
        clglogger.info(
            f"Total Number of Invalid Models {DUPLICATE_MODEL} and percentage {DUPLICATE_MODEL/len(all_json_files)}"
        )
        clglogger.info(f"Total Number of Unique Models {len(unique_uid)}")


@logger.catch()
def process_json(json_path, args):
    """
    Processes a JSON file and converts it to a vector representations of sketch and extrusion.

    Args:
        json_path (str): The path to the JSON file.
        bit (int): The bit depth of the vector.
        output_dir (str): The output directory.

    Returns:
        int: The number of sketches in the JSON file.
        int: The number of extrusions in the JSON file.

    """
    try:
        
        if args.dataset=="deepcad":
            uid = "/".join(json_path.strip(".json").split("/")[-2:])  # 0003/00003121
            
        elif args.dataset=="fusion360":
            uid = "/".join(json_path.split("/")[-4:-2])
        name = uid.split("/")[-1]  # 00003121

        # Open the JSON file.
        with open(json_path, "r") as f:
            data = json.load(f)

        # Reading From JSON -> Normalize -> Numericalize -> To Vector Representation
        cad_obj, cad_vec, flag_vec, index_vec = CADSequence.json_to_vec(
            data=data,
            bit=args.bit,
            padding=args.padding,
            max_cad_seq_len=MAX_CAD_SEQUENCE_LENGTH,
        )

        # Check for duplication
        to_save = True
        # Perform hashing for unique models
        if args.deduplicate:
            global unique_model_hash

            param = cad_vec[torch.where(cad_vec >= len(END_TOKEN))[0]].tolist()
            hash_vec = hash_map(param)
            if hash_vec in unique_model_hash:
                to_save = False

        cad_seq_dict = {
            "vec": {
                "cad_vec": cad_vec,
                "flag_vec": flag_vec,
                "index_vec": index_vec,
            },
            "mask_cad_dict": {
                "attn_mask": generate_attention_mask(cad_vec.shape[0] - 1),
                "key_padding_mask": cad_vec == END_TOKEN.index("PADDING"),
            },
        }

        # If save
        if to_save:
            # Save the data in .pth format
            output_dir = os.path.join(args.output_dir, uid, "seq")
            # print(output_dir)
            ensure_dir(output_dir)
            torch.save(cad_seq_dict, os.path.join(output_dir, name + ".pth"))
            if args.verbose:
                clglogger.success(f"Saved in {os.path.join(output_dir, name + '.pth')}")

            return 0, uid, len(cad_obj.all_curves)
        else:
            if args.verbose:
                clglogger.warning(f"Skipping {json_path} because of duplication.")
            return 1, uid, 0
    except Exception as e:
        # print(traceback.print_exc())
        if args.verbose:
            clglogger.error(f"Problem with json path {json_path} with error {e}")
        return 1, uid, 0


if __name__ == "__main__":
    main()


================================================
File: CadSeqProc/llm_client.py
================================================
""" LLM client module for Claude 3.5 integration. Handles communication and response parsing. """

import json
import re
from typing import Dict, Any, List, Optional, Union, TypedDict
import anthropic
from dataclasses import dataclass

class Message(TypedDict):
    role: str
    content: str

@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    api_key: str
    model: str = "claude-3-sonnet-20240229"
    temperature: float = 0.7
    max_tokens: int = 4096

class LLMClient:
    """Client for interacting with Claude 3.5."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)
        self.context: List[Dict[str, Any]] = []

    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Send analysis prompt to Claude and parse response.
            
        Args:
            prompt: Analysis prompt
            
        Returns:
            Parsed response as dictionary
        """
        # Add engineering context
        enriched_prompt = self._enrich_prompt(prompt)
            
        try:
            response = self._call_api(enriched_prompt)
            return self._parse_response(response)
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return {"error": str(e)}

    def generate(self, prompt: str) -> str:
        """
        Generate CAD descriptions or operations.
            
        Args:
            prompt: Generation prompt
            
        Returns:
            Generated text
        """
        # Add CAD-specific context
        enriched_prompt = self._enrich_prompt(prompt, context_type="cad")
            
        try:
            response = self._call_api(enriched_prompt)
            return response.strip()
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            return f"Error: {str(e)}"

    def _enrich_prompt(self, prompt: str, context_type: str = "analysis") -> str:
        """Add context and examples to prompt."""
        if context_type == "analysis":
            context = """
            You are an expert CAD engineer with deep knowledge of:
            - Mechanical engineering principles
            - Manufacturing processes
            - Assembly design
            - GD&T (Geometric Dimensioning and Tolerancing)
            - Material properties
            
            Analyze the following request considering all engineering aspects.
            Provide detailed, structured responses with clear rationale.
            """
        else:  # CAD generation
            context = """
            You are an expert CAD system generating precise geometric operations.
            Consider:
            - Construction sequence
            - Feature dependencies
            - Parametric relationships
            - Manufacturing constraints
            
            Generate detailed, executable CAD operations.
            """
            
        return f"{context}\n\n{prompt}"

    def _call_api(self, prompt: str) -> str:
        """Make API call to Claude."""
        try:
            message = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse analysis response into structured format."""
        try:
            # First try to parse as direct JSON
            try:
                return json.loads(response)
            except:
                # If not direct JSON, try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise ValueError("No JSON found in response")
        except Exception as e:
            return {"error": f"Failed to parse response: {str(e)}"}

class CADPromptGenerator:
    """Generate structured prompts for CAD operations."""
    
    @staticmethod
    def geometric_analysis(description: str) -> str:
        """Generate prompt for geometric analysis."""
        return f"""
        Analyze this geometric description:
        {description}
            
        Extract:
        1. Primary geometric features
        2. Dimensional parameters
        3. Spatial relationships
        4. Symmetry patterns
        5. Construction hierarchy
            
        Format response as JSON with:
        {{
            "features": [
                {{
                    "type": "feature_type",
                    "parameters": {{...}},
                    "relationships": [...]
                }}
            ],
            "parameters": {{
                "name": {{
                    "value": number,
                    "unit": "unit",
                    "constraints": {{...}}
                }}
            }},
            "construction_sequence": [...]
        }}
        """

    @staticmethod
    def manufacturing_analysis(geometry: Dict) -> str:
        """Generate prompt for manufacturing analysis."""
        return f"""
        Analyze manufacturing requirements for:
        {json.dumps(geometry, indent=2)}
            
        Consider:
        1. Material selection
        2. Manufacturing processes
        3. Feature manufacturability
        4. Required tolerances
        5. Cost implications
            
        Format response as JSON with:
        {{
            "material_options": [
                {{
                    "material": "name",
                    "properties": {{...}},
                    "suitability_score": number,
                    "rationale": "..."
                }}
            ],
            "manufacturing": {{
                "primary_process": "...",
                "secondary_processes": [...],
                "critical_features": [
                    {{
                        "feature": "name",
                        "challenges": [...],
                        "solutions": [...]
                    }}
                ]
            }},
            "tolerances": {{...}},
            "cost_analysis": {{...}}
        }}
        """

    @staticmethod
    def assembly_analysis(parts: List[Dict], constraints: List[Dict]) -> str:
        """Generate prompt for assembly analysis."""
        return f"""
        Analyze assembly requirements for:
        Parts: {json.dumps(parts, indent=2)}
        Constraints: {json.dumps(constraints, indent=2)}
            
        Determine:
        1. Assembly sequence
        2. Mating relationships
        3. Tolerance stack-up
        4. Required fixtures
        5. Potential issues
            
        Format response as JSON with:
        {{
            "assembly_sequence": [
                {{
                    "step": number,
                    "parts": [...],
                    "operations": [...],
                    "critical_notes": [...]
                }}
            ],
            "mating_features": [
                {{
                    "part1": "name",
                    "part2": "name",
                    "type": "mate_type",
                    "parameters": {{...}}
                }}
            ],
            "tolerance_analysis": {{...}},
            "fixtures": [...],
            "issues": [
                {{
                    "type": "issue_type",
                    "description": "...",
                    "severity": number,
                    "solutions": [...]
                }}
            ]
        }}
        """

class ResponseValidator:
    """Validate and normalize LLM responses."""
    
    @staticmethod
    def validate_geometric_analysis(response: Dict) -> Dict:
        """Validate geometric analysis response."""
        required_fields = {
            "features",
            "parameters",
            "construction_sequence"
        }
            
        if not all(field in response for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - set(response.keys())}")
            
        # Validate feature structure
        for feature in response["features"]:
            if not all(k in feature for k in ["type", "parameters"]):
                raise ValueError(f"Invalid feature structure: {feature}")
            
        return response

    @staticmethod
    def validate_manufacturing_analysis(response: Dict) -> Dict:
        """Validate manufacturing analysis response."""
        required_fields = {
            "material_options",
            "manufacturing",
            "tolerances",
            "cost_analysis"
        }
            
        if not all(field in response for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - set(response.keys())}")
            
        # Validate manufacturing structure
        mfg = response["manufacturing"]
        if not all(k in mfg for k in ["primary_process", "secondary_processes"]):
            raise ValueError(f"Invalid manufacturing structure: {mfg}")
            
        return response

    @staticmethod
    def validate_assembly_analysis(response: Dict) -> Dict:
        """Validate assembly analysis response."""
        required_fields = {
            "assembly_sequence",
            "mating_features",
            "tolerance_analysis",
            "fixtures",
            "issues"
        }
            
        if not all(field in response for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - set(response.keys())}")
            
        # Validate assembly sequence
        for step in response["assembly_sequence"]:
            if not all(k in step for k in ["step", "parts", "operations"]):
                raise ValueError(f"Invalid assembly step: {step}")
            
        return response 

================================================
File: CadSeqProc/merge_vlm_minimal.py
================================================
import os
import json
import argparse
from tqdm import tqdm
from glob import glob
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

 # ---------------------------------------------------------------------------- #
 #                            Only for Text2CAD v1.1                            #
 # ---------------------------------------------------------------------------- #


def extract_shape_info(input_string):
    # Function to extract content between tags
    def extract_content(tag, text):
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    # Extracting values
    name = extract_content("NAME", input_string)
    description = extract_content("DESCRIPTION", input_string)
    keywords = extract_content("KEYWORDS", input_string)

    # Converting keywords to a list
    if keywords:
        keywords = [keyword.strip() for keyword in keywords.split(",")]

    # Creating the dictionary
    return {
        "name": "" if name is None else name,
        "description": "" if description is None else description,
        "keywords": "" if keywords is None else keywords,
    }


def process_single(root_dir, uid):
    root_id, sample_id = uid.split("/")

    with open(
        os.path.join(root_dir, uid, "minimal_json", f"{sample_id}.json"), "r"
    ) as f:
        data = json.load(f)

    all_vlm_annotations = glob(
        os.path.join(root_dir, uid, "qwen2_vlm_annotation/*_*.json")
    )
    all_vlm_annot_dict = {}
    for vlm_annot in all_vlm_annotations:
        file_name = os.path.basename(vlm_annot)
        key_name = (
            "final"
            if "final" in file_name
            else "part_" + file_name.split("_")[-1].strip(".json")
        )

        with open(vlm_annot, "r") as f:
            vlm_data = json.load(f)
            all_vlm_annot_dict[key_name] = extract_shape_info(vlm_data)

    data["final_name"] = all_vlm_annot_dict["final"]["name"]
    data["final_shape"] = all_vlm_annot_dict["final"]["description"]
    for key, val in data["parts"].items():
        if key not in all_vlm_annot_dict:
            annot_key = "final"
        else:
            annot_key = key
        val["description"]["name"] = all_vlm_annot_dict[annot_key]["name"]
        val["description"]["shape"] = all_vlm_annot_dict[annot_key]["description"]

    return data


def process_uid(uid, root_dir, output_dir):
    try:
        root_id, sample_id = uid.split("/")
        merged_metadata = process_single(root_dir, uid)
        output_path = os.path.join(
            output_dir, uid, "minimal_json", f"{sample_id}_merged_vlm.json"
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(merged_metadata, f, indent=4)
    except Exception as e:
        return f"Error in processing {uid}: {e}"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str
    )
    parser.add_argument(
        "--split_json",
        type=str,
    )
    parser.add_argument(
        "--output_dir", type=str
    )
    parser.add_argument("--max_workers", type=int, default=8)
    
    args = parser.parse_args()

    with open(args.split_json, "r") as f:
        split_json_data = json.load(f)

    all_uids = (
        split_json_data["train"]
        + split_json_data["test"]
        + split_json_data["validation"]
    )

    errors = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_uid, uid, args.input_dir, args.output_dir): uid
            for uid in all_uids
        }

        for future in tqdm(as_completed(futures), desc="Processing", total=len(all_uids)):
            error = future.result()
            if error:
                errors.append(error)

    if errors:
        print("Some errors occurred during processing:")
        for error in errors:
            print(error)


if __name__ == "__main__":
    main()


================================================
File: CadSeqProc/minimal_cad_json.py
================================================
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")

# Adding Python Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from CadSeqProc.utility.decorator import measure_performance
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import ensure_dir, get_files_scan
from cad_sequence import CADSequence
import argparse
import traceback
# import multiprocessing
import json
from CadSeqProc.cad_sequence import CADSequence
import warnings

warnings.filterwarnings("ignore")
# multiprocessing.set_start_method("forkserver", force=True)
clglogger = CLGLogger().configure_logger().logger


# ---------------------------------------------------------------------------- #
#                         DeepCAD Json to Minimal Json                         #
# ---------------------------------------------------------------------------- #


@measure_performance
def main():
    """
    Parse DeepCAD Json into more human-readable json format
    """
    parser = argparse.ArgumentParser(
        description="Creating Sketch and Extrusion Sequence"
    )
    parser.add_argument(
        "-p", "--input_dir", help="Input Directory for DeepCAD Json dataset", type=str
    )
    parser.add_argument("--split_json", help="Train-test-validation split", type=str, default="")
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("--bit", type=int, default=N_BIT)
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepcad",
        choices=["deepcad", "fusion360", "cad_parser"],
    )
    parser.add_argument("--max_workers", type=int, default=32)

    args = parser.parse_args()
    # print(args)
    clglogger.info(f"Running Task with {args.max_workers} workers.")

    if args.dataset == "deepcad":
        process_deepcad(args, clglogger)
    elif args.dataset == "fusion360":
        process_fusion360(args, clglogger)

    clglogger.success(f"Task Complete")


def process_fusion360(args, clglogger):
    """
    Processes the Fusion360 dataset

    Args:
        args (argparse.Namespace): The arguments.
    """
    all_files = get_files_scan(args.input_dir, max_workers=args.max_workers)
    all_json_files = [
        file
        for file in all_files
        if file.endswith(".json") and file.split("/")[-2] == "json"
    ]
    clglogger.info(
        f"Preprocessing {len(all_json_files)} Fusion360 dataset using Method 1."
    )
    process_all_jsons(all_json_files, args, clglogger)


def process_deepcad(args, clglogger):
    """
    Processes the DeepCAD dataset

    Args:
        args (argparse.Namespace): The arguments.
    """
    with open(args.split_json, "r") as f:
        data = json.load(f)

    all_uids = (
        data["train"][:82000]
        + data["train"][84000:]
        + data["test"]
        + data["validation"]
    )

    # --------------------------------- Method 1 --------------------------------- #
    all_json_files=[os.path.join(args.input_dir, uid + ".json") for uid in all_uids]
    process_all_jsons(all_json_files, args, clglogger)

    extra_json_files = [
        os.path.join(args.input_dir, uid + ".json")
        for uid in data["train"][82000:84000]
    ]

    # --------------------------------- Method 2 --------------------------------- #
    clglogger.info(f"Preprocessing {len(extra_json_files)} using Method 2")
    for json_path in tqdm(extra_json_files):
        try:
            process_json(json_path, args)
        except:
            pass



def process_all_jsons(all_json_files, args, clglogger):
    clglogger.info(f"Found {len(all_json_files)} files.")
    clglogger.info(f"Saving Sequence in {args.output_dir}.")

    # --------------------------------- Method 1 --------------------------------- #

    # NOTE: METHOD 1 - Run the following for faster data processing

    executor = ThreadPoolExecutor(max_workers=args.max_workers)
    # If it fails, resume from the next iteration

    # Submit tasks to the executor
    futures = [
        executor.submit(process_json, json_path, args)
        for json_path in tqdm(all_json_files, desc="Submitting Tasks")
    ]

    for future in tqdm(as_completed(futures), desc="Processing Files", total=len(futures)):
        val, _ = future.result()  # complexity is number of curves


def process_json(json_path, args):
    """
    Processes a JSON file and converts it to a vector representations of sketch and extrusion.

    Args:
        json_path (str): The path to the JSON file.
        bit (int): The bit depth of the vector.
        output_dir (str): The output directory.

    Returns:
        int: The number of sketches in the JSON file.
        int: The number of extrusions in the JSON file.

    """
    try:

        if args.dataset == "deepcad":
            uid = "/".join(json_path.strip(".json").split("/")[-2:])  # 0003/00003121

        elif args.dataset == "fusion360":
            uid = "/".join(json_path.split("/")[-4:-2])
        name = uid.split("/")[-1]  # 00003121

        # Open the JSON file.
        with open(json_path, "r") as f:
            data = json.load(f)

        # Reading From JSON -> Normalize -> Numericalize -> To Vector Representation
        cad_seq = CADSequence.json_to_NormalizedCAD(data=data, bit=8)
        cad_metadata = cad_seq._json()

        # Save the data in .pth format
        output_dir = os.path.join(
            args.output_dir, uid, "minimal_json"
        )  # Output Directory
        ensure_dir(output_dir)
        output_name = os.path.join(output_dir, name + ".json")
        # clglogger.debug(f"Saving to {output_name}")

        if os.path.exists(output_name):
            os.remove(output_name)

        with open(output_name, "w") as f:
            json.dump(cad_metadata, f, indent=5)

        return 0, uid

    except Exception as e:
        # print(traceback.print_exc())
        clglogger.error(f"Problem with json path {json_path} with error {e}")
        return 1, uid


if __name__ == "__main__":
    main()


================================================
File: CadSeqProc/split_json.py
================================================
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")

# Adding Python Path
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)

from rich import print
from tqdm import tqdm
from CadSeqProc.utility.decorator import measure_performance
from CadSeqProc.utility.logger import CLGLogger
import torch
from loguru import logger
import numpy as np
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import get_files_scan, ensure_dir
import argparse
import multiprocessing
import json

import warnings

warnings.filterwarnings("ignore")

multiprocessing.set_start_method("forkserver", force=True)

clglogger = CLGLogger().configure_logger().logger

unique_model_hash = []


# ---------------------------------------------------------------------------- #
#           Create a nested Dictionary with all the annotation paths           #
# ---------------------------------------------------------------------------- #


@measure_performance
def main():
    """
    Create a nested Dictionary with all the annotation paths
    """
    parser = argparse.ArgumentParser(
        description="Generate Split JSON for Training, Test, and Validation"
    )
    parser.add_argument(
        "-p", "--mapper_path", help="Input Directory for DeepCAD JSON dataset", type=str
    )
    parser.add_argument("-o", "--output_dir", type=str)

    args = parser.parse_args()

    seq_prompt_pairs = []
    with open(args.mapper_path, "r") as f:
        mapper_data = json.load(f)

    # TODO: Work on this part
    for key, val in mapper_data.items():
        for sub_key, sub_val in val.items():

            uid = f"{key}/{sub_key}"

            if "vlm_annotation" in sub_val:
                num_prompts = len(val["llm_annotation"])
                if num_prompts > 0:
                    all_keys = [sub_val['']] * num_prompts
                    result = list(zip(all_keys, val["llm_annotation"]))
                    seq_prompt_pairs.extend(result)
            else:
                continue


if __name__ == "__main__":
    main()


================================================
File: CadSeqProc/test_recon_step.py
================================================
from cad_sequence import CADSequence
import os
import argparse
import glob
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# ---------------------------------------------------------------------------- #
#                Generate Step file from Predicted CAD Sequence                #
# ---------------------------------------------------------------------------- #


def process_uid(uid, data, output_dir):
    output_folder = os.path.join(output_dir, uid, "step")
    correct_keys = 0
    wrong_keys = 0
    count_saved = 0
    sample_id = uid.split("/")[-1]  # 00000007
    for i in range(1, 5):
        # check if the key exists
        if "level_" + str(i) not in data[uid]:
            wrong_keys += 1
        else:
            correct_keys += 1
        try:
            CADSequence.from_vec(
                data[uid]["level_" + str(i)]["pred_cad_vec"][0], 2, 8, True
            ).save_stp(
                filename=f"{sample_id}_final_level_" + str(i),
                output_folder=output_folder,
                type="step",
            )
            count_saved += 1
        except Exception as e:
            # print(f"Error in {uid} level {i}")
            # print(e)
            continue

    return count_saved, correct_keys, wrong_keys


def save_step(input_path, output_dir, max_workers):
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} records from {input_path}")

    uids = list(data.keys())
    print(f"Loaded {len(uids)} uids")

    total_count_saved = 0
    total_correct_keys = 0
    total_wrong_keys = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_uid, uid, data, output_dir): uid for uid in uids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            count_saved, correct_keys, wrong_keys = future.result()
            total_count_saved += count_saved
            total_correct_keys += correct_keys
            total_wrong_keys += wrong_keys

    print(
        f"{total_count_saved} step files saved in {output_dir} folder out of total {len(uids) * 4} samples"
    )
    print(f"Correct keys: {total_correct_keys}")
    print(f"Wrong keys: {total_wrong_keys}")
    print(f"Total keys: {total_correct_keys + total_wrong_keys}")
    print(
        f"Invalidity Ratio = {1-total_count_saved / (total_correct_keys + total_wrong_keys)}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--input_path", help="Predicted CAD Sequence in pkl format", required=True
    )
    parser.add_argument("--output_dir", help="Output dir", required=True)
    parser.add_argument("--max_workers", help="Number of workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    save_step(
        input_path=args.input_path,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )


================================================
File: CadSeqProc/test_recon_stl.py
================================================
from cad_sequence import CADSequence
import os
import argparse
import glob
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# ---------------------------------------------------------------------------- #
#                   Generate Mesh from Predicted CAD Sequence                  #
# ---------------------------------------------------------------------------- #



def process_uid(uid, data, output_dir):
    output_folder = os.path.join(output_dir, uid, "stl")
    correct_keys = 0
    wrong_keys = 0
    count_saved = 0
    sample_id = uid.split("/")[-1]  # 00000007
    for i in range(1, 5):
        # check if the key exists
        if "level_" + str(i) not in data[uid]:
            wrong_keys += 1
        else:
            correct_keys += 1
        try:
            CADSequence.from_vec(
                data[uid]["level_" + str(i)]["pred_cad_vec"][0], 2, 8, True
            ).save_stp(
                filename=f"{sample_id}_final_level_" + str(i),
                output_folder=output_folder,
                type="stl",
            )
            count_saved += 1
        except Exception as e:
            # print(f"Error in {uid} level {i}")
            # print(e)
            continue

    return count_saved, correct_keys, wrong_keys


def save_stl(input_path, output_dir, max_workers):
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} records from {input_path}")

    uids = list(data.keys())
    print(f"Loaded {len(uids)} uids")

    total_count_saved = 0
    total_correct_keys = 0
    total_wrong_keys = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_uid, uid, data, output_dir): uid for uid in uids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            count_saved, correct_keys, wrong_keys = future.result()
            total_count_saved += count_saved
            total_correct_keys += correct_keys
            total_wrong_keys += wrong_keys

    print(
        f"{total_count_saved} stl files saved in {output_dir} folder out of total {len(uids) * 4} samples"
    )
    print(f"Correct keys: {total_correct_keys}")
    print(f"Wrong keys: {total_wrong_keys}")
    print(f"Total keys: {total_correct_keys + total_wrong_keys}")
    print(
        f"Invalidity Ratio = {1-total_count_saved / (total_correct_keys + total_wrong_keys)}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--input_path", help="Predicted CAD Sequence in pkl format", required=True
    )
    parser.add_argument("--output_dir", help="Output dir", required=True)
    parser.add_argument("--max_workers", help="Number of workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    save_stl(
        input_path=args.input_path,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )


================================================
File: CadSeqProc/OCCUtils/Common.py
================================================
#! /usr/bin/python

##Copyright 2008-2015 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>.

import random

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.TColgp import (
    TColgp_HArray1OfPnt,
    TColgp_Array1OfPnt,
    TColgp_Array1OfPnt2d,
    TColgp_Array1OfVec,
)
from OCC.Core.TColStd import TColStd_HArray1OfBoolean
from OCC.Core.BRepAdaptor import (
    BRepAdaptor_Curve,
    BRepAdaptor_Curve,
    BRepAdaptor_CompCurve,
    BRepAdaptor_CompCurve,
)
from OCC.Core.GeomAPI import (
    GeomAPI_Interpolate,
    GeomAPI_PointsToBSpline,
    GeomAPI_ProjectPointOnCurve,
)
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Shape, TopoDS_Wire, TopoDS_Vertex
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_C1, GeomAbs_C2, GeomAbs_C3
from OCC.Core.BRepGProp import (
    brepgprop_LinearProperties,
    brepgprop_SurfaceProperties,
    brepgprop_VolumeProperties,
)
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.Geom import Geom_Curve

from OCC.Core import Graphic3d

# ===========================================================================
# No PythonOCC dependencies...
# ===========================================================================


class assert_isdone(object):
    """
    raises an assertion error when IsDone() returns false, with the error
    specified in error_statement
    """

    def __init__(self, to_check, error_statement):
        self.to_check = to_check
        self.error_statement = error_statement

    def __enter__(
        self,
    ):
        if self.to_check.IsDone():
            pass
        else:
            raise AssertionError(self.error_statement)

    def __exit__(self, assertion_type, value, traceback):
        pass


def roundlist(li, n_decimals=3):
    return [round(i, n_decimals) for i in li]


# ===========================================================================
# CONSTANTS
# ===========================================================================

TOLERANCE = 1e-6


def get_boundingbox(shape, tol=TOLERANCE):
    """
    :param shape: TopoDS_Shape such as TopoDS_Face
    :param tol: tolerance
    :return: xmin, ymin, zmin, xmax, ymax, zmax
    """
    bbox = Bnd_Box()
    bbox.SetGap(tol)
    brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return xmin, ymin, zmin, xmax, ymax, zmax


def smooth_pnts(pnts):
    smooth = [pnts[0]]
    for i in range(1, len(pnts) - 1):
        prev = pnts[i - 1]
        this = pnts[i]
        next_pnt = pnts[i + 1]
        pt = (prev + this + next_pnt) / 3.0
        smooth.append(pt)
    smooth.append(pnts[-1])
    return smooth


# ===========================================================================
# Data type utilities
# ===========================================================================


def color(r, g, b):
    return Quantity_Color(r, g, b, Quantity_TOC_RGB)


def to_string(_string):
    from OCC.Core.TCollection import TCollection_ExtendedString

    return TCollection_ExtendedString(_string)


def to_tcol_(_list, collection_type):
    array = collection_type(1, len(_list) + 1)
    for n, i in enumerate(_list):
        array.SetValue(n + 1, i)
    return array


def _Tcol_dim_1(li, _type):
    """function factory for 1-dimensional TCol* types"""
    pts = _type(0, len(li) - 1)
    for n, i in enumerate(li):
        pts.SetValue(n, i)
    pts.thisown = False
    return pts


def point_list_to_TColgp_Array1OfPnt(li):
    pts = TColgp_Array1OfPnt(0, len(li) - 1)
    for n, i in enumerate(li):
        pts.SetValue(n, i)
    return pts


def point2d_list_to_TColgp_Array1OfPnt2d(li):
    return _Tcol_dim_1(li, TColgp_Array1OfPnt2d)


# ===========================================================================
# --- INTERPOLATION ---
# ===========================================================================


def filter_points_by_distance(list_of_point, distance=0.1):
    """
    get rid of those point that lie within tolerance of a
    consequtive series of points
    """
    tmp = [list_of_point[0]]
    for a in list_of_point[1:]:
        if any([a.IsEqual(i, distance) for i in tmp]):
            continue
        else:
            tmp.append(a)
    return tmp


def points_to_bspline(pnts):
    """
    Points to bspline
    """
    pnts = point_list_to_TColgp_Array1OfPnt(pnts)
    crv = GeomAPI_PointsToBSpline(pnts)
    return crv.Curve()


def interpolate_points_to_spline(
    list_of_points, start_tangent, end_tangent, filter_pts=True, tolerance=TOLERANCE
):
    """
    GeomAPI_Interpolate is buggy: need to use `fix` in order
    to get the right points in...
    """

    def fix(li, _type):
        """function factory for 1-dimensional TCol* types"""
        pts = _type(1, len(li))
        for n, i in enumerate(li):
            pts.SetValue(n + 1, i)
        pts.thisown = False
        return pts

    if filter_pts:
        list_of_points = filter_points_by_distance(list_of_points, 0.1)

    fixed_points = fix(list_of_points, TColgp_HArray1OfPnt)
    try:
        interp = GeomAPI_Interpolate(fixed_points, False, tolerance)
        interp.Load(start_tangent, end_tangent, False)
        interp.Perform()
        if interp.IsDone():
            return interp.Curve()
    except RuntimeError:
        print("Failed to interpolate the shown points")


def interpolate_points_vectors_to_spline(
    list_of_points, list_of_vectors, vector_mask=None, tolerance=TOLERANCE
):
    """
    build a curve from a set of points and vectors
    the vectors describe the tangent vector at the corresponding point
    """
    # GeomAPI_Interpolate is buggy: need to use `fix` in order to
    # get the right points in...
    assert len(list_of_points) == len(
        list_of_vectors
    ), "vector and point list not of same length"

    def fix(li, _type):
        """function factory for 1-dimensional TCol* types"""
        pts = _type(1, len(li))
        for n, i in enumerate(li):
            pts.SetValue(n + 1, i)
        pts.thisown = False
        return pts

    if vector_mask is not None:
        assert len(vector_mask) == len(
            list_of_points
        ), "length vector mask is not of length points list nor []"
    else:
        vector_mask = [True for i in range(len(list_of_points))]

    fixed_mask = fix(vector_mask, TColStd_HArray1OfBoolean)
    fixed_points = fix(list_of_points, TColgp_HArray1OfPnt)
    fixed_vectors = fix(list_of_vectors, TColgp_Array1OfVec)

    try:
        interp = GeomAPI_Interpolate(fixed_points, False, tolerance)
        interp.Load(fixed_vectors, fixed_mask, False)
        interp.Perform()
        if interp.IsDone():
            return interp.Curve()
    except RuntimeError:
        # the exception was unclear
        raise RuntimeError("FAILED TO INTERPOLATE THE POINTS")


def interpolate_points_to_spline_no_tangency(
    list_of_points, filter_pts=True, closed=False, tolerance=TOLERANCE
):
    """
    GeomAPI_Interpolate is buggy: need to use `fix`
    in order to get the right points in...
    """

    def fix(li, _type):
        """function factory for 1-dimensional TCol* types"""
        pts = _type(1, len(li))
        for n, i in enumerate(li):
            pts.SetValue(n + 1, i)
        pts.thisown = False
        return pts

    if filter_pts:
        list_of_points = filter_points_by_distance(list_of_points, 0.1)

    fixed_points = fix(list_of_points, TColgp_HArray1OfPnt)
    try:
        interp = GeomAPI_Interpolate(fixed_points, closed, tolerance)
        interp.Perform()
        if interp.IsDone():
            return interp.Curve()

    except RuntimeError:
        # the exception was unclear
        raise RuntimeError("FAILED TO INTERPOLATE THE POINTS")


# ===========================================================================
# --- RANDOMNESS ---
# ===========================================================================


def random_vec():
    x, y, z = [random.uniform(-1, 1) for i in range(3)]
    return gp_Vec(x, y, z)


def random_colored_material_aspect():
    clrs = [i for i in dir(Graphic3d) if i.startswith("Graphic3d_NOM_")]
    color = random.sample(clrs, 1)[0]
    print("color", color)
    return Graphic3d.Graphic3d_MaterialAspect(getattr(Graphic3d, color))


def random_color():
    return color(random.random(), random.random(), random.random())


# ===========================================================================
# --- BUILD PATCHES ---
# ===========================================================================


def common_vertex(edg1, edg2):
    from OCC.Core.TopExp import topexp_CommonVertex

    vert = TopoDS_Vertex()
    if topexp_CommonVertex(edg1, edg2, vert):
        return vert
    else:
        raise ValueError("no common vertex found")


def midpoint(pntA, pntB):
    """
    computes the point that lies in the middle between pntA and pntB
    @param pntA:    gp_Pnt
    @param pntB:    gp_Pnt
    """
    vec1 = gp_Vec(pntA.XYZ())
    vec2 = gp_Vec(pntB.XYZ())
    veccie = (vec1 + vec2) / 2.0
    return gp_Pnt(veccie.XYZ())


def center_boundingbox(shape):
    """
    compute the center point of a TopoDS_Shape, based on its bounding box
    @param shape: TopoDS_* instance
    returns a gp_Pnt instance
    """
    xmin, ymin, zmin, xmax, ymax, zmax = get_boundingbox(shape, 1e-6)
    return midpoint(gp_Pnt(xmin, ymin, zmin), gp_Pnt(xmax, ymax, zmax))


def point_in_boundingbox(solid, pnt, tolerance=1e-5):
    """returns True if *pnt* lies in *boundingbox*, False if not
    this is a much speedier test than checking the TopoDS_Solid
    Args:
        solid   TopoDS_Solid
        pnt:    gp_Pnt

    Returns: bool
    """
    bbox = Bnd_Box()
    bbox.SetGap(tolerance)
    brepbndlib_Add(solid, bbox)
    return not bbox.IsOut(pnt)


def point_in_solid(solid, pnt, tolerance=1e-5):
    """returns True if *pnt* lies in *solid*, False if not
    Args:
        solid   TopoDS_Solid
        pnt:    gp_Pnt

    Returns: bool
    """
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.TopAbs import TopAbs_ON, TopAbs_OUT, TopAbs_IN

    _in_solid = BRepClass3d_SolidClassifier(solid, pnt, tolerance)
    print("State", _in_solid.State())
    if _in_solid.State() == TopAbs_ON:
        return None, "on"
    if _in_solid.State() == TopAbs_OUT:
        return False, "out"
    if _in_solid.State() == TopAbs_IN:
        return True, "in"


def intersection_from_three_planes(planeA, planeB, planeC):
    """
    intersection from 3 planes
    accepts both Geom_Plane and gp_Pln
    @param planeA:
    @param planeB:
    @param planeC:
    @param show:
    """
    from OCC.Core.IntAna import IntAna_Int3Pln

    planeA = planeA if not hasattr(planeA, "Pln") else planeA.Pln()
    planeB = planeB if not hasattr(planeB, "Pln") else planeB.Pln()
    planeC = planeC if not hasattr(planeC, "Pln") else planeC.Pln()

    intersection_planes = IntAna_Int3Pln(planeA, planeB, planeC)
    pnt = intersection_planes.Value()
    return pnt


def intersect_shape_by_line(
    topods_shape, line, low_parameter=0.0, hi_parameter=float("+inf")
):
    """
    finds the intersection of a shape and a line

    :param shape: any TopoDS_*
    :param line: gp_Lin
    :param low_parameter:
    :param hi_parameter:

    :return: a list with a number of tuples that corresponds to the number
    of intersections found
    the tuple contains ( gp_Pnt, TopoDS_Face, u,v,w ), respectively the
    intersection point, the intersecting face
    and the u,v,w parameters of the intersection point
    :raise:
    """
    from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector

    shape_inter = IntCurvesFace_ShapeIntersector()
    shape_inter.Load(topods_shape, TOLERANCE)
    shape_inter.PerformNearest(line, low_parameter, hi_parameter)

    with assert_isdone(shape_inter, "failed to computer shape / line intersection"):
        return (
            shape_inter.Pnt(1),
            shape_inter.Face(1),
            shape_inter.UParameter(1),
            shape_inter.VParameter(1),
            shape_inter.WParameter(1),
        )


def normal_vector_from_plane(plane, vec_length=1.0):
    """
    returns a vector normal to the plane of length vec_length
    @param plane:
    """
    trns = gp_Vec(plane.Axis().Direction())
    return trns.Normalized() * vec_length


# ===========================================================================
# FIX
# ===========================================================================


def fix_tolerance(shape, tolerance=TOLERANCE):
    from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance

    ShapeFix_ShapeTolerance().SetTolerance(shape, tolerance)


def fix_continuity(edge, continuity=1):
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_ShapeDivideContinuity

    su = ShapeUpgrade_ShapeDivideContinuity(edge)
    su.SetBoundaryCriterion(eval("GeomAbs_C" + str(continuity)))
    su.Perform()
    te = st(su.Result())
    return te


def resample_curve_with_uniform_deflection(
    curve,
    deflection=0.5,
    degreeMin=3,
    degreeMax=8,
    continuity=GeomAbs_C2,
    tolerance=1e-4,
):
    """
    fits a bspline through the samples on `curve`
    @param curve: TopoDS_Wire, TopoDS_Edge, curve
    @param n_samples:
    """
    from OCC.Core.GCPnts import GCPnts_UniformDeflection

    crv = to_adaptor_3d(curve)
    defl = GCPnts_UniformDeflection(crv, deflection)
    with assert_isdone(defl, "failed to compute UniformDeflection"):
        print("Number of points:", defl.NbPoints())
    sampled_pnts = [defl.Value(i) for i in range(1, defl.NbPoints())]
    resampled_curve = GeomAPI_PointsToBSpline(
        point_list_to_TColgp_Array1OfPnt(sampled_pnts),
        degreeMin,
        degreeMax,
        continuity,
        tolerance,
    )
    return resampled_curve.Curve().GetObject()


# ===========================================================================
# global properties
# ===========================================================================


class GpropsFromShape(object):
    def __init__(self, shape, tolerance=1e-5):
        self.shape = shape
        self.tolerance = tolerance

    def volume(self):
        """returns the volume of a solid"""
        prop = GProp_GProps()
        brepgprop_VolumeProperties(self.shape, prop, self.tolerance)
        return prop

    def surface(self):
        """returns the area of a surface"""
        prop = GProp_GProps()
        brepgprop_SurfaceProperties(self.shape, prop, self.tolerance)
        return prop

    def linear(self):
        """returns the length of a wire or edge"""
        prop = GProp_GProps()
        brepgprop_LinearProperties(self.shape, prop)
        return prop


def curve_length(crv):
    """
    get the length from a TopoDS_Edge or TopoDS_Wire
    """
    assert isinstance(crv, (TopoDS_Wire, TopoDS_Edge)), "either a wire or edge..."
    gprop = GpropsFromShape(crv)
    return gprop.linear().Mass()


# =======================================================================
# Distance
# =======================================================================


def minimum_distance(shp1, shp2):
    """
    compute minimum distance between 2 BREP's
    @param shp1:    any TopoDS_*
    @param shp2:    any TopoDS_*

    @return: minimum distance,
             minimum distance points on shp1
             minimum distance points on shp2
    """
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

    bdss = BRepExtrema_DistShapeShape(shp1, shp2)
    bdss.Perform()
    with assert_isdone(bdss, "failed computing minimum distances"):
        min_dist = bdss.Value()
        min_dist_shp1, min_dist_shp2 = [], []
        for i in range(1, bdss.NbSolution() + 1):
            min_dist_shp1.append(bdss.PointOnShape1(i))
            min_dist_shp2.append(bdss.PointOnShape2(i))
    return min_dist, min_dist_shp1, min_dist_shp2


def vertex2pnt(vertex):
    """returns a gp_Pnt from a TopoDS_Vertex"""
    from OCC.Core.Core.BRep import BRep_Tool

    return BRep_Tool.Pnt(vertex)


def adapt_edge_to_curve(edg):
    """
    returns a curve adaptor from an edge
    @param edg: TopoDS_Edge
    """
    return BRepAdaptor_Curve(edg)


def adapt_edge_to_hcurve(edg):
    c = BRepAdaptor_Curve()
    c.ChangeCurve().Initialize(edg)
    return c


def to_adaptor_3d(curveType):
    """
    abstract curve like type into an adaptor3d
    @param curveType:
    """
    if isinstance(curveType, TopoDS_Wire):
        return BRepAdaptor_CompCurve(curveType)
    elif isinstance(curveType, TopoDS_Edge):
        return BRepAdaptor_Curve(curveType)
    elif issubclass(curveType.__class__, Geom_Curve):
        return GeomAdaptor_Curve(curveType)
    elif hasattr(curveType, "GetObject"):
        _crv = curveType.GetObject()
        if issubclass(_crv.__class__, Geom_Curve):
            return GeomAdaptor_Curve(curveType)
    else:
        raise TypeError(
            "allowed types are Wire, Edge or a subclass of Geom_Curve\nGot a %s"
            % (curveType.__class__)
        )


def project_point_on_curve(crv, pnt):
    if isinstance(crv, TopoDS_Shape):
        # get the curve
        crv = adapt_edge_to_curve(crv).Curve().Curve()
    else:
        raise NotImplementedError("expected a TopoDS_Edge...")
    rrr = GeomAPI_ProjectPointOnCurve(pnt, crv)
    return rrr.LowerDistanceParameter(), rrr.NearestPoint()


def project_point_on_plane(plane, point):
    """
    project point on plane
    @param plane: Geom_Plane
    @param point: gp_Pnt
    """
    from OCC.Core.ProjLib import projlib_Project

    pl = plane.Pln()
    aa, bb = projlib_Project(pl, point).Coord()
    point = plane.Value(aa, bb)
    return point


def wire_to_curve(
    wire, tolerance=TOLERANCE, order=GeomAbs_C2, max_segment=200, max_order=12
):
    """
    a wire can consist of many edges.
    these edges are merged given a tolerance and a curve
    @param wire:
    """
    adap = BRepAdaptor_CompCurve(wire)
    hadap = BRepAdaptor_CompCurve(adap)
    from OCC.Core.Approx import Approx_Curve3d

    approx = Approx_Curve3d(hadap, tolerance, order, max_segment, max_order)
    with assert_isdone(approx, "not able to compute approximation from wire"):
        return approx.Curve().GetObject()


def adapt_edge_to_curve(edg):
    """
    returns a curve adaptor from an edge
    @param edg: TopoDS_Edge
    """
    return BRepAdaptor_Curve(edg)


def adapt_edge_to_hcurve(edg):
    c = BRepAdaptor_Curve()
    c.ChangeCurve().Initialize(edg)
    return c


================================================
File: CadSeqProc/OCCUtils/Construct.py
================================================
#!/usr/bin/env python

##Copyright 2011-2015 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>.

"""
This modules makes the construction of geometry a little easier
"""

from __future__ import with_statement
from functools import wraps
import warnings
import operator
import math

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepOffset import BRepOffset_Skin
from OCC.Core.Geom import Geom_TrimmedCurve
from OCC.Core.GeomConvert import GeomConvert_ApproxCurve
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Transform,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeShell,
    BRepBuilderAPI_MakeEdge2d,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_FindPlane,
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeEvolved
from OCC.Core.GeomAbs import (
    GeomAbs_Arc,
    GeomAbs_C2,
    GeomAbs_C0,
    GeomAbs_Tangent,
    GeomAbs_Intersection,
    GeomAbs_G1,
    GeomAbs_G2,
    GeomAbs_C1,
)
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopoDS import (
    TopoDS_Wire,
    TopoDS_Solid,
    TopoDS_Vertex,
    TopoDS_Shape,
    TopoDS_Builder,
    TopoDS_Compound,
)
from OCC.Core.TColgp import TColgp_SequenceOfVec, TColgp_HArray1OfPnt
from OCC.Core.gp import (
    gp_Vec,
    gp_Pnt,
    gp_Dir,
    gp_Trsf,
    gp_Ax1,
    gp_Quaternion,
    gp_Circ,
    gp_Pln,
)

from OCCUtils.Common import (
    TOLERANCE,
    assert_isdone,
    to_tcol_,
    to_adaptor_3d,
    vertex2pnt,
    smooth_pnts,
    points_to_bspline,
    project_point_on_curve,
)
from OCCUtils.types_lut import ShapeToTopology
from OCCUtils.Topology import Topo


EPSILON = TOLERANCE = 1e-6
ST = ShapeToTopology()


def point_to_vector(self):
    return gp_Vec(self.XYZ())


def vector_to_point(self):
    return gp_Pnt(self.XYZ())


def dir_to_vec(self):
    return gp_Vec(self)


def vec_to_dir(self):
    return gp_Dir(self)


def add_vector_to_point(self, vec):
    return (self.as_vec() + vec).as_pnt()


def gp_Pnt_get_state(self):
    """pack as a tuple

    used for copying / serializing the instance
    """
    return self.XYZ().Coord()


def gp_Pnt_set_state(self, state):
    """unpack tuple and return instance...

    used for copying / serializing the instance
    """
    self.__init__(*state)


def gp_Pnt_equal(self, other):
    return self.IsEqual(other, TOLERANCE)


def gp_pnt_print(self):
    x = self.X()
    y = self.Y()
    z = self.Z()
    return "< gp_Pnt: {0}, {1}, {2} >".format(x, y, z)


def gp_vec_print(self):
    x = self.X()
    y = self.Y()
    z = self.Z()
    magn = self.Magnitude()
    return "< gp_Vec: {0}, {1}, {2}, magnitude: {3} >".format(x, y, z, magn)


def gp_ax1_print(self):
    pX, pY, pZ = self.Location().Coord()
    dX, dY, dZ = self.Direction().Coord()
    return "< gp_Ax1: location: {pX}, {pY}, {pZ}, direction: {dX}, {dY}, {dZ} >".format(
        **vars()
    )


def gp_trsf_print(self):
    _f = lambda x: [self.Value(x, i) for i in range(1, 5)]
    a, b, c, d = _f(1)
    e, f, g, h = _f(2)
    i, j, k, l = _f(3)
    return "< gp_Trsf:\n {a:.3f}, {b:.3f}, {c:.3f}, {d:.3f}\n {e:.3f}, {f:.3f}, {g:.3f}, {h:.3f}\n {i:.3f}, {j:.3f}, {k:.3f}, {l:.3f} >".format(
        **vars()
    )


def gp_quat_print(self):
    w, x, y, z = self.W(), self.X(), self.Y(), self.Z()
    vec = gp_Vec()
    angle = math.degrees(self.GetVectorAndAngle(vec))
    return "< gp_Quaternion: w:{w}, x:{x}, y:{y}, z:{z} >\nvector:{vec} angle:{angle}".format(
        **vars()
    )


def _apply(pnt, other, _operator):
    if isinstance(other, gp_Pnt):
        return gp_Pnt(*map(lambda x: _operator(*x), zip(pnt.Coord(), other.Coord())))
    else:
        return gp_Pnt(*map(lambda x: _operator(x, other), pnt.Coord()))


def gp_pnt_add(self, other):
    return _apply(self, other, operator.add)


def gp_pnt_sub(self, other):
    return _apply(self, other, operator.sub)


def gp_pnt_mul(self, other):
    return _apply(self, other, operator.mul)


def gp_pnt_div(self, other):
    return _apply(self, other, operator.div)


# easier conversion between data types
gp_Vec.as_pnt = vector_to_point
gp_Pnt.as_vec = point_to_vector
gp_Pnt.add_vec = add_vector_to_point
gp_Dir.as_vec = dir_to_vec
gp_Vec.as_dir = vec_to_dir
# for copying / serializing
gp_Pnt.__getstate__ = gp_Pnt_get_state
gp_Pnt.__setstate__ = gp_Pnt_set_state
gp_Vec.__getstate__ = gp_Pnt_get_state
gp_Vec.__setstate__ = gp_Pnt_set_state
# equality, not identity comparison
gp_Pnt.__eq__ = gp_Pnt_equal
# print gp_Pnt() should return something informative...
gp_Vec.__repr__ = gp_vec_print
gp_Vec.__str__ = gp_vec_print
gp_Pnt.__repr__ = gp_pnt_print
gp_Pnt.__str__ = gp_pnt_print
gp_Ax1.__repr__ = gp_ax1_print
gp_Ax1.__str__ = gp_ax1_print
gp_Trsf.__repr__ = gp_trsf_print
gp_Trsf.__str__ = gp_trsf_print
gp_Quaternion.__repr__ = gp_quat_print
gp_Quaternion.__str__ = gp_quat_print
# gp_Pnt.__eq__ = gp_equal
gp_Pnt.__add__ = gp_pnt_add
gp_Pnt.__sub__ = gp_pnt_sub
gp_Pnt.__mul__ = gp_pnt_mul
gp_Pnt.__div__ = gp_pnt_div

# ===========================================================================
# ---TOPOLOGY---
# ===========================================================================


@wraps(BRepBuilderAPI_MakeSolid)
def make_solid(*args):
    sld = BRepBuilderAPI_MakeSolid(*args)
    with assert_isdone(sld, "failed to produce solid"):
        result = sld.Solid()
        return result


@wraps(BRepBuilderAPI_MakeShell)
def make_shell(*args):
    shell = BRepBuilderAPI_MakeShell(*args)
    st = ShapeToTopology()
    with assert_isdone(shell, "failed to produce shell"):
        result = shell.Shell()
        return st(result)


@wraps(BRepBuilderAPI_MakeFace)
def make_face(*args):
    face = BRepBuilderAPI_MakeFace(*args)
    with assert_isdone(face, "failed to produce face"):
        result = face.Face()
        return result


@wraps(BRepBuilderAPI_MakeEdge2d)
def make_edge2d(*args):
    edge = BRepBuilderAPI_MakeEdge2d(*args)
    with assert_isdone(edge, "failed to produce edge"):
        result = edge.Edge()
    return result


@wraps(BRepBuilderAPI_MakeEdge)
def make_edge(*args):
    edge = BRepBuilderAPI_MakeEdge(*args)
    with assert_isdone(edge, "failed to produce edge"):
        result = edge.Edge()
        return result


@wraps(BRepBuilderAPI_MakeVertex)
def make_vertex(*args):
    vert = BRepBuilderAPI_MakeVertex(*args)
    with assert_isdone(vert, "failed to produce vertex"):
        result = vert.Vertex()
        return result


@wraps(BRepBuilderAPI_MakeWire)
def make_wire(*args):
    # if we get an iterable, than add all edges to wire builder
    if isinstance(args[0], list) or isinstance(args[0], tuple):
        wire = BRepBuilderAPI_MakeWire()
        for i in args[0]:
            wire.Add(i)
        wire.Build()
        return wire.Wire()

    wire = BRepBuilderAPI_MakeWire(*args)
    wire.Build()
    with assert_isdone(wire, "failed to produce wire"):
        result = wire.Wire()
        return result


@wraps(BRepBuilderAPI_MakePolygon)
def make_polygon(args, closed=False):
    poly = BRepBuilderAPI_MakePolygon()
    for pt in args:
        # support nested lists
        if isinstance(pt, list) or isinstance(pt, tuple):
            for i in pt:
                poly.Add(i)
        else:
            poly.Add(pt)
    if closed:
        poly.Close()
    poly.Build()

    with assert_isdone(poly, "failed to produce wire"):
        result = poly.Wire()
        return result


@wraps(BRepBuilderAPI_MakePolygon)
def make_closed_polygon(*args):
    poly = BRepBuilderAPI_MakePolygon()
    for pt in args:
        if isinstance(pt, list) or isinstance(pt, tuple):
            for i in pt:
                poly.Add(i)
        else:
            poly.Add(pt)
    poly.Build()
    poly.Close()
    with assert_isdone(poly, "failed to produce wire"):
        result = poly.Wire()
        return result


# ===========================================================================
# PRIMITIVES
# ===========================================================================


def make_circle(pnt, radius):
    """
    returns an edge
    @param pnt:
    @param radius:
    """
    circ = gp_Circ()
    circ.SetLocation(pnt)
    circ.SetRadius(radius)
    return make_edge(circ)


def make_line(pnt1, pnt2):
    return make_edge(pnt1, pnt2)


def make_evolved(spine, profile):
    evol = BRepOffsetAPI_MakeEvolved(spine, profile)
    with assert_isdone(evol, "failed buillding evolved"):
        evol.Build()
        return evol.Evolved()


def make_pipe(spine, profile):
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipe

    pipe = BRepOffsetAPI_MakePipe(spine, profile)
    with assert_isdone(pipe, "failed building pipe"):
        pipe.Build()
        return pipe.Shape()


def make_prism(profile, vec):
    """
    makes a finite prism
    """
    pri = BRepPrimAPI_MakePrism(profile, vec, True)
    with assert_isdone(pri, "failed building prism"):
        pri.Build()
        return pri.Shape()


def make_offset_shape(
    shapeToOffset,
    offsetDistance,
    tolerance=TOLERANCE,
    offsetMode=BRepOffset_Skin,
    intersection=False,
    selfintersection=False,
    joinType=GeomAbs_Arc,
):
    """
    builds an offsetted shell from a shape
    construct an offsetted version of the shape
    """
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape

    try:
        offset = BRepOffsetAPI_MakeOffsetShape(
            shapeToOffset,
            offsetDistance,
            tolerance,
            offsetMode,
            intersection,
            selfintersection,
            joinType,
        )
        if offset.IsDone():
            return offset.Shape()
        else:
            return None
    except RuntimeError("failed to offset shape"):
        return None


def make_offset(wire_or_face, offsetDistance, altitude=0, joinType=GeomAbs_Arc):
    """
    builds a offsetted wire or face from a wire or face
    construct an offsetted version of the shape

    @param wire_or_face:        the wire or face to offset
    @param offsetDistance:      the distance to offset
    @param altitude:            move the offsetted shape to altitude
    from the normal of the wire or face
    @param joinType:            the type of offset you want
    can be one of GeomAbs_Arc, GeomAbs_Tangent, GeomAbs_Intersection

    note: a shape that has a negative offsetDistance will return
    a sharp corner
    """
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffset

    _joints = [GeomAbs_Arc, GeomAbs_Tangent, GeomAbs_Intersection]
    assert joinType in _joints, "%s is not one of %s" % (joinType, _joints)
    try:
        offset = BRepOffsetAPI_MakeOffset(wire_or_face, joinType)
        offset.Perform(offsetDistance, altitude)
        if offset.IsDone():
            return ST(offset.Shape())
        else:
            return None
    except RuntimeError("failed to offset shape"):
        return None


def make_loft(
    elements,
    ruled=False,
    tolerance=TOLERANCE,
    continuity=GeomAbs_C2,
    check_compatibility=True,
):
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections

    sections = BRepOffsetAPI_ThruSections(False, ruled, tolerance)
    for i in elements:
        if isinstance(i, TopoDS_Wire):
            sections.AddWire(i)
        elif isinstance(i, TopoDS_Vertex):
            sections.AddVertex(i)
        else:
            raise TypeError(
                "elements is a list of TopoDS_Wire or TopoDS_Vertex, found a %s fool"
                % i.__class__
            )

    sections.CheckCompatibility(check_compatibility)
    sections.SetContinuity(continuity)
    sections.Build()
    with assert_isdone(sections, "failed lofting"):
        te = ShapeToTopology()
        loft = te(sections.Shape())
        return loft


def make_ruled(edgeA, edgeB):
    from OCC.Core.BRepFill import brepfill_Face

    return brepfill_Face(edgeA, edgeB)


def make_plane(
    center=gp_Pnt(0, 0, 0),
    vec_normal=gp_Vec(0, 0, 1),
    extent_x_min=-100.0,
    extent_x_max=100.0,
    extent_y_min=-100.0,
    extent_y_max=100.0,
    depth=0.0,
):
    if depth != 0:
        center = center.add_vec(gp_Vec(0, 0, depth))
    PL = gp_Pln(center, vec_normal.as_dir())
    face = make_face(PL, extent_x_min, extent_x_max, extent_y_min, extent_y_max)
    return face


def make_oriented_box(v_corner, v_x, v_y, v_z):
    """
    produces an oriented box
    oriented meaning here that the x,y,z axis do not have to be
    cartesian aligned

    :param v_corner: the lower corner
    :param v_x: gp_Vec that describes the X-axis
    :param v_y: gp_Vec that describes the Y-axis
    :param v_z: gp_Vec that describes the Z-axis
    :return: TopoDS_Solid
    """
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipe

    verts = map(
        lambda x: x.as_pnt(),
        [v_corner, v_corner + v_x, v_corner + v_x + v_y, v_corner + v_y],
    )
    p = make_polygon(verts, closed=True)
    li = make_line(v_corner.as_pnt(), (v_corner + v_z).as_pnt())
    bmp = BRepOffsetAPI_MakePipe(p, li)
    bmp.Build()
    shp = bmp.Shape()

    bottom = make_face(p)
    top = translate_topods_from_vector(bottom, v_z, True)
    oriented_bbox = make_solid(sew_shapes([bottom, shp, top]))
    return oriented_bbox


@wraps(BRepPrimAPI_MakeBox)
def make_box(*args):
    box = BRepPrimAPI_MakeBox(*args)
    box.Build()
    with assert_isdone(box, "failed to built a cube..."):
        return box.Shape()


def make_n_sided(edges, points, continuity=GeomAbs_C0):
    """
    builds an n-sided patch, respecting the constraints defined by *edges*
    and *points*

    a simplified call to the BRepFill_Filling class
    its simplified in the sense that to all constraining edges and points
    the same level of *continuity* will be applied

    *continuity* represents:

    GeomAbs_C0 : the surface has to pass by 3D representation of the edge
    GeomAbs_G1 : the surface has to pass by 3D representation of the edge
    and to respect tangency with the given face
    GeomAbs_G2 : the surface has to pass by 3D representation of the edge
    and to respect tangency and curvature with the given face.

    NOTE: it is not required to set constraining points.
    just leave the tuple or list empty

    :param edges: the constraining edges
    :param points: the constraining points
    :param continuity: GeomAbs_0, 1, 2
    :return: TopoDS_Face
    """
    from OCC.Core.BRepFill import BRepFill_Filling

    n_sided = BRepFill_Filling()
    for edg in edges:
        n_sided.Add(edg, continuity)
    for pt in points:
        n_sided.Add(pt)
    n_sided.Build()
    face = n_sided.Face()
    return face


def make_n_sections(edges):
    from OCC.Core.TopTools import TopTools_SequenceOfShape
    from OCC.Core.BRepFill import BRepFill_NSections

    seq = TopTools_SequenceOfShape()
    for i in edges:
        seq.Append(i)
    n_sec = BRepFill_NSections(seq)
    return n_sec


def make_coons(edges):
    from OCC.GeomFill import GeomFill_BSplineCurves, GeomFill_StretchStyle

    if len(edges) == 4:
        spl1, spl2, spl3, spl4 = edges
        srf = GeomFill_BSplineCurves(spl1, spl2, spl3, spl4, GeomFill_StretchStyle)
    elif len(edges) == 3:
        spl1, spl2, spl3 = edges
        srf = GeomFill_BSplineCurves(spl1, spl2, spl3, GeomFill_StretchStyle)
    elif len(edges) == 2:
        spl1, spl2 = edges
        srf = GeomFill_BSplineCurves(spl1, spl2, GeomFill_StretchStyle)
    else:
        raise ValueError("give 2,3 or 4 curves")
    return srf.Surface()


def make_constrained_surface_from_edges(edges):
    """
    DOESNT RESPECT BOUNDARIES
    """
    from OCC.GeomPlate import GeomPlate_MakeApprox, GeomPlate_BuildPlateSurface
    from OCC.Core.BRepFill import BRepFill_CurveConstraint

    bpSrf = GeomPlate_BuildPlateSurface(3, 15, 2)
    for edg in edges:
        c = BRepAdaptor_Curve()
        c.ChangeCurve().Initialize(edg)
        constraint = BRepFill_CurveConstraint(c, 0)
        bpSrf.Add(constraint)
    bpSrf.Perform()
    maxSeg, maxDeg, critOrder = 9, 8, 0
    tol = 1e-4
    srf = bpSrf.Surface()
    plate = GeomPlate_MakeApprox(srf, tol, maxSeg, maxDeg, tol, critOrder)
    uMin, uMax, vMin, vMax = srf.Bounds()
    face = make_face(plate.Surface(), uMin, uMax, vMin, vMax)
    return face


def add_wire_to_face(face, wire, reverse=False):
    """
    apply a wire to a face
    use reverse to set the orientation of the wire to opposite
    @param face:
    @param wire:
    @param reverse:
    """
    face = BRepBuilderAPI_MakeFace(face)
    if reverse:
        wire.Reverse()
    face.Add(wire)
    result = face.Face()
    return result


def sew_shapes(shapes, tolerance=0.001):
    sew = BRepBuilderAPI_Sewing(tolerance)
    for shp in shapes:
        if isinstance(shp, list):
            for i in shp:
                sew.Add(i)
        else:
            sew.Add(shp)
    sew.Perform()
    print("n degenerated shapes", sew.NbDegeneratedShapes())
    print("n deleted faces:", sew.NbDeletedFaces())
    print("n free edges", sew.NbFreeEdges())
    print("n multiple edges:", sew.NbMultipleEdges())
    result = ShapeToTopology()(sew.SewedShape())
    return result


# ===========================================================================
# ---BOOL---
# ===========================================================================


def boolean_cut(shapeToCutFrom, cuttingShape):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut

    try:
        cut = BRepAlgoAPI_Cut(shapeToCutFrom, cuttingShape)
        print("Can work?", cut.BuilderCanWork())
        _error = {
            0: "- Ok",
            1: "- The Object is created but Nothing is Done",
            2: "- Null source shapes is not allowed",
            3: "- Check types of the arguments",
            4: "- Can not allocate memory for the DSFiller",
            5: "- The Builder can not work with such types of arguments",
            6: "- Unknown operation is not allowed",
            7: "- Can not allocate memory for the Builder",
        }
        print("Error status:", _error[cut.ErrorStatus()])
        cut.RefineEdges()
        cut.FuseEdges()
        shp = cut.Shape()
        cut.Destroy()
        return shp
    except:
        print("Failed to boolean cut")
        return shapeToCutFrom


def boolean_fuse(shapeToCutFrom, joiningShape):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

    join = BRepAlgoAPI_Fuse(shapeToCutFrom, joiningShape)
    join.RefineEdges()
    join.FuseEdges()
    shape = join.Shape()
    join.Destroy()
    return shape


def trim_wire(wire, shapeLimit1, shapeLimit2, periodic=False):
    """return the trimmed wire that lies between `shapeLimit1`
    and `shapeLimit2`
    returns TopoDS_Edge
    """
    adap = to_adaptor_3d(wire)
    bspl = adap.BSpline()
    if periodic:
        if bspl.IsClosed():
            bspl.SetPeriodic()
        else:
            warnings.warn(
                "the wire to be trimmed is not closed, hence cannot be made periodic"
            )
    p1 = project_point_on_curve(bspl, shapeLimit1)[0]
    p2 = project_point_on_curve(bspl, shapeLimit2)[0]
    a, b = sorted([p1, p2])
    tr = Geom_TrimmedCurve(bspl, a, b)
    return make_edge(tr)


# ===========================================================================
# ---FIXES---
# ===========================================================================


def fix_shape(shp, tolerance=1e-3):
    from OCC.ShapeFix import ShapeFix_Shape

    fix = ShapeFix_Shape(shp)
    fix.SetFixFreeShellMode(True)
    sf = fix.FixShellTool()
    sf.SetFixOrientationMode(True)
    fix.LimitTolerance(tolerance)
    fix.Perform()
    return fix.Shape()


def fix_face(shp, tolerance=1e-3):
    from OCC.ShapeFix import ShapeFix_Face

    fix = ShapeFix_Face(shp)
    fix.SetMaxTolerance(tolerance)
    fix.Perform()
    return fix.Face()


# ===========================================================================
# --- TRANSFORM ---
# ===========================================================================


def translate_topods_from_vector(brep_or_iterable, vec, copy=False):
    """
    translate a brep over a vector
    @param brep:    the Topo_DS to translate
    @param vec:     the vector defining the translation
    @param copy:    copies to brep if True
    """
    st = ShapeToTopology()
    trns = gp_Trsf()
    trns.SetTranslation(vec)
    if issubclass(brep_or_iterable.__class__, TopoDS_Shape):
        brep_trns = BRepBuilderAPI_Transform(brep_or_iterable, trns, copy)
        brep_trns.Build()
        return st(brep_trns.Shape())
    else:
        return [
            translate_topods_from_vector(brep_or_iterable, vec, copy)
            for i in brep_or_iterable
        ]


def scale_uniformal(brep, pnt, factor, copy=False):
    """
    translate a brep over a vector
    @param brep:    the Topo_DS to translate
    @param pnt:     a gp_Pnt
    @param triple:  scaling factor
    @param copy:    copies to brep if True
    """
    trns = gp_Trsf()
    trns.SetScale(pnt, factor)
    brep_trns = BRepBuilderAPI_Transform(brep, trns, copy)
    brep_trns.Build()
    return brep_trns.Shape()


def mirror_pnt_dir(brep, pnt, direction, copy=False):
    """
    @param brep:
    @param line:
    """
    trns = gp_Trsf()
    trns.SetMirror(gp_Ax1(pnt, direction))
    brep_trns = BRepBuilderAPI_Transform(brep, trns, copy)
    with assert_isdone(brep_trns, "could not produce mirror"):
        brep_trns.Build()
        return brep_trns.Shape()


def mirror_axe2(brep, axe2, copy=False):
    """
    @param brep:
    @param line:
    """
    trns = gp_Trsf()
    trns.SetMirror(axe2)
    brep_trns = BRepBuilderAPI_Transform(brep, trns, copy)
    with assert_isdone(brep_trns, "could not produce mirror"):
        brep_trns.Build()
        return brep_trns.Shape()


def rotate(brep, axe, degree, copy=False):
    """
    @param brep:
    @param axe:
    @param degree:
    """
    from math import radians

    trns = gp_Trsf()
    trns.SetRotation(axe, radians(degree))
    brep_trns = BRepBuilderAPI_Transform(brep, trns, copy)
    with assert_isdone(brep_trns, "could not produce rotation"):
        brep_trns.Build()
        return ST(brep_trns.Shape())


# =============================================================================
# Not so sure where this should be located
# =============================================================================


def face_normal(face):
    from OCC.Core.BRepTools import breptools_UVBounds

    umin, umax, vmin, vmax = breptools_UVBounds(face)
    surf = BRep_Tool().Surface(face)
    props = GeomLProp_SLProps(
        surf, (umin + umax) / 2.0, (vmin + vmax) / 2.0, 1, TOLERANCE
    )
    norm = props.Normal()
    if face.Orientation() == TopAbs_REVERSED:
        norm.Reverse()
    return norm


def face_from_plane(_geom_plane, lowerLimit=-1000, upperLimit=1000):
    from OCC.Geom import Geom_RectangularTrimmedSurface

    _trim_plane = make_face(
        Geom_RectangularTrimmedSurface(
            _geom_plane, lowerLimit, upperLimit, lowerLimit, upperLimit
        )
    )
    return _trim_plane


def find_plane_from_shape(shape, tolerance=-1):
    try:
        fpl = BRepBuilderAPI_FindPlane(shape, tolerance)
        if fpl.Found():
            return fpl.Plane()
        else:
            return None
    except:
        raise AssertionError("couldnt find plane in %s" % (shape))


def fit_plane_through_face_vertices(_face):
    """
    :param _face:   OCC.KBE.face.Face instance
    :return:        Geom_Plane
    """
    from OCC.GeomPlate import GeomPlate_BuildAveragePlane

    uvs_from_vertices = [
        _face.project_vertex(vertex2pnt(i)) for i in Topo(_face).vertices()
    ]
    normals = [gp_Vec(_face.DiffGeom.normal(*uv[0])) for uv in uvs_from_vertices]
    points = [i[1] for i in uvs_from_vertices]

    NORMALS = TColgp_SequenceOfVec()
    [NORMALS.Append(i) for i in normals]
    POINTS = to_tcol_(points, TColgp_HArray1OfPnt)

    pl = GeomPlate_BuildAveragePlane(NORMALS, POINTS).Plane()
    vec = gp_Vec(pl.Location(), _face.GlobalProperties.centre())
    pt = (pl.Location().as_vec() + vec).as_pnt()
    pl.SetLocation(pt)
    return pl


def project_edge_onto_plane(edg, plane):
    """
    :param edg:     kbe.edge.Edge
    :param plane:   Geom_Plane
    :return:        TopoDS_Edge projected on the plane
    """
    from OCC.GeomProjLib import geomprojlib_ProjectOnPlane

    proj = geomprojlib_ProjectOnPlane(
        edg.adaptor.Curve().Curve(), plane, plane.Axis().Direction(), 1
    )
    return make_edge(proj)


def curve_to_bspline(
    crv, tolerance=TOLERANCE, continuity=GeomAbs_C1, sections=300, degree=12
):
    approx_curve = GeomConvert_ApproxCurve(crv, tolerance, continuity, sections, degree)
    with assert_isdone(approx_curve, "could not compute bspline from curve"):
        return approx_curve.Curve()


def compound(topo):
    """
    accumulate a bunch of TopoDS_* in list `topo` to a TopoDS_Compound
    @param topo: list of TopoDS_* instances
    """
    bd = TopoDS_Builder()
    comp = TopoDS_Compound()
    bd.MakeCompound(comp)
    for i in topo:
        bd.Add(comp, i)
    return comp


def geodesic_path(
    pntA, pntB, edgA, edgB, kbe_face, n_segments=20, _tolerance=0.1, n_iter=20
):
    """
    :param pntA:        point to start from
    :param pntB:        point to move towards
    :param edgA:        edge to start from
    :param edgB:        edge to move towards
    :param kbe_face:    kbe.face.Face on which `edgA` and `edgB` lie
    :param n_segments:  the number of segments the geodesic is built from
    :param _tolerance:  tolerance when the geodesic is converged
    :param n_iter:      maximum number of iterations
    :return:            TopoDS_Edge
    """
    uvA, srf_pnt_A = kbe_face.project_vertex(pntA)
    uvB, srf_pnt_B = kbe_face.project_vertex(pntB)

    path = []
    for i in range(n_segments):
        t = i / float(n_segments)
        u = uvA[0] + t * (uvB[0] - uvA[0])
        v = uvA[1] + t * (uvB[1] - uvA[1])
        path.append(kbe_face.parameter_to_point(u, v))

    project_pnts = lambda x: [kbe_face.project_vertex(i)[1] for i in x]
    poly_length = lambda x: sum(
        [x[i].Distance(x[i + 1]) for i in range(len(x) - 1)]
    ) / len(x)

    length = poly_length(path)

    n = 0
    while True:
        path = smooth_pnts(path)
        path = project_pnts(path)
        newlength = poly_length(path)
        if abs(newlength - length) < _tolerance or n == n_iter:
            crv = points_to_bspline(path)
            return make_edge(crv)
        n += 1


================================================
File: CadSeqProc/OCCUtils/Image.py
================================================
#!/usr/bin/env python

##Copyright 2008-2015 Thomas Paviot (tpaviot@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>.

import os
import os.path


class Texture(object):
    """
    This class encapsulates the necessary texture properties:
    Filename, toScaleU, etc.
    """

    def __init__(self, filename):
        if not os.path.isfile(filename):
            raise IOError("File %s not found.\n" % filename)
        self._filename = filename
        self._toScaleU = 1.0
        self._toScaleV = 1.0
        self._toRepeatU = 1.0
        self._toRepeatV = 1.0
        self._originU = 0.0
        self._originV = 0.0

    def TextureScale(self, toScaleU, toScaleV):
        self._toScaleU = toScaleU
        self._toScaleV = toScaleV

    def TextureRepeat(self, toRepeatU, toRepeatV):
        self._toRepeatU = toRepeatU
        self._toRepeatV = toRepeatV

    def TextureOrigin(self, originU, originV):
        self._originU = originU
        self._originV = originV

    def GetProperties(self):
        return (
            self._filename,
            self._toScaleU,
            self._toScaleV,
            self._toRepeatU,
            self._toRepeatV,
            self._originU,
            self._originV,
        )


================================================
File: CadSeqProc/OCCUtils/Iteration.py
================================================
##Copyright 2008-2015 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>.

"""
This module helps looping through topology
"""
from OCC.Core.BRep import BRep_Tool

from OCCUtils.Topology import WireExplorer, Topo
from OCCUtils.edge import Edge


class EdgePairsFromWire(object):
    """
    helper class to loop through a wire and return ordered pairs of edges
    """

    def __init__(self, wire):
        self.wire = wire
        self.edge_pairs = []
        self.prev_edge = None
        self.we = WireExplorer(self.wire).ordered_edges()
        self.number_of_edges = self.we.__length_hint__()
        self.previous_edge = None
        self.current_edge = None
        self.first_edge = None
        self.index = 0

    def next(self):
        if self.index == 0:
            # first edge, need to set self.previous_edge
            self.previous_edge = next(self.we)
            self.current_edge = next(self.we)
            self.first_edge = self.previous_edge  # for the last iteration
            self.index += 1
            return [self.previous_edge, self.current_edge]
        elif self.index == self.number_of_edges - 1:
            # no next edge
            self.index += 1
            return [self.current_edge, self.first_edge]
        else:
            self.previous_edge = self.current_edge
            self.current_edge = next(self.we)
            self.index += 1
            return [self.previous_edge, self.current_edge]

    def __iter__(self):
        return self


class LoopWirePairs(object):
    """
    for looping through consequtive wires
    assures that the returned edge pairs are ordered
    """

    def __init__(self, wireA, wireB):
        self.wireA = wireA
        self.wireB = wireB
        self.we_A = WireExplorer(self.wireA)
        self.we_B = WireExplorer(self.wireB)
        self.tp_A = Topo(self.wireA)
        self.tp_B = Topo(self.wireB)
        self.bt = BRep_Tool()
        self.vertsA = [v for v in self.we_A.ordered_vertices()]
        self.vertsB = [v for v in self.we_B.ordered_vertices()]

        self.edgesA = [v for v in WireExplorer(wireA).ordered_edges()]
        self.edgesB = [v for v in WireExplorer(wireB).ordered_edges()]

        self.pntsB = [self.bt.Pnt(v) for v in self.vertsB]
        self.number_of_vertices = len(self.vertsA)
        self.index = 0

    def closest_point(self, vertexFromWireA):
        pt = self.bt.Pnt(vertexFromWireA)
        distances = [pt.Distance(i) for i in self.pntsB]
        indx_max_dist = distances.index(min(distances))
        return self.vertsB[indx_max_dist]

    def next(self):
        if self.index == self.number_of_vertices:
            raise StopIteration

        vert = self.vertsA[self.index]
        closest = self.closest_point(vert)
        edges_a = self.tp_A.edges_from_vertex(vert)
        edges_b = self.tp_B.edges_from_vertex(closest)
        a1, a2 = Edge(next(edges_a)), Edge(next(edges_a))
        b1, b2 = Edge(next(edges_b)), Edge(next(edges_b))
        mpA = a1.mid_point()
        self.index += 1

        if mpA.Distance(b1.mid_point()) < mpA.Distance(b2.mid_point()):
            return iter([a1, a2]), iter([b1, b2])
        else:
            return iter([a1, a2]), iter([b2, b1])

    def __iter__(self):
        return self


================================================
File: CadSeqProc/OCCUtils/Topology.py
================================================
#!/usr/bin/env python

##Copyright 2008-2015 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

__all__ = ["Topo", "WireExplorer", "dumpTopology"]

from OCC.Core.BRep import BRep_Tool

from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.TopAbs import (
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_WIRE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_COMPOUND,
    TopAbs_COMPSOLID,
)
from OCC.Core.TopExp import TopExp_Explorer, topexp_MapShapesAndAncestors
from OCC.Core.TopTools import (
    TopTools_ListOfShape,
    TopTools_ListIteratorOfListOfShape,
    TopTools_IndexedDataMapOfShapeListOfShape,
)
from OCC.Core.TopoDS import (
    topods,
    TopoDS_Wire,
    TopoDS_Vertex,
    TopoDS_Edge,
    TopoDS_Face,
    TopoDS_Shell,
    TopoDS_Solid,
    TopoDS_Compound,
    TopoDS_CompSolid,
    topods_Edge,
    topods_Vertex,
    TopoDS_Iterator,
)


class WireExplorer(object):
    """
    Wire traversal
    """

    def __init__(self, wire):
        assert isinstance(wire, TopoDS_Wire), "not a TopoDS_Wire"
        self.wire = wire
        self.wire_explorer = BRepTools_WireExplorer(self.wire)
        self.done = False

    def _reinitialize(self):
        self.wire_explorer = BRepTools_WireExplorer(self.wire)
        self.done = False

    def _loop_topo(self, edges=True):
        if self.done:
            self._reinitialize()
        topologyType = topods_Edge if edges else topods_Vertex
        seq = []
        hashes = []  # list that stores hashes to avoid redundancy
        occ_seq = TopTools_ListOfShape()
        while self.wire_explorer.More():
            # loop edges
            if edges:
                current_item = self.wire_explorer.Current()
            # loop vertices
            else:
                current_item = self.wire_explorer.CurrentVertex()
            current_item_hash = current_item.__hash__()
            if not current_item_hash in hashes:
                hashes.append(current_item_hash)
                occ_seq.Append(current_item)
            self.wire_explorer.Next()

        # Convert occ_seq to python list
        occ_iterator = TopTools_ListIteratorOfListOfShape(occ_seq)
        while occ_iterator.More():
            topo_to_add = topologyType(occ_iterator.Value())
            seq.append(topo_to_add)
            occ_iterator.Next()
        self.done = True
        return iter(seq)

    def ordered_edges(self):
        return self._loop_topo(edges=True)

    def ordered_vertices(self):
        return self._loop_topo(edges=False)


class Topo(object):
    """
    Topology traversal
    """

    def __init__(self, myShape, ignore_orientation=False):
        """

        implements topology traversal from any TopoDS_Shape
        this class lets you find how various topological entities are connected from one to another
        find the faces connected to an edge, find the vertices this edge is made from, get all faces connected to
        a vertex, and find out how many topological elements are connected from a source

        *note* when traversing TopoDS_Wire entities, its advised to use the specialized
        ``WireExplorer`` class, which will return the vertices / edges in the expected order

        :param myShape: the shape which topology will be traversed

        :param ignore_orientation: filter out TopoDS_* entities of similar TShape but different Orientation

        for instance, a cube has 24 edges, 4 edges for each of 6 faces

        that results in 48 vertices, while there are only 8 vertices that have a unique
        geometric coordinate

        in certain cases ( computing a graph from the topology ) its preferable to return
        topological entities that share similar geometry, though differ in orientation
        by setting the ``ignore_orientation`` variable
        to True, in case of a cube, just 12 edges and only 8 vertices will be returned

        for further reference see TopoDS_Shape IsEqual / IsSame methods

        """
        self.myShape = myShape
        self.ignore_orientation = ignore_orientation

        # the topoFactory dicts maps topology types and functions that can
        # create this topology
        self.topoFactory = {
            TopAbs_VERTEX: topods.Vertex,
            TopAbs_EDGE: topods.Edge,
            TopAbs_FACE: topods.Face,
            TopAbs_WIRE: topods.Wire,
            TopAbs_SHELL: topods.Shell,
            TopAbs_SOLID: topods.Solid,
            TopAbs_COMPOUND: topods.Compound,
            TopAbs_COMPSOLID: topods.CompSolid,
        }

    def _loop_topo(
        self, topologyType, topologicalEntity=None, topologyTypeToAvoid=None
    ):
        """
        this could be a faces generator for a python TopoShape class
        that way you can just do:
        for face in srf.faces:
            processFace(face)
        """
        topoTypes = {
            TopAbs_VERTEX: TopoDS_Vertex,
            TopAbs_EDGE: TopoDS_Edge,
            TopAbs_FACE: TopoDS_Face,
            TopAbs_WIRE: TopoDS_Wire,
            TopAbs_SHELL: TopoDS_Shell,
            TopAbs_SOLID: TopoDS_Solid,
            TopAbs_COMPOUND: TopoDS_Compound,
            TopAbs_COMPSOLID: TopoDS_CompSolid,
        }

        assert topologyType in topoTypes.keys(), "%s not one of %s" % (
            topologyType,
            topoTypes.keys(),
        )
        self.topExp = TopExp_Explorer()
        # use self.myShape if nothing is specified
        if topologicalEntity is None and topologyTypeToAvoid is None:
            self.topExp.Init(self.myShape, topologyType)
        elif topologicalEntity is None and topologyTypeToAvoid is not None:
            self.topExp.Init(self.myShape, topologyType, topologyTypeToAvoid)
        elif topologyTypeToAvoid is None:
            self.topExp.Init(topologicalEntity, topologyType)
        elif topologyTypeToAvoid:
            self.topExp.Init(topologicalEntity, topologyType, topologyTypeToAvoid)
        seq = []
        hashes = []  # list that stores hashes to avoid redundancy
        occ_seq = TopTools_ListOfShape()
        while self.topExp.More():
            current_item = self.topExp.Current()
            current_item_hash = current_item.__hash__()

            if not current_item_hash in hashes:
                hashes.append(current_item_hash)
                occ_seq.Append(current_item)

            self.topExp.Next()
        # Convert occ_seq to python list
        occ_iterator = TopTools_ListIteratorOfListOfShape(occ_seq)
        while occ_iterator.More():
            topo_to_add = self.topoFactory[topologyType](occ_iterator.Value())
            seq.append(topo_to_add)
            occ_iterator.Next()

        if self.ignore_orientation:
            # filter out those entities that share the same TShape
            # but do *not* share the same orientation
            filter_orientation_seq = []
            for i in seq:
                _present = False
                for j in filter_orientation_seq:
                    if i.IsSame(j):
                        _present = True
                        break
                if _present is False:
                    filter_orientation_seq.append(i)
            return filter_orientation_seq
        else:
            return iter(seq)

    def faces(self):
        """
        loops over all faces
        """
        return self._loop_topo(TopAbs_FACE)

    def _number_of_topo(self, iterable):
        n = 0
        for i in iterable:
            n += 1
        return n

    def number_of_faces(self):
        return self._number_of_topo(self.faces())

    def vertices(self):
        """
        loops over all vertices
        """
        return self._loop_topo(TopAbs_VERTEX)

    def number_of_vertices(self):
        return self._number_of_topo(self.vertices())

    def edges(self):
        """
        loops over all edges
        """
        return self._loop_topo(TopAbs_EDGE)

    def number_of_edges(self):
        return self._number_of_topo(self.edges())

    def wires(self):
        """
        loops over all wires
        """
        return self._loop_topo(TopAbs_WIRE)

    def number_of_wires(self):
        return self._number_of_topo(self.wires())

    def shells(self):
        """
        loops over all shells
        """
        return self._loop_topo(TopAbs_SHELL, None)

    def number_of_shells(self):
        return self._number_of_topo(self.shells())

    def solids(self):
        """
        loops over all solids
        """
        return self._loop_topo(TopAbs_SOLID, None)

    def number_of_solids(self):
        return self._number_of_topo(self.solids())

    def comp_solids(self):
        """
        loops over all compound solids
        """
        return self._loop_topo(TopAbs_COMPSOLID)

    def number_of_comp_solids(self):
        return self._number_of_topo(self.comp_solids())

    def compounds(self):
        """
        loops over all compounds
        """
        return self._loop_topo(TopAbs_COMPOUND)

    def number_of_compounds(self):
        return self._number_of_topo(self.compounds())

    def ordered_vertices_from_wire(self, wire):
        """
        @param wire: TopoDS_Wire
        """
        we = WireExplorer(wire)
        return we.ordered_vertices()

    def number_of_ordered_vertices_from_wire(self, wire):
        return self._number_of_topo(self.ordered_vertices_from_wire(wire))

    def ordered_edges_from_wire(self, wire):
        """
        @param wire: TopoDS_Wire
        """
        we = WireExplorer(wire)
        return we.ordered_edges()

    def number_of_ordered_edges_from_wire(self, wire):
        return self._number_of_topo(self.ordered_edges_from_wire(wire))

    def _map_shapes_and_ancestors(self, topoTypeA, topoTypeB, topologicalEntity):
        """
        using the same method
        @param topoTypeA:
        @param topoTypeB:
        @param topologicalEntity:
        """
        topo_set = set()
        _map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp_MapShapesAndAncestors(self.myShape, topoTypeA, topoTypeB, _map)
        results = _map.FindFromKey(topologicalEntity)
        if results.Size() == 0:
            yield None

        topology_iterator = TopTools_ListIteratorOfListOfShape(results)
        while topology_iterator.More():

            topo_entity = self.topoFactory[topoTypeB](topology_iterator.Value())

            # return the entity if not in set
            # to assure we're not returning entities several times
            if not topo_entity in topo_set:
                if self.ignore_orientation:
                    unique = True
                    for i in topo_set:
                        if i.IsSame(topo_entity):
                            unique = False
                            break
                    if unique:
                        yield topo_entity
                else:
                    yield topo_entity

            topo_set.add(topo_entity)
            topology_iterator.Next()

    def _number_shapes_ancestors(self, topoTypeA, topoTypeB, topologicalEntity):
        """returns the number of shape ancestors
        If you want to know how many edges a faces has:
        _number_shapes_ancestors(self, TopAbs_EDGE, TopAbs_FACE, edg)
        will return the number of edges a faces has
        @param topoTypeA:
        @param topoTypeB:
        @param topologicalEntity:
        """
        topo_set = set()
        _map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp_MapShapesAndAncestors(self.myShape, topoTypeA, topoTypeB, _map)
        results = _map.FindFromKey(topologicalEntity)
        if results.Size() == 0:
            return None
        topology_iterator = TopTools_ListIteratorOfListOfShape(results)
        while topology_iterator.More():
            topo_set.add(topology_iterator.Value())
            topology_iterator.Next()
        return len(topo_set)

    # ======================================================================
    # EDGE <-> FACE
    # ======================================================================
    def faces_from_edge(self, edge):
        """

        :param edge:
        :return:
        """
        return self._map_shapes_and_ancestors(TopAbs_EDGE, TopAbs_FACE, edge)

    def number_of_faces_from_edge(self, edge):
        """

        :param edge:
        :return:
        """
        return self._number_shapes_ancestors(TopAbs_EDGE, TopAbs_FACE, edge)

    def edges_from_face(self, face):
        """

        :param face:
        :return:
        """
        return self._loop_topo(TopAbs_EDGE, face)

    def number_of_edges_from_face(self, face):
        cnt = 0
        for i in self._loop_topo(TopAbs_EDGE, face):
            cnt += 1
        return cnt

    # ======================================================================
    # VERTEX <-> EDGE
    # ======================================================================
    def vertices_from_edge(self, edg):
        return self._loop_topo(TopAbs_VERTEX, edg)

    def number_of_vertices_from_edge(self, edg):
        cnt = 0
        for i in self._loop_topo(TopAbs_VERTEX, edg):
            cnt += 1
        return cnt

    def edges_from_vertex(self, vertex):
        return self._map_shapes_and_ancestors(TopAbs_VERTEX, TopAbs_EDGE, vertex)

    def number_of_edges_from_vertex(self, vertex):
        return self._number_shapes_ancestors(TopAbs_VERTEX, TopAbs_EDGE, vertex)

    # ======================================================================
    # WIRE <-> EDGE
    # ======================================================================
    def edges_from_wire(self, wire):
        return self._loop_topo(TopAbs_EDGE, wire)

    def number_of_edges_from_wire(self, wire):
        cnt = 0
        for i in self._loop_topo(TopAbs_EDGE, wire):
            cnt += 1
        return cnt

    def wires_from_edge(self, edg):
        return self._map_shapes_and_ancestors(TopAbs_EDGE, TopAbs_WIRE, edg)

    def wires_from_vertex(self, edg):
        return self._map_shapes_and_ancestors(TopAbs_VERTEX, TopAbs_WIRE, edg)

    def number_of_wires_from_edge(self, edg):
        return self._number_shapes_ancestors(TopAbs_EDGE, TopAbs_WIRE, edg)

    # ======================================================================
    # WIRE <-> FACE
    # ======================================================================
    def wires_from_face(self, face):
        return self._loop_topo(TopAbs_WIRE, face)

    def number_of_wires_from_face(self, face):
        cnt = 0
        for i in self._loop_topo(TopAbs_WIRE, face):
            cnt += 1
        return cnt

    def faces_from_wire(self, wire):
        return self._map_shapes_and_ancestors(TopAbs_WIRE, TopAbs_FACE, wire)

    def number_of_faces_from_wires(self, wire):
        return self._number_shapes_ancestors(TopAbs_WIRE, TopAbs_FACE, wire)

    # ======================================================================
    # VERTEX <-> FACE
    # ======================================================================
    def faces_from_vertex(self, vertex):
        return self._map_shapes_and_ancestors(TopAbs_VERTEX, TopAbs_FACE, vertex)

    def number_of_faces_from_vertex(self, vertex):
        return self._number_shapes_ancestors(TopAbs_VERTEX, TopAbs_FACE, vertex)

    def vertices_from_face(self, face):
        return self._loop_topo(TopAbs_VERTEX, face)

    def number_of_vertices_from_face(self, face):
        cnt = 0
        for i in self._loop_topo(TopAbs_VERTEX, face):
            cnt += 1
        return cnt

    # ======================================================================
    # FACE <-> SOLID
    # ======================================================================
    def solids_from_face(self, face):
        return self._map_shapes_and_ancestors(TopAbs_FACE, TopAbs_SOLID, face)

    def number_of_solids_from_face(self, face):
        return self._number_shapes_ancestors(TopAbs_FACE, TopAbs_SOLID, face)

    def faces_from_solids(self, solid):
        return self._loop_topo(TopAbs_FACE, solid)

    def number_of_faces_from_solids(self, solid):
        cnt = 0
        for i in self._loop_topo(TopAbs_FACE, solid):
            cnt += 1
        return cnt


def dumpTopology(shape, level=0):
    """
    Print the details of an object from the top down
    """
    brt = BRep_Tool()
    s = shape.ShapeType()
    if s == TopAbs_VERTEX:
        pnt = brt.Pnt(topods_Vertex(shape))
        print(
            ".." * level
            + "<Vertex %i: %s %s %s>" % (hash(shape), pnt.X(), pnt.Y(), pnt.Z())
        )
    else:
        print(".." * level, end="")
        print(shapeTypeString(shape))
    it = TopoDS_Iterator(shape)
    while it.More():
        shp = it.Value()
        it.Next()
        dumpTopology(shp, level + 1)


def shapeTypeString(shape):
    st = shape.ShapeType()
    s = "?"
    if st == TopAbs_VERTEX:
        s = "Vertex"
    if st == TopAbs_SOLID:
        s = "Solid"
    if st == TopAbs_EDGE:
        s = "Edge"
    if st == TopAbs_FACE:
        s = "Face"
    if st == TopAbs_SHELL:
        s = "Shell"
    if st == TopAbs_WIRE:
        s = "Wire"
    if st == TopAbs_COMPOUND:
        s = "Compound."
    if st == TopAbs_COMPSOLID:
        s = "Compsolid."
    return "%s: %i" % (s, hash(shape))


================================================
File: CadSeqProc/OCCUtils/__init__.py
================================================
from OCCUtils.Common import get_boundingbox
from OCCUtils.Topology import Topo


================================================
File: CadSeqProc/OCCUtils/base.py
================================================
##Copyright 2008-2013 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>

"""
Please note the following;
@readonly
means that the decorated method is a readonly descriptor
@property
means that the decorated method can be set / get using the descriptor
( irony is that just using @property *would*
    result in a readonly descriptor :")

Sometimes a set of methods should be contained in another module or class,
or simply grouped.
For instance the set of methods after:
#===========================================================================
#    Curve.local_properties
#===========================================================================

Can be a module, class or namespace.

"""

import functools

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCC.Core.BRepGProp import (
    brepgprop_VolumeProperties,
    brepgprop_LinearProperties,
    brepgprop_SurfaceProperties,
)
from OCC.Core.BRepCheck import (
    BRepCheck_Vertex,
    BRepCheck_Edge,
    BRepCheck_Wire,
    BRepCheck_Face,
    BRepCheck_Shell,
    BRepCheck_Analyzer,
)
from OCC.Core.GProp import GProp_GProps
from OCC.Display.SimpleGui import init_display

from OCCUtils.Common import get_boundingbox
from OCCUtils.Construct import make_vertex, TOLERANCE
from OCCUtils.types_lut import shape_lut, topo_lut, curve_lut, surface_lut

# ===========================================================================
# DISPLAY
# ===========================================================================
global display


class singleton(object):
    def __init__(self, cls):
        self.cls = cls
        self.instance_container = []

    def __call__(self, *args, **kwargs):
        if not len(self.instance_container):
            cls = functools.partial(self.cls, *args, **kwargs)
            self.instance_container.append(cls())
        return self.instance_container[0]


@singleton
class Display(object):
    def __init__(self):
        (
            self.display,
            self.start_display,
            self.add_menu,
            self.add_function_to_menu,
        ) = init_display()

    def __call__(self, *args, **kwargs):
        return self.display.DisplayShape(*args, **kwargs)


# ============
# base class
# ============


class BaseObject(object):
    """base class for all objects"""

    def __init__(self, name=None, tolerance=TOLERANCE):
        self.GlobalProperties = GlobalProperties(self)
        self.name = name
        self._dirty = False
        self.tolerance = tolerance
        self.display_set = False

    @property
    def is_dirty(self):
        """when an object is dirty, its topology will be
        rebuild when update is called"""
        return self._dirty

    @is_dirty.setter
    def is_dirty(self, _bool):
        self._dirty = bool(_bool)

    @property
    def topo_type(self):
        return topo_lut[self.ShapeType()]

    @property
    def geom_type(self):
        if self.topo_type == "edge":
            return curve_lut[self.ShapeType()]
        if self.topo_type == "face":
            return surface_lut[self.adaptor.GetType()]
        else:
            raise ValueError("geom_type works only for edges and faces...")

    def set_display(self, display):
        if hasattr(display, "DisplayShape"):
            self.display_set = True
            self.display = display
        else:
            raise ValueError("not a display")

    def check(self):
        """ """
        _check = dict(
            vertex=BRepCheck_Vertex,
            edge=BRepCheck_Edge,
            wire=BRepCheck_Wire,
            face=BRepCheck_Face,
            shell=BRepCheck_Shell,
        )
        _check[self.topo_type]
        # TODO: BRepCheck will be able to inform *what* actually is the matter,
        # though implementing this still is a bit of work...
        raise NotImplementedError

    def is_valid(self):
        analyse = BRepCheck_Analyzer(self)
        ok = analyse.IsValid()
        if ok:
            return True
        else:
            return False

    def copy(self):
        """

        :return:
        """
        cp = BRepBuilderAPI_Copy(self)
        cp.Perform(self)
        # get the class, construct a new instance
        # cast the cp.Shape() to its specific TopoDS topology
        _copy = self.__class__(shape_lut(cp.Shape()))
        return _copy

    def distance(self, other):
        """
        return the minimum distance

         :return: minimum distance,
             minimum distance points on shp1
             minimum distance points on shp2
        """
        return minimum_distance(self, other)

    def show(self, *args, **kwargs):
        """
        renders the topological entity in the viewer

        :param update: redraw the scene or not
        """
        if not self.display_set:
            Display()(self, *args, **kwargs)
        else:
            self.disp.DisplayShape(*args, **kwargs)

    def build(self):
        if self.name.startswith("Vertex"):
            self = make_vertex(self)

    def __eq__(self, other):
        return self.IsEqual(other)

    def __ne__(self, other):
        return not self.__eq__(other)


class GlobalProperties(object):
    """
    global properties for all topologies
    """

    def __init__(self, instance):
        self.instance = instance

    @property
    def system(self):
        self._system = GProp_GProps()
        # todo, type should be abstracted with TopoDS...
        _topo_type = self.instance.topo_type
        if _topo_type == "face" or _topo_type == "shell":
            brepgprop_SurfaceProperties(self.instance, self._system)
        elif _topo_type == "edge":
            brepgprop_LinearProperties(self.instance, self._system)
        elif _topo_type == "solid":
            brepgprop_VolumeProperties(self.instance, self._system)
        return self._system

    def centre(self):
        """
        :return: centre of the entity
        """
        return self.system.CentreOfMass()

    def inertia(self):
        """returns the inertia matrix"""
        return self.system.MatrixOfInertia(), self.system.MomentOfInertia()

    def area(self):
        """returns the area of the surface"""
        return self.system.Mass()

    def bbox(self):
        """
        returns the bounding box of the face
        """
        return get_boundingbox(self.instance)


================================================
File: CadSeqProc/OCCUtils/edge.py
================================================
##Copyright 2008-2015 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.Geom import Geom_OffsetCurve, Geom_TrimmedCurve
from OCC.Core.TopExp import topexp
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Vertex, TopoDS_Face
from OCC.Core.gp import gp_Vec, gp_Dir, gp_Pnt
from OCC.Core.GeomLProp import GeomLProp_CurveTool
from OCC.Core.BRepLProp import BRepLProp_CLProps
from OCC.Core.GeomLib import geomlib
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.BRep import BRep_Tool, BRep_Tool_Continuity
from OCC.Core.BRepIntCurveSurface import BRepIntCurveSurface_Inter

# high-level
from OCCUtils.Common import vertex2pnt, minimum_distance, assert_isdone, fix_continuity
from OCCUtils.Construct import make_edge
from OCCUtils.types_lut import geom_lut
from OCCUtils.base import BaseObject


class IntersectCurve(object):
    def __init__(self, instance):
        self.instance = instance

    def intersect(self, other, tolerance=1e-2):
        """Intersect self with a point, curve, edge, face, solid
        method wraps dealing with the various topologies
        """
        if isinstance(other, TopoDS_Face):
            face_curve_intersect = BRepIntCurveSurface_Inter()
            face_curve_intersect.Init(other, self.instance.adaptor.Curve(), tolerance)
            pnts = []
            while face_curve_intersect.More():
                next(face_curve_intersect)
                pnts.append(face_curve_intersect.Pnt())
            return pnts


class DiffGeomCurve(object):
    def __init__(self, instance):
        self.instance = instance
        self._local_props = BRepLProp_CLProps(
            self.instance.adaptor, 2, self.instance.tolerance
        )

    @property
    def _curvature(self):
        return self._local_props

    def radius(self, u):
        """returns the radius at u"""
        # NOT SO SURE IF THIS IS THE SAME THING!!!
        self._curvature.SetParameter(u)
        pnt = gp_Pnt()
        self._curvature.CentreOfCurvature(pnt)
        return pnt

    def curvature(self, u):
        # ugly
        self._curvature.SetParameter(u)
        return self._curvature.Curvature()

    def tangent(self, u):
        """sets or gets ( iff vector ) the tangency at the u parameter
        tangency can be constrained so when setting the tangency,
        you're constrainting it in fact
        """
        self._curvature.SetParameter(u)
        if self._curvature.IsTangentDefined():
            ddd = gp_Dir()
            self._curvature.Tangent(ddd)
            return ddd
        else:
            raise ValueError("no tangent defined")

    def normal(self, u):
        """returns the normal at u

        computes the main normal if no normal is found
        see:
        www.opencascade.org/org/forum/thread_645+&cd=10&hl=nl&ct=clnk&gl=nl
        """
        try:
            self._curvature.SetParameter(u)
            a_dir = gp_Dir()
            self._curvature.Normal(a_dir)
            return a_dir
        except:
            raise ValueError("no normal was found")

    def derivative(self, u, n):
        """
        returns n derivatives at parameter b
        """
        self._curvature.SetParameter(u)
        deriv = {
            1: self._curvature.D1,
            2: self._curvature.D2,
            3: self._curvature.D3,
        }
        try:
            return deriv[n]
        except KeyError:
            raise AssertionError("n of derivative is one of [1,2,3]")

    def points_from_tangential_deflection(self):
        pass


# ===========================================================================
#    Curve.Construct
# ===========================================================================


class ConstructFromCurve:
    def __init__(self, instance):
        self.instance = instance

    def make_offset(self, offset, vec):
        """
        returns an offsetted curve
        @param offset: the distance between self.crv and the curve to offset
        @param vec:    offset direction
        """
        return Geom_OffsetCurve(self.instance.h_crv, offset, vec)


class Edge(TopoDS_Edge, BaseObject):
    def __init__(self, edge):
        assert isinstance(edge, TopoDS_Edge), (
            "need a TopoDS_Edge, got a %s" % edge.__class__
        )
        assert not edge.IsNull()
        super(Edge, self).__init__()
        BaseObject.__init__(self, "edge")
        # we need to copy the base shape using the following three
        # lines
        assert self.IsNull()
        self.TShape(edge.TShape())
        self.Location(edge.Location())
        self.Orientation(edge.Orientation())
        assert not self.IsNull()

        # tracking state
        self._local_properties_init = False
        self._curvature_init = False
        self._geometry_lookup_init = False
        self._curve = None
        self._adaptor = None

        # instantiating cooperative classes
        # cooperative classes are distinct through CamelCaps from
        # normal method -> pep8
        self.DiffGeom = DiffGeomCurve(self)
        self.Intersect = IntersectCurve(self)
        self.Construct = ConstructFromCurve(self)

        # GeomLProp object
        self._curvature = None

    def is_closed(self):
        return self.adaptor.IsClosed()

    def is_periodic(self):
        return self.adaptor.IsPeriodic()

    def is_rational(self):
        return self.adaptor.IsRational()

    def continuity(self):
        return self.adaptor.Continuity

    def degree(self):
        if "line" in self.type:
            return 1
        elif "curve" in self.type:
            return self.adaptor.Degree()
        else:
            # hyperbola, parabola, circle
            return 2

    def nb_knots(self):
        return self.adaptor.NbKnots()

    def nb_poles(self):
        return self.adaptor.NbPoles()

    @property
    def curve(self):
        if self._curve is not None and not self.is_dirty:
            pass
        else:
            self._curve = BRep_Tool().Curve(self)[0]
        return self._curve

    @property
    def adaptor(self):
        if self._adaptor is not None and not self.is_dirty:
            pass
        else:
            self._adaptor = BRepAdaptor_Curve(self)
        return self._adaptor

    @property
    def type(self):
        return geom_lut[self.adaptor.Curve().GetType()]

    def pcurve(self, face):
        """
        computes the 2d parametric spline that lies on the surface of the face
        :return: Geom2d_Curve, u, v
        """
        crv, u, v = BRep_Tool().CurveOnSurface(self, face)
        return crv, u, v

    def _local_properties(self):
        self._lprops_curve_tool = GeomLProp_CurveTool()
        self._local_properties_init = True

    def domain(self):
        """returns the u,v domain of the curve"""
        return self.adaptor.FirstParameter(), self.adaptor.LastParameter()

    # ===========================================================================
    #    Curve.GlobalProperties
    # ===========================================================================

    def length(self, lbound=None, ubound=None, tolerance=1e-5):
        """returns the curve length
        if either lbound | ubound | both are given, than the length
        of the curve will be measured over that interval
        """
        _min, _max = self.domain()
        if _min < self.adaptor.FirstParameter():
            raise ValueError(
                "the lbound argument is lower than the first parameter of the curve: %s "
                % (self.adaptor.FirstParameter())
            )
        if _max > self.adaptor.LastParameter():
            raise ValueError(
                "the ubound argument is greater than the last parameter of the curve: %s "
                % (self.adaptor.LastParameter())
            )

        lbound = _min if lbound is None else lbound
        ubound = _max if ubound is None else ubound
        return GCPnts_AbscissaPoint().Length(self.adaptor, lbound, ubound, tolerance)

    # ===========================================================================
    #    Curve.modify
    # ===========================================================================

    def trim(self, lbound, ubound):
        """
        trim the curve
        @param lbound:
        @param ubound:
        """
        a, b = sorted([lbound, ubound])
        tr = Geom_TrimmedCurve(self.adaptor.Curve().Curve(), a, b)
        return Edge(make_edge(tr))

    def extend_by_point(self, pnt, degree=3, beginning=True):
        """extends the curve to point

        does not extend if the degree of self.curve > 3
        @param pnt:
        @param degree:
        @param beginning:
        """
        if self.degree > 3:
            raise ValueError(
                "to extend you self.curve should be <= 3, is %s" % (self.degree)
            )
        return geomlib.ExtendCurveToPoint(self.curve, pnt, degree, beginning)

    # ===========================================================================
    #    Curve.
    # ===========================================================================
    def closest(self, other):
        return minimum_distance(self, other)

    def project_vertex(self, pnt_or_vertex):
        """returns the closest orthogonal project on `pnt` on edge"""
        if isinstance(pnt_or_vertex, TopoDS_Vertex):
            pnt_or_vertex = vertex2pnt(pnt_or_vertex)

        poc = GeomAPI_ProjectPointOnCurve(pnt_or_vertex, self.curve)
        return poc.LowerDistanceParameter(), poc.NearestPoint()

    def distance_on_curve(self, distance, close_parameter, estimate_parameter):
        """returns the parameter if there is a parameter
        on the curve with a distance length from u
        raises OutOfBoundary if no such parameter exists
        """
        gcpa = GCPnts_AbscissaPoint(
            self.adaptor, distance, close_parameter, estimate_parameter, 1e-5
        )
        with assert_isdone(gcpa, "couldnt compute distance on curve"):
            return gcpa.Parameter()

    def mid_point(self):
        """
        :return: the parameter at the mid point of the curve, and
        its corresponding gp_Pnt
        """
        _min, _max = self.domain()
        _mid = (_min + _max) / 2.0
        return _mid, self.adaptor.Value(_mid)

    def divide_by_number_of_points(self, n_pts, lbound=None, ubound=None):
        """returns a nested list of parameters and points on the edge
        at the requested interval [(param, gp_Pnt),...]
        """
        _lbound, _ubound = self.domain()
        if lbound:
            _lbound = lbound
        elif ubound:
            _ubound = ubound

        # minimally two points or a Standard_ConstructionError is raised
        if n_pts <= 1:
            n_pts = 2

        try:
            npts = GCPnts_UniformAbscissa(self.adaptor, n_pts, _lbound, _ubound)
        except:
            print("Warning : GCPnts_UniformAbscissa failed")
        if npts.IsDone():
            tmp = []
            for i in xrange(1, npts.NbPoints() + 1):
                param = npts.Parameter(i)
                pnt = self.adaptor.Value(param)
                tmp.append((param, pnt))
            return tmp
        else:
            return None

    def __eq__(self, other):
        if hasattr(other, "topo"):
            return self.IsEqual(other)
        else:
            return self.IsEqual(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def first_vertex(self):
        return topexp.FirstVertex(self)

    def last_vertex(self):
        return topexp.LastVertex(self)

    def common_vertex(self, edge):
        vert = TopoDS_Vertex()
        if topexp.CommonVertex(self, edge, vert):
            return vert
        else:
            return False

    def as_vec(self):
        if self.is_line():
            first, last = map(vertex2pnt, [self.first_vertex(), self.last_vertex()])
            return gp_Vec(first, last)
        else:
            raise ValueError(
                "edge is not a line, hence no meaningful vector can be returned"
            )

    # ===========================================================================
    #    Curve.
    # ===========================================================================

    def parameter_to_point(self, u):
        """returns the coordinate at parameter u"""
        return self.adaptor.Value(u)

    def fix_continuity(self, continuity):
        """
        splits an edge to achieve a level of continuity
        :param continuity: GeomAbs_C*
        """
        return fix_continuity(self, continuity)

    def continuity_from_faces(self, f1, f2):
        return BRep_Tool_Continuity(self, f1, f2)

    # ===========================================================================
    #    Curve.
    # ===========================================================================

    def is_line(self):
        """checks if the curve is planar"""
        if self.nb_knots() == 2 and self.nb_poles() == 2:
            return True
        else:
            return False

    def is_seam(self, face):
        """
        :return: True if the edge has two pcurves on one surface
        ( in the case of a sphere for example... )
        """
        sae = ShapeAnalysis_Edge()
        return sae.IsSeam(self, face)

    def is_edge_on_face(self, face):
        """checks whether curve lies on a surface or a face"""
        return ShapeAnalysis_Edge().HasPCurve(self, face)

    # ===========================================================================
    #    Curve.graphic
    # ===========================================================================
    def show(self):
        """
        poles, knots, should render all slightly different.
        here's how...

        http://www.opencascade.org/org/forum/thread_1125/
        """
        super(Edge, self).show()


if __name__ == "__main__":
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCCUtils.Topology import Topo

    b = BRepPrimAPI_MakeBox(10, 20, 30).Shape()
    t = Topo(b)
    ed = next(t.edges())
    my_e = Edge(ed)
    print(my_e.tolerance)


================================================
File: CadSeqProc/OCCUtils/face.py
================================================
##Copyright 2008-2013 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>

from OCC.Core.BRep import BRep_Tool_Surface, BRep_Tool
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCC.Core.Geom import Geom_Curve
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomLib import GeomLib_IsPlanarSurface
from OCC.Core.TopAbs import TopAbs_IN
from OCC.Core.TopExp import topexp
from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Face, TopoDS_Edge
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.GeomProjLib import geomprojlib
from OCC.Core.Adaptor3d import Adaptor3d_IsoCurve
from OCC.Core.gp import gp_Pnt2d, gp_Dir

from OCCUtils.base import BaseObject
from OCCUtils.edge import Edge
from OCCUtils.Construct import TOLERANCE, to_adaptor_3d
from OCCUtils.Topology import Topo, WireExplorer


class DiffGeomSurface(object):
    def __init__(self, instance):
        self.instance = instance
        self._curvature = None
        self._curvature_initiated = False

    def curvature(self, u, v):
        """returns the curvature at the u parameter
        the curvature object can be returned too using
        curvatureType == curvatureType
        curvatureTypes are:
            gaussian
            minimum
            maximum
            mean
            curvatureType
        """
        if not self._curvature_initiated:
            self._curvature = GeomLProp_SLProps(self.instance.surface, u, v, 2, 1e-7)

        _domain = self.instance.domain()
        if u in _domain or v in _domain:
            print("<<<CORRECTING DOMAIN...>>>")
            div = 1000
            delta_u, delta_v = (_domain[0] - _domain[1]) / div, (
                _domain[2] - _domain[3]
            ) / div

            if u in _domain:
                low, hi = u - _domain[0], u - _domain[1]
                if low < hi:
                    u = u - delta_u
                else:
                    u = u + delta_u

            if v in _domain:
                low, hi = v - _domain[2], v - _domain[3]
                if low < hi:
                    v = v - delta_v
                else:
                    v = v + delta_v

        self._curvature.SetParameters(u, v)
        self._curvature_initiated = True

        return self._curvature

    def gaussian_curvature(self, u, v):
        return self.curvature(u, v).GaussianCurvature()

    def min_curvature(self, u, v):
        return self.curvature(u, v).MinCurvature()

    def mean_curvature(self, u, v):
        return self.curvature(u, v).MeanCurvature()

    def max_curvature(self, u, v):
        return self.curvature(u, v).MaxCurvature()

    def normal(self, u, v):
        # TODO: should make this return a gp_Vec
        curv = self.curvature(u, v)
        if curv.IsNormalDefined():
            return curv.Normal()
        else:
            raise ValueError("normal is not defined at u,v: {0}, {1}".format(u, v))

    def tangent(self, u, v):
        dU, dV = gp_Dir(), gp_Dir()
        curv = self.curvature(u, v)
        if curv.IsTangentUDefined() and curv.IsTangentVDefined():
            curv.TangentU(dU), curv.TangentV(dV)
            return dU, dV
        else:
            return None, None

    def radius(self, u, v):
        """returns the radius at u"""
        # TODO: SHOULD WE RETURN A SIGNED RADIUS? ( get rid of abs() )?
        try:
            _crv_min = 1.0 / self.min_curvature(u, v)
        except ZeroDivisionError:
            _crv_min = 0.0

        try:
            _crv_max = 1.0 / self.max_curvature(u, v)
        except ZeroDivisionError:
            _crv_max = 0.0
        return abs((_crv_min + _crv_max) / 2.0)


class Face(TopoDS_Face, BaseObject):
    """high level surface API
    object is a Face if part of a Solid
    otherwise the same methods do apply, apart from the topology obviously
    """

    def __init__(self, face):
        """ """
        assert isinstance(face, TopoDS_Face), (
            "need a TopoDS_Face, got a %s" % face.__class__
        )
        assert not face.IsNull()
        super(Face, self).__init__()
        BaseObject.__init__(self, "face")
        # we need to copy the base shape using the following three
        # lines
        assert self.IsNull()
        self.TShape(face.TShape())
        self.Location(face.Location())
        self.Orientation(face.Orientation())
        assert not self.IsNull()

        # cooperative classes
        self.DiffGeom = DiffGeomSurface(self)

        # STATE; whether cooperative classes are yet initialized
        self._curvature_initiated = False
        self._geometry_lookup_init = False

        # ===================================================================
        # properties
        # ===================================================================
        self._h_srf = None
        self._srf = None
        self._adaptor = None
        self._classify_uv = (
            None  # cache the u,v classifier, no need to rebuild for every sample
        )
        self._topo = None

        # aliasing of useful methods
        def is_u_periodic(self):
            return self.adaptor.IsUPeriodic()

        def is_v_periodic(self):
            return self.adaptor.IsVPeriodic()

        def is_u_closed(self):
            return self.adaptor.IsUClosed()

        def is_v_closed(self):
            return self.adaptor.IsVClosed()

        def is_u_rational(self):
            return self.adaptor.IsURational()

        def is_v_rational(self):
            return self.adaptor.IsVRational()

        def u_degree(self):
            return self.adaptor.UDegree()

        def v_degree(self):
            return self.adaptor.VDegree()

        def u_continuity(self):
            return self.adaptor.UContinuity()

        def v_continuity(self):
            return self.adaptor.VContinuity()

    def domain(self):
        """the u,v domain of the curve
        :return: UMin, UMax, VMin, VMax
        """
        return breptools_UVBounds(self)

    def mid_point(self):
        """
        :return: the parameter at the mid point of the face,
        and its corresponding gp_Pnt
        """
        u_min, u_max, v_min, v_max = self.domain()
        u_mid = (u_min + u_max) / 2.0
        v_mid = (v_min + v_max) / 2.0
        return ((u_mid, v_mid), self.adaptor.Value(u_mid, v_mid))

    @property
    def topo(self):
        if self._topo is not None:
            return self._topo
        else:
            self._topo = Topo(self)
            return self._topo

    @property
    def surface(self):
        if self._srf is None or self.is_dirty:
            self._srf = BRep_Tool_Surface(self)
        return self._srf

    @property
    def adaptor(self):
        if self._adaptor is not None and not self.is_dirty:
            pass
        else:
            self._adaptor = BRepAdaptor_Surface(self)
        return self._adaptor

    def is_closed(self):
        sa = ShapeAnalysis_Surface(self.surface)
        return sa.IsUClosed(), sa.IsVClosed()

    def is_planar(self, tol=TOLERANCE):
        """checks if the surface is planar within a tolerance
        :return: bool, gp_Pln
        """
        is_planar_surface = GeomLib_IsPlanarSurface(self.surface, tol)
        return is_planar_surface.IsPlanar()

    def is_trimmed(self):
        """
        :return: True if the Wire delimiting the Face lies on the bounds
        of the surface
        if this is not the case, the wire represents a contour that delimits
        the face [ think cookie cutter ]
        and implies that the surface is trimmed
        """
        _round = lambda x: round(x, 3)
        a = map(_round, breptools_UVBounds(self))
        b = map(_round, self.adaptor.Surface().Surface().Bounds())
        if a != b:
            print("a,b", a, b)
            return True
        return False

    def on_trimmed(self, u, v):
        """tests whether the surface at the u,v parameter has been trimmed"""
        if self._classify_uv is None:
            self._classify_uv = BRepTopAdaptor_FClass2d(self, 1e-9)
        uv = gp_Pnt2d(u, v)
        if self._classify_uv.Perform(uv) == TopAbs_IN:
            return True
        else:
            return False

    def parameter_to_point(self, u, v):
        """returns the coordinate at u,v"""
        return self.surface.Value(u, v)

    def point_to_parameter(self, pt):
        """
        returns the uv value of a point on a surface
        @param pt:
        """
        sas = ShapeAnalysis_Surface(self.surface)
        uv = sas.ValueOfUV(pt, self.tolerance)
        return uv.Coord()

    def continuity_edge_face(self, edge, face):
        """
        compute the continuity between two faces at :edge:

        :param edge: an Edge or TopoDS_Edge from :face:
        :param face: a Face or TopoDS_Face
        :return: bool, GeomAbs_Shape if it has continuity, otherwise
         False, None
        """
        bt = BRep_Tool()
        if bt.HasContinuity(edge, self, face):
            continuity = bt.Continuity(edge, self, face)
            return True, continuity
        else:
            return False, None

    # ===========================================================================
    #    Surface.project
    #    project curve, point on face
    # ===========================================================================

    def project_vertex(self, pnt, tol=TOLERANCE):
        """projects self with a point, curve, edge, face, solid
        method wraps dealing with the various topologies

        if other is a point:
            returns uv, point

        """
        if isinstance(pnt, TopoDS_Vertex):
            pnt = BRep_Tool.Pnt(pnt)

        proj = GeomAPI_ProjectPointOnSurf(pnt, self.surface, tol)
        uv = proj.LowerDistanceParameters()
        proj_pnt = proj.NearestPoint()

        return uv, proj_pnt

    def project_curve(self, other):
        # this way Geom_Circle and alike are valid too
        if (
            isinstance(other, TopoDS_Edge)
            or isinstance(other, Geom_Curve)
            or issubclass(other, Geom_Curve)
        ):
            # convert edge to curve
            first, last = topexp.FirstVertex(other), topexp.LastVertex(other)
            lbound, ubound = BRep_Tool().Parameter(first, other), BRep_Tool().Parameter(
                last, other
            )
            other = BRep_Tool.Curve(other, lbound, ubound)
            return geomprojlib.Project(other, self.surface)

    def project_edge(self, edg):
        if hasattr(edg, "adaptor"):
            return self.project_curve(self, self.adaptor)
        return self.project_curve(self, to_adaptor_3d(edg))

    def iso_curve(self, u_or_v, param):
        """
        get the iso curve from a u,v + parameter
        :param u_or_v:
        :param param:
        :return:
        """
        uv = 0 if u_or_v == "u" else 1
        iso = Adaptor3d_IsoCurve(self.adaptor, uv, param)
        return iso

    def edges(self):
        return [Edge(i) for i in WireExplorer(next(self.topo.wires())).ordered_edges()]

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere

    sph = BRepPrimAPI_MakeSphere(1, 1).Face()
    fc = Face(sph)
    print(fc.is_trimmed())
    print(fc.is_planar())


================================================
File: CadSeqProc/OCCUtils/shell.py
================================================
##Copyright 2008-2015 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>

from OCC.Core.TopoDS import TopoDS_Shell
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Shell

from OCCUtils.Topology import Topo
from OCCUtils.base import BaseObject, GlobalProperties


class Shell(TopoDS_Shell, BaseObject):
    _n = 0

    def __init__(self, shell):
        assert isinstance(shell, TopoDS_Shell), (
            "need a TopoDS_Shell, got a %s" % shell.__class__
        )
        assert not shell.IsNull()
        super(Shell, self).__init__()
        BaseObject.__init__(self, "shell")
        # we need to copy the base shape using the following three
        # lines
        assert self.IsNull()
        self.TShape(shell.TShape())
        self.Location(shell.Location())
        self.Orientation(shell.Orientation())
        assert not self.IsNull()

        self.GlobalProperties = GlobalProperties(self)
        self._n += 1

    def analyse(self):
        """

        :return:
        """
        ss = ShapeAnalysis_Shell(self)
        if ss.HasFreeEdges():
            bad_edges = [e for e in Topo(ss.BadEdges()).edges()]
        return bad_edges

    def Faces(self):
        """

        :return:
        """
        return Topo(self, True).faces()

    def Wires(self):
        """
        :return:
        """
        return Topo(self, True).wires()

    def Edges(self):
        return Topo(self, True).edges()


================================================
File: CadSeqProc/OCCUtils/solid.py
================================================
##Copyright 2008-2013 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>

from OCC.Core.TopoDS import TopoDS_Solid

from OCCUtils.Topology import Topo
from OCCUtils.base import GlobalProperties, BaseObject
from OCCUtils.shell import Shell


class Solid(TopoDS_Solid, BaseObject):
    def __init__(self, solid):
        assert isinstance(solid, TopoDS_Solid), (
            "need a TopoDS_Solid, got a %s" % solid.__class__
        )
        assert not solid.IsNull()
        super(Solid, self).__init__()
        BaseObject.__init__(self, "solid")
        # we need to copy the base shape using the following three
        # lines
        assert self.IsNull()
        self.TShape(solid.TShape())
        self.Location(solid.Location())
        self.Orientation(solid.Orientation())
        assert not self.IsNull()

        self.GlobalProperties = GlobalProperties(self)

    def shells(self):
        return (Shell(sh) for sh in Topo(self))


================================================
File: CadSeqProc/OCCUtils/types_lut.py
================================================
##Copyright 2008-2015 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>

from OCC.Core.BRepCheck import *
from OCC.Core.GeomAbs import *
from OCC.Core.TopoDS import topods, TopoDS_Shape
from OCC.Core.BRep import BRep_Tool_Surface
from OCC.Core.TopAbs import *
from OCC.Core.Geom import Geom_CylindricalSurface, Geom_Plane


class ShapeToTopology(object):
    """
    looks up the topology type and returns the corresponding topological entity
    """

    def __init__(self):
        self.topoTypes = {
            TopAbs_VERTEX: topods.Vertex,
            TopAbs_EDGE: topods.Edge,
            TopAbs_FACE: topods.Face,
            TopAbs_WIRE: topods.Wire,
            TopAbs_SHELL: topods.Shell,
            TopAbs_SOLID: topods.Solid,
            TopAbs_COMPOUND: topods.Compound,
            TopAbs_COMPSOLID: topods.CompSolid,
        }

    def __call__(self, shape):
        if isinstance(shape, TopoDS_Shape):
            return self.topoTypes[shape.ShapeType()](shape)
        else:
            raise AttributeError("shape has not method `ShapeType`")

    def __getitem__(self, item):
        return self(item)


class EnumLookup(object):
    """
    perform bi-directional lookup of Enums'...
    """

    def __init__(self, li_in, li_out):
        self.d = {}
        for a, b in zip(li_in, li_out):
            self.d[a] = b
            self.d[b] = a

    def __getitem__(self, item):
        return self.d[item]


_curve_typesA = (
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_Hyperbola,
    GeomAbs_Parabola,
    GeomAbs_BezierCurve,
    GeomAbs_BSplineCurve,
    GeomAbs_OtherCurve,
)
_curve_typesB = (
    "line",
    "circle",
    "ellipse",
    "hyperbola",
    "parabola",
    "bezier",
    "spline",
    "other",
)
_surface_typesA = (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
)
_surface_typesB = (
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "bezier",
    "spline",
    "revolution",
    "extrusion",
    "offset",
    "other",
)


_stateA = ("in", "out", "on", "unknown")
_stateB = (TopAbs_IN, TopAbs_OUT, TopAbs_ON, TopAbs_UNKNOWN)


_orientA = ["TopAbs_FORWARD", "TopAbs_REVERSED", "TopAbs_INTERNAL", "TopAbs_EXTERNAL"]
_orientB = [TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_INTERNAL, TopAbs_EXTERNAL]


_topoTypesA = [
    "vertex",
    "edge",
    "wire",
    "face",
    "shell",
    "solid",
    "compsolid",
    "compound",
    "shape",
]
_topoTypesB = [
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_WIRE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_COMPSOLID,
    TopAbs_COMPOUND,
    TopAbs_SHAPE,
]


_geom_types_a = [
    "line",
    "circle",
    "ellipse",
    "hyperbola",
    "parabola",
    "beziercurve",
    "bsplinecurve",
    "othercurve",
]
_geom_types_b = [
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_Hyperbola,
    GeomAbs_Parabola,
    GeomAbs_BezierCurve,
    GeomAbs_BSplineCurve,
    GeomAbs_OtherCurve,
]


# TODO: make a function that generalizes this, there is absolutely
# no need for 2 lists to define an EnumLookup


def fix_formatting(_str):
    return [i.strip() for i in _str.split(",")]


_brep_check_a = fix_formatting(
    "NoError, InvalidPointOnCurve,\
InvalidPointOnCurveOnSurface, InvalidPointOnSurface,\
No3DCurve, Multiple3DCurve, Invalid3DCurve, NoCurveOnSurface,\
InvalidCurveOnSurface, InvalidCurveOnClosedSurface, InvalidSameRangeFlag,\
InvalidSameParameterFlag,\
InvalidDegeneratedFlag, FreeEdge, InvalidMultiConnexity, InvalidRange,\
EmptyWire, RedundantEdge, SelfIntersectingWire, NoSurface,\
InvalidWire, RedundantWire, IntersectingWires, InvalidImbricationOfWires,\
EmptyShell, RedundantFace, UnorientableShape, NotClosed,\
NotConnected, SubshapeNotInShape, BadOrientation, BadOrientationOfSubshape,\
InvalidToleranceValue, CheckFail"
)

_brep_check_b = [
    BRepCheck_NoError,
    BRepCheck_InvalidPointOnCurve,
    BRepCheck_InvalidPointOnCurveOnSurface,
    BRepCheck_InvalidPointOnSurface,
    BRepCheck_No3DCurve,
    BRepCheck_Multiple3DCurve,
    BRepCheck_Invalid3DCurve,
    BRepCheck_NoCurveOnSurface,
    BRepCheck_InvalidCurveOnSurface,
    BRepCheck_InvalidCurveOnClosedSurface,
    BRepCheck_InvalidSameRangeFlag,
    BRepCheck_InvalidSameParameterFlag,
    BRepCheck_InvalidDegeneratedFlag,
    BRepCheck_FreeEdge,
    BRepCheck_InvalidMultiConnexity,
    BRepCheck_InvalidRange,
    BRepCheck_EmptyWire,
    BRepCheck_RedundantEdge,
    BRepCheck_SelfIntersectingWire,
    BRepCheck_NoSurface,
    BRepCheck_InvalidWire,
    BRepCheck_RedundantWire,
    BRepCheck_IntersectingWires,
    BRepCheck_InvalidImbricationOfWires,
    BRepCheck_EmptyShell,
    BRepCheck_RedundantFace,
    BRepCheck_UnorientableShape,
    BRepCheck_NotClosed,
    BRepCheck_NotConnected,
    BRepCheck_SubshapeNotInShape,
    BRepCheck_BadOrientation,
    BRepCheck_BadOrientationOfSubshape,
    BRepCheck_InvalidToleranceValue,
    BRepCheck_CheckFail,
]

brepcheck_lut = EnumLookup(_brep_check_a, _brep_check_b)
curve_lut = EnumLookup(_curve_typesA, _curve_typesB)
surface_lut = EnumLookup(_surface_typesA, _surface_typesB)
state_lut = EnumLookup(_stateA, _stateB)
orient_lut = EnumLookup(_orientA, _orientB)
topo_lut = EnumLookup(_topoTypesA, _topoTypesB)
shape_lut = ShapeToTopology()
geom_lut = EnumLookup(_geom_types_a, _geom_types_b)

# todo: refactor, these classes have been moved from the "Topology" directory
# which had too many overlapping methods & classes, that are
# now part of the KBE module...
# still need to think what to do with these...
# what_is_face should surely become a lut [ geom_lut? ]
# i'm not sure whether casting to a gp_* is useful...

classes = dir()
geom_classes = []
for elem in classes:
    if elem.startswith("Geom") and not "swig" in elem:
        geom_classes.append(elem)


def what_is_face(face):
    """Returns all class names for which this class can be downcasted"""
    if not face.ShapeType() == TopAbs_FACE:
        print("%s is not a TopAbs_FACE. Conversion impossible")
        return None
    hs = BRep_Tool_Surface(face)
    obj = hs.GetObject()
    result = []
    for elem in classes:
        if elem.startswith("Geom") and not "swig" in elem:
            geom_classes.append(elem)
    # Run the test for each class
    for geom_class in geom_classes:
        if obj.IsKind(geom_class) and not geom_class in result:
            result.append(geom_class)
    return result


def face_is_plane(face):
    """Returns True if the TopoDS_Shape is a plane, False otherwise"""
    hs = BRep_Tool_Surface(face)
    downcast_result = Geom_Plane().DownCast(hs)
    # the handle is null if downcast failed or is not possible,
    # that is to say the face is not a plane
    if downcast_result.IsNull():
        return False
    else:
        return True


def shape_is_cylinder(face):
    """Returns True is the TopoDS_Shape is a cylinder, False otherwise"""
    hs = BRep_Tool_Surface(face)
    downcast_result = Geom_CylindricalSurface().DownCast(hs)
    if downcast_result.IsNull():
        return False
    else:
        return True


================================================
File: CadSeqProc/OCCUtils/vertex.py
================================================
##Copyright 2008-2013 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_XYZ, gp_Pnt2d
from OCC.Core.TopoDS import TopoDS_Vertex
from OCC.Core.ShapeBuild import ShapeBuild_ReShape

from OCCUtils.base import BaseObject
from OCCUtils.Construct import make_vertex


class Vertex(TopoDS_Vertex, BaseObject):
    """
    wraps gp_Pnt
    """

    _n = 0

    def __init__(self, x, y, z):
        super(Vertex, self).__init__()
        """Constructor for KbeVertex"""
        BaseObject.__init__(self, name="Vertex #{0}".format(self._n))

        self._n += 1  # should be a property of KbeObject
        self._pnt = gp_Pnt(x, y, z)
        self._vertex = make_vertex(self._pnt)
        TopoDS_Vertex.__init__(self, self._vertex)

    def _update(self):
        """ """
        # TODO: perhaps should take an argument until which topological level
        # topological entities bound to the vertex should be updated too...
        reshape = ShapeBuild_ReShape()
        reshape.Replace(self._vertex, make_vertex(self._pnt))

    @staticmethod
    def from_pnt(cls, pnt):
        x, y, z = pnt.X(), pnt.Y(), pnt.Z()
        return cls(x, y, z)

    @property
    def x(self):
        return self._pnt.X()

    @x.setter
    def x(self, val):
        self._pnt.SetX(val)
        self._update()

    @property
    def y(self):
        return self._pnt.Y()

    @y.setter
    def y(self, val):
        self._pnt.SetY(val)
        self._update()

    @property
    def z(self):
        return self._pnt.Z()

    @z.setter
    def z(self, val):
        self._pnt.SetZ(val)
        self._update()

    @property
    def xyz(self):
        return self._pnt.Coord()

    @xyz.setter
    def xyz(self, *val):
        self._pnt.SetXYZ(*val)
        self._update()

    def __repr__(self):
        return self.name

    @property
    def as_vec(self):
        """returns a gp_Vec version of self"""
        return gp_Vec(*self._pnt.Coord())

    @property
    def as_dir(self):
        """returns a gp_Dir version of self"""
        return gp_Dir(*self._pnt.Coord())

    @property
    def as_xyz(self):
        """returns a gp_XYZ version of self"""
        return gp_XYZ(*self._pnt.Coord())

    @property
    def as_pnt(self):
        return self._pnt

    @property
    def as_2d(self):
        """returns a gp_Pnt2d version of self"""
        return gp_Pnt2d(*self._pnt.Coord()[:2])


================================================
File: CadSeqProc/OCCUtils/wire.py
================================================
#!/usr/bin/env python

##Copyright 2008-2013 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>

from OCC.Core.TopoDS import TopoDS_Wire

from OCCUtils.base import BaseObject


class Wire(TopoDS_Wire, BaseObject):
    def __init__(self, wire):
        """ """
        assert isinstance(wire, TopoDS_Wire), (
            "need a TopoDS_Wire, got a %s" % wire.__class__
        )
        assert not wire.IsNull()
        super(Wire, self).__init__()
        BaseObject.__init__(self, "wire")
        # we need to copy the base shape using the following three
        # lines
        assert self.IsNull()
        self.TShape(wire.TShape())
        self.Location(wire.Location())
        self.Orientation(wire.Orientation())
        assert not self.IsNull()


================================================
File: CadSeqProc/enhanced_geometry/__init__.py
================================================
"""
Enhanced geometry module for CAD generation.
Provides support for both geometric primitives and organic shapes.
"""

from .base import Point, GeometricEntity
from .nurbs import NURBSCurve, NURBSSurface
from .organic import (
    OrganicSurface,
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

================================================
File: CadSeqProc/enhanced_geometry/base.py
================================================
"""
Base classes and interfaces for enhanced geometry system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, cast
import numpy as np

class Point:
    """3D point representation."""
    
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Point':
        """Create point from numpy array."""
        return cls(arr[0], arr[1], arr[2])
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate distance to another point."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

class BoundingBox:
    """Axis-aligned bounding box."""
    
    def __init__(self, min_point: Point, max_point: Point):
        self.min_point = min_point
        self.max_point = max_point
    
    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Get box dimensions (width, height, depth)."""
        return (
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z
        )
    
    @property
    def volume(self) -> float:
        """Calculate box volume."""
        w, h, d = self.dimensions
        return w * h * d
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside box."""
        return (
            self.min_point.x <= point.x <= self.max_point.x and
            self.min_point.y <= point.y <= self.max_point.y and
            self.min_point.z <= point.z <= self.max_point.z
        )

class BaseGeometry(ABC):
    """Abstract base class for all geometric entities."""
    
    @abstractmethod
    def analyze_thickness(self) -> float:
        """Analyze minimum wall thickness."""
        pass
    
    @abstractmethod
    def analyze_overhangs(self) -> float:
        """Analyze maximum overhang angle."""
        pass
    
    @abstractmethod
    def analyze_stress_points(self) -> List[Tuple[float, float, float]]:
        """Analyze potential stress points."""
        pass
    
    @abstractmethod
    def thicken_walls(self, min_thickness: float) -> 'BaseGeometry':
        """Thicken walls to meet minimum thickness."""
        pass
    
    @abstractmethod
    def reduce_overhangs(self, max_angle: float) -> 'BaseGeometry':
        """Reduce overhangs to meet maximum angle."""
        pass
    
    @abstractmethod
    def reinforce_weak_points(self) -> 'BaseGeometry':
        """Reinforce identified weak points."""
        pass

class GeometricEntity(BaseGeometry):
    """Base class for geometric entities with common functionality."""
    
    def __init__(self):
        self._bounding_box: Optional[BoundingBox] = None
        self._volume: Optional[float] = None
        self._surface_area: Optional[float] = None
    
    @property
    def bounding_box(self) -> BoundingBox:
        """Get entity's bounding box."""
        if self._bounding_box is None:
            self._compute_bounding_box()
        return cast(BoundingBox, self._bounding_box)
    
    @property
    def volume(self) -> float:
        """Get entity's volume."""
        if self._volume is None:
            self._compute_volume()
        return cast(float, self._volume)
    
    @property
    def surface_area(self) -> float:
        """Get entity's surface area."""
        if self._surface_area is None:
            self._compute_surface_area()
        return cast(float, self._surface_area)
    
    @abstractmethod
    def _compute_bounding_box(self) -> None:
        """Compute bounding box."""
        pass
    
    @abstractmethod
    def _compute_volume(self) -> None:
        """Compute volume."""
        pass
    
    @abstractmethod
    def _compute_surface_area(self) -> None:
        """Compute surface area."""
        pass
    
    def analyze_thickness(self) -> float:
        """Default thickness analysis."""
        # Implement basic thickness analysis
        return float('inf')
    
    def analyze_overhangs(self) -> float:
        """Default overhang analysis."""
        # Implement basic overhang analysis
        return 0.0
    
    def analyze_stress_points(self) -> List[Tuple[float, float, float]]:
        """Default stress point analysis."""
        # Implement basic stress analysis
        return []
    
    def thicken_walls(self, min_thickness: float) -> 'GeometricEntity':
        """Default wall thickening."""
        # Implement basic wall thickening
        return self
    
    def reduce_overhangs(self, max_angle: float) -> 'GeometricEntity':
        """Default overhang reduction."""
        # Implement basic overhang reduction
        return self
    
    def reinforce_weak_points(self) -> 'GeometricEntity':
        """Default weak point reinforcement."""
        # Implement basic reinforcement
        return self

class NURBSEntity(GeometricEntity):
    """Base class for NURBS representations."""
    
    def __init__(self, 
                 control_points: List[Point],
                 weights: Optional[List[float]] = None,
                 knots: Optional[List[float]] = None,
                 degree: int = 3):
        self.control_points = control_points
        self.weights = weights if weights is not None else [1.0] * len(control_points)
        self.knots = knots
        self.degree = degree
        self._validate()
    
    def _validate(self):
        """Validate NURBS parameters."""
        if len(self.control_points) < self.degree + 1:
            raise ValueError("Not enough control points for specified degree")
        if len(self.weights) != len(self.control_points):
            raise ValueError("Number of weights must match number of control points")
        if self.knots is not None and len(self.knots) != len(self.control_points) + self.degree + 1:
            raise ValueError("Invalid number of knots")

    def evaluate(self, u: float) -> Point:
        """Evaluate NURBS at parameter u."""
        # Basic implementation - will be enhanced
        if not (0 <= u <= 1):
            raise ValueError("Parameter u must be in [0,1]")
        # TODO: Implement proper NURBS evaluation
        return self.control_points[0]  # Placeholder

class OrganicShape(GeometricEntity):
    """Base class for organic shapes."""
    
    def __init__(self, 
                 control_surfaces: List[NURBSEntity],
                 deformation_params: Optional[dict] = None):
        self.control_surfaces = control_surfaces
        self.deformation_params = deformation_params or {}
    
    def apply_deformation(self, deformation_type: str, params: dict) -> 'OrganicShape':
        """Apply deformation to shape."""
        # TODO: Implement deformation types (twist, bend, etc.)
        return self

    def to_nurbs(self) -> NURBSEntity:
        """Convert to NURBS representation."""
        # TODO: Implement conversion to NURBS
        return self.control_surfaces[0]

class ParametricEntity(GeometricEntity):
    """Base class for parametric entities."""
    
    def __init__(self, parameters: dict, constraints: Optional[dict] = None):
        self.parameters = parameters
        self.constraints = constraints or {}
        self._validate_constraints()
    
    def _validate_constraints(self):
        """Validate parameter constraints."""
        for param, value in self.parameters.items():
            if param in self.constraints:
                constraint = self.constraints[param]
                if 'min' in constraint and value < constraint['min']:
                    raise ValueError(f"Parameter {param} below minimum value")
                if 'max' in constraint and value > constraint['max']:
                    raise ValueError(f"Parameter {param} above maximum value")
    
    def update_parameter(self, param: str, value: float):
        """Update parameter value with validation."""
        self.parameters[param] = value
        self._validate_constraints()

class GeometryFactory:
    """Factory for creating geometric entities."""
    
    @staticmethod
    def create_nurbs_curve(control_points: List[Point], 
                          weights: Optional[List[float]] = None,
                          degree: int = 3) -> NURBSEntity:
        """Create a NURBS curve."""
        return NURBSEntity(control_points, weights, degree=degree)
    
    @staticmethod
    def create_organic_shape(control_points: List[List[Point]],
                           deformation_params: Optional[dict] = None) -> OrganicShape:
        """Create an organic shape."""
        control_surfaces = [
            NURBSEntity(points) for points in control_points
        ]
        return OrganicShape(control_surfaces, deformation_params) 

================================================
File: CadSeqProc/enhanced_geometry/factory.py
================================================
"""
Factory for creating geometric and organic shapes.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from .base import BaseGeometry, Point, BoundingBox
from .nurbs import NURBSCurve, NURBSSurface
from .organic import OrganicSurface
from ..utility.logger import setup_logger

logger = setup_logger(__name__)

class OrganicShapeFactory:
    """Factory for creating organic shapes."""
    
    def create_from_params(self, params: Dict[str, Any]) -> BaseGeometry:
        """Create shape from parameters."""
        try:
            # Extract basic parameters
            shape_type = params.get('shape_type', 'generic')
            scale = params.get('scale_factor', 1.0)
            detail_level = params.get('detail_level', 0.5)
            
            if shape_type == 'flower':
                return self._create_flower(
                    num_petals=params.get('num_petals', 5),
                    petal_length=0.5 * scale,
                    petal_width=0.2 * scale,
                    detail_level=detail_level
                )
            elif shape_type == 'leaf':
                return self._create_leaf(
                    length=1.0 * scale,
                    width=0.5 * scale,
                    detail_level=detail_level
                )
            else:
                return self._create_generic_organic(
                    scale=scale,
                    detail_level=detail_level
                )
                
        except Exception as e:
            logger.error(f"Error creating organic shape: {str(e)}")
            # Return a simple fallback shape
            return self._create_generic_organic(scale=1.0, detail_level=0.3)
    
    def _create_flower(self, num_petals: int, petal_length: float,
                      petal_width: float, detail_level: float) -> OrganicSurface:
        """Create a flower shape."""
        # Create center
        center_radius = petal_length * 0.2
        center = self._create_center(radius=center_radius, detail_level=detail_level)
        
        # Create petals
        petals = []
        for i in range(num_petals):
            angle = (2 * np.pi * i) / num_petals
            petal = self._create_petal(
                length=petal_length,
                width=petal_width,
                angle=angle,
                detail_level=detail_level
            )
            petals.append(petal)
        
        # Combine shapes
        return OrganicSurface.combine([center] + petals)
    
    def _create_leaf(self, length: float, width: float,
                    detail_level: float) -> OrganicSurface:
        """Create a leaf shape."""
        # Create main surface
        control_points = self._generate_leaf_points(length, width)
        surface = NURBSSurface.from_points(control_points)
        
        # Add veins
        veins = self._create_leaf_veins(length, width, detail_level)
        
        # Combine and add organic deformation
        combined = OrganicSurface.from_nurbs(surface)
        for vein in veins:
            combined.add_feature(vein)
        
        return combined
    
    def _create_generic_organic(self, scale: float,
                              detail_level: float) -> OrganicSurface:
        """Create a generic organic shape."""
        # Create base sphere
        radius = 0.5 * scale
        sphere = self._create_sphere(radius)
        
        # Add organic deformation
        surface = OrganicSurface.from_nurbs(sphere)
        surface.add_random_deformation(intensity=detail_level)
        
        return surface
    
    def _create_center(self, radius: float, detail_level: float) -> OrganicSurface:
        """Create flower center."""
        # Create base sphere
        sphere = self._create_sphere(radius)
        
        # Add organic texture
        surface = OrganicSurface.from_nurbs(sphere)
        surface.add_bumps(
            count=int(20 * detail_level),
            height=radius * 0.1,
            radius=radius * 0.1
        )
        
        return surface
    
    def _create_petal(self, length: float, width: float,
                     angle: float, detail_level: float) -> OrganicSurface:
        """Create a single petal."""
        # Create base curve
        control_points = [
            [0, 0, 0],
            [length * 0.3, width * 0.5, 0],
            [length * 0.7, width * 0.5, 0],
            [length, 0, 0]
        ]
        curve = NURBSCurve.from_points(control_points)
        
        # Create surface by sweeping
        surface = curve.sweep(width)
        
        # Add organic deformation
        organic = OrganicSurface.from_nurbs(surface)
        organic.add_random_deformation(intensity=detail_level * 0.3)
        
        # Rotate to position
        organic.rotate(angle)
        
        return organic
    
    def _create_sphere(self, radius: float) -> NURBSSurface:
        """Create a NURBS sphere."""
        # Create control points for sphere
        u_count, v_count = 10, 10
        control_points = []
        
        for i in range(u_count):
            u = (i / (u_count - 1)) * 2 * np.pi
            row = []
            for j in range(v_count):
                v = (j / (v_count - 1)) * np.pi
                x = radius * np.sin(v) * np.cos(u)
                y = radius * np.sin(v) * np.sin(u)
                z = radius * np.cos(v)
                row.append([x, y, z])
            control_points.append(row)
        
        return NURBSSurface.from_points(control_points)
    
    def _generate_leaf_points(self, length: float,
                            width: float) -> List[List[List[float]]]:
        """Generate control points for a leaf shape."""
        # Create control point grid
        u_count, v_count = 5, 3
        control_points = []
        
        for i in range(u_count):
            u = i / (u_count - 1)
            row = []
            for j in range(v_count):
                v = j / (v_count - 1) - 0.5
                
                # Create leaf shape
                x = length * u
                y = width * v * (1 - u) * (1 - u)
                z = 0.0
                
                row.append([x, y, z])
            control_points.append(row)
        
        return control_points
    
    def _create_leaf_veins(self, length: float, width: float,
                          detail_level: float) -> List[NURBSCurve]:
        """Create leaf vein curves."""
        veins = []
        
        # Main vein
        main_vein = NURBSCurve.from_points([
            [0, 0, 0],
            [length * 0.3, 0, 0],
            [length * 0.7, 0, 0],
            [length, 0, 0]
        ])
        veins.append(main_vein)
        
        # Side veins
        num_side_veins = int(5 * detail_level)
        for i in range(num_side_veins):
            t = (i + 1) / (num_side_veins + 1)
            start = main_vein.point_at(t)
            
            # Create side vein on both sides
            for side in [-1, 1]:
                end = [
                    start[0] + length * 0.2,
                    side * width * 0.4 * (1 - t),
                    0
                ]
                vein = NURBSCurve.from_points([start, end])
                veins.append(vein)
        
        return veins 

================================================
File: CadSeqProc/enhanced_geometry/integration.py
================================================
"""
Integration module to connect enhanced geometry system with existing CAD model.
Provides conversion and adaptation layers.
"""

import numpy as np
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from .base import Point, GeometricEntity
from .nurbs import NURBSCurve, NURBSSurface
from .organic import OrganicSurface
from .factory import OrganicShapeFactory
from .parametric import (
    ParametricCurve, FlowerPetal, PatternGenerator,
    RoseCurve, BezierCurve, EpicycloidCurve,
    PatternAnalyzer, CombinedPattern
)

class TextParser:
    """Advanced text parsing for shape parameters."""
    
    @staticmethod
    def extract_number(text: str, default: float) -> float:
        """Extract first number from text or return default."""
        if match := re.search(r'(\d*\.?\d+)', text):
            return float(match.group(1))
        return default
    
    @staticmethod
    def extract_size_modifier(text: str) -> float:
        """Extract size modifier from descriptive text."""
        size_modifiers = {
            'tiny': 0.5,
            'small': 0.7,
            'medium': 1.0,
            'large': 1.5,
            'huge': 2.0,
            'giant': 2.5
        }
        
        for word, modifier in size_modifiers.items():
            if word in text.lower():
                return modifier
        return 1.0
    
    @staticmethod
    def extract_pattern_type(text: str) -> str:
        """Determine pattern type from description using AI analysis."""
        # Use AI-driven pattern analysis
        requirements = PatternAnalyzer.analyze_shape_requirements(text)
        return PatternAnalyzer.get_optimal_pattern(requirements)
    
    @staticmethod
    def extract_curve_type(text: str) -> str:
        """Determine the best curve type for the shape."""
        if any(word in text.lower() for word in ['rose', 'flower', 'petal']):
            return 'rose'
        elif any(word in text.lower() for word in ['spiral', 'coil']):
            return 'spiral'
        elif any(word in text.lower() for word in ['complex', 'ornate']):
            return 'epicycloid'
        elif any(word in text.lower() for word in ['smooth', 'curved']):
            return 'bezier'
        return 'default'
    
    @staticmethod
    def extract_shape_complexity(text: str) -> Dict[str, float]:
        """Analyze shape complexity requirements."""
        complexity = {
            'detail_level': 1.0,  # Base detail level
            'symmetry': 1.0,      # Perfect symmetry
            'variation': 0.0,     # No random variation
            'layers': 1           # Single layer
        }
        
        # Adjust detail level
        if any(word in text.lower() for word in ['detailed', 'complex', 'intricate']):
            complexity['detail_level'] = 1.5
        elif any(word in text.lower() for word in ['simple', 'basic']):
            complexity['detail_level'] = 0.7
        
        # Adjust symmetry
        if any(word in text.lower() for word in ['irregular', 'natural', 'organic']):
            complexity['symmetry'] = 0.8
        elif any(word in text.lower() for word in ['perfect', 'exact']):
            complexity['symmetry'] = 1.0
        
        # Adjust variation
        if any(word in text.lower() for word in ['random', 'varied', 'diverse']):
            complexity['variation'] = 0.3
        
        # Adjust layers
        if 'layered' in text.lower():
            complexity['layers'] = 2
        elif 'multi-layered' in text.lower():
            complexity['layers'] = 3
        
        return complexity
    
    @staticmethod
    def extract_center_type(text: str) -> str:
        """Determine center type from description."""
        if any(word in text.lower() for word in ['spiral', 'swirl']):
            return 'spiral'
        elif any(word in text.lower() for word in ['complex', 'intricate']):
            return 'lissajous'
        return 'circle'
    
    @staticmethod
    def extract_color_hints(text: str) -> List[str]:
        """Extract color-related words from text."""
        color_words = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'white', 'black']
        return [word for word in color_words if word in text.lower()]

class GeometryAdapter:
    """Adapter to convert between enhanced geometry and CAD model formats."""
    
    @staticmethod
    def to_cad_sequence(
        entities: List[Union[GeometricEntity, OrganicSurface]]
    ) -> Dict[str, Any]:
        """Convert geometric entities to CAD sequence format."""
        sequence = {
            'type': 'composite',
            'operations': []
        }
        
        for entity in entities:
            if isinstance(entity, NURBSCurve):
                points = entity.sample_points(20)
                sequence['operations'].append({
                    'type': 'curve',
                    'points': [[p.x, p.y, p.z] for p in points],
                    'closed': False
                })
            elif isinstance(entity, NURBSSurface):
                points_2d = entity.sample_points_grid(20, 20)
                sequence['operations'].append({
                    'type': 'surface',
                    'points': [[[p.x, p.y, p.z] for p in row] for row in points_2d],
                    'closed_u': True,
                    'closed_v': True
                })
            elif isinstance(entity, OrganicSurface):
                for surface in entity.control_surfaces:
                    points_2d = surface.sample_points_grid(20, 20)
                    sequence['operations'].append({
                        'type': 'surface',
                        'points': [[[p.x, p.y, p.z] for p in row] for row in points_2d],
                        'closed_u': True,
                        'closed_v': True
                    })
        
        return sequence
    
    @staticmethod
    def from_cad_sequence(sequence: Dict[str, Any]) -> List[GeometricEntity]:
        """Convert CAD sequence back to geometric entities."""
        entities = []
        
        for op in sequence.get('operations', []):
            if op['type'] == 'curve':
                points = [Point(*p) for p in op['points']]
                entities.append(NURBSCurve.from_points(points))
            elif op['type'] == 'surface':
                points_2d = [[Point(*p) for p in row] for row in op['points']]
                entities.append(NURBSSurface.from_points(points_2d))
        
        return entities

class ShapeGenerator:
    """High-level interface for generating shapes from text descriptions."""
    
    @staticmethod
    def parse_flower_description(text: str) -> Dict[str, Any]:
        """Parse text description for flower parameters."""
        parser = TextParser()
        size_modifier = parser.extract_size_modifier(text)
        pattern_type = parser.extract_pattern_type(text)
        curve_type = parser.extract_curve_type(text)
        complexity = parser.extract_shape_complexity(text)
        
        # Base parameters
        params = {
            'num_petals': 5,
            'petal_length': 1.0 * size_modifier,
            'petal_width': 0.3 * size_modifier,
            'center_radius': 0.2 * size_modifier,
            'petal_curve_factor': 0.3,
            'pattern_type': pattern_type,
            'center_type': 'spiral' if complexity['detail_level'] > 1.2 else 'circle',
            'complexity': complexity
        }
        
        # Extract number of petals
        if match := re.search(r'(\d+)\s*petals?', text.lower()):
            params['num_petals'] = int(match.group(1))
        
        # Adjust curve factor based on descriptive words
        if any(word in text.lower() for word in ['wavy', 'curly', 'curved']):
            params['petal_curve_factor'] *= 1.5
        elif any(word in text.lower() for word in ['straight', 'flat']):
            params['petal_curve_factor'] *= 0.5
        
        # Add randomness based on complexity
        if complexity['variation'] > 0:
            params['petal_length'] *= (1 + 0.2 * np.random.random() - 0.1)
            params['petal_width'] *= (1 + 0.2 * np.random.random() - 0.1)
            params['petal_curve_factor'] *= (1 + 0.3 * np.random.random() - 0.15)
        
        return params
    
    @staticmethod
    def parse_tree_description(text: str) -> Dict[str, Any]:
        """Parse text description for tree parameters."""
        parser = TextParser()
        size_modifier = parser.extract_size_modifier(text)
        
        params = {
            'trunk_height': 2.0 * size_modifier,
            'trunk_radius': 0.2 * size_modifier,
            'num_branches': 5,
            'leaf_size': 0.5 * size_modifier
        }
        
        # Adjust trunk height
        if 'tall' in text.lower():
            params['trunk_height'] *= 1.5
        elif 'short' in text.lower():
            params['trunk_height'] *= 0.7
        
        # Adjust trunk thickness
        if any(word in text.lower() for word in ['thick', 'wide', 'broad']):
            params['trunk_radius'] *= 1.5
        elif any(word in text.lower() for word in ['thin', 'narrow', 'slender']):
            params['trunk_radius'] *= 0.7
        
        # Adjust number of branches
        if 'many branches' in text.lower():
            params['num_branches'] = 8
        elif 'few branches' in text.lower():
            params['num_branches'] = 3
        
        # Adjust leaf size
        if 'large leaves' in text.lower():
            params['leaf_size'] *= 1.5
        elif 'small leaves' in text.lower():
            params['leaf_size'] *= 0.7
        
        return params
    
    @staticmethod
    def create_flower_from_text(text: str) -> List[OrganicSurface]:
        """Generate a flower based on text description."""
        params = ShapeGenerator.parse_flower_description(text)
        requirements = PatternAnalyzer.analyze_shape_requirements(text)
        
        # Create curves based on requirements
        curves = []
        for curve_type in requirements['curve_types']:
            curve = PatternAnalyzer.create_curve_from_type(
                curve_type,
                size=params['petal_length'],
                complexity=requirements['complexity']
            )
            curves.append(curve)
        
        # Use default curve if none specified
        if not curves:
            curves = [FlowerPetal(
                length=params['petal_length'],
                width=params['petal_width'],
                curve_factor=params['petal_curve_factor']
            )]
        
        # Generate patterns for each curve
        patterns = []
        for curve in curves:
            if requirements['pattern_type'] == 'fractal':
                pattern = PatternGenerator.fractal_pattern(
                    curve,
                    params['num_petals'],
                    scale_range=(0.3, 1.0)
                )
            elif requirements['pattern_type'] == 'radial_wave':
                pattern = PatternGenerator.radial_wave_pattern(
                    curve,
                    params['num_petals'],
                    radius=params['center_radius'] * 2
                )
            else:
                pattern = PatternGenerator.circular_pattern(
                    curve,
                    params['num_petals'],
                    radius=params['center_radius']
                )
            patterns.append(pattern)
        
        # Combine patterns if specified
        if requirements['combination_mode'] == 'blend':
            combined = CombinedPattern(patterns)
            final_pattern = combined.blend(0.5)
        else:
            # Use the most complex pattern
            pattern_complexity = [len(p) for p in patterns]
            final_pattern = patterns[pattern_complexity.index(max(pattern_complexity))]
        
        # Create surfaces
        surfaces = []
        for curve in final_pattern:
            control_points = []
            for t in np.linspace(0, 1, 10):
                points = curve.sample_points(20)
                thickness = 0.1 * (1 - t)
                offset = np.array([0, 0, thickness])
                control_points.append([Point(p.x, p.y, p.z + offset[2]) for p in points])
            surfaces.append(NURBSSurface.from_points(control_points))
        
        return [OrganicSurface([surface]) for surface in surfaces]
    
    @staticmethod
    def create_tree_from_text(text: str) -> List[OrganicSurface]:
        """Generate a tree based on text description."""
        params = ShapeGenerator.parse_tree_description(text)
        return OrganicShapeFactory.create_tree(**params)

class ModelIntegration:
    """Integration with the main CAD model."""
    
    def __init__(self):
        self.adapter = GeometryAdapter()
        self.generator = ShapeGenerator()
    
    def process_text_input(self, text: str) -> Dict[str, Any]:
        """
        Process text input to generate CAD model.
        
        Args:
            text: Text description of desired shape
            
        Returns:
            Dictionary containing:
            - cad_sequence: CAD sequence for the shape
            - metadata: Additional information about the generation
        """
        try:
            # Determine shape type from text
            if any(word in text.lower() for word in ['flower', 'petal', 'bloom', 'sunflower', 'daisy']):
                entities = self.generator.create_flower_from_text(text)
            elif any(word in text.lower() for word in ['tree', 'branch', 'leaf', 'plant']):
                entities = self.generator.create_tree_from_text(text)
            else:
                raise ValueError("Unsupported shape type in text description")
            
            cad_sequence = self.adapter.to_cad_sequence(entities)
            
            # Add metadata about the generation
            metadata = {
                'input_text': text,
                'generation_status': 'success',
                'shape_type': cad_sequence['type'],
                'parameters': {
                    'color_hints': TextParser.extract_color_hints(text),
                    'pattern_type': TextParser.extract_pattern_type(text),
                    'size_modifier': TextParser.extract_size_modifier(text)
                }
            }
            
            return {
                'cad_sequence': cad_sequence,
                'metadata': metadata
            }
        except Exception as e:
            return {
                'cad_sequence': None,
                'metadata': {
                    'input_text': text,
                    'generation_status': 'error',
                    'error_message': str(e)
                }
            }
    
    def validate_sequence(self, sequence: Dict[str, Any]) -> bool:
        """
        Validate CAD sequence before processing.
        
        Args:
            sequence: CAD sequence to validate
            
        Returns:
            True if sequence is valid
        """
        if not isinstance(sequence, dict):
            return False
        
        if 'type' not in sequence or sequence['type'] != 'composite':
            return False
        
        if 'operations' not in sequence or not isinstance(sequence['operations'], list):
            return False
        
        for op in sequence['operations']:
            if 'type' not in op:
                return False
            
            if op['type'] not in ['curve', 'surface']:
                return False
            
            if op['type'] == 'curve':
                if 'points' not in op or not isinstance(op['points'], list):
                    return False
            elif op['type'] == 'surface':
                if 'points' not in op or not isinstance(op['points'], list):
                    return False
                if not all(isinstance(row, list) for row in op['points']):
                    return False
        
        return True 

================================================
File: CadSeqProc/enhanced_geometry/intelligent_cad.py
================================================
"""
Intelligent CAD system that bridges LLM output with CAD parameters.
"""

from typing import Dict, Any, List, Optional, Union
from .base import BaseGeometry
from .factory import OrganicShapeFactory
from .llm_client import LLMClient
from ..utility.logger import setup_logger

logger = setup_logger(__name__)

class IntelligentCAD:
    """Main class for intelligent CAD operations."""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.shape_factory = OrganicShapeFactory()
        self.context = {}
        
    def _map_llm_to_parameters(self, llm_output: dict) -> dict:
        """Convert LLM output to CAD parameters"""
        param_map = {
            'density': {
                'petals': ('num_petals', lambda x: 34 + int(21 * x)),
                'holes': ('hole_count', lambda x: 1 + int(4 * x))
            },
            'size': {
                'small': ('scale_factor', 0.5),
                'large': ('scale_factor', 1.5)
            },
            'complexity': {
                'simple': ('detail_level', 0.3),
                'complex': ('detail_level', 0.8)
            }
        }
        
        cad_params = {}
        for category, values in llm_output.get('properties', {}).items():
            for prop, intensity in values.items():
                if mapping := param_map.get(category, {}).get(prop):
                    param_name, converter = mapping
                    if callable(converter):
                        cad_params[param_name] = converter(intensity)
                    else:
                        cad_params[param_name] = converter
        
        # Add geometric constraints
        cad_params.setdefault('min_wall_thickness', 0.1)
        cad_params.setdefault('max_overhang_angle', 45)
        
        return cad_params
    
    def process_design_request(self, text_input: str) -> Dict[str, Any]:
        """Process a design request from text input."""
        try:
            # Get LLM analysis
            llm_response = self.llm_client.analyze_request(text_input)
            
            # Map to CAD parameters
            cad_params = self._map_llm_to_parameters(llm_response)
            
            # Generate geometry
            geometry = self.shape_factory.create_from_params(cad_params)
            
            # Validate manufacturability
            validation_result = self.validate_design(geometry)
            
            if not validation_result['valid']:
                # Attempt to fix issues
                geometry = self.optimize_for_manufacturing(geometry, validation_result['issues'])
                validation_result = self.validate_design(geometry)
            
            return {
                'status': 'success',
                'geometry': geometry,
                'parameters': cad_params,
                'validation': validation_result
            }
            
        except Exception as e:
            logger.error(f"Error processing design request: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def validate_design(self, geometry: BaseGeometry) -> Dict[str, Any]:
        """Validate design for manufacturability."""
        issues = []
        
        # Check minimum wall thickness
        min_thickness = geometry.analyze_thickness()
        if min_thickness < 0.1:
            issues.append(f"Minimum wall thickness {min_thickness:.2f}mm is below 0.1mm")
        
        # Check overhangs
        max_overhang = geometry.analyze_overhangs()
        if max_overhang > 45:
            issues.append(f"Maximum overhang angle {max_overhang:.1f}° exceeds 45°")
        
        # Check structural integrity
        stress_points = geometry.analyze_stress_points()
        if stress_points:
            issues.append(f"Found {len(stress_points)} potential stress points")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def optimize_for_manufacturing(self, geometry: BaseGeometry, issues: List[str]) -> BaseGeometry:
        """Optimize geometry to resolve manufacturing issues."""
        optimized = geometry
        
        for issue in issues:
            if 'wall thickness' in issue.lower():
                optimized = optimized.thicken_walls(min_thickness=0.1)
            elif 'overhang' in issue.lower():
                optimized = optimized.reduce_overhangs(max_angle=45)
            elif 'stress points' in issue.lower():
                optimized = optimized.reinforce_weak_points()
        
        return optimized 

================================================
File: CadSeqProc/enhanced_geometry/llm_client.py
================================================
"""
LLM client for handling interactions with Claude 3.5.
"""

import os
from typing import Dict, Any, Optional
from ..utility.logger import setup_logger

logger = setup_logger(__name__)

class LLMClient:
    """Client for interacting with Claude 3.5."""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
    
    def analyze_request(self, text: str) -> Dict[str, Any]:
        """Analyze a design request using Claude 3.5."""
        try:
            # Format prompt for design analysis
            prompt = f"""
            Analyze this CAD design request: "{text}"
            
            Extract and provide:
            1. Shape properties and their intensities (0-1 scale)
            2. Material suggestions
            3. Manufacturing considerations
            4. Key features and dimensions
            
            Format response as JSON with this structure:
            {{
                "properties": {{
                    "density": {{"petals": 0.8, "holes": 0.3}},
                    "size": {{"small": 0.2, "large": 0.8}},
                    "complexity": {{"simple": 0.3, "complex": 0.7}}
                }},
                "material": {{
                    "primary": "PLA",
                    "alternatives": ["ABS", "PETG"]
                }},
                "manufacturing": {{
                    "process": "3D printing",
                    "considerations": [
                        "Support structures needed",
                        "Layer adhesion critical"
                    ]
                }},
                "features": [
                    {{
                        "type": "hole",
                        "purpose": "mounting",
                        "count": 4
                    }}
                ]
            }}
            """
            
            # TODO: Implement actual Claude 3.5 API call
            # For now, return mock response for development
            mock_response = {
                "properties": {
                    "density": {"petals": 0.8, "holes": 0.3},
                    "size": {"small": 0.2, "large": 0.8},
                    "complexity": {"simple": 0.3, "complex": 0.7}
                },
                "material": {
                    "primary": "PLA",
                    "alternatives": ["ABS", "PETG"]
                },
                "manufacturing": {
                    "process": "3D printing",
                    "considerations": [
                        "Support structures needed",
                        "Layer adhesion critical"
                    ]
                },
                "features": [
                    {
                        "type": "hole",
                        "purpose": "mounting",
                        "count": 4
                    }
                ]
            }
            
            return mock_response
            
        except Exception as e:
            logger.error(f"Error analyzing request: {str(e)}")
            return {
                "error": str(e),
                "properties": {},
                "material": {},
                "manufacturing": {},
                "features": []
            }
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate the structure of an LLM response."""
        required_fields = ["properties", "material", "manufacturing", "features"]
        return all(field in response for field in required_fields)
    
    def extract_parameters(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract CAD parameters from LLM response."""
        parameters = {}
        
        # Extract property-based parameters
        if "properties" in response:
            for category, values in response["properties"].items():
                for prop, intensity in values.items():
                    param_name = f"{category}_{prop}"
                    parameters[param_name] = intensity
        
        # Extract feature-based parameters
        if "features" in response:
            for feature in response["features"]:
                if "type" in feature and "count" in feature:
                    parameters[f"{feature['type']}_count"] = feature["count"]
        
        return parameters 

================================================
File: CadSeqProc/enhanced_geometry/nurbs.py
================================================
"""
NURBS (Non-Uniform Rational B-Spline) implementation.
Provides advanced curve and surface manipulation capabilities.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, cast
from .base import Point, NURBSEntity, GeometricEntity, BaseGeometry, BoundingBox

def find_span(n: int, p: int, u: float, U: List[float]) -> int:
    """
    Find the knot span index.
    
    Args:
        n: Number of control points - 1
        p: Degree of curve
        u: Parameter value
        U: Knot vector
    
    Returns:
        Knot span index
    """
    if u == U[n + 1]:
        return n
    
    low = p
    high = n + 1
    mid = (low + high) // 2
    
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    
    return mid

def basis_funs(i: int, u: float, p: int, U: List[float]) -> List[float]:
    """
    Compute the nonzero basis functions.
    
    Args:
        i: Knot span index
        u: Parameter value
        p: Degree of curve
        U: Knot vector
    
    Returns:
        List of basis function values
    """
    N = [0.0] * (p + 1)
    left = [0.0] * (p + 1)
    right = [0.0] * (p + 1)
    
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        N[j] = saved
    
    return N

class NURBSCurve(BaseGeometry):
    """NURBS curve implementation."""
    
    def __init__(self, control_points: List[List[float]], degree: int = 3,
                 weights: Optional[List[float]] = None):
        """Initialize NURBS curve."""
        super().__init__()
        self.control_points = np.array(control_points)
        self.degree = degree
        self.weights = np.ones(len(control_points)) if weights is None else np.array(weights)
        self._knot_vector = self._generate_knot_vector()
    
    @classmethod
    def from_points(cls, points: List[List[float]], degree: int = 3) -> 'NURBSCurve':
        """Create curve from points."""
        return cls(points, degree)
    
    def point_at(self, t: float) -> List[float]:
        """Evaluate curve at parameter t."""
        # Implement De Boor's algorithm
        n = len(self.control_points) - 1
        p = self.degree
        
        # Find knot span
        span = self._find_span(t)
        
        # Calculate basis functions
        N = self._basis_functions(span, t)
        
        # Calculate point
        point = np.zeros(3)
        w_sum = 0.0
        
        for i in range(p + 1):
            weight = self.weights[span - p + i]
            point += N[i] * weight * self.control_points[span - p + i]
            w_sum += N[i] * weight
        
        return (point / w_sum).tolist()
    
    def sweep(self, width: float) -> 'NURBSSurface':
        """Create surface by sweeping curve."""
        # Create surface control points by sweeping curve
        surface_points = []
        for t in np.linspace(0, 1, 10):
            curve_point = self.point_at(t)
            row = []
            for s in np.linspace(-width/2, width/2, 5):
                # Create points perpendicular to curve
                point = [
                    curve_point[0],
                    curve_point[1] + s,
                    curve_point[2]
                ]
                row.append(point)
            surface_points.append(row)
        
        return NURBSSurface(surface_points)
    
    def _generate_knot_vector(self) -> np.ndarray:
        """Generate uniform knot vector."""
        n = len(self.control_points) - 1
        p = self.degree
        m = n + p + 1
        
        knots = np.zeros(m + 1)
        for i in range(m + 1):
            if i <= p:
                knots[i] = 0
            elif i >= m - p:
                knots[i] = 1
            else:
                knots[i] = (i - p) / (m - 2 * p)
        
        return knots
    
    def _find_span(self, t: float) -> int:
        """Find knot span index."""
        n = len(self.control_points) - 1
        p = self.degree
        
        if t >= 1.0:
            return n
        
        low = p
        high = n + 1
        mid = (low + high) // 2
        
        while t < self._knot_vector[mid] or t >= self._knot_vector[mid + 1]:
            if t < self._knot_vector[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
        
        return mid
    
    def _basis_functions(self, span: int, t: float) -> np.ndarray:
        """Calculate basis functions."""
        p = self.degree
        N = np.zeros(p + 1)
        left = np.zeros(p + 1)
        right = np.zeros(p + 1)
        
        N[0] = 1.0
        for j in range(1, p + 1):
            left[j] = t - self._knot_vector[span + 1 - j]
            right[j] = self._knot_vector[span + j] - t
            saved = 0.0
            
            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            
            N[j] = saved
        
        return N
    
    def analyze_thickness(self) -> float:
        """Analyze curve thickness (always 0 for curves)."""
        return 0.0
    
    def analyze_overhangs(self) -> float:
        """Analyze curve overhangs (not applicable for curves)."""
        return 0.0
    
    def analyze_stress_points(self) -> List[Tuple[float, float, float]]:
        """Analyze curve stress points (not applicable for curves)."""
        return []
    
    def thicken_walls(self, min_thickness: float) -> 'BaseGeometry':
        """Thicken curve (not applicable)."""
        return self
    
    def reduce_overhangs(self, max_angle: float) -> 'BaseGeometry':
        """Reduce curve overhangs (not applicable)."""
        return self
    
    def reinforce_weak_points(self) -> 'BaseGeometry':
        """Reinforce curve weak points (not applicable)."""
        return self

class NURBSSurface(BaseGeometry):
    """NURBS surface implementation."""
    
    def __init__(self, control_points: List[List[List[float]]], 
                 degree_u: int = 3, degree_v: int = 3,
                 weights: Optional[List[List[float]]] = None):
        """Initialize NURBS surface."""
        super().__init__()
        self.control_points = np.array(control_points)
        self.degree_u = degree_u
        self.degree_v = degree_v
        
        if weights is None:
            self.weights = np.ones((
                self.control_points.shape[0],
                self.control_points.shape[1]
            ))
        else:
            self.weights = np.array(weights)
        
        self._knot_vector_u = self._generate_knot_vector(
            self.control_points.shape[0] - 1,
            degree_u
        )
        self._knot_vector_v = self._generate_knot_vector(
            self.control_points.shape[1] - 1,
            degree_v
        )
    
    @classmethod
    def from_points(cls, points: List[List[List[float]]],
                   degree_u: int = 3, degree_v: int = 3) -> 'NURBSSurface':
        """Create surface from points."""
        return cls(points, degree_u, degree_v)
    
    def point_at(self, u: float, v: float) -> List[float]:
        """Evaluate surface at parameters (u,v)."""
        # Find spans
        span_u = self._find_span(u, self._knot_vector_u, self.control_points.shape[0] - 1, self.degree_u)
        span_v = self._find_span(v, self._knot_vector_v, self.control_points.shape[1] - 1, self.degree_v)
        
        # Calculate basis functions
        Nu = self._basis_functions(span_u, u, self._knot_vector_u, self.degree_u)
        Nv = self._basis_functions(span_v, v, self._knot_vector_v, self.degree_v)
        
        # Calculate point
        point = np.zeros(3)
        w_sum = 0.0
        
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                weight = self.weights[span_u - self.degree_u + i, span_v - self.degree_v + j]
                point += (Nu[i] * Nv[j] * weight * 
                         self.control_points[span_u - self.degree_u + i,
                                          span_v - self.degree_v + j])
                w_sum += Nu[i] * Nv[j] * weight
        
        return (point / w_sum).tolist()
    
    def _generate_knot_vector(self, n: int, p: int) -> np.ndarray:
        """Generate uniform knot vector."""
        m = n + p + 1
        knots = np.zeros(m + 1)
        
        for i in range(m + 1):
            if i <= p:
                knots[i] = 0
            elif i >= m - p:
                knots[i] = 1
            else:
                knots[i] = (i - p) / (m - 2 * p)
        
        return knots
    
    def _find_span(self, t: float, knot_vector: np.ndarray,
                  n: int, p: int) -> int:
        """Find knot span index."""
        if t >= 1.0:
            return n
        
        low = p
        high = n + 1
        mid = (low + high) // 2
        
        while t < knot_vector[mid] or t >= knot_vector[mid + 1]:
            if t < knot_vector[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
        
        return mid
    
    def _basis_functions(self, span: int, t: float,
                        knot_vector: np.ndarray, p: int) -> np.ndarray:
        """Calculate basis functions."""
        N = np.zeros(p + 1)
        left = np.zeros(p + 1)
        right = np.zeros(p + 1)
        
        N[0] = 1.0
        for j in range(1, p + 1):
            left[j] = t - knot_vector[span + 1 - j]
            right[j] = knot_vector[span + j] - t
            saved = 0.0
            
            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            
            N[j] = saved
        
        return N
    
    def analyze_thickness(self) -> float:
        """Analyze surface thickness."""
        # Simple thickness analysis based on bounding box
        bbox = self.bounding_box
        return min(bbox.dimensions)
    
    def analyze_overhangs(self) -> float:
        """Analyze surface overhangs."""
        # Simple overhang analysis
        max_angle = 0.0
        
        # Sample surface normals
        for u in np.linspace(0, 1, 10):
            for v in np.linspace(0, 1, 10):
                normal = self._compute_normal(u, v)
                angle = np.arccos(np.dot(normal, [0, 0, 1])) * 180 / np.pi
                max_angle = max(max_angle, angle)
        
        return max_angle
    
    def analyze_stress_points(self) -> List[Tuple[float, float, float]]:
        """Analyze surface stress points."""
        stress_points = []
        
        # Simple stress analysis based on curvature
        for u in np.linspace(0, 1, 10):
            for v in np.linspace(0, 1, 10):
                curvature = self._compute_curvature(u, v)
                if curvature > 1.0:  # Threshold for high curvature
                    point = self.point_at(u, v)
                    stress_points.append(tuple(point))  # type: ignore
        
        return stress_points
    
    def thicken_walls(self, min_thickness: float) -> 'BaseGeometry':
        """Thicken surface walls."""
        # Simple uniform thickening
        thickened_points = []
        
        for i in range(self.control_points.shape[0]):
            row = []
            for j in range(self.control_points.shape[1]):
                point = self.control_points[i, j]
                normal = self._compute_normal(i/(self.control_points.shape[0]-1),
                                           j/(self.control_points.shape[1]-1))
                offset = normal * min_thickness
                row.append((point + offset).tolist())
            thickened_points.append(row)
        
        return NURBSSurface(thickened_points, self.degree_u, self.degree_v)
    
    def reduce_overhangs(self, max_angle: float) -> 'BaseGeometry':
        """Reduce surface overhangs."""
        # Simple overhang reduction by rotating surface
        if self.analyze_overhangs() > max_angle:
            # Rotate surface to reduce overhangs
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(np.pi/6), -np.sin(np.pi/6)],
                [0, np.sin(np.pi/6), np.cos(np.pi/6)]
            ])
            
            rotated_points = []
            for i in range(self.control_points.shape[0]):
                row = []
                for j in range(self.control_points.shape[1]):
                    point = np.dot(rotation_matrix, self.control_points[i, j])
                    row.append(point.tolist())
                rotated_points.append(row)
            
            return NURBSSurface(rotated_points, self.degree_u, self.degree_v)
        
        return self
    
    def reinforce_weak_points(self) -> 'BaseGeometry':
        """Reinforce surface weak points."""
        # Simple reinforcement by thickening high-stress regions
        stress_points = self.analyze_stress_points()
        if not stress_points:
            return self
        
        # Add thickness near stress points
        reinforced_points = []
        for i in range(self.control_points.shape[0]):
            row = []
            for j in range(self.control_points.shape[1]):
                point = self.control_points[i, j]
                
                # Check if near stress point
                near_stress = False
                for stress_point in stress_points:
                    if np.linalg.norm(point - np.array(stress_point)) < 0.5:
                        near_stress = True
                        break
                
                if near_stress:
                    normal = self._compute_normal(i/(self.control_points.shape[0]-1),
                                               j/(self.control_points.shape[1]-1))
                    point = point + normal * 0.1  # Add thickness
                
                row.append(point.tolist())
            reinforced_points.append(row)
        
        return NURBSSurface(reinforced_points, self.degree_u, self.degree_v)
    
    def _compute_normal(self, u: float, v: float) -> np.ndarray:
        """Compute surface normal at point."""
        # Approximate normal using central differences
        delta = 0.01
        
        p1 = np.array(self.point_at(u + delta, v))
        p2 = np.array(self.point_at(u - delta, v))
        p3 = np.array(self.point_at(u, v + delta))
        p4 = np.array(self.point_at(u, v - delta))
        
        du = (p1 - p2) / (2 * delta)
        dv = (p3 - p4) / (2 * delta)
        
        normal = np.cross(du, dv)
        return normal / np.linalg.norm(normal)
    
    def _compute_curvature(self, u: float, v: float) -> float:
        """Compute surface curvature at point."""
        # Approximate curvature using second derivatives
        delta = 0.01
        
        p0 = np.array(self.point_at(u, v))
        p1 = np.array(self.point_at(u + delta, v))
        p2 = np.array(self.point_at(u - delta, v))
        p3 = np.array(self.point_at(u, v + delta))
        p4 = np.array(self.point_at(u, v - delta))
        
        # Second derivatives
        duu = (p1 - 2*p0 + p2) / (delta * delta)
        dvv = (p3 - 2*p0 + p4) / (delta * delta)
        
        # Mean curvature (simplified)
        return float(np.linalg.norm(duu + dvv) / 2) 

================================================
File: CadSeqProc/enhanced_geometry/organic.py
================================================
"""
Organic shape implementation with natural deformations.
"""

from typing import List, Dict, Any, Tuple, Optional, Union, cast
import numpy as np
from .base import BaseGeometry, Point, BoundingBox
from .nurbs import NURBSCurve, NURBSSurface

class OrganicSurface(BaseGeometry):
    """Organic surface with natural deformations."""
    
    def __init__(self, base_surface: NURBSSurface):
        """Initialize organic surface."""
        super().__init__()
        self.base_surface = base_surface
        self.deformations: List[Dict[str, Any]] = []
        self.features: List[Union[NURBSCurve, NURBSSurface]] = []
    
    @classmethod
    def from_nurbs(cls, surface: NURBSSurface) -> 'OrganicSurface':
        """Create organic surface from NURBS surface."""
        return cls(surface)
    
    @classmethod
    def combine(cls, surfaces: List['OrganicSurface']) -> 'OrganicSurface':
        """Combine multiple organic surfaces."""
        # Create base surface from first surface
        combined = cls(surfaces[0].base_surface)
        
        # Add deformations and features from all surfaces
        for surface in surfaces:
            combined.deformations.extend(surface.deformations)
            combined.features.extend(surface.features)
        
        return combined
    
    def add_random_deformation(self, intensity: float = 0.5) -> None:
        """Add random organic deformation."""
        self.deformations.append({
            'type': 'random',
            'intensity': intensity,
            'seed': np.random.randint(1000)
        })
    
    def add_bumps(self, count: int, height: float, radius: float) -> None:
        """Add organic bumps to surface."""
        self.deformations.append({
            'type': 'bumps',
            'count': count,
            'height': height,
            'radius': radius
        })
    
    def add_feature(self, feature: Union[NURBSCurve, NURBSSurface]) -> None:
        """Add geometric feature to surface."""
        self.features.append(feature)
    
    def rotate(self, angle: float) -> None:
        """Rotate surface around Z axis."""
        c = np.cos(angle)
        s = np.sin(angle)
        rotation = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to control points
        points = self.base_surface.control_points
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                points[i, j] = np.dot(rotation, points[i, j])
    
    def point_at(self, u: float, v: float) -> List[float]:
        """Evaluate surface at parameters (u,v) with deformations."""
        # Get base point
        point = np.array(self.base_surface.point_at(u, v))
        
        # Apply deformations
        for deform in self.deformations:
            if deform['type'] == 'random':
                point += self._apply_random_deform(u, v, deform)
            elif deform['type'] == 'bumps':
                point += self._apply_bumps(u, v, deform)
        
        return point.tolist()
    
    def _apply_random_deform(self, u: float, v: float,
                           deform: Dict[str, Any]) -> np.ndarray:
        """Apply random deformation."""
        np.random.seed(deform['seed'])
        
        # Generate Perlin-like noise
        freq = 5.0
        x = u * freq
        y = v * freq
        
        noise = 0.0
        amplitude = deform['intensity']
        for i in range(4):  # Octaves
            noise += amplitude * self._noise2d(x, y)
            x *= 2
            y *= 2
            amplitude *= 0.5
        
        # Get surface normal
        normal = self.base_surface._compute_normal(u, v)
        return normal * noise
    
    def _apply_bumps(self, u: float, v: float,
                    deform: Dict[str, Any]) -> np.ndarray:
        """Apply bump deformation."""
        # Generate random bump centers
        np.random.seed(0)  # For reproducibility
        centers = []
        for _ in range(deform['count']):
            centers.append((
                np.random.random(),  # u coordinate
                np.random.random()   # v coordinate
            ))
        
        # Calculate bump influence
        total = np.zeros(3)
        for cu, cv in centers:
            dist = np.sqrt((u - cu)**2 + (v - cv)**2)
            if dist < deform['radius']:
                factor = (1 - dist/deform['radius'])**2
                normal = self.base_surface._compute_normal(u, v)
                total += normal * factor * deform['height']
        
        return total
    
    def _noise2d(self, x: float, y: float) -> float:
        """Simple 2D noise function."""
        # Hash coordinates to pseudo-random gradient
        n = int(x + y * 57)
        n = (n << 13) ^ n
        rand = (n * (n * n * 15731 + 789221) + 1376312589)
        return 1.0 - float(rand & 0x7fffffff) / 1073741824.0
    
    def analyze_thickness(self) -> float:
        """Analyze surface thickness."""
        # Use base surface thickness as starting point
        thickness = self.base_surface.analyze_thickness()
        
        # Adjust for deformations
        for deform in self.deformations:
            if deform['type'] == 'random':
                thickness *= (1.0 - 0.2 * deform['intensity'])
            elif deform['type'] == 'bumps':
                thickness = min(thickness, deform['height'])
        
        return thickness
    
    def analyze_overhangs(self) -> float:
        """Analyze surface overhangs."""
        max_angle = self.base_surface.analyze_overhangs()
        
        # Sample deformed surface
        for u in np.linspace(0, 1, 10):
            for v in np.linspace(0, 1, 10):
                point = np.array(self.point_at(u, v))
                normal = self._compute_normal(u, v)
                angle = np.arccos(np.dot(normal, [0, 0, 1])) * 180 / np.pi
                max_angle = max(max_angle, angle)
        
        return max_angle
    
    def analyze_stress_points(self) -> List[Tuple[float, float, float]]:
        """Analyze surface stress points."""
        stress_points = []
        
        # Check base surface stress points
        base_points = self.base_surface.analyze_stress_points()
        stress_points.extend(base_points)
        
        # Check deformation stress points
        for u in np.linspace(0, 1, 10):
            for v in np.linspace(0, 1, 10):
                # Calculate curvature of deformed surface
                curvature = self._compute_curvature(u, v)
                if curvature > 1.0:  # High curvature threshold
                    point = self.point_at(u, v)
                    stress_points.append(tuple(point))  # type: ignore
        
        return stress_points
    
    def thicken_walls(self, min_thickness: float) -> 'BaseGeometry':
        """Thicken surface walls."""
        # Thicken base surface
        thickened_base = self.base_surface.thicken_walls(min_thickness)
        if not isinstance(thickened_base, NURBSSurface):
            return self
        
        # Create new organic surface with thickened base
        thickened = OrganicSurface(thickened_base)
        thickened.deformations = self.deformations.copy()
        thickened.features = self.features.copy()
        
        return thickened
    
    def reduce_overhangs(self, max_angle: float) -> 'BaseGeometry':
        """Reduce surface overhangs."""
        if self.analyze_overhangs() <= max_angle:
            return self
        
        # Reduce base surface overhangs
        reduced_base = self.base_surface.reduce_overhangs(max_angle)
        if not isinstance(reduced_base, NURBSSurface):
            return self
        
        # Create new organic surface with reduced overhangs
        reduced = OrganicSurface(reduced_base)
        
        # Reduce deformation intensities
        for deform in self.deformations:
            if deform['type'] == 'random':
                deform['intensity'] *= 0.7
            elif deform['type'] == 'bumps':
                deform['height'] *= 0.7
        reduced.deformations = self.deformations
        reduced.features = self.features
        
        return reduced
    
    def reinforce_weak_points(self) -> 'BaseGeometry':
        """Reinforce surface weak points."""
        # Reinforce base surface
        reinforced_base = self.base_surface.reinforce_weak_points()
        if not isinstance(reinforced_base, NURBSSurface):
            return self
        
        # Create new organic surface with reinforced base
        reinforced = OrganicSurface(reinforced_base)
        
        # Reduce deformation intensities near stress points
        stress_points = self.analyze_stress_points()
        if stress_points:
            for deform in self.deformations:
                if deform['type'] == 'random':
                    deform['intensity'] *= 0.8
                elif deform['type'] == 'bumps':
                    deform['height'] *= 0.8
        
        reinforced.deformations = self.deformations
        reinforced.features = self.features
        
        return reinforced
    
    def _compute_normal(self, u: float, v: float) -> np.ndarray:
        """Compute surface normal at point."""
        # Approximate normal using central differences
        delta = 0.01
        
        p1 = np.array(self.point_at(u + delta, v))
        p2 = np.array(self.point_at(u - delta, v))
        p3 = np.array(self.point_at(u, v + delta))
        p4 = np.array(self.point_at(u, v - delta))
        
        du = (p1 - p2) / (2 * delta)
        dv = (p3 - p4) / (2 * delta)
        
        normal = np.cross(du, dv)
        return normal / np.linalg.norm(normal)
    
    def _compute_curvature(self, u: float, v: float) -> float:
        """Compute surface curvature at point."""
        # Approximate curvature using second derivatives
        delta = 0.01
        
        p0 = np.array(self.point_at(u, v))
        p1 = np.array(self.point_at(u + delta, v))
        p2 = np.array(self.point_at(u - delta, v))
        p3 = np.array(self.point_at(u, v + delta))
        p4 = np.array(self.point_at(u, v - delta))
        
        # Second derivatives
        duu = (p1 - 2*p0 + p2) / (delta * delta)
        dvv = (p3 - 2*p0 + p4) / (delta * delta)
        
        # Mean curvature (simplified)
        return float(np.linalg.norm(duu + dvv) / 2) 

================================================
File: CadSeqProc/enhanced_geometry/parametric.py
================================================
"""
Parametric curve support for organic shape generation.
Provides mathematical functions for generating complex curves and patterns.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Union, Dict, Any
from .base import Point, GeometricEntity
from .nurbs import NURBSCurve

class ParametricCurve(GeometricEntity):
    """Base class for parametric curves."""
    
    def __init__(self, 
                 curve_func: Callable[[float], Tuple[float, float, float]],
                 t_range: Tuple[float, float] = (0, 2*np.pi),
                 num_samples: int = 100):
        self.curve_func = curve_func
        self.t_range = t_range
        self.num_samples = num_samples
        
    def sample_points(self) -> List[Point]:
        """Sample points along the curve."""
        t_vals = np.linspace(self.t_range[0], self.t_range[1], self.num_samples)
        return [Point(*self.curve_func(t)) for t in t_vals]
    
    def to_nurbs(self) -> NURBSCurve:
        """Convert to NURBS representation."""
        points = self.sample_points()
        return NURBSCurve.from_points(points)

class Spiral(ParametricCurve):
    """Logarithmic or Archimedean spiral curve."""
    
    def __init__(self,
                 a: float = 1.0,
                 b: float = 0.2,
                 height: float = 0.0,
                 num_turns: float = 2.0,
                 spiral_type: str = 'logarithmic'):
        """
        Args:
            a: Base radius
            b: Growth rate
            height: Total height change
            num_turns: Number of complete turns
            spiral_type: 'logarithmic' or 'archimedean'
        """
        t_range = (0, 2*np.pi * num_turns)
        
        def spiral_curve(t: float) -> Tuple[float, float, float]:
            if spiral_type == 'logarithmic':
                r = a * np.exp(b * t)
            else:  # archimedean
                r = a + b * t
            
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = height * t / (2*np.pi * num_turns)
            return (x, y, z)
        
        super().__init__(spiral_curve, t_range)

class FlowerPetal(ParametricCurve):
    """Parametric curve for flower petal shapes."""
    
    def __init__(self,
                 length: float = 1.0,
                 width: float = 0.3,
                 curve_factor: float = 0.3,
                 harmonics: int = 3,
                 asymmetry: float = 0.0):
        def petal_curve(t: float) -> Tuple[float, float, float]:
            # Add asymmetry to create more natural looking petals
            asym = 1.0 + asymmetry * np.sin(2*t)
            x = length * np.cos(t) * (1 + curve_factor * np.sin(harmonics*t)) * asym
            y = width * np.sin(t) * (1 + curve_factor * np.sin(harmonics*t))
            z = 0.1 * np.sin(t) * (1 + 0.5 * np.sin(2*t))  # More complex 3D curvature
            return (x, y, z)
            
        super().__init__(petal_curve)

class Helix(ParametricCurve):
    """Helical curve with variable radius."""
    
    def __init__(self,
                 radius: float = 1.0,
                 pitch: float = 1.0,
                 num_turns: float = 3.0,
                 taper: float = 0.0):
        t_range = (0, 2*np.pi * num_turns)
        
        def helix_curve(t: float) -> Tuple[float, float, float]:
            # Apply taper to radius
            r = radius * (1.0 - taper * t/(2*np.pi * num_turns))
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = pitch * t/(2*np.pi)
            return (x, y, z)
        
        super().__init__(helix_curve, t_range)

class Lissajous(ParametricCurve):
    """Lissajous curve for complex symmetric patterns."""
    
    def __init__(self,
                 a: float = 1.0,
                 b: float = 1.0,
                 freq_x: int = 3,
                 freq_y: int = 2,
                 phase: float = np.pi/2):
        def lissajous_curve(t: float) -> Tuple[float, float, float]:
            x = a * np.sin(freq_x * t)
            y = b * np.sin(freq_y * t + phase)
            z = 0.0
            return (x, y, z)
        
        super().__init__(lissajous_curve)

class SuperShape(ParametricCurve):
    """Superformula-based curve for complex organic shapes."""
    
    def __init__(self,
                 a: float = 1.0,
                 b: float = 1.0,
                 m1: float = 7.0,
                 m2: float = 3.0,
                 n1: float = 0.2,
                 n2: float = 1.7,
                 n3: float = 1.7):
        def supershape_curve(t: float) -> Tuple[float, float, float]:
            phi = t
            
            # Superformula
            part1 = (1/a) * np.abs(np.cos(m1*phi/4))**n2
            part2 = (1/b) * np.abs(np.sin(m2*phi/4))**n3
            r = (part1 + part2)**(-1/n1)
            
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = 0.0
            return (x, y, z)
        
        super().__init__(supershape_curve)

class BezierCurve(ParametricCurve):
    """Bézier curve with variable control points."""
    
    def __init__(self, control_points: List[Point]):
        def bezier_curve(t: float) -> Tuple[float, float, float]:
            n = len(control_points) - 1
            point = np.zeros(3)
            for i, p in enumerate(control_points):
                # Bernstein polynomial
                coeff = np.math.comb(n, i) * (1-t)**(n-i) * t**i
                point += coeff * np.array([p.x, p.y, p.z])
            return tuple(point)
        
        super().__init__(bezier_curve, t_range=(0, 1))

class RoseCurve(ParametricCurve):
    """Rose curve (rhodonea) for petal-like patterns."""
    
    def __init__(self,
                 radius: float = 1.0,
                 n: int = 3,
                 d: int = 1,
                 height_factor: float = 0.2):
        """
        Args:
            radius: Base radius
            n, d: Determines number of petals (n/d petals if n odd, 2n/d if n even)
            height_factor: Controls 3D variation
        """
        def rose_curve(t: float) -> Tuple[float, float, float]:
            k = n / d
            r = radius * np.cos(k * t)
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = height_factor * np.sin(k * t)
            return (x, y, z)
        
        super().__init__(rose_curve)

class EpicycloidCurve(ParametricCurve):
    """Epicycloid curve for complex geometric patterns."""
    
    def __init__(self,
                 R: float = 1.0,  # Fixed circle radius
                 r: float = 0.3,  # Moving circle radius
                 d: float = 0.5): # Distance from center
        def epicycloid_curve(t: float) -> Tuple[float, float, float]:
            x = (R+r) * np.cos(t) - d * np.cos((R+r)*t/r)
            y = (R+r) * np.sin(t) - d * np.sin((R+r)*t/r)
            z = 0.1 * np.sin(5*t)  # Add slight 3D variation
            return (x, y, z)
        
        super().__init__(epicycloid_curve)

class FractalCurve(ParametricCurve):
    """Base class for fractal-based curves."""
    
    def __init__(self,
                 iterations: int = 3,
                 scale: float = 1.0):
        self.iterations = iterations
        self.scale = scale
        super().__init__(self._fractal_curve)
    
    def _fractal_curve(self, t: float) -> Tuple[float, float, float]:
        raise NotImplementedError

class DragonCurve(FractalCurve):
    """Dragon curve implementation."""
    
    def _fractal_curve(self, t: float) -> Tuple[float, float, float]:
        angle = 0.0
        x, y = 0.0, 0.0
        for i in range(self.iterations):
            angle += t * np.pi/4
            factor = 0.7**i
            x += factor * np.cos(angle)
            y += factor * np.sin(angle)
        return (self.scale * x, self.scale * y, 0.0)

class HypocycloidCurve(ParametricCurve):
    """Hypocycloid curve for star-like patterns."""
    
    def __init__(self,
                 R: float = 1.0,    # Fixed circle radius
                 r: float = 0.3,    # Moving circle radius
                 d: float = 0.5,    # Distance from center
                 height_var: float = 0.1):
        def hypocycloid_curve(t: float) -> Tuple[float, float, float]:
            x = (R-r) * np.cos(t) + d * np.cos((R-r)*t/r)
            y = (R-r) * np.sin(t) - d * np.sin((R-r)*t/r)
            z = height_var * np.sin((R/r)*t)  # Add 3D variation
            return (x, y, z)
        
        super().__init__(hypocycloid_curve)

class TorusKnotCurve(ParametricCurve):
    """Torus knot curve for complex 3D patterns."""
    
    def __init__(self,
                 p: int = 2,        # Number of winds around torus
                 q: int = 3,        # Number of winds through torus
                 R: float = 1.0,    # Major radius
                 r: float = 0.3):   # Minor radius
        def torus_knot_curve(t: float) -> Tuple[float, float, float]:
            # Parametric equations for torus knot
            pt = p * t
            qt = q * t
            x = R * (2 + np.cos(qt)) * np.cos(pt) / 3
            y = R * (2 + np.cos(qt)) * np.sin(pt) / 3
            z = R * np.sin(qt) / 3
            return (x, y, z)
        
        super().__init__(torus_knot_curve)

class HermiteSpline(ParametricCurve):
    """Hermite spline for smooth interpolation."""
    
    def __init__(self,
                 p0: Point,         # Start point
                 p1: Point,         # End point
                 t0: Point,         # Start tangent
                 t1: Point):        # End tangent
        def hermite_curve(t: float) -> Tuple[float, float, float]:
            # Hermite basis functions
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2
            
            # Interpolate each component
            x = h00*p0.x + h10*t0.x + h01*p1.x + h11*t1.x
            y = h00*p0.y + h10*t0.y + h01*p1.y + h11*t1.y
            z = h00*p0.z + h10*t0.z + h01*p1.z + h11*t1.z
            return (x, y, z)
        
        super().__init__(hermite_curve, t_range=(0, 1))

class CombinedPattern:
    """Support for combining multiple patterns."""
    
    def __init__(self,
                 base_patterns: List[List[ParametricCurve]],
                 weights: Optional[List[float]] = None):
        self.patterns = base_patterns
        self.weights = weights or [1.0] * len(base_patterns)
        
    def blend(self, t: float = 0.5) -> List[ParametricCurve]:
        """Blend between patterns based on weights and parameter t."""
        result = []
        total_weight = sum(self.weights)
        normalized_weights = [w/total_weight for w in self.weights]
        
        # Find maximum number of curves in any pattern
        max_curves = max(len(pattern) for pattern in self.patterns)
        
        for i in range(max_curves):
            points = []
            total_points = 0
            
            # Collect points from each pattern
            for pattern, weight in zip(self.patterns, normalized_weights):
                if i < len(pattern):
                    curve_points = pattern[i].sample_points()
                    points.append((curve_points, weight))
                    total_points = max(total_points, len(curve_points))
            
            # Blend points
            blended_points = []
            for j in range(total_points):
                x, y, z = 0, 0, 0
                total_w = 0
                
                for curve_points, weight in points:
                    if j < len(curve_points):
                        p = curve_points[j]
                        w = weight * (1 - abs(2*t - 1))  # Smooth transition
                        x += p.x * w
                        y += p.y * w
                        z += p.z * w
                        total_w += w
                
                if total_w > 0:
                    blended_points.append(Point(x/total_w, y/total_w, z/total_w))
            
            if blended_points:
                result.append(NURBSCurve.from_points(blended_points))
        
        return result

class PatternGenerator:
    """Generate patterns of curves with transformations."""
    
    @staticmethod
    def circular_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        radius: float,
        z_offset: float = 0.0,
        scale_factor: Optional[float] = None,
        rotation_offset: float = 0.0
    ) -> List[ParametricCurve]:
        """Create a circular pattern of curves."""
        patterns = []
        for i in range(num_copies):
            angle = 2 * np.pi * i / num_copies + rotation_offset
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, radius * c],
                [s, c, 0, radius * s],
                [0, 0, 1, z_offset],
                [0, 0, 0, 1]
            ])
            
            # Apply scale if specified
            if scale_factor is not None:
                scale = 1.0 + (scale_factor - 1.0) * i / num_copies
                transform[:3, :3] *= scale
            
            # Create new curve with transformed points
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_points.append(Point(
                    transformed[0] / transformed[3],
                    transformed[1] / transformed[3],
                    transformed[2] / transformed[3]
                ))
            
            patterns.append(NURBSCurve.from_points(new_points))
            
        return patterns
    
    @staticmethod
    def spiral_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        start_radius: float,
        end_radius: float,
        height: float = 0.0,
        rotation_offset: float = 0.0
    ) -> List[ParametricCurve]:
        """Create a spiral pattern of curves."""
        patterns = []
        for i in range(num_copies):
            t = i / (num_copies - 1)
            angle = 2 * np.pi * t * 3 + rotation_offset  # 3 turns
            radius = start_radius + (end_radius - start_radius) * t
            
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, radius * c],
                [s, c, 0, radius * s],
                [0, 0, 1, height * t],
                [0, 0, 0, 1]
            ])
            
            # Create new curve
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_points.append(Point(
                    transformed[0] / transformed[3],
                    transformed[1] / transformed[3],
                    transformed[2] / transformed[3]
                ))
            
            patterns.append(NURBSCurve.from_points(new_points))
        
        return patterns
    
    @staticmethod
    def fibonacci_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        max_radius: float,
        scale_factor: float = 0.95
    ) -> List[ParametricCurve]:
        """Create a Fibonacci spiral pattern of curves."""
        patterns = []
        golden_angle = np.pi * (3 - np.sqrt(5))  # ≈ 137.5 degrees
        
        for i in range(num_copies):
            angle = i * golden_angle
            # Radius grows as square root of i
            radius = max_radius * np.sqrt(i / num_copies)
            scale = scale_factor ** i
            
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c*scale, -s*scale, 0, radius * c],
                [s*scale, c*scale, 0, radius * s],
                [0, 0, scale, 0],
                [0, 0, 0, 1]
            ])
            
            # Create new curve
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_points.append(Point(
                    transformed[0] / transformed[3],
                    transformed[1] / transformed[3],
                    transformed[2] / transformed[3]
                ))
            
            patterns.append(NURBSCurve.from_points(new_points))
        
        return patterns

    @staticmethod
    def fractal_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        scale_range: Tuple[float, float] = (0.2, 1.0),
        rotation_base: float = np.pi/3
    ) -> List[ParametricCurve]:
        """Create a fractal-like pattern of curves."""
        patterns = []
        for i in range(num_copies):
            scale = scale_range[0] + (scale_range[1] - scale_range[0]) * (i/num_copies)
            angle = rotation_base * i
            
            # Create transformation matrix with fractal properties
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c*scale, -s*scale, 0, scale * np.cos(i*np.pi/4)],
                [s*scale, c*scale, 0, scale * np.sin(i*np.pi/4)],
                [0, 0, scale, 0.1 * scale * np.sin(i*np.pi/3)],
                [0, 0, 0, 1]
            ])
            
            # Create new curve
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_points.append(Point(
                    transformed[0] / transformed[3],
                    transformed[1] / transformed[3],
                    transformed[2] / transformed[3]
                ))
            
            patterns.append(NURBSCurve.from_points(new_points))
        
        return patterns
    
    @staticmethod
    def radial_wave_pattern(
        base_curve: ParametricCurve,
        num_copies: int,
        radius: float,
        wave_freq: float = 3.0,
        wave_amp: float = 0.2
    ) -> List[ParametricCurve]:
        """Create a pattern with wave-like radial variation."""
        patterns = []
        for i in range(num_copies):
            angle = 2 * np.pi * i / num_copies
            # Add wave variation to radius
            r = radius * (1 + wave_amp * np.sin(wave_freq * angle))
            
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, r * c],
                [s, c, 0, r * s],
                [0, 0, 1, 0.1 * np.sin(wave_freq * angle)],
                [0, 0, 0, 1]
            ])
            
            # Create new curve
            base_points = base_curve.sample_points()
            new_points = []
            for p in base_points:
                p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
                transformed = transform @ p_homogeneous
                new_points.append(Point(
                    transformed[0] / transformed[3],
                    transformed[1] / transformed[3],
                    transformed[2] / transformed[3]
                ))
            
            patterns.append(NURBSCurve.from_points(new_points))
        
        return patterns

class OrganicPatternFactory:
    """Factory for creating organic patterns."""
    
    @staticmethod
    def create_flower(
        num_petals: int,
        petal_length: float,
        petal_width: float,
        curve_factor: float = 0.3,
        center_radius: float = 0.2,
        pattern_type: str = 'circular',
        center_type: str = 'spiral'
    ) -> List[GeometricEntity]:
        """Create a flower pattern with petals."""
        # Create base petal with random asymmetry
        base_petal = FlowerPetal(
            length=petal_length,
            width=petal_width,
            curve_factor=curve_factor,
            asymmetry=0.1 * np.random.random()
        )
        
        # Generate petal pattern based on type
        if pattern_type == 'fibonacci':
            petals = PatternGenerator.fibonacci_pattern(
                base_petal,
                num_petals,
                max_radius=petal_length * 0.2,
                scale_factor=0.97
            )
        elif pattern_type == 'spiral':
            petals = PatternGenerator.spiral_pattern(
                base_petal,
                num_petals,
                start_radius=center_radius,
                end_radius=center_radius * 2,
                height=petal_length * 0.1
            )
        else:  # circular
            petals = PatternGenerator.circular_pattern(
                base_petal,
                num_petals,
                radius=center_radius,
                rotation_offset=np.random.random() * np.pi/6  # Slight random rotation
            )
        
        # Create center based on type
        if center_type == 'spiral':
            center = Spiral(
                a=center_radius * 0.2,
                b=0.1,
                height=center_radius * 0.2,
                num_turns=3
            ).to_nurbs()
        elif center_type == 'lissajous':
            center = Lissajous(
                a=center_radius,
                b=center_radius,
                freq_x=3,
                freq_y=4
            ).to_nurbs()
        else:  # simple circle
            center = NURBSCurve.create_circle(center_radius)
        
        return [center] + petals 

class PatternAnalyzer:
    """AI-driven pattern analysis and selection."""
    
    @staticmethod
    def analyze_shape_requirements(description: str) -> Dict[str, Any]:
        """Analyze text description to determine optimal pattern parameters."""
        requirements = {
            'pattern_type': 'circular',  # default
            'complexity': 1.0,
            'regularity': 1.0,
            'dimensionality': '2D',
            'symmetry': True,
            'curve_types': [],
            'combination_mode': None
        }
        
        # Natural/organic patterns
        if any(word in description.lower() for word in 
               ['natural', 'organic', 'flowing', 'random', 'irregular']):
            requirements.update({
                'pattern_type': 'fibonacci',
                'regularity': 0.7,
                'complexity': 1.2,
                'curve_types': ['rose', 'bezier']
            })
        
        # Geometric/regular patterns
        if any(word in description.lower() for word in 
               ['geometric', 'regular', 'symmetric', 'even']):
            requirements.update({
                'pattern_type': 'circular',
                'regularity': 1.0,
                'complexity': 0.8,
                'curve_types': ['epicycloid', 'hypocycloid']
            })
        
        # Complex/intricate patterns
        if any(word in description.lower() for word in 
               ['complex', 'intricate', 'detailed', 'ornate']):
            requirements.update({
                'pattern_type': 'fractal',
                'complexity': 1.5,
                'regularity': 0.8,
                'curve_types': ['torus_knot', 'supershape']
            })
        
        # Spiral patterns
        if any(word in description.lower() for word in 
               ['spiral', 'swirl', 'twist', 'coil']):
            requirements.update({
                'pattern_type': 'spiral',
                'complexity': 1.2,
                'regularity': 0.9,
                'curve_types': ['spiral', 'helix']
            })
        
        # Wave patterns
        if any(word in description.lower() for word in 
               ['wave', 'ripple', 'undulating']):
            requirements.update({
                'pattern_type': 'radial_wave',
                'complexity': 1.1,
                'regularity': 0.85,
                'curve_types': ['lissajous', 'hermite']
            })
        
        # 3D variations
        if any(word in description.lower() for word in 
               ['3d', 'dimensional', 'depth', 'layered']):
            requirements['dimensionality'] = '3D'
            requirements['curve_types'].append('torus_knot')
        
        # Pattern combinations
        if any(word in description.lower() for word in 
               ['mixed', 'combined', 'blend', 'hybrid']):
            requirements['combination_mode'] = 'blend'
        elif any(word in description.lower() for word in 
                ['layered', 'stacked', 'overlaid']):
            requirements['combination_mode'] = 'layer'
        
        return requirements
    
    @staticmethod
    def get_optimal_pattern(requirements: Dict[str, Any]) -> str:
        """Determine the optimal pattern type based on requirements."""
        pattern_scores = {
            'circular': 0,
            'spiral': 0,
            'fibonacci': 0,
            'fractal': 0,
            'radial_wave': 0
        }
        
        # Score patterns based on requirements
        if requirements['regularity'] > 0.9:
            pattern_scores['circular'] += 2
        if requirements['complexity'] > 1.2:
            pattern_scores['fractal'] += 2
            pattern_scores['fibonacci'] += 1
        if requirements['dimensionality'] == '3D':
            pattern_scores['spiral'] += 1
            pattern_scores['radial_wave'] += 1
        
        # Consider curve types in scoring
        if 'torus_knot' in requirements['curve_types']:
            pattern_scores['spiral'] += 1
        if 'hypocycloid' in requirements['curve_types']:
            pattern_scores['circular'] += 1
        
        # Return pattern with highest score
        return max(pattern_scores.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def create_curve_from_type(
        curve_type: str,
        size: float = 1.0,
        complexity: float = 1.0
    ) -> ParametricCurve:
        """Create a curve instance based on type and parameters."""
        if curve_type == 'torus_knot':
            p = int(2 + complexity * 2)
            q = int(3 + complexity * 2)
            return TorusKnotCurve(p=p, q=q, R=size, r=size*0.3)
        elif curve_type == 'hypocycloid':
            return HypocycloidCurve(R=size, r=size*0.3, height_var=0.1*complexity)
        elif curve_type == 'hermite':
            # Create a smooth curve with controlled complexity
            p0 = Point(0, 0, 0)
            p1 = Point(size, 0, 0)
            t0 = Point(0, size*complexity, 0)
            t1 = Point(0, -size*complexity, 0)
            return HermiteSpline(p0, p1, t0, t1)
        # ... handle other curve types ...
        else:
            return FlowerPetal(length=size, curve_factor=0.3*complexity) 

================================================
File: CadSeqProc/enhanced_geometry/pattern_recognition.py
================================================
"""Design pattern recognition and analysis module for CAD models."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np  # type: ignore
from .base import GeometricEntity, Point
from .nurbs import NURBSCurve, NURBSSurface

@dataclass
class PatternFeature:
    """Represents a recognized pattern feature in the design."""
    pattern_type: str  # e.g., "linear_array", "circular_array", "symmetry"
    base_feature: GeometricEntity
    instances: List[GeometricEntity]
    parameters: Dict[str, Any]
    confidence: float  # Recognition confidence score

@dataclass
class DesignPattern:
    """Represents a higher-level design pattern."""
    name: str
    description: str
    features: List[PatternFeature]
    relationships: List[Dict[str, Any]]
    manufacturing_notes: Optional[Dict[str, Any]] = None
    reuse_suggestions: Optional[List[Dict[str, Any]]] = None

class PatternRecognizer:
    """Core class for recognizing and analyzing design patterns."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.pattern_library = self._initialize_pattern_library()
        
    def _initialize_pattern_library(self) -> Dict[str, Any]:
        """Initialize the library of known patterns and their characteristics."""
        return {
            "linear_array": {
                "detector": self._detect_linear_array,
                "validator": self._validate_linear_array,
                "parameters": ["spacing", "count", "direction"]
            },
            "circular_array": {
                "detector": self._detect_circular_array,
                "validator": self._validate_circular_array,
                "parameters": ["radius", "count", "angle"]
            },
            "symmetry": {
                "detector": self._detect_symmetry,
                "validator": self._validate_symmetry,
                "parameters": ["plane", "elements"]
            },
            "repetitive_feature": {
                "detector": self._detect_repetitive_feature,
                "validator": self._validate_repetitive_feature,
                "parameters": ["feature_type", "instances", "variations"]
            }
        }

    def analyze_geometry(self, geometry: GeometricEntity) -> List[DesignPattern]:
        """Analyze geometry to identify design patterns."""
        patterns = []
        
        # Decompose geometry into basic features
        features = self._decompose_geometry(geometry)
        
        # Analyze each pattern type
        for pattern_type, pattern_info in self.pattern_library.items():
            detector = pattern_info["detector"]
            validator = pattern_info["validator"]
            
            # Detect potential patterns
            potential_patterns = detector(features)
            
            # Validate and refine patterns
            for pattern in potential_patterns:
                if validator(pattern):
                    patterns.append(self._create_design_pattern(pattern))
        
        return patterns

    def _decompose_geometry(self, geometry: GeometricEntity) -> List[GeometricEntity]:
        """Decompose complex geometry into basic features."""
        features = []
        
        if hasattr(geometry, "sub_entities"):
            features.extend(geometry.sub_entities)
        else:
            features.append(geometry)
            
        return features

    def _detect_linear_array(self, features: List[GeometricEntity]) -> List[PatternFeature]:
        """Detect linear array patterns in features."""
        patterns: List[PatternFeature] = []
        for i, base_feature in enumerate(features):
            similar_features = self._find_similar_features(base_feature, features[i+1:])
            if len(similar_features) >= 2:  # Minimum 3 instances for a pattern
                spacing = self._calculate_linear_spacing(base_feature, similar_features)
                if spacing is not None:
                    patterns.append(PatternFeature(
                        pattern_type="linear_array",
                        base_feature=base_feature,
                        instances=similar_features,
                        parameters={"spacing": spacing},
                        confidence=0.9
                    ))
        return patterns

    def _detect_circular_array(self, features: List[GeometricEntity]) -> List[PatternFeature]:
        """Detect circular array patterns in features."""
        patterns: List[PatternFeature] = []
        for i, base_feature in enumerate(features):
            similar_features = self._find_similar_features(base_feature, features[i+1:])
            if len(similar_features) >= 2:
                center, radius = self._calculate_circular_parameters(base_feature, similar_features)
                if center is not None and radius is not None:
                    patterns.append(PatternFeature(
                        pattern_type="circular_array",
                        base_feature=base_feature,
                        instances=similar_features,
                        parameters={"center": center, "radius": radius},
                        confidence=0.85
                    ))
        return patterns

    def _detect_symmetry(self, features: List[GeometricEntity]) -> List[PatternFeature]:
        """Detect symmetry patterns in features."""
        patterns: List[PatternFeature] = []
        # Implementation for symmetry detection
        return patterns

    def _detect_repetitive_feature(self, features: List[GeometricEntity]) -> List[PatternFeature]:
        """Detect repetitive features that may not follow a strict pattern."""
        patterns: List[PatternFeature] = []
        # Implementation for repetitive feature detection
        return patterns

    def _find_similar_features(self, base: GeometricEntity, 
                             candidates: List[GeometricEntity]) -> List[GeometricEntity]:
        """Find features similar to the base feature."""
        similar = []
        for candidate in candidates:
            if self._compare_features(base, candidate) > 0.9:  # Similarity threshold
                similar.append(candidate)
        return similar

    def _compare_features(self, feature1: GeometricEntity, 
                         feature2: GeometricEntity) -> float:
        """Compare two features and return similarity score."""
        # Basic implementation - should be enhanced based on specific requirements
        if type(feature1) != type(feature2):
            return 0.0
            
        # Compare basic properties
        similarity = 1.0
        
        # Compare dimensions if available
        if hasattr(feature1, "dimensions") and hasattr(feature2, "dimensions"):
            dim_similarity = self._compare_dimensions(
                feature1.dimensions, feature2.dimensions)
            similarity *= dim_similarity
            
        return similarity

    def _compare_dimensions(self, dim1: Dict[str, float], dim2: Dict[str, float]) -> float:
        """Compare dimensions of two features and return similarity score."""
        if not dim1 or not dim2:
            return 0.0
            
        # Get common dimension keys
        common_dims = set(dim1.keys()) & set(dim2.keys())
        if not common_dims:
            return 0.0
            
        # Compare each dimension
        similarities = []
        for dim in common_dims:
            val1, val2 = dim1[dim], dim2[dim]
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            elif val1 == 0 or val2 == 0:
                similarities.append(0.0)
            else:
                ratio = min(val1, val2) / max(val1, val2)
                similarities.append(ratio)
                
        return sum(similarities) / len(similarities)

    def _calculate_linear_spacing(self, base: GeometricEntity, 
                                instances: List[GeometricEntity]) -> Optional[float]:
        """Calculate spacing for linear array pattern."""
        if not instances:
            return None
            
        spacings = []
        base_center = self._get_center(base)
        
        for instance in instances:
            instance_center = self._get_center(instance)
            spacing = np.linalg.norm(instance_center - base_center)
            spacings.append(spacing)
            
        # Check if spacings are consistent
        if self._are_spacings_consistent(spacings):
            return np.mean(spacings)
        return None

    def _calculate_circular_parameters(self, base: GeometricEntity,
                                    instances: List[GeometricEntity]) -> Tuple[Optional[Point], Optional[float]]:
        """Calculate center and radius for circular array pattern."""
        # Implementation for circular parameter calculation
        return None, None

    def _get_center(self, entity: GeometricEntity) -> np.ndarray:
        """Get the center point of a geometric entity."""
        if hasattr(entity, "center"):
            return np.array(entity.center)
        elif hasattr(entity, "bounds"):
            min_bound, max_bound = entity.bounds
            return (np.array(min_bound) + np.array(max_bound)) / 2
        return np.zeros(3)

    def _are_spacings_consistent(self, spacings: List[float], tolerance: float = 0.01) -> bool:
        """Check if spacings are consistent within tolerance."""
        if not spacings:
            return False
        mean_spacing = np.mean(spacings)
        return all(abs(s - mean_spacing) <= tolerance * mean_spacing for s in spacings)

    def _calculate_distance(self, entity1: GeometricEntity, entity2: GeometricEntity) -> float:
        """Calculate the distance between two geometric entities."""
        center1 = self._get_center(entity1)
        center2 = self._get_center(entity2)
        return float(np.linalg.norm(center2 - center1))

    def _create_design_pattern(self, pattern_feature: PatternFeature) -> DesignPattern:
        """Create a DesignPattern from a PatternFeature with additional analysis."""
        return DesignPattern(
            name=f"{pattern_feature.pattern_type}_pattern",
            description=self._generate_pattern_description(pattern_feature),
            features=[pattern_feature],
            relationships=self._analyze_pattern_relationships(pattern_feature),
            manufacturing_notes=self._generate_manufacturing_notes(pattern_feature),
            reuse_suggestions=self._generate_reuse_suggestions(pattern_feature)
        )

    def _generate_pattern_description(self, pattern: PatternFeature) -> str:
        """Generate human-readable description of the pattern."""
        if self.llm_client:
            # Use LLM for rich description
            return self.llm_client.generate_pattern_description(pattern)
        
        # Fallback basic description
        return f"{pattern.pattern_type} with {len(pattern.instances)} instances"

    def _analyze_pattern_relationships(self, pattern: PatternFeature) -> List[Dict[str, Any]]:
        """Analyze relationships between pattern elements."""
        relationships: List[Dict[str, Any]] = []
        
        # Analyze spatial relationships
        if pattern.pattern_type == "linear_array":
            relationships.append({
                "type": "spacing",
                "value": pattern.parameters.get("spacing"),
                "unit": "mm"
            })
        elif pattern.pattern_type == "circular_array":
            relationships.append({
                "type": "radius",
                "value": pattern.parameters.get("radius"),
                "unit": "mm"
            })
            
        # Analyze feature relationships
        for i, instance in enumerate(pattern.instances):
            relationships.append({
                "type": "instance",
                "index": i,
                "base_distance": self._calculate_distance(pattern.base_feature, instance)
            })
            
        return relationships

    def _generate_manufacturing_notes(self, pattern: PatternFeature) -> Dict[str, Any]:
        """Generate manufacturing considerations for the pattern."""
        if self.llm_client:
            return self.llm_client.generate_manufacturing_notes(pattern)
        return {"note": "Standard manufacturing process recommended"}

    def _generate_reuse_suggestions(self, pattern: PatternFeature) -> List[Dict[str, Any]]:
        """Generate suggestions for pattern reuse in other contexts."""
        if self.llm_client:
            return self.llm_client.generate_reuse_suggestions(pattern)
        return [{"suggestion": "Pattern can be reused in similar contexts"}]

    def _validate_linear_array(self, pattern: PatternFeature) -> bool:
        """Validate detected linear array pattern."""
        if len(pattern.instances) < 2:
            return False
        # Add more validation logic
        return True

    def _validate_circular_array(self, pattern: PatternFeature) -> bool:
        """Validate detected circular array pattern."""
        if len(pattern.instances) < 3:
            return False
        # Add more validation logic
        return True

    def _validate_symmetry(self, pattern: PatternFeature) -> bool:
        """Validate detected symmetry pattern."""
        # Implementation for symmetry validation
        return True

    def _validate_repetitive_feature(self, pattern: PatternFeature) -> bool:
        """Validate detected repetitive feature pattern."""
        # Implementation for repetitive feature validation
        return True 


================================================
File: CadSeqProc/enhanced_geometry/tests.py
================================================
"""
Test module for enhanced geometry system.
"""

import unittest
import numpy as np
from typing import List
from .base import Point, GeometricEntity
from .nurbs import NURBSCurve, NURBSSurface
from .organic import OrganicSurface, TwistDeformation
from .factory import OrganicShapeFactory
from .integration import GeometryAdapter, ShapeGenerator, ModelIntegration

class TestPoint(unittest.TestCase):
    def test_point_creation(self):
        p = Point(1.0, 2.0, 3.0)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 2.0)
        self.assertEqual(p.z, 3.0)
    
    def test_point_array_conversion(self):
        p = Point(1.0, 2.0, 3.0)
        arr = p.to_array()
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))
        
        p2 = Point.from_array(arr)
        self.assertEqual(p.x, p2.x)
        self.assertEqual(p.y, p2.y)
        self.assertEqual(p.z, p2.z)

class TestNURBS(unittest.TestCase):
    def test_curve_creation(self):
        points = [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(2.0, 0.0, 0.0)
        ]
        curve = NURBSCurve(points)
        
        # Test point sampling
        samples = curve.sample_points(5)
        self.assertEqual(len(samples), 5)
        
        # Test first and last points match control points
        np.testing.assert_array_almost_equal(
            samples[0].to_array(),
            points[0].to_array()
        )
        np.testing.assert_array_almost_equal(
            samples[-1].to_array(),
            points[-1].to_array()
        )
    
    def test_surface_creation(self):
        points = [
            [Point(0,0,0), Point(0,1,0)],
            [Point(1,0,0), Point(1,1,1)]
        ]
        surface = NURBSSurface(points)
        
        # Test point sampling
        samples = surface.sample_points(3)
        self.assertEqual(len(samples), 9)  # 3x3 grid
        
        # Test corners match control points
        np.testing.assert_array_almost_equal(
            samples[0].to_array(),
            points[0][0].to_array()
        )
        np.testing.assert_array_almost_equal(
            samples[-1].to_array(),
            points[-1][-1].to_array()
        )

class TestOrganicShape(unittest.TestCase):
    def test_deformation(self):
        # Create a simple surface
        points = [
            [Point(0,0,0), Point(0,1,0)],
            [Point(1,0,0), Point(1,1,0)]
        ]
        surface = NURBSSurface(points)
        organic = OrganicSurface([surface])
        
        # Apply twist deformation
        organic.apply_deformation('twist', {
            'axis': [0, 0, 1],
            'angle': np.pi/4,
            'center': [0.5, 0.5, 0]
        })
        
        # Sample points and verify they're not all in z=0 plane
        samples = organic.sample_points(5)
        z_coords = [p.z for p in samples]
        self.assertTrue(any(z != 0 for z in z_coords))

class TestFactory(unittest.TestCase):
    def test_flower_creation(self):
        factory = OrganicShapeFactory()
        flower = factory.create_flower(
            n_petals=5,
            petal_length=2.0,
            petal_width=1.0,
            center_radius=1.0,
            center_height=0.5
        )
        
        # Verify flower has correct number of surfaces
        # (1 center + n_petals surfaces)
        self.assertEqual(len(flower.control_surfaces), 6)
    
    def test_leaf_creation(self):
        factory = OrganicShapeFactory()
        leaf = factory.create_leaf(
            length=3.0,
            width=1.5,
            curve_factor=0.2
        )
        
        # Verify leaf has one surface
        self.assertEqual(len(leaf.control_surfaces), 1)
        
        # Sample points and verify dimensions
        samples = leaf.sample_points(10)
        xs = [p.x for p in samples]
        ys = [p.y for p in samples]
        
        self.assertLess(max(xs), 3.1)  # Length
        self.assertLess(max(ys), 0.8)  # Half width

class TestIntegration(unittest.TestCase):
    def test_shape_generation(self):
        integration = ModelIntegration()
        
        # Test flower generation
        result = integration.process_text_input("Create a flower with 5 petals")
        self.assertEqual(result['metadata']['generation_status'], 'success')
        self.assertEqual(result['cad_sequence']['type'], 'organic')
        
        # Test validation
        self.assertTrue(
            integration.validate_sequence(result['cad_sequence'])
        )
    
    def test_geometry_adapter(self):
        # Create a simple curve
        points = [Point(0,0,0), Point(1,1,0)]
        curve = NURBSCurve(points)
        
        # Convert to sequence and back
        adapter = GeometryAdapter()
        sequence = adapter.to_cad_sequence(curve)
        curve2 = adapter.from_cad_sequence(sequence)
        
        # Verify points match
        np.testing.assert_array_almost_equal(
            curve.control_points[0].to_array(),
            curve2.control_points[0].to_array()
        )

def run_tests():
    unittest.main() 

================================================
File: CadSeqProc/examples/pattern_recognition_demo.py
================================================
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

================================================
File: CadSeqProc/geometry/arc.py
================================================
import os
import sys
from pathlib import Path

sys.path.append("..")

sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import matplotlib.pyplot as plt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.GC import GC_MakeArcOfCircle
from loguru import logger
from rich import print
from CadSeqProc.utility.utils import (
    angle_from_vector_to_x,
    rads_to_degs,
    create_point_from_array,
    quantize,
    dequantize_verts,
    coord_to_pixel,
    create_point,
    get_orientation,
    pixel_to_coord,
    int_round,
    float_round,
    find_arc_geometry,
    point_distance,
)
import torch
import matplotlib.patches as patches
from CadSeqProc.geometry.line import Line
from CadSeqProc.geometry.curve import Curve
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.logger import CLGLogger
import numpy as np


clglogger = CLGLogger().configure_logger().logger


class Arc(Curve):
    def __init__(self, metadata):
        self.metadata = metadata
        self.is_numerical = False

    @staticmethod
    def from_dict(arc_entity: dict):
        metadata = {
            "start_point": np.array(
                [arc_entity["start_point"]["x"], arc_entity["start_point"]["y"]]
            ),
            "end_point": np.array(
                [arc_entity["end_point"]["x"], arc_entity["end_point"]["y"]]
            ),
            "center": np.array(
                [arc_entity["center_point"]["x"], arc_entity["center_point"]["y"]]
            ),
            "radius": arc_entity["radius"],
            "normal": np.array(
                [
                    arc_entity["normal"]["x"],
                    arc_entity["normal"]["y"],
                    arc_entity["normal"]["z"],
                ]
            ),
            "start_angle": arc_entity["start_angle"],
            "end_angle": arc_entity["end_angle"],
            "ref_vec": np.array(
                [
                    arc_entity["reference_vector"]["x"],
                    arc_entity["reference_vector"]["y"],
                ]
            ),
        }

        mid_angle = (metadata["start_angle"] + metadata["end_angle"]) / 2
        rot_mat = np.array(
            [
                [np.cos(mid_angle), -np.sin(mid_angle)],
                [np.sin(mid_angle), np.cos(mid_angle)],
            ]
        )
        mid_point = Arc.get_mid_point_arc(
            metadata["center"], metadata["radius"], metadata["ref_vec"], rot_mat
        )
        metadata["mid_point"] = mid_point
        metadata["rotation_matrix"] = rot_mat
        metadata["mid_angle"] = mid_angle

        return Arc(metadata)

    def to_vec(self):
        """
        vector representation of arc
        """
        assert self.is_numerical is True, clglogger.error(
            "The points are not quantized."
        )
        coord_token = [
            (self.metadata["start_point"] + END_PAD + BOOLEAN_PAD).tolist(),
            (self.metadata["mid_point"] + END_PAD + BOOLEAN_PAD).tolist(),
            [self.token_index, 0],
        ]
        return coord_token

    @staticmethod
    def get_mid_point_arc(center, radius, ref_vec, rot_mat):
        mid_vec = rot_mat @ ref_vec
        return center + mid_vec * radius

    def __repr__(self) -> str:
        if "center" in self.metadata:
            center = self.metadata["center"].round(4)
        else:
            center = "None"
        arc_repr = "{}: Start({}), Mid({}), End({}), Center({}) ".format(
            self.__class__.__name__,
            self.metadata["start_point"].round(4),
            self.metadata["mid_point"].round(4),
            self.metadata["end_point"].round(4),
            center,
        )
        
        return arc_repr


    @property
    def curve_type(self):
        return "arc"

    def reverse(self):
        self.metadata["start_point"], self.metadata["end_point"] = (
            self.metadata["end_point"],
            self.metadata["start_point"],
        )

    def get_point(self, point_type):
        return self.metadata[point_type]
   
    def transform(self, translate, scale):
        self.metadata["start_point"] = (
            self.metadata["start_point"] + translate
        ) * scale
        self.metadata["end_point"] = (
            self.metadata["end_point"] + translate
        ) * scale
        self.metadata["mid_point"] = (
            self.metadata["mid_point"] + translate
        ) * scale
        if "center" in self.metadata:
            self.metadata["center"] = (self.metadata["center"] + translate) * scale

        if "radius" in self.metadata:
            self.metadata["radius"] *= scale

    @staticmethod
    def from_vec(vec, bit=N_BIT, post_processing=False):
        
        metadata = {}

        vec -= END_PAD + BOOLEAN_PAD

        # pixel_to_coord_value=pixel_to_coord(vec,bit=bit).numpy()-(END_PAD+BOOLEAN_PAD)
        metadata["start_point"] = vec[0]
        metadata["mid_point"] = vec[1]
        metadata["end_point"] = vec[2]

        if (
            get_orientation(
                metadata["start_point"],
                metadata["mid_point"],
                metadata["end_point"],
            )
            == "collinear"
        ):
            if post_processing:
                # If three points are collinear,make them a line
                line = Line(metadata=metadata)
                line.quantized_metadata = metadata.copy()
                return line
            else:
                # raise Exception(f"Collinear points {metadata}")
                pass
        arc = Arc(metadata=metadata)
        arc.quantized_metadata = metadata.copy()
        arc.bit = bit
        return arc

    def get_angles_counterclockwise(self, eps=1e-8):
        c2s_vec = (self.metadata["start_point"] - self.metadata["center"]) / (
            np.linalg.norm(self.metadata["start_point"] - self.metadata["center"]) + eps
        )
        c2m_vec = (self.metadata["mid_point"] - self.metadata["center"]) / (
            np.linalg.norm(self.metadata["mid_point"] - self.metadata["center"]) + eps
        )
        c2e_vec = (self.metadata["end_point"] - self.metadata["center"]) / (
            np.linalg.norm(self.metadata["end_point"] - self.metadata["center"]) + eps
        )
        angle_s, angle_m, angle_e = (
            angle_from_vector_to_x(c2s_vec),
            angle_from_vector_to_x(c2m_vec),
            angle_from_vector_to_x(c2e_vec),
        )
        angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)
        if not angle_s < angle_m < angle_e:
            angle_s, angle_e = angle_e - np.pi * 2, angle_s
        return angle_s, angle_e

    def direction(self, from_start=True):
        if from_start:
            return self.metadata["mid_point"] - self.metadata["start_point"]
        else:
            return self.metadata["end_point"] - self.metadata["mid_point"]

    @property
    def bbox(self):
        points = [
            self.metadata["start_point"],
            self.metadata["mid_point"],
            self.metadata["end_point"],
        ]
        points = np.stack(points, axis=0)
        
        return np.stack([np.min(points, axis=0), np.max(points, axis=0)], axis=0)

    @property
    def bbox_size(self):
        bbox_size = np.max(np.abs(self.bbox[1] - self.bbox[0]))
        if bbox_size == 0:
            return 1
        else:
            return bbox_size

    @property
    def start_point(self):
        return self.metadata["start_point"]

    @property
    def clock_sign(self):
        """get a boolean sign indicating whether the arc is on top of s->e"""
        s2e = self.metadata["end_point"] - self.metadata["start_point"]
        s2m = self.metadata["mid_point"] - self.metadata["start_point"]
        sign = np.cross(s2m, s2e) >= 0  # counter-clockwise
        return sign

    def draw(self, ax=None, color="black"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,10))
        ref_vec_angle = rads_to_degs(angle_from_vector_to_x(self.get_point("ref_vec")))
        start_angle = rads_to_degs(self.get_point("start_angle"))
        end_angle = rads_to_degs(self.get_point("end_angle"))
        diameter = 2.0 * self.metadata["radius"]
        ap = patches.Arc(
            (self.metadata["center"][0], self.metadata["center"][1]),
            diameter,
            diameter,
            angle=ref_vec_angle,
            theta1=start_angle,
            theta2=end_angle,
            lw=1,
            color=color,
        )
        ax.add_patch(ap)
        
    def sample_points(self, n_points=32):
        if "center" not in self.metadata.keys():
            center, radius, _, _, _ = find_arc_geometry(
                self.metadata["start_point"],
                self.metadata["mid_point"],
                self.metadata["end_point"],
            )
            self.metadata["center"] = center
            self.metadata["radius"] = radius
        # print(self.metadata)
        c2s_vec = (
            self.metadata["start_point"] - self.metadata["center"]
        ) / np.linalg.norm(self.metadata["start_point"] - self.metadata["center"])
        c2m_vec = (
            self.metadata["mid_point"] - self.metadata["center"]
        ) / np.linalg.norm(self.metadata["mid_point"] - self.metadata["center"])
        c2e_vec = (
            self.metadata["end_point"] - self.metadata["center"]
        ) / np.linalg.norm(self.metadata["end_point"] - self.metadata["center"])
        angle_s, angle_m, angle_e = (
            angle_from_vector_to_x(c2s_vec),
            angle_from_vector_to_x(c2m_vec),
            angle_from_vector_to_x(c2e_vec),
        )
        angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)
        if not angle_s < angle_m < angle_e:
            angle_s, angle_e = angle_e - np.pi * 2, angle_s

        angles = np.linspace(angle_s, angle_e, num=n_points)
        points = (
            np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.metadata["radius"]
            + self.metadata["center"][np.newaxis]
        )
        
        return points

    def is_collinear(self, curve: Curve):
        return super().is_collinear()

    def build_body(self, coordsystem=None):
        """
        Requires start point, end point and mid point

        """
        assert coordsystem is not None, clglogger.error(
            f"Requires Coordinate system for building {self.curve_type}."
        )
        start_point = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["start_point"])
        )
        mid_point = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["mid_point"])
        )
        end_point = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["end_point"])
        )
        arc_occ = GC_MakeArcOfCircle(start_point, mid_point, end_point).Value()

        topo_edge = BRepBuilderAPI_MakeEdge(arc_occ).Edge()

        return topo_edge

    @property
    def one_point(self):
        return self.metadata["start_point"]

    def numericalize(self, bit=N_BIT):
        self.is_numerical = True
        self.bit = bit
        
        size = 2**bit - 1
        self.metadata["start_point"] = int_round(
            np.clip(self.metadata["start_point"], a_min=0, a_max=size)
        )
        self.metadata["mid_point"] = int_round(
            np.clip(self.metadata["mid_point"], a_min=0, a_max=size)
        )
        self.metadata["end_point"] = int_round(
            np.clip(self.metadata["end_point"], a_min=0, a_max=size)
        )

        # If the quantized values becomes invalid during json processing, perform some changes in the vector.
        # This slight change won't affect the model that much.

        if self.metadata["start_point"][0] == self.metadata["mid_point"][0]:
            if self.metadata["mid_point"][0] < 255:
                self.metadata["mid_point"][0] += 1
            else:
                self.metadata["mid_point"][0] -= 1
        if self.metadata["mid_point"][0] == self.metadata["end_point"][0]:
            if self.metadata["mid_point"][0] < 255:
                self.metadata["mid_point"][0] += 1
            else:
                self.metadata["mid_point"][0] -= 1

        if self.metadata["start_point"][1] == self.metadata["mid_point"][1]:
            if self.metadata["mid_point"][1] < 255:
                self.metadata["mid_point"][1] += 1
            else:
                self.metadata["mid_point"][1] -= 1
        if self.metadata["mid_point"][1] == self.metadata["end_point"][1]:
            if self.metadata["mid_point"][1] < 255:
                self.metadata["mid_point"][1] += 1
            else:
                self.metadata["mid_point"][1] -= 1

    def denumericalize(self, bit=N_BIT):
        self.is_numerical = True
        self.metadata["start_point"] = dequantize_verts(
            verts=self.metadata["start_point"], n_bits=bit, min_range=0, max_range=1
        )
        self.metadata["mid_point"] = dequantize_verts(
            verts=self.metadata["mid_point"], n_bits=bit, min_range=0, max_range=1
        )
        self.metadata["end_point"] = dequantize_verts(
            verts=self.metadata["end_point"], n_bits=bit, min_range=0, max_range=1
        )

    def accuracyReport(self, target, tolerance):
        # # De-quantize the parameters between (0 and 1) for comparison purposes
        # self.transform(translate=0,scale=1/255)
        # target.transform(translate=0,scale=1/255)


        self.arc_parameter_correct={"s":np.array([0,0]),
                                    "m":np.array([0,0]),
                                    "e":np.array([0,0])}
        # For Start Point 
        self.arc_parameter_correct['s'][0] +=  np.abs(self.metadata['start_point'][0]
                                                -target.metadata['start_point'][0])/self.bbox_size
        
        
        self.arc_parameter_correct['s'][1] +=  np.abs(self.metadata['start_point'][1]
                                                -target.metadata['start_point'][1])/self.bbox_size
        
        # For Mid Point 
        self.arc_parameter_correct['m'][0] +=  np.abs(self.metadata['mid_point'][0]
                                                -target.metadata['mid_point'][0])/self.bbox_size
        
        self.arc_parameter_correct['m'][1] +=  np.abs(self.metadata['mid_point'][1]
                                                -target.metadata['mid_point'][1])/self.bbox_size
        
        # For End Point 
        self.arc_parameter_correct['e'][0] +=  np.abs(self.metadata['end_point'][0]
                                                -target.metadata['end_point'][0])/self.bbox_size
        
        self.arc_parameter_correct['e'][1] += np.abs(self.metadata['end_point'][1]
                                                -target.metadata['end_point'][1])/self.bbox_size

        return self.arc_parameter_correct


    def curve_distance(self, pred_curve, scale):
        return super().curve_distance(pred_curve, scale)

    def _json(self):
        if "center" in self.metadata:
            center = self.metadata["center"].round(4)
        else:
            center = "None"
        arc_json = {
            "Start Point": list(float_round(self.metadata["start_point"])),
            "Mid Point": list(float_round(self.metadata["mid_point"])),
            "End Point": list(float_round(self.metadata["end_point"]))
        }

        return arc_json


if __name__ == "__main__":
    arc_dict = {
        "center_point": {"y": -0.00040928, "x": -0.00040928, "z": 0.0},
        "normal": {"y": 0.0, "x": 0.0, "z": 1.0},
        "end_point": {"y": -0.04871051, "x": -0.01, "z": 0.0},
        "start_angle": 0.0,
        "curve": "JGt",
        "end_angle": 1.1787740968698315,
        "radius": 0.0492442,
        "type": "Arc3D",
        "start_point": {"y": -0.01, "x": -0.04871051, "z": 0.0},
        "reference_vector": {
            "y": -0.19475838772210444,
            "x": -0.9808512478515213,
            "z": 0.0,
        },
    }

    arc = Arc.from_dict(arc_dict)
    print(arc._json())


================================================
File: CadSeqProc/geometry/circle.py
================================================
import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import (
    create_point_from_array,
    get_plane_normal,
    quantize,
    dequantize_verts,
    point_distance,
    coord_to_pixel,
    create_point,
    pixel_to_coord,
    int_round,
    float_round
)
import torch
from CadSeqProc.geometry.curve import Curve
import matplotlib.patches as patches
from rich import print
from loguru import logger
import matplotlib.pyplot as plt
from OCC.Core.gp import gp_Circ, gp_Ax2, gp_Dir
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

clglogger = CLGLogger().configure_logger().logger


class Circle(Curve):
    def __init__(self, metadata):
        self.metadata = metadata
        self.is_numerical = False

    @staticmethod
    def from_dict(circle_entity: dict):
        metadata = {
            "center": np.array(
                [circle_entity["center_point"]["x"], circle_entity["center_point"]["y"]]
            ),
            "radius": circle_entity["radius"],
            "normal": np.array(
                [
                    circle_entity["normal"]["x"],
                    circle_entity["normal"]["y"],
                    circle_entity["normal"]["z"],
                ]
            ),
        }

        # Get 4 points on the circle
        metadata["pt1"] = np.array(
            [metadata["center"][0], metadata["center"][1] + metadata["radius"]]
        )
        metadata["pt2"] = np.array(
            [metadata["center"][0], metadata["center"][1] - metadata["radius"]]
        )
        metadata["pt3"] = np.array(
            [metadata["center"][0] + metadata["radius"], metadata["center"][1]]
        )
        metadata["pt4"] = np.array(
            [metadata["center"][0] - metadata["radius"], metadata["center"][1]]
        )

        return Circle(metadata)

    @property
    def bbox(self):
        return np.stack(
            [
                self.metadata["center"] - self.metadata["radius"],
                self.metadata["center"] + self.metadata["radius"],
            ],
            axis=0,
        )
    
    @property
    def bbox_size(self):
        bbox_size = np.max(np.abs(self.bbox[1] - self.bbox[0]))
        if bbox_size == 0:
            return 1
        else:
            return bbox_size

    def direction(self):
        return self.metadata["center"] - self.start_point

    @property
    def start_point(self):
        """Changing start point will change circle quantized values as well as its translation"""
        return self.bbox[0]

    @property
    def end_point(self):
        return np.array(
            [
                self.metadata["center"][0] + self.metadata["radius"],
                self.metadata["center"][1],
            ]
        )

    def to_vec(self):
        """
        vector representation of circle
        """
        assert self.is_numerical is True, clglogger.error(
            "The points are not quantized."
        )
        coord_token = [
            (self.metadata["center"] + END_PAD + BOOLEAN_PAD).tolist(),
            (self.metadata["pt1"] + END_PAD + BOOLEAN_PAD).tolist(),
            [self.token_index, 0],
        ]
        return coord_token

    @staticmethod
    def from_vec(vec, bit, post_processing):
        metadata = {}
        vec -= END_PAD + BOOLEAN_PAD
        metadata["center"] = vec[0]
        metadata["pt1"] = vec[1]
        metadata["radius"] = point_distance(metadata["center"], metadata["pt1"])
        circle = Circle(metadata=metadata)
        circle.quantized_metadata = metadata.copy()
        circle.bit = bit
        return circle

    def sample_points(self, n_points=1024):
        angles = np.linspace(0, np.pi * 2, num=n_points, endpoint=False)
        points = (
            np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.metadata["radius"]
            + self.metadata["center"]
        )
        return points

    def draw(self, ax=None, color="black"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        ap = patches.Circle(
            (self.metadata["center"][0], self.metadata["center"][1]),
            self.metadata["radius"],
            lw=1,
            fill=None,
            color=color,
        )
        ax.add_patch(ap)
        # ax.plot(self.metadata['center'][0], self.metadata['center'][1], 'ok')

    def __repr__(self) -> str:
        circle_repr = f"{self.__class__.__name__}: center({self.metadata['center'].round(4)}), \
            radius({round(self.metadata['radius'], 4)}), pt1 {self.metadata['pt1']}"
    
        return circle_repr

    @property
    def curve_type(self):
        return "circle"

    def get_point(self, point_type):
        return self.metadata[point_type]
        
    def is_collinear(self, curve: Curve):
        return super().is_collinear()

    def transform(self, translate, scale=1):
        self.metadata["center"] = (self.metadata["center"] + translate) * scale
        self.metadata["pt1"] = (self.metadata["pt1"] + translate) * scale
        if "radius" in self.metadata:
            self.metadata["radius"] *= scale
        else:
            self.metadata["radius"] = abs(
                float(
                    point_distance(
                        self.metadata["center"], self.metadata["pt1"], type="l1"
                    )
                )
            )
            if hasattr(self, "quantized_metadata"):
                self.quantized_metadata["radius"] = int_round(
                    [
                        np.clip(
                            self.metadata["radius"] / scale,
                            a_min=0,
                            a_max=2**self.bit - 1,
                        )
                    ]
                )[0]

    def build_body(self,normal=None, coordsystem=None):
        """
        Requires Center, uppermost point and normal, transform(optional for build_type 2)
        """
        
        assert coordsystem is not None and normal is not None, clglogger.error(
            f"Requires Coordinate System for building {self.curve_type}."
        )

        center = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["center"])
        )
        radius = abs(
            float(
                point_distance(
                    self.metadata["center"], self.metadata["pt1"], type="l1"
                )
            )
        )

        axis = gp_Ax2(center, gp_Dir(*normal))
        circle = gp_Circ(axis, radius)
        topo_edge = BRepBuilderAPI_MakeEdge(circle).Edge()

        return topo_edge

    @property
    def one_point(self):
        return self.metadata["center"]

    def numericalize(self, bit=N_BIT):
        self.is_numerical = True
        self.bit = bit
        size = 2**bit - 1
        self.metadata["pt1"] = int_round(
            np.clip(self.metadata["pt1"], a_min=0, a_max=size)
        )
        self.metadata["pt2"] = int_round(
            np.clip(self.metadata["pt2"], a_min=0, a_max=size)
        )
        self.metadata["pt3"] = int_round(
            np.clip(self.metadata["pt3"], a_min=0, a_max=size)
        )
        self.metadata["pt4"] = int_round(
            np.clip(self.metadata["pt4"], a_min=0, a_max=size)
        )
        self.metadata["center"] = int_round(
            np.clip(self.metadata["center"], a_min=0, a_max=size)
        )
        self.metadata["radius"] = int_round(
            [np.clip(self.metadata["radius"], a_min=0, a_max=size)]
        )[0]

        if self.metadata["pt1"][1] == self.metadata["center"][1]:
            if self.metadata["pt1"][1] < 255:
                self.metadata["pt1"][1] += 1
            else:
                self.metadata["pt1"][1] -= 1

    def denumericalize(self, bit=N_BIT):
        self.is_numerical = False
        
        self.metadata["pt1"] = dequantize_verts(
            verts=self.metadata["pt1"], n_bits=bit, min_range=-1, max_range=1
        )
        self.metadata["pt2"] = dequantize_verts(
            verts=self.metadata["pt2"], n_bits=bit, min_range=-1, max_range=1
        )
        self.metadata["pt3"] = dequantize_verts(
            verts=self.metadata["pt3"], n_bits=bit, min_range=-1, max_range=1
        )
        self.metadata["pt4"] = dequantize_verts(
            verts=self.metadata["pt4"], n_bits=bit, min_range=-1, max_range=1
        )
        self.metadata["center"] = dequantize_verts(
            verts=self.metadata["center"], n_bits=bit, min_range=-1, max_range=1
        )
        self.metadata["radius"] = dequantize_verts(
            verts=self.metadata["radius"], n_bits=bit, min_range=-1, max_range=1
        )

    def accuracyReport(self, target, tolerance):

        # De-quantize the parameters between (0 and 1) for comparison purposes
        # self.transform(translate=0,scale=1/255)
        # target.transform(translate=0,scale=1/255)

        self.circle_parameter_report = {"c": np.array([0, 0]), "r": np.array([0, 0])}

        self.circle_parameter_report["c"][0] += (
            np.abs(self.metadata["center"][0] - target.metadata["center"][0])
            / self.bbox_size
        )

        self.circle_parameter_report["c"][1] += (
            np.abs(self.metadata["center"][1] - target.metadata["center"][1])
            / self.bbox_size
        )

        self.circle_parameter_report["r"][0] += (
            np.abs(self.metadata["radius"] - target.metadata["radius"]) / self.bbox_size
        )
        self.circle_parameter_report["r"][
            1
        ] += 1  # Number of radius considered (not used anymore. dummy value)

        return self.circle_parameter_report

    def curve_distance(self, pred_curve, scale):
        return super().curve_distance(pred_curve, scale)

    def _json(self):
        circle_json = {
            "Center": list(float_round(self.metadata["center"])),
            "Radius": float(float_round(self.metadata["radius"]))
        }
        return circle_json


if __name__ == "__main__":
    circle_dict = {
        "center_point": {"y": 0.0762, "x": 0.0, "z": 0.0},
        "type": "Circle3D",
        "radius": 0.06000001,
        "curve": "JGR",
        "normal": {"y": 0.0, "x": 1.0, "z": 0.0},
    }
    circle = Circle.from_dict(circle_dict)
    print(circle)


================================================
File: CadSeqProc/geometry/curve.py
================================================

import os
import sys
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3])) # ROOT_DIR/CADLGen/DataProc

from abc import ABC, abstractmethod
from loguru import logger
from CadSeqProc.utility.utils import point_distance
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.logger import CLGLogger
import numpy as np



clglogger = CLGLogger().configure_logger().logger


class Curve(ABC):

    def sample_points(self, n_points, sample_type):
        raise NotImplementedError

    def to_vec(self):
        raise NotImplementedError

    @staticmethod
    def from_dict():
        raise NotImplementedError

    @property
    def bbox(self):
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError

    @property
    def curve_type(self):
        return self.__class__.__name__

    @property
    def bbox(self):
        raise NotImplementedError

    def is_collinear(self):
        return False

    def build_body(self):
        raise NotImplementedError

    def numericalize(self):
        raise NotImplementedError

    def denumericalize(self):
        raise NotImplementedError

    @property
    def token_index(self):
        return SKETCH_TOKEN.index("END_CURVE")

    def curve_distance(self, pred_curve, scale):
        return point_distance(self.bbox*scale, pred_curve.bbox*scale, type="l2")


================================================
File: CadSeqProc/geometry/line.py
================================================
import copy
import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import (
    coord_to_pixel,
    float_round,
    create_point_from_array,
    dequantize_verts,
    int_round,
    pixel_to_coord,
    quantize,
    point_distance,
)
from CadSeqProc.geometry.curve import Curve
from rich import print
import torch
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.gp import gp_Pnt

clglogger = CLGLogger().configure_logger().logger


class Line(Curve):
    def __init__(self, metadata):
        self.metadata = metadata

        self.is_numerical = False

    @staticmethod
    def from_dict(line_entity: dict):
        metadata = {}
        metadata["start_point"] = np.array(
            [line_entity["start_point"]["x"], line_entity["start_point"]["y"]]
        )
        metadata["end_point"] = np.array(
            [line_entity["end_point"]["x"], line_entity["end_point"]["y"]]
        )
        return Line(metadata)

    def to_vec(self):
        """
        vector representation of line
        """
        assert self.is_numerical is True, clglogger.error(
            "The points are not quantized."
        )
        coord_token = [
            (self.metadata["start_point"] + END_PAD + BOOLEAN_PAD).tolist(),
            [self.token_index, 0],
        ]
        return coord_token

    @staticmethod
    def from_vec(vec, bit=N_BIT, post_processing=False):
        metadata = {}
        vec -= END_PAD + BOOLEAN_PAD
        metadata["start_point"] = vec[0]
        metadata["end_point"] = vec[1]
        line = Line(metadata=metadata)
        line.quantized_metadata = metadata.copy()
        line.bit = bit
        return line

    def sample_points(self, n_points=32):
        points = np.linspace(
            self.metadata["start_point"], self.metadata["end_point"], num=n_points
        )
        
        return points

    @property
    def min_point(self):
        if np.all(
            self.metadata["start_point"] <= self.metadata["end_point"]
        ):
            return self.metadata["start_point"]
        else:
            return self.metadata["end_point"]
        

    def __repr__(self):
        line_repr = "{}: Start({}), End({})".format(
            self.__class__.__name__,
            self.metadata["start_point"].round(4),
            self.metadata["end_point"].round(4),
        )
        return line_repr

    @property
    def curve_type(self):
        return "line"

    @property
    def start_point(self):
        return self.metadata["start_point"]

    def get_point(self, point_type):
        return self.metadata[point_type]
       

    @property
    def bbox(self):
        points = np.stack(
            [self.metadata["start_point"], self.metadata["end_point"]], axis=0
        )
        return np.stack([np.min(points, axis=0), np.max(points, axis=0)], axis=0)


    def transform(self, translate, scale):
        """
        Transform the 2d points if 3D transformation is not done

        """
        # clglogger.debug(f"Before {translate} {scale} {self.metadata}")
        self.metadata["start_point"] = (
            self.metadata["start_point"] + translate
        ) * scale
        self.metadata["end_point"] = (
            self.metadata["end_point"] + translate
        ) * scale
        # clglogger.debug(f"After {self.metadata}")

    def reverse(self):
        self.metadata["start_point"], self.metadata["end_point"] = (
            self.metadata["end_point"],
            self.metadata["start_point"],
        )

    def merge(self, line: Curve):
        self.metadata["end_point"] = line.metadata["end_point"]

    def draw(self, ax=None, color="black"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        xdata = [self.metadata["start_point"][0], self.metadata["end_point"][0]]
        ydata = [self.metadata["start_point"][1], self.metadata["end_point"][1]]
        l1 = lines.Line2D(xdata, ydata, lw=1, color=color, axes=ax)
        ax.add_line(l1)
        # ax.plot(self.metadata['start_point'][0], self.metadata['start_point'][1], 'ok', color=color)

    def is_collinear(self, curve: Curve):
        if curve.curve_type == "arc" or curve.curve_type == "circle":
            return False
        else:
            # Calculate the direction vectors of both lines
            direction_self = self.get_point("end_point") - self.get_point("start_point")
            direction_other = curve.get_point("end_point") - curve.get_point(
                "start_point"
            )

            # Normalize the direction vectors
            direction_self_norm = direction_self / np.linalg.norm(direction_self)
            direction_other_norm = direction_other / np.linalg.norm(direction_other)
            # Check if the direction vectors are parallel or anti-parallel
            dot_product = np.dot(direction_self_norm, direction_other_norm)
            if np.isclose(dot_product, 1.0) or np.isclose(dot_product, -1.0):
                return True
            else:
                return False

    def build_body(self, coordsystem=None):
        """
        Requires start point and end point ,transform(only for build type 2)
        """

        assert coordsystem is not None, clglogger.error(
            f"Requires Coordinate system for building {self.curve_type}."
        )
        start_point = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["start_point"])
        )
        end_point = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["end_point"])
        )

        topo_edge = BRepBuilderAPI_MakeEdge(start_point, end_point).Edge()

        return topo_edge

    @property
    def bbox_size(self):
        bbox_size = np.max(np.abs(self.bbox[1] - self.bbox[0]))
        if bbox_size == 0:
            return 1
        else:
            return bbox_size

    @property
    def one_point(self):
        return self.metadata["start_point"]

    def numericalize(self, bit=N_BIT):
        self.is_numerical = True
        self.bit = bit
        size = 2**bit - 1
        # clglogger.debug(f"{self.metadata['start_point']}")
        self.metadata["start_point"] = int_round(
            np.clip(self.metadata["start_point"], a_min=0, a_max=size)
        )
        self.metadata["end_point"] = int_round(
            np.clip(self.metadata["end_point"], a_min=0, a_max=size)
        )

    def denumericalize(self, bit=N_BIT):
        self.is_numerical = False
        self.metadata["start_point"] = dequantize_verts(
            verts=self.metadata["start_point"],
            n_bits=bit,
            min_range=-1,
            max_range=1,
        )
        self.metadata["end_point"] = dequantize_verts(
            verts=self.metadata["end_point"], n_bits=bit, min_range=-1, max_range=1
        )

    def accuracyReport(self, target, tolerance):

        # De-quantize the parameters between (0 and 1) for comparison purposes
        # self.transform(translate=0,scale=1/255)
        # target.transform(translate=0,scale=1/255)
        # print(self.bbox_size)

        self.line_parameter_correct = {"s": np.array([0, 0]), "e": np.array([0, 0])}

        # For Start Point
        self.line_parameter_correct["s"][0] += (
            np.abs(self.metadata["start_point"][0] - target.metadata["start_point"][0])
            / self.bbox_size
        )

        self.line_parameter_correct["s"][1] += (
            np.abs(self.metadata["start_point"][1] - target.metadata["start_point"][1])
            / self.bbox_size
        )

        # For End Point
        self.line_parameter_correct["e"][0] += (
            np.abs(self.metadata["end_point"][0] - target.metadata["end_point"][0])
            / self.bbox_size
        )

        self.line_parameter_correct["e"][1] += (
            np.abs(self.metadata["end_point"][1] - target.metadata["end_point"][1])
            / self.bbox_size
        )

        return self.line_parameter_correct

    def curve_distance(self, pred_curve, scale):
        return super().curve_distance(pred_curve, scale)

    def _json(self):
        line_json = {
            "Start Point": list(float_round(self.metadata["start_point"])),
            "End Point": list(float_round(self.metadata["end_point"]))
        }
        return line_json


if __name__ == "__main__":
    line_dict = {
        "type": "Line3D",
        "start_point": {"y": -0.01, "x": -0.01, "z": 0.0},
        "curve": "JGp",
        "end_point": {"y": -0.04871051, "x": -0.01, "z": 0.0},
    }

    line = Line.from_dict(line_dict)
    print(line)


================================================
File: CadSeqProc/geometry/nurbs.py
================================================
"""
NURBS (Non-Uniform Rational B-Spline) implementation for curve and surface manipulation.
"""

import numpy as np
from typing import List, Tuple, Optional
from .curve import Curve

def find_span(n: int, p: int, u: float, U: List[float]) -> int:
    """Find the knot span index."""
    if u >= U[n]:
        return n
    if u <= U[p]:
        return p
    
    low = p
    high = n + 1
    mid = (low + high) // 2
    
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    
    return mid

def basis_funs(i: int, u: float, p: int, U: List[float]) -> List[float]:
    """Compute the nonzero basis functions."""
    N = [0.0] * (p + 1)
    left = [0.0] * (p + 1) 
    right = [0.0] * (p + 1)
    
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        N[j] = saved
    
    return N

class NURBSCurve(Curve):
    """NURBS curve implementation."""
    
    def __init__(self, 
                 control_points: List[Tuple[float, float, float]],
                 weights: Optional[List[float]] = None,
                 knots: Optional[List[float]] = None,
                 degree: int = 3):
        super().__init__()
        self.control_points = control_points
        self.weights = weights if weights is not None else [1.0] * len(control_points)
        self.degree = degree
        
        if knots is None:
            # Generate uniform knot vector
            n = len(control_points) - 1
            m = n + degree + 1
            self.knots = [0.0] * (degree + 1) + \
                        list(np.linspace(0, 1, m - 2*degree)) + \
                        [1.0] * (degree + 1)
        else:
            self.knots = knots
    
    def evaluate(self, u: float) -> Tuple[float, float, float]:
        """Evaluate the NURBS curve at parameter u."""
        n = len(self.control_points) - 1
        p = self.degree
        
        span = find_span(n, p, u, self.knots)
        N = basis_funs(span, u, p, self.knots)
        
        x = y = z = w = 0.0
        for i in range(p + 1):
            j = span - p + i
            weight = self.weights[j]
            point = self.control_points[j]
            
            factor = N[i] * weight
            x += factor * point[0]
            y += factor * point[1]
            z += factor * point[2]
            w += factor
        
        return (x/w, y/w, z/w)
    
    @classmethod
    def from_points(cls, points: List[Tuple[float, float, float]], degree: int = 3) -> 'NURBSCurve':
        """Create a NURBS curve interpolating the given points."""
        return cls(points, degree=degree)

class NURBSSurface:
    """NURBS surface implementation."""
    
    def __init__(self,
                 control_points: List[List[Tuple[float, float, float]]],
                 weights: Optional[List[List[float]]] = None,
                 u_knots: Optional[List[float]] = None,
                 v_knots: Optional[List[float]] = None,
                 degree_u: int = 3,
                 degree_v: int = 3):
        self.control_points = control_points
        self.degree_u = degree_u
        self.degree_v = degree_v
        
        nu = len(control_points) - 1
        nv = len(control_points[0]) - 1
        
        if weights is None:
            self.weights = [[1.0] * (nv + 1) for _ in range(nu + 1)]
        else:
            self.weights = weights
        
        if u_knots is None:
            mu = nu + degree_u + 1
            self.u_knots = [0.0] * (degree_u + 1) + \
                          list(np.linspace(0, 1, mu - 2*degree_u)) + \
                          [1.0] * (degree_u + 1)
        else:
            self.u_knots = u_knots
            
        if v_knots is None:
            mv = nv + degree_v + 1
            self.v_knots = [0.0] * (degree_v + 1) + \
                          list(np.linspace(0, 1, mv - 2*degree_v)) + \
                          [1.0] * (degree_v + 1)
        else:
            self.v_knots = v_knots
    
    def evaluate(self, u: float, v: float) -> Tuple[float, float, float]:
        """Evaluate the NURBS surface at parameters (u,v)."""
        nu = len(self.control_points) - 1
        nv = len(self.control_points[0]) - 1
        
        u_span = find_span(nu, self.degree_u, u, self.u_knots)
        v_span = find_span(nv, self.degree_v, v, self.v_knots)
        
        Nu = basis_funs(u_span, u, self.degree_u, self.u_knots)
        Nv = basis_funs(v_span, v, self.degree_v, self.v_knots)
        
        x = y = z = w = 0.0
        
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                ui = u_span - self.degree_u + i
                vj = v_span - self.degree_v + j
                
                weight = self.weights[ui][vj]
                point = self.control_points[ui][vj]
                
                factor = Nu[i] * Nv[j] * weight
                x += factor * point[0]
                y += factor * point[1]
                z += factor * point[2]
                w += factor
        
        return (x/w, y/w, z/w)
    
    @classmethod
    def from_points(cls, points: List[List[Tuple[float, float, float]]], 
                   degree_u: int = 3, degree_v: int = 3) -> 'NURBSSurface':
        """Create a NURBS surface interpolating the given points."""
        return cls(points, degree_u=degree_u, degree_v=degree_v) 

================================================
File: CadSeqProc/geometry/organic.py
================================================
"""
Organic surface implementation for complex shape generation.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .nurbs import NURBSCurve, NURBSSurface
from ..sequence.transformation.deform import DeformationOp

class OrganicSurface:
    """Surface with organic deformation capabilities."""
    
    def __init__(self,
                 control_surfaces: List[NURBSSurface],
                 deformations: Optional[List[DeformationOp]] = None):
        self.control_surfaces = control_surfaces
        self.deformations = deformations or []
    
    def add_deformation(self, deform: DeformationOp):
        """Add a deformation operation."""
        self.deformations.append(deform)
    
    def clear_deformations(self):
        """Remove all deformations."""
        self.deformations = []
    
    def sample_points(self, 
                     num_u: int = 20,
                     num_v: int = 20) -> List[List[Tuple[float, float, float]]]:
        """Sample points from the surface with deformations applied."""
        points = []
        
        # Sample points from each control surface
        for surface in self.control_surfaces:
            surface_points = []
            for u in np.linspace(0, 1, num_u):
                row = []
                for v in np.linspace(0, 1, num_v):
                    point = surface.evaluate(u, v)
                    
                    # Apply deformations in sequence
                    current_point = point
                    for deform in self.deformations:
                        current_point = deform.apply([current_point])[0]
                    
                    row.append(current_point)
                surface_points.append(row)
            points.append(surface_points)
        
        return points
    
    def get_bounding_box(self) -> Tuple[Tuple[float, float, float], 
                                      Tuple[float, float, float]]:
        """Get the bounding box of the surface."""
        points = self.sample_points()
        all_points = [p for surface in points for row in surface for p in row]
        
        min_x = min(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        min_z = min(p[2] for p in all_points)
        max_x = max(p[0] for p in all_points)
        max_y = max(p[1] for p in all_points)
        max_z = max(p[2] for p in all_points)
        
        return ((min_x, min_y, min_z), (max_x, max_y, max_z))
    
    def transform(self, 
                 matrix: np.ndarray) -> 'OrganicSurface':
        """Apply transformation matrix to the surface."""
        transformed_surfaces = []
        for surface in self.control_surfaces:
            # Transform control points
            new_points = []
            for row in surface.control_points:
                new_row = []
                for point in row:
                    p_homogeneous = np.array([*point, 1.0])
                    transformed = matrix @ p_homogeneous
                    new_row.append(tuple(transformed[:3] / transformed[3]))
                new_points.append(new_row)
            
            # Create new surface with transformed points
            transformed_surface = NURBSSurface(
                new_points,
                weights=surface.weights,
                u_knots=surface.u_knots,
                v_knots=surface.v_knots,
                degree_u=surface.degree_u,
                degree_v=surface.degree_v
            )
            transformed_surfaces.append(transformed_surface)
        
        return OrganicSurface(transformed_surfaces, self.deformations.copy())
    
    def to_nurbs(self) -> List[NURBSSurface]:
        """Convert to NURBS representation."""
        return self.control_surfaces 

================================================
File: CadSeqProc/sequence/sketch/coord_system.py
================================================
import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))


import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.utils import (
    float_round,
    int_round,
    polar_parameterization,
    quantize,
)
from CadSeqProc.utility.macro import *
from loguru import logger
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Ax3, gp_Dir, gp_Ax1, gp_Pnt
from rich import print
from scipy.spatial.transform import Rotation as R
import math

clglogger = CLGLogger().configure_logger().logger


class CoordinateSystem(object):

    def __init__(self, metadata) -> None:
        self.metadata = metadata
        self.is_numerical = False

    @staticmethod
    def from_dict(transform_dict):
        metadata = {}

        metadata["origin"] = np.array(
            [
                transform_dict["origin"]["x"],
                transform_dict["origin"]["y"],
                transform_dict["origin"]["z"],
            ]
        )
        metadata["x_axis"] = np.array(
            [
                transform_dict["x_axis"]["x"],
                transform_dict["x_axis"]["y"],
                transform_dict["x_axis"]["z"],
            ]
        )
        metadata["y_axis"] = np.array(
            [
                transform_dict["y_axis"]["x"],
                transform_dict["y_axis"]["y"],
                transform_dict["y_axis"]["z"],
            ]
        )
        metadata["z_axis"] = np.array(
            [
                transform_dict["z_axis"]["x"],
                transform_dict["z_axis"]["y"],
                transform_dict["z_axis"]["z"],
            ]
        )

        # theta,phi,gamma=polar_parameterization(metadata['z_axis'],metadata['x_axis'])
        euler_angles = R.from_matrix(
            np.vstack((metadata["x_axis"], metadata["y_axis"], metadata["z_axis"]))
        ).as_euler("zyx", degrees=False)

        metadata["euler_angles"] = euler_angles

        coord = CoordinateSystem(metadata=metadata)

        return coord

    @staticmethod
    def from_vec(vec, bit, post_processing):
        assert len(vec) == 6, clglogger.error(f"Wrong number of inputs {vec}")
        metadata = {}
        metadata["origin"] = vec[:3]
        metadata["euler_angles"] = vec[3:6]
        coord = CoordinateSystem(metadata=metadata)
        coord.quantized_metadata = metadata.copy()

        return coord

    def create_transform(self):
        """
        Requires Origin and the z-axis and gamma angle

        """
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(*self.metadata["origin"]))

        # Calculate the rotation angle and axis
        rotation_axis = gp_Ax1(
            gp_Pnt(*self.metadata["origin"]),
            gp_Dir(*self.metadata["z_axis"].astype(np.float64)),
        )
        rotation_angle = self.metadata["euler_angles"][2]

        # Set the rotation using the calculated angle and axis
        transform.SetRotation(rotation_axis, rotation_angle)

        return transform

    @property
    def normal(self):
        return self.metadata["z_axis"]

    def get_property(self, key):
        return self.metadata[key]

    def rotate_vec(self, vec, translation=True):
        if vec.shape[-1] == 2:
            if len(vec.shape) == 1:
                vec = np.concatenate([vec, np.zeros(1)])
            else:
                vec = np.hstack([vec, np.zeros((len(vec), 1))])

        # Create a rotation matrix using the axes from metadata
        rotation_matrix = np.column_stack(
            (self.metadata["x_axis"], self.metadata["y_axis"], self.metadata["z_axis"])
        ).T

        # Rotate the vector
        rotated_vector = vec @ rotation_matrix

        rotated_vector = rotated_vector

        if translation:
            return rotated_vector + self.metadata["origin"]
        else:
            return rotated_vector

    def __repr__(self) -> str:
        try:
            rotation_matrix = (
                [*self.metadata["x_axis"]]
                + [*self.metadata["y_axis"]]
                + [*self.metadata["z_axis"]]
            )
        except:
            rotation_matrix = None
        s = f"{self.__class__.__name__}:\n            - Rotation Matrix {rotation_matrix},\n            - Translation {self.metadata['origin']}"

        return s

    def transform(self, translate, scale):
        if not isinstance(translate, int) and not isinstance(translate, float):
            if translate.shape[0] != 3:
                translate = np.concatenate([translate, np.zeros(3 - len(translate))])
        self.metadata["origin"] = (self.metadata["origin"] + translate) * scale

    def numericalize(self, bit: int):
        """
        Quantization
        """
        self.is_numerical = True
        size = 2**bit - 1
        self.metadata["origin"] = int_round(
            ((self.metadata["origin"] + 1.0) / 2 * (size + 1)).clip(min=0, max=size)
        )
        self.metadata["euler_angles"] = int_round(
            ((self.metadata["euler_angles"] / np.pi + 1.0) / 2 * (size + 1)).clip(
                min=0, max=size
            )
        )

    def denumericalize(self, bit):
        """
        Dequantization
        """

        self.is_numerical = False
        size = 2**bit
        self.metadata["origin"] = self.metadata["origin"] / size * 2 - 1.0
        self.metadata["euler_angles"] = (
            self.metadata["euler_angles"] / size * 2 - 1.0
        ) * np.pi

        rot_matrix = R.from_euler(
            seq="zyx", angles=self.metadata["euler_angles"], degrees=False
        ).as_matrix()

        # x_axis,y_axis,z_axis=euler_to_axis(*self.metadata['euler_angles'])
        self.metadata["x_axis"] = rot_matrix[0]
        self.metadata["y_axis"] = rot_matrix[1]
        self.metadata["z_axis"] = rot_matrix[2]


        self.is_numerical = False

    def _json(self):
        return {
            "Euler Angles": [
                float(float_round(math.degrees(r_val)))
                for r_val in self.metadata["euler_angles"]
            ],
            "Translation Vector": list(float_round(self.metadata["origin"]))
        }


if __name__ == "__main__":
    transform_dict = {
        "origin": {"y": 0.0, "x": 0.0, "z": 0.0},
        "y_axis": {"y": 0.0, "x": 0.0, "z": 1.0},
        "x_axis": {"y": 0.0, "x": 1.0, "z": 0.0},
        "z_axis": {"y": -1.0, "x": 0.0, "z": 0.0},
    }

    cd = CoordinateSystem.from_dict(transform_dict)
    print(cd._json())


================================================
File: CadSeqProc/sequence/sketch/face.py
================================================
import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))


import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from .loop import LoopSequence
from CadSeqProc.utility.utils import (
    random_sample_points,
    perform_op,
    split_array,
    write_stl_file,
)
from loguru import logger
from OCC.Core.BRepCheck import (
    BRepCheck_Analyzer,
    BRepCheck_Result,
    BRepCheck_ListOfStatus,
)
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.ShapeFix import ShapeFix_Face
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
import matplotlib.pyplot as plt
from typing import List

clglogger = CLGLogger().configure_logger().logger


class FaceSequence(object):

    def __init__(self, loopdata: List[LoopSequence], reorder: bool = True) -> None:
        self.loopdata = loopdata
        self.quantize_metadata = {}

        if reorder:
            # Reorder Faces according to the minimum bounding box coordinates
            self.reorder()

    @property
    def token_index(self):
        return SKETCH_TOKEN.index("END_FACE")

    @staticmethod
    def from_dict(face_entity: dict, loop_uid: str):
        # Faces consists of One loop
        loopdata = []

        loop_entity = face_entity["profiles"][loop_uid]
        for i, lp in enumerate(loop_entity["loops"]):
            loopdata.append(LoopSequence.from_dict(lp))

        return FaceSequence(loopdata, True)

    def to_vec(self):
        """
        Vector Representation of Face
        """
        coord_token = []
        for lp in self.loopdata:
            vec = lp.to_vec()
            coord_token += vec
        coord_token.append([self.token_index, 0])
        return coord_token

    def reorder(self):
        if len(self.loopdata) <= 1:
            return
        all_loops_bbox_min = np.stack(
            [loop.bbox[0] for loop in self.loopdata], axis=0
        ).round(6)
        ind = np.lexsort(all_loops_bbox_min.transpose()[[1, 0]])
        self.loopdata = [self.loopdata[i] for i in ind]

    @staticmethod
    def from_vec(vec, bit, post_processing, fix_collinearity):
        """
        Vec is the list of loops
        """
        lp = []
        merged_vec = split_array(vec, val=SKETCH_TOKEN.index("END_LOOP"))
        for lp_tokens in merged_vec:
            lp.append(
                LoopSequence.from_vec(
                    vec=lp_tokens,
                    bit=bit,
                    post_processing=post_processing,
                    fix_collinearity=fix_collinearity,
                )
            )
        if len(lp) == 0:
            raise Exception(f"No Loops Added for vec {vec}")
        return FaceSequence(
            loopdata=lp, reorder=False
        )  # No reordering during evaluation

    def __repr__(self):
        s = "Face:"  # Start with bold text for "Loop:"
        for loop in self.loopdata:
            s += f"\n          - {loop.__repr__()}"  # Add the curve representation with blue color

        return s + "\n"


    def transform(self, translate=None, scale=1):
        if translate is None:
            translate = 0
        for loop in self.loopdata:
            loop.transform(translate=translate, scale=scale)

    # @logger.catch()
    def sample_points(self, n_points):
        all_points = []

        for loop in self.loopdata:
            all_points.append(
                loop.sample_points(n_points=n_points)
            )

        all_points = np.vstack(all_points)
        random_points = random_sample_points(all_points, n_points)[0]
        # random_points=all_points
        return random_points

    @property
    def all_curves(self):
        all_curves = []
        for lp in self.loopdata:
            all_curves += lp.all_curves

        return all_curves

    @property
    def start_point(self):
        return self.loopdata[0].start_point

    @property
    def all_loops(self):
        all_loops = []
        for lp in self.loopdata:
            all_loops.append(lp)
        return all_loops

    @property
    def bbox(self):
        all_min_box = []
        all_max_box = []
        for lp in self.loopdata:
            bbox = lp.bbox
            all_min_box.append(bbox[0])
            all_max_box.append(bbox[1])
        return np.array([np.min(all_min_box, axis=0), np.max(all_max_box, axis=0)])

    def build_body(self, plane, normal, coordsystem):
        """
        plane: gp_Pln object. Sketch Plane where a face will be constructed
        normal: gp_Dir object
        transform: gp_Trsf object
        """
        face_list = []
        # plane=self.plane
        # Save all the loop
        for lp in self.loopdata:
            face_builder = BRepBuilderAPI_MakeFace(
                plane,
                lp.build_body(
                    normal=normal, coordsystem=coordsystem
                ),
            )
            if not face_builder.IsDone():
                raise Exception("face builder not done")
            face = face_builder.Face()

            # Fix face
            fixer = ShapeFix_Face(face)
            fixer.SetPrecision(PRECISION)
            fixer.FixOrientation()

            # analyzer = BRepCheck_Analyzer(fixer.Face())
            # if not analyzer.IsValid():
            #     clglogger.error(f"{lp}{normal}{coordsystem}")
            #     raise Exception(f"face check failed.")

            face_list.append(fixer.Face())

        # Find the outer wire (rest becomes inner wires)
        props = GProp_GProps()
        outer_idx = 0
        redo = True
        while redo:
            for f_idx, face in enumerate(face_list):
                # Skip outer face itself
                if f_idx == outer_idx:
                    continue
                # Cut inner face from outer
                cut_face = perform_op(face_list[outer_idx], face, "cut")
                # Compute area, check if inner is larger than outer
                brepgprop.SurfaceProperties(cut_face, props)
                area = props.Mass()
                if area == 0.0:
                    outer_idx = f_idx
                    break
            redo = False

        # Create final closed loop face
        inner_idx = list(set(list(range(0, len(face_list)))) - set([outer_idx]))
        inner_faces = [face_list[i] for i in inner_idx]
        final_face = face_list[outer_idx]
        for face in inner_faces:
            final_face = perform_op(final_face, face, "cut")

        return face_list[0], final_face

    def numericalize(self, bit=N_BIT):
        for lp in self.loopdata:
            lp.numericalize(bit=bit)

    def denumericalize(self, bit):
        for lp in self.loopdata:
            lp.denumericalize(bit=bit)

    def draw(self, ax=None, colors=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        if colors is None:
            colors = [
                "red",
                "blue",
                "green",
                "brown",
                "pink",
                "yellow",
                "purple",
                "black",
            ] * 10
        else:
            colors = [colors] * 100
        for i, loop in enumerate(self.loopdata):
            loop.draw(ax, colors[i])


    def _json(self):
        face_json = {}
        for i, loop in enumerate(self.loopdata):
            face_json[f"loop_{i+1}"] = loop._json()

        return face_json



if __name__ == "__main__":
    face_dict = {
        "transform": {
            "origin": {"y": 0.0, "x": 0.0, "z": 0.0},
            "y_axis": {"y": 0.0, "x": 0.0, "z": 1.0},
            "x_axis": {"y": 1.0, "x": 0.0, "z": 0.0},
            "z_axis": {"y": 0.0, "x": 1.0, "z": 0.0},
        },
        "type": "Sketch",
        "name": "Sketch 1",
        "profiles": {
            "JGC": {
                "loops": [
                    {
                        "is_outer": True,
                        "profile_curves": [
                            {
                                "center_point": {"y": 0.0762, "x": 0.0, "z": 0.0},
                                "type": "Circle3D",
                                "radius": 0.06000001,
                                "curve": "JGR",
                                "normal": {"y": 0.0, "x": 1.0, "z": 0.0},
                            }
                        ],
                    }
                ],
                "properties": {},
            },
            "JGK": {
                "loops": [
                    {
                        "is_outer": True,
                        "profile_curves": [
                            {
                                "type": "Line3D",
                                "start_point": {"y": 0.3048, "x": 0.3048, "z": 0.0},
                                "curve": "JGB",
                                "end_point": {"y": 0.3048, "x": -0.3048, "z": 0.0},
                            },
                            {
                                "type": "Line3D",
                                "start_point": {"y": 0.3048, "x": -0.3048, "z": 0.0},
                                "curve": "JGN",
                                "end_point": {"y": -0.3048, "x": -0.3048, "z": 0.0},
                            },
                            {
                                "type": "Line3D",
                                "start_point": {"y": -0.3048, "x": 0.3048, "z": 0.0},
                                "curve": "JGF",
                                "end_point": {"y": -0.3048, "x": -0.3048, "z": 0.0},
                            },
                            {
                                "type": "Line3D",
                                "start_point": {"y": 0.3048, "x": 0.3048, "z": 0.0},
                                "curve": "JGJ",
                                "end_point": {"y": -0.3048, "x": 0.3048, "z": 0.0},
                            },
                        ],
                    },
                    {
                        "is_outer": True,
                        "profile_curves": [
                            {
                                "center_point": {"y": 0.0762, "x": 0.0, "z": 0.0},
                                "type": "Circle3D",
                                "radius": 0.06000001,
                                "curve": "JGR",
                                "normal": {"y": 0.0, "x": 1.0, "z": 0.0},
                            }
                        ],
                    },
                    {
                        "is_outer": True,
                        "profile_curves": [
                            {
                                "center_point": {"y": -0.08540001, "x": 0.0, "z": 0.0},
                                "type": "Circle3D",
                                "radius": 0.06000001,
                                "curve": "JGV",
                                "normal": {"y": 0.0, "x": 1.0, "z": 0.0},
                            }
                        ],
                    },
                ],
                "properties": {},
            },
            "JGG": {
                "loops": [
                    {
                        "is_outer": True,
                        "profile_curves": [
                            {
                                "center_point": {"y": -0.08540001, "x": 0.0, "z": 0.0},
                                "type": "Circle3D",
                                "radius": 0.06000001,
                                "curve": "JGV",
                                "normal": {"y": 0.0, "x": 1.0, "z": 0.0},
                            }
                        ],
                    }
                ],
                "properties": {},
            },
        },
        "reference_plane": {},
    }

    face = FaceSequence.from_dict(face_dict, "JGK")
    print(face._json())
    # print(face.all_curves)

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)

    # # Save PointCloud object as PLY file
    # o3d.io.write_point_cloud("/home/mkhan/Codes/point2cad/output/output.ply", pcd)


================================================
File: CadSeqProc/sequence/sketch/loop.py
================================================
import os, sys
from typing import List

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))

import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.geometry.curve import Curve
from CadSeqProc.geometry.line import Line
from CadSeqProc.geometry.arc import Arc
from CadSeqProc.geometry.circle import Circle
from CadSeqProc.utility.utils import (
    get_orientation,
    merge_list,
    flatten,
    random_sample_points,
    merge_end_tokens_from_loop,
    write_stl_file,
    write_ply,
    point_distance,
    create_matched_pair,
)
from rich import print
from loguru import logger
import matplotlib.pyplot as plt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
from OCC.Core.ShapeFix import ShapeFix_Wire
from OCC.Extend.DataExchange import write_step_file
from scipy.optimize import linear_sum_assignment

clglogger = CLGLogger().configure_logger().logger


class LoopSequence(object):

    def __init__(
        self,
        curvedata: List[Curve],
        is_outer=False,
        post_processing=True,
        fix_collinearity=False,
    ) -> None:
        self.curvedata = curvedata
        self.is_outer = is_outer
        self.collinear_curves = []

        # if post_processing:
        # Reorder the loop to fix connectivity, orientation and collinearity
        self.reorder(orientation_fix=True, collinearity_fix=fix_collinearity)

        # If post processing is set on, fix the line start and end point error
        n_curve = len(self.curvedata)

        if post_processing:
            for i, cv in enumerate(self.curvedata):
                if n_curve == 1:
                    continue
                else:
                    if cv.curve_type == "line":
                        if (
                            np.sum(
                                cv.metadata["start_point"] - cv.metadata["end_point"]
                            )
                            == 0
                        ):
                            cv.metadata["end_point"] += 1
                            self.curvedata[(i + 1) % n_curve].metadata[
                                "start_point"
                            ] = cv.metadata["end_point"]

    @property
    def token_index(self):
        return SKETCH_TOKEN.index("END_LOOP")

    @staticmethod
    def from_dict(loop_entity: dict):
        is_outer = loop_entity["is_outer"]
        curvedata = []
        curves = loop_entity["profile_curves"]
        for i in range(len(curves)):
            curve_type = curves[i]["type"]

            if curve_type == "Line3D":
                curvedata.append(Line.from_dict(curves[i]))
            if curve_type == "Arc3D":
                curvedata.append(Arc.from_dict(curves[i]))
            if curve_type == "Circle3D":
                curvedata.append(Circle.from_dict(curves[i]))

        return LoopSequence(curvedata, is_outer, False)

    @property
    def start_point(self):
        # if len(self.curvedata)>1:
        #     return self.curvedata[0].start_point
        # else:
        return self.curvedata[0].start_point

    @property
    def bbox(self):
        if len(self.curvedata) <= 1:
            return self.curvedata[0].bbox
        else:
            all_min_box = []
            all_max_box = []
            for curve in self.curvedata:
                if curve is not None:
                    bbox = curve.bbox
                    all_min_box.append(bbox[0])
                    all_max_box.append(bbox[1])
        return np.array([np.min(all_min_box, axis=0), np.max(all_max_box, axis=0)])

    def to_vec(self):
        """
        Vector Representation of Loop
        """
        coord_token = []
        for cv in self.curvedata:
            vec = cv.to_vec()
            coord_token += vec
        coord_token.append([self.token_index, 0])
        return coord_token

    @staticmethod
    def from_vec(vec, bit, post_processing, fix_collinearity):
        """
        Vec is the list of curves
        """
        cv = []
        merged_vec = merge_end_tokens_from_loop(vec)[0]
        for cv_tokens in merged_vec:
            if len(merged_vec) == 1:
                cv.append(
                    Circle.from_vec(
                        vec=cv_tokens,
                        bit=bit,
                        post_processing=post_processing,
                    )
                )
            elif len(cv_tokens) == 2:
                cv.append(
                    Line.from_vec(
                        vec=cv_tokens,
                        bit=bit,
                        post_processing=post_processing,
                    )
                )
            elif len(cv_tokens) == 3:
                cv.append(
                    Arc.from_vec(
                        vec=cv_tokens,
                        bit=bit,
                        post_processing=post_processing,
                    )
                )
            else:
                raise ValueError(f"Invalid Curve Tokens {cv_tokens}")
        if len(cv) == 0:
            raise Exception(f"No Curves Added for vec {vec}")
        return LoopSequence(
            curvedata=cv,
            post_processing=post_processing,
            fix_collinearity=fix_collinearity,
        )

    @property
    def direction(self):
        first_curve = self.curvedata[0]
        if first_curve.curve_type == "circle":
            try:
                return get_orientation(
                    first_curve.get_point("pt1"),
                    first_curve.get_point("pt2"),
                    first_curve.get_point("pt3"),
                )
            except:
                return "counterclockwise"
        else:
            return get_orientation(
                first_curve.get_point("start_point"),
                first_curve.get_point("end_point"),
                self.curvedata[1].get_point("end_point"),
            )

    @staticmethod
    def is_connected(curvedata: List[Curve]):
        """
        Check if curve is connected
        """
        n = len(curvedata)
        if n == 1:
            return True

        for i, curve in enumerate(curvedata):
            if (
                i > 0
                and i < n - 1
                and not np.allclose(
                    curvedata[i - 1].get_point("end_point"),
                    curve.get_point("start_point"),
                )
            ):
                clglogger.critical(
                    f"Curve is not connected {curvedata} at {curvedata[i-1],curve}"
                )
                return False
            elif i == n - 1 and not np.allclose(
                curvedata[0].get_point("start_point"), curve.get_point("end_point")
            ):
                clglogger.critical(
                    f"Curve is not connected {curvedata} at {curve,curvedata[0]}"
                )
                return False
        clglogger.success("Curve is connected")
        return True

    @staticmethod
    def ensure_connectivity(curvedata: List[Curve], verbose=False):
        """
        Create a connected loop from the existing curves
        """
        if len(curvedata) <= 1:
            return curvedata

        new_curvedata = [curvedata[0]]

        n = len(curvedata)
        for i, curve in enumerate(curvedata):
            if i > 0:
                if i < n - 1 and np.allclose(
                    new_curvedata[-1].get_point("end_point"),
                    curve.get_point("end_point"),
                ):
                    curve.reverse()
                    new_curvedata.append(curve)
                elif (
                    i == n - 1
                    and np.allclose(
                        new_curvedata[-1].get_point("end_point"),
                        curve.get_point("end_point"),
                    )
                    or np.allclose(
                        curve.get_point("start_point"),
                        new_curvedata[0].get_point("start_point"),
                    )
                ):
                    curve.reverse()
                    new_curvedata.append(curve)
                else:
                    new_curvedata.append(curve)
        if verbose:
            LoopSequence.is_connected(curvedata)

        return new_curvedata

    def reorder(self, orientation_fix=True, collinearity_fix=True):
        """reorder by starting left most and counter-clockwise. Fix Collinearity if exists by merging (for lines only)"""
        if len(self.curvedata) <= 1:
            return

        start_curve_idx = -1
        sx, sy = 10000, 10000
        total_curve = len(self.curvedata)

        self.curvedata = LoopSequence.ensure_connectivity(
            self.curvedata, verbose=False
        )  # Connected Loop
        # LoopSequence.is_connected(self.curvedata) # Check if the loop is connected

        # correct start-end point order and find left-most point
        for i, curve in enumerate(self.curvedata):
            if round(curve.get_point("start_point")[0], 6) < round(sx, 6) or (
                round(curve.get_point("start_point")[0], 6) == round(sx, 6)
                and round(curve.get_point("start_point")[1], 6) < round(sy, 6)
            ):
                start_curve_idx = i
                sx, sy = curve.get_point("start_point")

        self.curvedata = (
            self.curvedata[start_curve_idx:] + self.curvedata[:start_curve_idx]
        )

        # Fix Orientation so that loop is created counter-clockwise
        if (
            self.direction == "clockwise"
            and orientation_fix
            and len(self.curvedata) > 1
        ):
            self.curvedata = self.curvedata[::-1]

            for i in range(len(self.curvedata)):
                self.curvedata[i].reverse()

        # Fix Collinearity
        if len(self.curvedata) > 1 and collinearity_fix:
            collinear_pair = []
            for i in range(len(self.curvedata) - 1):
                if self.curvedata[i].is_collinear(self.curvedata[i + 1]):
                    collinear_pair.append([i, i + 1])
                    self.collinear_curves.append(
                        [self.curvedata[i], self.curvedata[i + 1]]
                    )
                else:
                    collinear_pair.append([i])
            if len(self.curvedata) - 1 not in flatten(collinear_pair):
                collinear_pair.append([len(self.curvedata) - 1])
            collinear_pair = merge_list(collinear_pair)
            self.new_curvedata = []

            for p in collinear_pair:
                if len(p) == 1:
                    self.new_curvedata.append(self.curvedata[p[0]])
                else:
                    curve = self.curvedata[p[0]]
                    curve.merge(self.curvedata[p[-1]])
                    self.new_curvedata.append(curve)

            self.curvedata = self.new_curvedata

    @property
    def all_curves(self):
        return self.curvedata

    def transform3D(self, coordsystem):
        for curve in self.curvedata:
            curve.transform3D(coordsystem=coordsystem)

    def transform(self, translate=None, scale=1):
        if translate is None:
            translate = 0
        for curve in self.curvedata:
            curve.transform(translate=translate, scale=scale)

    def __repr__(self):
        s = f"Loop: Start Point: {list(np.round(self.start_point,4))}, Direction: {self.direction}"  # bbox {list(np.round(self.bbox,4))}"  # Start with bold text for "Loop:"
        for curve in self.curvedata:
            s += f"\n              - {curve.__repr__()}"  # Add the curve representation with blue color
        return s + "\n"

    def add_info(self, key_, val_):
        self.curvedata[key_] = val_

    def sample_points(self, n_points):
        all_points = []

        for curve in self.curvedata:
            all_points.append(
                curve.sample_points(n_points=n_points)
            )
        all_points = np.vstack(all_points)
        random_points = random_sample_points(all_points, n_points)[0]
        # random_points=all_points
        return random_points

    def draw(self, ax=None, colors=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        if colors is None:
            colors = [
                "red",
                "blue",
                "green",
                "brown",
                "pink",
                "yellow",
                "purple",
                "black",
            ] * 10
        else:
            colors = [colors] * 100
        for i, curve in enumerate(self.curvedata):
            curve.draw(ax, colors[i])

    def build_body(self, normal, coordsystem):
        topo_wire = BRepBuilderAPI_MakeWire()
        for cv in self.curvedata:
            if cv.curve_type.lower() == "circle":
                topo_wire.Add(
                    cv.build_body(
                        normal=normal, coordsystem=coordsystem
                    )
                )
            else:
                topo_wire.Add(
                    cv.build_body(coordsystem=coordsystem)
                )
            if not topo_wire.IsDone():
                raise Exception("wire builder not done")

        fixer = ShapeFix_Wire()
        fixer.Load(topo_wire.Wire())
        fixer.SetPrecision(PRECISION)
        fixer.FixClosed()
        fixer.Perform()

        return fixer.Wire()

    def numericalize(self, bit=N_BIT):
        for cv in self.curvedata:
            cv.numericalize(bit=bit)

        # Fix Invalidity
        # for i,cv in enumerate(self.curvedata[:-1]):
        #     if cv.curve_type.lower()=="line":
        #         if np.sum(cv.metadata['start_point']-cv.metadata['end_point'])==0:
        #             cv.metadata['end_point']+=1
        #             self.curvedata[i+1].metadata['start_point']=cv.metadata['end_point']

    def denumericalize(self, bit):
        for cv in self.curvedata:
            cv.denumericalize(bit=bit)

    def loop_distance(self, target_loop, scale: float):
        return point_distance(self.bbox * scale, target_loop.bbox * scale, type="l2")

    @staticmethod
    def match_primitives(gt_loop, pred_loop, scale: float, multiplier: int = 1):
        """
        Match primitives (e.g., curves) based on their bounding box distances.

        Args:
            gt_loop (object): Ground truth loop object.
            pred_loop (object): Predicted loop object.
            scale (float): The scaling factor.
            multiplier (int, optional): Multiplier for cost matrix. Defaults to 1.

        Returns:
            list: List containing matched ground truth and predicted curves.
        """
        if gt_loop is None:
            gt_curves = [None]
        else:
            gt_curves = gt_loop.all_curves

        if pred_loop is None:
            pred_curves = [None]
        else:
            pred_curves = pred_loop.all_curves

        n_gt = len(gt_curves)
        n_pred = len(pred_curves)
        n_max = max(n_gt, n_pred)

        # Initialize cost matrix with ones and apply multiplier
        cost_matrix = np.ones((n_max, n_max)) * multiplier

        # Pad lists with None if needed
        if n_gt < n_max:
            gt_curves += [None] * (n_max - n_gt)

        if n_pred < n_max:
            pred_curves += [None] * (n_max - n_pred)

        # Calculate Cost by calculating the distance between loops
        for ind_self in range(n_gt):
            for ind_pred in range(n_pred):
                if (
                    gt_curves[ind_self] is not None
                    and pred_curves[ind_pred] is not None
                ):
                    cost_matrix[ind_self, ind_pred] = gt_curves[
                        ind_self
                    ].curve_distance(pred_curves[ind_pred], scale)

        # Use Hungarian matching to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # row_indices=list(row_indices)
        # col_indices=list(col_indices)
        # print(row_indices, col_indices)

        # Create a new pair with matched ground truth and predicted curves
        new_pair = create_matched_pair(
            list1=gt_curves,
            list2=pred_curves,
            row_indices=row_indices,
            col_indices=col_indices,
        )
        return new_pair

    def _json(self):
        loop_json = {}
        curve_num_dict = {"line": 1, "arc": 1, "circle": 1}
        for i, curve in enumerate(self.curvedata):
            loop_json[f"{curve.curve_type}_{curve_num_dict[curve.curve_type]}"] = (
                curve._json()
            )
            curve_num_dict[curve.curve_type] += 1
        return loop_json


if __name__ == "__main__":
    loop_dict = {
        "is_outer": True,
        "profile_curves": [
            {
                "type": "Line3D",
                "start_point": {"y": 0.3048, "x": 0.3048, "z": 0.0},
                "curve": "JGB",
                "end_point": {"x": 0.166, "y": 0.3048, "z": 0.0},
            },
            {
                "type": "Line3D",
                "start_point": {"x": 0.166, "y": 0.3048, "z": 0.0},
                "curve": "JGB",
                "end_point": {"x": -0.3048, "y": 0.3048, "z": 0.0},
            },
            {
                "type": "Line3D",
                "start_point": {"y": 0.3048, "x": -0.3048, "z": 0.0},
                "curve": "JGN",
                "end_point": {"y": -0.3048, "x": -0.3048, "z": 0.0},
            },
            {
                "type": "Line3D",
                "start_point": {"y": -0.3048, "x": 0.3048, "z": 0.0},
                "curve": "JGF",
                "end_point": {"y": -0.3048, "x": -0.3048, "z": 0.0},
            },
            {
                "type": "Line3D",
                "start_point": {"y": 0.3048, "x": 0.3048, "z": 0.0},
                "curve": "JGJ",
                "end_point": {"y": -0.3048, "x": 0.3048, "z": 0.0},
            },
        ],
    }

    loop = LoopSequence.from_dict(loop_dict)
    loop.reorder()
    print(loop._json())


================================================
File: CadSeqProc/sequence/sketch/sketchsequence.py
================================================
import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))


import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.utils import (
    create_point_from_array,
    perform_op,
    random_sample_points,
    split_array,
    write_ply,
    create_matched_pair,
    create_colored_wire,
)
from CadSeqProc.utility.macro import *
from rich import print
from .face import FaceSequence, LoopSequence
from loguru import logger
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.gp import gp_Vec, gp_Pln, gp_Dir, gp_Ax3
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Extend.DataExchange import write_step_file
from .coord_system import CoordinateSystem
import copy
from scipy.optimize import linear_sum_assignment
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

clglogger = CLGLogger().configure_logger().logger


class SketchSequence(object):

    def __init__(
        self,
        facedata: List[FaceSequence],
        coordsystem: CoordinateSystem = None,
        reorder: bool = True,
    ):

        self.facedata = facedata
        self.quantized_facedata = {}
        self.coordsystem = coordsystem

        if reorder:
            # Reorder Faces
            self.reorder()

    @property
    def token_index(self):
        return SKETCH_TOKEN.index("END_SKETCH")

    def reorder(self):
        if len(self.facedata) <= 1:
            return
        all_faces_bbox_min = np.stack(
            [face.bbox[0] for face in self.facedata], axis=0
        ).round(6)
        ind = np.lexsort(all_faces_bbox_min.transpose()[[1, 0]])
        self.facedata = [self.facedata[i] for i in ind]

    @staticmethod
    def from_dict(all_stat, profile_uid_list):
        facedata = []
        coordsystem = CoordinateSystem.from_dict(
            all_stat["entities"][profile_uid_list[0][0]]["transform"]
        )

        for i in range(len(profile_uid_list)):
            sketch_entity = all_stat["entities"][profile_uid_list[i][0]]
            assert sketch_entity["type"] == "Sketch", clglogger.critical(
                f"Uid Mismatch for {profile_uid_list[i]}"
            )
            facedata.append(
                FaceSequence.from_dict(sketch_entity, profile_uid_list[i][1])
            )

        return SketchSequence(facedata=facedata, coordsystem=coordsystem, reorder=True)

    def sample_points(self, n_points, point_dimension=3):
        all_points = []

        for fc in self.facedata:
            all_points.append(
                fc.sample_points(n_points=n_points)
            )
        all_points = np.vstack(all_points)

        random_points = random_sample_points(all_points, n_points)[0]
        if random_points.shape[-1] == 2 and point_dimension == 3:
            random_points = self.coordsystem.rotate_vec(random_points)
        return random_points

    def __repr__(self):
        s = "Sketch:"
        s += f"\n       - {self.coordsystem.__repr__()}"
        for face in self.facedata:
            s += f"\n       - {face.__repr__()}"
        return s

    def to_vec(self):
        """
        Vector Representation of One Sketch sequence
        """
        coord_token = []
        for fc in self.facedata:
            vec = fc.to_vec()
            coord_token += vec
        coord_token.append([self.token_index, 0])
        return coord_token

    @staticmethod
    def from_vec(vec, bit, post_processing, fix_collinearity):
        """
        Vec is the list of faces
        """
        fc = []
        merged_vec = split_array(vec, val=SKETCH_TOKEN.index("END_FACE"))
        for fc_tokens in merged_vec:
            fc.append(
                FaceSequence.from_vec(
                    vec=fc_tokens,
                    bit=bit,
                    post_processing=post_processing,
                    fix_collinearity=fix_collinearity,
                )
            )
        if len(fc) == 0:
            raise Exception(f"No Loops Added for vec {vec}")
        return SketchSequence(facedata=fc, reorder=False)

    @property
    def bbox(self):
        all_min_box = []
        all_max_box = []
        for fc in self.facedata:
            bbox = fc.bbox
            all_min_box.append(bbox[0])
            all_max_box.append(bbox[1])
        return np.array([np.min(all_min_box, axis=0), np.max(all_max_box, axis=0)])

    @property
    def length(self):
        bbox_min=self.bbox[0]
        bbox_max=self.bbox[1]
        return abs(bbox_max[0]-bbox_min[0])
    
    @property
    def width(self):
        bbox_min=self.bbox[0]
        bbox_max=self.bbox[1]
        return abs(bbox_max[1]-bbox_min[1])

    @property
    def dimension(self):
        return self.length, self.width
    
    @property
    def all_loops(self):
        all_loops = []
        for fc in self.facedata:
            all_loops += fc.all_loops

        return all_loops

    @property
    def bbox_size(self):
        """compute bounding box size (max of height and width)"""
        bbox_min, bbox_max = self.bbox[0], self.bbox[1]
        bbox_size = np.max(
            np.abs(
                np.concatenate(
                    [bbox_max - self.start_point, bbox_min - self.start_point]
                )
            )
        )
        # clglogger.debug(f"{self.bbox} {bbox_size} {bbox_max-self.start_point}")
        return bbox_size

    def add_info(self, key: str, val: FaceSequence):
        self.facedata[key] = val

    def transform(self, translate=None, scale=1):
        for fc in self.facedata:
            fc.transform(translate=translate, scale=scale)

    @property
    def all_curves(self):
        curves = []

        for fc in self.facedata:
            curves += fc.all_curves

        return curves

    @property
    def start_point(self):
        # return self.facedata[0].start_point
        return self.bbox[0]

    @property
    def sketch_position(self):
        return (
            self.start_point[0] * self.coordsystem.get_property("x_axis")
            + self.start_point[1] * self.coordsystem.get_property("y_axis")
            + self.coordsystem.get_property("origin")
        )

    def sketch_plane(self):

        origin = create_point_from_array(self.sketch_position)
        return gp_Pln(origin, gp_Dir(*self.coordsystem.metadata["z_axis"]))

    def build_body(self, extrude_params: dict):
        """
        extrude params must contain {"extrude_values": [float,float]}
        """
        all_faces = []
        for fc in self.facedata:
            ref_face, face = fc.build_body(
                plane=self.sketch_plane(),
                normal=self.coordsystem.normal,
                coordsystem=self.coordsystem,
            )
            all_faces.append(face)
            # clglogger.debug("Success for a face")

        # Merge all faces in the same plane
        plane_face = all_faces[0]
        for face in all_faces[1:]:
            plane_face = perform_op(plane_face, face, "fuse")

        # Extrude face to 3d shape
        solid = self.extrude_face(ref_face, plane_face, extrude_params)

        return solid

    def extrude_face(self, ref_face, face, extrude_params):
        distance = extrude_params["extrude_values"]
        surf = BRepAdaptor_Surface(ref_face).Plane()
        normal = surf.Axis().Direction()
        extruded_shape = self.extrudeBasedOnType(face, normal, distance)
        return extruded_shape

    def extrudeBasedOnType(self, face, normal, distance):
        # Extrude based on the two bound values
        # if not (distance[0] < distance[1]):
        #     sorted(distance)
        # large_value = max(distance)
        # small_value = min(distance)
        if distance[0] == 0:
            ext_vec = gp_Vec(normal.Reversed()).Multiplied(distance[1])
            body = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
        else:
            ext_vec = gp_Vec(normal).Multiplied(distance[0])
            body = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
            if distance[1] > 0:
                ext_vec = gp_Vec(normal.Reversed()).Multiplied(distance[1])
                body_two = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
                body = perform_op(body, body_two, "fuse")
        return body

        # if large_value == 0:
        #     return self.build_prism(face, -normal, -small_value)
        # elif small_value == 0:
        #     return self.build_prism(face, normal, large_value)
        # elif np.sign(large_value) == np.sign(small_value):
        #     if large_value < 0:
        #         body1 = self.build_prism(face, -normal, -small_value)
        #         body2 = self.build_prism(face, -normal, -large_value)
        #         return perform_op(body1, body2, 'cut')
        #     else:
        #         assert large_value > 0
        #         body1 = self.build_prism(face, normal, small_value)
        #         body2 = self.build_prism(face, normal, large_value)
        #         return perform_op(body2, body1, 'cut')
        # else:
        #     assert np.sign(large_value) != np.sign(small_value)
        #     body1 = self.build_prism(face, normal, large_value)
        #     body2 = self.build_prism(face, -normal, -small_value)
        #     return perform_op(body1, body2, 'fuse')

    def build_prism(self, face, normal, value):
        extrusion_vec = gp_Vec(normal).Multiplied(value)
        make_prism = BRepPrimAPI_MakePrism(face, extrusion_vec)
        make_prism.Build()
        prism = make_prism.Prism()
        return prism.Shape()

    def normalize(self, translate=None, bit=N_BIT):
        """
        Normalize the sketch and shift the sketch to the start point.
        Only used for 2d representation
        """
        size = 2**bit
        cur_size = self.bbox_size
        # scale = (size / 2 * NORM_FACTOR - 1) / cur_size # prevent potential overflow if data augmentation applied
        scale = (size - 1) / self.bbox_size
        if translate is None:
            self.transform(-self.start_point, scale)
        else:
            self.transform(translate, scale)

    def denormalize(self, bbox_size=None, translate=0.0, bit=N_BIT):
        """
        Inverse operation of normalize. Only used for 2d representation.
        """
        size = 2**bit
        # if bbox_size is None:
        #     bbox_size=self.bbox_size
        # scale = bbox_size / (size / 2 * NORM_FACTOR - 1)
        scale = bbox_size / (size - 1)
        if translate is None:
            translate = -np.array((size / 2, size / 2))
        self.transform(translate, scale)

    def numericalize(self, bit):
        """
        Quantization
        """
        for fc in self.facedata:
            fc.numericalize(bit=bit)

    def denumericalize(self, bit):
        """
        Dequantization
        """
        for fc in self.facedata:
            fc.denumericalize(bit=bit)

    def create_skt3d_edge(self):
        """Creates TopoDS shape for 3d sketch visualization"""
        solid = self.build_body(2, {"extrude_values": [0.001, 0]})
        return solid

    @staticmethod
    def loop_match(gt_sketch, pred_sketch, scale: float, multiplier: int = 2):
        """
        Match Loops according to the bounding box.

        Args:
            gt_sketch (object): The current object. (self must be ground truth)
            pred_sketch (object): The pred sketch object. (pred is prediction)
            scale (float): The scaling factor.
            multiplier (int): cost of distance with None

        Returns:
            list: List of matched loop pairs.
        """

        if pred_sketch is None:
            pred_loops = [None]
        else:
            pred_loops = copy.deepcopy(pred_sketch.all_loops)

        if gt_sketch is None:
            gt_loops = [None]
        else:
            gt_loops = copy.deepcopy(gt_sketch.all_loops)

        num_gt_loops = len(gt_loops)
        num_pred_loops = len(pred_loops)

        n_max = max(num_gt_loops, num_pred_loops)

        # Pad lists with None if needed
        if len(gt_loops) < n_max:
            gt_loops += [None] * (n_max - len(gt_loops))
        if len(pred_loops) < n_max:
            pred_loops += [None] * (n_max - len(pred_loops))

        cost_matrix = (
            np.ones((n_max, n_max)) * multiplier
        )  # Fixed the shape of the cost matrix

        # Calculate Cost by calculating the distance between loops
        for ind_self in range(num_gt_loops):
            for ind_pred in range(num_pred_loops):
                if gt_loops[ind_self] is not None and pred_loops[ind_pred] is not None:
                    cost_matrix[ind_self, ind_pred] = gt_loops[ind_self].loop_distance(
                        pred_loops[ind_pred], scale
                    )

        # Use Hungarian matching to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Create matched loop pairs
        matched_loop_pair = create_matched_pair(
            list1=gt_loops,
            list2=pred_loops,
            row_indices=row_indices,
            col_indices=col_indices,
        )

        # After loops are matched, match primitives
        # (This will change the object from a pair of LoopSequences to a pair of list of CurveSequences)
        matched_curve_pair = []
        for i, pair in enumerate(matched_loop_pair):
            matched_curve_pair += LoopSequence.match_primitives(
                pair[0], pair[1], scale, multiplier
            )

        return matched_curve_pair, matched_loop_pair

    def draw(self, ax=None, colors=None):
        if ax is None:
            fig, ax = plt.subplots()
        if colors is None:
            colors = [
                "red",
                "blue",
                "green",
                "brown",
                "pink",
                "yellow",
                "purple",
                "black",
            ] * 10
        else:
            colors = [colors] * 100
        for i, face in enumerate(self.facedata):
            face.draw(ax, colors[i])

    def _json(self):
        sketch_json = {}
        for i, face in enumerate(self.facedata):
            sketch_json[f"face_{i+1}"] = face._json()

        # sketch_json["coordinate_system"]=self.coordsystem._json()
        return sketch_json





if __name__ == "__main__":
    import json

    json_path = "/data/3d_cluster/Brep2Seq/deepcad_data/cad_json/0043/00430950.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    lst = [["FcWd1Kjyasi3dQe_0", "JGC"], ["FcWd1Kjyasi3dQe_0", "JGG"]]
    skt = SketchSequence.from_dict(data, lst)

    # print(skt.start_point)
    # print(skt)
    skt.transform(translate=-skt.start_point)
    # print(skt)
    skt.transform3D()
    points = skt.sample_points(num_points=1000)
    # print("Points")
    # print(points)

    # print(points.shape)

    # points=np.hstack([points,np.zeros((len(points),1))])

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save PointCloud object as PLY file
    o3d.io.write_point_cloud("/home/mkhan/Codes/point2cad/output/output.ply", pcd)


================================================
File: CadSeqProc/sequence/transformation/deform.py
================================================
"""
Deformation operations for geometric transformations.
"""

import numpy as np
from typing import List, Tuple, Optional
from ...geometry.nurbs import NURBSCurve, NURBSSurface

class DeformationOp:
    """Base class for deformation operations."""
    
    def validate_params(self) -> bool:
        """Validate deformation parameters."""
        raise NotImplementedError
    
    def apply(self, points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Apply deformation to points."""
        raise NotImplementedError

class TwistDeformation(DeformationOp):
    """Twist points around an axis."""
    
    def __init__(self, 
                 angle: float,
                 axis: str = 'z',
                 center: Optional[Tuple[float, float, float]] = None):
        self.angle = angle
        self.axis = axis.lower()
        self.center = center or (0.0, 0.0, 0.0)
    
    def validate_params(self) -> bool:
        return self.axis in ['x', 'y', 'z']
    
    def apply(self, points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        if not self.validate_params():
            raise ValueError(f"Invalid axis: {self.axis}")
        
        result = []
        for point in points:
            # Convert to local coordinates
            local = np.array(point) - np.array(self.center)
            
            # Calculate twist angle based on position
            if self.axis == 'z':
                height = local[2]
                twist = self.angle * height
                c, s = np.cos(twist), np.sin(twist)
                x = local[0] * c - local[1] * s
                y = local[0] * s + local[1] * c
                z = local[2]
            elif self.axis == 'y':
                height = local[1]
                twist = self.angle * height
                c, s = np.cos(twist), np.sin(twist)
                x = local[0] * c - local[2] * s
                y = local[1]
                z = local[0] * s + local[2] * c
            else:  # x axis
                height = local[0]
                twist = self.angle * height
                c, s = np.cos(twist), np.sin(twist)
                x = local[0]
                y = local[1] * c - local[2] * s
                z = local[1] * s + local[2] * c
            
            # Convert back to global coordinates
            result.append(tuple(np.array([x, y, z]) + np.array(self.center)))
        
        return result

class BendDeformation(DeformationOp):
    """Bend points along a curve."""
    
    def __init__(self,
                 curve: NURBSCurve,
                 bend_factor: float = 1.0,
                 up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)):
        self.curve = curve
        self.bend_factor = bend_factor
        self.up_vector = np.array(up_vector)
        self.up_vector /= np.linalg.norm(self.up_vector)
    
    def validate_params(self) -> bool:
        return self.bend_factor > 0
    
    def apply(self, points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        if not self.validate_params():
            raise ValueError("Invalid bend factor")
        
        result = []
        for point in points:
            # Project point onto curve
            min_dist = float('inf')
            min_param = 0.0
            for t in np.linspace(0, 1, 100):
                curve_point = self.curve.evaluate(t)
                dist = np.linalg.norm(np.array(point) - np.array(curve_point))
                if dist < min_dist:
                    min_dist = dist
                    min_param = t
            
            # Calculate bend
            curve_point = np.array(self.curve.evaluate(min_param))
            bend_amount = min_dist * self.bend_factor
            
            # Apply bend
            bent_point = curve_point + self.up_vector * bend_amount
            result.append(tuple(bent_point))
        
        return result

class TaperDeformation(DeformationOp):
    """Taper points along an axis."""
    
    def __init__(self,
                 start_scale: float = 1.0,
                 end_scale: float = 0.5,
                 axis: str = 'z',
                 center: Optional[Tuple[float, float, float]] = None):
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.axis = axis.lower()
        self.center = center or (0.0, 0.0, 0.0)
    
    def validate_params(self) -> bool:
        return (self.axis in ['x', 'y', 'z'] and 
                self.start_scale > 0 and 
                self.end_scale > 0)
    
    def apply(self, points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        if not self.validate_params():
            raise ValueError("Invalid parameters")
        
        result = []
        for point in points:
            # Convert to local coordinates
            local = np.array(point) - np.array(self.center)
            
            # Calculate scale factor based on position
            if self.axis == 'z':
                t = (local[2] - self.center[2]) / (max(p[2] for p in points) - self.center[2])
            elif self.axis == 'y':
                t = (local[1] - self.center[1]) / (max(p[1] for p in points) - self.center[1])
            else:  # x axis
                t = (local[0] - self.center[0]) / (max(p[0] for p in points) - self.center[0])
            
            scale = self.start_scale + t * (self.end_scale - self.start_scale)
            
            # Apply scale
            if self.axis == 'z':
                x = local[0] * scale
                y = local[1] * scale
                z = local[2]
            elif self.axis == 'y':
                x = local[0] * scale
                y = local[1]
                z = local[2] * scale
            else:  # x axis
                x = local[0]
                y = local[1] * scale
                z = local[2] * scale
            
            # Convert back to global coordinates
            result.append(tuple(np.array([x, y, z]) + np.array(self.center)))
        
        return result 

================================================
File: CadSeqProc/sequence/transformation/extrude_sequence.py
================================================
import os, sys
from typing import Any

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))


import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import dequantize_verts, int_round, quantize, float_round
from loguru import logger
from CadSeqProc.sequence.sketch.coord_system import CoordinateSystem

clglogger = CLGLogger().configure_logger().logger


class ExtrudeSequence(object):
    """
    Extrusion Sequence for a sketch.

    This class represents an extrusion sequence for a sketch. It stores metadata related to the extrusion operation.
    """

    def __init__(self, metadata: dict, coordsystem: CoordinateSystem = None):
        """
        Initialize the ExtrudeSequence object with the provided metadata.

        Args:
            metadata (dict): Metadata dictionary containing information about the extrusion.
        """
        self.metadata = metadata
        self.quantized_metadata = {}
        self.is_numerical = False
        self.coordsystem = coordsystem

    @staticmethod
    def from_dict(all_stat, uid):
        """
        Create an ExtrudeSequence object from a dictionary.

        Args:
            all_stat (dict): Dictionary containing all the extrusion data.
            uid (str): Unique identifier for the extrusion entity.

        Returns:
            ExtrudeSequence: An instance of ExtrudeSequence created from the dictionary.

        Raises:
            AssertionError: If the extrusion entity type is not "ExtrudeFeature" or if the start extent type is not "ProfilePlaneStartDefinition".
        """
        metadata = {}
        extrude_entity = all_stat["entities"][uid]  # Only the extrusion entity

        # Verify the extrusion entity type
        assert extrude_entity["type"] == "ExtrudeFeature", clglogger.critical(
            f"uid {uid} is not extrusion"
        )

        # Verify the start extent type
        assert (
            extrude_entity["start_extent"]["type"] == "ProfilePlaneStartDefinition"
        ), clglogger.critical(f"Error with Extrusion uid {uid}")

        # Save the uids of the profiles
        metadata["profile_uids"] = [
            [profile["sketch"], profile["profile"]]
            for profile in extrude_entity["profiles"]
        ]

        # Extract extent values and boolean operation type
        metadata["extent_one"] = extrude_entity["extent_one"]["distance"][
            "value"
        ]  # Towards the direction of normal

        if extrude_entity["extent_type"] == "SymmetricFeatureExtentType":
            metadata["extent_two"] = metadata["extent_one"] / 2
            metadata["extent_one"] = metadata["extent_one"] / 2
        elif "extent_two" in metadata:
            metadata["extent_two"] = extrude_entity["extent_two"]["distance"][
                "value"
            ]  # Towards the opposite direction of normal
        else:
            metadata["extent_two"] = 0

        if metadata["extent_one"] < 0:
            metadata["extent_two"], metadata["extent_one"] = abs(
                metadata["extent_one"]
            ), abs(metadata["extent_two"])
        metadata["boolean"] = EXTRUDE_OPERATIONS.index(extrude_entity["operation"])

        return ExtrudeSequence(metadata)

    @property
    def token_index(self):
        return END_TOKEN.index("END_EXTRUSION")

    def add_info(self, key, val):
        self.metadata[key] = val

    def transform(self, translate, scale, merge_extent=False):
        # clglogger.debug(f"Extrusion Scale {scale}")
        if not isinstance(translate, int) and not isinstance(translate, float):
            if translate.shape[0] != 3:
                translate = np.concatenate([translate, np.zeros(3 - len(translate))])

        # Extrude Distance Transform
        self.metadata["extent_one"] *= scale
        self.metadata["extent_two"] *= scale

        # Two extents can be changed into one single extent by taking the mean and shifting the sketch position
        if merge_extent:
            self.metadata["extent"] = (
                abs(self.metadata["extent_one"]) + abs(self.metadata["extent_two"])
            ) / 2
            ext_translation = abs(self.metadata["extent"] - self.metadata["extent_one"])
        else:
            ext_translation = 0

        self.coordsystem.transform(translate, scale)  # Plane transformation
        self.metadata["sketch_size"] *= scale  # Sketch Size

    def __repr__(self) -> str:
        metadata_str = ", ".join(
            f"{key}: {value}" for key, value in self.metadata.items()
        )

        repr_str = f'{self.__class__.__name__}: ({metadata_str}) Euler Angles {self.coordsystem.metadata["euler_angles"]}'

        return repr_str

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, __value)

    def get_profile_uids(self):
        return self.metadata["profile_uids"]

    def get_total_extent(self, return_quantized=True):
        if hasattr(self, "quantized_metadata") and return_quantized:
            return (
                self.quantized_metadata["extent_one"]
                + self.quantized_metadata["extent_two"]
            )
        else:
            return abs(self.metadata["extent_one"]) + abs(self.metadata["extent_two"])

    def get_boolean(self):
        return self.metadata["boolean"]

    def numericalize(self, bit):
        self.is_numerical = True
        size = 2**bit - 1
        assert (
            -2.0 <= self.metadata["extent_one"] <= 2.0
            and -2.0 <= self.metadata["extent_two"] <= 2.0
        )
        self.metadata["extent_one"] = int_round(
            [
                ((self.metadata["extent_one"] + 1.0) / 2 * (size + 1)).clip(
                    min=0, max=size
                )
            ]
        )[0]
        self.metadata["extent_two"] = int_round(
            [
                ((self.metadata["extent_two"] + 1.0) / 2 * (size + 1)).clip(
                    min=0, max=size
                )
            ]
        )[0]
        self.metadata["boolean"] = int(self.metadata["boolean"])
        self.coordsystem.numericalize(bit)
        self.metadata["sketch_size"] = int_round(
            [(self.metadata["sketch_size"] / 2 * (size + 1)).clip(min=0, max=size)]
        )[0]

        # Due to quantization, small extent values can be quantized to zero so change the values to 1
        if self.metadata["extent_one"] == (2**bit) / 2 and self.metadata[
            "extent_two"
        ] == (2**bit / 2):
            self.metadata["extent_one"] = 1 + ((2**bit) // 2)
        if self.metadata["sketch_size"] == 0:
            self.metadata["sketch_size"] = 1

    def denumericalize(self, bit, post_processing=True):
        self.is_numerical = False
        size = 2**bit
        self.metadata["extent_one"] = self.metadata["extent_one"] / size * 2 - 1.0
        self.metadata["extent_two"] = self.metadata["extent_two"] / size * 2 - 1.0
        self.coordsystem.denumericalize(bit)
        # self.metadata['sketch_pos'] = self.metadata['sketch_pos'] / size * 2 - 1.0
        self.metadata["sketch_size"] = self.metadata["sketch_size"] / size * 2
        if post_processing:
            if (
                self.metadata["extent_one"] == 0
                and self.metadata["extent_two"] == 0
            ):  # Post Processing Step
                self.metadata["extent_one"] = 0.01

    def to_vec(self):
        """
        default Value

        END_PAD = 3 # ONE END TOKEN, Start/END SEQ Token and Pad Token
        EXT_OPERATION_PAD=4 # Boolean Operations

        So 0,1,2 are preserved for End tokens.
        """
        assert self.is_numerical is True, clglogger.error("Values are not quantized")
        vec = []
        distance1 = [self.metadata["extent_one"] + END_PAD + BOOLEAN_PAD, 0]
        distance2 = [self.metadata["extent_two"] + END_PAD + BOOLEAN_PAD, 0]
        origin = [
            [i, 0]
            for i in self.coordsystem.metadata["origin"] + END_PAD + BOOLEAN_PAD
        ]
        euler_angles = [
            [i, 0]
            for i in self.coordsystem.metadata["euler_angles"]
            + END_PAD
            + BOOLEAN_PAD
        ]
        boolean = [self.metadata["boolean"] + END_PAD, 0]
        sketch_size = [self.metadata["sketch_size"] + END_PAD + BOOLEAN_PAD, 0]
        token = [self.token_index, 0]

        vec = (
            [distance1]
            + [distance2]
            + origin
            + euler_angles
            + [boolean]
            + [sketch_size]
            + [token]
        )
        # (e1,e2,ox,oy,oz,theta,phi,gamma,b,s,END_EXTRUDE_SKETCH) -> 11
        return vec

    @staticmethod
    def from_vec(vec, bit, post_processing):
        if vec[-1][0] == END_TOKEN.index("END_EXTRUSION"):
            vec = vec[:-1]
        metadata = {}
        metadata["extent_one"] = vec[0][0] - (END_PAD + BOOLEAN_PAD)
        metadata["extent_two"] = vec[1][0] - (END_PAD + BOOLEAN_PAD)
        metadata["boolean"] = vec[-2][0] - (END_PAD)
        metadata["sketch_size"] = vec[-1][0] - (END_PAD + BOOLEAN_PAD)
        coordsystem = CoordinateSystem.from_vec(
            vec[2:8, 0] - (END_PAD + BOOLEAN_PAD), bit, post_processing
        )
        # if post_processing and metadata['extent_one']==0 and metadata['extent_two']==0:
        #     metadata['extent_one']=1

        ext = ExtrudeSequence(metadata=metadata, coordsystem=coordsystem)
        ext.quantized_metadata = metadata.copy()

        return ext

    def _json(self):
        extrude_json = {
            "extrude_depth_towards_normal": float(float_round(self.metadata["extent_one"])),
            "extrude_depth_opposite_normal": float(float_round(self.metadata["extent_two"])),
            "sketch_scale": float(float_round(self.metadata["sketch_size"])),
            "operation": EXTRUDE_OPERATIONS[self.metadata["boolean"]],
        }

        return extrude_json


================================================
File: CadSeqProc/tests/test_pattern_recognition.py
================================================
"""Tests for the pattern recognition module."""

import unittest
import numpy as np  # type: ignore
from typing import List, Dict, Any
from CadSeqProc.enhanced_geometry.pattern_recognition import (
    PatternRecognizer, PatternFeature, DesignPattern
)
from CadSeqProc.base import GeometricEntity, Point

class MockGeometricEntity(GeometricEntity):
    """Mock geometric entity for testing."""
    
    def __init__(self, center: Point, dimensions: Dict[str, float]):
        self.center = center
        self.dimensions = dimensions
        self.bounds = (
            Point(center.x - 1, center.y - 1, center.z - 1),
            Point(center.x + 1, center.y + 1, center.z + 1)
        )

class TestPatternRecognition(unittest.TestCase):
    """Test cases for pattern recognition functionality."""
    
    def setUp(self):
        """Set up test cases."""
        self.recognizer = PatternRecognizer()
        
        # Create test geometries
        self.linear_array = self._create_linear_array()
        self.circular_array = self._create_circular_array()
        
    def _create_linear_array(self) -> List[MockGeometricEntity]:
        """Create a linear array of mock entities."""
        entities = []
        for i in range(5):  # 5 entities in a line
            center = Point(i * 2.0, 0.0, 0.0)  # Spaced 2 units apart
            dimensions = {"width": 1.0, "height": 1.0, "depth": 1.0}
            entities.append(MockGeometricEntity(center, dimensions))
        return entities
        
    def _create_circular_array(self) -> List[MockGeometricEntity]:
        """Create a circular array of mock entities."""
        entities = []
        radius = 5.0
        num_entities = 8
        for i in range(num_entities):
            angle = (2 * np.pi * i) / num_entities
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            center = Point(x, y, 0.0)
            dimensions = {"width": 1.0, "height": 1.0, "depth": 1.0}
            entities.append(MockGeometricEntity(center, dimensions))
        return entities
        
    def test_linear_array_detection(self):
        """Test detection of linear array patterns."""
        patterns = self.recognizer.analyze_geometry(self.linear_array[0])
        
        # Should find one linear array pattern
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        self.assertEqual(pattern.features[0].pattern_type, "linear_array")
        
        # Check pattern properties
        feature = pattern.features[0]
        self.assertEqual(len(feature.instances), 4)  # 4 instances after base
        self.assertAlmostEqual(feature.parameters["spacing"], 2.0)  # 2 units spacing
        self.assertGreater(feature.confidence, 0.8)  # High confidence
        
    def test_circular_array_detection(self):
        """Test detection of circular array patterns."""
        patterns = self.recognizer.analyze_geometry(self.circular_array[0])
        
        # Should find one circular array pattern
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        self.assertEqual(pattern.features[0].pattern_type, "circular_array")
        
        # Check pattern properties
        feature = pattern.features[0]
        self.assertEqual(len(feature.instances), 7)  # 7 instances after base
        self.assertAlmostEqual(feature.parameters["radius"], 5.0)  # 5 units radius
        self.assertGreater(feature.confidence, 0.8)  # High confidence
        
    def test_pattern_similarity(self):
        """Test feature similarity comparison."""
        entity1 = MockGeometricEntity(
            Point(0.0, 0.0, 0.0),
            {"width": 1.0, "height": 1.0, "depth": 1.0}
        )
        entity2 = MockGeometricEntity(
            Point(2.0, 0.0, 0.0),
            {"width": 1.1, "height": 0.9, "depth": 1.0}
        )
        
        similarity = self.recognizer._compare_features(entity1, entity2)
        self.assertGreater(similarity, 0.8)  # Should be similar
        
    def test_pattern_relationships(self):
        """Test analysis of pattern relationships."""
        patterns = self.recognizer.analyze_geometry(self.linear_array[0])
        pattern = patterns[0]
        
        relationships = pattern.relationships
        self.assertGreater(len(relationships), 0)
        
        # Check spacing relationship
        spacing_rel = next(r for r in relationships if r["type"] == "spacing")
        self.assertAlmostEqual(spacing_rel["value"], 2.0)
        
    def test_manufacturing_notes(self):
        """Test generation of manufacturing notes."""
        patterns = self.recognizer.analyze_geometry(self.linear_array[0])
        pattern = patterns[0]
        
        self.assertIsNotNone(pattern.manufacturing_notes)
        self.assertIn("note", pattern.manufacturing_notes)
        
    def test_reuse_suggestions(self):
        """Test generation of reuse suggestions."""
        patterns = self.recognizer.analyze_geometry(self.linear_array[0])
        pattern = patterns[0]
        
        self.assertIsNotNone(pattern.reuse_suggestions)
        self.assertGreater(len(pattern.reuse_suggestions), 0)
        
if __name__ == '__main__':
    unittest.main() 

================================================
File: CadSeqProc/utility/decorator.py
================================================
from functools import wraps
import tracemalloc
import time
import gc
import torch
import gc
import datetime
from contextlib import ContextDecorator
from rich import print
# <---------------- Custom Decorators ---------------->

"""
def my_decorator_func(func):

    def wrapper_func():
        # Do something before the function.
        func()
        # Do something after the function.
        # May return the result of the func()
    return wrapper_func
"""


def convert_seconds_to_minutes_and_hours(seconds):
    """
    Converts seconds to minutes and hours.

    Args:
        seconds (int): The number of seconds.

    Returns:
        tuple: The minutes and hours.

    """

    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60

    if minutes==0:
        if seconds<0.1:
            return f"{seconds*1000} ms"
        return f"{seconds} seconds"
    elif hours==0:
        return f"{minutes} minutes"
    else:
        return f"{hours} hours {minutes} minutes"

def timeit(func):
    # Decorator for calculating time
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {convert_seconds_to_minutes_and_hours(total_time)}')
        return result
    return timeit_wrapper


def log_datetime(func):
    """Log the date and time of a function"""
    @wraps(func)
    def log_datetime_wrapper(*args,**kwargs):
        startInfo=f'Function: {func.__name__} \nRun on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        result=func(*args,**kwargs)
        print(startInfo)
        return result
        print(f'{"-"*30}')
    return log_datetime_wrapper


def measure_performance(func):
    '''Measure performance of a function'''

    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()
        result=func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        finish_time = time.perf_counter()
        print(f'{"-"*40}')
        print(f'Function: {func.__name__}')
        print(f'Method: {func.__doc__}')
        print(f'Memory usage:\t\t {current / 10**6:.6f} MB \n'
              f'Peak memory usage:\t {peak / 10**6:.6f} MB ')
        print(f'Time elapsed : {convert_seconds_to_minutes_and_hours(finish_time - start_time)}')
        print(f'{"-"*40}')
        tracemalloc.stop()
        return result
    return wrapper



def gpu_memory_usage(func):
    """
    Decorator that prints the GPU memory usage before and after a function is called.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    """

    def wrapper(*args, **kwargs):
        start_memory = torch.cuda.memory_allocated()
        func(*args, **kwargs)
        end_memory = torch.cuda.memory_allocated()
        print("GPU memory usage: " + str(end_memory - start_memory))

    return wrapper



# Code from https://gist.github.com/MarkTension/4783697ebd5212ba500cdd829b364338
# pytorch method to find number of tensors in the graph
def get_n_tensors():
    tensors= []
    for obj in gc.get_objects():
        try:
            if (torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))):
                tensors.append(obj)
        except:
            pass
        return len(tensors)
  
# this is our context decorator
class check_memory_leak_context(ContextDecorator):
    def __enter__(self):
        self.start = get_n_tensors()
        return self
    def __exit__(self, *exc):
        self.end = get_n_tensors()
        increase = self.end - self.start
        
        if increase > 0:
                print(f"num tensors increased with"\
                    f"{self.end - self.start} !")
        else:
                print("no added tensors")
        return False

"""
Decorator utilities for enhanced geometry system.
"""

import functools
import time
from typing import Any, Callable, TypeVar, cast
from .logger import setup_logger

logger = setup_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

def log_execution(func: F) -> F:
    """Log function execution time and parameters."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return cast(F, wrapper)

def validate_parameters(func: F) -> F:
    """Validate function parameters."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get function parameter names
        params = func.__code__.co_varnames[:func.__code__.co_argcount]
        
        # Check for None values
        for name, value in zip(params, args):
            if value is None:
                logger.warning(f"Parameter {name} is None in {func.__name__}")
        
        for name, value in kwargs.items():
            if value is None:
                logger.warning(f"Parameter {name} is None in {func.__name__}")
        
        return func(*args, **kwargs)
    return cast(F, wrapper)

def cache_result(func: F) -> F:
    """Cache function results."""
    cache: dict = {}
    
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            logger.debug(f"Cached result for {func.__name__}")
        else:
            logger.debug(f"Using cached result for {func.__name__}")
        
        return cache[key]
    return cast(F, wrapper)

def retry_on_error(max_attempts: int = 3, delay: float = 1.0) -> Callable[[F], F]:
    """Retry function on error."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Max retry attempts ({max_attempts}) reached for {func.__name__}")
                        raise
                    logger.warning(f"Attempt {attempts} failed for {func.__name__}: {str(e)}")
                    time.sleep(delay)
            return None  # Should never reach here
        return cast(F, wrapper)
    return decorator

def deprecated(func: F) -> F:
    """Mark function as deprecated."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.warning(f"Function {func.__name__} is deprecated")
        return func(*args, **kwargs)
    return cast(F, wrapper)


================================================
File: CadSeqProc/utility/factory.py
================================================
"""
Factory utility for creating complex geometric shapes.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from CadSeqProc.enhanced_geometry.base import BaseGeometry
from CadSeqProc.enhanced_geometry.nurbs import NURBSCurve, NURBSSurface
from CadSeqProc.enhanced_geometry.organic import OrganicSurface
from CadSeqProc.utility.shape_factory import (
    create_circle, create_sphere, create_cylinder,
    create_cone, create_torus, create_box
)
from CadSeqProc.utility.logger import setup_logger

logger = setup_logger(__name__)

class ShapeFactory:
    """Factory for creating complex geometric shapes."""
    
    @staticmethod
    def create_shape(shape_type: str, params: Dict[str, Any]) -> BaseGeometry:
        """Create shape from type and parameters."""
        try:
            if shape_type == 'organic':
                return ShapeFactory._create_organic_shape(params)
            elif shape_type == 'mechanical':
                return ShapeFactory._create_mechanical_shape(params)
            elif shape_type == 'architectural':
                return ShapeFactory._create_architectural_shape(params)
            else:
                return ShapeFactory._create_basic_shape(shape_type, params)
        except Exception as e:
            logger.error(f"Error creating shape: {str(e)}")
            # Return simple fallback shape
            return create_sphere(1.0)
    
    @staticmethod
    def _create_organic_shape(params: Dict[str, Any]) -> BaseGeometry:
        """Create organic shape from parameters."""
        shape_type = params.get('type', 'generic')
        scale = params.get('scale', 1.0)
        
        if shape_type == 'flower':
            return ShapeFactory._create_flower(
                num_petals=params.get('num_petals', 5),
                petal_length=0.5 * scale,
                petal_width=0.2 * scale,
                center_radius=0.2 * scale
            )
        elif shape_type == 'leaf':
            return ShapeFactory._create_leaf(
                length=1.0 * scale,
                width=0.5 * scale,
                vein_depth=0.05 * scale
            )
        else:
            # Create generic organic shape
            base = create_sphere(0.5 * scale)
            organic = OrganicSurface(base)
            organic.add_random_deformation(0.3)
            return organic
    
    @staticmethod
    def _create_mechanical_shape(params: Dict[str, Any]) -> BaseGeometry:
        """Create mechanical shape from parameters."""
        shape_type = params.get('type', 'generic')
        scale = params.get('scale', 1.0)
        
        if shape_type == 'gear':
            return ShapeFactory._create_gear(
                outer_radius=0.5 * scale,
                inner_radius=0.3 * scale,
                thickness=0.2 * scale,
                num_teeth=params.get('num_teeth', 12)
            )
        elif shape_type == 'bolt':
            return ShapeFactory._create_bolt(
                head_radius=0.3 * scale,
                shaft_radius=0.15 * scale,
                length=1.0 * scale,
                thread_pitch=0.1 * scale
            )
        else:
            # Create generic mechanical shape
            return create_cylinder(0.5 * scale, 1.0 * scale)
    
    @staticmethod
    def _create_architectural_shape(params: Dict[str, Any]) -> BaseGeometry:
        """Create architectural shape from parameters."""
        shape_type = params.get('type', 'generic')
        scale = params.get('scale', 1.0)
        
        if shape_type == 'column':
            return ShapeFactory._create_column(
                height=2.0 * scale,
                radius=0.3 * scale,
                capital_height=0.4 * scale,
                base_height=0.3 * scale
            )
        elif shape_type == 'arch':
            return ShapeFactory._create_arch(
                width=2.0 * scale,
                height=1.5 * scale,
                depth=0.5 * scale,
                thickness=0.2 * scale
            )
        else:
            # Create generic architectural shape
            return create_box(1.0 * scale, 2.0 * scale, 1.0 * scale)
    
    @staticmethod
    def _create_basic_shape(shape_type: str, params: Dict[str, Any]) -> BaseGeometry:
        """Create basic shape from type and parameters."""
        scale = params.get('scale', 1.0)
        center = params.get('center', [0.0, 0.0, 0.0])
        
        if shape_type == 'sphere':
            return create_sphere(0.5 * scale, center)
        elif shape_type == 'cylinder':
            return create_cylinder(0.5 * scale, 1.0 * scale, center)
        elif shape_type == 'cone':
            return create_cone(0.5 * scale, 1.0 * scale, center)
        elif shape_type == 'torus':
            return create_torus(0.5 * scale, 0.2 * scale, center)
        elif shape_type == 'box':
            return create_box(1.0 * scale, 1.0 * scale, 1.0 * scale, center)
        else:
            return create_sphere(0.5 * scale, center)
    
    @staticmethod
    def _create_flower(num_petals: int, petal_length: float,
                      petal_width: float, center_radius: float) -> BaseGeometry:
        """Create flower shape."""
        # Create center
        center = create_sphere(center_radius)
        center_organic = OrganicSurface(center)
        center_organic.add_bumps(20, center_radius * 0.1, center_radius * 0.1)
        
        # Create petals
        petals = []
        for i in range(num_petals):
            angle = (2 * np.pi * i) / num_petals
            
            # Create petal curve
            control_points = [
                [0, 0, 0],
                [petal_length * 0.3, petal_width * 0.5, 0],
                [petal_length * 0.7, petal_width * 0.5, 0],
                [petal_length, 0, 0]
            ]
            curve = NURBSCurve(control_points)
            
            # Create petal surface
            surface = curve.sweep(petal_width)
            petal = OrganicSurface(surface)
            petal.add_random_deformation(0.2)
            petal.rotate(angle)
            
            petals.append(petal)
        
        # Combine shapes
        return OrganicSurface.combine([center_organic] + petals)
    
    @staticmethod
    def _create_leaf(length: float, width: float, vein_depth: float) -> BaseGeometry:
        """Create leaf shape."""
        # Create main surface control points
        points = []
        for i in range(5):
            u = i / 4
            row = []
            for j in range(3):
                v = j / 2 - 0.5
                
                # Create leaf shape
                x = length * u
                y = width * v * (1 - u) * (1 - u)
                z = vein_depth * np.sin(np.pi * u) * (1 - abs(v))
                
                row.append([x, y, z])
            points.append(row)
        
        # Create surface
        surface = NURBSSurface(points)
        leaf = OrganicSurface(surface)
        
        # Add random deformation
        leaf.add_random_deformation(0.1)
        
        return leaf
    
    @staticmethod
    def _create_gear(outer_radius: float, inner_radius: float,
                    thickness: float, num_teeth: int) -> BaseGeometry:
        """Create gear shape."""
        # Create base cylinder
        base = create_cylinder(inner_radius, thickness)
        
        # Create teeth
        teeth_points = []
        for i in range(num_teeth):
            angle = (2 * np.pi * i) / num_teeth
            
            # Create tooth profile
            tooth_points = [
                [inner_radius * np.cos(angle), inner_radius * np.sin(angle), 0],
                [outer_radius * np.cos(angle - 0.1), outer_radius * np.sin(angle - 0.1), 0],
                [outer_radius * np.cos(angle), outer_radius * np.sin(angle), 0],
                [outer_radius * np.cos(angle + 0.1), outer_radius * np.sin(angle + 0.1), 0],
                [inner_radius * np.cos(angle), inner_radius * np.sin(angle), 0]
            ]
            
            # Extrude tooth
            for z in [0, thickness]:
                row = []
                for point in tooth_points:
                    row.append([point[0], point[1], z])
                teeth_points.append(row)
        
        teeth = NURBSSurface(teeth_points)
        return teeth
    
    @staticmethod
    def _create_bolt(head_radius: float, shaft_radius: float,
                    length: float, thread_pitch: float) -> BaseGeometry:
        """Create bolt shape."""
        # Create head
        head = create_cylinder(head_radius, head_radius * 0.8)
        
        # Create shaft
        shaft = create_cylinder(shaft_radius, length)
        
        # Create thread helix
        helix_points = []
        turns = int(length / thread_pitch)
        points_per_turn = 20
        
        for i in range(turns * points_per_turn):
            t = i / points_per_turn
            angle = 2 * np.pi * t
            x = shaft_radius * np.cos(angle)
            y = shaft_radius * np.sin(angle)
            z = t * thread_pitch
            helix_points.append([x, y, z])
        
        thread = NURBSCurve(helix_points)
        
        # TODO: Combine shapes properly
        return shaft
    
    @staticmethod
    def _create_column(height: float, radius: float,
                      capital_height: float, base_height: float) -> BaseGeometry:
        """Create column shape."""
        # Create base
        base = create_cylinder(radius * 1.2, base_height)
        
        # Create shaft
        shaft = create_cylinder(radius, height - capital_height - base_height)
        
        # Create capital
        capital = create_cylinder(radius * 1.3, capital_height)
        
        # TODO: Combine shapes properly
        return shaft
    
    @staticmethod
    def _create_arch(width: float, height: float,
                    depth: float, thickness: float) -> BaseGeometry:
        """Create arch shape."""
        # Create arch curve
        curve_points = []
        for t in np.linspace(0, np.pi, 20):
            x = width/2 * np.cos(t)
            y = height * np.sin(t)
            curve_points.append([x, y, 0])
        
        curve = NURBSCurve(curve_points)
        
        # Create arch surface
        arch = curve.sweep(depth)
        
        # TODO: Add thickness
        return arch 

================================================
File: CadSeqProc/utility/logger.py
================================================
"""
Logging utility for enhanced geometry system.
"""

from loguru import logger
import sys
import logging
from typing import Any, Union, Optional

class CLGLogger:
    """Logger class with enhanced configuration options."""
    
    def __init__(self) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)

    def configure_logger(self, verbose: bool = True) -> 'CLGLogger':
        """Configure logger with specified verbosity level."""
        logger.remove()
        
        if verbose:
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file.name}:{line}</cyan> | <level><level>{message}</level></level>",
                level="TRACE",
                colorize=True,
                backtrace=True,
                diagnose=True,
                enqueue=True,
                catch=True)
        else:
            # Disable logging by adding a null handler
            logger.add(logging.NullHandler())

        return self
    
    def add_log_file(self, file_path: str) -> 'CLGLogger':
        """Add a file handler to the logger."""
        # Add a log file handler
        logger.add(
            file_path,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file.relative}:{line}</cyan> | <level><level>{message}</level></level>",
            level="DEBUG",
            rotation="100 MB"
        )

        return self
    
    def add_log_rotation(self, rotation_size: Union[str, int], backup_count: int) -> 'CLGLogger':
        """Configure log rotation settings."""
        # Set log rotation size and backup count
        logger.configure(
            handlers=[{"sink": sys.stderr, "level": "TRACE", "diagnose": True},
                      {"sink": logging.FileHandler, "level": "DEBUG", "rotation": rotation_size, "backtrace": True, "enqueue": True, "catch": True, "diagnose": True}],
            backtrace=True,
            diagnose=True,
            rotation=rotation_size,
            compression="zip",
            retention=backup_count
        )

        return self

    def add_log_level(self, level: Union[str, int]) -> 'CLGLogger':
        """Set the logging level."""
        # Set the log level
        logger.level(level)

        return self


================================================
File: CadSeqProc/utility/macro.py
================================================
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



================================================
File: CadSeqProc/utility/shape_factory.py
================================================
"""
Factory class for generating organic shapes from basic primitives.
"""

import numpy as np
from typing import List, Tuple, Optional
from ..geometry.nurbs import NURBSCurve, NURBSSurface
from ..geometry.organic import OrganicSurface
from ..sequence.transformation.deform import TwistDeformation, BendDeformation, TaperDeformation

class OrganicShapeFactory:
    """Factory for creating organic shapes."""
    
    @staticmethod
    def create_petal(length: float = 1.0,
                    width: float = 0.3,
                    curve_factor: float = 0.3,
                    twist_angle: float = 0.0) -> OrganicSurface:
        """Create a petal shape."""
        # Create control points for petal surface
        num_u, num_v = 5, 3
        control_points = []
        
        for i in range(num_u):
            row = []
            u = i / (num_u - 1)
            
            # Base curve follows a quadratic shape
            base_x = length * u
            base_y = 0
            base_z = curve_factor * u * (1 - u)
            
            for j in range(num_v):
                v = j / (num_v - 1) - 0.5
                # Width follows a cubic falloff
                width_factor = 1 - (2*v)**2
                
                x = base_x
                y = width * v * width_factor
                z = base_z * width_factor
                
                row.append((x, y, z))
            control_points.append(row)
        
        surface = NURBSSurface(control_points)
        organic_surface = OrganicSurface([surface])
        
        # Add twist if specified
        if twist_angle != 0:
            organic_surface.add_deformation(
                TwistDeformation(twist_angle, axis='x')
            )
        
        return organic_surface
    
    @staticmethod
    def create_leaf(length: float = 1.0,
                   width: float = 0.3,
                   curve_factor: float = 0.2,
                   vein_depth: float = 0.05) -> OrganicSurface:
        """Create a leaf shape."""
        num_u, num_v = 7, 5
        control_points = []
        
        for i in range(num_u):
            row = []
            u = i / (num_u - 1)
            
            # Base curve with upward curve
            base_x = length * u
            base_y = 0
            base_z = curve_factor * np.sin(np.pi * u)
            
            for j in range(num_v):
                v = j / (num_v - 1) - 0.5
                # Width follows an elliptical shape
                width_factor = np.sqrt(1 - (2*v)**2)
                # Add vein pattern
                vein_z = vein_depth * np.cos(np.pi * v)
                
                x = base_x
                y = width * v * width_factor * (1 - u**0.5)  # Taper towards tip
                z = base_z + vein_z * width_factor
                
                row.append((x, y, z))
            control_points.append(row)
        
        surface = NURBSSurface(control_points)
        return OrganicSurface([surface])
    
    @staticmethod
    def create_flower(num_petals: int = 5,
                     petal_length: float = 1.0,
                     petal_width: float = 0.3,
                     center_radius: float = 0.2) -> List[OrganicSurface]:
        """Create a flower with multiple petals."""
        shapes = []
        
        # Create center
        center_points = []
        num_u, num_v = 5, 8
        for i in range(num_u):
            row = []
            u = i / (num_u - 1)
            radius = center_radius * (1 - u**2)
            height = center_radius * 0.5 * u
            
            for j in range(num_v):
                angle = 2 * np.pi * j / (num_v - 1)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = height
                row.append((x, y, z))
            center_points.append(row)
        
        center_surface = NURBSSurface(center_points)
        shapes.append(OrganicSurface([center_surface]))
        
        # Create and position petals
        for i in range(num_petals):
            angle = 2 * np.pi * i / num_petals
            petal = OrganicShapeFactory.create_petal(
                length=petal_length,
                width=petal_width,
                curve_factor=0.3,
                twist_angle=0.2
            )
            
            # Create transformation matrix
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, center_radius * c],
                [s, c, 0, center_radius * s],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            shapes.append(petal.transform(transform))
        
        return shapes
    
    @staticmethod
    def create_tree(trunk_height: float = 2.0,
                   trunk_radius: float = 0.2,
                   num_branches: int = 5,
                   leaf_size: float = 0.5) -> List[OrganicSurface]:
        """Create a simple tree structure."""
        shapes = []
        
        # Create trunk
        trunk_points = []
        num_u, num_v = 5, 8
        for i in range(num_u):
            row = []
            u = i / (num_u - 1)
            radius = trunk_radius * (1 - 0.3 * u)  # Slight taper
            height = trunk_height * u
            
            for j in range(num_v):
                angle = 2 * np.pi * j / (num_v - 1)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = height
                row.append((x, y, z))
            trunk_points.append(row)
        
        trunk_surface = NURBSSurface(trunk_points)
        shapes.append(OrganicSurface([trunk_surface]))
        
        # Add branches with leaves
        for i in range(num_branches):
            height_factor = 0.3 + 0.7 * i / (num_branches - 1)
            angle = 2 * np.pi * i / num_branches
            
            # Create leaf
            leaf = OrganicShapeFactory.create_leaf(
                length=leaf_size,
                width=leaf_size * 0.4,
                curve_factor=0.2
            )
            
            # Position leaf
            c, s = np.cos(angle), np.sin(angle)
            transform = np.array([
                [c, -s, 0, trunk_radius * 2 * c],
                [s, c, 0, trunk_radius * 2 * s],
                [0, 0, 1, trunk_height * height_factor],
                [0, 0, 0, 1]
            ])
            
            shapes.append(leaf.transform(transform))
        
        return shapes
    
    @staticmethod
    def create_vine(control_points: List[Tuple[float, float, float]],
                   thickness: float = 0.1,
                   num_leaves: int = 5,
                   leaf_size: float = 0.3) -> List[OrganicSurface]:
        """Create a vine with leaves."""
        shapes = []
        
        # Create vine curve
        vine_curve = NURBSCurve.from_points(control_points)
        
        # Create vine surface
        vine_points = []
        num_u, num_v = len(control_points), 8
        
        for i in range(num_u):
            row = []
            u = i / (num_u - 1)
            point = vine_curve.evaluate(u)
            
            for j in range(num_v):
                angle = 2 * np.pi * j / (num_v - 1)
                x = point[0] + thickness * np.cos(angle)
                y = point[1] + thickness * np.sin(angle)
                z = point[2]
                row.append((x, y, z))
            vine_points.append(row)
        
        vine_surface = NURBSSurface(vine_points)
        shapes.append(OrganicSurface([vine_surface]))
        
        # Add leaves along the vine
        for i in range(num_leaves):
            t = i / (num_leaves - 1)
            point = vine_curve.evaluate(t)
            
            # Create leaf
            leaf = OrganicShapeFactory.create_leaf(
                length=leaf_size,
                width=leaf_size * 0.4,
                curve_factor=0.2
            )
            
            # Calculate orientation based on curve tangent
            delta = 0.01
            next_point = vine_curve.evaluate(min(t + delta, 1.0))
            tangent = np.array(next_point) - np.array(point)
            tangent /= np.linalg.norm(tangent)
            
            # Create transformation matrix
            up = np.array([0, 0, 1])
            right = np.cross(tangent, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, tangent)
            
            transform = np.array([
                [right[0], tangent[0], up[0], point[0]],
                [right[1], tangent[1], up[1], point[1]],
                [right[2], tangent[2], up[2], point[2]],
                [0, 0, 0, 1]
            ])
            
            shapes.append(leaf.transform(transform))
        
        return shapes 

================================================
File: Cad_VLM/test.py
================================================
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import chamfer_dist, normalize_pc
from CadSeqProc.utility.logger import CLGLogger
from Cad_VLM.models.text2cad import Text2CAD
from Cad_VLM.models.utils import print_with_separator
from Cad_VLM.dataprep.t2c_dataset import get_dataloaders
from loguru import logger
from rich import print
import torch
import argparse
from tqdm import tqdm
import os
import datetime
import argparse
import yaml
import warnings
import logging.config
import pickle

warnings.filterwarnings("ignore")
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)

t2clogger = CLGLogger().configure_logger(verbose=True).logger

# ---------------------------------------------------------------------------- #
#                              Text2CAD Test Code                              #
# ---------------------------------------------------------------------------- #


def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def save_yaml_file(yaml_data, filename, output_dir):
    with open(os.path.join(output_dir, filename), "w+") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)


@logger.catch()
def main():
    print_with_separator("😊 Text2CAD Inference 😊")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config/inference.yaml",
    )
    args = parser.parse_args()
    config = parse_config_file(args.config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t2clogger.info(
        "Current Device {}",
        torch.cuda.get_device_properties(device),
    )

    # -------------------------------- Load Model -------------------------------- #
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    text2cad = Text2CAD(text_config=config["text_encoder"], cad_config=cad_config).to(
        device
    )

    dim = config["cad_decoder"]["cdim"]
    nlayers = config["cad_decoder"]["num_layers"]
    ca_level_start = config["cad_decoder"]["ca_level_start"]

    # --------------------------- Prepare Log Directory -------------------------- #
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = datetime.date.today()
    log_dir = os.path.join(
        config["test"]["log_dir"],
        f"{date_str}/{time_str}_d{dim}_nl{nlayers}_ca{ca_level_start}",
    )
    t2clogger.info(
        "Current Date {date_str} Time {time_str}\n",
        date_str=date_str,
        time_str=time_str,
    )

    # Create the log dir if it doesn't exist
    if not config["debug"]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        save_yaml_file(
            config, filename=args.config_path.split("/")[-1], output_dir=log_dir
        )

    # -------------------------------- Train Model ------------------------------- #

    test_model(
        model=text2cad,
        device=device,
        log_dir=log_dir,
        config=config,
        logger=t2clogger,
    )


def test_model(
    model,
    device,
    log_dir,
    config,
    logger,
):
    """
    Trains a deep learning model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        device (str): Device to train on ('cuda' for GPU, 'cpu' for CPU).
        log_dir (str): Directory to save logs and checkpoints.
        config (dict): Additional configuration parameters.

    Returns:
        None
    """

    # Create the dataloader for train
    test_loader = get_dataloaders(
        cad_seq_dir=config["test_data"]["cad_seq_dir"],
        prompt_path=config["test_data"]["prompt_path"],
        split_filepath=config["test_data"]["split_filepath"],
        subsets=["test"],
        batch_size=config["test"]["batch_size"],
        num_workers=config["test"]["num_workers"],
        pin_memory=True,
        shuffle=False,  # If curriculum learning is enabled, set to False else it will automatically shuffle
        prefetch_factor=config["test"]["prefetch_factor"],
    )[0]

    if not config["debug"]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    logger.info(f"Saving results in {log_dir}.")
    if config["test"]["checkpoint_path"] is not None:
        checkpoint_file = config["test"]["checkpoint_path"]

        print(f"Using saved checkpoint at {checkpoint_file}.")
        checkpoint_file_name = checkpoint_file.split("/")[-1]
        checkpoint = torch.load(checkpoint_file, map_location=device)

        if "epoch" in checkpoint:
            print(f"Model was trained for epoch {checkpoint['epoch']}.")

        pretrained_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.split(".")[0] == "module":
                pretrained_dict[".".join(key.split(".")[1:])] = value
            else:
                pretrained_dict[key] = value

        model.load_state_dict(pretrained_dict, strict=False)
        if not config["debug"]:
            save_yaml_file(
                config,
                filename=f"config_{checkpoint_file_name.split('.')[0]}.yaml",
                output_dir=log_dir,
            )

    # ---------------------------------- Inference ---------------------------------- #
    test_acc_uid = {}
    model.eval()
    if config["test"]["sampling_type"] == "max":
        TOPK = 1
    else:
        TOPK = 5

    with torch.no_grad():
        with tqdm(test_loader, ascii=True, desc=f"Inference✨") as pbar:
            for uid_level, vec_dict, prompt, _ in pbar:
                for key, value in vec_dict.items():
                    vec_dict[key] = value.to(device)

                for topk_index in range(1, TOPK + 1):
                    # Autoregressive Prediction (5 outputs per sample)
                    pred_cad_seq_dict = model.test_decode(
                        texts=prompt,
                        maxlen=MAX_CAD_SEQUENCE_LENGTH,
                        nucleus_prob=0,
                        topk_index=topk_index,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )

                    # Save the results batchwise
                    for i in range(vec_dict["cad_vec"].shape[0]):
                        uid, level = uid_level[i].split("_")
                        # if topk_index == 1:
                        if uid[i] not in test_acc_uid:
                            test_acc_uid[uid[i]] = {}

                        if level not in test_acc_uid[uid[i]]:
                            test_acc_uid[uid[i]][level] = {}  # {"response":response}

                        is_invalid = 0
                        try:
                            gt_cad = (
                                CADSequence.from_vec(
                                    vec_dict["cad_vec"][i].cpu().numpy(),
                                    bit=N_BIT,
                                    post_processing=True,
                                )
                                .create_cad_model()
                                .sample_points(n_points=8192)
                            )
                        except:
                            continue

                        try:
                            pred_cad = (
                                CADSequence.from_vec(
                                    pred_cad_seq_dict["cad_vec"][i].cpu().numpy(),
                                    bit=N_BIT,
                                    post_processing=True,
                                )
                                .create_cad_model()
                                .sample_points(n_points=8192)
                            )

                        except Exception as e:
                            is_invalid = 1
                            pred_cad = None

                        # Save the model prediction output
                        try:
                            test_acc_uid[uid[i]][level]["pred_cad_vec"].append(
                                pred_cad_seq_dict["cad_vec"][i].cpu().numpy()
                            )
                        except:
                            test_acc_uid[uid[i]][level]["pred_cad_vec"] = [
                                pred_cad_seq_dict["cad_vec"][i].cpu().numpy()
                            ]
                            # Adding Ground Truth Label
                            test_acc_uid[uid[i]][level]["gt_cad_vec"] = (
                                vec_dict["cad_vec"][i].cpu().numpy()
                            )
                            test_acc_uid[uid[i]][level]["cd"] = []

                        # If the model is valid, add the chamfer distance (Multiplied by 1000)
                        if is_invalid == 0:
                            cd = (
                                chamfer_dist(
                                    normalize_pc(gt_cad.points),
                                    normalize_pc(pred_cad.points),
                                )
                                * 1000
                            )
                        else:  # If the model is invalid, -1 chamfer distance (will be filtered in the evaluation stage)
                            cd = -1

                        test_acc_uid[uid[i]][level]["cd"].append(cd)

                        pbar.set_postfix({"uid": uid[i], "cd": cd})

                        test_acc_uid[uid[i]][level]["is_invalid"] = is_invalid

                # if not config['debug']:
                #     # Save the pkl files
                #     with open(log_dir+"/output.pkl", "wb") as f:
                #         pickle.dump(test_acc_uid, f,
                #                     protocol=pickle.HIGHEST_PROTOCOL)

    if not config["debug"]:
        # Save the pkl files (overwrites the previous file)
        with open(log_dir + "/output.pkl", "wb") as f:
            pickle.dump(test_acc_uid, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.success("Inference Complete")


if __name__ == "__main__":
    main()


================================================
File: Cad_VLM/test_user_input.py
================================================
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.logger import CLGLogger
from Cad_VLM.models.text2cad import Text2CAD
from Cad_VLM.models.utils import print_with_separator, text_prompt
from loguru import logger
from rich import print
import torch
import argparse
import os
import datetime
import argparse
import yaml
import warnings
import logging.config
import pickle

warnings.filterwarnings("ignore")
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)

t2clogger = CLGLogger().configure_logger(verbose=True).logger

# ---------------------------------------------------------------------------- #
#                    Generate CAD Sequence from User Inputs                    #
# ---------------------------------------------------------------------------- #


def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def save_yaml_file(yaml_data, filename, output_dir):
    with open(os.path.join(output_dir, filename), "w+") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)


@logger.catch()
def main():
    print_with_separator("⚡Text2CAD Test from User Input⚡")

    # --------------------------------- Argument --------------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config/inference_user_input.yaml",
    )
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    config = parse_config_file(args.config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t2clogger.info(
        "Current Device {}",
        torch.cuda.get_device_properties(device),
    )

    # -------------------------------- Load Model -------------------------------- #
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    text2cad = Text2CAD(text_config=config["text_encoder"], cad_config=cad_config).to(
        device
    )

    dim = config["cad_decoder"]["cdim"]
    nlayers = config["cad_decoder"]["num_layers"]
    ca_level_start = config["cad_decoder"]["ca_level_start"]

    # --------------------------- Prepare Log Directory -------------------------- #
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = datetime.date.today()
    log_dir = os.path.join(
        config["test"]["log_dir"],
        f"{date_str}/{time_str}_d{dim}_nl{nlayers}_ca{ca_level_start}",
    )
    t2clogger.info(
        "Current Date {date_str} Time {time_str}\n",
        date_str=date_str,
        time_str=time_str,
    )

    # Create the log dir if it doesn't exist
    if not config["debug"]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        save_yaml_file(
            config, filename=args.config_path.split("/")[-1], output_dir=log_dir
        )

    # -------------------------------- Train Model ------------------------------- #
    test_model(
        prompt=args.prompt,
        model=text2cad,
        device=device,
        log_dir=log_dir,
        config=config,
    )


def test_model(
    prompt,
    model,
    device,
    log_dir,
    config,
):
    """
    Trains a deep learning model.

    Parameters:
        prompt (str): Text prompt for CAD sequence generation.
        model (torch.nn.Module): Model to be used for inference.
        device (str): Device to be used for training.
        log_dir (str): Directory to save the results.
        config (dict): Configuration dictionary.

    Returns:
        None
    """

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"Saving results in {log_dir}")
    if config["test"]["checkpoint_path"] is not None:
        checkpoint_file = config["test"]["checkpoint_path"]

        print(f"Using saved checkpoint at {checkpoint_file}")
        # checkpoint_file_name = checkpoint_file.split("/")[-1]
        checkpoint = torch.load(checkpoint_file, map_location=device)
        pretrained_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.split(".")[0] == "module":
                pretrained_dict[".".join(key.split(".")[1:])] = value
            else:
                pretrained_dict[key] = value
        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
            t2clogger.info(f"Model was trained for epoch {epoch}.")

        model.load_state_dict(pretrained_dict, strict=False)

    # ---------------------------------- Testing ---------------------------------- #
    test_acc_uid = {}

    if config["test"]["sampling_type"] == "max":
        TOPK = 1
    else:
        TOPK = 5

    # Get the text prompts
    if prompt is None:
        text = text_prompt(config["test"]["prompt_file"])
    else:
        text = [prompt]
        t2clogger.info(f"Using the user input text prompt.")

    num_texts = len(text)
    if num_texts == 0:
        raise Exception(
            f'No text found in the prompt file 😥. Please check the prompt file in {config["test"]["prompt_file"]} 🔍.'
        )
    else:
        t2clogger.info(f"Found {num_texts} prompts in the prompt file.")

    model.eval()
    batch_size=min(config["test"]["batch_size"], num_texts)
    with torch.no_grad():
        t2clogger.info("Generating CAD Sequence.")
        for b in range(num_texts // batch_size):
            # Autoregressive Generation of CAD Sequences from Text Prompts
            pred_cad_seq_dict = model.test_decode(
                texts=text[
                    b
                    * batch_size : (b + 1)
                    * batch_size
                ],
                maxlen=MAX_CAD_SEQUENCE_LENGTH,
                nucleus_prob=0,
                topk_index=TOPK,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            # Save the results batchwise
            for i in range(
                len(
                    text[
                        b
                        * batch_size : (b + 1)
                        * batch_size
                    ]
                )
            ):
                index = i + b * batch_size
                try:
                    CADSequence.from_vec(
                        pred_cad_seq_dict["cad_vec"][i].cpu().numpy(),
                        bit=N_BIT,
                        post_processing=True,
                    ).save_stp(f"pred", os.path.join(log_dir, str(index)))

                except Exception as e:
                    print(f"Invalid Model Generated for example {index}")
                # Save the model prediction output
                test_acc_uid[index] = {}
                test_acc_uid[index]["pred_cad_vec"] = (
                    pred_cad_seq_dict["cad_vec"][i].cpu().numpy()
                )
                test_acc_uid[index]["text_prompt"] = text[index]
    # Save the pkl files
    with open(log_dir + "/output.pkl", "wb") as f:
        pickle.dump(test_acc_uid, f, protocol=pickle.HIGHEST_PROTOCOL)

    t2clogger.info(f"Cad Sequence Generation Complete. Results are saved in {log_dir}.")


if __name__ == "__main__":
    main()


================================================
File: Cad_VLM/train.py
================================================
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

import random
import numpy as np
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.logger import CLGLogger
from Cad_VLM.models.text2cad import Text2CAD
from Cad_VLM.models.loss import CELoss
from Cad_VLM.models.metrics import AccuracyCalculator
from Cad_VLM.models.utils import print_with_separator
from Cad_VLM.dataprep.t2c_dataset import get_dataloaders
from loguru import logger
import torch
import argparse
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from torch.optim.lr_scheduler import ExponentialLR
import datetime
import gc
import argparse
import yaml
import warnings
import logging.config

warnings.filterwarnings("ignore")
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)

t2clogger = CLGLogger().configure_logger(verbose=True).logger

# ---------------------------------------------------------------------------- #
#                            Text2CAD Training Code                            #
# ---------------------------------------------------------------------------- #


def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def save_yaml_file(yaml_data, filename, output_dir):
    with open(os.path.join(output_dir, filename), "w+") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)


@logger.catch()
def main():
    print_with_separator("😊 Text2CAD Training 😊")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config/trainer.yaml",
    )
    args = parser.parse_args()
    config = parse_config_file(args.config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t2clogger.info(
        "Current Device {}",
        torch.cuda.get_device_properties(device),
    )

    # -------------------------------- Load Model -------------------------------- #
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    text2cad = Text2CAD(text_config=config["text_encoder"], cad_config=cad_config).to(
        device
    )

    # Freeze the base text embedder
    for param in text2cad.base_text_embedder.parameters():
        param.requires_grad = False

    # text2cad = torch.nn.DataParallel(
    #     text2cad
    # )  # For Parallel Processing (during Training)

    optimizer = optim.AdamW(text2cad.parameters(), lr=config["train"]["lr"])
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    criterion = CELoss(device=device)

    lr = config["train"]["lr"]
    dim = config["cad_decoder"]["cdim"]
    nlayers = config["cad_decoder"]["num_layers"]
    batch = config["train"]["batch_size"]
    ca_level_start = config["cad_decoder"]["ca_level_start"]

    # --------------------------- Prepare Log Directory -------------------------- #
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = datetime.date.today()
    log_dir = os.path.join(
        config["train"]["log_dir"],
        f"{date_str}/{time_str}_d{dim}_nl{nlayers}_ca{ca_level_start}",
    )
    t2clogger.info(
        "Current Date {date_str} Time {time_str}\n",
        date_str=date_str,
        time_str=time_str,
    )

    # Create the log dir if it doesn't exist
    if not config["debug"]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        save_yaml_file(
            config, filename=args.config_path.split("/")[-1], output_dir=log_dir
        )

    # -------------------------------- Train Model ------------------------------- #

    train_model(
        model=text2cad,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=log_dir,
        num_epochs=config["train"]["num_epochs"],
        checkpoint_name=f"lr{lr}_d{dim}_nl{nlayers}_b{batch}_ca{ca_level_start}",
        config=config,
    )


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    log_dir,
    num_epochs,
    checkpoint_name,
    config,
):
    """
    Trains a deep learning model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        scheduler: Learning rate scheduler.
        device (str): Device to train on ('cuda' for GPU, 'cpu' for CPU).
        log_dir (str): Directory to save logs and checkpoints.
        num_epochs (int): Number of training epochs.
        checkpoint_name (str): Name to save the checkpoints.
        config (dict): Additional configuration parameters.

    Returns:
        None
    """

    # Create the dataloader for train
    train_loader, val_loader = get_dataloaders(
        cad_seq_dir=config["train_data"]["cad_seq_dir"],
        prompt_path=config["train_data"]["prompt_path"],
        split_filepath=config["train_data"]["split_filepath"],
        subsets=["train", "validation"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        shuffle=False,  # If curriculum learning is enabled, set to False else it will automatically shuffle
        prefetch_factor=config["train"]["prefetch_factor"],
    )

    tensorboard_dir = os.path.join(log_dir, f"summary")
    # ---------------------- Resume Training from checkpoint --------------------- #
    checkpoint_file = os.path.join(log_dir, f"t2c_{checkpoint_name}.pth")
    checkpoint_only_model_file = os.path.join(
        log_dir, f"t2c_{checkpoint_name}_model.pth"
    )

    if config["train"]["checkpoint_path"] is None:
        old_checkpoint_file = checkpoint_file
    else:
        old_checkpoint_file = config["train"]["checkpoint_path"]

    if os.path.exists(old_checkpoint_file):
        t2clogger.info("Using saved checkpoint at {}", old_checkpoint_file)
        checkpoint = torch.load(old_checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        step = checkpoint["step"]
    else:
        step = 0
        start_epoch = 1
    t2clogger.info("Saving checkpoint at {}", checkpoint_file)

    # Create the tensorboard summary writer
    writer = SummaryWriter(log_dir=tensorboard_dir, comment=f"{checkpoint_name}")
    
    model = torch.nn.DataParallel(
        model
    )  # For Parallel Processing (during Training)

    # ---------------------------------- Training ---------------------------------- #
    if start_epoch > config["train"]["curriculum_learning_epoch"]:
        t2clogger.warning("MIXED LEARNING...")
        random.shuffle(train_loader.dataset.uid_pair)
    else:
        t2clogger.info("CURRICULUM LEARNING...")

    # model=torch.compile(model)
    # Start training
    model.train()
    for epoch in range(start_epoch, num_epochs + 1):
        # ------------------------------- Single Epoch ------------------------------- #
        # Shuffle the data when curriculum learning stops
        if epoch == config["train"]["curriculum_learning_epoch"]:
            t2clogger.info("MIXED LEARNING...")
            optimizer = optim.AdamW(model.parameters(), lr=config["train"]["lr"])
            scheduler = ExponentialLR(optimizer, gamma=0.99)

        if epoch >= config["train"]["curriculum_learning_epoch"]:
            # Note: Works as dataloader(shuffle=True) for the current epoch
            random.shuffle(train_loader.dataset.uid_pair)

        # Train for one epoch
        train_loss = []
        train_loss_seq = {"seq": []}
        train_accuracy_seq = {"seq": []}
        val_accuracy_seq = {"seq": []}

        with tqdm(
            train_loader,
            ascii=True,
            desc=f"\033[94mText2CAD\033[0m: Epoch [{epoch}/{num_epochs+1}]✨",
        ) as pbar:
            for _, vec_dict, prompt, mask_cad_dict in pbar:
                step += 1

                for key, value in vec_dict.items():
                    vec_dict[key] = value.to(device)

                for key, value in mask_cad_dict.items():
                    mask_cad_dict[key] = value.to(device)

                # Padding mask for predicted Cad Sequence
                shifted_key_padding_mask = mask_cad_dict["key_padding_mask"][:, 1:]
                # Create Label for Training
                cad_vec_target = vec_dict["cad_vec"][:, 1:].clone()

                # ------------------ Forward pass by Teacher Forcing Method ------------------ #

                # Create training input by removing the last token
                for key, value in vec_dict.items():
                    vec_dict[key] = value[:, :-1]

                # Padding mask for input Cad Sequence
                mask_cad_dict["key_padding_mask"] = mask_cad_dict["key_padding_mask"][
                    :, :-1
                ]

                # Output from the model
                cad_vec_pred, _ = model(
                    vec_dict=vec_dict,
                    texts=prompt,
                    mask_cad_dict=mask_cad_dict,
                    metadata=False,
                )  # (B,N1,2,C1)

                # ----------------------------- Loss Calculation ----------------------------- #
                loss, loss_sep_dict = criterion(
                    {
                        "pred": cad_vec_pred,
                        "target": cad_vec_target,
                        "key_padding_mask": ~shifted_key_padding_mask,
                    }
                )

                # ------------------------------- Backward pass ------------------------------ #
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=0.9, norm_type=2.0
                )
                optimizer.step()

                # --------------------------- Log loss and accuracy -------------------------- #
                train_loss.append(loss.item())
                train_loss_seq["seq"].append(loss_sep_dict["loss_seq"])

                # Compute accuracy
                cad_accuracy = AccuracyCalculator(
                    discard_token=len(END_TOKEN)
                ).calculateAccMulti2DFromProbability(cad_vec_pred, cad_vec_target)

                # Add Accuracy report
                train_accuracy_seq["seq"].append(cad_accuracy)
                pbar_keys = ["Loss", "seq"]

                updated_dict = {
                    key: (
                        np.round(train_loss[-1], decimals=2)
                        if key == "Loss"
                        else np.round(train_accuracy_seq[key.lower()][-1], decimals=2)
                    )
                    for key in pbar_keys
                }
                # Update the progress bar
                pbar.set_postfix(updated_dict)

                # ---------------------------- Add to Tensorboard ---------------------------- #

                if not config["debug"]:
                    # Add Losses
                    writer.add_scalar(
                        "Seq Loss (Train)",
                        np.mean(train_loss_seq["seq"]),
                        step,
                        new_style=True,
                    )

                    # Add Accuracies
                    writer.add_scalar(
                        "Seq Accuracy (Train)",
                        np.mean(train_accuracy_seq["seq"]),
                        step,
                        new_style=True,
                    )

                    writer.add_scalar(
                        "Total Train Loss", np.mean(train_loss), step, new_style=True
                    )

        # Perform Validation
        val_cad_acc = validation_one_epoch(
            val_loader=val_loader,
            model=model,
            epoch=epoch,
            num_epochs=num_epochs,
            writer=writer,
            config=config,
            total_batch=config["val"]["val_batch"],
        )
        val_accuracy_seq["seq"].append(val_cad_acc)
        # ---------------- Save the model weights and optimizer state ---------------- #
        if not config["debug"]:
            # Save checkpoints
            if epoch % config["train"]["checkpoint_interval"] == 0:

                # Save only the model weights
                checkpoint_only_model_file = os.path.join(
                    log_dir, f"t2c_{checkpoint_name}_{epoch}_model.pth"
                )
                # torch.save(
                #     {
                #         "epoch": epoch,
                #         "model_state_dict": model.state_dict(),
                #         "step": step,
                #     },
                #     checkpoint_only_model_file,
                # )

                # Save the model weights and optimizer states
                # Only save trainable parameters


                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.get_trainable_state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                    },
                    checkpoint_file,
                )

        scheduler.step()
        # Print epoch summary
        logger.info(
            f"Epoch [{epoch}/{num_epochs+1}]✅,"
            f" Train Loss: {np.round(np.mean(train_loss), decimals=2)},"
            f" Train Seq Acc: {np.round(np.mean(train_accuracy_seq['seq']), decimals=2)},"
            f" Val Seq Acc: {np.round(np.mean(val_accuracy_seq['seq']), decimals=2)}",
        )

    # Close the tensorboard summary writer
    writer.close()
    t2clogger.success("Training Finished.")


def validation_one_epoch(
    val_loader,
    model,
    epoch=0,
    num_epochs=0,
    writer=None,
    topk=5,
    config=None,
    total_batch=5,
):
    """
    Perform one validation epoch on the given validation loader.

    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader for validation dataset.
        model (torch.nn.Module): The model to be validated.
        epoch (int, required): Current epoch number. Defaults to 0.
        num_epochs (int, required): Total number of epochs. Defaults to 0.
        writer (SummaryWriter, optional): TensorBoard SummaryWriter for logging. Defaults to None.
        topk (int, optional): Hybrid Sampling. Defaults to 5. Set to 1 for top-1
        config (dict, optional): Additional configuration parameters. Defaults to None.

    Returns:
        tuple: Mean Sequence Token Accuracy (mean_seq_token_acc)
    """
    seq_acc_all = []

    # Get available GPU ID and set the device to that GPU
    # gpu_id = get_available_gpu_ids()[0]
    device = torch.device(f"cuda")
    val_model = model.module.to(device)  # Move the model to the GPU
    val_model.eval()

    cur_batch = 0
    with torch.no_grad():
        # tqdm is used to create a progress bar for validation
        with tqdm(
            val_loader, ascii=True, desc=f"Epoch [{epoch}/{num_epochs+1}] Validation✨"
        ) as pbar:
            for _, vec_dict, prompt, mask_cad_dict in pbar:
                # If the number of batches specified by num_batch is reached, break the loop
                if cur_batch == total_batch:
                    break
                cur_batch += 1

                for key, val in vec_dict.items():
                    vec_dict[key] = val.to(device)

                for key, val in mask_cad_dict.items():
                    mask_cad_dict[key] = val.to(device)

                # Create a copy of the sequence dictionaries, and take only the start token
                sec_topk_acc = []
                for topk_index in range(1, topk + 1):
                    new_cad_seq_dict = vec_dict.copy()

                    for key, value in new_cad_seq_dict.items():
                        new_cad_seq_dict[key] = value[:, :1]
                        # new_mask_pc_seg_dict = {}

                    # Autoregressive Prediction (topk outputs per sample)
                    pred_cad_seq_dict = val_model.test_decode(
                        texts=prompt,
                        maxlen=MAX_CAD_SEQUENCE_LENGTH,
                        nucleus_prob=0,
                        topk_index=topk_index,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Calculate accuracies
                    try:
                        cad_seq_acc = AccuracyCalculator(
                            discard_token=len(END_TOKEN)
                        ).calculateAccMulti2DFromLabel(
                            pred_cad_seq_dict["cad_vec"].cpu(),
                            vec_dict["cad_vec"].cpu(),
                        )
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        cad_seq_acc = 0

                    # Update progress bar with current accuracy information
                    pbar.set_postfix({"Seq": np.round(cad_seq_acc, decimals=2)})

                    # Store accuracies for each batch
                    sec_topk_acc.append(cad_seq_acc)

                seq_acc_all.append(np.max(sec_topk_acc))
                sec_topk_acc = []

            # Calculate mean accuracies for sketches and extrusions
            mean_seq_acc = np.mean(seq_acc_all)
            gc.collect()
            torch.cuda.empty_cache()

            # If a writer is provided, log the mean accuracies to TensorBoard
            if writer is not None:
                writer.add_scalar(
                    "Seq Accuracy (Val)",
                    np.round(mean_seq_acc, decimals=2),
                    epoch,
                    new_style=True,
                )

            # Return mean accuracies for sketches and extrusions
            return mean_seq_acc


if __name__ == "__main__":
    main()


================================================
File: Cad_VLM/config/inference.yaml
================================================
# ---------------------- Configuration for text encoder ---------------------- #
text_encoder:
  # ----------------- Configuration for the Base Text Embedder ----------------- #
  text_embedder:
    # Name of the text encoder model
    model_name: "bert_large_uncased"
    # Maximum Text Sequence Length
    max_seq_len: 512
    # Cache Directory
    cache_dir: "YOUR DIR"

  # ----------------- Configuration for the Adaptive Layer ----------------- #
  adaptive_layer:
    # Input dimension of the text encoder (1024 for bert, else 4096)
    in_dim: 1024
    # Output dimension of the text encoder (1024 for bert, else 4096)
    out_dim: 1024
    # Number of attention heads in the text encoder
    num_heads: 8
    # Dropout probability in the text encoder
    dropout: 0.1

# ----------------------- Configuration for CAD Decoder ---------------------- #
cad_decoder:
  # Dimension of the latent variable z in the CAD decoder (1024 for bert, else 4096)
  tdim: 1024
  # Dimension of the state variable s in the CAD decoder
  cdim: 256
  # Number of transformer layers in the CAD decoder
  num_layers: 8
  # Number of attention heads in each layer of the CAD decoder
  num_heads: 8
  # Dropout probability in the CAD decoder
  dropout: 0.1
  # Starting level for channel attention in the CAD decoder
  ca_level_start: 2

# --------------- Configuration related to training dataloader --------------- #
test_data:
  # Root directory of the training data
  cad_seq_dir: "YOUR DIR"
  # Path to the CSV file containing the text prompts
  prompt_path: "YOUR PATH"
  # JSON file containing information about train, test, and validation splits
  split_filepath: "YOUR PATH"
  # Maximum sequence length for input data
  max_seq_len: 512

# --------------------- Configuration related to training -------------------- #
test:
  # Batch size for training
  batch_size: 4 # for 80 GB gpu
  # Number of workers for the DataLoader during training
  num_workers: 30
  # Prefetch factor for the DataLoader during training
  prefetch_factor: 10
  # Directory for logging training information
  log_dir: "YOUR DIR"
  # Path to saved model checkpoint (optional)
  checkpoint_path: "YOUR CHECKPOINT PATH"
  nucleus_prob: 0
  sampling_type: "max"

# ------------------------------ Debug mode flag ----------------------------- #
debug: False

# --------------- Additional information (leave empty for now) --------------- #
info: "Inference"


================================================
File: Cad_VLM/config/inference_user_input.yaml
================================================
# ---------------------- Configuration for text encoder ---------------------- #
text_encoder:
  # ----------------- Configuration for the Base Text Embedder ----------------- #
  text_embedder:
    # Name of the text encoder model
    model_name: "bert_large_uncased"
    # Maximum Text Sequence Length
    max_seq_len: 512
    # Cache Directory
    cache_dir: "/Users/babacar/.cache/huggingface"

  # ----------------- Configuration for the Adaptive Layer ----------------- #
  adaptive_layer:
    # Input dimension of the text encoder (1024 for bert, else 4096)
    in_dim: 1024
    # Output dimension of the text encoder (1024 for bert, else 4096)
    out_dim: 1024
    # Number of attention heads in the text encoder
    num_heads: 8
    # Dropout probability in the text encoder
    dropout: 0.1

# ----------------------- Configuration for CAD Decoder ---------------------- #
cad_decoder:
  # Dimension of the latent variable z in the CAD decoder (1024 for bert, else 4096)
  tdim: 1024
  # Dimension of the state variable s in the CAD decoder
  cdim: 256
  # Number of transformer layers in the CAD decoder
  num_layers: 8
  # Number of attention heads in each layer of the CAD decoder
  num_heads: 8
  # Dropout probability in the CAD decoder
  dropout: 0.1
  # Starting level for channel attention in the CAD decoder
  ca_level_start: 2


# --------------------- Configuration related to inference -------------------- #
test:
  # Batch size for inference
  batch_size: 1
  # Number of workers for the DataLoader during inference
  num_workers: 30
  # Prefetch factor for the DataLoader during inference
  prefetch_factor: 10
  # Directory for logging inference information
  log_dir: "logs"
  # Path to saved model checkpoint (optional)
  checkpoint_path: "../Cad_VLM/checkpoints/Text2CAD_1.0.pth"
  nucleus_prob: 0
  sampling_type: "max"
  prompt_file: "YOUR PROMPT FILE"

# ------------------------------ Debug mode flag ----------------------------- #
debug: False

# --------------- Additional information (leave empty for now) --------------- #
info: "Inference"


================================================
File: Cad_VLM/config/trainer.yaml
================================================
 # ---------------------------------------------------------------------------- #
 #                      Config for Text2CAD Model Training                      #
 # ---------------------------------------------------------------------------- #
 
 
 
 # ---------------------- Configuration for text encoder ---------------------- #
text_encoder:
  # ----------------- Configuration for the Base Text Embedder ----------------- #
  text_embedder:
    # Name of the text encoder model
    model_name: "bert_large_uncased"
    # Maximum Text Sequence Length
    max_seq_len: 512
    # Cache Directory
    cache_dir: "YOUR DIR"

  # ----------------- Configuration for the Adaptive Layer ----------------- #
  adaptive_layer:
    # Input dimension of the text encoder (1024 for bert, else 4096)
    in_dim: 1024 
    # Output dimension of the text encoder (1024 for bert, else 4096)
    out_dim: 1024
    # Number of attention heads in the text encoder
    num_heads: 8
    # Dropout probability in the text encoder
    dropout: 0.1

# ----------------------- Configuration for CAD Decoder ---------------------- #
cad_decoder:
  # Dimension of the latent variable z in the CAD decoder (1024 for bert, else 4096)
  tdim: 1024
  # Dimension of the state variable s in the CAD decoder
  cdim: 256
  # Number of transformer layers in the CAD decoder
  num_layers: 8
  # Number of attention heads in each layer of the CAD decoder
  num_heads: 8
  # Dropout probability in the CAD decoder
  dropout: 0.1
  # Starting level for channel attention in the CAD decoder
  ca_level_start: 2

# --------------- Configuration related to training dataloader --------------- #
train_data:
  # Root directory of the training data
  cad_seq_dir: "YOUR DIR"
  # Path to the CSV file containing the text prompts
  prompt_path: "YOUR PATH"
  # JSON file containing information about train, test, and validation splits
  split_filepath: "YOUR PATH"
  # Maximum sequence length for input data
  max_seq_len: 512

# --------------------- Configuration related to training -------------------- #
train:
  # Learning rate for training
  lr: 0.0001
  # Batch size for training
  batch_size: 16 # for 80 GB gpu
  # Number of epochs for training
  num_epochs: 150
  # Number of workers for the DataLoader during training
  num_workers: 30
  # Prefetch factor for the DataLoader during training
  prefetch_factor: 10
  # Directory for logging training information
  log_dir: "YOUR DIR"
  # Path to saved model checkpoint for Resuming Training (optional)
  checkpoint_path:  # Set to None if no checkpoint is available
  # Checkpoint interval
  checkpoint_interval: 10
  # Curriculum learning epoch (set to 0 for no curriculum learning)
  curriculum_learning_epoch: 0
  

# ------------------------- Validation configuration ------------------------- #
val:
  # Nucleus sampling probability for validation (set to 0 for greedy decoding)
  nucleus_prob: 0
  val_batch: 5

# ------------------------------ Debug mode flag ----------------------------- #
# In debug mode, the model weights are not saved
debug: False

# --------------- Additional information (leave empty for now) --------------- #
info: "Experiment 1: Base Model Training"


================================================
File: Cad_VLM/dataprep/t2c_dataset.py
================================================
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import pandas as pd
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor


class Text2CAD_Dataset(Dataset):
    def __init__(
        self,
        cad_seq_dir: str,
        prompt_path: str,
        split_filepath: str,
        subset: str,
        max_workers: int,
    ):
        """
        Args:
            cad_seq_dir (string): Directory with all the .pth files.
            prompt_path (string): Directory with all the .npz files.
            split_filepath (string): Train_Test_Val json file path.
            subset (string): "train", "test" or "val"
        """
        super(Text2CAD_Dataset, self).__init__()
        self.cad_seq_dir = cad_seq_dir
        self.prompt_path = prompt_path
        self.prompt_df = pd.read_csv(prompt_path)
        self.prompt_df = self.prompt_df[
            self.prompt_df["abstract"].notnull()
            & self.prompt_df["beginner"].notnull()
            & self.prompt_df["intermediate"].notnull()
            & self.prompt_df["expert"].notnull()
        ]
        self.all_prompt_choices = ["abstract", "beginner", "intermediate", "expert"]
        self.substrings_to_remove = ["*", "\n", '"', "\_", "\\", "\t", "-", ":"]
        # open spilt json
        with open(os.path.join(split_filepath), "r") as f:
            self.split = json.load(f)

        self.uid_pair = self.split[subset]
        func = self._prepare_data

        # Load the prompt data using ThreadPoolExecutor and _prepare_data function
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary to store the prompt data
            self.prompt_data = {}
            # Use ThreadPoolExecutor to process the prompt data in parallel
            for data in tqdm(
                executor.map(func, self.uid_pair),
                total=len(self.uid_pair),
                desc=f"Loading {subset} split",
            ):
                if data is not None:
                    uid, cad_vec, prompt, mask_cad_dict = data
                    if isinstance(prompt, dict):
                        for key, val in prompt.items():
                            self.prompt_data[uid + f"_{key}"] = (
                                cad_vec,
                                val,
                                mask_cad_dict,
                            )  # "0000/00001234" -> "0000/00001234_beginner"

        self.keys = list(self.prompt_data.keys())
        print(f"Found {len(self.prompt_data)} samples for {subset} split.")

    def __len__(self):
        return len(self.keys)

    def _prepare_data(self, uid):
        root_id, chunk_id = uid.split("/")
        if len(self.prompt_df[self.prompt_df["uid"] == uid]) == 0:
            return None
        cad_vec_dict = torch.load(
            os.path.join(self.cad_seq_dir, root_id, chunk_id, "seq", f"{chunk_id}.pth"),
            weights_only=True,
        )
        level_data = {}
        for prompt_choice in self.all_prompt_choices:
            prompt = self.prompt_df[self.prompt_df["uid"] == uid][prompt_choice].iloc[0]
            if isinstance(prompt, str):
                level_data[prompt_choice] = prompt
        if len(level_data) == 0:
            return None
        # Filter the prompt
        prompt = self.remove_substrings(prompt, self.substrings_to_remove).lower()
        return uid, cad_vec_dict["vec"], level_data, cad_vec_dict["mask_cad_dict"]

    def remove_substrings(self, text, substrings):
        """
        Remove specified substrings from the input text.

        Args:
            text (str): The input text to be cleaned.
            substrings (list): A list of substrings to be removed.

        Returns:
            str: The cleaned text with specified substrings removed.
        """
        # Escape special characters in substrings and join them to form the regex pattern
        regex_pattern = "|".join(re.escape(substring) for substring in substrings)
        # Use re.sub to replace occurrences of any substrings with an empty string
        cleaned_text = re.sub(regex_pattern, " ", text)
        # Remove extra white spaces
        cleaned_text = re.sub(" +", " ", cleaned_text)
        return cleaned_text

    def __getitem__(self, idx):
        uid = self.keys[idx]
        return uid, *self.prompt_data[uid]


def get_dataloaders(
    cad_seq_dir: str,
    prompt_path: str,
    split_filepath: str,
    subsets: list[str],
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int,
    prefetch_factor: int,
):
    """
    Generate a DataLoader for the Text2CADDataset.

    Args:
    - cad_seq_dir (str): The directory containing the CAD sequence files.
    - prompt_path (str): The path to the CSV file containing the prompts.
    - split_filepath (str): The path to the JSON file containing the train/test/validation split.
    - subsets (list[str]): The subset to use ("train", "test", or "val").
    - batch_size (int): The batch size.
    - shuffle (bool): Whether to shuffle the data.
    - pin_memory (bool): Whether to pin memory.
    - num_workers (int): The number of workers.
    - prefetch_factor (int): The prefetch factor.

    Returns:
    - dataloader (torch.utils.data.DataLoader): The DataLoader object.
    """

    all_dataloaders = []

    for subset in subsets:
        # Create an instance of the Text2CADDataset
        dataset = Text2CAD_Dataset(
            cad_seq_dir=cad_seq_dir,
            prompt_path=prompt_path,
            split_filepath=split_filepath,
            subset=subset,
            max_workers=num_workers,
        )

        # Create a DataLoader with the specified parameters
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,  # You can set this to True if you want to shuffle the data
            num_workers=num_workers,
            pin_memory=pin_memory,  # Set to True if using CUDA
            prefetch_factor=prefetch_factor,
        )
        all_dataloaders.append(dataloader)

    return all_dataloaders


if __name__ == "__main__":
    pass


================================================
File: Cad_VLM/models/decoder.py
================================================
import torch
import copy
import torch.nn as nn
import os, sys

sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import (
    generate_attention_mask,
    create_flag_vec,
    create_index_vec,
    top_p_sampling,
)
from Cad_VLM.models.layers.embedder import CADSequenceEmbedder, PositionalEncodingSinCos
from Cad_VLM.models.layers.attention import CrossAttention, MultiHeadAttention
from Cad_VLM.models.layers.functional import FeedForwardLayer
from Cad_VLM.models.utils import count_parameters, get_device, get_device_str
from rich import print
from typing import Optional


class CADDecoder(nn.Module):
    """
    Decodes the discrete latent codes into CAD Sequence autoregressively.

    Args:
        cad_class_info (dict, required): A dictionary containing information about CAD classes. Check `DataProc.CadSeqProc.utility.macro`
        tdim (int, required): Dimensionality of the continuous latent space.
        cdim (int, required): Dimensionality of the cad embedding.
        num_layers (int, required): Number of transformer layers.
        num_heads (int, required): Number of attention heads in the transformer.
        dropout (float, required): Dropout probability.
        mode (str, required): Either "train" or "test" for specifying the mode of operation.
        ca_level_start (int, required): The starting level of class attention layers.
        device (str, required): Device to run the model on, e.g., "cuda" or "cpu".

    Example:
       To initialize the CADDecoder, you can use the following example:

        ```python
        decoder = CADDecoder(
            cad_class_info={"one_hot_size": 267, "index_size": 11, "flag_size": 12}, tdim=128, cdim=128, num_layers=8,
            num_heads=8, dropout=0.1, mode="train", ca_level_start=0, device="cuda",
        )
        ```

    """

    def __init__(
        self,
        cad_class_info: dict,
        tdim: int,
        cdim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ca_level_start: int,
        device: str,
    ):
        super(CADDecoder, self).__init__()

        self.num_layers = num_layers

        # For Initial Sequence Embedding
        self.seq_embed = CADSequenceEmbedder(
            one_hot_size=cad_class_info["one_hot_size"],
            flag_size=cad_class_info["flag_size"],
            index_size=cad_class_info["index_size"],
            d_model=cdim,
            device=device,
        )

        # Positional Encoding
        self.pe = PositionalEncodingSinCos(
            embedding_size=cdim, max_seq_len=MAX_CAD_SEQUENCE_LENGTH, device=device
        )

        # List of booleans indicating which decoder layers support cross-attention
        self.use_ca = [False] * ca_level_start + [True] * (num_heads - ca_level_start)

        # A stack of num_layers x Decoder Layer
        self.cad_decoder_layers = nn.ModuleList(
            [
                copy.deepcopy(
                    CADDecoderLayer(
                        tdim=tdim,
                        cdim=cdim,
                        num_heads=num_heads,
                        block_level=i,
                        dropout=dropout,
                        use_ca=self.use_ca[i],
                        device=device,
                    )
                )
                for i in range(self.num_layers)
            ]
        )

        # Out Layer for Sequence Prediction
        self.seq_output_x = nn.Sequential(
            nn.Linear(in_features=cdim, out_features=cad_class_info["one_hot_size"])
        )
        self.seq_output_y = nn.Sequential(
            nn.Linear(in_features=cdim, out_features=cad_class_info["one_hot_size"])
        )

        # Metadata
        self.attention_scores = dict()

    def forward(
        self,
        vec_dict: dict,
        ZE: Optional[torch.tensor],
        mask_cad_dict: dict,
        cross_attn_mask_dict: dict = {"attn_mask": None, "key_padding_mask": None},
        metadata: bool = False,
    ):
        """
        vec_dict: dictionary with keys "cad_vec", "flag_vec", "index_vec"
        ZE: tensor of shape (batch, num_code, emd_dim). Context
        mask_cad_dict: dictionary with keys "key_padding_mask" and "attn_mask". SELF ATTENTION MASK
        cross_attn_mask_dict: dictionary with keys "attn_mask" and "key_padding_mask". CROSS ATTENTION MASK
        metadata: boolean indicating whether attention weights are saved. Turn off during training
        """
        num_seq = vec_dict["cad_vec"].shape[1]
        # Token Embedding and positional encoding
        S = self.pe(num_seq) + self.seq_embed(
            vec_dict, mask_cad_dict["key_padding_mask"]
        )  # (B,N1,cdim)
        # Pass through Decoder Layers
        for i in range(self.num_layers):
            S, self.attention_scores[f"block_level_{i}"] = self.cad_decoder_layers[i](
                S,
                ZE=ZE,
                mask_cad_dict=mask_cad_dict,
                cross_attn_mask_dict=cross_attn_mask_dict,
                metadata=metadata,
            )
        Sx = self.seq_output_x(S).unsqueeze(dim=2)  # (B,1,N1,one_hot_size)
        Sy = self.seq_output_y(S).unsqueeze(dim=2)  # (B,N1,1,one_hot_size)

        S = torch.cat([Sx, Sy], dim=2)  # (B,N1,2*one_hot_size

        # Save the metadata
        if metadata:
            self.metadata = {"attention_scores": self.attention_scores}
            return S, self.metadata
        else:
            return S, None

    @staticmethod
    def from_config(config):
        """Initialize the CADDecoder class from a config file"""
        device = get_device()
        cad_decoder = CADDecoder(
            cad_class_info=CAD_CLASS_INFO,
            tdim=config["tdim"],
            cdim=config["cdim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            ca_level_start=config["ca_level_start"],
            device=device,
        )
        return cad_decoder

    def decode(
        self,
        ZE,
        cross_attn_mask_dict,
        maxlen,
        nucleus_prob,
        topk_index,
        device,
    ):
        """
        Autoregressive Prediction for test and validation.
        :param new_cad_seq_dict: dictionary containing keys "cad_vec", "flag_vec", "index_vec". cad_vec shape (batch, 1, 2) and flag and index vec (batch,1,1)
        :param ZE: tensor of shape (batch, num_latent, emb_dim)
        :param cross_attn_mask_dict: dictionary containing keys "attn_mask", "key_padding_mask". Keep both None.
        :maxlen maximum length of sequence.
        :nucleus_prob: Probability for Nucleus sampling. If 0, then top1-sampling.
                        More probability for more diversity but less valid CAD models and less probability for less diversity but more correct CAD models.
        :device: "cuda" or "cpu"
        """
        self.eval()
        num_texts = ZE.shape[0]
        device = get_device()  # Use our device function
        new_cad_seq_dict={
            "cad_vec": torch.tensor([[[1, 0]]]).repeat(num_texts, 1, 1).to(device),
            "flag_vec": torch.zeros(num_texts, 1).int().to(device),
            "index_vec": torch.zeros(num_texts, 1).int().to(device),
        }
        
        # NOTE: Iteratively run the forward method till the end token is predicted.
        for t in range(1, maxlen):
            cad_mask = {
                "attn_mask": cross_attn_mask_dict["attn_mask"].repeat(1, t, 1),
                "key_padding_mask": cross_attn_mask_dict["key_padding_mask"],
            }
            cad_pred, _ = self(
                new_cad_seq_dict,
                ZE,
                {
                    "attn_mask": generate_attention_mask(t, t, device=device),
                    "key_padding_mask": (new_cad_seq_dict["cad_vec"] == 0),
                },
                cad_mask,
                False,
            )

            # --------------------------------- Sampling --------------------------------- #
            # Hybrid-Sampling
            if nucleus_prob == 0:
                if t == 1:  # NOTE: Remove this part for top-1 sampling
                    new_token = torch.topk(cad_pred, topk_index, dim=-1).indices[
                        :, t - 1 : t, :, -1
                    ]
                else:
                    # NOTE: Keep this part only for top-1 sampling
                    new_token = torch.argmax(cad_pred, dim=-1)[:, t - 1 : t]
            # Nucleus Sampling
            else:
                new_token = torch.cat(
                    [
                        top_p_sampling(cad_pred[:, t - 1 : t, 0], nucleus_prob),
                        top_p_sampling(cad_pred[:, t - 1 : t, 1], nucleus_prob),
                    ],
                    axis=-1,
                )

            # ------------------------------ CAD Sequence Update ------------------------------ #
            # Add the new token (no masking here)
            new_cad_seq_dict["cad_vec"] = torch.cat(
                [new_cad_seq_dict["cad_vec"], new_token], axis=1
            )

            # ------------------------------ Flag generation ----------------------------- #
            # Create flag seq (Very important. Wrong flag may result in invalid model)
            new_cad_seq_dict["flag_vec"] = torch.cat(
                [
                    new_cad_seq_dict["flag_vec"],
                    create_flag_vec(
                        new_cad_seq_dict["cad_vec"], new_cad_seq_dict["flag_vec"]
                    ),
                ],
                axis=1,
            )

            # ----------------------------- Index Generation ----------------------------- #
            # Create index seq  (Very important. Wrong index may result in invalid model)
            new_cad_seq_dict["index_vec"] = torch.cat(
                [
                    new_cad_seq_dict["index_vec"],
                    create_index_vec(
                        new_cad_seq_dict["cad_vec"], new_cad_seq_dict["index_vec"]
                    ),
                ],
                axis=1,
            )

            # ------------------------- Masking the dummy tokens ------------------------- #
            # Mask the dummy tokens in the new CAD tokens (Very important. Wrong masking may result in inaccurate model)

            end_tokens=torch.logical_or(new_cad_seq_dict['cad_vec'][:,:,0] <= END_TOKEN.index("END_EXTRUSION"),new_cad_seq_dict['flag_vec']>0)
    

            num_tokens=new_cad_seq_dict["cad_vec"][
                end_tokens
            ].shape[0]

            mask = torch.cat(
                [
                    torch.ones((num_tokens, 1), dtype=torch.int32),
                    torch.zeros((num_tokens, 1), dtype=torch.int32),
                ],
                axis=1,
            ).to(device)
            
            new_cad_seq_dict["cad_vec"][
                end_tokens
            ] *= mask

        return new_cad_seq_dict

    def total_parameters(self, description=False, in_millions=False):
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000  # Convert to millions
            print(f"Number of Parameters: {num_params_million:.1f}M")
        else:
            num_params = count_parameters(self, description)
            print(f"Number of Parameters: {num_params}")


class CADDecoderLayer(nn.Module):
    """
    CAD Decoder Layers

    """

    def __init__(
        self,
        tdim=128,
        cdim=128,
        num_heads=8,
        block_level=0,
        dropout=0.1,
        use_ca=True,
        device="cuda",
    ):
        super(CADDecoderLayer, self).__init__()

        # Multi-Head Self Attention for CAD Sequence
        # TODO: Check if Flash Attention is implemented
        #! Note: Dropout is set to 0 for attention otherwise the sum of attention weights > 1
        self.sa_seq = MultiHeadAttention(
            input_dim=cdim,
            embed_dim=cdim,
            dropout=0,
            num_heads=num_heads,
        )
        self.use_ca = use_ca

        if use_ca:
            # Cross Attention between discrete latent vectors and cad features
            # TODO: Implement Flash attention here
            self.ca_ze_seq = CrossAttention(
                input_dim_list=[cdim, tdim],
                output_dim=cdim,
                query_name="cad",
                context_1_name="vq",
                dropout=0,
                block_level=block_level,
            )

        # Text Embedding Downsampler
        self.downsampler = nn.Linear(tdim, cdim)

        # LayerNormalization
        self.norm_seq = nn.ModuleDict(
            {
                "norm_1": nn.LayerNorm(cdim),
                "norm_2": nn.LayerNorm(cdim),
                "norm_3": nn.LayerNorm(cdim),
            }
        )

        # Dropout
        self.dp_seq = nn.ModuleDict(
            {
                "dropout_1": nn.Dropout(dropout),
                "dropout_2": nn.Dropout(dropout),
                "dropout_3": nn.Dropout(dropout),
            }
        )

        # Feed forward Networks
        self.ffl_seq = FeedForwardLayer(input_dim=cdim)

        # Attention Scores
        self.attention_scores = dict()

    def forward(
        self,
        S: Optional[torch.Tensor],
        ZE: Optional[torch.Tensor],
        mask_cad_dict: dict,
        cross_attn_mask_dict: dict = {"attn_mask": None, "key_padding_mask": None},
        metadata: bool = False,
    ):
        """
        S: tensor of shape (bs, num_seq, emb_dim)
        ZE: tensor of shape (bs, num_code, emd_dim)
        mask_cad_dict: dictionary with keys "attn_mask", "key_padding_mask"
        cross_attn_mask_dict: dictionary with keys "attn_mask", "key_padding_mask"
        metadata: boolean. To save attention weights
        """

        self_attn_mask_dict = mask_cad_dict.copy()
        self_attn_mask_dict["key_padding_mask"] = torch.all(
            mask_cad_dict["key_padding_mask"], axis=2
        )

        # ? <----------  CAD SEQUENCE SELF-ATTENTION  ---------->
        S2 = self.norm_seq["norm_1"](S)  # (bs,num_seq,emb_dim)
        # exit()
        S2, S_score = self.sa_seq(
            S2,
            S2,
            S2,
            key_padding_mask=self_attn_mask_dict["key_padding_mask"],
            attn_mask=self_attn_mask_dict["attn_mask"],
        )  # (bs,num_seq,emb_dim) (Self-Attention)

        # (bs,num_seq,emb_dim) (Dropout + Addition)
        S = S + self.dp_seq["dropout_1"](S2)
        S2 = self.norm_seq["norm_2"](S)  # (bs,num_seq,emb_dim) (Normalization)

        # ? <----------  CROSS-ATTENTION BETWEEN LATENT EMBEDDING AND CAD EMBEDDING  ---------->
        if self.use_ca:
            if hasattr(self, "downsampler"):
                ZE = self.downsampler(ZE)
            S3, ZE_S_score = self.ca_ze_seq(
                S2,
                ZE,
                ZE,
                key_padding_mask=cross_attn_mask_dict["key_padding_mask"],
                attn_mask=cross_attn_mask_dict["attn_mask"],
            )

            # ? <----------  (CROSS-ATTENDED FEATURES + SELF-ATTENDED FEATURES) + DROPOUT + NORMALIZATION LAYER  ---------->
            S = S + self.dp_seq["dropout_2"](S3)
            S2 = self.norm_seq["norm_3"](S)  # (bs,num_seq,emb_dim)
            self.attention_scores["ca"] = ZE_S_score

        # ? <---------- FEED-FORWARD + DROPOUT + ADDITION +    ---------->
        S = S + self.dp_seq["dropout_3"](self.ffl_seq(S2))

        # Add the cross attention scores (metadata)
        self.attention_scores["sa"] = S_score

        return S, self.attention_scores


if __name__ == "__main__":
    pass
    

================================================
File: Cad_VLM/models/loss.py
================================================
import torch
import torch.nn as nn


class CELoss(nn.Module):
    """
    Cross Entropy Loss for Text2CAD
    """

    def __init__(self, device):
        super(CELoss, self).__init__()

        self.ce_cad = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)
        self.ce_pc = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss()

    def forward(self, cad_dict: dict):
        """
        cad_dict: dictionary containing 'pred', 'target' and 'key_padding_mask' key.
                pred: shape (B,N,2)
                target: shape (B,N)
                key_padding_mask: shape (B,N)
        """

        key_padding_mask = cad_dict["key_padding_mask"]
        loss = []
        if cad_dict["key_padding_mask"] is not None:
            self.loss_seq_x = torch.sum(
                self.ce_cad(
                    cad_dict["pred"][:, :, 0].permute(0, 2, 1),
                    cad_dict["target"][:, :, 0].long(),
                )
                * key_padding_mask[:, :, 0]
            ) / torch.sum(key_padding_mask[:, :, 0] * 1)
            self.loss_seq_y = torch.sum(
                self.ce_cad(
                    cad_dict["pred"][:, :, 1].permute(0, 2, 1),
                    cad_dict["target"][:, :, 1].long(),
                )
                * key_padding_mask[:, :, 1]
            ) / torch.sum(key_padding_mask[:, :, 1] * 1)

        self.loss_seq = (self.loss_seq_x + self.loss_seq_y) / 2
        loss_keys = ["loss_seq"]

        result_dict = {key: getattr(self, key).detach().item() for key in loss_keys}

        loss = self.loss_seq
        return loss, result_dict


================================================
File: Cad_VLM/models/metrics.py
================================================
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch

class AccuracyCalculator:
    def __init__(self,tolerance=3,discard_token=6,device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tol=tolerance
        self.discard_token=discard_token
        self.device=device
    
    def calculateAccMulti2DFromProbability(self,predProb,targetLabel):
        # Get the predicted classes
        predLabel=predProb.argmax(dim=-1)

        return self.calculateAccMulti2DFromLabel(predLabel,targetLabel)
    
    def calculateAccMulti2DFromLabel(self,predLabel,targetLabel):
        """
        predLabel: tensor of shape (B, N, 2)
        target: tensor of shape (B, N, 2)
        """
        if predLabel.shape[1]>targetLabel.shape[1]:
            predLabel=predLabel[:,:targetLabel.shape[1]]
        if targetLabel.shape[1]>predLabel.shape[1]:
            targetLabel=targetLabel[:,:predLabel.shape[1]]

        mask=(targetLabel>self.discard_token).any(axis=-1)
        mask=mask.to(targetLabel.device)
        mask=mask.unsqueeze(dim=-1).repeat(1,1,2)

        return self.calculateAccMultiFromLabel(predLabel=predLabel,targetLabel=targetLabel,mask=mask)

    def calculateAccMultiFromLabel(self,predLabel,targetLabel,mask=None):
        """
        pred: tensor of shape (B, N, 2)
        target: tensor of shape (B, N)
        """
        
        N_pred=predLabel.shape[1]
        N_gt=targetLabel.shape[1]
        N_min=min(N_pred,N_gt)
        predLabel=predLabel[:,:N_pred]
        targetLabel=targetLabel[:,:N_pred]

        if mask is None:
            mask=targetLabel>self.discard_token
            mask=mask.to(targetLabel.device)

        # Calculate the number of correct predictions (removing the padding)
        correct = (torch.abs(predLabel-targetLabel)<self.tol)*1*mask
        
        correct=correct.sum()

        # Calculate the total number of predictions excluding the discard token
        total = torch.sum(mask)

        # Calculate the accuracy
        accuracy = float(correct) / float(total)

        return accuracy

if __name__=="__main__":
    pass

================================================
File: Cad_VLM/models/text2cad.py
================================================
import os, sys

sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import torch.nn as nn
import torch
from Cad_VLM.models.layers.adaptive_layer import TextAdaptiveLayer
from Cad_VLM.models.layers.text_embed import TextEmbedder, prepare_cross_attention_mask_batch
from Cad_VLM.models.decoder import CADDecoder
from Cad_VLM.models.utils import count_parameters



class Text2CAD(nn.Module):
    """
    Text2CAD: Generating CAD Designs from beginner-to-expert text prompts

    """

    def __init__(self, text_config, cad_config):
        super().__init__()
        
        # Base Text Embedder
        self.base_text_embedder = TextEmbedder.from_config(text_config['text_embedder'])
        # Adaptive Layer to fine tune text embeddings
        self.adaptive_layer = TextAdaptiveLayer.from_config(text_config['adaptive_layer'])

        # Transformer Decoder for CAD sequence generation
        self.cad_decoder = CADDecoder.from_config(cad_config)
        
        self.cad_seq_len= cad_config["cad_seq_len"]-1

        self.attention_scores = dict()

    def test_decode(self, 
                        texts: list[str],
                        maxlen:int,
                        nucleus_prob,
                        topk_index,
                        device='cuda' if torch .cuda.is_available() else 'cpu'
                        ):
        """
        Auto-regressively decode CAD sequence from text prompts
        Args:
        - texts: list of text prompts
        - maxlen: maximum length of the generated CAD sequence
        - nucleus_prob: nucleus sampling probability
        - topk_index: top-k sampling index
        - device: device to run the model
        """
        
        ZE, key_padding_mask = self.base_text_embedder.get_embedding(texts)
        ca_mask={"attn_mask": prepare_cross_attention_mask_batch(key_padding_mask, cad_seq_len=1), 
                 "key_padding_mask": key_padding_mask}
        ZE, _  = self.adaptive_layer(ZE,
            {
                "attn_mask": None,
                "key_padding_mask": ca_mask["key_padding_mask"],
            },
            False,)
        S_output = self.cad_decoder.decode(
                        ZE=ZE, 
                        cross_attn_mask_dict=ca_mask, 
                        maxlen=maxlen, 
                        nucleus_prob=nucleus_prob,
                        topk_index=topk_index, 
                        device=device)
        
        return S_output

    def forward(
        self,
        vec_dict: dict,
        texts: list[str],
        mask_cad_dict: dict,
        metadata: bool = False,
    ):
        """
        vec_dict: dict contains cad_vec, flag_vec, and index_vec
        texts: list of text prompts
        mask_cad_dict: dict contains attention mask and key_padding_mask for CAD sequence
        metadata: bool to return attention scores
        """
        # ------------ Get the initial text embeddings ------------ #
        
        T, key_padding_mask = self.base_text_embedder.get_embedding(texts)
        ca_mask={"attn_mask": prepare_cross_attention_mask_batch(key_padding_mask, cad_seq_len=self.cad_seq_len), 
                 "key_padding_mask": key_padding_mask}
        
        # ------------ Pass the text embedding through the adaptive layer ------------ #
        T, text_attn_scores = self.adaptive_layer(
            T,
            {
                "attn_mask": None,
                "key_padding_mask": ca_mask["key_padding_mask"],
            },
            metadata,
        )
        if text_attn_scores is not None:
            self.attention_scores.update(text_attn_scores)

        # ------------ Pass the text embedding through the CAD Decoder as context ------------ #
        S_output, cad_attn_scores = self.cad_decoder(
            vec_dict, T, mask_cad_dict, ca_mask, metadata
        )
        if cad_attn_scores is not None:
            self.attention_scores.update(cad_attn_scores)

        if metadata:
            return S_output, self.attention_scores
        else:
            return S_output, None

    def total_parameters(self, description=False, in_millions=False):
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000  # Convert to millions
            print(f"Number of Parameters: {num_params_million:.1f}M")
        else:
            num_params = count_parameters(self, description)
            print(f"Number of Parameters: {num_params}")
        
    def get_trainable_state_dict(self):
        # Get the state dict of the model which are trainable parameters
        return {
            k: v for k, v in self.state_dict().items() if "base_text_embedder" not in k.split(".")
        }

================================================
File: Cad_VLM/models/utils.py
================================================
from prettytable import PrettyTable
import torch.nn as nn
import copy
import os
import re
import torch


def text_prompt(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # Use a regular expression to find all text within <prompt> tags
            prompts = re.findall(r'<prompt>(.*?)</prompt>', content, re.DOTALL)
            # Remove whitespace within each prompt
        return prompts
    except FileNotFoundError:
        print("File not found.")
        return []
    except Exception as e:
        print("An error occurred:", e)
        return []

# Calculate the total number of parameters
def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    return sents


def count_parameters(model, description=True):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if description:
        print(table)
    return total_params


def check_memory_usage(tensor):
    return tensor.element_size() * tensor.nelement() / 1024**2


def get_clones(module, num_layers=8):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])


def get_available_gpu_ids():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in cuda_visible_devices.split(",")]
    else:
        gpu_ids = []  # Empty list means no GPUs available for training.

    return gpu_ids


def print_with_separator(text):
    separator = "# ---------------------------------------------------------------------------- #"
    text_length = len(text)
    padding_total = 78 - 2 - text_length  # Total padding available for the inner line, subtracting 2 for the `#` characters
    left_padding = padding_total // 2
    right_padding = padding_total - left_padding

    print("\033[94m" + separator)
    print("#" + " " * left_padding + text + " " * right_padding + "#")
    print(separator + "\033[0m")


def get_device():
    """Get the appropriate device (MPS for Mac M-series, CUDA for NVIDIA, CPU otherwise)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_str():
    """Get the device string (mps, cuda, or cpu)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

================================================
File: Cad_VLM/models/layers/adaptive_layer.py
================================================
import torch
import copy
import torch.nn as nn
import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))


from Cad_VLM.models.layers.attention import MultiHeadAttention
from Cad_VLM.models.layers.functional import FeedForwardLayer
from rich import print
from Cad_VLM.models.layers.utils_decode import generate_attention_mask
from Cad_VLM.models.utils import count_parameters
from typing import Optional


class TextAdaptiveLayer(nn.Module):
    """
    Adaptive Layer for Text Embeddings

    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float):
        super(TextAdaptiveLayer, self).__init__()

        # Multi-Head Self Attention for CAD Sequence
        # TODO: Check if Flash Attention is implemented
        self.sa_seq = MultiHeadAttention(
            input_dim=in_dim,
            embed_dim=in_dim,
            dropout=dropout,
            num_heads=num_heads,
        )

        # LayerNormalization
        self.norm_seq = nn.ModuleDict(
            {"norm_1": nn.LayerNorm(in_dim), "norm_2": nn.LayerNorm(in_dim)}
        )

        # Dropout
        self.dp_seq = nn.ModuleDict(
            {"dropout_1": nn.Dropout(dropout), "dropout_2": nn.Dropout(dropout)}
        )

        # Feed forward Networks
        self.ffl_seq = FeedForwardLayer(input_dim=in_dim)

        if in_dim != out_dim:
            # Downsampler
            self.downsampler = nn.Linear(in_dim, out_dim)

        # Attention Scores
        self.attention_scores = dict()

    def forward(
        self,
        T: Optional[torch.Tensor],
        mask_prompt_dict: dict,
        metadata: bool = False,
    ):
        """
        T: tensor of shape (bs, num_seq, emb_dim). Text Embedding
        mask_prompt_dict: dictionary with keys "attn_mask", "key_padding_mask"
        metadata: boolean. To save attention weights
        """

        # self_attn_mask_dict = mask_cad_dict.copy()

        # ? <----------  TEXT EMBEDDING SELF-ATTENTION  ---------->
        T2 = self.norm_seq["norm_1"](T)  # (bs,num_seq,emb_dim)
        # exit()
        T2, T_score = self.sa_seq(
            T2,
            T2,
            T2,
            key_padding_mask=mask_prompt_dict["key_padding_mask"],
            attn_mask=mask_prompt_dict["attn_mask"],
        )  # (bs,num_seq,emb_dim) (Self-Attention)

        # (bs,num_seq,emb_dim) (Dropout + Addition)
        T = T + self.dp_seq["dropout_1"](T2)
        T2 = self.norm_seq["norm_2"](T)  # (bs,num_seq,emb_dim) (Normalization)

        # ? <---------- FEED-FORWARD + DROPOUT + ADDITION + DOWN-SAMPLER    ---------->
        T = T + self.dp_seq["dropout_2"](self.ffl_seq(T2))

        if hasattr(self, "downsampler"):
            T = self.downsampler(T)

        if metadata:
            # Add the cross attention scores (metadata)
            self.attention_scores["text_sattn"] = T_score
            return T, self.attention_scores
        else:
            return T, None

    def total_parameters(self, description=False, in_millions=False):
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000  # Convert to millions
            print(f"Number of Parameters: {num_params_million:.1f}M")
        else:
            num_params = count_parameters(self, description)
            print(f"Number of Parameters: {num_params}")

    @staticmethod
    def from_config(config):
        return TextAdaptiveLayer(**config)


if __name__ == "__main__":
    adaptive_layer = TextAdaptiveLayer(4096, 4096, 8, 0.1).cuda()
    input_tensor = torch.rand(32, 512, 4096).cuda()

    attn_mask = generate_attention_mask(512)

    output, attn_weight = adaptive_layer(
        input_tensor,
        {
            "attn_mask": None,
            "key_padding_mask": torch.randint(0, 2, (32, 512)).bool().cuda(),
        },
        metadata=True,
    )
    print(output.shape)

    print(attn_weight)
    print(adaptive_layer.total_parameters())


================================================
File: Cad_VLM/models/layers/attention.py
================================================
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from operator import itemgetter
from .layer_utils import *
from .functional import multi_head_attention_forward
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch


# TODO: Replace the MultiHeadAttention layer with the code written for CADLGen (Adding Flash and Memory efficient Attention)
class MultiHeadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        input_dim: Input dimension of the sequence
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    __annotations__ = {
        "bias_k": torch._jit_internal.Optional[torch.Tensor],
        "bias_v": torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = [
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "in_proj_weight",
    ]

    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiHeadAttention, self).__setstate__(state)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_3d=False,
    ):
        r"""
        Args:
            query: input Tensor of shape (batch_size,n_samples,n_features) for query generation
            key: input Tensor of shape (batch_size,n_samples,n_features) for key generation
            value: input Tensor of shape (batch_size,n_samples,n_features) for value generation
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        Shape:
            - Inputs:
            - query: :math:`(B, N, E)` where N is the target sequence length, B is the batch size, E is
              the embedding dimension.
            - key: :math:`(B, N, E)` where N is the target sequence length, B is the batch size, E is
            the embedding dimension.
            - vaue: :math:`(B, N, E)` where N is the target sequence length, B is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(B, S)`, ByteTensor, where B is the batch size, S is the source sequence length.
            - attn_mask: 2D mask :math:`(N, S)` where N is the target sequence length, S is the source sequence length.
              3D mask :math:`(B*num_heads, N, S)` where B is the batch size, N is the target sequence length,
              S is the source sequence length.
            - Outputs:
            - attn_output: :math:`(N, B, E)` where N is the target sequence length, B is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(B, N, S)` where B is the batch size,
              N is the target sequence length, S is the source sequence length.
        """

        query = query.transpose(0, 1)  # (N,B,E)
        key = key.transpose(0, 1)  # (N,B,E)
        value = value.transpose(0, 1)  # (N,B,E)

        # Atttention mask is same for all the batches
        if attn_mask is not None and attn_mask.dim() > 2 and not use_3d:
            attn_mask = attn_mask[0]  # (N,S)

        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )

class CrossAttention(nn.Module):
    """Cross Attention Layer for multi-modal Feature sharing

    Args:

    input_dim_list: List of dimenstion for input embeddings. The first index should be of the input used for query.
    output_dim: Output dimension. Int
    """

    def __init__(
        self,
        input_dim_list,
        output_dim=512,
        num_heads=1,
        query_name="pc",
        context_1_name="cad",
        dropout=0,
        block_level=1,
    ):
        super(CrossAttention, self).__init__()
        self.query_name = query_name
        self.context_1_name = context_1_name
        self.block_level = block_level

        self.mha = MultiHeadAttention(
            input_dim=input_dim_list[0],
            embed_dim=output_dim,
            dropout=dropout,
            num_heads=num_heads,
        )

    def forward(self, X, Y, Z,  attn_mask=None, key_padding_mask=None):
        """
        input:
        X: Input Embedding of shape. (B,N1,E)
        Y: Context Input Embedding of shape (B,N2,E)
        aggregate: "mean","max","sum"
        mask_dict: dictionary containing two keys "key_padding_mask" and "attn_mask"

        output:
        X: Input Embedding of shape. (B,N1,output_dim)
        cross_attention_scores: dict
        """
        attention_output, attention_scores = self.mha(
            X, Y, Z, attn_mask=attn_mask, use_3d=True
        )

        cross_attention_score = {
            f"{self.query_name}_{self.context_1_name}_{self.block_level}": attention_scores
        }

        return attention_output, cross_attention_score



================================================
File: Cad_VLM/models/layers/decorator.py
================================================
from functools import wraps
import tracemalloc
import time
import gc
import torch
import gc
import datetime
from contextlib import ContextDecorator
from rich import print
# <---------------- Custom Decorators ---------------->

"""
def my_decorator_func(func):

    def wrapper_func():
        # Do something before the function.
        func()
        # Do something after the function.
        # May return the result of the func()
    return wrapper_func
"""


def convert_seconds_to_minutes_and_hours(seconds):
    """
    Converts seconds to minutes and hours.

    Args:
        seconds (int): The number of seconds.

    Returns:
        tuple: The minutes and hours.

    """

    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60

    if minutes==0:
        if seconds<0.1:
            return f"{seconds*1000} ms"
        return f"{seconds} seconds"
    elif hours==0:
        return f"{minutes} minutes"
    else:
        return f"{hours} hours {minutes} minutes"

def timeit(func):
    # Decorator for calculating time
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {convert_seconds_to_minutes_and_hours(total_time)}')
        return result
    return timeit_wrapper


def log_datetime(func):
    """Log the date and time of a function"""
    @wraps(func)
    def log_datetime_wrapper(*args,**kwargs):
        startInfo=f'Function: {func.__name__} \nRun on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        result=func(*args,**kwargs)
        print(startInfo)
        return result
        print(f'{"-"*30}')
    return log_datetime_wrapper


def measure_performance(func):
    '''Measure performance of a function'''

    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()
        result=func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        finish_time = time.perf_counter()
        print(f'{"-"*40}')
        print(f'Function: {func.__name__}')
        print(f'Method: {func.__doc__}')
        print(f'Memory usage:\t\t {current / 10**6:.6f} MB \n'
              f'Peak memory usage:\t {peak / 10**6:.6f} MB ')
        print(f'Time elapsed : {convert_seconds_to_minutes_and_hours(finish_time - start_time)}')
        print(f'{"-"*40}')
        tracemalloc.stop()
        return result
    return wrapper



def gpu_memory_usage(func):
    """
    Decorator that prints the GPU memory usage before and after a function is called.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    """

    def wrapper(*args, **kwargs):
        start_memory = torch.cuda.memory_allocated()
        func(*args, **kwargs)
        end_memory = torch.cuda.memory_allocated()
        print("GPU memory usage: " + str(end_memory - start_memory))

    return wrapper



# Code from https://gist.github.com/MarkTension/4783697ebd5212ba500cdd829b364338
# pytorch method to find number of tensors in the graph
def get_n_tensors():
    tensors= []
    for obj in gc.get_objects():
        try:
            if (torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))):
                tensors.append(obj)
        except:
            pass
        return len(tensors)
  
# this is our context decorator
class check_memory_leak_context(ContextDecorator):
    def __enter__(self):
        self.start = get_n_tensors()
        return self
    def __exit__(self, *exc):
        self.end = get_n_tensors()
        increase = self.end - self.start
        
        if increase > 0:
                print(f"num tensors increased with"\
                    f"{self.end - self.start} !")
        else:
                print("no added tensors")
        return False


================================================
File: Cad_VLM/models/layers/embedder.py
================================================
from loguru import logger
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class PositionalEncodingSinCos(nn.Module):
    def __init__(self, embedding_size: int, max_seq_len: int, device: str):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len

        # create a matrix of shape (max_seq_len, embedding_size/2)
        self.pos_enc = torch.zeros(max_seq_len, embedding_size)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size)
        )
        self.pos_enc[:, 0::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)

        self.pos_enc = self.pos_enc.unsqueeze(0)

    def forward(self, seq_len):
        # x has shape (batch_size, seq_len, embedding_size)
        # add positional encoding to x
        x = self.pos_enc[:, :seq_len, :]
        return x.to(self.device)


class PositionalEncodingLUT(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingLUT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer("position", position)

        self.pos_embed = nn.Embedding(max_len, d_model)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[: x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class CADSequenceEmbedder(nn.Module):
    """
    CAD Sequence token embedding.

    parameters:

    one_hot_size: int. The dimension of one-hot embedding (default 4102).
    flag_size: Type of token
    index_size: Model index. Maximum is MAX_EXTRUSION
    d_model: Embedding dimension
    """

    def __init__(self, one_hot_size, flag_size, index_size, d_model, device="cpu"):
        super(CADSequenceEmbedder, self).__init__()

        # self.pe = PositionalEncodingSinCos(d_model,MAX_CAD_SEQUENCE_LENGTH, device=device)  # Positional encoding
        self.si = Embedder(index_size, d_model)  # CAD Sequence Index Encoding
        self.sf = Embedder(flag_size, d_model)  # CAD Sequence Flag Encoding
        self.cx = Embedder(one_hot_size, d_model)  # x-coordinate embedding
        self.cy = Embedder(one_hot_size, d_model)  # y-coordinate embedding
        self.device = torch.device(device)

    def forward(self, vec_dict, key_padding_mask):
        """
        vec_dict: contains key "cad_vec","flag_vec" and "index_vec"
        key_padding_mask: Tensor. Shape (N,2). Must be same with cad_vec
        """
        num_seq = vec_dict["cad_vec"].shape[1]  # Number of tokens
        x_seq = vec_dict["cad_vec"][:, :, 0] * (~key_padding_mask[:, :, 0] * 1)
        y_seq = vec_dict["cad_vec"][:, :, 1] * (~key_padding_mask[:, :, 1] * 1)

        return (
            self.sf(vec_dict["flag_vec"])
            + self.si(vec_dict["index_vec"])
            + self.cx(x_seq)
            + self.cy(y_seq)
        )


class VectorQuantizerEMA(nn.Module):
    """
    Code from SkexGen: Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks.
    https://github.com/samxuxiang/SkexGen
    """

    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        seqlen, bs = inputs.shape[0], inputs.shape[1]

        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).reshape(
            seqlen, bs, self._embedding_dim
        )

        encodings_flat = encodings.reshape(inputs.shape[0], inputs.shape[1], -1)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), encodings_flat, encoding_indices


if __name__ == "__main__":
    pass


================================================
File: Cad_VLM/models/layers/functional.py
================================================
from __future__ import division
from typing import Optional, Tuple
from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn


def multi_head_attention_forward(query,                           # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 # type: Optional[Tensor]
                                 bias_k,
                                 # type: Optional[Tensor]
                                 bias_v,
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 # type: Optional[Tensor]
                                 key_padding_mask=None,
                                 need_weights=True,               # type: bool
                                 # type: Optional[Tensor]
                                 attn_mask=None,
                                 use_separate_proj_weight=False,  # type: bool
                                 # type: Optional[Tensor]
                                 q_proj_weight=None,
                                 # type: Optional[Tensor]
                                 k_proj_weight=None,
                                 # type: Optional[Tensor]
                                 v_proj_weight=None,
                                 # type: Optional[Tensor]
                                 static_k=None,
                                 # type: Optional[Tensor]
                                 static_v=None
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check, f"{embed_dim} is not same as {embed_dim_to_check}"
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight,
                               in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt,
                         in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt,
                         in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt,
                         in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError(
                    f'The size of the 2D attn_mask is not correct. Size is {attn_mask.shape}. Query size is {query.shape}, Key size is {key.shape}')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError(f'The size of the 3D attn_mask is not correct. Expected ({bsz * num_heads, query.size(0), key.size(0)}) but got {attn_mask.size()}')
        else:
            raise RuntimeError(
                "attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len, f"key padding mask {key_padding_mask.size(1)} size mismatch with src length {src_len}"

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()
                      [2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()
                      [2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [
        bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    
    #! Note: Adding dropout here changes the sum of the attention weights to 1.11, so force it 0.
    
    attn_output_weights = F.dropout(
        attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(
        0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len, src_len)
        return attn_output.transpose(0, 1), attn_output_weights
    else:
        return attn_output.transpose(0, 1), None


class FeedForwardLayer(nn.Module):
    """ Feed forward network for Transformer
    Args:
        input_dim: Input Embedding Dimension
        d_ff: Feedforward dimension. Default=2048
        dropout: float
    """

    def __init__(self, input_dim, d_ff=2048, dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear_1 = nn.Linear(input_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, input_dim)

    def forward(self, x):
        """
        input:
            x: tensors of shape (B,N,input_dim)
        output:
            x: tensors of shape (B,N,input_dim)
        """
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x



================================================
File: Cad_VLM/models/layers/layer_utils.py
================================================
import torch

def perform_aggregate(X, Y, type):
    """
    X: Shape (B,N,D)
    Y: Shape (B,N,D)
    """
    if type == "sum":
        return X+Y
    elif type == "mean":
        return 0.5*(X+Y)
    elif type == "max":
        return torch.maximum(X, Y)


================================================
File: Cad_VLM/models/layers/text_embed.py
================================================
import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel
from ..utils import get_device, get_device_str


MODEL_NAME_DICT={"bert_large_uncased":"google-bert/bert-large-uncased"}

def prepare_cross_attention_mask_batch(mask, cad_seq_len=271):
    if mask.shape[0] > 1:
        length=mask.shape[1]
        batch_size=mask.shape[0]
        mask = mask.reshape(batch_size, 1, length)
    mask = torch.tile(mask, (1, cad_seq_len, 1))  # (512) -> (271, 512)
    mask = torch.where(
        mask, -torch.inf, 0
    )  # Changing the [True,False] format to [0,-inf] format

    return mask

class TextEmbedder(nn.Module):
    def __init__(self, model_name:str, cache_dir:str, max_seq_len:int):
        super(TextEmbedder, self).__init__()
        
        self.device = get_device()
        self.max_seq_len = max_seq_len
        self.model_name = MODEL_NAME_DICT.get(model_name, "bert_large_uncased")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.model = BertModel.from_pretrained(
                self.model_name, cache_dir=cache_dir, max_position_embeddings=max_seq_len
            ).to(self.device)
    
    def get_embedding(self, texts:list[str]):
        if isinstance(texts, str):
            texts = [texts]
        with torch.no_grad():
                input_ids = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    max_length=self.max_seq_len,
                    truncation=True,
                    padding=True,
                ).to(self.device)
                all_output = self.model(**input_ids)

                embedding = all_output[0]
                key_padding_mask = (
                    (input_ids["attention_mask"] == 0)
                )
                
        return embedding, key_padding_mask

    @staticmethod
    def from_config(config: dict):
        return TextEmbedder(
            **config
        )
        


================================================
File: Evaluation/eval_seq.py
================================================
import pandas as pd
import pickle
import os,sys
import argparse
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from tqdm import tqdm
from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.utils import (create_path_with_time,ensure_dir)
from CadSeqProc.utility.logger import CLGLogger
from tqdm import tqdm
import traceback
from rich import print
import json

csnLogger=CLGLogger().configure_logger().logger


def main():
    parser=argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input_path",help="Predicted CAD Sequence in pkl format",required=True)
    parser.add_argument("--output_dir",help="Output dir",required=True)
    parser.add_argument("--verbose",action='store_true')

    args=parser.parse_args()
    output_dir=create_path_with_time(args.output_dir)
    

    if args.verbose:
        csnLogger.info("Evaluation for Design History")
        csnLogger.info(f"Output Path {output_path}")

    with open(args.input_path,"rb") as f:
        data=pickle.load(f)
    
    for level in range(1,5):
        csnLogger.info(f"Level {level}")
        output_path=os.path.join(output_dir,'level_'+str(level))
        ensure_dir(output_path)
        generate_analysis_report(data=data,output_path=output_path,
                                logger=csnLogger,verbose=args.verbose, level='level_'+str(level))


def generate_analysis_report(data,output_path,logger,verbose,level):
    report_df = pd.DataFrame() # Dataframe for analysis
    # cm=np.zeros((4,4)) # Confusion Matrix

    uids=list(data.keys())

    for uid in tqdm(uids):
        best_report_df=process_uid_(uid,data,level=level)
        if best_report_df is not None:
            report_df=pd.concat([report_df,best_report_df])
    csv_path=os.path.join(output_path,f"report_df_{level}.csv")

    try:
        report_df.to_csv(csv_path, index=None)
        # logger.success(f"Report is saved at {csv_path}")
    except Exception as e:
        logger.error(f"Error saving csv file at {csv_path}")
        if verbose:
           print(traceback.print_exc())

    if verbose:
        logger.info("Calculating Metrics...")

    eval_dict = {}

    line_metrics = report_df[(report_df['line_total_gt'] > 0)][['line_recall', 'line_precision', 'line_f1']].mean() * 100
    eval_dict['line'] = {
        'recall': line_metrics['line_recall'],
        'precision': line_metrics['line_precision'],
        'f1': line_metrics['line_f1']
    }

    # Mean Recall, Precision, F1 for Arc
    arc_metrics = report_df[(report_df['arc_total_gt'] > 0)][['arc_recall', 'arc_precision', 'arc_f1']].mean() * 100
    eval_dict['arc'] = {
        'recall': arc_metrics['arc_recall'],
        'precision': arc_metrics['arc_precision'],
        'f1': arc_metrics['arc_f1']
    }

    # Mean Recall, Precision, F1 for Circle
    circle_metrics = report_df[(report_df['circle_total_gt'] > 0)][['circle_recall', 'circle_precision', 'circle_f1']].mean() * 100
    eval_dict['circle'] = {
        'recall': circle_metrics['circle_recall'],
        'precision': circle_metrics['circle_precision'],
        'f1': circle_metrics['circle_f1']
    }

    # Mean Recall, Precision, F1 for Extrusion
    ext_recall = report_df['num_ext'] / report_df['num_ext_gt']
    ext_precision = report_df['num_ext'] / report_df['num_ext_pred']
    ext_f1 = 2 * ext_recall * ext_precision / (ext_recall + ext_precision)
    extrusion_metrics = {
        'recall': ext_recall.mean() * 100,
        'precision': ext_precision.mean() * 100,
        'f1': ext_f1.mean() * 100
    }
    eval_dict.update({'extrusion': extrusion_metrics})

    
    # Update Chamfer Distance
    eval_dict['cd']={}
    eval_dict['cd']['median']=report_df['cd'][report_df['cd']>0].median()
    eval_dict['cd']['mean']=report_df['cd'][report_df['cd']>0].mean()
    eval_dict['invalidity_ratio_percentage']=report_df['cd'][report_df['cd']<0].count()*100/len(report_df)

    if verbose:
        json_formatted_str = json.dumps(eval_dict, indent=4)
        print(json_formatted_str)

    mean_report_path=os.path.join(output_path,f"mean_report_{level}.json")

    with open(mean_report_path,"w") as f:
        json.dump(eval_dict,f, indent=4)



def process_vec(pred_vec,gt_vec,bit,uid):
    try:
        pred_cad=CADSequence.from_vec(pred_vec,2,8,denumericalize=False)
        gt_cad=CADSequence.from_vec(gt_vec,2,8,denumericalize=False)

        report_df,cm=gt_cad.generate_report(pred_cad,uid)
        
        return report_df,cm
    except Exception as e:
        #print(e)
        return None,None

def process_uid_(uid,data,level):
    try:
        gt_vec = data[uid][level]['gt_cad_vec']
        all_cd = data[uid][level]['cd']
        best_index = 0
        pred_vec = data[uid][level]['pred_cad_vec'][best_index]
        df, _ = process_vec(pred_vec, gt_vec, 8, uid)
        df['cd'] = all_cd[best_index]

        return df

    except Exception as e:
        return None

if __name__=="__main__":
    main()

================================================
File: structure-git/cad3dify.md
================================================
Directory structure:
└── neka-nat-cad3dify/
    ├── README.md
    ├── LICENSE
    ├── pyproject.toml
    ├── .env.sample
    ├── cad3dify/
    │   ├── __init__.py
    │   ├── agents.py
    │   ├── chat_models.py
    │   ├── image.py
    │   ├── pipeline.py
    │   ├── render.py
    │   └── v1/
    │       ├── __init__.py
    │       ├── cad_code_generator.py
    │       └── cad_code_refiner.py
    ├── sample_data/
    └── scripts/
        ├── app.py
        └── cli.py


================================================
File: structure-git/cadroid-now.md
================================================
Directory structure:
└── princemuichkine-cadroid/
    ├── README.md
    ├── LICENSE
    ├── environment.yml
    ├── .env.sample
    ├── App/
    │   └── app.py
    ├── CadSeqProc/
    │   ├── README.md
    │   ├── cad_sequence.py
    │   ├── eda.py
    │   ├── integration.py
    │   ├── json2step.py
    │   ├── json2stl_skt3d.py
    │   ├── json2vec.py
    │   ├── merge_vlm_minimal.py
    │   ├── minimal_cad_json.py
    │   ├── split_json.py
    │   ├── test_recon_step.py
    │   ├── test_recon_stl.py
    │   ├── OCCUtils/
    │   │   ├── Common.py
    │   │   ├── Construct.py
    │   │   ├── Image.py
    │   │   ├── Iteration.py
    │   │   ├── Topology.py
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── edge.py
    │   │   ├── face.py
    │   │   ├── shell.py
    │   │   ├── solid.py
    │   │   ├── types_lut.py
    │   │   ├── vertex.py
    │   │   └── wire.py
    │   ├── enhanced_geometry/
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── factory.py
    │   │   ├── integration.py
    │   │   ├── intelligent_cad.py
    │   │   ├── llm_client.py
    │   │   ├── nurbs.py
    │   │   ├── organic.py
    │   │   ├── parametric.py
    │   │   └── tests.py
    │   ├── geometry/
    │   │   ├── arc.py
    │   │   ├── circle.py
    │   │   ├── curve.py
    │   │   ├── line.py
    │   │   ├── nurbs.py
    │   │   └── organic.py
    │   ├── sequence/
    │   │   ├── sketch/
    │   │   │   ├── coord_system.py
    │   │   │   ├── face.py
    │   │   │   ├── loop.py
    │   │   │   └── sketchsequence.py
    │   │   └── transformation/
    │   │       ├── deform.py
    │   │       └── extrude_sequence.py
    │   └── utility/
    │       ├── decorator.py
    │       ├── factory.py
    │       ├── logger.py
    │       ├── macro.py
    │       ├── shape_factory.py
    │       └── utils.py
    ├── Cad_VLM/
    │   ├── test.py
    │   ├── test_user_input.py
    │   ├── train.py
    │   ├── config/
    │   │   ├── inference.yaml
    │   │   ├── inference_user_input.yaml
    │   │   └── trainer.yaml
    │   ├── dataprep/
    │   │   └── t2c_dataset.py
    │   └── models/
    │       ├── decoder.py
    │       ├── loss.py
    │       ├── metrics.py
    │       ├── text2cad.py
    │       ├── utils.py
    │       └── layers/
    │           ├── __init__.py
    │           ├── adaptive_layer.py
    │           ├── attention.py
    │           ├── decorator.py
    │           ├── embedder.py
    │           ├── functional.py
    │           ├── layer_utils.py
    │           ├── text_embed.py
    │           └── utils_decode.py
    └── Evaluation/
        └── eval_seq.py


================================================
File: structure-git/text-to-cad.md
================================================
Directory structure:
└── sadilkhan-text2cad.git/
    ├── README.md
    ├── LICENSE
    ├── environment.yml
    ├── App/
    │   └── app.py
    ├── CadSeqProc/
    │   ├── README.md
    │   ├── cad_sequence.py
    │   ├── eda.py
    │   ├── json2step.py
    │   ├── json2stl_skt3d.py
    │   ├── json2vec.py
    │   ├── merge_vlm_minimal.py
    │   ├── minimal_cad_json.py
    │   ├── split_json.py
    │   ├── test_recon_step.py
    │   ├── test_recon_stl.py
    │   ├── OCCUtils/
    │   │   ├── Common.py
    │   │   ├── Construct.py
    │   │   ├── Image.py
    │   │   ├── Iteration.py
    │   │   ├── Topology.py
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── edge.py
    │   │   ├── face.py
    │   │   ├── shell.py
    │   │   ├── solid.py
    │   │   ├── types_lut.py
    │   │   ├── vertex.py
    │   │   └── wire.py
    │   ├── geometry/
    │   │   ├── arc.py
    │   │   ├── circle.py
    │   │   ├── curve.py
    │   │   └── line.py
    │   ├── sequence/
    │   │   ├── sketch/
    │   │   │   ├── coord_system.py
    │   │   │   ├── face.py
    │   │   │   ├── loop.py
    │   │   │   └── sketchsequence.py
    │   │   └── transformation/
    │   │       └── extrude_sequence.py
    │   └── utility/
    │       ├── decorator.py
    │       ├── logger.py
    │       ├── macro.py
    │       └── utils.py
    ├── Cad_VLM/
    │   ├── test.py
    │   ├── test_user_input.py
    │   ├── train.py
    │   ├── config/
    │   │   ├── inference.yaml
    │   │   ├── inference_user_input.yaml
    │   │   └── trainer.yaml
    │   ├── dataprep/
    │   │   └── t2c_dataset.py
    │   └── models/
    │       ├── decoder.py
    │       ├── loss.py
    │       ├── metrics.py
    │       ├── text2cad.py
    │       ├── utils.py
    │       └── layers/
    │           ├── __init__.py
    │           ├── adaptive_layer.py
    │           ├── attention.py
    │           ├── decorator.py
    │           ├── embedder.py
    │           ├── functional.py
    │           ├── layer_utils.py
    │           ├── text_embed.py
    │           └── utils_decode.py
    └── Evaluation/
        └── eval_seq.py

