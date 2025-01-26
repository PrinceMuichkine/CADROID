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
