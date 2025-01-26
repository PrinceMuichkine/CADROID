import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from Cad_VLM.models.text2cad import Text2CAD
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT
from CadSeqProc.cad_sequence import CADSequence
import gradio as gr
import yaml
import torch


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
                
            # Load checkpoint with CPU as fallback
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
        # Use device-agnostic code
        pred_cad_seq_dict = model.test_decode(
            texts=text,
            maxlen=MAX_CAD_SEQUENCE_LENGTH,
            nucleus_prob=0,
            topk_index=1,
            device=device
        )
        
        pred_cad = CADSequence.from_vec(
            pred_cad_seq_dict["cad_vec"][0].cpu().numpy(),
            bit=N_BIT,
            post_processing=True,
        ).create_mesh()

        return pred_cad.mesh, pred_cad
    except Exception as e:
        print(f"Error in test_model: {str(e)}")
        return None

def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

# Set up device - use MPS if available on Mac, otherwise CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

config_path = "../Cad_VLM/config/inference_user_input.yaml"
config = parse_config_file(config_path)
model = load_model(config, device)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_cad_model_from_text(text):
    global model, config
    mesh, *extra = test_model(model=model, text=text, config=config, device=device)
    if mesh is not None:
        output_path = os.path.join(OUTPUT_DIR, "output.stl")
        mesh.export(output_path)
        return output_path
    else:
        raise gr.Error("Error generating CAD model from text")

examples = [
    "A ring.",
    "A rectangular prism.",
    "A 3D star shape with 5 points.",
    "A cylindrical object with a cylindrical hole in the center.",
    "A rectangular metal plate with four holes along its length."
]

title = "Text2CAD: Generating Sequential CAD Designs from Text Prompts"
description = """
Generate 3D CAD models from text descriptions. This demo runs on CPU/MPS for Mac compatibility.

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

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_cad_model_from_text,
    inputs=gr.Textbox(label="Text", placeholder="Enter a text description of the CAD model"),
    outputs=gr.Model3D(clear_color=[0.678, 0.847, 0.902, 1.0], label="3D CAD Model"),
    examples=examples,
    title=title,
    description=description,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=True)
