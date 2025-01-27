Directory structure:
└── neka-nat-cad3dify.git/
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
File: README.md
================================================
# cad3dify

Using GPT-4o (or Claude 3.5 sonnet, Gemini 2.0 flash, Llama 3.2 on Vertex AI), generate a 3D CAD model (STEP file) from a 2D CAD image.

## Getting started

Installation.

```bash
git clone git@github.com:neka-nat/cad3dify.git
cd cad3dify
poetry install
```

Run script.
A STEP`file ("output.step") will be generated.

```bash
cd scripts
export OPENAI_API_KEY=<YOUR API KEY>
python cli.py <2D CAD Image File>
```

Or run streamlit spp

```bash
streamlit run scripts/app.py
streamlit run scripts/app.py -- --model_type claude  # Use Claude 3.5 sonnet
streamlit run scripts/app.py -- --model_type gemini  # Use Gemini 2.0 flash
streamlit run scripts/app.py -- --model_type llama  # Use Llama 3.2 on Vertex AI
```

## Demo

We will use the sample file [here](http://cad.wp.xdomain.jp/).

### Input image

![input](sample_data/g1-3.jpg)

### Generated 3D CAD model

![output](sample_data/gen_result1.png)


================================================
File: LICENSE
================================================
MIT License

Copyright (c) 2024 Shirokuma (k tanaka)

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
File: pyproject.toml
================================================
[tool.poetry]
name = "cad3dify"
version = "0.1.0"
description = ""
authors = ["neka-nat <nekanat.stock@gmail.com>"]
license = "MIT"
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
langchain = "^0.3.0"
langchain-openai = "^0.2.0"
svglib = "^1.5.1"
langchain-experimental = "^0.3.1"
cadquery = "^2.4.0"
rlpycairo = "^0.3.0"
loguru = "^0.7.2"
langchainhub = "^0.1.15"
streamlit = "^1.37.1"
numpy = "1.26.4"
langchain-google-vertexai = {version = "^2.0.1", optional = true}
python-dotenv = "^1.0.1"
langchain-anthropic = "^0.3.0"
langchain-google-genai = "^2.0.7"


[tool.poetry.group.dev.dependencies]
black = "^24.4.2"
isort = "^5.13.2"

[tool.poetry.extras]
vertexai = ["langchain-google-vertexai"]

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"


================================================
File: .env.sample
================================================
# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Anthropic (Optional)
ANTHROPIC_API_KEY=your-anthropic-api-key

# Google GenAI (Optional)
GOOGLE_API_KEY=your-google-api-key

# VertexAI (Optional)
VERTEXAI_PROJECT=your-project-id
VERTEXAI_LOCATION=your-location
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account.json


================================================
File: cad3dify/__init__.py
================================================
from .v1.cad_code_refiner import CadCodeRefinerChain
from .v1.cad_code_generator import CadCodeGeneratorChain
from .image import ImageData
from .pipeline import generate_step_from_2d_cad_image


================================================
File: cad3dify/agents.py
================================================
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

from .chat_models import MODEL_TYPE, ChatModelParameters

_instructions = """You are an agent designed to execute and debug the given Python code.
Please make corrections so that the code runs successfully without changing the intended purpose of the given code.
`cadquery` is installed in the environment, so you can use it without setting it up.
Even if you can tell that no corrections are needed without executing it, you still need to run the code to confirm it works properly.
If it is difficult to make the code run successfully despite making corrections, respond with "I cannot fix it."
"""


def execute_python_code(code: str, model_type: MODEL_TYPE = "gpt", only_execute: bool = False) -> str:
    tools = [PythonREPLTool()]
    if only_execute:
        return tools[0].run(code)
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=_instructions)
    agent = create_react_agent(
        ChatModelParameters.from_model_name(model_type).create_chat_model(),
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    output = agent_executor.invoke(
        {
            "input": f"Please execute the following code. If it doesn't work, fix the errors and make it run.\n```python\n{code}\n```\n"
        }
    )["output"]
    return output


================================================
File: cad3dify/chat_models.py
================================================
from typing import Literal

try:
    import os

    import vertexai
    vertexai.init(project=os.environ["VERTEXAI_PROJECT"], location=os.environ["VERTEXAI_LOCATION"])
except KeyError:
    print("VertexAI is not initialized. Please set VERTEXAI_PROJECT and VERTEXAI_LOCATION environment variables.")
except Exception as e:
    pass
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

MODEL_TYPE = Literal["gpt", "claude", "gemini", "llama"]
PROVIDER_TYPE = Literal["openai", "anthropic", "google", "vertex_ai"]


class ChatModelParameters(BaseModel):
    provider: PROVIDER_TYPE
    model_name: str
    temperature: float
    max_tokens: int | None = None

    @classmethod
    def default(cls) -> "ChatModelParameters":
        return cls(
            provider="openai",
            model_name="gpt-4o-2024-08-06",
            temperature=0.0,
            max_tokens=16384,
        )

    @classmethod
    def from_model_name(
        cls,
        model_type: MODEL_TYPE,
        temperature: float = 0.0,
    ) -> "ChatModelParameters":
        model_type_to_parameters = {
            "gpt": cls(
                provider="openai",
                model_name="gpt-4o-2024-08-06",
                temperature=temperature,
                max_tokens=16384,
            ),
            "claude": cls(
                provider="anthropic",
                model_name="claude-3-5-sonnet-20241022",
                temperature=temperature,
                max_tokens=8192,
            ),
            "gemini": cls(
                provider="google",
                model_name="gemini-2.0-flash-exp",
                temperature=temperature,
                max_tokens=8192,
            ),
            "llama": cls(
                provider="vertex_ai",
                model_name="meta/llama-3.2-90b-vision-instruct-maas",
                temperature=temperature,
            ),
        }
        return model_type_to_parameters.get(model_type, cls.default())

    def create_chat_model(self) -> BaseChatModel:
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider == "anthropic":
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider == "google":
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider == "vertex_ai":
            from google.auth import default
            from langchain_google_vertexai.model_garden_maas.llama import (
                VertexModelGardenLlama,
            )

            credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            return VertexModelGardenLlama(
                model_name=self.model_name,
                temperature=self.temperature,
                credentials=credentials,
            )
        else:
            raise ValueError(f"provider {self.provider} is not supported.")


================================================
File: cad3dify/image.py
================================================
import base64
import io
import os
from typing import Literal

from pydantic import BaseModel
from PIL import Image

ImageTypes = Literal["jpg", "jpeg", "png", "gif"]


class ImageData(BaseModel):
    """画像データのクラス

    Args:
        data (str): 画像データ(base64エンコード)
        type (ImageTypes): 画像の拡張子
    """

    data: str
    type: ImageTypes

    @classmethod
    def load_from_file(cls, file_path: str) -> "ImageData":
        """ファイルから画像データを読み込む

        Args:
            file_path (str): 画像ファイルのパス

        Returns:
            ImageData: 画像データ
        """
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return cls(data=data, type=os.path.splitext(file_path)[1][1:])

    def merge(self, other: "ImageData") -> "ImageData":
        """2つの画像データをマージする

        Args:
            other (ImageData): マージする画像データ

        Returns:
            ImageData: マージされた画像データ
        """
        img1 = Image.open(io.BytesIO(base64.b64decode(self.data)))
        img2 = Image.open(io.BytesIO(base64.b64decode(other.data)))
        dst = Image.new("RGB", (img1.width + img2.width, img1.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width, 0))
        output = io.BytesIO()
        dst.save(output, format=self.type)
        return ImageData(data=base64.b64encode(output.getvalue()).decode("utf-8"), type=self.type)

    def convert(self, type: ImageTypes) -> "ImageData":
        """画像データを指定された形式に変換する

        Args:
            type (ImageTypes): 変換する形式
        """
        img = Image.open(io.BytesIO(base64.b64decode(self.data)))
        output = io.BytesIO()
        img.save(output, format=type)
        return ImageData(data=base64.b64encode(output.getvalue()).decode("utf-8"), type=type)


================================================
File: cad3dify/pipeline.py
================================================
import tempfile

from loguru import logger

from .agents import execute_python_code
from .chat_models import MODEL_TYPE
from .image import ImageData
from .render import render_and_export_image
from .v1.cad_code_refiner import CadCodeRefinerChain
from .v1.cad_code_generator import CadCodeGeneratorChain

def index_map(index: int) -> str:
    if index == 0:
        return "1st"
    elif index == 1:
        return "2nd"
    elif index == 2:
        return "3rd"
    else:
        return f"{index + 1}th"


def generate_step_from_2d_cad_image(
    image_filepath: str,
    output_filepath: str,
    num_refinements: int = 3,
    model_type: MODEL_TYPE = "gpt",
):
    """Generate a STEP file from a 2D CAD image

    Args:
        image_filepath (str): Path to the 2D CAD image
        output_filepath (str): Path to the output STEP file
    """
    only_execute = (model_type == "llama")  # llamaだとagentがうまく動かない
    image_data = ImageData.load_from_file(image_filepath)
    chain = CadCodeGeneratorChain(model_type=model_type)

    result = chain.invoke(image_data)["result"]
    code = result.format(output_filename=output_filepath)
    logger.info("1st code generation complete. Running code...")
    logger.debug("Generated 1st code:")
    logger.debug(code)
    output = execute_python_code(code, model_type=model_type, only_execute=only_execute)
    logger.debug(output)

    refiner_chain = CadCodeRefinerChain(model_type=model_type)

    for i in range(num_refinements):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            render_and_export_image(output_filepath, f.name)
            logger.info(f"Temporarily rendered image to {f.name}")
            rendered_image = ImageData.load_from_file(f.name)
            result = refiner_chain.invoke(
                {"code": code, "original_input": image_data, "rendered_result": rendered_image}
            )["result"]
            if result is None:
                logger.error(f"Refinement failed. Skipping to the next step.")
                continue
            code = result.format(output_filename=output_filepath)
            logger.info("Refined code generation complete. Running code...")
            logger.debug(f"Generated {index_map(i)} refined code:")
            logger.debug(code)
            try:
                output = execute_python_code(code, model_type=model_type, only_execute=only_execute)
                logger.debug(output)
            except Exception as e:
                logger.error(f"Error occurred during code execution: {e}")
                continue


================================================
File: cad3dify/render.py
================================================
import tempfile

import cadquery as cq
from cadquery import exporters
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg


def render_and_export_image(cat_filepath: str, output_filepath: str):
    """Render a CAD file and export it as an SVG file

    Args:
        cat_file (str): Path to the CAD file
        output_filename (str): Path to the output PNG file
    """
    cad = cq.importers.importStep(cat_filepath)
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=True) as f:
        exporters.export(cad, f.name)
        drawing = svg2rlg(f.name)

    renderPM.drawToFile(drawing, output_filepath, fmt="PNG")


================================================
File: cad3dify/v1/cad_code_generator.py
================================================
import re
import textwrap
from typing import Any, Union

from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate

from ..chat_models import MODEL_TYPE, ChatModelParameters
from ..image import ImageData


def _parse_code(input: dict) -> dict:
    match = re.search(r"```(?:python)?\n(.*?)\n```", input["text"], re.DOTALL)
    if match:
        code_output = match.group(1).strip()
        return {"result": code_output}
    else:
        return {"result": None}


_cad_query_examples = [
    (
        "You can use a list of points to construct multiple objects at once. Most construction methods, "
        "like Workplane.circle() and Workplane.rect(), will operate on multiple points if they are on the stack",
        """r = cq.Workplane("front").circle(2.0)  # make base
r = r.pushPoints(
    [(1.5, 0), (0, 1.5), (-1.5, 0), (0, -1.5)]
)  # now four points are on the stack
r = r.circle(0.25)  # circle will operate on all four points
result = r.extrude(0.125)  # make prism""",
    ),
    (
        "This example uses a polyline to create one half of an i-beam shape, which is mirrored to create the final profile.",
        """(L, H, W, t) = (100.0, 20.0, 20.0, 1.0)
pts = [
    (0, H / 2.0),
    (W / 2.0, H / 2.0),
    (W / 2.0, (H / 2.0 - t)),
    (t / 2.0, (H / 2.0 - t)),
    (t / 2.0, (t - H / 2.0)),
    (W / 2.0, (t - H / 2.0)),
    (W / 2.0, H / -2.0),
    (0, H / -2.0),
]
result = cq.Workplane("front").polyline(pts).mirrorY().extrude(L)""",
    ),
    (
        "Combining a few basic functions, its possible to make a very good parametric bearing pillow block, with just a few lines of code.",
        """(length, height, bearing_diam, thickness, padding) = (30.0, 40.0, 22.0, 10.0, 8.0)

result = (
    cq.Workplane("XY")
    .box(length, height, thickness)
    .faces(">Z")
    .workplane()
    .hole(bearing_diam)
    .faces(">Z")
    .workplane()
    .rect(length - padding, height - padding, forConstruction=True)
    .vertices()
    .cboreHole(2.4, 4.4, 2.1)
)""",
    ),
    (
        "This sample is one of the longer ones at 13 lines, but that's very short compared to the pythonOCC version, which is 10x longer!",
        """(L, w, t) = (20.0, 6.0, 3.0)
s = cq.Workplane("XY")

# Draw half the profile of the bottle and extrude it
p = (
    s.center(-L / 2.0, 0)
    .vLine(w / 2.0)
    .threePointArc((L / 2.0, w / 2.0 + t), (L, w / 2.0))
    .vLine(-w / 2.0)
    .mirrorX()
    .extrude(30.0, True)
)

# Make the neck
p = p.faces(">Z").workplane(centerOption="CenterOfMass").circle(3.0).extrude(2.0, True)

# Make a shell
result = p.faces(">Z").shell(0.3)""",
    ),
    (
        "This specific examples generates a helical cycloidal gear.",
        """import cadquery as cq
from math import sin, cos, pi, floor


# define the generating function
def hypocycloid(t, r1, r2):
    return (
        (r1 - r2) * cos(t) + r2 * cos(r1 / r2 * t - t),
        (r1 - r2) * sin(t) + r2 * sin(-(r1 / r2 * t - t)),
    )


def epicycloid(t, r1, r2):
    return (
        (r1 + r2) * cos(t) - r2 * cos(r1 / r2 * t + t),
        (r1 + r2) * sin(t) - r2 * sin(r1 / r2 * t + t),
    )


def gear(t, r1=4, r2=1):
    if (-1) ** (1 + floor(t / 2 / pi * (r1 / r2))) < 0:
        return epicycloid(t, r1, r2)
    else:
        return hypocycloid(t, r1, r2)


# create the gear profile and extrude it
result = (
    cq.Workplane("XY")
    .parametricCurve(lambda t: gear(t * 2 * pi, 6, 1))
    .twistExtrude(15, 90)
    .faces(">Z")
    .workplane()
    .circle(2)
    .cutThruAll()
)
""",
    ),
    (
        "This script will produce any size regular rectangular Lego(TM) brick. Its only tricky because of the logic regarding the underside of the brick.",
        """#####
# Inputs
######
lbumps = 6  # number of bumps long
wbumps = 2  # number of bumps wide
thin = True  # True for thin, False for thick

#
# Lego Brick Constants-- these make a Lego brick a Lego :)
#
pitch = 8.0
clearance = 0.1
bumpDiam = 4.8
bumpHeight = 1.8
if thin:
    height = 3.2
else:
    height = 9.6

t = (pitch - (2 * clearance) - bumpDiam) / 2.0
postDiam = pitch - t  # works out to 6.5
total_length = lbumps * pitch - 2.0 * clearance
total_width = wbumps * pitch - 2.0 * clearance

# make the base
s = cq.Workplane("XY").box(total_length, total_width, height)

# shell inwards not outwards
s = s.faces("<Z").shell(-1.0 * t)

# make the bumps on the top
s = (
    s.faces(">Z")
    .workplane()
    .rarray(pitch, pitch, lbumps, wbumps, True)
    .circle(bumpDiam / 2.0)
    .extrude(bumpHeight)
)

# add posts on the bottom. posts are different diameter depending on geometry
# solid studs for 1 bump, tubes for multiple, none for 1x1
tmp = s.faces("<Z").workplane(invert=True)

if lbumps > 1 and wbumps > 1:
    tmp = (
        tmp.rarray(pitch, pitch, lbumps - 1, wbumps - 1, center=True)
        .circle(postDiam / 2.0)
        .circle(bumpDiam / 2.0)
        .extrude(height - t)
    )
elif lbumps > 1:
    tmp = (
        tmp.rarray(pitch, pitch, lbumps - 1, 1, center=True)
        .circle(t)
        .extrude(height - t)
    )
elif wbumps > 1:
    tmp = (
        tmp.rarray(pitch, pitch, 1, wbumps - 1, center=True)
        .circle(t)
        .extrude(height - t)
    )
else:
    tmp = s""",
    ),
]


_design_steps = """
1. **Model Construction Steps**
   - **Setting the Workplane**:
     - Use the `Workplane()` method to select the appropriate workplane (e.g., "XY", "XZ", "YZ").
     - If necessary, offset the plane using `workplane(offset=value)`.
     - **Note**: The workplane serves as the basis for sketches and operations, so it's crucial to set it correctly. Choosing the wrong plane may result in shapes being created in unintended positions.
   - **Drawing Basic Shapes**:
     - Use methods like `circle()`, `rect()`, `polygon()` to draw 2D shapes.
     - Specify dimensions (e.g., radius, width, height) clearly.
     - **Note**: In CadQuery, the order of method chaining is important. Always set the workplane before drawing shapes, and specify dimensions accurately.
   - **Creating 3D Shapes**:
     - Convert 2D shapes into 3D using methods like `extrude()`, `revolve()`.
     - Specify the direction and distance of extrusion precisely.
     - **Note**: Using negative values will extrude in the opposite direction, so pay attention to the direction.
   - **Adding Details**:
     - **Fillets and Chamfers**:
       - Use the `fillet()`, `chamfer()` methods to process edges.
       - Use selectors like `edges()`, `faces()` to accurately select the edges to apply.
       - **Note**: Incorrect edge selection may result in unintended areas being processed.
     - **Creating Holes and Pockets**:
       - Remove material using methods like `hole()`, `cutBlind()`, `cutThruAll()`.
       - Specify positions and dimensions in detail.
       - **Note**: Before making holes, select the appropriate workplane or face.
     - **Patterning**:
       - Repeat features using methods like `mirror()`, `array()`, `polarArray()`.
       - **Note**: Specify the direction and spacing of repetitions accurately.
   - **Defining Dimensions and Parameters**:
     - Define important dimensions as variables and use them consistently in the code.
     - **Note**: Parameterizing makes it easier to change dimensions later.
   - **Adding Comments**:
     - Add comments to each step to enhance code readability and maintainability.
2. **Setting Constraints**
   - **Dimensional Constraints**:
     - Specify dimensions accurately to ensure model precision.
     - **Note**: CadQuery supports parametric modeling, and changes in dimensions reflect throughout the model.
   - **Geometric Constraints**:
     - Use appropriate sketching techniques to maintain relationships between features.
     - **Note**: CadQuery itself does not have constraint functions within sketches, so dimensions and positions need to be set carefully.
   - **Defining Relationships Between Features**:
     - Position features using methods like `translate()`, `rotate()`, `align()`.
     - **Note**: Specify alignment using absolute or relative coordinates accurately.
3. **Verification and Adjustment**
   - **Model Verification**:
     - Confirm the validity of the model using the `val().isValid()` method.
     - **Note**: If errors occur, check boolean operations and shape consistency.
   - **Making Adjustments**:
     - Change parameters or dimensions and regenerate the model.
     - **Note**: After changing parameters, ensure all related parts are updated correctly.
   - **Error Checking and Debugging**:
     - Review error messages and identify problematic areas.
     - Visualize intermediate results using the `show_object()` method.
     - **Note**: CadQuery errors may not be specific, so check results at each step.
4. **Points Specific to CadQuery**
   - **Order of Method Chaining**:
     - The order of method chaining affects the result, so describe the sequence of operations precisely.
     - **Example**: Set the plane with `workplane()` before drawing shapes with `circle()`.
   - **Using Selectors**:
     - Use appropriate selectors like `faces()`, `edges()`, `vertices()` when selecting specific parts of the shape.
     - **Note**: Ambiguous selector specifications may result in unintended parts being selected.
   - **Workplane Context**:
     - Using `workplane()` changes the workplane, affecting subsequent operations.
     - **Note**: Be mindful of which workplane is active for each operation.
   - **Handling Boolean Operations**:
     - Confirm the validity and consistency of shapes when combining them using `union()`, `cut()`, `intersect()`.
     - **Note**: Tiny gaps or overlaps may cause errors.
   - **Ensuring Parametric Design**:
     - Manage all dimensions with variables so that the model updates dynamically.
     - **Note**: Manually entering numbers makes changes difficult later.
"""


class CadCodeGeneratorChain(SequentialChain):
    model_type: MODEL_TYPE = "gpt"

    def __init__(self, model_type: MODEL_TYPE = "gpt") -> None:
        sample_codes = "\n\n".join(
            [f"{explanation}\n```python\n{code}\n```" for explanation, code in _cad_query_examples]
        )
        gen_cad_code_prompt = (
            "You are a highly skilled CAD designer. Please write code that converts the attached 2D CAD image into a 3D CAD model using a Python CAD library called 'cadquery.'\n"
            "## Points to Note\n"
            "* Please use the `cadquery.exporters.export` function to output the created 3D model as a STEP file.\n"
            "* Where you describe the output file path, use the template string `{{output_filename}}`. The 'output_filename' includes the file extension.\n"
            "* Surround the code with a markdown code block.\n"
            "* Refer to the sample code for how to use Cadquery.\n"
            "* Write CAD code following these steps:\n"
            f"{textwrap.indent(_design_steps, prefix='  ')}\n"
            "## Cadquery Sample Code\n"
            f"{sample_codes}\n"
            "## Start here\n"
            "Output code:"
        )
        prompt = ChatPromptTemplate(
            input_variables=["image_type", "image_data"],
            messages=[
                HumanMessagePromptTemplate(
                    prompt=[
                        PromptTemplate(input_variables=[], template=gen_cad_code_prompt),
                        ImagePromptTemplate(
                            input_variables=["image_type", "image_data"],
                            template={"url": "data:image/{image_type};base64,{image_data}"},
                        ),
                    ]
                )
            ],
        )
        llm = ChatModelParameters.from_model_name(model_type).create_chat_model()

        super().__init__(
            chains=[
                LLMChain(prompt=prompt, llm=llm),  # type: ignore
                TransformChain(
                    input_variables=["text"],
                    output_variables=["result"],
                    transform=_parse_code,
                    atransform=None,
                ),
            ],
            input_variables=["image_type", "image_data"],
            output_variables=["result"],
            verbose=True,
        )
        self.model_type = model_type

    def prep_inputs(self, inputs: Union[dict[str, Any], Any]) -> dict[str, str]:
        assert isinstance(inputs, ImageData) or (
            "input" in inputs and isinstance(inputs["input"], ImageData)
        ), "inputs must be ImageData or dict with 'input' and 'input' must be ImageData"
        if isinstance(inputs, ImageData):
            inputs = {"input": inputs}
        if self.model_type == "claude" and inputs["input"].type != "png":
            inputs["input"] = inputs["input"].convert("png")
        inputs["image_type"] = inputs["input"].type
        inputs["image_data"] = inputs["input"].data
        return inputs


================================================
File: cad3dify/v1/cad_code_refiner.py
================================================
from typing import Any, Union

from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate

from .cad_code_generator import _parse_code
from ..chat_models import MODEL_TYPE, ChatModelParameters
from ..image import ImageData


class CadCodeRefinerChain(SequentialChain):
    model_type: MODEL_TYPE = "gpt"

    def __init__(self, model_type: MODEL_TYPE = "gpt") -> None:
        refine_cad_code_prompt = (
            "You are a highly skilled CAD designer. You have created the following code that converts the attached 2D CAD image into a 3D CAD model using a Python CAD library called 'cadquery.'\n"
            "When the CAD model obtained from this code is rendered in 3D, the attached 3D view image is obtained.\n"
            "Please compare this 3D view image with the 2D CAD drawing and modify the code to correct the CAD model.\n"
            "## Code\n"
            "```python\n"
            "{code}\n"
            "```\n"
            "## Start here\n"
            "Corrected code:"
        )
        if model_type in ["gpt", "claude", "gemini"]:
            prompt = ChatPromptTemplate(
                input_variables=[
                    "code",
                    "original_image_type",
                    "original_image_data",
                    "rendered_image_type",
                    "rendered_image_data",
                ],
                messages=[
                    HumanMessagePromptTemplate(
                        prompt=[
                            PromptTemplate(input_variables=["code"], template=refine_cad_code_prompt),
                            ImagePromptTemplate(
                                input_variables=["original_image_type", "original_image_data"],
                                template={
                                    "url": "data:image/{original_image_type};base64,{original_image_data}",
                                },
                            ),
                            ImagePromptTemplate(
                                input_variables=["rendered_image_type", "rendered_image_data"],
                                template={
                                    "url": "data:image/{rendered_image_type};base64,{rendered_image_data}",
                                },
                            ),
                        ]
                    )
                ],
            )
        elif model_type == "llama":
            prompt = ChatPromptTemplate(
                input_variables=[
                    "code",
                    "original_and_rendered_image_type",
                    "original_and_rendered_image_data",
                ],
                messages=[
                    HumanMessagePromptTemplate(
                        prompt=[
                            PromptTemplate(input_variables=["code"], template=refine_cad_code_prompt),
                            ImagePromptTemplate(
                                input_variables=["original_and_rendered_image_type", "original_and_rendered_image_data"],
                                template={
                                    "url": "data:image/{original_and_rendered_image_type};base64,{original_and_rendered_image_data}",
                                },
                            ),
                            
                        ]
                    )
                ],
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        llm = ChatModelParameters.from_model_name(model_type).create_chat_model()

        super().__init__(
            chains=[
                LLMChain(prompt=prompt, llm=llm),  # type: ignore
                TransformChain(
                    input_variables=["text"],
                    output_variables=["result"],
                    transform=_parse_code,
                    atransform=None,
                ),
            ],
            input_variables=prompt.input_variables,
            output_variables=["result"],
            verbose=True,
        )
        self.model_type = model_type

    def prep_inputs(self, inputs: Union[dict[str, Any], Any]) -> dict[str, str]:
        assert (
            "original_input" in inputs
            and isinstance(inputs["original_input"], ImageData)
            and "rendered_result" in inputs
            and isinstance(inputs["rendered_result"], ImageData)
            and "code" in inputs
            and isinstance(inputs["code"], str)
        ), "inputs must have 'original_input' and 'rendered_result' and 'code' keys"
        if self.model_type in ["gpt", "claude", "gemini"]:
            if self.model_type == "claude" and inputs["original_input"].type != "png":
                # if the image type is not png and the model is claude, convert it to png.
                inputs["original_input"] = inputs["original_input"].convert("png")
                inputs["rendered_result"] = inputs["rendered_result"].convert("png")
            inputs["original_image_type"] = inputs["original_input"].type
            inputs["original_image_data"] = inputs["original_input"].data
            inputs["rendered_image_type"] = inputs["rendered_result"].type
            inputs["rendered_image_data"] = inputs["rendered_result"].data
        elif self.model_type == "llama":
            inputs["original_and_rendered_image_type"] = inputs["original_input"].type
            inputs["original_and_rendered_image_data"] = inputs["original_input"].merge(
                inputs["rendered_result"]
            )
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        inputs["code"] = inputs["code"]
        return inputs


================================================
File: scripts/app.py
================================================
import argparse

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
from cad3dify import generate_step_from_2d_cad_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt")
    return parser.parse_args()


args = parse_args()

st.title("2D図面 to 3DCAD")

uploaded_file = st.sidebar.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png"])

# 画像がアップロードされたら表示
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)
    st.write("画像のサイズ: ", image.size)
    with open("temp.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner("画像を処理中..."):
        generate_step_from_2d_cad_image("temp.png", "output.step", model_type=args.model_type)
    st.success("3DCADデータの生成が完了しました。")
else:
    st.write("画像がアップロードされていません。")


================================================
File: scripts/cli.py
================================================
from cad3dify import generate_step_from_2d_cad_image


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_filepath", type=str, help="Path to the image file")
    parser.add_argument("--output_filepath", type=str, default="output.step", help="Path to the output STEP file")
    args = parser.parse_args()

    generate_step_from_2d_cad_image(args.image_filepath, args.output_filepath)


if __name__ == "__main__":
    main()

