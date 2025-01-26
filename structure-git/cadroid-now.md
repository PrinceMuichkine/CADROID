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
