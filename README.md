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