# Testing MoE for LLMs and VLLMs

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-yellow.svg)

A comprehensive testing framework for exploring and evaluating Mixture of Experts (MoE) architectures in Large Language Models (LLMs) and Vision-Language Models (VLLMs). This project provides tools for analyzing expert behavior, routing mechanisms, and performance metrics.

## 🌟 Key Features

- **Text Processing MoE**
  - Token-level expert routing
  - Complexity analysis
  - Real-time performance metrics
  - Detailed token analysis

- **Image Processing MoE**
  - Region-based expert assignment
  - Feature extraction
  - Pattern recognition
  - Visual complexity metrics

- **Advanced MoE Variants**
  - Switched MoE implementation
  - Customizable routing strategies
  - Extensible base architecture
  - Performance benchmarking

## 📊 Interactive Dashboard

Monitor and analyze MoE behavior in real-time:
- Complexity metrics visualization
- Expert utilization tracking
- Performance metrics
- Pattern analysis

## 🚀 Quick Start

1. **Installation**
```bash
# Clone the repository
https://github.com/edujbarrios/MoE_testing_framework_on_LLMs_and_VLLMs.git
cd MoE_testing_framework_on_LLMs_and_VLLMs

# Install dependencies
pip install -e .
```

2. **Basic Usage**
```python
from text_processing.text_moe import TextMoE
from image_processing.image_moe import ImageMoE

# Text Processing Example
text_moe = TextMoE(num_experts=3)
result = text_moe.process("Example text for processing")
print(result)

# Image Processing Example
image_moe = ImageMoE(num_experts=4)
result = image_moe.process(image_data)
print(result)
```

## 📁 Project Structure

```
.
├── benchmarking/           # Performance benchmarking tools
├── image_processing/       # Image MoE implementation
├── moe_variants/          # Different MoE architectures
├── notebooks/             # Jupyter notebooks for analysis
├── text_processing/       # Text MoE implementation
└── utils/                 # Utility functions and tools
```

## 🔍 Components

### Text Processing MoE
- Token-level complexity analysis
- Expert specialization:
  - Short text specialist
  - Medium text specialist
  - Long text specialist

### Image Processing MoE
- Region-based analysis
- Expert types:
  - Dark uniform regions
  - Bright uniform regions
  - Edge detection
  - Texture analysis

### Switched MoE
- Dynamic routing based on input complexity
- Confidence scoring
- Performance metrics

## 📈 Benchmarking

The project includes comprehensive benchmarking tools:
```python
from benchmarking import run_benchmark, generate_report

# Run benchmarks
results = run_benchmark(moe_variant, input_type='text')
generate_report(results, variant_name)
```

## 📓 Jupyter Notebooks

Three detailed notebooks are provided:
1. `01_basic_moe.ipynb` - Basic concepts and implementation
2. `02_text_processing.ipynb` - Text analysis and metrics
3. `03_image_processing.ipynb` - Image processing and visualization

## 🛠️ Development

### Requirements
- Python 3.11+
- Rich (for console interface)
- NumPy (for numerical operations)
- Matplotlib (for visualization)
- Jupyter (for notebooks)

### Running Tests
```bash
python -m pytest tests/
```

## 📚 Documentation

Detailed documentation is available in the [docs](./docs) directory, including:
- Architecture overview
- API reference
- Implementation details
- Performance optimization tips

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📖 Citation

If you use this project in your research, please cite:
```bibtex
@software{moe_testing_framework,
  title = {Testing MoE for LLMs and VLLMs},
  year = {2025},
  author = {Eduardo José Barrios García},
  url = {https://github.com/edujbarrios/MoE_testing_framework_on_LLMs_and_VLLMs}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [Sparse MoE Paper](https://arxiv.org/abs/2201.05596)
- [ViT Architecture](https://arxiv.org/abs/2010.11929)
- [LLM with MoE](https://arxiv.org/abs/2006.16668)

## 👥 Authors

- Eduardo José Barrios García - *Initial work* - [GitHub](https://github.com/EduardoBarrios)

## 🙏 Acknowledgments

- Thanks to all contributors
- Inspired by recent advances in LLMs and VLLMs
- Special thanks to the research community
