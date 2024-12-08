# Quantum Net

This repository contains an implementation of a quantum-inspired neural network architecture for text classification. The model demonstrates strong performance on the AG News dataset while using significantly fewer parameters than traditional transformers.

## Key Concepts

### Wave-Based Representations
The model represents text using complex vector representations inspired by quantum mechanics and wave physics:

1. **Global Semantic Vector (Magnitude)**
   - Encodes semantic strength through vector magnitudes
   - Shows a balanced, Gaussian-like distribution
   - Typical magnitude range: -0.4 to 0.4
   - Follows probability amplitude principles from quantum mechanics

2. **Phase Relationships**
   - Encodes semantic relationships through phase angles
   - Exhibits structured patterns in phase space
   - Shows hierarchical organization through radial clustering
   - Enables interference-based information processing

### Wave Operations
The model supports two modes of combining information:

1. **Wave Interference**
   ```
   Combined = Wave1 + Wave2  # Complex addition
   ```
   - Models direct interaction between semantic components
   - Allows constructive/destructive interference based on phase alignment

2. **Wave Modulation**
   ```
   Combined = Wave1 * Wave2  # Complex multiplication
   ```
   - Enables amplitude and phase modulation
   - More effective for modeling complex semantic interactions

## Results

We successfully reproduced the paper's results on the AG News dataset:
- **Training Accuracy**: 92.75% (final epoch)
- **Validation Accuracy**: 90.57% (best model)
- **Parameter Count**: 2.4M (vs 100M+ for BERT)

### Empirical Observations

Our analysis shows several interesting properties:

1. **Magnitude Distribution**
   - Bell-shaped, symmetric distribution around 0
   - Most values concentrated between -0.2 and 0.2
   - Suggests balanced semantic representations

2. **Phase Organization**
   - Non-random distribution of phases
   - Structured clustering patterns
   - Hierarchical organization from center to periphery

## Model Architecture

The Wave Network implements quantum-inspired neural processing with:
- Wave transformation layers
- Amplitude and phase encoding
- Wave interference modeling
- Quantum measurement inspired final layer

```python
# Example wave operation
if use_modulation:
    # Complex multiplication
    combined_real = Z1_real * Z2_real - Z1_imag * Z2_imag
    combined_imag = Z1_real * Z2_imag + Z1_imag * Z2_real
else:
    # Complex addition
    combined_real = Z1_real + Z2_real
    combined_imag = Z1_imag + Z2_imag
```

## Requirements

```
python>=3.9
torch>=2.1.0
numpy>=1.24.0
pandas>=2.0.0
transformers>=4.35.0
scikit-learn>=1.3.0
```

## Setup

1. Clone and setup environment:
```bash
git clone https://github.com/jblazick/quantum_net.git
cd quantum_net
./setup.sh
```

2. Install dependencies:
```bash
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

```python
from quantum_net.model import WaveNetwork

# Initialize model
model = WaveNetwork(
    input_dim=768,
    hidden_dim=64,
    output_dim=4
)

# Training/inference follows standard PyTorch patterns
outputs = model(inputs, use_modulation=True)
```

## Training

```bash
python train.py --epochs 5 --batch_size 32 --learning_rate 0.001
```

### Hyperparameters
- Epochs: 4
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Cross-entropy

## Analysis Tools

The repository includes tools for analyzing wave representations:

1. Wave Dynamics Analyzer:
```python
from wave_analysis.wave_dynamics import WaveDynamicsAnalyzer

analyzer = WaveDynamicsAnalyzer(model)
analyzer.visualize_wave_evolution(inputs)
```

2. Semantic Probe:
```python
from wave_analysis.semantic_probe import SemanticProbe

probe = SemanticProbe(model)
probe_results = probe.analyze_phase_separation(embeddings, labels)
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@misc{blazick2024quantum,
  author = {Blazick, Joseph},
  title = {Quantum Net: A Quantum-Inspired Neural Network for Text Classification},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jblazick/quantum_net}}
}
```

## Contact

- **Author**: Joseph Blazick
- **Email**: jblazick@example.com
- **GitHub**: [@jblazick](https://github.com/jblazick)