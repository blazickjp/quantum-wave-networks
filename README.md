# Quantum Net

This repository contains an implementation of a quantum-inspired neural network architecture, with results reproduced on the AG News classification dataset.

## Results

We successfully reproduced the paper's results on the AG News dataset, achieving:
- **Training Accuracy**: 92.75% (final epoch)
- **Validation Accuracy**: 90.57% (best model)

These results demonstrate strong performance on par with the original paper's findings for text classification tasks.

## Dataset

The AG News dataset is a collection of news articles from 4 major categories:
- World
- Sports
- Business
- Science/Technology

The dataset contains:
- Training set: 120,000 articles
- Test set: 7,600 articles

## Model Architecture

The Wave Network architecture implements quantum-inspired neural processing with the following key components:
- Quantum-inspired wave transformation layers
- Amplitude and phase encoding
- Wave interference modeling
- Quantum measurement inspired final layer

## Requirements

```
python>=3.7
torch
numpy
pandas
transformers
scikit-learn
tqdm
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/jblazick/quantum_net.git
cd quantum_net
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
python train.py --epochs 5 --batch_size 32 --learning_rate 0.001
```

Key hyperparameters used in our experiments:
- Epochs: 4
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Cross-entropy

## Model Checkpoints

The best performing model is automatically saved to:
```
/models/wave_network_best.pth
```

## Training Logs

Training metrics from our best run:
- Final epoch training accuracy: 92.75%
- Best validation accuracy: 90.57%
- Final training loss: 0.3311

Detailed training progression showed consistent improvement across epochs with stable convergence.

## Usage

To use the trained model for inference:

```python
from model import WaveNetwork
from utils import preprocess_text

# Load the model
model = WaveNetwork.load_from_checkpoint('models/wave_network_best.pth')
model.eval()

# Make predictions
text = "Your news article text here"
processed_text = preprocess_text(text)
prediction = model.predict(processed_text)
```

## Citation

If you use this implementation in your research, please cite:
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Before contributing:
1. Check existing issues or create a new one to discuss your proposed changes
2. Fork the repository and create your branch from `main`
3. Ensure tests pass and add new ones for your features
4. Update documentation as needed
5. Submit your pull request

## License

MIT License

Copyright (c) 2024 Joseph Blazick

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

## Contact

- **Author**: Joseph Blazick
- **Email**: jblazick@example.com
- **GitHub**: [@jblazick](https://github.com/jblazick)
- **Project Link**: [https://github.com/jblazick/quantum_net](https://github.com/jblazick/quantum_net)

For questions, feature requests, or support, please open an issue in the GitHub repository.