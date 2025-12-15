# 🔢 MNIST Digit Classifier

A deep learning project that classifies handwritten digits (0-9) using a Convolutional Neural Network (CNN), achieving **99.33% accuracy** on the MNIST dataset. Optimized for Apple M1 using Metal Performance Shaders (MPS).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- **High Accuracy**: 99.33% accuracy on MNIST test set
- **Modern Architecture**: CNN with 2 convolutional layers and dropout regularization
- **M1 Optimized**: Leverages Apple Silicon GPU (MPS) for faster training
- **Interactive Web UI**: Draw digits in your browser and get real-time predictions
- **Well-Structured**: Clean, modular code following best practices


## 📁 Project Structure
```
deep-learning-mnist/
├── models/
│   ├── __init__.py
│   └── cnn.py              # CNN architecture
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   └── helpers.py          # Training/testing helpers
├── data/                   # MNIST dataset (auto-downloaded)
├── checkpoints/            # Saved model weights
├── results/                # Training plots and visualizations
├── config.py               # Configuration settings
├── train.py                # Training script
├── test.py                 # Testing script
├── app.py                  # Gradio web interface
├── requirements.txt        # Dependencies
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/vigp17/DEEP-LEARNING-MNIST.git
cd DEEP-LERANING-MNIST
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🎓 Usage

### Training the Model

Train the CNN on MNIST dataset:
```bash
python train.py
```

This will:
- Download MNIST dataset automatically
- Train for 10 epochs (~3-5 minutes on M1 Mac)
- Save the best model to `checkpoints/mnist_model.pth`
- Generate training plots in `results/`

**Training output:**
```
Device: mps
Training samples: 60000
Test samples: 10000

Epoch 10/10
Train Loss: 0.0234 | Train Acc: 99.26%
Val Loss: 0.0312 | Val Acc: 99.33%
```

### Testing the Model

Evaluate the trained model:
```bash
python test.py
```

This will:
- Load the trained model
- Calculate per-digit accuracy
- Generate visualization plots
- Show sample predictions

### Running the Web Interface

Launch the interactive web app:
```bash
python app.py
```

Then open your browser at `http://127.0.0.1:7860`

Draw a digit and watch the model predict it in real-time!

## 🧠 Model Architecture
```
Input (28x28 grayscale image)
    ↓
Conv2D (32 filters, 3x3) → ReLU → MaxPool2D
    ↓
Conv2D (64 filters, 3x3) → ReLU → MaxPool2D
    ↓
Flatten (3136 features)
    ↓
Dense (128 neurons) → ReLU → Dropout(0.25)
    ↓
Dense (10 neurons, softmax)
    ↓
Output (probabilities for digits 0-9)
```

**Total Parameters:** ~100,000

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 99.33% |
| Training Time | ~3-5 min (M1 Mac) |
| Model Size | ~400 KB |

### Accuracy Per Digit

| Digit | Accuracy |
|-------|----------|
| 0 | 99.5% |
| 1 | 99.7% |
| 2 | 99.1% |
| 3 | 99.2% |
| 4 | 99.3% |
| 5 | 98.9% |
| 6 | 99.4% |
| 7 | 99.0% |
| 8 | 98.8% |
| 9 | 99.1% |

## 🔧 Configuration

Modify hyperparameters in `config.py`:
```python
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
```

## 📚 Technologies Used

- **PyTorch 2.9.1** - Deep learning framework
- **torchvision** - Dataset and transforms
- **Gradio** - Web interface
- **matplotlib** - Visualizations
- **NumPy** - Numerical operations

## 👨‍💻 Author

**Vignesh Pai**
- GitHub: [@vigp17](https://github.com/vigp17)
- LinkedIn: [Vignesh Pai](https://www.linkedin.com/in/vigneshpaib/)

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- PyTorch team for the amazing framework
- Gradio team for the easy-to-use web interface library

---

⭐ If you found this project helpful, please give it a star!