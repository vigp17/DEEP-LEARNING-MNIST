import torch

class Config:
    """Configuration class for MNIST training"""

    # Device configuration
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Data parameters
    DATA_DIR = './data'
    BATCH_SIZE = 128
    NUM_WORKERS = 0  # Set to 0 for M1 compatibility

     # Model parameters
    INPUT_SIZE = 28 * 28  # MNIST images are 28x28
    HIDDEN_SIZE = 128
    NUM_CLASSES = 10  # Digits 0-9

    # Training parameters
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

     # Paths
    CHECKPOINT_DIR = './checkpoints'
    RESULTS_DIR = './results'
    MODEL_SAVE_PATH = './checkpoints/mnist_model.pth'

    # Logging
    LOG_INTERVAL = 100  # Print training stats every N batches

    # Random seed for reproducibility
    SEED = 42

    # Create a global config instance
config = Config()
