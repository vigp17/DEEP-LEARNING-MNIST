import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config

def get_data_loaders():
    """
    Create and return train and test data loaders for MNIST
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    
    # Define transformations
    # Converts PIL images to tensors and normalizes them
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std of MNIST dataset
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Shuffle training data
        num_workers=config.NUM_WORKERS,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # Don't shuffle test data
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader