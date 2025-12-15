import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification
    
    Architecture:
    - Conv Layer 1: 1 -> 32 channels
    - Conv Layer 2: 32 -> 64 channels
    - Fully Connected 1: 64*7*7 -> 128
    - Fully Connected 2: 128 -> 10 (output)
    """
    
    def __init__(self):
        super(MNISTCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, 10)
        """
        # First conv block: Conv -> ReLU -> Pool
        # Input: (batch, 1, 28, 28) -> Output: (batch, 32, 14, 14)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second conv block: Conv -> ReLU -> Pool
        # Input: (batch, 32, 14, 14) -> Output: (batch, 64, 7, 7)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten for fully connected layers
        # Input: (batch, 64, 7, 7) -> Output: (batch, 64*7*7)
        x = x.view(-1, 64 * 7 * 7)
        
        # First fully connected layer with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer (no activation - will use CrossEntropyLoss)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)