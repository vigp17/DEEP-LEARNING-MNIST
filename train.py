import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from config import config
from models import MNISTCNN
from utils import get_data_loaders, train_epoch, validate, save_checkpoint, plot_metrics

def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main():
    """Main training function"""
    
    # Set random seed for reproducibility
    set_seed(config.SEED)
    
    print("="*60)
    print("MNIST Digit Classification - Training")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Number of Epochs: {config.NUM_EPOCHS}")
    print("="*60)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_data_loaders()
    
    # Initialize model
    print("\nInitializing model...")
    model = MNISTCNN().to(config.DEVICE)
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_accuracy = 0.0
    
    # Training loop
    print("\nStarting training...")
    print("="*60)
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, config.DEVICE
        )
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                config.MODEL_SAVE_PATH
            )
            print(f"  ✓ New best model saved! (Accuracy: {best_accuracy:.2f}%)")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print("="*60)
    
    # Plot and save training metrics
    print("\nGenerating training plots...")
    plot_metrics(
        train_losses, train_accuracies,
        val_losses, val_accuracies,
        save_path=f"{config.RESULTS_DIR}/training_metrics.png"
    )
    
    print("\nAll done! Model and plots saved.")


if __name__ == "__main__":
    main()