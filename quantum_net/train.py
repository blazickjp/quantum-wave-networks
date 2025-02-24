import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import WaveNetwork  # Assumes the WaveNetwork class is defined in model.py
from utils import (
    load_ag_news,
    create_dataloaders,
)  # Assumes data utilities are in utils.py
import logging
import os

# Set up logging to track training progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)

# Define hyperparameters
embedding_dim = 768  # Embedding dimension, matching BERT-like models
hidden_dim = 64  # Hidden dimension for the model
output_dim = 4  # Number of classes in AG News dataset
learning_rate = 0.0001  # Learning rate for the optimizer
batch_size = 16  # Batch size for training
epochs = 5  # Maximum number of training epochs
patience = 5  # Patience for early stopping
use_modulation = False  # Toggle between wave modulation (True) or interference (False)
max_length = 128  # Maximum sequence length for tokenization
vocab_size = 30000  # Vocabulary size limit

# Load the AG News dataset
file_path = "data/train.csv"  # Replace with the actual path to your AG News CSV file
X_train, X_test, y_train, y_test, vocab = load_ag_news(
    file_path, max_length=max_length, vocab_size=vocab_size
)

# Create DataLoaders for training and testing
train_loader, test_loader = create_dataloaders(
    X_train, X_test, y_train, y_test, batch_size
)

# Initialize the Wave Network model
model = WaveNetwork(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
)

# Set the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with early stopping
best_val_acc = 0  # Track the best validation accuracy
counter = 0  # Counter for early stopping
train_losses = []  # Store training losses for plotting
val_accuracies = []  # Store validation accuracies for plotting

for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()  # Clear gradients
        predictions = model(batch_x, use_modulation=use_modulation)  # Forward pass
        loss = criterion(predictions, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x, use_modulation=use_modulation)
            _, predicted = torch.max(predictions, 1)  # Get predicted class
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)

    # Log progress
    logger.info(
        f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
    )

    # Early stopping and model saving
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        counter = 0
        os.makedirs(
            "models", exist_ok=True
        )  # Create models directory if it doesn't exist
        model_path = f"models/wave_network_{'modulation' if use_modulation else 'interference'}.pth"
        torch.save(model.state_dict(), model_path)  # Save the best model
    else:
        counter += 1
        if counter >= patience:
            logger.info("Early stopping triggered")
            break

# Plot training loss and validation accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Val Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()
