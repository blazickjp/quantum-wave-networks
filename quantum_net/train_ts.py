#!/usr/bin/env python
"""Training script for time series data using Wave-LSTM."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from wave_lstm import WaveLSTM


def create_time_series_data(
    num_samples: int = 1000,
    seq_length: int = 50,
    num_features: int = 10,
    num_classes: int = 4,
) -> tuple:
    """Create synthetic time series data for testing.

    Args:
        num_samples: Number of sequences to generate
        seq_length: Length of each sequence
        num_features: Number of features per timestep
        num_classes: Number of classes for classification

    Returns:
        Tuple of (X, y) with shapes ((num_samples, seq_length, num_features), (num_samples,))
    """
    # Generate random features
    X = np.random.randn(num_samples, seq_length, num_features)

    # Generate class labels based on sequence patterns
    y = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        # Create pattern-based labels
        pattern = np.mean(X[i], axis=0)  # Mean across time steps
        y[i] = hash(tuple(pattern)) % num_classes

    return torch.FloatTensor(X), torch.LongTensor(y)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    with tqdm(train_loader, desc="Training") as pbar:
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Get predictions and hidden states
            predictions, _ = model(batch_x)

            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            pred = predictions.argmax(dim=1)
            correct += pred.eq(batch_y).sum().item()
            total += batch_y.size(0)

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
            )

    return total_loss / len(train_loader), 100.0 * correct / total


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Get predictions and hidden states
        predictions, _ = model(batch_x)

        loss = criterion(predictions, batch_y)
        total_loss += loss.item()

        pred = predictions.argmax(dim=1)
        correct += pred.eq(batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / len(val_loader), 100.0 * correct / total


def main():
    # Parameters
    batch_size = 32
    num_samples = 1000
    seq_length = 50
    num_features = 10
    num_classes = 4
    train_ratio = 0.8
    epochs = 10
    learning_rate = 0.001

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate data
    X, y = create_time_series_data(
        num_samples=num_samples,
        seq_length=seq_length,
        num_features=num_features,
        num_classes=num_classes,
    )

    # Split into train/val
    train_size = int(train_ratio * num_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = WaveLSTM(
        input_dim=num_features,
        hidden_dim=64,
        lstm_hidden_dim=128,
        output_dim=num_classes,
        num_lstm_layers=2,
        dropout=0.1,
        bidirectional=True,
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model!")

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
