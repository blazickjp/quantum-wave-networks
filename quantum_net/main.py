import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import WaveNetwork
from utils import load_ag_news, create_dataloaders
import logging


def train_wave_network():
    # Configuration
    config = {
        "input_dim": 128,
        "hidden_dim": 512,
        "output_dim": 4,  # AG News has 4 classes
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 4,
        "use_modulation": True,  # True for modulation, False for interference
    }

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Setup data paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "train.csv")
    test_path = os.path.join(base_dir, "data", "test.csv")

    logger.info(f"Loading data from {train_path}")

    # Load AG News dataset
    X_train, X_test, y_train, y_test = load_ag_news(
        file_path=train_path,
        max_length=config["input_dim"],
        vocab_size=30000,  # As per paper's specifications
    )

    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=config["batch_size"]
    )

    # Initialize model and training components
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = WaveNetwork(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")

    # Training loop
    best_accuracy = 0

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            try:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs, use_modulation=config["use_modulation"])
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if batch_idx % 100 == 0:
                    logger.info(
                        f"Epoch: {epoch+1}, Batch: {batch_idx}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Acc: {100.*correct/total:.2f}%"
                    )

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                try:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs, use_modulation=config["use_modulation"])
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                except Exception as e:
                    logger.error(f"Error in validation: {str(e)}")
                    continue

        accuracy = 100.0 * val_correct / val_total
        logger.info(f"Epoch {epoch+1} Validation Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_save_path = os.path.join(base_dir, "models", "wave_network_best.pth")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model to {model_save_path}")

    logger.info(f"Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    try:
        train_wave_network()
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
