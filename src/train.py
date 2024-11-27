import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import QuantumNet
from torch.utils.data import DataLoader, TensorDataset

# Configuration
input_size = 10
hidden_size = 32
output_size = 1
learning_rate = 0.001
epochs = 20
batch_size = 16

# Generate synthetic data
x = torch.randn(1000, input_size)
y = torch.randn(1000, output_size)

# Create DataLoader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = QuantumNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "../models/quantum_net.pth")

# Plot the loss curve
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
