"""Wave Network model implementation."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveNetwork(nn.Module):
    """Wave Network implementation using quantum-inspired complex vector representations."""

    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=64, output_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.global_semantics = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.wave_variant1 = nn.Linear(hidden_dim, hidden_dim)
        self.wave_variant2 = nn.Linear(hidden_dim, hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def compute_global_semantics(self, x: torch.Tensor) -> torch.Tensor:
        """Compute global semantics vector.

        Args:
            x: Input tensor of shape (batch_size, seq_length, hidden_dim)

        Returns:
            Global semantics vector of shape (batch_size, 1, hidden_dim)
        """
        return torch.norm(x, p=2, dim=1, keepdim=True)

    def compute_phase(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """Compute phase vector.

        Args:
            x: Input tensor of shape (batch_size, seq_length, hidden_dim)
            G: Global semantics tensor of shape (batch_size, 1, hidden_dim)

        Returns:
            Phase vector of shape (batch_size, seq_length, hidden_dim)
        """
        ratio = x / (G + 1e-8)
        numerator = torch.sqrt(1 - ratio**2 + 1e-8)
        phase = torch.atan2(numerator, ratio)
        return phase

    def forward(self, x: torch.Tensor, use_modulation: bool = True) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length)
            use_modulation: Whether to use wave modulation (True) or interference (False)

        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        batch_size, seq_length = x.shape  # Fix: Unpack two dimensions
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        hidden = self.global_semantics(x)

        # Generate two variants
        hidden1 = self.wave_variant1(hidden)
        hidden2 = self.wave_variant2(hidden)

        # Compute global semantics and phase for each variant
        G1 = self.compute_global_semantics(hidden1)
        G2 = self.compute_global_semantics(hidden2)
        alpha1 = self.compute_phase(hidden1, G1)
        alpha2 = self.compute_phase(hidden2, G2)

        # Create complex representations
        Z1_real = G1 * torch.cos(alpha1)
        Z1_imag = G1 * torch.sin(alpha1)
        Z2_real = G2 * torch.cos(alpha2)
        Z2_imag = G2 * torch.sin(alpha2)

        # Apply wave operation
        if use_modulation:
            combined_real = Z1_real * Z2_real - Z1_imag * Z2_imag
            combined_imag = Z1_real * Z2_imag + Z1_imag * Z2_real
        else:
            combined_real = Z1_real + Z2_real
            combined_imag = Z1_imag + Z2_imag

        # Compute magnitude
        combined = torch.sqrt(
            torch.clamp(combined_real**2 + combined_imag**2, min=1e-8)
        )

        # Normalize
        combined = self.layer_norm1(combined)

        # Pool across sequence
        pooled = torch.mean(combined, dim=1)

        # Feed-forward
        output = self.feed_forward(pooled)
        output = self.layer_norm2(output)

        # Output
        logits = self.output_layer(output)
        return logits

    def log_model_statistics(self) -> dict:
        """Log model statistics for debugging."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding_params = sum(p.numel() for p in self.embedding.parameters())
        wave_params = (
            sum(p.numel() for p in self.global_semantics.parameters())
            + sum(p.numel() for p in self.wave_variant1.parameters())
            + sum(p.numel() for p in self.wave_variant2.parameters())
        )
        ff_params = sum(p.numel() for p in self.feed_forward.parameters())
        output_params = sum(p.numel() for p in self.output_layer.parameters())

        return {
            "Total Parameters": total_params,
            "Embedding Parameters": embedding_params,
            "Wave Operation Parameters": wave_params,
            "Feed-forward Parameters": ff_params,
            "Output Layer Parameters": output_params,
        }


# Example usage
if __name__ == "__main__":
    vocab_size = 30000  # Define a reasonable vocab size
    model = WaveNetwork(
        vocab_size=vocab_size, embedding_dim=768, hidden_dim=64, output_dim=4
    )
    x = torch.randint(0, vocab_size, (32, 10))  # Generate token indices
    logits = model(x, use_modulation=True)
    print(logits.shape)  # Should be (32, 4)
    print(model.log_model_statistics())
