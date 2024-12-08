import torch
import torch.nn as nn


class WaveNetwork(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64, output_dim=4, num_layers=1):
        super(WaveNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Token embedding layer (vocab_size set to 30000 as per AG News dataset)
        self.embedding = nn.Embedding(30000, input_dim)

        # Wave representation layers
        self.global_semantics = nn.Linear(input_dim, hidden_dim)
        self.phase_vector = nn.Linear(input_dim, hidden_dim)

        # Wave variant layers
        self.wave_variant1 = nn.Linear(hidden_dim, hidden_dim)
        self.wave_variant2 = nn.Linear(hidden_dim, hidden_dim)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def compute_global_semantics(self, x):
        # Input shape: (batch_size, seq_length, input_dim)
        # Compute L2 norm across sequence length dimension
        return torch.norm(x, p=2, dim=1, keepdim=True)

    def compute_phase(self, x, G):
        # Following paper's formula for phase computation
        ratio = x / (G + 1e-8)
        squared_ratio = torch.clamp(ratio**2, max=0.99)  # Prevent numerical instability
        phase = torch.atan2(torch.sqrt(1 - squared_ratio + 1e-8), ratio)
        return phase

    def forward(self, x, use_modulation=True):
        # x shape: (batch_size, seq_length)
        batch_size, seq_length = x.shape

        # Embed tokens
        # Shape: (batch_size, seq_length, input_dim)
        embedded = self.embedding(x)

        # Project to hidden dimension
        # Shape: (batch_size, seq_length, hidden_dim)
        hidden = self.global_semantics(embedded)

        # Compute global semantics (magnitude)
        # Shape: (batch_size, 1, hidden_dim)
        G = self.compute_global_semantics(hidden)

        # Compute phase
        # Shape: (batch_size, seq_length, hidden_dim)
        alpha = self.compute_phase(hidden, G)

        # Create complex representation
        # Shape: (batch_size, seq_length, hidden_dim)
        real = G * torch.cos(alpha)
        imag = G * torch.sin(alpha)

        # Generate two variants
        Z1_real = self.wave_variant1(real)
        Z1_imag = self.wave_variant1(imag)
        Z2_real = self.wave_variant2(real)
        Z2_imag = self.wave_variant2(imag)

        # Apply wave operation (interference or modulation)
        if use_modulation:
            # Complex multiplication
            combined_real = Z1_real * Z2_real - Z1_imag * Z2_imag
            combined_imag = Z1_real * Z2_imag + Z1_imag * Z2_real
        else:
            # Complex addition
            combined_real = Z1_real + Z2_real
            combined_imag = Z1_imag + Z2_imag

        # Take magnitude of the combined representation
        combined = torch.sqrt(combined_real**2 + combined_imag**2 + 1e-8)

        # Apply layer normalization
        combined = self.layer_norm1(combined)

        # Mean pooling across sequence length
        # Shape: (batch_size, hidden_dim)
        pooled = torch.mean(combined, dim=1)

        # Feed-forward network
        output = self.feed_forward(pooled)
        output = self.layer_norm2(output)

        # Final classification layer
        # Shape: (batch_size, output_dim)
        logits = self.output_layer(output)

        return logits

    def log_model_statistics(self):
        """Log model statistics for debugging."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding_params = self.embedding.weight.numel()
        wave_params = sum(p.numel() for p in self.global_semantics.parameters()) + sum(
            p.numel() for p in self.phase_vector.parameters()
        )
        ff_params = sum(p.numel() for p in self.feed_forward.parameters())

        stats = {
            "Total Parameters": total_params,
            "Embedding Parameters": embedding_params,
            "Wave Operation Parameters": wave_params,
            "Feed-forward Parameters": ff_params,
        }

        return stats
