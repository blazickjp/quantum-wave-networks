"""Hybrid Wave-LSTM Network implementation."""

from typing import Tuple, Optional
import torch
import torch.nn as nn

from model import WaveNetwork


class WaveLSTM(nn.Module):
    """Hybrid model combining Wave Network with LSTM for temporal-semantic processing."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 64,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        output_dim: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        """Initialize Wave-LSTM Network.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of wave network hidden layers
            lstm_hidden_dim: Dimension of LSTM hidden states
            num_lstm_layers: Number of LSTM layers
            output_dim: Dimension of output (number of classes)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        # Wave Network component
        self.wave_net = WaveNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output to LSTM dimensionality
        )

        # LSTM component
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output dimension adjustments
        lstm_output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim

        # Final layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, output_dim),
        )

        # Save dimensions for later use
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional

    def _init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden states for LSTM.

        Args:
            batch_size: Size of input batch
            device: Device to create tensors on

        Returns:
            Tuple of (hidden state, cell state)
        """
        num_directions = 2 if self.bidirectional else 1
        # Initialize hidden state and cell state
        h0 = torch.zeros(
            self.num_lstm_layers * num_directions,
            batch_size,
            self.lstm_hidden_dim,
            device=device,
        )
        c0 = torch.zeros(
            self.num_lstm_layers * num_directions,
            batch_size,
            self.lstm_hidden_dim,
            device=device,
        )
        return h0, c0

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_wave_modulation: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the hybrid network.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            hidden: Optional initial hidden state for LSTM
            use_wave_modulation: Whether to use wave modulation

        Returns:
            Tuple of (output logits, final hidden state)
        """
        batch_size = x.size(0)
        device = x.device

        # Process through Wave Network
        # Shape: (batch_size, seq_length, hidden_dim)
        wave_output = self.wave_net(x, use_modulation=use_wave_modulation)

        # Initialize LSTM hidden state if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, device)

        # Process through LSTM
        # lstm_output shape: (batch_size, seq_length, lstm_hidden_dim * num_directions)
        lstm_output, hidden = self.lstm(wave_output, hidden)

        # Take final sequence output for classification
        # Shape: (batch_size, lstm_hidden_dim * num_directions)
        if self.bidirectional:
            final_hidden = torch.cat((hidden[0][-2, :, :], hidden[0][-1, :, :]), dim=1)
        else:
            final_hidden = hidden[0][-1]

        # Final output layers
        # Shape: (batch_size, output_dim)
        output = self.output_layers(final_hidden)

        return output, hidden

    def get_wave_representations(
        self, x: torch.Tensor, use_wave_modulation: bool = True
    ) -> torch.Tensor:
        """Get intermediate wave representations for analysis.

        Args:
            x: Input tensor
            use_wave_modulation: Whether to use wave modulation

        Returns:
            Wave network representations
        """
        with torch.no_grad():
            return self.wave_net(x, use_modulation=use_wave_modulation)
