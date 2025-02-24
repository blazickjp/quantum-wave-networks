"""Tests for Wave-LSTM hybrid model."""

import torch
import pytest
from wave_lstm import WaveLSTM


@pytest.fixture
def model():
    """Create a model instance for testing."""
    return WaveLSTM(
        input_dim=768,
        hidden_dim=64,
        lstm_hidden_dim=128,
        num_lstm_layers=2,
        output_dim=4,
        dropout=0.1,
        bidirectional=True,
    )


def test_model_initialization(model):
    """Test model initialization."""
    assert isinstance(model, WaveLSTM)
    assert model.hidden_dim == 64
    assert model.lstm_hidden_dim == 128
    assert model.num_lstm_layers == 2
    assert model.bidirectional == True


def test_forward_pass(model):
    """Test model forward pass."""
    batch_size = 2
    seq_length = 10
    input_dim = 768
    x = torch.randint(0, 30000, (batch_size, seq_length))
    
    # Test with wave modulation
    output, hidden = model(x, use_wave_modulation=True)
    assert output.shape == (batch_size, model.output_layers[-1].out_features)
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2  # (hidden_state, cell_state)
    
    # Test with wave interference
    output, hidden = model(x, use_wave_modulation=False)
    assert output.shape == (batch_size, model.output_layers[-1].out_features)


def test_hidden_state_initialization(model):
    """Test LSTM hidden state initialization."""
    batch_size = 2
    device = torch.device("cpu")
    
    hidden = model._init_hidden(batch_size, device)
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2
    
    num_directions = 2 if model.bidirectional else 1
    expected_shape = (
        model.num_lstm_layers * num_directions,
        batch_size,
        model.lstm_hidden_dim,
    )
    
    assert hidden[0].shape == expected_shape  # hidden state
    assert hidden[1].shape == expected_shape  # cell state


def test_wave_representations(model):
    """Test getting intermediate wave representations."""
    batch_size = 2
    seq_length = 10
    x = torch.randint(0, 30000, (batch_size, seq_length))
    
    wave_output = model.get_wave_representations(x)
    assert wave_output.shape == (batch_size, seq_length, model.hidden_dim)


def test_with_provided_hidden_state(model):
    """Test forward pass with provided initial hidden state."""
    batch_size = 2
    seq_length = 10
    x = torch.randint(0, 30000, (batch_size, seq_length))
    
    # Get initial hidden state
    hidden = model._init_hidden(batch_size, x.device)
    
    # Forward pass with provided hidden state
    output, new_hidden = model(x, hidden=hidden)
    
    assert output.shape == (batch_size, model.output_layers[-1].out_features)
    assert isinstance(new_hidden, tuple)
    assert len(new_hidden) == 2
    
    # Hidden state should be updated
    assert not torch.equal(hidden[0], new_hidden[0])
    assert not torch.equal(hidden[1], new_hidden[1])
