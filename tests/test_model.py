"""Tests for Wave Network model."""

import torch
import pytest
from quantum_net.model import WaveNetwork


@pytest.fixture
def model():
    """Create a model instance for testing."""
    return WaveNetwork(
        input_dim=768,
        hidden_dim=64,
        output_dim=4,
        num_layers=1,
    )


def test_model_initialization(model):
    """Test model initialization."""
    assert isinstance(model, WaveNetwork)
    assert model.input_dim == 768
    assert model.hidden_dim == 64
    assert model.output_dim == 4


def test_forward_pass(model):
    """Test model forward pass."""
    batch_size = 2
    seq_length = 10
    x = torch.randint(0, 30000, (batch_size, seq_length))
    
    # Test with modulation
    output = model(x, use_modulation=True)
    assert output.shape == (batch_size, model.output_dim)
    
    # Test with interference
    output = model(x, use_modulation=False)
    assert output.shape == (batch_size, model.output_dim)


def test_model_statistics(model):
    """Test model statistics computation."""
    stats = model.log_model_statistics()
    
    assert isinstance(stats, dict)
    assert "Total Parameters" in stats
    assert "Embedding Parameters" in stats
    assert "Wave Operation Parameters" in stats
    assert "Feed-forward Parameters" in stats
    
    # Check that parameters add up
    component_params = (
        stats["Embedding Parameters"] 
        + stats["Wave Operation Parameters"]
        + stats["Feed-forward Parameters"]
    )
    assert stats["Total Parameters"] >= component_params


def test_global_semantics(model):
    """Test global semantics computation."""
    batch_size = 2
    seq_length = 10
    input_dim = 768
    x = torch.randn(batch_size, seq_length, input_dim)
    
    G = model.compute_global_semantics(x)
    assert G.shape == (batch_size, 1, input_dim)
    assert torch.all(G >= 0)  # Magnitudes should be non-negative


def test_phase_computation(model):
    """Test phase computation."""
    batch_size = 2
    seq_length = 10
    hidden_dim = 64
    
    x = torch.randn(batch_size, seq_length, hidden_dim)
    G = torch.ones(batch_size, 1, hidden_dim)  # Mock global semantics
    
    phase = model.compute_phase(x, G)
    assert phase.shape == (batch_size, seq_length, hidden_dim)
    assert torch.all(phase >= -torch.pi) and torch.all(phase <= torch.pi)
