import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
from model import WaveNetwork  # Import your WaveNetwork class from model.py


def load_model(model_path, vocab_size, embedding_dim, hidden_dim, output_dim):
    """Load the WaveNetwork model with the given weights."""
    model = WaveNetwork(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def compute_intermediates(model, x):
    """Compute intermediate complex vectors Z1 and Z2 using hooks."""
    hidden1_list = []
    hidden2_list = []

    def hook_hidden1(module, input, output):
        hidden1_list.append(output.detach().cpu().numpy())

    def hook_hidden2(module, input, output):
        hidden2_list.append(output.detach().cpu().numpy())

    # Attach hooks to wave_variant1 and wave_variant2
    handle1 = model.wave_variant1.register_forward_hook(hook_hidden1)
    handle2 = model.wave_variant2.register_forward_hook(hook_hidden2)

    # Pass input through the model
    with torch.no_grad():
        _ = model(x)

    # Remove hooks after use
    handle1.remove()
    handle2.remove()

    # Retrieve hidden1 and hidden2
    hidden1 = hidden1_list[0]  # (batch_size, seq_length, hidden_dim)
    hidden2 = hidden2_list[0]

    # Compute G1 and G2 (global semantics)
    G1 = np.linalg.norm(hidden1, ord=2, axis=1, keepdims=True)
    G2 = np.linalg.norm(hidden2, ord=2, axis=1, keepdims=True)

    # Compute alpha1 and alpha2 (phase vectors)
    def compute_phase_np(x, G):
        ratio = x / (G + 1e-8)
        numerator = np.sqrt(1 - ratio**2 + 1e-8)
        phase = np.arctan2(numerator, ratio)
        return phase

    alpha1 = compute_phase_np(hidden1, G1)
    alpha2 = compute_phase_np(hidden2, G2)

    # Compute complex vectors Z1 and Z2
    Z1 = G1 * np.exp(1j * alpha1)
    Z2 = G2 * np.exp(1j * alpha2)

    return Z1, Z2


def plot_magnitude_distribution(magnitudes, save_path=None):
    """Plot the distribution of magnitudes."""
    plt.figure(figsize=(12, 6))
    sns.histplot(magnitudes, bins=50, kde=True)
    plt.title("Distribution of Magnitudes")
    plt.xlabel("Magnitude")
    plt.ylabel("Count")
    if save_path:
        plt.savefig(f"{save_path}/magnitude_dist.png")
    plt.close()


def plot_phase_relationships(phases, save_path=None):
    """Visualize phase relationships in a polar plot."""
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111, projection="polar")
    theta = np.linspace(0, 2 * np.pi, len(phases))
    ax.scatter(theta, phases)
    plt.title("Phase Relationships")
    if save_path:
        plt.savefig(f"{save_path}/phase_relationships.png")
    plt.close()


def plot_complex_plane(Z, num_points=1000, save_path=None):
    """Visualize complex vectors in the complex plane."""
    plt.figure(figsize=(12, 12))
    if len(Z) > num_points:
        indices = np.random.choice(len(Z), num_points, replace=False)
        Z = Z[indices]
    plt.scatter(Z.real, Z.imag, alpha=0.5)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    plt.title("Complex Vector Representations")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.axis("equal")
    if save_path:
        plt.savefig(f"{save_path}/complex_plane.png")
    plt.close()


def plot_wave_interference(Z1, Z2, save_path=None):
    """Visualize wave interference between two complex vectors."""
    plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3)

    # Original vectors
    ax1 = plt.subplot(gs[0])
    ax1.scatter(Z1.real, Z1.imag, alpha=0.5, label="Vector 1")
    ax1.scatter(Z2.real, Z2.imag, alpha=0.5, label="Vector 2")
    ax1.set_title("Original Vectors")
    ax1.legend()

    # Interference pattern (sum of Z1 and Z2)
    Z_sum = Z1 + Z2
    ax2 = plt.subplot(gs[1])
    ax2.scatter(Z_sum.real, Z_sum.imag, alpha=0.5, c="red")
    ax2.set_title("Interference Pattern")

    # Magnitude comparison
    ax3 = plt.subplot(gs[2])
    ax3.hist(
        [np.abs(Z1), np.abs(Z2), np.abs(Z_sum)],
        label=["Vector 1", "Vector 2", "Combined"],
        alpha=0.5,
    )
    ax3.set_title("Magnitude Distribution")
    ax3.legend()

    if save_path:
        plt.savefig(f"{save_path}/wave_interference.png")
    plt.close()


def analyze_wave_network(
    model_path,
    vocab_size,
    embedding_dim,
    hidden_dim,
    output_dim,
    save_dir="wave_analysis",
):
    """Main function to analyze the Wave Network model."""
    # Load the model
    model = load_model(model_path, vocab_size, embedding_dim, hidden_dim, output_dim)

    # Prepare sample input (random token indices)
    batch_size = 1
    seq_length = 10  # Adjust based on your data
    x = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Compute intermediate complex vectors Z1 and Z2
    Z1, Z2 = compute_intermediates(model, x)

    # Flatten Z1 and Z2 for plotting
    Z1_flat = Z1.flatten()
    Z2_flat = Z2.flatten()

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create visualizations
    plot_magnitude_distribution(np.abs(Z1_flat), save_dir)
    plot_phase_relationships(np.angle(Z1_flat), save_dir)
    plot_complex_plane(Z1_flat, save_path=save_dir)
    plot_wave_interference(Z1_flat, Z2_flat, save_path=save_dir)

    print("Analysis complete! Visualizations saved to:", save_dir)


if __name__ == "__main__":
    model_path = "/Users/josephblazick/Documents/quantum_net/models/wave_network_interference.pth"
    vocab_size = 30000  # Adjust to your actual vocab size
    embedding_dim = 768
    hidden_dim = 64
    output_dim = 4  # Adjust based on your dataset (e.g., 4 for AG News)
    analyze_wave_network(model_path, vocab_size, embedding_dim, hidden_dim, output_dim)
