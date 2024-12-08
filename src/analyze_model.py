import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os


def load_and_analyze_weights(model_path):
    """
    Load the wave network weights and extract complex vector components.

    Args:
        model_path (str): Path to the PyTorch model weights

    Returns:
        dict: Model state dictionary with weights
    """
    # Load model weights safely with weights_only=True
    weights = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=True
    )

    # Print available keys to help debug
    print("Available keys in model state dict:", weights.keys())
    return weights


def plot_magnitude_distribution(G, save_path=None):
    """Plot the distribution of magnitudes in the global semantic vector."""
    plt.figure(figsize=(12, 6))

    # Plot histogram of magnitudes
    sns.histplot(G.flatten(), bins=50, kde=True)
    plt.title("Distribution of Global Semantic Vector Magnitudes")
    plt.xlabel("Magnitude")
    plt.ylabel("Count")

    if save_path:
        plt.savefig(f"{save_path}/magnitude_dist.png")
    plt.close()


def plot_phase_relationships(alpha, save_path=None):
    """Visualize phase relationships between tokens."""
    plt.figure(figsize=(12, 6))

    # Create polar plot
    ax = plt.subplot(111, projection="polar")
    theta = np.linspace(0, 2 * np.pi, len(alpha))
    ax.scatter(theta, alpha)
    plt.title("Phase Relationships")

    if save_path:
        plt.savefig(f"{save_path}/phase_relationships.png")
    plt.close()


def plot_complex_plane(G, alpha, num_points=1000, save_path=None):
    """
    Visualize vectors in the complex plane.

    Args:
        G (np.ndarray): Magnitude vectors
        alpha (np.ndarray): Phase vectors
        num_points (int): Number of points to plot (to avoid overcrowding)
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 12))

    # Ensure G and alpha are compatible shapes
    if len(G.shape) > 1:
        G = G.flatten()
    if len(alpha.shape) > 1:
        alpha = alpha.flatten()

    # Take min length to avoid index errors
    min_len = min(len(G), len(alpha))

    # Sample points if we have more than num_points
    if min_len > num_points:
        indices = np.random.choice(min_len, num_points, replace=False)
        G = G[indices]
        alpha = alpha[indices]

    # Convert to complex numbers
    Z = G * np.exp(1j * alpha)

    # Plot points in complex plane
    plt.scatter(Z.real, Z.imag, alpha=0.5)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    plt.title("Complex Vector Representations")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")

    # Make plot square
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

    # Interference pattern
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


def analyze_wave_network(model_path, save_dir="wave_analysis"):
    """
    Main analysis function.

    Args:
        model_path (str): Path to the model weights
        save_dir (str): Directory to save analysis plots
    """
    # Load weights
    print("Loading model weights...")
    weights = load_and_analyze_weights(model_path)

    # Extract components based on actual model structure
    if "state_dict" in weights:
        weights = weights["state_dict"]

    # Based on the printed keys, we can now use the exact keys
    G = weights["global_semantics.weight"].numpy()
    alpha = weights["phase_vector.weight"].numpy()

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create visualizations
    print("Creating visualizations...")
    plot_magnitude_distribution(G, save_dir)
    plot_phase_relationships(alpha, save_dir)
    plot_complex_plane(G, alpha, save_dir)

    # For wave interference, ensure we have at least 2 vectors
    if len(G) >= 2 and len(alpha) >= 2:
        Z1 = G[0] * np.exp(1j * alpha[0])
        Z2 = G[1] * np.exp(1j * alpha[1])
        plot_wave_interference(Z1, Z2, save_dir)
    else:
        print("Not enough vectors for interference analysis")

    print("Analysis complete! Visualizations saved to:", save_dir)


if __name__ == "__main__":
    model_path = (
        "/Users/josephblazick/Documents/quantum_net/models/wave_network_best.pth"
    )
    analyze_wave_network(model_path)
