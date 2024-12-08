import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import torch.nn.functional as F


class WaveDynamicsAnalyzer:
    def __init__(self, model, save_dir="wave_analysis/dynamics"):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Register hooks for collecting intermediate outputs
        self.layer_outputs = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to collect outputs from wave operation layers"""
        def hook_fn(module, input, output):
            self.layer_outputs.append(output.detach())

        # Register hooks for relevant layers
        for name, module in self.model.named_modules():
            if any(x in name for x in ['wave_variant1', 'wave_variant2', 'global_semantics']):
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)

    def extract_wave_components(self, tensor):
        """Extract magnitude and phase from complex representation"""
        if isinstance(tensor, tuple):
            real, imag = tensor
        else:
            # Assume first half is real, second half is imaginary
            split_size = tensor.size(-1) // 2
            real, imag = torch.split(tensor, split_size, dim=-1)
        
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real)
        return magnitude, phase

    def visualize_wave_evolution(self, input_ids, labels=None):
        """Visualize how wave representations evolve through the network"""
        # Forward pass to collect layer outputs
        self.layer_outputs = []
        with torch.no_grad():
            _ = self.model(input_ids)

        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, len(self.layer_outputs))

        # Plot magnitude and phase evolution
        for i, layer_output in enumerate(self.layer_outputs):
            magnitude, phase = self.extract_wave_components(layer_output)
            
            # Magnitude distribution
            ax1 = fig.add_subplot(gs[0, i])
            sns.histplot(magnitude.flatten().cpu().numpy(), bins=50, ax=ax1)
            ax1.set_title(f'Layer {i+1} Magnitude')
            ax1.set_yscale('log')
            
            # Phase distribution
            ax2 = fig.add_subplot(gs[1, i], projection='polar')
            phase_hist, bins = np.histogram(phase.cpu().numpy(), bins=50)
            ax2.plot(bins[:-1], phase_hist)
            ax2.set_title(f'Layer {i+1} Phase')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'wave_evolution.png')
        plt.close()

    def compute_pairwise_phase_differences(self, embeddings):
        """Compute phase differences between all pairs of embeddings"""
        _, phase = self.extract_wave_components(embeddings)
        phase = phase.cpu().numpy()
        
        # Compute pairwise differences
        n_samples = phase.shape[0]
        differences = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Compute circular difference
                diff = np.abs(np.angle(np.exp(1j * (phase[i] - phase[j]))))
                differences[i,j] = diff
                differences[j,i] = diff
                
        return differences

    def compute_semantic_similarity(self, labels):
        """Compute semantic similarity based on labels"""
        n_samples = len(labels)
        similarity = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Simple similarity metric: 1 if same label, 0 if different
                sim = 1.0 if labels[i] == labels[j] else 0.0
                similarity[i,j] = sim
                similarity[j,i] = sim
                
        return similarity

    def analyze_semantic_correlations(self, embeddings, labels):
        """Analyze correlation between phase relationships and semantic similarity"""
        phase_diffs = self.compute_pairwise_phase_differences(embeddings)
        semantic_sims = self.compute_semantic_similarity(labels)
        
        # Flatten the matrices (excluding diagonal)
        mask = ~np.eye(phase_diffs.shape[0], dtype=bool)
        phase_diffs_flat = phase_diffs[mask]
        semantic_sims_flat = semantic_sims[mask]
        
        # Compute correlation
        correlation, p_value = stats.pearsonr(phase_diffs_flat, semantic_sims_flat)
        
        # Visualization
        plt.figure(figsize=(10, 5))
        
        # Scatter plot
        plt.subplot(121)
        plt.scatter(phase_diffs_flat, semantic_sims_flat, alpha=0.1)
        plt.xlabel('Phase Difference')
        plt.ylabel('Semantic Similarity')
        plt.title(f'Correlation: {correlation:.3f}\np-value: {p_value:.3e}')
        
        # Distribution comparison
        plt.subplot(122)
        plt.hist(phase_diffs_flat[semantic_sims_flat == 1], bins=50, alpha=0.5, 
                label='Same Class', density=True)
        plt.hist(phase_diffs_flat[semantic_sims_flat == 0], bins=50, alpha=0.5, 
                label='Different Class', density=True)
        plt.xlabel('Phase Difference')
        plt.ylabel('Density')
        plt.legend()
        plt.title('Phase Difference Distribution')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'semantic_correlation.png')
        plt.close()
        
        return correlation, p_value

    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
