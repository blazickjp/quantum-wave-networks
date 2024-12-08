import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class SemanticProbe:
    def __init__(self, model, save_dir="wave_analysis/probing"):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_representations(self, input_ids, layer_name):
        """Extract representations from a specific layer"""
        representations = {}
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                representations['output'] = output[0].detach()
            else:
                representations['output'] = output.detach()
            
        # Register hook for the specified layer
        for name, module in self.model.named_modules():
            if layer_name in name:
                hook = module.register_forward_hook(hook_fn)
                break
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids)
            
        # Remove hook
        hook.remove()
        
        return representations['output']

    def train_probe(self, embeddings, labels, test_size=0.2):
        """Train a linear probe on the embeddings"""
        # Convert to numpy arrays
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
            
        # Split into train and test
        n_samples = len(embeddings)
        n_test = int(n_samples * test_size)
        indices = np.random.permutation(n_samples)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = embeddings[train_indices]
        y_train = labels[train_indices]
        X_test = embeddings[test_indices]
        y_test = labels[test_indices]
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        
        # Confusion matrix
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'Confusion Matrix\nTest Accuracy: {test_acc:.3f}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classifier': clf,
            'confusion_matrix': cm
        }

    def analyze_phase_separation(self, embeddings, labels):
        """Analyze how well phase separates different classes"""
        # Extract phase from complex representations
        if isinstance(embeddings, tuple):
            real, imag = embeddings
        else:
            split_size = embeddings.size(-1) // 2
            real, imag = torch.split(embeddings, split_size, dim=-1)
        
        phase = torch.atan2(imag, real).cpu().numpy()
        
        # Compute class-wise phase statistics
        unique_labels = np.unique(labels)
        phase_stats = {}
        
        plt.figure(figsize=(12, 6))
        
        for label in unique_labels:
            mask = labels == label
            class_phase = phase[mask]
            
            # Compute statistics
            phase_stats[label] = {
                'mean': np.mean(class_phase),
                'std': np.std(class_phase),
                'median': np.median(class_phase)
            }
            
            # Plot phase distribution for each class
            plt.hist(class_phase.flatten(), bins=50, alpha=0.5, 
                    label=f'Class {label}', density=True)
        
        plt.xlabel('Phase')
        plt.ylabel('Density')
        plt.title('Phase Distribution by Class')
        plt.legend()
        plt.savefig(self.save_dir / 'phase_distribution.png')
        plt.close()
        
        return phase_stats

    def analyze_magnitude_patterns(self, embeddings, labels):
        """Analyze patterns in magnitude across different classes"""
        # Extract magnitude from complex representations
        if isinstance(embeddings, tuple):
            real, imag = embeddings
        else:
            split_size = embeddings.size(-1) // 2
            real, imag = torch.split(embeddings, split_size, dim=-1)
        
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8).cpu().numpy()
        
        # Compute class-wise magnitude statistics
        unique_labels = np.unique(labels)
        magnitude_stats = {}
        
        plt.figure(figsize=(12, 6))
        
        for label in unique_labels:
            mask = labels == label
            class_magnitude = magnitude[mask]
            
            # Compute statistics
            magnitude_stats[label] = {
                'mean': np.mean(class_magnitude),
                'std': np.std(class_magnitude),
                'median': np.median(class_magnitude)
            }
            
            # Plot magnitude distribution for each class
            plt.hist(class_magnitude.flatten(), bins=50, alpha=0.5, 
                    label=f'Class {label}', density=True)
        
        plt.xlabel('Magnitude')
        plt.ylabel('Density')
        plt.title('Magnitude Distribution by Class')
        plt.legend()
        plt.savefig(self.save_dir / 'magnitude_distribution.png')
        plt.close()
        
        return magnitude_stats
