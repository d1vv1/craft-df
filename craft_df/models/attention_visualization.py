"""
Attention Visualization and Interpretability Module for CRAFT-DF

This module provides comprehensive visualization and interpretability tools for the
cross-attention mechanism in the CRAFT-DF model. It includes utilities for:
- Attention weight visualization and analysis
- Feature importance analysis
- Attention pattern interpretation
- Statistical analysis of attention distributions
- Export functionality for external visualization tools

The visualization tools help researchers and practitioners understand:
1. Which frequency patterns the model focuses on for different spatial inputs
2. How attention patterns change across different samples and classes
3. The stability and consistency of attention mechanisms
4. Potential biases or failure modes in the attention mechanism
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


@dataclass
class AttentionAnalysis:
    """
    Data class for storing comprehensive attention analysis results.
    
    Attributes:
        sample_id: Identifier for the analyzed sample
        attention_weights: Raw attention weights (num_heads, seq_len, seq_len)
        attention_stats: Statistical measures of attention distribution
        entropy_scores: Entropy measures for attention concentration
        head_similarities: Similarity measures between attention heads
        feature_importance: Importance scores for different features
        metadata: Additional metadata about the analysis
    """
    sample_id: str
    attention_weights: np.ndarray
    attention_stats: Dict[str, float]
    entropy_scores: Dict[str, float]
    head_similarities: np.ndarray
    feature_importance: Dict[str, float]
    metadata: Dict[str, Any]


class AttentionVisualizer:
    """
    Comprehensive attention visualization and analysis toolkit.
    
    This class provides methods for visualizing and analyzing attention patterns
    in the CRAFT-DF cross-attention mechanism. It supports various visualization
    formats and statistical analyses to help understand model behavior.
    
    Args:
        model: The CrossAttentionFusion model to analyze
        save_dir: Directory to save visualization outputs
        figsize: Default figure size for plots
        dpi: Resolution for saved figures
        style: Matplotlib style for plots
    """
    
    def __init__(
        self,
        model: 'CrossAttentionFusion',
        save_dir: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        style: str = 'seaborn-v0_8'
    ) -> None:
        self.model = model
        self.save_dir = Path(save_dir) if save_dir else Path('./attention_analysis')
        self.figsize = figsize
        self.dpi = dpi
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except OSError:
            logger.warning(f"Style '{style}' not available, using default")
            plt.style.use('default')
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analysis cache
        self.analysis_cache: Dict[str, AttentionAnalysis] = {}
        
        logger.info(f"AttentionVisualizer initialized with save_dir: {self.save_dir}")
    
    def analyze_attention_pattern(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
        sample_ids: Optional[List[str]] = None,
        return_raw: bool = False
    ) -> Union[List[AttentionAnalysis], Dict[str, torch.Tensor]]:
        """
        Perform comprehensive analysis of attention patterns.
        
        Args:
            spatial_features: Spatial domain features (batch_size, spatial_dim)
            frequency_features: Frequency domain features (batch_size, frequency_dim)
            sample_ids: Optional identifiers for samples
            return_raw: Whether to return raw tensors instead of analysis objects
            
        Returns:
            List of AttentionAnalysis objects or raw tensor dictionary
        """
        batch_size = spatial_features.shape[0]
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(batch_size)]
        
        # Get attention weights and other intermediate outputs
        with torch.no_grad():
            self.model.eval()
            
            # Get attention weights
            attention_weights = self.model.get_attention_weights(spatial_features, frequency_features)
            # Shape: (batch_size, num_heads, 1, 1)
            
            # Get fused features for feature importance analysis
            fused_features, _ = self.model(spatial_features, frequency_features)
            
            if return_raw:
                return {
                    'attention_weights': attention_weights,
                    'spatial_features': spatial_features,
                    'frequency_features': frequency_features,
                    'fused_features': fused_features
                }
        
        analyses = []
        for i, sample_id in enumerate(sample_ids):
            # Extract attention weights for this sample
            sample_attention = attention_weights[i].cpu().numpy()  # (num_heads, 1, 1)
            
            # Compute attention statistics
            attention_stats = self._compute_attention_statistics(sample_attention)
            
            # Compute entropy scores
            entropy_scores = self._compute_entropy_scores(sample_attention)
            
            # Compute head similarities
            head_similarities = self._compute_head_similarities(sample_attention)
            
            # Compute feature importance (simplified for single query-key pair)
            feature_importance = self._compute_feature_importance(
                spatial_features[i:i+1], frequency_features[i:i+1], fused_features[i:i+1]
            )
            
            # Create metadata
            metadata = {
                'batch_index': i,
                'num_heads': self.model.num_heads,
                'spatial_dim': self.model.spatial_dim,
                'frequency_dim': self.model.frequency_dim,
                'embed_dim': self.model.embed_dim,
                'spatial_norm': torch.norm(spatial_features[i]).item(),
                'frequency_norm': torch.norm(frequency_features[i]).item(),
                'fused_norm': torch.norm(fused_features[i]).item()
            }
            
            # Create analysis object
            analysis = AttentionAnalysis(
                sample_id=sample_id,
                attention_weights=sample_attention,
                attention_stats=attention_stats,
                entropy_scores=entropy_scores,
                head_similarities=head_similarities,
                feature_importance=feature_importance,
                metadata=metadata
            )
            
            analyses.append(analysis)
            self.analysis_cache[sample_id] = analysis
        
        return analyses
    
    def _compute_attention_statistics(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """Compute statistical measures of attention distribution."""
        # Flatten attention weights across heads
        flat_attention = attention_weights.flatten()
        
        # Handle edge cases for numerical stability
        if len(flat_attention) == 0:
            return {stat: 0.0 for stat in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'skewness', 'kurtosis', 'range']}
        
        # Compute basic statistics
        mean_val = float(np.mean(flat_attention))
        std_val = float(np.std(flat_attention))
        min_val = float(np.min(flat_attention))
        max_val = float(np.max(flat_attention))
        median_val = float(np.median(flat_attention))
        q25_val = float(np.percentile(flat_attention, 25))
        q75_val = float(np.percentile(flat_attention, 75))
        range_val = max_val - min_val
        
        # Compute skewness and kurtosis with error handling
        try:
            if std_val > 1e-8:  # Only compute if there's sufficient variance
                skewness_val = float(stats.skew(flat_attention))
                kurtosis_val = float(stats.kurtosis(flat_attention))
            else:
                skewness_val = 0.0
                kurtosis_val = 0.0
        except (RuntimeWarning, FloatingPointError):
            # Handle numerical instability
            skewness_val = 0.0
            kurtosis_val = 0.0
        
        # Ensure all values are finite
        result = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'median': median_val,
            'q25': q25_val,
            'q75': q75_val,
            'skewness': skewness_val if np.isfinite(skewness_val) else 0.0,
            'kurtosis': kurtosis_val if np.isfinite(kurtosis_val) else 0.0,
            'range': range_val
        }
        
        return result
    
    def _compute_entropy_scores(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """Compute entropy measures for attention concentration."""
        # Normalize attention weights to proper probabilities
        total_sum = np.sum(attention_weights)
        if total_sum <= 1e-8:
            # Handle zero or near-zero attention weights
            return {
                'shannon_entropy': 0.0,
                'normalized_entropy': 0.0,
                'gini_coefficient': 0.0,
                'concentration_ratio': 1.0
            }
        
        normalized_attention = attention_weights / total_sum
        
        # Add small epsilon to prevent log(0)
        eps = 1e-8
        normalized_attention = normalized_attention + eps
        
        # Renormalize after adding epsilon
        normalized_attention = normalized_attention / np.sum(normalized_attention)
        
        # Compute Shannon entropy
        shannon_entropy = -np.sum(normalized_attention * np.log(normalized_attention))
        
        # Compute normalized entropy (0 to 1 scale)
        max_entropy = np.log(normalized_attention.size)
        if max_entropy > 1e-8:
            normalized_entropy = shannon_entropy / max_entropy
            # Clamp to [0, 1] to handle floating point precision issues
            normalized_entropy = max(0.0, min(1.0, normalized_entropy))
        else:
            normalized_entropy = 0.0
        
        # Compute Gini coefficient (measure of inequality)
        sorted_attention = np.sort(normalized_attention.flatten())
        n = len(sorted_attention)
        if n > 1:
            cumsum = np.cumsum(sorted_attention)
            if cumsum[-1] > 1e-8:
                gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
                gini = max(0.0, min(1.0, gini))  # Clamp to [0, 1]
            else:
                gini = 0.0
        else:
            gini = 0.0
        
        # Compute concentration ratio
        mean_attention = np.mean(normalized_attention)
        if mean_attention > 1e-8:
            concentration_ratio = np.max(normalized_attention) / mean_attention
        else:
            concentration_ratio = 1.0
        
        return {
            'shannon_entropy': float(shannon_entropy),
            'normalized_entropy': float(normalized_entropy),
            'gini_coefficient': float(gini),
            'concentration_ratio': float(concentration_ratio)
        }
    
    def _compute_head_similarities(self, attention_weights: np.ndarray) -> np.ndarray:
        """Compute similarity matrix between attention heads."""
        num_heads = attention_weights.shape[0]
        similarities = np.zeros((num_heads, num_heads))
        
        for i in range(num_heads):
            for j in range(num_heads):
                # Compute cosine similarity between heads
                head_i = attention_weights[i].flatten()
                head_j = attention_weights[j].flatten()
                
                # Handle zero vectors
                norm_i = np.linalg.norm(head_i)
                norm_j = np.linalg.norm(head_j)
                
                if norm_i > 1e-8 and norm_j > 1e-8:
                    similarities[i, j] = np.dot(head_i, head_j) / (norm_i * norm_j)
                else:
                    similarities[i, j] = 1.0 if i == j else 0.0
        
        return similarities
    
    def _compute_feature_importance(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
        fused_features: torch.Tensor
    ) -> Dict[str, float]:
        """Compute feature importance scores using gradient-based methods."""
        # Enable gradients for input features
        spatial_features = spatial_features.clone().detach().requires_grad_(True)
        frequency_features = frequency_features.clone().detach().requires_grad_(True)
        
        # Forward pass
        output, _ = self.model(spatial_features, frequency_features)
        
        # Compute gradients with respect to output norm
        output_norm = torch.norm(output)
        output_norm.backward()
        
        # Compute importance scores
        spatial_importance = torch.norm(spatial_features.grad).item() if spatial_features.grad is not None else 0.0
        frequency_importance = torch.norm(frequency_features.grad).item() if frequency_features.grad is not None else 0.0
        
        # Normalize importance scores
        total_importance = spatial_importance + frequency_importance
        if total_importance > 1e-8:
            spatial_importance /= total_importance
            frequency_importance /= total_importance
        
        return {
            'spatial_importance': spatial_importance,
            'frequency_importance': frequency_importance,
            'importance_ratio': spatial_importance / (frequency_importance + 1e-8)
        }
    
    def plot_attention_heatmap(
        self,
        analysis: AttentionAnalysis,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Create heatmap visualization of attention weights across heads.
        
        Args:
            analysis: AttentionAnalysis object containing attention data
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        num_heads = analysis.attention_weights.shape[0]
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        fig.suptitle(f'Attention Heatmap - {analysis.sample_id}', fontsize=16)
        
        # Handle single subplot case
        if num_heads == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot attention weights for each head
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            
            # Get attention weights for this head
            head_attention = analysis.attention_weights[head_idx]  # (1, 1)
            
            # Create a simple visualization (since we have single query-key pair)
            im = ax.imshow([[head_attention[0, 0]]], cmap='Blues', aspect='auto')
            ax.set_title(f'Head {head_idx + 1}')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar to the side of the figure instead of individual subplots
            # This reduces the total number of axes
        
        # Hide unused subplots
        for head_idx in range(num_heads, len(axes)):
            axes[head_idx].set_visible(False)
        
        # Add a single colorbar for all subplots
        if num_heads > 0:
            # Create colorbar on the right side
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            # Use the last image for colorbar reference
            fig.colorbar(im, cax=cbar_ax)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)  # Make room for colorbar
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        elif self.save_dir:
            save_path = self.save_dir / f'attention_heatmap_{analysis.sample_id}.png'
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_attention_statistics(
        self,
        analyses: List[AttentionAnalysis],
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Create statistical visualization of attention patterns across samples.
        
        Args:
            analyses: List of AttentionAnalysis objects
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Attention Statistics Across Samples', fontsize=16)
        
        # Extract statistics
        stats_data = {
            'mean': [a.attention_stats['mean'] for a in analyses],
            'std': [a.attention_stats['std'] for a in analyses],
            'entropy': [a.entropy_scores['shannon_entropy'] for a in analyses],
            'gini': [a.entropy_scores['gini_coefficient'] for a in analyses],
            'spatial_importance': [a.feature_importance['spatial_importance'] for a in analyses],
            'frequency_importance': [a.feature_importance['frequency_importance'] for a in analyses]
        }
        
        # Plot distributions
        axes[0, 0].hist(stats_data['mean'], bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Attention Mean Distribution')
        axes[0, 0].set_xlabel('Mean Attention Weight')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(stats_data['std'], bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Attention Std Distribution')
        axes[0, 1].set_xlabel('Std Attention Weight')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[0, 2].hist(stats_data['entropy'], bins=20, alpha=0.7, color='red')
        axes[0, 2].set_title('Attention Entropy Distribution')
        axes[0, 2].set_xlabel('Shannon Entropy')
        axes[0, 2].set_ylabel('Frequency')
        
        axes[1, 0].hist(stats_data['gini'], bins=20, alpha=0.7, color='purple')
        axes[1, 0].set_title('Gini Coefficient Distribution')
        axes[1, 0].set_xlabel('Gini Coefficient')
        axes[1, 0].set_ylabel('Frequency')
        
        # Feature importance comparison
        axes[1, 1].scatter(stats_data['spatial_importance'], stats_data['frequency_importance'], alpha=0.6)
        axes[1, 1].set_title('Feature Importance Comparison')
        axes[1, 1].set_xlabel('Spatial Importance')
        axes[1, 1].set_ylabel('Frequency Importance')
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Diagonal line
        
        # Box plot of attention statistics
        box_data = [stats_data['mean'], stats_data['std'], stats_data['entropy']]
        axes[1, 2].boxplot(box_data, tick_labels=['Mean', 'Std', 'Entropy'])
        axes[1, 2].set_title('Attention Statistics Summary')
        axes[1, 2].set_ylabel('Value')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        elif self.save_dir:
            save_path = self.save_dir / 'attention_statistics.png'
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_head_similarity_matrix(
        self,
        analysis: AttentionAnalysis,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Create visualization of attention head similarity matrix.
        
        Args:
            analysis: AttentionAnalysis object containing similarity data
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(analysis.head_similarities, cmap='RdYlBu_r', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
        
        # Set labels
        ax.set_title(f'Attention Head Similarity Matrix - {analysis.sample_id}')
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Head Index')
        
        # Set ticks
        num_heads = analysis.head_similarities.shape[0]
        ax.set_xticks(range(num_heads))
        ax.set_yticks(range(num_heads))
        ax.set_xticklabels([f'H{i+1}' for i in range(num_heads)])
        ax.set_yticklabels([f'H{i+1}' for i in range(num_heads)])
        
        # Add text annotations
        for i in range(num_heads):
            for j in range(num_heads):
                text = ax.text(j, i, f'{analysis.head_similarities[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        elif self.save_dir:
            save_path = self.save_dir / f'head_similarity_{analysis.sample_id}.png'
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def export_analysis_data(
        self,
        analyses: List[AttentionAnalysis],
        export_path: Optional[str] = None,
        format: str = 'json'
    ) -> str:
        """
        Export attention analysis data to file.
        
        Args:
            analyses: List of AttentionAnalysis objects to export
            export_path: Optional path for export file
            format: Export format ('json', 'csv', 'npz')
            
        Returns:
            Path to exported file
        """
        if export_path is None:
            export_path = self.save_dir / f'attention_analysis.{format}'
        else:
            export_path = Path(export_path)
        
        if format == 'json':
            # Convert to JSON-serializable format
            export_data = []
            for analysis in analyses:
                data = asdict(analysis)
                # Convert numpy arrays to lists
                data['attention_weights'] = analysis.attention_weights.tolist()
                data['head_similarities'] = analysis.head_similarities.tolist()
                export_data.append(data)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format == 'npz':
            # Save as numpy archive
            save_dict = {}
            for i, analysis in enumerate(analyses):
                prefix = f'sample_{i}_'
                save_dict[f'{prefix}attention_weights'] = analysis.attention_weights
                save_dict[f'{prefix}head_similarities'] = analysis.head_similarities
                save_dict[f'{prefix}metadata'] = np.array([analysis.metadata], dtype=object)
            
            np.savez_compressed(export_path, **save_dict)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Analysis data exported to: {export_path}")
        return str(export_path)
    
    def generate_report(
        self,
        analyses: List[AttentionAnalysis],
        report_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive HTML report of attention analysis.
        
        Args:
            analyses: List of AttentionAnalysis objects
            report_path: Optional path for report file
            
        Returns:
            Path to generated report
        """
        if report_path is None:
            report_path = self.save_dir / 'attention_report.html'
        else:
            report_path = Path(report_path)
        
        # Generate summary statistics
        all_entropies = [a.entropy_scores['shannon_entropy'] for a in analyses]
        all_ginis = [a.entropy_scores['gini_coefficient'] for a in analyses]
        all_spatial_importance = [a.feature_importance['spatial_importance'] for a in analyses]
        
        summary_stats = {
            'num_samples': len(analyses),
            'avg_entropy': np.mean(all_entropies),
            'std_entropy': np.std(all_entropies),
            'avg_gini': np.mean(all_ginis),
            'avg_spatial_importance': np.mean(all_spatial_importance),
            'num_heads': analyses[0].metadata['num_heads'] if analyses else 0
        }
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CRAFT-DF Attention Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .stats-table {{ border-collapse: collapse; width: 100%; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .stats-table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CRAFT-DF Cross-Attention Analysis Report</h1>
                <p>Generated on: {np.datetime64('now')}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <table class="stats-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Number of Samples</td><td>{summary_stats['num_samples']}</td></tr>
                    <tr><td>Number of Attention Heads</td><td>{summary_stats['num_heads']}</td></tr>
                    <tr><td>Average Attention Entropy</td><td>{summary_stats['avg_entropy']:.4f}</td></tr>
                    <tr><td>Entropy Standard Deviation</td><td>{summary_stats['std_entropy']:.4f}</td></tr>
                    <tr><td>Average Gini Coefficient</td><td>{summary_stats['avg_gini']:.4f}</td></tr>
                    <tr><td>Average Spatial Importance</td><td>{summary_stats['avg_spatial_importance']:.4f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Individual Sample Analysis</h2>
        """
        
        # Add individual sample details
        for analysis in analyses[:10]:  # Limit to first 10 samples for report size
            html_content += f"""
                <h3>Sample: {analysis.sample_id}</h3>
                <p><strong>Attention Statistics:</strong></p>
                <ul>
                    <li>Mean: {analysis.attention_stats['mean']:.4f}</li>
                    <li>Standard Deviation: {analysis.attention_stats['std']:.4f}</li>
                    <li>Shannon Entropy: {analysis.entropy_scores['shannon_entropy']:.4f}</li>
                    <li>Gini Coefficient: {analysis.entropy_scores['gini_coefficient']:.4f}</li>
                </ul>
                <p><strong>Feature Importance:</strong></p>
                <ul>
                    <li>Spatial: {analysis.feature_importance['spatial_importance']:.4f}</li>
                    <li>Frequency: {analysis.feature_importance['frequency_importance']:.4f}</li>
                </ul>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Analysis report generated: {report_path}")
        return str(report_path)
    
    def validate_attention_stability(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
        num_runs: int = 10,
        noise_std: float = 0.01
    ) -> Dict[str, float]:
        """
        Validate stability of attention patterns under small input perturbations.
        
        Args:
            spatial_features: Input spatial features
            frequency_features: Input frequency features
            num_runs: Number of perturbation runs
            noise_std: Standard deviation of Gaussian noise to add
            
        Returns:
            Dictionary containing stability metrics
        """
        self.model.eval()
        
        # Get baseline attention
        with torch.no_grad():
            baseline_attention = self.model.get_attention_weights(spatial_features, frequency_features)
        
        # Run perturbation tests
        attention_variations = []
        
        for run in range(num_runs):
            # Add small random noise
            spatial_noise = torch.randn_like(spatial_features) * noise_std
            frequency_noise = torch.randn_like(frequency_features) * noise_std
            
            perturbed_spatial = spatial_features + spatial_noise
            perturbed_frequency = frequency_features + frequency_noise
            
            with torch.no_grad():
                perturbed_attention = self.model.get_attention_weights(perturbed_spatial, perturbed_frequency)
            
            # Compute difference from baseline
            attention_diff = torch.abs(perturbed_attention - baseline_attention)
            attention_variations.append(attention_diff.cpu().numpy())
        
        # Compute stability metrics
        variations_array = np.array(attention_variations)
        
        stability_metrics = {
            'mean_variation': float(np.mean(variations_array)),
            'std_variation': float(np.std(variations_array)),
            'max_variation': float(np.max(variations_array)),
            'stability_score': float(1.0 / (1.0 + np.mean(variations_array))),  # Higher is more stable
            'consistency_ratio': float(np.mean(variations_array < 0.1))  # Fraction of small variations
        }
        
        logger.info(f"Attention stability analysis completed: stability_score={stability_metrics['stability_score']:.4f}")
        
        return stability_metrics