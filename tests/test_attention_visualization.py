"""
Unit tests for attention visualization and interpretability features.

This test suite validates the visualization tools, analysis methods,
and interpretability features for the cross-attention mechanism.
"""

import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import json
from unittest.mock import patch, MagicMock

# Import modules to test
from craft_df.models.cross_attention import CrossAttentionFusion
from craft_df.models.attention_visualization import AttentionVisualizer, AttentionAnalysis


class TestAttentionVisualization:
    """Test suite for attention visualization and analysis tools."""
    
    @pytest.fixture
    def device(self):
        """Get available device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        """Create CrossAttentionFusion model for testing."""
        model = CrossAttentionFusion(
            spatial_dim=1280,
            frequency_dim=512,
            embed_dim=512,
            num_heads=8,
            dropout_rate=0.1
        )
        return model.to(device)
    
    @pytest.fixture
    def sample_features(self, device):
        """Create sample features for testing."""
        batch_size = 4
        spatial_features = torch.randn(batch_size, 1280, device=device)
        frequency_features = torch.randn(batch_size, 512, device=device)
        return spatial_features, frequency_features
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def visualizer(self, model, temp_dir):
        """Create AttentionVisualizer instance."""
        return AttentionVisualizer(model, save_dir=temp_dir)
    
    def test_visualizer_initialization(self, model, temp_dir):
        """Test AttentionVisualizer initialization."""
        visualizer = AttentionVisualizer(model, save_dir=temp_dir)
        
        assert visualizer.model == model
        assert visualizer.save_dir == temp_dir
        assert visualizer.figsize == (12, 8)
        assert visualizer.dpi == 300
        assert isinstance(visualizer.analysis_cache, dict)
        assert len(visualizer.analysis_cache) == 0
    
    def test_analyze_attention_pattern(self, visualizer, sample_features):
        """Test attention pattern analysis."""
        spatial_features, frequency_features = sample_features
        
        # Test analysis with default sample IDs
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        assert len(analyses) == spatial_features.shape[0]
        assert all(isinstance(analysis, AttentionAnalysis) for analysis in analyses)
        
        # Check first analysis object
        analysis = analyses[0]
        assert analysis.sample_id == "sample_0"
        assert analysis.attention_weights.shape == (8, 1, 1)  # num_heads, 1, 1
        assert isinstance(analysis.attention_stats, dict)
        assert isinstance(analysis.entropy_scores, dict)
        assert isinstance(analysis.feature_importance, dict)
        assert isinstance(analysis.metadata, dict)
        
        # Test with custom sample IDs
        custom_ids = ["test_1", "test_2", "test_3", "test_4"]
        analyses_custom = visualizer.analyze_attention_pattern(
            spatial_features, frequency_features, sample_ids=custom_ids
        )
        
        assert len(analyses_custom) == 4
        assert analyses_custom[0].sample_id == "test_1"
        assert analyses_custom[1].sample_id == "test_2"
    
    def test_analyze_attention_pattern_raw_output(self, visualizer, sample_features):
        """Test attention pattern analysis with raw output."""
        spatial_features, frequency_features = sample_features
        
        raw_output = visualizer.analyze_attention_pattern(
            spatial_features, frequency_features, return_raw=True
        )
        
        assert isinstance(raw_output, dict)
        required_keys = ['attention_weights', 'spatial_features', 'frequency_features', 'fused_features']
        for key in required_keys:
            assert key in raw_output
            assert isinstance(raw_output[key], torch.Tensor)
        
        # Check shapes
        batch_size = spatial_features.shape[0]
        assert raw_output['attention_weights'].shape == (batch_size, 8, 1, 1)
        assert raw_output['fused_features'].shape == (batch_size, 512)
    
    def test_attention_statistics_computation(self, visualizer, sample_features):
        """Test attention statistics computation."""
        spatial_features, frequency_features = sample_features
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        analysis = analyses[0]
        stats = analysis.attention_stats
        
        # Check required statistics
        required_stats = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'skewness', 'kurtosis', 'range']
        for stat in required_stats:
            assert stat in stats
            assert isinstance(stats[stat], float)
            assert np.isfinite(stats[stat])
        
        # Check logical relationships
        assert stats['min'] <= stats['q25'] <= stats['median'] <= stats['q75'] <= stats['max']
        assert stats['range'] == stats['max'] - stats['min']
        assert stats['std'] >= 0
    
    def test_entropy_scores_computation(self, visualizer, sample_features):
        """Test entropy scores computation."""
        spatial_features, frequency_features = sample_features
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        analysis = analyses[0]
        entropy = analysis.entropy_scores
        
        # Check required entropy measures
        required_entropy = ['shannon_entropy', 'normalized_entropy', 'gini_coefficient', 'concentration_ratio']
        for measure in required_entropy:
            assert measure in entropy
            assert isinstance(entropy[measure], float)
            assert np.isfinite(entropy[measure])
        
        # Check value ranges
        assert entropy['shannon_entropy'] >= 0
        assert 0 <= entropy['normalized_entropy'] <= 1
        assert 0 <= entropy['gini_coefficient'] <= 1
        assert entropy['concentration_ratio'] >= 1
    
    def test_head_similarities_computation(self, visualizer, sample_features):
        """Test attention head similarity computation."""
        spatial_features, frequency_features = sample_features
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        analysis = analyses[0]
        similarities = analysis.head_similarities
        
        # Check shape and properties
        assert similarities.shape == (8, 8)  # num_heads x num_heads
        assert np.allclose(similarities, similarities.T)  # Should be symmetric
        assert np.allclose(np.diag(similarities), 1.0)  # Diagonal should be 1
        assert np.all(similarities >= -1) and np.all(similarities <= 1)  # Cosine similarity range
    
    def test_feature_importance_computation(self, visualizer, sample_features):
        """Test feature importance computation."""
        spatial_features, frequency_features = sample_features
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        analysis = analyses[0]
        importance = analysis.feature_importance
        
        # Check required importance measures
        required_importance = ['spatial_importance', 'frequency_importance', 'importance_ratio']
        for measure in required_importance:
            assert measure in importance
            assert isinstance(importance[measure], float)
            assert np.isfinite(importance[measure])
        
        # Check value ranges and relationships
        assert importance['spatial_importance'] >= 0
        assert importance['frequency_importance'] >= 0
        assert abs(importance['spatial_importance'] + importance['frequency_importance'] - 1.0) < 1e-6  # Should sum to 1
    
    @patch('matplotlib.pyplot.show')
    def test_plot_attention_heatmap(self, mock_show, visualizer, sample_features, temp_dir):
        """Test attention heatmap plotting."""
        spatial_features, frequency_features = sample_features
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        analysis = analyses[0]
        
        # Test plotting
        fig = visualizer.plot_attention_heatmap(analysis, show_plot=False)
        
        assert isinstance(fig, plt.Figure)
        # The figure should have 8 main subplot axes plus 1 colorbar axis
        # But we count only the main plot axes, not the colorbar
        main_axes = [ax for ax in fig.axes if ax.get_position().width > 0.1]  # Filter out narrow colorbar axes
        assert len(main_axes) == 8  # Should have 8 main subplots for 8 heads
        
        # Test saving
        save_path = temp_dir / 'test_heatmap.png'
        fig = visualizer.plot_attention_heatmap(analysis, save_path=str(save_path), show_plot=False)
        assert save_path.exists()
        
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_attention_statistics(self, mock_show, visualizer, sample_features, temp_dir):
        """Test attention statistics plotting."""
        spatial_features, frequency_features = sample_features
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        # Test plotting
        fig = visualizer.plot_attention_statistics(analyses, show_plot=False)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 6  # Should have 6 subplots
        
        # Test saving
        save_path = temp_dir / 'test_statistics.png'
        fig = visualizer.plot_attention_statistics(analyses, save_path=str(save_path), show_plot=False)
        assert save_path.exists()
        
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_head_similarity_matrix(self, mock_show, visualizer, sample_features, temp_dir):
        """Test head similarity matrix plotting."""
        spatial_features, frequency_features = sample_features
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        analysis = analyses[0]
        
        # Test plotting
        fig = visualizer.plot_head_similarity_matrix(analysis, show_plot=False)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Main plot + colorbar
        
        # Test saving
        save_path = temp_dir / 'test_similarity.png'
        fig = visualizer.plot_head_similarity_matrix(analysis, save_path=str(save_path), show_plot=False)
        assert save_path.exists()
        
        plt.close(fig)
    
    def test_export_analysis_data_json(self, visualizer, sample_features, temp_dir):
        """Test exporting analysis data to JSON."""
        spatial_features, frequency_features = sample_features
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        # Test JSON export
        export_path = visualizer.export_analysis_data(analyses, format='json')
        
        assert Path(export_path).exists()
        
        # Verify JSON content
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        assert len(data) == len(analyses)
        assert 'sample_id' in data[0]
        assert 'attention_weights' in data[0]
        assert 'attention_stats' in data[0]
    
    def test_export_analysis_data_npz(self, visualizer, sample_features, temp_dir):
        """Test exporting analysis data to NPZ."""
        spatial_features, frequency_features = sample_features
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        # Test NPZ export
        export_path = visualizer.export_analysis_data(analyses, format='npz')
        
        assert Path(export_path).exists()
        
        # Verify NPZ content
        data = np.load(export_path, allow_pickle=True)
        
        # Check that data contains expected keys
        expected_keys = ['sample_0_attention_weights', 'sample_0_head_similarities', 'sample_0_metadata']
        for key in expected_keys:
            assert key in data
    
    def test_export_analysis_data_invalid_format(self, visualizer, sample_features):
        """Test export with invalid format."""
        spatial_features, frequency_features = sample_features
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            visualizer.export_analysis_data(analyses, format='invalid')
    
    def test_generate_report(self, visualizer, sample_features, temp_dir):
        """Test HTML report generation."""
        spatial_features, frequency_features = sample_features
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        # Generate report
        report_path = visualizer.generate_report(analyses)
        
        assert Path(report_path).exists()
        assert Path(report_path).suffix == '.html'
        
        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()
        
        assert 'CRAFT-DF Cross-Attention Analysis Report' in content
        assert 'Summary Statistics' in content
        assert 'Individual Sample Analysis' in content
    
    def test_validate_attention_stability(self, visualizer, sample_features):
        """Test attention stability validation."""
        spatial_features, frequency_features = sample_features
        
        # Test stability validation
        stability_metrics = visualizer.validate_attention_stability(
            spatial_features, frequency_features, num_runs=5, noise_std=0.01
        )
        
        # Check required metrics
        required_metrics = ['mean_variation', 'std_variation', 'max_variation', 'stability_score', 'consistency_ratio']
        for metric in required_metrics:
            assert metric in stability_metrics
            assert isinstance(stability_metrics[metric], float)
            assert np.isfinite(stability_metrics[metric])
        
        # Check value ranges
        assert stability_metrics['mean_variation'] >= 0
        assert stability_metrics['std_variation'] >= 0
        assert stability_metrics['max_variation'] >= 0
        assert 0 <= stability_metrics['stability_score'] <= 1
        assert 0 <= stability_metrics['consistency_ratio'] <= 1
    
    def test_cache_functionality(self, visualizer, sample_features):
        """Test analysis caching functionality."""
        spatial_features, frequency_features = sample_features
        
        # First analysis should populate cache
        sample_ids = ["cached_1", "cached_2", "cached_3", "cached_4"]
        analyses = visualizer.analyze_attention_pattern(
            spatial_features, frequency_features, sample_ids=sample_ids
        )
        
        # Check cache is populated
        assert len(visualizer.analysis_cache) == 4
        for sample_id in sample_ids:
            assert sample_id in visualizer.analysis_cache
            assert isinstance(visualizer.analysis_cache[sample_id], AttentionAnalysis)
    
    def test_edge_cases(self, temp_dir, device):
        """Test edge cases and boundary conditions."""
        # Test with minimal model configuration
        model = CrossAttentionFusion(
            spatial_dim=4, frequency_dim=2, embed_dim=4, num_heads=2
        ).to(device)
        
        visualizer = AttentionVisualizer(model, save_dir=temp_dir)
        
        # Test with single sample
        spatial_features = torch.randn(1, 4, device=device)
        frequency_features = torch.randn(1, 2, device=device)
        
        analyses = visualizer.analyze_attention_pattern(spatial_features, frequency_features)
        
        assert len(analyses) == 1
        assert analyses[0].attention_weights.shape == (2, 1, 1)  # 2 heads
        assert analyses[0].head_similarities.shape == (2, 2)
    
    def test_error_handling(self, visualizer, device):
        """Test error handling in visualization methods."""
        # Test with mismatched feature dimensions
        spatial_features = torch.randn(2, 100, device=device)  # Wrong dimension
        frequency_features = torch.randn(2, 512, device=device)
        
        with pytest.raises((AssertionError, RuntimeError)):
            visualizer.analyze_attention_pattern(spatial_features, frequency_features)


class TestCrossAttentionExtensions:
    """Test suite for extended CrossAttentionFusion methods."""
    
    @pytest.fixture
    def device(self):
        """Get available device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        """Create CrossAttentionFusion model for testing."""
        model = CrossAttentionFusion()
        return model.to(device)
    
    @pytest.fixture
    def sample_features(self, device):
        """Create sample features for testing."""
        batch_size = 2
        spatial_features = torch.randn(batch_size, 1280, device=device)
        frequency_features = torch.randn(batch_size, 512, device=device)
        return spatial_features, frequency_features
    
    def test_validate_tensor_shapes(self, model, sample_features):
        """Test comprehensive tensor shape validation."""
        spatial_features, frequency_features = sample_features
        
        # Test input stage validation
        results = model.validate_tensor_shapes(spatial_features, frequency_features, stage="input")
        
        assert 'spatial_2d' in results
        assert 'frequency_2d' in results
        assert 'batch_match' in results
        assert 'spatial_dim_correct' in results
        assert 'frequency_dim_correct' in results
        assert 'all_valid' in results
        
        # All should be True for valid inputs
        assert all(results[key] for key in results if key != 'all_valid')
        assert results['all_valid'] is True
        
        # Test projection stage validation
        results_proj = model.validate_tensor_shapes(spatial_features, frequency_features, stage="projection")
        assert 'queries_shape' in results_proj
        assert 'keys_shape' in results_proj
        assert 'values_shape' in results_proj
        
        # Test attention stage validation
        results_att = model.validate_tensor_shapes(spatial_features, frequency_features, stage="attention")
        assert 'queries_reshaped' in results_att
        assert 'keys_reshaped' in results_att
        assert 'attention_scores_shape' in results_att
    
    def test_analyze_attention_gradients(self, model, sample_features):
        """Test attention gradient analysis."""
        spatial_features, frequency_features = sample_features
        
        # Enable gradients
        spatial_features.requires_grad_(True)
        frequency_features.requires_grad_(True)
        
        # Analyze gradients
        gradient_info = model.analyze_attention_gradients(spatial_features, frequency_features)
        
        # Check required keys
        required_keys = [
            'spatial_gradients', 'frequency_gradients', 'spatial_grad_norm', 
            'frequency_grad_norm', 'attention_weights', 'fused_features'
        ]
        for key in required_keys:
            assert key in gradient_info
        
        # Check gradient properties
        assert isinstance(gradient_info['spatial_gradients'], torch.Tensor)
        assert isinstance(gradient_info['frequency_gradients'], torch.Tensor)
        assert gradient_info['spatial_grad_norm'] >= 0
        assert gradient_info['frequency_grad_norm'] >= 0
        
        # Check shapes
        assert gradient_info['spatial_gradients'].shape == spatial_features.shape
        assert gradient_info['frequency_gradients'].shape == frequency_features.shape
    
    def test_compute_attention_rollout(self, model, sample_features):
        """Test attention rollout computation."""
        spatial_features, frequency_features = sample_features
        
        rollout = model.compute_attention_rollout(spatial_features, frequency_features)
        
        # Check shape and properties
        batch_size = spatial_features.shape[0]
        assert rollout.shape == (batch_size, 1, 1)  # Averaged across heads
        assert torch.all(rollout >= 0)  # Should be non-negative
        assert torch.all(torch.isfinite(rollout))
    
    def test_get_attention_maps_for_visualization(self, model, sample_features):
        """Test attention maps extraction for visualization."""
        spatial_features, frequency_features = sample_features
        
        attention_maps = model.get_attention_maps_for_visualization(spatial_features, frequency_features)
        
        # Check required keys
        required_keys = [
            'attention_weights', 'attention_scores', 'queries', 'keys', 'values',
            'head_averaged_attention', 'max_attention_per_head', 'min_attention_per_head'
        ]
        for key in required_keys:
            assert key in attention_maps
            assert isinstance(attention_maps[key], torch.Tensor)
        
        # Check shapes
        batch_size = spatial_features.shape[0]
        assert attention_maps['attention_weights'].shape == (batch_size, 8, 1, 1)
        assert attention_maps['attention_scores'].shape == (batch_size, 8, 1, 1)
        assert attention_maps['queries'].shape == (batch_size, 512)
        assert attention_maps['keys'].shape == (batch_size, 512)
        assert attention_maps['values'].shape == (batch_size, 512)
    
    def test_invalid_tensor_shapes(self, model, device):
        """Test validation with invalid tensor shapes."""
        # Test with wrong dimensions
        spatial_wrong = torch.randn(2, 100, device=device)  # Wrong spatial dim
        frequency_correct = torch.randn(2, 512, device=device)
        
        results = model.validate_tensor_shapes(spatial_wrong, frequency_correct, stage="input")
        
        assert results['spatial_dim_correct'] is False
        assert results['all_valid'] is False
        
        # Test with mismatched batch sizes
        spatial_batch2 = torch.randn(2, 1280, device=device)
        frequency_batch3 = torch.randn(3, 512, device=device)
        
        results = model.validate_tensor_shapes(spatial_batch2, frequency_batch3, stage="input")
        
        assert results['batch_match'] is False
        assert results['all_valid'] is False


if __name__ == '__main__':
    pytest.main([__file__])