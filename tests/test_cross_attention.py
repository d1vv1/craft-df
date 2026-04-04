"""
Unit tests for CrossAttentionFusion module.

This test suite validates the cross-attention mechanism implementation,
including attention weight computation, gradient flow, numerical stability,
and edge case handling.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import logging

# Import the module to test
from craft_df.models.cross_attention import CrossAttentionFusion


class TestCrossAttentionFusion:
    """Test suite for CrossAttentionFusion module."""
    
    @pytest.fixture
    def device(self):
        """Get available device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def default_model(self, device):
        """Create default CrossAttentionFusion model for testing."""
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
        """Create sample spatial and frequency features for testing."""
        batch_size = 4
        spatial_features = torch.randn(batch_size, 1280, device=device)
        frequency_features = torch.randn(batch_size, 512, device=device)
        return spatial_features, frequency_features
    
    def test_initialization_valid_parameters(self):
        """Test model initialization with valid parameters."""
        model = CrossAttentionFusion(
            spatial_dim=1280,
            frequency_dim=512,
            embed_dim=512,
            num_heads=8,
            dropout_rate=0.1,
            temperature=1.0
        )
        
        assert model.spatial_dim == 1280
        assert model.frequency_dim == 512
        assert model.embed_dim == 512
        assert model.num_heads == 8
        assert model.head_dim == 64  # 512 // 8
        assert model.dropout_rate == 0.1
        assert model.temperature == 1.0
    
    def test_initialization_invalid_parameters(self):
        """Test model initialization with invalid parameters."""
        # Test negative dimensions
        with pytest.raises(AssertionError):
            CrossAttentionFusion(spatial_dim=-1, frequency_dim=512)
        
        with pytest.raises(AssertionError):
            CrossAttentionFusion(spatial_dim=1280, frequency_dim=-1)
        
        # Test embed_dim not divisible by num_heads
        with pytest.raises(AssertionError):
            CrossAttentionFusion(embed_dim=513, num_heads=8)
        
        # Test invalid dropout rate
        with pytest.raises(AssertionError):
            CrossAttentionFusion(dropout_rate=-0.1)
        
        with pytest.raises(AssertionError):
            CrossAttentionFusion(dropout_rate=1.1)
        
        # Test invalid temperature
        with pytest.raises(AssertionError):
            CrossAttentionFusion(temperature=-1.0)
    
    def test_forward_pass_basic(self, default_model, sample_features, device):
        """Test basic forward pass functionality."""
        spatial_features, frequency_features = sample_features
        
        # Test forward pass without attention weights
        fused_features, attention_weights = default_model(spatial_features, frequency_features)
        
        # Validate output shape
        expected_shape = (spatial_features.shape[0], default_model.embed_dim)
        assert fused_features.shape == expected_shape
        assert attention_weights is None
        
        # Validate output is finite
        assert torch.all(torch.isfinite(fused_features))
    
    def test_forward_pass_with_attention_weights(self, default_model, sample_features, device):
        """Test forward pass with attention weight return."""
        spatial_features, frequency_features = sample_features
        
        # Set model to eval mode to disable dropout
        default_model.eval()
        
        # Test forward pass with attention weights
        fused_features, attention_weights = default_model(
            spatial_features, frequency_features, return_attention=True
        )
        
        # Validate output shapes
        batch_size = spatial_features.shape[0]
        assert fused_features.shape == (batch_size, default_model.embed_dim)
        assert attention_weights.shape == (batch_size, default_model.num_heads, 1, 1)
        
        # Validate attention weights are probabilities (sum to 1)
        # Note: Since we have single query-key pair per head, each attention weight is 1.0
        # The softmax of a single element is always 1.0
        assert torch.allclose(attention_weights, torch.ones_like(attention_weights), atol=1e-5)
        
        # Validate all outputs are finite
        assert torch.all(torch.isfinite(fused_features))
        assert torch.all(torch.isfinite(attention_weights))
    
    def test_tensor_shape_validation(self, default_model, device):
        """Test input tensor shape validation."""
        # Test invalid spatial features shape (3D instead of 2D)
        spatial_3d = torch.randn(4, 1280, 10, device=device)
        frequency_2d = torch.randn(4, 512, device=device)
        
        with pytest.raises(AssertionError, match="Expected 2D spatial features"):
            default_model(spatial_3d, frequency_2d)
        
        # Test invalid frequency features shape (3D instead of 2D)
        spatial_2d = torch.randn(4, 1280, device=device)
        frequency_3d = torch.randn(4, 512, 10, device=device)
        
        with pytest.raises(AssertionError, match="Expected 2D frequency features"):
            default_model(spatial_2d, frequency_3d)
        
        # Test batch size mismatch
        spatial_batch4 = torch.randn(4, 1280, device=device)
        frequency_batch2 = torch.randn(2, 512, device=device)
        
        with pytest.raises(AssertionError, match="Batch size mismatch"):
            default_model(spatial_batch4, frequency_batch2)
        
        # Test wrong feature dimensions
        spatial_wrong_dim = torch.randn(4, 1000, device=device)  # Should be 1280
        frequency_correct = torch.randn(4, 512, device=device)
        
        with pytest.raises(AssertionError, match="Expected spatial_dim"):
            default_model(spatial_wrong_dim, frequency_correct)
        
        spatial_correct = torch.randn(4, 1280, device=device)
        frequency_wrong_dim = torch.randn(4, 256, device=device)  # Should be 512
        
        with pytest.raises(AssertionError, match="Expected frequency_dim"):
            default_model(spatial_correct, frequency_wrong_dim)
    
    def test_gradient_flow(self, default_model, sample_features, device):
        """Test gradient computation and backpropagation."""
        spatial_features, frequency_features = sample_features
        
        # Enable gradient computation
        spatial_features.requires_grad_(True)
        frequency_features.requires_grad_(True)
        
        # Forward pass
        fused_features, _ = default_model(spatial_features, frequency_features)
        
        # Compute loss and backward pass
        loss = fused_features.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert spatial_features.grad is not None
        assert frequency_features.grad is not None
        assert torch.all(torch.isfinite(spatial_features.grad))
        assert torch.all(torch.isfinite(frequency_features.grad))
        
        # Check model parameter gradients
        for name, param in default_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"
                assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient for parameter {name}"
    
    def test_attention_weight_extraction(self, default_model, sample_features, device):
        """Test attention weight extraction functionality."""
        spatial_features, frequency_features = sample_features
        
        # Set model to eval mode to disable dropout
        default_model.eval()
        
        # Test get_attention_weights method
        attention_weights = default_model.get_attention_weights(spatial_features, frequency_features)
        
        batch_size = spatial_features.shape[0]
        expected_shape = (batch_size, default_model.num_heads, 1, 1)
        assert attention_weights.shape == expected_shape
        
        # Validate attention weights are probabilities
        # Note: With single query-key pair, softmax always produces 1.0
        assert torch.all(attention_weights >= 0)
        # Each attention weight should be 1.0 (softmax of single element)
        assert torch.allclose(attention_weights, torch.ones_like(attention_weights), atol=1e-5)
    
    def test_attention_visualization(self, default_model, sample_features, device):
        """Test attention pattern visualization functionality."""
        spatial_features, frequency_features = sample_features
        
        # Test visualization data generation
        viz_data = default_model.visualize_attention_pattern(
            spatial_features, frequency_features, sample_idx=0
        )
        
        # Validate visualization data structure
        assert 'attention_weights' in viz_data
        assert 'attention_stats' in viz_data
        assert 'feature_norms' in viz_data
        assert 'num_heads' in viz_data
        assert 'sample_idx' in viz_data
        
        # Validate attention statistics
        stats = viz_data['attention_stats']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'max' in stats
        assert 'min' in stats
        
        # Validate feature norms
        norms = viz_data['feature_norms']
        assert 'spatial' in norms
        assert 'frequency' in norms
        assert norms['spatial'] > 0
        assert norms['frequency'] > 0
    
    def test_attention_entropy_computation(self, default_model, sample_features, device):
        """Test attention entropy computation for measuring attention concentration."""
        spatial_features, frequency_features = sample_features
        
        # Compute attention entropy
        entropy = default_model.compute_attention_entropy(spatial_features, frequency_features)
        
        batch_size = spatial_features.shape[0]
        assert entropy.shape == (batch_size,)
        assert torch.all(entropy >= 0)  # Entropy should be non-negative
        assert torch.all(torch.isfinite(entropy))
    
    def test_numerical_stability_validation(self, default_model, sample_features, device):
        """Test numerical stability validation functionality."""
        spatial_features, frequency_features = sample_features
        
        # Test with normal features
        results = default_model.validate_numerical_stability(spatial_features, frequency_features)
        
        # All checks should pass with normal inputs
        for key, value in results.items():
            assert value is True, f"Numerical stability check failed for {key}"
    
    def test_non_finite_input_handling(self, default_model, device):
        """Test handling of non-finite input values."""
        batch_size = 4
        
        # Create features with NaN values
        spatial_with_nan = torch.randn(batch_size, 1280, device=device)
        spatial_with_nan[0, 0] = float('nan')
        
        frequency_normal = torch.randn(batch_size, 512, device=device)
        
        # Test that model handles NaN gracefully (should replace with zeros and continue)
        with patch('craft_df.models.cross_attention.logger') as mock_logger:
            fused_features, _ = default_model(spatial_with_nan, frequency_normal)
            
            # Should log warning about non-finite values
            mock_logger.warning.assert_called()
            
            # Output should still be finite
            assert torch.all(torch.isfinite(fused_features))
    
    def test_memory_usage_estimation(self, default_model):
        """Test memory usage estimation functionality."""
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            memory_info = default_model.get_memory_usage(batch_size)
            
            # Validate memory info structure
            required_keys = [
                'parameter_memory_mb', 'spatial_input_mb', 'frequency_input_mb',
                'qkv_projections_mb', 'attention_weights_mb', 'output_memory_mb',
                'total_estimated_mb', 'batch_size'
            ]
            
            for key in required_keys:
                assert key in memory_info
                assert memory_info[key] >= 0
            
            # Total memory should be sum of components
            component_sum = (
                memory_info['parameter_memory_mb'] + 
                memory_info['spatial_input_mb'] + 
                memory_info['frequency_input_mb'] + 
                memory_info['qkv_projections_mb'] + 
                memory_info['attention_weights_mb'] + 
                memory_info['output_memory_mb']
            )
            
            assert abs(memory_info['total_estimated_mb'] - component_sum) < 1e-6
    
    def test_model_info_generation(self, default_model):
        """Test model information generation."""
        model_info = default_model.get_model_info()
        
        # Validate model info structure
        required_keys = [
            'model_name', 'spatial_dim', 'frequency_dim', 'embed_dim',
            'num_heads', 'head_dim', 'temperature', 'dropout_rate',
            'total_parameters', 'trainable_parameters', 'parameter_efficiency'
        ]
        
        for key in required_keys:
            assert key in model_info
        
        # Validate specific values
        assert model_info['model_name'] == 'CrossAttentionFusion'
        assert model_info['spatial_dim'] == 1280
        assert model_info['frequency_dim'] == 512
        assert model_info['embed_dim'] == 512
        assert model_info['num_heads'] == 8
        assert model_info['head_dim'] == 64
        
        # Parameter counts should be positive
        assert model_info['total_parameters'] > 0
        assert model_info['trainable_parameters'] > 0
        assert 0 <= model_info['parameter_efficiency'] <= 1
    
    def test_different_configurations(self, device):
        """Test model with different configuration parameters."""
        configs = [
            {'spatial_dim': 512, 'frequency_dim': 256, 'embed_dim': 256, 'num_heads': 4},
            {'spatial_dim': 2048, 'frequency_dim': 1024, 'embed_dim': 1024, 'num_heads': 16},
            {'spatial_dim': 128, 'frequency_dim': 64, 'embed_dim': 128, 'num_heads': 2}
        ]
        
        for config in configs:
            model = CrossAttentionFusion(**config).to(device)
            
            batch_size = 2
            spatial_features = torch.randn(batch_size, config['spatial_dim'], device=device)
            frequency_features = torch.randn(batch_size, config['frequency_dim'], device=device)
            
            # Test forward pass
            fused_features, _ = model(spatial_features, frequency_features)
            
            # Validate output shape
            expected_shape = (batch_size, config['embed_dim'])
            assert fused_features.shape == expected_shape
            assert torch.all(torch.isfinite(fused_features))
    
    def test_training_vs_eval_mode(self, default_model, sample_features, device):
        """Test behavior differences between training and evaluation modes."""
        spatial_features, frequency_features = sample_features
        
        # Test in training mode
        default_model.train()
        train_output, _ = default_model(spatial_features, frequency_features)
        
        # Test in evaluation mode
        default_model.eval()
        eval_output, _ = default_model(spatial_features, frequency_features)
        
        # Outputs should have same shape
        assert train_output.shape == eval_output.shape
        
        # Due to dropout, outputs might be different in training vs eval
        # But both should be finite
        assert torch.all(torch.isfinite(train_output))
        assert torch.all(torch.isfinite(eval_output))
    
    def test_optimization_methods(self, default_model, sample_features, device):
        """Test model optimization methods."""
        spatial_features, frequency_features = sample_features
        
        # Test gradient checkpointing enablement
        default_model.enable_gradient_checkpointing()
        assert hasattr(default_model, 'use_gradient_checkpointing')
        assert default_model.use_gradient_checkpointing is True
        
        # Test inference optimization
        default_model.optimize_for_inference()
        
        # After optimization, model should still work
        fused_features, _ = default_model(spatial_features, frequency_features)
        assert torch.all(torch.isfinite(fused_features))
        
        # Dropout should be disabled (p=0.0)
        for module in default_model.modules():
            if isinstance(module, nn.Dropout):
                assert module.p == 0.0
    
    def test_edge_cases(self, device):
        """Test edge cases and boundary conditions."""
        # Test with minimum valid configuration
        model = CrossAttentionFusion(
            spatial_dim=1, frequency_dim=1, embed_dim=2, num_heads=1
        ).to(device)
        
        spatial_features = torch.randn(1, 1, device=device)
        frequency_features = torch.randn(1, 1, device=device)
        
        fused_features, _ = model(spatial_features, frequency_features)
        assert fused_features.shape == (1, 2)
        assert torch.all(torch.isfinite(fused_features))
        
        # Test with single sample batch
        batch_size = 1
        spatial_single = torch.randn(batch_size, 1280, device=device)
        frequency_single = torch.randn(batch_size, 512, device=device)
        
        default_model = CrossAttentionFusion().to(device)
        fused_single, _ = default_model(spatial_single, frequency_single)
        
        assert fused_single.shape == (batch_size, 512)
        assert torch.all(torch.isfinite(fused_single))
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_specific_functionality(self, device):
        """Test CUDA-specific functionality like mixed precision."""
        if device.type != 'cuda':
            pytest.skip("CUDA not available")
        
        model = CrossAttentionFusion().to(device)
        spatial_features = torch.randn(4, 1280, device=device)
        frequency_features = torch.randn(4, 512, device=device)
        
        # Test with mixed precision
        with torch.cuda.amp.autocast():
            fused_features, _ = model(spatial_features, frequency_features)
            assert torch.all(torch.isfinite(fused_features))
        
        # Test memory management
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        fused_features, _ = model(spatial_features, frequency_features)
        
        # Memory should increase
        after_forward_memory = torch.cuda.memory_allocated()
        assert after_forward_memory >= initial_memory
    
    def test_reproducibility(self, device):
        """Test reproducibility with fixed random seeds."""
        # Set seeds for reproducible initialization
        torch.manual_seed(42)
        if device.type == 'cuda':
            torch.cuda.manual_seed(42)
        np.random.seed(42)
        
        # Create first model
        model1 = CrossAttentionFusion().to(device)
        
        # Reset seeds and create second model
        torch.manual_seed(42)
        if device.type == 'cuda':
            torch.cuda.manual_seed(42)
        np.random.seed(42)
        
        model2 = CrossAttentionFusion().to(device)
        
        # Ensure models have identical parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-8), "Model parameters should be identical"
        
        # Create identical inputs
        torch.manual_seed(123)
        spatial_features = torch.randn(4, 1280, device=device)
        frequency_features = torch.randn(4, 512, device=device)
        
        # Set models to eval mode to disable dropout randomness
        model1.eval()
        model2.eval()
        
        # Forward pass should produce identical results
        output1, _ = model1(spatial_features, frequency_features)
        output2, _ = model2(spatial_features, frequency_features)
        
        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__])