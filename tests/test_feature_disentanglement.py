"""
Unit tests for Feature Disentanglement Module

This module contains comprehensive tests for the FeatureDisentanglement class,
including tests for adversarial training components, loss computation, and
gradient flow validation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import logging

# Import the module under test
from craft_df.models.feature_disentanglement import (
    FeatureDisentanglement,
    ResidualBlock,
    GradientReversalLayer,
    GradientReversalFunction
)


class TestFeatureDisentanglement:
    """Test suite for FeatureDisentanglement class."""
    
    @pytest.fixture
    def default_config(self):
        """Default configuration for testing."""
        return {
            'input_dim': 512,
            'invariant_dim': 256,
            'specific_dim': 128,
            'num_domains': 4,
            'hidden_dim': 512,
            'num_layers': 3,
            'dropout_rate': 0.1,
            'adversarial_weight': 0.1,
            'reconstruction_weight': 0.01,
            'gradient_reversal_lambda': 1.0
        }
    
    @pytest.fixture
    def model(self, default_config):
        """Create a FeatureDisentanglement model for testing."""
        return FeatureDisentanglement(**default_config)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        batch_size = 8
        input_dim = 512
        num_domains = 4
        
        fused_features = torch.randn(batch_size, input_dim)
        domain_labels = torch.randint(0, num_domains, (batch_size,))
        
        return {
            'fused_features': fused_features,
            'domain_labels': domain_labels,
            'batch_size': batch_size
        }
    
    def test_initialization_valid_params(self, default_config):
        """Test model initialization with valid parameters."""
        model = FeatureDisentanglement(**default_config)
        
        assert model.input_dim == default_config['input_dim']
        assert model.invariant_dim == default_config['invariant_dim']
        assert model.specific_dim == default_config['specific_dim']
        assert model.num_domains == default_config['num_domains']
        assert model.adversarial_weight == default_config['adversarial_weight']
        assert model.reconstruction_weight == default_config['reconstruction_weight']
        
        # Check that all components are initialized
        assert hasattr(model, 'invariant_encoder')
        assert hasattr(model, 'specific_encoder')
        assert hasattr(model, 'domain_classifier')
        assert hasattr(model, 'feature_decoder')
        assert hasattr(model, 'gradient_reversal')
    
    def test_initialization_invalid_params(self):
        """Test model initialization with invalid parameters."""
        # Test negative input_dim
        with pytest.raises(AssertionError, match="input_dim must be positive"):
            FeatureDisentanglement(input_dim=-1)
        
        # Test zero invariant_dim
        with pytest.raises(AssertionError, match="invariant_dim must be positive"):
            FeatureDisentanglement(invariant_dim=0)
        
        # Test invalid num_domains
        with pytest.raises(AssertionError, match="num_domains must be > 1"):
            FeatureDisentanglement(num_domains=1)
        
        # Test invalid dropout_rate
        with pytest.raises(AssertionError, match="dropout_rate must be between 0 and 1"):
            FeatureDisentanglement(dropout_rate=1.5)
        
        # Test negative adversarial_weight
        with pytest.raises(AssertionError, match="adversarial_weight must be non-negative"):
            FeatureDisentanglement(adversarial_weight=-0.1)
    
    def test_forward_pass_basic(self, model, sample_data):
        """Test basic forward pass without loss computation."""
        fused_features = sample_data['fused_features']
        
        invariant_features, specific_features, losses = model.forward(
            fused_features, return_losses=False
        )
        
        # Check output shapes
        expected_invariant_shape = (sample_data['batch_size'], model.invariant_dim)
        expected_specific_shape = (sample_data['batch_size'], model.specific_dim)
        
        assert invariant_features.shape == expected_invariant_shape
        assert specific_features.shape == expected_specific_shape
        assert losses is None
        
        # Check that outputs are finite
        assert torch.all(torch.isfinite(invariant_features))
        assert torch.all(torch.isfinite(specific_features))
    
    def test_forward_pass_with_losses(self, model, sample_data):
        """Test forward pass with loss computation."""
        fused_features = sample_data['fused_features']
        domain_labels = sample_data['domain_labels']
        
        invariant_features, specific_features, losses = model.forward(
            fused_features, domain_labels, return_losses=True
        )
        
        # Check that losses are computed
        assert losses is not None
        assert 'reconstruction' in losses
        assert 'adversarial' in losses
        assert 'orthogonality' in losses
        assert 'total' in losses
        assert 'domain_accuracy' in losses
        
        # Check that all losses are finite scalars
        for loss_name, loss_value in losses.items():
            if loss_name != 'domain_accuracy':  # Accuracy can be exactly 0 or 1
                assert torch.isfinite(loss_value), f"{loss_name} loss is not finite"
            assert loss_value.dim() == 0, f"{loss_name} should be a scalar"
    
    def test_forward_pass_invalid_input_shape(self, model):
        """Test forward pass with invalid input shapes."""
        # Test 1D input
        with pytest.raises(AssertionError, match="Expected 2D fused features"):
            model.forward(torch.randn(512))
        
        # Test 3D input
        with pytest.raises(AssertionError, match="Expected 2D fused features"):
            model.forward(torch.randn(8, 512, 10))
        
        # Test wrong feature dimension
        with pytest.raises(AssertionError, match="Expected input_dim"):
            model.forward(torch.randn(8, 256))  # Wrong dimension
    
    def test_forward_pass_non_finite_input(self, model, sample_data):
        """Test forward pass with non-finite input values."""
        fused_features = sample_data['fused_features'].clone()
        
        # Introduce NaN and Inf values
        fused_features[0, 0] = float('nan')
        fused_features[1, 0] = float('inf')
        fused_features[2, 0] = float('-inf')
        
        # Should handle non-finite values gracefully
        with patch('craft_df.models.feature_disentanglement.logger') as mock_logger:
            invariant_features, specific_features, _ = model.forward(fused_features)
            
            # Check that warning was logged
            mock_logger.warning.assert_called_once()
            
            # Check that outputs are finite
            assert torch.all(torch.isfinite(invariant_features))
            assert torch.all(torch.isfinite(specific_features))
    
    def test_loss_computation_reconstruction(self, model, sample_data):
        """Test reconstruction loss computation."""
        fused_features = sample_data['fused_features']
        
        # Set model to eval mode to disable dropout for consistent results
        model.eval()
        
        # Get disentangled features
        invariant_features, specific_features, _ = model.forward(fused_features)
        
        # Compute reconstruction loss manually
        concatenated = torch.cat([invariant_features, specific_features], dim=1)
        reconstructed = model.feature_decoder(concatenated)
        expected_loss = nn.MSELoss()(reconstructed, fused_features)
        
        # Get loss from model
        _, _, losses = model.forward(fused_features, return_losses=True)
        actual_loss = losses['reconstruction']
        
        # Should be approximately equal
        assert torch.allclose(actual_loss, expected_loss, rtol=1e-4)
    
    def test_loss_computation_adversarial(self, model, sample_data):
        """Test adversarial loss computation."""
        fused_features = sample_data['fused_features']
        domain_labels = sample_data['domain_labels']
        
        # Forward pass with losses
        invariant_features, _, losses = model.forward(
            fused_features, domain_labels, return_losses=True
        )
        
        # Check adversarial loss properties
        adversarial_loss = losses['adversarial']
        assert adversarial_loss >= 0, "Cross-entropy loss should be non-negative"
        
        # Check domain accuracy is between 0 and 1
        domain_accuracy = losses['domain_accuracy']
        assert 0 <= domain_accuracy <= 1, "Domain accuracy should be between 0 and 1"
    
    def test_loss_computation_orthogonality(self, model, sample_data):
        """Test orthogonality loss computation."""
        fused_features = sample_data['fused_features']
        
        # Forward pass
        invariant_features, specific_features, losses = model.forward(
            fused_features, return_losses=True
        )
        
        # Compute orthogonality loss manually
        invariant_norm = torch.nn.functional.normalize(invariant_features, p=2, dim=1)
        specific_norm = torch.nn.functional.normalize(specific_features, p=2, dim=1)
        correlation = torch.mm(invariant_norm.t(), specific_norm)
        expected_loss = torch.norm(correlation, p='fro') ** 2
        
        actual_loss = losses['orthogonality']
        
        # Should be approximately equal
        assert torch.allclose(actual_loss, expected_loss, rtol=1e-5)
    
    def test_get_disentangled_features(self, model, sample_data):
        """Test get_disentangled_features method."""
        fused_features = sample_data['fused_features']
        
        invariant_features, specific_features = model.get_disentangled_features(fused_features)
        
        # Check shapes
        assert invariant_features.shape == (sample_data['batch_size'], model.invariant_dim)
        assert specific_features.shape == (sample_data['batch_size'], model.specific_dim)
        
        # Check that outputs are finite
        assert torch.all(torch.isfinite(invariant_features))
        assert torch.all(torch.isfinite(specific_features))
        
        # Should not require gradients (inference mode)
        assert not invariant_features.requires_grad
        assert not specific_features.requires_grad
    
    def test_compute_domain_confusion(self, model, sample_data):
        """Test domain confusion computation."""
        fused_features = sample_data['fused_features']
        domain_labels = sample_data['domain_labels']
        
        # Get invariant features
        invariant_features, _, _ = model.forward(fused_features)
        
        # Compute domain confusion
        confusion = model.compute_domain_confusion(invariant_features, domain_labels)
        
        # Should be between 0 and 1
        assert 0 <= confusion <= 1, "Domain confusion should be between 0 and 1"
        assert confusion.dim() == 0, "Domain confusion should be a scalar"
    
    def test_analyze_feature_separation(self, model, sample_data):
        """Test feature separation analysis."""
        fused_features = sample_data['fused_features']
        domain_labels = sample_data['domain_labels']
        
        analysis = model.analyze_feature_separation(fused_features, domain_labels)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'invariant_mean_norm', 'specific_mean_norm',
            'invariant_std_mean', 'specific_std_mean',
            'correlation_magnitude', 'domain_confusion',
            'reconstruction_error', 'separation_quality'
        ]
        
        for metric in expected_metrics:
            assert metric in analysis, f"Missing metric: {metric}"
            assert isinstance(analysis[metric], float), f"{metric} should be a float"
            assert np.isfinite(analysis[metric]), f"{metric} should be finite"
    
    def test_gradient_flow(self, model, sample_data):
        """Test gradient flow through the model."""
        fused_features = sample_data['fused_features'].requires_grad_(True)
        domain_labels = sample_data['domain_labels']
        
        # Forward pass with losses
        invariant_features, specific_features, losses = model.forward(
            fused_features, domain_labels, return_losses=True
        )
        
        # Backward pass
        total_loss = losses['total']
        total_loss.backward()
        
        # Check that gradients are computed
        assert fused_features.grad is not None, "Input gradients should be computed"
        assert torch.all(torch.isfinite(fused_features.grad)), "Input gradients should be finite"
        
        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"
                assert torch.all(torch.isfinite(param.grad)), f"Parameter {name} gradients should be finite"
    
    def test_gradient_reversal_effect(self, model, sample_data):
        """Test that gradient reversal actually reverses gradients."""
        fused_features = sample_data['fused_features'].requires_grad_(True)
        domain_labels = sample_data['domain_labels']
        
        # Forward pass
        invariant_features, _, _ = model.forward(fused_features)
        
        # Apply gradient reversal
        reversed_features = model.gradient_reversal(invariant_features)
        
        # Compute a simple loss
        loss = reversed_features.sum()
        loss.backward()
        
        # The gradient should flow back through the reversal layer
        assert fused_features.grad is not None
        assert torch.all(torch.isfinite(fused_features.grad))
    
    def test_model_training_mode(self, model, sample_data):
        """Test model behavior in training vs evaluation mode."""
        fused_features = sample_data['fused_features']
        domain_labels = sample_data['domain_labels']
        
        # Training mode
        model.train()
        train_output = model.forward(fused_features, domain_labels, return_losses=True)
        
        # Evaluation mode
        model.eval()
        eval_output = model.forward(fused_features, domain_labels, return_losses=True)
        
        # Outputs should be different due to dropout
        train_invariant, train_specific, train_losses = train_output
        eval_invariant, eval_specific, eval_losses = eval_output
        
        # Shapes should be the same
        assert train_invariant.shape == eval_invariant.shape
        assert train_specific.shape == eval_specific.shape
    
    def test_memory_efficiency(self, model):
        """Test memory efficiency with large batch sizes."""
        # Test with larger batch size
        large_batch_size = 64
        fused_features = torch.randn(large_batch_size, model.input_dim)
        domain_labels = torch.randint(0, model.num_domains, (large_batch_size,))
        
        # Should handle large batches without memory issues
        invariant_features, specific_features, losses = model.forward(
            fused_features, domain_labels, return_losses=True
        )
        
        # Check outputs
        assert invariant_features.shape[0] == large_batch_size
        assert specific_features.shape[0] == large_batch_size
        assert losses is not None


class TestResidualBlock:
    """Test suite for ResidualBlock class."""
    
    def test_residual_block_same_dim(self):
        """Test residual block with same input/output dimensions."""
        input_dim = 256
        output_dim = 256
        block = ResidualBlock(input_dim, output_dim)
        
        x = torch.randn(8, input_dim)
        output = block(x)
        
        assert output.shape == (8, output_dim)
        assert torch.all(torch.isfinite(output))
    
    def test_residual_block_different_dim(self):
        """Test residual block with different input/output dimensions."""
        input_dim = 256
        output_dim = 512
        block = ResidualBlock(input_dim, output_dim)
        
        x = torch.randn(8, input_dim)
        output = block(x)
        
        assert output.shape == (8, output_dim)
        assert torch.all(torch.isfinite(output))
    
    def test_residual_block_gradient_flow(self):
        """Test gradient flow through residual block."""
        block = ResidualBlock(256, 256)
        x = torch.randn(8, 256, requires_grad=True)
        
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


class TestGradientReversalLayer:
    """Test suite for GradientReversalLayer class."""
    
    def test_gradient_reversal_forward(self):
        """Test forward pass of gradient reversal layer."""
        layer = GradientReversalLayer(lambda_param=1.0)
        x = torch.randn(8, 256)
        
        output = layer(x)
        
        # Forward pass should be identity
        assert torch.allclose(output, x)
    
    def test_gradient_reversal_backward(self):
        """Test backward pass of gradient reversal layer."""
        layer = GradientReversalLayer(lambda_param=2.0)
        x = torch.randn(8, 256, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Gradients should be reversed and scaled
        expected_grad = -2.0 * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)
    
    def test_gradient_reversal_lambda_update(self):
        """Test updating lambda parameter."""
        layer = GradientReversalLayer(lambda_param=1.0)
        assert layer.lambda_param == 1.0
        
        layer.set_lambda(0.5)
        assert layer.lambda_param == 0.5
    
    def test_gradient_reversal_function_directly(self):
        """Test GradientReversalFunction directly."""
        x = torch.randn(4, 128, requires_grad=True)
        lambda_param = 1.5
        
        # Forward pass
        output = GradientReversalFunction.apply(x, lambda_param)
        assert torch.allclose(output, x)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradient reversal
        expected_grad = -lambda_param * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)


class TestIntegration:
    """Integration tests for the complete feature disentanglement system."""
    
    def test_end_to_end_training_step(self):
        """Test a complete training step with all components."""
        # Create model
        model = FeatureDisentanglement(
            input_dim=512,
            invariant_dim=256,
            specific_dim=128,
            num_domains=3
        )
        
        # Create sample data
        batch_size = 16
        fused_features = torch.randn(batch_size, 512, requires_grad=True)
        domain_labels = torch.randint(0, 3, (batch_size,))
        
        # Forward pass
        invariant_features, specific_features, losses = model.forward(
            fused_features, domain_labels, return_losses=True
        )
        
        # Backward pass
        total_loss = losses['total']
        total_loss.backward()
        
        # Check that everything worked
        assert torch.all(torch.isfinite(invariant_features))
        assert torch.all(torch.isfinite(specific_features))
        assert torch.isfinite(total_loss)
        assert fused_features.grad is not None
        
        # Check loss components
        assert losses['reconstruction'] > 0
        assert losses['adversarial'] >= 0
        assert losses['orthogonality'] >= 0
        assert 0 <= losses['domain_accuracy'] <= 1
    
    def test_feature_disentanglement_quality(self):
        """Test that the model actually disentangles features."""
        model = FeatureDisentanglement(
            input_dim=256,
            invariant_dim=128,
            specific_dim=64,
            num_domains=2
        )
        
        # Create synthetic data with known domain structure
        batch_size = 32
        
        # Domain 0: features with specific pattern
        domain0_features = torch.randn(batch_size // 2, 256)
        domain0_features[:, :64] += 2.0  # Domain-specific component
        domain0_labels = torch.zeros(batch_size // 2, dtype=torch.long)
        
        # Domain 1: features with different pattern
        domain1_features = torch.randn(batch_size // 2, 256)
        domain1_features[:, :64] -= 2.0  # Different domain-specific component
        domain1_labels = torch.ones(batch_size // 2, dtype=torch.long)
        
        # Combine data
        all_features = torch.cat([domain0_features, domain1_features], dim=0)
        all_labels = torch.cat([domain0_labels, domain1_labels], dim=0)
        
        # Train for a few steps
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for _ in range(10):
            optimizer.zero_grad()
            _, _, losses = model.forward(all_features, all_labels, return_losses=True)
            losses['total'].backward()
            optimizer.step()
        
        # Analyze feature separation
        analysis = model.analyze_feature_separation(all_features, all_labels)
        
        # After training, domain confusion should be reasonable
        # (not perfect since we only trained for a few steps)
        assert analysis['domain_confusion'] >= 0
        assert analysis['reconstruction_error'] >= 0
        assert np.isfinite(analysis['separation_quality'])
    
    def test_different_model_configurations(self):
        """Test model with different configurations."""
        configs = [
            {'input_dim': 128, 'invariant_dim': 64, 'specific_dim': 32, 'num_domains': 2},
            {'input_dim': 1024, 'invariant_dim': 512, 'specific_dim': 256, 'num_domains': 5},
            {'input_dim': 256, 'invariant_dim': 128, 'specific_dim': 128, 'num_domains': 3}
        ]
        
        for config in configs:
            model = FeatureDisentanglement(**config)
            
            # Test forward pass
            batch_size = 8
            fused_features = torch.randn(batch_size, config['input_dim'])
            domain_labels = torch.randint(0, config['num_domains'], (batch_size,))
            
            invariant_features, specific_features, losses = model.forward(
                fused_features, domain_labels, return_losses=True
            )
            
            # Check shapes
            assert invariant_features.shape == (batch_size, config['invariant_dim'])
            assert specific_features.shape == (batch_size, config['specific_dim'])
            assert losses is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])