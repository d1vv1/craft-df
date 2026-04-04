"""
Integration tests for CRAFT-DF Model with Feature Disentanglement

This module contains comprehensive integration tests for the complete CRAFT-DF model,
including tests for the integration of feature disentanglement with the main architecture,
training loop functionality, and end-to-end model behavior.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os

# Import the modules under test
from craft_df.models.craft_df_model import CRAFTDFModel


class TestCRAFTDFModelIntegration:
    """Integration test suite for complete CRAFT-DF model."""
    
    @pytest.fixture
    def model_configs(self):
        """Default model configurations for testing."""
        return {
            'spatial_config': {
                'pretrained': False,  # Use False for testing to avoid downloading weights
                'freeze_layers': 5,
                'feature_dim': 128,  # Smaller for testing
                'dropout_rate': 0.1
            },
            'frequency_config': {
                'input_channels': 3,
                'dwt_levels': 2,  # Smaller for testing
                'feature_dim': 64,  # Smaller for testing
                'dropout_rate': 0.1
            },
            'attention_config': {
                'spatial_dim': 128,
                'frequency_dim': 64,
                'embed_dim': 128,
                'num_heads': 4,  # Smaller for testing
                'dropout_rate': 0.1
            },
            'disentanglement_config': {
                'input_dim': 128,
                'invariant_dim': 64,
                'specific_dim': 32,
                'num_domains': 3,
                'hidden_dim': 128,
                'adversarial_weight': 0.1,
                'reconstruction_weight': 0.01
            }
        }
    
    @pytest.fixture
    def sample_batch(self):
        """Generate sample batch data for testing."""
        batch_size = 4
        
        # Create DWT coefficients in the expected format
        dwt_coefficients = {
            'll': torch.randn(batch_size, 3, 56, 56),  # Approximation coefficients
            'lh_1': torch.randn(batch_size, 3, 112, 112),  # Level 1 horizontal details
            'hl_1': torch.randn(batch_size, 3, 112, 112),  # Level 1 vertical details
            'hh_1': torch.randn(batch_size, 3, 112, 112),  # Level 1 diagonal details
            'lh_2': torch.randn(batch_size, 3, 56, 56),   # Level 2 horizontal details
            'hl_2': torch.randn(batch_size, 3, 56, 56),   # Level 2 vertical details
            'hh_2': torch.randn(batch_size, 3, 56, 56),   # Level 2 diagonal details
        }
        
        return {
            'spatial_input': torch.randn(batch_size, 3, 224, 224),
            'frequency_input': dwt_coefficients,
            'labels': torch.randint(0, 2, (batch_size,)),
            'domain_labels': torch.randint(0, 3, (batch_size,))
        }
    
    def test_model_initialization_with_disentanglement(self, model_configs):
        """Test model initialization with feature disentanglement enabled."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        # Check that all components are initialized
        assert hasattr(model, 'spatial_stream')
        assert hasattr(model, 'frequency_stream')
        assert hasattr(model, 'cross_attention')
        assert hasattr(model, 'feature_disentanglement')
        assert hasattr(model, 'classifier')
        
        # Check that feature disentanglement is enabled
        assert model.feature_disentanglement is not None
        assert model.adversarial_training is True
        
        # Check model summary
        summary = model.get_model_summary()
        assert 'feature_disentanglement' in summary['components']
        assert summary['adversarial_training'] is True
    
    def test_model_initialization_without_disentanglement(self, model_configs):
        """Test model initialization without feature disentanglement."""
        model = CRAFTDFModel(
            adversarial_training=False,
            **model_configs
        )
        
        # Check that disentanglement is disabled
        assert model.feature_disentanglement is None
        assert model.adversarial_training is False
        
        # Check model summary
        summary = model.get_model_summary()
        assert 'feature_disentanglement' not in summary['components']
        assert summary['adversarial_training'] is False
    
    def test_forward_pass_with_disentanglement(self, model_configs, sample_batch):
        """Test forward pass with feature disentanglement enabled."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        spatial_input = sample_batch['spatial_input']
        frequency_input = sample_batch['frequency_input']
        domain_labels = sample_batch['domain_labels']
        
        # Forward pass
        outputs = model.forward(
            spatial_input, frequency_input, domain_labels,
            return_features=True, return_attention=True
        )
        
        # Check required outputs
        assert 'logits' in outputs
        assert 'predictions' in outputs
        assert 'invariant_features' in outputs
        assert 'specific_features' in outputs
        
        # Check shapes
        batch_size = spatial_input.shape[0]
        assert outputs['logits'].shape == (batch_size, model.num_classes)
        assert outputs['predictions'].shape == (batch_size, model.num_classes)
        assert outputs['invariant_features'].shape == (batch_size, 64)  # invariant_dim
        assert outputs['specific_features'].shape == (batch_size, 32)   # specific_dim
        
        # Check that all outputs are finite
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert torch.all(torch.isfinite(value)), f"{key} contains non-finite values"
    
    def test_forward_pass_without_disentanglement(self, model_configs, sample_batch):
        """Test forward pass without feature disentanglement."""
        model = CRAFTDFModel(
            adversarial_training=False,
            **model_configs
        )
        
        spatial_input = sample_batch['spatial_input']
        frequency_input = sample_batch['frequency_input']
        
        # Forward pass
        outputs = model.forward(
            spatial_input, frequency_input,
            return_features=True, return_attention=True
        )
        
        # Check required outputs
        assert 'logits' in outputs
        assert 'predictions' in outputs
        assert 'fused_features' in outputs
        
        # Should not have disentanglement outputs
        assert 'invariant_features' not in outputs
        assert 'specific_features' not in outputs
        assert 'disentanglement_losses' not in outputs
        
        # Check shapes
        batch_size = spatial_input.shape[0]
        assert outputs['logits'].shape == (batch_size, model.num_classes)
        assert outputs['predictions'].shape == (batch_size, model.num_classes)
    
    def test_training_step_with_disentanglement(self, model_configs, sample_batch):
        """Test training step with feature disentanglement."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        # Mock optimizers
        model.automatic_optimization = False
        opt_main = torch.optim.Adam(model.parameters(), lr=1e-4)
        opt_domain = torch.optim.Adam(model.feature_disentanglement.domain_classifier.parameters(), lr=1e-5)
        
        with patch.object(model, 'optimizers', return_value=[opt_main, opt_domain]):
            with patch.object(model, 'lr_schedulers', return_value=[None, None]):
                with patch.object(model, 'log'):
                    with patch.object(model, 'manual_backward'):
                        # Training step
                        loss = model.training_step(sample_batch, 0)
                        
                        # Check that loss is computed
                        assert isinstance(loss, torch.Tensor)
                        assert loss.dim() == 0  # Scalar loss
                        assert torch.isfinite(loss)
    
    def test_validation_step(self, model_configs, sample_batch):
        """Test validation step."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        with patch.object(model, 'log'):
            # Validation step
            outputs = model.validation_step(sample_batch, 0)
            
            # Check outputs
            assert 'val_loss' in outputs
            assert 'val_accuracy' in outputs
            assert 'predictions' in outputs
            assert 'labels' in outputs
            
            # Check types and shapes
            assert isinstance(outputs['val_loss'], torch.Tensor)
            assert torch.isfinite(outputs['val_loss'])
            
            batch_size = sample_batch['spatial_input'].shape[0]
            assert outputs['predictions'].shape == (batch_size,)
            assert outputs['labels'].shape == (batch_size,)
    
    def test_test_step(self, model_configs, sample_batch):
        """Test test step."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        with patch.object(model, 'log'):
            # Test step
            outputs = model.test_step(sample_batch, 0)
            
            # Check outputs
            assert 'test_loss' in outputs
            assert 'test_accuracy' in outputs
            assert 'predictions' in outputs
            assert 'labels' in outputs
            
            # Check that loss is finite
            assert torch.isfinite(outputs['test_loss'])
    
    def test_predict_step(self, model_configs, sample_batch):
        """Test prediction step."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        # Prediction step
        outputs = model.predict_step(sample_batch, 0)
        
        # Check outputs
        assert 'predictions' in outputs
        assert 'probabilities' in outputs
        assert 'logits' in outputs
        assert 'features' in outputs
        
        # Check shapes
        batch_size = sample_batch['spatial_input'].shape[0]
        assert outputs['predictions'].shape == (batch_size,)
        assert outputs['probabilities'].shape == (batch_size, model.num_classes)
        assert outputs['logits'].shape == (batch_size, model.num_classes)
        
        # Check features
        features = outputs['features']
        assert 'spatial' in features
        assert 'frequency' in features
        assert 'fused' in features
        assert 'invariant' in features
        assert 'specific' in features
    
    def test_configure_optimizers_with_disentanglement(self, model_configs):
        """Test optimizer configuration with adversarial training."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        # Mock trainer
        model.trainer = MagicMock()
        model.trainer.max_epochs = 100
        
        # Configure optimizers
        config = model.configure_optimizers()
        
        # Check that we have multiple optimizers for adversarial training
        assert 'optimizer' in config
        assert len(config['optimizer']) >= 1  # At least main optimizer
        
        # Check scheduler configuration
        if 'lr_scheduler' in config:
            assert len(config['lr_scheduler']) >= 0
    
    def test_configure_optimizers_without_disentanglement(self, model_configs):
        """Test optimizer configuration without adversarial training."""
        model = CRAFTDFModel(
            adversarial_training=False,
            **model_configs
        )
        
        # Mock trainer
        model.trainer = MagicMock()
        model.trainer.max_epochs = 100
        
        # Configure optimizers
        config = model.configure_optimizers()
        
        # Check that we have single optimizer
        assert 'optimizer' in config
        assert len(config['optimizer']) == 1  # Only main optimizer
    
    def test_analyze_feature_disentanglement(self, model_configs, sample_batch):
        """Test feature disentanglement analysis."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        spatial_input = sample_batch['spatial_input']
        frequency_input = sample_batch['frequency_input']
        domain_labels = sample_batch['domain_labels']
        
        # Analyze disentanglement
        analysis = model.analyze_feature_disentanglement(
            spatial_input, frequency_input, domain_labels
        )
        
        # Check analysis results
        expected_metrics = [
            'invariant_mean_norm', 'specific_mean_norm',
            'invariant_std_mean', 'specific_std_mean',
            'correlation_magnitude', 'domain_confusion',
            'reconstruction_error', 'separation_quality'
        ]
        
        for metric in expected_metrics:
            assert metric in analysis
            assert isinstance(analysis[metric], float)
            assert np.isfinite(analysis[metric])
    
    def test_analyze_feature_disentanglement_disabled(self, model_configs, sample_batch):
        """Test that analysis fails when disentanglement is disabled."""
        model = CRAFTDFModel(
            adversarial_training=False,
            **model_configs
        )
        
        spatial_input = sample_batch['spatial_input']
        frequency_input = sample_batch['frequency_input']
        domain_labels = sample_batch['domain_labels']
        
        # Should raise error when disentanglement is disabled
        with pytest.raises(ValueError, match="Feature disentanglement is not enabled"):
            model.analyze_feature_disentanglement(
                spatial_input, frequency_input, domain_labels
            )
    
    def test_model_summary(self, model_configs):
        """Test model summary generation."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        summary = model.get_model_summary()
        
        # Check required fields
        assert 'model_name' in summary
        assert 'total_parameters' in summary
        assert 'trainable_parameters' in summary
        assert 'parameter_efficiency' in summary
        assert 'adversarial_training' in summary
        assert 'num_classes' in summary
        assert 'components' in summary
        
        # Check components
        components = summary['components']
        expected_components = [
            'spatial_stream', 'frequency_stream', 
            'cross_attention', 'classifier', 'feature_disentanglement'
        ]
        
        for component in expected_components:
            assert component in components
            assert isinstance(components[component], int)
            assert components[component] > 0
    
    def test_gradient_flow_integration(self, model_configs, sample_batch):
        """Test gradient flow through the complete integrated model."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        spatial_input = sample_batch['spatial_input'].requires_grad_(True)
        frequency_input = sample_batch['frequency_input'].requires_grad_(True)
        domain_labels = sample_batch['domain_labels']
        labels = sample_batch['labels']
        
        # Forward pass
        outputs = model.forward(spatial_input, frequency_input, domain_labels)
        
        # Compute loss
        classification_loss = nn.CrossEntropyLoss()(outputs['logits'], labels)
        
        # Add disentanglement losses if available
        total_loss = classification_loss
        if 'disentanglement_losses' in outputs:
            disentanglement_losses = outputs['disentanglement_losses']
            if 'total' in disentanglement_losses:
                total_loss += disentanglement_losses['total']
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients flow to inputs
        assert spatial_input.grad is not None
        assert frequency_input.grad is not None
        assert torch.all(torch.isfinite(spatial_input.grad))
        assert torch.all(torch.isfinite(frequency_input.grad))
        
        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"
                assert torch.all(torch.isfinite(param.grad)), f"Parameter {name} gradients should be finite"
    
    def test_different_input_sizes(self, model_configs):
        """Test model with different input sizes."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        # Test different batch sizes
        batch_sizes = [1, 2, 8, 16]
        
        for batch_size in batch_sizes:
            spatial_input = torch.randn(batch_size, 3, 224, 224)
            
            # Create DWT coefficients for this batch size
            dwt_coefficients = {
                'll': torch.randn(batch_size, 3, 56, 56),
                'lh_1': torch.randn(batch_size, 3, 112, 112),
                'hl_1': torch.randn(batch_size, 3, 112, 112),
                'hh_1': torch.randn(batch_size, 3, 112, 112),
                'lh_2': torch.randn(batch_size, 3, 56, 56),
                'hl_2': torch.randn(batch_size, 3, 56, 56),
                'hh_2': torch.randn(batch_size, 3, 56, 56),
            }
            
            domain_labels = torch.randint(0, 3, (batch_size,))
            
            # Forward pass should work
            outputs = model.forward(spatial_input, dwt_coefficients, domain_labels)
            
            # Check output shapes
            assert outputs['logits'].shape[0] == batch_size
            assert outputs['predictions'].shape[0] == batch_size
            assert outputs['invariant_features'].shape[0] == batch_size
            assert outputs['specific_features'].shape[0] == batch_size
    
    def test_model_modes(self, model_configs, sample_batch):
        """Test model behavior in training vs evaluation mode."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        spatial_input = sample_batch['spatial_input']
        frequency_input = sample_batch['frequency_input']
        domain_labels = sample_batch['domain_labels']
        
        # Training mode
        model.train()
        train_outputs = model.forward(spatial_input, frequency_input, domain_labels)
        
        # Evaluation mode
        model.eval()
        eval_outputs = model.forward(spatial_input, frequency_input, domain_labels)
        
        # Outputs should have same structure
        assert set(train_outputs.keys()) == set(eval_outputs.keys())
        
        # Shapes should be the same
        for key in train_outputs.keys():
            if isinstance(train_outputs[key], torch.Tensor):
                assert train_outputs[key].shape == eval_outputs[key].shape
    
    def test_memory_efficiency(self, model_configs):
        """Test memory efficiency with larger inputs."""
        model = CRAFTDFModel(
            adversarial_training=True,
            **model_configs
        )
        
        # Test with larger batch
        batch_size = 32
        spatial_input = torch.randn(batch_size, 3, 224, 224)
        
        # Create DWT coefficients for larger batch
        dwt_coefficients = {
            'll': torch.randn(batch_size, 3, 56, 56),
            'lh_1': torch.randn(batch_size, 3, 112, 112),
            'hl_1': torch.randn(batch_size, 3, 112, 112),
            'hh_1': torch.randn(batch_size, 3, 112, 112),
            'lh_2': torch.randn(batch_size, 3, 56, 56),
            'hl_2': torch.randn(batch_size, 3, 56, 56),
            'hh_2': torch.randn(batch_size, 3, 56, 56),
        }
        
        domain_labels = torch.randint(0, 3, (batch_size,))
        
        # Should handle larger batches
        outputs = model.forward(spatial_input, dwt_coefficients, domain_labels)
        
        # Check that outputs are correct
        assert outputs['logits'].shape[0] == batch_size
        assert torch.all(torch.isfinite(outputs['logits']))


class TestEndToEndTraining:
    """End-to-end training tests."""
    
    def test_minimal_training_loop(self):
        """Test a minimal training loop with the complete model."""
        # Create small model for testing
        model_configs = {
            'spatial_config': {
                'pretrained': False,
                'freeze_layers': 2,
                'feature_dim': 64,
                'dropout_rate': 0.1
            },
            'frequency_config': {
                'input_channels': 3,
                'dwt_levels': 2,
                'feature_dim': 32,
                'dropout_rate': 0.1
            },
            'attention_config': {
                'spatial_dim': 64,
                'frequency_dim': 32,
                'embed_dim': 64,
                'num_heads': 2,
                'dropout_rate': 0.1
            },
            'disentanglement_config': {
                'input_dim': 64,
                'invariant_dim': 32,
                'specific_dim': 16,
                'num_domains': 2,
                'hidden_dim': 64
            }
        }
        
        model = CRAFTDFModel(
            adversarial_training=True,
            learning_rate=1e-3,
            **model_configs
        )
        
        # Create synthetic dataset
        batch_size = 8
        num_batches = 4
        
        spatial_data = torch.randn(num_batches * batch_size, 3, 224, 224)
        
        # Create DWT coefficients for the dataset
        dwt_data = {}
        for key in ['ll', 'lh_1', 'hl_1', 'hh_1', 'lh_2', 'hl_2', 'hh_2']:
            if key == 'll' or key.endswith('_2'):
                dwt_data[key] = torch.randn(num_batches * batch_size, 3, 56, 56)
            else:  # Level 1 coefficients
                dwt_data[key] = torch.randn(num_batches * batch_size, 3, 112, 112)
        
        labels = torch.randint(0, 2, (num_batches * batch_size,))
        domain_labels = torch.randint(0, 2, (num_batches * batch_size,))
        
        # Manual training loop (simplified)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            optimizer.zero_grad()
            
            # Get batch data
            spatial_input = spatial_data[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            batch_domain_labels = domain_labels[start_idx:end_idx]
            
            # Create DWT coefficients for this batch
            batch_dwt = {}
            for key, data in dwt_data.items():
                batch_dwt[key] = data[start_idx:end_idx]
            
            # Forward pass
            outputs = model.forward(
                spatial_input, batch_dwt, batch_domain_labels
            )
            
            # Compute loss
            classification_loss = nn.CrossEntropyLoss()(outputs['logits'], batch_labels)
            total_loss = classification_loss
            
            # Add disentanglement losses
            if 'disentanglement_losses' in outputs:
                disentanglement_losses = outputs['disentanglement_losses']
                if 'total' in disentanglement_losses:
                    total_loss += 0.1 * disentanglement_losses['total']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Check that loss is finite
            assert torch.isfinite(total_loss)
            
            # Only train for a few batches in test
            if batch_idx >= 2:
                break
        
        # Test evaluation
        model.eval()
        with torch.no_grad():
            # Create test batch
            test_spatial = spatial_data[:batch_size]
            test_labels = labels[:batch_size]
            
            test_dwt = {}
            for key, data in dwt_data.items():
                test_dwt[key] = data[:batch_size]
            
            outputs = model.forward(test_spatial, test_dwt)
            predictions = torch.argmax(outputs['logits'], dim=1)
            
            # Check that predictions are valid
            assert predictions.shape == test_labels.shape
            assert torch.all((predictions >= 0) & (predictions < model.num_classes))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])