"""
Unit tests for complete CRAFT-DF Model implementation

This module contains comprehensive unit tests for the main CRAFT-DF model,
focusing on the integration of all components and the training/validation logic
implemented in task 8.1 and 8.2.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

# Import the module under test
from craft_df.models.craft_df_model import CRAFTDFModel


class TestCRAFTDFModelComplete:
    """Test suite for complete CRAFT-DF model implementation."""
    
    @pytest.fixture
    def model_config(self):
        """Standard model configuration for testing."""
        return {
            'spatial_config': {
                'pretrained': False,
                'freeze_layers': 3,
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
                'num_heads': 4,
                'dropout_rate': 0.1
            },
            'disentanglement_config': {
                'input_dim': 64,
                'invariant_dim': 32,
                'specific_dim': 16,
                'num_domains': 3,
                'hidden_dim': 64
            },
            'num_classes': 2,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'scheduler_type': 'cosine',
            'adversarial_training': True
        }
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        batch_size = 4
        
        # DWT coefficients in expected format
        dwt_coefficients = {
            'll': torch.randn(batch_size, 3, 56, 56),
            'lh_1': torch.randn(batch_size, 3, 112, 112),
            'hl_1': torch.randn(batch_size, 3, 112, 112),
            'hh_1': torch.randn(batch_size, 3, 112, 112),
            'lh_2': torch.randn(batch_size, 3, 56, 56),
            'hl_2': torch.randn(batch_size, 3, 56, 56),
            'hh_2': torch.randn(batch_size, 3, 56, 56),
        }
        
        return {
            'spatial_input': torch.randn(batch_size, 3, 224, 224),
            'frequency_input': dwt_coefficients,
            'labels': torch.randint(0, 2, (batch_size,)),
            'domain_labels': torch.randint(0, 3, (batch_size,))
        }
    
    def test_model_initialization_complete(self, model_config):
        """Test complete model initialization with all components."""
        model = CRAFTDFModel(**model_config)
        
        # Check all components are initialized
        assert hasattr(model, 'spatial_stream')
        assert hasattr(model, 'frequency_stream')
        assert hasattr(model, 'cross_attention')
        assert hasattr(model, 'feature_disentanglement')
        assert hasattr(model, 'classifier')
        
        # Check configuration is stored
        assert model.adversarial_training == model_config['adversarial_training']
        assert model.num_classes == model_config['num_classes']
        assert model.learning_rate == model_config['learning_rate']
        
        # Check that model is a PyTorch Lightning module
        assert isinstance(model, pl.LightningModule)
        
        # Check automatic optimization setting
        assert model.automatic_optimization == False  # Manual optimization for adversarial training
    
    def test_forward_pass_complete_integration(self, model_config, sample_data):
        """Test complete forward pass through all integrated components."""
        model = CRAFTDFModel(**model_config)
        
        spatial_input = sample_data['spatial_input']
        frequency_input = sample_data['frequency_input']
        domain_labels = sample_data['domain_labels']
        
        # Test forward pass with all features
        outputs = model.forward(
            spatial_input, frequency_input, domain_labels,
            return_features=True, return_attention=True
        )
        
        # Check all expected outputs are present
        expected_keys = [
            'logits', 'predictions', 'invariant_features', 'specific_features',
            'spatial_features', 'frequency_features', 'fused_features',
            'classification_features', 'attention_weights', 'disentanglement_losses'
        ]
        
        for key in expected_keys:
            assert key in outputs, f"Missing output key: {key}"
        
        # Check output shapes
        batch_size = spatial_input.shape[0]
        assert outputs['logits'].shape == (batch_size, model_config['num_classes'])
        assert outputs['predictions'].shape == (batch_size, model_config['num_classes'])
        assert outputs['invariant_features'].shape == (batch_size, model_config['disentanglement_config']['invariant_dim'])
        assert outputs['specific_features'].shape == (batch_size, model_config['disentanglement_config']['specific_dim'])
        
        # Check that all outputs are finite
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert torch.all(torch.isfinite(value)), f"Non-finite values in {key}"
    
    def test_forward_pass_without_disentanglement(self, model_config, sample_data):
        """Test forward pass without feature disentanglement."""
        model_config_no_adv = model_config.copy()
        model_config_no_adv['adversarial_training'] = False
        
        model = CRAFTDFModel(**model_config_no_adv)
        
        spatial_input = sample_data['spatial_input']
        frequency_input = sample_data['frequency_input']
        
        outputs = model.forward(
            spatial_input, frequency_input,
            return_features=True, return_attention=True
        )
        
        # Should have basic outputs but not disentanglement
        assert 'logits' in outputs
        assert 'predictions' in outputs
        assert 'fused_features' in outputs
        assert 'attention_weights' in outputs
        
        # Should not have disentanglement outputs
        assert 'invariant_features' not in outputs
        assert 'specific_features' not in outputs
        assert 'disentanglement_losses' not in outputs
    
    def test_training_step_comprehensive(self, model_config, sample_data):
        """Test comprehensive training step implementation."""
        model = CRAFTDFModel(**model_config)
        
        # Mock the required PyTorch Lightning components
        model.trainer = MagicMock()
        model.trainer.is_last_batch = False
        
        # Mock optimizers and schedulers
        opt_main = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt_domain = torch.optim.Adam(model.feature_disentanglement.domain_classifier.parameters(), lr=1e-5)
        
        with patch.object(model, 'optimizers', return_value=[opt_main, opt_domain]):
            with patch.object(model, 'lr_schedulers', return_value=[None, None]):
                with patch.object(model, 'log') as mock_log:
                    with patch.object(model, 'manual_backward') as mock_backward:
                        # Execute training step
                        loss = model.training_step(sample_data, 0)
                        
                        # Check that loss is computed correctly
                        assert isinstance(loss, torch.Tensor)
                        assert loss.dim() == 0  # Scalar
                        assert torch.isfinite(loss)
                        assert loss.item() > 0  # Should be positive
                        
                        # Check that logging was called
                        assert mock_log.called
                        
                        # Check that backward was called
                        assert mock_backward.called
                        
                        # Verify logged metrics
                        logged_metrics = [call[0][0] for call in mock_log.call_args_list]
                        expected_metrics = [
                            'train/classification_loss', 'train/total_loss', 'train/accuracy'
                        ]
                        for metric in expected_metrics:
                            assert metric in logged_metrics, f"Missing logged metric: {metric}"
    
    def test_validation_step_comprehensive(self, model_config, sample_data):
        """Test comprehensive validation step implementation."""
        model = CRAFTDFModel(**model_config)
        
        with patch.object(model, 'log') as mock_log:
            # Execute validation step
            outputs = model.validation_step(sample_data, 0)
            
            # Check required outputs
            expected_keys = [
                'val_loss', 'val_accuracy', 'predictions', 'labels',
                'probabilities', 'confidence_scores', 'mean_confidence'
            ]
            for key in expected_keys:
                assert key in outputs, f"Missing validation output: {key}"
            
            # Check output types and shapes
            batch_size = sample_data['spatial_input'].shape[0]
            assert isinstance(outputs['val_loss'], torch.Tensor)
            assert outputs['predictions'].shape == (batch_size,)
            assert outputs['labels'].shape == (batch_size,)
            assert outputs['probabilities'].shape == (batch_size, model_config['num_classes'])
            
            # Check that logging was called
            assert mock_log.called
            
            # Verify logged metrics
            logged_metrics = [call[0][0] for call in mock_log.call_args_list]
            expected_metrics = ['val/classification_loss', 'val/accuracy']
            for metric in expected_metrics:
                assert metric in logged_metrics, f"Missing logged metric: {metric}"
    
    def test_configure_optimizers_comprehensive(self, model_config):
        """Test comprehensive optimizer configuration."""
        model = CRAFTDFModel(**model_config)
        
        # Mock trainer for scheduler configuration
        model.trainer = MagicMock()
        model.trainer.max_epochs = 100
        
        # Configure optimizers
        config = model.configure_optimizers()
        
        # Check configuration structure
        assert 'optimizer' in config
        assert 'lr_scheduler' in config
        
        # Check optimizers
        optimizers = config['optimizer']
        assert len(optimizers) >= 1  # At least main optimizer
        assert isinstance(optimizers[0], torch.optim.AdamW)
        
        # Check schedulers
        schedulers = config['lr_scheduler']
        if schedulers:  # May be empty for some scheduler types
            for scheduler_config in schedulers:
                assert 'scheduler' in scheduler_config
                assert 'interval' in scheduler_config
                assert 'frequency' in scheduler_config
    
    def test_different_scheduler_types(self, model_config):
        """Test different learning rate scheduler configurations."""
        scheduler_types = ['cosine', 'plateau', 'warmup_cosine']
        
        for scheduler_type in scheduler_types:
            config = model_config.copy()
            config['scheduler_type'] = scheduler_type
            
            model = CRAFTDFModel(**config)
            model.trainer = MagicMock()
            model.trainer.max_epochs = 100
            
            # Should not raise an error
            optimizer_config = model.configure_optimizers()
            assert 'optimizer' in optimizer_config
    
    def test_model_summary_complete(self, model_config):
        """Test complete model summary generation."""
        model = CRAFTDFModel(**model_config)
        
        summary = model.get_model_summary()
        
        # Check all required fields
        required_fields = [
            'model_name', 'total_parameters', 'trainable_parameters',
            'parameter_efficiency', 'adversarial_training', 'num_classes', 'components'
        ]
        for field in required_fields:
            assert field in summary, f"Missing summary field: {field}"
        
        # Check components
        components = summary['components']
        expected_components = [
            'spatial_stream', 'frequency_stream', 'cross_attention',
            'classifier', 'feature_disentanglement'
        ]
        for component in expected_components:
            assert component in components, f"Missing component: {component}"
            assert isinstance(components[component], int)
            assert components[component] > 0
        
        # Check parameter counts are reasonable
        assert summary['total_parameters'] > 0
        assert summary['trainable_parameters'] > 0
        assert summary['parameter_efficiency'] <= 1.0
    
    def test_feature_disentanglement_analysis(self, model_config, sample_data):
        """Test feature disentanglement analysis functionality."""
        model = CRAFTDFModel(**model_config)
        
        spatial_input = sample_data['spatial_input']
        frequency_input = sample_data['frequency_input']
        domain_labels = sample_data['domain_labels']
        
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
            assert metric in analysis, f"Missing analysis metric: {metric}"
            assert isinstance(analysis[metric], float)
            assert np.isfinite(analysis[metric])
    
    def test_predict_step_complete(self, model_config, sample_data):
        """Test complete prediction step functionality."""
        model = CRAFTDFModel(**model_config)
        
        # Execute prediction step
        outputs = model.predict_step(sample_data, 0)
        
        # Check all expected outputs
        expected_keys = [
            'predictions', 'probabilities', 'logits', 'attention_weights', 'features'
        ]
        for key in expected_keys:
            assert key in outputs, f"Missing prediction output: {key}"
        
        # Check features dictionary
        features = outputs['features']
        expected_features = ['spatial', 'frequency', 'fused', 'invariant', 'specific']
        for feature_type in expected_features:
            assert feature_type in features, f"Missing feature type: {feature_type}"
        
        # Check shapes
        batch_size = sample_data['spatial_input'].shape[0]
        assert outputs['predictions'].shape == (batch_size,)
        assert outputs['probabilities'].shape == (batch_size, model_config['num_classes'])
        assert outputs['logits'].shape == (batch_size, model_config['num_classes'])
    
    def test_error_handling_forward_pass(self, model_config):
        """Test error handling in forward pass."""
        model = CRAFTDFModel(**model_config)
        
        # Test with invalid spatial input shape
        with pytest.raises(AssertionError):
            invalid_spatial = torch.randn(4, 3, 100, 100)  # Wrong size
            valid_frequency = {
                'll': torch.randn(4, 3, 56, 56),
                'lh_1': torch.randn(4, 3, 112, 112),
                'hl_1': torch.randn(4, 3, 112, 112),
                'hh_1': torch.randn(4, 3, 112, 112),
                'lh_2': torch.randn(4, 3, 56, 56),
                'hl_2': torch.randn(4, 3, 56, 56),
                'hh_2': torch.randn(4, 3, 56, 56),
            }
            model.forward(invalid_spatial, valid_frequency)
        
        # Test with batch size mismatch
        with pytest.raises(AssertionError):
            valid_spatial = torch.randn(4, 3, 224, 224)
            mismatched_frequency = {
                'll': torch.randn(3, 3, 56, 56),  # Different batch size
                'lh_1': torch.randn(3, 3, 112, 112),
                'hl_1': torch.randn(3, 3, 112, 112),
                'hh_1': torch.randn(3, 3, 112, 112),
                'lh_2': torch.randn(3, 3, 56, 56),
                'hl_2': torch.randn(3, 3, 56, 56),
                'hh_2': torch.randn(3, 3, 56, 56),
            }
            model.forward(valid_spatial, mismatched_frequency)
    
    def test_gradient_flow_complete(self, model_config, sample_data):
        """Test gradient flow through complete integrated model."""
        model = CRAFTDFModel(**model_config)
        
        spatial_input = sample_data['spatial_input'].requires_grad_(True)
        
        # Set requires_grad for each tensor in the frequency input dictionary
        frequency_input = {}
        for key, tensor in sample_data['frequency_input'].items():
            frequency_input[key] = tensor.requires_grad_(True)
        
        domain_labels = sample_data['domain_labels']
        labels = sample_data['labels']
        
        # Forward pass
        outputs = model.forward(spatial_input, frequency_input, domain_labels)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(outputs['logits'], labels)
        
        # Add disentanglement losses
        if 'disentanglement_losses' in outputs:
            disentanglement_losses = outputs['disentanglement_losses']
            if 'total' in disentanglement_losses:
                loss += 0.1 * disentanglement_losses['total']
        
        # Backward pass
        loss.backward()
        
        # Check gradients flow to inputs
        assert spatial_input.grad is not None
        assert torch.all(torch.isfinite(spatial_input.grad))
        
        # Check gradients flow to frequency inputs
        for key, tensor in frequency_input.items():
            assert tensor.grad is not None, f"No gradient for frequency input: {key}"
            assert torch.all(torch.isfinite(tensor.grad)), f"Non-finite gradients for frequency input: {key}"
        
        # Check model parameter gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"
                assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradients for: {name}"
    
    def test_model_modes_training_eval(self, model_config, sample_data):
        """Test model behavior in training vs evaluation modes."""
        model = CRAFTDFModel(**model_config)
        
        spatial_input = sample_data['spatial_input']
        frequency_input = sample_data['frequency_input']
        domain_labels = sample_data['domain_labels']
        
        # Training mode
        model.train()
        train_outputs = model.forward(spatial_input, frequency_input, domain_labels)
        
        # Evaluation mode
        model.eval()
        with torch.no_grad():
            eval_outputs = model.forward(spatial_input, frequency_input, domain_labels)
        
        # Check that core outputs are present in both modes
        core_keys = ['logits', 'predictions', 'invariant_features', 'specific_features']
        for key in core_keys:
            assert key in train_outputs, f"Missing {key} in training outputs"
            assert key in eval_outputs, f"Missing {key} in evaluation outputs"
        
        # Check that shapes are consistent for core outputs
        for key in core_keys:
            if isinstance(train_outputs[key], torch.Tensor) and isinstance(eval_outputs[key], torch.Tensor):
                assert train_outputs[key].shape == eval_outputs[key].shape, f"Shape mismatch for {key}"
        
        # Disentanglement losses should only be present in training mode
        if model.adversarial_training:
            assert 'disentanglement_losses' in train_outputs, "Missing disentanglement losses in training mode"
            assert 'disentanglement_losses' not in eval_outputs, "Disentanglement losses should not be present in eval mode"
    
    def test_memory_efficiency_large_batch(self, model_config):
        """Test memory efficiency with larger batch sizes."""
        model = CRAFTDFModel(**model_config)
        
        # Test with larger batch
        batch_size = 16
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
        
        # Should handle larger batches without issues
        outputs = model.forward(spatial_input, dwt_coefficients, domain_labels)
        
        # Check outputs are correct
        assert outputs['logits'].shape[0] == batch_size
        assert torch.all(torch.isfinite(outputs['logits']))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])