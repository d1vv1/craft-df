"""
Integration tests for training and validation logic

This module tests the comprehensive training and validation logic implemented
in task 8.2, including loss computation, metric tracking, and optimizer configuration.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import pytorch_lightning as pl

from craft_df.models.craft_df_model import CRAFTDFModel


class TestTrainingValidationIntegration:
    """Test suite for training and validation logic integration."""
    
    @pytest.fixture
    def model_config(self):
        """Model configuration for testing."""
        return {
            'spatial_config': {
                'pretrained': False,
                'freeze_layers': 2,
                'feature_dim': 32,
                'dropout_rate': 0.1
            },
            'frequency_config': {
                'input_channels': 3,
                'dwt_levels': 2,
                'feature_dim': 16,
                'dropout_rate': 0.1
            },
            'attention_config': {
                'spatial_dim': 32,
                'frequency_dim': 16,
                'embed_dim': 32,
                'num_heads': 2,
                'dropout_rate': 0.1
            },
            'disentanglement_config': {
                'input_dim': 32,
                'invariant_dim': 16,
                'specific_dim': 8,
                'num_domains': 2,
                'hidden_dim': 32
            },
            'num_classes': 2,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'scheduler_type': 'cosine',
            'adversarial_training': True,
            'domain_adaptation_weight': 0.1,
            'reconstruction_weight': 0.01,
            'orthogonality_weight': 0.01
        }
    
    @pytest.fixture
    def sample_batch(self):
        """Generate sample batch for testing."""
        batch_size = 4
        
        dwt_coefficients = {
            'll': torch.randn(batch_size, 3, 28, 28),
            'lh_1': torch.randn(batch_size, 3, 56, 56),
            'hl_1': torch.randn(batch_size, 3, 56, 56),
            'hh_1': torch.randn(batch_size, 3, 56, 56),
            'lh_2': torch.randn(batch_size, 3, 28, 28),
            'hl_2': torch.randn(batch_size, 3, 28, 28),
            'hh_2': torch.randn(batch_size, 3, 28, 28),
        }
        
        return {
            'spatial_input': torch.randn(batch_size, 3, 224, 224),
            'frequency_input': dwt_coefficients,
            'labels': torch.randint(0, 2, (batch_size,)),
            'domain_labels': torch.randint(0, 2, (batch_size,))
        }
    
    def test_training_step_loss_computation(self, model_config, sample_batch):
        """Test comprehensive loss computation in training step."""
        model = CRAFTDFModel(**model_config)
        
        # Mock trainer and optimizers
        model._trainer = MagicMock()
        model._trainer.is_last_batch = False
        
        opt_main = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        with patch.object(model, 'optimizers', return_value=[opt_main]):
            with patch.object(model, 'lr_schedulers', return_value=[]):
                with patch.object(model, 'log') as mock_log:
                    with patch.object(model, 'manual_backward') as mock_backward:
                        # Execute training step
                        loss = model.training_step(sample_batch, 0)
                        
                        # Verify loss properties
                        assert isinstance(loss, torch.Tensor)
                        assert loss.dim() == 0  # Scalar
                        assert torch.isfinite(loss)
                        assert loss.item() > 0
                        
                        # Verify backward was called
                        mock_backward.assert_called_once_with(loss)
                        
                        # Check logged metrics
                        logged_metrics = {call[0][0]: call[0][1] for call in mock_log.call_args_list}
                        
                        # Required metrics
                        assert 'train/classification_loss' in logged_metrics
                        assert 'train/total_loss' in logged_metrics
                        assert 'train/accuracy' in logged_metrics
                        
                        # Disentanglement metrics (should be present with adversarial training)
                        assert 'train/adversarial_loss' in logged_metrics
                        assert 'train/reconstruction_loss' in logged_metrics
                        
                        # Verify metric values are reasonable
                        assert torch.isfinite(logged_metrics['train/classification_loss'])
                        assert torch.isfinite(logged_metrics['train/total_loss'])
                        assert 0.0 <= logged_metrics['train/accuracy'] <= 1.0
    
    def test_validation_step_comprehensive_metrics(self, model_config, sample_batch):
        """Test comprehensive metric computation in validation step."""
        model = CRAFTDFModel(**model_config)
        
        with patch.object(model, 'log') as mock_log:
            # Execute validation step
            outputs = model.validation_step(sample_batch, 0)
            
            # Check required outputs
            required_keys = [
                'val_loss', 'val_accuracy', 'predictions', 'labels',
                'probabilities', 'confidence_scores', 'mean_confidence'
            ]
            for key in required_keys:
                assert key in outputs, f"Missing validation output: {key}"
            
            # Check output properties
            batch_size = sample_batch['spatial_input'].shape[0]
            assert outputs['predictions'].shape == (batch_size,)
            assert outputs['labels'].shape == (batch_size,)
            assert outputs['probabilities'].shape == (batch_size, model_config['num_classes'])
            assert outputs['confidence_scores'].shape == (batch_size,)
            
            # Check that all values are finite
            assert torch.all(torch.isfinite(outputs['val_loss']))
            assert torch.all(torch.isfinite(outputs['probabilities']))
            assert torch.all(torch.isfinite(outputs['confidence_scores']))
            
            # Check logged metrics
            logged_metrics = {call[0][0]: call[0][1] for call in mock_log.call_args_list}
            
            # Required validation metrics
            assert 'val/classification_loss' in logged_metrics
            assert 'val/accuracy' in logged_metrics
            assert 'val/mean_confidence' in logged_metrics
            
            # Feature analysis metrics (should be present with disentanglement)
            assert 'val/invariant_feature_norm' in logged_metrics
            assert 'val/specific_feature_norm' in logged_metrics
            assert 'val/invariant_diversity' in logged_metrics
            assert 'val/specific_diversity' in logged_metrics
    
    def test_optimizer_configuration_comprehensive(self, model_config):
        """Test comprehensive optimizer and scheduler configuration."""
        model = CRAFTDFModel(**model_config)
        
        # Mock trainer for scheduler configuration
        model._trainer = MagicMock()
        model._trainer.max_epochs = 50
        
        # Configure optimizers
        config = model.configure_optimizers()
        
        # Check configuration structure
        assert 'optimizer' in config
        assert 'lr_scheduler' in config
        
        # Check optimizers
        optimizers = config['optimizer']
        assert len(optimizers) >= 1  # At least main optimizer
        
        # Main optimizer should be AdamW
        main_optimizer = optimizers[0]
        assert isinstance(main_optimizer, torch.optim.AdamW)
        assert main_optimizer.param_groups[0]['lr'] == model_config['learning_rate']
        assert main_optimizer.param_groups[0]['weight_decay'] == model_config['weight_decay']
        
        # Check schedulers
        schedulers = config['lr_scheduler']
        if schedulers:
            for scheduler_config in schedulers:
                assert isinstance(scheduler_config, dict)
                assert 'scheduler' in scheduler_config
                assert 'interval' in scheduler_config
                assert 'frequency' in scheduler_config
    
    def test_different_scheduler_configurations(self, model_config):
        """Test different learning rate scheduler configurations."""
        scheduler_types = ['cosine', 'plateau', 'warmup_cosine']
        
        for scheduler_type in scheduler_types:
            config = model_config.copy()
            config['scheduler_type'] = scheduler_type
            
            model = CRAFTDFModel(**config)
            model._trainer = MagicMock()
            model._trainer.max_epochs = 50
            
            # Should configure without errors
            optimizer_config = model.configure_optimizers()
            
            assert 'optimizer' in optimizer_config
            assert 'lr_scheduler' in optimizer_config
            
            # Check that schedulers are properly configured
            schedulers = optimizer_config['lr_scheduler']
            if scheduler_type != 'none':  # Some scheduler types may not create schedulers
                # At least verify the structure is correct
                if schedulers:
                    for scheduler_config in schedulers:
                        assert 'scheduler' in scheduler_config
    
    def test_training_without_adversarial(self, model_config, sample_batch):
        """Test training step without adversarial training."""
        config = model_config.copy()
        config['adversarial_training'] = False
        
        model = CRAFTDFModel(**config)
        
        with patch.object(model, 'log') as mock_log:
            # Execute training step (should use automatic optimization)
            loss = model.training_step(sample_batch, 0)
            
            # Check loss is computed
            assert isinstance(loss, torch.Tensor)
            assert torch.isfinite(loss)
            
            # Check logged metrics
            logged_metrics = {call[0][0]: call[0][1] for call in mock_log.call_args_list}
            
            # Should have basic metrics
            assert 'train/classification_loss' in logged_metrics
            assert 'train/total_loss' in logged_metrics
            assert 'train/accuracy' in logged_metrics
            
            # Should not have adversarial metrics
            assert 'train/adversarial_loss' not in logged_metrics
            assert 'train/domain_accuracy' not in logged_metrics
    
    def test_validation_without_adversarial(self, model_config, sample_batch):
        """Test validation step without adversarial training."""
        config = model_config.copy()
        config['adversarial_training'] = False
        
        model = CRAFTDFModel(**config)
        
        with patch.object(model, 'log') as mock_log:
            # Execute validation step
            outputs = model.validation_step(sample_batch, 0)
            
            # Check basic outputs are present
            assert 'val_loss' in outputs
            assert 'val_accuracy' in outputs
            
            # Check logged metrics
            logged_metrics = {call[0][0]: call[0][1] for call in mock_log.call_args_list}
            
            # Should have basic validation metrics
            assert 'val/classification_loss' in logged_metrics
            assert 'val/accuracy' in logged_metrics
            
            # Should not have disentanglement-specific metrics
            disentanglement_metrics = [key for key in logged_metrics.keys() 
                                     if 'domain_confusion' in key or 'invariant' in key]
            assert len(disentanglement_metrics) == 0
    
    def test_metric_tracking_consistency(self, model_config, sample_batch):
        """Test consistency of metric tracking across training and validation."""
        model = CRAFTDFModel(**model_config)
        
        # Mock trainer
        model._trainer = MagicMock()
        model._trainer.is_last_batch = False
        
        opt_main = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Training step
        with patch.object(model, 'optimizers', return_value=[opt_main]):
            with patch.object(model, 'lr_schedulers', return_value=[]):
                with patch.object(model, 'log') as train_log:
                    with patch.object(model, 'manual_backward'):
                        train_loss = model.training_step(sample_batch, 0)
                        train_metrics = {call[0][0]: call[0][1] for call in train_log.call_args_list}
        
        # Validation step
        with patch.object(model, 'log') as val_log:
            val_outputs = model.validation_step(sample_batch, 0)
            val_metrics = {call[0][0]: call[0][1] for call in val_log.call_args_list}
        
        # Check that both have accuracy metrics
        assert 'train/accuracy' in train_metrics
        assert 'val/accuracy' in val_metrics
        
        # Check that both accuracy values are in valid range
        assert 0.0 <= train_metrics['train/accuracy'] <= 1.0
        assert 0.0 <= val_metrics['val/accuracy'] <= 1.0
        
        # Check that both have loss metrics
        assert 'train/classification_loss' in train_metrics
        assert 'val/classification_loss' in val_metrics
        
        # Both losses should be positive and finite
        assert train_metrics['train/classification_loss'] > 0
        assert val_metrics['val/classification_loss'] > 0
        assert torch.isfinite(train_metrics['train/classification_loss'])
        assert torch.isfinite(val_metrics['val/classification_loss'])
    
    def test_gradient_clipping_integration(self, model_config, sample_batch):
        """Test gradient clipping integration in training step."""
        model = CRAFTDFModel(**model_config)
        
        # Mock trainer and optimizers
        model._trainer = MagicMock()
        model._trainer.is_last_batch = False
        
        opt_main = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        with patch.object(model, 'optimizers', return_value=[opt_main]):
            with patch.object(model, 'lr_schedulers', return_value=[]):
                with patch.object(model, 'log'):
                    with patch.object(model, 'manual_backward') as mock_backward:
                        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
                            # Execute training step
                            loss = model.training_step(sample_batch, 0)
                            
                            # Verify gradient clipping was called
                            mock_clip.assert_called_once()
                            args, kwargs = mock_clip.call_args
                            assert args[0] == model.parameters()
                            assert kwargs.get('max_norm', args[1] if len(args) > 1 else None) == 1.0
    
    def test_learning_rate_logging(self, model_config, sample_batch):
        """Test learning rate logging in training step."""
        model = CRAFTDFModel(**model_config)
        
        # Mock trainer and optimizers
        model._trainer = MagicMock()
        model._trainer.is_last_batch = False
        
        opt_main = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        with patch.object(model, 'optimizers', return_value=[opt_main]):
            with patch.object(model, 'lr_schedulers', return_value=[]):
                with patch.object(model, 'log') as mock_log:
                    with patch.object(model, 'manual_backward'):
                        # Execute training step
                        loss = model.training_step(sample_batch, 0)
                        
                        # Check that learning rate was logged
                        logged_metrics = {call[0][0]: call[0][1] for call in mock_log.call_args_list}
                        assert 'train/learning_rate' in logged_metrics
                        assert logged_metrics['train/learning_rate'] == 1e-3
    
    def test_batch_size_logging(self, model_config, sample_batch):
        """Test batch size logging in training and validation steps."""
        model = CRAFTDFModel(**model_config)
        
        # Training step
        model._trainer = MagicMock()
        model._trainer.is_last_batch = False
        opt_main = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        with patch.object(model, 'optimizers', return_value=[opt_main]):
            with patch.object(model, 'lr_schedulers', return_value=[]):
                with patch.object(model, 'log') as train_log:
                    with patch.object(model, 'manual_backward'):
                        model.training_step(sample_batch, 0)
                        
                        train_metrics = {call[0][0]: call[0][1] for call in train_log.call_args_list}
                        assert 'train/batch_size' in train_metrics
                        assert train_metrics['train/batch_size'] == float(sample_batch['spatial_input'].shape[0])
        
        # Validation step
        with patch.object(model, 'log') as val_log:
            model.validation_step(sample_batch, 0)
            
            val_metrics = {call[0][0]: call[0][1] for call in val_log.call_args_list}
            assert 'val/batch_size' in val_metrics
            assert val_metrics['val/batch_size'] == float(sample_batch['spatial_input'].shape[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])