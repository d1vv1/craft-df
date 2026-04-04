"""
Unit tests for SpatialStream module.

This module contains comprehensive tests for the MobileNetV2-based spatial feature
extractor, including functionality tests, gradient flow validation, and numerical
stability checks.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from craft_df.models.spatial_stream import SpatialStream


class TestSpatialStream:
    """Test suite for SpatialStream class."""
    
    @pytest.fixture
    def device(self):
        """Get available device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_input(self, device):
        """Create sample input tensor for testing."""
        return torch.randn(4, 3, 224, 224, device=device)
    
    @pytest.fixture
    def spatial_stream(self, device):
        """Create SpatialStream instance for testing."""
        model = SpatialStream(
            pretrained=False,  # Use False for faster testing
            freeze_layers=5,
            feature_dim=512,
            dropout_rate=0.1
        )
        return model.to(device)
    
    def test_initialization_default_params(self):
        """Test SpatialStream initialization with default parameters."""
        model = SpatialStream(pretrained=False)
        
        assert model.feature_dim == 1280
        assert model.freeze_layers == 10
        assert isinstance(model.backbone, nn.Module)
        assert isinstance(model.classifier, nn.Sequential)
    
    def test_initialization_custom_params(self):
        """Test SpatialStream initialization with custom parameters."""
        model = SpatialStream(
            pretrained=False,
            freeze_layers=5,
            feature_dim=512,
            dropout_rate=0.2
        )
        
        assert model.feature_dim == 512
        assert model.freeze_layers == 5
    
    def test_initialization_invalid_params(self):
        """Test SpatialStream initialization with invalid parameters."""
        # Test invalid freeze_layers
        with pytest.raises(AssertionError):
            SpatialStream(freeze_layers=-1)
        
        with pytest.raises(AssertionError):
            SpatialStream(freeze_layers=20)
        
        # Test invalid feature_dim
        with pytest.raises(AssertionError):
            SpatialStream(feature_dim=0)
        
        with pytest.raises(AssertionError):
            SpatialStream(feature_dim=-100)
        
        # Test invalid dropout_rate
        with pytest.raises(AssertionError):
            SpatialStream(dropout_rate=-0.1)
        
        with pytest.raises(AssertionError):
            SpatialStream(dropout_rate=1.5)
    
    def test_forward_pass_shape(self, spatial_stream, sample_input):
        """Test forward pass output shape."""
        output = spatial_stream(sample_input)
        
        expected_shape = (sample_input.shape[0], spatial_stream.feature_dim)
        assert output.shape == expected_shape
        assert output.dtype == torch.float32
    
    def test_forward_pass_invalid_input_shape(self, spatial_stream):
        """Test forward pass with invalid input shapes."""
        # Test 3D input
        with pytest.raises(AssertionError):
            invalid_input = torch.randn(4, 3, 224)
            spatial_stream(invalid_input)
        
        # Test wrong number of channels
        with pytest.raises(AssertionError):
            invalid_input = torch.randn(4, 1, 224, 224)
            spatial_stream(invalid_input)
        
        # Test wrong spatial dimensions
        with pytest.raises(AssertionError):
            invalid_input = torch.randn(4, 3, 128, 128)
            spatial_stream(invalid_input)
    
    def test_gradient_flow(self, spatial_stream, sample_input):
        """Test gradient flow through the model."""
        spatial_stream.train()
        
        # Forward pass
        output = spatial_stream(sample_input)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed for trainable parameters
        trainable_params = [p for p in spatial_stream.parameters() if p.requires_grad]
        assert len(trainable_params) > 0
        
        for param in trainable_params:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()
    
    def test_layer_freezing(self):
        """Test layer freezing functionality."""
        model = SpatialStream(pretrained=False, freeze_layers=5)
        
        # Check that first 5 layers are frozen
        features = model.backbone.features
        for i in range(min(5, len(features))):
            for param in features[i].parameters():
                assert not param.requires_grad
        
        # Check that remaining layers are trainable
        for i in range(5, len(features)):
            for param in features[i].parameters():
                assert param.requires_grad
    
    def test_unfreeze_layers(self):
        """Test unfreezing layers functionality."""
        model = SpatialStream(pretrained=False, freeze_layers=10)
        
        # Unfreeze 3 layers
        model.unfreeze_layers(3)
        assert model.freeze_layers == 7
        
        # Check that layers 7-9 are now trainable
        features = model.backbone.features
        for i in range(7, min(10, len(features))):
            for param in features[i].parameters():
                assert param.requires_grad
    
    def test_get_feature_maps(self, spatial_stream, sample_input):
        """Test feature map extraction."""
        feature_maps = spatial_stream.get_feature_maps(sample_input)
        
        # MobileNetV2 feature maps should be (batch_size, 1280, 7, 7)
        expected_shape = (sample_input.shape[0], 1280, 7, 7)
        assert feature_maps.shape == expected_shape
    
    def test_get_trainable_parameters(self, spatial_stream):
        """Test trainable parameter counting."""
        trainable_count = spatial_stream.get_trainable_parameters()
        
        # Manually count trainable parameters
        manual_count = sum(p.numel() for p in spatial_stream.parameters() if p.requires_grad)
        
        assert trainable_count == manual_count
        assert trainable_count > 0
    
    def test_get_model_info(self, spatial_stream):
        """Test model information extraction."""
        info = spatial_stream.get_model_info()
        
        required_keys = [
            'model_name', 'backbone', 'total_parameters', 
            'trainable_parameters', 'frozen_layers', 'feature_dim', 'frozen_ratio'
        ]
        
        for key in required_keys:
            assert key in info
        
        assert info['model_name'] == 'SpatialStream'
        assert info['backbone'] == 'MobileNetV2'
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert 0 <= info['frozen_ratio'] <= 1
    
    def test_numerical_stability(self, spatial_stream, device):
        """Test numerical stability with extreme inputs."""
        spatial_stream.eval()
        
        # Test with very small values
        small_input = torch.full((2, 3, 224, 224), 1e-6, device=device)
        output_small = spatial_stream(small_input)
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()
        
        # Test with large values (within reasonable range)
        large_input = torch.full((2, 3, 224, 224), 10.0, device=device)
        output_large = spatial_stream(large_input)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()
    
    def test_batch_consistency(self, spatial_stream, device):
        """Test that different batch sizes produce consistent results."""
        spatial_stream.eval()
        
        # Single sample
        single_input = torch.randn(1, 3, 224, 224, device=device)
        single_output = spatial_stream(single_input)
        
        # Batch of same sample
        batch_input = single_input.repeat(4, 1, 1, 1)
        batch_output = spatial_stream(batch_input)
        
        # Check that all outputs in batch are identical
        for i in range(4):
            torch.testing.assert_close(
                single_output[0], 
                batch_output[i], 
                rtol=1e-5, 
                atol=1e-6
            )
    
    def test_deterministic_output(self, device):
        """Test that model produces deterministic outputs with same input."""
        # Set seeds for reproducibility
        torch.manual_seed(42)
        model1 = SpatialStream(pretrained=False, freeze_layers=0).to(device)
        
        torch.manual_seed(42)
        model2 = SpatialStream(pretrained=False, freeze_layers=0).to(device)
        
        # Both models should have identical weights
        input_tensor = torch.randn(2, 3, 224, 224, device=device)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(input_tensor)
            output2 = model2(input_tensor)
        
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-6)
    
    def test_memory_efficiency(self, spatial_stream, device):
        """Test memory usage during forward pass."""
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Process a batch
            input_tensor = torch.randn(8, 3, 224, 224, device=device)
            output = spatial_stream(input_tensor)
            
            peak_memory = torch.cuda.memory_allocated()
            
            # Clear intermediate tensors
            del input_tensor, output
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            
            # Memory should be released after clearing tensors
            assert final_memory <= initial_memory + 1024 * 1024  # Allow 1MB tolerance
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_different_batch_sizes(self, spatial_stream, device, batch_size):
        """Test model with different batch sizes."""
        input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
        output = spatial_stream(input_tensor)
        
        expected_shape = (batch_size, spatial_stream.feature_dim)
        assert output.shape == expected_shape
    
    @pytest.mark.parametrize("feature_dim", [256, 512, 1024, 1280])
    def test_different_feature_dimensions(self, device, feature_dim):
        """Test model with different feature dimensions."""
        model = SpatialStream(
            pretrained=False,
            feature_dim=feature_dim,
            freeze_layers=0
        ).to(device)
        
        input_tensor = torch.randn(2, 3, 224, 224, device=device)
        output = model(input_tensor)
        
        assert output.shape == (2, feature_dim)


class TestSpatialStreamIntegration:
    """Integration tests for SpatialStream with other components."""
    
    def test_with_optimizer(self):
        """Test SpatialStream with PyTorch optimizer."""
        model = SpatialStream(pretrained=False, freeze_layers=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        input_tensor = torch.randn(4, 3, 224, 224)
        target = torch.randn(4, model.feature_dim)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        output = model(input_tensor)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        
        # Check that loss is computed and gradients are applied
        assert loss.item() > 0
    
    def test_with_dataloader(self):
        """Test SpatialStream with PyTorch DataLoader."""
        from torch.utils.data import DataLoader, TensorDataset
        
        model = SpatialStream(pretrained=False, freeze_layers=0)
        
        # Create dummy dataset
        data = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, 2, (20,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        model.eval()
        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                output = model(batch_data)
                assert output.shape[0] == batch_data.shape[0]
                break  # Test only first batch


if __name__ == "__main__":
    pytest.main([__file__])