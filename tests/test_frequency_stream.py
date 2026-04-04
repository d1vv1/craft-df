"""
Unit tests for FrequencyStream and DWTLayer functionality.

This module provides comprehensive testing for the frequency stream architecture,
including DWT layer functionality, gradient computation, and integration tests.
The tests ensure numerical stability, proper tensor handling, and correct
gradient flow through the frequency processing pipeline.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import logging

from craft_df.models.frequency_stream import FrequencyStream, DWTLayer

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDWTLayer:
    """Test cases for DWTLayer functionality."""
    
    @pytest.fixture
    def sample_dwt_coefficients(self) -> Dict[str, torch.Tensor]:
        """Create sample DWT coefficients for testing."""
        batch_size = 2
        channels = 3
        
        # Create coefficients with realistic sizes for 3-level decomposition
        # Starting from 224x224 image
        coefficients = {
            'll': torch.randn(batch_size, channels, 28, 28),  # 224 / (2^3) = 28
            'lh_1': torch.randn(batch_size, channels, 112, 112),  # 224 / 2 = 112
            'hl_1': torch.randn(batch_size, channels, 112, 112),
            'hh_1': torch.randn(batch_size, channels, 112, 112),
            'lh_2': torch.randn(batch_size, channels, 56, 56),   # 224 / 4 = 56
            'hl_2': torch.randn(batch_size, channels, 56, 56),
            'hh_2': torch.randn(batch_size, channels, 56, 56),
            'lh_3': torch.randn(batch_size, channels, 28, 28),   # 224 / 8 = 28
            'hl_3': torch.randn(batch_size, channels, 28, 28),
            'hh_3': torch.randn(batch_size, channels, 28, 28),
        }
        return coefficients
    
    def test_dwt_layer_initialization(self):
        """Test DWTLayer initialization with various parameters."""
        # Test default initialization
        layer = DWTLayer()
        assert layer.input_channels == 3
        assert layer.dwt_levels == 3
        assert layer.feature_dim == 256
        assert layer.dropout_rate == 0.1
        
        # Test custom initialization
        layer = DWTLayer(input_channels=1, dwt_levels=2, feature_dim=512, dropout_rate=0.2)
        assert layer.input_channels == 1
        assert layer.dwt_levels == 2
        assert layer.feature_dim == 512
        assert layer.dropout_rate == 0.2
    
    def test_dwt_layer_parameter_validation(self):
        """Test parameter validation in DWTLayer initialization."""
        # Test invalid input_channels
        with pytest.raises(AssertionError):
            DWTLayer(input_channels=0)
        
        # Test invalid dwt_levels
        with pytest.raises(AssertionError):
            DWTLayer(dwt_levels=0)
        with pytest.raises(AssertionError):
            DWTLayer(dwt_levels=7)
        
        # Test invalid feature_dim
        with pytest.raises(AssertionError):
            DWTLayer(feature_dim=0)
        
        # Test invalid dropout_rate
        with pytest.raises(AssertionError):
            DWTLayer(dropout_rate=-0.1)
        with pytest.raises(AssertionError):
            DWTLayer(dropout_rate=1.1)
    
    def test_dwt_layer_forward_pass(self, sample_dwt_coefficients):
        """Test forward pass through DWTLayer."""
        layer = DWTLayer(input_channels=3, dwt_levels=3, feature_dim=256)
        
        # Test forward pass
        output = layer(sample_dwt_coefficients)
        
        # Validate output shape
        batch_size = sample_dwt_coefficients['ll'].shape[0]
        assert output.shape == (batch_size, 256)
        
        # Validate output properties
        assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
        assert not torch.allclose(output, torch.zeros_like(output)), "Output is all zeros"
        
        # Test L2 normalization
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), "Features are not L2 normalized"
    
    def test_dwt_layer_input_validation(self, sample_dwt_coefficients):
        """Test input validation in DWTLayer forward pass."""
        layer = DWTLayer(input_channels=3, dwt_levels=3)
        
        # Test missing approximation coefficients
        invalid_coeffs = sample_dwt_coefficients.copy()
        del invalid_coeffs['ll']
        with pytest.raises((AssertionError, ValueError, RuntimeError)):
            layer(invalid_coeffs)
        
        # Test missing detail coefficients
        invalid_coeffs = sample_dwt_coefficients.copy()
        del invalid_coeffs['lh_1']
        with pytest.raises((AssertionError, ValueError, RuntimeError)):
            layer(invalid_coeffs)
        
        # Test wrong number of channels
        invalid_coeffs = sample_dwt_coefficients.copy()
        invalid_coeffs['ll'] = torch.randn(2, 1, 28, 28)  # Wrong channels
        with pytest.raises((AssertionError, ValueError, RuntimeError)):
            layer(invalid_coeffs)
        
        # Test wrong tensor dimensions
        invalid_coeffs = sample_dwt_coefficients.copy()
        invalid_coeffs['ll'] = torch.randn(2, 3, 28)  # 3D instead of 4D
        with pytest.raises((AssertionError, ValueError, RuntimeError)):
            layer(invalid_coeffs)
    
    def test_dwt_layer_gradient_computation(self, sample_dwt_coefficients):
        """Test gradient computation through DWTLayer."""
        layer = DWTLayer(input_channels=3, dwt_levels=3, feature_dim=256)
        
        # Enable gradient computation
        for coeff in sample_dwt_coefficients.values():
            coeff.requires_grad_(True)
        
        # Forward pass
        output = layer(sample_dwt_coefficients)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, coeff in sample_dwt_coefficients.items():
            assert coeff.grad is not None, f"No gradient computed for {name}"
            assert torch.all(torch.isfinite(coeff.grad)), f"Non-finite gradients in {name}"
            assert not torch.allclose(coeff.grad, torch.zeros_like(coeff.grad)), f"Zero gradients in {name}"
        
        # Check model parameter gradients
        for name, param in layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient computed for parameter {name}"
                assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradients in parameter {name}"
    
    def test_dwt_layer_different_levels(self):
        """Test DWTLayer with different numbers of decomposition levels."""
        batch_size = 2
        channels = 3
        
        for levels in [1, 2, 3, 4]:
            # Create appropriate coefficients for this level
            coefficients = {'ll': torch.randn(batch_size, channels, 28, 28)}
            
            for level in range(1, levels + 1):
                size = 224 // (2 ** level)
                coefficients[f'lh_{level}'] = torch.randn(batch_size, channels, size, size)
                coefficients[f'hl_{level}'] = torch.randn(batch_size, channels, size, size)
                coefficients[f'hh_{level}'] = torch.randn(batch_size, channels, size, size)
            
            layer = DWTLayer(input_channels=channels, dwt_levels=levels, feature_dim=128)
            output = layer(coefficients)
            
            assert output.shape == (batch_size, 128)
            assert torch.all(torch.isfinite(output))


class TestFrequencyStream:
    """Test cases for FrequencyStream functionality."""
    
    @pytest.fixture
    def sample_dwt_coefficients(self) -> Dict[str, torch.Tensor]:
        """Create sample DWT coefficients for testing."""
        batch_size = 2
        channels = 3
        
        coefficients = {
            'll': torch.randn(batch_size, channels, 28, 28),
            'lh_1': torch.randn(batch_size, channels, 112, 112),
            'hl_1': torch.randn(batch_size, channels, 112, 112),
            'hh_1': torch.randn(batch_size, channels, 112, 112),
            'lh_2': torch.randn(batch_size, channels, 56, 56),
            'hl_2': torch.randn(batch_size, channels, 56, 56),
            'hh_2': torch.randn(batch_size, channels, 56, 56),
            'lh_3': torch.randn(batch_size, channels, 28, 28),
            'hl_3': torch.randn(batch_size, channels, 28, 28),
            'hh_3': torch.randn(batch_size, channels, 28, 28),
        }
        return coefficients
    
    def test_frequency_stream_initialization(self):
        """Test FrequencyStream initialization with various parameters."""
        # Test default initialization
        stream = FrequencyStream()
        assert stream.input_channels == 3
        assert stream.dwt_levels == 3
        assert stream.feature_dim == 512
        assert stream.use_attention == True
        
        # Test custom initialization
        stream = FrequencyStream(
            input_channels=1,
            dwt_levels=2,
            feature_dim=256,
            hidden_dim=512,
            dropout_rate=0.2,
            use_attention=False
        )
        assert stream.input_channels == 1
        assert stream.dwt_levels == 2
        assert stream.feature_dim == 256
        assert stream.hidden_dim == 512
        assert stream.use_attention == False
    
    def test_frequency_stream_parameter_validation(self):
        """Test parameter validation in FrequencyStream initialization."""
        # Test invalid parameters
        with pytest.raises(AssertionError):
            FrequencyStream(input_channels=0)
        
        with pytest.raises(AssertionError):
            FrequencyStream(dwt_levels=0)
        
        with pytest.raises(AssertionError):
            FrequencyStream(feature_dim=0)
        
        with pytest.raises(AssertionError):
            FrequencyStream(hidden_dim=0)
        
        with pytest.raises(AssertionError):
            FrequencyStream(dropout_rate=-0.1)
    
    def test_frequency_stream_forward_pass(self, sample_dwt_coefficients):
        """Test forward pass through FrequencyStream."""
        stream = FrequencyStream(input_channels=3, dwt_levels=3, feature_dim=512)
        
        # Test forward pass
        output = stream(sample_dwt_coefficients)
        
        # Validate output shape
        batch_size = sample_dwt_coefficients['ll'].shape[0]
        assert output.shape == (batch_size, 512)
        
        # Validate output properties
        assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
        assert not torch.allclose(output, torch.zeros_like(output)), "Output is all zeros"
        
        # Test L2 normalization
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), "Features are not L2 normalized"
    
    def test_frequency_stream_with_attention(self, sample_dwt_coefficients):
        """Test FrequencyStream with attention mechanism."""
        stream = FrequencyStream(use_attention=True, feature_dim=256)
        
        # Test forward pass
        output = stream(sample_dwt_coefficients)
        batch_size = sample_dwt_coefficients['ll'].shape[0]
        assert output.shape == (batch_size, 256)
        
        # Test attention weight extraction
        attention_weights = stream.get_attention_weights(sample_dwt_coefficients)
        assert attention_weights is not None
        assert attention_weights.shape[0] == batch_size
    
    def test_frequency_stream_without_attention(self, sample_dwt_coefficients):
        """Test FrequencyStream without attention mechanism."""
        stream = FrequencyStream(use_attention=False, feature_dim=256)
        
        # Test forward pass
        output = stream(sample_dwt_coefficients)
        batch_size = sample_dwt_coefficients['ll'].shape[0]
        assert output.shape == (batch_size, 256)
        
        # Test that attention weights are None
        attention_weights = stream.get_attention_weights(sample_dwt_coefficients)
        assert attention_weights is None
    
    def test_frequency_stream_gradient_computation(self, sample_dwt_coefficients):
        """Test gradient computation through FrequencyStream."""
        stream = FrequencyStream(input_channels=3, dwt_levels=3, feature_dim=256)
        
        # Enable gradient computation
        for coeff in sample_dwt_coefficients.values():
            coeff.requires_grad_(True)
        
        # Forward pass
        output = stream(sample_dwt_coefficients)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed for inputs
        for name, coeff in sample_dwt_coefficients.items():
            assert coeff.grad is not None, f"No gradient computed for {name}"
            assert torch.all(torch.isfinite(coeff.grad)), f"Non-finite gradients in {name}"
        
        # Check model parameter gradients
        for name, param in stream.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient computed for parameter {name}"
                assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradients in parameter {name}"
    
    def test_frequency_stream_feature_maps(self, sample_dwt_coefficients):
        """Test feature map extraction from FrequencyStream."""
        stream = FrequencyStream(use_attention=True, feature_dim=256)
        
        # Extract feature maps
        feature_maps = stream.get_feature_maps(sample_dwt_coefficients)
        
        # Validate feature maps
        assert 'dwt_features' in feature_maps
        assert 'refined_features' in feature_maps
        assert 'attended_features' in feature_maps
        
        batch_size = sample_dwt_coefficients['ll'].shape[0]
        assert feature_maps['refined_features'].shape == (batch_size, 256)
        assert feature_maps['attended_features'].shape == (batch_size, 256)
    
    def test_frequency_stream_model_info(self):
        """Test model information extraction."""
        stream = FrequencyStream(feature_dim=256, hidden_dim=512)
        
        info = stream.get_model_info()
        
        assert info['model_name'] == 'FrequencyStream'
        assert info['feature_dim'] == 256
        assert info['hidden_dim'] == 512
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert info['trainable_parameters'] <= info['total_parameters']
    
    def test_frequency_stream_memory_usage(self):
        """Test memory usage estimation."""
        stream = FrequencyStream(feature_dim=256)
        
        memory_info = stream.get_memory_usage(batch_size=4, input_size=(224, 224))
        
        assert 'parameter_memory_mb' in memory_info
        assert 'll_memory_mb' in memory_info
        assert 'detail_memory_mb' in memory_info
        assert 'feature_memory_mb' in memory_info
        assert 'total_estimated_mb' in memory_info
        assert memory_info['batch_size'] == 4
        assert memory_info['input_size'] == (224, 224)
        
        # All memory values should be positive
        for key, value in memory_info.items():
            if key.endswith('_mb'):
                assert value >= 0, f"Negative memory value for {key}: {value}"
    
    def test_frequency_stream_optimization(self, sample_dwt_coefficients):
        """Test model optimization for inference."""
        stream = FrequencyStream(feature_dim=256)
        
        # Test optimization
        stream.optimize_for_inference()
        
        # Model should be in eval mode
        assert not stream.training
        
        # Test that optimized model still works
        with torch.no_grad():
            output = stream(sample_dwt_coefficients)
            batch_size = sample_dwt_coefficients['ll'].shape[0]
            assert output.shape == (batch_size, 256)


class TestFrequencyStreamIntegration:
    """Integration tests for FrequencyStream with other components."""
    
    @pytest.fixture
    def sample_dwt_coefficients(self) -> Dict[str, torch.Tensor]:
        """Create sample DWT coefficients for integration testing."""
        batch_size = 2
        channels = 3
        
        coefficients = {
            'll': torch.randn(batch_size, channels, 28, 28),
            'lh_1': torch.randn(batch_size, channels, 112, 112),
            'hl_1': torch.randn(batch_size, channels, 112, 112),
            'hh_1': torch.randn(batch_size, channels, 112, 112),
            'lh_2': torch.randn(batch_size, channels, 56, 56),
            'hl_2': torch.randn(batch_size, channels, 56, 56),
            'hh_2': torch.randn(batch_size, channels, 56, 56),
            'lh_3': torch.randn(batch_size, channels, 28, 28),
            'hl_3': torch.randn(batch_size, channels, 28, 28),
            'hh_3': torch.randn(batch_size, channels, 28, 28),
        }
        return coefficients
    
    def test_frequency_stream_with_spatial_compatibility(self, sample_dwt_coefficients):
        """Test that FrequencyStream output is compatible with spatial stream features."""
        from craft_df.models.spatial_stream import SpatialStream
        
        # Create streams with compatible feature dimensions
        frequency_stream = FrequencyStream(feature_dim=512)
        spatial_stream = SpatialStream(feature_dim=512)
        
        # Test frequency stream
        freq_output = frequency_stream(sample_dwt_coefficients)
        
        # Test spatial stream with dummy input
        batch_size = sample_dwt_coefficients['ll'].shape[0]
        spatial_input = torch.randn(batch_size, 3, 224, 224)
        spatial_output = spatial_stream(spatial_input)
        
        # Outputs should have compatible shapes for fusion
        assert freq_output.shape == spatial_output.shape
        assert freq_output.shape == (batch_size, 512)
    
    def test_frequency_stream_numerical_stability(self, sample_dwt_coefficients):
        """Test numerical stability with extreme input values."""
        stream = FrequencyStream(feature_dim=256)
        
        # Test with very small values
        small_coeffs = {}
        for key, tensor in sample_dwt_coefficients.items():
            small_coeffs[key] = tensor * 1e-6
        
        output_small = stream(small_coeffs)
        assert torch.all(torch.isfinite(output_small)), "Numerical instability with small values"
        
        # Test with very large values
        large_coeffs = {}
        for key, tensor in sample_dwt_coefficients.items():
            large_coeffs[key] = tensor * 1e6
        
        output_large = stream(large_coeffs)
        assert torch.all(torch.isfinite(output_large)), "Numerical instability with large values"
        
        # Test with mixed positive/negative values
        mixed_coeffs = {}
        for key, tensor in sample_dwt_coefficients.items():
            mixed_coeffs[key] = tensor * torch.randn_like(tensor).sign()
        
        output_mixed = stream(mixed_coeffs)
        assert torch.all(torch.isfinite(output_mixed)), "Numerical instability with mixed values"
    
    def test_frequency_stream_batch_consistency(self, sample_dwt_coefficients):
        """Test that processing individual samples vs batches gives consistent results."""
        stream = FrequencyStream(feature_dim=256)
        stream.eval()  # Disable dropout for consistency
        
        # Process full batch
        with torch.no_grad():
            batch_output = stream(sample_dwt_coefficients)
        
        # Process individual samples
        individual_outputs = []
        for i in range(sample_dwt_coefficients['ll'].shape[0]):
            individual_coeffs = {}
            for key, tensor in sample_dwt_coefficients.items():
                individual_coeffs[key] = tensor[i:i+1]  # Keep batch dimension
            
            with torch.no_grad():
                individual_output = stream(individual_coeffs)
                individual_outputs.append(individual_output)
        
        # Combine individual outputs
        combined_output = torch.cat(individual_outputs, dim=0)
        
        # Should be approximately equal (allowing for small numerical differences)
        assert torch.allclose(batch_output, combined_output, atol=1e-5), \
            "Batch processing inconsistent with individual processing"
    
    def test_frequency_stream_device_compatibility(self, sample_dwt_coefficients):
        """Test FrequencyStream compatibility with different devices."""
        stream = FrequencyStream(feature_dim=256)
        
        # Test CPU
        cpu_output = stream(sample_dwt_coefficients)
        assert cpu_output.device.type == 'cpu'
        
        # Test GPU if available
        if torch.cuda.is_available():
            # Move model and data to GPU
            stream_gpu = stream.cuda()
            gpu_coeffs = {}
            for key, tensor in sample_dwt_coefficients.items():
                gpu_coeffs[key] = tensor.cuda()
            
            gpu_output = stream_gpu(gpu_coeffs)
            assert gpu_output.device.type == 'cuda'
            
            # Results should be approximately equal
            assert torch.allclose(cpu_output, gpu_output.cpu(), atol=1e-4), \
                "GPU and CPU outputs differ significantly"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])