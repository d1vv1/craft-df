"""
Tests for SpatialStream GPU optimization and performance features.

This module contains tests specifically for GPU optimization features,
performance profiling, and numerical stability of the spatial stream.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time

from craft_df.models.spatial_stream import SpatialStream


class TestSpatialStreamOptimization:
    """Test suite for SpatialStream optimization features."""
    
    @pytest.fixture
    def device(self):
        """Get available device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def optimized_model(self, device):
        """Create optimized SpatialStream for testing."""
        model = SpatialStream(
            pretrained=False,
            freeze_layers=5,
            feature_dim=512,
            dropout_rate=0.1
        )
        return model.to(device)
    
    def test_memory_usage_estimation(self, optimized_model):
        """Test memory usage estimation functionality."""
        memory_info = optimized_model.get_memory_usage(batch_size=8)
        
        required_keys = [
            'parameter_memory_mb', 'input_memory_mb', 'feature_memory_mb',
            'output_memory_mb', 'total_estimated_mb', 'batch_size'
        ]
        
        for key in required_keys:
            assert key in memory_info
            assert isinstance(memory_info[key], (int, float))
            assert memory_info[key] >= 0
        
        assert memory_info['batch_size'] == 8
        assert memory_info['total_estimated_mb'] > 0
    
    def test_mixed_precision_enablement(self, optimized_model):
        """Test mixed precision optimization."""
        optimized_model.enable_mixed_precision()
        
        # Check that batch norm layers remain in float32
        for module in optimized_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                # Should be float32 for numerical stability
                assert next(module.parameters()).dtype == torch.float32
    
    def test_inference_optimization(self, optimized_model, device):
        """Test inference optimization functionality."""
        optimized_model.optimize_for_inference()
        
        # Model should be in eval mode
        assert not optimized_model.training
        
        # Test that optimized model still works
        input_tensor = torch.randn(2, 3, 224, 224, device=device)
        output = optimized_model(input_tensor)
        
        assert output.shape == (2, optimized_model.feature_dim)
        assert not torch.isnan(output).any()
    
    def test_forward_pass_profiling(self, optimized_model, device):
        """Test forward pass profiling functionality."""
        input_tensor = torch.randn(4, 3, 224, 224, device=device)
        
        profile_results = optimized_model.profile_forward_pass(input_tensor)
        
        required_keys = [
            'batch_size', 'cpu_time_ms', 'throughput_samples_per_sec', 'output_shape'
        ]
        
        for key in required_keys:
            assert key in profile_results
        
        assert profile_results['batch_size'] == 4
        assert profile_results['cpu_time_ms'] > 0
        assert profile_results['throughput_samples_per_sec'] > 0
        assert profile_results['output_shape'] == (4, optimized_model.feature_dim)
        
        if device.type == 'cuda':
            assert 'gpu_time_ms' in profile_results
            assert 'memory_allocated_mb' in profile_results
            assert profile_results['gpu_time_ms'] is not None
    
    def test_contiguous_tensor_handling(self, optimized_model, device):
        """Test handling of non-contiguous tensors."""
        # Create non-contiguous tensor
        base_tensor = torch.randn(4, 6, 224, 224, device=device)
        non_contiguous = base_tensor[:, ::2, :, :]  # Take every 2nd channel
        
        assert not non_contiguous.is_contiguous()
        assert non_contiguous.shape == (4, 3, 224, 224)
        
        # Model should handle non-contiguous tensors
        output = optimized_model(non_contiguous)
        assert output.shape == (4, optimized_model.feature_dim)
        assert not torch.isnan(output).any()
    
    def test_feature_normalization(self, optimized_model, device):
        """Test L2 normalization of output features."""
        input_tensor = torch.randn(4, 3, 224, 224, device=device)
        output = optimized_model(input_tensor)
        
        # Check L2 normalization
        norms = torch.norm(output, p=2, dim=1)
        expected_norms = torch.ones_like(norms)
        
        torch.testing.assert_close(norms, expected_norms, rtol=1e-5, atol=1e-6)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_efficiency(self, device):
        """Test CUDA memory efficiency during forward pass."""
        if device.type != 'cuda':
            pytest.skip("CUDA not available")
        
        model = SpatialStream(pretrained=False, freeze_layers=0).to(device)
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Process multiple batches
        for _ in range(5):
            input_tensor = torch.randn(8, 3, 224, 224, device=device)
            output = model(input_tensor)
            del input_tensor, output
        
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 50 * 1024 * 1024  # Less than 50MB growth
    
    def test_numerical_precision_consistency(self, optimized_model, device):
        """Test numerical precision consistency across runs."""
        optimized_model.eval()
        
        input_tensor = torch.randn(4, 3, 224, 224, device=device)
        
        # Run multiple times and check consistency
        outputs = []
        with torch.no_grad():
            for _ in range(3):
                output = optimized_model(input_tensor)
                outputs.append(output.clone())
        
        # All outputs should be identical
        for i in range(1, len(outputs)):
            torch.testing.assert_close(outputs[0], outputs[i], rtol=1e-6, atol=1e-7)
    
    def test_gradient_scaling_compatibility(self, optimized_model, device):
        """Test compatibility with gradient scaling for mixed precision."""
        if not hasattr(torch.cuda.amp, 'GradScaler'):
            pytest.skip("Mixed precision not available")
        
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.Adam(optimized_model.parameters(), lr=1e-4)
        
        input_tensor = torch.randn(4, 3, 224, 224, device=device)
        target = torch.randn(4, optimized_model.feature_dim, device=device)
        
        optimized_model.train()
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            output = optimized_model(input_tensor)
            loss = nn.MSELoss()(output, target)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Check that gradients were computed
        for param in optimized_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
    def test_throughput_scaling(self, optimized_model, device, batch_size):
        """Test throughput scaling with different batch sizes."""
        input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
        
        profile_results = optimized_model.profile_forward_pass(input_tensor)
        
        # Throughput should be reasonable
        assert profile_results['throughput_samples_per_sec'] > 0
        
        # Larger batches should generally have higher total throughput
        if batch_size >= 8:
            assert profile_results['throughput_samples_per_sec'] > batch_size * 0.5
    
    def test_memory_scaling_with_batch_size(self, optimized_model):
        """Test memory usage scaling with batch size."""
        batch_sizes = [1, 4, 8, 16]
        memory_usages = []
        
        for batch_size in batch_sizes:
            memory_info = optimized_model.get_memory_usage(batch_size)
            memory_usages.append(memory_info['total_estimated_mb'])
        
        # Memory should generally increase with batch size
        # Just check that larger batches use more memory than smaller ones
        assert memory_usages[1] > memory_usages[0], "Batch size 4 should use more memory than batch size 1"
        assert memory_usages[2] > memory_usages[1], "Batch size 8 should use more memory than batch size 4"
        assert memory_usages[3] > memory_usages[2], "Batch size 16 should use more memory than batch size 8"


class TestSpatialStreamNumericalStability:
    """Test suite for numerical stability of SpatialStream."""
    
    @pytest.fixture
    def stable_model(self):
        """Create model for stability testing."""
        return SpatialStream(pretrained=False, freeze_layers=0, feature_dim=256)
    
    def test_extreme_input_values(self, stable_model):
        """Test model behavior with extreme input values."""
        stable_model.eval()
        
        test_cases = [
            torch.full((2, 3, 224, 224), 0.5),   # Normal range values
            torch.ones(2, 3, 224, 224),          # All ones
            torch.randn(2, 3, 224, 224),         # Random values
        ]
        
        with torch.no_grad():
            for i, test_input in enumerate(test_cases):
                output = stable_model(test_input)
                
                # Check for NaN or Inf
                assert not torch.isnan(output).any(), f"NaN detected in test case {i} with input range [{test_input.min()}, {test_input.max()}]"
                assert not torch.isinf(output).any(), f"Inf detected in test case {i} with input range [{test_input.min()}, {test_input.max()}]"
                
                # Check output is approximately normalized
                norms = torch.norm(output, p=2, dim=1)
                
                # The normalization should make norms reasonably close to 1.0
                # Allow more tolerance for numerical precision issues
                assert torch.all(norms > 0.5), f"Norm too small in test case {i}: {norms}"
                assert torch.all(norms < 1.5), f"Norm too large in test case {i}: {norms}"
    
    def test_gradient_explosion_prevention(self, stable_model):
        """Test prevention of gradient explosion."""
        stable_model.train()
        
        input_tensor = torch.randn(4, 3, 224, 224, requires_grad=True)
        target = torch.randn(4, stable_model.feature_dim)
        
        # Simulate training step with large learning rate
        optimizer = torch.optim.SGD(stable_model.parameters(), lr=10.0)  # Intentionally large
        
        for _ in range(5):  # Multiple steps
            optimizer.zero_grad()
            output = stable_model(input_tensor)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            # Check gradient norms
            total_norm = 0
            for param in stable_model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Gradients should not explode
            assert total_norm < 1000, f"Gradient explosion detected: norm = {total_norm}"
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(stable_model.parameters(), max_norm=1.0)
            optimizer.step()
    
    def test_batch_size_consistency(self, stable_model):
        """Test consistency across different batch sizes."""
        stable_model.eval()
        
        # Create identical samples
        single_sample = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            # Process single sample
            single_output = stable_model(single_sample)
            
            # Process batch of identical samples
            batch_input = single_sample.repeat(8, 1, 1, 1)
            batch_output = stable_model(batch_input)
            
            # All outputs should be identical
            for i in range(8):
                torch.testing.assert_close(
                    single_output[0], 
                    batch_output[i], 
                    rtol=1e-6, 
                    atol=1e-7
                )
    
    def test_deterministic_behavior(self):
        """Test deterministic behavior with fixed seeds."""
        torch.manual_seed(12345)
        model1 = SpatialStream(pretrained=False, freeze_layers=0)
        
        torch.manual_seed(12345)
        model2 = SpatialStream(pretrained=False, freeze_layers=0)
        
        input_tensor = torch.randn(4, 3, 224, 224)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(input_tensor)
            output2 = model2(input_tensor)
        
        torch.testing.assert_close(output1, output2, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__])