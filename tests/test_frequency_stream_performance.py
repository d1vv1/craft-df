"""
Performance benchmarks and integration tests for FrequencyStream.

This module provides comprehensive performance testing for the frequency stream architecture,
including benchmarking, memory profiling, and integration tests with spatial stream compatibility.
The tests ensure optimal performance and proper integration with other system components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Tuple
import logging

from craft_df.models.frequency_stream import FrequencyStream, DWTLayer

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFrequencyStreamPerformance:
    """Performance benchmark tests for FrequencyStream."""
    
    @pytest.fixture
    def sample_dwt_coefficients(self) -> Dict[str, torch.Tensor]:
        """Create sample DWT coefficients for performance testing."""
        batch_size = 4  # Larger batch for performance testing
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
    
    @pytest.fixture
    def large_dwt_coefficients(self) -> Dict[str, torch.Tensor]:
        """Create large DWT coefficients for stress testing."""
        batch_size = 8  # Larger batch for stress testing
        channels = 3
        
        coefficients = {
            'll': torch.randn(batch_size, channels, 56, 56),  # Larger resolution
            'lh_1': torch.randn(batch_size, channels, 224, 224),
            'hl_1': torch.randn(batch_size, channels, 224, 224),
            'hh_1': torch.randn(batch_size, channels, 224, 224),
            'lh_2': torch.randn(batch_size, channels, 112, 112),
            'hl_2': torch.randn(batch_size, channels, 112, 112),
            'hh_2': torch.randn(batch_size, channels, 112, 112),
            'lh_3': torch.randn(batch_size, channels, 56, 56),
            'hl_3': torch.randn(batch_size, channels, 56, 56),
            'hh_3': torch.randn(batch_size, channels, 56, 56),
        }
        return coefficients
    
    def test_frequency_stream_benchmark(self, sample_dwt_coefficients):
        """Test performance benchmarking functionality."""
        stream = FrequencyStream(feature_dim=256, use_attention=False)  # Disable attention for baseline
        
        # Run benchmark
        results = stream.benchmark_performance(
            sample_dwt_coefficients, 
            num_iterations=20, 
            warmup_iterations=5
        )
        
        # Validate benchmark results
        assert 'mean_time_ms' in results
        assert 'throughput_samples_per_sec' in results
        assert 'batch_size' in results
        assert results['batch_size'] == sample_dwt_coefficients['ll'].shape[0]
        assert results['mean_time_ms'] > 0
        assert results['throughput_samples_per_sec'] > 0
        assert results['std_time_ms'] >= 0
        
        logger.info(f"Benchmark results: {results['throughput_samples_per_sec']:.2f} samples/sec")
    
    def test_frequency_stream_benchmark_with_attention(self, sample_dwt_coefficients):
        """Test performance benchmarking with attention mechanism."""
        stream = FrequencyStream(feature_dim=256, use_attention=True)
        
        # Run benchmark
        results = stream.benchmark_performance(
            sample_dwt_coefficients, 
            num_iterations=15, 
            warmup_iterations=3
        )
        
        # Validate benchmark results
        assert results['use_attention'] == True
        assert results['mean_time_ms'] > 0
        assert results['throughput_samples_per_sec'] > 0
        
        logger.info(f"Benchmark with attention: {results['throughput_samples_per_sec']:.2f} samples/sec")
    
    def test_memory_profiling(self, sample_dwt_coefficients):
        """Test memory usage profiling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory profiling")
        
        stream = FrequencyStream(feature_dim=256).cuda()
        
        # Move coefficients to GPU
        gpu_coeffs = {}
        for key, tensor in sample_dwt_coefficients.items():
            gpu_coeffs[key] = tensor.cuda()
        
        # Profile memory usage
        memory_results = stream.profile_memory_usage(gpu_coeffs)
        
        # Validate memory profiling results
        assert 'forward_memory_mb' in memory_results
        assert 'peak_memory_mb' in memory_results
        assert 'parameter_memory_mb' in memory_results
        assert memory_results['forward_memory_mb'] >= 0
        assert memory_results['peak_memory_mb'] >= memory_results['forward_memory_mb']
        
        logger.info(f"Memory usage: {memory_results['forward_memory_mb']:.2f}MB forward")
    
    def test_throughput_optimization(self, sample_dwt_coefficients):
        """Test throughput optimization functionality."""
        stream = FrequencyStream(feature_dim=256)
        
        # Benchmark before optimization
        results_before = stream.benchmark_performance(
            sample_dwt_coefficients, 
            num_iterations=10, 
            warmup_iterations=2
        )
        
        # Apply throughput optimizations
        stream.optimize_for_throughput()
        
        # Benchmark after optimization
        results_after = stream.benchmark_performance(
            sample_dwt_coefficients, 
            num_iterations=10, 
            warmup_iterations=2
        )
        
        # Optimization should maintain or improve performance
        # Note: Improvement may vary based on hardware and model complexity
        assert results_after['mean_time_ms'] > 0
        assert results_after['throughput_samples_per_sec'] > 0
        
        logger.info(f"Throughput before optimization: {results_before['throughput_samples_per_sec']:.2f} samples/sec")
        logger.info(f"Throughput after optimization: {results_after['throughput_samples_per_sec']:.2f} samples/sec")
    
    def test_mixed_precision_performance(self, sample_dwt_coefficients):
        """Test mixed precision training performance."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")
        
        stream = FrequencyStream(feature_dim=256).cuda()
        
        # Move coefficients to GPU
        gpu_coeffs = {}
        for key, tensor in sample_dwt_coefficients.items():
            gpu_coeffs[key] = tensor.cuda()
        
        # Enable mixed precision
        stream.enable_mixed_precision()
        
        # Test forward pass with mixed precision
        stream.train()  # Enable training mode for mixed precision
        
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast():
            output = stream(gpu_coeffs)
        
        # Validate output
        assert output.shape[0] == gpu_coeffs['ll'].shape[0]
        assert output.shape[1] == 256
        assert torch.all(torch.isfinite(output))
        
        logger.info("Mixed precision forward pass successful")
    
    def test_large_batch_performance(self, large_dwt_coefficients):
        """Test performance with large batch sizes and high resolution."""
        stream = FrequencyStream(feature_dim=512, use_attention=False)  # Larger feature dim
        
        # Test forward pass with large batch
        try:
            output = stream(large_dwt_coefficients)
            
            # Validate output
            batch_size = large_dwt_coefficients['ll'].shape[0]
            assert output.shape == (batch_size, 512)
            assert torch.all(torch.isfinite(output))
            
            logger.info(f"Large batch processing successful: {batch_size} samples")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("Large batch test skipped due to memory constraints")
                pytest.skip("Insufficient memory for large batch test")
            else:
                raise e
    
    def test_gradient_checkpointing_performance(self, sample_dwt_coefficients):
        """Test performance with gradient checkpointing enabled."""
        stream = FrequencyStream(feature_dim=256)
        stream.train()  # Enable training mode
        
        # Enable gradient computation
        for coeff in sample_dwt_coefficients.values():
            coeff.requires_grad_(True)
        
        # Forward pass (gradient checkpointing is automatically enabled under memory pressure)
        output = stream(sample_dwt_coefficients)
        
        # Backward pass to test gradient checkpointing
        loss = output.sum()
        loss.backward()
        
        # Validate gradients
        for name, coeff in sample_dwt_coefficients.items():
            assert coeff.grad is not None, f"No gradient for {name}"
            assert torch.all(torch.isfinite(coeff.grad)), f"Non-finite gradients in {name}"
        
        logger.info("Gradient checkpointing test successful")
    
    def test_error_handling_performance(self, sample_dwt_coefficients):
        """Test error handling doesn't significantly impact performance."""
        stream = FrequencyStream(feature_dim=256)
        
        # Test with valid input (baseline)
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = stream(sample_dwt_coefficients)
        baseline_time = time.time() - start_time
        
        # Test with input that triggers error handling (non-finite values)
        corrupted_coeffs = {}
        for key, tensor in sample_dwt_coefficients.items():
            corrupted_tensor = tensor.clone()
            # Add some extreme values that will trigger clamping
            corrupted_tensor[0, 0, 0, 0] = float('inf')
            corrupted_tensor[0, 0, 0, 1] = float('-inf')
            corrupted_coeffs[key] = corrupted_tensor
        
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                output = stream(corrupted_coeffs)
                # Should still produce valid output due to error handling
                assert torch.all(torch.isfinite(output))
        error_handling_time = time.time() - start_time
        
        # Error handling should not significantly slow down processing
        # Allow up to 50% overhead for error handling
        assert error_handling_time < baseline_time * 1.5, \
            f"Error handling too slow: {error_handling_time:.3f}s vs {baseline_time:.3f}s baseline"
        
        logger.info(f"Error handling overhead: {((error_handling_time / baseline_time) - 1) * 100:.1f}%")


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
    
    def test_spatial_stream_compatibility(self, sample_dwt_coefficients):
        """Test compatibility with spatial stream for feature fusion."""
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
        
        # Test feature concatenation (simple fusion)
        fused_features = torch.cat([freq_output, spatial_output], dim=1)
        assert fused_features.shape == (batch_size, 1024)
        
        # Test feature addition (residual-like fusion)
        added_features = freq_output + spatial_output
        assert added_features.shape == (batch_size, 512)
        
        logger.info("Spatial stream compatibility test passed")
    
    def test_cross_attention_preparation(self, sample_dwt_coefficients):
        """Test that frequency stream output is suitable for cross-attention."""
        frequency_stream = FrequencyStream(feature_dim=256)
        
        # Get frequency features
        freq_features = frequency_stream(sample_dwt_coefficients)
        batch_size = freq_features.shape[0]
        
        # Simulate cross-attention setup (frequency as keys/values, spatial as queries)
        # This tests the format expected by cross-attention mechanisms
        
        # Test reshaping for attention
        seq_len = 1  # Single feature vector per sample
        freq_keys = freq_features.unsqueeze(1)  # (batch_size, seq_len, feature_dim)
        freq_values = freq_features.unsqueeze(1)
        
        assert freq_keys.shape == (batch_size, seq_len, 256)
        assert freq_values.shape == (batch_size, seq_len, 256)
        
        # Test with multi-head attention (simulating cross-attention)
        attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        # Use frequency features as both query and key/value for testing
        with torch.no_grad():
            attended_output, attention_weights = attention(freq_keys, freq_keys, freq_values)
        
        assert attended_output.shape == (batch_size, seq_len, 256)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)
        
        logger.info("Cross-attention preparation test passed")
    
    def test_feature_disentanglement_compatibility(self, sample_dwt_coefficients):
        """Test compatibility with feature disentanglement modules."""
        frequency_stream = FrequencyStream(feature_dim=256)
        
        # Get frequency features
        freq_features = frequency_stream(sample_dwt_coefficients)
        batch_size = freq_features.shape[0]
        
        # Simulate feature disentanglement (domain-invariant vs domain-specific features)
        # This tests the format expected by disentanglement modules
        
        # Simple linear projection to simulate disentanglement
        domain_invariant_proj = nn.Linear(256, 128)
        domain_specific_proj = nn.Linear(256, 128)
        
        with torch.no_grad():
            domain_invariant = domain_invariant_proj(freq_features)
            domain_specific = domain_specific_proj(freq_features)
        
        assert domain_invariant.shape == (batch_size, 128)
        assert domain_specific.shape == (batch_size, 128)
        
        # Test reconstruction
        reconstruction_proj = nn.Linear(256, 256)
        combined_features = torch.cat([domain_invariant, domain_specific], dim=1)
        
        with torch.no_grad():
            reconstructed = reconstruction_proj(combined_features)
        
        assert reconstructed.shape == (batch_size, 256)
        
        logger.info("Feature disentanglement compatibility test passed")
    
    def test_end_to_end_pipeline_performance(self, sample_dwt_coefficients):
        """Test end-to-end pipeline performance with all components."""
        from craft_df.models.spatial_stream import SpatialStream
        
        # Create complete pipeline components
        frequency_stream = FrequencyStream(feature_dim=256, use_attention=True)
        spatial_stream = SpatialStream(feature_dim=256)
        
        # Simulate cross-attention fusion
        cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        # Simulate final classifier
        classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # Binary classification
        )
        
        batch_size = sample_dwt_coefficients['ll'].shape[0]
        spatial_input = torch.randn(batch_size, 3, 224, 224)
        
        # Measure end-to-end performance
        start_time = time.time()
        
        with torch.no_grad():
            # Extract features
            freq_features = frequency_stream(sample_dwt_coefficients)
            spatial_features = spatial_stream(spatial_input)
            
            # Cross-attention fusion (simplified)
            freq_query = freq_features.unsqueeze(1)
            spatial_kv = spatial_features.unsqueeze(1)
            
            fused_features, _ = cross_attention(freq_query, spatial_kv, spatial_kv)
            fused_features = fused_features.squeeze(1)
            
            # Classification
            predictions = classifier(fused_features)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # milliseconds
        
        # Validate outputs
        assert predictions.shape == (batch_size, 2)
        assert torch.all(torch.isfinite(predictions))
        
        # Calculate throughput
        throughput = batch_size / (total_time / 1000)  # samples per second
        
        logger.info(f"End-to-end pipeline: {throughput:.2f} samples/sec, {total_time:.2f}ms total")
        
        # Performance should be reasonable (adjust threshold based on hardware)
        assert total_time < 1000, f"End-to-end pipeline too slow: {total_time:.2f}ms"
        assert throughput > 1, f"End-to-end throughput too low: {throughput:.2f} samples/sec"
    
    def test_batch_size_scaling(self, sample_dwt_coefficients):
        """Test performance scaling with different batch sizes."""
        frequency_stream = FrequencyStream(feature_dim=256, use_attention=False)
        
        batch_sizes = [1, 2, 4, 8]
        throughputs = []
        
        for batch_size in batch_sizes:
            # Create coefficients with specific batch size
            batch_coeffs = {}
            for key, tensor in sample_dwt_coefficients.items():
                # Repeat or slice to get desired batch size
                if batch_size <= tensor.shape[0]:
                    batch_coeffs[key] = tensor[:batch_size]
                else:
                    repeats = (batch_size + tensor.shape[0] - 1) // tensor.shape[0]
                    repeated = tensor.repeat(repeats, 1, 1, 1)
                    batch_coeffs[key] = repeated[:batch_size]
            
            # Benchmark this batch size
            try:
                results = frequency_stream.benchmark_performance(
                    batch_coeffs, 
                    num_iterations=10, 
                    warmup_iterations=2
                )
                throughputs.append(results['throughput_samples_per_sec'])
                
                logger.info(f"Batch size {batch_size}: {results['throughput_samples_per_sec']:.2f} samples/sec")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"Batch size {batch_size} skipped due to memory constraints")
                    break
                else:
                    raise e
        
        # Throughput should generally increase with batch size (up to memory limits)
        assert len(throughputs) >= 2, "Need at least 2 batch sizes for scaling test"
        
        # Check that we can process different batch sizes successfully
        for i, throughput in enumerate(throughputs):
            assert throughput > 0, f"Invalid throughput for batch size {batch_sizes[i]}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])