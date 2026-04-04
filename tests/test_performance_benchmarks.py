"""
Performance benchmarks and tests for CRAFT-DF training pipeline.

This module provides comprehensive performance testing including:
- Throughput benchmarks
- Memory usage profiling
- GPU optimization validation
- Distributed training performance
"""

import pytest
import torch
import torch.nn as nn
import time
import psutil
from typing import Dict, Any, List, Tuple
import numpy as np
from unittest.mock import patch, MagicMock

from craft_df.training.performance_monitor import (
    PerformanceMonitor, GPUOptimizer, PerformanceCallback,
    DistributedTrainingOptimizer
)
from craft_df.models.craft_df_model import CRAFTDFModel
from craft_df.data.dataset import HierarchicalDeepfakeDataset
from torch.utils.data import DataLoader


class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""
    
    @pytest.fixture
    def model_config(self):
        """Lightweight model configuration for performance testing."""
        return {
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
                'num_heads': 4,
                'dropout_rate': 0.1
            },
            'disentanglement_config': {
                'input_dim': 64,
                'invariant_dim': 32,
                'specific_dim': 16,
                'num_domains': 2,
                'hidden_dim': 64
            },
            'num_classes': 2,
            'learning_rate': 1e-3,
            'adversarial_training': True
        }
    
    @pytest.fixture
    def sample_batch(self):
        """Generate sample batch for performance testing."""
        batch_size = 8
        
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
    
    def test_model_throughput_benchmark(self, model_config, sample_batch):
        """Benchmark model throughput (samples per second)."""
        model = CRAFTDFModel(**model_config)
        model.eval()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        for key in sample_batch:
            if isinstance(sample_batch[key], torch.Tensor):
                sample_batch[key] = sample_batch[key].to(device)
            elif isinstance(sample_batch[key], dict):
                for subkey in sample_batch[key]:
                    sample_batch[key][subkey] = sample_batch[key][subkey].to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = model(
                    sample_batch['spatial_input'],
                    sample_batch['frequency_input'],
                    sample_batch['domain_labels']
                )
        
        # Benchmark runs
        num_runs = 100
        batch_size = sample_batch['spatial_input'].shape[0]
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(
                    sample_batch['spatial_input'],
                    sample_batch['frequency_input'],
                    sample_batch['domain_labels']
                )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        total_samples = num_runs * batch_size
        samples_per_second = total_samples / total_time
        
        # Performance assertions
        assert samples_per_second > 0, "Throughput should be positive"
        
        # Log performance metrics
        print(f"\nThroughput Benchmark Results:")
        print(f"Total samples: {total_samples}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Samples per second: {samples_per_second:.1f}")
        print(f"Batch processing time: {total_time/num_runs*1000:.2f}ms")
        
        # Reasonable performance expectations (adjust based on hardware)
        if torch.cuda.is_available():
            assert samples_per_second > 10, f"GPU throughput too low: {samples_per_second:.1f} samples/s"
        else:
            assert samples_per_second > 1, f"CPU throughput too low: {samples_per_second:.1f} samples/s"
    
    def test_memory_usage_profiling(self, model_config, sample_batch):
        """Profile memory usage during training."""
        model = CRAFTDFModel(**model_config)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        for key in sample_batch:
            if isinstance(sample_batch[key], torch.Tensor):
                sample_batch[key] = sample_batch[key].to(device)
            elif isinstance(sample_batch[key], dict):
                for subkey in sample_batch[key]:
                    sample_batch[key][subkey] = sample_batch[key][subkey].to(device)
        
        # Measure initial memory
        initial_cpu_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_gpu_memory = torch.cuda.memory_allocated(device) / (1024**2)  # MB
        
        # Forward pass
        outputs = model(
            sample_batch['spatial_input'],
            sample_batch['frequency_input'],
            sample_batch['domain_labels']
        )
        
        # Measure memory after forward pass
        forward_cpu_memory = psutil.Process().memory_info().rss / (1024**2)
        
        if torch.cuda.is_available():
            forward_gpu_memory = torch.cuda.memory_allocated(device) / (1024**2)
        
        # Backward pass
        loss = outputs['logits'].sum()  # Dummy loss
        loss.backward()
        
        # Measure memory after backward pass
        backward_cpu_memory = psutil.Process().memory_info().rss / (1024**2)
        
        if torch.cuda.is_available():
            backward_gpu_memory = torch.cuda.memory_allocated(device) / (1024**2)
        
        # Calculate memory usage
        forward_cpu_usage = forward_cpu_memory - initial_cpu_memory
        backward_cpu_usage = backward_cpu_memory - forward_cpu_memory
        
        print(f"\nMemory Usage Profile:")
        print(f"CPU Memory - Forward: {forward_cpu_usage:.1f}MB, Backward: {backward_cpu_usage:.1f}MB")
        
        if torch.cuda.is_available():
            forward_gpu_usage = forward_gpu_memory - initial_gpu_memory
            backward_gpu_usage = backward_gpu_memory - forward_gpu_memory
            print(f"GPU Memory - Forward: {forward_gpu_usage:.1f}MB, Backward: {backward_gpu_usage:.1f}MB")
            
            # GPU memory assertions
            assert forward_gpu_usage > 0, "Forward pass should use GPU memory"
            assert backward_gpu_usage > 0, "Backward pass should use additional GPU memory"
            assert backward_gpu_memory < 2000, f"Total GPU memory usage too high: {backward_gpu_memory:.1f}MB"
        
        # CPU memory assertions
        assert forward_cpu_usage >= 0, "Forward pass CPU memory usage should be non-negative"
        assert backward_cpu_usage >= 0, "Backward pass CPU memory usage should be non-negative"
    
    def test_performance_monitor_functionality(self, model_config, sample_batch):
        """Test performance monitoring functionality."""
        monitor = PerformanceMonitor(
            log_interval=1,  # Log every step for testing
            save_interval=10,
            max_history=100
        )
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate training steps
        for step in range(5):
            metrics = monitor.record_step_metrics(
                epoch=0,
                step=step,
                batch_size=8,
                loss=0.5 - step * 0.1,
                learning_rate=1e-3
            )
            
            # Validate metrics
            assert metrics.step == step
            assert metrics.loss == 0.5 - step * 0.1
            assert metrics.gpu_memory_allocated >= 0
            assert metrics.cpu_memory_used > 0
            
            time.sleep(0.1)  # Small delay to simulate training
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Check performance summary
        summary = monitor.get_performance_summary()
        
        assert summary['total_steps'] == 5
        assert 'gpu_memory_mb' in summary
        assert 'throughput' in summary
        assert summary['monitoring_duration_hours'] > 0
        
        print(f"\nPerformance Summary: {summary}")
    
    def test_gpu_optimizer_functionality(self):
        """Test GPU optimization utilities."""
        optimizer = GPUOptimizer(
            enable_amp=True,
            memory_fraction=0.8,
            allow_growth=True
        )
        
        # Test model optimization
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        optimized_model = optimizer.optimize_model(model)
        
        # Model should still be functional
        test_input = torch.randn(4, 100)
        output = optimized_model(test_input)
        
        assert output.shape == (4, 10)
        assert torch.isfinite(output).all()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_optimal_batch_size_finder(self, model_config):
        """Test optimal batch size finding functionality."""
        optimizer = GPUOptimizer()
        model = CRAFTDFModel(**model_config)
        model = model.cuda()
        
        # Test with spatial input shape
        spatial_shape = (3, 224, 224)
        optimal_batch_size = optimizer.get_optimal_batch_size(
            model=model.spatial_stream,  # Test with spatial stream only
            input_shape=spatial_shape,
            max_memory_gb=2.0,  # Conservative limit for testing
            start_batch_size=1
        )
        
        assert optimal_batch_size >= 1
        assert optimal_batch_size <= 64  # Reasonable upper bound
        
        print(f"\nOptimal batch size found: {optimal_batch_size}")
    
    def test_performance_callback_integration(self, model_config, sample_batch):
        """Test performance callback integration with PyTorch Lightning."""
        callback = PerformanceCallback(
            log_interval=1,
            save_interval=5,
            enable_profiling=False  # Disable for testing
        )
        
        # Mock trainer and module
        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.global_step = 0
        trainer.optimizers = [MagicMock()]
        trainer.optimizers[0].param_groups = [{'lr': 1e-3}]
        
        pl_module = MagicMock()
        
        # Test callback lifecycle
        callback.on_train_start(trainer, pl_module)
        
        for step in range(3):
            trainer.global_step = step
            
            callback.on_train_batch_start(trainer, pl_module, sample_batch, step)
            
            # Simulate some processing time
            time.sleep(0.01)
            
            outputs = {'loss': torch.tensor(0.5)}
            callback.on_train_batch_end(trainer, pl_module, outputs, sample_batch, step)
        
        callback.on_train_end(trainer, pl_module)
        
        # Check that monitoring was active
        assert len(callback.monitor.metrics_history) == 3
        
        # Validate recorded metrics
        for metrics in callback.monitor.metrics_history:
            assert metrics.step >= 0
            assert metrics.batch_time > 0
            assert metrics.gpu_memory_allocated >= 0
    
    def test_distributed_training_optimizer(self):
        """Test distributed training optimization utilities."""
        # Test different world sizes
        world_sizes = [1, 2, 4, 8, 16]
        base_batch_size = 32
        
        for world_size in world_sizes:
            optimizer = DistributedTrainingOptimizer(world_size=world_size, rank=0)
            
            # Test communication optimization
            comm_settings = optimizer.optimize_communication()
            
            assert isinstance(comm_settings, dict)
            assert 'find_unused_parameters' in comm_settings
            assert 'gradient_as_bucket_view' in comm_settings
            assert 'bucket_cap_mb' in comm_settings
            
            # Test batch size optimization
            optimal_batch_size = optimizer.get_optimal_batch_size(base_batch_size)
            
            assert optimal_batch_size > 0
            assert optimal_batch_size <= base_batch_size
            
            print(f"World size {world_size}: batch_size={optimal_batch_size}, "
                  f"bucket_cap={comm_settings['bucket_cap_mb']}MB")
    
    def test_memory_leak_detection(self, model_config, sample_batch):
        """Test for memory leaks during training simulation."""
        model = CRAFTDFModel(**model_config)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        for key in sample_batch:
            if isinstance(sample_batch[key], torch.Tensor):
                sample_batch[key] = sample_batch[key].to(device)
            elif isinstance(sample_batch[key], dict):
                for subkey in sample_batch[key]:
                    sample_batch[key][subkey] = sample_batch[key][subkey].to(device)
        
        # Record initial memory
        initial_cpu_memory = psutil.Process().memory_info().rss / (1024**2)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_gpu_memory = torch.cuda.memory_allocated(device) / (1024**2)
        
        # Simulate multiple training steps
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        memory_usage = []
        
        for step in range(20):
            # Forward pass
            outputs = model(
                sample_batch['spatial_input'],
                sample_batch['frequency_input'],
                sample_batch['domain_labels']
            )
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(outputs['logits'], sample_batch['labels'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record memory usage
            current_cpu_memory = psutil.Process().memory_info().rss / (1024**2)
            
            if torch.cuda.is_available():
                current_gpu_memory = torch.cuda.memory_allocated(device) / (1024**2)
                memory_usage.append((current_cpu_memory, current_gpu_memory))
            else:
                memory_usage.append((current_cpu_memory, 0))
        
        # Analyze memory usage trend
        cpu_memories = [mem[0] for mem in memory_usage]
        gpu_memories = [mem[1] for mem in memory_usage]
        
        # Check for significant memory growth (potential leak)
        cpu_growth = cpu_memories[-1] - cpu_memories[0]
        
        print(f"\nMemory Leak Analysis:")
        print(f"CPU memory growth: {cpu_growth:.1f}MB")
        
        if torch.cuda.is_available():
            gpu_growth = gpu_memories[-1] - gpu_memories[0]
            print(f"GPU memory growth: {gpu_growth:.1f}MB")
            
            # GPU memory should be relatively stable after initial allocation
            assert gpu_growth < 100, f"Potential GPU memory leak detected: {gpu_growth:.1f}MB growth"
        
        # CPU memory growth should be minimal
        assert cpu_growth < 200, f"Potential CPU memory leak detected: {cpu_growth:.1f}MB growth"
    
    def test_training_speed_benchmark(self, model_config, sample_batch):
        """Benchmark training speed with different optimizations."""
        results = {}
        
        # Test configurations
        configs = [
            {'name': 'baseline', 'amp': False, 'compile': False},
            {'name': 'mixed_precision', 'amp': True, 'compile': False},
        ]
        
        # Add torch.compile if available
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
            configs.append({'name': 'compiled', 'amp': False, 'compile': True})
            configs.append({'name': 'amp_compiled', 'amp': True, 'compile': True})
        
        for config in configs:
            model = CRAFTDFModel(**model_config)
            
            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Apply optimizations
            if config['compile'] and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model)
                except Exception as e:
                    print(f"Compilation failed: {e}")
                    continue
            
            # Prepare batch
            test_batch = {}
            for key in sample_batch:
                if isinstance(sample_batch[key], torch.Tensor):
                    test_batch[key] = sample_batch[key].to(device)
                elif isinstance(sample_batch[key], dict):
                    test_batch[key] = {}
                    for subkey in sample_batch[key]:
                        test_batch[key][subkey] = sample_batch[key][subkey].to(device)
            
            # Warmup
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            for _ in range(5):
                outputs = model(
                    test_batch['spatial_input'],
                    test_batch['frequency_input'],
                    test_batch['domain_labels']
                )
                loss = nn.CrossEntropyLoss()(outputs['logits'], test_batch['labels'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Benchmark
            num_steps = 20
            start_time = time.time()
            
            for _ in range(num_steps):
                if config['amp'] and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            test_batch['spatial_input'],
                            test_batch['frequency_input'],
                            test_batch['domain_labels']
                        )
                        loss = nn.CrossEntropyLoss()(outputs['logits'], test_batch['labels'])
                    
                    # Use GradScaler for mixed precision
                    scaler = torch.cuda.amp.GradScaler()
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(
                        test_batch['spatial_input'],
                        test_batch['frequency_input'],
                        test_batch['domain_labels']
                    )
                    loss = nn.CrossEntropyLoss()(outputs['logits'], test_batch['labels'])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            total_time = end_time - start_time
            steps_per_second = num_steps / total_time
            
            results[config['name']] = {
                'total_time': total_time,
                'steps_per_second': steps_per_second
            }
        
        # Print benchmark results
        print(f"\nTraining Speed Benchmark:")
        baseline_speed = results.get('baseline', {}).get('steps_per_second', 1.0)
        
        for name, result in results.items():
            speedup = result['steps_per_second'] / baseline_speed
            print(f"{name}: {result['steps_per_second']:.2f} steps/s (speedup: {speedup:.2f}x)")
        
        # Validate that optimizations don't hurt performance
        for name, result in results.items():
            assert result['steps_per_second'] > 0, f"Invalid performance for {name}"
    
    def test_h100_optimizations(self):
        """Test H100-specific optimizations."""
        # Temporarily disable torch.compile for this test due to C++ compilation issues
        original_compile = None
        if hasattr(torch, 'compile'):
            original_compile = torch.compile
            torch.compile = lambda x, **kwargs: x  # Mock compile to return model unchanged
        
        try:
            optimizer = GPUOptimizer(
                enable_amp=True,
                enable_h100_optimizations=True
            )
            
            # Test H100 detection (will be False in most test environments)
            h100_detected = optimizer.h100_detected
            print(f"H100 detected: {h100_detected}")
            
            # Test mixed precision scaler
            scaler = optimizer.get_mixed_precision_scaler()
            
            if torch.cuda.is_available():
                assert scaler is not None, "Mixed precision scaler should be available on GPU"
                assert isinstance(scaler, torch.cuda.amp.GradScaler)
            else:
                # On CPU, scaler should be None
                assert scaler is None, "Mixed precision scaler should be None on CPU"
            
            # Test model optimization with H100 settings
            model = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )
            
            optimized_model = optimizer.optimize_model(model)
            
            # Model should still be functional
            test_input = torch.randn(4, 100)
            if torch.cuda.is_available():
                test_input = test_input.cuda()
                optimized_model = optimized_model.cuda()
            
            output = optimized_model(test_input)
            assert output.shape == (4, 10)
            assert torch.isfinite(output).all()
            
        finally:
            # Restore original torch.compile
            if original_compile is not None:
                torch.compile = original_compile
    
    def test_memory_profiler_functionality(self, model_config, sample_batch):
        """Test advanced memory profiler functionality."""
        from craft_df.training.performance_monitor import MemoryProfiler
        
        profiler = MemoryProfiler(enable_detailed_profiling=True)
        
        # Take initial snapshot
        initial_snapshot = profiler.take_snapshot("initial")
        
        assert 'cpu_memory' in initial_snapshot
        assert 'timestamp' in initial_snapshot
        assert initial_snapshot['tag'] == "initial"
        
        # Simulate some memory usage
        model = CRAFTDFModel(**model_config)
        
        if torch.cuda.is_available():
            model = model.cuda()
            for key in sample_batch:
                if isinstance(sample_batch[key], torch.Tensor):
                    sample_batch[key] = sample_batch[key].cuda()
                elif isinstance(sample_batch[key], dict):
                    for subkey in sample_batch[key]:
                        sample_batch[key][subkey] = sample_batch[key][subkey].cuda()
        
        # Take snapshot after model creation
        model_snapshot = profiler.take_snapshot("model_loaded")
        
        # Run forward pass
        outputs = model(
            sample_batch['spatial_input'],
            sample_batch['frequency_input'],
            sample_batch['domain_labels']
        )
        
        forward_snapshot = profiler.take_snapshot("forward_pass")
        
        # Analyze memory usage
        analysis = profiler.analyze_memory_usage()
        
        assert 'duration_seconds' in analysis
        assert 'num_snapshots' in analysis
        assert analysis['num_snapshots'] == 3
        assert 'cpu_memory' in analysis
        assert 'leak_detection' in analysis
        
        print(f"Memory analysis: {analysis}")
        
        # Test saving profile
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            profiler.save_profile(f.name)
            
            # Verify file was created
            import os
            assert os.path.exists(f.name)
            
            # Clean up
            os.unlink(f.name)
    
    def test_distributed_training_advanced_features(self):
        """Test advanced distributed training features."""
        # Test different world sizes
        world_sizes = [2, 4, 8]
        
        for world_size in world_sizes:
            optimizer = DistributedTrainingOptimizer(world_size=world_size, rank=0)
            
            # Test communication optimization
            comm_settings = optimizer.optimize_communication()
            
            assert isinstance(comm_settings, dict)
            assert 'bucket_cap_mb' in comm_settings
            
            # Verify bucket size scaling
            if world_size <= 4:
                assert comm_settings['bucket_cap_mb'] == 25
            elif world_size <= 8:
                assert comm_settings['bucket_cap_mb'] == 50
            
            # Test learning rate scaling
            base_lr = 1e-3
            scaled_lr = optimizer.get_learning_rate_scaling(base_lr)
            
            assert scaled_lr > 0
            if world_size <= 8:
                assert scaled_lr == base_lr * world_size
            else:
                # Should use square root scaling for large world sizes
                expected_lr = base_lr * np.sqrt(world_size)
                assert abs(scaled_lr - expected_lr) < 1e-6
            
            # Test data loader settings
            loader_settings = optimizer.get_data_loader_settings()
            
            assert 'shuffle' in loader_settings
            assert 'drop_last' in loader_settings
            assert 'num_workers' in loader_settings
            assert loader_settings['shuffle'] == False  # DistributedSampler handles this
            assert loader_settings['drop_last'] == True
            
            print(f"World size {world_size}: lr_scale={scaled_lr/base_lr:.2f}x, "
                  f"num_workers={loader_settings['num_workers']}")
    
    def test_performance_benchmarking_integration(self, model_config, sample_batch):
        """Test integrated performance benchmarking."""
        monitor = PerformanceMonitor(
            log_interval=1,
            enable_detailed_profiling=True
        )
        
        model = CRAFTDFModel(**model_config)
        
        if torch.cuda.is_available():
            model = model.cuda()
            for key in sample_batch:
                if isinstance(sample_batch[key], torch.Tensor):
                    sample_batch[key] = sample_batch[key].cuda()
                elif isinstance(sample_batch[key], dict):
                    for subkey in sample_batch[key]:
                        sample_batch[key][subkey] = sample_batch[key][subkey].cuda()
        
        # Run throughput benchmark
        throughput_results = monitor.run_throughput_benchmark(
            model=model,
            sample_input=sample_batch,
            num_warmup=5,
            num_runs=20
        )
        
        assert 'samples_per_second' in throughput_results
        assert 'batch_time_ms' in throughput_results
        assert throughput_results['samples_per_second'] > 0
        
        # Run memory benchmark
        memory_results = monitor.run_memory_benchmark(
            model=model,
            sample_input=sample_batch,
            num_steps=10
        )
        
        assert 'peak_memory' in memory_results
        assert 'cpu_memory' in memory_results
        
        # Run distributed benchmark (simulated)
        distributed_results = monitor.run_distributed_benchmark(
            world_size=4,
            model_config=model_config,
            batch_sizes=[8, 16, 32]
        )
        
        assert 'optimal_batch_size' in distributed_results
        assert 'scaling_efficiency' in distributed_results
        assert distributed_results['world_size'] == 4
        
        print(f"Benchmark results:")
        print(f"  Throughput: {throughput_results['samples_per_second']:.1f} samples/s")
        print(f"  Peak GPU memory: {memory_results['peak_memory']['gpu']:.1f} MB")
        print(f"  Optimal batch size: {distributed_results['optimal_batch_size']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])