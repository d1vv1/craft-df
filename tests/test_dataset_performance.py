"""
Performance tests for HierarchicalDeepfakeDataset.

Tests memory usage, loading speed, caching efficiency, and scalability
under various configurations and dataset sizes.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import time
import psutil
import gc
from torch.utils.data import DataLoader

from craft_df.data.dataset import HierarchicalDeepfakeDataset
from craft_df.data.transforms import create_train_transforms, create_val_transforms


class TestDatasetPerformance:
    """Performance test suite for HierarchicalDeepfakeDataset."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create a larger dataset for performance testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create 100 samples for performance testing
            num_samples = 100
            spatial_paths = []
            freq_paths = []
            labels = []
            
            for i in range(num_samples):
                # Create realistic-sized data
                spatial_data = np.random.rand(224, 224, 3).astype(np.float32)
                freq_data = np.random.rand(112, 112, 12).astype(np.float32)
                
                spatial_path = temp_path / f"spatial_{i:03d}.npy"
                freq_path = temp_path / f"freq_{i:03d}.npy"
                
                np.save(spatial_path, spatial_data)
                np.save(freq_path, freq_data)
                
                spatial_paths.append(str(spatial_path))
                freq_paths.append(str(freq_path))
                labels.append(i % 2)  # Alternating labels
            
            # Create metadata
            metadata = pd.DataFrame({
                'spatial_path': spatial_paths,
                'frequency_path': freq_paths,
                'label': labels,
                'video_id': [f'video_{i//10:03d}' for i in range(num_samples)],
                'frame_number': [i % 10 for i in range(num_samples)]
            })
            
            metadata_path = temp_path / "metadata.csv"
            metadata.to_csv(metadata_path, index=False)
            
            yield {
                'metadata_path': metadata_path,
                'num_samples': num_samples,
                'temp_dir': temp_path
            }
    
    def test_loading_speed_baseline(self, large_dataset):
        """Test baseline loading speed without optimizations."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            cache_size=0,  # No caching
            use_memory_mapping=False,
            validate_files=False
        )
        
        # Time loading first 20 samples
        start_time = time.time()
        for i in range(min(20, len(dataset))):
            spatial, frequency, label = dataset[i]
        
        baseline_time = time.time() - start_time
        
        # Get performance stats
        stats = dataset.get_performance_stats()
        
        assert stats['total_samples_loaded'] == 20
        assert stats['avg_load_time_ms'] > 0
        assert stats['samples_per_second'] > 0
        
        print(f"Baseline loading: {baseline_time:.3f}s for 20 samples")
        print(f"Average load time: {stats['avg_load_time_ms']:.2f}ms")
        print(f"Samples per second: {stats['samples_per_second']:.1f}")
    
    def test_loading_speed_with_caching(self, large_dataset):
        """Test loading speed improvement with caching."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            cache_size=50,  # Cache 50 samples
            use_memory_mapping=True,
            validate_files=False
        )
        
        # First pass - populate cache
        start_time = time.time()
        for i in range(min(20, len(dataset))):
            spatial, frequency, label = dataset[i]
        first_pass_time = time.time() - start_time
        
        # Second pass - should use cache
        start_time = time.time()
        for i in range(min(20, len(dataset))):
            spatial, frequency, label = dataset[i]
        second_pass_time = time.time() - start_time
        
        # Cache should improve performance
        assert second_pass_time < first_pass_time
        
        # Check cache statistics
        cache_info = dataset.get_cache_info()
        assert cache_info['spatial_cache']['hits'] > 0
        assert cache_info['frequency_cache']['hits'] > 0
        
        print(f"First pass: {first_pass_time:.3f}s")
        print(f"Second pass (cached): {second_pass_time:.3f}s")
        print(f"Speedup: {first_pass_time / second_pass_time:.2f}x")
    
    def test_memory_mapping_performance(self, large_dataset):
        """Test performance difference with memory mapping."""
        # Test without memory mapping
        dataset_no_mmap = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            cache_size=0,
            use_memory_mapping=False,
            validate_files=False
        )
        
        start_time = time.time()
        for i in range(min(10, len(dataset_no_mmap))):
            spatial, frequency, label = dataset_no_mmap[i]
        no_mmap_time = time.time() - start_time
        
        # Test with memory mapping
        dataset_mmap = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            cache_size=0,
            use_memory_mapping=True,
            validate_files=False
        )
        
        start_time = time.time()
        for i in range(min(10, len(dataset_mmap))):
            spatial, frequency, label = dataset_mmap[i]
        mmap_time = time.time() - start_time
        
        print(f"Without memory mapping: {no_mmap_time:.3f}s")
        print(f"With memory mapping: {mmap_time:.3f}s")
        
        # Memory mapping should not significantly degrade performance
        assert mmap_time < no_mmap_time * 2.0  # Allow up to 2x slower
    
    def test_memory_usage_monitoring(self, large_dataset):
        """Test memory usage monitoring and limits."""
        initial_memory = psutil.Process().memory_info().rss
        
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            cache_size=100,
            memory_limit_gb=0.1,  # Very low limit to trigger monitoring
            validate_files=False
        )
        
        # Load samples to trigger memory monitoring
        for i in range(min(50, len(dataset))):
            spatial, frequency, label = dataset[i]
        
        # Check that memory monitoring is working
        stats = dataset.get_performance_stats()
        assert 'avg_memory_usage_mb' in stats
        assert stats['max_memory_usage_mb'] > 0
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024**2  # MB
        
        print(f"Memory increase: {memory_increase:.1f}MB")
        print(f"Max memory usage: {stats['max_memory_usage_mb']:.1f}MB")
    
    def test_preloading_performance(self, large_dataset):
        """Test preloading vs lazy loading performance."""
        # Test lazy loading
        dataset_lazy = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            preload_data=False,
            cache_size=0,
            validate_files=False
        )
        
        start_time = time.time()
        for i in range(min(20, len(dataset_lazy))):
            spatial, frequency, label = dataset_lazy[i]
        lazy_time = time.time() - start_time
        
        # Test preloading (with small dataset to avoid memory issues)
        dataset_preload = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            preload_data=True,
            memory_limit_gb=1.0,  # Allow some memory for preloading
            validate_files=False
        )
        
        start_time = time.time()
        for i in range(min(20, len(dataset_preload))):
            spatial, frequency, label = dataset_preload[i]
        preload_time = time.time() - start_time
        
        print(f"Lazy loading: {lazy_time:.3f}s")
        print(f"Preloaded: {preload_time:.3f}s")
        
        # Preloading should be faster for repeated access
        if hasattr(dataset_preload, 'preloaded_spatial'):
            assert preload_time <= lazy_time
    
    def test_dataloader_performance(self, large_dataset):
        """Test performance with PyTorch DataLoader."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            cache_size=50,
            use_memory_mapping=True,
            validate_files=False
        )
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]
        results = {}
        
        for batch_size in batch_sizes:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0  # Single-threaded for consistent testing
            )
            
            start_time = time.time()
            samples_processed = 0
            
            for batch_idx, (spatial, frequency, labels) in enumerate(dataloader):
                samples_processed += len(labels)
                if batch_idx >= 10:  # Process 10 batches
                    break
            
            batch_time = time.time() - start_time
            samples_per_second = samples_processed / batch_time if batch_time > 0 else 0
            
            results[batch_size] = {
                'time': batch_time,
                'samples_per_second': samples_per_second,
                'samples_processed': samples_processed
            }
            
            print(f"Batch size {batch_size}: {samples_per_second:.1f} samples/sec")
        
        # Larger batch sizes should generally be more efficient
        assert results[16]['samples_per_second'] >= results[1]['samples_per_second'] * 0.5
    
    def test_transform_performance_impact(self, large_dataset):
        """Test performance impact of data transforms."""
        # Create transforms
        spatial_transform, freq_transform = create_train_transforms(
            spatial_augmentation=True,
            frequency_augmentation=True
        )
        
        # Test without transforms
        dataset_no_transform = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            cache_size=20,
            validate_files=False
        )
        
        start_time = time.time()
        for i in range(min(10, len(dataset_no_transform))):
            spatial, frequency, label = dataset_no_transform[i]
        no_transform_time = time.time() - start_time
        
        # Test with transforms
        dataset_with_transform = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            transform=spatial_transform,
            freq_transform=freq_transform,
            cache_size=20,
            validate_files=False
        )
        
        start_time = time.time()
        for i in range(min(10, len(dataset_with_transform))):
            spatial, frequency, label = dataset_with_transform[i]
        with_transform_time = time.time() - start_time
        
        print(f"Without transforms: {no_transform_time:.3f}s")
        print(f"With transforms: {with_transform_time:.3f}s")
        print(f"Transform overhead: {(with_transform_time - no_transform_time):.3f}s")
        
        # Transforms should add some overhead but not be excessive
        assert with_transform_time < no_transform_time * 5.0  # Allow up to 5x slower
    
    def test_cache_size_optimization(self, large_dataset):
        """Test automatic cache size optimization."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            cache_size=10,  # Start with small cache
            validate_files=False
        )
        
        # Test cache size optimization
        recommended_size = dataset.optimize_cache_size(target_memory_gb=0.1)
        
        assert isinstance(recommended_size, int)
        assert recommended_size > 0
        assert recommended_size <= len(dataset)
        
        print(f"Recommended cache size: {recommended_size}")
    
    def test_concurrent_access_performance(self, large_dataset):
        """Test performance with concurrent access patterns."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(large_dataset['metadata_path']),
            cache_size=50,
            validate_files=False
        )
        
        # Simulate random access pattern
        indices = np.random.choice(len(dataset), size=30, replace=True)
        
        start_time = time.time()
        for idx in indices:
            spatial, frequency, label = dataset[idx]
        random_access_time = time.time() - start_time
        
        # Simulate sequential access pattern
        start_time = time.time()
        for i in range(30):
            spatial, frequency, label = dataset[i % len(dataset)]
        sequential_access_time = time.time() - start_time
        
        print(f"Random access: {random_access_time:.3f}s")
        print(f"Sequential access: {sequential_access_time:.3f}s")
        
        # Both should complete successfully
        assert random_access_time > 0
        assert sequential_access_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])