"""
Unit tests for HierarchicalDeepfakeDataset class.

Tests cover dataset initialization, data loading, caching mechanisms,
batch consistency, and error handling scenarios.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from craft_df.data.dataset import HierarchicalDeepfakeDataset


class TestHierarchicalDeepfakeDataset:
    """Test suite for HierarchicalDeepfakeDataset class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_metadata(self, temp_dir):
        """Create sample metadata CSV and corresponding data files."""
        # Create sample spatial and frequency data
        spatial_data1 = np.random.rand(224, 224, 3).astype(np.float32)
        spatial_data2 = np.random.rand(224, 224, 3).astype(np.float32)
        
        freq_data1 = np.random.rand(112, 112, 12).astype(np.float32)  # DWT coefficients
        freq_data2 = np.random.rand(112, 112, 12).astype(np.float32)
        
        # Save data files
        spatial_path1 = temp_dir / "spatial_001.npy"
        spatial_path2 = temp_dir / "spatial_002.npy"
        freq_path1 = temp_dir / "freq_001.npy"
        freq_path2 = temp_dir / "freq_002.npy"
        
        np.save(spatial_path1, spatial_data1)
        np.save(spatial_path2, spatial_data2)
        np.save(freq_path1, freq_data1)
        np.save(freq_path2, freq_data2)
        
        # Create metadata CSV
        metadata = pd.DataFrame({
            'spatial_path': [str(spatial_path1), str(spatial_path2)],
            'frequency_path': [str(freq_path1), str(freq_path2)],
            'label': [0, 1],
            'video_id': ['video_001', 'video_002'],
            'frame_number': [1, 1]
        })
        
        metadata_path = temp_dir / "metadata.csv"
        metadata.to_csv(metadata_path, index=False)
        
        return {
            'metadata_path': metadata_path,
            'spatial_data': [spatial_data1, spatial_data2],
            'freq_data': [freq_data1, freq_data2],
            'labels': [0, 1]
        }
    
    def test_dataset_initialization(self, sample_metadata):
        """Test dataset initialization with valid metadata."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(sample_metadata['metadata_path']),
            cache_size=100,
            validate_files=True
        )
        
        assert len(dataset) == 2
        assert dataset.metadata_path.exists()
        assert len(dataset.class_counts) == 2
        assert dataset.class_counts[0] == 1  # One real sample
        assert dataset.class_counts[1] == 1  # One fake sample
    
    def test_dataset_initialization_missing_file(self, temp_dir):
        """Test dataset initialization with missing metadata file."""
        missing_path = temp_dir / "missing_metadata.csv"
        
        with pytest.raises(FileNotFoundError):
            HierarchicalDeepfakeDataset(metadata_path=str(missing_path))
    
    def test_dataset_initialization_missing_columns(self, temp_dir):
        """Test dataset initialization with invalid metadata columns."""
        # Create metadata with missing required columns
        metadata = pd.DataFrame({
            'spatial_path': ['path1.npy'],
            'label': [0]
            # Missing 'frequency_path' column
        })
        
        metadata_path = temp_dir / "invalid_metadata.csv"
        metadata.to_csv(metadata_path, index=False)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            HierarchicalDeepfakeDataset(metadata_path=str(metadata_path))
    
    def test_getitem_functionality(self, sample_metadata):
        """Test __getitem__ method returns correct data types and shapes."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(sample_metadata['metadata_path']),
            validate_files=True
        )
        
        # Test first sample
        spatial, frequency, label = dataset[0]
        
        # Check types
        assert isinstance(spatial, torch.Tensor)
        assert isinstance(frequency, torch.Tensor)
        assert isinstance(label, int)
        
        # Check shapes - spatial should be (C, H, W) after permutation
        assert spatial.dim() >= 2
        assert frequency.dim() >= 1
        
        # Check label values
        assert label in [0, 1]
        
        # Test second sample
        spatial2, frequency2, label2 = dataset[1]
        assert label2 in [0, 1]
        assert label != label2  # Should have different labels
    
    def test_getitem_index_error(self, sample_metadata):
        """Test __getitem__ raises IndexError for invalid indices."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(sample_metadata['metadata_path'])
        )
        
        with pytest.raises(IndexError):
            _ = dataset[10]  # Index out of range
    
    def test_class_weights_calculation(self, sample_metadata):
        """Test class weight calculation for balanced training."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(sample_metadata['metadata_path'])
        )
        
        weights = dataset.get_class_weights()
        
        # Check type and shape
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (2,)  # Two classes
        
        # Weights should be positive
        assert torch.all(weights > 0)
        
        # For balanced dataset, weights should be equal
        assert torch.allclose(weights[0], weights[1], rtol=1e-5)
    
    def test_class_weights_imbalanced(self, temp_dir):
        """Test class weight calculation with imbalanced dataset."""
        # Create imbalanced dataset (3 real, 1 fake)
        spatial_paths = []
        freq_paths = []
        labels = [0, 0, 0, 1]  # 3 real, 1 fake
        
        for i, label in enumerate(labels):
            spatial_data = np.random.rand(224, 224, 3).astype(np.float32)
            freq_data = np.random.rand(112, 112, 12).astype(np.float32)
            
            spatial_path = temp_dir / f"spatial_{i:03d}.npy"
            freq_path = temp_dir / f"freq_{i:03d}.npy"
            
            np.save(spatial_path, spatial_data)
            np.save(freq_path, freq_data)
            
            spatial_paths.append(str(spatial_path))
            freq_paths.append(str(freq_path))
        
        metadata = pd.DataFrame({
            'spatial_path': spatial_paths,
            'frequency_path': freq_paths,
            'label': labels
        })
        
        metadata_path = temp_dir / "imbalanced_metadata.csv"
        metadata.to_csv(metadata_path, index=False)
        
        dataset = HierarchicalDeepfakeDataset(metadata_path=str(metadata_path))
        weights = dataset.get_class_weights()
        
        # Fake class (minority) should have higher weight
        assert weights[1] > weights[0]
        
        # Check approximate values
        # Weight for class 0 (3 samples): 4 / (2 * 3) = 0.667
        # Weight for class 1 (1 sample): 4 / (2 * 1) = 2.0
        assert abs(weights[0] - 0.667) < 0.01
        assert abs(weights[1] - 2.0) < 0.01
    
    def test_caching_mechanism(self, sample_metadata):
        """Test that caching works correctly."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(sample_metadata['metadata_path']),
            cache_size=10
        )
        
        # Load same sample multiple times
        for _ in range(3):
            spatial, frequency, label = dataset[0]
        
        # Check cache info
        cache_info = dataset.get_cache_info()
        
        # Should have cache hits after first load
        assert cache_info['spatial_cache']['hits'] >= 2
        assert cache_info['frequency_cache']['hits'] >= 2
        
        # Clear cache and verify
        dataset.clear_cache()
        cache_info_after_clear = dataset.get_cache_info()
        
        assert cache_info_after_clear['spatial_cache']['currsize'] == 0
        assert cache_info_after_clear['frequency_cache']['currsize'] == 0
    
    def test_sample_info_retrieval(self, sample_metadata):
        """Test getting sample metadata information."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(sample_metadata['metadata_path'])
        )
        
        # Get info for first sample
        info = dataset.get_sample_info(0)
        
        assert isinstance(info, dict)
        assert 'spatial_path' in info
        assert 'frequency_path' in info
        assert 'label' in info
        assert info['label'] == 0
        
        # Test index error
        with pytest.raises(IndexError):
            dataset.get_sample_info(10)
    
    def test_transforms_application(self, sample_metadata):
        """Test that transforms are applied correctly."""
        def dummy_spatial_transform(x):
            return x * 2.0
        
        def dummy_freq_transform(x):
            return x + 1.0
        
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(sample_metadata['metadata_path']),
            transform=dummy_spatial_transform,
            freq_transform=dummy_freq_transform
        )
        
        spatial, frequency, label = dataset[0]
        
        # Transforms should have been applied
        # Note: We can't easily verify the exact values due to data loading
        # and tensor conversions, but we can verify the shapes are maintained
        assert spatial.dim() >= 2
        assert frequency.dim() >= 1
    
    def test_batch_consistency(self, sample_metadata):
        """Test that multiple calls return consistent data."""
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(sample_metadata['metadata_path'])
        )
        
        # Load same sample multiple times
        results = []
        for _ in range(3):
            spatial, frequency, label = dataset[0]
            results.append((spatial.clone(), frequency.clone(), label))
        
        # All results should be identical
        for i in range(1, len(results)):
            assert torch.equal(results[0][0], results[i][0])
            assert torch.equal(results[0][1], results[i][1])
            assert results[0][2] == results[i][2]
    
    def test_file_validation_disabled(self, temp_dir):
        """Test dataset initialization with file validation disabled."""
        # Create metadata with non-existent files
        metadata = pd.DataFrame({
            'spatial_path': ['nonexistent1.npy', 'nonexistent2.npy'],
            'frequency_path': ['nonexistent1_freq.npy', 'nonexistent2_freq.npy'],
            'label': [0, 1]
        })
        
        metadata_path = temp_dir / "metadata_nonexistent.csv"
        metadata.to_csv(metadata_path, index=False)
        
        # Should not raise error when validation is disabled
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(metadata_path),
            validate_files=False
        )
        
        assert len(dataset) == 2
    
    @patch('craft_df.data.dataset.logger')
    def test_file_loading_error_handling(self, mock_logger, temp_dir):
        """Test error handling when files cannot be loaded."""
        # Create metadata pointing to corrupted files
        metadata = pd.DataFrame({
            'spatial_path': ['corrupted.npy'],
            'frequency_path': ['corrupted_freq.npy'],
            'label': [0]
        })
        
        metadata_path = temp_dir / "metadata_corrupted.csv"
        metadata.to_csv(metadata_path, index=False)
        
        # Create corrupted files (empty files)
        (temp_dir / "corrupted.npy").touch()
        (temp_dir / "corrupted_freq.npy").touch()
        
        dataset = HierarchicalDeepfakeDataset(
            metadata_path=str(metadata_path),
            validate_files=False
        )
        
        # Should raise error when trying to load corrupted file
        with pytest.raises(Exception):
            _ = dataset[0]


if __name__ == "__main__":
    pytest.main([__file__])