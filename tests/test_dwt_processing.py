"""
Unit tests for DWT processing functionality.

This module contains comprehensive tests for the DWTProcessor class,
including numerical stability tests, edge cases, and theoretical validation.
"""

import pytest
import numpy as np
import pywt
from unittest.mock import Mock, patch
from craft_df.data.dwt_processing import DWTProcessor


class TestDWTProcessorInit:
    """Test suite for DWTProcessor initialization."""
    
    def test_init_valid_parameters(self):
        """Test DWTProcessor initialization with valid parameters."""
        processor = DWTProcessor(
            wavelet='db8',
            levels=2,
            mode='periodization'
        )
        
        assert processor.wavelet == 'db8'
        assert processor.levels == 2
        assert processor.mode == 'periodization'
        assert processor.filter_length > 0
    
    def test_init_default_parameters(self):
        """Test DWTProcessor initialization with default parameters."""
        processor = DWTProcessor()
        
        assert processor.wavelet == 'db4'
        assert processor.levels == 3
        assert processor.mode == 'symmetric'
    
    def test_init_invalid_wavelet(self):
        """Test DWTProcessor initialization with invalid wavelet."""
        with pytest.raises(ValueError, match="Unsupported wavelet type"):
            DWTProcessor(wavelet='invalid_wavelet')
    
    def test_init_invalid_levels(self):
        """Test DWTProcessor initialization with invalid levels."""
        with pytest.raises(ValueError, match="levels must be an integer between 1 and 6"):
            DWTProcessor(levels=0)
        
        with pytest.raises(ValueError, match="levels must be an integer between 1 and 6"):
            DWTProcessor(levels=7)
        
        with pytest.raises(ValueError, match="levels must be an integer between 1 and 6"):
            DWTProcessor(levels=1.5)
    
    def test_init_invalid_mode(self):
        """Test DWTProcessor initialization with invalid mode."""
        with pytest.raises(ValueError, match="Unsupported mode"):
            DWTProcessor(mode='invalid_mode')


class TestDWTProcessorDecomposition:
    """Test suite for DWT decomposition methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DWTProcessor(wavelet='db4', levels=2, mode='symmetric')
        
        # Create test images
        self.grayscale_image = np.random.rand(64, 64).astype(np.float32)
        self.color_image = np.random.rand(64, 64, 3).astype(np.float32)
        self.uint8_image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    
    def test_decompose_2d_grayscale(self):
        """Test 2D DWT decomposition on grayscale image."""
        coefficients = self.processor.decompose_2d(self.grayscale_image)
        
        # Should have levels + 1 coefficient sets (approximation + detail levels)
        assert len(coefficients) == self.processor.levels + 1
        
        # First coefficient should be approximation (LL)
        assert isinstance(coefficients[0], np.ndarray)
        assert coefficients[0].ndim == 2
        
        # Remaining coefficients should be detail tuples (LH, HL, HH)
        for i in range(1, len(coefficients)):
            assert isinstance(coefficients[i], tuple)
            assert len(coefficients[i]) == 3
            for detail in coefficients[i]:
                assert isinstance(detail, np.ndarray)
                assert detail.ndim == 2
    
    def test_decompose_2d_color(self):
        """Test 2D DWT decomposition on color image."""
        coefficients = self.processor.decompose_2d(self.color_image)
        
        # Should have levels + 1 coefficient sets
        assert len(coefficients) == self.processor.levels + 1
        
        # First coefficient should be approximation with 3 channels
        assert isinstance(coefficients[0], np.ndarray)
        assert coefficients[0].ndim == 3
        assert coefficients[0].shape[2] == 3
        
        # Detail coefficients should also have 3 channels
        for i in range(1, len(coefficients)):
            assert isinstance(coefficients[i], tuple)
            assert len(coefficients[i]) == 3
            for detail in coefficients[i]:
                assert isinstance(detail, np.ndarray)
                assert detail.ndim == 3
                assert detail.shape[2] == 3
    
    def test_decompose_2d_uint8_conversion(self):
        """Test DWT decomposition with uint8 input (should be converted to float)."""
        coefficients = self.processor.decompose_2d(self.uint8_image)
        
        # Should successfully decompose uint8 image
        assert len(coefficients) == self.processor.levels + 1
        
        # All coefficients should be float type
        assert coefficients[0].dtype in [np.float32, np.float64]
        for i in range(1, len(coefficients)):
            for detail in coefficients[i]:
                assert detail.dtype in [np.float32, np.float64]
    
    def test_decompose_2d_invalid_input(self):
        """Test decompose_2d with invalid inputs."""
        # Test non-numpy array
        with pytest.raises(ValueError, match="Image must be a numpy array"):
            self.processor.decompose_2d("not_an_array")
        
        # Test wrong dimensions
        with pytest.raises(ValueError, match="Image must be 2D \\(grayscale\\) or 3D \\(color\\)"):
            self.processor.decompose_2d(np.zeros((64,)))
        
        # Test wrong dimensions (4D)
        with pytest.raises(ValueError, match="Image must be 2D \\(grayscale\\) or 3D \\(color\\)"):
            self.processor.decompose_2d(np.zeros((64, 64, 3, 2)))
    
    def test_decompose_2d_assertions(self):
        """Test decompose_2d input assertions."""
        # Test unsupported dtype
        invalid_dtype_image = np.zeros((64, 64), dtype=np.int32)
        with pytest.raises(AssertionError, match="Expected uint8, float32, or float64 image"):
            self.processor.decompose_2d(invalid_dtype_image)
        
        # Test zero dimensions
        zero_dim_image = np.zeros((0, 64), dtype=np.float32)
        with pytest.raises(AssertionError, match="Image dimensions must be positive"):
            self.processor.decompose_2d(zero_dim_image)
    
    def test_decompose_2d_coefficient_shapes(self):
        """Test that coefficient shapes are reasonable."""
        image = np.random.rand(128, 128).astype(np.float32)
        coefficients = self.processor.decompose_2d(image)
        
        # Check that we have the expected number of levels
        assert len(coefficients) == self.processor.levels + 1
        
        # Check that all coefficients have positive dimensions
        for i, coeff in enumerate(coefficients):
            if i == 0:  # Approximation
                assert coeff.shape[0] > 0 and coeff.shape[1] > 0
            else:  # Details
                assert len(coeff) == 3  # LH, HL, HH
                for detail in coeff:
                    assert detail.shape[0] > 0 and detail.shape[1] > 0
        
        # Check that approximation coefficients are smaller than original image
        approx_size = coefficients[0].shape[0] * coefficients[0].shape[1]
        original_size = image.shape[0] * image.shape[1]
        assert approx_size < original_size


class TestDWTProcessorFeatureExtraction:
    """Test suite for DWT feature extraction methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DWTProcessor(wavelet='db4', levels=2, mode='symmetric')
        
        # Create test coefficients
        self.test_image = np.random.rand(64, 64).astype(np.float32)
        self.test_coefficients = self.processor.decompose_2d(self.test_image)
    
    def test_extract_features_valid_coefficients(self):
        """Test feature extraction with valid coefficients."""
        features = self.processor.extract_features(self.test_coefficients)
        
        # Should return a 1D numpy array
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert len(features) > 0
        
        # All features should be finite
        assert np.all(np.isfinite(features))
        
        # Features should be float32
        assert features.dtype == np.float32
    
    def test_extract_features_expected_length(self):
        """Test that feature vector has expected length."""
        features = self.processor.extract_features(self.test_coefficients)
        
        # Calculate expected feature length
        # Approximation: 8 features
        # Each detail level: 3 subbands * 8 features + 3 cross-correlation = 27 features
        # Total: 8 + (levels * 27)
        expected_length = 8 + (self.processor.levels * 27)
        
        assert len(features) == expected_length
    
    def test_extract_features_invalid_input(self):
        """Test extract_features with invalid inputs."""
        # Test empty list
        with pytest.raises(ValueError, match="Coefficients must be a non-empty list"):
            self.processor.extract_features([])
        
        # Test non-list input
        with pytest.raises(ValueError, match="Coefficients must be a non-empty list"):
            self.processor.extract_features("not_a_list")
    
    def test_extract_statistical_features(self):
        """Test statistical feature extraction from coefficients."""
        # Create test coefficient array
        test_coeffs = np.random.randn(32, 32).astype(np.float32)
        
        features = self.processor._extract_statistical_features(test_coeffs, "test")
        
        # Should return 8 statistical features
        assert len(features) == 8
        assert all(isinstance(f, float) for f in features)
        assert all(np.isfinite(f) for f in features)
    
    def test_extract_statistical_features_edge_cases(self):
        """Test statistical feature extraction with edge cases."""
        # Test with constant array
        constant_coeffs = np.ones((32, 32), dtype=np.float32)
        features = self.processor._extract_statistical_features(constant_coeffs, "constant")
        
        assert len(features) == 8
        assert features[1] == 0.0  # Standard deviation should be 0
        assert all(np.isfinite(f) for f in features)
        
        # Test with array containing NaN
        nan_coeffs = np.random.randn(32, 32).astype(np.float32)
        nan_coeffs[0, 0] = np.nan
        features = self.processor._extract_statistical_features(nan_coeffs, "nan_test")
        
        assert len(features) == 8
        assert all(np.isfinite(f) for f in features)
    
    def test_extract_cross_correlation_features(self):
        """Test cross-correlation feature extraction."""
        # Create test detail coefficients
        lh = np.random.randn(16, 16).astype(np.float32)
        hl = np.random.randn(16, 16).astype(np.float32)
        hh = np.random.randn(16, 16).astype(np.float32)
        
        features = self.processor._extract_cross_correlation_features(lh, hl, hh, 1)
        
        # Should return 3 correlation features
        assert len(features) == 3
        assert all(isinstance(f, float) for f in features)
        assert all(np.isfinite(f) for f in features)
        assert all(-1.0 <= f <= 1.0 for f in features)  # Correlations should be in [-1, 1]
    
    def test_extract_cross_correlation_edge_cases(self):
        """Test cross-correlation with edge cases."""
        # Test with constant arrays (should give 0 correlation)
        lh = np.ones((16, 16), dtype=np.float32)
        hl = np.ones((16, 16), dtype=np.float32) * 2
        hh = np.ones((16, 16), dtype=np.float32) * 3
        
        features = self.processor._extract_cross_correlation_features(lh, hl, hh, 1)
        
        assert len(features) == 3
        assert all(f == 0.0 for f in features)  # Should be 0 for constant arrays
        
        # Test with arrays containing NaN
        lh_nan = np.random.randn(16, 16).astype(np.float32)
        lh_nan[0, 0] = np.nan
        hl_nan = np.random.randn(16, 16).astype(np.float32)
        hh_nan = np.random.randn(16, 16).astype(np.float32)
        
        features = self.processor._extract_cross_correlation_features(lh_nan, hl_nan, hh_nan, 1)
        
        assert len(features) == 3
        assert all(np.isfinite(f) for f in features)


class TestDWTProcessorPipeline:
    """Test suite for complete DWT processing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DWTProcessor()
        
        # Create test face crops
        self.face_crop_gray = np.random.rand(224, 224).astype(np.float32)
        self.face_crop_color = np.random.rand(224, 224, 3).astype(np.float32)
        self.face_crop_uint8 = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    
    def test_process_face_crop_grayscale(self):
        """Test complete processing pipeline with grayscale face crop."""
        features = self.processor.process_face_crop(self.face_crop_gray)
        
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert len(features) > 0
        assert np.all(np.isfinite(features))
        assert features.dtype == np.float32
    
    def test_process_face_crop_color(self):
        """Test complete processing pipeline with color face crop."""
        features = self.processor.process_face_crop(self.face_crop_color)
        
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert len(features) > 0
        assert np.all(np.isfinite(features))
        assert features.dtype == np.float32
    
    def test_process_face_crop_uint8(self):
        """Test complete processing pipeline with uint8 face crop."""
        features = self.processor.process_face_crop(self.face_crop_uint8)
        
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert len(features) > 0
        assert np.all(np.isfinite(features))
        assert features.dtype == np.float32
    
    def test_process_face_crop_invalid_input(self):
        """Test process_face_crop with invalid inputs."""
        # Test non-numpy array
        with pytest.raises(ValueError, match="Face image must be a numpy array"):
            self.processor.process_face_crop("not_an_array")
        
        # Test wrong dimensions
        with pytest.raises(ValueError, match="Face image must be 2D \\(grayscale\\) or 3D \\(color\\)"):
            self.processor.process_face_crop(np.zeros((224,)))
        
        # Test zero dimensions
        zero_dim_image = np.zeros((0, 224, 3), dtype=np.float32)
        with pytest.raises(AssertionError, match="Face image dimensions must be positive"):
            self.processor.process_face_crop(zero_dim_image)
    
    def test_process_face_crop_consistency(self):
        """Test that processing the same image gives consistent results."""
        features1 = self.processor.process_face_crop(self.face_crop_color)
        features2 = self.processor.process_face_crop(self.face_crop_color)
        
        # Should get identical results for the same input
        np.testing.assert_array_equal(features1, features2)
    
    def test_get_feature_names(self):
        """Test feature name generation."""
        feature_names = self.processor.get_feature_names()
        
        # Should return a list of strings
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)
        
        # Number of names should match feature vector length
        test_features = self.processor.process_face_crop(self.face_crop_color)
        assert len(feature_names) == len(test_features)
        
        # Names should be descriptive
        assert any('LL_0' in name for name in feature_names)  # Approximation features
        assert any('LH_1' in name for name in feature_names)  # Detail features
        assert any('corr_' in name for name in feature_names)  # Correlation features
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.processor)
        
        assert isinstance(repr_str, str)
        assert 'DWTProcessor' in repr_str
        assert 'db4' in repr_str
        assert 'levels=3' in repr_str


class TestDWTProcessorNumericalStability:
    """Test suite for numerical stability and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DWTProcessor()
    
    def test_small_image_processing(self):
        """Test processing with very small images."""
        # Test minimum viable image size
        small_image = np.random.rand(8, 8).astype(np.float32)
        
        try:
            features = self.processor.process_face_crop(small_image)
            assert isinstance(features, np.ndarray)
            assert np.all(np.isfinite(features))
        except RuntimeError:
            # Small images might not be decomposable to the requested levels
            # This is acceptable behavior
            pass
    
    def test_extreme_values(self):
        """Test processing with extreme pixel values."""
        # Test with very large values
        large_image = np.ones((64, 64), dtype=np.float32) * 1e6
        features_large = self.processor.process_face_crop(large_image)
        assert np.all(np.isfinite(features_large))
        
        # Test with very small values
        small_image = np.ones((64, 64), dtype=np.float32) * 1e-6
        features_small = self.processor.process_face_crop(small_image)
        assert np.all(np.isfinite(features_small))
        
        # Test with negative values
        negative_image = np.ones((64, 64), dtype=np.float32) * -1.0
        features_negative = self.processor.process_face_crop(negative_image)
        assert np.all(np.isfinite(features_negative))
    
    def test_different_wavelets(self):
        """Test processing with different wavelet types."""
        test_image = np.random.rand(64, 64).astype(np.float32)
        
        wavelets_to_test = ['db2', 'db4', 'db8', 'haar', 'bior2.2']
        
        for wavelet in wavelets_to_test:
            if wavelet in pywt.wavelist():
                processor = DWTProcessor(wavelet=wavelet, levels=2)
                features = processor.process_face_crop(test_image)
                
                assert isinstance(features, np.ndarray)
                assert np.all(np.isfinite(features))
                assert len(features) > 0
    
    def test_different_levels(self):
        """Test processing with different decomposition levels."""
        test_image = np.random.rand(128, 128).astype(np.float32)
        
        for levels in [1, 2, 3, 4]:
            processor = DWTProcessor(levels=levels)
            features = processor.process_face_crop(test_image)
            
            assert isinstance(features, np.ndarray)
            assert np.all(np.isfinite(features))
            
            # More levels should generally produce more features
            expected_length = 8 + (levels * 27)
            assert len(features) == expected_length


class TestDWTProcessorIntegration:
    """Integration tests for DWT processing with realistic data."""
    
    def test_realistic_face_processing(self):
        """Test DWT processing on realistic face-like images."""
        processor = DWTProcessor()
        
        # Create a more realistic face-like image (gradient with some structure)
        x, y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
        face_like = np.exp(-(x**2 + y**2)) + 0.1 * np.random.randn(224, 224)
        face_like = np.clip(face_like, 0, 1).astype(np.float32)
        
        # Add color channels
        face_rgb = np.stack([face_like, face_like * 0.8, face_like * 0.6], axis=-1)
        
        features = processor.process_face_crop(face_rgb)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert np.all(np.isfinite(features))
        
        # Features should have reasonable ranges (not all zeros or extremely large)
        assert np.std(features) > 0  # Should have some variation
        assert np.max(np.abs(features)) < 1e6  # Should not be extremely large
    
    def test_batch_processing_consistency(self):
        """Test that batch processing gives consistent results."""
        processor = DWTProcessor()
        
        # Create multiple similar images
        base_image = np.random.rand(64, 64, 3).astype(np.float32)
        images = [base_image + 0.01 * np.random.randn(64, 64, 3) for _ in range(5)]
        
        # Process each image
        features_list = [processor.process_face_crop(img.astype(np.float32)) for img in images]
        
        # All feature vectors should have the same length
        feature_lengths = [len(f) for f in features_list]
        assert all(length == feature_lengths[0] for length in feature_lengths)
        
        # All features should be finite
        for features in features_list:
            assert np.all(np.isfinite(features))


if __name__ == "__main__":
    pytest.main([__file__])