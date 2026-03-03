"""
Unit tests for face detection and cropping functionality.

This module contains comprehensive tests for the FaceDetector class,
including accuracy tests, edge cases, and error handling validation.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from craft_df.data.face_detection import FaceDetector


class TestFaceDetectorInit:
    """Test suite for FaceDetector initialization."""
    
    def test_init_valid_parameters(self):
        """Test FaceDetector initialization with valid parameters."""
        with patch.object(FaceDetector, '_load_face_detection_model'):
            detector = FaceDetector(
                min_detection_confidence=0.5,
                model_selection=1,
                target_size=(256, 256)
            )
            
            assert detector.min_detection_confidence == 0.5
            assert detector.model_selection == 1
            assert detector.target_size == (256, 256)
    
    def test_init_invalid_confidence(self):
        """Test FaceDetector initialization with invalid confidence."""
        with pytest.raises(ValueError, match="min_detection_confidence must be between"):
            FaceDetector(min_detection_confidence=1.5)
        
        with pytest.raises(ValueError, match="min_detection_confidence must be between"):
            FaceDetector(min_detection_confidence=-0.1)
    
    def test_init_invalid_model_selection(self):
        """Test FaceDetector initialization with invalid model selection."""
        with pytest.raises(ValueError, match="model_selection must be 0"):
            FaceDetector(model_selection=2)
    
    def test_init_invalid_target_size(self):
        """Test FaceDetector initialization with invalid target size."""
        with pytest.raises(ValueError, match="target_size must be a tuple"):
            FaceDetector(target_size=(224,))
        
        with pytest.raises(ValueError, match="target_size must be a tuple"):
            FaceDetector(target_size=(0, 224))


class TestFaceDetectorMethods:
    """Test suite for FaceDetector methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create detector with mocked model loading
        with patch.object(FaceDetector, '_load_face_detection_model'):
            self.detector = FaceDetector(
                min_detection_confidence=0.7,
                model_selection=0,
                target_size=(224, 224)
            )
        
        # Create a synthetic test image (blue square)
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_image[:, :] = [255, 0, 0]  # Blue in BGR
    
    def test_detect_faces_invalid_input(self):
        """Test detect_faces with invalid input."""
        # Test non-numpy array
        with pytest.raises(ValueError, match="Image must be a numpy array"):
            self.detector.detect_faces("not_an_array")
        
        # Test wrong dimensions
        with pytest.raises(ValueError, match="Image must be 3-channel"):
            self.detector.detect_faces(np.zeros((100, 100)))
        
        # Test wrong number of channels
        with pytest.raises(ValueError, match="Image must be 3-channel"):
            self.detector.detect_faces(np.zeros((100, 100, 4)))
    
    def test_detect_faces_assertions(self):
        """Test detect_faces input assertions."""
        # Test wrong dtype
        wrong_dtype_image = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(AssertionError, match="Expected uint8 image"):
            self.detector.detect_faces(wrong_dtype_image)
        
        # Test zero dimensions
        zero_dim_image = np.zeros((0, 100, 3), dtype=np.uint8)
        with pytest.raises(AssertionError, match="Image dimensions must be positive"):
            self.detector.detect_faces(zero_dim_image)
    
    def test_detect_faces_no_detections(self):
        """Test detect_faces when no faces are found."""
        # Mock the face cascade to return no detections
        mock_classifier = Mock()
        mock_classifier.detectMultiScale.return_value = np.array([])
        self.detector.face_cascade = mock_classifier
        
        result = self.detector.detect_faces(self.test_image)
        assert result == []
    
    def test_detect_faces_with_detections(self):
        """Test detect_faces with mock detections."""
        # Mock OpenCV to return detections
        # Format: (x, y, width, height)
        mock_detections = np.array([[64, 96, 192, 192]])
        
        mock_classifier = Mock()
        mock_classifier.detectMultiScale.return_value = mock_detections
        self.detector.face_cascade = mock_classifier
        
        result = self.detector.detect_faces(self.test_image)
        
        assert len(result) == 1
        confidence, bbox = result[0]
        assert confidence == 0.8  # Fixed confidence for Haar cascades
        
        # Check bbox
        expected_x = 64
        expected_y = 96
        expected_w = 192
        expected_h = 192
        
        assert bbox == (expected_x, expected_y, expected_w, expected_h)
    
    def test_crop_face_invalid_input(self):
        """Test crop_face with invalid inputs."""
        bbox = (10, 10, 100, 100)
        
        # Test invalid image
        with pytest.raises(ValueError, match="Image must be a 3D numpy array"):
            self.detector.crop_face(np.zeros((100, 100)), bbox)
        
        # Test invalid bbox
        with pytest.raises(ValueError, match="Bounding box must contain 4 non-negative"):
            self.detector.crop_face(self.test_image, (10, 10, -100, 100))
        
        # Test invalid padding factor
        with pytest.raises(ValueError, match="padding_factor must be between"):
            self.detector.crop_face(self.test_image, bbox, padding_factor=1.5)
    
    def test_crop_face_valid_input(self):
        """Test crop_face with valid inputs."""
        bbox = (100, 100, 200, 200)  # x, y, width, height
        
        cropped = self.detector.crop_face(self.test_image, bbox, padding_factor=0.1)
        
        # Check output shape
        assert cropped.shape == (224, 224, 3)
        assert cropped.dtype == np.uint8
    
    def test_crop_face_edge_cases(self):
        """Test crop_face with edge cases."""
        # Bbox at image edge
        bbox = (0, 0, 50, 50)
        cropped = self.detector.crop_face(self.test_image, bbox)
        assert cropped.shape == (224, 224, 3)
        
        # Bbox extending beyond image
        bbox = (600, 400, 100, 100)  # Extends beyond 640x480 image
        cropped = self.detector.crop_face(self.test_image, bbox)
        assert cropped.shape == (224, 224, 3)
    
    def test_extract_faces_invalid_input(self):
        """Test extract_faces with invalid inputs."""
        with pytest.raises(ValueError, match="max_faces must be positive"):
            self.detector.extract_faces(self.test_image, max_faces=0)
    
    def test_extract_faces_no_detections(self):
        """Test extract_faces when no faces are detected."""
        with patch.object(self.detector, 'detect_faces', return_value=[]):
            result = self.detector.extract_faces(self.test_image)
            assert result == []
    
    def test_extract_faces_with_detections(self):
        """Test extract_faces with mock detections."""
        # Mock detections (sorted by confidence)
        mock_detections = [
            (0.95, (100, 100, 200, 200)),
            (0.85, (300, 300, 150, 150))
        ]
        
        with patch.object(self.detector, 'detect_faces', return_value=mock_detections):
            with patch.object(self.detector, 'crop_face') as mock_crop:
                # Mock cropped faces
                mock_face1 = np.zeros((224, 224, 3), dtype=np.uint8)
                mock_face2 = np.ones((224, 224, 3), dtype=np.uint8) * 255
                mock_crop.side_effect = [mock_face1, mock_face2]
                
                result = self.detector.extract_faces(self.test_image, max_faces=2)
                
                assert len(result) == 2
                
                # Check first face (highest confidence)
                face1, conf1 = result[0]
                assert conf1 == 0.95
                assert face1.shape == (224, 224, 3)
                
                # Check second face
                face2, conf2 = result[1]
                assert conf2 == 0.85
                assert face2.shape == (224, 224, 3)
    
    def test_extract_faces_crop_failure(self):
        """Test extract_faces when cropping fails for some faces."""
        mock_detections = [
            (0.95, (100, 100, 200, 200)),
            (0.85, (300, 300, 150, 150))
        ]
        
        with patch.object(self.detector, 'detect_faces', return_value=mock_detections):
            with patch.object(self.detector, 'crop_face') as mock_crop:
                # First crop succeeds, second fails
                mock_face = np.zeros((224, 224, 3), dtype=np.uint8)
                mock_crop.side_effect = [mock_face, RuntimeError("Crop failed")]
                
                result = self.detector.extract_faces(self.test_image, max_faces=2)
                
                # Should only return the successful crop
                assert len(result) == 1
                face, conf = result[0]
                assert conf == 0.95
    
    def test_extract_faces_max_faces_limit(self):
        """Test extract_faces respects max_faces limit."""
        # Mock 3 detections
        mock_detections = [
            (0.95, (100, 100, 200, 200)),
            (0.85, (300, 300, 150, 150)),
            (0.75, (400, 400, 100, 100))
        ]
        
        with patch.object(self.detector, 'detect_faces', return_value=mock_detections):
            with patch.object(self.detector, 'crop_face') as mock_crop:
                mock_face = np.zeros((224, 224, 3), dtype=np.uint8)
                mock_crop.return_value = mock_face
                
                # Request only 2 faces
                result = self.detector.extract_faces(self.test_image, max_faces=2)
                
                assert len(result) == 2
                # Should get the two highest confidence faces
                assert result[0][1] == 0.95
                assert result[1][1] == 0.85
    
    def test_tensor_shape_assertions(self):
        """Test that tensor shape assertions work correctly."""
        # Test valid image passes assertions
        valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock the cascade to avoid actual detection
        mock_classifier = Mock()
        mock_classifier.detectMultiScale.return_value = np.array([])
        self.detector.face_cascade = mock_classifier
        
        try:
            self.detector.detect_faces(valid_image)
        except Exception as e:
            # Should not raise assertion errors for valid input
            assert "Expected uint8 image" not in str(e)
            assert "Image dimensions must be positive" not in str(e)
    
    def test_cleanup(self):
        """Test proper cleanup of resources."""
        # Call destructor (should not raise any errors)
        self.detector.__del__()


class TestFaceDetectorIntegration:
    """Integration tests for FaceDetector with real OpenCV."""
    
    def test_real_face_detection_synthetic_image(self):
        """Test face detection on a synthetic image (should find no faces)."""
        # This test uses the real OpenCV implementation
        try:
            detector = FaceDetector(min_detection_confidence=0.5)
            
            # Create a simple synthetic image with no faces
            synthetic_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            
            faces = detector.detect_faces(synthetic_image)
            
            # Should detect no faces in random noise (usually)
            assert isinstance(faces, list)
            # Note: We don't assert len(faces) == 0 because Haar cascades might 
            # occasionally detect false positives in random noise
        except RuntimeError as e:
            # If OpenCV model loading fails, skip this test
            pytest.skip(f"OpenCV model loading failed: {e}")
    
    def test_real_face_extraction_synthetic_image(self):
        """Test face extraction on a synthetic image."""
        try:
            detector = FaceDetector(min_detection_confidence=0.5)
            
            synthetic_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            
            extracted_faces = detector.extract_faces(synthetic_image, max_faces=1)
            
            # Should return a list (empty or with false positives)
            assert isinstance(extracted_faces, list)
            
            # If any faces were extracted, verify their format
            for face, confidence in extracted_faces:
                assert face.shape == (224, 224, 3)
                assert 0.0 <= confidence <= 1.0
                assert face.dtype == np.uint8
        except RuntimeError as e:
            # If OpenCV model loading fails, skip this test
            pytest.skip(f"OpenCV model loading failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])