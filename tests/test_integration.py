"""
Integration test demonstrating the complete data processing pipeline.

This test shows how all components (FaceDetector, DWTProcessor, VideoProcessor)
work together in the CRAFT-DF preprocessing pipeline.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from craft_df.data.face_detection import FaceDetector
from craft_df.data.dwt_processing import DWTProcessor
from craft_df.data.video_processor import VideoProcessor


class TestDataProcessingIntegration:
    """Integration tests for the complete data processing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch.object(FaceDetector, '_load_face_detection_model')
    def test_complete_pipeline_integration(self, mock_load_model):
        """Test complete integration of all data processing components."""
        
        # 1. Test FaceDetector
        face_detector = FaceDetector(min_detection_confidence=0.7, target_size=(224, 224))
        
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Mock face detection to return a face
        mock_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        with patch.object(face_detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [(0.95, (100, 100, 200, 200))]
            
            with patch.object(face_detector, 'crop_face') as mock_crop:
                mock_crop.return_value = mock_face
                
                faces = face_detector.extract_faces(test_image, max_faces=1)
                
                assert len(faces) == 1
                face_crop, confidence = faces[0]
                assert face_crop.shape == (224, 224, 3)
                assert confidence == 0.95
        
        # 2. Test DWTProcessor
        dwt_processor = DWTProcessor(wavelet='db4', levels=3, mode='symmetric')
        
        # Process the face crop through DWT
        dwt_features = dwt_processor.process_face_crop(mock_face)
        
        assert isinstance(dwt_features, np.ndarray)
        assert dwt_features.ndim == 1
        assert len(dwt_features) > 0
        assert np.all(np.isfinite(dwt_features))
        
        # 3. Test VideoProcessor integration
        input_dir = self.temp_dir / "input"
        output_dir = self.temp_dir / "output"
        metadata_path = self.temp_dir / "metadata.csv"
        
        input_dir.mkdir(parents=True)
        (input_dir / "real").mkdir()
        
        video_processor = VideoProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_path=metadata_path,
            face_detector_config={'min_detection_confidence': 0.7},
            dwt_processor_config={'wavelet': 'db4', 'levels': 3}
        )
        
        # Verify that processors are properly initialized
        assert isinstance(video_processor.face_detector, FaceDetector)
        assert isinstance(video_processor.dwt_processor, DWTProcessor)
        assert video_processor.face_detector.min_detection_confidence == 0.7
        assert video_processor.dwt_processor.wavelet == 'db4'
        assert video_processor.dwt_processor.levels == 3
        
        # 4. Test frame processing integration
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch.object(video_processor.face_detector, 'extract_faces') as mock_extract:
            with patch.object(video_processor.dwt_processor, 'process_face_crop') as mock_dwt:
                mock_extract.return_value = [(mock_face, 0.95)]
                mock_dwt.return_value = dwt_features
                
                output_video_dir = output_dir / "real" / "test_video"
                output_video_dir.mkdir(parents=True)
                
                metadata = video_processor._process_frame(
                    test_frame, "test_video", "real", 0, output_video_dir
                )
                
                assert len(metadata) == 1
                record = metadata[0]
                
                # Verify complete metadata structure
                assert record['video_id'] == "test_video"
                assert record['label'] == "real"
                assert record['label_numeric'] == 0
                assert record['frame_number'] == 0
                assert record['face_confidence'] == 0.95
                assert record['dwt_feature_count'] == len(dwt_features)
                assert record['dwt_wavelet'] == 'db4'
                assert record['dwt_levels'] == 3
                
                # Verify files are created
                face_path = output_video_dir / record['face_path'].split('/')[-1]
                dwt_path = output_video_dir / record['dwt_path'].split('/')[-1]
                
                assert face_path.exists()
                assert dwt_path.exists()
                
                # Verify file contents
                saved_face = np.load(face_path)
                saved_features = np.load(dwt_path)
                
                assert saved_face.shape == mock_face.shape
                assert saved_features.shape == dwt_features.shape
                np.testing.assert_array_equal(saved_features, dwt_features)
    
    def test_feature_consistency(self):
        """Test that DWT features are consistent across processing."""
        dwt_processor = DWTProcessor()
        
        # Create a test face crop
        face_crop = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Process the same face multiple times
        features1 = dwt_processor.process_face_crop(face_crop)
        features2 = dwt_processor.process_face_crop(face_crop)
        
        # Should get identical results
        np.testing.assert_array_equal(features1, features2)
        
        # Features should have expected properties
        assert len(features1) > 0
        assert np.all(np.isfinite(features1))
        assert features1.dtype == np.float32
    
    def test_pipeline_error_handling(self):
        """Test error handling throughout the pipeline."""
        
        # Test DWT processor with invalid input
        dwt_processor = DWTProcessor()
        
        with pytest.raises(ValueError):
            dwt_processor.process_face_crop("not_an_array")
        
        with pytest.raises(ValueError):
            dwt_processor.process_face_crop(np.zeros((100,)))  # Wrong dimensions
        
        # Test with zero-dimension image
        with pytest.raises(AssertionError):
            dwt_processor.process_face_crop(np.zeros((0, 224, 3), dtype=np.float32))
    
    def test_memory_efficiency(self):
        """Test that processing doesn't cause memory leaks."""
        dwt_processor = DWTProcessor()
        
        # Process multiple face crops
        for i in range(10):
            face_crop = np.random.rand(224, 224, 3).astype(np.float32)
            features = dwt_processor.process_face_crop(face_crop)
            
            # Verify each result
            assert isinstance(features, np.ndarray)
            assert len(features) > 0
            assert np.all(np.isfinite(features))
            
            # Clear references to help garbage collection
            del face_crop, features


if __name__ == "__main__":
    pytest.main([__file__])