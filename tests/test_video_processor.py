"""
Integration tests for video processing pipeline.

This module contains comprehensive tests for the VideoProcessor class,
including end-to-end pipeline tests, metadata validation, and error handling.
"""

import pytest
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import json

from craft_df.data.video_processor import VideoProcessor
from craft_df.data.face_detection import FaceDetector
from craft_df.data.dwt_processing import DWTProcessor


class TestVideoProcessorInit:
    """Test suite for VideoProcessor initialization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.metadata_path = self.temp_dir / "metadata.csv"
        
        # Create input directory structure
        self.input_dir.mkdir(parents=True)
        (self.input_dir / "real").mkdir()
        (self.input_dir / "fake").mkdir()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init_valid_parameters(self):
        """Test VideoProcessor initialization with valid parameters."""
        processor = VideoProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            metadata_path=self.metadata_path,
            max_faces_per_frame=2,
            frame_skip=2,
            max_frames_per_video=100
        )
        
        assert processor.input_dir == self.input_dir
        assert processor.output_dir == self.output_dir
        assert processor.metadata_path == self.metadata_path
        assert processor.max_faces_per_frame == 2
        assert processor.frame_skip == 2
        assert processor.max_frames_per_video == 100
        
        # Output directory should be created
        assert processor.output_dir.exists()
        
        # Processors should be initialized
        assert isinstance(processor.face_detector, FaceDetector)
        assert isinstance(processor.dwt_processor, DWTProcessor)
    
    def test_init_with_configs(self):
        """Test initialization with custom processor configurations."""
        face_config = {'min_detection_confidence': 0.8, 'target_size': (256, 256)}
        dwt_config = {'wavelet': 'db8', 'levels': 2}
        
        processor = VideoProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            metadata_path=self.metadata_path,
            face_detector_config=face_config,
            dwt_processor_config=dwt_config
        )
        
        assert processor.face_detector.min_detection_confidence == 0.8
        assert processor.face_detector.target_size == (256, 256)
        assert processor.dwt_processor.wavelet == 'db8'
        assert processor.dwt_processor.levels == 2
    
    def test_init_invalid_input_dir(self):
        """Test initialization with non-existent input directory."""
        with pytest.raises(ValueError, match="Input directory does not exist"):
            VideoProcessor(
                input_dir=self.temp_dir / "nonexistent",
                output_dir=self.output_dir,
                metadata_path=self.metadata_path
            )
    
    def test_init_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Invalid max_faces_per_frame
        with pytest.raises(ValueError, match="max_faces_per_frame must be at least 1"):
            VideoProcessor(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                metadata_path=self.metadata_path,
                max_faces_per_frame=0
            )
        
        # Invalid frame_skip
        with pytest.raises(ValueError, match="frame_skip must be at least 1"):
            VideoProcessor(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                metadata_path=self.metadata_path,
                frame_skip=0
            )
        
        # Invalid max_frames_per_video
        with pytest.raises(ValueError, match="max_frames_per_video must be at least 1"):
            VideoProcessor(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                metadata_path=self.metadata_path,
                max_frames_per_video=0
            )


class TestVideoProcessorUtilities:
    """Test suite for VideoProcessor utility methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.metadata_path = self.temp_dir / "metadata.csv"
        
        # Create input directory structure with test files
        self.input_dir.mkdir(parents=True)
        (self.input_dir / "real").mkdir()
        (self.input_dir / "fake").mkdir()
        
        # Create mock video files
        (self.input_dir / "real" / "video1.mp4").touch()
        (self.input_dir / "real" / "video2.avi").touch()
        (self.input_dir / "fake" / "video3.mp4").touch()
        (self.input_dir / "test.txt").touch()  # Non-video file
        
        self.processor = VideoProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            metadata_path=self.metadata_path
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_get_video_files(self):
        """Test video file discovery."""
        video_files = self.processor.get_video_files()
        
        # Should find 3 video files
        assert len(video_files) == 3
        
        # Should not include text file
        video_names = [f.name for f in video_files]
        assert "test.txt" not in video_names
        assert "video1.mp4" in video_names
        assert "video2.avi" in video_names
        assert "video3.mp4" in video_names
    
    def test_get_video_files_custom_extensions(self):
        """Test video file discovery with custom extensions."""
        video_files = self.processor.get_video_files(extensions=['.mp4'])
        
        # Should find only MP4 files
        assert len(video_files) == 2
        video_names = [f.name for f in video_files]
        assert "video1.mp4" in video_names
        assert "video3.mp4" in video_names
        assert "video2.avi" not in video_names
    
    def test_extract_label_from_path(self):
        """Test label extraction from file paths."""
        real_path = self.input_dir / "real" / "video1.mp4"
        fake_path = self.input_dir / "fake" / "video3.mp4"
        
        assert self.processor.extract_label_from_path(real_path) == "real"
        assert self.processor.extract_label_from_path(fake_path) == "fake"
    
    def test_extract_label_from_path_invalid(self):
        """Test label extraction with invalid paths."""
        invalid_path = self.input_dir / "unknown" / "video.mp4"
        
        with pytest.raises(ValueError, match="Cannot determine label from path"):
            self.processor.extract_label_from_path(invalid_path)
    
    def test_get_processing_summary(self):
        """Test processing summary generation."""
        summary = self.processor.get_processing_summary()
        
        assert 'configuration' in summary
        assert 'statistics' in summary
        
        config = summary['configuration']
        assert config['input_dir'] == str(self.input_dir)
        assert config['output_dir'] == str(self.output_dir)
        assert config['max_faces_per_frame'] == 1
        
        stats = summary['statistics']
        assert 'videos_processed' in stats
        assert 'frames_processed' in stats


class TestVideoProcessorFrameProcessing:
    """Test suite for frame processing methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.metadata_path = self.temp_dir / "metadata.csv"
        
        self.input_dir.mkdir(parents=True)
        
        # Create processor with mocked components
        with patch.object(FaceDetector, '_load_face_detection_model'):
            self.processor = VideoProcessor(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                metadata_path=self.metadata_path
            )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_process_frame_valid_input(self):
        """Test frame processing with valid input."""
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Mock face detection and DWT processing
        mock_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        mock_features = np.random.randn(89).astype(np.float32)
        
        with patch.object(self.processor.face_detector, 'extract_faces') as mock_extract:
            with patch.object(self.processor.dwt_processor, 'process_face_crop') as mock_dwt:
                mock_extract.return_value = [(mock_face, 0.95)]
                mock_dwt.return_value = mock_features
                
                # Create output directory
                output_dir = self.output_dir / "real" / "test_video"
                output_dir.mkdir(parents=True)
                
                metadata = self.processor._process_frame(
                    frame, "test_video", "real", 0, output_dir
                )
                
                assert len(metadata) == 1
                record = metadata[0]
                
                assert record['video_id'] == "test_video"
                assert record['label'] == "real"
                assert record['label_numeric'] == 0
                assert record['frame_number'] == 0
                assert record['face_index'] == 0
                assert record['face_confidence'] == 0.95
                assert record['dwt_feature_count'] == len(mock_features)
    
    def test_process_frame_no_faces(self):
        """Test frame processing when no faces are detected."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch.object(self.processor.face_detector, 'extract_faces') as mock_extract:
            mock_extract.return_value = []  # No faces detected
            
            output_dir = self.output_dir / "real" / "test_video"
            output_dir.mkdir(parents=True)
            
            metadata = self.processor._process_frame(
                frame, "test_video", "real", 0, output_dir
            )
            
            assert len(metadata) == 0
    
    def test_process_frame_multiple_faces(self):
        """Test frame processing with multiple faces."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Mock multiple faces
        mock_face1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        mock_face2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        mock_features = np.random.randn(89).astype(np.float32)
        
        with patch.object(self.processor.face_detector, 'extract_faces') as mock_extract:
            with patch.object(self.processor.dwt_processor, 'process_face_crop') as mock_dwt:
                mock_extract.return_value = [(mock_face1, 0.95), (mock_face2, 0.85)]
                mock_dwt.return_value = mock_features
                
                output_dir = self.output_dir / "real" / "test_video"
                output_dir.mkdir(parents=True)
                
                metadata = self.processor._process_frame(
                    frame, "test_video", "real", 0, output_dir
                )
                
                assert len(metadata) == 2
                assert metadata[0]['face_index'] == 0
                assert metadata[1]['face_index'] == 1
                assert metadata[0]['face_confidence'] == 0.95
                assert metadata[1]['face_confidence'] == 0.85
    
    def test_process_frame_invalid_input(self):
        """Test frame processing with invalid input."""
        # Test with None frame
        output_dir = self.output_dir / "real" / "test_video"
        output_dir.mkdir(parents=True)
        
        metadata = self.processor._process_frame(
            None, "test_video", "real", 0, output_dir
        )
        
        assert len(metadata) == 0
        
        # Test with empty frame
        empty_frame = np.array([])
        metadata = self.processor._process_frame(
            empty_frame, "test_video", "real", 0, output_dir
        )
        
        assert len(metadata) == 0


class TestVideoProcessorMetadata:
    """Test suite for metadata validation and generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.metadata_path = self.temp_dir / "metadata.csv"
        
        self.input_dir.mkdir(parents=True)
        
        with patch.object(FaceDetector, '_load_face_detection_model'):
            self.processor = VideoProcessor(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                metadata_path=self.metadata_path
            )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_validate_metadata_schema_valid(self):
        """Test metadata schema validation with valid data."""
        # Create valid metadata
        metadata_data = [
            {
                'video_id': 'video1',
                'label': 'real',
                'label_numeric': 0,
                'frame_number': 0,
                'face_index': 0,
                'face_confidence': 0.95,
                'face_path': 'real/video1/frame_000000_face_00.npy',
                'dwt_path': 'real/video1/frame_000000_face_00_dwt.npy',
                'dwt_feature_count': 89,
                'processing_timestamp': '2024-01-01T12:00:00',
                'face_shape': '224x224x3',
                'dwt_wavelet': 'db4',
                'dwt_levels': 3,
                'dwt_mode': 'symmetric'
            }
        ]
        
        df = pd.DataFrame(metadata_data)
        validated_df = self.processor._validate_metadata_schema(df)
        
        assert len(validated_df) == 1
        assert validated_df['label_numeric'].dtype == int
        assert validated_df['frame_number'].dtype == int
        assert validated_df['face_confidence'].dtype == float
    
    def test_validate_metadata_schema_missing_columns(self):
        """Test metadata schema validation with missing columns."""
        # Create incomplete metadata
        metadata_data = [
            {
                'video_id': 'video1',
                'label': 'real',
                # Missing required columns
            }
        ]
        
        df = pd.DataFrame(metadata_data)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.processor._validate_metadata_schema(df)
    
    def test_validate_metadata_schema_invalid_ranges(self):
        """Test metadata schema validation with invalid value ranges."""
        # Create metadata with invalid confidence
        metadata_data = [
            {
                'video_id': 'video1',
                'label': 'real',
                'label_numeric': 0,
                'frame_number': 0,
                'face_index': 0,
                'face_confidence': 1.5,  # Invalid: > 1.0
                'face_path': 'real/video1/frame_000000_face_00.npy',
                'dwt_path': 'real/video1/frame_000000_face_00_dwt.npy',
                'dwt_feature_count': 89,
                'processing_timestamp': '2024-01-01T12:00:00'
            }
        ]
        
        df = pd.DataFrame(metadata_data)
        
        with pytest.raises(ValueError, match="face_confidence must be between 0 and 1"):
            self.processor._validate_metadata_schema(df)
    
    def test_generate_metadata_csv(self):
        """Test metadata CSV generation."""
        # Create test metadata
        metadata_data = [
            {
                'video_id': 'video1',
                'label': 'real',
                'label_numeric': 0,
                'frame_number': 0,
                'face_index': 0,
                'face_confidence': 0.95,
                'face_path': 'real/video1/frame_000000_face_00.npy',
                'dwt_path': 'real/video1/frame_000000_face_00_dwt.npy',
                'dwt_feature_count': 89,
                'processing_timestamp': '2024-01-01T12:00:00'
            }
        ]
        
        df = pd.DataFrame(metadata_data)
        self.processor.generate_metadata_csv(df)
        
        # Check that files were created
        assert self.metadata_path.exists()
        assert self.metadata_path.with_suffix('.stats.json').exists()
        
        # Verify CSV content
        loaded_df = pd.read_csv(self.metadata_path)
        assert len(loaded_df) == 1
        assert loaded_df['video_id'].iloc[0] == 'video1'
        
        # Verify stats content
        with open(self.metadata_path.with_suffix('.stats.json')) as f:
            stats = json.load(f)
        
        assert stats['total_records'] == 1
        assert stats['unique_videos'] == 1
        assert stats['real_samples'] == 1
        assert stats['fake_samples'] == 0


class TestVideoProcessorIntegration:
    """Integration tests for complete video processing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.metadata_path = self.temp_dir / "metadata.csv"
        
        # Create directory structure
        self.input_dir.mkdir(parents=True)
        (self.input_dir / "real").mkdir()
        (self.input_dir / "fake").mkdir()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_video(self, path: Path, num_frames: int = 5) -> None:
        """Create a test video file."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, 1.0, (640, 480))
        
        for i in range(num_frames):
            # Create a simple test frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
    
    @patch.object(FaceDetector, '_load_face_detection_model')
    def test_process_video_complete_pipeline(self, mock_load_model):
        """Test complete video processing pipeline."""
        # Create test video
        video_path = self.input_dir / "real" / "test_video.mp4"
        self.create_test_video(video_path, num_frames=3)
        
        processor = VideoProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            metadata_path=self.metadata_path,
            frame_skip=1  # Process all frames
        )
        
        # Mock face detection and DWT processing
        mock_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        mock_features = np.random.randn(89).astype(np.float32)
        
        with patch.object(processor.face_detector, 'extract_faces') as mock_extract:
            with patch.object(processor.dwt_processor, 'process_face_crop') as mock_dwt:
                mock_extract.return_value = [(mock_face, 0.95)]
                mock_dwt.return_value = mock_features
                
                metadata = processor.process_video(video_path)
                
                # Should have processed 3 frames with 1 face each
                assert len(metadata) == 3
                
                # Check metadata structure
                for i, record in enumerate(metadata):
                    assert record['video_id'] == 'test_video'
                    assert record['label'] == 'real'
                    assert record['frame_number'] == i
                    assert record['face_confidence'] == 0.95
                
                # Check that files were created
                output_video_dir = processor.output_dir / "real" / "test_video"
                assert output_video_dir.exists()
                
                # Should have face and DWT files for each frame
                face_files = list(output_video_dir.glob("*_face_*.npy"))
                dwt_files = list(output_video_dir.glob("*_dwt.npy"))
                
                # Filter out DWT files from face files (since glob pattern overlaps)
                face_only_files = [f for f in face_files if '_dwt.npy' not in str(f)]
                
                assert len(face_only_files) == 3
                assert len(dwt_files) == 3
    
    @patch.object(FaceDetector, '_load_face_detection_model')
    def test_process_video_batch_complete(self, mock_load_model):
        """Test complete batch processing pipeline."""
        # Create multiple test videos
        real_video = self.input_dir / "real" / "real_video.mp4"
        fake_video = self.input_dir / "fake" / "fake_video.mp4"
        
        self.create_test_video(real_video, num_frames=2)
        self.create_test_video(fake_video, num_frames=2)
        
        processor = VideoProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            metadata_path=self.metadata_path
        )
        
        # Mock processing components
        mock_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        mock_features = np.random.randn(89).astype(np.float32)
        
        with patch.object(processor.face_detector, 'extract_faces') as mock_extract:
            with patch.object(processor.dwt_processor, 'process_face_crop') as mock_dwt:
                mock_extract.return_value = [(mock_face, 0.95)]
                mock_dwt.return_value = mock_features
                
                # Process batch
                metadata_df = processor.process_video_batch(progress_bar=False)
                
                # Should have processed both videos
                assert len(metadata_df) == 4  # 2 frames per video
                assert metadata_df['video_id'].nunique() == 2
                assert (metadata_df['label'] == 'real').sum() == 2
                assert (metadata_df['label'] == 'fake').sum() == 2
                
                # Generate final metadata
                processor.generate_metadata_csv(metadata_df)
                
                # Verify final outputs
                assert processor.metadata_path.exists()
                assert processor.metadata_path.with_suffix('.stats.json').exists()
                
                # Check processing statistics
                assert processor.stats['videos_processed'] == 2
                assert processor.stats['frames_processed'] == 4
                assert processor.stats['faces_extracted'] == 4
    
    @patch.object(FaceDetector, '_load_face_detection_model')
    def test_error_handling_invalid_video(self, mock_load_model):
        """Test error handling with invalid video files."""
        # Create invalid video file (empty file)
        invalid_video = self.input_dir / "real" / "invalid.mp4"
        invalid_video.touch()
        
        processor = VideoProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            metadata_path=self.metadata_path
        )
        
        # Should handle invalid video gracefully
        with pytest.raises(RuntimeError, match="Video processing failed"):
            processor.process_video(invalid_video)
        
        # Error should be recorded in stats
        assert processor.stats['errors'] > 0


if __name__ == "__main__":
    pytest.main([__file__])