"""
Video processing pipeline for deepfake detection preprocessing.

This module provides a comprehensive video processing pipeline that integrates
face detection, cropping, and DWT feature extraction for deepfake detection.
It handles batch processing of video files and generates hierarchical data
organization with metadata management.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Any
import logging
from datetime import datetime
import json
import os
from tqdm import tqdm

from .face_detection import FaceDetector
from .dwt_processing import DWTProcessor

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Comprehensive video processing pipeline for deepfake detection preprocessing.
    
    This class integrates face detection, cropping, and DWT feature extraction
    into a unified pipeline for processing video datasets. It provides:
    
    1. Batch video processing with progress tracking
    2. Hierarchical file organization (real/fake/video_id/frame_xxx.npy)
    3. Metadata CSV generation with comprehensive schema validation
    4. Error handling and recovery mechanisms
    5. Memory-efficient processing for large datasets
    
    The pipeline is designed to handle massive video datasets while maintaining
    data integrity and providing detailed processing logs.
    
    Attributes:
        input_dir (Path): Directory containing input videos
        output_dir (Path): Directory for processed outputs
        metadata_path (Path): Path to metadata CSV file
        face_detector (FaceDetector): Face detection and cropping instance
        dwt_processor (DWTProcessor): DWT feature extraction instance
    """
    
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        metadata_path: Union[str, Path],
        face_detector_config: Optional[Dict[str, Any]] = None,
        dwt_processor_config: Optional[Dict[str, Any]] = None,
        max_faces_per_frame: int = 1,
        frame_skip: int = 1,
        max_frames_per_video: Optional[int] = None
    ):
        """
        Initialize the video processor.
        
        Args:
            input_dir: Directory containing input video files
            output_dir: Directory for saving processed outputs
            metadata_path: Path for metadata CSV file
            face_detector_config: Configuration for FaceDetector initialization
            dwt_processor_config: Configuration for DWTProcessor initialization
            max_faces_per_frame: Maximum number of faces to extract per frame
            frame_skip: Process every Nth frame (1 = process all frames)
            max_frames_per_video: Maximum frames to process per video (None = all)
        
        Raises:
            ValueError: If directories don't exist or configuration is invalid
        """
        # Validate and set paths
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metadata_path = Path(metadata_path)
        
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate processing parameters
        if max_faces_per_frame < 1:
            raise ValueError("max_faces_per_frame must be at least 1")
        
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")
        
        if max_frames_per_video is not None and max_frames_per_video < 1:
            raise ValueError("max_frames_per_video must be at least 1 or None")
        
        self.max_faces_per_frame = max_faces_per_frame
        self.frame_skip = frame_skip
        self.max_frames_per_video = max_frames_per_video
        
        # Initialize processors
        face_config = face_detector_config or {}
        dwt_config = dwt_processor_config or {}
        
        self.face_detector = FaceDetector(**face_config)
        self.dwt_processor = DWTProcessor(**dwt_config)
        
        # Processing statistics
        self.stats = {
            'videos_processed': 0,
            'frames_processed': 0,
            'faces_extracted': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(f"VideoProcessor initialized: input={self.input_dir}, "
                   f"output={self.output_dir}, max_faces={max_faces_per_frame}, "
                   f"frame_skip={frame_skip}")
    
    def get_video_files(self, extensions: List[str] = None) -> List[Path]:
        """
        Get list of video files from input directory.
        
        Args:
            extensions: List of video file extensions to include
            
        Returns:
            List of video file paths
        """
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
        video_files = []
        for ext in extensions:
            video_files.extend(self.input_dir.glob(f"**/*{ext}"))
            video_files.extend(self.input_dir.glob(f"**/*{ext.upper()}"))
        
        logger.info(f"Found {len(video_files)} video files")
        return sorted(video_files)

    def get_image_files(self, extensions: List[str] = None) -> List[Path]:
        """
        Get list of image files from input directory.

        Args:
            extensions: List of image file extensions to include

        Returns:
            List of image file paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

        image_files = []
        for ext in extensions:
            image_files.extend(self.input_dir.glob(f"**/*{ext}"))
            image_files.extend(self.input_dir.glob(f"**/*{ext.upper()}"))

        logger.info(f"Found {len(image_files)} image files")
        return sorted(image_files)

    def process_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single image file — same pipeline as one video frame.

        Args:
            image_path: Path to image file

        Returns:
            List of metadata dictionaries for extracted faces
        """
        try:
            label    = self.extract_label_from_path(image_path)
            video_id = image_path.stem          # treat each image as its own "video"

            label_dir = self.output_dir / label / video_id
            label_dir.mkdir(parents=True, exist_ok=True)

            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Cannot read image: {image_path}")

            metadata_records = self._process_frame(
                frame, video_id, label, frame_number=0, output_dir=label_dir
            )

            self.stats['videos_processed'] += 1
            self.stats['frames_processed'] += 1
            return metadata_records

        except Exception as e:
            logger.error(f"Image processing failed for {image_path}: {str(e)}")
            self.stats['errors'] += 1
            raise RuntimeError(f"Image processing failed: {str(e)}")
    
    def extract_label_from_path(self, video_path: Path) -> str:
        """
        Extract label (real/fake) from video file path.
        
        This method uses directory structure to determine labels.
        Expected structure: .../real/... or .../fake/...
        
        Args:
            video_path: Path to video file
            
        Returns:
            Label string ('real' or 'fake')
            
        Raises:
            ValueError: If label cannot be determined from path
        """
        path_parts = video_path.parts
        
        # Look for 'real' or 'fake' in path components
        for part in path_parts:
            part_lower = part.lower()
            if 'real' in part_lower and 'fake' not in part_lower:
                return 'real'
            elif 'fake' in part_lower:
                return 'fake'
        
        # Fallback: check parent directory names
        parent_name = video_path.parent.name.lower()
        if 'real' in parent_name and 'fake' not in parent_name:
            return 'real'
        elif 'fake' in parent_name:
            return 'fake'
        
        raise ValueError(f"Cannot determine label from path: {video_path}")
    
    def process_video(self, video_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single video file and extract face crops with DWT features.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of metadata dictionaries for processed frames
            
        Raises:
            RuntimeError: If video processing fails
        """
        try:
            # Extract label and video ID
            label = self.extract_label_from_path(video_path)
            video_id = video_path.stem
            
            # Create output directory structure
            label_dir = self.output_dir / label / video_id
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Processing video: {video_path.name} "
                       f"({total_frames} frames, {fps:.2f} fps)")
            
            metadata_records = []
            frame_count = 0
            processed_frames = 0
            
            # Determine frames to process
            max_frames = self.max_frames_per_video or total_frames
            frames_to_process = min(max_frames, total_frames)
            
            while cap.isOpened() and processed_frames < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on frame_skip parameter
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                try:
                    # Process frame
                    frame_metadata = self._process_frame(
                        frame, video_id, label, frame_count, label_dir
                    )
                    metadata_records.extend(frame_metadata)
                    
                    processed_frames += 1
                    self.stats['frames_processed'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process frame {frame_count} "
                                 f"in {video_path.name}: {str(e)}")
                    self.stats['errors'] += 1
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Completed video {video_path.name}: "
                       f"{len(metadata_records)} faces extracted from "
                       f"{processed_frames} frames")
            
            self.stats['videos_processed'] += 1
            return metadata_records
            
        except Exception as e:
            logger.error(f"Video processing failed for {video_path}: {str(e)}")
            self.stats['errors'] += 1
            raise RuntimeError(f"Video processing failed: {str(e)}")
    
    def _process_frame(
        self, 
        frame: np.ndarray, 
        video_id: str, 
        label: str, 
        frame_number: int,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Process a single frame to extract faces and compute DWT features.
        
        Args:
            frame: Video frame as numpy array
            video_id: Video identifier
            label: Video label ('real' or 'fake')
            frame_number: Frame number in video
            output_dir: Directory to save processed data
            
        Returns:
            List of metadata dictionaries for extracted faces
        """
        metadata_records = []
        
        # Validate frame
        if frame is None or frame.size == 0:
            logger.warning(f"Invalid frame {frame_number} in video {video_id}")
            return metadata_records
        
        assert frame.dtype == np.uint8, f"Expected uint8 frame, got {frame.dtype}"
        assert len(frame.shape) == 3, f"Expected 3D frame, got shape {frame.shape}"
        
        try:
            # Extract faces from frame
            faces = self.face_detector.extract_faces(
                frame, max_faces=self.max_faces_per_frame
            )
            
            for face_idx, (face_crop, confidence) in enumerate(faces):
                try:
                    # Validate face crop
                    assert face_crop.shape == (224, 224, 3), \
                        f"Expected face shape (224, 224, 3), got {face_crop.shape}"
                    assert face_crop.dtype == np.uint8, \
                        f"Expected uint8 face crop, got {face_crop.dtype}"
                    
                    # Generate file paths
                    face_filename = f"frame_{frame_number:06d}_face_{face_idx:02d}.npy"
                    dwt_filename = f"frame_{frame_number:06d}_face_{face_idx:02d}_dwt.npy"
                    
                    face_path = output_dir / face_filename
                    dwt_path = output_dir / dwt_filename
                    
                    # Save face crop
                    np.save(face_path, face_crop)
                    
                    # Compute and save DWT features
                    dwt_features = self.dwt_processor.process_face_crop(face_crop)
                    np.save(dwt_path, dwt_features)
                    
                    # Create metadata record
                    metadata_record = {
                        'video_id': video_id,
                        'label': label,
                        'label_numeric': 1 if label == 'fake' else 0,
                        'frame_number': frame_number,
                        'face_index': face_idx,
                        'face_confidence': float(confidence),
                        'face_path': str(face_path.relative_to(self.output_dir)),
                        'dwt_path': str(dwt_path.relative_to(self.output_dir)),
                        'dwt_feature_count': len(dwt_features),
                        'processing_timestamp': datetime.now().isoformat(),
                        'face_shape': f"{face_crop.shape[0]}x{face_crop.shape[1]}x{face_crop.shape[2]}",
                        'dwt_wavelet': self.dwt_processor.wavelet,
                        'dwt_levels': self.dwt_processor.levels,
                        'dwt_mode': self.dwt_processor.mode
                    }
                    
                    metadata_records.append(metadata_record)
                    self.stats['faces_extracted'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process face {face_idx} "
                                 f"in frame {frame_number}: {str(e)}")
                    self.stats['errors'] += 1
            
        except Exception as e:
            logger.warning(f"Face extraction failed for frame {frame_number}: {str(e)}")
            self.stats['errors'] += 1
        
        return metadata_records
    
    def process_video_batch(
        self, 
        video_paths: Optional[List[Path]] = None,
        save_intermediate: bool = True,
        progress_bar: bool = True
    ) -> pd.DataFrame:
        """
        Process a batch of videos with progress tracking and error handling.
        
        Args:
            video_paths: List of video paths to process (None = all videos in input_dir)
            save_intermediate: Save metadata after each video for recovery
            progress_bar: Show progress bar during processing
            
        Returns:
            DataFrame containing all metadata records
            
        Raises:
            RuntimeError: If batch processing fails
        """
        try:
            self.stats['start_time'] = datetime.now()
            
            # Get video files to process
            if video_paths is None:
                video_paths = self.get_video_files()

            # If no videos found, fall back to images
            if not video_paths:
                logger.info("No video files found, looking for images instead...")
                video_paths = self.get_image_files()
            
            if not video_paths:
                logger.warning("No video or image files found to process")
                return pd.DataFrame()
            
            logger.info(f"Starting batch processing of {len(video_paths)} files")
            
            all_metadata = []
            
            # Process videos with optional progress bar
            iterator = tqdm(video_paths, desc="Processing files") if progress_bar else video_paths
            
            IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

            for video_path in iterator:
                try:
                    if video_path.suffix.lower() in IMAGE_EXTS:
                        file_metadata = self.process_image(video_path)
                    else:
                        file_metadata = self.process_video(video_path)
                    all_metadata.extend(file_metadata)
                    
                    # Save intermediate results for recovery
                    if save_intermediate and all_metadata:
                        self._save_intermediate_metadata(all_metadata)
                    
                except Exception as e:
                    logger.error(f"Failed to process video {video_path}: {str(e)}")
                    self.stats['errors'] += 1
                    continue
            
            # Create final metadata DataFrame
            if all_metadata:
                metadata_df = pd.DataFrame(all_metadata)
                metadata_df = self._validate_metadata_schema(metadata_df)
            else:
                metadata_df = pd.DataFrame()
                logger.warning("No metadata records generated")
            
            self.stats['end_time'] = datetime.now()
            
            # Log final statistics
            self._log_processing_stats()
            
            return metadata_df
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise RuntimeError(f"Batch processing failed: {str(e)}")
    
    def _save_intermediate_metadata(self, metadata_records: List[Dict[str, Any]]) -> None:
        """Save intermediate metadata for recovery purposes."""
        try:
            temp_path = self.metadata_path.with_suffix('.tmp.csv')
            temp_df = pd.DataFrame(metadata_records)
            temp_df.to_csv(temp_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to save intermediate metadata: {str(e)}")
    
    def _validate_metadata_schema(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean metadata DataFrame schema.
        
        Args:
            metadata_df: Raw metadata DataFrame
            
        Returns:
            Validated and cleaned DataFrame
        """
        required_columns = [
            'video_id', 'label', 'label_numeric', 'frame_number', 'face_index',
            'face_confidence', 'face_path', 'dwt_path', 'dwt_feature_count',
            'processing_timestamp'
        ]
        
        # Check required columns
        missing_columns = set(required_columns) - set(metadata_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types and ranges
        try:
            metadata_df['label_numeric'] = metadata_df['label_numeric'].astype(int)
            metadata_df['frame_number'] = metadata_df['frame_number'].astype(int)
            metadata_df['face_index'] = metadata_df['face_index'].astype(int)
            metadata_df['face_confidence'] = metadata_df['face_confidence'].astype(float)
            metadata_df['dwt_feature_count'] = metadata_df['dwt_feature_count'].astype(int)
            
            # Validate ranges
            assert metadata_df['label_numeric'].isin([0, 1]).all(), \
                "label_numeric must be 0 or 1"
            assert (metadata_df['face_confidence'] >= 0).all() and \
                   (metadata_df['face_confidence'] <= 1).all(), \
                "face_confidence must be between 0 and 1"
            assert (metadata_df['frame_number'] >= 0).all(), \
                "frame_number must be non-negative"
            assert (metadata_df['face_index'] >= 0).all(), \
                "face_index must be non-negative"
            assert (metadata_df['dwt_feature_count'] > 0).all(), \
                "dwt_feature_count must be positive"
            
        except Exception as e:
            logger.error(f"Metadata validation failed: {str(e)}")
            raise ValueError(f"Metadata validation failed: {str(e)}")
        
        logger.info(f"Metadata validation passed: {len(metadata_df)} records")
        return metadata_df
    
    def generate_metadata_csv(self, metadata_df: pd.DataFrame) -> None:
        """
        Generate and save the final metadata CSV file.
        
        Args:
            metadata_df: Validated metadata DataFrame
        """
        try:
            # Add summary statistics
            summary_stats = {
                'total_records': int(len(metadata_df)),
                'unique_videos': int(metadata_df['video_id'].nunique()),
                'real_samples': int((metadata_df['label'] == 'real').sum()),
                'fake_samples': int((metadata_df['label'] == 'fake').sum()),
                'avg_face_confidence': float(metadata_df['face_confidence'].mean()),
                'processing_stats': self.stats
            }
            
            # Save metadata CSV
            metadata_df.to_csv(self.metadata_path, index=False)
            
            # Save summary statistics
            stats_path = self.metadata_path.with_suffix('.stats.json')
            with open(stats_path, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            logger.info(f"Metadata saved: {self.metadata_path}")
            logger.info(f"Summary stats saved: {stats_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            raise RuntimeError(f"Failed to save metadata: {str(e)}")
    
    def _log_processing_stats(self) -> None:
        """Log comprehensive processing statistics."""
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            
            logger.info("=== Processing Statistics ===")
            logger.info(f"Videos processed: {self.stats['videos_processed']}")
            logger.info(f"Frames processed: {self.stats['frames_processed']}")
            logger.info(f"Faces extracted: {self.stats['faces_extracted']}")
            logger.info(f"Errors encountered: {self.stats['errors']}")
            logger.info(f"Processing time: {duration}")
            
            if self.stats['frames_processed'] > 0:
                fps = self.stats['frames_processed'] / duration.total_seconds()
                logger.info(f"Processing speed: {fps:.2f} frames/second")
            
            if self.stats['faces_extracted'] > 0:
                faces_per_frame = self.stats['faces_extracted'] / self.stats['frames_processed']
                logger.info(f"Average faces per frame: {faces_per_frame:.2f}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive processing summary.
        
        Returns:
            Dictionary containing processing statistics and configuration
        """
        return {
            'configuration': {
                'input_dir': str(self.input_dir),
                'output_dir': str(self.output_dir),
                'metadata_path': str(self.metadata_path),
                'max_faces_per_frame': self.max_faces_per_frame,
                'frame_skip': self.frame_skip,
                'max_frames_per_video': self.max_frames_per_video,
                'face_detector_config': {
                    'min_detection_confidence': self.face_detector.min_detection_confidence,
                    'target_size': self.face_detector.target_size
                },
                'dwt_processor_config': {
                    'wavelet': self.dwt_processor.wavelet,
                    'levels': self.dwt_processor.levels,
                    'mode': self.dwt_processor.mode
                }
            },
            'statistics': self.stats.copy()
        }