#!/usr/bin/env python3
"""
Comprehensive data preparation script for CRAFT-DF deepfake detection.

This script provides a complete data preprocessing pipeline that integrates
face detection, cropping, and DWT feature extraction for deepfake detection.
It handles batch processing of video files and generates hierarchical data
organization with metadata management.

Usage:
    python data_prep.py --input_dir /path/to/videos --output_dir /path/to/output
    python data_prep.py --config config.yaml
    python data_prep.py --help

Features:
    - Batch video processing with progress tracking
    - Face detection and cropping using OpenCV
    - DWT feature extraction for frequency domain analysis
    - Hierarchical file organization (real/fake/video_id/frame_xxx.npy)
    - Comprehensive metadata CSV generation
    - Error handling and recovery mechanisms
    - Configurable processing parameters
    - Resume capability for interrupted processing
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import json
from datetime import datetime
import traceback

# Import CRAFT-DF data processing components
from craft_df.data.video_processor import VideoProcessor
from craft_df.data.face_detection import FaceDetector
from craft_df.data.dwt_processing import DWTProcessor


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path for persistent logging
    """
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Set up handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )
    
    # Set specific logger levels
    logging.getLogger("craft_df").setLevel(getattr(logging, log_level.upper()))
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    logger = logging.getLogger(__name__)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML configuration: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    logger = logging.getLogger(__name__)
    
    # Required parameters
    required_params = ['input_dir', 'output_dir', 'metadata_path']
    missing_params = [param for param in required_params if param not in config]
    
    if missing_params:
        raise ValueError(f"Missing required configuration parameters: {missing_params}")
    
    # Validate paths
    input_dir = Path(config['input_dir'])
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Validate numeric parameters
    numeric_params = {
        'max_faces_per_frame': (1, 10),
        'frame_skip': (1, 100),
        'max_frames_per_video': (1, None)
    }
    
    for param, (min_val, max_val) in numeric_params.items():
        if param in config:
            value = config[param]
            if not isinstance(value, int) or value < min_val:
                raise ValueError(f"{param} must be an integer >= {min_val}")
            if max_val is not None and value > max_val:
                raise ValueError(f"{param} must be <= {max_val}")
    
    # Validate face detector config
    if 'face_detector' in config:
        face_config = config['face_detector']
        if 'min_detection_confidence' in face_config:
            conf = face_config['min_detection_confidence']
            if not 0.0 <= conf <= 1.0:
                raise ValueError("min_detection_confidence must be between 0.0 and 1.0")
        
        if 'target_size' in face_config:
            size = face_config['target_size']
            if not (isinstance(size, list) and len(size) == 2 and all(s > 0 for s in size)):
                raise ValueError("target_size must be a list of two positive integers")
    
    # Validate DWT processor config
    if 'dwt_processor' in config:
        dwt_config = config['dwt_processor']
        if 'levels' in dwt_config:
            levels = dwt_config['levels']
            if not isinstance(levels, int) or not 1 <= levels <= 6:
                raise ValueError("DWT levels must be an integer between 1 and 6")
    
    logger.info("Configuration validation passed")


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration dictionary.
    
    Returns:
        Default configuration parameters
    """
    return {
        # Required parameters
        'input_dir': './input_videos',
        'output_dir': './processed_data',
        'metadata_path': './metadata.csv',
        
        # Processing parameters
        'max_faces_per_frame': 1,
        'frame_skip': 1,
        'max_frames_per_video': None,
        'save_intermediate': True,
        'progress_bar': True,
        
        # Face detector configuration
        'face_detector': {
            'min_detection_confidence': 0.7,
            'model_selection': 0,
            'target_size': [224, 224]
        },
        
        # DWT processor configuration
        'dwt_processor': {
            'wavelet': 'db4',
            'levels': 3,
            'mode': 'symmetric'
        },
        
        # Video file extensions
        'video_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'],
        
        # Logging configuration
        'logging': {
            'level': 'INFO',
            'log_file': None
        },
        
        # Resume processing
        'resume': False
    }


def save_config_template(output_path: Path) -> None:
    """
    Save a template configuration file.
    
    Args:
        output_path: Path where to save the template
    """
    logger = logging.getLogger(__name__)
    
    config = create_default_config()
    
    # Add comments to the configuration
    config_with_comments = {
        '# CRAFT-DF Data Preparation Configuration': None,
        '# Required Parameters': None,
        'input_dir': config['input_dir'],
        'output_dir': config['output_dir'], 
        'metadata_path': config['metadata_path'],
        
        '# Processing Parameters': None,
        'max_faces_per_frame': config['max_faces_per_frame'],
        'frame_skip': config['frame_skip'],
        'max_frames_per_video': config['max_frames_per_video'],
        'save_intermediate': config['save_intermediate'],
        'progress_bar': config['progress_bar'],
        
        '# Face Detector Configuration': None,
        'face_detector': config['face_detector'],
        
        '# DWT Processor Configuration': None,
        'dwt_processor': config['dwt_processor'],
        
        '# Video Extensions': None,
        'video_extensions': config['video_extensions'],
        
        '# Logging Configuration': None,
        'logging': config['logging'],
        
        '# Resume Processing': None,
        'resume': config['resume']
    }
    
    # Remove comment entries for actual YAML output
    clean_config = {k: v for k, v in config_with_comments.items() if not k.startswith('#')}
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(clean_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration template saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration template: {e}")
        raise


def check_resume_capability(config: Dict[str, Any]) -> bool:
    """
    Check if processing can be resumed from previous run.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if resume is possible, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if not config.get('resume', False):
        return False
    
    metadata_path = Path(config['metadata_path'])
    temp_metadata_path = metadata_path.with_suffix('.tmp.csv')
    
    if temp_metadata_path.exists():
        logger.info(f"Found temporary metadata file: {temp_metadata_path}")
        logger.info("Resume capability detected")
        return True
    
    return False


def resume_processing(config: Dict[str, Any]) -> Optional[List[str]]:
    """
    Resume processing from previous interrupted run.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of already processed video IDs, or None if resume not possible
    """
    logger = logging.getLogger(__name__)
    
    metadata_path = Path(config['metadata_path'])
    temp_metadata_path = metadata_path.with_suffix('.tmp.csv')
    
    if not temp_metadata_path.exists():
        return None
    
    try:
        import pandas as pd
        temp_df = pd.read_csv(temp_metadata_path)
        processed_videos = temp_df['video_id'].unique().tolist()
        
        logger.info(f"Resuming processing: {len(processed_videos)} videos already processed")
        return processed_videos
        
    except Exception as e:
        logger.warning(f"Failed to load temporary metadata for resume: {e}")
        return None


def process_videos(config: Dict[str, Any]) -> None:
    """
    Main video processing function.
    
    Args:
        config: Configuration dictionary containing all processing parameters
        
    Raises:
        RuntimeError: If processing fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize video processor
        logger.info("Initializing video processor...")
        
        processor = VideoProcessor(
            input_dir=config['input_dir'],
            output_dir=config['output_dir'],
            metadata_path=config['metadata_path'],
            face_detector_config=config.get('face_detector', {}),
            dwt_processor_config=config.get('dwt_processor', {}),
            max_faces_per_frame=config.get('max_faces_per_frame', 1),
            frame_skip=config.get('frame_skip', 1),
            max_frames_per_video=config.get('max_frames_per_video')
        )
        
        logger.info("Video processor initialized successfully")
        
        # Get video files to process
        video_extensions = config.get('video_extensions', ['.mp4', '.avi', '.mov'])
        video_files = processor.get_video_files(extensions=video_extensions)
        
        if not video_files:
            logger.warning("No video files found to process")
            return
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        # Check for resume capability
        processed_videos = []
        if check_resume_capability(config):
            processed_videos = resume_processing(config) or []
        
        # Filter out already processed videos
        if processed_videos:
            original_count = len(video_files)
            video_files = [vf for vf in video_files if vf.stem not in processed_videos]
            logger.info(f"Skipping {original_count - len(video_files)} already processed videos")
        
        if not video_files:
            logger.info("All videos have already been processed")
            return
        
        # Process videos
        logger.info(f"Starting batch processing of {len(video_files)} videos")
        
        metadata_df = processor.process_video_batch(
            video_paths=video_files,
            save_intermediate=config.get('save_intermediate', True),
            progress_bar=config.get('progress_bar', True)
        )
        
        # Generate final metadata CSV
        if not metadata_df.empty:
            logger.info("Generating final metadata CSV...")
            processor.generate_metadata_csv(metadata_df)
            
            # Clean up temporary files
            temp_metadata_path = Path(config['metadata_path']).with_suffix('.tmp.csv')
            if temp_metadata_path.exists():
                temp_metadata_path.unlink()
                logger.info("Cleaned up temporary metadata file")
        
        # Log final processing summary
        summary = processor.get_processing_summary()
        logger.info("=== Processing Complete ===")
        logger.info(f"Videos processed: {summary['statistics']['videos_processed']}")
        logger.info(f"Frames processed: {summary['statistics']['frames_processed']}")
        logger.info(f"Faces extracted: {summary['statistics']['faces_extracted']}")
        logger.info(f"Errors encountered: {summary['statistics']['errors']}")
        
        if summary['statistics']['start_time'] and summary['statistics']['end_time']:
            duration = summary['statistics']['end_time'] - summary['statistics']['start_time']
            logger.info(f"Total processing time: {duration}")
        
        # Save processing summary
        summary_path = Path(config['output_dir']) / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Processing summary saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Video processing failed: {e}")


def main():
    """Main entry point for the data preparation script."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="CRAFT-DF Data Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with command line arguments
    python data_prep.py --input_dir ./videos --output_dir ./processed --metadata_path ./metadata.csv
    
    # Use configuration file
    python data_prep.py --config config.yaml
    
    # Generate configuration template
    python data_prep.py --generate_config config_template.yaml
    
    # Resume interrupted processing
    python data_prep.py --config config.yaml --resume
    
    # Enable debug logging
    python data_prep.py --config config.yaml --log_level DEBUG --log_file processing.log
        """
    )
    
    # Configuration options
    parser.add_argument('--config', type=Path, help='Path to YAML configuration file')
    parser.add_argument('--generate_config', type=Path, help='Generate configuration template and exit')
    
    # Basic parameters (can override config file)
    parser.add_argument('--input_dir', type=Path, help='Input directory containing videos')
    parser.add_argument('--output_dir', type=Path, help='Output directory for processed data')
    parser.add_argument('--metadata_path', type=Path, help='Path for metadata CSV file')
    
    # Processing parameters
    parser.add_argument('--max_faces_per_frame', type=int, help='Maximum faces to extract per frame')
    parser.add_argument('--frame_skip', type=int, help='Process every Nth frame')
    parser.add_argument('--max_frames_per_video', type=int, help='Maximum frames to process per video')
    
    # Face detector parameters
    parser.add_argument('--min_detection_confidence', type=float, help='Minimum face detection confidence')
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('HEIGHT', 'WIDTH'), 
                       help='Target size for face crops')
    
    # DWT processor parameters
    parser.add_argument('--wavelet', type=str, help='Wavelet type for DWT')
    parser.add_argument('--dwt_levels', type=int, help='Number of DWT decomposition levels')
    parser.add_argument('--dwt_mode', type=str, help='DWT boundary condition mode')
    
    # Control options
    parser.add_argument('--resume', action='store_true', help='Resume interrupted processing')
    parser.add_argument('--no_progress', action='store_true', help='Disable progress bar')
    parser.add_argument('--no_intermediate', action='store_true', help='Disable intermediate saves')
    
    # Logging options
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log_file', type=Path, help='Log file path')
    
    args = parser.parse_args()
    
    # Handle configuration template generation
    if args.generate_config:
        try:
            save_config_template(args.generate_config)
            print(f"Configuration template saved to {args.generate_config}")
            return 0
        except Exception as e:
            print(f"Error generating configuration template: {e}")
            return 1
    
    # Set up logging (basic setup, will be reconfigured after loading config)
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config = load_config(args.config)
        else:
            logger.info("Using default configuration")
            config = create_default_config()
        
        # Override config with command line arguments
        if args.input_dir:
            config['input_dir'] = str(args.input_dir)
        if args.output_dir:
            config['output_dir'] = str(args.output_dir)
        if args.metadata_path:
            config['metadata_path'] = str(args.metadata_path)
        if args.max_faces_per_frame:
            config['max_faces_per_frame'] = args.max_faces_per_frame
        if args.frame_skip:
            config['frame_skip'] = args.frame_skip
        if args.max_frames_per_video:
            config['max_frames_per_video'] = args.max_frames_per_video
        if args.resume:
            config['resume'] = True
        if args.no_progress:
            config['progress_bar'] = False
        if args.no_intermediate:
            config['save_intermediate'] = False
        
        # Override face detector config
        if 'face_detector' not in config:
            config['face_detector'] = {}
        if args.min_detection_confidence:
            config['face_detector']['min_detection_confidence'] = args.min_detection_confidence
        if args.target_size:
            config['face_detector']['target_size'] = args.target_size
        
        # Override DWT processor config
        if 'dwt_processor' not in config:
            config['dwt_processor'] = {}
        if args.wavelet:
            config['dwt_processor']['wavelet'] = args.wavelet
        if args.dwt_levels:
            config['dwt_processor']['levels'] = args.dwt_levels
        if args.dwt_mode:
            config['dwt_processor']['mode'] = args.dwt_mode
        
        # Override logging config
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['level'] = args.log_level
        if args.log_file:
            config['logging']['log_file'] = str(args.log_file)
        
        # Reconfigure logging with final settings
        log_file = Path(config['logging']['log_file']) if config['logging'].get('log_file') else None
        setup_logging(config['logging']['level'], log_file)
        
        # Validate configuration
        validate_config(config)
        
        # Log configuration summary
        logger.info("=== Configuration Summary ===")
        logger.info(f"Input directory: {config['input_dir']}")
        logger.info(f"Output directory: {config['output_dir']}")
        logger.info(f"Metadata path: {config['metadata_path']}")
        logger.info(f"Max faces per frame: {config['max_faces_per_frame']}")
        logger.info(f"Frame skip: {config['frame_skip']}")
        logger.info(f"Max frames per video: {config.get('max_frames_per_video', 'unlimited')}")
        logger.info(f"Resume processing: {config.get('resume', False)}")
        
        # Start processing
        logger.info("Starting CRAFT-DF data preparation pipeline...")
        process_videos(config)
        
        logger.info("Data preparation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())