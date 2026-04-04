#!/usr/bin/env python3
"""
Data Preparation Example for CRAFT-DF

This script demonstrates how to prepare video data for training,
including face detection, DWT processing, and metadata generation.

Usage:
    python examples/data_preparation.py
"""

import sys
from pathlib import Path
import tempfile
import shutil
import cv2
import numpy as np
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from craft_df.data.video_processor import VideoProcessor
from craft_df.data.face_detection import FaceDetector
from craft_df.data.dwt_processing import DWTProcessor


def create_sample_videos(output_dir: Path) -> List[Path]:
    """Create sample video files for demonstration."""
    
    print("Creating sample video files...")
    
    # Create directories
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    video_paths = []
    
    # Create sample videos (simple colored frames)
    for category, color in [("real", (0, 255, 0)), ("fake", (255, 0, 0))]:
        for i in range(2):  # 2 videos per category
            video_path = output_dir / category / f"sample_{i+1}.mp4"
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
            
            # Write 30 frames (1 second at 30fps)
            for frame_idx in range(30):
                # Create a frame with some variation
                frame = np.full((480, 640, 3), color, dtype=np.uint8)
                
                # Add some noise and patterns to make it more realistic
                noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
                frame = cv2.add(frame, noise)
                
                # Add a simple face-like rectangle
                cv2.rectangle(frame, (250, 150), (390, 330), (255, 255, 255), 2)
                cv2.circle(frame, (290, 200), 10, (0, 0, 0), -1)  # Left eye
                cv2.circle(frame, (350, 200), 10, (0, 0, 0), -1)  # Right eye
                cv2.rectangle(frame, (310, 250), (330, 280), (0, 0, 0), -1)  # Nose
                cv2.ellipse(frame, (320, 300), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Mouth
                
                writer.write(frame)
            
            writer.release()
            video_paths.append(video_path)
            print(f"Created: {video_path}")
    
    return video_paths


def demonstrate_individual_components():
    """Demonstrate individual data processing components."""
    
    print("\n" + "="*50)
    print("Demonstrating Individual Components")
    print("="*50)
    
    # 1. Face Detection
    print("\n1. Face Detection Component:")
    face_detector = FaceDetector()
    
    # Create a sample image with a face-like pattern
    sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(sample_image, (250, 150), (390, 330), (200, 200, 200), -1)
    cv2.circle(sample_image, (290, 200), 10, (0, 0, 0), -1)
    cv2.circle(sample_image, (350, 200), 10, (0, 0, 0), -1)
    
    faces = face_detector.detect_faces(sample_image)
    print(f"   Detected {len(faces)} faces in sample image")
    
    if faces:
        face_crop = face_detector.crop_face(sample_image, faces[0])
        print(f"   Face crop shape: {face_crop.shape}")
    
    # 2. DWT Processing
    print("\n2. DWT Processing Component:")
    dwt_processor = DWTProcessor()
    
    if faces:
        dwt_coeffs = dwt_processor.process_face_crop(face_crop)
        print(f"   DWT coefficients shape: {dwt_coeffs.shape}")
        print(f"   DWT coefficient range: [{dwt_coeffs.min():.3f}, {dwt_coeffs.max():.3f}]")
    
    print("   Available wavelets:", dwt_processor.get_available_wavelets()[:5], "...")


def demonstrate_full_pipeline():
    """Demonstrate the complete data preparation pipeline."""
    
    print("\n" + "="*50)
    print("Demonstrating Full Pipeline")
    print("="*50)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input_videos"
        output_dir = temp_path / "processed_data"
        metadata_path = temp_path / "metadata.csv"
        
        # Create sample videos
        video_paths = create_sample_videos(input_dir)
        
        # Initialize video processor
        print(f"\nInitializing VideoProcessor...")
        processor = VideoProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_path=metadata_path,
            max_faces_per_frame=1,
            frame_skip=5,  # Process every 5th frame for speed
            face_confidence_threshold=0.3  # Lower threshold for demo
        )
        
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Metadata path: {metadata_path}")
        
        # Process all videos
        print(f"\nProcessing {len(video_paths)} videos...")
        try:
            processor.process_all_videos()
            
            # Display results
            print(f"\nProcessing completed!")
            print(f"Processing statistics:")
            stats = processor.get_processing_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            # Check output structure
            print(f"\nOutput directory structure:")
            for item in sorted(output_dir.rglob("*")):
                if item.is_file():
                    rel_path = item.relative_to(output_dir)
                    print(f"   {rel_path}")
            
            # Display metadata info
            if metadata_path.exists():
                import pandas as pd
                metadata = pd.read_csv(metadata_path)
                print(f"\nMetadata summary:")
                print(f"   Total samples: {len(metadata)}")
                print(f"   Real samples: {len(metadata[metadata['label'] == 0])}")
                print(f"   Fake samples: {len(metadata[metadata['label'] == 1])}")
                print(f"   Columns: {list(metadata.columns)}")
                
                # Show first few rows
                print(f"\nFirst 3 metadata entries:")
                print(metadata.head(3).to_string(index=False))
        
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()


def demonstrate_configuration_options():
    """Demonstrate different configuration options."""
    
    print("\n" + "="*50)
    print("Configuration Options")
    print("="*50)
    
    print("\nVideoProcessor Configuration Options:")
    print("   max_faces_per_frame: Maximum faces to extract per frame (default: 1)")
    print("   frame_skip: Process every Nth frame (default: 1)")
    print("   max_frames_per_video: Limit frames per video (default: None)")
    print("   face_confidence_threshold: Minimum face detection confidence (default: 0.5)")
    print("   save_intermediate: Save intermediate results (default: True)")
    
    print("\nFaceDetector Configuration Options:")
    print("   model_selection: 0 (short-range) or 1 (full-range) (default: 0)")
    print("   min_detection_confidence: Detection confidence threshold (default: 0.5)")
    print("   target_size: Output face crop size (default: [224, 224])")
    
    print("\nDWTProcessor Configuration Options:")
    print("   wavelet: Wavelet type - 'db4', 'db8', 'haar', etc. (default: 'db4')")
    print("   levels: Decomposition levels 1-6 (default: 3)")
    print("   mode: Boundary condition - 'symmetric', 'periodization' (default: 'symmetric')")
    
    # Show example configurations
    print("\nExample Configurations:")
    
    print("\n1. High Quality (Slow):")
    print("   frame_skip: 1")
    print("   face_confidence_threshold: 0.8")
    print("   dwt_levels: 4")
    print("   wavelet: 'db8'")
    
    print("\n2. Fast Processing (Lower Quality):")
    print("   frame_skip: 10")
    print("   face_confidence_threshold: 0.3")
    print("   dwt_levels: 2")
    print("   wavelet: 'haar'")
    
    print("\n3. Memory Optimized:")
    print("   max_frames_per_video: 100")
    print("   target_size: [112, 112]")
    print("   save_intermediate: False")


def main():
    """Run data preparation examples."""
    
    print("CRAFT-DF Data Preparation Examples")
    print("=" * 40)
    
    # Demonstrate individual components
    demonstrate_individual_components()
    
    # Demonstrate full pipeline
    demonstrate_full_pipeline()
    
    # Show configuration options
    demonstrate_configuration_options()
    
    print("\n" + "="*50)
    print("Examples completed successfully!")
    print("="*50)
    
    print("\nNext steps:")
    print("1. Prepare your video dataset in the required structure:")
    print("   input_videos/")
    print("   ├── real/")
    print("   │   ├── video1.mp4")
    print("   │   └── ...")
    print("   └── fake/")
    print("       ├── video1.mp4")
    print("       └── ...")
    
    print("\n2. Run data preparation:")
    print("   python data_prep.py --input_dir ./input_videos --output_dir ./processed_data")
    
    print("\n3. Start training:")
    print("   python train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()