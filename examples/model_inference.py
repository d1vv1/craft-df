#!/usr/bin/env python3
"""
Model Inference Example for CRAFT-DF

This script demonstrates how to load a trained model and perform
inference on new video data for deepfake detection.

Usage:
    python examples/model_inference.py --checkpoint path/to/model.ckpt --video path/to/video.mp4
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from craft_df.models.craft_df_model import CRAFTDFModel
from craft_df.data.face_detection import FaceDetector
from craft_df.data.dwt_processing import DWTProcessor
from craft_df.utils.config import load_default_config
from craft_df.utils.reproducibility import seed_everything


class CRAFTDFInference:
    """CRAFT-DF inference pipeline for deepfake detection."""
    
    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None):
        """
        Initialize inference pipeline.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Optional path to configuration file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load configuration
        if config_path:
            from craft_df.utils.config import load_config
            self.config = load_config(config_path)
        else:
            self.config = load_default_config()
        
        # Set up reproducibility
        seed_everything(self.config.reproducibility.seed)
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.dwt_processor = DWTProcessor(
            wavelet=self.config.data.wavelet_type,
            levels=self.config.data.dwt_levels
        )
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        print("Inference pipeline initialized successfully")
    
    def _load_model(self, checkpoint_path: str) -> CRAFTDFModel:
        """Load trained model from checkpoint."""
        
        print(f"Loading model from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model with same configuration as training
        model = CRAFTDFModel(
            spatial_dim=getattr(self.config.model, 'spatial_dim', 1280),
            freq_dim=getattr(self.config.model, 'freq_dim', 512),
            attention_heads=self.config.model.attention_heads,
            attention_dim=self.config.model.attention_dim,
            num_classes=2,
            dropout_rate=self.config.model.dropout_rate
        )
        
        # Load state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        
        # Display model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded with {total_params:,} parameters")
        
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Preprocess a single frame for inference.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            
        Returns:
            Tuple of (spatial_features, frequency_features) or None if no face detected
        """
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        if not faces:
            return None
        
        # Use the first (most confident) face
        face_crop = self.face_detector.crop_face(frame, faces[0])
        
        # Process spatial features (normalize to [0, 1])
        spatial_input = face_crop.astype(np.float32) / 255.0
        spatial_tensor = torch.from_numpy(spatial_input).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        # Process frequency features
        dwt_coeffs = self.dwt_processor.process_face_crop(face_crop)
        freq_tensor = torch.from_numpy(dwt_coeffs).unsqueeze(0)  # (1, features)
        
        return spatial_tensor.to(self.device), freq_tensor.to(self.device)
    
    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Predict deepfake probability for a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess frame
        features = self.preprocess_frame(frame)
        
        if features is None:
            return {
                'face_detected': False,
                'prediction': None,
                'confidence': None,
                'is_deepfake': None
            }
        
        spatial_features, freq_features = features
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(spatial_features, freq_features)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            deepfake_prob = probabilities[0, 1].item()  # Probability of being fake
        
        return {
            'face_detected': True,
            'prediction': predicted_class,  # 0: real, 1: fake
            'confidence': confidence,
            'deepfake_probability': deepfake_prob,
            'is_deepfake': predicted_class == 1
        }
    
    def predict_video(self, video_path: str, frame_skip: int = 5) -> Dict[str, Any]:
        """
        Predict deepfake probability for an entire video.
        
        Args:
            video_path: Path to input video
            frame_skip: Process every Nth frame
            
        Returns:
            Dictionary with aggregated video results
        """
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_predictions = []
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for efficiency
                if frame_count % frame_skip != 0:
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get prediction for this frame
                result = self.predict_frame(frame_rgb)
                
                if result['face_detected']:
                    frame_predictions.append(result)
                    processed_count += 1
                
                # Progress update
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} frames with faces...")
        
        finally:
            cap.release()
        
        # Aggregate results
        if not frame_predictions:
            return {
                'total_frames': frame_count,
                'processed_frames': 0,
                'faces_detected': 0,
                'average_deepfake_probability': None,
                'video_prediction': None,
                'confidence': None
            }
        
        # Calculate statistics
        deepfake_probs = [p['deepfake_probability'] for p in frame_predictions]
        avg_deepfake_prob = np.mean(deepfake_probs)
        std_deepfake_prob = np.std(deepfake_probs)
        
        # Video-level prediction (majority vote with confidence weighting)
        weighted_votes = []
        for pred in frame_predictions:
            weight = pred['confidence']
            vote = 1 if pred['is_deepfake'] else 0
            weighted_votes.append(vote * weight)
        
        video_prediction = 1 if np.mean(weighted_votes) > 0.5 else 0
        video_confidence = abs(np.mean(weighted_votes) - 0.5) * 2  # Scale to [0, 1]
        
        return {
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'faces_detected': len(frame_predictions),
            'average_deepfake_probability': avg_deepfake_prob,
            'std_deepfake_probability': std_deepfake_prob,
            'video_prediction': video_prediction,  # 0: real, 1: fake
            'video_confidence': video_confidence,
            'is_deepfake': video_prediction == 1,
            'frame_predictions': frame_predictions
        }


def create_demo_video(output_path: str) -> str:
    """Create a demo video for testing inference."""
    
    print(f"Creating demo video: {output_path}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    # Create 60 frames (2 seconds)
    for frame_idx in range(60):
        # Create frame with face-like pattern
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add face-like features
        cv2.rectangle(frame, (250, 150), (390, 330), (180, 180, 180), -1)
        cv2.circle(frame, (290, 200), 8, (0, 0, 0), -1)  # Left eye
        cv2.circle(frame, (350, 200), 8, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(frame, (315, 250), (325, 270), (0, 0, 0), -1)  # Nose
        cv2.ellipse(frame, (320, 300), (25, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        # Add some temporal variation
        if frame_idx > 30:  # Second half looks more "artificial"
            cv2.rectangle(frame, (250, 150), (390, 330), (255, 0, 0), 3)  # Blue border
        
        writer.write(frame)
    
    writer.release()
    return output_path


def main():
    """Run inference examples."""
    
    parser = argparse.ArgumentParser(description="CRAFT-DF Inference Example")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--demo", action="store_true", help="Run with demo video")
    parser.add_argument("--frame_skip", type=int, default=5, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    print("CRAFT-DF Model Inference Example")
    print("=" * 40)
    
    # Handle demo mode
    if args.demo:
        print("Running in demo mode...")
        
        # Create demo video
        demo_video_path = "demo_video.mp4"
        create_demo_video(demo_video_path)
        
        # Use demo checkpoint (would need to be created)
        if not args.checkpoint:
            print("Demo mode requires a trained checkpoint.")
            print("Please train a model first or provide --checkpoint argument.")
            return
        
        args.video = demo_video_path
    
    # Validate arguments
    if not args.checkpoint:
        print("Error: --checkpoint argument is required")
        print("Please provide path to a trained model checkpoint")
        return
    
    if not args.video:
        print("Error: --video argument is required (or use --demo)")
        return
    
    # Check if files exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    try:
        # Initialize inference pipeline
        print("Initializing inference pipeline...")
        inference = CRAFTDFInference(args.checkpoint, args.config)
        
        # Run inference on video
        print(f"Running inference on video...")
        results = inference.predict_video(args.video, frame_skip=args.frame_skip)
        
        # Display results
        print("\n" + "="*50)
        print("INFERENCE RESULTS")
        print("="*50)
        
        print(f"Video: {args.video}")
        print(f"Total frames: {results['total_frames']}")
        print(f"Processed frames: {results['processed_frames']}")
        print(f"Faces detected: {results['faces_detected']}")
        
        if results['faces_detected'] > 0:
            print(f"\nPrediction Results:")
            print(f"Average deepfake probability: {results['average_deepfake_probability']:.3f}")
            print(f"Standard deviation: {results['std_deepfake_probability']:.3f}")
            print(f"Video prediction: {'DEEPFAKE' if results['is_deepfake'] else 'REAL'}")
            print(f"Video confidence: {results['video_confidence']:.3f}")
            
            # Show frame-by-frame statistics
            frame_preds = results['frame_predictions']
            deepfake_frames = sum(1 for p in frame_preds if p['is_deepfake'])
            print(f"\nFrame Statistics:")
            print(f"Frames classified as deepfake: {deepfake_frames}/{len(frame_preds)}")
            print(f"Percentage deepfake frames: {deepfake_frames/len(frame_preds)*100:.1f}%")
            
            # Show confidence distribution
            confidences = [p['confidence'] for p in frame_preds]
            print(f"Average frame confidence: {np.mean(confidences):.3f}")
            print(f"Min/Max confidence: {np.min(confidences):.3f}/{np.max(confidences):.3f}")
        else:
            print("\nNo faces detected in video!")
            print("This could indicate:")
            print("- Video quality issues")
            print("- No faces present in the video")
            print("- Face detection threshold too high")
        
        print("\nInference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up demo video if created
        if args.demo and Path("demo_video.mp4").exists():
            Path("demo_video.mp4").unlink()


if __name__ == "__main__":
    main()