"""
Data processing and dataset management for CRAFT-DF.

This module provides utilities for face detection, DWT processing,
video processing, and hierarchical dataset management.
"""

from .face_detection import FaceDetector
from .dwt_processing import DWTProcessor
from .video_processor import VideoProcessor
from .dataset import HierarchicalDeepfakeDataset
from .transforms import (
    SpatialAugmentation,
    FrequencyAugmentation,
    ValidationTransforms,
    create_train_transforms,
    create_val_transforms
)

__all__ = [
    'FaceDetector',
    'DWTProcessor', 
    'VideoProcessor',
    'HierarchicalDeepfakeDataset',
    'SpatialAugmentation',
    'FrequencyAugmentation',
    'ValidationTransforms',
    'create_train_transforms',
    'create_val_transforms'
]