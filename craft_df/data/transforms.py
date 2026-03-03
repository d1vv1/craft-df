"""
Data augmentation and transformation utilities for CRAFT-DF.

This module provides specialized transforms for spatial and frequency domain data
to improve training robustness and model generalization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Tuple, Union
import random
from torchvision import transforms as T


class SpatialAugmentation:
    """
    Data augmentation pipeline for spatial domain face crops.
    
    Applies various geometric and photometric transformations while preserving
    face structure and deepfake artifacts that are crucial for detection.
    
    Args:
        rotation_range: Maximum rotation angle in degrees (default: 15)
        brightness_range: Brightness adjustment range (default: 0.2)
        contrast_range: Contrast adjustment range (default: 0.2)
        saturation_range: Saturation adjustment range (default: 0.2)
        hue_range: Hue adjustment range (default: 0.1)
        horizontal_flip_prob: Probability of horizontal flip (default: 0.5)
        gaussian_noise_std: Standard deviation for Gaussian noise (default: 0.02)
        normalize: Whether to apply ImageNet normalization (default: True)
    """
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        saturation_range: float = 0.2,
        hue_range: float = 0.1,
        horizontal_flip_prob: float = 0.5,
        gaussian_noise_std: float = 0.02,
        normalize: bool = True
    ):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.horizontal_flip_prob = horizontal_flip_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.normalize = normalize
        
        # Build transformation pipeline
        transforms_list = []
        
        # Geometric transformations
        if rotation_range > 0:
            transforms_list.append(
                T.RandomRotation(
                    degrees=rotation_range,
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=0
                )
            )
        
        # Horizontal flip
        if horizontal_flip_prob > 0:
            transforms_list.append(T.RandomHorizontalFlip(p=horizontal_flip_prob))
        
        # Color jittering
        if any([brightness_range, contrast_range, saturation_range, hue_range]):
            transforms_list.append(
                T.ColorJitter(
                    brightness=brightness_range,
                    contrast=contrast_range,
                    saturation=saturation_range,
                    hue=hue_range
                )
            )
        
        # Normalization
        if normalize:
            transforms_list.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225]   # ImageNet stds
                )
            )
        
        self.transform = T.Compose(transforms_list)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial augmentations to input tensor.
        
        Args:
            tensor: Input tensor of shape (C, H, W) or (H, W, C)
            
        Returns:
            Augmented tensor with same shape as input
        """
        # Ensure tensor is in (C, H, W) format
        if tensor.dim() == 3 and tensor.shape[-1] in [1, 3]:
            tensor = tensor.permute(2, 0, 1)
        
        # Apply transformations
        augmented = self.transform(tensor)
        
        # Add Gaussian noise if specified
        if self.gaussian_noise_std > 0:
            noise = torch.randn_like(augmented) * self.gaussian_noise_std
            augmented = augmented + noise
            augmented = torch.clamp(augmented, 0, 1)
        
        return augmented


class FrequencyAugmentation:
    """
    Data augmentation for frequency domain DWT coefficients.
    
    Applies transformations that preserve frequency structure while adding
    robustness to variations in compression and processing artifacts.
    
    Args:
        coefficient_dropout_prob: Probability of dropping individual coefficients (default: 0.1)
        coefficient_noise_std: Standard deviation for coefficient noise (default: 0.05)
        subband_dropout_prob: Probability of dropping entire subbands (default: 0.05)
        amplitude_scaling_range: Range for random amplitude scaling (default: 0.1)
    """
    
    def __init__(
        self,
        coefficient_dropout_prob: float = 0.1,
        coefficient_noise_std: float = 0.05,
        subband_dropout_prob: float = 0.05,
        amplitude_scaling_range: float = 0.1
    ):
        self.coefficient_dropout_prob = coefficient_dropout_prob
        self.coefficient_noise_std = coefficient_noise_std
        self.subband_dropout_prob = subband_dropout_prob
        self.amplitude_scaling_range = amplitude_scaling_range
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency domain augmentations to DWT coefficients.
        
        Args:
            tensor: DWT coefficient tensor
            
        Returns:
            Augmented coefficient tensor
        """
        augmented = tensor.clone()
        
        # Random coefficient dropout
        if self.coefficient_dropout_prob > 0:
            dropout_mask = torch.rand_like(augmented) > self.coefficient_dropout_prob
            augmented = augmented * dropout_mask
        
        # Add noise to coefficients
        if self.coefficient_noise_std > 0:
            noise = torch.randn_like(augmented) * self.coefficient_noise_std
            augmented = augmented + noise
        
        # Random subband dropout (if tensor has multiple channels/subbands)
        if self.subband_dropout_prob > 0 and augmented.dim() >= 3:
            num_subbands = augmented.shape[-1] if augmented.dim() == 3 else augmented.shape[0]
            for i in range(num_subbands):
                if random.random() < self.subband_dropout_prob:
                    if augmented.dim() == 3:
                        augmented[:, :, i] = 0
                    else:
                        augmented[i] = 0
        
        # Random amplitude scaling
        if self.amplitude_scaling_range > 0:
            scale_factor = 1.0 + random.uniform(
                -self.amplitude_scaling_range, 
                self.amplitude_scaling_range
            )
            augmented = augmented * scale_factor
        
        return augmented


class ValidationTransforms:
    """
    Validation/test time transforms without augmentation.
    
    Provides consistent preprocessing for evaluation while maintaining
    the same normalization as training data.
    """
    
    def __init__(self, normalize_spatial: bool = True):
        self.normalize_spatial = normalize_spatial
        
        if normalize_spatial:
            self.spatial_transform = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.spatial_transform = None
    
    def spatial_transform_fn(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply validation transforms to spatial data."""
        # Ensure tensor is in (C, H, W) format
        if tensor.dim() == 3 and tensor.shape[-1] in [1, 3]:
            tensor = tensor.permute(2, 0, 1)
        
        if self.spatial_transform is not None:
            tensor = self.spatial_transform(tensor)
        
        return tensor
    
    def frequency_transform_fn(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply validation transforms to frequency data (no-op for now)."""
        return tensor


def create_train_transforms(
    spatial_augmentation: bool = True,
    frequency_augmentation: bool = True,
    **kwargs
) -> Tuple[Optional[Callable], Optional[Callable]]:
    """
    Create training transforms for spatial and frequency data.
    
    Args:
        spatial_augmentation: Whether to apply spatial augmentations
        frequency_augmentation: Whether to apply frequency augmentations
        **kwargs: Additional arguments for augmentation classes
        
    Returns:
        Tuple of (spatial_transform, frequency_transform)
    """
    spatial_transform = None
    frequency_transform = None
    
    if spatial_augmentation:
        spatial_transform = SpatialAugmentation(**kwargs)
    
    if frequency_augmentation:
        frequency_transform = FrequencyAugmentation(**kwargs)
    
    return spatial_transform, frequency_transform


def create_val_transforms(normalize_spatial: bool = True) -> Tuple[Callable, Callable]:
    """
    Create validation transforms for spatial and frequency data.
    
    Args:
        normalize_spatial: Whether to normalize spatial data
        
    Returns:
        Tuple of (spatial_transform, frequency_transform)
    """
    val_transforms = ValidationTransforms(normalize_spatial=normalize_spatial)
    return val_transforms.spatial_transform_fn, val_transforms.frequency_transform_fn


def get_transforms(
    input_size: Tuple[int, int] = (224, 224),
    augmentation: bool = True,
    **kwargs
) -> Tuple[Callable, Callable]:
    """
    Get appropriate transforms for training and validation.
    
    Args:
        input_size: Target input size (height, width)
        augmentation: Whether to apply augmentation for training
        **kwargs: Additional arguments for augmentation
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    if augmentation:
        train_spatial, train_freq = create_train_transforms(
            spatial_augmentation=True,
            frequency_augmentation=True,
            **kwargs
        )
    else:
        train_spatial, train_freq = create_val_transforms()
    
    val_spatial, val_freq = create_val_transforms()
    
    # Combine spatial and frequency transforms
    def combined_train_transform(sample):
        spatial_data, freq_data, label = sample
        if train_spatial:
            spatial_data = train_spatial(spatial_data)
        if train_freq:
            freq_data = train_freq(freq_data)
        return spatial_data, freq_data, label
    
    def combined_val_transform(sample):
        spatial_data, freq_data, label = sample
        if val_spatial:
            spatial_data = val_spatial(spatial_data)
        if val_freq:
            freq_data = val_freq(freq_data)
        return spatial_data, freq_data, label
    
    return combined_train_transform, combined_val_transform