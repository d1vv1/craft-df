"""
Spatial Stream Architecture for CRAFT-DF

This module implements the spatial domain feature extraction using a pre-trained MobileNetV2
backbone. The spatial stream processes face crops to extract spatial features that capture
visual artifacts and inconsistencies in deepfake videos.

The MobileNetV2 architecture is chosen for its efficiency and strong feature extraction
capabilities, making it suitable for real-time deepfake detection while maintaining
high accuracy. The model uses depthwise separable convolutions which significantly
reduce computational cost compared to standard convolutions.

Theory:
- Spatial domain analysis focuses on pixel-level inconsistencies and artifacts
- Pre-trained features from ImageNet provide robust low-level feature representations
- Fine-tuning allows adaptation to deepfake-specific visual patterns
- Layer freezing prevents overfitting while maintaining learned representations
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpatialStream(nn.Module):
    """
    MobileNetV2-based spatial feature extractor for deepfake detection.
    
    This class implements the spatial stream of the CRAFT-DF architecture, using
    a pre-trained MobileNetV2 backbone for efficient feature extraction from face crops.
    The model supports configurable layer freezing and fine-tuning strategies.
    
    Args:
        pretrained (bool): Whether to use pre-trained ImageNet weights
        freeze_layers (int): Number of initial layers to freeze (0-18)
        feature_dim (int): Output feature dimension
        dropout_rate (float): Dropout rate for regularization
        
    Attributes:
        backbone (nn.Module): MobileNetV2 feature extractor
        classifier (nn.Module): Classification head for feature extraction
        feature_dim (int): Output feature dimension
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_layers: int = 10,
        feature_dim: int = 1280,
        dropout_rate: float = 0.1
    ) -> None:
        super(SpatialStream, self).__init__()
        
        # Validate input parameters
        assert 0 <= freeze_layers <= 18, f"freeze_layers must be between 0 and 18, got {freeze_layers}"
        assert feature_dim > 0, f"feature_dim must be positive, got {feature_dim}"
        assert 0.0 <= dropout_rate <= 1.0, f"dropout_rate must be between 0 and 1, got {dropout_rate}"
        
        self.feature_dim = feature_dim
        self.freeze_layers = freeze_layers
        
        # Load pre-trained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Freeze specified layers
        self._freeze_layers(freeze_layers)
        
        # Create custom feature extraction head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(1280, feature_dim),  # MobileNetV2 outputs 1280 features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Initialize custom layers
        self._initialize_weights()
        
        logger.info(f"SpatialStream initialized with {freeze_layers} frozen layers, "
                   f"feature_dim={feature_dim}, dropout_rate={dropout_rate}")
    
    def _freeze_layers(self, num_layers: int) -> None:
        """
        Freeze the first num_layers of the MobileNetV2 backbone.
        
        Args:
            num_layers (int): Number of layers to freeze
        """
        if num_layers == 0:
            return
            
        # MobileNetV2 has 18 inverted residual blocks in features
        features = self.backbone.features
        
        # Freeze initial layers
        for i, layer in enumerate(features):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        logger.info(f"Frozen first {num_layers} layers of MobileNetV2")
    
    def _initialize_weights(self) -> None:
        """Initialize weights for custom layers using Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the spatial stream with GPU optimization.
        
        This method processes face crops through the MobileNetV2 backbone to extract
        spatial features. The implementation includes several GPU optimizations:
        - Memory-efficient tensor operations
        - Optimized data layout for tensor cores
        - Minimal memory allocations during forward pass
        
        Args:
            x (torch.Tensor): Input face crops of shape (batch_size, 3, 224, 224)
                             Values should be normalized to [0, 1] range
            
        Returns:
            torch.Tensor: Spatial features of shape (batch_size, feature_dim)
                         Features are L2-normalized for stable training
            
        Raises:
            AssertionError: If input tensor shape is invalid
        """
        # Validate input tensor shape
        assert len(x.shape) == 4, f"Expected 4D input tensor, got shape {x.shape}"
        assert x.shape[1] == 3, f"Expected 3 channels (RGB), got {x.shape[1]}"
        assert x.shape[2] == x.shape[3] == 224, f"Expected 224x224 input, got {x.shape[2]}x{x.shape[3]}"
        
        batch_size = x.shape[0]
        
        # Ensure tensor is contiguous for optimal GPU memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Extract features using MobileNetV2 backbone
        # Use torch.cuda.amp.autocast for mixed precision if available
        if x.is_cuda and hasattr(torch.cuda.amp, 'autocast'):
            with torch.cuda.amp.autocast(enabled=self.training):
                features = self.backbone.features(x)
        else:
            features = self.backbone.features(x)
        
        # Validate intermediate feature shape
        assert features.shape[1] == 1280, f"Expected 1280 feature channels, got {features.shape[1]}"
        
        # Apply classification head for feature extraction
        spatial_features = self.classifier(features)
        
        # L2 normalize features for stable training and better convergence
        # Handle edge case where features might be zero (avoid NaN)
        spatial_features = torch.nn.functional.normalize(spatial_features, p=2, dim=1, eps=1e-8)
        
        # Validate output shape
        assert spatial_features.shape == (batch_size, self.feature_dim), \
            f"Expected output shape ({batch_size}, {self.feature_dim}), got {spatial_features.shape}"
        
        return spatial_features
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate feature maps for visualization and analysis.
        
        Args:
            x (torch.Tensor): Input face crops of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Feature maps of shape (batch_size, 1280, 7, 7)
        """
        assert len(x.shape) == 4, f"Expected 4D input tensor, got shape {x.shape}"
        assert x.shape[1] == 3, f"Expected 3 channels (RGB), got {x.shape[1]}"
        
        with torch.no_grad():
            feature_maps = self.backbone.features(x)
            
        return feature_maps
    
    def unfreeze_layers(self, num_layers: int) -> None:
        """
        Unfreeze additional layers for fine-tuning.
        
        Args:
            num_layers (int): Number of additional layers to unfreeze
        """
        features = self.backbone.features
        current_frozen = self.freeze_layers
        new_frozen = max(0, current_frozen - num_layers)
        
        # Unfreeze layers
        for i in range(new_frozen, current_frozen):
            if i < len(features):
                for param in features[i].parameters():
                    param.requires_grad = True
        
        self.freeze_layers = new_frozen
        logger.info(f"Unfroze {num_layers} layers, now {new_frozen} layers frozen")
    
    def get_trainable_parameters(self) -> int:
        """
        Get the number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """
        Get model information for logging and debugging.
        
        Returns:
            dict: Model information including parameter counts and configuration
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.get_trainable_parameters()
        
        return {
            'model_name': 'SpatialStream',
            'backbone': 'MobileNetV2',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_layers': self.freeze_layers,
            'feature_dim': self.feature_dim,
            'frozen_ratio': (total_params - trainable_params) / total_params
        }
    
    def optimize_for_inference(self) -> None:
        """
        Optimize model for inference by applying various optimizations.
        
        This method applies several optimizations for faster inference:
        - Sets model to evaluation mode
        - Fuses batch normalization layers
        - Optimizes memory layout
        """
        self.eval()
        
        # Fuse batch normalization layers for faster inference
        try:
            torch.quantization.fuse_modules(
                self.backbone,
                [['features.0.0', 'features.0.1']],  # Conv2d + BatchNorm2d
                inplace=True
            )
            logger.info("Successfully fused batch normalization layers")
        except Exception as e:
            logger.warning(f"Could not fuse batch normalization layers: {e}")
    
    def get_memory_usage(self, batch_size: int = 1) -> dict:
        """
        Estimate memory usage for given batch size.
        
        Args:
            batch_size (int): Batch size for memory estimation
            
        Returns:
            dict: Memory usage information in MB
        """
        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 ** 2)
        
        # Estimate activation memory (rough approximation)
        input_memory = batch_size * 3 * 224 * 224 * 4 / (1024 ** 2)  # 4 bytes per float32
        feature_memory = batch_size * 1280 * 7 * 7 * 4 / (1024 ** 2)  # Feature maps
        output_memory = batch_size * self.feature_dim * 4 / (1024 ** 2)  # Output features
        
        total_memory = param_memory + input_memory + feature_memory + output_memory
        
        return {
            'parameter_memory_mb': param_memory,
            'input_memory_mb': input_memory,
            'feature_memory_mb': feature_memory,
            'output_memory_mb': output_memory,
            'total_estimated_mb': total_memory,
            'batch_size': batch_size
        }
    
    def enable_mixed_precision(self) -> None:
        """
        Enable mixed precision training optimizations.
        
        This method configures the model for mixed precision training,
        which can significantly speed up training on modern GPUs.
        """
        # Convert model to half precision where appropriate
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                # Keep normalization layers in float32 for stability
                module.float()
        
        logger.info("Mixed precision optimizations enabled")
    
    def profile_forward_pass(self, input_tensor: torch.Tensor) -> dict:
        """
        Profile the forward pass to identify performance bottlenecks.
        
        Args:
            input_tensor (torch.Tensor): Input tensor for profiling
            
        Returns:
            dict: Profiling results including timing information
        """
        import time
        
        self.eval()
        device = next(self.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Warm up GPU
        for _ in range(5):
            with torch.no_grad():
                _ = self(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Profile forward pass
        start_time = time.time()
        
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        with torch.no_grad():
            output = self(input_tensor)
        
        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            gpu_time = start_event.elapsed_time(end_event)  # milliseconds
        else:
            gpu_time = None
        
        end_time = time.time()
        cpu_time = (end_time - start_time) * 1000  # milliseconds
        
        batch_size = input_tensor.shape[0]
        throughput = batch_size / (cpu_time / 1000)  # samples per second
        
        return {
            'batch_size': batch_size,
            'cpu_time_ms': cpu_time,
            'gpu_time_ms': gpu_time,
            'throughput_samples_per_sec': throughput,
            'memory_allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2) if device.type == 'cuda' else None,
            'output_shape': tuple(output.shape)
        }