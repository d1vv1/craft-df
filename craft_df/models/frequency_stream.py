"""
Frequency Stream Architecture for CRAFT-DF

This module implements the frequency domain feature extraction using Discrete Wavelet Transform (DWT)
coefficients. The frequency stream processes multi-level wavelet decompositions to extract frequency
artifacts and anomalies that are characteristic of deepfake generation processes.

The frequency domain analysis is crucial for deepfake detection because:
1. Generative models often introduce frequency artifacts during synthesis
2. Compression artifacts from video encoding create distinctive frequency patterns
3. DWT provides multi-resolution analysis capturing both coarse and fine frequency details
4. High-frequency detail coefficients contain the most discriminative information

Theory:
- DWT decomposes images into approximation (LL) and detail (LH, HL, HH) coefficients
- Multi-level decomposition provides frequency analysis at different scales
- Convolutional layers learn to detect frequency-domain artifacts
- Attention mechanisms focus on the most discriminative frequency patterns
- Feature disentanglement improves generalization across different deepfake methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import logging
import numpy as np
import numpy as np

logger = logging.getLogger(__name__)


class DWTLayer(nn.Module):
    """
    Custom DWT processing layer for handling wavelet coefficients.
    
    This layer processes DWT coefficients from multiple decomposition levels,
    applying learned transformations to extract frequency-domain features
    relevant for deepfake detection.
    
    The layer handles the complex structure of DWT coefficients:
    - Approximation coefficients (LL) from the deepest level
    - Detail coefficients (LH, HL, HH) from all levels
    - Multi-channel processing for RGB images
    
    Args:
        input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
        dwt_levels (int): Number of DWT decomposition levels
        feature_dim (int): Output feature dimension
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        dwt_levels: int = 3,
        feature_dim: int = 256,
        dropout_rate: float = 0.1
    ) -> None:
        super(DWTLayer, self).__init__()
        
        # Validate parameters
        assert input_channels > 0, f"input_channels must be positive, got {input_channels}"
        assert 1 <= dwt_levels <= 6, f"dwt_levels must be between 1 and 6, got {dwt_levels}"
        assert feature_dim > 0, f"feature_dim must be positive, got {feature_dim}"
        assert 0.0 <= dropout_rate <= 1.0, f"dropout_rate must be between 0 and 1, got {dropout_rate}"
        
        self.input_channels = input_channels
        self.dwt_levels = dwt_levels
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        
        # Calculate expected input dimensions for each level
        # Each level has 4 subbands (LL, LH, HL, HH) except the deepest level which only has LL
        # Total subbands = 1 (LL) + 3 * dwt_levels (detail coefficients)
        total_subbands = 1 + 3 * dwt_levels
        
        # Convolutional layers for processing each type of coefficient
        self.ll_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.detail_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Feature fusion layers
        # LL features: 128 channels, Detail features: 64 channels * 3 subbands * levels
        ll_features = 128 * 8 * 8  # 8192
        detail_features = 64 * 8 * 8 * 3 * dwt_levels  # Variable based on levels
        total_features = ll_features + detail_features
        
        self.feature_fusion = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(total_features, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"DWTLayer initialized: channels={input_channels}, levels={dwt_levels}, "
                   f"feature_dim={feature_dim}, total_features={total_features}")
    
    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier initialization for better gradient flow."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, dwt_coefficients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through DWT processing layer with optimized tensor operations.
        
        This method processes DWT coefficients from multiple decomposition levels,
        applying learned transformations to extract frequency-domain features.
        Optimized for GPU performance with efficient memory usage and tensor operations.
        
        Args:
            dwt_coefficients: Dictionary containing DWT coefficients with keys:
                - 'll': Approximation coefficients (batch_size, channels, H, W)
                - 'lh_1', 'hl_1', 'hh_1': Level 1 detail coefficients
                - 'lh_2', 'hl_2', 'hh_2': Level 2 detail coefficients
                - ... (up to dwt_levels)
                
        Returns:
            torch.Tensor: Frequency features of shape (batch_size, feature_dim)
            
        Raises:
            AssertionError: If input format is invalid
            RuntimeError: If processing fails due to memory or computation issues
        """
        try:
            # Validate input format
            if not isinstance(dwt_coefficients, dict):
                raise ValueError("dwt_coefficients must be a dictionary")
            if 'll' not in dwt_coefficients:
                raise ValueError("Missing approximation coefficients (ll)")
            
            batch_size = dwt_coefficients['ll'].shape[0]
            
            # Process approximation coefficients (LL) with error handling
            ll_coeff = dwt_coefficients['ll']
            if len(ll_coeff.shape) != 4:
                raise ValueError(f"Expected 4D LL tensor, got shape {ll_coeff.shape}")
            if ll_coeff.shape[1] != self.input_channels:
                raise ValueError(f"Expected {self.input_channels} channels, got {ll_coeff.shape[1]}")
            
            # Check for invalid values
            if not torch.all(torch.isfinite(ll_coeff)):
                logger.warning("Non-finite values detected in LL coefficients, clamping...")
                ll_coeff = torch.clamp(ll_coeff, -1e6, 1e6)
            
            # Ensure tensor is contiguous for optimal GPU memory access
            if not ll_coeff.is_contiguous():
                ll_coeff = ll_coeff.contiguous()
            
            # Process LL coefficients with mixed precision if available
            if ll_coeff.is_cuda and hasattr(torch.cuda.amp, 'autocast'):
                with torch.cuda.amp.autocast(enabled=self.training):
                    ll_features = self.ll_conv(ll_coeff)
            else:
                ll_features = self.ll_conv(ll_coeff)
            
            # MPS workaround: adaptive pooling with non-divisible sizes crashes on MPS
            if ll_features.device.type == 'mps':
                ll_features = self.adaptive_pool(ll_features.cpu()).to('mps')
            else:
                ll_features = self.adaptive_pool(ll_features)
            ll_features = ll_features.view(batch_size, -1)
            
            # Process detail coefficients from all levels with batch processing
            detail_features_list = []
            
            # Batch process detail coefficients for efficiency
            detail_tensors = []
            for level in range(1, self.dwt_levels + 1):
                for subband in ['lh', 'hl', 'hh']:
                    key = f"{subband}_{level}"
                    if key not in dwt_coefficients:
                        raise ValueError(f"Missing detail coefficients ({key})")
                    
                    detail_coeff = dwt_coefficients[key]
                    if len(detail_coeff.shape) != 4:
                        raise ValueError(f"Expected 4D {key} tensor, got shape {detail_coeff.shape}")
                    if detail_coeff.shape[1] != self.input_channels:
                        raise ValueError(f"Expected {self.input_channels} channels in {key}, got {detail_coeff.shape[1]}")
                    
                    # Check for invalid values and handle gracefully
                    if not torch.all(torch.isfinite(detail_coeff)):
                        logger.warning(f"Non-finite values detected in {key} coefficients, clamping...")
                        detail_coeff = torch.clamp(detail_coeff, -1e6, 1e6)
                    
                    # Ensure tensor is contiguous
                    if not detail_coeff.is_contiguous():
                        detail_coeff = detail_coeff.contiguous()
                    
                    detail_tensors.append(detail_coeff)
            
            # Process all detail coefficients in batches for better GPU utilization
            for detail_coeff in detail_tensors:
                # Process detail coefficients with mixed precision
                if detail_coeff.is_cuda and hasattr(torch.cuda.amp, 'autocast'):
                    with torch.cuda.amp.autocast(enabled=self.training):
                        detail_feat = self.detail_conv(detail_coeff)
                else:
                    detail_feat = self.detail_conv(detail_coeff)
                
                # MPS workaround: adaptive pooling with non-divisible sizes crashes on MPS
                if detail_feat.device.type == 'mps':
                    detail_feat = self.adaptive_pool(detail_feat.cpu()).to('mps')
                else:
                    detail_feat = self.adaptive_pool(detail_feat)
                detail_feat = detail_feat.view(batch_size, -1)
                detail_features_list.append(detail_feat)
            
            # Efficient concatenation of detail features
            if detail_features_list:
                detail_features = torch.cat(detail_features_list, dim=1)
            else:
                # Fallback if no detail features (shouldn't happen with valid input)
                detail_features = torch.zeros(batch_size, 0, device=ll_features.device, dtype=ll_features.dtype)
            
            # Combine LL and detail features efficiently
            combined_features = torch.cat([ll_features, detail_features], dim=1)
            
            # Apply feature fusion with gradient checkpointing for memory efficiency
            if self.training and combined_features.requires_grad:
                frequency_features = torch.utils.checkpoint.checkpoint(
                    self.feature_fusion, combined_features, use_reentrant=False
                )
            else:
                frequency_features = self.feature_fusion(combined_features)
            
            # L2 normalize features for stable training with numerical stability
            frequency_features = F.normalize(frequency_features, p=2, dim=1, eps=1e-8)
            
            # Final validation
            if not torch.all(torch.isfinite(frequency_features)):
                logger.error("Non-finite values in output features")
                raise RuntimeError("Non-finite values in output features")
            
            # Validate output shape
            expected_shape = (batch_size, self.feature_dim)
            if frequency_features.shape != expected_shape:
                raise RuntimeError(f"Expected output shape {expected_shape}, got {frequency_features.shape}")
            
            return frequency_features
            
        except Exception as e:
            logger.error(f"DWTLayer forward pass failed: {str(e)}")
            raise RuntimeError(f"DWTLayer forward pass failed: {str(e)}")
    
    def get_memory_usage(self, batch_size: int = 1, input_size: Tuple[int, int] = (224, 224)) -> Dict[str, float]:
        """
        Estimate memory usage for given batch size and input dimensions.
        
        Args:
            batch_size: Batch size for memory estimation
            input_size: Input image dimensions (H, W)
            
        Returns:
            dict: Memory usage information in MB
        """
        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 ** 2)
        
        # Estimate activation memory for DWT coefficients
        h, w = input_size
        
        # Approximation coefficients memory (LL at deepest level)
        ll_h, ll_w = h // (2 ** self.dwt_levels), w // (2 ** self.dwt_levels)
        ll_memory = batch_size * self.input_channels * ll_h * ll_w * 4 / (1024 ** 2)
        
        # Detail coefficients memory (LH, HL, HH at each level)
        detail_memory = 0
        for level in range(1, self.dwt_levels + 1):
            level_h, level_w = h // (2 ** level), w // (2 ** level)
            level_memory = batch_size * self.input_channels * level_h * level_w * 4 * 3  # 3 subbands
            detail_memory += level_memory / (1024 ** 2)
        
        # Feature memory
        feature_memory = batch_size * self.feature_dim * 4 / (1024 ** 2)
        
        # Intermediate feature memory (conv outputs)
        ll_conv_memory = batch_size * 128 * 8 * 8 * 4 / (1024 ** 2)  # LL conv output
        detail_conv_memory = batch_size * 64 * 8 * 8 * 4 * 3 * self.dwt_levels / (1024 ** 2)  # Detail conv outputs
        
        total_memory = param_memory + ll_memory + detail_memory + feature_memory + ll_conv_memory + detail_conv_memory
        
        return {
            'parameter_memory_mb': param_memory,
            'll_memory_mb': ll_memory,
            'detail_memory_mb': detail_memory,
            'feature_memory_mb': feature_memory,
            'll_conv_memory_mb': ll_conv_memory,
            'detail_conv_memory_mb': detail_conv_memory,
            'total_estimated_mb': total_memory,
            'batch_size': batch_size,
            'input_size': input_size
        }


class FrequencyStream(nn.Module):
    """
    DWT-based frequency feature extractor for deepfake detection.
    
    This class implements the frequency stream of the CRAFT-DF architecture, using
    multi-level DWT coefficients to extract frequency-domain artifacts that are
    characteristic of deepfake generation processes.
    
    The frequency stream architecture:
    1. DWT coefficient processing through custom convolutional layers
    2. Multi-scale feature extraction from different decomposition levels
    3. Frequency artifact detection through learned filters
    4. Feature fusion and normalization for stable training
    
    Args:
        input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
        dwt_levels (int): Number of DWT decomposition levels
        feature_dim (int): Output feature dimension
        hidden_dim (int): Hidden layer dimension for feature processing
        dropout_rate (float): Dropout rate for regularization
        use_attention (bool): Whether to use self-attention for feature refinement
        
    Attributes:
        dwt_layer (DWTLayer): Core DWT processing layer
        feature_refiner (nn.Module): Additional feature processing layers
        attention (nn.Module): Optional self-attention mechanism
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        dwt_levels: int = 3,
        feature_dim: int = 512,
        hidden_dim: int = 1024,
        dropout_rate: float = 0.1,
        use_attention: bool = True
    ) -> None:
        super(FrequencyStream, self).__init__()
        
        # Validate parameters
        assert input_channels > 0, f"input_channels must be positive, got {input_channels}"
        assert 1 <= dwt_levels <= 6, f"dwt_levels must be between 1 and 6, got {dwt_levels}"
        assert feature_dim > 0, f"feature_dim must be positive, got {feature_dim}"
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"
        assert 0.0 <= dropout_rate <= 1.0, f"dropout_rate must be between 0 and 1, got {dropout_rate}"
        
        self.input_channels = input_channels
        self.dwt_levels = dwt_levels
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # Core DWT processing layer
        self.dwt_layer = DWTLayer(
            input_channels=input_channels,
            dwt_levels=dwt_levels,
            feature_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        # Feature refinement layers
        self.feature_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Optional self-attention for feature refinement
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(feature_dim)
        else:
            self.attention = None
            self.attention_norm = None
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"FrequencyStream initialized: channels={input_channels}, levels={dwt_levels}, "
                   f"feature_dim={feature_dim}, hidden_dim={hidden_dim}, attention={use_attention}")
    
    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, dwt_coefficients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the frequency stream with GPU optimization and error handling.
        
        This method processes DWT coefficients through the frequency stream to extract
        frequency-domain features for deepfake detection. The implementation includes
        several optimizations for GPU performance, numerical stability, and error handling.
        
        The frequency domain analysis theory:
        - DWT coefficients capture frequency information at multiple scales
        - High-frequency detail coefficients often contain deepfake artifacts
        - Learned convolutional filters detect frequency-domain anomalies
        - Self-attention refines features by focusing on discriminative patterns
        
        Args:
            dwt_coefficients: Dictionary containing DWT coefficients from multi-level decomposition
                Expected format:
                - 'll': Approximation coefficients (batch_size, channels, H, W)
                - 'lh_1', 'hl_1', 'hh_1': Level 1 detail coefficients
                - 'lh_2', 'hl_2', 'hh_2': Level 2 detail coefficients
                - ... (up to dwt_levels)
                
        Returns:
            torch.Tensor: Frequency features of shape (batch_size, feature_dim)
                         Features are L2-normalized for stable training
            
        Raises:
            ValueError: If input format is invalid
            RuntimeError: If processing fails due to memory or computation issues
        """
        try:
            # Validate input format with detailed error messages
            if not isinstance(dwt_coefficients, dict):
                raise ValueError("dwt_coefficients must be a dictionary")
            if 'll' not in dwt_coefficients:
                raise ValueError("Missing approximation coefficients (ll)")
            
            batch_size = dwt_coefficients['ll'].shape[0]
            
            # Validate all required coefficients are present
            missing_keys = []
            for level in range(1, self.dwt_levels + 1):
                for subband in ['lh', 'hl', 'hh']:
                    key = f"{subband}_{level}"
                    if key not in dwt_coefficients:
                        missing_keys.append(key)
            
            if missing_keys:
                raise ValueError(f"Missing DWT coefficients: {missing_keys}")
            
            # Check for memory constraints and adjust processing accordingly
            device = dwt_coefficients['ll'].device
            if device.type == 'cuda':
                available_memory = torch.cuda.get_device_properties(device).total_memory
                current_memory = torch.cuda.memory_allocated(device)
                memory_usage_ratio = current_memory / available_memory
                
                if memory_usage_ratio > 0.8:
                    logger.warning(f"High GPU memory usage ({memory_usage_ratio:.2%}), enabling memory optimizations")
                    # Enable gradient checkpointing for memory efficiency
                    use_checkpointing = True
                else:
                    use_checkpointing = False
            else:
                use_checkpointing = False
            
            # Process DWT coefficients through core layer with error handling
            try:
                if use_checkpointing and self.training:
                    dwt_features = torch.utils.checkpoint.checkpoint(
                        self.dwt_layer, dwt_coefficients, use_reentrant=False
                    )
                else:
                    dwt_features = self.dwt_layer(dwt_coefficients)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("GPU out of memory during DWT processing")
                    # Clear cache and retry with gradient checkpointing
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    dwt_features = torch.utils.checkpoint.checkpoint(
                        self.dwt_layer, dwt_coefficients, use_reentrant=False
                    )
                else:
                    raise e
            
            # Refine features through additional layers with mixed precision
            if dwt_features.is_cuda and hasattr(torch.cuda.amp, 'autocast'):
                with torch.cuda.amp.autocast(enabled=self.training):
                    refined_features = self.feature_refiner(dwt_features)
            else:
                refined_features = self.feature_refiner(dwt_features)
            
            # Apply self-attention if enabled with optimizations
            if self.use_attention and self.attention is not None:
                # Reshape for attention (add sequence dimension)
                attention_input = refined_features.unsqueeze(1)  # (batch_size, 1, feature_dim)
                
                try:
                    # Apply multi-head self-attention with mixed precision
                    if attention_input.is_cuda and hasattr(torch.cuda.amp, 'autocast'):
                        with torch.cuda.amp.autocast(enabled=self.training):
                            attended_features, attention_weights = self.attention(
                                attention_input, attention_input, attention_input
                            )
                    else:
                        attended_features, attention_weights = self.attention(
                            attention_input, attention_input, attention_input
                        )
                    
                    # Remove sequence dimension and apply residual connection
                    attended_features = attended_features.squeeze(1)  # (batch_size, feature_dim)
                    refined_features = self.attention_norm(refined_features + attended_features)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning("GPU out of memory during attention, skipping attention mechanism")
                        # Skip attention if out of memory
                        pass
                    else:
                        raise e
            
            # Final output projection with error handling
            try:
                frequency_features = self.output_projection(refined_features)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("GPU out of memory during output projection")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    frequency_features = self.output_projection(refined_features)
                else:
                    raise e
            
            # L2 normalize features for stable training and better convergence
            frequency_features = F.normalize(frequency_features, p=2, dim=1, eps=1e-8)
            
            # Validate output for numerical stability
            if not torch.all(torch.isfinite(frequency_features)):
                logger.error("Non-finite values detected in frequency features")
                # Clamp extreme values and re-normalize
                frequency_features = torch.clamp(frequency_features, -10.0, 10.0)
                frequency_features = F.normalize(frequency_features, p=2, dim=1, eps=1e-8)
                
                if not torch.all(torch.isfinite(frequency_features)):
                    raise RuntimeError("Unable to resolve non-finite values in frequency features")
            
            # Validate output shape
            expected_shape = (batch_size, self.feature_dim)
            if frequency_features.shape != expected_shape:
                raise RuntimeError(f"Expected output shape {expected_shape}, got {frequency_features.shape}")
            
            return frequency_features
            
        except Exception as e:
            logger.error(f"FrequencyStream forward pass failed: {str(e)}")
            raise RuntimeError(f"FrequencyStream forward pass failed: {str(e)}")
    
    def enable_mixed_precision(self) -> None:
        """
        Enable mixed precision training optimizations for better GPU performance.
        
        This method configures the model for mixed precision training,
        which can significantly speed up training on modern GPUs while
        maintaining numerical stability.
        """
        # Convert appropriate layers to half precision
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                # Keep normalization layers in float32 for stability
                module.float()
            elif isinstance(module, nn.MultiheadAttention):
                # Keep attention layers in float32 for stability
                module.float()
        
        logger.info("Mixed precision optimizations enabled for FrequencyStream")
    
    def optimize_for_throughput(self) -> None:
        """
        Apply optimizations specifically for high throughput inference.
        
        This method applies several optimizations:
        - Fuses batch normalization layers
        - Optimizes memory layout
        - Prepares for potential quantization
        - Disables gradient computation
        """
        self.eval()
        
        # Disable gradient computation for all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Optimize memory layout for all parameters
        for module in self.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data = module.weight.data.contiguous()
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.contiguous()
        
        # Try to fuse batch normalization layers
        try:
            # Fuse conv + bn layers in DWT layer
            if hasattr(self.dwt_layer, 'll_conv'):
                torch.quantization.fuse_modules(
                    self.dwt_layer.ll_conv,
                    [['0', '1']],  # Conv2d + BatchNorm2d
                    inplace=True
                )
            if hasattr(self.dwt_layer, 'detail_conv'):
                torch.quantization.fuse_modules(
                    self.dwt_layer.detail_conv,
                    [['0', '1'], ['2', '3']],  # Conv2d + BatchNorm2d pairs
                    inplace=True
                )
            logger.info("Successfully fused batch normalization layers")
        except Exception as e:
            logger.warning(f"Could not fuse batch normalization layers: {e}")
        
        logger.info("FrequencyStream optimized for high throughput inference")
    
    def benchmark_performance(
        self, 
        dwt_coefficients: Dict[str, torch.Tensor], 
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark the performance of the frequency stream.
        
        Args:
            dwt_coefficients: Sample DWT coefficients for benchmarking
            num_iterations: Number of iterations for benchmarking
            warmup_iterations: Number of warmup iterations
            
        Returns:
            dict: Performance metrics including timing and throughput
        """
        import time
        
        self.eval()
        device = next(self.parameters()).device
        
        # Move coefficients to device
        dwt_coeffs_device = {}
        for key, tensor in dwt_coefficients.items():
            dwt_coeffs_device[key] = tensor.to(device)
        
        batch_size = dwt_coeffs_device['ll'].shape[0]
        
        # Warmup iterations
        logger.info(f"Warming up with {warmup_iterations} iterations...")
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = self(dwt_coeffs_device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark iterations
        logger.info(f"Benchmarking with {num_iterations} iterations...")
        times = []
        
        for i in range(num_iterations):
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()
            
            with torch.no_grad():
                output = self(dwt_coeffs_device)
            
            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                iteration_time = start_event.elapsed_time(end_event)  # milliseconds
            else:
                end_time = time.time()
                iteration_time = (end_time - start_time) * 1000  # milliseconds
            
            times.append(iteration_time)
        
        # Calculate statistics
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        median_time = np.median(times)
        
        # Calculate throughput
        throughput = (batch_size * 1000) / mean_time  # samples per second
        
        # Memory usage
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
            memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB
        else:
            memory_allocated = None
            memory_reserved = None
        
        results = {
            'batch_size': batch_size,
            'num_iterations': num_iterations,
            'mean_time_ms': float(mean_time),
            'std_time_ms': float(std_time),
            'min_time_ms': float(min_time),
            'max_time_ms': float(max_time),
            'median_time_ms': float(median_time),
            'throughput_samples_per_sec': float(throughput),
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'device': str(device),
            'output_shape': tuple(output.shape),
            'feature_dim': self.feature_dim,
            'dwt_levels': self.dwt_levels,
            'use_attention': self.use_attention
        }
        
        logger.info(f"Benchmark results: {throughput:.2f} samples/sec, {mean_time:.2f}±{std_time:.2f}ms per batch")
        
        return results
    
    def profile_memory_usage(self, dwt_coefficients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Profile detailed memory usage during forward pass.
        
        Args:
            dwt_coefficients: Sample DWT coefficients for profiling
            
        Returns:
            dict: Detailed memory usage information
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, memory profiling limited")
            return {'error': 'CUDA not available'}
        
        device = next(self.parameters()).device
        if device.type != 'cuda':
            device = torch.device('cuda')
            self.to(device)
            dwt_coeffs_device = {}
            for key, tensor in dwt_coefficients.items():
                dwt_coeffs_device[key] = tensor.to(device)
        else:
            dwt_coeffs_device = dwt_coefficients
        
        # Clear cache and measure baseline
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        
        # Measure memory during forward pass
        with torch.no_grad():
            output = self(dwt_coeffs_device)
        
        peak_memory = torch.cuda.max_memory_allocated(device)
        final_memory = torch.cuda.memory_allocated(device)
        
        # Calculate memory usage
        forward_memory = final_memory - baseline_memory
        peak_additional = peak_memory - baseline_memory
        
        # Get model parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters())
        
        # Calculate input memory
        input_memory = sum(tensor.numel() * tensor.element_size() for tensor in dwt_coeffs_device.values())
        
        # Calculate output memory
        output_memory = output.numel() * output.element_size()
        
        results = {
            'baseline_memory_mb': baseline_memory / (1024 ** 2),
            'final_memory_mb': final_memory / (1024 ** 2),
            'peak_memory_mb': peak_memory / (1024 ** 2),
            'forward_memory_mb': forward_memory / (1024 ** 2),
            'peak_additional_mb': peak_additional / (1024 ** 2),
            'parameter_memory_mb': param_memory / (1024 ** 2),
            'input_memory_mb': input_memory / (1024 ** 2),
            'output_memory_mb': output_memory / (1024 ** 2),
            'batch_size': dwt_coeffs_device['ll'].shape[0],
            'device': str(device)
        }
        
        logger.info(f"Memory usage: {forward_memory / (1024 ** 2):.2f}MB forward, "
                   f"{peak_additional / (1024 ** 2):.2f}MB peak additional")
        
        return results
    
    def get_attention_weights(self, dwt_coefficients: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Extract attention weights for visualization and interpretability.
        
        Args:
            dwt_coefficients: DWT coefficients dictionary
            
        Returns:
            Attention weights tensor if attention is enabled, None otherwise
        """
        if not self.use_attention or self.attention is None:
            return None
        
        with torch.no_grad():
            # Process through DWT layer and feature refiner
            dwt_features = self.dwt_layer(dwt_coefficients)
            refined_features = self.feature_refiner(dwt_features)
            
            # Get attention weights
            attention_input = refined_features.unsqueeze(1)
            _, attention_weights = self.attention(
                attention_input, attention_input, attention_input
            )
            
            return attention_weights
    
    def get_feature_maps(self, dwt_coefficients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps for analysis and debugging.
        
        Args:
            dwt_coefficients: DWT coefficients dictionary
            
        Returns:
            Dictionary containing intermediate feature maps
        """
        feature_maps = {}
        
        with torch.no_grad():
            # DWT layer features
            dwt_features = self.dwt_layer(dwt_coefficients)
            feature_maps['dwt_features'] = dwt_features
            
            # Refined features
            refined_features = self.feature_refiner(dwt_features)
            feature_maps['refined_features'] = refined_features
            
            # Attention features (if available)
            if self.use_attention and self.attention is not None:
                attention_input = refined_features.unsqueeze(1)
                attended_features, _ = self.attention(
                    attention_input, attention_input, attention_input
                )
                feature_maps['attended_features'] = attended_features.squeeze(1)
        
        return feature_maps
    
    def get_trainable_parameters(self) -> int:
        """
        Get the number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get model information for logging and debugging.
        
        Returns:
            dict: Model information including parameter counts and configuration
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.get_trainable_parameters()
        
        return {
            'model_name': 'FrequencyStream',
            'input_channels': self.input_channels,
            'dwt_levels': self.dwt_levels,
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'use_attention': self.use_attention,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate
        }
    
    def optimize_for_inference(self) -> None:
        """
        Optimize model for inference by applying various optimizations.
        
        This method applies several optimizations for faster inference:
        - Sets model to evaluation mode
        - Optimizes memory layout
        - Prepares for potential quantization
        """
        self.eval()
        
        # Optimize memory layout
        for module in self.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data = module.weight.data.contiguous()
        
        logger.info("FrequencyStream optimized for inference")
    
    def get_memory_usage(self, batch_size: int = 1, input_size: Tuple[int, int] = (224, 224)) -> Dict[str, float]:
        """
        Estimate memory usage for given batch size and input dimensions.
        
        Args:
            batch_size: Batch size for memory estimation
            input_size: Input image dimensions (H, W)
            
        Returns:
            dict: Memory usage information in MB
        """
        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 ** 2)
        
        # Estimate activation memory for DWT coefficients
        h, w = input_size
        
        # Approximation coefficients memory (LL at deepest level)
        ll_h, ll_w = h // (2 ** self.dwt_levels), w // (2 ** self.dwt_levels)
        ll_memory = batch_size * self.input_channels * ll_h * ll_w * 4 / (1024 ** 2)
        
        # Detail coefficients memory (LH, HL, HH at each level)
        detail_memory = 0
        for level in range(1, self.dwt_levels + 1):
            level_h, level_w = h // (2 ** level), w // (2 ** level)
            level_memory = batch_size * self.input_channels * level_h * level_w * 4 * 3  # 3 subbands
            detail_memory += level_memory / (1024 ** 2)
        
        # Feature memory
        feature_memory = batch_size * self.feature_dim * 4 / (1024 ** 2)
        
        total_memory = param_memory + ll_memory + detail_memory + feature_memory
        
        return {
            'parameter_memory_mb': param_memory,
            'll_memory_mb': ll_memory,
            'detail_memory_mb': detail_memory,
            'feature_memory_mb': feature_memory,
            'total_estimated_mb': total_memory,
            'batch_size': batch_size,
            'input_size': input_size
        }
    
    def profile_forward_pass(self, dwt_coefficients: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """
        Profile the forward pass to identify performance bottlenecks.
        
        Args:
            dwt_coefficients: DWT coefficients for profiling
            
        Returns:
            dict: Profiling results including timing information
        """
        import time
        
        self.eval()
        device = next(self.parameters()).device
        
        # Move coefficients to device
        dwt_coeffs_device = {}
        for key, tensor in dwt_coefficients.items():
            dwt_coeffs_device[key] = tensor.to(device)
        
        # Warm up GPU
        for _ in range(5):
            with torch.no_grad():
                _ = self(dwt_coeffs_device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Profile forward pass
        start_time = time.time()
        
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        with torch.no_grad():
            output = self(dwt_coeffs_device)
        
        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            gpu_time = start_event.elapsed_time(end_event)  # milliseconds
        else:
            gpu_time = None
        
        end_time = time.time()
        cpu_time = (end_time - start_time) * 1000  # milliseconds
        
        batch_size = dwt_coeffs_device['ll'].shape[0]
        throughput = batch_size / (cpu_time / 1000)  # samples per second
        
        return {
            'batch_size': batch_size,
            'cpu_time_ms': cpu_time,
            'gpu_time_ms': gpu_time,
            'throughput_samples_per_sec': throughput,
            'memory_allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2) if device.type == 'cuda' else None,
            'output_shape': tuple(output.shape),
            'dwt_levels': self.dwt_levels,
            'feature_dim': self.feature_dim
        }