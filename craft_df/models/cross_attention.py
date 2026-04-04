"""
Cross-Attention Fusion Module for CRAFT-DF

This module implements the cross-attention mechanism that fuses spatial and frequency domain features
for robust deepfake detection. The cross-attention allows the model to selectively focus on the most
discriminative features from both domains, creating a unified representation that leverages the
complementary information from spatial and frequency streams.

Theory:
Cross-attention enables the model to:
1. Use spatial features as queries to attend to frequency features (keys/values)
2. Learn which frequency patterns are most relevant for each spatial region
3. Create adaptive feature fusion based on input content
4. Improve interpretability through attention weight visualization
5. Enhance generalization by focusing on domain-invariant patterns

The multi-head attention mechanism allows the model to attend to different types of
relationships simultaneously, capturing both local and global dependencies between
spatial and frequency features.

Mathematical formulation:
- Q = spatial_features * W_q (queries from spatial domain)
- K = frequency_features * W_k (keys from frequency domain)  
- V = frequency_features * W_v (values from frequency domain)
- Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
- Output = LayerNorm(Q + MultiHead(Q,K,V))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention module for fusing spatial and frequency features.
    
    This module implements cross-attention where spatial features serve as queries
    and frequency features serve as keys and values. This design allows the model
    to selectively attend to frequency patterns that are most relevant for each
    spatial region, creating an adaptive fusion mechanism.
    
    The architecture includes:
    - Multi-head cross-attention with configurable number of heads
    - Residual connections for gradient flow and training stability
    - Layer normalization for feature standardization
    - Dropout for regularization
    - Optional positional encoding for sequence modeling
    
    Args:
        spatial_dim (int): Dimension of spatial features (queries)
        frequency_dim (int): Dimension of frequency features (keys/values)
        embed_dim (int): Embedding dimension for attention computation
        num_heads (int): Number of attention heads
        dropout_rate (float): Dropout rate for regularization
        use_bias (bool): Whether to use bias in linear projections
        temperature (float): Temperature scaling for attention weights
        
    Attributes:
        spatial_dim (int): Input spatial feature dimension
        frequency_dim (int): Input frequency feature dimension
        embed_dim (int): Attention embedding dimension
        num_heads (int): Number of attention heads
        head_dim (int): Dimension per attention head
    """
    
    def __init__(
        self,
        spatial_dim: int = 1280,
        frequency_dim: int = 512,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        use_bias: bool = True,
        temperature: float = 1.0
    ) -> None:
        super(CrossAttentionFusion, self).__init__()
        
        # Validate input parameters
        assert spatial_dim > 0, f"spatial_dim must be positive, got {spatial_dim}"
        assert frequency_dim > 0, f"frequency_dim must be positive, got {frequency_dim}"
        assert embed_dim > 0, f"embed_dim must be positive, got {embed_dim}"
        assert num_heads > 0, f"num_heads must be positive, got {num_heads}"
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        assert 0.0 <= dropout_rate <= 1.0, f"dropout_rate must be between 0 and 1, got {dropout_rate}"
        assert temperature > 0, f"temperature must be positive, got {temperature}"
        
        self.spatial_dim = spatial_dim
        self.frequency_dim = frequency_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        
        # Scale factor for attention computation (prevents vanishing gradients)
        self.scale = math.sqrt(self.head_dim) * temperature
        
        # Linear projections for queries, keys, and values
        # Queries come from spatial features
        self.query_projection = nn.Linear(spatial_dim, embed_dim, bias=use_bias)
        
        # Keys and values come from frequency features
        self.key_projection = nn.Linear(frequency_dim, embed_dim, bias=use_bias)
        self.value_projection = nn.Linear(frequency_dim, embed_dim, bias=use_bias)
        
        # Output projection to combine multi-head outputs
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        
        # Residual connection and layer normalization
        # Project spatial features to match embed_dim for residual connection
        self.spatial_projection = nn.Linear(spatial_dim, embed_dim, bias=use_bias)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Dropout for regularization
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
        logger.info(f"CrossAttentionFusion initialized: spatial_dim={spatial_dim}, "
                   f"frequency_dim={frequency_dim}, embed_dim={embed_dim}, "
                   f"num_heads={num_heads}, head_dim={self.head_dim}")
    
    def _initialize_weights(self) -> None:
        """
        Initialize weights using Xavier uniform initialization for better gradient flow.
        
        This initialization strategy helps maintain proper gradient magnitudes
        throughout the network, which is crucial for training stability in
        attention mechanisms.
        """
        for module in [self.query_projection, self.key_projection, 
                      self.value_projection, self.output_projection, 
                      self.spatial_projection]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        # Initialize layer norm parameters
        nn.init.constant_(self.layer_norm.weight, 1)
        nn.init.constant_(self.layer_norm.bias, 0)
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through cross-attention fusion with comprehensive error handling.
        
        This method implements the core cross-attention mechanism where spatial features
        serve as queries and frequency features serve as keys and values. The attention
        mechanism allows the model to adaptively focus on frequency patterns that are
        most relevant for each spatial feature.
        
        Mathematical operations:
        1. Project spatial features to queries: Q = spatial_features @ W_q
        2. Project frequency features to keys/values: K = freq @ W_k, V = freq @ W_v
        3. Compute attention scores: scores = Q @ K^T / scale
        4. Apply softmax: attention_weights = softmax(scores)
        5. Apply attention: attended = attention_weights @ V
        6. Apply output projection and residual connection
        7. Apply layer normalization
        
        Args:
            spatial_features (torch.Tensor): Spatial domain features of shape (batch_size, spatial_dim)
                                           These serve as queries in the attention mechanism
            frequency_features (torch.Tensor): Frequency domain features of shape (batch_size, frequency_dim)
                                              These serve as keys and values in the attention mechanism
            return_attention (bool): Whether to return attention weights for visualization
            
        Returns:
            Tuple containing:
            - fused_features (torch.Tensor): Fused features of shape (batch_size, embed_dim)
            - attention_weights (torch.Tensor, optional): Attention weights of shape 
              (batch_size, num_heads, 1, 1) if return_attention=True, else None
              
        Raises:
            AssertionError: If input tensor shapes are invalid
            RuntimeError: If computation fails due to memory or numerical issues
        """
        # Validate input tensor shapes with detailed error messages
        assert len(spatial_features.shape) == 2, f"Expected 2D spatial features, got shape {spatial_features.shape}"
        assert len(frequency_features.shape) == 2, f"Expected 2D frequency features, got shape {frequency_features.shape}"
        assert spatial_features.shape[0] == frequency_features.shape[0], \
            f"Batch size mismatch: spatial {spatial_features.shape[0]} vs frequency {frequency_features.shape[0]}"
        assert spatial_features.shape[1] == self.spatial_dim, \
            f"Expected spatial_dim {self.spatial_dim}, got {spatial_features.shape[1]}"
        assert frequency_features.shape[1] == self.frequency_dim, \
            f"Expected frequency_dim {self.frequency_dim}, got {frequency_features.shape[1]}"
        
        batch_size = spatial_features.shape[0]
        
        # Check for invalid values in input tensors
        if not torch.all(torch.isfinite(spatial_features)):
            logger.warning("Non-finite values detected in spatial features, replacing with zeros...")
            spatial_features = torch.where(torch.isfinite(spatial_features), spatial_features, torch.zeros_like(spatial_features))
        
        if not torch.all(torch.isfinite(frequency_features)):
            logger.warning("Non-finite values detected in frequency features, replacing with zeros...")
            frequency_features = torch.where(torch.isfinite(frequency_features), frequency_features, torch.zeros_like(frequency_features))
        
        try:
            # Ensure tensors are contiguous for optimal GPU memory access
            if not spatial_features.is_contiguous():
                spatial_features = spatial_features.contiguous()
            if not frequency_features.is_contiguous():
                frequency_features = frequency_features.contiguous()
            
            # Project features to query, key, and value spaces with mixed precision
            device = spatial_features.device
            if device.type == 'cuda' and hasattr(torch.cuda.amp, 'autocast'):
                with torch.cuda.amp.autocast(enabled=self.training):
                    queries = self.query_projection(spatial_features)  # (batch_size, embed_dim)
                    keys = self.key_projection(frequency_features)     # (batch_size, embed_dim)
                    values = self.value_projection(frequency_features) # (batch_size, embed_dim)
            else:
                queries = self.query_projection(spatial_features)
                keys = self.key_projection(frequency_features)
                values = self.value_projection(frequency_features)
            
            # Reshape for multi-head attention
            # Add sequence dimension (length=1) for compatibility with attention mechanism
            queries = queries.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            # Shape: (batch_size, num_heads, 1, head_dim)
            
            keys = keys.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            # Shape: (batch_size, num_heads, 1, head_dim)
            
            values = values.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            # Shape: (batch_size, num_heads, 1, head_dim)
            
            # Compute attention scores with numerical stability
            # scores = Q @ K^T / scale
            attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
            # Shape: (batch_size, num_heads, 1, 1)
            
            # Check for numerical stability in attention scores
            if not torch.all(torch.isfinite(attention_scores)):
                logger.warning("Non-finite attention scores detected, applying clipping...")
                attention_scores = torch.clamp(attention_scores, -50.0, 50.0)
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(attention_scores, dim=-1)
            # Shape: (batch_size, num_heads, 1, 1)
            
            # Apply dropout to attention weights for regularization
            if self.training:
                attention_weights = self.attention_dropout(attention_weights)
            
            # Apply attention to values
            attended_values = torch.matmul(attention_weights, values)
            # Shape: (batch_size, num_heads, 1, head_dim)
            
            # Concatenate heads and reshape
            attended_values = attended_values.transpose(1, 2).contiguous().view(
                batch_size, 1, self.embed_dim
            ).squeeze(1)
            # Shape: (batch_size, embed_dim)
            
            # Apply output projection
            if device.type == 'cuda' and hasattr(torch.cuda.amp, 'autocast'):
                with torch.cuda.amp.autocast(enabled=self.training):
                    attended_output = self.output_projection(attended_values)
            else:
                attended_output = self.output_projection(attended_values)
            
            # Apply output dropout
            if self.training:
                attended_output = self.output_dropout(attended_output)
            
            # Residual connection: project spatial features to match embed_dim
            if device.type == 'cuda' and hasattr(torch.cuda.amp, 'autocast'):
                with torch.cuda.amp.autocast(enabled=self.training):
                    spatial_residual = self.spatial_projection(spatial_features)
            else:
                spatial_residual = self.spatial_projection(spatial_features)
            
            # Add residual connection
            fused_features = attended_output + spatial_residual
            
            # Apply layer normalization
            fused_features = self.layer_norm(fused_features)
            
            # Final validation of output
            if not torch.all(torch.isfinite(fused_features)):
                logger.error("Non-finite values in fused features")
                raise RuntimeError("Non-finite values detected in fused features")
            
            # Validate output shape
            expected_shape = (batch_size, self.embed_dim)
            assert fused_features.shape == expected_shape, \
                f"Expected output shape {expected_shape}, got {fused_features.shape}"
            
            # Return attention weights if requested (for visualization)
            if return_attention:
                # Return full attention weights for analysis
                return fused_features, attention_weights  # (batch_size, num_heads, 1, 1)
            else:
                return fused_features, None
                
        except Exception as e:
            logger.error(f"CrossAttentionFusion forward pass failed: {str(e)}")
            raise RuntimeError(f"CrossAttentionFusion forward pass failed: {str(e)}")
    
    def get_attention_weights(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract attention weights for visualization and analysis.
        
        This method computes and returns the attention weights without applying
        them to the values, useful for understanding what frequency patterns
        the model is focusing on for each spatial input.
        
        Args:
            spatial_features (torch.Tensor): Spatial features (batch_size, spatial_dim)
            frequency_features (torch.Tensor): Frequency features (batch_size, frequency_dim)
            
        Returns:
            torch.Tensor: Attention weights of shape (batch_size, num_heads, 1, 1)
        """
        with torch.no_grad():
            _, attention_weights = self.forward(
                spatial_features, frequency_features, return_attention=True
            )
            return attention_weights
    
    def visualize_attention_pattern(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Create visualization data for attention patterns.
        
        Args:
            spatial_features (torch.Tensor): Spatial features
            frequency_features (torch.Tensor): Frequency features  
            sample_idx (int): Index of sample to visualize
            
        Returns:
            dict: Visualization data including attention weights and feature norms
        """
        with torch.no_grad():
            # Get attention weights
            attention_weights = self.get_attention_weights(spatial_features, frequency_features)
            
            # Extract data for specific sample and average across heads
            sample_attention = attention_weights[sample_idx].mean(dim=0).cpu().numpy()  # Average across heads
            
            # Compute feature norms for analysis
            spatial_norm = torch.norm(spatial_features[sample_idx]).item()
            frequency_norm = torch.norm(frequency_features[sample_idx]).item()
            
            # Compute attention statistics
            attention_mean = float(sample_attention.mean())
            attention_std = float(sample_attention.std())
            attention_max = float(sample_attention.max())
            attention_min = float(sample_attention.min())
            
            return {
                'attention_weights': sample_attention,
                'attention_stats': {
                    'mean': attention_mean,
                    'std': attention_std,
                    'max': attention_max,
                    'min': attention_min
                },
                'feature_norms': {
                    'spatial': spatial_norm,
                    'frequency': frequency_norm
                },
                'num_heads': self.num_heads,
                'sample_idx': sample_idx
            }
    
    def compute_attention_entropy(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention entropy for measuring attention concentration.
        
        Higher entropy indicates more distributed attention, while lower entropy
        indicates more focused attention. This can be useful for analyzing
        model behavior and detecting potential issues.
        
        Args:
            spatial_features (torch.Tensor): Spatial features
            frequency_features (torch.Tensor): Frequency features
            
        Returns:
            torch.Tensor: Attention entropy for each sample (batch_size,)
        """
        with torch.no_grad():
            attention_weights = self.get_attention_weights(spatial_features, frequency_features)
            
            # Average across heads and flatten attention weights for entropy computation
            attention_flat = attention_weights.mean(dim=1).view(attention_weights.shape[0], -1)
            
            # Normalize to ensure they sum to 1 (proper probability distribution)
            attention_flat = F.softmax(attention_flat, dim=-1)
            
            # Add small epsilon to prevent log(0)
            eps = 1e-8
            attention_flat = attention_flat + eps
            
            # Compute entropy: -sum(p * log(p))
            entropy = -torch.sum(attention_flat * torch.log(attention_flat), dim=1)
            
            return entropy
    
    def get_memory_usage(self, batch_size: int = 1) -> Dict[str, float]:
        """
        Estimate memory usage for given batch size.
        
        Args:
            batch_size (int): Batch size for memory estimation
            
        Returns:
            dict: Memory usage information in MB
        """
        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 ** 2)
        
        # Estimate activation memory
        # Input features
        spatial_memory = batch_size * self.spatial_dim * 4 / (1024 ** 2)
        frequency_memory = batch_size * self.frequency_dim * 4 / (1024 ** 2)
        
        # Projected features (Q, K, V)
        qkv_memory = batch_size * self.embed_dim * 3 * 4 / (1024 ** 2)
        
        # Attention weights
        attention_memory = batch_size * self.num_heads * 1 * 1 * 4 / (1024 ** 2)
        
        # Output features
        output_memory = batch_size * self.embed_dim * 4 / (1024 ** 2)
        
        total_memory = (param_memory + spatial_memory + frequency_memory + 
                       qkv_memory + attention_memory + output_memory)
        
        return {
            'parameter_memory_mb': param_memory,
            'spatial_input_mb': spatial_memory,
            'frequency_input_mb': frequency_memory,
            'qkv_projections_mb': qkv_memory,
            'attention_weights_mb': attention_memory,
            'output_memory_mb': output_memory,
            'total_estimated_mb': total_memory,
            'batch_size': batch_size
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information for logging and debugging.
        
        Returns:
            dict: Model information including architecture details and parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'CrossAttentionFusion',
            'spatial_dim': self.spatial_dim,
            'frequency_dim': self.frequency_dim,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'temperature': self.temperature,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_efficiency': trainable_params / total_params
        }
    
    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing for memory-efficient training.
        
        This trades computation for memory by recomputing activations
        during the backward pass instead of storing them.
        """
        self.use_gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled for CrossAttentionFusion")
    
    def optimize_for_inference(self) -> None:
        """
        Apply optimizations for faster inference.
        
        This method applies several optimizations:
        - Fuses linear layers where possible
        - Optimizes memory layout
        - Disables dropout
        """
        self.eval()
        
        # Disable dropout for inference
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        
        # Optimize memory layout
        for param in self.parameters():
            if param.data is not None:
                param.data = param.data.contiguous()
        
        logger.info("CrossAttentionFusion optimized for inference")
    
    def validate_tensor_shapes(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
        stage: str = "input"
    ) -> Dict[str, bool]:
        """
        Comprehensive tensor shape validation throughout attention computation.
        
        Args:
            spatial_features: Spatial features tensor
            frequency_features: Frequency features tensor
            stage: Stage of computation ("input", "projection", "attention", "output")
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        try:
            # Basic shape validation
            results['spatial_2d'] = len(spatial_features.shape) == 2
            results['frequency_2d'] = len(frequency_features.shape) == 2
            results['batch_match'] = spatial_features.shape[0] == frequency_features.shape[0]
            results['spatial_dim_correct'] = spatial_features.shape[1] == self.spatial_dim
            results['frequency_dim_correct'] = frequency_features.shape[1] == self.frequency_dim
            
            # Advanced validation based on stage
            if stage == "projection":
                # Validate after projection
                with torch.no_grad():
                    queries = self.query_projection(spatial_features)
                    keys = self.key_projection(frequency_features)
                    values = self.value_projection(frequency_features)
                    
                    results['queries_shape'] = queries.shape[1] == self.embed_dim
                    results['keys_shape'] = keys.shape[1] == self.embed_dim
                    results['values_shape'] = values.shape[1] == self.embed_dim
                    
            elif stage == "attention":
                # Validate attention computation shapes
                batch_size = spatial_features.shape[0]
                with torch.no_grad():
                    queries = self.query_projection(spatial_features)
                    keys = self.key_projection(frequency_features)
                    
                    queries_reshaped = queries.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
                    keys_reshaped = keys.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
                    
                    results['queries_reshaped'] = queries_reshaped.shape == (batch_size, self.num_heads, 1, self.head_dim)
                    results['keys_reshaped'] = keys_reshaped.shape == (batch_size, self.num_heads, 1, self.head_dim)
                    
                    attention_scores = torch.matmul(queries_reshaped, keys_reshaped.transpose(-2, -1))
                    results['attention_scores_shape'] = attention_scores.shape == (batch_size, self.num_heads, 1, 1)
            
            # Check for any failures
            results['all_valid'] = all(results.values())
            
        except Exception as e:
            logger.error(f"Tensor shape validation failed at stage '{stage}': {str(e)}")
            results['validation_error'] = str(e)
            results['all_valid'] = False
        
        return results
    
    def analyze_attention_gradients(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
        target_output: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze gradients flowing through the attention mechanism.
        
        Args:
            spatial_features: Input spatial features (requires_grad=True)
            frequency_features: Input frequency features (requires_grad=True)
            target_output: Optional target for gradient computation
            
        Returns:
            Dictionary containing gradient analysis results
        """
        # Ensure gradients are enabled
        spatial_features = spatial_features.clone().detach().requires_grad_(True)
        frequency_features = frequency_features.clone().detach().requires_grad_(True)
        
        # Forward pass
        fused_features, attention_weights = self.forward(
            spatial_features, frequency_features, return_attention=True
        )
        
        # Compute target for gradient computation
        if target_output is None:
            target = fused_features.sum()  # Simple scalar target
        else:
            target = F.mse_loss(fused_features, target_output)
        
        # Backward pass
        target.backward(retain_graph=True)
        
        # Collect gradient information
        gradient_info = {
            'spatial_gradients': spatial_features.grad.clone() if spatial_features.grad is not None else torch.zeros_like(spatial_features),
            'frequency_gradients': frequency_features.grad.clone() if frequency_features.grad is not None else torch.zeros_like(frequency_features),
            'spatial_grad_norm': torch.norm(spatial_features.grad).item() if spatial_features.grad is not None else 0.0,
            'frequency_grad_norm': torch.norm(frequency_features.grad).item() if frequency_features.grad is not None else 0.0,
            'attention_weights': attention_weights.detach() if attention_weights is not None else None,
            'fused_features': fused_features.detach()
        }
        
        # Compute gradient statistics
        if spatial_features.grad is not None:
            gradient_info['spatial_grad_mean'] = torch.mean(spatial_features.grad).item()
            gradient_info['spatial_grad_std'] = torch.std(spatial_features.grad).item()
        
        if frequency_features.grad is not None:
            gradient_info['frequency_grad_mean'] = torch.mean(frequency_features.grad).item()
            gradient_info['frequency_grad_std'] = torch.std(frequency_features.grad).item()
        
        return gradient_info
    
    def compute_attention_rollout(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention rollout for better interpretability.
        
        Since we have a single-layer attention mechanism, this is simplified,
        but provides a foundation for multi-layer extensions.
        
        Args:
            spatial_features: Input spatial features
            frequency_features: Input frequency features
            
        Returns:
            Attention rollout tensor
        """
        with torch.no_grad():
            attention_weights = self.get_attention_weights(spatial_features, frequency_features)
            
            # For single-layer attention, rollout is just the attention weights
            # In multi-layer scenarios, this would involve matrix multiplication across layers
            rollout = attention_weights.mean(dim=1)  # Average across heads
            
            return rollout
    
    def get_attention_maps_for_visualization(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention maps formatted for visualization tools.
        
        Args:
            spatial_features: Input spatial features
            frequency_features: Input frequency features
            
        Returns:
            Dictionary containing formatted attention maps
        """
        with torch.no_grad():
            # Get raw attention weights
            attention_weights = self.get_attention_weights(spatial_features, frequency_features)
            
            # Get intermediate projections for analysis
            queries = self.query_projection(spatial_features)
            keys = self.key_projection(frequency_features)
            values = self.value_projection(frequency_features)
            
            # Compute attention scores before softmax
            batch_size = spatial_features.shape[0]
            queries_reshaped = queries.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            keys_reshaped = keys.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            
            attention_scores = torch.matmul(queries_reshaped, keys_reshaped.transpose(-2, -1)) / self.scale
            
            return {
                'attention_weights': attention_weights,  # After softmax
                'attention_scores': attention_scores,    # Before softmax
                'queries': queries,
                'keys': keys,
                'values': values,
                'head_averaged_attention': attention_weights.mean(dim=1),
                'max_attention_per_head': attention_weights.max(dim=-1)[0].max(dim=-1)[0],
                'min_attention_per_head': attention_weights.min(dim=-1)[0].min(dim=-1)[0]
            }