"""
Feature Disentanglement Module for CRAFT-DF

This module implements adversarial feature disentanglement for domain generalization
in deepfake detection. The disentanglement mechanism separates features into
domain-specific and domain-invariant components, promoting better generalization
across different deepfake generation methods and datasets.

Theory:
Feature disentanglement aims to:
1. Separate domain-specific features (e.g., compression artifacts, generation method signatures)
2. Extract domain-invariant features (e.g., fundamental facial inconsistencies)
3. Use adversarial training to ensure domain-invariant features cannot predict domain
4. Improve generalization by focusing on universal deepfake indicators
5. Reduce overfitting to specific generation methods or datasets

The adversarial training involves:
- Feature extractor: Learns to separate domain-specific and domain-invariant features
- Domain classifier: Tries to predict domain from domain-invariant features
- Adversarial loss: Encourages domain-invariant features to be domain-agnostic

Mathematical formulation:
- Let F be the fused features from cross-attention
- Domain-invariant features: F_inv = Encoder_inv(F)
- Domain-specific features: F_spec = Encoder_spec(F)
- Domain prediction: D = DomainClassifier(F_inv)
- Adversarial loss: L_adv = -log(1/K) where K is number of domains (maximum entropy)
- Reconstruction loss: L_rec = ||F - Decoder(F_inv, F_spec)||²
- Total loss: L = L_task + λ_adv * L_adv + λ_rec * L_rec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import math
import logging

logger = logging.getLogger(__name__)


class FeatureDisentanglement(nn.Module):
    """
    Adversarial feature disentanglement module for domain generalization.
    
    This module separates fused features into domain-invariant and domain-specific
    components using adversarial training. The domain-invariant features are
    encouraged to be domain-agnostic through adversarial loss, while domain-specific
    features capture dataset/method-specific characteristics.
    
    The architecture consists of:
    - Domain-invariant encoder: Extracts features that should be consistent across domains
    - Domain-specific encoder: Extracts features specific to generation methods/datasets
    - Domain classifier: Adversarial component that tries to predict domain from invariant features
    - Feature decoder: Reconstructs original features from disentangled components
    - Gradient reversal layer: Implements adversarial training dynamics
    
    Args:
        input_dim (int): Dimension of input fused features
        invariant_dim (int): Dimension of domain-invariant features
        specific_dim (int): Dimension of domain-specific features
        num_domains (int): Number of domains/datasets for adversarial training
        hidden_dim (int): Hidden dimension for internal layers
        num_layers (int): Number of layers in encoders/decoders
        dropout_rate (float): Dropout rate for regularization
        adversarial_weight (float): Weight for adversarial loss component
        reconstruction_weight (float): Weight for reconstruction loss component
        gradient_reversal_lambda (float): Strength of gradient reversal
        
    Attributes:
        input_dim (int): Input feature dimension
        invariant_dim (int): Domain-invariant feature dimension
        specific_dim (int): Domain-specific feature dimension
        num_domains (int): Number of domains for classification
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        invariant_dim: int = 256,
        specific_dim: int = 128,
        num_domains: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        adversarial_weight: float = 0.1,
        reconstruction_weight: float = 0.01,
        gradient_reversal_lambda: float = 1.0
    ) -> None:
        super(FeatureDisentanglement, self).__init__()
        
        # Validate input parameters
        assert input_dim > 0, f"input_dim must be positive, got {input_dim}"
        assert invariant_dim > 0, f"invariant_dim must be positive, got {invariant_dim}"
        assert specific_dim > 0, f"specific_dim must be positive, got {specific_dim}"
        assert num_domains > 1, f"num_domains must be > 1, got {num_domains}"
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"
        assert num_layers >= 1, f"num_layers must be >= 1, got {num_layers}"
        assert 0.0 <= dropout_rate <= 1.0, f"dropout_rate must be between 0 and 1, got {dropout_rate}"
        assert adversarial_weight >= 0, f"adversarial_weight must be non-negative, got {adversarial_weight}"
        assert reconstruction_weight >= 0, f"reconstruction_weight must be non-negative, got {reconstruction_weight}"
        assert gradient_reversal_lambda >= 0, f"gradient_reversal_lambda must be non-negative, got {gradient_reversal_lambda}"
        
        self.input_dim = input_dim
        self.invariant_dim = invariant_dim
        self.specific_dim = specific_dim
        self.num_domains = num_domains
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.adversarial_weight = adversarial_weight
        self.reconstruction_weight = reconstruction_weight
        self.gradient_reversal_lambda = gradient_reversal_lambda
        
        # Domain-invariant feature encoder
        self.invariant_encoder = self._build_encoder(
            input_dim, invariant_dim, hidden_dim, num_layers, dropout_rate
        )
        
        # Domain-specific feature encoder
        self.specific_encoder = self._build_encoder(
            input_dim, specific_dim, hidden_dim, num_layers, dropout_rate
        )
        
        # Domain classifier for adversarial training
        self.domain_classifier = self._build_domain_classifier(
            invariant_dim, num_domains, hidden_dim, dropout_rate
        )
        
        # Feature decoder for reconstruction loss
        self.feature_decoder = self._build_decoder(
            invariant_dim + specific_dim, input_dim, hidden_dim, num_layers, dropout_rate
        )
        
        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer(gradient_reversal_lambda)
        
        # Loss functions
        self.domain_criterion = nn.CrossEntropyLoss()
        self.reconstruction_criterion = nn.MSELoss()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"FeatureDisentanglement initialized: input_dim={input_dim}, "
                   f"invariant_dim={invariant_dim}, specific_dim={specific_dim}, "
                   f"num_domains={num_domains}, adversarial_weight={adversarial_weight}")
    
    def _build_encoder(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float
    ) -> nn.Module:
        """
        Build encoder network with residual connections and layer normalization.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            dropout_rate: Dropout rate
            
        Returns:
            nn.Module: Encoder network
        """
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        ])
        
        # Hidden layers with residual connections
        for i in range(num_layers - 2):
            layers.extend([
                ResidualBlock(hidden_dim, hidden_dim, dropout_rate),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
        
        # Output layer
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        ])
        
        return nn.Sequential(*layers)
    
    def _build_decoder(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float
    ) -> nn.Module:
        """
        Build decoder network for feature reconstruction.
        
        Args:
            input_dim: Input feature dimension (invariant + specific)
            output_dim: Output feature dimension (original features)
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            dropout_rate: Dropout rate
            
        Returns:
            nn.Module: Decoder network
        """
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        ])
        
        # Hidden layers
        for i in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
        
        # Output layer (no activation for reconstruction)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _build_domain_classifier(
        self,
        input_dim: int,
        num_domains: int,
        hidden_dim: int,
        dropout_rate: float
    ) -> nn.Module:
        """
        Build domain classifier for adversarial training.
        
        Args:
            input_dim: Input feature dimension (invariant features)
            num_domains: Number of domains to classify
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout rate
            
        Returns:
            nn.Module: Domain classifier network
        """
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_domains)
        )
    
    def _initialize_weights(self) -> None:
        """
        Initialize weights using Xavier initialization for better gradient flow.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        fused_features: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        return_losses: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through feature disentanglement with optional loss computation.
        
        Args:
            fused_features (torch.Tensor): Input fused features of shape (batch_size, input_dim)
            domain_labels (torch.Tensor, optional): Domain labels for adversarial training
                                                   of shape (batch_size,)
            return_losses (bool): Whether to compute and return loss components
            
        Returns:
            Tuple containing:
            - invariant_features (torch.Tensor): Domain-invariant features (batch_size, invariant_dim)
            - specific_features (torch.Tensor): Domain-specific features (batch_size, specific_dim)
            - losses (Dict[str, torch.Tensor], optional): Loss components if return_losses=True
        """
        # Validate input tensor shapes
        assert len(fused_features.shape) == 2, f"Expected 2D fused features, got shape {fused_features.shape}"
        assert fused_features.shape[1] == self.input_dim, \
            f"Expected input_dim {self.input_dim}, got {fused_features.shape[1]}"
        
        batch_size = fused_features.shape[0]
        
        # Check for invalid values
        if not torch.all(torch.isfinite(fused_features)):
            logger.warning("Non-finite values detected in fused features, replacing with zeros...")
            fused_features = torch.where(torch.isfinite(fused_features), fused_features, torch.zeros_like(fused_features))
        
        try:
            # Ensure tensor is contiguous for optimal memory access
            if not fused_features.is_contiguous():
                fused_features = fused_features.contiguous()
            
            # Extract domain-invariant features
            invariant_features = self.invariant_encoder(fused_features)
            # Shape: (batch_size, invariant_dim)
            
            # Extract domain-specific features
            specific_features = self.specific_encoder(fused_features)
            # Shape: (batch_size, specific_dim)
            
            # Validate intermediate shapes
            assert invariant_features.shape == (batch_size, self.invariant_dim), \
                f"Expected invariant shape {(batch_size, self.invariant_dim)}, got {invariant_features.shape}"
            assert specific_features.shape == (batch_size, self.specific_dim), \
                f"Expected specific shape {(batch_size, self.specific_dim)}, got {specific_features.shape}"
            
            losses = None
            if return_losses:
                losses = self._compute_losses(
                    fused_features, invariant_features, specific_features, domain_labels
                )
            
            return invariant_features, specific_features, losses
            
        except Exception as e:
            logger.error(f"FeatureDisentanglement forward pass failed: {str(e)}")
            raise RuntimeError(f"FeatureDisentanglement forward pass failed: {str(e)}")
    
    def _compute_losses(
        self,
        original_features: torch.Tensor,
        invariant_features: torch.Tensor,
        specific_features: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components for disentanglement training.
        
        Args:
            original_features: Original fused features
            invariant_features: Domain-invariant features
            specific_features: Domain-specific features
            domain_labels: Domain labels for adversarial training
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Reconstruction loss
        concatenated_features = torch.cat([invariant_features, specific_features], dim=1)
        reconstructed_features = self.feature_decoder(concatenated_features)
        reconstruction_loss = self.reconstruction_criterion(reconstructed_features, original_features)
        losses['reconstruction'] = reconstruction_loss
        
        # Adversarial loss (if domain labels provided)
        if domain_labels is not None:
            # Apply gradient reversal to invariant features
            reversed_invariant = self.gradient_reversal(invariant_features)
            
            # Domain classification on reversed features
            domain_predictions = self.domain_classifier(reversed_invariant)
            adversarial_loss = self.domain_criterion(domain_predictions, domain_labels)
            losses['adversarial'] = adversarial_loss
            
            # Compute domain classification accuracy for monitoring
            with torch.no_grad():
                domain_pred_labels = torch.argmax(domain_predictions, dim=1)
                domain_accuracy = (domain_pred_labels == domain_labels).float().mean()
                losses['domain_accuracy'] = domain_accuracy
        
        # Orthogonality loss to encourage feature separation
        orthogonality_loss = self._compute_orthogonality_loss(invariant_features, specific_features)
        losses['orthogonality'] = orthogonality_loss
        
        # Total weighted loss
        total_loss = reconstruction_loss * self.reconstruction_weight
        if 'adversarial' in losses:
            total_loss += adversarial_loss * self.adversarial_weight
        total_loss += orthogonality_loss * 0.01  # Small weight for orthogonality
        
        losses['total'] = total_loss
        
        return losses
    
    def _compute_orthogonality_loss(
        self,
        invariant_features: torch.Tensor,
        specific_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute orthogonality loss to encourage feature separation.
        
        This loss encourages the invariant and specific features to be orthogonal,
        promoting better disentanglement by reducing redundancy between the two
        feature spaces.
        
        Args:
            invariant_features: Domain-invariant features
            specific_features: Domain-specific features
            
        Returns:
            torch.Tensor: Orthogonality loss
        """
        # Normalize features
        invariant_norm = F.normalize(invariant_features, p=2, dim=1)
        specific_norm = F.normalize(specific_features, p=2, dim=1)
        
        # Compute correlation matrix between feature spaces
        # We want this to be close to zero (orthogonal)
        correlation = torch.mm(invariant_norm.t(), specific_norm)
        
        # L2 norm of correlation matrix (should be minimized)
        orthogonality_loss = torch.norm(correlation, p='fro') ** 2
        
        return orthogonality_loss
    
    def get_disentangled_features(
        self,
        fused_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract disentangled features without loss computation.
        
        Args:
            fused_features: Input fused features
            
        Returns:
            Tuple of (invariant_features, specific_features)
        """
        with torch.no_grad():
            invariant_features, specific_features, _ = self.forward(
                fused_features, return_losses=False
            )
            return invariant_features, specific_features
    
    def compute_domain_confusion(
        self,
        invariant_features: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute domain confusion metric to measure disentanglement quality.
        
        Higher confusion (lower accuracy) indicates better domain-invariant features.
        
        Args:
            invariant_features: Domain-invariant features
            domain_labels: True domain labels
            
        Returns:
            torch.Tensor: Domain confusion score (1 - accuracy)
        """
        with torch.no_grad():
            domain_predictions = self.domain_classifier(invariant_features)
            predicted_labels = torch.argmax(domain_predictions, dim=1)
            accuracy = (predicted_labels == domain_labels).float().mean()
            confusion = 1.0 - accuracy
            return confusion
    
    def analyze_feature_separation(
        self,
        fused_features: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze the quality of feature separation.
        
        Args:
            fused_features: Input fused features
            domain_labels: Domain labels
            
        Returns:
            Dictionary of separation metrics
        """
        with torch.no_grad():
            invariant_features, specific_features, _ = self.forward(fused_features)
            
            # Compute feature statistics
            invariant_mean = torch.mean(invariant_features, dim=0)
            specific_mean = torch.mean(specific_features, dim=0)
            invariant_std = torch.std(invariant_features, dim=0)
            specific_std = torch.std(specific_features, dim=0)
            
            # Compute correlation between feature spaces
            invariant_norm = F.normalize(invariant_features, p=2, dim=1)
            specific_norm = F.normalize(specific_features, p=2, dim=1)
            correlation = torch.mm(invariant_norm.t(), specific_norm)
            correlation_magnitude = torch.norm(correlation, p='fro').item()
            
            # Compute domain confusion
            domain_confusion = self.compute_domain_confusion(invariant_features, domain_labels).item()
            
            # Compute reconstruction quality
            concatenated = torch.cat([invariant_features, specific_features], dim=1)
            reconstructed = self.feature_decoder(concatenated)
            reconstruction_error = F.mse_loss(reconstructed, fused_features).item()
            
            return {
                'invariant_mean_norm': torch.norm(invariant_mean).item(),
                'specific_mean_norm': torch.norm(specific_mean).item(),
                'invariant_std_mean': torch.mean(invariant_std).item(),
                'specific_std_mean': torch.mean(specific_std).item(),
                'correlation_magnitude': correlation_magnitude,
                'domain_confusion': domain_confusion,
                'reconstruction_error': reconstruction_error,
                'separation_quality': domain_confusion - correlation_magnitude  # Higher is better
            }


class ResidualBlock(nn.Module):
    """
    Residual block for encoder networks.
    
    Implements a residual connection with layer normalization and dropout
    for improved gradient flow and training stability.
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Skip connection projection if dimensions don't match
        self.skip_connection = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = self.skip_connection(x)
        
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        # Add residual connection
        out = out + residual
        out = self.layer_norm(out)
        
        return out


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for adversarial training.
    
    This layer acts as identity during forward pass but reverses gradients
    during backward pass, enabling adversarial training dynamics.
    
    The gradient reversal is scaled by a lambda parameter that can be
    adjusted during training to control the strength of adversarial training.
    """
    
    def __init__(self, lambda_param: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_param = lambda_param
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (identity) with gradient reversal in backward pass."""
        return GradientReversalFunction.apply(x, self.lambda_param)
    
    def set_lambda(self, lambda_param: float) -> None:
        """Update the gradient reversal strength."""
        self.lambda_param = lambda_param


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Function implementation.
    
    This function implements the gradient reversal operation:
    - Forward pass: y = x (identity)
    - Backward pass: dx = -lambda * dy (gradient reversal)
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_param: float) -> torch.Tensor:
        """Forward pass - identity function."""
        ctx.lambda_param = lambda_param
        return x
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass - reverse and scale gradients."""
        return -ctx.lambda_param * grad_output, None