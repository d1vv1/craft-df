"""
CRAFT-DF Main Model Architecture

This module implements the complete CRAFT-DF (Cross-Attentive Frequency-Temporal 
Disentanglement for Generalizable Deepfake Detection) model architecture. It integrates
all components: spatial stream, frequency stream, cross-attention fusion, and 
feature disentanglement for robust deepfake detection.

The model follows a dual-stream architecture:
1. Spatial Stream: MobileNetV2-based feature extraction from face crops
2. Frequency Stream: DWT-based frequency domain analysis
3. Cross-Attention Fusion: Adaptive fusion of spatial and frequency features
4. Feature Disentanglement: Domain generalization through adversarial training
5. Classification Head: Final deepfake probability prediction

Theory:
The CRAFT-DF architecture addresses key challenges in deepfake detection:
- Multi-domain generalization through feature disentanglement
- Complementary spatial and frequency domain analysis
- Adaptive feature fusion through cross-attention
- Robust training through adversarial domain adaptation
- Efficient inference through optimized architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, List
import logging
import math
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Import CRAFT-DF components
from .spatial_stream import SpatialStream
from .frequency_stream import FrequencyStream
from .cross_attention import CrossAttentionFusion
from .feature_disentanglement import FeatureDisentanglement

logger = logging.getLogger(__name__)


class CRAFTDFModel(pl.LightningModule):
    """
    Complete CRAFT-DF model with integrated feature disentanglement.
    
    This class implements the full CRAFT-DF architecture as a PyTorch Lightning module,
    providing modular training, validation, and inference capabilities. The model
    integrates all components and handles the complex training dynamics including
    adversarial feature disentanglement.
    
    Args:
        spatial_config (Dict[str, Any]): Configuration for spatial stream
        frequency_config (Dict[str, Any]): Configuration for frequency stream
        attention_config (Dict[str, Any]): Configuration for cross-attention
        disentanglement_config (Dict[str, Any]): Configuration for feature disentanglement
        num_classes (int): Number of output classes (2 for binary classification)
        learning_rate (float): Initial learning rate
        weight_decay (float): Weight decay for regularization
        scheduler_type (str): Type of learning rate scheduler
        adversarial_training (bool): Whether to use adversarial training
        domain_adaptation_weight (float): Weight for domain adaptation loss
        
    Attributes:
        spatial_stream (SpatialStream): Spatial feature extractor
        frequency_stream (FrequencyStream): Frequency feature extractor
        cross_attention (CrossAttentionFusion): Cross-attention fusion module
        feature_disentanglement (FeatureDisentanglement): Feature disentanglement module
        classifier (nn.Module): Final classification head
    """
    
    def __init__(
        self,
        spatial_config: Optional[Dict[str, Any]] = None,
        frequency_config: Optional[Dict[str, Any]] = None,
        attention_config: Optional[Dict[str, Any]] = None,
        disentanglement_config: Optional[Dict[str, Any]] = None,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler_type: str = "cosine",
        adversarial_training: bool = True,
        domain_adaptation_weight: float = 0.1,
        classification_weight: float = 1.0,
        reconstruction_weight: float = 0.01,
        orthogonality_weight: float = 0.01,
        gradient_reversal_lambda: float = 1.0,
        warmup_epochs: int = 5,
        **kwargs
    ) -> None:
        super(CRAFTDFModel, self).__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model configuration
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.adversarial_training = adversarial_training
        self.domain_adaptation_weight = domain_adaptation_weight
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight
        self.orthogonality_weight = orthogonality_weight
        self.gradient_reversal_lambda = gradient_reversal_lambda
        self.warmup_epochs = warmup_epochs
        
        # Default configurations
        spatial_config = spatial_config or {
            'pretrained': True,
            'freeze_layers': 10,
            'feature_dim': 1280,
            'dropout_rate': 0.1
        }
        
        frequency_config = frequency_config or {
            'input_channels': 3,
            'dwt_levels': 3,
            'feature_dim': 512,
            'dropout_rate': 0.1
        }
        
        attention_config = attention_config or {
            'spatial_dim': spatial_config['feature_dim'],
            'frequency_dim': frequency_config['feature_dim'],
            'embed_dim': 512,
            'num_heads': 8,
            'dropout_rate': 0.1
        }
        
        disentanglement_config = disentanglement_config or {
            'input_dim': attention_config['embed_dim'],
            'invariant_dim': 256,
            'specific_dim': 128,
            'num_domains': 4,
            'hidden_dim': 512,
            'adversarial_weight': domain_adaptation_weight,
            'reconstruction_weight': reconstruction_weight,
            'gradient_reversal_lambda': gradient_reversal_lambda
        }
        
        # Initialize model components
        self.spatial_stream = SpatialStream(**spatial_config)
        self.frequency_stream = FrequencyStream(**frequency_config)
        self.cross_attention = CrossAttentionFusion(**attention_config)
        
        if adversarial_training:
            self.feature_disentanglement = FeatureDisentanglement(**disentanglement_config)
            classifier_input_dim = disentanglement_config['invariant_dim']
        else:
            self.feature_disentanglement = None
            classifier_input_dim = attention_config['embed_dim']
        
        # Classification head
        self.classifier = self._build_classifier(classifier_input_dim, num_classes)
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        try:
            # Try newer torchmetrics import
            from torchmetrics import Accuracy
            self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
            self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
            self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        except ImportError:
            # Fallback to manual accuracy computation
            self.train_accuracy = None
            self.val_accuracy = None
            self.test_accuracy = None
        
        # Training state — manual optimization only needed for adversarial training
        self.automatic_optimization = not adversarial_training
        
        logger.info(f"CRAFTDFModel initialized with adversarial_training={adversarial_training}")
    
    def _build_classifier(self, input_dim: int, num_classes: int) -> nn.Module:
        """
        Build the final classification head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            
        Returns:
            Classification head module
        """
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 4, num_classes)
        )
    
    def forward(
        self,
        spatial_input: torch.Tensor,
        frequency_input: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete CRAFT-DF model.
        
        Args:
            spatial_input: Face crop images (batch_size, 3, 224, 224)
            frequency_input: DWT coefficients (batch_size, channels, height, width)
            domain_labels: Domain labels for adversarial training (batch_size,)
            return_features: Whether to return intermediate features
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing model outputs and optional intermediate results
        """
        # Validate inputs
        assert len(spatial_input.shape) == 4, f"Expected 4D spatial input, got {spatial_input.shape}"
        
        # Handle frequency input - can be tensor or dictionary
        if isinstance(frequency_input, dict):
            # DWT coefficients format - validate batch size consistency
            if 'll' in frequency_input:
                freq_batch_size = frequency_input['ll'].shape[0]
            else:
                raise ValueError("DWT coefficients dictionary must contain 'll' key")
        else:
            # Tensor format
            assert len(frequency_input.shape) == 4, f"Expected 4D frequency input, got {frequency_input.shape}"
            freq_batch_size = frequency_input.shape[0]
        
        assert spatial_input.shape[0] == freq_batch_size, "Batch size mismatch"
        
        batch_size = spatial_input.shape[0]
        outputs = {}
        
        try:
            # Extract spatial features
            spatial_features = self.spatial_stream(spatial_input)
            # Shape: (batch_size, spatial_dim)
            
            # If frequency_input is a raw image tensor, compute DWT coefficients
            if not isinstance(frequency_input, dict):
                frequency_input = self._compute_dwt_coefficients(frequency_input)
            
            # Extract frequency features
            frequency_features = self.frequency_stream(frequency_input)
            # Shape: (batch_size, frequency_dim)
            
            # Cross-attention fusion
            fused_features, attention_weights = self.cross_attention(
                spatial_features, frequency_features, return_attention=return_attention
            )
            # Shape: (batch_size, embed_dim)
            
            # Feature disentanglement (if enabled)
            if self.feature_disentanglement is not None:
                invariant_features, specific_features, disentanglement_losses = \
                    self.feature_disentanglement(
                        fused_features, domain_labels, return_losses=self.training
                    )
                
                # Use invariant features for classification
                classification_features = invariant_features
                
                # Store disentanglement outputs
                outputs['invariant_features'] = invariant_features
                outputs['specific_features'] = specific_features
                if disentanglement_losses is not None:
                    outputs['disentanglement_losses'] = disentanglement_losses
            else:
                # Use fused features directly for classification
                classification_features = fused_features
            
            # Final classification
            logits = self.classifier(classification_features)
            outputs['logits'] = logits
            outputs['predictions'] = F.softmax(logits, dim=1)
            
            # Optional returns
            if return_features:
                outputs['spatial_features'] = spatial_features
                outputs['frequency_features'] = frequency_features
                outputs['fused_features'] = fused_features
                outputs['classification_features'] = classification_features
            
            if return_attention and attention_weights is not None:
                outputs['attention_weights'] = attention_weights
            
            return outputs
            
        except AssertionError:
            # Re-raise assertion errors for proper error handling in tests
            raise
        except Exception as e:
            logger.error(f"CRAFTDFModel forward pass failed: {str(e)}")
            raise RuntimeError(f"CRAFTDFModel forward pass failed: {str(e)}")
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step with comprehensive loss computation and metric tracking.
        
        This method implements the complete training logic for CRAFT-DF including:
        - Classification loss computation
        - Adversarial feature disentanglement losses
        - Gradient clipping for training stability
        - Comprehensive metric logging
        - Learning rate scheduling
        
        Args:
            batch: Training batch containing spatial_input, frequency_input, labels, domain_labels
            batch_idx: Batch index
            
        Returns:
            Total training loss for optimization
        """
        # Handle both automatic and manual optimization
        if self.adversarial_training:
            # Manual optimization for adversarial training
            optimizers = self.optimizers()
            if isinstance(optimizers, list):
                opt_main = optimizers[0]
                opt_domain = optimizers[1] if len(optimizers) > 1 else None
            else:
                opt_main = optimizers
                opt_domain = None
        else:
            # Automatic optimization for standard training
            opt_main = None
            opt_domain = None
        
        # Extract batch data with validation
        spatial_input = batch['spatial_input']
        frequency_input = batch['frequency_input']
        labels = batch['labels']
        domain_labels = batch.get('domain_labels', None)
        
        # Validate batch data
        assert spatial_input.shape[0] == labels.shape[0], "Batch size mismatch between spatial input and labels"
        if domain_labels is not None:
            assert spatial_input.shape[0] == domain_labels.shape[0], "Batch size mismatch with domain labels"
        
        # Forward pass with error handling
        try:
            outputs = self.forward(
                spatial_input, frequency_input, domain_labels, return_features=True
            )
        except Exception as e:
            logger.error(f"Forward pass failed in training step: {str(e)}")
            raise RuntimeError(f"Training forward pass failed: {str(e)}")
        
        # Classification loss
        classification_loss = self.classification_criterion(outputs['logits'], labels)
        
        # Initialize total loss with classification component
        total_loss = self.classification_weight * classification_loss
        
        # Add disentanglement losses if available
        disentanglement_metrics = {}
        if 'disentanglement_losses' in outputs:
            disentanglement_losses = outputs['disentanglement_losses']
            
            # Reconstruction loss
            if 'reconstruction' in disentanglement_losses:
                reconstruction_loss = disentanglement_losses['reconstruction']
                total_loss += self.reconstruction_weight * reconstruction_loss
                disentanglement_metrics['reconstruction_loss'] = reconstruction_loss
            
            # Orthogonality loss
            if 'orthogonality' in disentanglement_losses:
                orthogonality_loss = disentanglement_losses['orthogonality']
                total_loss += self.orthogonality_weight * orthogonality_loss
                disentanglement_metrics['orthogonality_loss'] = orthogonality_loss
            
            # Adversarial loss (with gradient reversal)
            if 'adversarial' in disentanglement_losses:
                adversarial_loss = disentanglement_losses['adversarial']
                total_loss += self.domain_adaptation_weight * adversarial_loss
                disentanglement_metrics['adversarial_loss'] = adversarial_loss
                
                # Domain accuracy for monitoring
                if 'domain_accuracy' in disentanglement_losses:
                    domain_accuracy = disentanglement_losses['domain_accuracy']
                    disentanglement_metrics['domain_accuracy'] = domain_accuracy
        
        # Optimization step
        if self.adversarial_training and opt_main is not None:
            # Manual optimization for adversarial training
            opt_main.zero_grad()
            self.manual_backward(total_loss)
            
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            opt_main.step()
            
            # Update learning rate schedulers
            schedulers = self.lr_schedulers()
            # Check if trainer is available and is last batch
            try:
                if schedulers and hasattr(self, '_trainer') and self._trainer is not None:
                    if self._trainer.is_last_batch:
                        if isinstance(schedulers, list):
                            for scheduler in schedulers:
                                if scheduler is not None:
                                    scheduler.step()
                        else:
                            schedulers.step()
            except (AttributeError, RuntimeError):
                # Handle case where trainer is not properly attached or accessible
                pass
        else:
            # Automatic optimization (standard PyTorch Lightning)
            # Gradient clipping is handled in configure_optimizers or trainer
            pass
        
        # Compute accuracy metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs['logits'], dim=1)
            if self.train_accuracy is not None:
                accuracy = self.train_accuracy(predictions, labels)
            else:
                accuracy = (predictions == labels).float().mean()
        
        # Comprehensive logging
        self.log('train/classification_loss', classification_loss, prog_bar=True, sync_dist=True)
        self.log('train/total_loss', total_loss, prog_bar=True, sync_dist=True)
        self.log('train/accuracy', accuracy, prog_bar=True, sync_dist=True)
        
        # Log learning rate
        if opt_main is not None:
            self.log('train/learning_rate', opt_main.param_groups[0]['lr'], sync_dist=True)
        
        # Log disentanglement metrics
        for metric_name, metric_value in disentanglement_metrics.items():
            prog_bar = metric_name in ['adversarial_loss', 'reconstruction_loss']
            self.log(f'train/{metric_name}', metric_value, prog_bar=prog_bar, sync_dist=True)
        
        # Additional metrics for monitoring
        self.log('train/batch_size', float(spatial_input.shape[0]), sync_dist=True)
        
        # Log gradient norms for debugging
        if self.global_step % 100 == 0:  # Log every 100 steps
            total_grad_norm = 0.0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_grad_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5
            self.log('train/grad_norm', total_grad_norm, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step with comprehensive metric computation.
        
        This method implements validation logic including:
        - Classification loss and accuracy computation
        - Disentanglement quality metrics
        - Feature analysis and visualization data
        - Comprehensive logging for monitoring
        
        Args:
            batch: Validation batch containing spatial_input, frequency_input, labels, domain_labels
            batch_idx: Batch index
            
        Returns:
            Dictionary of validation metrics and outputs
        """
        # Extract batch data
        spatial_input = batch['spatial_input']
        frequency_input = batch['frequency_input']
        labels = batch['labels']
        domain_labels = batch.get('domain_labels', None)
        
        # Validate batch data
        assert spatial_input.shape[0] == labels.shape[0], "Batch size mismatch between spatial input and labels"
        if domain_labels is not None:
            assert spatial_input.shape[0] == domain_labels.shape[0], "Batch size mismatch with domain labels"
        
        # Forward pass with error handling
        try:
            outputs = self.forward(
                spatial_input, frequency_input, domain_labels, 
                return_features=True, return_attention=True
            )
        except Exception as e:
            logger.error(f"Forward pass failed in validation step: {str(e)}")
            raise RuntimeError(f"Validation forward pass failed: {str(e)}")
        
        # Classification loss and metrics
        classification_loss = self.classification_criterion(outputs['logits'], labels)
        
        # Compute accuracy
        predictions = torch.argmax(outputs['logits'], dim=1)
        if self.val_accuracy is not None:
            accuracy = self.val_accuracy(predictions, labels)
        else:
            accuracy = (predictions == labels).float().mean()
        
        # Compute additional classification metrics
        with torch.no_grad():
            # Confidence scores
            probabilities = F.softmax(outputs['logits'], dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
            mean_confidence = confidence_scores.mean()
            
            # Prediction distribution
            pred_distribution = torch.bincount(predictions, minlength=self.num_classes).float()
            pred_distribution = pred_distribution / pred_distribution.sum()
        
        # Initialize validation metrics
        val_metrics = {
            'val_loss': classification_loss,
            'val_accuracy': accuracy,
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities,
            'confidence_scores': confidence_scores,
            'mean_confidence': mean_confidence
        }
        
        # Disentanglement metrics if available
        disentanglement_metrics = {}
        if 'disentanglement_losses' in outputs:
            disentanglement_losses = outputs['disentanglement_losses']
            
            # Log individual disentanglement losses
            for loss_name, loss_value in disentanglement_losses.items():
                if isinstance(loss_value, torch.Tensor) and loss_value.dim() == 0:
                    disentanglement_metrics[f'val/{loss_name}'] = loss_value
            
            # Domain confusion analysis
            if domain_labels is not None and 'domain_accuracy' in disentanglement_losses:
                domain_accuracy = disentanglement_losses['domain_accuracy']
                domain_confusion = 1.0 - domain_accuracy  # Higher confusion = better disentanglement
                disentanglement_metrics['val/domain_confusion'] = domain_confusion
        
        # Feature analysis metrics
        feature_metrics = {}
        if 'invariant_features' in outputs and 'specific_features' in outputs:
            invariant_features = outputs['invariant_features']
            specific_features = outputs['specific_features']
            
            # Feature norm analysis
            invariant_norm = torch.norm(invariant_features, p=2, dim=1).mean()
            specific_norm = torch.norm(specific_features, p=2, dim=1).mean()
            
            feature_metrics['val/invariant_feature_norm'] = invariant_norm
            feature_metrics['val/specific_feature_norm'] = specific_norm
            
            # Feature diversity (standard deviation across batch)
            invariant_diversity = torch.std(invariant_features, dim=0).mean()
            specific_diversity = torch.std(specific_features, dim=0).mean()
            
            feature_metrics['val/invariant_diversity'] = invariant_diversity
            feature_metrics['val/specific_diversity'] = specific_diversity
        
        # Attention analysis if available
        attention_metrics = {}
        if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
            attention_weights = outputs['attention_weights']
            
            # Attention statistics
            attention_mean = attention_weights.mean()
            attention_std = attention_weights.std()
            attention_entropy = self._compute_attention_entropy(attention_weights)
            
            attention_metrics['val/attention_mean'] = attention_mean
            attention_metrics['val/attention_std'] = attention_std
            attention_metrics['val/attention_entropy'] = attention_entropy.mean()
        
        # Comprehensive logging
        self.log('val/classification_loss', classification_loss, prog_bar=True, sync_dist=True)
        self.log('val/accuracy', accuracy, prog_bar=True, sync_dist=True)
        self.log('val/mean_confidence', mean_confidence, sync_dist=True)
        
        # Log class distribution for monitoring data balance
        for class_idx in range(self.num_classes):
            class_ratio = pred_distribution[class_idx]
            self.log(f'val/pred_class_{class_idx}_ratio', class_ratio, sync_dist=True)
        
        # Log disentanglement metrics
        for metric_name, metric_value in disentanglement_metrics.items():
            prog_bar = 'domain_confusion' in metric_name or 'adversarial' in metric_name
            self.log(metric_name, metric_value, prog_bar=prog_bar, sync_dist=True)
        
        # Log feature metrics
        for metric_name, metric_value in feature_metrics.items():
            self.log(metric_name, metric_value, sync_dist=True)
        
        # Log attention metrics
        for metric_name, metric_value in attention_metrics.items():
            self.log(metric_name, metric_value, sync_dist=True)
        
        # Additional validation metrics
        self.log('val/batch_size', float(spatial_input.shape[0]), sync_dist=True)
        
        # Store additional data for epoch-end analysis
        val_metrics.update({
            'spatial_features': outputs.get('spatial_features', None),
            'frequency_features': outputs.get('frequency_features', None),
            'fused_features': outputs.get('fused_features', None),
            'invariant_features': outputs.get('invariant_features', None),
            'specific_features': outputs.get('specific_features', None),
            'attention_weights': outputs.get('attention_weights', None)
        })
        
        return val_metrics
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention entropy for measuring attention concentration.
        
        Args:
            attention_weights: Attention weights tensor
            
        Returns:
            Attention entropy for each sample
        """
        # Flatten attention weights and normalize
        batch_size = attention_weights.shape[0]
        attention_flat = attention_weights.view(batch_size, -1)
        attention_flat = F.softmax(attention_flat, dim=-1)
        
        # Add small epsilon to prevent log(0)
        eps = 1e-8
        attention_flat = attention_flat + eps
        
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(attention_flat * torch.log(attention_flat), dim=1)
        
        return entropy
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Test batch
            batch_idx: Batch index
            
        Returns:
            Dictionary of test metrics
        """
        spatial_input = batch['spatial_input']
        frequency_input = batch['frequency_input']
        labels = batch['labels']
        
        # Forward pass (no domain labels needed for testing)
        outputs = self.forward(spatial_input, frequency_input)
        
        # Classification loss
        classification_loss = self.classification_criterion(outputs['logits'], labels)
        
        # Compute accuracy
        predictions = torch.argmax(outputs['logits'], dim=1)
        if self.test_accuracy is not None:
            accuracy = self.test_accuracy(predictions, labels)
        else:
            accuracy = (predictions == labels).float().mean()
        
        # Log metrics
        self.log('test/classification_loss', classification_loss)
        self.log('test/accuracy', accuracy)
        
        return {
            'test_loss': classification_loss,
            'test_accuracy': accuracy,
            'predictions': predictions,
            'labels': labels
        }
    
    def _compute_dwt_coefficients(self, images: torch.Tensor) -> dict:
        """
        Compute multi-level DWT coefficients from raw image tensors using PyWavelets.

        Args:
            images: (B, C, H, W) float tensor in [0, 1]

        Returns:
            Dict with keys 'll', 'lh_1', 'hl_1', 'hh_1', ..., 'lh_N', 'hl_N', 'hh_N'
        """
        import pywt
        import numpy as np

        dwt_levels = self.frequency_stream.dwt_levels
        device = images.device
        B, C, H, W = images.shape

        coeffs_dict: dict = {}
        imgs_np = images.detach().cpu().numpy()  # (B, C, H, W)

        ll_list = []
        detail_lists: dict = {f"{s}_{l}": [] for l in range(1, dwt_levels + 1)
                               for s in ['lh', 'hl', 'hh']}

        for b in range(B):
            per_channel_coeffs = []
            for c in range(C):
                coeffs = pywt.wavedec2(imgs_np[b, c], wavelet='db4',
                                       level=dwt_levels, mode='symmetric')
                per_channel_coeffs.append(coeffs)

            # LL: deepest approximation — shape (H', W')
            ll = np.stack([per_channel_coeffs[c][0] for c in range(C)], axis=0)  # (C, H', W')
            ll_list.append(ll)

            for level in range(1, dwt_levels + 1):
                lh, hl, hh = zip(*[per_channel_coeffs[c][level] for c in range(C)])
                detail_lists[f'lh_{level}'].append(np.stack(lh, axis=0))
                detail_lists[f'hl_{level}'].append(np.stack(hl, axis=0))
                detail_lists[f'hh_{level}'].append(np.stack(hh, axis=0))

        coeffs_dict['ll'] = torch.from_numpy(np.stack(ll_list, axis=0)).float().to(device)
        for key, arrays in detail_lists.items():
            coeffs_dict[key] = torch.from_numpy(np.stack(arrays, axis=0)).float().to(device)

        return coeffs_dict

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers with comprehensive setup.
        
        This method sets up:
        - Main optimizer for all model parameters
        - Optional domain optimizer for adversarial training
        - Learning rate schedulers with warmup and decay
        - Gradient clipping configuration
        - Optimizer-specific hyperparameters
        
        Returns:
            Dictionary containing optimizers and schedulers configuration
        """
        # Separate parameters for different optimization strategies
        main_params = []
        domain_params = []
        
        # Collect main model parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'domain_classifier' in name and self.adversarial_training:
                    domain_params.append(param)
                else:
                    main_params.append(param)
        
        # Main optimizer configuration
        main_optimizer = AdamW(
            main_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=False  # Can be enabled for more stable training
        )
        
        optimizers = [main_optimizer]
        schedulers = []
        
        # Configure main scheduler
        if self.scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                main_optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.learning_rate * 0.01
            )
            schedulers.append({
                'scheduler': main_scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'main_lr'
            })
            
        elif self.scheduler_type == "plateau":
            main_scheduler = ReduceLROnPlateau(
                main_optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=2,
                min_lr=self.learning_rate * 1e-3
            )
            schedulers.append({
                'scheduler': main_scheduler,
                'monitor': 'val/classification_loss',
                'interval': 'epoch',
                'frequency': 1,
                'name': 'main_lr'
            })
            
        elif self.scheduler_type == "warmup_cosine":
            # Warmup + Cosine Annealing
            from torch.optim.lr_scheduler import LambdaLR
            
            def warmup_cosine_schedule(step):
                if step < self.warmup_epochs:
                    # Linear warmup
                    return step / self.warmup_epochs
                else:
                    # Cosine annealing
                    progress = (step - self.warmup_epochs) / (self.trainer.max_epochs - self.warmup_epochs)
                    return 0.5 * (1 + math.cos(math.pi * progress))
            
            main_scheduler = LambdaLR(main_optimizer, warmup_cosine_schedule)
            schedulers.append({
                'scheduler': main_scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'main_lr'
            })
        
        # Domain optimizer for adversarial training
        if self.adversarial_training and domain_params:
            domain_optimizer = Adam(
                domain_params,
                lr=self.learning_rate * 0.1,  # Lower learning rate for domain classifier
                weight_decay=self.weight_decay * 0.1,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            optimizers.append(domain_optimizer)
            
            # Domain scheduler (simpler than main scheduler)
            if self.scheduler_type in ["cosine", "warmup_cosine"]:
                domain_scheduler = CosineAnnealingLR(
                    domain_optimizer,
                    T_max=self.trainer.max_epochs,
                    eta_min=self.learning_rate * 0.001
                )
                schedulers.append({
                    'scheduler': domain_scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'name': 'domain_lr'
                })
        
        # Configure optimization strategy — return format depends on optimizer count
        if not self.adversarial_training:
            # Single optimizer: return clean dict PL expects
            if schedulers:
                return {'optimizer': main_optimizer, 'lr_scheduler': schedulers[0]}
            return main_optimizer

        # Multiple optimizers (adversarial): return lists
        config = {
            'optimizer': optimizers,
            'lr_scheduler': schedulers
        }
        
        logger.info(f"Configured {len(optimizers)} optimizers and {len(schedulers)} schedulers")
        logger.info(f"Main optimizer: AdamW(lr={self.learning_rate}, weight_decay={self.weight_decay})")
        if len(optimizers) > 1:
            logger.info(f"Domain optimizer: Adam(lr={self.learning_rate * 0.1})")
        logger.info(f"Scheduler type: {self.scheduler_type}")
        
        return config
    
    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Prediction step for inference.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: Dataloader index
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        spatial_input = batch['spatial_input']
        frequency_input = batch['frequency_input']
        
        # Forward pass
        outputs = self.forward(
            spatial_input, frequency_input, 
            return_features=True, return_attention=True
        )
        
        predictions = torch.argmax(outputs['logits'], dim=1)
        probabilities = outputs['predictions']
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'logits': outputs['logits'],
            'attention_weights': outputs.get('attention_weights', None),
            'features': {
                'spatial': outputs.get('spatial_features', None),
                'frequency': outputs.get('frequency_features', None),
                'fused': outputs.get('fused_features', None),
                'invariant': outputs.get('invariant_features', None),
                'specific': outputs.get('specific_features', None)
            }
        }
    
    def analyze_feature_disentanglement(
        self,
        spatial_input: torch.Tensor,
        frequency_input: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Analyze the quality of feature disentanglement.
        
        Args:
            spatial_input: Spatial input tensor
            frequency_input: Frequency input tensor
            domain_labels: Domain labels
            
        Returns:
            Dictionary of disentanglement analysis results
        """
        if self.feature_disentanglement is None:
            raise ValueError("Feature disentanglement is not enabled")
        
        with torch.no_grad():
            # Get fused features
            spatial_features = self.spatial_stream(spatial_input)
            frequency_features = self.frequency_stream(frequency_input)
            fused_features, _ = self.cross_attention(spatial_features, frequency_features)
            
            # Analyze disentanglement
            analysis = self.feature_disentanglement.analyze_feature_separation(
                fused_features, domain_labels
            )
            
            return analysis
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = {
            'model_name': 'CRAFT-DF',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_efficiency': trainable_params / total_params,
            'adversarial_training': self.adversarial_training,
            'num_classes': self.num_classes,
            'components': {
                'spatial_stream': sum(p.numel() for p in self.spatial_stream.parameters()),
                'frequency_stream': sum(p.numel() for p in self.frequency_stream.parameters()),
                'cross_attention': sum(p.numel() for p in self.cross_attention.parameters()),
                'classifier': sum(p.numel() for p in self.classifier.parameters())
            }
        }
        
        if self.feature_disentanglement is not None:
            summary['components']['feature_disentanglement'] = sum(
                p.numel() for p in self.feature_disentanglement.parameters()
            )
        
        return summary