"""
Discrete Wavelet Transform (DWT) processing for frequency domain analysis.

This module provides comprehensive DWT-based feature extraction capabilities
optimized for deepfake detection. It implements multi-level wavelet decomposition
using Daubechies wavelets to extract frequency domain artifacts that are
characteristic of deepfake generation processes.

The DWT decomposes images into different frequency subbands, allowing the model
to analyze both low-frequency content (approximation coefficients) and high-frequency
details (horizontal, vertical, and diagonal detail coefficients) that often contain
telltale signs of deepfake manipulation.
"""

import numpy as np
import pywt
from typing import Tuple, List, Optional, Union, Dict
import logging
import cv2

logger = logging.getLogger(__name__)


class DWTProcessor:
    """
    Discrete Wavelet Transform processor for frequency domain feature extraction.
    
    This class implements multi-level DWT decomposition using Daubechies wavelets
    to extract frequency domain features from face images. The DWT analysis is
    particularly effective for deepfake detection as it can reveal compression
    artifacts and frequency anomalies introduced by generative models.
    
    The theoretical foundation:
    - DWT decomposes signals into different frequency bands using wavelet basis functions
    - Daubechies wavelets (db4) provide good time-frequency localization
    - Multi-level decomposition captures features at different scales
    - High-frequency detail coefficients often contain deepfake artifacts
    
    Attributes:
        wavelet (str): Wavelet type (default: 'db4' - Daubechies 4)
        levels (int): Number of decomposition levels (default: 3)
        mode (str): Border condition handling mode (default: 'symmetric')
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        levels: int = 3,
        mode: str = 'symmetric'
    ):
        """
        Initialize the DWT processor.
        
        Args:
            wavelet: Wavelet type for decomposition. 'db4' (Daubechies 4) provides
                    good balance between smoothness and compact support, making it
                    effective for image analysis and artifact detection.
            levels: Number of decomposition levels. 3 levels provide good frequency
                   resolution while maintaining computational efficiency.
            mode: Boundary condition handling. 'symmetric' reduces boundary artifacts
                 by extending the signal symmetrically at borders.
        
        Raises:
            ValueError: If wavelet type is not supported or levels is invalid
        """
        # Validate wavelet type
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Unsupported wavelet type: {wavelet}. "
                           f"Available wavelets: {pywt.wavelist()}")
        
        if not isinstance(levels, int) or levels < 1 or levels > 6:
            raise ValueError("levels must be an integer between 1 and 6")
        
        if mode not in pywt.Modes.modes:
            raise ValueError(f"Unsupported mode: {mode}. "
                           f"Available modes: {pywt.Modes.modes}")
        
        self.wavelet = wavelet
        self.levels = levels
        self.mode = mode
        
        # Validate wavelet properties
        wavelet_obj = pywt.Wavelet(wavelet)
        self.filter_length = len(wavelet_obj.dec_lo)
        
        logger.info(f"DWTProcessor initialized with wavelet={wavelet}, "
                   f"levels={levels}, mode={mode}, filter_length={self.filter_length}")
    
    def decompose_2d(self, image: np.ndarray) -> List[Tuple[np.ndarray, ...]]:
        """
        Perform 2D multi-level DWT decomposition on an image.
        
        The 2D DWT decomposes an image into four subbands at each level:
        - LL (Low-Low): Approximation coefficients (low-frequency content)
        - LH (Low-High): Horizontal detail coefficients (vertical edges)
        - HL (High-Low): Vertical detail coefficients (horizontal edges)  
        - HH (High-High): Diagonal detail coefficients (diagonal features)
        
        For deepfake detection, the detail coefficients (LH, HL, HH) are particularly
        important as they capture high-frequency artifacts introduced by generative models.
        
        Args:
            image: Input image as numpy array (H, W) for grayscale or (H, W, C) for color
            
        Returns:
            List of coefficient tuples for each decomposition level.
            Format: [(LL_n, (LH_n, HL_n, HH_n)), ..., (LL_1, (LH_1, HL_1, HH_1))]
            where n is the deepest level and 1 is the first level.
            
        Raises:
            ValueError: If image format is invalid
            RuntimeError: If DWT decomposition fails
        """
        # Validate input image
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if len(image.shape) not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")
        
        assert image.dtype in [np.uint8, np.float32, np.float64], \
            f"Expected uint8, float32, or float64 image, got {image.dtype}"
        assert image.shape[0] > 0 and image.shape[1] > 0, \
            "Image dimensions must be positive"
        
        try:
            # Convert to float for DWT processing
            if image.dtype == np.uint8:
                image_float = image.astype(np.float32) / 255.0
            else:
                image_float = image.astype(np.float32)
            
            # Handle color images by processing each channel separately
            if len(image.shape) == 3:
                # Process each color channel
                channels_coeffs = []
                for c in range(image.shape[2]):
                    channel = image_float[:, :, c]
                    coeffs = pywt.wavedec2(channel, self.wavelet, 
                                         level=self.levels, mode=self.mode)
                    channels_coeffs.append(coeffs)
                
                # Combine coefficients from all channels
                # Stack coefficients along a new axis for multi-channel processing
                combined_coeffs = []
                for level in range(len(channels_coeffs[0])):
                    if level == 0:
                        # Approximation coefficients (LL)
                        ll_combined = np.stack([ch_coeffs[level] for ch_coeffs in channels_coeffs], axis=-1)
                        combined_coeffs.append(ll_combined)
                    else:
                        # Detail coefficients (LH, HL, HH)
                        lh_combined = np.stack([ch_coeffs[level][0] for ch_coeffs in channels_coeffs], axis=-1)
                        hl_combined = np.stack([ch_coeffs[level][1] for ch_coeffs in channels_coeffs], axis=-1)
                        hh_combined = np.stack([ch_coeffs[level][2] for ch_coeffs in channels_coeffs], axis=-1)
                        combined_coeffs.append((lh_combined, hl_combined, hh_combined))
                
                coefficients = combined_coeffs
            else:
                # Grayscale image
                coefficients = pywt.wavedec2(image_float, self.wavelet,
                                           level=self.levels, mode=self.mode)
            
            # Validate coefficient shapes
            for i, coeff in enumerate(coefficients):
                if i == 0:  # Approximation coefficients
                    assert coeff.shape[0] > 0 and coeff.shape[1] > 0, \
                        f"Invalid approximation coefficient shape at level {i}: {coeff.shape}"
                else:  # Detail coefficients
                    assert len(coeff) == 3, f"Expected 3 detail coefficients at level {i}, got {len(coeff)}"
                    for j, detail in enumerate(coeff):
                        assert detail.shape[0] > 0 and detail.shape[1] > 0, \
                            f"Invalid detail coefficient shape at level {i}, component {j}: {detail.shape}"
            
            logger.debug(f"DWT decomposition completed: {len(coefficients)} levels, "
                        f"approximation shape: {coefficients[0].shape}")
            
            return coefficients
            
        except Exception as e:
            logger.error(f"DWT decomposition failed: {str(e)}")
            raise RuntimeError(f"DWT decomposition failed: {str(e)}")
    
    def extract_features(self, coefficients: List[Tuple[np.ndarray, ...]]) -> np.ndarray:
        """
        Extract statistical features from DWT coefficients for deepfake detection.
        
        This method computes various statistical measures from the wavelet coefficients
        that are indicative of deepfake artifacts:
        
        1. Energy measures: Capture the distribution of frequency content
        2. Statistical moments: Mean, variance, skewness, kurtosis reveal distribution properties
        3. Entropy measures: Quantify information content and randomness
        4. Cross-subband correlations: Detect unnatural frequency relationships
        
        The feature extraction focuses on detail coefficients (LH, HL, HH) as these
        high-frequency components often contain the most discriminative information
        for deepfake detection.
        
        Args:
            coefficients: DWT coefficients from decompose_2d()
            
        Returns:
            Feature vector as 1D numpy array containing:
            - Energy features (per level, per subband)
            - Statistical moments (mean, std, skewness, kurtosis)
            - Entropy measures
            - Cross-correlation features
            
        Raises:
            ValueError: If coefficients format is invalid
        """
        if not isinstance(coefficients, list) or len(coefficients) == 0:
            raise ValueError("Coefficients must be a non-empty list")
        
        features = []
        
        try:
            # Process each decomposition level
            for level_idx, coeff in enumerate(coefficients):
                if level_idx == 0:
                    # Approximation coefficients (LL)
                    ll_coeff = coeff
                    
                    # Extract statistical features from approximation
                    ll_features = self._extract_statistical_features(ll_coeff, f"LL_{level_idx}")
                    features.extend(ll_features)
                    
                else:
                    # Detail coefficients (LH, HL, HH)
                    lh_coeff, hl_coeff, hh_coeff = coeff
                    
                    # Extract features from each detail subband
                    for subband_name, subband_coeff in [("LH", lh_coeff), ("HL", hl_coeff), ("HH", hh_coeff)]:
                        subband_features = self._extract_statistical_features(
                            subband_coeff, f"{subband_name}_{level_idx}"
                        )
                        features.extend(subband_features)
                    
                    # Cross-subband correlation features
                    cross_corr_features = self._extract_cross_correlation_features(
                        lh_coeff, hl_coeff, hh_coeff, level_idx
                    )
                    features.extend(cross_corr_features)
            
            # Convert to numpy array and validate
            feature_vector = np.array(features, dtype=np.float32)
            
            assert len(feature_vector) > 0, "Feature vector is empty"
            assert np.all(np.isfinite(feature_vector)), "Feature vector contains non-finite values"
            
            logger.debug(f"Extracted {len(feature_vector)} DWT features")
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise RuntimeError(f"Feature extraction failed: {str(e)}")
    
    def _extract_statistical_features(self, coefficients: np.ndarray, subband_name: str) -> List[float]:
        """
        Extract statistical features from a coefficient subband.
        
        Args:
            coefficients: Coefficient array for a specific subband
            subband_name: Name of the subband for logging
            
        Returns:
            List of statistical features
        """
        # Flatten coefficients for statistical analysis
        flat_coeffs = coefficients.flatten()
        
        # Remove any potential NaN or infinite values
        valid_coeffs = flat_coeffs[np.isfinite(flat_coeffs)]
        
        if len(valid_coeffs) == 0:
            logger.warning(f"No valid coefficients in {subband_name}")
            return [0.0] * 8  # Return zeros for all features
        
        features = []
        
        # Basic statistical moments
        features.append(float(np.mean(valid_coeffs)))  # Mean
        features.append(float(np.std(valid_coeffs)))   # Standard deviation
        
        # Higher-order moments (normalized)
        if len(valid_coeffs) > 1 and np.std(valid_coeffs) > 1e-10:
            # Skewness (third moment)
            skewness = np.mean(((valid_coeffs - np.mean(valid_coeffs)) / np.std(valid_coeffs)) ** 3)
            features.append(float(skewness))
            
            # Kurtosis (fourth moment)
            kurtosis = np.mean(((valid_coeffs - np.mean(valid_coeffs)) / np.std(valid_coeffs)) ** 4) - 3
            features.append(float(kurtosis))
        else:
            features.extend([0.0, 0.0])
        
        # Energy measures
        energy = float(np.sum(valid_coeffs ** 2))
        features.append(energy)
        
        # Use adaptive histogram binning for entropy estimation
        # Reduce bins for arrays with small range
        range_val = np.max(valid_coeffs) - np.min(valid_coeffs)
        if range_val < 1e-10:
            # Constant or near-constant array
            entropy = 0.0
        else:
            # Adaptive number of bins based on data range and size
            n_bins = min(50, max(5, int(np.sqrt(len(valid_coeffs)))))
            try:
                hist, _ = np.histogram(valid_coeffs, bins=n_bins, density=True)
                hist = hist[hist > 0]  # Remove zero bins
                if len(hist) > 0:
                    entropy = -np.sum(hist * np.log2(hist + 1e-10))
                else:
                    entropy = 0.0
            except ValueError:
                # Fallback for problematic cases
                entropy = 0.0
        
        features.append(float(entropy))
        
        # Range and percentile features
        features.append(float(np.max(valid_coeffs) - np.min(valid_coeffs)))  # Range
        features.append(float(np.percentile(valid_coeffs, 95) - np.percentile(valid_coeffs, 5)))  # IQR-like
        
        return features
    
    def _extract_cross_correlation_features(
        self, 
        lh_coeff: np.ndarray, 
        hl_coeff: np.ndarray, 
        hh_coeff: np.ndarray,
        level: int
    ) -> List[float]:
        """
        Extract cross-correlation features between detail subbands.
        
        Cross-correlations between different detail subbands can reveal unnatural
        frequency relationships that are characteristic of deepfake generation.
        
        Args:
            lh_coeff: Horizontal detail coefficients
            hl_coeff: Vertical detail coefficients  
            hh_coeff: Diagonal detail coefficients
            level: Decomposition level
            
        Returns:
            List of cross-correlation features
        """
        features = []
        
        try:
            # Flatten coefficients
            lh_flat = lh_coeff.flatten()
            hl_flat = hl_coeff.flatten()
            hh_flat = hh_coeff.flatten()
            
            # Ensure all arrays have the same length (they should, but safety check)
            min_len = min(len(lh_flat), len(hl_flat), len(hh_flat))
            lh_flat = lh_flat[:min_len]
            hl_flat = hl_flat[:min_len]
            hh_flat = hh_flat[:min_len]
            
            # Remove non-finite values
            valid_mask = np.isfinite(lh_flat) & np.isfinite(hl_flat) & np.isfinite(hh_flat)
            lh_valid = lh_flat[valid_mask]
            hl_valid = hl_flat[valid_mask]
            hh_valid = hh_flat[valid_mask]
            
            if len(lh_valid) < 2:
                return [0.0] * 3
            
            # Pearson correlation coefficients
            corr_lh_hl = float(np.corrcoef(lh_valid, hl_valid)[0, 1]) if np.std(lh_valid) > 1e-10 and np.std(hl_valid) > 1e-10 else 0.0
            corr_lh_hh = float(np.corrcoef(lh_valid, hh_valid)[0, 1]) if np.std(lh_valid) > 1e-10 and np.std(hh_valid) > 1e-10 else 0.0
            corr_hl_hh = float(np.corrcoef(hl_valid, hh_valid)[0, 1]) if np.std(hl_valid) > 1e-10 and np.std(hh_valid) > 1e-10 else 0.0
            
            # Handle NaN correlations (can occur with constant arrays)
            corr_lh_hl = 0.0 if np.isnan(corr_lh_hl) else corr_lh_hl
            corr_lh_hh = 0.0 if np.isnan(corr_lh_hh) else corr_lh_hh
            corr_hl_hh = 0.0 if np.isnan(corr_hl_hh) else corr_hl_hh
            
            features.extend([corr_lh_hl, corr_lh_hh, corr_hl_hh])
            
        except Exception as e:
            logger.warning(f"Cross-correlation computation failed at level {level}: {str(e)}")
            features.extend([0.0] * 3)
        
        return features
    
    def process_face_crop(self, face_image: np.ndarray) -> np.ndarray:
        """
        Complete DWT processing pipeline for a face crop.
        
        This method provides a convenient interface for processing face crops
        through the complete DWT analysis pipeline:
        1. Input validation and preprocessing
        2. Multi-level DWT decomposition
        3. Feature extraction from coefficients
        
        Args:
            face_image: Face crop as numpy array (H, W, C) or (H, W)
            
        Returns:
            Feature vector as 1D numpy array
            
        Raises:
            ValueError: If face image format is invalid
            RuntimeError: If processing fails
        """
        # Validate input
        if not isinstance(face_image, np.ndarray):
            raise ValueError("Face image must be a numpy array")
        
        if len(face_image.shape) not in [2, 3]:
            raise ValueError("Face image must be 2D (grayscale) or 3D (color)")
        
        assert face_image.shape[0] > 0 and face_image.shape[1] > 0, \
            "Face image dimensions must be positive"
        
        try:
            # Perform DWT decomposition
            coefficients = self.decompose_2d(face_image)
            
            # Extract features
            features = self.extract_features(coefficients)
            
            logger.debug(f"DWT processing completed: input shape {face_image.shape}, "
                        f"output features {len(features)}")
            
            return features
            
        except Exception as e:
            logger.error(f"DWT processing failed: {str(e)}")
            raise RuntimeError(f"DWT processing failed: {str(e)}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get descriptive names for the extracted features.
        
        Returns:
            List of feature names corresponding to the feature vector
        """
        feature_names = []
        
        # Approximation coefficients features
        for stat in ['mean', 'std', 'skewness', 'kurtosis', 'energy', 'entropy', 'range', 'iqr']:
            feature_names.append(f"LL_0_{stat}")
        
        # Detail coefficients features for each level
        for level in range(1, self.levels + 1):
            for subband in ['LH', 'HL', 'HH']:
                for stat in ['mean', 'std', 'skewness', 'kurtosis', 'energy', 'entropy', 'range', 'iqr']:
                    feature_names.append(f"{subband}_{level}_{stat}")
            
            # Cross-correlation features
            feature_names.extend([
                f"corr_LH_HL_{level}",
                f"corr_LH_HH_{level}",
                f"corr_HL_HH_{level}"
            ])
        
        return feature_names
    
    def __repr__(self) -> str:
        """String representation of the DWT processor."""
        return (f"DWTProcessor(wavelet='{self.wavelet}', levels={self.levels}, "
                f"mode='{self.mode}', filter_length={self.filter_length})")