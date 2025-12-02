"""
Postprocessing Pipeline for Enhanced Ghibli Processor
Handles image enhancement and quality assessment after generation
"""
import cv2
import numpy as np
import logging
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple
from skimage.metrics import structural_similarity as ssim

from core.models import QualityMetrics


class PostprocessingPipeline:
    """Handles image postprocessing and quality assessment"""
    
    def __init__(
        self,
        sharpen_amount: float = 0.3,
        saturation_factor: float = 1.2,
        warm_tone_strength: float = 0.15,
        min_ssim_threshold: float = 0.85,
        logger: logging.Logger = None
    ):
        """
        Initialize PostprocessingPipeline
        
        Args:
            sharpen_amount: Amount of sharpening (0-1)
            saturation_factor: Saturation multiplier
            warm_tone_strength: Warm tone strength (0-1)
            min_ssim_threshold: Minimum SSIM to avoid over-processing
            logger: Logger instance
        """
        self.sharpen_amount = sharpen_amount
        self.saturation_factor = saturation_factor
        self.warm_tone_strength = warm_tone_strength
        self.min_ssim_threshold = min_ssim_threshold
        self.logger = logger or logging.getLogger(__name__)
    
    def sharpen(self, image: Image.Image, amount: float = None) -> Image.Image:
        """
        Apply sharpening to enhance details
        
        Args:
            image: Input image
            amount: Sharpening amount (default: self.sharpen_amount)
        
        Returns:
            Sharpened image
        """
        if amount is None:
            amount = self.sharpen_amount
        
        try:
            # Apply unsharp mask
            blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
            
            # Calculate sharpened image
            img_array = np.array(image, dtype=np.float32)
            blurred_array = np.array(blurred, dtype=np.float32)
            
            sharpened_array = img_array + amount * (img_array - blurred_array)
            sharpened_array = np.clip(sharpened_array, 0, 255).astype(np.uint8)
            
            sharpened_image = Image.fromarray(sharpened_array)
            
            self.logger.debug(f"Applied sharpening with amount: {amount}")
            return sharpened_image
            
        except Exception as e:
            self.logger.warning(f"Sharpening failed: {e}")
            return image
    
    def adjust_saturation(self, image: Image.Image, factor: float = None) -> Image.Image:
        """
        Adjust color saturation
        
        Args:
            image: Input image
            factor: Saturation factor (default: self.saturation_factor)
        
        Returns:
            Saturation-adjusted image
        """
        if factor is None:
            factor = self.saturation_factor
        
        try:
            enhancer = ImageEnhance.Color(image)
            enhanced_image = enhancer.enhance(factor)
            
            self.logger.debug(f"Adjusted saturation with factor: {factor}")
            return enhanced_image
            
        except Exception as e:
            self.logger.warning(f"Saturation adjustment failed: {e}")
            return image
    
    def apply_warm_tone(self, image: Image.Image, strength: float = None) -> Image.Image:
        """
        Apply warm color tone
        
        Args:
            image: Input image
            strength: Warm tone strength (default: self.warm_tone_strength)
        
        Returns:
            Warm-toned image
        """
        if strength is None:
            strength = self.warm_tone_strength
        
        try:
            img_array = np.array(image, dtype=np.float32)
            
            # Increase red channel, slightly decrease blue channel
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + strength), 0, 255)  # Red
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - strength * 0.5), 0, 255)  # Blue
            
            warm_image = Image.fromarray(img_array.astype(np.uint8))
            
            self.logger.debug(f"Applied warm tone with strength: {strength}")
            return warm_image
            
        except Exception as e:
            self.logger.warning(f"Warm tone application failed: {e}")
            return image
    
    def calculate_sharpness(self, image: Image.Image) -> float:
        """
        Calculate image sharpness using Laplacian variance
        
        Args:
            image: Input image
        
        Returns:
            Sharpness score (0-100)
        """
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize to 0-100 scale
            sharpness = min(laplacian_var / 10, 100)
            
            return float(sharpness)
            
        except Exception as e:
            self.logger.warning(f"Sharpness calculation failed: {e}")
            return 0.0
    
    def calculate_edge_clarity(self, image: Image.Image) -> float:
        """
        Calculate edge clarity using Canny edge detection
        
        Args:
            image: Input image
        
        Returns:
            Edge clarity score (0-100)
        """
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Normalize to 0-100 scale
            edge_clarity = min(edge_density * 500, 100)
            
            return float(edge_clarity)
            
        except Exception as e:
            self.logger.warning(f"Edge clarity calculation failed: {e}")
            return 0.0
    
    def calculate_color_harmony(self, image: Image.Image) -> float:
        """
        Calculate color harmony (inverse of color variance)
        
        Args:
            image: Input image
        
        Returns:
            Color harmony score (0-100)
        """
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                # Calculate color variance
                color_std = np.std(img_array, axis=(0, 1))
                avg_std = np.mean(color_std)
                
                # Lower variance = higher harmony
                # Normalize to 0-100 scale (inverse relationship)
                harmony = max(0, 100 - (avg_std / 2))
                
                return float(harmony)
            else:
                return 50.0  # Default for grayscale
                
        except Exception as e:
            self.logger.warning(f"Color harmony calculation failed: {e}")
            return 0.0
    
    def calculate_brightness(self, image: Image.Image) -> float:
        """
        Calculate average brightness
        
        Args:
            image: Input image
        
        Returns:
            Average brightness (0-255)
        """
        try:
            img_array = np.array(image)
            brightness = np.mean(img_array)
            return float(brightness)
            
        except Exception as e:
            self.logger.warning(f"Brightness calculation failed: {e}")
            return 0.0
    
    def calculate_saturation(self, image: Image.Image) -> float:
        """
        Calculate average saturation
        
        Args:
            image: Input image
        
        Returns:
            Average saturation (0-100)
        """
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                # Convert to HSV
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                saturation = np.mean(hsv[:, :, 1])
                # Normalize to 0-100
                saturation = (saturation / 255) * 100
                return float(saturation)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Saturation calculation failed: {e}")
            return 0.0
    
    def calculate_ssim(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Calculate structural similarity between two images
        
        Args:
            image1: First image
            image2: Second image
        
        Returns:
            SSIM score (0-1)
        """
        try:
            # Ensure same size
            if image1.size != image2.size:
                image2 = image2.resize(image1.size, Image.LANCZOS)
            
            # Convert to grayscale arrays
            img1_array = np.array(image1.convert('L'))
            img2_array = np.array(image2.convert('L'))
            
            # Calculate SSIM
            ssim_score = ssim(img1_array, img2_array)
            
            return float(ssim_score)
            
        except Exception as e:
            self.logger.warning(f"SSIM calculation failed: {e}")
            return 1.0  # Return 1.0 to avoid blocking
    
    def calculate_quality_metrics(
        self,
        original: Image.Image,
        processed: Image.Image
    ) -> QualityMetrics:
        """
        Calculate comprehensive quality metrics
        
        Args:
            original: Original image
            processed: Processed image
        
        Returns:
            Quality metrics
        """
        metrics = QualityMetrics()
        
        try:
            metrics.sharpness = self.calculate_sharpness(processed)
            metrics.edge_clarity = self.calculate_edge_clarity(processed)
            metrics.color_harmony = self.calculate_color_harmony(processed)
            metrics.brightness = self.calculate_brightness(processed)
            metrics.saturation = self.calculate_saturation(processed)
            
            # Calculate overall score (weighted average)
            metrics.overall_score = (
                metrics.sharpness * 0.25 +
                metrics.edge_clarity * 0.25 +
                metrics.color_harmony * 0.20 +
                min(metrics.brightness / 2.55, 100) * 0.15 +
                metrics.saturation * 0.15
            )
            
            self.logger.debug(
                f"Quality metrics - Sharpness: {metrics.sharpness:.1f}, "
                f"Edge: {metrics.edge_clarity:.1f}, Harmony: {metrics.color_harmony:.1f}, "
                f"Overall: {metrics.overall_score:.1f}"
            )
            
        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {e}")
        
        return metrics
    
    def postprocess(
        self,
        image: Image.Image,
        original_image: Image.Image = None,
        enable_sharpen: bool = True,
        enable_saturation: bool = True,
        enable_warm_tone: bool = True
    ) -> Tuple[Image.Image, QualityMetrics]:
        """
        Complete postprocessing pipeline
        
        Args:
            image: Generated image
            original_image: Original image for quality comparison
            enable_sharpen: Whether to apply sharpening
            enable_saturation: Whether to adjust saturation
            enable_warm_tone: Whether to apply warm tone
        
        Returns:
            Tuple of (postprocessed image, quality metrics)
        """
        # Store pre-postprocessing image for SSIM check
        pre_postprocess = image.copy()
        
        try:
            # Apply sharpening
            if enable_sharpen:
                image = self.sharpen(image)
            
            # Apply saturation adjustment
            if enable_saturation:
                image = self.adjust_saturation(image)
            
            # Apply warm tone
            if enable_warm_tone:
                image = self.apply_warm_tone(image)
            
            # Check for over-processing
            ssim_score = self.calculate_ssim(pre_postprocess, image)
            if ssim_score < self.min_ssim_threshold:
                self.logger.warning(
                    f"Over-processing detected (SSIM: {ssim_score:.3f}). "
                    f"Returning pre-postprocessed image."
                )
                image = pre_postprocess
            
            # Calculate quality metrics
            reference_image = original_image if original_image else pre_postprocess
            quality_metrics = self.calculate_quality_metrics(reference_image, image)
            
            self.logger.info("Postprocessing completed")
            return image, quality_metrics
            
        except Exception as e:
            self.logger.error(f"Postprocessing failed: {e}")
            # Return original image with default metrics
            quality_metrics = QualityMetrics()
            return pre_postprocess, quality_metrics
