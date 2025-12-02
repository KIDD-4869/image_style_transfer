"""
Preprocessing Pipeline for Enhanced Ghibli Processor
Handles image analysis and preprocessing before generation
"""
import cv2
import numpy as np
import logging
from PIL import Image, ImageEnhance, ImageStat
from typing import Tuple, Optional

try:
    from controlnet_aux import CannyDetector, MidasDetector
    CONTROLNET_AUX_AVAILABLE = True
except (ImportError, Exception) as e:
    CONTROLNET_AUX_AVAILABLE = False
    CannyDetector = None
    MidasDetector = None

from core.models import ContentType


class PreprocessingPipeline:
    """Handles image preprocessing and analysis"""
    
    def __init__(
        self,
        brightness_threshold: int = 100,
        brightness_boost: int = 20,
        contrast_threshold: float = 3.0,
        contrast_reduction: float = 0.15,
        target_size: int = 512,
        logger: logging.Logger = None
    ):
        """
        Initialize PreprocessingPipeline
        
        Args:
            brightness_threshold: Threshold for dark image detection
            brightness_boost: Amount to boost brightness
            contrast_threshold: Threshold for high contrast detection
            contrast_reduction: Amount to reduce contrast
            target_size: Target size for processing
            logger: Logger instance
        """
        self.brightness_threshold = brightness_threshold
        self.brightness_boost = brightness_boost
        self.contrast_threshold = contrast_threshold
        self.contrast_reduction = contrast_reduction
        self.target_size = target_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize ControlNet processors if available
        self.canny_detector = None
        self.depth_detector = None
        if CONTROLNET_AUX_AVAILABLE:
            try:
                self.canny_detector = CannyDetector()
                self.depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
                self.logger.debug("ControlNet auxiliary processors initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ControlNet processors: {e}")
    
    def analyze_content(self, image: Image.Image) -> ContentType:
        """
        Analyze image content type
        
        Args:
            image: Input image
        
        Returns:
            Detected content type
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Simple heuristic-based content detection
            # This is a placeholder - could be enhanced with ML models
            
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            # Analyze color distribution
            if len(img_array.shape) == 3:
                # Calculate color variance
                color_std = np.std(img_array, axis=(0, 1))
                avg_color_std = np.mean(color_std)
                
                # Detect edges
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges > 0) / (height * width)
                
                # Heuristic classification
                if edge_density > 0.15 and avg_color_std < 50:
                    content_type = ContentType.ARCHITECTURE
                elif aspect_ratio > 1.5 or aspect_ratio < 0.67:
                    content_type = ContentType.LANDSCAPE
                elif edge_density > 0.1 and 0.8 < aspect_ratio < 1.2:
                    content_type = ContentType.PORTRAIT
                else:
                    content_type = ContentType.MIXED
            else:
                content_type = ContentType.UNKNOWN
            
            self.logger.debug(f"Detected content type: {content_type.value}")
            return content_type
            
        except Exception as e:
            self.logger.warning(f"Content analysis failed: {e}. Using UNKNOWN.")
            return ContentType.UNKNOWN
    
    def adjust_brightness(self, image: Image.Image) -> Image.Image:
        """
        Adjust brightness for dark images
        
        Args:
            image: Input image
        
        Returns:
            Brightness-adjusted image
        """
        try:
            # Calculate average brightness
            stat = ImageStat.Stat(image)
            avg_brightness = sum(stat.mean) / len(stat.mean)
            
            if avg_brightness < self.brightness_threshold:
                # Calculate boost factor
                boost_factor = 1.0 + (self.brightness_boost / avg_brightness)
                boost_factor = min(boost_factor, 2.0)  # Cap at 2x
                
                # Apply brightness enhancement
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(boost_factor)
                
                self.logger.debug(
                    f"Brightness adjusted: {avg_brightness:.1f} -> "
                    f"{avg_brightness * boost_factor:.1f} (factor: {boost_factor:.2f})"
                )
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Brightness adjustment failed: {e}")
            return image
    
    def soften_contrast(self, image: Image.Image) -> Image.Image:
        """
        Soften contrast for high-contrast images
        
        Args:
            image: Input image
        
        Returns:
            Contrast-adjusted image
        """
        try:
            # Calculate contrast ratio
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            contrast_ratio = gray.std() / gray.mean() if gray.mean() > 0 else 0
            
            if contrast_ratio > self.contrast_threshold:
                # Calculate reduction factor
                reduction_factor = 1.0 - self.contrast_reduction
                
                # Apply contrast reduction
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(reduction_factor)
                
                self.logger.debug(
                    f"Contrast softened: ratio {contrast_ratio:.2f} "
                    f"(factor: {reduction_factor:.2f})"
                )
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Contrast adjustment failed: {e}")
            return image
    
    def generate_control_image(
        self,
        image: Image.Image,
        control_type: str = "canny",
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> Optional[Image.Image]:
        """
        Generate ControlNet condition image
        
        Args:
            image: Input image
            control_type: Type of control (canny or depth)
            low_threshold: Low threshold for Canny
            high_threshold: High threshold for Canny
        
        Returns:
            Control image or None if failed
        """
        try:
            if control_type == "canny":
                if self.canny_detector:
                    control_image = self.canny_detector(
                        image,
                        low_threshold=low_threshold,
                        high_threshold=high_threshold
                    )
                else:
                    # Fallback to OpenCV Canny
                    img_array = np.array(image)
                    if len(img_array.shape) == 3:
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img_array
                    
                    edges = cv2.Canny(gray, low_threshold, high_threshold)
                    control_image = Image.fromarray(edges)
                
                self.logger.debug("Canny control image generated")
                return control_image
                
            elif control_type == "depth":
                if self.depth_detector:
                    control_image = self.depth_detector(image)
                    self.logger.debug("Depth control image generated")
                    return control_image
                else:
                    self.logger.warning("Depth detector not available")
                    return None
            
            else:
                self.logger.warning(f"Unknown control type: {control_type}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Control image generation failed: {e}")
            return None
    
    def resize_image(
        self,
        image: Image.Image,
        target_size: int = None
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        Resize image to target size while maintaining aspect ratio
        
        Args:
            image: Input image
            target_size: Target size (default: self.target_size)
        
        Returns:
            Tuple of (resized image, original size)
        """
        if target_size is None:
            target_size = self.target_size
        
        original_size = image.size
        width, height = original_size
        
        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
        
        # Ensure dimensions are multiples of 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        self.logger.debug(f"Image resized: {original_size} -> {(new_width, new_height)}")
        return resized_image, original_size
    
    def preprocess(
        self,
        image: Image.Image,
        enable_brightness_adjust: bool = True,
        enable_contrast_adjust: bool = True
    ) -> Tuple[Image.Image, ContentType, Tuple[int, int]]:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image
            enable_brightness_adjust: Whether to adjust brightness
            enable_contrast_adjust: Whether to adjust contrast
        
        Returns:
            Tuple of (preprocessed image, content type, original size)
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            self.logger.debug(f"Converted image to RGB from {image.mode}")
        
        # Analyze content
        content_type = self.analyze_content(image)
        
        # Adjust brightness if needed
        if enable_brightness_adjust:
            image = self.adjust_brightness(image)
        
        # Soften contrast if needed
        if enable_contrast_adjust:
            image = self.soften_contrast(image)
        
        # Resize image
        image, original_size = self.resize_image(image)
        
        return image, content_type, original_size
