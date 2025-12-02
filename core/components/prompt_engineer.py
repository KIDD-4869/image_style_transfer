"""
Prompt Engineer for Enhanced Ghibli Processor
Generates optimized prompts for Ghibli-style image generation
"""
import logging
from typing import Dict
from core.models import ContentType


class PromptEngineer:
    """Generates and optimizes prompts for image generation"""
    
    # Base Ghibli style prompt (强调保持原图内容)
    BASE_GHIBLI_PROMPT = (
        "Studio Ghibli anime style, Hayao Miyazaki art, "
        "hand-drawn animation, cel shading, soft lighting, "
        "vibrant colors, dreamy atmosphere, whimsical, "
        "detailed background, painterly, watercolor style, "
        "magical realism, high quality, masterpiece, "
        "preserve original composition, keep original subject, "
        "maintain original content and structure"
    )
    
    # Content-specific keywords (强调保持原图主体)
    CONTENT_KEYWORDS = {
        ContentType.PORTRAIT: "anime character, expressive eyes, soft features, detailed face, keep original subject",
        ContentType.LANDSCAPE: "natural scenery, lush vegetation, sky and clouds, scenic view, preserve landscape",
        ContentType.ARCHITECTURE: "detailed buildings, clean lines, perspective, architectural details, maintain structure",
        ContentType.MIXED: "balanced composition, harmonious elements, detailed scene, preserve original elements",
        ContentType.UNKNOWN: "detailed illustration, artistic composition, keep original content"
    }
    
    # Negative prompt
    NEGATIVE_PROMPT = (
        "photorealistic, photo, realistic, 3d render, cgi, "
        "blurry, low quality, bad anatomy, deformed, ugly, "
        "watermark, text, signature, cropped, out of frame, "
        "worst quality, low res, jpeg artifacts, duplicate, "
        "morbid, mutilated, extra limbs, poorly drawn, "
        "bad proportions, gross proportions, malformed"
    )
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize PromptEngineer
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def build_prompt(
        self,
        content_type: ContentType = ContentType.UNKNOWN,
        style_intensity: float = 1.0,
        custom_keywords: str = ""
    ) -> str:
        """
        Build positive prompt for generation
        
        Args:
            content_type: Type of content in the image
            style_intensity: Intensity of style application (0-1)
            custom_keywords: Additional custom keywords
        
        Returns:
            Constructed prompt string
        """
        # Start with base prompt
        prompt_parts = [self.BASE_GHIBLI_PROMPT]
        
        # Add content-specific keywords
        if content_type in self.CONTENT_KEYWORDS:
            content_keywords = self.CONTENT_KEYWORDS[content_type]
            prompt_parts.append(content_keywords)
            self.logger.debug(f"Added {content_type.value} keywords: {content_keywords}")
        
        # Add custom keywords if provided
        if custom_keywords:
            prompt_parts.append(custom_keywords)
            self.logger.debug(f"Added custom keywords: {custom_keywords}")
        
        # Join all parts
        prompt = ", ".join(prompt_parts)
        
        # Apply style intensity through prompt weighting (if needed)
        if style_intensity < 1.0:
            # Could implement prompt weighting here
            pass
        
        self.logger.debug(f"Built prompt: {prompt[:100]}...")
        return prompt
    
    def build_negative_prompt(self, additional_negatives: str = "") -> str:
        """
        Build negative prompt
        
        Args:
            additional_negatives: Additional negative keywords
        
        Returns:
            Negative prompt string
        """
        negative_parts = [self.NEGATIVE_PROMPT]
        
        if additional_negatives:
            negative_parts.append(additional_negatives)
            self.logger.debug(f"Added additional negatives: {additional_negatives}")
        
        negative_prompt = ", ".join(negative_parts)
        self.logger.debug(f"Built negative prompt: {negative_prompt[:100]}...")
        return negative_prompt
    
    def apply_weights(self, prompt: str, weights: Dict[str, float] = None) -> str:
        """
        Apply weights to prompt keywords (for advanced usage)
        
        Args:
            prompt: Base prompt
            weights: Dictionary of keyword -> weight mappings
        
        Returns:
            Weighted prompt string
        """
        if not weights:
            return prompt
        
        # This is a placeholder for advanced prompt weighting
        # Could implement (keyword:weight) syntax here
        weighted_prompt = prompt
        
        for keyword, weight in weights.items():
            if keyword in weighted_prompt:
                # Replace keyword with weighted version
                weighted_keyword = f"({keyword}:{weight:.2f})"
                weighted_prompt = weighted_prompt.replace(keyword, weighted_keyword)
        
        return weighted_prompt
    
    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate that prompt contains required keywords
        
        Args:
            prompt: Prompt to validate
        
        Returns:
            True if valid, False otherwise
        """
        required_keywords = ["Studio Ghibli", "Hayao Miyazaki"]
        
        for keyword in required_keywords:
            if keyword not in prompt:
                self.logger.warning(f"Prompt missing required keyword: {keyword}")
                return False
        
        return True
    
    def validate_negative_prompt(self, negative_prompt: str) -> bool:
        """
        Validate that negative prompt contains required exclusions
        
        Args:
            negative_prompt: Negative prompt to validate
        
        Returns:
            True if valid, False otherwise
        """
        required_exclusions = ["photorealistic", "photo", "realistic"]
        
        for exclusion in required_exclusions:
            if exclusion not in negative_prompt.lower():
                self.logger.warning(f"Negative prompt missing required exclusion: {exclusion}")
                return False
        
        return True
