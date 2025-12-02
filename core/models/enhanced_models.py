"""
Data models for Enhanced Ghibli Processor
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from PIL import Image


class ProcessingMode(Enum):
    """Processing mode enumeration"""
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    ULTRA = "ultra"


class ContentType(Enum):
    """Image content type enumeration"""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    ARCHITECTURE = "architecture"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class ProcessorConfig:
    """Configuration for the enhanced processor"""
    base_model: str = "Linaqruf/anything-v3.0"
    lora_model: Optional[str] = "ghibli-style-lora-v1"
    lora_weight: float = 0.8
    use_controlnet: bool = True
    controlnet_type: str = "canny"
    device: str = "auto"
    dtype: str = "float16"
    model_cache_dir: str = "default"  # 模型缓存目录，"default"使用HuggingFace默认缓存


@dataclass
class GenerationParams:
    """Parameters for image generation"""
    strength: float
    num_inference_steps: int
    guidance_scale: float
    scheduler: str
    controlnet_conditioning_scale: float = 0.5


@dataclass
class QualityMetrics:
    """Quality metrics for processed images"""
    sharpness: float = 0.0
    edge_clarity: float = 0.0
    color_harmony: float = 0.0
    brightness: float = 0.0
    saturation: float = 0.0
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'sharpness': self.sharpness,
            'edge_clarity': self.edge_clarity,
            'color_harmony': self.color_harmony,
            'brightness': self.brightness,
            'saturation': self.saturation,
            'overall_score': self.overall_score
        }


@dataclass
class ProcessingResult:
    """Result of image processing"""
    success: bool
    image: Optional[Image.Image] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Optional[QualityMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding image)"""
        result = {
            'success': self.success,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'metadata': self.metadata
        }
        if self.quality_metrics:
            result['quality_metrics'] = self.quality_metrics.to_dict()
        return result
