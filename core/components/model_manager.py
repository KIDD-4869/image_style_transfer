"""
Model Manager for Enhanced Ghibli Processor
Handles loading and configuration of SD models, LoRA, and ControlNet
"""
import torch
import logging
from typing import Optional
from pathlib import Path

try:
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        ControlNetModel,
        StableDiffusionControlNetImg2ImgPipeline,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        DDIMScheduler
    )
    from peft import PeftModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    StableDiffusionImg2ImgPipeline = None
    ControlNetModel = None

from core.models import ProcessorConfig


class ModelManager:
    """Manages model loading and configuration"""
    
    def __init__(self, config: ProcessorConfig, logger: logging.Logger = None, progress_callback=None):
        """
        Initialize ModelManager
        
        Args:
            config: Processor configuration
            logger: Logger instance
            progress_callback: Optional callback for progress updates
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers library not available. "
                "Install with: pip install diffusers transformers accelerate"
            )
        
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = self._detect_device()
        self.dtype = self._get_dtype()
        self.pipeline = None
        self.controlnet = None
        self.progress_callback = progress_callback
        
    def _detect_device(self) -> str:
        """Detect available device"""
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available():
            self.logger.info("CUDA device detected")
            return "cuda"
        elif torch.backends.mps.is_available():
            self.logger.info("MPS device detected (Apple Silicon)")
            return "mps"
        else:
            self.logger.info("Using CPU device")
            return "cpu"
    
    def _get_dtype(self) -> torch.dtype:
        """Get appropriate dtype for device"""
        if self.device == "cpu":
            return torch.float32
        elif self.config.dtype == "float16":
            return torch.float16
        else:
            return torch.float32
    
    def load_base_model(self) -> StableDiffusionImg2ImgPipeline:
        """
        Load base Stable Diffusion model
        
        Returns:
            Loaded pipeline
        """
        try:
            if self.progress_callback:
                self.progress_callback(1, "开始加载 Stable Diffusion 模型...")
            
            self.logger.info(f"Loading base model: {self.config.base_model}")
            
            # Get cache directory from config (None means use HuggingFace default)
            cache_dir = getattr(self.config, 'model_cache_dir', None)
            if cache_dir and cache_dir != 'default':
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Using custom model cache directory: {cache_path.absolute()}")
            else:
                cache_path = None
                self.logger.info("Using HuggingFace default cache directory (~/.cache/huggingface)")
            
            if self.progress_callback:
                self.progress_callback(3, "正在加载模型文件（首次会自动下载）...")
            
            # Load pipeline with cache directory
            kwargs = {
                'torch_dtype': self.dtype,
                'safety_checker': None,
                'requires_safety_checker': False
            }
            if cache_path:
                kwargs['cache_dir'] = str(cache_path)
            
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.config.base_model,
                **kwargs
            )
            
            if self.progress_callback:
                self.progress_callback(8, "模型文件加载完成，正在初始化...")
            
            # Move to device
            pipeline = pipeline.to(self.device)
            
            if self.progress_callback:
                self.progress_callback(12, "正在优化模型性能...")
            
            # Enable optimizations
            if self.device == "cuda":
                pipeline.enable_attention_slicing()
                if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                    try:
                        pipeline.enable_xformers_memory_efficient_attention()
                        self.logger.debug("xformers memory efficient attention enabled")
                    except Exception as e:
                        self.logger.warning(f"Could not enable xformers: {e}")
            
            if self.progress_callback:
                self.progress_callback(15, "Stable Diffusion 模型加载完成")
            
            self.logger.info("Base model loaded successfully")
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            # Try fallback to SD 1.5
            try:
                self.logger.warning("Attempting fallback to SD 1.5")
                cache_dir = getattr(self.config, 'model_cache_dir', None)
                if cache_dir and cache_dir != 'default':
                    cache_path = Path(cache_dir)
                    cache_path.mkdir(parents=True, exist_ok=True)
                else:
                    cache_path = None
                
                kwargs = {
                    'torch_dtype': self.dtype,
                    'safety_checker': None,
                    'requires_safety_checker': False
                }
                if cache_path:
                    kwargs['cache_dir'] = str(cache_path)
                
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    **kwargs
                )
                pipeline = pipeline.to(self.device)
                self.logger.info("Fallback model loaded successfully")
                return pipeline
            except Exception as fallback_error:
                self.logger.error(f"Fallback also failed: {fallback_error}")
                raise
    
    def load_lora(self, pipeline: StableDiffusionImg2ImgPipeline, lora_path: str, weight: float = 0.8):
        """
        Load and apply LoRA model
        
        Args:
            pipeline: Base pipeline
            lora_path: Path to LoRA model
            weight: LoRA weight (0-1)
        
        Returns:
            Pipeline with LoRA applied (or original if failed)
        """
        if not self.config.lora_model:
            self.logger.debug("LoRA not configured, skipping")
            return pipeline
        
        try:
            self.logger.info(f"Loading LoRA: {lora_path} with weight {weight}")
            
            # Check if LoRA file exists
            lora_file = Path(lora_path)
            if not lora_file.exists():
                self.logger.warning(f"LoRA file not found: {lora_path}")
                return pipeline
            
            # Load LoRA weights
            pipeline.load_lora_weights(lora_path)
            pipeline.fuse_lora(lora_scale=weight)
            
            self.logger.info("LoRA loaded and applied successfully")
            return pipeline
            
        except Exception as e:
            self.logger.warning(f"Failed to load LoRA: {e}. Continuing without LoRA.")
            return pipeline
    
    def load_controlnet(self, controlnet_type: str = "canny") -> Optional[ControlNetModel]:
        """
        Load ControlNet model
        
        Args:
            controlnet_type: Type of ControlNet (canny, depth, etc.)
        
        Returns:
            ControlNet model or None if failed
        """
        if not self.config.use_controlnet:
            self.logger.debug("ControlNet not enabled, skipping")
            return None
        
        try:
            if self.progress_callback:
                self.progress_callback(16, "开始加载 ControlNet 模型...")
            
            self.logger.info(f"Loading ControlNet: {controlnet_type}")
            
            # Map controlnet type to model
            controlnet_models = {
                "canny": "lllyasviel/control_v11p_sd15_canny",
                "depth": "lllyasviel/control_v11f1p_sd15_depth"
            }
            
            model_id = controlnet_models.get(controlnet_type, controlnet_models["canny"])
            
            # Get cache directory from config (None means use HuggingFace default)
            cache_dir = getattr(self.config, 'model_cache_dir', None)
            if cache_dir and cache_dir != 'default':
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
            else:
                cache_path = None
            
            if self.progress_callback:
                self.progress_callback(18, "正在加载 ControlNet 文件（首次会自动下载约1.5GB）...")
            
            kwargs = {'torch_dtype': self.dtype}
            if cache_path:
                kwargs['cache_dir'] = str(cache_path)
            
            controlnet = ControlNetModel.from_pretrained(
                model_id,
                **kwargs
            )
            
            if self.progress_callback:
                self.progress_callback(28, "ControlNet 文件加载完成，正在初始化...")
            
            controlnet = controlnet.to(self.device)
            
            if self.progress_callback:
                self.progress_callback(30, "ControlNet 模型加载完成")
            
            self.logger.info("ControlNet loaded successfully")
            return controlnet
            
        except Exception as e:
            self.logger.warning(f"Failed to load ControlNet: {e}. Continuing without ControlNet.")
            return None
    
    def get_pipeline(self) -> StableDiffusionImg2ImgPipeline:
        """
        Get configured pipeline with all models loaded
        
        Returns:
            Configured pipeline
        """
        if self.pipeline is not None:
            return self.pipeline
        
        # Load base model
        pipeline = self.load_base_model()
        
        # Load LoRA if configured
        if self.config.lora_model:
            pipeline = self.load_lora(pipeline, self.config.lora_model, self.config.lora_weight)
        
        # Load ControlNet if configured
        if self.config.use_controlnet:
            self.controlnet = self.load_controlnet(self.config.controlnet_type)
            
            # If ControlNet loaded, create ControlNet pipeline
            if self.controlnet is not None:
                try:
                    from diffusers import StableDiffusionControlNetImg2ImgPipeline
                    
                    # Create new pipeline with ControlNet
                    pipeline = StableDiffusionControlNetImg2ImgPipeline(
                        vae=pipeline.vae,
                        text_encoder=pipeline.text_encoder,
                        tokenizer=pipeline.tokenizer,
                        unet=pipeline.unet,
                        controlnet=self.controlnet,
                        scheduler=pipeline.scheduler,
                        safety_checker=None,
                        feature_extractor=None,
                        requires_safety_checker=False
                    )
                    pipeline = pipeline.to(self.device)
                    self.logger.info("ControlNet pipeline created")
                except Exception as e:
                    self.logger.warning(f"Failed to create ControlNet pipeline: {e}")
        
        self.pipeline = pipeline
        return pipeline
    
    def get_device_info(self) -> dict:
        """Get device information"""
        return {
            'device': self.device,
            'dtype': str(self.dtype),
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available()
        }
