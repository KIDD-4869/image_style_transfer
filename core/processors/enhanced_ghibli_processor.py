"""
Enhanced Ghibli Processor
Main processor that coordinates all components for high-quality Ghibli-style conversion
"""
import time
import logging
from typing import Optional, Callable
from PIL import Image
import yaml
from pathlib import Path

from core.processors.base import BaseProcessor
from core.models import (
    ProcessingMode,
    ProcessorConfig,
    ProcessingResult,
    ContentType
)
from core.components.model_manager import ModelManager
from core.components.prompt_engineer import PromptEngineer
from core.components.preprocessing_pipeline import PreprocessingPipeline
from core.components.generation_engine import GenerationEngine
from core.components.postprocessing_pipeline import PostprocessingPipeline
from utils.enhanced_logging import (
    setup_enhanced_logging,
    log_processing_start,
    log_model_loading,
    log_generation_params,
    log_processing_complete,
    log_processing_error
)


class EnhancedGhibliProcessor(BaseProcessor):
    """
    Enhanced processor for high-quality Ghibli-style image conversion
    
    Coordinates all components:
    - ModelManager: Loads and manages SD models, LoRA, ControlNet
    - PromptEngineer: Generates optimized prompts
    - PreprocessingPipeline: Analyzes and preprocesses images
    - GenerationEngine: Generates images with optimized parameters
    - PostprocessingPipeline: Enhances and evaluates quality
    """
    
    def __init__(
        self,
        config_path: str = "config/enhanced_processor_config.yaml",
        logger: logging.Logger = None
    ):
        """
        Initialize Enhanced Ghibli Processor
        
        Args:
            config_path: Path to configuration file
            logger: Logger instance
        """
        super().__init__(
            name="EnhancedGhibliProcessor",
            description="Enhanced Ghibli-style processor with SD + ControlNet + LoRA"
        )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        if logger is None:
            log_config = self.config.get('logging', {})
            self.logger = setup_enhanced_logging(
                log_level=log_config.get('level', 'INFO'),
                log_file=log_config.get('log_file'),
                log_format=log_config.get('log_format')
            )
        else:
            self.logger = logger
        
        self.logger.info("Initializing Enhanced Ghibli Processor")
        
        # Initialize components
        try:
            # Create processor config
            model_config_dict = self.config.get('models', {})
            device_config = self.config.get('device', {})
            
            # Get cache configuration
            cache_config = self.config.get('cache', {})
            model_cache_dir = cache_config.get('model_cache_dir', 'default')
            
            processor_config = ProcessorConfig(
                base_model=model_config_dict.get('base_model', 'Linaqruf/anything-v3.0'),
                lora_model=model_config_dict.get('lora_model'),
                lora_weight=model_config_dict.get('lora_weight', 0.8),
                use_controlnet=self.config.get('controlnet', {}).get('enabled', True),
                controlnet_type=self.config.get('controlnet', {}).get('type', 'canny'),
                device=device_config.get('preferred', 'auto') if device_config.get('auto_detect', True) else 'cpu',
                dtype=device_config.get('dtype', 'float16'),
                model_cache_dir=model_cache_dir
            )
            
            # 使用全局模型缓存（避免每次都重新加载）
            from core.components.global_model_cache import get_global_model_manager, is_model_cached
            
            if is_model_cached():
                self.logger.info("✅ 检测到缓存的模型，将快速加载")
            else:
                self.logger.info("🔄 首次加载模型，需要较长时间")
                log_model_loading(
                    self.logger,
                    processor_config.base_model,
                    processor_config.lora_model is not None,
                    processor_config.use_controlnet
                )
            
            # 获取全局模型管理器（如果已缓存则秒级返回）
            self.model_manager = get_global_model_manager(
                processor_config,
                self.logger,
                progress_callback=None  # 初始化时不需要进度回调
            )
            
            # Initialize other components
            self.prompt_engineer = PromptEngineer(self.logger)
            
            preprocess_config = self.config.get('preprocessing', {})
            self.preprocessor = PreprocessingPipeline(
                brightness_threshold=preprocess_config.get('brightness_threshold', 100),
                brightness_boost=preprocess_config.get('brightness_boost', 20),
                contrast_threshold=preprocess_config.get('contrast_threshold', 3.0),
                contrast_reduction=preprocess_config.get('contrast_reduction', 0.15),
                target_size=preprocess_config.get('target_size', 512),
                logger=self.logger
            )
            
            self.generator = GenerationEngine(self.logger)
            
            postprocess_config = self.config.get('postprocessing', {})
            self.postprocessor = PostprocessingPipeline(
                sharpen_amount=postprocess_config.get('sharpen_amount', 0.3),
                saturation_factor=postprocess_config.get('saturation_factor', 1.2),
                warm_tone_strength=postprocess_config.get('warm_tone_strength', 0.15),
                min_ssim_threshold=postprocess_config.get('min_ssim_threshold', 0.85),
                logger=self.logger
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}", exc_info=True)
            raise
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                return config
            else:
                self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {}
    
    def process(
        self,
        image: Image.Image,
        mode: ProcessingMode = ProcessingMode.QUALITY,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        **kwargs
    ) -> ProcessingResult:
        """
        Process image with enhanced Ghibli style conversion
        
        Args:
            image: Input PIL Image
            mode: Processing mode (FAST, BALANCED, QUALITY, ULTRA)
            progress_callback: Optional callback for progress updates
            **kwargs: Additional parameters
        
        Returns:
            ProcessingResult with generated image and metadata
        """
        start_time = time.time()
        
        def update_progress(percent: int, message: str):
            """Helper to update progress"""
            if progress_callback:
                progress_callback(percent, message)
            self.logger.debug(f"Progress: {percent}% - {message}")
        
        try:
            # Stage 1: Model Loading (0-30%)
            from core.components.global_model_cache import is_model_cached
            
            if is_model_cached():
                # 模型已缓存，快速加载
                update_progress(0, "使用缓存的模型（秒级加载）...")
                pipeline = self.model_manager.get_pipeline()
                update_progress(15, "✅ 模型加载完成（使用缓存）")
            else:
                # 首次加载，需要较长时间
                update_progress(0, "首次加载模型，请稍候...")
                
                # 设置 ModelManager 的进度回调
                self.model_manager.progress_callback = update_progress
                
                pipeline = self.model_manager.get_pipeline()
                
                # 如果没有 ControlNet，进度应该在 15%
                if not self.model_manager.controlnet:
                    update_progress(15, "模型加载完成")
                # 否则进度已经在 30%（由 ControlNet 加载更新）
            
            # Stage 2: Preprocessing (30-40%)
            update_progress(30, "正在预处理图像...")
            log_processing_start(self.logger, image.size, mode.value)
            
            preprocessed_image, content_type, original_size = self.preprocessor.preprocess(image)
            update_progress(35, f"预处理完成 - 内容类型: {content_type.value}")
            
            # Generate ControlNet condition image if enabled
            control_image = None
            if self.model_manager.controlnet is not None:
                update_progress(36, "正在生成 ControlNet 控制图...")
                controlnet_config = self.config.get('controlnet', {})
                control_image = self.preprocessor.generate_control_image(
                    preprocessed_image,
                    control_type=controlnet_config.get('type', 'canny'),
                    low_threshold=controlnet_config.get('canny_low_threshold', 100),
                    high_threshold=controlnet_config.get('canny_high_threshold', 200)
                )
                update_progress(38, "ControlNet 控制图生成完成")
            
            # Stage 3: Prompt Generation (38-40%)
            update_progress(38, "正在生成提示词...")
            prompt = self.prompt_engineer.build_prompt(content_type)
            negative_prompt = self.prompt_engineer.build_negative_prompt()
            update_progress(40, "提示词生成完成")
            
            # Stage 4: Image Generation (40-85%)
            update_progress(40, "开始 AI 图像生成...")
            params = self.generator.configure_parameters(mode)
            log_generation_params(
                self.logger,
                params.strength,
                params.num_inference_steps,
                params.guidance_scale,
                params.scheduler
            )
            
            generated_image = self.generator.generate(
                pipeline=pipeline,
                image=preprocessed_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                params=params,
                control_image=control_image,
                seed=kwargs.get('seed'),
                progress_callback=update_progress
            )
            update_progress(85, "AI 图像生成完成")
            
            # Stage 5: Postprocessing (85-95%)
            update_progress(85, "正在后处理优化...")
            postprocess_config = self.config.get('postprocessing', {})
            processed_image, quality_metrics = self.postprocessor.postprocess(
                generated_image,
                original_image=image,
                enable_sharpen=postprocess_config.get('sharpen_enabled', True),
                enable_saturation=postprocess_config.get('saturation_enabled', True),
                enable_warm_tone=postprocess_config.get('warm_tone_enabled', True)
            )
            update_progress(90, "后处理优化完成")
            
            # Stage 6: Resize to original size (90-95%)
            update_progress(90, "Resizing to original size...")
            final_image = processed_image.resize(original_size, Image.LANCZOS)
            update_progress(95, "Resize complete")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Build metadata
            metadata = {
                'mode': mode.value,
                'content_type': content_type.value,
                'model': self.model_manager.config.base_model,
                'lora_used': self.model_manager.config.lora_model is not None,
                'controlnet_used': control_image is not None,
                'device': self.model_manager.device,
                'dtype': str(self.model_manager.dtype),
                'strength': params.strength,
                'steps': params.num_inference_steps,
                'guidance_scale': params.guidance_scale,
                'scheduler': params.scheduler,
                'original_size': original_size,
                'processing_size': preprocessed_image.size
            }
            
            # Create result
            result = ProcessingResult(
                success=True,
                image=final_image,
                processing_time=processing_time,
                metadata=metadata,
                quality_metrics=quality_metrics
            )
            
            update_progress(100, "Processing complete")
            log_processing_complete(self.logger, processing_time, quality_metrics.overall_score)
            
            # Check quality and log warning if needed
            if quality_metrics.overall_score < self.config.get('quality_metrics', {}).get('min_overall_score', 70):
                self.logger.warning(
                    f"Quality score below threshold: {quality_metrics.overall_score:.1f}. "
                    f"Consider adjusting parameters."
                )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            log_processing_error(self.logger, "generation", e)
            
            return ProcessingResult(
                success=False,
                processing_time=processing_time,
                error_message=f"{type(e).__name__}: {str(e)}"
            )
    
    def get_info(self) -> dict:
        """Get processor information"""
        return {
            'name': 'Enhanced Ghibli Processor',
            'version': '1.0.0',
            'device': self.model_manager.device,
            'base_model': self.model_manager.config.base_model,
            'lora_enabled': self.model_manager.config.lora_model is not None,
            'controlnet_enabled': self.model_manager.config.use_controlnet,
            'supported_modes': [mode.value for mode in ProcessingMode]
        }
