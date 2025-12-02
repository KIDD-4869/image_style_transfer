"""
Generation Engine for Enhanced Ghibli Processor
Handles image generation with optimized parameters
"""
import logging
from typing import Optional
from PIL import Image

try:
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        DDIMScheduler
    )
    DIFFUSERS_AVAILABLE = True
except (ImportError, Exception) as e:
    DIFFUSERS_AVAILABLE = False
    StableDiffusionImg2ImgPipeline = None
    DPMSolverMultistepScheduler = None
    EulerAncestralDiscreteScheduler = None
    DDIMScheduler = None

from core.models import ProcessingMode, GenerationParams


class GenerationEngine:
    """Handles image generation with SD pipeline"""
    
    # Mode configuration mapping (进一步降低 strength 以保留原图内容)
    MODE_CONFIGS = {
        ProcessingMode.FAST: GenerationParams(
            strength=0.30,  # 进一步降低以保留原图内容
            num_inference_steps=25,
            guidance_scale=7.0,
            scheduler="euler_a",
            controlnet_conditioning_scale=0.5
        ),
        ProcessingMode.BALANCED: GenerationParams(
            strength=0.35,  # 进一步降低以保留原图内容
            num_inference_steps=40,
            guidance_scale=7.5,
            scheduler="dpm_2m_karras",
            controlnet_conditioning_scale=0.6
        ),
        ProcessingMode.QUALITY: GenerationParams(
            strength=0.40,  # 进一步降低以保留原图内容
            num_inference_steps=60,
            guidance_scale=8.0,
            scheduler="dpm_2m_karras",
            controlnet_conditioning_scale=0.7
        ),
        ProcessingMode.ULTRA: GenerationParams(
            strength=0.45,  # 进一步降低以保留原图内容
            num_inference_steps=80,
            guidance_scale=8.5,
            scheduler="dpm_2m_karras",
            controlnet_conditioning_scale=0.8
        )
    }
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize GenerationEngine
        
        Args:
            logger: Logger instance
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library not available")
        
        self.logger = logger or logging.getLogger(__name__)
    
    def configure_parameters(self, mode: ProcessingMode) -> GenerationParams:
        """
        Get generation parameters for mode
        
        Args:
            mode: Processing mode
        
        Returns:
            Generation parameters
        """
        params = self.MODE_CONFIGS.get(mode, self.MODE_CONFIGS[ProcessingMode.BALANCED])
        self.logger.debug(
            f"Configured parameters for {mode.value}: "
            f"strength={params.strength}, steps={params.num_inference_steps}, "
            f"guidance={params.guidance_scale}, scheduler={params.scheduler}"
        )
        return params
    
    def select_scheduler(self, pipeline: StableDiffusionImg2ImgPipeline, scheduler_name: str):
        """
        Select and configure scheduler
        
        Args:
            pipeline: SD pipeline
            scheduler_name: Name of scheduler to use
        
        Returns:
            Pipeline with configured scheduler
        """
        try:
            if scheduler_name == "dpm_2m_karras":
                scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config,
                    use_karras_sigmas=True
                )
                self.logger.debug("Using DPM++ 2M Karras scheduler")
                
            elif scheduler_name == "euler_a":
                scheduler = EulerAncestralDiscreteScheduler.from_config(
                    pipeline.scheduler.config
                )
                self.logger.debug("Using Euler Ancestral scheduler")
                
            elif scheduler_name == "ddim":
                scheduler = DDIMScheduler.from_config(
                    pipeline.scheduler.config
                )
                self.logger.debug("Using DDIM scheduler")
                
            else:
                self.logger.warning(f"Unknown scheduler: {scheduler_name}, using DDIM")
                scheduler = DDIMScheduler.from_config(
                    pipeline.scheduler.config
                )
            
            pipeline.scheduler = scheduler
            return pipeline
            
        except Exception as e:
            self.logger.warning(f"Failed to set scheduler {scheduler_name}: {e}. Using default.")
            # Fallback to DDIM
            try:
                scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
                pipeline.scheduler = scheduler
                self.logger.info("Fallback to DDIM scheduler")
            except Exception as fallback_error:
                self.logger.error(f"Scheduler fallback also failed: {fallback_error}")
            
            return pipeline
    
    def generate(
        self,
        pipeline: StableDiffusionImg2ImgPipeline,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        params: GenerationParams,
        control_image: Optional[Image.Image] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Image.Image:
        """
        Generate image using SD pipeline
        
        Args:
            pipeline: Configured SD pipeline
            image: Input image
            prompt: Positive prompt
            negative_prompt: Negative prompt
            params: Generation parameters
            control_image: ControlNet condition image (optional)
            seed: Random seed (optional)
        
        Returns:
            Generated image
        """
        try:
            # Configure scheduler
            pipeline = self.select_scheduler(pipeline, params.scheduler)
            
            # Prepare generation kwargs
            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "strength": params.strength,
                "num_inference_steps": params.num_inference_steps,
                "guidance_scale": params.guidance_scale,
            }
            
            # Add ControlNet image if available
            if control_image is not None and hasattr(pipeline, 'controlnet'):
                gen_kwargs["control_image"] = control_image
                gen_kwargs["controlnet_conditioning_scale"] = params.controlnet_conditioning_scale
                self.logger.debug(
                    f"Using ControlNet with conditioning scale: "
                    f"{params.controlnet_conditioning_scale}"
                )
            
            # Set seed if provided
            if seed is not None:
                import torch
                generator = torch.Generator(device=pipeline.device).manual_seed(seed)
                gen_kwargs["generator"] = generator
                self.logger.debug(f"Using seed: {seed}")
            
            # Add progress callback
            if progress_callback:
                def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                    # Calculate progress (40% to 85% of total)
                    progress = 40 + int((step_index + 1) / params.num_inference_steps * 45)
                    progress_callback(progress, f"AI 生成中... 第 {step_index + 1}/{params.num_inference_steps} 步")
                    return callback_kwargs
                
                gen_kwargs["callback_on_step_end"] = callback_on_step_end
            
            # Generate image
            self.logger.info("Starting image generation...")
            result = pipeline(**gen_kwargs)
            
            generated_image = result.images[0]
            self.logger.info("Image generation completed")
            
            return generated_image
            
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}", exc_info=True)
            raise
    
    def get_scheduler_name(self, pipeline: StableDiffusionImg2ImgPipeline) -> str:
        """
        Get current scheduler name
        
        Args:
            pipeline: SD pipeline
        
        Returns:
            Scheduler name
        """
        scheduler_class = pipeline.scheduler.__class__.__name__
        
        # Map class names to friendly names
        scheduler_map = {
            "DPMSolverMultistepScheduler": "dpm_2m_karras",
            "EulerAncestralDiscreteScheduler": "euler_a",
            "DDIMScheduler": "ddim"
        }
        
        return scheduler_map.get(scheduler_class, scheduler_class)
