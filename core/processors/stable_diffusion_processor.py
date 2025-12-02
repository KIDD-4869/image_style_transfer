#!/usr/bin/env python3
"""
Stable Diffusion处理器 - 真正的AI艺术风格转换
使用Stable Diffusion img2img进行宫崎骏风格转换
"""

import torch
import numpy as np
from PIL import Image
import time
import logging
import os

from .base import BaseProcessor, ProcessingStrategy, ProcessingResult

logger = logging.getLogger(__name__)


class StableDiffusionProcessor(BaseProcessor):
    """Stable Diffusion处理器 - 最强的风格转换"""
    
    def __init__(self):
        super().__init__(
            name="StableDiffusionProcessor",
            description="Stable Diffusion - AI艺术风格转换"
        )
        
        # 设备检测 - Requirements 4.1, 4.2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipe = None
        self.model_loaded = False
        
        # 宫崎骏风格提示词 - Requirements 5.1, 5.2
        # 强调重新绘制和动漫风格
        self.ghibli_prompt = (
            "Studio Ghibli anime style, Hayao Miyazaki art, "
            "hand-drawn anime illustration, cel shading, anime painting, "
            "vibrant colors, soft lighting, dreamy atmosphere, "
            "detailed anime artwork, high quality anime art, "
            "whimsical, magical realism, anime character design, "
            "traditional animation style, painterly anime style"
        )
        
        self.negative_prompt = (
            "photorealistic, photo, realistic, photograph, real life, "
            "3d render, cgi, ugly, blurry, low quality, bad anatomy, "
            "watermark, text, signature, deformed, distorted"
        )
        
        # 记录设备信息 - Requirements 4.5
        logger.info(f"StableDiffusionProcessor初始化，设备: {self.device}")
    
    def _load_model(self):
        """
        加载Stable Diffusion模型
        Requirements 1.1, 1.4, 4.1, 4.2, 4.3, 4.4
        """
        if self.model_loaded:
            return True
        
        try:
            # 记录模型加载开始 - Requirements 7.5
            logger.info("正在加载Stable Diffusion模型...")
            logger.info("⚠️ 首次加载需要下载模型（约4GB），请耐心等待...")
            
            from diffusers import StableDiffusionImg2ImgPipeline
            
            # 加载模型 - Requirements 1.1
            model_id = "runwayml/stable-diffusion-v1-5"
            
            # 根据设备选择精度 - Requirements 4.1, 4.2
            if self.device == 'cpu':
                # CPU模式：使用float32精度 - Requirements 4.2
                logger.info("使用CPU模式，float32精度")
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                # GPU模式：使用float16精度 - Requirements 4.1
                logger.info("使用GPU模式，float16精度，CUDA加速")
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            self.pipe = self.pipe.to(self.device)
            
            # 应用优化设置 - Requirements 4.3, 4.4
            if self.device == 'cpu':
                # CPU优化 - Requirements 4.3
                logger.info("启用CPU优化设置")
            else:
                # GPU优化：启用attention slicing - Requirements 4.4
                logger.info("启用attention slicing优化")
                self.pipe.enable_attention_slicing()
            
            # 记录成功 - Requirements 7.5
            logger.info(f"✅ Stable Diffusion模型加载成功，设备: {self.device}")
            self.model_loaded = True
            return True
            
        except Exception as e:
            # 错误处理 - Requirements 1.5, 7.4, 7.5
            logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
            return False
    
    def _preprocess(self, image: Image.Image, target_size=512):
        """
        预处理图像
        Requirements 1.2, 6.1, 6.2, 6.3, 6.5
        
        Args:
            image: 输入图像
            target_size: 目标尺寸（默认512）
            
        Returns:
            预处理后的图像
        """
        # 转换为RGB模式 - Requirements 6.1
        if image.mode != 'RGB':
            logger.info(f"转换图像模式: {image.mode} -> RGB")
            image = image.convert('RGB')
        
        # 调整大小（保持宽高比）- Requirements 1.2, 6.2
        w, h = image.size
        aspect_ratio = w / h
        
        if aspect_ratio > 1:
            # 宽图
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
        else:
            # 高图
            new_h = target_size
            new_w = int(target_size * aspect_ratio)
        
        # 确保是8的倍数 - Requirements 6.3
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        
        # 使用LANCZOS插值 - Requirements 6.5
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        logger.info(f"预处理完成: {w}x{h} -> {new_w}x{new_h}")
        
        return image
    
    def process(
        self, 
        image: Image.Image,
        strategy: ProcessingStrategy = ProcessingStrategy.BALANCED,
        **kwargs
    ) -> ProcessingResult:
        """
        使用Stable Diffusion处理图像
        
        Args:
            image: 输入图像
            strategy: 处理策略
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始Stable Diffusion风格转换，策略: {strategy.value}")
            
            # 加载模型 - Requirements 3.1
            self.update_progress(5, 1, 10, 0)
            logger.info("阶段1: 加载模型...")
            if not self._load_model():
                raise Exception("Stable Diffusion模型加载失败")
            
            # 预处理 - Requirements 3.2
            self.update_progress(15, 2, 10, 0)
            logger.info("阶段2: 预处理图像...")
            original_size = image.size
            processed_image = self._preprocess(image)
            
            # 根据策略设置参数 - Requirements 2.1, 2.2, 2.3
            # 提高 strength 以实现更强的风格转换和重新绘制
            if strategy == ProcessingStrategy.FAST:
                # 快速模式 - Requirements 2.1
                strength = 0.75  # 提高从 0.5 到 0.75，更多重新生成
                num_inference_steps = 25  # 提高从 20 到 25
                guidance_scale = 8.0  # 提高从 7.0 到 8.0，更强的风格引导
            elif strategy == ProcessingStrategy.QUALITY:
                # 高质量模式 - Requirements 2.3
                strength = 0.95  # 提高从 0.75 到 0.95，几乎完全重新生成
                num_inference_steps = 60  # 提高从 50 到 60
                guidance_scale = 9.0  # 提高从 8.0 到 9.0
            else:  # BALANCED
                # 标准模式 - Requirements 2.2
                strength = 0.85  # 提高从 0.65 到 0.85
                num_inference_steps = 40  # 提高从 30 到 40
                guidance_scale = 8.5  # 提高从 7.5 到 8.5
            
            logger.info(f"策略参数: strength={strength}, steps={num_inference_steps}, guidance={guidance_scale}")
            logger.info(f"💡 使用高 strength 值以实现更强的动漫风格重绘")
            
            # Stable Diffusion推理 - Requirements 3.3
            self.update_progress(30, 3, 10, 0)
            logger.info(f"阶段3: Stable Diffusion生成中... (steps={num_inference_steps})")
            logger.info(f"🎨 使用 strength={strength:.2f} 进行{'强力重绘' if strength > 0.8 else '风格转换'}")
            
            # 使用更高的 strength 实现更强的重绘效果
            result_image = self.pipe(
                prompt=self.ghibli_prompt,
                negative_prompt=self.negative_prompt,
                image=processed_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                # 添加更多控制参数以提高质量
                eta=0.0  # 确定性采样，提高一致性
            ).images[0]
            
            # 后处理：调整回原始大小 - Requirements 3.4, 6.4
            self.update_progress(90, 9, 10, 0)
            logger.info("阶段4: 后处理...")
            if result_image.size != original_size:
                logger.info(f"调整输出尺寸: {result_image.size} -> {original_size}")
                result_image = result_image.resize(original_size, Image.LANCZOS)
            
            self.update_progress(100, 10, 10, 0)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Stable Diffusion转换完成，耗时: {processing_time:.2f}秒")
            
            # 返回结果 - Requirements 7.1, 7.2, 7.3
            return ProcessingResult(
                success=True,
                image=result_image,
                processing_time=processing_time,
                metadata={
                    'strategy': strategy.value,
                    'model': 'Stable Diffusion v1.5',
                    'device': self.device,
                    'strength': strength,
                    'steps': num_inference_steps,
                    'guidance_scale': guidance_scale
                }
            )
            
        except Exception as e:
            # 错误处理 - Requirements 3.5, 7.4, 7.5
            processing_time = time.time() - start_time
            logger.error(f"❌ Stable Diffusion处理失败: {e}", exc_info=True, extra={
                'image_size': image.size if image else None,
                'strategy': strategy.value,
                'device': self.device
            })
            
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
