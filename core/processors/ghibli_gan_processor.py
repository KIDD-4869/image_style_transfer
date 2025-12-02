#!/usr/bin/env python3
"""
宫崎骏GAN风格处理器 - 使用深度学习模型进行真正的风格转换
"""

import torch
import numpy as np
from PIL import Image
import time
import logging
import os
import sys

# 添加archive路径以导入GAN模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../archive/deprecated_processors'))

from .base import BaseProcessor, ProcessingStrategy, ProcessingResult

logger = logging.getLogger(__name__)


class GhibliGANProcessor(BaseProcessor):
    """宫崎骏GAN风格处理器 - 真正的深度学习风格转换"""
    
    def __init__(self):
        super().__init__(
            name="GhibliGANProcessor",
            description="基于GAN的宫崎骏动画风格转换器 - 真正的风格迁移"
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gan_model = None
        self.model_loaded = False
        
        logger.info(f"GhibliGANProcessor初始化，设备: {self.device}")
    
    def _load_model(self):
        """延迟加载GAN模型"""
        if self.model_loaded:
            return True
        
        try:
            from ghibli_gan import GhibliGAN
            
            logger.info("正在加载GhibliGAN模型...")
            self.gan_model = GhibliGAN(device=self.device)
            
            # 尝试加载最佳模型
            model_path = "models/ghibli_gan/ghibli_gan_best.pth"
            if os.path.exists(model_path):
                try:
                    # 尝试标准加载
                    self.gan_model.load_model(model_path)
                    logger.info(f"✅ 成功加载模型: {model_path}")
                except Exception as e:
                    logger.warning(f"标准加载失败: {e}，尝试兼容加载...")
                    # 兼容旧格式
                    state = torch.load(model_path, map_location=self.device)
                    if 'model_state_dict' in state:
                        # 旧格式：只有一个生成器
                        self.gan_model.generator.load_state_dict(state['model_state_dict'])
                        logger.info(f"✅ 使用兼容模式加载模型: {model_path}")
                    else:
                        raise Exception("无法识别的模型格式")
                
                self.model_loaded = True
                return True
            else:
                logger.warning(f"⚠️ 模型文件不存在: {model_path}")
                logger.warning("将使用未训练的模型（效果可能不佳）")
                self.model_loaded = True
                return True
                
        except Exception as e:
            logger.error(f"❌ 加载GAN模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process(
        self, 
        image: Image.Image,
        strategy: ProcessingStrategy = ProcessingStrategy.BALANCED,
        **kwargs
    ) -> ProcessingResult:
        """
        使用GAN模型处理图像
        
        Args:
            image: 输入图像
            strategy: 处理策略（GAN模式下主要影响后处理）
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始GAN风格转换，策略: {strategy.value}")
            
            # 加载模型
            self.update_progress(5, 1, 10, 0)
            if not self._load_model():
                raise Exception("GAN模型加载失败")
            
            # 预处理
            self.update_progress(15, 2, 10, 0)
            logger.info("预处理图像...")
            
            # GAN推理
            self.update_progress(30, 3, 10, 0)
            logger.info("GAN生成中...")
            
            result_image = self.gan_model.inference(image)
            
            # 后处理（根据策略）
            self.update_progress(70, 7, 10, 0)
            if strategy == ProcessingStrategy.QUALITY:
                result_image = self._post_process_quality(result_image)
            elif strategy == ProcessingStrategy.BALANCED:
                result_image = self._post_process_balanced(result_image)
            else:  # FAST
                result_image = self._post_process_fast(result_image)
            
            self.update_progress(100, 10, 10, 0)
            
            processing_time = time.time() - start_time
            logger.info(f"GAN风格转换完成，耗时: {processing_time:.2f}秒")
            
            return ProcessingResult(
                success=True,
                image=result_image,
                processing_time=processing_time,
                metadata={
                    'strategy': strategy.value,
                    'model': 'GhibliGAN',
                    'device': self.device
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"GAN处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _post_process_fast(self, image: Image.Image) -> Image.Image:
        """快速后处理"""
        # 轻微锐化
        import cv2
        img_np = np.array(image)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(img_np, -1, kernel)
        return Image.fromarray(sharpened)
    
    def _post_process_balanced(self, image: Image.Image) -> Image.Image:
        """平衡后处理"""
        import cv2
        img_np = np.array(image)
        
        # 轻微锐化
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(img_np, -1, kernel)
        
        # 混合原图和锐化图
        result = cv2.addWeighted(img_np, 0.7, sharpened, 0.3, 0)
        
        return Image.fromarray(result)
    
    def _post_process_quality(self, image: Image.Image) -> Image.Image:
        """高质量后处理"""
        import cv2
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 细节增强
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(img_bgr, -1, kernel)
        
        # 混合
        result = cv2.addWeighted(img_bgr, 0.6, enhanced, 0.4, 0)
        
        # 轻微去噪
        result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)
        
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
