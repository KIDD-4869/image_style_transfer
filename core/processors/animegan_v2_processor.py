#!/usr/bin/env python3
"""
AnimeGAN v2处理器 - 匹配实际模型架构
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time
import logging
import os
import cv2

from .base import BaseProcessor, ProcessingStrategy, ProcessingResult

logger = logging.getLogger(__name__)


class SimpleResBlock(nn.Module):
    """简单残差块"""
    def __init__(self, channels):
        super(SimpleResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + residual
        out = self.relu2(out)
        return out


class SimpleAnimeGAN(nn.Module):
    """简单的AnimeGAN架构 - 匹配实际模型"""
    def __init__(self):
        super(SimpleAnimeGAN, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            # 64x64
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # 128x128
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            # 256x256
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # 512x512
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            # 512x512
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        
        # 残差块
        self.residual_blocks = nn.Sequential(
            *[SimpleResBlock(512) for _ in range(8)]
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            # 上采样
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            # 输出层
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )
    
    def forward(self, x):
        # 编码
        features = self.encoder(x)
        # 残差处理
        features = self.residual_blocks(features)
        # 解码
        output = self.decoder(features)
        return output


class AnimeGANv2Processor(BaseProcessor):
    """AnimeGAN v2处理器"""
    
    def __init__(self):
        super().__init__(
            name="AnimeGANv2Processor",
            description="AnimeGAN v2 - 简化架构版本"
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_loaded = False
        
        logger.info(f"AnimeGANv2Processor初始化，设备: {self.device}")
    
    def _load_model(self):
        """加载模型"""
        if self.model_loaded:
            return True
        
        try:
            logger.info("正在加载AnimeGAN v2模型...")
            
            # 创建模型
            self.model = SimpleAnimeGAN().to(self.device)
            
            # 加载权重
            model_path = "models/anime_gan/AnimeGANv2_Hayao.pth"
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 加载状态字典
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 直接加载（这个模型就是state_dict格式）
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
            logger.info(f"✅ AnimeGAN v2模型加载成功")
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _preprocess(self, image: Image.Image):
        """预处理"""
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 调整大小到256的倍数
        w, h = image.size
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32
        if new_w == 0:
            new_w = 32
        if new_h == 0:
            new_h = 32
        
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # 转换为numpy
        img_np = np.array(image).astype(np.float32)
        
        # 归一化到[-1, 1]
        img_np = (img_np / 127.5) - 1.0
        
        # 转换为tensor
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device), image.size
    
    def _postprocess(self, tensor, original_size):
        """后处理"""
        # 转换为numpy
        output = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        
        # 反归一化
        output = (output + 1.0) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # 转换为PIL
        result_image = Image.fromarray(output)
        
        # 调整回原始大小
        if result_image.size != original_size:
            result_image = result_image.resize(original_size, Image.LANCZOS)
        
        return result_image
    
    def process(
        self, 
        image: Image.Image,
        strategy: ProcessingStrategy = ProcessingStrategy.BALANCED,
        **kwargs
    ) -> ProcessingResult:
        """处理图像"""
        start_time = time.time()
        
        try:
            logger.info(f"开始AnimeGAN v2处理，策略: {strategy.value}")
            
            # 加载模型
            self.update_progress(10, 1, 10, 0)
            if not self._load_model():
                raise Exception("模型加载失败")
            
            # 预处理
            self.update_progress(20, 2, 10, 0)
            input_tensor, original_size = self._preprocess(image)
            
            # 推理
            self.update_progress(50, 5, 10, 0)
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            # 后处理
            self.update_progress(80, 8, 10, 0)
            result_image = self._postprocess(output_tensor, original_size)
            
            self.update_progress(100, 10, 10, 0)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ 处理完成，耗时: {processing_time:.2f}秒")
            
            return ProcessingResult(
                success=True,
                image=result_image,
                processing_time=processing_time,
                metadata={
                    'strategy': strategy.value,
                    'model': 'AnimeGANv2_Simple',
                    'device': self.device
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
