#!/usr/bin/env python3
"""
AnimeGANv2处理器 - 真正的深度学习动漫风格转换
基于AnimeGANv2 Hayao模型
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


class ConvNormLReLU(nn.Sequential):
    """卷积+归一化+激活"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError
        
        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


class InvertedResBlock(nn.Module):
    """倒残差块"""
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()
        
        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))
        
        # Depthwise
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # Pointwise
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class AnimeGANv2Generator(nn.Module):
    """AnimeGANv2生成器"""
    def __init__(self):
        super(AnimeGANv2Generator, self).__init__()
        
        # 编码器
        self.encode_blocks = nn.Sequential(
            ConvNormLReLU(3, 64, kernel_size=7, padding=3),
            ConvNormLReLU(64, 128, stride=2, padding=1),
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 256, stride=2, padding=1),
            ConvNormLReLU(256, 256),
        )
        
        # 残差块
        self.res_blocks = nn.Sequential(
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
            InvertedResBlock(256, 256),
        )
        
        # 解码器
        self.decode_blocks = nn.Sequential(
            ConvNormLReLU(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 3, kernel_size=7, padding=3),
        )
    
    def forward(self, input):
        out = self.encode_blocks(input)
        out = self.res_blocks(out)
        out = self.decode_blocks(out)
        return out


class AnimeGANProcessor(BaseProcessor):
    """AnimeGAN处理器 - 真正的深度学习风格转换"""
    
    def __init__(self):
        super().__init__(
            name="AnimeGANProcessor",
            description="AnimeGANv2 Hayao - 宫崎骏风格深度学习转换器"
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_loaded = False
        
        logger.info(f"AnimeGANProcessor初始化，设备: {self.device}")
    
    def _load_model(self):
        """加载AnimeGAN模型"""
        if self.model_loaded:
            return True
        
        try:
            logger.info("正在加载AnimeGANv2模型...")
            
            # 创建模型
            self.model = AnimeGANv2Generator().to(self.device)
            
            # 加载权重
            model_path = "models/anime_gan/AnimeGANv2_Hayao.pth"
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 加载状态字典
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 处理可能的键名不匹配
            if 'generator' in state_dict:
                state_dict = state_dict['generator']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
            logger.info(f"✅ AnimeGAN模型加载成功: {model_path}")
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ AnimeGAN模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _preprocess(self, image: Image.Image, target_size=512):
        """预处理图像"""
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 调整大小（保持宽高比）
        w, h = image.size
        if max(w, h) > target_size:
            if w > h:
                new_w = target_size
                new_h = int(h * target_size / w)
            else:
                new_h = target_size
                new_w = int(w * target_size / h)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # 转换为numpy数组
        img_np = np.array(image).astype(np.float32)
        
        # 归一化到[-1, 1]
        img_np = (img_np / 127.5) - 1.0
        
        # 转换为tensor [1, 3, H, W]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device), image.size
    
    def _postprocess(self, tensor, original_size):
        """后处理输出"""
        # 转换为numpy [H, W, 3]
        output = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        
        # 反归一化到[0, 255]
        output = (output + 1.0) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # 转换为PIL图像
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
        """
        使用AnimeGAN处理图像
        
        Args:
            image: 输入图像
            strategy: 处理策略
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始AnimeGAN风格转换，策略: {strategy.value}")
            
            # 加载模型
            self.update_progress(10, 1, 10, 0)
            if not self._load_model():
                raise Exception("AnimeGAN模型加载失败")
            
            # 预处理
            self.update_progress(20, 2, 10, 0)
            logger.info("预处理图像...")
            input_tensor, original_size = self._preprocess(image)
            
            # AnimeGAN推理
            self.update_progress(40, 4, 10, 0)
            logger.info("AnimeGAN生成中...")
            
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            # 后处理
            self.update_progress(70, 7, 10, 0)
            logger.info("后处理...")
            result_image = self._postprocess(output_tensor, original_size)
            
            # 根据策略进行额外处理
            self.update_progress(85, 8, 10, 0)
            if strategy == ProcessingStrategy.QUALITY:
                result_image = self._enhance_quality(result_image)
            elif strategy == ProcessingStrategy.BALANCED:
                result_image = self._enhance_balanced(result_image)
            
            self.update_progress(100, 10, 10, 0)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ AnimeGAN风格转换完成，耗时: {processing_time:.2f}秒")
            
            return ProcessingResult(
                success=True,
                image=result_image,
                processing_time=processing_time,
                metadata={
                    'strategy': strategy.value,
                    'model': 'AnimeGANv2_Hayao',
                    'device': self.device,
                    'original_size': original_size
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ AnimeGAN处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _enhance_balanced(self, image: Image.Image) -> Image.Image:
        """平衡模式增强"""
        img_np = np.array(image)
        
        # 轻微锐化
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(img_np, -1, kernel)
        
        # 混合
        result = cv2.addWeighted(img_np, 0.7, sharpened, 0.3, 0)
        
        return Image.fromarray(result)
    
    def _enhance_quality(self, image: Image.Image) -> Image.Image:
        """高质量模式增强"""
        img_np = np.array(image)
        
        # 细节增强
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(img_np, -1, kernel)
        
        # 混合
        result = cv2.addWeighted(img_np, 0.6, enhanced, 0.4, 0)
        
        # 轻微去噪
        result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)
        
        return Image.fromarray(result)
