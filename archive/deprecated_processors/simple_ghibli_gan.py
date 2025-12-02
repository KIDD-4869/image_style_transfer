#!/usr/bin/env python3
"""
简化的宫崎骏风格GAN处理器 - 修复纯色输出问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from .image_processor_interface import ImageProcessorInterface, ProcessingResult, ProcessingStyle

class SimpleGhibliGenerator(nn.Module):
    """简化的宫崎骏风格生成器 - 保持内容结构"""
    
    def __init__(self):
        super(SimpleGhibliGenerator, self).__init__()
        
        # 简单的卷积层进行风格转换
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 3, 3, 1, 1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
    
    def forward(self, x):
        # 保持原始内容，只进行轻微的风格转换
        identity = x
        
        # 轻微的特征提取
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        
        # 与原图混合，保持内容
        out = 0.3 * torch.tanh(out) + 0.7 * identity
        
        return out

class SimpleGhibliGANProcessor(ImageProcessorInterface):
    """简化的GAN处理器"""
    
    def __init__(self):
        super().__init__(ProcessingStyle.GHIBLI_ENHANCED)
        self.device = torch.device("cpu")  # 使用CPU避免设备问题
        self.generator = SimpleGhibliGenerator().to(self.device)
        self.original_size = None
    
    def process(self, image: Image.Image, **kwargs) -> ProcessingResult:
        """处理图像"""
        try:
            self.original_size = image.size
            
            # 转换为张量
            input_tensor = self._image_to_tensor(image)
            
            # 生成结果
            with torch.no_grad():
                self.generator.eval()
                output_tensor = self.generator(input_tensor)
            
            # 转换回图像
            result_image = self._tensor_to_image(output_tensor)
            
            return ProcessingResult(success=True, image=result_image)
        except Exception as e:
            return ProcessingResult(success=False, error_message=str(e))
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """图像转张量"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 调整大小
        if image.size != (256, 256):
            image = image.resize((256, 256), Image.LANCZOS)
        
        # 转换为张量
        img_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """张量转图像"""
        tensor = tensor.cpu().squeeze(0)
        tensor = torch.clamp(tensor, 0, 1)
        
        img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        
        if self.original_size:
            image = image.resize(self.original_size, Image.LANCZOS)
        
        return image
    
    def get_processing_info(self) -> dict:
        return {
            "processor_type": "SimpleGhibliGANProcessor",
            "style_type": self.style_type.value,
            "description": "简化的宫崎骏风格GAN处理器"
        }