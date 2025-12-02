#!/usr/bin/env python3
"""
修复版宫崎骏GAN - 解决纯色输出问题
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class ContentPreservingGenerator(nn.Module):
    """内容保持生成器 - 修复纯色问题"""
    
    def __init__(self):
        super().__init__()
        
        # 编码器 - 简化版本
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 减少残差块数量：9→4
        self.residual_blocks = nn.Sequential(
            *[self._make_residual_block(256) for _ in range(4)]
        )
        
        # 内容保持分支
        self.content_branch = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.Sigmoid()
        )
        
        # 风格转换分支
        self.style_branch = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.Tanh()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )
    
    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        
        # 残差处理
        residual_out = encoded
        for block in self.residual_blocks:
            residual_out = residual_out + block(residual_out)
        
        # 内容保持权重
        content_weights = self.content_branch(residual_out)
        preserved_content = encoded * content_weights
        
        # 风格转换
        style_features = self.style_branch(residual_out)
        
        # 内容和风格融合 - 关键修复
        # 大幅提高内容保持权重
        final_features = preserved_content * 0.8 + style_features * 0.2
        
        # 解码
        output = self.decoder(final_features)
        
        return output

class FixedGhibliGAN:
    """修复版宫崎骏GAN"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.generator = ContentPreservingGenerator().to(device)
        
        # 加载预训练权重（如果存在）
        self._load_pretrained()
    
    def _load_pretrained(self):
        """加载预训练权重"""
        model_paths = [
            'models/ghibli_gan/ghibli_gan_best.pth',
            'models/ghibli_gan/ghibli_gan_v3.pth',
            'models/anime_gan/AnimeGANv2_Hayao.pth'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=self.device)
                    
                    if 'generator' in checkpoint:
                        # 尝试加载生成器权重
                        try:
                            self.generator.load_state_dict(checkpoint['generator'], strict=False)
                            print(f"✅ 加载预训练模型: {path}")
                            return
                        except:
                            print(f"⚠️ 权重不匹配，跳过: {path}")
                    
                except Exception as e:
                    print(f"⚠️ 加载失败: {path} - {e}")
        
        print("⚠️ 未找到兼容的预训练模型，使用随机初始化")
    
    def process_image(self, image_path: str) -> Image.Image:
        """处理图像"""
        self.generator.eval()
        
        with torch.no_grad():
            # 预处理
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # 降低分辨率提高稳定性
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # 生成
            output_tensor = self.generator(input_tensor)
            
            # 后处理
            output_tensor = (output_tensor + 1) / 2  # 反归一化
            output_tensor = torch.clamp(output_tensor, 0, 1)
            
            # 转换为PIL图像
            output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
            
            # 恢复原始尺寸
            output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
            
            return output_image

# 全局实例
fixed_ghibli_gan = FixedGhibliGAN()

def process_with_fixed_gan(image_path: str) -> Image.Image:
    """使用修复版GAN处理图像"""
    return fixed_ghibli_gan.process_image(image_path)