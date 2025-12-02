#!/usr/bin/env python3
"""
宫崎骏风格GAN生成器 - 实现从真实照片到动漫风格的端到端生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2
from .image_processor_interface import ImageProcessorInterface, ProcessingResult, ProcessingStyle
import os

class GhibliGANGenerator(nn.Module):
    """宫崎骏风格GAN生成器网络"""
    
    def __init__(self):
        super(GhibliGANGenerator, self).__init__()
        
        # 编码器部分 - 提取真实照片特征 (与预训练模型完全匹配)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),      # 64 x 256 x 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, 2, 1),    # 128 x 128 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, 2, 1),   # 256 x 64 x 64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 3, 2, 1),   # 512 x 32 x 32
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 风格全连接层 (与预训练模型完全匹配)
        self.style_fc = nn.Linear(512, 256)
        
        # 添加一个图像生成模块，从风格特征生成图像
        # 使用转置卷积来重建图像
        self.image_generator = nn.Sequential(
            # 将256维特征扩展为初步特征图
            nn.Linear(256, 512 * 4 * 4),  # 生成4x4的特征图
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 4, 4)),
            
            # 逐步上采样到256x256
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 3, 4, 2, 1),     # 256x256
            nn.Tanh()
        )
    
    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        
        # 全局平均池化
        encoded = F.adaptive_avg_pool2d(encoded, (1, 1))
        encoded = encoded.view(encoded.size(0), -1)
        
        # 风格特征处理
        style_features = self.style_fc(encoded)
        
        # 从风格特征生成图像
        generated_image = self.image_generator(style_features)
        
        return generated_image


class GhibliGANDiscriminator(nn.Module):
    """宫崎骏风格GAN判别器网络"""
    
    def __init__(self):
        super(GhibliGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 输入: 3 x 256 x 256
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出层
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class PerceptualLoss(nn.Module):
    """感知损失 - 基于VGG特征"""
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # 使用预训练VGG网络
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg.eval()
        
        # 选择特定层进行特征提取
        self.layer_names = {
            '3': 'relu1_2',
            '8': 'relu2_2', 
            '17': 'relu3_3',
            '26': 'relu4_3'
        }
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, generated, target):
        # 提取特征
        gen_features = self._extract_features(generated)
        target_features = self._extract_features(target)
        
        # 计算感知损失
        loss = 0
        for gen_feat, target_feat in zip(gen_features, target_features):
            loss += self.mse_loss(gen_feat, target_feat)
        
        return loss
    
    def _extract_features(self, x):
        """提取VGG特征"""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layer_names:
                features.append(x)
        return features


class GhibliGANProcessor(ImageProcessorInterface):
    """基于GAN的宫崎骏风格处理器"""
    
    def __init__(self):
        super().__init__(ProcessingStyle.GHIBLI_ENHANCED)
        self.progress_callback = None
        self.task_id = None
        
        # 初始化设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # 初始化模型
        self.generator = GhibliGANGenerator().to(self.device)
        self.discriminator = GhibliGANDiscriminator().to(self.device)
        
        # 损失函数
        self.adversarial_loss = nn.BCELoss()
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.l1_loss = nn.L1Loss()
        
        # 存储原始图像尺寸
        self.original_size = None
        
        # 加载预训练模型（如果存在）
        self._load_pretrained_model()
    
    def process(self, image: Image.Image, **kwargs) -> ProcessingResult:
        """使用GAN处理图像为宫崎骏风格"""
        try:
            # 保存原始尺寸
            self.original_size = image.size
            
            # 将图像转换为张量并移动到设备
            input_tensor = self._image_to_tensor(image).to(self.device)
            
            # 使用生成器进行推理
            with torch.no_grad():
                self.generator.eval()
                output_tensor = self.generator(input_tensor)
            
            # 将输出张量转换回图像
            result_image = self._tensor_to_image(output_tensor)
            
            return ProcessingResult(success=True, image=result_image)
        except Exception as e:
            return ProcessingResult(success=False, error_message=str(e))
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """将PIL图像转换为张量"""
        # 确保图片是RGB模式
        if image.mode in ('RGBA', 'LA', 'P'):
            # 创建白色背景
            background = Image.new('RGB', image.size, (255, 255, 255))
            # 处理调色板模式
            if image.mode == 'P':
                image = image.convert('RGBA')
            # 粘贴图片到白色背景上
            if image.mode == 'RGBA' or image.mode == 'LA':
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        
        # 调整图像大小
        if image.size != (256, 256):
            image = image.resize((256, 256), Image.LANCZOS)
        
        # 转换为numpy数组并归一化
        img_array = np.array(image).astype(np.float32)
        
        # 归一化到[-1, 1]
        img_array = (img_array / 127.5) - 1.0
        
        # 转换为张量
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return tensor
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """将张量转换为PIL图像"""
        # 移动到CPU并移除批次维度
        tensor = tensor.cpu().squeeze(0)
        
        # 反归一化到[0, 255]
        tensor = (tensor + 1.0) * 127.5
        tensor = torch.clamp(tensor, 0, 255)
        
        # 转换为numpy数组
        img_array = tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        
        # 转换为PIL图像
        image = Image.fromarray(img_array)
        
        # 如果有原始尺寸信息，则调整回原始尺寸
        if self.original_size is not None:
            image = image.resize(self.original_size, Image.LANCZOS)
        
        return image
    
    def _load_pretrained_model(self):
        """加载预训练模型"""
        try:
            # 检查不同的模型路径
            model_paths = [
                "models/ghibli_gan/ghibli_gan_best.pth",
                "models/ghibli_gan/ghibli_gan_v3.pth",
                "models/ghibli_gan/ghibli_gan_v2.pth",
                "models/ghibli_gan/ghibli_gan_v1.pth"
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        checkpoint = torch.load(model_path, map_location=self.device)
                        # 处理不同的模型保存格式
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            self.generator.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        elif isinstance(checkpoint, dict) and 'generator' in checkpoint:
                            self.generator.load_state_dict(checkpoint['generator'], strict=False)
                        else:
                            self.generator.load_state_dict(checkpoint, strict=False)
                        print(f"✅ 成功加载预训练的宫崎骏GAN模型: {model_path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"⚠️ 尝试加载模型 {model_path} 失败: {e}")
                        continue
            
            if not model_loaded:
                print("⚠️ 未找到预训练模型，将使用随机初始化的模型")
        except Exception as e:
            print(f"❌ 加载预训练模型失败: {e}")
    
    def set_progress_callback(self, callback, task_id):
        """设置进度回调"""
        self.progress_callback = callback
        self.task_id = task_id
    
    def get_processing_info(self) -> dict:
        """获取处理器信息"""
        return {
            "processor_type": "GhibliGANProcessor",
            "style_type": self.style_type.value,
            "device": str(self.device),
            "description": "基于GAN的宫崎骏风格处理器 - 端到端生成"
        }


# 创建全局实例
ghibli_gan_processor = GhibliGANProcessor()