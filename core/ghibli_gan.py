#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
宫崎骏风格专用生成对抗网络 (GhibliGAN)
端到端的动漫风格转换系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vgg19
import torchvision.models as models
import numpy as np
from PIL import Image
import os
import json
import time

class ResidualBlock(nn.Module):
    """残差块 - 用于构建深度生成器网络"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu2(out)
        
        return out

class SelfAttention(nn.Module):
    """自注意力机制 - 捕获长距离依赖关系"""
    
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 计算查询、键、值
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        
        # 计算注意力权重
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # 应用注意力权重
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out

class GhibliGenerator(nn.Module):
    """宫崎骏风格生成器"""
    
    def __init__(self, input_channels=3, output_channels=3):
        super(GhibliGenerator, self).__init__()
        
        # 编码器 - 提取特征
        self.encoder = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(input_channels, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 下采样层1
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 下采样层2
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 下采样层3
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 残差块 - 深度特征提取
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(512, 512) for _ in range(9)]
        )
        
        # 自注意力机制
        self.attention = SelfAttention(512)
        
        # 解码器 - 重建图像
        self.decoder = nn.Sequential(
            # 上采样层1
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 上采样层2
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 上采样层3
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出层
            nn.Conv2d(64, output_channels, 7, 1, 3),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        # 宫崎骏风格特征增强
        self.style_enhancer = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 编码特征
        features = self.encoder(x)
        
        # 残差处理
        residual_out = self.residual_blocks(features)
        
        # 自注意力
        attention_out = self.attention(residual_out)
        
        # 风格增强
        style_weights = self.style_enhancer(attention_out)
        enhanced_features = attention_out * style_weights
        
        # 解码生成
        output = self.decoder(enhanced_features)
        
        return output

class GhibliDiscriminator(nn.Module):
    """宫崎骏风格判别器"""
    
    def __init__(self, input_channels=3):
        super(GhibliDiscriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, stride=2, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            discriminator_block(input_channels, 64, normalize=False),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512),
            discriminator_block(512, 1024),
            
            # 输出层
            nn.Conv2d(1024, 1, 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class PerceptualLoss(nn.Module):
    """感知损失 - 基于VGG19特征"""
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # 加载预训练VGG19
        vgg = vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        # 选择特征层
        self.features = nn.Sequential()
        for i, layer in enumerate(list(vgg)[:35]):  # 到conv4_3
            self.features.add_module(str(i), layer)
        
        # MSE损失
        self.mse_loss = nn.MSELoss()
    
    def forward(self, generated, target):
        # 特征提取
        gen_features = self.features(generated)
        target_features = self.features(target)
        
        # 计算感知损失
        return self.mse_loss(gen_features, target_features)

class StyleLoss(nn.Module):
    """风格损失 - Gram矩阵匹配"""
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        
        # 加载预训练VGG19
        vgg = vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        # 选择多个特征层
        self.style_layers = {
            '3': 'conv1_2',
            '8': 'conv2_2', 
            '15': 'conv3_3',
            '22': 'conv4_3'
        }
        
        self.features = {}
        
        def hook_fn(module, input, output, layer_name):
            self.features[layer_name] = output
        
        # 注册钩子
        for name, module in vgg._modules.items():
            if name in self.style_layers:
                module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
        
        self.vgg = vgg
        self.mse_loss = nn.MSELoss()
    
    def gram_matrix(self, tensor):
        """计算Gram矩阵"""
        batch_size, channels, height, width = tensor.size()
        
        # 重塑为 [batch_size, channels, height*width]
        tensor = tensor.view(batch_size, channels, height * width)
        
        # 计算Gram矩阵
        gram = torch.bmm(tensor, tensor.transpose(1, 2))
        
        # 归一化
        gram = gram / (channels * height * width)
        
        return gram
    
    def forward(self, generated, target):
        # 提取特征
        self.vgg(generated)
        gen_features = self.features.copy()
        
        self.features.clear()
        self.vgg(target)
        target_features = self.features.copy()
        
        # 计算风格损失
        style_loss = 0
        for layer_name in gen_features:
            gen_gram = self.gram_matrix(gen_features[layer_name])
            target_gram = self.gram_matrix(target_features[layer_name])
            style_loss += self.mse_loss(gen_gram, target_gram)
        
        return style_loss / len(gen_features)

class GhibliGAN:
    """宫崎骏风格生成对抗网络训练和推理"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # 初始化模型
        self.generator = GhibliGenerator().to(device)
        self.discriminator = GhibliDiscriminator().to(device)
        
        # 损失函数
        self.adversarial_loss = nn.BCELoss()
        self.perceptual_loss = PerceptualLoss().to(device)
        self.style_loss = StyleLoss().to(device)
        self.l1_loss = nn.L1Loss()
        
        # 优化器
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        
        # 训练历史
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'perceptual_loss': [],
            'style_loss': []
        }
    
    def train_step(self, real_photos, anime_styles):
        """单步训练"""
        batch_size = real_photos.size(0)
        
        # 真实和虚假标签
        real_label = torch.ones(batch_size, 1, 16, 16).to(self.device)
        fake_label = torch.zeros(batch_size, 1, 16, 16).to(self.device)
        
        # 训练判别器
        self.d_optimizer.zero_grad()
        
        # 真实照片的判别
        real_output = self.discriminator(real_photos)
        d_loss_real = self.adversarial_loss(real_output, real_label)
        
        # 生成动漫图像的判别
        fake_anime = self.generator(real_photos)
        fake_output = self.discriminator(fake_anime.detach())
        d_loss_fake = self.adversarial_loss(fake_output, fake_label)
        
        # 判别器总损失
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        self.d_optimizer.step()
        
        # 训练生成器
        self.g_optimizer.zero_grad()
        
        # 重新生成并判别
        fake_anime = self.generator(real_photos)
        fake_output = self.discriminator(fake_anime)
        
        # 生成器损失
        g_adv_loss = self.adversarial_loss(fake_output, real_label)
        g_perceptual_loss = self.perceptual_loss(fake_anime, anime_styles)
        g_style_loss = self.style_loss(fake_anime, anime_styles)
        g_l1_loss = self.l1_loss(fake_anime, anime_styles)
        
        # 总损失（加权组合）
        g_loss = (g_adv_loss * 1.0 + 
                 g_perceptual_loss * 10.0 + 
                 g_style_loss * 100.0 + 
                 g_l1_loss * 10.0)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        # 记录损失
        self.training_history['g_loss'].append(g_loss.item())
        self.training_history['d_loss'].append(d_loss.item())
        self.training_history['perceptual_loss'].append(g_perceptual_loss.item())
        self.training_history['style_loss'].append(g_style_loss.item())
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'perceptual_loss': g_perceptual_loss.item(),
            'style_loss': g_style_loss.item()
        }
    
    def save_model(self, path, epoch=None):
        """保存模型"""
        state = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'training_history': self.training_history
        }
        
        if epoch is not None:
            state['epoch'] = epoch
        
        torch.save(state, path)
        print(f"✅ 模型已保存到: {path}")
    
    def load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            print(f"❌ 模型文件不存在: {path}")
            return False
        
        state = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(state['generator'])
        self.discriminator.load_state_dict(state['discriminator'])
        
        if 'g_optimizer' in state:
            self.g_optimizer.load_state_dict(state['g_optimizer'])
        if 'd_optimizer' in state:
            self.d_optimizer.load_state_dict(state['d_optimizer'])
        if 'training_history' in state:
            self.training_history = state['training_history']
        
        print(f"✅ 模型已从 {path} 加载")
        return True
    
    def inference(self, image):
        """推理转换"""
        self.generator.eval()
        
        with torch.no_grad():
            # 预处理
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            if isinstance(image, Image.Image):
                input_tensor = transform(image).unsqueeze(0).to(self.device)
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
                input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
            else:
                raise ValueError("输入必须是PIL图像或numpy数组")
            
            # 生成
            output_tensor = self.generator(input_tensor)
            
            # 后处理
            output_tensor = (output_tensor + 1) / 2  # 反归一化到[0,1]
            output_tensor = torch.clamp(output_tensor, 0, 1)
            
            # 转换为PIL图像
            output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
            
            return output_image

# 全局实例
ghibli_gan = GhibliGAN()

def convert_with_ghibli_gan(image, model_path=None, progress_callback=None):
    """
    使用GhibliGAN进行风格转换
    
    Args:
        image: PIL图像或numpy数组
        model_path: 模型路径，如果为None则使用预训练模型
        progress_callback: 进度回调函数
    
    Returns:
        PIL图像
    """
    try:
        if progress_callback:
            progress_callback("加载模型...", 10)
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            ghibli_gan.load_model(model_path)
        elif model_path:
            print(f"⚠️ 模型文件不存在，使用随机初始化: {model_path}")
        
        if progress_callback:
            progress_callback("预处理图像...", 30)
        
        # 推理转换
        if progress_callback:
            progress_callback("生成宫崎骏风格...", 60)
        
        result = ghibli_gan.inference(image)
        
        if progress_callback:
            progress_callback("完成!", 100)
        
        return result
        
    except Exception as e:
        print(f"❌ GhibliGAN转换失败: {e}")
        return None

def create_sample_training_data():
    """创建示例训练数据配置"""
    config = {
        "dataset_config": {
            "photo_dir": "training_data/photos",
            "style_dir": "training_data/ghibli_styles",
            "batch_size": 4,
            "image_size": 512,
            "num_workers": 4
        },
        "training_config": {
            "epochs": 100,
            "save_interval": 10,
            "lr": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999
        },
        "loss_weights": {
            "adversarial": 1.0,
            "perceptual": 10.0,
            "style": 100.0,
            "l1": 10.0
        }
    }
    
    # 保存配置
    os.makedirs("models", exist_ok=True)
    with open("models/ghibli_gan_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ GhibliGAN训练配置已创建")
    return config

if __name__ == "__main__":
    # 创建配置
    create_sample_training_data()
    
    # 测试模型
    print("🎨 测试GhibliGAN...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    def test_progress(stage, progress):
        print(f"📊 {stage}: {progress}%")
    
    result = convert_with_ghibli_gan(test_image, progress_callback=test_progress)
    
    if result:
        print("✅ GhibliGAN测试成功")
        result.save("test_ghibli_gan_output.jpg")
        print("📁 输出保存到: test_ghibli_gan_output.jpg")
    else:
        print("❌ GhibliGAN测试失败")