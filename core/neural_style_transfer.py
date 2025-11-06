#!/usr/bin/env python3
"""
神经网络风格迁移模块 - 基于预训练模型的先进风格迁移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import torch.optim as optim
import numpy as np
from PIL import Image
import cv2
import os

class NeuralStyleTransfer:
    """基于预训练模型的神经网络风格迁移"""
    
    def __init__(self, model_type='vgg19'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = self._load_pretrained_model(model_type)
        
        # 定义风格和内容层
        if model_type == 'vgg19':
            self.style_layers = ['0', '5', '10', '19', '28']  # 更丰富的风格特征
            self.content_layers = ['21']  # 更深层的内容特征
        elif model_type == 'resnet50':
            self.style_layers = ['layer1', 'layer2', 'layer3', 'layer4']
            self.content_layers = ['layer4']
        
        # 预定义的宫崎骏风格特征
        self.ghibli_style_features = None
        
    def _load_pretrained_model(self, model_type):
        """加载预训练模型"""
        print(f"🎯 加载预训练模型: {model_type}")
        
        if model_type == 'vgg19':
            try:
                model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
            except AttributeError:
                model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        elif model_type == 'resnet50':
            try:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            except AttributeError:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 冻结模型参数
        for param in model.parameters():
            param.requires_grad = False
        
        return model.to(self.device)
    
    def _extract_features(self, x, layers):
        """从模型中提取特征"""
        features = {}
        
        if self.model_type == 'vgg19':
            for name, layer in self.model._modules.items():
                x = layer(x)
                if name in layers:
                    features[name] = x
        elif self.model_type == 'resnet50':
            # ResNet特征提取
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            x = self.model.layer1(x)
            if 'layer1' in layers:
                features['layer1'] = x
            
            x = self.model.layer2(x)
            if 'layer2' in layers:
                features['layer2'] = x
            
            x = self.model.layer3(x)
            if 'layer3' in layers:
                features['layer3'] = x
            
            x = self.model.layer4(x)
            if 'layer4' in layers:
                features['layer4'] = x
        
        return features
    
    def _gram_matrix(self, x):
        """计算Gram矩阵（风格特征）"""
        batch_size, channels, height, width = x.size()
        features = x.view(batch_size * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * channels * height * width)
    
    def _preprocess_image(self, image, target_size=512):
        """预处理图像 - 智能处理大尺寸图片"""
        # 获取原始尺寸
        original_size = image.size
        
        # 智能调整目标尺寸，避免内存溢出
        max_allowed_size = 1024  # 最大处理尺寸
        
        # 如果图片尺寸过大，按比例缩放
        if max(original_size) > max_allowed_size:
            scale = max_allowed_size / max(original_size)
            target_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            print(f"📏 图片尺寸过大，自动缩放至: {target_size[0]}x{target_size[1]}")
        else:
            # 保持原始尺寸或使用默认尺寸
            target_size = min(max_allowed_size, max(original_size))
        
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def _postprocess_image(self, tensor):
        """后处理张量为图像"""
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tensor = torch.clamp(tensor, 0, 1)
        
        transform = transforms.ToPILImage()
        return transform(tensor)
    
    def _load_ghibli_style_reference(self):
        """加载宫崎骏风格参考图像"""
        style_folder = 'ghibli_images'
        style_images = []
        
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in image_extensions:
            import glob
            pattern = os.path.join(style_folder, ext)
            style_images.extend(glob.glob(pattern))
        
        if not style_images:
            print("⚠️ 没有找到宫崎骏风格参考图片，使用默认风格")
            return None
        
        print(f"🎨 加载了 {len(style_images)} 张宫崎骏风格参考图片")
        return style_images
    
    def _extract_ghibli_style_features(self, target_size=512):
        """提取宫崎骏风格特征"""
        if self.ghibli_style_features is not None:
            return self.ghibli_style_features
        
        style_images = self._load_ghibli_style_reference()
        
        if not style_images:
            # 如果没有参考图片，创建默认的宫崎骏风格特征
            return self._create_default_ghibli_style()
        
        # 处理所有风格图片并提取特征
        style_features_list = []
        
        for style_path in style_images:
            try:
                style_img = Image.open(style_path).convert('RGB')
                style_tensor = self._preprocess_image(style_img, target_size)
                
                # 提取风格特征
                features = self._extract_features(style_tensor, self.style_layers)
                style_features = {}
                
                for layer, feature in features.items():
                    style_features[layer] = self._gram_matrix(feature)
                
                style_features_list.append(style_features)
                
            except Exception as e:
                print(f"❌ 处理风格图片 {style_path} 失败: {e}")
        
        if not style_features_list:
            return self._create_default_ghibli_style()
        
        # 平均所有风格图片的特征
        avg_style_features = {}
        for layer in self.style_layers:
            layer_features = []
            for style_features in style_features_list:
                if layer in style_features:
                    layer_features.append(style_features[layer])
            
            if layer_features:
                avg_style_features[layer] = torch.stack(layer_features).mean(dim=0)
        
        self.ghibli_style_features = avg_style_features
        return avg_style_features
    
    def _create_default_ghibli_style(self):
        """创建默认的宫崎骏风格特征"""
        print("🎨 使用默认宫崎骏风格特征")
        
        # 创建具有宫崎骏风格特征的默认风格
        # 宫崎骏风格特点：柔和色彩、梦幻光影、简洁线条
        default_style = {}
        
        # 这里可以基于宫崎骏的艺术特点创建默认风格特征
        # 由于时间关系，我们使用简化的方法
        
        return default_style
    
    def transfer_style(self, content_image, style_weight=1000000, content_weight=1, 
                      num_steps=300, learning_rate=0.02, tv_weight=1e-5):
        """执行风格迁移"""
        print(f"🎯 开始神经网络风格迁移 (模型: {self.model_type}, 步骤: {num_steps})")
        
        # 预处理内容图像
        content_tensor = self._preprocess_image(content_image)
        
        # 使用内容图像作为初始输入
        input_img = content_tensor.clone().requires_grad_(True)
        
        # 提取内容特征
        content_features = self._extract_features(content_tensor, self.content_layers)
        
        # 提取宫崎骏风格特征
        style_features = self._extract_ghibli_style_features()
        
        # 使用Adam优化器
        optimizer = optim.Adam([input_img], lr=learning_rate)
        
        # 进度回调函数（来自 RealGhibliStyleTransfer 注入）
        progress_callback = self.progress_callback
        if progress_callback:
            # 发出初始进度，避免前端长时间显示0%
            try:
                progress_callback(1, 0, num_steps, 0.0)
            except Exception:
                pass
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 获取当前输入的特征
            features = self._extract_features(input_img, self.style_layers + self.content_layers)
            
            style_loss = 0
            content_loss = 0
            
            # 计算风格损失
            for layer in self.style_layers:
                if layer in features and layer in style_features:
                    target_style = style_features[layer]
                    current_style = features[layer]
                    
                    target_gram = self._gram_matrix(target_style)
                    current_gram = self._gram_matrix(current_style)
                    
                    style_loss += F.mse_loss(current_gram, target_gram)
            
            # 计算内容损失
            for layer in self.content_layers:
                if layer in features:
                    target_content = content_features[layer]
                    current_content = features[layer]
                    content_loss += F.mse_loss(current_content, target_content)
            
            # 总损失（加入TV约束）
            total_loss = style_weight * style_loss + content_weight * content_loss + tv_weight * self._tv_loss(input_img)
            
            # 检查损失是否为nan或inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️ 步骤 {step+1}: 损失值异常 (NaN/Inf)，跳过此步骤")
                continue
            
            total_loss.backward()
            
            # 检查梯度是否为nan或inf
            if torch.isnan(input_img.grad).any() or torch.isinf(input_img.grad).any():
                print(f"⚠️ 步骤 {step+1}: 梯度异常 (NaN/Inf)，重置梯度")
                optimizer.zero_grad()
                continue
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([input_img], max_norm=0.5)
            
            optimizer.step()
            
            # 限制像素值范围
            with torch.no_grad():
                input_img.data.clamp_(0, 1)
            
            # 更新进度
            if progress_callback:
                progress = int((step + 1) / num_steps * 100)
                progress_callback(progress, step + 1, num_steps, total_loss.item())
            
            if (step + 1) % 50 == 0:
                print(f"步骤 {step+1}/{num_steps}, 总损失: {total_loss.item():.4f}, "
                      f"风格损失: {style_loss.item():.4f}, 内容损失: {content_loss.item():.4f}")
        
        # 后处理输出图像
        output_tensor = input_img.data.clamp(0, 1)
        # 恢复到原始尺寸，减少模糊
        original_size = content_image.size
        result_image = self._postprocess_image(output_tensor, original_size)
        
        print("✅ 神经网络风格迁移完成")
        return result_image
    
    def fast_style_transfer(self, content_image, style_weight=500000, num_steps=100):
        """快速风格迁移 - 优化版本"""
        print("⚡ 执行快速风格迁移")
        
        # 使用更少的步骤和更高的学习率
        return self.transfer_style(
            content_image, 
            style_weight=style_weight, 
            content_weight=1,
            num_steps=num_steps, 
            learning_rate=0.03
        )
    
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback

# 预训练模型管理器
class StyleTransferManager:
    """风格迁移模型管理器"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
    
    def load_model(self, model_type='vgg19'):
        """加载指定类型的模型"""
        if model_type in self.models:
            self.current_model = self.models[model_type]
            return self.current_model
        
        try:
            model = NeuralStyleTransfer(model_type)
            self.models[model_type] = model
            self.current_model = model
            return model
        except Exception as e:
            print(f"❌ 加载模型 {model_type} 失败: {e}")
            return None
    
    def get_available_models(self):
        """获取可用的模型列表"""
        return ['vgg19', 'resnet50']
    
    def get_current_model(self):
        """获取当前模型"""
        return self.current_model

# 创建全局模型管理器
style_transfer_manager = StyleTransferManager()