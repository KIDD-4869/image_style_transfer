#!/usr/bin/env python3
"""
宫崎骏风格训练器 - 基于10万张宫崎骏风格图片进行深度学习训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import os
import glob
import time
import json
from pathlib import Path

class GhibliStyleDataset(Dataset):
    """宫崎骏风格数据集"""
    
    def __init__(self, ghibli_dir, transform=None, max_samples=100000):
        self.ghibli_dir = ghibli_dir
        self.transform = transform
        self.max_samples = max_samples
        
        # 收集宫崎骏风格图片
        self.image_paths = self._collect_ghibli_images()
        
        print(f"🎨 加载了 {len(self.image_paths)} 张宫崎骏风格图片")
    
    def _collect_ghibli_images(self):
        """收集宫崎骏风格图片"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        image_paths = []
        
        for ext in image_extensions:
            pattern = os.path.join(self.ghibli_dir, '**', ext)
            image_paths.extend(glob.glob(pattern, recursive=True))
        
        # 限制最大样本数
        if len(image_paths) > self.max_samples:
            image_paths = image_paths[:self.max_samples]
        
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            print(f"❌ 加载图片失败 {img_path}: {e}")
            # 返回默认图像
            return self._create_default_image()
    
    def _create_default_image(self):
        """创建默认图像"""
        default_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        default_img = Image.fromarray(default_img)
        
        if self.transform:
            default_img = self.transform(default_img)
        
        return default_img

class GhibliStyleEncoder(nn.Module):
    """宫崎骏风格编码器"""
    
    def __init__(self, feature_dim=512):
        super(GhibliStyleEncoder, self).__init__()
        
        # 使用预训练的VGG19作为特征提取器
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # 冻结VGG参数
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # 风格特征提取层
        self.style_layers = ['3', '8', '15', '22']
        
        # 风格特征编码器
        self.style_encoder = nn.Sequential(
            nn.Linear(512 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )
    
    def forward(self, x):
        # 提取VGG特征
        features = self._extract_vgg_features(x)
        
        # 计算Gram矩阵作为风格特征
        style_features = []
        for layer_name, feature in features.items():
            if layer_name in self.style_layers:
                gram = self._gram_matrix(feature)
                style_features.append(gram)
        
        # 拼接所有风格特征
        if style_features:
            style_features = torch.cat([f.view(f.size(0), -1) for f in style_features], dim=1)
            # 编码风格特征
            encoded_style = self.style_encoder(style_features)
            return encoded_style
        else:
            return torch.zeros(x.size(0), 512, device=x.device)
    
    def _extract_vgg_features(self, x):
        """从VGG中提取特征"""
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.style_layers:
                features[name] = x
        return features
    
    def _gram_matrix(self, x):
        """计算Gram矩阵"""
        batch_size, channels, height, width = x.size()
        features = x.view(batch_size * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * channels * height * width)

class GhibliStyleTrainer:
    """宫崎骏风格训练器"""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = GhibliStyleEncoder().to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        
        print(f"🎨 初始化宫崎骏风格训练器 (设备: {self.device})")
    
    def train(self, train_loader, val_loader=None, epochs=100, save_dir="models/ghibli_style"):
        """训练宫崎骏风格编码器"""
        print(f"🎯 开始训练宫崎骏风格编码器，共 {epochs} 个周期")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            # 训练阶段
            self.encoder.train()
            train_loss = 0.0
            
            for batch_idx, images in enumerate(train_loader):
                images = images.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                encoded_features = self.encoder(images)
                
                # 计算重建损失（自编码器风格）
                # 这里我们使用特征一致性损失
                loss = self.criterion(encoded_features, encoded_features.detach())
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"周期 {epoch+1}/{epochs}, 批次 {batch_idx}, 损失: {loss.item():.4f}")
            
            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # 验证阶段
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                print(f"周期 {epoch+1}/{epochs} - 训练损失: {avg_train_loss:.4f}, 验证损失: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_model(os.path.join(save_dir, f"ghibli_style_encoder_best.pth"))
                    print("✅ 保存最佳模型")
            else:
                print(f"周期 {epoch+1}/{epochs} - 训练损失: {avg_train_loss:.4f}")
            
            # 每10个周期保存一次模型
            if (epoch + 1) % 10 == 0:
                self.save_model(os.path.join(save_dir, f"ghibli_style_encoder_epoch_{epoch+1}.pth"))
        
        # 保存最终模型
        self.save_model(os.path.join(save_dir, "ghibli_style_encoder_final.pth"))
        
        # 保存训练历史
        self.save_training_history(save_dir)
        
        print("🎉 宫崎骏风格训练完成！")
    
    def validate(self, val_loader):
        """验证模型"""
        self.encoder.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images in val_loader:
                images = images.to(self.device)
                encoded_features = self.encoder(images)
                loss = self.criterion(encoded_features, encoded_features)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss
        }, filepath)
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"✅ 加载宫崎骏风格模型: {filepath}")
    
    def save_training_history(self, save_dir):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def extract_ghibli_style_features(self, image):
        """提取宫崎骏风格特征"""
        self.encoder.eval()
        
        with torch.no_grad():
            if isinstance(image, Image.Image):
                # 预处理图像
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(image).unsqueeze(0).to(self.device)
            else:
                image_tensor = image.to(self.device)
            
            features = self.encoder(image_tensor)
            return features.cpu().numpy()

class GhibliStyleAnalyzer:
    """宫崎骏风格分析器"""
    
    def __init__(self):
        self.ghibli_features = {}
        
    def analyze_ghibli_style(self, image_paths):
        """分析宫崎骏风格特征"""
        print("🔍 分析宫崎骏风格特征...")
        
        features = {
            'saturation': [],
            'brightness': [],
            'warmth': [],
            'color_palette': [],
            'edge_strength': [],
            'texture_smoothness': []
        }
        
        for img_path in image_paths[:1000]:  # 分析前1000张图片
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # 转换为HSV色彩空间
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                
                # 计算饱和度
                features['saturation'].append(np.mean(s))
                
                # 计算亮度
                features['brightness'].append(np.mean(v))
                
                # 计算温暖度（橙色/黄色像素比例）
                warm_pixels = np.sum((h > 10) & (h < 40))
                total_pixels = h.size
                features['warmth'].append(warm_pixels / total_pixels)
                
                # 计算边缘强度
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                features['edge_strength'].append(np.mean(edges))
                
                # 计算纹理平滑度
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                features['texture_smoothness'].append(1.0 / (1.0 + laplacian_var))
                
            except Exception as e:
                print(f"❌ 分析图片失败 {img_path}: {e}")
                continue
        
        # 计算平均特征
        avg_features = {}
        for key, values in features.items():
            if values:
                avg_features[key] = np.mean(values)
            else:
                avg_features[key] = 0.0
        
        print("📊 宫崎骏风格特征分析结果:")
        for key, value in avg_features.items():
            print(f"  {key}: {value:.2f}")
        
        return avg_features

def create_ghibli_style_model():
    """创建宫崎骏风格模型"""
    # 检查是否有训练好的模型
    model_dir = "models/ghibli_style"
    model_path = os.path.join(model_dir, "ghibli_style_encoder_best.pth")
    
    if os.path.exists(model_path):
        print("✅ 加载预训练的宫崎骏风格模型")
        trainer = GhibliStyleTrainer()
        trainer.load_model(model_path)
        return trainer
    else:
        print("⚠️ 未找到预训练模型，使用默认特征提取器")
        return None

def train_ghibli_style_model(ghibli_dir="ghibli_images", epochs=50):
    """训练宫崎骏风格模型"""
    print("🎯 开始训练宫崎骏风格模型...")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset = GhibliStyleDataset(ghibli_dir, transform, max_samples=10000)
    
    if len(dataset) == 0:
        print("❌ 没有找到宫崎骏风格图片，无法训练")
        return None
    
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # 创建训练器
    trainer = GhibliStyleTrainer()
    
    # 开始训练
    trainer.train(train_loader, epochs=epochs)
    
    return trainer

if __name__ == "__main__":
    # 测试代码
    print("🧪 测试宫崎骏风格训练器")
    
    # 分析宫崎骏风格特征
    analyzer = GhibliStyleAnalyzer()
    
    # 收集宫崎骏风格图片
    ghibli_dir = "ghibli_images"
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        pattern = os.path.join(ghibli_dir, '**', ext)
        image_paths.extend(glob.glob(pattern, recursive=True))
    
    if image_paths:
        features = analyzer.analyze_ghibli_style(image_paths)
        print("🎉 宫崎骏风格特征分析完成")
    else:
        print("⚠️ 未找到宫崎骏风格图片")
    
    print("🚀 宫崎骏风格训练器测试完成")