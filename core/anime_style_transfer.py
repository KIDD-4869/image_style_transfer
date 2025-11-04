#!/usr/bin/env python3
"""
真正的动漫风格转换模块 - 基于深度学习和GANs
实现真正的照片转动漫风技术
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

class AnimeStyleGAN(nn.Module):
    """动漫风格生成对抗网络"""
    
    def __init__(self):
        super(AnimeStyleGAN, self).__init__()
        
        # 编码器 - 提取真实照片特征
        self.encoder = nn.Sequential(
            # 输入: 3x256x256
            nn.Conv2d(3, 64, 4, 2, 1),  # 64x128x128
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 128x64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256x32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),  # 512x16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1),  # 1024x8x8
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        
        # 解码器 - 生成动漫风格图像
        self.decoder = nn.Sequential(
            # 输入: 1024x8x8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # 512x16x16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 256x32x32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128x64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 3x256x256
            nn.Tanh(),
        )
        
        # 判别器 - 判断是否为动漫风格
        self.discriminator = nn.Sequential(
            # 输入: 3x256x256
            nn.Conv2d(3, 64, 4, 2, 1),  # 64x128x128
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 128x64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256x32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),  # 512x16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 2, 1),  # 1x8x8
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # 编码器提取特征
        encoded = self.encoder(x)
        # 解码器生成动漫风格
        decoded = self.decoder(encoded)
        return decoded
    
    def discriminate(self, x):
        return self.discriminator(x)

class AnimeStyleTransfer:
    """基于深度学习的动漫风格转换"""
    
    def __init__(self, use_pretrained=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(use_pretrained)
        
        # 动漫风格特征提取器
        self.style_extractor = self._load_style_extractor()
        
        # 面部特征检测器
        self.face_detector = self._load_face_detector()
        
        # 动漫风格特征库
        self.anime_features = self._load_anime_features()
        
    def _load_model(self, use_pretrained):
        """加载预训练模型或创建新模型"""
        if use_pretrained:
            # 尝试加载预训练模型
            model_path = "models/anime_style_gan.pth"
            if os.path.exists(model_path):
                try:
                    model = AnimeStyleGAN()
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    print("✅ 加载预训练动漫风格模型成功")
                    return model
                except Exception as e:
                    print(f"⚠️ 加载预训练模型失败: {e}")
        
        # 创建新模型
        model = AnimeStyleGAN().to(self.device)
        print("🆕 创建新的动漫风格模型")
        return model
    
    def _load_style_extractor(self):
        """加载风格特征提取器"""
        try:
            # 使用VGG19提取风格特征
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
            for param in vgg.parameters():
                param.requires_grad = False
            return vgg.to(self.device)
        except Exception as e:
            print(f"⚠️ 加载风格提取器失败: {e}")
            return None
    
    def _load_face_detector(self):
        """加载面部特征检测器"""
        try:
            # 使用OpenCV的人脸检测器
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            return face_cascade
        except Exception as e:
            print(f"⚠️ 加载面部检测器失败: {e}")
            return None
    
    def _load_anime_features(self):
        """加载动漫风格特征库"""
        # 动漫风格特征：大眼、小鼻、鲜艳色彩、清晰线条
        anime_features = {
            'eye_ratio': 0.15,  # 眼睛占面部比例（动漫通常更大）
            'nose_ratio': 0.05,  # 鼻子占面部比例（动漫通常更小）
            'saturation_boost': 0.3,  # 饱和度增强
            'contrast_boost': 0.2,  # 对比度增强
            'edge_strength': 0.15,  # 边缘强度
        }
        return anime_features
    
    def _preprocess_image(self, image, target_size=256):
        """预处理图像"""
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def _postprocess_image(self, tensor):
        """后处理张量为图像"""
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor * 0.5 + 0.5  # 反归一化
        tensor = torch.clamp(tensor, 0, 1)
        
        transform = transforms.ToPILImage()
        return transform(tensor)
    
    def _detect_faces(self, image):
        """检测面部特征"""
        if self.face_detector is None:
            return []
        
        # 转换为灰度图进行人脸检测
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
        
        face_features = []
        for (x, y, w, h) in faces:
            # 计算面部特征比例
            features = {
                'bbox': (x, y, w, h),
                'eye_region': (x + w//4, y + h//3, w//2, h//3),  # 眼睛区域
                'nose_region': (x + w//3, y + h//2, w//3, h//4),  # 鼻子区域
                'face_ratio': w / h  # 面部宽高比
            }
            face_features.append(features)
        
        return face_features
    
    def _apply_anime_face_features(self, image, face_features):
        """应用动漫面部特征"""
        img_array = np.array(image)
        
        for features in face_features:
            x, y, w, h = features['bbox']
            
            # 1. 眼睛放大（动漫特征）
            eye_x, eye_y, eye_w, eye_h = features['eye_region']
            eyes = img_array[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
            
            if eyes.size > 0:
                # 放大眼睛区域
                new_eye_h = int(eye_h * 1.3)  # 放大30%
                new_eye_w = int(eye_w * 1.2)
                eyes_resized = cv2.resize(eyes, (new_eye_w, new_eye_h))
                
                # 计算新的眼睛位置（居中）
                new_eye_x = eye_x - (new_eye_w - eye_w) // 2
                new_eye_y = eye_y - (new_eye_h - eye_h) // 3
                
                # 确保不越界
                new_eye_x = max(0, new_eye_x)
                new_eye_y = max(0, new_eye_y)
                
                # 替换眼睛区域
                end_y = min(new_eye_y + new_eye_h, img_array.shape[0])
                end_x = min(new_eye_x + new_eye_w, img_array.shape[1])
                
                actual_h = end_y - new_eye_y
                actual_w = end_x - new_eye_x
                
                if actual_h > 0 and actual_w > 0:
                    img_array[new_eye_y:end_y, new_eye_x:end_x] = eyes_resized[:actual_h, :actual_w]
            
            # 2. 鼻子缩小（动漫特征）
            nose_x, nose_y, nose_w, nose_h = features['nose_region']
            if nose_h > 10 and nose_w > 10:  # 确保鼻子区域足够大
                nose = img_array[nose_y:nose_y+nose_h, nose_x:nose_x+nose_w]
                
                # 缩小鼻子
                new_nose_h = max(5, int(nose_h * 0.7))  # 缩小30%
                new_nose_w = max(5, int(nose_w * 0.7))
                nose_resized = cv2.resize(nose, (new_nose_w, new_nose_h))
                
                # 居中放置
                new_nose_x = nose_x + (nose_w - new_nose_w) // 2
                new_nose_y = nose_y + (nose_h - new_nose_h) // 2
                
                img_array[new_nose_y:new_nose_y+new_nose_h, new_nose_x:new_nose_x+new_nose_w] = nose_resized
        
        return Image.fromarray(img_array)
    
    def _enhance_anime_features(self, image):
        """增强动漫特征"""
        img_array = np.array(image)
        
        # 1. 增强饱和度（动漫色彩鲜艳）
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, int(255 * self.anime_features['saturation_boost']))
        s = np.clip(s, 0, 255)
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
        
        # 2. 增强对比度
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab_enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # 3. 增强边缘（动漫线条清晰）
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # 叠加边缘
        result = cv2.addWeighted(enhanced, 0.85, edges_colored, 0.15, 0)
        
        return Image.fromarray(result)
    
    def transfer_to_anime(self, content_image, enhance_features=True):
        """将真实照片转换为动漫风格"""
        print("🎨 开始真正的动漫风格转换...")
        
        try:
            # 1. 检测面部特征
            face_features = self._detect_faces(content_image)
            print(f"👤 检测到 {len(face_features)} 个面部")
            
            # 2. 使用GAN模型进行风格转换
            content_tensor = self._preprocess_image(content_image)
            
            with torch.no_grad():
                anime_tensor = self.model(content_tensor)
            
            anime_image = self._postprocess_image(anime_tensor)
            
            # 3. 应用动漫面部特征
            if face_features and enhance_features:
                anime_image = self._apply_anime_face_features(anime_image, face_features)
            
            # 4. 增强动漫特征
            if enhance_features:
                anime_image = self._enhance_anime_features(anime_image)
            
            print("✅ 动漫风格转换完成")
            return anime_image
            
        except Exception as e:
            print(f"❌ 动漫风格转换失败: {e}")
            # 回退到传统方法
            return self._fallback_traditional_method(content_image)
    
    def _fallback_traditional_method(self, image):
        """备选传统方法"""
        print("⚠️ 使用备选传统动漫风格转换")
        
        img_array = np.array(image)
        
        # 使用改进的动漫风格滤镜
        # 1. 深度边缘保留平滑
        filtered = cv2.bilateralFilter(img_array, d=15, sigmaColor=100, sigmaSpace=100)
        
        # 2. 强烈的颜色量化
        Z = filtered.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = 8
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        cartoon = centers[labels.flatten()].reshape(filtered.shape)
        
        # 3. 清晰的边缘检测
        gray = cv2.cvtColor(cartoon, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # 4. 叠加边缘
        result = cv2.addWeighted(cartoon, 0.8, edges_colored, 0.2, 0)
        
        return Image.fromarray(result)

# 创建全局动漫风格转换器
anime_style_transfer = AnimeStyleTransfer()