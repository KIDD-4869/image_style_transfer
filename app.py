import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import io
import base64
from flask import Flask, render_template, request, jsonify
import os
import requests
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class GhibliStyleTransfer(nn.Module):
    """宫崎骏动漫风格转换模型 - 基于优化的传统方法"""
    def __init__(self):
        super(GhibliStyleTransfer, self).__init__()
        # 使用优化的传统图像处理方法
        self.use_deep_learning = False  # 暂时禁用深度学习
        
        if self.use_deep_learning:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._init_simple_model()
        else:
            print("使用优化的传统图像处理方法")
    
    def _build_simple_generator(self):
        """构建简化的生成器网络"""
        class SimpleGenerator(nn.Module):
            def __init__(self):
                super(SimpleGenerator, self).__init__()
                # 简化的编码器-解码器结构
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 9, 1, 4),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, 2, 1),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, 2, 1),
                    nn.InstanceNorm2d(128),
                    nn.ReLU(),
                )
                
                # 残差块
                self.res_blocks = nn.Sequential(*[
                    nn.Sequential(
                        nn.Conv2d(128, 128, 3, 1, 1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, 3, 1, 1),
                        nn.InstanceNorm2d(128),
                    ) for _ in range(4)
                ])
                
                # 解码器
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 3, 9, 1, 4),
                    nn.Tanh()
                )
            
            def forward(self, x):
                # 编码
                encoded = self.encoder(x)
                
                # 残差连接
                residual = self.res_blocks(encoded)
                
                # 解码
                decoded = self.decoder(encoded + residual)
                
                return decoded
        
        return SimpleGenerator().to(self.device)
    
    def _try_load_pretrained(self):
        """尝试加载预训练权重"""
        # 如果没有预训练权重，使用随机初始化
        print("使用随机初始化的模型（无预训练权重）")
    
    def _load_vgg(self):
        """加载预训练的VGG网络"""
        try:
            vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
            vgg = vgg.features[:16].to(self.device)
            for param in vgg.parameters():
                param.requires_grad = False
            return vgg
        except:
            # 如果加载失败，返回None
            return None
    
    def forward(self, image, style_type="ghibli"):
        """
        应用宫崎骏动漫风格转换（基于优化的传统方法）
        
        Args:
            image: PIL图像
            style_type: 风格类型，可选 "ghibli"（宫崎骏）、"anime"（通用动漫）
        """
        if self.use_deep_learning:
            # 深度学习路径
            input_tensor = self._preprocess_image(image)
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            result_image = self._postprocess_image(output_tensor)
            if style_type == "ghibli":
                result_image = self._apply_ghibli_optimization(result_image)
            return result_image
        else:
            # 优化的传统方法路径
            return self._apply_optimized_traditional_method(image, style_type)
    
    def _preprocess_image(self, image):
        """预处理图像：调整到固定尺寸、归一化"""
        # 调整到固定尺寸（512x512），确保尺寸对齐
        target_size = 512
        
        # 保持宽高比调整大小，然后填充到正方形
        w, h = image.size
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 调整大小
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 创建正方形画布并居中放置图像
        square_image = Image.new('RGB', (target_size, target_size), (128, 128, 128))
        left = (target_size - new_w) // 2
        top = (target_size - new_h) // 2
        square_image.paste(resized_image, (left, top))
        
        # 转换为tensor并归一化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        tensor = transform(square_image).unsqueeze(0).to(self.device)
        return tensor
    
    def _postprocess_image(self, tensor):
        """后处理tensor为PIL图像"""
        # 反归一化
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为PIL图像
        transform = transforms.ToPILImage()
        image = transform(tensor)
        
        return image
    
    def _apply_ghibli_optimization(self, image):
        """应用宫崎骏风格优化：色彩、线条、光影"""
        # 转换为numpy数组进行处理
        img_np = np.array(image)
        
        # 1. 色彩优化 - 增强宫崎骏风格的柔和色彩
        img_np = self._optimize_ghibli_colors(img_np)
        
        # 2. 线条优化 - 增强动漫风格的清晰线条
        img_np = self._optimize_anime_lines(img_np)
        
        # 3. 光影优化 - 添加宫崎骏风格的梦幻光影
        img_np = self._optimize_ghibli_lighting(img_np)
        
        return Image.fromarray(img_np)
    
    def _optimize_ghibli_colors(self, img_np):
        """优化宫崎骏风格色彩"""
        # 转换为LAB色彩空间进行更精确的色彩调整
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强色彩饱和度（宫崎骏风格色彩鲜艳但不刺眼）
        a = cv2.addWeighted(a, 1.1, np.zeros_like(a), 0, 0)
        b = cv2.addWeighted(b, 1.1, np.zeros_like(b), 0, 0)
        
        # 合并LAB通道
        lab = cv2.merge([l, a, b])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return img_np
    
    def _optimize_anime_lines(self, img_np):
        """优化动漫风格线条"""
        # 使用边缘检测增强线条
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 多尺度边缘检测
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_medium = cv2.Canny(gray, 30, 100)
        
        # 组合边缘
        edges = cv2.bitwise_or(edges_fine, edges_medium)
        
        # 对边缘进行艺术化处理
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 将边缘融合到原图（轻微加深边缘区域）
        edge_mask = edges > 0
        for c in range(3):
            channel = img_np[:, :, c]
            channel[edge_mask] = np.clip(channel[edge_mask] * 0.8, 0, 255)
            img_np[:, :, c] = channel
        
        return img_np
    
    def _optimize_ghibli_lighting(self, img_np):
        """优化宫崎骏风格光影"""
        # 添加柔和的全局光照效果
        h, w = img_np.shape[:2]
        
        # 创建中心渐变光照
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        center_y, center_x = h // 2, w // 2
        
        # 计算距离中心的距离
        dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # 创建光照图（中心亮，边缘暗）
        light_map = 1.0 - (dist / max_dist) * 0.15
        
        # 应用光照效果
        for c in range(3):
            img_np[:, :, c] = np.clip(img_np[:, :, c] * light_map, 0, 255)
        
        return img_np.astype(np.uint8)
    
    def _extract_features(self, x, layers):
        """从VGG网络中提取特征"""
        features = []
        for layer in self.vgg[:layers]:
            x = layer(x)
            features.append(x)
        return features
    
    def compute_gram_matrix(self, x):
        """计算Gram矩阵用于风格损失"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def train_step(self, real_images, style_images):
        """训练步骤"""
        # 生成器训练
        self.generator.train()
        self.discriminator.train()
        
        # 生成假图像
        fake_images = self.generator(real_images)
        
        # 计算各种损失
        content_loss = self._compute_content_loss(real_images, fake_images)
        style_loss = self._compute_style_loss(style_images, fake_images)
        adversarial_loss = self._compute_adversarial_loss(fake_images)
        
        # 总损失
        total_loss = content_loss + style_loss + adversarial_loss
        
        return total_loss
    
    def _compute_content_loss(self, real, fake):
        """计算内容损失"""
        real_features = self._extract_features(real, 5)  # 浅层特征
        fake_features = self._extract_features(fake, 5)
        return self.content_criterion(fake_features[-1], real_features[-1])
    
    def _compute_style_loss(self, style, fake):
        """计算风格损失"""
        style_features = self._extract_features(style, 16)  # 深层特征
        fake_features = self._extract_features(fake, 16)
        
        style_loss = 0
        for s_feat, f_feat in zip(style_features, fake_features):
            style_gram = self.compute_gram_matrix(s_feat)
            fake_gram = self.compute_gram_matrix(f_feat)
            style_loss += self.style_criterion(fake_gram, style_gram)
        
        return style_loss
    
    def _compute_adversarial_loss(self, fake_images):
        """计算对抗损失"""
        fake_logits = self.discriminator(fake_images)
        real_labels = torch.ones_like(fake_logits)
        return self.adversarial_criterion(fake_logits, real_labels)
    
    def _apply_ghibli_style_preserve_details(self, img):
        """应用宫崎骏风格（重写版）- 基于区域色彩分布和高画质动漫化"""
        original_img = img.copy()
        
        # 1. 保持高画质预处理：不缩小尺寸，保持原始细节
        preprocessed = self._high_quality_preprocess(img)
        
        # 2. 区域色彩分析：分析不同区域的色彩分布
        regional_analysis = self._analyze_regional_colors(preprocessed)
        
        # 3. 基于区域特征的动漫化：不同区域采用不同处理
        regional_anime = self._regional_anime_conversion(preprocessed, regional_analysis)
        
        # 4. 宫崎骏风格色彩映射：基于真实宫崎骏图片的区域色彩
        ghibli_colors = self._regional_ghibli_color_mapping(regional_anime, regional_analysis)
        
        # 5. 高画质细节保留：在动漫化基础上恢复重要细节
        detailed_result = self._high_quality_detail_preservation(ghibli_colors, original_img)
        
        # 6. 宫崎骏光影氛围：添加梦幻光影效果
        final = self._ghibli_dreamy_lighting(detailed_result)
        
        return final
    
    def _apply_anime_style_preserve_details(self, img):
        """应用通用动漫风格（细节保留版）"""
        original_img = img.copy()
        
        # 1. 基础动漫化
        cartoon_base = self._create_cartoon_base(img)
        
        # 2. 细节提取和保留
        details = self._extract_important_details(original_img)
        
        # 3. 色彩增强
        enhanced_colors = self._enhance_anime_colors(cartoon_base)
        
        # 4. 细节融合
        result = self._smart_detail_fusion(enhanced_colors, details)
        
        return result
    
    def _apply_painting_style_preserve_details(self, img):
        """应用绘画风格（细节保留版）"""
        original_img = img.copy()
        
        # 1. 绘画效果基础
        painting_base = self._create_painting_base(img)
        
        # 2. 关键细节保留
        key_details = self._extract_key_details(original_img)
        
        # 3. 艺术色彩处理
        artistic_colors = self._apply_artistic_colors(painting_base)
        
        # 4. 智能融合
        result = self._artistic_detail_fusion(artistic_colors, key_details)
        
        return result
    
    def _apply_optimized_traditional_method(self, image, style_type):
        """应用优化的传统图像处理方法"""
        # 将PIL图像转换为numpy数组
        img_np = np.array(image)
        
        # 转换为BGR格式（OpenCV使用）
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            # 如果是灰度图，转换为彩色
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # 根据风格类型选择处理方式
        if style_type == "ghibli":
            result = self._apply_ghibli_style_preserve_details(img_bgr)
        elif style_type == "anime":
            result = self._apply_anime_style_preserve_details(img_bgr)
        elif style_type == "painting":
            result = self._apply_painting_style_preserve_details(img_bgr)
        else:
            # 默认使用宫崎骏风格
            result = self._apply_ghibli_style_preserve_details(img_bgr)
        
        # 将结果转换回RGB格式
        if len(result.shape) == 3:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        else:
            result_rgb = result
        
        return result_rgb
    
    def _high_quality_preprocess(self, img):
        """高画质预处理：保持原始分辨率，增强细节可见度"""
        h, w = img.shape[:2]
        
        # 保持原始分辨率，不进行缩小（宫崎骏风格需要高画质）
        # 仅当图片过大时进行适度缩小（>2000像素）
        max_size = 2000
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 轻微锐化，保持细节清晰度
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # 与原图混合，保持自然感
        final = cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)
        
        return final
    
    def _analyze_regional_colors(self, img):
        """分析区域色彩分布：采用更精细的5x5=25个区域，避免相邻颜色互相影响"""
        h, w = img.shape[:2]
        
        # 将图片分为5x5=25个更小的区域，避免大区域的平均色值失真
        regions = []
        region_features = []
        
        for i in range(5):
            for j in range(5):
                y1, y2 = i*h//5, (i+1)*h//5
                x1, x2 = j*w//5, (j+1)*w//5
                region = img[y1:y2, x1:x2]
                regions.append(region)
                
                # 分析区域色彩特征
                if region.size > 0:
                    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                    h_mean, s_mean, v_mean = np.mean(hsv, axis=(0, 1))
                    h_std, s_std, v_std = np.std(hsv, axis=(0, 1))
                    
                    # 判断区域类型（使用更精细的分类）
                    region_type = self._classify_region_type_fine(h_mean, s_mean, v_mean, h_std, s_std, v_std)
                    
                    region_features.append({
                        'coords': (y1, y2, x1, x2),
                        'h_mean': h_mean, 's_mean': s_mean, 'v_mean': v_mean,
                        'h_std': h_std, 's_std': s_std, 'v_std': v_std,
                        'type': region_type,
                        'original_region': region.copy()  # 保存原始区域用于参考
                    })
        
        return region_features
    
    def _classify_region_type_fine(self, h_mean, s_mean, v_mean, h_std, s_std, v_std):
        """更精细的区域分类：基于像素级特征，避免平均色值失真"""
        # 更精细的分类逻辑，考虑更多特征
        if v_mean > 200:  # 极高亮度
            if s_mean < 50:  # 极低饱和度
                return "bright_white"  # 亮白色区域（如高光）
            elif s_mean < 100:
                return "bright_pastel"  # 亮色柔和区域
            else:
                return "bright_colorful"  # 亮色鲜艳区域
        elif v_mean > 160:  # 高亮度
            if s_mean < 80:
                return "light_neutral"  # 浅色中性区域
            else:
                return "light_colorful"  # 浅色鲜艳区域
        elif v_mean > 100:  # 中等亮度
            if h_std > 40:  # 色彩变化很大
                return "complex_detailed"  # 复杂细节区域
            elif s_mean > 120:
                return "medium_colorful"  # 中等鲜艳区域
            else:
                return "medium_neutral"  # 中等中性区域
        elif v_mean > 60:  # 低亮度
            if s_mean > 100:
                return "dark_colorful"  # 暗色鲜艳区域
            else:
                return "dark_neutral"  # 暗色中性区域
        else:  # 极低亮度
            return "very_dark"  # 极暗区域
    
    def _classify_region_type(self, h_mean, s_mean, v_mean, h_std, s_std, v_std):
        """根据色彩特征分类区域类型"""
        # 基于宫崎骏图片分析的区域分类
        if v_mean > 180:  # 高亮度区域
            if s_mean < 100:  # 低饱和度
                return "sky"  # 天空
            else:
                return "highlight"  # 高光
        elif v_mean > 120:  # 中等亮度
            if h_std > 30:  # 色彩变化大
                return "complex"  # 复杂区域
            else:
                return "main_object"  # 主要物体
        else:  # 低亮度
            if s_mean > 120:  # 高饱和度
                return "shadow_colorful"  # 彩色阴影
            else:
                return "shadow"  # 普通阴影
    
    def _regional_anime_conversion(self, img, region_features):
        """基于区域特征的动漫化：采用更保守的处理，保持原有色值"""
        result = img.copy()
        
        for feature in region_features:
            y1, y2, x1, x2 = feature['coords']
            region = img[y1:y2, x1:x2]
            
            if region.size == 0:
                continue
            
            # 根据精细区域类型应用不同的动漫化处理
            if feature['type'] in ["bright_white", "bright_pastel"]:
                # 亮色区域：极轻微简化，保持原有亮度和色彩
                processed = cv2.bilateralFilter(region, d=3, sigmaColor=5, sigmaSpace=5)
            elif feature['type'] == "bright_colorful":
                # 亮色鲜艳区域：轻微简化，保持鲜艳度
                processed = cv2.bilateralFilter(region, d=5, sigmaColor=8, sigmaSpace=8)
            elif feature['type'] in ["light_neutral", "light_colorful"]:
                # 浅色区域：适度简化，保持原有色调
                processed = cv2.bilateralFilter(region, d=7, sigmaColor=12, sigmaSpace=12)
            elif feature['type'] == "complex_detailed":
                # 复杂细节区域：保留更多细节，轻微简化
                processed = cv2.bilateralFilter(region, d=3, sigmaColor=6, sigmaSpace=6)
            elif feature['type'] in ["medium_colorful", "medium_neutral"]:
                # 中等区域：标准简化
                processed = cv2.bilateralFilter(region, d=9, sigmaColor=15, sigmaSpace=15)
            elif feature['type'] in ["dark_colorful", "dark_neutral"]:
                # 暗色区域：适度简化，保持阴影层次
                processed = cv2.bilateralFilter(region, d=11, sigmaColor=20, sigmaSpace=20)
            else:  # very_dark
                # 极暗区域：轻微简化，避免过度处理
                processed = cv2.bilateralFilter(region, d=5, sigmaColor=10, sigmaSpace=10)
            
            # 与原图混合，保持更多原有特征（混合比例根据区域类型调整）
            if feature['type'] in ["bright_white", "bright_pastel", "complex_detailed"]:
                # 重要区域：保持更多原图特征
                blended = cv2.addWeighted(region, 0.7, processed, 0.3, 0)
            elif feature['type'] in ["bright_colorful", "light_neutral", "light_colorful"]:
                # 主要区域：平衡处理
                blended = cv2.addWeighted(region, 0.6, processed, 0.4, 0)
            else:
                # 次要区域：较多动漫化处理
                blended = cv2.addWeighted(region, 0.4, processed, 0.6, 0)
            
            # 将处理后的区域放回原图
            result[y1:y2, x1:x2] = blended
        
        return result
    
    def _extract_ghibli_key_details(self, img):
        """提取宫崎骏关键细节：基于真实宫崎骏图片边缘密度优化"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 基于宫崎骏图片分析：边缘密度较低(0.027-0.153)，注重重要轮廓
        
        # 多尺度细节提取，适应不同粗细的轮廓
        # 精细细节（眼睛、毛发等细微特征）
        laplacian_fine = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
        laplacian_fine = np.absolute(laplacian_fine)
        
        # 中等细节（主要轮廓和形状）
        laplacian_medium = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian_medium = np.absolute(laplacian_medium)
        
        # 粗细节（整体结构）
        laplacian_coarse = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        laplacian_coarse = np.absolute(laplacian_coarse)
        
        # 加权组合：宫崎骏风格注重中等细节，减少过多细碎边缘
        combined_details = (laplacian_fine * 0.3 + 
                          laplacian_medium * 0.45 + 
                          laplacian_coarse * 0.25)
        
        # 归一化并转换为8位
        combined_details = cv2.normalize(combined_details, None, 0, 255, cv2.NORM_MINMAX)
        combined_details = combined_details.astype(np.uint8)
        
        # 自适应阈值：宫崎骏风格边缘清晰但不密集
        detail_mask = cv2.adaptiveThreshold(combined_details, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 15, 3)  # 增大块大小，减少细碎边缘
        
        # 形态学优化：连接重要轮廓，去除孤立噪点
        kernel = np.ones((2, 2), np.uint8)
        detail_mask = cv2.morphologyEx(detail_mask, cv2.MORPH_CLOSE, kernel)
        detail_mask = cv2.morphologyEx(detail_mask, cv2.MORPH_OPEN, kernel)
        
        # 进一步减少边缘密度，保持宫崎骏的简洁风格
        detail_mask = cv2.medianBlur(detail_mask, 3)
        
        return detail_mask
    
    def _regional_ghibli_color_mapping(self, img, region_features):
        """基于区域特征的宫崎骏色彩映射：采用更保守的色彩增强"""
        result = img.copy()
        
        for feature in region_features:
            y1, y2, x1, x2 = feature['coords']
            region = img[y1:y2, x1:x2]
            
            if region.size == 0:
                continue
            
            # 根据精细区域类型应用不同的色彩处理（更保守）
            if feature['type'] in ["bright_white", "bright_pastel"]:
                # 亮色区域：极轻微色彩增强，保持原有亮度
                processed = self._conservative_enhance_bright_colors(region)
            elif feature['type'] == "bright_colorful":
                # 亮色鲜艳区域：轻微增强饱和度
                processed = self._conservative_enhance_colorful_colors(region)
            elif feature['type'] in ["light_neutral", "light_colorful"]:
                # 浅色区域：适度色彩增强
                processed = self._conservative_enhance_light_colors(region)
            elif feature['type'] == "complex_detailed":
                # 复杂细节区域：保持原有色彩，仅轻微增强
                processed = self._conservative_enhance_complex_colors(region)
            elif feature['type'] in ["medium_colorful", "medium_neutral"]:
                # 中等区域：标准色彩增强
                processed = self._conservative_enhance_medium_colors(region)
            elif feature['type'] in ["dark_colorful", "dark_neutral"]:
                # 暗色区域：适度提亮和增强
                processed = self._conservative_enhance_dark_colors(region)
            else:  # very_dark
                # 极暗区域：轻微提亮，保持阴影感
                processed = self._conservative_enhance_very_dark_colors(region)
            
            # 与原图混合，保持更多原有色彩（混合比例根据区域类型调整）
            if feature['type'] in ["bright_white", "bright_pastel", "complex_detailed"]:
                # 重要区域：保持更多原图色彩
                blended = cv2.addWeighted(region, 0.8, processed, 0.2, 0)
            elif feature['type'] in ["bright_colorful", "light_neutral", "light_colorful"]:
                # 主要区域：平衡色彩处理
                blended = cv2.addWeighted(region, 0.7, processed, 0.3, 0)
            else:
                # 次要区域：较多色彩增强
                blended = cv2.addWeighted(region, 0.6, processed, 0.4, 0)
            
            # 将处理后的区域放回原图
            result[y1:y2, x1:x2] = blended
        
        return result
    
    def _enhance_sky_colors(self, region):
        """增强天空色彩：宫崎骏风格的蓝天白云"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 增强蓝色调（H=90-120）
        blue_mask = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 120)
        hsv[blue_mask, 1] = hsv[blue_mask, 1] * 1.3  # 增强饱和度
        hsv[blue_mask, 2] = hsv[blue_mask, 2] * 1.2  # 提高亮度
        
        # 增强白色区域（高亮度低饱和度）
        white_mask = (hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 50)
        hsv[white_mask, 2] = 220  # 设置为标准白色亮度
        
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _enhance_highlight_colors(self, region):
        """增强高光区域色彩：温暖明亮"""
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        # 提高亮度，增强温暖感
        l = np.clip(l * 1.1, 0, 255)
        a = np.clip(a * 1.05 + 5, 0, 255)  # 偏向红色
        b = np.clip(b * 1.08 + 8, 0, 255)  # 偏向黄色
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def _enhance_main_object_colors(self, region):
        """增强主要物体色彩：保持真实感，适度鲜艳"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 适度增强饱和度，保持自然
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 60, 180)
        
        # 轻微提高亮度
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 80, 220)
        
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _enhance_complex_colors(self, region):
        """增强复杂区域色彩：保持多样性"""
        # 复杂区域保持原有色彩分布，仅轻微增强
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.02, 0, 255)
        
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _enhance_shadow_colors(self, region):
        """增强阴影区域色彩：提高可见度，保持阴影感"""
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        # 提高阴影区域亮度，但保持阴影层次
        l = np.clip(l * 1.3, 0, 180)  # 限制最大亮度，保持阴影感
        
        # 增强色彩饱和度
        a = np.clip(a * 1.2, 0, 255)
        b = np.clip(b * 1.2, 0, 255)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def _enhance_shadow_lighting(self, region):
        """增强普通阴影照明：轻微提亮"""
        # 简单提亮处理
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        l = np.clip(l * 1.2, 0, 150)  # 限制亮度，保持阴影感
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def _conservative_enhance_bright_colors(self, region):
        """保守增强亮色区域：极轻微处理，保持原有特征"""
        # 几乎不改变原有色彩，仅轻微调整
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.05, 0, 255)  # 极轻微饱和度增强
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.02, 0, 255)  # 极轻微亮度调整
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _conservative_enhance_colorful_colors(self, region):
        """保守增强鲜艳色彩：轻微增强饱和度"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)  # 轻微饱和度增强
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)  # 轻微亮度调整
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _conservative_enhance_light_colors(self, region):
        """保守增强浅色区域：适度色彩增强"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)   # 适度饱和度增强
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.08, 0, 255)  # 适度亮度调整
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _conservative_enhance_complex_colors(self, region):
        """保守增强复杂区域：保持色彩多样性"""
        # 复杂区域保持原有色彩分布，仅极轻微增强
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.08, 0, 255)  # 极轻微饱和度增强
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.03, 0, 255)  # 极轻微亮度调整
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _conservative_enhance_medium_colors(self, region):
        """保守增强中等区域：标准色彩增强"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)  # 标准饱和度增强
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)   # 标准亮度调整
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _conservative_enhance_dark_colors(self, region):
        """保守增强暗色区域：适度提亮和增强"""
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        l = np.clip(l * 1.25, 0, 180)  # 适度提亮，保持阴影感
        a = np.clip(a * 1.15, 0, 255)  # 轻微色彩增强
        b = np.clip(b * 1.15, 0, 255)  # 轻微色彩增强
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def _conservative_enhance_very_dark_colors(self, region):
        """保守增强极暗区域：轻微提亮"""
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        l = np.clip(l * 1.3, 0, 120)   # 轻微提亮，保持极暗感
        a = np.clip(a * 1.1, 0, 255)   # 极轻微色彩增强
        b = np.clip(b * 1.1, 0, 255)   # 极轻微色彩增强
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def _high_quality_detail_preservation(self, ghibli_colors, original_img):
        """高画质细节保留：在动漫化基础上恢复重要细节"""
        # 1. 提取宫崎骏关键细节
        detail_mask = self._extract_ghibli_key_details(original_img)
        
        # 2. 宫崎骏细节融合
        detailed_result = self._ghibli_detail_fusion(ghibli_colors, detail_mask, original_img)
        
        # 3. 宫崎骏轮廓优化
        contour_optimized = self._ghibli_contour_optimization(detailed_result, original_img)
        
        # 4. 最终宫崎骏风格调整
        final = self._final_ghibli_adjustment(contour_optimized)
        
        return final
    
    def _apply_ghibli_lighting(self, img):
        """应用宫崎骏光影处理：柔和的光线和阴影"""
        # 宫崎骏光影特征：柔和、自然、梦幻
        
        # 1. 创建柔和的光照效果
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 模拟柔和的环境光
        height, width = gray.shape
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # 创建中心渐变光照（模拟宫崎骏的柔和光源）
        center_y, center_x = height // 2, width // 2
        dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        light_map = 1.0 - (dist_from_center / max_dist) * 0.3
        
        # 应用光照效果到每个通道
        b, g, r = cv2.split(img)
        
        b = np.clip(b * light_map, 0, 255).astype(np.uint8)
        g = np.clip(g * light_map, 0, 255).astype(np.uint8)
        r = np.clip(r * light_map, 0, 255).astype(np.uint8)
        
        lighted = cv2.merge([b, g, r])
        
        # 2. 添加轻微的辉光效果
        blurred = cv2.GaussianBlur(lighted, (15, 15), 0)
        final = cv2.addWeighted(lighted, 0.85, blurred, 0.15, 0)
        
        return final
    
    def _ghibli_dreamy_lighting(self, img):
        """应用宫崎骏梦幻光影效果：柔和梦幻的光照"""
        # 使用现有的光影处理方法
        return self._apply_ghibli_lighting(img)
    
    def _ghibli_detail_fusion(self, base_img, detail_mask, original_img):
        """宫崎骏细节融合：精确保留和增强重要细节"""
        # 从原图提取高质量细节
        gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        # 使用双边滤波替代导向滤波，实现类似效果
        detail_enhanced = cv2.bilateralFilter(gray_original, d=5, sigmaColor=10, sigmaSpace=10)
        
        # 将细节增强结果扩展到彩色图像
        detail_enhanced_color = cv2.cvtColor(detail_enhanced, cv2.COLOR_GRAY2BGR)
        
        # 创建细节增强版本（更强烈的对比度）
        enhanced_contrast = cv2.addWeighted(base_img, 0.4, detail_enhanced_color, 0.6, 0)
        
        # 使用自适应融合：根据细节强度精确控制融合
        detail_intensity = detail_mask.astype(np.float32) / 255.0
        detail_intensity = cv2.GaussianBlur(detail_intensity, (3, 3), 0)
        detail_intensity_3d = np.stack([detail_intensity] * 3, axis=2)
        
        # 精确融合：在细节区域使用增强版本
        result = base_img * (1 - detail_intensity_3d) + enhanced_contrast * detail_intensity_3d
        
        # 最终锐化增强清晰度
        kernel = np.array([[-0.02, -0.02, -0.02],
                          [-0.02,  1.12, -0.02],
                          [-0.02, -0.02, -0.02]])
        result = cv2.filter2D(result.astype(np.float32), -1, kernel)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _ghibli_contour_optimization(self, img, original_img):
        """宫崎骏轮廓优化：清晰但不生硬的线条"""
        gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        # 宫崎骏轮廓特征：清晰但不生硬，有艺术感
        
        # 1. 多尺度边缘检测，适应不同粗细的轮廓
        edges_fine = cv2.Canny(gray_original, 25, 80)   # 精细边缘
        edges_medium = cv2.Canny(gray_original, 40, 120) # 中等边缘
        edges_coarse = cv2.Canny(gray_original, 60, 150) # 粗边缘
        
        # 组合多尺度边缘
        combined_edges = edges_fine | edges_medium | edges_coarse
        
        # 2. 对边缘进行艺术化处理
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(combined_edges, kernel, iterations=1)
        edges = cv2.GaussianBlur(edges.astype(np.float32), (3, 3), 0)
        
        # 3. 创建柔和的轮廓效果
        edge_mask = edges > 30
        
        # 宫崎骏风格：轮廓处轻微加深，但不生硬
        darkened_edges = cv2.multiply(img, 0.75).astype(np.uint8)
        
        # 渐进式融合：边缘强度越高，加深程度越大
        edge_strength = edges / 255.0
        edge_strength_3d = np.stack([edge_strength] * 3, axis=2)
        
        # 计算融合结果
        contour_blend = img * (1 - edge_strength_3d) + darkened_edges * edge_strength_3d
        
        # 最终与原始图像混合，保持自然感
        result = cv2.addWeighted(img, 0.3, contour_blend.astype(np.uint8), 0.7, 0)
        
        return result.astype(np.uint8)
    
    def _final_ghibli_adjustment(self, img):
        """最终宫崎骏风格调整：梦幻氛围"""
        # 宫崎骏最终调整：创造梦幻、和谐的氛围
        
        # 1. 轻微的色彩统一化
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # 减少色彩差异，创造和谐感
        lab_mean = np.mean(lab, axis=(0, 1))
        lab_std = np.std(lab, axis=(0, 1))
        
        # 限制色彩范围，避免过度鲜艳
        lab[:, :, 1] = np.clip(lab[:, :, 1], 
                              lab_mean[1] - lab_std[1] * 1.2, 
                              lab_mean[1] + lab_std[1] * 1.2)
        lab[:, :, 2] = np.clip(lab[:, :, 2], 
                              lab_mean[2] - lab_std[2] * 1.2, 
                              lab_mean[2] + lab_std[2] * 1.2)
        
        unified = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. 添加梦幻的辉光效果
        blurred_dream = cv2.GaussianBlur(unified, (21, 21), 0)
        dreamy = cv2.addWeighted(unified, 0.88, blurred_dream, 0.12, 0)
        
        # 3. 最终锐化，增强清晰度但不破坏柔和感
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        sharpened = cv2.filter2D(dreamy, -1, kernel)
        
        final = cv2.addWeighted(dreamy, 0.7, sharpened, 0.3, 0)
        
        return final
    
    def _final_optimization_preserve_details(self, img, original_img):
        """最终优化（细节保留）"""
        # 轻微锐化以增强清晰度
        kernel = np.array([[0, -0.25, 0],
                          [-0.25, 2, -0.25],
                          [0, -0.25, 0]])
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # 与原图混合，保留更多细节
        final = cv2.addWeighted(img, 0.8, sharpened, 0.2, 0)
        
        # 轻微的光晕效果（不影响细节）
        blurred = cv2.GaussianBlur(final, (5, 5), 0)
        final = cv2.addWeighted(final, 0.9, blurred, 0.1, 0)
        
        return final
    
    def _create_cartoon_base(self, img):
        """创建卡通基础层"""
        filtered = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
        return filtered
    
    def _extract_important_details(self, img):
        """提取重要细节"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)
        
        _, detail_mask = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)
        return detail_mask
    
    def _enhance_anime_colors(self, img):
        """增强动漫色彩"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.4)
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _smart_detail_fusion(self, base_img, detail_mask):
        """智能细节融合"""
        detail_mask_3d = cv2.cvtColor(detail_mask, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(base_img, 0.8, base_img, 0.2, 0)
        result = np.where(detail_mask_3d > 128, enhanced, base_img)
        return result.astype(np.uint8)
    
    def _create_painting_base(self, img):
        """创建绘画基础层"""
        oil = cv2.medianBlur(img, 3)
        watercolor = cv2.bilateralFilter(oil, d=7, sigmaColor=40, sigmaSpace=40)
        return watercolor
    
    def _extract_key_details(self, img):
        """提取关键细节"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)
        
        _, detail_mask = cv2.threshold(laplacian, 25, 255, cv2.THRESH_BINARY)
        return detail_mask
    
    def _apply_artistic_colors(self, img):
        """应用艺术色彩"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _artistic_detail_fusion(self, base_img, detail_mask):
        """艺术细节融合"""
        detail_mask_3d = cv2.cvtColor(detail_mask, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(base_img, 0.75, base_img, 0.25, 0)
        result = np.where(detail_mask_3d > 128, enhanced, base_img)
        return result.astype(np.uint8)

def image_to_base64(img):
    """将PIL图像转换为base64字符串"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# 初始化模型
model = GhibliStyleTransfer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 获取风格类型参数
    style_type = request.form.get('style_type', 'ghibli')
    
    if file:
        try:
            # 读取图像
            image = Image.open(file.stream)
            
            # 调整图像大小（限制最大尺寸）
            max_size = 800
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # 应用风格转换
            result_image = model(image, style_type=style_type)
            
            # 如果结果是numpy数组，转换为PIL图像
            if isinstance(result_image, np.ndarray):
                # 确保数据类型正确
                if result_image.dtype == np.float32 or result_image.dtype == np.float64:
                    result_image = (result_image * 255).astype(np.uint8)
                result_image = Image.fromarray(result_image)
            
            # 转换为base64
            original_b64 = image_to_base64(image)
            result_b64 = image_to_base64(result_image)
            
            return jsonify({
                'success': True,
                'original': original_b64,
                'result': result_b64,
                'style_type': style_type
            })
            
        except Exception as e:
            return jsonify({'error': f'处理图像时出错: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)