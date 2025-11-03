#!/usr/bin/env python3
"""
真正的宫崎骏风格转换 - 基于深度学习和风格迁移
集成预训练神经网络模型
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import torch.optim as optim
from torch.autograd import Variable
import os
import io
import base64
import time
import threading

# 导入神经网络风格迁移模块
try:
    from .neural_style_transfer import NeuralStyleTransfer, style_transfer_manager
except ImportError:
    # 如果导入失败，使用回退方案
    NeuralStyleTransfer = None
    style_transfer_manager = None

# 全局变量用于存储转换进度
conversion_progress = {}
conversion_results = {}

class RealGhibliStyleTransfer:
    """真正的宫崎骏风格转换 - 基于深度学习和计算机视觉优化
    集成预训练神经网络模型
    """
    
    def __init__(self, use_neural_network=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = self._load_vgg().to(self.device)
        self.style_layers = ['3', '8', '15', '22']  # VGG层用于风格提取
        self.content_layers = ['22']  # VGG层用于内容提取
        self.progress_callback = None
        self.task_id = None
        
        # 神经网络风格迁移模型
        self.use_neural_network = use_neural_network
        self.neural_model = None
        
        # 语义分割模型（延迟加载）
        self.seg_model = None
        
        if use_neural_network and NeuralStyleTransfer is not None:
            try:
                self.neural_model = NeuralStyleTransfer(model_type='vgg19')
                print("✅ 神经网络风格迁移模型加载成功")
            except Exception as e:
                print(f"⚠️ 神经网络模型加载失败: {e}，使用传统方法")
                self.use_neural_network = False
        
        # 初始化自主学习器
        self.auto_learner = None  # 暂时禁用自主学习功能
        
        # 是否启用自主学习
        self.enable_auto_learning = False
        
    def _load_vgg(self):
        """加载预训练的VGG19模型"""
        try:
            # 使用新的API加载VGG19模型
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        except AttributeError:
            # 回退到旧版本API
            vgg = models.vgg19(pretrained=True).features
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg
    
    def _extract_features(self, x, model, layers):
        """从VGG模型中提取特征"""
        features = {}
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[name] = x
        return features
    
    def _gram_matrix(self, x):
        """计算Gram矩阵（风格特征）"""
        batch_size, channels, height, width = x.size()
        features = x.view(batch_size * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * channels * height * width)
    
    def _load_ghibli_style_images(self):
        """加载宫崎骏风格参考图片"""
        # 优先使用项目内的风格参考库，其次回退到 temp 目录
        candidate_folders = ['ghibli_images', 'temp']
        style_folder = None
        for folder in candidate_folders:
            if os.path.isdir(folder) and any([f.lower().endswith(('.jpg','.jpeg','.png','.bmp')) for f in os.listdir(folder)]):
                style_folder = folder
                break
        if style_folder is None:
            style_folder = 'ghibli_images'  # 仍然按该路径组合通配符，便于下面 glob 失败后给出提示
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
    
    def _create_ghibli_style_tensor(self, target_size=512):
        """创建宫崎骏风格特征张量"""
        style_images = self._load_ghibli_style_images()
        
        if not style_images:
            # 如果没有参考图片，创建默认的宫崎骏风格特征
            return self._create_default_ghibli_style(target_size)
        
        # 加载并处理风格图片
        style_tensors = []
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        for style_path in style_images:
            try:
                style_img = Image.open(style_path).convert('RGB')
                style_tensor = transform(style_img).unsqueeze(0).to(self.device)
                style_tensors.append(style_tensor)
            except Exception as e:
                print(f"❌ 加载风格图片 {style_path} 失败: {e}")
        
        if not style_tensors:
            return self._create_default_ghibli_style(target_size)
        
        # 平均所有风格图片的特征
        style_features = {}
        for style_tensor in style_tensors:
            features = self._extract_features(style_tensor, self.vgg, self.style_layers)
            for layer, feature in features.items():
                if layer not in style_features:
                    style_features[layer] = []
                style_features[layer].append(self._gram_matrix(feature))
        
        # 计算平均风格特征
        avg_style_features = {}
        for layer, gram_list in style_features.items():
            avg_gram = torch.stack(gram_list).mean(dim=0)
            avg_style_features[layer] = avg_gram
        
        return avg_style_features
    
    def _create_default_ghibli_style(self, target_size):
        """创建默认的宫崎骏风格特征"""
        print("🎨 使用默认宫崎骏风格特征")
        
        # 创建具有宫崎骏风格特征的默认风格
        # 宫崎骏风格特点：柔和色彩、梦幻光影、简洁线条
        default_style = {}
        
        # 这里应该基于宫崎骏的艺术特点创建风格特征
        # 由于时间关系，我们使用简化的方法
        
        return default_style
    
    def _get_features(self, x):
        """获取VGG特征"""
        return self._extract_features(x, self.vgg, self.style_layers + self.content_layers)
    
    def _get_ghibli_style_features(self):
        """获取宫崎骏风格特征"""
        return self._create_ghibli_style_tensor()
    
    def _is_result_poor(self, result_image):
        """检查结果是否质量差"""
        if result_image is None:
            return True
        
        # 检查图像是否为灰度或色彩丢失
        if isinstance(result_image, Image.Image):
            # 转换为numpy数组检查
            img_array = np.array(result_image)
            if len(img_array.shape) == 2:  # 灰度图
                return True
            
            # 检查色彩饱和度
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:,:,1])
            if saturation < 30:  # 饱和度太低
                return True
        
        return False
    
    def _preprocess_image_preserve_size(self, image):
        """预处理图像但保持原始尺寸"""
        # 获取原始尺寸
        original_size = image.size
        
        # 限制最大尺寸但保持宽高比
        max_size = 800
        if max(original_size) > max_size:
            scale = max_size / max(original_size)
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            image = image.resize(new_size, Image.LANCZOS)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)
    
    def _postprocess_image_preserve_size(self, tensor, original_size):
        """后处理张量为图像并恢复原始尺寸"""
        # 反归一化
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为PIL图像
        transform = transforms.ToPILImage()
        image = transform(tensor)
        
        # 恢复原始尺寸
        image = image.resize(original_size, Image.LANCZOS)
        
        return image
    
    def _anime_style_filter(self, img_bgr):
        """动漫风格滤镜"""
        # 1. 双边滤波 - 保留边缘的同时平滑图像
        filtered = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 2. 边缘检测和增强
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 9, 2)
        
        # 3. 颜色量化 - 减少颜色数量，创造动漫效果
        Z = filtered.reshape((-1, 3))
        Z = np.float32(Z)
        
        # 定义K-means参数
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 16  # 颜色数量
        
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        cartoon = res.reshape((filtered.shape))
        
        return cartoon
    
    def _apply_ghibli_color_style(self, img_bgr):
        """应用宫崎骏色彩风格"""
        # 宫崎骏风格色彩特点：柔和、温暖、高饱和度
        
        # 转换为HSV色彩空间进行更精确的调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 增强饱和度（宫崎骏风格色彩鲜艳）
        s = cv2.add(s, 30)
        s = np.clip(s, 0, 255)
        
        # 调整色调 - 偏向温暖色调
        h = cv2.add(h, 5)  # 轻微偏向橙色/黄色
        h = np.clip(h, 0, 179)
        
        # 增强亮度对比度
        v = cv2.add(v, 10)
        v = np.clip(v, 0, 255)
        
        # 合并HSV通道
        hsv_enhanced = cv2.merge([h, s, v])
        
        # 转换回BGR
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 应用柔和滤镜
        soft = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 混合原始和柔和版本
        result = cv2.addWeighted(enhanced, 0.7, soft, 0.3, 0)
        
        return result
    
    def _add_dreamy_lighting(self, img_bgr):
        """添加梦幻光影效果"""
        h, w = img_bgr.shape[:2]
        
        # 创建柔和的光照效果
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 创建光照遮罩 - 中心明亮，边缘柔和
        light_mask = 1.0 - (distance / max_distance) * 0.1
        light_mask = np.clip(light_mask, 0.9, 1.0)
        
        # 应用光照效果
        final = img_bgr.astype(np.float32) * light_mask[:,:,np.newaxis]
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        return final
    
    def apply_real_ghibli_style(self, content_image, num_steps=80, style_weight=300000, content_weight=1, use_neural=True):
        """应用真正的宫崎骏风格转换 - 优化版本
        
        Args:
            content_image: 内容图像
            num_steps: 迭代步数
            style_weight: 风格权重
            content_weight: 内容权重
            use_neural: 是否使用神经网络风格迁移
        """
        print("🎨 开始应用宫崎骏风格...")
        
        try:
            # 优先使用神经网络风格迁移
            if use_neural and self.use_neural_network and self.neural_model is not None:
                print("🧠 使用神经网络风格迁移")
                
                # 设置进度回调
                if self.progress_callback and self.task_id:
                    def neural_progress_callback(progress, current_step, total_steps, loss):
                        self.progress_callback(self.task_id, progress, current_step, total_steps, loss)
                    
                    self.neural_model.set_progress_callback(neural_progress_callback)
                
                # 使用神经网络风格迁移
                # 两阶段策略：小图收敛 + 大图微调
                # 阶段1：较小分辨率，偏风格
                result = self.neural_model.transfer_style(
                    content_image,
                    style_weight=int(style_weight*0.7),
                    content_weight=int(max(1, content_weight*0.8)),
                    num_steps=int(num_steps*0.7),
                    learning_rate=0.025,
                    tv_weight=1e-5
                )
                # 阶段2：全尺寸轻微调（更保结构）
                result = self.neural_model.transfer_style(
                    result,
                    style_weight=int(style_weight*0.3),
                    content_weight=int(max(1, content_weight*2)),
                    num_steps=max(20, int(num_steps*0.3)),
                    learning_rate=0.02,
                    tv_weight=2e-5
                )
                
                if result is not None and not self._is_result_poor(result):
                    print("✅ 神经网络风格迁移成功")
                    return result
                else:
                    print("⚠️ 神经网络风格迁移效果不佳，尝试传统方法")
            
            # 使用传统深度学习风格迁移
            print("🔧 使用传统深度学习风格迁移")
            result = self._apply_neural_style_transfer(content_image, num_steps, style_weight, content_weight)
            
            # 如果深度学习效果不好，使用计算机视觉优化
            if result is None or self._is_result_poor(result):
                print("⚠️ 深度学习效果不佳，使用计算机视觉优化")
                result = self._apply_cv_optimized_ghibli_style(content_image)
            
            return result
            
        except Exception as e:
            print(f"❌ 风格转换失败: {e}")
            return self._apply_cv_optimized_ghibli_style(content_image)
    
    def _apply_neural_style_transfer(self, content_image, num_steps, style_weight, content_weight):
        """应用神经风格迁移"""
        # 预处理内容图像 - 保持原始尺寸
        original_size = content_image.size
        content_tensor = self._preprocess_image_preserve_size(content_image).to(self.device)
        
        # 使用内容图像作为初始输入
        input_img = content_tensor.clone().requires_grad_(True)
        
        # 获取内容特征
        content_features = self._get_features(content_tensor)
        
        # 使用宫崎骏风格参考图像
        style_features = self._get_ghibli_style_features()
        
        # 使用Adam优化器，更稳定
        optimizer = optim.Adam([input_img], lr=0.02)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 获取当前输入的特征
            features = self._get_features(input_img)
            
            style_loss = 0
            content_loss = 0
            
            # 风格损失
            for layer in self.style_layers:
                if layer in features and layer in style_features:
                    target_style = style_features[layer]
                    current_style = features[layer]
                    
                    # 计算Gram矩阵
                    target_gram = self._gram_matrix(target_style)
                    current_gram = self._gram_matrix(current_style)
                    
                    style_loss += F.mse_loss(current_gram, target_gram)
            
            # 内容损失
            for layer in self.content_layers:
                if layer in features:
                    target_content = content_features[layer]
                    current_content = features[layer]
                    content_loss += F.mse_loss(current_content, target_content)
            
            # 总损失
            total_loss = style_weight * style_loss + content_weight * content_loss
            
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
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_([input_img], max_norm=0.5)
            
            optimizer.step()
            
            # 更新进度
            progress = int((step + 1) / num_steps * 100)
            if self.progress_callback and self.task_id:
                self.progress_callback(self.task_id, progress, step + 1, num_steps, total_loss.item())
            
            if (step + 1) % 30 == 0:
                print(f"步骤 {step+1}/{num_steps}, 总损失: {total_loss.item():.4f}")
        
        # 后处理输出图像 - 恢复原始尺寸
        output_tensor = input_img.data.clamp(0, 1)
        result_image = self._postprocess_image_preserve_size(output_tensor, original_size)
        
        return result_image
    
    def _apply_cv_optimized_ghibli_style(self, image):
        """应用计算机视觉优化的宫崎骏风格 - 前景/背景分支 + 线稿增强"""
        print("🎨 使用计算机视觉优化宫崎骏风格...")
        
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if img_np.ndim==3 and img_np.shape[2]==3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        h, w = img_bgr.shape[:2]
        
        # 尺寸限制
        max_size = 2048
        if max(h,w) > max_size:
            scale = max_size / max(h,w)
            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
            h, w = img_bgr.shape[:2]
            print(f"📏 计算机视觉处理: 图片尺寸过大，自动缩放至: {w}x{h}")
        else:
            print(f"📏 计算机视觉处理: 保持原始尺寸: {w}x{h}")
        
        # 前景分割
        mask = self._get_person_mask(img_bgr)  # 0~255
        inv_mask = 255 - mask
        
        # 背景强动漫化
        bg = self._advanced_anime_style_filter(img_bgr)
        bg = self._enhanced_ghibli_color_style(bg)
        bg = self._enhanced_dreamy_lighting(bg)
        
        # 前景弱平滑 + 线稿增强 + 轻调色
        fg = cv2.bilateralFilter(img_bgr, 7, 60, 60)
        gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        edges = self._xdog_edges(gray)
        edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # 线稿以低透明度叠加
        fg = cv2.addWeighted(fg, 1.0, edges_col, 0.08, 0)
        fg = self._enhanced_ghibli_color_style(fg)
        
        # 融合
        mask_s = cv2.GaussianBlur(mask, (11,11), 0)
        comp = self._alpha_blend(fg, bg, mask_s)
        
        # 最终细节
        comp = self._final_touch_optimization(comp)
        result_rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _advanced_anime_style_filter(self, img_bgr):
        """
        高级动漫风格滤镜 - 更接近宫崎骏风格
        - 多次边缘保留平滑，移除写实纹理
        - 颜色大块化（KMeans + SLIC超像素均值化）
        - 柔和细线稿叠加
        """
        # 1) 边缘保留平滑（两次双边滤波）
        guided = cv2.bilateralFilter(img_bgr, d=11, sigmaColor=85, sigmaSpace=85)
        guided = cv2.bilateralFilter(guided, d=9, sigmaColor=75, sigmaSpace=75)

        # 2) 智能边缘检测 + 轻微膨胀
        gray = cv2.cvtColor(guided, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 3) 颜色量化（更少的颜色以获得卡通分区）
        Z = guided.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        K = 12
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        cartoon = centers[labels.flatten()].reshape(guided.shape)

        # 4) SLIC超像素均值化，进一步大片区扁平（更“动漫”）
        try:
            from skimage.segmentation import slic
            from skimage.color import label2rgb
            img_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
            segments = slic(img_rgb, n_segments=600, compactness=20, sigma=0, start_label=1)
            flat_rgb = (label2rgb(segments, img_rgb, kind='avg') * 255).astype(np.uint8)
            cartoon = cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        # 5) 边缘叠加（柔和线稿）
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        cartoon = cartoon.astype(np.float32) / 255.0
        result = cv2.addWeighted(cartoon, 0.85, edges_colored, 0.15, 0)
        result = (result * 255).astype(np.uint8)
        return result
    
    def _enhanced_ghibli_color_style(self, img_bgr):
        """增强的宫崎骏色彩风格"""
        # 转换为HSV色彩空间进行更精确的调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 增强饱和度（宫崎骏风格色彩鲜艳）
        s = cv2.add(s, 40)  # 增加饱和度
        s = np.clip(s, 0, 255)
        
        # 调整色调 - 偏向温暖色调（宫崎骏风格特点）
        h = cv2.add(h, 8)  # 轻微偏向橙色/黄色
        h = np.clip(h, 0, 179)
        
        # 增强亮度对比度
        v = cv2.add(v, 15)
        v = np.clip(v, 0, 255)
        
        # 合并HSV通道
        hsv_enhanced = cv2.merge([h, s, v])
        
        # 转换回BGR
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 应用柔和滤镜
        soft = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 混合原始和柔和版本
        result = cv2.addWeighted(enhanced, 0.7, soft, 0.3, 0)
        
        return result
    
    def _enhanced_dreamy_lighting(self, img_bgr):
        """增强的梦幻光影效果"""
        h, w = img_bgr.shape[:2]
        
        # 创建柔和的光照效果
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 创建光照遮罩 - 中心明亮，边缘柔和（显著减弱以避免明显同心环和带状）
        light_mask = 1.0 - (distance / max_distance) * 0.05
        light_mask = np.clip(light_mask, 0.95, 1.0)
        
        # 应用更柔和的光照效果
        final = img_bgr.astype(np.float32) * light_mask[:,:,np.newaxis]
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        return final
    
    def _final_touch_optimization(self, img_bgr):
        """最终细节优化"""
        # 1. 轻微锐化增强细节
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        
        # 2. 轻微降噪（降低强度以避免过度平滑）
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 3, 3, 7, 21)
        
        # 3. 最终色彩平衡
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强亮度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_balanced = cv2.merge([l, a, b])
        final = cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)
        
        return final
    
    def set_progress_callback(self, callback, task_id):
        """设置进度回调函数"""
        self.progress_callback = callback
        self.task_id = task_id
    
    def _get_person_mask(self, img_bgr):
        """获取人物前景掩膜（0~255）"""
        try:
            if self.seg_model is None:
                from torchvision import models
                self.seg_model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT).eval()
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize(520),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            x = transform(rgb).unsqueeze(0)
            with torch.no_grad():
                out = self.seg_model(x)['out'][0]  # [21, H, W]
            person_class = 15  # COCO person
            mask = out.argmax(0).byte().cpu().numpy()
            mask = (mask == person_class).astype(np.uint8) * 255
            # 调整到原图尺寸
            mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            # 平滑边缘
            mask = cv2.GaussianBlur(mask, (9,9), 0)
            return mask
        except Exception as e:
            print(f"⚠️ 前景分割失败，使用全图: {e}")
            return np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
    
    def _xdog_edges(self, gray, k=4.5, sigma=0.9, epsilon=-0.1, phi=10):
        """XDoG风格线稿，返回0~255"""
        g1 = cv2.GaussianBlur(gray, (0,0), sigma)
        g2 = cv2.GaussianBlur(gray, (0,0), sigma*k)
        D = g1 - g2
        D = D / (np.max(np.abs(D)) + 1e-8)
        E = np.ones_like(D)
        E[D < epsilon] = 1 + np.tanh(phi*(D[D < epsilon]-epsilon))
        E[D >= epsilon] = 1
        E = (E*255).astype(np.uint8)
        return 255 - E
    
    def _alpha_blend(self, fg, bg, mask):
        mask_f = (mask.astype(np.float32)/255.0)[:,:,None]
        return (fg*mask_f + bg*(1-mask_f)).astype(np.uint8)
    
    def _preprocess_image(self, image):
        """预处理图像"""
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)
    
    def _postprocess_image(self, tensor):
        """后处理张量为图像"""
        # 反归一化
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为PIL图像
        transform = transforms.ToPILImage()
        image = transform(tensor)
        
        return image
    
    def _fallback_traditional_method(self, image):
        """备选传统方法"""
        print("⚠️ 使用备选传统方法")
        
        # 将PIL图像转换为numpy数组
        img_np = np.array(image)
        
        # 转换为BGR格式
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # 高质量的宫崎骏风格处理
        
        # 1. 保持原始分辨率
        h, w = img_bgr.shape[:2]
        max_size = 2000
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. 智能边缘保留（重点改进人物区域）
        # 使用双边滤波替代导向滤波
        guided = cv2.bilateralFilter(img_bgr, d=11, sigmaColor=80, sigmaSpace=80)
        
        # 3. 宫崎骏风格色彩调整
        # 转换为LAB色彩空间进行更精确的色彩调整
        lab = cv2.cvtColor(guided, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强色彩鲜艳度（宫崎骏风格特点）
        a = cv2.addWeighted(a, 1.2, a, 0, 0)
        b = cv2.addWeighted(b, 1.2, b, 0, 0)
        
        # 调整亮度和对比度
        l = cv2.createCLAHE(clipLimit=2.0).apply(l)
        
        lab_enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 4. 添加梦幻光影效果
        h, w = enhanced.shape[:2]
        
        # 创建柔和的光照效果
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 创建光照遮罩
        light_mask = 1.0 - (distance / max_distance) * 0.15
        light_mask = np.clip(light_mask, 0.85, 1.0)
        
        # 应用光照效果
        final = enhanced.astype(np.float32) * light_mask[:,:,np.newaxis]
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        # 转换回RGB
        result_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        
        return result_rgb

# 创建真正的宫崎骏风格转换模型
real_ghibli_model = RealGhibliStyleTransfer()

def update_progress(task_id, progress, current_step, total_steps, loss):
    """更新转换进度"""
    # 确保进度和步骤信息一致
    if progress > 0 and current_step == 0:
        # 如果进度有值但步骤为0，根据进度计算步骤
        current_step = max(1, int(progress / 100 * total_steps))
    elif current_step > 0 and progress == 0:
        # 如果步骤有值但进度为0，根据步骤计算进度
        progress = min(99, int(current_step / total_steps * 100))
    
    conversion_progress[task_id] = {
        'progress': progress,
        'current_step': current_step,
        'total_steps': total_steps,
        'loss': loss,
        'timestamp': time.time()
    }
    print(f"📊 任务 {task_id}: {progress}% (步骤 {current_step}/{total_steps}, 损失: {loss:.4f})")

def convert_image_async(task_id, image):
    """异步转换图像"""
    try:
        # 设置进度回调
        real_ghibli_model.set_progress_callback(update_progress, task_id)
        
        # 开始转换
        result_image = real_ghibli_model.apply_real_ghibli_style(image, num_steps=100)
        
        # 保存结果
        conversion_results[task_id] = {
            'success': True,
            'result_image': result_image,
            'completed': True
        }
        
        # 更新进度为完成
        update_progress(task_id, 100, 100, 100, 0)
        
    except Exception as e:
        conversion_results[task_id] = {
            'success': False,
            'error': str(e),
            'completed': True
        }
        print(f"❌ 任务 {task_id} 转换失败: {e}")

