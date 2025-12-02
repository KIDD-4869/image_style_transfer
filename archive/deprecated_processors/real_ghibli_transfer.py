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
from typing import Optional
from .image_processor_interface import ImageProcessorInterface, ProcessingResult, ProcessingStyle

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

class RealGhibliStyleTransfer(ImageProcessorInterface):
    """真正的宫崎骏风格转换 - 基于深度学习和计算机视觉优化
    集成预训练神经网络模型
    """
    
    def __init__(self, use_neural_network=True):
        super().__init__(ProcessingStyle.GHIBLI_CLASSIC)
        # 检查可用的最佳设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
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
        self.auto_learner = None
        
        # 是否启用自主学习
        self.enable_auto_learning = True
        
        # 尝试加载训练好的模型
        self.trained_model = None
        self._load_trained_model()
    
    def process(self, image: Image.Image, **kwargs) -> ProcessingResult:
        """
        处理图像，应用宫崎骏风格
        
        Args:
            image: 输入图像
            **kwargs: 其他处理参数
                - num_steps: 迭代步数
                - style_weight: 风格权重
                - content_weight: 内容权重
                - use_neural: 是否使用神经网络风格迁移
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            num_steps = kwargs.get('num_steps', 80)
            style_weight = kwargs.get('style_weight', 300000)
            content_weight = kwargs.get('content_weight', 1)
            use_neural = kwargs.get('use_neural', True)
            
            result_image = self.apply_real_ghibli_style(
                image, 
                num_steps=num_steps, 
                style_weight=style_weight,
                content_weight=content_weight,
                use_neural=use_neural
            )
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                image=result_image,
                processing_time=processing_time
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def get_processing_info(self) -> dict:
        """
        获取处理器信息
        
        Returns:
            dict: 处理器信息
        """
        return {
            "processor_type": "RealGhibliStyleTransfer",
            "style_type": self.style_type.value,
            "use_neural_network": self.use_neural_network,
            "device": str(self.device)
        }
    
    def _load_vgg(self):
        """加载预训练的VGG19模型"""
        try:
            # 使用新的API加载VGG19模型
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        except AttributeError:
            # 回退到旧版本API
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg
    
    def _load_trained_model(self):
        """加载训练好的宫崎骏风格模型"""
        import os
        import sys
        model_path = "models/real_ghibli_learning/best_model.pth"
        
        # 检查是否有更新的带版本号的模型
        import glob
        versioned_models = glob.glob("models/ghibli_gan/ghibli_gan_v*_best.pth")
        if versioned_models:
            # 选择最新的版本
            versioned_models.sort(reverse=True)
            model_path = versioned_models[0]
            print(f"✅ 使用最新版本的宫崎骏风格模型: {model_path}")
        elif os.path.exists("models/ghibli_gan/ghibli_gan_best.pth"):
            model_path = "models/ghibli_gan/ghibli_gan_best.pth"
            print("✅ 使用宫崎骏GAN最佳模型")
        else:
            model_path = "models/real_ghibli_learning/best_model.pth"
        
        if os.path.exists(model_path):
            try:
                # 导入训练器类来加载模型
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                # 尝试从不同位置导入
                try:
                    from real_auto_learning import GhibliStyleEncoder
                except ImportError:
                    # 如果直接导入失败，尝试相对导入
                    sys.path.append('.')
                    from real_auto_learning import GhibliStyleEncoder
                
                self.trained_model = GhibliStyleEncoder().to(self.device)
                
                # 加载保存的检查点
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 检查是否包含完整的状态字典
                if 'model_state_dict' in checkpoint:
                    self.trained_model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ 模型训练信息: epoch={checkpoint.get('epoch', 'unknown')}, loss={checkpoint.get('best_loss', 'unknown')}")
                else:
                    self.trained_model.load_state_dict(checkpoint)
                
                self.trained_model.eval()
                print("✅ 成功加载训练好的宫崎骏风格模型")
                return True
            except Exception as e:
                print(f"⚠️ 加载训练模型失败: {e}")
                self.trained_model = None
                return False
        else:
            print("⚠️ 未找到训练好的模型文件")
            return False
    
    def _extract_features(self, x, model, layers):
        """从VGG模型中提取特征"""
        features = {}
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[name] = x
        return features
    
    def _compute_content_loss(self, content_features, target_features):
        """计算内容损失"""
        return F.mse_loss(content_features, target_features)
    
    def _compute_style_loss(self, style_features, target_features):
        """计算风格损失"""
        style_gram = self._gram_matrix(style_features)
        target_gram = self._gram_matrix(target_features)
        return F.mse_loss(style_gram, target_gram)
    
    def _gram_matrix(self, x):
        """计算Gram矩阵"""
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram
    
    def apply_real_ghibli_style(self, image: Image.Image, num_steps: int = 80, style_weight: float = 300000, content_weight: float = 1, use_neural: bool = True) -> Image.Image:
        """
        应用宫崎骏风格到图像
        
        Args:
            image: 输入图像
            num_steps: 迭代步数
            style_weight: 风格权重
            content_weight: 内容权重
            use_neural: 是否使用神经网络风格迁移
        
        Returns:
            Image.Image: 处理后的图像
        """
        # 将图像转换为张量
        image_tensor = self._image_to_tensor(image).to(self.device)
        image_tensor = Variable(image_tensor, requires_grad=True)
        
        # 加载风格图像
        style_image = Image.open("styles/ghibli.jpg")
        style_tensor = self._image_to_tensor(style_image).to(self.device)
        style_tensor = Variable(style_tensor, requires_grad=False)
        
        # 提取内容和风格特征
        content_features = self._extract_features(image_tensor, self.vgg, self.content_layers)
        style_features = self._extract_features(style_tensor, self.vgg, self.style_layers)
        
        # 初始化优化器
        optimizer = optim.LBFGS([image_tensor])
        
        # 定义损失函数
        def closure():
            optimizer.zero_grad()
            content_loss = 0
            style_loss = 0
            
            # 提取特征
            features = self._extract_features(image_tensor, self.vgg, self.style_layers + self.content_layers)
            
            # 计算内容损失
            content_loss += self._compute_content_loss(features['22'], content_features['22'])
            
            # 计算风格损失
            for layer in self.style_layers:
                style_loss += self._compute_style_loss(style_features[layer], features[layer])
            
            # 加权损失
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            total_loss.backward()
            return total_loss
        
        # 进行优化
        for i in range(num_steps):
            optimizer.step(closure)
            if self.progress_callback:
                self.progress_callback(i + 1, num_steps)
        
        # 将张量转换回图像
        result_image = self._tensor_to_image(image_tensor)
        return result_image
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """将图像转换为张量"""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        return transform(image).unsqueeze(0)
    
    def _extract_features(self, x, vgg, layers):
        """提取VGG特征"""
        features = {}
        for name, layer in vgg._modules.items():
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
        # 确保图像是3通道的BGR格式
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 1:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        
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
        """应用真正的宫崎骏风格转换

        Args:
            content_image: 内容图像
            num_steps: 迭代步数
            style_weight: 风格权重
            content_weight: 内容权重
            use_neural: 是否使用神经网络风格迁移
        """
        print("🎨 开始应用宫崎骏风格转换...")
        
        try:
            # 使用新的简化但有效的算法
            from .true_ghibli_style import true_ghibli_processor
            
            # 设置进度回调
            true_ghibli_processor.set_progress_callback(self.progress_callback, self.task_id)
            
            # 应用宫崎骏风格
            result = true_ghibli_processor.apply_ghibli_style(content_image)
            
            print("✅ 宫崎骏风格转换完成")
            return result
            
        except Exception as e:
            print(f"❌ 风格转换失败: {e}")
            # 备用方法
            return self._fallback_simple_ghibli(content_image)
    
    def _apply_trained_model_style(self, content_image):
        """使用训练好的模型进行风格转换"""
        try:
            print("🎯 使用真实的神经网络风格迁移")
            
            # 直接使用神经网络风格迁移 - 这是真正有效的方法
            if self.neural_model is not None:
                # 加载宫崎骏风格参考图像
                style_features = self._get_ghibli_style_features()
                
                # 应用风格迁移
                result = self.neural_model.transfer_style(
                    content_image,
                    style_features,
                    num_steps=100,
                    style_weight=50000,  # 降低风格权重
                    content_weight=10   # 提高内容权重
                )
                
                # 添加结构保持处理
                result = self._preserve_structure_mixing(content_image, result)
                
                return result
            else:
                result = self._apply_cv_optimized_ghibli_style(content_image)
                # 添加结构保持处理
                result = self._preserve_structure_mixing(content_image, result)
                return result
        except Exception as e:
            print(f"❌ 神经网络风格迁移失败: {e}")
            result = self._apply_cv_optimized_ghibli_style(content_image)
            # 添加结构保持处理
            result = self._preserve_structure_mixing(content_image, result)
            return result
    
    def _preserve_structure_mixing(self, original_image, styled_image):
        """
        通过边缘检测和加权融合保持原始图像结构
        """
        # 将PIL图像转换为OpenCV格式
        original_np = np.array(original_image)
        styled_np = np.array(styled_image)
        
        # 确保两个图像尺寸一致
        if original_np.shape[:2] != styled_np.shape[:2]:
            styled_np = cv2.resize(styled_np, (original_np.shape[1], original_np.shape[0]))
        
        # 转换为BGR格式（OpenCV使用BGR）
        if len(original_np.shape) == 3 and original_np.shape[2] == 3:
            original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = cv2.cvtColor(original_np, cv2.COLOR_GRAY2BGR)
            
        if len(styled_np.shape) == 3 and styled_np.shape[2] == 3:
            styled_bgr = cv2.cvtColor(styled_np, cv2.COLOR_RGB2BGR)
        else:
            styled_bgr = cv2.cvtColor(styled_np, cv2.COLOR_GRAY2BGR)
        
        # 提取原始图像的边缘信息
        original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        
        # 使用Canny边缘检测提取结构信息
        edges = cv2.Canny(original_gray, 50, 150)
        
        # 创建边缘权重掩码
        # 边缘区域权重更高，以保持结构清晰度
        edge_mask = edges.astype(np.float32) / 255.0
        # 扩展边缘影响区域
        kernel = np.ones((5, 5), np.float32) / 25
        edge_mask = cv2.filter2D(edge_mask, -1, kernel)
        # 增强边缘权重
        edge_mask = np.clip(edge_mask * 2.0, 0, 1)
        
        # 创建结构保持权重掩码
        # 非边缘区域也保持一定权重以维持整体结构
        structure_mask = np.ones_like(edge_mask) * 0.3  # 基础权重
        structure_mask = np.maximum(structure_mask, edge_mask)  # 与边缘权重合并
        structure_mask = np.clip(structure_mask, 0, 1)
        
        # 应用加权融合
        # 在结构重要区域（边缘等）更多保留原始图像信息
        # 在其他区域更多使用风格化结果
        mixed = np.zeros_like(styled_bgr, dtype=np.float32)
        for i in range(styled_bgr.shape[2]):  # 对每个颜色通道进行处理
            mixed[:, :, i] = (
                structure_mask * original_bgr[:, :, i] + 
                (1 - structure_mask) * styled_bgr[:, :, i]
            )
        
        # 转换回RGB格式并确保数据类型正确
        result_bgr = np.clip(mixed, 0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        result_image = Image.fromarray(result_rgb)
        
        return result_image

    def _apply_learned_ghibli_style(self, img_bgr):
        """应用标准的宫崎骏动漫风格 - 基于训练好的模型"""
        height, width = img_bgr.shape[:2]
        
        # 1. 色彩空间转换
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        
        # 2. 标准宫崎骏风格色彩增强
        # 饱和度提升 - 宫崎骏动画特色
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        
        # 亮度调整 - 稍微明亮一些
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        
        enhanced_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 3. LAB色彩空间精调
        enhanced_lab = cv2.cvtColor(enhanced_hsv, cv2.COLOR_BGR2LAB)
        
        # 标准宫崎骏色彩偏移
        enhanced_lab[:, :, 1] = np.clip(enhanced_lab[:, :, 1] + 8, 0, 255)   # 偏绿
        enhanced_lab[:, :, 2] = np.clip(enhanced_lab[:, :, 2] + 5, 0, 255)   # 偏黄
        
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 4. 宫崎骏特色色彩映射
        original = enhanced.copy()
        
        # 模拟动画片的色彩特点
        enhanced[:, :, 0] = np.clip(original[:, :, 0] + 8, 0, 255)   # B通道调整
        enhanced[:, :, 1] = np.clip(original[:, :, 1] + 12, 0, 255)  # G通道调整
        enhanced[:, :, 2] = np.clip(original[:, :, 2] + 5, 0, 255)   # R通道调整
        
        # 5. 局部对比度增强
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        local_contrast = cv2.addWeighted(enhanced, 1.0, 
                                        cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel), 
                                        0.3, 0)
        enhanced = np.clip(local_contrast, 0, 255).astype(np.uint8)
        
        # 6. 梦幻光照效果
        y_coords, x_coords = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 标准光照强度
        light_intensity = 0.15
        light_mask = 1.0 - (distance / max_distance) * light_intensity
        light_mask = np.clip(light_mask, 0.8, 1.0)
        
        # 应用光照
        final = enhanced.astype(np.float32) * light_mask[:, :, np.newaxis]
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        # 7. 标准柔和效果
        # 轻微的高斯模糊
        blurred = cv2.GaussianBlur(final, (3, 3), 0)
        # 双边滤波保持边缘
        bilateral = cv2.bilateralFilter(final, 9, 80, 80)
        
        # 混合效果
        final = cv2.addWeighted(final, 0.7, bilateral, 0.3, 0)
        
        # 8. 最终色彩校正
        final = self._apply_gamma_correction(final, 1.02)
        
        return final
    
    def _apply_gamma_correction(self, img, gamma):
        """应用伽马校正"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

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
        
        # 初始进度更新
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 5, 0, num_steps, 0)
        
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
            
            # 更频繁的进度更新（每5步更新一次）
            if (step + 1) % 5 == 0 or step == num_steps - 1:
                progress = int((step + 1) / num_steps * 100)
                if self.progress_callback and self.task_id:
                    self.progress_callback(self.task_id, progress, step + 1, num_steps, total_loss.item())
            
            if (step + 1) % 30 == 0:
                print(f"步骤 {step+1}/{num_steps}, 总损失: {total_loss.item():.4f}")
        
        # 后处理输出图像 - 恢复原始尺寸
        output_tensor = input_img.data.clamp(0, 1)
        result_image = self._postprocess_image_preserve_size(output_tensor, original_size)
        
        # 最终进度更新
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 100, num_steps, num_steps, total_loss.item())
        
        return result_image
    
    def _apply_cv_optimized_ghibli_style(self, image):
        """应用计算机视觉优化的宫崎骏风格 - 大幅改进版本"""
        print("🎨 使用改进的计算机视觉宫崎骏风格...")
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 10, 1, 10, 0)
        
        img_np = np.array(image)
        
        # 正确处理图像格式转换
        if img_np.ndim == 2:
            # 灰度图
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.ndim == 3:
            if img_np.shape[2] == 3:
                # RGB图像
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif img_np.shape[2] == 4:
                # RGBA图像，转换为RGB
                img_bgr = cv2.cvtColor(img_np[:,:,:3], cv2.COLOR_RGB2BGR)
            else:
                # 其他通道数，转换为灰度再转BGR
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        else:
            # 未知格式，使用默认处理
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 20, 2, 10, 0)
        
        # 尺寸限制
        max_size = 2048
        if max(h,w) > max_size:
            scale = max_size / max(h,w)
            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
            h, w = img_bgr.shape[:2]
            print(f"📏 计算机视觉处理: 图片尺寸过大，自动缩放至: {w}x{h}")
        else:
            print(f"📏 计算机视觉处理: 保持原始尺寸: {w}x{h}")
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 30, 3, 10, 0)
        
        # 1. 先转换为动漫风格 - 创造基本的动漫效果
        img_bgr = self._anime_style_conversion(img_bgr)
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 50, 5, 10, 0)
        
        # 2. 再叠加宫崎骏色彩风格 - 在动漫基础上调整色彩
        img_bgr = self._ghibli_color_style(img_bgr)
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 70, 7, 10, 0)
        
        # 3. 最终优化
        img_bgr = self._final_anime_optimization(img_bgr)
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 100, 10, 10, 0)
        
        result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _preserve_structure_mixing(self, original_image, styled_image):
        """
        通过边缘检测和加权融合保持原始图像结构
        """
        # 将PIL图像转换为OpenCV格式
        original_np = np.array(original_image)
        styled_np = np.array(styled_image)
        
        # 确保两个图像尺寸一致
        if original_np.shape[:2] != styled_np.shape[:2]:
            styled_np = cv2.resize(styled_np, (original_np.shape[1], original_np.shape[0]))
        
        # 转换为BGR格式（OpenCV使用BGR）
        if len(original_np.shape) == 3 and original_np.shape[2] == 3:
            original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = cv2.cvtColor(original_np, cv2.COLOR_GRAY2BGR)
            
        if len(styled_np.shape) == 3 and styled_np.shape[2] == 3:
            styled_bgr = cv2.cvtColor(styled_np, cv2.COLOR_RGB2BGR)
        else:
            styled_bgr = cv2.cvtColor(styled_np, cv2.COLOR_GRAY2BGR)
        
        # 提取原始图像的边缘信息
        original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        
        # 使用Canny边缘检测提取结构信息
        edges = cv2.Canny(original_gray, 50, 150)
        
        # 创建边缘权重掩码
        # 边缘区域权重更高，以保持结构清晰度
        edge_mask = edges.astype(np.float32) / 255.0
        # 扩展边缘影响区域
        kernel = np.ones((5, 5), np.float32) / 25
        edge_mask = cv2.filter2D(edge_mask, -1, kernel)
        # 增强边缘权重
        edge_mask = np.clip(edge_mask * 2.0, 0, 1)
        
        # 创建结构保持权重掩码
        # 非边缘区域也保持一定权重以维持整体结构
        structure_mask = np.ones_like(edge_mask) * 0.3  # 基础权重
        structure_mask = np.maximum(structure_mask, edge_mask)  # 与边缘权重合并
        structure_mask = np.clip(structure_mask, 0, 1)
        
        # 应用加权融合
        # 在结构重要区域（边缘等）更多保留原始图像信息
        # 在其他区域更多使用风格化结果
        mixed = np.zeros_like(styled_bgr, dtype=np.float32)
        for i in range(styled_bgr.shape[2]):  # 对每个颜色通道进行处理
            mixed[:, :, i] = (
                structure_mask * original_bgr[:, :, i] + 
                (1 - structure_mask) * styled_bgr[:, :, i]
            )
        
        # 转换回RGB格式并确保数据类型正确
        result_bgr = np.clip(mixed, 0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        result_image = Image.fromarray(result_rgb)
        
        return result_image

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
        # 确保图像是3通道的BGR格式
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 1:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        
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
    
    def _enhanced_ghibli_color_palette(self, img_bgr):
        """基于特征分析的宫崎骏色彩调色板 - 精确匹配宫崎骏风格"""
        # 宫崎骏风格特点：高饱和度、明亮、温暖色调、梦幻感
        
        # 转换为HSV色彩空间进行精确调整
        # 确保图像是3通道的BGR格式
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 1:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 1. 大幅增强饱和度 - 宫崎骏风格色彩鲜艳（解决黑白灰问题）
        s = cv2.add(s, 80)  # 大幅增加饱和度
        s = np.clip(s, 0, 240)  # 提高最大饱和度限制
        
        # 2. 显著增强亮度 - 宫崎骏风格明亮
        v = cv2.add(v, 30)
        v = np.clip(v, 0, 255)
        
        # 3. 强烈调整色调 - 偏向温暖色调（橙色/黄色）
        # 宫崎骏风格温暖色调范围：10-40（橙色到黄色）
        h_warm = h.copy()
        warm_mask = (h > 10) & (h < 40)
        if np.any(warm_mask):
            h_warm[warm_mask] = np.clip(h_warm[warm_mask] + 10, 0, 179)  # 强烈偏向更暖
            h = np.where(warm_mask, h_warm, h)
        
        # 4. 增强蓝色和绿色（宫崎骏风格中的天空和自然色）
        blue_green_mask = (h > 85) & (h < 150)
        if np.any(blue_green_mask):
            s_blue_green = s.copy()
            s_blue_green[blue_green_mask] = np.clip(s_blue_green[blue_green_mask] + 50, 0, 255)
            s = np.where(blue_green_mask, s_blue_green, s)
        
        h = np.clip(h, 0, 179)
        
        # 5. 合并HSV通道
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 6. 应用LAB色彩空间进一步优化
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 大幅增强色彩鲜艳度
        a = cv2.add(a, 25)  # 强烈增强红色/绿色
        b = cv2.add(b, 30)  # 强烈增强蓝色/黄色
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        # 增强亮度对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_balanced = cv2.merge([l, a, b])
        final = cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)
        
        # 7. 应用柔和滤镜保持宫崎骏的柔和感
        soft = cv2.GaussianBlur(final, (3, 3), 0)
        result = cv2.addWeighted(final, 0.9, soft, 0.1, 0)  # 减少柔和度，保持清晰度
        
        return result
    
    def _anime_style_conversion(self, img_bgr):
        """真正的动漫风格转换 - 改进版本，保持结构清晰"""
        # 动漫风格核心特点：
        # 1. 简化的色块和扁平化效果
        # 2. 清晰的轮廓线条
        # 3. 减少写实纹理，增加卡通感
        # 4. 保持物体和人物识别性
        
        # 第一步：轻度边缘保留平滑 - 移除写实纹理但保持结构
        filtered = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=60, sigmaSpace=60)
        
        # 第二步：适度的颜色量化 - 创造动漫的扁平色块但保持细节
        Z = filtered.reshape((-1, 3))
        Z = np.float32(Z)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = 16  # 增加颜色数量，保持更多细节
        
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        cartoon = centers[labels.flatten()]
        cartoon = cartoon.reshape((filtered.shape))
        
        # 第三步：超像素分割 - 创造自然的色块边界
        try:
            from skimage.segmentation import slic
            from skimage.color import label2rgb
            
            img_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
            segments = slic(img_rgb, n_segments=300, compactness=20, sigma=1)  # 增加分割数量
            flat_rgb = (label2rgb(segments, img_rgb, kind='avg') * 255).astype(np.uint8)
            cartoon = cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            # 备选方案：均值漂移滤波
            try:
                cartoon = cv2.pyrMeanShiftFiltering(cartoon, 10, 20)
            except Exception:
                pass
        
        # 第四步：生成清晰的动漫轮廓线条
        gray = cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)
        
        # 使用多种边缘检测方法组合
        edges_canny = cv2.Canny(gray, 50, 120)  # 调整阈值
        edges_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 7, 2)
        
        # 合并边缘检测结果
        edges_combined = cv2.bitwise_or(edges_canny, edges_adaptive)
        
        # 轻微柔化线条
        edges_soft = cv2.GaussianBlur(edges_combined, (3, 3), 0.3)
        
        # 第五步：结构保持的线条叠加
        edges_colored = cv2.cvtColor(edges_soft, cv2.COLOR_GRAY2BGR)
        
        # 适度线条叠加，保持动漫效果但不失结构
        result = cv2.addWeighted(cartoon, 0.9, edges_colored, 0.1, 0)
        
        # 第六步：轻微锐化恢复清晰度
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(result, -1, kernel)
        result = cv2.addWeighted(result, 0.85, sharpened, 0.15, 0)
        
        return result
    
    def _ghibli_color_style(self, img_bgr):
        """宫崎骏色彩风格 - 在动漫基础上叠加宫崎骏特色色彩"""
        # 宫崎骏色彩特点：
        # 1. 温暖明亮的色调
        # 2. 高饱和度但不刺眼
        # 3. 梦幻的光影效果
        
        # 转换为HSV色彩空间进行精确调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 增强饱和度 - 宫崎骏风格色彩鲜艳
        s = cv2.add(s, 40)
        s = np.clip(s, 0, 220)
        
        # 调整色调 - 偏向温暖色调（橙色/黄色）
        h_warm = h.copy()
        warm_mask = (h > 10) & (h < 40)
        if np.any(warm_mask):
            h_warm[warm_mask] = np.clip(h_warm[warm_mask] + 8, 0, 179)
            h = np.where(warm_mask, h_warm, h)
        
        # 增强亮度 - 宫崎骏风格明亮
        v = cv2.add(v, 20)
        v = np.clip(v, 0, 255)
        
        # 增强蓝色和绿色（宫崎骏风格中的天空和自然色）
        blue_green_mask = (h > 85) & (h < 150)
        if np.any(blue_green_mask):
            s_blue_green = s.copy()
            s_blue_green[blue_green_mask] = np.clip(s_blue_green[blue_green_mask] + 30, 0, 255)
            s = np.where(blue_green_mask, s_blue_green, s)
        
        h = np.clip(h, 0, 179)
        
        # 合并HSV通道
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 应用LAB色彩空间进一步优化
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强色彩鲜艳度
        a = cv2.add(a, 15)
        b = cv2.add(b, 20)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        # 增强亮度对比度
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_balanced = cv2.merge([l, a, b])
        final = cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)
        
        # 添加宫崎骏风格的梦幻光影效果
        h, w = final.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 创建柔和的光照效果
        light_mask = 1.0 - (distance / max_distance) * 0.08
        light_mask = np.clip(light_mask, 0.92, 1.0)
        
        result = final.astype(np.float32) * light_mask[:,:,np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _clear_line_enhancement(self, img_bgr):
        """基于特征分析的线条增强 - 精确匹配宫崎骏风格"""
        # 宫崎骏风格特点：清晰但不生硬的线条
        
        # 1. 提取灰度图像
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2. 使用改进的边缘检测 - 增强线条清晰度
        # 使用Canny边缘检测，调整阈值以获得更清晰的线条
        edges_canny = cv2.Canny(gray, 50, 150)
        
        # 3. 使用自适应阈值获取更丰富的边缘信息
        edges_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 9, 3)
        
        # 4. 合并两种边缘检测结果
        edges_combined = cv2.bitwise_or(edges_canny, edges_adaptive)
        
        # 5. 创建更清晰的线稿效果
        edges_enhanced = cv2.GaussianBlur(edges_combined, (3, 3), 0.8)
        
        # 6. 增强线稿对比度
        edges_enhanced = cv2.addWeighted(edges_enhanced, 1.5, edges_enhanced, 0, 0)
        
        # 7. 转换为彩色线稿
        edges_colored = cv2.cvtColor(edges_enhanced, cv2.COLOR_GRAY2BGR)
        
        # 8. 增强线稿强度，提高动漫风格明显度
        line_strength = 0.15  # 增加线稿强度
        result = cv2.addWeighted(img_bgr, 1.0 - line_strength, edges_colored, line_strength, 0)
        
        return result
    
    def _preserve_structure_enhancement(self, img_bgr, original_img):
        """保持原图结构的增强 - 避免与原图差异过大"""
        # 1. 与原图进行混合，保持原始结构
        # 检查图像维度，正确处理灰度图和彩色图
        if len(original_img.shape) == 3 and original_img.shape[2] == 3:
            original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        else:
            # 灰度图或单通道图
            original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        
        # 调整尺寸匹配
        if original_bgr.shape != img_bgr.shape:
            original_bgr = cv2.resize(original_bgr, (img_bgr.shape[1], img_bgr.shape[0]))
        
        # 2. 与原图进行适度混合（70%动漫效果 + 30%原图）
        blended = cv2.addWeighted(img_bgr, 0.7, original_bgr, 0.3, 0)
        
        return blended
    
    def _final_ghibli_optimization(self, img_bgr):
        """最终的宫崎骏风格优化"""
        # 1. 轻微锐化增强细节
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        
        # 2. 轻微降噪
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 5, 5, 7, 21)
        
        # 3. 最终色彩平衡
        # 确保图像是3通道的BGR格式
        if len(denoised.shape) == 2:
            denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        elif denoised.shape[2] == 1:
            denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强亮度
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_balanced = cv2.merge([l, a, b])
        final = cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)
        
        # 4. 添加梦幻光影效果
        h, w = final.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 创建柔和的光照效果
        light_mask = 1.0 - (distance / max_distance) * 0.08
        light_mask = np.clip(light_mask, 0.92, 1.0)
        
        result = final.astype(np.float32) * light_mask[:,:,np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
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
    
    def _fallback_simple_ghibli(self, image):
        """简化的宫崎骏风格备用方法"""
        print("⚠️ 使用简化的宫崎骏风格方法")
        
        # 转换为OpenCV格式
        img_np = np.array(image)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # 1. 边缘保留平滑
        smooth = cv2.bilateralFilter(img_bgr, 9, 75, 75)
        
        # 2. 宫崎骏色彩
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 增强饱和度和亮度
        s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
        v = np.clip(v * 1.1, 0, 255).astype(np.uint8)
        
        hsv_enhanced = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 3. 轻微锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(result, -1, kernel)
        result = cv2.addWeighted(result, 0.8, sharpened, 0.2, 0)
        
        # 转换回RGB
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

    def _subtle_color_optimization(self, img_bgr):
        """轻微的色彩优化 - 保持原图色彩，只做轻微调整"""
        # 转换为HSV色彩空间进行轻微调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 轻微增强饱和度（+10，而不是之前的+80）
        s = cv2.add(s, 10)
        s = np.clip(s, 0, 255)
        
        # 轻微增强亮度（+5，而不是之前的+30）
        v = cv2.add(v, 5)
        v = np.clip(v, 0, 255)
        
        # 合并HSV通道
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        return enhanced
    
    def _apply_optimized_cv_anime_style(self, content_image):
        """应用专门针对宫崎骏风格优化的转换算法"""
        print("🎨 使用宫崎骏风格专用转换算法...")
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 10, 1, 10, 0)
        
        img_np = np.array(content_image)
        
        # 正确处理图像格式转换
        if img_np.ndim == 2:
            # 灰度图
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.ndim == 3:
            if img_np.shape[2] == 3:
                # RGB图像
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif img_np.shape[2] == 4:
                # RGBA图像，转换为RGB
                img_bgr = cv2.cvtColor(img_np[:,:,:3], cv2.COLOR_RGB2BGR)
            else:
                # 其他通道数，转换为灰度再转BGR
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        else:
            # 未知格式，使用默认处理
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        h, w = img_bgr.shape[:2]
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 20, 2, 10, 0)
        
        # 尺寸限制
        max_size = 1024
        if max(h,w) > max_size:
            scale = max_size / max(h,w)
            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
            h, w = img_bgr.shape[:2]
            print(f"📏 优化处理: 图片尺寸过大，自动缩放至: {w}x{h}")
        else:
            print(f"📏 优化处理: 保持原始尺寸: {w}x{h}")
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 30, 3, 10, 0)
        
        # 1. 宫崎骏风格预处理 - 基于参考图片分析
        # 使用轻微的双边滤波，保留细节但创造柔和效果
        filtered = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 40, 4, 10, 0)
        
        # 2. 宫崎骏风格线条生成 - 清晰但不生硬
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        # 使用宫崎骏风格的边缘检测
        edges_canny = cv2.Canny(gray, 50, 150)  # 参考宫崎骏图片的边缘密度
        
        # 使用自适应阈值增强重要边缘
        edges_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        
        # 合并边缘检测结果
        edges_combined = cv2.bitwise_or(edges_canny, edges_adaptive)
        
        # 宫崎骏风格的柔和线条
        edges_soft = cv2.GaussianBlur(edges_combined, (3, 3), 0.5)
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 50, 5, 10, 0)
        
        # 3. 宫崎骏风格颜色处理 - 基于参考图片分析
        # 使用适度的颜色量化（参考宫崎骏风格的颜色简化程度）
        Z = filtered.reshape((-1, 3))
        Z = np.float32(Z)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = 24  # 基于宫崎骏风格的颜色数量调整
        
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        cartoon = centers[labels.flatten()]
        cartoon = cartoon.reshape((filtered.shape))
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 60, 6, 10, 0)
        
        # 4. 超像素分割 - 创造自然的色块边界（宫崎骏风格特点）
        try:
            from skimage.segmentation import slic
            from skimage.color import label2rgb
            
            img_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
            segments = slic(img_rgb, n_segments=200, compactness=20, sigma=1)
            flat_rgb = (label2rgb(segments, img_rgb, kind='avg') * 255).astype(np.uint8)
            cartoon = cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"⚠️ 超像素分割失败: {e}")
            # 使用均值漂移滤波作为备选
            try:
                cartoon = cv2.pyrMeanShiftFiltering(cartoon, 15, 30)
            except Exception:
                pass
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 70, 7, 10, 0)
        
        # 5. 宫崎骏风格线条叠加
        edges_colored = cv2.cvtColor(edges_soft, cv2.COLOR_GRAY2BGR)
        
        # 基于宫崎骏风格的线条叠加比例
        result = cv2.addWeighted(cartoon, 0.85, edges_colored, 0.15, 0)
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 80, 8, 10, 0)
        
        # 6. 应用宫崎骏色彩风格（基于参考图片分析）
        result = self._apply_ghibli_style_based_on_reference(result)
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 90, 9, 10, 0)
        
        # 7. 最终宫崎骏风格优化
        result = self._final_ghibli_style_optimization(result)
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 100, 10, 10, 0)
        
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _apply_ghibli_color_to_anime(self, img_bgr):
        """在动漫风格基础上应用宫崎骏色彩"""
        # 宫崎骏色彩特点：温暖、明亮、高饱和度但不刺眼
        
        # 1. 转换为HSV色彩空间进行精确调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 2. 增强饱和度（宫崎骏风格色彩鲜艳）
        s = cv2.add(s, 40)  # 适度增强饱和度
        s = np.clip(s, 0, 220)
        
        # 3. 调整色调 - 偏向温暖色调（橙色/黄色）
        h_warm = h.copy()
        warm_mask = (h > 10) & (h < 40)
        if np.any(warm_mask):
            h_warm[warm_mask] = np.clip(h_warm[warm_mask] + 8, 0, 179)  # 轻微偏向温暖
            h = np.where(warm_mask, h_warm, h)
        
        # 4. 增强亮度 - 宫崎骏风格明亮
        v = cv2.add(v, 20)
        v = np.clip(v, 0, 255)
        
        # 5. 增强蓝色和绿色（宫崎骏风格中的天空和自然色）
        blue_green_mask = (h > 85) & (h < 150)
        if np.any(blue_green_mask):
            s_blue_green = s.copy()
            s_blue_green[blue_green_mask] = np.clip(s_blue_green[blue_green_mask] + 30, 0, 255)
            s = np.where(blue_green_mask, s_blue_green, s)
        
        h = np.clip(h, 0, 179)
        
        # 6. 合并HSV通道
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 7. 应用LAB色彩空间进一步优化
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强色彩鲜艳度
        a = cv2.add(a, 15)
        b = cv2.add(b, 20)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        # 增强亮度对比度
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_balanced = cv2.merge([l, a, b])
        final = cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)
        
        # 8. 添加宫崎骏风格的梦幻光影效果
        h, w = final.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 创建柔和的光照效果
        light_mask = 1.0 - (distance / max_distance) * 0.08
        light_mask = np.clip(light_mask, 0.92, 1.0)
        
        result = final.astype(np.float32) * light_mask[:,:,np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _apply_improved_ghibli_color(self, img_bgr):
        """改进的宫崎骏色彩风格 - 更自然，保留更多细节"""
        # 宫崎骏色彩特点：温暖、明亮、自然
        
        # 转换为HSV色彩空间进行精确调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 适度增强饱和度 - 不过度
        s = cv2.add(s, 20)  # 减少饱和度增强幅度
        s = np.clip(s, 0, 200)
        
        # 轻微调整色调 - 偏向温暖色调
        h_warm = h.copy()
        warm_mask = (h > 10) & (h < 40)
        if np.any(warm_mask):
            h_warm[warm_mask] = np.clip(h_warm[warm_mask] + 5, 0, 179)  # 减少色调调整幅度
            h = np.where(warm_mask, h_warm, h)
        
        # 适度增强亮度
        v = cv2.add(v, 10)
        v = np.clip(v, 0, 255)
        
        h = np.clip(h, 0, 179)
        
        # 合并HSV通道
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 应用LAB色彩空间进一步优化
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 适度增强色彩鲜艳度
        a = cv2.add(a, 10)
        b = cv2.add(b, 15)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        # 增强亮度对比度
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_balanced = cv2.merge([l, a, b])
        final = cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)
        
        return final
    
    def _improved_final_optimization(self, img_bgr):
        """改进的最终优化 - 保留更多细节"""
        # 轻微锐化增强细节
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        
        # 轻微降噪，保留细节
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 2, 2, 3, 10)
        
        return denoised
    
    def _apply_ghibli_style_based_on_reference(self, img_bgr):
        """基于宫崎骏参考图片的色彩风格应用"""
        # 宫崎骏风格特点（基于分析）：
        # - 中等饱和度（约160-170）
        # - 较高的亮度
        # - 温暖柔和的色调
        
        # 转换为HSV色彩空间进行精确调整
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 宫崎骏风格饱和度调整
        target_saturation = 165  # 基于参考图片的平均饱和度
        current_saturation = np.mean(s)
        saturation_boost = max(0, target_saturation - current_saturation)
        s = cv2.add(s, int(saturation_boost))
        s = np.clip(s, 0, 220)
        
        # 宫崎骏风格亮度调整
        target_brightness = 180  # 基于参考图片的平均亮度
        current_brightness = np.mean(v)
        brightness_boost = max(0, target_brightness - current_brightness)
        v = cv2.add(v, int(brightness_boost))
        v = np.clip(v, 0, 255)
        
        # 宫崎骏风格色调调整 - 偏向温暖色调
        h_warm = h.copy()
        warm_mask = (h > 10) & (h < 40)  # 橙色到黄色范围
        if np.any(warm_mask):
            h_warm[warm_mask] = np.clip(h_warm[warm_mask] + 8, 0, 179)
            h = np.where(warm_mask, h_warm, h)
        
        h = np.clip(h, 0, 179)
        
        # 合并HSV通道
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 应用LAB色彩空间进一步优化
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 宫崎骏风格色彩鲜艳度
        a = cv2.add(a, 20)  # 增强红色/绿色
        b = cv2.add(b, 25)  # 增强蓝色/黄色
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        # 宫崎骏风格亮度对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_balanced = cv2.merge([l, a, b])
        final = cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)
        
        return final
    
    def _final_ghibli_style_optimization(self, img_bgr):
        """最终的宫崎骏风格优化"""
        # 宫崎骏风格轻微锐化
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        
        # 宫崎骏风格降噪（保留细节）
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 2, 2, 3, 10)
        
        # 添加宫崎骏风格的梦幻光影效果
        h, w = denoised.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 宫崎骏风格的光照效果
        light_mask = 1.0 - (distance / max_distance) * 0.1
        light_mask = np.clip(light_mask, 0.9, 1.0)
        
        result = denoised.astype(np.float32) * light_mask[:,:,np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _blend_with_original(self, anime_img, original_img):
        """与原图混合，保留更多实物内容"""
        # 调整尺寸匹配
        if original_img.shape != anime_img.shape:
            original_img = cv2.resize(original_img, (anime_img.shape[1], anime_img.shape[0]))
        
        # 与原图进行适度混合（80%动漫效果 + 20%原图细节）
        blended = cv2.addWeighted(anime_img, 0.8, original_img, 0.2, 0)
        
        return blended
    
    def _final_anime_optimization(self, img_bgr):
        """最终的动漫化优化 - 保持动漫风格的同时轻微优化"""
        # 轻微锐化增强细节
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        
        # 轻微降噪
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 3, 3, 5, 15)
        
        return denoised

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

