#!/usr/bin/env python3
"""
增强版宫崎骏风格转换系统
集成人脸检测、语义分割、背景处理等专业功能
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import os
import time
import threading
from typing import Optional
from .image_processor_interface import ImageProcessorInterface, ProcessingResult, ProcessingStyle

# 导入高级处理器和AnimeGAN
from .advanced_anime_processor import advanced_processor
from .anime_gan_processor import convert_with_anime_gan

class GhibliEnhancedTransfer(ImageProcessorInterface):
    """增强版宫崎骏风格转换系统
    集成人脸检测、语义分割、背景处理等专业功能
    """
    
    def __init__(self, use_face_detection=True, use_background_separation=True):
        super().__init__(ProcessingStyle.GHIBLI_ENHANCED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_face_detection = use_face_detection
        self.use_background_separation = use_background_separation
        
        # 加载模型
        self.face_detector = None
        self.seg_model = None
        self.face_landmark_model = None
        
        # 初始化模型
        self._initialize_models()
        
        # 进度回调
        self.progress_callback = None
        self.task_id = None
    
    def process(self, image: Image.Image, **kwargs) -> ProcessingResult:
        """
        处理图像，应用增强版宫崎骏风格
        
        Args:
            image: 输入图像
            **kwargs: 其他处理参数
                - use_face_enhancement: 是否使用人脸增强
                - use_bg_separation: 是否使用背景分离
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            use_face_enhancement = kwargs.get('use_face_enhancement', True)
            use_bg_separation = kwargs.get('use_bg_separation', True)
            
            result_image = self.apply_enhanced_ghibli_style(
                image,
                use_face_enhancement=use_face_enhancement,
                use_bg_separation=use_bg_separation
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
            "processor_type": "GhibliEnhancedTransfer",
            "style_type": self.style_type.value,
            "use_face_detection": self.use_face_detection,
            "use_background_separation": self.use_background_separation,
            "device": str(self.device)
        }
    
    def _initialize_models(self):
        """初始化所有需要的模型"""
        print("🔧 初始化增强版宫崎骏风格转换模型...")
        
        # 1. 人脸检测模型
        if self.use_face_detection:
            self._initialize_face_detector()
        
        # 2. 语义分割模型
        if self.use_background_separation:
            self._initialize_segmentation_model()
        
        print("✅ 模型初始化完成")
    
    def _initialize_face_detector(self):
        """初始化人脸检测器"""
        try:
            # 使用OpenCV的DNN模块加载人脸检测模型
            model_path = "models/face_detector/opencv_face_detector_uint8.pb"
            config_path = "models/face_detector/opencv_face_detector.pbtxt"
            
            # 如果模型文件不存在，使用内置的Haar级联分类器
            if os.path.exists(model_path) and os.path.exists(config_path):
                self.face_detector = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                print("✅ 加载DNN人脸检测模型成功")
            else:
                # 使用OpenCV内置的Haar级联分类器
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.face_detector = cv2.CascadeClassifier(cascade_path)
                print("✅ 加载Haar级联人脸检测器成功")
                
        except Exception as e:
            print(f"⚠️ 人脸检测器初始化失败: {e}")
            self.face_detector = None
    
    def _initialize_segmentation_model(self):
        """初始化语义分割模型"""
        try:
            # 使用预训练的DeepLabV3模型进行语义分割
            self.seg_model = models.segmentation.deeplabv3_resnet50(
                weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
            ).eval().to(self.device)
            print("✅ 加载语义分割模型成功")
        except Exception as e:
            print(f"⚠️ 语义分割模型初始化失败: {e}")
            self.seg_model = None
    
    def detect_faces(self, image):
        """检测图像中的人脸"""
        if not self.use_face_detection or self.face_detector is None:
            return []
        
        try:
            # 转换为灰度图
            if isinstance(image, Image.Image):
                img_np = np.array(image)
                if len(img_np.shape) == 3:
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_np
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            if isinstance(self.face_detector, cv2.dnn_Net):
                # DNN模型检测
                blob = cv2.dnn.blobFromImage(gray, 1.0, (300, 300), [104, 117, 123])
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()
                
                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:  # 置信度阈值
                        x1 = int(detections[0, 0, i, 3] * gray.shape[1])
                        y1 = int(detections[0, 0, i, 4] * gray.shape[0])
                        x2 = int(detections[0, 0, i, 5] * gray.shape[1])
                        y2 = int(detections[0, 0, i, 6] * gray.shape[0])
                        faces.append((x1, y1, x2-x1, y2-y1))
            else:
                # Haar级联检测
                faces = self.face_detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
            
            print(f"👤 检测到 {len(faces)} 张人脸")
            return faces
            
        except Exception as e:
            print(f"❌ 人脸检测失败: {e}")
            return []
    
    def segment_person(self, image):
        """语义分割，提取人物前景"""
        if not self.use_background_separation or self.seg_model is None:
            # 如果没有分割模型，返回全图掩码
            if isinstance(image, Image.Image):
                img_np = np.array(image)
                mask = np.ones(img_np.shape[:2], dtype=np.uint8) * 255
            else:
                mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            return mask
        
        try:
            # 预处理图像
            if isinstance(image, Image.Image):
                img_np = np.array(image)
                rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                rgb = image
            
            # 调整尺寸
            h, w = rgb.shape[:2]
            max_size = 520
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                rgb_resized = cv2.resize(rgb, (new_w, new_h))
            else:
                rgb_resized = rgb
            
            # 转换为tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(rgb_resized).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                output = self.seg_model(input_tensor)['out'][0]
            
            # 获取人物类别（COCO数据集中人物类别为15）
            person_class = 15
            mask = (output.argmax(0) == person_class).cpu().numpy().astype(np.uint8) * 255
            
            # 调整到原图尺寸
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 平滑边缘
            mask = cv2.GaussianBlur(mask, (9, 9), 0)
            
            print("🎯 语义分割完成")
            return mask
            
        except Exception as e:
            print(f"❌ 语义分割失败: {e}")
            # 返回全图掩码作为回退
            if isinstance(image, Image.Image):
                img_np = np.array(image)
                mask = np.ones(img_np.shape[:2], dtype=np.uint8) * 255
            else:
                mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            return mask
    
    def enhance_faces(self, image, face_regions):
        """增强人脸区域 - 宫崎骏风格美化"""
        if not face_regions:
            return image
        
        # 转换为numpy数组
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            if len(img_np.shape) == 3:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = image.copy()
        
        enhanced_image = img_bgr.copy()
        
        for (x, y, w, h) in face_regions:
            # 提取人脸区域
            face_roi = img_bgr[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue
            
            # 应用宫崎骏风格的人脸美化
            enhanced_face = self._ghibli_face_enhancement(face_roi)
            
            # 将增强后的人脸放回原图
            enhanced_image[y:y+h, x:x+w] = enhanced_face
        
        return enhanced_image
    
    def _ghibli_face_enhancement(self, face_region):
        """宫崎骏风格的人脸美化"""
        # 1. 皮肤平滑
        smoothed = cv2.bilateralFilter(face_region, 9, 75, 75)
        
        # 2. 眼睛增强
        enhanced_eyes = self._enhance_eyes(smoothed)
        
        # 3. 嘴唇增强
        enhanced_lips = self._enhance_lips(enhanced_eyes)
        
        # 4. 色彩调整
        final_face = self._adjust_face_colors(enhanced_lips)
        
        return final_face
    
    def _enhance_eyes(self, face_region):
        """增强眼睛区域"""
        h, w = face_region.shape[:2]
        
        # 定义眼睛区域（相对位置）
        eye_top = int(h * 0.25)
        eye_bottom = int(h * 0.45)
        eye_left = int(w * 0.25)
        eye_right = int(w * 0.75)
        
        # 增强眼睛区域亮度
        eye_region = face_region[eye_top:eye_bottom, eye_left:eye_right]
        
        if eye_region.size > 0:
            # 转换为LAB色彩空间
            lab = cv2.cvtColor(eye_region, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 增强亮度
            l = cv2.add(l, 20)
            l = np.clip(l, 0, 255)
            
            # 合并通道
            lab_enhanced = cv2.merge([l, a, b])
            enhanced_eyes = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # 将增强后的眼睛区域放回
            face_region[eye_top:eye_bottom, eye_left:eye_right] = enhanced_eyes
        
        return face_region
    
    def _enhance_lips(self, face_region):
        """增强嘴唇区域"""
        h, w = face_region.shape[:2]
        
        # 定义嘴唇区域（相对位置）
        lip_top = int(h * 0.6)
        lip_bottom = int(h * 0.75)
        lip_left = int(w * 0.35)
        lip_right = int(w * 0.65)
        
        # 增强嘴唇区域饱和度
        lip_region = face_region[lip_top:lip_bottom, lip_left:lip_right]
        
        if lip_region.size > 0:
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(lip_region, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 增强饱和度
            s = cv2.add(s, 30)
            s = np.clip(s, 0, 255)
            
            # 合并通道
            hsv_enhanced = cv2.merge([h, s, v])
            enhanced_lips = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
            
            # 将增强后的嘴唇区域放回
            face_region[lip_top:lip_bottom, lip_left:lip_right] = enhanced_lips
        
        return face_region
    
    def _adjust_face_colors(self, face_region):
        """调整面部色彩 - 宫崎骏风格"""
        # 转换为LAB色彩空间进行精确调整
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 调整亮度（宫崎骏风格皮肤明亮）
        l = cv2.add(l, 10)
        l = np.clip(l, 0, 255)
        
        # 调整色彩平衡（偏向温暖色调）
        a = cv2.add(a, 5)  # 偏向红色
        b = cv2.add(b, 3)  # 偏向黄色
        
        # 合并通道
        lab_balanced = cv2.merge([l, a, b])
        final_face = cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)
        
        return final_face
    
    def process_background(self, image, person_mask, style_template="classic"):
        """处理背景 - 宫崎骏风格背景优化"""
        if not self.use_background_separation:
            return image
        
        # 转换为numpy数组
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            if len(img_np.shape) == 3:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = image.copy()
        
        # 分离前景和背景
        foreground = img_bgr.copy()
        background = img_bgr.copy()
        
        # 创建背景掩码
        bg_mask = cv2.bitwise_not(person_mask)
        
        # 对背景应用宫崎骏风格处理
        if style_template == "fantasy":
            bg_processed = self._fantasy_ghibli_style(background, intensity=0.6)
        elif style_template == "nostalgic":
            bg_processed = self._nostalgic_ghibli_style(background, intensity=0.7)
        elif style_template == "vibrant":
            bg_processed = self._vibrant_ghibli_style(background, intensity=0.8)
        else:
            bg_processed = self._classic_ghibli_style(background, intensity=0.5)
        
        # 合并前景和背景
        result = np.zeros_like(img_bgr)
        
        # 使用掩码混合前景和背景
        for i in range(3):  # 对每个通道进行处理
            result[:,:,i] = (
                foreground[:,:,i] * (person_mask / 255.0) + 
                bg_processed[:,:,i] * (bg_mask / 255.0)
            ).astype(np.uint8)
        
        return result
    
    def _classic_ghibli_style(self, image, intensity=0.5):
        """经典宫崎骏风格背景"""
        # 柔和色彩，梦幻光影
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 增强饱和度
        s = cv2.add(s, int(30 * intensity))
        s = np.clip(s, 0, 255)
        
        # 调整色调偏向温暖
        h = cv2.add(h, int(5 * intensity))
        h = np.clip(h, 0, 179)
        
        # 应用柔和滤镜
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 添加梦幻光影
        final = self._add_dreamy_lighting(enhanced, intensity)
        
        return final
    
    def _fantasy_ghibli_style(self, image, intensity=0.6):
        """梦幻宫崎骏风格背景"""
        # 更强烈的色彩和光影效果
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强色彩鲜艳度
        a = cv2.add(a, int(40 * intensity))
        b = cv2.add(b, int(30 * intensity))
        
        # 增强亮度对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 更强的梦幻光影
        final = self._add_dreamy_lighting(enhanced, intensity * 1.2)
        
        return final
    
    def _nostalgic_ghibli_style(self, image, intensity=0.7):
        """怀旧宫崎骏风格背景"""
        # 柔和的怀旧色调
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 降低饱和度，创造怀旧感
        s = cv2.subtract(s, int(20 * intensity))
        s = np.clip(s, 0, 255)
        
        # 调整色调偏向暖黄
        h = cv2.add(h, int(10 * intensity))
        h = np.clip(h, 0, 179)
        
        hsv_nostalgic = cv2.merge([h, s, v])
        nostalgic = cv2.cvtColor(hsv_nostalgic, cv2.COLOR_HSV2BGR)
        
        # 添加轻微的胶片颗粒效果
        noise = np.random.normal(0, 3 * intensity, nostalgic.shape).astype(np.uint8)
        final = cv2.add(nostalgic, noise)
        
        return final
    
    def _vibrant_ghibli_style(self, image, intensity=0.8):
        """鲜艳宫崎骏风格背景"""
        # 高饱和度鲜艳色彩
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 大幅增强饱和度
        s = cv2.add(s, int(50 * intensity))
        s = np.clip(s, 0, 255)
        
        # 增强亮度
        v = cv2.add(v, int(20 * intensity))
        v = np.clip(v, 0, 255)
        
        hsv_vibrant = cv2.merge([h, s, v])
        vibrant = cv2.cvtColor(hsv_vibrant, cv2.COLOR_HSV2BGR)
        
        # 锐化增强细节
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        final = cv2.filter2D(vibrant, -1, kernel)
        
        return final
    
    def _add_dreamy_lighting(self, image, intensity=1.0):
        """添加梦幻光影效果"""
        h, w = image.shape[:2]
        
        # 创建中心明亮的光照效果
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 创建光照遮罩
        light_mask = 1.0 - (distance / max_distance) * 0.1 * intensity
        light_mask = np.clip(light_mask, 0.9, 1.0)
        
        # 应用光照效果
        final = image.astype(np.float32) * light_mask[:,:,np.newaxis]
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        return final
    
    def apply_enhanced_ghibli_style(self, image, use_face_enhancement=True, use_bg_separation=True):
        """应用增强版宫崎骏风格转换"""
        print("🎨 开始应用增强版宫崎骏风格...")
        
        # 更新进度
        if self.progress_callback and self.task_id:
            self.progress_callback(self.task_id, 10, 1, 10, 0)
        
        try:
            # 1. 人脸检测
            faces = []
            if use_face_enhancement:
                faces = self.detect_faces(image)
                print(f"👤 检测到 {len(faces)} 张人脸")
            
            if self.progress_callback and self.task_id:
                self.progress_callback(self.task_id, 30, 2, 10, 0)
            
            # 2. 语义分割
            person_mask = None
            if use_bg_separation:
                person_mask = self.segment_person(image)
                print("🎯 语义分割完成")
            
            if self.progress_callback and self.task_id:
                self.progress_callback(self.task_id, 50, 3, 10, 0)
            
            # 3. 转换为numpy数组
            if isinstance(image, Image.Image):
                img_np = np.array(image)
                if len(img_np.shape) == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = image.copy()
            
            # 4. 首先尝试使用AnimeGAN进行端到端转换
            print("🎨 尝试使用AnimeGAN进行端到端转换...")
            
            try:
                # 转换为PIL格式
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # 使用AnimeGAN转换（宫崎骏风格）
                def anime_gan_progress(stage, progress):
                    if self.progress_callback and self.task_id:
                        self.progress_callback(self.task_id, 20 + int(progress * 0.3), 3, 10, 0)
                
                anime_result = convert_with_anime_gan(
                    img_pil, 
                    model_type='hayao',  # 使用宫崎骏风格
                    progress_callback=anime_gan_progress
                )
                
                # 转换回BGR格式
                img_array = np.array(anime_result)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                print("✅ AnimeGAN转换成功")
                
            except Exception as e:
                print(f"⚠️ AnimeGAN转换失败，使用高级处理器回退: {e}")
                
                # 使用高级处理器作为回退方案
                img_bgr = advanced_processor.process_anime_style(
                    img_bgr, 
                    use_slic=True, 
                    use_xdog=True, 
                    use_multiscale=True, 
                    use_color_mapping=False  # 先不应用色彩映射，后面单独处理
                )
            
            # 保存原始结构用于后续混合
            original_structure = img_bgr.copy()
            
            if self.progress_callback and self.task_id:
                self.progress_callback(self.task_id, 50, 3, 10, 0)
            
            # 5. 人脸美化
            if faces and use_face_enhancement:
                img_bgr = self.enhance_faces(img_bgr, faces)
                print("💄 人脸美化完成")
            
            if self.progress_callback and self.task_id:
                self.progress_callback(self.task_id, 60, 4, 10, 0)
            
            # 6. 背景处理
            if person_mask is not None and use_bg_separation:
                img_bgr = self.process_background(img_bgr, person_mask)
                print("🌅 背景处理完成")
            
            if self.progress_callback and self.task_id:
                self.progress_callback(self.task_id, 70, 5, 10, 0)
            
            # 7. 应用智能宫崎骏色彩映射
            print("🎨 应用智能宫崎骏色彩映射...")
            img_bgr = advanced_processor.intelligent_color_mapping(img_bgr)
            
            if self.progress_callback and self.task_id:
                self.progress_callback(self.task_id, 80, 6, 10, 0)
            
            # 8. 结构保持混合 - 保持物体和人物识别性
            print("🔧 保持结构清晰度...")
            img_bgr = self._preserve_structure_mixing(img_bgr, original_structure)
            
            if self.progress_callback and self.task_id:
                self.progress_callback(self.task_id, 90, 7, 10, 0)
            
            # 9. 最终光照优化
            img_bgr = apply_final_lighting(img_bgr)
            
            # 转换回PIL格式
            result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = Image.fromarray(result_rgb)
            
            if self.progress_callback and self.task_id:
                self.progress_callback(self.task_id, 100, 10, 10, 0)
            
            print("✅ 增强版宫崎骏风格转换完成")
            return result
            
        except Exception as e:
            print(f"❌ 增强版风格转换失败: {e}")
            # 回退到基础版本
            from .real_ghibli_transfer import RealGhibliStyleTransfer
            ghibli_model = RealGhibliStyleTransfer()
            return ghibli_model.apply_real_ghibli_style(image, use_neural=False)
    
    def _preserve_structure_mixing(self, styled_img, original_structure):
        """保持结构混合 - 确保物体和人物清晰可识别"""
        # 1. 检测结构信息
        gray_original = cv2.cvtColor(original_structure, cv2.COLOR_BGR2GRAY)
        gray_styled = cv2.cvtColor(styled_img, cv2.COLOR_BGR2GRAY)
        
        # 2. 计算结构相似度
        # 使用Sobel算子检测边缘
        edges_original = cv2.Sobel(gray_original, cv2.CV_64F, 1, 1, ksize=3)
        edges_styled = cv2.Sobel(gray_styled, cv2.CV_64F, 1, 1, ksize=3)
        
        # 归一化边缘信息
        edges_original = np.abs(edges_original)
        edges_styled = np.abs(edges_styled)
        
        # 3. 创建结构保持掩码
        # 在结构信息强的区域，更多地保留原始结构
        structure_mask = edges_original / (edges_original.max() + 1e-8)
        structure_mask = np.clip(structure_mask * 2, 0, 1)  # 增强结构区域
        
        # 4. 分层混合
        result = np.zeros_like(styled_img)
        
        # 在结构强的区域，更多地保留原始结构
        for i in range(3):
            result[:, :, i] = (
                styled_img[:, :, i] * (1 - structure_mask * 0.3) + 
                original_structure[:, :, i] * structure_mask * 0.3
            )
        
        # 5. 局部对比度增强
        lab = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强局部对比度
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab_enhanced = cv2.merge([l, a, b])
        final = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return final
    
    def set_progress_callback(self, callback, task_id):
        """设置进度回调函数"""
        self.progress_callback = callback
        self.task_id = task_id

# 创建增强版宫崎骏风格转换模型
ghibli_enhanced_model = GhibliEnhancedTransfer()

def update_enhanced_progress(task_id, progress, current_step, total_steps, loss):
    """更新增强版转换进度"""
    print(f"📊 增强版任务 {task_id}: {progress}% (步骤 {current_step}/{total_steps}, 损失: {loss:.4f})")

def convert_image_enhanced_async(task_id, image):
    """异步转换图像 - 增强版"""
    try:
        # 设置进度回调
        ghibli_enhanced_model.set_progress_callback(update_enhanced_progress, task_id)
        
        # 开始转换
        result_image = ghibli_enhanced_model.apply_enhanced_ghibli_style(image)
        
        print(f"✅ 增强版任务 {task_id} 转换完成")
        return result_image
        
    except Exception as e:
        print(f"❌ 增强版任务 {task_id} 转换失败: {e}")
        # 回退到基础版本
        from .real_ghibli_transfer import RealGhibliStyleTransfer
        ghibli_model = RealGhibliStyleTransfer()
        return ghibli_model.apply_real_ghibli_style(image, use_neural=False)

def apply_cartoon_effect(img_bgr):
    """应用卡通化效果 - 改进版本，保持清晰度和结构"""
    # 1. 轻度双边滤波 - 减少强度以保持细节
    bilateral = cv2.bilateralFilter(img_bgr, 9, 60, 60)
    
    # 2. 多层次边缘检测 - 获取更丰富的边缘信息
    gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    
    # Canny边缘检测 - 获取主要轮廓
    edges_canny = cv2.Canny(gray, 30, 100)
    
    # 自适应阈值 - 获取细节边缘
    edges_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 7, 2)
    
    # 合并边缘结果
    edges_combined = cv2.bitwise_or(edges_canny, edges_adaptive)
    
    # 3. 增强颜色量化 - 增加聚类数量保持细节
    data = bilateral.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 16, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  # 增加到16个颜色
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(bilateral.shape)
    
    # 4. 智能边缘叠加 - 保持结构清晰
    edges_colored = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)
    
    # 分层混合：主要使用量化图像，轻度叠加边缘
    cartoon = cv2.addWeighted(quantized, 0.9, edges_colored, 0.1, 0)
    
    # 5. 轻微锐化 - 恢复一些清晰度
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(cartoon, -1, kernel)
    
    # 混合锐化结果
    result = cv2.addWeighted(cartoon, 0.8, sharpened, 0.2, 0)
    
    return result

def apply_ghibli_color_style(img_bgr):
    """应用宫崎骏色彩风格 - 改进版本，保持结构清晰"""
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 适度增强饱和度 - 避免过度饱和导致失真
    s = cv2.add(s, 25)  # 固定增量而非倍数
    s = np.clip(s, 0, 240)
    
    # 轻微调整色调 - 偏向温暖色调
    h = cv2.add(h, 5)
    h = np.clip(h, 0, 179)
    
    # 适度增强亮度
    v = cv2.add(v, 15)
    v = np.clip(v, 0, 255)
    
    # 重新组合
    hsv_enhanced = cv2.merge([h, s, v])
    result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    # 应用LAB色彩空间精调 - 保持自然感
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 增强对比度但不失真
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # 轻微色彩调整
    a = cv2.add(a, 8)
    b = cv2.add(b, 5)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    
    lab_enhanced = cv2.merge([l, a, b])
    final = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return final

def apply_final_lighting(img_bgr):
    """应用最终光照优化 - 改进版本，保持清晰度"""
    height, width = img_bgr.shape[:2]
    
    # 创建更自然的径向光照效果 - 减少强度
    y_coords, x_coords = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # 光照遮罩 - 更柔和的光照效果
    light_mask = 1.0 - (distance / max_distance) * 0.08  # 减少光照强度
    light_mask = np.clip(light_mask, 0.92, 1.0)
    
    # 应用光照
    result = img_bgr.astype(np.float32) * light_mask[:, :, np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 轻微锐化 - 恢复细节
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(result, -1, kernel)
    
    # 混合锐化结果
    final = cv2.addWeighted(result, 0.9, sharpened, 0.1, 0)
    
    # 非常轻微的柔化 - 去除锐化带来的噪点
    final = cv2.bilateralFilter(final, 3, 30, 30)
    
    return final