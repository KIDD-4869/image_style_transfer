#!/usr/bin/env python3
"""
宫崎骏风格专用数据管道
为训练宫崎骏风格GAN模型准备高质量数据集
包含数据收集、预处理、增强、加载等功能
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import time
import glob
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class GhibliDataPipeline:
    """宫崎骏风格专用数据管道"""
    
    def __init__(self, config_path=None):
        """
        初始化数据管道
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        
        # 数据目录
        self.photo_dir = self.config.get('photo_dir', 'training_data/photos')
        self.ghibli_dir = self.config.get('ghibli_dir', 'ghibli_images')
        self.output_dir = self.config.get('output_dir', 'training_data/processed')
        
        # 处理参数
        self.image_size = self.config.get('image_size', 512)
        self.quality_threshold = self.config.get('quality_threshold', 0.3)
        self.augmentation_enabled = self.config.get('augmentation_enabled', True)
        
        # 确保目录存在
        os.makedirs(self.photo_dir, exist_ok=True)
        os.makedirs(self.ghibli_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'photos_processed': 0,
            'ghibli_processed': 0,
            'pairs_created': 0,
            'errors': 0
        }
        
        print("🎨 宫崎骏风格数据管道初始化完成")
    
    def _load_config(self, config_path):
        """加载配置文件"""
        default_config = {
            'photo_dir': 'training_data/photos',
            'ghibli_dir': 'ghibli_images', 
            'output_dir': 'training_data/processed',
            'image_size': 512,
            'quality_threshold': 0.3,
            'augmentation_enabled': True,
            'augmentation_params': {
                'rotation_range': 10,
                'brightness_range': 0.1,
                'contrast_range': 0.1,
                'saturation_range': 0.1
            },
            'preprocessing': {
                'face_detection': True,
                'blur_detection': True,
                'noise_detection': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 合并配置
                for key, value in user_config.items():
                    if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                print(f"📋 加载配置文件: {config_path}")
            except Exception as e:
                print(f"⚠️ 配置文件加载失败，使用默认配置: {e}")
        
        return default_config
    
    def collect_photo_data(self, source_dirs=None):
        """
        收集真实照片数据
        
        Args:
            source_dirs: 源目录列表
        """
        print("📸 开始收集真实照片数据...")
        
        if source_dirs is None:
            source_dirs = ['temp', '.', 'downloads']  # 默认搜索目录
        
        collected_count = 0
        
        for source_dir in source_dirs:
            if not os.path.exists(source_dir):
                print(f"⚠️ 目录不存在: {source_dir}")
                continue
            
            print(f"🔍 搜索目录: {source_dir}")
            
            # 搜索图片文件
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            found_images = []
            
            for ext in image_extensions:
                pattern = os.path.join(source_dir, ext)
                found_images.extend(glob.glob(pattern))
                pattern = os.path.join(source_dir, ext.upper())
                found_images.extend(glob.glob(pattern))
            
            # 去重
            found_images = list(set(found_images))
            
            print(f"   📊 找到 {len(found_images)} 张图片")
            
            # 处理每张图片
            for img_path in found_images:
                try:
                    if self._process_photo_image(img_path):
                        collected_count += 1
                        if collected_count % 100 == 0:
                            print(f"   ✅ 已处理 {collected_count} 张图片")
                except Exception as e:
                    print(f"   ❌ 处理失败 {img_path}: {e}")
                    self.stats['errors'] += 1
        
        print(f"✅ 照片数据收集完成，成功处理 {collected_count} 张图片")
        return collected_count
    
    def collect_ghibli_data(self, extract_frames=True):
        """
        收集宫崎骏风格数据
        
        Args:
            extract_frames: 是否从视频中提取帧
        """
        print("🎬 开始收集宫崎骏风格数据...")
        
        collected_count = 0
        
        # 收集现有的宫崎骏图片
        print("🔍 搜索现有宫崎骏图片...")
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in image_extensions:
            pattern = os.path.join(self.ghibli_dir, ext)
            found_images = glob.glob(pattern)
            
            for img_path in found_images:
                try:
                    if self._process_ghibli_image(img_path):
                        collected_count += 1
                except Exception as e:
                    print(f"❌ 处理失败 {img_path}: {e}")
                    self.stats['errors'] += 1
        
        # 从视频提取帧（如果有）
        if extract_frames:
            video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov']
            
            for ext in video_extensions:
                pattern = os.path.join(self.ghibli_dir, ext)
                found_videos = glob.glob(pattern)
                
                for video_path in found_videos:
                    try:
                        frames_count = self._extract_frames_from_video(video_path)
                        collected_count += frames_count
                    except Exception as e:
                        print(f"❌ 视频处理失败 {video_path}: {e}")
                        self.stats['errors'] += 1
        
        print(f"✅ 宫崎骏数据收集完成，成功处理 {collected_count} 张图片")
        return collected_count
    
    def _process_photo_image(self, img_path):
        """处理单张真实照片"""
        try:
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                return False
            
            # 质量检查
            if not self._check_image_quality(img):
                return False
            
            # 预处理
            processed_img = self._preprocess_image(img)
            
            # 保存处理后的图片
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(self.photo_dir, f"{name}_processed.jpg")
            cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            self.stats['photos_processed'] += 1
            return True
            
        except Exception as e:
            return False
    
    def _process_ghibli_image(self, img_path):
        """处理单张宫崎骏风格图片"""
        try:
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                return False
            
            # 预处理
            processed_img = self._preprocess_ghibli_image(img)
            
            # 保存处理后的图片
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(self.ghibli_dir, f"{name}_processed.jpg")
            cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            self.stats['ghibli_processed'] += 1
            return True
            
        except Exception as e:
            return False
    
    def _extract_frames_from_video(self, video_path):
        """从视频中提取帧"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return 0
        
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 每秒提取一帧
        frame_interval = int(fps)
        
        filename = os.path.splitext(os.path.basename(video_path))[0]
        
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if ret:
                # 预处理帧
                processed_frame = self._preprocess_ghibli_image(frame)
                
                # 保存帧
                output_path = os.path.join(self.ghibli_dir, f"{filename}_frame_{frame_count:06d}.jpg")
                cv2.imwrite(output_path, processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
                frame_count += 1
        
        cap.release()
        return frame_count
    
    def _check_image_quality(self, img):
        """检查图像质量"""
        # 模糊检测
        if self.config['preprocessing']['blur_detection']:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if blur_score < 100:  # 模糊阈值
                return False
        
        # 噪声检测
        if self.config['preprocessing']['noise_detection']:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            noise_score = np.var(gray)
            
            # 噪声过高或过低都排除
            if noise_score < 50 or noise_score > 2000:
                return False
        
        # 尺寸检查
        h, w = img.shape[:2]
        if h < 256 or w < 256:
            return False
        
        return True
    
    def _preprocess_image(self, img):
        """预处理真实照片"""
        # 调整尺寸
        h, w = img.shape[:2]
        if max(h, w) != self.image_size:
            scale = self.image_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 中心裁剪到目标尺寸
        h, w = img.shape[:2]
        start_y = (h - self.image_size) // 2
        start_x = (w - self.image_size) // 2
        img = img[start_y:start_y+self.image_size, start_x:start_x+self.image_size]
        
        # 人脸检测（可选）
        if self.config['preprocessing']['face_detection']:
            try:
                # 使用OpenCV的人脸检测
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # 如果检测到人脸，优先保留包含人脸的图片
                if len(faces) > 0:
                    pass  # 保留图片
                # 可以添加其他逻辑...
            except:
                pass
        
        return img
    
    def _preprocess_ghibli_image(self, img):
        """预处理宫崎骏风格图片"""
        # 调整尺寸
        h, w = img.shape[:2]
        if max(h, w) != self.image_size:
            scale = self.image_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 中心裁剪到目标尺寸
        h, w = img.shape[:2]
        start_y = (h - self.image_size) // 2
        start_x = (w - self.image_size) // 2
        img = img[start_y:start_y+self.image_size, start_x:start_x+self.image_size]
        
        # 宫崎骏风格特有的预处理
        # 增强饱和度
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return img
    
    def create_training_pairs(self):
        """创建训练配对数据"""
        print("🔗 开始创建训练配对...")
        
        # 获取处理后的图片列表
        photo_files = glob.glob(os.path.join(self.photo_dir, "*_processed.jpg"))
        ghibli_files = glob.glob(os.path.join(self.ghibli_dir, "*_processed.jpg"))
        
        print(f"📊 照片数量: {len(photo_files)}")
        print(f"📊 宫崎骏图片数量: {len(ghibli_files)}")
        
        if len(photo_files) == 0 or len(ghibli_files) == 0:
            print("❌ 缺少训练数据")
            return 0
        
        # 创建配对
        pairs_created = 0
        
        # 为每张照片创建配对
        for photo_path in photo_files:
            try:
                # 随机选择宫崎骏风格图片
                ghibli_path = random.choice(ghibli_files)
                
                # 创建配对目录
                pair_id = f"pair_{pairs_created:06d}"
                pair_dir = os.path.join(self.output_dir, pair_id)
                os.makedirs(pair_dir, exist_ok=True)
                
                # 复制文件
                photo_name = os.path.basename(photo_path)
                ghibli_name = os.path.basename(ghibli_path)
                
                os.system(f"cp '{photo_path}' '{os.path.join(pair_dir, 'photo.jpg')}'")
                os.system(f"cp '{ghibli_path}' '{os.path.join(pair_dir, 'ghibli.jpg')}'")
                
                # 创建配对信息文件
                pair_info = {
                    'pair_id': pair_id,
                    'photo_file': photo_name,
                    'ghibli_file': ghibli_name,
                    'created_time': time.time()
                }
                
                with open(os.path.join(pair_dir, 'info.json'), 'w') as f:
                    json.dump(pair_info, f, indent=2)
                
                pairs_created += 1
                
                if pairs_created % 100 == 0:
                    print(f"   ✅ 已创建 {pairs_created} 个配对")
                
            except Exception as e:
                print(f"❌ 创建配对失败: {e}")
                self.stats['errors'] += 1
        
        self.stats['pairs_created'] = pairs_created
        print(f"✅ 训练配对创建完成，共创建 {pairs_created} 个配对")
        return pairs_created
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'photos_processed': self.stats['photos_processed'],
            'ghibli_processed': self.stats['ghibli_processed'], 
            'pairs_created': self.stats['pairs_created'],
            'errors': self.stats['errors'],
            'photo_dir_size': len(glob.glob(os.path.join(self.photo_dir, "*.jpg"))),
            'ghibli_dir_size': len(glob.glob(os.path.join(self.ghibli_dir, "*.jpg"))),
            'output_dir_size': len(glob.glob(os.path.join(self.output_dir, "pair_*")))
        }

class GhibliDataset(Dataset):
    """宫崎骏风格训练数据集"""
    
    def __init__(self, data_dir, transform=None, augmentation=True):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            transform: 图像变换
            augmentation: 是否启用数据增强
        """
        self.data_dir = data_dir
        self.transform = transform
        self.augmentation = augmentation
        
        # 加载配对数据
        self.pairs = []
        pair_dirs = glob.glob(os.path.join(data_dir, "pair_*"))
        
        for pair_dir in pair_dirs:
            photo_path = os.path.join(pair_dir, 'photo.jpg')
            ghibli_path = os.path.join(pair_dir, 'ghibli.jpg')
            
            if os.path.exists(photo_path) and os.path.exists(ghibli_path):
                self.pairs.append((photo_path, ghibli_path))
        
        print(f"📊 加载了 {len(self.pairs)} 个训练配对")
        
        # 数据增强变换
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        photo_path, ghibli_path = self.pairs[idx]
        
        # 读取图像
        photo = Image.open(photo_path).convert('RGB')
        ghibli = Image.open(ghibli_path).convert('RGB')
        
        # 数据增强
        if self.augmentation and random.random() > 0.5:
            # 同样的随机种子确保两个图像应用相同的变换
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            photo = self.aug_transform(photo)
            
            random.seed(seed)
            torch.manual_seed(seed)
            ghibli = self.aug_transform(ghibli)
        
        # 应用基础变换
        if self.transform:
            photo = self.transform(photo)
            ghibli = self.transform(ghibli)
        
        return {
            'photo': photo,
            'ghibli': ghibli,
            'photo_path': photo_path,
            'ghibli_path': ghibli_path
        }

def create_dataloaders(data_dir, batch_size=4, num_workers=2, train_split=0.8):
    """
    创建训练和验证数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作进程数
        train_split: 训练集比例
        
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 创建数据集
    full_dataset = GhibliDataset(data_dir, transform=transform, augmentation=True)
    
    # 分割数据集
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"📊 训练集: {len(train_dataset)} 个样本")
    print(f"📊 验证集: {len(val_dataset)} 个样本")
    
    return train_loader, val_loader

    def collect_training_data(self, photo_dir, style_dir):
        """
        收集训练数据
        
        Args:
            photo_dir: 照片目录
            style_dir: 风格目录
        
        Returns:
            tuple: (照片列表, 风格列表)
        """
        photos = []
        styles = []
        
        # 收集照片
        if os.path.exists(photo_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                photos.extend(glob.glob(os.path.join(photo_dir, ext)))
                photos.extend(glob.glob(os.path.join(photo_dir, ext.upper())))
        
        # 收集风格图像
        if os.path.exists(style_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                styles.extend(glob.glob(os.path.join(style_dir, ext)))
                styles.extend(glob.glob(os.path.join(style_dir, ext.upper())))
        
        return photos, styles

# 创建全局数据管道实例
ghibli_pipeline = GhibliDataPipeline()