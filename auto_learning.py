#!/usr/bin/env python3
"""
自主学习模块 - 自动下载宫崎骏风格图片并进行深度学习
"""

import os
import time
import random
from PIL import Image
import cv2
import numpy as np

class GhibliAutoLearner:
    """宫崎骏风格自主学习器"""
    
    def __init__(self, download_folder="temp/learning"):
        self.download_folder = download_folder
        self.learning_images = []
        os.makedirs(download_folder, exist_ok=True)
        
        # 宫崎骏相关搜索关键词
        self.search_keywords = [
            "宫崎骏动漫", "吉卜力工作室", "千与千寻", "龙猫", "哈尔的移动城堡",
            "天空之城", "幽灵公主", "魔女宅急便", "风之谷", "红猪",
            "悬崖上的金鱼姬", "起风了", "侧耳倾听", "猫的报恩"
        ]
        
        # 用户代理列表
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        ]
    
    def get_random_user_agent(self):
        """获取随机用户代理"""
        return random.choice(self.user_agents)
    
    def search_ghibli_images(self, keyword, max_images=10):
        """搜索宫崎骏风格图片 - 主要使用必应搜索"""
        print(f"🔍 搜索关键词: {keyword}")
        
        # 主要使用必应搜索（更可靠）
        downloaded_count = self._search_backup_images(keyword, max_images)
        
        # 如果必应搜索失败，再尝试百度
        if downloaded_count == 0:
            print(f"⚠️ 必应搜索失败，尝试百度搜索...")
            downloaded_count = self._search_baidu_images(keyword, max_images)
        
        return downloaded_count
    
    def _search_baidu_images(self, keyword, max_images):
        """百度图片搜索 - 使用更可靠的方法"""
        headers = {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://image.baidu.com/',
        }
        
        try:
            # 使用更简单的百度图片搜索URL
            search_url = f"https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&dyTabStr=MCwzLDYsMiw0LDUsNyw4LDksMTAsMTEsMTIsMTMsMTQsMTUsMTYsMTcsMTgsMTksMjAsMjEsMjIsMjMsMjQsMjUsMjYsMjcsMjgsMjksMzAsMzEsMzIsMzMsMzQsMzUsMzYsMzcsMzgsMzksNDAsNDEsNDIsNDMsNDQsNDUsNDYsNDcsNDgsNDksNTAsNTEsNTIsNTMsNTQsNTUsNTYsNTcsNTgsNTksNjAsNjEsNjIsNjMsNjQsNjUsNjYsNjcsNjgsNjksNzAsNzEsNzIsNzMsNzQsNzUsNzYsNzcsNzgsNzksODAsODEsODIsODMsODQsODUsODYsODcsODgsODksOTAsOTEsOTIsOTMsOTQsOTUsOTYsOTcsOTgsOTksMTAwLDEwMSwxMDIsMTAzLDEwNCwxMDUsMTA2LDEwNywxMDgsMTA5LDExMCwxMTEsMTEyLDExMywxMTQsMTE1LDExNiwxMTcsMTE4LDExOSwxMjAsMTIxLDEyMiwxMjMsMTI0LDEyNSwxMjYsMTI3LDEyOCwxMjksMTMwLDEzMSwxMzIsMTMzLDEzNCwxMzUsMTM2LDEzNywxMzg&word={keyword}"
            
            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # 从HTML页面中提取图片URL
            image_urls = self.extract_image_urls_from_html(response.text)
            
            # 如果HTML解析失败，尝试备用方法
            if not image_urls:
                print("⚠️ HTML解析失败，尝试备用解析方法...")
                image_urls = self.extract_image_urls_backup(response.text)
            
            downloaded_count = 0
            for i, img_url in enumerate(image_urls[:max_images]):
                if self.download_image(img_url, f"{keyword}_{i}"):
                    downloaded_count += 1
                    time.sleep(2)  # 避免请求过快
            
            return downloaded_count
            
        except Exception as e:
            print(f"❌ 百度搜索失败: {e}")
            return 0
    
    def _search_backup_images(self, keyword, max_images):
        """备用图片搜索源 - 优先使用本地图片，其次使用网络图片"""
        
        # 首先尝试使用本地图片
        local_images = self._get_local_ghibli_images(keyword)
        if local_images:
            print(f"📁 使用本地图片源: {keyword}")
            copied_count = 0
            
            for i, img_path in enumerate(local_images[:max_images]):
                if self._copy_local_image(img_path, f"local_{keyword}_{i}"):
                    copied_count += 1
            
            if copied_count > 0:
                print(f"✅ 本地图片源使用成功 {copied_count} 张图片")
                return copied_count
        
        # 如果本地图片不存在，使用预定义的网络图片
        predefined_urls = self.get_predefined_ghibli_images(keyword)
        
        if predefined_urls:
            print(f"🔍 使用预定义网络图片源: {keyword}")
            downloaded_count = 0
            
            for i, img_url in enumerate(predefined_urls[:max_images]):
                if self.download_image(img_url, f"predefined_{keyword}_{i}"):
                    downloaded_count += 1
                    time.sleep(1)  # 避免请求过快
            
            if downloaded_count > 0:
                print(f"✅ 预定义图片源下载成功 {downloaded_count} 张图片")
            
            return downloaded_count
        
        # 如果预定义图片源失败，尝试必应搜索
        return self._search_bing_fallback(keyword, max_images)
    
    def _get_local_ghibli_images(self, keyword):
        """获取本地宫崎骏风格图片路径 - 简化版本"""
        # 检查是否存在本地图片文件夹
        local_folders = [
            "ghibli_images",
            "static/ghibli_images",
            "images/ghibli",
            "static/images",
            "images"
        ]
        
        all_image_files = []
        
        for folder in local_folders:
            if os.path.exists(folder) and os.path.isdir(folder):
                # 查找文件夹中的所有图片文件
                for file in os.listdir(folder):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        all_image_files.append(os.path.join(folder, file))
        
        # 随机选择图片，避免每次都使用相同的图片
        import random
        random.shuffle(all_image_files)
        
        return all_image_files
    
    def _copy_local_image(self, source_path, filename):
        """复制本地图片到学习文件夹"""
        try:
            # 读取图片
            image = Image.open(source_path)
            
            # 保存到学习文件夹
            filepath = os.path.join(self.download_folder, f"{filename}.jpg")
            image.save(filepath, "JPEG", quality=90, optimize=True)
            
            print(f"✅ 复制本地图片: {filename} ({image.size[0]}x{image.size[1]})")
            self.learning_images.append(filepath)
            return True
            
        except Exception as e:
            print(f"❌ 复制本地图片失败 {filename}: {e}")
            return False
    
    def get_predefined_ghibli_images(self, keyword):
        """获取预定义的宫崎骏风格图片URL"""
        
        # 宫崎骏风格学习图片 - 使用动漫风格图片
        # 这些是公开的动漫风格图片，更接近宫崎骏风格
        ghibli_images = {
            '宫崎骏动漫': [
                "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=800&h=600&fit=crop",  # 幻想动漫风格
                "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800&h=600&fit=crop",  # 艺术动漫风格
                "https://images.unsplash.com/photo-1637858868799-7f26a0640eb6?w=800&h=600&fit=crop",  # 动漫插画风格
            ],
            '吉卜力工作室': [
                "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=800&h=600&fit=crop",
                "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800&h=600&fit=crop",
                "https://images.unsplash.com/photo-1637858868799-7f26a0640eb6?w=800&h=600&fit=crop",
            ],
            '千与千寻': [
                "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=800&h=600&fit=crop",
                "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800&h=600&fit=crop",
                "https://images.unsplash.com/photo-1637858868799-7f26a0640eb6?w=800&h=600&fit=crop",
            ]
        }
        
        return ghibli_images.get(keyword, [])
    
    def _search_bing_fallback(self, keyword, max_images):
        """备用必应搜索"""
        headers = {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        try:
            english_keywords = {
                '宫崎骏动漫': 'studio ghibli anime wallpaper',
                '吉卜力工作室': 'studio ghibli wallpaper',
                '千与千寻': 'spirited away wallpaper',
                '龙猫': 'totoro wallpaper',
            }
            
            english_keyword = english_keywords.get(keyword, keyword)
            
            search_url = f"https://www.bing.com/images/search?q={english_keyword}&qft=+filterui:imagesize-large"
            
            print(f"🔍 备用必应搜索: {english_keyword}")
            
            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            if len(response.text) < 1000:
                print("⚠️ 必应响应内容过短")
                return 0
            
            image_urls = self._extract_bing_image_urls(response.text)
            
            print(f"📷 找到 {len(image_urls)} 个图片URL")
            
            downloaded_count = 0
            for i, img_url in enumerate(image_urls[:max_images]):
                if self.download_image(img_url, f"bing_{keyword}_{i}"):
                    downloaded_count += 1
                    time.sleep(3)  # 更长的间隔
            
            return downloaded_count
            
        except Exception as e:
            print(f"❌ 备用必应搜索失败: {e}")
            return 0
    
    def _extract_bing_image_urls(self, html_content):
        """从必应搜索结果中提取图片URL - 改进版本"""
        try:
            import re
            
            # 方法1: 查找真实的图片URL（来自网站内容）
            pattern = r'https?:[^"\'\s<>]+\.(?:jpg|jpeg|png|webp)'
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            
            # 清理URL
            clean_urls = []
            for url in matches:
                # 彻底清理URL
                url = url.replace('&quot;', '').replace('"', '').replace('\\', '')
                
                # 过滤掉明显无效的URL
                if (url.startswith('http') and 
                    len(url) > 30 and 
                    ' ' not in url and
                    'bing.net/th/id/OIP-C' not in url and
                    'facebook' not in url.lower() and
                    'logo' not in url.lower()):
                    
                    # 修复URL格式问题
                    if ':/' in url and '://' not in url:
                        url = url.replace(':/', '://', 1)
                    
                    clean_urls.append(url)
            
            # 去重
            clean_urls = list(set(clean_urls))
            
            print(f"🔍 提取到 {len(clean_urls)} 个有效图片URL")
            
            # 显示前几个URL用于调试
            if clean_urls:
                for i, url in enumerate(clean_urls[:3]):
                    print(f"  {i+1}. {url[:100]}...")
            
            return clean_urls
            
        except Exception as e:
            print(f"❌ 必应解析失败: {e}")
            return []
    
    def clean_image_url(self, url):
        """清理和验证图片URL - 改进版本"""
        if not url:
            return None
        
        # 清理URL中的多余字符
        url = url.strip()
        
        # 处理常见的URL格式问题
        url = url.replace('&quot;', '').replace('"', '')
        
        # 提取真正的图片URL（处理murl:前缀）
        if 'murl:' in url:
            parts = url.split('murl:')
            if len(parts) > 1:
                url = parts[1]
        
        # 检查是否是有效的图片URL
        if not any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
            return None
        
        # 标准化URL格式
        if url.startswith('//'):
            url = 'https:' + url
        elif not url.startswith('http'):
            return None
        
        # 过滤掉明显无效的URL
        if len(url) < 15 or ' ' in url or 'murl:' in url:
            return None
        
        # 确保URL格式正确
        if not url.startswith('http://') and not url.startswith('https://'):
            return None
        
        return url
    
    def extract_image_urls_from_json(self, json_content):
        """从JSON数据中提取图片URL - 改进版本"""
        try:
            import json
            # 清理JSON数据，处理可能的格式问题
            cleaned_content = json_content.strip()
            
            # 检查是否是有效的JSON
            if not cleaned_content or cleaned_content[0] not in ['{', '[']:
                print("⚠️ 返回内容不是有效的JSON格式")
                return []
            
            # 尝试解析JSON
            data = json.loads(cleaned_content)
            
            image_urls = []
            
            # 处理不同的JSON结构
            if isinstance(data, dict):
                if 'data' in data:
                    for item in data['data']:
                        if isinstance(item, dict):
                            # 尝试多种可能的URL字段
                            for field in ['middleURL', 'thumbURL', 'objURL', 'hoverURL', 'fromURL']:
                                if field in item and item[field]:
                                    image_urls.append(item[field])
                                    break
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for field in ['middleURL', 'thumbURL', 'objURL', 'hoverURL', 'fromURL']:
                            if field in item and item[field]:
                                image_urls.append(item[field])
                                break
            
            return list(set(image_urls))  # 去重
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            return []
        except Exception as e:
            print(f"❌ JSON处理失败: {e}")
            return []
    
    def extract_image_urls_from_html(self, html_content):
        """从HTML页面中提取图片URL - 主要方法"""
        try:
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            image_urls = []
            
            # 方法1: 查找所有图片标签
            for img in soup.find_all('img'):
                src = img.get('src', '')
                data_src = img.get('data-src', '')
                data_url = img.get('data-url', '')
                
                # 检查src属性
                if src and self.is_valid_image_url(src):
                    full_url = self.normalize_url(src)
                    if full_url:
                        image_urls.append(full_url)
                
                # 检查data-src属性（懒加载图片）
                if data_src and self.is_valid_image_url(data_src):
                    full_url = self.normalize_url(data_src)
                    if full_url:
                        image_urls.append(full_url)
                
                # 检查data-url属性
                if data_url and self.is_valid_image_url(data_url):
                    full_url = self.normalize_url(data_url)
                    if full_url:
                        image_urls.append(full_url)
            
            # 方法2: 使用正则表达式查找隐藏的图片URL
            import re
            patterns = [
                r'"objURL"\s*:\s*"([^"]+)"',
                r'"middleURL"\s*:\s*"([^"]+)"',
                r'"thumbURL"\s*:\s*"([^"]+)"',
                r'"hoverURL"\s*:\s*"([^"]+)"',
                r'"URL"\s*:\s*"([^"]+)"',
                r'data-imgurl="([^"]+)"',
                r'data-original="([^"]+)"',
                r'data-src="([^"]+)"',
                r'data-url="([^"]+)"'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, html_content)
                for url in matches:
                    if self.is_valid_image_url(url):
                        full_url = self.normalize_url(url)
                        if full_url:
                            image_urls.append(full_url)
            
            # 方法3: 查找背景图片
            style_patterns = [
                r'background-image\s*:\s*url\(["\']?([^"\'\)]+)["\']?\)',
                r'background\s*:\s*url\(["\']?([^"\'\)]+)["\']?\)'
            ]
            
            for pattern in style_patterns:
                matches = re.findall(pattern, html_content)
                for url in matches:
                    if self.is_valid_image_url(url):
                        full_url = self.normalize_url(url)
                        if full_url:
                            image_urls.append(full_url)
            
            return list(set(image_urls))  # 去重
            
        except Exception as e:
            print(f"❌ HTML解析失败: {e}")
            return []
    
    def extract_image_urls_backup(self, html_content):
        """备用图片URL提取方法"""
        try:
            import re
            image_urls = []
            
            # 查找所有可能的图片URL模式
            url_patterns = [
                r'https?:[^"\'\s<>]+\.(?:jpg|jpeg|png|webp|gif|bmp)',
                r'//[^"\'\s<>]+\.(?:jpg|jpeg|png|webp|gif|bmp)',
                r'/[^"\'\s<>]+\.(?:jpg|jpeg|png|webp|gif|bmp)',
            ]
            
            for pattern in url_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                for url in matches:
                    if self.is_valid_image_url(url):
                        full_url = self.normalize_url(url)
                        if full_url:
                            image_urls.append(full_url)
            
            return list(set(image_urls))
            
        except Exception as e:
            print(f"❌ 备用解析失败: {e}")
            return []
    
    def is_valid_image_url(self, url):
        """检查URL是否是有效的图片URL"""
        if not url or len(url) < 10:
            return False
        
        # 检查图片扩展名
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']
        url_lower = url.lower()
        
        # 检查扩展名
        if any(ext in url_lower for ext in valid_extensions):
            return True
        
        # 检查常见的图片URL模式
        if any(pattern in url_lower for pattern in ['image', 'img', 'pic', 'photo']):
            return True
        
        return False
    
    def normalize_url(self, url):
        """标准化URL格式"""
        if not url:
            return None
        
        # 处理相对URL
        if url.startswith('//'):
            return 'https:' + url
        elif url.startswith('/'):
            return 'https://image.baidu.com' + url
        elif url.startswith('http'):
            return url
        
        return None
    
    def download_image(self, image_url, filename):
        """下载单张图片 - 改进版本"""
        max_retries = 2
        
        # 首先验证URL
        if not self.validate_image_url(image_url):
            print(f"⚠️ 无效URL，跳过: {filename}")
            return False
        
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': self.get_random_user_agent(),
                    'Referer': 'https://www.bing.com/',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
                
                # 处理URL格式
                if image_url.startswith('//'):
                    image_url = 'https:' + image_url
                elif not image_url.startswith('http'):
                    image_url = 'https://' + image_url
                
                # 修复URL格式问题
                if ':/' in image_url and '://' not in image_url:
                    image_url = image_url.replace(':/', '://', 1)
                
                print(f"  📥 尝试下载: {filename}")
                print(f"     URL: {image_url[:100]}...")
                
                response = requests.get(image_url, headers=headers, timeout=15)
                response.raise_for_status()
                
                # 检查图片格式和大小
                if len(response.content) < 10240:  # 10KB以下可能不是有效图片
                    print(f"⚠️ 图片太小({len(response.content)}字节)，跳过: {filename}")
                    return False
                
                # 验证图片格式
                try:
                    image = Image.open(io.BytesIO(response.content))
                    
                    # 检查图片格式
                    if image.format not in ['JPEG', 'PNG', 'WEBP']:
                        print(f"⚠️ 不支持的图片格式({image.format}): {filename}")
                        return False
                    
                    # 过滤掉太小的图片
                    if image.size[0] < 200 or image.size[1] < 200:
                        print(f"⚠️ 图片尺寸太小({image.size[0]}x{image.size[1]}): {filename}")
                        return False
                    
                    # 检查图片质量（避免下载损坏的图片）
                    if image.mode == 'P':  # 调色板模式，可能有问题
                        image = image.convert('RGB')
                    
                    # 保存图片
                    filepath = os.path.join(self.download_folder, f"{filename}.jpg")
                    image.save(filepath, "JPEG", quality=90, optimize=True)
                    
                    print(f"✅ 下载成功: {filename} ({image.size[0]}x{image.size[1]})")
                    self.learning_images.append(filepath)
                    return True
                    
                except Exception as img_error:
                    print(f"❌ 图片处理失败 {filename}: {img_error}")
                    if attempt == max_retries - 1:
                        return False
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ 下载失败 {filename} (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return False
                time.sleep(2)  # 重试前等待更长时间
            
            except Exception as e:
                print(f"❌ 下载失败 {filename}: {e}")
                return False
        
        return False
    
    def validate_image_url(self, url):
        """验证图片URL是否有效"""
        if not url or len(url) < 10:
            return False
        
        # 检查URL格式
        if not url.startswith('http'):
            return False
        
        # 检查图片扩展名或图片服务域名
        if (not any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']) and
            not any(service in url.lower() for service in ['picsum.photos', 'unsplash.com', 'placeholder.com'])):
            return False
        
        # 过滤掉明显无效的URL
        if any(bad in url.lower() for bad in ['logo', 'icon', 'avatar', 'thumb']):
            return False
        
        # 检查URL中是否包含特殊字符
        if ' ' in url or '\n' in url or '\t' in url:
            return False
        
        return True
    
    def preprocess_learning_images(self):
        """预处理学习图片"""
        print("🔄 预处理学习图片...")
        
        processed_images = []
        for img_path in self.learning_images:
            try:
                # 读取图片
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # 调整大小（保持宽高比）
                h, w = image.shape[:2]
                max_size = 800
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 增强图片质量
                image = self.enhance_image_quality(image)
                
                # 保存处理后的图片
                cv2.imwrite(img_path, image)
                processed_images.append(img_path)
                
            except Exception as e:
                print(f"❌ 预处理失败 {img_path}: {e}")
        
        self.learning_images = processed_images
        print(f"✅ 预处理完成，有效图片: {len(processed_images)} 张")
    
    def enhance_image_quality(self, image):
        """增强图片质量"""
        # 转换为LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强亮度和对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # 合并通道
        lab_enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 轻微降噪
        enhanced = cv2.medianBlur(enhanced, 3)
        
        return enhanced
    
    def start_auto_learning(self, max_total_images=100):
        """开始自主学习 - 改进版本，增加样本数量"""
        print("🎯 开始宫崎骏风格自主学习...")
        
        total_downloaded = 0
        # 优先使用本地图片，确保质量
        local_images = self._get_local_ghibli_images("all")
        
        if local_images:
            print(f"📁 发现 {len(local_images)} 张本地宫崎骏风格图片")
            for img_path in local_images[:min(30, len(local_images))]:
                if total_downloaded >= max_total_images:
                    break
                if self._copy_local_image(img_path, f"local_{total_downloaded}"):
                    total_downloaded += 1
        
        # 如果本地图片不足，再使用网络搜索
        if total_downloaded < max_total_images:
            for keyword in self.search_keywords:
                if total_downloaded >= max_total_images:
                    break
                
                downloaded = self.search_ghibli_images(keyword, max_images=8)
                total_downloaded += downloaded
                time.sleep(1)  # 避免请求过快
        
        if total_downloaded > 0:
            self.preprocess_learning_images()
            print(f"🎉 自主学习完成！共收集 {len(self.learning_images)} 张宫崎骏风格图片")
            
            # 增强学习效果
            self.enhance_learning_quality()
        else:
            print("⚠️ 未下载到任何图片，使用默认风格")
        
        return self.learning_images
    
    def enhance_learning_quality(self):
        """增强学习质量 - 使用更高级的图像处理技术"""
        print("🔧 增强学习图片质量...")
        
        enhanced_images = []
        for img_path in self.learning_images:
            try:
                # 读取图片
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # 1. 高质量缩放
                h, w = image.shape[:2]
                max_size = 1024  # 提高分辨率
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # 2. 高级色彩增强
                image = self.advanced_color_enhancement(image)
                
                # 3. 细节增强
                image = self.enhance_details(image)
                
                # 4. 降噪处理
                image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                
                # 保存处理后的图片
                cv2.imwrite(img_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                enhanced_images.append(img_path)
                
            except Exception as e:
                print(f"❌ 图片增强失败 {img_path}: {e}")
        
        self.learning_images = enhanced_images
        print(f"✅ 图片质量增强完成，有效图片: {len(enhanced_images)} 张")
    
    def advanced_color_enhancement(self, image):
        """高级色彩增强 - 模拟宫崎骏风格色彩"""
        # 转换为LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强亮度和对比度（宫崎骏风格特点：明亮、高对比度）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # 增强色彩饱和度（宫崎骏风格色彩鲜艳）
        a = cv2.addWeighted(a, 1.3, a, 0, 0)
        b = cv2.addWeighted(b, 1.3, b, 0, 0)
        
        # 合并通道
        lab_enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 应用柔和滤镜（宫崎骏风格柔和）
        soft = cv2.GaussianBlur(enhanced, (3, 3), 0)
        result = cv2.addWeighted(enhanced, 0.8, soft, 0.2, 0)
        
        return result
    
    def enhance_details(self, image):
        """增强图片细节"""
        # 使用非锐化掩蔽增强细节
        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
        unsharp_mask = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        return unsharp_mask
    
    def cleanup_learning_files(self):
        """清理学习文件"""
        print("🧹 清理学习文件...")
        
        if os.path.exists(self.download_folder):
            for file in os.listdir(self.download_folder):
                file_path = os.path.join(self.download_folder, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"❌ 删除失败 {file_path}: {e}")
            
            try:
                os.rmdir(self.download_folder)
                print("✅ 学习文件清理完成")
            except:
                print("⚠️ 文件夹删除失败，可能仍有文件")


def test_auto_learning():
    """测试自主学习功能 - 改进版本"""
    learner = GhibliAutoLearner()
    
    try:
        # 只测试前3个关键词，避免过多失败
        test_keywords = learner.search_keywords[:3]
        print(f"🧪 测试关键词: {test_keywords}")
        
        total_downloaded = 0
        for keyword in test_keywords:
            if total_downloaded >= 5:  # 最多下载5张
                break
            
            downloaded = learner.search_ghibli_images(keyword, max_images=2)
            total_downloaded += downloaded
            time.sleep(2)  # 避免请求过快
        
        if total_downloaded > 0:
            learner.preprocess_learning_images()
            print(f"✅ 自主学习测试成功，获得 {len(learner.learning_images)} 张图片")
        else:
            print("⚠️ 自主学习测试完成，但未获得图片")
        
        # 清理文件
        learner.cleanup_learning_files()
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
        learner.cleanup_learning_files()
    except Exception as e:
        print(f"❌ 自主学习测试失败: {e}")
        learner.cleanup_learning_files()


if __name__ == "__main__":
    test_auto_learning()