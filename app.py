import io
import base64
from flask import Flask, render_template, request, jsonify
import os
import time
import threading
from PIL import Image
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 导入配置和核心模块
from config import get_config, config

# 获取配置实例
app_config = get_config()

try:
    from core.image_processor_interface import ProcessingStyle
except ImportError:
    from enum import Enum
    class ProcessingStyle(Enum):
        GHIBLI_ENHANCED = "ghibli_enhanced"

# 使用改进的任务管理器
from utils.improved_task_manager import task_manager, TaskStatus

# 导入缓存管理器
from utils.cache_manager import get_cache_manager

app = Flask(__name__)

# 初始化缓存管理器（全局单例）
cache_manager = None

# 加载配置
app.config.from_object(config['default'])

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化缓存管理器
try:
    if app_config.cache_enabled:
        cache_manager = get_cache_manager(
            memory_size_mb=app_config.cache_memory_size_mb,
            disk_size_mb=app_config.cache_disk_size_mb,
            cache_dir=app_config.cache_dir,
            ttl_hours=app_config.cache_ttl_hours,
            enable_disk_cache=app_config.cache_enable_disk
        )
        logger.info(f"✅ 缓存系统已启用 (内存:{app_config.cache_memory_size_mb}MB, 磁盘:{app_config.cache_disk_size_mb}MB)")
    else:
        cache_manager = None
        logger.info("ℹ️ 缓存系统已禁用")
except Exception as e:
    logger.warning(f"⚠️ 缓存系统初始化失败: {e}，将不使用缓存")
    cache_manager = None

def convert_image_async(task_id, image, processor_type="ghibli", style_strategy="balanced"):
    """异步转换图像 - 使用新的统一处理器架构（支持缓存）
    
    Args:
        task_id: 任务ID
        image: 输入图像
        processor_type: 处理器类型
        style_strategy: 风格策略 (fast, balanced, quality)
    """
    try:
        # 更新任务状态为处理中
        task_manager.set_task_status(task_id, TaskStatus.PROCESSING)
        
        # 使用新的统一处理器
        from core.processors import GhibliProcessor, ProcessingStrategy
        
        # 映射策略名称
        strategy_map = {
            'fast': ProcessingStrategy.FAST,
            'classic': ProcessingStrategy.FAST,
            'balanced': ProcessingStrategy.BALANCED,
            'perfect': ProcessingStrategy.BALANCED,
            'quality': ProcessingStrategy.QUALITY,
            'true': ProcessingStrategy.QUALITY,
            'gan': ProcessingStrategy.QUALITY,  # GAN使用高质量策略
            'sd': ProcessingStrategy.BALANCED,  # Stable Diffusion
            'sd_fast': ProcessingStrategy.FAST,
            'sd_quality': ProcessingStrategy.QUALITY,
            # Enhanced 模式映射
            'enhanced_fast': ProcessingStrategy.FAST,
            'enhanced_balanced': ProcessingStrategy.BALANCED,
            'enhanced_quality': ProcessingStrategy.QUALITY,
            'enhanced_ultra': ProcessingStrategy.QUALITY
        }
        
        strategy = strategy_map.get(style_strategy, ProcessingStrategy.BALANCED)
        strategy_str = style_strategy  # 保存原始策略字符串
        
        # 检查是否使用特殊模式
        use_gan = (style_strategy == 'gan')
        use_sd = (style_strategy in ['sd', 'sd_fast', 'sd_quality'])
        use_enhanced = style_strategy.startswith('enhanced_')
        
        # 1. 禁用缓存检查 - 始终重新处理以确保使用正确的处理器
        result = None
        cache_hit = False
        cache_key = None
        
        # 缓存已禁用，始终重新处理
        logger.info(f"🔄 缓存已禁用，将重新处理任务 {task_id}")
        
        # 2. 进行处理（不检查缓存）
        if True:  # 始终处理
            # 根据模式选择处理器
            # 优先级: Enhanced > SD > GAN > CV
            # use_enhanced 已在上面定义
            
            # 初始化 processing_mode 为 None
            processing_mode = None
            
            if use_enhanced:
                logger.info(f"✨ 使用Enhanced Ghibli处理器处理任务 {task_id}")
                try:
                    from core.processors.enhanced_ghibli_processor import EnhancedGhibliProcessor
                    from core.models import ProcessingMode
                    
                    processor = EnhancedGhibliProcessor()
                    
                    # 映射策略到处理模式
                    mode_map = {
                        'enhanced_fast': ProcessingMode.FAST,
                        'enhanced_balanced': ProcessingMode.BALANCED,
                        'enhanced_quality': ProcessingMode.QUALITY,
                        'enhanced_ultra': ProcessingMode.ULTRA
                    }
                    processing_mode = mode_map.get(strategy_str, ProcessingMode.QUALITY)
                    
                    logger.info(f"✅ Enhanced处理器加载成功，模式: {processing_mode.value}")
                except Exception as e:
                    logger.warning(f"⚠️ Enhanced处理器加载失败: {e}，回退到AnimeGAN处理器")
                    logger.exception("Enhanced处理器加载详细错误:")
                    use_enhanced = False
                    use_gan = True
                    processing_mode = None
            
            if not use_enhanced and use_gan:
                logger.info(f"🤖 使用AnimeGAN模型处理任务 {task_id}")
                try:
                    from core.processors.animegan_processor import AnimeGANProcessor
                    processor = AnimeGANProcessor()
                    logger.info("✅ AnimeGAN处理器加载成功")
                except Exception as e:
                    logger.warning(f"⚠️ AnimeGAN处理器加载失败: {e}，回退到CV处理器")
                    processor = GhibliProcessor()
                    use_gan = False
            elif not use_enhanced:
                processor = GhibliProcessor()
            
            # 设置进度回调
            if use_enhanced and processing_mode is not None:
                # Enhanced处理器使用不同的进度回调接口
                def enhanced_progress_callback(percent, message):
                    # 将百分比转换为步骤
                    current_step = int(percent / 10)
                    total_steps = 10
                    task_manager.update_task_progress(task_id, percent, current_step, total_steps, 0)
                
                # 处理图像
                mode_str = f"Enhanced Ghibli ({processing_mode.value})"
                logger.info(f"✨ 开始处理任务 {task_id}，模式: {mode_str}")
                result = processor.process(
                    image,
                    mode=processing_mode,
                    progress_callback=enhanced_progress_callback
                )
            else:
                # 传统处理器使用原有的进度回调
                processor.set_progress_callback(
                    lambda tid, progress, current_step, total_steps, loss: 
                        task_manager.update_task_progress(tid, progress, current_step, total_steps, loss), 
                    task_id
                )
                
                # 处理图像
                if use_sd:
                    mode_str = "Stable Diffusion"
                elif use_gan:
                    mode_str = "AnimeGAN"
                else:
                    mode_str = "CV算法"
                logger.info(f"🎨 开始处理任务 {task_id}，模式: {mode_str}，策略: {strategy.value}")
                
                try:
                    result = processor.process(image, strategy=strategy)
                    # 如果 SD 或 GAN 处理失败，回退到 CV
                    if not result.success and (use_sd or use_gan):
                        logger.warning(f"⚠️ {mode_str}处理失败，回退到CV处理器")
                        processor = GhibliProcessor()
                        processor.set_progress_callback(
                            lambda tid, progress, current_step, total_steps, loss: 
                                task_manager.update_task_progress(tid, progress, current_step, total_steps, loss), 
                            task_id
                        )
                        result = processor.process(image, strategy=strategy)
                except Exception as e:
                    logger.error(f"处理器执行错误: {e}")
                    if use_sd or use_gan:
                        logger.warning(f"⚠️ {mode_str}处理异常，回退到CV处理器")
                        processor = GhibliProcessor()
                        processor.set_progress_callback(
                            lambda tid, progress, current_step, total_steps, loss: 
                                task_manager.update_task_progress(tid, progress, current_step, total_steps, loss), 
                            task_id
                        )
                        result = processor.process(image, strategy=strategy)
                    else:
                        raise
            
            # 3. 缓存已禁用 - 不保存到缓存
            # if result.success and cache_manager:
            #     try:
            #         cache_manager.set(image, strategy, result)
            #         logger.info(f"💾 任务 {task_id} 结果已保存到缓存")
            #     except Exception as e:
            #         logger.warning(f"缓存保存失败: {e}")
            logger.debug(f"缓存已禁用，不保存结果到缓存")
        
        # 4. 处理结果
        if result.success:
            result_image = result.image
        else:
            raise Exception(result.error_message)
        
        # 关键修复：先保存结果，再更新状态
        # 这确保当状态变为COMPLETED时，结果已经可用
        result_data = {
            'result_image': result_image,
            'original_image': image
        }
        
        # 计算结果大小用于日志
        import sys
        result_size_mb = sys.getsizeof(result_data) / (1024 * 1024)
        
        # 保存结果
        task_manager.set_task_result(task_id, result_data)
        logger.info(f"任务 {task_id} 结果已保存，大小: {result_size_mb:.2f}MB")
        
        # 更新任务状态为完成
        task_manager.set_task_status(task_id, TaskStatus.COMPLETED)
        logger.info(f"任务 {task_id} 状态更新为: completed")
        
        # 缓存已禁用
        logger.info(f"✅ 任务 {task_id} 宫崎骏风格转换完成（耗时: {result.processing_time:.2f}秒）")
        
    except Exception as e:
        # 记录错误信息（包含完整堆栈）
        error_msg = str(e)
        logger.exception(f"❌ 任务 {task_id} 处理失败: {error_msg}")
        logger.error(f"任务详情 - ID: {task_id}, 策略: {strategy.value if 'strategy' in locals() else 'unknown'}")
        task_manager.set_task_error(task_id, error_msg)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """健康检查端点"""
    try:
        stats = task_manager.get_stats()
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'timestamp': time.time(),
            'tasks': stats
        })
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/stats')
def get_stats():
    """获取系统统计信息"""
    try:
        stats = task_manager.get_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """获取转换进度"""
    task_info = task_manager.get_task(task_id)
    if task_info:
        return jsonify(task_info.to_dict())
    
    # 任务不存在
    return jsonify({'success': False, 'error': '任务不存在'}), 404

def ensure_task_consistency(task_id: str):
    """确保任务状态一致性"""
    task = task_manager.get_task(task_id)
    
    if task is None:
        return False, "任务不存在"
    
    # 检查状态和结果的一致性
    if task.status == TaskStatus.COMPLETED:
        if task.result is None:
            logger.error(f"❌ 任务 {task_id} 状态为completed但结果为空，状态不一致")
            # 修复：重新设置为processing
            task.status = TaskStatus.PROCESSING
            task.progress = 0
            return False, "任务状态不一致，正在修复"
    
    if task.progress >= 100 and task.status != TaskStatus.COMPLETED:
        logger.warning(f"⚠️ 任务 {task_id} 进度100%但状态不是completed，自动修复")
        # 修复：更新状态
        task.status = TaskStatus.COMPLETED
    
    return True, "状态一致"

@app.route('/result/<task_id>')
def get_result(task_id):
    """获取转换结果"""
    task_info = task_manager.get_task(task_id)
    if not task_info:
        return jsonify({'success': False, 'error': '任务不存在'}), 404
    
    # 检查任务状态一致性
    is_consistent, message = ensure_task_consistency(task_id)
    if not is_consistent:
        logger.warning(f"任务 {task_id} 状态不一致: {message}")
    
    # 检查任务状态
    if task_info.status == TaskStatus.COMPLETED:
        result_data = task_info.result
        if result_data:
            try:
                # 转换为base64
                result_image = result_data['result_image']
                
                # 检查结果类型并正确处理
                if isinstance(result_image, np.ndarray):
                    if result_image.dtype == np.float32 or result_image.dtype == np.float64:
                        result_image = (result_image * 255).astype(np.uint8)
                    result_image = Image.fromarray(result_image)
                
                # 确保结果图片是RGB模式以支持JPEG格式
                if result_image.mode in ('RGBA', 'LA', 'P'):
                    # 创建白色背景
                    background = Image.new('RGB', result_image.size, (255, 255, 255))
                    # 处理调色板模式
                    if result_image.mode == 'P':
                        result_image = result_image.convert('RGBA')
                    # 粘贴图片到白色背景上
                    if result_image.mode == 'RGBA' or result_image.mode == 'LA':
                        background.paste(result_image, mask=result_image.split()[-1] if result_image.mode == 'RGBA' else None)
                    result_image = background
                
                buffered = io.BytesIO()
                result_image.save(buffered, format="JPEG", quality=95)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # 同时返回原图
                original_image = result_data.get('original_image')
                if original_image:
                    original_buffered = io.BytesIO()
                    # 确保原图是RGB模式以支持JPEG格式
                    if original_image.mode in ('RGBA', 'LA', 'P'):
                        # 创建白色背景
                        background = Image.new('RGB', original_image.size, (255, 255, 255))
                        # 处理调色板模式
                        if original_image.mode == 'P':
                            original_image = original_image.convert('RGBA')
                        # 粘贴图片到白色背景上
                        if original_image.mode == 'RGBA' or original_image.mode == 'LA':
                            background.paste(original_image, mask=original_image.split()[-1] if original_image.mode == 'RGBA' else None)
                        original_image = background
                    original_image.save(original_buffered, format="JPEG", quality=95)
                    original_img_str = base64.b64encode(original_buffered.getvalue()).decode()
                else:
                    # 如果没有保存原图，返回默认值
                    original_img_str = ""
                
                return jsonify({
                    'success': True,
                    'result': f"data:image/jpeg;base64,{img_str}",
                    'original': f"data:image/jpeg;base64,{original_img_str}",
                    'completed': True
                })
            except Exception as e:
                error_msg = f"结果处理失败: {str(e)}"
                logger.error(error_msg)
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'completed': True
                })
        else:
            return jsonify({
                'success': False,
                'error': '结果数据丢失',
                'completed': True
            })
    elif task_info.status == TaskStatus.FAILED:
        return jsonify({
            'success': False,
            'error': task_info.error_message,
            'completed': True
        })
    elif task_info.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
        return jsonify({
            'success': False, 
            'error': '任务仍在处理中', 
            'completed': False,
            'progress': task_info.progress,
            'current_step': task_info.current_step,
            'total_steps': task_info.total_steps
        })
    
    # 其他状态
    return jsonify({'success': False, 'error': '任务状态未知', 'completed': True})

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和风格转换"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        # 检查文件类型
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'error': f'不支持的文件格式: {file_ext}，请上传 {allowed_extensions} 格式的图片'})
        
        # 检查文件大小
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置文件指针
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'success': False, 'error': f'文件太大，最大支持 {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'})
        
        # 读取图片
        try:
            image = Image.open(file.stream)
            # 处理RGBA模式图片，转换为RGB模式以支持JPEG格式
            if image.mode in ('RGBA', 'LA', 'P'):
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                # 处理调色板模式
                if image.mode == 'P':
                    image = image.convert('RGBA')
                # 粘贴图片到白色背景上
                if image.mode == 'RGBA' or image.mode == 'LA':
                    background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
        except Exception as e:
            return jsonify({'success': False, 'error': f'图片文件损坏或格式不支持: {str(e)}'})
        
        # 检查图片尺寸 - 移除尺寸限制，支持任意尺寸图片
        max_size = app.config.get('MAX_IMAGE_SIZE', 0)
        if max_size > 0 and max(image.size) > max_size * 2:  # 如果设置了最大尺寸才检查
            return jsonify({'success': False, 'error': f'图片尺寸过大，最大支持 {max_size}x{max_size} 像素'})
        
        # 记录图片尺寸信息
        print(f"📊 图片尺寸: {image.size[0]}x{image.size[1]}, 格式: {image.format}")
        
        # 保存原图用于显示
        original_buffered = io.BytesIO()
        # 确保图片是RGB模式以支持JPEG格式
        if image.mode in ('RGBA', 'LA', 'P'):
            # 创建白色背景
            background = Image.new('RGB', image.size, (255, 255, 255))
            # 处理调色板模式
            if image.mode == 'P':
                image = image.convert('RGBA')
            # 粘贴图片到白色背景上
            if image.mode == 'RGBA' or image.mode == 'LA':
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        image.save(original_buffered, format="JPEG", quality=95)
        original_img_str = base64.b64encode(original_buffered.getvalue()).decode()
        
        # 生成任务ID
        task_id = str(int(time.time() * 1000))
        
        # 创建任务
        task_manager.create_task(task_id, "image_conversion", {
            'image_size': image.size,
            'image_format': image.format
        })
        
        # 获取处理参数
        processor_type = request.form.get('processor_type', 'ghibli')  # 默认使用宫崎骏风格
        style_strategy = request.form.get('style_strategy', 'perfect')  # 默认使用完美策略
        
        # 启动异步转换
        thread = threading.Thread(
            target=convert_image_async, 
            args=(task_id, image, processor_type, style_strategy)
        )
        thread.daemon = True
        thread.start()
        
        print(f"🎨 开始异步宫崎骏风格转换，任务ID: {task_id}")
        print(f"📊 图片信息: {image.size[0]}x{image.size[1]}, 格式: {image.format}")
        print(f"⚙️ 使用处理器: {processor_type}, 策略: {style_strategy}")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'original': f"data:image/jpeg;base64,{original_img_str}",
            'message': '转换任务已开始，请等待完成',
            'estimated_time': '预计处理时间: 30-60秒'
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        
        # 分类错误信息
        if 'image file is truncated' in error_msg.lower():
            error_msg = '图片文件损坏，请重新上传'
        elif 'cannot identify image file' in error_msg.lower():
            error_msg = '无法识别图片格式，请上传有效的图片文件'
        
        logger.exception(f"❌ 转换错误: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/cache/stats')
def cache_stats():
    """获取缓存统计信息"""
    try:
        if cache_manager is None:
            return jsonify({
                'success': False,
                'error': '缓存系统未启用'
            })
        
        stats = cache_manager.get_stats()
        return jsonify({
            'success': True,
            'stats': {
                'hits': stats.hits,
                'misses': stats.misses,
                'hit_rate': f"{stats.hit_rate * 100:.2f}%",
                'total_requests': stats.total_requests,
                'memory': {
                    'items': stats.memory_items,
                    'size_mb': f"{stats.memory_size_mb:.2f}"
                },
                'disk': {
                    'items': stats.disk_items,
                    'size_mb': f"{stats.disk_size_mb:.2f}"
                }
            }
        })
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """清空缓存"""
    try:
        if cache_manager is None:
            return jsonify({
                'success': False,
                'error': '缓存系统未启用'
            })
        
        cache_manager.clear()
        logger.info("🗑️ 缓存已清空")
        
        return jsonify({
            'success': True,
            'message': '缓存已清空'
        })
    except Exception as e:
        logger.error(f"清空缓存失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/config')
def get_app_config():
    """获取应用配置信息"""
    try:
        config_dict = app_config.to_dict()
        # 移除敏感信息
        safe_config = {k: v for k, v in config_dict.items() if k not in ['debug']}
        return jsonify({
            'success': True,
            'config': safe_config
        })
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model/status')
def get_model_status():
    """获取模型缓存状态"""
    try:
        from core.components.global_model_cache import is_model_cached
        
        is_cached = is_model_cached()
        
        return jsonify({
            'success': True,
            'model_cached': is_cached,
            'message': '模型已缓存，处理速度快' if is_cached else '模型未缓存，首次处理需要加载'
        })
    except Exception as e:
        logger.error(f"获取模型状态失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model/clear', methods=['POST'])
def clear_model_cache():
    """清除模型缓存"""
    try:
        from core.components.global_model_cache import clear_global_model_cache
        
        clear_global_model_cache()
        logger.info("🗑️ 模型缓存已清除")
        
        return jsonify({
            'success': True,
            'message': '模型缓存已清除'
        })
    except Exception as e:
        logger.error(f"清除模型缓存失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/tasks')
def get_all_tasks():
    """获取所有任务信息"""
    try:
        tasks = task_manager.get_all_tasks()
        return jsonify({
            'success': True,
            'tasks': tasks
        })
    except Exception as e:
        logger.error(f"获取任务信息失败: {e}")
        return jsonify({
            'success': False,
            'error': f"获取任务信息失败: {str(e)}"
        })

@app.route('/tasks/active')
def get_active_tasks():
    """获取活跃任务信息"""
    try:
        tasks = task_manager.get_active_tasks()
        return jsonify({
            'success': True,
            'tasks': tasks
        })
    except Exception as e:
        logger.error(f"获取活跃任务信息失败: {e}")
        return jsonify({
            'success': False,
            'error': f"获取活跃任务信息失败: {str(e)}"
        })

# 全局错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': '请求的资源不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"内部服务器错误: {error}")
    return jsonify({'success': False, 'error': '服务器内部错误'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"未处理的异常: {e}")
    return jsonify({'success': False, 'error': '服务器发生未知错误'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)