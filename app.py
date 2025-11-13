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

from config.settings import config
from config.dependency_injection import get_container
from core import ImageProcessorFactory, ProcessingStyle
from core.model_manager import model_manager
from utils.optimized_cache import optimized_cache_manager
from utils.task_manager import task_manager, TaskStatus
from utils.async_processor import get_async_processor
from utils.error_handler import handle_errors, error_handler
from utils.logging_config import setup_logging, logging_manager, perf_logger
from utils.health_check import health_monitor, setup_health_monitoring

app = Flask(__name__)

# 加载配置
app.config.from_object(config['default'])

# 设置日志系统
setup_logging(
    level=app.config.get('MONITORING', {}).get('log_level', 'INFO'),
    enable_console=True,
    enable_file=True,
    enable_structured=True
)

# 设置健康监控
setup_health_monitoring(model_manager, optimized_cache_manager)

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 配置依赖注入
container = get_container()
container.register_instance(type(model_manager), model_manager)
container.register_instance(type(optimized_cache_manager), optimized_cache_manager)
container.register_instance(type(task_manager), task_manager)

def convert_image_async(task_id, image, processor_type="ghibli", style_intensity=1.0):
    """异步转换图像 - 使用新的宫崎骏风格转换器
    
    Args:
        task_id: 任务ID
        image: 输入图像
        processor_type: 处理器类型 (固定为"ghibli")
        style_intensity: 风格强度 (忽略，固定使用标准宫崎骏风格)
    """
    try:
        # 更新任务状态为处理中
        task_manager.set_task_status(task_id, TaskStatus.PROCESSING)
        task_manager.update_task_progress(task_id, 5, 1, 10, 0)
        
        # 使用新的宫崎骏风格转换器
        from core.true_ghibli_style import true_ghibli_processor
        
        # 设置进度回调
        true_ghibli_processor.set_progress_callback(
            lambda tid, progress, current_step, total_steps, loss: 
                task_manager.update_task_progress(tid, progress, current_step, total_steps, loss), 
            task_id
        )
        
        # 应用宫崎骏风格
        result_image = true_ghibli_processor.apply_ghibli_style(image)
        
        # 保存结果
        task_manager.set_task_result(task_id, {
            'result_image': result_image,
            'original_image': image
        })
        
        # 更新任务状态为完成
        task_manager.set_task_status(task_id, TaskStatus.COMPLETED)
        task_manager.update_task_progress(task_id, 100, 10, 10, 0)
        
        print(f"✅ 任务 {task_id} 宫崎骏风格转换完成")
        
    except Exception as e:
        # 记录错误信息
        error_msg = str(e)
        logger.error(f"任务 {task_id} 处理失败: {error_msg}")
        task_manager.set_task_error(task_id, error_msg)
        print(f"❌ 任务 {task_id} 转换失败: {error_msg}")

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """获取转换进度"""
    task_info = task_manager.get_task(task_id)
    if task_info:
        return jsonify(task_info.to_dict())
    
    # 任务不存在
    return jsonify({'success': False, 'error': '任务不存在'}), 404

@app.route('/result/<task_id>')
def get_result(task_id):
    """获取转换结果"""
    task_info = task_manager.get_task(task_id)
    if not task_info:
        return jsonify({'success': False, 'error': '任务不存在'}), 404
    
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
                
                buffered = io.BytesIO()
                result_image.save(buffered, format="JPEG", quality=95)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # 同时返回原图
                original_image = result_data.get('original_image')
                if original_image:
                    original_buffered = io.BytesIO()
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
        image.save(original_buffered, format="JPEG", quality=95)
        original_img_str = base64.b64encode(original_buffered.getvalue()).decode()
        
        # 生成任务ID
        task_id = str(int(time.time() * 1000))
        
        # 创建任务
        task_manager.create_task(task_id, "image_conversion", {
            'image_size': image.size,
            'image_format': image.format
        })
        
        # 固定使用宫崎骏风格
        processor_type = "ghibli"  # 固定使用宫崎骏风格
        style_intensity = 1.0  # 固定风格强度
        
        # 启动异步转换
        thread = threading.Thread(
            target=convert_image_async, 
            args=(task_id, image, processor_type, style_intensity)
        )
        thread.daemon = True
        thread.start()
        
        print(f"🎨 开始异步宫崎骏风格转换，任务ID: {task_id}")
        print(f"📊 图片信息: {image.size[0]}x{image.size[1]}, 格式: {image.format}")
        print(f"⚙️ 使用标准宫崎骏风格转换器")
        
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
        
        logger.error(f"❌ 转换错误: {error_msg}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': error_msg})

@app.route('/cache/stats')
def cache_stats():
    """获取缓存统计信息"""
    try:
        stats = optimized_cache_manager.get_cache_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"获取缓存统计信息失败: {e}")
        return jsonify({
            'success': False,
            'error': f"获取缓存统计信息失败: {str(e)}"
        })

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """清空缓存"""
    try:
        optimized_cache_manager.clear_cache()
        return jsonify({
            'success': True,
            'message': '缓存已清空'
        })
    except Exception as e:
        logger.error(f"清空缓存失败: {e}")
        return jsonify({
            'success': False,
            'error': f"清空缓存失败: {str(e)}"
        })

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