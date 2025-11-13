#!/usr/bin/env python3
"""
优化后的Flask应用主文件
集成所有性能优化和架构改进
"""

import io
import base64
from flask import Flask, render_template, request, jsonify
import os
import time
import threading
from PIL import Image
import numpy as np
import logging

# 优化组件导入
from config.settings import config
from config.dependency_injection import get_container
from core import ImageProcessorFactory, ProcessingStyle
from core.model_manager import model_manager
from utils.optimized_cache import optimized_cache_manager
from utils.task_manager import task_manager, TaskStatus
from utils.async_processor import get_async_processor
from utils.error_handler import handle_errors, error_handler, ProcessingError
from utils.logging_config import setup_logging, logging_manager, perf_logger
from utils.health_check import health_monitor, setup_health_monitoring

# 创建Flask应用
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

logger = logging.getLogger(__name__)

# 设置健康监控
setup_health_monitoring(model_manager, optimized_cache_manager)

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 配置依赖注入
container = get_container()
container.register_instance(type(model_manager), model_manager)
container.register_instance(type(optimized_cache_manager), optimized_cache_manager)
container.register_instance(type(task_manager), task_manager)

# 获取异步处理器
async_processor = get_async_processor(
    max_workers=app.config.get('PERFORMANCE', {}).get('max_workers'),
    use_process_pool=app.config.get('PERFORMANCE', {}).get('use_process_pool', False)
)

@handle_errors({'component': 'image_conversion'})
@perf_logger('image_conversion')
def convert_image_async(task_id, image, processor_type="enhanced", style_intensity=1.0):
    """优化的异步图像转换"""
    try:
        # 更新任务状态为处理中
        task_manager.set_task_status(task_id, TaskStatus.PROCESSING)
        task_manager.update_task_progress(task_id, 5, 1, 20, 0)
        
        # 根据类型选择处理器
        if processor_type == "enhanced":
            style_type = ProcessingStyle.GHIBLI_ENHANCED
            params = {
                "use_face_enhancement": True,
                "use_bg_separation": True
            }
        elif processor_type == "neural":
            style_type = ProcessingStyle.GHIBLI_NEURAL
            params = {
                "num_steps": 100,
                "style_weight": int(300000 * style_intensity),
                "content_weight": 1,
                "use_neural": True
            }
        else:  # classic
            style_type = ProcessingStyle.GHIBLI_CLASSIC
            params = {
                "num_steps": 80,
                "style_weight": int(300000 * style_intensity),
                "content_weight": 1,
                "use_neural": False
            }
        
        # 检查优化缓存
        cached_result = optimized_cache_manager.get_cached_result(image, processor_type, params)
        if cached_result:
            result_image = cached_result
            task_manager.update_task_progress(task_id, 90, 18, 20, 0)
            logger.info(f"Cache hit for task {task_id}")
        else:
            # 创建处理器
            processor = ImageProcessorFactory.create_processor(style_type)
            processor.set_progress_callback(
                lambda tid, progress, current_step, total_steps, loss: 
                task_manager.update_task_progress(tid, progress, current_step, total_steps, loss), 
                task_id
            )
            
            # 处理图像
            result = processor.process(image, **params)
            
            if not result.success:
                raise ProcessingError(result.error_message, error_handler.ErrorCode.PROCESSING_ERROR)
            
            result_image = result.image
            
            # 保存到优化缓存
            optimized_cache_manager.save_result(image, result_image, processor_type, params)
            logger.info(f"Processed and cached result for task {task_id}")
        
        # 保存结果
        task_manager.set_task_result(task_id, {
            'result_image': result_image,
            'original_image': image
        })
        
        # 更新任务状态为完成
        task_manager.set_task_status(task_id, TaskStatus.COMPLETED)
        task_manager.update_task_progress(task_id, 100, 20, 20, 0)
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Task {task_id} processing failed: {error_msg}")
        task_manager.set_task_error(task_id, error_msg)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """健康检查端点"""
    try:
        health_report = health_monitor.run_checks()
        status_code = 200 if health_report['overall_status'] == 'healthy' else 503
        return jsonify(health_report), status_code
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'overall_status': 'critical',
            'error': str(e),
            'timestamp': time.time()
        }), 503

@app.route('/metrics')
def metrics():
    """系统指标端点"""
    try:
        metrics_data = {
            'system': {
                'memory_info': model_manager.get_memory_info(),
                'cache_stats': optimized_cache_manager.get_cache_stats(),
                'task_stats': {
                    'active_tasks': len(task_manager.get_active_tasks()),
                    'total_tasks': len(task_manager.get_all_tasks())
                },
                'error_stats': error_handler.get_error_stats(),
                'async_processor_stats': async_processor.get_system_stats()
            },
            'health_summary': health_monitor.get_health_summary(),
            'log_stats': logging_manager.get_log_stats()
        }
        
        return jsonify(metrics_data)
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """获取转换进度"""
    task_info = task_manager.get_task(task_id)
    if task_info:
        return jsonify(task_info.to_dict())
    
    return jsonify({'success': False, 'error': '任务不存在'}), 404

@app.route('/result/<task_id>')
def get_result(task_id):
    """获取转换结果"""
    task_info = task_manager.get_task(task_id)
    if not task_info:
        return jsonify({'success': False, 'error': '任务不存在'}), 404
    
    if task_info.status == TaskStatus.COMPLETED:
        result_data = task_info.result
        if result_data:
            try:
                # 转换为base64
                result_image = result_data['result_image']
                
                if isinstance(result_image, np.ndarray):
                    if result_image.dtype == np.float32 or result_image.dtype == np.float64:
                        result_image = (result_image * 255).astype(np.uint8)
                    result_image = Image.fromarray(result_image)
                
                buffered = io.BytesIO()
                result_image.save(buffered, format="JPEG", quality=95)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # 原图处理
                original_image = result_data.get('original_image')
                original_img_str = ""
                if original_image:
                    original_buffered = io.BytesIO()
                    original_image.save(original_buffered, format="JPEG", quality=95)
                    original_img_str = base64.b64encode(original_buffered.getvalue()).decode()
                
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
    
    return jsonify({'success': False, 'error': '任务状态未知', 'completed': True})

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和风格转换"""
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
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'success': False, 'error': f'文件太大，最大支持 {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'})
    
    # 读取图片
    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({'success': False, 'error': f'图片文件损坏或格式不支持: {str(e)}'})
    
    # 记录图片信息
    logger.info(f"Processing image: {image.size[0]}x{image.size[1]}, format: {image.format}")
    
    # 保存原图用于显示
    original_buffered = io.BytesIO()
    image.save(original_buffered, format="JPEG", quality=95)
    original_img_str = base64.b64encode(original_buffered.getvalue()).decode()
    
    # 生成任务ID
    task_id = str(int(time.time() * 1000))
    
    # 创建任务
    task_manager.create_task(task_id, "image_conversion", {
        'image_size': image.size,
        'image_format': image.format,
        'file_size': file_size
    })
    
    # 获取处理参数
    processor_type = request.form.get('processor_type', 'enhanced')
    style_intensity = float(request.form.get('style_intensity', 1.0))
    
    # 验证参数
    if processor_type not in ['classic', 'enhanced', 'neural']:
        processor_type = 'enhanced'
        
    if not (0.5 <= style_intensity <= 2.0):
        style_intensity = 1.0
    
    # 启动异步转换
    thread = threading.Thread(
        target=convert_image_async, 
        args=(task_id, image, processor_type, style_intensity)
    )
    thread.daemon = True
    thread.start()
    
    logger.info(f"Started async processing for task {task_id}")
    
    return jsonify({
        'success': True,
        'task_id': task_id,
        'original': f"data:image/jpeg;base64,{original_img_str}",
        'message': '转换任务已开始，请等待完成',
        'estimated_time': '预计处理时间: 30-60秒'
    })

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
    optimized_cache_manager.clear_cache()
    return jsonify({
        'success': True,
        'message': '缓存已清空'
    })

# 全局错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': '请求的资源不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"内部服务器错误: {error}")
    return jsonify({'success': False, 'error': '服务器内部错误'}), 500

@app.errorhandler(ProcessingError)
def handle_processing_error(error):
    logger.error(f"处理错误: {error.message}")
    return jsonify({
        'success': False, 
        'error': error.message,
        'error_code': error.error_code.value
    }), 400

if __name__ == '__main__':
    logger.info("Starting optimized Flask application")
    app.run(debug=False, host='0.0.0.0', port=5003)