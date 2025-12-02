"""
Enhanced logging configuration for Ghibli Quality Enhancement
"""
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_enhanced_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_format: str = None
) -> logging.Logger:
    """
    Setup enhanced logging for the processor
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("enhanced_ghibli_processor")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_processing_start(logger: logging.Logger, image_size: tuple, mode: str, content_type: str = None):
    """Log processing start with input details"""
    logger.info(
        f"Processing started - Size: {image_size}, Mode: {mode}, Content: {content_type or 'unknown'}"
    )


def log_model_loading(logger: logging.Logger, base_model: str, lora_enabled: bool, controlnet_enabled: bool):
    """Log model loading details"""
    logger.debug(
        f"Loading models - Base: {base_model}, LoRA: {lora_enabled}, ControlNet: {controlnet_enabled}"
    )


def log_generation_params(logger: logging.Logger, strength: float, steps: int, guidance: float, scheduler: str):
    """Log generation parameters"""
    logger.info(
        f"Generation params - Strength: {strength}, Steps: {steps}, Guidance: {guidance}, Scheduler: {scheduler}"
    )


def log_processing_complete(logger: logging.Logger, processing_time: float, quality_score: float = None):
    """Log processing completion"""
    msg = f"Processing complete - Time: {processing_time:.2f}s"
    if quality_score is not None:
        msg += f", Quality: {quality_score:.1f}"
    logger.info(msg)


def log_processing_error(logger: logging.Logger, stage: str, error: Exception):
    """Log processing error with context"""
    logger.error(
        f"Processing failed at {stage}: {type(error).__name__}: {str(error)}",
        exc_info=True
    )
