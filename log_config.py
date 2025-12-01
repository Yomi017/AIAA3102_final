from loguru import logger
import sys
import os
from datetime import datetime

def setup_logger():
    """配置loguru日志系统"""
    # 移除默认handler
    logger.remove()
    
    # 创建基于时间戳的日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/session_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 控制台输出 - 彩色，简洁格式
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # 详细日志文件 - 包含所有级别
    logger.add(
        f"{log_dir}/agent.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8"
    )
    
    # 错误日志单独文件
    logger.add(
        f"{log_dir}/error.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        encoding="utf-8"
    )
    
    logger.info(f"日志系统初始化完成，日志目录: {log_dir}")
    return logger
