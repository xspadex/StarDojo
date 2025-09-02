# my_app_logging.py

import logging
import os
from pathlib import Path

# --- 这部分代码在模块被导入时，只会被执行一次 ---

logger_name = 'stardojo_logger'
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# 检查是否已经有handlers，防止在某些特殊情况下重复添加
if not logger.hasHandlers():
    log_directory = Path("backup_logs")
    log_directory.mkdir(exist_ok=True)

    pid = os.getpid()
    log_file_path = log_directory / f"stardojo_pid_{pid}.log"
    
    handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)

# --- 最终，我们导出一个已经配置好的logger实例 ---
# 为了方便，可以直接导出这个实例
stardojo_log = logger