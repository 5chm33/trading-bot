import logging
from logging.handlers import RotatingFileHandler
import coloredlogs  # pip install coloredlogs
import yaml
import os

def setup_logger(name):
    """Creates a configured logger instance with settings from config.yaml"""
    
    # 1. Load logging config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)['logging']
    
    # 2. Create logger
    logger = logging.getLogger(name)
    logger.setLevel(config['level'])
    
    # 3. File Handler (Rotating logs)
    if config['file']['enabled']:
        os.makedirs(os.path.dirname(config['file']['path']), exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=config['file']['path'],
            maxBytes=config['file']['max_size'] * 1024 * 1024,  # Convert MB to bytes
            backupCount=config['file']['backups']
        )
        file_handler.setFormatter(logging.Formatter(config['format']))
        logger.addHandler(file_handler)
    
    # 4. Console Handler (Colored output)
    if config['console']['enabled'] and config['console']['colors']:
        coloredlogs.install(
            level=config['level'],
            fmt=config['format'],
            logger=logger
        )
    elif config['console']['enabled']:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(config['format']))
        logger.addHandler(console_handler)
    
    return logger