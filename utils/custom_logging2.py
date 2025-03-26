# logging.py
import logging
import sys
import os

def setup_logger():
    """Set up the logger."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create a file handler
    file_handler = logging.FileHandler('trading_bot.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger