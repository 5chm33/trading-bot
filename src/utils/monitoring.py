from prometheus_client import Gauge, Counter, REGISTRY, start_http_server
import logging
from typing import Dict, Any
import time
import os

logger = logging.getLogger(__name__)

class TradingMonitor:
    """Lightweight metrics exporter service"""
    
    _instance = None

    def __new__(cls, port: int = 8000):
        if cls._instance is None:
            cls._instance = super(TradingMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, port: int = 8000):
        if not self._initialized:
            self.port = port
            self._init_server()
            self._initialized = True

    def _init_server(self):
        """Initialize only the metrics server"""
        # Clear existing metrics to avoid duplicates
        for metric in list(REGISTRY._collector_to_names.keys()):
            REGISTRY.unregister(metric)
            
        start_http_server(self.port)
        logger.info(f"Metrics server started on port {self.port}")