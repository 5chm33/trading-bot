<<<<<<< HEAD
from prometheus_client import Gauge, Counter, REGISTRY, start_http_server
import logging
from typing import Dict, Any
import time
import os

logger = logging.getLogger(__name__)

class TradingMonitor:
    """Enhanced metrics exporter with proper port handling"""

    _instance = None
    DEFAULT_PORT = 8000  # Class-level default

    def __new__(cls, config: Dict = None):
        if cls._instance is None:
            cls._instance = super(TradingMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Dict = None):
        if not self._initialized:
            self.port = self._get_port_from_config(config)
            self._init_server()
            self._initialized = True

    def _get_port_from_config(self, config: Dict) -> int:
        """Extract port from config with fallbacks"""
        if config is None:
            return self.DEFAULT_PORT

        try:
            # Try getting port from monitoring config section
            port = config.get('monitoring', {}).get('port', self.DEFAULT_PORT)
            return self._validate_port(port)
        except Exception as e:
            logger.warning(f"Failed to parse port from config: {str(e)}")
            return self.DEFAULT_PORT

    def _validate_port(self, port: Any) -> int:
        """Ensure port is valid integer within range"""
        try:
            port = int(port) if port is not None else self.DEFAULT_PORT
            assert 1024 <= port <= 65535
            return port
        except (ValueError, AssertionError):
            logger.warning(
                f"Invalid port {port}, using default {self.DEFAULT_PORT}",
                exc_info=True
            )
            return self.DEFAULT_PORT

    def _init_server(self):
        """Safe server initialization with cleanup"""
        # Clear existing metrics
        for metric in list(REGISTRY._collector_to_names.keys()):
            REGISTRY.unregister(metric)

        try:
            start_http_server(self.port)
            logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            logger.critical(
                f"Failed to start metrics server on port {self.port}: {str(e)}",
                exc_info=True
            )
            raise
=======
from prometheus_client import Gauge, Counter, REGISTRY, start_http_server
import logging
from typing import Dict, Any
import time
import os

logger = logging.getLogger(__name__)

class TradingMonitor:
    """Enhanced metrics exporter with proper port handling"""
    
    _instance = None
    DEFAULT_PORT = 8000  # Class-level default

    def __new__(cls, config: Dict = None):
        if cls._instance is None:
            cls._instance = super(TradingMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Dict = None):
        if not self._initialized:
            self.port = self._get_port_from_config(config)
            self._init_server()
            self._initialized = True

    def _get_port_from_config(self, config: Dict) -> int:
        """Extract port from config with fallbacks"""
        if config is None:
            return self.DEFAULT_PORT
            
        try:
            # Try getting port from monitoring config section
            port = config.get('monitoring', {}).get('port', self.DEFAULT_PORT)
            return self._validate_port(port)
        except Exception as e:
            logger.warning(f"Failed to parse port from config: {str(e)}")
            return self.DEFAULT_PORT

    def _validate_port(self, port: Any) -> int:
        """Ensure port is valid integer within range"""
        try:
            port = int(port) if port is not None else self.DEFAULT_PORT
            assert 1024 <= port <= 65535
            return port
        except (ValueError, AssertionError):
            logger.warning(
                f"Invalid port {port}, using default {self.DEFAULT_PORT}",
                exc_info=True
            )
            return self.DEFAULT_PORT

    def _init_server(self):
        """Safe server initialization with cleanup"""
        # Clear existing metrics
        for metric in list(REGISTRY._collector_to_names.keys()):
            REGISTRY.unregister(metric)
            
        try:
            start_http_server(self.port)
            logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            logger.critical(
                f"Failed to start metrics server on port {self.port}: {str(e)}",
                exc_info=True
            )
            raise
>>>>>>> 60870aec3b9ed2c2cb804ceb4f1eeb5c6af9d852
