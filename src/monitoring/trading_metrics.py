# src/monitoring/trading_metrics.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from prometheus_client import Counter, Gauge, Histogram, REGISTRY
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetricDefinition:
    name: str
    description: str
    metric_class: type
    labels: Optional[List[str]] = None
    buckets: Optional[List[float]] = None

class TradeMetrics:
    """Complete trading metrics implementation with all required methods"""
    
    _shared_registry = {}
    
    METRIC_SPECS = [
        # Trading Metrics
        MetricDefinition(
            name='trading_trades',
            description='Trade executions by ticker/direction/status',
            metric_class=Counter,
            labels=['ticker', 'direction', 'status']
        ),
        MetricDefinition(
            name='trading_portfolio_value',
            description='Current portfolio value',
            metric_class=Gauge,
            labels=['ticker', 'currency']
        ),
        MetricDefinition(
            name='trading_position_size',
            description='Current position size',
            metric_class=Gauge,
            labels=['ticker']
        ),
        MetricDefinition(
            name='trading_price_updates',
            description='Price updates received',
            metric_class=Counter
        ),
        
        # Risk Metrics
        MetricDefinition(
            name='trading_sharpe_ratio',
            description='21-day rolling Sharpe ratio',
            metric_class=Gauge
        ),
        MetricDefinition(
            name='trading_max_drawdown',
            description='Current portfolio drawdown percentage',
            metric_class=Gauge
        ),
        
        # System Metrics
        MetricDefinition(
            name='system_component_init',
            description='Component initialization status',
            metric_class=Counter,
            labels=['component', 'status']
        ),
        MetricDefinition(
            name='system_broker_init',
            description='Broker initialization status',
            metric_class=Counter,
            labels=['status']
        ),
        MetricDefinition(
            name='system_env_init',
            description='Environment initialization status',
            metric_class=Counter,
            labels=['env_type', 'status']
        ),
        MetricDefinition(
            name='system_processor_init',
            description='Feature processor initialization',
            metric_class=Counter,
            labels=['mode']
        ),
        MetricDefinition(
            name='system_processor_failures',
            description='Feature processor failures',
            metric_class=Counter,
            labels=['stage']
        ),
        MetricDefinition(
            name='system_startup',
            description='System startup status',
            metric_class=Counter,
            labels=['version', 'mode', 'status']
        ),
        MetricDefinition(
            name='system_broker_errors',
            description='Broker API errors',
            metric_class=Counter,
            labels=['error_type', 'method']
        ),
        MetricDefinition(
            name='system_state_failures',
            description='State retrieval failures',
            metric_class=Counter
        ),
        
        # Execution Metrics
        MetricDefinition(
            name='execution_latency',
            description='Operation latency in seconds',
            metric_class=Histogram,
            buckets=[0.1, 0.5, 1, 2, 5]
        ),
        
        # RL Agent Metrics
        MetricDefinition(
            name='rl_decision_failures',
            description='RL agent decision failures',
            metric_class=Counter,
            labels=['error_type']
        ),
        
        # Session Metrics
        MetricDefinition(
            name='session_status',
            description='Trading session status',
            metric_class=Counter,
            labels=['status']
        )
    ]

    def __init__(self, registry=None):
        self.registry = registry or REGISTRY
        self._initialize_metrics()
        self._add_compatibility_layer()
        logger.info("Metrics system initialized")

    def _initialize_metrics(self):
        """Create all metric instances from specifications"""
        for spec in self.METRIC_SPECS:
            try:
                if spec.name in self._shared_registry:
                    setattr(self, spec.name, self._shared_registry[spec.name])
                    continue
                    
                kwargs = {
                    'name': spec.name,
                    'documentation': spec.description,
                    'labelnames': spec.labels or [],
                    'registry': self.registry
                }
                
                if spec.metric_class == Histogram:
                    kwargs['buckets'] = spec.buckets or [0.1, 0.5, 1, 2, 5]
                
                metric = spec.metric_class(**kwargs)
                self._shared_registry[spec.name] = metric
                setattr(self, spec.name, metric)
                
            except Exception as e:
                logger.error(f"Failed to initialize metric {spec.name}", exc_info=True)
                raise

    def _add_compatibility_layer(self):
        """Maintain backward compatibility with legacy metric names"""
        self.TRADE_COUNTER = self.trading_trades
        self.PORTFOLIO_GAUGE = self.trading_portfolio_value
        self.POSITION_GAUGE = self.trading_position_size
        self.SHARPE_GAUGE = self.trading_sharpe_ratio
        self.DRAWDOWN_GAUGE = self.trading_max_drawdown
        self.COMPONENT_INIT = self.system_component_init
        self.BROKER_INIT = self.system_broker_init
        self.ENV_INIT = self.system_env_init
        self.PROCESSOR_INIT = self.system_processor_init
        self.PROCESSOR_INIT_FAILURES = self.system_processor_failures
        self.SYSTEM_STARTUP = self.system_startup
        self.BROKER_ERRORS = self.system_broker_errors
        self.STATE_FAILURE = self.system_state_failures
        self.DECISION_FAILURES = self.rl_decision_failures
        self.SESSION_STATUS = self.session_status
        self.PRICE_UPDATE = self.trading_price_updates
        
        self.TRADE_SUCCESS = _LegacyTradeMetric(self.trading_trades, 'success')
        self.TRADE_FAILURE = _LegacyTradeMetric(self.trading_trades, 'failed')

    # Unified metric recording methods
    def record_trade(self, ticker: str, direction: str, status: str):
        self.trading_trades.labels(ticker=ticker, direction=direction, status=status).inc()

    def record_portfolio_value(self, ticker: str, value: float, currency: str = 'USD'):
        self.trading_portfolio_value.labels(ticker=ticker, currency=currency).set(value)

    def record_position(self, ticker: str, size: float):
        self.trading_position_size.labels(ticker=ticker).set(size)

    def record_price_update(self):
        self.trading_price_updates.inc()

    def record_component_init(self, component: str, success: bool):
        self.system_component_init.labels(component=component, status='success' if success else 'failed').inc()

    def record_broker_init(self, success: bool):
        self.system_broker_init.labels(status='success' if success else 'failed').inc()

    def record_broker_error(self, error_type: str, method: str):
        self.system_broker_errors.labels(error_type=error_type, method=method).inc()

    def record_env_init(self, env_type: str, success: bool):
        self.system_env_init.labels(env_type=env_type, status='success' if success else 'failed').inc()

    def record_processor_init(self, mode: str):
        self.system_processor_init.labels(mode=mode).inc()

    def record_processor_failure(self, stage: str):
        self.system_processor_failures.labels(stage=stage).inc()

    def record_system_startup(self, version: str, mode: str, success: bool):
        self.system_startup.labels(version=version, mode=mode, status='success' if success else 'failed').inc()

    def record_state_failure(self):
        self.system_state_failures.inc()

    def record_decision_failure(self, error_type: str):
        self.rl_decision_failures.labels(error_type=error_type).inc()

    def record_session_status(self, status: str):
        self.session_status.labels(status=status).inc()

class _LegacyTradeMetric:
    """Wrapper class to support both .labels() and direct calling"""
    def __init__(self, metric, status):
        self.metric = metric
        self.status = status
    
    def __call__(self, ticker: str, side: str):
        self.metric.labels(ticker=ticker, direction=side, status=self.status).inc()
    
    def labels(self, ticker: str, side: str):
        self(ticker, side)