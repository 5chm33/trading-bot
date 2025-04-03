# tests/test_metrics.py
import pytest
from src.monitoring.trading_metrics import TradeMetrics, MetricDefinition
import torch
from prometheus_client import CollectorRegistry

class TestTradeMetrics:
    @pytest.fixture
    def metrics(self):
        """Fixture providing a clean metrics instance"""
        # Use fresh registry to avoid test pollution
        registry = CollectorRegistry()
        return TradeMetrics(registry=registry)

    def test_required_metrics_exist(self, metrics):
        """Verify all critical metrics are properly defined"""
        required_metrics = {
            # Core Trading Metrics
            'trading_trades_executed': (Counter, ['ticker', 'direction', 'status']),
            'trading_portfolio_value': (Gauge, ['ticker', 'currency']),
            'trading_position_size': (Gauge, ['ticker']),
            
            # Risk Metrics
            'trading_sharpe_ratio': (Gauge, None),
            'trading_max_drawdown': (Gauge, None),
            
            # System Metrics
            'system_component_init': (Counter, ['component', 'status']),
            'system_errors_total': (Counter, ['error_type', 'component']),
        }

        for name, (mtype, labels) in required_metrics.items():
            metric = getattr(metrics, name)
            assert isinstance(metric, mtype), f"{name} is not {mtype.__name__}"
            if labels:
                assert sorted(metric._labelnames) == sorted(labels), f"{name} label mismatch"

    def test_gpu_info_recording(self, metrics, mocker):
        """Test GPU metrics with mock CUDA"""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available for testing")

        mock_name = mocker.patch.object(torch.cuda, 'get_device_name', return_value="RTX3090")
        mock_props = mocker.patch.object(
            torch.cuda, 
            'get_device_properties',
            return_value=mocker.Mock(total_memory=24e9)
        )

        metrics.record_gpu_info()
        
        # Verify metric values
        samples = list(metrics.GPU_INFO._metrics.values())
        assert len(samples) == 1
        assert samples[0].labels['name'] == "RTX3090"
        assert samples[0].labels['memory'] == "24.0GB"

    def test_metric_validation(self, metrics, caplog):
        """Test error handling for invalid metric usage"""
        # Test non-existent metric
        metrics.safe_record('NON_EXISTENT_METRIC')
        assert "Metric NON_EXISTENT_METRIC not found" in caplog.text
        
        # Test missing labels
        metrics.safe_record('system_component_init', labels={'component': 'broker'})
        assert "Missing labels for system_component_init: {'status'}" in caplog.text
        
        # Test extra labels
        metrics.safe_record('trading_position_size', 
                          labels={'ticker': 'AAPL', 'extra': 'invalid'})
        assert "Extra labels for trading_position_size: {'extra'}" in caplog.text

    def test_trade_recording(self, metrics):
        """Verify trade metrics are recorded correctly"""
        metrics.record_trade('AAPL', 'buy', 'filled')
        metrics.record_trade('TSLA', 'sell', 'partial')
        
        samples = metrics.trading_trades_executed._metrics
        assert samples[('AAPL', 'buy', 'filled')]._value == 1
        assert samples[('TSLA', 'sell', 'partial')]._value == 1

    def test_portfolio_recording(self, metrics):
        """Test portfolio value updates"""
        metrics.record_portfolio_value('SPY', 10000.0, 'USD')
        assert metrics.trading_portfolio_value.labels(ticker='SPY', currency='USD')._value == 10000.0

    def test_error_handling(self, metrics):
        """Verify error metrics are properly recorded"""
        metrics.system_errors_total.labels(
            error_type='ConnectionError',
            component='broker'
        ).inc()
        
        samples = list(metrics.system_errors_total._metrics.values())
        assert samples[0]._value == 1
        assert samples[0].labels == {'error_type': 'ConnectionError', 'component': 'broker'}

    def test_histogram_metrics(self, metrics):
        """Test latency metric recording"""
        metrics.trading_rollback_latency_seconds.observe(0.5)
        metrics.trading_rollback_latency_seconds.observe(1.2)
        
        # Verify bucket counts
        assert metrics.trading_rollback_latency_seconds._buckets[0.5]._value == 1
        assert metrics.trading_rollback_latency_seconds._buckets[2.0]._value == 2

    def test_legacy_compatibility(self, metrics):
        """Verify backward compatibility layer works"""
        assert metrics.TRADE_COUNTER == metrics.trading_trades_executed
        assert metrics.PORTFOLIO_GAUGE == metrics.trading_portfolio_value
        
        # Test legacy dict-style access
        legacy_metrics = metrics.core_metrics
        assert legacy_metrics['trade_counter'] == metrics.trading_trades_executed
        assert legacy_metrics['position_gauge'] == metrics.trading_position_size