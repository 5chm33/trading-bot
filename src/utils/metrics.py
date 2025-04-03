from prometheus_client import Counter, Gauge

# Trading Execution Metrics
TRADE_COUNTER = Counter(
    'trading_trades_executed',
    'Count of trades by direction and ticker',
    ['ticker', 'direction']
)

# Portfolio Metrics
PORTFOLIO_GAUGE = Gauge(
    'trading_portfolio_value',
    'Current portfolio value in USD',
    ['ticker']
)

POSITION_GAUGE = Gauge(
    'trading_position_size',
    'Current position size [-1,1]',
    ['ticker']
)

# Risk Metrics
SHARPE_GAUGE = Gauge(
    'trading_sharpe_ratio',
    '21-day rolling Sharpe ratio'
)

DRAWDOWN_GAUGE = Gauge(
    'trading_max_drawdown',
    'Current portfolio drawdown percentage'
)