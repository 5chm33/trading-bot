# Metrics Reference Documentation

## System Metrics

### SYSTEM_STARTUP
- **Type**: Counter
- **Labels**:
  - `version`: System version string
  - `mode`: Operation mode ("paper" or "live")
- **Purpose**: Tracks system startup events
- **Recorded in**: Main application initialization

### COMPONENT_INIT
- **Type**: Counter
- **Labels**:
  - `component`: Component name ("broker", "env", "agent")
  - `status`: Initialization status ("success" or "failed")
- **Purpose**: Tracks component initialization results
- **Recorded in**: Each component's initialization

## Hardware Metrics

### GPU_INFO
- **Type**: Gauge
- **Labels**:
  - `name`: GPU model name (e.g. "NVIDIA RTX 3090")
  - `memory`: GPU memory in GB (e.g. "24.0")
- **Purpose**: Tracks GPU hardware information
- **Recorded in**: Agent initialization when GPU is available

## Feature Processing Metrics

### PROCESSOR_INIT
- **Type**: Counter
- **Labels**:
  - `mode`: Processor mode ("live" or "backtest")
- **Purpose**: Tracks feature processor initializations
- **Recorded in**: Feature processor initialization

## Environment Metrics

### ENV_INIT
- **Type**: Counter
- **Labels**:
  - `env_type`: Environment type ("backtest" or "live")
- **Purpose**: Tracks environment initializations
- **Recorded in**: Environment initialization

## Session Metrics

### SESSION_START
- **Type**: Counter
- **Purpose**: Tracks trading session starts
- **Recorded in**: Session manager

### SESSION_END
- **Type**: Gauge
- **Purpose**: Records last session end timestamp
- **Recorded in**: Session shutdown

## Trading Execution Metrics

### TRADE_SUCCESS
- **Type**: Counter
- **Labels**:
  - `ticker`: Ticker symbol (e.g. "AAPL")
  - `side`: Trade side ("buy" or "sell")
- **Purpose**: Counts successful trade executions
- **Recorded in**: Trade execution handler

### TRADE_FAILURE
- **Type**: Counter
- **Labels**:
  - `ticker`: Ticker symbol
  - `reason`: Failure reason ("timeout", "rejected", etc.)
- **Purpose**: Counts failed trade executions
- **Recorded in**: Trade error handler

## Performance Metrics

### CYCLE_TIME
- **Type**: Histogram
- **Buckets**: [0.1, 0.5, 1, 2, 5] seconds
- **Purpose**: Measures trading cycle duration
- **Recorded in**: Main trading loop

## Agent Metrics

### AGENT_INIT
- **Type**: Gauge
- **Labels**:
  - `device`: Device type ("cpu" or "cuda")
  - `policy`: Policy name (e.g. "MultiInputPolicy")
  - `version`: Agent version string
- **Purpose**: Tracks agent initialization status
- **Values**:
  - 1: Initialized successfully
  - 0: Initialization failed
- **Recorded in**: Agent initialization

## Risk Management Metrics

### RISK_REDUCTIONS
- **Type**: Counter
- **Labels**:
  - `trigger`: Reduction trigger ("volatility", "drawdown", etc.)
- **Purpose**: Counts risk-induced position reductions
- **Recorded in**: Risk management system

## Portfolio Metrics

### PORTFOLIO_VALUE
- **Type**: Gauge
- **Purpose**: Current portfolio value in base currency
- **Recorded in**: Portfolio manager updates

## Shutdown Metrics

### SHUTDOWN_SUCCESS
- **Type**: Counter
- **Purpose**: Counts successful shutdowns
- **Recorded in**: Shutdown sequence

## Metric Types Guide

| Type      | Best For                          | Example Use Cases               |
|-----------|-----------------------------------|--------------------------------|
| Counter   | Cumulative events                 | Trades executed, errors counted |
| Gauge     | Current values                    | Portfolio value, positions      |
| Histogram | Measurements with distribution    | Latency, processing times       |

## Best Practices

1. **Use Specific Recorders**: Prefer methods like `record_trade()` over raw `safe_record()`
2. **Validate Early**: Check metrics during initialization
3. **Document Changes**: Update both METRIC_SPECS and reference docs
4. **Monitor Cardinality**: Keep label values bounded
5. **Standardize Labels**: Use consistent label values across recordings