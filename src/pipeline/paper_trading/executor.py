# src/pipeline/paper_trading/executor.py
import time
import json
import numpy as np
from typing import Dict, Any, Optional
from prometheus_client import start_http_server
from src.brokers.alpaca.paper import AlpacaPaperBroker
from src.utils.logging import setup_logger
from src.utils.config_loader import ConfigLoader
from src.agents.rl_agent import RLAgent
from src.models.rl.env import LiveTradingEnv
from src.monitoring.trading_metrics import TradeMetrics

logger = setup_logger(__name__)

class PaperTradingExecutor:
    def __init__(self, config_path: str = "config/config.yaml"):
        start_http_server(8000)
        self.config = ConfigLoader.load_config(config_path)
        self._validate_config()
        self.metrics = TradeMetrics()  # Single metrics instance
        self._cycle_count = 0
        self._initialize_components()
        self.test_connectivity()
        logger.info("Paper trading executor initialized")
        self.metrics.record_system_startup(
            version=self.config.get('meta', {}).get('version', 'unknown'),
            mode='paper',
            success=True
        )

    def test_connectivity(self):
        try:
            balance = self.broker.get_account_balance()
            prices = self.broker.get_current_prices()
            logger.info(f"✅ Connectivity OK | Balance: ${balance:,.2f} | Latest Prices: {prices}")
            return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {str(e)}")
            return False

    def _validate_config(self):
        required_sections = ['trading', 'brokers', 'tickers']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing config section: {section}")

    def _initialize_components(self):
        try:
            self.broker = AlpacaPaperBroker(
                config=self.config,
                metrics=self.metrics
            )
            self.env = LiveTradingEnv(
                broker=self.broker,
                config=self.config,
                core_metrics=self.metrics,  # Changed from metrics to core_metrics
                agent_metrics=self.metrics
            )
            self.agent = RLAgent(
                env=self.env,
                config=self.config,
                metrics=self.metrics
            )
            self.metrics.record_component_init(
                component='full_stack',
                success=True
            )
            logger.info("All components initialized successfully")
        except Exception as e:
            self.metrics.record_component_init(
                component=self._identify_failed_component(str(e)),
                success=False
            )
            logger.critical(f"Component initialization failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize trading components: {str(e)}") from e

    def _identify_failed_component(self, error_msg: str) -> str:
        error_msg = error_msg.lower()
        if 'broker' in error_msg:
            return 'broker'
        elif 'environment' in error_msg or 'env' in error_msg:
            return 'environment'
        elif 'agent' in error_msg or 'rl' in error_msg:
            return 'agent'
        return 'unknown'

    def run(self, interval: int = 60) -> None:
        logger.info(f"Starting trading session (interval={interval}s)")
        self.metrics.record_trade(
            ticker='system',
            direction='start',
            status='session'
        )
        
        try:
            while True:
                cycle_start = time.time()
                self._cycle_count += 1
                
                with self.metrics.execution_latency.time():
                    state = self._get_market_state()
                    action = self._get_trading_action(state)
                    
                    if self._should_execute(action):
                        self._execute_trade(action)
                    
                    self._monitor_cycle(cycle_start, interval)
                    
        except KeyboardInterrupt:
            logger.info("Graceful shutdown initiated")
            self._shutdown()
        except Exception as e:
            self.metrics.record_trade(
                ticker='system',
                direction='error',
                status='failed'
            )
            logger.error(f"Trading session failed: {str(e)}")
            raise

    def _get_market_state(self) -> Dict[str, Any]:
        try:
            return self.env.get_state()
        except Exception as e:
            self.metrics.record_trade(
                ticker='system',
                direction='state',
                status='failed'
            )
            logger.error(f"Failed to get market state: {str(e)}")
            raise

    def _get_trading_action(self, state: Dict) -> Optional[Dict]:
        try:
            logger.debug(f"Market State:\n{json.dumps(state, indent=2)}")
            
            with self.metrics.execution_latency.time():
                raw_action = self.agent.decide(state)
                logger.debug(f"Raw Agent Output: {raw_action} (Type: {type(raw_action)})")
                
                if isinstance(raw_action, np.ndarray):
                    primary_ticker = self.config['tickers']['primary'][0]
                    action = {
                        primary_ticker: float(raw_action[0]),
                        'raw_array': raw_action.tolist()
                    }
                    logger.info(f"Processed Action: {action}")
                    return action
                elif isinstance(raw_action, dict):
                    logger.info(f"Direct Action: {raw_action}")
                    return raw_action
                else:
                    raise ValueError(f"Unexpected action type: {type(raw_action)}")
                
        except Exception as e:
            self.metrics.record_trade(
                ticker='system',
                direction='decision',
                status='failed'
            )
            logger.error(f"Decision Pipeline Failed: {str(e)}", exc_info=True)
            return None

    def _should_execute(self, action: Optional[Dict]) -> bool:
        if action is None:
            return False
            
        primary_ticker = self.config['tickers']['primary'][0]
        action_value = action.get(primary_ticker, 0)
        
        current_pos = self.broker.get_positions().get(primary_ticker, {}).get('qty', 0)
        if abs(float(current_pos) + action_value) > self.config['risk']['position_limits']['max']:
            logger.warning("Would exceed position limits - skipping execution")
            return False
            
        return abs(action_value) > 0.01

    def _execute_trade(self, action: Dict) -> None:
        primary_ticker = self.config['tickers']['primary'][0]
        trade_amount = action[primary_ticker]
        
        try:
            execution = self.broker.execute({
                'symbol': primary_ticker,
                'qty': abs(trade_amount) * self.config['trading']['initial_balance'],
                'side': 'buy' if trade_amount > 0 else 'sell',
                'type': 'market'
            })
            
            self.metrics.record_trade(
                ticker=primary_ticker,
                direction=execution['side'],
                status='executed'
            )
            
            self.metrics.record_position(
                ticker=primary_ticker,
                size=trade_amount
            )
            
            logger.debug(f"Trade executed: {execution}")
            
        except Exception as e:
            self.metrics.record_trade(
                ticker=primary_ticker,
                direction='buy' if trade_amount > 0 else 'sell',
                status='failed'
            )
            logger.error(f"Trade execution failed: {str(e)}")
            raise

    def _monitor_cycle(self, start_time: float, interval: int) -> None:
        cycle_time = time.time() - start_time
        sleep_time = max(0, interval - cycle_time)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif cycle_time > interval * 1.5:
            self.metrics.record_trade(
                ticker='system',
                direction='cycle',
                status='overrun'
            )
            logger.warning(f"Cycle overrun: {cycle_time:.2f}s (target {interval}s)")

    def _shutdown(self) -> None:
        logger.info("Initiating shutdown sequence...")
        try:
            self.broker.cancel_all_orders()
            self.metrics.record_portfolio_value(
                ticker=self.config['tickers']['primary'][0],
                value=self.broker.get_account_balance()
            )
            logger.info("Shutdown completed successfully")
        except Exception as e:
            self.metrics.record_trade(
                ticker='system',
                direction='shutdown',
                status='failed'
            )
            logger.error(f"Shutdown error: {str(e)}")
            raise

def main():
    try:
        executor = PaperTradingExecutor()
        executor.run(interval=60)
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        raise