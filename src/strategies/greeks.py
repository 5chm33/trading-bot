from py_vollib.black_scholes.greeks import analytical

class GreekCalculator:
    @staticmethod
    def calculate_all(
        S: float,  # Spot
        K: float,  # Strike
        t: float,  # DTE (years)
        r: float,  # Risk-free
        sigma: float,  # Volatility
        flag: str  # 'c' or 'p'
    ) -> dict:
        return {
            'delta': analytical.delta(flag, S, K, t, r, sigma),
            'gamma': analytical.gamma(flag, S, K, t, r, sigma),
            'theta': analytical.theta(flag, S, K, t, r, sigma),
            'vega': analytical.vega(flag, S, K, t, r, sigma)
        }