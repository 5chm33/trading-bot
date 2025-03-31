Updated README.md:
markdown
Copy

# Algorithmic Trading System v2.0
*Multi-model trading platform combining Transformer forecasting with RL optimization*

## 🚀 Key Features
- **Hybrid Architecture**
  - Transformer-based price prediction
  - SAC RL agent for strategy optimization
  - Kalman Filter smoothing

- **Real-time Monitoring**
  ```mermaid
  graph TD
    A[Trading Bot] -->|Metrics| B(Prometheus)
    B --> C{Grafana}
    C --> D[Dashboard]
    C --> E[Alerts]

Install the dependencies using:
```bash
pip install -r requirements.txt
