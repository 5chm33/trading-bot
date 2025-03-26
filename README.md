# Trading Bot with Reinforcement Learning

This project implements a trading bot using **Reinforcement Learning (RL)** and **Transformer-based models** to predict and trade stocks. The bot is designed to optimize trading strategies using historical stock data and technical indicators.

---

## **Features**
- **Transformer Model**: Predicts stock prices using a Transformer-based architecture.
- **Reinforcement Learning Agent**: Uses the Soft Actor-Critic (SAC) algorithm to optimize trading strategies.
- **Kalman Filter**: Smoothes predictions for better decision-making.
- **Hyperparameter Tuning**: Uses Ray Tune for efficient hyperparameter optimization.
- **Evaluation Metrics**: Includes metrics like MSE, MAE, R², Sharpe Ratio, and Sortino Ratio.

---

## **Requirements**
To run this project, you'll need the following dependencies:
- Python 3.8+
- Libraries listed in `requirements.txt`

Install the dependencies using:
```bash
pip install -r requirements.txt
