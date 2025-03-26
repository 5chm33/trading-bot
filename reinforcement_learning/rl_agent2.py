import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

class RLAgent:
    def __init__(self, env, policy="MlpPolicy", learning_rate=0.0001, buffer_size=1000000, batch_size=512, tau=0.005, gamma=0.99, ent_coef=0.01, verbose=1):
        """
        Initialize the RL agent with SAC.

        Args:
            env: The trading environment.
            policy (str): The policy network architecture (e.g., "MlpPolicy").
            learning_rate (float): Learning rate for the optimizer.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Mini-batch size for training.
            tau (float): Soft update coefficient for the target network.
            gamma (float): Discount factor.
            ent_coef (float or str): Entropy coefficient for exploration.
            verbose (int): Verbosity level.
        """
        # Wrap the environment for stable-baselines3
        self.env = DummyVecEnv([lambda: env])

        # Initialize the SAC model with updated hyperparameters
        self.model = SAC(
            policy=policy,
            env=self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            ent_coef=ent_coef,  # Fixed entropy coefficient
            verbose=verbose
        )

    def choose_action(self, state):
        """
        Choose an action based on the current state.

        Args:
            state (np.array): Current state of the environment.

        Returns:
            float: Action to take (between -1 and 1).
        """
        # Use the SAC model to predict the action
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def train(self, total_timesteps=10000, callback=None):
        """
        Train the RL agent.

        Args:
            total_timesteps (int): Total number of timesteps to train for.
            callback (BaseCallback): Callback for training.
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        """
        Save the trained model.

        Args:
            path (str): Path to save the model.
        """
        self.model.save(path)

    def load(self, path):
        """
        Load a trained model.

        Args:
            path (str): Path to load the model from.
        """
        self.model = SAC.load(path, env=self.env)