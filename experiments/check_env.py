"""Script for running the check_env utility."""

import s5  # noqa: F401
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

env = gym.make("s5/static-v0")
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

check_env(env.unwrapped)
