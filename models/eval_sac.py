"""
This evaluates SAC on the gymnasium Pendulum:
https://gymnasium.farama.org/environments/classic_control/pendulum/
"""

import gymnasium as gym

env = gym.make("Pendulum-v1", render_mode="human")

observation, info = env.reset()

episode_over = False
total_reward = 0

while not episode_over:

    # TODO: Replace with agents action
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished. Total reward: {total_reward}")
env.close()