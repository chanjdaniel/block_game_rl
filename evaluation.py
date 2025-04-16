import gymnasium as gym
import gymnasium_env
from collections import defaultdict
import numpy as np
import pickle
from agent import Agent

env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")

with open("q_table.pkl", "rb") as f:
    saved_q_table = pickle.load(f)

total_actions = int(np.prod(env.action_space.nvec))

learning_rate = 0.01
n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

agent.q_values = defaultdict(lambda: np.zeros(total_actions), saved_q_table)
agent.epsilon = 0.0  # Greedy policy for evaluation

# 4. Run a single episode
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}")
    total_reward += reward
    done = terminated or truncated
    env.render()  # 👈 optional (remove for headless run)

print(f"Final reward: {total_reward}")
env.close()