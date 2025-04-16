from collections import defaultdict
import gymnasium as gym
import gymnasium_env
import numpy as np
import math
import cProfile
from matplotlib import pyplot as plt
import pickle

class Agent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        total_actions = int(np.prod(env.action_space.nvec))
        self.q_values = defaultdict(lambda: np.zeros(total_actions))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    
    def obs_to_key(self, obs):
        return (
            obs["grid"].tobytes(),
            obs["shapes"].tobytes(),
            obs["used_shapes"].tobytes()
        )

    def get_action(self, obs) -> tuple:
        obs_key = self.obs_to_key(obs)
        if np.random.random() < self.epsilon:
            return tuple(self.env.action_space.sample())
        else:
            q_vals = self.q_values[obs_key]
            flat_action = int(np.argmax(q_vals))
            return tuple(np.unravel_index(flat_action, self.env.action_space.nvec))

    def update(self, obs, action, reward, terminated, next_obs, episode_td_errors):
        obs_key = self.obs_to_key(obs)
        next_obs_key = self.obs_to_key(next_obs)

        action_index = np.ravel_multi_index(action, self.env.action_space.nvec)

        future_q_value = 0 if terminated else np.max(self.q_values[next_obs_key])
        td_error = reward + self.discount_factor * future_q_value - self.q_values[obs_key][action_index]
        self.q_values[obs_key][action_index] += self.lr * td_error
        episode_td_errors.append(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# hyperparameters
learning_rate = 0.001
n_episodes = 1_000_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# training
if __name__ == "__main__":
    from tqdm import tqdm

    for episode in tqdm(range(n_episodes), mininterval=2.0):
        obs, info = env.reset()
        episode_td_errors = []
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # if episode == n_episodes - 1:
            #     env.render()

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs, episode_td_errors)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        
        agent.training_error.append(np.mean(episode_td_errors))
        agent.decay_epsilon()
        
        if episode % 10000 == 0:
            with open("q_table.pkl", "wb") as f:
                pickle.dump(dict(agent.q_values), f)
        
    print("Done")

    with open("q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.q_values), f)

    def get_moving_avgs(arr, window, convolution_mode):
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    # Smooth over a 500 episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()

    env.close()