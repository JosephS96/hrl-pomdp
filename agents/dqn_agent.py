import matplotlib.pyplot as plt
import numpy as np
import random
import time
import gym
from gym_minigrid.wrappers import ImgObsWrapper

from replay_buffer import ReplayBuffer
from common.schedules import LinearSchedule
from networks.dqn import DQN


class DQNAgent:
    def __init__(self, env, num_episodes=200, render=False):

        self.identifier = 'dqn'
        self.env = env
        self.num_episodes = num_episodes
        self.render = render

        # Replay Buffer
        self.buffer_size = 10000
        self.min_size_batch = 3000
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.batch_size = 32
        # self.her = False

        # Hyper-parameters
        self.gamma = 0.99  # discount factor
        self.alpha = 0.001  # Learning rate
        self.epsilon = 1  # Exploration rate
        self.epsilon_min = 0.1
        self.schedule = LinearSchedule(num_episodes, self.epsilon_min)

        self.model = DQN(input_shape=self.env.observation_space.shape, output_shape=self.env.action_space.n, n_neurons=32)

        self.mode = 'train'

    def choose_action(self, state):
        if self.mode == 'train':
            action = self.__epsilon_greedy(state)
            return action
        else:  # Only Exploitation
            output = self.model.predict(state)
            action = np.argmax(output, axis=1)
            return action

    def __epsilon_greedy(self, state):
        if random.random() > self.epsilon:  # Exploitation
            output = self.model.predict(state)
            action = np.argmax(output, axis=0)
            return action
        else: # Exploration
            # Reduce the exploration probability
            return random.randint(0, 3)
            # return self.env.action_space.sample()

    def learn(self):
        start_training_time = time.time()
        reward_per_episode = []
        success_rate_history = []
        n_success = 0

        for episode in range(self.num_episodes):
            episode_time_start = time.time()
            state = self.env.reset()
            done = False
            t = 0  # episode length
            episode_reward = 0
            action_history = []
            prev_agent_pos = self.env.agent_pos  # Initial position of the agent

            # Execute one episode of the environment until the agent is done
            while not done:

                action = self.choose_action(state)
                state_next, reward, done, _ = self.env.step(action)
                agent_pos = self.env.agent_pos  # Current agent position

                # Save statistics for plotting
                episode_reward += reward
                action_history.append(action)

                # If goal was not achieved, penalize
                if reward == 0:
                    reward = -0.01

                # Save transitions into memory
                self.replay_buffer.add(state, action, reward, state_next, done)

                state = state_next
                prev_agent_pos = agent_pos
                t += 1

                if self.replay_buffer.__len__() > self.min_size_batch:
                    batch_memory = self.replay_buffer.sample(self.batch_size)
                    self.model.update_params(batch_memory, self.gamma)
                    self.epsilon = self.schedule.value(episode)  # Anneal epsilon

                if done and (reward > 0):
                    n_success += 1

                if self.render:
                    self.env.render()

                # End of while

            # Save reward for plotting
            reward_per_episode.append(episode_reward)
            success_rate_history.append(n_success / (episode + 1))

            # print stuff for logging - End of Episode -
            print(
                f'Episode: {episode}, Reward: {episode_reward}, Epsilon: {self.epsilon}, Memory length: {self.replay_buffer.__len__()}, Episode length: {t}')
            print(f'Success rate: {success_rate_history[-1]}')
            print(f'Action history: {action_history[-15:]}')
            print(f'Episode duration in seconds: {time.time() - episode_time_start}')
            print('\n')

        return reward_per_episode, success_rate_history


if __name__ == "__main__":
    PATH = "/Users/josesanchez/Documents/IAS/Thesis-Results"
    env_name = 'MiniGrid-Empty-16x16-v0'
    env = gym.make(env_name)
    # env = gym.make('MiniGrid-FourRooms-v0', agent_pos=(5, 5), goal_pos=(13, 13))
    env = ImgObsWrapper(env)

    agent = DQNAgent(env=env, num_episodes=200, render=False)
    rewards, success = agent.learn()

    results = {}
    results['success'] = success
    results['rewards'] = rewards

    save_path = f"{PATH}/{env_name}/{agent.identifier}-{time.time()}.npy"
    np.save(save_path, results)

    # plt.figure()
    # plotting.plot_rewards([stats], smoothing_window=10)
    plt.ylim(0, 1.0)
    plt.plot(range(len(rewards)), rewards)
    plt.show()

    plt.ylim(0, 1.0)
    plt.plot(range(len(success)), success)
    plt.show()