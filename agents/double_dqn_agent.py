import matplotlib.pyplot as plt
import numpy as np
import random
import time
import gym
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper

from common.Logger import Logger
from replay_buffer import ReplayBuffer
from common.schedules import LinearSchedule
from networks.dqn import DoubleDQN
from envs.staticfourrooms import StaticFourRoomsEnv
from envs.randomempty import RandomEmpyEnv


class DoubleDQNAgent:
    def __init__(self, env, num_episodes=200, render=False):

        self.identifier = 'double-dqn'
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

        # Steps to give before updating target model nn
        self.target_update_steps = 2000
        self.model = DoubleDQN(input_shape=self.env.observation_space.shape, output_shape=3, n_neurons=32)

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
            return random.randint(0, 2)
            # return self.env.action_space.sample()

    def learn(self):
        logger = Logger()

        update_nn_steps = 0
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
                update_nn_steps += 1

                action = self.choose_action(state)
                state_next, reward, done, _ = self.env.step(action)
                agent_pos = self.env.agent_pos  # Current agent position

                # Save statistics for plotting
                episode_reward += reward
                action_history.append(action)

                logger.extrinsic_reward_per_step.append(reward)
                logger.intrinsic_reward_per_step.append(0)

                # Save transitions into memory
                self.replay_buffer.add(state, action, reward, state_next, done)

                state = state_next
                prev_agent_pos = agent_pos
                t += 1

                if self.replay_buffer.__len__() > self.min_size_batch:
                    batch_memory = self.replay_buffer.sample(self.batch_size)
                    self.model.update_params(batch_memory, self.gamma)
                    self.epsilon = self.schedule.value(episode)  # Anneal epsilon

                if update_nn_steps % self.target_update_steps == 0:
                    self.model.update_target_network()

                if done and (reward > 0):
                    n_success += 1

                if self.render:
                    self.env.render()

                # End of while

            # Save reward for plotting
            reward_per_episode.append(episode_reward)
            success_rate_history.append(n_success / (episode + 1))

            logger.extrinsic_reward_per_episode.append(episode_reward)
            logger.intrinsic_reward_per_episode.append(0)
            logger.steps_per_episode.append(t)
            logger.success_rate.append(n_success/ (episode + 1))

            if episode > 10:
                logger.print_latest_statistics()

            # print stuff for logging - End of Episode -
            print(
                f'Episode: {episode}, Reward: {episode_reward}, Epsilon: {self.epsilon}, Memory length: {self.replay_buffer.__len__()}, Episode length: {t}')
            print(f'Success rate: {success_rate_history[-1]}')
            print(f'Action history: {action_history[-15:]}')
            print(f'Episode duration in seconds: {time.time() - episode_time_start}')
            print('\n')

        return logger


if __name__ == "__main__":
    PATH = "/Users/josesanchez/Documents/IAS/Thesis-Results"
    # env_name = 'MiniGrid-Empty-16x16-v0'
    #env = gym.make(env_name)

    # env_name = 'MiniGrid-Empty-11x11'
    # env = StaticFourRoomsEnv(grid_size=13, max_steps=500)
    # env = RandomEmpyEnv(grid_size=11, max_steps=400, goal_pos=(9, 9), agent_pos=(1, 1))
    # env = gym.make('MiniGrid-Empty-8x8-v0', size=11)
    # env = gym.make('MiniGrid-FourRooms-v0', agent_pos=(5, 5), goal_pos=(13, 13))
    # env = FullyObsWrapper(env)
    env_name = "StaticFourRooms-11x11"
    goal_pos = (8, 7)
    env = StaticFourRoomsEnv(agent_pos=(2, 2), goal_pos=goal_pos, grid_size=11, max_steps=400)

    env = ImgObsWrapper(env)

    agent = DoubleDQNAgent(env=env, num_episodes=1000, render=False)
    logger = agent.learn()

    logger.save(env_name, agent.identifier)

    # plt.figure()
    # plotting.plot_rewards([stats], smoothing_window=10)
    plt.ylim(0, 1.0)
    plt.plot(range(len(logger.extrinsic_reward_per_episode)), logger.extrinsic_reward_per_episode)
    plt.show()

    plt.ylim(0, 1.0)
    plt.plot(range(len(logger.success_rate)), logger.success_rate)
    plt.show()