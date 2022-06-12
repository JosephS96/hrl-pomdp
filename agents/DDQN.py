import random
import time
import gym
from gym_minigrid.wrappers import ImgObsWrapper
import matplotlib.pyplot as plt


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import Huber

from replay_buffer import ReplayBuffer
from common.schedules import LinearSchedule


class DDQNAgent():
    def __init__(
            self,
            env,
            num_episodes=1000,
            render=False
    ):
        self.identifier = 'dqn_old'

        self.env = env
        self.num_episodes = num_episodes
        self.render = render

        # Replay Buffer
        self.buffer_size = 10000
        self.min_size_batch = 3000
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.batch_size = 32
        self.her = False

        # Hyper-parameters
        self.gamma = 0.99  # discount factor
        self.alpha = 0.001  # Learning rate
        self.epsilon = 1  # Exploration rate
        self.epsilon_min = 0.1
        self.schedule = LinearSchedule(num_episodes, self.epsilon_min)
        self.loss_function = Huber()
        self.optimizer = Adam(learning_rate=self.alpha)
        # self.optimizer = SGD(learning_rate=self.alpha)

        # NN models for Double Deep Q Learning
        self.model = self.__create_model()
        self.target_model = self.__create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_nn_update = 1500

        self.mode = 'train'

    def __create_model(self):

        model = Sequential()
        model.add(InputLayer(input_shape=self.env.observation_space.shape))  # , kernel_initializer = 'zeros')
        model.add(Dense(64, activation='relu'))  # , kernel_initializer = 'zeros'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))  # , kernel_initializer = 'zeros'))
        model.add(Dense(self.env.action_space.n, activation='linear'))  # , kernel_initializer = 'zeros'))
        model.compile(loss='mse', optimizer=self.optimizer)

        model.summary()

        return model

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        if self.mode == 'train':
            action = self.__epsilon_greedy(state)
            return action
        else:  # Only Exploitation
            output = self.model(state)
            action = np.argmax(output, axis=1)[0]
            return action

    def __epsilon_greedy(self, state):
        if random.random() > self.epsilon:  # Exploitation
            output = self.model(state)
            action = np.argmax(output, axis=1)[0]
            return action
        else: # Exploration
            # Reduce the exploration probability
            return random.randint(0, 3)
            # return self.env.action_space.sample()

    def update_params(self, model, replay_memory):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_memory.sample(self.batch_size)

        for state, action, reward, next_state, done in zip(state_batch, action_batch, reward_batch, next_state_batch, done_batch):
            next_state = np.expand_dims(next_state, axis=0)  # Adding batch dimension
            Q_next = self.target_model(next_state).numpy()[0]
            target = reward
            if not done:
                best_action_next = np.argmax(Q_next)
                target = reward + self.gamma * Q_next[best_action_next]

            target_arr = model(np.expand_dims(state, axis=0)).numpy()
            target_arr[0][action] = target
            # print('t', target_arr)
            # print(target_arr[0])
            model.fit(np.expand_dims(state, axis=0), target_arr, epochs=1, verbose=0)

        self.model = model

    def update_params_alt(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        q = np.zeros(self.batch_size)

        for index in range(self.batch_size):
            if done_batch[index]:
                q[index] = reward_batch[index]
            else:
                next_state = np.expand_dims(next_state_batch[index], axis=0)
                future_q = self.target_model(next_state)
                q[index] = reward_batch[index] + self.gamma * tf.reduce_max(future_q, axis=1)

        masks = tf.one_hot(action_batch, self.env.action_space.n)

        with tf.GradientTape() as tape:
            q_values = self.model(state_batch) # Predicted
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_function(q, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_params_2(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        target_q_value = self.model.predict(state_batch)  # Current state
        q_value = self.model.predict(next_state_batch)  # Next state

        for index in range(self.batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                target_q_value[index][action_batch[index]] = reward_batch[index] + self.gamma * (np.amax(q_value[index]))

        # Train the model
        self.model.fit(state_batch, target_q_value, batch_size=self.batch_size, epochs=1, verbose=0)

    def learn(self):

        update_nn_steps = 0
        reward_per_episode = []
        success_rate_history = []
        n_success = 0

        for episode in range(self.num_episodes):

            # if episode % 100 == 0:
            #   print('Episode ', episode)
            episode_time_start = time.time()
            state = self.env.reset()
            done = False
            t = 0
            episode_reward = 0
            action_history = []
            prev_agent_pos = self.env.agent_pos # Initial position of the agent

            if self.render:
                self.env.render()

            # Execute one episode of the environment until the agent is done
            while not done:
                update_nn_steps += 1

                action = self.choose_action(state)
                state_next, reward, done, _ = self.env.step(action)
                agent_pos = self.env.agent_pos  # Current agent position

                # Save statistics for plotting
                episode_reward += reward
                action_history.append(action)

                # Save transitions into memory
                self.replay_buffer.add(state, action, reward, state_next, done)

                # Modify the transitions to have a good reward if the agent actually moved only
                if self.her:
                    if agent_pos is not prev_agent_pos:
                        # if the agent has moved from the position, then add the reward
                        self.replay_buffer.add(state, action, 1, state_next, done)

                state = state_next
                prev_agent_pos = agent_pos
                t += 1

                if self.replay_buffer.__len__() > self.min_size_batch:
                    self.update_params_2()
                    # Anneal epsilon
                    # self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
                    self.epsilon = self.schedule.value(episode)

                # Copy values from model to target model
                if self.target_nn_update % update_nn_steps == 0:
                    self.target_model.set_weights(self.model.get_weights())
                    # print("Updated target network")

                if done and (reward > 0):
                    n_success += 1

                if self.render:
                    self.env.render()

            # Save reward for plotting
            reward_per_episode.append(episode_reward)
            success_rate_history.append(n_success / (episode+1))

            # print stuff for logging
            print(f'Episode: {episode}, Reward: {episode_reward}, Epsilon: {self.epsilon}, Memory length: {self.replay_buffer.__len__()}, Episode length: {t}')
            print(f'Success rate: {success_rate_history[-1]}')
            print(f'Action history: {action_history[-15:]}')
            print(f'Episode duration in ms: {time.time() - episode_time_start}')
            print('\n')

        return reward_per_episode, success_rate_history

if __name__ == "__main__":
    PATH = "/Users/josesanchez/Documents/IAS/Thesis-Results"
    env_name = 'MiniGrid-Empty-16x16-v0'
    env = gym.make(env_name)
    # env = gym.make('MiniGrid-FourRooms-v0', agent_pos=(5, 5), goal_pos=(13, 13))
    env = ImgObsWrapper(env)

    agent = DDQNAgent(env=env, num_episodes=200, render=False)
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
