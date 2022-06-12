import time

import numpy as np
import random
import gym
from gym_minigrid.wrappers import ImgObsWrapper
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow

from collections import defaultdict, deque
from envs.mdp import StochasticMDPEnv as MDP
from envs.cmdp import ContinuousStochasticMDPEnv as cMDP
from envs.chmdp import ContinuousStochastichMDPEnv as chMDP
import utils.plotting as plotting

class DQNAgent():

    def __init__(self, env = cMDP(),  num_episodes = 20000, \
                    gamma = 0.9, batch_size = 1, \
                    epsilon_anneal = 1/2000):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_anneal = epsilon_anneal

        self.model = self.__get_model()

    def __get_model(self):

        model = Sequential()
        model.add(InputLayer(input_shape=env.observation_space.shape))  # , kernel_initializer = 'zeros')
        model.add(Flatten())
        model.add(Dense(24, activation='relu'))  # , kernel_initializer = 'zeros'))
        model.add(Dense(24, activation='relu'))  # , kernel_initializer = 'zeros'))
        model.add(Dense(self.env.action_space.n, activation='linear'))  # , kernel_initializer = 'zeros'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        model.summary()

        return model

    def epsGreedy(self, state, B, eps, model):
        # Q = model.predict(np.array([state]).reshape(1, -1))[0]
        state = np.expand_dims(state, axis=0)
        Q = model(state).numpy()[0]
        #print(Q)
        action_probabilities = np.ones_like(Q) * eps / len(Q)
        best_action = np.argmax(Q)
        action_probabilities[best_action] += (1.0 - eps)
        action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
        return action

    def QValueUpdate(self, model, D):
        batch_size = min(self.batch_size, len(D))
        mini_batch = random.sample(D, batch_size)
        for s, action, f, s_next, done in mini_batch:
            s_next = np.expand_dims(s_next, axis=0)
            Q_next = model(s_next).numpy()[0]
            target = f
            if not done:
                best_next_action = np.argmax(Q_next)
                target = f + self.gamma * Q_next[best_next_action]
            target_arr = model(np.expand_dims(s, axis=0)).numpy()
            target_arr[0][action] = target
            #print('t', target_arr)
            #print(target_arr[0])
            model.fit(np.expand_dims(s, axis=0), target_arr, epochs=1, verbose=0)

    def learn(self):

        stats = plotting.Stats(num_episodes=self.num_episodes, continuous=True)

        D = deque(maxlen=1000)
        Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        A = self.env.action_space

        epsilon = 1.0

        for i in range(self.num_episodes):
            if i % 100 == 0:
                print('Episode ', i)
            s = self.env.reset()
            done = False
            t = 0
            while not done:# and t < 500:
                action = self.epsGreedy(s, A, epsilon, self.model)
                #print(action)
                s_next, f, done, _ = self.env.step(action)
                stats.episode_rewards[i] += f
                stats.episode_lengths[i] = t
                #print(s_next)
                #stats.visitation_count[s_next, i] += 1

                D.append((s, action, f, s_next, done))
                #print(s, epsilon)
                s = s_next
                t += 1
            self.QValueUpdate(self.model, D)
            epsilon = max(epsilon - self.epsilon_anneal, 0.1)
            print('time', t, ', epsilon ', epsilon)

                
        # plt.figure()
        #plotting.plot_q_values(model, self.env.observation_space, self.env.action_space)

        return stats
        #plotting.plot_episode_stats(stats, smoothing_window=100)


if __name__ == "__main__":
    # env = chMDP()
    # env = MDP()
    env = gym.make('MiniGrid-Empty-6x6-v0')
    env = ImgObsWrapper(env)
    # env = gym.make('CartPole-v1')
    agent = DQNAgent(env=env, num_episodes=3000, batch_size=32)
    stats = agent.learn()

    # plt.figure()
    # plotting.plot_rewards([stats], smoothing_window=10)
    plt.plot(range(len(stats.episode_rewards)), stats.episode_rewards)
    plt.show()

    state = env.reset()
    done = False
    while not done:
        action = agent.epsGreedy(state, 0, 0.2, agent.model)
        print(action)

        state, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.1)

    env.close()