# keras, based on Paper 'Balacing_CartPole_RL'

import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
# testing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

EPISODES = 100

class DQN_Actor:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.train_start = 1000
        self.model = self._build_model()

        # build environment
        #self.env = gym.make('CartPole-v0')

    def _build_model(self):
        # Neural Network for Deep-Q learnign model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0]) # action return

    def replay_batch(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        current_state = np.zeros((batch_size, self.state_size))
        next_state = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            current_state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        targetQvalue = self.model.predict(current_state) # current state
        Qvalue_ns = self.model.predict(next_state) # next state

        for i in range(batch_size):
            if done[i]:
                targetQvalue[i][action[i]] = reward[i]
            else:
                targetQvalue[i][action[i]] = reward[i] + self.gamma * (np.amax(Qvalue_ns[i]))

        # train the model
        self.model.fit(current_state, targetQvalue, batch_size=batch_size, epochs=1, verbose=1)
        if (self.epsilon > self.epsilon_min) and (len(self.memory) >= self.train_start):
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# TODO: refactor in main method
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    print(env.observation_space.shape)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN_Actor(state_size, action_size)

    done = False
    batch_size = 64
    score1 = []
    episode1 = []
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False
        t = 0
        while not done:
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -100
            next_state = np.reshape(next_state, [1, state_size])

            # remember
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            t += 1

            # experience replay
            if len(agent.memory) > batch_size:
                agent.replay_batch(batch_size)
                #agent.replay_incremental(batch_size)
        print("episode: {}, score: {}, e: {:.2}, memory length: {}".format(e, t, agent.epsilon, len(agent.memory)))
        score1.append(t)
        episode1.append(e)

        if np.mean(score1[-min(50, len(score1)):]) >= (env.spec.max_episode_steps-5):
            print('Problem is solved in {} episodes'.format(e))
            break
    # agent.save("./saved_networks/cartple_dqn1.h5")

    plt.plot(episode1, score1)
    plt.xlabel('Episodes')
    plt.ylabel('Time steps')
