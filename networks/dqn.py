import keras
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Dense, Flatten, LSTM, Lambda, Add, Reshape, Input, Conv2D
from tensorflow.python.keras.losses import MSE

from tqdm.keras import TqdmCallback
import numpy as np

"""
Classic Deep Q network to estimate Q values for Reinforcement Learning
TODO: DuelingQNetwork still has the incorrect output values
https://github.com/PacktPublishing/Tensorflow-2-Reinforcement-Learning-Cookbook/blob/master/Chapter03/2_dueling_dqn.py
"""


class DQN:
    def __init__(self, input_shape, output_shape, n_neurons=64, activation='linear'):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_neurons = n_neurons
        self.activation = activation

        self.model = self.__create_model()

    def __create_model(self):
        model = Sequential([
            InputLayer(input_shape=self.input_shape),
            Dense(self.n_neurons, activation='relu'),
            Flatten(),
            Dense(self.n_neurons, activation='relu'),
            Dense(self.output_shape, activation=self.activation)
        ])

        model.compile(loss='mse', optimizer='adam')
        model.summary()

        return model

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        output = self.model(state)[0].numpy()

        return output

    def update_params(self, batch_memory, gamma):
        """
        batch_memory: tuple of shape (state, action, reward, next_state, done)
        batch_size: size of the batch memory tuple
        gamma: discount value for the future Q-values
        """
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch_memory

        batch_size = len(state_batch)  # Getting batch_size

        target_q_value = self.model.predict(state_batch)  # Current state
        q_value = self.model.predict(next_state_batch)  # Next state

        for index in range(batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                target_q_value[index][action_batch[index]] = reward_batch[index] + gamma * (
                    np.amax(q_value[index]))

        # Train the model
        self.model.fit(state_batch, target_q_value, batch_size=batch_size, epochs=1, verbose=0)


class DoubleDQN():
    """
    Double Deep Q Network, features a target network to make the overall training more stable
    TODO: Check performance and if implementation is actually correct
    """
    def __init__(self, input_shape, output_shape, n_neurons=64, activation='linear'):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_neurons = n_neurons
        self.activation = activation

        self.model = self.__create_model()
        self.target = self.__create_model()

        # Copy weights from one network to the other to make it exactly the same
        self.update_target_network()

    def __create_model(self):
        model = Sequential([
            InputLayer(input_shape=self.input_shape),
            Dense(self.n_neurons, activation='relu'),
            Flatten(),
            Dense(self.n_neurons, activation='relu'),
            Dense(self.output_shape, activation=self.activation)
        ])

        model.compile(loss='mse', optimizer='adam')
        model.summary()

        return model

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        output = self.model(state)[0].numpy()

        return output

    def update_target_network(self):
        self.target.set_weights(self.model.get_weights())

    def update_params(self, batch_memory, gamma):
        """
        batch_memory: tuple of shape (state, action, reward, next_state, done)
        batch_size: size of the batch memory tuple
        gamma: discount value for the future Q-values
        """
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch_memory

        batch_size = len(state_batch)  # Getting batch_size

        target_q_value = self.model.predict(state_batch)  # Current state with target network
        q_value = self.target.predict(next_state_batch)  # Next state

        for index in range(batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                target_q_value[index][action_batch[index]] = reward_batch[index] + gamma * (
                    np.amax(q_value[index]))

        # Train the model
        self.model.fit(state_batch, target_q_value, batch_size=batch_size, epochs=1, verbose=0)


"""
Dueling Deep Q Network, as proposed by DeepMind
This approach divides features into Advantage and Value streams, which 
are later aggregated to calculate the Q-values
"""


class DuelingDQN():
    def __init__(self, input_shape, output_shape, n_neurons=64, activation='linear'):
        super(DuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_neurons = n_neurons
        self.activation = activation

        self.model = self.__create_model()

    def __create_model(self):
        # state_input = InputLayer(input_shape=self.input_shape)
        state_input = Input(shape=self.input_shape)
        backbone_1 = Dense(32, activation='relu')(state_input)
        backbone_2 = Dense(16, activation='relu')(backbone_1)
        value_output = Dense(1)(backbone_2)
        advantage_output = Dense(self.output_shape)(backbone_2)

        output = Add()([value_output, advantage_output])

        model = Model(state_input, output)
        model.compile(loss='mse', optimizer='adam')
        # model.build(self.input_shape)
        model.summary()

        return model

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        output = self.model(state)[0].numpy()

        return output

    def update_params(self, batch_memory, gamma):
        """
        batch_memory: tuple of shape (state, action, reward, next_state, done)
        batch_size: size of the batch memory tuple
        gamma: discount value for the future Q-values
        """
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch_memory

        batch_size = len(state_batch)  # Getting batch_size

        target_q_value = self.model.predict(state_batch)  # Current state with target network
        q_value = self.model.predict(next_state_batch)  # Next state

        for index in range(batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                target_q_value[index][action_batch[index]] = reward_batch[index] + gamma * (
                    np.amax(q_value[index]))

        # Train the model
        self.model.fit(state_batch, target_q_value, batch_size=batch_size, epochs=1, verbose=0)


class RecurrentDQN():
    def __init__(self, input_shape, output_shape, n_neurons=64, activation='linear', hidden_units=224, trace_n=7):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_neurons = n_neurons
        self.activation = activation
        self.hidden_units = hidden_units
        self.trace_length=trace_n

        self.model = self.__create_model()

    def __create_model(self):

        model = Sequential([
            InputLayer(input_shape=self.input_shape),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Flatten(),
            # Dense(256, activation='relu'),
            Reshape(target_shape=(self.trace_length, self.hidden_units)),
            # LSTM input: [batch, timesteps, feature]
            LSTM(units=self.hidden_units),  # RNN layer should help at memorizing stuff from the environment
            Dense(self.output_shape, activation=self.activation)
        ])

        model.compile(loss='mse', optimizer='adam')
        model.summary()

        return model

    def init_hidden_state(self):
        pass

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        output = self.model(state)[0].numpy()

        return output

    def update_params(self, batch_memory, gamma):
        """
        batch_memory: tuple of shape (state, action, reward, next_state, done)
        batch_size: size of the batch memory tuple
        gamma: discount value for the future Q-values
        """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        for episode in batch_memory:
            state_batch.append(episode[0])
            action_batch.append(episode[1])
            reward_batch.append(episode[2])
            next_state_batch.append(episode[3])
            done_batch.append(episode[4])

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)
        # state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch_memory

        batch_size = len(state_batch)  # Getting batch_size

        target_q_value = self.model.predict(state_batch)  # Current state
        q_value = self.model.predict(next_state_batch)  # Next state

        for index in range(batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                target_q_value[index][action_batch[index]] = reward_batch[index] + gamma * (
                    np.amax(q_value[index]))

        # Train the model
        self.model.fit(state_batch, target_q_value, batch_size=batch_size, epochs=1, verbose=0)

class RecurrentDDQN():
    """"
    Recurrent Double Deep Q learning
    """""
    def __init__(self, input_shape, output_shape, n_neurons=64, activation='linear', hidden_units=224, trace_n=7):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_neurons = n_neurons
        self.activation = activation
        self.hidden_units = hidden_units
        self.trace_length=trace_n

        self.model = self.__create_model()
        self.target = self.__create_model()

        self.update_target_network()

    def __create_model(self):

        model = Sequential([
            InputLayer(input_shape=self.input_shape),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Flatten(),
            # Dense(256, activation='relu'),
            Reshape(target_shape=(self.trace_length, self.hidden_units)),
            # LSTM input: [batch, timesteps, feature]
            LSTM(units=self.hidden_units),  # RNN layer should help at memorizing stuff from the environment
            Dense(self.output_shape, activation=self.activation)
        ])

        model.compile(loss='mse', optimizer='adam')
        model.summary()

        return model

    def init_hidden_state(self):
        pass

    def update_target_network(self):
        self.target.set_weights(self.model.get_weights())

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        output = self.model(state)[0].numpy()

        return output

    def update_params(self, batch_memory, gamma):
        """
        batch_memory: tuple of shape (state, action, reward, next_state, done)
        batch_size: size of the batch memory tuple
        gamma: discount value for the future Q-values
        """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        for episode in batch_memory:
            state_batch.append(episode[0])
            action_batch.append(episode[1])
            reward_batch.append(episode[2])
            next_state_batch.append(episode[3])
            done_batch.append(episode[4])

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)
        # state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch_memory

        batch_size = len(state_batch)  # Getting batch_size

        target_q_value = self.model.predict(state_batch)  # Current state
        q_value = self.target.predict(next_state_batch)  # Next state

        for index in range(batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                target_q_value[index][action_batch[index]] = reward_batch[index] + gamma * (
                    np.amax(q_value[index]))

        # Train the model
        self.model.fit(state_batch, target_q_value, batch_size=batch_size, epochs=1, verbose=0)

class RecurrentDuelingDQN():
    def __init__(self):
        pass


class RecurrentDuelingDDQN():
    """
        Recurrent LSTM
        Dueling
        Double
         Deep Q Network
    """

    def __init__(self):
        pass
