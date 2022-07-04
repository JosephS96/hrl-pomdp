from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Dense, Flatten, LSTM, Lambda, Add, Reshape, Input, Conv2D

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

"""
Main difference is that these take as input the goal as well, so that
input shape is state + goal
"""


class DoubleDQN():
    """
    Double Deep Q Network, features a target network to make the overall training more stable
    TODO: Check performance and if implementation is actually correct
    """
    def __init__(self, state_shape, output_shape, n_neurons=64, activation='linear'):

        state_placeholder = np.zeros(state_shape)
        state_goal = np.concatenate((state_placeholder, state_placeholder), axis=2)

        self.input_shape = state_goal.shape
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

    def predict(self, state, goal):
        """
        Predicts a single value of the model
        state: shape (7,7,3)
        goal: shape (7,7,3)
        """

        state_goal = np.concatenate([state, goal], axis=2)
        state = np.expand_dims(state_goal, axis=0)
        output = self.model(state)[0].numpy()

        return output

    def update_target_network(self):
        self.target.set_weights(self.model.get_weights())

    def update_params(self, batch_memory, gamma):
        """
        batch_memory: tuple of shape (state, action, reward, next_state, goal, done)
        batch_size: size of the batch memory tuple
        gamma: discount value for the future Q-values
        """
        state_batch, action_batch, reward_batch, next_state_batch, goal_batch, done_batch = batch_memory

        batch_size = len(state_batch)  # Getting batch_size

        # Encode the goal into the states
        state_goal_batch = np.concatenate([state_batch, goal_batch], axis=3)
        next_state_goal_batch = np.concatenate([next_state_batch, goal_batch], axis=3)

        target_q_value = self.model.predict(state_goal_batch)  # Current state with target network
        q_value = self.target.predict(next_state_goal_batch)  # Next state

        for index in range(batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                target_q_value[index][action_batch[index]] = reward_batch[index] + gamma * (
                    np.amax(q_value[index]))

        # Train the model
        self.model.fit(state_goal_batch, target_q_value, batch_size=batch_size, epochs=1, verbose=0)


class RecurrentDDQN():
    """"
    Recurrent Double Deep Q learning
    """""
    def __init__(self, input_shape, output_shape, n_neurons=64, activation='linear', hidden_units=224, trace_n=7):

        state_placeholder = np.zeros(input_shape)
        state_dir = np.zeros(shape=(input_shape[0], input_shape[1], 1))
        state_goal = np.concatenate((state_placeholder, state_placeholder, state_dir), axis=2).shape

        self.input_shape = state_goal
        self.output_shape = output_shape
        self.n_neurons = n_neurons
        self.activation = activation
        self.hidden_units = hidden_units
        self.trace_length = trace_n

        self.model = self.__create_model()
        self.target = self.__create_model()

        self.update_target_network()

    def __create_model(self):
        """
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
        """

        # input_layer = InputLayer(input_shape=self.input_shape)
        input_layer = Input(shape=self.input_shape)
        fc1 = Dense(64, activation='relu')(input_layer)
        fc2 = Dense(32, activation='relu')(fc1)
        flatten = Flatten()(fc2)
        # Dense(256, activation='relu'),
        #1,7,7,32
        #28,7,7,32

        reshape = Reshape(target_shape=(self.trace_length, self.hidden_units))(fc2)
        # LSTM input: [batch, timesteps, feature]
        lstm = LSTM(units=self.hidden_units)(reshape, initial_state=None)  # RNN layer should help at memorizing stuff from the environment
        output = Dense(self.output_shape, activation=self.activation)(lstm)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer='adam')
        model.summary()

        return model

    def init_hidden_state(self):
        pass

    def update_target_network(self):
        self.target.set_weights(self.model.get_weights())

    def predict(self, state, goal):
        """
                Predicts a single value of the model
                state: shape (7,7,3)
                goal: shape (7,7,3)
                """

        state_goal = np.concatenate([state, goal], axis=2)
        state = np.expand_dims(state_goal, axis=0)
        output = self.model(state)[0].numpy()

        return output

    def update_params(self, batch_memory, gamma):
        """
        batch_memory: tuple of shape (state, action, reward, next_state, goal, done)
        batch_size: size of the batch memory tuple
        gamma: discount value for the future Q-values
        """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        goal_batch = []
        done_batch = []
        for episode in batch_memory:
            state_batch.append(episode[0])
            action_batch.append(episode[1])
            reward_batch.append(episode[2])
            next_state_batch.append(episode[3])
            goal_batch.append(episode[4])
            done_batch.append(episode[5])

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        goal_batch = np.array(goal_batch)
        done_batch = np.array(done_batch)
        # state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch_memory

        # Encode the goal into the states
        state_goal_batch = np.concatenate([state_batch, goal_batch], axis=3)
        next_state_goal_batch = np.concatenate([next_state_batch, goal_batch], axis=3)

        batch_size = len(state_goal_batch)  # Getting batch_size

        target_q_value = self.model.predict(state_goal_batch)  # Current state
        q_value = self.target.predict(next_state_goal_batch)  # Next state

        for index in range(batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                target_q_value[index][action_batch[index]] = reward_batch[index] + gamma * (
                    np.amax(q_value[index]))

        # Train the model
        self.model.fit(state_goal_batch, target_q_value, batch_size=batch_size, epochs=1, verbose=0)

class Learner(nn.Module):

    def __init__(self, hidden_units, output_shape, trace_length):
        super().__init__()
        self.trace_length = trace_length
        self.hidden_units = hidden_units
        self.output_shape = output_shape
        self.linear1 = nn.Linear(7 , 64)
        self.linear2 = nn.Linear(64, 32)
        self.lstm = nn.LSTM(hidden_units, hidden_units)
        self.adv = nn.Linear(hidden_units, output_shape)
        self.val = nn.Linear(hidden_units, 1)


    def predict(self,  state, hidden_state, cell_state):
        return self.forward(state, hidden_state, cell_state)

    def forward(self, state, hidden_state, cell_state):
        x = F.relu(self.linear1(torch.tensor(state, dtype=torch.float)))
        x = F.relu(self.linear2(x))
        reshape = x.view( x.shape[0], self.trace_length, self.hidden_units)
        lstm_out = self.lstm(reshape, (hidden_state, cell_state))

        out = lstm_out[0] [:, 7 - 1, :]
        hidden = lstm_out[1][0]
        cell = lstm_out[1][1]

        adv_out = self.adv(out)
        val_out = self.val(out)

        model_out = val_out.expand(x.shape[0], self.output_shape) + (
                     adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(x.shape[0], self.output_shape))

        return model_out, hidden, cell

    def init_hidden_states(self):
        h = torch.zeros( 1, 7 , 224).float()
        c = torch.zeros( 1, 7,  224).float()
        return h, c

class RecurrentDDQNPyTorch():
    """"
    Recurrent Double Deep Q learning
    """""

    def __init__(self, input_shape, output_shape, n_neurons=64, activation='linear', hidden_units=224, trace_n=7):

        state_placeholder = np.zeros(input_shape)
        state_dir = np.zeros(shape=(input_shape[0], input_shape[1], 1))
        state_goal = np.concatenate((state_placeholder, state_placeholder, state_dir), axis=2).shape

        self.input_shape = state_goal
        self.output_shape = output_shape
        self.n_neurons = n_neurons
        self.activation = activation
        self.hidden_units = hidden_units
        self.trace_length = trace_n

        self.model = self.__create_model()
        self.target = self.__create_model()

        self.update_target_network()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.hidden_state, self.cell_state =  self.model.init_hidden_states()

    def __create_model(self):
        return Learner(self.hidden_units, self.output_shape, self.trace_length)

    def update_target_network(self):
        self.target.load_state_dict(self.model.state_dict())

    def predict(self, state, goal):
        """
                Predicts a single value of the model
                state: shape (7,7,3)
                goal: shape (7,7,3)
                """
        state_goal = np.concatenate([state, goal], axis=2)
        state_goal = np.expand_dims(state_goal, axis=0)
        output,  self.hidden_state, self.cell_state = self.model(state_goal, self.hidden_state, self.cell_state)
        output = output.detach().numpy()[0]

        return output

    def update_params(self, batch_memory, gamma):
        """
        batch_memory: tuple of shape (state, action, reward, next_state, goal, done)
        batch_size: size of the batch memory tuple
        gamma: discount value for the future Q-values
        """

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        goal_batch = []
        done_batch = []
        for episode in batch_memory:
            state_batch.append(episode[0])
            action_batch.append(episode[1])
            reward_batch.append(episode[2])
            next_state_batch.append(episode[3])
            goal_batch.append(episode[4])
            done_batch.append(episode[5])

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        goal_batch = np.array(goal_batch)
        done_batch = np.array(done_batch)

        hidden_state, cell_state = self.model.init_hidden_states()

        # Encode the goal into the states
        state_goal_batch = np.concatenate([state_batch, goal_batch], axis=3)
        next_state_goal_batch = np.concatenate([next_state_batch, goal_batch], axis=3)

        batch_size = len(state_goal_batch)

        q_value, _, _ = self.model.predict(state_goal_batch, hidden_state, cell_state)  # Current state
        target_q_value, _, _ = self.target.predict(next_state_goal_batch, hidden_state, cell_state)  # Next state

        for index in range(batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                q_next_max, _ = target_q_value[index].detach().max(dim=0)
                target_q_value[index][action_batch[index]] = reward_batch[index] + gamma * q_next_max

        loss = self.criterion(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ConvRecurrentDQN:
    """"
        Recurrent Double Deep Q learning
        """""

    def __init__(self, input_shape, output_shape, n_neurons=64, activation='linear', hidden_units=224, trace_n=7):

        state_placeholder = np.zeros(input_shape)
        state_dir = np.zeros(shape=(input_shape[0], input_shape[1], 1))
        state_goal = np.concatenate((state_placeholder, state_placeholder, state_dir), axis=2).shape

        self.input_shape = state_goal
        self.output_shape = output_shape
        self.n_neurons = n_neurons
        self.activation = activation
        self.hidden_units = hidden_units
        self.trace_length = trace_n

        self.model = self.__create_model()
        self.target = self.__create_model()

        self.update_target_network()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.hidden_state, self.cell_state = self.model.init_hidden_states()

    def __create_model(self):
        return ConvLearnerSub(self.hidden_units, self.output_shape, self.trace_length)

    def update_target_network(self):
        self.target.load_state_dict(self.model.state_dict())

    def predict(self, state, goal):
        """
                Predicts a single value of the model
                state: shape (7,7,3)
                goal: shape (7,7,3)
                """
        state_goal = np.concatenate([state, goal], axis=2)
        state_goal = np.expand_dims(state_goal, axis=0)
        output, self.hidden_state, self.cell_state = self.model(state_goal, self.hidden_state, self.cell_state)
        output = output.detach().numpy()[0]

        return output

    def update_params(self, batch_memory, gamma):
        """
        batch_memory: tuple of shape (state, action, reward, next_state, goal, done)
        batch_size: size of the batch memory tuple
        gamma: discount value for the future Q-values
        """

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        goal_batch = []
        done_batch = []
        for episode in batch_memory:
            state_batch.append(episode[0])
            action_batch.append(episode[1])
            reward_batch.append(episode[2])
            next_state_batch.append(episode[3])
            goal_batch.append(episode[4])
            done_batch.append(episode[5])

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        goal_batch = np.array(goal_batch)
        done_batch = np.array(done_batch)

        hidden_state, cell_state = self.model.init_hidden_states()

        # Encode the goal into the states
        state_goal_batch = np.concatenate([state_batch, goal_batch], axis=3)
        next_state_goal_batch = np.concatenate([next_state_batch, goal_batch], axis=3)

        batch_size = len(state_goal_batch)

        target_q_value, _, _ = self.model.predict(state_goal_batch, hidden_state, cell_state)  # Current state
        q_value, _, _ = self.target.predict(next_state_goal_batch, hidden_state, cell_state)  # Next state

        for index in range(batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                target_q_value[index][action_batch[index]] = reward_batch[index] + gamma * (
                    torch.amax(q_value[index]))

        loss = self.criterion(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ConvLearnerSub(nn.Module):

    def __init__(self, hidden_units, output_shape, trace_length):
        super().__init__()
        self.trace_length = trace_length
        self.hidden_units = hidden_units
        self.output_shape = output_shape
        self.linear1 = nn.Linear(7 , 64)
        self.linear2 = nn.Linear(64, 32)
        self.lstm = nn.LSTM(hidden_units, hidden_units)
        self.adv = nn.Linear(hidden_units, output_shape)
        self.val = nn.Linear(hidden_units, 1)

        self.conv1 = nn.Conv2d(in_channels=7, out_channels=32, kernel_size=(3, 3), stride=(4, 4))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(4, 4))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=self.hidden_units, kernel_size=(3, 3), stride=(1, 1))

    def predict(self,  state, hidden_state, cell_state):
        return self.forward(state, hidden_state, cell_state)

    def forward(self, state, hidden_state, cell_state):
        state = torch.tensor(state, dtype=torch.float)
        state = state.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        conv_flat = torch.flatten(x, 1)  # flatten all dimensions except batch

        # reshape = x.view( x.shape[0], self.trace_length, self.hidden_units)
        lstm_out = self.lstm(conv_flat, (hidden_state, cell_state))

        out = lstm_out[0] [:, 7 - 1, :]
        hidden = lstm_out[1][0]
        cell = lstm_out[1][1]

        adv_out = self.adv(out)
        val_out = self.val(out)

        model_out = val_out.expand(x.shape[0], self.output_shape) + (
                     adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(x.shape[0], self.output_shape))

        return model_out, hidden, cell

    def init_hidden_states(self):
        h = torch.zeros( 1, 7 , 224).float()
        c = torch.zeros( 1, 7,  224).float()
        return h, c