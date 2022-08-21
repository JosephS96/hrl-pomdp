from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Dense, Flatten, LSTM, Lambda, Add, Reshape, Input, Conv2D

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np


class ConvLearner(nn.Module):

    def __init__(self, hidden_units, output_shape, trace_length):
        super().__init__()
        self.trace_length = trace_length
        self.hidden_units = hidden_units
        self.output_shape = output_shape

        # 44, 44, 3
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=(3, 3), stride=(4, 4))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=self.hidden_units, kernel_size=(3, 3), stride=(1, 1))

        self.lstm = nn.LSTM(hidden_units, hidden_units)
        self.adv = nn.Linear(hidden_units, output_shape)
        self.val = nn.Linear(hidden_units, 1)
        # 1, 256

    def predict(self,  state, hidden_state, cell_state, trace_length, batch_size):
        return self.forward(state, hidden_state, cell_state, trace_length, batch_size)

    def forward(self, state, hidden_state, cell_state, trace_length, batch_size):
        state = torch.tensor(state, dtype=torch.float)
        state = state.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        conv_flat = torch.flatten(x, 1)  # flatten all dimensions except batch

        reshape = conv_flat.view( batch_size, trace_length, self.hidden_units)
        lstm_out = self.lstm(reshape, (hidden_state, cell_state))

        out = lstm_out[0] # [:, trace_length - 1, :]
        hidden = lstm_out[1][0]
        cell = lstm_out[1][1]

        out = torch.reshape(out, shape=(-1, self.hidden_units))

        split_values = torch.split(out, self.hidden_units // 2, dim=1)

        streamA = split_values[0]
        streamV = split_values[1]
        AW = torch.randn(size=(self.hidden_units // 2, self.output_shape))
        VW = torch.randn(size=(self.hidden_units // 2, 1))

        advantage = torch.matmul(streamA, AW)
        value = torch.matmul(streamV, VW)

        model_out = value + torch.subtract(advantage, torch.mean(advantage, dim=1, keepdim=True))

        # adv_out = self.adv(out)
        # val_out = self.val(out)

        # model_out = val_out.expand(batch_size, self.output_shape) + (adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(batch_size, self.output_shape))

        return model_out, hidden, cell

    def init_hidden_states(self, trace_length):
        h = torch.zeros( 1, trace_length , self.hidden_units).float()
        c = torch.zeros( 1, trace_length,  self.hidden_units).float()
        return h, c


class ConvRecurrentDQN:
    """"
        Recurrent Double Deep Q learning
    """""

    def __init__(self, input_shape, output_shape, n_neurons=64, activation='linear', hidden_units=256, trace_n=8):

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

        self.hidden_state = None
        self.cell_state = None
        self.init_hidden(trace_length=1)

    def __create_model(self):
        return ConvLearner(self.hidden_units, self.output_shape, self.trace_length)

    def init_hidden(self, trace_length):
        self.hidden_state, self.cell_state = self.model.init_hidden_states(trace_length)

    def set_hidden(self, hidden, cell):
        self.hidden_state = hidden
        self.cell_state = cell

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
        output, self.hidden_state, self.cell_state = self.model.predict(state_goal, self.hidden_state, self.cell_state, trace_length=1, batch_size=1)
        output = output.detach().numpy()[0]

        return output

    def update_params(self, batch_memory, gamma, batch_size):
        state_batch = np.array([row[0] for row in batch_memory])
        action_batch = np.array([row[1] for row in batch_memory])
        reward_batch = np.array([row[2] for row in batch_memory])
        next_state_batch = np.array([row[3] for row in batch_memory])
        goal_batch = np.array([row[4] for row in batch_memory])
        done_batch = np.array([row[5] for row in batch_memory])

        # Update is always perform by resetting the lstm hidden
        hidden_state, cell_state = self.model.init_hidden_states(trace_length=self.trace_length)

        # Encode the goal into the states
        state_goal_batch = np.concatenate([state_batch, goal_batch], axis=3)
        next_state_goal_batch = np.concatenate([next_state_batch, goal_batch], axis=3)

        # Below we perform Double DQN

        # Get actions from main network
        q_value, _, _ = self.model.predict(next_state_goal_batch, hidden_state, cell_state, trace_length=self.trace_length, batch_size=batch_size)  # Current state
        q_value_max = np.argmax(q_value.detach().numpy(),
                                axis=1)  # Represents the actions the network would have taken for each state of the batch

        # Get Q values from the target network
        target_q_value, _, _ = self.target.predict(next_state_goal_batch, hidden_state, cell_state, trace_length=self.trace_length, batch_size=batch_size)  # Next state
        target_q_value = target_q_value.detach().numpy()

        end_multiplier = -(done_batch - 1)
        double_Q = target_q_value[range(batch_size * self.trace_length), q_value_max]
        target_Q = reward_batch + (gamma * double_Q * end_multiplier)
        target_Q = torch.from_numpy(target_Q)

        # === Update the network with target values ===
        n_actions = len(q_value[0])  # get the number of actions
        actions_onehot = F.one_hot(torch.from_numpy(action_batch), n_actions)

        # Q-values of current state-goal
        q_out, _, _ = self.model.predict(state_goal_batch, hidden_state, cell_state, trace_length=self.trace_length, batch_size=batch_size)
        Q = torch.sum(torch.multiply(q_out, actions_onehot), dim=1)

        td_error = torch.square(target_Q - Q)

        if self.trace_length > 1:
            # Get loss on masked values
            maskA = torch.zeros(size=(batch_size, self.trace_length // 2))
            maskB = torch.ones(size=(batch_size, self.trace_length // 2))
            mask = torch.concat([maskA, maskB], dim=1)
            mask = torch.reshape(mask, shape=(-1,))

            loss = torch.mean(td_error * mask)
        else:
            loss = torch.mean(td_error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()