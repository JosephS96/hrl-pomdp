import random

import numpy as np

from replay_buffer import ExperienceBuffer

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer, Flatten


class Layer:
    def __init__(self, input_dim, output_dim, level):

        self.level = level   # Each layer needs to be aware of its own level
        self.max_subgoal_steps = 10
        self.model = self.__create_model(input_dim, output_dim)
        self.replay_buffer = ExperienceBuffer(size=3000)
        self.batch_size = 64
        self.gamma = 0.95

    def __create_model(self, input_dim, output_dim):
        model = Sequential()
        model.add(InputLayer(input_shape=input_dim))  # , kernel_initializer = 'zeros')
        model.add(Dense(24, activation='relu'))  # , kernel_initializer = 'zeros'))
        model.add(Flatten())
        model.add(Dense(24, activation='relu'))  # , kernel_initializer = 'zeros'))

        if self.level > 0:
            model.add(Dense(output_dim, activation='relu'))
        else:
            model.add(Dense(output_dim, activation='linear'))

        model.compile(loss='mse', optimizer='adam')

        model.summary()

        return model

    def update_params(self, num_updates):
        state_batch, action_batch, reward_batch, next_state_batch, goal_batch, done_batch = self.replay_buffer.sample(
            self.batch_size)

        target_q_value = self.model.predict(state_batch)  # Current state
        q_value = self.model.predict(next_state_batch)  # Next state

        for index in range(self.batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                target_q_value[index][action_batch[index]] = reward_batch[index] + self.gamma * (
                    np.amax(q_value[index]))

        # Train the model
        self.model.fit(state_batch, target_q_value, batch_size=self.batch_size, epochs=1, verbose=0)

    def goal_reached(self, agent, goal):
        agent_pos = agent.env.agent_pos
        if type(agent_pos) is not tuple:
            agent_pos = (agent_pos[0], agent_pos[1])

        # check whether the agent reached the subgoal
        return agent_pos == goal

    def get_random_subgoal(self):
        # TODO: Only return reasonable subgoals that are not overlapping
        x = random.randint(1, 15)
        y = random.randint(1, 15)

        return (x, y)

    def choose_subgoal(self, state, goal):
        state = np.expand_dims(state, axis=0)
        if np.random.sample() > 0.2:
            output = self.model(state)[0].numpy()
            goal = (round(output[0]), round(output[1]))
        else:
            goal = self.get_random_subgoal()

        return goal

    def choose_action(self, state, goal):
        state = np.expand_dims(state, axis=0)
        if np.random.sample() > 0.2:
            output = self.model(state)
            action = np.argmax(output, axis=1)[0]
        else:
            action = random.randint(0, 2)

        return action

    # This is basically the equivalent of the TRAIN-LEVEL function in the pseudocode
    def learn(self, level, state, goal, agent):
        next_state = None
        is_subgoal_test = False
        is_next_subgoal_test = False
        done = False

        goal_transitions = []

        for _ in range(self.max_subgoal_steps):
            action = None
            if level > 0:
                action = self.choose_subgoal(state, goal)
            else:
                action = self.choose_action(state, goal)   # Sample noisy action form policy

            # ====== High Level Policy =======
            if level > 0:
                if not is_subgoal_test:
                    action = self.choose_subgoal(state, goal)

                # Determine whether to test subgoal action
                if np.random.sample() > 0.6:
                    is_next_subgoal_test = True

                # Train next sub-level of the hierarchy using action as subgoal
                # Pass subgoal to next layer
                next_state, done = agent.layers[level-1].learn(level-1, state, action, agent)

                # if subgoal was tested but not achieved, add subgoal testing transition
                if is_next_subgoal_test and not self.goal_reached(agent, action):
                    self.replay_buffer.add(state, action, -self.max_subgoal_steps, next_state, goal, done)

                # For hindsight action transition
                action = next_state

            # ====== Low Level Policy =======
            else:
                if not is_subgoal_test:
                    action = self.choose_action(state, goal)

                # Execute primitive action and observe new state
                next_state, reward, done, _ = agent.env.step(action)

                if agent.render:
                    agent.env.render()

            # check if goal was achieved
            goal_achieved = self.goal_reached(agent, goal)

            # hindsight action transition
            if goal_achieved:
                self.replay_buffer.add(state, action, 1.0, next_state, goal, done)
            else:
                self.replay_buffer.add(state, action, -1.0, next_state, goal, done)

            # copy for goal transitions
            goal_transitions.append([state, action, -1.0, next_state, None, done])

            # TODO: Might need to replace this next state with agent position
            state = next_state

            if done or goal_achieved:
                break

        # Finish H attemps/steps

        # hindsight goal transition
        # last transition reward is 0
        goal_transitions[-1][2] = 0
        for transition in goal_transitions:
            # last state is goal for all transitions
            transition[4] = next_state
            self.replay_buffer.add(transition[0], transition[1],transition[2],transition[3],transition[4], transition[5],)

        # Output current state: next_state
        return next_state, done