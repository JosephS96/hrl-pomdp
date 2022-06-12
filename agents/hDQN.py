from replay_buffer import ReplayBuffer
from common.schedules import LinearSchedule

import numpy as np
import random

from gym_minigrid.minigrid import Floor, SubGoal

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.optimizers import Adam


class hDQNAgent():
    def __init__(self,
                 env,
                 num_episodes=100,
                 meta_goals=[(2,2), (3,3), (4,4)],
                 goal_pos=(1,1),
                 render=False
                 ):
        self.env = env
        self.num_episodes = num_episodes
        self.render = render
        self.goal_pos = goal_pos

        # Replay Buffer
        self.buffer_size = 8000
        self.min_size_batch = 1000
        self.batch_size = 32
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Meta controller replay buffer
        self.min_meta_size_batch = 100
        self.meta_replay_buffer = ReplayBuffer(self.buffer_size)
        self.her = False

        # Meta controller
        self.meta_goals = meta_goals
        self.meta_goals.append(self.goal_pos)
        self.epsilon_meta = 1.0
        self.epsilon_min = 0.1
        self.meta_epsilon_scheduler = LinearSchedule(num_episodes, self.epsilon_min)

        # Hyper-parameters
        self.gamma = 0.99  # discount factor
        self.alpha = 0.001  # Learning rate
        self.epsilon_decay = 0.85
        self.epsilon = {}   # Exploration rate of every goal
        self.goal_success =  {}  # Times the agent reached the goal
        self.goal_selected = {} # Times the goal was selected
        for goal in self.meta_goals:
            self.epsilon[goal] = 1.0
            self.goal_success[goal] = 0.0
            self.goal_selected[goal] = 1.0

        # NN models for Double Deep Q Learning
        self.optimizer = Adam(learning_rate=self.alpha)
        # self.optimizer = SGD(learning_rate=self.alpha)
        self.model = self.__create_model(self.env.observation_space.shape, self.env.action_space.n)
        self.h_model = self.__create_model(self.env.observation_space.shape, len(self.meta_goals))

        self.mode = 'train'

    def __create_model(self, input_dim, output_dim):
        model = Sequential()
        model.add(InputLayer(input_shape=input_dim))  # , kernel_initializer = 'zeros')
        model.add(Dense(24, activation='relu'))  # , kernel_initializer = 'zeros'))
        model.add(Flatten())
        model.add(Dense(24, activation='relu'))  # , kernel_initializer = 'zeros'))
        model.add(Dense(output_dim, activation='linear'))  # , kernel_initializer = 'zeros'))
        model.compile(loss='mse', optimizer=self.optimizer)

        model.summary()

        return model

    def choose_action(self, state, epsilon):
        state = np.expand_dims(state, axis=0)
        if self.mode == 'train':
            action = self.__epsilon_greedy(state, epsilon)
            return action
        else:  # Only Exploitation
            output = self.model(state)
            action = np.argmax(output, axis=1)[0]
            return action

    # Epsilon greedy to choose the controller actions
    def __epsilon_greedy(self, state, epsilon):
        if random.random() > epsilon:  # Exploitation
            output = self.model(state)
            action = np.argmax(output, axis=1)[0]
            return action
        else: # Exploration
            # Reduce the exploration probability
            return random.randint(0, 3)
            # return self.env.action_space.sample()

    def choose_goal(self, state):
        state = np.expand_dims(state, axis=0)
        if self.mode == 'train':
            goal = self.__epsilon_greedy_meta(state)
            return goal
        else:  # Only Exploitation
            output = self.model(state)
            action = np.argmax(output, axis=1)[0]
            return action

    # Epsilon greedy for the meta controller to choose the goals
    # This returns the index for the meta_goal, not the actual goal
    def __epsilon_greedy_meta(self, state):
        if random.random() > self.epsilon_meta:
            output = self.h_model(state)
            goal_index = np.argmax(output, axis=1)[0]
            return goal_index
        else:
            goal_index = random.randint(0, len(self.meta_goals)-1)
            return goal_index

    def update_params(self):
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

    def update_meta_params(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.meta_replay_buffer.sample(self.batch_size)

        target_q_value = self.h_model.predict(state_batch)  # Current state
        q_value = self.h_model.predict(next_state_batch)  # Next state

        for index in range(self.batch_size):
            if done_batch[index]:
                target_q_value[index][action_batch[index]] = reward_batch[index]
            else:
                target_q_value[index][action_batch[index]] = reward_batch[index] + self.gamma * (np.amax(q_value[index]))

        # Train the model
        self.h_model.fit(state_batch, target_q_value, batch_size=self.batch_size, epochs=1, verbose=0)

    def intrinsic_reward(self, goal):
        agent_pos = self.env.agent_pos
        if type(agent_pos) is not tuple:
            agent_pos = (agent_pos[0], agent_pos[1])

        # If the agent has reached the subgoal, provide a reward
        return 1.0 if agent_pos == (goal or self.goal_pos) else 0.0

    def place_subgoal(self, old_pos, new_pos):
        # Remove the previous goal
        if old_pos is not None:
            self.env.grid.set(*old_pos, None)

        # Place the new subgoal in the environment
        if new_pos == self.goal_pos:
            return

        # self.env.grid.set(*new_pos, SubGoal())
        self.env.place_obj(SubGoal(), top=new_pos, size=(1, 1))

    def update_goal_epsilon(self, goal):
        success_rate = self.goal_success[goal] / self.goal_selected[goal]
        if success_rate > 0.5:
            # reduce epsilon
            self.epsilon[goal] = max(self.epsilon[goal] * self.epsilon_decay, self.epsilon_min)
        else:  #Increase epsilon
            self.epsilon[goal] = min(self.epsilon[goal] / self.epsilon_decay, 1.0)

    def get_goal_success_rate(self):
        success_rate = {}
        for goal in self.meta_goals:
            success = self.goal_success[goal] / self.goal_selected[goal]
            success_rate[goal] = success

        return success_rate

    def learn(self):

        reward_per_episode = []

        success_rate_history = []
        n_success = 0

        for episode in range(self.num_episodes):

            print(f" === Episode {episode} ===")

            state = self.env.reset()
            done = False
            t = 0
            episode_reward = 0
            action_history = []

            goal_idx = self.choose_goal(state) # Get some goal to look for
            goal = self.meta_goals[goal_idx]
            self.goal_selected[goal] += 1

            # self.env.place_obj(SubGoal(), top=goal, size=(1, 1))  # Place the new subgoal in the environment
            self.place_subgoal(None, goal)  # Place the new subgoal in the environment

            # Reduce the chance to explore the selected goal
            # self.epsilon = self.epsilon_scheduler.value(episode)
            # self.epsilon[self.meta_goals[goal_idx]] = self.update_goal_epsilon(goal)
            self.update_goal_epsilon(goal)

            if self.render:
                self.env.render()

            while not done:
                global_reward = 0  # reward
                initial_state = state
                r = 0  # intrinsic reward
                while not (done or r > 0):
                    # Choose action for movement in epsilon greedy
                    action = self.choose_action(state, self.epsilon[goal])
                    state_next, reward, done, _ = self.env.step(action)

                    r = self.intrinsic_reward(goal)  #intrinsic reward from subgoals

                    # Store transition
                    self.replay_buffer.add(state, action, r, state_next, done)
                    global_reward += reward
                    state = state_next
                    t += 1

                    if self.render:
                        self.env.render()

                    if self.replay_buffer.__len__() > self.min_size_batch:
                        self.update_params()

                    if self.meta_replay_buffer.__len__() > self.min_meta_size_batch:
                        self.update_meta_params()

                    action_history.append(action)
                    if r > 0:  # If subgoal was reached
                        self.goal_success[goal] += 1

                    episode_reward = global_reward

                # Save transitions for the meta controller, regarding goal and extrinsic rewards
                # Save the index of the selected goal
                self.meta_replay_buffer.add(initial_state, self.meta_goals.index(goal), global_reward, state, done)

                if done and (global_reward > 0):
                    n_success += 1

                # Logg stuff for debugging
                print(f'Intrinsic reward: {r}, Extrinsic_reward: {global_reward}, Current goal: {goal}')
                if not done:
                    # Remove the previous goal
                    # self.env.grid.set(*goal, None)
                    prev_goal = goal

                    goal_idx = self.choose_goal(state)  # Choose a new goal (returns index)
                    goal = self.meta_goals[goal_idx] # The actual goal coordinates
                    # self.epsilon[self.meta_goals[goal_idx]] = max(self.epsilon[self.meta_goals[goal_idx]] * self.epsilon_decay, 0.1)
                    self.update_goal_epsilon(goal)
                    self.goal_selected[goal] += 1

                    # Place the new subgoal in the environment
                    # self.env.grid.set(*goal, SubGoal())
                    self.place_subgoal(prev_goal, goal)
                    print(f"Changed environment subgoal to {goal}")

            # self.epsilon_meta = max(self.epsilon_meta - self.meta_epsilon_decay, self.epsilon_min)
            self.epsilon_meta = self.meta_epsilon_scheduler.value(episode)

            reward_per_episode.append(episode_reward)
            success_rate_history.append(n_success / (episode + 1))
            print(f"Success rate: {success_rate_history[-1]}")

            # print stuff for logging
            print(f'Episode: {episode}, Reward: {episode_reward}, Episode length: {t}')
            print(f'Epsilon: {self.epsilon}, Memory length: {self.replay_buffer.__len__()}')
            print(f'Goal success rate: {self.get_goal_success_rate()}')
            print(f'Epsilon Meta: {self.epsilon_meta}, Memory length: {self.meta_replay_buffer.__len__()}')
            print(f'Action history: {action_history[-15:]}')
            print('\n')

        if self.render:
            self.env.close()

        return reward_per_episode, success_rate_history


