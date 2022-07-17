import matplotlib.pyplot as plt
import numpy as np
import random
import time
import gym
import torch
from gym_minigrid.minigrid import SubGoal, Goal
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper

from common.Logger import Logger
from envs import StaticFourRoomsEnv
from envs.closed_four_rooms import ClosedFourRoomsEnv
from envs.randomempty import RandomEmpyEnv
from replay_buffer import ExperienceEpisodeReplayBuffer
from common.schedules import LinearSchedule
from networks.hdqn import RecurrentDDQN, RecurrentDDQNPyTorch
from common.utils import *


class HierarchicalDDRQNAgent:
    """
    Hierarchical
    Double Deep
    Recurrent
    Q-Network
    """
    def __init__(self, env,
                 num_episodes=100,
                 meta_goals=[(2, 2), (3, 3), (4, 4)],
                 goal_pos=None,
                 render=False):
        self.identifier = 'hierarchical-ddrqn'
        self.env = env
        self.num_episodes = num_episodes
        self.render = render
        self.goal_pos = goal_pos

        # Replay Buffer
        self.buffer_size = 3500
        self.min_size_batch = 400
        self.replay_buffer = ExperienceEpisodeReplayBuffer(self.buffer_size)
        self.batch_size = 4
        self.sub_trace_length = 8

        # Meta controller replay buffer
        self.meta_trace_length = 1
        self.meta_batch_size = 12
        self.min_meta_size_batch = 25
        self.meta_replay_buffer = ExperienceEpisodeReplayBuffer(self.buffer_size)

        # Meta controller
        self.meta_goals = meta_goals

        # Last position is always the goal position, starts as None
        self.meta_goals.append(self.goal_pos)  # Include goal position into the subgoals
        self.epsilon_meta = 1.0
        self.epsilon_min = 0.1
        self.meta_epsilon_scheduler = LinearSchedule(num_episodes, self.epsilon_min)

        # Hyper-parameters
        self.max_steps_per_goal = 500
        self.gamma = 0.99  # discount factor
        self.alpha = 0.001  # Learning rate
        self.epsilon_decay = 0.95
        self.epsilon = {}  # Exploration rate of every goal
        self.goal_success = {}  # Times the agent reached the goal
        self.goal_selected = {}  # Times the goal was selected
        for goal in self.meta_goals:
            self.epsilon[goal] = 1.0
            self.goal_success[goal] = 0.0
            self.goal_selected[goal] = 1.0

        # Steps to give before updating target model nn
        self.target_update_steps = 2000
        self.meta_target_update_steps = 4000
        self.model = RecurrentDDQNPyTorch(
            input_shape=self.env.observation_space.shape,
            output_shape=3,
            trace_n=self.sub_trace_length,
            meta=False
        )
        self.h_model = RecurrentDDQNPyTorch(
            input_shape=self.env.observation_space.shape,
            output_shape=len(self.meta_goals),
            trace_n=self.meta_trace_length,
            meta=True
        )

        self.mode = 'train'

    def choose_action(self, state, goal_state, epsilon):
        if self.mode == 'train':
            action = self.__epsilon_greedy(state, goal_state, epsilon)
            return action
        else:  # Only Exploitation
            output = self.model.predict(state, goal_state)
            action = np.argmax(output, axis=0)
            return action

    # Epsilon greedy to choose the controller actions
    def __epsilon_greedy(self, state, goal_state, epsilon):
        if random.random() > epsilon:  # Exploitation
            output = self.model.predict(state, goal_state)
            action = np.argmax(output, axis=0)
            return action
        else:  # Exploration
            # Reduce the exploration probability
            # Prediction is done in order to update the LSTM hidden state
            # but result is not used anyway
            _ = self.model.predict(state, goal_state)
            return random.randint(0, 2)
            # return self.env.action_space.sample()

    def choose_goal(self, state, goal_state):
        if self.mode == 'train':
            goal = self.__epsilon_greedy_meta(state, goal_state)
            return goal
        else:  # Only Exploitation
            output = self.h_model.predict(state, goal_state)
            action = np.argmax(output, axis=0)
            return action

    # Epsilon greedy for the meta controller to choose the goals
    # This returns the index for the meta_goal, not the actual goal
    def __epsilon_greedy_meta(self, state, goal_state):
        if random.random() > self.epsilon_meta:
            output = self.h_model.predict(state, goal_state)
            goal_index = np.argmax(output, axis=0)
            return goal_index
        else:
            _ = self.h_model.predict(state, goal_state)
            goal_index = random.randint(0, len(self.meta_goals) - 1)
            return goal_index

    def intrinsic_reward(self, goal):
        agent_pos = self.env.agent_pos
        if type(agent_pos) is not tuple:
            agent_pos = (agent_pos[0], agent_pos[1])

        # If the agent has reached the subgoal, provide a reward
        # return 1.0 if agent_pos == (self.meta_goals[goal] or self.goal_pos) else 0.0
        return 1.0 if agent_pos == (self.meta_goals[goal]) else 0

    def place_subgoal(self, old_pos, new_pos):
        # Remove the previous goal
        # Do not remove previous goal if:
        # Goal is None or if previous goal is the global goal
        if (old_pos is not None):
            old_pos = self.meta_goals[old_pos]
            self.env.grid.set(*old_pos, None)

            # If the old position was the global Goal, put it back on the environment
            if old_pos == self.goal_pos:
                self.env.grid.set(*old_pos, Goal())

        # Place the new subgoal in the environment
        # if new_pos == self.meta_goals.index(self.goal_pos):
        #     return

        new_pos = self.meta_goals[new_pos]
        self.env.grid.set(*new_pos, SubGoal())
        # self.env.place_obj(SubGoal(), top=new_pos, size=(1, 1))

    def update_goal_epsilon(self, goal):
        goal = self.meta_goals[goal]
        success_rate = self.goal_success[goal] / self.goal_selected[goal]
        if success_rate > 0.5:
            # reduce epsilon
            self.epsilon[goal] = max(self.epsilon[goal] * self.epsilon_decay, self.epsilon_min)
        else:  # Increase epsilon
            # This will ensure that the goal epsilon will decrease over time
            self.epsilon[goal] = min(self.epsilon[goal] / self.epsilon_decay, self.epsilon_meta)

    def get_goal_success_rate(self):
        success_rate = {}
        for goal in self.meta_goals:
            success = self.goal_success[goal] / self.goal_selected[goal]
            success_rate[goal] = success

        return success_rate

    # Put the state values between 0.0 and 1.0
    def norm_state(self, state):
        return state / 10.0

    def process_state(self, state):
        dir_state = np.ones(shape=(state.shape[0], state.shape[1], 1)) * self.env.agent_dir

        # Put the state values between 0.0 and 1.0
        state = state / 10.0
        dir_state = dir_state / 3

        state = np.concatenate([state, dir_state], axis=2)

        return state

    def extrinsic_reward(self):
        agent_pos = self.env.agent_pos
        if type(agent_pos) is not tuple:
            agent_pos = (agent_pos[0], agent_pos[1])

        reward = 1 - 0.9 * (self.env.step_count / self.env.max_steps)

        # If the agent has reached the subgoal, provide a reward
        # return 1.0 if agent_pos == (self.meta_goals[goal] or self.goal_pos) else 0.0
        return reward if agent_pos == (self.goal_pos) else 0.0

    def learn(self):

        update_nn_steps = 0
        n_success = 0

        logger = Logger()

        for episode in range(self.num_episodes):
            episode_time_start = time.time()
            print(f" === Episode {episode} ===")

            meta_episode_buffer = []
            sub_episode_buffer = []

            state = self.env.reset()
            state = self.process_state(state)
            done = False
            t = 0
            episode_reward = 0
            intrinsic_reward_per_episode = 0
            action_history = []

            # Reset meta-controller hidden state
            self.h_model.init_hidden()

            # Start selecting the final goal
            goal = len(self.meta_goals) - 1
            goal_state = get_subview(self.env, self.meta_goals[goal])  # Obtain view of the subgoal
            goal_state = self.norm_state(goal_state)
            goal = self.choose_goal(state, goal_state)  # Get some goal to look for
            self.goal_selected[self.meta_goals[goal]] += 1

            # self.env.place_obj(SubGoal(), top=goal, size=(1, 1))  # Place the new subgoal in the environment
            self.place_subgoal(None, goal)  # Place the new subgoal in the environment

            # Reduce the chance to explore the selected goal
            # self.epsilon = self.epsilon_scheduler.value(episode)
            # self.epsilon[self.meta_goals[goal_idx]] = self.update_goal_epsilon(goal)
            self.update_goal_epsilon(goal)

            if self.render:
                self.env.render()

            # Global goal
            while not done:
                global_reward = 0  # reward
                initial_state = state
                r = 0  # intrinsic reward
                steps_per_goal = 0

                # Reset sub-controller hidden state to zeros
                # Every time a new subgoal is chosen
                self.model.init_hidden()

                # Global reward or intrinsic reward
                while not (done or r > 0):
                    update_nn_steps += 1
                    # Choose action for movement in epsilon greedy
                    goal_state = get_subview(self.env, self.meta_goals[goal])
                    goal_state = self.norm_state(goal_state)
                    action = self.choose_action(state, goal_state, self.epsilon[self.meta_goals[goal]])

                    state_next, reward, done, _ = self.env.step(action)

                    reward = self.extrinsic_reward()
                    if reward > 0:
                        done = True

                    state_next = self.process_state(state_next)

                    r = self.intrinsic_reward(goal)  # intrinsic reward from subgoals
                    intrinsic_reward_per_episode += r
                    intrinsic_done = r > 0

                    # Save rewards per episode for metrics
                    logger.intrinsic_reward_per_step.append(r)
                    logger.extrinsic_reward_per_step.append(reward)

                    # Store transition
                    transition = np.reshape(
                        np.array([state, action, r, state_next, self.norm_state(get_subview(self.env, self.meta_goals[goal])), intrinsic_done]),
                        newshape=(1, 6)
                    )
                    sub_episode_buffer.append(transition)
                    global_reward += reward
                    state = state_next
                    t += 1

                    if self.render:
                        self.env.render()

                    if self.replay_buffer.__len__() > self.min_size_batch:
                        batch_memory = self.replay_buffer.sample(self.batch_size, self.sub_trace_length)
                        # self.model.update_params(batch_memory, self.gamma)
                        self.model.update_params2(batch_memory, self.gamma, self.batch_size)

                    if self.meta_replay_buffer.__len__() > self.min_meta_size_batch:
                        batch_memory = self.meta_replay_buffer.sample(self.meta_batch_size, self.meta_trace_length)
                        # self.h_model.update_params(batch_memory, self.gamma)
                        self.h_model.update_params2(batch_memory, self.gamma, self.meta_batch_size)

                    action_history.append(action)
                    if r > 0:  # If subgoal was reached
                        self.goal_success[self.meta_goals[goal]] += 1

                    episode_reward = global_reward

                    # If the max number of steps has been achieved for a given goal
                    # break this loops and choose a different one
                    if steps_per_goal == self.max_steps_per_goal:
                        break

                    steps_per_goal += 1
                    # Finish one step or transition in real environment

                # Finish of H steps on environment
                # === Sub-Controller ====

                # Save transitions for the meta controller, regarding goal and extrinsic rewards
                # Save the index of the selected goal
                transition = np.reshape(
                    np.array([initial_state, goal, global_reward, state, self.norm_state(get_subview(self.env, self.meta_goals[goal])), done]),
                    newshape=(1, 6)
                )
                meta_episode_buffer.append(transition)

                if done and (global_reward > 0):
                    n_success += 1

                if self.target_update_steps % update_nn_steps == 0:
                    self.model.update_target_network()

                if self.meta_target_update_steps % update_nn_steps == 0:
                    self.h_model.update_target_network()

                if len(sub_episode_buffer) > self.sub_trace_length:
                    self.replay_buffer.add(sub_episode_buffer)

                # Logg stuff for debugging
                print(f'Intrinsic reward: {r}, Extrinsic_reward: {global_reward}, Current goal: {self.meta_goals[goal]}')

                # If the goal has not been achieved
                if not done:
                    # Remove the previous goal
                    # self.env.grid.set(*goal, None)
                    prev_goal = goal

                    prev_goal_state = get_subview(self.env, self.meta_goals[prev_goal])
                    prev_goal_state = self.norm_state(prev_goal_state)
                    goal = self.choose_goal(state, prev_goal_state)  # Choose a new goal (returns index)
                    # self.epsilon[self.meta_goals[goal_idx]] = max(self.epsilon[self.meta_goals[goal_idx]] * self.epsilon_decay, 0.1)
                    self.update_goal_epsilon(goal)
                    self.goal_selected[self.meta_goals[goal]] += 1

                    # Place the new subgoal in the environment
                    # self.env.grid.set(*goal, SubGoal())
                    self.place_subgoal(prev_goal, goal)
                    print(f"Changed environment subgoal to {self.meta_goals[goal]}")

            # self.epsilon_meta = max(self.epsilon_meta - self.meta_epsilon_decay, self.epsilon_min)
            self.epsilon_meta = self.meta_epsilon_scheduler.value(episode)

            if len(meta_episode_buffer) > self.meta_trace_length:
                self.meta_replay_buffer.add(meta_episode_buffer)

            logger.success_rate.append(n_success / (episode + 1))
            logger.extrinsic_reward_per_episode.append(episode_reward)
            logger.intrinsic_reward_per_episode.append(intrinsic_reward_per_episode)
            logger.steps_per_episode.append(t)

            print(f"Success rate: {logger.success_rate[-1]}")

            # print stuff for logging
            print(f'Episode: {episode}, Reward: {episode_reward}, Episode length: {t}')
            print(f'Epsilon: {self.epsilon}, Memory length: {self.replay_buffer.__len__()}')
            print(f'Goal success rate: {self.get_goal_success_rate()}')
            print(f'Epsilon Meta: {self.epsilon_meta}, Memory length: {self.meta_replay_buffer.__len__()}')
            print(f'Action history: {action_history[-15:]}')
            print(f'Episode duration in seconds: {time.time() - episode_time_start}')
            print('\n')

            if episode > 10:
                logger.print_latest_statistics()

        if self.render:
            self.env.close()

        return logger


if __name__ == "__main__":
    print(torch.initial_seed())

    """
    env_name = 'MiniGrid-Empty-11x11'
    env = RandomEmpyEnv(grid_size=11, max_steps=400, goal_pos=(9, 9), agent_pos=(1, 1))
    
        sub_goals = [
            (2, 2), (2, 5), (2, 8),
            (5, 2), (5, 5), (5, 8),
            (8, 2), (8, 5), (8, 8),
        ]
    """

    """
    env_name = "StaticFourRooms-11x11"
    env = StaticFourRoomsEnv(agent_pos=(2, 2), goal_pos=(9, 9), grid_size=11, max_steps=400)
    sub_goals = [
        (2, 2), (2, 5), (2, 8),
        (5, 2), (5, 8),
        (8, 2), (8, 5), (8, 8),
    ]
    """

    env_name = "ClosedFourRooms-11x11"
    goal_pos = (9, 1)
    env = ClosedFourRoomsEnv(agent_pos=(2, 2), goal_pos=goal_pos, grid_size=11, max_steps=400)
    sub_goals = [
        (2, 8),     (8, 2),
        (2, 5),     (8, 5),
        (2, 8), (5, 8), (8, 8),
    ]

    # env = gym.make(env_name)
    # env = gym.make('MiniGrid-Empty-8x8-v0', size=11)
    # env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    agent = HierarchicalDDRQNAgent(env=env, num_episodes=500, render=True, meta_goals=sub_goals, goal_pos=goal_pos)
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