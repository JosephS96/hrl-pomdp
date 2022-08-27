import matplotlib.pyplot as plt
import numpy as np
import random
import time
import gym
from gym_minigrid.minigrid import SubGoal, Goal
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper

from common.Logger import Logger
from envs import StaticFourRoomsEnv
from envs.closed_four_rooms import ClosedFourRoomsEnv
from envs.randomempty import RandomEmpyEnv
from replay_buffer import ExperienceEpisodeReplayBuffer
from common.schedules import LinearSchedule
from networks.hdqn import RecurrentDDQNPyTorch
from common.utils import *


class HierarchicalDDRQNAgent:
    """
    Hierarchical
    Double Deep
    Recurrent
    Q-Network
    Hindsight Experience Replay
    """
    def __init__(self, env,
                 num_episodes=100,
                 meta_goals=[(2, 2), (3, 3), (4, 4)],
                 goal_pos=None,
                 render=False):
        self.identifier = 'hierarchical-ddrqn-her'
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
        self.max_steps_per_goal = 25
        self.is_subgoal_test = False  # Whether the current goal is being tested
        self.update_steps = 5

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
        if self.mode == 'train' and not self.is_subgoal_test:
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
            _ = self.model.predict(state, goal_state)
            return random.randint(0, 2)
            # return self.env.action_space.sample()

    def choose_goal(self, state, goal_state):
        if self.mode == 'train' and not self.is_subgoal_test:
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

        env_intrinsic = 0.0
        transition_intrinsic = -1.0
        if agent_pos == self.meta_goals[goal]:
            env_intrinsic = 1.0
            transition_intrinsic = 1.0

        # If the agent has reached the subgoal, provide a reward
        # otherwise penalize. Hindsight Action Transition
        return env_intrinsic, transition_intrinsic

    def get_subgoal_reached(self):
        agent_pos = self.env.agent_pos
        if type(agent_pos) is not tuple:
            agent_pos = (agent_pos[0], agent_pos[1])

        if agent_pos in self.meta_goals:
            return self.meta_goals.index(agent_pos)
        else:
            return None

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

        env_reward = 0
        transition_reward = -1.0
        if agent_pos == (self.goal_pos):
            # This is for logging
            env_reward = 1 - 0.9 * (self.env.step_count / self.env.max_steps)

            # If the agent has reached the subgoal, provide a reward
            # This is for storing in transitions
            transition_reward = 1.0

        # return 1.0 if agent_pos == (self.meta_goals[goal] or self.goal_pos) else 0.0
        return env_reward, transition_reward

    def learn(self):

        update_nn_steps = 0
        n_success = 0

        global_goal = len(self.meta_goals) - 1
        global_goal_state = get_subview(self.env, self.meta_goals[global_goal])
        global_goal_state = self.norm_state(global_goal_state)

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

            # Reset meta controller hidden state at the beginning of every episode
            self.h_model.init_hidden()

            # Start selecting the final goal
            goal = self.choose_goal(state, global_goal_state)  # Get some goal to look for
            self.goal_selected[self.meta_goals[goal]] += 1

            # self.env.place_obj(SubGoal(), top=goal, size=(1, 1))  # Place the new subgoal in the environment
            self.place_subgoal(None, goal)  # Place the new subgoal in the environment

            # Reduce the chance to explore the selected goal
            # self.epsilon = self.epsilon_scheduler.value(episode)
            # self.epsilon[self.meta_goals[goal_idx]] = self.update_goal_epsilon(goal)
            self.update_goal_epsilon(goal)

            if self.render:
                self.env.render()

            meta_goal_transitions = []

            # Global goal
            # === Meta-Controller ===
            while not done:
                global_reward = 0  # reward
                initial_state = state
                r = 0  # intrinsic reward
                steps_per_goal = 0

                # Determine whether to test the subgoal
                self.is_subgoal_test = np.random.random_sample() < 0.2

                sub_goal_transitions = []

                # reset sub-controller hidden state to zeros
                self.model.init_hidden()

                # Global reward or intrinsic reward
                # === Sub-Controller ====
                while not (done or r > 0):
                    update_nn_steps += 1
                    # Choose action for movement in epsilon greedy
                    goal_state = get_subview(self.env, self.meta_goals[goal])
                    goal_state = self.norm_state(goal_state)
                    action = self.choose_action(state, goal_state, self.epsilon[self.meta_goals[goal]])

                    # Make a step in the environment
                    state_next, reward, done, _ = self.env.step(action)
                    state_next = self.process_state(state_next)  # Add direction

                    reward, transition_reward = self.extrinsic_reward()
                    if transition_reward > 0:
                        done = True

                    r, transition_intrinsic = self.intrinsic_reward(goal)
                    intrinsic_done = r > 0
                    intrinsic_reward_per_episode += r

                    # Save rewards per episode for metrics
                    logger.intrinsic_reward_per_step.append(r)
                    logger.extrinsic_reward_per_step.append(reward)

                    # Hindsight Action Transition - SubController
                    # Store transition
                    transition = np.reshape(
                        np.array([state, action, transition_intrinsic, state_next, self.norm_state(get_subview(self.env, self.meta_goals[goal])), intrinsic_done]),
                        newshape=(1, 6)
                    )
                    sub_episode_buffer.append(transition)

                    # Copy for goal transition
                    sub_goal_transitions.append([state, action, -1.0, state_next, None, intrinsic_done])

                    global_reward += reward
                    state = state_next
                    t += 1

                    if self.render:
                        self.env.render()

                    if self.replay_buffer.__len__() > self.min_size_batch and t % self.update_steps == 0:
                        batch_memory = self.replay_buffer.sample(self.batch_size, self.sub_trace_length)
                        # self.model.update_params(batch_memory, self.gamma)
                        self.model.update_params2(batch_memory, self.gamma, self.batch_size)

                    if self.meta_replay_buffer.__len__() > self.min_meta_size_batch and t % self.update_steps == 0:
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

                # Hindsight Goal Transitions  - Sub Controller
                # If the agent ended in any of the subgoals, utilize that subgoal as the
                # goal intended for the transition
                goal_reached = self.get_subgoal_reached()

                if goal_reached is not None and len(sub_goal_transitions) > self.sub_trace_length:
                    goal_transition_buffer = []
                    sub_goal_transitions[-1][2] = 1  # set reward to 1, because goal was reached
                    sub_goal_transitions[-1][5] = True  # Set done to True, because a goal was achieved
                    new_goal_state = get_subview(self.env, self.meta_goals[goal])
                    # print(" ===== Create Hindisght Goal Transitions! SUB======")
                    for transition in sub_goal_transitions:
                        transition[4] = new_goal_state
                        new_transition = np.reshape(np.array([transition[0], transition[1], transition[2], transition[3],new_goal_state, transition[5]]),newshape=(1, 6))
                        goal_transition_buffer.append(new_transition)
                    self.replay_buffer.add(goal_transition_buffer)

                # If subgoal testes and not achieved, penalize
                if self.is_subgoal_test and self.intrinsic_reward(goal) == 0:
                    transition = np.reshape(
                        np.array([initial_state, goal, -self.max_steps_per_goal, state,
                                  get_subview(self.env, self.meta_goals[goal]), done]),
                        newshape=(1, 6)
                    )
                    meta_episode_buffer.append(transition)
                    self.is_subgoal_test = False

                # Hindsight Action Transition - Meta Controller
                # Save transitions for the meta controller, regarding goal and extrinsic rewards
                # Save the index of the selected goal
                if goal_reached is not None:
                    meta_transition_reward = global_reward if global_reward > 0 else -1
                    # print(" ===== Create Hindisght Goal Transitions! META======")
                    transition = np.reshape(
                        np.array([initial_state, goal_reached, meta_transition_reward, state, get_subview(self.env, self.meta_goals[goal]), done]),
                        newshape=(1, 6)
                    )
                    meta_episode_buffer.append(transition)
                    meta_goal_transitions.append(transition)

                if done and (global_reward > 0):
                    n_success += 1

                if update_nn_steps % self.target_update_steps == 0:
                    self.model.update_target_network()

                if update_nn_steps % self.meta_target_update_steps == 0:
                    self.h_model.update_target_network()

                # Corresponds to a list of at least n transitions
                if len(sub_episode_buffer) > self.sub_trace_length:
                    self.replay_buffer.add(sub_episode_buffer)

                # Logg stuff for debugging
                # print(f'Intrinsic reward: {r}, Extrinsic_reward: {global_reward}, Current goal: {self.meta_goals[goal]}')

                # If the goal has not been achieved
                # Choose a new goal
                if not done:
                    # Remove the previous goal
                    # self.env.grid.set(*goal, None)
                    prev_goal = goal

                    # prev_goal_state = get_subview(self.env, self.meta_goals[prev_goal])
                    # prev_goal_state = self.norm_state(prev_goal_state)
                    goal = self.choose_goal(state, global_goal_state)  # Choose a new goal (returns index)
                    # self.epsilon[self.meta_goals[goal_idx]] = max(self.epsilon[self.meta_goals[goal_idx]] * self.epsilon_decay, 0.1)
                    self.update_goal_epsilon(goal)
                    self.goal_selected[self.meta_goals[goal]] += 1

                    # Place the new subgoal in the environment
                    # self.env.grid.set(*goal, SubGoal())
                    self.place_subgoal(prev_goal, goal)
                    # print(f"Changed environment subgoal to {self.meta_goals[goal]}")

            # After Choosing N amount of subgoals

            # Hindsightht Goal Transitions - Meta Controller
            goal_reached = self.get_subgoal_reached()
            if goal_reached is not None:
                temp_buffer = []
                reached_goal_state = get_subview(self.env, self.meta_goals[goal_reached])
                meta_goal_transitions[-1][0][2] = 1
                meta_goal_transitions[-1][0][5] = True
                # print(" ===== Create Hindisght Action Transitions! META======")
                for transition in meta_goal_transitions:
                    transition[0][4] = reached_goal_state
                    new_transition = np.reshape(np.array([transition[0][0], transition[0][1], transition[0][2], transition[0][3],reached_goal_state, transition[0][5]]),newshape=(1, 6))
                    temp_buffer.append(new_transition)
                self.meta_replay_buffer.add(temp_buffer)

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
    PATH = "/Users/josesanchez/Documents/IAS/Thesis-Results"

    env_name = 'MiniGrid-Empty-11x11'
    goal_pos=(9,9)
    env = RandomEmpyEnv(grid_size=11, max_steps=400, goal_pos=goal_pos, agent_pos=(1, 1))

    sub_goals = [
        (2, 2), (2, 5), (2, 8),
        (5, 2), (5, 5), (5, 8),
        (8, 2), (8, 5), (8, 8),
    ]

    """
    env_name = "StaticFourRooms-11x11"
    goal_pos = (8, 7)
    env = StaticFourRoomsEnv(agent_pos=(2, 2), goal_pos=goal_pos, grid_size=11, max_steps=400)
    sub_goals = [
        (2, 2), (8, 2),
        (3, 3), (7, 3), (5, 2), (5, 3),
        (2, 5), (3, 5), (7, 5), (8, 5),
        (3, 7), (5, 7), (7, 7),
        (2, 8), (5, 8), (8, 8),
    ]
    """

    """
    env_name = "ClosedFourRooms-11x11"
    goal_pos = (8, 7)
    env = ClosedFourRoomsEnv(agent_pos=(2, 2), goal_pos=goal_pos, grid_size=11, max_steps=400)
    sub_goals = [
        (2, 2), (8, 2),
        (3, 3), (7, 3),
        (2, 5), (3, 5), (7, 5), (8, 5),
        (3, 7), (5, 7), (7, 7),
        (2, 8), (5, 8), (8, 8)
    ]
    """
    # Only get the image as observation
    env = ImgObsWrapper(env)

    agent = HierarchicalDDRQNAgent(env=env, num_episodes=1000, render=False, meta_goals=sub_goals, goal_pos=goal_pos)
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