import random
import time

import numpy as np
from gym_minigrid.minigrid import Goal, SubGoal
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from matplotlib import pyplot as plt

from common.Logger import Logger
from common.schedules import LinearSchedule
from common.utils import get_subview
from envs import StaticFourRoomsEnv
from envs.closed_four_rooms import ClosedFourRoomsEnv
from envs.hdrqn_fourrooms import Fourrooms
from networks.conv_dqn import ConvRecurrentDQN
from replay_buffer import ExperienceEpisodeReplayBuffer
import cv2 as cv2
from PIL import Image


class Conv_hDDRQNAgent:
    def __init__(
            self,
            env,
            num_episodes=100,
            meta_goals=[],
            goal_pos=None,
            render=False
    ):
        self.identifier = 'conv-hierarchical-ddrqn-her'
        self.env = env
        self.num_episodes = num_episodes
        self.render = render
        self.goal_pos = goal_pos

        # Hyper-parameters
        self.max_steps_per_goal = 25
        self.gamma = 0.99  # discount factor
        self.alpha = 0.001  # Learning rate
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.1
        self.is_subgoal_test = False  # Whether the current goal is being tested
        self.update_steps = 5 # Update networks every n steps

        # == Meta Controller ==

        self.meta_goals = meta_goals
        self.meta_target_update_steps = 2000
        # Last position is always the goal position, starts as None
        self.meta_goals.append(self.goal_pos)  # Include goal position into the subgoals
        self.epsilon_meta = 1.0
        self.meta_epsilon_scheduler = LinearSchedule(num_episodes, self.epsilon_min)

        # Replay buffer
        self.meta_trace_length = 1
        self.meta_batch_size = 12
        self.min_meta_size_batch = 25
        self.meta_buffer_size = 1000
        self.meta_replay_buffer = ExperienceEpisodeReplayBuffer(self.meta_buffer_size)

        # Network
        self.meta_controller = ConvRecurrentDQN(
            input_shape=self.env.observation_space.shape,
            output_shape=len(self.meta_goals),
            trace_n=self.meta_trace_length,
        )

        # == Sub Controller ==

        self.target_update_steps = 1000
        self.epsilon = 1.0
        self.epsilon_scheduler = LinearSchedule(num_episodes, self.epsilon_min)

        # Replay Buffer
        self.sub_trace_length = 8
        self.batch_size = 4
        self.min_size_batch = 400
        self.buffer_size = 3500
        self.replay_buffer = ExperienceEpisodeReplayBuffer(self.buffer_size)

        # Network
        self.sub_controller = ConvRecurrentDQN(
            input_shape=self.env.observation_space.shape,
            output_shape=3,
            trace_n=self.sub_trace_length,
        )

    def choose_action(self, state, goal_state):
        if not self.is_subgoal_test:
            if random.random() > self.epsilon: # Exploitation
                output = self.sub_controller.predict(state, goal_state)
                action = np.argmax(output, axis=0)
                return action
            else:
                # Change the hidden state
                _ = self.sub_controller.predict(state, goal_state)
                return random.randint(0, 2)
        else:
            output = self.sub_controller.predict(state, goal_state)
            action = np.argmax(output, axis=0)
            return action

    # Goal state should always be the global goal
    def choose_goal(self, state, goal_state):
        if not self.is_subgoal_test:
            if random.random() > self.epsilon: # Exploitation
                output = self.meta_controller.predict(state, goal_state)
                goal_index = np.argmax(output, axis=0)
                return goal_index
            else:
                # Change the hidden state
                _ = self.meta_controller.predict(state, goal_state)
                goal_index = random.randint(0, len(self.meta_goals) - 1)
                return goal_index
        else:
            output = self.meta_controller.predict(state, goal_state)
            goal_index = np.argmax(output, axis=0)
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

    def process_state(self, state):
        img = Image.fromarray(state)
        img = img.resize(size=(44, 44))
        state = np.asarray(img) / 255.0

        return state

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

    def create_hindsight_goal_transition(self, transitions, reached_goal):
        goal_transition_buffer = []
        transitions[-1][0][2] = 1  # set reward to 1, because goal was reached
        transitions[-1][0][5] = True  # Set done to True, because a goal was achieved
        new_goal_state = get_subview(self.env, self.meta_goals[reached_goal], rgb=True)
        new_goal_state = self.process_state(new_goal_state)
        for transition in transitions:
            transition[0][4] = new_goal_state
            goal_transition = np.array(transition)
            goal_transition_buffer.append(goal_transition)

        return goal_transition_buffer

    def create_transition(self, state, action, reward, state_next, goal, done):
        transition = np.reshape(
            np.array([state, action, reward, state_next, goal, done]),
            newshape=(1, 6)
        )
        return transition

    def learn(self):
        # Steps to update the target networks
        update_nn_steps = 0
        n_success = 0

        global_goal = len(self.meta_goals) - 1
        global_goal_state = get_subview(self.env, self.meta_goals[global_goal], rgb=True)
        global_goal_state = self.process_state(global_goal_state)

        logger = Logger()

        for episode in range(self.num_episodes):

            print(f" === Episode {episode} ===")
            episode_time_start = time.time()

            # Store individual transitions to save to the replay buffer
            meta_episode_buffer = []
            sub_episode_buffer = []

            # For Hindsight Goal Transitions
            meta_goal_transitions = []

            state = self.env.reset()
            state = self.process_state(state)

            done = False
            timestep = 0
            episode_reward = 0
            episode_intrinsic_reward = 0

            # Update epsilon values
            self.epsilon_meta = self.meta_epsilon_scheduler.value(episode)
            self.epsilon = self.epsilon_scheduler.value(episode)

            # Reset meta controller hidden state at the beginning of every episode
            self.meta_controller.init_hidden(trace_length=1)

            # Choose a goal
            goal = self.choose_goal(state, global_goal_state)

            # Place the new subgoal in the environment
            self.place_subgoal(None, goal)

            if self.render:
                self.env.render()

            # == Meta controller horizon ==
            while not done:
                initial_state = state

                # reset sub-controller hidden state to zeros
                self.sub_controller.init_hidden(trace_length=1)

                # Determine whether to test subgoal
                self.is_subgoal_test = np.random.random_sample() < 0.2

                # For saving Hindsight Goal Transitions
                sub_goal_transitions = []

                # Choose goal
                prev_goal = goal
                goal = self.choose_goal(state, global_goal_state)

                # Place the new subgoal in the environment
                # self.env.grid.set(*goal, SubGoal())
                self.place_subgoal(prev_goal, goal)
                # print(f"Changed environment subgoal to {self.meta_goals[goal]}")

                #  === Start H steps ===
                for _ in range(self.max_steps_per_goal):
                    update_nn_steps += 1

                    goal_state = get_subview(self.env, self.meta_goals[goal], rgb=True)
                    goal_state = self.process_state(goal_state)
                    action = self.choose_action(state, goal_state)

                    # Make a step in the environment
                    state_next, reward, done, _ = self.env.step(action)
                    state_next = self.process_state(state_next)

                    reward, transition_reward = self.extrinsic_reward()
                    if transition_reward > 0:
                        done = True

                    intrinsic_reward, transition_intrinsic = self.intrinsic_reward(goal)
                    intrinsic_done = intrinsic_reward > 0  # If reward was not zero, a subgoal was reached

                    # Store current transition with: Hindsight Action Transition
                    transition = self.create_transition(state, action, transition_intrinsic, state_next, goal_state, intrinsic_done)
                    sub_episode_buffer.append(transition)

                    sub_goal_transitions.append(transition)

                    # Statistics for loggings
                    episode_reward += reward
                    episode_intrinsic_reward += intrinsic_reward

                    logger.intrinsic_reward_per_step.append(intrinsic_reward)
                    logger.extrinsic_reward_per_step.append(reward)

                    state = state_next
                    timestep += 1

                    if self.render:
                        self.env.render()

                    if self.replay_buffer.__len__() > self.min_size_batch and timestep % self.update_steps == 0:
                        batch_memory = self.replay_buffer.sample(self.batch_size, self.sub_trace_length)
                        # self.model.update_params(batch_memory, self.gamma)
                        # print("Update params!")
                        self.sub_controller.update_params(batch_memory, self.gamma, self.batch_size)

                    if self.meta_replay_buffer.__len__() > self.min_meta_size_batch and timestep % self.update_steps == 0:
                        batch_memory = self.meta_replay_buffer.sample(self.meta_batch_size, self.meta_trace_length)
                        # self.h_model.update_params(batch_memory, self.gamma)
                        # print("Update meta params!")
                        self.meta_controller.update_params(batch_memory, self.gamma, self.meta_batch_size)

                    # If the episode has been completed and there was a reward, then success
                    if done and reward > 0:
                        n_success += 1

                    # If the goal or subgoal has been achieved, exit
                    if done or intrinsic_done:
                        break

                # === End of H steps ===

                # Sub Controller - Hindsight Goal Transitions
                goal_reached = self.get_subgoal_reached()
                if goal_reached is not None and len(sub_goal_transitions) > self.sub_trace_length:
                    goal_transitions = self.create_hindsight_goal_transition(sub_goal_transitions, goal_reached)
                    self.replay_buffer.add(goal_transitions)

                # Meta Controller - Hindsight Action Transition
                if goal_reached is not None:
                    reached_goal_state = get_subview(self.env, self.meta_goals[goal_reached], rgb=True)
                    reached_goal_state = self.process_state(reached_goal_state)
                    meta_transition_reward = 1 if done else -1
                    transition = self.create_transition(initial_state, goal_reached, meta_transition_reward, state, reached_goal_state, done)
                    meta_episode_buffer.append(transition)
                    meta_goal_transitions.append(transition)

                # If goal was not achieved, penalize it
                subgoal_reward, _ = self.intrinsic_reward(goal)
                if self.is_subgoal_test and subgoal_reward < 1:
                    subgoal_test_view = get_subview(self.env, self.meta_goals[goal], rgb=True)
                    subgoal_test_view = self.process_state(subgoal_test_view)
                    transition = self.create_transition(initial_state, goal, -self.max_steps_per_goal, state, subgoal_test_view, done)
                    meta_episode_buffer.append(transition)
                    self.is_subgoal_test = False

                # if episode % self.target_update_steps == 0:
                if update_nn_steps % self.target_update_steps == 0:
                    print("Update target netowrk!")
                    self.sub_controller.update_target_network()

                if self.meta_target_update_steps % update_nn_steps == 0:
                    self.meta_controller.update_target_network()

                # Corresponds to a list of at least n transitions
                if len(sub_episode_buffer) > self.sub_trace_length:
                    # print(len(sub_episode_buffer))
                    self.replay_buffer.add(sub_episode_buffer)

            # == Meta controller horizon ==

            # Meta Controller - Hindsight Goal Transitions
            goal_reached = self.get_subgoal_reached()
            if goal_reached is not None:
                goal_transitions = self.create_hindsight_goal_transition(meta_goal_transitions, goal_reached)
                self.meta_replay_buffer.add(goal_transitions)

            if len(meta_episode_buffer) > self.meta_trace_length:
                self.meta_replay_buffer.add(meta_episode_buffer)

            logger.success_rate.append(n_success / (episode+1))
            logger.extrinsic_reward_per_episode.append(episode_reward)
            logger.intrinsic_reward_per_episode.append(episode_intrinsic_reward)
            logger.steps_per_episode.append(timestep)

            print(f"Success rate: {logger.success_rate[-1]}")

            # Print for Logging stuff
            print(f'Episode: {episode}, Reward: {episode_reward}, Episode length: {timestep}')
            print(f'Epsilon: {self.epsilon}, Memory length: {self.replay_buffer.__len__()}')
            print(f'Epsilon Meta: {self.epsilon_meta}, Memory length: {self.meta_replay_buffer.__len__()}')
            print(f'Episode duration in seconds: {time.time() - episode_time_start}')
            print('\n')

            if episode > 10:
                logger.print_latest_statistics()

        if self.render:
            self.env.close()

        return logger


if __name__ == "__main__":
    PATH = "/Users/josesanchez/Documents/IAS/Thesis-Results"

    env_name = "StaticFourRooms-11x11"
    goal_pos = (8,7)
    env = StaticFourRoomsEnv(agent_pos=(2, 2), goal_pos=goal_pos, grid_size=11, max_steps=400)
    sub_goals = [
        (2, 2), (8, 2),
        (3, 3), (7, 3), (5, 2), (5, 3),
        (2, 5), (3, 5), (7, 5), (8, 5),
        (3, 7), (5, 7), (7, 7),
        (2, 8), (5, 8), (8, 8),
    ]

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
    env = RGBImgPartialObsWrapper(env)
    # env = RGBImgObsWrapper(env) # Fully Observable
    env = ImgObsWrapper(env)

    agent = Conv_hDDRQNAgent(env=env, num_episodes=500, render=False, meta_goals=sub_goals, goal_pos=goal_pos)
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
