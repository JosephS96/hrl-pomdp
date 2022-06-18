import matplotlib.pyplot as plt
import numpy as np
import random
import time
import gym
from gym_minigrid.minigrid import SubGoal
from gym_minigrid.wrappers import ImgObsWrapper

from replay_buffer import ReplayBuffer
from common.schedules import LinearSchedule
from networks.dqn import DQN

class HierarchicalDQNAgent:
    def __init__(self, env,
                 num_episodes=100,
                 meta_goals=[(2,2), (3,3), (4,4)],
                 goal_pos=(1,1),
                 render=False):
        self.identifier = 'hierarchical-dqn'
        self.env = env
        self.num_episodes = num_episodes
        self.render = render
        self.goal_pos = goal_pos

        # Replay Buffer
        self.buffer_size = 10000
        self.min_size_batch = 3000
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.batch_size = 32

        # Meta controller replay buffer
        self.min_meta_size_batch = 100
        self.meta_replay_buffer = ReplayBuffer(self.buffer_size)

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
        self.epsilon = {}  # Exploration rate of every goal
        self.goal_success = {}  # Times the agent reached the goal
        self.goal_selected = {}  # Times the goal was selected
        for goal in self.meta_goals:
            self.epsilon[goal] = 1.0
            self.goal_success[goal] = 0.0
            self.goal_selected[goal] = 1.0

        # Steps to give before updating target model nn
        self.target_update_steps = 2000
        self.model = DQN(input_shape=self.env.observation_space.shape, output_shape=self.env.action_space.n,
                               n_neurons=32)
        self.h_model = DQN(input_shape=self.env.observation_space.shape, output_shape=len(self.meta_goals),
                         n_neurons=32)

        self.mode = 'train'

    def choose_action(self, state, epsilon):
        if self.mode == 'train':
            action = self.__epsilon_greedy(state, epsilon)
            return action
        else:  # Only Exploitation
            output = self.model.predict(state)
            action = np.argmax(output, axis=0)
            return action

    # Epsilon greedy to choose the controller actions
    def __epsilon_greedy(self, state, epsilon):
        if random.random() > epsilon:  # Exploitation
            output = self.model.predict(state)
            action = np.argmax(output, axis=0)
            return action
        else:  # Exploration
            # Reduce the exploration probability
            return random.randint(0, 2)
            # return self.env.action_space.sample()

    def choose_goal(self, state):
        if self.mode == 'train':
            goal = self.__epsilon_greedy_meta(state)
            return goal
        else:  # Only Exploitation
            output = self.model.predict(state)
            action = np.argmax(output, axis=0)
            return action

    # Epsilon greedy for the meta controller to choose the goals
    # This returns the index for the meta_goal, not the actual goal
    def __epsilon_greedy_meta(self, state):
        if random.random() > self.epsilon_meta:
            output = self.h_model.predict(state)
            goal_index = np.argmax(output, axis=0)
            return goal_index
        else:
            goal_index = random.randint(0, len(self.meta_goals) - 1)
            return goal_index

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
            episode_time_start = time.time()
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
                        batch_memory = self.replay_buffer.sample(self.batch_size)
                        self.model.update_params(batch_memory, self.gamma)

                    if self.meta_replay_buffer.__len__() > self.min_meta_size_batch:
                        batch_memory = self.meta_replay_buffer.sample(self.batch_size)
                        self.h_model.update_params(batch_memory, self.gamma)

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
            print(f'Episode duration in seconds: {time.time() - episode_time_start}')
            print('\n')

        if self.render:
            self.env.close()

        return reward_per_episode, success_rate_history

if __name__ == "__main__":
    PATH = "/Users/josesanchez/Documents/IAS/Thesis-Results"
    env_name = 'MiniGrid-Empty-16x16-v0'
    env = gym.make(env_name)
    # env = gym.make('MiniGrid-FourRooms-v0', agent_pos=(5, 5), goal_pos=(13, 13))
    env = ImgObsWrapper(env)

    sub_goals = [(3, 3), (6, 6), (9, 9), (11, 11)]

    agent = HierarchicalDQNAgent(env=env, num_episodes=200, render=False, meta_goals=sub_goals)
    rewards, success = agent.learn()

    results = {}
    results['success'] = success
    results['rewards'] = rewards

    save_path = f"{PATH}/{env_name}/{agent.identifier}-{time.time()}.npy"
    np.save(save_path, results)

    # plt.figure()
    # plotting.plot_rewards([stats], smoothing_window=10)
    plt.ylim(0, 1.0)
    plt.plot(range(len(rewards)), rewards)
    plt.show()

    plt.ylim(0, 1.0)
    plt.plot(range(len(success)), success)
    plt.show()