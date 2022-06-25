import random

import numpy as np
from gym_minigrid.minigrid import SubGoal

from networks.hdqn import DoubleDQN
from replay_buffer import ExperienceBuffer
from common.utils import *
from networks.hdqn import DoubleDQN

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer, Flatten


class Layer:
    def __init__(self, input_dim, output_dim, level, subgoals=None):

        self.level = level   # Each layer needs to be aware of its own level
        self.max_subgoal_steps = 10
        # self.model = self.__create_model(input_dim, output_dim)
        self.model = DoubleDQN(state_shape=input_dim, output_shape=output_dim)
        self.replay_buffer = ExperienceBuffer(size=3000)
        self.batch_size = 64
        self.gamma = 0.95

        # Only sent this if layer > 0
        self.subgoals = subgoals
        self.selected_goal = None

    def place_subgoal(self, old_pos, new_pos, env):
        """
        Removes the previous position (x, y) and put the new pos (x, y)
        """
        # Remove the previous goal
        if (old_pos is not None) and (old_pos is not len(self.subgoals)-1):
            old_pos = self.subgoals[old_pos]
            env.grid.set(*old_pos, None)

        # Place the new subgoal in the environment
        new_pos = self.subgoals[new_pos]
        if new_pos == self.subgoals[-1]:
            return

        env.grid.set(*new_pos, SubGoal())
        # self.env.place_obj(SubGoal(), top=new_pos, size=(1, 1))

    def goal_reached(self, agent, goal):
        agent_pos = agent.env.agent_pos
        goal_pos = self.subgoals[goal]
        if type(agent_pos) is not tuple:
            agent_pos = (agent_pos[0], agent_pos[1])

        # check whether the agent reached the subgoal
        return agent_pos == goal_pos

    def get_random_subgoal(self):
        # TODO: Only return reasonable subgoals that are not overlapping
        # x = random.randint(1, 15)
        # y = random.randint(1, 15)

        # goal = random.(self.subgoals)
        goal = random.randint(0, len(self.subgoals)-1)

        return goal

    def choose_subgoal(self, state, goal):
        if np.random.sample() > 0.2:
            output = self.model.predict(state, goal)
            goal_idx = np.argmax(output, axis=0)
            goal = goal_idx
        else:
            goal = self.get_random_subgoal()

        return goal

    def choose_action(self, state, goal):
        if np.random.sample() > 0.2:
            output = self.model.predict(state, goal)
            action = np.argmax(output, axis=0)
        else:
            action = random.randint(0, 2)

        return action

    def update_params(self):
        batch_memory = self.replay_buffer.sample(self.batch_size)
        self.model.update_params(batch_memory, self.gamma)

    # This is basically the equivalent of the TRAIN-LEVEL function in the pseudocode
    """
    params: 
        - level: current hierarchy level
        - state: current state from higher layer
        - goal: Goal selected from the higher hierarchy, index
    """
    def learn(self, level, state, goal, agent):
        next_state = None
        is_subgoal_test = False
        is_next_subgoal_test = False
        done = False
        self.selected_goal = goal

        global_reward = 0

        goal_transitions = []
        self.place_subgoal(None, goal, agent.env)

        for _ in range(self.max_subgoal_steps):
            action = None

            # Transforms the goal coordinates into a Grid representation
            goal_state = get_subview(agent.env, self.subgoals[goal])

            if level > 0:
                action = self.choose_subgoal(state, goal_state)
                self.place_subgoal(self.selected_goal, action, agent.env)
                self.selected_goal = action
            else:
                action = self.choose_action(state, goal_state)   # Sample noisy action form policy

            # ====== High Level Policy =======
            if level > 0:
                if not is_subgoal_test:
                    action = self.choose_subgoal(state, goal_state)
                    self.place_subgoal(self.selected_goal, action, agent.env)
                    self.selected_goal = action

                # Determine whether to test subgoal action
                if np.random.sample() > 0.6:
                    is_next_subgoal_test = True

                # Train next sub-level of the hierarchy using action as subgoal
                # Pass subgoal to next layer
                next_state, done, agent_pos = agent.layers[level-1].learn(level-1, state, action, agent)

                # if subgoal was tested but not achieved, add subgoal testing transition
                if is_next_subgoal_test and not self.goal_reached(agent, action):
                    goal_state = get_subview(agent.env, self.subgoals[action])
                    self.replay_buffer.add(state, action, -self.max_subgoal_steps, next_state, goal_state, done)

                # Hindsight Action Transition
                # Save transition into reply buffer
                if self.goal_reached(agent, goal):
                    self.replay_buffer.add(state, action, 1.0, next_state, goal_state, done)
                else:
                    self.replay_buffer.add(state, action, -1.0, next_state, goal_state, done)

                # For hindsight action transition
                # WHY ???
                # action = next_state

            # ====== Low Level Policy =======
            else:
                if not is_subgoal_test:
                    action = self.choose_action(state, goal_state)

                # Execute primitive action and observe new state
                next_state, reward, done, _ = agent.env.step(action)

                # Hindsight Action Transition
                # Save transition into reply buffer
                if self.goal_reached(agent, goal):
                    self.replay_buffer.add(state, action, 1.0, next_state, goal_state, done)
                else:
                    self.replay_buffer.add(state, action, -1.0, next_state, goal_state, done)

                if agent.render:
                    agent.env.render()

                if reward > 0 or self.goal_reached(agent, goal):
                    print(f'Reward Achieved!: {reward}')

            # check if goal was achieved
            goal_achieved = self.goal_reached(agent, goal)

            """
            # hindsight action transition
            # Save transition into reply buffer
            if self.goal_reached(agent, goal):
                self.replay_buffer.add(state, action, 1.0, next_state, goal_state, done)
            else:
                self.replay_buffer.add(state, action, -1.0, next_state, goal_state, done)
            """

            # copy for goal transitions
            goal_transitions.append([state, action, -1.0, next_state, None, done])

            # TODO: Might need to replace this next state with agent position
            state = next_state

            if done or goal_achieved:
                break

        # ==== Finish H attempts/steps ====



        # Hindsight Goal Transition
        # last transition reward is 0
        goal_transitions[-1][2] = 0
        for transition in goal_transitions:
            # last state is goal for all transitions
            # transition[4] = next_state
            transition[4] = get_subview(agent.env, agent.env.agent_pos)
            self.replay_buffer.add(transition[0], transition[1], transition[2], transition[3], transition[4], transition[5])


        # Output current state: next_state
        return next_state, done, agent.env.agent_pos
