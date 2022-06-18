import math
import operator
from functools import reduce

import numpy as np
import gym
from gym import error, spaces, utils
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, Goal, SubGoal

class PartialSubgoalWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_shape = env.observation_space['image'].shape

        self.observation_space.spaces

    def find_subgoal(self):
        pass

    def observation(self, obs):
        """"
        Here I want to return:
        -normal observations
        -frame of 7x7 of the goal in the center
        -direction?
        """""

        grid, vis_mask = self.env.gen_obs_grid()

        return {
            'image': obs['image'],
            'goal': 'hello there'
        }