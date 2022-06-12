import time

import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from envs.staticfourrooms import StaticFourRoomsEnv
from envs.apartment import Apartment
from common.schedules import LinearSchedule

if __name__ == "__main__":

    # env = gym.make('MiniGrid-SimpleCrossingS9N1-v0')
    # env = gym.make('MiniGrid-Empty-16x16-v0')
    env = StaticFourRoomsEnv(grid_size=11)
    # env = Apartment()
    # env = gym.make('MiniGrid-FourRooms-v0', agent_pos=(5,5), goal_pos=(13,13))
    # env = ImgObsWrapper(env)
    env = FullyObsWrapper(env)
    env.reset()
    done = False
    while not done:
        env.render()
        # action = agent.choose_action(state)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        time.sleep(0.05)

    env.close()

