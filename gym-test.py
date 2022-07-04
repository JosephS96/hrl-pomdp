import time

import gym
import gym_minigrid
import matplotlib.pyplot as plt
from gym_minigrid.minigrid import SubGoal, Grid
from gym_minigrid.wrappers import *
from envs.staticfourrooms import StaticFourRoomsEnv
from envs.randomempty import RandomEmpyEnv
from envs.apartment import Apartment
from common.schedules import LinearSchedule
from envs.wrappers import PartialSubgoalWrapper


def get_subgoal_view(env, goal_pos):
    """
            Get the extents of the square set of tiles visible to the agent
            Note: the bottom extent indices are not included in the set

            TODO: Get subview of the subgoal in relation to the agent to stack it in the state
     """

    # Facing right
    if env.agent_dir == 0:
        topX = goal_pos[0]
        topY = goal_pos[1] - env.agent_view_size // 2
    # Facing down
    elif env.agent_dir == 1:
        topX = env.agent_pos[0] - env.agent_view_size // 2
        topY = env.agent_pos[1]
    # Facing left
    elif env.agent_dir == 2:
        topX = env.agent_pos[0] - env.agent_view_size + 1
        topY = env.agent_pos[1] - env.agent_view_size // 2
    # Facing up
    elif env.agent_dir == 3:
        topX = env.agent_pos[0] - env.agent_view_size // 2
        topY = env.agent_pos[1] - env.agent_view_size + 1
    else:
        assert False, "invalid agent direction"

    botX = topX + env.agent_view_size
    botY = topY + env.agent_view_size

    topX, topY, botX, botY


    """
    Generate the sub-grid observed by the agent.
    This method also outputs a visibility mask telling us which grid
    cells the agent can actually see.
    """

    grid = env.grid.slice(topX, topY, env.agent_view_size, env.agent_view_size)

    print(env.agent_pos)

    for i in range(env.agent_dir + 1):
        grid = grid.rotate_left()

    # Process occluders and visibility
    # Note that this incurs some performance cost
    if not env.see_through_walls:
        vis_mask = grid.process_vis(agent_pos=(env.agent_view_size // 2, env.agent_view_size // 2)) # Put the subgoal in the center
    else:
        vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

    # Encode the partially observable view into a numpy array
    image = grid.encode(vis_mask)

    # to rgb
    image = env.get_obs_render(image, 8)

    return image


def get_subview(env, pos, view_size=7, rgb=False):
    # get topX and topY
    top_offset = int(view_size / 2)

    topX = pos[0] - top_offset
    topY = pos[1] - top_offset

    print(env.agent_pos)

    grid = env.grid.slice(topX, topY, width=view_size, height=view_size)
    grid = grid.encode()

    if rgb:
        grid, mask = Grid.decode(grid)
        agent_pos = to_relative_pos(env.agent_pos, topX, topY, view_size)

        rgb_img = grid.render(
            8,
            agent_pos= agent_pos,
            agent_dir=env.agent_dir,
            highlight_mask=mask
        )
        return rgb_img

    return grid


def to_relative_pos(agent_pos, topX, topY, size):
    relative_x = agent_pos[0] - topX
    relative_y = agent_pos[1] - topY

    valid_x = relative_x >= 0 and relative_x < size
    valid_y = relative_y >= 0 and relative_y < size

    if valid_x and valid_y:
        relative_pos = (relative_x, relative_y)
        return relative_pos

    return None


if __name__ == "__main__":

    # env = gym.make('MiniGrid-SimpleCrossingS9N1-v0')
    # env = gym.make('MiniGrid-Empty-16x16-v0')
    env = StaticFourRoomsEnv(agent_pos= (2, 2), goal_pos=(9, 9) , grid_size=11, max_steps=20)
    # env = RandomEmpyEnv()
    # env = Apartment()
    # env = gym.make('MiniGrid-FourRooms-v0', agent_pos=(5,5), goal_pos=(13,13))
    # env = FullyObsWrapper(env)
    # env = RGBImgPartialObsWrapper(env)
    # env = PartialSubgoalWrapper(env)
    env = ImgObsWrapper(env)
    env.reset()
    done = False

    env.place_obj(SubGoal(), top=(3, 3), size=(1, 1))
    # sub_view = get_subgoal_view(env, (2, 2))
    sub_view = get_subview(env, (3, 3))

    # plt.imshow(sub_view, cmap='hot', interpolation='nearest')
    # plt.show()

    for i in range(10):
        env.reset()
        done = False
        while not done:
            env.render()
            # action = agent.choose_action(state)
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            # print(state.shape)
            time.sleep(0.05)

    env.close()

