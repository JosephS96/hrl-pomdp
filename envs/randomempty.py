import math

from gym_minigrid.minigrid import *

class RandomEmpyEnv(MiniGridEnv):

    """"
    Empty environment with random agent and goal position if the position
    is not given at initialization
    """
    def __init__(self, agent_pos=None, goal_pos=None, grid_size=13, max_steps=50):
        self.agent_pos = agent_pos
        self.goal_pos = goal_pos
        self.grid_size = grid_size,
        self.max_steps = max_steps

        self.agent_start_pos = agent_pos
        self.agent_start_dir = 0

        super().__init__(
            grid_size=grid_size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # self.agent_pos = None
        # pos = self.place_obj(Goal(), None, None, max_tries=math.inf)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"