from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class Apartment(MiniGridEnv):
    def __init__(self, agent_pos=None, goal_pos=None, max_steps=100):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(width=20, height=12, max_steps=max_steps)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self.grid.vert_wall(4, 1, length=8)
        self.grid.vert_wall(7, 1, length=7)
        self.grid.vert_wall(11, 1, length=7)
        self.grid.vert_wall(15, 10, length=1)

        self.grid.vert_wall(15, 1, length=5)
        self.grid.vert_wall(15, 9, length=2)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info

register(
    id='MiniGrid-Apartment-v0',
    entry_point='envs:ApartmentEnv'
)