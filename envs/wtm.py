from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class WTMEnv(MiniGridEnv):
    def __init__(self, agent_pos=None, goal_pos=None, max_steps=100):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(width=15, height=16, max_steps=max_steps)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Generate walls from top to bottom
        self.grid.horz_wall(0, 5)
        self.grid.horz_wall(0, 10)
        self.grid.vert_wall(5, 0)
        self.grid.vert_wall(9, 0)

        # Remove blocks to create doors
        self.grid.set(5, 1, None)
        self.grid.set(5, 2, None)
        self.grid.set(5, 3, None)
        self.grid.set(5, 4, None)
        self.grid.set(5, 7, None)
        self.grid.set(5, 8, None)
        self.grid.set(5, 12, None)
        self.grid.set(5, 13, None)

        self.grid.set(9, 2, None)
        self.grid.set(9, 3, None)
        self.grid.set(9, 7, None)
        self.grid.set(9, 8, None)
        self.grid.set(9, 12, None)
        self.grid.set(9, 13, None)

        self.grid.set(11, 10, None)
        self.grid.set(12, 10, None)

        # Hallway
        self.grid.set(6, 5, None)
        self.grid.set(7, 5, None)
        self.grid.set(8, 5, None)

        self.grid.set(6, 10, None)
        self.grid.set(7, 10, None)
        self.grid.set(8, 10, None)

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
    id='MiniGrid-WTM-v0',
    entry_point='envs:WTMEnv'
)