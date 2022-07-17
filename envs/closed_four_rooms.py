from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class ClosedFourRoomsEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, grid_size=19, max_steps=50):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=grid_size, max_steps=max_steps)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    # pos = (xR, self._rand_int(yT + 1, yB))
                    # pos1 = (xR, 2)
                    # pos2 = (xR, 3)
                    pos3 = (xR, 7)
                    pos4 = (xR, 8)
                    # self.grid.set(*pos1, None)
                    # self.grid.set(*pos2, None)
                    self.grid.set(*pos3, None)
                    self.grid.set(*pos4, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    # pos = (self._rand_int(xL + 1, xR), yB)
                    pos1 = (2, yB)
                    pos2 = (3, yB)
                    pos3 = (7, yB)
                    pos4 = (8, yB)
                    self.grid.set(*pos1, None)
                    self.grid.set(*pos2, None)
                    self.grid.set(*pos3, None)
                    self.grid.set(*pos4, None)

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

    def _reward(self):
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info

register(
    id='MiniGrid-ClosedFourRooms-v0',
    entry_point='envs:ClosedFourRoomsEnv'
)