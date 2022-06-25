from agents.custom_hac.layer import Layer

import gym
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper


class HAC:
    def __init__(self,
                 env,
                 num_episodes,
                 n_layers=2,
                 max_subgoal_steps=10,
                 goal_pos=(15,15),
                 subgoal_testing_freq=0.6,
                 subgoals=[],
                 render=True
                ):

        self.env = env
        self.num_episodes = num_episodes
        self.render = render
        self.n_layers = n_layers
        self.goal_pos = goal_pos
        self.max_subgoal_steps = max_subgoal_steps
        self.num_updates = 10

        self.subgoals = subgoals
        self.subgoals.append(goal_pos)

        # Final goal to achieve
        self.agent_pos = (1, 1)

        # Goal array will store goal for each layer of the agent
        self.goal_array = [None for i in range(n_layers)]

        self.layers = self.get_model()

    def get_model(self):
        hierarchies = []
        for level in range(self.n_layers):
            if level == 0:
                primitive = Layer(self.env.observation_space.shape, 3, level=level, subgoals=self.subgoals)
                hierarchies.append(primitive)
            else:
                layer = Layer(self.env.observation_space.shape, len(self.subgoals), level, subgoals=self.subgoals)
                hierarchies.append(layer)

        return hierarchies

    # Update all NN for each layer
    def update_params(self):
        for i in range(self.n_layers):
            self.layers[i].update_params()

    def learn(self):

        for episode in range(self.num_episodes):
            print(f" === Episode {episode} ===")

            # Select final goal from final goal space
            goal = self.subgoals.index(self.goal_pos)

            # Get initial state from environment
            state = self.env.reset()

            # Reset steps counter

            # Train for an episode
            next_state, done, agent_pos = self.layers[self.n_layers-1].learn(self.n_layers-1, state, goal, self)

            # Update all networks layers
            self.update_params()

            print(episode)


if __name__ == "__main__":
    print("Custom HAC")

    PATH = "/Users/josesanchez/Documents/IAS/Thesis-Results"
    env_name = 'RandomMiniGrid-11x11'
    # env = RandomEmpyEnv(grid_size=11, max_steps=80, goal_pos=(9,9), agent_pos=(1, 1))
    # env = gym.make(env_name)
    env = gym.make('MiniGrid-Empty-8x8-v0', size=11)
    # env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    sub_goals = [
        (2, 2), (2, 5), (2, 8),
        (5, 2), (5, 5), (5, 8),
        (8, 2), (8, 5), (8, 8),
    ]

    subgoals = [(2, 2), (3, 3), (4, 4)]

    agent = HAC(env,
                num_episodes=50,
                n_layers=2,
                max_subgoal_steps=10,
                goal_pos=(9, 9),
                subgoal_testing_freq=0.6,
                subgoals=subgoals,
                render=True)
    agent.learn()
