from agents.custom_hac.layer import Layer

import gym
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper


class HAC():
    def __init__(self,
                 env,
                 num_episodes,
                 n_layers=2,
                 max_subgoal_steps=10,
                 goal_pos=(15,15),
                 subgoal_testing_freq = 0.6,
                 render=True
                ):

        self.env = env
        self.num_episodes = num_episodes
        self.render = render
        self.n_layers = n_layers
        self.goal_pos = goal_pos
        self.max_subgoal_steps = max_subgoal_steps
        self.num_updates = 10

        # Final goal to achieve
        self.agent_pos = (1, 1)

        # Goal array will store goal for each layer of the agent
        self.goal_array = [None for i in range(n_layers)]

        self.layers = self.get_model()

    def get_model(self):
        hierarchies = []
        for level in range(self.n_layers):
            if level == 0:
                primitive = Layer(self.env.observation_space.shape, self.env.action_space.n, level=level)
                hierarchies.append(primitive)
            else:
                layer = Layer(self.env.observation_space.shape, 2, level)
                hierarchies.append(layer)

        return hierarchies

    # Update all NN for each layer
    def update_params(self):
        for i in range(self.n_layers):
            self.layers[i].update_params(self.num_updates)

    def learn(self):

        for episode in range(self.num_episodes):
            # Select final goal from final goal space
            goal = self.goal_pos

            # Get initial state from environment
            state = self.env.reset()

            # Reset steps counter

            # Train for an episode
            next_state, done = self.layers[self.n_layers-1].learn(self.n_layers-1, state, goal, self)

            # Update all networks layers
            self.update_params()

            print(episode)


if __name__ == "__main__":
    print("Custom HAC")

    env = gym.make('MiniGrid-Empty-16x16-v0')
    env = ImgObsWrapper(env)

    agent = HAC(env, num_episodes=10, n_layers=2, max_subgoal_steps=10, goal_pos=(1, 1), subgoal_testing_freq=0.6, render=False)
    agent.learn()
