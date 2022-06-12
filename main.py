import gym
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper
from gym_minigrid.minigrid import SubGoal, Floor
from agents.DDQN import DDQNAgent
from agents.hDQN import hDQNAgent

import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    env = gym.make('MiniGrid-Empty-16x16-v0')
    # env = gym.make('MiniGrid-FourRooms-v0', agent_pos=(5, 5), goal_pos=(13, 13))
    env = ImgObsWrapper(env)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    print(state_dim)
    print(action_dim)

    # agent = DDQNAgent(env=env, num_episodes=200, render=False)

    # sub_goals = [(3, 3), (6, 6), (9, 9), (11, 11)]
    sub_goals = [(4, 8), (4, 11), (6, 12), (11, 12)]
    agent = hDQNAgent(env=env, num_episodes=200, meta_goals=sub_goals, goal_pos=(13, 13), render=True)
    rewards, success = agent.learn()

    # plt.figure()
    # plotting.plot_rewards([stats], smoothing_window=10)
    plt.ylim(0, 1.0)
    plt.plot(range(len(rewards)), rewards)
    plt.show()

    plt.ylim(0, 1.0)
    plt.plot(range(len(success)), success)
    plt.show()

    """
    agent.mode = 'test'
    state = env.reset()
    done = False
    while not done:
        env.render()
        # action = agent.choose_action(state)
        action = agent
        state, reward, done, info = env.step(action)
        time.sleep(0.1)


    env.close()
    """
