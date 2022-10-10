
# Navigating in Partial Observability using Hierarchical Reinforcement Learning with Hindsight Experience

Navigating through an environment is a fundamental capability for many service, warehouse or domestic robots. 
Traditional navigation approaches involve using constant localization, mapping and planning under the assumption of full knowledge of the environment. 
This thesis focuses on expanding current reinforcement learning approaches to solve a navigation task in a partially observable environment. 
The approach makes use of hierarchical recurrent deep reinforcement learning, with the addition of hindsight experience replay in order to enhance learning by learning from failed attempts. Such architecture is evaluated in maze-like environments designed. The results show that hindsight effectively helps the model to learn form failed experiences even in partial observability.

This code is part of the Master Thesis for the University of Hamburg for the program 
of M. Sc Intelligent Adaptive Systems


## Features

This work makes use of the following concepts. 
- Hierarchical Reinforcement Learning
- Deep Recurrent Q-Learning
- Hindsight Experience Replay
- OpenAI Gym for the POMDP environments

The main neural network models 
are developed using PyTorch, while some complementary are in TensorFlow

## Environments

The algorithm was evaluated in three different environments with different complexities. For the environments
[Gym-Minigrid](https://github.com/Farama-Foundation/Minigrid) was used on top of OpenAI Gym to handle the dynamics. 
The specific environments can be found in the ``/envs`` folder. Concretely the tested environments are the following:
 * RandomEmpyEnv
 * StaticFourRoomsEnv
 * WTM

Example of environment for partial observability with the results showing the average reward obtained during training.

![Four Rooms Environment](https://github.com/JosephS96/hrl-pomdp/blob/main/images/fourrooms-env.png)

![Reward obtained for Four Rooms Environment](https://github.com/JosephS96/hrl-pomdp/blob/main/images/extrinsic_fourrooms.png)

## Installing
This project uses mostly standard libraries like PyTorch, Numpy, etc. For a complete set up, use the 
`requirements.txt`.

```bash
  pip install -r requirements.txt
```

The main libraries required for the experiments to function are the follwoing:
* PyTorch
* TensorFlow (to be removed later)
* Gym-Minigrid
* Numpy

Lastly, replace the ``minigrid.py`` file inside your local Gym-Minigrid installation environment with 
the one provided in the root folder of this repository. The new file contains the source code for the
SubGoals objects and is needed to work properly. In the future this will be changed so that
there is no need to override the original package.


## How to Run the experiments

Unfortunately there are no command line arguments for the environments or the agents. In order
to modify any of the parameters the file need to be directly edited. Each agent already has commented
the environments and the setting for them in their files, so it is only a matter of uncommenting
the desired environment to test.

DQN agent
```bash
  python3 agents/double_dqn_agent.py
```

hDRQN agent
```bash
  python3 agents/h_ddrqn.py
```

hDRQN+HER agent
```bash
  python3 agents/h_ddrqn+her.py
```


## ToDo

The code in this repository is not final and is a work in progress. Although
the main algorithms already yield the desired results the code still need
an overall cleanup.

Things to do in the nearby future
- Migrate most neural models to PyTorch (from TensorFlow)
- Delete unused code, models and agent
- Better organize the main entry point for the code
- Allow easy use from the command line with arguments