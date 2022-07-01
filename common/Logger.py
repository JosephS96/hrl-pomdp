import numpy as np
from datetime import datetime


class Logger:
    def __init__(self):
        self.success_rate = []
        self.extrinsic_reward_per_episode = []
        self.extrinsic_reward_per_step = []
        self.intrinsic_reward_per_episode = []
        self.intrinsic_reward_per_step = []
        self.steps_per_episode = []

    def print_latest_statistics(self, n_steps=1000, n_episodes=10):
        reward_per_episode = np.mean(self.extrinsic_reward_per_episode[-n_episodes:])
        reward_per_step = np.mean(self.extrinsic_reward_per_step[-n_steps:])

        intrinsic_per_episode = np.mean(self.intrinsic_reward_per_episode[-n_episodes:])
        intrinsic_per_step = np.mean(self.intrinsic_reward_per_step[-n_steps:])

        steps = np.mean(self.steps_per_episode[-n_episodes:])

        print("==== Statistics =====")
        print(f"Last episodes: {n_episodes}, Last steps: {n_steps}")
        print(f"Episode reward: {reward_per_episode}, Steps reward: {reward_per_step}")
        print(f"Episode intrinsic: {intrinsic_per_episode}, Steps intrinsic: {intrinsic_per_step}")
        print(f"Steps per episode: {steps}")
        print("\n")

    def save(self, env_name, identifier):
        PATH = "/Users/josesanchez/Documents/IAS/Thesis-Results"
        now = datetime.now()
        current_datetime = now.strftime("%m-%d-%Y-%H-%M-%S")

        save_path = f"{PATH}/{env_name}/{identifier}_{current_datetime}.npy"

        results = {}
        results['algorithm'] = identifier
        results['datetime'] = current_datetime
        results['success_rate'] = self.success_rate
        results['extrinsic_reward_per_episode'] = self.extrinsic_reward_per_episode
        results['extrinsic_reward_per_step'] = self.extrinsic_reward_per_step
        results['intrinsic_reward_per_episode'] = self.intrinsic_reward_per_episode
        results['intrinsic_reward_per_step'] = self.intrinsic_reward_per_step
        results['steps_per_episode'] = self.steps_per_episode

        np.save(save_path, results)

        print("Saved results!")
